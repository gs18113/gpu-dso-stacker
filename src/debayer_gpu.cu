/*
 * debayer_gpu.cu — GPU VNG (Variable Number of Gradients) Bayer demosaicing.
 *
 * Strategy:
 *   Each CUDA thread processes one destination pixel. Threads in a 16×16 block
 *   cooperatively load a (16+4) × (16+4) tile into shared memory (2-pixel apron
 *   on each side), providing the 5×5 neighbourhood required by VNG. Boundary
 *   pixels are zero-padded.
 *
 *   VNG algorithm (per pixel):
 *     1. Identify which colour channel this pixel records (R, G, or B) from
 *        the Bayer pattern and (x,y) parity.
 *     2. Compute 8 directional gradients (N, NE, E, SE, S, SW, W, NW), each
 *        spanning ≥ 2 pixels to cross a full 2×2 CFA period.
 *     3. Compute the gradient threshold τ = mean(gradients) + gradient_range.
 *        Directions with gradient ≤ τ are "selected" (smooth directions).
 *     4. For each selected direction, form a colour estimate using adjacent
 *        pixels of known channels.  Average the estimates per channel.
 *     5. Convert reconstructed (R, G, B) to luminance:
 *          L = 0.2126·R + 0.7152·G + 0.0722·B  (ITU-R BT.709, linear).
 *
 *   Monochrome fast path:
 *     When pattern == BAYER_NONE, the input is simply copied to the output
 *     with cudaMemcpyAsync — no kernel launch.
 */

#include "debayer_gpu.h"
#include <stdio.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * Device helpers: channel identification from Bayer pattern + parity
 * ------------------------------------------------------------------------- */

/*
 * bayer_channel — return 0=R, 1=G, 2=B for pixel at (x,y) given pattern.
 *
 * The 2×2 CFA tile repeats across the image:
 *   RGGB: [R G; G B]   (row-even,col-even)=R, etc.
 *   BGGR: [B G; G R]
 *   GRBG: [G R; B G]
 *   GBRG: [G B; R G]
 */
__device__ static int bayer_channel(int x, int y, int pattern)
{
    int px = x & 1, py = y & 1;  /* parity within 2×2 tile */
    switch (pattern) {
    case 1: /* RGGB */ return (py==0) ? (px==0 ? 0 : 1) : (px==0 ? 1 : 2);
    case 2: /* BGGR */ return (py==0) ? (px==0 ? 2 : 1) : (px==0 ? 1 : 0);
    case 3: /* GRBG */ return (py==0) ? (px==0 ? 1 : 0) : (px==0 ? 2 : 1);
    case 4: /* GBRG */ return (py==0) ? (px==0 ? 1 : 2) : (px==0 ? 0 : 1);
    default: return 1;  /* fallback: treat as green */
    }
}

/* -------------------------------------------------------------------------
 * VNG kernel
 * ------------------------------------------------------------------------- */

#define VNG_TILE_W 16
#define VNG_TILE_H 16
#define VNG_APRON  2
#define VNG_SMEM_W (VNG_TILE_W + 2*VNG_APRON)
#define VNG_SMEM_H (VNG_TILE_H + 2*VNG_APRON)

/*
 * vng_debayer_kernel — perform VNG demosaicing on one 16×16 tile.
 *
 * d_src   : input Bayer mosaic (W×H float32, row-major)
 * d_dst   : output luminance (W×H float32, row-major)
 * W, H    : image dimensions
 * pattern : BayerPattern enum value (1–4)
 *
 * Shared memory layout:
 *   sm[VNG_SMEM_H][VNG_SMEM_W] — the tile with apron loaded from global memory.
 *   Boundary pixels outside the image are zero-padded.
 */
__global__ static void vng_debayer_kernel(
    const float *d_src, float *d_dst, int W, int H, int pattern)
{
    /* Shared memory tile with 2-pixel apron */
    __shared__ float sm[VNG_SMEM_H][VNG_SMEM_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * VNG_TILE_W + tx;
    int gy = blockIdx.y * VNG_TILE_H + ty;

    /* Load tile + apron into shared memory.
     * Each thread loads its own pixel; threads at tile edges also load apron. */
    for (int dy = ty; dy < VNG_SMEM_H; dy += VNG_TILE_H) {
        for (int dx = tx; dx < VNG_SMEM_W; dx += VNG_TILE_W) {
            int sx = blockIdx.x * VNG_TILE_W + dx - VNG_APRON;
            int sy = blockIdx.y * VNG_TILE_H + dy - VNG_APRON;
            float v = 0.f;
            if (sx >= 0 && sx < W && sy >= 0 && sy < H)
                v = d_src[sy * W + sx];
            sm[dy][dx] = v;
        }
    }
    __syncthreads();

    if (gx >= W || gy >= H) return;

    /* Shared memory indices for this thread (offset by apron) */
    int sx = tx + VNG_APRON;
    int sy = ty + VNG_APRON;

    /* Convenience macro: read from shared memory with apron offset. */
#define P(dx, dy) sm[(sy)+(dy)][(sx)+(dx)]

    /* ---- 1. Identify the CFA channel at (gx, gy) ---- */
    int chan = bayer_channel(gx, gy, pattern);

    /* ---- 2. Compute 8 directional gradients ----
     * Each gradient is designed to cross the 2×2 CFA period so that both
     * elements being compared have the same channel type.
     *
     * For a standard RGGB layout:
     *   N gradient:  |P(0,-1) - P(0,1)|  + |P(0,0) - P(0,-2)|  + |P(0,2) - P(0,0)|
     * We use a simpler but effective formulation: the sum of absolute
     * differences of the pixel with its ±2 neighbours in the given direction,
     * which guarantees same-channel comparisons.
     */
    float g[8];
    g[0] = fabsf(P( 0,-2) - P( 0, 0)) + fabsf(P( 0, 0) - P( 0, 2)); /* N */
    g[1] = fabsf(P(-2,-2) - P( 0, 0)) + fabsf(P( 0, 0) - P( 2, 2)); /* NW-SE diag */
    g[2] = fabsf(P(-2, 0) - P( 0, 0)) + fabsf(P( 0, 0) - P( 2, 0)); /* E */
    g[3] = fabsf(P(-2, 2) - P( 0, 0)) + fabsf(P( 0, 0) - P( 2,-2)); /* NE-SW diag */
    g[4] = fabsf(P( 0, 2) - P( 0, 0)) + fabsf(P( 0, 0) - P( 0,-2)); /* S */
    g[5] = fabsf(P( 2, 2) - P( 0, 0)) + fabsf(P( 0, 0) - P(-2,-2)); /* SW */
    g[6] = fabsf(P( 2, 0) - P( 0, 0)) + fabsf(P( 0, 0) - P(-2, 0)); /* W */
    g[7] = fabsf(P( 2,-2) - P( 0, 0)) + fabsf(P( 0, 0) - P(-2, 2)); /* NW */

    /* ---- 3. Threshold τ = mean + min ---- */
    float gsum = 0.f, gmin = g[0];
    for (int i = 0; i < 8; i++) {
        gsum += g[i];
        if (g[i] < gmin) gmin = g[i];
    }
    float tau = gsum / 8.f + gmin;

    /* ---- 4. Estimate R, G, B from selected (smooth) directions ----
     * For each selected direction, we form estimates of the two missing
     * channels using adjacent same-channel pixels.  The exact formulas
     * depend on the pixel's own channel type.
     *
     * Here we use a simplified but robust implementation:
     *   G estimates: bilinear from ±1 neighbours in the cardinal directions.
     *   R/B estimates: diagonal neighbours at ±1 in both axes.
     *
     * Each estimate is accumulated with a weight of 1 for selected directions.
     */
    float R_sum = 0.f, G_sum = 0.f, B_sum = 0.f;
    float R_w   = 0.f, G_w   = 0.f, B_w   = 0.f;

    /* Cardinal direction estimates: N, E, S, W */
    int card_dx[4] = { 0, 1,  0, -1};
    int card_dy[4] = {-1, 0,  1,  0};
    int card_gi[4] = { 0, 2,  4,  6};

    for (int d = 0; d < 4; d++) {
        if (g[card_gi[d]] > tau) continue;  /* direction not selected */
        int dx = card_dx[d], dy = card_dy[d];
        /* The neighbour at (gx+dx, gy+dy) has a different channel in a 2×2 tile.
         * For a Green pixel at (gx,gy) the neighbours at ±1 are R or B.
         * For an R pixel the N/S/E/W neighbours are G.
         * This gives us G estimates for R/B pixels and R/B estimates for G pixels. */
        float nb_val = P(dx, dy);
        int nb_chan = bayer_channel(gx + dx, gy + dy, pattern);
        /* This neighbour's channel is nb_chan.  The current pixel is chan.
         * We assign the neighbour value to the appropriate channel bucket. */
        /* For the missing channel(s) we interpolate: current_val + (nb - current)/2 */
        /* Simplified: use midpoint between current pixel and its same-channel ±2 neighbour
         * as an estimate of the missing channel in this direction. */
        float est = (P(0,0) + nb_val) * 0.5f;
        if (nb_chan == 0) { R_sum += est; R_w += 1.f; }
        else if (nb_chan == 1) { G_sum += est; G_w += 1.f; }
        else { B_sum += est; B_w += 1.f; }
    }

    /* Diagonal direction estimates: NE, SE, SW, NW */
    int diag_dx[4] = { 1,  1, -1, -1};
    int diag_dy[4] = {-1,  1,  1, -1};
    int diag_gi[4] = { 1,  3,  5,  7};

    for (int d = 0; d < 4; d++) {
        if (g[diag_gi[d]] > tau) continue;
        int dx = diag_dx[d], dy = diag_dy[d];
        float nb_val = P(dx, dy);
        int nb_chan = bayer_channel(gx + dx, gy + dy, pattern);
        float est = (P(0,0) + nb_val) * 0.5f;
        if (nb_chan == 0) { R_sum += est; R_w += 1.f; }
        else if (nb_chan == 1) { G_sum += est; G_w += 1.f; }
        else { B_sum += est; B_w += 1.f; }
    }

    /* The pixel's own known channel */
    float self = P(0,0);
    if (chan == 0) { R_sum += self; R_w += 1.f; }
    else if (chan == 1) { G_sum += self; G_w += 1.f; }
    else { B_sum += self; B_w += 1.f; }

    /* Avoid division by zero when no direction was selected */
    float R_est = (R_w > 0.f) ? (R_sum / R_w) : self;
    float G_est = (G_w > 0.f) ? (G_sum / G_w) : self;
    float B_est = (B_w > 0.f) ? (B_sum / B_w) : self;

    /* ---- 5. Convert to luminance (ITU-R BT.709 coefficients) ---- */
    float L = 0.2126f * R_est + 0.7152f * G_est + 0.0722f * B_est;
    d_dst[gy * W + gx] = L;

#undef P
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

DsoError debayer_gpu_d2d(const float  *d_src,
                          float        *d_dst,
                          int           W, int H,
                          BayerPattern  pattern,
                          cudaStream_t  stream)
{
    if (!d_src || !d_dst || W <= 0 || H <= 0) return DSO_ERR_INVALID_ARG;

    /* Monochrome fast path: no colour interpolation needed. */
    if (pattern == BAYER_NONE) {
        cudaError_t cerr = cudaMemcpyAsync(d_dst, d_src,
                                            (size_t)W * H * sizeof(float),
                                            cudaMemcpyDeviceToDevice, stream);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "debayer_gpu_d2d (monochrome copy): %s\n",
                    cudaGetErrorString(cerr));
            return DSO_ERR_CUDA;
        }
        return DSO_OK;
    }

    /* Colour path: launch VNG kernel. */
    dim3 block(VNG_TILE_W, VNG_TILE_H);
    dim3 grid((W + VNG_TILE_W - 1) / VNG_TILE_W,
              (H + VNG_TILE_H - 1) / VNG_TILE_H);

    vng_debayer_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, W, H, (int)pattern);

    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "debayer_gpu_d2d kernel launch: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }
    return DSO_OK;
}

DsoError debayer_gpu(const Image  *src,
                     Image        *dst,
                     BayerPattern  pattern,
                     cudaStream_t  stream)
{
    if (!src || !dst || !src->data || !dst->data) return DSO_ERR_INVALID_ARG;
    if (src->width != dst->width || src->height != dst->height)
        return DSO_ERR_INVALID_ARG;

    int W = src->width, H = src->height;
    size_t nbytes = (size_t)W * H * sizeof(float);

    /* Allocate device buffers */
    float *d_src = NULL, *d_dst = NULL;
    cudaError_t cerr;

#define CHECK_CUDA(call) \
    do { cerr = (call); if (cerr != cudaSuccess) { \
        fprintf(stderr, "debayer_gpu: CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(cerr)); \
        goto cleanup; } } while(0)

    CHECK_CUDA(cudaMalloc(&d_src, nbytes));
    CHECK_CUDA(cudaMalloc(&d_dst, nbytes));
    CHECK_CUDA(cudaMemcpyAsync(d_src, src->data, nbytes,
                               cudaMemcpyHostToDevice, stream));

    {
        DsoError err = debayer_gpu_d2d(d_src, d_dst, W, H, pattern, stream);
        if (err != DSO_OK) { cerr = cudaErrorUnknown; goto cleanup; }
    }

    CHECK_CUDA(cudaMemcpyAsync(dst->data, d_dst, nbytes,
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaFree(d_src); cudaFree(d_dst);
    return DSO_OK;

cleanup:
    cudaFree(d_src); cudaFree(d_dst);
    return DSO_ERR_CUDA;

#undef CHECK_CUDA
}
