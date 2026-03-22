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

    /* 1. Calculate 8 gradients in a 5x5 area */
    float g[8];
    float self = P(0, 0);

    /* North, South, East, West */
    g[0] = fabsf(P(0,-2) - P(0,0)) + fabsf(P(0,-1) - P(0,1))*0.5f + fabsf(P(-1,-1) - P(-1,1))*0.5f + fabsf(P(1,-1) - P(1,1))*0.5f; /* N */
    g[1] = fabsf(P(0, 2) - P(0,0)) + fabsf(P(0, 1) - P(0,-1))*0.5f + fabsf(P(-1, 1) - P(-1,-1))*0.5f + fabsf(P(1, 1) - P(1,-1))*0.5f; /* S */
    g[2] = fabsf(P( 2,0) - P(0,0)) + fabsf(P( 1,0) - P(-1,0))*0.5f + fabsf(P( 1,-1) - P(-1,-1))*0.5f + fabsf(P( 1,1) - P(-1,1))*0.5f; /* E */
    g[3] = fabsf(P(-2,0) - P(0,0)) + fabsf(P(-1,0) - P( 1,0))*0.5f + fabsf(P(-1,-1) - P( 1,-1))*0.5f + fabsf(P(-1,1) - P( 1,1))*0.5f; /* W */

    /* Diagonals */
    g[4] = fabsf(P( 2,-2) - P(0,0)) + fabsf(P( 1,-1) - P(-1, 1))*0.5f + fabsf(P(0,-1) - P(-1,0))*0.5f + fabsf(P( 1,0) - P(0, 1))*0.5f; /* NE */
    g[5] = fabsf(P(-2,-2) - P(0,0)) + fabsf(P(-1,-1) - P( 1, 1))*0.5f + fabsf(P(0,-1) - P( 1,0))*0.5f + fabsf(P(-1,0) - P(0, 1))*0.5f; /* NW */
    g[6] = fabsf(P( 2, 2) - P(0,0)) + fabsf(P( 1, 1) - P(-1,-1))*0.5f + fabsf(P(0, 1) - P( 1,0))*0.5f + fabsf(P(-1,0) - P(0,-1))*0.5f; /* SE */
    g[7] = fabsf(P(-2, 2) - P(0,0)) + fabsf(P(-1, 1) - P( 1,-1))*0.5f + fabsf(P(0, 1) - P(-1,0))*0.5f + fabsf(P( 1,0) - P(0,-1))*0.5f; /* SW */

    /* 2. Threshold */
    float gsum = 0.f, gmin = g[0], gmax = g[0];
    for (int i = 0; i < 8; i++) {
        gsum += g[i];
        if (g[i] < gmin) gmin = g[i];
        if (g[i] > gmax) gmax = g[i];
    }
    float tau = 1.5f * gmin + 0.5f * (gmax - gmin);

    /* 3. Interpolate R, G, B */
    float R_sum = 0.f, G_sum = 0.f, B_sum = 0.f;
    int R_count = 0, G_count = 0, B_count = 0;
    int chan = bayer_channel(gx, gy, pattern);

    const int dx[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};

    for (int i = 0; i < 8; i++) {
        if (g[i] <= tau) {
            int n_gx = gx + dx[i], n_gy = gy + dy[i];
            int nchan = bayer_channel(n_gx, n_gy, pattern);
            float nv = P(dx[i], dy[i]);
            
            if (nchan == 0)      { R_sum += nv; R_count++; }
            else if (nchan == 1) { G_sum += nv; G_count++; }
            else                 { B_sum += nv; B_count++; }
        }
    }

    if (chan == 0)      { R_sum += self; R_count++; }
    else if (chan == 1) { G_sum += self; G_count++; }
    else                { B_sum += self; B_count++; }

    float R_est = (R_count > 0) ? (R_sum / R_count) : self;
    float G_est = (G_count > 0) ? (G_sum / G_count) : self;
    float B_est = (B_count > 0) ? (B_sum / B_count) : self;

    /* Convert to luminance (ITU-R BT.709) */
    d_dst[gy * W + gx] = 0.2126f * R_est + 0.7152f * G_est + 0.0722f * B_est;

#undef P
}

/*
 * vng_debayer_rgb_kernel — same VNG algorithm as vng_debayer_kernel but
 * writes reconstructed R, G, B to three separate output arrays instead of
 * collapsing to luminance.
 */
__global__ static void vng_debayer_rgb_kernel(
    const float *d_src,
    float *d_dst_r, float *d_dst_g, float *d_dst_b,
    int W, int H, int pattern)
{
    __shared__ float sm[VNG_SMEM_H][VNG_SMEM_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * VNG_TILE_W + tx;
    int gy = blockIdx.y * VNG_TILE_H + ty;

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

    int sx = tx + VNG_APRON;
    int sy = ty + VNG_APRON;

#define P(dx, dy) sm[(sy)+(dy)][(sx)+(dx)]

    float g[8];
    float self = P(0, 0);

    g[0] = fabsf(P(0,-2) - P(0,0)) + fabsf(P(0,-1) - P(0,1))*0.5f + fabsf(P(-1,-1) - P(-1,1))*0.5f + fabsf(P(1,-1) - P(1,1))*0.5f;
    g[1] = fabsf(P(0, 2) - P(0,0)) + fabsf(P(0, 1) - P(0,-1))*0.5f + fabsf(P(-1, 1) - P(-1,-1))*0.5f + fabsf(P(1, 1) - P(1,-1))*0.5f;
    g[2] = fabsf(P( 2,0) - P(0,0)) + fabsf(P( 1,0) - P(-1,0))*0.5f + fabsf(P( 1,-1) - P(-1,-1))*0.5f + fabsf(P( 1,1) - P(-1,1))*0.5f;
    g[3] = fabsf(P(-2,0) - P(0,0)) + fabsf(P(-1,0) - P( 1,0))*0.5f + fabsf(P(-1,-1) - P( 1,-1))*0.5f + fabsf(P(-1,1) - P( 1,1))*0.5f;
    g[4] = fabsf(P( 2,-2) - P(0,0)) + fabsf(P( 1,-1) - P(-1, 1))*0.5f + fabsf(P(0,-1) - P(-1,0))*0.5f + fabsf(P( 1,0) - P(0, 1))*0.5f;
    g[5] = fabsf(P(-2,-2) - P(0,0)) + fabsf(P(-1,-1) - P( 1, 1))*0.5f + fabsf(P(0,-1) - P( 1,0))*0.5f + fabsf(P(-1,0) - P(0, 1))*0.5f;
    g[6] = fabsf(P( 2, 2) - P(0,0)) + fabsf(P( 1, 1) - P(-1,-1))*0.5f + fabsf(P(0, 1) - P( 1,0))*0.5f + fabsf(P(-1,0) - P(0,-1))*0.5f;
    g[7] = fabsf(P(-2, 2) - P(0,0)) + fabsf(P(-1, 1) - P( 1,-1))*0.5f + fabsf(P(0, 1) - P(-1,0))*0.5f + fabsf(P( 1,0) - P(0,-1))*0.5f;

    float gmin = g[0], gmax = g[0];
    for (int i = 1; i < 8; i++) {
        if (g[i] < gmin) gmin = g[i];
        if (g[i] > gmax) gmax = g[i];
    }
    float tau = 1.5f * gmin + 0.5f * (gmax - gmin);

    float R_sum = 0.f, G_sum = 0.f, B_sum = 0.f;
    int R_count = 0, G_count = 0, B_count = 0;
    int chan = bayer_channel(gx, gy, pattern);

    const int ddx[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const int ddy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};

    for (int i = 0; i < 8; i++) {
        if (g[i] <= tau) {
            int n_gx = gx + ddx[i], n_gy = gy + ddy[i];
            int nchan = bayer_channel(n_gx, n_gy, pattern);
            float nv = P(ddx[i], ddy[i]);
            if (nchan == 0)      { R_sum += nv; R_count++; }
            else if (nchan == 1) { G_sum += nv; G_count++; }
            else                 { B_sum += nv; B_count++; }
        }
    }

    if (chan == 0)      { R_sum += self; R_count++; }
    else if (chan == 1) { G_sum += self; G_count++; }
    else                { B_sum += self; B_count++; }

    int px = gy * W + gx;
    d_dst_r[px] = (R_count > 0) ? (R_sum / R_count) : self;
    d_dst_g[px] = (G_count > 0) ? (G_sum / G_count) : self;
    d_dst_b[px] = (B_count > 0) ? (B_sum / B_count) : self;

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

DsoError debayer_gpu_rgb_d2d(const float  *d_src,
                              float        *d_r,
                              float        *d_g,
                              float        *d_b,
                              int           W, int H,
                              BayerPattern  pattern,
                              cudaStream_t  stream)
{
    if (!d_src || !d_r || !d_g || !d_b || W <= 0 || H <= 0)
        return DSO_ERR_INVALID_ARG;

    if (pattern == BAYER_NONE) {
        size_t nbytes = (size_t)W * H * sizeof(float);
        cudaError_t ce;
        ce = cudaMemcpyAsync(d_r, d_src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) goto cuda_err;
        ce = cudaMemcpyAsync(d_g, d_src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) goto cuda_err;
        ce = cudaMemcpyAsync(d_b, d_src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) goto cuda_err;
        return DSO_OK;
    cuda_err:
        fprintf(stderr, "debayer_gpu_rgb_d2d (monochrome copy): %s\n",
                cudaGetErrorString(ce));
        return DSO_ERR_CUDA;
    }

    dim3 block(VNG_TILE_W, VNG_TILE_H);
    dim3 grid((W + VNG_TILE_W - 1) / VNG_TILE_W,
              (H + VNG_TILE_H - 1) / VNG_TILE_H);

    vng_debayer_rgb_kernel<<<grid, block, 0, stream>>>(
        d_src, d_r, d_g, d_b, W, H, (int)pattern);

    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "debayer_gpu_rgb_d2d kernel launch: %s\n",
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
