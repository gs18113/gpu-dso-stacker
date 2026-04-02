/*
 * star_detect_gpu.cu — GPU Moffat convolution and sigma-threshold masking.
 *
 * Moffat convolution:
 *   The kernel K(i,j) = [1 + (i²+j²)/alpha²]^(-beta) is computed on the CPU,
 *   normalised to sum=1, and stored in GPU constant memory (up to 31×31 = 961
 *   elements, ≈ 3.8 KB, well within the 64 KB constant cache limit).
 *
 *   The convolution kernel uses shared-memory tiling: each 16×16 block loads
 *   a (16+2R)×(16+2R) apron into shared memory and then each thread multiplies
 *   the (2R+1)×(2R+1) kernel against shared memory to produce one output pixel.
 *   Boundary pixels outside the image are zero-padded.
 *
 * Threshold masking:
 *   A two-pass GPU reduction computes the global mean μ and sample standard
 *   deviation σ of the convolved image; an element-wise kernel then writes
 *   mask[i] = (conv[i] > μ + sigma_k * σ) ? 1 : 0.
 *
 * The d2d variant keeps all data on the device and executes asynchronously
 * on the caller-supplied CUDA stream.  The h2h wrapper allocates device memory
 * and synchronises before returning so the result is immediately available
 * on the host.
 */

#include "star_detect_gpu.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------
 * Moffat kernel in constant memory
 *
 * Maximum kernel diameter: 31 pixels (alpha ≤ 5, radius = ceil(3*alpha) = 15,
 * diameter = 31).  The actual used portion is determined at runtime.
 * ------------------------------------------------------------------------- */
#define MOFFAT_MAX_RADIUS 15
#define MOFFAT_MAX_DIAM   (2*MOFFAT_MAX_RADIUS + 1)  /* = 31 */
#define MOFFAT_MAX_ELEMS  (MOFFAT_MAX_DIAM * MOFFAT_MAX_DIAM)  /* = 961 */

__constant__ float c_kernel[MOFFAT_MAX_ELEMS];
__constant__ int   c_kradius;   /* kernel radius currently uploaded */

/* -------------------------------------------------------------------------
 * Convolution kernel (shared memory tiling)
 * ------------------------------------------------------------------------- */
#define CONV_TILE_W 16
#define CONV_TILE_H 16

/*
 * moffat_conv_kernel — tile-based 2D convolution with kernel in constant mem.
 *
 * d_src : input image (W×H float32)
 * d_dst : output image (W×H float32), written by this kernel
 * W, H  : image dimensions
 *
 * Shared memory size is computed at launch from the kernel radius.
 * The kernel assumes zero-padding at image boundaries.
 */
__global__ static void moffat_conv_kernel(
    const float *d_src, float *d_dst, int W, int H)
{
    extern __shared__ float sm[];  /* (CONV_TILE_W + 2*kradius)^2 floats */

    int kradius = c_kradius;
    int kw = 2 * kradius + 1;     /* kernel width */
    int smw = CONV_TILE_W + 2 * kradius;  /* shared memory tile width */

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * CONV_TILE_W + tx;
    int gy = blockIdx.y * CONV_TILE_H + ty;

    /* Load shared memory tile with apron.
     * Threads cooperatively load the (smw × smh) region. */
    int smh = CONV_TILE_H + 2 * kradius;
    for (int dy = ty; dy < smh; dy += CONV_TILE_H) {
        for (int dx = tx; dx < smw; dx += CONV_TILE_W) {
            int sx = blockIdx.x * CONV_TILE_W + dx - kradius;
            int sy = blockIdx.y * CONV_TILE_H + dy - kradius;
            float v = 0.f;
            if (sx >= 0 && sx < W && sy >= 0 && sy < H)
                v = d_src[sy * W + sx];
            sm[dy * smw + dx] = v;
        }
    }
    __syncthreads();

    if (gx >= W || gy >= H) return;

    /* Apply convolution using shared memory */
    float acc = 0.f;
    int sy0 = ty;  /* top of kernel in shared memory */
    int sx0 = tx;

    for (int ky = 0; ky < kw; ky++) {
        for (int kx = 0; kx < kw; kx++) {
            float kv = c_kernel[ky * kw + kx];
            acc += kv * sm[(sy0 + ky) * smw + (sx0 + kx)];
        }
    }

    d_dst[gy * W + gx] = acc;
}

/* -------------------------------------------------------------------------
 * Reduction kernels for mean + variance
 * ------------------------------------------------------------------------- */

#define REDUCE_BLOCK 256

/* Pass 1: compute partial sums into d_partials. */
__global__ static void reduce_sum_kernel(
    const float *d_in, double *d_partials, int N)
{
    __shared__ double sm_sum[REDUCE_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (gid < N) val = (double)d_in[gid];
    sm_sum[tid] = val;
    __syncthreads();

    /* Tree reduction */
    for (int s = REDUCE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sm_sum[tid] += sm_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_partials[blockIdx.x] = sm_sum[0];
}

/* Final pass: reduces partials to a single value. */
__global__ static void reduce_final_kernel(
    const double *d_partials, int n_partials, double *d_out)
{
    __shared__ double sm_sum[REDUCE_BLOCK];
    int tid = threadIdx.x;

    double val = 0.0;
    for (int i = tid; i < n_partials; i += REDUCE_BLOCK) {
        val += d_partials[i];
    }
    sm_sum[tid] = val;
    __syncthreads();

    for (int s = REDUCE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sm_sum[tid] += sm_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) *d_out = sm_sum[0];
}

/* Helper to divide by N on device */
__global__ static void reduce_div_n_kernel(double *d_val, int N)
{
    if (N > 0) *d_val /= N;
}

/* Pass 2: compute partial sum-of-squares (for variance) given mean. */
__global__ static void reduce_sumsq_kernel(
    const float *d_in, const double *d_mean, double *d_partials, int N)
{
    __shared__ double sm_sq[REDUCE_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double mean = *d_mean;

    double v = 0.0;
    if (gid < N) { double d = (double)d_in[gid] - mean; v = d * d; }
    sm_sq[tid] = v;
    __syncthreads();

    for (int s = REDUCE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sm_sq[tid] += sm_sq[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_partials[blockIdx.x] = sm_sq[0];
}

/* Threshold application kernel using device-resident mean and sq_sum */
__global__ static void threshold_auto_kernel(
    const float *d_conv, uint8_t *d_mask, 
    const double *d_mean, const double *d_sq_sum,
    float sigma_k, int N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    double mean   = *d_mean;
    double sq_sum = *d_sq_sum;
    double var    = (N > 1) ? (sq_sum / (N - 1)) : 0.0;
    double sigma  = sqrt(var);
    float  thresh = (float)(mean + (double)sigma_k * sigma);

    d_mask[gid] = (d_conv[gid] > thresh) ? 1 : 0;
}

/* -------------------------------------------------------------------------
 * Build Moffat kernel on CPU and upload to constant memory
 * ------------------------------------------------------------------------- */

/* Cached Moffat parameters to avoid redundant per-frame uploads */
static float s_cached_alpha = -1.f;
static float s_cached_beta  = -1.f;
static int   s_cached_R     = 0;

/* Pre-allocated reduction scratch buffer (persists across frames) */
static double *s_reduce_scratch      = NULL;
static size_t  s_reduce_scratch_size = 0;

static DsoError upload_moffat_kernel(const MoffatParams *params,
                                      int *radius_out)
{
    if (!params || params->alpha <= 0.f || params->beta <= 0.f)
        return DSO_ERR_INVALID_ARG;

    /* Kernel radius = ceil(3 * alpha); clamp to MOFFAT_MAX_RADIUS. */
    int R = (int)ceilf(3.0f * params->alpha);
    if (R > MOFFAT_MAX_RADIUS) {
        fprintf(stderr, "star_detect_gpu: Moffat alpha=%.2f yields radius %d, "
                "clamping to MOFFAT_MAX_RADIUS=%d (kernel will be under-sized)\n",
                params->alpha, R, MOFFAT_MAX_RADIUS);
        R = MOFFAT_MAX_RADIUS;
    }
    if (R < 1) R = 1;

    /* Skip re-upload if parameters haven't changed */
    if (params->alpha == s_cached_alpha && params->beta == s_cached_beta) {
        *radius_out = s_cached_R;
        return DSO_OK;
    }

    int kw = 2 * R + 1;
    float kbuf[MOFFAT_MAX_ELEMS];
    double ksum = 0.0;

    float alpha2 = params->alpha * params->alpha;
    float neg_beta = -params->beta;

    for (int j = -R; j <= R; j++) {
        for (int i = -R; i <= R; i++) {
            float r2 = (float)(i*i + j*j);
            float v = powf(1.0f + r2 / alpha2, neg_beta);
            kbuf[(j+R) * kw + (i+R)] = v;
            ksum += (double)v;
        }
    }

    /* Normalise kernel to sum = 1 (double division for accuracy). */
    double inv_ksum = 1.0 / ksum;
    for (int k = 0; k < kw * kw; k++) kbuf[k] = (float)((double)kbuf[k] * inv_ksum);

    /* Upload to constant memory. */
    cudaError_t cerr;
    cerr = cudaMemcpyToSymbol(c_kernel, kbuf, (size_t)kw*kw * sizeof(float));
    if (cerr != cudaSuccess) {
        fprintf(stderr, "upload_moffat_kernel: %s\n", cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }
    cerr = cudaMemcpyToSymbol(c_kradius, &R, sizeof(int));
    if (cerr != cudaSuccess) {
        fprintf(stderr, "upload_moffat_kernel (radius): %s\n", cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    /* Cache the parameters */
    s_cached_alpha = params->alpha;
    s_cached_beta  = params->beta;
    s_cached_R     = R;

    *radius_out = R;
    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * Public d2d implementation
 * ------------------------------------------------------------------------- */

DsoError star_detect_gpu_d2d(const float        *d_src,
                              float              *d_conv,
                              uint8_t            *d_mask,
                              int                 W, int H,
                              const MoffatParams *params,
                              float               sigma_k,
                              cudaStream_t        stream)
{
    if (!d_src || !d_conv || !d_mask || W <= 0 || H <= 0 || !params)
        return DSO_ERR_INVALID_ARG;

    int R;
    DsoError err = upload_moffat_kernel(params, &R);
    if (err != DSO_OK) return err;

    /* Launch Moffat convolution kernel */
    int smw = CONV_TILE_W + 2 * R;
    int smh = CONV_TILE_H + 2 * R;
    size_t smem = (size_t)smw * smh * sizeof(float);

    dim3 block(CONV_TILE_W, CONV_TILE_H);
    dim3 grid((W + CONV_TILE_W - 1) / CONV_TILE_W,
              (H + CONV_TILE_H - 1) / CONV_TILE_H);

    moffat_conv_kernel<<<grid, block, smem, stream>>>(d_src, d_conv, W, H);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "moffat_conv_kernel: %s\n", cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    /* Compute mean and sigma fully on GPU (Asynchronously).
     * The reduction scratch buffer is pre-allocated and reused across frames
     * to avoid per-frame cudaMalloc + cudaStreamSynchronize + cudaFree. */
    int N = W * H;
    int n_partials = (N + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
    size_t needed = (size_t)n_partials * sizeof(double) + 2 * sizeof(double);

    /* Grow the persistent scratch buffer if needed */
    if (needed > s_reduce_scratch_size) {
        if (s_reduce_scratch) cudaFree(s_reduce_scratch);
        cerr = cudaMalloc(&s_reduce_scratch, needed);
        if (cerr != cudaSuccess) {
            s_reduce_scratch = NULL;
            s_reduce_scratch_size = 0;
            return DSO_ERR_ALLOC;
        }
        s_reduce_scratch_size = needed;
    }

    double *d_partials = s_reduce_scratch;
    double *d_mean     = d_partials + n_partials;
    double *d_sq_sum   = d_mean + 1;

    /* 1. Sum partials */
    reduce_sum_kernel<<<n_partials, REDUCE_BLOCK, 0, stream>>>(d_conv, d_partials, N);
    /* 2. Final sum -> mean */
    reduce_final_kernel<<<1, REDUCE_BLOCK, 0, stream>>>(d_partials, n_partials, d_mean);
    reduce_div_n_kernel<<<1, 1, 0, stream>>>(d_mean, N);

    /* 3. Sumsq partials */
    reduce_sumsq_kernel<<<n_partials, REDUCE_BLOCK, 0, stream>>>(d_conv, d_mean, d_partials, N);
    /* 4. Final sumsq */
    reduce_final_kernel<<<1, REDUCE_BLOCK, 0, stream>>>(d_partials, n_partials, d_sq_sum);

    /* 5. Threshold application */
    dim3 blk_t(REDUCE_BLOCK);
    dim3 grd_t((N + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
    threshold_auto_kernel<<<grd_t, blk_t, 0, stream>>>(d_conv, d_mask, d_mean, d_sq_sum, sigma_k, N);

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "star_detect_gpu_d2d reduction/threshold: %s\n", cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * h2h wrapper: star_detect_gpu_moffat_convolve
 * ------------------------------------------------------------------------- */

DsoError star_detect_gpu_moffat_convolve(const Image        *src,
                                          Image              *dst,
                                          const MoffatParams *params,
                                          cudaStream_t        stream)
{
    if (!src || !dst || !src->data || !dst->data || !params)
        return DSO_ERR_INVALID_ARG;
    if (src->width != dst->width || src->height != dst->height)
        return DSO_ERR_INVALID_ARG;

    int W = src->width, H = src->height;
    size_t nbytes = (size_t)W * H * sizeof(float);
    float *d_src = NULL, *d_dst = NULL;
    cudaError_t cerr;

    int R;
    DsoError derr = upload_moffat_kernel(params, &R);
    if (derr != DSO_OK) return derr;

    cerr = cudaMalloc(&d_src, nbytes); if (cerr) { return DSO_ERR_CUDA; }
    cerr = cudaMalloc(&d_dst, nbytes); if (cerr) { cudaFree(d_src); return DSO_ERR_CUDA; }

    cerr = cudaMemcpyAsync(d_src, src->data, nbytes, cudaMemcpyHostToDevice, stream);
    if (cerr) goto cleanup;

    {
        int smw = CONV_TILE_W + 2 * R, smh = CONV_TILE_H + 2 * R;
        size_t smem = (size_t)smw * smh * sizeof(float);
        dim3 block(CONV_TILE_W, CONV_TILE_H);
        dim3 grid((W + CONV_TILE_W - 1) / CONV_TILE_W,
                  (H + CONV_TILE_H - 1) / CONV_TILE_H);
        moffat_conv_kernel<<<grid, block, smem, stream>>>(d_src, d_dst, W, H);
        cerr = cudaGetLastError(); if (cerr) goto cleanup;
    }

    cerr = cudaMemcpyAsync(dst->data, d_dst, nbytes, cudaMemcpyDeviceToHost, stream);
    if (cerr) goto cleanup;
    cerr = cudaStreamSynchronize(stream); if (cerr) goto cleanup;

    cudaFree(d_src); cudaFree(d_dst);
    return DSO_OK;
cleanup:
    cudaFree(d_src); cudaFree(d_dst);
    fprintf(stderr, "star_detect_gpu_moffat_convolve: %s\n", cudaGetErrorString(cerr));
    return DSO_ERR_CUDA;
}

/* -------------------------------------------------------------------------
 * h2h wrapper: star_detect_gpu_threshold
 * ------------------------------------------------------------------------- */

DsoError star_detect_gpu_threshold(const Image  *convolved,
                                    uint8_t      *mask_out,
                                    float         sigma_k,
                                    cudaStream_t  stream)
{
    if (!convolved || !convolved->data || !mask_out) return DSO_ERR_INVALID_ARG;

    int W = convolved->width, H = convolved->height;
    int N = W * H;
    size_t nbytes = (size_t)N * sizeof(float);
    size_t mbytes = (size_t)N;

    float   *d_conv = NULL;
    uint8_t *d_mask = NULL;
    cudaError_t cerr;

    cerr = cudaMalloc(&d_conv, nbytes); if (cerr) goto cleanup;
    cerr = cudaMalloc(&d_mask, mbytes); if (cerr) goto cleanup;

    cerr = cudaMemcpyAsync(d_conv, convolved->data, nbytes,
                           cudaMemcpyHostToDevice, stream);
    if (cerr) goto cleanup;

    {
        int n_partials = (N + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
        double *d_scratch = NULL;
        cerr = cudaMalloc(&d_scratch, (size_t)n_partials * sizeof(double) + 2 * sizeof(double));
        if (cerr != cudaSuccess) goto cleanup;
        double *d_mean = d_scratch + n_partials;
        double *d_sq_sum = d_mean + 1;

        reduce_sum_kernel<<<n_partials, REDUCE_BLOCK, 0, stream>>>(d_conv, d_scratch, N);
        reduce_final_kernel<<<1, REDUCE_BLOCK, 0, stream>>>(d_scratch, n_partials, d_mean);
        reduce_div_n_kernel<<<1, 1, 0, stream>>>(d_mean, N);

        reduce_sumsq_kernel<<<n_partials, REDUCE_BLOCK, 0, stream>>>(d_conv, d_mean, d_scratch, N);
        reduce_final_kernel<<<1, REDUCE_BLOCK, 0, stream>>>(d_scratch, n_partials, d_sq_sum);

        dim3 blk(REDUCE_BLOCK), grd((N + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
        threshold_auto_kernel<<<grd, blk, 0, stream>>>(d_conv, d_mask, d_mean, d_sq_sum, sigma_k, N);
        
        cudaStreamSynchronize(stream);
        cudaFree(d_scratch);
    }

    cerr = cudaMemcpyAsync(mask_out, d_mask, mbytes,
                           cudaMemcpyDeviceToHost, stream);
    if (cerr) goto cleanup;
    cerr = cudaStreamSynchronize(stream);

cleanup:
    cudaFree(d_conv); cudaFree(d_mask);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "star_detect_gpu_threshold: %s\n", cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }
    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * star_detect_gpu_cleanup — free persistent GPU resources.
 * ------------------------------------------------------------------------- */

void star_detect_gpu_cleanup(void)
{
    if (s_reduce_scratch) {
        cudaFree(s_reduce_scratch);
        s_reduce_scratch = NULL;
        s_reduce_scratch_size = 0;
    }
    s_cached_alpha = -1.f;
    s_cached_beta  = -1.f;
    s_cached_R     = 0;
}
