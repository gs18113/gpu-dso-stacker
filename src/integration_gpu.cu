/*
 * integration_gpu.cu — GPU mini-batch kappa-sigma integration.
 *
 * Strategy
 * --------
 * True whole-dataset kappa-sigma is impractical when N frames exceed GPU
 * memory.  Instead we use a mini-batch approximation:
 *
 *   For each batch of M frames (all pre-transformed, resident in d_frames[]):
 *     kappa_sigma_batch_kernel — one thread per pixel; loads M values from
 *     global memory into register arrays, runs iterative sigma-clipping
 *     (Bessel-corrected stddev), accumulates survivors into persistent
 *     d_combined_sum / d_combined_count buffers.
 *
 *   A parallel unclipped d_rawsum accumulates Σ all frame values for every
 *   pixel, providing a fallback for pixels whose every batch clips everything.
 *
 *   finalize_kernel converts accumulators to the final pixel value:
 *     out[px] = combined_sum / combined_count  if count > 0
 *             = rawsum / n_frames               otherwise
 *
 * Mean mode (no clipping)
 * -----------------------
 * mean_batch_kernel sums M frame values into d_combined_sum directly; no
 * clipping occurs and every frame is a survivor.
 *
 * Passing frame pointers to the kernel
 * -------------------------------------
 * d_frame_ptrs is a device-side float*[batch_size] array holding the M
 * device frame addresses for the current batch.  Updated via cudaMemcpyAsync
 * before each kernel launch so the kernel can index d_frame_ptrs[i][px].
 *
 * Register pressure
 * -----------------
 * The kappa_sigma_batch_kernel allocates float vals[M] + char active[M]
 * per thread.  For M ≤ 32 this fits in registers (~40 regs); for larger M
 * values may spill to local (L1-cached) memory.  Recommended range: M ≤ 32.
 */

#include "integration_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* IntegrationGpuCtx is defined in integration_gpu.h (must be visible to pipeline.cu). */

/* -------------------------------------------------------------------------
 * CUDA error-check macro
 * ------------------------------------------------------------------------- */

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "integration_gpu CUDA error %s:%d — %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

/* -------------------------------------------------------------------------
 * GPU kernels
 * ------------------------------------------------------------------------- */

/*
 * kappa_sigma_batch_kernel — per-pixel kappa-sigma clipping over M frames.
 *
 * Each thread handles one pixel, loading M values into registers, running
 * iterative sigma-clipping, then updating the persistent accumulators.
 *
 * d_frame_ptrs : array of M device pointers, each pointing to one W*H frame
 * M            : number of frames in this batch (1 ≤ M ≤ INTEGRATION_GPU_MAX_BATCH)
 * kappa        : sigma-clipping threshold (e.g. 3.0)
 * iterations   : max clipping passes (e.g. 3)
 * d_combined_sum   : accumulator for survivor-weighted sums (read-modify-write)
 * d_combined_count : accumulator for total survivor counts (read-modify-write)
 * d_rawsum         : accumulator for all raw values (read-modify-write, fallback)
 * npix         : W * H
 */
__global__ static void kappa_sigma_batch_kernel(
    float * const *d_frame_ptrs,
    int            M,
    float          kappa,
    int            iterations,
    float         *d_combined_sum,
    int           *d_combined_count,
    float         *d_rawsum,
    int            npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    /* Load M values into registers; accumulate raw sum for fallback */
    float vals[INTEGRATION_GPU_MAX_BATCH];
    char  active[INTEGRATION_GPU_MAX_BATCH];   /* 1 = not yet clipped */
    float raw_total = 0.f;

    for (int i = 0; i < M; i++) {
        float v = d_frame_ptrs[i][px];
        vals[i]   = v;
        active[i] = 1;
        raw_total += v;
    }

    /* Accumulate raw sum across batches for the all-clipped fallback */
    d_rawsum[px] += raw_total;

    /* Iterative kappa-sigma clipping (mirrors integrate_kappa_sigma in integration.c) */
    int n_active = M;

    for (int iter = 0; iter < iterations; iter++) {
        if (n_active < 2) break;  /* need ≥ 2 samples for Bessel-corrected stddev */

        /* Mean of active values */
        float sum = 0.f;
        for (int i = 0; i < M; i++) if (active[i]) sum += vals[i];
        float mean = sum / (float)n_active;

        /* Bessel-corrected sample variance */
        float sq = 0.f;
        for (int i = 0; i < M; i++) {
            if (active[i]) {
                float d = vals[i] - mean;
                sq += d * d;
            }
        }
        float sigma  = sqrtf(sq / (float)(n_active - 1));
        float thresh = kappa * sigma;

        /* Reject outliers */
        int rejected = 0;
        for (int i = 0; i < M; i++) {
            if (active[i] && fabsf(vals[i] - mean) > thresh) {
                active[i] = 0;
                rejected++;
            }
        }
        n_active -= rejected;
        if (rejected == 0) break;  /* converged */
    }

    /* Accumulate surviving values into the persistent combined buffers.
     * When n_active == 0 (all clipped), skip — finalize will use rawsum. */
    if (n_active > 0) {
        float clipped_sum = 0.f;
        for (int i = 0; i < M; i++) if (active[i]) clipped_sum += vals[i];
        d_combined_sum[px]   += clipped_sum;
        d_combined_count[px] += n_active;
    }
}

/*
 * mean_batch_kernel — accumulate M frames without any sigma clipping.
 *
 * All M values are treated as survivors and added to both combined and raw
 * accumulators, so integration_gpu_finalize works with its standard formula.
 */
__global__ static void mean_batch_kernel(
    float * const *d_frame_ptrs,
    int            M,
    float         *d_combined_sum,
    int           *d_combined_count,
    float         *d_rawsum,
    int            npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    float sum = 0.f;
    for (int i = 0; i < M; i++) sum += d_frame_ptrs[i][px];

    d_combined_sum[px]   += sum;
    d_combined_count[px] += M;
    d_rawsum[px]         += sum;
}

/*
 * finalize_kernel — compute the final pixel value from accumulators.
 *
 * Primary: combined_sum / combined_count  (survivor-weighted mean)
 * Fallback: rawsum / n_frames             (unclipped mean, when all clipped)
 */
__global__ static void finalize_kernel(
    const float *d_combined_sum,
    const int   *d_combined_count,
    const float *d_rawsum,
    float       *d_out,
    int          n_frames,
    int          npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    int c = d_combined_count[px];
    if (c > 0) {
        d_out[px] = d_combined_sum[px] / (float)c;
    } else {
        /* Degenerate: every value was clipped in every batch — fall back to
         * the unclipped mean (same behaviour as the CPU implementation). */
        d_out[px] = (n_frames > 0) ? d_rawsum[px] / (float)n_frames : 0.f;
    }
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

DsoError integration_gpu_init(int W, int H, int batch_size,
                               IntegrationGpuCtx **ctx_out)
{
    if (!ctx_out || W <= 0 || H <= 0 ||
        batch_size < 1 || batch_size > INTEGRATION_GPU_MAX_BATCH)
        return DSO_ERR_INVALID_ARG;

    IntegrationGpuCtx *ctx =
        (IntegrationGpuCtx *)calloc(1, sizeof(IntegrationGpuCtx));
    if (!ctx) return DSO_ERR_ALLOC;

    ctx->W          = W;
    ctx->H          = H;
    ctx->batch_size = batch_size;

    size_t npix_f   = (size_t)W * H * sizeof(float);
    size_t npix_i   = (size_t)W * H * sizeof(int);
    cudaError_t cerr;

    /* Persistent accumulator buffers (zeroed) */
    if ((cerr = cudaMalloc(&ctx->d_combined_sum,   npix_f)) != cudaSuccess) goto oom_cuda;
    if ((cerr = cudaMalloc(&ctx->d_combined_count, npix_i)) != cudaSuccess) goto oom_cuda;
    if ((cerr = cudaMalloc(&ctx->d_rawsum,         npix_f)) != cudaSuccess) goto oom_cuda;
    cudaMemset(ctx->d_combined_sum,   0, npix_f);
    cudaMemset(ctx->d_combined_count, 0, npix_i);
    cudaMemset(ctx->d_rawsum,         0, npix_f);

    /* Finalize staging output buffer */
    if ((cerr = cudaMalloc(&ctx->d_out, npix_f)) != cudaSuccess) goto oom_cuda;

    /* Lanczos coordinate-map buffers (shared across frames in phase 2) */
    if ((cerr = cudaMalloc(&ctx->d_xmap, npix_f)) != cudaSuccess) goto oom_cuda;
    if ((cerr = cudaMalloc(&ctx->d_ymap, npix_f)) != cudaSuccess) goto oom_cuda;

    /* Per-batch frame buffers */
    for (int i = 0; i < batch_size; i++) {
        if ((cerr = cudaMalloc(&ctx->d_frames[i], npix_f)) != cudaSuccess)
            goto oom_cuda;
    }

    /* Device-side array of frame pointers (updated before each batch launch) */
    if ((cerr = cudaMalloc(&ctx->d_frame_ptrs,
                            (size_t)batch_size * sizeof(float *)))
        != cudaSuccess) goto oom_cuda;

    *ctx_out = ctx;
    return DSO_OK;

oom_cuda:
    fprintf(stderr, "integration_gpu_init: cudaMalloc failed — %s\n",
            cudaGetErrorString(cerr));
    integration_gpu_cleanup(ctx);
    return DSO_ERR_CUDA;
}

void integration_gpu_cleanup(IntegrationGpuCtx *ctx)
{
    if (!ctx) return;

    cudaFree(ctx->d_combined_sum);
    cudaFree(ctx->d_combined_count);
    cudaFree(ctx->d_rawsum);
    cudaFree(ctx->d_out);
    cudaFree(ctx->d_xmap);
    cudaFree(ctx->d_ymap);
    cudaFree(ctx->d_frame_ptrs);

    for (int i = 0; i < ctx->batch_size; i++)
        cudaFree(ctx->d_frames[i]);

    free(ctx);
}

DsoError integration_gpu_process_batch(IntegrationGpuCtx *ctx,
                                        int                M,
                                        float              kappa,
                                        int                iterations,
                                        cudaStream_t       stream)
{
    if (!ctx || M < 1 || M > ctx->batch_size) return DSO_ERR_INVALID_ARG;

    int npix = ctx->W * ctx->H;

    /* Copy the M frame pointers (host values) to the device pointer array.
     * This must be enqueued into the same stream before the kernel so the
     * device sees the correct addresses. */
    cudaError_t cerr = cudaMemcpyAsync(
        ctx->d_frame_ptrs,
        ctx->d_frames,                    /* host array of float* device ptrs  */
        (size_t)M * sizeof(float *),
        cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch: memcpy ptrs: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    /* Launch kappa-sigma kernel: 256 threads / block, 1-D grid */
    dim3 block(256);
    dim3 grid((npix + 255) / 256);

    kappa_sigma_batch_kernel<<<grid, block, 0, stream>>>(
        ctx->d_frame_ptrs, M, kappa, iterations,
        ctx->d_combined_sum, ctx->d_combined_count, ctx->d_rawsum, npix);

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch kernel: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }
    return DSO_OK;
}

DsoError integration_gpu_process_batch_mean(IntegrationGpuCtx *ctx,
                                             int                M,
                                             cudaStream_t       stream)
{
    if (!ctx || M < 1 || M > ctx->batch_size) return DSO_ERR_INVALID_ARG;

    int npix = ctx->W * ctx->H;

    /* Update device-side frame pointer array for this batch */
    cudaError_t cerr = cudaMemcpyAsync(
        ctx->d_frame_ptrs,
        ctx->d_frames,
        (size_t)M * sizeof(float *),
        cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch_mean: memcpy ptrs: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    dim3 block(256);
    dim3 grid((npix + 255) / 256);

    mean_batch_kernel<<<grid, block, 0, stream>>>(
        ctx->d_frame_ptrs, M,
        ctx->d_combined_sum, ctx->d_combined_count, ctx->d_rawsum, npix);

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch_mean kernel: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }
    return DSO_OK;
}

DsoError integration_gpu_finalize(IntegrationGpuCtx *ctx,
                                   int                n_frames,
                                   Image             *out,
                                   cudaStream_t       stream)
{
    if (!ctx || !out || !out->data || n_frames <= 0) return DSO_ERR_INVALID_ARG;
    if (out->width != ctx->W || out->height != ctx->H)  return DSO_ERR_INVALID_ARG;

    int npix = ctx->W * ctx->H;

    /* Compute final pixel values on device */
    dim3 block(256);
    dim3 grid((npix + 255) / 256);

    finalize_kernel<<<grid, block, 0, stream>>>(
        ctx->d_combined_sum, ctx->d_combined_count, ctx->d_rawsum,
        ctx->d_out, n_frames, npix);

    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_finalize kernel: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    /* Download result to host */
    cerr = cudaMemcpyAsync(out->data, ctx->d_out,
                            (size_t)npix * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_finalize D2H: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    /* Synchronise so the caller can use out->data immediately */
    cerr = cudaStreamSynchronize(stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_finalize sync: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    /* Reset accumulators so the context can be reused for another run */
    size_t npix_f = (size_t)npix * sizeof(float);
    size_t npix_i = (size_t)npix * sizeof(int);
    cudaMemset(ctx->d_combined_sum,   0, npix_f);
    cudaMemset(ctx->d_combined_count, 0, npix_i);
    cudaMemset(ctx->d_rawsum,         0, npix_f);

    return DSO_OK;
}
