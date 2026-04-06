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
    double        *d_combined_sum,
    int           *d_combined_count,
    double        *d_rawsum,
    int           *d_rawcount,
    int            npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    /* Load M values into registers; skip NaN (OOB sentinel from Lanczos warp) */
    float vals[INTEGRATION_GPU_MAX_BATCH];
    char  active[INTEGRATION_GPU_MAX_BATCH];   /* 1 = not yet clipped */
    double raw_total = 0.0;
    int   n_valid   = 0;

    for (int i = 0; i < M; i++) {
        float v = d_frame_ptrs[i][px];
        if (isnan(v)) {
            vals[i]   = 0.f;
            active[i] = 0;
        } else {
            vals[i]   = v;
            active[i] = 1;
            raw_total += (double)v;
            n_valid++;
        }
    }

    /* Accumulate raw sum and count across batches for the all-clipped fallback */
    d_rawsum[px]   += raw_total;
    d_rawcount[px] += n_valid;

    /* If no frames have valid data at this pixel in this batch, skip */
    if (n_valid == 0) return;

    /* Iterative kappa-sigma clipping (mirrors integrate_kappa_sigma in integration.c) */
    int n_active = n_valid;

    for (int iter = 0; iter < iterations; iter++) {
        if (n_active < 2) break;  /* need ≥ 2 samples for Bessel-corrected stddev */

        /* Mean of active values (double precision to match CPU path) */
        double sum = 0.0;
        for (int i = 0; i < M; i++) if (active[i]) sum += (double)vals[i];
        double mean = sum / (double)n_active;

        /* Bessel-corrected sample variance (double to avoid FP loss for
         * high-dynamic-range pixels; matches CPU integrate_kappa_sigma) */
        double sq = 0.0;
        for (int i = 0; i < M; i++) {
            if (active[i]) {
                double d = (double)vals[i] - mean;
                sq += d * d;
            }
        }
        float sigma  = (float)sqrt(sq / (double)(n_active - 1));
        float thresh = kappa * sigma;

        /* Reject outliers */
        int rejected = 0;
        for (int i = 0; i < M; i++) {
            if (active[i] && fabs((double)vals[i] - mean) > (double)thresh) {
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
        double clipped_sum = 0.0;
        for (int i = 0; i < M; i++) if (active[i]) clipped_sum += (double)vals[i];
        d_combined_sum[px]   += clipped_sum;
        d_combined_count[px] += n_active;
    }
}

/*
 * aawa_batch_kernel — Auto Adaptive Weighted Average (Stetson 1989).
 *
 * Each thread handles one pixel: loads M values, runs iterative weighted
 * averaging with Stetson weight w = 1/(1+(|r|/α)²), α=2.0, max 10 iters.
 * Accumulates result into persistent combined buffers.
 */
#define AAWA_GPU_ALPHA    2.0
#define AAWA_GPU_MAX_ITER 10

__global__ static void aawa_batch_kernel(
    float * const *d_frame_ptrs,
    int            M,
    double        *d_combined_sum,
    int           *d_combined_count,
    double        *d_rawsum,
    int           *d_rawcount,
    int            npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    /* Load M values; skip NaN */
    float  vals[INTEGRATION_GPU_MAX_BATCH];
    double raw_total = 0.0;
    int    n_valid   = 0;

    for (int i = 0; i < M; i++) {
        float v = d_frame_ptrs[i][px];
        if (!isnan(v)) {
            vals[n_valid++] = v;
            raw_total += (double)v;
        }
    }

    d_rawsum[px]   += raw_total;
    d_rawcount[px] += n_valid;

    if (n_valid == 0) return;

    /* Initial mean */
    double mu = 0.0;
    for (int i = 0; i < n_valid; i++) mu += (double)vals[i];
    mu /= (double)n_valid;

    if (n_valid == 1) {
        d_combined_sum[px]   += mu;
        d_combined_count[px] += 1;
        return;
    }

    /* Initial Bessel-corrected stddev */
    double sq = 0.0;
    for (int i = 0; i < n_valid; i++) {
        double d = (double)vals[i] - mu;
        sq += d * d;
    }
    double sigma = sqrt(sq / (double)(n_valid - 1));

    if (sigma < 1e-12) {
        d_combined_sum[px]   += mu * n_valid;
        d_combined_count[px] += n_valid;
        return;
    }

    /* Iterative Stetson weighted average */
    for (int iter = 0; iter < AAWA_GPU_MAX_ITER; iter++) {
        double sum_w  = 0.0;
        double sum_wv = 0.0;

        for (int i = 0; i < n_valid; i++) {
            double r  = ((double)vals[i] - mu) / sigma;
            double ra = fabs(r) / AAWA_GPU_ALPHA;
            double w  = 1.0 / (1.0 + ra * ra);
            sum_w  += w;
            sum_wv += w * (double)vals[i];
        }

        if (sum_w < 1e-12) break;

        double mu_new = sum_wv / sum_w;

        double sum_wsq = 0.0;
        for (int i = 0; i < n_valid; i++) {
            double r  = ((double)vals[i] - mu) / sigma;
            double ra = fabs(r) / AAWA_GPU_ALPHA;
            double w  = 1.0 / (1.0 + ra * ra);
            double d  = (double)vals[i] - mu_new;
            sum_wsq += w * d * d;
        }
        double sigma_new = sqrt(sum_wsq / sum_w);

        double denom = fabs(mu) > 1e-10 ? fabs(mu) : 1e-10;
        if (fabs(mu_new - mu) / denom < 1e-6) {
            mu = mu_new;
            break;
        }

        mu    = mu_new;
        sigma = (sigma_new > 1e-12) ? sigma_new : 1e-12;
    }

    d_combined_sum[px]   += mu * n_valid;
    d_combined_count[px] += n_valid;
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
    double        *d_combined_sum,
    int           *d_combined_count,
    double        *d_rawsum,
    int            npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    double sum = 0.0;
    int   valid = 0;
    for (int i = 0; i < M; i++) {
        float v = d_frame_ptrs[i][px];
        if (!isnan(v)) { sum += (double)v; valid++; }
    }

    if (valid > 0) {
        d_combined_sum[px]   += sum;
        d_combined_count[px] += valid;
    }
    d_rawsum[px] += sum;
}

/*
 * median_batch_kernel — per-pixel median over M frames (mini-batch).
 *
 * Each thread loads M values, skips NaN, sorts valid values via insertion
 * sort in registers, computes the median, and accumulates into persistent
 * buffers weighted by n_valid.
 */
__global__ static void median_batch_kernel(
    float * const *d_frame_ptrs,
    int            M,
    double        *d_combined_sum,
    int           *d_combined_count,
    double        *d_rawsum,
    int           *d_rawcount,
    int            npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    /* Load M values into registers; skip NaN */
    float vals[INTEGRATION_GPU_MAX_BATCH];
    double raw_total = 0.0;
    int   n_valid   = 0;

    for (int i = 0; i < M; i++) {
        float v = d_frame_ptrs[i][px];
        if (!isnan(v)) {
            vals[n_valid++] = v;
            raw_total += (double)v;
        }
    }

    /* Accumulate raw sum/count for the all-clipped fallback */
    d_rawsum[px]   += raw_total;
    d_rawcount[px] += n_valid;

    if (n_valid == 0) return;

    /* Insertion sort — M ≤ 32 so this is fast in registers */
    for (int i = 1; i < n_valid; i++) {
        float key = vals[i];
        int   j   = i - 1;
        while (j >= 0 && vals[j] > key) {
            vals[j + 1] = vals[j];
            j--;
        }
        vals[j + 1] = key;
    }

    /* Compute median */
    float median;
    if (n_valid & 1) {
        median = vals[n_valid / 2];
    } else {
        median = (vals[n_valid / 2 - 1] + vals[n_valid / 2]) / 2.0f;
    }

    /* Weight by n_valid so batches with more valid frames contribute more */
    d_combined_sum[px]   += (double)median * n_valid;
    d_combined_count[px] += n_valid;
}

/*
 * finalize_kernel — compute the final pixel value from accumulators.
 *
 * Primary: combined_sum / combined_count  (survivor-weighted mean)
 * Fallback: rawsum / n_frames             (unclipped mean, when all clipped)
 */
__global__ static void finalize_kernel(
    const double *d_combined_sum,
    const int    *d_combined_count,
    const double *d_rawsum,
    const int    *d_rawcount,
    float        *d_out,
    int           npix)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= npix) return;

    int c = d_combined_count[px];
    if (c > 0) {
        d_out[px] = (float)(d_combined_sum[px] / (double)c);
    } else {
        /* Degenerate: every value was clipped in every batch — fall back to
         * the unclipped mean of valid (non-NaN) frames.  If no frames
         * contributed valid data (all OOB), output NAN. */
        int rc = d_rawcount[px];
        d_out[px] = (rc > 0) ? (float)(d_rawsum[px] / (double)rc) : NAN;
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
    size_t npix_d   = (size_t)W * H * sizeof(double);
    size_t npix_i   = (size_t)W * H * sizeof(int);
    cudaError_t cerr;

    /* Persistent accumulator buffers (zeroed).
     * Sum accumulators use double to prevent precision loss for large stacks. */
    if ((cerr = cudaMalloc(&ctx->d_combined_sum,   npix_d)) != cudaSuccess) goto oom_cuda;
    if ((cerr = cudaMalloc(&ctx->d_combined_count, npix_i)) != cudaSuccess) goto oom_cuda;
    if ((cerr = cudaMalloc(&ctx->d_rawsum,         npix_d)) != cudaSuccess) goto oom_cuda;
    if ((cerr = cudaMalloc(&ctx->d_rawcount,       npix_i)) != cudaSuccess) goto oom_cuda;
    cudaMemset(ctx->d_combined_sum,   0, npix_d);
    cudaMemset(ctx->d_combined_count, 0, npix_i);
    cudaMemset(ctx->d_rawsum,         0, npix_d);
    cudaMemset(ctx->d_rawcount,       0, npix_i);

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
    cudaFree(ctx->d_rawcount);
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
        ctx->d_combined_sum, ctx->d_combined_count,
        ctx->d_rawsum, ctx->d_rawcount, npix);

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

DsoError integration_gpu_process_batch_median(IntegrationGpuCtx *ctx,
                                               int                M,
                                               cudaStream_t       stream)
{
    if (!ctx || M < 1 || M > ctx->batch_size) return DSO_ERR_INVALID_ARG;

    int npix = ctx->W * ctx->H;

    cudaError_t cerr = cudaMemcpyAsync(
        ctx->d_frame_ptrs,
        ctx->d_frames,
        (size_t)M * sizeof(float *),
        cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch_median: memcpy ptrs: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    dim3 block(256);
    dim3 grid((npix + 255) / 256);

    median_batch_kernel<<<grid, block, 0, stream>>>(
        ctx->d_frame_ptrs, M,
        ctx->d_combined_sum, ctx->d_combined_count,
        ctx->d_rawsum, ctx->d_rawcount, npix);

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch_median kernel: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }
    return DSO_OK;
}

DsoError integration_gpu_process_batch_aawa(IntegrationGpuCtx *ctx,
                                             int                M,
                                             cudaStream_t       stream)
{
    if (!ctx || M < 1 || M > ctx->batch_size) return DSO_ERR_INVALID_ARG;

    int npix = ctx->W * ctx->H;

    cudaError_t cerr = cudaMemcpyAsync(
        ctx->d_frame_ptrs,
        ctx->d_frames,
        (size_t)M * sizeof(float *),
        cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch_aawa: memcpy ptrs: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    dim3 block(256);
    dim3 grid((npix + 255) / 256);

    aawa_batch_kernel<<<grid, block, 0, stream>>>(
        ctx->d_frame_ptrs, M,
        ctx->d_combined_sum, ctx->d_combined_count,
        ctx->d_rawsum, ctx->d_rawcount, npix);

    cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "integration_gpu_process_batch_aawa kernel: %s\n",
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
        ctx->d_combined_sum, ctx->d_combined_count,
        ctx->d_rawsum, ctx->d_rawcount, ctx->d_out, npix);

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
    size_t npix_d = (size_t)npix * sizeof(double);
    size_t npix_i = (size_t)npix * sizeof(int);
    cudaMemset(ctx->d_combined_sum,   0, npix_d);
    cudaMemset(ctx->d_combined_count, 0, npix_i);
    cudaMemset(ctx->d_rawsum,         0, npix_d);
    cudaMemset(ctx->d_rawcount,       0, npix_i);

    return DSO_OK;
}
