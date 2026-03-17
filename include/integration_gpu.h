/*
 * integration_gpu.h — GPU mini-batch kappa-sigma clipping integration.
 *
 * Overview
 * --------
 * After Lanczos alignment, N float32 frames need to be combined into a
 * single output image while rejecting cosmic rays, satellite trails, and
 * other transient artefacts.  Keeping all N frames in GPU memory
 * simultaneously is impractical for large datasets; instead this module
 * works in mini-batches of M frames.
 *
 * Mini-batch strategy
 * -------------------
 *   For each batch of M frames (all already transformed and resident in
 *   GPU memory):
 *     1. Run the kappa-sigma clipping kernel: each pixel independently
 *        collects its M values, performs iterative sigma-clipping (same
 *        algorithm as the CPU integrate_kappa_sigma), and writes:
 *          d_partial_mean[px] = clipped mean for this batch
 *          d_count[px]        = number of surviving samples (≥ 1)
 *     2. Accumulate into persistent global buffers:
 *          d_combined_sum[px]   += d_partial_mean[px] * d_count[px]
 *          d_combined_count[px] += d_count[px]
 *        A parallel unclipped accumulator d_rawsum[px] += sum(vals) is
 *        maintained for the degenerate all-clipped fallback.
 *
 *   Final output = d_combined_sum[px] / d_combined_count[px].
 *   If d_combined_count[px] == 0 (all values clipped in every batch),
 *   the pixel falls back to the unclipped mean.
 *
 * Kappa-sigma GPU kernel
 * ----------------------
 *   Block: 256 threads (1-D), Grid: ceil(W*H / 256).
 *   Each thread owns one pixel and loops over M frame values in registers
 *   (stack arrays; safe for M ≤ 32 which is the recommended range).
 *   The per-pixel clipping loop mirrors the CPU implementation:
 *     repeat up to `iterations` times:
 *       compute mean and Bessel-corrected stddev of active values
 *       reject values |v - mean| > kappa * stddev
 *       break if nothing rejected or fewer than 2 values remain
 *
 * Maximum batch size
 * ------------------
 *   INTEGRATION_GPU_MAX_BATCH (64) is the compile-time upper bound.
 *   Values beyond this require refactoring the fixed-size device arrays.
 *   Practical recommendation: M = 8–32 for good GPU occupancy.
 */

#pragma once

#include "dso_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * INTEGRATION_GPU_MAX_BATCH — compile-time maximum frames per batch.
 * The GPU kappa-sigma kernel uses stack arrays of this size.
 */
#define INTEGRATION_GPU_MAX_BATCH 64

/*
 * IntegrationGpuCtx — GPU integration context.
 *
 * Holds all persistent device buffers.  The pipeline accesses d_frames[],
 * d_xmap, and d_ymap directly to write transformed frames and Lanczos maps.
 *
 * Not thread-safe; use one context per pipeline instance.
 */
typedef struct IntegrationGpuCtx {
    float  *d_combined_sum;                      /* W*H floats — survivor-weighted sum  */
    float  *d_rawsum;                            /* W*H floats — unclipped sum          */
    int    *d_combined_count;                    /* W*H ints   — survivor count         */
    float  *d_frames[INTEGRATION_GPU_MAX_BATCH]; /* batch device frame buffers          */
    float  *d_xmap;                              /* W*H floats — Lanczos x-coord map    */
    float  *d_ymap;                              /* W*H floats — Lanczos y-coord map    */
    float **d_frame_ptrs;                        /* device array of M frame pointers    */
    float  *d_out;                               /* W*H floats — finalize staging buf   */
    int     W, H, batch_size;
} IntegrationGpuCtx;

/*
 * integration_gpu_init — allocate all GPU resources needed for integration.
 *
 * W, H        : output image dimensions (pixels)
 * batch_size  : maximum M per batch; must be 1 ≤ batch_size ≤
 *               INTEGRATION_GPU_MAX_BATCH
 * ctx_out     : receives a heap-allocated context pointer on success;
 *               must be freed with integration_gpu_cleanup()
 *
 * Allocates:
 *   - batch_size device frame buffers (d_frames[0..batch_size-1])
 *   - d_xmap, d_ymap for Lanczos coordinate maps (reused across frames)
 *   - d_combined_sum, d_combined_count, d_rawsum (global accumulators,
 *     zeroed at init and after each integration_gpu_finalize call)
 *   - d_partial_mean, d_count (per-batch temporaries)
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_ALLOC / DSO_ERR_INVALID_ARG.
 */
DsoError integration_gpu_init(int                  W,
                               int                  H,
                               int                  batch_size,
                               IntegrationGpuCtx  **ctx_out);

/*
 * integration_gpu_cleanup — free all GPU and CPU resources held by ctx.
 *
 * Safe to call with ctx == NULL.
 */
void integration_gpu_cleanup(IntegrationGpuCtx *ctx);

/*
 * integration_gpu_process_batch — run kappa-sigma on M pre-transformed frames.
 *
 * All M frames must already reside in ctx->d_frames[0..M-1] (placed there
 * by the pipeline after lanczos_transform_gpu_d2d).
 *
 * M          : number of frames in this batch; must be 1 ≤ M ≤ batch_size
 * kappa      : rejection threshold in units of sample stddev (e.g. 3.0)
 * iterations : maximum sigma-clipping passes per pixel (e.g. 3)
 * stream     : CUDA stream; the function does NOT synchronise — caller
 *              must synchronise before reading the accumulators
 *
 * Side effects:
 *   Updates ctx->d_combined_sum, ctx->d_combined_count, ctx->d_rawsum
 *   via the kappa-sigma result and the survivor-count-weighted accumulation.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError integration_gpu_process_batch(IntegrationGpuCtx *ctx,
                                        int                M,
                                        float              kappa,
                                        int                iterations,
                                        cudaStream_t       stream);

/*
 * integration_gpu_process_batch_mean — accumulate M frames without clipping.
 *
 * Used when the integration method is "mean".  Each frame is added to the
 * unclipped rawsum accumulator and the combined_count is incremented by M.
 *
 * M      : number of frames in this batch
 * stream : CUDA stream; no implicit synchronisation
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError integration_gpu_process_batch_mean(IntegrationGpuCtx *ctx,
                                             int                M,
                                             cudaStream_t       stream);

/*
 * integration_gpu_finalize — download the final result image.
 *
 * Computes the final pixel value from the accumulated sums:
 *   out[px] = combined_sum[px] / combined_count[px]   if count > 0
 *           = rawsum[px] / n_frames                    otherwise (fallback)
 *
 * n_frames : total number of frames processed (for the unclipped fallback)
 * out      : output Image; out->data must be pre-allocated (W*H floats)
 * stream   : CUDA stream; the function synchronises before returning so
 *            out->data is immediately usable on the host
 *
 * After this call the accumulators are zeroed so the context can be reused
 * for a subsequent integration run.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError integration_gpu_finalize(IntegrationGpuCtx *ctx,
                                   int                n_frames,
                                   Image             *out,
                                   cudaStream_t       stream);

#ifdef __cplusplus
}
#endif
