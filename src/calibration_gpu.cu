/*
 * calibration_gpu.cu — GPU calibration frame application.
 *
 * Uploads dark and flat masters to device memory once per pipeline run.
 * A single CUDA kernel applies both corrections in-place to each raw
 * (pre-debayer) frame buffer before it is passed to the debayer kernel.
 *
 * Kernel
 * ------
 *   calib_apply_kernel<<<grid, 256, 0, stream>>>(
 *       d_frame, d_dark, d_flat, N, has_dark, has_flat)
 *
 * For pixel i:
 *   v = d_frame[i]
 *   if has_dark: v -= d_dark[i]
 *   if has_flat: f = d_flat[i]; v = (f < 1e-6) ? 0 : v / f
 *   d_frame[i] = v
 *
 * The has_dark / has_flat flags are integers; nvcc will hoist the
 * invariant branches out of the inner loop via compile-time constant
 * propagation when the flags are uniform across the warp (which they are —
 * they never change within a kernel launch).
 */

#include "calibration_gpu.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------
 * CUDA error helpers (local to this file)
 * ------------------------------------------------------------------------- */
#define CU_CHECK(call, label)                                            \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            fprintf(stderr,                                              \
                    "calibration_gpu CUDA error %s:%d — %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));         \
            err = DSO_ERR_CUDA; goto label;                             \
        }                                                                \
    } while (0)

/* -------------------------------------------------------------------------
 * Kernel: per-pixel dark subtract + flat divide
 * ------------------------------------------------------------------------- */
__global__ static void calib_apply_kernel(float       *frame,
                                           const float *dark,
                                           const float *flat,
                                           int          N,
                                           int          has_dark,
                                           int          has_flat)
{
    int i = (int)(blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N) return;

    float v = frame[i];

    if (has_dark)
        v -= dark[i];

    if (has_flat) {
        float f = flat[i];
        v = (f < 1e-6f) ? 0.0f : v / f;
    }

    frame[i] = v;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

DsoError calib_gpu_init(const CalibFrames *calib, CalibGpuCtx **ctx_out)
{
    if (!calib || !ctx_out) return DSO_ERR_INVALID_ARG;

    DsoError      err = DSO_OK;
    CalibGpuCtx  *ctx = NULL;

    ctx = (CalibGpuCtx *)calloc(1, sizeof(CalibGpuCtx));
    if (!ctx) return DSO_ERR_ALLOC;

    /* Determine device dimensions from whichever master is available */
    if (calib->has_dark) {
        ctx->W = calib->dark.width;
        ctx->H = calib->dark.height;
    } else if (calib->has_flat) {
        ctx->W = calib->flat.width;
        ctx->H = calib->flat.height;
    } else {
        /* Nothing to upload; return a valid but empty context */
        *ctx_out = ctx;
        return DSO_OK;
    }

    size_t nbytes = (size_t)ctx->W * ctx->H * sizeof(float);

    if (calib->has_dark) {
        CU_CHECK(cudaMalloc(&ctx->d_dark, nbytes), cleanup);
        CU_CHECK(cudaMemcpy(ctx->d_dark, calib->dark.data, nbytes,
                            cudaMemcpyHostToDevice), cleanup);
    }

    if (calib->has_flat) {
        CU_CHECK(cudaMalloc(&ctx->d_flat, nbytes), cleanup);
        CU_CHECK(cudaMemcpy(ctx->d_flat, calib->flat.data, nbytes,
                            cudaMemcpyHostToDevice), cleanup);
    }

    *ctx_out = ctx;
    return DSO_OK;

cleanup:
    cudaFree(ctx->d_dark);
    cudaFree(ctx->d_flat);
    free(ctx);
    return err;
}

DsoError calib_gpu_apply_d2d(float *d_frame, int W, int H,
                               const CalibGpuCtx *ctx, cudaStream_t stream)
{
    if (!ctx) return DSO_OK;
    if (!ctx->d_dark && !ctx->d_flat) return DSO_OK;

    if (W != ctx->W || H != ctx->H) {
        fprintf(stderr,
                "calib_gpu_apply_d2d: frame %d×%d != master %d×%d\n",
                W, H, ctx->W, ctx->H);
        return DSO_ERR_INVALID_ARG;
    }

    int  N    = W * H;
    dim3 block(256);
    dim3 grid((unsigned int)((N + 255) / 256));

    calib_apply_kernel<<<grid, block, 0, stream>>>(
        d_frame,
        ctx->d_dark,
        ctx->d_flat,
        N,
        ctx->d_dark ? 1 : 0,
        ctx->d_flat ? 1 : 0);

    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "calib_apply_kernel launch error: %s\n",
                cudaGetErrorString(ce));
        return DSO_ERR_CUDA;
    }

    return DSO_OK;
}

void calib_gpu_cleanup(CalibGpuCtx *ctx)
{
    if (!ctx) return;
    cudaFree(ctx->d_dark);
    cudaFree(ctx->d_flat);
    free(ctx);
}
