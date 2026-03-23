/*
 * calibration_gpu.h — GPU-side application of calibration master frames.
 *
 * Uploads dark and flat masters to the GPU once per pipeline run and
 * provides a single D2D kernel call to calibrate each raw frame in-place
 * before debayering.
 *
 * Operation
 * ---------
 * The CUDA kernel applies, per pixel:
 *   1. Subtract dark master (if has_dark)
 *   2. Divide by flat master (if has_flat); pixels where flat < 1e-6 → 0
 *
 * Usage in pipeline.cu
 * --------------------
 *   CalibGpuCtx *calib_ctx = NULL;
 *   if (config->calib)
 *       calib_gpu_init(config->calib, &calib_ctx);
 *
 *   // After each H2D transfer, before debayer:
 *   calib_gpu_apply_d2d(d_raw, W, H, calib_ctx, stream_compute);
 *
 *   calib_gpu_cleanup(calib_ctx);
 */

#pragma once

#include "calibration.h"      /* CalibFrames */
#include "dso_types.h"        /* DsoError */
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * CalibGpuCtx — device-side master frame buffers.
 * ------------------------------------------------------------------------- */
typedef struct {
    float *d_dark;   /* device dark master  (NULL = not used) */
    float *d_flat;   /* device flat master  (NULL = not used) */
    int    W;
    int    H;
} CalibGpuCtx;

/*
 * calib_gpu_init — allocate device buffers and upload master frames.
 *
 * Creates a CalibGpuCtx on the heap and populates it from the host-side
 * CalibFrames.  Passes calib->dark / calib->flat dimensions; both must
 * match the light-frame dimensions that will be passed to calib_gpu_apply_d2d.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_ALLOC on failure.
 */
DsoError calib_gpu_init(const CalibFrames *calib, CalibGpuCtx **ctx_out);

/*
 * calib_gpu_apply_d2d — apply calibration in-place to a device buffer.
 *
 * d_frame must point to W*H floats on the device.
 * stream may be 0 (default stream).
 * W and H must match ctx->W and ctx->H.
 */
DsoError calib_gpu_apply_d2d(float *d_frame, int W, int H,
                               const CalibGpuCtx *ctx, cudaStream_t stream);

/*
 * calib_gpu_cleanup — free device buffers and the context itself.
 * Safe to call with NULL.
 */
void calib_gpu_cleanup(CalibGpuCtx *ctx);

#ifdef __cplusplus
}
#endif
