/*
 * lanczos_gpu.h — GPU Lanczos-3 image transformation API
 *
 * Accelerates the homographic warp with CUDA and NPP+:
 *   1. A CUDA kernel computes the inverse-homography coordinate maps
 *      (xmap, ymap) for every destination pixel.
 *   2. nppiRemap_32f_C1R_Ctx samples the source at those coordinates
 *      with NPPI_INTER_LANCZOS (Lanczos-3) interpolation.
 *
 * This two-step approach is necessary because nppiWarpPerspective only
 * supports NN, bilinear, and bicubic interpolation modes.
 *
 * Usage:
 *   lanczos_gpu_init(stream);          // once at startup
 *   lanczos_transform_gpu(&src, &dst, &H);  // per frame
 *   lanczos_gpu_cleanup();             // once at shutdown
 */

#pragma once

#include "dso_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * lanczos_gpu_init — initialise NPP stream context from device properties.
 *
 * Must be called before the first lanczos_transform_gpu call.
 * Pass stream = 0 to use the CUDA default stream.
 *
 * Returns DSO_OK or DSO_ERR_CUDA.
 */
DsoError lanczos_gpu_init(cudaStream_t stream);

/*
 * lanczos_gpu_cleanup — release any GPU-side resources.
 *
 * Currently a no-op (per-frame memory is freed inside lanczos_transform_gpu);
 * provided for API symmetry and future extensibility.
 */
void lanczos_gpu_cleanup(void);

/*
 * lanczos_transform_gpu — warp src into dst using Lanczos-3 on the GPU.
 *
 * H is the BACKWARD homography (ref → src) and is used directly without
 * inversion.  dst->width, dst->height, and dst->data must be pre-allocated
 * by the caller.  Destination pixels that map outside the source are written
 * as 0.  Allocates / frees device memory internally; synchronises before
 * returning.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError lanczos_transform_gpu(const Image *src, Image *dst, const Homography *H);

/*
 * lanczos_transform_gpu_d2d — device-to-device Lanczos-3 warp for the
 * overlapped pipeline.
 *
 * Unlike lanczos_transform_gpu, this variant:
 *   - Accepts pre-allocated device pointers (d_src, d_dst, d_xmap, d_ymap).
 *   - Performs no H2D/D2H transfers.
 *   - Executes asynchronously on `stream`; caller must synchronise.
 *   - Derives NppStreamContext from the module-global g_nppCtx (populated
 *     by lanczos_gpu_init) but overrides the stream field to use `stream`.
 *
 * Caller responsibilities:
 *   - d_src  : SW×SH float32 on device (source image, row-major)
 *   - d_dst  : DW×DH float32 on device (written by this call)
 *   - d_xmap : DW×DH float32 temp buffer (written internally)
 *   - d_ymap : DW×DH float32 temp buffer (written internally)
 *   - d_dst must be zeroed before calling (out-of-bounds pixels are unwritten)
 *
 * H is the BACKWARD homography (ref → src); used directly without inversion.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError lanczos_transform_gpu_d2d(
    const float       *d_src,
    float             *d_dst,
    float             *d_xmap,
    float             *d_ymap,
    int                SW, int SH,
    int                DW, int DH,
    const Homography  *H,
    cudaStream_t       stream);

#ifdef __cplusplus
}
#endif
