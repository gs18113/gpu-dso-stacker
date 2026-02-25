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
 * H is the *forward* homography (source → reference).  Internally inverted.
 *
 * dst->width, dst->height, and dst->data must be pre-allocated by the caller.
 * Destination pixels that map outside the source are written as 0.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError lanczos_transform_gpu(const Image *src, Image *dst, const Homography *H);

#ifdef __cplusplus
}
#endif
