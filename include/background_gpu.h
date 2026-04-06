/*
 * background_gpu.h — GPU per-frame background normalization kernel.
 *
 * Applies the affine transform  pixel = (pixel - frame_bg) * scale_ratio + ref_bg
 * in-place on a device buffer.  NaN pixels are left unchanged.
 */

#pragma once

#include "dso_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * bg_normalize_gpu — normalize a device frame in-place.
 *
 * d_data      : device pointer to float image (W×H pixels)
 * npix        : total pixel count (W * H)
 * frame_bg    : frame background (median)
 * scale_ratio : ref_scale / frame_scale (pre-computed by caller)
 * ref_bg      : reference background (median)
 * stream      : CUDA stream for async launch
 *
 * 256 threads/block, one thread per pixel.  NaN pixels unchanged.
 * Returns DSO_ERR_CUDA on kernel launch failure, DSO_OK otherwise.
 * ------------------------------------------------------------------------- */
DsoError bg_normalize_gpu(float *d_data, int npix,
                           float frame_bg, float scale_ratio, float ref_bg,
                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif
