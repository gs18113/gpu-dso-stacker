/*
 * lanczos_cpu.h — CPU Lanczos-3 image transformation API
 *
 * Applies a homographic (perspective) warp from a source image to a
 * destination image using backward mapping with a 6×6-tap Lanczos-3
 * interpolation kernel.  Runs on the host CPU; no GPU required.
 *
 * Suitable for testing correctness or for systems without a CUDA GPU.
 * For large images the GPU path (lanczos_gpu.h) is significantly faster.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * lanczos_transform_cpu — warp src into dst using Lanczos-3 interpolation.
 *
 * H is the *backward* homography (reference pixel coords → source pixel coords).
 * It is used directly for pixel sampling — do not invert before passing.
 *
 * dst->width, dst->height, and dst->data must be set by the caller before
 * the call (typically matching the reference frame dimensions).
 *
 * Boundary handling: taps outside the source image are skipped and the weight
 * sum is renormalised.  Pixels with no valid taps are written as 0.
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG (NULL pointers or singular H).
 */
DsoError lanczos_transform_cpu(const Image *src, Image *dst, const Homography *H);

#ifdef __cplusplus
}
#endif
