/*
 * debayer_gpu.h — GPU Bayer-mosaic demosaicing (VNG) → float32 luminance.
 *
 * Raw astronomical images from colour cameras arrive as a single-channel
 * Bayer mosaic: each pixel records only one colour channel (R, G, or B)
 * according to the CFA (Colour Filter Array) layout.  This module converts
 * such a mosaic to a single-channel float32 luminance image suitable for
 * star detection and Lanczos alignment.
 *
 * Algorithm: Variable Number of Gradients (VNG)
 *   For each destination pixel the surrounding 5×5 neighbourhood is examined.
 *   Eight directional gradients (N, NE, E, SE, S, SW, W, NW) are computed;
 *   only directions with gradient ≤ mean + threshold are considered "smooth"
 *   and used to estimate the missing colour channels.  After reconstructing
 *   R, G, B at every pixel, luminance is derived as:
 *     L = 0.2126·R + 0.7152·G + 0.0722·B   (ITU-R BT.709, linear light)
 *
 * Monochrome fast path:
 *   When pattern == BAYER_NONE the output is a straight copy of the input;
 *   no interpolation is performed.
 *
 * Memory convention:
 *   Both input and output are row-major float32, pixel (x,y) at index y*W+x.
 *   Input pixel values are expected in ADU (linear detector counts).
 */

#pragma once

#include "dso_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * debayer_gpu — host-to-host VNG debayering convenience wrapper.
 *
 * src:     host-side float32 Bayer mosaic image (row-major)
 * dst:     host-side float32 luminance output; dst->data, dst->width, and
 *          dst->height must be pre-allocated and set by the caller to the
 *          same dimensions as src.
 * pattern: CFA layout; BAYER_NONE copies src unchanged.
 * stream:  CUDA stream (0 = default stream); the function synchronises
 *          before returning so the result is immediately available on the
 *          host.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError debayer_gpu(const Image  *src,
                     Image        *dst,
                     BayerPattern  pattern,
                     cudaStream_t  stream);

/*
 * debayer_gpu_d2d — device-to-device VNG debayering for the pipeline.
 *
 * All buffers are already on the device; no H2D / D2H transfers are done.
 * Executes on `stream` asynchronously — caller must synchronise when needed.
 *
 * d_src : device float32 input  (W × H floats, Bayer mosaic)
 * d_dst : device float32 output (W × H floats, luminance); must be pre-
 *         allocated by the caller.
 * W, H  : image dimensions in pixels.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError debayer_gpu_d2d(const float  *d_src,
                          float        *d_dst,
                          int           W, int H,
                          BayerPattern  pattern,
                          cudaStream_t  stream);

/*
 * debayer_gpu_rgb_d2d — device-to-device VNG debayering into three planes.
 *
 * Same VNG algorithm as debayer_gpu_d2d but writes reconstructed R, G, B
 * values to three separate device buffers instead of collapsing to luminance.
 * All output buffers must be pre-allocated by the caller (each W × H floats).
 * Executes on `stream` asynchronously.
 *
 * BAYER_NONE fast path: d_src is copied to all three output buffers.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 */
DsoError debayer_gpu_rgb_d2d(const float  *d_src,
                               float        *d_r,
                               float        *d_g,
                               float        *d_b,
                               int           W, int H,
                               BayerPattern  pattern,
                               cudaStream_t  stream);

#ifdef __cplusplus
}
#endif
