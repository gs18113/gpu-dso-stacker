/*
 * debayer_cpu.h — CPU VNG (Variable Number of Gradients) Bayer demosaicing.
 *
 * Mirrors the algorithm implemented in debayer_gpu.cu but runs entirely on
 * the CPU with OpenMP parallelism across pixels.
 *
 * Algorithm overview (same as GPU path):
 *   For each pixel (x,y):
 *     1. Identify the CFA channel (R/G/B) from the Bayer pattern and position.
 *     2. Compute 8 directional gradients spanning ≥2 pixels (to cross one full
 *        2×2 CFA period), ensuring same-channel comparisons.
 *     3. Select smooth directions: gradient ≤ τ = mean(gradients) + min(gradients).
 *     4. Estimate the two missing colour channels from selected-direction neighbours.
 *     5. Emit luminance: L = 0.2126·R + 0.7152·G + 0.0722·B (ITU-R BT.709).
 *
 * Boundary handling:
 *   Pixels outside the image boundary are treated as zero (same as the GPU
 *   zero-padding strategy). Edge pixels may have slightly reduced colour
 *   accuracy, which is acceptable for star-detection use.
 *
 * Monochrome fast path:
 *   When pattern == BAYER_NONE the function performs a plain memcpy.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * debayer_cpu — VNG demosaic src → dst on the CPU.
 *
 * src     : raw Bayer mosaic image, W × H float32, row-major
 * dst     : output luminance image, W × H float32, pre-allocated by caller
 * W, H    : image dimensions (must be > 0)
 * pattern : CFA layout; BAYER_NONE triggers a fast memcpy path
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG.
 */
DsoError debayer_cpu(const float *src, float *dst,
                     int W, int H, BayerPattern pattern);

#ifdef __cplusplus
}
#endif
