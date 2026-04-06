/*
 * integration.h — Frame integration API
 *
 * Combines N aligned float32 images of identical dimensions into one
 * output image.  Three methods are provided:
 *
 * integrate_mean:
 *   Simple pixel-wise arithmetic mean.  Fast; no outlier rejection.
 *
 * integrate_kappa_sigma:
 *   Iterative sigma-clipping per pixel.  On each iteration, values further
 *   than kappa·σ from the current mean are rejected and the statistics are
 *   recomputed on the survivors.  This suppresses cosmic rays, hot pixels,
 *   and satellite trails that appear in only a fraction of the frames.
 *
 * integrate_aawa:
 *   Auto Adaptive Weighted Average (Stetson 1989).  Iteratively computes a
 *   robust weighted mean where outlier frames are down-weighted rather than
 *   hard-rejected.  Uses the Stetson weight function w = 1/(1+(|r|/α)²)
 *   with α = 2.0.  Best general-purpose method for most DSO datasets.
 *
 * All frames must have the same width and height; a mismatch returns
 * DSO_ERR_INVALID_ARG.  The caller must free out->data via image_free()
 * after use.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * integrate_mean — pixel-wise arithmetic mean over all frames.
 *
 * Allocates out->data; caller must free with image_free().
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG / DSO_ERR_ALLOC.
 */
DsoError integrate_mean(const Image **frames, int n, Image *out);

/*
 * integrate_kappa_sigma — sigma-clipping integration.
 *
 * kappa      : rejection threshold in units of sample stddev (e.g. 3.0)
 * iterations : maximum number of clipping passes per pixel (e.g. 3)
 *
 * Allocates out->data; caller must free with image_free().
 * If all values for a pixel are clipped (degenerate case), the unclipped
 * mean is used as a fallback.
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG / DSO_ERR_ALLOC.
 */
DsoError integrate_kappa_sigma(const Image **frames, int n, Image *out,
                                float kappa, int iterations);

/*
 * integrate_aawa — Auto Adaptive Weighted Average (Stetson 1989).
 *
 * Iteratively computes a robust weighted mean per pixel.  On each iteration
 * the normalized residual r[i] = (v[i] - μ) / σ is used to compute a
 * Stetson weight w[i] = 1 / (1 + (|r[i]| / 2.0)²), and the weighted mean
 * and standard deviation are updated.  Converges when Δμ/|μ| < 1e-6 or
 * after 10 iterations.
 *
 * Allocates out->data; caller must free with image_free().
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG / DSO_ERR_ALLOC.
 */
DsoError integrate_aawa(const Image **frames, int n, Image *out);

#ifdef __cplusplus
}
#endif
