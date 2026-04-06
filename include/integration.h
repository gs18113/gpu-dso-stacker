/*
 * integration.h — Frame integration API
 *
 * Combines N aligned float32 images of identical dimensions into one
 * output image.  Two methods are provided:
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
 * integrate_median — pixel-wise median over all frames.
 *
 * Per pixel, collects all non-NaN values, sorts them, and returns the
 * middle value (or average of the two middle values for even counts).
 * Naturally resistant to outliers (satellites, cosmic rays) without
 * any tuning parameters.
 *
 * Allocates out->data; caller must free with image_free().
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG / DSO_ERR_ALLOC.
 */
DsoError integrate_median(const Image **frames, int n, Image *out);

#ifdef __cplusplus
}
#endif
