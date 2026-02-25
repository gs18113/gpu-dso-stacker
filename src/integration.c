/*
 * integration.c — Frame integration (mean and kappa-sigma clipping)
 *
 * Combines N float32 images of identical dimensions into one.
 *
 * integrate_mean:
 *   Simple pixel-wise arithmetic mean over all frames.
 *
 * integrate_kappa_sigma:
 *   Per-pixel iterative sigma-clipping:
 *     1. Collect all N values and initialise a boolean mask (all active).
 *     2. Repeat up to `iterations` times:
 *          a. Compute mean and sample stddev (Bessel correction, ÷ (n−1))
 *             of currently active values.
 *          b. Reject values with |val − mean| > kappa × stddev.
 *          c. Break early if nothing was rejected or fewer than 2 remain.
 *     3. Output = mean of surviving values.
 *        If all values were rejected (degenerate), fall back to unclipped mean.
 */

#include "integration.h"
#include "fits_io.h"   /* image_free */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

DsoError integrate_mean(const Image **frames, int n, Image *out)
{
    if (!frames || n <= 0 || !out) return DSO_ERR_INVALID_ARG;

    int W = frames[0]->width;
    int H = frames[0]->height;

    /* Validate that every frame has the same dimensions */
    for (int i = 1; i < n; i++) {
        if (frames[i]->width != W || frames[i]->height != H) {
            fprintf(stderr, "integrate_mean: frame %d size mismatch (%dx%d vs %dx%d)\n",
                    i, frames[i]->width, frames[i]->height, W, H);
            return DSO_ERR_INVALID_ARG;
        }
    }

    out->data = (float *)calloc((size_t)W * H, sizeof(float));
    if (!out->data) return DSO_ERR_ALLOC;
    out->width  = W;
    out->height = H;

    long  npix  = (long)W * H;
    float inv_n = 1.f / (float)n;

    for (int i = 0; i < n; i++) {
        const float *src = frames[i]->data;
        float       *dst = out->data;
        for (long p = 0; p < npix; p++) {
            dst[p] += src[p] * inv_n;
        }
    }

    return DSO_OK;
}

DsoError integrate_kappa_sigma(const Image **frames, int n, Image *out,
                                float kappa, int iterations)
{
    if (!frames || n <= 0 || !out || kappa <= 0.f || iterations < 1)
        return DSO_ERR_INVALID_ARG;

    int W = frames[0]->width;
    int H = frames[0]->height;

    for (int i = 1; i < n; i++) {
        if (frames[i]->width != W || frames[i]->height != H) {
            fprintf(stderr,
                    "integrate_kappa_sigma: frame %d size mismatch (%dx%d vs %dx%d)\n",
                    i, frames[i]->width, frames[i]->height, W, H);
            return DSO_ERR_INVALID_ARG;
        }
    }

    out->data = (float *)malloc((size_t)W * H * sizeof(float));
    if (!out->data) return DSO_ERR_ALLOC;
    out->width  = W;
    out->height = H;

    /* Allocate per-pixel work buffers once outside the pixel loop.
     * Using heap (not VLA/stack) to be safe for large frame counts. */
    float *vals = (float *)malloc((size_t)n * sizeof(float));
    int   *mask = (int   *)malloc((size_t)n * sizeof(int));
    if (!vals || !mask) {
        free(vals);
        free(mask);
        image_free(out);
        return DSO_ERR_ALLOC;
    }

    long npix = (long)W * H;

    for (long p = 0; p < npix; p++) {
        /* Gather values from all frames for this pixel */
        for (int i = 0; i < n; i++) {
            vals[i] = frames[i]->data[p];
            mask[i] = 1;
        }

        int n_active = n;

        for (int iter = 0; iter < iterations; iter++) {
            if (n_active < 2) break;   /* can't compute stddev with < 2 samples */

            /* Two-pass mean */
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                if (mask[i]) sum += vals[i];
            }
            double mean = sum / n_active;

            /* Sample variance with Bessel correction (÷ (n_active − 1)) */
            double sq_sum = 0.0;
            for (int i = 0; i < n; i++) {
                if (mask[i]) {
                    double d = vals[i] - mean;
                    sq_sum += d * d;
                }
            }
            double stddev = sqrt(sq_sum / (n_active - 1));

            /* Reject any value more than kappa·σ from the mean */
            int n_rejected = 0;
            double threshold = kappa * stddev;
            for (int i = 0; i < n; i++) {
                if (mask[i] && fabs(vals[i] - mean) > threshold) {
                    mask[i] = 0;
                    n_rejected++;
                }
            }
            n_active -= n_rejected;

            if (n_rejected == 0) break;   /* converged */
            if (n_active < 2)    break;   /* too few left for next iteration */
        }

        /* Output = mean of surviving values */
        if (n_active > 0) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                if (mask[i]) sum += vals[i];
            }
            out->data[p] = (float)(sum / n_active);
        } else {
            /* Degenerate: all pixels clipped — fall back to plain mean */
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += vals[i];
            out->data[p] = (float)(sum / n);
        }
    }

    free(vals);
    free(mask);
    return DSO_OK;
}
