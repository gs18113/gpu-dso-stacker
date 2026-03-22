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
#include <omp.h>

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

    /* Pixel-outer loop: each pixel is independent → safe to parallelise. */
#pragma omp parallel for schedule(static)
    for (long p = 0; p < npix; p++) {
        float sum = 0.f;
        for (int i = 0; i < n; i++)
            sum += frames[i]->data[p];
        out->data[p] = sum * inv_n;
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

    long npix = (long)W * H;

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif

    /* Per-thread workspaces to avoid VLAs on stack */
    float *all_vals = (float *)malloc((size_t)max_threads * n * sizeof(float));
    int   *all_actv = (int   *)malloc((size_t)max_threads * n * sizeof(int));

    if (!all_vals || !all_actv) {
        free(all_vals); free(all_actv);
        free(out->data); out->data = NULL;
        return DSO_ERR_ALLOC;
    }

    /*
     * Each pixel is fully independent. Each OpenMP thread uses its own
     * dedicated slice of the all_vals/all_actv workspace.
     */
#pragma omp parallel for schedule(dynamic, 64)
    for (long p = 0; p < npix; p++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *vals = &all_vals[tid * n];
        int   *actv = &all_actv[tid * n];

        for (int i = 0; i < n; i++) {
            vals[i] = frames[i]->data[p];
            actv[i] = 1;
        }

        int n_active = n;

        for (int iter = 0; iter < iterations; iter++) {
            if (n_active < 2) break;

            double sum = 0.0;
            for (int i = 0; i < n; i++)
                if (actv[i]) sum += vals[i];
            double mean = sum / n_active;

            double sq_sum = 0.0;
            for (int i = 0; i < n; i++) {
                if (actv[i]) {
                    double d = vals[i] - mean;
                    sq_sum += d * d;
                }
            }
            double stddev    = sqrt(sq_sum / (n_active - 1));
            double threshold = kappa * stddev;

            int n_rejected = 0;
            for (int i = 0; i < n; i++) {
                if (actv[i] && fabs(vals[i] - mean) > threshold) {
                    actv[i] = 0;
                    n_rejected++;
                }
            }
            n_active -= n_rejected;
            if (n_rejected == 0) break;
            if (n_active < 2)    break;
        }

        if (n_active > 0) {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
                if (actv[i]) sum += vals[i];
            out->data[p] = (float)(sum / n_active);
        } else {
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += vals[i];
            out->data[p] = (float)(sum / n);
        }
    }

    free(all_vals);
    free(all_actv);
    return DSO_OK;
}
