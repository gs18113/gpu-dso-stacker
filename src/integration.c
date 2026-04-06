/*
 * integration.c — Frame integration (mean, kappa-sigma clipping, AAWA)
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
 *
 * integrate_aawa:
 *   Auto Adaptive Weighted Average (Stetson 1989):
 *     1. Compute initial mean μ and sample stddev σ.
 *     2. Iterate (max 10): compute Stetson weights w[i] = 1/(1+(|r|/α)²)
 *        where r = (v-μ)/σ and α=2.0, then update μ and σ as weighted stats.
 *     3. Converge when |Δμ|/max(|μ|,1e-10) < 1e-6.
 *
 * NaN sentinel:
 *   Warped frames use NAN to mark out-of-bounds pixels.  All integration
 *   routines skip NaN values so that OOB regions do not contaminate the
 *   stacked result.  If every frame is NaN at a pixel, the output is NAN.
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
    long  p;

    /* Pixel-outer loop: each pixel is independent → safe to parallelise.
     * NaN-valued pixels (OOB sentinel from Lanczos warp) are skipped. */
#pragma omp parallel for schedule(static)
    for (p = 0; p < npix; p++) {
        double sum = 0.0;
        int   valid = 0;
        for (int i = 0; i < n; i++) {
            float v = frames[i]->data[p];
            if (!isnan(v)) { sum += (double)v; valid++; }
        }
        out->data[p] = (valid > 0) ? (float)(sum / (double)valid) : NAN;
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
    long p;
#pragma omp parallel for schedule(dynamic, 64)
    for (p = 0; p < npix; p++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *vals = &all_vals[tid * n];
        int   *actv = &all_actv[tid * n];

        int n_valid = 0;
        for (int i = 0; i < n; i++) {
            float v = frames[i]->data[p];
            if (isnan(v)) {
                vals[i] = 0.f;
                actv[i] = 0;
            } else {
                vals[i] = v;
                actv[i] = 1;
                n_valid++;
            }
        }

        /* If no frames have valid data at this pixel, output NAN (OOB) */
        if (n_valid == 0) { out->data[p] = NAN; continue; }

        int n_active = n_valid;

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
            /* All-clipped fallback: unclipped mean of valid (non-NaN) values */
            double sum = 0.0;
            for (int i = 0; i < n; i++)
                if (!isnan(frames[i]->data[p])) sum += vals[i];
            out->data[p] = (float)(sum / n_valid);
        }
    }

    free(all_vals);
    free(all_actv);
    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * AAWA — Auto Adaptive Weighted Average (Stetson 1989)
 * ------------------------------------------------------------------------- */

#define AAWA_ALPHA    2.0
#define AAWA_MAX_ITER 10

DsoError integrate_aawa(const Image **frames, int n, Image *out)
{
    if (!frames || n <= 0 || !out) return DSO_ERR_INVALID_ARG;

    int W = frames[0]->width;
    int H = frames[0]->height;

    for (int i = 1; i < n; i++) {
        if (frames[i]->width != W || frames[i]->height != H) {
            fprintf(stderr,
                    "integrate_aawa: frame %d size mismatch (%dx%d vs %dx%d)\n",
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

    /* Per-thread workspaces: values + weights (double for precision) */
    float  *all_vals = (float  *)malloc((size_t)max_threads * n * sizeof(float));
    double *all_wts  = (double *)malloc((size_t)max_threads * n * sizeof(double));

    if (!all_vals || !all_wts) {
        free(all_vals); free(all_wts);
        free(out->data); out->data = NULL;
        return DSO_ERR_ALLOC;
    }

    long p;
#pragma omp parallel for schedule(dynamic, 64)
    for (p = 0; p < npix; p++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float  *vals = &all_vals[(size_t)tid * n];
        double *wts  = &all_wts[(size_t)tid * n];

        /* Collect non-NaN values */
        int n_valid = 0;
        for (int i = 0; i < n; i++) {
            float v = frames[i]->data[p];
            if (!isnan(v)) vals[n_valid++] = v;
        }

        if (n_valid == 0) { out->data[p] = NAN; continue; }
        if (n_valid == 1) { out->data[p] = vals[0]; continue; }

        /* Initial mean and sample stddev (Bessel-corrected) */
        double mu = 0.0;
        for (int i = 0; i < n_valid; i++) mu += (double)vals[i];
        mu /= n_valid;

        double sq = 0.0;
        for (int i = 0; i < n_valid; i++) {
            double d = (double)vals[i] - mu;
            sq += d * d;
        }
        double sigma = sqrt(sq / (n_valid - 1));

        /* If σ ≈ 0, all values are identical — output mean directly */
        if (sigma < 1e-12) { out->data[p] = (float)mu; continue; }

        /* Iterative Stetson weighted average */
        for (int iter = 0; iter < AAWA_MAX_ITER; iter++) {
            double sum_w  = 0.0;
            double sum_wv = 0.0;

            for (int i = 0; i < n_valid; i++) {
                double r = ((double)vals[i] - mu) / sigma;
                double ra = fabs(r) / AAWA_ALPHA;
                wts[i] = 1.0 / (1.0 + ra * ra);
                sum_w  += wts[i];
                sum_wv += wts[i] * (double)vals[i];
            }

            /* Guard against degenerate sum of weights */
            if (sum_w < 1e-12) break;

            double mu_new = sum_wv / sum_w;

            /* Weighted standard deviation */
            double sum_wsq = 0.0;
            for (int i = 0; i < n_valid; i++) {
                double d = (double)vals[i] - mu_new;
                sum_wsq += wts[i] * d * d;
            }
            double sigma_new = sqrt(sum_wsq / sum_w);

            /* Convergence check */
            double denom = fabs(mu) > 1e-10 ? fabs(mu) : 1e-10;
            if (fabs(mu_new - mu) / denom < 1e-6) {
                mu = mu_new;
                break;
            }

            mu    = mu_new;
            sigma = (sigma_new > 1e-12) ? sigma_new : 1e-12;
        }

        out->data[p] = (float)mu;
    }

    free(all_vals);
    free(all_wts);
    return DSO_OK;
}
