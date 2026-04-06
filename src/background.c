/*
 * background.c — Per-frame background normalization (CPU).
 *
 * Uses stride-16 sampling for efficient median/MAD computation on large
 * images.  For a 16 MP image this processes ~1M samples — accurate enough
 * for background estimation and much faster than a full sort.
 */

#include "background.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Stride for subsampling pixels.  Every 16th pixel gives ~1M samples for
 * a 16 MP image — sufficient accuracy for background estimation. */
#define BG_SAMPLE_STRIDE 16

/* 1.4826 converts MAD to an estimate of σ for normal distributions. */
#define MAD_TO_SIGMA 1.4826f

/* -------------------------------------------------------------------------
 * float comparator for qsort
 * ------------------------------------------------------------------------- */
static int float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

/* -------------------------------------------------------------------------
 * bg_compute_stats
 * ------------------------------------------------------------------------- */
DsoError bg_compute_stats(const float *data, int npix, BgStats *out)
{
    if (!data || npix <= 0 || !out)
        return DSO_ERR_INVALID_ARG;

    /* Allocate scratch for the subsample (worst case: npix / stride + 1). */
    int max_samples = npix / BG_SAMPLE_STRIDE + 1;
    float *samples = (float *)malloc((size_t)max_samples * sizeof(float));
    if (!samples)
        return DSO_ERR_ALLOC;

    /* Collect every 16th non-NaN pixel. */
    int n = 0;
    for (int i = 0; i < npix; i += BG_SAMPLE_STRIDE) {
        float v = data[i];
        if (!isnan(v))
            samples[n++] = v;
    }

    if (n == 0) {
        /* All sampled pixels are NaN — degenerate frame. */
        out->background = 0.0f;
        out->scale      = 0.0f;
        free(samples);
        return DSO_OK;
    }

    /* Sort to find median. */
    qsort(samples, (size_t)n, sizeof(float), float_cmp);
    float median = (n % 2 == 1)
                     ? samples[n / 2]
                     : 0.5f * (samples[n / 2 - 1] + samples[n / 2]);

    /* Compute absolute deviations from median (reuse samples array). */
    for (int i = 0; i < n; i++)
        samples[i] = fabsf(samples[i] - median);

    /* Sort absolute deviations to find MAD. */
    qsort(samples, (size_t)n, sizeof(float), float_cmp);
    float mad = (n % 2 == 1)
                  ? samples[n / 2]
                  : 0.5f * (samples[n / 2 - 1] + samples[n / 2]);

    out->background = median;
    out->scale      = MAD_TO_SIGMA * mad;

    free(samples);
    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * bg_normalize_cpu
 * ------------------------------------------------------------------------- */
DsoError bg_normalize_cpu(float *data, int npix,
                           const BgStats *frame_stats,
                           const BgStats *ref_stats)
{
    if (!data || npix <= 0 || !frame_stats || !ref_stats)
        return DSO_ERR_INVALID_ARG;

    /* Guard: if frame is essentially flat, skip normalization. */
    if (frame_stats->scale < 1e-10f)
        return DSO_OK;

    /* Guard: if reference scale is near zero, skip. */
    if (ref_stats->scale < 1e-10f)
        return DSO_OK;

    float ratio  = ref_stats->scale / frame_stats->scale;
    float frm_bg = frame_stats->background;
    float ref_bg = ref_stats->background;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < npix; i++) {
        float v = data[i];
        if (!isnan(v))
            data[i] = (v - frm_bg) * ratio + ref_bg;
    }

    return DSO_OK;
}
