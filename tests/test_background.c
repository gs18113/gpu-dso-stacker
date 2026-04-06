/*
 * test_background.c — Unit tests for per-frame background normalization.
 *
 * Tests cover:
 *   - bg_compute_stats: constant image, known median/MAD, NaN handling,
 *     degenerate (all-NaN) image
 *   - bg_normalize_cpu: background shift, scale adjustment, NaN preservation,
 *     degenerate frame (flat / zero scale) skip
 */

#include "test_framework.h"
#include "background.h"

#include <stdlib.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * bg_compute_stats tests
 * ------------------------------------------------------------------------- */

/* Constant image → background = constant, scale ≈ 0 */
static int test_bg_stats_constant(void)
{
    int npix = 1024;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    for (int i = 0; i < npix; i++) data[i] = 42.0f;

    BgStats stats;
    ASSERT_OK(bg_compute_stats(data, npix, &stats));
    ASSERT_NEAR(stats.background, 42.0f, 1e-5f);
    ASSERT_NEAR(stats.scale, 0.0f, 1e-5f);

    free(data);
    return 0;
}

/* Image with known median and MAD.
 * 50 samples at 100.0 and 50 samples at 200.0 (stride-16, so we need
 * enough pixels).  Median = 150 (average of 100 and 200 at midpoint).
 * MAD = 50.0.  scale = 1.4826 * 50 = 74.13. */
static int test_bg_stats_known(void)
{
    /* Create 1600 pixels so stride-16 gives exactly 100 samples. */
    int npix = 1600;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    /* Fill so that every 16th pixel alternates: first half = 100, second half = 200. */
    for (int i = 0; i < npix; i++) {
        /* Stride-16 sampling picks i=0,16,32,...,1584 → 100 samples.
         * First 50 (i=0..799) → 100.0, last 50 (i=800..1599) → 200.0. */
        if (i < npix / 2)
            data[i] = 100.0f;
        else
            data[i] = 200.0f;
    }

    BgStats stats;
    ASSERT_OK(bg_compute_stats(data, npix, &stats));
    /* Median of [100×50, 200×50] = (100+200)/2 = 150 */
    ASSERT_NEAR(stats.background, 150.0f, 1.0f);
    /* MAD of |vals - 150| = all 50.0, so MAD = 50.0, scale = 74.13 */
    ASSERT_NEAR(stats.scale, 1.4826f * 50.0f, 1.0f);

    free(data);
    return 0;
}

/* NaN pixels should be excluded from stats computation. */
static int test_bg_stats_nan_handling(void)
{
    int npix = 1600;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);

    /* Set every pixel to 500, but sprinkle NaN at every 16th pixel
     * for the first half.  The sampled non-NaN values should all be 500. */
    for (int i = 0; i < npix; i++) data[i] = 500.0f;
    /* Make first 50 sampled positions NaN */
    for (int i = 0; i < npix / 2; i += 16) data[i] = NAN;

    BgStats stats;
    ASSERT_OK(bg_compute_stats(data, npix, &stats));
    /* All non-NaN sampled values are 500.0 */
    ASSERT_NEAR(stats.background, 500.0f, 1e-3f);
    ASSERT_NEAR(stats.scale, 0.0f, 1e-3f);

    free(data);
    return 0;
}

/* All NaN → degenerate: background=0, scale=0. */
static int test_bg_stats_all_nan(void)
{
    int npix = 256;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    for (int i = 0; i < npix; i++) data[i] = NAN;

    BgStats stats;
    ASSERT_OK(bg_compute_stats(data, npix, &stats));
    ASSERT_NEAR(stats.background, 0.0f, 1e-10f);
    ASSERT_NEAR(stats.scale, 0.0f, 1e-10f);

    free(data);
    return 0;
}

/* Invalid args. */
static int test_bg_stats_invalid_args(void)
{
    BgStats stats;
    float dummy = 1.0f;
    ASSERT_ERR(bg_compute_stats(NULL, 100, &stats), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(bg_compute_stats(&dummy, 0, &stats), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(bg_compute_stats(&dummy, 100, NULL), DSO_ERR_INVALID_ARG);
    return 0;
}

/* -------------------------------------------------------------------------
 * bg_normalize_cpu tests
 * ------------------------------------------------------------------------- */

/* Frame with bg=100, ref with bg=200, both same scale → shift by +100. */
static int test_bg_normalize_shifts_background(void)
{
    int npix = 256;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    for (int i = 0; i < npix; i++) data[i] = 100.0f + (float)i * 0.01f;

    BgStats frame_stats = {100.0f, 10.0f};  /* bg=100, scale=10 */
    BgStats ref_stats   = {200.0f, 10.0f};  /* bg=200, scale=10 */

    ASSERT_OK(bg_normalize_cpu(data, npix, &frame_stats, &ref_stats));

    /* ratio = 10/10 = 1.0, so pixel = (pixel - 100) * 1.0 + 200 = pixel + 100 */
    for (int i = 0; i < npix; i++) {
        float expected = 200.0f + (float)i * 0.01f;
        ASSERT_NEAR(data[i], expected, 1e-3f);
    }

    free(data);
    return 0;
}

/* Frame with different signal scale → verify scaling. */
static int test_bg_normalize_scales(void)
{
    int npix = 256;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    /* Frame signal: background 100, signal at 200 (100 above bg) */
    for (int i = 0; i < npix; i++) data[i] = 200.0f;

    BgStats frame_stats = {100.0f, 20.0f};  /* bg=100, scale=20 */
    BgStats ref_stats   = {150.0f, 40.0f};  /* bg=150, scale=40 */

    ASSERT_OK(bg_normalize_cpu(data, npix, &frame_stats, &ref_stats));

    /* ratio = 40/20 = 2.0
     * pixel = (200 - 100) * 2.0 + 150 = 200 + 150 = 350 */
    for (int i = 0; i < npix; i++) {
        ASSERT_NEAR(data[i], 350.0f, 1e-3f);
    }

    free(data);
    return 0;
}

/* NaN pixels should remain NaN after normalization. */
static int test_bg_normalize_nan_preserved(void)
{
    int npix = 64;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    for (int i = 0; i < npix; i++) data[i] = 100.0f;
    data[0]  = NAN;
    data[10] = NAN;
    data[63] = NAN;

    BgStats frame_stats = {100.0f, 10.0f};
    BgStats ref_stats   = {200.0f, 10.0f};

    ASSERT_OK(bg_normalize_cpu(data, npix, &frame_stats, &ref_stats));

    ASSERT(isnan(data[0]));
    ASSERT(isnan(data[10]));
    ASSERT(isnan(data[63]));
    /* Non-NaN pixels should be shifted */
    ASSERT_NEAR(data[1], 200.0f, 1e-3f);

    free(data);
    return 0;
}

/* Degenerate frame (scale ≈ 0) → no modification. */
static int test_bg_normalize_degenerate_skip(void)
{
    int npix = 64;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    for (int i = 0; i < npix; i++) data[i] = 42.0f;

    BgStats frame_stats = {42.0f, 0.0f};  /* scale=0 → flat frame */
    BgStats ref_stats   = {200.0f, 10.0f};

    ASSERT_OK(bg_normalize_cpu(data, npix, &frame_stats, &ref_stats));

    /* Data should be unchanged */
    for (int i = 0; i < npix; i++) {
        ASSERT_NEAR(data[i], 42.0f, 1e-10f);
    }

    free(data);
    return 0;
}

/* Degenerate reference (scale ≈ 0) → no modification. */
static int test_bg_normalize_degenerate_ref_skip(void)
{
    int npix = 64;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(data);
    for (int i = 0; i < npix; i++) data[i] = 42.0f;

    BgStats frame_stats = {42.0f, 10.0f};
    BgStats ref_stats   = {200.0f, 0.0f};  /* ref scale=0 → skip */

    ASSERT_OK(bg_normalize_cpu(data, npix, &frame_stats, &ref_stats));

    /* Data should be unchanged */
    for (int i = 0; i < npix; i++) {
        ASSERT_NEAR(data[i], 42.0f, 1e-10f);
    }

    free(data);
    return 0;
}

/* Invalid args for normalize. */
static int test_bg_normalize_invalid_args(void)
{
    float data[4] = {1, 2, 3, 4};
    BgStats fs = {1.0f, 1.0f};
    BgStats rs = {2.0f, 1.0f};

    ASSERT_ERR(bg_normalize_cpu(NULL, 4, &fs, &rs), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(bg_normalize_cpu(data, 0, &fs, &rs), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(bg_normalize_cpu(data, 4, NULL, &rs), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(bg_normalize_cpu(data, 4, &fs, NULL), DSO_ERR_INVALID_ARG);
    return 0;
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(void)
{
    SUITE("bg_compute_stats");
    RUN(test_bg_stats_constant);
    RUN(test_bg_stats_known);
    RUN(test_bg_stats_nan_handling);
    RUN(test_bg_stats_all_nan);
    RUN(test_bg_stats_invalid_args);

    SUITE("bg_normalize_cpu");
    RUN(test_bg_normalize_shifts_background);
    RUN(test_bg_normalize_scales);
    RUN(test_bg_normalize_nan_preserved);
    RUN(test_bg_normalize_degenerate_skip);
    RUN(test_bg_normalize_degenerate_ref_skip);
    RUN(test_bg_normalize_invalid_args);

    return SUMMARY();
}
