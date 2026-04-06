/*
 * test_frame_quality.c — Unit tests for frame quality scoring.
 */

#include "test_framework.h"
#include "frame_quality.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Stamp a 2D Gaussian into buf at (cx, cy) with given sigma and peak. */
static void stamp_gaussian(float *buf, int W, int H,
                           float cx, float cy,
                           float sigma_x, float sigma_y, float peak)
{
    int r = (int)(4.0f * fmaxf(sigma_x, sigma_y)) + 1;
    int x0 = (int)(cx - r); if (x0 < 0) x0 = 0;
    int x1 = (int)(cx + r); if (x1 >= W) x1 = W - 1;
    int y0 = (int)(cy - r); if (y0 < 0) y0 = 0;
    int y1 = (int)(cy + r); if (y1 >= H) y1 = H - 1;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            float ex = dx * dx / (2.0f * sigma_x * sigma_x);
            float ey = dy * dy / (2.0f * sigma_y * sigma_y);
            buf[y * W + x] += peak * expf(-(ex + ey));
        }
    }
}

static float *make_zero_image(int W, int H)
{
    return (float *)calloc((size_t)W * H, sizeof(float));
}

static float *make_const_image(int W, int H, float val)
{
    float *buf = (float *)malloc((size_t)W * H * sizeof(float));
    if (!buf) return NULL;
    for (long i = 0; i < (long)W * H; i++) buf[i] = val;
    return buf;
}

/* -------------------------------------------------------------------------
 * Tests
 * ------------------------------------------------------------------------- */

/* Test FWHM measurement on a single Gaussian star.
 * Gaussian FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma.
 * For sigma=2.0, expected FWHM ≈ 4.71 pixels. */
static int test_fwhm_gaussian(void)
{
    int W = 64, H = 64;
    float *conv = make_zero_image(W, H);
    float *lum  = make_const_image(W, H, 100.0f);
    ASSERT_NE(conv, NULL);
    ASSERT_NE(lum, NULL);

    float sigma = 2.0f;
    float peak = 1000.0f;
    stamp_gaussian(conv, W, H, 32.0f, 32.0f, sigma, sigma, peak);
    /* Also stamp into lum so background is slightly elevated */
    stamp_gaussian(lum, W, H, 32.0f, 32.0f, sigma, sigma, peak);

    StarPos sp = {32.0f, 32.0f, 5000.0f};
    StarList sl = {&sp, 1};

    FrameQuality fq;
    ASSERT_OK(frame_quality_compute(conv, lum, &sl, W, H, &fq));

    float expected_fwhm = 2.355f * sigma;  /* ~4.71 */
    /* Allow 25% tolerance — we're walking on discrete pixels */
    ASSERT_NEAR(fq.fwhm, expected_fwhm, expected_fwhm * 0.25f);
    ASSERT_EQ(fq.star_count, 1);

    free(conv);
    free(lum);
    return 0;
}

/* Tight stars should have lower FWHM than loose stars. */
static int test_fwhm_tight_vs_loose(void)
{
    int W = 128, H = 128;
    float *conv_tight = make_zero_image(W, H);
    float *conv_loose = make_zero_image(W, H);
    float *lum        = make_const_image(W, H, 100.0f);
    ASSERT_NE(conv_tight, NULL);
    ASSERT_NE(conv_loose, NULL);
    ASSERT_NE(lum, NULL);

    /* 3 tight stars (sigma=1.5) */
    stamp_gaussian(conv_tight, W, H, 30.0f, 30.0f, 1.5f, 1.5f, 1000.0f);
    stamp_gaussian(conv_tight, W, H, 60.0f, 30.0f, 1.5f, 1.5f, 1000.0f);
    stamp_gaussian(conv_tight, W, H, 90.0f, 60.0f, 1.5f, 1.5f, 1000.0f);

    /* 3 loose stars (sigma=4.0) */
    stamp_gaussian(conv_loose, W, H, 30.0f, 30.0f, 4.0f, 4.0f, 1000.0f);
    stamp_gaussian(conv_loose, W, H, 60.0f, 30.0f, 4.0f, 4.0f, 1000.0f);
    stamp_gaussian(conv_loose, W, H, 90.0f, 60.0f, 4.0f, 4.0f, 1000.0f);

    StarPos sps[] = {
        {30.0f, 30.0f, 5000.0f},
        {60.0f, 30.0f, 5000.0f},
        {90.0f, 60.0f, 5000.0f}
    };
    StarList sl = {sps, 3};

    FrameQuality fq_tight, fq_loose;
    ASSERT_OK(frame_quality_compute(conv_tight, lum, &sl, W, H, &fq_tight));
    ASSERT_OK(frame_quality_compute(conv_loose, lum, &sl, W, H, &fq_loose));

    /* Tight stars should have smaller FWHM */
    ASSERT_NE(fq_tight.fwhm >= fq_loose.fwhm, 1);

    free(conv_tight);
    free(conv_loose);
    free(lum);
    return 0;
}

/* Circular star should have high roundness. */
static int test_roundness_circular(void)
{
    int W = 64, H = 64;
    float *conv = make_zero_image(W, H);
    float *lum  = make_const_image(W, H, 100.0f);
    ASSERT_NE(conv, NULL);
    ASSERT_NE(lum, NULL);

    stamp_gaussian(conv, W, H, 32.0f, 32.0f, 3.0f, 3.0f, 1000.0f);
    StarPos sp = {32.0f, 32.0f, 5000.0f};
    StarList sl = {&sp, 1};

    FrameQuality fq;
    ASSERT_OK(frame_quality_compute(conv, lum, &sl, W, H, &fq));

    /* Circular star: roundness should be > 0.9 */
    ASSERT_NE(fq.roundness > 0.9f, 0);

    free(conv);
    free(lum);
    return 0;
}

/* Elongated star should have low roundness. */
static int test_roundness_elongated(void)
{
    int W = 64, H = 64;
    float *conv = make_zero_image(W, H);
    float *lum  = make_const_image(W, H, 100.0f);
    ASSERT_NE(conv, NULL);
    ASSERT_NE(lum, NULL);

    /* Elongated: sigma_x=2, sigma_y=6 → FWHM ratio ≈ 2/6 = 0.33 */
    stamp_gaussian(conv, W, H, 32.0f, 32.0f, 2.0f, 6.0f, 1000.0f);
    StarPos sp = {32.0f, 32.0f, 5000.0f};
    StarList sl = {&sp, 1};

    FrameQuality fq;
    ASSERT_OK(frame_quality_compute(conv, lum, &sl, W, H, &fq));

    /* Elongated star: roundness should be < 0.6 */
    ASSERT_NE(fq.roundness < 0.6f, 0);

    free(conv);
    free(lum);
    return 0;
}

/* Background estimation on a uniform image. */
static int test_background(void)
{
    int W = 64, H = 64;
    float bg_val = 500.0f;
    float *lum  = make_const_image(W, H, bg_val);
    float *conv = make_zero_image(W, H);
    ASSERT_NE(lum, NULL);
    ASSERT_NE(conv, NULL);

    StarList sl = {NULL, 0};
    FrameQuality fq;
    ASSERT_OK(frame_quality_compute(conv, lum, &sl, W, H, &fq));

    /* Background should be very close to the uniform value */
    ASSERT_NEAR(fq.background, bg_val, 1.0f);

    free(lum);
    free(conv);
    return 0;
}

/* Better frames should get higher composite scores. */
static int test_score_ordering(void)
{
    int W = 128, H = 128;
    float *lum  = make_const_image(W, H, 100.0f);
    ASSERT_NE(lum, NULL);

    /* Good frame: many tight, round stars */
    float *conv_good = make_zero_image(W, H);
    StarPos sps_good[5];
    for (int i = 0; i < 5; i++) {
        float x = 20.0f + 20.0f * i;
        float y = 64.0f;
        stamp_gaussian(conv_good, W, H, x, y, 2.0f, 2.0f, 1000.0f);
        sps_good[i] = (StarPos){x, y, 5000.0f};
    }
    StarList sl_good = {sps_good, 5};

    /* Bad frame: few loose, elongated stars */
    float *conv_bad = make_zero_image(W, H);
    StarPos sps_bad[2];
    for (int i = 0; i < 2; i++) {
        float x = 40.0f + 40.0f * i;
        float y = 64.0f;
        stamp_gaussian(conv_bad, W, H, x, y, 5.0f, 2.0f, 500.0f);
        sps_bad[i] = (StarPos){x, y, 2000.0f};
    }
    StarList sl_bad = {sps_bad, 2};

    FrameQuality fq_good, fq_bad;
    ASSERT_OK(frame_quality_compute(conv_good, lum, &sl_good, W, H, &fq_good));
    ASSERT_OK(frame_quality_compute(conv_bad,  lum, &sl_bad,  W, H, &fq_bad));

    /* Good frame should have higher composite score */
    ASSERT_NE(fq_good.composite > fq_bad.composite, 0);

    free(conv_good);
    free(conv_bad);
    free(lum);
    return 0;
}

/* Reference normalization should set normalized to 100.0. */
static int test_normalize_reference(void)
{
    FrameQuality fq;
    fq.composite = 42.5f;

    /* Normalize to itself = 100.0 */
    frame_quality_normalize(&fq, fq.composite);
    ASSERT_NEAR(fq.normalized, 100.0f, 0.001f);

    /* Another frame with half the composite → 50.0 */
    FrameQuality fq2;
    fq2.composite = 21.25f;
    frame_quality_normalize(&fq2, fq.composite);
    ASSERT_NEAR(fq2.normalized, 50.0f, 0.001f);

    return 0;
}

/* Zero stars should not crash, composite should be 0. */
static int test_zero_stars(void)
{
    int W = 32, H = 32;
    float *conv = make_zero_image(W, H);
    float *lum  = make_const_image(W, H, 100.0f);
    ASSERT_NE(conv, NULL);
    ASSERT_NE(lum, NULL);

    StarList sl = {NULL, 0};
    FrameQuality fq;
    ASSERT_OK(frame_quality_compute(conv, lum, &sl, W, H, &fq));

    ASSERT_EQ(fq.star_count, 0);
    ASSERT_NEAR(fq.composite, 0.0f, 0.001f);

    free(conv);
    free(lum);
    return 0;
}

/* NULL arguments should return DSO_ERR_INVALID_ARG. */
static int test_null_args(void)
{
    int W = 16, H = 16;
    float *conv = make_zero_image(W, H);
    float *lum  = make_const_image(W, H, 100.0f);
    StarList sl = {NULL, 0};
    FrameQuality fq;

    ASSERT_ERR(frame_quality_compute(NULL, lum,  &sl, W, H, &fq), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(frame_quality_compute(conv, NULL, &sl, W, H, &fq), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(frame_quality_compute(conv, lum,  NULL, W, H, &fq), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(frame_quality_compute(conv, lum,  &sl, W, H, NULL), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(frame_quality_compute(conv, lum,  &sl, 0, H, &fq), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(frame_quality_compute(conv, lum,  &sl, W, 0, &fq), DSO_ERR_INVALID_ARG);

    free(conv);
    free(lum);
    return 0;
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(void)
{
    RUN(test_fwhm_gaussian);
    RUN(test_fwhm_tight_vs_loose);
    RUN(test_roundness_circular);
    RUN(test_roundness_elongated);
    RUN(test_background);
    RUN(test_score_ordering);
    RUN(test_normalize_reference);
    RUN(test_zero_stars);
    RUN(test_null_args);
    SUMMARY();
}
