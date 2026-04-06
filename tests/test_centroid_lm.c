/*
 * test_centroid_lm.c — Tests for Levenberg-Marquardt 2D Gaussian centroid fitting
 */
#include "test_framework.h"
#include "centroid_lm.h"
#include "dso_types.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Helper: generate a synthetic image with a 2D Gaussian star
 * f(x,y) = A * exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2)) + B
 * ------------------------------------------------------------------------- */
static float *make_gaussian_image(int W, int H,
                                   float cx, float cy,
                                   float A, float sigma, float B)
{
    float *img = (float *)calloc((size_t)W * H, sizeof(float));
    if (!img) return NULL;
    float s2 = sigma * sigma;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            img[y * W + x] = A * expf(-(dx * dx + dy * dy) / (2.0f * s2)) + B;
        }
    }
    return img;
}

/* -------------------------------------------------------------------------
 * Test: perfect Gaussian at known sub-pixel position
 * Verify fitted centroid matches within 0.02 pixels
 * ------------------------------------------------------------------------- */
static int test_lm_perfect_gaussian(void)
{
    int W = 31, H = 31;
    float true_cx = 15.37f, true_cy = 15.71f;
    float A = 1000.0f, sigma = 2.0f, B = 100.0f;

    float *img = make_gaussian_image(W, H, true_cx, true_cy, A, sigma, B);
    ASSERT_NOT_NULL(img);

    /* Create a single-star StarList starting from an offset CoM */
    StarList stars;
    stars.n = 1;
    stars.stars = (StarPos *)malloc(sizeof(StarPos));
    stars.stars[0].x = 15.0f;  /* intentionally offset from truth */
    stars.stars[0].y = 16.0f;
    stars.stars[0].flux = 5000.0f;

    DsoError err = centroid_lm_refine(&stars, img, W, H, sigma, 0.0f, 15);
    ASSERT_OK(err);

    ASSERT_NEAR(stars.stars[0].x, true_cx, 0.02f);
    ASSERT_NEAR(stars.stars[0].y, true_cy, 0.02f);

    free(stars.stars);
    free(img);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: Gaussian with noise — verify within 0.1 pixels
 * ------------------------------------------------------------------------- */
static int test_lm_noisy_gaussian(void)
{
    int W = 31, H = 31;
    float true_cx = 15.42f, true_cy = 15.63f;
    float A = 1000.0f, sigma = 2.5f, B = 100.0f;

    float *img = make_gaussian_image(W, H, true_cx, true_cy, A, sigma, B);
    ASSERT_NOT_NULL(img);

    /* Add deterministic pseudo-noise (seeded LCG) */
    unsigned int seed = 42;
    for (int i = 0; i < W * H; i++) {
        seed = seed * 1103515245u + 12345u;
        float noise = ((float)(seed >> 16) / 32768.0f - 1.0f) * (A * 0.05f);
        img[i] += noise;
    }

    StarList stars;
    stars.n = 1;
    stars.stars = (StarPos *)malloc(sizeof(StarPos));
    stars.stars[0].x = 15.0f;
    stars.stars[0].y = 16.0f;
    stars.stars[0].flux = 5000.0f;

    DsoError err = centroid_lm_refine(&stars, img, W, H, sigma, 0.0f, 20);
    ASSERT_OK(err);

    ASSERT_NEAR(stars.stars[0].x, true_cx, 0.1f);
    ASSERT_NEAR(stars.stars[0].y, true_cy, 0.1f);

    free(stars.stars);
    free(img);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: convergence from significant CoM offset (~0.5 px)
 * ------------------------------------------------------------------------- */
static int test_lm_convergence_from_offset(void)
{
    int W = 31, H = 31;
    float true_cx = 15.25f, true_cy = 15.75f;
    float A = 2000.0f, sigma = 2.0f, B = 50.0f;

    float *img = make_gaussian_image(W, H, true_cx, true_cy, A, sigma, B);
    ASSERT_NOT_NULL(img);

    StarList stars;
    stars.n = 1;
    stars.stars = (StarPos *)malloc(sizeof(StarPos));
    /* Start 0.5px away from true center */
    stars.stars[0].x = 14.75f;
    stars.stars[0].y = 16.25f;
    stars.stars[0].flux = 5000.0f;

    DsoError err = centroid_lm_refine(&stars, img, W, H, sigma, 0.0f, 20);
    ASSERT_OK(err);

    ASSERT_NEAR(stars.stars[0].x, true_cx, 0.02f);
    ASSERT_NEAR(stars.stars[0].y, true_cy, 0.02f);

    free(stars.stars);
    free(img);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: flat image — LM should not converge, CoM preserved
 * ------------------------------------------------------------------------- */
static int test_lm_no_convergence_fallback(void)
{
    int W = 31, H = 31;
    float *img = (float *)malloc((size_t)W * H * sizeof(float));
    ASSERT_NOT_NULL(img);

    /* Flat constant image — no star */
    for (int i = 0; i < W * H; i++) img[i] = 100.0f;

    float original_x = 15.5f, original_y = 15.5f;

    StarList stars;
    stars.n = 1;
    stars.stars = (StarPos *)malloc(sizeof(StarPos));
    stars.stars[0].x = original_x;
    stars.stars[0].y = original_y;
    stars.stars[0].flux = 100.0f;

    DsoError err = centroid_lm_refine(&stars, img, W, H, 2.0f, 0.0f, 15);
    ASSERT_OK(err);

    /* Position should be unchanged (LM couldn't fit a Gaussian to flat data) */
    ASSERT_NEAR(stars.stars[0].x, original_x, 0.01f);
    ASSERT_NEAR(stars.stars[0].y, original_y, 0.01f);

    free(stars.stars);
    free(img);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: star near image edge — no crash, reasonable result
 * ------------------------------------------------------------------------- */
static int test_lm_edge_star(void)
{
    int W = 31, H = 31;
    float true_cx = 2.3f, true_cy = 2.7f;
    float A = 1000.0f, sigma = 2.0f, B = 50.0f;

    float *img = make_gaussian_image(W, H, true_cx, true_cy, A, sigma, B);
    ASSERT_NOT_NULL(img);

    StarList stars;
    stars.n = 1;
    stars.stars = (StarPos *)malloc(sizeof(StarPos));
    stars.stars[0].x = 2.0f;
    stars.stars[0].y = 3.0f;
    stars.stars[0].flux = 3000.0f;

    DsoError err = centroid_lm_refine(&stars, img, W, H, sigma, 0.0f, 15);
    ASSERT_OK(err);

    /* Should be close to true position (fitting window is clipped at edge) */
    ASSERT_NEAR(stars.stars[0].x, true_cx, 0.15f);
    ASSERT_NEAR(stars.stars[0].y, true_cy, 0.15f);

    free(stars.stars);
    free(img);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: multiple stars — all refined correctly
 * ------------------------------------------------------------------------- */
static int test_lm_multiple_stars(void)
{
    int W = 64, H = 64;
    float *img = (float *)calloc((size_t)W * H, sizeof(float));
    ASSERT_NOT_NULL(img);

    /* Background */
    for (int i = 0; i < W * H; i++) img[i] = 50.0f;

    /* Place 3 stars with known positions */
    float stars_cx[] = {15.3f, 45.7f, 30.5f};
    float stars_cy[] = {15.6f, 20.2f, 50.1f};
    float A = 800.0f, sigma = 2.0f, B = 50.0f;
    float s2 = sigma * sigma;

    for (int s = 0; s < 3; s++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float dx = (float)x - stars_cx[s];
                float dy = (float)y - stars_cy[s];
                img[y * W + x] += A * expf(-(dx * dx + dy * dy) / (2.0f * s2));
            }
        }
    }

    StarList sl;
    sl.n = 3;
    sl.stars = (StarPos *)malloc(3 * sizeof(StarPos));
    /* Start with integer-rounded positions */
    for (int i = 0; i < 3; i++) {
        sl.stars[i].x = floorf(stars_cx[i]);
        sl.stars[i].y = floorf(stars_cy[i]);
        sl.stars[i].flux = 5000.0f;
    }

    DsoError err = centroid_lm_refine(&sl, img, W, H, sigma, 0.0f, 15);
    ASSERT_OK(err);

    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(sl.stars[i].x, stars_cx[i], 0.05f);
        ASSERT_NEAR(sl.stars[i].y, stars_cy[i], 0.05f);
    }

    free(sl.stars);
    free(img);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: API argument validation
 * ------------------------------------------------------------------------- */
static int test_lm_api_args(void)
{
    ASSERT_ERR(centroid_lm_refine(NULL, NULL, 0, 0, 2.0f, 0.0f, 15),
               DSO_ERR_INVALID_ARG);

    /* Empty star list should return OK without doing anything */
    float img[4] = {0};
    StarList stars = {NULL, 0};
    ASSERT_OK(centroid_lm_refine(&stars, img, 2, 2, 2.0f, 0.0f, 15));

    return 0;
}

/* -------------------------------------------------------------------------
 * Test: fit with explicit radius parameter
 * ------------------------------------------------------------------------- */
static int test_lm_explicit_radius(void)
{
    int W = 41, H = 41;
    float true_cx = 20.4f, true_cy = 20.6f;
    float A = 1500.0f, sigma = 3.0f, B = 80.0f;

    float *img = make_gaussian_image(W, H, true_cx, true_cy, A, sigma, B);
    ASSERT_NOT_NULL(img);

    StarList stars;
    stars.n = 1;
    stars.stars = (StarPos *)malloc(sizeof(StarPos));
    stars.stars[0].x = 20.0f;
    stars.stars[0].y = 21.0f;
    stars.stars[0].flux = 5000.0f;

    /* Use explicit radius = 10 */
    DsoError err = centroid_lm_refine(&stars, img, W, H, sigma, 10.0f, 20);
    ASSERT_OK(err);

    ASSERT_NEAR(stars.stars[0].x, true_cx, 0.02f);
    ASSERT_NEAR(stars.stars[0].y, true_cy, 0.02f);

    free(stars.stars);
    free(img);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: narrow PSF (sigma = 1.0) — still converges
 * ------------------------------------------------------------------------- */
static int test_lm_narrow_psf(void)
{
    int W = 21, H = 21;
    float true_cx = 10.3f, true_cy = 10.7f;
    float A = 2000.0f, sigma = 1.0f, B = 30.0f;

    float *img = make_gaussian_image(W, H, true_cx, true_cy, A, sigma, B);
    ASSERT_NOT_NULL(img);

    StarList stars;
    stars.n = 1;
    stars.stars = (StarPos *)malloc(sizeof(StarPos));
    stars.stars[0].x = 10.0f;
    stars.stars[0].y = 11.0f;
    stars.stars[0].flux = 3000.0f;

    DsoError err = centroid_lm_refine(&stars, img, W, H, sigma, 0.0f, 20);
    ASSERT_OK(err);

    ASSERT_NEAR(stars.stars[0].x, true_cx, 0.05f);
    ASSERT_NEAR(stars.stars[0].y, true_cy, 0.05f);

    free(stars.stars);
    free(img);
    return 0;
}

/* ========================================================================= */

int main(void)
{
    SUITE("LM Gaussian Centroid Fitting — Basic");
    RUN(test_lm_perfect_gaussian);
    RUN(test_lm_noisy_gaussian);
    RUN(test_lm_convergence_from_offset);
    RUN(test_lm_narrow_psf);

    SUITE("LM Gaussian Centroid Fitting — Robustness");
    RUN(test_lm_no_convergence_fallback);
    RUN(test_lm_edge_star);
    RUN(test_lm_multiple_stars);
    RUN(test_lm_explicit_radius);

    SUITE("LM Gaussian Centroid Fitting — API");
    RUN(test_lm_api_args);

    return SUMMARY();
}
