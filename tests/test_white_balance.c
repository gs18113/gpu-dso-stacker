/*
 * test_white_balance.c — Unit tests for white balance module.
 *
 * Tests cover:
 *   bayer_color:
 *     - All four Bayer patterns: position → channel mapping
 *
 *   wb_apply_bayer:
 *     - All four Bayer patterns (RGGB, BGGR, GRBG, GBRG)
 *     - BAYER_NONE passthrough (no-op)
 *     - Identity multipliers (1, 1, 1) leave data unchanged
 *     - Argument validation (null data, zero dimensions)
 *
 *   wb_auto_compute:
 *     - Gray-world with known channel imbalance
 *     - Uniform mosaic produces multipliers near 1.0
 *     - BAYER_NONE returns DSO_ERR_INVALID_ARG
 *     - Argument validation
 */

#include "test_framework.h"
#include "white_balance.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =========================================================================
 * Helpers
 * ========================================================================= */

static float *alloc_fill(int W, int H, float val)
{
    int n = W * H;
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) return NULL;
    for (int i = 0; i < n; i++) p[i] = val;
    return p;
}

static float *make_mosaic(BayerPattern pat, int W, int H,
                           float r_val, float g_val, float b_val)
{
    float *data = (float *)malloc((size_t)W * H * sizeof(float));
    if (!data) return NULL;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int c = bayer_color(pat, x, y);
            float v = (c == 0) ? r_val : (c == 1) ? g_val : b_val;
            data[y * W + x] = v;
        }
    }
    return data;
}

/* =========================================================================
 * bayer_color tests
 * ========================================================================= */

static int test_bayer_color_rggb(void)
{
    ASSERT_EQ(bayer_color(BAYER_RGGB, 0, 0), 0);
    ASSERT_EQ(bayer_color(BAYER_RGGB, 1, 0), 1);
    ASSERT_EQ(bayer_color(BAYER_RGGB, 0, 1), 1);
    ASSERT_EQ(bayer_color(BAYER_RGGB, 1, 1), 2);
    return 0;
}

static int test_bayer_color_bggr(void)
{
    ASSERT_EQ(bayer_color(BAYER_BGGR, 0, 0), 2);
    ASSERT_EQ(bayer_color(BAYER_BGGR, 1, 0), 1);
    ASSERT_EQ(bayer_color(BAYER_BGGR, 0, 1), 1);
    ASSERT_EQ(bayer_color(BAYER_BGGR, 1, 1), 0);
    return 0;
}

static int test_bayer_color_grbg(void)
{
    ASSERT_EQ(bayer_color(BAYER_GRBG, 0, 0), 1);
    ASSERT_EQ(bayer_color(BAYER_GRBG, 1, 0), 0);
    ASSERT_EQ(bayer_color(BAYER_GRBG, 0, 1), 2);
    ASSERT_EQ(bayer_color(BAYER_GRBG, 1, 1), 1);
    return 0;
}

static int test_bayer_color_gbrg(void)
{
    ASSERT_EQ(bayer_color(BAYER_GBRG, 0, 0), 1);
    ASSERT_EQ(bayer_color(BAYER_GBRG, 1, 0), 2);
    ASSERT_EQ(bayer_color(BAYER_GBRG, 0, 1), 0);
    ASSERT_EQ(bayer_color(BAYER_GBRG, 1, 1), 1);
    return 0;
}

/* =========================================================================
 * wb_apply_bayer tests
 * ========================================================================= */

static int test_wb_apply_rggb(void)
{
    int W = 4, H = 4;
    float *data = make_mosaic(BAYER_RGGB, W, H, 1.0f, 1.0f, 1.0f);
    ASSERT_NOT_NULL(data);

    DsoError err = wb_apply_bayer(data, W, H, BAYER_RGGB, 2.0f, 1.0f, 0.5f);
    ASSERT_EQ(err, DSO_OK);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int c = bayer_color(BAYER_RGGB, x, y);
            float expected = (c == 0) ? 2.0f : (c == 1) ? 1.0f : 0.5f;
            ASSERT_NEAR(data[y * W + x], expected, 1e-6f);
        }
    }
    free(data);
    return 0;
}

static int test_wb_apply_bggr(void)
{
    int W = 4, H = 4;
    float *data = make_mosaic(BAYER_BGGR, W, H, 1.0f, 1.0f, 1.0f);
    ASSERT_NOT_NULL(data);

    DsoError err = wb_apply_bayer(data, W, H, BAYER_BGGR, 2.0f, 1.0f, 0.5f);
    ASSERT_EQ(err, DSO_OK);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int c = bayer_color(BAYER_BGGR, x, y);
            float expected = (c == 0) ? 2.0f : (c == 1) ? 1.0f : 0.5f;
            ASSERT_NEAR(data[y * W + x], expected, 1e-6f);
        }
    }
    free(data);
    return 0;
}

static int test_wb_apply_grbg(void)
{
    int W = 4, H = 4;
    float *data = make_mosaic(BAYER_GRBG, W, H, 1.0f, 1.0f, 1.0f);
    ASSERT_NOT_NULL(data);

    DsoError err = wb_apply_bayer(data, W, H, BAYER_GRBG, 2.0f, 1.0f, 0.5f);
    ASSERT_EQ(err, DSO_OK);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int c = bayer_color(BAYER_GRBG, x, y);
            float expected = (c == 0) ? 2.0f : (c == 1) ? 1.0f : 0.5f;
            ASSERT_NEAR(data[y * W + x], expected, 1e-6f);
        }
    }
    free(data);
    return 0;
}

static int test_wb_apply_gbrg(void)
{
    int W = 4, H = 4;
    float *data = make_mosaic(BAYER_GBRG, W, H, 1.0f, 1.0f, 1.0f);
    ASSERT_NOT_NULL(data);

    DsoError err = wb_apply_bayer(data, W, H, BAYER_GBRG, 2.0f, 1.0f, 0.5f);
    ASSERT_EQ(err, DSO_OK);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int c = bayer_color(BAYER_GBRG, x, y);
            float expected = (c == 0) ? 2.0f : (c == 1) ? 1.0f : 0.5f;
            ASSERT_NEAR(data[y * W + x], expected, 1e-6f);
        }
    }
    free(data);
    return 0;
}

static int test_wb_apply_bayer_none_noop(void)
{
    int W = 4, H = 4;
    float *data = alloc_fill(W, H, 0.42f);
    ASSERT_NOT_NULL(data);

    DsoError err = wb_apply_bayer(data, W, H, BAYER_NONE, 2.0f, 3.0f, 4.0f);
    ASSERT_EQ(err, DSO_OK);

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(data[i], 0.42f, 1e-6f);

    free(data);
    return 0;
}

static int test_wb_apply_identity(void)
{
    int W = 4, H = 4;
    float *data = make_mosaic(BAYER_RGGB, W, H, 0.3f, 0.5f, 0.7f);
    ASSERT_NOT_NULL(data);

    float *orig = (float *)malloc((size_t)(W * H) * sizeof(float));
    ASSERT_NOT_NULL(orig);
    memcpy(orig, data, (size_t)(W * H) * sizeof(float));

    DsoError err = wb_apply_bayer(data, W, H, BAYER_RGGB, 1.0f, 1.0f, 1.0f);
    ASSERT_EQ(err, DSO_OK);

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(data[i], orig[i], 1e-6f);

    free(data);
    free(orig);
    return 0;
}

static int test_wb_apply_arg_validation(void)
{
    float data[16] = {0};
    ASSERT_EQ(wb_apply_bayer(NULL, 4, 4, BAYER_RGGB, 1, 1, 1),
              DSO_ERR_INVALID_ARG);
    ASSERT_EQ(wb_apply_bayer(data, 0, 4, BAYER_RGGB, 1, 1, 1),
              DSO_ERR_INVALID_ARG);
    ASSERT_EQ(wb_apply_bayer(data, 4, 0, BAYER_RGGB, 1, 1, 1),
              DSO_ERR_INVALID_ARG);
    return 0;
}

/* =========================================================================
 * wb_auto_compute tests
 * ========================================================================= */

static int test_wb_auto_gray_world(void)
{
    int W = 4, H = 4;
    float *data = make_mosaic(BAYER_RGGB, W, H, 2.0f, 1.0f, 1.0f);
    ASSERT_NOT_NULL(data);

    float r_mul = 0, g_mul = 0, b_mul = 0;
    DsoError err = wb_auto_compute(data, W, H, BAYER_RGGB,
                                    &r_mul, &g_mul, &b_mul);
    ASSERT_EQ(err, DSO_OK);
    ASSERT_NEAR(r_mul, 0.5f, 0.01f);
    ASSERT_NEAR(g_mul, 1.0f, 0.01f);
    ASSERT_NEAR(b_mul, 1.0f, 0.01f);

    free(data);
    return 0;
}

static int test_wb_auto_uniform(void)
{
    int W = 8, H = 8;
    float *data = make_mosaic(BAYER_RGGB, W, H, 0.5f, 0.5f, 0.5f);
    ASSERT_NOT_NULL(data);

    float r_mul = 0, g_mul = 0, b_mul = 0;
    DsoError err = wb_auto_compute(data, W, H, BAYER_RGGB,
                                    &r_mul, &g_mul, &b_mul);
    ASSERT_EQ(err, DSO_OK);
    ASSERT_NEAR(r_mul, 1.0f, 0.01f);
    ASSERT_NEAR(g_mul, 1.0f, 0.01f);
    ASSERT_NEAR(b_mul, 1.0f, 0.01f);

    free(data);
    return 0;
}

static int test_wb_auto_blue_heavy(void)
{
    int W = 4, H = 4;
    float *data = make_mosaic(BAYER_BGGR, W, H, 1.0f, 1.0f, 3.0f);
    ASSERT_NOT_NULL(data);

    float r_mul = 0, g_mul = 0, b_mul = 0;
    DsoError err = wb_auto_compute(data, W, H, BAYER_BGGR,
                                    &r_mul, &g_mul, &b_mul);
    ASSERT_EQ(err, DSO_OK);
    ASSERT_NEAR(r_mul, 1.0f, 0.01f);
    ASSERT_NEAR(g_mul, 1.0f, 0.01f);
    ASSERT_NEAR(b_mul, 0.333f, 0.02f);

    free(data);
    return 0;
}

static int test_wb_auto_bayer_none_rejected(void)
{
    float data[16] = {0};
    float r, g, b;
    ASSERT_EQ(wb_auto_compute(data, 4, 4, BAYER_NONE, &r, &g, &b),
              DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_wb_auto_arg_validation(void)
{
    float data[16] = {0};
    float r, g, b;
    ASSERT_EQ(wb_auto_compute(NULL, 4, 4, BAYER_RGGB, &r, &g, &b),
              DSO_ERR_INVALID_ARG);
    ASSERT_EQ(wb_auto_compute(data, 4, 4, BAYER_RGGB, NULL, &g, &b),
              DSO_ERR_INVALID_ARG);
    ASSERT_EQ(wb_auto_compute(data, 0, 4, BAYER_RGGB, &r, &g, &b),
              DSO_ERR_INVALID_ARG);
    return 0;
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    SUITE("bayer_color — CFA position lookup");
    RUN(test_bayer_color_rggb);
    RUN(test_bayer_color_bggr);
    RUN(test_bayer_color_grbg);
    RUN(test_bayer_color_gbrg);

    SUITE("wb_apply_bayer — per-pattern multiplier application");
    RUN(test_wb_apply_rggb);
    RUN(test_wb_apply_bggr);
    RUN(test_wb_apply_grbg);
    RUN(test_wb_apply_gbrg);
    RUN(test_wb_apply_bayer_none_noop);
    RUN(test_wb_apply_identity);
    RUN(test_wb_apply_arg_validation);

    SUITE("wb_auto_compute — gray-world multiplier computation");
    RUN(test_wb_auto_gray_world);
    RUN(test_wb_auto_uniform);
    RUN(test_wb_auto_blue_heavy);
    RUN(test_wb_auto_bayer_none_rejected);
    RUN(test_wb_auto_arg_validation);

    return SUMMARY();
}
