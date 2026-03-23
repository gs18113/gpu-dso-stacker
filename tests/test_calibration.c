/*
 * test_calibration.c — Unit tests for the calibration module.
 *
 * Tests cover:
 *   - calib_apply_cpu: dark subtract, flat divide, dead-pixel guard,
 *     dimension validation, null-argument handling
 *   - calib_load_or_generate: FITS master loading, frame-list stacking,
 *     bias subtraction from darks, bias/darkflat subtraction from flats,
 *     flat per-frame normalization, winsorized-mean outlier rejection,
 *     median stacking, save-dir output
 *   - calib_free: safe on zero-init and populated structs
 *
 * Design for failure detection
 * ----------------------------
 * Each test is designed to fail with at least one common wrong implementation:
 *   - "dark not subtracted"          → test_apply_dark_only
 *   - "flat not divided"             → test_apply_flat_only
 *   - "no dead-pixel guard"          → test_apply_dead_pixel_guard
 *   - "wrong dimension not caught"   → test_apply_dimension_mismatch_*
 *   - "plain mean instead of wsor."  → test_winsorized_mean_clips_outlier
 *   - "bias not subtracted from dark"→ test_bias_subtracted_from_dark
 *   - "flat not normalised per-frame"→ test_flat_per_frame_normalization
 *   - "darkflat not subtracted"      → test_darkflat_subtracted_from_flat
 */

#include "test_framework.h"
#include "calibration.h"
#include "fits_io.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * Helper utilities
 * ------------------------------------------------------------------------- */

/* Create a W×H FITS file where every pixel equals `val`. */
static DsoError make_fits_const(const char *path, int W, int H, float val)
{
    Image img;
    img.data   = (float *)malloc((size_t)W * H * sizeof(float));
    img.width  = W;
    img.height = H;
    if (!img.data) return DSO_ERR_ALLOC;
    for (int i = 0; i < W * H; i++) img.data[i] = val;
    DsoError err = fits_save(path, &img);
    free(img.data);
    return err;
}

/* Create a 1×N FITS file (row of N pixels) from an array of values. */
static DsoError make_fits_row(const char *path, const float *vals, int N)
{
    Image img;
    img.data   = (float *)malloc((size_t)N * sizeof(float));
    img.width  = N;
    img.height = 1;
    if (!img.data) return DSO_ERR_ALLOC;
    memcpy(img.data, vals, (size_t)N * sizeof(float));
    DsoError err = fits_save(path, &img);
    free(img.data);
    return err;
}

/* Write a frame-list text file with `n` FITS paths (one per line). */
static DsoError write_framelist(const char *list_path,
                                const char **fits_paths, int n)
{
    FILE *fp = fopen(list_path, "w");
    if (!fp) return DSO_ERR_IO;
    for (int i = 0; i < n; i++)
        fprintf(fp, "%s\n", fits_paths[i]);
    fclose(fp);
    return DSO_OK;
}

/* Build a CalibFrames with caller-supplied pixel arrays (no heap ownership). */
static CalibFrames make_calib_inline(float *dark_data, int dw, int dh,
                                     float *flat_data, int fw, int fh)
{
    CalibFrames c = {0};
    if (dark_data) {
        c.dark.data   = dark_data;
        c.dark.width  = dw;
        c.dark.height = dh;
        c.has_dark    = 1;
    }
    if (flat_data) {
        c.flat.data   = flat_data;
        c.flat.width  = fw;
        c.flat.height = fh;
        c.has_flat    = 1;
    }
    return c;
}

/* =========================================================================
 * calib_apply_cpu — argument validation
 * ========================================================================= */

static int test_apply_null_img(void)
{
    CalibFrames c = {0};
    ASSERT_ERR(calib_apply_cpu(NULL, &c), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_apply_null_calib(void)
{
    Image img = {0};
    float buf[4] = {1, 2, 3, 4};
    img.data = buf; img.width = 2; img.height = 2;
    ASSERT_ERR(calib_apply_cpu(&img, NULL), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_apply_no_calib_is_noop(void)
{
    /* has_dark=0, has_flat=0 → DSO_OK, data unchanged */
    CalibFrames c = {0};
    float buf[4] = {10.f, 20.f, 30.f, 40.f};
    Image img = {buf, 2, 2};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    ASSERT_NEAR(buf[0], 10.f, 1e-5f);
    ASSERT_NEAR(buf[3], 40.f, 1e-5f);
    return 0;
}

/* =========================================================================
 * calib_apply_cpu — dark subtraction
 * ========================================================================= */

static int test_apply_dark_only(void)
{
    /* light = 100, dark = 30 → expected = 70
     * Wrong implementation (no dark sub) would give 100. */
    float dark_buf[4] = {30.f, 30.f, 30.f, 30.f};
    CalibFrames c = make_calib_inline(dark_buf, 2, 2, NULL, 0, 0);

    float data[4] = {100.f, 100.f, 100.f, 100.f};
    Image img = {data, 2, 2};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(data[i], 70.f, 1e-4f);
    return 0;
}

static int test_apply_dark_varying(void)
{
    /* Verify per-pixel subtraction with different values */
    float dark_buf[4] = {10.f, 20.f, 30.f, 40.f};
    CalibFrames c = make_calib_inline(dark_buf, 2, 2, NULL, 0, 0);
    float data[4] = {50.f, 60.f, 70.f, 80.f};
    Image img = {data, 2, 2};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    ASSERT_NEAR(data[0], 40.f, 1e-4f);
    ASSERT_NEAR(data[1], 40.f, 1e-4f);
    ASSERT_NEAR(data[2], 40.f, 1e-4f);
    ASSERT_NEAR(data[3], 40.f, 1e-4f);
    return 0;
}

static int test_apply_negative_preserved(void)
{
    /* Dark subtraction can produce negatives; they must not be clamped.
     * Wrong implementation: clamp to 0 → would give 0 instead of -20. */
    float dark_buf[1] = {120.f};
    CalibFrames c = make_calib_inline(dark_buf, 1, 1, NULL, 0, 0);
    float data[1] = {100.f};
    Image img = {data, 1, 1};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    ASSERT_NEAR(data[0], -20.f, 1e-4f);
    return 0;
}

/* =========================================================================
 * calib_apply_cpu — flat division
 * ========================================================================= */

static int test_apply_flat_only(void)
{
    /* light = 60, flat = 2.0 → expected = 30
     * Wrong implementation (no flat div) would give 60. */
    float flat_buf[4] = {2.f, 2.f, 2.f, 2.f};
    CalibFrames c = make_calib_inline(NULL, 0, 0, flat_buf, 2, 2);
    float data[4] = {60.f, 60.f, 60.f, 60.f};
    Image img = {data, 2, 2};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(data[i], 30.f, 1e-4f);
    return 0;
}

static int test_apply_dead_pixel_guard(void)
{
    /* flat[0] = 0 (dead pixel) → output must be 0, not ±inf/NaN.
     * Wrong implementation (no guard) → division by 0 → inf/NaN. */
    float flat_buf[2] = {0.f, 2.f};
    CalibFrames c = make_calib_inline(NULL, 0, 0, flat_buf, 2, 1);
    float data[2] = {100.f, 100.f};
    Image img = {data, 2, 1};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    ASSERT_NEAR(data[0], 0.f, 1e-6f);    /* dead pixel → 0 */
    ASSERT_NEAR(data[1], 50.f, 1e-4f);   /* normal pixel */
    return 0;
}

static int test_apply_small_flat_guard(void)
{
    /* flat = 5e-7 < 1e-6 → treated as dead pixel */
    float flat_buf[1] = {5e-7f};
    CalibFrames c = make_calib_inline(NULL, 0, 0, flat_buf, 1, 1);
    float data[1] = {42.f};
    Image img = {data, 1, 1};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    ASSERT_NEAR(data[0], 0.f, 1e-6f);
    return 0;
}

static int test_apply_dark_and_flat(void)
{
    /* light=100, dark=20, flat=2 → (100-20)/2 = 40
     * Order matters: dark sub THEN flat div. */
    float dark_buf[1] = {20.f};
    float flat_buf[1] = {2.f};
    CalibFrames c = make_calib_inline(dark_buf, 1, 1, flat_buf, 1, 1);
    float data[1] = {100.f};
    Image img = {data, 1, 1};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    ASSERT_NEAR(data[0], 40.f, 1e-4f);
    return 0;
}

/* =========================================================================
 * calib_apply_cpu — dimension validation
 * ========================================================================= */

static int test_apply_dim_mismatch_dark(void)
{
    /* Dark master is 3×3 but image is 2×2 → DSO_ERR_INVALID_ARG
     * Wrong implementation: no check → accesses out-of-bounds memory. */
    float dark_buf[9] = {0};
    CalibFrames c = make_calib_inline(dark_buf, 3, 3, NULL, 0, 0);
    float data[4] = {1.f, 2.f, 3.f, 4.f};
    Image img = {data, 2, 2};
    ASSERT_ERR(calib_apply_cpu(&img, &c), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_apply_dim_mismatch_flat(void)
{
    /* Flat master is 1×1 but image is 2×2 */
    float flat_buf[1] = {1.f};
    CalibFrames c = make_calib_inline(NULL, 0, 0, flat_buf, 1, 1);
    float data[4] = {100.f, 100.f, 100.f, 100.f};
    Image img = {data, 2, 2};
    ASSERT_ERR(calib_apply_cpu(&img, &c), DSO_ERR_INVALID_ARG);
    return 0;
}

/* =========================================================================
 * calib_free
 * ========================================================================= */

static int test_free_null(void)
{
    calib_free(NULL);   /* must not crash */
    return 0;
}

static int test_free_empty(void)
{
    CalibFrames c = {0};
    calib_free(&c);     /* must not crash; data pointer is NULL */
    return 0;
}

static int test_free_clears_flags(void)
{
    float buf[1] = {1.f};
    CalibFrames c = make_calib_inline(NULL, 0, 0, NULL, 0, 0);
    /* Manually set a heap-allocated buffer to test the free path */
    c.dark.data   = (float *)malloc(sizeof(float));
    c.dark.width  = 1;
    c.dark.height = 1;
    c.has_dark    = 1;
    if (!c.dark.data) return 1;
    c.dark.data[0] = 99.f;
    (void)buf;

    calib_free(&c);
    ASSERT_EQ(c.has_dark, 0);
    ASSERT_EQ(c.has_flat, 0);
    return 0;
}

/* =========================================================================
 * calib_load_or_generate — FITS master loading
 *
 * When the path is already a FITS file, load it directly without stacking.
 * ========================================================================= */

static int test_load_fits_master_dark(void)
{
    /* Save a FITS file with constant value 50, load it as a master dark */
    char tmp[512]; TEST_TMPPATH(tmp, "dso_calib_test_master_dark.fits");
    ASSERT_OK(make_fits_const(tmp, 4, 4, 50.f));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(tmp, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_dark, 1);
    ASSERT_EQ(c.has_flat, 0);
    ASSERT_EQ(c.dark.width,  4);
    ASSERT_EQ(c.dark.height, 4);
    ASSERT_NEAR(c.dark.data[0], 50.f, 1e-4f);
    calib_free(&c);
    return 0;
}

static int test_load_fits_master_flat(void)
{
    char tmp[512]; TEST_TMPPATH(tmp, "dso_calib_test_master_flat.fits");
    ASSERT_OK(make_fits_const(tmp, 2, 2, 1.5f));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     tmp,  CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_dark, 0);
    ASSERT_EQ(c.has_flat, 1);
    ASSERT_NEAR(c.flat.data[0], 1.5f, 1e-4f);
    calib_free(&c);
    return 0;
}

/* =========================================================================
 * calib_load_or_generate — frame-list stacking (winsorized mean, median)
 * ========================================================================= */

static int test_generate_dark_constant_frames(void)
{
    /* 3 dark frames all with value 30 → master dark = 30 */
    char f1[512]; TEST_TMPPATH(f1, "dso_calib_dark1.fits");
    char f2[512]; TEST_TMPPATH(f2, "dso_calib_dark2.fits");
    char f3[512]; TEST_TMPPATH(f3, "dso_calib_dark3.fits");
    char lst[512]; TEST_TMPPATH(lst, "dso_calib_darklist.txt");
    ASSERT_OK(make_fits_const(f1, 2, 2, 30.f));
    ASSERT_OK(make_fits_const(f2, 2, 2, 30.f));
    ASSERT_OK(make_fits_const(f3, 2, 2, 30.f));
    const char *paths[] = {f1, f2, f3};
    ASSERT_OK(write_framelist(lst, paths, 3));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(lst, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_dark, 1);
    ASSERT_NEAR(c.dark.data[0], 30.f, 1e-3f);
    calib_free(&c);
    return 0;
}

static int test_winsorized_mean_clips_outlier(void)
{
    /* 10 dark frames: 9 with value 5.0, 1 with value 500.0 (outlier).
     * N=10, g=floor(0.1*10)=1.
     * Sorted: [5,5,5,5,5,5,5,5,5,500]
     * After clamping top 1: [5,5,5,5,5,5,5,5,5,5] → mean = 5.0
     * Plain mean (wrong): (9*5+500)/10 = 54.5
     * This test fails if winsorized mean is replaced by plain mean. */
    #define WSOR_N 10
    char paths[WSOR_N][64];
    const char *pptrs[WSOR_N];
    const int N = WSOR_N;
    for (int i = 0; i < N; i++) {
        snprintf(paths[i], sizeof(paths[i]),
                 "%s/dso_wsor_dark_%d.fits", test_tmpdir(), i);
        float val = (i == N - 1) ? 500.f : 5.f;
        ASSERT_OK(make_fits_const(paths[i], 1, 1, val));
        pptrs[i] = paths[i];
    }
    char lst[512]; TEST_TMPPATH(lst, "dso_wsor_dark_list.txt");
    ASSERT_OK(write_framelist(lst, pptrs, N));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(lst, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_dark, 1);
    /* Winsorized mean must be close to 5, not the plain-mean 54.5 */
    ASSERT_NEAR(c.dark.data[0], 5.f, 1.0f);   /* tol=1 to be safe */
    /* Verify it is far from the plain mean (which would indicate clipping worked) */
    ASSERT(fabsf(c.dark.data[0] - 54.5f) > 10.f);
    calib_free(&c);
    return 0;
}

static int test_median_method(void)
{
    /* 5 frames with values [1, 2, 3, 100, 200].
     * Median = 3.0.
     * Wrong (winsorized mean, g=0 for n=5): mean = 61.2 */
    float vals[] = {1.f, 2.f, 3.f, 100.f, 200.f};
    #define MED_N 5
    char paths[MED_N][64];
    const char *pptrs[MED_N];
    const int N = MED_N;
    for (int i = 0; i < N; i++) {
        snprintf(paths[i], sizeof(paths[i]),
                 "%s/dso_med_dark_%d.fits", test_tmpdir(), i);
        ASSERT_OK(make_fits_const(paths[i], 1, 1, vals[i]));
        pptrs[i] = paths[i];
    }
    char lst[512]; TEST_TMPPATH(lst, "dso_med_dark_list.txt");
    ASSERT_OK(write_framelist(lst, pptrs, N));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(lst, CALIB_MEDIAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_dark, 1);
    ASSERT_NEAR(c.dark.data[0], 3.f, 0.5f);
    /* Must not be the plain mean (61.2) */
    ASSERT(fabsf(c.dark.data[0] - 61.2f) > 10.f);
    calib_free(&c);
    return 0;
}

/* =========================================================================
 * calib_load_or_generate — bias subtracted from dark frames
 * ========================================================================= */

static int test_bias_subtracted_from_dark(void)
{
    /* bias master = 10, raw dark = 30 → dark_master = 20.
     * Wrong (no bias sub): dark_master = 30.
     * When applied to light=100: correct → 80, wrong → 70. */
    char bias_fits[512]; TEST_TMPPATH(bias_fits, "dso_calib_bias.fits");
    char dark_fits[512]; TEST_TMPPATH(dark_fits, "dso_calib_darkraw.fits");
    char dark_list[512]; TEST_TMPPATH(dark_list, "dso_calib_darklist2.txt");
    ASSERT_OK(make_fits_const(bias_fits, 2, 2, 10.f));
    ASSERT_OK(make_fits_const(dark_fits, 2, 2, 30.f));
    const char *dpaths[] = {dark_fits};
    ASSERT_OK(write_framelist(dark_list, dpaths, 1));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(dark_list,  CALIB_WINSORIZED_MEAN,
                                     bias_fits,   CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_dark, 1);
    /* dark_master should be dark - bias = 30 - 10 = 20 */
    ASSERT_NEAR(c.dark.data[0], 20.f, 1e-3f);
    /* Verify it is not 30 (which would mean bias was not subtracted) */
    ASSERT(fabsf(c.dark.data[0] - 30.f) > 5.f);

    /* Apply and verify end-to-end: light=100, dark_master=20 → 80 */
    float data[4] = {100.f, 100.f, 100.f, 100.f};
    Image img = {data, 2, 2};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    ASSERT_NEAR(data[0], 80.f, 1e-3f);

    calib_free(&c);
    return 0;
}

/* =========================================================================
 * calib_load_or_generate — flat per-frame normalization
 * ========================================================================= */

static int test_flat_per_frame_normalization(void)
{
    /* Two flat frames with uniform values 2000 and 4000.
     * After per-frame normalization each becomes 1.0 (mean=1.0).
     * Stacked master flat = 1.0.
     *
     * Wrong (no normalization): stack([2000, 4000]) = 3000.
     * In that case applying to a light frame changes the result drastically. */
    char f1[512]; TEST_TMPPATH(f1, "dso_flat_norm1.fits");
    char f2[512]; TEST_TMPPATH(f2, "dso_flat_norm2.fits");
    char flst[512]; TEST_TMPPATH(flst, "dso_flat_norm_list.txt");
    ASSERT_OK(make_fits_const(f1, 3, 3, 2000.f));
    ASSERT_OK(make_fits_const(f2, 3, 3, 4000.f));
    const char *fpaths[] = {f1, f2};
    ASSERT_OK(write_framelist(flst, fpaths, 2));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     flst, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_flat, 1);
    /* Master flat should be ≈ 1.0 (normalized frames stacked) */
    ASSERT_NEAR(c.flat.data[0], 1.f, 0.05f);
    /* Verify it is not the raw average 3000 */
    ASSERT(fabsf(c.flat.data[0] - 3000.f) > 100.f);
    calib_free(&c);
    return 0;
}

/* =========================================================================
 * calib_load_or_generate — bias subtracted from flat frames
 * ========================================================================= */

static int test_bias_subtracted_from_flat(void)
{
    /* bias = 100, flat_raw pixel = 1100.
     * flat_cal = flat_raw - bias = 1000; normalized = 1.0.
     *
     * Wrong (no bias sub): flat_raw mean = 1100; normalized also = 1.0
     * In this case both are 1.0, so the test needs to check an asymmetric case. */

    /* Use a 2-pixel image: pixel0 = 2100, pixel1 = 1100; bias = 100.
     * flat_cal[0] = 2000, flat_cal[1] = 1000.
     * mean(flat_cal) = 1500; normalized: flat[0] = 2000/1500 ≈ 1.333, flat[1] ≈ 0.667.
     *
     * Without bias sub: flat_raw mean = (2100+1100)/2 = 1600.
     * normalized: flat[0] = 2100/1600 ≈ 1.3125, flat[1] = 1100/1600 ≈ 0.6875.
     *
     * Not identical → can distinguish. */
    char bias_fits[512]; TEST_TMPPATH(bias_fits, "dso_flat_bias.fits");
    char flat_raw[512]; TEST_TMPPATH(flat_raw, "dso_flat_raw.fits");
    char flat_lst[512]; TEST_TMPPATH(flat_lst, "dso_flat_raw_list.txt");

    /* 2-pixel image (width=2, height=1) */
    float bias_pix[2] = {100.f, 100.f};
    float flat_pix[2] = {2100.f, 1100.f};
    ASSERT_OK(make_fits_row(bias_fits, bias_pix, 2));
    ASSERT_OK(make_fits_row(flat_raw, flat_pix, 2));
    const char *fpaths[] = {flat_raw};
    ASSERT_OK(write_framelist(flat_lst, fpaths, 1));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(NULL, CALIB_WINSORIZED_MEAN,
                                     bias_fits, CALIB_WINSORIZED_MEAN,
                                     flat_lst,  CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_flat, 1);
    /* Expected: (2000/1500) ≈ 1.333 */
    ASSERT_NEAR(c.flat.data[0], 2000.f / 1500.f, 0.01f);
    /* Expected: (1000/1500) ≈ 0.667 */
    ASSERT_NEAR(c.flat.data[1], 1000.f / 1500.f, 0.01f);
    calib_free(&c);
    return 0;
}

/* =========================================================================
 * calib_load_or_generate — darkflat subtracted from flat frames
 * ========================================================================= */

static int test_darkflat_subtracted_from_flat(void)
{
    /* darkflat = 50, flat_raw = 1050 → flat_cal = 1000; normalized = 1.0.
     * Wrong (no darkflat sub): flat_raw = 1050, normalized also ≈ 1.0.
     * Same issue as above; use asymmetric pixels to distinguish. */
    char df_fits[512]; TEST_TMPPATH(df_fits, "dso_darkflat.fits");
    char flat_raw[512]; TEST_TMPPATH(flat_raw, "dso_flat_raw2.fits");
    char flat_lst[512]; TEST_TMPPATH(flat_lst, "dso_flat_raw2_list.txt");

    /* 2-pixel: darkflat = [50, 50], flat_raw = [2050, 1050] */
    float df_pix[2]   = {50.f, 50.f};
    float flat_pix[2] = {2050.f, 1050.f};
    ASSERT_OK(make_fits_row(df_fits,  df_pix,   2));
    ASSERT_OK(make_fits_row(flat_raw, flat_pix, 2));
    const char *fpaths[] = {flat_raw};
    ASSERT_OK(write_framelist(flat_lst, fpaths, 1));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     flat_lst, CALIB_WINSORIZED_MEAN,
                                     df_fits,  CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));
    ASSERT_EQ(c.has_flat, 1);
    /* flat_cal = [2000, 1000], mean = 1500 → normalized [1.333, 0.667] */
    ASSERT_NEAR(c.flat.data[0], 2000.f / 1500.f, 0.01f);
    ASSERT_NEAR(c.flat.data[1], 1000.f / 1500.f, 0.01f);
    calib_free(&c);
    return 0;
}

/* =========================================================================
 * calib_load_or_generate — end-to-end calibration pipeline
 * ========================================================================= */

static int test_end_to_end_dark_and_flat(void)
{
    /* light = 2100, dark_raw = 100 (bias already applied externally for
     * simplicity here), flat_raw = 2.0 (pre-normalized, loaded as master).
     * Calibrated = (2100 - 100) / 2.0 = 1000. */
    char dark_fits[512]; TEST_TMPPATH(dark_fits, "dso_e2e_dark.fits");
    char flat_fits[512]; TEST_TMPPATH(flat_fits, "dso_e2e_flat.fits");
    ASSERT_OK(make_fits_const(dark_fits, 2, 2, 100.f));
    ASSERT_OK(make_fits_const(flat_fits, 2, 2, 2.f));

    CalibFrames c = {0};
    ASSERT_OK(calib_load_or_generate(dark_fits, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     flat_fits, CALIB_WINSORIZED_MEAN,
                                     NULL, CALIB_WINSORIZED_MEAN,
                                     NULL, 0.1f, &c));

    float data[4] = {2100.f, 2100.f, 2100.f, 2100.f};
    Image img = {data, 2, 2};
    ASSERT_OK(calib_apply_cpu(&img, &c));
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(data[i], 1000.f, 0.1f);

    calib_free(&c);
    return 0;
}

static int test_end_to_end_dead_pixels_zeroed(void)
{
    /* Flat has one dead pixel (value 0.0) → that pixel must become 0 in output.
     * All other pixels pass normally. */
    char flat_fits[512]; TEST_TMPPATH(flat_fits, "dso_e2e_deadflat.fits");
    float flat_pix[4] = {0.f, 2.f, 2.f, 2.f};
    ASSERT_OK(make_fits_row(flat_fits, flat_pix, 4));
    /* Use width=4, height=1 for simplicity */

    CalibFrames c = {0};
    c.flat.data   = flat_pix;
    c.flat.width  = 4;
    c.flat.height = 1;
    c.has_flat    = 1;

    float data[4] = {100.f, 100.f, 100.f, 100.f};
    Image img = {data, 4, 1};
    ASSERT_OK(calib_apply_cpu(&img, &c));

    ASSERT_NEAR(data[0], 0.f, 1e-5f);   /* dead pixel */
    ASSERT_NEAR(data[1], 50.f, 1e-4f);
    ASSERT_NEAR(data[2], 50.f, 1e-4f);
    ASSERT_NEAR(data[3], 50.f, 1e-4f);
    return 0;
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    SUITE("calib_apply_cpu — argument validation");
    RUN(test_apply_null_img);
    RUN(test_apply_null_calib);
    RUN(test_apply_no_calib_is_noop);

    SUITE("calib_apply_cpu — dark subtraction");
    RUN(test_apply_dark_only);
    RUN(test_apply_dark_varying);
    RUN(test_apply_negative_preserved);

    SUITE("calib_apply_cpu — flat division");
    RUN(test_apply_flat_only);
    RUN(test_apply_dead_pixel_guard);
    RUN(test_apply_small_flat_guard);
    RUN(test_apply_dark_and_flat);

    SUITE("calib_apply_cpu — dimension validation");
    RUN(test_apply_dim_mismatch_dark);
    RUN(test_apply_dim_mismatch_flat);

    SUITE("calib_free");
    RUN(test_free_null);
    RUN(test_free_empty);
    RUN(test_free_clears_flags);

    SUITE("calib_load_or_generate — FITS master loading");
    RUN(test_load_fits_master_dark);
    RUN(test_load_fits_master_flat);

    SUITE("calib_load_or_generate — frame-list stacking");
    RUN(test_generate_dark_constant_frames);
    RUN(test_winsorized_mean_clips_outlier);
    RUN(test_median_method);

    SUITE("calib_load_or_generate — bias/darkflat subtraction");
    RUN(test_bias_subtracted_from_dark);
    RUN(test_flat_per_frame_normalization);
    RUN(test_bias_subtracted_from_flat);
    RUN(test_darkflat_subtracted_from_flat);

    SUITE("calib_load_or_generate — end-to-end");
    RUN(test_end_to_end_dark_and_flat);
    RUN(test_end_to_end_dead_pixels_zeroed);

    return SUMMARY();
}
