/*
 * test_cpu.c — unit tests for the DSO stacker CPU-side library
 *
 * Covers: csv_parser, fits_io, integration (mean + kappa-sigma), lanczos_cpu.
 *
 * Design principle: every test is chosen so that a plausible but incorrect
 * implementation (wrong loop bounds, wrong weight formula, missing rejection
 * pass, wrong FITS pixel ordering, …) will produce a measurably different
 * result and therefore fail.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_framework.h"
#include "dso_types.h"
#include "csv_parser.h"
#include "fits_io.h"
#include "integration.h"
#include "lanczos_cpu.h"

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

/* Allocate an Image with all pixels set to `val`. */
static Image make_image_const(int w, int h, float val)
{
    Image img = { NULL, w, h };
    img.data = (float *)malloc((size_t)w * h * sizeof(float));
    if (!img.data) { fprintf(stderr, "OOM in make_image_const\n"); exit(1); }
    for (int i = 0; i < w * h; i++) img.data[i] = val;
    return img;
}

/* Allocate an Image where pixel (x,y) = (float)(y*w + x). */
static Image make_image_gradient(int w, int h)
{
    Image img = { NULL, w, h };
    img.data = (float *)malloc((size_t)w * h * sizeof(float));
    if (!img.data) { fprintf(stderr, "OOM in make_image_gradient\n"); exit(1); }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img.data[y * w + x] = (float)(y * w + x);
    return img;
}

/* Write a minimal 2-column frame-list CSV to `path`. */
static void write_test_csv(const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) { perror("write_test_csv"); exit(1); }
    fprintf(fp, "filepath, is_reference\n");
    fprintf(fp, "%s/dso_frame1.fits, 1\n", test_tmpdir());
    fprintf(fp, "%s/dso_frame2.fits, 0\n", test_tmpdir());
    fclose(fp);
}

/* =========================================================================
 * CSV PARSER TESTS
 * ====================================================================== */

/* Parsing a valid 2-row 2-col CSV must return exactly 2 frames. */
static int test_csv_count(void)
{
    char csv_path[512]; TEST_TMPPATH(csv_path, "dso_test.csv");
    write_test_csv(csv_path);
    FrameInfo *f = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv_path, &f, &n));
    ASSERT_EQ(n, 2);
    ASSERT_NOT_NULL(f);
    free(f);
    return 0;
}

/* The first row (header) must be skipped; filepaths come from data rows. */
static int test_csv_filepath(void)
{
    char csv_path[512]; TEST_TMPPATH(csv_path, "dso_test.csv");
    char exp1[512], exp2[512];
    snprintf(exp1, sizeof(exp1), "%s/dso_frame1.fits", test_tmpdir());
    snprintf(exp2, sizeof(exp2), "%s/dso_frame2.fits", test_tmpdir());
    write_test_csv(csv_path);
    FrameInfo *f = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv_path, &f, &n));
    ASSERT_EQ(strcmp(f[0].filepath, exp1), 0);
    ASSERT_EQ(strcmp(f[1].filepath, exp2), 0);
    free(f);
    return 0;
}

/* is_reference field must be parsed as an integer (1 or 0). */
static int test_csv_reference_flag(void)
{
    char csv_path[512]; TEST_TMPPATH(csv_path, "dso_test.csv");
    write_test_csv(csv_path);
    FrameInfo *f = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv_path, &f, &n));
    ASSERT_EQ(f[0].is_reference, 1);
    ASSERT_EQ(f[1].is_reference, 0);
    free(f);
    return 0;
}

/* Blank lines in the CSV body must be silently skipped. */
static int test_csv_blank_lines_skipped(void)
{
    char path[512]; TEST_TMPPATH(path, "dso_test_blank.csv");
    FILE *fp = fopen(path, "w");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "filepath, is_reference\n");
    fprintf(fp, "\n");
    fprintf(fp, "%s/f.fits, 1\n", test_tmpdir());
    fprintf(fp, "\n");
    fclose(fp);

    FrameInfo *f = NULL; int n = 0;
    ASSERT_OK(csv_parse(path, &f, &n));
    ASSERT_EQ(n, 1);   /* only the data row, not the blank lines */
    free(f);
    return 0;
}

/* Opening a nonexistent file must return an I/O error. */
static int test_csv_nonexistent_file(void)
{
    FrameInfo *f = NULL; int n = 0;
    char nofile[512]; TEST_TMPPATH(nofile, "does_not_exist_xyz.csv");
    DsoError err = csv_parse(nofile, &f, &n);
    ASSERT_EQ(err, DSO_ERR_IO);
    return 0;
}

/*
 * 2-column CSV: filepaths and is_reference flags must be correct; homography
 * fields must be zero-initialised (they are populated later by RANSAC).
 */
static int test_csv_2col_basic(void)
{
    char csv2[512]; TEST_TMPPATH(csv2, "dso_test_2col.csv");
    char exp1[512]; snprintf(exp1, sizeof(exp1), "%s/dso_frame1.fits", test_tmpdir());
    write_test_csv(csv2);
    FrameInfo *f = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv2, &f, &n));
    ASSERT_EQ(n, 2);
    /* Filepaths must be correct */
    ASSERT_EQ(strcmp(f[0].filepath, exp1), 0);
    ASSERT_EQ(f[0].is_reference, 1);
    ASSERT_EQ(f[1].is_reference, 0);
    /* Homography must be zero-initialised */
    for (int i = 0; i < 9; i++) ASSERT_NEAR(f[0].H.h[i], 0.0, 1e-12);
    for (int i = 0; i < 9; i++) ASSERT_NEAR(f[1].H.h[i], 0.0, 1e-12);
    free(f);
    return 0;
}

/*
 * A CSV with a column count other than 2 must return DSO_ERR_CSV.
 * Prevents silent misparse of files in the wrong format.
 */
static int test_csv_bad_col_count(void)
{
    char path[512]; TEST_TMPPATH(path, "dso_test_badcols.csv");
    FILE *fp = fopen(path, "w");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "filepath, is_reference, foo, bar, baz\n");
    fprintf(fp, "%s/f.fits, 1, 1.0, 2.0, 3.0\n", test_tmpdir());
    fclose(fp);

    FrameInfo *f = NULL; int n = 0;
    ASSERT_ERR(csv_parse(path, &f, &n), DSO_ERR_CSV);
    return 0;
}

/* =========================================================================
 * FITS I/O TESTS
 * ====================================================================== */

/*
 * Round-trip: save a known 4×3 float image to disk, reload it, compare
 * every pixel. Detects wrong BITPIX, transposed axes, scale errors, etc.
 */
static int test_fits_roundtrip(void)
{
    const int W = 4, H = 3;
    Image src = make_image_gradient(W, H);

    char rpath[512]; TEST_TMPPATH(rpath, "dso_roundtrip.fits");
    ASSERT_OK(fits_save(rpath, &src));

    Image dst = {NULL, 0, 0};
    ASSERT_OK(fits_load(rpath, &dst));

    ASSERT_EQ(dst.width,  W);
    ASSERT_EQ(dst.height, H);
    for (int i = 0; i < W * H; i++) {
        ASSERT_NEAR(dst.data[i], src.data[i], 1e-5f);
    }

    image_free(&src);
    image_free(&dst);
    return 0;
}

/*
 * Pixel ordering: a known value at (x=2, y=1) must survive the round-trip
 * at exactly that position (not transposed or shifted).
 */
static int test_fits_pixel_ordering(void)
{
    const int W = 8, H = 6;
    Image src = make_image_const(W, H, 0.f);
    /* stamp a unique value at (x=2, y=1) */
    src.data[1 * W + 2] = 42.f;

    char opath[512]; TEST_TMPPATH(opath, "dso_ordering.fits");
    ASSERT_OK(fits_save(opath, &src));

    Image dst = {NULL, 0, 0};
    ASSERT_OK(fits_load(opath, &dst));

    ASSERT_NEAR(dst.data[1 * W + 2], 42.f, 1e-5f);
    /* neighbouring pixel must be 0 (not the same value) */
    ASSERT_NEAR(dst.data[1 * W + 3], 0.f, 1e-5f);

    image_free(&src);
    image_free(&dst);
    return 0;
}

/* Loading a nonexistent file must return DSO_ERR_FITS. */
static int test_fits_load_missing(void)
{
    Image img = {NULL, 0, 0};
    char mpath[512]; TEST_TMPPATH(mpath, "no_such_file_abc.fits");
    ASSERT_ERR(fits_load(mpath, &img), DSO_ERR_FITS);
    return 0;
}

/* image_free must null the data pointer so double-free is safe. */
static int test_image_free_nulls_ptr(void)
{
    Image img = make_image_const(4, 4, 1.f);
    ASSERT_NOT_NULL(img.data);
    image_free(&img);
    ASSERT_NULL(img.data);
    return 0;
}

/* image_free on a zero-initialised Image must not crash. */
static int test_image_free_on_null_data(void)
{
    Image img = {NULL, 0, 0};
    image_free(&img);   /* must not crash */
    return 0;
}

/* =========================================================================
 * INTEGRATION TESTS — mean
 * ====================================================================== */

/* Mean of N identical constant images must equal that constant. */
static int test_integrate_mean_constant(void)
{
    const int N = 5, W = 6, H = 4;
    Image imgs[5];
    const Image *ptrs[5];
    for (int i = 0; i < N; i++) { imgs[i] = make_image_const(W, H, 7.5f); ptrs[i] = &imgs[i]; }

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, N, &out));
    for (int i = 0; i < W * H; i++) ASSERT_NEAR(out.data[i], 7.5f, 1e-5f);

    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/*
 * Mean of three single-pixel images with values 1, 2, 3 must be exactly 2.
 * This fails if the accumulation loop is wrong (e.g. off-by-one in N).
 */
static int test_integrate_mean_arithmetic(void)
{
    float vals[] = { 1.f, 2.f, 3.f };
    Image imgs[3];
    const Image *ptrs[3];
    for (int i = 0; i < 3; i++) { imgs[i] = make_image_const(1, 1, vals[i]); ptrs[i] = &imgs[i]; }

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 3, &out));
    ASSERT_NEAR(out.data[0], 2.f, 1e-5f);

    for (int i = 0; i < 3; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/* Mismatched image dimensions must return DSO_ERR_INVALID_ARG. */
static int test_integrate_mean_size_mismatch(void)
{
    Image a = make_image_const(4, 4, 1.f);
    Image b = make_image_const(5, 4, 1.f);
    const Image *ptrs[] = { &a, &b };
    Image out = {NULL, 0, 0};
    ASSERT_ERR(integrate_mean(ptrs, 2, &out), DSO_ERR_INVALID_ARG);
    image_free(&a);
    image_free(&b);
    return 0;
}

/* NULL frames pointer must return DSO_ERR_INVALID_ARG. */
static int test_integrate_mean_null_frames(void)
{
    Image out = {NULL, 0, 0};
    ASSERT_ERR(integrate_mean(NULL, 3, &out), DSO_ERR_INVALID_ARG);
    return 0;
}

/* =========================================================================
 * INTEGRATION TESTS — kappa-sigma clipping
 * ====================================================================== */

/*
 * Outlier rejection: 4 frames with pixel value 1.0 and 1 frame with 100.0.
 * With kappa=1.5 the outlier must be rejected.
 *
 * Mathematical verification:
 *   n=5, vals=[1,1,1,1,100], mean=104/5=20.8
 *   sq_sum = 4*(1-20.8)^2 + (100-20.8)^2 = 4*392.04 + 6280.64 = 7848.80
 *   stddev = sqrt(7848.80 / 4) ≈ 44.3
 *   threshold = 1.5 * 44.3 ≈ 66.4
 *   |100 - 20.8| = 79.2 > 66.4 → outlier rejected
 *   survivors: mean(1,1,1,1) = 1.0
 */
static int test_kappa_sigma_removes_outlier(void)
{
    const int N = 5;
    Image imgs[5];
    const Image *ptrs[5];
    for (int i = 0; i < 4; i++) { imgs[i] = make_image_const(1, 1, 1.f);   ptrs[i] = &imgs[i]; }
    imgs[4] = make_image_const(1, 1, 100.f);
    ptrs[4] = &imgs[4];

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, N, &out, 1.5f, 5));
    /* Output must be close to 1.0, not influenced by the 100.0 outlier */
    ASSERT_NEAR(out.data[0], 1.f, 1e-4f);

    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/* Uniform data (no outliers) must produce the plain mean. */
static int test_kappa_sigma_uniform(void)
{
    const int N = 6, W = 3, H = 3;
    Image imgs[6];
    const Image *ptrs[6];
    for (int i = 0; i < N; i++) { imgs[i] = make_image_const(W, H, 5.f); ptrs[i] = &imgs[i]; }

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, N, &out, 3.f, 5));
    for (int i = 0; i < W * H; i++) ASSERT_NEAR(out.data[i], 5.f, 1e-4f);

    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/*
 * Degenerate fallback: with a very tight kappa all values get rejected;
 * the implementation must fall back to the unclipped mean instead of
 * producing NaN, 0, or undefined behaviour.
 *
 * Setup: 2 frames, values 1.0 and 3.0, kappa=0.1
 *   mean=2, stddev=sqrt(2)≈1.414, threshold=0.141
 *   |1-2|=1>0.141 → rejected; |3-2|=1>0.141 → rejected; n_active=0
 *   fallback mean = (1+3)/2 = 2.0
 */
static int test_kappa_sigma_all_clipped_fallback(void)
{
    Image a = make_image_const(1, 1, 1.f);
    Image b = make_image_const(1, 1, 3.f);
    const Image *ptrs[] = { &a, &b };

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, 2, &out, 0.1f, 1));
    ASSERT_NEAR(out.data[0], 2.f, 1e-4f);   /* fallback = (1+3)/2 */

    image_free(&a);
    image_free(&b);
    image_free(&out);
    return 0;
}

/* kappa-sigma on a multi-pixel image must process every pixel independently. */
static int test_kappa_sigma_multi_pixel(void)
{
    /*
     * 5 frames, 1x2 image:
     *   pixel 0: [1, 1, 1, 1, 100] → outlier=100 rejected (same math as
     *                                 test_kappa_sigma_removes_outlier) → mean=1
     *   pixel 1: [5, 5, 5, 5, 5]   → no outlier                         → mean=5
     *
     * With n=3 the outlier inflates stddev enough that kappa=1.5 does NOT reject it
     * (|100-34|=66 < 1.5*57.2=85.7).  n=5 gives a tighter distribution.
     */
    const int N = 5;
    Image imgs[5];
    const Image *ptrs[5];
    for (int i = 0; i < N; i++) {
        imgs[i].width  = 2;
        imgs[i].height = 1;
        imgs[i].data   = (float *)malloc(2 * sizeof(float));
    }
    imgs[0].data[0] = 1.f;   imgs[0].data[1] = 5.f;
    imgs[1].data[0] = 1.f;   imgs[1].data[1] = 5.f;
    imgs[2].data[0] = 1.f;   imgs[2].data[1] = 5.f;
    imgs[3].data[0] = 1.f;   imgs[3].data[1] = 5.f;
    imgs[4].data[0] = 100.f; imgs[4].data[1] = 5.f;
    for (int i = 0; i < N; i++) ptrs[i] = &imgs[i];

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, N, &out, 1.5f, 5));
    ASSERT_NEAR(out.data[0], 1.f, 1e-4f);
    ASSERT_NEAR(out.data[1], 5.f, 1e-4f);

    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/* =========================================================================
 * LANCZOS CPU TESTS
 * ====================================================================== */

/*
 * Identity homography must map every interior destination pixel to the exact
 * same source pixel value.
 *
 * This relies on the Lanczos sinc property: sinc(n)=0 for non-zero integer n,
 * so at an integer source coordinate only the centre tap has non-zero weight.
 * Any error in the weight formula or backward-mapping math will break this.
 */
static int test_lanczos_cpu_identity(void)
{
    const int W = 24, H = 24;
    Image src = make_image_gradient(W, H);
    Image dst = make_image_const(W, H, 0.f);

    /* Identity: hom = diag(1,1,1) */
    Homography hom = {{ 1,0,0, 0,1,0, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));

    /*
     * Interior pixels: [4..19, 4..19].
     * Offset 4 ensures all 6 taps (offsets -2..3) are within [0,23].
     */
    for (int y = 4; y <= 19; y++) {
        for (int x = 4; x <= 19; x++) {
            float expected = src.data[y * W + x];
            float actual   = dst.data[y * W + x];
            if (fabsf(actual - expected) > 1e-4f) {
                printf("\n    pixel (%d,%d): expected %.4f, got %.4f\n",
                       x, y, expected, actual);
                image_free(&src); image_free(&dst);
                return 1;
            }
        }
    }

    image_free(&src);
    image_free(&dst);
    return 0;
}

/*
 * Integer-pixel translation: backward map shifts dst lookup by (−4, −3).
 *
 * H = [[1,0,-4],[0,1,-3],[0,0,1]]  (backward map: ref → src)
 *
 * For dst pixel (dx,dy): sx = dx-4, sy = dy-3
 * Interior check range: dx in [7..19], dy in [6..19]
 *   sx in [3..15], sy in [3..16] — all taps within [0..23].
 *
 * Expected: dst[dy][dx] == src[dy-3][dx-4]
 */
static int test_lanczos_cpu_integer_shift(void)
{
    const int W = 24, H = 24;
    Image src = make_image_gradient(W, H);
    Image dst = make_image_const(W, H, 0.f);

    /* Backward map: dst(dx,dy) samples src at (dx-4, dy-3) */
    Homography hom = {{ 1,0,-4, 0,1,-3, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));

    for (int dy = 6; dy <= 19; dy++) {
        for (int dx = 7; dx <= 19; dx++) {
            float expected = src.data[(dy - 3) * W + (dx - 4)];
            float actual   = dst.data[dy * W + dx];
            if (fabsf(actual - expected) > 1e-4f) {
                printf("\n    pixel (%d,%d): expected %.4f, got %.4f\n",
                       dx, dy, expected, actual);
                image_free(&src); image_free(&dst);
                return 1;
            }
        }
    }

    image_free(&src);
    image_free(&dst);
    return 0;
}

/*
 * Pixels that map outside the source bounds must be written as NAN.
 * A large translation moves all source content out of the destination view.
 * NAN sentinel allows integration to skip OOB pixels from warped frames.
 */
static int test_lanczos_cpu_oob_pixels_zero(void)
{
    const int W = 10, H = 10;
    Image src = make_image_const(W, H, 99.f);
    Image dst = make_image_const(W, H, -1.f);  /* sentinel: non-NAN */

    /* Backward map with +1000 offset: src lookup at (dx+1000, dy+1000) — always OOB */
    Homography hom = {{ 1,0,1000, 0,1,1000, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));

    /* Every destination pixel should be NAN (OOB sentinel) */
    for (int i = 0; i < W * H; i++) {
        if (!isnan(dst.data[i])) {
            printf("\n    dst[%d] = %f, expected NAN\n", i, dst.data[i]);
            image_free(&src); image_free(&dst);
            return 1;
        }
    }

    image_free(&src);
    image_free(&dst);
    return 0;
}

/*
 * Degenerate H (sw = 0 everywhere): all destination pixels map to OOB
 * and must be written as NAN. No error is returned — degenerate pixels
 * get the NAN sentinel. (H is the backward map used directly; there is
 * no inversion step that would detect a singular matrix.)
 */
static int test_lanczos_cpu_singular_h(void)
{
    Image src = make_image_const(8, 8, 1.f);
    Image dst = make_image_const(8, 8, -1.f);  /* sentinel */

    /* All-zero matrix: sw = 0 for every pixel → OOB → written as NAN */
    Homography H = {{ 0,0,0, 0,0,0, 0,0,0 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H));
    for (int i = 0; i < 8 * 8; i++) {
        if (!isnan(dst.data[i])) {
            printf("\n    dst[%d] = %f, expected NAN\n", i, dst.data[i]);
            image_free(&src); image_free(&dst);
            return 1;
        }
    }

    image_free(&src);
    image_free(&dst);
    return 0;
}

/* NULL arguments must return DSO_ERR_INVALID_ARG without crashing. */
static int test_lanczos_cpu_null_args(void)
{
    Image img = make_image_const(4, 4, 1.f);
    Homography H = {{ 1,0,0, 0,1,0, 0,0,1 }};

    ASSERT_ERR(lanczos_transform_cpu(NULL, &img, &H),  DSO_ERR_INVALID_ARG);
    ASSERT_ERR(lanczos_transform_cpu(&img, NULL, &H),  DSO_ERR_INVALID_ARG);
    ASSERT_ERR(lanczos_transform_cpu(&img, &img, NULL), DSO_ERR_INVALID_ARG);

    image_free(&img);
    return 0;
}

/*
 * Sub-pixel interpolation sanity: transform a flat-ramp image by a non-integer
 * shift (0.5 pixels in x). The interpolated value must lie between the two
 * neighbouring source pixels, not equal to either.
 *
 * H = [[1,0,-0.5],[0,1,0],[0,0,1]]  (backward map: dst(x,y) samples src at (x-0.5, y))
 * For dst pixel (10, 10): sx=9.5 — interpolated between src[10*W+9] and [10*W+10].
 */
static int test_lanczos_cpu_subpixel_range(void)
{
    const int W = 30, H = 20;
    Image src = make_image_gradient(W, H);  /* pixel value = index */
    Image dst = make_image_const(W, H, 0.f);

    Homography hom = {{ 1,0,-0.5, 0,1,0, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));

    /* dst(10,10) samples src at sx=9.5, sy=10 (interior pixel) */
    int x = 10, y = 10;
    float val   = dst.data[y * W + x];
    float left  = src.data[y * W + (x - 1)];  /* src at sx=9 */
    float right = src.data[y * W + x];         /* src at sx=10 */

    /* Interpolated value must be strictly between left and right */
    ASSERT_GT(val, left);
    ASSERT_LT(val, right);

    image_free(&src);
    image_free(&dst);
    return 0;
}

/* =========================================================================
 * INTEGRATION — NaN handling
 * ====================================================================== */

/* NaN pixels excluded from mean (valid frames only contribute). */
static int test_integrate_mean_nan_skipped(void)
{
    const int W = 4, H = 3;
    Image f1 = make_image_const(W, H, 10.0f);
    Image f2 = make_image_const(W, H, 20.0f);
    Image f3 = make_image_const(W, H, NAN);   /* entire frame NaN */

    const Image *ptrs[3] = {&f1, &f2, &f3};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 3, &out));
    /* Mean of [10, 20] (NaN skipped) = 15 */
    ASSERT_NEAR(out.data[0], 15.0f, 0.01f);

    image_free(&f1); image_free(&f2); image_free(&f3); image_free(&out);
    return 0;
}

/* All frames NaN at a pixel → output NaN. */
static int test_integrate_mean_all_nan_pixel(void)
{
    const int W = 4, H = 3;
    Image f1 = make_image_const(W, H, 5.0f);
    Image f2 = make_image_const(W, H, 5.0f);
    f1.data[0] = NAN;
    f2.data[0] = NAN;

    const Image *ptrs[2] = {&f1, &f2};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 2, &out));
    ASSERT(isnan(out.data[0]));
    ASSERT_NEAR(out.data[1], 5.0f, 0.01f);

    image_free(&f1); image_free(&f2); image_free(&out);
    return 0;
}

/* Single frame → output = input. */
static int test_integrate_mean_single_frame(void)
{
    const int W = 4, H = 3;
    Image f1 = make_image_const(W, H, 42.0f);
    f1.data[0] = 99.0f;

    const Image *ptrs[1] = {&f1};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 1, &out));
    ASSERT_NEAR(out.data[0], 99.0f, 0.01f);
    ASSERT_NEAR(out.data[1], 42.0f, 0.01f);

    image_free(&f1); image_free(&out);
    return 0;
}

/* NaN frames excluded from kappa-sigma. */
static int test_kappa_sigma_nan_handling(void)
{
    const int W = 4, H = 3;
    Image f1 = make_image_const(W, H, 10.0f);
    Image f2 = make_image_const(W, H, 10.0f);
    Image f3 = make_image_const(W, H, NAN);

    const Image *ptrs[3] = {&f1, &f2, &f3};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, 3, &out, 3.0f, 3));
    ASSERT_NEAR(out.data[0], 10.0f, 0.01f);

    image_free(&f1); image_free(&f2); image_free(&f3); image_free(&out);
    return 0;
}

/* n=1 kappa-sigma → no clipping, output = input. */
static int test_kappa_sigma_single_frame(void)
{
    const int W = 4, H = 3;
    Image f1 = make_image_const(W, H, 77.0f);

    const Image *ptrs[1] = {&f1};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, 1, &out, 3.0f, 3));
    ASSERT_NEAR(out.data[0], 77.0f, 0.01f);

    image_free(&f1); image_free(&out);
    return 0;
}

/* n=2 kappa-sigma. */
static int test_kappa_sigma_two_frames(void)
{
    const int W = 4, H = 3;
    Image f1 = make_image_const(W, H, 10.0f);
    Image f2 = make_image_const(W, H, 20.0f);

    const Image *ptrs[2] = {&f1, &f2};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, 2, &out, 3.0f, 3));
    /* With only 2 values, kappa=3 is very generous — both survive */
    ASSERT_NEAR(out.data[0], 15.0f, 0.1f);

    image_free(&f1); image_free(&f2); image_free(&out);
    return 0;
}

/* =========================================================================
 * LANCZOS CPU — additional edge cases
 * ====================================================================== */

/* Minimum viable 2×2 image. */
static int test_lanczos_cpu_small_image_2x2(void)
{
    Image src = make_image_const(2, 2, 50.0f);
    Image dst = make_image_const(2, 2, 0.0f);
    Homography hom = {{ 1,0,0, 0,1,0, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));
    /* With identity H, interior (or all for 2×2) should be near 50 */
    /* Small image — boundary effects dominate, just check no crash */
    image_free(&src); image_free(&dst);
    return 0;
}

/* Shift larger than image → all NaN output. */
static int test_lanczos_cpu_large_shift_all_oob(void)
{
    const int W = 8, H = 8;
    Image src = make_image_const(W, H, 100.0f);
    Image dst = make_image_const(W, H, 0.0f);
    /* Shift by 1000 pixels — everything OOB */
    Homography hom = {{ 1,0,1000, 0,1,1000, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));
    /* All output should be NaN (OOB sentinel) */
    for (int i = 0; i < W * H; i++)
        ASSERT(isnan(dst.data[i]));
    image_free(&src); image_free(&dst);
    return 0;
}

/* 0.25px shift — verify interpolation accuracy. */
static int test_lanczos_cpu_fractional_shift_accuracy(void)
{
    const int W = 30, H = 20;
    Image src = make_image_gradient(W, H);
    Image dst = make_image_const(W, H, 0.0f);
    /* H = [[1,0,-0.25],[0,1,0],[0,0,1]]: dst(x,y) samples src(x-0.25, y) */
    Homography hom = {{ 1,0,-0.25, 0,1,0, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));

    /* Interior pixel (10,10): src(9.75, 10) ≈ 0.75*src[10*W+9] + 0.25*src[10*W+10] */
    float expected_approx = 0.75f * src.data[10*W+9] + 0.25f * src.data[10*W+10];
    float actual = dst.data[10*W+10];
    ASSERT_NEAR(actual, expected_approx, 1.0f); /* Lanczos not linear interp, allow tolerance */

    image_free(&src); image_free(&dst);
    return 0;
}

/* =========================================================================
 * AAWA (Auto Adaptive Weighted Average)
 * ====================================================================== */

/* Uniform frames → output equals the constant. */
static int test_integrate_aawa_uniform(void)
{
    const int W = 4, H = 3, N = 5;
    Image imgs[5];
    const Image *ptrs[5];
    for (int i = 0; i < N; i++) {
        imgs[i] = make_image_const(W, H, 42.0f);
        ptrs[i] = &imgs[i];
    }
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_aawa(ptrs, N, &out));
    ASSERT_EQ(out.width, W);
    ASSERT_EQ(out.height, H);
    for (int p = 0; p < W * H; p++)
        ASSERT_NEAR(out.data[p], 42.0f, 1e-4f);
    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/* 9 frames at 100, 1 at 10000 → AAWA should strongly down-weight the outlier. */
static int test_integrate_aawa_outlier_downweight(void)
{
    const int W = 2, H = 2, N = 10;
    Image imgs[10];
    const Image *ptrs[10];
    for (int i = 0; i < 9; i++) {
        imgs[i] = make_image_const(W, H, 100.0f);
        ptrs[i] = &imgs[i];
    }
    imgs[9] = make_image_const(W, H, 10000.0f);
    ptrs[9] = &imgs[9];

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_aawa(ptrs, N, &out));
    /* Simple mean would give 1090; AAWA should be much closer to 100 */
    for (int p = 0; p < W * H; p++) {
        ASSERT(out.data[p] < 150.0f);
        ASSERT(out.data[p] > 50.0f);
    }
    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/* NaN values should be skipped. */
static int test_integrate_aawa_nan_handling(void)
{
    const int W = 4, H = 3, N = 5;
    Image imgs[5];
    const Image *ptrs[5];
    for (int i = 0; i < N; i++) {
        imgs[i] = make_image_const(W, H, 50.0f);
        ptrs[i] = &imgs[i];
    }
    /* Mark frames 3 and 4 as NaN for all pixels */
    for (int p = 0; p < W * H; p++) {
        imgs[3].data[p] = NAN;
        imgs[4].data[p] = NAN;
    }
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_aawa(ptrs, N, &out));
    for (int p = 0; p < W * H; p++)
        ASSERT_NEAR(out.data[p], 50.0f, 1e-4f);
    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out);
    return 0;
}

/* Two frames → should produce result close to mean. */
static int test_integrate_aawa_two_frames(void)
{
    const int W = 4, H = 3;
    Image f1 = make_image_const(W, H, 10.0f);
    Image f2 = make_image_const(W, H, 20.0f);
    const Image *ptrs[2] = {&f1, &f2};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_aawa(ptrs, 2, &out));
    for (int p = 0; p < W * H; p++)
        ASSERT_NEAR(out.data[p], 15.0f, 0.5f);
    image_free(&f1); image_free(&f2); image_free(&out);
    return 0;
}

/* Deterministic: running twice produces the same result. */
static int test_integrate_aawa_convergence(void)
{
    const int W = 4, H = 3, N = 8;
    Image imgs[8];
    const Image *ptrs[8];
    /* Create frames with varied values */
    for (int i = 0; i < N; i++) {
        imgs[i] = make_image_const(W, H, 100.0f + (float)(i * 5));
        ptrs[i] = &imgs[i];
    }
    imgs[N - 1] = make_image_const(W, H, 5000.0f); /* outlier */
    ptrs[N - 1] = &imgs[N - 1];

    Image out1 = {NULL, 0, 0};
    Image out2 = {NULL, 0, 0};
    ASSERT_OK(integrate_aawa(ptrs, N, &out1));
    ASSERT_OK(integrate_aawa(ptrs, N, &out2));
    for (int p = 0; p < W * H; p++)
        ASSERT_NEAR(out1.data[p], out2.data[p], 1e-6f);
    for (int i = 0; i < N; i++) image_free(&imgs[i]);
    image_free(&out1); image_free(&out2);
    return 0;
}

/* =========================================================================
 * CSV — additional edge cases
 * ====================================================================== */

/* CSV with path containing spaces. */
static int test_csv_whitespace_in_path(void)
{
    char csv_path[512]; TEST_TMPPATH(csv_path, "dso_ws_test.csv");
    FILE *fp = fopen(csv_path, "w");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "filepath, is_reference\n");
    fprintf(fp, "%s/path with spaces/frame.fits, 1\n", test_tmpdir());
    fclose(fp);

    FrameInfo *f = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv_path, &f, &n));
    ASSERT_EQ(n, 1);
    /* Path should contain the space */
    ASSERT(strstr(f[0].filepath, "path with spaces") != NULL);
    free(f);
    return 0;
}

/* Multiple is_reference=1 rows. */
static int test_csv_multiple_references(void)
{
    char csv_path[512]; TEST_TMPPATH(csv_path, "dso_multiref.csv");
    FILE *fp = fopen(csv_path, "w");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "filepath, is_reference\n");
    fprintf(fp, "%s/frame1.fits, 1\n", test_tmpdir());
    fprintf(fp, "%s/frame2.fits, 1\n", test_tmpdir());
    fclose(fp);

    FrameInfo *f = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv_path, &f, &n));
    ASSERT_EQ(n, 2);
    ASSERT_EQ(f[0].is_reference, 1);
    ASSERT_EQ(f[1].is_reference, 1);
    free(f);
    return 0;
}

/* ------------------------------------------------------------------
 * Integration — median
 * ------------------------------------------------------------------ */

static int test_integrate_median_basic(void)
{
    /* 5 frames of constant value → median = that value */
    Image imgs[5];
    for (int i = 0; i < 5; i++) imgs[i] = make_image_const(4, 4, 42.0f);
    const Image *ptrs[5];
    for (int i = 0; i < 5; i++) ptrs[i] = &imgs[i];

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_median(ptrs, 5, &out));
    ASSERT_EQ(out.width, 4);
    ASSERT_EQ(out.height, 4);
    for (int i = 0; i < 16; i++) ASSERT_NEAR(out.data[i], 42.0f, 1e-6f);

    image_free(&out);
    for (int i = 0; i < 5; i++) image_free(&imgs[i]);
    return 0;
}

static int test_integrate_median_outlier(void)
{
    /* 5 frames: 3 at 10.0, 2 outliers at 1000.0 → median = 10.0 */
    Image imgs[5];
    imgs[0] = make_image_const(4, 4, 10.0f);
    imgs[1] = make_image_const(4, 4, 10.0f);
    imgs[2] = make_image_const(4, 4, 10.0f);
    imgs[3] = make_image_const(4, 4, 1000.0f);
    imgs[4] = make_image_const(4, 4, 1000.0f);
    const Image *ptrs[5];
    for (int i = 0; i < 5; i++) ptrs[i] = &imgs[i];

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_median(ptrs, 5, &out));
    for (int i = 0; i < 16; i++) ASSERT_NEAR(out.data[i], 10.0f, 1e-6f);

    image_free(&out);
    for (int i = 0; i < 5; i++) image_free(&imgs[i]);
    return 0;
}

static int test_integrate_median_even_count(void)
{
    /* 4 frames: values 1,2,3,4 → median = (2+3)/2 = 2.5 */
    Image imgs[4];
    imgs[0] = make_image_const(4, 4, 1.0f);
    imgs[1] = make_image_const(4, 4, 2.0f);
    imgs[2] = make_image_const(4, 4, 3.0f);
    imgs[3] = make_image_const(4, 4, 4.0f);
    const Image *ptrs[4];
    for (int i = 0; i < 4; i++) ptrs[i] = &imgs[i];

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_median(ptrs, 4, &out));
    for (int i = 0; i < 16; i++) ASSERT_NEAR(out.data[i], 2.5f, 1e-6f);

    image_free(&out);
    for (int i = 0; i < 4; i++) image_free(&imgs[i]);
    return 0;
}

static int test_integrate_median_nan_skip(void)
{
    /* 3 frames: [5, NaN, 7] → median of {5,7} = 6.0.
     * All-NaN pixel → output NaN. */
    Image imgs[3];
    imgs[0] = make_image_const(2, 2, 5.0f);
    imgs[1] = make_image_const(2, 2, NAN);
    imgs[2] = make_image_const(2, 2, 7.0f);
    /* Make pixel 0 all-NaN */
    imgs[0].data[0] = NAN;
    imgs[2].data[0] = NAN;
    const Image *ptrs[3];
    for (int i = 0; i < 3; i++) ptrs[i] = &imgs[i];

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_median(ptrs, 3, &out));
    ASSERT(isnan(out.data[0]));         /* all-NaN → NaN */
    ASSERT_NEAR(out.data[1], 6.0f, 1e-6f);  /* median of {5,7} = 6 */
    ASSERT_NEAR(out.data[2], 6.0f, 1e-6f);
    ASSERT_NEAR(out.data[3], 6.0f, 1e-6f);

    image_free(&out);
    for (int i = 0; i < 3; i++) image_free(&imgs[i]);
    return 0;
}

static int test_integrate_median_single_frame(void)
{
    /* N=1 → output equals input */
    Image img = make_image_const(4, 4, 99.0f);
    const Image *ptrs[1] = { &img };

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_median(ptrs, 1, &out));
    for (int i = 0; i < 16; i++) ASSERT_NEAR(out.data[i], 99.0f, 1e-6f);

    image_free(&out);
    image_free(&img);
    return 0;
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    SUITE("CSV Parser");
    RUN(test_csv_count);
    RUN(test_csv_filepath);
    RUN(test_csv_reference_flag);
    RUN(test_csv_blank_lines_skipped);
    RUN(test_csv_nonexistent_file);
    RUN(test_csv_2col_basic);
    RUN(test_csv_bad_col_count);

    SUITE("FITS I/O");
    RUN(test_fits_roundtrip);
    RUN(test_fits_pixel_ordering);
    RUN(test_fits_load_missing);
    RUN(test_image_free_nulls_ptr);
    RUN(test_image_free_on_null_data);

    SUITE("Integration — mean");
    RUN(test_integrate_mean_constant);
    RUN(test_integrate_mean_arithmetic);
    RUN(test_integrate_mean_size_mismatch);
    RUN(test_integrate_mean_null_frames);

    SUITE("Integration — kappa-sigma clipping");
    RUN(test_kappa_sigma_removes_outlier);
    RUN(test_kappa_sigma_uniform);
    RUN(test_kappa_sigma_all_clipped_fallback);
    RUN(test_kappa_sigma_multi_pixel);

    SUITE("Integration — median");
    RUN(test_integrate_median_basic);
    RUN(test_integrate_median_outlier);
    RUN(test_integrate_median_even_count);
    RUN(test_integrate_median_nan_skip);
    RUN(test_integrate_median_single_frame);

    SUITE("Lanczos CPU");
    RUN(test_lanczos_cpu_identity);
    RUN(test_lanczos_cpu_integer_shift);
    RUN(test_lanczos_cpu_oob_pixels_zero);
    RUN(test_lanczos_cpu_singular_h);
    RUN(test_lanczos_cpu_null_args);
    RUN(test_lanczos_cpu_subpixel_range);

    SUITE("Integration — NaN handling");
    RUN(test_integrate_mean_nan_skipped);
    RUN(test_integrate_mean_all_nan_pixel);
    RUN(test_integrate_mean_single_frame);
    RUN(test_kappa_sigma_nan_handling);
    RUN(test_kappa_sigma_single_frame);
    RUN(test_kappa_sigma_two_frames);

    SUITE("Integration — AAWA");
    RUN(test_integrate_aawa_uniform);
    RUN(test_integrate_aawa_outlier_downweight);
    RUN(test_integrate_aawa_nan_handling);
    RUN(test_integrate_aawa_two_frames);
    RUN(test_integrate_aawa_convergence);

    SUITE("Lanczos CPU — edge cases");
    RUN(test_lanczos_cpu_small_image_2x2);
    RUN(test_lanczos_cpu_large_shift_all_oob);
    RUN(test_lanczos_cpu_fractional_shift_accuracy);

    SUITE("CSV — edge cases");
    RUN(test_csv_whitespace_in_path);
    RUN(test_csv_multiple_references);

    return SUMMARY();
}
