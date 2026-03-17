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

/* Write a minimal 11-column frame-list CSV (pre-computed homographies) to `path`. */
static void write_test_csv(const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) { perror("write_test_csv"); exit(1); }
    fprintf(fp, "filepath, is_reference, h00, h01, h02, h10, h11, h12, h20, h21, h22\n");
    fprintf(fp, "/tmp/dso_frame1.fits, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1\n");
    fprintf(fp, "/tmp/dso_frame2.fits, 0, 1, 0, 2.5, 0, 1, 1.3, 0, 0, 1\n");
    fclose(fp);
}

/* Write a minimal 2-column frame-list CSV (no pre-computed homographies) to `path`. */
static void write_test_csv_2col(const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) { perror("write_test_csv_2col"); exit(1); }
    fprintf(fp, "filepath, is_reference\n");
    fprintf(fp, "/tmp/dso_frame1.fits, 1\n");
    fprintf(fp, "/tmp/dso_frame2.fits, 0\n");
    fclose(fp);
}

/* =========================================================================
 * CSV PARSER TESTS
 * ====================================================================== */

/* Parsing a valid 2-row 11-col CSV must return exactly 2 frames. */
static int test_csv_count(void)
{
    write_test_csv("/tmp/dso_test.csv");
    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_OK(csv_parse("/tmp/dso_test.csv", &f, &n, &ht));
    ASSERT_EQ(n, 2);
    ASSERT_NOT_NULL(f);
    free(f);
    return 0;
}

/* The first row (header) must be skipped; filepaths come from data rows. */
static int test_csv_filepath(void)
{
    write_test_csv("/tmp/dso_test.csv");
    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_OK(csv_parse("/tmp/dso_test.csv", &f, &n, &ht));
    ASSERT_EQ(strcmp(f[0].filepath, "/tmp/dso_frame1.fits"), 0);
    ASSERT_EQ(strcmp(f[1].filepath, "/tmp/dso_frame2.fits"), 0);
    free(f);
    return 0;
}

/* is_reference field must be parsed as an integer (1 or 0). */
static int test_csv_reference_flag(void)
{
    write_test_csv("/tmp/dso_test.csv");
    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_OK(csv_parse("/tmp/dso_test.csv", &f, &n, &ht));
    ASSERT_EQ(f[0].is_reference, 1);
    ASSERT_EQ(f[1].is_reference, 0);
    free(f);
    return 0;
}

/* All 9 homography values must be parsed in row-major order (h00..h22). */
static int test_csv_homography_values(void)
{
    write_test_csv("/tmp/dso_test.csv");
    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_OK(csv_parse("/tmp/dso_test.csv", &f, &n, &ht));

    /* Frame 0: identity */
    ASSERT_NEAR(f[0].H.h[0], 1.0, 1e-9);  /* h00 */
    ASSERT_NEAR(f[0].H.h[1], 0.0, 1e-9);  /* h01 */
    ASSERT_NEAR(f[0].H.h[2], 0.0, 1e-9);  /* h02 */
    ASSERT_NEAR(f[0].H.h[8], 1.0, 1e-9);  /* h22 */

    /* Frame 1: translation with non-zero off-diagonals */
    ASSERT_NEAR(f[1].H.h[2], 2.5, 1e-9);  /* h02 = x-translation */
    ASSERT_NEAR(f[1].H.h[5], 1.3, 1e-9);  /* h12 = y-translation */

    free(f);
    return 0;
}

/* Blank lines in the CSV body must be silently skipped. */
static int test_csv_blank_lines_skipped(void)
{
    const char *path = "/tmp/dso_test_blank.csv";
    FILE *fp = fopen(path, "w");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "filepath, is_reference, h00, h01, h02, h10, h11, h12, h20, h21, h22\n");
    fprintf(fp, "\n");
    fprintf(fp, "/tmp/f.fits, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1\n");
    fprintf(fp, "\n");
    fclose(fp);

    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_OK(csv_parse(path, &f, &n, &ht));
    ASSERT_EQ(n, 1);   /* only the data row, not the blank lines */
    free(f);
    return 0;
}

/* Opening a nonexistent file must return an I/O error. */
static int test_csv_nonexistent_file(void)
{
    FrameInfo *f = NULL; int n = 0; int ht = -1;
    DsoError err = csv_parse("/tmp/does_not_exist_xyz.csv", &f, &n, &ht);
    ASSERT_EQ(err, DSO_ERR_IO);
    return 0;
}

/*
 * 11-column CSV must set has_transforms=1.
 * Verifies the flag accurately reflects that homography values were parsed.
 */
static int test_csv_has_transforms_11col(void)
{
    write_test_csv("/tmp/dso_test.csv");
    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_OK(csv_parse("/tmp/dso_test.csv", &f, &n, &ht));
    ASSERT_EQ(ht, 1);
    free(f);
    return 0;
}

/*
 * 2-column CSV must set has_transforms=0 and zero-init all homographies.
 * Verifies that filepaths and is_reference are still correctly parsed.
 */
static int test_csv_2col_no_transforms(void)
{
    write_test_csv_2col("/tmp/dso_test_2col.csv");
    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_OK(csv_parse("/tmp/dso_test_2col.csv", &f, &n, &ht));
    ASSERT_EQ(n, 2);
    ASSERT_EQ(ht, 0);
    /* Filepaths must be correct */
    ASSERT_EQ(strcmp(f[0].filepath, "/tmp/dso_frame1.fits"), 0);
    ASSERT_EQ(f[0].is_reference, 1);
    ASSERT_EQ(f[1].is_reference, 0);
    /* Homography must be zero-initialised when not provided */
    for (int i = 0; i < 9; i++) ASSERT_NEAR(f[0].H.h[i], 0.0, 1e-12);
    for (int i = 0; i < 9; i++) ASSERT_NEAR(f[1].H.h[i], 0.0, 1e-12);
    free(f);
    return 0;
}

/*
 * A CSV with an unexpected column count (e.g. 5) must return DSO_ERR_CSV.
 * Prevents silent misparse of files in wrong format.
 */
static int test_csv_bad_col_count(void)
{
    const char *path = "/tmp/dso_test_badcols.csv";
    FILE *fp = fopen(path, "w");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "filepath, is_reference, foo, bar, baz\n");
    fprintf(fp, "/tmp/f.fits, 1, 1.0, 2.0, 3.0\n");
    fclose(fp);

    FrameInfo *f = NULL; int n = 0; int ht = -1;
    ASSERT_ERR(csv_parse(path, &f, &n, &ht), DSO_ERR_CSV);
    return 0;
}

/*
 * NULL has_transforms_out pointer must return DSO_ERR_INVALID_ARG.
 */
static int test_csv_null_has_transforms_out(void)
{
    write_test_csv("/tmp/dso_test.csv");
    FrameInfo *f = NULL; int n = 0;
    ASSERT_ERR(csv_parse("/tmp/dso_test.csv", &f, &n, NULL), DSO_ERR_INVALID_ARG);
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

    ASSERT_OK(fits_save("/tmp/dso_roundtrip.fits", &src));

    Image dst = {NULL, 0, 0};
    ASSERT_OK(fits_load("/tmp/dso_roundtrip.fits", &dst));

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

    ASSERT_OK(fits_save("/tmp/dso_ordering.fits", &src));

    Image dst = {NULL, 0, 0};
    ASSERT_OK(fits_load("/tmp/dso_ordering.fits", &dst));

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
    ASSERT_ERR(fits_load("/tmp/no_such_file_abc.fits", &img), DSO_ERR_FITS);
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
 * Pixels that map outside the source bounds must be written as 0.
 * A large translation moves all source content out of the destination view.
 */
static int test_lanczos_cpu_oob_pixels_zero(void)
{
    const int W = 10, H = 10;
    Image src = make_image_const(W, H, 99.f);
    Image dst = make_image_const(W, H, -1.f);  /* sentinel: non-zero */

    /* Backward map with +1000 offset: src lookup at (dx+1000, dy+1000) — always OOB */
    Homography hom = {{ 1,0,1000, 0,1,1000, 0,0,1 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &hom));

    /* Every destination pixel should be 0 (out-of-bounds fill value) */
    for (int i = 0; i < W * H; i++) {
        if (dst.data[i] != 0.f) {
            printf("\n    dst[%d] = %f, expected 0\n", i, dst.data[i]);
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
 * and must be written as 0. No error is returned — degenerate pixels are
 * silently zeroed, not rejected. (H is the backward map used directly;
 * there is no inversion step that would detect a singular matrix.)
 */
static int test_lanczos_cpu_singular_h(void)
{
    Image src = make_image_const(8, 8, 1.f);
    Image dst = make_image_const(8, 8, -1.f);  /* sentinel */

    /* All-zero matrix: sw = 0 for every pixel → OOB → written as 0 */
    Homography H = {{ 0,0,0, 0,0,0, 0,0,0 }};
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H));
    for (int i = 0; i < 8 * 8; i++)
        ASSERT_NEAR(dst.data[i], 0.f, 1e-5f);

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
 * main
 * ====================================================================== */

int main(void)
{
    SUITE("CSV Parser");
    RUN(test_csv_count);
    RUN(test_csv_filepath);
    RUN(test_csv_reference_flag);
    RUN(test_csv_homography_values);
    RUN(test_csv_blank_lines_skipped);
    RUN(test_csv_nonexistent_file);
    RUN(test_csv_has_transforms_11col);
    RUN(test_csv_2col_no_transforms);
    RUN(test_csv_bad_col_count);
    RUN(test_csv_null_has_transforms_out);

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

    SUITE("Lanczos CPU");
    RUN(test_lanczos_cpu_identity);
    RUN(test_lanczos_cpu_integer_shift);
    RUN(test_lanczos_cpu_oob_pixels_zero);
    RUN(test_lanczos_cpu_singular_h);
    RUN(test_lanczos_cpu_null_args);
    RUN(test_lanczos_cpu_subpixel_range);

    return SUMMARY();
}
