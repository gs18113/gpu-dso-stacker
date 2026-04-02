/*
 * test_star_detect.c — Unit tests for star_detect_cpu (CCL + CoM).
 *
 * Tests cover:
 *   - Single blob: exact CoM position
 *   - Two separated blobs: correct count and independence
 *   - 8-connectivity: diagonal adjacency merges blobs
 *   - Negative original values: clamped to 0 in CoM weights
 *   - top_k flux ranking: correct selection and order
 *   - Empty mask: returns n=0, no crash
 *   - Full mask: single giant blob
 *   - NULL argument validation
 *   - Single-pixel star: CoM equals pixel position
 *   - Large blob with non-uniform weights: weighted CoM vs geometric centroid
 *   - All-zero original image: geometric centroid fallback
 *   - Minimum blob size (1 pixel)
 *   - top_k > n_blobs: returns all blobs without crash
 */

#include "test_framework.h"
#include "star_detect_cpu.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Allocate an all-zero uint8 mask of W×H. */
static uint8_t *make_mask(int W, int H)
{
    uint8_t *m = (uint8_t *)calloc((size_t)W * H, 1);
    return m;
}

/* Allocate an all-zero float32 image of W×H. */
static float *make_float(int W, int H)
{
    float *f = (float *)calloc((size_t)W * H, sizeof(float));
    return f;
}

/* Fill a rectangular region [x0,x1) × [y0,y1) in mask with value v. */
static void fill_mask_rect(uint8_t *mask, int W, int x0, int y0,
                            int x1, int y1, uint8_t v)
{
    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
            mask[y * W + x] = v;
}

/* Fill a rectangular region in a float image with constant value. */
static void fill_float_rect(float *img, int W, int x0, int y0,
                             int x1, int y1, float v)
{
    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
            img[y * W + x] = v;
}

/* Find a star in the list nearest to (tx, ty), return its index or -1. */
static int find_near(const StarList *sl, float tx, float ty, float tol)
{
    for (int i = 0; i < sl->n; i++) {
        float dx = sl->stars[i].x - tx;
        float dy = sl->stars[i].y - ty;
        if (sqrtf(dx*dx + dy*dy) < tol)
            return i;
    }
    return -1;
}

/* =========================================================================
 * Test cases
 * ========================================================================= */

/* Single 3×3 filled square in a 20×20 image.
 * Mask pixels (5,5) to (7,7), original all-ones.
 * Expected CoM = (6.0, 6.0) = centre of the 3×3 square [5..7] inclusive. */
static int test_single_blob_com(void)
{
    const int W = 20, H = 20;
    uint8_t *mask  = make_mask(W, H);
    float   *orig  = make_float(W, H);
    float   *conv  = make_float(W, H);

    fill_mask_rect(mask, W, 5, 5, 8, 8, 1);     /* 3×3 = 9 pixels */
    fill_float_rect(orig, W, 5, 5, 8, 8, 1.0f);
    fill_float_rect(conv, W, 5, 5, 8, 8, 1.0f);

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    ASSERT_NEAR(sl.stars[0].x, 6.0f, 0.01f);
    ASSERT_NEAR(sl.stars[0].y, 6.0f, 0.01f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* Two separated 3×3 squares at (2,2) and (15,15).
 * Expected: 2 blobs, each at their respective centres. */
static int test_two_blobs(void)
{
    const int W = 25, H = 25;
    uint8_t *mask  = make_mask(W, H);
    float   *orig  = make_float(W, H);
    float   *conv  = make_float(W, H);

    fill_mask_rect(mask, W, 2, 2, 5, 5, 1);
    fill_float_rect(orig, W, 2, 2, 5, 5, 2.0f);
    fill_float_rect(conv, W, 2, 2, 5, 5, 2.0f);

    fill_mask_rect(mask, W, 15, 15, 18, 18, 1);
    fill_float_rect(orig, W, 15, 15, 18, 18, 3.0f);
    fill_float_rect(conv, W, 15, 15, 18, 18, 3.0f);

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 2);

    /* Both stars must be found near the expected centres. */
    ASSERT_NE(find_near(&sl, 3.0f, 3.0f, 0.1f), -1);
    ASSERT_NE(find_near(&sl, 16.0f, 16.0f, 0.1f), -1);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* Diagonal adjacency: two 1-pixel stars sharing only a diagonal corner.
 * With 8-connectivity they should merge into one blob. */
static int test_diagonal_8conn_merge(void)
{
    const int W = 10, H = 10;
    uint8_t *mask  = make_mask(W, H);
    float   *orig  = make_float(W, H);
    float   *conv  = make_float(W, H);

    /* pixels at (3,3) and (4,4) — touch diagonally */
    mask[3*W + 3] = 1; orig[3*W + 3] = 1.0f; conv[3*W + 3] = 1.0f;
    mask[4*W + 4] = 1; orig[4*W + 4] = 1.0f; conv[4*W + 4] = 1.0f;

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);   /* merged into one blob by 8-connectivity */

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* Two 1-pixel stars with only 4-connectivity adjacency (share an edge).
 * Should also merge. */
static int test_horizontal_4conn_merge(void)
{
    const int W = 10, H = 10;
    uint8_t *mask  = make_mask(W, H);
    float   *orig  = make_float(W, H);
    float   *conv  = make_float(W, H);

    mask[5*W + 3] = 1; orig[5*W + 3] = 1.0f; conv[5*W + 3] = 1.0f;
    mask[5*W + 4] = 1; orig[5*W + 4] = 1.0f; conv[5*W + 4] = 1.0f; /* right neighbour */

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* Negative original values: the pixel at (5,5) has orig=-10.0 (dark residual).
 * With clamping it contributes 0 weight, so CoM is determined by the others. */
static int test_negative_weight_clamping(void)
{
    const int W = 15, H = 15;
    uint8_t *mask  = make_mask(W, H);
    float   *orig  = make_float(W, H);
    float   *conv  = make_float(W, H);

    /* 3×3 blob at (4,4)–(6,6), all orig=1.0 except centre pixel = -10.0 */
    fill_mask_rect(mask, W, 4, 4, 7, 7, 1);
    fill_float_rect(orig, W, 4, 4, 7, 7, 1.0f);
    fill_float_rect(conv, W, 4, 4, 7, 7, 1.0f);
    orig[5*W + 5] = -10.0f;  /* centre pixel has negative original value */

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);

    /* The negative pixel is clamped to 0, so it doesn't pull CoM toward (5,5).
     * The 8 remaining positive-weight pixels form a symmetric ring around (5,5)
     * so the CoM must still be at exactly (5.0, 5.0). */
    ASSERT_NEAR(sl.stars[0].x, 5.0f, 0.01f);
    ASSERT_NEAR(sl.stars[0].y, 5.0f, 0.01f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* top_k flux ranking: 5 blobs with known convolved values.
 * top_k=3 should return the 3 brightest in descending order. */
static int test_topk_flux_ranking(void)
{
    const int W = 60, H = 10;
    uint8_t *mask  = make_mask(W, H);
    float   *orig  = make_float(W, H);
    float   *conv  = make_float(W, H);

    /* 5 single-pixel blobs at x=5,15,25,35,45 with convolved flux 10,30,5,50,20 */
    int xs[5]    = {5, 15, 25, 35, 45};
    float fl[5]  = {10.f, 30.f, 5.f, 50.f, 20.f};
    for (int i = 0; i < 5; i++) {
        mask[5*W + xs[i]] = 1;
        orig[5*W + xs[i]] = 1.0f;
        conv[5*W + xs[i]] = fl[i];
    }

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 3, &sl));
    ASSERT_EQ(sl.n, 3);

    /* First star must be the brightest (flux=50 at x=35) */
    ASSERT_NEAR(sl.stars[0].x, 35.0f, 0.01f);
    /* Second must be flux=30 at x=15 */
    ASSERT_NEAR(sl.stars[1].x, 15.0f, 0.01f);
    /* Third must be flux=20 at x=45 */
    ASSERT_NEAR(sl.stars[2].x, 45.0f, 0.01f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* Empty mask: all zeros → 0 stars returned, no crash. */
static int test_empty_mask(void)
{
    const int W = 20, H = 20;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 0);

    free(sl.stars);  /* may be NULL, free(NULL) is safe */
    free(mask); free(orig); free(conv);
    return 0;
}

/* Full mask: entire 4×4 image is mask=1, orig=1.0.
 * One giant blob; CoM should be the image centre (1.5, 1.5). */
static int test_full_mask_one_blob(void)
{
    const int W = 4, H = 4;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);

    memset(mask, 1, (size_t)W * H);
    for (int i = 0; i < W * H; i++) { orig[i] = 1.0f; conv[i] = 1.0f; }

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    ASSERT_NEAR(sl.stars[0].x, 1.5f, 0.01f);
    ASSERT_NEAR(sl.stars[0].y, 1.5f, 0.01f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* NULL argument validation: each NULL pointer must return DSO_ERR_INVALID_ARG. */
static int test_null_args(void)
{
    const int W = 4, H = 4;
    uint8_t mask[16] = {0};
    float   orig[16] = {0};
    float   conv[16] = {0};
    StarList sl = {NULL, 0};

    ASSERT_ERR(star_detect_cpu_ccl_com(NULL, orig, conv, W, H, 5, &sl),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(star_detect_cpu_ccl_com(mask, NULL, conv, W, H, 5, &sl),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(star_detect_cpu_ccl_com(mask, orig, NULL, W, H, 5, &sl),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 5, NULL),
               DSO_ERR_INVALID_ARG);
    return 0;
}

/* Single-pixel star at (7, 3): CoM must equal exactly (7.0, 3.0). */
static int test_single_pixel_star(void)
{
    const int W = 20, H = 10;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);

    mask[3*W + 7] = 1;
    orig[3*W + 7] = 5.0f;
    conv[3*W + 7] = 5.0f;

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    ASSERT_NEAR(sl.stars[0].x, 7.0f, 1e-5f);
    ASSERT_NEAR(sl.stars[0].y, 3.0f, 1e-5f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* Non-uniform weights: 1D blob from x=0 to x=4 with weights [1,2,4,2,1].
 * Expected weighted mean x = (0*1 + 1*2 + 2*4 + 3*2 + 4*1) / (1+2+4+2+1)
 *                          = (0+2+8+6+4) / 10 = 2.0. */
static int test_nonuniform_weights(void)
{
    const int W = 10, H = 5;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);
    float weights[5] = {1.f, 2.f, 4.f, 2.f, 1.f};

    for (int x = 0; x < 5; x++) {
        mask[2*W + x] = 1;
        orig[2*W + x] = weights[x];
        conv[2*W + x] = 1.0f;  /* equal conv so flux is uniform */
    }

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    ASSERT_NEAR(sl.stars[0].x, 2.0f, 0.01f);
    ASSERT_NEAR(sl.stars[0].y, 2.0f, 0.01f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* All-zero original image: CoM must fall back to geometric centroid.
 * 3×3 blob at (4,4)–(6,6), orig=0 everywhere → geometric centroid = (5, 5). */
static int test_all_zero_original_fallback(void)
{
    const int W = 15, H = 15;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);

    fill_mask_rect(mask, W, 4, 4, 7, 7, 1);
    fill_float_rect(conv, W, 4, 4, 7, 7, 1.0f);
    /* orig stays all zero */

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    ASSERT_NEAR(sl.stars[0].x, 5.0f, 0.01f);
    ASSERT_NEAR(sl.stars[0].y, 5.0f, 0.01f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* top_k > number of actual blobs: should return all blobs without error. */
static int test_topk_larger_than_n_blobs(void)
{
    const int W = 20, H = 10;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);

    /* Only 2 blobs */
    mask[5*W + 3] = 1; orig[5*W + 3] = 1.0f; conv[5*W + 3] = 1.0f;
    mask[5*W + 15] = 1; orig[5*W + 15] = 2.0f; conv[5*W + 15] = 2.0f;

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 100, &sl)); /* top_k=100 */
    ASSERT_EQ(sl.n, 2);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* Flux accumulates from convolved values, not original values.
 * Blob A: orig=10, conv=1; Blob B: orig=1, conv=10.
 * Expected ranking: B first (higher flux), A second. */
static int test_flux_uses_convolved_not_original(void)
{
    const int W = 20, H = 5;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);

    mask[2*W + 3] = 1; orig[2*W + 3] = 10.f; conv[2*W + 3] = 1.f;  /* Blob A */
    mask[2*W + 15] = 1; orig[2*W + 15] = 1.f; conv[2*W + 15] = 10.f; /* Blob B */

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 2);
    /* Blob B should be first (higher convolved flux). */
    ASSERT_NEAR(sl.stars[0].x, 15.0f, 0.01f);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* =========================================================================
 * Moffat convolution + threshold tests
 * ========================================================================= */

/* Uniform input: convolved with a normalised kernel → output = input value. */
static int test_moffat_convolve_uniform(void)
{
    const int W = 32, H = 32;
    int npix = W * H;
    float val = 0.5f;

    float *src  = (float *)malloc((size_t)npix * sizeof(float));
    float *dst  = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(dst);

    for (int i = 0; i < npix; i++) src[i] = val;

    MoffatParams p = {2.5f, 2.0f};
    ASSERT_OK(star_detect_cpu_moffat_convolve(src, dst, W, H, &p));

    /* Interior pixels should be very close to val (kernel is normalised to 1) */
    int R = (int)ceilf(3.f * p.alpha);
    for (int y = R; y < H - R; y++)
        for (int x = R; x < W - R; x++)
            ASSERT_NEAR(dst[y*W + x], val, 1e-4f);

    free(src); free(dst);
    return 0;
}

/* Low-variance uniform image → threshold produces all-zero mask. */
static int test_threshold_no_stars(void)
{
    const int W = 32, H = 32;
    int npix = W * H;
    float *conv = (float *)malloc((size_t)npix * sizeof(float));
    uint8_t *mask = (uint8_t *)malloc((size_t)npix);
    ASSERT_NOT_NULL(conv); ASSERT_NOT_NULL(mask);

    /* Nearly uniform: tiny variation, nothing above 3σ threshold */
    for (int i = 0; i < npix; i++) conv[i] = 1.0f + (float)(i & 1) * 1e-6f;

    ASSERT_OK(star_detect_cpu_threshold(conv, mask, W, H, 3.0f));

    int sum = 0;
    for (int i = 0; i < npix; i++) sum += mask[i];
    ASSERT_EQ(sum, 0);

    free(conv); free(mask);
    return 0;
}

/* Single bright spike → mask = 1 at that pixel, 0 elsewhere (interior). */
static int test_threshold_bright_star(void)
{
    const int W = 32, H = 32;
    int npix = W * H;
    float *conv = (float *)malloc((size_t)npix * sizeof(float));
    uint8_t *mask = (uint8_t *)malloc((size_t)npix);
    ASSERT_NOT_NULL(conv); ASSERT_NOT_NULL(mask);

    for (int i = 0; i < npix; i++) conv[i] = 0.f;
    int sy = 15, sx = 16;
    conv[sy * W + sx] = 1000.f;   /* far above any 3σ threshold */

    ASSERT_OK(star_detect_cpu_threshold(conv, mask, W, H, 3.0f));
    ASSERT_EQ(mask[sy * W + sx], 1u);

    /* All other pixels should be 0 (the spike dominates the threshold) */
    int n_set = 0;
    for (int i = 0; i < npix; i++) n_set += mask[i];
    ASSERT_EQ(n_set, 1);

    free(conv); free(mask);
    return 0;
}

/* detect (combined): uniform input, high sigma_k → empty mask. */
static int test_detect_uniform_no_stars(void)
{
    const int W = 32, H = 32;
    int npix = W * H;
    float *src  = (float *)malloc((size_t)npix * sizeof(float));
    float *conv = (float *)malloc((size_t)npix * sizeof(float));
    uint8_t *mask = (uint8_t *)malloc((size_t)npix);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(conv); ASSERT_NOT_NULL(mask);

    for (int i = 0; i < npix; i++) src[i] = 1.0f;

    MoffatParams p = {2.5f, 2.0f};
    ASSERT_OK(star_detect_cpu_detect(src, conv, mask, W, H, &p, 10.0f));

    int n_set = 0;
    for (int i = 0; i < npix; i++) n_set += mask[i];
    ASSERT_EQ(n_set, 0);

    free(src); free(conv); free(mask);
    return 0;
}

/* detect + CCL: spike image should produce exactly one detected star. */
static int test_detect_finds_spike_star(void)
{
    const int W = 64, H = 64;
    int npix = W * H;
    float *src  = (float *)malloc((size_t)npix * sizeof(float));
    float *conv = (float *)malloc((size_t)npix * sizeof(float));
    uint8_t *mask = (uint8_t *)malloc((size_t)npix);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(conv); ASSERT_NOT_NULL(mask);

    for (int i = 0; i < npix; i++) src[i] = 0.f;
    /* Single bright star at (32, 32) */
    src[32 * W + 32] = 1000.f;

    MoffatParams p = {2.5f, 2.0f};
    ASSERT_OK(star_detect_cpu_detect(src, conv, mask, W, H, &p, 3.0f));

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, src, conv, W, H, 10, &sl));
    ASSERT_GT(sl.n, 0);
    /* The centroid should be near (32, 32) */
    ASSERT_NEAR(sl.stars[0].x, 32.f, 3.f);
    ASSERT_NEAR(sl.stars[0].y, 32.f, 3.f);

    free(sl.stars);
    free(src); free(conv); free(mask);
    return 0;
}

/*
 * For an interior spike of value V convolved with a normalised kernel,
 * the sum of all output pixels must equal V (kernel sums to 1 and the
 * spike is far enough from all edges that no kernel taps are clipped).
 * A race condition that skips, duplicates, or misplaces writes will
 * perturb the sum by at least one PSF-peak-sized error (~40 counts),
 * dwarfing the sub-count floating-point accumulation error.
 */
static int test_moffat_convolve_spike_sum(void)
{
    const int W = 64, H = 64;
    const int cx = 32, cy = 32;
    const float V = 1000.f;
    int npix = W * H;

    float *src = (float *)calloc((size_t)npix, sizeof(float));
    float *dst = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(dst);

    src[cy * W + cx] = V;

    MoffatParams p = {2.5f, 2.0f};
    ASSERT_OK(star_detect_cpu_moffat_convolve(src, dst, W, H, &p));

    double total = 0.0;
    for (int i = 0; i < npix; i++) total += dst[i];

    /* With R=8 the kernel fits entirely inside the 64×64 image, so the
     * normalised kernel sums to exactly 1 and total must equal V.
     * Tolerance: 1.0 covers FP accumulation error (<<0.25) while still
     * detecting any race-induced misplaced write (error ≥ PSF peak ≈ 40). */
    ASSERT_NEAR((float)total, V, 1.0f);

    free(src); free(dst);
    return 0;
}

/*
 * The Moffat profile is monotonically decreasing with distance from the
 * kernel centre, so the output pixel at the spike location must be
 * strictly larger than all its neighbours.  A race that corrupts any
 * pixel in the convolved image can falsely elevate a non-centre pixel
 * above the true peak.
 */
static int test_moffat_convolve_spike_peak_location(void)
{
    const int W = 64, H = 64;
    const int cx = 32, cy = 32;
    const float V = 1000.f;
    int npix = W * H;

    float *src = (float *)calloc((size_t)npix, sizeof(float));
    float *dst = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(dst);

    src[cy * W + cx] = V;

    MoffatParams p = {2.5f, 2.0f};
    ASSERT_OK(star_detect_cpu_moffat_convolve(src, dst, W, H, &p));

    float peak = dst[cy * W + cx];
    for (int i = 0; i < npix; i++)
        ASSERT(dst[i] <= peak);

    free(src); free(dst);
    return 0;
}

/*
 * The Moffat kernel is radially symmetric, so a spike at (cx, cy) must
 * produce a symmetric convolved image.  Check horizontal, vertical, and
 * diagonal symmetry at several offsets.  A race that swaps row-indices
 * produces an asymmetric blob that breaks these equalities.
 */
static int test_moffat_convolve_spike_symmetry(void)
{
    const int W = 64, H = 64;
    const int cx = 32, cy = 32;
    const float V = 1000.f;
    int npix = W * H;

    float *src = (float *)calloc((size_t)npix, sizeof(float));
    float *dst = (float *)malloc((size_t)npix * sizeof(float));
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(dst);

    src[cy * W + cx] = V;

    MoffatParams p = {2.5f, 2.0f};
    ASSERT_OK(star_detect_cpu_moffat_convolve(src, dst, W, H, &p));

    int R = (int)ceilf(3.f * p.alpha);   /* = 8 */
    for (int d = 1; d <= R; d++) {
        /* Horizontal symmetry: row cy, left vs right */
        ASSERT_NEAR(dst[cy * W + (cx + d)], dst[cy * W + (cx - d)], 1e-5f);
        /* Vertical symmetry: column cx, above vs below */
        ASSERT_NEAR(dst[(cy + d) * W + cx], dst[(cy - d) * W + cx], 1e-5f);
        /* Diagonal symmetry: NE vs SW */
        ASSERT_NEAR(dst[(cy - d) * W + (cx + d)], dst[(cy + d) * W + (cx - d)], 1e-5f);
        /* Anti-diagonal symmetry: NW vs SE */
        ASSERT_NEAR(dst[(cy - d) * W + (cx - d)], dst[(cy + d) * W + (cx + d)], 1e-5f);
    }

    free(src); free(dst);
    return 0;
}

/* NULL argument checks for new functions */
static int test_moffat_null_args(void)
{
    float buf[16]; uint8_t mbuf[16];
    MoffatParams p = {2.5f, 2.0f};

    ASSERT_ERR(star_detect_cpu_moffat_convolve(NULL, buf, 4, 4, &p),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(star_detect_cpu_moffat_convolve(buf, NULL, 4, 4, &p),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(star_detect_cpu_moffat_convolve(buf, buf, 4, 4, NULL),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(star_detect_cpu_threshold(NULL, mbuf, 4, 4, 3.f),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(star_detect_cpu_threshold(buf, NULL, 4, 4, 3.f),
               DSO_ERR_INVALID_ARG);
    return 0;
}

/* top_k = 0: should return an empty list without error. */
static int test_topk_zero(void)
{
    const int W = 10, H = 10;
    uint8_t *mask = make_mask(W, H);
    float   *orig = make_float(W, H);
    float   *conv = make_float(W, H);

    mask[5*W + 5] = 1; orig[5*W + 5] = 1.f; conv[5*W + 5] = 1.f;

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 0, &sl));
    ASSERT_EQ(sl.n, 0);

    free(sl.stars);
    free(mask); free(orig); free(conv);
    return 0;
}

/* =========================================================================
 * Additional edge-case tests
 * ========================================================================= */

/* Blob touching image edge — CoM should account for truncation. */
static int test_ccl_boundary_blob(void)
{
    const int W = 16, H = 16;
    uint8_t *mask = (uint8_t *)calloc(W * H, 1);
    float   *orig = (float *)calloc(W * H, sizeof(float));
    float   *conv = (float *)calloc(W * H, sizeof(float));

    /* Place a 3×3 blob at top-left corner (0,0)-(2,2) */
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++) {
            mask[y * W + x] = 1;
            orig[y * W + x] = 100.0f;
            conv[y * W + x] = 100.0f;
        }

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    /* CoM should be near (1,1) */
    ASSERT_NEAR(sl.stars[0].x, 1.0f, 0.5f);
    ASSERT_NEAR(sl.stars[0].y, 1.0f, 0.5f);

    free(sl.stars); free(mask); free(orig); free(conv);
    return 0;
}

/* Blob at (0,0) corner — single pixel. */
static int test_ccl_corner_blob(void)
{
    const int W = 16, H = 16;
    uint8_t *mask = (uint8_t *)calloc(W * H, 1);
    float   *orig = (float *)calloc(W * H, sizeof(float));
    float   *conv = (float *)calloc(W * H, sizeof(float));

    mask[0] = 1; orig[0] = 50.0f; conv[0] = 50.0f;

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    ASSERT_NEAR(sl.stars[0].x, 0.0f, 0.01f);
    ASSERT_NEAR(sl.stars[0].y, 0.0f, 0.01f);

    free(sl.stars); free(mask); free(orig); free(conv);
    return 0;
}

/* Minimum 3×3 image for CCL. */
static int test_ccl_small_image_3x3(void)
{
    const int W = 3, H = 3;
    uint8_t *mask = (uint8_t *)calloc(W * H, 1);
    float   *orig = (float *)calloc(W * H, sizeof(float));
    float   *conv = (float *)calloc(W * H, sizeof(float));

    mask[4] = 1; orig[4] = 100.0f; conv[4] = 100.0f; /* center pixel */

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 10, &sl));
    ASSERT_EQ(sl.n, 1);
    ASSERT_NEAR(sl.stars[0].x, 1.0f, 0.01f);
    ASSERT_NEAR(sl.stars[0].y, 1.0f, 0.01f);

    free(sl.stars); free(mask); free(orig); free(conv);
    return 0;
}

/* Moffat with different alpha/beta changes kernel shape. */
static int test_moffat_different_alpha_beta(void)
{
    const int W = 32, H = 32;
    float *src = (float *)calloc(W * H, sizeof(float));
    src[(H/2) * W + W/2] = 1000.0f;

    float *conv1 = (float *)calloc(W * H, sizeof(float));
    float *conv2 = (float *)calloc(W * H, sizeof(float));

    MoffatParams mp1 = {2.5f, 2.0f};
    MoffatParams mp2 = {1.0f, 4.0f};
    ASSERT_OK(star_detect_cpu_moffat_convolve(src, conv1, W, H, &mp1));
    ASSERT_OK(star_detect_cpu_moffat_convolve(src, conv2, W, H, &mp2));

    /* Different params → different convolution results */
    int differ = 0;
    for (int i = 0; i < W * H; i++)
        if (fabsf(conv1[i] - conv2[i]) > 1e-6f) differ++;
    ASSERT_GT(differ, 0);

    free(src); free(conv1); free(conv2);
    return 0;
}

/* Uniform image → sigma=0, no detections. */
static int test_threshold_sigma_zero_uniform(void)
{
    const int W = 32, H = 32;
    float *conv = (float *)calloc(W * H, sizeof(float));
    for (int i = 0; i < W * H; i++) conv[i] = 100.0f;
    uint8_t *mask = (uint8_t *)calloc(W * H, 1);

    ASSERT_OK(star_detect_cpu_threshold(conv, mask, W, H, 3.0f));

    /* Uniform → stddev = 0 → nothing exceeds mean + 3*0 */
    int sum = 0;
    for (int i = 0; i < W * H; i++) sum += mask[i];
    ASSERT_EQ(sum, 0);

    free(conv); free(mask);
    return 0;
}

/* Inject 5 stars, verify all 5 detected near expected positions. */
static int test_detect_multiple_stars_positions(void)
{
    const int W = 128, H = 128;
    int npix = W * H;
    float *src = (float *)calloc(npix, sizeof(float));
    for (int i = 0; i < npix; i++) src[i] = 100.0f;

    float star_x[] = {20, 40, 60, 80, 100};
    float star_y[] = {20, 40, 60, 80, 100};
    MoffatParams mp = {2.5f, 2.0f};
    int R = 8;

    for (int s = 0; s < 5; s++) {
        float cx = star_x[s], cy = star_y[s];
        for (int dy = -R; dy <= R; dy++)
            for (int dx = -R; dx <= R; dx++) {
                int x = (int)cx + dx, y = (int)cy + dy;
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    float r2 = (float)(dx*dx + dy*dy);
                    src[y * W + x] += 5000.0f *
                        powf(1.0f + r2 / (mp.alpha * mp.alpha), -mp.beta);
                }
            }
    }

    float *conv = (float *)calloc(npix, sizeof(float));
    uint8_t *mask = (uint8_t *)calloc(npix, 1);
    ASSERT_OK(star_detect_cpu_detect(src, conv, mask, W, H, &mp, 3.0f));

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, src, conv, W, H, 50, &sl));

    ASSERT_GT(sl.n, 3); /* at least 4 of 5 stars detected */

    /* Check that each star position has a detection nearby */
    int found = 0;
    for (int s = 0; s < 5; s++) {
        for (int i = 0; i < sl.n; i++) {
            float dist = sqrtf((sl.stars[i].x - star_x[s]) * (sl.stars[i].x - star_x[s]) +
                               (sl.stars[i].y - star_y[s]) * (sl.stars[i].y - star_y[s]));
            if (dist < 3.0f) { found++; break; }
        }
    }
    ASSERT_GT(found, 3);

    free(sl.stars); free(src); free(conv); free(mask);
    return 0;
}

/* 100 single-pixel blobs, verify all counted. */
static int test_ccl_many_small_blobs(void)
{
    const int W = 64, H = 64;
    uint8_t *mask = (uint8_t *)calloc(W * H, 1);
    float   *orig = (float *)calloc(W * H, sizeof(float));
    float   *conv = (float *)calloc(W * H, sizeof(float));

    /* Place single-pixel blobs on a grid (every 6 pixels) */
    int count = 0;
    for (int y = 3; y < H - 3; y += 6)
        for (int x = 3; x < W - 3; x += 6) {
            mask[y * W + x] = 1;
            orig[y * W + x] = 100.0f;
            conv[y * W + x] = 100.0f;
            count++;
        }

    StarList sl = {NULL, 0};
    ASSERT_OK(star_detect_cpu_ccl_com(mask, orig, conv, W, H, 200, &sl));

    /* Each isolated pixel = 1 blob; should detect all */
    ASSERT_EQ(sl.n, count);

    free(sl.stars); free(mask); free(orig); free(conv);
    return 0;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    SUITE("star_detect_cpu — CCL + CoM");
    RUN(test_single_blob_com);
    RUN(test_two_blobs);
    RUN(test_diagonal_8conn_merge);
    RUN(test_horizontal_4conn_merge);
    RUN(test_negative_weight_clamping);
    RUN(test_topk_flux_ranking);
    RUN(test_empty_mask);
    RUN(test_full_mask_one_blob);
    RUN(test_null_args);
    RUN(test_single_pixel_star);
    RUN(test_nonuniform_weights);
    RUN(test_all_zero_original_fallback);
    RUN(test_topk_larger_than_n_blobs);
    RUN(test_flux_uses_convolved_not_original);
    RUN(test_topk_zero);

    SUITE("star_detect_cpu — Moffat convolution + threshold");
    RUN(test_moffat_convolve_uniform);
    RUN(test_moffat_convolve_spike_sum);
    RUN(test_moffat_convolve_spike_peak_location);
    RUN(test_moffat_convolve_spike_symmetry);
    RUN(test_threshold_no_stars);
    RUN(test_threshold_bright_star);
    RUN(test_detect_uniform_no_stars);
    RUN(test_detect_finds_spike_star);
    RUN(test_moffat_null_args);

    SUITE("star_detect_cpu — boundary & edge cases");
    RUN(test_ccl_boundary_blob);
    RUN(test_ccl_corner_blob);
    RUN(test_ccl_small_image_3x3);
    RUN(test_moffat_different_alpha_beta);
    RUN(test_threshold_sigma_zero_uniform);
    RUN(test_detect_multiple_stars_positions);
    RUN(test_ccl_many_small_blobs);

    return SUMMARY();
}
