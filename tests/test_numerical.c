/*
 * test_numerical.c — NaN/Inf/numerical stability tests across modules.
 *
 * Verifies correct handling of floating-point edge cases (NaN, Inf,
 * denormals, large values, negative values) as they propagate through
 * debayer, Lanczos, integration, star detection, and DLT.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#include "test_framework.h"
#include "dso_types.h"
#include "fits_io.h"
#include "debayer_cpu.h"
#include "lanczos_cpu.h"
#include "integration.h"
#include "star_detect_cpu.h"
#include "ransac.h"

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

#define TW 32
#define TH 32
#define NPIX (TW * TH)

static float *alloc_f(int n, float val) {
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < n; i++) p[i] = val;
    return p;
}

static Image make_const(int iw, int ih, float val) {
    Image img = {NULL, iw, ih};
    img.data = alloc_f(iw * ih, val);
    return img;
}

/* =========================================================================
 * NaN PROPAGATION
 * ====================================================================== */

/* NaN pixel in Bayer mosaic → debayer output should not crash. */
static int test_nan_through_debayer(void)
{
    float *src = alloc_f(NPIX, 100.0f);
    float *dst = alloc_f(NPIX, 0.0f);
    src[TH/2 * TW + TW/2] = NAN;

    ASSERT_OK(debayer_cpu(src, dst, TW, TH, BAYER_RGGB));
    /* No crash is the primary assertion. Output near NaN pixel may be NaN. */

    free(src); free(dst);
    return 0;
}

/* NaN in source → Lanczos mapped destination contains NaN. */
static int test_nan_through_lanczos(void)
{
    float *data = alloc_f(NPIX, 50.0f);
    data[TH/2 * TW + TW/2] = NAN;

    Homography H_id;
    double id[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(H_id.h, id, sizeof(id));

    Image src = {data, TW, TH};
    Image dst = {NULL, TW, TH};
    dst.data = alloc_f(NPIX, 0.0f);
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H_id));

    /* Primary check: no crash and output is finite elsewhere */
    int finite_count = 0;
    for (int i = 0; i < NPIX; i++)
        if (isfinite(dst.data[i])) finite_count++;
    /* Most pixels should still be finite */
    ASSERT_GT(finite_count, NPIX - 50);

    free(data); free(dst.data);
    return 0;
}

/* NaN excluded from mean integration. */
static int test_nan_through_integration_mean(void)
{
    Image f1 = make_const(TW, TH, 10.0f);
    Image f2 = make_const(TW, TH, 20.0f);
    Image f3 = make_const(TW, TH, NAN);

    const Image *ptrs[3] = {&f1, &f2, &f3};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 3, &out));

    /* Mean of 10 and 20 (NaN skipped) = 15 */
    ASSERT_NEAR(out.data[0], 15.0f, 0.01f);

    image_free(&f1); image_free(&f2); image_free(&f3); image_free(&out);
    return 0;
}

/* NaN excluded from kappa-sigma. */
static int test_nan_through_integration_kappa(void)
{
    Image f1 = make_const(TW, TH, 10.0f);
    Image f2 = make_const(TW, TH, 10.0f);
    Image f3 = make_const(TW, TH, NAN);

    const Image *ptrs[3] = {&f1, &f2, &f3};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, 3, &out, 3.0f, 3));

    /* Kappa-sigma of [10, 10] (NaN skipped) = 10 */
    ASSERT_NEAR(out.data[0], 10.0f, 0.01f);

    image_free(&f1); image_free(&f2); image_free(&f3); image_free(&out);
    return 0;
}

/* All frames NaN at a pixel → output NaN. */
static int test_all_nan_pixel_integration(void)
{
    Image f1 = make_const(TW, TH, 5.0f);
    Image f2 = make_const(TW, TH, 5.0f);
    f1.data[0] = NAN;
    f2.data[0] = NAN;

    const Image *ptrs[2] = {&f1, &f2};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 2, &out));
    ASSERT(isnan(out.data[0]));

    /* Other pixels should be 5.0 */
    ASSERT_NEAR(out.data[1], 5.0f, 0.01f);

    image_free(&f1); image_free(&f2); image_free(&out);
    return 0;
}

/* =========================================================================
 * INF HANDLING
 * ====================================================================== */

/* +Inf input to debayer should not crash. */
static int test_inf_through_debayer(void)
{
    float *src = alloc_f(NPIX, 100.0f);
    float *dst = alloc_f(NPIX, 0.0f);
    src[TH/2 * TW + TW/2] = INFINITY;

    ASSERT_OK(debayer_cpu(src, dst, TW, TH, BAYER_RGGB));
    /* No crash */

    free(src); free(dst);
    return 0;
}

/* +Inf in Lanczos source. */
static int test_inf_through_lanczos(void)
{
    float *data = alloc_f(NPIX, 50.0f);
    data[NPIX / 2] = INFINITY;

    Homography H_id;
    double id[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(H_id.h, id, sizeof(id));

    Image src = {data, TW, TH};
    Image dst = {NULL, TW, TH};
    dst.data = alloc_f(NPIX, 0.0f);
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H_id));

    /* No crash; most pixels should be finite */
    int finite_count = 0;
    for (int i = 0; i < NPIX; i++)
        if (isfinite(dst.data[i])) finite_count++;
    ASSERT_GT(finite_count, NPIX - 50);

    free(data); free(dst.data);
    return 0;
}

/* =========================================================================
 * LARGE VALUES & OVERFLOW
 * ====================================================================== */

/* Pixel values ~65535, N=100 frames → no overflow in mean integration. */
static int test_large_values_integration(void)
{
    int N = 100;
    Image *frames = (Image *)malloc((size_t)N * sizeof(Image));
    const Image **ptrs = (const Image **)malloc((size_t)N * sizeof(Image *));
    for (int i = 0; i < N; i++) {
        frames[i] = make_const(TW, TH, 65535.0f);
        ptrs[i] = &frames[i];
    }

    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, N, &out));

    /* Mean of 100 frames of 65535 = 65535 */
    ASSERT_NEAR(out.data[0], 65535.0f, 1.0f);

    for (int i = 0; i < N; i++) image_free(&frames[i]);
    image_free(&out);
    free(frames); free(ptrs);
    return 0;
}

/* Very small values (denormals) accumulate correctly. */
static int test_denormal_through_integration(void)
{
    Image f1 = make_const(TW, TH, 1e-38f);
    Image f2 = make_const(TW, TH, 1e-38f);

    const Image *ptrs[2] = {&f1, &f2};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 2, &out));

    ASSERT(out.data[0] > 0.0f);
    ASSERT(out.data[0] < 1e-37f);

    image_free(&f1); image_free(&f2); image_free(&out);
    return 0;
}

/* Negative pixel values preserved through stages. */
static int test_negative_values_through_pipeline(void)
{
    float *data = alloc_f(NPIX, -50.0f);

    /* Debayer should handle negatives */
    float *lum = alloc_f(NPIX, 0.0f);
    ASSERT_OK(debayer_cpu(data, lum, TW, TH, BAYER_NONE));
    ASSERT_NEAR(lum[NPIX / 2], -50.0f, 0.01f);

    /* Lanczos with identity H should preserve negatives */
    Homography H_id;
    double id[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(H_id.h, id, sizeof(id));
    Image src = {data, TW, TH};
    Image dst = {NULL, TW, TH};
    dst.data = alloc_f(NPIX, 0.0f);
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H_id));
    ASSERT_NEAR(dst.data[NPIX / 2], -50.0f, 0.5f);

    /* Integration of negative values */
    Image f1 = make_const(TW, TH, -10.0f);
    Image f2 = make_const(TW, TH, -20.0f);
    const Image *ptrs[2] = {&f1, &f2};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 2, &out));
    ASSERT_NEAR(out.data[0], -15.0f, 0.01f);

    free(data); free(lum); free(dst.data);
    image_free(&f1); image_free(&f2); image_free(&out);
    return 0;
}

/* =========================================================================
 * ZERO IMAGE
 * ====================================================================== */

/* All-zero image through debayer → all-zero output. */
static int test_zero_image_through_debayer(void)
{
    float *src = alloc_f(NPIX, 0.0f);
    float *dst = alloc_f(NPIX, 1.0f);

    ASSERT_OK(debayer_cpu(src, dst, TW, TH, BAYER_RGGB));

    for (int i = 0; i < NPIX; i++)
        ASSERT_NEAR(dst[i], 0.0f, 1e-6f);

    free(src); free(dst);
    return 0;
}

/* All-zero image → all-zero through Lanczos (identity H). */
static int test_zero_image_through_lanczos(void)
{
    float *data = alloc_f(NPIX, 0.0f);
    Homography H_id;
    double id[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(H_id.h, id, sizeof(id));

    Image src = {data, TW, TH};
    Image dst = {NULL, TW, TH};
    dst.data = alloc_f(NPIX, 1.0f);
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H_id));

    int margin = 4;
    for (int y = margin; y < TH - margin; y++)
        for (int x = margin; x < TW - margin; x++)
            ASSERT_NEAR(dst.data[y * TW + x], 0.0f, 1e-6f);

    free(data); free(dst.data);
    return 0;
}

/* =========================================================================
 * NUMERICAL PRECISION
 * ====================================================================== */

/* Kappa-sigma with high mean and small stddev. */
static int test_kappa_sigma_precision_high_mean(void)
{
    Image f1 = make_const(TW, TH, 50000.0f);
    Image f2 = make_const(TW, TH, 50000.0f);
    Image f3 = make_const(TW, TH, 50000.0f);
    Image f4 = make_const(TW, TH, 50000.0f);
    Image f5 = make_const(TW, TH, 50100.0f);

    const Image *ptrs[5] = {&f1, &f2, &f3, &f4, &f5};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, 5, &out, 1.5f, 3));

    /* Outlier 50100 should be rejected; result ≈ 50000 */
    ASSERT_NEAR(out.data[0], 50000.0f, 1.0f);

    image_free(&f1); image_free(&f2); image_free(&f3);
    image_free(&f4); image_free(&f5); image_free(&out);
    return 0;
}

/* Moffat convolution energy conservation. */
static int test_moffat_convolve_energy_conservation(void)
{
    int sz = 64;
    int npix = sz * sz;
    float *src = alloc_f(npix, 0.0f);
    src[(sz/2) * sz + sz/2] = 1000.0f;

    float *conv = alloc_f(npix, 0.0f);
    MoffatParams mp = {2.5f, 2.0f};
    ASSERT_OK(star_detect_cpu_moffat_convolve(src, conv, sz, sz, &mp));

    double sum_src = 0, sum_conv = 0;
    for (int i = 0; i < npix; i++) {
        sum_src += src[i];
        sum_conv += conv[i];
    }
    ASSERT_NEAR((float)sum_conv, (float)sum_src, 50.0f);

    free(src); free(conv);
    return 0;
}

/* DLT numerical conditioning with points far from origin. */
static int test_dlt_numerical_conditioning(void)
{
    StarPos ref_pts[6] = {
        {5000, 5000, 1}, {5010, 5000, 1}, {5000, 5010, 1},
        {5010, 5010, 1}, {5020, 5000, 1}, {5000, 5020, 1}
    };
    StarPos src_pts[6];
    for (int i = 0; i < 6; i++) {
        src_pts[i].x = ref_pts[i].x + 10.0f;
        src_pts[i].y = ref_pts[i].y + 5.0f;
        src_pts[i].flux = 1.0f;
    }

    Homography hom;
    ASSERT_OK(dlt_homography(ref_pts, src_pts, 6, &hom));

    ASSERT_NEAR(hom.h[0], 1.0, 0.01);
    ASSERT_NEAR(hom.h[4], 1.0, 0.01);
    ASSERT_NEAR(hom.h[2], 10.0, 0.5);
    ASSERT_NEAR(hom.h[5], 5.0, 0.5);

    return 0;
}

/* Single-frame integration: n=1 → output = input. */
static int test_single_frame_integration(void)
{
    Image f1 = make_const(TW, TH, 42.0f);
    f1.data[0] = 123.0f;

    const Image *ptrs[1] = {&f1};
    Image out_mean = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 1, &out_mean));
    ASSERT_NEAR(out_mean.data[0], 123.0f, 0.01f);
    ASSERT_NEAR(out_mean.data[1], 42.0f, 0.01f);

    Image out_ks = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(ptrs, 1, &out_ks, 3.0f, 3));
    ASSERT_NEAR(out_ks.data[0], 123.0f, 0.01f);

    image_free(&f1); image_free(&out_mean); image_free(&out_ks);
    return 0;
}

/* =========================================================================
 * MAIN
 * ====================================================================== */

int main(void)
{
    SUITE("NaN propagation");
    RUN(test_nan_through_debayer);
    RUN(test_nan_through_lanczos);
    RUN(test_nan_through_integration_mean);
    RUN(test_nan_through_integration_kappa);
    RUN(test_all_nan_pixel_integration);

    SUITE("Inf handling");
    RUN(test_inf_through_debayer);
    RUN(test_inf_through_lanczos);

    SUITE("Large values & overflow");
    RUN(test_large_values_integration);
    RUN(test_denormal_through_integration);
    RUN(test_negative_values_through_pipeline);

    SUITE("Zero image propagation");
    RUN(test_zero_image_through_debayer);
    RUN(test_zero_image_through_lanczos);

    SUITE("Numerical precision");
    RUN(test_kappa_sigma_precision_high_mean);
    RUN(test_moffat_convolve_energy_conservation);
    RUN(test_dlt_numerical_conditioning);
    RUN(test_single_frame_integration);

    return SUMMARY();
}
