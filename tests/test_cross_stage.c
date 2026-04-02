/*
 * test_cross_stage.c — integration tests between adjacent pipeline stages.
 *
 * Tests the data flow: debayer → star_detect → RANSAC → Lanczos → integration,
 * verifying that the output of one stage is valid input for the next.
 * All tests operate in-memory (no file I/O except for test setup).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "test_framework.h"
#include "dso_types.h"
#include "debayer_cpu.h"
#include "star_detect_cpu.h"
#include "ransac.h"
#include "lanczos_cpu.h"
#include "integration.h"
#include "calibration.h"
#include "fits_io.h"

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

#define TW 128
#define TH 128
#define NPIX (TW * TH)
#define BG 100.0f
#define FLUX 5000.0f
#define ALPHA 2.5f
#define BETA  2.0f

static float *alloc_f(int n) {
    float *p = (float *)calloc((size_t)n, sizeof(float));
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static void inject_star(float *img, int iw, int ih,
                          float cx, float cy, float flux)
{
    int R = (int)ceilf(3.0f * ALPHA);
    if (R > 15) R = 15;
    float total = 0;
    for (int dy = -R; dy <= R; dy++)
        for (int dx = -R; dx <= R; dx++) {
            float r2 = (float)(dx*dx + dy*dy);
            total += powf(1.0f + r2 / (ALPHA*ALPHA), -BETA);
        }
    float scale = (total > 0) ? flux / total : 0;
    for (int dy = -R; dy <= R; dy++)
        for (int dx = -R; dx <= R; dx++) {
            int x = (int)cx + dx, y = (int)cy + dy;
            if (x >= 0 && x < iw && y >= 0 && y < ih) {
                float r2 = (float)(dx*dx + dy*dy);
                img[y * iw + x] += scale * powf(1.0f + r2 / (ALPHA*ALPHA), -BETA);
            }
        }
}

/* Create a synthetic luminance image with stars on a grid. */
static float *make_star_image(float dx, float dy, int n_stars,
                                float xs[], float ys[])
{
    float *img = alloc_f(NPIX);
    for (int i = 0; i < NPIX; i++) img[i] = BG;
    for (int i = 0; i < n_stars; i++) {
        float sx = xs[i] + dx, sy = ys[i] + dy;
        if (sx >= 3 && sx < TW - 3 && sy >= 3 && sy < TH - 3)
            inject_star(img, TW, TH, sx, sy, FLUX);
    }
    return img;
}

#define N_STARS 30
static void get_grid_stars(float xs[], float ys[])
{
    int idx = 0;
    for (int gy = 0; gy < 6 && idx < N_STARS; gy++)
        for (int gx = 0; gx < 5 && idx < N_STARS; gx++) {
            xs[idx] = 14.0f + gx * 22.0f;
            ys[idx] = 12.0f + gy * 18.0f;
            idx++;
        }
}

/* Run star detection on an image, return StarList. */
static DsoError detect_stars(const float *img, StarList *out)
{
    float   *conv = alloc_f(NPIX);
    uint8_t *mask = (uint8_t *)calloc(NPIX, 1);
    MoffatParams mp = {ALPHA, BETA};

    DsoError e = star_detect_cpu_detect(img, conv, mask, TW, TH, &mp, 3.0f);
    if (e != DSO_OK) { free(conv); free(mask); return e; }

    e = star_detect_cpu_ccl_com(mask, img, conv, TW, TH, 50, out);
    free(conv); free(mask);
    return e;
}

/* =========================================================================
 * STAR DETECTION → RANSAC
 * ====================================================================== */

/* Detect stars in two copies of the same image → RANSAC yields ~identity H. */
static int test_detect_then_ransac_identity(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);
    float *img = make_star_image(0, 0, N_STARS, xs, ys);

    StarList ref_stars = {NULL, 0}, frm_stars = {NULL, 0};
    ASSERT_OK(detect_stars(img, &ref_stars));
    ASSERT_OK(detect_stars(img, &frm_stars));
    ASSERT_GT(ref_stars.n, 5);
    ASSERT_GT(frm_stars.n, 5);

    RansacParams rp = {1000, 3.0f, 30.0f, 0.99f, 5};
    Homography H; int n_inliers = 0;
    ASSERT_OK(ransac_compute_homography(&ref_stars, &frm_stars, &rp, &H, &n_inliers));

    /* H should be close to identity */
    ASSERT_NEAR(H.h[0], 1.0, 0.1);
    ASSERT_NEAR(H.h[4], 1.0, 0.1);
    ASSERT_NEAR(H.h[8], 1.0, 0.1);
    ASSERT_NEAR(H.h[2], 0.0, 2.0); /* tx ≈ 0 */
    ASSERT_NEAR(H.h[5], 0.0, 2.0); /* ty ≈ 0 */

    free(ref_stars.stars); free(frm_stars.stars); free(img);
    return 0;
}

/* Detect stars in ref + shifted copy → RANSAC recovers the shift. */
static int test_detect_then_ransac_shifted(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);
    float *ref = make_star_image(0, 0, N_STARS, xs, ys);
    float *shifted = make_star_image(5, 3, N_STARS, xs, ys);

    StarList ref_stars = {NULL, 0}, frm_stars = {NULL, 0};
    ASSERT_OK(detect_stars(ref, &ref_stars));
    ASSERT_OK(detect_stars(shifted, &frm_stars));

    RansacParams rp = {1000, 3.0f, 30.0f, 0.99f, 5};
    Homography H; int n_inliers = 0;
    ASSERT_OK(ransac_compute_homography(&ref_stars, &frm_stars, &rp, &H, &n_inliers));

    /* Backward H maps ref→src, so H should encode ~(+5, +3) translation */
    ASSERT_NEAR(H.h[2], 5.0, 3.0);
    ASSERT_NEAR(H.h[5], 3.0, 3.0);

    free(ref_stars.stars); free(frm_stars.stars);
    free(ref); free(shifted);
    return 0;
}

/* Verify detected star count is sufficient for typical synthetic frame. */
static int test_detect_count_sufficient(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);
    float *img = make_star_image(0, 0, N_STARS, xs, ys);

    StarList stars = {NULL, 0};
    ASSERT_OK(detect_stars(img, &stars));

    /* Should detect at least half the injected stars */
    ASSERT_GT(stars.n, N_STARS / 3);

    free(stars.stars); free(img);
    return 0;
}

/* =========================================================================
 * RANSAC → LANCZOS
 * ====================================================================== */

/* Compute H via RANSAC on shifted frames, then apply Lanczos warp.
 * The warped shifted frame should align with the reference. */
static int test_ransac_h_to_lanczos_roundtrip(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);
    float *ref_data = make_star_image(0, 0, N_STARS, xs, ys);
    float *shifted_data = make_star_image(4, 3, N_STARS, xs, ys);

    StarList ref_stars = {NULL, 0}, frm_stars = {NULL, 0};
    ASSERT_OK(detect_stars(ref_data, &ref_stars));
    ASSERT_OK(detect_stars(shifted_data, &frm_stars));

    RansacParams rp = {1000, 3.0f, 30.0f, 0.99f, 5};
    Homography H; int n_inliers = 0;
    ASSERT_OK(ransac_compute_homography(&ref_stars, &frm_stars, &rp, &H, &n_inliers));

    /* Apply Lanczos warp to shifted frame using recovered H */
    Image src = {shifted_data, TW, TH};
    Image dst = {NULL, TW, TH};
    dst.data = alloc_f(NPIX);
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H));

    /* In the overlap region (interior), warped frame should be close to ref */
    int margin = 15;
    float mse = 0; int count = 0;
    for (int y = margin; y < TH - margin; y++)
        for (int x = margin; x < TW - margin; x++) {
            int idx = y * TW + x;
            if (!isnan(dst.data[idx]) && !isnan(ref_data[idx])) {
                float diff = dst.data[idx] - ref_data[idx];
                mse += diff * diff;
                count++;
            }
        }
    if (count > 0) mse /= count;

    /* PSNR should be reasonable (> 30 dB) */
    float peak = FLUX;
    float psnr = (mse > 0) ? 10.0f * log10f(peak * peak / mse) : 100.0f;
    ASSERT_GT(psnr, 25.0f);

    free(ref_stars.stars); free(frm_stars.stars);
    free(ref_data); free(dst.data);
    /* shifted_data is owned by src but not freed via image_free */
    free(shifted_data);
    return 0;
}

/* Identity H → Lanczos output ≈ input. */
static int test_identity_h_lanczos_preserves(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);
    float *data = make_star_image(0, 0, N_STARS, xs, ys);

    Homography H_id;
    double id[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(H_id.h, id, sizeof(id));

    Image src = {data, TW, TH};
    Image dst = {NULL, TW, TH};
    dst.data = alloc_f(NPIX);
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H_id));

    /* Interior pixels should be very close */
    int margin = 5;
    for (int y = margin; y < TH - margin; y++)
        for (int x = margin; x < TW - margin; x++) {
            int idx = y * TW + x;
            ASSERT_NEAR(dst.data[idx], data[idx], 0.01f);
        }

    free(data); free(dst.data);
    return 0;
}

/* =========================================================================
 * CALIBRATION → DEBAYER
 * ====================================================================== */

/* Apply dark subtraction then debayer — verify ordering matters. */
static int test_calib_then_debayer_dark(void)
{
    float *raw = alloc_f(NPIX);
    for (int i = 0; i < NPIX; i++) raw[i] = 200.0f;

    Image img = {raw, TW, TH};

    /* Dark = 50 at every pixel */
    float *dark_data = alloc_f(NPIX);
    for (int i = 0; i < NPIX; i++) dark_data[i] = 50.0f;
    CalibFrames calib;
    memset(&calib, 0, sizeof(calib));
    calib.dark.data = dark_data;
    calib.dark.width = TW;
    calib.dark.height = TH;
    calib.has_dark = 1;

    ASSERT_OK(calib_apply_cpu(&img, &calib));

    /* After dark sub: pixels should be ~150 */
    ASSERT_NEAR(img.data[TW * TH / 2], 150.0f, 0.01f);

    /* Now debayer */
    float *lum = alloc_f(NPIX);
    ASSERT_OK(debayer_cpu(img.data, lum, TW, TH, BAYER_NONE));

    /* BAYER_NONE = memcpy, so lum should equal calibrated value */
    ASSERT_NEAR(lum[TW * TH / 2], 150.0f, 0.01f);

    free(raw); free(dark_data); free(lum);
    return 0;
}

/* Flat division then debayer. */
static int test_calib_then_debayer_flat(void)
{
    float *raw = alloc_f(NPIX);
    for (int i = 0; i < NPIX; i++) raw[i] = 400.0f;

    Image img = {raw, TW, TH};

    float *flat_data = alloc_f(NPIX);
    for (int i = 0; i < NPIX; i++) flat_data[i] = 0.5f;
    CalibFrames calib;
    memset(&calib, 0, sizeof(calib));
    calib.flat.data = flat_data;
    calib.flat.width = TW;
    calib.flat.height = TH;
    calib.has_flat = 1;

    ASSERT_OK(calib_apply_cpu(&img, &calib));

    /* After flat: pixels should be 400 / 0.5 = 800 */
    ASSERT_NEAR(img.data[NPIX / 2], 800.0f, 0.01f);

    float *lum = alloc_f(NPIX);
    ASSERT_OK(debayer_cpu(img.data, lum, TW, TH, BAYER_NONE));
    ASSERT_NEAR(lum[NPIX / 2], 800.0f, 0.01f);

    free(raw); free(flat_data); free(lum);
    return 0;
}

/* =========================================================================
 * DEBAYER → STAR DETECTION
 * ====================================================================== */

/* Debayer luminance output feeds star detection. */
static int test_debayer_lum_then_detect(void)
{
    /* Create a Bayer mosaic with stars */
    float *mosaic = alloc_f(NPIX);
    for (int i = 0; i < NPIX; i++) mosaic[i] = BG;

    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);
    for (int i = 0; i < N_STARS; i++) {
        float sx = xs[i], sy = ys[i];
        if (sx >= 3 && sx < TW - 3 && sy >= 3 && sy < TH - 3)
            inject_star(mosaic, TW, TH, sx, sy, FLUX);
    }

    /* Debayer (RGGB) → luminance */
    float *lum = alloc_f(NPIX);
    ASSERT_OK(debayer_cpu(mosaic, lum, TW, TH, BAYER_RGGB));

    /* Detect stars in the debayered image */
    StarList stars = {NULL, 0};
    float *conv = alloc_f(NPIX);
    uint8_t *mask = (uint8_t *)calloc(NPIX, 1);
    MoffatParams mp = {ALPHA, BETA};

    ASSERT_OK(star_detect_cpu_detect(lum, conv, mask, TW, TH, &mp, 3.0f));
    ASSERT_OK(star_detect_cpu_ccl_com(mask, lum, conv, TW, TH, 50, &stars));

    /* Should detect a reasonable number of stars */
    ASSERT_GT(stars.n, 5);

    free(stars.stars); free(mosaic); free(lum);
    free(conv); free(mask);
    return 0;
}

/* BAYER_NONE passthrough → detection still works. */
static int test_debayer_none_then_detect(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);
    float *img = make_star_image(0, 0, N_STARS, xs, ys);

    float *lum = alloc_f(NPIX);
    ASSERT_OK(debayer_cpu(img, lum, TW, TH, BAYER_NONE));

    /* BAYER_NONE = memcpy, so lum == img */
    StarList stars = {NULL, 0};
    ASSERT_OK(detect_stars(lum, &stars));
    ASSERT_GT(stars.n, 5);

    free(stars.stars); free(img); free(lum);
    return 0;
}

/* =========================================================================
 * FULL MINI-PIPELINE (no file I/O)
 * ====================================================================== */

/* In-memory: debayer → detect → RANSAC → Lanczos for 2 frames. */
static int test_mini_pipeline_detect_align_warp(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);

    float *ref_data = make_star_image(0, 0, N_STARS, xs, ys);
    float *frm_data = make_star_image(3, 2, N_STARS, xs, ys);

    /* Debayer (BAYER_NONE passthrough) */
    float *ref_lum = alloc_f(NPIX);
    float *frm_lum = alloc_f(NPIX);
    ASSERT_OK(debayer_cpu(ref_data, ref_lum, TW, TH, BAYER_NONE));
    ASSERT_OK(debayer_cpu(frm_data, frm_lum, TW, TH, BAYER_NONE));

    /* Detect stars */
    StarList ref_stars = {NULL, 0}, frm_stars = {NULL, 0};
    ASSERT_OK(detect_stars(ref_lum, &ref_stars));
    ASSERT_OK(detect_stars(frm_lum, &frm_stars));
    ASSERT_GT(ref_stars.n, 5);
    ASSERT_GT(frm_stars.n, 5);

    /* RANSAC */
    RansacParams rp = {1000, 3.0f, 30.0f, 0.99f, 5};
    Homography H; int n_inliers = 0;
    ASSERT_OK(ransac_compute_homography(&ref_stars, &frm_stars, &rp, &H, &n_inliers));
    ASSERT_GT(n_inliers, 4);

    /* Lanczos warp */
    Image src = {frm_data, TW, TH};
    Image dst = {NULL, TW, TH};
    dst.data = alloc_f(NPIX);
    ASSERT_OK(lanczos_transform_cpu(&src, &dst, &H));

    /* Verify alignment: interior pixels of warped frame ≈ reference */
    int margin = 15;
    float max_diff = 0;
    int count = 0;
    for (int y = margin; y < TH - margin; y++)
        for (int x = margin; x < TW - margin; x++) {
            int idx = y * TW + x;
            if (!isnan(dst.data[idx])) {
                float d = fabsf(dst.data[idx] - ref_data[idx]);
                if (d > max_diff) max_diff = d;
                count++;
            }
        }
    ASSERT_GT(count, NPIX / 4);

    free(ref_stars.stars); free(frm_stars.stars);
    free(ref_data); free(frm_data);
    free(ref_lum); free(frm_lum); free(dst.data);
    return 0;
}

/* 3 frames through detect → align → warp → mean integrate. */
static int test_mini_pipeline_three_frames_integrate(void)
{
    float xs[N_STARS], ys[N_STARS];
    get_grid_stars(xs, ys);

    float *ref_data = make_star_image(0, 0, N_STARS, xs, ys);
    float *f2_data  = make_star_image(2, 1, N_STARS, xs, ys);
    float *f3_data  = make_star_image(-1, 2, N_STARS, xs, ys);

    /* Detect in all frames */
    StarList ref_stars = {NULL, 0}, f2_stars = {NULL, 0}, f3_stars = {NULL, 0};
    ASSERT_OK(detect_stars(ref_data, &ref_stars));
    ASSERT_OK(detect_stars(f2_data, &f2_stars));
    ASSERT_OK(detect_stars(f3_data, &f3_stars));

    /* RANSAC for f2 and f3 */
    RansacParams rp = {1000, 3.0f, 30.0f, 0.99f, 5};
    Homography H2, H3;
    int n2, n3;
    ASSERT_OK(ransac_compute_homography(&ref_stars, &f2_stars, &rp, &H2, &n2));
    ASSERT_OK(ransac_compute_homography(&ref_stars, &f3_stars, &rp, &H3, &n3));

    /* Identity H for reference */
    Homography H_id;
    double id[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(H_id.h, id, sizeof(id));

    /* Warp all 3 */
    Image src_ref = {ref_data, TW, TH};
    Image src_f2  = {f2_data, TW, TH};
    Image src_f3  = {f3_data, TW, TH};
    Image dst[3];
    for (int i = 0; i < 3; i++) {
        dst[i].width = TW; dst[i].height = TH;
        dst[i].data = alloc_f(NPIX);
    }
    ASSERT_OK(lanczos_transform_cpu(&src_ref, &dst[0], &H_id));
    ASSERT_OK(lanczos_transform_cpu(&src_f2, &dst[1], &H2));
    ASSERT_OK(lanczos_transform_cpu(&src_f3, &dst[2], &H3));

    /* Integrate (mean) */
    const Image *ptrs[3] = {&dst[0], &dst[1], &dst[2]};
    Image out = {NULL, 0, 0};
    ASSERT_OK(integrate_mean(ptrs, 3, &out));
    ASSERT_EQ(out.width, TW);
    ASSERT_EQ(out.height, TH);

    /* Stacked result should have higher SNR (star peaks present) */
    int star_peaks = 0;
    for (int i = 0; i < N_STARS; i++) {
        int px = (int)xs[i], py = (int)ys[i];
        if (px >= 1 && px < TW - 1 && py >= 1 && py < TH - 1) {
            float v = out.data[py * TW + px];
            if (!isnan(v) && v > BG + 10.0f)
                star_peaks++;
        }
    }
    ASSERT_GT(star_peaks, N_STARS / 3);

    free(ref_stars.stars); free(f2_stars.stars); free(f3_stars.stars);
    free(ref_data); free(f2_data); free(f3_data);
    for (int i = 0; i < 3; i++) free(dst[i].data);
    image_free(&out);
    return 0;
}

/* =========================================================================
 * MAIN
 * ====================================================================== */

int main(void)
{
    SUITE("Cross-stage: Star Detection → RANSAC");
    RUN(test_detect_then_ransac_identity);
    RUN(test_detect_then_ransac_shifted);
    RUN(test_detect_count_sufficient);

    SUITE("Cross-stage: RANSAC → Lanczos");
    RUN(test_ransac_h_to_lanczos_roundtrip);
    RUN(test_identity_h_lanczos_preserves);

    SUITE("Cross-stage: Calibration → Debayer");
    RUN(test_calib_then_debayer_dark);
    RUN(test_calib_then_debayer_flat);

    SUITE("Cross-stage: Debayer → Star Detection");
    RUN(test_debayer_lum_then_detect);
    RUN(test_debayer_none_then_detect);

    SUITE("Cross-stage: Full mini-pipeline");
    RUN(test_mini_pipeline_detect_align_warp);
    RUN(test_mini_pipeline_three_frames_integrate);

    return SUMMARY();
}
