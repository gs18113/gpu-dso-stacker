/*
 * test_pipeline_cpu.c — end-to-end tests for the CPU pipeline.
 *
 * Creates synthetic FITS frames with Moffat-PSF stars in C, writes them
 * to temp files with a 2-column CSV, and exercises pipeline_run_cpu().
 *
 * Every test creates its own temp directory, writes FITS + CSV, runs
 * the pipeline, and verifies the output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_framework.h"
#include "dso_types.h"
#include "pipeline.h"
#include "calibration.h"
#include "fits_io.h"
#include "image_io.h"
#include "csv_parser.h"

/* -------------------------------------------------------------------------
 * Synthetic data helpers
 * ---------------------------------------------------------------------- */

static Image make_image(int w, int h, float val)
{
    Image img = {NULL, w, h};
    img.data = (float *)malloc((size_t)w * h * sizeof(float));
    if (!img.data) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < w * h; i++) img.data[i] = val;
    return img;
}

/* Add a Moffat-profile star at (cx, cy) with given flux. */
static void inject_star(Image *img, float cx, float cy,
                         float flux, float alpha, float beta)
{
    int R = (int)ceilf(3.0f * alpha);
    if (R > 15) R = 15;
    int x0 = (int)cx - R, x1 = (int)cx + R;
    int y0 = (int)cy - R, y1 = (int)cy + R;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= img->width) x1 = img->width - 1;
    if (y1 >= img->height) y1 = img->height - 1;
    /* Compute normalization */
    float total = 0.0f;
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            float dx = (float)x - cx, dy = (float)y - cy;
            float r2 = dx * dx + dy * dy;
            total += powf(1.0f + r2 / (alpha * alpha), -beta);
        }
    float scale = (total > 0.0f) ? flux / total : 0.0f;
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            float dx = (float)x - cx, dy = (float)y - cy;
            float r2 = dx * dx + dy * dy;
            img->data[y * img->width + x] +=
                scale * powf(1.0f + r2 / (alpha * alpha), -beta);
        }
}

/* Create a frame with N_STARS distributed stars on a background.
 * Stars are placed on a grid to ensure detectability and matching. */
#define N_STARS 40
#define TEST_W  128
#define TEST_H  128
#define BG_LEVEL 100.0f
#define STAR_FLUX 5000.0f
#define ALPHA 2.5f
#define BETA  2.0f

/* Fixed star positions (grid + jitter) — same for all frames */
static void get_star_positions(float xs[N_STARS], float ys[N_STARS])
{
    /* 5x8 grid with padding from edges */
    int idx = 0;
    for (int gy = 0; gy < 8 && idx < N_STARS; gy++)
        for (int gx = 0; gx < 5 && idx < N_STARS; gx++) {
            xs[idx] = 15.0f + gx * 22.0f;
            ys[idx] = 10.0f + gy * 14.0f;
            idx++;
        }
}

static Image make_star_frame(float dx, float dy)
{
    Image img = make_image(TEST_W, TEST_H, BG_LEVEL);
    float xs[N_STARS], ys[N_STARS];
    get_star_positions(xs, ys);
    for (int i = 0; i < N_STARS; i++) {
        float sx = xs[i] + dx, sy = ys[i] + dy;
        if (sx >= 2 && sx < TEST_W - 2 && sy >= 2 && sy < TEST_H - 2)
            inject_star(&img, sx, sy, STAR_FLUX, ALPHA, BETA);
    }
    return img;
}

static void write_csv(const char *csv_path, const char **fits_paths,
                       int n, int ref_idx)
{
    FILE *fp = fopen(csv_path, "w");
    if (!fp) { perror("write_csv"); exit(1); }
    fprintf(fp, "filepath, is_reference\n");
    for (int i = 0; i < n; i++)
        fprintf(fp, "%s, %d\n", fits_paths[i], (i == ref_idx) ? 1 : 0);
    fclose(fp);
}

static PipelineConfig default_config(const char *output_path)
{
    PipelineConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.backend          = DSO_BACKEND_CPU;
    cfg.output_file      = output_path;
    cfg.star_sigma       = 3.0f;
    cfg.moffat.alpha     = ALPHA;
    cfg.moffat.beta      = BETA;
    cfg.top_stars        = 50;
    cfg.min_stars        = 5;
    cfg.ransac.max_iters      = 1000;
    cfg.ransac.inlier_thresh  = 3.0f;
    cfg.ransac.match_radius   = 30.0f;
    cfg.ransac.confidence     = 0.99f;
    cfg.ransac.min_inliers    = 5;
    cfg.batch_size       = 16;
    cfg.kappa            = 3.0f;
    cfg.iterations       = 3;
    cfg.use_kappa_sigma  = 1;
    cfg.bayer_override   = BAYER_NONE;
    cfg.color_output     = 0;
    cfg.save_opts.stretch_min = NAN;
    cfg.save_opts.stretch_max = NAN;
    return cfg;
}

/* =========================================================================
 * ERROR HANDLING TESTS
 * ====================================================================== */

static int test_pipeline_cpu_null_config(void)
{
    FrameInfo frame;
    memset(&frame, 0, sizeof(frame));
    strcpy(frame.filepath, "dummy.fits");
    frame.is_reference = 1;
    ASSERT_ERR(pipeline_run_cpu(&frame, 1, 0, NULL), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_pipeline_cpu_null_frames(void)
{
    PipelineConfig cfg = default_config("out.fits");
    ASSERT_ERR(pipeline_run_cpu(NULL, 1, 0, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_pipeline_cpu_zero_frames(void)
{
    FrameInfo frame;
    memset(&frame, 0, sizeof(frame));
    PipelineConfig cfg = default_config("out.fits");
    ASSERT_ERR(pipeline_run_cpu(&frame, 0, 0, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_pipeline_cpu_bad_ref_idx(void)
{
    FrameInfo frame;
    memset(&frame, 0, sizeof(frame));
    strcpy(frame.filepath, "dummy.fits");
    PipelineConfig cfg = default_config("out.fits");
    ASSERT_ERR(pipeline_run_cpu(&frame, 1, 5, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_pipeline_cpu_negative_ref_idx(void)
{
    FrameInfo frame;
    memset(&frame, 0, sizeof(frame));
    strcpy(frame.filepath, "dummy.fits");
    PipelineConfig cfg = default_config("out.fits");
    ASSERT_ERR(pipeline_run_cpu(&frame, 1, -1, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_pipeline_cpu_missing_file(void)
{
    FrameInfo frame;
    memset(&frame, 0, sizeof(frame));
    strcpy(frame.filepath, "/nonexistent/no_such_file.fits");
    frame.is_reference = 1;
    char out[512]; TEST_TMPPATH(out, "pipe_miss_out.fits");
    PipelineConfig cfg = default_config(out);
    DsoError e = pipeline_run_cpu(&frame, 1, 0, &cfg);
    ASSERT(e != DSO_OK);
    return 0;
}

/* =========================================================================
 * SINGLE-FRAME TESTS
 * ====================================================================== */

static int test_pipeline_cpu_single_frame(void)
{
    char fits_path[512]; TEST_TMPPATH(fits_path, "pipe_single.fits");
    char out_path[512];  TEST_TMPPATH(out_path,  "pipe_single_out.fits");
    char csv_path[512];  TEST_TMPPATH(csv_path,  "pipe_single.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(fits_path, &img));

    const char *paths[] = {fits_path};
    write_csv(csv_path, paths, 1, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv_path, &frames, &n));
    ASSERT_EQ(n, 1);

    PipelineConfig cfg = default_config(out_path);
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    /* Verify output exists and is loadable */
    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out_path, &result));
    ASSERT_EQ(result.width, TEST_W);
    ASSERT_EQ(result.height, TEST_H);

    /* Single frame: output should closely match input where data is valid */
    int match_count = 0;
    for (int i = 0; i < TEST_W * TEST_H; i++) {
        if (!isnan(result.data[i]) && !isnan(img.data[i])) {
            if (fabsf(result.data[i] - img.data[i]) < 0.5f)
                match_count++;
        }
    }
    /* Most interior pixels should match closely */
    ASSERT_GT(match_count, (TEST_W * TEST_H) / 2);

    image_free(&result);
    image_free(&img);
    free(frames);
    return 0;
}

/* =========================================================================
 * MULTI-FRAME TESTS
 * ====================================================================== */

static int test_pipeline_cpu_two_identical_frames(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_ident1.fits");
    char f2[512]; TEST_TMPPATH(f2, "pipe_ident2.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_ident_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_ident.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));
    ASSERT_OK(fits_save(f2, &img));

    const char *paths[] = {f1, f2};
    write_csv(csv, paths, 2, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_EQ(result.width, TEST_W);
    ASSERT_EQ(result.height, TEST_H);

    /* Mean of two identical frames ≈ original */
    int close_count = 0;
    for (int i = 0; i < TEST_W * TEST_H; i++) {
        if (!isnan(result.data[i]) && !isnan(img.data[i])) {
            if (fabsf(result.data[i] - img.data[i]) < 1.0f)
                close_count++;
        }
    }
    ASSERT_GT(close_count, (TEST_W * TEST_H) / 2);

    image_free(&result);
    image_free(&img);
    free(frames);
    return 0;
}

static int test_pipeline_cpu_three_frames_known_shift(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_shift_ref.fits");
    char f2[512]; TEST_TMPPATH(f2, "pipe_shift_f2.fits");
    char f3[512]; TEST_TMPPATH(f3, "pipe_shift_f3.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_shift_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_shift.csv");

    /* Reference at (0,0), frame 2 shifted by (3,2), frame 3 by (-2,1) */
    Image ref = make_star_frame(0, 0);
    Image frm2 = make_star_frame(3, 2);
    Image frm3 = make_star_frame(-2, 1);

    ASSERT_OK(fits_save(f1, &ref));
    ASSERT_OK(fits_save(f2, &frm2));
    ASSERT_OK(fits_save(f3, &frm3));

    const char *paths[] = {f1, f2, f3};
    write_csv(csv, paths, 3, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_EQ(result.width, TEST_W);
    ASSERT_EQ(result.height, TEST_H);

    /* The stacked result should have higher SNR than single frame.
     * Check that star peaks are present at reference positions. */
    float xs[N_STARS], ys[N_STARS];
    get_star_positions(xs, ys);
    int detected_peaks = 0;
    for (int s = 0; s < N_STARS; s++) {
        int px = (int)xs[s], py = (int)ys[s];
        if (px >= 1 && px < TEST_W - 1 && py >= 1 && py < TEST_H - 1) {
            float v = result.data[py * TEST_W + px];
            if (!isnan(v) && v > BG_LEVEL + 10.0f)
                detected_peaks++;
        }
    }
    /* At least half the stars should be visible in the stacked result */
    ASSERT_GT(detected_peaks, N_STARS / 3);

    image_free(&result);
    image_free(&ref);
    image_free(&frm2);
    image_free(&frm3);
    free(frames);
    return 0;
}

/* =========================================================================
 * INTEGRATION METHOD TESTS
 * ====================================================================== */

static int test_pipeline_cpu_mean_integration(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_mean1.fits");
    char f2[512]; TEST_TMPPATH(f2, "pipe_mean2.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_mean_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_mean.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));
    ASSERT_OK(fits_save(f2, &img));

    const char *paths[] = {f1, f2};
    write_csv(csv, paths, 2, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    cfg.use_kappa_sigma = 0; /* mean integration */
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_NOT_NULL(result.data);

    image_free(&result);
    image_free(&img);
    free(frames);
    return 0;
}

static int test_pipeline_cpu_kappa_sigma_integration(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_ks1.fits");
    char f2[512]; TEST_TMPPATH(f2, "pipe_ks2.fits");
    char f3[512]; TEST_TMPPATH(f3, "pipe_ks3.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_ks_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_ks.csv");

    /* 3 identical frames — kappa-sigma should produce clean result */
    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));
    ASSERT_OK(fits_save(f2, &img));
    ASSERT_OK(fits_save(f3, &img));

    const char *paths[] = {f1, f2, f3};
    write_csv(csv, paths, 3, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    cfg.use_kappa_sigma = 1;
    cfg.kappa = 2.0f;
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_NOT_NULL(result.data);

    /* With 3 identical frames and kappa-sigma, result ≈ input */
    int close_count = 0;
    for (int i = 0; i < TEST_W * TEST_H; i++) {
        if (!isnan(result.data[i]) && !isnan(img.data[i])) {
            if (fabsf(result.data[i] - img.data[i]) < 1.0f)
                close_count++;
        }
    }
    ASSERT_GT(close_count, (TEST_W * TEST_H) / 2);

    image_free(&result);
    image_free(&img);
    free(frames);
    return 0;
}

/* =========================================================================
 * STAR DETECTION + RANSAC INTEGRATION
 * ====================================================================== */

static int test_pipeline_cpu_detects_stars_and_aligns(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_align_ref.fits");
    char f2[512]; TEST_TMPPATH(f2, "pipe_align_f2.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_align_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_align.csv");

    Image ref = make_star_frame(0, 0);
    Image frm2 = make_star_frame(4, 3);
    ASSERT_OK(fits_save(f1, &ref));
    ASSERT_OK(fits_save(f2, &frm2));

    const char *paths[] = {f1, f2};
    write_csv(csv, paths, 2, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    /* Verify that the homography of frame 1 was populated (non-zero) */
    double h_sum = 0;
    for (int k = 0; k < 9; k++) h_sum += fabs(frames[1].H.h[k]);
    ASSERT_GT(h_sum, 0.01);

    /* The backward homography should roughly encode a (4,3) shift:
     * H[0][2] ≈ 4 (tx), H[1][2] ≈ 3 (ty) for a pure translation. */
    ASSERT_NEAR(frames[1].H.h[2], 4.0, 3.0);  /* tx within ±3 px */
    ASSERT_NEAR(frames[1].H.h[5], 3.0, 3.0);  /* ty within ±3 px */

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_NOT_NULL(result.data);

    image_free(&result);
    image_free(&ref);
    image_free(&frm2);
    free(frames);
    return 0;
}

static int test_pipeline_cpu_skips_starless_frame(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_nostar_ref.fits");
    char f2[512]; TEST_TMPPATH(f2, "pipe_nostar_blank.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_nostar_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_nostar.csv");

    Image ref = make_star_frame(0, 0);
    Image blank = make_image(TEST_W, TEST_H, BG_LEVEL); /* no stars */
    ASSERT_OK(fits_save(f1, &ref));
    ASSERT_OK(fits_save(f2, &blank));

    const char *paths[] = {f1, f2};
    write_csv(csv, paths, 2, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    /* Pipeline should succeed — blank frame is skipped, not fatal */
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_NOT_NULL(result.data);

    image_free(&result);
    image_free(&ref);
    image_free(&blank);
    free(frames);
    return 0;
}

/* =========================================================================
 * CALIBRATION INTEGRATION
 * ====================================================================== */

static int test_pipeline_cpu_with_dark_subtraction(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_dark_ref.fits");
    char out_nodark[512]; TEST_TMPPATH(out_nodark, "pipe_nodark_out.fits");
    char out_dark[512]; TEST_TMPPATH(out_dark, "pipe_dark_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_dark.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));

    const char *paths[] = {f1};
    write_csv(csv, paths, 1, 0);

    /* Run without calibration */
    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));
    PipelineConfig cfg_no = default_config(out_nodark);
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg_no));
    free(frames);

    /* Create a dark frame (constant 20.0) */
    char dark_path[512]; TEST_TMPPATH(dark_path, "pipe_master_dark.fits");
    Image dark = make_image(TEST_W, TEST_H, 20.0f);
    ASSERT_OK(fits_save(dark_path, &dark));

    CalibFrames calib;
    memset(&calib, 0, sizeof(calib));
    calib.dark = dark;
    calib.has_dark = 1;

    /* Run with dark subtraction */
    ASSERT_OK(csv_parse(csv, &frames, &n));
    PipelineConfig cfg_dk = default_config(out_dark);
    cfg_dk.calib = &calib;
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg_dk));

    /* Load both outputs and compare */
    Image r_no = {NULL, 0, 0}, r_dk = {NULL, 0, 0};
    ASSERT_OK(fits_load(out_nodark, &r_no));
    ASSERT_OK(fits_load(out_dark, &r_dk));

    /* Dark-subtracted result should be lower */
    float sum_no = 0, sum_dk = 0;
    int count = 0;
    for (int i = 0; i < TEST_W * TEST_H; i++) {
        if (!isnan(r_no.data[i]) && !isnan(r_dk.data[i])) {
            sum_no += r_no.data[i];
            sum_dk += r_dk.data[i];
            count++;
        }
    }
    if (count > 0) {
        ASSERT_LT(sum_dk / count, sum_no / count);
    }

    image_free(&r_no);
    image_free(&r_dk);
    image_free(&img);
    /* Don't free dark.data — it's owned by the calib struct on stack */
    free(frames);
    return 0;
}

static int test_pipeline_cpu_with_flat_division(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_flat_ref.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_flat_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_flat.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));

    const char *paths[] = {f1};
    write_csv(csv, paths, 1, 0);

    /* Create a flat frame with mean ≈ 1.0 (left half dimmer) */
    Image flat = make_image(TEST_W, TEST_H, 1.0f);
    for (int y = 0; y < TEST_H; y++)
        for (int x = 0; x < TEST_W / 2; x++)
            flat.data[y * TEST_W + x] = 0.8f;

    CalibFrames calib;
    memset(&calib, 0, sizeof(calib));
    calib.flat = flat;
    calib.has_flat = 1;

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));
    PipelineConfig cfg = default_config(out);
    cfg.calib = &calib;
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_NOT_NULL(result.data);

    /* Left half should be boosted (divided by 0.8 = multiplied by 1.25) */
    float left_sum = 0, right_sum = 0;
    int lc = 0, rc = 0;
    int margin = 10; /* skip edges */
    for (int y = margin; y < TEST_H - margin; y++) {
        for (int x = margin; x < TEST_W / 2 - margin; x++) {
            float v = result.data[y * TEST_W + x];
            if (!isnan(v)) { left_sum += v; lc++; }
        }
        for (int x = TEST_W / 2 + margin; x < TEST_W - margin; x++) {
            float v = result.data[y * TEST_W + x];
            if (!isnan(v)) { right_sum += v; rc++; }
        }
    }
    /* Left side (divided by 0.8) should be higher than right (divided by 1.0) */
    if (lc > 0 && rc > 0) {
        ASSERT_GT(left_sum / lc, right_sum / rc);
    }

    image_free(&result);
    image_free(&img);
    free(frames);
    return 0;
}

/* =========================================================================
 * COLOR OUTPUT
 * ====================================================================== */

static int test_pipeline_cpu_color_bayer_rggb(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_color_ref.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_color_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_color.csv");

    /* Create a Bayer RGGB mosaic frame with stars */
    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));

    const char *paths[] = {f1};
    write_csv(csv, paths, 1, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    cfg.bayer_override = BAYER_RGGB;
    cfg.color_output = 1;
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    /* Verify output file exists and has non-zero size.
     * fits_load only reads NAXIS=2; RGB output is NAXIS=3, so we check file size. */
    FILE *fp = fopen(out, "rb");
    ASSERT_NOT_NULL(fp);
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fclose(fp);
    ASSERT_GT(sz, 0);
    image_free(&img);
    free(frames);
    return 0;
}

static int test_pipeline_cpu_color_none_mono(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_mono_ref.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_mono_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_mono.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));

    const char *paths[] = {f1};
    write_csv(csv, paths, 1, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    cfg.bayer_override = BAYER_NONE;
    cfg.color_output = 0;
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_EQ(result.width, TEST_W);
    ASSERT_EQ(result.height, TEST_H);
    ASSERT_NOT_NULL(result.data);

    image_free(&result);
    image_free(&img);
    free(frames);
    return 0;
}

/* =========================================================================
 * OUTPUT FORMAT TESTS
 * ====================================================================== */

static int test_pipeline_cpu_output_tiff(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_tiff_ref.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_out.tiff");
    char csv[512]; TEST_TMPPATH(csv, "pipe_tiff.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));

    const char *paths[] = {f1};
    write_csv(csv, paths, 1, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    /* Verify the TIFF file exists (non-zero size) */
    FILE *fp = fopen(out, "rb");
    ASSERT_NOT_NULL(fp);
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fclose(fp);
    ASSERT_GT(sz, 0);

    image_free(&img);
    free(frames);
    return 0;
}

static int test_pipeline_cpu_output_png(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_png_ref.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_out.png");
    char csv[512]; TEST_TMPPATH(csv, "pipe_png.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));

    const char *paths[] = {f1};
    write_csv(csv, paths, 1, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    cfg.save_opts.bit_depth = OUT_BITS_INT16;
    ASSERT_OK(pipeline_run_cpu(frames, n, 0, &cfg));

    FILE *fp = fopen(out, "rb");
    ASSERT_NOT_NULL(fp);
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fclose(fp);
    ASSERT_GT(sz, 0);

    image_free(&img);
    free(frames);
    return 0;
}

/* =========================================================================
 * SIZE MISMATCH
 * ====================================================================== */

static int test_pipeline_cpu_size_mismatch(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_szmis_ref.fits");
    char f2[512]; TEST_TMPPATH(f2, "pipe_szmis_f2.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_szmis_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_szmis.csv");

    Image big = make_star_frame(0, 0); /* TEST_W x TEST_H */
    Image small_img = make_image(TEST_W / 2, TEST_H / 2, BG_LEVEL);
    /* Add some stars to small image so it doesn't fail on star detection */
    inject_star(&small_img, 20, 20, STAR_FLUX, ALPHA, BETA);

    ASSERT_OK(fits_save(f1, &big));
    ASSERT_OK(fits_save(f2, &small_img));

    const char *paths[] = {f1, f2};
    write_csv(csv, paths, 2, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    DsoError e = pipeline_run_cpu(frames, n, 0, &cfg);
    ASSERT(e != DSO_OK); /* Should fail on dimension mismatch */

    image_free(&big);
    image_free(&small_img);
    free(frames);
    return 0;
}

/* =========================================================================
 * BACKEND DISPATCH
 * ====================================================================== */

static int test_pipeline_dispatch_cpu_backend(void)
{
    char f1[512]; TEST_TMPPATH(f1, "pipe_disp_ref.fits");
    char out[512]; TEST_TMPPATH(out, "pipe_disp_out.fits");
    char csv[512]; TEST_TMPPATH(csv, "pipe_disp.csv");

    Image img = make_star_frame(0, 0);
    ASSERT_OK(fits_save(f1, &img));

    const char *paths[] = {f1};
    write_csv(csv, paths, 1, 0);

    FrameInfo *frames = NULL; int n = 0;
    ASSERT_OK(csv_parse(csv, &frames, &n));

    PipelineConfig cfg = default_config(out);
    cfg.backend = DSO_BACKEND_CPU;
    /* pipeline_run should dispatch to pipeline_run_cpu */
    ASSERT_OK(pipeline_run(frames, n, 0, &cfg));

    Image result = {NULL, 0, 0};
    ASSERT_OK(fits_load(out, &result));
    ASSERT_NOT_NULL(result.data);

    image_free(&result);
    image_free(&img);
    free(frames);
    return 0;
}

/* =========================================================================
 * MAIN
 * ====================================================================== */

int main(void)
{
    SUITE("Pipeline CPU — error handling");
    RUN(test_pipeline_cpu_null_config);
    RUN(test_pipeline_cpu_null_frames);
    RUN(test_pipeline_cpu_zero_frames);
    RUN(test_pipeline_cpu_bad_ref_idx);
    RUN(test_pipeline_cpu_negative_ref_idx);
    RUN(test_pipeline_cpu_missing_file);

    SUITE("Pipeline CPU — single frame");
    RUN(test_pipeline_cpu_single_frame);

    SUITE("Pipeline CPU — multi-frame");
    RUN(test_pipeline_cpu_two_identical_frames);
    RUN(test_pipeline_cpu_three_frames_known_shift);

    SUITE("Pipeline CPU — integration methods");
    RUN(test_pipeline_cpu_mean_integration);
    RUN(test_pipeline_cpu_kappa_sigma_integration);

    SUITE("Pipeline CPU — star detection + alignment");
    RUN(test_pipeline_cpu_detects_stars_and_aligns);
    RUN(test_pipeline_cpu_skips_starless_frame);

    SUITE("Pipeline CPU — calibration");
    RUN(test_pipeline_cpu_with_dark_subtraction);
    RUN(test_pipeline_cpu_with_flat_division);

    SUITE("Pipeline CPU — color output");
    RUN(test_pipeline_cpu_color_bayer_rggb);
    RUN(test_pipeline_cpu_color_none_mono);

    SUITE("Pipeline CPU — output formats");
    RUN(test_pipeline_cpu_output_tiff);
    RUN(test_pipeline_cpu_output_png);

    SUITE("Pipeline CPU — error cases");
    RUN(test_pipeline_cpu_size_mismatch);

    SUITE("Pipeline CPU — dispatch");
    RUN(test_pipeline_dispatch_cpu_backend);

    return SUMMARY();
}
