/*
 * pipeline_cpu.c — Full CPU DSO stacking pipeline (no CUDA).
 *
 * Implements pipeline_run_cpu(), which mirrors pipeline_run() in pipeline.cu
 * but uses only CPU code.  All computationally intensive steps use OpenMP
 * parallelism via the annotations in debayer_cpu.c, star_detect_cpu.c,
 * lanczos_cpu.c, and integration.c.
 *
 * Data-flow (2-column CSV, full pipeline):
 *   Phase 1 — Star detection and alignment
 *     for each frame:
 *       fits_load → debayer_cpu → star_detect_cpu_detect → star_detect_cpu_ccl_com
 *     for each non-reference frame:
 *       ransac_compute_homography (against reference star list)
 *
 *   Phase 2 — Lanczos transform + integration
 *     for each frame:
 *       fits_load → debayer_cpu → lanczos_transform_cpu → store transformed frame
 *     integrate_kappa_sigma (or integrate_mean) over all transformed frames
 *     fits_save output
 *
 * Data-flow (11-column CSV, pre-computed transforms):
 *   Phase 1 is skipped.  Phase 2 proceeds as above.
 *
 * Memory Optimization:
 *   Previous implementation held all N transformed frames in RAM.  Current
 *   implementation uses mini-batching (default B=32) to cap memory usage.
 *   For kappa-sigma integration, this uses a mini-batch approximation
 *   identical to the GPU pipeline.
 */

#include "pipeline.h"
#include "calibration.h"
#include "image_io.h"
#include "fits_io.h"
#include "debayer_cpu.h"
#include "star_detect_cpu.h"
#include "star_detect_gpu.h"   /* MoffatParams (now from dso_types.h via this header) */
#include "ransac.h"
#include "lanczos_cpu.h"
#include "integration.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define PIPELINE_CPU_BATCH_SIZE 32

/* -------------------------------------------------------------------------
 * Convenience error-check macro: jump to label on non-OK DsoError.
 * Includes the current filename in the error message for context.
 * ------------------------------------------------------------------------- */
#define PIPE_CHECK(call, label, path)                \
    do {                                             \
        DsoError _pe = (call);                       \
        if (_pe != DSO_OK) {                         \
            fprintf(stderr, "pipeline error %d at %s\n", (int)_pe, (path)); \
            err = _pe; goto label;                   \
        }                                            \
    } while (0)

/* -------------------------------------------------------------------------
 * Phase 1: per-frame star detection and RANSAC alignment
 * ------------------------------------------------------------------------- */

static DsoError phase1_cpu(FrameInfo            *frames,
                            int                   n_frames,
                            int                   ref_idx,
                            int                   W, int H,
                            const PipelineConfig *config)
{
    DsoError  err       = DSO_OK;
    long      npix      = (long)W * H;
    StarList *star_lists = NULL;
    float    *conv_buf   = NULL;
    uint8_t  *mask_buf   = NULL;

    star_lists = (StarList *)calloc((size_t)n_frames, sizeof(StarList));
    conv_buf   = (float   *)malloc((size_t)npix * sizeof(float));
    mask_buf   = (uint8_t *)malloc((size_t)npix);
    if (!star_lists || !conv_buf || !mask_buf) {
        err = DSO_ERR_ALLOC;
        goto cleanup;
    }

    /* Detect stars in every frame */
    for (int i = 0; i < n_frames; i++) {
        printf("[Phase 1 CPU] Frame %d/%d: %s\n",
               i + 1, n_frames, frames[i].filepath);

        /* Load FITS */
        Image raw = {NULL, 0, 0};
        PIPE_CHECK(fits_load(frames[i].filepath, &raw), cleanup, frames[i].filepath);
        if (raw.width != W || raw.height != H) {
            fprintf(stderr, "pipeline_cpu phase1: frame %d size mismatch "
                    "(%d×%d vs %d×%d)\n", i, raw.width, raw.height, W, H);
            image_free(&raw);
            err = DSO_ERR_INVALID_ARG;
            goto cleanup;
        }

        /* Apply calibration (dark subtract + flat divide) before debayering */
        if (config->calib) {
            err = calib_apply_cpu(&raw, config->calib);
            if (err != DSO_OK) { 
                fprintf(stderr, "pipeline_cpu phase1: calibration failed for %s\n", frames[i].filepath);
                image_free(&raw); goto cleanup; 
            }
        }

        /* Determine Bayer pattern */
        BayerPattern pat = config->bayer_override;
        if (pat == BAYER_NONE)
            fits_get_bayer_pattern(frames[i].filepath, &pat);

        /* Debayer → luminance */
        Image lum = {NULL, W, H};
        lum.data = (float *)malloc((size_t)npix * sizeof(float));
        if (!lum.data) { image_free(&raw); err = DSO_ERR_ALLOC; goto cleanup; }

        err = debayer_cpu(raw.data, lum.data, W, H, pat);
        image_free(&raw);
        if (err != DSO_OK) { 
            fprintf(stderr, "pipeline_cpu phase1: debayer failed for %s\n", frames[i].filepath);
            image_free(&lum); goto cleanup; 
        }

        /* Cache metadata for Phase 2 */
        frames[i].width   = W;
        frames[i].height  = H;
        frames[i].pattern = (int)pat;

        /* Moffat convolve + threshold */
        err = star_detect_cpu_detect(lum.data, conv_buf, mask_buf, W, H,
                                      &config->moffat, config->star_sigma);
        if (err != DSO_OK) { 
            fprintf(stderr, "pipeline_cpu phase1: detection failed for %s\n", frames[i].filepath);
            image_free(&lum); goto cleanup; 
        }

        /* CCL + weighted CoM */
        err = star_detect_cpu_ccl_com(mask_buf, lum.data, conv_buf, W, H,
                                       config->top_stars, &star_lists[i]);
        image_free(&lum);
        if (err != DSO_OK) {
            fprintf(stderr, "pipeline_cpu phase1: CCL failed for %s\n", frames[i].filepath);
            goto cleanup;
        }
        printf("[Phase 1 CPU] Frame %d: %d star(s) detected\n",
               i, star_lists[i].n);
    }

    /* RANSAC: align each non-reference frame against reference */
    for (int i = 0; i < n_frames; i++) {
        if (i == ref_idx) continue;

        if (star_lists[i].n < config->min_stars ||
            star_lists[ref_idx].n < config->min_stars) {
            fprintf(stderr, "pipeline_cpu phase1: frame %d has too few stars "
                    "(%d) for RANSAC (need %d)\n",
                    i, star_lists[i].n, config->min_stars);
            err = DSO_ERR_STAR_DETECT;
            goto cleanup;
        }

        int n_inliers = 0;
        PIPE_CHECK(ransac_compute_homography(
                       &star_lists[ref_idx], &star_lists[i],
                       &config->ransac, &frames[i].H, &n_inliers), cleanup, frames[i].filepath);

        printf("[Phase 1 CPU] Frame %d: aligned with %d inlier star match(es)\n",
               i, n_inliers);
    }

    /* Reference frame gets identity homography */
    {
        double id[9] = {1,0,0, 0,1,0, 0,0,1};
        memcpy(frames[ref_idx].H.h, id, sizeof(id));
    }

cleanup:
    if (star_lists) {
        for (int i = 0; i < n_frames; i++)
            free(star_lists[i].stars);
        free(star_lists);
    }
    free(conv_buf);
    free(mask_buf);
    return err;
}

/* -------------------------------------------------------------------------
 * Phase 2: Lanczos transform + integration (mini-batched)
 * ------------------------------------------------------------------------- */

static DsoError phase2_cpu(FrameInfo            *frames,
                            int                   n_frames,
                            int                   W, int H,
                            const PipelineConfig *config)
{
    DsoError     err     = DSO_OK;
    long         npix    = (long)W * H;
    int          color   = config->color_output;

    /* Per-channel accumulators (mono uses only _r slot) */
    float *global_sum_r   = NULL;
    float *global_sum_g   = NULL;
    float *global_sum_b   = NULL;
    int   *global_count_r = NULL;
    int   *global_count_g = NULL;
    int   *global_count_b = NULL;

    global_sum_r   = (float *)calloc((size_t)npix, sizeof(float));
    global_count_r = (int   *)calloc((size_t)npix, sizeof(int));
    if (!global_sum_r || !global_count_r) { err = DSO_ERR_ALLOC; goto cleanup; }

    if (color) {
        global_sum_g   = (float *)calloc((size_t)npix, sizeof(float));
        global_sum_b   = (float *)calloc((size_t)npix, sizeof(float));
        global_count_g = (int   *)calloc((size_t)npix, sizeof(int));
        global_count_b = (int   *)calloc((size_t)npix, sizeof(int));
        if (!global_sum_g || !global_sum_b || !global_count_g || !global_count_b) {
            err = DSO_ERR_ALLOC; goto cleanup;
        }
    }

    int batch_size = PIPELINE_CPU_BATCH_SIZE;

    /* For mono: xformed[i] holds the warped luminance frame.
     * For color: xformed_r/g/b[i] hold the warped R/G/B frames. */
    Image       *xformed_r = (Image *)calloc((size_t)batch_size, sizeof(Image));
    Image       *xformed_g = color ? (Image *)calloc((size_t)batch_size, sizeof(Image)) : NULL;
    Image       *xformed_b = color ? (Image *)calloc((size_t)batch_size, sizeof(Image)) : NULL;
    const Image **ptrs     = (const Image **)malloc((size_t)batch_size * sizeof(Image *));
    if (!xformed_r || !ptrs) { err = DSO_ERR_ALLOC; goto cleanup; }
    if (color && (!xformed_g || !xformed_b)) { err = DSO_ERR_ALLOC; goto cleanup; }

    for (int b_start = 0; b_start < n_frames; b_start += batch_size) {
        int b_end = b_start + batch_size;
        if (b_end > n_frames) b_end = n_frames;
        int n_batch = b_end - b_start;

        printf("[Phase 2 CPU] Processing batch %d-%d/%d...\n",
               b_start + 1, b_end, n_frames);

        /* 1. Transform batch frames */
        for (int i = 0; i < n_batch; i++) {
            int f_idx = b_start + i;
            Image raw = {NULL, 0, 0};
            PIPE_CHECK(fits_load(frames[f_idx].filepath, &raw), cleanup, frames[f_idx].filepath);

            if (config->calib) calib_apply_cpu(&raw, config->calib);

            BayerPattern pat = (BayerPattern)frames[f_idx].pattern;
            if (pat == BAYER_NONE && config->bayer_override == BAYER_NONE) {
                fits_get_bayer_pattern(frames[f_idx].filepath, &pat);
            } else if (config->bayer_override != BAYER_NONE) {
                pat = config->bayer_override;
            }

            if (color) {
                /* Debayer → three separate channel planes */
                Image ch_r = {NULL, W, H}, ch_g = {NULL, W, H}, ch_b = {NULL, W, H};
                ch_r.data = (float *)malloc((size_t)npix * sizeof(float));
                ch_g.data = (float *)malloc((size_t)npix * sizeof(float));
                ch_b.data = (float *)malloc((size_t)npix * sizeof(float));
                if (!ch_r.data || !ch_g.data || !ch_b.data) {
                    image_free(&raw); image_free(&ch_r);
                    image_free(&ch_g); image_free(&ch_b);
                    err = DSO_ERR_ALLOC; goto cleanup;
                }
                err = debayer_cpu_rgb(raw.data, ch_r.data, ch_g.data, ch_b.data, W, H, pat);
                image_free(&raw);
                if (err != DSO_OK) {
                    image_free(&ch_r); image_free(&ch_g); image_free(&ch_b);
                    goto cleanup;
                }

                /* Lanczos warp each channel with the same homography */
                xformed_r[i] = (Image){(float *)calloc((size_t)npix, sizeof(float)), W, H};
                xformed_g[i] = (Image){(float *)calloc((size_t)npix, sizeof(float)), W, H};
                xformed_b[i] = (Image){(float *)calloc((size_t)npix, sizeof(float)), W, H};
                if (!xformed_r[i].data || !xformed_g[i].data || !xformed_b[i].data) {
                    image_free(&ch_r); image_free(&ch_g); image_free(&ch_b);
                    err = DSO_ERR_ALLOC; goto cleanup;
                }
                err = lanczos_transform_cpu(&ch_r, &xformed_r[i], &frames[f_idx].H);
                if (err == DSO_OK)
                    err = lanczos_transform_cpu(&ch_g, &xformed_g[i], &frames[f_idx].H);
                if (err == DSO_OK)
                    err = lanczos_transform_cpu(&ch_b, &xformed_b[i], &frames[f_idx].H);
                image_free(&ch_r); image_free(&ch_g); image_free(&ch_b);
                if (err != DSO_OK) {
                    fprintf(stderr, "pipeline_cpu phase2: lanczos failed for %s\n", frames[f_idx].filepath);
                    goto cleanup;
                }

                ptrs[i] = &xformed_r[i]; /* ptrs used for R channel integration */
            } else {
                /* Mono path (unchanged) */
                Image lum = {NULL, W, H};
                lum.data = (float *)malloc((size_t)npix * sizeof(float));
                PIPE_CHECK(debayer_cpu(raw.data, lum.data, W, H, pat), cleanup, frames[f_idx].filepath);
                image_free(&raw);

                xformed_r[i].width  = W;
                xformed_r[i].height = H;
                xformed_r[i].data   = (float *)calloc((size_t)npix, sizeof(float));
                PIPE_CHECK(lanczos_transform_cpu(&lum, &xformed_r[i], &frames[f_idx].H), cleanup, frames[f_idx].filepath);
                image_free(&lum);

                ptrs[i] = &xformed_r[i];
            }
        }

        /* 2. Integrate batch */
        Image b_out_r = {NULL, W, H}, b_out_g = {NULL, W, H}, b_out_b = {NULL, W, H};

        /* R (or mono) */
        if (config->use_kappa_sigma) {
            PIPE_CHECK(integrate_kappa_sigma(ptrs, n_batch, &b_out_r,
                                              config->kappa, config->iterations),
                       cleanup, "batch integration R");
        } else {
            PIPE_CHECK(integrate_mean(ptrs, n_batch, &b_out_r), cleanup, "batch integration R");
        }

        if (color) {
            /* Build ptrs arrays for G and B channels */
            const Image **ptrs_g = (const Image **)malloc((size_t)n_batch * sizeof(Image *));
            const Image **ptrs_b = (const Image **)malloc((size_t)n_batch * sizeof(Image *));
            if (!ptrs_g || !ptrs_b) {
                free(ptrs_g); free(ptrs_b);
                image_free(&b_out_r);
                err = DSO_ERR_ALLOC; goto cleanup;
            }
            for (int i = 0; i < n_batch; i++) {
                ptrs_g[i] = &xformed_g[i];
                ptrs_b[i] = &xformed_b[i];
            }

            if (config->use_kappa_sigma) {
                err = integrate_kappa_sigma(ptrs_g, n_batch, &b_out_g, config->kappa, config->iterations);
                if (err == DSO_OK)
                    err = integrate_kappa_sigma(ptrs_b, n_batch, &b_out_b, config->kappa, config->iterations);
            } else {
                err = integrate_mean(ptrs_g, n_batch, &b_out_g);
                if (err == DSO_OK)
                    err = integrate_mean(ptrs_b, n_batch, &b_out_b);
            }
            free(ptrs_g); free(ptrs_b);
            if (err != DSO_OK) {
                image_free(&b_out_r); goto cleanup;
            }
        }

        /* 3. Accumulate batch into global */
#pragma omp parallel for schedule(static)
        for (long p = 0; p < npix; p++) {
            global_sum_r[p]   += b_out_r.data[p] * n_batch;
            global_count_r[p] += n_batch;
        }
        image_free(&b_out_r);

        if (color) {
#pragma omp parallel for schedule(static)
            for (long p = 0; p < npix; p++) {
                global_sum_g[p]   += b_out_g.data[p] * n_batch;
                global_count_g[p] += n_batch;
                global_sum_b[p]   += b_out_b.data[p] * n_batch;
                global_count_b[p] += n_batch;
            }
            image_free(&b_out_g);
            image_free(&b_out_b);
        }

        /* 4. Free batch transformed frames */
        for (int i = 0; i < n_batch; i++) {
            image_free(&xformed_r[i]);
            if (color) {
                image_free(&xformed_g[i]);
                image_free(&xformed_b[i]);
            }
        }
    }

    /* Finalize */
    if (color) {
        Image img_r = {NULL, W, H}, img_g = {NULL, W, H}, img_b = {NULL, W, H};
        img_r.data = (float *)malloc((size_t)npix * sizeof(float));
        img_g.data = (float *)malloc((size_t)npix * sizeof(float));
        img_b.data = (float *)malloc((size_t)npix * sizeof(float));
        if (!img_r.data || !img_g.data || !img_b.data) {
            image_free(&img_r); image_free(&img_g); image_free(&img_b);
            err = DSO_ERR_ALLOC; goto cleanup;
        }
#pragma omp parallel for schedule(static)
        for (long p = 0; p < npix; p++) {
            img_r.data[p] = (global_count_r[p] > 0) ? (global_sum_r[p] / global_count_r[p]) : 0.0f;
            img_g.data[p] = (global_count_g[p] > 0) ? (global_sum_g[p] / global_count_g[p]) : 0.0f;
            img_b.data[p] = (global_count_b[p] > 0) ? (global_sum_b[p] / global_count_b[p]) : 0.0f;
        }
        err = image_save_rgb(config->output_file, &img_r, &img_g, &img_b, &config->save_opts);
        image_free(&img_r); image_free(&img_g); image_free(&img_b);
        if (err != DSO_OK)
            fprintf(stderr, "pipeline_cpu phase2: image_save_rgb failed for %s\n", config->output_file);
    } else {
        Image out = {NULL, W, H};
        out.data = (float *)malloc((size_t)npix * sizeof(float));
        if (!out.data) { err = DSO_ERR_ALLOC; goto cleanup; }
#pragma omp parallel for schedule(static)
        for (long p = 0; p < npix; p++) {
            out.data[p] = (global_count_r[p] > 0) ? (global_sum_r[p] / global_count_r[p]) : 0.0f;
        }
        PIPE_CHECK(image_save(config->output_file, &out, &config->save_opts), cleanup, config->output_file);
        image_free(&out);
    }

cleanup:
    free(global_sum_r); free(global_sum_g); free(global_sum_b);
    free(global_count_r); free(global_count_g); free(global_count_b);
    if (xformed_r) {
        for (int i = 0; i < batch_size; i++) image_free(&xformed_r[i]);
        free(xformed_r);
    }
    if (xformed_g) {
        for (int i = 0; i < batch_size; i++) image_free(&xformed_g[i]);
        free(xformed_g);
    }
    if (xformed_b) {
        for (int i = 0; i < batch_size; i++) image_free(&xformed_b[i]);
        free(xformed_b);
    }
    free(ptrs);
    return err;
}

/* -------------------------------------------------------------------------
 * Public entry point
 * ------------------------------------------------------------------------- */

DsoError pipeline_run_cpu(FrameInfo            *frames,
                           int                   n_frames,
                           int                   has_transforms,
                           int                   ref_idx,
                           const PipelineConfig *config)
{
    if (!frames || n_frames <= 0 || !config ||
        ref_idx < 0 || ref_idx >= n_frames) return DSO_ERR_INVALID_ARG;

    /* Determine output dimensions from the reference frame */
    int W = 0, H = 0;
    {
        Image ref_img = {NULL, 0, 0};
        DsoError e = fits_load(frames[ref_idx].filepath, &ref_img);
        if (e != DSO_OK) return e;
        W = ref_img.width;
        H = ref_img.height;
        image_free(&ref_img);
    }

    printf("pipeline_cpu: output dimensions %d × %d, %d frame(s)\n",
           W, H, n_frames);

    DsoError err = DSO_OK;

    /* Phase 1: star detection + RANSAC (only when transforms are not given) */
    if (!has_transforms) {
        printf("pipeline_cpu: Phase 1 — star detection (%d frames)\n", n_frames);
        err = phase1_cpu(frames, n_frames, ref_idx, W, H, config);
        if (err != DSO_OK) return err;
        printf("pipeline_cpu: Phase 1 complete\n");
    } else {
        /* Reference frame needs identity homography if not already set */
        double id[9] = {1,0,0, 0,1,0, 0,0,1};
        memcpy(frames[ref_idx].H.h, id, sizeof(id));
    }

    /* Phase 2: transform + integrate */
    printf("pipeline_cpu: Phase 2 — transform + %s integration (%d frames)\n",
           config->use_kappa_sigma ? "kappa-sigma" : "mean", n_frames);
    err = phase2_cpu(frames, n_frames, W, H, config);
    if (err != DSO_OK) return err;

    printf("pipeline_cpu: saving to %s\n", config->output_file);
    printf("pipeline_cpu: done.\n");
    return DSO_OK;
}
