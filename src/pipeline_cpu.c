/*
 * pipeline_cpu.c — Full CPU DSO stacking pipeline (no CUDA).
 *
 * Single-pass streaming: each frame is read from disk exactly once, then
 * processed end-to-end before the next frame is loaded.
 *
 *   fits_load → calib → debayer_lum → star detect → [triangle matching] → Lanczos warp
 *   → accumulate into mini-batch → integrate batch when full → repeat
 *
 * Frame ordering: reference frame processed first (builds ref_stars for triangle matching
 * of all subsequent frames), then all non-reference frames in CSV order.
 *
 * Memory at any point:
 *   · One raw frame  (freed after debayer)
 *   · Up to batch_size warped frames  (freed after each batch integration)
 *   · Two fixed scratch buffers (conv_buf, mask_buf)  reused per frame
 *   · Global sum/count accumulators  (always resident)
 */

#include "pipeline.h"
#include "calibration.h"
#include "image_io.h"
#include "fits_io.h"
#include "frame_load.h"
#include "debayer_cpu.h"
#include "star_detect_cpu.h"
#include "ransac.h"
#include "lanczos_cpu.h"
#include "integration.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define PIPELINE_CPU_BATCH_SIZE 32

#define PIPE_CHECK(call, label, path)                \
    do {                                             \
        DsoError _pe = (call);                       \
        if (_pe != DSO_OK) {                         \
            fprintf(stderr, "pipeline error %d at %s\n", (int)_pe, (path)); \
            err = _pe; goto label;                   \
        }                                            \
    } while (0)

/* -------------------------------------------------------------------------
 * flush_batch — integrate and accumulate the current mini-batch.
 * All batch frames are freed on return (success or error).
 * ------------------------------------------------------------------------- */
static DsoError flush_batch(
    Image        *xformed_r, Image *xformed_g, Image *xformed_b,
    int           n_batch, int color, int W, int H, long npix,
    float kappa, int iterations, int use_kappa_sigma,
    const Image **ptrs_r, const Image **ptrs_g, const Image **ptrs_b,
    float *global_sum_r, float *global_sum_g, float *global_sum_b,
    int   *global_count_r, int   *global_count_g, int   *global_count_b)
{
    int      i;
    long     pp;
    DsoError err = DSO_OK;

    Image b_out_r = {NULL, W, H};
    Image b_out_g = {NULL, W, H};
    Image b_out_b = {NULL, W, H};

    for (i = 0; i < n_batch; i++) ptrs_r[i] = &xformed_r[i];

    if (use_kappa_sigma == 1) {
        PIPE_CHECK(integrate_kappa_sigma(ptrs_r, n_batch, &b_out_r, kappa, iterations),
                   cleanup, "batch integration R");
    } else if (use_kappa_sigma == 2) {
        PIPE_CHECK(integrate_aawa(ptrs_r, n_batch, &b_out_r),
                   cleanup, "batch integration R");
    } else {
        PIPE_CHECK(integrate_mean(ptrs_r, n_batch, &b_out_r), cleanup, "batch integration R");
    }

    if (color) {
        for (i = 0; i < n_batch; i++) { ptrs_g[i] = &xformed_g[i]; ptrs_b[i] = &xformed_b[i]; }
        if (use_kappa_sigma == 1) {
            err = integrate_kappa_sigma(ptrs_g, n_batch, &b_out_g, kappa, iterations);
            if (err == DSO_OK)
                err = integrate_kappa_sigma(ptrs_b, n_batch, &b_out_b, kappa, iterations);
        } else if (use_kappa_sigma == 2) {
            err = integrate_aawa(ptrs_g, n_batch, &b_out_g);
            if (err == DSO_OK)
                err = integrate_aawa(ptrs_b, n_batch, &b_out_b);
        } else {
            err = integrate_mean(ptrs_g, n_batch, &b_out_g);
            if (err == DSO_OK)
                err = integrate_mean(ptrs_b, n_batch, &b_out_b);
        }
        if (err != DSO_OK) { image_free(&b_out_r); goto cleanup; }
    }

    /* Count per-pixel valid (non-NaN) frames in this batch.  R/G/B share
     * the same homography so their OOB patterns are identical — count once. */
    {
        int *batch_valid = (int *)calloc((size_t)npix, sizeof(int));
        if (!batch_valid) { err = DSO_ERR_ALLOC; image_free(&b_out_r);
                            if (color) { image_free(&b_out_g); image_free(&b_out_b); }
                            goto cleanup; }
#pragma omp parallel for schedule(static)
        for (pp = 0; pp < npix; pp++) {
            int vc = 0;
            for (int fi = 0; fi < n_batch; fi++)
                if (!isnan(xformed_r[fi].data[pp])) vc++;
            batch_valid[pp] = vc;
        }

#pragma omp parallel for schedule(static)
        for (pp = 0; pp < npix; pp++) {
            int vc = batch_valid[pp];
            if (vc > 0 && !isnan(b_out_r.data[pp])) {
                global_sum_r[pp]   += b_out_r.data[pp] * vc;
                global_count_r[pp] += vc;
            }
        }
        image_free(&b_out_r);

        if (color) {
#pragma omp parallel for schedule(static)
            for (pp = 0; pp < npix; pp++) {
                int vc = batch_valid[pp];
                if (vc > 0) {
                    if (!isnan(b_out_g.data[pp])) {
                        global_sum_g[pp]   += b_out_g.data[pp] * vc;
                        global_count_g[pp] += vc;
                    }
                    if (!isnan(b_out_b.data[pp])) {
                        global_sum_b[pp]   += b_out_b.data[pp] * vc;
                        global_count_b[pp] += vc;
                    }
                }
            }
            image_free(&b_out_g);
            image_free(&b_out_b);
        }
        free(batch_valid);
    }

cleanup:
    for (i = 0; i < n_batch; i++) {
        image_free(&xformed_r[i]);
        if (color) { image_free(&xformed_g[i]); image_free(&xformed_b[i]); }
    }
    return err;
}

/* -------------------------------------------------------------------------
 * Public entry point
 * ------------------------------------------------------------------------- */

DsoError pipeline_run_cpu(FrameInfo            *frames,
                           int                   n_frames,
                           int                   ref_idx,
                           const PipelineConfig *config)
{
    if (!frames || n_frames <= 0 || !config ||
        ref_idx < 0 || ref_idx >= n_frames) return DSO_ERR_INVALID_ARG;

    /* Determine output dimensions from the reference frame */
    int W = 0, H = 0;
    {
        Image ref_img = {NULL, 0, 0};
        DsoError e = frame_load(frames[ref_idx].filepath, &ref_img);
        if (e != DSO_OK) return e;
        W = ref_img.width;
        H = ref_img.height;
        image_free(&ref_img);
    }

    printf("pipeline_cpu: output dimensions %d × %d, %d frame(s)\n", W, H, n_frames);

    DsoError  err        = DSO_OK;
    long      npix       = (long)W * H;
    long      pp;
    int       color      = config->color_output;
    int       batch_size = PIPELINE_CPU_BATCH_SIZE;
    int       batch_n    = 0;
    int       successful_frames = 0;
    int       skipped_frames    = 0;

    /* Shared scratch (reused per frame) */
    float   *conv_buf    = NULL;
    uint8_t *mask_buf    = NULL;

    /* Reference star list kept until all frames are processed */
    StarList  ref_stars  = {NULL, 0};

    /* Batch buffers */
    Image       *xformed_r = NULL;
    Image       *xformed_g = NULL;
    Image       *xformed_b = NULL;
    const Image **ptrs_r   = NULL;
    const Image **ptrs_g   = NULL;
    const Image **ptrs_b   = NULL;

    /* Global accumulators */
    float *global_sum_r   = NULL;
    float *global_sum_g   = NULL;
    float *global_sum_b   = NULL;
    int   *global_count_r = NULL;
    int   *global_count_g = NULL;
    int   *global_count_b = NULL;

    /* ---- Allocations ---- */
    conv_buf = (float   *)malloc((size_t)npix * sizeof(float));
    mask_buf = (uint8_t *)malloc((size_t)npix);
    if (!conv_buf || !mask_buf) { err = DSO_ERR_ALLOC; goto cleanup; }

    xformed_r = (Image *)calloc((size_t)batch_size, sizeof(Image));
    ptrs_r    = (const Image **)malloc((size_t)batch_size * sizeof(Image *));
    if (!xformed_r || !ptrs_r) { err = DSO_ERR_ALLOC; goto cleanup; }

    if (color) {
        xformed_g = (Image *)calloc((size_t)batch_size, sizeof(Image));
        xformed_b = (Image *)calloc((size_t)batch_size, sizeof(Image));
        ptrs_g    = (const Image **)malloc((size_t)batch_size * sizeof(Image *));
        ptrs_b    = (const Image **)malloc((size_t)batch_size * sizeof(Image *));
        if (!xformed_g || !xformed_b || !ptrs_g || !ptrs_b) {
            err = DSO_ERR_ALLOC; goto cleanup;
        }
    }

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

    /* Identity homography for the reference frame */
    {
        double id[9] = {1,0,0, 0,1,0, 0,0,1};
        memcpy(frames[ref_idx].H.h, id, sizeof(id));
    }

    /* ---- Build visit order: reference frame first, then non-reference ---- */
    {
        int *order = (int *)malloc((size_t)n_frames * sizeof(int));
        if (!order) { err = DSO_ERR_ALLOC; goto cleanup; }

        int pos = 0;
        order[pos++] = ref_idx;
        for (int k = 0; k < n_frames; k++) {
            if (k != ref_idx) order[pos++] = k;
        }

        for (pos = 0; pos < n_frames; pos++) {
            int i = order[pos];

            printf("[Pipeline CPU] Frame %d/%d: %s\n",
                   pos + 1, n_frames, frames[i].filepath);

            /* ---- Load ---- */
            Image raw = {NULL, 0, 0};
            PIPE_CHECK(frame_load(frames[i].filepath, &raw),
                       order_cleanup, frames[i].filepath);
            if (raw.width != W || raw.height != H) {
                fprintf(stderr,
                        "pipeline_cpu: frame %d size mismatch (%d×%d vs %d×%d)\n",
                        i, raw.width, raw.height, W, H);
                image_free(&raw); err = DSO_ERR_INVALID_ARG;
                free(order); goto cleanup;
            }

            /* ---- Calibration ---- */
            if (config->calib) {
                err = calib_apply_cpu(&raw, config->calib);
                if (err != DSO_OK) {
                    fprintf(stderr, "pipeline_cpu: calibration failed for %s\n",
                            frames[i].filepath);
                    image_free(&raw); free(order); goto cleanup;
                }
            }

            /* ---- Bayer pattern ---- */
            BayerPattern pat = config->bayer_override;
            if (pat == BAYER_NONE)
                frame_get_bayer_pattern(frames[i].filepath, &pat);
            frames[i].width   = W;
            frames[i].height  = H;
            frames[i].pattern = (int)pat;

            /* ---- Debayer → luminance (for star detection) ---- */
            Image lum = {NULL, W, H};
            lum.data = (float *)malloc((size_t)npix * sizeof(float));
            if (!lum.data) {
                image_free(&raw); err = DSO_ERR_ALLOC;
                free(order); goto cleanup;
            }
            err = debayer_cpu(raw.data, lum.data, W, H, pat);
            if (err != DSO_OK) {
                image_free(&lum); image_free(&raw);
                free(order); goto cleanup;
            }
            /* For mono, raw is no longer needed after luminance debayer */
            if (!color) { image_free(&raw); }

            /* ---- Star detection ---- */
            err = star_detect_cpu_detect(lum.data, conv_buf, mask_buf,
                                          W, H, &config->moffat,
                                          config->star_sigma);
            if (err != DSO_OK) {
                image_free(&lum);
                if (color) image_free(&raw);
                free(order); goto cleanup;
            }
            StarList stars = {NULL, 0};
            err = star_detect_cpu_ccl_com(mask_buf, lum.data, conv_buf,
                                           W, H, config->top_stars, &stars);
            if (err != DSO_OK) {
                image_free(&lum);
                if (color) image_free(&raw);
                free(order); goto cleanup;
            }
            printf("[Pipeline CPU] Frame %d: %d star(s) detected\n", i, stars.n);

            if (i == ref_idx) {
                ref_stars = stars;
            } else {
                /* ---- Triangle matching against reference ---- */
                if (stars.n < config->min_stars ||
                    ref_stars.n < config->min_stars) {
                    fprintf(stderr,
                            "pipeline_cpu: skipping frame %d/%d "
                            "(csv index=%d, path=%s): insufficient stars for triangle matching "
                            "(ref=%d, frame=%d, min=%d)\n",
                            pos + 1, n_frames, i + 1, frames[i].filepath,
                            ref_stars.n, stars.n, config->min_stars);
                    free(stars.stars); image_free(&lum);
                    if (color) image_free(&raw);
                    skipped_frames++;
                    err = DSO_OK;
                    continue;
                }
                int n_inliers = 0;
                err = ransac_compute_homography(&ref_stars, &stars,
                                                 &config->ransac,
                                                 &frames[i].H, &n_inliers);
                free(stars.stars);
                if (err != DSO_OK) {
                    fprintf(stderr,
                            "pipeline_cpu: skipping frame %d/%d "
                            "(csv index=%d, path=%s): triangle-matching mismatch (err=%d)\n",
                            pos + 1, n_frames, i + 1, frames[i].filepath, (int)err);
                    image_free(&lum);
                    if (color) image_free(&raw);
                    skipped_frames++;
                    err = DSO_OK;
                    continue;
                }
                printf("[Pipeline CPU] Frame %d/%d (csv index=%d): aligned with %d inlier(s)\n",
                       pos + 1, n_frames, i + 1, n_inliers);
            }

            /* ---- Warp ---- */
            int slot = batch_n;
            if (color) {
                Image ch_r = {NULL, W, H};
                Image ch_g = {NULL, W, H};
                Image ch_b = {NULL, W, H};
                ch_r.data = (float *)malloc((size_t)npix * sizeof(float));
                ch_g.data = (float *)malloc((size_t)npix * sizeof(float));
                ch_b.data = (float *)malloc((size_t)npix * sizeof(float));
                if (!ch_r.data || !ch_g.data || !ch_b.data) {
                    image_free(&lum); image_free(&raw);
                    image_free(&ch_r); image_free(&ch_g); image_free(&ch_b);
                    err = DSO_ERR_ALLOC; free(order); goto cleanup;
                }
                err = debayer_cpu_rgb(raw.data, ch_r.data, ch_g.data,
                                       ch_b.data, W, H, pat);
                image_free(&raw);
                image_free(&lum);
                if (err != DSO_OK) {
                    image_free(&ch_r); image_free(&ch_g); image_free(&ch_b);
                    free(order); goto cleanup;
                }
                xformed_r[slot].data = (float *)calloc((size_t)npix, sizeof(float));
                xformed_g[slot].data = (float *)calloc((size_t)npix, sizeof(float));
                xformed_b[slot].data = (float *)calloc((size_t)npix, sizeof(float));
                xformed_r[slot].width = W; xformed_r[slot].height = H;
                xformed_g[slot].width = W; xformed_g[slot].height = H;
                xformed_b[slot].width = W; xformed_b[slot].height = H;
                if (!xformed_r[slot].data || !xformed_g[slot].data ||
                    !xformed_b[slot].data) {
                    image_free(&ch_r); image_free(&ch_g); image_free(&ch_b);
                    err = DSO_ERR_ALLOC; free(order); goto cleanup;
                }
                err = lanczos_transform_cpu(&ch_r, &xformed_r[slot], &frames[i].H);
                if (err == DSO_OK)
                    err = lanczos_transform_cpu(&ch_g, &xformed_g[slot], &frames[i].H);
                if (err == DSO_OK)
                    err = lanczos_transform_cpu(&ch_b, &xformed_b[slot], &frames[i].H);
                image_free(&ch_r); image_free(&ch_g); image_free(&ch_b);
            } else {
                xformed_r[slot].data   = (float *)calloc((size_t)npix, sizeof(float));
                xformed_r[slot].width  = W;
                xformed_r[slot].height = H;
                if (!xformed_r[slot].data) {
                    image_free(&lum); err = DSO_ERR_ALLOC;
                    free(order); goto cleanup;
                }
                err = lanczos_transform_cpu(&lum, &xformed_r[slot], &frames[i].H);
                image_free(&lum);
            }
            if (err != DSO_OK) {
                fprintf(stderr, "pipeline_cpu: lanczos failed for %s\n",
                        frames[i].filepath);
                free(order); goto cleanup;
            }

            batch_n++;
            successful_frames++;

            /* ---- Flush batch when full ---- */
            if (batch_n == batch_size) {
                printf("[Pipeline CPU] Integrating batch of %d frames...\n", batch_size);
                err = flush_batch(xformed_r, xformed_g, xformed_b,
                                  batch_size, color, W, H, npix,
                                  config->kappa, config->iterations,
                                  config->use_kappa_sigma,
                                  ptrs_r, ptrs_g, ptrs_b,
                                  global_sum_r, global_sum_g, global_sum_b,
                                  global_count_r, global_count_g, global_count_b);
                if (err != DSO_OK) { free(order); goto cleanup; }
                batch_n = 0;
            }

            continue;
        order_cleanup:
            free(order);
            goto cleanup;
        }

        free(order);
    }

    /* ---- Flush final partial batch ---- */
    if (batch_n > 0) {
        printf("[Pipeline CPU] Integrating final batch of %d frame(s)...\n", batch_n);
        err = flush_batch(xformed_r, xformed_g, xformed_b,
                          batch_n, color, W, H, npix,
                          config->kappa, config->iterations,
                          config->use_kappa_sigma,
                          ptrs_r, ptrs_g, ptrs_b,
                          global_sum_r, global_sum_g, global_sum_b,
                          global_count_r, global_count_g, global_count_b);
        if (err != DSO_OK) goto cleanup;
        batch_n = 0;
    }

    /* ---- Finalize: divide accumulators and save ---- */
    printf("pipeline_cpu: saving to %s\n", config->output_file);
    {
        if (color) {
            Image img_r = {NULL, W, H};
            Image img_g = {NULL, W, H};
            Image img_b = {NULL, W, H};
            img_r.data = (float *)malloc((size_t)npix * sizeof(float));
            img_g.data = (float *)malloc((size_t)npix * sizeof(float));
            img_b.data = (float *)malloc((size_t)npix * sizeof(float));
            if (!img_r.data || !img_g.data || !img_b.data) {
                image_free(&img_r); image_free(&img_g); image_free(&img_b);
                err = DSO_ERR_ALLOC; goto cleanup;
            }
#pragma omp parallel for schedule(static)
            for (pp = 0; pp < npix; pp++) {
                img_r.data[pp] = (global_count_r[pp] > 0) ?
                                 (global_sum_r[pp] / global_count_r[pp]) : 0.0f;
                img_g.data[pp] = (global_count_g[pp] > 0) ?
                                 (global_sum_g[pp] / global_count_g[pp]) : 0.0f;
                img_b.data[pp] = (global_count_b[pp] > 0) ?
                                 (global_sum_b[pp] / global_count_b[pp]) : 0.0f;
            }
            err = image_save_rgb(config->output_file, &img_r, &img_g, &img_b,
                                 &config->save_opts);
            image_free(&img_r); image_free(&img_g); image_free(&img_b);
            if (err != DSO_OK)
                fprintf(stderr, "pipeline_cpu: image_save_rgb failed for %s\n",
                        config->output_file);
        } else {
            Image out = {NULL, W, H};
            out.data = (float *)malloc((size_t)npix * sizeof(float));
            if (!out.data) { err = DSO_ERR_ALLOC; goto cleanup; }
#pragma omp parallel for schedule(static)
            for (pp = 0; pp < npix; pp++) {
                out.data[pp] = (global_count_r[pp] > 0) ?
                               (global_sum_r[pp] / global_count_r[pp]) : 0.0f;
            }
            PIPE_CHECK(image_save(config->output_file, &out, &config->save_opts),
                       save_cleanup, config->output_file);
        save_cleanup:
            image_free(&out);
        }
    }

    if (err == DSO_OK)
        printf("pipeline_cpu: done. successful frames: %d/%d (skipped: %d)\n",
               successful_frames, n_frames, skipped_frames);

cleanup:
    free(ref_stars.stars);
    free(conv_buf);
    free(mask_buf);
    free(global_sum_r); free(global_sum_g); free(global_sum_b);
    free(global_count_r); free(global_count_g); free(global_count_b);
    if (xformed_r) {
        int j;
        for (j = 0; j < batch_size; j++) image_free(&xformed_r[j]);
        free(xformed_r);
    }
    if (xformed_g) {
        int j;
        for (j = 0; j < batch_size; j++) image_free(&xformed_g[j]);
        free(xformed_g);
    }
    if (xformed_b) {
        int j;
        for (j = 0; j < batch_size; j++) image_free(&xformed_b[j]);
        free(xformed_b);
    }
    free(ptrs_r); free(ptrs_g); free(ptrs_b);
    return err;
}
