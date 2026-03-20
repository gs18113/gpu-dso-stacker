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
 * Memory:
 *   All N transformed frames are held in RAM simultaneously before integration.
 *   For the reference frame the identity transform produces a direct copy.
 */

#include "pipeline.h"
#include "calibration.h"
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

/* -------------------------------------------------------------------------
 * Convenience error-check macro: jump to label on non-OK DsoError.
 * ------------------------------------------------------------------------- */
#define PIPE_CHECK(call, label)                      \
    do {                                             \
        DsoError _pe = (call);                       \
        if (_pe != DSO_OK) { err = _pe; goto label; }\
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
        PIPE_CHECK(fits_load(frames[i].filepath, &raw), cleanup);
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
            if (err != DSO_OK) { image_free(&raw); goto cleanup; }
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
        if (err != DSO_OK) { image_free(&lum); goto cleanup; }

        /* Moffat convolve + threshold */
        err = star_detect_cpu_detect(lum.data, conv_buf, mask_buf, W, H,
                                      &config->moffat, config->star_sigma);
        if (err != DSO_OK) { image_free(&lum); goto cleanup; }

        /* CCL + weighted CoM */
        err = star_detect_cpu_ccl_com(mask_buf, lum.data, conv_buf, W, H,
                                       config->top_stars, &star_lists[i]);
        image_free(&lum);
        if (err != DSO_OK) goto cleanup;
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
                       &config->ransac, &frames[i].H, &n_inliers), cleanup);

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
 * Phase 2: Lanczos transform + integration
 * ------------------------------------------------------------------------- */

static DsoError phase2_cpu(FrameInfo            *frames,
                            int                   n_frames,
                            int                   W, int H,
                            const PipelineConfig *config)
{
    DsoError     err     = DSO_OK;
    long         npix    = (long)W * H;
    Image       *xformed = NULL;   /* array of n_frames transformed Images */
    const Image **ptrs   = NULL;   /* pointer array for integrate_* */
    Image        out     = {NULL, 0, 0};

    xformed = (Image *)calloc((size_t)n_frames, sizeof(Image));
    ptrs    = (const Image **)malloc((size_t)n_frames * sizeof(Image *));
    if (!xformed || !ptrs) { err = DSO_ERR_ALLOC; goto cleanup; }

    /* Transform every frame */
    for (int i = 0; i < n_frames; i++) {
        Image raw = {NULL, 0, 0};
        Image lum = {NULL, W, H};

        PIPE_CHECK(fits_load(frames[i].filepath, &raw), cleanup);
        if (raw.width != W || raw.height != H) {
            fprintf(stderr, "pipeline_cpu phase2: frame %d size mismatch\n", i);
            image_free(&raw);
            err = DSO_ERR_INVALID_ARG;
            goto cleanup;
        }

        /* Apply calibration (dark subtract + flat divide) before debayering */
        if (config->calib) {
            err = calib_apply_cpu(&raw, config->calib);
            if (err != DSO_OK) { image_free(&raw); goto cleanup; }
        }

        /* Bayer pattern */
        BayerPattern pat = config->bayer_override;
        if (pat == BAYER_NONE)
            fits_get_bayer_pattern(frames[i].filepath, &pat);

        /* Debayer → luminance */
        lum.data = (float *)malloc((size_t)npix * sizeof(float));
        if (!lum.data) { image_free(&raw); err = DSO_ERR_ALLOC; goto cleanup; }

        err = debayer_cpu(raw.data, lum.data, W, H, pat);
        image_free(&raw);
        if (err != DSO_OK) { image_free(&lum); goto cleanup; }

        /* Allocate output frame (zero-filled; lanczos leaves OOB pixels unset) */
        xformed[i].data   = (float *)calloc((size_t)npix, sizeof(float));
        xformed[i].width  = W;
        xformed[i].height = H;
        if (!xformed[i].data) { image_free(&lum); err = DSO_ERR_ALLOC; goto cleanup; }

        /* Lanczos-3 warp using backward homography (ref → src) */
        err = lanczos_transform_cpu(&lum, &xformed[i], &frames[i].H);
        image_free(&lum);
        if (err != DSO_OK) goto cleanup;

        ptrs[i] = &xformed[i];
    }

    /* Integration */
    if (config->use_kappa_sigma) {
        PIPE_CHECK(integrate_kappa_sigma(ptrs, n_frames, &out,
                                          config->kappa, config->iterations),
                   cleanup);
    } else {
        PIPE_CHECK(integrate_mean(ptrs, n_frames, &out), cleanup);
    }

    /* Save result */
    PIPE_CHECK(fits_save(config->output_file, &out), cleanup);

cleanup:
    image_free(&out);
    if (xformed) {
        for (int i = 0; i < n_frames; i++)
            image_free(&xformed[i]);
        free(xformed);
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
