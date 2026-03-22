/*
 * pipeline.cu — Full DSO stacking pipeline orchestrator.
 *
 * Two-phase execution:
 *
 *   Phase 1 — Star detection and RANSAC alignment (only when the input CSV
 *   does not contain pre-computed homographies).  For each frame:
 *     1. Load FITS → upload to device
 *     2. VNG debayer → luminance
 *     3. Moffat convolution + sigma threshold → binary mask
 *     4. D2H mask + debayered + convolved images
 *     5. CPU CCL + weighted CoM → StarList
 *   Then RANSAC each non-reference frame's star list against the reference
 *   to compute the backward homography H[i] (ref → src).
 *   Phase 1 is sequential (one frame at a time) because it is a one-time
 *   setup cost dominated by disk I/O.
 *
 *   Phase 2 — Lanczos alignment + GPU integration.  Uses two CUDA streams
 *   and double-buffered pinned host memory to overlap CPU disk reads with
 *   GPU processing:
 *
 *     stream_copy (B)  : H2D DMA transfers only
 *     stream_compute (A): debayer → Lanczos → integration kernels
 *
 *   Within each mini-batch of M frames:
 *     For frame m:
 *       stream_compute: waitEvent(e_h2d[slot]); debayer; lanczos_d2d;
 *                       recordEvent(e_gpu[slot])
 *       CPU:            [while GPU runs] fits_load next frame → pinned[next_slot]
 *       stream_copy:    H2D pinned[next_slot] → d_raw[next_slot]
 *       CPU:            waitEvent(e_gpu[next_slot])  [instant — prev batch]
 *     After batch: integration_gpu_process_batch
 *
 *   The key overlap: CPU disk I/O for frame m+1 runs concurrently with GPU
 *   processing frame m on stream_compute.
 */

#include "pipeline.h"
#include "calibration.h"
#include "calibration_gpu.h"
#include "fits_io.h"
#include "debayer_gpu.h"
#include "star_detect_gpu.h"
#include "star_detect_cpu.h"
#include "ransac.h"
#include "integration_gpu.h"
#include "lanczos_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* -------------------------------------------------------------------------
 * Internal error-check helpers
 * ------------------------------------------------------------------------- */

/* Wrap a DsoError call; jump to label on failure. */
#define PIPE_CHECK(call, label, path)                                \
    do {                                                             \
        DsoError _pe = (call);                                       \
        if (_pe != DSO_OK) {                                         \
            fprintf(stderr, "pipeline error %d at %s\n", (int)_pe, (path)); \
            err = _pe; goto label;                                   \
        }                                                            \
    } while (0)

/* Wrap a CUDA call; print message, set err, jump to label on failure. */
#define CUDA_CHECK(call, label, path)                                \
    do {                                                             \
        cudaError_t _ce = (call);                                    \
        if (_ce != cudaSuccess) {                                    \
            fprintf(stderr, "pipeline CUDA error %s:%d at %s — %s\n", \
                    __FILE__, __LINE__, (path), cudaGetErrorString(_ce)); \
            err = DSO_ERR_CUDA; goto label;                         \
        }                                                            \
    } while (0)

/* -------------------------------------------------------------------------
 * Phase 1: per-frame star detection (sequential, no stream overlap)
 *
 * For each frame: upload raw → debayer (d2d) → Moffat+threshold (d2d)
 * → D2H three host arrays → CCL+CoM on CPU.
 * After all frames: RANSAC each non-reference frame against ref_stars.
 * ------------------------------------------------------------------------- */

static DsoError phase1_detect_stars(
    FrameInfo            *frames,
    int                 n_frames,
    int                 ref_idx,
    int                 W, int H,
    const PipelineConfig *config,
    CalibGpuCtx        *calib_ctx,    /* NULL = no calibration */
    StarList           *star_lists)   /* caller-allocated array of n_frames StarList */
{
    DsoError  err           = DSO_OK;
    size_t    npix_f        = (size_t)W * H * sizeof(float);
    size_t    npix_b        = (size_t)W * H;

    /* Device scratch buffers (reused for every frame) */
    float   *d_raw          = NULL;
    float   *d_debayed      = NULL;
    float   *d_conv         = NULL;
    uint8_t *d_mask         = NULL;

    /* Host buffers for CCL input (allocated once, reused per frame) */
    float   *original_host  = NULL;
    float   *convolved_host = NULL;
    uint8_t *mask_host      = NULL;

    CUDA_CHECK(cudaMalloc(&d_raw,     npix_f), cleanup, "d_raw");
    CUDA_CHECK(cudaMalloc(&d_debayed, npix_f), cleanup, "d_debayed");
    CUDA_CHECK(cudaMalloc(&d_conv,    npix_f), cleanup, "d_conv");
    CUDA_CHECK(cudaMalloc(&d_mask,    npix_b), cleanup, "d_mask");

    original_host  = (float   *)malloc(npix_f);
    convolved_host = (float   *)malloc(npix_f);
    mask_host      = (uint8_t *)malloc(npix_b);
    if (!original_host || !convolved_host || !mask_host) {
        err = DSO_ERR_ALLOC; goto cleanup;
    }

    for (int i = 0; i < n_frames; i++) {
        printf("[Phase 1] Frame %d/%d: %s\n", i + 1, n_frames, frames[i].filepath);

        /* Load raw FITS frame */
        Image raw = {NULL, 0, 0};
        PIPE_CHECK(fits_load(frames[i].filepath, &raw), cleanup, frames[i].filepath);
        if (raw.width != W || raw.height != H) {
            fprintf(stderr, "pipeline: frame %d size %dx%d != ref %dx%d\n",
                    i, raw.width, raw.height, W, H);
            image_free(&raw); err = DSO_ERR_INVALID_ARG; goto cleanup;
        }

        /* Determine Bayer pattern (override or auto-detect from FITS header) */
        BayerPattern pat = config->bayer_override;
        if (pat == BAYER_NONE)
            fits_get_bayer_pattern(frames[i].filepath, &pat);

        /* Cache metadata for Phase 2 */
        frames[i].width   = W;
        frames[i].height  = H;
        frames[i].pattern = (int)pat;

        /* H2D raw → device */
        CUDA_CHECK(cudaMemcpy(d_raw, raw.data, npix_f, cudaMemcpyHostToDevice), frame_err, frames[i].filepath);

        /* Apply calibration (dark subtract + flat divide) before debayering */
        PIPE_CHECK(calib_gpu_apply_d2d(d_raw, W, H, calib_ctx, 0), frame_err, frames[i].filepath);

        /* Debayer → luminance */
        PIPE_CHECK(debayer_gpu_d2d(d_raw, d_debayed, W, H, pat, 0), frame_err, frames[i].filepath);

        /* Moffat convolution + sigma threshold → d_conv, d_mask */
        PIPE_CHECK(star_detect_gpu_d2d(d_debayed, d_conv, d_mask,
                                        W, H, &config->moffat,
                                        config->star_sigma, 0), frame_err, frames[i].filepath);

        /* D2H: three arrays needed by CCL+CoM */
        CUDA_CHECK(cudaMemcpy(original_host,  d_debayed, npix_f,
                              cudaMemcpyDeviceToHost), frame_err, frames[i].filepath);
        CUDA_CHECK(cudaMemcpy(convolved_host, d_conv,    npix_f,
                              cudaMemcpyDeviceToHost), frame_err, frames[i].filepath);
        CUDA_CHECK(cudaMemcpy(mask_host,      d_mask,    npix_b,
                              cudaMemcpyDeviceToHost), frame_err, frames[i].filepath);

        /* CPU: connected-component labeling + weighted center-of-mass */
        PIPE_CHECK(star_detect_cpu_ccl_com(mask_host, original_host, convolved_host,
                                            W, H, config->top_stars,
                                            &star_lists[i]), frame_err, frames[i].filepath);
        printf("[Phase 1] Frame %d: %d star(s) detected\n", i, star_lists[i].n);

        image_free(&raw);
        continue;

    frame_err:
        image_free(&raw);
        goto cleanup;
    }

    /* ---- RANSAC: compute H[i] (ref → frame i) for every non-reference frame ---- */

    /* Validate reference frame has enough stars */
    if (star_lists[ref_idx].n < config->min_stars) {
        fprintf(stderr, "pipeline: reference frame %d has only %d star(s) "
                        "(min_stars = %d)\n",
                ref_idx, star_lists[ref_idx].n, config->min_stars);
        err = DSO_ERR_STAR_DETECT;
        goto cleanup;
    }

    for (int i = 0; i < n_frames; i++) {
        if (i == ref_idx) {
            /* Reference frame: identity backward homography */
            memset(&frames[i].H, 0, sizeof(Homography));
            frames[i].H.h[0] = frames[i].H.h[4] = frames[i].H.h[8] = 1.0;
            continue;
        }

        if (star_lists[i].n < config->min_stars) {
            fprintf(stderr, "pipeline: frame %d has only %d star(s) "
                            "(min_stars = %d) — cannot align\n",
                    i, star_lists[i].n, config->min_stars);
            err = DSO_ERR_STAR_DETECT;
            goto cleanup;
        }

        int n_inliers = 0;
        err = ransac_compute_homography(&star_lists[ref_idx], &star_lists[i],
                                         &config->ransac,
                                         &frames[i].H, &n_inliers);
        if (err != DSO_OK) {
            fprintf(stderr, "pipeline: RANSAC failed for frame %d (err = %d)\n",
                    i, (int)err);
            goto cleanup;
        }
        printf("[Phase 1] Frame %d: aligned with %d inlier star match(es)\n",
               i, n_inliers);
    }

cleanup:
    cudaFree(d_raw); cudaFree(d_debayed);
    cudaFree(d_conv); cudaFree(d_mask);
    free(original_host); free(convolved_host); free(mask_host);
    return err;
}

/* -------------------------------------------------------------------------
 * Phase 2: Lanczos transform + GPU integration with stream overlap.
 *
 * Uses double-buffered pinned host memory and two CUDA streams so that
 * CPU disk I/O for frame m+1 overlaps with GPU processing frame m.
 * All frames in a mini-batch are transformed before calling process_batch.
 * ------------------------------------------------------------------------- */

/*
 * phase2_transform_integrate — Lanczos + integration with stream overlap.
 *
 * For mono (ctx_g == NULL): debayer → luminance → single Lanczos → ctx_r.
 * For color (ctx_g != NULL): debayer → RGB → 3× Lanczos → ctx_r/g/b.
 *
 * The stream_copy / stream_compute double-buffer overlap is fully preserved
 * in both paths. e_gpu[slot] is recorded after the last Lanczos call for
 * each slot, so the H2D of the next frame on stream_copy overlaps with all
 * three Lanczos warps on stream_compute in color mode.
 */
static DsoError phase2_transform_integrate(
    FrameInfo            *frames,
    int                   n_frames,
    int                   W, int H,
    const PipelineConfig *config,
    CalibGpuCtx          *calib_ctx,   /* NULL = no calibration */
    IntegrationGpuCtx    *ctx_r,       /* always non-NULL (mono or R channel) */
    IntegrationGpuCtx    *ctx_g,       /* NULL = mono mode */
    IntegrationGpuCtx    *ctx_b)       /* NULL = mono mode */
{
    DsoError     err            = DSO_OK;
    size_t       npix_f        = (size_t)W * H * sizeof(float);
    int          color          = (ctx_g != NULL);
    cudaStream_t stream_copy   = 0;
    cudaStream_t stream_compute = 0;
    cudaEvent_t  e_h2d[2]      = {0, 0};
    cudaEvent_t  e_gpu[2]      = {0, 0};
    float       *pinned[2]     = {NULL, NULL};
    float       *d_raw[2]      = {NULL, NULL};
    float       *d_debayed     = NULL;   /* mono luminance or R channel */
    float       *d_ch_g        = NULL;   /* G channel (color only) */
    float       *d_ch_b        = NULL;   /* B channel (color only) */
    int          processed     = 0;

    CUDA_CHECK(cudaStreamCreate(&stream_copy),    cleanup, "stream_copy");
    CUDA_CHECK(cudaStreamCreate(&stream_compute), cleanup, "stream_compute");

    for (int j = 0; j < 2; j++) {
        CUDA_CHECK(cudaEventCreate(&e_h2d[j]), cleanup, "e_h2d");
        CUDA_CHECK(cudaEventCreate(&e_gpu[j]),  cleanup, "e_gpu");
    }

    CUDA_CHECK(cudaMallocHost(&pinned[0], npix_f), cleanup, "pinned[0]");
    CUDA_CHECK(cudaMallocHost(&pinned[1], npix_f), cleanup, "pinned[1]");
    CUDA_CHECK(cudaMalloc(&d_raw[0], npix_f), cleanup, "d_raw[0]");
    CUDA_CHECK(cudaMalloc(&d_raw[1], npix_f), cleanup, "d_raw[1]");
    CUDA_CHECK(cudaMalloc(&d_debayed, npix_f), cleanup, "d_debayed");

    if (color) {
        CUDA_CHECK(cudaMalloc(&d_ch_g, npix_f), cleanup, "d_ch_g");
        CUDA_CHECK(cudaMalloc(&d_ch_b, npix_f), cleanup, "d_ch_b");
    }

    while (processed < n_frames) {
        int M           = n_frames - processed;
        if (M > config->batch_size) M = config->batch_size;
        int batch_start = processed;

        /* -----------------------------------------------------------------
         * Kickstart: load frame 0 of this batch into pinned[0], start H2D.
         * ----------------------------------------------------------------- */
        PIPE_CHECK(fits_load_to_buffer(frames[batch_start].filepath, pinned[0], W, H), cleanup, frames[batch_start].filepath);
        CUDA_CHECK(cudaMemcpyAsync(d_raw[0], pinned[0], npix_f,
                                   cudaMemcpyHostToDevice, stream_copy), cleanup, frames[batch_start].filepath);
        CUDA_CHECK(cudaEventRecord(e_h2d[0], stream_copy), cleanup, "event_record");

        /* -----------------------------------------------------------------
         * Process each frame in the mini-batch.
         * ----------------------------------------------------------------- */
        for (int m = 0; m < M; m++) {
            int global_idx = batch_start + m;
            int slot       = m % 2;
            int next_slot  = (m + 1) % 2;

            BayerPattern pat = (BayerPattern)frames[global_idx].pattern;
            if (pat == BAYER_NONE && config->bayer_override == BAYER_NONE) {
                fits_get_bayer_pattern(frames[global_idx].filepath, &pat);
            } else if (config->bayer_override != BAYER_NONE) {
                pat = config->bayer_override;
            }

            /* ---- stream_compute: process frame m ---- */

            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, e_h2d[slot], 0), cleanup, "stream_wait");

            PIPE_CHECK(calib_gpu_apply_d2d(d_raw[slot], W, H,
                                            calib_ctx, stream_compute), cleanup, frames[global_idx].filepath);

            if (color) {
                /* Debayer → three separate R/G/B device planes */
                PIPE_CHECK(debayer_gpu_rgb_d2d(d_raw[slot],
                                                d_debayed, d_ch_g, d_ch_b,
                                                W, H, pat, stream_compute),
                           cleanup, frames[global_idx].filepath);

                /* Lanczos warp each channel into its integration slot.
                 * All three calls are on stream_compute (sequential on device,
                 * overlapping H2D for the next frame on stream_copy). */
                CUDA_CHECK(cudaMemsetAsync(ctx_r->d_frames[m], 0, npix_f, stream_compute), cleanup, "memset_r");
                PIPE_CHECK(lanczos_transform_gpu_d2d(
                               d_debayed, ctx_r->d_frames[m],
                               ctx_r->d_xmap, ctx_r->d_ymap,
                               W, H, W, H, &frames[global_idx].H,
                               stream_compute), cleanup, frames[global_idx].filepath);

                CUDA_CHECK(cudaMemsetAsync(ctx_g->d_frames[m], 0, npix_f, stream_compute), cleanup, "memset_g");
                PIPE_CHECK(lanczos_transform_gpu_d2d(
                               d_ch_g, ctx_g->d_frames[m],
                               ctx_g->d_xmap, ctx_g->d_ymap,
                               W, H, W, H, &frames[global_idx].H,
                               stream_compute), cleanup, frames[global_idx].filepath);

                CUDA_CHECK(cudaMemsetAsync(ctx_b->d_frames[m], 0, npix_f, stream_compute), cleanup, "memset_b");
                PIPE_CHECK(lanczos_transform_gpu_d2d(
                               d_ch_b, ctx_b->d_frames[m],
                               ctx_b->d_xmap, ctx_b->d_ymap,
                               W, H, W, H, &frames[global_idx].H,
                               stream_compute), cleanup, frames[global_idx].filepath);
            } else {
                /* Mono path (unchanged) */
                if (pat != BAYER_NONE) {
                    PIPE_CHECK(debayer_gpu_d2d(d_raw[slot], d_debayed,
                                                W, H, pat, stream_compute), cleanup, frames[global_idx].filepath);
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(d_debayed, d_raw[slot], npix_f,
                                               cudaMemcpyDeviceToDevice, stream_compute),
                               cleanup, frames[global_idx].filepath);
                }
                CUDA_CHECK(cudaMemsetAsync(ctx_r->d_frames[m], 0, npix_f, stream_compute), cleanup, "memset");
                PIPE_CHECK(lanczos_transform_gpu_d2d(
                               d_debayed, ctx_r->d_frames[m],
                               ctx_r->d_xmap, ctx_r->d_ymap,
                               W, H, W, H, &frames[global_idx].H,
                               stream_compute), cleanup, frames[global_idx].filepath);
            }

            /* Record GPU-done event after the last Lanczos call for this slot */
            CUDA_CHECK(cudaEventRecord(e_gpu[slot], stream_compute), cleanup, "event_record");

            /* ---- CPU overlap: load next frame while GPU runs ---- */
            if (m + 1 < M) {
                int global_next = batch_start + m + 1;

                if (m >= 1) {
                    CUDA_CHECK(cudaEventSynchronize(e_gpu[next_slot]), cleanup, "event_sync");
                }

                PIPE_CHECK(fits_load_to_buffer(frames[global_next].filepath, pinned[next_slot], W, H),
                           cleanup, frames[global_next].filepath);
                CUDA_CHECK(cudaMemcpyAsync(d_raw[next_slot], pinned[next_slot], npix_f,
                                           cudaMemcpyHostToDevice, stream_copy), cleanup, frames[global_next].filepath);
                CUDA_CHECK(cudaEventRecord(e_h2d[next_slot], stream_copy), cleanup, "event_record");
            }
        } /* end per-frame loop */

        CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup, "stream_sync");

        /* -----------------------------------------------------------------
         * Integration: accumulate this batch into the global accumulators.
         * ----------------------------------------------------------------- */
        printf("[Phase 2] Integrating batch frames %d–%d\n",
               batch_start, batch_start + M - 1);

        if (config->use_kappa_sigma) {
            PIPE_CHECK(integration_gpu_process_batch(ctx_r, M, config->kappa,
                                                      config->iterations,
                                                      stream_compute), cleanup, "batch integration R");
            if (color) {
                PIPE_CHECK(integration_gpu_process_batch(ctx_g, M, config->kappa,
                                                          config->iterations,
                                                          stream_compute), cleanup, "batch integration G");
                PIPE_CHECK(integration_gpu_process_batch(ctx_b, M, config->kappa,
                                                          config->iterations,
                                                          stream_compute), cleanup, "batch integration B");
            }
        } else {
            PIPE_CHECK(integration_gpu_process_batch_mean(ctx_r, M, stream_compute),
                       cleanup, "batch integration R");
            if (color) {
                PIPE_CHECK(integration_gpu_process_batch_mean(ctx_g, M, stream_compute),
                           cleanup, "batch integration G");
                PIPE_CHECK(integration_gpu_process_batch_mean(ctx_b, M, stream_compute),
                           cleanup, "batch integration B");
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup, "stream_sync");

        processed += M;
    } /* end batch loop */

cleanup:
    if (pinned[0]) cudaFreeHost(pinned[0]);
    if (pinned[1]) cudaFreeHost(pinned[1]);
    cudaFree(d_raw[0]);
    cudaFree(d_raw[1]);
    cudaFree(d_debayed);
    cudaFree(d_ch_g);
    cudaFree(d_ch_b);
    if (e_h2d[0]) cudaEventDestroy(e_h2d[0]);
    if (e_h2d[1]) cudaEventDestroy(e_h2d[1]);
    if (e_gpu[0]) cudaEventDestroy(e_gpu[0]);
    if (e_gpu[1]) cudaEventDestroy(e_gpu[1]);
    if (stream_copy)    cudaStreamDestroy(stream_copy);
    if (stream_compute) cudaStreamDestroy(stream_compute);
    return err;
}

/* -------------------------------------------------------------------------
 * Public API: pipeline_run
 * ------------------------------------------------------------------------- */

DsoError pipeline_run(FrameInfo            *frames,
                       int                   n_frames,
                       int                   has_transforms,
                       int                   ref_idx,
                       const PipelineConfig *config)
{
    if (!frames || n_frames <= 0 || !config || !config->output_file)
        return DSO_ERR_INVALID_ARG;
    if (ref_idx < 0 || ref_idx >= n_frames)
        return DSO_ERR_INVALID_ARG;

    /* Dispatch to CPU-only pipeline when GPU Lanczos is disabled. */
    if (!config->use_gpu_lanczos)
        return pipeline_run_cpu(frames, n_frames, has_transforms, ref_idx, config);

    DsoError           err        = DSO_OK;
    int                W          = 0;
    int                H          = 0;
    IntegrationGpuCtx *ctx_r      = NULL;   /* mono luminance or R channel */
    IntegrationGpuCtx *ctx_g      = NULL;   /* G channel (color only) */
    IntegrationGpuCtx *ctx_b      = NULL;   /* B channel (color only) */
    CalibGpuCtx       *calib_ctx  = NULL;
    StarList          *star_lists = NULL;

    /* ---- Load reference frame to determine output dimensions ---- */
    {
        Image ref_img = {NULL, 0, 0};
        PIPE_CHECK(fits_load(frames[ref_idx].filepath, &ref_img), done, frames[ref_idx].filepath);
        W = ref_img.width;
        H = ref_img.height;
        image_free(&ref_img);
    }
    printf("pipeline: output dimensions %d × %d, %d frame(s)\n", W, H, n_frames);

    /* ---- Initialise GPU subsystems ---- */
    PIPE_CHECK(lanczos_gpu_init(0 /* default stream */), done, "lanczos_gpu_init");
    PIPE_CHECK(integration_gpu_init(W, H, config->batch_size, &ctx_r), done, "integration_gpu_init R");
    if (config->color_output) {
        PIPE_CHECK(integration_gpu_init(W, H, config->batch_size, &ctx_g), done, "integration_gpu_init G");
        PIPE_CHECK(integration_gpu_init(W, H, config->batch_size, &ctx_b), done, "integration_gpu_init B");
    }

    /* Upload calibration master frames to device (if calibration was requested) */
    if (config->calib)
        PIPE_CHECK(calib_gpu_init(config->calib, &calib_ctx), done, "calib_gpu_init");

    if (!has_transforms) {
        printf("pipeline: Phase 1 — star detection (%d frames)\n", n_frames);

        star_lists = (StarList *)calloc((size_t)n_frames, sizeof(StarList));
        if (!star_lists) { err = DSO_ERR_ALLOC; goto done; }

        PIPE_CHECK(phase1_detect_stars(frames, n_frames, ref_idx,
                                        W, H, config, calib_ctx, star_lists), done, "Phase 1");
        printf("pipeline: Phase 1 complete\n");
    }

    /* ---- Phase 2: Lanczos transform + GPU integration ---- */
    printf("pipeline: Phase 2 — transform + %s integration (%d frames, batch=%d)\n",
           config->use_kappa_sigma ? "kappa-sigma" : "mean",
           n_frames, config->batch_size);

    PIPE_CHECK(phase2_transform_integrate(frames, n_frames, W, H,
                                           config, calib_ctx,
                                           ctx_r, ctx_g, ctx_b), done, "Phase 2");
    printf("pipeline: Phase 2 complete\n");

    /* ---- Finalise: compute output image(s) and save ---- */
    printf("pipeline: saving to %s\n", config->output_file);
    if (config->color_output) {
        Image img_r = {NULL, W, H}, img_g = {NULL, W, H}, img_b = {NULL, W, H};
        img_r.data = (float *)calloc((size_t)W * H, sizeof(float));
        img_g.data = (float *)calloc((size_t)W * H, sizeof(float));
        img_b.data = (float *)calloc((size_t)W * H, sizeof(float));
        if (!img_r.data || !img_g.data || !img_b.data) {
            image_free(&img_r); image_free(&img_g); image_free(&img_b);
            err = DSO_ERR_ALLOC; goto done;
        }
        err = integration_gpu_finalize(ctx_r, n_frames, &img_r, 0);
        if (err == DSO_OK) err = integration_gpu_finalize(ctx_g, n_frames, &img_g, 0);
        if (err == DSO_OK) err = integration_gpu_finalize(ctx_b, n_frames, &img_b, 0);
        if (err == DSO_OK) err = fits_save_rgb(config->output_file, &img_r, &img_g, &img_b);
        image_free(&img_r); image_free(&img_g); image_free(&img_b);
    } else {
        Image out = {NULL, W, H};
        out.data = (float *)calloc((size_t)W * H, sizeof(float));
        if (!out.data) { err = DSO_ERR_ALLOC; goto done; }
        if (integration_gpu_finalize(ctx_r, n_frames, &out, 0) == DSO_OK)
            err = fits_save(config->output_file, &out);
        else
            err = DSO_ERR_CUDA;
        image_free(&out);
    }

done:
    /* Free star lists (if Phase 1 ran) */
    if (star_lists) {
        for (int i = 0; i < n_frames; i++) free(star_lists[i].stars);
        free(star_lists);
    }

    calib_gpu_cleanup(calib_ctx);
    integration_gpu_cleanup(ctx_r);
    integration_gpu_cleanup(ctx_g);
    integration_gpu_cleanup(ctx_b);
    lanczos_gpu_cleanup();

    if (err == DSO_OK)
        printf("pipeline: done.\n");
    else
        fprintf(stderr, "pipeline: failed with error code %d\n", (int)err);

    return err;
}
