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
#define PIPE_CHECK(call, label)                                      \
    do {                                                             \
        DsoError _pe = (call);                                       \
        if (_pe != DSO_OK) { err = _pe; goto label; }               \
    } while (0)

/* Wrap a CUDA call; print message, set err, jump to label on failure. */
#define CUDA_CHECK(call, label)                                      \
    do {                                                             \
        cudaError_t _ce = (call);                                    \
        if (_ce != cudaSuccess) {                                    \
            fprintf(stderr, "pipeline CUDA error %s:%d — %s\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(_ce));    \
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
    FrameInfo          *frames,
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

    CUDA_CHECK(cudaMalloc(&d_raw,     npix_f), cleanup);
    CUDA_CHECK(cudaMalloc(&d_debayed, npix_f), cleanup);
    CUDA_CHECK(cudaMalloc(&d_conv,    npix_f), cleanup);
    CUDA_CHECK(cudaMalloc(&d_mask,    npix_b), cleanup);

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
        PIPE_CHECK(fits_load(frames[i].filepath, &raw), cleanup);
        if (raw.width != W || raw.height != H) {
            fprintf(stderr, "pipeline: frame %d size %dx%d != ref %dx%d\n",
                    i, raw.width, raw.height, W, H);
            image_free(&raw); err = DSO_ERR_INVALID_ARG; goto cleanup;
        }

        /* Determine Bayer pattern (override or auto-detect from FITS header) */
        BayerPattern pat = config->bayer_override;
        if (pat == BAYER_NONE)
            fits_get_bayer_pattern(frames[i].filepath, &pat);

        /* H2D raw → device */
        CUDA_CHECK(cudaMemcpy(d_raw, raw.data, npix_f, cudaMemcpyHostToDevice), frame_err);

        /* Apply calibration (dark subtract + flat divide) before debayering */
        PIPE_CHECK(calib_gpu_apply_d2d(d_raw, W, H, calib_ctx, 0), frame_err);

        /* Debayer → luminance */
        PIPE_CHECK(debayer_gpu_d2d(d_raw, d_debayed, W, H, pat, 0), frame_err);

        /* Moffat convolution + sigma threshold → d_conv, d_mask */
        PIPE_CHECK(star_detect_gpu_d2d(d_debayed, d_conv, d_mask,
                                        W, H, &config->moffat,
                                        config->star_sigma, 0), frame_err);

        /* D2H: three arrays needed by CCL+CoM */
        CUDA_CHECK(cudaMemcpy(original_host,  d_debayed, npix_f,
                              cudaMemcpyDeviceToHost), frame_err);
        CUDA_CHECK(cudaMemcpy(convolved_host, d_conv,    npix_f,
                              cudaMemcpyDeviceToHost), frame_err);
        CUDA_CHECK(cudaMemcpy(mask_host,      d_mask,    npix_b,
                              cudaMemcpyDeviceToHost), frame_err);

        /* CPU: connected-component labeling + weighted center-of-mass */
        PIPE_CHECK(star_detect_cpu_ccl_com(mask_host, original_host, convolved_host,
                                            W, H, config->top_stars,
                                            &star_lists[i]), frame_err);
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

static DsoError phase2_transform_integrate(
    FrameInfo            *frames,
    int                   n_frames,
    int                   W, int H,
    const PipelineConfig *config,
    CalibGpuCtx          *calib_ctx,   /* NULL = no calibration */
    IntegrationGpuCtx    *ctx)
{
    DsoError     err            = DSO_OK;
    size_t       npix_f        = (size_t)W * H * sizeof(float);
    cudaStream_t stream_copy   = 0;
    cudaStream_t stream_compute = 0;
    cudaEvent_t  e_h2d[2]      = {0, 0};
    cudaEvent_t  e_gpu[2]      = {0, 0};
    float       *pinned[2]     = {NULL, NULL};
    float       *d_raw[2]      = {NULL, NULL};
    float       *d_debayed     = NULL;
    int          processed     = 0;

    CUDA_CHECK(cudaStreamCreate(&stream_copy),    cleanup);
    CUDA_CHECK(cudaStreamCreate(&stream_compute), cleanup);

    for (int j = 0; j < 2; j++) {
        CUDA_CHECK(cudaEventCreate(&e_h2d[j]), cleanup);
        CUDA_CHECK(cudaEventCreate(&e_gpu[j]),  cleanup);
    }

    CUDA_CHECK(cudaMallocHost(&pinned[0], npix_f), cleanup);
    CUDA_CHECK(cudaMallocHost(&pinned[1], npix_f), cleanup);
    CUDA_CHECK(cudaMalloc(&d_raw[0], npix_f), cleanup);
    CUDA_CHECK(cudaMalloc(&d_raw[1], npix_f), cleanup);
    CUDA_CHECK(cudaMalloc(&d_debayed, npix_f), cleanup);

    while (processed < n_frames) {
        /* Clamp batch to remaining frames */
        int M           = n_frames - processed;
        if (M > config->batch_size) M = config->batch_size;
        int batch_start = processed;

        /* -----------------------------------------------------------------
         * Kickstart: load frame 0 of this batch into pinned[0], start H2D.
         * ----------------------------------------------------------------- */
        {
            Image raw = {NULL, 0, 0};
            PIPE_CHECK(fits_load(frames[batch_start].filepath, &raw), cleanup);
            if (raw.width != W || raw.height != H) {
                image_free(&raw); err = DSO_ERR_INVALID_ARG; goto cleanup;
            }
            memcpy(pinned[0], raw.data, npix_f);
            image_free(&raw);
        }
        CUDA_CHECK(cudaMemcpyAsync(d_raw[0], pinned[0], npix_f,
                                   cudaMemcpyHostToDevice, stream_copy), cleanup);
        CUDA_CHECK(cudaEventRecord(e_h2d[0], stream_copy), cleanup);

        /* -----------------------------------------------------------------
         * Process each frame in the mini-batch.
         * ----------------------------------------------------------------- */
        for (int m = 0; m < M; m++) {
            int global_idx = batch_start + m;
            int slot       = m % 2;
            int next_slot  = (m + 1) % 2;

            /* Determine Bayer pattern */
            BayerPattern pat = config->bayer_override;
            if (pat == BAYER_NONE)
                fits_get_bayer_pattern(frames[global_idx].filepath, &pat);

            /* ---- stream_compute: process frame m ---- */

            /* Wait (GPU-side) for H2D of this slot to complete */
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, e_h2d[slot], 0), cleanup);

            /* Apply calibration (dark subtract + flat divide) before debayering */
            PIPE_CHECK(calib_gpu_apply_d2d(d_raw[slot], W, H,
                                            calib_ctx, stream_compute), cleanup);

            /* Debayer raw → luminance (or plain device-to-device for monochrome) */
            if (pat != BAYER_NONE) {
                PIPE_CHECK(debayer_gpu_d2d(d_raw[slot], d_debayed,
                                            W, H, pat, stream_compute), cleanup);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_debayed, d_raw[slot], npix_f,
                                           cudaMemcpyDeviceToDevice, stream_compute),
                           cleanup);
            }

            /* Zero-fill destination: lanczos_transform_gpu_d2d leaves out-of-bounds
             * pixels unwritten, so the destination must start as all-zero. */
            CUDA_CHECK(cudaMemsetAsync(ctx->d_frames[m], 0, npix_f,
                                       stream_compute), cleanup);

            /* Lanczos-3 warp using pre-computed backward homography (ref → src) */
            PIPE_CHECK(lanczos_transform_gpu_d2d(
                           d_debayed, ctx->d_frames[m],
                           ctx->d_xmap, ctx->d_ymap,
                           W, H,   /* source dimensions  */
                           W, H,   /* destination dimensions */
                           &frames[global_idx].H,
                           stream_compute), cleanup);

            /* Record GPU-done event for this slot */
            CUDA_CHECK(cudaEventRecord(e_gpu[slot], stream_compute), cleanup);

            /* ---- CPU overlap: load next frame while GPU runs ---- */
            if (m + 1 < M) {
                int global_next = batch_start + m + 1;

                /* Ensure d_raw[next_slot] is no longer needed by stream_compute.
                 * next_slot was last used for frame m-1 (two steps ago);
                 * e_gpu[next_slot] fires when stream_compute finishes that frame.
                 * For m == 0 the event was never set, so skip the sync. */
                if (m >= 1) {
                    CUDA_CHECK(cudaEventSynchronize(e_gpu[next_slot]), cleanup);
                }

                /* CPU: read next frame from disk into pinned[next_slot].
                 * This is the key overlap: disk I/O runs concurrently with
                 * GPU processing the current frame on stream_compute. */
                {
                    Image raw_next = {NULL, 0, 0};
                    PIPE_CHECK(fits_load(frames[global_next].filepath, &raw_next),
                               cleanup);
                    if (raw_next.width != W || raw_next.height != H) {
                        image_free(&raw_next);
                        err = DSO_ERR_INVALID_ARG;
                        goto cleanup;
                    }
                    memcpy(pinned[next_slot], raw_next.data, npix_f);
                    image_free(&raw_next);
                }

                /* H2D next frame into device slot */
                CUDA_CHECK(cudaMemcpyAsync(d_raw[next_slot], pinned[next_slot], npix_f,
                                           cudaMemcpyHostToDevice, stream_copy), cleanup);
                CUDA_CHECK(cudaEventRecord(e_h2d[next_slot], stream_copy), cleanup);
            }
        } /* end per-frame loop */

        /* Wait for all GPU work in this batch to complete */
        CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup);

        /* -----------------------------------------------------------------
         * Integration: accumulate this batch into the global accumulators.
         * ----------------------------------------------------------------- */
        printf("[Phase 2] Integrating batch frames %d–%d\n",
               batch_start, batch_start + M - 1);

        if (config->use_kappa_sigma) {
            PIPE_CHECK(integration_gpu_process_batch(ctx, M, config->kappa,
                                                      config->iterations,
                                                      stream_compute), cleanup);
        } else {
            PIPE_CHECK(integration_gpu_process_batch_mean(ctx, M, stream_compute),
                       cleanup);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup);

        processed += M;
    } /* end batch loop */

cleanup:
    cudaFreeHost(pinned[0]);
    cudaFreeHost(pinned[1]);
    cudaFree(d_raw[0]);
    cudaFree(d_raw[1]);
    cudaFree(d_debayed);
    cudaEventDestroy(e_h2d[0]); cudaEventDestroy(e_h2d[1]);
    cudaEventDestroy(e_gpu[0]);  cudaEventDestroy(e_gpu[1]);
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
    IntegrationGpuCtx *ctx        = NULL;
    CalibGpuCtx       *calib_ctx  = NULL;
    StarList          *star_lists = NULL;

    /* ---- Load reference frame to determine output dimensions ---- */
    {
        Image ref_img = {NULL, 0, 0};
        PIPE_CHECK(fits_load(frames[ref_idx].filepath, &ref_img), done);
        W = ref_img.width;
        H = ref_img.height;
        image_free(&ref_img);
    }
    printf("pipeline: output dimensions %d × %d, %d frame(s)\n", W, H, n_frames);

    /* ---- Initialise GPU subsystems ---- */
    PIPE_CHECK(lanczos_gpu_init(0 /* default stream */), done);
    PIPE_CHECK(integration_gpu_init(W, H, config->batch_size, &ctx), done);

    /* Upload calibration master frames to device (if calibration was requested) */
    if (config->calib)
        PIPE_CHECK(calib_gpu_init(config->calib, &calib_ctx), done);

    if (!has_transforms) {
        printf("pipeline: Phase 1 — star detection (%d frames)\n", n_frames);

        star_lists = (StarList *)calloc((size_t)n_frames, sizeof(StarList));
        if (!star_lists) { err = DSO_ERR_ALLOC; goto done; }

        PIPE_CHECK(phase1_detect_stars(frames, n_frames, ref_idx,
                                        W, H, config, calib_ctx, star_lists), done);
        printf("pipeline: Phase 1 complete\n");
    }

    /* ---- Phase 2: Lanczos transform + GPU integration ---- */
    printf("pipeline: Phase 2 — transform + %s integration (%d frames, batch=%d)\n",
           config->use_kappa_sigma ? "kappa-sigma" : "mean",
           n_frames, config->batch_size);

    PIPE_CHECK(phase2_transform_integrate(frames, n_frames, W, H,
                                           config, calib_ctx, ctx), done);
    printf("pipeline: Phase 2 complete\n");

    /* ---- Finalise: compute output image and save ---- */
    {
        Image out = {NULL, W, H};
        out.data = (float *)calloc((size_t)W * H, sizeof(float));
        if (!out.data) { err = DSO_ERR_ALLOC; goto done; }

        PIPE_CHECK(integration_gpu_finalize(ctx, n_frames, &out, 0), finalize_err);

        printf("pipeline: saving to %s\n", config->output_file);
        err = fits_save(config->output_file, &out);
        image_free(&out);
        goto done;

    finalize_err:
        image_free(&out);
    }

done:
    /* Free star lists (if Phase 1 ran) */
    if (star_lists) {
        for (int i = 0; i < n_frames; i++) free(star_lists[i].stars);
        free(star_lists);
    }

    calib_gpu_cleanup(calib_ctx);
    integration_gpu_cleanup(ctx);
    lanczos_gpu_cleanup();

    if (err == DSO_OK)
        printf("pipeline: done.\n");
    else
        fprintf(stderr, "pipeline: failed with error code %d\n", (int)err);

    return err;
}
