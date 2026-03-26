/*
 * pipeline.cu — Full DSO stacking pipeline orchestrator.
 *
 * Single-pass execution: every frame is loaded from disk exactly once.
 * The reference frame is processed first (to build ref_stars), then all
 * non-reference frames follow.  Per-frame sequence:
 *
 *   fits_load_to_buffer → H2D (stream_copy)
 *   → calib → debayer_lum → Moffat+threshold  (stream_compute)
 *   → D2H → CCL+CoM → RANSAC                  (CPU)
 *   → Lanczos warp → integration batch slot    (stream_compute)
 *
 * I/O overlap:
 *   After RANSAC the Lanczos warp for frame m runs on stream_compute while
 *   the CPU simultaneously loads frame m+1 from disk and streams it to the
 *   device via stream_copy.  At batch boundaries the H2D of frame m+1 also
 *   overlaps with mini-batch kappa-sigma integration on stream_compute.
 *
 * Memory safety for double-buffered d_raw[2]:
 *   cudaStreamSynchronize(stream_compute) called for the D2H sync point
 *   serialises all prior stream_compute work, which includes the Lanczos
 *   warp from the previous frame.  By the time we pre-load d_raw[next_slot]
 *   that slot is guaranteed free — no extra cudaEvent guard is needed.
 *
 * Mini-batch kappa-sigma integration caps peak GPU memory to batch_size
 * warped frames (default 16).
 */

#include "pipeline.h"
#include "calibration.h"
#include "calibration_gpu.h"
#include "image_io.h"
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

#define PIPE_CHECK(call, label, path)                                \
    do {                                                             \
        DsoError _pe = (call);                                       \
        if (_pe != DSO_OK) {                                         \
            fprintf(stderr, "pipeline error %d at %s\n", (int)_pe, (path)); \
            err = _pe; goto label;                                   \
        }                                                            \
    } while (0)

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
 * phase_warp — warp a single raw device frame into the next integration slot.
 *
 * For mono: debayer_lum is already in d_lum; warp it directly.
 * For color: re-debayer d_raw → RGB on stream, then warp each channel.
 *
 * d_lum is an in/out scratch buffer (re-used as R in color mode).
 * batch_n is the slot index within ctx_r->d_frames[].
 * ------------------------------------------------------------------------- */
static DsoError phase_warp(
    float *d_raw, float *d_lum, float *d_ch_g, float *d_ch_b,
    int W, int H, int color, BayerPattern pat,
    const Homography *H_frame, int batch_n,
    IntegrationGpuCtx *ctx_r, IntegrationGpuCtx *ctx_g, IntegrationGpuCtx *ctx_b,
    cudaStream_t stream, const char *label)
{
    DsoError err = DSO_OK;
    size_t npix_f = (size_t)W * H * sizeof(float);

    if (color) {
        /* Re-debayer raw → R (into d_lum), G (d_ch_g), B (d_ch_b) */
        PIPE_CHECK(debayer_gpu_rgb_d2d(d_raw, d_lum, d_ch_g, d_ch_b,
                                        W, H, pat, stream),
                   done, label);
        CUDA_CHECK(cudaMemsetAsync(ctx_r->d_frames[batch_n], 0, npix_f, stream),
                   done, label);
        PIPE_CHECK(lanczos_transform_gpu_d2d(d_lum, ctx_r->d_frames[batch_n],
                                              ctx_r->d_xmap, ctx_r->d_ymap,
                                              W, H, W, H, H_frame, stream),
                   done, label);
        CUDA_CHECK(cudaMemsetAsync(ctx_g->d_frames[batch_n], 0, npix_f, stream),
                   done, label);
        PIPE_CHECK(lanczos_transform_gpu_d2d(d_ch_g, ctx_g->d_frames[batch_n],
                                              ctx_g->d_xmap, ctx_g->d_ymap,
                                              W, H, W, H, H_frame, stream),
                   done, label);
        CUDA_CHECK(cudaMemsetAsync(ctx_b->d_frames[batch_n], 0, npix_f, stream),
                   done, label);
        PIPE_CHECK(lanczos_transform_gpu_d2d(d_ch_b, ctx_b->d_frames[batch_n],
                                              ctx_b->d_xmap, ctx_b->d_ymap,
                                              W, H, W, H, H_frame, stream),
                   done, label);
    } else {
        /* d_lum already contains the luminance; warp directly */
        CUDA_CHECK(cudaMemsetAsync(ctx_r->d_frames[batch_n], 0, npix_f, stream),
                   done, label);
        PIPE_CHECK(lanczos_transform_gpu_d2d(d_lum, ctx_r->d_frames[batch_n],
                                              ctx_r->d_xmap, ctx_r->d_ymap,
                                              W, H, W, H, H_frame, stream),
                   done, label);
    }
done:
    return err;
}

/* -------------------------------------------------------------------------
 * integrate_batch — flush the current mini-batch into the accumulators.
 * ------------------------------------------------------------------------- */
static DsoError integrate_batch(
    int M, int color, float kappa, int iterations, int use_kappa_sigma,
    cudaStream_t stream,
    IntegrationGpuCtx *ctx_r, IntegrationGpuCtx *ctx_g, IntegrationGpuCtx *ctx_b)
{
    DsoError err = DSO_OK;
    printf("[Pipeline] Integrating batch of %d frame(s)...\n", M);

    if (use_kappa_sigma) {
        PIPE_CHECK(integration_gpu_process_batch(ctx_r, M, kappa, iterations, stream),
                   done, "batch integration R");
        if (color) {
            PIPE_CHECK(integration_gpu_process_batch(ctx_g, M, kappa, iterations, stream),
                       done, "batch integration G");
            PIPE_CHECK(integration_gpu_process_batch(ctx_b, M, kappa, iterations, stream),
                       done, "batch integration B");
        }
    } else {
        PIPE_CHECK(integration_gpu_process_batch_mean(ctx_r, M, stream),
                   done, "batch integration R");
        if (color) {
            PIPE_CHECK(integration_gpu_process_batch_mean(ctx_g, M, stream),
                       done, "batch integration G");
            PIPE_CHECK(integration_gpu_process_batch_mean(ctx_b, M, stream),
                       done, "batch integration B");
        }
    }
done:
    return err;
}

/* -------------------------------------------------------------------------
 * phase_detect_warp_integrate — single-pass streaming pipeline.
 *
 * Processes frames in the order: [ref_idx, non-ref0, non-ref1, ...].
 * For each frame:
 *   1. GPU:  calib → debayer_lum → Moffat+threshold  (stream_compute)
 *   2. SYNC: cudaStreamSynchronize(stream_compute)
 *            Also clears the Lanczos warp from the previous frame,
 *            ensuring d_raw[next_slot] is safe to overwrite.
 *   3. CPU:  D2H (lum/conv/mask) → CCL+CoM → RANSAC
 *   4. GPU:  Lanczos warp  (stream_compute)
 *   5. CPU:  Load next frame from disk + async H2D  (stream_copy)
 *            Overlaps with the GPU warp in step 4.
 *   6. Batch full → sync + integrate + sync  (stream_compute)
 * ------------------------------------------------------------------------- */
static DsoError phase_detect_warp_integrate(
    FrameInfo            *frames,
    int                   n_frames,
    int                   ref_idx,
    int                   W, int H,
    const PipelineConfig *config,
    CalibGpuCtx          *calib_ctx,
    IntegrationGpuCtx    *ctx_r,
    IntegrationGpuCtx    *ctx_g,
    IntegrationGpuCtx    *ctx_b)
{
    DsoError     err           = DSO_OK;
    size_t       npix_f        = (size_t)W * H * sizeof(float);
    size_t       npix_b        = (size_t)W * H;
    int          color         = (ctx_g != NULL);
    int          batch_n       = 0;
    int         *order         = NULL;

    cudaStream_t stream_copy   = 0;
    cudaStream_t stream_compute = 0;
    cudaEvent_t  e_h2d[2]      = {0, 0};

    float       *pinned[2]     = {NULL, NULL};
    float       *d_raw[2]      = {NULL, NULL};
    float       *d_lum         = NULL;
    float       *d_conv        = NULL;
    uint8_t     *d_mask        = NULL;
    float       *d_ch_g        = NULL;
    float       *d_ch_b        = NULL;

    float       *lum_host      = NULL;
    float       *conv_host     = NULL;
    uint8_t     *mask_host     = NULL;
    StarList     ref_stars      = {NULL, 0};

    /* --- Streams and events --- */
    CUDA_CHECK(cudaStreamCreate(&stream_copy),    cleanup, "stream_copy");
    CUDA_CHECK(cudaStreamCreate(&stream_compute), cleanup, "stream_compute");
    for (int j = 0; j < 2; j++)
        CUDA_CHECK(cudaEventCreate(&e_h2d[j]), cleanup, "e_h2d");

    /* --- Device allocations --- */
    CUDA_CHECK(cudaMallocHost(&pinned[0], npix_f), cleanup, "pinned[0]");
    CUDA_CHECK(cudaMallocHost(&pinned[1], npix_f), cleanup, "pinned[1]");
    CUDA_CHECK(cudaMalloc(&d_raw[0], npix_f), cleanup, "d_raw[0]");
    CUDA_CHECK(cudaMalloc(&d_raw[1], npix_f), cleanup, "d_raw[1]");
    CUDA_CHECK(cudaMalloc(&d_lum,    npix_f), cleanup, "d_lum");
    CUDA_CHECK(cudaMalloc(&d_conv,   npix_f), cleanup, "d_conv");
    CUDA_CHECK(cudaMalloc(&d_mask,   npix_b), cleanup, "d_mask");
    if (color) {
        CUDA_CHECK(cudaMalloc(&d_ch_g, npix_f), cleanup, "d_ch_g");
        CUDA_CHECK(cudaMalloc(&d_ch_b, npix_f), cleanup, "d_ch_b");
    }

    /* --- Host staging buffers for D2H --- */
    lum_host  = (float   *)malloc(npix_f);
    conv_host = (float   *)malloc(npix_f);
    mask_host = (uint8_t *)malloc(npix_b);
    if (!lum_host || !conv_host || !mask_host) { err = DSO_ERR_ALLOC; goto cleanup; }

    /* --- Processing order: reference frame first --- */
    order = (int *)malloc((size_t)n_frames * sizeof(int));
    if (!order) { err = DSO_ERR_ALLOC; goto cleanup; }
    order[0] = ref_idx;
    {
        int npos = 1;
        for (int k = 0; k < n_frames; k++)
            if (k != ref_idx) order[npos++] = k;
    }

    /* Set identity homography for the reference frame */
    memset(&frames[ref_idx].H, 0, sizeof(Homography));
    frames[ref_idx].H.h[0] = frames[ref_idx].H.h[4] = frames[ref_idx].H.h[8] = 1.0;

    /* --- Kickstart: load frame 0 and begin H2D --- */
    printf("[Pipeline] Loading frame 1/%d: %s\n", n_frames, frames[order[0]].filepath);
    PIPE_CHECK(fits_load_to_buffer(frames[order[0]].filepath, pinned[0], W, H),
               cleanup, frames[order[0]].filepath);
    CUDA_CHECK(cudaMemcpyAsync(d_raw[0], pinned[0], npix_f,
                               cudaMemcpyHostToDevice, stream_copy),
               cleanup, frames[order[0]].filepath);
    CUDA_CHECK(cudaEventRecord(e_h2d[0], stream_copy), cleanup, "e_h2d kickstart");

    /* ================================================================
     * Main frame loop
     * ============================================================== */
    for (int pos = 0; pos < n_frames; pos++) {
        int       fi_idx    = order[pos];
        int       slot      = pos % 2;
        int       next_slot = 1 - slot;
        FrameInfo *fi       = &frames[fi_idx];

        BayerPattern pat = config->bayer_override;
        if (pat == BAYER_NONE) fits_get_bayer_pattern(fi->filepath, &pat);

        /* ----------------------------------------------------------
         * 1. GPU: calib + debayer_lum + star_detect on stream_compute.
         *    Stream_compute waits for the H2D of this slot first.
         * ---------------------------------------------------------- */
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, e_h2d[slot], 0),
                   cleanup, "stream_wait_e_h2d");
        PIPE_CHECK(calib_gpu_apply_d2d(d_raw[slot], W, H, calib_ctx, stream_compute),
                   cleanup, fi->filepath);
        PIPE_CHECK(debayer_gpu_d2d(d_raw[slot], d_lum, W, H, pat, stream_compute),
                   cleanup, fi->filepath);
        PIPE_CHECK(star_detect_gpu_d2d(d_lum, d_conv, d_mask,
                                        W, H, &config->moffat,
                                        config->star_sigma, stream_compute),
                   cleanup, fi->filepath);

        /* ----------------------------------------------------------
         * 2. Sync stream_compute.
         *    Guarantees star_detect outputs are ready for D2H AND
         *    that the Lanczos warp from pos-1 (on d_raw[next_slot])
         *    is complete — making next_slot safe to reuse.
         * ---------------------------------------------------------- */
        CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup, "star_detect sync");

        /* ----------------------------------------------------------
         * 3. D2H: copy lum/conv/mask to host for CCL+RANSAC.
         * ---------------------------------------------------------- */
        CUDA_CHECK(cudaMemcpy(lum_host,  d_lum,  npix_f, cudaMemcpyDeviceToHost),
                   cleanup, fi->filepath);
        CUDA_CHECK(cudaMemcpy(conv_host, d_conv, npix_f, cudaMemcpyDeviceToHost),
                   cleanup, fi->filepath);
        CUDA_CHECK(cudaMemcpy(mask_host, d_mask, npix_b, cudaMemcpyDeviceToHost),
                   cleanup, fi->filepath);

        /* ----------------------------------------------------------
         * 4. CPU: CCL + CoM, then RANSAC (or store ref_stars).
         * ---------------------------------------------------------- */
        StarList stars = {NULL, 0};
        PIPE_CHECK(star_detect_cpu_ccl_com(mask_host, lum_host, conv_host,
                                            W, H, config->top_stars, &stars),
                   cleanup, fi->filepath);
        printf("[Pipeline] Frame %d/%d: %d star(s) — %s\n",
               pos + 1, n_frames, stars.n, fi->filepath);

        if (fi_idx == ref_idx) {
            if (stars.n < config->min_stars) {
                fprintf(stderr,
                        "pipeline: reference frame has only %d star(s) "
                        "(min_stars=%d)\n",
                        stars.n, config->min_stars);
                free(stars.stars); err = DSO_ERR_STAR_DETECT; goto cleanup;
            }
            ref_stars = stars;  /* take ownership */
        } else {
            if (stars.n < config->min_stars) {
                fprintf(stderr,
                        "pipeline: frame %d has only %d star(s) "
                        "(min_stars=%d) — cannot align\n",
                        fi_idx, stars.n, config->min_stars);
                free(stars.stars); err = DSO_ERR_STAR_DETECT; goto cleanup;
            }
            int n_inliers = 0;
            err = ransac_compute_homography(&ref_stars, &stars, &config->ransac,
                                            &fi->H, &n_inliers);
            free(stars.stars);
            if (err != DSO_OK) {
                fprintf(stderr,
                        "pipeline: RANSAC failed for frame %d/%d "
                        "(csv index=%d, path=%s, err=%d)\n",
                        pos + 1, n_frames, fi_idx + 1, fi->filepath, (int)err);
                goto cleanup;
            }
            printf("[Pipeline] Frame %d/%d (csv index=%d): aligned with %d inlier(s)\n",
                   pos + 1, n_frames, fi_idx + 1, n_inliers);
        }

        /* ----------------------------------------------------------
         * 5. GPU: Lanczos warp on stream_compute.
         *    d_raw[slot] is still valid (H2D completed at step 2 sync).
         * ---------------------------------------------------------- */
        PIPE_CHECK(phase_warp(d_raw[slot], d_lum, d_ch_g, d_ch_b,
                               W, H, color, pat, &fi->H, batch_n,
                               ctx_r, ctx_g, ctx_b, stream_compute, fi->filepath),
                   cleanup, fi->filepath);
        batch_n++;

        /* ----------------------------------------------------------
         * 6. Pre-load next frame (overlaps with GPU warp above).
         *    d_raw[next_slot] is safe: step 2 sync cleared its prior
         *    warp before we queued the current one on stream_compute.
         * ---------------------------------------------------------- */
        if (pos + 1 < n_frames) {
            int fi_next_idx = order[pos + 1];
            printf("[Pipeline] Loading frame %d/%d: %s\n",
                   pos + 2, n_frames, frames[fi_next_idx].filepath);
            PIPE_CHECK(fits_load_to_buffer(frames[fi_next_idx].filepath,
                                            pinned[next_slot], W, H),
                       cleanup, frames[fi_next_idx].filepath);
            CUDA_CHECK(cudaMemcpyAsync(d_raw[next_slot], pinned[next_slot], npix_f,
                                       cudaMemcpyHostToDevice, stream_copy),
                       cleanup, frames[fi_next_idx].filepath);
            CUDA_CHECK(cudaEventRecord(e_h2d[next_slot], stream_copy),
                       cleanup, "e_h2d next");
        }

        /* ----------------------------------------------------------
         * 7. If mini-batch is full: sync warp → integrate → sync.
         *    H2D of frame pos+1 on stream_copy overlaps integration.
         * ---------------------------------------------------------- */
        if (batch_n == config->batch_size) {
            CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup, "batch warp sync");
            PIPE_CHECK(integrate_batch(batch_n, color,
                                        config->kappa, config->iterations,
                                        config->use_kappa_sigma, stream_compute,
                                        ctx_r, ctx_g, ctx_b),
                       cleanup, "batch integration");
            CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup, "batch int sync");
            batch_n = 0;
        }
    } /* end frame loop */

    /* --- Flush remaining partial batch --- */
    if (batch_n > 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup, "final warp sync");
        PIPE_CHECK(integrate_batch(batch_n, color,
                                    config->kappa, config->iterations,
                                    config->use_kappa_sigma, stream_compute,
                                    ctx_r, ctx_g, ctx_b),
                   cleanup, "final integration");
        CUDA_CHECK(cudaStreamSynchronize(stream_compute), cleanup, "final int sync");
    }

cleanup:
    free(order);
    free(ref_stars.stars);
    free(lum_host); free(conv_host); free(mask_host);
    if (pinned[0]) cudaFreeHost(pinned[0]);
    if (pinned[1]) cudaFreeHost(pinned[1]);
    cudaFree(d_raw[0]); cudaFree(d_raw[1]);
    cudaFree(d_lum); cudaFree(d_conv); cudaFree(d_mask);
    cudaFree(d_ch_g); cudaFree(d_ch_b);
    if (e_h2d[0]) cudaEventDestroy(e_h2d[0]);
    if (e_h2d[1]) cudaEventDestroy(e_h2d[1]);
    if (stream_copy)    cudaStreamDestroy(stream_copy);
    if (stream_compute) cudaStreamDestroy(stream_compute);
    return err;
}

/* -------------------------------------------------------------------------
 * Public API: pipeline_run
 * ------------------------------------------------------------------------- */

DsoError pipeline_run(FrameInfo            *frames,
                       int                   n_frames,
                       int                   ref_idx,
                       const PipelineConfig *config)
{
    if (!frames || n_frames <= 0 || !config || !config->output_file)
        return DSO_ERR_INVALID_ARG;
    if (ref_idx < 0 || ref_idx >= n_frames)
        return DSO_ERR_INVALID_ARG;

    /* Dispatch to CPU-only pipeline when GPU Lanczos is disabled */
    if (!config->use_gpu_lanczos)
        return pipeline_run_cpu(frames, n_frames, ref_idx, config);

    DsoError           err       = DSO_OK;
    int                W         = 0;
    int                H         = 0;
    IntegrationGpuCtx *ctx_r     = NULL;
    IntegrationGpuCtx *ctx_g     = NULL;
    IntegrationGpuCtx *ctx_b     = NULL;
    CalibGpuCtx       *calib_ctx = NULL;

    /* Determine output dimensions from reference frame */
    {
        Image ref_img = {NULL, 0, 0};
        PIPE_CHECK(fits_load(frames[ref_idx].filepath, &ref_img),
                   done, frames[ref_idx].filepath);
        W = ref_img.width;
        H = ref_img.height;
        image_free(&ref_img);
    }
    printf("pipeline: output dimensions %d × %d, %d frame(s)\n", W, H, n_frames);

    /* Initialise GPU subsystems */
    PIPE_CHECK(lanczos_gpu_init(0), done, "lanczos_gpu_init");
    PIPE_CHECK(integration_gpu_init(W, H, config->batch_size, &ctx_r),
               done, "integration_gpu_init R");
    if (config->color_output) {
        PIPE_CHECK(integration_gpu_init(W, H, config->batch_size, &ctx_g),
                   done, "integration_gpu_init G");
        PIPE_CHECK(integration_gpu_init(W, H, config->batch_size, &ctx_b),
                   done, "integration_gpu_init B");
    }

    if (config->calib)
        PIPE_CHECK(calib_gpu_init(config->calib, &calib_ctx), done, "calib_gpu_init");

    /* ---- Single-pass with double-buffered I/O overlap ---- */
    printf("pipeline: single-pass star-detect + align + %s integration "
           "(%d frames, batch=%d)\n",
           config->use_kappa_sigma ? "kappa-sigma" : "mean",
           n_frames, config->batch_size);

    PIPE_CHECK(phase_detect_warp_integrate(frames, n_frames, ref_idx, W, H,
                                            config, calib_ctx,
                                            ctx_r, ctx_g, ctx_b),
               done, "phase_detect_warp_integrate");

    /* ---- Finalise: compute output image(s) and save ---- */
    printf("pipeline: saving to %s\n", config->output_file);
    if (config->color_output) {
        Image img_r = {NULL, W, H};
        Image img_g = {NULL, W, H};
        Image img_b = {NULL, W, H};
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
        if (err == DSO_OK)
            err = image_save_rgb(config->output_file, &img_r, &img_g, &img_b,
                                 &config->save_opts);
        image_free(&img_r); image_free(&img_g); image_free(&img_b);
    } else {
        Image out = {NULL, W, H};
        out.data = (float *)calloc((size_t)W * H, sizeof(float));
        if (!out.data) { err = DSO_ERR_ALLOC; goto done; }
        if (integration_gpu_finalize(ctx_r, n_frames, &out, 0) == DSO_OK)
            err = image_save(config->output_file, &out, &config->save_opts);
        else
            err = DSO_ERR_CUDA;
        image_free(&out);
    }

done:
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
