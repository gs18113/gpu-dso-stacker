/*
 * pipeline.h — Full DSO-stacking pipeline orchestrator with CUDA stream
 * overlap.
 *
 * This module wires together all processing stages into a single call that
 * reads FITS frames, optionally computes alignment transforms, and produces
 * a stacked output image.
 *
 * Data-flow (2-column CSV, full pipeline):
 *   Phase 1 — Star detection and transform computation
 *     for each frame:
 *       CPU : read FITS → pinned host buffer
 *       streamB (copy): H2D to device src slot
 *       streamA (compute): debayer (VNG) → Moffat convolve → threshold
 *       streamA → D2H threshold mask to host
 *       CPU : CCL + weighted CoM → StarList
 *       CPU : RANSAC → Homography H[i]  (against reference star list)
 *     (CPU reads frame i+1 while GPU processes frame i — I/O overlap)
 *
 *   Phase 2 — Lanczos alignment + integration
 *     for each mini-batch of M frames:
 *       for each frame in batch:
 *         CPU : read FITS → pinned host buffer
 *         streamB: H2D src frame
 *         streamA: debayer → cudaMemset(d_dst, 0) → lanczos_transform_gpu_d2d
 *       streamA: integration_gpu_process_batch (kappa-sigma)
 *     streamA: integration_gpu_finalize
 *     CPU: fits_save output
 *
 * Data-flow (11-column CSV, pre-computed transforms):
 *   Phase 1 is skipped entirely.
 *   Phase 2 proceeds as above, using the H values from the CSV.
 *
 * CUDA stream architecture:
 *   stream_copy  : only cudaMemcpyAsync H2D transfers (no kernels)
 *   stream_compute: all GPU kernels
 *   Pinned host memory double-buffer: pinned[i % 2] avoids stall between
 *   disk I/O and DMA.  Events e_h2d[2] and e_gpu[2] enforce ordering.
 */

#pragma once

#include "dso_types.h"         /* MoffatParams lives here */
#include "ransac.h"            /* RansacParams */

/* Forward declaration — include calibration.h where the full type is needed */
typedef struct CalibFrames CalibFrames;

#ifdef __cplusplus
#include <stdbool.h>
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * PipelineConfig — complete configuration for one pipeline run.
 * ------------------------------------------------------------------------- */
typedef struct {
    /* --- Star detection (ignored when has_transforms = 1) --- */
    float        star_sigma;      /* detection threshold: mean + star_sigma*σ (3.0) */
    MoffatParams moffat;          /* Moffat kernel shape {alpha=2.5, beta=2.0}       */
    int          top_stars;       /* top-K stars to use for matching (50)            */
    int          min_stars;       /* minimum stars required for RANSAC (6)           */

    /* --- RANSAC (ignored when has_transforms = 1) --- */
    RansacParams ransac;          /* {max_iters=1000, inlier_thresh=2.0,
                                      match_radius=30.0, confidence=0.99,
                                      min_inliers=4}                                */

    /* --- Integration --- */
    int          batch_size;      /* frames per GPU mini-batch (16)                  */
    float        kappa;           /* kappa-sigma threshold (3.0)                     */
    int          iterations;      /* sigma-clipping passes per pixel (3)             */
    int          use_kappa_sigma; /* 1 = kappa-sigma, 0 = plain mean                 */

    /* --- I/O --- */
    const char         *output_file;     /* output FITS path                                */
    BayerPattern        bayer_override;  /* BAYER_NONE = auto-detect per frame header       */
    int                 use_gpu_lanczos; /* 1 = GPU Lanczos (default), 0 = CPU Lanczos      */

    /* --- Calibration --- */
    const CalibFrames  *calib;           /* NULL = no calibration; applied before debayer   */
} PipelineConfig;

/*
 * pipeline_run — execute the full DSO stacking pipeline.
 *
 * frames         : parsed FrameInfo array (from csv_parse); the H field of
 *                  non-reference frames is populated in-place when
 *                  has_transforms == 0.
 * n_frames       : number of frames (must be ≥ 1)
 * has_transforms : 1 → skip star detection + RANSAC (use CSV homographies)
 *                  0 → run full detection + alignment
 * ref_idx        : index of the reference frame in frames[] (0 ≤ ref_idx < n)
 * config         : pipeline configuration; must not be NULL
 *
 * Returns DSO_OK or the first non-OK error encountered.  All allocated GPU
 * and host resources are freed before returning, even on error.
 */
DsoError pipeline_run(FrameInfo            *frames,
                       int                   n_frames,
                       int                   has_transforms,
                       int                   ref_idx,
                       const PipelineConfig *config);

/*
 * pipeline_run_cpu — full CPU-only pipeline (no CUDA).
 *
 * Same signature as pipeline_run.  Called automatically by pipeline_run when
 * config->use_gpu_lanczos == 0.  All stages (debayer, star detection, RANSAC,
 * Lanczos warp, integration) execute on the CPU with OpenMP parallelism.
 */
DsoError pipeline_run_cpu(FrameInfo            *frames,
                           int                   n_frames,
                           int                   has_transforms,
                           int                   ref_idx,
                           const PipelineConfig *config);

#ifdef __cplusplus
}
#endif
