/*
 * pipeline.h — Full DSO-stacking pipeline orchestrator.
 *
 * Single-pass execution: each frame is read from disk exactly once.
 * Reference frame is processed first (to build ref_stars), then all
 * non-reference frames follow.  Per-frame sequence:
 *
 *   fits_load_to_buffer → H2D (stream_copy)
 *   → calib → debayer_lum → Moffat+threshold  (stream_compute)
 *   → D2H → CCL+CoM → RANSAC                  (CPU)
 *   → Lanczos warp → integration batch slot    (stream_compute)
 *
 * I/O overlap: after RANSAC the GPU warp for frame m runs concurrently
 * with CPU loading frame m+1 from disk + async H2D on stream_copy.
 * At batch boundaries the H2D of frame m+1 also overlaps batch integration.
 *
 * Mini-batch kappa-sigma integration caps peak GPU memory to batch_size
 * warped frames (default 16).
 */

#pragma once

#include "dso_types.h"         /* MoffatParams lives here */
#include "ransac.h"            /* RansacParams */
#include "image_io.h"          /* ImageSaveOptions */

/* Forward declaration — include calibration.h where the full type is needed */
typedef struct CalibFrames CalibFrames;

#ifdef __cplusplus
#include <stdbool.h>
extern "C" {
#endif

typedef enum {
    DSO_BACKEND_AUTO  = 0,
    DSO_BACKEND_CPU   = 1,
    DSO_BACKEND_CUDA  = 2,
    DSO_BACKEND_METAL = 3
} DsoBackend;

/* -------------------------------------------------------------------------
 * IntegrationMethod — stacking combination strategy.
 * ------------------------------------------------------------------------- */
typedef enum {
    DSO_INTEGRATE_MEAN        = 0,
    DSO_INTEGRATE_KAPPA_SIGMA = 1,
    DSO_INTEGRATE_MEDIAN      = 2
} IntegrationMethod;

/* -------------------------------------------------------------------------
 * PipelineConfig — complete configuration for one pipeline run.
 * ------------------------------------------------------------------------- */
typedef struct {
    /* --- Backend selection --- */
    DsoBackend    backend;         /* auto | cpu | cuda | metal (default: auto) */

    /* --- Star detection --- */
    float        star_sigma;      /* detection threshold: mean + star_sigma*σ (3.0) */
    MoffatParams moffat;          /* Moffat kernel shape {alpha=2.5, beta=2.0}       */
    int          top_stars;       /* top-K stars to use for matching (50)            */
    int          min_stars;       /* minimum detected stars to attempt alignment (20)*/

    /* --- RANSAC --- */
    RansacParams ransac;          /* {max_iters=1000, inlier_thresh=2.0,
                                      match_radius=30.0, confidence=0.99,
                                      min_inliers=10}                                */

    /* --- Integration --- */
    int          batch_size;      /* frames per GPU mini-batch (16)                  */
    float        kappa;           /* kappa-sigma threshold (3.0)                     */
    int          iterations;      /* sigma-clipping passes per pixel (3)             */
    IntegrationMethod integration_method; /* mean / kappa-sigma / median             */

    /* --- I/O --- */
    const char         *output_file;     /* output FITS path                                */
    BayerPattern        bayer_override;  /* BAYER_NONE = auto-detect per frame header       */
    int                 use_gpu_lanczos; /* 1 = GPU Lanczos (default), 0 = CPU Lanczos      */
    int                 use_gpu_ransac;  /* 1 = GPU triangle matching, 0 = CPU triangle
                                             matching; default follows use_gpu_lanczos       */

    /* --- Calibration --- */
    const CalibFrames  *calib;           /* NULL = no calibration; applied before debayer   */

    /* --- Color output --- */
    int color_output;  /* 1 = produce 3-plane RGB output; 0 = luminance (default) */

    /* --- Output format --- */
    ImageSaveOptions save_opts; /* bit depth, compression, stretch; see image_io.h */
} PipelineConfig;

/*
 * pipeline_run — execute the full DSO stacking pipeline.
 *
 * frames   : parsed FrameInfo array (from csv_parse); H fields of
 *            non-reference frames are populated in-place by RANSAC.
 * n_frames : number of frames (must be ≥ 1)
 * ref_idx  : index of the reference frame in frames[] (0 ≤ ref_idx < n)
 * config   : pipeline configuration; must not be NULL
 *
 * Returns DSO_OK or the first non-OK error encountered.  All allocated GPU
 * and host resources are freed before returning, even on error.
 */
DsoError pipeline_run(FrameInfo            *frames,
                       int                   n_frames,
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
                            int                   ref_idx,
                            const PipelineConfig *config);

/* CUDA backend entry point (internal dispatch target). */
DsoError pipeline_run_cuda(FrameInfo            *frames,
                            int                   n_frames,
                            int                   ref_idx,
                            const PipelineConfig *config);

/*
 * pipeline_run_metal — Metal backend entry point.
 *
 * Phase-1 implementation provides orchestration and safe fallback behavior.
 * Full Metal stage kernels are introduced incrementally.
 */
DsoError pipeline_run_metal(FrameInfo            *frames,
                             int                   n_frames,
                             int                   ref_idx,
                             const PipelineConfig *config);

#ifdef __cplusplus
}
#endif
