/*
 * background.h — Per-frame background normalization.
 *
 * Matches each frame's background level and signal scale to the reference
 * frame before integration.  Uses median for background estimation and
 * MAD (Median Absolute Deviation) scaled by 1.4826 for robust σ estimation.
 *
 * Two modes:
 *   DSO_BG_PER_CHANNEL — normalize R, G, B independently (handles coloured
 *                         light pollution such as sodium streetlights).
 *   DSO_BG_RGB         — normalize all channels by the same luminance-based
 *                         statistics (preserves colour ratios).
 *
 * For mono images both modes behave identically.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * BgCalibMode — background calibration strategy.
 * ------------------------------------------------------------------------- */
typedef enum {
    DSO_BG_NONE        = 0,   /* disabled (default) */
    DSO_BG_PER_CHANNEL = 1,   /* per-channel normalization */
    DSO_BG_RGB         = 2    /* uniform normalization from luminance stats */
} BgCalibMode;

/* -------------------------------------------------------------------------
 * BgStats — background statistics for one image plane.
 * ------------------------------------------------------------------------- */
typedef struct {
    float background;   /* median pixel value (NaN-aware, stride-16 sample) */
    float scale;        /* 1.4826 * MAD (robust σ estimate) */
} BgStats;

/* -------------------------------------------------------------------------
 * bg_compute_stats — estimate background level and scale.
 *
 * Samples every 16th pixel (skipping NaN), sorts the sample to obtain the
 * median, then computes MAD from absolute deviations.  O(npix/16 · log).
 *
 * Returns DSO_ERR_ALLOC on allocation failure, DSO_ERR_INVALID_ARG if
 * data is NULL or npix <= 0, DSO_OK otherwise.  If the sample is empty
 * (all NaN), sets background = 0, scale = 0.
 * ------------------------------------------------------------------------- */
DsoError bg_compute_stats(const float *data, int npix, BgStats *out);

/* -------------------------------------------------------------------------
 * bg_normalize_cpu — normalize a frame in-place to match ref stats.
 *
 * For each non-NaN pixel:
 *   pixel = (pixel - frame_bg) * (ref_scale / frame_scale) + ref_bg
 *
 * If frame_stats->scale < 1e-10 (essentially flat frame), the function
 * returns DSO_OK without modifying data.  OpenMP-parallelized.
 * ------------------------------------------------------------------------- */
DsoError bg_normalize_cpu(float *data, int npix,
                           const BgStats *frame_stats,
                           const BgStats *ref_stats);

#ifdef __cplusplus
}
#endif
