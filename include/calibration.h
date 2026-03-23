/*
 * calibration.h — Astronomical calibration frame support.
 *
 * Provides master-frame generation (dark, bias, flat, darkflat) and
 * application of calibration to raw light frames before debayering.
 *
 * Calibration formula
 * -------------------
 * With bias:
 *   dark_master = stack(dark_raw - bias_master)
 *   flat_master = stack(normalize(flat_raw - bias_master))
 *
 * With darkflat (preferred when flat exposures > 1 s):
 *   dark_master = stack(dark_raw)
 *   flat_master = stack(normalize(flat_raw - darkflat_master))
 *
 * Applied to each light frame (before debayering):
 *   light_cal = (light_raw - dark_master) / flat_master
 *
 * bias and darkflat are mutually exclusive.  Each of the four calibration
 * types is individually optional.
 *
 * Master-frame generation
 * -----------------------
 * Each calibration path argument may be:
 *   (a) A FITS file — treated as a pre-computed master frame.
 *   (b) A plain-text file — one FITS path per line; frames are stacked to
 *       produce a master using the chosen method.
 *
 * Stacking methods
 * ----------------
 * CALIB_WINSORIZED_MEAN (default)
 *   Sort the N values per pixel, replace the bottom g = floor(0.1*N) and
 *   top g values with their respective boundary values, then compute the
 *   mean.  Uses double-precision accumulation to avoid overflow.
 *
 * CALIB_MEDIAN
 *   Sort the N values per pixel, return the middle element (or the average
 *   of the two middle elements when N is even).
 *
 * Flat normalization
 * ------------------
 * Before stacking, each flat frame is divided by its own mean so that all
 * flats contribute equally regardless of illumination level.  The stacked
 * master flat has a mean close to 1.0 and is used directly as a divisor.
 *
 * Dead-pixel guard
 * ----------------
 * Flat pixels below 1e-6 are treated as dead; division is replaced by 0.
 */

#pragma once

#include "dso_types.h"   /* Image, DsoError */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * CalibMethod — stacking algorithm for master frame generation.
 * ------------------------------------------------------------------------- */
typedef enum {
    CALIB_WINSORIZED_MEAN = 0,
    CALIB_MEDIAN
} CalibMethod;

/* -------------------------------------------------------------------------
 * CalibFrames — ready-to-use master calibration frames.
 *
 * After calib_load_or_generate() returns DSO_OK:
 *   has_dark == 1  →  dark.data is valid (bias-subtracted if bias provided)
 *   has_flat == 1  →  flat.data is valid (normalized, mean ≈ 1.0)
 *
 * Zero-initialize this struct before passing to calib_load_or_generate().
 * Call calib_free() when done.
 * ------------------------------------------------------------------------- */
typedef struct CalibFrames {
    Image dark;
    Image flat;
    int   has_dark;
    int   has_flat;
} CalibFrames;

/* -------------------------------------------------------------------------
 * calib_load_or_generate — build CalibFrames from CLI-provided paths.
 *
 * Any NULL path is silently skipped.
 * save_dir may be NULL (no saving); created with mkdir -p if it does not exist.
 *
 * wsor_clip : winsorized-mean clipping fraction per side (default 0.1).
 *   Fraction of values replaced on each end before computing the mean.
 *   E.g. 0.1 replaces the bottom 10% and top 10% with boundary values.
 *   Valid range: [0.0, 0.49].  Values outside this range are clamped.
 *   Only used when method == CALIB_WINSORIZED_MEAN.
 *
 * Generated master frames are saved to:
 *   <save_dir>/master_dark.fits
 *   <save_dir>/master_flat.fits
 *
 * Returns DSO_OK or the first error encountered.
 * ------------------------------------------------------------------------- */
DsoError calib_load_or_generate(
    const char *dark_path,     CalibMethod dark_method,
    const char *bias_path,     CalibMethod bias_method,
    const char *flat_path,     CalibMethod flat_method,
    const char *darkflat_path, CalibMethod darkflat_method,
    const char *save_dir,
    float        wsor_clip,
    CalibFrames *calib_out);

/* -------------------------------------------------------------------------
 * calib_apply_cpu — apply calibration in-place to a raw pre-debayer image.
 *
 * img dimensions must match calib master dimensions; returns
 * DSO_ERR_INVALID_ARG if they differ.
 * ------------------------------------------------------------------------- */
DsoError calib_apply_cpu(Image *img, const CalibFrames *calib);

/* -------------------------------------------------------------------------
 * calib_free — release master frame memory.  Safe to call on zero-init struct.
 * ------------------------------------------------------------------------- */
void calib_free(CalibFrames *calib);

#ifdef __cplusplus
}
#endif
