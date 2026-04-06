/*
 * frame_quality.h — Per-frame quality metrics for sub-frame scoring.
 *
 * Computes FWHM, roundness, background, and a composite quality score
 * from star detection data (convolution map + star list) that is already
 * available after the Moffat detection stage. Entirely CPU-side.
 */
#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * FrameQuality — quality metrics for a single frame.
 *
 * fwhm       : median full-width-at-half-maximum across detected stars (pixels)
 * roundness  : median min(fwhm_x,fwhm_y)/max(fwhm_x,fwhm_y) per star (1.0=circular)
 * background : sigma-clipped mean of the luminance image
 * star_count : number of detected stars
 * composite  : raw composite quality score (before normalization)
 * normalized : score normalized so the reference frame = 100.0
 */
typedef struct {
    float fwhm;
    float roundness;
    float background;
    int   star_count;
    float composite;
    float normalized;
} FrameQuality;

/*
 * frame_quality_compute — compute all quality metrics for one frame.
 *
 * conv_data   : Moffat convolution output (W × H float32)
 * lum_data    : debayered luminance image (W × H float32)
 * stars       : detected star list from ccl_com (positions + flux)
 * W, H        : image dimensions
 * quality_out : filled on success
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG (NULL pointers).
 */
DsoError frame_quality_compute(const float    *conv_data,
                               const float    *lum_data,
                               const StarList *stars,
                               int W, int H,
                               FrameQuality   *quality_out);

/*
 * frame_quality_normalize — scale composite score relative to the reference
 * frame so that reference = 100.0.
 *
 * q             : quality struct whose .normalized field will be set
 * ref_composite : the reference frame's .composite value
 */
void frame_quality_normalize(FrameQuality *q, float ref_composite);

/*
 * frame_quality_print_table — print a summary table of all frames' quality.
 *
 * qualities     : array of FrameQuality (one per scored frame)
 * frame_indices : CSV-order index for each entry
 * rejected      : 1 = rejected by quality gate, 0 = accepted
 * n             : number of entries
 */
void frame_quality_print_table(const FrameQuality *qualities,
                               const int          *frame_indices,
                               const int          *rejected,
                               int                 n);

#ifdef __cplusplus
}
#endif
