/*
 * ransac.h — Star-based homography estimation via RANSAC + DLT.
 *
 * Given two sets of star centroids (reference frame and current frame),
 * this module:
 *   1. Matches stars by nearest-neighbour with a Lowe ratio test.
 *   2. Estimates the 3×3 homography H that maps reference pixels to source
 *      pixels (the backward map, ref → src) via the Direct Linear Transform
 *      (DLT) solved with Jacobi eigendecomposition of AᵀA (9×9).
 *   3. Runs RANSAC with adaptive early termination to reject false matches.
 *   4. Re-estimates H from all inliers for a final least-squares refinement.
 *
 * Homography convention:
 *   H is the BACKWARD map (ref → src), consistent with the rest of the
 *   library.  The DLT row setup directly encodes this convention:
 *     Row 1: [-rx, -ry, -1,   0,   0,  0, sx*rx, sx*ry, sx]
 *     Row 2: [  0,   0,  0, -rx, -ry, -1, sy*rx, sy*ry, sy]
 *   so the null vector h satisfies H*p_ref ∝ p_src without any inversion.
 *
 * Normalisation:
 *   Before building the DLT matrix the reference points are normalised:
 *   centroid shifted to origin, mean distance to origin scaled to √2.
 *   The same normalisation is applied to source points.  The recovered H
 *   is de-normalised before return.  This improves conditioning when star
 *   coordinates have large absolute values.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * RansacParams — configuration for the RANSAC homography estimator.
 * ------------------------------------------------------------------------- */
typedef struct {
    int   max_iters;     /* maximum RANSAC iterations (default: 1000)          */
    float inlier_thresh; /* reprojection error threshold in pixels (default: 2) */
    float match_radius;  /* max star-to-star distance for a valid match (px)    */
    float confidence;    /* desired success probability (default: 0.99)         */
    int   min_inliers;   /* minimum inliers to declare success (default: 4)     */
} RansacParams;

/* -------------------------------------------------------------------------
 * dlt_homography — compute a backward homography from N ≥ 4 correspondences.
 *
 * ref_pts : N reference-frame star positions (input of H)
 * src_pts : N source-frame star positions    (output of H)
 * n       : number of correspondences (must be ≥ 4)
 * H_out   : receives the normalised-then-denormalised 3×3 backward homography.
 *           The homogeneous scale is fixed by setting h[8] = 1 (last element).
 *
 * Algorithm:
 *   Build the 2N×9 matrix A from DLT equations; compute M = AᵀA (9×9
 *   symmetric); find the eigenvector of M corresponding to the smallest
 *   eigenvalue via classical Jacobi iteration; reshape to 3×3 and
 *   de-normalise.
 *
 * Returns DSO_OK, DSO_ERR_INVALID_ARG (n < 4 or degenerate), or
 *         DSO_ERR_ALLOC.
 * ------------------------------------------------------------------------- */
DsoError dlt_homography(const StarPos *ref_pts,
                         const StarPos *src_pts,
                         int            n,
                         Homography    *H_out);

/* -------------------------------------------------------------------------
 * ransac_compute_homography — match stars and estimate backward homography.
 *
 * ref_list     : StarList of reference-frame stars (sorted by flux)
 * frm_list     : StarList of current-frame stars   (sorted by flux)
 * params       : RANSAC configuration (use NULL for defaults)
 * H_out        : backward homography on success
 * n_inliers_out: if non-NULL, receives the final inlier count
 *
 * Star matching:
 *   For each star in ref_list, find the closest star in frm_list within
 *   params->match_radius.  The Lowe ratio test (d1/d2 < 0.8) rejects
 *   ambiguous matches (requires ≥ 2 candidate neighbours).
 *
 * RANSAC:
 *   Draws 4-point minimal samples, estimates H via dlt_homography, counts
 *   inliers (reprojection error < inlier_thresh after homogeneous divide).
 *   Uses adaptive termination: after each improvement the remaining
 *   iteration count is recomputed from N = log(1-conf)/log(1-p^4).
 *
 * Returns DSO_OK, DSO_ERR_RANSAC (not enough inliers), DSO_ERR_STAR_DETECT
 *         (too few stars for a 4-point sample), or DSO_ERR_INVALID_ARG.
 * ------------------------------------------------------------------------- */
DsoError ransac_compute_homography(const StarList     *ref_list,
                                    const StarList     *frm_list,
                                    const RansacParams *params,
                                    Homography         *H_out,
                                    int                *n_inliers_out);

#ifdef __cplusplus
}
#endif
