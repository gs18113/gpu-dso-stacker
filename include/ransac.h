/*
 * ransac.h — Star-based homography estimation via triangle/asterism matching.
 *
 * Given two sets of star centroids (reference frame and source frame), this
 * module:
 *   1. Generates all valid 3-point triangles in both sets.
 *   2. Computes invariant triangle hashes (r1=a/c, r2=b/c with a≤b≤c).
 *   3. Matches hashes within tolerance and accumulates a src×ref vote matrix.
 *   4. Extracts high-vote correspondences and solves backward H (ref→src) with
 *      DLT + Jacobi eigendecomposition.
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
 * ransac_compute_homography — triangle matching + DLT refinement (CPU, C11).
 *
 * ref_list     : StarList of reference-frame stars
 * src_list     : StarList of source-frame stars
 * params       : matching/consensus settings (NULL = internal defaults)
 * H_out        : backward homography (ref → src)
 * n_inliers_out: optional final inlier count from reprojection thresholding
 * ------------------------------------------------------------------------- */
DsoError ransac_compute_homography(const StarList     *ref_list,
                                    const StarList     *src_list,
                                    const RansacParams *params,
                                    Homography         *H_out,
                                    int                *n_inliers_out);

#ifdef __cplusplus
}
#endif
