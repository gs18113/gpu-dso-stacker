/*
 * transform.h — Polynomial alignment transform models.
 *
 * Provides bilinear (affine, 6 DOF), bisquared (quadratic, 12 DOF),
 * and bicubic (cubic, 20 DOF) coordinate mappings as alternatives to
 * the projective homography.  All transforms are BACKWARD maps:
 * reference pixel coordinates → source pixel coordinates, consistent
 * with the Homography convention used throughout the library.
 *
 * Polynomial models:
 *
 *   Bilinear  (6 params):  sx = a0 + a1*dx + a2*dy
 *                           sy = b0 + b1*dx + b2*dy
 *
 *   Bisquared (12 params): sx = a0 + a1*dx + a2*dy + a3*dx² + a4*dx*dy + a5*dy²
 *                           sy = b0 + b1*dx + b2*dy + b3*dx² + b4*dx*dy + b5*dy²
 *
 *   Bicubic   (20 params): sx = a0 + a1*dx + a2*dy + a3*dx² + a4*dx*dy + a5*dy²
 *                                + a6*dx³ + a7*dx²*dy + a8*dx*dy² + a9*dy³
 *                           sy = b0 + b1*dx + b2*dy + b3*dx² + b4*dx*dy + b5*dy²
 *                                + b6*dx³ + b7*dx²*dy + b8*dx*dy² + b9*dy³
 *
 * Coefficients are stored sequentially: {a0..aN, b0..bN} in coeffs[].
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * TransformModel, PolyTransform, and TRANSFORM_MAX_COEFFS are defined in
 * dso_types.h so that all translation units can reference them without
 * pulling in the transform function declarations or CUDA headers.
 */

/* Number of coefficients per axis for each model. */
#define TRANSFORM_BILINEAR_NCOEFFS   3
#define TRANSFORM_BISQUARED_NCOEFFS  6
#define TRANSFORM_BICUBIC_NCOEFFS   10

/* Minimum correspondences required for a robust fit. */
#define TRANSFORM_BILINEAR_MIN_PTS   3
#define TRANSFORM_BISQUARED_MIN_PTS  6
#define TRANSFORM_BICUBIC_MIN_PTS   10

/* Auto-selection thresholds (practical minimums for robust fitting). */
#define TRANSFORM_AUTO_BICUBIC_THRESH   20
#define TRANSFORM_AUTO_BISQUARED_THRESH 12
#define TRANSFORM_AUTO_BILINEAR_THRESH   6

/* -------------------------------------------------------------------------
 * API
 * ------------------------------------------------------------------------- */

/*
 * transform_eval — evaluate the polynomial transform at (dx, dy).
 *
 * Writes the mapped source coordinates to (*sx, *sy).
 * For TRANSFORM_PROJECTIVE, this function is a no-op (use Homography directly).
 */
void transform_eval(const PolyTransform *T, double dx, double dy,
                    double *sx, double *sy);

/*
 * transform_fit — solve polynomial coefficients from N correspondences.
 *
 * ref_pts are the "input" coordinates (reference frame);
 * src_pts are the "output" coordinates (source frame).
 * model must be BILINEAR, BISQUARED, or BICUBIC.
 *
 * Uses least-squares via normal equations (A^T A x = A^T b) with
 * Cholesky decomposition.  Coordinates are normalized (centroid + scale)
 * before fitting for numerical stability.
 *
 * Returns DSO_ERR_INVALID_ARG if N < minimum required correspondences
 * or if the normal matrix is singular (e.g. collinear points).
 */
DsoError transform_fit(const StarPos *ref_pts, const StarPos *src_pts,
                        int n, TransformModel model, PolyTransform *out);

/*
 * transform_from_homography — wrap a Homography as a PROJECTIVE PolyTransform.
 *
 * Coefficients are unused; the model field is set to TRANSFORM_PROJECTIVE
 * so callers know to use the Homography H directly.
 */
void transform_from_homography(const Homography *H, PolyTransform *out);

/*
 * transform_identity — set identity transform coefficients for the given model.
 *
 * Identity: sx = dx, sy = dy.  For BILINEAR: a0=0, a1=1, a2=0, b0=0, b1=0, b2=1.
 * Higher-order coefficients are zero.
 */
void transform_identity(TransformModel model, PolyTransform *out);

/*
 * transform_auto_select — choose the best model for the given inlier count.
 *
 *   n >= 20 → BICUBIC
 *   n >= 12 → BISQUARED
 *   n >= 6  → BILINEAR
 *   else    → PROJECTIVE
 */
TransformModel transform_auto_select(int n_inliers);

/*
 * transform_reproj_err_sq — squared reprojection error for polynomial transforms.
 *
 * Evaluates T at (rx, ry) and returns the squared Euclidean distance to (sx, sy).
 */
double transform_reproj_err_sq(const PolyTransform *T,
                                float rx, float ry,
                                float sx, float sy);

/*
 * transform_ncoeffs_per_axis — return number of coefficients per axis for a model.
 *
 * Returns 0 for TRANSFORM_PROJECTIVE or TRANSFORM_AUTO.
 */
int transform_ncoeffs_per_axis(TransformModel model);

#ifdef __cplusplus
}
#endif
