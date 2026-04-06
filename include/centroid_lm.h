/* -------------------------------------------------------------------------
 * centroid_lm.h — Levenberg-Marquardt 2D Gaussian centroid refinement (CPU)
 *
 * Refines star centroids from center-of-mass (CoM) initial guesses by
 * fitting a 5-parameter circular Gaussian:
 *   f(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2)) + B
 *
 * Parameters: (x0, y0, A, sigma, B)
 * All internal computation uses double precision.
 * Stars that fail to converge keep their original CoM positions.
 * ------------------------------------------------------------------------- */
#ifndef CENTROID_LM_H
#define CENTROID_LM_H

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Refine star centroids in-place using LM 2D Gaussian fitting.
 *
 * stars      : StarList from CCL+CoM (modified in-place with refined cx, cy)
 * image      : source luminance image (float32, row-major, W*H)
 * W, H       : image dimensions
 * sigma_init : initial guess for Gaussian sigma (typically moffat_alpha * 0.8)
 * fit_radius : fitting window half-size in pixels (0 = auto from sigma_init)
 * max_iter   : maximum LM iterations per star (0 = default 15)
 *
 * Stars that fail to converge retain their original CoM positions.
 * Returns DSO_OK on success, DSO_ERR_INVALID_ARG for bad inputs. */
DsoError centroid_lm_refine(StarList *stars, const float *image,
                             int W, int H, float sigma_init,
                             float fit_radius, int max_iter);

#ifdef __cplusplus
}
#endif

#endif /* CENTROID_LM_H */
