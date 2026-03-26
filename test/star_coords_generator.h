#pragma once

#include "ransac.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generate synthetic star lists for RANSAC testing.
 *
 * - Creates n_inliers random coordinates in the reference list.
 * - Applies H_ref_to_frame to those coordinates to create matching frame stars.
 * - Appends random outlier stars to each list independently.
 *
 * Output ordering is deterministic:
 *   [0, n_inliers)            = true correspondences
 *   [n_inliers, list->n)      = outliers
 */
int star_coords_generate(int n_inliers,
                         int n_ref_outliers,
                         int n_frame_outliers,
                         int width,
                         int height,
                         const Homography *H_ref_to_frame,
                         unsigned int seed,
                         StarList *ref_out,
                         StarList *frame_out);

void star_coords_free(StarList *list);

#ifdef __cplusplus
}
#endif
