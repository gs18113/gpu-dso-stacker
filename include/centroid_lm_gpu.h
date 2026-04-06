/* -------------------------------------------------------------------------
 * centroid_lm_gpu.h — GPU Levenberg-Marquardt 2D Gaussian centroid refinement
 *
 * Warp-per-star architecture: each CUDA warp (32 threads) cooperatively
 * fits one star using warp-level reductions (__shfl_down_sync).
 * All LM math uses double precision; 5x5 Cholesky on lane 0.
 * ------------------------------------------------------------------------- */
#ifndef CENTROID_LM_GPU_H
#define CENTROID_LM_GPU_H

#include "dso_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Refine star centroids on GPU using warp-per-star LM fitting.
 *
 * stars      : HOST-SIDE StarList (modified in-place with refined positions)
 * d_image    : device float32 source image (W*H, already resident)
 * W, H       : image dimensions
 * sigma_init : initial guess for Gaussian sigma
 * fit_radius : fitting window half-size in pixels (0 = auto from sigma_init)
 * max_iter   : maximum LM iterations per star (0 = default 15)
 * stream     : CUDA stream
 *
 * Internally: H2D the star data, run kernel, D2H the refined positions.
 * Stars that fail to converge retain their original CoM positions. */
DsoError centroid_lm_refine_gpu(StarList *stars, const float *d_image,
                                 int W, int H, float sigma_init,
                                 float fit_radius, int max_iter,
                                 cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* CENTROID_LM_GPU_H */
