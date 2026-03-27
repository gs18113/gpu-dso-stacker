#pragma once

#include "dso_types.h"
#include "ransac.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ransac_compute_homography_gpu
 *
 * Triangle/asterism matching on GPU using device-resident star arrays.
 * Returns backward homography (ref -> src).
 */
DsoError ransac_compute_homography_gpu(const StarPos     *d_ref_stars,
                                        int                n_ref,
                                        const StarPos     *d_src_stars,
                                        int                n_src,
                                        const RansacParams *params,
                                        Homography         *H_out,
                                        int                *n_inliers_out,
                                        cudaStream_t        stream);

#ifdef __cplusplus
}
#endif
