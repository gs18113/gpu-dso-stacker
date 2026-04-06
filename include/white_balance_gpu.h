/*
 * white_balance_gpu.h — GPU white balance for raw Bayer mosaics.
 *
 * Provides a device-to-device kernel that applies per-channel multipliers
 * to the raw Bayer mosaic in-place.  Called on stream_compute between
 * calib_gpu_apply_d2d and debayer_gpu_d2d.
 */

#pragma once

#include "dso_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wb_apply_bayer_gpu_d2d — apply white balance to a device Bayer mosaic.
 *
 * Each pixel is multiplied in-place by the channel multiplier matching
 * its CFA position.  BAYER_NONE → immediate return (no-op).
 *
 * d_data : device buffer, W*H floats (modified in-place)
 * stream : CUDA stream (0 = default)
 */
DsoError wb_apply_bayer_gpu_d2d(float *d_data, int W, int H,
                                 BayerPattern pattern,
                                 float r_mul, float g_mul, float b_mul,
                                 cudaStream_t stream);

#ifdef __cplusplus
}
#endif
