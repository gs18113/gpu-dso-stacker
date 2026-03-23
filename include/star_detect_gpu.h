/*
 * star_detect_gpu.h — GPU star detection via Moffat convolution and
 * sigma-threshold masking.
 *
 * Star detection proceeds in three GPU passes:
 *
 *   1. Moffat convolution
 *      The image is convolved with a normalised 2D Moffat kernel:
 *        K(i,j) = [1 + (i²+j²) / alpha²]^(-beta)   normalised so Σ K = 1.
 *      This enhances circular point-spread-function (PSF) features (stars)
 *      while suppressing extended sources and noise.  The kernel is stored
 *      in GPU constant memory and evaluated on a (2R+1)×(2R+1) window where
 *      R = ceil(3·alpha).  Tile-based convolution with shared-memory apron
 *      reduces global-memory traffic.
 *
 *   2. Global mean+sigma reduction
 *      Two GPU reductions compute the mean μ and sample standard deviation σ
 *      of the entire convolved image.  These statistics describe the noise
 *      floor in the Moffat-response domain.
 *
 *   3. Threshold masking
 *      Each pixel receives mask value 1 if its convolved response exceeds
 *      μ + sigma_k·σ, and 0 otherwise.  sigma_k is the user-settable
 *      centre-of-mass threshold (default 3.0).
 *
 * The resulting uint8 mask is transferred to the host, where the CPU-side
 * CCL+CoM step (star_detect_cpu.h) extracts centroid coordinates.
 */

#pragma once

#include "dso_types.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * star_detect_gpu_moffat_convolve — convolve image with normalised Moffat
 * kernel on the GPU.
 *
 * src:    host-side float32 input image.
 * dst:    host-side float32 output (pre-allocated, same size as src).
 * params: Moffat shape parameters.
 * stream: CUDA stream (0 = default); synchronises before returning.
 *
 * The kernel is computed from params on the CPU, normalised, and uploaded
 * to GPU constant memory each call.  Shared-memory tiling with a (2R)-pixel
 * apron is used; zero-padding at the boundary is acceptable for detection.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_ALLOC / DSO_ERR_INVALID_ARG.
 * ------------------------------------------------------------------------- */
DsoError star_detect_gpu_moffat_convolve(const Image        *src,
                                          Image              *dst,
                                          const MoffatParams *params,
                                          cudaStream_t        stream);

/* -------------------------------------------------------------------------
 * star_detect_gpu_threshold — compute mean+k·sigma and write binary mask.
 *
 * convolved : float32 Moffat-response image (host-side input).
 * mask_out  : uint8 output mask, W × H bytes (caller must pre-allocate);
 *             1 where convolved > mean + sigma_k · sigma, else 0.
 * sigma_k   : threshold multiplier (e.g. 3.0).
 * stream    : CUDA stream (0 = default); synchronises before returning.
 *
 * The mean and stddev are computed via a two-pass GPU tree reduction.
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 * ------------------------------------------------------------------------- */
DsoError star_detect_gpu_threshold(const Image  *convolved,
                                    uint8_t      *mask_out,
                                    float         sigma_k,
                                    cudaStream_t  stream);

/* -------------------------------------------------------------------------
 * star_detect_gpu_d2d — device-to-device combined convolve + threshold.
 *
 * This variant is used in the pipelined orchestrator where both input and
 * scratch buffers are already on the device.  The threshold mask is written
 * to d_mask (device uint8, W × H bytes) and must be transferred to the host
 * by the caller before CCL+CoM.
 *
 * d_src  : device float32 input (W × H floats, already debayered)
 * d_conv : device float32 scratch buffer for Moffat output (W × H floats,
 *          pre-allocated; contents overwritten by this call)
 * d_mask : device uint8 output threshold mask (W × H bytes, pre-allocated)
 * W, H   : image dimensions
 * params : Moffat kernel parameters
 * sigma_k: threshold multiplier
 * stream : CUDA stream; executes asynchronously — caller synchronises
 *
 * Returns DSO_OK or DSO_ERR_CUDA / DSO_ERR_INVALID_ARG.
 * ------------------------------------------------------------------------- */
DsoError star_detect_gpu_d2d(const float        *d_src,
                              float              *d_conv,
                              uint8_t            *d_mask,
                              int                 W, int H,
                              const MoffatParams *params,
                              float               sigma_k,
                              cudaStream_t        stream);

#ifdef __cplusplus
}
#endif
