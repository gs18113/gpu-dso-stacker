/*
 * star_detect_cpu.h — CPU-side star detection.
 *
 * Provides two complementary stages:
 *
 * Stage A — Moffat convolution + sigma threshold (new, mirrors star_detect_gpu)
 *   star_detect_cpu_moffat_convolve: 2-D convolution with a normalised Moffat
 *     PSF kernel.  Enhances circular star profiles and suppresses noise.
 *   star_detect_cpu_threshold: compute global mean+σ of the convolved image;
 *     write a binary mask where convolved > mean + sigma_k·σ.
 *   star_detect_cpu_detect: combined entry point (convolve + threshold).
 *
 * Stage B — CCL + weighted center-of-mass (original)
 *   star_detect_cpu_ccl_com: groups mask pixels into blobs via 8-connectivity
 *     CCL with union-find; computes sub-pixel centroids and flux rankings.
 *
 * Typical call sequence (CPU-only pipeline):
 *   star_detect_cpu_detect(lum, conv, mask, W, H, &moffat, sigma_k);
 *   star_detect_cpu_ccl_com(mask, lum, conv, W, H, top_k, &stars);
 */

#pragma once

#include "dso_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Stage A — Moffat convolution + sigma threshold
 * ------------------------------------------------------------------------- */

/*
 * star_detect_cpu_moffat_convolve — 2-D Moffat convolution on the CPU.
 *
 * Convolves src with a normalised Moffat kernel K(i,j) = [1+(i²+j²)/α²]^(-β).
 * Kernel radius R = min((int)ceilf(3·alpha), 15); kernel size (2R+1)².
 * Out-of-bounds pixels are treated as 0 (zero-padding, matches GPU behaviour).
 * Inner loops (per output pixel) are parallelised with OpenMP.
 *
 * src    : float32 input image, W×H
 * dst    : float32 output image, W×H (pre-allocated)
 * W, H   : image dimensions
 * params : Moffat kernel parameters {alpha, beta}
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG / DSO_ERR_ALLOC.
 */
DsoError star_detect_cpu_moffat_convolve(const float        *src,
                                          float              *dst,
                                          int                 W, int H,
                                          const MoffatParams *params);

/*
 * star_detect_cpu_threshold — sigma-based binary mask from a convolved image.
 *
 * Computes the global mean μ and Bessel-corrected sample standard deviation σ
 * of the convolved image, then writes mask[i] = (convolved[i] > μ + sigma_k·σ).
 *
 * convolved : float32 Moffat-response image, W×H
 * mask      : uint8 output (1=star, 0=sky), W×H bytes, pre-allocated
 * W, H      : image dimensions
 * sigma_k   : detection threshold multiplier (e.g. 3.0)
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG.
 */
DsoError star_detect_cpu_threshold(const float *convolved,
                                    uint8_t     *mask,
                                    int          W, int H,
                                    float        sigma_k);

/*
 * star_detect_cpu_detect — combined Moffat convolution + threshold.
 *
 * Convenience wrapper: calls moffat_convolve then threshold in sequence.
 * conv_out and mask_out must be pre-allocated (W×H floats and W×H bytes).
 *
 * Returns the first non-OK error or DSO_OK.
 */
DsoError star_detect_cpu_detect(const float        *src,
                                 float              *conv_out,
                                 uint8_t            *mask_out,
                                 int                 W, int H,
                                 const MoffatParams *params,
                                 float               sigma_k);

/*
 * star_detect_cpu_ccl_com — label blobs and compute star centroids.
 *
 * Inputs:
 *   mask      : uint8 binary image from GPU threshold (1 = star, 0 = sky);
 *               must be W × H bytes.
 *   original  : float32 original (non-convolved) image, W × H floats;
 *               used as CoM weights (values clamped to ≥ 0 to handle
 *               imperfect sky subtraction).
 *   convolved : float32 Moffat-convolved image, W × H floats;
 *               used only for per-blob flux accumulation and ranking.
 *   W, H      : image width and height in pixels.
 *   top_k     : maximum number of stars to return, ranked by flux descending.
 *               Pass INT_MAX to return all detected blobs.
 *
 * Output:
 *   list_out  : on success, list_out->stars points to a heap-allocated array
 *               of min(n_blobs, top_k) StarPos structs sorted by flux
 *               descending; list_out->n is the count.
 *               Caller must free(list_out->stars) when done.
 *               On entry, list_out->stars must be NULL (or the call will leak).
 *
 * Algorithm (two-pass 8-connectivity CCL with union-find):
 *   Pass 1 (raster scan): for each star pixel (x,y), look at the 4 already-
 *     visited neighbours in the raster order: N(0,-1), NW(-1,-1), W(-1,0),
 *     NE(+1,-1).  Assign the minimum active label; union their roots.
 *   Pass 2 (raster scan): replace each label by its fully-compressed root.
 *   Stats pass: accumulate flux, sum_w, sum_wx, sum_wy per root label.
 *   CoM: x = sum_wx / sum_w, y = sum_wy / sum_w.
 *     If sum_w ≈ 0 (all-negative original image), fall back to the unweighted
 *     geometric centroid (sum of (x,y) / pixel count).
 *   Sort by flux descending, truncate to top_k.
 *
 * Returns DSO_OK or DSO_ERR_INVALID_ARG / DSO_ERR_ALLOC.
 */
DsoError star_detect_cpu_ccl_com(const uint8_t *mask,
                                  const float   *original,
                                  const float   *convolved,
                                  int            W, int H,
                                  int            top_k,
                                  StarList       *list_out);

#ifdef __cplusplus
}
#endif
