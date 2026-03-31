/*
 * raw_io.h — RAW camera file I/O via LibRaw.
 *
 * Provides functions to load RAW camera files (CR2, NEF, ARW, DNG, etc.)
 * as float32 Bayer mosaic data, extract Bayer pattern and dimensions.
 *
 * LibRaw is used ONLY to unpack the raw sensor data — no demosaicing is
 * performed.  The stacker's own VNG debayer pipeline handles color
 * reconstruction.
 *
 * Pixel conversion:
 *   pixel_out = clamp((raw_pixel - black[c]) / (maximum - black[c]), 0, 1)
 * where black[c] is the per-channel black level (cblack[c] + global black)
 * and maximum is the sensor white point.
 *
 * Only compiled when DSO_HAS_LIBRAW=1 (cmake -DDSO_ENABLE_LIBRAW=ON).
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * raw_load — load a RAW camera file as a float32 Bayer mosaic.
 *
 * Uses LibRaw to unpack raw sensor data WITHOUT demosaicing.
 * Output is black-subtracted and normalized to [0.0, 1.0].
 * Uses the usable image area (sizes.width × sizes.height), not the full
 * sensor area including masked border pixels.
 *
 * out->data is heap-allocated; caller must free with image_free().
 *
 * Returns DSO_OK or DSO_ERR_RAW / DSO_ERR_ALLOC / DSO_ERR_INVALID_ARG.
 */
DsoError raw_load(const char *filepath, Image *out);

/*
 * raw_load_to_buffer — load RAW data into a pre-allocated float32 buffer.
 *
 * Used by the GPU pipeline for direct loading into pinned memory.
 * W and H must match the usable sensor dimensions exactly.
 *
 * Returns DSO_OK or DSO_ERR_RAW / DSO_ERR_INVALID_ARG.
 */
DsoError raw_load_to_buffer(const char *filepath, float *buffer, int W, int H);

/*
 * raw_get_bayer_pattern — extract the CFA pattern from a RAW file header.
 *
 * Maps LibRaw's idata.filters to BayerPattern.  Only opens the file header
 * (no unpack), so this is a lightweight metadata read.
 *
 * Sets *pattern_out to BAYER_NONE for non-Bayer sensors (X-Trans, Foveon).
 *
 * Returns DSO_OK or DSO_ERR_RAW / DSO_ERR_INVALID_ARG.
 */
DsoError raw_get_bayer_pattern(const char *filepath, BayerPattern *pattern_out);

/*
 * raw_get_dimensions — read usable image width and height from a RAW file.
 *
 * Only opens the file header (no unpack).
 *
 * Returns DSO_OK or DSO_ERR_RAW / DSO_ERR_INVALID_ARG.
 */
DsoError raw_get_dimensions(const char *filepath, int *width_out, int *height_out);

#ifdef __cplusplus
}
#endif
