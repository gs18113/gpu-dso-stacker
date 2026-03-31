/*
 * frame_load.h — Format-agnostic image loading dispatch layer.
 *
 * Routes load/metadata requests to the appropriate backend (FITS or RAW)
 * based on file extension.  When DSO_HAS_LIBRAW=0, RAW extensions produce
 * a diagnostic error.
 *
 * Supported RAW extensions (case-insensitive):
 *   .cr2  .cr3  .nef  .arw  .orf  .rw2  .raf  .dng
 *   .pef  .srw  .raw  .3fr  .iiq  .rwl  .nrw
 *
 * All other extensions are assumed to be FITS.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * frame_is_raw — return 1 if the file extension is a recognized RAW format.
 *
 * Pure extension check; does not open the file.
 */
int frame_is_raw(const char *filepath);

/*
 * frame_load — load any supported image format as float32.
 *
 * Dispatches to raw_load() or fits_load() based on file extension.
 * out->data is heap-allocated; caller must free with image_free().
 *
 * Returns DSO_OK or the backend-specific error code.
 */
DsoError frame_load(const char *filepath, Image *out);

/*
 * frame_load_to_buffer — load into a pre-allocated float32 buffer.
 *
 * Used by the GPU pipeline for direct loading into pinned memory.
 * W and H must match the image dimensions exactly.
 */
DsoError frame_load_to_buffer(const char *filepath, float *buffer, int W, int H);

/*
 * frame_get_bayer_pattern — read the Bayer/CFA pattern from any supported format.
 *
 * For FITS: reads the BAYERPAT keyword from the header.
 * For RAW: reads the CFA pattern from LibRaw metadata.
 *
 * Sets *pattern_out to BAYER_NONE if the pattern cannot be determined.
 */
DsoError frame_get_bayer_pattern(const char *filepath, BayerPattern *pattern_out);

/*
 * frame_get_dimensions — read image width and height without loading pixels.
 *
 * For RAW: lightweight header-only read via LibRaw.
 * For FITS: opens and reads NAXIS1/NAXIS2 via CFITSIO.
 */
DsoError frame_get_dimensions(const char *filepath, int *width_out, int *height_out);

#ifdef __cplusplus
}
#endif
