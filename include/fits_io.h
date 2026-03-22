/*
 * fits_io.h — FITS image I/O API (backed by CFITSIO)
 *
 * Provides three functions for loading and saving float32 astronomical images
 * in the FITS format, and for freeing the allocated pixel buffer.
 *
 * FITS layout: NAXIS1 = width (columns), NAXIS2 = height (rows).
 * Internal layout: row-major float32, pixel (x, y) at data[y*width + x].
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * fits_load — read a FITS image into a host float32 buffer.
 *
 * Supports any BITPIX value; CFITSIO converts to float automatically.
 * On success, out->data is heap-allocated and must be freed with image_free().
 *
 * Returns DSO_OK or DSO_ERR_FITS / DSO_ERR_ALLOC / DSO_ERR_INVALID_ARG.
 */
DsoError fits_load(const char *filepath, Image *out);

/*
 * fits_load_to_buffer — Load FITS image data directly into a provided buffer.
 *
 * Avoids extra allocations/memcpys when loading into pre-allocated memory
 * (e.g., pinned memory for CUDA).
 * W, H must match the actual file dimensions exactly.
 */
DsoError fits_load_to_buffer(const char *filepath, float *buffer, int W, int H);

/*
 * fits_save — write a host float32 image to a FITS file (BITPIX=-32).
 *
 * Overwrites any existing file at filepath (CFITSIO '!' prefix).
 *
 * Returns DSO_OK or DSO_ERR_FITS / DSO_ERR_INVALID_ARG.
 */
DsoError fits_save(const char *filepath, const Image *img);

/*
 * image_free — release the data buffer inside an Image struct.
 *
 * Sets img->data = NULL after freeing.  Safe to call on a zero-initialised
 * or already-freed Image.
 */
void image_free(Image *img);

/*
 * fits_get_bayer_pattern — read the BAYERPAT keyword from a FITS header.
 *
 * Recognised values (case-insensitive): "RGGB", "BGGR", "GRBG", "GBRG".
 * If the keyword is absent or the file cannot be opened, *pattern_out is set
 * to BAYER_NONE and DSO_OK is returned — a missing keyword is not an error.
 *
 * Returns DSO_OK or DSO_ERR_FITS / DSO_ERR_INVALID_ARG.
 */
DsoError fits_get_bayer_pattern(const char *filepath, BayerPattern *pattern_out);

#ifdef __cplusplus
}
#endif
