/*
 * csv_parser.h — CSV frame-list parser API
 *
 * Parses an input CSV that lists FITS frames together with their
 * pre-computed homographies.  Expected format (first row = header, skipped):
 *
 *   filepath, is_reference, h00, h01, h02, h10, h11, h12, h20, h21, h22
 *
 * The nine h-values are the row-major 3×3 forward homography coefficients
 * that map source pixel coordinates to reference pixel coordinates.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * csv_parse — read a frame-list CSV into a dynamically-allocated array.
 *
 * On success:
 *   *frames_out   → heap-allocated array of n FrameInfo structs
 *   *n_frames_out → number of valid rows parsed
 *
 * Caller must free(*frames_out) when done.
 * Malformed rows are skipped with a warning; the function still returns DSO_OK
 * if at least one row was parsed successfully.
 *
 * Returns DSO_OK or DSO_ERR_IO / DSO_ERR_ALLOC / DSO_ERR_CSV / DSO_ERR_INVALID_ARG.
 */
DsoError csv_parse(const char *filepath, FrameInfo **frames_out, int *n_frames_out);

#ifdef __cplusplus
}
#endif
