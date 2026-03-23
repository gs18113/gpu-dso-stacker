/*
 * csv_parser.h — CSV frame-list parser API
 *
 * Accepts only 2-column format (header row always skipped):
 *   filepath, is_reference
 *
 * Any other column count is treated as a parse error.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * csv_parse — read a 2-column frame-list CSV into a dynamically-allocated array.
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
DsoError csv_parse(const char   *filepath,
                   FrameInfo   **frames_out,
                   int          *n_frames_out);

#ifdef __cplusplus
}
#endif
