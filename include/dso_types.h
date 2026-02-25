/*
 * dso_types.h — Shared data types for the DSO stacker library
 *
 * All types are plain C structs/enums so they can be shared across
 * C translation units (fits_io.c, lanczos_cpu.c, …) and included from
 * C++ code (main.cpp, lanczos_gpu.cu) without name-mangling issues.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Homography — row-major 3×3 perspective transform matrix.
 *
 *   h[r*3 + c] == H(row r, column c)
 *
 * Convention used throughout the library:
 *   H maps source pixel coordinates → reference pixel coordinates (forward).
 *   The transform functions receive this H and internally invert it to
 *   perform backward mapping (reference pixel → source sample position).
 */
typedef struct {
    double h[9];
} Homography;

/*
 * Image — float32 host image buffer.
 *
 *   Pixel at column x, row y: data[y * width + x]   (row-major)
 *
 * Ownership: whoever allocates data is responsible for calling image_free()
 * (defined in fits_io.h).  A zero-initialised Image is safe to pass to
 * image_free() without allocating first.
 */
typedef struct {
    float *data;
    int    width;
    int    height;
} Image;

/*
 * FrameInfo — one row from the input CSV file.
 *
 * filepath     : absolute or relative path to the FITS frame
 * is_reference : 1 if this frame is the alignment reference, 0 otherwise
 * H            : forward homography (source → reference)
 */
typedef struct {
    char       filepath[4096];
    int        is_reference;
    Homography H;
} FrameInfo;

/*
 * DsoError — library-wide error codes.
 *
 * DSO_OK              (0)  — success
 * DSO_ERR_IO          (-1) — file open / read / write failure
 * DSO_ERR_ALLOC       (-2) — malloc/calloc returned NULL
 * DSO_ERR_FITS        (-3) — CFITSIO error (status printed to stderr)
 * DSO_ERR_CUDA        (-4) — CUDA runtime error
 * DSO_ERR_NPP         (-5) — NPP error (NppStatus != NPP_SUCCESS)
 * DSO_ERR_CSV         (-6) — CSV parse failure (empty / malformed)
 * DSO_ERR_INVALID_ARG (-7) — NULL pointer, singular matrix, size mismatch, …
 */
typedef enum {
    DSO_OK              =  0,
    DSO_ERR_IO          = -1,
    DSO_ERR_ALLOC       = -2,
    DSO_ERR_FITS        = -3,
    DSO_ERR_CUDA        = -4,
    DSO_ERR_NPP         = -5,
    DSO_ERR_CSV         = -6,
    DSO_ERR_INVALID_ARG = -7
} DsoError;

#ifdef __cplusplus
}
#endif
