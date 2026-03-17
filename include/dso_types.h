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
 *   H is the BACKWARD map: reference pixel coordinates → source pixel
 *   coordinates (ref → src).  Transform functions use H directly for
 *   pixel sampling — they do NOT invert it.
 *
 *   To transform reference pixel (dx, dy) to source sample (sx, sy):
 *     [sx_h, sy_h, w]^T = H * [dx, dy, 1]^T
 *     sx = sx_h / w,  sy = sy_h / w
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
 * H            : backward homography (ref → src); zero-init when not provided
 */
typedef struct {
    char       filepath[4096];
    int        is_reference;
    Homography H;
} FrameInfo;

/*
 * StarPos — sub-pixel position of a detected star with integrated flux.
 *
 * x, y  : center-of-mass coordinates (pixels, 0-based, column-major x)
 * flux  : sum of convolved pixel weights in the blob (used for brightness ranking)
 */
typedef struct {
    float x;
    float y;
    float flux;
} StarPos;

/*
 * StarList — heap-allocated array of detected star positions.
 *
 * stars : pointer to StarPos array; caller must free() when done
 * n     : number of valid entries in stars[]
 */
typedef struct {
    StarPos *stars;
    int      n;
} StarList;

/*
 * BayerPattern — mosaic color filter array layout.
 *
 * BAYER_NONE  : monochrome sensor; no debayering required
 * BAYER_RGGB  : top-left pixel is Red (most common DSLR pattern)
 * BAYER_BGGR  : top-left pixel is Blue (common in some cooled cameras)
 * BAYER_GRBG  : top-left pixel is Green-Red row
 * BAYER_GBRG  : top-left pixel is Green-Blue row
 *
 * Detected automatically from the FITS keyword BAYERPAT; overridable via CLI.
 */
typedef enum {
    BAYER_NONE = 0,
    BAYER_RGGB,
    BAYER_BGGR,
    BAYER_GRBG,
    BAYER_GBRG
} BayerPattern;

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
 * DSO_ERR_STAR_DETECT (-8) — insufficient stars found for alignment
 * DSO_ERR_RANSAC      (-9) — RANSAC failed to find a valid homography
 */
typedef enum {
    DSO_OK              =  0,
    DSO_ERR_IO          = -1,
    DSO_ERR_ALLOC       = -2,
    DSO_ERR_FITS        = -3,
    DSO_ERR_CUDA        = -4,
    DSO_ERR_NPP         = -5,
    DSO_ERR_CSV         = -6,
    DSO_ERR_INVALID_ARG = -7,
    DSO_ERR_STAR_DETECT = -8,
    DSO_ERR_RANSAC      = -9
} DsoError;

#ifdef __cplusplus
}
#endif
