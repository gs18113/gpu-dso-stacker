/*
 * image_io.h — Format-agnostic image save layer.
 *
 * Detects the output format from the file extension (.fits/.fit/.fts,
 * .tif/.tiff, .png) and dispatches to the appropriate writer.  The only
 * supported input pixel type is float32, matching the internal Image
 * representation used throughout the pipeline.
 *
 * Supported output formats and options:
 *
 *   Extension         Format   Bit depths              Compression
 *   ───────────────── ──────── ──────────────────────  ──────────────────────
 *   .fits / .fit / .fts FITS   FP32 only               none
 *   .tif / .tiff      TIFF     INT8, INT16, FP16, FP32  none, zip, lzw, rle
 *   .png              PNG      INT8, INT16              always lossless DEFLATE
 *
 * Integer scaling formula (INT8 / INT16 output only):
 *   quantised = round(clamp((val - lo) / (hi - lo) * MAX_INT, 0, MAX_INT))
 * where lo/hi are the effective stretch bounds (auto = per-image min/max),
 * MAX_INT = 255 for INT8, 65535 for INT16.  For RGB output, the same lo/hi
 * are derived globally across all three channels to preserve colour ratios.
 *
 * FP16 conversion uses portable IEEE 754 bit manipulation (no __fp16
 * compiler extension).  Values outside the FP16 representable range
 * (approximately [6×10⁻⁸, 65504]) are flushed to zero or infinity.
 */

#pragma once

#include "dso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * OutputFormat — detected from the output filepath extension.
 * Exposed here so callers can perform pre-flight format validation.
 * ------------------------------------------------------------------------- */
typedef enum {
    FMT_FITS    = 0,   /* .fits / .fit / .fts */
    FMT_TIFF    = 1,   /* .tif / .tiff        */
    FMT_PNG     = 2,   /* .png                */
    FMT_UNKNOWN = 3    /* no recognized extension */
} OutputFormat;

/* -------------------------------------------------------------------------
 * TiffCompression — lossless compression codec for TIFF output.
 * Ignored for FITS and PNG output.
 * ------------------------------------------------------------------------- */
typedef enum {
    TIFF_COMPRESS_NONE = 0,   /* no compression (default)    */
    TIFF_COMPRESS_ZIP,        /* DEFLATE / ZIP (lossless)    */
    TIFF_COMPRESS_LZW,        /* Lempel-Ziv-Welch (lossless) */
    TIFF_COMPRESS_RLE         /* PackBits run-length (lossless) */
} TiffCompression;

/* -------------------------------------------------------------------------
 * OutputBitDepth — sample depth written to the output file.
 *
 *   OUT_BITS_FP32  : IEEE 754 single-precision float; exact, no quantisation.
 *                    Valid for FITS and TIFF.  Default.
 *   OUT_BITS_FP16  : IEEE 754 half-precision float (TIFF only).  Precision
 *                    loss for values outside ~[6e-8, 65504].
 *   OUT_BITS_INT16 : unsigned 16-bit integer [0, 65535]; linear scaling.
 *                    Valid for TIFF and PNG.
 *   OUT_BITS_INT8  : unsigned 8-bit integer [0, 255]; linear scaling.
 *                    Valid for TIFF and PNG.
 *
 * FITS output always uses FP32 regardless of this field.
 * PNG requires INT8 or INT16; FP32/FP16 are errors.
 * ------------------------------------------------------------------------- */
typedef enum {
    OUT_BITS_FP32  = 0,
    OUT_BITS_FP16,
    OUT_BITS_INT16,
    OUT_BITS_INT8
} OutputBitDepth;

/* -------------------------------------------------------------------------
 * ImageSaveOptions — controls bit depth, compression, and integer scaling.
 *
 * Zero-initialise for lossless FP32 output with no compression.
 * For integer formats (INT8/INT16), stretch_min/stretch_max must be set to
 * NAN (use auto min/max of the image) or to explicit float values.
 * ------------------------------------------------------------------------- */
typedef struct {
    TiffCompression tiff_compress;   /* compression for TIFF; ignored for FITS/PNG */
    OutputBitDepth  bit_depth;       /* sample depth; ignored for FITS (always FP32) */
    float           stretch_min;     /* lower bound for INT scaling; NAN = auto */
    float           stretch_max;     /* upper bound for INT scaling; NAN = auto */
} ImageSaveOptions;

/* -------------------------------------------------------------------------
 * image_detect_format — infer OutputFormat from a filepath extension.
 * The comparison is case-insensitive (.TIFF = .tiff).
 * Returns FMT_UNKNOWN if the extension is not recognized.
 * ------------------------------------------------------------------------- */
OutputFormat image_detect_format(const char *filepath);

/* -------------------------------------------------------------------------
 * image_save — save a single-channel (mono / luminance) float32 image.
 *
 * filepath : output path; format is inferred from the extension.
 * img      : float32 row-major image; must not be NULL.
 * opts     : save options; NULL is equivalent to a zero-initialised struct
 *            (FP32, no compression, auto stretch with NAN bounds).
 *
 * Returns DSO_OK on success, DSO_ERR_IO on file I/O failure,
 * DSO_ERR_ALLOC on memory allocation failure, DSO_ERR_INVALID_ARG if
 * opts are incompatible with the detected format, or FMT_UNKNOWN.
 * ------------------------------------------------------------------------- */
DsoError image_save(const char *filepath, const Image *img,
                    const ImageSaveOptions *opts);

/* -------------------------------------------------------------------------
 * image_save_rgb — save a three-channel (R/G/B) float32 image.
 *
 * The three planes must have identical dimensions.  For integer output
 * (INT8/INT16), stretch bounds are derived globally across all three planes
 * so that colour ratios are preserved.
 * ------------------------------------------------------------------------- */
DsoError image_save_rgb(const char *filepath,
                         const Image *r, const Image *g, const Image *b,
                         const ImageSaveOptions *opts);

#ifdef __cplusplus
}
#endif
