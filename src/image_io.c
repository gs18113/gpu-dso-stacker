/*
 * image_io.c — Format-agnostic image save dispatch layer.
 *
 * Detects the output format from the file extension and routes to the
 * appropriate writer:
 *
 *   .fits / .fit / .fts  →  fits_save / fits_save_rgb  (CFITSIO, always FP32)
 *   .tif / .tiff         →  libtiff writer
 *                              bit depths : FP32, FP16, INT16, INT8
 *                              compression: none, zip (DEFLATE), lzw, rle (PackBits)
 *   .png                 →  libpng writer
 *                              bit depths : INT8, INT16 (always DEFLATE)
 *
 * Integer quantisation (INT8 / INT16 paths):
 *   quantised = round(clamp((val - lo) / (hi - lo) * MAX_INT, 0, MAX_INT))
 * where [lo, hi] defaults to the per-image [min, max] (RGB uses global
 * min/max across all three channels to preserve colour ratios), MAX_INT is
 * 255 or 65535.
 *
 * FP16 conversion uses portable IEEE 754 bit manipulation; no __fp16
 * compiler extension is required.
 *
 * TIFF RGB output uses interleaved PLANARCONFIG_CONTIG layout (RGBRGB…),
 * which is the format expected by Photoshop, Lightroom, and SIRIL.
 *
 * PNG 16-bit output is written in big-endian byte order as required by the
 * PNG specification; the byte swap is done explicitly in this file.
 */

#include "image_io.h"
#include "fits_io.h"

#include <tiffio.h>
#include <png.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Extension detection
 * ------------------------------------------------------------------------- */

OutputFormat image_detect_format(const char *path)
{
    const char *dot = strrchr(path, '.');
    if (!dot) return FMT_UNKNOWN;

    /* Copy extension and lower-case it */
    char ext[8] = {0};
    size_t len = strlen(dot + 1);
    if (len == 0 || len >= sizeof(ext)) return FMT_UNKNOWN;
    for (size_t i = 0; i <= len; i++)
        ext[i] = (char)tolower((unsigned char)dot[1 + i]);

    if (strcmp(ext, "fits") == 0 || strcmp(ext, "fit") == 0 || strcmp(ext, "fts") == 0)
        return FMT_FITS;
    if (strcmp(ext, "tif") == 0 || strcmp(ext, "tiff") == 0)
        return FMT_TIFF;
    if (strcmp(ext, "png") == 0)
        return FMT_PNG;
    return FMT_UNKNOWN;
}

/* -------------------------------------------------------------------------
 * Default options helper
 * Returns ImageSaveOptions with NAN stretch bounds (auto), FP32, no compression.
 * Used when the caller passes opts == NULL.
 * ------------------------------------------------------------------------- */
static ImageSaveOptions make_default_opts(void)
{
    ImageSaveOptions o;
    o.tiff_compress = TIFF_COMPRESS_NONE;
    o.bit_depth     = OUT_BITS_FP32;
    o.stretch_min   = (float)NAN;
    o.stretch_max   = (float)NAN;
    return o;
}

/* -------------------------------------------------------------------------
 * FP16 conversion — IEEE 754 bit manipulation, no compiler extension.
 *
 * Values outside the half-precision representable range are flushed to
 * zero (underflow) or infinity (overflow). NaN is preserved.
 * ------------------------------------------------------------------------- */
static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);

    uint32_t sign = (x >> 31) & 1u;
    uint32_t exp  = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 0xFFu) {
        /* Infinity or NaN — preserve */
        return (uint16_t)((sign << 15) | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp == 0u) {
        /* Zero / denormal — flush to signed zero */
        return (uint16_t)(sign << 15);
    }

    int e16 = (int)exp - 127 + 15;
    if (e16 >= 31) {
        /* Overflow — map to infinity */
        return (uint16_t)((sign << 15) | 0x7C00u);
    }
    if (e16 <= 0) {
        /* Underflow — flush to signed zero */
        return (uint16_t)(sign << 15);
    }
    return (uint16_t)((sign << 15) | ((uint16_t)e16 << 10) | (uint16_t)(mant >> 13));
}

/* -------------------------------------------------------------------------
 * Integer stretch helpers
 * ------------------------------------------------------------------------- */

static void stretch_bounds_mono(const float *data, int n,
                                 float user_lo, float user_hi,
                                 float *lo_out, float *hi_out)
{
    float lo = user_lo, hi = user_hi;
    if (isnan(lo) || isnan(hi)) {
        float mn = data[0], mx = data[0];
        for (int i = 1; i < n; i++) {
            if (data[i] < mn) mn = data[i];
            if (data[i] > mx) mx = data[i];
        }
        if (isnan(lo)) lo = mn;
        if (isnan(hi)) hi = mx;
    }
    *lo_out = lo;
    *hi_out = hi;
}

/* For RGB, derive a single lo/hi from all three planes to preserve colour ratios */
static void stretch_bounds_rgb(const float *r, const float *g, const float *b, int n,
                                float user_lo, float user_hi,
                                float *lo_out, float *hi_out)
{
    float lo = user_lo, hi = user_hi;
    if (isnan(lo) || isnan(hi)) {
        float mn = r[0], mx = r[0];
        for (int i = 0; i < n; i++) {
            if (r[i] < mn) mn = r[i]; if (r[i] > mx) mx = r[i];
            if (g[i] < mn) mn = g[i]; if (g[i] > mx) mx = g[i];
            if (b[i] < mn) mn = b[i]; if (b[i] > mx) mx = b[i];
        }
        if (isnan(lo)) lo = mn;
        if (isnan(hi)) hi = mx;
    }
    *lo_out = lo;
    *hi_out = hi;
}

static inline uint16_t quantise16(float val, float lo, float hi)
{
    if (hi <= lo) return 0;
    float t = (val - lo) / (hi - lo) * 65535.0f;
    if (t < 0.0f)       t = 0.0f;
    if (t > 65535.0f)   t = 65535.0f;
    return (uint16_t)(t + 0.5f);
}

static inline uint8_t quantise8(float val, float lo, float hi)
{
    if (hi <= lo) return 0;
    float t = (val - lo) / (hi - lo) * 255.0f;
    if (t < 0.0f)  t = 0.0f;
    if (t > 255.0f) t = 255.0f;
    return (uint8_t)(t + 0.5f);
}

/* -------------------------------------------------------------------------
 * TIFF helpers
 * ------------------------------------------------------------------------- */

static uint16_t tiff_compress_tag(TiffCompression c)
{
    switch (c) {
    case TIFF_COMPRESS_ZIP: return COMPRESSION_ADOBE_DEFLATE;
    case TIFF_COMPRESS_LZW: return COMPRESSION_LZW;
    case TIFF_COMPRESS_RLE: return COMPRESSION_PACKBITS;
    default:                return COMPRESSION_NONE;
    }
}

static DsoError tiff_save_mono(const char *path, const Image *img,
                                const ImageSaveOptions *opts)
{
    TIFF *tif = TIFFOpen(path, "w");
    if (!tif) return DSO_ERR_IO;

    int W = img->width, H = img->height;

    uint16_t bps, sfmt;
    switch (opts->bit_depth) {
    case OUT_BITS_FP16:  bps = 16; sfmt = SAMPLEFORMAT_IEEEFP; break;
    case OUT_BITS_INT16: bps = 16; sfmt = SAMPLEFORMAT_UINT;   break;
    case OUT_BITS_INT8:  bps =  8; sfmt = SAMPLEFORMAT_UINT;   break;
    default:             bps = 32; sfmt = SAMPLEFORMAT_IEEEFP; break; /* FP32 */
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,       (uint32_t)W);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,      (uint32_t)H);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,    bps);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT,     sfmt);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL,  (uint16_t)1);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,      PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,     PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,      tiff_compress_tag(opts->tiff_compress));
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,     TIFFDefaultStripSize(tif, (uint32_t)-1));

    float lo = 0.0f, hi = 1.0f;
    if (opts->bit_depth == OUT_BITS_INT8 || opts->bit_depth == OUT_BITS_INT16)
        stretch_bounds_mono(img->data, W * H, opts->stretch_min, opts->stretch_max, &lo, &hi);

    size_t row_bytes = (size_t)W * (bps / 8);
    uint8_t *row = (uint8_t *)malloc(row_bytes);
    if (!row) { TIFFClose(tif); return DSO_ERR_ALLOC; }

    DsoError err = DSO_OK;
    for (int y = 0; y < H && err == DSO_OK; y++) {
        const float *src = img->data + (size_t)y * W;
        switch (opts->bit_depth) {
        case OUT_BITS_FP32:
            memcpy(row, src, row_bytes);
            break;
        case OUT_BITS_FP16: {
            uint16_t *p = (uint16_t *)row;
            for (int x = 0; x < W; x++) p[x] = f32_to_f16(src[x]);
            break;
        }
        case OUT_BITS_INT16: {
            uint16_t *p = (uint16_t *)row;
            for (int x = 0; x < W; x++) p[x] = quantise16(src[x], lo, hi);
            break;
        }
        case OUT_BITS_INT8:
            for (int x = 0; x < W; x++) row[x] = quantise8(src[x], lo, hi);
            break;
        }
        if (TIFFWriteScanline(tif, row, (uint32_t)y, 0) < 0)
            err = DSO_ERR_IO;
    }

    free(row);
    TIFFClose(tif);
    return err;
}

static DsoError tiff_save_rgb(const char *path,
                               const Image *r, const Image *g, const Image *b,
                               const ImageSaveOptions *opts)
{
    TIFF *tif = TIFFOpen(path, "w");
    if (!tif) return DSO_ERR_IO;

    int W = r->width, H = r->height;

    uint16_t bps, sfmt;
    switch (opts->bit_depth) {
    case OUT_BITS_FP16:  bps = 16; sfmt = SAMPLEFORMAT_IEEEFP; break;
    case OUT_BITS_INT16: bps = 16; sfmt = SAMPLEFORMAT_UINT;   break;
    case OUT_BITS_INT8:  bps =  8; sfmt = SAMPLEFORMAT_UINT;   break;
    default:             bps = 32; sfmt = SAMPLEFORMAT_IEEEFP; break;
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,       (uint32_t)W);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,      (uint32_t)H);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,    bps);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT,     sfmt);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL,  (uint16_t)3);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,      PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,     PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,      tiff_compress_tag(opts->tiff_compress));
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,     TIFFDefaultStripSize(tif, (uint32_t)-1));

    int npix = W * H;
    float lo = 0.0f, hi = 1.0f;
    if (opts->bit_depth == OUT_BITS_INT8 || opts->bit_depth == OUT_BITS_INT16)
        stretch_bounds_rgb(r->data, g->data, b->data, npix,
                           opts->stretch_min, opts->stretch_max, &lo, &hi);

    size_t row_bytes = (size_t)W * 3 * (bps / 8);
    uint8_t *row = (uint8_t *)malloc(row_bytes);
    if (!row) { TIFFClose(tif); return DSO_ERR_ALLOC; }

    DsoError err = DSO_OK;
    for (int y = 0; y < H && err == DSO_OK; y++) {
        const float *sr = r->data + (size_t)y * W;
        const float *sg = g->data + (size_t)y * W;
        const float *sb = b->data + (size_t)y * W;
        switch (opts->bit_depth) {
        case OUT_BITS_FP32: {
            float *p = (float *)row;
            for (int x = 0; x < W; x++) {
                p[x * 3 + 0] = sr[x];
                p[x * 3 + 1] = sg[x];
                p[x * 3 + 2] = sb[x];
            }
            break;
        }
        case OUT_BITS_FP16: {
            uint16_t *p = (uint16_t *)row;
            for (int x = 0; x < W; x++) {
                p[x * 3 + 0] = f32_to_f16(sr[x]);
                p[x * 3 + 1] = f32_to_f16(sg[x]);
                p[x * 3 + 2] = f32_to_f16(sb[x]);
            }
            break;
        }
        case OUT_BITS_INT16: {
            uint16_t *p = (uint16_t *)row;
            for (int x = 0; x < W; x++) {
                p[x * 3 + 0] = quantise16(sr[x], lo, hi);
                p[x * 3 + 1] = quantise16(sg[x], lo, hi);
                p[x * 3 + 2] = quantise16(sb[x], lo, hi);
            }
            break;
        }
        case OUT_BITS_INT8:
            for (int x = 0; x < W; x++) {
                row[x * 3 + 0] = quantise8(sr[x], lo, hi);
                row[x * 3 + 1] = quantise8(sg[x], lo, hi);
                row[x * 3 + 2] = quantise8(sb[x], lo, hi);
            }
            break;
        }
        if (TIFFWriteScanline(tif, row, (uint32_t)y, 0) < 0)
            err = DSO_ERR_IO;
    }

    free(row);
    TIFFClose(tif);
    return err;
}

/* -------------------------------------------------------------------------
 * PNG helpers
 * ------------------------------------------------------------------------- */

static DsoError png_save_mono(const char *path, const Image *img,
                               const ImageSaveOptions *opts)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return DSO_ERR_IO;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return DSO_ERR_ALLOC; }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return DSO_ERR_ALLOC;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return DSO_ERR_IO;
    }

    png_init_io(png, fp);

    int W = img->width, H = img->height;
    int bit_depth = (opts->bit_depth == OUT_BITS_INT8) ? 8 : 16;
    png_set_IHDR(png, info, (png_uint_32)W, (png_uint_32)H,
                 bit_depth, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    float lo = 0.0f, hi = 1.0f;
    stretch_bounds_mono(img->data, W * H, opts->stretch_min, opts->stretch_max, &lo, &hi);

    size_t row_bytes = (bit_depth == 8) ? (size_t)W : (size_t)W * 2;
    uint8_t *row = (uint8_t *)malloc(row_bytes);
    if (!row) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return DSO_ERR_ALLOC;
    }

    for (int y = 0; y < H; y++) {
        const float *src = img->data + (size_t)y * W;
        if (bit_depth == 8) {
            for (int x = 0; x < W; x++)
                row[x] = quantise8(src[x], lo, hi);
        } else {
            /* 16-bit: big-endian as required by PNG spec */
            for (int x = 0; x < W; x++) {
                uint16_t v = quantise16(src[x], lo, hi);
                row[x * 2 + 0] = (uint8_t)(v >> 8);
                row[x * 2 + 1] = (uint8_t)(v & 0xFF);
            }
        }
        png_write_row(png, row);
    }

    free(row);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return DSO_OK;
}

static DsoError png_save_rgb(const char *path,
                              const Image *r, const Image *g, const Image *b,
                              const ImageSaveOptions *opts)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return DSO_ERR_IO;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return DSO_ERR_ALLOC; }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return DSO_ERR_ALLOC;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return DSO_ERR_IO;
    }

    png_init_io(png, fp);

    int W = r->width, H = r->height;
    int bit_depth = (opts->bit_depth == OUT_BITS_INT8) ? 8 : 16;
    png_set_IHDR(png, info, (png_uint_32)W, (png_uint_32)H,
                 bit_depth, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    int npix = W * H;
    float lo = 0.0f, hi = 1.0f;
    stretch_bounds_rgb(r->data, g->data, b->data, npix,
                       opts->stretch_min, opts->stretch_max, &lo, &hi);

    size_t row_bytes = (bit_depth == 8) ? (size_t)W * 3 : (size_t)W * 6;
    uint8_t *row = (uint8_t *)malloc(row_bytes);
    if (!row) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return DSO_ERR_ALLOC;
    }

    for (int y = 0; y < H; y++) {
        const float *sr = r->data + (size_t)y * W;
        const float *sg = g->data + (size_t)y * W;
        const float *sb = b->data + (size_t)y * W;
        if (bit_depth == 8) {
            for (int x = 0; x < W; x++) {
                row[x * 3 + 0] = quantise8(sr[x], lo, hi);
                row[x * 3 + 1] = quantise8(sg[x], lo, hi);
                row[x * 3 + 2] = quantise8(sb[x], lo, hi);
            }
        } else {
            /* 16-bit big-endian: R0hi R0lo G0hi G0lo B0hi B0lo … */
            for (int x = 0; x < W; x++) {
                uint16_t rv = quantise16(sr[x], lo, hi);
                uint16_t gv = quantise16(sg[x], lo, hi);
                uint16_t bv = quantise16(sb[x], lo, hi);
                row[x * 6 + 0] = (uint8_t)(rv >> 8);
                row[x * 6 + 1] = (uint8_t)(rv & 0xFF);
                row[x * 6 + 2] = (uint8_t)(gv >> 8);
                row[x * 6 + 3] = (uint8_t)(gv & 0xFF);
                row[x * 6 + 4] = (uint8_t)(bv >> 8);
                row[x * 6 + 5] = (uint8_t)(bv & 0xFF);
            }
        }
        png_write_row(png, row);
    }

    free(row);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

DsoError image_save(const char *filepath, const Image *img,
                    const ImageSaveOptions *opts)
{
    ImageSaveOptions eff = opts ? *opts : make_default_opts();
    OutputFormat fmt = image_detect_format(filepath);

    switch (fmt) {
    case FMT_FITS:
        return fits_save(filepath, img);

    case FMT_TIFF:
        return tiff_save_mono(filepath, img, &eff);

    case FMT_PNG:
        if (eff.bit_depth == OUT_BITS_FP32 || eff.bit_depth == OUT_BITS_FP16) {
            fprintf(stderr,
                    "image_save: PNG does not support FP32/FP16 output. "
                    "Use --bit-depth 8 or --bit-depth 16.\n");
            return DSO_ERR_INVALID_ARG;
        }
        return png_save_mono(filepath, img, &eff);

    default:
        fprintf(stderr,
                "image_save: unrecognized file extension for '%s'. "
                "Use .fits, .fit, .fts, .tif, .tiff, or .png.\n",
                filepath);
        return DSO_ERR_INVALID_ARG;
    }
}

DsoError image_save_rgb(const char *filepath,
                         const Image *r, const Image *g, const Image *b,
                         const ImageSaveOptions *opts)
{
    ImageSaveOptions eff = opts ? *opts : make_default_opts();
    OutputFormat fmt = image_detect_format(filepath);

    switch (fmt) {
    case FMT_FITS:
        return fits_save_rgb(filepath, r, g, b);

    case FMT_TIFF:
        return tiff_save_rgb(filepath, r, g, b, &eff);

    case FMT_PNG:
        if (eff.bit_depth == OUT_BITS_FP32 || eff.bit_depth == OUT_BITS_FP16) {
            fprintf(stderr,
                    "image_save_rgb: PNG does not support FP32/FP16 output. "
                    "Use --bit-depth 8 or --bit-depth 16.\n");
            return DSO_ERR_INVALID_ARG;
        }
        return png_save_rgb(filepath, r, g, b, &eff);

    default:
        fprintf(stderr,
                "image_save_rgb: unrecognized file extension for '%s'. "
                "Use .fits, .fit, .fts, .tif, .tiff, or .png.\n",
                filepath);
        return DSO_ERR_INVALID_ARG;
    }
}
