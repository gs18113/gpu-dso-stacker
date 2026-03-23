/*
 * test_image_io.c — Round-trip tests for the image_io save layer.
 *
 * Tests all combinations of:
 *   - Format: FITS passthrough, TIFF (all bit depths + compressions), PNG
 *   - Channels: mono and RGB
 *
 * Each test writes a synthetic image to /tmp/, reads it back using the
 * appropriate library (CFITSIO, libtiff, libpng), and verifies pixel values.
 * All temporary files are removed at test exit.
 *
 * FP32 TIFF is verified to be bit-exact.
 * FP16 TIFF is verified to be within FP16 epsilon (~2e-3 relative).
 * INT16 and INT8 are verified by checking the round-trip quantised value
 * matches the expected quantisation formula.
 * PNG 16-bit is verified for correct big-endian byte order.
 */

#include "test_framework.h"
#include "image_io.h"
#include "fits_io.h"

#include <tiffio.h>
#include <png.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Linear gradient: data[i] = (i / (n-1)) * scale */
static Image make_gradient(int W, int H, float scale)
{
    Image img = {NULL, W, H};
    int n = W * H;
    img.data = (float *)malloc((size_t)n * sizeof(float));
    if (!img.data) return img;
    for (int i = 0; i < n; i++)
        img.data[i] = (float)i / (float)(n > 1 ? n - 1 : 1) * scale;
    return img;
}

static void rm(const char *path) { remove(path); }

/* Portable IEEE 754 FP16 → FP32 for test verification */
static float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp == 31u)      x = sign | 0x7F800000u | (mant << 13);
    else if (exp == 0u)  x = sign;
    else                 x = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

/* -------------------------------------------------------------------------
 * TIFF reader helpers
 * ------------------------------------------------------------------------- */

static float *tiff_read_mono_fp32(const char *path, int *W_out, int *H_out)
{
    TIFF *tif = TIFFOpen(path, "r");
    if (!tif) return NULL;
    uint32_t W, H;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,  &W);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &H);
    *W_out = (int)W; *H_out = (int)H;
    size_t rowbytes = (size_t)TIFFScanlineSize(tif);
    uint8_t *rowbuf = (uint8_t *)_TIFFmalloc((tmsize_t)rowbytes);
    float   *out    = (float *)malloc((size_t)W * H * sizeof(float));
    if (!rowbuf || !out) { if (rowbuf) _TIFFfree(rowbuf); free(out); TIFFClose(tif); return NULL; }
    for (uint32_t y = 0; y < H; y++) {
        TIFFReadScanline(tif, rowbuf, y, 0);
        memcpy(out + (size_t)y * W, rowbuf, (size_t)W * sizeof(float));
    }
    _TIFFfree(rowbuf); TIFFClose(tif);
    return out;
}

static uint16_t *tiff_read_mono_u16(const char *path, int *W_out, int *H_out)
{
    TIFF *tif = TIFFOpen(path, "r");
    if (!tif) return NULL;
    uint32_t W, H;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,  &W);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &H);
    *W_out = (int)W; *H_out = (int)H;
    size_t rowbytes = (size_t)TIFFScanlineSize(tif);
    uint8_t  *rowbuf = (uint8_t *)_TIFFmalloc((tmsize_t)rowbytes);
    uint16_t *out    = (uint16_t *)malloc((size_t)W * H * sizeof(uint16_t));
    if (!rowbuf || !out) { if (rowbuf) _TIFFfree(rowbuf); free(out); TIFFClose(tif); return NULL; }
    for (uint32_t y = 0; y < H; y++) {
        TIFFReadScanline(tif, rowbuf, y, 0);
        memcpy(out + (size_t)y * W, rowbuf, (size_t)W * sizeof(uint16_t));
    }
    _TIFFfree(rowbuf); TIFFClose(tif);
    return out;
}

static uint8_t *tiff_read_mono_u8(const char *path, int *W_out, int *H_out)
{
    TIFF *tif = TIFFOpen(path, "r");
    if (!tif) return NULL;
    uint32_t W, H;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,  &W);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &H);
    *W_out = (int)W; *H_out = (int)H;
    size_t rowbytes = (size_t)TIFFScanlineSize(tif);
    uint8_t *rowbuf = (uint8_t *)_TIFFmalloc((tmsize_t)rowbytes);
    uint8_t *out    = (uint8_t *)malloc((size_t)W * H);
    if (!rowbuf || !out) { if (rowbuf) _TIFFfree(rowbuf); free(out); TIFFClose(tif); return NULL; }
    for (uint32_t y = 0; y < H; y++) {
        TIFFReadScanline(tif, rowbuf, y, 0);
        memcpy(out + (size_t)y * W, rowbuf, W);
    }
    _TIFFfree(rowbuf); TIFFClose(tif);
    return out;
}

/* -------------------------------------------------------------------------
 * PNG reader helpers
 * ------------------------------------------------------------------------- */

static uint8_t *png_read_gray8(const char *path, int *W_out, int *H_out)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return NULL; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return NULL; }
    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return NULL; }
    png_init_io(png, fp);
    png_read_info(png, info);
    int W = (int)png_get_image_width(png, info);
    int H = (int)png_get_image_height(png, info);
    *W_out = W; *H_out = H;
    uint8_t *out = (uint8_t *)malloc((size_t)W * H);
    if (!out) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return NULL; }
    for (int y = 0; y < H; y++) png_read_row(png, out + (size_t)y * W, NULL);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return out;
}

/* Read 16-bit gray PNG; returns host-order uint16 array */
static uint16_t *png_read_gray16(const char *path, int *W_out, int *H_out)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return NULL; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return NULL; }
    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return NULL; }
    png_init_io(png, fp);
    png_read_info(png, info);
    int W = (int)png_get_image_width(png, info);
    int H = (int)png_get_image_height(png, info);
    *W_out = W; *H_out = H;
    uint8_t  *raw = (uint8_t *)malloc((size_t)W * H * 2);
    uint16_t *out = (uint16_t *)malloc((size_t)W * H * sizeof(uint16_t));
    if (!raw || !out) { free(raw); free(out); png_destroy_read_struct(&png, &info, NULL); fclose(fp); return NULL; }
    for (int y = 0; y < H; y++) png_read_row(png, raw + (size_t)y * W * 2, NULL);
    /* Big-endian → host */
    for (int i = 0; i < W * H; i++)
        out[i] = (uint16_t)(((uint16_t)raw[i*2] << 8) | raw[i*2+1]);
    free(raw);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return out;
}

/* -------------------------------------------------------------------------
 * Format detection tests
 * ------------------------------------------------------------------------- */

static int test_detect_fits(void)
{
    ASSERT_EQ((int)image_detect_format("out.fits"), (int)FMT_FITS);
    ASSERT_EQ((int)image_detect_format("out.fit"),  (int)FMT_FITS);
    ASSERT_EQ((int)image_detect_format("out.fts"),  (int)FMT_FITS);
    ASSERT_EQ((int)image_detect_format("out.FITS"), (int)FMT_FITS);
    return 0;
}

static int test_detect_tiff(void)
{
    ASSERT_EQ((int)image_detect_format("out.tif"),  (int)FMT_TIFF);
    ASSERT_EQ((int)image_detect_format("out.tiff"), (int)FMT_TIFF);
    ASSERT_EQ((int)image_detect_format("out.TIFF"), (int)FMT_TIFF);
    ASSERT_EQ((int)image_detect_format("out.TIF"),  (int)FMT_TIFF);
    return 0;
}

static int test_detect_png(void)
{
    ASSERT_EQ((int)image_detect_format("out.png"), (int)FMT_PNG);
    ASSERT_EQ((int)image_detect_format("out.PNG"), (int)FMT_PNG);
    return 0;
}

static int test_detect_unknown(void)
{
    ASSERT_EQ((int)image_detect_format("out.jpg"),  (int)FMT_UNKNOWN);
    ASSERT_EQ((int)image_detect_format("noext"),    (int)FMT_UNKNOWN);
    return 0;
}

/* -------------------------------------------------------------------------
 * FITS passthrough (regression)
 * ------------------------------------------------------------------------- */

static int test_fits_passthrough(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_fits.fits");
    int W = 8, H = 8;
    Image img = make_gradient(W, H, 1000.0f);
    ASSERT(img.data != NULL);

    ASSERT_OK(image_save(path, &img, NULL));

    Image loaded = {NULL, 0, 0};
    ASSERT_OK(fits_load(path, &loaded));
    ASSERT_EQ(loaded.width, W);
    ASSERT_EQ(loaded.height, H);
    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(loaded.data[i], img.data[i], 1e-4f);

    image_free(&loaded);
    free(img.data);
    rm(path);
    return 0;
}

/* -------------------------------------------------------------------------
 * TIFF mono tests
 * ------------------------------------------------------------------------- */

static int test_tiff_fp32_mono(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_fp32.tiff");
    int W = 16, H = 12;
    Image img = make_gradient(W, H, 500.0f);
    ASSERT(img.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_FP32, (float)NAN, (float)NAN};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    float *got = tiff_read_mono_fp32(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);
    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(got[i], img.data[i], 0.0f);  /* bit-exact */

    free(got); free(img.data); rm(path);
    return 0;
}

static int test_tiff_fp16_mono(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_fp16.tiff");
    int W = 8, H = 8;
    Image img = make_gradient(W, H, 1.0f);  /* [0,1]: well within FP16 range */
    ASSERT(img.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_FP16, (float)NAN, (float)NAN};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    uint16_t *raw = tiff_read_mono_u16(path, &rW, &rH);
    ASSERT(raw != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);

    for (int i = 0; i < W * H; i++) {
        float reconstructed = f16_to_f32(raw[i]);
        float expected = img.data[i];
        float eps = (expected < 1e-6f) ? 1e-4f : fabsf(expected) * 2e-3f;
        ASSERT_NEAR(reconstructed, expected, eps);
    }

    free(raw); free(img.data); rm(path);
    return 0;
}

static int test_tiff_int16_mono(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_int16.tiff");
    int W = 8, H = 8;
    Image img = make_gradient(W, H, 1.0f);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT16, 0.0f, 1.0f};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    uint16_t *got = tiff_read_mono_u16(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);

    for (int i = 0; i < W * H; i++) {
        uint16_t expected = (uint16_t)(img.data[i] * 65535.0f + 0.5f);
        ASSERT_EQ((int)got[i], (int)expected);
    }

    free(got); free(img.data); rm(path);
    return 0;
}

static int test_tiff_int8_mono(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_int8.tiff");
    int W = 8, H = 8;
    Image img = make_gradient(W, H, 1.0f);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT8, 0.0f, 1.0f};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    uint8_t *got = tiff_read_mono_u8(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);

    for (int i = 0; i < W * H; i++) {
        uint8_t expected = (uint8_t)(img.data[i] * 255.0f + 0.5f);
        ASSERT_EQ((int)got[i], (int)expected);
    }

    free(got); free(img.data); rm(path);
    return 0;
}

/* --- TIFF compression round-trips (FP32, bit-exact) --- */

static int test_tiff_zip_roundtrip(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_zip.tiff");
    int W = 32, H = 32;
    Image img = make_gradient(W, H, 100.0f);
    ASSERT(img.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_ZIP, OUT_BITS_FP32, (float)NAN, (float)NAN};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    float *got = tiff_read_mono_fp32(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);
    for (int i = 0; i < W * H; i++) ASSERT_NEAR(got[i], img.data[i], 0.0f);

    free(got); free(img.data); rm(path);
    return 0;
}

static int test_tiff_lzw_roundtrip(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_lzw.tiff");
    int W = 32, H = 32;
    Image img = make_gradient(W, H, 100.0f);
    ASSERT(img.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_LZW, OUT_BITS_FP32, (float)NAN, (float)NAN};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    float *got = tiff_read_mono_fp32(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);
    for (int i = 0; i < W * H; i++) ASSERT_NEAR(got[i], img.data[i], 0.0f);

    free(got); free(img.data); rm(path);
    return 0;
}

static int test_tiff_rle_roundtrip(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_rle.tiff");
    int W = 32, H = 32;
    Image img = make_gradient(W, H, 100.0f);
    ASSERT(img.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_RLE, OUT_BITS_FP32, (float)NAN, (float)NAN};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    float *got = tiff_read_mono_fp32(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);
    for (int i = 0; i < W * H; i++) ASSERT_NEAR(got[i], img.data[i], 0.0f);

    free(got); free(img.data); rm(path);
    return 0;
}

/* --- TIFF RGB --- */

static int test_tiff_fp32_rgb(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_fp32_rgb.tiff");
    int W = 8, H = 6;
    Image r = make_gradient(W, H, 1.0f);
    Image g = make_gradient(W, H, 0.5f);
    Image b = make_gradient(W, H, 0.25f);
    ASSERT(r.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_FP32, (float)NAN, (float)NAN};
    ASSERT_OK(image_save_rgb(path, &r, &g, &b, &opts));

    TIFF *tif = TIFFOpen(path, "r");
    ASSERT(tif != NULL);

    uint32_t rW, rH;
    uint16_t spp;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,      &rW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH,     &rH);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    ASSERT_EQ((int)rW, W); ASSERT_EQ((int)rH, H); ASSERT_EQ((int)spp, 3);

    size_t rowbytes = (size_t)TIFFScanlineSize(tif);
    float *rowbuf = (float *)_TIFFmalloc((tmsize_t)rowbytes);
    ASSERT(rowbuf != NULL);

    int ok = 1;
    for (uint32_t y = 0; y < rH && ok; y++) {
        TIFFReadScanline(tif, rowbuf, y, 0);
        for (int x = 0; x < W && ok; x++) {
            int i = (int)y * W + x;
            if (rowbuf[x*3+0] != r.data[i]) ok = 0;
            if (rowbuf[x*3+1] != g.data[i]) ok = 0;
            if (rowbuf[x*3+2] != b.data[i]) ok = 0;
        }
    }
    _TIFFfree(rowbuf);
    TIFFClose(tif);
    ASSERT_EQ(ok, 1);

    free(r.data); free(g.data); free(b.data); rm(path);
    return 0;
}

static int test_tiff_int16_rgb_global_scale(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_int16_rgb.tiff");
    int W = 4, H = 4;
    Image r = make_gradient(W, H, 1.0f);
    Image g = make_gradient(W, H, 0.5f);
    Image b = make_gradient(W, H, 0.25f);
    ASSERT(r.data != NULL);

    /* Explicit [0,1] stretch: all channels use the same bounds */
    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT16, 0.0f, 1.0f};
    ASSERT_OK(image_save_rgb(path, &r, &g, &b, &opts));

    TIFF *tif = TIFFOpen(path, "r");
    ASSERT(tif != NULL);

    uint32_t rW, rH;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,  &rW);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &rH);
    ASSERT_EQ((int)rW, W); ASSERT_EQ((int)rH, H);

    size_t rowbytes = (size_t)TIFFScanlineSize(tif);
    uint16_t *rowbuf = (uint16_t *)_TIFFmalloc((tmsize_t)rowbytes);
    ASSERT(rowbuf != NULL);

    int ok = 1;
    for (uint32_t y = 0; y < rH && ok; y++) {
        TIFFReadScanline(tif, rowbuf, y, 0);
        for (int x = 0; x < W && ok; x++) {
            int i = (int)y * W + x;
            uint16_t er = (uint16_t)(r.data[i] * 65535.0f + 0.5f);
            uint16_t eg = (uint16_t)(g.data[i] * 65535.0f + 0.5f);
            uint16_t eb = (uint16_t)(b.data[i] * 65535.0f + 0.5f);
            if (rowbuf[x*3+0] != er) ok = 0;
            if (rowbuf[x*3+1] != eg) ok = 0;
            if (rowbuf[x*3+2] != eb) ok = 0;
        }
    }
    _TIFFfree(rowbuf);
    TIFFClose(tif);
    ASSERT_EQ(ok, 1);

    free(r.data); free(g.data); free(b.data); rm(path);
    return 0;
}

/* -------------------------------------------------------------------------
 * PNG mono tests
 * ------------------------------------------------------------------------- */

static int test_png_8bit_mono(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_png8.png");
    int W = 16, H = 8;
    Image img = make_gradient(W, H, 1.0f);
    ASSERT(img.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT8, 0.0f, 1.0f};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    uint8_t *got = png_read_gray8(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);

    for (int i = 0; i < W * H; i++) {
        uint8_t expected = (uint8_t)(img.data[i] * 255.0f + 0.5f);
        ASSERT_EQ((int)got[i], (int)expected);
    }

    free(got); free(img.data); rm(path);
    return 0;
}

static int test_png_16bit_mono(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_png16.png");
    int W = 16, H = 8;
    Image img = make_gradient(W, H, 1.0f);
    ASSERT(img.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT16, 0.0f, 1.0f};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    uint16_t *got = png_read_gray16(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);

    for (int i = 0; i < W * H; i++) {
        uint16_t expected = (uint16_t)(img.data[i] * 65535.0f + 0.5f);
        ASSERT_EQ((int)got[i], (int)expected);
    }

    free(got); free(img.data); rm(path);
    return 0;
}

/* -------------------------------------------------------------------------
 * PNG RGB tests
 * ------------------------------------------------------------------------- */

static int test_png_8bit_rgb(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_png8_rgb.png");
    int W = 8, H = 4;
    Image r = make_gradient(W, H, 1.0f);
    Image g = make_gradient(W, H, 0.5f);
    Image b = make_gradient(W, H, 0.25f);
    ASSERT(r.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT8, 0.0f, 1.0f};
    ASSERT_OK(image_save_rgb(path, &r, &g, &b, &opts));

    FILE *fp = fopen(path, "rb");
    ASSERT(fp != NULL);
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_read_info(png, info);
    int rW = (int)png_get_image_width(png, info);
    int rH = (int)png_get_image_height(png, info);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);
    ASSERT_EQ((int)png_get_color_type(png, info), (int)PNG_COLOR_TYPE_RGB);

    uint8_t *row = (uint8_t *)malloc((size_t)W * 3);
    int ok = 1;
    for (int y = 0; y < H && ok; y++) {
        png_read_row(png, row, NULL);
        for (int x = 0; x < W && ok; x++) {
            int i = y * W + x;
            if (row[x*3+0] != (uint8_t)(r.data[i] * 255.0f + 0.5f)) ok = 0;
            if (row[x*3+1] != (uint8_t)(g.data[i] * 255.0f + 0.5f)) ok = 0;
            if (row[x*3+2] != (uint8_t)(b.data[i] * 255.0f + 0.5f)) ok = 0;
        }
    }
    free(row);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    ASSERT_EQ(ok, 1);

    free(r.data); free(g.data); free(b.data); rm(path);
    return 0;
}

static int test_png_16bit_rgb(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_png16_rgb.png");
    int W = 8, H = 4;
    Image r = make_gradient(W, H, 1.0f);
    Image g = make_gradient(W, H, 0.5f);
    Image b = make_gradient(W, H, 0.25f);
    ASSERT(r.data != NULL);

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT16, 0.0f, 1.0f};
    ASSERT_OK(image_save_rgb(path, &r, &g, &b, &opts));

    FILE *fp = fopen(path, "rb");
    ASSERT(fp != NULL);
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_read_info(png, info);
    int rW = (int)png_get_image_width(png, info);
    int rH = (int)png_get_image_height(png, info);
    ASSERT_EQ(rW, W); ASSERT_EQ(rH, H);

    /* 16-bit RGB: 6 bytes per pixel big-endian */
    uint8_t *row = (uint8_t *)malloc((size_t)W * 6);
    int ok = 1;
    for (int y = 0; y < H && ok; y++) {
        png_read_row(png, row, NULL);
        for (int x = 0; x < W && ok; x++) {
            int i = y * W + x;
            uint16_t rv = (uint16_t)(((uint16_t)row[x*6+0] << 8) | row[x*6+1]);
            uint16_t gv = (uint16_t)(((uint16_t)row[x*6+2] << 8) | row[x*6+3]);
            uint16_t bv = (uint16_t)(((uint16_t)row[x*6+4] << 8) | row[x*6+5]);
            if (rv != (uint16_t)(r.data[i] * 65535.0f + 0.5f)) ok = 0;
            if (gv != (uint16_t)(g.data[i] * 65535.0f + 0.5f)) ok = 0;
            if (bv != (uint16_t)(b.data[i] * 65535.0f + 0.5f)) ok = 0;
        }
    }
    free(row);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    ASSERT_EQ(ok, 1);

    free(r.data); free(g.data); free(b.data); rm(path);
    return 0;
}

/* -------------------------------------------------------------------------
 * Error cases
 * ------------------------------------------------------------------------- */

static int test_png_fp32_returns_error(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_png_fp32_err.png");
    int W = 4, H = 4;
    Image img = make_gradient(W, H, 1.0f);
    ASSERT(img.data != NULL);

    /* NULL opts defaults to FP32 — should be rejected for PNG */
    ASSERT_ERR(image_save(path, &img, NULL), DSO_ERR_INVALID_ARG);

    free(img.data); rm(path);
    return 0;
}

static int test_unknown_ext_returns_error(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_unknown.xyz");
    int W = 4, H = 4;
    Image img = make_gradient(W, H, 1.0f);
    ASSERT(img.data != NULL);

    ASSERT_ERR(image_save(path, &img, NULL), DSO_ERR_INVALID_ARG);

    free(img.data);
    return 0;
}

/* -------------------------------------------------------------------------
 * Auto stretch
 * ------------------------------------------------------------------------- */

static int test_tiff_int16_auto_stretch(void)
{
    char path[512]; TEST_TMPPATH(path, "test_io_tiff_auto.tiff");
    int W = 4, H = 4;
    /* Gradient [100, 200]; auto stretch should map 100→0 and 200→65535 */
    Image img = make_gradient(W, H, 100.0f);
    ASSERT(img.data != NULL);
    for (int i = 0; i < W * H; i++) img.data[i] += 100.0f;

    ImageSaveOptions opts = {TIFF_COMPRESS_NONE, OUT_BITS_INT16, (float)NAN, (float)NAN};
    ASSERT_OK(image_save(path, &img, &opts));

    int rW, rH;
    uint16_t *got = tiff_read_mono_u16(path, &rW, &rH);
    ASSERT(got != NULL);
    ASSERT_EQ((int)got[0], 0);            /* minimum maps to 0 */
    ASSERT_EQ((int)got[W*H-1], 65535);    /* maximum maps to 65535 */

    free(got); free(img.data); rm(path);
    return 0;
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(void)
{
    SUITE("image_io");

    RUN(test_detect_fits);
    RUN(test_detect_tiff);
    RUN(test_detect_png);
    RUN(test_detect_unknown);
    RUN(test_fits_passthrough);
    RUN(test_tiff_fp32_mono);
    RUN(test_tiff_fp16_mono);
    RUN(test_tiff_int16_mono);
    RUN(test_tiff_int8_mono);
    RUN(test_tiff_zip_roundtrip);
    RUN(test_tiff_lzw_roundtrip);
    RUN(test_tiff_rle_roundtrip);
    RUN(test_tiff_fp32_rgb);
    RUN(test_tiff_int16_rgb_global_scale);
    RUN(test_png_8bit_mono);
    RUN(test_png_16bit_mono);
    RUN(test_png_8bit_rgb);
    RUN(test_png_16bit_rgb);
    RUN(test_png_fp32_returns_error);
    RUN(test_unknown_ext_returns_error);
    RUN(test_tiff_int16_auto_stretch);

    return SUMMARY();
}
