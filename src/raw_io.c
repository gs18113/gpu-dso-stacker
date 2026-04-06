/*
 * raw_io.c — RAW camera file I/O via LibRaw.
 *
 * Loads RAW camera files (CR2, NEF, ARW, DNG, etc.) as float32 Bayer
 * mosaic data suitable for the DSO stacker's VNG debayer pipeline.
 *
 * LibRaw workflow:
 *   libraw_init() → libraw_open_file() → libraw_unpack() → read raw_image
 *   → convert uint16 to float32 with per-channel black subtraction
 *   → libraw_close()
 *
 * libraw_dcraw_process() is intentionally NOT called — we want the raw
 * undemosaiced Bayer mosaic, not LibRaw's own interpolated output.
 *
 * The usable image area is raw_image[top_margin..][left_margin..] with
 * dimensions sizes.width × sizes.height.  Masked border pixels are excluded.
 *
 * Per-channel black subtraction:
 *   black[c] = color.cblack[c] + color.black   (c = COLOR(row, col))
 *   pixel_out = clamp((raw_val - black[c]) / (maximum - black[c]), 0, 1)
 */

#include "raw_io.h"
#include <libraw/libraw.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/* -------------------------------------------------------------------------
 * Internal: map LibRaw idata.filters to BayerPattern enum.
 *
 * LibRaw encodes the 2×2 CFA pattern as a repeating 32-bit mask.
 * Standard Bayer sensors use one of four canonical patterns.
 * Non-Bayer sensors (X-Trans, Foveon, monochrome) → BAYER_NONE.
 * ------------------------------------------------------------------------- */
static BayerPattern filters_to_bayer(unsigned filters)
{
    /* LibRaw repeats the 2×2 pattern across 8 nibbles:
     * RGGB = 0x94949494, BGGR = 0x16161616,
     * GRBG = 0x61616161, GBRG = 0x49494949   */
    unsigned pat = filters & 0xFFFFFFFF;

    if (pat == 0x94949494u) return BAYER_RGGB;
    if (pat == 0x16161616u) return BAYER_BGGR;
    if (pat == 0x61616161u) return BAYER_GRBG;
    if (pat == 0x49494949u) return BAYER_GBRG;

    return BAYER_NONE;
}


/* -------------------------------------------------------------------------
 * Internal: unpack raw data and write float32 Bayer mosaic to buffer.
 *
 * Shared logic for raw_load() and raw_load_to_buffer().
 * If W_expect / H_expect are > 0, validates dimensions match.
 * On success, *W_out and *H_out are set to the usable area dimensions.
 * ------------------------------------------------------------------------- */
static DsoError raw_unpack_to_float(const char *filepath,
                                     float *buffer,
                                     int W_expect, int H_expect,
                                     int *W_out, int *H_out)
{
    libraw_data_t *raw = libraw_init(LIBRAW_OPTIONS_NONE);
    if (!raw) {
        fprintf(stderr, "raw_io: libraw_init failed\n");
        return DSO_ERR_RAW;
    }

    int ret = libraw_open_file(raw, filepath);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "raw_io: cannot open '%s': %s\n",
                filepath, libraw_strerror(ret));
        libraw_close(raw);
        return DSO_ERR_RAW;
    }

    int W = (int)raw->sizes.width;
    int H = (int)raw->sizes.height;

    if (W_expect > 0 && H_expect > 0 && (W != W_expect || H != H_expect)) {
        fprintf(stderr, "raw_io: size mismatch for '%s' "
                "(expected %dx%d, got %dx%d)\n",
                filepath, W_expect, H_expect, W, H);
        libraw_close(raw);
        return DSO_ERR_INVALID_ARG;
    }

    ret = libraw_unpack(raw);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "raw_io: unpack failed for '%s': %s\n",
                filepath, libraw_strerror(ret));
        libraw_close(raw);
        return DSO_ERR_RAW;
    }

    /* Verify raw_image is available (some exotic formats use color4_image) */
    if (!raw->rawdata.raw_image) {
        fprintf(stderr, "raw_io: no single-channel raw_image for '%s' "
                "(non-Bayer sensor?)\n", filepath);
        libraw_close(raw);
        return DSO_ERR_RAW;
    }

    /* Per-channel black levels */
    float black[4];
    for (int c = 0; c < 4; c++)
        black[c] = (float)(raw->color.cblack[c] + raw->color.black);
    float white = (float)raw->color.maximum;

    int top    = (int)raw->sizes.top_margin;
    int left   = (int)raw->sizes.left_margin;
    int raw_w  = (int)raw->sizes.raw_width;

    /* Convert uint16 Bayer mosaic to float32 with black subtraction */
    for (int row = 0; row < H; row++) {
        int raw_row = row + top;
        for (int col = 0; col < W; col++) {
            int raw_col = col + left;
            unsigned short val =
                raw->rawdata.raw_image[raw_row * raw_w + raw_col];
            int c = libraw_COLOR(raw, raw_row, raw_col);
            float range = white - black[c];
            float f;
            if (range > 0.0f)
                f = ((float)val - black[c]) / range;
            else
                f = 0.0f;
            if (f < 0.0f) f = 0.0f;
            if (f > 1.0f) f = 1.0f;
            buffer[row * W + col] = f;
        }
    }

    if (W_out) *W_out = W;
    if (H_out) *H_out = H;

    libraw_close(raw);
    return DSO_OK;
}


/* =========================================================================
 * Public API
 * ========================================================================= */

DsoError raw_load(const char *filepath, Image *out)
{
    if (!filepath || !out) return DSO_ERR_INVALID_ARG;

    /* First pass: open to get dimensions (no unpack) */
    libraw_data_t *probe = libraw_init(LIBRAW_OPTIONS_NONE);
    if (!probe) return DSO_ERR_RAW;

    int ret = libraw_open_file(probe, filepath);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "raw_load: cannot open '%s': %s\n",
                filepath, libraw_strerror(ret));
        libraw_close(probe);
        return DSO_ERR_RAW;
    }
    int W = (int)probe->sizes.width;
    int H = (int)probe->sizes.height;
    libraw_close(probe);

    /* Allocate output buffer */
    long long npix = (long long)W * H;
    float *data = (float *)malloc((size_t)npix * sizeof(float));
    if (!data) return DSO_ERR_ALLOC;

    /* Second pass: unpack and convert */
    DsoError err = raw_unpack_to_float(filepath, data, W, H, NULL, NULL);
    if (err != DSO_OK) {
        free(data);
        return err;
    }

    out->data   = data;
    out->width  = W;
    out->height = H;
    return DSO_OK;
}


DsoError raw_load_to_buffer(const char *filepath, float *buffer, int W, int H)
{
    if (!filepath || !buffer) return DSO_ERR_INVALID_ARG;
    return raw_unpack_to_float(filepath, buffer, W, H, NULL, NULL);
}


DsoError raw_get_bayer_pattern(const char *filepath, BayerPattern *pattern_out)
{
    if (!filepath || !pattern_out) return DSO_ERR_INVALID_ARG;

    *pattern_out = BAYER_NONE;

    libraw_data_t *raw = libraw_init(LIBRAW_OPTIONS_NONE);
    if (!raw) return DSO_ERR_RAW;

    int ret = libraw_open_file(raw, filepath);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "raw_get_bayer_pattern: cannot open '%s': %s\n",
                filepath, libraw_strerror(ret));
        libraw_close(raw);
        return DSO_ERR_RAW;
    }

    BayerPattern pat = filters_to_bayer(raw->idata.filters);
    if (pat == BAYER_NONE && raw->idata.filters != 0) {
        fprintf(stderr, "raw_get_bayer_pattern: non-standard CFA filter "
                "0x%08x for '%s', treating as monochrome\n",
                raw->idata.filters, filepath);
    }
    *pattern_out = pat;

    libraw_close(raw);
    return DSO_OK;
}


DsoError raw_get_dimensions(const char *filepath, int *width_out, int *height_out)
{
    if (!filepath || !width_out || !height_out) return DSO_ERR_INVALID_ARG;

    libraw_data_t *raw = libraw_init(LIBRAW_OPTIONS_NONE);
    if (!raw) return DSO_ERR_RAW;

    int ret = libraw_open_file(raw, filepath);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "raw_get_dimensions: cannot open '%s': %s\n",
                filepath, libraw_strerror(ret));
        libraw_close(raw);
        return DSO_ERR_RAW;
    }

    *width_out  = (int)raw->sizes.width;
    *height_out = (int)raw->sizes.height;

    libraw_close(raw);
    return DSO_OK;
}


DsoError raw_get_wb_multipliers(const char *filepath,
                                 float *r_mul, float *g_mul, float *b_mul)
{
    if (!filepath || !r_mul || !g_mul || !b_mul)
        return DSO_ERR_INVALID_ARG;

    /* Defaults */
    *r_mul = 1.0f;
    *g_mul = 1.0f;
    *b_mul = 1.0f;

    libraw_data_t *raw = libraw_init(LIBRAW_OPTIONS_NONE);
    if (!raw) return DSO_ERR_RAW;

    int ret = libraw_open_file(raw, filepath);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "raw_get_wb_multipliers: cannot open '%s': %s\n",
                filepath, libraw_strerror(ret));
        libraw_close(raw);
        return DSO_ERR_RAW;
    }

    /* cam_mul[4]: R, G, B, G2 — normalize so green (index 1) = 1.0 */
    float g = raw->color.cam_mul[1];
    if (g < 1e-10f) {
        fprintf(stderr, "raw_get_wb_multipliers: green cam_mul is zero "
                "for '%s', using defaults\n", filepath);
        libraw_close(raw);
        return DSO_OK;
    }

    *r_mul = raw->color.cam_mul[0] / g;
    *g_mul = 1.0f;
    *b_mul = raw->color.cam_mul[2] / g;

    libraw_close(raw);
    return DSO_OK;
}
