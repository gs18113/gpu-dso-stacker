/*
 * frame_load.c — Format-agnostic image loading dispatch layer.
 *
 * Detects file format by extension and routes to the appropriate backend:
 *   - RAW camera files (.cr2, .nef, .arw, .dng, …) → raw_io.c (LibRaw)
 *   - Everything else → fits_io.c (CFITSIO)
 *
 * When DSO_HAS_LIBRAW=0 (built without LibRaw), RAW file extensions produce
 * DSO_ERR_IO with a diagnostic message pointing to the build flag.
 */

#include "frame_load.h"
#include "fits_io.h"

#if DSO_HAS_LIBRAW
#include "raw_io.h"
#endif

#include <string.h>
#include <stdio.h>
#include <ctype.h>


/* -------------------------------------------------------------------------
 * Extension table for recognized RAW camera formats
 * ------------------------------------------------------------------------- */
static const char *raw_extensions[] = {
    ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".dng",
    ".pef", ".srw", ".raw", ".3fr", ".iiq", ".rwl", ".nrw",
    NULL
};


/* Case-insensitive extension comparison */
static int ext_eq(const char *a, const char *b)
{
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}


int frame_is_raw(const char *filepath)
{
    if (!filepath) return 0;
    const char *dot = strrchr(filepath, '.');
    if (!dot) return 0;

    for (int i = 0; raw_extensions[i]; i++) {
        if (ext_eq(dot, raw_extensions[i]))
            return 1;
    }
    return 0;
}


/* =========================================================================
 * Load dispatch
 * ========================================================================= */

DsoError frame_load(const char *filepath, Image *out)
{
    if (!filepath || !out) return DSO_ERR_INVALID_ARG;

    if (frame_is_raw(filepath)) {
#if DSO_HAS_LIBRAW
        return raw_load(filepath, out);
#else
        fprintf(stderr, "frame_load: RAW file '%s' not supported "
                "(build with -DDSO_ENABLE_LIBRAW=ON)\n", filepath);
        return DSO_ERR_IO;
#endif
    }
    return fits_load(filepath, out);
}


DsoError frame_load_to_buffer(const char *filepath, float *buffer, int W, int H)
{
    if (!filepath || !buffer) return DSO_ERR_INVALID_ARG;

    if (frame_is_raw(filepath)) {
#if DSO_HAS_LIBRAW
        return raw_load_to_buffer(filepath, buffer, W, H);
#else
        fprintf(stderr, "frame_load_to_buffer: RAW file '%s' not supported "
                "(build with -DDSO_ENABLE_LIBRAW=ON)\n", filepath);
        return DSO_ERR_IO;
#endif
    }
    return fits_load_to_buffer(filepath, buffer, W, H);
}


DsoError frame_get_bayer_pattern(const char *filepath, BayerPattern *pattern_out)
{
    if (!filepath || !pattern_out) return DSO_ERR_INVALID_ARG;

    if (frame_is_raw(filepath)) {
#if DSO_HAS_LIBRAW
        return raw_get_bayer_pattern(filepath, pattern_out);
#else
        fprintf(stderr, "frame_get_bayer_pattern: RAW file '%s' not supported "
                "(build with -DDSO_ENABLE_LIBRAW=ON)\n", filepath);
        *pattern_out = BAYER_NONE;
        return DSO_ERR_IO;
#endif
    }
    return fits_get_bayer_pattern(filepath, pattern_out);
}


DsoError frame_get_dimensions(const char *filepath, int *width_out, int *height_out)
{
    if (!filepath || !width_out || !height_out) return DSO_ERR_INVALID_ARG;

    if (frame_is_raw(filepath)) {
#if DSO_HAS_LIBRAW
        return raw_get_dimensions(filepath, width_out, height_out);
#else
        fprintf(stderr, "frame_get_dimensions: RAW file '%s' not supported "
                "(build with -DDSO_ENABLE_LIBRAW=ON)\n", filepath);
        return DSO_ERR_IO;
#endif
    }

    /* FITS path: open, read dimensions, close.
     * CFITSIO has no header-only API, so we load + free.
     * This is consistent with how pipeline_cpu.c reads dimensions. */
    Image tmp = {NULL, 0, 0};
    DsoError err = fits_load(filepath, &tmp);
    if (err != DSO_OK) return err;
    *width_out  = tmp.width;
    *height_out = tmp.height;
    image_free(&tmp);
    return DSO_OK;
}


DsoError frame_get_wb_multipliers(const char *filepath,
                                   float *r_mul, float *g_mul, float *b_mul)
{
    if (!filepath || !r_mul || !g_mul || !b_mul) return DSO_ERR_INVALID_ARG;

    if (frame_is_raw(filepath)) {
#if DSO_HAS_LIBRAW
        return raw_get_wb_multipliers(filepath, r_mul, g_mul, b_mul);
#else
        fprintf(stderr, "frame_get_wb_multipliers: RAW file '%s' not supported "
                "(build with -DDSO_ENABLE_LIBRAW=ON)\n", filepath);
        *r_mul = 1.0f; *g_mul = 1.0f; *b_mul = 1.0f;
        return DSO_ERR_IO;
#endif
    }
    return fits_get_wb_multipliers(filepath, r_mul, g_mul, b_mul);
}
