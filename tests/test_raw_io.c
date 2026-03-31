/*
 * test_raw_io.c — Tests for the frame_load dispatch layer and raw_io module.
 *
 * Tests that always run (regardless of DSO_HAS_LIBRAW):
 *   - frame_is_raw() extension detection for all RAW + non-RAW extensions
 *   - frame_load() FITS fallback (create synthetic FITS, load via frame_load)
 *   - frame_get_bayer_pattern() FITS path through dispatch layer
 *   - frame_load() with RAW extension returns DSO_ERR_IO when LibRaw disabled
 *
 * Conditional tests (DSO_HAS_LIBRAW=1):
 *   - raw_load() error handling for nonexistent file
 *   - raw_get_bayer_pattern() error handling for nonexistent file
 *   - raw_get_dimensions() error handling for nonexistent file
 */

#include "test_framework.h"
#include "frame_load.h"
#include "fits_io.h"
#include "dso_types.h"

#if DSO_HAS_LIBRAW
#include "raw_io.h"
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/* =========================================================================
 * Extension detection tests
 * ========================================================================= */

static int test_frame_is_raw_positive(void)
{
    /* All recognized RAW extensions (lower and upper case) */
    ASSERT(frame_is_raw("photo.cr2"));
    ASSERT(frame_is_raw("photo.CR2"));
    ASSERT(frame_is_raw("photo.cr3"));
    ASSERT(frame_is_raw("photo.CR3"));
    ASSERT(frame_is_raw("photo.nef"));
    ASSERT(frame_is_raw("photo.NEF"));
    ASSERT(frame_is_raw("photo.arw"));
    ASSERT(frame_is_raw("photo.ARW"));
    ASSERT(frame_is_raw("photo.orf"));
    ASSERT(frame_is_raw("photo.rw2"));
    ASSERT(frame_is_raw("photo.raf"));
    ASSERT(frame_is_raw("photo.dng"));
    ASSERT(frame_is_raw("photo.DNG"));
    ASSERT(frame_is_raw("photo.pef"));
    ASSERT(frame_is_raw("photo.srw"));
    ASSERT(frame_is_raw("photo.raw"));
    ASSERT(frame_is_raw("photo.3fr"));
    ASSERT(frame_is_raw("photo.iiq"));
    ASSERT(frame_is_raw("photo.rwl"));
    ASSERT(frame_is_raw("photo.nrw"));

    /* Mixed case */
    ASSERT(frame_is_raw("/path/to/IMG_1234.Cr2"));
    ASSERT(frame_is_raw("C:\\Photos\\DSC_5678.NeF"));
    return 0;
}

static int test_frame_is_raw_negative(void)
{
    /* FITS and other non-RAW formats */
    ASSERT(!frame_is_raw("image.fits"));
    ASSERT(!frame_is_raw("image.fit"));
    ASSERT(!frame_is_raw("image.fts"));
    ASSERT(!frame_is_raw("image.tiff"));
    ASSERT(!frame_is_raw("image.tif"));
    ASSERT(!frame_is_raw("image.png"));
    ASSERT(!frame_is_raw("image.jpg"));
    ASSERT(!frame_is_raw("image.txt"));
    ASSERT(!frame_is_raw("image.csv"));
    ASSERT(!frame_is_raw("noext"));
    ASSERT(!frame_is_raw(""));
    ASSERT(!frame_is_raw(NULL));
    return 0;
}


/* =========================================================================
 * FITS fallback tests — frame_load dispatches to fits_load for non-RAW
 * ========================================================================= */

static int test_frame_load_fits_fallback(void)
{
    /* Create a small synthetic FITS file */
    const int W = 4, H = 3;
    float data[12];
    for (int i = 0; i < 12; i++) data[i] = (float)i * 100.0f;

    Image src = { data, W, H };
    const char *path = "/tmp/test_frame_load_fallback.fits";
    ASSERT_OK(fits_save(path, &src));

    /* Load through frame_load (should dispatch to fits_load) */
    Image out = {NULL, 0, 0};
    ASSERT_OK(frame_load(path, &out));
    ASSERT(out.width == W);
    ASSERT(out.height == H);

    /* Verify pixel data */
    for (int i = 0; i < 12; i++) {
        ASSERT(fabsf(out.data[i] - data[i]) < 0.01f);
    }

    image_free(&out);
    remove(path);
    return 0;
}

static int test_frame_get_bayer_fits(void)
{
    /* Create a FITS with BAYERPAT keyword */
    const int W = 4, H = 4;
    float data[16] = {0};
    Image src = { data, W, H };
    const char *path = "/tmp/test_frame_bayer.fits";
    ASSERT_OK(fits_save(path, &src));

    /* fits_save doesn't write BAYERPAT, so we should get BAYER_NONE */
    BayerPattern pat = BAYER_RGGB;  /* pre-set to non-NONE */
    ASSERT_OK(frame_get_bayer_pattern(path, &pat));
    ASSERT(pat == BAYER_NONE);

    remove(path);
    return 0;
}

static int test_frame_load_invalid_args(void)
{
    Image out = {NULL, 0, 0};
    ASSERT_ERR(frame_load(NULL, &out), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(frame_load("test.fits", NULL), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_frame_get_dimensions_fits(void)
{
    const int W = 8, H = 6;
    float data[48] = {0};
    Image src = { data, W, H };
    const char *path = "/tmp/test_frame_dims.fits";
    ASSERT_OK(fits_save(path, &src));

    int w_out = 0, h_out = 0;
    ASSERT_OK(frame_get_dimensions(path, &w_out, &h_out));
    ASSERT(w_out == W);
    ASSERT(h_out == H);

    remove(path);
    return 0;
}


/* =========================================================================
 * RAW-disabled tests (always run)
 * ========================================================================= */

#if !DSO_HAS_LIBRAW
static int test_frame_load_raw_returns_error(void)
{
    /* When LibRaw is not compiled in, loading a RAW file should fail */
    Image out = {NULL, 0, 0};
    DsoError err = frame_load("nonexistent.cr2", &out);
    ASSERT(err == DSO_ERR_IO);
    ASSERT(out.data == NULL);
    return 0;
}
#endif


/* =========================================================================
 * RAW-enabled tests (only when DSO_HAS_LIBRAW=1)
 * ========================================================================= */

#if DSO_HAS_LIBRAW
static int test_raw_load_missing_file(void)
{
    Image out = {NULL, 0, 0};
    DsoError err = raw_load("/tmp/nonexistent_file.cr2", &out);
    ASSERT(err == DSO_ERR_RAW);
    ASSERT(out.data == NULL);
    return 0;
}

static int test_raw_get_bayer_missing_file(void)
{
    BayerPattern pat = BAYER_RGGB;
    DsoError err = raw_get_bayer_pattern("/tmp/nonexistent_file.nef", &pat);
    ASSERT(err == DSO_ERR_RAW);
    return 0;
}

static int test_raw_get_dims_missing_file(void)
{
    int w = 0, h = 0;
    DsoError err = raw_get_dimensions("/tmp/nonexistent_file.arw", &w, &h);
    ASSERT(err == DSO_ERR_RAW);
    return 0;
}

static int test_raw_load_invalid_args(void)
{
    Image out = {NULL, 0, 0};
    ASSERT_ERR(raw_load(NULL, &out), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(raw_load("test.cr2", NULL), DSO_ERR_INVALID_ARG);
    return 0;
}
#endif


/* =========================================================================
 * Test runner
 * ========================================================================= */

int main(void)
{
    SUITE("frame_is_raw — extension detection");
    RUN(test_frame_is_raw_positive);
    RUN(test_frame_is_raw_negative);

    SUITE("frame_load — FITS fallback");
    RUN(test_frame_load_fits_fallback);
    RUN(test_frame_get_bayer_fits);
    RUN(test_frame_load_invalid_args);
    RUN(test_frame_get_dimensions_fits);

#if !DSO_HAS_LIBRAW
    SUITE("frame_load — RAW disabled");
    RUN(test_frame_load_raw_returns_error);
#endif

#if DSO_HAS_LIBRAW
    SUITE("raw_io — error handling");
    RUN(test_raw_load_missing_file);
    RUN(test_raw_get_bayer_missing_file);
    RUN(test_raw_get_dims_missing_file);
    RUN(test_raw_load_invalid_args);
#endif

    return SUMMARY();
}
