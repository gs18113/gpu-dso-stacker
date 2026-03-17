/*
 * fits_io.c — CFITSIO-based FITS image I/O
 *
 * Loads any FITS image (BITPIX 16 or -32) as a float32 buffer and saves
 * float32 images back to FITS. CFITSIO handles all format conversions
 * transparently (e.g., uint16 dark/flat → float).
 *
 * Memory layout: row-major, pixel (x, y) is data[y * width + x].
 * FITS NAXIS1 = width (columns), NAXIS2 = height (rows).
 */

#include "fits_io.h"
#include <fitsio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

DsoError fits_load(const char *filepath, Image *out)
{
    if (!filepath || !out) return DSO_ERR_INVALID_ARG;

    fitsfile *fptr = NULL;
    int status = 0;

    if (ffopen(&fptr, filepath, READONLY, &status)) {
        fprintf(stderr, "fits_load: cannot open '%s' (status=%d)\n", filepath, status);
        return DSO_ERR_FITS;
    }

    /* Read pixel type and image dimensions.
     * naxes[0] = NAXIS1 = number of columns (width)
     * naxes[1] = NAXIS2 = number of rows    (height) */
    int bitpix = 0;
    long naxes[2] = {0, 0};
    ffgidt(fptr, &bitpix, &status);
    ffgisz(fptr, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        ffclos(fptr, &status);
        return DSO_ERR_FITS;
    }

    int width  = (int)naxes[0];
    int height = (int)naxes[1];
    LONGLONG nelem = (LONGLONG)width * height;

    float *data = (float *)malloc((size_t)nelem * sizeof(float));
    if (!data) {
        ffclos(fptr, &status);
        return DSO_ERR_ALLOC;
    }

    /* CFITSIO converts any BITPIX to TFLOAT automatically.
     * firstpix is a 1-based pixel index array per NAXIS dimension. */
    float nulval = 0.f;   /* replacement value for undefined pixels */
    int anynul = 0;
    long firstpix[2] = {1, 1};
    ffgpxv(fptr, TFLOAT, firstpix, nelem, &nulval, data, &anynul, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(data);
        ffclos(fptr, &status);
        return DSO_ERR_FITS;
    }

    ffclos(fptr, &status);

    out->data   = data;
    out->width  = width;
    out->height = height;
    return DSO_OK;
}

DsoError fits_save(const char *filepath, const Image *img)
{
    if (!filepath || !img || !img->data) return DSO_ERR_INVALID_ARG;

    /* Prepend '!' so CFITSIO overwrites any existing file at that path */
    char overwrite_path[4097];
    overwrite_path[0] = '!';
    strncpy(overwrite_path + 1, filepath, sizeof(overwrite_path) - 2);
    overwrite_path[sizeof(overwrite_path) - 1] = '\0';

    fitsfile *fptr = NULL;
    int status = 0;
    long naxes[2] = { img->width, img->height };

    /* Create new file and write primary image HDU as FLOAT_IMG (-32 BITPIX) */
    ffinit(&fptr, overwrite_path, &status);
    ffcrim(fptr, FLOAT_IMG, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        if (fptr) { int s2 = 0; ffclos(fptr, &s2); }
        return DSO_ERR_FITS;
    }

    LONGLONG nelem = (LONGLONG)img->width * img->height;
    long firstpix[2] = {1, 1};
    ffppx(fptr, TFLOAT, firstpix, nelem, img->data, &status);
    if (status) {
        fits_report_error(stderr, status);
        ffclos(fptr, &status);
        return DSO_ERR_FITS;
    }

    ffclos(fptr, &status);
    return DSO_OK;
}

/* Free the data buffer inside an Image struct and null the pointer.
 * Safe to call on a zeroed/stack-allocated Image. */
void image_free(Image *img)
{
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

DsoError fits_get_bayer_pattern(const char *filepath, BayerPattern *pattern_out)
{
    if (!filepath || !pattern_out) return DSO_ERR_INVALID_ARG;

    /* Default: assume monochrome (no Bayer pattern) */
    *pattern_out = BAYER_NONE;

    fitsfile *fptr = NULL;
    int status = 0;

    if (ffopen(&fptr, filepath, READONLY, &status)) {
        /* Cannot open file — treat as no pattern rather than hard error,
         * since this function is often called speculatively. */
        fprintf(stderr, "fits_get_bayer_pattern: cannot open '%s' (status=%d)\n",
                filepath, status);
        return DSO_ERR_FITS;
    }

    /* Read the BAYERPAT keyword as a string value.
     * ffgkys returns a non-zero status if the keyword is absent — that is
     * normal for monochrome images and should not propagate as an error. */
    char bayerpat[FLEN_VALUE] = {0};
    char comment[FLEN_COMMENT] = {0};
    int ks = 0;
    ffgkys(fptr, "BAYERPAT", bayerpat, comment, &ks);

    ffclos(fptr, &status);

    if (ks != 0) {
        /* Keyword absent — monochrome sensor, BAYER_NONE already set */
        return DSO_OK;
    }

    /* Strip leading/trailing whitespace and single-quotes that CFITSIO may add */
    char *p = bayerpat;
    while (*p == ' ' || *p == '\'' || *p == '"') p++;
    char *end = p + strlen(p);
    while (end > p && (*(end-1) == ' ' || *(end-1) == '\'' || *(end-1) == '"'))
        end--;
    *end = '\0';

    /* Case-insensitive comparison against known Bayer patterns */
    if      (strcasecmp(p, "RGGB") == 0) *pattern_out = BAYER_RGGB;
    else if (strcasecmp(p, "BGGR") == 0) *pattern_out = BAYER_BGGR;
    else if (strcasecmp(p, "GRBG") == 0) *pattern_out = BAYER_GRBG;
    else if (strcasecmp(p, "GBRG") == 0) *pattern_out = BAYER_GBRG;
    else {
        fprintf(stderr,
                "fits_get_bayer_pattern: unrecognised BAYERPAT='%s', assuming NONE\n", p);
    }

    return DSO_OK;
}
