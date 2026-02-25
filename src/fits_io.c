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
