/*
 * calibration.c — Astronomical calibration frame generation and application.
 *
 * Implements:
 *   - FITS vs. text-list detection
 *   - Frame list parsing
 *   - Per-pixel stacking with winsorized mean or median
 *   - Master frame generation: bias, darkflat, dark, flat
 *   - CPU in-place application of master frames to raw light data
 *
 * Flat normalization
 * ------------------
 * Each flat frame is divided by its own (double-precision) mean before
 * stacking.  This preserves the per-pixel vignetting / dust-ring pattern
 * while cancelling out session-to-session illumination differences.
 * The stacked master flat therefore has mean ≈ 1.0 and is used directly
 * as a divisor for light frames.
 *
 * Winsorized mean
 * ---------------
 * Controlled by wsor_clip ∈ [0.0, 0.49] (fraction per side, default 0.1).
 * For N values per pixel sorted ascending:
 *   g = floor(wsor_clip * N)
 *   vals[0..g-1]       = vals[g]      (clamp bottom)
 *   vals[N-g..N-1]     = vals[N-1-g]  (clamp top)
 *   result = mean(vals) using double accumulation
 *
 * OpenMP
 * ------
 * Per-pixel loops use schedule(dynamic, 64) because the per-pixel work
 * is variable (insertion sort on N elements).
 */

#define _POSIX_C_SOURCE 200809L   /* for strdup */

#include "calibration.h"
#include "fits_io.h"
#include "compat.h"

#include <fitsio.h>   /* fits_open_file, fits_close_file (for FITS probe) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _MSC_VER
#include <sys/stat.h>  /* mkdir (POSIX) */
#endif
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/* -------------------------------------------------------------------------
 * FITS file probe
 *
 * Returns 1 if 'path' is a valid FITS file, 0 otherwise.
 * Uses CFITSIO: if fits_open_file succeeds the file is FITS.
 * ------------------------------------------------------------------------- */
static int is_fits_file(const char *path)
{
    fitsfile *fp  = NULL;
    int       status = 0;
    fits_open_file(&fp, path, READONLY, &status);
    if (status == 0) {
        fits_close_file(fp, &status);
        return 1;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Frame list parser
 *
 * Opens a plain-text file with one FITS path per line.
 * Trims trailing whitespace/CR/LF.  Skips blank lines and '#' comments.
 * Caller must free(*paths_out)[i] and *paths_out.
 * ------------------------------------------------------------------------- */
static DsoError parse_framelist(const char *path, char ***paths_out, int *n_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "calib: cannot open frame list '%s'\n", path);
        return DSO_ERR_IO;
    }

    int    capacity = 16;
    int    n        = 0;
    char **paths    = (char **)malloc((size_t)capacity * sizeof(char *));
    if (!paths) { fclose(fp); return DSO_ERR_ALLOC; }

    char line[4096];
    while (fgets(line, (int)sizeof(line), fp)) {
        /* Trim trailing whitespace/newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r' ||
                           line[len-1] == ' '  || line[len-1] == '\t'))
            line[--len] = '\0';

        /* Skip empty lines and comments */
        if (len == 0 || line[0] == '#')
            continue;

        /* Grow array if needed */
        if (n >= capacity) {
            capacity *= 2;
            char **np = (char **)realloc(paths, (size_t)capacity * sizeof(char *));
            if (!np) {
                for (int i = 0; i < n; i++) free(paths[i]);
                free(paths);
                fclose(fp);
                return DSO_ERR_ALLOC;
            }
            paths = np;
        }

        paths[n] = strdup(line);
        if (!paths[n]) {
            for (int i = 0; i < n; i++) free(paths[i]);
            free(paths);
            fclose(fp);
            return DSO_ERR_ALLOC;
        }
        n++;
    }
    fclose(fp);

    *paths_out = paths;
    *n_out     = n;
    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * stack_frames — per-pixel stacking of N pre-loaded float buffers.
 *
 * paths[]       : FITS file paths to load (paths[0] sets output dimensions)
 * n             : number of frames
 * method        : stacking algorithm
 * subtract_buf  : if non-NULL, subtracted from each loaded frame before
 *                 stacking (must have same dimensions)
 * normalize     : if 1, each frame is divided by its own mean (flat mode)
 * master_out    : receives the stacked image; caller must image_free() it
 * ------------------------------------------------------------------------- */
static DsoError stack_frames(const char **paths, int n,
                              CalibMethod method,
                              const float *subtract_buf,
                              int normalize,
                              float wsor_clip,
                              Image *master_out)
{
    if (n <= 0) return DSO_ERR_INVALID_ARG;

    DsoError  err      = DSO_OK;
    float   **bufs     = NULL;
    float    *all_vals = NULL;
    int       W = 0, H = 0;
    long npix = 0;

    bufs = (float **)calloc((size_t)n, sizeof(float *));
    if (!bufs) return DSO_ERR_ALLOC;

    /* Load all frames; derive dimensions from first frame */
    for (int i = 0; i < n; i++) {
        Image img = {NULL, 0, 0};
        err = fits_load(paths[i], &img);
        if (err != DSO_OK) {
            fprintf(stderr, "calib: failed to load '%s' (err=%d)\n", paths[i], (int)err);
            goto cleanup;
        }

        if (i == 0) {
            W    = img.width;
            H    = img.height;
            npix = (long)W * H;
        } else if (img.width != W || img.height != H) {
            fprintf(stderr,
                    "calib: frame '%s' size %d×%d does not match first frame %d×%d\n",
                    paths[i], img.width, img.height, W, H);
            image_free(&img);
            err = DSO_ERR_INVALID_ARG;
            goto cleanup;
        }

        bufs[i] = img.data;   /* take ownership of pixel buffer */
        /* img.data is now owned by bufs[i]; clear to avoid double-free */
        img.data = NULL;

        /* Optional bias/darkflat subtraction (in-place) */
        if (subtract_buf) {
            for (long p = 0; p < npix; p++)
                bufs[i][p] -= subtract_buf[p];
        }

        /* Optional per-frame mean normalization (flat mode) */
        if (normalize) {
            double sum = 0.0;
            for (long p = 0; p < npix; p++)
                sum += (double)bufs[i][p];
            double mean = sum / (double)npix;
            if (mean > 1e-9) {
                float inv_mean = (float)(1.0 / mean);
                for (long p = 0; p < npix; p++)
                    bufs[i][p] *= inv_mean;
            } else {
                fprintf(stderr,
                        "calib: flat frame '%s' has near-zero mean (%.3g), "
                        "skipping normalization\n", paths[i], mean);
            }
        }
    }

    /* Allocate master frame buffer */
    master_out->data   = (float *)malloc((size_t)npix * sizeof(float));
    master_out->width  = W;
    master_out->height = H;
    if (!master_out->data) { err = DSO_ERR_ALLOC; goto cleanup; }

    float *master = master_out->data;

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    all_vals = (float *)malloc((size_t)max_threads * n * sizeof(float));
    if (!all_vals) { err = DSO_ERR_ALLOC; goto cleanup; }

    /* Per-pixel stacking (parallelised across pixels) */
#pragma omp parallel for schedule(dynamic, 64)
    for (long p = 0; p < npix; p++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        /* Collect pixel values across frames into a local buffer */
        float *vals = &all_vals[tid * n];
        for (int i = 0; i < n; i++)
            vals[i] = bufs[i][p];

        /* Insertion sort (n typically < 100; fast for small arrays) */
        for (int j = 1; j < n; j++) {
            float tmp = vals[j];
            int   k   = j - 1;
            while (k >= 0 && vals[k] > tmp) {
                vals[k + 1] = vals[k];
                k--;
            }
            vals[k + 1] = tmp;
        }

        if (method == CALIB_MEDIAN) {
            if (n & 1)
                master[p] = vals[n / 2];
            else
                master[p] = 0.5f * (vals[n / 2 - 1] + vals[n / 2]);
        } else {
            /* Winsorized mean: clamp outer floor(wsor_clip*n) on each side */
            int g = (int)(wsor_clip * n);
            if (g > 0) {
                float lo = vals[g];
                float hi = vals[n - 1 - g];
                for (int j = 0;     j < g; j++) vals[j]     = lo;
                for (int j = n - g; j < n; j++) vals[j]     = hi;
            }
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += (double)vals[j];
            master[p] = (float)(sum / n);
        }
    }

cleanup:
    free(all_vals);
    for (int i = 0; i < n; i++) free(bufs[i]);
    free(bufs);
    return err;
}

/* -------------------------------------------------------------------------
 * load_or_generate_one — load a pre-made FITS master or stack from a list.
 *
 * If path is a FITS file: loads it directly.
 * If path is a text list: parses, stacks, optionally saves.
 *
 * was_generated is set to 1 when stacking occurred, 0 for a direct load.
 * subtract_buf and normalize are forwarded to stack_frames().
 * ------------------------------------------------------------------------- */
static DsoError load_or_generate_one(const char *path, CalibMethod method,
                                     const float *subtract_buf, int normalize,
                                     float wsor_clip,
                                     Image *out, int *was_generated)
{
    *was_generated = 0;

    if (is_fits_file(path)) {
        /* Pre-computed master FITS — load directly */
        return fits_load(path, out);
    }

    /* Text frame list — generate master */
    char  **paths = NULL;
    int     n     = 0;
    DsoError err  = parse_framelist(path, &paths, &n);
    if (err != DSO_OK) return err;

    if (n == 0) {
        fprintf(stderr, "calib: frame list '%s' contains no entries\n", path);
        free(paths);
        return DSO_ERR_IO;
    }

    printf("calib: generating master from %d frame(s) in '%s'\n", n, path);

    err = stack_frames((const char **)paths, n, method,
                       subtract_buf, normalize, wsor_clip, out);

    for (int i = 0; i < n; i++) free(paths[i]);
    free(paths);

    if (err == DSO_OK)
        *was_generated = 1;

    return err;
}

/* -------------------------------------------------------------------------
 * ensure_dir — create directory (and ignore EEXIST).
 * ------------------------------------------------------------------------- */
static void ensure_dir(const char *dir)
{
    if (mkdir(dir, 0755) != 0 && errno != EEXIST)
        fprintf(stderr, "calib: warning: could not create '%s': %s\n",
                dir, strerror(errno));
}

/* =========================================================================
 * Public API
 * ========================================================================= */

DsoError calib_load_or_generate(
    const char *dark_path,     CalibMethod dark_method,
    const char *bias_path,     CalibMethod bias_method,
    const char *flat_path,     CalibMethod flat_method,
    const char *darkflat_path, CalibMethod darkflat_method,
    const char *save_dir,
    float        wsor_clip,
    CalibFrames *calib_out)
{
    if (!calib_out) return DSO_ERR_INVALID_ARG;

    /* Clamp wsor_clip to valid range [0.0, 0.49] */
    if (wsor_clip < 0.0f)  wsor_clip = 0.0f;
    if (wsor_clip > 0.49f) wsor_clip = 0.49f;

    DsoError err        = DSO_OK;
    Image    bias       = {NULL, 0, 0};
    Image    darkflat   = {NULL, 0, 0};
    int      bias_gen   = 0;
    int      df_gen     = 0;
    int      dark_gen   = 0;
    int      flat_gen   = 0;

    /* --- Step 1: Load / generate master bias (if requested) --- */
    if (bias_path) {
        printf("calib: processing bias frames...\n");
        err = load_or_generate_one(bias_path, bias_method,
                                   /*subtract_buf=*/NULL, /*normalize=*/0,
                                   wsor_clip, &bias, &bias_gen);
        if (err != DSO_OK) goto cleanup;
        printf("calib: master bias ready (%d×%d)\n", bias.width, bias.height);

        if (bias_gen && save_dir) {
            ensure_dir(save_dir);
            char save_path[4096];
            snprintf(save_path, sizeof(save_path), "%s/master_bias.fits", save_dir);
            err = fits_save(save_path, &bias);
            if (err != DSO_OK) {
                fprintf(stderr, "calib: warning: failed to save master bias to '%s'\n",
                        save_path);
                err = DSO_OK;  /* non-fatal — continue */
            } else {
                printf("calib: saved master bias to '%s'\n", save_path);
            }
        }
    }

    /* --- Step 2: Load / generate master darkflat (if requested) --- */
    if (darkflat_path) {
        printf("calib: processing darkflat frames...\n");
        err = load_or_generate_one(darkflat_path, darkflat_method,
                                   /*subtract_buf=*/NULL, /*normalize=*/0,
                                   wsor_clip, &darkflat, &df_gen);
        if (err != DSO_OK) goto cleanup;
        printf("calib: master darkflat ready (%d×%d)\n",
               darkflat.width, darkflat.height);

        if (df_gen && save_dir) {
            ensure_dir(save_dir);
            char save_path[4096];
            snprintf(save_path, sizeof(save_path), "%s/master_darkflat.fits", save_dir);
            err = fits_save(save_path, &darkflat);
            if (err != DSO_OK) {
                fprintf(stderr, "calib: warning: failed to save master darkflat\n");
                err = DSO_OK;
            } else {
                printf("calib: saved master darkflat to '%s'\n", save_path);
            }
        }
    }

    /* --- Step 3: Load / generate master dark (if requested) ---
     * If bias is available, subtract it from each dark frame before stacking.
     * Standard procedure: dark_cal[i] = dark_raw[i] - bias_master */
    if (dark_path) {
        printf("calib: processing dark frames...\n");
        const float *sub = bias.data;   /* NULL if no bias loaded */
        err = load_or_generate_one(dark_path, dark_method,
                                   sub, /*normalize=*/0,
                                   wsor_clip, &calib_out->dark, &dark_gen);
        if (err != DSO_OK) goto cleanup;
        calib_out->has_dark = 1;
        printf("calib: master dark ready (%d×%d)\n",
               calib_out->dark.width, calib_out->dark.height);

        if (dark_gen && save_dir) {
            ensure_dir(save_dir);
            char save_path[4096];
            snprintf(save_path, sizeof(save_path), "%s/master_dark.fits", save_dir);
            err = fits_save(save_path, &calib_out->dark);
            if (err != DSO_OK) {
                fprintf(stderr, "calib: warning: failed to save master dark\n");
                err = DSO_OK;
            } else {
                printf("calib: saved master dark to '%s'\n", save_path);
            }
        }
    }

    /* --- Step 4: Load / generate master flat (if requested) ---
     * Each flat frame is bias- or darkflat-subtracted, then normalised by its
     * own mean so that all flats carry equal weight regardless of illumination.
     * The stacked master flat has mean ≈ 1.0. */
    if (flat_path) {
        printf("calib: processing flat frames...\n");
        /* Use darkflat if available, else bias, else no subtraction */
        const float *flat_sub = darkflat.data ? darkflat.data : bias.data;
        err = load_or_generate_one(flat_path, flat_method,
                                   flat_sub, /*normalize=*/1,
                                   wsor_clip, &calib_out->flat, &flat_gen);
        if (err != DSO_OK) goto cleanup;
        calib_out->has_flat = 1;
        printf("calib: master flat ready (%d×%d)\n",
               calib_out->flat.width, calib_out->flat.height);

        if (flat_gen && save_dir) {
            ensure_dir(save_dir);
            char save_path[4096];
            snprintf(save_path, sizeof(save_path), "%s/master_flat.fits", save_dir);
            err = fits_save(save_path, &calib_out->flat);
            if (err != DSO_OK) {
                fprintf(stderr, "calib: warning: failed to save master flat\n");
                err = DSO_OK;
            } else {
                printf("calib: saved master flat to '%s'\n", save_path);
            }
        }
    }

cleanup:
    image_free(&bias);
    image_free(&darkflat);
    if (err != DSO_OK) {
        /* Release any partially-built masters */
        image_free(&calib_out->dark);
        image_free(&calib_out->flat);
        calib_out->has_dark = 0;
        calib_out->has_flat = 0;
    }
    return err;
}

/* -------------------------------------------------------------------------
 * calib_apply_cpu
 * ------------------------------------------------------------------------- */
DsoError calib_apply_cpu(Image *img, const CalibFrames *calib)
{
    if (!img || !calib) return DSO_ERR_INVALID_ARG;
    if (!calib->has_dark && !calib->has_flat) return DSO_OK;

    /* Dimension checks */
    if (calib->has_dark &&
        (img->width != calib->dark.width || img->height != calib->dark.height)) {
        fprintf(stderr,
                "calib_apply_cpu: image %d×%d != dark master %d×%d\n",
                img->width, img->height,
                calib->dark.width, calib->dark.height);
        return DSO_ERR_INVALID_ARG;
    }
    if (calib->has_flat &&
        (img->width != calib->flat.width || img->height != calib->flat.height)) {
        fprintf(stderr,
                "calib_apply_cpu: image %d×%d != flat master %d×%d\n",
                img->width, img->height,
                calib->flat.width, calib->flat.height);
        return DSO_ERR_INVALID_ARG;
    }

    long npix = (long)img->width * img->height;
    float       *data = img->data;
    const float *dark = calib->has_dark ? calib->dark.data : NULL;
    const float *flat = calib->has_flat ? calib->flat.data : NULL;
#pragma omp parallel for schedule(static)
    for (long p = 0; p < npix; p++) {
        float v = data[p];
        if (dark) v -= dark[p];
        if (flat) {
            float f = flat[p];
            v = (f < 1e-6f) ? 0.0f : v / f;
        }
        data[p] = v;
    }

    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * calib_free
 * ------------------------------------------------------------------------- */
void calib_free(CalibFrames *calib)
{
    if (!calib) return;
    image_free(&calib->dark);
    image_free(&calib->flat);
    calib->has_dark = 0;
    calib->has_flat = 0;
}
