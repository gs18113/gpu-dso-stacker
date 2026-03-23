/*
 * test_color.c — Exhaustive unit tests for OSC (one-shot color) output.
 *
 * Tests cover:
 *
 *   debayer_cpu_rgb:
 *     - Argument validation: null src/r/g/b, zero width/height
 *     - BAYER_NONE passthrough: all three planes are exact copies of src
 *     - Uniform mosaic (all four patterns): interior R = G = B = constant
 *     - Pure-channel dominance (RGGB R/G/B; BGGR/GRBG/GBRG R):
 *         set only one channel's Bayer pixels to 0.8, rest to 0;
 *         the corresponding output channel must be > 0.7 and the other
 *         two must be < 0.1 at a known interior pixel
 *     - Luminance consistency: 0.2126·R + 0.7152·G + 0.0722·B output of
 *         debayer_cpu_rgb must exactly match debayer_cpu for the same mosaic
 *     - Channel distinctness: non-uniform mosaic produces distinct R, G, B
 *     - Non-negative output: all pixel values ≥ 0 after VNG interpolation
 *
 *   fits_save_rgb:
 *     - Argument validation: null filepath, null plane pointer, dim mismatch
 *     - NAXIS = 3 and NAXIS3 = 3 in the written file
 *     - Per-plane round-trip: R, G, B values are recovered correctly
 *
 *   fits_get_bayer_pattern:
 *     - RGGB / BGGR / GRBG / GBRG keywords correctly decoded
 *     - Absent BAYERPAT keyword returns BAYER_NONE
 */

#include "test_framework.h"
#include "debayer_cpu.h"
#include "fits_io.h"
#include <fitsio.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

static float *alloc_float(int n, float fill)
{
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) return NULL;
    for (int i = 0; i < n; i++) p[i] = fill;
    return p;
}

/*
 * channel_at — return the Bayer channel index (0=R, 1=G, 2=B) of pixel (x,y).
 * Mirrors the bayer_channel() helper in debayer_cpu.c so tests can predict
 * which pixel positions belong to which colour channel.
 */
static int channel_at(int x, int y, BayerPattern pattern)
{
    int px = x & 1, py = y & 1;
    switch ((int)pattern) {
    case 1: /* RGGB */ return (py==0) ? (px==0 ? 0 : 1) : (px==0 ? 1 : 2);
    case 2: /* BGGR */ return (py==0) ? (px==0 ? 2 : 1) : (px==0 ? 1 : 0);
    case 3: /* GRBG */ return (py==0) ? (px==0 ? 1 : 0) : (px==0 ? 2 : 1);
    case 4: /* GBRG */ return (py==0) ? (px==0 ? 1 : 2) : (px==0 ? 0 : 1);
    default: return 1;
    }
}

/*
 * fill_channel — set every pixel whose Bayer position equals `target_chan`
 * to `val`; set every other pixel to 0.  Produces a pure single-channel
 * Bayer mosaic that drives the channel-dominance tests.
 */
static void fill_channel(float *mosaic, int W, int H,
                          BayerPattern pattern, int target_chan, float val)
{
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            mosaic[y * W + x] =
                (channel_at(x, y, pattern) == target_chan) ? val : 0.f;
}

/*
 * fill_pattern — fill a Bayer mosaic with distinct per-channel constants:
 *   channel 0 (R) = r_val, channel 1 (G) = g_val, channel 2 (B) = b_val.
 */
static void fill_pattern(float *mosaic, int W, int H, BayerPattern pattern,
                          float r_val, float g_val, float b_val)
{
    float vals[3] = {r_val, g_val, b_val};
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            mosaic[y * W + x] = vals[channel_at(x, y, pattern)];
}

/*
 * fits_check_axes — open a FITS file and return the number of image
 * dimensions and the size of each axis (up to 3).
 * Returns 0 on success, -1 on error.
 */
static int fits_check_axes(const char *path, int *ndim_out, long naxes_out[3])
{
    fitsfile *fptr = NULL;
    int status = 0;
    if (ffopen(&fptr, path, READONLY, &status)) return -1;

    int ndim = 0;
    ffgidm(fptr, &ndim, &status);

    long naxes[3] = {0, 0, 0};
    if (ndim >= 1 && ndim <= 3)
        ffgisz(fptr, ndim, naxes, &status);

    ffclos(fptr, &status);
    if (status) return -1;

    if (ndim_out)  *ndim_out = ndim;
    if (naxes_out) {
        naxes_out[0] = naxes[0];
        naxes_out[1] = naxes[1];
        naxes_out[2] = naxes[2];
    }
    return 0;
}

/*
 * fits_load_plane — read one plane (1-indexed) from a 3-D FITS file into
 * a caller-supplied buffer of W×H floats.
 */
static DsoError fits_load_plane(const char *path, int plane,
                                 float *buf, int W, int H)
{
    fitsfile *fptr = NULL;
    int status = 0;
    if (ffopen(&fptr, path, READONLY, &status)) return DSO_ERR_FITS;

    float nulval = 0.f;
    int anynul = 0;
    long firstpix[3] = {1, 1, (long)plane};
    LONGLONG nelem = (LONGLONG)W * H;
    ffgpxv(fptr, TFLOAT, firstpix, nelem, &nulval, buf, &anynul, &status);
    ffclos(fptr, &status);
    return status ? DSO_ERR_FITS : DSO_OK;
}

/*
 * make_fits_bayerpat — create a minimal W×H FITS file (all pixels = val)
 * and, when bayerpat is non-NULL, write a BAYERPAT string keyword.
 */
static DsoError make_fits_bayerpat(const char *path, int W, int H,
                                    float val, const char *bayerpat)
{
    char ovr[4097];
    ovr[0] = '!';
    strncpy(ovr + 1, path, sizeof(ovr) - 2);
    ovr[sizeof(ovr) - 1] = '\0';

    fitsfile *fptr = NULL;
    int status = 0;
    long naxes[2] = {W, H};
    ffinit(&fptr, ovr, &status);
    ffcrim(fptr, FLOAT_IMG, 2, naxes, &status);
    if (status) {
        if (fptr) { int s2 = 0; ffclos(fptr, &s2); }
        return DSO_ERR_FITS;
    }

    float *data = (float *)malloc((size_t)W * H * sizeof(float));
    if (!data) { ffclos(fptr, &status); return DSO_ERR_ALLOC; }
    for (int i = 0; i < W * H; i++) data[i] = val;

    LONGLONG nelem = (LONGLONG)W * H;
    long firstpix[2] = {1, 1};
    ffppx(fptr, TFLOAT, firstpix, nelem, data, &status);
    free(data);

    if (bayerpat)
        ffpkys(fptr, "BAYERPAT", bayerpat, "Bayer color filter pattern", &status);

    ffclos(fptr, &status);
    return DSO_OK;
}

/* =========================================================================
 * debayer_cpu_rgb — argument validation
 * ========================================================================= */

static int test_rgb_null_src(void)
{
    float buf[4] = {0};
    ASSERT_ERR(debayer_cpu_rgb(NULL, buf, buf, buf, 2, 2, BAYER_RGGB),
               DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_rgb_null_r(void)
{
    float src[4] = {0}, buf[4] = {0};
    ASSERT_ERR(debayer_cpu_rgb(src, NULL, buf, buf, 2, 2, BAYER_RGGB),
               DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_rgb_null_g(void)
{
    float src[4] = {0}, buf[4] = {0};
    ASSERT_ERR(debayer_cpu_rgb(src, buf, NULL, buf, 2, 2, BAYER_RGGB),
               DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_rgb_null_b(void)
{
    float src[4] = {0}, buf[4] = {0};
    ASSERT_ERR(debayer_cpu_rgb(src, buf, buf, NULL, 2, 2, BAYER_RGGB),
               DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_rgb_zero_width(void)
{
    float buf[4] = {0};
    ASSERT_ERR(debayer_cpu_rgb(buf, buf, buf, buf, 0, 4, BAYER_RGGB),
               DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_rgb_zero_height(void)
{
    float buf[4] = {0};
    ASSERT_ERR(debayer_cpu_rgb(buf, buf, buf, buf, 4, 0, BAYER_RGGB),
               DSO_ERR_INVALID_ARG);
    return 0;
}

/* =========================================================================
 * debayer_cpu_rgb — BAYER_NONE passthrough
 * ========================================================================= */

static int test_rgb_none_passthrough(void)
{
    int W = 8, H = 8;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    for (int i = 0; i < W * H; i++) src[i] = (float)i * 0.01f;

    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_NONE));

    for (int i = 0; i < W * H; i++) {
        ASSERT_NEAR(r[i], src[i], 1e-6f);
        ASSERT_NEAR(g[i], src[i], 1e-6f);
        ASSERT_NEAR(b[i], src[i], 1e-6f);
    }

    free(src); free(r); free(g); free(b);
    return 0;
}

/* =========================================================================
 * debayer_cpu_rgb — uniform mosaics (all four patterns)
 *
 * When every mosaic pixel equals the same constant, each channel estimate
 * also equals that constant, so R = G = B = val in the image interior.
 * ========================================================================= */

static int test_rgb_uniform_rggb(void)
{
    int W = 16, H = 16;
    float val = 0.7f;
    float *src = alloc_float(W * H, val);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_RGGB));

    for (int y = 2; y < H - 2; y++) {
        for (int x = 2; x < W - 2; x++) {
            ASSERT_NEAR(r[y*W+x], val, 1e-4f);
            ASSERT_NEAR(g[y*W+x], val, 1e-4f);
            ASSERT_NEAR(b[y*W+x], val, 1e-4f);
        }
    }

    free(src); free(r); free(g); free(b);
    return 0;
}

static int test_rgb_uniform_bggr(void)
{
    int W = 16, H = 16;
    float val = 0.5f;
    float *src = alloc_float(W * H, val);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_BGGR));

    for (int y = 2; y < H - 2; y++) {
        for (int x = 2; x < W - 2; x++) {
            ASSERT_NEAR(r[y*W+x], val, 1e-4f);
            ASSERT_NEAR(g[y*W+x], val, 1e-4f);
            ASSERT_NEAR(b[y*W+x], val, 1e-4f);
        }
    }

    free(src); free(r); free(g); free(b);
    return 0;
}

static int test_rgb_uniform_grbg(void)
{
    int W = 16, H = 16;
    float val = 0.3f;
    float *src = alloc_float(W * H, val);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_GRBG));

    for (int y = 2; y < H - 2; y++) {
        for (int x = 2; x < W - 2; x++) {
            ASSERT_NEAR(r[y*W+x], val, 1e-4f);
            ASSERT_NEAR(g[y*W+x], val, 1e-4f);
            ASSERT_NEAR(b[y*W+x], val, 1e-4f);
        }
    }

    free(src); free(r); free(g); free(b);
    return 0;
}

static int test_rgb_uniform_gbrg(void)
{
    int W = 16, H = 16;
    float val = 0.6f;
    float *src = alloc_float(W * H, val);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_GBRG));

    for (int y = 2; y < H - 2; y++) {
        for (int x = 2; x < W - 2; x++) {
            ASSERT_NEAR(r[y*W+x], val, 1e-4f);
            ASSERT_NEAR(g[y*W+x], val, 1e-4f);
            ASSERT_NEAR(b[y*W+x], val, 1e-4f);
        }
    }

    free(src); free(r); free(g); free(b);
    return 0;
}

/* =========================================================================
 * debayer_cpu_rgb — pure-channel dominance (RGGB)
 *
 * RGGB layout on a 16×16 image:
 *   R at even_x, even_y  — e.g. pixel (4,4)
 *   G at odd_x,  even_y  — e.g. pixel (5,4)
 *   G at even_x, odd_y   — e.g. pixel (4,5)
 *   B at odd_x,  odd_y   — e.g. pixel (5,5)
 *
 * When only one channel's Bayer pixels carry a non-zero value, VNG uses
 * only those pixels (via gradient-directed selection) so the corresponding
 * output channel is close to that value and the others are close to zero.
 * ========================================================================= */

static int test_rgb_rggb_r_dominates(void)
{
    /* R at (4,4): py=0, px=0 → R.  With all G/B = 0, R_est ≈ 0.8. */
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    fill_channel(src, W, H, BAYER_RGGB, 0 /* R */, 0.8f);
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_RGGB));

    int px = 4 * W + 4; /* interior R pixel */
    ASSERT_GT(r[px], 0.7f);
    ASSERT_LT(g[px], 0.1f);
    ASSERT_LT(b[px], 0.1f);

    free(src); free(r); free(g); free(b);
    return 0;
}

static int test_rgb_rggb_g_dominates(void)
{
    /* G at (5,4): py=0, px=1 → G.  With all R/B = 0, G_est ≈ 0.8. */
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    fill_channel(src, W, H, BAYER_RGGB, 1 /* G */, 0.8f);
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_RGGB));

    int px = 4 * W + 5; /* interior G pixel (odd_x, even_y) */
    ASSERT_GT(g[px], 0.7f);
    ASSERT_LT(r[px], 0.1f);
    ASSERT_LT(b[px], 0.1f);

    free(src); free(r); free(g); free(b);
    return 0;
}

static int test_rgb_rggb_b_dominates(void)
{
    /* B at (5,5): py=1, px=1 → B.  With all R/G = 0, B_est ≈ 0.8. */
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    fill_channel(src, W, H, BAYER_RGGB, 2 /* B */, 0.8f);
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_RGGB));

    int px = 5 * W + 5; /* interior B pixel (odd_x, odd_y) */
    ASSERT_GT(b[px], 0.7f);
    ASSERT_LT(r[px], 0.1f);
    ASSERT_LT(g[px], 0.1f);

    free(src); free(r); free(g); free(b);
    return 0;
}

/* =========================================================================
 * debayer_cpu_rgb — pure-R dominance for BGGR, GRBG, GBRG
 *
 * BGGR: R at (odd_x, odd_y) — check (5,5)
 * GRBG: R at (odd_x, even_y) — check (5,4)
 * GBRG: R at (even_x, odd_y) — check (4,5)
 * ========================================================================= */

static int test_rgb_bggr_r_dominates(void)
{
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    fill_channel(src, W, H, BAYER_BGGR, 0 /* R */, 0.8f);
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_BGGR));

    int px = 5 * W + 5; /* BGGR R pixel: py=1,px=1 → R */
    ASSERT_GT(r[px], 0.7f);
    ASSERT_LT(g[px], 0.1f);
    ASSERT_LT(b[px], 0.1f);

    free(src); free(r); free(g); free(b);
    return 0;
}

static int test_rgb_grbg_r_dominates(void)
{
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    fill_channel(src, W, H, BAYER_GRBG, 0 /* R */, 0.8f);
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_GRBG));

    int px = 4 * W + 5; /* GRBG R pixel: py=0,px=1 → R */
    ASSERT_GT(r[px], 0.7f);
    ASSERT_LT(g[px], 0.1f);
    ASSERT_LT(b[px], 0.1f);

    free(src); free(r); free(g); free(b);
    return 0;
}

static int test_rgb_gbrg_r_dominates(void)
{
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    fill_channel(src, W, H, BAYER_GBRG, 0 /* R */, 0.8f);
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_GBRG));

    int px = 5 * W + 4; /* GBRG R pixel: py=1,px=0 → R */
    ASSERT_GT(r[px], 0.7f);
    ASSERT_LT(g[px], 0.1f);
    ASSERT_LT(b[px], 0.1f);

    free(src); free(r); free(g); free(b);
    return 0;
}

/* =========================================================================
 * debayer_cpu_rgb — luminance consistency
 *
 * debayer_cpu_rgb and debayer_cpu implement the same VNG computation.
 * For the same input mosaic their outputs must satisfy:
 *   lum[px] == 0.2126·r[px] + 0.7152·g[px] + 0.0722·b[px]
 * (within single-precision rounding) at every interior pixel.
 * ========================================================================= */

static int test_rgb_luminance_consistency(void)
{
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *lum = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(lum);
    ASSERT_NOT_NULL(r);   ASSERT_NOT_NULL(g); ASSERT_NOT_NULL(b);

    /* Non-trivial mosaic: distinct constant per channel so all estimates
     * are non-zero and meaningful. */
    fill_pattern(src, W, H, BAYER_RGGB, 0.8f, 0.5f, 0.3f);

    ASSERT_OK(debayer_cpu(src, lum, W, H, BAYER_RGGB));
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_RGGB));

    for (int y = 2; y < H - 2; y++) {
        for (int x = 2; x < W - 2; x++) {
            int px = y * W + x;
            float recon = 0.2126f * r[px] + 0.7152f * g[px] + 0.0722f * b[px];
            ASSERT_NEAR(lum[px], recon, 1e-4f);
        }
    }

    free(src); free(lum); free(r); free(g); free(b);
    return 0;
}

/* =========================================================================
 * debayer_cpu_rgb — channel distinctness
 *
 * With a non-uniform Bayer mosaic (different values per channel position),
 * the R, G, B output planes must not all be identical to each other at
 * interior pixels.
 * ========================================================================= */

static int test_rgb_channels_distinct(void)
{
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    fill_pattern(src, W, H, BAYER_RGGB, 0.8f, 0.5f, 0.2f);
    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_RGGB));

    /* Count interior pixels where R ≠ G or R ≠ B */
    int distinct = 0;
    for (int y = 2; y < H - 2; y++) {
        for (int x = 2; x < W - 2; x++) {
            int px = y * W + x;
            if (fabsf(r[px] - g[px]) > 1e-3f || fabsf(r[px] - b[px]) > 1e-3f)
                distinct++;
        }
    }

    /* Virtually all interior pixels must show channel separation */
    ASSERT_GT(distinct, 100);

    free(src); free(r); free(g); free(b);
    return 0;
}

/* =========================================================================
 * debayer_cpu_rgb — non-negative output
 *
 * VNG interpolation must never produce negative pixel values.
 * ========================================================================= */

static int test_rgb_nonnegative(void)
{
    int W = 16, H = 16;
    float *src = alloc_float(W * H, 0.f);
    float *r   = alloc_float(W * H, 0.f);
    float *g   = alloc_float(W * H, 0.f);
    float *b   = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(src); ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(g);   ASSERT_NOT_NULL(b);

    /* Alternating high/low to stress the gradient logic */
    for (int i = 0; i < W * H; i++)
        src[i] = (i & 1) ? 0.9f : 0.1f;

    ASSERT_OK(debayer_cpu_rgb(src, r, g, b, W, H, BAYER_RGGB));

    for (int i = 0; i < W * H; i++) {
        ASSERT_GT(r[i], -1e-6f);
        ASSERT_GT(g[i], -1e-6f);
        ASSERT_GT(b[i], -1e-6f);
    }

    free(src); free(r); free(g); free(b);
    return 0;
}

/* =========================================================================
 * fits_save_rgb — argument validation
 * ========================================================================= */

static int test_save_rgb_null_filepath(void)
{
    float data[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    Image img = {data, 2, 2};
    ASSERT_ERR(fits_save_rgb(NULL, &img, &img, &img), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_save_rgb_null_plane(void)
{
    char path[512];
    TEST_TMPPATH(path, "tc_null_plane.fits");
    float data[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    Image img = {data, 2, 2};
    ASSERT_ERR(fits_save_rgb(path, NULL, &img, &img),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(fits_save_rgb(path, &img, NULL, &img),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(fits_save_rgb(path, &img, &img, NULL),
               DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_save_rgb_dim_mismatch(void)
{
    char path[512];
    TEST_TMPPATH(path, "tc_dim_mismatch.fits");
    float *dr = alloc_float(4, 0.1f);
    float *dg = alloc_float(4, 0.2f);
    float *db = alloc_float(9, 0.3f); /* different size */
    ASSERT_NOT_NULL(dr); ASSERT_NOT_NULL(dg); ASSERT_NOT_NULL(db);

    Image r = {dr, 2, 2};
    Image g = {dg, 2, 2};
    Image b = {db, 3, 3}; /* mismatched dimensions */

    ASSERT_ERR(fits_save_rgb(path, &r, &g, &b),
               DSO_ERR_INVALID_ARG);

    free(dr); free(dg); free(db);
    return 0;
}

/* =========================================================================
 * fits_save_rgb — round-trip: NAXIS = 3, NAXIS3 = 3
 * ========================================================================= */

static int test_save_rgb_naxis3(void)
{
    int W = 8, H = 8;
    float *dr = alloc_float(W * H, 0.1f);
    float *dg = alloc_float(W * H, 0.2f);
    float *db = alloc_float(W * H, 0.3f);
    ASSERT_NOT_NULL(dr); ASSERT_NOT_NULL(dg); ASSERT_NOT_NULL(db);

    Image r = {dr, W, H}, g = {dg, W, H}, b = {db, W, H};
    char path[512];
    TEST_TMPPATH(path, "tc_rgb_naxis3.fits");
    ASSERT_OK(fits_save_rgb(path, &r, &g, &b));

    int ndim = 0;
    long naxes[3] = {0, 0, 0};
    ASSERT_EQ(fits_check_axes(path, &ndim, naxes), 0);
    ASSERT_EQ(ndim, 3);
    ASSERT_EQ(naxes[0], (long)W);
    ASSERT_EQ(naxes[1], (long)H);
    ASSERT_EQ(naxes[2], 3L);

    free(dr); free(dg); free(db);
    return 0;
}

/* =========================================================================
 * fits_save_rgb — round-trip: per-plane pixel values
 * ========================================================================= */

static int test_save_rgb_r_plane(void)
{
    int W = 8, H = 8;
    float *dr = alloc_float(W * H, 0.1f);
    float *dg = alloc_float(W * H, 0.2f);
    float *db = alloc_float(W * H, 0.3f);
    float *back = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(dr); ASSERT_NOT_NULL(dg);
    ASSERT_NOT_NULL(db); ASSERT_NOT_NULL(back);

    Image r = {dr, W, H}, g = {dg, W, H}, b = {db, W, H};
    char path[512];
    TEST_TMPPATH(path, "tc_rgb_planes.fits");
    ASSERT_OK(fits_save_rgb(path, &r, &g, &b));
    ASSERT_OK(fits_load_plane(path, 1, back, W, H));

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(back[i], 0.1f, 1e-5f);

    free(dr); free(dg); free(db); free(back);
    return 0;
}

static int test_save_rgb_g_plane(void)
{
    int W = 8, H = 8;
    float *dr = alloc_float(W * H, 0.1f);
    float *dg = alloc_float(W * H, 0.2f);
    float *db = alloc_float(W * H, 0.3f);
    float *back = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(dr); ASSERT_NOT_NULL(dg);
    ASSERT_NOT_NULL(db); ASSERT_NOT_NULL(back);

    Image r = {dr, W, H}, g = {dg, W, H}, b = {db, W, H};
    char path[512];
    TEST_TMPPATH(path, "tc_rgb_planes.fits");
    ASSERT_OK(fits_save_rgb(path, &r, &g, &b));
    ASSERT_OK(fits_load_plane(path, 2, back, W, H));

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(back[i], 0.2f, 1e-5f);

    free(dr); free(dg); free(db); free(back);
    return 0;
}

static int test_save_rgb_b_plane(void)
{
    int W = 8, H = 8;
    float *dr = alloc_float(W * H, 0.1f);
    float *dg = alloc_float(W * H, 0.2f);
    float *db = alloc_float(W * H, 0.3f);
    float *back = alloc_float(W * H, 0.f);
    ASSERT_NOT_NULL(dr); ASSERT_NOT_NULL(dg);
    ASSERT_NOT_NULL(db); ASSERT_NOT_NULL(back);

    Image r = {dr, W, H}, g = {dg, W, H}, b = {db, W, H};
    char path[512];
    TEST_TMPPATH(path, "tc_rgb_planes.fits");
    ASSERT_OK(fits_save_rgb(path, &r, &g, &b));
    ASSERT_OK(fits_load_plane(path, 3, back, W, H));

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(back[i], 0.3f, 1e-5f);

    free(dr); free(dg); free(db); free(back);
    return 0;
}

/*
 * Round-trip with non-constant planes: verify pixel-by-pixel recovery.
 * R[i] = i/npix, G[i] = 1 - i/npix, B[i] = 0.5.
 */
static int test_save_rgb_gradient_planes(void)
{
    int W = 12, H = 10;
    int npix = W * H;
    float *dr = alloc_float(npix, 0.f);
    float *dg = alloc_float(npix, 0.f);
    float *db = alloc_float(npix, 0.5f);
    float *back = alloc_float(npix, 0.f);
    ASSERT_NOT_NULL(dr); ASSERT_NOT_NULL(dg);
    ASSERT_NOT_NULL(db); ASSERT_NOT_NULL(back);

    for (int i = 0; i < npix; i++) {
        dr[i] = (float)i / (float)npix;
        dg[i] = 1.f - dr[i];
    }

    Image r = {dr, W, H}, g = {dg, W, H}, b = {db, W, H};
    char path[512];
    TEST_TMPPATH(path, "tc_rgb_gradient.fits");
    ASSERT_OK(fits_save_rgb(path, &r, &g, &b));

    /* Check R plane */
    ASSERT_OK(fits_load_plane(path, 1, back, W, H));
    for (int i = 0; i < npix; i++)
        ASSERT_NEAR(back[i], dr[i], 1e-5f);

    /* Check G plane */
    ASSERT_OK(fits_load_plane(path, 2, back, W, H));
    for (int i = 0; i < npix; i++)
        ASSERT_NEAR(back[i], dg[i], 1e-5f);

    /* Check B plane */
    ASSERT_OK(fits_load_plane(path, 3, back, W, H));
    for (int i = 0; i < npix; i++)
        ASSERT_NEAR(back[i], 0.5f, 1e-5f);

    free(dr); free(dg); free(db); free(back);
    return 0;
}

/* =========================================================================
 * fits_get_bayer_pattern — BAYERPAT keyword detection
 * ========================================================================= */

static int test_bayerpat_rggb(void)
{
    char path[512];
    TEST_TMPPATH(path, "tc_bayerpat_rggb.fits");
    ASSERT_OK(make_fits_bayerpat(path, 4, 4, 0.5f, "RGGB"));
    BayerPattern pat = BAYER_NONE;
    ASSERT_OK(fits_get_bayer_pattern(path, &pat));
    ASSERT_EQ(pat, BAYER_RGGB);
    return 0;
}

static int test_bayerpat_bggr(void)
{
    char path[512];
    TEST_TMPPATH(path, "tc_bayerpat_bggr.fits");
    ASSERT_OK(make_fits_bayerpat(path, 4, 4, 0.5f, "BGGR"));
    BayerPattern pat = BAYER_NONE;
    ASSERT_OK(fits_get_bayer_pattern(path, &pat));
    ASSERT_EQ(pat, BAYER_BGGR);
    return 0;
}

static int test_bayerpat_grbg(void)
{
    char path[512];
    TEST_TMPPATH(path, "tc_bayerpat_grbg.fits");
    ASSERT_OK(make_fits_bayerpat(path, 4, 4, 0.5f, "GRBG"));
    BayerPattern pat = BAYER_NONE;
    ASSERT_OK(fits_get_bayer_pattern(path, &pat));
    ASSERT_EQ(pat, BAYER_GRBG);
    return 0;
}

static int test_bayerpat_gbrg(void)
{
    char path[512];
    TEST_TMPPATH(path, "tc_bayerpat_gbrg.fits");
    ASSERT_OK(make_fits_bayerpat(path, 4, 4, 0.5f, "GBRG"));
    BayerPattern pat = BAYER_NONE;
    ASSERT_OK(fits_get_bayer_pattern(path, &pat));
    ASSERT_EQ(pat, BAYER_GBRG);
    return 0;
}

static int test_bayerpat_absent(void)
{
    char path[512];
    TEST_TMPPATH(path, "tc_bayerpat_absent.fits");
    ASSERT_OK(make_fits_bayerpat(path, 4, 4, 0.5f, NULL /* no keyword */));
    BayerPattern pat = BAYER_RGGB; /* pre-set to non-NONE to detect no-write */
    ASSERT_OK(fits_get_bayer_pattern(path, &pat));
    ASSERT_EQ(pat, BAYER_NONE);
    return 0;
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    SUITE("debayer_cpu_rgb — argument validation");
    RUN(test_rgb_null_src);
    RUN(test_rgb_null_r);
    RUN(test_rgb_null_g);
    RUN(test_rgb_null_b);
    RUN(test_rgb_zero_width);
    RUN(test_rgb_zero_height);

    SUITE("debayer_cpu_rgb — BAYER_NONE passthrough");
    RUN(test_rgb_none_passthrough);

    SUITE("debayer_cpu_rgb — uniform mosaic (all four patterns)");
    RUN(test_rgb_uniform_rggb);
    RUN(test_rgb_uniform_bggr);
    RUN(test_rgb_uniform_grbg);
    RUN(test_rgb_uniform_gbrg);

    SUITE("debayer_cpu_rgb — pure-channel dominance (RGGB)");
    RUN(test_rgb_rggb_r_dominates);
    RUN(test_rgb_rggb_g_dominates);
    RUN(test_rgb_rggb_b_dominates);

    SUITE("debayer_cpu_rgb — pure-R dominance (BGGR, GRBG, GBRG)");
    RUN(test_rgb_bggr_r_dominates);
    RUN(test_rgb_grbg_r_dominates);
    RUN(test_rgb_gbrg_r_dominates);

    SUITE("debayer_cpu_rgb — luminance consistency");
    RUN(test_rgb_luminance_consistency);

    SUITE("debayer_cpu_rgb — channel distinctness and output quality");
    RUN(test_rgb_channels_distinct);
    RUN(test_rgb_nonnegative);

    SUITE("fits_save_rgb — argument validation");
    RUN(test_save_rgb_null_filepath);
    RUN(test_save_rgb_null_plane);
    RUN(test_save_rgb_dim_mismatch);

    SUITE("fits_save_rgb — NAXIS=3 structure");
    RUN(test_save_rgb_naxis3);

    SUITE("fits_save_rgb — per-plane round-trip");
    RUN(test_save_rgb_r_plane);
    RUN(test_save_rgb_g_plane);
    RUN(test_save_rgb_b_plane);
    RUN(test_save_rgb_gradient_planes);

    SUITE("fits_get_bayer_pattern — BAYERPAT keyword");
    RUN(test_bayerpat_rggb);
    RUN(test_bayerpat_bggr);
    RUN(test_bayerpat_grbg);
    RUN(test_bayerpat_gbrg);
    RUN(test_bayerpat_absent);

    return SUMMARY();
}
