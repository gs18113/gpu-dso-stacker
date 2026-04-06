/*
 * white_balance.c — CPU white balance for raw Bayer mosaics.
 *
 * Applies per-channel multipliers to the raw mosaic before debayering.
 * OpenMP-parallelised where beneficial.
 */

#include "white_balance.h"
#include "compat.h"

#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
DsoError wb_apply_bayer(float *data, int W, int H,
                        BayerPattern pattern,
                        float r_mul, float g_mul, float b_mul)
{
    if (!data || W <= 0 || H <= 0)
        return DSO_ERR_INVALID_ARG;

    if (pattern == BAYER_NONE)
        return DSO_OK;   /* monochrome — nothing to do */

    const float mul[3] = { r_mul, g_mul, b_mul };

    int y;
    OMP_PARALLEL_FOR_COLLAPSE2
    for (y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int c = bayer_color(pattern, x, y);
            data[y * W + x] *= mul[c];
        }
    }
    return DSO_OK;
}

/* ------------------------------------------------------------------ */
DsoError wb_auto_compute(const float *data, int W, int H,
                         BayerPattern pattern,
                         float *r_mul, float *g_mul, float *b_mul)
{
    if (!data || W <= 0 || H <= 0 || !r_mul || !g_mul || !b_mul)
        return DSO_ERR_INVALID_ARG;

    if (pattern == BAYER_NONE)
        return DSO_ERR_INVALID_ARG;   /* gray-world needs Bayer data */

    /* Accumulate per-channel sums using double for precision.
     * Counts use double (not long) for MSVC OpenMP 2.0 reduction compat. */
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    double cnt_r = 0.0, cnt_g = 0.0, cnt_b = 0.0;
    int y;

    #pragma omp parallel for schedule(static) reduction(+:sum_r,sum_g,sum_b,cnt_r,cnt_g,cnt_b)
    for (y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int c = bayer_color(pattern, x, y);
            double v = (double)data[y * W + x];
            if (c == 0)      { sum_r += v; cnt_r += 1.0; }
            else if (c == 1) { sum_g += v; cnt_g += 1.0; }
            else             { sum_b += v; cnt_b += 1.0; }
        }
    }

    if (cnt_r < 1.0 || cnt_g < 1.0 || cnt_b < 1.0)
        return DSO_ERR_STAR_DETECT;   /* degenerate — no pixels for a channel */

    double mean_r = sum_r / cnt_r;
    double mean_g = sum_g / cnt_g;
    double mean_b = sum_b / cnt_b;

    if (mean_r < 1e-10 || mean_b < 1e-10) {
        fprintf(stderr, "wb_auto_compute: channel mean near zero "
                "(R=%.6g, G=%.6g, B=%.6g), cannot compute WB\n",
                mean_r, mean_g, mean_b);
        return DSO_ERR_STAR_DETECT;
    }

    *r_mul = (float)(mean_g / mean_r);
    *g_mul = 1.0f;
    *b_mul = (float)(mean_g / mean_b);

    /* Warn on unusual multipliers */
    if (*r_mul < 0.1f || *r_mul > 10.0f || *b_mul < 0.1f || *b_mul > 10.0f) {
        fprintf(stderr, "wb_auto_compute: unusual multipliers "
                "(R=%.3f, G=1.000, B=%.3f) — gray-world assumption "
                "may not suit this image\n", *r_mul, *b_mul);
    }

    return DSO_OK;
}
