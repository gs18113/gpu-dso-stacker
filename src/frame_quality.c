/*
 * frame_quality.c — Per-frame quality scoring from star detection data.
 *
 * FWHM is estimated from the Moffat convolution map (matched-filter response),
 * not the raw image. This means absolute FWHM values include the kernel width,
 * but for relative quality comparison (which is all we need) this is ideal —
 * the convolution map suppresses noise and provides a clean radial profile.
 */

#include "frame_quality.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------- */

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

/* Return median of arr[0..n-1]. Modifies arr in-place (partial sort). */
static float median_float(float *arr, int n)
{
    if (n <= 0) return 0.0f;
    if (n == 1) return arr[0];
    qsort(arr, (size_t)n, sizeof(float), cmp_float);
    if (n % 2 == 1) return arr[n / 2];
    return 0.5f * (arr[n / 2 - 1] + arr[n / 2]);
}

/*
 * Walk outward from (cx, cy) in direction (dx, dy) on the convolution map
 * until the value drops below half_max. Return the interpolated distance
 * from (cx, cy) to the half-max crossing, or -1 if not found within bounds.
 *
 * max_steps limits the walk to avoid running off the image.
 */
static float walk_to_half_max(const float *conv, int W, int H,
                              float cx, float cy,
                              int dx, int dy,
                              float half_max, int max_steps)
{
    float prev_val = half_max * 2.0f; /* start above half_max */
    int ix = (int)(cx + 0.5f);
    int iy = (int)(cy + 0.5f);

    /* Value at center */
    if (ix >= 0 && ix < W && iy >= 0 && iy < H)
        prev_val = conv[iy * W + ix];

    for (int step = 1; step <= max_steps; step++) {
        int nx = ix + dx * step;
        int ny = iy + dy * step;
        if (nx < 0 || nx >= W || ny < 0 || ny >= H)
            return -1.0f;

        float cur_val = conv[ny * W + nx];
        if (cur_val <= half_max) {
            /* Linear interpolation between step-1 and step */
            float denom = prev_val - cur_val;
            if (denom < 1e-10f) return (float)step;
            float frac = (prev_val - half_max) / denom;
            return (float)(step - 1) + frac;
        }
        prev_val = cur_val;
    }
    return -1.0f; /* didn't drop below half_max within max_steps */
}

/*
 * Measure FWHM for a single star at (cx, cy) from the convolution map.
 * Also sets *fwhm_x and *fwhm_y for the horizontal and vertical directions.
 * Returns the overall FWHM (mean of all valid directions × 2).
 * Returns -1 if measurement fails.
 */
static float measure_star_fwhm(const float *conv, int W, int H,
                               float cx, float cy,
                               float *fwhm_x, float *fwhm_y)
{
    int ix = (int)(cx + 0.5f);
    int iy = (int)(cy + 0.5f);
    if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
        *fwhm_x = -1.0f;
        *fwhm_y = -1.0f;
        return -1.0f;
    }

    float peak = conv[iy * W + ix];
    if (peak <= 0.0f) {
        *fwhm_x = -1.0f;
        *fwhm_y = -1.0f;
        return -1.0f;
    }

    float half_max = peak * 0.5f;
    int max_steps = 20;

    /* Walk in 4 cardinal directions: +x, -x, +y, -y */
    float r_px = walk_to_half_max(conv, W, H, cx, cy,  1,  0, half_max, max_steps);
    float r_mx = walk_to_half_max(conv, W, H, cx, cy, -1,  0, half_max, max_steps);
    float r_py = walk_to_half_max(conv, W, H, cx, cy,  0,  1, half_max, max_steps);
    float r_my = walk_to_half_max(conv, W, H, cx, cy,  0, -1, half_max, max_steps);

    /* FWHM_x from horizontal walks */
    float sum_x = 0.0f;
    int   cnt_x = 0;
    if (r_px > 0.0f) { sum_x += r_px; cnt_x++; }
    if (r_mx > 0.0f) { sum_x += r_mx; cnt_x++; }

    /* FWHM_y from vertical walks */
    float sum_y = 0.0f;
    int   cnt_y = 0;
    if (r_py > 0.0f) { sum_y += r_py; cnt_y++; }
    if (r_my > 0.0f) { sum_y += r_my; cnt_y++; }

    if (cnt_x == 0 && cnt_y == 0) {
        *fwhm_x = -1.0f;
        *fwhm_y = -1.0f;
        return -1.0f;
    }

    /* If one direction is missing, use the other as fallback */
    float fx = (cnt_x > 0) ? (2.0f * sum_x / cnt_x) : -1.0f;
    float fy = (cnt_y > 0) ? (2.0f * sum_y / cnt_y) : -1.0f;

    *fwhm_x = fx;
    *fwhm_y = fy;

    /* Overall FWHM: mean of all valid radii × 2 */
    float total = 0.0f;
    int   total_cnt = 0;
    if (r_px > 0.0f) { total += r_px; total_cnt++; }
    if (r_mx > 0.0f) { total += r_mx; total_cnt++; }
    if (r_py > 0.0f) { total += r_py; total_cnt++; }
    if (r_my > 0.0f) { total += r_my; total_cnt++; }

    return (total_cnt > 0) ? (2.0f * total / total_cnt) : -1.0f;
}

/*
 * Sigma-clipped mean of an image buffer.
 * sigma_k = clipping threshold in sigma units, iters = clipping passes.
 * Uses double precision accumulation. Skips NaN values.
 */
static float sigma_clipped_mean(const float *data, long n,
                                float sigma_k, int iters)
{
    /* First pass: compute initial mean and stddev */
    double sum = 0.0;
    long   count = 0;
    long   i;

    for (i = 0; i < n; i++) {
        float v = data[i];
        if (v != v) continue;  /* skip NaN */
        sum += (double)v;
        count++;
    }
    if (count == 0) return 0.0f;

    double mean = sum / count;

    for (int iter = 0; iter < iters; iter++) {
        /* Compute stddev */
        double sq_sum = 0.0;
        long   sq_count = 0;
        for (i = 0; i < n; i++) {
            float v = data[i];
            if (v != v) continue;
            double d = (double)v - mean;
            sq_sum += d * d;
            sq_count++;
        }
        if (sq_count < 2) break;
        double sigma = sqrt(sq_sum / (sq_count - 1));
        if (sigma < 1e-10) break;

        double lo = mean - sigma_k * sigma;
        double hi = mean + sigma_k * sigma;

        /* Recompute mean with clipping */
        double new_sum = 0.0;
        long   new_count = 0;
        for (i = 0; i < n; i++) {
            float v = data[i];
            if (v != v) continue;
            if ((double)v >= lo && (double)v <= hi) {
                new_sum += (double)v;
                new_count++;
            }
        }
        if (new_count == 0) break;
        mean = new_sum / new_count;
    }

    return (float)mean;
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

DsoError frame_quality_compute(const float    *conv_data,
                               const float    *lum_data,
                               const StarList *stars,
                               int W, int H,
                               FrameQuality   *quality_out)
{
    if (!conv_data || !lum_data || !stars || !quality_out || W <= 0 || H <= 0)
        return DSO_ERR_INVALID_ARG;

    memset(quality_out, 0, sizeof(FrameQuality));
    quality_out->star_count = stars->n;

    /* --- Background --- */
    quality_out->background = sigma_clipped_mean(lum_data, (long)W * H, 3.0f, 3);

    /* --- FWHM and roundness --- */
    if (stars->n == 0) {
        quality_out->fwhm = 1e6f;  /* very large = very bad */
        quality_out->roundness = 0.0f;
        quality_out->composite = 0.0f;
        quality_out->normalized = 0.0f;
        return DSO_OK;
    }

    float *fwhm_arr  = (float *)malloc((size_t)stars->n * sizeof(float));
    float *round_arr = (float *)malloc((size_t)stars->n * sizeof(float));
    if (!fwhm_arr || !round_arr) {
        free(fwhm_arr);
        free(round_arr);
        return DSO_ERR_ALLOC;
    }

    int valid = 0;
    for (int s = 0; s < stars->n; s++) {
        float fx, fy;
        float fwhm = measure_star_fwhm(conv_data, W, H,
                                        stars->stars[s].x,
                                        stars->stars[s].y,
                                        &fx, &fy);
        if (fwhm <= 0.0f) continue;

        fwhm_arr[valid] = fwhm;

        /* Roundness: if both directions available, use ratio; otherwise 1.0 */
        if (fx > 0.0f && fy > 0.0f) {
            float mn = (fx < fy) ? fx : fy;
            float mx = (fx > fy) ? fx : fy;
            round_arr[valid] = mn / mx;
        } else {
            round_arr[valid] = 1.0f;
        }
        valid++;
    }

    if (valid == 0) {
        quality_out->fwhm = 1e6f;
        quality_out->roundness = 0.0f;
        quality_out->composite = 0.0f;
        quality_out->normalized = 0.0f;
        free(fwhm_arr);
        free(round_arr);
        return DSO_OK;
    }

    quality_out->fwhm      = median_float(fwhm_arr, valid);
    quality_out->roundness  = median_float(round_arr, valid);

    free(fwhm_arr);
    free(round_arr);

    /* --- Composite score --- */
    /* composite = roundness * log2(1 + star_count) / (fwhm * (1 + 0.001 * bg)) */
    {
        float fwhm_val = quality_out->fwhm;
        if (fwhm_val < 0.1f) fwhm_val = 0.1f;  /* guard against near-zero */
        float bg_factor = 1.0f + 0.001f * quality_out->background;
        if (bg_factor < 0.01f) bg_factor = 0.01f;
        float star_factor = (float)(log(1.0 + stars->n) / log(2.0));
        quality_out->composite = quality_out->roundness * star_factor
                                 / (fwhm_val * bg_factor);
    }

    quality_out->normalized = 0.0f;  /* set by frame_quality_normalize */
    return DSO_OK;
}

void frame_quality_normalize(FrameQuality *q, float ref_composite)
{
    if (!q) return;
    if (ref_composite <= 1e-10f) {
        q->normalized = 0.0f;
        return;
    }
    q->normalized = 100.0f * q->composite / ref_composite;
}

void frame_quality_print_table(const FrameQuality *qualities,
                               const int          *frame_indices,
                               const int          *rejected,
                               int                 n)
{
    if (!qualities || !frame_indices || !rejected || n <= 0) return;

    printf("\nFrame quality summary:\n");
    printf("  %-8s %-8s %-8s %-8s %-8s %s\n",
           "Frame", "FWHM", "Round", "Stars", "Score", "");

    for (int i = 0; i < n; i++) {
        const FrameQuality *q = &qualities[i];
        int idx = frame_indices[i];
        const char *tag = "";
        if (i == 0) tag = "(ref)";
        else if (rejected[i]) tag = "[SKIPPED]";

        printf("  %-8d %-8.2f %-8.3f %-8d %-8.1f %s\n",
               idx + 1, q->fwhm, q->roundness, q->star_count,
               q->normalized, tag);
    }
    printf("\n");
}
