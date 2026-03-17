/*
 * debayer_cpu.c — CPU VNG Bayer demosaicing with OpenMP.
 *
 * Direct port of the vng_debayer_kernel in debayer_gpu.cu.  The algorithm is
 * identical; the only differences are:
 *   - No shared memory: pixels are read directly from the source array with an
 *     inline bounds-clamping helper (out-of-bounds → 0, matching GPU behaviour).
 *   - OpenMP parallelism: the outer y × x pixel loops are parallelised with
 *     #pragma omp parallel for collapse(2); each pixel is fully independent.
 */

#include "debayer_cpu.h"
#include <string.h>
#include <math.h>
#include <omp.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------- */

/*
 * src_at — read source pixel with zero out-of-bounds padding.
 * Matches the GPU shared-memory apron strategy (boundary pixels = 0).
 */
static inline float src_at(const float *src, int x, int y, int W, int H)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return 0.f;
    return src[y * W + x];
}

/*
 * bayer_channel — return 0=R, 1=G, 2=B for pixel at (x,y) given pattern.
 * Exact copy of the device function in debayer_gpu.cu.
 */
static inline int bayer_channel(int x, int y, int pattern)
{
    int px = x & 1, py = y & 1;
    switch (pattern) {
    case 1: /* RGGB */ return (py==0) ? (px==0 ? 0 : 1) : (px==0 ? 1 : 2);
    case 2: /* BGGR */ return (py==0) ? (px==0 ? 2 : 1) : (px==0 ? 1 : 0);
    case 3: /* GRBG */ return (py==0) ? (px==0 ? 1 : 0) : (px==0 ? 2 : 1);
    case 4: /* GBRG */ return (py==0) ? (px==0 ? 1 : 2) : (px==0 ? 0 : 1);
    default: return 1;
    }
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

DsoError debayer_cpu(const float *src, float *dst,
                     int W, int H, BayerPattern pattern)
{
    if (!src || !dst || W <= 0 || H <= 0) return DSO_ERR_INVALID_ARG;

    /* Monochrome fast path */
    if (pattern == BAYER_NONE) {
        memcpy(dst, src, (size_t)W * H * sizeof(float));
        return DSO_OK;
    }

    int pat = (int)pattern;   /* integer for switch inside parallel region */

    /* Cardinal direction offsets and gradient-index mapping (N, E, S, W) */
    static const int card_dx[4] = { 0, 1,  0, -1};
    static const int card_dy[4] = {-1, 0,  1,  0};
    static const int card_gi[4] = { 0, 2,  4,  6};

    /* Diagonal direction offsets and gradient-index mapping (NE, SE, SW, NW) */
    static const int diag_dx[4] = { 1,  1, -1, -1};
    static const int diag_dy[4] = {-1,  1,  1, -1};
    static const int diag_gi[4] = { 1,  3,  5,  7};

#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {

            /* ---- 1. Channel of this CFA pixel ---- */
            int chan = bayer_channel(x, y, pat);

            /* ---- 2. Eight directional gradients ----
             * Each gradient uses ±2-pixel offsets so both ends sample the same
             * channel (they span one full 2×2 CFA period). */
            float g[8];
            g[0] = fabsf(src_at(src,x, y-2,W,H) - src_at(src,x,   y,  W,H))
                 + fabsf(src_at(src,x, y,  W,H) - src_at(src,x,   y+2,W,H)); /* N  */
            g[1] = fabsf(src_at(src,x-2,y-2,W,H) - src_at(src,x,  y,  W,H))
                 + fabsf(src_at(src,x,  y,  W,H) - src_at(src,x+2,y+2,W,H)); /* NE */
            g[2] = fabsf(src_at(src,x-2,y,  W,H) - src_at(src,x,  y,  W,H))
                 + fabsf(src_at(src,x,  y,  W,H) - src_at(src,x+2,y,  W,H)); /* E  */
            g[3] = fabsf(src_at(src,x-2,y+2,W,H) - src_at(src,x,  y,  W,H))
                 + fabsf(src_at(src,x,  y,  W,H) - src_at(src,x+2,y-2,W,H)); /* SE */
            g[4] = fabsf(src_at(src,x, y+2,W,H) - src_at(src,x,  y,  W,H))
                 + fabsf(src_at(src,x, y,  W,H) - src_at(src,x,  y-2,W,H)); /* S  */
            g[5] = fabsf(src_at(src,x+2,y+2,W,H) - src_at(src,x,  y,  W,H))
                 + fabsf(src_at(src,x,  y,  W,H) - src_at(src,x-2,y-2,W,H)); /* SW */
            g[6] = fabsf(src_at(src,x+2,y,  W,H) - src_at(src,x,  y,  W,H))
                 + fabsf(src_at(src,x,  y,  W,H) - src_at(src,x-2,y,  W,H)); /* W  */
            g[7] = fabsf(src_at(src,x+2,y-2,W,H) - src_at(src,x,  y,  W,H))
                 + fabsf(src_at(src,x,  y,  W,H) - src_at(src,x-2,y+2,W,H)); /* NW */

            /* ---- 3. Gradient threshold τ = mean + min ---- */
            float gsum = 0.f, gmin = g[0];
            for (int i = 0; i < 8; i++) {
                gsum += g[i];
                if (g[i] < gmin) gmin = g[i];
            }
            float tau = gsum / 8.f + gmin;

            /* ---- 4. Accumulate colour estimates from selected directions ---- */
            float R_sum = 0.f, G_sum = 0.f, B_sum = 0.f;
            float R_w   = 0.f, G_w   = 0.f, B_w   = 0.f;
            float self  = src_at(src, x, y, W, H);

            for (int d = 0; d < 4; d++) {
                if (g[card_gi[d]] > tau) continue;
                int nx = x + card_dx[d], ny = y + card_dy[d];
                float nb_val  = src_at(src, nx, ny, W, H);
                int   nb_chan = bayer_channel(nx, ny, pat);
                float est     = (self + nb_val) * 0.5f;
                if      (nb_chan == 0) { R_sum += est; R_w += 1.f; }
                else if (nb_chan == 1) { G_sum += est; G_w += 1.f; }
                else                  { B_sum += est; B_w += 1.f; }
            }

            for (int d = 0; d < 4; d++) {
                if (g[diag_gi[d]] > tau) continue;
                int nx = x + diag_dx[d], ny = y + diag_dy[d];
                float nb_val  = src_at(src, nx, ny, W, H);
                int   nb_chan = bayer_channel(nx, ny, pat);
                float est     = (self + nb_val) * 0.5f;
                if      (nb_chan == 0) { R_sum += est; R_w += 1.f; }
                else if (nb_chan == 1) { G_sum += est; G_w += 1.f; }
                else                  { B_sum += est; B_w += 1.f; }
            }

            /* Own channel contributes directly */
            if      (chan == 0) { R_sum += self; R_w += 1.f; }
            else if (chan == 1) { G_sum += self; G_w += 1.f; }
            else                { B_sum += self; B_w += 1.f; }

            float R_est = (R_w > 0.f) ? (R_sum / R_w) : self;
            float G_est = (G_w > 0.f) ? (G_sum / G_w) : self;
            float B_est = (B_w > 0.f) ? (B_sum / B_w) : self;

            /* ---- 5. ITU-R BT.709 luminance ---- */
            dst[y * W + x] = 0.2126f * R_est + 0.7152f * G_est + 0.0722f * B_est;
        }
    }

    return DSO_OK;
}
