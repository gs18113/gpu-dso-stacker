/*
 * debayer_cpu.c — CPU VNG Bayer demosaicing with OpenMP.
 *
 * Implements the Variable Number of Gradients (VNG) algorithm.
 * For each pixel, 8 directional gradients are computed in a 5x5
 * neighborhood. Missing colors are interpolated by averaging colors along
 * directions with low gradients.
 */

#include "debayer_cpu.h"
#include <string.h>
#include <math.h>
#include <omp.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------- */

static inline float src_at(const float *src, int x, int y, int W, int H)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return 0.f;
    return src[y * W + x];
}

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

DsoError debayer_cpu(const float *src, float *dst,
                     int W, int H, BayerPattern pattern)
{
    if (!src || !dst || W <= 0 || H <= 0) return DSO_ERR_INVALID_ARG;

    if (pattern == BAYER_NONE) {
        memcpy(dst, src, (size_t)W * H * sizeof(float));
        return DSO_OK;
    }

    int pat = (int)pattern;

#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            /* 1. Calculate 8 gradients in a 5x5 area */
            float g[8];
            float self = src_at(src, x, y, W, H);

            /* North, South, East, West */
            g[0] = fabsf(src_at(src,x,y-2,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x,y-1,W,H) - src_at(src,x,y+1,W,H)) * 0.5f +
                   fabsf(src_at(src,x-1,y-1,W,H) - src_at(src,x-1,y+1,W,H)) * 0.5f +
                   fabsf(src_at(src,x+1,y-1,W,H) - src_at(src,x+1,y+1,W,H)) * 0.5f; /* N */
            
            g[1] = fabsf(src_at(src,x,y+2,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x,y+1,W,H) - src_at(src,x,y-1,W,H)) * 0.5f +
                   fabsf(src_at(src,x-1,y+1,W,H) - src_at(src,x-1,y-1,W,H)) * 0.5f +
                   fabsf(src_at(src,x+1,y+1,W,H) - src_at(src,x+1,y-1,W,H)) * 0.5f; /* S */

            g[2] = fabsf(src_at(src,x+2,y,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x+1,y,W,H) - src_at(src,x-1,y,W,H)) * 0.5f +
                   fabsf(src_at(src,x+1,y-1,W,H) - src_at(src,x-1,y-1,W,H)) * 0.5f +
                   fabsf(src_at(src,x+1,y+1,W,H) - src_at(src,x-1,y+1,W,H)) * 0.5f; /* E */

            g[3] = fabsf(src_at(src,x-2,y,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x-1,y,W,H) - src_at(src,x+1,y,W,H)) * 0.5f +
                   fabsf(src_at(src,x-1,y-1,W,H) - src_at(src,x+1,y-1,W,H)) * 0.5f +
                   fabsf(src_at(src,x-1,y+1,W,H) - src_at(src,x+1,y+1,W,H)) * 0.5f; /* W */

            /* Diagonals */
            g[4] = fabsf(src_at(src,x+2,y-2,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x+1,y-1,W,H) - src_at(src,x-1,y+1,W,H)) * 0.5f +
                   fabsf(src_at(src,x,y-1,W,H) - src_at(src,x-1,y,W,H)) * 0.5f +
                   fabsf(src_at(src,x+1,y,W,H) - src_at(src,x,y+1,W,H)) * 0.5f; /* NE */

            g[5] = fabsf(src_at(src,x-2,y-2,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x-1,y-1,W,H) - src_at(src,x+1,y+1,W,H)) * 0.5f +
                   fabsf(src_at(src,x,y-1,W,H) - src_at(src,x+1,y,W,H)) * 0.5f +
                   fabsf(src_at(src,x-1,y,W,H) - src_at(src,x,y+1,W,H)) * 0.5f; /* NW */

            g[6] = fabsf(src_at(src,x+2,y+2,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x+1,y+1,W,H) - src_at(src,x-1,y-1,W,H)) * 0.5f +
                   fabsf(src_at(src,x,y+1,W,H) - src_at(src,x+1,y,W,H)) * 0.5f +
                   fabsf(src_at(src,x-1,y,W,H) - src_at(src,x,y-1,W,H)) * 0.5f; /* SE */

            g[7] = fabsf(src_at(src,x-2,y+2,W,H) - src_at(src,x,y,W,H)) +
                   fabsf(src_at(src,x-1,y+1,W,H) - src_at(src,x+1,y-1,W,H)) * 0.5f +
                   fabsf(src_at(src,x,y+1,W,H) - src_at(src,x-1,y,W,H)) * 0.5f +
                   fabsf(src_at(src,x+1,y,W,H) - src_at(src,x,y-1,W,H)) * 0.5f; /* SW */

            /* 2. Threshold */
            float gsum = 0.f, gmin = g[0], gmax = g[0];
            for (int i = 0; i < 8; i++) {
                gsum += g[i];
                if (g[i] < gmin) gmin = g[i];
                if (g[i] > gmax) gmax = g[i];
            }
            float tau = 1.5f * gmin + 0.5f * (gmax - gmin);

            /* 3. Interpolate R, G, B */
            float R_sum = 0.f, G_sum = 0.f, B_sum = 0.f;
            int R_count = 0, G_count = 0, B_count = 0;
            int chan = bayer_channel(x, y, pat);

            /* Directions: N, S, E, W, NE, NW, SE, SW */
            const int dx[8] = {0, 0, 1, -1, 1, -1, 1, -1};
            const int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};

            for (int i = 0; i < 8; i++) {
                if (g[i] <= tau) {
                    int nx = x + dx[i], ny = y + dy[i];
                    int nchan = bayer_channel(nx, ny, pat);
                    float nv = src_at(src, nx, ny, W, H);
                    
                    if (nchan == 0)      { R_sum += nv; R_count++; }
                    else if (nchan == 1) { G_sum += nv; G_count++; }
                    else                 { B_sum += nv; B_count++; }
                }
            }

            /* Own channel */
            if (chan == 0)      { R_sum += self; R_count++; }
            else if (chan == 1) { G_sum += self; G_count++; }
            else                { B_sum += self; B_count++; }

            float R_est = (R_count > 0) ? (R_sum / R_count) : self;
            float G_est = (G_count > 0) ? (G_sum / G_count) : self;
            float B_est = (B_count > 0) ? (B_sum / B_count) : self;

            dst[y * W + x] = 0.2126f * R_est + 0.7152f * G_est + 0.0722f * B_est;
        }
    }

    return DSO_OK;
}
