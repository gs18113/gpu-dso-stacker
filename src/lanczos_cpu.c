/*
 * lanczos_cpu.c — CPU Lanczos-3 image transformation
 *
 * Applies a homographic (perspective) warp to a float32 image using
 * backward mapping with a Lanczos-3 interpolation kernel.
 *
 * Algorithm:
 *   1. H is the backward homography (ref → src); use it directly.
 *   2. For each destination pixel (dx, dy):
 *        [sx_h, sy_h, sw]^T = H * [dx, dy, 1]^T
 *        sx = sx_h / sw,  sy = sy_h / sw
 *   3. Sample the source at (sx, sy) with a 6×6 Lanczos-3 tap window.
 *      Boundary taps outside [0, W) × [0, H) are skipped; weights are
 *      renormalised so partial windows near edges are handled cleanly.
 *      Pixels with no valid taps are set to NAN (out-of-bounds sentinel).
 *      Integration stages skip NAN-valued pixels so that OOB regions
 *      from warped frames do not contaminate the stacked result.
 */

#include "lanczos_cpu.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Lanczos-3 kernel weight for offset x.
 *
 * L(x) = sinc(x) * sinc(x/3)   for |x| < 3
 *       = 0                      otherwise
 *
 * Using the unnormalised sinc: sinc(t) = sin(π·t) / (π·t)
 * The leading factor of 3 in the numerator cancels the denominator's /3.
 */
static float lanczos_weight(float x)
{
    if (x == 0.f) return 1.f;
    if (fabsf(x) >= 3.f) return 0.f;
    float px = (float)M_PI * x;
    return 3.f * sinf(px) * sinf(px / 3.f) / (px * px);
}

/*
 * Analytically invert a 3×3 homography via the cofactor/adjugate method.
 * Returns DSO_ERR_INVALID_ARG if the matrix is singular (|det| < 1e-12).
 */
static DsoError invert_homography(const Homography *H, Homography *H_inv)
{
    const double *h = H->h;

    /* Cofactors of each element (these are the minors with sign) */
    double c00 =  h[4]*h[8] - h[5]*h[7];
    double c01 = -(h[3]*h[8] - h[5]*h[6]);
    double c02 =  h[3]*h[7] - h[4]*h[6];
    double c10 = -(h[1]*h[8] - h[2]*h[7]);
    double c11 =  h[0]*h[8] - h[2]*h[6];
    double c12 = -(h[0]*h[7] - h[1]*h[6]);
    double c20 =  h[1]*h[5] - h[2]*h[4];
    double c21 = -(h[0]*h[5] - h[2]*h[3]);
    double c22 =  h[0]*h[4] - h[1]*h[3];

    /* Determinant via cofactor expansion along the first row */
    double det = h[0]*c00 + h[1]*c01 + h[2]*c02;

    if (fabs(det) < 1e-12) {
        fprintf(stderr, "lanczos_cpu: homography is singular (det=%g)\n", det);
        return DSO_ERR_INVALID_ARG;
    }

    /* H_inv = (1/det) * adjugate(H) = (1/det) * cofactor_matrix^T */
    double inv_det = 1.0 / det;
    H_inv->h[0] = c00 * inv_det;  H_inv->h[1] = c10 * inv_det;  H_inv->h[2] = c20 * inv_det;
    H_inv->h[3] = c01 * inv_det;  H_inv->h[4] = c11 * inv_det;  H_inv->h[5] = c21 * inv_det;
    H_inv->h[6] = c02 * inv_det;  H_inv->h[7] = c12 * inv_det;  H_inv->h[8] = c22 * inv_det;

    return DSO_OK;
}

DsoError lanczos_transform_cpu(const Image *src, Image *dst, const Homography *H)
{
    if (!src || !dst || !H || !src->data || !dst->data) return DSO_ERR_INVALID_ARG;

    /* H is already the backward map (ref → src); use it directly. */
    const double *hi = H->h;
    int SW = src->width,  SH = src->height;
    int DW = dst->width,  DH = dst->height;

    /* Check for identity transform (fast path) */
    int is_identity = (fabs(hi[0]-1.0) < 1e-9 && fabs(hi[4]-1.0) < 1e-9 && fabs(hi[8]-1.0) < 1e-9 &&
                       fabs(hi[1]) < 1e-9 && fabs(hi[2]) < 1e-9 && fabs(hi[3]) < 1e-9 &&
                       fabs(hi[5]) < 1e-9 && fabs(hi[6]) < 1e-9 && fabs(hi[7]) < 1e-9);

    if (is_identity && SW == DW && SH == DH) {
        memcpy(dst->data, src->data, (size_t)SW * SH * sizeof(float));
        return DSO_OK;
    }

    int dy;
#pragma omp parallel for schedule(static)
    for (dy = 0; dy < DH; dy++) {
        for (int dx = 0; dx < DW; dx++) {

            /* Map destination pixel (dx, dy) to source coordinates (sx, sy)
             * using H (the backward map) directly in homogeneous coordinates. */
            double sx_h = hi[0]*dx + hi[1]*dy + hi[2];
            double sy_h = hi[3]*dx + hi[4]*dy + hi[5];
            double sw   = hi[6]*dx + hi[7]*dy + hi[8];

            if (fabs(sw) < 1e-12) {
                dst->data[dy * DW + dx] = NAN;
                continue;
            }

            float sx = (float)(sx_h / sw);
            float sy = (float)(sy_h / sw);

            /* Integer part of the source coordinate (floor toward −∞) */
            int ix = (int)floorf(sx);
            int iy = (int)floorf(sy);

            /*
             * 6×6 Lanczos-3 tap window: offsets j, i in {−2, −1, 0, 1, 2, 3}.
             * Taps that fall outside the source image are skipped and the
             * weight sum is renormalised, so the filter degrades gracefully
             * at image boundaries rather than ringing or clamping.
             */
            double accum      = 0.0;
            double weight_sum = 0.0;

            /* Pre-calculate 6 weights for x and 6 for y */
            float wx_arr[6], wy_arr[6];
            for (int k = 0; k < 6; k++) {
                wx_arr[k] = lanczos_weight(sx - (float)(ix - 2 + k));
                wy_arr[k] = lanczos_weight(sy - (float)(iy - 2 + k));
            }

            for (int j = 0; j < 6; j++) {
                int row = iy - 2 + j;
                if (row < 0 || row >= SH) continue;
                float wy = wy_arr[j];

                for (int i = 0; i < 6; i++) {
                    int col = ix - 2 + i;
                    if (col < 0 || col >= SW) continue;

                    float wx = wx_arr[i];
                    float w  = wx * wy;
                    accum      += w * src->data[row * SW + col];
                    weight_sum += w;
                }
            }

            dst->data[dy * DW + dx] = (fabs(weight_sum) < 1e-6) ? NAN : (float)(accum / weight_sum);
        }
    }

    return DSO_OK;
}
