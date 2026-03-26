#include "star_coords_generator.h"
#include "compat.h"

#include <stdlib.h>
#include <math.h>

static float rand_uniform(unsigned int *seed, float lo, float hi)
{
    float u = (float)rand_r(seed) / (float)RAND_MAX;
    return lo + (hi - lo) * u;
}

static void sample_point(unsigned int *seed, int width, int height, float *x, float *y)
{
    const float pad = 2.0f;
    *x = rand_uniform(seed, pad, (float)width - pad);
    *y = rand_uniform(seed, pad, (float)height - pad);
}

static void sample_point_with_min_dist(unsigned int *seed,
                                       int width, int height,
                                       const StarPos *existing, int n_existing,
                                       float min_dist,
                                       float *x, float *y)
{
    const float min_dist_sq = min_dist * min_dist;
    for (int tries = 0; tries < 1000; tries++) {
        float cx = 0.f, cy = 0.f;
        sample_point(seed, width, height, &cx, &cy);
        int ok = 1;
        for (int i = 0; i < n_existing; i++) {
            float dx = existing[i].x - cx;
            float dy = existing[i].y - cy;
            if (dx * dx + dy * dy < min_dist_sq) {
                ok = 0;
                break;
            }
        }
        if (ok) {
            *x = cx;
            *y = cy;
            return;
        }
    }
    sample_point(seed, width, height, x, y);
}

static int is_inside(float x, float y, int width, int height, float pad)
{
    return x >= pad && x <= (float)width - pad &&
           y >= pad && y <= (float)height - pad;
}

static void apply_h(const Homography *H, float x, float y, float *ox, float *oy)
{
    const double *h = H->h;
    double qx = h[0] * x + h[1] * y + h[2];
    double qy = h[3] * x + h[4] * y + h[5];
    double qw = h[6] * x + h[7] * y + h[8];
    if (fabs(qw) < 1e-12) {
        *ox = x;
        *oy = y;
        return;
    }
    *ox = (float)(qx / qw);
    *oy = (float)(qy / qw);
}

void star_coords_free(StarList *list)
{
    if (!list) return;
    free(list->stars);
    list->stars = NULL;
    list->n = 0;
}

int star_coords_generate(int n_inliers,
                         int n_ref_outliers,
                         int n_frame_outliers,
                         int width,
                         int height,
                         const Homography *H_ref_to_frame,
                         unsigned int seed,
                         StarList *ref_out,
                         StarList *frame_out)
{
    if (!H_ref_to_frame || !ref_out || !frame_out) return -1;
    if (n_inliers < 0 || n_ref_outliers < 0 || n_frame_outliers < 0) return -1;
    if (width <= 4 || height <= 4) return -1;

    ref_out->stars = NULL;
    ref_out->n = 0;
    frame_out->stars = NULL;
    frame_out->n = 0;

    const int n_ref = n_inliers + n_ref_outliers;
    const int n_frame = n_inliers + n_frame_outliers;
    StarPos *ref = (StarPos *)malloc((size_t)n_ref * sizeof(StarPos));
    StarPos *frm = (StarPos *)malloc((size_t)n_frame * sizeof(StarPos));
    if (!ref || !frm) {
        free(ref);
        free(frm);
        return -1;
    }

    for (int i = 0; i < n_inliers; i++) {
        float rx = 0.f, ry = 0.f;
        float sx = 0.f, sy = 0.f;
        int found = 0;
        for (int tries = 0; tries < 2000; tries++) {
            sample_point_with_min_dist(&seed, width, height, ref, i, 22.0f, &rx, &ry);
            apply_h(H_ref_to_frame, rx, ry, &sx, &sy);
            if (!is_inside(sx, sy, width, height, 2.0f)) continue;
            int src_far = 1;
            for (int k = 0; k < i; k++) {
                float dx = frm[k].x - sx;
                float dy = frm[k].y - sy;
                if (dx * dx + dy * dy < 22.0f * 22.0f) {
                    src_far = 0;
                    break;
                }
            }
            if (src_far) {
                found = 1;
                break;
            }
        }
        if (!found) {
            free(ref);
            free(frm);
            return -1;
        }

        ref[i].x = rx;
        ref[i].y = ry;
        ref[i].flux = 1000.f - (float)i;

        frm[i].x = sx;
        frm[i].y = sy;
        frm[i].flux = 1000.f - (float)i;
    }

    for (int i = 0; i < n_ref_outliers; i++) {
        int idx = n_inliers + i;
        sample_point_with_min_dist(&seed, width, height, ref, idx, 45.0f,
                                   &ref[idx].x, &ref[idx].y);
        ref[idx].flux = 500.f - (float)i;
    }

    for (int i = 0; i < n_frame_outliers; i++) {
        int idx = n_inliers + i;
        sample_point_with_min_dist(&seed, width, height, frm, idx, 45.0f,
                                   &frm[idx].x, &frm[idx].y);
        frm[idx].flux = 500.f - (float)i;
    }

    ref_out->stars = ref;
    ref_out->n = n_ref;
    frame_out->stars = frm;
    frame_out->n = n_frame;
    return 0;
}
