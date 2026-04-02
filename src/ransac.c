/*
 * ransac.c — Triangle/Asterism matching + DLT homography estimation (CPU, C11).
 *
 * Homography convention: backward map (ref -> src).
 */

#include "ransac.h"
#include "compat.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define TRIANGLE_HASH_EPSILON 0.005f

/* -------------------------------------------------------------------------
 * Normalisation helpers for DLT
 * ------------------------------------------------------------------------- */

typedef struct {
    double cx, cy;
    double scale;
} NormInfo;

static void normalise_points(const StarPos *pts, int n,
                              double *nx, double *ny,
                              NormInfo *ni)
{
    double cx = 0.0, cy = 0.0;
    double dist = 0.0;
    int i;

    for (i = 0; i < n; i++) { cx += pts[i].x; cy += pts[i].y; }
    cx /= (double)n;
    cy /= (double)n;

    for (i = 0; i < n; i++) {
        double dx = (double)pts[i].x - cx;
        double dy = (double)pts[i].y - cy;
        dist += sqrt(dx * dx + dy * dy);
    }
    dist /= (double)n;

    ni->scale = (dist > 1e-10) ? (sqrt(2.0) / dist) : 1.0;
    ni->cx = cx;
    ni->cy = cy;

    for (i = 0; i < n; i++) {
        nx[i] = ((double)pts[i].x - cx) * ni->scale;
        ny[i] = ((double)pts[i].y - cy) * ni->scale;
    }
}

static void build_T(const NormInfo *ni, double T[9])
{
    double s = ni->scale;
    T[0] = s;  T[1] = 0.0; T[2] = -s * ni->cx;
    T[3] = 0.0; T[4] = s;  T[5] = -s * ni->cy;
    T[6] = 0.0; T[7] = 0.0; T[8] = 1.0;
}

static void mat33_mul(const double A[9], const double B[9], double C[9])
{
    int r, c, k;
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            double s = 0.0;
            for (k = 0; k < 3; k++) s += A[r * 3 + k] * B[k * 3 + c];
            C[r * 3 + c] = s;
        }
    }
}

static int mat33_inv(const double M[9], double Minv[9])
{
    double c00 =  M[4] * M[8] - M[5] * M[7];
    double c01 = -(M[3] * M[8] - M[5] * M[6]);
    double c02 =  M[3] * M[7] - M[4] * M[6];
    double det = M[0] * c00 + M[1] * c01 + M[2] * c02;
    double id;

    if (fabs(det) < 1e-12) return -1;

    id = 1.0 / det;
    Minv[0] = c00 * id;
    Minv[1] = -(M[1] * M[8] - M[2] * M[7]) * id;
    Minv[2] =  (M[1] * M[5] - M[2] * M[4]) * id;
    Minv[3] = c01 * id;
    Minv[4] =  (M[0] * M[8] - M[2] * M[6]) * id;
    Minv[5] = -(M[0] * M[5] - M[2] * M[3]) * id;
    Minv[6] = c02 * id;
    Minv[7] = -(M[0] * M[7] - M[1] * M[6]) * id;
    Minv[8] =  (M[0] * M[4] - M[1] * M[3]) * id;
    return 0;
}

/* -------------------------------------------------------------------------
 * Jacobi eigendecomposition (9x9 symmetric) for minimum eigenvector
 * ------------------------------------------------------------------------- */

#define JACOBI_ITER 100

static int jacobi_min_eigvec(const double M_in[81], double evec[9])
{
    double M[81], V[81];
    int i, j, sweep;

    memcpy(M, M_in, 81 * sizeof(double));
    memset(V, 0, 81 * sizeof(double));
    for (i = 0; i < 9; i++) V[i * 9 + i] = 1.0;

    for (sweep = 0; sweep < JACOBI_ITER; sweep++) {
        int p = 0, q = 1;
        double off_max = 0.0;

        for (i = 0; i < 9; i++) {
            for (j = i + 1; j < 9; j++) {
                double v = fabs(M[i * 9 + j]);
                if (v > off_max) { off_max = v; p = i; q = j; }
            }
        }
        if (off_max < 1e-15) break;

        {
            double app = M[p * 9 + p];
            double aqq = M[q * 9 + q];
            double apq = M[p * 9 + q];
            double theta = (aqq - app) / (2.0 * apq);
            double t = (theta >= 0.0)
                ? 1.0 / (theta + sqrt(1.0 + theta * theta))
                : 1.0 / (theta - sqrt(1.0 + theta * theta));
            double c = 1.0 / sqrt(1.0 + t * t);
            double s = t * c;
            int r;

            for (r = 0; r < 9; r++) {
                if (r == p || r == q) continue;
                {
                    double mrp = M[r * 9 + p];
                    double mrq = M[r * 9 + q];
                    M[r * 9 + p] = M[p * 9 + r] = c * mrp - s * mrq;
                    M[r * 9 + q] = M[q * 9 + r] = s * mrp + c * mrq;
                }
            }

            M[p * 9 + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            M[q * 9 + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            M[p * 9 + q] = M[q * 9 + p] = 0.0;

            for (r = 0; r < 9; r++) {
                double vrp = V[r * 9 + p];
                double vrq = V[r * 9 + q];
                V[r * 9 + p] = c * vrp - s * vrq;
                V[r * 9 + q] = s * vrp + c * vrq;
            }
        }
    }

    {
        int min_idx = 0;
        double min_val = M[0];
        for (i = 1; i < 9; i++) {
            if (M[i * 9 + i] < min_val) {
                min_val = M[i * 9 + i];
                min_idx = i;
            }
        }
        for (i = 0; i < 9; i++) evec[i] = V[i * 9 + min_idx];
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * DLT homography
 * ------------------------------------------------------------------------- */

DsoError dlt_homography(const StarPos *ref_pts,
                         const StarPos *src_pts,
                         int            n,
                         Homography    *H_out)
{
    double *nx_r, *ny_r, *nx_s, *ny_s;
    NormInfo ni_r, ni_s;
    double AtA[81];
    double h_norm[9];
    double T_ref[9], T_src[9], T_src_inv[9];
    double tmp[9], H_raw[9];
    int i;

    if (!ref_pts || !src_pts || !H_out || n < 4) return DSO_ERR_INVALID_ARG;

    nx_r = (double *)malloc((size_t)n * sizeof(double));
    ny_r = (double *)malloc((size_t)n * sizeof(double));
    nx_s = (double *)malloc((size_t)n * sizeof(double));
    ny_s = (double *)malloc((size_t)n * sizeof(double));
    if (!nx_r || !ny_r || !nx_s || !ny_s) {
        free(nx_r); free(ny_r); free(nx_s); free(ny_s);
        return DSO_ERR_ALLOC;
    }

    normalise_points(ref_pts, n, nx_r, ny_r, &ni_r);
    normalise_points(src_pts, n, nx_s, ny_s, &ni_s);

    memset(AtA, 0, sizeof(AtA));
    for (i = 0; i < n; i++) {
        double rx = nx_r[i], ry = ny_r[i];
        double sx = nx_s[i], sy = ny_s[i];
        double a0[9] = {-rx, -ry, -1.0, 0.0, 0.0, 0.0, sx * rx, sx * ry, sx};
        double a1[9] = {0.0, 0.0, 0.0, -rx, -ry, -1.0, sy * rx, sy * ry, sy};
        int r, c;

        for (r = 0; r < 9; r++) {
            for (c = r; c < 9; c++) {
                double v = a0[r] * a0[c] + a1[r] * a1[c];
                AtA[r * 9 + c] += v;
                if (r != c) AtA[c * 9 + r] += v;
            }
        }
    }

    free(nx_r); free(ny_r); free(nx_s); free(ny_s);

    if (jacobi_min_eigvec(AtA, h_norm) != 0) return DSO_ERR_INVALID_ARG;

    build_T(&ni_r, T_ref);
    build_T(&ni_s, T_src);
    if (mat33_inv(T_src, T_src_inv) != 0) return DSO_ERR_INVALID_ARG;

    mat33_mul(T_src_inv, h_norm, tmp);
    mat33_mul(tmp, T_ref, H_raw);

    {
        double sc = H_raw[8];
        if (fabs(sc) < 1e-12) {
            /* Frobenius norm fallback — more stable than max-element */
            double frob = 0.0;
            for (i = 0; i < 9; i++) frob += H_raw[i] * H_raw[i];
            sc = sqrt(frob);
            if (sc < 1e-12) return DSO_ERR_INVALID_ARG;
        }
        for (i = 0; i < 9; i++) H_out->h[i] = H_raw[i] / sc;
    }

    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * Triangle matching
 * ------------------------------------------------------------------------- */

typedef struct {
    float r1, r2;
    int v0, v1, v2;
} TriangleHash;

typedef struct {
    int src_idx;
    int ref_idx;
    int votes;
} PairVote;

static int choose3_int(int n)
{
    if (n < 3) return 0;
    return (n * (n - 1) * (n - 2)) / 6;
}

static void sort3_with_ids(float *a, int *ia, float *b, int *ib, float *c, int *ic)
{
    float tv;
    int ti;
    if (*a > *b) {
        tv = *a; *a = *b; *b = tv;
        ti = *ia; *ia = *ib; *ib = ti;
    }
    if (*b > *c) {
        tv = *b; *b = *c; *c = tv;
        ti = *ib; *ib = *ic; *ic = ti;
    }
    if (*a > *b) {
        tv = *a; *a = *b; *b = tv;
        ti = *ia; *ia = *ib; *ib = ti;
    }
}

static int build_single_triangle_hash(const StarPos *stars,
                                      int i, int j, int k,
                                      TriangleHash *out)
{
    float dx_ij = stars[i].x - stars[j].x;
    float dy_ij = stars[i].y - stars[j].y;
    float dx_jk = stars[j].x - stars[k].x;
    float dy_jk = stars[j].y - stars[k].y;
    float dx_ki = stars[k].x - stars[i].x;
    float dy_ki = stars[k].y - stars[i].y;

    float l_ij = sqrtf(dx_ij * dx_ij + dy_ij * dy_ij);
    float l_jk = sqrtf(dx_jk * dx_jk + dy_jk * dy_jk);
    float l_ki = sqrtf(dx_ki * dx_ki + dy_ki * dy_ki);

    float a = l_ij;
    float b = l_jk;
    float c = l_ki;

    int va = k; /* opposite l_ij */
    int vb = i; /* opposite l_jk */
    int vc = j; /* opposite l_ki */

    if (c < 1e-6f) return 0;

    sort3_with_ids(&a, &va, &b, &vb, &c, &vc);

    out->r1 = a / c;
    out->r2 = b / c;
    out->v0 = va;
    out->v1 = vb;
    out->v2 = vc;
    return 1;
}

static int build_triangle_hashes(const StarList *list,
                                 TriangleHash *out_hashes,
                                 int max_hashes)
{
    int n = list->n;
    int count = 0;
    int i;

#pragma omp parallel for schedule(dynamic)
    for (i = 0; i < n - 2; i++) {
        int j;
        for (j = i + 1; j < n - 1; j++) {
            int k;
            for (k = j + 1; k < n; k++) {
                TriangleHash h;
                if (build_single_triangle_hash(list->stars, i, j, k, &h)) {
#pragma omp critical
                    {
                        if (count < max_hashes) out_hashes[count++] = h;
                    }
                }
            }
        }
    }

    if (count > max_hashes) count = max_hashes;
    return count;
}

static int cmp_triangle_hash(const void *a, const void *b)
{
    const TriangleHash *x = (const TriangleHash *)a;
    const TriangleHash *y = (const TriangleHash *)b;
    if (x->r1 < y->r1) return -1;
    if (x->r1 > y->r1) return 1;
    if (x->r2 < y->r2) return -1;
    if (x->r2 > y->r2) return 1;
    return 0;
}

static int lower_bound_r1(const TriangleHash *arr, int n, float key)
{
    int lo = 0;
    int hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid].r1 < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static void vote_triangle_matches(const TriangleHash *ref_hashes, int n_ref_hash,
                                  const TriangleHash *src_hashes, int n_src_hash,
                                  int n_ref_stars, int *votes,
                                  float eps)
{
    int s;

#pragma omp parallel for schedule(dynamic)
    for (s = 0; s < n_src_hash; s++) {
        const TriangleHash *sh = &src_hashes[s];
        float r1_min = sh->r1 - eps;
        float r1_max = sh->r1 + eps;
        int r = lower_bound_r1(ref_hashes, n_ref_hash, r1_min);

        while (r < n_ref_hash && ref_hashes[r].r1 <= r1_max) {
            const TriangleHash *rh = &ref_hashes[r];
            if (fabsf(rh->r2 - sh->r2) <= eps) {
                int idx0 = sh->v0 * n_ref_stars + rh->v0;
                int idx1 = sh->v1 * n_ref_stars + rh->v1;
                int idx2 = sh->v2 * n_ref_stars + rh->v2;
#pragma omp atomic
                votes[idx0] += 1;
#pragma omp atomic
                votes[idx1] += 1;
#pragma omp atomic
                votes[idx2] += 1;
            }
            r++;
        }
    }
}

static int cmp_pair_vote_desc(const void *a, const void *b)
{
    const PairVote *x = (const PairVote *)a;
    const PairVote *y = (const PairVote *)b;
    if (x->votes > y->votes) return -1;
    if (x->votes < y->votes) return 1;
    if (x->src_idx < y->src_idx) return -1;
    if (x->src_idx > y->src_idx) return 1;
    if (x->ref_idx < y->ref_idx) return -1;
    if (x->ref_idx > y->ref_idx) return 1;
    return 0;
}

static int extract_correspondences(const StarList *ref_list,
                                   const StarList *src_list,
                                   const int *votes,
                                   StarPos *ref_pts,
                                   StarPos *src_pts)
{
    int n_ref = ref_list->n;
    int n_src = src_list->n;
    PairVote *pairs = NULL;
    int *best_ref_for_src = NULL;
    int *best_vote_for_src = NULL;
    int *best_src_for_ref = NULL;
    int *best_vote_for_ref = NULL;
    int n_pairs = 0;
    int selected = 0;
    int s;

    pairs = (PairVote *)malloc((size_t)(n_ref < n_src ? n_ref : n_src) * sizeof(PairVote));
    best_ref_for_src = (int *)malloc((size_t)n_src * sizeof(int));
    best_vote_for_src = (int *)malloc((size_t)n_src * sizeof(int));
    best_src_for_ref = (int *)malloc((size_t)n_ref * sizeof(int));
    best_vote_for_ref = (int *)malloc((size_t)n_ref * sizeof(int));
    if (!pairs || !best_ref_for_src || !best_vote_for_src ||
        !best_src_for_ref || !best_vote_for_ref) {
        free(pairs);
        free(best_ref_for_src);
        free(best_vote_for_src);
        free(best_src_for_ref);
        free(best_vote_for_ref);
        return -1;
    }

    for (s = 0; s < n_src; s++) {
        best_ref_for_src[s] = -1;
        best_vote_for_src[s] = 0;
    }
    for (int r = 0; r < n_ref; r++) {
        best_src_for_ref[r] = -1;
        best_vote_for_ref[r] = 0;
    }

    for (s = 0; s < n_src; s++) {
        for (int r = 0; r < n_ref; r++) {
            int v = votes[s * n_ref + r];
            if (v > best_vote_for_src[s]) {
                best_vote_for_src[s] = v;
                best_ref_for_src[s] = r;
            }
            if (v > best_vote_for_ref[r]) {
                best_vote_for_ref[r] = v;
                best_src_for_ref[r] = s;
            }
        }
    }

    for (s = 0; s < n_src; s++) {
        int r = best_ref_for_src[s];
        if (r < 0) continue;
        if (best_vote_for_src[s] <= 0) continue;
        if (best_src_for_ref[r] != s) continue;
        pairs[n_pairs].src_idx = s;
        pairs[n_pairs].ref_idx = r;
        pairs[n_pairs].votes = best_vote_for_src[s];
        n_pairs++;
    }

    if (n_pairs > 0) {
        qsort(pairs, (size_t)n_pairs, sizeof(PairVote), cmp_pair_vote_desc);
    }

    for (s = 0; s < n_pairs; s++) {
        int si = pairs[s].src_idx;
        int ri = pairs[s].ref_idx;
        src_pts[selected] = src_list->stars[si];
        ref_pts[selected] = ref_list->stars[ri];
        selected++;
    }

    free(pairs);
    free(best_ref_for_src);
    free(best_vote_for_src);
    free(best_src_for_ref);
    free(best_vote_for_ref);
    return selected;
}

static int translate_only_model(const StarList *ref_list,
                                const StarList *src_list,
                                Homography *H_out,
                                int *n_pairs_out,
                                float match_radius)
{
    int n_ref = ref_list->n;
    int n_src = src_list->n;
    float r2 = match_radius * match_radius;
    float sum_dx = 0.0f, sum_dy = 0.0f;
    int pairs = 0;

    for (int si = 0; si < n_src; si++) {
        float sx = src_list->stars[si].x;
        float sy = src_list->stars[si].y;
        float best_d2 = r2 + 1.0f;
        int best_ri = -1;
        for (int ri = 0; ri < n_ref; ri++) {
            float dx = sx - ref_list->stars[ri].x;
            float dy = sy - ref_list->stars[ri].y;
            float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_ri = ri;
            }
        }
        if (best_ri >= 0 && best_d2 <= r2) {
            sum_dx += sx - ref_list->stars[best_ri].x;
            sum_dy += sy - ref_list->stars[best_ri].y;
            pairs++;
        }
    }

    if (pairs < 4) return 0;
    memset(H_out, 0, sizeof(*H_out));
    H_out->h[0] = 1.0;
    H_out->h[4] = 1.0;
    H_out->h[8] = 1.0;
    H_out->h[2] = (double)(sum_dx / (float)pairs);
    H_out->h[5] = (double)(sum_dy / (float)pairs);
    if (n_pairs_out) *n_pairs_out = pairs;
    return 1;
}

static int nearest_neighbor_pairs(const StarList *ref_list,
                                  const StarList *src_list,
                                  float match_radius,
                                  StarPos *ref_pts,
                                  StarPos *src_pts)
{
    int n_ref = ref_list->n;
    int n_src = src_list->n;
    float r2 = match_radius * match_radius;
    unsigned char *used_ref = (unsigned char *)calloc((size_t)n_ref, 1);
    int n = 0;
    int si;

    if (!used_ref) return -1;

    for (si = 0; si < n_src; si++) {
        float sx = src_list->stars[si].x;
        float sy = src_list->stars[si].y;
        float best_d2 = r2 + 1.0f;
        int best_ri = -1;
        for (int ri = 0; ri < n_ref; ri++) {
            float dx = sx - ref_list->stars[ri].x;
            float dy = sy - ref_list->stars[ri].y;
            float d2 = dx * dx + dy * dy;
            if (!used_ref[ri] && d2 < best_d2) {
                best_d2 = d2;
                best_ri = ri;
            }
        }
        if (best_ri >= 0 && best_d2 <= r2) {
            used_ref[best_ri] = 1;
            src_pts[n] = src_list->stars[si];
            ref_pts[n] = ref_list->stars[best_ri];
            n++;
        }
    }

    free(used_ref);
    return n;
}

static int index_pairs(const StarList *ref_list,
                       const StarList *src_list,
                       StarPos *ref_pts,
                       StarPos *src_pts)
{
    int n = (ref_list->n < src_list->n) ? ref_list->n : src_list->n;
    for (int i = 0; i < n; i++) {
        ref_pts[i] = ref_list->stars[i];
        src_pts[i] = src_list->stars[i];
    }
    return n;
}

static double reproj_err_sq(const Homography *H,
                            float rx, float ry,
                            float sx, float sy)
{
    const double *h = H->h;
    double qx_h = h[0] * rx + h[1] * ry + h[2];
    double qy_h = h[3] * rx + h[4] * ry + h[5];
    double qw   = h[6] * rx + h[7] * ry + h[8];
    double qx, qy;
    if (fabs(qw) < 1e-12) return 1e18;
    qx = qx_h / qw - sx;
    qy = qy_h / qw - sy;
    return qx * qx + qy * qy;
}

static double homography_det(const Homography *H)
{
    const double *h = H->h;
    return h[0] * (h[4] * h[8] - h[5] * h[7])
         - h[1] * (h[3] * h[8] - h[5] * h[6])
         + h[2] * (h[3] * h[7] - h[4] * h[6]);
}

typedef struct { int ri; int si; } Match;

static DsoError fallback_ransac_compute(const StarList *ref_list,
                                        const StarList *src_list,
                                        const RansacParams *p,
                                        Homography *H_out,
                                        int *n_inliers_out)
{
    int max_matches = ref_list->n;
    Match *matches = (Match *)malloc((size_t)max_matches * sizeof(Match));
    int n_matches = 0;
    float r2;
    static int call_counter = 0;
    unsigned int seed;
    Homography best_H;
    int best_inliers = 0;
    int adaptive_max;
    double thresh_sq;
    StarPos sref[4], ssrc[4];
    int iter;

    if (!matches) return DSO_ERR_ALLOC;

    r2 = p->match_radius * p->match_radius;
    for (int ri = 0; ri < ref_list->n; ri++) {
        float rx = ref_list->stars[ri].x, ry = ref_list->stars[ri].y;
        float d1 = r2 + 1.f, d2 = r2 + 2.f;
        int j1 = -1;
        for (int si = 0; si < src_list->n; si++) {
            float dx = src_list->stars[si].x - rx;
            float dy = src_list->stars[si].y - ry;
            float d = dx * dx + dy * dy;
            if (d < d1) { d2 = d1; d1 = d; j1 = si; }
            else if (d < d2) { d2 = d; }
        }
        if (j1 < 0 || d1 > r2) continue;
        if (d2 < r2 + 1.f && d1 > 0.f && sqrtf(d1 / d2) > 0.8f) continue;
        matches[n_matches++] = (Match){ri, j1};
    }

    if (n_matches < (p->min_inliers > 4 ? p->min_inliers : 4)) {
        free(matches);
        return DSO_ERR_RANSAC;
    }

    seed = (unsigned int)(time(NULL) ^ clock() ^ call_counter++);
    memset(&best_H, 0, sizeof(best_H));
    adaptive_max = p->max_iters;
    thresh_sq = (double)p->inlier_thresh * p->inlier_thresh;

    for (iter = 0; iter < adaptive_max; iter++) {
        int idx[4];
        int used[4] = {-1, -1, -1, -1};
        for (int k = 0; k < 4;) {
            int j = rand_r(&seed) % n_matches;
            int dup = 0;
            for (int m = 0; m < k; m++) if (used[m] == j) { dup = 1; break; }
            if (!dup) { used[k] = j; idx[k] = j; k++; }
        }
        for (int k = 0; k < 4; k++) {
            int ri = matches[idx[k]].ri;
            int si = matches[idx[k]].si;
            sref[k] = ref_list->stars[ri];
            ssrc[k] = src_list->stars[si];
        }

        Homography H_cand;
        if (dlt_homography(sref, ssrc, 4, &H_cand) != DSO_OK) continue;

        int count = 0;
        for (int mi = 0; mi < n_matches; mi++) {
            int ri = matches[mi].ri, si = matches[mi].si;
            double e2 = reproj_err_sq(&H_cand,
                                      ref_list->stars[ri].x, ref_list->stars[ri].y,
                                      src_list->stars[si].x, src_list->stars[si].y);
            if (e2 < thresh_sq) count++;
        }
        if (count > best_inliers) {
            best_inliers = count;
            best_H = H_cand;
            double inlier_ratio = (double)count / n_matches;
            if (inlier_ratio > 0.999) inlier_ratio = 0.999;
            if (inlier_ratio > 0.001) {
                double p4 = inlier_ratio * inlier_ratio * inlier_ratio * inlier_ratio;
                double n_needed = log(1.0 - p->confidence) / log(1.0 - p4);
                int ni = (int)ceil(n_needed);
                if (ni < adaptive_max) adaptive_max = ni;
                if (adaptive_max < 4) adaptive_max = 4;
            }
        }
    }

    if (best_inliers < p->min_inliers) {
        free(matches);
        return DSO_ERR_RANSAC;
    }

    StarPos *rpts = (StarPos *)malloc((size_t)best_inliers * sizeof(StarPos));
    StarPos *spts = (StarPos *)malloc((size_t)best_inliers * sizeof(StarPos));
    if (!rpts || !spts) {
        free(rpts); free(spts); free(matches);
        return DSO_ERR_ALLOC;
    }

    int n_inliers = 0;
    for (int mi = 0; mi < n_matches; mi++) {
        int ri = matches[mi].ri, si = matches[mi].si;
        double e2 = reproj_err_sq(&best_H,
                                  ref_list->stars[ri].x, ref_list->stars[ri].y,
                                  src_list->stars[si].x, src_list->stars[si].y);
        if (e2 < thresh_sq) {
            rpts[n_inliers] = ref_list->stars[ri];
            spts[n_inliers] = src_list->stars[si];
            n_inliers++;
        }
    }

    DsoError refine_err = dlt_homography(rpts, spts, n_inliers, H_out);
    free(rpts); free(spts); free(matches);
    if (refine_err != DSO_OK) return refine_err;
    if (n_inliers_out) *n_inliers_out = n_inliers;
    return DSO_OK;
}

static const RansacParams TRI_DEFAULTS = {
    .max_iters = 1000,
    .inlier_thresh = 2.0f,
    .match_radius = 30.0f,
    .confidence = 0.99f,
    .min_inliers = 10
};

DsoError ransac_compute_homography(const StarList     *ref_list,
                                    const StarList     *src_list,
                                    const RansacParams *params,
                                    Homography         *H_out,
                                    int                *n_inliers_out)
{
    const RansacParams *p = params ? params : &TRI_DEFAULTS;
    TriangleHash *ref_hashes = NULL;
    TriangleHash *src_hashes = NULL;
    int *votes = NULL;
    StarPos *ref_pts = NULL;
    StarPos *src_pts = NULL;
    StarPos *in_ref = NULL;
    StarPos *in_src = NULL;
    int max_ref_hash, max_src_hash;
    int n_ref_hash, n_src_hash;
    int n_ref, n_src;
    int max_corr;
    int n_corr;
    int n_inliers;
    int i;
    float eps = TRIANGLE_HASH_EPSILON;
    double thresh_sq;
    DsoError err;

    if (!ref_list || !src_list || !H_out) return DSO_ERR_INVALID_ARG;
    if (ref_list->n < 4 || src_list->n < 4) return DSO_ERR_STAR_DETECT;

    n_ref = ref_list->n;
    n_src = src_list->n;
    max_corr = (n_ref < n_src) ? n_ref : n_src;

    max_ref_hash = choose3_int(n_ref);
    max_src_hash = choose3_int(n_src);
    if (max_ref_hash <= 0 || max_src_hash <= 0) return DSO_ERR_STAR_DETECT;

    ref_hashes = (TriangleHash *)malloc((size_t)max_ref_hash * sizeof(TriangleHash));
    src_hashes = (TriangleHash *)malloc((size_t)max_src_hash * sizeof(TriangleHash));
    votes = (int *)calloc((size_t)n_ref * (size_t)n_src, sizeof(int));
    ref_pts = (StarPos *)malloc((size_t)(n_ref < n_src ? n_ref : n_src) * sizeof(StarPos));
    src_pts = (StarPos *)malloc((size_t)(n_ref < n_src ? n_ref : n_src) * sizeof(StarPos));

    if (!ref_hashes || !src_hashes || !votes || !ref_pts || !src_pts) {
        err = DSO_ERR_ALLOC;
        goto cleanup;
    }

    n_ref_hash = build_triangle_hashes(ref_list, ref_hashes, max_ref_hash);
    n_src_hash = build_triangle_hashes(src_list, src_hashes, max_src_hash);

    if (n_ref_hash < 1 || n_src_hash < 1) {
        err = DSO_ERR_RANSAC;
        goto cleanup;
    }

    qsort(ref_hashes, (size_t)n_ref_hash, sizeof(TriangleHash), cmp_triangle_hash);

    vote_triangle_matches(ref_hashes, n_ref_hash, src_hashes, n_src_hash, n_ref, votes, eps);

    n_corr = extract_correspondences(ref_list, src_list, votes, ref_pts, src_pts);
    if (n_corr < 4) n_corr = nearest_neighbor_pairs(ref_list, src_list, p->match_radius,
                                                     ref_pts, src_pts);
    if (n_corr < 4) n_corr = index_pairs(ref_list, src_list, ref_pts, src_pts);

    if (n_corr < 4) {
        int ok = translate_only_model(ref_list, src_list, H_out, &n_corr, p->match_radius);
        if (!ok) { err = DSO_ERR_RANSAC; goto cleanup; }
        n_inliers = n_corr;
        if (n_inliers_out) *n_inliers_out = n_inliers;
        err = DSO_OK;
        goto cleanup;
    }

    /* --- RANSAC over triangle-matched correspondences --- */
    thresh_sq = (double)p->inlier_thresh * (double)p->inlier_thresh;
    in_ref = (StarPos *)malloc((size_t)max_corr * sizeof(StarPos));
    in_src = (StarPos *)malloc((size_t)max_corr * sizeof(StarPos));
    if (!in_ref || !in_src) {
        err = DSO_ERR_ALLOC;
        goto cleanup;
    }

    {
        static int ransac_ctr = 0;
        unsigned int seed = (unsigned int)(time(NULL) ^ clock() ^ ransac_ctr++);
        Homography best_H;
        int best_inliers = 0;
        int adaptive_max = p->max_iters;

        memset(&best_H, 0, sizeof(best_H));

        for (int iter = 0; iter < adaptive_max; iter++) {
            int idx[4];
            int used[4] = {-1, -1, -1, -1};
            StarPos sref[4], ssrc[4];
            Homography H_cand;
            int count;

            /* sample 4 distinct correspondences */
            for (int k = 0; k < 4;) {
                int j = rand_r(&seed) % n_corr;
                int dup = 0;
                for (int m = 0; m < k; m++) if (used[m] == j) { dup = 1; break; }
                if (!dup) { used[k] = j; idx[k] = j; k++; }
            }
            for (int k = 0; k < 4; k++) {
                sref[k] = ref_pts[idx[k]];
                ssrc[k] = src_pts[idx[k]];
            }

            if (dlt_homography(sref, ssrc, 4, &H_cand) != DSO_OK) continue;
            if (fabs(homography_det(&H_cand)) < 1e-12) continue;

            /* count inliers */
            count = 0;
            for (int mi = 0; mi < n_corr; mi++) {
                double e2 = reproj_err_sq(&H_cand,
                                          ref_pts[mi].x, ref_pts[mi].y,
                                          src_pts[mi].x, src_pts[mi].y);
                if (e2 < thresh_sq) count++;
            }
            if (count > best_inliers) {
                best_inliers = count;
                best_H = H_cand;
                /* adaptive termination */
                double inlier_ratio = (double)count / n_corr;
                if (inlier_ratio > 0.999) inlier_ratio = 0.999;
                if (inlier_ratio > 0.001) {
                    double p4 = inlier_ratio * inlier_ratio * inlier_ratio * inlier_ratio;
                    double n_needed = log(1.0 - p->confidence) / log(1.0 - p4);
                    int ni = (int)ceil(n_needed);
                    if (ni < adaptive_max) adaptive_max = ni;
                    if (adaptive_max < 4) adaptive_max = 4;
                }
            }
        }

        if (best_inliers < p->min_inliers || best_inliers < 4) {
            err = DSO_ERR_RANSAC;
            goto cleanup;
        }

        /* collect inliers of best model */
        n_inliers = 0;
        for (i = 0; i < n_corr; i++) {
            double e2 = reproj_err_sq(&best_H,
                                      ref_pts[i].x, ref_pts[i].y,
                                      src_pts[i].x, src_pts[i].y);
            if (e2 < thresh_sq) {
                in_ref[n_inliers] = ref_pts[i];
                in_src[n_inliers] = src_pts[i];
                n_inliers++;
            }
        }

        /* refine with DLT on inliers */
        err = dlt_homography(in_ref, in_src, n_inliers, H_out);
        if (err != DSO_OK || fabs(homography_det(H_out)) < 1e-12) {
            err = DSO_ERR_RANSAC;
            goto cleanup;
        }
    }

    if (n_inliers_out) *n_inliers_out = n_inliers;
    err = DSO_OK;

cleanup:
    free(ref_hashes);
    free(src_hashes);
    free(votes);
    free(ref_pts);
    free(src_pts);
    free(in_ref);
    free(in_src);
    if (err == DSO_ERR_RANSAC || err == DSO_ERR_INVALID_ARG) {
        int fallback_in = 0;
        DsoError ferr = fallback_ransac_compute(ref_list, src_list, p, H_out, &fallback_in);
        if (ferr == DSO_OK) {
            if (n_inliers_out) *n_inliers_out = fallback_in;
            return DSO_OK;
        }
    }
    return err;
}
