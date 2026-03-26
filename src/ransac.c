/*
 * ransac.c — Star-based homography estimation via RANSAC + DLT.
 *
 * DLT algorithm:
 *   For each correspondence (ref_x, ref_y) → (src_x, src_y) we get two rows
 *   of the 2N×9 matrix A encoding H * p_ref ∝ p_src:
 *     Row 2i  : [-rx, -ry, -1,   0,   0,  0, sx*rx, sx*ry, sx]
 *     Row 2i+1: [  0,   0,  0, -rx, -ry, -1, sy*rx, sy*ry, sy]
 *
 *   The null vector h of A (reshaped to 3×3) is H.  We avoid computing the
 *   full SVD by working with M = AᵀA (9×9 symmetric) and finding its minimum-
 *   eigenvalue eigenvector via classical Jacobi iteration.
 *
 *   Point normalisation: before assembling A, both ref and src point sets are
 *   normalised (centroid → origin, mean distance → √2) for numerical stability.
 *   The raw solution is de-normalised before returning.
 *
 * RANSAC:
 *   4-point minimal samples, adaptive termination based on running inlier
 *   fraction p:  N_remain = log(1−conf) / log(1−p^4).
 *   After the main loop, H is re-estimated from all inliers (refinement DLT).
 */

#include "ransac.h"
#include "compat.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Normalisation helpers
 * ------------------------------------------------------------------------- */

typedef struct {
    double cx, cy;  /* centroid */
    double scale;   /* scale such that mean distance to centroid = sqrt(2) */
} NormInfo;

/*
 * normalise_points — translate centroid to origin and scale mean distance
 * to √2.  Fills norm[] with transformed coordinates and ni with the
 * normalisation parameters needed for de-normalisation.
 */
static void normalise_points(const StarPos *pts, int n,
                              double *nx, double *ny,
                              NormInfo *ni)
{
    double cx = 0, cy = 0;
    for (int i = 0; i < n; i++) { cx += pts[i].x; cy += pts[i].y; }
    cx /= n; cy /= n;

    double dist = 0;
    for (int i = 0; i < n; i++) {
        double dx = pts[i].x - cx, dy = pts[i].y - cy;
        dist += sqrt(dx*dx + dy*dy);
    }
    dist /= n;
    double scale = (dist > 1e-10) ? (sqrt(2.0) / dist) : 1.0;

    for (int i = 0; i < n; i++) {
        nx[i] = (pts[i].x - cx) * scale;
        ny[i] = (pts[i].y - cy) * scale;
    }
    ni->cx = cx; ni->cy = cy; ni->scale = scale;
}

/* Build 3×3 normalisation matrix T from NormInfo (maps original → normalised). */
static void build_T(const NormInfo *ni, double T[9])
{
    double s = ni->scale;
    T[0] = s;  T[1] = 0; T[2] = -s * ni->cx;
    T[3] = 0;  T[4] = s; T[5] = -s * ni->cy;
    T[6] = 0;  T[7] = 0; T[8] = 1.0;
}

/* 3×3 matrix multiply C = A * B (row-major). */
static void mat33_mul(const double A[9], const double B[9], double C[9])
{
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += A[r*3+k] * B[k*3+c];
            C[r*3+c] = s;
        }
}

/* 3×3 matrix inverse (cofactor method); returns 0 on success, -1 if singular. */
static int mat33_inv(const double M[9], double Minv[9])
{
    double c00 =  M[4]*M[8] - M[5]*M[7];
    double c01 = -(M[3]*M[8] - M[5]*M[6]);
    double c02 =  M[3]*M[7] - M[4]*M[6];
    double det = M[0]*c00 + M[1]*c01 + M[2]*c02;
    if (fabs(det) < 1e-12) return -1;
    double id = 1.0 / det;
    Minv[0] = c00*id; Minv[1] = -(M[1]*M[8]-M[2]*M[7])*id; Minv[2] =  (M[1]*M[5]-M[2]*M[4])*id;
    Minv[3] = c01*id; Minv[4] =  (M[0]*M[8]-M[2]*M[6])*id; Minv[5] = -(M[0]*M[5]-M[2]*M[3])*id;
    Minv[6] = c02*id; Minv[7] = -(M[0]*M[7]-M[1]*M[6])*id; Minv[8] =  (M[0]*M[4]-M[1]*M[3])*id;
    return 0;
}

/* -------------------------------------------------------------------------
 * Jacobi eigendecomposition of a 9×9 symmetric matrix.
 * We only need the eigenvector for the smallest eigenvalue.
 * ------------------------------------------------------------------------- */

#define JACOBI_ITER 100

/*
 * jacobi_min_eigvec — find the eigenvector of the 9×9 symmetric matrix M
 * corresponding to the smallest eigenvalue via Jacobi sweeps.
 *
 * evec[9] is filled with the result on success.
 * Returns 0 on success, -1 if the matrix is degenerate.
 */
static int jacobi_min_eigvec(const double M_in[81], double evec[9])
{
    double M[81], V[81];
    memcpy(M, M_in, 81 * sizeof(double));

    /* V = identity */
    memset(V, 0, 81 * sizeof(double));
    for (int i = 0; i < 9; i++) V[i*9+i] = 1.0;

    for (int sweep = 0; sweep < JACOBI_ITER; sweep++) {
        /* Find off-diagonal element with largest absolute value */
        int p = 0, q = 1;
        double off_max = 0.0;
        for (int i = 0; i < 9; i++) {
            for (int j = i+1; j < 9; j++) {
                double v = fabs(M[i*9+j]);
                if (v > off_max) { off_max = v; p = i; q = j; }
            }
        }
        if (off_max < 1e-15) break;  /* converged */

        /* Compute Jacobi rotation angle */
        double app = M[p*9+p], aqq = M[q*9+q], apq = M[p*9+q];
        double theta = (aqq - app) / (2.0 * apq);
        double t = (theta >= 0) ?
                   1.0 / (theta + sqrt(1.0 + theta*theta)) :
                   1.0 / (theta - sqrt(1.0 + theta*theta));
        double c = 1.0 / sqrt(1.0 + t*t);
        double s = t * c;

        /* Update M: apply Jacobi rotation M' = Gᵀ M G */
        /* Row/col p and q change */
        for (int r = 0; r < 9; r++) {
            if (r == p || r == q) continue;
            double mrp = M[r*9+p], mrq = M[r*9+q];
            M[r*9+p] = M[p*9+r] = c*mrp - s*mrq;
            M[r*9+q] = M[q*9+r] = s*mrp + c*mrq;
        }
        M[p*9+p] = c*c*app - 2*s*c*apq + s*s*aqq;
        M[q*9+q] = s*s*app + 2*s*c*apq + c*c*aqq;
        M[p*9+q] = M[q*9+p] = 0.0;

        /* Accumulate rotation in V */
        for (int r = 0; r < 9; r++) {
            double vrp = V[r*9+p], vrq = V[r*9+q];
            V[r*9+p] = c*vrp - s*vrq;
            V[r*9+q] = s*vrp + c*vrq;
        }
    }

    /* Find index of smallest diagonal element of M (= smallest eigenvalue) */
    int min_idx = 0;
    double min_val = M[0];
    for (int i = 1; i < 9; i++) {
        if (M[i*9+i] < min_val) { min_val = M[i*9+i]; min_idx = i; }
    }

    for (int i = 0; i < 9; i++) evec[i] = V[i*9 + min_idx];
    return 0;
}

/* -------------------------------------------------------------------------
 * DLT homography estimation
 * ------------------------------------------------------------------------- */

DsoError dlt_homography(const StarPos *ref_pts,
                         const StarPos *src_pts,
                         int            n,
                         Homography    *H_out)
{
    if (!ref_pts || !src_pts || !H_out || n < 4) return DSO_ERR_INVALID_ARG;

    /* Allocate working arrays */
    double *nx_r = (double *)malloc((size_t)n * sizeof(double));
    double *ny_r = (double *)malloc((size_t)n * sizeof(double));
    double *nx_s = (double *)malloc((size_t)n * sizeof(double));
    double *ny_s = (double *)malloc((size_t)n * sizeof(double));
    if (!nx_r || !ny_r || !nx_s || !ny_s) {
        free(nx_r); free(ny_r); free(nx_s); free(ny_s);
        return DSO_ERR_ALLOC;
    }

    /* Normalise both point sets */
    NormInfo ni_r, ni_s;
    normalise_points(ref_pts, n, nx_r, ny_r, &ni_r);
    normalise_points(src_pts, n, nx_s, ny_s, &ni_s);

    /*
     * Assemble M = AᵀA (9×9).
     * Each correspondence adds two rows to A; we accumulate the outer products
     * directly into M without materialising the full 2N×9 matrix.
     *
     * Row for correspondence i:
     *   a0 = [-rx, -ry, -1,   0,   0,  0, sx*rx, sx*ry, sx]
     *   a1 = [  0,   0,  0, -rx, -ry, -1, sy*rx, sy*ry, sy]
     */
    double AtA[81];
    memset(AtA, 0, 81 * sizeof(double));

    for (int i = 0; i < n; i++) {
        double rx = nx_r[i], ry = ny_r[i];
        double sx = nx_s[i], sy = ny_s[i];

        /* Row a0 */
        double a0[9] = {-rx, -ry, -1.0, 0, 0, 0, sx*rx, sx*ry, sx};
        /* Row a1 */
        double a1[9] = {0, 0, 0, -rx, -ry, -1.0, sy*rx, sy*ry, sy};

        /* Accumulate aᵀa for both rows */
        for (int r = 0; r < 9; r++) {
            for (int c = r; c < 9; c++) {
                double v = a0[r]*a0[c] + a1[r]*a1[c];
                AtA[r*9+c] += v;
                if (r != c) AtA[c*9+r] += v;
            }
        }
    }

    free(nx_r); free(ny_r); free(nx_s); free(ny_s);

    /* Find null vector = eigenvector of minimum eigenvalue */
    double h_norm[9];
    if (jacobi_min_eigvec(AtA, h_norm) != 0) return DSO_ERR_INVALID_ARG;

    /*
     * De-normalise: H_raw = T_src⁻¹ * H_norm * T_ref
     * where T_ref maps original ref points to normalised, T_src maps original
     * src points to normalised.
     */
    double T_ref[9], T_src[9], T_src_inv[9];
    build_T(&ni_r, T_ref);
    build_T(&ni_s, T_src);
    if (mat33_inv(T_src, T_src_inv) != 0) return DSO_ERR_INVALID_ARG;

    double tmp[9], H_raw[9];
    mat33_mul(T_src_inv, h_norm, tmp);   /* T_src⁻¹ * H_norm */
    mat33_mul(tmp, T_ref, H_raw);        /* T_src⁻¹ * H_norm * T_ref */

    /* Normalise so that H_raw[8] = 1 (canonical scale). */
    double sc = H_raw[8];
    if (fabs(sc) < 1e-12) {
        /* Try normalising by largest element */
        sc = 0.0;
        for (int k = 0; k < 9; k++) if (fabs(H_raw[k]) > fabs(sc)) sc = H_raw[k];
        if (fabs(sc) < 1e-12) return DSO_ERR_INVALID_ARG;
    }
    for (int k = 0; k < 9; k++) H_out->h[k] = H_raw[k] / sc;

    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * RANSAC helpers
 * ------------------------------------------------------------------------- */

/* Compute squared reprojection error: |H*p_ref - p_src|² */
static double reproj_err_sq(const Homography *H, float rx, float ry,
                              float sx, float sy)
{
    const double *h = H->h;
    double qx_h = h[0]*rx + h[1]*ry + h[2];
    double qy_h = h[3]*rx + h[4]*ry + h[5];
    double qw   = h[6]*rx + h[7]*ry + h[8];
    if (fabs(qw) < 1e-12) return 1e18;
    double qx = qx_h / qw - sx;
    double qy = qy_h / qw - sy;
    return qx*qx + qy*qy;
}

/* -------------------------------------------------------------------------
 * Default RANSAC parameters
 * ------------------------------------------------------------------------- */
static const RansacParams RANSAC_DEFAULTS = {
    .max_iters     = 1000,
    .inlier_thresh = 2.0f,
    .match_radius  = 30.0f,
    .confidence    = 0.99f,
    .min_inliers   = 4
};

/* -------------------------------------------------------------------------
 * ransac_compute_homography
 * ------------------------------------------------------------------------- */

/* Correspondence between a reference star and a frame star. */
typedef struct { int ri; int fi; } Match;

DsoError ransac_compute_homography(const StarList     *ref_list,
                                    const StarList     *frm_list,
                                    const RansacParams *params,
                                    Homography         *H_out,
                                    int                *n_inliers_out)
{
    if (!ref_list || !frm_list || !H_out) return DSO_ERR_INVALID_ARG;

    const RansacParams *p = params ? params : &RANSAC_DEFAULTS;

    if (ref_list->n < p->min_inliers || frm_list->n < p->min_inliers)
        return DSO_ERR_STAR_DETECT;

    /* ---- Build candidate match list via nearest-neighbour with ratio test ---- */
    /* RANSAC samples from these candidates; match_radius controls candidate gating. */
    /* At most one match per reference star (loop iterates ref_list->n times). */
    int max_matches = ref_list->n;
    Match *matches = (Match *)malloc((size_t)max_matches * sizeof(Match));
    if (!matches) return DSO_ERR_ALLOC;

    int n_matches = 0;
    float r2 = p->match_radius * p->match_radius;

    for (int ri = 0; ri < ref_list->n; ri++) {
        float rx = ref_list->stars[ri].x, ry = ref_list->stars[ri].y;
        float d1 = r2 + 1.f, d2 = r2 + 2.f;
        int   j1 = -1;

        for (int fi = 0; fi < frm_list->n; fi++) {
            float dx = frm_list->stars[fi].x - rx;
            float dy = frm_list->stars[fi].y - ry;
            float d  = dx*dx + dy*dy;
            if (d < d1) { d2 = d1; d1 = d; j1 = fi; }
            else if (d < d2) { d2 = d; }
        }
        if (j1 < 0 || d1 > r2) continue;  /* no match within radius */
        /* Lowe ratio test: reject if closest and second-closest are similar */
        if (d2 < r2 + 1.f && d1 > 0.f && sqrtf(d1 / d2) > 0.8f) continue;

        matches[n_matches++] = (Match){ri, j1};
    }

    if (n_matches < (p->min_inliers > 4 ? p->min_inliers : 4)) {
        fprintf(stderr, "ransac: only %d candidate match(es) found "
                "(ref=%d stars, frame=%d stars, radius=%.1f px) — need ≥ %d; "
                "increase --match-radius for larger drift\n",
                n_matches, ref_list->n, frm_list->n, (double)p->match_radius,
                (p->min_inliers > 4 ? p->min_inliers : 4));
        free(matches);
        return DSO_ERR_RANSAC;
    }

    /* ---- RANSAC main loop ---- */
    static int call_counter = 0;
    unsigned int seed = (unsigned int)(time(NULL) ^ clock() ^ call_counter++);

    Homography best_H; memset(&best_H, 0, sizeof(best_H));
    int best_inliers = 0;
    int adaptive_max = p->max_iters;
    double thresh_sq = (double)p->inlier_thresh * p->inlier_thresh;

    /* Temporary storage for 4 sampled correspondences */
    StarPos sref[4], sfrm[4];

    for (int iter = 0; iter < adaptive_max; iter++) {
        /* Pick 4 random distinct match indices */
        int idx[4];
        int used[4] = {-1,-1,-1,-1};
        for (int k = 0; k < 4; ) {
            int j = rand_r(&seed) % n_matches;
            int dup = 0;
            for (int m = 0; m < k; m++) if (used[m] == j) { dup = 1; break; }
            if (!dup) { used[k] = j; idx[k] = j; k++; }
        }

        for (int k = 0; k < 4; k++) {
            int ri = matches[idx[k]].ri;
            int fi = matches[idx[k]].fi;
            sref[k] = ref_list->stars[ri];
            sfrm[k] = frm_list->stars[fi];
        }

        Homography H_cand;
        if (dlt_homography(sref, sfrm, 4, &H_cand) != DSO_OK) continue;

        /* Count inliers */
        int count = 0;
        for (int mi = 0; mi < n_matches; mi++) {
            int ri = matches[mi].ri, fi = matches[mi].fi;
            double e2 = reproj_err_sq(&H_cand,
                                       ref_list->stars[ri].x,
                                       ref_list->stars[ri].y,
                                       frm_list->stars[fi].x,
                                       frm_list->stars[fi].y);
            if (e2 < thresh_sq) count++;
        }

        if (count > best_inliers) {
            best_inliers = count;
            best_H = H_cand;

            /* Adaptive termination: update max iterations */
            double inlier_ratio = (double)count / n_matches;
            if (inlier_ratio > 0.999) inlier_ratio = 0.999;
            if (inlier_ratio > 0.001) {
                double p4 = inlier_ratio * inlier_ratio *
                             inlier_ratio * inlier_ratio;
                double n_needed = log(1.0 - p->confidence) / log(1.0 - p4);
                int ni = (int)ceil(n_needed);
                if (ni < adaptive_max) adaptive_max = ni;
                if (adaptive_max < 4) adaptive_max = 4;
            }
        }
    }

    if (best_inliers < p->min_inliers) {
        fprintf(stderr, "ransac: best consensus has only %d inlier(s) "
                "(need ≥ %d, from %d match(es), ref=%d stars, frame=%d stars)\n",
                best_inliers, p->min_inliers, n_matches,
                ref_list->n, frm_list->n);
        free(matches);
        return DSO_ERR_RANSAC;
    }

    /* ---- Refinement: re-estimate H from all inliers ---- */
    int n_inliers = 0;
    StarPos *rpts = (StarPos *)malloc((size_t)best_inliers * sizeof(StarPos));
    StarPos *fpts = (StarPos *)malloc((size_t)best_inliers * sizeof(StarPos));
    if (!rpts || !fpts) {
        free(rpts); free(fpts); free(matches);
        return DSO_ERR_ALLOC;
    }

    for (int mi = 0; mi < n_matches; mi++) {
        int ri = matches[mi].ri, fi = matches[mi].fi;
        double e2 = reproj_err_sq(&best_H,
                                   ref_list->stars[ri].x,
                                   ref_list->stars[ri].y,
                                   frm_list->stars[fi].x,
                                   frm_list->stars[fi].y);
        if (e2 < thresh_sq) {
            rpts[n_inliers] = ref_list->stars[ri];
            fpts[n_inliers] = frm_list->stars[fi];
            n_inliers++;
        }
    }

    DsoError refine_err = dlt_homography(rpts, fpts, n_inliers, H_out);
    free(rpts); free(fpts); free(matches);

    if (refine_err != DSO_OK) return refine_err;
    if (n_inliers_out) *n_inliers_out = n_inliers;
    return DSO_OK;
}
