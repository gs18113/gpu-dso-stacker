/*
 * transform.c — Polynomial alignment transform fitting and evaluation.
 *
 * Implements bilinear (affine), bisquared (quadratic), and bicubic (cubic)
 * coordinate mappings solved via least-squares normal equations with
 * Cholesky decomposition.
 *
 * Coordinate normalization (centroid + scale to unit mean distance)
 * is applied before fitting for numerical stability, matching the
 * normalisation strategy used for DLT in ransac.c.
 */

#include "transform.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------
 * Internal constants
 * ------------------------------------------------------------------------- */
#define MAX_K 10  /* max coefficients per axis (bicubic) */

/* -------------------------------------------------------------------------
 * transform_ncoeffs_per_axis
 * ------------------------------------------------------------------------- */
int transform_ncoeffs_per_axis(TransformModel model)
{
    switch (model) {
    case TRANSFORM_BILINEAR:  return TRANSFORM_BILINEAR_NCOEFFS;
    case TRANSFORM_BISQUARED: return TRANSFORM_BISQUARED_NCOEFFS;
    case TRANSFORM_BICUBIC:   return TRANSFORM_BICUBIC_NCOEFFS;
    default:                  return 0;
    }
}

/* -------------------------------------------------------------------------
 * transform_eval
 * ------------------------------------------------------------------------- */
void transform_eval(const PolyTransform *T, double dx, double dy,
                    double *sx, double *sy)
{
    const double *c = T->coeffs;
    switch (T->model) {
    case TRANSFORM_BILINEAR:
        *sx = c[0] + c[1]*dx + c[2]*dy;
        *sy = c[3] + c[4]*dx + c[5]*dy;
        break;
    case TRANSFORM_BISQUARED: {
        double dx2 = dx*dx, dxy = dx*dy, dy2 = dy*dy;
        *sx = c[0] + c[1]*dx + c[2]*dy + c[3]*dx2 + c[4]*dxy + c[5]*dy2;
        *sy = c[6] + c[7]*dx + c[8]*dy + c[9]*dx2 + c[10]*dxy + c[11]*dy2;
        break;
    }
    case TRANSFORM_BICUBIC: {
        double dx2 = dx*dx, dy2 = dy*dy, dxy = dx*dy;
        double dx3 = dx2*dx, dy3 = dy2*dy;
        *sx = c[0]  + c[1]*dx  + c[2]*dy  + c[3]*dx2  + c[4]*dxy  + c[5]*dy2
            + c[6]*dx3 + c[7]*dx2*dy + c[8]*dx*dy2 + c[9]*dy3;
        *sy = c[10] + c[11]*dx + c[12]*dy + c[13]*dx2 + c[14]*dxy + c[15]*dy2
            + c[16]*dx3 + c[17]*dx2*dy + c[18]*dx*dy2 + c[19]*dy3;
        break;
    }
    default:
        /* PROJECTIVE / AUTO — caller should use Homography directly */
        *sx = dx;
        *sy = dy;
        break;
    }
}

/* -------------------------------------------------------------------------
 * transform_from_homography
 * ------------------------------------------------------------------------- */
void transform_from_homography(const Homography *H, PolyTransform *out)
{
    memset(out, 0, sizeof(*out));
    out->model = TRANSFORM_PROJECTIVE;
    /* coeffs unused for projective; zero-init is fine */
    (void)H;
}

/* -------------------------------------------------------------------------
 * transform_identity
 * ------------------------------------------------------------------------- */
void transform_identity(TransformModel model, PolyTransform *out)
{
    memset(out, 0, sizeof(*out));
    out->model = model;
    switch (model) {
    case TRANSFORM_BILINEAR:
        /* sx = 0 + 1*dx + 0*dy,  sy = 0 + 0*dx + 1*dy */
        out->coeffs[1] = 1.0;  /* a1 */
        out->coeffs[5] = 1.0;  /* b2 */
        break;
    case TRANSFORM_BISQUARED:
        out->coeffs[1] = 1.0;  /* a1 */
        out->coeffs[8] = 1.0;  /* b2 (index 6+2) */
        break;
    case TRANSFORM_BICUBIC:
        out->coeffs[1]  = 1.0; /* a1 */
        out->coeffs[12] = 1.0; /* b2 (index 10+2) */
        break;
    case TRANSFORM_PROJECTIVE:
        /* Identity homography — coeffs unused */
        break;
    default:
        break;
    }
}

/* -------------------------------------------------------------------------
 * transform_auto_select
 * ------------------------------------------------------------------------- */
TransformModel transform_auto_select(int n_inliers)
{
    if (n_inliers >= TRANSFORM_AUTO_BICUBIC_THRESH)   return TRANSFORM_BICUBIC;
    if (n_inliers >= TRANSFORM_AUTO_BISQUARED_THRESH)  return TRANSFORM_BISQUARED;
    if (n_inliers >= TRANSFORM_AUTO_BILINEAR_THRESH)   return TRANSFORM_BILINEAR;
    return TRANSFORM_PROJECTIVE;
}

/* -------------------------------------------------------------------------
 * transform_reproj_err_sq
 * ------------------------------------------------------------------------- */
double transform_reproj_err_sq(const PolyTransform *T,
                                float rx, float ry,
                                float sx, float sy)
{
    double px, py;
    transform_eval(T, (double)rx, (double)ry, &px, &py);
    double ex = px - (double)sx;
    double ey = py - (double)sy;
    return ex*ex + ey*ey;
}

/* =========================================================================
 * Least-squares polynomial fitting via normal equations + Cholesky
 * =========================================================================
 *
 * Given N correspondences (ref_x_i, ref_y_i) → (src_x_i, src_y_i):
 *
 *   1. Normalise ref and src coordinates (centroid + scale).
 *   2. Build design matrix A (N × K) from ref coordinates.
 *   3. Form normal equations: G = A^T A (K×K), rhs_x = A^T src_x, rhs_y = A^T src_y.
 *   4. Solve G * cx = rhs_x and G * cy = rhs_y via Cholesky.
 *   5. De-normalise coefficients back to pixel coordinates.
 */

/* -------------------------------------------------------------------------
 * build_design_row — fill a K-element row of the design matrix.
 *
 * For normalised reference coordinates (nx, ny), writes the polynomial
 * basis evaluated at (nx, ny) into row[0..K-1].
 * ------------------------------------------------------------------------- */
static void build_design_row(double nx, double ny, TransformModel model,
                              double *row, int K)
{
    (void)K;
    switch (model) {
    case TRANSFORM_BILINEAR:
        row[0] = 1.0;  row[1] = nx;  row[2] = ny;
        break;
    case TRANSFORM_BISQUARED:
        row[0] = 1.0;  row[1] = nx;  row[2] = ny;
        row[3] = nx*nx; row[4] = nx*ny; row[5] = ny*ny;
        break;
    case TRANSFORM_BICUBIC: {
        double nx2 = nx*nx, ny2 = ny*ny;
        row[0] = 1.0;  row[1] = nx;  row[2] = ny;
        row[3] = nx2;  row[4] = nx*ny;  row[5] = ny2;
        row[6] = nx2*nx;  row[7] = nx2*ny;  row[8] = nx*ny2;  row[9] = ny2*ny;
        break;
    }
    default:
        break;
    }
}

/* -------------------------------------------------------------------------
 * Cholesky decomposition and solve (in-place, K×K, row-major).
 *
 * Decomposes G = L L^T.  Returns 0 on success, -1 if not positive definite.
 * After cholesky_decompose, cholesky_solve solves L L^T x = b in-place.
 * ------------------------------------------------------------------------- */
static int cholesky_decompose(double *G, int K)
{
    for (int i = 0; i < K; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = G[i*K + j];
            for (int k = 0; k < j; k++)
                sum -= G[i*K + k] * G[j*K + k];
            if (i == j) {
                if (sum <= 1e-15) return -1;  /* not positive definite */
                G[i*K + j] = sqrt(sum);
            } else {
                G[i*K + j] = sum / G[j*K + j];
            }
        }
        /* Zero upper triangle (L is lower-triangular) */
        for (int j = i + 1; j < K; j++)
            G[i*K + j] = 0.0;
    }
    return 0;
}

static void cholesky_solve(const double *L, int K, double *b)
{
    /* Forward substitution: L y = b */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < i; j++)
            b[i] -= L[i*K + j] * b[j];
        b[i] /= L[i*K + i];
    }
    /* Backward substitution: L^T x = y */
    for (int i = K - 1; i >= 0; i--) {
        for (int j = i + 1; j < K; j++)
            b[i] -= L[j*K + i] * b[j];
        b[i] /= L[i*K + i];
    }
}

/* -------------------------------------------------------------------------
 * Normalisation: centroid subtraction + scale to mean distance sqrt(2).
 * Same approach as normalise_points() in ransac.c.
 * ------------------------------------------------------------------------- */
typedef struct {
    double cx, cy;   /* centroid */
    double scale;    /* scale factor */
} NormInfo;

static NormInfo compute_norm(const StarPos *pts, int n)
{
    NormInfo info = {0.0, 0.0, 1.0};
    double sx = 0.0, sy = 0.0;
    for (int i = 0; i < n; i++) {
        sx += (double)pts[i].x;
        sy += (double)pts[i].y;
    }
    info.cx = sx / n;
    info.cy = sy / n;

    double dist = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = (double)pts[i].x - info.cx;
        double dy = (double)pts[i].y - info.cy;
        dist += sqrt(dx*dx + dy*dy);
    }
    dist /= n;
    if (dist > 1e-12)
        info.scale = 1.41421356237 / dist;  /* sqrt(2) / avg_dist */
    return info;
}

static inline double norm_x(const NormInfo *info, double x)
{
    return (x - info->cx) * info->scale;
}

static inline double norm_y(const NormInfo *info, double y)
{
    return (y - info->cy) * info->scale;
}

/* -------------------------------------------------------------------------
 * De-normalise polynomial coefficients.
 *
 * The fitting solves in normalised coordinates:
 *   s_norm = P(r_norm)   where  r_norm = s_r * (r - c_r),
 *                                s_norm = s_s * (s - c_s)
 *
 * We need pixel-space coefficients: s_pixel = P_pixel(r_pixel).
 *
 * Strategy: for each polynomial term, expand the normalised monomials
 * in terms of pixel coordinates using the substitution
 *   nx = sr*(rx - crx),  ny = sr*(ry - cry)
 * Then invert the source normalisation:
 *   s_pixel = s_norm / s_s + c_s
 *
 * Rather than algebraic expansion (complex for cubic), we use a
 * numerical approach: evaluate the normalised polynomial at enough
 * pixel-space points and re-fit in pixel space (identity normalisation).
 * This is exact for polynomials since the basis spans all terms.
 * ------------------------------------------------------------------------- */
static DsoError denormalise_coeffs(const double *cx_norm, const double *cy_norm,
                                    int K, TransformModel model,
                                    const NormInfo *ref_norm,
                                    const NormInfo *src_norm,
                                    double *cx_pixel, double *cy_pixel)
{
    /*
     * Generate K sample points spread across the normalised range,
     * evaluate the normalised polynomial, convert both input and output
     * back to pixel space, then fit in pixel space with no normalisation.
     *
     * We need at least K points; use 2*K for robustness (still exact
     * for polynomial fitting with the right basis).
     */
    int N = K * 3;
    if (N < 10) N = 10;

    double *A  = (double *)calloc((size_t)N * K, sizeof(double));
    double *bx = (double *)malloc((size_t)N * sizeof(double));
    double *by = (double *)malloc((size_t)N * sizeof(double));
    if (!A || !bx || !by) { free(A); free(bx); free(by); return DSO_ERR_ALLOC; }

    /* Generate sample points in pixel space on a grid covering the range */
    double inv_sr = 1.0 / ref_norm->scale;
    double inv_ss = 1.0 / src_norm->scale;
    int idx = 0;
    /* Use a grid in normalised ref space [-2, 2] × [-2, 2] */
    int side = (int)ceil(sqrt((double)N));
    for (int gi = 0; gi < side && idx < N; gi++) {
        for (int gj = 0; gj < side && idx < N; gj++) {
            double nr = -2.0 + 4.0 * gi / (side - 1);
            double nc = -2.0 + 4.0 * gj / (side - 1);

            /* Pixel-space ref coordinates */
            double px = nr * inv_sr + ref_norm->cx;
            double py = nc * inv_sr + ref_norm->cy;

            /* Evaluate normalised polynomial */
            double snx = 0.0, sny = 0.0;
            double row[MAX_K];
            build_design_row(nr, nc, model, row, K);
            for (int k = 0; k < K; k++) {
                snx += cx_norm[k] * row[k];
                sny += cy_norm[k] * row[k];
            }

            /* Convert source back to pixel space */
            bx[idx] = snx * inv_ss + src_norm->cx;
            by[idx] = sny * inv_ss + src_norm->cy;

            /* Build design row in pixel space (no normalisation) */
            build_design_row(px, py, model, &A[idx * K], K);
            idx++;
        }
    }
    N = idx;

    /* Form normal equations in pixel space: G = A^T A, rhs = A^T b */
    double G[MAX_K * MAX_K];
    double rhsx[MAX_K], rhsy[MAX_K];
    memset(G, 0, sizeof(double) * K * K);
    memset(rhsx, 0, sizeof(double) * K);
    memset(rhsy, 0, sizeof(double) * K);

    for (int i = 0; i < N; i++) {
        const double *ai = &A[i * K];
        for (int r = 0; r < K; r++) {
            for (int c = 0; c < K; c++)
                G[r*K + c] += ai[r] * ai[c];
            rhsx[r] += ai[r] * bx[i];
            rhsy[r] += ai[r] * by[i];
        }
    }

    free(A); free(bx); free(by);

    /* Cholesky solve */
    double G2[MAX_K * MAX_K];
    memcpy(G2, G, sizeof(double) * K * K);

    if (cholesky_decompose(G, K) != 0) return DSO_ERR_INVALID_ARG;
    cholesky_solve(G, K, rhsx);
    memcpy(cx_pixel, rhsx, sizeof(double) * K);

    /* Solve again for y with fresh copy of G */
    if (cholesky_decompose(G2, K) != 0) return DSO_ERR_INVALID_ARG;
    cholesky_solve(G2, K, rhsy);
    memcpy(cy_pixel, rhsy, sizeof(double) * K);

    return DSO_OK;
}

/* -------------------------------------------------------------------------
 * transform_fit
 * ------------------------------------------------------------------------- */
DsoError transform_fit(const StarPos *ref_pts, const StarPos *src_pts,
                        int n, TransformModel model, PolyTransform *out)
{
    if (!ref_pts || !src_pts || !out) return DSO_ERR_INVALID_ARG;
    if (model == TRANSFORM_PROJECTIVE || model == TRANSFORM_AUTO)
        return DSO_ERR_INVALID_ARG;

    int K = transform_ncoeffs_per_axis(model);
    if (K == 0) return DSO_ERR_INVALID_ARG;

    /* Check minimum point count */
    int min_pts;
    switch (model) {
    case TRANSFORM_BILINEAR:  min_pts = TRANSFORM_BILINEAR_MIN_PTS;  break;
    case TRANSFORM_BISQUARED: min_pts = TRANSFORM_BISQUARED_MIN_PTS; break;
    case TRANSFORM_BICUBIC:   min_pts = TRANSFORM_BICUBIC_MIN_PTS;   break;
    default:                  return DSO_ERR_INVALID_ARG;
    }
    if (n < min_pts) return DSO_ERR_INVALID_ARG;

    /* Normalise coordinates */
    NormInfo ref_norm = compute_norm(ref_pts, n);
    NormInfo src_norm = compute_norm(src_pts, n);

    /* Build normal equations: G = A^T A (K×K), rhsx/rhsy = A^T b */
    double G[MAX_K * MAX_K];
    double rhsx[MAX_K], rhsy[MAX_K];
    memset(G, 0, sizeof(double) * K * K);
    memset(rhsx, 0, sizeof(double) * K);
    memset(rhsy, 0, sizeof(double) * K);

    double row[MAX_K];
    for (int i = 0; i < n; i++) {
        double nx = norm_x(&ref_norm, (double)ref_pts[i].x);
        double ny = norm_y(&ref_norm, (double)ref_pts[i].y);
        double sx = norm_x(&src_norm, (double)src_pts[i].x);
        double sy = norm_y(&src_norm, (double)src_pts[i].y);

        build_design_row(nx, ny, model, row, K);

        /* Accumulate A^T A and A^T b */
        for (int r = 0; r < K; r++) {
            for (int c = 0; c < K; c++)
                G[r*K + c] += row[r] * row[c];
            rhsx[r] += row[r] * sx;
            rhsy[r] += row[r] * sy;
        }
    }

    /* Cholesky decomposition and solve */
    double G2[MAX_K * MAX_K];
    memcpy(G2, G, sizeof(double) * K * K);

    if (cholesky_decompose(G, K) != 0) {
        fprintf(stderr, "transform_fit: singular normal matrix "
                "(collinear or insufficient points)\n");
        return DSO_ERR_INVALID_ARG;
    }
    cholesky_solve(G, K, rhsx);

    if (cholesky_decompose(G2, K) != 0)
        return DSO_ERR_INVALID_ARG;
    cholesky_solve(G2, K, rhsy);

    /* De-normalise coefficients to pixel space */
    double cx_pixel[MAX_K], cy_pixel[MAX_K];
    DsoError derr = denormalise_coeffs(rhsx, rhsy, K, model,
                                        &ref_norm, &src_norm,
                                        cx_pixel, cy_pixel);
    if (derr != DSO_OK) return derr;

    /* Pack into PolyTransform */
    memset(out, 0, sizeof(*out));
    out->model = model;
    memcpy(&out->coeffs[0], cx_pixel, sizeof(double) * K);
    memcpy(&out->coeffs[K], cy_pixel, sizeof(double) * K);

    return DSO_OK;
}
