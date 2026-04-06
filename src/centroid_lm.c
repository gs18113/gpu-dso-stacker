/* -------------------------------------------------------------------------
 * centroid_lm.c — Levenberg-Marquardt 2D Gaussian centroid refinement (CPU)
 *
 * Fits a 5-parameter circular Gaussian to each detected star blob:
 *   f(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2)) + B
 *
 * Parameters: p = (x0, y0, A, sigma, B)
 * Analytical Jacobian, 5x5 Cholesky solver, all in double precision.
 * OpenMP-parallelized over stars.
 * ------------------------------------------------------------------------- */
#include "centroid_lm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* -------------------------------------------------------------------------
 * 5x5 Cholesky decomposition + solve: A * x = b
 * A is 5x5 symmetric positive definite (row-major, modified in-place).
 * Returns 0 on success, -1 if not positive definite.
 * ------------------------------------------------------------------------- */
static int cholesky_solve_5x5(double A[25], const double b[5], double x[5])
{
    /* Cholesky: A = L * L^T */
    double L[25];
    memset(L, 0, sizeof(L));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i * 5 + j];
            for (int k = 0; k < j; k++)
                sum -= L[i * 5 + k] * L[j * 5 + k];
            if (i == j) {
                if (sum <= 1e-30) return -1;  /* not positive definite */
                L[i * 5 + j] = sqrt(sum);
            } else {
                L[i * 5 + j] = sum / L[j * 5 + j];
            }
        }
    }

    /* Forward substitution: L * y = b */
    double y[5];
    for (int i = 0; i < 5; i++) {
        double sum = b[i];
        for (int k = 0; k < i; k++)
            sum -= L[i * 5 + k] * y[k];
        y[i] = sum / L[i * 5 + i];
    }

    /* Back substitution: L^T * x = y */
    for (int i = 4; i >= 0; i--) {
        double sum = y[i];
        for (int k = i + 1; k < 5; k++)
            sum -= L[k * 5 + i] * x[k];
        x[i] = sum / L[i * 5 + i];
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Compute border median for initial background estimate.
 * Samples the outermost ring of pixels in the fitting window.
 * ------------------------------------------------------------------------- */
static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

static float border_median(const float *image, int W, int H,
                            int x0, int y0, int x1, int y1)
{
    int cap = 2 * (x1 - x0 + 1) + 2 * (y1 - y0 - 1);
    if (cap <= 0) return 0.0f;

    float *buf = (float *)malloc((size_t)cap * sizeof(float));
    if (!buf) return 0.0f;

    int n = 0;
    /* Top and bottom rows */
    for (int x = x0; x <= x1; x++) {
        if (y0 >= 0 && y0 < H && x >= 0 && x < W)
            buf[n++] = image[y0 * W + x];
        if (y1 != y0 && y1 >= 0 && y1 < H && x >= 0 && x < W)
            buf[n++] = image[y1 * W + x];
    }
    /* Left and right columns (excluding corners already counted) */
    for (int y = y0 + 1; y < y1; y++) {
        if (y >= 0 && y < H) {
            if (x0 >= 0 && x0 < W)
                buf[n++] = image[y * W + x0];
            if (x1 != x0 && x1 >= 0 && x1 < W)
                buf[n++] = image[y * W + x1];
        }
    }

    if (n == 0) { free(buf); return 0.0f; }

    qsort(buf, (size_t)n, sizeof(float), cmp_float);
    float med = (n % 2 == 1) ? buf[n / 2]
                              : 0.5f * (buf[n / 2 - 1] + buf[n / 2]);
    free(buf);
    return med;
}

/* -------------------------------------------------------------------------
 * LM fit for one star. Returns 1 if converged, 0 if not.
 * On convergence, *out_x and *out_y are updated with refined centroid.
 * ------------------------------------------------------------------------- */
static int lm_fit_one_star(const float *image, int W, int H,
                            float com_x, float com_y,
                            float sigma_init, int R, int max_iter,
                            float *out_x, float *out_y)
{
    /* Fitting window */
    int x0 = (int)floorf(com_x) - R;
    int y0 = (int)floorf(com_y) - R;
    int x1 = (int)floorf(com_x) + R;
    int y1 = (int)floorf(com_y) + R;

    /* Clamp to image bounds */
    if (x0 < 0)     x0 = 0;
    if (y0 < 0)     y0 = 0;
    if (x1 >= W)    x1 = W - 1;
    if (y1 >= H)    y1 = H - 1;

    int fw = x1 - x0 + 1;
    int fh = y1 - y0 + 1;
    int npix = fw * fh;

    if (npix < 6) return 0;  /* need more pixels than parameters */

    /* Initial background estimate from border median */
    double B = (double)border_median(image, W, H, x0, y0, x1, y1);

    /* Peak amplitude: max pixel in window minus background */
    double max_val = -1e30;
    for (int iy = y0; iy <= y1; iy++)
        for (int ix = x0; ix <= x1; ix++) {
            double v = (double)image[iy * W + ix];
            if (v > max_val) max_val = v;
        }
    double A = max_val - B;
    if (A < 1e-10) A = 1.0;  /* guard against flat regions */

    double cx = (double)com_x;
    double cy = (double)com_y;
    double sigma = (double)sigma_init;

    double lambda = 1e-3;
    double prev_cost = 1e30;
    int consecutive_rejects = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Accumulate J^T*J (5x5) and J^T*r (5x1) */
        double JtJ[25];
        double Jtr[5];
        double cost = 0.0;
        memset(JtJ, 0, sizeof(JtJ));
        memset(Jtr, 0, sizeof(Jtr));

        double s2 = sigma * sigma;

        for (int iy = y0; iy <= y1; iy++) {
            for (int ix = x0; ix <= x1; ix++) {
                double px = (double)ix;
                double py = (double)iy;
                double dx = px - cx;
                double dy = py - cy;
                double d2 = dx * dx + dy * dy;
                double e  = exp(-d2 / (2.0 * s2));
                double model = A * e + B;
                double data_val = (double)image[iy * W + ix];
                double r = model - data_val;
                cost += r * r;

                /* Jacobian: [dF/dx0, dF/dy0, dF/dA, dF/dsigma, dF/dB] */
                double J[5];
                J[0] = A * e * dx / s2;            /* dF/dx0 */
                J[1] = A * e * dy / s2;            /* dF/dy0 */
                J[2] = e;                           /* dF/dA  */
                J[3] = A * e * d2 / (s2 * sigma);  /* dF/dsigma */
                J[4] = 1.0;                         /* dF/dB  */

                /* Accumulate upper triangle of J^T*J and J^T*r */
                for (int a = 0; a < 5; a++) {
                    Jtr[a] += J[a] * r;
                    for (int b = a; b < 5; b++)
                        JtJ[a * 5 + b] += J[a] * J[b];
                }
            }
        }

        /* Fill symmetric lower triangle */
        for (int a = 0; a < 5; a++)
            for (int b = 0; b < a; b++)
                JtJ[a * 5 + b] = JtJ[b * 5 + a];

        /* Add LM damping: H_ii *= (1 + lambda) */
        for (int a = 0; a < 5; a++)
            JtJ[a * 5 + a] *= (1.0 + lambda);

        /* Solve (J^T*J + lambda*diag) * delta = J^T*r */
        double delta[5];
        if (cholesky_solve_5x5(JtJ, Jtr, delta) != 0) {
            /* Singular matrix — increase damping and retry */
            lambda *= 10.0;
            consecutive_rejects++;
            if (consecutive_rejects >= 5 || lambda > 1e10) break;
            continue;
        }

        /* Trial step: p_new = p - delta */
        double cx_new    = cx    - delta[0];
        double cy_new    = cy    - delta[1];
        double A_new     = A     - delta[2];
        double sigma_new = sigma - delta[3];
        double B_new     = B     - delta[4];

        /* Parameter bounds */
        if (sigma_new < 0.3 || sigma_new > 20.0 || A_new < 0.0) {
            lambda *= 10.0;
            consecutive_rejects++;
            if (consecutive_rejects >= 5 || lambda > 1e10) break;
            continue;
        }

        /* Clamp center to within fitting window */
        if (cx_new < (double)x0 || cx_new > (double)x1 ||
            cy_new < (double)y0 || cy_new > (double)y1) {
            lambda *= 10.0;
            consecutive_rejects++;
            if (consecutive_rejects >= 5 || lambda > 1e10) break;
            continue;
        }

        /* Compute new cost */
        double new_cost = 0.0;
        double s2_new = sigma_new * sigma_new;
        for (int iy = y0; iy <= y1; iy++) {
            for (int ix = x0; ix <= x1; ix++) {
                double dx = (double)ix - cx_new;
                double dy = (double)iy - cy_new;
                double e = exp(-(dx * dx + dy * dy) / (2.0 * s2_new));
                double r = A_new * e + B_new - (double)image[iy * W + ix];
                new_cost += r * r;
            }
        }

        if (new_cost < cost) {
            cx    = cx_new;
            cy    = cy_new;
            A     = A_new;
            sigma = sigma_new;
            B     = B_new;
            lambda *= 0.1;
            if (lambda < 1e-10) lambda = 1e-10;
            consecutive_rejects = 0;

            /* Convergence check: position change < 1e-4 px */
            if (fabs(delta[0]) + fabs(delta[1]) < 1e-4) {
                *out_x = (float)cx;
                *out_y = (float)cy;
                return 1;
            }
            prev_cost = new_cost;
        } else {
            lambda *= 10.0;
            consecutive_rejects++;
            if (consecutive_rejects >= 5 || lambda > 1e10) break;
        }
    }

    /* Check if we improved at all vs. starting position */
    /* Accept if the center moved and cost improved from initial */
    double final_dist = fabs(cx - (double)com_x) + fabs(cy - (double)com_y);
    if (final_dist > 1e-6 && prev_cost < 1e29) {
        *out_x = (float)cx;
        *out_y = (float)cy;
        return 1;
    }

    return 0;  /* not converged — keep CoM */
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

DsoError centroid_lm_refine(StarList *stars, const float *image,
                             int W, int H, float sigma_init,
                             float fit_radius, int max_iter)
{
    if (!stars || !image || W <= 0 || H <= 0)
        return DSO_ERR_INVALID_ARG;

    if (stars->n == 0 || !stars->stars)
        return DSO_OK;

    if (sigma_init <= 0.0f) sigma_init = 2.0f;
    if (max_iter <= 0) max_iter = 15;

    int R;
    if (fit_radius > 0.0f) {
        R = (int)ceilf(fit_radius);
    } else {
        R = (int)ceilf(3.0f * sigma_init);
    }
    if (R < 3)  R = 3;
    if (R > 15) R = 15;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int s = 0; s < stars->n; s++) {
        float out_x = stars->stars[s].x;
        float out_y = stars->stars[s].y;

        int ok = lm_fit_one_star(image, W, H,
                                  stars->stars[s].x, stars->stars[s].y,
                                  sigma_init, R, max_iter,
                                  &out_x, &out_y);
        if (ok) {
            stars->stars[s].x = out_x;
            stars->stars[s].y = out_y;
        }
        /* If !ok, star keeps its original CoM position */
    }

    return DSO_OK;
}
