/* -------------------------------------------------------------------------
 * centroid_lm_gpu.cu — GPU Levenberg-Marquardt 2D Gaussian centroid refinement
 *
 * Warp-per-star architecture: 32 threads cooperate on each star via
 * __shfl_down_sync for JtJ/Jtr reductions, lane 0 solves the 5x5 Cholesky.
 * All LM internal math in double precision.
 * ------------------------------------------------------------------------- */
#include "centroid_lm_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Star fit input/output structures for H2D / D2H transfers
 * ------------------------------------------------------------------------- */
struct StarFitInput {
    float com_x, com_y;   /* initial CoM position */
    float flux;           /* original flux (passed through) */
};

struct StarFitOutput {
    float x, y;           /* refined centroid */
    int   converged;      /* 1 if LM converged */
};

/* -------------------------------------------------------------------------
 * Device: warp-level double-precision sum reduction
 * ------------------------------------------------------------------------- */
__device__ static double warp_reduce_sum_d(double val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/* -------------------------------------------------------------------------
 * Device: 5x5 Cholesky decomposition + solve (lane 0 only)
 * A is 5x5 row-major SPD, b is 5x1, x is 5x1 output.
 * Returns 0 on success, -1 if not positive definite.
 * ------------------------------------------------------------------------- */
__device__ static int cholesky_solve_5x5_device(double A[25],
                                                  const double b[5],
                                                  double x[5])
{
    double L[25];
    for (int i = 0; i < 25; i++) L[i] = 0.0;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i * 5 + j];
            for (int k = 0; k < j; k++)
                sum -= L[i * 5 + k] * L[j * 5 + k];
            if (i == j) {
                if (sum <= 1e-30) return -1;
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
        for (int k = 0; k < i; k++) sum -= L[i * 5 + k] * y[k];
        y[i] = sum / L[i * 5 + i];
    }

    /* Back substitution: L^T * x = y */
    for (int i = 4; i >= 0; i--) {
        double sum = y[i];
        for (int k = i + 1; k < 5; k++) sum -= L[k * 5 + i] * x[k];
        x[i] = sum / L[i * 5 + i];
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * GPU kernel: one warp (32 threads) per star
 * Launch: <<<ceil(n_stars / warps_per_block), threads_per_block>>>
 * with threads_per_block = warps_per_block * 32
 * ------------------------------------------------------------------------- */
__global__ void lm_fit_warp_kernel(
    const float       *d_image,
    int                W, int H,
    const StarFitInput *d_inputs,
    StarFitOutput      *d_outputs,
    int                n_stars,
    float              sigma_init_f,
    int                R,       /* fitting window half-size */
    int                max_iter)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id >= n_stars) return;

    StarFitInput inp = d_inputs[warp_id];

    /* Fitting window */
    int cx_i = (int)floorf(inp.com_x);
    int cy_i = (int)floorf(inp.com_y);
    int x0 = max(0, cx_i - R);
    int y0 = max(0, cy_i - R);
    int x1 = min(W - 1, cx_i + R);
    int y1 = min(H - 1, cy_i + R);
    int fw = x1 - x0 + 1;
    int fh = y1 - y0 + 1;
    int npix = fw * fh;

    if (npix < 6) {
        if (lane == 0) {
            d_outputs[warp_id].x = inp.com_x;
            d_outputs[warp_id].y = inp.com_y;
            d_outputs[warp_id].converged = 0;
        }
        return;
    }

    double sigma_init = (double)sigma_init_f;

    /* --- Cooperative initial guess computation --- */

    /* Background: average of border pixels (cooperative) */
    double local_border_sum = 0.0;
    int local_border_n = 0;
    int border_count = 2 * fw + 2 * max(0, fh - 2);
    /* Enumerate border pixels: top row, bottom row, left/right cols */
    for (int p = lane; p < border_count; p += 32) {
        int gx, gy;
        if (p < fw) {
            /* top row */
            gx = x0 + p;
            gy = y0;
        } else if (p < 2 * fw) {
            /* bottom row */
            gx = x0 + (p - fw);
            gy = y1;
        } else {
            /* left/right columns (excluding corners) */
            int side_idx = p - 2 * fw;
            int row = side_idx / 2 + 1;  /* row offset from y0 */
            if (side_idx % 2 == 0) {
                gx = x0;
            } else {
                gx = x1;
            }
            gy = y0 + row;
        }
        if (gx >= 0 && gx < W && gy >= 0 && gy < H && gy >= y0 && gy <= y1) {
            local_border_sum += (double)d_image[gy * W + gx];
            local_border_n++;
        }
    }
    local_border_sum = warp_reduce_sum_d(local_border_sum);
    double border_n_d = warp_reduce_sum_d((double)local_border_n);
    double B_init = (border_n_d > 0.5) ? (local_border_sum / border_n_d) : 0.0;
    B_init = __shfl_sync(0xFFFFFFFF, B_init, 0);

    /* Peak: max pixel in window (cooperative) */
    double local_max = -1e30;
    for (int p = lane; p < npix; p += 32) {
        int ix = x0 + (p % fw);
        int iy = y0 + (p / fw);
        double v = (double)d_image[iy * W + ix];
        if (v > local_max) local_max = v;
    }
    /* Warp max reduction */
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    double A_init = __shfl_sync(0xFFFFFFFF, local_max, 0) - B_init;
    if (A_init < 1e-10) A_init = 1.0;

    /* Parameters: all lanes hold the same copy */
    double cx = (double)inp.com_x;
    double cy = (double)inp.com_y;
    double A  = A_init;
    double sigma = sigma_init;
    double B  = B_init;
    double lambda = 1e-3;
    int consecutive_rejects = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Each thread accumulates partial JtJ (15 upper-tri) and Jtr (5) */
        double local_JtJ[15];
        double local_Jtr[5];
        double local_cost = 0.0;
        for (int i = 0; i < 15; i++) local_JtJ[i] = 0.0;
        for (int i = 0; i < 5; i++)  local_Jtr[i]  = 0.0;

        double s2 = sigma * sigma;

        /* Distribute pixels across 32 lanes */
        for (int p = lane; p < npix; p += 32) {
            int ix = x0 + (p % fw);
            int iy = y0 + (p / fw);
            double px = (double)ix;
            double py = (double)iy;
            double dx = px - cx;
            double dy = py - cy;
            double d2 = dx * dx + dy * dy;
            double e  = exp(-d2 / (2.0 * s2));
            double model = A * e + B;
            double data_val = (double)d_image[iy * W + ix];
            double r = model - data_val;
            local_cost += r * r;

            double J[5];
            J[0] = A * e * dx / s2;            /* dF/dx0 */
            J[1] = A * e * dy / s2;            /* dF/dy0 */
            J[2] = e;                           /* dF/dA  */
            J[3] = A * e * d2 / (s2 * sigma);  /* dF/dsigma */
            J[4] = 1.0;                         /* dF/dB  */

            /* Accumulate upper triangle of JtJ (15 elements) */
            int idx = 0;
            for (int a = 0; a < 5; a++) {
                local_Jtr[a] += J[a] * r;
                for (int b = a; b < 5; b++)
                    local_JtJ[idx++] += J[a] * J[b];
            }
        }

        /* Warp reduction: sum JtJ, Jtr, cost */
        for (int i = 0; i < 15; i++)
            local_JtJ[i] = warp_reduce_sum_d(local_JtJ[i]);
        for (int i = 0; i < 5; i++)
            local_Jtr[i] = warp_reduce_sum_d(local_Jtr[i]);
        local_cost = warp_reduce_sum_d(local_cost);

        /* Lane 0: solve normal equations */
        double delta[5] = {0, 0, 0, 0, 0};
        int solve_ok = 0;
        int accept = 0;
        int done = 0;

        if (lane == 0) {
            /* Expand upper triangle to full 5x5 */
            double M[25];
            int idx = 0;
            for (int a = 0; a < 5; a++) {
                for (int b = a; b < 5; b++) {
                    M[a * 5 + b] = M[b * 5 + a] = local_JtJ[idx++];
                }
                /* LM damping */
                M[a * 5 + a] *= (1.0 + lambda);
            }

            solve_ok = (cholesky_solve_5x5_device(M, local_Jtr, delta) == 0);

            if (!solve_ok) {
                lambda *= 10.0;
                consecutive_rejects++;
                done = (consecutive_rejects >= 5 || lambda > 1e10);
            }
        }

        solve_ok = __shfl_sync(0xFFFFFFFF, solve_ok, 0);
        done     = __shfl_sync(0xFFFFFFFF, done, 0);

        if (done) break;
        if (!solve_ok) {
            lambda = __shfl_sync(0xFFFFFFFF, lambda, 0);
            continue;
        }

        /* Broadcast delta from lane 0 */
        for (int i = 0; i < 5; i++)
            delta[i] = __shfl_sync(0xFFFFFFFF, delta[i], 0);

        /* Trial step */
        double cx_new    = cx    - delta[0];
        double cy_new    = cy    - delta[1];
        double A_new     = A     - delta[2];
        double sigma_new = sigma - delta[3];
        double B_new     = B     - delta[4];

        /* Bounds check (lane 0 decides, broadcast) */
        int bounds_ok = 1;
        if (lane == 0) {
            if (sigma_new < 0.3 || sigma_new > 20.0 || A_new < 0.0)
                bounds_ok = 0;
            if (cx_new < (double)x0 || cx_new > (double)x1 ||
                cy_new < (double)y0 || cy_new > (double)y1)
                bounds_ok = 0;
        }
        bounds_ok = __shfl_sync(0xFFFFFFFF, bounds_ok, 0);

        if (!bounds_ok) {
            if (lane == 0) {
                lambda *= 10.0;
                consecutive_rejects++;
            }
            lambda = __shfl_sync(0xFFFFFFFF, lambda, 0);
            consecutive_rejects = __shfl_sync(0xFFFFFFFF, consecutive_rejects, 0);
            if (consecutive_rejects >= 5 || lambda > 1e10) break;
            continue;
        }

        /* Compute trial cost (parallel across lanes) */
        double new_cost = 0.0;
        double s2_new = sigma_new * sigma_new;
        for (int p = lane; p < npix; p += 32) {
            int ix = x0 + (p % fw);
            int iy = y0 + (p / fw);
            double dx = (double)ix - cx_new;
            double dy = (double)iy - cy_new;
            double e = exp(-(dx * dx + dy * dy) / (2.0 * s2_new));
            double r = A_new * e + B_new - (double)d_image[iy * W + ix];
            new_cost += r * r;
        }
        new_cost = warp_reduce_sum_d(new_cost);

        /* Lane 0 decides accept/reject */
        if (lane == 0) {
            if (new_cost < local_cost) {
                cx = cx_new;
                cy = cy_new;
                A  = A_new;
                sigma = sigma_new;
                B  = B_new;
                lambda *= 0.1;
                if (lambda < 1e-10) lambda = 1e-10;
                consecutive_rejects = 0;
                accept = 1;
                if (fabs(delta[0]) + fabs(delta[1]) < 1e-4)
                    done = 1;
            } else {
                lambda *= 10.0;
                consecutive_rejects++;
                if (consecutive_rejects >= 5 || lambda > 1e10)
                    done = 1;
            }
        }

        /* Broadcast everything */
        accept = __shfl_sync(0xFFFFFFFF, accept, 0);
        done   = __shfl_sync(0xFFFFFFFF, done, 0);
        cx     = __shfl_sync(0xFFFFFFFF, cx, 0);
        cy     = __shfl_sync(0xFFFFFFFF, cy, 0);
        A      = __shfl_sync(0xFFFFFFFF, A, 0);
        sigma  = __shfl_sync(0xFFFFFFFF, sigma, 0);
        B      = __shfl_sync(0xFFFFFFFF, B, 0);
        lambda = __shfl_sync(0xFFFFFFFF, lambda, 0);
        consecutive_rejects = __shfl_sync(0xFFFFFFFF, consecutive_rejects, 0);

        if (done) break;
    }

    /* Lane 0 writes output */
    if (lane == 0) {
        double dist = fabs(cx - (double)inp.com_x) + fabs(cy - (double)inp.com_y);
        int conv = (dist > 1e-6) ? 1 : 0;
        d_outputs[warp_id].x = (float)cx;
        d_outputs[warp_id].y = (float)cy;
        d_outputs[warp_id].converged = conv;
    }
}

/* -------------------------------------------------------------------------
 * Host API
 * ------------------------------------------------------------------------- */

DsoError centroid_lm_refine_gpu(StarList *stars, const float *d_image,
                                 int W, int H, float sigma_init,
                                 float fit_radius, int max_iter,
                                 cudaStream_t stream)
{
    if (!stars || !d_image || W <= 0 || H <= 0)
        return DSO_ERR_INVALID_ARG;

    if (stars->n == 0 || !stars->stars)
        return DSO_OK;

    int n = stars->n;
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

    /* Prepare host input array */
    StarFitInput *h_inputs = (StarFitInput *)malloc(n * sizeof(StarFitInput));
    if (!h_inputs) return DSO_ERR_ALLOC;

    for (int i = 0; i < n; i++) {
        h_inputs[i].com_x = stars->stars[i].x;
        h_inputs[i].com_y = stars->stars[i].y;
        h_inputs[i].flux  = stars->stars[i].flux;
    }

    /* Device allocations */
    StarFitInput  *d_inputs  = NULL;
    StarFitOutput *d_outputs = NULL;
    StarFitOutput *h_outputs = NULL;
    cudaError_t ce;

    ce = cudaMalloc(&d_inputs, n * sizeof(StarFitInput));
    if (ce != cudaSuccess) { free(h_inputs); return DSO_ERR_CUDA; }

    ce = cudaMalloc(&d_outputs, n * sizeof(StarFitOutput));
    if (ce != cudaSuccess) {
        cudaFree(d_inputs);
        free(h_inputs);
        return DSO_ERR_CUDA;
    }

    h_outputs = (StarFitOutput *)malloc(n * sizeof(StarFitOutput));
    if (!h_outputs) {
        cudaFree(d_outputs);
        cudaFree(d_inputs);
        free(h_inputs);
        return DSO_ERR_ALLOC;
    }

    /* H2D */
    ce = cudaMemcpyAsync(d_inputs, h_inputs, n * sizeof(StarFitInput),
                          cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) goto cuda_fail;

    /* Launch kernel: 8 warps per block = 256 threads */
    {
        int warps_per_block = 8;
        int threads_per_block = warps_per_block * 32;  /* 256 */
        int n_blocks = (n + warps_per_block - 1) / warps_per_block;

        lm_fit_warp_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
            d_image, W, H, d_inputs, d_outputs, n,
            sigma_init, R, max_iter);

        ce = cudaGetLastError();
        if (ce != cudaSuccess) goto cuda_fail;
    }

    /* D2H */
    ce = cudaMemcpyAsync(h_outputs, d_outputs, n * sizeof(StarFitOutput),
                          cudaMemcpyDeviceToHost, stream);
    if (ce != cudaSuccess) goto cuda_fail;

    /* Sync to ensure D2H complete */
    ce = cudaStreamSynchronize(stream);
    if (ce != cudaSuccess) goto cuda_fail;

    /* Update star positions from fitted results */
    for (int i = 0; i < n; i++) {
        if (h_outputs[i].converged) {
            stars->stars[i].x = h_outputs[i].x;
            stars->stars[i].y = h_outputs[i].y;
        }
    }

    cudaFree(d_outputs);
    cudaFree(d_inputs);
    free(h_outputs);
    free(h_inputs);
    return DSO_OK;

cuda_fail:
    fprintf(stderr, "centroid_lm_gpu: CUDA error: %s\n",
            cudaGetErrorString(ce));
    cudaFree(d_outputs);
    cudaFree(d_inputs);
    free(h_outputs);
    free(h_inputs);
    return DSO_ERR_CUDA;
}
