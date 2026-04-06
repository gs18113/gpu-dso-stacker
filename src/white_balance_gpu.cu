/*
 * white_balance_gpu.cu — GPU white balance for raw Bayer mosaics.
 *
 * Simple 1D-grid kernel: each thread multiplies one pixel by the channel
 * multiplier corresponding to its Bayer position.  Memory-bound; no shared
 * memory optimisation needed.
 */

#include "white_balance_gpu.h"

#include <cuda_runtime.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Device helper: map (x, y) + pattern → channel multiplier.
 *
 * Encodes the 2x2 CFA as a 4-element LUT per pattern:
 *   RGGB: R G / G B   BGGR: B G / G R   GRBG: G R / B G   GBRG: G B / R G
 * ------------------------------------------------------------------------- */
__device__ static float wb_pixel_mul(int x, int y, int pattern,
                                      float r_mul, float g_mul, float b_mul)
{
    int xm = x & 1;
    int ym = y & 1;
    int idx = ym * 2 + xm;

    /* pattern values match BayerPattern enum: 1=RGGB, 2=BGGR, 3=GRBG, 4=GBRG */
    switch (pattern) {
    case 1: { /* RGGB: 0=R 1=G 2=G 3=B */
        const float m[4] = { r_mul, g_mul, g_mul, b_mul };
        return m[idx];
    }
    case 2: { /* BGGR: 0=B 1=G 2=G 3=R */
        const float m[4] = { b_mul, g_mul, g_mul, r_mul };
        return m[idx];
    }
    case 3: { /* GRBG: 0=G 1=R 2=B 3=G */
        const float m[4] = { g_mul, r_mul, b_mul, g_mul };
        return m[idx];
    }
    case 4: { /* GBRG: 0=G 1=B 2=R 3=G */
        const float m[4] = { g_mul, b_mul, r_mul, g_mul };
        return m[idx];
    }
    default:
        return 1.0f;  /* BAYER_NONE — should not reach here */
    }
}

/* -------------------------------------------------------------------------
 * Kernel: per-pixel white balance multiply
 * ------------------------------------------------------------------------- */
__global__ static void wb_apply_kernel(float *data, int N, int W,
                                        int pattern,
                                        float r_mul, float g_mul, float b_mul)
{
    int i = (int)(blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= N) return;

    int x = i % W;
    int y = i / W;

    data[i] *= wb_pixel_mul(x, y, pattern, r_mul, g_mul, b_mul);
}

/* =========================================================================
 * Public API
 * ========================================================================= */

DsoError wb_apply_bayer_gpu_d2d(float *d_data, int W, int H,
                                 BayerPattern pattern,
                                 float r_mul, float g_mul, float b_mul,
                                 cudaStream_t stream)
{
    if (!d_data || W <= 0 || H <= 0)
        return DSO_ERR_INVALID_ARG;

    if (pattern == BAYER_NONE)
        return DSO_OK;   /* monochrome — nothing to do */

    int  N = W * H;
    dim3 block(256);
    dim3 grid((unsigned int)((N + 255) / 256));

    wb_apply_kernel<<<grid, block, 0, stream>>>(
        d_data, N, W, (int)pattern, r_mul, g_mul, b_mul);

    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "wb_apply_kernel launch error: %s\n",
                cudaGetErrorString(ce));
        return DSO_ERR_CUDA;
    }

    return DSO_OK;
}
