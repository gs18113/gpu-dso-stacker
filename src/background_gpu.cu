/*
 * background_gpu.cu — GPU per-frame background normalization kernel.
 */

#include "background_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * CUDA kernel: per-pixel affine transform, NaN-aware.
 * ------------------------------------------------------------------------- */
__global__ void bg_normalize_kernel(float *data, int npix,
                                     float frame_bg, float scale_ratio,
                                     float ref_bg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;

    float v = data[i];
    if (!isnan(v))
        data[i] = (v - frame_bg) * scale_ratio + ref_bg;
}

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */
DsoError bg_normalize_gpu(float *d_data, int npix,
                           float frame_bg, float scale_ratio, float ref_bg,
                           cudaStream_t stream)
{
    if (!d_data || npix <= 0)
        return DSO_ERR_INVALID_ARG;

    const int threads = 256;
    int blocks = (npix + threads - 1) / threads;

    bg_normalize_kernel<<<blocks, threads, 0, stream>>>(
        d_data, npix, frame_bg, scale_ratio, ref_bg);

    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "bg_normalize_gpu: kernel launch failed: %s\n",
                cudaGetErrorString(ce));
        return DSO_ERR_CUDA;
    }

    return DSO_OK;
}
