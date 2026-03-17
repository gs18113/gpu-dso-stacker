/*
 * lanczos_gpu.cu — GPU Lanczos-3 image transformation via NPP + CUDA
 *
 * Strategy:
 *   nppiWarpPerspective only supports NN/LINEAR/CUBIC interpolation.
 *   To obtain true Lanczos-3 quality with a homographic warp we:
 *     1. H is the backward (ref → src) homography; use it directly.
 *     2. Launch a CUDA kernel (build_coord_maps) that evaluates H
 *        per destination pixel and writes two float device maps:
 *        xmap[dy*W+dx] and ymap[dy*W+dx] = source (sx, sy).
 *     3. Call nppiRemap_32f_C1R_Ctx with NPPI_INTER_LANCZOS, which reads
 *        the precomputed maps and performs the sampling on the GPU.
 *
 * Memory lifecycle:
 *   d_src / d_dst / d_xmap / d_ymap are allocated and freed per call inside
 *   lanczos_transform_gpu. The NppStreamContext (g_nppCtx) is initialised
 *   once in lanczos_gpu_init and reused across calls.
 */

#include "lanczos_gpu.h"
#include <nppi_geometry_transforms.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Module-level NPP stream context and CUDA stream handle */
static NppStreamContext g_nppCtx;
static cudaStream_t     g_stream = 0;

/* -------------------------------------------------------------------------- */
/* Coord-map kernel                                                            */
/* -------------------------------------------------------------------------- */

/*
 * build_coord_maps — compute backward-map source coordinates per pixel.
 *
 * For each destination pixel (dx, dy) the kernel computes:
 *   [sx_h, sy_h, sw]^T = H * [dx, dy, 1]^T
 *   xmap[dy*W+dx] = sx_h / sw
 *   ymap[dy*W+dx] = sy_h / sw
 *
 * H is the backward (ref → src) map and is used directly without inversion.
 * Pixels where sw is near zero are mapped to (−1, −1), which nppiRemap
 * treats as out-of-bounds and leaves as 0 (destination is zero-initialised).
 */
__global__ static void build_coord_maps(
    float *xmap, float *ymap, int W, int H,
    double h00, double h01, double h02,
    double h10, double h11, double h12,
    double h20, double h21, double h22)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= W || dy >= H) return;

    double sx_h = h00*(double)dx + h01*(double)dy + h02;
    double sy_h = h10*(double)dx + h11*(double)dy + h12;
    double sw   = h20*(double)dx + h21*(double)dy + h22;

    int idx = dy * W + dx;
    if (fabs(sw) > 1e-12) {
        xmap[idx] = (float)(sx_h / sw);
        ymap[idx] = (float)(sy_h / sw);
    } else {
        xmap[idx] = -1.f;
        ymap[idx] = -1.f;
    }
}

/* -------------------------------------------------------------------------- */
/* Host-side 3×3 homography inversion (unused; kept for reference)            */
/* -------------------------------------------------------------------------- */

static DsoError invert_homography_h(const double *h, double *hi)
{
    double c00 =  h[4]*h[8] - h[5]*h[7];
    double c01 = -(h[3]*h[8] - h[5]*h[6]);
    double c02 =  h[3]*h[7] - h[4]*h[6];
    double c10 = -(h[1]*h[8] - h[2]*h[7]);
    double c11 =  h[0]*h[8] - h[2]*h[6];
    double c12 = -(h[0]*h[7] - h[1]*h[6]);
    double c20 =  h[1]*h[5] - h[2]*h[4];
    double c21 = -(h[0]*h[5] - h[2]*h[3]);
    double c22 =  h[0]*h[4] - h[1]*h[3];

    double det = h[0]*c00 + h[1]*c01 + h[2]*c02;
    if (fabs(det) < 1e-12) {
        fprintf(stderr, "lanczos_gpu: homography is singular (det=%g)\n", det);
        return DSO_ERR_INVALID_ARG;
    }

    double inv = 1.0 / det;
    hi[0] = c00*inv; hi[1] = c10*inv; hi[2] = c20*inv;
    hi[3] = c01*inv; hi[4] = c11*inv; hi[5] = c21*inv;
    hi[6] = c02*inv; hi[7] = c12*inv; hi[8] = c22*inv;
    return DSO_OK;
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                  */
/* -------------------------------------------------------------------------- */

/*
 * lanczos_gpu_init — populate NppStreamContext from device properties.
 *
 * Must be called once before any lanczos_transform_gpu calls.
 * Pass stream=0 to use the CUDA default stream.
 */
DsoError lanczos_gpu_init(cudaStream_t stream)
{
    g_stream = stream;
    memset(&g_nppCtx, 0, sizeof(g_nppCtx));

    int dev = 0;
    cudaError_t cerr = cudaGetDevice(&dev);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "lanczos_gpu_init: cudaGetDevice: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    cudaDeviceProp prop;
    cerr = cudaGetDeviceProperties(&prop, dev);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "lanczos_gpu_init: cudaGetDeviceProperties: %s\n",
                cudaGetErrorString(cerr));
        return DSO_ERR_CUDA;
    }

    /* Fill in all fields that NPP+ requires for stream-aware dispatch */
    g_nppCtx.hStream                          = stream;
    g_nppCtx.nCudaDeviceId                    = dev;
    g_nppCtx.nMultiProcessorCount             = prop.multiProcessorCount;
    g_nppCtx.nMaxThreadsPerMultiProcessor     = prop.maxThreadsPerMultiProcessor;
    g_nppCtx.nMaxThreadsPerBlock              = prop.maxThreadsPerBlock;
    g_nppCtx.nSharedMemPerBlock               = prop.sharedMemPerBlock;
    g_nppCtx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    g_nppCtx.nCudaDevAttrComputeCapabilityMinor = prop.minor;

    /* Non-default streams have non-blocking flags */
    unsigned int flags = 0;
    if (stream != 0) cudaStreamGetFlags(stream, &flags);
    g_nppCtx.nStreamFlags = flags;

    return DSO_OK;
}

/* No persistent GPU resources to release; kept for API symmetry. */
void lanczos_gpu_cleanup(void) {}

/*
 * lanczos_transform_gpu — warp src → dst with Lanczos-3 on the GPU.
 *
 * Per-frame steps:
 *   1. Allocate d_src, d_dst, d_xmap, d_ymap on the device.
 *   2. Async-copy src pixels to d_src.
 *   3. Zero d_dst (NPP leaves out-of-bounds pixels unwritten).
 *   4. Launch build_coord_maps kernel with 16×16 thread blocks.
 *   5. Call nppiRemap_32f_C1R_Ctx (NPPI_INTER_LANCZOS).
 *   6. Async-copy result back, synchronise, free device buffers.
 *
 * Steps in bytes for row-major float32: step = width * sizeof(float).
 */
DsoError lanczos_transform_gpu(const Image *src, Image *dst, const Homography *H)
{
    if (!src || !dst || !H || !src->data || !dst->data) return DSO_ERR_INVALID_ARG;

    int SW = src->width,  SH = src->height;
    int DW = dst->width,  DH = dst->height;
    size_t src_bytes = (size_t)SW * SH * sizeof(float);
    size_t dst_bytes = (size_t)DW * DH * sizeof(float);
    size_t map_bytes = (size_t)DW * DH * sizeof(float);

    /* H is already the backward map (ref → src); use it directly. */
    const double *hi = H->h;

    float *d_src = NULL, *d_dst = NULL, *d_xmap = NULL, *d_ymap = NULL;
    cudaError_t cerr;

    /* Declare dim3 before any goto to keep C++ happy about jump-over-init */
    dim3 block(16, 16);
    dim3 grid((DW + 15) / 16, (DH + 15) / 16);

#define CHECK_CUDA(call) \
    do { cerr = (call); if (cerr != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(cerr)); \
        goto cuda_err; } } while(0)

    CHECK_CUDA(cudaMalloc(&d_src,  src_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst,  dst_bytes));
    CHECK_CUDA(cudaMalloc(&d_xmap, map_bytes));
    CHECK_CUDA(cudaMalloc(&d_ymap, map_bytes));

    /* Upload source image asynchronously */
    CHECK_CUDA(cudaMemcpyAsync(d_src, src->data, src_bytes,
                               cudaMemcpyHostToDevice, g_stream));

    /* Zero destination buffer — nppiRemap leaves out-of-bounds pixels as-is */
    CHECK_CUDA(cudaMemsetAsync(d_dst, 0, dst_bytes, g_stream));

    /* Build the (xmap, ymap) coordinate lookup tables on device */
    build_coord_maps<<<grid, block, 0, g_stream>>>(
        d_xmap, d_ymap, DW, DH,
        hi[0], hi[1], hi[2],
        hi[3], hi[4], hi[5],
        hi[6], hi[7], hi[8]);
    CHECK_CUDA(cudaGetLastError());

    /* nppiRemap_32f_C1R_Ctx — Lanczos-3 resampling driven by precomputed maps.
     * Row step = width × sizeof(float) for all row-major float32 buffers. */
    {
        NppiSize oSrcSize = { SW, SH };
        NppiRect oSrcROI  = { 0, 0, SW, SH };
        NppiSize oDstSize = { DW, DH };
        int src_step = SW * (int)sizeof(float);
        int map_step = DW * (int)sizeof(float);
        int dst_step = DW * (int)sizeof(float);

        NppStatus npp_err = nppiRemap_32f_C1R_Ctx(
            d_src,  oSrcSize, src_step, oSrcROI,
            d_xmap, map_step,
            d_ymap, map_step,
            d_dst,  dst_step, oDstSize,
            NPPI_INTER_LANCZOS,
            g_nppCtx);

        if (npp_err != NPP_SUCCESS) {
            fprintf(stderr, "nppiRemap_32f_C1R_Ctx failed: %d\n", (int)npp_err);
            goto cuda_err;
        }
    }

    /* Download result and synchronise before returning to caller */
    CHECK_CUDA(cudaMemcpyAsync(dst->data, d_dst, dst_bytes,
                               cudaMemcpyDeviceToHost, g_stream));
    CHECK_CUDA(cudaStreamSynchronize(g_stream));

    cudaFree(d_src);  cudaFree(d_dst);
    cudaFree(d_xmap); cudaFree(d_ymap);
    return DSO_OK;

cuda_err:
    cudaFree(d_src);  cudaFree(d_dst);
    cudaFree(d_xmap); cudaFree(d_ymap);
    return DSO_ERR_CUDA;

#undef CHECK_CUDA
}

/*
 * lanczos_transform_gpu_d2d — device-to-device Lanczos-3 warp.
 *
 * All buffers are already on the device; no H2D/D2H transfers are performed.
 * Executes on `stream` asynchronously — caller must synchronise.
 *
 * The NppStreamContext is derived from g_nppCtx (populated once by
 * lanczos_gpu_init) with only the hStream field overridden.  This avoids a
 * repeated cudaGetDeviceProperties call while still directing NPP work to the
 * correct stream for multi-stream pipeline overlap.
 */
DsoError lanczos_transform_gpu_d2d(
    const float       *d_src,
    float             *d_dst,
    float             *d_xmap,
    float             *d_ymap,
    int                SW, int SH,
    int                DW, int DH,
    const Homography  *H,
    cudaStream_t       stream)
{
    if (!d_src || !d_dst || !d_xmap || !d_ymap || !H) return DSO_ERR_INVALID_ARG;
    if (SW <= 0 || SH <= 0 || DW <= 0 || DH <= 0)     return DSO_ERR_INVALID_ARG;

    /* Build a per-call NppStreamContext that routes NPP work to `stream`. */
    NppStreamContext ctx = g_nppCtx;
    ctx.hStream = stream;
    /* Update stream flags in case this stream has non-default flags */
    if (stream != 0) cudaStreamGetFlags(stream, &ctx.nStreamFlags);

    /* H is already the backward map (ref → src); used directly. */
    const double *hi = H->h;

    cudaError_t cerr;

    /* dim3 declarations before any CHECK_CUDA to avoid jump-over-init */
    dim3 block(16, 16);
    dim3 grid((DW + 15) / 16, (DH + 15) / 16);

#define CHECK_CUDA_D2D(call) \
    do { cerr = (call); if (cerr != cudaSuccess) { \
        fprintf(stderr, "CUDA error (d2d) %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(cerr)); \
        return DSO_ERR_CUDA; } } while(0)

    /* Build coordinate maps from the backward homography */
    build_coord_maps<<<grid, block, 0, stream>>>(
        d_xmap, d_ymap, DW, DH,
        hi[0], hi[1], hi[2],
        hi[3], hi[4], hi[5],
        hi[6], hi[7], hi[8]);
    CHECK_CUDA_D2D(cudaGetLastError());

    /* Remap with Lanczos-3 interpolation.
     * Row step = width × sizeof(float) for all row-major float32 buffers. */
    {
        NppiSize oSrcSize = { SW, SH };
        NppiRect oSrcROI  = { 0, 0, SW, SH };
        NppiSize oDstSize = { DW, DH };
        int src_step = SW * (int)sizeof(float);
        int map_step = DW * (int)sizeof(float);
        int dst_step = DW * (int)sizeof(float);

        NppStatus npp_err = nppiRemap_32f_C1R_Ctx(
            d_src,  oSrcSize, src_step, oSrcROI,
            d_xmap, map_step,
            d_ymap, map_step,
            d_dst,  dst_step, oDstSize,
            NPPI_INTER_LANCZOS,
            ctx);

        if (npp_err != NPP_SUCCESS) {
            fprintf(stderr, "nppiRemap_32f_C1R_Ctx (d2d) failed: %d\n", (int)npp_err);
            return DSO_ERR_NPP;
        }
    }

    return DSO_OK;

#undef CHECK_CUDA_D2D
}
