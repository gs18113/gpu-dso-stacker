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
#include "transform.h"
#include <nppi_geometry_transforms.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Module-level NPP stream context and CUDA stream handle */
static NppStreamContext g_nppCtx;
static cudaStream_t     g_stream = 0;

/* -------------------------------------------------------------------------- */
/* Utility kernel: fill a float buffer with NAN (OOB sentinel)                 */
/* -------------------------------------------------------------------------- */

__global__ static void fill_nan_kernel(float *buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = NAN;
}

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
 * treats as out-of-bounds and leaves unwritten (destination is NAN-initialised
 * so that integration stages can distinguish OOB from real dark pixels).
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
 * NOTE: This function synchronises internally before returning to safely
 * free the temporary device-local scratch buffers.  For high-performance
 * multi-stream overlap, use the `d2d` variant with pre-allocated scratch.
 *
 * Steps in bytes for row-major float32: step = width * sizeof(float).
 */
DsoError lanczos_transform_gpu(const Image *src, Image *dst, const Homography *H)
{
    if (!src || !dst || !H || !src->data || !dst->data) return DSO_ERR_INVALID_ARG;

    /* Reject singular homographies before touching any GPU resources. */
    const double *hi = H->h;
    double det = hi[0]*(hi[4]*hi[8] - hi[5]*hi[7])
               - hi[1]*(hi[3]*hi[8] - hi[5]*hi[6])
               + hi[2]*(hi[3]*hi[7] - hi[4]*hi[6]);
    if (fabs(det) < 1e-12) {
        fprintf(stderr, "lanczos_transform_gpu: singular homography (det=%g)\n", det);
        return DSO_ERR_INVALID_ARG;
    }

    int SW = src->width,  SH = src->height;
    int DW = dst->width,  DH = dst->height;
    size_t src_bytes = (size_t)SW * SH * sizeof(float);
    size_t dst_bytes = (size_t)DW * DH * sizeof(float);
    size_t map_bytes = (size_t)DW * DH * sizeof(float);

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

    /* NAN-fill destination buffer — nppiRemap leaves out-of-bounds pixels
     * unwritten, so they retain NAN as the OOB sentinel for integration. */
    {
        dim3 fb(256);
        dim3 fg((DW * DH + 255) / 256);
        fill_nan_kernel<<<fg, fb, 0, g_stream>>>(d_dst, DW * DH);
        CHECK_CUDA(cudaGetLastError());
    }

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

/* -------------------------------------------------------------------------- */
/* Polynomial coordinate mapping kernel                                        */
/* -------------------------------------------------------------------------- */

/*
 * build_coord_maps_poly — polynomial backward-map per pixel.
 *
 * Evaluates the polynomial transform at each destination pixel (dx, dy)
 * and writes xmap/ymap.  Coefficients are passed as kernel parameters
 * (20 doubles = 160 bytes, well within CUDA's 4KB parameter limit).
 *
 * model: 1=BILINEAR (3 coeffs/axis), 2=BISQUARED (6), 3=BICUBIC (10).
 * c[0..9] = sx coeffs (a0..a9),  c[10..19] = sy coeffs (b0..b9).
 */
__global__ static void build_coord_maps_poly(
    float *xmap, float *ymap, int W, int H,
    int model,
    double c0,  double c1,  double c2,  double c3,  double c4,
    double c5,  double c6,  double c7,  double c8,  double c9,
    double c10, double c11, double c12, double c13, double c14,
    double c15, double c16, double c17, double c18, double c19)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= W || dy >= H) return;

    double x = (double)dx;
    double y = (double)dy;
    double sx, sy;

    if (model == 1) {
        /* BILINEAR */
        sx = c0 + c1*x + c2*y;
        sy = c3 + c4*x + c5*y;
    } else if (model == 2) {
        /* BISQUARED */
        double x2 = x*x, xy = x*y, y2 = y*y;
        sx = c0 + c1*x + c2*y + c3*x2 + c4*xy + c5*y2;
        sy = c6 + c7*x + c8*y + c9*x2 + c10*xy + c11*y2;
    } else {
        /* BICUBIC */
        double x2 = x*x, y2 = y*y, xy = x*y;
        double x3 = x2*x, y3 = y2*y;
        sx = c0  + c1*x  + c2*y  + c3*x2  + c4*xy  + c5*y2
           + c6*x3 + c7*x2*y + c8*x*y2 + c9*y3;
        sy = c10 + c11*x + c12*y + c13*x2 + c14*xy + c15*y2
           + c16*x3 + c17*x2*y + c18*x*y2 + c19*y3;
    }

    int idx = dy * W + dx;
    xmap[idx] = (float)sx;
    ymap[idx] = (float)sy;
}

/* -------------------------------------------------------------------------- */
/* D2D polynomial warp                                                         */
/* -------------------------------------------------------------------------- */

DsoError lanczos_transform_gpu_d2d_poly(
    const float       *d_src,
    float             *d_dst,
    float             *d_xmap,
    float             *d_ymap,
    int                SW, int SH,
    int                DW, int DH,
    const PolyTransform *T,
    cudaStream_t       stream)
{
    if (!d_src || !d_dst || !d_xmap || !d_ymap || !T) return DSO_ERR_INVALID_ARG;
    if (SW <= 0 || SH <= 0 || DW <= 0 || DH <= 0)     return DSO_ERR_INVALID_ARG;

    NppStreamContext ctx = g_nppCtx;
    ctx.hStream = stream;
    if (stream != 0) cudaStreamGetFlags(stream, &ctx.nStreamFlags);

    const double *c = T->coeffs;
    int model = (int)T->model;

    cudaError_t cerr;
    dim3 block(16, 16);
    dim3 grid((DW + 15) / 16, (DH + 15) / 16);

#define CHECK_CUDA_POLY(call) \
    do { cerr = (call); if (cerr != cudaSuccess) { \
        fprintf(stderr, "CUDA error (poly) %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(cerr)); \
        return DSO_ERR_CUDA; } } while(0)

    build_coord_maps_poly<<<grid, block, 0, stream>>>(
        d_xmap, d_ymap, DW, DH, model,
        c[0],  c[1],  c[2],  c[3],  c[4],
        c[5],  c[6],  c[7],  c[8],  c[9],
        c[10], c[11], c[12], c[13], c[14],
        c[15], c[16], c[17], c[18], c[19]);
    CHECK_CUDA_POLY(cudaGetLastError());

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
            fprintf(stderr, "nppiRemap_32f_C1R_Ctx (poly) failed: %d\n", (int)npp_err);
            return DSO_ERR_NPP;
        }
    }

    return DSO_OK;

#undef CHECK_CUDA_POLY
}
