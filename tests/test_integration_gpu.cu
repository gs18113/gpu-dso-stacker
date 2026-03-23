/*
 * test_integration_gpu.cu — Unit tests for GPU mini-batch kappa-sigma
 * integration (integration_gpu.h).
 *
 * Tests cover:
 *   - Init / cleanup: no crash, context is non-NULL
 *   - Mean integration: constant frames → exact mean
 *   - Kappa-sigma: outlier in one frame correctly rejected
 *   - Kappa-sigma vs CPU reference: GPU matches CPU integrate_kappa_sigma
 *   - Batch boundary: N=17 frames with batch_size=16 (partial last batch)
 *   - All-clipped fallback: when every value is an outlier → unclipped mean
 *   - Mean integration method: plain mean across batches
 *   - Large values: no float32 overflow or precision loss at moderate N
 *
 * Exit code 77 = SKIP (CTest convention) when no CUDA device is found.
 */

#include "test_framework.h"
#include "integration_gpu.h"
#include "integration.h"   /* CPU reference: integrate_kappa_sigma */
#include "fits_io.h"       /* image_free */
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Check for CUDA device; return 1 if available. */
static int has_cuda(void)
{
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
}

/* Allocate a host Image filled with constant value v. */
static Image make_const_img(int W, int H, float v)
{
    Image img;
    img.width  = W;
    img.height = H;
    img.data   = (float *)malloc((size_t)W * H * sizeof(float));
    for (int i = 0; i < W * H; i++) img.data[i] = v;
    return img;
}

/* Max absolute pixel difference between two images (same size). */
static float max_pixel_diff(const Image *a, const Image *b)
{
    float mx = 0.f;
    int n = a->width * a->height;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a->data[i] - b->data[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/*
 * Run integration on a set of images using the GPU pipeline.
 * Fills ctx frame slots one batch at a time, calls process_batch,
 * then finalizes.
 *
 * frames[]    : array of N host-side Images (all W×H)
 * N           : total frame count
 * batch_size  : ctx batch size
 * kappa       : sigma-clipping threshold
 * iterations  : max clipping passes
 * use_ks      : 1 = kappa-sigma, 0 = mean
 * out         : must be pre-allocated (W×H floats set to 0)
 */
static DsoError run_gpu_integration(const Image **frames, int N,
                                     int batch_size,
                                     float kappa, int iterations,
                                     int use_ks, Image *out)
{
    IntegrationGpuCtx *ctx = NULL;
    int W = frames[0]->width, H = frames[0]->height;
    DsoError err = integration_gpu_init(W, H, batch_size, &ctx);
    if (err != DSO_OK) return err;

    int processed = 0;
    while (processed < N) {
        int M = N - processed;
        if (M > batch_size) M = batch_size;

        /* Upload M frames into ctx frame slots */
        for (int m = 0; m < M; m++) {
            size_t nbytes = (size_t)W * H * sizeof(float);
            cudaError_t cerr = cudaMemcpy(ctx->d_frames[m],
                                          frames[processed + m]->data,
                                          nbytes,
                                          cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) {
                integration_gpu_cleanup(ctx);
                return DSO_ERR_CUDA;
            }
        }

        if (use_ks)
            err = integration_gpu_process_batch(ctx, M, kappa, iterations, 0);
        else
            err = integration_gpu_process_batch_mean(ctx, M, 0);

        if (err != DSO_OK) { integration_gpu_cleanup(ctx); return err; }
        processed += M;
    }

    err = integration_gpu_finalize(ctx, N, out, 0);
    integration_gpu_cleanup(ctx);
    return err;
}

/* =========================================================================
 * Tests
 * ========================================================================= */

static int test_init_cleanup(void)
{
    IntegrationGpuCtx *ctx = NULL;
    ASSERT_OK(integration_gpu_init(64, 64, 8, &ctx));
    ASSERT_NOT_NULL(ctx);
    integration_gpu_cleanup(ctx);
    integration_gpu_cleanup(NULL); /* must not crash */
    return 0;
}

/* 10 constant frames, all value 5.0 → mean = 5.0, max error < 1e-4. */
static int test_constant_frames_mean(void)
{
    const int W = 32, H = 32, N = 10;
    const Image *frames[10];
    Image storage[10];
    for (int i = 0; i < N; i++) {
        storage[i] = make_const_img(W, H, 5.0f);
        frames[i] = &storage[i];
    }

    Image out = make_const_img(W, H, 0.f);
    ASSERT_OK(run_gpu_integration(frames, N, 8, 3.0f, 3, 0, &out)); /* mean mode */

    /* Every pixel should be 5.0 */
    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(out.data[i], 5.0f, 1e-4f);

    for (int i = 0; i < N; i++) image_free(&storage[i]);
    image_free(&out);
    return 0;
}

/* 9 frames of value 1.0 + 1 frame of value 100.0.
 * kappa=2.0, iterations=3 → outlier (100) should be rejected.
 * Expected output ≈ 1.0. */
static int test_kappa_sigma_rejects_outlier(void)
{
    const int W = 8, H = 8, N = 10;
    const Image *frames[10];
    Image storage[10];
    for (int i = 0; i < N - 1; i++) {
        storage[i] = make_const_img(W, H, 1.0f);
        frames[i] = &storage[i];
    }
    storage[N-1] = make_const_img(W, H, 100.0f);
    frames[N-1] = &storage[N-1];

    Image out = make_const_img(W, H, 0.f);
    ASSERT_OK(run_gpu_integration(frames, N, 10, 2.0f, 3, 1, &out));

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(out.data[i], 1.0f, 0.05f);

    for (int i = 0; i < N; i++) image_free(&storage[i]);
    image_free(&out);
    return 0;
}

/* GPU kappa-sigma result must match CPU integrate_kappa_sigma.
 * Use 8 frames with one outlier. */
static int test_kappa_sigma_matches_cpu(void)
{
    const int W = 16, H = 16, N = 8;
    Image storage[8];
    const Image *frames[8];

    for (int i = 0; i < N; i++) {
        storage[i] = make_const_img(W, H, (float)(i + 1));
        frames[i] = &storage[i];
    }
    /* Make frame 7 (value=8) a strong outlier on one pixel */
    storage[7].data[0] = 1000.f;

    /* CPU reference */
    Image cpu_out = {NULL, 0, 0};
    ASSERT_OK(integrate_kappa_sigma(frames, N, &cpu_out, 2.5f, 3));

    /* GPU result */
    Image gpu_out = make_const_img(W, H, 0.f);
    ASSERT_OK(run_gpu_integration(frames, N, 8, 2.5f, 3, 1, &gpu_out));

    /* Allow up to 1.0 difference (mini-batch kappa-sigma is an approximation) */
    float diff = max_pixel_diff(&cpu_out, &gpu_out);
    ASSERT(diff < 1.0f);

    for (int i = 0; i < N; i++) image_free(&storage[i]);
    image_free(&cpu_out);
    image_free(&gpu_out);
    return 0;
}

/* N=17 frames with batch_size=16.
 * First batch: 16 frames (all value 2.0).
 * Second batch: 1 frame (value 2.0).
 * Final result must be 2.0. */
static int test_batch_boundary(void)
{
    const int W = 16, H = 16, N = 17;
    Image storage[17];
    const Image *frames[17];
    for (int i = 0; i < N; i++) {
        storage[i] = make_const_img(W, H, 2.0f);
        frames[i] = &storage[i];
    }

    Image out = make_const_img(W, H, 0.f);
    ASSERT_OK(run_gpu_integration(frames, N, 16, 3.0f, 3, 1, &out));

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(out.data[i], 2.0f, 0.01f);

    for (int i = 0; i < N; i++) image_free(&storage[i]);
    image_free(&out);
    return 0;
}

/* All-clipped fallback: a 1-pixel image with 5 identical frames + very tight
 * kappa.  If kappa is ridiculously tight (0.001), all values get clipped
 * because stddev≈0 but the mean±kappa*σ window has no tolerance, or because
 * with n=5 identical values stddev=0 exactly and no clipping occurs (all
 * identical means σ=0).  Actually test with n=2 frames having extreme spread:
 * frame0=0, frame1=10000; kappa=0.0001.  Both clipped → fallback to (0+10000)/2=5000. */
static int test_all_clipped_fallback(void)
{
    const int W = 1, H = 1, N = 2;
    Image storage[2];
    storage[0] = make_const_img(W, H, 0.f);
    storage[1] = make_const_img(W, H, 10000.f);
    const Image *frames[2] = {&storage[0], &storage[1]};

    Image out = make_const_img(W, H, -1.f);
    /* With kappa=0.0001, both values should be clipped (each is ~0.7σ from mean=5000,
     * but with Bessel correction n=2 gives σ=7071; 0.0001*7071≈0.7, so |0-5000|=5000 > 0.7 → both clipped) */
    ASSERT_OK(run_gpu_integration(frames, N, 2, 0.0001f, 3, 1, &out));

    /* Fallback to unclipped mean = (0 + 10000) / 2 = 5000 */
    ASSERT_NEAR(out.data[0], 5000.f, 1.f);

    image_free(&storage[0]); image_free(&storage[1]);
    image_free(&out);
    return 0;
}

/* Mean integration (not kappa-sigma): all values contribute.
 * Frames with values 1,2,3,4,5 → mean = 3.0. */
static int test_mean_integration_method(void)
{
    const int W = 8, H = 8, N = 5;
    Image storage[5];
    const Image *frames[5];
    for (int i = 0; i < N; i++) {
        storage[i] = make_const_img(W, H, (float)(i + 1));
        frames[i] = &storage[i];
    }

    Image out = make_const_img(W, H, 0.f);
    ASSERT_OK(run_gpu_integration(frames, N, 4, 3.0f, 3, 0, &out)); /* mean mode */

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(out.data[i], 3.0f, 1e-4f);

    for (int i = 0; i < N; i++) image_free(&storage[i]);
    image_free(&out);
    return 0;
}

/* Large pixel values: N=10 frames, all value 30000.0 → mean = 30000.
 * Tests that float32 accumulation doesn't saturate at moderate frame counts. */
static int test_large_values_no_overflow(void)
{
    const int W = 16, H = 16, N = 10;
    Image storage[10];
    const Image *frames[10];
    for (int i = 0; i < N; i++) {
        storage[i] = make_const_img(W, H, 30000.f);
        frames[i] = &storage[i];
    }

    Image out = make_const_img(W, H, 0.f);
    ASSERT_OK(run_gpu_integration(frames, N, 8, 3.0f, 3, 0, &out));

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(out.data[i], 30000.f, 1.f); /* 1 ADU tolerance */

    for (int i = 0; i < N; i++) image_free(&storage[i]);
    image_free(&out);
    return 0;
}

/* Single frame: should return the frame unchanged. */
static int test_single_frame(void)
{
    const int W = 8, H = 8;
    Image storage = make_const_img(W, H, 42.5f);
    const Image *frames[1] = {&storage};

    Image out = make_const_img(W, H, 0.f);
    ASSERT_OK(run_gpu_integration(frames, 1, 4, 3.0f, 3, 1, &out));

    for (int i = 0; i < W * H; i++)
        ASSERT_NEAR(out.data[i], 42.5f, 1e-3f);

    image_free(&storage);
    image_free(&out);
    return 0;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    if (!has_cuda()) {
        printf("No CUDA device found — skipping GPU integration tests\n");
        return 77;  /* CTest SKIP */
    }

    SUITE("integration_gpu — mini-batch kappa-sigma");
    RUN(test_init_cleanup);
    RUN(test_constant_frames_mean);
    RUN(test_kappa_sigma_rejects_outlier);
    RUN(test_kappa_sigma_matches_cpu);
    RUN(test_batch_boundary);
    RUN(test_all_clipped_fallback);
    RUN(test_mean_integration_method);
    RUN(test_large_values_no_overflow);
    RUN(test_single_frame);

    return SUMMARY();
}
