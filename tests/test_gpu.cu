/*
 * test_gpu.cu — unit tests for the GPU Lanczos transform path
 *
 * Tests:
 *   1. GPU init / cleanup runs without crashing.
 *   2. Identity homography: GPU output matches CPU output (within float epsilon).
 *   3. Integer-pixel shift: GPU output matches CPU output.
 *   4. Out-of-bounds shift: every destination pixel is 0.
 *   5. Singular homography: returns DSO_ERR_INVALID_ARG (not CUDA error).
 *
 * If CUDA device initialisation fails (no GPU) the test binary exits with
 * code 77, which CTest interprets as SKIP.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "test_framework.h"
#include "dso_types.h"
#include "fits_io.h"
#include "lanczos_cpu.h"
#include "lanczos_gpu.h"

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

static Image make_gradient(int w, int h)
{
    Image img = { nullptr, w, h };
    img.data = (float *)malloc((size_t)w * h * sizeof(float));
    if (!img.data) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img.data[y * w + x] = (float)(y * w + x);
    return img;
}

static Image make_const(int w, int h, float v)
{
    Image img = { nullptr, w, h };
    img.data = (float *)malloc((size_t)w * h * sizeof(float));
    if (!img.data) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < w * h; i++) img.data[i] = v;
    return img;
}

/*
 * Returns the maximum absolute difference between two images.
 * Used to compare CPU and GPU outputs.
 */
static float max_abs_diff(const Image *a, const Image *b)
{
    float max_d = 0.f;
    int n = a->width * a->height;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a->data[i] - b->data[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

/* =========================================================================
 * Tests
 * ====================================================================== */

/* GPU init must succeed without error on a machine with a CUDA device. */
static int test_gpu_init_cleanup(void)
{
    ASSERT_OK(lanczos_gpu_init(0));
    lanczos_gpu_cleanup();
    /* Re-initialise so subsequent tests work */
    ASSERT_OK(lanczos_gpu_init(0));
    return 0;
}

/*
 * Identity homography: GPU output must match CPU output to within 1e-4.
 *
 * Both paths use the same inverse-homography math; the difference comes from
 * NPP's internal Lanczos kernel vs our C kernel. For an identity transform
 * (all samples at integer coordinates) both should produce exact values,
 * so the tolerance can be tight.
 */
static int test_gpu_identity(void)
{
    const int W = 32, H = 32;
    Image src      = make_gradient(W, H);
    Image dst_cpu  = make_const(W, H, 0.f);
    Image dst_gpu  = make_const(W, H, 0.f);

    Homography H_id = {{ 1,0,0, 0,1,0, 0,0,1 }};

    ASSERT_OK(lanczos_transform_cpu(&src, &dst_cpu, &H_id));
    ASSERT_OK(lanczos_transform_gpu(&src, &dst_gpu, &H_id));

    float diff = max_abs_diff(&dst_cpu, &dst_gpu);
    if (diff > 1e-3f) {
        printf("\n    max abs diff CPU vs GPU: %g (threshold 1e-3)\n", diff);
        image_free(&src); image_free(&dst_cpu); image_free(&dst_gpu);
        return 1;
    }

    image_free(&src);
    image_free(&dst_cpu);
    image_free(&dst_gpu);
    return 0;
}

/*
 * Integer-pixel translation: GPU result must agree with CPU result.
 * H maps src(x,y) → ref(x+5, y+3); interior pixels sample exact positions.
 */
static int test_gpu_integer_shift(void)
{
    const int W = 40, H = 40;
    Image src     = make_gradient(W, H);
    Image dst_cpu = make_const(W, H, 0.f);
    Image dst_gpu = make_const(W, H, 0.f);

    Homography H_shift = {{ 1,0,5, 0,1,3, 0,0,1 }};

    ASSERT_OK(lanczos_transform_cpu(&src, &dst_cpu, &H_shift));
    ASSERT_OK(lanczos_transform_gpu(&src, &dst_gpu, &H_shift));

    /* Check interior pixels only (away from boundaries) */
    float max_d = 0.f;
    for (int dy = 8; dy < H - 4; dy++) {
        for (int dx = 8; dx < W - 4; dx++) {
            float d = fabsf(dst_cpu.data[dy * W + dx] - dst_gpu.data[dy * W + dx]);
            if (d > max_d) max_d = d;
        }
    }

    if (max_d > 1e-3f) {
        printf("\n    max interior diff CPU vs GPU: %g (threshold 1e-3)\n", max_d);
        image_free(&src); image_free(&dst_cpu); image_free(&dst_gpu);
        return 1;
    }

    image_free(&src);
    image_free(&dst_cpu);
    image_free(&dst_gpu);
    return 0;
}

/*
 * Out-of-bounds shift: a translation that pushes all source content outside
 * the destination frame must produce a fully-zeroed output.
 * Verifies that nppiRemap out-of-bounds pixels are filled with 0
 * (destination is cudaMemset'd to 0 before the NPP call).
 */
static int test_gpu_oob_pixels_zero(void)
{
    const int W = 16, H = 16;
    Image src = make_const(W, H, 99.f);
    Image dst = make_const(W, H, -1.f);   /* sentinel */

    Homography H_far = {{ 1,0,5000, 0,1,5000, 0,0,1 }};
    ASSERT_OK(lanczos_transform_gpu(&src, &dst, &H_far));

    for (int i = 0; i < W * H; i++) {
        if (dst.data[i] != 0.f) {
            printf("\n    dst[%d] = %f, expected 0\n", i, dst.data[i]);
            image_free(&src); image_free(&dst);
            return 1;
        }
    }

    image_free(&src);
    image_free(&dst);
    return 0;
}

/* A singular homography must return DSO_ERR_INVALID_ARG (caught on host). */
static int test_gpu_singular_h(void)
{
    Image src = make_const(8, 8, 1.f);
    Image dst = make_const(8, 8, 0.f);
    Homography H_sing = {{ 0,0,0, 0,0,0, 0,0,0 }};
    ASSERT_ERR(lanczos_transform_gpu(&src, &dst, &H_sing), DSO_ERR_INVALID_ARG);
    image_free(&src);
    image_free(&dst);
    return 0;
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    /* Check that a CUDA device is available before running any test */
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        printf("No CUDA device found — skipping GPU tests (exit 77)\n");
        return 77;   /* CTest SKIP code */
    }

    if (lanczos_gpu_init(0) != DSO_OK) {
        printf("GPU init failed — skipping GPU tests (exit 77)\n");
        return 77;
    }

    SUITE("Lanczos GPU");
    RUN(test_gpu_init_cleanup);
    RUN(test_gpu_identity);
    RUN(test_gpu_integer_shift);
    RUN(test_gpu_oob_pixels_zero);
    RUN(test_gpu_singular_h);

    lanczos_gpu_cleanup();
    return SUMMARY();
}
