#include "test_framework.h"

#include "pipeline.h"

static int test_dispatch_prefers_cpu_backend(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_CPU;
    cfg.use_gpu_lanczos = 1;
    cfg.use_gpu_ransac = 1;

    ASSERT_ERR(pipeline_run(&frame, 1, 0, &cfg), DSO_ERR_IO);
    return 0;
}

static int test_dispatch_metal_stub_when_disabled(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_METAL;

    ASSERT_ERR(pipeline_run(&frame, 1, 0, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

static int test_dispatch_auto_with_cpu_flag(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_AUTO;
    cfg.use_gpu_lanczos = 0;
    cfg.use_gpu_ransac = 0;

    ASSERT_ERR(pipeline_run(&frame, 1, 0, &cfg), DSO_ERR_IO);
    return 0;
}

int main(void)
{
    SUITE("Pipeline backend dispatch");
    RUN(test_dispatch_prefers_cpu_backend);
    RUN(test_dispatch_metal_stub_when_disabled);
    RUN(test_dispatch_auto_with_cpu_flag);
    return SUMMARY();
}
