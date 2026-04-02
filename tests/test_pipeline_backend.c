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

    ASSERT_ERR(pipeline_run(&frame, 1, 0, &cfg), DSO_ERR_FITS);
    return 0;
}

static int test_dispatch_metal_stub_when_disabled(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_METAL;

#if defined(DSO_HAS_METAL) && DSO_HAS_METAL
    ASSERT_ERR(pipeline_run(&frame, 1, 0, &cfg), DSO_ERR_FITS);
#else
    ASSERT_ERR(pipeline_run(&frame, 1, 0, &cfg), DSO_ERR_INVALID_ARG);
#endif
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

    ASSERT_ERR(pipeline_run(&frame, 1, 0, &cfg), DSO_ERR_FITS);
    return 0;
}

/* Single frame with dummy path should not attempt RANSAC (ref only). */
static int test_pipeline_single_frame_no_ransac(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_CPU;
    /* With 1 frame, pipeline should fail on fits_load (dummy path)
     * but not on RANSAC */
    DsoError e = pipeline_run(&frame, 1, 0, &cfg);
    ASSERT(e != DSO_OK); /* Expected: file not found, not RANSAC error */
    return 0;
}

/* Negative ref_idx → DSO_ERR_INVALID_ARG */
static int test_pipeline_negative_ref_idx(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_CPU;
    ASSERT_ERR(pipeline_run(&frame, 1, -1, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

/* ref_idx out of range → DSO_ERR_INVALID_ARG */
static int test_pipeline_ref_idx_out_of_range(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_CPU;
    ASSERT_ERR(pipeline_run(&frame, 1, 5, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

/* NULL frames → DSO_ERR_INVALID_ARG */
static int test_pipeline_null_frames(void)
{
    PipelineConfig cfg = {0};
    cfg.output_file = "out.fits";
    cfg.backend = DSO_BACKEND_CPU;
    ASSERT_ERR(pipeline_run(NULL, 1, 0, &cfg), DSO_ERR_INVALID_ARG);
    return 0;
}

/* NULL config → DSO_ERR_INVALID_ARG */
static int test_pipeline_null_config(void)
{
    FrameInfo frame = {"dummy", 1, {{0}}};
    ASSERT_ERR(pipeline_run(&frame, 1, 0, NULL), DSO_ERR_INVALID_ARG);
    return 0;
}

int main(void)
{
    SUITE("Pipeline backend dispatch");
    RUN(test_dispatch_prefers_cpu_backend);
    RUN(test_dispatch_metal_stub_when_disabled);
    RUN(test_dispatch_auto_with_cpu_flag);

    SUITE("Pipeline argument validation");
    RUN(test_pipeline_single_frame_no_ransac);
    RUN(test_pipeline_negative_ref_idx);
    RUN(test_pipeline_ref_idx_out_of_range);
    RUN(test_pipeline_null_frames);
    RUN(test_pipeline_null_config);

    return SUMMARY();
}
