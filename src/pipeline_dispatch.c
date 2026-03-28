#include "pipeline.h"

DsoError pipeline_run(FrameInfo            *frames,
                       int                   n_frames,
                       int                   ref_idx,
                       const PipelineConfig *config)
{
    if (!frames || n_frames <= 0 || !config || !config->output_file)
        return DSO_ERR_INVALID_ARG;
    if (ref_idx < 0 || ref_idx >= n_frames)
        return DSO_ERR_INVALID_ARG;

    if (config->backend == DSO_BACKEND_CPU)
        return pipeline_run_cpu(frames, n_frames, ref_idx, config);
    if (config->backend == DSO_BACKEND_METAL)
        return pipeline_run_metal(frames, n_frames, ref_idx, config);
#if defined(DSO_HAS_CUDA) && DSO_HAS_CUDA
    if (config->backend == DSO_BACKEND_CUDA)
        return pipeline_run_cuda(frames, n_frames, ref_idx, config);
#else
    if (config->backend == DSO_BACKEND_CUDA)
        return DSO_ERR_CUDA;
#endif

    /*
     * AUTO: preserve legacy behavior for backward compatibility.
     * Historically use_gpu_lanczos==0 selected the full CPU pipeline.
     */
    if (!config->use_gpu_lanczos)
        return pipeline_run_cpu(frames, n_frames, ref_idx, config);

#if defined(DSO_HAS_CUDA) && DSO_HAS_CUDA
    return pipeline_run_cuda(frames, n_frames, ref_idx, config);
#elif defined(DSO_HAS_METAL) && DSO_HAS_METAL
    return pipeline_run_metal(frames, n_frames, ref_idx, config);
#else
    return pipeline_run_cpu(frames, n_frames, ref_idx, config);
#endif
}
