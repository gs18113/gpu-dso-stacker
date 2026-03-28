#include "pipeline.h"

#include <stdio.h>

DsoError pipeline_run_metal(FrameInfo            *frames,
                             int                   n_frames,
                             int                   ref_idx,
                             const PipelineConfig *config)
{
    if (!frames || n_frames <= 0 || !config || !config->output_file)
        return DSO_ERR_INVALID_ARG;
    if (ref_idx < 0 || ref_idx >= n_frames)
        return DSO_ERR_INVALID_ARG;

    fprintf(stderr,
            "pipeline_metal: Metal backend requested but this build was compiled without Metal support.\n");
    return DSO_ERR_INVALID_ARG;
}
