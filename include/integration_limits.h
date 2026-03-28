#pragma once

/*
 * Shared integration limits used by both CUDA and non-CUDA builds.
 *
 * Keep this header free of CUDA includes so CLI/config validation can use
 * the same constants even when the CUDA backend is disabled.
 */
#define INTEGRATION_GPU_MAX_BATCH 32
