# GPU DSO Stacker: Implementation Process

This document tracks the progress of fixes and optimizations based on the technical audit in `GEMINI-ERRORS.md`.

## Current Status: All Phases Complete

### Completed: Technical Audit & Baseline Verification
- [x] **Full Codebase Audit**: Identified risks in memory management, concurrency, and algorithm efficiency.
- [x] **Verification Suite**: Created `tests/test_audit.c` to empirically verify identified issues (e.g., RANSAC non-determinism, CCL memory overhead).
- [x] **Risk Assessment Update**: Downgraded VLA risks after calculating real-world stack usage vs. thread stack limits.

### Completed: Phase 1 — Safety & Stability
- [x] **VLA Elimination**: Removed stack-allocated Variable Length Arrays in `src/integration.c` and `src/calibration.c`, replacing them with per-thread heap-allocated workspaces.
- [x] **CCL Memory Optimization**: Refactored `star_detect_cpu_ccl_com` to re-map unique labels to a contiguous range, reducing the `CompStats` allocation from $O(W \times H)$ to $O(N_{stars})$.
- [x] **Thread-safe Randomness**: Replaced global `srand`/`rand` in `src/ransac.c` with `rand_r` using a robust, call-unique seed.
- [x] **Verification**: All Phase 1 fixes verified passing via `test_audit`.

### Completed: Phase 2 — Performance Optimization
- [x] **CPU Pipeline Mini-batching**: Refactored `src/pipeline_cpu.c` to use a mini-batching strategy, lowering peak RAM usage from $O(N_{frames})$ to $O(BatchSize)$.
- [x] **GPU Async Restoration**: Removed multiple internal `cudaStreamSynchronize` calls in `src/star_detect_gpu.cu`, moving reduction logic fully to the GPU.
- [x] **Lanczos Weight Pre-computation**: Optimized `src/lanczos_cpu.c` to pre-calculate weights for $x$ and $y$ dimensions and added an identity fast-path.
- [x] **Integration Kernel Optimization**: Reduced `INTEGRATION_GPU_MAX_BATCH` to 32 in `include/integration_gpu.h` to mitigate register spilling.
- [x] **Redundant Memory Copy Removal**: Introduced `fits_load_to_buffer` in `src/fits_io.c` and used it in `src/pipeline.cu` to load directly into pinned memory.

### Completed: Phase 3 — Accuracy & Robustness
- [x] **VNG Refinement**: Implemented a more robust 8-directional gradient formula for VNG debayering in both CPU and GPU implementations.
- [x] **Metadata Caching**: Added caching for image dimensions and Bayer patterns in `FrameInfo`, avoiding redundant FITS header parsing in Phase 2.
- [x] **Enhanced Error Context**: Updated `PIPE_CHECK` and `CUDA_CHECK` macros in both pipelines to include the file path being processed.
- [x] **Final Verification**: Full build and `test_audit` passing.
