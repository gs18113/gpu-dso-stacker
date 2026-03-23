# GPU DSO Stacker: Technical Audit & Improvement Plan

This document identifies potential dangers, errors, and misimplementations in the `gpu-dso-stacker` project and provides a roadmap for resolution.

## 1. Identified Dangers & Errors

### 1.1 Numerical & Mathematical Risks
*   **VLA Usage & Stack Robustness (Low/Medium Risk)**: In `integration.c` and `calibration.c`, `float vals[n]` and `int actv[n]` are allocated as Variable Length Arrays (VLAs) on the stack within OpenMP loops. While typical DSO frame counts (N < 1000) use only a few kilobytes of stack per thread and are safe under standard `OMP_STACKSIZE` defaults (2-8 MB), this approach has drawbacks:
    *   **Portability**: VLAs are optional in C11 and not supported by MSVC.
    *   **Unbounded Growth**: For extreme datasets (e.g., planetary stacking with $N > 100,000$), or on systems with very small thread stacks, this remains a potential failure point.
    *   **Alternative**: A per-thread heap-allocated workspace is more robust.
*   **Simplified VNG Debayering (Medium Risk)**: The current VNG implementation in `debayer_gpu.cu` and `debayer_cpu.c` uses a highly simplified 8-directional gradient formula and a basic averaging scheme for missing colors. This may produce more demosaicing artifacts (zippering, color bleeding) than a standard VNG implementation.
*   **RANSAC `rand()` Usage (Medium Risk)**: `ransac_compute_homography` uses `srand(time(NULL))` and `rand()`. `rand()` is not thread-safe and has a limited period. While Phase 1 is currently serial, any future parallelization of alignment will suffer from race conditions or identical random sequences.

### 1.2 Performance Bottlenecks
*   **Massive Heap Allocation in CCL (High Risk)**: `star_detect_cpu_ccl_com` allocates a `CompStats` array of size `npix + 1` (e.g., ~900MB for a 16MP frame) every frame. This is extremely inefficient as most entries remain unused.
*   **CPU Pipeline Memory Bloat (High Risk)**: `pipeline_run_cpu` holds *all* transformed frames in RAM simultaneously before integration. For 100 frames of 16MP, this consumes ~6.4GB of RAM. The CPU pipeline should adopt a mini-batching strategy similar to the GPU pipeline.
*   **Broken Asynchronous Pipeline (Medium Risk)**:
    *   `star_detect_gpu_d2d` and `lanczos_transform_gpu` (h2h) call `cudaStreamSynchronize` internally. This prevents overlapping computation with host-side tasks or other GPU streams.
    *   `upload_moffat_kernel` uses `cudaMemcpyToSymbol`, which is a synchronous operation.
*   **GPU Register Spilling (Medium Risk)**: In `integration_gpu.cu`, `kappa_sigma_batch_kernel` uses `float vals[64]` and `char active[64]` per thread. This exceeds the register file limit on most architectures, causing spilling to slow local memory.
*   **Redundant Lanczos Computations (Low Risk)**: `lanczos_cpu.c` recomputes `lanczos_weight` 36 times per pixel. Pre-calculating 6 weights for $x$ and 6 for $y$ once per pixel would be more efficient.
*   **I/O Inefficiency (Low Risk)**: FITS headers are parsed multiple times for the same file to detect the Bayer pattern in both Phase 1 and Phase 2.

### 1.3 Logical & Architectural Issues
*   **Missing Error Context**: Many `PIPE_CHECK` and `CUDA_CHECK` macros report that an error occurred but do not identify the specific file path being processed at that moment.
*   **Extra Memory Copy in Pipeline**: In `pipeline.cu` Phase 2, `fits_load` loads into a temporary buffer which is then `memcpy`'d to pinned memory. `fits_load` should ideally load directly into pinned memory to avoid the extra copy and save time.

---

## 2. Improvement Plan

### Phase 1: Safety & Stability (Immediate)
1.  **Eliminate VLAs**: Replace stack-allocated VLAs in `integration.c` and `calibration.c` with a single heap-allocated buffer per thread (using OpenMP's `threadprivate` or by passing a pre-allocated workspace).
2.  **Fix CCL Memory**: Refactor `star_detect_cpu_ccl_com` to use a more memory-efficient storage for `CompStats` (e.g., a hash map or re-mapping labels to a contiguous range).
3.  **Thread-safe Randomness**: Replace `rand()` in `ransac.c` with `rand_r()` or a modern PCG/Xorshift generator to prepare for parallel alignment.

### Phase 2: Performance Optimization
1.  **CPU Mini-batching**: Update `pipeline_run_cpu` to process frames in mini-batches, accumulating sums/counts to reduce the peak memory footprint from O(N) to O(BatchSize).
2.  **Restore GPU Async**: Remove internal `cudaStreamSynchronize` calls from `star_detect_gpu_d2d`. Move the threshold calculation to a dedicated GPU kernel or use `cudaMemcpyAsync` for partial sums.
3.  **Optimize Integration Kernel**: Reduce `INTEGRATION_GPU_MAX_BATCH` or use shared memory to mitigate register spilling in the kappa-sigma kernel.
4.  **Lanczos Weight Pre-computation**: Optimize `lanczos_transform_cpu` to pre-calculate the 12 Lanczos weights per pixel.
5.  **Direct Pinned I/O**: Modify `fits_load` or provide a `fits_load_to_buffer` variant to allow loading directly into the pinned memory used by CUDA streams.

### Phase 3: Accuracy & Robustness
1.  **VNG Refinement**: Implement a more robust VNG gradient formula (e.g., the 5x5 sum-of-differences) to improve demosaicing quality.
2.  **Metadata Caching**: Cache Bayer patterns and image dimensions after the first read to avoid redundant FITS header parsing.
3.  **Enhanced Logging**: Update error macros to include the `filepath` currently being processed.
