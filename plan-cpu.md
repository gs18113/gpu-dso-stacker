# DSO Stacker — Full CPU Pipeline with OpenMP

## Context

The `--cpu` flag currently does nothing useful: `pipeline.cu` always uses GPU, and `--cpu` is rejected outright when the CSV has no pre-computed transforms (because star detection was GPU-only). The goal is a complete CPU execution path for all pipeline stages, accelerated with OpenMP, with zero impact on the GPU path's performance.

---

## What Exists vs What's Missing

| Stage | GPU path | CPU path |
|---|---|---|
| Debayering (VNG) | `debayer_gpu.cu` | **Nothing** |
| Moffat conv + threshold | `star_detect_gpu.cu` | **Nothing** |
| CCL + CoM | — | `star_detect_cpu.c` ✅ |
| RANSAC + DLT | — | `ransac.c` ✅ |
| Lanczos warp | `lanczos_gpu.cu` (used by pipeline) | `lanczos_cpu.c` ✅ (not wired into pipeline) |
| Integration | `integration_gpu.cu` (used by pipeline) | `integration.c` ✅ (not wired into pipeline) |
| Pipeline orchestrator | `pipeline.cu` (GPU always) | **Nothing** |

---

## Implementation Plan

### 1. Move `MoffatParams` to `dso_types.h`

`MoffatParams` is currently in `include/star_detect_gpu.h`, which pulls in `cuda_runtime.h`. A pure-C CPU pipeline can't include CUDA headers. Moving `MoffatParams` to `dso_types.h` (which has no CUDA deps) unblocks `pipeline_cpu.c` and `star_detect_cpu.c`.

- **`include/dso_types.h`** — append:
  ```c
  typedef struct { float alpha; float beta; } MoffatParams;
  ```
- **`include/star_detect_gpu.h`** — remove the `MoffatParams` typedef (it now comes from `dso_types.h`).
- **`include/pipeline.h`** — remove `#include "star_detect_gpu.h"`; `MoffatParams` now comes from `dso_types.h` which `pipeline.h` already includes.

---

### 2. `include/debayer_cpu.h` + `src/debayer_cpu.c` — VNG Debayer on CPU

Port the VNG algorithm from `debayer_gpu.cu` to plain C.

**Header** (`include/debayer_cpu.h`):
```c
DsoError debayer_cpu(const float *src, float *dst, int W, int H, BayerPattern pattern);
```
- `BAYER_NONE`: plain `memcpy`.
- Other patterns: pixel loop with 8-directional gradients.

**Implementation** (`src/debayer_cpu.c`):
- Same VNG algorithm as the GPU kernel: 8 directional gradients, threshold τ = mean(grads) + min(grads), select smooth directions, blend per-channel estimates, emit luminance L = 0.2126·R + 0.7152·G + 0.0722·B.
- Boundary pixels: clamp coordinates to [0, W-1] / [0, H-1] (match GPU zero-pad behaviour for simplicity: treat out-of-bounds reads as 0).
- **OpenMP**: `#pragma omp parallel for collapse(2) schedule(static)` on the outer `y, x` loops. Each pixel is independent.

---

### 3. New CPU functions in `star_detect_cpu.h` / `star_detect_cpu.c`

Add Moffat convolution + sigma threshold to the existing CPU star detection module.

**New declarations** in `include/star_detect_cpu.h`:
```c
/* 2-D convolution of src with Moffat kernel → dst (same size W×H). */
DsoError star_detect_cpu_moffat_convolve(const float *src, float *dst,
                                          int W, int H, const MoffatParams *params);

/* Compute global mean+σ of convolved; write mask: 1 where conv > mean + sigma_k*σ. */
DsoError star_detect_cpu_threshold(const float *convolved, uint8_t *mask,
                                    int W, int H, float sigma_k);

/* Combined: convolve + threshold in one call (used by pipeline_cpu). */
DsoError star_detect_cpu_detect(const float *src, float *conv_out, uint8_t *mask_out,
                                 int W, int H, const MoffatParams *params, float sigma_k);
```

**`src/star_detect_cpu.c`** — add to file:
- `star_detect_cpu_moffat_convolve`: pre-compute Moffat kernel (same formula as GPU: `K(i,j) = [1 + (i²+j²)/α²]^(-β)`, normalized, R = min((int)ceilf(3*α), 15)). Outer `y, x` pixel loops with `#pragma omp parallel for collapse(2) schedule(static)`. Inner kernel loops are serial per pixel. Clamp boundary reads to 0.
- `star_detect_cpu_threshold`: single-pass double-precision accumulation for mean and variance (Bessel-corrected). `#pragma omp parallel for reduction(+:sum)` and `reduction(+:sq)` for statistics; separate `#pragma omp parallel for` for mask write.
- `star_detect_cpu_detect`: calls `moffat_convolve` then `threshold`.

---

### 4. OpenMP in existing CPU modules

No algorithmic changes — add pragma annotations only.

**`src/lanczos_cpu.c`**:
- `#pragma omp parallel for schedule(static)` on the outer destination-row `dy` loop. Inner `dx` loop and 6×6 tap window are serial per pixel. Add `#include <omp.h>`.

**`src/integration.c`** (`integrate_mean`):
- `#pragma omp parallel for` on the outer pixel loop.

**`src/integration.c`** (`integrate_kappa_sigma`):
- `#pragma omp parallel for schedule(dynamic, 64)` on the outer pixel loop.
- Per-pixel VLAs (`float vals[n]`, `char active[n]`) are safe: each OpenMP thread has its own stack.

CCL pass in `star_detect_cpu.c` is **not parallelized** — the union-find raster scan has data dependencies between adjacent pixels. The post-CCL stats accumulation pass could be parallelized with `reduction` but is negligible compared to the convolution step.

---

### 5. `src/pipeline_cpu.c` + declarations in `pipeline.h`

Pure C orchestrator — no CUDA headers, no CUDA API calls.

**Declare in `pipeline.h`** (extern "C"):
```c
DsoError pipeline_run_cpu(FrameInfo *frames, int n_frames, int has_transforms,
                           int ref_idx, const PipelineConfig *config);
```

**Phase 1** (`!has_transforms` only):
```
determine W, H from reference frame (fits_load + image_free)
for each frame i:
    fits_load → raw
    debayer_cpu(raw, lum, W, H, bayer)
    alloc conv[W*H], mask[W*H]
    star_detect_cpu_detect(lum, conv, mask, params, sigma_k) → stars[i]
    free raw, lum, conv, mask
for each non-ref frame i:
    ransac_compute_homography(&ref_stars, &stars[i], &ransac, &frames[i].H, NULL)
free star lists
```

**Phase 2**:
```
alloc transformed[n_frames]  // Image structs, data calloc'd per frame
for each frame i:
    fits_load → raw
    debayer_cpu(raw, lum, W, H, bayer)
    lanczos_transform_cpu(&lum, &transformed[i], &frames[i].H)
    image_free(raw), image_free(lum)

const Image *ptrs[n_frames] = { &transformed[0], ... }
if use_kappa_sigma:
    integrate_kappa_sigma(ptrs, n_frames, &out, kappa, iters)
else:
    integrate_mean(ptrs, n_frames, &out)

image_free all transformed frames
fits_save(config->output_file, &out)
image_free(&out)
```

Memory: `n_frames × W × H × 4` bytes (≈ 654 MB for the 10-frame test set — acceptable).

---

### 6. `pipeline.cu` — early dispatch to CPU path

Add at the very top of `pipeline_run()` body, before any CUDA resource allocation:
```c
if (!config->use_gpu_lanczos)
    return pipeline_run_cpu(frames, n_frames, has_transforms, ref_idx, config);
```
Everything below this line is the existing GPU path — completely unchanged.

---

### 7. `main.cpp` — remove `--cpu` restriction

Remove these lines (around line 249–252):
```cpp
if (!has_transforms && use_cpu) {
    fprintf(stderr, "Error: star detection requires GPU; "
                    "--cpu is incompatible with 2-column CSV input\n");
    return 1;
}
```
`--cpu` now works with both CSV formats.

---

### 8. `CMakeLists.txt`

```cmake
find_package(OpenMP REQUIRED)

# Add to LIB_SOURCES:
#   src/debayer_cpu.c
#   src/pipeline_cpu.c

# Add to target_link_libraries(dso_stacker_lib PUBLIC ...):
#   OpenMP::OpenMP_C

# New test target:
add_executable(test_debayer_cpu tests/test_debayer_cpu.c)
target_include_directories(test_debayer_cpu PRIVATE tests)
target_link_libraries(test_debayer_cpu PRIVATE dso_stacker_lib m)
add_test(NAME test_debayer_cpu COMMAND test_debayer_cpu)
```

---

### 9. `tests/test_debayer_cpu.c`

| Test | What it checks |
|---|---|
| `test_none_passthrough` | `BAYER_NONE` → output buffer identical to input |
| `test_null_args` | NULL src/dst → `DSO_ERR_INVALID_ARG` |
| `test_zero_dims` | W=0 or H=0 → `DSO_ERR_INVALID_ARG` |
| `test_rggb_pure_red` | 4×4 image with only R channel set → luminance = 0.2126·R |
| `test_rggb_pure_green` | Green-only → luminance = 0.7152·G |
| `test_uniform_any_pattern` | Uniform image (all pixels same) → uniform luminance output |
| `test_bggr_pattern` | BGGR dispatches without crash, output non-zero |

Also extend `tests/test_star_detect.c` with tests for `star_detect_cpu_detect`:
- `test_moffat_convolve_uniform`: uniform input → uniform output (normalization check)
- `test_threshold_no_stars`: low-variance image → empty mask
- `test_threshold_bright_star`: spike pixel → mask = 1 at that location

---

## Files Summary

| Action | File |
|---|---|
| **Modify** | `include/dso_types.h` — add `MoffatParams` |
| **Modify** | `include/star_detect_gpu.h` — remove `MoffatParams` typedef |
| **Modify** | `include/pipeline.h` — remove `star_detect_gpu.h` include; add `pipeline_run_cpu` decl |
| **Modify** | `include/star_detect_cpu.h` — add Moffat+threshold+detect declarations |
| **Modify** | `src/star_detect_cpu.c` — add moffat_convolve, threshold, detect with OpenMP |
| **Modify** | `src/lanczos_cpu.c` — `omp parallel for` on outer pixel loop |
| **Modify** | `src/integration.c` — `omp parallel for` on pixel loops |
| **Modify** | `src/pipeline.cu` — early dispatch to `pipeline_run_cpu()` |
| **Modify** | `main.cpp` — remove `--cpu` + 2-col CSV restriction |
| **Modify** | `CMakeLists.txt` — OpenMP, new sources, new test |
| **Create** | `include/debayer_cpu.h` |
| **Create** | `src/debayer_cpu.c` — VNG with OpenMP |
| **Create** | `src/pipeline_cpu.c` — full CPU orchestrator (no CUDA) |
| **Create** | `tests/test_debayer_cpu.c` |

---

## Verification

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --parallel $(nproc)

# All tests must pass (including new test_debayer_cpu)
cd build && ctest --output-on-failure -V

# GPU pipeline — unchanged behavior
./build/dso_stacker -f data/frames.csv -o stacked_gpu.fits

# Full CPU pipeline — both 11-col and 2-col CSV
./build/dso_stacker -f data/frames.csv -o stacked_cpu.fits --cpu
```
