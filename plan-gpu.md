# DSO Stacker — Full Pipeline Implementation Plan

## Context

The existing repo has stages 4 (Lanczos transform) and 5 (CPU kappa-sigma integration) implemented. This plan fills in stages 1–3 (preprocessing/debayering, star detection, RANSAC homography) plus a GPU mini-batch integration engine, wiring them together with a pipelined orchestrator (`pipeline.cu`) that overlaps computation and data transfer using CUDA streams.

The CSV parser already supports both 2-column (no transforms) and 11-column (pre-computed homographies) formats. When the 11-column format is used, star detection and RANSAC are skipped entirely.

---

## Files Overview

### Already Implemented (read-only unless noted)
- `src/fits_io.c` + `include/fits_io.h` — FITS I/O; `fits_get_bayer_pattern` declared (may need implementation)
- `src/csv_parser.c` + `include/csv_parser.h` — CSV parsing
- `src/lanczos_cpu.c` + `include/lanczos_cpu.h` — CPU Lanczos-3
- `src/lanczos_gpu.cu` + `include/lanczos_gpu.h` — GPU Lanczos-3 (h2h and d2d variants)
- `src/integration.c` + `include/integration.h` — CPU mean + kappa-sigma
- `include/dso_types.h` — StarPos, StarList, BayerPattern, DsoError, Homography, Image, FrameInfo
- `CMakeLists.txt` — already lists all new source files and test targets (no changes needed)

### To Create
| File | Purpose |
|---|---|
| `include/debayer_gpu.h` + `src/debayer_gpu.cu` | VNG debayer → luminance on GPU |
| `include/star_detect_gpu.h` + `src/star_detect_gpu.cu` | Moffat convolution + sigma threshold |
| `include/star_detect_cpu.h` + `src/star_detect_cpu.c` | CCL + weighted center-of-mass |
| `include/ransac.h` + `src/ransac.c` | Star matching + RANSAC + DLT homography |
| `include/integration_gpu.h` + `src/integration_gpu.cu` | GPU mini-batch kappa-sigma + accumulation |
| `include/pipeline.h` + `src/pipeline.cu` | Full pipeline orchestrator with stream overlap |
| `tests/test_star_detect.c` | CCL + CoM unit tests |
| `tests/test_ransac.c` | DLT + RANSAC unit tests |
| `tests/test_integration_gpu.cu` | GPU integration unit tests |

### To Modify
- `src/fits_io.c` — implement `fits_get_bayer_pattern` if missing
- `main.cpp` — add new CLI args, replace transform loop with `pipeline_run()`

---

## Module Specifications

### 1. `debayer_gpu` — VNG Debayering → Luminance

**Algorithm**: VNG (Variable Number of Gradients) debayering. For each pixel:
1. Determine which channel it represents based on BayerPattern and position parity.
2. Compute gradients in 8 compass directions (N/NE/E/SE/S/SW/W/NW), each spanning 2 pixels to cross a full CFA period.
3. Threshold: only use directions with gradient below `mean_gradient + threshold`. This is the "variable number" of VNG.
4. Estimate missing channels from the selected smooth directions.
5. Convert to luminance: `L = 0.2126*R + 0.7152*G + 0.0722*B` (ITU-R BT.709, correct for linear astronomical data).

**CUDA kernel**: `vng_debayer_kernel<<<grid, block, smem>>>`
- Block: 16×16 threads
- Grid: `((W+15)/16, (H+15)/16)`
- Shared memory: `(16+4) × (16+4) × sizeof(float) = 1600 bytes` (2-pixel apron for 5×5 VNG neighborhood)
- Boundary handling: zero-pad apron (acceptable edge artifacts for star detection)

**Fast path**: When `pattern == BAYER_NONE`, `cudaMemcpyAsync(d_dst, d_src, ...)` instead of launching kernel.

**API**:
```c
DsoError debayer_gpu_d2d(const float *d_src, float *d_dst,
                          int W, int H, BayerPattern pattern, cudaStream_t stream);
// h2h convenience wrapper (used in tests):
DsoError debayer_gpu(const Image *src, Image *dst, BayerPattern pattern, cudaStream_t stream);
```

---

### 2. `star_detect_gpu` — Moffat Convolution + Sigma Thresholding

**Moffat kernel**: `K(i,j) = [1 + (i²+j²)/alpha²]^(-beta)`, normalized to sum=1.
- Kernel radius: `R = ceil(3 * alpha)` (e.g., R=8 for alpha=2.5)
- Pre-computed on CPU as a `(2R+1)²` float array, uploaded to device constant memory `__constant__ float c_kernel[]` (max kernel diameter 31, fits 961 floats × 4 = 3.8 KB within 64 KB constant limit).

**Convolution kernel**: `moffat_conv_kernel<<<grid, block, smem>>>`
- Block: 16×16 threads; Grid: `((W+15)/16, (H+15)/16)`
- Shared memory: `(16+2R)² × sizeof(float)` — for R=8 this is 32²×4 = 4 KB
- Reads from shared memory tile, multiplies by `c_kernel`, accumulates
- Boundary: zero-pad (skipping boundary reweighting acceptable for detection)

**Threshold**: Two-pass GPU reduction for mean and variance of the convolved image, then element-wise `d_mask[i] = (d_conv[i] > mean + sigma_k * sigma) ? 1 : 0`.
- Pass 1: `reduce_mean_kernel` (256 threads/block, tree reduction to partial sums, host-side divide)
- Pass 2: `reduce_variance_kernel` (same pattern, using pre-computed mean)
- Pass 3: `threshold_kernel`

**API**:
```c
typedef struct { float alpha; float beta; } MoffatParams;

// Combined convolve + threshold (device-to-device for pipeline):
DsoError star_detect_gpu_d2d(const float *d_src, float *d_conv, uint8_t *d_mask,
                              int W, int H, const MoffatParams *params,
                              float sigma_k, cudaStream_t stream);
// h2h wrapper for testing:
DsoError star_detect_gpu_moffat_convolve(const Image *src, Image *dst,
                                          const MoffatParams *params, cudaStream_t stream);
DsoError star_detect_gpu_threshold(const Image *convolved, uint8_t *mask_out,
                                    float sigma_k, cudaStream_t stream);
```

---

### 3. `star_detect_cpu` — CCL + Weighted Center-of-Mass

**Algorithm**: Two-pass 8-connectivity CCL using union-find:
1. **Pass 1** (top-left scan): assign provisional labels; when pixel at (x,y) is a star pixel (mask=1), check 4 already-visited neighbors (N, NW, W, NE). Assign min label of all active neighbors; union their sets.
2. **Pass 2**: resolve each pixel's label to its union-find root (path-compressed).
3. **Per-component statistics**: for each component, accumulate:
   - `flux = Σ conv_pixel` (convolved values, for ranking)
   - `sum_w = Σ max(0, orig_pixel)` (clamped original values, for CoM weights)
   - `sum_wx = Σ max(0, orig_pixel) * x`
   - `sum_wy = Σ max(0, orig_pixel) * y`
4. **CoM**: `x_com = sum_wx / sum_w`, `y_com = sum_wy / sum_w`. If `sum_w ≈ 0`, use geometric centroid instead.
5. **Ranking**: sort components by `flux` descending, return top `top_k` as `StarList`.

**Mathematical note**: Using original (non-convolved) pixels as weights is correct astrophotometry practice — the convolved image is only used for detection, while CoM precision comes from the raw signal distribution.

**API**:
```c
DsoError star_detect_cpu_ccl_com(const uint8_t *mask, const float *original,
                                   const float *convolved, int W, int H,
                                   int top_k, StarList *list_out);
```

---

### 4. `ransac` — Star Matching + RANSAC Homography

**Star matching**: For each star in `ref_list`, find nearest star in `frm_list` within `match_radius` pixels (brute-force O(N²) for N ≤ 50 — 2500 comparisons, trivial). Apply ratio test: accept match only if `d_nearest / d_second_nearest < 0.8`. Build a list of `(ref_star, frm_star)` correspondences.

**DLT homography** (for RANSAC inner loop, N=4 correspondences):
For each correspondence `(ref_x, ref_y) → (src_x, src_y)`, two rows of the matrix A:
```
Row 1: [-ref_x, -ref_y, -1,     0,     0,  0, src_x*ref_x, src_x*ref_y, src_x]
Row 2: [    0,      0,  0, -ref_x, -ref_y, -1, src_y*ref_x, src_y*ref_y, src_y]
```
H (row-major) is the null vector of A = right singular vector for smallest singular value.

**SVD via normal equations**: Compute `M = AᵀA` (9×9 symmetric), find eigenvector for smallest eigenvalue via Jacobi iteration. This avoids implementing full SVD while being numerically adequate for pixel-precision star coordinates.

**Normalization**: Translate ref_pts centroid to origin, scale so mean distance to origin = √2 before building A. Apply inverse normalization to H afterward. This improves numerical stability.

**RANSAC loop**:
```
best_H, best_inliers = I, 0
adaptive_max_iters = ransac_iters  // updated each time we find a better model
for iter in [0, adaptive_max_iters):
    sample = random 4 correspondences
    H = dlt_homography(sample)       // null-vector of 8×9 A
    inliers = count where |H * p_ref - p_src| / w < inlier_thresh
    if inliers > best_inliers:
        best_H = H; best_inliers = inliers
        // Adaptive termination:
        p = inliers / n_matches
        adaptive_max_iters = min(ransac_iters, ceil(log(1-0.99) / log(1 - p^4)))
if best_inliers < min_inliers: return DSO_ERR_RANSAC
// Refinement: refit DLT using all inlier correspondences
H_out = dlt_homography(best_inlier_correspondences)
```

**Result H is the backward map (ref → src)**: The DLT row setup above directly produces this convention because ref_x/y are the "input" of H and src_x/y are the "output". **No inversion needed.** (See CLAUDE.md homography convention.)

**API**:
```c
typedef struct {
    int   max_iters;      /* default: 1000 */
    float inlier_thresh;  /* pixels, default: 2.0 */
    float match_radius;   /* pixels, default: 30.0 */
    float confidence;     /* default: 0.99 */
    int   min_inliers;    /* default: 4 */
} RansacParams;

DsoError ransac_compute_homography(const StarList *ref_list, const StarList *frm_list,
                                    const RansacParams *params,
                                    Homography *H_out, int *n_inliers_out);

DsoError dlt_homography(const StarPos *ref_pts, const StarPos *src_pts,
                         int n, Homography *H_out);
```

---

### 5. `integration_gpu` — GPU Mini-Batch Kappa-Sigma

**Strategy**: Per-batch kappa-sigma; combine batches with per-pixel survivor-weighted mean.

For each batch of M frames:
1. All M transformed frames reside in GPU memory (`d_frames[0..M-1]`).
2. `kappa_sigma_gpu_kernel` — 1 thread per pixel, inner loop over M values:
   ```
   float vals[M];      // M ≤ 32 stack array (fits in registers for M ≤ 32)
   int   mask[M];      // 1 = active
   // iterative sigma clipping (mirrors CPU integrate_kappa_sigma)
   // → writes d_partial_mean[px] and d_count[px] (surviving pixel count)
   ```
   - Block: 256 threads (1D); Grid: `((W*H + 255)/256)`
   - Register usage for M=16: ~26 registers/thread → high occupancy on sm_86

3. **Combining batches** (all-GPU, separate kernel):
   ```
   // After each batch:
   d_combined_sum[px]   += d_partial_mean[px] * d_count[px]
   d_combined_count[px] += d_count[px]
   // At the end:
   d_out[px] = (d_combined_count[px] > 0)
       ? d_combined_sum[px] / d_combined_count[px]
       : d_rawsum[px] / n_frames          // degenerate fallback
   ```
   The `d_rawsum` accumulator is updated alongside (unclipped per-pixel sum for fallback).

**GPU memory footprint** (example: 4096×4096, M=16):
- M destination frame buffers: 16 × 64 MB = 1024 MB
- xmap, ymap (Lanczos): 2 × 64 MB = 128 MB
- Accumulators (sum, count, rawsum): 3 × 64 MB ≈ 192 MB
- Total ≈ 1.4 GB — comfortable on 8GB GPU

**Opaque context struct**:
```c
typedef struct {
    float  *d_combined_sum;   /* W*H floats, zeroed at init */
    float  *d_rawsum;         /* W*H floats, unclipped accumulator */
    int    *d_combined_count; /* W*H ints, zeroed at init */
    float  *d_frames[MAX_BATCH]; /* batch_size device frame buffers */
    float  *d_xmap, *d_ymap; /* Lanczos coord maps, shared across frames */
    int     W, H, batch_size;
} IntegrationGpuCtx;
```

**API**:
```c
DsoError integration_gpu_init(int W, int H, int batch_size, IntegrationGpuCtx **ctx_out);
void     integration_gpu_cleanup(IntegrationGpuCtx *ctx);

// Process one mini-batch (M ≤ batch_size transformed frames already on device):
DsoError integration_gpu_process_batch(IntegrationGpuCtx *ctx, int M,
                                        float kappa, int iterations,
                                        cudaStream_t stream);

// Final pass: divide combined_sum by combined_count → output host image:
DsoError integration_gpu_finalize(IntegrationGpuCtx *ctx, int n_frames,
                                   Image *out, cudaStream_t stream);
```

---

### 6. `pipeline` — Orchestrator with CUDA Stream Overlap

**Two streams + double-buffered pinned memory**:
- `stream_copy` (streamB): async H2D DMA transfers
- `stream_compute` (streamA): GPU kernels (debayer, detect, Lanczos, integrate)
- Pinned host memory: `float *pinned[2]` — double buffer, `pinned[i % 2]` for frame i
- CUDA events: `e_h2d[2]` (H2D done), `e_gpu[2]` (GPU processing done)

**Phase 1: Star Detection (only for 2-col CSV)**

```
Frame i=0:  [CPU: read to pinned[0]] [streamB: H2D f0] → [streamA: debayer+detect f0] → [CPU: D2H mask, CCL+CoM f0]
Frame i=1:  [CPU: read to pinned[1]] ← overlaps with streamA processing f0
            → [streamB: H2D f1] ← waits for e_gpu[0] (reuses d_src slot 0)
            → [streamA: debayer+detect f1]
            → [CPU: CCL+CoM f1; RANSAC f1 against ref_stars]
```

After CCL+CoM for all frames and reference frame, run RANSAC for each non-reference frame against ref_stars. Store computed `H` back into `frames[i].H`.

**Phase 2: Lanczos Transform + Integration**

For each mini-batch (M frames):
```
For i in batch:
  pinned[i%2] ← fits_load (CPU, blocking disk I/O)
  streamB: cudaMemcpyAsync(d_frames[i%batch]*) ← overlaps with streamA on previous frame
  cudaStreamWaitEvent(streamA, e_h2d[i%2])
  streamA: debayer_gpu_d2d (if color)
  streamA: cudaMemsetAsync(d_dst[i], 0)
  streamA: lanczos_transform_gpu_d2d(d_debay, d_dst[i], d_xmap, d_ymap, ..., streamA)
  cudaEventRecord(e_gpu[i%2], streamA)
  // While streamA runs Lanczos for frame i, CPU reads frame i+1 from disk
  // and streamB transfers it

// After all M frames transformed:
cudaStreamSynchronize(streamA)
integration_gpu_process_batch(ctx, M, kappa, iterations, streamA)
cudaStreamSynchronize(streamA)
```

**Important synchronization points**:
1. Before overwriting `pinned[i%2]` with next frame: ensure H2D for current frame on `stream_copy` is done → `cudaEventSynchronize(e_h2d[i%2])`
2. Before launching GPU kernel on `stream_compute` that reads from `d_src[i%2]`: `cudaStreamWaitEvent(stream_compute, e_h2d[i%2], 0)`
3. Before CPU can use the D2H mask: `cudaEventSynchronize(e_gpu[i%2])`

**Key overlap**: CPU disk I/O for frame i+1 overlaps with GPU Lanczos transform for frame i. Since disk I/O is typically the bottleneck, this overlap is where most of the performance gain comes from.

**`PipelineConfig` struct** (passed from `main.cpp`):
```c
typedef struct {
    float        star_sigma;      /* threshold sigma multiplier (3.0) */
    float        moffat_alpha;    /* (2.5) */
    float        moffat_beta;     /* (2.0) */
    int          top_stars;       /* top-K stars for matching (50) */
    int          min_stars;       /* min stars for RANSAC (6) */
    int          ransac_iters;    /* (1000) */
    float        ransac_thresh;   /* inlier threshold px (2.0) */
    float        match_radius;    /* star match search radius px (30.0) */
    int          batch_size;      /* frames per mini-batch (16) */
    float        kappa;           /* kappa-sigma threshold (3.0) */
    int          iterations;      /* kappa-sigma iterations (3) */
    bool         use_kappa_sigma; /* true=kappa-sigma, false=mean */
    const char  *output_file;
    BayerPattern bayer_override;  /* BAYER_NONE = auto-detect from FITS header */
    bool         use_gpu_lanczos;
} PipelineConfig;

DsoError pipeline_run(FrameInfo *frames, int n_frames, int has_transforms,
                       int ref_idx, const PipelineConfig *config);
```

---

## `main.cpp` Changes

Replace the existing transform loop + integration block with a single `pipeline_run()` call.

New CLI arguments (use `enum { OPT_STAR_SIGMA=256, ... }` to avoid collision with existing single-char options):

| Long option | Default | Description |
|---|---|---|
| `--star-sigma <float>` | 3.0 | CoM threshold: accept pixels > mean + sigma_k × sigma |
| `--batch-size <int>` | 16 | Frames per integration mini-batch |
| `--moffat-alpha <float>` | 2.5 | Moffat kernel alpha (FWHM control) |
| `--moffat-beta <float>` | 2.0 | Moffat kernel beta (wing heaviness) |
| `--top-stars <int>` | 50 | Top-K stars to use for matching |
| `--min-stars <int>` | 6 | Minimum stars for RANSAC to attempt |
| `--ransac-iters <int>` | 1000 | Max RANSAC iterations |
| `--ransac-thresh <float>` | 2.0 | Inlier reprojection threshold (pixels) |
| `--match-radius <float>` | 30.0 | Star matching search radius (pixels) |
| `--bayer <pattern>` | auto | Override FITS BAYERPAT: none/rggb/bggr/grbg/gbrg |

Validation: if 2-col CSV and `--cpu` flag → error (star detection requires GPU).

---

## `fits_io.c` Addition

If `fits_get_bayer_pattern` is not yet implemented, add it: open the FITS file with CFITSIO, read the `BAYERPAT` keyword string, map to `BayerPattern` enum case-insensitively, close file. Missing keyword → `BAYER_NONE` + `DSO_OK`.

---

## Key Pitfalls to Avoid

1. **DLT direction**: Row setup uses `(ref_x, ref_y)` as H's input and `(src_x, src_y)` as output → produces backward map directly (no inversion).
2. **Homogeneous divide in RANSAC reprojection**: `q = H * p_ref`, then `q_x = q[0]/q[2]`, `q_y = q[1]/q[2]` before computing reprojection error.
3. **Pinned buffer reuse**: must call `cudaEventSynchronize(e_h2d[i%2])` before CPU overwrites `pinned[i%2]`.
4. **Moffat kernel constant memory cap**: `(2*ceil(3*alpha)+1)² ≤ ~961` floats for alpha ≤ 5. Assert or clamp alpha at startup.
5. **CCL union-find path compression**: always chase to root (`while (parent[x] != x) x = parent[x]`), not just one hop.
6. **Negative pixel weights in CoM**: clamp `max(0, original_val)` before using as CoM weight to prevent centroid drift.
7. **`dim3` before `CHECK_CUDA` in .cu files**: follow existing `lanczos_gpu.cu` convention to avoid C++ jump-over-initialization.
8. **`d_mask` D2H timing**: the thresholded mask D2H must complete before CCL runs on CPU. Use `cudaEventSynchronize` on the event recorded after `threshold_kernel` + `cudaMemcpyAsync`.

---

## Implementation Sequence

1. `src/fits_io.c` — implement `fits_get_bayer_pattern` if missing (no deps)
2. `include/star_detect_cpu.h` + `src/star_detect_cpu.c` — CCL + CoM (no CUDA)
3. `include/ransac.h` + `src/ransac.c` — DLT + RANSAC (no CUDA)
4. `tests/test_star_detect.c` + `tests/test_ransac.c` — test steps 2–3
5. `include/debayer_gpu.h` + `src/debayer_gpu.cu` — VNG debayer kernel
6. `include/star_detect_gpu.h` + `src/star_detect_gpu.cu` — Moffat + threshold
7. `include/integration_gpu.h` + `src/integration_gpu.cu` — mini-batch kappa-sigma
8. `tests/test_integration_gpu.cu` — test step 7
9. `include/pipeline.h` + `src/pipeline.cu` — full orchestrator
10. `main.cpp` — new CLI args + call `pipeline_run()`

---

## Verification

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --parallel $(nproc)

# Run all tests
cd build && ctest --output-on-failure -V

# End-to-end test (2-col CSV, star detection mode):
./build/dso_stacker -f frames_2col.csv -o stacked.fits \
    --star-sigma 3.0 --batch-size 8 --top-stars 30

# End-to-end test (11-col CSV, pre-computed transforms):
./build/dso_stacker -f data/transform_mat.csv -o stacked.fits \
    --kappa 3.0 --batch-size 16

# Compare GPU integration output to CPU reference:
# python3 python/stacker.py -f data/transform_mat.csv -o ref.fits
# Then compare pixel statistics between stacked.fits and ref.fits
```

Test cases for new modules (see Plan agent output for full list):
- CCL: single blob CoM, two blobs, 8-connectivity merge, negative weight clamping, top-K ranking, empty mask
- RANSAC: identity H, pure translation, known homography, outlier rejection, insufficient stars
- GPU integration: constant frames = exact mean, outlier rejection vs CPU reference, batch boundary (N=17 with M=16)
