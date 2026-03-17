# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Maintenance rule**: Whenever an important discovery, convention clarification, new implementation, API changes, or "aha moment" arises during a session, update this file immediately so future sessions start with the full picture.

## Project Overview

A high-performance Deep Sky Object (DSO) image stacker using C/CUDA for GPU-accelerated processing. Python reference stacker and transform-verification tools live in `python/`.

## Processing Pipeline

All stages are implemented for both GPU and CPU execution paths. The pipeline runs end-to-end from raw FITS frames to a stacked output.

| Stage | GPU path | CPU path |
|---|---|---|
| Debayering (VNG) | `debayer_gpu.cu` | `debayer_cpu.c` |
| Moffat conv + threshold | `star_detect_gpu.cu` | `star_detect_cpu.c` |
| CCL + CoM | — (CPU always) | `star_detect_cpu.c` |
| RANSAC + DLT | — (CPU always) | `ransac.c` |
| Lanczos warp | `lanczos_gpu.cu` | `lanczos_cpu.c` |
| Integration | `integration_gpu.cu` | `integration.c` |
| Pipeline orchestrator | `pipeline.cu` | `pipeline_cpu.c` |

Pass `--cpu` to run the complete CPU path; omit it for the GPU path (default).
When the input CSV already contains homographies (11-column format), stages 1–3 are skipped in both paths.

## Technology Stack

- **C11** — Core library (`fits_io.c`, `csv_parser.c`, `lanczos_cpu.c`, `integration.c`, `debayer_cpu.c`, `star_detect_cpu.c`, `ransac.c`, `pipeline_cpu.c`)
- **OpenMP** — CPU parallelism (debayer, Moffat convolution, Lanczos, integration all use `#pragma omp parallel for`)
- **CUDA 12 / NPP+** — GPU acceleration (`lanczos_gpu.cu`, `debayer_gpu.cu`, `star_detect_gpu.cu`, `integration_gpu.cu`, `pipeline.cu`)
- **CFITSIO 4.6.3** — FITS image I/O
- **C++17** — CLI entry point (`main.cpp`)

---

## File Structure

```
gpu-dso-stacker/
├── CMakeLists.txt               ← CMake build definition
├── main.cpp                     ← CLI entry point (getopt_long)
├── bench.sh                     ← GPU vs CPU benchmark script
├── include/
│   ├── dso_types.h              ← Shared types: Image, Homography, FrameInfo, DsoError,
│   │                               StarPos, StarList, BayerPattern, MoffatParams
│   ├── fits_io.h                ← FITS load/save/free + fits_get_bayer_pattern
│   ├── csv_parser.h             ← CSV frame-list parser (2-col and 11-col formats)
│   ├── lanczos_cpu.h            ← CPU Lanczos-3 transform API
│   ├── lanczos_gpu.h            ← GPU Lanczos-3 transform API (h2h and d2d)
│   ├── integration.h            ← CPU mean / kappa-sigma integration API
│   ├── debayer_cpu.h            ← VNG debayer → luminance (CPU, OpenMP)
│   ├── debayer_gpu.h            ← VNG debayer → luminance (GPU, h2h and d2d)
│   ├── star_detect_gpu.h        ← Moffat convolution + sigma threshold (GPU, h2h and d2d)
│   ├── star_detect_cpu.h        ← Moffat conv + threshold + CCL + CoM (CPU)
│   ├── ransac.h                 ← DLT homography + RANSAC alignment
│   ├── integration_gpu.h        ← GPU mini-batch kappa-sigma context + API
│   └── pipeline.h               ← pipeline_run (GPU) + pipeline_run_cpu + PipelineConfig
├── src/
│   ├── fits_io.c                ← CFITSIO-based I/O
│   ├── csv_parser.c             ← CSV parser
│   ├── lanczos_cpu.c            ← CPU backward-mapping Lanczos-3 (OpenMP)
│   ├── lanczos_gpu.cu           ← CUDA coord-map kernel + nppiRemap
│   ├── integration.c            ← CPU mean and kappa-sigma clipping (OpenMP)
│   ├── debayer_cpu.c            ← VNG debayer CPU implementation (OpenMP)
│   ├── debayer_gpu.cu           ← VNG debayer kernel (16×16 tiles, 2px apron)
│   ├── star_detect_gpu.cu       ← Moffat conv kernel + reduction + threshold
│   ├── star_detect_cpu.c        ← Moffat conv + threshold (OpenMP) + CCL + CoM
│   ├── ransac.c                 ← Jacobi eigendecomp DLT + RANSAC loop
│   ├── integration_gpu.cu       ← Mini-batch kappa-sigma + finalize kernels
│   ├── pipeline.cu              ← GPU orchestrator; dispatches to pipeline_cpu if --cpu
│   └── pipeline_cpu.c           ← Pure-C CPU orchestrator (no CUDA)
├── tests/
│   ├── test_framework.h         ← Minimal test harness (ASSERT_*, RUN, SUITE)
│   ├── test_cpu.c               ← 29 tests: CSV, FITS, integration, Lanczos CPU
│   ├── test_gpu.cu              ← 5 tests: GPU Lanczos (2 known pre-existing failures)
│   ├── test_star_detect.c       ← 21 tests: CCL + CoM + Moffat conv + threshold
│   ├── test_ransac.c            ← 13 tests: DLT + RANSAC
│   ├── test_debayer_cpu.c       ← 10 tests: VNG debayer CPU (all patterns + edge cases)
│   └── test_integration_gpu.cu  ← 9 tests: GPU mini-batch kappa-sigma
└── python/
    ├── stacker.py               ← Reference stacker: OpenCV Lanczos4 + kappa-sigma
    └── compute_transforms.py    ← Independently compute homographies via astroalign
```

---

## Build Instructions

### Prerequisites

| Dependency | Version | Location |
|---|---|---|
| CFITSIO | 4.6.3 | `/home/donut/.local` |
| CUDA Toolkit | 12.x | `/usr/local/cuda` |
| NPP+ (`libnppig`) | bundled with CUDA | `/usr/local/cuda/lib64` |
| OpenMP | any | system (GCC) |
| CMake | >= 3.18 | system |
| pkg-config | any | system |

### Build

```bash
# Configure (Debug)
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Configure (Release)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Compile
cmake --build build --parallel $(nproc)

# Binary location
./build/dso_stacker
```

CUDA architectures are hardcoded to `86;89` (RTX 30xx / 40xx) in `CMakeLists.txt`.
Adjust `CUDA_ARCHITECTURES` if targeting a different GPU family.

---

## CLI Usage

```
dso_stacker -f <frames.csv> [options]

I/O:
  -f, --file <path>              Input CSV file (required)
  -o, --output <path>            Output FITS file (default: output.fits)

Integration:
      --cpu                      Run ALL pipeline stages on CPU (OpenMP-accelerated)
      --integration <method>     mean | kappa-sigma (default: kappa-sigma)
      --kappa <float>            Sigma clipping threshold (default: 3.0)
      --iterations <int>         Max clipping passes per pixel (default: 3)
      --batch-size <int>         GPU integration mini-batch size (default: 16)

Star detection (2-column CSV only):
      --star-sigma <float>       Detection threshold in σ units (default: 3.0)
      --moffat-alpha <float>     Moffat PSF alpha / FWHM (default: 2.5)
      --moffat-beta <float>      Moffat PSF beta / wing slope (default: 2.0)
      --top-stars <int>          Top-K stars for matching (default: 50)
      --min-stars <int>          Minimum stars for RANSAC (default: 6)

RANSAC (2-column CSV only):
      --ransac-iters <int>       Max RANSAC iterations (default: 1000)
      --ransac-thresh <float>    Inlier reprojection threshold px (default: 2.0)
      --match-radius <float>     Star matching search radius px (default: 30.0)

Sensor:
      --bayer <pattern>          CFA override: none | rggb | bggr | grbg | gbrg
```

### Input CSV Format

```
filepath, is_reference, h00, h01, h02, h10, h11, h12, h20, h21, h22
/data/frame1.fits, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1
/data/frame2.fits, 0, 1, 0, 2.5, 0, 1, 1.3, 0, 0, 1
```

- First row is a header and is always skipped.
- Exactly **one** row must have `is_reference = 1`.
- The nine `h` values form a **row-major 3x3 backward homography** mapping reference pixel coordinates to source pixel coordinates (ref → src). Despite being labelled "forward" in some upstream docs, empirical testing confirms this is the backward/inverse map — use it directly for pixel sampling without inverting.

---

## API Reference

### `dso_types.h` — Shared Types

```c
typedef struct { double h[9]; }  Homography;   // row-major 3x3, backward map (ref → src)
typedef struct { float *data; int width; int height; } Image;  // row-major float32
typedef struct { char filepath[4096]; int is_reference; Homography H; } FrameInfo;
typedef struct { float x, y, flux; } StarPos;  // sub-pixel CoM position + integrated flux
typedef struct { StarPos *stars; int n; } StarList;  // heap-alloc'd, caller must free()
typedef struct { float alpha; float beta; } MoffatParams;  // Moffat PSF params (default: 2.5, 2.0)
typedef enum { BAYER_NONE=0, BAYER_RGGB, BAYER_BGGR, BAYER_GRBG, BAYER_GBRG } BayerPattern;
typedef enum { DSO_OK=0, DSO_ERR_IO=-1, DSO_ERR_ALLOC=-2, DSO_ERR_FITS=-3,
               DSO_ERR_CUDA=-4, DSO_ERR_NPP=-5, DSO_ERR_CSV=-6,
               DSO_ERR_INVALID_ARG=-7, DSO_ERR_STAR_DETECT=-8,
               DSO_ERR_RANSAC=-9 } DsoError;
```

`MoffatParams` lives in `dso_types.h` (not `star_detect_gpu.h`) so pure-C CPU code can use it without pulling in CUDA headers.

### `fits_io.h`

```c
DsoError fits_load(const char *filepath, Image *out);
DsoError fits_save(const char *filepath, const Image *img);
void     image_free(Image *img);
```

- `fits_load`: opens FITS, reads any BITPIX as float32. Caller owns `out->data`.
- `fits_save`: writes float32 image as BITPIX=-32. Overwrites existing files.
- `image_free`: frees `img->data` and nulls the pointer. Safe on zero-init Images.

### `csv_parser.h`

```c
DsoError csv_parse(const char *filepath, FrameInfo **frames_out, int *n_frames_out);
```

Parses CSV, returns heap-allocated array. Caller must `free(*frames_out)`.

### `lanczos_cpu.h`

```c
DsoError lanczos_transform_cpu(const Image *src, Image *dst, const Homography *H);
```

CPU Lanczos-3 warp. `dst->data`, `dst->width`, `dst->height` must be pre-set by the caller.
H is the **backward homography (ref → src)**, used directly for pixel sampling — not inverted. Boundary taps are skipped and weights renormalised.

### `lanczos_gpu.h`

```c
DsoError lanczos_gpu_init(cudaStream_t stream);   // call once at startup
void     lanczos_gpu_cleanup(void);               // call once at shutdown
DsoError lanczos_transform_gpu(const Image *src, Image *dst, const Homography *H);
```

GPU path using CUDA coord-map kernel + `nppiRemap_32f_C1R_Ctx` (NPPI_INTER_LANCZOS).
`lanczos_gpu_init` populates `NppStreamContext` from `cudaGetDeviceProperties`.
Per-frame device allocations (d_src, d_dst, d_xmap, d_ymap) are freed inside the call.

### `integration.h`

```c
DsoError integrate_mean(const Image **frames, int n, Image *out);
DsoError integrate_kappa_sigma(const Image **frames, int n, Image *out,
                                float kappa, int iterations);
```

All frames must be the same size. `out->data` is heap-allocated; free with `image_free()`.
Kappa-sigma uses two-pass Bessel-corrected stddev per iteration; degenerate pixels (all clipped) fall back to unclipped mean.

### `debayer_gpu.h`

```c
DsoError debayer_gpu_d2d(const float *d_src, float *d_dst,
                          int W, int H, BayerPattern pattern, cudaStream_t stream);
DsoError debayer_gpu(const Image *src, Image *dst, BayerPattern pattern, cudaStream_t stream);
```

VNG debayer: 16×16 tiles with 2-pixel apron in shared memory; 8 directional gradients; luminance output via ITU-R BT.709 (L = 0.2126·R + 0.7152·G + 0.0722·B). `BAYER_NONE` = device-to-device copy (fast path).

### `debayer_cpu.h`

```c
DsoError debayer_cpu(const float *src, float *dst, int W, int H, BayerPattern pattern);
```

Same VNG algorithm as the GPU kernel. `BAYER_NONE` = `memcpy`. Boundary reads clamp to 0. Parallelized with `#pragma omp parallel for collapse(2) schedule(static)`.

### `star_detect_gpu.h`

```c
DsoError star_detect_gpu_d2d(const float *d_src, float *d_conv, uint8_t *d_mask,
                              int W, int H, const MoffatParams *params,
                              float sigma_k, cudaStream_t stream);
```

Moffat kernel `K(i,j) = [1 + (i²+j²)/alpha²]^(-beta)` stored in 64 KB constant memory (max radius 15). Convolution via 16×16 shared-memory tiles. Two-pass GPU reduction for mean+stddev; element-wise threshold `mask[i] = (conv[i] > mean + sigma_k·σ)`.

### `star_detect_cpu.h`

```c
/* Moffat convolution + sigma threshold (CPU equivalents of star_detect_gpu) */
DsoError star_detect_cpu_moffat_convolve(const float *src, float *dst,
                                          int W, int H, const MoffatParams *params);
DsoError star_detect_cpu_threshold(const float *convolved, uint8_t *mask,
                                    int W, int H, float sigma_k);
DsoError star_detect_cpu_detect(const float *src, float *conv_out, uint8_t *mask_out,
                                 int W, int H, const MoffatParams *params, float sigma_k);

/* CCL + center-of-mass (shared by both GPU and CPU pipelines) */
DsoError star_detect_cpu_ccl_com(const uint8_t *mask, const float *original,
                                   const float *convolved, int W, int H,
                                   int top_k, StarList *list_out);
```

`moffat_convolve`: pre-computed kernel (R = min(⌈3α⌉, 15)), zero-boundary padding, `collapse(2) omp parallel for`.
`threshold`: double-precision Bessel-corrected σ via `reduction(+:)`, then parallel mask write.
`ccl_com`: two-pass 8-connectivity union-find; not parallelized (raster scan has data dependencies).

### `ransac.h`

```c
typedef struct { int max_iters; float inlier_thresh; float match_radius;
                 float confidence; int min_inliers; } RansacParams;
DsoError dlt_homography(const StarPos *ref_pts, const StarPos *src_pts,
                         int n, Homography *H_out);
DsoError ransac_compute_homography(const StarList *ref_list, const StarList *frm_list,
                                    const RansacParams *params,
                                    Homography *H_out, int *n_inliers_out);
```

DLT via Jacobi eigendecomposition of A^T·A (9×9); point normalization for numerical stability. Nearest-neighbour star matching with Lowe ratio test (d1/d2 < 0.8). Adaptive RANSAC termination. Produces **backward homography (ref → src)** directly — no inversion needed.

### `integration_gpu.h`

```c
#define INTEGRATION_GPU_MAX_BATCH 64
DsoError integration_gpu_init(int W, int H, int batch_size, IntegrationGpuCtx **ctx_out);
void     integration_gpu_cleanup(IntegrationGpuCtx *ctx);
DsoError integration_gpu_process_batch(IntegrationGpuCtx *ctx, int M,
                                        float kappa, int iterations, cudaStream_t stream);
DsoError integration_gpu_process_batch_mean(IntegrationGpuCtx *ctx, int M, cudaStream_t stream);
DsoError integration_gpu_finalize(IntegrationGpuCtx *ctx, int n_frames,
                                   Image *out, cudaStream_t stream);
```

`IntegrationGpuCtx` is now a **public** struct (in the header) exposing `d_frames[]`, `d_xmap`, `d_ymap` so `pipeline.cu` can fill frames without extra indirection. Mini-batch approximation: per-batch kappa-sigma combined with survivor-count-weighted mean across batches. All-clipped fallback uses raw (unclipped) per-pixel sum.

### `pipeline.h`

```c
DsoError pipeline_run(FrameInfo *frames, int n_frames, int has_transforms,
                       int ref_idx, const PipelineConfig *config);
DsoError pipeline_run_cpu(FrameInfo *frames, int n_frames, int has_transforms,
                           int ref_idx, const PipelineConfig *config);
```

`pipeline_run`: GPU orchestrator. First line of the function body dispatches to `pipeline_run_cpu` when `config->use_gpu_lanczos == 0`, so the GPU path has zero overhead. Phase 2 uses double-buffered `stream_copy` + `stream_compute` overlap.

`pipeline_run_cpu`: pure-C orchestrator in `pipeline_cpu.c` (no CUDA headers). Phase 1: debayer_cpu → star_detect_cpu_detect → ccl_com → ransac per frame. Phase 2: debayer_cpu → lanczos_transform_cpu per frame (all frames kept in RAM), then integrate_kappa_sigma or integrate_mean, then fits_save.

---

## Key Implementation Notes

- **Homography convention**: H is the *backward* map (ref → src). Transform functions use it directly for pixel sampling — **do not invert**. The `invert_homography` helpers remain in the source but are dead code.
- **GPU Lanczos strategy**: `nppiWarpPerspective` only supports NN/LINEAR/CUBIC. We instead pre-compute backward-homography coordinate maps (H used directly, no inversion) in a CUDA kernel, then feed them to `nppiRemap_32f_C1R_Ctx` with `NPPI_INTER_LANCZOS` (= 16).
- **CPU dispatch is a no-cost early return**: `pipeline_run()` checks `!config->use_gpu_lanczos` as its very first statement and returns `pipeline_run_cpu(...)`. No GPU resources are allocated, no CUDA context is created.
- **`MoffatParams` lives in `dso_types.h`**: moved from `star_detect_gpu.h` (which includes `cuda_runtime.h`) so that `pipeline_cpu.c` and `star_detect_cpu.c` can use it without any CUDA dependency.
- **OpenMP parallelism strategy**:
  - `debayer_cpu`: `collapse(2) schedule(static)` — each pixel is independent.
  - `star_detect_cpu_moffat_convolve`: `collapse(2) schedule(static)` — inner kernel loops serial per pixel.
  - `star_detect_cpu_threshold`: three passes — `reduction(+:sum)`, `reduction(+:sq)`, then parallel mask write.
  - `lanczos_cpu`: `schedule(static)` on outer `dy` row loop.
  - `integrate_mean`: pixel-outer loop (restructured from frame-outer) — `schedule(static)`.
  - `integrate_kappa_sigma`: `schedule(dynamic, 64)` — VLA `float vals[n]` / `int actv[n]` declared inside loop body for per-thread stack allocation.
  - `star_detect_cpu_ccl_com` (CCL pass): **not parallelized** — union-find raster scan has cross-pixel data dependencies.
- **C goto rule (C11 and C++17)**: variables that appear between a `goto` source and its label must be declared before the first `goto`. In `pipeline_cpu.c`, all variables are declared at the top of each function; error paths use explicit `image_free()` before `goto cleanup` rather than a PIPE_CHECK macro, to avoid jumping over initializers.
- **Row steps for NPP**: all row steps are `width * sizeof(float)` (row-major float32).
- **FITS pixel index**: `ffgpxv` / `ffppx` take `long *firstpix` (1-based per axis); pass `{1, 1}`.
- **Out-of-bounds pixels**: both CPU and GPU paths write 0 for destination pixels that map outside the source bounds.
- **`IntegrationGpuCtx` is public**: the struct definition lives in `integration_gpu.h` (not just the `.cu`) so `pipeline.cu` can access `d_frames[]`, `d_xmap`, `d_ymap` directly. Opaque handle pattern was abandoned in favour of direct field access.
- **DLT produces backward H directly**: row setup uses `(ref_x, ref_y)` as H input and `(src_x, src_y)` as output → null vector = backward map (ref → src). No post-inversion needed.
- **Moffat constant memory cap**: max kernel radius is 15 (alpha ≤ 5 → R = ceil(3·alpha) ≤ 15, diameter 31, 961 floats × 4 = ~3.8 KB within 64 KB constant limit).
- **Phase 1 reads each file twice**: once for star detection, once for Phase 2 transform+integration. Acceptable trade-off versus caching all frames in RAM.
- **CPU vs GPU output agreement**: not bit-identical due to different Moffat conv precision (→ slightly different homographies), different Lanczos implementations (nppi vs hand-coded), and different integration paths. Empirical PSNR ≈ 44.6 dB; mean relative error ≈ 0.25% in the interior on 10 × 4656×3520 frames.

---

## Environment

| Resource | Path |
|---|---|
| CFITSIO headers | `/home/donut/.local/include/fitsio.h` |
| CFITSIO lib | `/home/donut/.local/lib/libcfitsio.so` |
| CFITSIO pkg-config | `/home/donut/.local/lib/pkgconfig/cfitsio.pc` |
| CUDA Toolkit | `/usr/local/cuda` |
| NPP geometry lib | `/usr/local/cuda/lib64/libnppig.so` |

---

## Python Tools (`python/`)

| Script | Purpose |
|---|---|
| `python/stacker.py` | Reference stacker: OpenCV Lanczos4 warp + kappa-sigma integration |
| `python/compute_transforms.py` | Independently compute homographies via astroalign and compare to CSV |

### astroalign convention note
`astroalign.find_transform(source, target)` returns an `AffineTransform` whose `.params` maps **source → target** (the forward direction: frame → ref). Since the CSV stores the **backward** map (ref → src), always call `np.linalg.inv(transform.params)` before comparing to CSV homographies.

### Verified findings (2026-03-17)
- **`data/transform_mat.csv` transforms are correct** — verified against independently computed astroalign transforms; max element diff < 0.04 px across all 9 non-reference frames.
- **Root cause of misalignment identified**: all three implementations (stacker.py, lanczos_cpu.c, lanczos_gpu.cu) were incorrectly inverting H before use. H is already the backward map — inverting it produced the forward map, which then misaligned every non-reference frame.
- **Fix**: use H directly without inversion in all transform paths. In OpenCV, pass `cv2.WARP_INVERSE_MAP` so H is used as-is.

### Python dependencies
```
astropy, numpy, opencv-python (cv2), astroalign
```
