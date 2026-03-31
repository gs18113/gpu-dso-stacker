# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Maintenance rule**: Whenever an important discovery, convention clarification, new implementation, API changes, or "aha moment" arises during a session, update this file immediately so future sessions start with the full picture.

## Project Overview

A high-performance Deep Sky Object (DSO) image stacker using C/CUDA for GPU-accelerated processing, with additive backend scaffolding for Apple Silicon Metal. Python reference stacker and transform-verification tools live in `python/`.

## Processing Pipeline

All stages are implemented for both GPU and CPU execution paths. The pipeline runs end-to-end from raw FITS frames to a stacked output.

| Stage | GPU path | CPU path |
|---|---|---|
| Debayering (VNG) | `debayer_gpu.cu` | `debayer_cpu.c` |
| Moffat conv + threshold | `star_detect_gpu.cu` | `star_detect_cpu.c` |
| CCL + CoM | — (CPU always) | `star_detect_cpu.c` |
| Triangle matching + DLT | `ransac_gpu.cu` (selectable via `--match-device`) | `ransac.c` |
| Lanczos warp | `lanczos_gpu.cu` | `lanczos_cpu.c` |
| Integration | `integration_gpu.cu` | `integration.c` |
| Pipeline orchestrator | `pipeline.cu` | `pipeline_cpu.c` |

Pass `--cpu` to run the complete CPU path.
Pass `--backend auto|cpu|cuda|metal` for explicit backend selection (default: `auto`).
Triangle matching device is controlled by `--match-device auto|cpu|gpu` (default `auto` follows stacking device).
The CSV input is always 2-column (`filepath, is_reference`). Star detection + alignment always run.

## Technology Stack

- **C11** — Core library (`fits_io.c`, `image_io.c`, `csv_parser.c`, `lanczos_cpu.c`, `integration.c`, `debayer_cpu.c`, `star_detect_cpu.c`, `ransac.c`, `pipeline_cpu.c`)
- **OpenMP** — CPU parallelism (debayer, Moffat convolution, Lanczos, integration all use `#pragma omp parallel for`)
- **CUDA 12 / NPP+** — GPU acceleration (`lanczos_gpu.cu`, `debayer_gpu.cu`, `star_detect_gpu.cu`, `integration_gpu.cu`, `pipeline.cu`)
- **CFITSIO 4.6.3** — FITS image I/O
- **libtiff 4.5.1** — TIFF output (FP32, FP16, INT16, INT8; none/zip/lzw/rle compression)
- **libpng 1.6.43** — PNG output (INT8, INT16)
- **C++17** — CLI entry point (`main.cpp`)

---

## File Structure

```
gpu-dso-stacker/
├── LICENSE                      ← GNU General Public License v3.0 (GPLv3)
├── README.ko.md                 ← Korean README translation (README.md remains default)
├── THIRD_PARTY_LICENSES         ← Third-party license texts and attribution notices
├── CMakeLists.txt               ← CMake build definition
├── main.cpp                     ← CLI entry point (getopt_long)
├── bench.sh                     ← GPU vs CPU benchmark script
├── include/
│   ├── dso_types.h              ← Shared types: Image, Homography, FrameInfo, DsoError,
│   │                               StarPos, StarList, BayerPattern, MoffatParams
│   ├── compat.h                 ← POSIX compatibility shims for MSVC (getopt, strtok_r, rand_r, mkdir, usleep, OMP collapse)
│   ├── getopt_port.h            ← Portable getopt_long for MSVC (BSD-licensed)
│   ├── fits_io.h                ← FITS load/save/free + fits_save_rgb + fits_get_bayer_pattern
│   ├── image_io.h               ← Format-agnostic save layer: ImageSaveOptions, image_save, image_save_rgb
│   ├── csv_parser.h             ← CSV frame-list parser (2-col only: filepath, is_reference)
│   ├── lanczos_cpu.h            ← CPU Lanczos-3 transform API
│   ├── lanczos_gpu.h            ← GPU Lanczos-3 transform API (h2h and d2d)
│   ├── integration.h            ← CPU mean / kappa-sigma integration API
│   ├── debayer_cpu.h            ← VNG debayer → luminance or RGB planes (CPU, OpenMP)
│   ├── debayer_gpu.h            ← VNG debayer → luminance or RGB planes (GPU, h2h and d2d)
│   ├── star_detect_gpu.h        ← Moffat convolution + sigma threshold (GPU, h2h and d2d)
│   ├── star_detect_cpu.h        ← Moffat conv + threshold + CCL + CoM (CPU)
│   ├── ransac.h                 ← DLT homography + RANSAC alignment
│   ├── integration_gpu.h        ← GPU mini-batch kappa-sigma context + API
│   ├── calibration.h            ← CalibFrames, CalibMethod, calib_load_or_generate, calib_apply_cpu
│   ├── calibration_gpu.h        ← CalibGpuCtx, calib_gpu_init/apply_d2d/cleanup
│   └── pipeline.h               ← pipeline_run dispatch + pipeline_run_{cpu,cuda,metal} + PipelineConfig
├── src/
│   ├── fits_io.c                ← CFITSIO-based I/O
│   ├── image_io.c               ← Format dispatch (FITS/TIFF/PNG) + libtiff + libpng writers
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
│   ├── calibration.c            ← Master frame generation (winsorized mean / median) + CPU apply
│   ├── calibration_gpu.cu       ← GPU calibration kernel (dark subtract + flat divide, D2D)
│   ├── pipeline.cu              ← CUDA orchestrator implementation (`pipeline_run_cuda`)
│   ├── pipeline_dispatch.c      ← backend dispatch (`pipeline_run`)
│   ├── pipeline_cpu.c           ← Pure-C CPU orchestrator (no CUDA)
│   ├── pipeline_metal.mm        ← Metal scaffold entry point (currently CPU fallback)
│   ├── pipeline_cuda_stub.c     ← CUDA-disabled build stub (`DSO_ERR_CUDA`)
│   └── pipeline_metal_stub.c    ← Metal-disabled build stub (`DSO_ERR_INVALID_ARG`)
│   └── getopt_port.c            ← Portable getopt_long (compiled only on MSVC)
├── tests/
│   ├── test_framework.h         ← Minimal test harness (ASSERT_*, RUN, SUITE)
│   ├── test_cpu.c               ← 29 tests: CSV, FITS, integration, Lanczos CPU
│   ├── test_gpu.cu              ← 5 tests: GPU Lanczos (all passing)
│   ├── test_star_detect.c       ← 21 tests: CCL + CoM + Moffat conv + threshold
│   ├── test_ransac.c            ← 13 tests: DLT + RANSAC
│   ├── test_debayer_cpu.c       ← 10 tests: VNG debayer CPU (all patterns + edge cases)
│   ├── test_integration_gpu.cu  ← 9 tests: GPU mini-batch kappa-sigma
│   ├── test_calibration.c       ← 26 tests: calib_apply_cpu (dark/flat/guard/dim), calib_load_or_generate (FITS master, frame-list stacking, winsorized mean, median, bias sub, flat normalization)
│   ├── test_audit.c             ← 4 tests: integration stability at N=1000, CCL large-frame, Lanczos numerical baseline, RANSAC non-determinism verification
│   ├── test_color.c             ← 33 tests: debayer_cpu_rgb (arg validation, BAYER_NONE passthrough, channel separation), fits_save_rgb (NAXIS=3, round-trip), color auto-detection
│   └── test_image_io.c          ← 21 tests: format detection, FITS passthrough, TIFF (FP32/FP16/INT16/INT8, all compressions, mono+RGB), PNG (8/16-bit mono+RGB), error cases, auto stretch
├── test/
│   └── star_detect_overlay.cpp  ← Standalone CLI helper that runs CPU/GPU star detection and writes PNG overlays with detected-star circles
├── .github/workflows/
│   ├── ci.yml                   ← CI: build + test on push/PR (Linux + Windows)
│   └── release.yml              ← Release: build + package + GitHub Release on tag v*
├── vcpkg.json                   ← vcpkg dependency manifest (Windows builds)
├── dso_stacker_gui.spec         ← PyInstaller spec for GUI packaging
└── python/
    ├── stacker.py               ← Reference stacker: OpenCV Lanczos4 + kappa-sigma
    ├── compute_transforms.py    ← Independently compute homographies via astroalign
    └── generate_test_frames.py  ← Synthetic FITS frame generator (Bayer + noise + guiding errors + calibration frames)
```

---

## Build Instructions

### Prerequisites

| Dependency | Version | Location |
|---|---|---|
| CFITSIO | 4.6.3 | `/home/donut/.local` (Linux); vcpkg (Windows) |
| CUDA Toolkit | 12.x | `/usr/local/cuda` (Linux); installer (Windows) |
| NPP+ (`libnppig`) | bundled with CUDA | `/usr/local/cuda/lib64` |
| OpenMP | any | system (GCC on Linux; MSVC on Windows) |
| libtiff | 4.5.1 | system pkg-config (Linux); vcpkg (Windows) |
| libpng | 1.6.43 | system pkg-config (Linux); vcpkg (Windows) |
| CMake | >= 3.18 | system |
| pkg-config | any | system (Linux only) |

### Build (Linux)

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

CPU-only build without CUDA toolkit:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDSO_ENABLE_CUDA=OFF
cmake --build build --parallel $(nproc)
```

### Build (Windows)

Requires Visual Studio 2022, CUDA Toolkit 12.x, and [vcpkg](https://vcpkg.io/).

```powershell
vcpkg install cfitsio tiff libpng --triplet x64-windows

cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
      -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"

cmake --build build --config Release --parallel
```

### CUDA Architectures

Default: `86;89` (RTX 30xx / 40xx). Override at configure time:

```bash
cmake -B build -DDSO_CUDA_ARCHITECTURES="75;80;86;89;90" ...
```

### Metal scaffold build (macOS / Apple Silicon)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DDSO_ENABLE_CUDA=OFF \
      -DDSO_ENABLE_METAL=ON
cmake --build build --parallel
```

`DSO_ENABLE_METAL` is guarded to Apple platforms and currently wires a safe
Phase-1 backend path (`pipeline_run_metal`) that falls back to `pipeline_run_cpu`.

### CFITSIO in CI

Set `CFITSIO_PREFIX` env var to the CFITSIO install prefix so CMake can find it via pkg-config:

```bash
CFITSIO_PREFIX=/opt/cfitsio cmake -B build ...
```

---

## CLI Usage

```
dso_stacker -f <frames.csv> [options]

I/O:
  -f, --file <path>              Input CSV file (required)
  -o, --output <path>            Output FITS file (default: output.fits)

Integration:
      --cpu                      Run ALL pipeline stages on CPU (OpenMP-accelerated)
      --backend <backend>        auto | cpu | cuda | metal (default: auto)
      --integration <method>     mean | kappa-sigma (default: kappa-sigma)
      --kappa <float>            Sigma clipping threshold (default: 3.0)
      --iterations <int>         Max clipping passes per pixel (default: 3)
      --batch-size <int>         GPU integration mini-batch size (default: 16)

Star detection (2-column CSV only):
      --star-sigma <float>       Detection threshold in σ units (default: 3.0)
      --moffat-alpha <float>     Moffat PSF alpha / FWHM (default: 2.5)
      --moffat-beta <float>      Moffat PSF beta / wing slope (default: 2.0)
      --top-stars <int>          Top-K stars for matching (default: 50)
      --min-stars <int>          Minimum stars for triangle matching (default: 6)

Triangle matching (2-column CSV only):
      --triangle-iters <int>     Max triangle-matching iterations (default: 1000)
      --triangle-thresh <float>  Inlier reprojection threshold px (default: 2.0)
      --ransac-iters <int>       Deprecated alias of --triangle-iters
      --ransac-thresh <float>    Deprecated alias of --triangle-thresh
      --match-radius <float>     Star matching search radius px (default: 30.0)
      --match-device <device>    auto | cpu | gpu (default: auto = stacking device)

Sensor:
      --bayer <pattern>          CFA override: none | rggb | bggr | grbg | gbrg

Calibration (applied before debayering; bias and darkflat are mutually exclusive):
      --dark <path>              Master dark FITS or text list of dark FITS paths
      --bias <path>              Master bias FITS or text list of bias FITS paths
      --flat <path>              Master flat FITS or text list of flat FITS paths
      --darkflat <path>          Master darkflat FITS or text list of darkflat FITS paths
      --save-master-frames <dir> Directory to save generated masters (default: ./master)
      --dark-method <method>     winsorized-mean | median (default: winsorized-mean)
      --bias-method <method>     winsorized-mean | median (default: winsorized-mean)
      --flat-method <method>     winsorized-mean | median (default: winsorized-mean)
      --darkflat-method <method> winsorized-mean | median (default: winsorized-mean)
      --wsor-clip <float>        Winsorized mean clipping fraction per side (default: 0.1)
                                 Valid range: [0.0, 0.49]
```

### Input CSV Format

```
filepath, is_reference
/data/frame1.fits, 1
/data/frame2.fits, 0
```

- First row is a header and is always skipped.
- Exactly **one** row must have `is_reference = 1`.
- Only 2-column format is accepted. Any other column count returns `DSO_ERR_CSV`.
- Homographies are computed at runtime via star detection + triangle matching.

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
DsoError fits_save_rgb(const char *filepath,
                        const Image *r, const Image *g, const Image *b);
void     image_free(Image *img);
```

- `fits_load`: opens FITS, reads any BITPIX as float32. Caller owns `out->data`.
- `fits_save`: writes float32 image as BITPIX=-32, NAXIS=2. Overwrites existing files.
- `fits_save_rgb`: writes three planes as NAXIS=3 (NAXIS1=W, NAXIS2=H, NAXIS3=3), planes ordered R=1/G=2/B=3. Compatible with DS9, SIRIL, PixInsight.
- `image_free`: frees `img->data` and nulls the pointer. Safe on zero-init Images.

### `image_io.h`

```c
typedef enum { FMT_FITS=0, FMT_TIFF, FMT_PNG, FMT_UNKNOWN } OutputFormat;
typedef enum { TIFF_COMPRESS_NONE=0, TIFF_COMPRESS_ZIP, TIFF_COMPRESS_LZW, TIFF_COMPRESS_RLE } TiffCompression;
typedef enum { OUT_BITS_FP32=0, OUT_BITS_FP16, OUT_BITS_INT16, OUT_BITS_INT8 } OutputBitDepth;

typedef struct {
    TiffCompression tiff_compress;   /* ignored for FITS/PNG */
    OutputBitDepth  bit_depth;       /* ignored for FITS (always FP32) */
    float           stretch_min;     /* NAN = auto; used for INT8/INT16 */
    float           stretch_max;     /* NAN = auto; used for INT8/INT16 */
} ImageSaveOptions;

OutputFormat image_detect_format(const char *filepath);
DsoError     image_save(const char *filepath, const Image *img, const ImageSaveOptions *opts);
DsoError     image_save_rgb(const char *filepath,
                              const Image *r, const Image *g, const Image *b,
                              const ImageSaveOptions *opts);
```

- `image_detect_format`: returns format based on file extension (case-insensitive). `.fits`/`.fit`/`.fts` → FMT_FITS; `.tif`/`.tiff` → FMT_TIFF; `.png` → FMT_PNG.
- `image_save` / `image_save_rgb`: dispatch to the appropriate writer. `opts == NULL` is equivalent to zero-init with NAN stretch (FP32, no compression, auto stretch).
- **Integer scaling**: `quantised = round(clamp((val - lo) / (hi - lo) * MAX_INT, 0, MAX_INT))` where lo/hi default to per-image min/max when `stretch_min`/`stretch_max` are NAN. For RGB, the same lo/hi are derived globally across all three planes to preserve colour ratios.
- **FP16 conversion**: portable IEEE 754 bit manipulation (no `__fp16`). Values outside ~[6e-8, 65504] are flushed to zero/infinity.
- **PNG 16-bit byte order**: written big-endian per PNG spec.
- **TIFF RGB layout**: interleaved PLANARCONFIG_CONTIG (RGBRGB…), compatible with Photoshop/Lightroom.
- **Valid combinations**: FP16 → TIFF only; FP32 → FITS or TIFF; INT8/INT16 → TIFF or PNG; FITS always writes FP32 regardless of bit_depth.

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
DsoError debayer_gpu_rgb_d2d(const float *d_src,
                               float *d_r, float *d_g, float *d_b,
                               int W, int H, BayerPattern pattern, cudaStream_t stream);
DsoError debayer_gpu(const Image *src, Image *dst, BayerPattern pattern, cudaStream_t stream);
```

VNG debayer: 16×16 tiles with 2-pixel apron in shared memory; 8 directional gradients.
- `debayer_gpu_d2d`: luminance output via ITU-R BT.709 (L = 0.2126·R + 0.7152·G + 0.0722·B). Used by Phase 1 (star detection) and mono Phase 2.
- `debayer_gpu_rgb_d2d`: writes reconstructed R, G, B values to three separate device buffers (each W×H). Used by color Phase 2. Separate kernel `vng_debayer_rgb_kernel`; same shared-memory tile strategy.
- `BAYER_NONE` fast path: `cudaMemcpyAsync` (d2d variants) or D2D copy.

### `debayer_cpu.h`

```c
DsoError debayer_cpu(const float *src, float *dst, int W, int H, BayerPattern pattern);
DsoError debayer_cpu_rgb(const float *src,
                          float *r, float *g, float *b,
                          int W, int H, BayerPattern pattern);
```

Same VNG algorithm as the GPU kernels. `BAYER_NONE` = `memcpy` (all three output planes for the RGB variant). Boundary reads clamp to 0. Both parallelized with `#pragma omp parallel for collapse(2) schedule(static)`.
- `debayer_cpu`: luminance output (mono mode and Phase 1).
- `debayer_cpu_rgb`: writes R, G, B into three pre-allocated arrays (color Phase 2).

### `star_detect_gpu.h`

```c
DsoError star_detect_gpu_d2d(const float *d_src, float *d_conv, uint8_t *d_mask,
                              int W, int H, const MoffatParams *params,
                              float sigma_k, cudaStream_t stream);
```

Moffat kernel `K(i,j) = [1 + (i²+j²)/alpha²]^(-beta)` stored in 64 KB constant memory (max radius 15). Convolution via 16×16 shared-memory tiles. Fully on-device reduction pipeline: `reduce_sum_kernel` → `reduce_final_kernel` → `reduce_div_n_kernel` (mean) → `reduce_sumsq_kernel` → `reduce_final_kernel` (variance) → `threshold_auto_kernel` reads device-resident mean/σ and writes `mask[i]`. One `cudaStreamSynchronize` at end for `cudaFree`; no host-side partial-sum copies.

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
`ccl_com`: two-pass 8-connectivity union-find; not parallelized (raster scan has data dependencies). After Pass 2, a label re-mapping pass compacts sparse root labels to contiguous `[1, n_unique]`, so `CompStats` is allocated as `calloc(n_unique+1)` (O(n_stars)) rather than `calloc(npix+1)` (O(pixels)).

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

DLT via Jacobi eigendecomposition of A^T·A (9×9); point normalization for numerical stability. `ransac_compute_homography` uses a two-phase approach: (1) triangle voting to extract correspondences, then (2) RANSAC over those correspondences (sample 4, DLT + degenerate check, inlier counting, adaptive iteration termination, final refinement on best inliers). If the primary RANSAC fails, falls back to `fallback_ransac_compute` which uses nearest-neighbour star matching with Lowe ratio test (d1/d2 < 0.8). Produces **backward homography (ref → src)** directly — no inversion needed. Uses `rand_r(&seed)` with per-call seeds derived from `time(NULL) ^ clock() ^ counter++` (static counters) for thread-safe, per-call independent random sequences.

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

### `calibration.h`

```c
typedef enum { CALIB_WINSORIZED_MEAN = 0, CALIB_MEDIAN } CalibMethod;

typedef struct CalibFrames {
    Image dark;    /* master dark (bias-subtracted if bias was provided) */
    Image flat;    /* master flat (bias/darkflat-subtracted, normalized to mean ≈ 1.0) */
    int   has_dark;
    int   has_flat;
} CalibFrames;

DsoError calib_load_or_generate(
    const char *dark_path,    CalibMethod dark_method,
    const char *bias_path,    CalibMethod bias_method,
    const char *flat_path,    CalibMethod flat_method,
    const char *darkflat_path, CalibMethod darkflat_method,
    const char *save_dir,
    float        wsor_clip,   /* clipping fraction per side [0.0, 0.49], default 0.1 */
    CalibFrames *calib_out);

DsoError calib_apply_cpu(Image *img, const CalibFrames *calib);
void     calib_free(CalibFrames *calib);
```

- Each path argument accepts either a FITS file (loaded directly as master) or a text file (one FITS path per line; frames are stacked).
- Generated masters are saved to `<save_dir>/master_{dark,flat,bias,darkflat}.fits`.
- `calib_apply_cpu`: validates dimensions, subtracts dark, divides by flat (in-place). Returns `DSO_ERR_INVALID_ARG` on dimension mismatch.

### `calibration_gpu.h`

```c
typedef struct {
    float *d_dark;   /* device dark master (NULL = not used) */
    float *d_flat;   /* device flat master (NULL = not used) */
    int    W, H;
} CalibGpuCtx;

DsoError calib_gpu_init(const CalibFrames *calib, CalibGpuCtx **ctx_out);
void     calib_gpu_cleanup(CalibGpuCtx *ctx);
DsoError calib_gpu_apply_d2d(float *d_frame, int W, int H,
                               const CalibGpuCtx *ctx, cudaStream_t stream);
```

Single CUDA kernel (256 threads/block): subtract dark, divide by flat, dead-pixel guard (flat < 1e-6 → 0). Upload happens once in `pipeline_run` before Phase 1.

### `pipeline.h`

```c
DsoError pipeline_run(FrameInfo *frames, int n_frames,
                       int ref_idx, const PipelineConfig *config);
DsoError pipeline_run_cpu(FrameInfo *frames, int n_frames,
                           int ref_idx, const PipelineConfig *config);
DsoError pipeline_run_cuda(FrameInfo *frames, int n_frames,
                            int ref_idx, const PipelineConfig *config);
DsoError pipeline_run_metal(FrameInfo *frames, int n_frames,
                             int ref_idx, const PipelineConfig *config);
```

`pipeline_run` is now a backend dispatcher in `pipeline_dispatch.c` using `PipelineConfig.backend` (`AUTO/CPU/CUDA/METAL`) with legacy `use_gpu_lanczos` preserved for AUTO behavior.

`pipeline_run_cuda` is the CUDA orchestrator. Processes frames in order `[ref_idx, non_ref_0, ...]` via `phase_detect_warp_integrate`: double-buffered `pinned[2]`/`d_raw[2]` + `stream_copy`/`stream_compute`. Per frame: `stream_compute` waits for `e_h2d[slot]` → calib → debayer_lum → star_detect → `cudaStreamSynchronize` → D2H → CCL+CoM → RANSAC → Lanczos warp on `stream_compute`; then CPU pre-loads next frame + async H2D on `stream_copy` (overlaps with the warp). At batch boundaries, H2D of the next frame also overlaps mini-batch integration. The `cudaStreamSynchronize` for D2H also implicitly frees `d_raw[next_slot]` (clears the prior warp), so no `e_gpu` event is needed. For color output allocates ctx_r/g/b + d_ch_g/d_ch_b; `phase_warp` calls `debayer_gpu_rgb_d2d` + 3× Lanczos per frame. Finalizes via `image_save_rgb`/`image_save`. **Non-reference frames that fail alignment (too few stars or RANSAC mismatch) are skipped, not fatal; successful-frame count is passed to `integration_gpu_finalize`.**

`pipeline_run_cpu`: pure-C orchestrator in `pipeline_cpu.c` (no CUDA headers). Processes frames in order `[ref_idx, non_ref_0, ...]` (single pass, each file loaded once). Per frame: debayer_cpu (lum) → star_detect_cpu_detect → ccl_com → ransac → batch warp+integrate. Mini-batched (PIPELINE_CPU_BATCH_SIZE=32). For color: debayer_cpu_rgb → 3× lanczos_transform_cpu; finalizes via `image_save_rgb`. **Non-reference frames that fail alignment (too few stars or RANSAC mismatch) are skipped and processing continues.**

`pipeline_run_metal`: Phase-1 scaffold implemented in `pipeline_metal.mm`; currently logs backend selection and falls back to `pipeline_run_cpu` to preserve numerical semantics while Metal kernels are ported incrementally.

---

## Key Implementation Notes

- **Homography convention**: H is the *backward* map (ref → src). Transform functions use it directly for pixel sampling — **do not invert**. The `invert_homography` helpers remain in the source but are dead code.
- **GPU Lanczos strategy**: `nppiWarpPerspective` only supports NN/LINEAR/CUBIC. We instead pre-compute backward-homography coordinate maps (H used directly, no inversion) in a CUDA kernel, then feed them to `nppiRemap_32f_C1R_Ctx` with `NPPI_INTER_LANCZOS` (= 16).
- **Backend dispatch is centralized**: `pipeline_run()` now lives in `pipeline_dispatch.c` and dispatches via `PipelineConfig.backend` (`AUTO/CPU/CUDA/METAL`). AUTO preserves legacy behavior: `use_gpu_lanczos==0` forces CPU; otherwise it prefers compiled GPU backend(s).
- **`MoffatParams` lives in `dso_types.h`**: moved from `star_detect_gpu.h` (which includes `cuda_runtime.h`) so that `pipeline_cpu.c` and `star_detect_cpu.c` can use it without any CUDA dependency.
- **OpenMP parallelism strategy**:
  - `debayer_cpu`: `OMP_PARALLEL_FOR_COLLAPSE2` (collapse(2) on GCC/Clang, outer-only on MSVC) — each pixel is independent.
  - `star_detect_cpu_moffat_convolve`: `OMP_PARALLEL_FOR_COLLAPSE2` — inner kernel loops serial per pixel.
  - `star_detect_cpu_threshold`: three passes — `reduction(+:sum)`, `reduction(+:sq)`, then parallel mask write.
  - `lanczos_cpu`: `schedule(static)` on outer `dy` row loop.
  - `integrate_mean`: pixel-outer loop (restructured from frame-outer) — `schedule(static)`.
  - `integrate_kappa_sigma`: `schedule(dynamic, 64)` — per-thread heap slabs `all_vals` / `all_actv` (size `max_threads × n`, allocated once before the parallel region, indexed by `omp_get_thread_num() * n`). No VLAs.
  - `star_detect_cpu_ccl_com` (CCL pass): **not parallelized** — union-find raster scan has cross-pixel data dependencies.
- **C goto rule (C11 and C++17)**: variables that appear between a `goto` source and its label must be declared before the first `goto`. In `pipeline_cpu.c`, all variables are declared at the top of each function; error paths use explicit `image_free()` before `goto cleanup` rather than a PIPE_CHECK macro, to avoid jumping over initializers.
- **Row steps for NPP**: all row steps are `width * sizeof(float)` (row-major float32).
- **FITS pixel index**: `ffgpxv` / `ffppx` take `long *firstpix` (1-based per axis); pass `{1, 1}` for 2D, `{1, 1, plane}` for 3D (NAXIS=3 RGB output).
- **Out-of-bounds pixels**: both CPU and GPU paths write 0 for destination pixels that map outside the source bounds.
- **`IntegrationGpuCtx` is public**: the struct definition lives in `integration_gpu.h` (not just the `.cu`) so `pipeline.cu` can access `d_frames[]`, `d_xmap`, `d_ymap` directly. Opaque handle pattern was abandoned in favour of direct field access.
- **DLT produces backward H directly**: row setup uses `(ref_x, ref_y)` as H input and `(src_x, src_y)` as output → null vector = backward map (ref → src). No post-inversion needed.
- **Moffat constant memory cap**: max kernel radius is 15 (alpha ≤ 5 → R = ceil(3·alpha) ≤ 15, diameter 31, 961 floats × 4 = ~3.8 KB within 64 KB constant limit).
- **Single-pass pipeline**: each frame is loaded from disk exactly once. Star detection, RANSAC, and warp all run in the same pass before the next frame is loaded. No separate Phase 1 / Phase 2; no 11-column pre-computed transform path. The only CSV format is 2-column (`filepath, is_reference`).
- **Triangle-matching mismatch policy**: non-reference frames that fail alignment are skipped (not fatal). Both CPU and GPU pipelines print a final summary: `successful frames: X/Y (skipped: Z)`. Reference-frame alignment failure still aborts.
- **CPU vs GPU output agreement**: not bit-identical due to different Moffat conv precision (→ slightly different homographies), different Lanczos implementations (nppi vs hand-coded), and different integration paths. Empirical PSNR ≈ 44.6 dB; mean relative error ≈ 0.25% in the interior on 10 × 4656×3520 frames.
- **`lanczos_transform_cpu` identity fast path**: if H is identity and src/dst dimensions match, falls back to `memcpy` before entering the warp loop.
- **`lanczos_transform_cpu` weight precomputation**: `wx_arr[6]` and `wy_arr[6]` are computed once before the 6×6 tap loop, reducing `lanczos_weight` calls from 42 to 12 per destination pixel.
- **`lanczos_transform_cpu` weight_sum guard**: uses `fabsf(weight_sum) < 1e-6f`, not `== 0.f`. When the Lanczos kernel center tap lands exactly on an integer OOB coordinate, `sinf(k·π)` floating-point error leaves `weight_sum` near but not exactly zero; an exact equality test would divide by ≈−1e-9 and amplify noise to ≈1560. The threshold correctly returns 0 for the OOB case.
- **`lanczos_transform_gpu` singular-H guard**: checks `det(H) < 1e-12` before any CUDA allocation and returns `DSO_ERR_INVALID_ARG`.
- **GPU kappa-sigma variance uses `double`**: `kappa_sigma_batch_kernel` accumulates squared deviations into `double sq` (not `float`). For pixel values up to 65535 and batch size 64, max sq ≈ 2.75×10¹¹ exceeds float precision; using double matches the CPU `integrate_kappa_sigma` implementation.
- **Synthetic RANSAC stress tests**: `test/star_coords_generator.{h,c}` generates deterministic synthetic star lists (shared inlier transform + independent outliers); `tests/test_ransac.c` uses it for seed sweeps and high-outlier scenarios, and `test_ransac` target links the generator directly in `CMakeLists.txt`.
- **Calibration formula**: `light_cal = (light - dark_master) / flat_master`. Applied before debayering. With bias: `dark_master = stack(dark_raw - bias)`, `flat_master = stack(normalize(flat_raw - bias))`. With darkflat: `dark_master = stack(dark_raw)`, `flat_master = stack(normalize(flat_raw - darkflat))`. Bias and darkflat are mutually exclusive.
- **Winsorized mean (γ=0.1)**: Sort N pixel values per pixel; replace bottom `g = floor(0.1·N)` values with `vals[g]` and top g with `vals[N-1-g]`; compute mean using `double` accumulator to prevent overflow for N·(65535)² accumulations. Insertion sort used for per-pixel sort (N typically < 100).
- **Flat normalization**: Each flat frame is divided by its own (double-precision) mean before stacking. The stacked master flat has mean ≈ 1.0 and is used directly as a divisor. Flat inputs must be **raw ADU** — `calibration.c` subtracts bias first, then normalises. Pre-normalised flat frames (mean≈1.0) produce `flat_raw − bias ≈ −249`, triggering "near-zero mean, skipping normalisation" and an inverted/invalid master flat.
- **Dead-pixel guard**: Flat pixels below `1e-6f` → output 0 (not divide). Both CPU and GPU paths apply this guard.
- **Calibration dimension validation**: `calib_apply_cpu` and `calib_gpu_apply_d2d` return `DSO_ERR_INVALID_ARG` if the frame dimensions do not match the master frame dimensions.
- **FITS vs. text-list detection**: `is_fits_file()` probes with CFITSIO `fits_open_file`; if it succeeds the path is a pre-computed master, otherwise it is treated as a newline-separated text frame list.
- **CalibFrames forward declaration**: `pipeline.h` uses `typedef struct CalibFrames CalibFrames` (forward decl with tag) so that it can hold a pointer without pulling in `calibration.h` and its CFITSIO dependency into all translation units. `calibration.h` defines `typedef struct CalibFrames { ... } CalibFrames` with the same tag — both declarations must agree on the tag name.
- **GPU calibration timing**: `calib_gpu_init` uploads masters once at the start of `pipeline_run`. `calib_gpu_apply_d2d` is inserted on `stream_compute` after `cudaStreamWaitEvent(e_h2d[slot])` and before `debayer_gpu_d2d`, in both Phase 1 and Phase 2.
- **VNG debayer gradients are one-sided**: each of the 8 directional gradients samples only the pixel in its named direction — `g[i] = |P(neighbor_i) − P(0,0)|`. The previous two-sided formula `|A−center| + |center−B|` made opposite pairs equal (g[4]≡g[0] etc.), reducing effective direction selection from 8-way to 4-way. Fixed in both `debayer_gpu.cu` and `debayer_cpu.c`.
- **Color output** (`PipelineConfig::color_output`): auto-set to 1 in `main.cpp` when the Bayer pattern is not `BAYER_NONE` (either from `--bayer` flag or auto-detected from the reference frame FITS BAYERPAT keyword). Phase 1 (star detection, RANSAC) always uses luminance; only Phase 2 (warp + integrate) and output change. `Image` struct remains single-channel — R, G, B are three independent `Image` instances throughout.
- **Color GPU stream overlap**: for color mode, `phase2_transform_integrate` allocates `d_debayed` (R), `d_ch_g`, `d_ch_b` plus three `IntegrationGpuCtx`. Per frame on `stream_compute`: calibrate → `debayer_gpu_rgb_d2d` → memset+Lanczos(R) → memset+Lanczos(G) → memset+Lanczos(B) → `cudaEventRecord(e_gpu[slot])`. The event fires after the third Lanczos, so `stream_copy` H2D of the next frame overlaps all three warps — the overlap is at least as efficient as mono mode since the compute phase is ~3× longer.
- **Color CPU batch**: three xformed_r/g/b arrays allocated for the batch; integration called once per channel per batch; ptrs arrays for G and B built as temporaries inside the batch loop (freed after integration).
- **Output format dispatch** (`PipelineConfig::save_opts`): `image_save` / `image_save_rgb` (in `image_io.c`) detect the format from the output file extension at call time. `pipeline.cu` and `pipeline_cpu.c` no longer call `fits_save` directly. `opts == NULL` or a zero-init struct with NAN stretch defaults to FP32, no compression, auto stretch.
- **Integer stretch semantics**: for INT8/INT16 output, `stretch_min`/`stretch_max` are NAN by default (auto min/max of the image). For RGB, the stretch bounds are derived globally across all three planes so that colour ratios are preserved. Setting both to the same value (hi == lo) produces a black image.
- **FP16 TIFF precision**: values outside the IEEE 754 half-precision representable range (~[6e-8, 65504]) are flushed to ±0 / ±infinity. For astronomical data with wide dynamic range, prefer FP32 or INT16.
- **TIFF RGB interleaving**: `PLANARCONFIG_CONTIG` (RGBRGB…) — the three `Image` planes are interleaved into a per-row buffer at write time, no full-image allocation.

---

## Cross-Platform Compatibility (`compat.h`)

`include/compat.h` provides a POSIX-to-MSVC shim layer. On GCC/Clang it is effectively empty. Include it in any source file that uses POSIX APIs not available on MSVC.

| POSIX API | MSVC Replacement | Files |
|---|---|---|
| `getopt_long` | `getopt_port.h/.c` (bundled BSD impl) | `main.cpp` |
| `strtok_r` | `strtok_s` (same signature) | `csv_parser.c` |
| `rand_r` | inline LCG (`compat_rand_r`) | `ransac.c` |
| `mkdir(dir, mode)` | `_mkdir(dir)` via `<direct.h>` | `calibration.c` |
| `usleep(us)` | `Sleep(us/1000)` via `<windows.h>` | `test_audit.c` |
| `collapse(2)` pragma | `OMP_PARALLEL_FOR_COLLAPSE2` macro (drops collapse on MSVC OpenMP 2.0) | `debayer_cpu.c`, `star_detect_cpu.c` |

**MSVC OpenMP**: MSVC only supports OpenMP 2.0. The `OMP_PARALLEL_FOR_COLLAPSE2` macro falls back to `#pragma omp parallel for schedule(static)` (outer loop only). Performance impact is minimal since the outer loop iterates over image rows (thousands of iterations).

**`getopt_port.c`**: compiled only on MSVC via conditional `list(APPEND LIB_SOURCES ...)` in `CMakeLists.txt`. BSD-2-Clause licensed.

**Optional CUDA/Metal build toggles**: `CMakeLists.txt` now supports `DSO_ENABLE_CUDA` (default ON) and `DSO_ENABLE_METAL` (default OFF, Apple-only). CUDA sources/tests are conditionally added; CPU-only builds compile without NVCC by setting `-DDSO_ENABLE_CUDA=OFF`.

**`test_framework.h`**: `SUMMARY()` macro uses `static inline` function instead of GCC statement expression `({...})` for MSVC compatibility.

---

## CI/CD

### Workflows

| File | Trigger | Jobs |
|---|---|---|
| `.github/workflows/ci.yml` | Push / PR to `main` | `linux`, `windows`, `gui-linux`, `gui-windows` |
| `.github/workflows/release.yml` | Tag push `v*` or manual dispatch (`workflow_dispatch`) | Same 4 build jobs + `create-release` |

### Linux CI

- Container: `nvidia/cuda:12.6.3-devel-ubuntu22.04` (includes nvcc, NPP, CUDA headers)
- CFITSIO 4.6.3 built from source and cached at `/opt/cfitsio`
- libtiff, libpng via `apt-get`
- CUDA architectures: `61;70;75;80;86;89` (Pascal GTX 10xx through Ada Lovelace RTX 40xx)
- Builds **two** CLI variants on Linux: CUDA-enabled (`dso-stacker-linux-x86_64-gpu`) and CPU-only (`dso-stacker-linux-x86_64-cpu`)
- GPU tests auto-skip via exit code 77 (no GPU on CI runners)

### Windows CI

- Runner: `windows-2022`
- CUDA via `Jimver/cuda-toolkit@v0.2.19` with sub-packages: `nvcc, cudart, npp, npp_dev, visual_studio_integration`
- C libraries via vcpkg (manifest: `vcpkg.json`): cfitsio, tiff, libpng
- CMake generator: `Visual Studio 17 2022` with vcpkg toolchain file
- Builds **two** CLI variants on Windows: CUDA-enabled (`dso-stacker-windows-x86_64-gpu`) and CPU-only (`dso-stacker-windows-x86_64-cpu`)
- GPU tests auto-skip

### GUI Packaging

- PyInstaller spec: `dso_stacker_gui.spec`
- Bundles CLI binary in `bin/` subdirectory alongside the GUI executable
- `src/GUI/utils.py:_binary_path()` resolves binary via: (1) PyInstaller frozen `<exe_dir>/bin/`, (2) dev layout `<repo>/build/`
- Hidden imports: `shiboken6`, `PySide6.QtCore`, `PySide6.QtGui`, `PySide6.QtWidgets`

### Release Artifacts

| Archive | Contents |
|---|---|
| `dso-stacker-cli-linux-x86_64-cpu.tar.gz` | CPU-only CLI binary |
| `dso-stacker-cli-linux-x86_64-gpu.tar.gz` | GPU-selectable CLI binary (CUDA runtime required) |
| `dso-stacker-cli-macos-arm64-cpu.tar.gz` | CPU-only CLI binary |
| `dso-stacker-cli-macos-arm64-metal.tar.gz` | Metal-enabled CLI binary |
| `dso-stacker-gui-linux-x86_64-cpu.tar.gz` | GUI + CPU-only CLI bundle |
| `dso-stacker-gui-linux-x86_64-gpu.tar.gz` | GUI + GPU-selectable CLI bundle |
| `dso-stacker-gui-macos-arm64-cpu.tar.gz` | GUI + CPU-only CLI bundle |
| `dso-stacker-gui-macos-arm64-metal.tar.gz` | GUI + Metal-enabled CLI bundle |
| `dso-stacker-cli-windows-x86_64-cpu.zip` | CPU-only CLI .exe + DLLs |
| `dso-stacker-cli-windows-x86_64-gpu.zip` | GPU-selectable CLI .exe + DLLs |
| `dso-stacker-gui-windows-x86_64-cpu.zip` | GUI + CPU-only CLI .exe + DLLs |
| `dso-stacker-gui-windows-x86_64-gpu.zip` | GUI + GPU-selectable CLI .exe + DLLs |

Before creating release archives, `release.yml` `create-release` now re-applies execute permissions for macOS binaries (`dso_stacker`, `DSOStacker`, and bundled `bin/dso_stacker`) so CLI and GUI artifacts remain directly runnable after download.

---

## Licensing

- **Project license**: GNU General Public License v3.0 (GPLv3) — see `LICENSE`.
- **NVIDIA CUDA Toolkit + NPP**: Both covered under the single [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/). NPP does not require any additional license beyond the CUDA EULA. Dynamically linked — users must have the CUDA runtime installed; they accept the NVIDIA EULA when installing the CUDA Toolkit.
- **Third-party components**: CFITSIO (U.S. Government permissive), libtiff (BSD-style), libpng (libpng-2.0), PySide6 (LGPL v3), PyYAML (MIT), getopt\_port (BSD-2-Clause). Full texts in `THIRD_PARTY_LICENSES`.

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
| `python/generate_test_frames.py` | Synthetic FITS frame generator for pipeline testing |

### `generate_test_frames.py`

Generates synthetic FITS test frames that exercise all pipeline stages: Bayer mosaics (`--bayer rggb|bggr|grbg|gbrg`), Moffat-PSF stars (alpha/beta match pipeline defaults), per-frame homographic guiding offsets, and optional calibration sets (bias/dark/flat, `--gen-calibration`). Star colours are modelled via a B-V colour index distribution (Beta(2.5, 1.5) biased toward G/K-type), with per-channel rendering in Bayer mode and ITU-R BT.709 luminance weighting in mono mode. Always writes a 2-column CSV (`filepath, is_reference`).

Key flags: `-n` (frames), `-s` (stars), `-o` (output dir), `--bayer`, `--gen-calibration`, `--num-calib-frames`, `--no-star-colors`.

**Flat frame generation note**: `make_flat_frame` returns **raw ADU values** (not normalised). `calibration.c` subtracts bias from each flat *before* normalising — passing pre-normalised flats (mean≈1.0) gives `flat_raw − bias ≈ −249`, triggering the "near-zero mean, skipping normalisation" warning and producing an inverted master flat.

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

---

## GUI Frontend (`src/GUI/`)

A PySide6 desktop frontend for the `dso_stacker` CLI.

### Launch

```bash
pip install PySide6 pyyaml astropy   # one-time
python src/GUI/main.py
```

### File structure

```
src/GUI/
├── main.py                      # Entry point: QApplication + dark palette + MainWindow
├── main_window.py               # MainWindow: QTabWidget, menu bar, Run/Abort, log panel
├── project.py                   # ProjectState dataclass + YAML save/load (pyyaml)
├── runner.py                    # SubprocessRunner(QThread): Popen + line-by-line emit
├── fits_meta.py                 # FitsMetaWorker(QRunnable): reads NAXIS1/2, BAYERPAT
├── utils.py                     # build_command(), write_csv(), write_calib_list(),
│                                #   format_size(), detect_output_format()
└── widgets/
    ├── __init__.py
    ├── frame_table.py           # FrameTableWidget: DnD QTableWidget base for all tabs
    ├── light_tab.py             # LightTab: frame list + QRadioButton reference column
    ├── calib_tab.py             # CalibTab: frame list + stacking-method combo
    └── stacking_options_tab.py  # All CLI params; single _update_visibility() controls
                                 #   conditional widget show/hide per output format
```

### Key design decisions

- **2-column CSV only**: the GUI always writes a 2-col CSV (star-detection mode). Pre-computed homography workflows remain a CLI concern.
- **Temp directory per run**: `tempfile.TemporaryDirectory` created on Run, deleted after subprocess join. Holds `frames.csv` and per-calibration `*_list.txt` files.
- **Async FITS metadata**: `FitsMetaWorker(QRunnable)` submitted to `QThreadPool.globalInstance()` so reading FITS headers never blocks the UI thread. Uses a minimal pure-Python FITS header parser (`_read_fits_keywords` in `fits_meta.py`) — reads 2880-byte blocks, 80-byte cards, stops at END — no external library needed.
- **Bias / Darkflat mutual exclusion**: `QTabWidget.setTabEnabled(False)` grays out and disables the opposing tab; `setTabToolTip` explains why. Checked again at run time.
- **Conditional visibility**: single `_update_visibility()` slot in `StackingOptionsTab` connected to all relevant widget change signals (integration combo, CPU checkbox, output path, bit depth).
- **Triangle-matching device in GUI**: `StackingOptionsTab` exposes `match_device` (`auto|cpu|gpu`) and `utils.build_command()` emits `--match-device` when GPU mode is active and the value is not `auto`.
- **Windows CUDA runtime PATH fallback in GUI**: `SubprocessRunner` now builds a subprocess env via `_build_subprocess_env()`; on Windows, if `CUDA_PATH` exists and `%CUDA_PATH%\bin` is missing from `PATH`, it prepends that directory before launching `dso_stacker`. This covers newer CUDA installers that set `CUDA_PATH` but don't update `PATH`.
- **Bit depth combo item disabling**: uses `QStandardItemModel` — items are disabled (not removed) based on output format. Snaps to nearest valid selection automatically.
- **Binary path resolution**: `utils._binary_path()` resolves `<repo>/build/dso_stacker` relative to `utils.py`. Raises `FileNotFoundError` with a helpful build instruction if absent.
- **Dark theme**: Fusion style + custom `QPalette` applied in `main.py`. No external theme library required.

### YAML project schema

```yaml
version: 1
light_frames:
  reference_index: 0
  files: [/abs/path/frame.fit, ...]
dark_frames:    { method: winsorized-mean, files: [] }
flat_frames:    { method: winsorized-mean, files: [] }
bias_frames:    { method: winsorized-mean, files: [] }
darkflat_frames: { method: winsorized-mean, files: [] }
options:
  output_path: output.fits     use_cpu: false
  integration: kappa-sigma     kappa: 3.0     iterations: 3     batch_size: 16
  star_sigma: 3.0     moffat_alpha: 2.5     moffat_beta: 2.0
  top_stars: 50     min_stars: 6
  triangle_iters: 1000     triangle_thresh: 2.0     match_radius: 30.0     match_device: auto
  bayer: auto     bit_depth: f32     tiff_compression: none
  stretch_min: null     stretch_max: null
  save_master_dir: ./master     wsor_clip: 0.1
  dark_method: winsorized-mean     ...
```
