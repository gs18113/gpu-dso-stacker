# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Maintenance rule**: Whenever an important discovery, convention clarification, or "aha moment" arises during a session, update this file immediately so future sessions start with the full picture.

## Project Overview

A high-performance Deep Sky Object (DSO) image stacker using C/CUDA for GPU-accelerated processing. Python reference stacker and transform-verification tools live in `python/`.

## Processing Pipeline

1. **Preprocessing** — Initial image preparation and calibration (upstream, not in this repo)
2. **Star Detection & PSF Fitting** — Identify stars and fit point spread functions (upstream)
3. **Transform Computation** — Calculate alignment using RANSAC (upstream)
4. **Image Transformation** — Lanczos-3 interpolation to align frames to reference (`lanczos_cpu.c` / `lanczos_gpu.cu`)
5. **Integration** — Combine images using kappa-sigma clipping (`integration.c`)

## Technology Stack

- **C11** — Core library (`fits_io.c`, `csv_parser.c`, `lanczos_cpu.c`, `integration.c`)
- **CUDA 12 / NPP+** — GPU acceleration (`lanczos_gpu.cu`)
- **CFITSIO 4.6.3** — FITS image I/O
- **C++17** — CLI entry point (`main.cpp`)

---

## File Structure

```
gpu-dso-stacker/
├── CMakeLists.txt          ← CMake build definition
├── main.cpp                ← CLI entry point (getopt_long)
├── include/
│   ├── dso_types.h         ← Shared types: Image, Homography, FrameInfo, DsoError
│   ├── fits_io.h           ← FITS load/save/free API
│   ├── csv_parser.h        ← CSV frame-list parser API
│   ├── lanczos_cpu.h       ← CPU Lanczos-3 transform API
│   ├── lanczos_gpu.h       ← GPU Lanczos-3 transform API
│   └── integration.h       ← Mean / kappa-sigma integration API
├── src/
│   ├── fits_io.c           ← CFITSIO-based I/O implementation
│   ├── csv_parser.c        ← CSV parser implementation
│   ├── lanczos_cpu.c       ← CPU backward-mapping Lanczos-3
│   ├── lanczos_gpu.cu      ← CUDA coord-map kernel + nppiRemap
│   └── integration.c       ← Mean and kappa-sigma clipping
├── tests/
│   ├── test_framework.h    ← Minimal test harness (ASSERT_*, RUN, SUITE)
│   ├── test_cpu.c          ← Unit tests for CPU-side library
│   └── test_gpu.cu         ← Unit tests for GPU transform
└── python/
    ├── stacker.py          ← Reference stacker: OpenCV Lanczos4 + kappa-sigma
    └── compute_transforms.py ← Independent transform computation via astroalign
```

---

## Build Instructions

### Prerequisites

| Dependency | Version | Location |
|---|---|---|
| CFITSIO | 4.6.3 | `/home/donut/.local` |
| CUDA Toolkit | 12.x | `/usr/local/cuda` |
| NPP+ (`libnppig`) | bundled with CUDA | `/usr/local/cuda/lib64` |
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

Options:
  -f, --file <path>              Input CSV file (required)
  -o, --output <path>            Output FITS file (default: output.fits)
      --cpu                      Use CPU Lanczos instead of GPU
      --integration <method>     mean | kappa-sigma (default: kappa-sigma)
      --kappa <float>            Sigma clipping threshold (default: 3.0)
      --iterations <int>         Max clipping passes per pixel (default: 3)
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
typedef enum { DSO_OK=0, DSO_ERR_IO=-1, DSO_ERR_ALLOC=-2, DSO_ERR_FITS=-3,
               DSO_ERR_CUDA=-4, DSO_ERR_NPP=-5, DSO_ERR_CSV=-6,
               DSO_ERR_INVALID_ARG=-7 } DsoError;
```

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

---

## Key Implementation Notes

- **Homography convention**: H is the *backward* map (ref → src). Transform functions use it directly for pixel sampling — **do not invert**. The `invert_homography` helpers remain in the source but are dead code.
- **GPU Lanczos strategy**: `nppiWarpPerspective` only supports NN/LINEAR/CUBIC. We instead pre-compute backward-homography coordinate maps (H used directly, no inversion) in a CUDA kernel, then feed them to `nppiRemap_32f_C1R_Ctx` with `NPPI_INTER_LANCZOS` (= 16).
- **Row steps for NPP**: all row steps are `width * sizeof(float)` (row-major float32).
- **FITS pixel index**: `ffgpxv` / `ffppx` take `long *firstpix` (1-based per axis); pass `{1, 1}`.
- **Out-of-bounds pixels**: both CPU and GPU paths write 0 for destination pixels that map outside the source bounds.
- **C++ goto rule**: in `lanczos_gpu.cu`, `dim3` variables are declared before any `CHECK_CUDA` macros to avoid "jump over initialisation" errors.

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
