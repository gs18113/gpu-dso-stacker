# gpu-dso-stacker

> A high-performance DSO (Deep Sky Object) stacker using CUDA for GPU-accelerated processing

---

## Technology Stack

- **C11** — Core library (FITS I/O, CSV parser, Lanczos CPU, integration, debayer CPU, star detection, RANSAC, CPU pipeline)
- **OpenMP** — CPU parallelism for debayer, Moffat convolution, Lanczos warp, and integration
- **CUDA 12** — GPU acceleration (VNG debayer, Moffat convolution, Lanczos warp, kappa-sigma integration, GPU pipeline)
- **CFITSIO 4.6.3** — FITS image I/O
- **C++17** — CLI entry point

---

## Pipeline

| Stage | GPU (default) | CPU (`--cpu`) |
|---|---|---|
| 1. Debayering (star detection) | VNG demosaic → luminance (CUDA kernel) | VNG demosaic → luminance (OpenMP) |
| 2. Star Detection | Moffat PSF conv + threshold (CUDA) | Moffat PSF conv + threshold (OpenMP) |
| 3. RANSAC Alignment | DLT homography + RANSAC (CPU always) | DLT homography + RANSAC (CPU always) |
| 4. Debayering (warp) | VNG demosaic → luminance **or R/G/B** | VNG demosaic → luminance **or R/G/B** |
| 5. Lanczos Warp | nppiRemap + coord-map kernel (CUDA) | 6-tap backward-map warp (OpenMP) |
| 6. Integration | Mini-batch kappa-sigma (CUDA) | Full kappa-sigma (OpenMP) |

When the input CSV already contains pre-computed homographies (11-column format), stages 1–3 are skipped for both paths.

**Color output**: when a Bayer pattern is active (from `--bayer` or the FITS `BAYERPAT` keyword), stage 4 debayers to separate R, G, B planes; stages 5–6 run once per channel; the output FITS has `NAXIS=3` with planes R=1/G=2/B=3. Star detection (stages 1–2) always uses luminance regardless of color mode.

**Calibration pre-processing** (applied to every raw frame before debayering when `--dark`/`--flat` are provided):

| Step | What it does |
|---|---|
| Subtract dark master | Removes thermal noise and hot pixels |
| Divide by flat master | Corrects pixel sensitivity, vignetting, and dust |

---

## Build

### Prerequisites

| Dependency | Version |
|---|---|
| CUDA Toolkit | 12.x |
| CFITSIO | 4.6.3 |
| OpenMP | any (GCC) |
| CMake | >= 3.18 |

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Compile
cmake --build build --parallel $(nproc)
```

CUDA architectures are set to `86;89` (RTX 30xx / 40xx). Edit `CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPU families.

---

## Usage

```
dso_stacker -f <frames.csv> [options]
```

### Input CSV Formats

**2-column format** (star detection and RANSAC alignment are run automatically):

```csv
filepath, is_reference
/data/frame1.fits, 1
/data/frame2.fits, 0
/data/frame3.fits, 0
```

**11-column format** (pre-computed homographies, stages 1–3 skipped):

```csv
filepath, is_reference, h00, h01, h02, h10, h11, h12, h20, h21, h22
/data/frame1.fits, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1
/data/frame2.fits, 0, 1, 0, 2.5, 0, 1, 1.3, 0, 0, 1
```

- Exactly **one** row must have `is_reference = 1`.
- The nine `h` values form a row-major 3×3 **backward homography** (ref → src).

### Options

```
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

Sensor:
      --bayer <pattern>          CFA override: none | rggb | bggr | grbg | gbrg
                                 (default: auto-detect from FITS BAYERPAT keyword)
```

### Examples

Stack frames using automatic star detection and alignment (GPU):

```bash
dso_stacker -f frames.csv -o stacked.fits
```

Stack entirely on CPU (no GPU required):

```bash
dso_stacker -f frames.csv -o stacked.fits --cpu
```

Stack with pre-computed transforms, mean integration, and a larger batch:

```bash
dso_stacker -f transforms.csv -o stacked.fits --integration mean --batch-size 32
```

Stack a color camera image (RGGB sensor) with tighter outlier rejection:

```bash
dso_stacker -f frames.csv -o stacked.fits --bayer rggb --kappa 2.5 --iterations 5
```

Stack with calibration frames generated from lists (bias + dark + flat):

```bash
dso_stacker -f frames.csv -o stacked.fits \
    --bias  bias_frames.txt \
    --dark  dark_frames.txt \
    --flat  flat_frames.txt \
    --save-master-frames ./masters
```

Stack with pre-computed master FITS files and darkflat instead of bias:

```bash
dso_stacker -f frames.csv -o stacked.fits \
    --dark     master_dark.fits \
    --flat     flat_frames.txt \
    --darkflat master_darkflat.fits
```

---

## Tests

```bash
cd build && ctest --output-on-failure -V
```

| Test suite | Tests | What it covers |
|---|---|---|
| `test_cpu` | 29 | CSV parser, FITS I/O, integration, Lanczos CPU |
| `test_gpu` | 5 | GPU Lanczos (2 known pre-existing failures without GPU) |
| `test_star_detect` | 21 | CCL + CoM; Moffat convolution + threshold (CPU) |
| `test_ransac` | 13 | DLT homography + RANSAC |
| `test_debayer_cpu` | 10 | VNG debayer CPU: all patterns, uniform, non-uniform, edge cases |
| `test_integration_gpu` | 9 | GPU mini-batch kappa-sigma |
| `test_calibration` | 26 | CPU calibration: dark/flat apply, dead-pixel guard, dimension validation, FITS master loading, frame-list stacking, winsorized mean, median, bias/darkflat subtraction, flat normalization |
| `test_color` | 33 | OSC color output: `debayer_cpu_rgb` (validation, BAYER_NONE passthrough, uniform all 4 patterns, per-channel dominance RGGB/BGGR/GRBG/GBRG, luminance consistency, channel distinctness, non-negative); `fits_save_rgb` (validation, NAXIS=3, per-plane round-trip, gradient planes); `fits_get_bayer_pattern` (all 4 patterns + absent keyword) |

GPU test suites return exit code 77 (CTest SKIP) when no CUDA device is found.

---

## Benchmark

`bench.sh` times both paths and prints a speedup summary:

```bash
./bench.sh              # 3 runs, default CSV
./bench.sh -r 5         # 5 runs
./bench.sh -f other.csv # different input
```

Measured on 10 × 4656×3520 frames (star detection mode):

| Path | Wall time | Notes |
|---|---|---|
| GPU | ~4.1 s | Double-buffered CUDA stream overlap |
| CPU (OpenMP) | ~16.4 s | All stages parallelized |
| Speedup | **~4×** | |

Output agreement: PSNR ≈ 44.6 dB, mean relative error ≈ 0.25% in the image interior.
Differences arise from distinct floating-point paths (GPU nppi Lanczos vs hand-coded CPU,
GPU mini-batch vs single-pass CPU kappa-sigma, slight homography differences from Moffat conv precision).

---

## Python Tools

| Script | Purpose |
|---|---|
| `python/stacker.py` | Reference stacker: OpenCV Lanczos4 warp + kappa-sigma integration |
| `python/compute_transforms.py` | Compute homographies independently via astroalign and compare to CSV |

```bash
python3 python/stacker.py -f data/transform_mat.csv -o ref.fits
```
