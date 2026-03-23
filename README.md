# gpu-dso-stacker

[![CI](https://github.com/youruser/gpu-dso-stacker/actions/workflows/ci.yml/badge.svg)](https://github.com/youruser/gpu-dso-stacker/actions/workflows/ci.yml)

> A high-performance DSO (Deep Sky Object) stacker using CUDA for GPU-accelerated processing

**Pre-built binaries** (CLI + GUI) for Linux and Windows are available on the [Releases](https://github.com/youruser/gpu-dso-stacker/releases) page.

---

## Runtime Requirements

Pre-built binaries require an NVIDIA GPU and the CUDA 12.x runtime libraries installed on your system. The `--cpu` flag works without a GPU but the CUDA shared libraries must still be present.

### Linux

Install the CUDA 12 runtime packages from the NVIDIA repository:

```bash
# 1. Install the cuda-keyring package (sets up the NVIDIA apt repository)
#    Replace <distro> with: ubuntu2404, ubuntu2204, debian12, etc.
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 2. Install the runtime libraries (no compiler or dev headers needed)
sudo apt-get install cuda-cudart-12-9 libnpp-12-9
```

This installs `libcudart.so.12`, `libnppc.so.12`, and `libnppig.so.12`.

For RHEL / Fedora:

```bash
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/cuda-<distro>.repo
sudo dnf install cuda-cudart-12-9 libnpp-12-9
```

Replace `<distro>` with `rhel8`, `rhel9`, `fedora42`, etc.

> Any CUDA 12.x minor version will work (e.g. `cuda-cudart-12-6 libnpp-12-6`). The packages provide the `*.so.12` symlinks that the binary needs.

### Windows

Download the CUDA Toolkit 12.x installer from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) (or the [archive](https://developer.nvidia.com/cuda-toolkit-archive) for a specific version).

During installation, select **Custom** and enable at minimum:

- **CUDA Runtime** (`cudart`)
- **NPP** (NVIDIA Performance Primitives)
- **Display Driver** (if not already installed)

This places `cudart64_12.dll`, `nppc64_12.dll`, and `nppig64_12.dll` in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`, which the installer adds to `PATH`.

For unattended / silent installation:

```powershell
# Download the installer, then:
cuda_12.9.1_windows.exe -s cudart_12.9 npp_12.9 Display.Driver -n
```

---

## Technology Stack

- **C11** — Core library (FITS I/O, image I/O dispatch, CSV parser, Lanczos CPU, integration, debayer CPU, star detection, RANSAC, CPU pipeline)
- **OpenMP** — CPU parallelism for debayer, Moffat convolution, Lanczos warp, and integration
- **CUDA 12** — GPU acceleration (VNG debayer, Moffat convolution, Lanczos warp, kappa-sigma integration, GPU pipeline)
- **CFITSIO 4.6.3** — FITS image I/O
- **libtiff 4.5.1** — TIFF output (FP32, FP16, INT16, INT8; none/zip/lzw/rle compression)
- **libpng 1.6.43** — PNG output (INT8 and INT16)
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
| libtiff | 4.x |
| libpng | 1.6.x |
| OpenMP | any (GCC) |
| CMake | >= 3.18 |

### Linux

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Compile
cmake --build build --parallel $(nproc)
```

### Windows

Requires Visual Studio 2022, CUDA Toolkit 12.x, and [vcpkg](https://vcpkg.io/) for C library dependencies.

```powershell
# Install C dependencies via vcpkg
vcpkg install cfitsio tiff libpng --triplet x64-windows

# Configure
cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
      -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"

# Compile
cmake --build build --config Release --parallel
```

### CUDA Architectures

Default: `86;89` (RTX 30xx / 40xx). Override with:

```bash
cmake -B build -DDSO_CUDA_ARCHITECTURES="75;80;86;89;90" ...
```

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

Output format (format inferred from --output extension):
      --bit-depth <depth>        8 | 16 | f16 | f32  (default: f32)
                                 f16 is TIFF only; 8/16 require TIFF or PNG
                                 FITS always uses f32 regardless of this flag
      --tiff-compression <c>     none | zip | lzw | rle  (default: none; TIFF only)
      --stretch-min <float>      Lower bound for integer [0,MAX_INT] scaling (default: auto)
      --stretch-max <float>      Upper bound for integer [0,MAX_INT] scaling (default: auto)
                                 stretch-min/max are ignored for f16/f32 output
```

### Output Formats

The output format is determined by the extension of `--output`:

| Extension | Format | Supported bit depths | Compression |
|---|---|---|---|
| `.fits` `.fit` `.fts` | FITS | f32 (always) | none |
| `.tif` `.tiff` | TIFF | f32, f16, 16, 8 | none, zip, lzw, rle |
| `.png` | PNG | 16, 8 | always lossless DEFLATE |

**Integer scaling** (bit depths `8` and `16`): pixel values are linearly mapped to `[0, 255]` or `[0, 65535]`. By default the image min and max are used as bounds; use `--stretch-min` / `--stretch-max` to override. For RGB output the same bounds apply to all three channels to preserve colour ratios.

**FP16 precision**: values outside ~[6×10⁻⁸, 65504] are flushed to zero/infinity. For wide-dynamic-range astronomical data prefer `f32` (lossless) or `16` (quantised but full range).

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

Save as lossless FP32 TIFF (identical precision to FITS):

```bash
dso_stacker -f frames.csv -o stacked.tiff
```

Save as 16-bit TIFF with ZIP compression (good for sharing):

```bash
dso_stacker -f frames.csv -o stacked.tiff --bit-depth 16 --tiff-compression zip
```

Save as 16-bit PNG (web-compatible, lossless):

```bash
dso_stacker -f frames.csv -o stacked.png --bit-depth 16
```

Save as 8-bit PNG with explicit stretch bounds:

```bash
dso_stacker -f frames.csv -o preview.png --bit-depth 8 --stretch-min 0 --stretch-max 65535
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
| `test_image_io` | 21 | Format detection; FITS passthrough; TIFF FP32/FP16/INT16/INT8 mono+RGB; TIFF zip/lzw/rle round-trips; PNG 8-bit/16-bit mono+RGB; error cases (FP32→PNG, unknown ext); auto stretch |

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

## GUI Frontend

A PySide6 desktop application wrapping the CLI. Provides drag-and-drop frame management, all stacking options with conditional visibility, and YAML project save/load.

Pre-built GUI bundles (Linux and Windows) are available on the [Releases](https://github.com/youruser/gpu-dso-stacker/releases) page — no Python installation required.

### Install (from source)

```bash
pip install PySide6 pyyaml
```

### Launch

```bash
python src/GUI/main.py
```

### Features

- **Tabs**: Light, Dark, Flat, Bias, Darkflat, Stacking Options
- **Drag-and-drop** FITS frames (`.fit` / `.fits` / `.fts`) onto any tab
- **Reference frame** selection in the Light tab (radio button column; default: first frame)
- **File info**: filename, path, size, and dimensions (W×H) loaded asynchronously from FITS headers
- **Conditional options**: kappa/iterations hidden for mean integration; batch size hidden in CPU mode; TIFF compression hidden for non-TIFF output; stretch bounds hidden for floating-point output; bit depth options restricted per output format
- **Bias / Darkflat mutual exclusion**: loading frames in one disables the other tab
- **Project files** (`.yaml`): save and reload complete project state (frame lists + all options)
- **Live log**: subprocess stdout/stderr streamed to a collapsible log panel
- **Abort**: terminate a running stack without leaving zombie processes

### Source layout

```
src/GUI/
├── main.py                      # Entry point
├── main_window.py               # MainWindow: tabs, menu, Run/Abort, log panel
├── project.py                   # ProjectState dataclass + YAML save/load
├── runner.py                    # SubprocessRunner (QThread)
├── fits_meta.py                 # Async FITS header reader (astropy, QThreadPool)
├── utils.py                     # build_command, format_size, detect_output_format
└── widgets/
    ├── frame_table.py           # FrameTableWidget: DnD file list base
    ├── light_tab.py             # Light tab + reference radio column
    ├── calib_tab.py             # Dark/Flat/Bias/Darkflat tabs
    └── stacking_options_tab.py  # All CLI parameters with conditional visibility
```

---

## CI/CD

GitHub Actions build and test on every push to `main` and every PR.

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Push / PR to `main` | Build + test (Linux & Windows), package GUI bundles |
| `release.yml` | Tag push `v*` | Same as CI, then creates a GitHub Release with archives |

**Linux** builds inside `nvidia/cuda:12.6.3-devel-ubuntu22.04`; **Windows** uses `Jimver/cuda-toolkit` + vcpkg. GPU tests auto-skip (exit code 77) on runners without a GPU. GUI packaging uses PyInstaller.

Release artifacts:

| Archive | Contents |
|---|---|
| `dso-stacker-cli-linux-x86_64.tar.gz` | CLI binary |
| `dso-stacker-gui-linux-x86_64.tar.gz` | GUI bundle (includes CLI) |
| `dso-stacker-cli-windows-x86_64.zip` | CLI .exe + DLLs |
| `dso-stacker-gui-windows-x86_64.zip` | GUI bundle (includes CLI) |

---

## Python Tools

| Script | Purpose |
|---|---|
| `python/stacker.py` | Reference stacker: OpenCV Lanczos4 warp + kappa-sigma integration |
| `python/compute_transforms.py` | Compute homographies independently via astroalign and compare to CSV |

```bash
python3 python/stacker.py -f data/transform_mat.csv -o ref.fits
```

---

## License

This software is **proprietary**. See [LICENSE](LICENSE) for terms.

This software dynamically links the NVIDIA CUDA Toolkit and NVIDIA
Performance Primitives (NPP). Users must have the CUDA runtime
installed on their system. Both libraries are covered under the single
[NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/), which users
accept when installing the CUDA Toolkit.

Third-party open-source components (CFITSIO, libtiff, libpng, PySide6,
PyYAML, getopt\_port) are used under their respective permissive or
LGPL licenses. See [THIRD\_PARTY\_LICENSES](THIRD_PARTY_LICENSES) for
full attribution and license texts.
