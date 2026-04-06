# gpu-dso-stacker

[English](README.md) | [한국어](README.ko.md)

[![CI](https://github.com/gs18113/gpu-dso-stacker/actions/workflows/ci.yml/badge.svg)](https://github.com/gs18113/gpu-dso-stacker/actions/workflows/ci.yml)

> CUDA(Linux/Windows) 및 Metal 스캐폴딩(Apple Silicon) 백엔드를 지원하는 고성능 DSO(Deep Sky Object) 스태커

Linux, macOS, Windows용 **사전 빌드 바이너리**(CLI + GUI)는 [Releases](https://github.com/gs18113/gpu-dso-stacker/releases) 페이지에서 받을 수 있습니다.

---

## 런타임 요구사항

런타임 요구사항은 릴리스 아티팩트 유형에 따라 다릅니다:

- **CPU 전용 빌드** (`*-cpu` 아카이브): NVIDIA GPU나 CUDA 런타임이 **필요 없습니다**.
- **GPU 선택 빌드** (`*-gpu` 아카이브): NVIDIA GPU와 CUDA 12.x 런타임 라이브러리가 필요합니다.

### Linux

NVIDIA 저장소에서 CUDA 12 런타임 패키지를 설치하세요.

**Debian / Ubuntu:**

```bash
# 1. cuda-keyring 패키지 설치 (NVIDIA apt 저장소 설정)
#    <distro>를 ubuntu2404, ubuntu2204, debian12 등으로 대체하세요.
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 2. 런타임 라이브러리 설치 (컴파일러나 개발 헤더는 필요 없음)
sudo apt-get install cuda-cudart-12-9 libnpp-12-9
```

`libcudart.so.12`, `libnppc.so.12`, `libnppig.so.12`가 설치됩니다.

**RHEL / Fedora:**

```bash
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/cuda-<distro>.repo
sudo dnf install cuda-cudart-12-9 libnpp-12-9
```

`<distro>`를 `rhel8`, `rhel9`, `fedora42` 등으로 대체하세요.

> CUDA 12.x의 모든 마이너 버전이 호환됩니다(예: `cuda-cudart-12-6 libnpp-12-6`). 해당 패키지가 바이너리에 필요한 `*.so.12` 심볼릭 링크를 제공합니다.

### Windows

[developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)에서 CUDA Toolkit 12.x 설치 프로그램을 다운로드하세요 (특정 버전은 [아카이브](https://developer.nvidia.com/cuda-toolkit-archive) 참조).

설치 중 **사용자 지정**을 선택하고 최소한 다음 항목을 활성화하세요:

- **CUDA Runtime** (`cudart`)
- **NPP** (NVIDIA Performance Primitives)
- **Display Driver** (미설치 시)

`cudart64_12.dll`, `nppc64_12.dll`, `nppig64_12.dll`이 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`에 설치됩니다.
최근 CUDA 설치 프로그램은 `CUDA_PATH`를 설정하지만 `PATH`를 업데이트하지 않을 수 있습니다. GUI 런처가 CLI 서브프로세스에 대해 `%CUDA_PATH%\bin`을 `PATH`에 자동으로 추가하므로 GPU 실행에 문제가 없습니다.

무인/자동 설치:

```powershell
# 설치 프로그램 다운로드 후:
cuda_12.9.1_windows.exe -s cudart_12.9 npp_12.9 Display.Driver -n
```

### macOS (Gatekeeper 우회)

다운로드, 압축 해제, 격리 해제를 한 번에:

```bash
mkdir -p ~/DSOStacker && curl -fL https://github.com/gs18113/gpu-dso-stacker/releases/latest/download/dso-stacker-gui-macos-arm64-metal.tar.gz | tar xz -C ~/DSOStacker && xattr -cr ~/DSOStacker && chmod +x ~/DSOStacker/DSOStacker ~/DSOStacker/_internal/bin/dso_stacker
```

Metal 가속이 필요 없으면 URL에서 `metal`을 `cpu`로 변경하세요.

---

## 기술 스택

- **C11** — 핵심 라이브러리 (FITS I/O, 이미지 I/O 디스패치, CSV 파서, Lanczos CPU, 적분, 디베이어 CPU, 별 검출, 삼각형 매칭, CPU 파이프라인)
- **OpenMP** — 디베이어, Moffat 컨볼루션, Lanczos 워프, 적분의 CPU 병렬화
- **CUDA 12** — GPU 가속 (VNG 디베이어, Moffat 컨볼루션, Lanczos 워프, 카파-시그마 적분, GPU 파이프라인)
- **Metal (스캐폴딩)** — Apple Silicon 백엔드 진입점 (단계적 커널 포팅 계획)
- **CFITSIO 4.6.3** — FITS 이미지 I/O
- **libtiff 4.5.1** — TIFF 출력 (FP32, FP16, INT16, INT8; none/zip/lzw/rle 압축)
- **libpng 1.6.43** — PNG 출력 (INT8 및 INT16)
- **LibRaw 0.21+** — RAW 카메라 파일 입력 (CR2, NEF, ARW, DNG 등); 선택사항, `-DDSO_ENABLE_LIBRAW=ON`
- **C++17** — CLI 엔트리포인트

---

## 파이프라인

| 단계 | GPU (기본값) | CPU (`--cpu`) |
|---|---|---|
| 1. 디베이어링 (별 검출) | VNG 디모자이크 → 휘도 (CUDA 커널) | VNG 디모자이크 → 휘도 (OpenMP) |
| 2. 별 검출 | Moffat PSF 컨볼루션 + 임계값 (CUDA) | Moffat PSF 컨볼루션 + 임계값 (OpenMP) |
| 3. 삼각형 매칭 정렬 | 삼각형/별자리 매칭 + DLT (`--match-device`로 CPU/GPU 선택; 기본값은 스태킹 장치를 따름) | 삼각형/별자리 매칭 + DLT (`--cpu` 사용 시 기본 CPU) |
| 4. 디베이어링 (워프) | VNG 디모자이크 → 휘도 **또는 R/G/B** | VNG 디모자이크 → 휘도 **또는 R/G/B** |
| 5. Lanczos 워프 | nppiRemap + 좌표맵 커널 (CUDA) | 6-탭 역방향 매핑 워프 (OpenMP) |
| 6. 적분 | 미니배치 카파-시그마 (CUDA) | 풀 카파-시그마 (OpenMP) |

**컬러 출력**: 베이어 패턴이 활성화되면 (`--bayer` 또는 FITS `BAYERPAT` 키워드) 4단계에서 별도의 R, G, B 평면으로 디베이어링합니다. 5~6단계는 채널별로 한 번씩 실행되며 출력 FITS는 `NAXIS=3` (R=1/G=2/B=3)입니다. 별 검출(1~2단계)은 컬러 모드와 관계없이 항상 휘도를 사용합니다.

**삼각형 매칭 실패 처리**: 비기준 프레임의 정렬이 실패하면(별이 너무 적거나 삼각형 매칭 불일치) 해당 프레임을 건너뛰고 처리를 계속합니다. 완료 시 CLI에서 성공 및 건너뛴 프레임 수를 요약 출력합니다.

**보정 전처리** (`--dark`/`--flat` 제공 시 디베이어링 전 모든 원본 프레임에 적용):

| 단계 | 수행 내용 |
|---|---|
| 다크 마스터 차감 | 열잡음 및 핫픽셀 제거 |
| 플랫 마스터 나누기 | 픽셀 감도, 비네팅, 먼지 보정 |

---

## 빌드

### 필수 구성 요소

| 의존성 | 버전 |
|---|---|
| CUDA Toolkit | 12.x |
| CFITSIO | 4.6.3 |
| libtiff | 4.x |
| libpng | 1.6.x |
| LibRaw | >= 0.21 (선택사항) |
| OpenMP | any (GCC) |
| CMake | >= 3.18 |

### Linux

```bash
# 구성
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# 컴파일
cmake --build build --parallel $(nproc)
```

CPU 전용 구성 (CUDA 툴킷 불필요):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDSO_ENABLE_CUDA=OFF
cmake --build build --parallel $(nproc)
```

RAW 카메라 파일 지원 활성화 (LibRaw 필요):

```bash
# LibRaw 설치: apt install libraw-dev (Linux), brew install libraw (macOS)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDSO_ENABLE_LIBRAW=ON ...
```

### Windows

Visual Studio 2022, CUDA Toolkit 12.x, C 라이브러리 의존성을 위한 [vcpkg](https://vcpkg.io/)가 필요합니다.

```powershell
# vcpkg로 C 의존성 설치
vcpkg install cfitsio tiff libpng libraw --triplet x64-windows

# 구성
cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
      -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"

# 컴파일
cmake --build build --config Release --parallel
```

### Apple Silicon / macOS (스캐폴딩)

Metal 백엔드 지원은 CMake 옵션으로 스캐폴딩되어 있습니다. Phase-1 동작은
Metal 커널을 단계적으로 포팅하는 동안 CPU 파이프라인으로 폴백하여
수치 의미론을 안전하게 유지합니다.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DDSO_ENABLE_CUDA=OFF \
      -DDSO_ENABLE_METAL=ON
cmake --build build --parallel
```

런타임 백엔드 선택:
- `--backend auto` (기본값): 기존 동작 유지 (`--cpu`는 여전히 CPU를 강제)
- `--backend cpu`
- `--backend cuda`
- `--backend metal`

### CUDA 아키텍처

기본값: `86;89` (RTX 30xx / 40xx). 다음으로 오버라이드:

```bash
cmake -B build -DDSO_CUDA_ARCHITECTURES="75;80;86;89;90" ...
```

---

## 사용법

```
dso_stacker -f <frames.csv> [options]
```

### 입력 CSV 형식

```csv
filepath, is_reference
/data/frame1.fits, 1
/data/frame2.fits, 0
/data/frame3.fits, 0
```

정확히 **하나**의 행이 `is_reference = 1`이어야 합니다. 입력 프레임은 **FITS** (`.fits`, `.fit`, `.fts`) 또는 **RAW 카메라 파일** (`.cr2`, `.cr3`, `.nef`, `.arw`, `.orf`, `.rw2`, `.raf`, `.dng`, `.pef`, `.srw`, `.raw`, `.3fr`, `.iiq`, `.rwl`, `.nrw`)을 사용할 수 있으며, `-DDSO_ENABLE_LIBRAW=ON`으로 빌드해야 합니다. 해상도가 일치하면 FITS와 RAW 프레임을 같은 CSV에 혼합할 수 있습니다.

### 옵션

```
I/O:
  -f, --file <path>              입력 CSV 파일 (필수)
  -o, --output <path>            출력 FITS 파일 (기본값: output.fits)

적분:
      --cpu                      모든 파이프라인 단계를 CPU에서 실행 (OpenMP 가속)
      --integration <method>     mean | kappa-sigma | auto-adaptive (기본값: kappa-sigma)
      --kappa <float>            시그마 클리핑 임계값 (기본값: 3.0)
      --iterations <int>         픽셀당 최대 클리핑 반복 횟수 (기본값: 3)
      --batch-size <int>         GPU 적분 미니배치 크기 (기본값: 16)

별 검출 (2열 CSV 전용):
      --star-sigma <float>       검출 임계값 σ 단위 (기본값: 3.0)
      --moffat-alpha <float>     Moffat PSF alpha / FWHM (기본값: 2.5)
      --moffat-beta <float>      Moffat PSF beta / 날개 기울기 (기본값: 2.0)
      --top-stars <int>          매칭용 Top-K 별 (기본값: 50)
      --min-stars <int>          삼각형 매칭을 위한 최소 별 수 (기본값: 6)

삼각형 매칭 (2열 CSV 전용):
      --triangle-iters <int>     최대 삼각형 매칭 반복 횟수 (기본값: 1000)
      --triangle-thresh <float>  인라이어 재투영 임계값 px (기본값: 2.0)
      --ransac-iters <int>       --triangle-iters의 구 별칭
      --ransac-thresh <float>    --triangle-thresh의 구 별칭
      --match-radius <float>     별 매칭 검색 반경 px (기본값: 30.0)
      --match-device <device>    auto | cpu | gpu (기본값: auto = 스태킹 장치)
      --backend <backend>        auto | cpu | cuda | metal (기본값: auto)

보정 (디베이어링 전 적용; bias와 darkflat은 상호 배타적):
      --dark <path>              마스터 다크 FITS 또는 다크 FITS 경로의 텍스트 목록
      --bias <path>              마스터 바이어스 FITS 또는 바이어스 FITS 경로의 텍스트 목록
      --flat <path>              마스터 플랫 FITS 또는 플랫 FITS 경로의 텍스트 목록
      --darkflat <path>          마스터 다크플랫 FITS 또는 다크플랫 FITS 경로의 텍스트 목록
      --save-master-frames <dir> 생성된 마스터 저장 디렉터리 (기본값: ./master)
      --dark-method <method>     winsorized-mean | median (기본값: winsorized-mean)
      --bias-method <method>     winsorized-mean | median (기본값: winsorized-mean)
      --flat-method <method>     winsorized-mean | median (기본값: winsorized-mean)
      --darkflat-method <method> winsorized-mean | median (기본값: winsorized-mean)
      --wsor-clip <float>        윈저화 평균 클리핑 비율 (기본값: 0.1)
                                 유효 범위: [0.0, 0.49]

센서:
      --bayer <pattern>          CFA 오버라이드: none | rggb | bggr | grbg | gbrg
                                 (기본값: FITS BAYERPAT 키워드에서 자동 감지)

출력 형식 (--output 확장자에서 추론):
      --bit-depth <depth>        8 | 16 | f16 | f32  (기본값: f32)
                                 f16은 TIFF 전용; 8/16은 TIFF 또는 PNG 필요
                                 FITS는 이 플래그와 관계없이 항상 f32 사용
      --tiff-compression <c>     none | zip | lzw | rle  (기본값: none; TIFF 전용)
      --stretch-min <float>      정수 [0,MAX_INT] 스케일링 하한 (기본값: auto)
      --stretch-max <float>      정수 [0,MAX_INT] 스케일링 상한 (기본값: auto)
                                 stretch-min/max는 f16/f32 출력 시 무시됨
```

### 출력 형식

출력 형식은 `--output`의 확장자에 따라 결정됩니다:

| 확장자 | 형식 | 지원 비트 심도 | 압축 |
|---|---|---|---|
| `.fits` `.fit` `.fts` | FITS | f32 (항상) | 없음 |
| `.tif` `.tiff` | TIFF | f32, f16, 16, 8 | none, zip, lzw, rle |
| `.png` | PNG | 16, 8 | 항상 무손실 DEFLATE |

**정수 스케일링** (비트 심도 `8` 및 `16`): 픽셀 값이 `[0, 255]` 또는 `[0, 65535]`로 선형 매핑됩니다. 기본적으로 이미지의 min/max를 경계로 사용하며 `--stretch-min` / `--stretch-max`로 오버라이드할 수 있습니다. RGB 출력 시 색상 비율 보존을 위해 세 채널 모두에 동일한 경계가 적용됩니다.

**FP16 정밀도**: ~[6×10⁻⁸, 65504] 범위 밖의 값은 0/infinity로 플러시됩니다. 넓은 다이나믹 레인지의 천체 데이터에는 `f32`(무손실) 또는 `16`(양자화되지만 전체 범위)을 권장합니다.

### 예제

자동 별 검출 및 정렬로 프레임 스태킹 (GPU):

```bash
dso_stacker -f frames.csv -o stacked.fits
```

CPU 전용으로 스태킹 (GPU 불필요):

```bash
dso_stacker -f frames.csv -o stacked.fits --cpu
```

평균 적분과 더 큰 배치로 스태킹:

```bash
dso_stacker -f frames.csv -o stacked.fits --integration mean --batch-size 32
```

컬러 카메라 이미지 (RGGB 센서)를 더 엄격한 이상치 제거로 스태킹:

```bash
dso_stacker -f frames.csv -o stacked.fits --bayer rggb --kappa 2.5 --iterations 5
```

보정 프레임 목록에서 생성하여 스태킹 (바이어스 + 다크 + 플랫):

```bash
dso_stacker -f frames.csv -o stacked.fits \
    --bias  bias_frames.txt \
    --dark  dark_frames.txt \
    --flat  flat_frames.txt \
    --save-master-frames ./masters
```

무손실 FP32 TIFF 저장 (FITS와 동일한 정밀도):

```bash
dso_stacker -f frames.csv -o stacked.tiff
```

16비트 TIFF + ZIP 압축 (공유에 적합):

```bash
dso_stacker -f frames.csv -o stacked.tiff --bit-depth 16 --tiff-compression zip
```

16비트 PNG (웹 호환, 무손실):

```bash
dso_stacker -f frames.csv -o stacked.png --bit-depth 16
```

명시적 스트레치 범위의 8비트 PNG:

```bash
dso_stacker -f frames.csv -o preview.png --bit-depth 8 --stretch-min 0 --stretch-max 65535
```

사전 계산된 마스터 FITS 파일과 바이어스 대신 다크플랫 사용:

```bash
dso_stacker -f frames.csv -o stacked.fits \
    --dark     master_dark.fits \
    --flat     flat_frames.txt \
    --darkflat master_darkflat.fits
```

---

## 테스트

```bash
cd build && ctest --output-on-failure -V
```

| 테스트 모음 | 테스트 수 | 커버리지 |
|---|---|---|
| `test_cpu` | 29 | CSV 파서, FITS I/O, 적분, Lanczos CPU |
| `test_gpu` | 5 | GPU Lanczos (GPU 없이 2개 기존 실패) |
| `test_star_detect` | 21 | CCL + CoM; Moffat 컨볼루션 + 임계값 (CPU) |
| `test_ransac` | 13 | DLT 호모그래피 + 삼각형 매칭 |
| `test_debayer_cpu` | 10 | VNG 디베이어 CPU: 모든 패턴, 균일, 비균일, 엣지 케이스 |
| `test_integration_gpu` | 9 | GPU 미니배치 카파-시그마 |
| `test_calibration` | 26 | CPU 보정: 다크/플랫 적용, 데드픽셀 가드, 차원 검증, FITS 마스터 로딩, 프레임 목록 스태킹, 윈저화 평균, 중앙값, 바이어스/다크플랫 차감, 플랫 정규화 |
| `test_color` | 33 | OSC 컬러 출력: `debayer_cpu_rgb` (검증, BAYER_NONE 패스스루, 균일 4패턴, 채널별 우세, 휘도 일관성, 채널 구별성, 비음수); `fits_save_rgb` (검증, NAXIS=3, 평면별 왕복, 그래디언트 평면); `fits_get_bayer_pattern` (4패턴 + 키워드 부재) |
| `test_image_io` | 21 | 형식 감지; FITS 패스스루; TIFF FP32/FP16/INT16/INT8 모노+RGB; TIFF zip/lzw/rle 왕복; PNG 8비트/16비트 모노+RGB; 에러 케이스; 자동 스트레치 |
| `test_pipeline_backend` | — | 백엔드 디스패치 검증 |
| `test_raw_io` | 10 | `frame_is_raw` 확장자 검출 (긍정 + 부정); `frame_load` FITS 폴백; `frame_get_bayer_pattern` 디스패치; `frame_get_dimensions` 디스패치; RAW 비활성화 시 에러 경로; LibRaw 에러 핸들링 (조건부) |

GPU 테스트 모음은 CUDA 장치가 없으면 종료 코드 77 (CTest SKIP)을 반환합니다.

---

## 벤치마크

`bench.sh`가 두 경로의 시간을 측정하고 속도 향상 요약을 출력합니다:

```bash
./bench.sh              # 3회, 기본 CSV
./bench.sh -r 5         # 5회
./bench.sh -f other.csv # 다른 입력
```

10 × 4656×3520 프레임 (별 검출 모드) 측정:

| 경로 | 실행 시간 | 비고 |
|---|---|---|
| GPU | ~2.6 s | 더블 버퍼링 CUDA 스트림 오버랩 |
| CPU (OpenMP) | ~11.5 s | 모든 단계 병렬화 |
| 속도 향상 | **~4.4×** | |

출력 일치도: PSNR ≈ 82.4 dB, 이미지 내부 평균 상대 오차 ≈ 0.02%.
차이는 서로 다른 부동소수점 경로(GPU nppi Lanczos vs 수동 코딩 CPU, GPU 미니배치 vs 단일 패스 CPU 카파-시그마, Moffat 컨볼루션 정밀도에 따른 미세한 호모그래피 차이)에 기인합니다.

---

## GUI 프론트엔드

CLI를 래핑하는 PySide6 데스크톱 애플리케이션입니다. 드래그 앤 드롭 프레임 관리, 모든 스태킹 옵션의 조건부 표시/숨김, YAML 프로젝트 저장/로드를 제공합니다.

사전 빌드 GUI 번들은 [Releases](https://github.com/gs18113/gpu-dso-stacker/releases) 페이지에서 받을 수 있습니다 — Python 설치가 필요 없습니다.

### 설치 (소스)

```bash
pip install PySide6 pyyaml
```

### 실행

```bash
python src/GUI/main.py
```

### 기능

- **탭**: Light, Dark, Flat, Bias, Darkflat, Stacking Options
- **드래그 앤 드롭**: FITS 프레임 (`.fit` / `.fits` / `.fts`) 및 RAW 카메라 파일 (`.cr2`, `.nef`, `.arw`, `.dng` 등)을 아무 탭에나 드롭
- **기준 프레임 선택**: Light 탭의 라디오 버튼 열 (기본값: 첫 번째 프레임)
- **파일 정보**: 파일명, 경로, 크기, 해상도 (W×H)를 FITS 헤더 또는 RAW 메타데이터(선택적 `rawpy` 사용)에서 비동기 로딩
- **조건부 옵션**: 평균 적분 시 카파/반복 숨김; CPU 모드 시 배치 크기 숨김; 비TIFF 출력 시 TIFF 압축 숨김; 부동소수점 출력 시 스트레치 범위 숨김; 출력 형식별 비트 심도 제한
- **바이어스 / 다크플랫 상호 배제**: 한쪽에 프레임을 로딩하면 다른 쪽 탭이 비활성화
- **프로젝트 파일** (`.yaml`): 전체 프로젝트 상태 저장 및 복원 (프레임 목록 + 모든 옵션)
- **실시간 로그**: 서브프로세스 stdout/stderr를 접이식 로그 패널에 스트리밍
- **중단**: 좀비 프로세스 없이 실행 중인 스태킹 종료

### 소스 레이아웃

```
src/GUI/
├── main.py                      # 엔트리포인트
├── main_window.py               # MainWindow: 탭, 메뉴, Run/Abort, 로그 패널
├── project.py                   # ProjectState 데이터클래스 + YAML 저장/로드
├── runner.py                    # SubprocessRunner (QThread)
├── fits_meta.py                 # 비동기 FITS 헤더 리더 (astropy, QThreadPool)
├── utils.py                     # build_command, format_size, detect_output_format
└── widgets/
    ├── frame_table.py           # FrameTableWidget: DnD 파일 목록 베이스
    ├── light_tab.py             # Light 탭 + 기준 라디오 열
    ├── calib_tab.py             # Dark/Flat/Bias/Darkflat 탭
    └── stacking_options_tab.py  # 모든 CLI 파라미터 + 조건부 표시
```

---

## CI/CD

GitHub Actions가 `main`에 대한 모든 푸시와 PR에서 빌드 및 테스트를 실행합니다.

| 워크플로 | 트리거 | 수행 내용 |
|---|---|---|
| `ci.yml` | `main` 푸시 / PR | 빌드 + 테스트 (Linux, Windows, macOS), GUI 번들 패키징 |
| `release.yml` | 태그 푸시 `v*` | CI와 동일 + GitHub Release 생성 |

**Linux**는 `nvidia/cuda:12.6.2-devel-ubuntu22.04` 컨테이너에서 빌드합니다. **Windows**는 `Jimver/cuda-toolkit` + vcpkg 사용 (병렬 `windows-gpu` 및 `windows-cpu` 작업). **macOS**는 `macos-14` (Apple Silicon)에서 Metal 활성화 및 CPU 전용 변형을 빌드하고 필요한 dylib를 번들합니다. GPU 테스트는 GPU가 없는 러너에서 자동 건너뜀(종료 코드 77)됩니다. GUI 패키징은 PyInstaller를 사용합니다.

릴리스 아티팩트:

| 아카이브 | 내용 |
|---|---|
| `dso-stacker-cli-linux-x86_64-cpu.tar.gz` | CPU 전용 CLI 바이너리 |
| `dso-stacker-cli-linux-x86_64-gpu.tar.gz` | GPU 선택 CLI 바이너리 |
| `dso-stacker-cli-macos-arm64-cpu.tar.gz` | CPU 전용 CLI 바이너리 |
| `dso-stacker-cli-macos-arm64-metal.tar.gz` | Metal 활성화 CLI 바이너리 |
| `dso-stacker-gui-linux-x86_64-cpu.tar.gz` | CPU 전용 CLI (`bin/dso_stacker`) 포함 GUI 번들 |
| `dso-stacker-gui-linux-x86_64-gpu.tar.gz` | GPU 선택 CLI (`bin/dso_stacker`) 포함 GUI 번들 |
| `dso-stacker-gui-macos-arm64-cpu.tar.gz` | CPU 전용 CLI (`bin/dso_stacker`) 포함 GUI 번들 |
| `dso-stacker-gui-macos-arm64-metal.tar.gz` | Metal 활성화 CLI (`bin/dso_stacker`) 포함 GUI 번들 |
| `dso-stacker-cli-windows-x86_64-cpu.zip` | CPU 전용 CLI .exe + DLL |
| `dso-stacker-cli-windows-x86_64-gpu.zip` | GPU 선택 CLI .exe + DLL |
| `dso-stacker-gui-windows-x86_64-cpu.zip` | CPU 전용 CLI (`bin/dso_stacker.exe`) 포함 GUI 번들 |
| `dso-stacker-gui-windows-x86_64-gpu.zip` | GPU 선택 CLI (`bin/dso_stacker.exe`) 포함 GUI 번들 |

---

## Python 도구

| 스크립트 | 용도 |
|---|---|
| `python/stacker.py` | 레퍼런스 스태커: OpenCV Lanczos4 워프 + 카파-시그마 적분 |
| `python/compute_transforms.py` | astroalign으로 호모그래피를 독립 계산하고 CSV와 비교 |

```bash
python3 python/stacker.py -f data/transform_mat.csv -o ref.fits
```

---

## 라이선스

이 프로젝트는 **GNU General Public License v3.0**으로 배포됩니다. [LICENSE](LICENSE)를 참조하세요.

GPU 선택 바이너리는 NVIDIA CUDA Toolkit 및 NVIDIA Performance
Primitives(NPP)를 동적으로 링크합니다. GPU 선택 빌드를 사용하는 경우
시스템에 CUDA 런타임이 설치되어 있어야 합니다. CPU 전용 빌드는 CUDA
런타임 라이브러리에 의존하지 않습니다. CUDA와 NPP는
[NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/)에 의해 관리되며,
사용자는 CUDA Toolkit 설치 시 이에 동의합니다.

재배포 관점에서, 이 저장소에서 번들/사용되는 서드파티 라이선스는 일반적으로
GPLv3와 호환됩니다(허용적 및 LGPLv3 구성 요소). CUDA/NPP는 선택사항이지만
NVIDIA CUDA EULA에 의해 관리되므로, GPU 아티팩트를 재배포하는 경우 이
프로젝트에 대한 GPLv3 의무와 NVIDIA 런타임 구성 요소에 대한 CUDA EULA
조건을 모두 충족해야 합니다.

서드파티 오픈소스 구성 요소(CFITSIO, libtiff, libpng, LibRaw, PySide6,
PyYAML, getopt\_port)는 각각의 허용적 또는 LGPL 라이선스에 따라
사용됩니다. LibRaw는 LGPL 2.1과 CDDL 1.0의 이중 라이선스이며, 이
프로젝트에서는 GPLv3와 호환되는 LGPL 2.1에 따라 사용합니다. 전체
저작자 표시 및 라이선스 텍스트는
[THIRD\_PARTY\_LICENSES](THIRD_PARTY_LICENSES)를 참조하세요.
