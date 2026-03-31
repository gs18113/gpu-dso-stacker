# gpu-dso-stacker

[English](README.md) | [한국어](README.ko.md)

[![CI](https://github.com/gs18113/gpu-dso-stacker/actions/workflows/ci.yml/badge.svg)](https://github.com/gs18113/gpu-dso-stacker/actions/workflows/ci.yml)

> CUDA(Linux/Windows) 및 Metal 스캐폴딩(Apple Silicon) 백엔드를 지원하는 고성능 DSO(Deep Sky Object) 스태커

Linux, macOS, Windows용 **사전 빌드 바이너리**(CLI + GUI)는 [Releases](https://github.com/gs18113/gpu-dso-stacker/releases)에서 받을 수 있습니다.

---

## 런타임 요구사항

런타임 요구사항은 릴리스 아티팩트 유형에 따라 다릅니다:

- **CPU 전용 빌드** (`*-cpu` 아카이브): NVIDIA GPU나 CUDA 런타임이 **필요 없습니다**.
- **GPU 선택 빌드** (`*-gpu` 아카이브): NVIDIA GPU와 CUDA 12.x 런타임 라이브러리가 필요합니다.

### Linux

```bash
# 1) NVIDIA apt 저장소 설정
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 2) 런타임 라이브러리 설치
sudo apt-get install cuda-cudart-12-9 libnpp-12-9
```

RHEL/Fedora 계열:

```bash
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/cuda-<distro>.repo
sudo dnf install cuda-cudart-12-9 libnpp-12-9
```

### Windows

[CUDA 다운로드 페이지](https://developer.nvidia.com/cuda-downloads)에서 CUDA Toolkit 12.x를 설치하세요. 최소 구성으로 Runtime(`cudart`), NPP, Display Driver를 선택하면 됩니다.

---

## 기술 스택

- **C11**: 핵심 라이브러리(FITS I/O, 이미지 I/O, CSV, Lanczos CPU, 적분, 디베이어, 별 검출, 매칭, CPU 파이프라인)
- **OpenMP**: CPU 병렬화
- **CUDA 12**: GPU 가속
- **Metal (스캐폴딩)**: Apple Silicon 백엔드 진입점
- **CFITSIO 4.6.3**, **libtiff 4.5.1**, **libpng 1.6.43**
- **C++17**: CLI 엔트리포인트

---

## 빌드

### Linux

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --parallel $(nproc)
./build/dso_stacker
```

CPU 전용 빌드:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDSO_ENABLE_CUDA=OFF
cmake --build build --parallel $(nproc)
```

### Windows

```powershell
vcpkg install cfitsio tiff libpng --triplet x64-windows
cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
      -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
cmake --build build --config Release --parallel
```

---

## CLI 사용법

```text
dso_stacker -f <frames.csv> [options]
```

핵심 옵션:

- `-f, --file <path>`: 입력 CSV(필수)
- `-o, --output <path>`: 출력 파일
- `--backend auto|cpu|cuda|metal`: 백엔드 선택(기본 auto)
- `--integration mean|kappa-sigma`: 적분 방식
- `--kappa <float>`, `--iterations <int>`: kappa-sigma 파라미터
- `--match-device auto|cpu|gpu`: 삼각형 매칭 장치
- `--bayer none|rggb|bggr|grbg|gbrg`: CFA 패턴 지정
- `--dark`, `--bias`, `--flat`, `--darkflat`: 보정 프레임

입력 CSV는 2열 형식(`filepath, is_reference`)만 지원하며, 기준 프레임(`is_reference=1`)은 정확히 1개여야 합니다.

---

## GUI 실행

```bash
pip install PySide6 pyyaml astropy
python src/GUI/main.py
```

---

## 테스트

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

ctest --test-dir build --output-on-failure
```

주요 테스트 실행 파일:

- `test_cpu`
- `test_star_detect`
- `test_ransac`
- `test_debayer_cpu`
- `test_calibration`
- `test_image_io`
- `test_color`
- `test_audit`
- `test_gpu` / `test_integration_gpu` (GPU 환경에서 실행)

---

## 라이선스

이 프로젝트는 **GNU General Public License v3.0 (GPLv3)** 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

CUDA Toolkit 및 NPP는 [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/) 조건을 따릅니다. 기타 서드파티 컴포넌트(CFITSIO, libtiff, libpng, PySide6, PyYAML, getopt_port)의 라이선스 고지는 [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)에 포함되어 있습니다.
