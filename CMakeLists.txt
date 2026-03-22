cmake_minimum_required(VERSION 3.18)
project(dso_stacker LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDAToolkit_ROOT "/usr/local/cuda")
find_package(CUDAToolkit 12 REQUIRED)

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "/home/donut/.local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(CFITSIO REQUIRED cfitsio)
pkg_check_modules(LIBTIFF REQUIRED libtiff-4)
pkg_check_modules(LIBPNG  REQUIRED libpng)

set(LIB_SOURCES
    src/fits_io.c
    src/image_io.c
    src/csv_parser.c
    src/lanczos_cpu.c
    src/integration.c
    src/debayer_cpu.c
    src/star_detect_cpu.c
    src/ransac.c
    src/calibration.c
    src/pipeline_cpu.c
    src/lanczos_gpu.cu
    src/debayer_gpu.cu
    src/star_detect_gpu.cu
    src/integration_gpu.cu
    src/calibration_gpu.cu
    src/pipeline.cu
)

add_library(dso_stacker_lib STATIC ${LIB_SOURCES})
target_include_directories(dso_stacker_lib PUBLIC
    include
    ${CFITSIO_INCLUDE_DIRS}
    ${LIBTIFF_INCLUDE_DIRS}
    ${LIBPNG_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_directories(dso_stacker_lib PUBLIC
    ${CFITSIO_LIBRARY_DIRS}
    ${LIBTIFF_LIBRARY_DIRS}
    ${LIBPNG_LIBRARY_DIRS}
)
target_link_libraries(dso_stacker_lib PUBLIC
    ${CFITSIO_LIBRARIES}
    ${LIBTIFF_LIBRARIES}
    ${LIBPNG_LIBRARIES}
    CUDA::cudart
    CUDA::nppc
    CUDA::nppig
    OpenMP::OpenMP_C
)
set_target_properties(dso_stacker_lib PROPERTIES
    CUDA_ARCHITECTURES "86;89"
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(dso_stacker main.cpp)
target_link_libraries(dso_stacker PRIVATE dso_stacker_lib)

# ---- Tests ----------------------------------------------------------------
enable_testing()

# CPU tests (C11): csv_parser, fits_io, integration, lanczos_cpu
add_executable(test_cpu tests/test_cpu.c)
target_include_directories(test_cpu PRIVATE tests)
target_link_libraries(test_cpu PRIVATE dso_stacker_lib m)
add_test(NAME test_cpu COMMAND test_cpu)

# GPU tests (CUDA): lanczos_gpu init/transform vs CPU baseline
add_executable(test_gpu tests/test_gpu.cu)
set_target_properties(test_gpu PROPERTIES
    CUDA_ARCHITECTURES "86;89"
    CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_gpu PRIVATE tests)
target_link_libraries(test_gpu PRIVATE dso_stacker_lib)
add_test(NAME test_gpu COMMAND test_gpu)
# Exit code 77 = SKIP (no GPU available)
set_property(TEST test_gpu PROPERTY SKIP_RETURN_CODE 77)

# CPU tests for star detection (CCL, CoM) and RANSAC homography
add_executable(test_star_detect tests/test_star_detect.c)
target_include_directories(test_star_detect PRIVATE tests)
target_link_libraries(test_star_detect PRIVATE dso_stacker_lib m)
add_test(NAME test_star_detect COMMAND test_star_detect)

add_executable(test_ransac tests/test_ransac.c)
target_include_directories(test_ransac PRIVATE tests)
target_link_libraries(test_ransac PRIVATE dso_stacker_lib m)
add_test(NAME test_ransac COMMAND test_ransac)

# CPU debayer tests
add_executable(test_debayer_cpu tests/test_debayer_cpu.c)
target_include_directories(test_debayer_cpu PRIVATE tests)
target_link_libraries(test_debayer_cpu PRIVATE dso_stacker_lib m)
add_test(NAME test_debayer_cpu COMMAND test_debayer_cpu)

# CPU tests for calibration (dark/flat/bias/darkflat preprocessing)
add_executable(test_calibration tests/test_calibration.c)
target_include_directories(test_calibration PRIVATE tests)
target_link_libraries(test_calibration PRIVATE dso_stacker_lib m)
add_test(NAME test_calibration COMMAND test_calibration)

# Audit/Fixes verification tests
add_executable(test_audit tests/test_audit.c)
target_include_directories(test_audit PRIVATE tests)
target_link_libraries(test_audit PRIVATE dso_stacker_lib m)
add_test(NAME test_audit COMMAND test_audit)

# OSC (color output) tests: debayer_cpu_rgb, fits_save_rgb, fits_get_bayer_pattern
add_executable(test_color tests/test_color.c)
target_include_directories(test_color PRIVATE tests)
target_link_libraries(test_color PRIVATE dso_stacker_lib m)
add_test(NAME test_color COMMAND test_color)

# GPU tests for mini-batch kappa-sigma integration
add_executable(test_integration_gpu tests/test_integration_gpu.cu)
set_target_properties(test_integration_gpu PROPERTIES
    CUDA_ARCHITECTURES "86;89"
    CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_integration_gpu PRIVATE tests)
target_link_libraries(test_integration_gpu PRIVATE dso_stacker_lib)
add_test(NAME test_integration_gpu COMMAND test_integration_gpu)
set_property(TEST test_integration_gpu PROPERTY SKIP_RETURN_CODE 77)

# image_io tests: TIFF and PNG save/load round-trips for all bit depths and compressions
add_executable(test_image_io tests/test_image_io.c)
target_include_directories(test_image_io PRIVATE tests)
target_link_libraries(test_image_io PRIVATE dso_stacker_lib m)
add_test(NAME test_image_io COMMAND test_image_io)
