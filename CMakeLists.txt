cmake_minimum_required(VERSION 3.18)

option(DSO_ENABLE_CUDA "Enable CUDA backend" ON)
option(DSO_ENABLE_METAL "Enable Metal backend (macOS only)" OFF)
option(DSO_ENABLE_LIBRAW "Enable RAW camera file support via LibRaw" OFF)

set(DSO_PROJECT_LANGUAGES C CXX)
if(DSO_ENABLE_CUDA)
    list(APPEND DSO_PROJECT_LANGUAGES CUDA)
endif()
if(DSO_ENABLE_METAL)
    list(APPEND DSO_PROJECT_LANGUAGES OBJCXX)
endif()

project(dso_stacker LANGUAGES ${DSO_PROJECT_LANGUAGES})

if(DSO_ENABLE_CUDA)
    add_compile_definitions(DSO_HAS_CUDA=1)
else()
    add_compile_definitions(DSO_HAS_CUDA=0)
endif()
if(DSO_ENABLE_METAL)
    add_compile_definitions(DSO_HAS_METAL=1)
else()
    add_compile_definitions(DSO_HAS_METAL=0)
endif()
if(DSO_ENABLE_LIBRAW)
    add_compile_definitions(DSO_HAS_LIBRAW=1)
else()
    add_compile_definitions(DSO_HAS_LIBRAW=0)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
if(DSO_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# ---- Configurable CUDA architectures (override with -DDSO_CUDA_ARCHITECTURES=) ----
if(DSO_ENABLE_CUDA)
    set(DSO_CUDA_ARCHITECTURES "86;89" CACHE STRING "CUDA architectures to compile for")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
endif()

# ---- CUDA Toolkit ----
if(DSO_ENABLE_CUDA)
    if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "/usr/local/cuda")
        set(CUDAToolkit_ROOT "/usr/local/cuda")
    endif()
    find_package(CUDAToolkit 12 REQUIRED)
endif()

# ---- OpenMP (C and CXX only; NVCC does not support OpenMP) ----
find_package(OpenMP COMPONENTS C CXX REQUIRED)

# ---- Math library (MSVC links math automatically) ----
if(NOT MSVC)
    set(MATH_LIB m)
else()
    set(MATH_LIB "")
endif()

# ---- External libraries (platform-dependent discovery) ----
if(WIN32)
    # On Windows, use find_package (vcpkg provides Config-mode packages).
    # vcpkg's cfitsio config uses lowercase package name.
    find_package(cfitsio CONFIG REQUIRED)
    find_package(TIFF REQUIRED)
    find_package(PNG REQUIRED)

    if(DSO_ENABLE_LIBRAW)
        find_package(libraw CONFIG REQUIRED)
        set(LIBRAW_TARGET libraw::raw)
    endif()

    # Imported target names from vcpkg
    set(CFITSIO_TARGET CFITSIO::cfitsio)
    set(TIFF_TARGET    TIFF::TIFF)
    set(PNG_TARGET     PNG::PNG)
else()
    # On Linux/macOS, use pkg-config.
    find_package(PkgConfig REQUIRED)
    if(DEFINED ENV{CFITSIO_PREFIX})
        set(ENV{PKG_CONFIG_PATH} "$ENV{CFITSIO_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    elseif(EXISTS "/home/donut/.local/lib/pkgconfig")
        set(ENV{PKG_CONFIG_PATH} "/home/donut/.local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    endif()
    pkg_check_modules(CFITSIO REQUIRED cfitsio)
    pkg_check_modules(LIBTIFF REQUIRED libtiff-4)
    pkg_check_modules(LIBPNG  REQUIRED libpng)
    if(DSO_ENABLE_LIBRAW)
        pkg_check_modules(LIBRAW REQUIRED libraw)
    endif()
endif()

# ---- Library sources ----
set(LIB_SOURCES
    src/fits_io.c
    src/image_io.c
    src/csv_parser.c
    src/lanczos_cpu.c
    src/integration.c
    src/debayer_cpu.c
    src/star_detect_cpu.c
    src/frame_quality.c
    src/ransac.c
    src/calibration.c
    src/white_balance.c
    src/background.c
    src/pipeline_dispatch.c
    src/pipeline_cpu.c
    src/frame_load.c
    src/centroid_lm.c
)

if(DSO_ENABLE_LIBRAW)
    list(APPEND LIB_SOURCES src/raw_io.c)
endif()

if(DSO_ENABLE_CUDA)
    list(APPEND LIB_SOURCES
        src/lanczos_gpu.cu
        src/debayer_gpu.cu
        src/star_detect_gpu.cu
        src/integration_gpu.cu
        src/ransac_gpu.cu
        src/calibration_gpu.cu
        src/white_balance_gpu.cu
        src/background_gpu.cu
        src/pipeline.cu
        src/centroid_lm_gpu.cu
    )
else()
    list(APPEND LIB_SOURCES src/pipeline_cuda_stub.c)
endif()

if(DSO_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "DSO_ENABLE_METAL is only supported on Apple platforms")
    endif()
    list(APPEND LIB_SOURCES src/pipeline_metal.mm)
else()
    list(APPEND LIB_SOURCES src/pipeline_metal_stub.c)
endif()

# Bundle portable getopt on MSVC (no system <getopt.h>)
if(MSVC)
    list(APPEND LIB_SOURCES src/getopt_port.c)
endif()

add_library(dso_stacker_lib STATIC ${LIB_SOURCES})

if(WIN32)
    # vcpkg imported targets
    target_include_directories(dso_stacker_lib PUBLIC
        include
    )
    if(DSO_ENABLE_CUDA)
        target_include_directories(dso_stacker_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    set(_DSO_WIN_LIBS
        ${CFITSIO_TARGET}
        ${TIFF_TARGET}
        ${PNG_TARGET}
        OpenMP::OpenMP_C
    )
    if(DSO_ENABLE_LIBRAW)
        list(APPEND _DSO_WIN_LIBS ${LIBRAW_TARGET})
    endif()
    target_link_libraries(dso_stacker_lib PUBLIC ${_DSO_WIN_LIBS})
    if(DSO_ENABLE_CUDA)
        target_link_libraries(dso_stacker_lib PUBLIC
            CUDA::cudart
            CUDA::nppc
            CUDA::nppig
        )
    endif()
else()
    # pkg-config variables
    set(_DSO_PKG_INCLUDE_DIRS
        ${CFITSIO_INCLUDE_DIRS}
        ${LIBTIFF_INCLUDE_DIRS}
        ${LIBPNG_INCLUDE_DIRS}
    )
    set(_DSO_PKG_LIBRARY_DIRS
        ${CFITSIO_LIBRARY_DIRS}
        ${LIBTIFF_LIBRARY_DIRS}
        ${LIBPNG_LIBRARY_DIRS}
    )
    set(_DSO_PKG_LIBRARIES
        ${CFITSIO_LIBRARIES}
        ${LIBTIFF_LIBRARIES}
        ${LIBPNG_LIBRARIES}
    )
    if(DSO_ENABLE_LIBRAW)
        list(APPEND _DSO_PKG_INCLUDE_DIRS ${LIBRAW_INCLUDE_DIRS})
        list(APPEND _DSO_PKG_LIBRARY_DIRS ${LIBRAW_LIBRARY_DIRS})
        list(APPEND _DSO_PKG_LIBRARIES    ${LIBRAW_LIBRARIES})
    endif()
    target_include_directories(dso_stacker_lib PUBLIC
        include
        ${_DSO_PKG_INCLUDE_DIRS}
    )
    if(DSO_ENABLE_CUDA)
        target_include_directories(dso_stacker_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    target_link_directories(dso_stacker_lib PUBLIC
        ${_DSO_PKG_LIBRARY_DIRS}
    )
    target_link_libraries(dso_stacker_lib PUBLIC
        ${_DSO_PKG_LIBRARIES}
        OpenMP::OpenMP_C
    )
    if(DSO_ENABLE_CUDA)
        target_link_libraries(dso_stacker_lib PUBLIC
            CUDA::cudart
            CUDA::nppc
            CUDA::nppig
        )
    endif()
endif()

if(DSO_ENABLE_CUDA)
    set_target_properties(dso_stacker_lib PROPERTIES
        CUDA_ARCHITECTURES "${DSO_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

add_executable(dso_stacker main.cpp)
target_link_libraries(dso_stacker PRIVATE dso_stacker_lib)

# Star detection visualization helper (CLI)
if(DSO_ENABLE_CUDA)
    add_executable(star_detect_overlay test/star_detect_overlay.cpp)
    target_link_libraries(star_detect_overlay PRIVATE dso_stacker_lib ${MATH_LIB})
endif()

# RANSAC status diagnostic helper (CLI)
add_executable(ransac_status test/ransac_status.cpp)
target_link_libraries(ransac_status PRIVATE dso_stacker_lib ${MATH_LIB})

# ---- Install rules ----
install(TARGETS dso_stacker RUNTIME DESTINATION bin)

# ---- Tests ----------------------------------------------------------------
enable_testing()

# CPU tests (C11): csv_parser, fits_io, integration, lanczos_cpu
add_executable(test_cpu tests/test_cpu.c)
target_include_directories(test_cpu PRIVATE tests)
target_link_libraries(test_cpu PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_cpu COMMAND test_cpu)

if(DSO_ENABLE_CUDA)
    # GPU tests (CUDA): lanczos_gpu init/transform vs CPU baseline
    add_executable(test_gpu tests/test_gpu.cu)
    set_target_properties(test_gpu PROPERTIES
        CUDA_ARCHITECTURES "${DSO_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(test_gpu PRIVATE tests)
    target_link_libraries(test_gpu PRIVATE dso_stacker_lib)
    add_test(NAME test_gpu COMMAND test_gpu)
    # Exit code 77 = SKIP (no GPU available)
    set_property(TEST test_gpu PROPERTY SKIP_RETURN_CODE 77)
endif()

# CPU tests for star detection (CCL, CoM) and RANSAC homography
add_executable(test_star_detect tests/test_star_detect.c)
target_include_directories(test_star_detect PRIVATE tests)
target_link_libraries(test_star_detect PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_star_detect COMMAND test_star_detect)

add_executable(test_centroid_lm tests/test_centroid_lm.c)
target_include_directories(test_centroid_lm PRIVATE tests)
target_link_libraries(test_centroid_lm PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_centroid_lm COMMAND test_centroid_lm)

add_executable(test_frame_quality tests/test_frame_quality.c)
target_include_directories(test_frame_quality PRIVATE tests)
target_link_libraries(test_frame_quality PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_frame_quality COMMAND test_frame_quality)

add_executable(test_ransac tests/test_ransac.c test/star_coords_generator.c)
target_include_directories(test_ransac PRIVATE tests test)
target_link_libraries(test_ransac PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_ransac COMMAND test_ransac)

# CPU debayer tests
add_executable(test_debayer_cpu tests/test_debayer_cpu.c)
target_include_directories(test_debayer_cpu PRIVATE tests)
target_link_libraries(test_debayer_cpu PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_debayer_cpu COMMAND test_debayer_cpu)

# CPU tests for calibration (dark/flat/bias/darkflat preprocessing)
add_executable(test_calibration tests/test_calibration.c)
target_include_directories(test_calibration PRIVATE tests)
target_link_libraries(test_calibration PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_calibration COMMAND test_calibration)

# Audit/Fixes verification tests
add_executable(test_audit tests/test_audit.c)
target_include_directories(test_audit PRIVATE tests)
target_link_libraries(test_audit PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_audit COMMAND test_audit)

# OSC (color output) tests: debayer_cpu_rgb, fits_save_rgb, fits_get_bayer_pattern
add_executable(test_color tests/test_color.c)
target_include_directories(test_color PRIVATE tests)
target_link_libraries(test_color PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_color COMMAND test_color)

# White balance tests: wb_apply_bayer, wb_auto_compute
add_executable(test_white_balance tests/test_white_balance.c)
target_include_directories(test_white_balance PRIVATE tests)
target_link_libraries(test_white_balance PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_white_balance COMMAND test_white_balance)

if(DSO_ENABLE_CUDA)
    # GPU tests for mini-batch kappa-sigma integration
    add_executable(test_integration_gpu tests/test_integration_gpu.cu)
    set_target_properties(test_integration_gpu PROPERTIES
        CUDA_ARCHITECTURES "${DSO_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(test_integration_gpu PRIVATE tests)
    target_link_libraries(test_integration_gpu PRIVATE dso_stacker_lib)
    add_test(NAME test_integration_gpu COMMAND test_integration_gpu)
    set_property(TEST test_integration_gpu PROPERTY SKIP_RETURN_CODE 77)
endif()

# image_io tests: TIFF and PNG save/load round-trips for all bit depths and compressions
add_executable(test_image_io tests/test_image_io.c)
target_include_directories(test_image_io PRIVATE tests)
target_link_libraries(test_image_io PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_image_io COMMAND test_image_io)

# backend-dispatch tests
add_executable(test_pipeline_backend tests/test_pipeline_backend.c)
target_include_directories(test_pipeline_backend PRIVATE tests)
target_link_libraries(test_pipeline_backend PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_pipeline_backend COMMAND test_pipeline_backend)

# frame_load dispatch + raw_io tests
add_executable(test_raw_io tests/test_raw_io.c)
target_include_directories(test_raw_io PRIVATE tests)
target_link_libraries(test_raw_io PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_raw_io COMMAND test_raw_io)

# End-to-end CPU pipeline tests
add_executable(test_pipeline_cpu tests/test_pipeline_cpu.c)
target_include_directories(test_pipeline_cpu PRIVATE tests)
target_link_libraries(test_pipeline_cpu PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_pipeline_cpu COMMAND test_pipeline_cpu)

# Cross-stage integration tests
add_executable(test_cross_stage tests/test_cross_stage.c test/star_coords_generator.c)
target_include_directories(test_cross_stage PRIVATE tests test)
target_link_libraries(test_cross_stage PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_cross_stage COMMAND test_cross_stage)

# Numerical edge case tests (NaN/Inf/precision)
add_executable(test_numerical tests/test_numerical.c)
target_include_directories(test_numerical PRIVATE tests)
target_link_libraries(test_numerical PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_numerical COMMAND test_numerical)

# Background normalization tests
add_executable(test_background tests/test_background.c)
target_include_directories(test_background PRIVATE tests)
target_link_libraries(test_background PRIVATE dso_stacker_lib ${MATH_LIB})
add_test(NAME test_background COMMAND test_background)
