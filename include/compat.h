/*
 * compat.h — Cross-platform compatibility layer.
 *
 * Provides POSIX API shims for MSVC builds. On GCC/Clang this header is
 * effectively empty. Include it in any source file that uses:
 *   - getopt_long  (main.cpp)
 *   - strtok_r     (csv_parser.c)
 *   - rand_r       (ransac.c)
 *   - mkdir        (calibration.c)
 *   - usleep       (test_audit.c)
 *
 * Also provides an OpenMP collapse(2) macro that degrades gracefully on
 * MSVC (which only supports OpenMP 2.0 — no collapse clause).
 */

#pragma once

/* ------------------------------------------------------------------ */
/*  OpenMP collapse(2) — safe on all compilers                        */
/* ------------------------------------------------------------------ */
/*
 * MSVC's OpenMP is version 2.0 (200203); collapse requires 3.0 (200805).
 * When collapse is unavailable, only the outer loop is parallelized —
 * still effective for image-processing row loops (thousands of iterations).
 */
#if defined(_OPENMP) && _OPENMP >= 200805
  #define OMP_PARALLEL_FOR_COLLAPSE2 \
      _Pragma("omp parallel for collapse(2) schedule(static)")
#else
  #define OMP_PARALLEL_FOR_COLLAPSE2 \
      _Pragma("omp parallel for schedule(static)")
#endif

/* ------------------------------------------------------------------ */
/*  MSVC-specific POSIX shims                                         */
/* ------------------------------------------------------------------ */
#ifdef _MSC_VER

/* getopt_long: bundled BSD-licensed implementation */
#include "getopt_port.h"

/*
 * strtok_r → strtok_s
 * MSVC's strtok_s has the same signature as POSIX strtok_r:
 *   char *strtok_s(char *str, const char *delim, char **context)
 */
#define strtok_r(str, delim, saveptr)  strtok_s(str, delim, saveptr)

/*
 * rand_r — thread-safe PRNG (POSIX, absent on MSVC).
 * Simple LCG matching glibc's rand_r constants.
 */
static __inline int compat_rand_r(unsigned int *seed)
{
    *seed = *seed * 1103515245u + 12345u;
    return (int)((*seed >> 16) & 0x7fff);
}
#define rand_r(seed) compat_rand_r(seed)

/*
 * mkdir — POSIX mkdir(path, mode) vs MSVC _mkdir(path).
 * The mode argument is silently ignored on Windows.
 */
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)

/*
 * usleep — POSIX microsecond sleep vs Windows Sleep(ms).
 */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static __inline void compat_usleep(unsigned int us)
{
    Sleep(us / 1000u);  /* convert microseconds to milliseconds */
}
#define usleep(us) compat_usleep(us)

/*
 * strcasecmp → _stricmp (case-insensitive string comparison).
 */
#include <string.h>
#define strcasecmp(a, b) _stricmp(a, b)

#endif /* _MSC_VER */
