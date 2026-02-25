/*
 * test_framework.h — minimal single-header test runner
 *
 * Usage:
 *   #include "test_framework.h"
 *   static int test_something(void) {
 *       ASSERT(1 + 1 == 2);
 *       ASSERT_NEAR(0.1f + 0.2f, 0.3f, 1e-5f);
 *       return 0;   // 0 = pass
 *   }
 *   int main(void) {
 *       SUITE("My Suite");
 *       RUN(test_something);
 *       return SUMMARY();
 *   }
 *
 * ASSERT macros return 1 (fail) immediately from the test function on failure.
 * RUN() calls the function and records pass/fail.
 * SUMMARY() prints totals and returns 0 if all passed, 1 otherwise.
 */

#pragma once
#include <stdio.h>
#include <math.h>

static int _g_total   = 0;
static int _g_passed  = 0;
static int _g_failed  = 0;

#define SUITE(name) \
    printf("\n=== %s ===\n", name)

#define RUN(fn) do { \
    _g_total++; \
    printf("  %-56s", #fn); \
    fflush(stdout); \
    int _rc = fn(); \
    if (_rc == 0) { printf("PASS\n"); _g_passed++; } \
    else          { printf("FAIL\n"); _g_failed++; } \
} while(0)

#define SUMMARY() ({ \
    printf("\n--- %d passed, %d failed, %d total ---\n", \
           _g_passed, _g_failed, _g_total); \
    (_g_failed > 0) ? 1 : 0; \
})

/* All ASSERT macros immediately return 1 on failure, printing the location. */
#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("\n    ASSERT FAILED: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

#define ASSERT_EQ(a, b)      ASSERT((a) == (b))
#define ASSERT_NE(a, b)      ASSERT((a) != (b))
#define ASSERT_LT(a, b)      ASSERT((a) <  (b))
#define ASSERT_GT(a, b)      ASSERT((a) >  (b))
#define ASSERT_NULL(p)       ASSERT((p) == NULL)
#define ASSERT_NOT_NULL(p)   ASSERT((p) != NULL)

/* Floating-point approximate equality */
#define ASSERT_NEAR(a, b, tol) \
    ASSERT(fabsf((float)(a) - (float)(b)) <= (float)(tol))

/* Convenience: check DsoError return value */
#define ASSERT_OK(call)  ASSERT((call) == DSO_OK)
#define ASSERT_ERR(call, expected_err) ASSERT((call) == (expected_err))
