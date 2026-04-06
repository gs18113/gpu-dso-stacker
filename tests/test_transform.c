/*
 * test_transform.c — Unit tests for polynomial alignment transforms.
 *
 * Tests cover:
 *   - transform_eval: identity, bilinear, bisquared, bicubic evaluation
 *   - transform_fit: recover known transforms from correspondences
 *   - transform_auto_select: threshold-based model selection
 *   - transform_identity: verify identity coefficients
 *   - transform_reproj_err_sq: reprojection error
 *   - transform_fit edge cases: insufficient points, degenerate input
 */

#include "test_framework.h"
#include "transform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Generate N StarPos points on a grid in [100, 900] × [100, 900] */
static void make_grid_pts(StarPos *pts, int n, int seed)
{
    int side = (int)ceil(sqrt((double)n));
    int idx = 0;
    for (int i = 0; i < side && idx < n; i++) {
        for (int j = 0; j < side && idx < n; j++) {
            pts[idx].x = 100.0f + 800.0f * i / (side - 1);
            pts[idx].y = 100.0f + 800.0f * j / (side - 1);
            pts[idx].flux = 1.0f;
            idx++;
        }
    }
    (void)seed;
}

/* Apply a known bilinear transform to generate source points */
static void apply_bilinear(const double *c, const StarPos *ref,
                            StarPos *src, int n)
{
    for (int i = 0; i < n; i++) {
        double dx = (double)ref[i].x;
        double dy = (double)ref[i].y;
        src[i].x = (float)(c[0] + c[1]*dx + c[2]*dy);
        src[i].y = (float)(c[3] + c[4]*dx + c[5]*dy);
        src[i].flux = ref[i].flux;
    }
}

/* Apply a known bisquared transform */
static void apply_bisquared(const double *c, const StarPos *ref,
                             StarPos *src, int n)
{
    for (int i = 0; i < n; i++) {
        double dx = (double)ref[i].x;
        double dy = (double)ref[i].y;
        double dx2 = dx*dx, dxy = dx*dy, dy2 = dy*dy;
        src[i].x = (float)(c[0] + c[1]*dx + c[2]*dy + c[3]*dx2 + c[4]*dxy + c[5]*dy2);
        src[i].y = (float)(c[6] + c[7]*dx + c[8]*dy + c[9]*dx2 + c[10]*dxy + c[11]*dy2);
        src[i].flux = ref[i].flux;
    }
}

/* Apply a known bicubic transform */
static void apply_bicubic(const double *c, const StarPos *ref,
                           StarPos *src, int n)
{
    for (int i = 0; i < n; i++) {
        double dx = (double)ref[i].x;
        double dy = (double)ref[i].y;
        double dx2 = dx*dx, dy2 = dy*dy, dxy = dx*dy;
        double dx3 = dx2*dx, dy3 = dy2*dy;
        src[i].x = (float)(c[0] + c[1]*dx + c[2]*dy + c[3]*dx2 + c[4]*dxy + c[5]*dy2
                         + c[6]*dx3 + c[7]*dx2*dy + c[8]*dx*dy2 + c[9]*dy3);
        src[i].y = (float)(c[10] + c[11]*dx + c[12]*dy + c[13]*dx2 + c[14]*dxy + c[15]*dy2
                         + c[16]*dx3 + c[17]*dx2*dy + c[18]*dx*dy2 + c[19]*dy3);
        src[i].flux = ref[i].flux;
    }
}

/* -------------------------------------------------------------------------
 * transform_eval tests
 * ------------------------------------------------------------------------- */

static int test_eval_identity_bilinear(void)
{
    PolyTransform T;
    transform_identity(TRANSFORM_BILINEAR, &T);
    double sx, sy;
    transform_eval(&T, 123.0, 456.0, &sx, &sy);
    ASSERT(fabs(sx - 123.0) < 1e-10);
    ASSERT(fabs(sy - 456.0) < 1e-10);
    return 0;
}

static int test_eval_identity_bisquared(void)
{
    PolyTransform T;
    transform_identity(TRANSFORM_BISQUARED, &T);
    double sx, sy;
    transform_eval(&T, 500.0, 300.0, &sx, &sy);
    ASSERT(fabs(sx - 500.0) < 1e-10);
    ASSERT(fabs(sy - 300.0) < 1e-10);
    return 0;
}

static int test_eval_identity_bicubic(void)
{
    PolyTransform T;
    transform_identity(TRANSFORM_BICUBIC, &T);
    double sx, sy;
    transform_eval(&T, 1000.0, 2000.0, &sx, &sy);
    ASSERT(fabs(sx - 1000.0) < 1e-10);
    ASSERT(fabs(sy - 2000.0) < 1e-10);
    return 0;
}

static int test_eval_bilinear_translation(void)
{
    PolyTransform T;
    memset(&T, 0, sizeof(T));
    T.model = TRANSFORM_BILINEAR;
    /* sx = 10 + 1*dx + 0*dy,  sy = 20 + 0*dx + 1*dy */
    T.coeffs[0] = 10.0; T.coeffs[1] = 1.0; T.coeffs[2] = 0.0;
    T.coeffs[3] = 20.0; T.coeffs[4] = 0.0; T.coeffs[5] = 1.0;

    double sx, sy;
    transform_eval(&T, 100.0, 200.0, &sx, &sy);
    ASSERT(fabs(sx - 110.0) < 1e-10);
    ASSERT(fabs(sy - 220.0) < 1e-10);
    return 0;
}

static int test_eval_bisquared_quadratic(void)
{
    PolyTransform T;
    memset(&T, 0, sizeof(T));
    T.model = TRANSFORM_BISQUARED;
    /* sx = 1*dx + 0.001*dx^2,  sy = 1*dy + 0.001*dy^2 */
    T.coeffs[1] = 1.0;  T.coeffs[3] = 0.001;
    T.coeffs[7] = 0.0;  T.coeffs[8] = 1.0; T.coeffs[11] = 0.001;

    double sx, sy;
    transform_eval(&T, 100.0, 200.0, &sx, &sy);
    ASSERT(fabs(sx - (100.0 + 10.0)) < 1e-6);
    ASSERT(fabs(sy - (200.0 + 40.0)) < 1e-6);
    return 0;
}

/* -------------------------------------------------------------------------
 * transform_auto_select tests
 * ------------------------------------------------------------------------- */

static int test_auto_select_thresholds(void)
{
    ASSERT_EQ(transform_auto_select(25), TRANSFORM_BICUBIC);
    ASSERT_EQ(transform_auto_select(20), TRANSFORM_BICUBIC);
    ASSERT_EQ(transform_auto_select(19), TRANSFORM_BISQUARED);
    ASSERT_EQ(transform_auto_select(12), TRANSFORM_BISQUARED);
    ASSERT_EQ(transform_auto_select(11), TRANSFORM_BILINEAR);
    ASSERT_EQ(transform_auto_select(6), TRANSFORM_BILINEAR);
    ASSERT_EQ(transform_auto_select(5), TRANSFORM_PROJECTIVE);
    ASSERT_EQ(transform_auto_select(0), TRANSFORM_PROJECTIVE);
    return 0;
}

/* -------------------------------------------------------------------------
 * transform_fit tests
 * ------------------------------------------------------------------------- */

static int test_fit_bilinear_identity(void)
{
    int n = 10;
    StarPos ref[10], src[10];
    make_grid_pts(ref, n, 42);

    /* Identity: src = ref */
    double id_coeffs[6] = {0, 1, 0,  0, 0, 1};
    apply_bilinear(id_coeffs, ref, src, n);

    PolyTransform T;
    DsoError err = transform_fit(ref, src, n, TRANSFORM_BILINEAR, &T);
    ASSERT_EQ(err, DSO_OK);
    ASSERT_EQ(T.model, TRANSFORM_BILINEAR);

    /* Verify: evaluate at all ref points, should match src */
    for (int i = 0; i < n; i++) {
        double sx, sy;
        transform_eval(&T, (double)ref[i].x, (double)ref[i].y, &sx, &sy);
        ASSERT(fabs(sx - (double)src[i].x) < 0.1);
        ASSERT(fabs(sy - (double)src[i].y) < 0.1);
    }
    return 0;
}

static int test_fit_bilinear_translation(void)
{
    int n = 10;
    StarPos ref[10], src[10];
    make_grid_pts(ref, n, 42);

    double c[6] = {5.5, 1, 0,  -3.2, 0, 1};  /* translation: (5.5, -3.2) */
    apply_bilinear(c, ref, src, n);

    PolyTransform T;
    DsoError err = transform_fit(ref, src, n, TRANSFORM_BILINEAR, &T);
    ASSERT_EQ(err, DSO_OK);

    for (int i = 0; i < n; i++) {
        double sx, sy;
        transform_eval(&T, (double)ref[i].x, (double)ref[i].y, &sx, &sy);
        ASSERT(fabs(sx - (double)src[i].x) < 0.1);
        ASSERT(fabs(sy - (double)src[i].y) < 0.1);
    }
    return 0;
}

static int test_fit_bilinear_rotation_scale(void)
{
    int n = 10;
    StarPos ref[10], src[10];
    make_grid_pts(ref, n, 42);

    /* Rotation by 5° + scale 1.02 + translation (3, -2) */
    double angle = 5.0 * 3.14159265 / 180.0;
    double s = 1.02;
    double ca = s * cos(angle), sa = s * sin(angle);
    double c[6] = {3.0, ca, -sa,  -2.0, sa, ca};
    apply_bilinear(c, ref, src, n);

    PolyTransform T;
    DsoError err = transform_fit(ref, src, n, TRANSFORM_BILINEAR, &T);
    ASSERT_EQ(err, DSO_OK);

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double sx, sy;
        transform_eval(&T, (double)ref[i].x, (double)ref[i].y, &sx, &sy);
        double ex = fabs(sx - (double)src[i].x);
        double ey = fabs(sy - (double)src[i].y);
        if (ex > max_err) max_err = ex;
        if (ey > max_err) max_err = ey;
    }
    ASSERT(max_err < 0.5);
    return 0;
}

static int test_fit_bisquared(void)
{
    int n = 25;
    StarPos ref[25], src[25];
    make_grid_pts(ref, n, 42);

    /* Quadratic distortion: small barrel effect */
    double c[12] = {
        2.0, 1.001, -0.0001, 1e-7, 5e-8, -3e-8,   /* sx */
        -1.5, 0.0002, 0.999,  -2e-8, 1e-7, 8e-8    /* sy */
    };
    apply_bisquared(c, ref, src, n);

    PolyTransform T;
    DsoError err = transform_fit(ref, src, n, TRANSFORM_BISQUARED, &T);
    ASSERT_EQ(err, DSO_OK);
    ASSERT_EQ(T.model, TRANSFORM_BISQUARED);

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double sx, sy;
        transform_eval(&T, (double)ref[i].x, (double)ref[i].y, &sx, &sy);
        double ex = fabs(sx - (double)src[i].x);
        double ey = fabs(sy - (double)src[i].y);
        if (ex > max_err) max_err = ex;
        if (ey > max_err) max_err = ey;
    }
    ASSERT(max_err < 1.0);
    return 0;
}

static int test_fit_bicubic(void)
{
    int n = 36;
    StarPos ref[36], src[36];
    make_grid_pts(ref, n, 42);

    /* Cubic distortion */
    double c[20] = {
        1.0, 1.0, 0.0, 1e-7, 0.0, -5e-8,
        1e-11, 0.0, 0.0, -1e-11,                     /* sx: a0..a9 */
        -0.5, 0.0, 1.0, 0.0, 1e-7, 3e-8,
        0.0, -1e-11, 0.0, 1e-11                       /* sy: b0..b9 */
    };
    apply_bicubic(c, ref, src, n);

    PolyTransform T;
    DsoError err = transform_fit(ref, src, n, TRANSFORM_BICUBIC, &T);
    ASSERT_EQ(err, DSO_OK);
    ASSERT_EQ(T.model, TRANSFORM_BICUBIC);

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double sx, sy;
        transform_eval(&T, (double)ref[i].x, (double)ref[i].y, &sx, &sy);
        double ex = fabs(sx - (double)src[i].x);
        double ey = fabs(sy - (double)src[i].y);
        if (ex > max_err) max_err = ex;
        if (ey > max_err) max_err = ey;
    }
    ASSERT(max_err < 1.0);
    return 0;
}

/* -------------------------------------------------------------------------
 * transform_fit edge cases
 * ------------------------------------------------------------------------- */

static int test_fit_insufficient_points(void)
{
    StarPos ref[2] = {{100,100,1}, {200,200,1}};
    StarPos src[2] = {{101,101,1}, {201,201,1}};
    PolyTransform T;

    DsoError err = transform_fit(ref, src, 2, TRANSFORM_BILINEAR, &T);
    ASSERT_EQ(err, DSO_ERR_INVALID_ARG);

    err = transform_fit(ref, src, 2, TRANSFORM_BISQUARED, &T);
    ASSERT_EQ(err, DSO_ERR_INVALID_ARG);

    err = transform_fit(ref, src, 2, TRANSFORM_BICUBIC, &T);
    ASSERT_EQ(err, DSO_ERR_INVALID_ARG);

    return 0;
}

static int test_fit_projective_rejected(void)
{
    StarPos ref[4] = {{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
    StarPos src[4] = {{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
    PolyTransform T;

    DsoError err = transform_fit(ref, src, 4, TRANSFORM_PROJECTIVE, &T);
    ASSERT_EQ(err, DSO_ERR_INVALID_ARG);

    err = transform_fit(ref, src, 4, TRANSFORM_AUTO, &T);
    ASSERT_EQ(err, DSO_ERR_INVALID_ARG);

    return 0;
}

/* -------------------------------------------------------------------------
 * transform_reproj_err_sq
 * ------------------------------------------------------------------------- */

static int test_reproj_err_sq(void)
{
    PolyTransform T;
    transform_identity(TRANSFORM_BILINEAR, &T);

    /* Identity: error should be zero */
    double e2 = transform_reproj_err_sq(&T, 100.0f, 200.0f, 100.0f, 200.0f);
    ASSERT(e2 < 1e-10);

    /* With offset: error should be distance squared */
    e2 = transform_reproj_err_sq(&T, 100.0f, 200.0f, 103.0f, 204.0f);
    ASSERT(fabs(e2 - (9.0 + 16.0)) < 1e-6);

    return 0;
}

/* -------------------------------------------------------------------------
 * transform_ncoeffs_per_axis
 * ------------------------------------------------------------------------- */

static int test_ncoeffs(void)
{
    ASSERT_EQ(transform_ncoeffs_per_axis(TRANSFORM_BILINEAR),  3);
    ASSERT_EQ(transform_ncoeffs_per_axis(TRANSFORM_BISQUARED), 6);
    ASSERT_EQ(transform_ncoeffs_per_axis(TRANSFORM_BICUBIC),  10);
    ASSERT_EQ(transform_ncoeffs_per_axis(TRANSFORM_PROJECTIVE), 0);
    return 0;
}

/* -------------------------------------------------------------------------
 * transform_from_homography
 * ------------------------------------------------------------------------- */

static int test_from_homography(void)
{
    Homography H;
    for (int i = 0; i < 9; i++) H.h[i] = (double)i;
    PolyTransform T;
    transform_from_homography(&H, &T);
    ASSERT_EQ(T.model, TRANSFORM_PROJECTIVE);
    return 0;
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    int fail = 0;

    printf("=== test_transform ===\n");

    /* Evaluation */
    RUN(test_eval_identity_bilinear);
    RUN(test_eval_identity_bisquared);
    RUN(test_eval_identity_bicubic);
    RUN(test_eval_bilinear_translation);
    RUN(test_eval_bisquared_quadratic);

    /* Auto selection */
    RUN(test_auto_select_thresholds);

    /* Fitting */
    RUN(test_fit_bilinear_identity);
    RUN(test_fit_bilinear_translation);
    RUN(test_fit_bilinear_rotation_scale);
    RUN(test_fit_bisquared);
    RUN(test_fit_bicubic);

    /* Edge cases */
    RUN(test_fit_insufficient_points);
    RUN(test_fit_projective_rejected);

    /* Reprojection error */
    RUN(test_reproj_err_sq);

    /* Utility */
    RUN(test_ncoeffs);
    RUN(test_from_homography);

    SUMMARY();
}
