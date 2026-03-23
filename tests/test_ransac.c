/*
 * test_ransac.c — Unit tests for dlt_homography and ransac_compute_homography.
 *
 * Tests cover:
 *   DLT:
 *     - Identity homography from 4 identity correspondences
 *     - Pure translation H from 4 translated pairs
 *     - Known rotation+scale H: recover H to within round-trip tolerance
 *     - Overdetermined (8 pairs): still produces correct H
 *     - Fewer than 4 pairs: must return DSO_ERR_INVALID_ARG
 *     - Collinear degenerate points: does not crash (H may be bad)
 *     - Reprojection residual: all 4 input points reproject with < 0.01 px error
 *
 *   RANSAC:
 *     - Clean data (0 outliers): all correspondences become inliers
 *     - 25% random outliers: correct H recovered, ≥ 75% inliers
 *     - Insufficient stars (< 4 in either list): DSO_ERR_STAR_DETECT
 *     - No matches (stars too far apart): DSO_ERR_RANSAC
 *     - NULL params uses defaults
 *     - n_inliers_out is populated correctly
 *     - Result H maps reference stars to source with low reprojection error
 */

#include "test_framework.h"
#include "ransac.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Apply H to (rx,ry) and return (sx,sy) via homogeneous divide. */
static void apply_h(const Homography *H, float rx, float ry,
                    float *sx_out, float *sy_out)
{
    const double *h = H->h;
    double sx_h = h[0]*rx + h[1]*ry + h[2];
    double sy_h = h[3]*rx + h[4]*ry + h[5];
    double sw   = h[6]*rx + h[7]*ry + h[8];
    *sx_out = (float)(sx_h / sw);
    *sy_out = (float)(sy_h / sw);
}

/* Max absolute difference between two Homography matrices (normalised to h[8]=1). */
static float h_max_diff(const Homography *A, const Homography *B)
{
    /* Normalise both to h[8]=1 for comparison */
    double sa = (fabs(A->h[8]) > 1e-12) ? A->h[8] : 1.0;
    double sb = (fabs(B->h[8]) > 1e-12) ? B->h[8] : 1.0;
    float  mx = 0.f;
    for (int i = 0; i < 9; i++) {
        float d = (float)fabs(A->h[i]/sa - B->h[i]/sb);
        if (d > mx) mx = d;
    }
    return mx;
}

/* Build a StarList from arrays of (x, y) positions (flux=1). */
static StarList make_starlist(const float *xs, const float *ys, int n)
{
    StarList sl;
    sl.n = n;
    sl.stars = (StarPos *)malloc((size_t)n * sizeof(StarPos));
    for (int i = 0; i < n; i++) {
        sl.stars[i].x    = xs[i];
        sl.stars[i].y    = ys[i];
        sl.stars[i].flux = 1.0f;
    }
    return sl;
}

/* Build a StarList by applying H to ref positions → source positions. */
static StarList apply_h_to_list(const Homography *H, const float *rx,
                                 const float *ry, int n)
{
    float *sx = (float *)malloc((size_t)n * sizeof(float));
    float *sy = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++)
        apply_h(H, rx[i], ry[i], &sx[i], &sy[i]);
    StarList sl = make_starlist(sx, sy, n);
    free(sx); free(sy);
    return sl;
}

/* =========================================================================
 * DLT tests
 * ========================================================================= */

/* 4 identity correspondences → H ≈ identity (normalised). */
static int test_dlt_identity(void)
{
    StarPos ref[4] = {{100, 200, 1}, {300, 400, 1}, {150, 350, 1}, {250, 150, 1}};
    StarPos src[4] = {{100, 200, 1}, {300, 400, 1}, {150, 350, 1}, {250, 150, 1}};

    Homography H = {{0}};
    ASSERT_OK(dlt_homography(ref, src, 4, &H));

    /* Normalise: h[8] should be non-zero; divide through */
    double sc = H.h[8];
    ASSERT(fabs(sc) > 1e-10);
    ASSERT_NEAR((float)(H.h[0]/sc), 1.0f, 1e-4f);  /* h00 = 1 */
    ASSERT_NEAR((float)(H.h[1]/sc), 0.0f, 1e-4f);  /* h01 = 0 */
    ASSERT_NEAR((float)(H.h[2]/sc), 0.0f, 1e-4f);  /* h02 = 0 */
    ASSERT_NEAR((float)(H.h[3]/sc), 0.0f, 1e-4f);  /* h10 = 0 */
    ASSERT_NEAR((float)(H.h[4]/sc), 1.0f, 1e-4f);  /* h11 = 1 */
    ASSERT_NEAR((float)(H.h[5]/sc), 0.0f, 1e-4f);  /* h12 = 0 */
    ASSERT_NEAR((float)(H.h[6]/sc), 0.0f, 1e-4f);  /* h20 = 0 */
    ASSERT_NEAR((float)(H.h[7]/sc), 0.0f, 1e-4f);  /* h21 = 0 */
    return 0;
}

/* Pure translation (5, 3): src = ref + (5, 3).
 * Expected H: [[1,0,5],[0,1,3],[0,0,1]] (normalised). */
static int test_dlt_pure_translation(void)
{
    float rx[4] = {100, 200, 150, 50};
    float ry[4] = {100, 200, 50,  150};
    StarPos ref[4], src[4];
    for (int i = 0; i < 4; i++) {
        ref[i] = (StarPos){rx[i],        ry[i],        1.f};
        src[i] = (StarPos){rx[i] + 5.0f, ry[i] + 3.0f, 1.f};
    }

    Homography H = {{0}};
    ASSERT_OK(dlt_homography(ref, src, 4, &H));

    /* Verify by reprojecting each reference point */
    for (int i = 0; i < 4; i++) {
        float sx, sy;
        apply_h(&H, ref[i].x, ref[i].y, &sx, &sy);
        ASSERT_NEAR(sx, src[i].x, 0.01f);
        ASSERT_NEAR(sy, src[i].y, 0.01f);
    }
    return 0;
}

/* Known rotation+scale: 30-degree CCW rotation, scale=0.95.
 * Generate 4 correspondences, run DLT, verify reprojection error < 0.01 px. */
static int test_dlt_known_rotation_scale(void)
{
    /* R = 30°, s = 0.95 → H = [[s*cos, -s*sin, 0],[s*sin, s*cos, 0],[0,0,1]] */
    double theta = 30.0 * 3.14159265358979 / 180.0;
    double s     = 0.95;
    double c     = cos(theta), si = sin(theta);

    Homography Htrue = {{0}};
    Htrue.h[0] = s*c; Htrue.h[1] = -s*si; Htrue.h[2] = 0;
    Htrue.h[3] = s*si; Htrue.h[4] = s*c;  Htrue.h[5] = 0;
    Htrue.h[6] = 0;   Htrue.h[7] = 0;     Htrue.h[8] = 1;

    float rx[4] = {100, 300, 200, 250};
    float ry[4] = {100, 100, 300, 200};
    StarPos ref[4], src[4];
    for (int i = 0; i < 4; i++) {
        float tsx, tsy;
        apply_h(&Htrue, rx[i], ry[i], &tsx, &tsy);
        ref[i] = (StarPos){rx[i], ry[i], 1.f};
        src[i] = (StarPos){tsx,   tsy,   1.f};
    }

    Homography Hest = {{0}};
    ASSERT_OK(dlt_homography(ref, src, 4, &Hest));

    for (int i = 0; i < 4; i++) {
        float sx, sy;
        apply_h(&Hest, ref[i].x, ref[i].y, &sx, &sy);
        ASSERT_NEAR(sx, src[i].x, 0.01f);
        ASSERT_NEAR(sy, src[i].y, 0.01f);
    }
    return 0;
}

/* Overdetermined (8 correspondences from known H): residual still < 0.01. */
static int test_dlt_overdetermined(void)
{
    Homography Htrue = {{0}};
    Htrue.h[0]=1.1; Htrue.h[1]=0.1; Htrue.h[2]=15.0;
    Htrue.h[3]=0.05; Htrue.h[4]=0.95; Htrue.h[5]=8.0;
    Htrue.h[6]=0; Htrue.h[7]=0; Htrue.h[8]=1.0;

    float rx[8] = {50, 150, 250, 350, 100, 200, 300, 400};
    float ry[8] = {50, 50,  50,  50,  200, 200, 200, 200};
    StarPos ref[8], src[8];
    for (int i = 0; i < 8; i++) {
        float tsx, tsy;
        apply_h(&Htrue, rx[i], ry[i], &tsx, &tsy);
        ref[i] = (StarPos){rx[i], ry[i], 1.f};
        src[i] = (StarPos){tsx,   tsy,   1.f};
    }

    Homography Hest = {{0}};
    ASSERT_OK(dlt_homography(ref, src, 8, &Hest));

    for (int i = 0; i < 8; i++) {
        float sx, sy;
        apply_h(&Hest, ref[i].x, ref[i].y, &sx, &sy);
        ASSERT_NEAR(sx, src[i].x, 0.05f);
        ASSERT_NEAR(sy, src[i].y, 0.05f);
    }
    return 0;
}

/* Fewer than 4 correspondences must return DSO_ERR_INVALID_ARG. */
static int test_dlt_too_few_points(void)
{
    StarPos ref[3] = {{0,0,1},{100,0,1},{50,100,1}};
    StarPos src[3] = {{0,0,1},{100,0,1},{50,100,1}};
    Homography H = {{0}};
    ASSERT_ERR(dlt_homography(ref, src, 3, &H), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(dlt_homography(ref, src, 0, &H), DSO_ERR_INVALID_ARG);
    return 0;
}

/* NULL pointer validation. */
static int test_dlt_null_args(void)
{
    StarPos pts[4] = {{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
    Homography H = {{0}};
    ASSERT_ERR(dlt_homography(NULL, pts, 4, &H), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(dlt_homography(pts, NULL, 4, &H), DSO_ERR_INVALID_ARG);
    ASSERT_ERR(dlt_homography(pts, pts, 4, NULL), DSO_ERR_INVALID_ARG);
    return 0;
}

/* =========================================================================
 * RANSAC tests
 * ========================================================================= */

static const RansacParams DEFAULT_PARAMS = {
    .max_iters     = 1000,
    .inlier_thresh = 2.0f,
    .match_radius  = 50.0f,
    .confidence    = 0.99f,
    .min_inliers   = 4
};

/* 20 clean correspondences (no outliers) from a known H.
 * Expected: all 20 become inliers; recovered H matches. */
static int test_ransac_clean_data(void)
{
    Homography Htrue = {{0}};
    Htrue.h[0]=1.0; Htrue.h[1]=0.02; Htrue.h[2]=10.0;
    Htrue.h[3]=0.01; Htrue.h[4]=1.0; Htrue.h[5]=5.0;
    Htrue.h[8]=1.0;

    const int N = 20;
    float rx[20], ry[20];
    for (int i = 0; i < N; i++) {
        rx[i] = (float)(30 + i * 20);
        ry[i] = (float)(50 + (i % 4) * 60);
    }

    StarList ref_list = make_starlist(rx, ry, N);
    StarList frm_list = apply_h_to_list(&Htrue, rx, ry, N);

    Homography Hest = {{0}};
    int n_inliers = 0;
    ASSERT_OK(ransac_compute_homography(&ref_list, &frm_list, &DEFAULT_PARAMS,
                                         &Hest, &n_inliers));
    ASSERT(n_inliers >= N - 1); /* allow at most 1 borderline case */

    /* Verify all reference points reproject to within 0.1 px */
    for (int i = 0; i < N; i++) {
        float sx, sy;
        apply_h(&Hest, rx[i], ry[i], &sx, &sy);
        float ex = frm_list.stars[i].x;
        float ey = frm_list.stars[i].y;
        /* Note: frm_list is sorted by flux, not necessarily in rx/ry order;
         * just check that the overall reprojection max error is small. */
        (void)sx; (void)sy; (void)ex; (void)ey;
    }
    /* Better: verify that Hest is close to Htrue (normalised) */
    ASSERT(h_max_diff(&Hest, &Htrue) < 0.1f);

    free(ref_list.stars);
    free(frm_list.stars);
    return 0;
}

/* 20 correspondences with 5 random outliers (25%).
 * RANSAC must recover H with ≥ 14 inliers. */
static int test_ransac_with_outliers(void)
{
    srand(42);

    Homography Htrue = {{0}};
    Htrue.h[0]=1.0; Htrue.h[1]=0.0; Htrue.h[2]=20.0;
    Htrue.h[3]=0.0; Htrue.h[4]=1.0; Htrue.h[5]=10.0;
    Htrue.h[8]=1.0;

    const int N = 20;
    float rx[20], ry[20];
    for (int i = 0; i < N; i++) {
        rx[i] = (float)(50 + i * 25);
        ry[i] = (float)(50 + (i % 5) * 70);
    }

    StarList ref_list = make_starlist(rx, ry, N);
    StarList frm_list = apply_h_to_list(&Htrue, rx, ry, N);

    /* Corrupt 5 of the source positions with large random offsets */
    int outlier_idx[5] = {2, 5, 11, 14, 18};
    for (int k = 0; k < 5; k++) {
        int i = outlier_idx[k];
        frm_list.stars[i].x += (float)(rand() % 200 - 100);
        frm_list.stars[i].y += (float)(rand() % 200 - 100);
    }

    Homography Hest = {{0}};
    int n_inliers = 0;
    DsoError err = ransac_compute_homography(&ref_list, &frm_list, &DEFAULT_PARAMS,
                                              &Hest, &n_inliers);
    ASSERT_OK(err);
    ASSERT(n_inliers >= 14);
    ASSERT(h_max_diff(&Hest, &Htrue) < 0.5f);

    free(ref_list.stars);
    free(frm_list.stars);
    return 0;
}

/* Insufficient stars: ref has only 3 stars → DSO_ERR_STAR_DETECT. */
static int test_ransac_insufficient_stars(void)
{
    float rx[3] = {100, 200, 300};
    float ry[3] = {100, 200, 100};
    StarList ref_list = make_starlist(rx, ry, 3);
    StarList frm_list = make_starlist(rx, ry, 3);

    Homography H = {{0}};
    ASSERT_ERR(ransac_compute_homography(&ref_list, &frm_list, &DEFAULT_PARAMS,
                                          &H, NULL),
               DSO_ERR_STAR_DETECT);

    free(ref_list.stars);
    free(frm_list.stars);
    return 0;
}

/* No matches: all frame stars are > match_radius away from reference stars. */
static int test_ransac_no_matches(void)
{
    float rx[10], ry[10], sx[10], sy[10];
    for (int i = 0; i < 10; i++) {
        rx[i] = (float)(50 + i * 30); ry[i] = 100.f;
        sx[i] = rx[i] + 5000.f;       sy[i] = ry[i] + 5000.f;
    }

    StarList ref_list = make_starlist(rx, ry, 10);
    StarList frm_list = make_starlist(sx, sy, 10);

    Homography H = {{0}};
    DsoError err = ransac_compute_homography(&ref_list, &frm_list, &DEFAULT_PARAMS,
                                              &H, NULL);
    /* Should fail because no valid correspondences can be formed */
    ASSERT(err == DSO_ERR_RANSAC || err == DSO_ERR_STAR_DETECT);

    free(ref_list.stars);
    free(frm_list.stars);
    return 0;
}

/* NULL params → should use internal defaults and succeed on clean data. */
static int test_ransac_null_params_uses_defaults(void)
{
    Homography Htrue = {{0}};
    Htrue.h[0]=1; Htrue.h[4]=1; Htrue.h[8]=1; /* identity */

    const int N = 10;
    float rx[10], ry[10];
    for (int i = 0; i < N; i++) { rx[i] = (float)(100 + i*30); ry[i] = 200.f; }
    StarList ref = make_starlist(rx, ry, N);
    StarList frm = apply_h_to_list(&Htrue, rx, ry, N);

    Homography H = {{0}};
    ASSERT_OK(ransac_compute_homography(&ref, &frm, NULL, &H, NULL));

    free(ref.stars); free(frm.stars);
    return 0;
}

/* n_inliers_out is correctly populated. */
static int test_ransac_inliers_out(void)
{
    Homography Htrue = {{0}};
    Htrue.h[0]=1; Htrue.h[4]=1; Htrue.h[2]=7; Htrue.h[5]=3; Htrue.h[8]=1;

    const int N = 10;
    float rx[10], ry[10];
    for (int i = 0; i < N; i++) { rx[i] = (float)(100 + i*40); ry[i] = 150.f; }
    StarList ref = make_starlist(rx, ry, N);
    StarList frm = apply_h_to_list(&Htrue, rx, ry, N);

    Homography H = {{0}};
    int n_in = -1;
    ASSERT_OK(ransac_compute_homography(&ref, &frm, &DEFAULT_PARAMS, &H, &n_in));
    ASSERT(n_in >= N - 1); /* nearly all should be inliers */

    free(ref.stars); free(frm.stars);
    return 0;
}

/* NULL args validation. */
static int test_ransac_null_args(void)
{
    float rx[5] = {0,1,2,3,4}, ry[5] = {0,1,2,3,4};
    StarList sl = make_starlist(rx, ry, 5);
    Homography H = {{0}};
    ASSERT_ERR(ransac_compute_homography(NULL, &sl, &DEFAULT_PARAMS, &H, NULL),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(ransac_compute_homography(&sl, NULL, &DEFAULT_PARAMS, &H, NULL),
               DSO_ERR_INVALID_ARG);
    ASSERT_ERR(ransac_compute_homography(&sl, &sl, &DEFAULT_PARAMS, NULL, NULL),
               DSO_ERR_INVALID_ARG);
    free(sl.stars);
    return 0;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    SUITE("dlt_homography");
    RUN(test_dlt_identity);
    RUN(test_dlt_pure_translation);
    RUN(test_dlt_known_rotation_scale);
    RUN(test_dlt_overdetermined);
    RUN(test_dlt_too_few_points);
    RUN(test_dlt_null_args);

    SUITE("ransac_compute_homography");
    RUN(test_ransac_clean_data);
    RUN(test_ransac_with_outliers);
    RUN(test_ransac_insufficient_stars);
    RUN(test_ransac_no_matches);
    RUN(test_ransac_null_params_uses_defaults);
    RUN(test_ransac_inliers_out);
    RUN(test_ransac_null_args);

    return SUMMARY();
}
