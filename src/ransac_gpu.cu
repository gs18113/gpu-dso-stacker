#include "ransac_gpu.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RANSAC_GPU_THREADS 256
#define TRIANGLE_HASH_EPSILON 0.005f
#define DEGENERATE_HASH_SENTINEL 2.0f

static inline int choose3_int(int n)
{
    if (n < 3) return 0;
    return (n * (n - 1) * (n - 2)) / 6;
}

struct HashTupleLess {
    __host__ __device__
    bool operator()(const thrust::tuple<float, float> &a,
                    const thrust::tuple<float, float> &b) const
    {
        float a1 = thrust::get<0>(a);
        float a2 = thrust::get<1>(a);
        float b1 = thrust::get<0>(b);
        float b2 = thrust::get<1>(b);
        if (a1 < b1) return true;
        if (a1 > b1) return false;
        return a2 < b2;
    }
};

/*
 * unrank_triangle_idx maps a linear triangle index t in [0, C(n,3)) to
 * lexicographic (i, j, k) with i < j < k.
 *
 * This avoids naive nested-if filtering in kernels (if i<j<k) that would
 * create heavy warp divergence. Each thread receives an exact valid triangle.
 */
__device__ static void unrank_triangle_idx(int n, int t, int *i_out, int *j_out, int *k_out)
{
    int i = 0;
    while (i < n - 2) {
        int cnt_i = ((n - i - 1) * (n - i - 2)) / 2;
        if (t < cnt_i) break;
        t -= cnt_i;
        i++;
    }

    int j = i + 1;
    while (j < n - 1) {
        int cnt_j = (n - j - 1);
        if (t < cnt_j) break;
        t -= cnt_j;
        j++;
    }

    *i_out = i;
    *j_out = j;
    *k_out = j + 1 + t;
}

__device__ static void sort3_with_ids(float *a, int *ia, float *b, int *ib, float *c, int *ic)
{
    float tv;
    int ti;
    if (*a > *b) { tv = *a; *a = *b; *b = tv; ti = *ia; *ia = *ib; *ib = ti; }
    if (*b > *c) { tv = *b; *b = *c; *c = tv; ti = *ib; *ib = *ic; *ic = ti; }
    if (*a > *b) { tv = *a; *a = *b; *b = tv; ti = *ia; *ia = *ib; *ib = ti; }
}

__global__ static void generate_triangle_hashes_kernel(
    const StarPos *stars,
    int n,
    float *r1,
    float *r2,
    int *v0,
    int *v1,
    int *v2,
    int n_tri)
{
    extern __shared__ unsigned char smem[];
    StarPos *s_stars = (StarPos *)smem;
    int t = threadIdx.x;
    int idx;

    for (idx = t; idx < n; idx += blockDim.x) s_stars[idx] = stars[idx];
    __syncthreads();

    for (idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_tri;
         idx += gridDim.x * blockDim.x) {
        int i, j, k;
        float lij, ljk, lki;
        float a, b, c;
        int va, vb, vc;
        float dx, dy;

        unrank_triangle_idx(n, idx, &i, &j, &k);

        dx = s_stars[i].x - s_stars[j].x;
        dy = s_stars[i].y - s_stars[j].y;
        lij = sqrtf(dx * dx + dy * dy);

        dx = s_stars[j].x - s_stars[k].x;
        dy = s_stars[j].y - s_stars[k].y;
        ljk = sqrtf(dx * dx + dy * dy);

        dx = s_stars[k].x - s_stars[i].x;
        dy = s_stars[k].y - s_stars[i].y;
        lki = sqrtf(dx * dx + dy * dy);

        a = lij; b = ljk; c = lki;
        va = k; vb = i; vc = j;

        if (c < 1e-6f) {
            r1[idx] = DEGENERATE_HASH_SENTINEL;
            r2[idx] = DEGENERATE_HASH_SENTINEL;
            v0[idx] = -1; v1[idx] = -1; v2[idx] = -1;
            continue;
        }

        sort3_with_ids(&a, &va, &b, &vb, &c, &vc);

        r1[idx] = a / c;
        r2[idx] = b / c;
        v0[idx] = va;
        v1[idx] = vb;
        v2[idx] = vc;
    }
}

__global__ static void vote_matches_kernel(
    const float *ref_r1,
    const float *ref_r2,
    const int *ref_v0,
    const int *ref_v1,
    const int *ref_v2,
    int n_ref_hash,
    const float *src_r1,
    const float *src_r2,
    const int *src_v0,
    const int *src_v1,
    const int *src_v2,
    const int *src_lb,
    const int *src_ub,
    int n_src_hash,
    int n_ref_stars,
    float eps,
    int *votes)
{
    int sidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sidx >= n_src_hash) return;

    int sv0 = src_v0[sidx];
    int sv1 = src_v1[sidx];
    int sv2 = src_v2[sidx];
    int lb = src_lb[sidx];
    int ub = src_ub[sidx];
    int r;

    if (sv0 < 0 || sv1 < 0 || sv2 < 0) return;
    if (lb < 0) lb = 0;
    if (ub > n_ref_hash) ub = n_ref_hash;

    for (r = lb; r < ub; r++) {
        int rv0 = ref_v0[r];
        int rv1 = ref_v1[r];
        int rv2 = ref_v2[r];
        if (rv0 < 0 || rv1 < 0 || rv2 < 0) continue;
        if (fabsf(ref_r2[r] - src_r2[sidx]) > eps) continue;
        if (fabsf(ref_r1[r] - src_r1[sidx]) > eps) continue;

        atomicAdd(&votes[sv0 * n_ref_stars + rv0], 1);
        atomicAdd(&votes[sv1 * n_ref_stars + rv1], 1);
        atomicAdd(&votes[sv2 * n_ref_stars + rv2], 1);
    }
}

__global__ static void make_r1_bounds_kernel(const float *src_r1,
                                             float *src_r1_lo,
                                             float *src_r1_hi,
                                             int n,
                                             float eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    src_r1_lo[i] = src_r1[i] - eps;
    src_r1_hi[i] = src_r1[i] + eps;
}

static double reproj_err_sq(const Homography *H,
                            float rx, float ry,
                            float sx, float sy)
{
    const double *h = H->h;
    double qx_h = h[0] * rx + h[1] * ry + h[2];
    double qy_h = h[3] * rx + h[4] * ry + h[5];
    double qw = h[6] * rx + h[7] * ry + h[8];
    double qx, qy;

    if (fabs(qw) < 1e-12) return 1e18;
    qx = qx_h / qw - sx;
    qy = qy_h / qw - sy;
    return qx * qx + qy * qy;
}

static double homography_det(const Homography *H)
{
    const double *h = H->h;
    return h[0] * (h[4] * h[8] - h[5] * h[7])
         - h[1] * (h[3] * h[8] - h[5] * h[6])
         + h[2] * (h[3] * h[7] - h[4] * h[6]);
}

typedef struct {
    int src_idx;
    int ref_idx;
    int votes;
} PairVote;

static int cmp_pair_vote_desc(const void *a, const void *b)
{
    const PairVote *x = (const PairVote *)a;
    const PairVote *y = (const PairVote *)b;
    if (x->votes > y->votes) return -1;
    if (x->votes < y->votes) return 1;
    if (x->src_idx < y->src_idx) return -1;
    if (x->src_idx > y->src_idx) return 1;
    if (x->ref_idx < y->ref_idx) return -1;
    if (x->ref_idx > y->ref_idx) return 1;
    return 0;
}

static int extract_correspondences_from_votes(const StarPos *h_ref_stars,
                                              int n_ref,
                                              const StarPos *h_src_stars,
                                              int n_src,
                                              const int *h_votes,
                                              StarPos *out_ref,
                                              StarPos *out_src)
{
    PairVote *pairs = (PairVote *)malloc((size_t)(n_ref < n_src ? n_ref : n_src) * sizeof(PairVote));
    int *best_ref_for_src = (int *)malloc((size_t)n_src * sizeof(int));
    int *best_vote_for_src = (int *)malloc((size_t)n_src * sizeof(int));
    int *best_src_for_ref = (int *)malloc((size_t)n_ref * sizeof(int));
    int *best_vote_for_ref = (int *)malloc((size_t)n_ref * sizeof(int));
    int n_pairs = 0;
    int selected = 0;
    int s;

    if (!pairs || !best_ref_for_src || !best_vote_for_src ||
        !best_src_for_ref || !best_vote_for_ref) {
        free(pairs);
        free(best_ref_for_src);
        free(best_vote_for_src);
        free(best_src_for_ref);
        free(best_vote_for_ref);
        return -1;
    }

    for (s = 0; s < n_src; s++) {
        best_ref_for_src[s] = -1;
        best_vote_for_src[s] = 0;
    }
    for (int r = 0; r < n_ref; r++) {
        best_src_for_ref[r] = -1;
        best_vote_for_ref[r] = 0;
    }

    for (s = 0; s < n_src; s++) {
        for (int r = 0; r < n_ref; r++) {
            int v = h_votes[s * n_ref + r];
            if (v > best_vote_for_src[s]) {
                best_vote_for_src[s] = v;
                best_ref_for_src[s] = r;
            }
            if (v > best_vote_for_ref[r]) {
                best_vote_for_ref[r] = v;
                best_src_for_ref[r] = s;
            }
        }
    }

    for (s = 0; s < n_src; s++) {
        int r = best_ref_for_src[s];
        if (r < 0) continue;
        if (best_vote_for_src[s] <= 0) continue;
        if (best_src_for_ref[r] != s) continue;
        pairs[n_pairs].src_idx = s;
        pairs[n_pairs].ref_idx = r;
        pairs[n_pairs].votes = best_vote_for_src[s];
        n_pairs++;
    }

    if (n_pairs > 0) {
        qsort(pairs, (size_t)n_pairs, sizeof(PairVote), cmp_pair_vote_desc);
    }

    for (s = 0; s < n_pairs; s++) {
        int si = pairs[s].src_idx;
        int ri = pairs[s].ref_idx;
        out_src[selected] = h_src_stars[si];
        out_ref[selected] = h_ref_stars[ri];
        selected++;
    }

    free(pairs);
    free(best_ref_for_src);
    free(best_vote_for_src);
    free(best_src_for_ref);
    free(best_vote_for_ref);
    return selected;
}

static int translate_only_model(const StarPos *h_ref_stars,
                                int n_ref,
                                const StarPos *h_src_stars,
                                int n_src,
                                Homography *H_out,
                                int *n_pairs_out,
                                float match_radius)
{
    float r2 = match_radius * match_radius;
    float sum_dx = 0.0f, sum_dy = 0.0f;
    int pairs = 0;

    for (int si = 0; si < n_src; si++) {
        float sx = h_src_stars[si].x;
        float sy = h_src_stars[si].y;
        float best_d2 = r2 + 1.0f;
        int best_ri = -1;
        for (int ri = 0; ri < n_ref; ri++) {
            float dx = sx - h_ref_stars[ri].x;
            float dy = sy - h_ref_stars[ri].y;
            float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_ri = ri;
            }
        }
        if (best_ri >= 0 && best_d2 <= r2) {
            sum_dx += sx - h_ref_stars[best_ri].x;
            sum_dy += sy - h_ref_stars[best_ri].y;
            pairs++;
        }
    }

    if (pairs < 4) return 0;

    memset(H_out, 0, sizeof(*H_out));
    H_out->h[0] = 1.0;
    H_out->h[4] = 1.0;
    H_out->h[8] = 1.0;
    H_out->h[2] = (double)(sum_dx / (float)pairs);
    H_out->h[5] = (double)(sum_dy / (float)pairs);
    if (n_pairs_out) *n_pairs_out = pairs;
    return 1;
}

static const RansacParams RANSAC_DEFAULTS = {
    1000, 2.0f, 30.0f, 0.99f, 4
};

extern "C" DsoError ransac_compute_homography_gpu(const StarPos     *d_ref_stars,
                                                     int                n_ref,
                                                     const StarPos     *d_src_stars,
                                                     int                n_src,
                                                     const RansacParams *params,
                                                     Homography         *H_out,
                                                     int                *n_inliers_out,
                                                     cudaStream_t        stream)
{
    const RansacParams *p = params ? params : &RANSAC_DEFAULTS;
    DsoError err = DSO_OK;

    int n_ref_tri;
    int n_src_tri;
    int ref_blocks;
    int src_blocks;

    float *d_ref_r1 = NULL, *d_ref_r2 = NULL;
    int *d_ref_v0 = NULL, *d_ref_v1 = NULL, *d_ref_v2 = NULL;
    float *d_src_r1 = NULL, *d_src_r2 = NULL;
    int *d_src_v0 = NULL, *d_src_v1 = NULL, *d_src_v2 = NULL;
    int *d_lb = NULL, *d_ub = NULL;
    int *d_votes = NULL;
    float *d_src_r1_lo = NULL;
    float *d_src_r1_hi = NULL;

    int *h_votes = NULL;
    StarPos *h_ref_stars = NULL;
    StarPos *h_src_stars = NULL;
    StarPos *corr_ref = NULL;
    StarPos *corr_src = NULL;
    StarPos *in_ref = NULL;
    StarPos *in_src = NULL;

    int n_corr;
    int n_inliers;
    int i;
    int max_corr;
    size_t shmem_ref;
    size_t shmem_src;
    thrust::device_ptr<float> ref_r1_ptr;
    thrust::device_ptr<float> src_r1_ptr;

    if (!d_ref_stars || !d_src_stars || !H_out) return DSO_ERR_INVALID_ARG;
    if (n_ref < 4 || n_src < 4) return DSO_ERR_STAR_DETECT;

    n_ref_tri = choose3_int(n_ref);
    n_src_tri = choose3_int(n_src);
    if (n_ref_tri <= 0 || n_src_tri <= 0) return DSO_ERR_STAR_DETECT;

    if (cudaMalloc((void **)&d_ref_r1, (size_t)n_ref_tri * sizeof(float)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_ref_r2, (size_t)n_ref_tri * sizeof(float)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_ref_v0, (size_t)n_ref_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_ref_v1, (size_t)n_ref_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_ref_v2, (size_t)n_ref_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    if (cudaMalloc((void **)&d_src_r1, (size_t)n_src_tri * sizeof(float)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_src_r2, (size_t)n_src_tri * sizeof(float)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_src_v0, (size_t)n_src_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_src_v1, (size_t)n_src_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_src_v2, (size_t)n_src_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    if (cudaMalloc((void **)&d_lb, (size_t)n_src_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_ub, (size_t)n_src_tri * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_src_r1_lo, (size_t)n_src_tri * sizeof(float)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMalloc((void **)&d_src_r1_hi, (size_t)n_src_tri * sizeof(float)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    if (cudaMalloc((void **)&d_votes, (size_t)n_src * (size_t)n_ref * sizeof(int)) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMemsetAsync(d_votes, 0, (size_t)n_src * (size_t)n_ref * sizeof(int), stream) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    ref_blocks = (n_ref_tri + RANSAC_GPU_THREADS - 1) / RANSAC_GPU_THREADS;
    src_blocks = (n_src_tri + RANSAC_GPU_THREADS - 1) / RANSAC_GPU_THREADS;
    if (ref_blocks < 1) ref_blocks = 1;
    if (src_blocks < 1) src_blocks = 1;
    if (ref_blocks > 1024) ref_blocks = 1024;
    if (src_blocks > 1024) src_blocks = 1024;

    shmem_ref = (size_t)n_ref * sizeof(StarPos);
    shmem_src = (size_t)n_src * sizeof(StarPos);

    generate_triangle_hashes_kernel<<<ref_blocks, RANSAC_GPU_THREADS, shmem_ref, stream>>>(
        d_ref_stars, n_ref, d_ref_r1, d_ref_r2, d_ref_v0, d_ref_v1, d_ref_v2, n_ref_tri);
    if (cudaGetLastError() != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    generate_triangle_hashes_kernel<<<src_blocks, RANSAC_GPU_THREADS, shmem_src, stream>>>(
        d_src_stars, n_src, d_src_r1, d_src_r2, d_src_v0, d_src_v1, d_src_v2, n_src_tri);
    if (cudaGetLastError() != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    {
        auto key_first = thrust::make_zip_iterator(thrust::make_tuple(d_ref_r1, d_ref_r2));
        auto key_last = key_first + n_ref_tri;
        auto val_first = thrust::make_zip_iterator(thrust::make_tuple(d_ref_v0, d_ref_v1, d_ref_v2));
        thrust::sort_by_key(thrust::cuda::par.on(stream), key_first, key_last, val_first, HashTupleLess());
    }

    ref_r1_ptr = thrust::device_pointer_cast(d_ref_r1);
    src_r1_ptr = thrust::device_pointer_cast(d_src_r1);

    make_r1_bounds_kernel<<<src_blocks, RANSAC_GPU_THREADS, 0, stream>>>(
        d_src_r1, d_src_r1_lo, d_src_r1_hi, n_src_tri, TRIANGLE_HASH_EPSILON);
    if (cudaGetLastError() != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    thrust::lower_bound(thrust::cuda::par.on(stream),
                        ref_r1_ptr, ref_r1_ptr + n_ref_tri,
                        thrust::device_pointer_cast(d_src_r1_lo),
                        thrust::device_pointer_cast(d_src_r1_lo) + n_src_tri,
                        thrust::device_pointer_cast(d_lb));

    thrust::upper_bound(thrust::cuda::par.on(stream),
                        ref_r1_ptr, ref_r1_ptr + n_ref_tri,
                        thrust::device_pointer_cast(d_src_r1_hi),
                        thrust::device_pointer_cast(d_src_r1_hi) + n_src_tri,
                        thrust::device_pointer_cast(d_ub));

    vote_matches_kernel<<<src_blocks, RANSAC_GPU_THREADS, 0, stream>>>(
        d_ref_r1, d_ref_r2, d_ref_v0, d_ref_v1, d_ref_v2, n_ref_tri,
        d_src_r1, d_src_r2, d_src_v0, d_src_v1, d_src_v2, d_lb, d_ub, n_src_tri,
        n_ref, TRIANGLE_HASH_EPSILON, d_votes);
    if (cudaGetLastError() != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    h_votes = (int *)malloc((size_t)n_src * (size_t)n_ref * sizeof(int));
    h_ref_stars = (StarPos *)malloc((size_t)n_ref * sizeof(StarPos));
    h_src_stars = (StarPos *)malloc((size_t)n_src * sizeof(StarPos));
    max_corr = (n_ref < n_src) ? n_ref : n_src;
    corr_ref = (StarPos *)malloc((size_t)max_corr * sizeof(StarPos));
    corr_src = (StarPos *)malloc((size_t)max_corr * sizeof(StarPos));
    in_ref = (StarPos *)malloc((size_t)max_corr * sizeof(StarPos));
    in_src = (StarPos *)malloc((size_t)max_corr * sizeof(StarPos));
    if (!h_votes || !h_ref_stars || !h_src_stars || !corr_ref || !corr_src || !in_ref || !in_src) {
        err = DSO_ERR_ALLOC;
        goto cleanup;
    }

    if (cudaMemcpyAsync(h_votes, d_votes,
                        (size_t)n_src * (size_t)n_ref * sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMemcpyAsync(h_ref_stars, d_ref_stars,
                        (size_t)n_ref * sizeof(StarPos),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaMemcpyAsync(h_src_stars, d_src_stars,
                        (size_t)n_src * sizeof(StarPos),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = DSO_ERR_CUDA; goto cleanup; }

    n_corr = extract_correspondences_from_votes(h_ref_stars, n_ref, h_src_stars, n_src,
                                                h_votes, corr_ref, corr_src);
    if (n_corr < 4) {
        int ok = translate_only_model(h_ref_stars, n_ref, h_src_stars, n_src,
                                      H_out, &n_corr, p->match_radius);
        if (!ok) { err = DSO_ERR_RANSAC; goto cleanup; }
        n_inliers = n_corr;
        if (n_inliers_out) *n_inliers_out = n_inliers;
        err = DSO_OK;
        goto cleanup;
    }

    err = dlt_homography(corr_ref, corr_src, n_corr, H_out);
    if (err != DSO_OK) { err = DSO_ERR_RANSAC; goto cleanup; }

    if (fabs(homography_det(H_out)) < 1e-12) {
        err = DSO_ERR_INVALID_ARG;
        goto cleanup;
    }

    n_inliers = 0;
    {
        double thresh_sq = (double)p->inlier_thresh * (double)p->inlier_thresh;
        for (i = 0; i < n_corr; i++) {
            double e2 = reproj_err_sq(H_out,
                                      corr_ref[i].x, corr_ref[i].y,
                                      corr_src[i].x, corr_src[i].y);
            if (e2 <= thresh_sq) {
                in_ref[n_inliers] = corr_ref[i];
                in_src[n_inliers] = corr_src[i];
                n_inliers++;
            }
        }
    }

    if (n_inliers < p->min_inliers || n_inliers < 4) {
        err = DSO_ERR_RANSAC;
        goto cleanup;
    }

    err = dlt_homography(in_ref, in_src, n_inliers, H_out);
    if (err != DSO_OK) { err = DSO_ERR_RANSAC; goto cleanup; }

    if (fabs(homography_det(H_out)) < 1e-12) {
        err = DSO_ERR_INVALID_ARG;
        goto cleanup;
    }

    if (n_inliers_out) *n_inliers_out = n_inliers;
    err = DSO_OK;

cleanup:
    cudaFree(d_ref_r1); cudaFree(d_ref_r2);
    cudaFree(d_ref_v0); cudaFree(d_ref_v1); cudaFree(d_ref_v2);
    cudaFree(d_src_r1); cudaFree(d_src_r2);
    cudaFree(d_src_v0); cudaFree(d_src_v1); cudaFree(d_src_v2);
    cudaFree(d_src_r1_lo); cudaFree(d_src_r1_hi);
    cudaFree(d_lb); cudaFree(d_ub);
    cudaFree(d_votes);

    free(h_votes);
    free(h_ref_stars);
    free(h_src_stars);
    free(corr_ref);
    free(corr_src);
    free(in_ref);
    free(in_src);

    return err;
}
