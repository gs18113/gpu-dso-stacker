/*
 * star_detect_cpu.c — Connected-component labeling and weighted center-of-mass
 * for star detection.
 *
 * Algorithm overview:
 *   1. Two-pass 8-connectivity CCL using a union-find (disjoint-set) data
 *      structure with path compression.
 *   2. Per-component accumulation of flux (from the convolved image) and
 *      weighted position sums (from the original image, clamped to ≥ 0).
 *   3. Sort by flux descending; return the top-K brightest blobs as StarPos.
 *
 * Memory usage: O(W × H) for the label array and union-find parent array.
 * Temporary component statistics are stored in a hash-like flat array indexed
 * by label, which is then compacted before sorting.
 */

#include "star_detect_cpu.h"
#include "compat.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

/* -------------------------------------------------------------------------
 * Stage A: Moffat convolution + threshold
 * ------------------------------------------------------------------------- */

#define MOFFAT_MAX_RADIUS 15   /* matches GPU constant-memory cap */

DsoError star_detect_cpu_moffat_convolve(const float        *src,
                                          float              *dst,
                                          int                 W, int H,
                                          const MoffatParams *params)
{
    if (!src || !dst || !params || W <= 0 || H <= 0) return DSO_ERR_INVALID_ARG;

    int R = (int)ceilf(3.f * params->alpha);
    if (R > MOFFAT_MAX_RADIUS) {
        fprintf(stderr, "star_detect_cpu: Moffat alpha=%.2f yields radius %d, "
                "clamping to MOFFAT_MAX_RADIUS=%d (kernel will be under-sized)\n",
                params->alpha, R, MOFFAT_MAX_RADIUS);
        R = MOFFAT_MAX_RADIUS;
    }
    int kw = 2 * R + 1;
    int kn  = kw * kw;

    /* Build normalised Moffat kernel on the stack / heap */
    float *kern = (float *)malloc((size_t)kn * sizeof(float));
    if (!kern) return DSO_ERR_ALLOC;

    float alpha2 = params->alpha * params->alpha;
    float ksum   = 0.f;
    for (int ky = -R; ky <= R; ky++) {
        for (int kx = -R; kx <= R; kx++) {
            float r2 = (float)(kx*kx + ky*ky);
            float v  = powf(1.f + r2 / alpha2, -params->beta);
            kern[(ky+R)*kw + (kx+R)] = v;
            ksum += v;
        }
    }
    /* Normalise */
    for (int i = 0; i < kn; i++) kern[i] /= ksum;

    /* 2-D convolution with zero-boundary padding */
    int y, x;
OMP_PARALLEL_FOR_COLLAPSE2
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            float acc = 0.f;
            for (int ky = -R; ky <= R; ky++) {
                int sy = y + ky;
                if (sy < 0 || sy >= H) continue;
                for (int kx = -R; kx <= R; kx++) {
                    int sx = x + kx;
                    if (sx < 0 || sx >= W) continue;
                    acc += kern[(ky+R)*kw + (kx+R)] * src[sy*W + sx];
                }
            }
            dst[y*W + x] = acc;
        }
    }

    free(kern);
    return DSO_OK;
}

DsoError star_detect_cpu_threshold(const float *convolved,
                                    uint8_t     *mask,
                                    int          W, int H,
                                    float        sigma_k)
{
    if (!convolved || !mask || W <= 0 || H <= 0) return DSO_ERR_INVALID_ARG;

    long npix = (long)W * H;

    /* Pass 1 — mean (double-precision accumulation for accuracy) */
    double sum = 0.0;
    long i;
#pragma omp parallel for reduction(+:sum) schedule(static)
    for (i = 0; i < npix; i++)
        sum += (double)convolved[i];
    double mean = sum / (double)npix;

    /* Pass 2 — Bessel-corrected variance */
    double sq = 0.0;
#pragma omp parallel for reduction(+:sq) schedule(static)
    for (i = 0; i < npix; i++) {
        double d = (double)convolved[i] - mean;
        sq += d * d;
    }
    double sigma = (npix > 1) ? sqrt(sq / (double)(npix - 1)) : 0.0;

    float thresh = (float)(mean + (double)sigma_k * sigma);

    /* Pass 3 — write mask */
#pragma omp parallel for schedule(static)
    for (i = 0; i < npix; i++)
        mask[i] = (convolved[i] > thresh) ? 1u : 0u;

    return DSO_OK;
}

DsoError star_detect_cpu_detect(const float        *src,
                                 float              *conv_out,
                                 uint8_t            *mask_out,
                                 int                 W, int H,
                                 const MoffatParams *params,
                                 float               sigma_k)
{
    DsoError err = star_detect_cpu_moffat_convolve(src, conv_out, W, H, params);
    if (err != DSO_OK) return err;
    return star_detect_cpu_threshold(conv_out, mask_out, W, H, sigma_k);
}

/* -------------------------------------------------------------------------
 * Union-find with path compression
 * ------------------------------------------------------------------------- */

/* Find root of label x with full path compression. */
static int uf_find(int *parent, int x)
{
    while (parent[x] != x)
        x = parent[x] = parent[parent[x]];  /* path halving */
    return x;
}

/* Union the sets containing a and b; return the new root. */
static int uf_union(int *parent, int a, int b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return a;
    if (a < b) { parent[b] = a; return a; }
    parent[a] = b; return b;
}

/* -------------------------------------------------------------------------
 * Component statistics (one per unique label)
 * ------------------------------------------------------------------------- */
typedef struct {
    double sum_w;    /* Σ max(0, original_pixel) — CoM weight sum            */
    double sum_wx;   /* Σ max(0, original_pixel) * x                         */
    double sum_wy;   /* Σ max(0, original_pixel) * y                         */
    double sum_gx;   /* Σ x — geometric centroid numerator (fallback)         */
    double sum_gy;   /* Σ y — geometric centroid numerator (fallback)         */
    double flux;     /* Σ convolved_pixel — for brightness ranking            */
    int    count;    /* pixel count in this component                         */
    int    valid;    /* 1 if this label was actually used                     */
} CompStats;

/* Comparator for qsort: descending flux. */
static int cmp_flux_desc(const void *a, const void *b)
{
    const StarPos *sa = (const StarPos *)a;
    const StarPos *sb = (const StarPos *)b;
    if (sb->flux > sa->flux) return  1;
    if (sb->flux < sa->flux) return -1;
    return 0;
}

/* -------------------------------------------------------------------------
 * Public function
 * ------------------------------------------------------------------------- */

DsoError star_detect_cpu_ccl_com(const uint8_t *mask,
                                  const float   *original,
                                  const float   *convolved,
                                  int            W, int H,
                                  int            top_k,
                                  StarList       *list_out)
{
    if (!mask || !original || !convolved || !list_out || W <= 0 || H <= 0)
        return DSO_ERR_INVALID_ARG;

    list_out->stars = NULL;
    list_out->n     = 0;

    /* Early exit for empty mask or top_k == 0. */
    if (top_k == 0) return DSO_OK;

    int npix = W * H;

    /* ---- Allocate label and parent arrays ---- */
    int *label  = (int *)malloc((size_t)npix * sizeof(int));
    int *parent = (int *)malloc((size_t)(npix + 1) * sizeof(int));
    if (!label || !parent) {
        free(label); free(parent);
        return DSO_ERR_ALLOC;
    }

    /*
     * Label 0 is reserved for "background" (mask == 0).
     * Star pixels start at label 1.  We use the pixel index + 1 as the
     * initial provisional label for each star pixel.
     */
    for (int i = 0; i <= npix; i++) parent[i] = i; /* identity = own root */

    /* ---- Pass 1: assign provisional labels + union ----
     * For each star pixel at (x, y), check the 4 already-visited neighbours
     * in raster order: N(0,-1), NW(-1,-1), W(-1,0), NE(+1,-1).
     * Assign the minimum neighbour label; union all found labels.
     */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            if (!mask[idx]) {
                label[idx] = 0;
                continue;
            }
            /* Provisional label = this pixel's own index + 1 */
            label[idx] = idx + 1;

            /* Check 4 raster-order neighbours */
            int nbrs[4];
            int nn = 0;
            /* N */
            if (y > 0 && mask[(y-1)*W + x])
                nbrs[nn++] = label[(y-1)*W + x];
            /* NW */
            if (y > 0 && x > 0 && mask[(y-1)*W + (x-1)])
                nbrs[nn++] = label[(y-1)*W + (x-1)];
            /* W */
            if (x > 0 && mask[y*W + (x-1)])
                nbrs[nn++] = label[y*W + (x-1)];
            /* NE */
            if (y > 0 && x < W-1 && mask[(y-1)*W + (x+1)])
                nbrs[nn++] = label[(y-1)*W + (x+1)];

            if (nn == 0) continue;  /* isolated pixel; keep its own label */

            /* Find minimum active neighbour label root */
            int min_root = uf_find(parent, nbrs[0]);
            for (int k = 1; k < nn; k++) {
                int r = uf_find(parent, nbrs[k]);
                if (r < min_root) min_root = r;
            }
            /* Assign minimum root to this pixel */
            label[idx] = min_root;
            /* Union all neighbour roots together */
            for (int k = 0; k < nn; k++)
                uf_union(parent, min_root, nbrs[k]);
        }
    }

    /* ---- Pass 2: resolve each pixel label to its root ---- */
    for (int i = 0; i < npix; i++) {
        if (label[i])
            label[i] = uf_find(parent, label[i]);
    }

    /* ---- Pass 3: Re-map roots to a contiguous range [1, n_unique] ----
     * This avoids allocating stats[npix+1] which is mostly empty. */
    int *label_map = (int *)malloc((size_t)(npix + 1) * sizeof(int));
    if (!label_map) {
        free(label); free(parent);
        return DSO_ERR_ALLOC;
    }
    memset(label_map, 0, (size_t)(npix + 1) * sizeof(int));

    int n_unique = 0;
    for (int i = 0; i < npix; i++) {
        int root = label[i];
        if (root > 0 && label_map[root] == 0) {
            label_map[root] = ++n_unique;
        }
    }
    /* Apply map to label array */
    for (int i = 0; i < npix; i++) {
        if (label[i]) label[i] = label_map[label[i]];
    }
    free(label_map);
    free(parent);

    if (n_unique == 0) {
        free(label);
        return DSO_OK;
    }

    /* ---- Allocate component statistics indexed by re-mapped label (1..n_unique) ---- */
    CompStats *stats = (CompStats *)calloc((size_t)(n_unique + 1), sizeof(CompStats));
    if (!stats) {
        free(label);
        return DSO_ERR_ALLOC;
    }

    /* ---- Accumulate per-component statistics ---- */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            int lbl = label[idx];
            if (!lbl) continue;  /* background */

            CompStats *s = &stats[lbl];
            s->valid = 1;

            float orig_v  = original[idx];
            float conv_v  = convolved[idx];
            float w       = orig_v > 0.f ? orig_v : 0.f;  /* clamp negatives */

            s->sum_w  += w;
            s->sum_wx += w * x;
            s->sum_wy += w * y;
            s->sum_gx += x;              /* geometric fallback */
            s->sum_gy += y;
            s->flux   += conv_v;
            s->count  ++;
        }
    }

    free(label);

    /* ---- Build flat array of StarPos from valid component stats ---- */
    int n_out = (n_unique < top_k) ? n_unique : top_k;
    /* We need all components first to sort, then truncate to top_k. */
    StarPos *tmp = (StarPos *)malloc((size_t)n_unique * sizeof(StarPos));
    if (!tmp) {
        free(stats);
        return DSO_ERR_ALLOC;
    }

    int k = 0;
    for (int lbl = 1; lbl <= n_unique; lbl++) {
        CompStats *s = &stats[lbl];
        if (!s->valid) continue;

        float cx, cy;
        if (s->sum_w > 1e-12) {
            cx = (float)(s->sum_wx / s->sum_w);
            cy = (float)(s->sum_wy / s->sum_w);
        } else {
            /* Geometric centroid fallback when all original values are ≤ 0 */
            cx = (float)(s->sum_gx / s->count);
            cy = (float)(s->sum_gy / s->count);
        }
        tmp[k++] = (StarPos){ cx, cy, (float)s->flux };
    }

    free(stats);

    /* Sort by flux descending */
    qsort(tmp, (size_t)n_unique, sizeof(StarPos), cmp_flux_desc);

    /* Allocate final output array of size min(n_unique, top_k) */
    n_out = (n_unique < top_k) ? n_unique : top_k;
    if (n_out == 0) {
        free(tmp);
        return DSO_OK;
    }

    StarPos *result = (StarPos *)malloc((size_t)n_out * sizeof(StarPos));
    if (!result) {
        free(tmp);
        return DSO_ERR_ALLOC;
    }
    memcpy(result, tmp, (size_t)n_out * sizeof(StarPos));
    free(tmp);

    list_out->stars = result;
    list_out->n     = n_out;
    return DSO_OK;
}
