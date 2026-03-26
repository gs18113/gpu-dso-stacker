#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef _MSC_VER
#include "compat.h"
#else
#include <getopt.h>
#endif

#include <vector>
#include <string>

#include "dso_types.h"
#include "fits_io.h"
#include "debayer_cpu.h"
#include "star_detect_cpu.h"
#include "ransac.h"

struct DetectResult {
    StarList stars;
    int width;
    int height;
};

struct MatchDiag {
    int accepted;
    int rejected_radius;
    int rejected_ratio;
};

static constexpr float LOWE_RATIO_THRESHOLD = 0.8f;

static void usage(const char *prog)
{
    std::fprintf(stderr,
        "Usage: %s --ref <ref.fits> --frame <frame.fits> [options]\n"
        "\n"
        "Star detection:\n"
        "  --star-sigma <float>      Detection threshold in sigma units (default: 3.0)\n"
        "  --moffat-alpha <float>    Moffat alpha (default: 2.5)\n"
        "  --moffat-beta <float>     Moffat beta (default: 2.0)\n"
        "  --top-stars <int>         Keep top-K stars per image (default: 50)\n"
        "  --bayer <pattern>         none | rggb | bggr | grbg | gbrg (default: auto)\n"
        "\n"
        "RANSAC:\n"
        "  --ransac-iters <int>      Max iterations (default: 1000)\n"
        "  --ransac-thresh <float>   Inlier reprojection threshold px (default: 2.0)\n"
        "  --match-radius <float>    Match search radius px (default: 30.0)\n"
        "  --confidence <float>      Desired success probability (default: 0.99)\n"
        "  --min-inliers <int>       Minimum inliers to accept model (default: 4)\n",
        prog);
}

static bool parse_bayer(const char *s, BayerPattern *out)
{
    if (!s || !out) return false;
    if      (std::strcmp(s, "none") == 0) *out = BAYER_NONE;
    else if (std::strcmp(s, "rggb") == 0) *out = BAYER_RGGB;
    else if (std::strcmp(s, "bggr") == 0) *out = BAYER_BGGR;
    else if (std::strcmp(s, "grbg") == 0) *out = BAYER_GRBG;
    else if (std::strcmp(s, "gbrg") == 0) *out = BAYER_GBRG;
    else return false;
    return true;
}

static DsoError detect_stars(const char *path,
                             const MoffatParams *moffat,
                             float star_sigma,
                             int top_stars,
                             bool bayer_override_set,
                             BayerPattern bayer_override,
                             DetectResult *out)
{
    if (!path || !moffat || !out) return DSO_ERR_INVALID_ARG;

    Image raw = {};
    Image lum = {};
    Image conv = {};
    std::vector<uint8_t> mask;

    DsoError err = fits_load(path, &raw);
    if (err != DSO_OK) return err;

    BayerPattern pat = BAYER_NONE;
    if (bayer_override_set) {
        pat = bayer_override;
    } else {
        DsoError berr = fits_get_bayer_pattern(path, &pat);
        if (berr != DSO_OK) pat = BAYER_NONE;
    }

    lum.width = raw.width;
    lum.height = raw.height;
    conv.width = raw.width;
    conv.height = raw.height;

    const size_t npix = (size_t)raw.width * (size_t)raw.height;
    lum.data = (float *)std::malloc(npix * sizeof(float));
    conv.data = (float *)std::malloc(npix * sizeof(float));
    if (!lum.data || !conv.data) {
        image_free(&raw);
        image_free(&lum);
        image_free(&conv);
        return DSO_ERR_ALLOC;
    }
    mask.resize(npix);

    err = debayer_cpu(raw.data, lum.data, raw.width, raw.height, pat);
    if (err != DSO_OK) {
        image_free(&raw);
        image_free(&lum);
        image_free(&conv);
        return err;
    }

    err = star_detect_cpu_detect(lum.data, conv.data, mask.data(),
                                 raw.width, raw.height, moffat, star_sigma);
    if (err != DSO_OK) {
        image_free(&raw);
        image_free(&lum);
        image_free(&conv);
        return err;
    }

    out->stars.stars = nullptr;
    out->stars.n = 0;
    err = star_detect_cpu_ccl_com(mask.data(), lum.data, conv.data,
                                  raw.width, raw.height, top_stars,
                                  &out->stars);
    if (err == DSO_OK) {
        out->width = raw.width;
        out->height = raw.height;
    }

    image_free(&raw);
    image_free(&lum);
    image_free(&conv);
    return err;
}

static MatchDiag diagnose_matches(const StarList *ref_list,
                                  const StarList *frm_list,
                                  float match_radius)
{
    MatchDiag d = {0, 0, 0};
    if (!ref_list || !frm_list || match_radius <= 0.0f) return d;

    const float r2 = match_radius * match_radius;
    for (int ri = 0; ri < ref_list->n; ++ri) {
        const float rx = ref_list->stars[ri].x;
        const float ry = ref_list->stars[ri].y;

        float d1 = r2 + 1.0f;
        float d2 = r2 + 2.0f;
        int best_idx = -1;

        for (int fi = 0; fi < frm_list->n; ++fi) {
            const float dx = frm_list->stars[fi].x - rx;
            const float dy = frm_list->stars[fi].y - ry;
            const float dist2 = dx * dx + dy * dy;
            if (dist2 < d1) {
                d2 = d1;
                d1 = dist2;
                best_idx = fi;
            } else if (dist2 < d2) {
                d2 = dist2;
            }
        }

        if (best_idx < 0 || d1 > r2) {
            ++d.rejected_radius;
            continue;
        }

        if (d2 > 0.0f && d2 < r2 + 1.0f && d1 > 0.0f &&
            std::sqrt(d1 / d2) > LOWE_RATIO_THRESHOLD) {
            ++d.rejected_ratio;
            continue;
        }

        ++d.accepted;
    }

    return d;
}

int main(int argc, char **argv)
{
    enum {
        OPT_REF = 256,
        OPT_FRAME,
        OPT_STAR_SIGMA,
        OPT_MOFFAT_ALPHA,
        OPT_MOFFAT_BETA,
        OPT_TOP_STARS,
        OPT_BAYER,
        OPT_RANSAC_ITERS,
        OPT_RANSAC_THRESH,
        OPT_MATCH_RADIUS,
        OPT_CONFIDENCE,
        OPT_MIN_INLIERS
    };

    const char *ref_path = nullptr;
    const char *frame_path = nullptr;
    float star_sigma = 3.0f;
    MoffatParams moffat = {2.5f, 2.0f};
    int top_stars = 50;
    bool bayer_override_set = false;
    BayerPattern bayer_override = BAYER_NONE;

    RansacParams params = {
        1000,
        2.0f,
        30.0f,
        0.99f,
        4
    };

    static struct option long_opts[] = {
        {"ref",           required_argument, nullptr, OPT_REF},
        {"frame",         required_argument, nullptr, OPT_FRAME},
        {"star-sigma",    required_argument, nullptr, OPT_STAR_SIGMA},
        {"moffat-alpha",  required_argument, nullptr, OPT_MOFFAT_ALPHA},
        {"moffat-beta",   required_argument, nullptr, OPT_MOFFAT_BETA},
        {"top-stars",     required_argument, nullptr, OPT_TOP_STARS},
        {"bayer",         required_argument, nullptr, OPT_BAYER},
        {"ransac-iters",  required_argument, nullptr, OPT_RANSAC_ITERS},
        {"ransac-thresh", required_argument, nullptr, OPT_RANSAC_THRESH},
        {"match-radius",  required_argument, nullptr, OPT_MATCH_RADIUS},
        {"confidence",    required_argument, nullptr, OPT_CONFIDENCE},
        {"min-inliers",   required_argument, nullptr, OPT_MIN_INLIERS},
        {nullptr, 0, nullptr, 0}
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "", long_opts, nullptr)) != -1) {
        switch (opt) {
        case OPT_REF: ref_path = optarg; break;
        case OPT_FRAME: frame_path = optarg; break;
        case OPT_STAR_SIGMA: star_sigma = std::strtof(optarg, nullptr); break;
        case OPT_MOFFAT_ALPHA: moffat.alpha = std::strtof(optarg, nullptr); break;
        case OPT_MOFFAT_BETA: moffat.beta = std::strtof(optarg, nullptr); break;
        case OPT_TOP_STARS: top_stars = std::atoi(optarg); break;
        case OPT_BAYER:
            if (!parse_bayer(optarg, &bayer_override)) {
                std::fprintf(stderr, "Error: unknown --bayer '%s'\n", optarg);
                return 1;
            }
            bayer_override_set = true;
            break;
        case OPT_RANSAC_ITERS: params.max_iters = std::atoi(optarg); break;
        case OPT_RANSAC_THRESH: params.inlier_thresh = std::strtof(optarg, nullptr); break;
        case OPT_MATCH_RADIUS: params.match_radius = std::strtof(optarg, nullptr); break;
        case OPT_CONFIDENCE: params.confidence = std::strtof(optarg, nullptr); break;
        case OPT_MIN_INLIERS: params.min_inliers = std::atoi(optarg); break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (!ref_path || !frame_path) {
        usage(argv[0]);
        return 1;
    }

    if (star_sigma <= 0.f || moffat.alpha <= 0.f || moffat.beta <= 0.f ||
        top_stars <= 0 || params.max_iters <= 0 || params.inlier_thresh <= 0.f ||
        params.match_radius <= 0.f || params.confidence <= 0.f ||
        params.confidence >= 1.f || params.min_inliers < 4) {
        std::fprintf(stderr, "Error: invalid numeric option\n");
        return 1;
    }

    DetectResult ref = {};
    DetectResult frm = {};

    DsoError err = detect_stars(ref_path, &moffat, star_sigma, top_stars,
                                bayer_override_set, bayer_override, &ref);
    if (err != DSO_OK) {
        std::fprintf(stderr, "Error: star detection failed for ref image '%s' (err=%d)\n",
                     ref_path, (int)err);
        return 1;
    }

    err = detect_stars(frame_path, &moffat, star_sigma, top_stars,
                       bayer_override_set, bayer_override, &frm);
    if (err != DSO_OK) {
        std::fprintf(stderr, "Error: star detection failed for frame image '%s' (err=%d)\n",
                     frame_path, (int)err);
        std::free(ref.stars.stars);
        return 1;
    }

    std::printf("Reference image: %s\n", ref_path);
    std::printf("  size: %dx%d, detected stars: %d\n", ref.width, ref.height, ref.stars.n);
    std::printf("Frame image: %s\n", frame_path);
    std::printf("  size: %dx%d, detected stars: %d\n", frm.width, frm.height, frm.stars.n);

    MatchDiag md = diagnose_matches(&ref.stars, &frm.stars, params.match_radius);
    std::printf("Matching diagnostics (radius=%.2f, ratio=%.2f):\n",
                (double)params.match_radius, (double)LOWE_RATIO_THRESHOLD);
    std::printf("  accepted: %d\n", md.accepted);
    std::printf("  rejected (no candidate within radius): %d\n", md.rejected_radius);
    std::printf("  rejected (Lowe ratio): %d\n", md.rejected_ratio);

    Homography H = {};
    int n_inliers = 0;
    err = ransac_compute_homography(&ref.stars, &frm.stars, &params, &H, &n_inliers);

    if (err == DSO_OK) {
        std::printf("RANSAC: SUCCESS\n");
        std::printf("  inliers: %d\n", n_inliers);
        std::printf("  homography (backward map ref->frame):\n");
        std::printf("    [%.9g %.9g %.9g]\n", H.h[0], H.h[1], H.h[2]);
        std::printf("    [%.9g %.9g %.9g]\n", H.h[3], H.h[4], H.h[5]);
        std::printf("    [%.9g %.9g %.9g]\n", H.h[6], H.h[7], H.h[8]);
    } else {
        std::printf("RANSAC: FAILED (err=%d)\n", (int)err);
        std::printf("Hint: if accepted matches are < %d, increase --match-radius or star count (--top-stars)\n",
                    params.min_inliers > 4 ? params.min_inliers : 4);
    }

    std::free(ref.stars.stars);
    std::free(frm.stars.stars);

    return (err == DSO_OK) ? 0 : 2;
}
