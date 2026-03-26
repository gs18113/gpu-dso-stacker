#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#ifdef _MSC_VER
#include "compat.h"
#else
#include <getopt.h>
#endif

#include "dso_types.h"
#include "csv_parser.h"
#include "fits_io.h"
#include "debayer_cpu.h"
#include "star_detect_cpu.h"
#include "star_detect_gpu.h"
#include "image_io.h"

struct CircleColor {
    float r;
    float g;
    float b;
};

static void usage(const char *prog)
{
    std::fprintf(stderr,
        "Usage: %s -f <input> -o <output-prefix|output.png> [options]\n"
        "\n"
        "Input:\n"
        "  -f, --file <path>         Input file list: CSV (filepath,is_reference),\n"
        "                            plain text list (one filepath per line), or single FITS file\n"
        "  -o, --output <path>       Output PNG path (single input) or prefix (multiple inputs)\n"
        "                            Default: stars (=> stars.png or stars_<idx>_<name>.png)\n"
        "\n"
        "Execution:\n"
        "      --cpu                 Use CPU Moffat+threshold (default)\n"
        "      --gpu                 Use GPU Moffat+threshold, then CPU CCL+CoM\n"
        "\n"
        "Star detection:\n"
        "      --star-sigma <float>  Threshold in sigma units (default: 3.0)\n"
        "      --moffat-alpha <f>    Moffat alpha (default: 2.5)\n"
        "      --moffat-beta <f>     Moffat beta (default: 2.0)\n"
        "      --top-stars <int>     Maximum stars to render (default: 200)\n"
        "      --bayer <pattern>     none | rggb | bggr | grbg | gbrg (default: auto)\n"
        "\n"
        "Drawing:\n"
        "      --circle-color <c>    red | green | blue | yellow | white | cyan | magenta\n"
        "                            or R,G,B floats in [0,1] (default: red)\n"
        "      --circle-radius <px>  Circle radius in pixels (default: 8)\n"
        "      --thickness <px>      Ring thickness; <=0 means filled disk (default: 2)\n"
        "\n", prog);
}

static bool ends_with_ci(const std::string &s, const std::string &suffix)
{
    if (suffix.size() > s.size()) return false;
    const size_t off = s.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); ++i) {
        const char a = (char)std::tolower((unsigned char)s[off + i]);
        const char b = (char)std::tolower((unsigned char)suffix[i]);
        if (a != b) return false;
    }
    return true;
}

static std::string trim(std::string s)
{
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char)s[a])) ++a;
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

static std::string basename_no_ext(const std::string &path)
{
    const size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name.empty() ? "frame" : name;
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

static bool parse_color(const char *s, CircleColor *out)
{
    if (!s || !out) return false;
    if (std::strcmp(s, "red") == 0)      { *out = {1.f, 0.f, 0.f}; return true; }
    if (std::strcmp(s, "green") == 0)    { *out = {0.f, 1.f, 0.f}; return true; }
    if (std::strcmp(s, "blue") == 0)     { *out = {0.f, 0.f, 1.f}; return true; }
    if (std::strcmp(s, "yellow") == 0)   { *out = {1.f, 1.f, 0.f}; return true; }
    if (std::strcmp(s, "white") == 0)    { *out = {1.f, 1.f, 1.f}; return true; }
    if (std::strcmp(s, "cyan") == 0)     { *out = {0.f, 1.f, 1.f}; return true; }
    if (std::strcmp(s, "magenta") == 0)  { *out = {1.f, 0.f, 1.f}; return true; }

    std::stringstream ss(s);
    std::string tok;
    float vals[3] = {0.f, 0.f, 0.f};
    int n = 0;
    while (std::getline(ss, tok, ',') && n < 3) {
        vals[n++] = std::strtof(trim(tok).c_str(), nullptr);
    }
    if (n != 3) return false;
    for (int i = 0; i < 3; ++i) {
        if (!(vals[i] >= 0.f && vals[i] <= 1.f)) return false;
    }
    *out = {vals[0], vals[1], vals[2]};
    return true;
}

static void draw_circle(float *r, float *g, float *b,
                        int W, int H, float cx, float cy,
                        int radius, int thickness,
                        float rv, float gv, float bv)
{
    if (!r || !g || !b || radius <= 0) return;

    const float r_out = (float)radius + (thickness > 0 ? 0.5f * (float)thickness : 0.0f);
    const float r_in  = (thickness > 0) ? fmaxf(0.f, (float)radius - 0.5f * (float)thickness) : 0.f;
    const float r_out2 = r_out * r_out;
    const float r_in2  = r_in  * r_in;

    const int min_x = (int)floorf(cx - r_out);
    const int max_x = (int)ceilf (cx + r_out);
    const int min_y = (int)floorf(cy - r_out);
    const int max_y = (int)ceilf (cy + r_out);

    for (int y = min_y; y <= max_y; ++y) {
        if (y < 0 || y >= H) continue;
        for (int x = min_x; x <= max_x; ++x) {
            if (x < 0 || x >= W) continue;
            const float dx = (float)x - cx;
            const float dy = (float)y - cy;
            const float d2 = dx * dx + dy * dy;
            if (d2 <= r_out2 && d2 >= r_in2) {
                const int idx = y * W + x;
                r[idx] = rv;
                g[idx] = gv;
                b[idx] = bv;
            }
        }
    }
}

static bool load_inputs(const std::string &input_path, std::vector<std::string> *out_paths)
{
    if (!out_paths) return false;

    if (ends_with_ci(input_path, ".csv")) {
        FrameInfo *frames = nullptr;
        int n_frames = 0;
        if (csv_parse(input_path.c_str(), &frames, &n_frames) != DSO_OK || !frames || n_frames <= 0) {
            return false;
        }
        out_paths->reserve((size_t)n_frames);
        for (int i = 0; i < n_frames; ++i) out_paths->push_back(frames[i].filepath);
        std::free(frames);
        return !out_paths->empty();
    }

    if (ends_with_ci(input_path, ".fits") || ends_with_ci(input_path, ".fit") || ends_with_ci(input_path, ".fts")) {
        out_paths->push_back(input_path);
        return true;
    }

    std::ifstream f(input_path.c_str());
    if (!f) return false;

    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t comma = line.find(',');
        if (comma != std::string::npos) line = trim(line.substr(0, comma));
        if (!line.empty()) out_paths->push_back(line);
    }
    return !out_paths->empty();
}

static std::string make_output_path(const std::string &out_arg,
                                    const std::string &input_file,
                                    int idx, int total)
{
    if (total == 1) {
        if (ends_with_ci(out_arg, ".png")) return out_arg;
        return out_arg + ".png";
    }
    std::string prefix = out_arg;
    if (ends_with_ci(prefix, ".png")) prefix = prefix.substr(0, prefix.size() - 4);
    return prefix + "_" + std::to_string(idx) + "_" + basename_no_ext(input_file) + ".png";
}

int main(int argc, char **argv)
{
    enum {
        OPT_CPU = 256,
        OPT_GPU,
        OPT_STAR_SIGMA,
        OPT_MOFFAT_ALPHA,
        OPT_MOFFAT_BETA,
        OPT_TOP_STARS,
        OPT_BAYER,
        OPT_CIRCLE_COLOR,
        OPT_CIRCLE_RADIUS,
        OPT_THICKNESS
    };

    const char *input = nullptr;
    const char *output = "stars";
    bool use_gpu = false;
    float star_sigma = 3.0f;
    MoffatParams moffat = {2.5f, 2.0f};
    int top_stars = 200;
    int circle_radius = 8;
    int thickness = 2;
    CircleColor color = {1.f, 0.f, 0.f};
    bool bayer_override_set = false;
    BayerPattern bayer_override = BAYER_NONE;

    static struct option long_opts[] = {
        {"file",          required_argument, nullptr, 'f'},
        {"output",        required_argument, nullptr, 'o'},
        {"cpu",           no_argument,       nullptr, OPT_CPU},
        {"gpu",           no_argument,       nullptr, OPT_GPU},
        {"star-sigma",    required_argument, nullptr, OPT_STAR_SIGMA},
        {"moffat-alpha",  required_argument, nullptr, OPT_MOFFAT_ALPHA},
        {"moffat-beta",   required_argument, nullptr, OPT_MOFFAT_BETA},
        {"top-stars",     required_argument, nullptr, OPT_TOP_STARS},
        {"bayer",         required_argument, nullptr, OPT_BAYER},
        {"circle-color",  required_argument, nullptr, OPT_CIRCLE_COLOR},
        {"circle-radius", required_argument, nullptr, OPT_CIRCLE_RADIUS},
        {"thickness",     required_argument, nullptr, OPT_THICKNESS},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:o:", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'f': input = optarg; break;
        case 'o': output = optarg; break;
        case OPT_CPU: use_gpu = false; break;
        case OPT_GPU: use_gpu = true; break;
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
        case OPT_CIRCLE_COLOR:
            if (!parse_color(optarg, &color)) {
                std::fprintf(stderr, "Error: invalid --circle-color '%s'\n", optarg);
                return 1;
            }
            break;
        case OPT_CIRCLE_RADIUS: circle_radius = std::atoi(optarg); break;
        case OPT_THICKNESS: thickness = std::atoi(optarg); break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (!input) {
        std::fprintf(stderr, "Error: --file is required\n\n");
        usage(argv[0]);
        return 1;
    }
    if (star_sigma <= 0.f || moffat.alpha <= 0.f || moffat.beta <= 0.f ||
        top_stars <= 0 || circle_radius <= 0) {
        std::fprintf(stderr, "Error: invalid numeric option\n");
        return 1;
    }

    std::vector<std::string> files;
    if (!load_inputs(input, &files)) {
        std::fprintf(stderr, "Error: failed to parse input list '%s'\n", input);
        return 1;
    }

    ImageSaveOptions opts = {};
    opts.bit_depth = OUT_BITS_INT8;
    opts.stretch_min = NAN;
    opts.stretch_max = NAN;

    int ok_count = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        const std::string &fits_path = files[i];

        Image raw = {};
        Image lum = {};
        Image conv = {};
        Image r = {}, g = {}, b = {};
        StarList stars = {};
        std::vector<uint8_t> mask;

        DsoError err = fits_load(fits_path.c_str(), &raw);
        if (err != DSO_OK) {
            std::fprintf(stderr, "[%zu/%zu] fits_load failed: %s (err=%d)\n",
                         i + 1, files.size(), fits_path.c_str(), (int)err);
            goto frame_cleanup;
        }

        BayerPattern pat = BAYER_NONE;
        if (bayer_override_set) pat = bayer_override;
        else {
            const DsoError berr = fits_get_bayer_pattern(fits_path.c_str(), &pat);
            if (berr != DSO_OK) {
                std::fprintf(stderr,
                             "Warning: fits_get_bayer_pattern failed for '%s' (err=%d); using BAYER_NONE\n",
                             fits_path.c_str(), (int)berr);
                pat = BAYER_NONE;
            }
        }

        lum.width = raw.width; lum.height = raw.height;
        conv.width = raw.width; conv.height = raw.height;
        r.width = g.width = b.width = raw.width;
        r.height = g.height = b.height = raw.height;

        lum.data  = (float *)std::malloc((size_t)raw.width * raw.height * sizeof(float));
        conv.data = (float *)std::malloc((size_t)raw.width * raw.height * sizeof(float));
        r.data    = (float *)std::malloc((size_t)raw.width * raw.height * sizeof(float));
        g.data    = (float *)std::malloc((size_t)raw.width * raw.height * sizeof(float));
        b.data    = (float *)std::malloc((size_t)raw.width * raw.height * sizeof(float));
        if (!lum.data || !conv.data || !r.data || !g.data || !b.data) {
            std::fprintf(stderr, "Allocation failed for %s\n", fits_path.c_str());
            goto frame_cleanup;
        }
        mask.resize((size_t)raw.width * raw.height);

        err = debayer_cpu(raw.data, lum.data, raw.width, raw.height, pat);
        if (err != DSO_OK) {
            std::fprintf(stderr, "debayer_cpu failed: %s (err=%d)\n", fits_path.c_str(), (int)err);
            goto frame_cleanup;
        }

        if (use_gpu) {
            const cudaStream_t stream = (cudaStream_t)0;
            err = star_detect_gpu_moffat_convolve(&lum, &conv, &moffat, stream);
            if (err == DSO_OK) {
                err = star_detect_gpu_threshold(&conv, mask.data(), star_sigma, stream);
            }
        } else {
            err = star_detect_cpu_detect(lum.data, conv.data, mask.data(),
                                         raw.width, raw.height, &moffat, star_sigma);
        }
        if (err != DSO_OK) {
            std::fprintf(stderr, "star detection failed: %s (err=%d)\n", fits_path.c_str(), (int)err);
            goto frame_cleanup;
        }

        err = star_detect_cpu_ccl_com(mask.data(), lum.data, conv.data,
                                      raw.width, raw.height, top_stars, &stars);
        if (err != DSO_OK) {
            std::fprintf(stderr, "CCL+CoM failed: %s (err=%d)\n", fits_path.c_str(), (int)err);
            goto frame_cleanup;
        }

        float min_v = lum.data[0];
        float max_v = lum.data[0];
        const int npix = raw.width * raw.height;
        for (int p = 0; p < npix; ++p) {
            const float v = lum.data[p];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            r.data[p] = v;
            g.data[p] = v;
            b.data[p] = v;
        }

        const float span = (max_v > min_v) ? (max_v - min_v) : 1.0f;
        const float hi = max_v + 0.15f * span;
        const float rv = min_v + color.r * (hi - min_v);
        const float gv = min_v + color.g * (hi - min_v);
        const float bv = min_v + color.b * (hi - min_v);

        for (int s = 0; s < stars.n; ++s) {
            draw_circle(r.data, g.data, b.data,
                        raw.width, raw.height,
                        stars.stars[s].x, stars.stars[s].y,
                        circle_radius, thickness, rv, gv, bv);
        }

        {
            std::string out_path = make_output_path(output, fits_path, (int)i, (int)files.size());
            err = image_save_rgb(out_path.c_str(), &r, &g, &b, &opts);
            if (err != DSO_OK) {
                std::fprintf(stderr, "image_save_rgb failed: %s (err=%d)\n", out_path.c_str(), (int)err);
                goto frame_cleanup;
            }
            std::printf("[%zu/%zu] %s: %d stars -> %s\n",
                        i + 1, files.size(), fits_path.c_str(), stars.n, out_path.c_str());
            ++ok_count;
        }

frame_cleanup:
        if (stars.stars) std::free(stars.stars);
        image_free(&raw);
        image_free(&lum);
        image_free(&conv);
        image_free(&r);
        image_free(&g);
        image_free(&b);
    }

    if (ok_count == 0) {
        std::fprintf(stderr, "No frames processed successfully.\n");
        return 1;
    }

    std::printf("Done. Processed %d/%zu frame(s).\n", ok_count, files.size());
    return 0;
}
