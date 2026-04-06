/*
 * main.cpp — DSO Stacker CLI
 *
 * Entry point for the gpu-dso-stacker pipeline. Parses a CSV describing
 * input FITS frames, optionally runs star detection and triangle matching
 * (when the CSV has no pre-computed transforms), then applies Lanczos-3
 * alignment and GPU mini-batch kappa-sigma integration via pipeline_run().
 *
 * Usage:
 *   dso_stacker -f <frames.csv> [options]
 *
 * Options (I/O):
 *   -f, --file <path>                Input CSV file (required)
 *   -o, --output <path>              Output FITS file (default: output.fits)
 *
 * Options (integration):
 *       --cpu                        Run ALL pipeline stages on CPU (default: GPU)
 *       --integration <method>       mean | kappa-sigma (default: kappa-sigma)
 *       --kappa <float>              Sigma multiplier for clipping (default: 3.0)
 *       --iterations <int>           Max clipping iterations (default: 3)
 *       --batch-size <int>           GPU integration mini-batch size (default: 16)
 *
 * Options (star detection — 2-column CSV only):
 *       --star-sigma <float>         Detection threshold: accept pixels above
 *                                    mean + star_sigma × σ (default: 3.0)
 *       --moffat-alpha <float>       Moffat PSF alpha (FWHM control, default: 2.5)
 *       --moffat-beta <float>        Moffat PSF beta (wing slope, default: 2.0)
 *       --top-stars <int>            Top-K stars to use for matching (default: 50)
 *       --min-stars <int>            Minimum stars required for triangle matching (default: 6)
 *       --triangle-iters <int>       Max triangle-matching iterations (default: 1000)
 *       --triangle-thresh <float>    Inlier reprojection threshold in px (default: 2.0)
 *       --match-radius <float>       Star matching search radius in px (default: 30.0)
 *       --match-device <device>      auto | cpu | gpu (default: auto)
 *
 * Options (calibration):
 *       --dark <path>                Master dark FITS or list of dark FITS paths
 *       --bias <path>                Master bias FITS or list of bias FITS paths
 *                                    (mutually exclusive with --darkflat)
 *       --flat <path>                Master flat FITS or list of flat FITS paths
 *       --darkflat <path>            Master darkflat FITS or list of darkflat FITS paths
 *                                    (mutually exclusive with --bias)
 *       --save-master-frames <dir>   Where to save generated masters (default: ./master)
 *       --dark-method <method>       winsorized-mean | median (default: winsorized-mean)
 *       --bias-method <method>       winsorized-mean | median (default: winsorized-mean)
 *       --flat-method <method>       winsorized-mean | median (default: winsorized-mean)
 *       --darkflat-method <method>   winsorized-mean | median (default: winsorized-mean)
 *       --wsor-clip <float>          Winsorized mean clipping fraction per side (default: 0.1)
 *                                    Valid range: [0.0, 0.49]
 *
 * Options (sensor):
 *       --bayer <pattern>            CFA override: none | rggb | bggr | grbg | gbrg
 *                                    (default: auto-detect from FITS header)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#ifdef _MSC_VER
#include "compat.h"
#else
#include <getopt.h>
#endif

#include "dso_types.h"
#include "csv_parser.h"
#include "calibration.h"
#include "fits_io.h"           /* image_free */
#include "frame_load.h"        /* frame_get_bayer_pattern */
#include "image_io.h"          /* ImageSaveOptions, image_detect_format */
#include "integration_limits.h"/* INTEGRATION_GPU_MAX_BATCH */
#include "pipeline.h"

/* -------------------------------------------------------------------------
 * Long-option enum values (avoid collision with ASCII short-option chars)
 * ------------------------------------------------------------------------- */
enum {
    OPT_CPU             = 256,
    OPT_INTEGRATION,
    OPT_KAPPA,
    OPT_ITERATIONS,
    OPT_STAR_SIGMA,
    OPT_MOFFAT_ALPHA,
    OPT_MOFFAT_BETA,
    OPT_TOP_STARS,
    OPT_MIN_STARS,
    OPT_RANSAC_ITERS,
    OPT_RANSAC_THRESH,
    OPT_MATCH_RADIUS,
    OPT_MIN_INLIERS,
    OPT_MATCH_DEVICE,
    OPT_BACKEND,
    OPT_BATCH_SIZE,
    OPT_BAYER,
    /* Calibration options */
    OPT_DARK,
    OPT_BIAS,
    OPT_FLAT,
    OPT_DARKFLAT,
    OPT_SAVE_MASTER_FRAMES,
    OPT_DARK_METHOD,
    OPT_BIAS_METHOD,
    OPT_FLAT_METHOD,
    OPT_DARKFLAT_METHOD,
    OPT_WSOR_CLIP,
    OPT_CALIB_KAPPA,
    OPT_CALIB_ITERATIONS,
    /* Output format options */
    OPT_BIT_DEPTH,
    OPT_TIFF_COMPRESSION,
    OPT_STRETCH_MIN,
    OPT_STRETCH_MAX,
    /* Query */
    OPT_LIST_BACKENDS,
};

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -f <frames.csv> [options]\n"
        "\n"
        "I/O:\n"
        "  -f, --file <path>              Input CSV file (required)\n"
        "  -o, --output <path>            Output FITS file (default: output.fits)\n"
        "\n"
        "Integration:\n"
        "      --cpu                      Run ALL pipeline stages on CPU (OpenMP-accelerated)\n"
        "      --integration <method>     mean | kappa-sigma | auto-adaptive\n"
        "                                 (default: kappa-sigma)\n"
        "                                 auto-adaptive: Stetson (1989) iterative\n"
        "                                 weighted average; down-weights outliers\n"
        "                                 instead of hard-rejecting them\n"
        "      --kappa <float>            Sigma clipping threshold (default: 3.0)\n"
        "      --iterations <int>         Max clipping passes per pixel (default: 3)\n"
        "      --batch-size <int>         GPU integration mini-batch size (default: 16)\n"
        "\n"
        "Star detection (used only for 2-column CSV input):\n"
        "      --star-sigma <float>       Detection threshold in σ units (default: 3.0)\n"
        "      --moffat-alpha <float>     Moffat PSF alpha / FWHM (default: 2.5)\n"
        "      --moffat-beta <float>      Moffat PSF beta / wing slope (default: 2.0)\n"
        "      --top-stars <int>          Top-K stars for matching (default: 50)\n"
        "      --min-stars <int>          Minimum detected stars to attempt alignment (default: 20)\n"
        "\n"
        "Triangle matching alignment (used only for 2-column CSV input):\n"
        "      --triangle-iters <int>     Max triangle-matching iterations (default: 1000)\n"
        "      --triangle-thresh <float>  Inlier reprojection threshold px (default: 2.0)\n"
        "      --ransac-iters <int>       Deprecated alias of --triangle-iters\n"
        "      --ransac-thresh <float>    Deprecated alias of --triangle-thresh\n"
        "      --match-radius <float>     Star matching radius px (default: 30.0)\n"
        "      --min-inliers <int>        Minimum RANSAC inliers for acceptance (default: 10)\n"
        "      --match-device <device>    auto | cpu | gpu (default: auto = stacking device)\n"
        "      --backend <backend>        auto | cpu"
#if defined(DSO_HAS_CUDA) && DSO_HAS_CUDA
        " | cuda"
#endif
#if defined(DSO_HAS_METAL) && DSO_HAS_METAL
        " | metal"
#endif
        " (default: auto)\n"
        "\n"
        "Calibration:\n"
        "      --dark <path>              Master dark FITS or text list of dark FITS paths\n"
        "      --bias <path>              Master bias FITS or text list of bias FITS paths\n"
        "                                 (mutually exclusive with --darkflat)\n"
        "      --flat <path>              Master flat FITS or text list of flat FITS paths\n"
        "      --darkflat <path>          Master darkflat FITS or text list of darkflat paths\n"
        "                                 (mutually exclusive with --bias)\n"
        "      --save-master-frames <dir> Directory to save generated master frames\n"
        "                                 (default: ./master)\n"
        "      --dark-method <method>     winsorized-mean | median | kappa-sigma\n"
        "                                 (default: winsorized-mean)\n"
        "      --bias-method <method>     winsorized-mean | median | kappa-sigma\n"
        "                                 (default: winsorized-mean)\n"
        "      --flat-method <method>     winsorized-mean | median | kappa-sigma\n"
        "                                 (default: winsorized-mean)\n"
        "      --darkflat-method <method> winsorized-mean | median | kappa-sigma\n"
        "                                 (default: winsorized-mean)\n"
        "      --wsor-clip <float>        Winsorized mean clipping fraction per side\n"
        "                                 Valid range: [0.0, 0.49] (default: 0.1)\n"
        "      --calib-kappa <float>      Kappa-sigma rejection threshold for calibration\n"
        "                                 stacking (default: 2.5)\n"
        "      --calib-iterations <int>   Max kappa-sigma clipping passes for calibration\n"
        "                                 stacking (default: 5)\n"
        "\n"
        "Sensor:\n"
        "      --bayer <pattern>          CFA override: none | rggb | bggr | grbg | gbrg\n"
        "                                 (default: auto-detect from FITS BAYERPAT keyword)\n"
        "\n"
        "Output format (format inferred from --output extension):\n"
        "      --bit-depth <depth>        8 | 16 | f16 | f32  (default: f32)\n"
        "                                 f16 is TIFF only; 8/16 require TIFF or PNG;\n"
        "                                 FITS always uses f32 regardless of this flag\n"
        "      --tiff-compression <c>     none | zip | lzw | rle  (default: none; TIFF only)\n"
        "      --stretch-min <float>      Lower bound for integer scaling (default: auto)\n"
        "      --stretch-max <float>      Upper bound for integer scaling (default: auto)\n"
        "                                 stretch-min/max are ignored for f16/f32 output\n"
        "\n",
        prog);
}

static void check(DsoError err, const char *ctx)
{
    if (err != DSO_OK) {
        fprintf(stderr, "Error in %s: code %d\n", ctx, (int)err);
        exit(1);
    }
}

/* Parse a method string → CalibMethod; returns -1 on unknown value */
static int parse_calib_method(const char *s, CalibMethod *out)
{
    if (strcmp(s, "winsorized-mean") == 0) { *out = CALIB_WINSORIZED_MEAN; return 0; }
    if (strcmp(s, "median")          == 0) { *out = CALIB_MEDIAN;          return 0; }
    if (strcmp(s, "kappa-sigma")     == 0) { *out = CALIB_KAPPA_SIGMA;     return 0; }
    return -1;
}

static const char *backend_name(DsoBackend backend)
{
    switch (backend) {
    case DSO_BACKEND_CPU:   return "cpu";
    case DSO_BACKEND_CUDA:  return "cuda";
    case DSO_BACKEND_METAL: return "metal";
    case DSO_BACKEND_AUTO:
    default:                return "auto";
    }
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    /* ---- Defaults ---- */
    const char *csv_file    = nullptr;
    const char *output_file = "output.fits";
    bool        use_cpu     = false;
    const char *integ_str   = "kappa-sigma";
    bool        backend_explicit = false;
    bool        list_backends    = false;

    /* Pipeline config defaults */
    PipelineConfig cfg = {};
    cfg.star_sigma      = 3.0f;
    cfg.moffat          = {2.5f, 2.0f};
    cfg.top_stars       = 50;
    cfg.min_stars       = 20;
    cfg.ransac          = {1000, 2.0f, 30.0f, 0.99f, 10};
    cfg.batch_size      = 16;
    cfg.kappa           = 3.0f;
    cfg.iterations      = 3;
    cfg.use_kappa_sigma = 1;
    cfg.output_file     = "output.fits";
    cfg.backend         = DSO_BACKEND_AUTO;
    cfg.bayer_override  = BAYER_NONE;   /* auto-detect */
    cfg.use_gpu_lanczos = 1;
    cfg.use_gpu_ransac  = -1; /* auto: follow stacking device */
    cfg.calib           = nullptr;
    /* save_opts defaults: FP32, no compression, auto stretch (NAN) */
    cfg.save_opts.tiff_compress = TIFF_COMPRESS_NONE;
    cfg.save_opts.bit_depth     = OUT_BITS_FP32;
    cfg.save_opts.stretch_min   = (float)NAN;
    cfg.save_opts.stretch_max   = (float)NAN;

    /* Calibration defaults */
    const char *dark_path        = nullptr;
    const char *bias_path        = nullptr;
    const char *flat_path        = nullptr;
    const char *darkflat_path    = nullptr;
    const char *save_master_dir  = "./master";
    CalibMethod dark_method      = CALIB_WINSORIZED_MEAN;
    CalibMethod bias_method      = CALIB_WINSORIZED_MEAN;
    CalibMethod flat_method      = CALIB_WINSORIZED_MEAN;
    CalibMethod darkflat_method  = CALIB_WINSORIZED_MEAN;
    float       wsor_clip        = 0.1f;
    float       calib_kappa      = 2.5f;
    int         calib_iterations = 5;

    static struct option long_opts[] = {
        {"file",              required_argument, nullptr, 'f'},
        {"output",            required_argument, nullptr, 'o'},
        {"cpu",               no_argument,       nullptr, OPT_CPU},
        {"integration",       required_argument, nullptr, OPT_INTEGRATION},
        {"kappa",             required_argument, nullptr, OPT_KAPPA},
        {"iterations",        required_argument, nullptr, OPT_ITERATIONS},
        {"star-sigma",        required_argument, nullptr, OPT_STAR_SIGMA},
        {"moffat-alpha",      required_argument, nullptr, OPT_MOFFAT_ALPHA},
        {"moffat-beta",       required_argument, nullptr, OPT_MOFFAT_BETA},
        {"top-stars",         required_argument, nullptr, OPT_TOP_STARS},
        {"min-stars",         required_argument, nullptr, OPT_MIN_STARS},
        {"triangle-iters",    required_argument, nullptr, OPT_RANSAC_ITERS},
        {"triangle-thresh",   required_argument, nullptr, OPT_RANSAC_THRESH},
        {"ransac-iters",      required_argument, nullptr, OPT_RANSAC_ITERS},  /* deprecated alias */
        {"ransac-thresh",     required_argument, nullptr, OPT_RANSAC_THRESH}, /* deprecated alias */
        {"match-radius",      required_argument, nullptr, OPT_MATCH_RADIUS},
        {"min-inliers",       required_argument, nullptr, OPT_MIN_INLIERS},
        {"match-device",      required_argument, nullptr, OPT_MATCH_DEVICE},
        {"backend",           required_argument, nullptr, OPT_BACKEND},
        {"batch-size",        required_argument, nullptr, OPT_BATCH_SIZE},
        {"bayer",             required_argument, nullptr, OPT_BAYER},
        /* Calibration */
        {"dark",              required_argument, nullptr, OPT_DARK},
        {"bias",              required_argument, nullptr, OPT_BIAS},
        {"flat",              required_argument, nullptr, OPT_FLAT},
        {"darkflat",          required_argument, nullptr, OPT_DARKFLAT},
        {"save-master-frames",required_argument, nullptr, OPT_SAVE_MASTER_FRAMES},
        {"dark-method",       required_argument, nullptr, OPT_DARK_METHOD},
        {"bias-method",       required_argument, nullptr, OPT_BIAS_METHOD},
        {"flat-method",       required_argument, nullptr, OPT_FLAT_METHOD},
        {"darkflat-method",   required_argument, nullptr, OPT_DARKFLAT_METHOD},
        {"wsor-clip",         required_argument, nullptr, OPT_WSOR_CLIP},
        {"calib-kappa",       required_argument, nullptr, OPT_CALIB_KAPPA},
        {"calib-iterations",  required_argument, nullptr, OPT_CALIB_ITERATIONS},
        /* Output format */
        {"bit-depth",         required_argument, nullptr, OPT_BIT_DEPTH},
        {"tiff-compression",  required_argument, nullptr, OPT_TIFF_COMPRESSION},
        {"stretch-min",       required_argument, nullptr, OPT_STRETCH_MIN},
        {"stretch-max",       required_argument, nullptr, OPT_STRETCH_MAX},
        {"list-backends",     no_argument,       nullptr, OPT_LIST_BACKENDS},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:o:", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'f': csv_file    = optarg; break;
        case 'o': output_file = optarg; break;

        case OPT_CPU:          use_cpu     = true;                             break;
        case OPT_INTEGRATION:  integ_str   = optarg;                          break;
        case OPT_KAPPA:        cfg.kappa   = strtof(optarg, nullptr);         break;
        case OPT_ITERATIONS:   cfg.iterations = atoi(optarg);                 break;

        case OPT_STAR_SIGMA:   cfg.star_sigma        = strtof(optarg, nullptr); break;
        case OPT_MOFFAT_ALPHA: cfg.moffat.alpha      = strtof(optarg, nullptr); break;
        case OPT_MOFFAT_BETA:  cfg.moffat.beta       = strtof(optarg, nullptr); break;
        case OPT_TOP_STARS:    cfg.top_stars         = atoi(optarg);            break;
        case OPT_MIN_STARS:    cfg.min_stars         = atoi(optarg);            break;

        case OPT_RANSAC_ITERS:  cfg.ransac.max_iters     = atoi(optarg);             break;
        case OPT_RANSAC_THRESH: cfg.ransac.inlier_thresh  = strtof(optarg, nullptr); break;
        case OPT_MATCH_RADIUS:  cfg.ransac.match_radius   = strtof(optarg, nullptr); break;
        case OPT_MIN_INLIERS:  cfg.ransac.min_inliers    = atoi(optarg);             break;
        case OPT_MATCH_DEVICE:
            if (strcmp(optarg, "auto") == 0) {
                cfg.use_gpu_ransac = -1;
            } else if (strcmp(optarg, "cpu") == 0) {
                cfg.use_gpu_ransac = 0;
            } else if (strcmp(optarg, "gpu") == 0) {
                cfg.use_gpu_ransac = 1;
            } else {
                fprintf(stderr, "Error: unknown --match-device '%s'; use auto, cpu, or gpu\n",
                        optarg);
                return 1;
            }
            break;
        case OPT_BACKEND:
            backend_explicit = true;
            if (strcmp(optarg, "auto") == 0) {
                cfg.backend = DSO_BACKEND_AUTO;
            } else if (strcmp(optarg, "cpu") == 0) {
                cfg.backend = DSO_BACKEND_CPU;
            } else if (strcmp(optarg, "cuda") == 0) {
                cfg.backend = DSO_BACKEND_CUDA;
            } else if (strcmp(optarg, "metal") == 0) {
                cfg.backend = DSO_BACKEND_METAL;
            } else {
                fprintf(stderr, "Error: unknown --backend '%s'; use auto, cpu, cuda, or metal\n",
                        optarg);
                return 1;
            }
            break;

        case OPT_BATCH_SIZE: cfg.batch_size = atoi(optarg); break;

        case OPT_BAYER:
            if      (strcmp(optarg, "none") == 0) cfg.bayer_override = BAYER_NONE;
            else if (strcmp(optarg, "rggb") == 0) cfg.bayer_override = BAYER_RGGB;
            else if (strcmp(optarg, "bggr") == 0) cfg.bayer_override = BAYER_BGGR;
            else if (strcmp(optarg, "grbg") == 0) cfg.bayer_override = BAYER_GRBG;
            else if (strcmp(optarg, "gbrg") == 0) cfg.bayer_override = BAYER_GBRG;
            else {
                fprintf(stderr, "Error: unknown Bayer pattern '%s'\n", optarg);
                return 1;
            }
            break;

        /* Calibration options */
        case OPT_DARK:               dark_path       = optarg; break;
        case OPT_BIAS:               bias_path       = optarg; break;
        case OPT_FLAT:               flat_path       = optarg; break;
        case OPT_DARKFLAT:           darkflat_path   = optarg; break;
        case OPT_SAVE_MASTER_FRAMES: save_master_dir = optarg; break;

        case OPT_DARK_METHOD:
            if (parse_calib_method(optarg, &dark_method) != 0) {
                fprintf(stderr, "Error: unknown dark method '%s'\n", optarg);
                return 1;
            }
            break;
        case OPT_BIAS_METHOD:
            if (parse_calib_method(optarg, &bias_method) != 0) {
                fprintf(stderr, "Error: unknown bias method '%s'\n", optarg);
                return 1;
            }
            break;
        case OPT_FLAT_METHOD:
            if (parse_calib_method(optarg, &flat_method) != 0) {
                fprintf(stderr, "Error: unknown flat method '%s'\n", optarg);
                return 1;
            }
            break;
        case OPT_DARKFLAT_METHOD:
            if (parse_calib_method(optarg, &darkflat_method) != 0) {
                fprintf(stderr, "Error: unknown darkflat method '%s'\n", optarg);
                return 1;
            }
            break;

        case OPT_WSOR_CLIP:
            wsor_clip = strtof(optarg, nullptr);
            if (wsor_clip < 0.0f || wsor_clip > 0.49f) {
                fprintf(stderr,
                        "Error: --wsor-clip must be in [0.0, 0.49] (got %.4f)\n",
                        (double)wsor_clip);
                return 1;
            }
            break;

        case OPT_CALIB_KAPPA:
            calib_kappa = strtof(optarg, nullptr);
            if (calib_kappa <= 0.0f) {
                fprintf(stderr,
                        "Error: --calib-kappa must be > 0 (got %.4f)\n",
                        (double)calib_kappa);
                return 1;
            }
            break;
        case OPT_CALIB_ITERATIONS:
            calib_iterations = atoi(optarg);
            if (calib_iterations < 1) {
                fprintf(stderr,
                        "Error: --calib-iterations must be >= 1 (got %d)\n",
                        calib_iterations);
                return 1;
            }
            break;

        case OPT_BIT_DEPTH:
            if      (strcmp(optarg, "8")   == 0) cfg.save_opts.bit_depth = OUT_BITS_INT8;
            else if (strcmp(optarg, "16")  == 0) cfg.save_opts.bit_depth = OUT_BITS_INT16;
            else if (strcmp(optarg, "f16") == 0) cfg.save_opts.bit_depth = OUT_BITS_FP16;
            else if (strcmp(optarg, "f32") == 0) cfg.save_opts.bit_depth = OUT_BITS_FP32;
            else {
                fprintf(stderr, "Error: unknown --bit-depth '%s'; use 8, 16, f16, or f32\n",
                        optarg);
                return 1;
            }
            break;

        case OPT_TIFF_COMPRESSION:
            if      (strcmp(optarg, "none") == 0) cfg.save_opts.tiff_compress = TIFF_COMPRESS_NONE;
            else if (strcmp(optarg, "zip")  == 0) cfg.save_opts.tiff_compress = TIFF_COMPRESS_ZIP;
            else if (strcmp(optarg, "lzw")  == 0) cfg.save_opts.tiff_compress = TIFF_COMPRESS_LZW;
            else if (strcmp(optarg, "rle")  == 0) cfg.save_opts.tiff_compress = TIFF_COMPRESS_RLE;
            else {
                fprintf(stderr,
                        "Error: unknown --tiff-compression '%s'; use none, zip, lzw, or rle\n",
                        optarg);
                return 1;
            }
            break;

        case OPT_STRETCH_MIN: cfg.save_opts.stretch_min = strtof(optarg, nullptr); break;
        case OPT_STRETCH_MAX: cfg.save_opts.stretch_max = strtof(optarg, nullptr); break;

        case OPT_LIST_BACKENDS: list_backends = true; break;

        default: usage(argv[0]); return 1;
        }
    }

    /* --list-backends: print available backends and exit (no -f required) */
    if (list_backends) {
        printf("auto\ncpu\n");
#if defined(DSO_HAS_CUDA) && DSO_HAS_CUDA
        printf("cuda\n");
#endif
#if defined(DSO_HAS_METAL) && DSO_HAS_METAL
        printf("metal\n");
#endif
        return 0;
    }

    if (!csv_file) {
        fprintf(stderr, "Error: --file is required\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Validate integration method */
    if (strcmp(integ_str, "kappa-sigma") == 0) {
        cfg.use_kappa_sigma = 1;
    } else if (strcmp(integ_str, "mean") == 0) {
        cfg.use_kappa_sigma = 0;
    } else if (strcmp(integ_str, "auto-adaptive") == 0) {
        cfg.use_kappa_sigma = 2;
    } else {
        fprintf(stderr, "Error: unknown integration method '%s'\n", integ_str);
        return 1;
    }

    /* Validate batch size */
    if (cfg.batch_size < 1 || cfg.batch_size > INTEGRATION_GPU_MAX_BATCH) {
        fprintf(stderr, "Error: --batch-size must be 1–%d\n",
                INTEGRATION_GPU_MAX_BATCH);
        return 1;
    }

    /* Validate calibration: bias and darkflat are mutually exclusive */
    if (bias_path && darkflat_path) {
        fprintf(stderr,
                "Error: --bias and --darkflat are mutually exclusive.\n"
                "  Use --bias for short flat exposures (< 1 s).\n"
                "  Use --darkflat for longer flat exposures to capture dark current.\n");
        return 1;
    }

    /* Warn when no calibration is provided */
    if (!dark_path && !bias_path && !flat_path && !darkflat_path) {
        fprintf(stderr,
                "Warning: no calibration frames provided.\n"
                "  The stacked output may contain dark current, hot pixels, "
                "and optical vignetting.\n"
                "  Consider providing --dark, --flat, and --bias/--darkflat.\n");
    }

    /* Validate output format vs. bit-depth and compression options */
    {
        OutputFormat ofmt = image_detect_format(output_file);
        if (ofmt == FMT_UNKNOWN) {
            fprintf(stderr,
                    "Error: unrecognized output file extension for '%s'.\n"
                    "  Supported: .fits, .fit, .fts, .tif, .tiff, .png\n",
                    output_file);
            return 1;
        }
        if (ofmt == FMT_FITS && cfg.save_opts.bit_depth != OUT_BITS_FP32) {
            fprintf(stderr,
                    "Error: FITS output only supports f32 bit depth.\n"
                    "  Remove --bit-depth or use a .tif / .png output path.\n");
            return 1;
        }
        if (ofmt == FMT_PNG &&
            (cfg.save_opts.bit_depth == OUT_BITS_FP32 ||
             cfg.save_opts.bit_depth == OUT_BITS_FP16)) {
            fprintf(stderr,
                    "Error: PNG output does not support f32 or f16 bit depth.\n"
                    "  Use --bit-depth 8 or --bit-depth 16.\n");
            return 1;
        }
        if (ofmt == FMT_PNG && cfg.save_opts.bit_depth == OUT_BITS_FP32) {
            /* default f32 with PNG — upgrade to INT16 silently? No: error above covers it. */
        }
        if (cfg.save_opts.bit_depth == OUT_BITS_FP16 && ofmt != FMT_TIFF) {
            fprintf(stderr,
                    "Error: f16 bit depth is only supported for TIFF output.\n"
                    "  Use a .tif / .tiff output path, or choose a different --bit-depth.\n");
            return 1;
        }
        if (cfg.save_opts.tiff_compress != TIFF_COMPRESS_NONE && ofmt != FMT_TIFF) {
            fprintf(stderr,
                    "Warning: --tiff-compression is ignored for non-TIFF output (%s).\n",
                    output_file);
        }
    }

    /* Resolve backend selection with backward-compatible semantics. */
    if (use_cpu && backend_explicit &&
        (cfg.backend == DSO_BACKEND_CUDA || cfg.backend == DSO_BACKEND_METAL)) {
        fprintf(stderr,
                "Error: --cpu cannot be combined with --backend %s.\n"
                "  Use either --cpu or --backend cpu.\n",
                backend_name(cfg.backend));
        return 1;
    }
    if (use_cpu) cfg.backend = DSO_BACKEND_CPU;

    /* Reject backends not compiled into this build */
#if !defined(DSO_HAS_CUDA) || !DSO_HAS_CUDA
    if (cfg.backend == DSO_BACKEND_CUDA) {
        fprintf(stderr,
                "Error: --backend cuda is not available in this build "
                "(compiled without CUDA support).\n"
                "  Available backends: auto, cpu"
#if defined(DSO_HAS_METAL) && DSO_HAS_METAL
                ", metal"
#endif
                "\n");
        return 1;
    }
#endif
#if !defined(DSO_HAS_METAL) || !DSO_HAS_METAL
    if (cfg.backend == DSO_BACKEND_METAL) {
        fprintf(stderr,
                "Error: --backend metal is not available in this build "
                "(compiled without Metal support).\n"
                "  Available backends: auto, cpu"
#if defined(DSO_HAS_CUDA) && DSO_HAS_CUDA
                ", cuda"
#endif
                "\n");
        return 1;
    }
#endif

    if (cfg.backend == DSO_BACKEND_AUTO) {
        if (!cfg.use_gpu_lanczos) cfg.backend = DSO_BACKEND_CPU;
#if defined(DSO_HAS_CUDA) && DSO_HAS_CUDA
        else cfg.backend = DSO_BACKEND_CUDA;
#elif defined(DSO_HAS_METAL) && DSO_HAS_METAL
        else cfg.backend = DSO_BACKEND_METAL;
#else
        else cfg.backend = DSO_BACKEND_CPU;
#endif
    }

    /* Wire output path and per-stage defaults into config */
    cfg.output_file     = output_file;
    cfg.use_gpu_lanczos = (cfg.backend != DSO_BACKEND_CPU);
    if (cfg.use_gpu_ransac < 0) cfg.use_gpu_ransac = cfg.use_gpu_lanczos;
    if (cfg.backend == DSO_BACKEND_CPU && cfg.use_gpu_ransac) {
        fprintf(stderr,
                "Error: --match-device gpu is not supported with CPU backend.\n"
                "  Use --match-device cpu, or leave it as auto.\n");
        return 1;
    }
    /* color_output is set after CSV parsing and ref_idx resolution below */

    /* ---- Generate / load calibration master frames ---- */
    CalibFrames calib   = {};
    bool        has_calib = dark_path || bias_path || flat_path || darkflat_path;

    if (has_calib) {
        check(calib_load_or_generate(
                  dark_path,    dark_method,
                  bias_path,    bias_method,
                  flat_path,    flat_method,
                  darkflat_path, darkflat_method,
                  save_master_dir,
                  wsor_clip,
                  calib_kappa,
                  calib_iterations,
                  &calib),
              "calib_load_or_generate");
        cfg.calib = &calib;
    }

    /* ---- Parse CSV ---- */
    FrameInfo *frames   = nullptr;
    int        n_frames = 0;
    check(csv_parse(csv_file, &frames, &n_frames), "csv_parse");

    if (n_frames == 0) {
        fprintf(stderr, "Error: CSV contains no frames\n");
        calib_free(&calib);
        return 1;
    }

    /* Validate exactly one reference frame */
    int ref_idx = -1;
    for (int i = 0; i < n_frames; i++) {
        if (frames[i].is_reference) {
            if (ref_idx != -1) {
                fprintf(stderr, "Error: multiple reference frames found\n");
                free(frames);
                calib_free(&calib);
                return 1;
            }
            ref_idx = i;
        }
    }
    if (ref_idx == -1) {
        fprintf(stderr, "Error: no reference frame found in CSV\n");
        free(frames);
        calib_free(&calib);
        return 1;
    }

    /* Auto-detect color mode: use bayer_override if set; otherwise peek at
     * the reference frame FITS header. Color output requires a known Bayer
     * pattern — BAYER_NONE (monochrome sensor) always produces mono output. */
    {
        BayerPattern detected = cfg.bayer_override;
        if (detected == BAYER_NONE)
            frame_get_bayer_pattern(frames[ref_idx].filepath, &detected);
        cfg.color_output = (detected != BAYER_NONE) ? 1 : 0;
    }

    const char *selected_backend = backend_name(cfg.backend);
    const char *compute_name = "CPU";
    switch (cfg.backend) {
    case DSO_BACKEND_CPU:   compute_name = "CPU";   break;
    case DSO_BACKEND_CUDA:  compute_name = "CUDA";  break;
    case DSO_BACKEND_METAL: compute_name = "METAL"; break;
    case DSO_BACKEND_AUTO:  compute_name = "AUTO";  break;
    default:                compute_name = "AUTO";  break;
    }
    printf("Parsed %d frame(s), reference = %d\n", n_frames, ref_idx);
    static const char *bit_depth_names[] = {"f32", "f16", "16-bit", "8-bit"};
    printf("Integration: %s (kappa=%.1f, iter=%d, batch=%d) | Backend: %s | Lanczos: %s | Matching: %s | Output: %s %s\n",
           integ_str, (double)cfg.kappa, cfg.iterations, cfg.batch_size,
            selected_backend,
            compute_name,
            cfg.use_gpu_ransac ? "GPU" : "CPU",
            cfg.color_output ? "color RGB" : "mono luminance",
            bit_depth_names[cfg.save_opts.bit_depth]);
    if (has_calib) {
        printf("Calibration: dark=%s flat=%s\n",
               calib.has_dark ? "yes" : "no",
               calib.has_flat ? "yes" : "no");
    }

    /* ---- Run pipeline ---- */
    check(pipeline_run(frames, n_frames, ref_idx, &cfg), "pipeline_run");

    free(frames);
    calib_free(&calib);
    return 0;
}
