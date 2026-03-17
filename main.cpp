/*
 * main.cpp — DSO Stacker CLI
 *
 * Entry point for the gpu-dso-stacker pipeline. Parses a CSV describing
 * input FITS frames, optionally runs star detection and RANSAC alignment
 * (when the CSV has no pre-computed transforms), then applies Lanczos-3
 * alignment and GPU mini-batch kappa-sigma integration via pipeline_run().
 *
 * Usage:
 *   dso_stacker -f <frames.csv> [options]
 *
 * Options (existing):
 *   -f, --file <path>                Input CSV file (required)
 *   -o, --output <path>              Output FITS file (default: output.fits)
 *       --cpu                        Use CPU Lanczos (default: GPU)
 *       --integration <method>       mean | kappa-sigma (default: kappa-sigma)
 *       --kappa <float>              Sigma multiplier for clipping (default: 3.0)
 *       --iterations <int>           Max clipping iterations (default: 3)
 *
 * Options (new — star detection and alignment):
 *       --star-sigma <float>         Detection threshold: accept pixels above
 *                                    mean + star_sigma × σ (default: 3.0)
 *       --moffat-alpha <float>       Moffat PSF alpha (FWHM control, default: 2.5)
 *       --moffat-beta <float>        Moffat PSF beta (wing slope, default: 2.0)
 *       --top-stars <int>            Top-K stars to use for matching (default: 50)
 *       --min-stars <int>            Minimum stars required for RANSAC (default: 6)
 *       --ransac-iters <int>         Max RANSAC iterations (default: 1000)
 *       --ransac-thresh <float>      Inlier reprojection threshold in px (default: 2.0)
 *       --match-radius <float>       Star matching search radius in px (default: 30.0)
 *       --batch-size <int>           Frames per GPU integration mini-batch (default: 16)
 *       --bayer <pattern>            Bayer override: none | rggb | bggr | grbg | gbrg
 *                                    (default: auto-detect from FITS header)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include "dso_types.h"
#include "csv_parser.h"
#include "integration_gpu.h"   /* INTEGRATION_GPU_MAX_BATCH */
#include "pipeline.h"

/* -------------------------------------------------------------------------
 * Long-option enum values (avoid collision with ASCII short-option chars)
 * ------------------------------------------------------------------------- */
enum {
    OPT_CPU          = 256,
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
    OPT_BATCH_SIZE,
    OPT_BAYER,
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
        "      --cpu                      Use CPU Lanczos instead of GPU\n"
        "      --integration <method>     mean | kappa-sigma (default: kappa-sigma)\n"
        "      --kappa <float>            Sigma clipping threshold (default: 3.0)\n"
        "      --iterations <int>         Max clipping passes per pixel (default: 3)\n"
        "      --batch-size <int>         GPU integration mini-batch size (default: 16)\n"
        "\n"
        "Star detection (used only for 2-column CSV input):\n"
        "      --star-sigma <float>       Detection threshold in σ units (default: 3.0)\n"
        "      --moffat-alpha <float>     Moffat PSF alpha / FWHM (default: 2.5)\n"
        "      --moffat-beta <float>      Moffat PSF beta / wing slope (default: 2.0)\n"
        "      --top-stars <int>          Top-K stars for matching (default: 50)\n"
        "      --min-stars <int>          Minimum stars for RANSAC (default: 6)\n"
        "\n"
        "RANSAC alignment (used only for 2-column CSV input):\n"
        "      --ransac-iters <int>       Max RANSAC iterations (default: 1000)\n"
        "      --ransac-thresh <float>    Inlier reprojection threshold px (default: 2.0)\n"
        "      --match-radius <float>     Star matching radius px (default: 30.0)\n"
        "\n"
        "Sensor:\n"
        "      --bayer <pattern>          CFA override: none | rggb | bggr | grbg | gbrg\n"
        "                                 (default: auto-detect from FITS BAYERPAT keyword)\n"
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

    /* Pipeline config defaults */
    PipelineConfig cfg = {};
    cfg.star_sigma      = 3.0f;
    cfg.moffat          = {2.5f, 2.0f};
    cfg.top_stars       = 50;
    cfg.min_stars       = 6;
    cfg.ransac          = {1000, 2.0f, 30.0f, 0.99f, 4};
    cfg.batch_size      = 16;
    cfg.kappa           = 3.0f;
    cfg.iterations      = 3;
    cfg.use_kappa_sigma = 1;
    cfg.output_file     = "output.fits";
    cfg.bayer_override  = BAYER_NONE;   /* auto-detect */
    cfg.use_gpu_lanczos = 1;

    static struct option long_opts[] = {
        {"file",          required_argument, nullptr, 'f'},
        {"output",        required_argument, nullptr, 'o'},
        {"cpu",           no_argument,       nullptr, OPT_CPU},
        {"integration",   required_argument, nullptr, OPT_INTEGRATION},
        {"kappa",         required_argument, nullptr, OPT_KAPPA},
        {"iterations",    required_argument, nullptr, OPT_ITERATIONS},
        {"star-sigma",    required_argument, nullptr, OPT_STAR_SIGMA},
        {"moffat-alpha",  required_argument, nullptr, OPT_MOFFAT_ALPHA},
        {"moffat-beta",   required_argument, nullptr, OPT_MOFFAT_BETA},
        {"top-stars",     required_argument, nullptr, OPT_TOP_STARS},
        {"min-stars",     required_argument, nullptr, OPT_MIN_STARS},
        {"ransac-iters",  required_argument, nullptr, OPT_RANSAC_ITERS},
        {"ransac-thresh", required_argument, nullptr, OPT_RANSAC_THRESH},
        {"match-radius",  required_argument, nullptr, OPT_MATCH_RADIUS},
        {"batch-size",    required_argument, nullptr, OPT_BATCH_SIZE},
        {"bayer",         required_argument, nullptr, OPT_BAYER},
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

        case OPT_RANSAC_ITERS:  cfg.ransac.max_iters     = atoi(optarg);          break;
        case OPT_RANSAC_THRESH: cfg.ransac.inlier_thresh  = strtof(optarg, nullptr); break;
        case OPT_MATCH_RADIUS:  cfg.ransac.match_radius   = strtof(optarg, nullptr); break;

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

        default: usage(argv[0]); return 1;
        }
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

    /* Wire output path into config */
    cfg.output_file     = output_file;
    cfg.use_gpu_lanczos = !use_cpu;

    /* ---- Parse CSV ---- */
    FrameInfo *frames        = nullptr;
    int        n_frames      = 0;
    int        has_transforms = 0;
    check(csv_parse(csv_file, &frames, &n_frames, &has_transforms), "csv_parse");

    if (n_frames == 0) {
        fprintf(stderr, "Error: CSV contains no frames\n");
        return 1;
    }

    /* Validate exactly one reference frame */
    int ref_idx = -1;
    for (int i = 0; i < n_frames; i++) {
        if (frames[i].is_reference) {
            if (ref_idx != -1) {
                fprintf(stderr, "Error: multiple reference frames found\n");
                return 1;
            }
            ref_idx = i;
        }
    }
    if (ref_idx == -1) {
        fprintf(stderr, "Error: no reference frame found in CSV\n");
        return 1;
    }

    printf("Parsed %d frame(s), reference = %d, %s\n",
           n_frames, ref_idx,
           has_transforms ? "pre-computed transforms" : "star detection mode");
    printf("Integration: %s (kappa=%.1f, iter=%d, batch=%d) | Lanczos: %s\n",
           integ_str, (double)cfg.kappa, cfg.iterations, cfg.batch_size,
           use_cpu ? "CPU" : "GPU");

    /* ---- Run pipeline ---- */
    check(pipeline_run(frames, n_frames, has_transforms, ref_idx, &cfg),
          "pipeline_run");

    free(frames);
    return 0;
}
