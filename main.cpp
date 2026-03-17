/*
 * main.cpp — DSO Stacker CLI
 *
 * Entry point for the gpu-dso-stacker pipeline. Reads a CSV describing
 * input frames and their pre-computed homographies, aligns each frame to the
 * reference via Lanczos-3 interpolation (CPU or GPU), then integrates via
 * mean or kappa-sigma clipping.
 *
 * Usage:
 *   dso_stacker -f frames.csv [options]
 *
 * Options:
 *   -f / --file <path>          Input CSV (required)
 *   -o / --output <path>        Output FITS file (default: output.fits)
 *   --cpu                       Use CPU Lanczos (default: GPU)
 *   --integration <mean|kappa-sigma>  Integration method (default: kappa-sigma)
 *   --kappa <float>             Sigma multiplier for clipping (default: 3.0)
 *   --iterations <int>          Max clipping iterations (default: 3)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include "dso_types.h"
#include "fits_io.h"
#include "csv_parser.h"
#include "lanczos_cpu.h"
#include "lanczos_gpu.h"
#include "integration.h"

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -f <frames.csv> [options]\n"
        "\n"
        "Options:\n"
        "  -f, --file <path>              Input CSV file (required)\n"
        "  -o, --output <path>            Output FITS file (default: output.fits)\n"
        "      --cpu                      Use CPU Lanczos interpolation\n"
        "      --integration <method>     mean | kappa-sigma (default: kappa-sigma)\n"
        "      --kappa <float>            Kappa multiplier (default: 3.0)\n"
        "      --iterations <int>         Max clipping iterations (default: 3)\n"
        "\n", prog);
}

static void check(DsoError err, const char *ctx)
{
    if (err != DSO_OK) {
        fprintf(stderr, "Error in %s: code %d\n", ctx, (int)err);
        exit(1);
    }
}

/* -------------------------------------------------------------------------- */
/* main                                                                        */
/* -------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    /* ---- Parse CLI arguments ---- */
    const char *csv_file    = nullptr;
    const char *output_file = "output.fits";
    bool        use_cpu     = false;
    const char *integ_str   = "kappa-sigma";
    float       kappa       = 3.0f;
    int         iterations  = 3;

    static struct option long_opts[] = {
        {"file",        required_argument, nullptr, 'f'},
        {"output",      required_argument, nullptr, 'o'},
        {"cpu",         no_argument,       nullptr, 'c'},
        {"integration", required_argument, nullptr, 'i'},
        {"kappa",       required_argument, nullptr, 'k'},
        {"iterations",  required_argument, nullptr, 'n'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:o:", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'f': csv_file    = optarg;        break;
        case 'o': output_file = optarg;        break;
        case 'c': use_cpu     = true;          break;
        case 'i': integ_str   = optarg;        break;
        case 'k': kappa       = strtof(optarg, nullptr); break;
        case 'n': iterations  = atoi(optarg);  break;
        default:  usage(argv[0]); return 1;
        }
    }

    if (!csv_file) {
        fprintf(stderr, "Error: --file is required\n\n");
        usage(argv[0]);
        return 1;
    }

    bool use_kappa_sigma = (strcmp(integ_str, "kappa-sigma") == 0);
    if (!use_kappa_sigma && strcmp(integ_str, "mean") != 0) {
        fprintf(stderr, "Error: unknown integration method '%s'\n", integ_str);
        return 1;
    }

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

    printf("Parsed %d frames, reference = frame %d\n", n_frames, ref_idx);
    printf("Integration: %s | Lanczos: %s\n",
           integ_str, use_cpu ? "CPU" : "GPU");

    /* ---- Load reference to get output dimensions ---- */
    Image ref_img = {};
    check(fits_load(frames[ref_idx].filepath, &ref_img), "fits_load (reference)");
    int W = ref_img.width, H = ref_img.height;
    printf("Output dimensions: %d x %d\n", W, H);
    image_free(&ref_img);

    /* ---- GPU init (if needed) ---- */
    if (!use_cpu) {
        check(lanczos_gpu_init(0 /* default stream */), "lanczos_gpu_init");
    }

    /* ---- Allocate transformed frame array ---- */
    Image *transformed = (Image *)calloc((size_t)n_frames, sizeof(Image));
    if (!transformed) { fprintf(stderr, "Out of memory\n"); return 1; }

    /* ---- Transform each frame ---- */
    for (int i = 0; i < n_frames; i++) {
        printf("Processing frame %d/%d: %s\n", i + 1, n_frames, frames[i].filepath);

        /* Load raw frame */
        Image raw = {};
        check(fits_load(frames[i].filepath, &raw), "fits_load (frame)");

        /* Allocate output buffer (same dimensions as reference) */
        transformed[i].width  = W;
        transformed[i].height = H;
        transformed[i].data   = (float *)calloc((size_t)W * H, sizeof(float));
        if (!transformed[i].data) {
            fprintf(stderr, "Out of memory allocating transformed frame %d\n", i);
            return 1;
        }

        if (frames[i].is_reference) {
            /* Reference frame: no transform needed; copy data directly.
             * If the reference has the same dimensions, this is a straight copy. */
            if (raw.width == W && raw.height == H) {
                memcpy(transformed[i].data, raw.data,
                       (size_t)W * H * sizeof(float));
            } else {
                fprintf(stderr,
                    "Warning: reference frame dimensions mismatch (%dx%d vs %dx%d); "
                    "applying identity transform\n",
                    raw.width, raw.height, W, H);
                DsoError err = use_cpu
                    ? lanczos_transform_cpu(&raw, &transformed[i], &frames[i].H)
                    : lanczos_transform_gpu(&raw, &transformed[i], &frames[i].H);
                check(err, "lanczos_transform (reference fallback)");
            }
        } else {
            /* Non-reference: apply homographic transform */
            DsoError err = use_cpu
                ? lanczos_transform_cpu(&raw, &transformed[i], &frames[i].H)
                : lanczos_transform_gpu(&raw, &transformed[i], &frames[i].H);
            check(err, "lanczos_transform");
        }

        image_free(&raw);
    }

    /* ---- GPU cleanup ---- */
    if (!use_cpu) {
        lanczos_gpu_cleanup();
    }

    /* ---- Integration ---- */
    printf("Integrating %d frames via %s...\n", n_frames, integ_str);

    /* Build const pointer array required by integration API */
    const Image **frame_ptrs = (const Image **)malloc((size_t)n_frames * sizeof(Image *));
    if (!frame_ptrs) { fprintf(stderr, "Out of memory\n"); return 1; }
    for (int i = 0; i < n_frames; i++) frame_ptrs[i] = &transformed[i];

    Image output = {};
    if (use_kappa_sigma) {
        check(integrate_kappa_sigma(frame_ptrs, n_frames, &output, kappa, iterations),
              "integrate_kappa_sigma");
    } else {
        check(integrate_mean(frame_ptrs, n_frames, &output), "integrate_mean");
    }
    free(frame_ptrs);

    /* ---- Save output ---- */
    printf("Saving output to %s\n", output_file);
    check(fits_save(output_file, &output), "fits_save");

    /* ---- Cleanup ---- */
    image_free(&output);
    for (int i = 0; i < n_frames; i++) image_free(&transformed[i]);
    free(transformed);
    free(frames);

    printf("Done.\n");
    return 0;
}
