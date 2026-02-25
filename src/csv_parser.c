/*
 * csv_parser.c — CSV frame-list parser
 *
 * Reads a CSV file where the first row is a header (skipped) and each
 * subsequent row describes one input frame:
 *
 *   filepath, is_reference, h00, h01, h02, h10, h11, h12, h20, h21, h22
 *
 * The nine h-values form a row-major 3×3 homography that maps source pixel
 * coordinates to reference pixel coordinates (forward direction).
 *
 * The returned array is heap-allocated; caller must free(*frames_out).
 */

#include "csv_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Strip leading and trailing ASCII whitespace in-place (returns same ptr). */
static char *strip(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)*(end - 1))) end--;
    *end = '\0';
    return s;
}

DsoError csv_parse(const char *filepath, FrameInfo **frames_out, int *n_frames_out)
{
    if (!filepath || !frames_out || !n_frames_out) return DSO_ERR_INVALID_ARG;

    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "csv_parse: cannot open '%s'\n", filepath);
        return DSO_ERR_IO;
    }

    char line[8192];

    /* Skip the header row */
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return DSO_ERR_CSV;
    }

    FrameInfo *frames   = NULL;
    int        capacity = 0;
    int        n        = 0;

    while (fgets(line, sizeof(line), fp)) {
        /* Skip blank lines */
        char *trimmed = strip(line);
        if (trimmed[0] == '\0') continue;

        /* Grow the output array with doubling strategy */
        if (n >= capacity) {
            int new_cap = (capacity == 0) ? 16 : capacity * 2;
            FrameInfo *tmp = (FrameInfo *)realloc(frames,
                                                  (size_t)new_cap * sizeof(FrameInfo));
            if (!tmp) {
                free(frames);
                fclose(fp);
                return DSO_ERR_ALLOC;
            }
            frames   = tmp;
            capacity = new_cap;
        }

        FrameInfo *fi = &frames[n];
        memset(fi, 0, sizeof(*fi));

        char *saveptr = NULL;
        char *tok;

        /* Field 1: file path */
        tok = strtok_r(line, ",", &saveptr);
        if (!tok) goto bad_line;
        strncpy(fi->filepath, strip(tok), sizeof(fi->filepath) - 1);

        /* Field 2: is_reference flag (0 or 1) */
        tok = strtok_r(NULL, ",", &saveptr);
        if (!tok) goto bad_line;
        fi->is_reference = atoi(strip(tok));

        /* Fields 3-11: homography coefficients h00..h22 (row-major) */
        for (int i = 0; i < 9; i++) {
            tok = strtok_r(NULL, ",", &saveptr);
            if (!tok) goto bad_line;
            fi->H.h[i] = strtod(strip(tok), NULL);
        }

        n++;
        continue;

    bad_line:
        fprintf(stderr, "csv_parse: malformed line %d, skipping\n", n + 2);
    }

    fclose(fp);

    *frames_out   = frames;
    *n_frames_out = n;
    return DSO_OK;
}
