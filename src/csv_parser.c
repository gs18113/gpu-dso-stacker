/*
 * csv_parser.c — CSV frame-list parser
 *
 * Accepts only 2-column format (header row always skipped):
 *   filepath, is_reference
 *
 * Any other column count causes DSO_ERR_CSV.
 * Malformed data rows are skipped with a warning.
 */

#include "csv_parser.h"
#include "compat.h"
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

/*
 * Count comma-separated tokens in a line (does not modify the string).
 * Returns 0 for an empty or whitespace-only line.
 */
static int count_columns(const char *line)
{
    char buf[8192];
    strncpy(buf, line, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *trimmed = strip(buf);
    if (trimmed[0] == '\0') return 0;

    int cols = 1;
    for (const char *p = trimmed; *p; p++) {
        if (*p == ',') cols++;
    }
    return cols;
}

DsoError csv_parse(const char   *filepath,
                   FrameInfo   **frames_out,
                   int          *n_frames_out)
{
    if (!filepath || !frames_out || !n_frames_out)
        return DSO_ERR_INVALID_ARG;

    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "csv_parse: cannot open '%s'\n", filepath);
        return DSO_ERR_IO;
    }

    char line[8192];

    /* Read header row and verify column count */
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return DSO_ERR_CSV;
    }

    int ncols = count_columns(line);
    if (ncols != 2) {
        fprintf(stderr,
                "csv_parse: expected 2 columns, got %d in header of '%s'\n",
                ncols, filepath);
        fclose(fp);
        return DSO_ERR_CSV;
    }

    FrameInfo *frames   = NULL;
    int        capacity = 0;
    int        n        = 0;
    int        line_num = 1; /* header was line 1 */

    while (fgets(line, sizeof(line), fp)) {
        line_num++;

        char *trimmed = strip(line);
        if (trimmed[0] == '\0') continue;

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

        n++;
        continue;

    bad_line:
        fprintf(stderr, "csv_parse: malformed line %d, skipping\n", line_num);
    }

    fclose(fp);

    *frames_out   = frames;
    *n_frames_out = n;
    return DSO_OK;
}
