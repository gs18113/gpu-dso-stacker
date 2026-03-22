/*
 * getopt_port.h — Portable getopt / getopt_long for platforms lacking
 *                 POSIX <getopt.h> (i.e. MSVC on Windows).
 *
 * This is a minimal, self-contained implementation derived from public-
 * domain and BSD-licensed sources. It supports:
 *   - getopt()       (short options)
 *   - getopt_long()  (GNU-style long options)
 *
 * On non-MSVC compilers this header should NOT be included — use the
 * system <getopt.h> instead.  The compat.h header handles this dispatch.
 *
 * License: BSD-2-Clause
 */

#pragma once

#ifdef _MSC_VER

#ifdef __cplusplus
extern "C" {
#endif

extern char *optarg;
extern int   optind;
extern int   opterr;
extern int   optopt;

struct option {
    const char *name;   /* long option name              */
    int         has_arg;/* no_argument / required_argument / optional_argument */
    int        *flag;   /* if non-NULL, set *flag = val  */
    int         val;    /* value to return (or store)    */
};

#define no_argument       0
#define required_argument 1
#define optional_argument 2

int getopt(int argc, char *const argv[], const char *optstring);

int getopt_long(int argc, char *const argv[], const char *optstring,
                const struct option *longopts, int *longindex);

#ifdef __cplusplus
}
#endif

#endif /* _MSC_VER */
