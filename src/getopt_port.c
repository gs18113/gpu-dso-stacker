/*
 * getopt_port.c — Portable getopt / getopt_long implementation.
 *
 * Compiled only on MSVC (Windows) where POSIX <getopt.h> is absent.
 * Provides getopt() and getopt_long() with standard GNU semantics.
 *
 * Derived from public-domain implementations (musl libc, NetBSD).
 * License: BSD-2-Clause
 *
 * Copyright (c) 2012-2024  Various contributors
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.
 */

#ifdef _MSC_VER

#include "getopt_port.h"
#include <stdio.h>
#include <string.h>

char *optarg = NULL;
int   optind = 1;
int   opterr = 1;
int   optopt = '?';

static int optpos = 0;  /* position within clustered short opts */

static void permute(char *const *argv, int dest, int src)
{
    char *tmp   = (char *)argv[src];
    int   i;
    for (i = src; i > dest; i--)
        ((char **)argv)[i] = (char *)argv[i - 1];
    ((char **)argv)[dest] = tmp;
}

int getopt(int argc, char *const argv[], const char *optstring)
{
    int i;
    const char *p;

    if (optind == 0) {
        optind = 1;
        optpos = 0;
    }

    if (optind >= argc || !argv[optind])
        return -1;

    /* skip non-option arguments */
    if (argv[optind][0] != '-' || argv[optind][1] == '\0')
        return -1;

    /* "--" terminates options */
    if (argv[optind][0] == '-' && argv[optind][1] == '-' &&
        argv[optind][2] == '\0') {
        optind++;
        return -1;
    }

    if (optpos == 0)
        optpos = 1;  /* skip leading '-' */

    optopt = (unsigned char)argv[optind][optpos];
    optpos++;

    p = strchr(optstring, optopt);
    if (!p || optopt == ':') {
        if (opterr && optstring[0] != ':')
            fprintf(stderr, "%s: unknown option '-%c'\n", argv[0], optopt);
        if (!argv[optind][optpos]) {
            optind++;
            optpos = 0;
        }
        return '?';
    }

    if (p[1] == ':') {
        /* option requires an argument */
        if (argv[optind][optpos]) {
            /* argument is rest of current argv element */
            optarg = (char *)&argv[optind][optpos];
            optind++;
            optpos = 0;
        } else if (p[2] != ':') {
            /* argument is next argv element (required) */
            optind++;
            if (optind >= argc) {
                if (opterr && optstring[0] != ':')
                    fprintf(stderr, "%s: option '-%c' requires an argument\n",
                            argv[0], optopt);
                optpos = 0;
                return (optstring[0] == ':') ? ':' : '?';
            }
            optarg = (char *)argv[optind];
            optind++;
            optpos = 0;
        } else {
            /* optional argument not present */
            optarg = NULL;
            if (!argv[optind][optpos]) {
                optind++;
                optpos = 0;
            }
        }
    } else {
        /* no argument */
        optarg = NULL;
        if (!argv[optind][optpos]) {
            optind++;
            optpos = 0;
        }
    }

    return optopt;
}

int getopt_long(int argc, char *const argv[], const char *optstring,
                const struct option *longopts, int *longindex)
{
    int i;
    const char *arg;

    if (optind == 0) {
        optind = 1;
        optpos = 0;
    }

    if (optind >= argc || !argv[optind])
        return -1;

    arg = argv[optind];

    /* not an option */
    if (arg[0] != '-' || arg[1] == '\0')
        return -1;

    /* short option(s) */
    if (arg[1] != '-')
        return getopt(argc, argv, optstring);

    /* "--" terminates */
    if (arg[2] == '\0') {
        optind++;
        return -1;
    }

    /* long option: skip leading "--" */
    arg += 2;

    for (i = 0; longopts[i].name; i++) {
        size_t namelen = strlen(longopts[i].name);

        if (strncmp(arg, longopts[i].name, namelen) != 0)
            continue;

        /* exact match or match up to '=' */
        if (arg[namelen] != '\0' && arg[namelen] != '=')
            continue;

        /* found a match */
        if (longindex)
            *longindex = i;

        optind++;

        if (longopts[i].has_arg != no_argument) {
            if (arg[namelen] == '=') {
                optarg = (char *)&arg[namelen + 1];
            } else if (longopts[i].has_arg == required_argument) {
                if (optind >= argc) {
                    if (opterr)
                        fprintf(stderr, "%s: option '--%s' requires an argument\n",
                                argv[0], longopts[i].name);
                    optopt = longopts[i].val;
                    return '?';
                }
                optarg = (char *)argv[optind++];
            } else {
                optarg = NULL;  /* optional_argument, not provided */
            }
        } else {
            optarg = NULL;
            if (arg[namelen] == '=') {
                if (opterr)
                    fprintf(stderr, "%s: option '--%s' doesn't allow an argument\n",
                            argv[0], longopts[i].name);
                optopt = longopts[i].val;
                return '?';
            }
        }

        if (longopts[i].flag) {
            *longopts[i].flag = longopts[i].val;
            return 0;
        }
        return longopts[i].val;
    }

    /* no match found */
    if (opterr)
        fprintf(stderr, "%s: unrecognized option '%s'\n", argv[0], argv[optind]);
    optind++;
    return '?';
}

#endif /* _MSC_VER */
