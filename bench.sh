#!/usr/bin/env bash
# bench.sh — Benchmark GPU vs CPU pipeline on a given CSV.
#
# Usage:
#   ./bench.sh [options]
#
# Options:
#   -f <csv>      Input CSV (default: data/frames.csv)
#   -b <binary>   dso_stacker binary (default: build/dso_stacker)
#   -r <n>        Number of runs to average (default: 3)
#   -h            Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSV="$SCRIPT_DIR/data/frames.csv"
BIN="$SCRIPT_DIR/build/dso_stacker"
RUNS=3

while getopts "f:b:r:h" opt; do
    case $opt in
        f) CSV="$OPTARG" ;;
        b) BIN="$OPTARG" ;;
        r) RUNS="$OPTARG" ;;
        h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option -$OPTARG" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$BIN" ]]; then
    echo "Error: binary not found: $BIN" >&2
    echo "Build first with: cmake --build build --parallel \$(nproc)" >&2
    exit 1
fi

if [[ ! -f "$CSV" ]]; then
    echo "Error: CSV not found: $CSV" >&2
    exit 1
fi

TMPOUT=$(mktemp /tmp/dso_bench_XXXXXX.fits)
trap 'rm -f "$TMPOUT"' EXIT

# Run one timed execution; print wall seconds to stdout, pipeline output to stderr.
run_once() {
    local extra_args=("$@")
    local start end
    start=$(date +%s%N)
    "$BIN" -f "$CSV" -o "$TMPOUT" "${extra_args[@]}" >/dev/null
    end=$(date +%s%N)
    echo "scale=3; ($end - $start) / 1000000000" | bc
}

bench() {
    local label="$1"; shift
    local args=("$@")
    local total=0
    echo "[$label — ${RUNS} run(s)]" >&2
    for ((i=1; i<=RUNS; i++)); do
        t=$(run_once "${args[@]}" 2>/dev/null)
        printf "  run %d: %s s\n" "$i" "$t" >&2
        total=$(echo "scale=3; $total + $t" | bc)
    done
    avg=$(echo "scale=3; $total / $RUNS" | bc)
    printf "  avg  : %s s\n\n" "$avg" >&2
    echo "$avg"
}

echo ""
echo "=== dso_stacker benchmark ==="
echo "CSV    : $CSV"
echo "Binary : $BIN"
echo "Runs   : $RUNS"
echo ""

gpu_avg=$(bench "GPU pipeline" )
cpu_avg=$(bench "CPU pipeline" --cpu)

# Speedup
speedup=$(echo "scale=2; $cpu_avg / $gpu_avg" | bc)

echo "=== Summary ==="
printf "  GPU avg : %s s\n" "$gpu_avg"
printf "  CPU avg : %s s\n" "$cpu_avg"
printf "  Speedup : GPU is %sx faster\n" "$speedup"
echo ""
