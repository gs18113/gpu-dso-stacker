# Pipeline Refactor — Linux Build & Test Handoff

> Generated 2026-03-23.  Apply this document in the Linux build environment.

## What Changed

### Round 1 — Single-pass pipeline (no double disk read)
`pipeline_cpu.c` and `pipeline.cu` were rewritten so each frame is loaded from
disk exactly **once**.  The old design read every frame twice (Phase 1 for star
detection, Phase 2 for warp+integration).

Key design:
- An `order[]` array places the reference frame at position 0, then all
  non-reference frames.  This ensures `ref_stars` is populated before any RANSAC
  call.
- Per frame: load → calibrate → debayer_lum → star_detect → D2H → CCL+CoM →
  RANSAC → Lanczos warp → mini-batch accumulate.  All in one pass.

### Round 2 — Remove 11-column CSV path + add GPU I/O overlap

**Removed entirely:**
- 11-column CSV format (pre-computed homographies).  Any non-2-column CSV now
  returns `DSO_ERR_CSV`.
- `has_transforms` parameter from `pipeline_run`, `pipeline_run_cpu`, and
  `csv_parse`.
- `phase_with_detection` (sequential GPU path, no overlap).
- `phase_transform_integrate` (double-buffer for 11-col path).

**Added:**
- `phase_detect_warp_integrate` in `pipeline.cu` — single function combining
  star detection + RANSAC + Lanczos warp with double-buffered I/O overlap:
  - `stream_copy` carries H2D DMA transfers.
  - `stream_compute` carries calib/debayer/star_detect/warp/integration.
  - After RANSAC, the GPU warp for frame *m* runs while the CPU loads frame
    *m+1* from disk and streams it via `stream_copy`.
  - At batch boundaries the H2D of frame *m+1* overlaps mini-batch integration.
  - No `e_gpu` events needed: `cudaStreamSynchronize(stream_compute)` at the
    D2H sync point implicitly clears the previous warp, making `d_raw[next_slot]`
    safe to reuse.

---

## Files Modified

| File | Change |
|---|---|
| `src/pipeline.cu` | Full rewrite — new `phase_detect_warp_integrate`, no `has_transforms` |
| `src/pipeline_cpu.c` | Full rewrite — single-pass with `order[]` array, no `has_transforms` |
| `src/csv_parser.c` | 11-col support removed; rejects any non-2-col CSV |
| `include/csv_parser.h` | Removed `has_transforms_out` from `csv_parse` signature |
| `include/pipeline.h` | Removed `has_transforms` from both function signatures; updated doc-comment |
| `main.cpp` | Updated `csv_parse` call (3 args), `pipeline_run` call (4 args), removed `has_transforms` variable |
| `tests/test_cpu.c` | Removed 11-col CSV tests; updated all `csv_parse` call sites to 3 args |
| `CLAUDE.md` | Updated pipeline description, API reference, Input CSV Format section |

---

## Current API Signatures

```c
/* csv_parser.h */
DsoError csv_parse(const char *filepath, FrameInfo **frames_out, int *n_frames_out);

/* pipeline.h */
DsoError pipeline_run    (FrameInfo *frames, int n_frames, int ref_idx,
                          const PipelineConfig *config);
DsoError pipeline_run_cpu(FrameInfo *frames, int n_frames, int ref_idx,
                          const PipelineConfig *config);
```

---

## Build Steps (Linux)

```bash
# From repo root — Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCFITSIO_PREFIX=/opt/cfitsio   # or wherever CFITSIO is installed

cmake --build build --parallel $(nproc)
```

For Release:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --parallel $(nproc)
```

---

## Tests to Run

### CPU tests (no GPU required)
```bash
./build/test_cpu
./build/test_star_detect
./build/test_ransac
./build/test_debayer_cpu
./build/test_calibration
./build/test_color
./build/test_image_io
./build/test_audit
```

### GPU tests (skip if no GPU — exit code 77)
```bash
./build/test_gpu
./build/test_integration_gpu
```

### Expected: all passing
The CSV suite now has 7 tests (down from 10 — removed `test_csv_has_transforms_11col`,
`test_csv_null_has_transforms_out`, `test_csv_homography_values`).

---

## End-to-End Smoke Test

Generate synthetic frames and run the full pipeline:

```bash
# Generate 5 RGGB frames with stars
python3 python/generate_test_frames.py -n 5 -s 30 -o /tmp/test_frames \
        --bayer rggb

# GPU pipeline
./build/dso_stacker -f /tmp/test_frames/frames.csv -o /tmp/out_gpu.fits

# CPU pipeline
./build/dso_stacker -f /tmp/test_frames/frames.csv -o /tmp/out_cpu.fits --cpu
```

Both should complete without error and produce a valid FITS output.

### Verify I/O overlap (GPU path)
The log should show interleaved "Loading frame N+1" messages appearing
**during** frame N processing (between "aligned with K inliers" and
"Integrating batch"):

```
[Pipeline] Frame 2/5: 31 star(s) — /tmp/test_frames/frame_01.fits
[Pipeline] Frame 2: aligned with 28 inlier(s)
[Pipeline] Loading frame 3/5: /tmp/test_frames/frame_02.fits   ← overlap
[Pipeline] Integrating batch of 2 frame(s)...
```

---

## Key Design Notes

- **Homography convention**: H is always the *backward* map (ref → src).  Used
  directly for pixel sampling — do not invert.
- **d_raw[next_slot] safety**: `cudaStreamSynchronize(stream_compute)` called
  before D2H also waits for the Lanczos warp from `pos-1` (since stream_compute
  is sequential).  By the time the pre-load writes to `d_raw[next_slot]`, that
  slot's prior warp is done.  No `cudaEvent` guard needed.
- **CPU dispatch**: `pipeline_run()` checks `!config->use_gpu_lanczos` as its
  very first statement and returns `pipeline_run_cpu(...)` — no CUDA context
  created, zero GPU overhead.
- **Reference frame**: always processed at `pos=0` regardless of its position
  in the input CSV, so `ref_stars` is always set before any RANSAC call.
