# ERRORS.md — Code Review Findings

Reviewed 2026-03-20. Previous entries superseded (those issues were already fixed).

---

## Bug 1 — `src/lanczos_cpu.c:137` — Float comparison amplifies FP noise at OOB boundary

**Severity:** High — causes `test_gpu_integer_shift` to fail
**Status:** Fixed

`weight_sum == 0.f` exact comparison fails when the Lanczos kernel center tap lands on an
out-of-bounds integer coordinate. In that case the only tap with weight 1 is OOB and
skipped; the remaining taps at offsets ±1, ±2, ±3 are mathematically zero (`sin(k·π)=0`)
but `sinf` returns a tiny non-zero value (≈8.7e-8 for `sinf(π)`). These accumulate into
`weight_sum ≈ −1e-9`, which is not `== 0.f`, so `accum / weight_sum` amplifies the noise
to ≈1560 instead of 0.

The GPU (nppiRemap) correctly returns 0 for an OOB source coordinate, so CPU and GPU
diverge by up to 1560 at boundary pixels.

**Fix:** `(fabsf(weight_sum) < 1e-6f) ? 0.f : accum / weight_sum`

---

## Bug 2 — `src/lanczos_gpu.cu:164` — Missing singular-H check in `lanczos_transform_gpu`

**Severity:** High — causes `test_gpu_singular_h` to fail
**Status:** Fixed

`lanczos_transform_gpu` never checks the determinant of `H`. For a singular (all-zero)
homography, `build_coord_maps` writes `(-1, -1)` for every pixel and nppiRemap leaves the
zeroed `d_dst` unchanged. The function returns `DSO_OK` instead of `DSO_ERR_INVALID_ARG`.

The dead helper `invert_homography_h` (lines 74–97) already contains the required
determinant check; it just is never called from `lanczos_transform_gpu`.

**Fix:** Determinant check before any CUDA allocation.

---

## Bug 3 — `src/integration_gpu.cu:122` — Float variance accumulation in GPU kappa-sigma

**Severity:** Medium — reduced precision for high-dynamic-range images
**Status:** Fixed

`float sq = 0.f` accumulates squared deviations in single precision. For pixel values up
to 65535 and batch size 64, max `sq ≈ 64 × 65535² ≈ 2.75 × 10¹¹`. Float has only ~7
significant digits; values that large lose several digits of precision, causing incorrect
sigma estimates and wrong clipping decisions.

The CPU implementation (`integration.c:120`) correctly uses `double sq_sum = 0.0`.

**Fix:** Promote `sq` to `double`.

---

## Issue 4 — `src/debayer_gpu.cu` and `src/debayer_cpu.c` — Symmetric gradients reduce VNG to 4-way direction selection

**Severity:** Low — consistent CPU/GPU quality degradation, tests pass
**Status:** Fixed

Each "directional" gradient was computed as the sum of absolute differences in **both**
directions along the axis:

```c
g[0] = |P(0,-2) - P(0,0)| + |P(0,0) - P(0,2)|  // N
g[4] = |P(0, 2) - P(0,0)| + |P(0,0) - P(0,-2)|  // S  ← always equals g[0]
```

Because `|a−b| = |b−a|`, `g[4] ≡ g[0]`, `g[5] ≡ g[1]`, `g[6] ≡ g[2]`, `g[7] ≡ g[3]`.
The VNG direction-selection threshold was identical for opposite directions, so the
algorithm could not distinguish "smooth north, steep south" from the reverse.

**Fix:** Replace each two-sided formula with a one-sided formula sampling only the pixel
in the named direction:

```c
g[0] = |P( 0,-2) - P(0,0)|  // N
g[1] = |P( 2,-2) - P(0,0)|  // NE
g[2] = |P( 2, 0) - P(0,0)|  // E
g[3] = |P( 2, 2) - P(0,0)|  // SE
g[4] = |P( 0, 2) - P(0,0)|  // S
g[5] = |P(-2, 2) - P(0,0)|  // SW
g[6] = |P(-2, 0) - P(0,0)|  // W
g[7] = |P(-2,-2) - P(0,0)|  // NW
```

Applied to both `src/debayer_gpu.cu` and `src/debayer_cpu.c`.

---

## Issue 5 — Dead code: `invert_homography` / `invert_homography_h` helpers

**Severity:** Cosmetic
**Status:** Documented, not fixed (retained "for reference" per code comment)

`src/lanczos_cpu.c:50–80` defines `invert_homography()` and
`src/lanczos_gpu.cu:74–97` defines `invert_homography_h()`. Neither is called anywhere.
Both files now use `H` directly as the backward map (no inversion needed).
