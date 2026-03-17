# Code Review — Errors and Issues

Identified during full source review on 2026-03-18.

---

## 1. `include/lanczos_cpu.h` lines 23–24 — Wrong documentation (H convention)

**Type:** Documentation error
**Severity:** Medium (misleading API docs)

The doc comment states:
> "H is the *forward* homography (source pixel coords → reference pixel coords).
> Internally the function inverts H and uses the inverse for backward mapping."

The actual implementation (`src/lanczos_cpu.c`) uses H **directly** as the backward map
(ref → src) without inversion. The internal `invert_homography()` function is dead code.
This contradicts the rest of the project where all transform functions explicitly document
H as the backward map.

---

## 2. `src/ransac.c` line 252 — Dead code: redundant `mat33_mul`

**Type:** Dead code / wasted computation
**Severity:** Low (no incorrect output, just wasted work)

```c
mat33_mul(T_src_inv, h_norm, tmp);   /* T_src⁻¹ * H_norm */   // ← dead

/* Reshape h_norm (vector) to 3×3 first */
double H_norm33[9];
memcpy(H_norm33, h_norm, 9 * sizeof(double));
mat33_mul(T_src_inv, H_norm33, tmp);   // ← overwrites tmp from line above
mat33_mul(tmp, T_ref, H_raw);
```

Line 252 computes `T_src_inv * h_norm → tmp`. Lines 254–257 then copy `h_norm` to a
local `H_norm33` and recompute the identical product, overwriting `tmp` before it is
ever read. The first `mat33_mul` call and the `H_norm33`/`memcpy` scaffolding are
both dead code.

---

## 3. `src/ransac.c` lines 370–381 — Dead code: abandoned reservoir sampling

**Type:** Dead code
**Severity:** Low (no incorrect output)

```c
/* Use a small reservoir of 4 indices */
for (int k = 0; k < 4; k++) {
    int j = k + rand() % (pool_size - k);
    if (k == 0) {
        idx[0] = j % n_matches;
        idx[1] = (j + 1) % n_matches;
        idx[2] = (j + 2) % n_matches;
        idx[3] = (j + 3) % n_matches;
        break;   // ← exits immediately; k=1,2,3 never execute
    }
}
/* Simpler: just pick 4 random distinct indices */
int used[4] = {-1,-1,-1,-1};
for (int k = 0; k < 4; ) {
    ...   // ← overwrites idx[] entirely
}
```

The first loop breaks unconditionally after `k == 0`, making `k=1,2,3` unreachable.
The "simpler" block that follows immediately overwrites `idx[]`, making the first
block's output irrelevant. The first block is completely dead.

---

## 4. `src/ransac.c` line 424 — Dead guard on adaptive termination

**Type:** Dead code
**Severity:** Low (no incorrect output)

```c
double log_arg = 1.0 - p->confidence;
if (log_arg < -1.0 + 1e-10) log_arg = -1.0 + 1e-10;   // ← never triggers
double n_needed = log(log_arg) / log(1.0 - p4);
```

`log_arg = 1.0 - confidence`. For any valid confidence ∈ (0, 1), `log_arg` ∈ (0, 1).
The guard clamps if `log_arg < −0.9999…`, which requires `confidence > 1.9999` —
impossible for a valid confidence value. The guard never executes.

The formula itself is correct (`log(1−conf) / log(1−p^4)`); only the guard is dead code.

---

## 5. `src/fits_io.c` lines 89–93 — File handle leak on `ffcrim` failure

**Type:** Resource leak
**Severity:** High (file descriptor leak on error path)

```c
ffinit(&fptr, overwrite_path, &status);   // opens file, fptr now valid
ffcrim(fptr, FLOAT_IMG, 2, naxes, &status);
if (status) {
    fits_report_error(stderr, status);
    return DSO_ERR_FITS;   // ← fptr is never closed
}
```

If `ffinit` succeeds (fptr is valid) but `ffcrim` fails (status ≠ 0), the function
returns without calling `ffclos(fptr, &status)`. The FITS file handle leaks.
Repeated failures accumulate open file descriptors until the process runs out.

---

## 6. `src/pipeline_cpu.c` line 251 — Missing `ref_idx` bounds check

**Type:** Safety / potential out-of-bounds array access
**Severity:** High (undefined behaviour on invalid input)

```c
DsoError pipeline_run_cpu(..., int ref_idx, ...) {
    if (!frames || n_frames <= 0 || !config) return DSO_ERR_INVALID_ARG;
    // ref_idx is never validated here ↑
    ...
    DsoError e = fits_load(frames[ref_idx].filepath, &ref_img);  // ← UB if ref_idx < 0 or ≥ n_frames
```

`pipeline_run()` in `pipeline.cu` correctly validates `ref_idx < 0 || ref_idx >= n_frames`
before delegating to `pipeline_run_cpu()`. However `pipeline_run_cpu()` is a public
function and performs no such check itself.

---

## 7. `src/debayer_gpu.cu` lines 126, 128 — Wrong diagonal gradient direction labels

**Type:** Documentation error
**Severity:** Low (comments only; algorithm is correct)

`P(dx, dy) = sm[sy+dy][sx+dx]`, so `P(-2,-2)` is the pixel 2 columns left and 2 rows
up (upper-left = NW), and `P(2,2)` is lower-right (SE).

```c
g[1] = fabsf(P(-2,-2) - P( 0, 0)) + fabsf(P( 0, 0) - P( 2, 2)); /* NE→SE diag */  // ← wrong: NW-SE
g[3] = fabsf(P(-2, 2) - P( 0, 0)) + fabsf(P( 0, 0) - P( 2,-2)); /* NW→SE cross */ // ← wrong: NE-SW
```

- `g[1]` spans `P(-2,-2)` ↔ `P(2,2)` (upper-left ↔ lower-right) → **NW-SE diagonal**
- `g[3]` spans `P(-2,2)` ↔ `P(2,-2)` (lower-left ↔ upper-right) → **NE-SW diagonal**

Both labels are transposed.

---

## 8. `src/star_detect_cpu.c` line 38 + `src/star_detect_gpu.cu` line 173 — Silent Moffat radius clamp

**Type:** Silent incorrect result / robustness
**Severity:** Medium (wrong kernel for alpha > 5, no user warning)

```c
int R = (int)ceilf(3.0f * params->alpha);
if (R > MOFFAT_MAX_RADIUS) R = MOFFAT_MAX_RADIUS;   // ← silent truncation
```

When `alpha > 5`, the intended kernel radius (`ceil(3·alpha) > 15`) exceeds
`MOFFAT_MAX_RADIUS = 15`. The kernel is silently truncated, producing an under-sized
PSF and incorrect (under-smoothed) convolution output. No warning or error is
emitted, making the problem invisible to the caller.

The same silent clamp exists in both the CPU (`star_detect_cpu.c`) and GPU
(`star_detect_gpu.cu`) paths.
