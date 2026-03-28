# Apple Silicon (Metal) Implementation Plan

This document lays out a practical, performance-oriented, and numerically safe plan for adding an Apple Silicon (Metal) backend to this codebase **without regressing existing Linux/Windows behavior**.

It is written to be directly executable by human developers and AI coding assistants.

---

## 1) Goals and Constraints

### Primary goals

1. Add a Metal backend for Apple Silicon (M1/M2/M3+).
2. Preserve current CUDA backend behavior and performance on Linux/Windows.
3. Preserve current CPU-only path behavior and numerical characteristics.
4. Keep homography conventions and image-processing semantics identical across backends.

### Hard constraints

- Do **not** alter existing CUDA execution path semantics.
- Do **not** change CSV format or pipeline behavior contracts.
- Do **not** silently change numerical conventions (homography direction, clipping behavior, luminance coefficients, edge handling).
- Keep backend selection explicit and testable.

---

## 2) Current Architecture (What Exists Today)

The project already has the right high-level separation to support another GPU backend:

- CPU implementations exist for all core stages.
- GPU implementations exist in CUDA `.cu` files.
- `pipeline_run()` dispatches to CPU path when configured for CPU.
- Shared domain types are already backend-agnostic in `include/dso_types.h`.

Important existing stage map:

- Debayer: `src/debayer_cpu.c` / `src/debayer_gpu.cu`
- Star detect: `src/star_detect_cpu.c` / `src/star_detect_gpu.cu`
- Match/RANSAC: `src/ransac.c` and optional GPU triangle matcher `src/ransac_gpu.cu`
- Warp: `src/lanczos_cpu.c` / `src/lanczos_gpu.cu`
- Integrate: `src/integration.c` / `src/integration_gpu.cu`
- Pipeline orchestrators: `src/pipeline_cpu.c` / `src/pipeline.cu`

---

## 3) Design Principles for a Safe Metal Port

1. **Backend additive, not replacing**: introduce Metal path alongside CUDA and CPU.
2. **Minimal public API churn**: keep existing CLI options valid; add backend/device selection in a backward-compatible way.
3. **Semantic parity first, optimization second**: establish correctness baseline before aggressive tuning.
4. **Single source of truth for math conventions**:
   - Backward homography (`ref -> src`) usage must remain unchanged.
   - Kappa-sigma clipping rules must match existing CPU/GPU semantics.
   - Debayer and star threshold behavior must match existing logic.
5. **Backend isolation**: no Linux/Windows code should compile Objective-C++/Metal.

---

## 4) Proposed Backend Abstraction

Use a thin orchestrator-level backend selection while keeping most current modules intact.

### 4.1 Add backend enum in `PipelineConfig`

Introduce:

- `DSO_BACKEND_AUTO`
- `DSO_BACKEND_CPU`
- `DSO_BACKEND_CUDA`
- `DSO_BACKEND_METAL`

Behavior:

- AUTO chooses CUDA when available on supported builds, Metal on macOS build when enabled, else CPU fallback.
- Existing `--cpu` remains supported and maps to `DSO_BACKEND_CPU`.

### 4.2 Add Metal orchestrator entry point

Add parallel entry point:

- `DsoError pipeline_run_metal(FrameInfo*, int, int, const PipelineConfig*);`

Then in top-level dispatch:

- CPU -> `pipeline_run_cpu`
- CUDA -> existing `pipeline_run` CUDA path internals (or renamed internal helper)
- METAL -> `pipeline_run_metal`

### 4.3 Keep stage APIs conceptually aligned

For each CUDA stage, create Metal analogs with similar signatures and semantics:

- `debayer_metal_*`
- `star_detect_metal_*`
- `lanczos_metal_*`
- `integration_metal_*`
- optional `ransac_metal_*` (can defer and use CPU RANSAC first)

---

## 5) Implementation Phases (Recommended Order)

### Phase 0 — Build-system scaffolding (no functional changes)

Scope:

- Introduce macOS/Metal build option guarded behind CMake flag, e.g. `DSO_ENABLE_METAL`.
- Ensure default Linux/Windows build remains unchanged.
- Keep CUDA requirements scoped so macOS Metal build does not require NVCC.

Validation:

- Linux/Windows CI unchanged and green.
- macOS build compiles CPU-only + Metal stubs (even before kernels).

---

### Phase 1 — Metal pipeline skeleton + CPU compute fallback in mixed mode

Scope:

- Add `pipeline_run_metal` that handles device initialization and frame orchestration skeleton.
- For first milestone, reuse CPU implementations for hard parts while plumbing Metal memory/command flow.

Why:

- Derisks integration, CLI/backend selection, and resource lifecycle before optimizing kernels.

Validation:

- End-to-end runs on macOS with equivalent outputs to CPU path.

---

### Phase 2 — Port compute-heavy kernels (highest ROI first)

Recommended order by performance impact:

1. Lanczos warp (typically dominant)
2. Debayer
3. Star detection convolution + threshold
4. Integration mini-batch kappa-sigma
5. Optional GPU matching/RANSAC path

Validation:

- Stage-by-stage output comparisons against CPU/CUDA references.

---

### Phase 3 — Performance tuning

Scope:

- Optimize threadgroup sizes and memory access patterns.
- Reduce host-device sync points.
- Introduce pipelining overlap analogous to existing copy/compute overlap.

Validation:

- Benchmark vs CPU path on representative frame sets.
- Confirm no numerical drift beyond accepted tolerance.

---

### Phase 4 — Hardening + CI

Scope:

- Add macOS CI job for build + CPU tests + Metal tests where possible.
- Add targeted regression tests for backend consistency.

Validation:

- Cross-platform CI matrix stable.

---

## 6) Numerical Correctness Checklist

For every Metal stage, verify:

1. **Homography direction** preserved: backward map (`ref -> src`) used directly.
2. **Lanczos edge behavior**: out-of-bounds tap handling and renormalization semantics match existing implementation.
3. **Debayer luminance conversion** uses same coefficients (BT.709).
4. **Moffat kernel generation and normalization** consistent with current radius/normalization rules.
5. **Threshold math** (mean/stddev, Bessel correction where applicable) consistent with current behavior.
6. **Kappa-sigma loop termination and degenerate fallback** exactly preserved.
7. **Floating-point precision choices** deliberate:
   - Use `double` where current code relies on it for stability/overflow safety.
   - Do not silently downcast accumulators that impact clipping decisions.

Recommended acceptance metrics:

- Pixel-wise max absolute error threshold per stage.
- PSNR comparison for full stacked output.
- Relative error in integrated output mean/variance.

---

## 7) Performance Optimization Guidance (Metal-specific)

1. Use threadgroup memory for reusable neighborhoods (debayer/star kernels).
2. Minimize command buffer commits and avoid unnecessary synchronization.
3. Pipeline transfers and compute similarly to current CUDA overlap strategy.
4. Reuse persistent buffers across frames/batches.
5. Tune threadgroup sizes empirically per kernel (do not assume CUDA-optimal sizes are optimal on Apple GPU).
6. Keep data layout contiguous and coalesced for SIMD-group efficiency.
7. Profile with Xcode GPU tools before micro-optimizing.

---

## 8) File/Module Plan (Concrete)

Likely new files:

- `include/metal_common.h` (opaque context, common helpers)
- `include/*_metal.h` for each ported stage
- `src/pipeline_metal.mm`
- `src/debayer_metal.mm`
- `src/star_detect_metal.mm`
- `src/lanczos_metal.mm`
- `src/integration_metal.mm`
- Metal shader sources (e.g. `.metal`) grouped under `src/metal/`

Likely touched existing files:

- `CMakeLists.txt` (backend/build flags, macOS framework links)
- `include/pipeline.h` (backend enum + new entry point)
- `main.cpp` (backend selection flags mapping)
- docs (`README.md`, `CLAUDE.md`)

---

## 9) Testing Strategy

### Unit / stage tests

- Add Metal-targeted tests mirroring existing CPU/CUDA stage tests where practical.
- For deterministic checks, use synthetic fixtures with known outcomes.

### Integration tests

- End-to-end stack comparisons across:
  - CPU
  - CUDA (Linux/Windows)
  - Metal (macOS)

### Regression gates

- Existing tests must still pass unchanged on Linux/Windows.
- New Metal tests should be isolated by platform/backend capability checks.

---

## 10) Risk Register and Mitigations

1. **Build coupling to CUDA in root CMake**
   - Mitigation: isolate CUDA requirements behind option/conditions; do not require NVCC for Metal-only builds.

2. **Numerical drift due to precision differences**
   - Mitigation: stage-level golden checks and explicit precision policy.

3. **Performance regressions from frequent sync**
   - Mitigation: preserve asynchronous orchestration pattern and use profiling-driven sync minimization.

4. **Cross-platform regressions**
   - Mitigation: additive code paths, guarded compile options, unchanged defaults for current users.

---

## 11) Suggested Milestone Definition of Done

Milestone A (scaffold):

- macOS build (CPU + Metal stubs) succeeds.
- Linux/Windows build/test unchanged and green.

Milestone B (correctness):

- Metal backend produces numerically acceptable output on standard datasets.

Milestone C (performance):

- Metal backend demonstrates meaningful speedup over CPU on Apple Silicon.

Milestone D (production):

- CI coverage added, docs updated, fallback behavior robust.

---

## 12) High-Performance Prompt for AI Coding Assistants

Use the following prompt with Copilot/Claude Code when implementing this feature:

```text
You are implementing Apple Silicon (Metal) support in the gs18113/gpu-dso-stacker repository.

Primary objective:
- Add an Apple Metal backend while preserving existing Linux/Windows CUDA + CPU behavior.

Critical constraints:
1) Do NOT break existing CUDA path or CPU path.
2) Make additive, minimal, surgical changes.
3) Preserve numerical conventions exactly:
   - homography is backward map (ref -> src), do not invert
   - debayer luminance coefficients unchanged
   - kappa-sigma clipping semantics unchanged
   - edge/out-of-bounds handling consistent with existing implementations
4) Keep defaults backward-compatible for current users.

Implementation plan:
- Introduce backend selection enum in pipeline config (AUTO/CPU/CUDA/METAL).
- Add pipeline_run_metal entry point in parallel with existing pipeline_run / pipeline_run_cpu.
- Add Metal build path in CMake guarded by DSO_ENABLE_METAL option.
- Ensure Linux/Windows still build with existing CUDA settings.
- Implement Metal modules in phases:
  Phase 1: pipeline skeleton + CPU fallback compute for parity
  Phase 2: port Lanczos, debayer, star detection, integration kernels
  Phase 3: optimize and reduce synchronization
- Keep RANSAC on CPU initially unless profiling shows bottleneck.

Testing requirements:
- Run existing relevant tests before and after changes.
- Add targeted tests for new backend selection and Metal stage parity.
- Validate end-to-end output parity versus CPU/CUDA baseline with quantitative metrics (PSNR / max abs error / relative error).
- Ensure no regressions in existing test suites.

Code quality requirements:
- Follow existing style and error handling patterns.
- Do not add unrelated refactors.
- Prefer explicit resource lifecycle and robust cleanup on failure paths.
- Update README and CLAUDE.md with concise, accurate backend/build instructions.

Deliverables:
1) Code changes for backend abstraction + Metal scaffold
2) Incremental commits with small scope
3) Tests and validation outputs
4) Brief performance and numerical correctness summary
```

---

## 13) Practical Notes for Running This Work Incrementally

When implementing, iterate in small PR-sized chunks:

1. Build-system/backend enum scaffolding only.
2. Pipeline metal stub + safe dispatch.
3. One kernel at a time with stage tests.
4. End-to-end performance tuning after correctness locks in.

This sequencing minimizes risk and keeps review manageable.

