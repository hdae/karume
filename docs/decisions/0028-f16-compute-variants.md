# 0028: f16 compute variants (shader-f16 — `:c16` for attention and linear)

- Status: accepted (2026-08-04); partly superseded by [0058](0058-numerics-opt-in-contract.md)
  (the default of decision 4 now rests on 0058 decision 2) and
  [0110](0110-practical-tier-numerics-contract.md) (the e2e check of decision 6 is gone; its shape
  is generalised in 0110 decision 5); the scope limit of decision 3 and the w8a16 rejection of
  decision 4 re-graded to MAY on 2026-10-18
- Date: 2026-08-04
- Related: [0018](0018-f16-weight-execution.md) (f16 **storage** — another axis) /
  [0021](0021-gpu-timing-diagnostics.md) decision 4 (amended: "no shader feature dependency" holds
  for the default path only) / [0022](0022-gemm-register-blocking.md) decision 3 /
  [0023](0023-fused-attention.md) / [0025](0025-w8a8-linear-execution.md) (opt-in knobs) / recon:
  [shader-f16-recon](../research/2026-08-03-shader-f16-recon.md)
- Checks (as of 2026-10-18): the numeric contract of decision 5 is
  `packages/runtime/tests/gpu_f16_compute_test.ts` (atol 0). The default path is guarded by the
  sha256 regression checks (for Anima, `packages/e2e/tests/e2e_anima_test.ts`).

## Context

Maintainer ruling (2026-08-04): implement and ship the f16 compute variants **even if they do not
win on the development machine**. The motive is browsers and GPUs with FP16:FP32 = 2:1; the
development GPU (GA102) is 1:1 and has no ALU gain (recon §1.4). Like the quantized variants
(w8, w8a8), f16 compute stays explicitly opt-in. Microbenchmarks are a record, not a check.

## Decision

1. **Feature wiring.** `AcquireGpuOptions.shaderF16?: boolean`; **`undefined` means "do not
   request"**, unlike the tri-state auto of `gpuTiming`. f16 compute changes the numbers; if the
   adapter's capability switched it on, changing the machine would silently change the output, so
   there is no auto mode. The plan is the pure function `planShaderF16Feature`; whether it is
   enabled is `GpuContext.shaderF16Enabled`, derived from the enabled features
   (`packages/runtime/src/gpu/acquire.ts`, `context.ts`).
2. **Run-time canary.** `acquireGpu` with `shaderF16: true` runs one tiny f16 kernel with a known
   answer and compares the result. **A listed feature is not evidence that f16 works.** The canary
   passes run-time values through an f16 shared tile, because a register round trip is removed by
   the compiler (see Detection limits) and would not prove that rounding happens. Rationale
   (restated 2026-10-18): a defence against implementation-dependent failures. The original trigger
   was `denoland/deno#23125` (an `enable f16` shader returns all zeros without a warning); Deno is
   now a secondary runtime (ADR 0127 decisions 1 and 3), but the canary is cheap and catches the
   same class of failure in any implementation (`acquire.ts:214`).
3. **Variant shape.** The GEMM skeleton (`packages/runtime/src/kernels/gemm.ts`) with **f16 tiles
   and f32 accumulation**. The rounding sequence is documented on `GemmCompute`: (1) round once,
   when writing the shared tile; (2) widen once, at register load (no per-MAC conversion);
   (3) accumulate in f32; (4) output f32 — with one exception: fused attention passes S as f16,
   which **halves the transient** (1,073.7 → 536.9 MB at 1024 px). The key word is `:c16`,
   distinct from the storage word `:wf16` (both can appear together). Wiring covers the **three
   attention kernels and linear**; the three kernels share the storage form of S and switch
   together.
   matmul / bmm / conv are not wired; they MAY be wired when measurements on another device show a
   gain (re-graded 2026-10-18 to MAY: the limit was a priority call from one device, where GEMM
   lost with f16 tiles; perf ledger K-68 re-measures on other devices).
4. **Knobs.** `SessionOptions.attentionCompute?: "f32" | "f16"` is added and `linearCompute` gains
   `"f16"` (the type is `ComputePrecision = "f32" | "f16" | "a8"` today —
   `packages/runtime/src/runtime/session-types.ts:223`). Both default to `"f32"` (MUST — the
   existing PNG sha256 checks and tolerances must not move). (Superseded by ADR 0058 decision 2
   on 2026-08-14: the default being the reference path is now stated there in general form.)
   w8a16 (i8 weights × f16 compute) is not implemented; it MAY be reopened when a speed
   measurement on an FP16 2:1 device shows a gain (re-graded 2026-10-18 to MAY: the rejection
   rested on the ALU ratio of one device — on a 1:1 device w8a16 gives up dp4a and only halves
   shared bytes; ADR 0029 found w8a16 equal to w8 in quality; perf ledger K-68). Until then the
   combination fails loudly at codegen (`CodegenError`, `gemm.ts:1825`) and at encode
   (`ExecutionError`, `packages/runtime/src/runtime/recipe-builders/linear.ts:222`).
5. **Numeric contract: atol = 0.** **The f16 variant's output ≡ the f32 variant's output on inputs
   rounded to f16** (same reduction order and rounding sequence, so bit-identical). The oracle lets
   the host do only the rounding (`roundToF16` = `Math.f16round`) and leaves the f32 reduction to
   the existing f32 kernel. A CPU rewrite of the f32 reduction would differ by the driver's freedom
   in FFMA fusion and `exp`. Reference: `packages/runtime/tests/gpu_f16_compute_test.ts` (bit
   equality on 13 shapes on a real GPU; to avoid a vacuous pass, every case also checks that the
   output **differs** from the f32 variant).
6. **The e2e check has three tiers** (`packages/runtime/tests/e2e_anima_f16compute_test.ts`): a
   ceiling (about 1.6 × the measurement), a **difference floor** (step 1 noise 1e-3 — a silent
   fallback to f32 makes the error *smaller*, which a ceiling alone cannot detect), and a key
   census (f16 key dispatch count; 0 f32 keys). A torch mirror of f16 compute is impossible (tile
   shape and reduction order differ), so decision 5 carries the numeric contract.
   (Superseded by ADR 0110 decision 5 on 2026-09-26: the file was deleted in the move to
   `packages/models` (2026-08-15), and the floor-plus-census-plus-ceiling shape is generalised
   there.)

## Measurements (2026-08-04, RTX 3080 Ti, Deno 2.9.4)

No time win on this device: attention_qk 0.91–0.92 ×, attention_stats 1.45–1.58 ×, attention_pv
0.97 ×, linear (wf16) 0.91–0.92 ×; DiT at 1024 px 0.92 ×, end-to-end even. GEMM slows down despite
halved bytes (instruction issue bound); only bandwidth-bound kernels win. The default path did not
move (PNG sha256, 74 WGSL snapshots, tolerances). Full record: commit `99a3c649`, this file.

## Detection limits

- **The compiler removes an `f32 → f16 → f32` round trip in registers** (fault injection: inserting
  it into the accumulation changes no output bit). Rounding is guaranteed only through f16
  **storage**; this implementation rounds by the shared tile's element type for that reason. A
  register-rounding design would have produced the same numbers as f32 while appearing to run in
  f16 — a silent failure. The canary uses a shared tile for the same reason.
- Rounding-position errors that the compiler removes do not show in the numbers; only the WGSL
  snapshots detect them. Leaving the tile in f32, however, turns five checks red (oracle mismatch,
  `assertDiffers`, snapshot, structural checks) — better than predicted.
- `Math.f16round` and a bit-level implementation (double → f32 → f16) differ on doubles that f32
  cannot represent (double rounding: 65519.999 gives 65504 vs Inf). Kernels always round f32
  values, so the contract is unaffected; this is a trap when replacing the oracle.
- The difference floor is a design that the w8a8 e2e check did not have: a lapsed opt-in (silent
  fallback to f32) reduces the error, so an upper bound alone cannot detect it.

## Consequences

- f16 compute variants are selectable where the feature and the canary pass (opt-in). On this
  device the value is the halved S transient; time is 0.92–0.98 ×. There are no measurements on
  browsers or FP16 2:1 GPUs, where a win is the motive.
- Bandwidth-bound kernels win with f16 (stats 1.58 ×), supporting recon option γ (f16 intermediate
  buffers for non-GEMM kernels) as the next candidate; starting it is a separate ruling.
- The default path did not change by one byte or bit (snapshots, PNG checks, existing tolerances).
