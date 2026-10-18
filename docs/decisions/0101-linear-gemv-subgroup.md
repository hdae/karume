# 0101: Run the parallel GEMV summation tree with value exchange inside 32 lanes

- Status: accepted (2026-09-13; opt-in only, not in the manifest `session` vocabulary); partly
  superseded by [0098](0098-linear-gemv-parallel.md) decision 4 and its addendum of 2026-09-26 (the
  shared shape table, which now holds unmeasured rows — decision 1's "25 measured shapes") and by
  [0128](0128-default-preset-original-weights.md) decision 2 C1 (2026-10-17: now the reason it stays
  opt-in, replacing decision 5's Deno / Chrome scope) and by [0127](0127-node-dawn-toolchain.md) /
  [0129](0129-test-layout-and-lanes.md) (2026-10-16: the real-GPU lane runs on Node + Dawn — the Deno
  SKIP in the verification list); on 2026-10-18 decision 6, the page selection
  after the M2 check and the browser entry point of the 2026-09-25 addendum were withdrawn
- Date: 2026-09-13
- Related: [0098](0098-linear-gemv-parallel.md), [0100](0100-rms-subgroup-reduction.md)

## Context and alternatives

After the M2 re-run of RMS `subgroup32` (about 0.5%, no case for a default), matrix work still
dominated the Chrome profile (2026-09-13). The existing `parallel` keeps partial sums in shared
memory and synchronizes the workgroup at each step; the 2 to 32 lanes of one output fit in one
32-lane subgroup, so exchanging values with the same input distribution and summation tree removes
synchronization without weight copies or more shared memory. Numbers: the
[research record](../research/2026-09-13-gemv-subgroup.md).

## Decision

1. Add `"parallel-subgroup32"` to `SessionOptions.linearGemvReduce`. The meaning, eligible shapes,
   WGSL and keys of the default `sequential` and of `parallel` stay. The new mode is limited to f32
   compute, INT2 / 4 / 8 storage, the 25 measured shapes and M = 1..8; other storage, shapes and
   prefill use the existing path, and f16 / a8 compute is rejected at construction. (Superseded by
   ADR 0098 decision 4 on 2026-09-26: the shape range is the static table shared with `parallel`
   (`linearGemvParallelLanes`, `packages/runtime/src/kernels/linear-gemv.ts`), now 47 rows, 22 of
   them unmeasured E4B rows added on 2026-09-26.)
2. The mode requires `acquireGpu({ subgroups: true })`, reusing ADR 0100's feature and known-answer
   checks; the Session also rejects missing device / WGSL features before the weights are uploaded.
   No automatic fallback. Acquiring the capability alone does not switch the computation.
3. 128 threads, a fixed 32 lanes. The output column comes from the subgroup ID and the invocation ID
   inside the subgroup; no correspondence with the local ID is assumed. The packed-word scan,
   dequantization and multiply-accumulate of each lane are those of `parallel`. `subgroupShuffleXor`
   runs with widths L/2 down to 1 and only the lower lane adds the matching partial sum; lanes of
   edge columns take part too. Bias is added once at the end. No shared partial sums or workgroup
   barrier. The key gains `subgroup32` to tell it from the reference kernel.
4. Bit identity with `parallel` holds within the measured range; rounding on untested GPUs is not
   guaranteed. The goldens and tolerances stay, and first-row identity for M = 1 / 4 / 8 is checked
   on the new path. Whole MTP, E4B, long contexts and other GPUs are not accepted by this check
   alone.
5. The regular and QAT Gemma pipelines pass the setting from their existing options to target and
   drafter; when they acquire the GPU they request the features only when the mode is set, and an
   external GPU is checked at Session construction. The quant storage vocabulary, recipes, default
   presets, distribution format and published revisions do not change. Deno lacked the features, so
   the scope was explicit use on Chrome. (Superseded by ADR 0128 decision 2 C1 on 2026-10-17 with
   the M2 measurement below: it stays opt-in because a default preset runs on baseline WebGPU and it
   was about 9.3% slower on M2. On Node + Dawn (ADR 0127)
   `packages/runtime/tests/gpu_linear_gemv_subgroup_test.ts` skips by adapter features and runs.)
6. Withdrawn 2026-10-18: the comparison page selection (not ported — ADR 0127 decision 5).

## Verification and stages

- Isolated unit experiment: short K, edge columns, every lane count, the FP64 tolerance band for
  M = 1 / 4 / 8, u32 identity with `parallel`, GPU timestamps on real shapes.
- Isolated end-to-end comparison: regular and QAT, English and Japanese prompts, repeated ABBA,
  saving speed and token / stop / text.
- Product checks: original snapshots kept, a snapshot for the new key, Session rejections and
  diagnostic keys.
- Deno skipped the new feature and Chrome ran the same checks. (Superseded by ADR 0127 decision 1 /
  ADR 0129 on 2026-10-16: the real-GPU lane runs on Node + Dawn.)
- The scope is decided by the M2 re-run; a gain on RTX is not treated as an M2 measurement.

Value exchange follows the
[WGSL `subgroupShuffleXor` specification](https://www.w3.org/TR/WGSL/#subgroupshufflexor-builtin):
the mask is a power of two of at most 16, uniform in the subgroup, and every lane takes part. No
external source code is copied.

## Addendum (2026-09-13) — adoption after the M2 check

The maintainer's 80 generations on M2 matched `parallel` in token / stop / text; regular was about
9.3% slower and QAT showed no speed-up. Not adopted as an M2 default; the option stays because the
QAT gain reproduces on RTX. Fewer shared-memory synchronizations are not assumed to speed up other
GPUs. Data: [M2 adoption record](../research/2026-09-13-m2-gemv-subgroup-adoption.md).

## Addendum (2026-09-25) — decision 3's multiply-accumulate spelled with explicit `fma()`

When addendum 4 (2026-09-20) of [ADR 0105](0105-packed-static-quantize-activations.md) changed the
parallel family to `acc = fma(x, d, acc)`, `subgroup32` kept `acc = acc + x * d`. Metal contracts to
fma per expression form, so decision 3 ("the same multiply-accumulate as `parallel`") did not hold
literally and the u32 identity check (`packages/runtime/tests/helpers/gemv-subgroup-check.ts`) had no
guarantee on Metal (full review of 2026-09-24, W-RT6-1 / W-AD3-2). `subgroup32` computes the same sum
as `parallel` through value exchange; it is not the reference path.

- `linearGemvSubgroupWgsl` uses the same `mac(fma = true)` as `parallel`. The key does not change
  (as in ADR 0105 addendum 4: the key identifies the pipeline cache entry; the snapshots carry the
  determinism check). The 3 snapshots were regenerated.
- Standing check: `packages/runtime/tests/linear_gemv_rows_test.ts` requires, over all 30 forms
  (storage × group × lanes), identical `acc = fma(…)` lines in `parallel` and `subgroup32` and no
  `acc + x * d` in the multiply-accumulate.
- RTX / Vulkan numbers are expected not to change (the mechanism of ADR 0105 addendum 4's 540-pair
  sweep: the compiler contracts). **On Metal the `subgroup32` output bytes change** (towards
  `parallel`); the M2 identity record of 2026-09-13 predates the change.
- Re-checked 2026-09-25 by the maintainer at checkout `8d3bd8b4`: u32 identity with `parallel` PASS
  on M2 (Apple metal-3, Chrome 153) and RTX 5070 Ti (Vulkan, Chrome 153), 180 cases / 28,080
  elements each. Withdrawn 2026-10-18: the browser entry point used for it (ADR 0127 decision 5);
  the u32 check runs in `packages/runtime/tests/gpu_linear_gemv_subgroup_test.ts`.
