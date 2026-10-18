# 0011: Layout execution strategy — alias reshape and strided materializing copy

- Status: accepted (2026-08-02); the rule that keeps the elementwise / reduce / matmul codegen
  unchanged was re-graded to MAY on 2026-10-18
- Date: 2026-08-02
- Related: [0004](0004-execution-model.md) / [0014](0014-layout-ops-full-write.md) /
  [0016](0016-anima-chain-export.md) / [0118](0118-wan21-video-generation.md)

## Context

M0 had no notion of layout; two DeBERTa layers contain view 28 / permute 22 / clone 24 / expand 5 /
squeeze 5 / unsqueeze 13 ([recon](../research/2026-08-02-deberta-front-recon.md)).

## Decision

- **Ops that keep the element order (view / squeeze / unsqueeze)** are normalized to the single IR op
  `reshape` and run as a **buffer alias** (no copy). The arena counts alias references as ordinary uses
  for retain / release; alias sharing is derived by `planAliases`
  (`packages/runtime/src/runtime/fusion.ts`).
- **Ops that change the element order or need a materialized tensor (permute / expand / slice)** run on
  **one family of strided materializing-copy kernels**: the input is read through
  `(offset, strides[STRIDED_RANK])` and the output is always contiguous; `expand` uses stride 0.
  **`STRIDED_RANK = 4`** (`packages/runtime/src/codegen/strided.ts`). Rank ≥ 5 values are lowered
  first by the exporter's rank-lowering normalization (a precedent from the earlier prototype). The
  first premise (every DeBERTa value has rank ≤ 4) is gone; ADR 0016 keeps 4 for Anima, and ADR 0118
  admits rank 5 only for conv3d weights and keeps the value rank cap at 4.
- Addendum (2026-08-08, performance): an `expand` whose bound input and output shapes match exactly,
  rank included, is an identity map and runs as a buffer alias like `reshape`. A non-identity `expand`
  still runs as a strided copy. The check is a pure-function pass in `fusion.ts`; the count appears
  in `Session.diagnostics().lastRunFusions`.
- The elementwise / reduce / matmul codegen keeps its contiguous-input assumption for now. Fusing
  strided reads into the consumer kernel to remove the copy MAY be done when measurement justifies it;
  it needs only a different execution strategy, not a different IR vocabulary.
  (re-graded 2026-10-18 to MAY: the record itself called strategy B provisional, and the M0.1 codegen
  baseline it protected has since been changed — ADRs 0022 and 0023, the 2026-09-14 addendum below.)
- Ruling of 2026-08-02 (the maintainer): B is provisional and correctness-first. The performance goal
  is "correct and fast"; A is one candidate. The golden E2E tests on the same IR are the regression net.

## Alternatives considered

- **A: fused strided reads in every codegen from the start** (the prototype's approach). No copies,
  but it rewrites the M0.1 elementwise / reduce codegen and its determinism snapshots (the largest
  blast radius of P2).
- **B: metadata reshape plus materializing copy (adopted).**

## Rationale

1. **The more reversible side.** Same IR vocabulary; B moves to A by swapping the execution strategy.
2. **Smallest blast radius.** The M0.1 codegen and snapshots stay; new risk is one kernel family.
3. The P2 goal is a correct E2E. At B = 1 and T ≤ 512 the copies are MB-sized (the gather-index expand
   is two tensors of about 16 MB after CSE) and harm neither correctness nor VRAM.

Disclosure: the recon judged A to leave less debt for the SBV2 front (rank ≥ 5). Both options lower
rank ≥ 5 first, so A's advantage is fusion performance only — a deliberate deviation.

## Consequences

- New IR ops: `reshape` / `permute` / `expand` / `slice`. Identity `clone` and identity `repeat` are
  removed by the exporter's normalization and never reach the IR.
- The discard rules for alias values (flush-before-destroy / discard) are decided per backing buffer.

## Addendum (2026-08-02) — `slice` deferred, then reinstated

`slice` was deferred at implementation time (its only measured use folded away under ADR 0010). ADR
[0014](0014-layout-ops-full-write.md) withdrew the deferral and added `slice` for SBV2.

## Addendum (2026-09-14) — permute that keeps element order

Gemma (regular and QAT E2B) ran 100 `permute` copies in decode and 30 in prefill that kept the flat
element order. An axis of length 1 always has coordinate 0, so if the other axes keep their order the
contiguous buffer can be reused.

- Drop the length-1 axes of the bound input shape; if the remaining axis numbers in `dims` are strictly
  ascending, the `permute` is an alias. Swapping non-unit axes of equal length still materializes.
  Zero-element shapes are out of scope.
- `planAliases` derives aliases and their backing origin in declaration order; the execution plan and
  the memory estimate share it, and the recipe builder emits no copy for an alias. Retain / release,
  output pinning and borrowing of resident weights follow the existing `reshape` / `expand` aliases.
- Only values whose backing buffer is allocated inside the graph are newly aliased. The first
  `permute` reached from an input or a weight (or their `reshape` / identity `expand`) keeps its copy:
  otherwise the existing `copyOutputs` write-back to the same resident input becomes a rejected
  self-copy. The existing alias rules and the self-copy rejection are unchanged.
- No new IR, public option or numerics setting; WGSL, goldens and tolerances are unchanged. The
  dtype capability check of `permute` stays (i32 / bool are rejected even for shapes that could alias).
- Checks: all 2,127 (shape, dims) pairs of rank 1–4 with axis lengths 1 / 2 / 3 against flat indices
  (`packages/runtime/tests/runtime_permute_alias_test.ts`), plus bindings, empty tensors, special
  values on a real GPU, shared sources, resident weights, re-runs, `copyOutputs` and the estimate.

An RMS norm + RoPE region fusion was put on hold (small gain); see the
[research record](../research/2026-09-14-reference-rope-optimization.md).
