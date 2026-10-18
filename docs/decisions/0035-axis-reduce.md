# 0035: Reduction axis for the reduce family (required `dim` attribute and a coalesced non-last-axis variant)

- Status: accepted (2026-08-05; the main check — Uint32 identity with the permute + row-reduce path
  — held in 9 cases and the sha256 checks stayed bit-identical, so no visual check was needed)
- Date: 2026-08-05
- Related: recon [2026-08-04-vae-axis-reduce-recon.md](../research/2026-08-04-vae-axis-reduce-recon.md)
  (attribution, the A3 design, §5.2 symbolic verification; its risk R4 was refuted — detection
  limit 1) / [0011](0011-layout-strategy.md) (layout strategy — the largest case of "remove the
  permutes") / [0015](0015-conv-family-extension.md) (required attributes) /
  [0033](0033-vae-fixed-tile-decode.md) and [0034](0034-dit-dynamic-tokens.md) (variable resolution)

## Context

The VAE decoder's `QwenImageRMS_norm` (L2 over the channel axis) was exported with a permute that
moved the reduction axis to the last dimension and back. Those 60 permute dispatches made up 99% of
the VAE's uncoalesced traffic (181 ms), and with the barrier-bound row reduce (39 ms) **17% of VAE
GPU time** (recon §2–3). The root cause was that the IR's `sum` reduced only the last dimension.

## Decision

1. **The reduce family (sum / amax / amin) has an attribute `dim`, and it is required** (no
   default; allowing omission would let an IR meant for the channel axis silently reduce the last
   dimension — the approach of ADR 0015). Negative axes are normalised at the exporter boundary; an
   axis outside the rank fails loudly on both sides (`reduceDim`,
   `packages/runtime/src/ops/shapes.ts`).
2. **A non-last axis uses a dedicated variant `reduce:v2:<op>:<from>><to>:axis:wg256`**
   (`axisReduceKey`, `packages/runtime/src/codegen/reduce.ts`): one thread per output, walking the
   reduction axis with the `inner` stride, so adjacent threads read coalesced and no barrier is
   needed. **The reduction order reproduces the row reduce's "256 lanes + bit-reversed binary tree"
   exactly, with a bit-reversed carry stack in one thread** (the 21-case symbolic verification of
   recon §5.2 is the design basis; the identity +0.0 is not folded away). The last dimension keeps
   the existing row kernel unchanged. **The branch MUST depend on the axis only**: allowing a choice
   by speed would make bit-invariance of the default path depend on runtime conditions
   (`buildRowReduce`, `packages/runtime/src/runtime/recipe-builders/elementwise.ts`).
3. **Permanent check: `packages/runtime/tests/gpu_reduce_axis_parity_test.ts`** — **Uint32 identity**
   with the permute + row-reduce path (real shapes with dim = 96 / 192 / 384 and rows on both sides
   of and at 65,535, a permute of cycle length 3, reduction length 300 > 256, amax / amin, bool sum;
   15 cases today, including NaN propagation and the ±Inf identity values), a check that the two
   kernels are routed by axis, and a guard against a vacuous pass.
4. **Exporter:** `_h_sum` and `_h_row_reduce` (amax / amin, extended in the same change for
   symmetry; `tools/exporter/src/karume/aten_handlers.py`) carry the single axis into attrs. Anima's
   channel L2 now calls `l2_normalize(x, dim=1)` (`tools/export-recipes/_shared/vae_rank4.py`)
   instead of the permute round trip. On 2026-08-05 the VAE decoder series, the SBV2 front series
   (which hold 3 bool sums over the last dimension, numerically unchanged) and 5 tiny goldens were
   re-exported.

## Measurements (2026-08-04 to 05)

VAE decoder IR 438 → 378 nodes (−60, exactly the norm's permute pairs); the 9 parity cases
Uint32-identical; **no tolerance changed** (re-derived values matched the old records to three
digits); sha256 checks unchanged. VAE keys: strided 191.25 ms / 71 dispatches → 5.80 ms / 11; the
reduce 41.10 ms (row) → 22.25 ms (axis variant, −46%; above the recon's optimistic 6–10 ms because
of the fixed 256 iterations per thread, as risk R3 foresaw). VAE wall time 1.6–1.7 → 1.4–1.5 s;
1024 px generation 22.5 → 22.0 s. Full record: commit `99a3c649`, this file.

## Detection limits

1. **Moving an axis without a layout change does not move torch's reference values by one ulp**
   (refuting recon R4): torch's permute is a view, so `permute + sum(dim=-1)` and `sum(dim=1)` reach
   the same reduction (`torch.equal` on four real shapes; the 26 tiny-golden io files
   byte-identical). **Loosening a tolerance because "the reduction order must have changed" needs a
   measurement first.**
2. The recon's census claim ("only the VAE decoder has `sum`") was wrong — a check that walks every
   IR (57 files) found the 3 in the SBV2 front. **Trust a census only after it walks the IR
   mechanically, not after a grep.**
3. Making `dim` required turns silent execution of old IR into a loud failure (fixed by re-exporting
   5 series). The required declaration prevents by structure the accident of "an old IR silently
   computing something else".

## Consequences

- The unsaturated uncoalesced strided reads (126 GB/s) and row reduce (124 GB/s) of the VAE are gone
  by structure; −60 dispatches per run also helps the non-GPU side.
- Reduction over any axis is a general capability, ready for future NCHW work (group_norm, batch
  norm statistics).
- The 3 tolerances of the f32 Anima series (assets deleted) were kept: torch's values are
  bit-unchanged, so a regenerated series is not expected to need re-derivation (regeneration is
  parked in the [backlog](../backlog.md)).
