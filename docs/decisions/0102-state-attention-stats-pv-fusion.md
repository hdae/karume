# 0102: Optional fusion of row statistics and PV in states attention

- Status: accepted (2026-09-15, within the maintainer's attention-optimization request); partly
  superseded by [0104](0104-gemma-fast-quant.md) addendum 2026-09-26 (`parallel-fused` is declared in
  the Gemma `i4-fast` presets, which are default presets — "defaults unchanged" and the M2 outcome
  "not promoted"); on 2026-10-18 the browser comparison selection and the M2 condition for promotion
  were withdrawn
- Date: 2026-09-15
- Related: [0067](0067-autoregressive-attention-vocabulary.md),
  [0058](0058-numerics-opt-in-contract.md), [0096](0096-speculative-decoding.md)
- Evidence: [research and measurements](../research/2026-09-15-attention-fusion.md)

## Problem and choice

Gemma E2B decode issues three dispatches per layer: QK, row statistics, PV. One kernel does the row
statistics and PV, saving the temporary row-statistics buffer and one dispatch. QK, append, state
ownership and commit order do not change.

The row maximum and denominator use 256 lanes and the PV partial sums 16 lanes, in the existing
order; the row-statistics lane number is `lid.y * 16 + lid.x`. Shared memory is overwritten with the
denominator only after every thread has read the maximum, and reused for PV partial sums only after
every thread has read the denominator. An empty row stays `(amax, inv) = (0, 0)`, padded rows stay
exactly 0, and the ring modulus stays the capacity.

## Option and compatibility

Add `"parallel-fused"` to `SessionOptions.stateAttentionReduce`; the meaning, default and WGSL
snapshots of `"sequential"` and `"parallel"` stay. The regular and QAT Gemma pipelines take it through
the existing option of the same name. The quant / `defaultQuant` and model defaults do not change at
this stage. (Superseded by ADR 0104 addendum on 2026-09-26: `stateAttentionReduce` is in the
manifest `session` vocabulary, and the Gemma 4 `i4-fast` presets declare `parallel-fused`; `i4-fast`
is the default preset of regular E2B and QAT E2B / E4B in the local mirrors on 2026-10-18.)

The fusion is limited to states shapes with `M <= 8` and a static column cap `colCap <= 1024`
(`stateStatsPvEligible`, `packages/runtime/src/kernels/state-attention-stats-pv.ts`); the cap is the
capacity for full attention and window − 1 + M for sliding. A short column does not select another
pipeline from the logical length at run time. Other shapes and readonly attention use `parallel`.
Row statistics are recomputed per depth tile, so long columns are not included unconditionally.

The summation order of unfused `parallel` stays, and real GPUs are accepted by u32 comparison; no
new identity with `sequential` is claimed. Tolerances and goldens are not updated; MTP verify row 0
and decode identity are checked separately.

## Memory and acceptance

`planStateAttention` decides the application and the intermediate bytes, shared by execution and
estimation. `EstimateOptions.stateAttentionReduce` estimates the same path; omitted, the estimate
follows the default reference path. The Gemma estimate receives the invariant settings used at
Session construction. No change to the stored IR or the manifest is needed.

Checks: non-finite classes, GQA, padding, row splitting, ring wrap-around and logical positions near
the u32 range in the unit; applied / not-applied keys through a Session; the memory estimate;
generation and MTP of the real model. Withdrawn 2026-10-18: the browser comparison selection and the
M2 condition for promotion (page not ported — ADR 0127 decision 5; defaults: ADR 0128 decision 1).

## Rejected alternatives

Widening the PV lanes to 64 / 256 slowed down with longer context, and 256 lanes exceeded the
absolute tolerance of 5e-6 in one 4096-column case; the check is not relaxed. Sharing normalized
weights gave no clear end-to-end gain, so the fusion without extra shared arrays came first. No
external source code is copied.

## Addendum (2026-09-15) — after the M2 check

The maintainer's 80 generations on M2 matched (regular E2B −0.88%, QAT E2B −0.16%): no case for
promotion, the option stayed opt-in (user M2 attention results in
[research 2026-09-15](../research/2026-09-15-linear-static-quantize-fusion.md)). ADR 0104 promoted it
on 2026-09-26 on Intel Arc B570 flag-bench measurements.
