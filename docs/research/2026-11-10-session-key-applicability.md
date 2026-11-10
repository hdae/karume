> This document is a point-in-time snapshot (2026-11-09 investigation, recounted 2026-11-10) of which `session` keys have a
> place in the graphs of six families whose acceptance tables (`FamilySessionPolicy`) reject almost every key. It is the
> evidence behind the Context of ADR [0147](../decisions/0147-numerics-and-kernel-adoption.md) (decision 7, the derived
> applicability check). It is not a design source of truth. The recount script and the raw records of the investigation are
> local and untracked.

# Which session keys have a place in six families' graphs

## 1. Question

ADR 0111 decision 2 lets each family hold a hand-written table of the `SessionSpec` keys it accepts. In six families the
table rejects every key, or every key but one. A `false` entry can mean two different things: "no op in the graph could be
affected by this key" (a fact the graph shows) or "the key could take effect, but nobody verified it for this family". This
record counts which reading each `false` entry has.

## 2. Method

- **The tables** are read from the current source: `SBV2_SESSION_POLICY`, `BIREFNET_SESSION_POLICY`,
  `DEPTH_ANYTHING_SESSION_POLICY`, `SIGLIP2_SESSION_POLICY`, `VOWEL_DETECTOR_SESSION_POLICY` and
  `UMT5_ENCODER_SESSION_POLICY` (one per family under `packages/models/src/`). Each has the twelve keys of `SessionSpec`.
- **The graphs** are read from part 0 only (the header and the two descriptors; no weight block) of every container of the
  local distribution mirrors `models/karume-sbv2-jvnv` (26 containers: every speaker, component and quant),
  `models/karume-birefnet-hr` (2), `models/karume-depth-anything-v2` (1), `models/karume-siglip2` (2) and
  `models/karume-umt5-xxl` (1). The vowel detector has no distribution; its graph is read from the export output
  `outputs/series/vowel-detector-crnn-epoch3` (1 container). The script uses the runtime's own readers
  (`readContainerGraph` of the runtime test helpers, `planWeightResidency`) and the recipe builders' own predicates
  (`linearTakesI8a8`, `linearGemvParallelEligible`, `integerGemvVariantFor`, `planesGemvVariantFor`), with M = 1 … 8
  for the decode-only predicates.
- **Key → op.** Each key is matched to the ops whose recipe builder or fusion rule reads it:
  `convCompute` → `conv3d` with i8-resident weights; `attentionCompute`, `attentionScoreStorage`, `stateAttentionReduce`
  → the fused `attention` op (the builders of `bmm` and `softmax`, into which the other families decompose attention, read
  no option); `linearGemvReduce` → a `linear` whose (storage, n, k, group) is a row of the parallel-GEMV table;
  `fuseRmsNormAdd` → an `rms_norm` that feeds an `add`; `fuseLinearStaticQuantize`, `fuseGeluMulStaticQuantize`,
  `packedStaticQuantize` → a `static_quantize`; `integerGemv` → a `linear` fed by a packed `static_quantize` with a measured
  integer variant; `linearGemvPlanes` → an i2-resident `linear`; `linearCompute` → `"a8"` on an i8- or i4-resident `linear`
  with k % 4 = 0, `"f16"` on a `linear` whose weights are not i8 / i4 / i2.

## 3. Graph facts

| Family | Containers | `linear` weights (summed over the containers) | Fused `attention` | `conv3d` | `static_quantize` | `rms_norm` → `add` | i2 | Parallel-GEMV row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sbv2 | 26 | 200 i8, 200 i4, 24 f16 | 0 | 0 | 0 | 0 (no `rms_norm`) | 0 | 0 |
| birefnet | 2 | 396 f16 | 0 | 0 | 0 | 0 (no `rms_norm`) | 0 | 0 |
| depth-anything | 1 | 72 f32 | 0 | 0 | 0 | 0 (no `rms_norm`) | 0 | 0 |
| siglip2 | 2 | 246 f32 | 0 | 0 | 0 | 0 (no `rms_norm`) | 0 | 0 |
| vowel-detector | 1 (export output) | 5 f32 | 0 | 0 | 0 | 0 (no `rms_norm`) | 0 | 0 |
| umT5 | 1 | 168 i8 | 0 | 0 | 0 | 0 (`rms_norm` feeds the q / k / v linears only) | 0 | 0 |

In all six, attention is decomposed into `bmm` and `softmax`.

## 4. The table entries

The six tables hold 6 × 12 = 72 entries. 71 are `false`: every entry except sbv2's `linearCompute`.

| Reading | Entries | Which |
| --- | --- | --- |
| No op could be affected (the graph shows it) | 66 | the eleven keys other than `linearCompute`, in all six families |
| The key has a place for some value | 5 | `linearCompute` in birefnet, depth-anything, siglip2 and vowel-detector (`"f16"` on f16 / f32 weights; not verified), and in umT5 (`"a8"` on its i8 weights) |

umT5's `"a8"` is not unverified: it was measured and not made a preset (perf-ledger K-80 and K-81 — per-token int8 collapses
some rows). sbv2's accepted `linearCompute` has places for both values (`"a8"` on the i8 / i4 linears, `"f16"` on the f16
ones).

## 5. Limits

- The recount reads the local mirrors; a published revision can carry another graph.
- The vowel detector's graph is an export output, not a distribution.
- The decode-only predicates were evaluated at M = 1 … 8, the range their tables cover.
- The key → op mapping follows the builders and rules as of 2026-11-10; a new consumer of a key changes it.
