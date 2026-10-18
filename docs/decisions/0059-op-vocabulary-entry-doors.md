# 0059: Entry-door model for the op vocabulary (layer numbers retired; one Core ATen test decides)

- Status: accepted (2026-08-14); amended by [0064](0064-entry-door-generality-axes.md) (two more
  axes; the inventory of decision 6 is a snapshot), inventory table of decision 6 withdrawn
  2026-10-18
- Date: 2026-08-14
- Related: supersedes [0043](0043-op-addition-layers.md) (layer definitions and procedure) /
  [op-vocabulary.md](../op-vocabulary.md) (the procedure) / [0012](0012-attrs-and-fused-ops.md) /
  [0015](0015-conv-family-extension.md) / [0040](0040-fusion-pass.md)

## Context

ADR 0043 decided whether an op needs an ADR from atomic / composite × inside / outside Core ATen ×
requester. The atomic / composite boundary ("exactly expressible, but does not hold up in memory or
speed") can split on a judgement of magnitude, yet it changed the procedure. Layer 1' patched this
with the requester axis, and the recon for new models found two holes: no place for a model's
atomic op outside Core ATen, and two documents drawing "does not hold up" differently. This ADR
changes only the written rule for future additions (implementations, assets and the place of
existing ops do not change) and closes both holes.

## Decision

1. **Layers have names, not numbers**: export-eliminated layer / Core ATen layer / extended atomic
   layer / extended composite layer / fusion layer. Old numbers: 0 = export-eliminated; 1 = Core
   ATen; 1' = extended atomic; 2 = split between Core ATen (attr variants of core ops) and extended
   composite (non-core); 3 = fusion. Old numbers stay in past ADRs and ledger NOTEs as snapshots and
   are not used in new text.
2. **Whether an ADR is needed depends on one test: Core ATen membership.** The test is the
   measured `torch.Tag.core in op.tags` (a machine test; the ledger records the command). Core ops
   and attr variants of them under another op name (`gelu_tanh` — a consequence of the empty-attrs
   contract, ADR 0012) belong to the **Core ATen layer: a ledger NOTE only**. IR names derived from
   core ops include overload variants (`ge.Scalar` → `ge_scalar` and the like) and pure renames
   (`_to_copy` → `cast`, `mm` → `matmul`, `view` → `reshape`) (inherits ADR 0015: the rest of the
   vocabulary additions need no ADR). The NOTE records, as before, the measurement that rejected
   steps 1 and 2 (folding, decomposition) (template: the `upsample_bilinear2d` NOTE). The commitment
   is bounded because it converges (160 Core ATen ops plus their attr surface). **Everything else
   (non-core aten, torchvision, Karume's own names) MUST have an ADR.**
3. **Atomic / composite is no longer a layer but the type of proof inside the ADR.** Extended
   atomic: proof that the op **cannot** be written as an exact equivalent from other ops of the
   vocabulary (`deform_conv2d`). Extended composite: it can be written, with the measured reason to
   keep it; the entry conditions inherit ① to ④ of the old layer 2 (reconstruction is fragile or
   loses meaning / materialising the intermediate does not hold up / recurrence / attr-variant
   separation) (`gru_scan`). **The requester (IR mechanism or model) does not split layers**
   (`sym_prefix_slice` and `deform_conv2d` go through the same door; one sentence in the ADR states
   the difference).
4. **"Does not hold up in memory or speed" does not change the procedure.** Inside Core ATen either
   answer lands at the same door. The engineering choice between decomposing and adding an op has
   no binding threshold; the procedure carries a **precedent guide**. Rewrites that barely add
   intermediates or FLOPs (reciprocal → division, softplus decomposition, real-valued RoPE)
   decompose by default. When intermediates grow, **there is already a precedent for keeping the op
   at 1.5–2×** (leaky_relu — ADR 0015, kept because the memory estimate breaks). Compositions that
   grow by orders of magnitude (upsample_bilinear2d FLOPs ×184, batch_norm 537 MB — the ledger's
   counter-examples) are grounds to **reject the decomposition**; whether to add the op is decided
   by demand (recurrence, entry conditions).
5. **Non-core ops kept from decomposition sit in the extended composite layer.** Of the 12 ops whose
   decomposition is forbidden, strictly 3 carry `Tag.core` (embedding / gelu / leaky_relu; torch,
   measured 2026-08-14). The 9 non-core ones (conv1d / conv2d / conv_transpose1d / layer_norm /
   linear / masked_fill / rms_norm / softmax / attention (SDPA, per target)) are composites whose
   decomposition the exporter chose to stop; they belong to the extended composite layer (grounds in
   existing ADRs and the ledger; no retroactive procedure). Consequence: **a non-core op (by the
   measured `Tag.core`, not by the spelling of its IR name) appears in the IR only when Karume
   chose it on purpose** (kept or invented), and the choice comes with an ADR. (Corrects the count
   "9 of 12 come from Core ATen" in the ADR 0043 addendum, which matched by meaning, such as conv1d
   → convolution.)
6. **Existing ops are re-placed without retroactive change** (no change to their place,
   implementation, assets or ADR requirement). Fused attention (ADR 0023) is not in the fusion
   layer: it is a run-time kernel variant of the IR op `attention`, and execution variants play no
   part in the vocabulary doors. Withdrawn 2026-10-18: the inventory table of 2026-08-14 (for
   example "the fusion layer is exactly four rules"); the current inventory lives in the source and
   the ledger (ADR 0064 decision 3) — `FUSION_RULES` in `packages/runtime/src/runtime/fusion.ts`
   holds seven rules today.
7. **Inherited unchanged**: the "better not to do" list (no additions for symmetry; filling the 160
   is not a goal) overrides every door; the fusion layer MUST have observation points (ADR 0040);
   the no-decomposition list is an axis independent of layers (ADR 0043 addendum, decision 3); the
   contract set of the Core ATen layer (`OP_CONTRACTS`, `ops.py`, `shapes.py`, fixtures, CPU
   reference, golden `COVERAGE` in `tools/exporter/src/karume/goldens.py`).

## Consequences

- The procedure in [op-vocabulary.md](../op-vocabulary.md) follows this model (door table, machine
  test, precedent guide, old-number mapping); dated NOTEs stay as snapshots.
- The two holes left known-issues (2026-08-14); this ADR holds their history. K-5 landed as a
  run-time variant of `attention` with no new door ([ADR 0060](0060-row-block-attention.md),
  2026-08-15).
