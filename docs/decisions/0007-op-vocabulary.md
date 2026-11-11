# 0007: Op vocabulary (Core ATen population of 160 and the extension discipline)

- Status: accepted (2026-08-01; carrying over the ledger was the maintainer's explicit instruction);
  record corrected by [0043](0043-op-addition-layers.md) (of the row-reduce family only sum / amax /
  amin were implemented first); the golden part of the one-set rule re-graded to SHOULD on 2026-10-18.
- Date: 2026-08-01
- Related: [../op-vocabulary.md](../op-vocabulary.md) (the canonical ledger), the prototype recon §5.

## Decision

- The population of the op vocabulary is **the 160 ops of the Core ATen IR** (`torch.Tag.core`). The
  five categories (not needed 25 / primitives 75 / compositions 38 / nice to have 7 / hard 15; 120 to
  implement) and the implementation order from the prototype's study are carried over into
  [docs/op-vocabulary.md](../op-vocabulary.md) and maintained in Karume from then on.
- The IR op vocabulary is a **frozen allowlist**. Adding an op is an explicit act: the ledger update,
  the contract tables and the CPU reference MUST land as one set; a torch golden SHOULD be added in
  the same set (re-graded 2026-10-18 to SHOULD: the required evidence is the TypeScript oracle —
  ADR [0129](0129-test-layout-and-lanes.md) decision 3 — and whether op-level goldens stay required
  is open in ADR 0129). The current set is listed in docs/op-vocabulary.md.
- **Implement the row-reduce family (amax / amin / max / min / argmax / argmin) first** (the
  prototype's largest gap). Corrected by ADR 0043: only sum / amax / amin were implemented first.
- **Ops preserved from decomposition (kept for fusion)**: the exporter has a decomposition-suppression
  list from the start — linear, layer_norm, softmax, gelu, conv1d, conv2d, conv_transpose1d,
  embedding, masked_fill (9 ops). ADR [0015](0015-conv-family-extension.md) added `leaky_relu` (its
  decomposed form inflates intermediate buffers 1.5–2×), ADR [0017](0017-rms-norm-conv2d-clamp-min.md)
  added `rms_norm` and ADR [0118](0118-wan21-video-generation.md) decision 1 added `conv3d`. The
  canonical list is `karume.convert.PRESERVED_OP_PREFIXES` (12 ops; 13 since ADR
  [0138](0138-moss-image-dit-vae.md) decision 7 added `group_norm`, updated 2026-11-11); attention is preserved only
  per target through `PRESERVED_OP_PREFIXES_WITH_ATTENTION` (`tools/exporter/src/karume/convert.py`).
- **A judgement that an op "can be decomposed" MUST pass the torch comparison check.** Adversarial
  checks in the prototype proved many decompositions wrong (missing log1p, pow with a negative base,
  NaN in max, var correction, inf − inf = NaN).
- Scatter ops **MUST NOT be implemented with CAS** (WGSL has no `atomic<f32>`, and a
  non-deterministic order of additions breaks the determinism invariant). Static indices are inverted
  into gather. A general scatter_add needs its own ADR.
- Kernel contracts (acceptance limits such as B = 1) do not leak into the IR specification; they are
  expressed as **runtime capabilities with clear diagnostics** (a lesson from v0).

## Consequences

- The ledger tracks its open measurement items (`sign(NaN)`, zero-length slice bindings; WGSL NaN
  comparison was measured on 2026-08-02).
