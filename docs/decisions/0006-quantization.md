# 0006: Quantization semantics and the initial scope

- Status: accepted (2026-08-01); partly superseded by [0009](0009-dtype-i32-bool.md) (2026-08-02:
  the declared IR dtype is no longer always f32), by [0028](0028-f16-compute-variants.md) /
  [0031](0031-attention-score-f16-storage.md) decision 7 / [0110](0110-practical-tier-numerics-contract.md)
  (activation f16 as an opt-in) and by [0025](0025-w8a8-linear-execution.md) decision 5 (no startup
  benchmark for w8a8). Storage-only quantization with f32 compute on the reference path is in force.
- Date: 2026-08-01
- Related: [0003](0003-ir-v1.md) / [0019](0019-i8-weight-execution.md) /
  [0069](0069-packed-w4-storage.md) / [0108](0108-container-format.md) /
  [docs/quantization.md](../quantization.md) (index of storage dtypes and presets). Sources: the
  prototype recon ([../research/2026-08-01-prototype-recon.md](../research/2026-08-01-prototype-recon.md)
  §6); the adopted methods follow the prototype's measured rulings (listening, SNR, accuracy checks).

## Decision

- The semantics are **storage-only quantization with f32 compute**. Quantization is confined to the
  storage description of the initializers (ADR 0003) and stays out of the graph semantics. (The
  declared IR dtype being always f32 was superseded by ADR 0009's semantic i32 and bool.)
- **Exporter-side fake-quant** is the reference for verification: weights are rounded to
  representable values before export and golden generation, which separates quantization error from
  implementation error. Storing a weight checks that dequantization is bit-identical.
- Initial scope (2026-08-01): **f16 and w8** (i8 with a per-channel f32 scale, `unpack4xI8`, no
  feature dependency). The current storage dtypes (`i4` — ADR 0069, `bf16` — ADR
  [0131](0131-bf16-weight-execution.md), the QAT integer formats — ADR
  [0097](0097-gemma4-qat-integration.md)) are indexed in [docs/quantization.md](../quantization.md).
- **Bias (and folded constants) are always read as f32** and are not part of the low-precision
  eligibility check. This is the root fix for a prototype bug: an f32 bias dragged its weight out of
  f16, so a 1024px DiT had 0 MB eligible, a 32 s load and no VRAM saving.
- Quantization targets are matched by **FQN (`state_dict` key) plus the expected count**. The match
  MUST NOT rely on `id(tensor)` identity, an implementation detail of `torch.export`.
- **The diagnostics are permanent**: the eligible bytes and the bytes expanded on the CPU are
  available after initialization, so "f16 requested but 0 MB eligible" is never silent (today
  `Session.diagnostics().storage` — `packages/runtime/src/runtime/session-types.ts`).

## VRAM effect (measured on the prototype)

The storage dtype also applies to the GPU weight buffers: eligible weights take ≈ 1/2 (f16) or ≈ 1/4
(w8) of the f32 VRAM; ineligible weights are expanded to f32 on the CPU and save no VRAM. Details:
[../research/2026-08-01-quant-vram.md](../research/2026-08-01-quant-vram.md).

## Not adopted or deferred (inherited from the prototype's measured rulings)

- Per-tensor i8 SHOULD NOT be used (re-graded 2026-10-18 to SHOULD NOT: the rejection rests on one
  measurement of one model — SBV2 voice, SNR 4.7–5.7 dB — and ADR 0069 decision 1 already limited
  the same kind of rejection of w4 to its measurement).
- w4: rejected at the time (SNR −1.5 to +5.1 dB, measured on int4 group quantization — attribution
  corrected 2026-08-18 by ADR 0069 decision 1). Group-wise w4 is enabled as storage `i4` by ADR 0069.
- w8a8 (DP4a): deferred at the time; implemented as an opt-in by ADR 0025. No startup benchmark: the
  dp4a and emulated variants are bit-identical, so ADR 0025 decision 5 picks one by
  `wgslLanguageFeatures`, for speed only.
- Activation f16: not adopted, because it blurs the separation of errors (superseded by ADR 0028 /
  0031 decision 7 / 0110, recorded 2026-10-18: not adopted on the reference path; allowed as an opt-in).

## Addendum (2026-09-24) — the storage description moved to the container

ADR 0108 decision 17 (IR v2 = IR v1 − `storage`) moved the storage description from the graph JSON
to the `encoding` of the container's binding table. Quantization stays outside the graph; the
numeric contract (storage-only, f32 compute, symmetric ±127 per-channel i8, f32 bias) is unchanged.

## Addendum (2026-09-24) — the reach of the core decision

The core of this ADR is the decision for the **reference path**. Later decisions stand outside it
without overturning it:

- **Reference path (this ADR)**: storage-only quantization, f32 compute. Every execution variant of
  `SessionOptions` defaults to the f32 reference path (ADR [0058](0058-numerics-opt-in-contract.md)).
- **Opt-in compute options**: variants that lower activation or compute precision are named runtime
  options that leave the graph unchanged — `linearCompute: "a8"` (ADR 0025), the f16 compute variant
  `:c16` (ADR 0028), attention a8 (ADR [0030](0030-attention-a8-execution.md)), w4a8 (ADR
  [0076](0076-w4a8-linear-execution.md)), packed int8 static SRQ activations (ADR
  [0105](0105-packed-static-quantize-activations.md)). A distribution that chooses a default
  execution form declares it in the manifest's preset; the graph stays unchanged.
- **The static SRQ of QAT is the model's own semantics**: the activation rounding of an official QAT
  model is part of its computation since training, so it enters the graph as the IR op
  `static_quantize` (ADR 0097 addendum 2; ADR 0108 decision 17). It is not PTQ storage; ADR 0025
  decision 1's reason against an IR op applies only to PTQ execution options.
