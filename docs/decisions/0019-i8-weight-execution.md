# 0019: i8 (w8) weight storage — execution path (per-channel scale, dequantized in the kernel)

- Status: accepted (2026-08-03); partly superseded by [0069](0069-packed-w4-storage.md) decision 1 (w4,
  2026-08-18), [0108](0108-container-format.md) (scale shape and declaration, `group_size`, I8 ordering,
  2026-09-22; see the addendum), [0097](0097-gemma4-qat-integration.md) / 0108 (mixed storage,
  2026-09-11 / 2026-09-22) and [0058](0058-numerics-opt-in-contract.md) /
  [0110](0110-practical-tier-numerics-contract.md) decision 2 (the condition for faster forms,
  2026-08-14 / 2026-09-26); the per-dtype asset directory withdrawn 2026-10-18. The numeric contract
  is in force (the base of 0069).
- Date: 2026-08-03
- Related: [0006](0006-quantization.md) (storage-only quantization, fake-quant, f32 bias, FQN matching,
  diagnostics) / [0018](0018-f16-weight-execution.md) (weight-variant generator, verification discipline)

## Context

This ADR maps the prototype's verified i8 design onto `WEIGHT_SLOTS` and
`packages/runtime/src/kernels/weight-storage.ts`, changing one point on purpose: where the scale applies.

## Decision

- **Method: per-channel symmetric int8.** `scale = clamp(amax / 127, f32 tiny)`, `q = round(w / scale)`
  **clamped to ±127** (−128 unused, so the largest-magnitude element is restored exactly and fake-quant
  is idempotent). No zero-point. The channel axis is the output channel: 0 for linear / conv1d / conv2d
  / conv3d / embedding, **1** for conv_transpose1d (transposed layout; `WEIGHT_CHANNEL_AXES`).
  Per-tensor scales SHOULD NOT be used (re-graded 2026-10-18 to SHOULD: the rejection rests on one
  prototype measurement of SBV2 voice quality, SNR 4.7–5.7dB, with no basis to generalize to other
  models). w4 was rejected on the same kind of measurement (SNR −1.5 to +5.1dB, systematically shorter
  utterances) and `group_size` was rejected at execution (superseded by ADR 0069 decision 1 on
  2026-08-18: that measurement covers SBV2 voice only; w4 is storage `i4` with K-direction groups).
- **The scale is a companion tensor** (F32), declared explicitly and validated at load (existence,
  dtype, shape) with fail loudly; key collisions with real tensor names are checked at emit. The shape
  and the declaration are now those of ADR 0108 (see the addendum).
- **GPU-resident, dequantized in the kernel**: the payload is bound as `array<u32>` and unpacked with
  **`unpack4xI8`** (core WGSL, no optional feature — consistent with ADRs 0002 / 0018). MUST: word and
  lane come from the **flat index** (`unpack4xI8(w[i >> 2u])[i & 3u]`); a row-relative index matches
  only by accident when the row length is a multiple of 4 (the same trap as f16's parity; the detector
  is **a test whose row length is not a multiple of 4**). The tail is zero-padded to 4-byte alignment.
- **The scale is applied per element on read**: `out = Σ x·(q·s) + bias`. GPU dequantization and the
  CPU expansion (the f32 path for ineligible slots) are **bitwise equal**, so ADR 0018's discipline
  (pinned by unit tests; E2E sees only implementation error) holds for i8. The prototype's form
  `(Σ x·q)·s` saves multiplies but differs by about 1 ulp in principle, turning verification into a
  tolerance argument. Cost: linear multiplies once when loading the shared-memory tile (not per MAC);
  conv hoists the loop-invariant `s`. The original condition "switching to the outer-scale form or DP4a
  requires revising this ADR and re-deriving tolerances" is superseded by ADRs 0058 (2026-08-14) and
  0110 decision 2 (2026-09-26): a faster form enters as an opt-in option with a contract class and
  leaves the default path unchanged (DP4a did so in ADR 0025, today `linearCompute: "a8"`).
- **Eligible = every slot in `WEIGHT_SLOTS`** (five at the time; conv3d added by ADR 0118). The
  prototype covered three ops; the reason (hand-written kernels) is gone with the shared
  `weight-storage.ts`. Embedding has no reduction, so per-row dequantization is trivially bitwise
  equal. Eligibility reuses `eligibleCompressedInitializers` unchanged (bias is never in the table).
  Ineligible i8 declarations are expanded on the CPU at load (`q·s` rounded once in f32 = the GPU
  value).
- **`weight-storage.ts` grows in three places only**: (1) `WeightStorage` gains `"i8"` (key
  discriminator `:wi8`; `WEIGHT_STORAGES` drives snapshots and the degenerate-shape harness);
  (2) the extra scale binding (the bind group layout differs per variant; its generation stays in
  `weight-storage.ts`, paired with the executor's bind entries); (3) `weightRead` takes the
  **output-channel scale expression** (ignored by f32 / f16 — **their generated WGSL MUST stay byte
  identical**). A bound local's expression is passed rather than a channel index (revised 2026-08-03 so
  conv can hoist the scale; a missing binding is a WGSL compile error). Scale bytes count in
  `residentCompressedBytes`.
- **Exporter**: fake-quant applies to the **effective weights** — MUST run after weight_norm removal,
  LoRA baking and the conv3d → conv2d slice, and before references are taken (unused elements would
  shift amax and the scale). Matching is by FQN plus the expected count (`id()` forbidden — ADR 0006);
  i8 requested with zero matches fails loudly. Emit writes the scale fake-quant used **as is**
  (recomputing breaks the golden through f32 rounding) and gates storage on
  `torch.equal(q8.to(f32)·s, t)`. "One dtype per target" is superseded by ADRs 0097 (2026-09-11) /
  0108 (2026-09-22) (mixed I2 / I4 / I8 and per-initializer encodings are implemented).
- Withdrawn 2026-10-18: "`models/anima-i8/` holds the transformer only" — distributions are containers
  and the storage dtype is chosen by a manifest preset (ADR 0109). Size then: DiT f32 7,465MiB → i8 ≈
  1,867MiB. conv2d / embedding `wi8` stay covered by a tiny golden plus unit tests
  (`packages/runtime/tests/gpu_f16_i8_weights_test.ts`).
- **Quality check**: with fake-quant, E2E measures only implementation error — **quantization quality
  is measured separately**: (1) port the DiT / VAE orthogonal-decomposition script (two denoise runs and
  four decode conditions, PSNR / relRMS); (2) generate w8 demo images and **decide by eye** (prototype:
  DiT-only i8 at PSNR 27.28dB was adopted by eye — diffusion changes trajectory, so numbers do not
  separate "degraded" from "a different picture").

## Alternatives considered

- Scale outside the reduction (prototype form): fewest multiplies, loses bitwise equality. Deferred.
- Per-tensor / w4 group quantization: rejected on the prototype's numeric and listening checks (w4
  reopened by ADR 0069; per-tensor see the SHOULD above).
- DP4a (`dot4I8Packed`): 4.73× in the prototype, but a new compute contract; opt-in in ADR 0025.
- i8 for every target: VAE / text savings are small (VAE −24.6MB vs DiT −1.87GiB) while verification
  cost grows; transformer alone plus tiny goldens suffice. Rejected.

## Addendum (2026-09-24) — ADR 0108 replaced the scale shape and declaration

ADR [0108](0108-container-format.md) decisions 11 / 17 / 18 (from 2026-09-22, IR v2 and the `krm`
container) replaced three storage statements:

- Scale shape and declaration: "keepdim broadcast shape of the weight's rank, declared by IR v1
  `storage.scale`" → the rank-2 group shape `[shape[rowAxis], rowLength / groupSize]` named by
  `encoding.scale` in the binding table (per-channel i8 is `[N, 1]`). The channel axis is the declared
  `rowAxis` (1 only for i8 `conv_transpose1d`); GPU-resident presets check it against the axis derived
  from the consuming op ([glossary](../glossary.md), `rowAxis`).
- `group_size`: "rejected when present" → the per-channel codec (`int8-sym`) MUST have `groupSize` =
  row length ([container-v1](../container-v1.md) §6); per-channel is a special case of the group form.
- I8 placed last: retired with the safetensors layout rules (ADR 0063).

The numeric contract is unchanged: ±127 symmetric per-channel without zero-point, per-element
dequantization on read (`out = Σ x·(q·s) + bias`), bitwise equality of GPU and CPU expansion, f32
bias, flat indexing and the fake-quant discipline.
