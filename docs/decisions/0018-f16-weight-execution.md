# 0018: f16 weight storage — execution path (GPU-resident, dequantized in the kernel)

- Status: accepted (2026-08-03); partly superseded — the eligible-slot list by `WEIGHT_SLOTS` (conv3d,
  [0118](0118-wan21-video-generation.md), 2026-10-02); the per-dtype asset directory withdrawn
  2026-10-18; bf16 added on the same skeleton by [0131](0131-bf16-weight-execution.md)
- Date: 2026-08-03
- Related: [0006](0006-quantization.md) (storage-only quantization, compute in f32; fake-quant is the
  truth; bias and folded constants always f32; diagnostics always on)

## Context

The storage schema was fixed in IR v1 (today the container's binding table, ADR 0108); this ADR
designs the **execution path**. Motivation (Deno's WebGPU, 2026-08-03): total GPUBuffer allocation
topped out at 7,280MiB, while the f32 weights of Anima's full 28-layer DiT need 7,465MiB (f16: about
3.7GiB) — see [limitations.md](../limitations.md).

## Decision

- **Eligible weights stay f16 and resident on the GPU**; dequantization happens in the kernel (ADR
  0006's consequence — only this halves VRAM; CPU expansion before upload only shrinks the download).
- **Reads use `unpack2x16float`** (core WGSL, no optional feature): the payload is bound as
  `array<u32>` and unpacked two elements at a time. Karume dropped optional features in M0.1 (ADR 0002);
  shader-f16 later arrived only as opt-in compute variants (ADR 0028), and weight reads keep the unpack
  form. Odd element counts are zero-padded by two bytes to 4-byte alignment (values unaffected).
- **Eligible = the weight slot of a fused op**, as listed by `WEIGHT_SLOTS`
  (`packages/runtime/src/ops/names.ts`). Each kernel has a **weight storage variant** (`w=f32` /
  `w=f16`) and the pipeline key carries the storage discriminator (determinism-key convention). Bias,
  norm weights and other initializers declared f16 are **expanded to f32 on the CPU at load**
  (correct, no VRAM saving — ADR 0006's ineligible rule). The 2026-08-03 list had five slots (linear /
  conv1d / conv2d / conv_transpose1d / embedding) (superseded by `WEIGHT_SLOTS` on 2026-10-02: ADR 0118
  added conv3d).
- **Diagnostics always on** (ADR 0006): the session reports GPU-resident compressed bytes
  (`residentCompressedBytes`) and CPU-expanded bytes, so "f16 requested but 0MB eligible" is not silent.
- **Fake-quant is the truth** (ADR 0006): the exporter rounds weights to f16-representable values
  before producing references and goldens. A unit test pins **bitwise equality** of the GPU unpack and
  the CPU f16 → f32 expansion; E2E sees only the implementation error.
- Withdrawn 2026-10-18: "the f16 asset series lives in its own directory (`models/anima-f16/`)" —
  distributions are containers now, and the storage dtype is a manifest preset (ADR 0109).
- Out of scope at the time: bf16 and i8 execution (i8 came in ADR 0019 the same day, bf16 in ADR 0131)
  and f16 activations (rejected in ADR 0006; attention-score storage later allowed by ADR 0031).

## Alternatives considered

- shader-f16 (`enable f16`, direct `array<f16>` reads): faster, but needs feature wiring and a fallback
  branch (duplicated pipelines). One unpack form keeps the first implementation narrow. Deferred.
- CPU expansion at load only (GPU always f32): VRAM does not shrink. Used only for ineligible slots.
- Every initializer f16-eligible: conflicts with the f32 bias rule (ADR 0006, the root fix of the
  prototype's demotion bug). Rejected.

## Addendum (2026-10-17) — bf16 uses the same skeleton

bf16 ([0131](0131-bf16-weight-execution.md), proposed) shares the `array<u32>` binding, eligibility and
odd-length padding; only the unpack differs (`bitcast<f32>` of the low / high 16 bits).
