> This document is a point-in-time snapshot (measurements of 2026-11-15) of one candidate way to ship a LoRA: a
> quantized weight delta. It is not a design source of truth: the candidate is on hold (the maintainer, 2026-11-15) and
> its revival condition is in [backlog](../backlog.md), section "parked". The script and its output are under
> `outputs/diag/quantized-delta-2026-11-15/` (local, untracked); the numbers below are that output.

# Quantized weight delta — size measurement

## 1. What was asked

Merging a LoRA into the upstream weights and quantizing gives `Q(W + ΔW)`. The candidate ships the integer difference
between that and the distributed `Q(W)`: the user adds the difference to the distributed int8 weights and gets the same
bits as merging before quantization (ADR [0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md)
decision 10), without the upstream weights. The question was how large that difference is.

## 2. Reading

- As a distribution format in place of a LoRA file it is too heavy: 16–41% of the int8 payload for the real pairs
  (322–809 MB of 1,956 MB), which is the size of a LoRA file or several times it. On hold.
- As a way to store a converted int8 per LoRA it would save roughly 6 to 8 tenths against a full copy.
- The strength is fixed when the difference is made, and several differences do not combine exactly.
- Not measured: a real LoRA file, and `int4` storage.
- By-product: re-quantizing the bf16 base with the exporter's formula reproduces the distributed int8 base exactly on the
  sample (Python only; the TS encoder of ADR 0152 decision 10 does not exist yet).

## 3. Measurement

Command (CPU only, 5 min 42 s): `cd tools/exporter && uv run python ../../outputs/diag/quantized-delta-2026-11-15/delta_size.py ../../models 6`

Sample: every 6th int8 weight tensor of the Anima transformer (76 of 454 tensors, 306,708,480 of 1,956,454,400 bytes).
"Size" = deflate (zlib level 6) of the int8 difference plus 4 bytes per changed scale, as a fraction of the int8 payload;
the MB column scales that fraction to the whole transformer (1,956 MB).

Re-quantization check: Q(bf16 base) with the exporter's formula equals the distributed int8 base — 0 mismatches in 306,708,480 elements.

### Real pairs (distributed fine-tune vs anima-v1.0)

| Fine-tune | median ‖ΔW‖/‖W‖ | non-zero | beyond ±1 | deflate | order-0 entropy | MB |
| --- | --- | --- | --- | --- | --- | --- |
| anima-turbo-v1.1 | 0.0030 | 19.8% | 0.4% | 16.4% | 11.5% | 322 |
| anima-aesthetic-v1.0 | 0.0023 | 27.7% | 0.9% | 19.8% | 14.4% | 387 |
| anima-turbo-v1.0 | 0.0050 | 40.3% | 3.6% | 25.0% | 19.2% | 489 |
| anima-copycat-20260610 | 0.0132 | 64.9% | 23.2% | 36.6% | 30.6% | 716 |
| anima-wai-v1.0 | 0.0193 | 71.3% | 34.6% | 41.4% | 35.4% | 809 |

Energy of ΔW in its top 32 singular values (median over sampled layers): 0.83–0.87 for four pairs (NaN for turbo-v1.1: a sampled tensor is identical to the base).

### Synthetic rank-32 delta on the bf16 base (Gaussian factors, scale re-derived)

| ‖ΔW‖/‖W‖ | non-zero | beyond ±1 | deflate | order-0 entropy | MB |
| --- | --- | --- | --- | --- | --- |
| 0.003 | 8.9% | 0.1% | 10.5% | 6.7% | 205 |
| 0.01 | 28.3% | 1.0% | 20.5% | 15.1% | 402 |
| 0.03 | 62.2% | 18.0% | 35.3% | 29.0% | 690 |
| 0.1 | 87.1% | 63.6% | 55.8% | 49.4% | 1,091 |
| 0.3 | 95.5% | 86.7% | 71.2% | 68.2% | 1,392 |

Keeping the base's scale instead of re-deriving it changes the size by less than 1 point at every magnitude.
