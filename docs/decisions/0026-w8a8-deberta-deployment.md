# 0026: w8a8 on DeBERTa (validation on a second model)

- Status: accepted (2026-08-03; listening check accepted by the maintainer on 2026-08-04); partly
  superseded by [0129](0129-test-layout-and-lanes.md) (the fake-quant golden comparisons of
  decisions 2 and 3 are opt-in, 2026-10-17) and [0127](0127-node-dawn-toolchain.md) (the demo
  knobs of decision 4 went with the Deno CLI examples, 2026-10-16)
- Date: 2026-08-03
- Related: [0019](0019-i8-weight-execution.md) (i8 storage) / [0025](0025-w8a8-linear-execution.md)
  (w8a8 execution — its decision 5 is the parent of this ADR) / [0045](0045-deberta-layer-trim.md)
  (DeBERTa layer trim) / [0110](0110-practical-tier-numerics-contract.md) (band derivation rules) /
  [dp4a-w8a8-design](../research/2026-08-03-dp4a-w8a8-design.md) §7
- Checks (as of 2026-10-18): the w8 check (`packages/e2e/tests/e2e_deberta_test.ts`) and the w8a8
  fake-quant golden check (`packages/e2e/tests/e2e_deberta_w8a8_test.ts`) exist; their comparisons
  with torch goldens run only with `KARUME_UPSTREAM_PARITY=1` (ADR 0129, addendum 2026-10-17 (2)).
  The third tier of the w8a8 check uses per-output relative RMS ceilings (addendum 2026-10-07). The
  output.0 / output.1 tolerance and the census counts are historical values from 2026-08-03.

## Context

ADR 0025 decision 5 named DeBERTa full-24layer as the second model for validating w8a8 (linear is
86.8% of its FLOPs; SBV2 is dominated by conv1d and cannot serve). The runtime already had the
opt-in w8a8 value of `SessionOptions.linearCompute` (spelled `"i8a8"` then, `"a8"` since 0.5.0 —
ADR 0074 decision 3). What was needed was an i8 series, the checks, and quality and speed
measurements. Whether this works with zero runtime changes is the substance of the generality test.

## Decision

1. **Series.** The DeBERTa recipe takes `--dtype` (f32 and i8 at the time; f32 / i8 / i4 today —
   `tools/export-recipes/deberta/export.py:106`). The i8 series is its own directory
   (`outputs/series/deberta-i8/{dev-2layer,full-24layer}/` — `export.py:142`; 24 layers
   334,545,336 B = 25.4% of f32). Fake-quant is applied to the exported wrapper itself, before the
   goldens are captured (the FQN discipline of `karume.quantize`).
   f16 is not added; it MAY be added when needed (re-graded 2026-10-18 to MAY: the SBV2 f16
   decision this waited for, ADR 0027, did not decide DeBERTa f16, and no work item is filed).
2. **The w8 (f32 compute) check.** `packages/e2e/tests/e2e_deberta_test.ts` is parameterised by
   series. The i8 series tolerance is measured: atol 7e-4 (5.7 × the raw worst case 1.23e-4). That
   it lands in the same decade as the f32 series (8.32e-5) shows that the golden was captured with
   the fake-quantized weights; a forgotten fake-quant would show the quantization error itself
   (1e-1 order, three decades higher). The torch comparison runs only with
   `KARUME_UPSTREAM_PARITY=1` (`e2e_deberta_test.ts:431`).
3. **The w8a8 check.** `packages/e2e/tests/e2e_deberta_w8a8_test.ts` compares against the
   fake-quant golden (`--act-quant` → `io-i8a8.<case>`; `karume.act_quant` is the numeric
   reference) and checks the diagnostic keys. Every tolerance, band, ceiling and count of this
   check is relaxed only on the basis of a measurement of the kind recorded below, never on the
   specification band of ADR 0106. Numeric detection power is concentrated in a strict
   tolerance on output.0 / output.1 (atol 5e-5; 1,150 × margin to the f32 path — see Detection
   limits). (Superseded by ADR 0129 on 2026-10-17: the golden comparison runs only with
   `KARUME_UPSTREAM_PARITY=1` (`e2e_deberta_w8a8_test.ts:345`); the census checks run by default.)
4. **Quality gate.** Per-layer SNR (numbers below) and listening on the SBV2 demo; the maintainer's
   listening verdict is final. Withdrawn 2026-10-18: the demo knobs `--bert {f32,i8}` /
   `--bert-compute` went with the Deno CLI examples (ADR 0127 decision 5).

## Measurements (2026-08-03, RTX 3080 Ti)

- Speed (T = 512, full-24layer): w8a8 makes the linear kernels 2.80 × and the GPU total 1.74 ×
  faster; wall clock only 1.16 ×, because the graph outputs all 25 hidden states and the readback
  (~52 MB) dominates. w8 alone does not speed up compute; it gives 3.2 × faster loading and 1/4 of
  the assets. On the SBV2 demo (T = 17) the BERT stage went from 279 ms to 228 ms.
- Quality: SNR falls by about 1 dB per layer; hidden[-3] (the layer SBV2 uses) is 20.7–23.3 dB.
  Listening (f32 / w8 / w8a8, same seed): accepted by the maintainer on 2026-08-04 ("no degradation
  heard"). Full tables: commit `99a3c649`, this file.

## Detection limits

- **The discontinuity of activation quantization saturates within a few encoder layers in one
  forward pass.** At the last layer, GPU i8a8 vs the torch fake-quant golden has maxAbs 1.46,
  about the same as the f32 path vs the golden (1.33): the two cannot be told apart. ADR 0025
  decision 6 ("another sample of the same distribution") reappears within 24 layers. A separation
  margin exists only at the exit of the first layer (output.1: i8a8 5.72e-6 vs f32 path 5.76e-2,
  10,058 ×). Deriving tolerances as "5–10 × the measurement" would admit the f32 path and make the
  check vacuous, so numeric detection power sits in the strict output.0 / output.1 tolerance and the
  diagnostic key census (192 keys).
- **The expected key count is not the module count.** DeBERTa-v2 reuses query/key_proj for the
  relative-position projection (`share_att_key`): 6 modules but 8 graph nodes per layer (16 for 2
  layers, 192 for 24). torch hooks fire per call, so the numeric correspondence with the
  fake-quant golden stays 1:1.
- `rel_embeddings.weight` is fake-quantized but stored as f32 because it is consumed outside a
  weight slot (ineligible; emit ignores the extra scale). The consumer is the data slot of a
  `layer_norm` (`norm_rel_ebd="layer_norm"`), not a linear input (corrected 2026-08-11 —
  [research/2026-08-11-deberta-size-recon.md](../research/2026-08-11-deberta-size-recon.md) §3).
  The conv1d weight of ConvLayer is i8-eligible (w8 execution, not w8a8).

## Consequences

- Generality is confirmed: w8a8 runs on a second model with zero runtime changes and gives linear
  2.80 × / GPU 1.74 ×. The opt-in design is unchanged.
- For SBV2's small T, the gains are loading (3.2 ×) and asset size (1/4), not compute.
- Quality: w8 and w8a8 both passed the listening check (2026-08-04). Numerically, hidden[-3] SNR is
  above the level accepted for Anima.

## Addendum (2026-10-07) — the third tier of the w8a8 check uses relative RMS error ceilings

After the GPU swap, `i8/full-24layer / case2` failed on one element of output.19 (maxAbs 3.544 >
atol 3). It was not a GPU defect: regenerating the fake-quant golden on the new CPU moves the same
element by 6.076, and CPU variants of the golden differ by up to 7.173. atol 3 was about twice the
last-layer value 1.46, applied with maxAbs to every intermediate output. Details: commit `b9d9ac01`
(this file); investigation notes in `.claude/reviews/2026-10-07_deberta-w8a8/` and
`outputs/diag/deberta-w8a8-2026-10-07/` (both untracked).

**Ruling (maintainer, 2026-10-07).** The third tier becomes a per-output ceiling on the relative
RMS error ‖gpu − golden‖₂ / ‖golden‖₂. The non-finite check (NaN / ±Inf on either side) stays on
every element. The first tier (output.0 / output.1, atol 5e-5) and the second tier (key census)
are unchanged.

- **Judgement.** For each of output.2–24: no non-finite value and relative RMS error ≤ that
  output's ceiling (0.020 for output.2 up to 0.33 for output.24). maxAbs is recorded next to it in
  `results.json` (`comparisons`) but not judged.
- **Derivation.** Per output, the worst relative RMS error over all legitimate samples and cases,
  × 2, rounded up to two significant digits. Samples: GPU a8 (4 cases), 6 CPU variants of the golden
  (AVX512 / AVX2 / DEFAULT × 1 / 6 threads), and 8 runs per case with a relative ±2^-23
  perturbation at the embedding exit (60 samples, no non-finite value). The GPU f32 path is not a
  legitimate sample. This follows ADR 0110 decision 5-3 (ceiling ≈ 2 × measurement) and avoids the
  MUST NOT of decision 4: one scalar (worst 0.162 × 5 = 0.81) would have admitted about 80 × the
  legitimate worst case at output.2. No theoretical bound is recorded, because activation rounding
  is discontinuous and none exists; a8 correctness belongs to the atol = 0 contract of
  `gpu_i8a8_test.ts`.
- **Where the table lives.** `COLLAPSE_CEILINGS` in
  `packages/models/tests/helpers/collapse-ceilings.ts` — a declared table, not a per-environment
  reference row (ADR 0110 decision 4). `packages/models/tests/collapse_ceilings_test.ts` (core
  lane, no GPU) pins the judgement; flattening, swapping or mis-scaling table rows turned it red
  (fault injection, 2026-10-07).
- **What it does not catch.** A × 1.25 scaling at deep outputs (11 of the 12 outputs
  output.22–24), a single-element jump in a middle layer, and a silent fallback to the f32 path.
  Those belong to the first two tiers and to `gpu_i8a8_test.ts`.
- **Only measurements of this kind justify relaxing a ceiling**; a specification band
  (ADR 0106) does not.
- **Re-deriving.** The CPU steps are `tools/export-recipes/deberta/collapse_ceilings.py`
  (`mirror` per ISA and thread count, then `derive`, which writes `derive.json` with `ceilings`).
  The GPU dump and fault-injection steps used `tools/diag/deberta-collapse.ts`, which was removed
  with the Deno tools in `bb35fe2a` (ADR 0127 decision 5); a re-derivation first needs a new GPU
  dump step.
- **Re-capturing the fake-quant golden.** The golden (`io-i8a8.<case>`) SHOULD NOT be re-captured
  on another machine's CPU; if it is, the first-tier tolerance is re-derived with it
  (re-graded 2026-10-18 to SHOULD: the strict first tier depends on the rounding direction of the
  CPU that captured the golden — on the new CPU output.1 of case0 / padded moves by 1.285e-3 — but
  the machine that captured it is gone, and the comparison is opt-in).
