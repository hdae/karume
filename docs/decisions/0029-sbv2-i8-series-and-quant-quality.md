# 0029: SBV2 synthesis network i8 series (w8a32) and measured quality of quantization patterns

- Status: accepted (2026-08-04; listening check accepted by the maintainer: "at least it does not
  collapse; only w8-real-gpu sounds a little different, but I do not perceive it as worse");
  the series directory of decision 1 superseded by [0074](0074-quant-seat-naming.md) and
  [docs/assets-layout.md](../assets-layout.md); decision 4 re-graded to MAY on 2026-10-18;
  decision 4's listening condition split by [0147](0147-numerics-and-kernel-adoption.md) decision 3 on 2026-11-10
  (option admission machine-only; the preset decision stays with the maintainer)
- Date: 2026-08-04
- Related: [0019](0019-i8-weight-execution.md) (the w8 method — execution path unchanged) /
  [0027](0027-sbv2-f16-series.md) (the series pattern; its decision 1 "no i8" is overridden here —
  maintainer ruling 2026-08-04) / [0025](0025-w8a8-linear-execution.md) decision 5 (SBV2 is not a
  w8a8 target — unchanged) / [0006](0006-quantization.md) (fake-quant)

## Context

The maintainer asked to try w8a32 on the SBV2 synthesis network (dp / front / flow / dec / voice),
expected to "work with some loss of audio quality", and to measure the degradation of w8a8, w8a16
and f16 as well. The prototype's per-channel i8 measurement had voice SNR 12.8–24.2 dB and was used
in practice.

## Decision

1. **Wire `--dtype i8` (the w8a32 series) into the SBV2 recipe** — zero runtime changes (the ADR
   0019 execution path suffices). Fake-quant applies the shared `karume.quantize.fake_quant_int8`
   (per-channel symmetric ±127) in the same order as f16, which is a MUST (after
   `remove_weight_norm` and the patches, before the reference and golden are captured).
   **Idempotence** on real weights (±127-closed quantization is a fixed point in f32) is pinned by
   pytest (`tools/export-recipes/sbv2/tests/test_export.py`, `fake_quant_int8` applied twice).
   Assets are 25.9% of f32 (121.81 MB; companion scales 0.42%).
   `--dtype` is emit-only (the same structural reason as ADR 0027 decision 3; the CLI also rejects
   i8 with `--verify`).
   The series was `models/sbv2-i8/`. (Superseded by ADR 0074 on 2026-08-25 and
   `docs/assets-layout.md`: the exporter writes `outputs/series/sbv2-F1-i8/`
   (`tools/export-recipes/sbv2/export.py:21`) and the distribution ships the `i8` preset — ADR
   0128, Context.)
2. **The e2e check adds the i8 series**, with its 5 tolerances derived independently from raw
   measurements (every target in the same decade as the f32 / f16 series — evidence that the
   golden was captured with the quantized weights; front alone is 1.4 × larger, the spline path,
   as with f16). **The series × storage dtype check is strengthened to equality of the set of
   compressed dtypes** (blocks i8 mixed into the f16 series), and for i8 both the declared scales
   and the actual ones are checked.
3. **The quality of w8a8 / w8a16 is measured by torch simulation before any execution kernel is
   built** (`python -m sbv2.measure_quant` in `tools/export-recipes` — as Anima's Q0). Activations
   are replaced **at op granularity** (`torch.nn.functional`): the patch layer calls
   `functional.conv1d` directly, so module hooks do not fire. Activation quantization of conv1d uses
   per-time-step granularity in tap-decomposed form, the finest granularity an implementation could
   use, so read it as an **optimistic upper bound**.
4. **Execution paths for w8a8 / w8a16 on conv1d (dp4a / f16 tiles) are not implemented.** w8a8 is
   closed on quality (below). w8a16 MAY be implemented if a speed measurement shows a gain;
   admitting either path as a runtime option needs machine checks only (ADR
   [0147](0147-numerics-and-kernel-adoption.md) decision 3 (1)); making it a preset is the maintainer's decision under ADR
   0147 decision 3 (2), whose human review is a listening check for audio (amended 2026-11-10), with the
   measurements below as the material (re-graded 2026-10-18 to MAY: w8a16 equals w8 in quality, so speed decides alone, and the
   negative speed finding rests on one 1:1 device — tied to ADR 0028 decision 4; perf ledger K-68).
   Today the SBV2 `i8-a8` preset sets only `linearCompute: "a8"`
   (`tools/export-recipes/sbv2/distribution.py:315`).

## Measurements (2026-08-04, same utterance and noise, BERT in f32 for every configuration)

SNR vs f32 / LSD (main metric): f16 40.5 dB / 0.36 dB (effectively transparent); w8 8.6 / 3.20;
w8a16 8.6 / 3.20 (≈ w8); w8a8 6.0 / 4.08 (clearly worse). w8 changes the utterance length
(198 → 196 frames); the main degradation is on the front side. Full table: commit `99a3c649`, this
file.

## Detection limits

- **The patch layer calls `functional.conv1d` directly.** Module-level hooks, quantization or
  measurement silently miss the FFN convs (12 in enc_p, 48 in flow) and understate the
  degradation. Op granularity is the only safe point of application (a side effect of the ADR 0013
  patch layer).
- **Waveform SNR reorders configurations in the 10–16 dB range** (it is sensitive to phase shift).
  Read LSD first and SNR second.
- The undetectable series-root mix-up of ADR 0027 recurs with i8; the only detector is the series ×
  storage dtype check (strengthened here to a set comparison).
- Checks that assume exact integer agreement of `w_ceil` (the torch parity form of the f32 demo) do
  not hold for the w8 series: the duration itself moves.

## Consequences

- The SBV2 i8 series (w8a32) runs: 25.9% of the assets, no runtime change. Numerically clearly
  worse (LSD 3.2 dB, length changes), but listening was accepted (2026-08-04). "Only w8 on the real
  GPU sounds a little different" matches the measured prosody change (198 → 196 frames).
- A conv1d w8a8 path is at a clear quality disadvantage (worse than w8, weak speed motive for
  SBV2); a w8a16 path would be built for speed only.
