# 0027: SBV2 f16 storage series

- Status: accepted (2026-08-03; listening check accepted 2026-08-04); partly superseded by
  [0029](0029-sbv2-i8-series-and-quant-quality.md) (decision 1, 2026-08-04),
  [0074](0074-quant-seat-naming.md) (decision 4) and [0129](0129-test-layout-and-lanes.md)
  (decision 5, 2026-10-17); decision 6 withdrawn 2026-10-18
- Date: 2026-08-03
- Related: [0018](0018-f16-weight-execution.md) (f16 storage) / [0013](0013-sbv2-chain-export.md)
  (SBV2 export, `--verify` exclusivity) / [0025](0025-w8a8-linear-execution.md) decision 5

## Context

The maintainer asked for quantization support in SBV2 (f16 was known to lose little quality). The
f16 execution path (ADR 0018) is generic; only exporter wiring, a series and checks were needed.

## Decision

1. The SBV2 recipe takes `--dtype {f32,f16}`. i8 is not added: all 5 targets are 86–90% conv1d
   (ADR 0025 decision 5), so w8 would only save size. (Superseded by ADR 0029 on 2026-08-04: the
   i8 series was adopted; `--dtype` accepts f32 / f16 / i8 / i4 —
   `tools/export-recipes/sbv2/export.py:137`.)
2. Fake-quant applies the shared `round_weights_to_f16` (`karume.quantize`) to the exported module
   itself, **after** `remove_weight_norm` and the patches and **before** the reference and golden
   are captured. This order is a MUST (`_fake_quant` docstring, `export.py:773`). `g` and
   `style_vec` are graph inputs and are not rounded.
3. **`--dtype` is emit-only**; the CLI rejects it with `--verify` (`export.py:1513`): the dec / voice
   `--verify` asserts bit equality around `remove_weight_norm` with the reference taken before it,
   which rounding after it (decision 2) breaks. ADR 0013 ruled out a front / flow-only exception.
4. Series `models/sbv2-f16/`: 470.34 → 237.57 MB = **50.5%** (front 53.8%: its baked
   relative-position table, 2.1 MB, is ineligible and stays f32). (Superseded by ADR 0074 on
   2026-08-25 and [docs/assets-layout.md](../assets-layout.md): the exporter writes
   `outputs/series/sbv2-F1-f16/` (`export.py:20`); the distribution ships presets — ADR 0128,
   Context.)
5. The e2e check is parameterised by series, with a **series × storage dtype check** (Detection
   limits). Measured f16 tolerances: dp atol 1e-6 / rtol 1e-5, front 3e-4, flow 2e-5, dec 2e-5,
   voice 1.5e-5 (`packages/e2e/tests/e2e_sbv2_test.ts`). (Superseded by ADR 0129 on 2026-10-17:
   the torch comparison runs only with `KARUME_UPSTREAM_PARITY=1`; the dtype check runs by default.)
6. Withdrawn 2026-10-18: the demo knob `--sbv2 {f32,f16}` went with the Deno CLI examples (ADR 0127
   decision 5). Listening (WAV) remains the maintainer's call.

## Detection limits

- **A mixed-up series root cannot be detected numerically**: both series have errors in the same
  decade, so each passes the other's tolerances (fault injection: 52 passed). Only the series ×
  storage dtype check detects it; DeBERTa has one (`compressedStorage`, `e2e_deberta_test.ts`),
  and the Anima check now runs on the distribution (`e2e_anima_test.ts`), not on a series root.
- **Whether the e2e check catches a forgotten rounding depends on the model.** SBV2's checkpoint is
  true f32, so skipping the rounding turns both the emit-side eligibility check and the e2e check
  red (atol exceeded > 31 ×). Anima ships BF16 weights, where f16 rounding is nearly the identity,
  so only the emit check catches it (`tools/export-recipes/anima/README.md`, fault-injection note).

## Consequences

- The whole SBV2 chain runs on f16 assets (50.5% of the size); listening accepted on 2026-08-04
  ("no degradation heard"). The i8 question deferred here went to ADR 0029. Measurements: only
  front's `logw_sdp` exceeded the f32 error (2.1 ×, spline path); full record in commit `99a3c649`.
