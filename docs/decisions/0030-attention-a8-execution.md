# 0030: a8 for fused attention (int8 QK / PV activations with dp4a integer dot products)

- Status: accepted (2026-08-04; visual check accepted by the maintainer); partly superseded by
  [0058](0058-numerics-opt-in-contract.md) (decision 1 with [0128](0128-default-preset-original-weights.md);
  decision 5, addendum 2026-10-10) and [0120](0120-wan-dit-w8a8-seat.md) (decision 7: Wan DiT also
  uses attention a8)
- Date: 2026-08-04
- Related: [0023](0023-fused-attention.md) (fused attention) / [0025](0025-w8a8-linear-execution.md)
  (linear w8a8 — quantization parts, atol = 0 contract) / [0028](0028-f16-compute-variants.md)
  (`ComputePrecision`) / [0074](0074-quant-seat-naming.md) (`"i8a8"` renamed `"a8"` in 0.5.0) /
  design recon: [2026-08-04-attention-a8-design.md](../research/2026-08-04-attention-a8-design.md)
  (qk / stats / pv breakdown, the P̃ scale collapse, options A / B, the reasons for every ruling)
- Checks (as of 2026-10-18): `packages/runtime/tests/gpu_attention_pv_i8a8_test.ts` (①QK and ③PV;
  `gpu_attention_i8a8_test.ts` was merged into it in `80a8100b`) and the sha256 regression checks
  of `packages/e2e/tests/e2e_anima_test.ts`. The original e2e check (`e2e_anima_attn_i8a8_test.ts`)
  was deleted on 2026-08-15; its tolerances and floor are historical.

## Context

In 1024 px turbo w8a8, attention was the largest GPU cost (46%: qk 44.2 / stats 7.3 / pv 48.5%),
running at about 1/3.4 of linear's throughput. This ADR extends the dp4a path proven for linear
(ADR 0025) to QK and PV. Approved by the maintainer on 2026-08-04 (option B introduced in stages,
the Q0 quality simulation first, fall back to option A if PV fails).

## Decision

1. **Opt-in `SessionOptions.attentionCompute: "a8"`** (spelled `"i8a8"` until 0.5.0). The default
   stays `"f32"`; `"a8"` and `"f16"` are mutually exclusive values of one enum — q / k / v are all
   activations with no storage axis, so there is no product. It does not request `shader-f16` and
   is orthogonal to the storage dtype of the assets. (Superseded by ADR 0058 decision 2 on
   2026-08-14 and ADR 0128 on 2026-10-17: the runtime default — every knob omitted — is the
   reference path; the distributed default presets of anima and wan declare
   `attentionCompute: "a8"`.)
2. **①QK.** `quantize_rows` runs unchanged on q and on k (the reduction axis D is innermost and
   contiguous, the same indexing as linear), then the `attention_qk` i8a8 kernel (key
   `attention_qk:v<N>:i8a8:<geometry>:dp4a|dp4aEmu`; v3 today —
   `packages/runtime/src/kernels/attention-i8a8.ts:285`). The half scale (the √scale of ADR 0023)
   is applied **on the dequant side, not before quantization**, twice: folding
   `(qs·scale)·(ks·scale)` first is a MUST. **GPU vs the TS reference holds at atol = 0** (6 shapes,
   NaN row / column propagation, exact zeros on zero rows, dp4a and emulation identical).
3. **③PV.** **P stays unmaterialised**; the A-tile fill computes `qP = round(127·exp(S−m))`. **The
   per-token scale of P̃ collapses to 1/127 by construction** (the row max is exp(0) = 1; taking
   amax gives the same value — no room to adapt, no division, no clamp). V is transposed with the
   existing `strided` and passed through `quantize_rows` (rows = (b, h, d)), which yields the
   **per-column scale and the N-contiguous dp4a packing at once** (no new kernel; per-token scales
   would change along the reduction axis and are mathematically invalid). Dequant folds
   `f32(acc)·((inv·(1/127))·vs[d])`. The numeric contract has **two stages**: the pure function up
   to the integers at atol = 0 (V = identity reads qP out), and qP generation checked as "±1 step
   only and a mismatch rate below 2%".
4. **②Row statistics stay f32, unchanged to the byte.** Using the post-quantization sum as the
   denominator (design §4.3) was **refuted by the Q0 measurement** (cross-attn O relRMS 17 ×, PSNR
   19.4 → 16.0 — on sharp rows the f32 denominator, which shrinks by the mass lost to rounding, is
   the better approximation) and is rejected.
5. **Eligibility is per stage** (① needs D % 4 = 0, ③ needs N % 4 = 0 — the packing direction
   differs). A stage that does not qualify falls back to f32 silently (mixing allowed, as with
   linear's k % 4). The only detector is the key-count check (fault injection showed the numeric
   band and floor both stay green). A reduction length above `LINEAR_I8A8_MAX_K` (2^17) fails
   loudly. (Superseded by ADR 0058, addendum 2026-10-10: eligibility is decided by the pure function
   `attentionRoute` (`packages/runtime/src/runtime/attention-route.ts`), and the series checks test
   full eligibility as a ratio — `assertAttentionSeatsComplete`,
   `packages/runtime/tests/helpers/pipeline-census.ts`.)
6. **The quality gate runs a torch simulation first** (`tools/export-recipes/anima/measure_quant.py`
   configurations (d)–(h); the granularity MUST match the runtime implementation exactly). At
   512 px: median P̃ row peak/rms 12.35 ≤ 16, latent ratio ≤ 1.185 × (vs the accepted w8a8). **At
   1024 px the distribution shifts right** (attn1 median 29.4 = 1.84 × the 512 px value; rows at the
   structural limit √N exist), so the latent check was re-run directly at 1024 px: qk 0.960 × /
   pv 1.084 × / qkpv 1.066 × (≤ 1.5 ×), passed. The Q0 outputs were kept in
   `models/anima-demo/q0-attn-a8{,-1024}/` (outside git). The final gate is the maintainer's visual
   check (as in ADR 0025).
7. Scope = **Anima DiT only**. The VAE decoder is out (attention is 5.65% and hits image quality
   directly). SBV2 and DeBERTa do not use the fused attention op, so they cannot host it. No memory
   gain (S stays f32; the memory lever is the f16 hand-off of ADR 0028). (Superseded by ADR 0120 on
   2026-10-03: the Wan DiT presets also use attention a8 —
   `tools/export-recipes/wan/distribution.py:332`, and ADR 0128 lists the wan2.1 / wan2.2 default
   preset `f16+dit8-a8-attn8-s16`. The VAE exclusion and the absence of a host in SBV2 / DeBERTa
   still hold.)

## Measurements (2026-08-04, 1024 px turbo, w8a8, seed 42)

Normalised by the linear time of the same run: qk 3.05 ×, pv 3.43 ×, attention with quantization
and Vᵀ 2.56 ×, DiT GPU −30.7% (the upper end of the recon estimate); about −4% at 512 px. The
wall-clock "non-GPU fixed cost" of the time was later removed by
[ADR 0032](0032-deserialize-submit-and-optin-timing.md). In the 512 px e2e check, attention a8 alone
was 1,969 × from the f32 control but only 1.35 × with w8a8, so no floor fits and opt-in detection
rests on the key count. The default path did not move. Full record: commit `99a3c649`, this file.

## Detection limits

1. **The P̃ scale collapses by construction.** "P ∈ [0, 1], so it quantizes easily" is half true:
   the range is narrow but the in-row dynamic range is wide, and quality is set by the row's
   peak/rms (`relRMS ≈ (peak/rms)/440` matched measurements to 84%). The distribution shifts right
   with N; do not extrapolate a 512 px pass to 1024 px.
2. **GPU / TS qP mismatches were 0 / 62,088 on this device.** The expectation that `exp`
   implementation differences always cause mismatches was refuted, so the ±1-step assert is
   effectively vacuous; **the working detector is the mismatch-rate ceiling** (126 grid injections
   turned it red at 3.15%).
3. **The silent fallback (D % 4 / N % 4) is detected only by the key-count check** (injection showed
   the numeric band and floor stay green).
4. **A one-sided break of the out-of-range zero fill of the P̃ tile is undetectable** (the other
   side's zero cancels it — the same shape as ADR 0022 detection limit 1; breaking both sides turns
   red).
5. **Test data whose heads differ only in magnitude cannot detect a wrong payload base** (the
   quantized integers coincide). Per-head **patterns** are a MUST (fixed in the test docs).
6. The post-quantization denominator for stats is clean in theory but **worse in measurement**
   (decision 4): the "only the numerator shrinks" bias is in fact close to the true contribution of
   the lost mass (≈ 0).

## Consequences

- Attention, the largest GPU cost of 1024 px w8a8, is resolved (−30.7%). Linear is the largest cost
  again (61%); further levers are bandwidth and non-GPU fixed cost (recon §11: f16 intermediate
  buffers, fusing Vᵀ and column quantization into one kernel to save 56 dispatches).
- attention_stats rises to 22.6% of attention. f16 measured 1.45–1.58 × there (ADR 0028), but the
  S storage form is incompatible with i8a8 — recorded only.
- Images differ from f32 / w8a8 (trajectory divergence), but the **visual check was accepted by the
  maintainer** (2026-08-04, "the change is at the error level", four before/after images at 512 and
  1024 px). This matches Q0's relative measurement (latent ratio ≤ 1.1 ×): the perceived difference
  is smaller than the "different picture" of w8a8 (ADR 0025).

## Addendum (2026-10-10) — the detector of decision 5

Eligibility is decided by `attentionRoute`; the series checks test full eligibility as a ratio
([ADR 0058](0058-numerics-opt-in-contract.md), addendum 2026-10-10 — see decision 5).
