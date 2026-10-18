# 0120: Add w8a8 presets to the Wan2.1 DiT — reference preset `f16+dit8` and practical preset `f16+dit8-a8-attn8-s16`

- Status: accepted (2026-10-03 — the maintainer's rulings on the two open points, attention a8 to be
  confirmed visually; 2026-10-04 — the practical preset's quality passes the visual review and it
  becomes the default preset); partly superseded: decision 6 (default stays `f16`) by the ruling of
  2026-10-04 (4), generalized by [0128](0128-default-preset-original-weights.md) decision 1 on 2026-10-17;
  decision 3's r check is opt-in since 2026-10-17 ([0129](0129-test-layout-and-lanes.md) addendum
  (2)). Decision 1's "no intermediate presets" re-graded 2026-10-18 to SHOULD.
- Date: 2026-10-03
- Related: research [2026-10-03-wan-w8a8-recon](../research/2026-10-03-wan-w8a8-recon.md)
  ("investigation §n" below — source of the facts and numbers) / ADR
  [0118](0118-wan21-video-generation.md) (Wan2.1 adoption; this ADR takes over decision 7's "i8 / i4
  after measuring quality" and decision 8's numeric comparison) /
  [0110](0110-practical-tier-numerics-contract.md) (reference path and practical presets, self A/B
  check, classes of sha256 rows, quality review for the default preset) /
  [0025](0025-w8a8-linear-execution.md) (linear a8) / [0030](0030-attention-a8-execution.md)
  (attention a8) / [0031](0031-attention-score-f16-storage.md) (f16 storage of S) /
  [0076](0076-w4a8-linear-execution.md) (w4a8 — Anima's i4 preset dropped a8) /
  [0006](0006-quantization.md), [0019](0019-i8-weight-execution.md) (quantization stays out of graph
  semantics; references use fake-quant weights) / [0074](0074-quant-seat-naming.md),
  [0075](0075-quant-presentation.md) (preset naming and presentation) /
  [0058](0058-numerics-opt-in-contract.md) (addendum 2026-09-13, the visual review procedure) /
  [0106](0106-device-keyed-references.md) (sha256 rows per environment key) /
  [0109](0109-manifest-v5-container.md) (manifest `karume/5`, one container per storage label).

## Context

### Need

Add to the Wan2.1 T2V 1.3B DiT (ADR 0118) quantized presets with the same machinery as Anima's default
`f16+dit8-a8-attn8-s16`. Wan had one preset, `f16`, with an empty `session` (`WAN_QUANTS` in
`tools/export-recipes/wan/distribution.py`); ADR 0118 decision 7 deferred i8 / i4 until quality was
measured.

### Terms

| Term                                | Meaning                                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `dit8`                              | Component override in the preset name: the DiT weights (manifest role `transformer`) stored as i8 per-channel (ADR 0074 decision 4) |
| `a8`                                | `linearCompute: "a8"`: linear inputs quantized per token to i8, integer dot products with exact i32 sums (ADR 0025)            |
| `attn8`                             | `attentionCompute: "a8"`: attention q / k / P̃ / V in i8 with integer dot products (ADR 0030)                                   |
| `s16`                               | `attentionScoreStorage: "f16"`: attention scores S stored as f16 (ADR 0031)                                                   |
| w8a8                                | i8 weights × i8 activations (`dit8` + `a8`)                                                                                   |
| reference path / practical preset   | The runtime's omitted-value path / the opt-in path bundled by a preset's `session` (ADR 0110 decision 1)                      |
| reference preset / practical preset | Same weights with an empty `session` / with a non-empty `session`; the reference preset is the partner of the self A/B check |
| r (normalized ratio)                | ADR 0118 decision 8's metric: GPU error against the CPU f64 reference ÷ the CPU f32 reference's own error against f64         |
| relRMS                              | Relative RMS error; the measure of the self A/B check (ADR 0110 decision 5)                                                   |
| pass                                | One DiT forward; one generation is 100 passes (50 steps × CFG 2)                                                              |
| S                                   | DiT token count: 1,560 × latent frames at 832×480; 14,040 at 33 frames, 32,760 at 81 frames                                   |

### Findings of the investigation

- **No runtime addition expected.** The DiT weights are 307 linears, all with k divisible by 4 (the a8
  predicate). The 60 attentions (30 self, 30 cross) have no mask, no GQA, D = 128 and S divisible by
  4, so ①QK, ③PV and `s16` are eligible on every accepted shape. Row blocks (splitting S to fit the
  binding limit) × a8 / `s16` already have a Uint32 identity check.
- **No conv3d in the DiT**: the recipe replaces the patch Conv3d with `Linear(64→1536)`.
- **adaLN has one shared `time_proj` weight**; the per-block `scale_shift_table` is an additive f32
  table.
- **Weights 2.65 → about 1.33 GiB**; download 3.14 → about 1.72 GB.
- **i8 weights alone are not faster** (DiT is not weight-bandwidth bound); speed comes from a8 (Anima on
  the B570: linear 2.68×, attention 1.69×).
- **No a8 gain on Metal (M2)**: the i8a8 linear takes as long as f32.
- **K-75 does not apply**: the f32 GEMM K-reduction rounding (perf-ledger K-75) disappears with exact
  i32 sums; activation quantization error, an order of magnitude larger, replaces it.
- **Two layers of numeric checks**: `f16+dit8` (empty `session`) is the reference path, where GPU
  dequantization is bit-identical to the CPU expansion (ADR 0019), so ADR 0118 decision 8's r check
  applies. Presets with a8 are practical presets; their kernels are in contract class E (atol = 0
  against the TS reference).
- Pre-implementation estimates (pass time, VRAM peak) were replaced by the measurements of steps 3–4
  (addendum "steps 3, 4 and 5").

## Decision

### 1. Two presets — reference preset `f16+dit8` and practical preset `f16+dit8-a8-attn8-s16`

| Preset                  | `weights.transformer` | `session`                                                                       | Path             |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------- | ---------------- |
| `f16` (existing)        | `f16`                 | empty                                                                           | reference path   |
| `f16+dit8`              | `i8`                  | empty (i8 weights, f32 compute)                                                 | reference path   |
| `f16+dit8-a8-attn8-s16` | `i8`                  | `linearCompute: "a8"`, `attentionCompute: "a8"`, `attentionScoreStorage: "f16"` | practical preset |

- Intermediate presets (`f16+dit8-a8` / `f16+dit8-a8-attn8`) SHOULD NOT be added (re-graded 2026-10-18
  to SHOULD: the cost per preset still holds, but a sibling preset without a8 for devices without dp4a
  — ADR 0128 decision 5 — may be wanted).
- Short name `{"dit": "transformer"}` as Anima. Containers per storage label:
  `transformer/model.f16.krm` (existing path) and `transformer/model.i8.krm`.

Why:

- The reference preset is needed in the manifest as the difference floor's partner of the self A/B
  check; `referencePartnerOf` throws when none matches.
- Each preset multiplies the cost of visual reviews, sha256 rows and card entries (a Wan visual review
  took 15–30 minutes per video on the B570).
- Knob-by-knob isolation (a8 only, + attn8) is done with manifest overrides in tests
  (`withManifestOverride`, as Anima's fault injection; `knobRungs` in `e2e_wan_ab_test.ts`).

Protects: the difference floor of the self A/B check never becomes vacuous (the partner has the same
weights — ADR 0110 decision 5 ②); the running cost that grows with the preset count stays minimal.

### 2. i8 covers all 307 DiT linears; bias and norm stay upstream f32; the VAE is out of scope

- All 307 DiT linears (self / cross attention, FFN, five in `condition_embedder`, `patch_embedding`,
  `proj_out`) are i8 — the same scope as Anima's i8 series.
- Fallback: keeping `condition_embedder` / `patch_embedding` / `proj_out` in f16 if step 3 or the visual
  review found a problem, expressible with the exporter's `weight_dtype_overrides`. Not needed (ruling
  2026-10-04).
- Bias, norm weights and `scale_shift_table` keep the upstream f32 values; no f16 rounding enters the
  i8 series (`fake_quant_int8` rounds only `weight` of Linear and similar).
- No calibration: weights use per-channel RTN, activations per-token dynamic quantization.
- The VAE stays f16.

Why: all-i8 is the simplest and Anima's default passed the visual review with adaLN in i8 + a8; the
only ground for a mixed form was diffusers' layer-wise cast exclusion list, unmeasured on Wan. Rounding
bias / norm to f16 would add wording to the NOTICE. The VAE attention was excluded by ADR 0030 decision
7, and the VAE stage is 4% of an 81-frame run.

Protects: one rule for building the series, the same as Anima's i8 series. The NOTICE wording MUST
match the distribution.

### 3. Numeric check of the reference preset `f16+dit8` — ADR 0118 decision 8's r check at S = 192 and S = 14,040

- The reference is re-taken with i8 fake-quant weights (ADR 0006, 0019): CPU f64 and f32 goldens.
- The metric is r; the band is derived per S (worst of six decision cases × 5), accepted on separate
  cases, and fault injections must land outside it.
- **S = 32,760 (81 frames) is excluded**: its CPU reference costs about 5.5 hours for eight cases;
  i8-specific errors (scale axis, binding, dequantization) do not depend on S; the growth with S is a
  property of the GEMM K-reduction shared with the `f16` preset, whose S = 32,760 check (band 104)
  already watches it. S = 14,040 stays as the one full-size point, so the S = 192 band is not carried
  to full size.

**(superseded by ADR 0129 addendum (2) on 2026-10-17: the r check and its fault injections run only
with `KARUME_UPSTREAM_PARITY=1` — `packages/e2e/tests/e2e_wan_dit_test.ts:1912` (S = 14,040) and
`:1957` (S = 192). Parity was proven at implementation; later changes are caught by the sha256 rows
and the self A/B check. The default run keeps the measured-mode structure checks and the per-submit
GPU time limit.)**

Protects: "the i8 weights were ported correctly" is stated numerically; i8 dequantization, scale or
binding mistakes are caught on the reference path before the practical preset's check.

### 4. Check of the practical preset — the self A/B check at 33 and 81 frames

The self A/B check of ADR 0110 decision 5 in the same form as Anima's
(`packages/e2e/tests/e2e_anima_ab_test.ts`, `helpers/ab-gate.ts`); Wan's is
`packages/e2e/tests/e2e_wan_ab_test.ts` with `helpers/wan-ab-gate.ts`.

- **Partner**: the reference preset `f16+dit8` on the same machine with the same i8 weights; no torch
  golden.
- **Observation point**: the step 1 latent; generation stops once it is taken.
- **Difference floor**: red when bit-identical to the reference preset (MUST). `floor` goes above 0 only
  when removing one knob is shown to give clearly less difference than the full bundle.
- **Collapse ceiling**: measured relRMS × 2, derived per case (33 / 81 frames). Before derivation
  `band: undefined` makes the check print the measurement and fail. The ceilings 2.2e-1 / 4.2e-1
  (`e2e_wan_ab_test.ts:94-103`) were derived on the B570 and not re-derived on the RTX 3080 Ti; ADR
  0110 decision 5 does not require a per-machine derivation.
- **Census**: every knob's variant actually ran — exactly 307 linear rows in the census table; attention
  "at least one" (its row-block count depends on the device limit).
- **Determinism**: two independent practical runs are bit-identical (ADR 0110 decision 6).
- **Fault injections**: empty `session` → floor red; a different seed on the second run → determinism
  red; `attentionCompute` removed → census red (extended to five in the addendum below).
- **sha256 rows**: in the default lane (2 steps × 33 frames), a reference row for the reference preset
  and a practical row for the practical preset (ADR 0110 decision 7); 81-frame rows in the 50-step
  opt-in (`KARUME_WAN_FULL_PIPELINE=1`).
- **Per-submit GPU time**: the maximum per-submit GPU time in measured mode (`gpuTiming`) is at most
  1 s, measured per preset (`SUBMIT_GPU_LIMIT_MS`, `packages/e2e/tests/e2e_wan_dit_test.ts:451`,
  `PRACTICAL_TIMING_CASES` at `:442`). The value stays; its basis is now the Windows TDR (2 s) and
  the browsers' GPU watchdog, not the xe `job_timeout` of the former B570 machine (as ADR 0118
  decision 6).
- 81-frame A/B runs first, in descending order of allocation (on the B570 `destroy()` released late and
  a later probe ran out of memory — ADR 0118 step 8).

Why both frame counts: attention a8 error grows with the row length N (③PV's P̃ scale is fixed at
1/127, so sharp rows lose mass — ADR 0030 decision 6). Anima was accepted up to N = 4,096; Wan's
self-attention has N = 14,040 / 32,760, so 33-frame results are not extrapolated to 81 frames.

Protects: silent degradation of the whole bundle (floor), collapse (ceiling), per-knob degradation
(census) and determinism are caught at full size, long rows included.

### 5. Measure quality before the visual review by comparing presets on the GPU — no torch CPU shim

- Step 3 adds knobs one at a time on the same i8 weights through manifest overrides (`a8` →
  `+ attn8` → `+ s16`) and records the step 1 relRMS against `f16+dit8` at 33 / 81 frames. Values are
  recorded, not banded.
- Why: Anima's torch CPU shim (`measure_quant.py`, the Q0 of ADR 0025 / 0030) was the check before the
  kernels existed. The kernels are now in contract class E (atol = 0 against TS), so the GPU measures
  the real thing in minutes; a shim at S = 14,040 would take hours on the CPU.

Protects: per-knob contributions and the N dependence are seen on the real kernels before the visual
review.

### 6. Default preset stays `f16` — whether the practical preset becomes the default is ruled after the step 6 visual review

- This ADR leaves `defaultQuant` at `f16`; the default is ruled separately after the visual A/B (ADR
  0110 decision 8).

**(superseded by the ruling of 2026-10-04 (4) and by ADR 0128 decision 1 on 2026-10-17: the default is
the practical preset `f16+dit8-a8-attn8-s16` — `WAN_DEFAULT_QUANT` in
`tools/export-recipes/wan/distribution.py:343`; devices without dp4a get a warning, ADR 0128
decision 5.)**

### 7. The public API is only the existing `quant` option

- Presets are selected with `WanPipeline`'s existing `quant` option.
- The acceptance table `WAN_SESSION_POLICY` (`packages/models/src/wan/family.ts:231`) sets
  `linearCompute`, `attentionCompute` and `attentionScoreStorage` to `true` so the manifest's
  `session` may declare them.
- No explicit user options (`linearCompute` and the like) are added to the public surface; the third
  argument of `resolveSessionOptions` stays empty (`family.ts:419-424`). They can be added on demand
  since the acceptance table is shared.

Why: manifest overrides suffice for tests; no public surface without a use (ADR 0008).

Protects: a minimal public surface; a declaration outside the acceptance table still fails loudly
before the weights are fetched.

### 8. Steps and acceptance

Seven steps (table under "Acceptance"); independent of ADR 0118 step 9 (Chrome) and step 10 (umT5 on
the GPU). GPU steps (3–6) ran one at a time; the test runner now serializes test files
(`fileParallelism: false`, `vite.config.ts:40`).

Protects: each step can be accepted alone; GPU measurements do not interfere.

## Alternatives not taken

- **Anima's four presets** (`f16+dit8` / `-a8` / `-a8-attn8` / `-a8-attn8-s16`) — cost grows with the
  preset count; manifest overrides isolate knobs (decision 1).
- **Only the practical preset** — the self A/B check loses its partner; `referencePartnerOf` throws.
- **Mixed form with the condition path and the ends in f16 from the start** — grounded only in
  upstream's cast exclusion list; kept as the fallback (decision 2).
- **Round i8-series bias / norm to f16** — more NOTICE wording (decision 2).
- **i8 VAE** — ADR 0030 decision 7 precedent; 4% of the run.
- **r check at every S including 32,760** — about 7.6 CPU hours; the extra 5.5 hours re-measure a kernel
  property shared with `f16` (decision 3).
- **r check at S = 192 only** — would carry the S = 192 band to full size (decision 3).
- **No numeric check for `f16+dit8`** — the self A/B partner's correctness would be unchecked.
- **Self A/B at 33 frames only** — attention a8 error grows with N (decision 4).
- **Band relative to the `f16` preset's output** — different weights make the floor vacuous; rejected
  by name in ADR 0110 decision 5 ②.
- **Comparison with upstream bf16 execution as a check** — hours on the CPU, and diffusion trajectories
  diverge, so pixel distance does not measure quality
  ([research 2026-08-21](../research/2026-08-21-anima-i4-seat-speed.md)).
- **Port the torch CPU shim (Q0)** — the E-class kernels make the GPU measurement real (decision 5).
- **Go to the visual review without measuring** — cannot isolate per-knob contributions or N.
- **Make the practical preset the default in this ADR** — needed the visual review first (ADR 0110
  decision 8).
- **Expose `linearCompute` and the like** — no demand (decision 7).

## Consequences

- No runtime change (no new kernel, key or op); codegen snapshots unchanged.
- The `f16` preset does not change by one bit: the f16 series, `transformer/model.f16.krm` and the
  `f16` sha256 rows stay byte-identical.
- The distribution `models/karume-wan2.1/` gains `transformer/model.i8.krm` (1,426,806,948 B); an i8
  preset downloads 1.72 GB.
- Public surface: the types of `@karume/models/wan` are unchanged; `quant` accepts two more values.
- The affected-file list and the cost estimates of the plan are history; the measurements are in the
  addenda.

### Risks (investigation §6)

1. **Attention a8 on long rows (the largest risk)**: ③PV's P̃ is `round(127·exp(S − m))` with the row
   scale fixed at 1/127; in sharp rows, elements with `exp(S − m) < 1/254` round to 0 and mass is lost.
   Confirmed on measurement (addendum "steps 3, 4 and 5"); accepted by the visual review.
2. **adaLN and the condition path**: the shared `time_proj` input (M = 1) is quantized with one row
   scale, and its error reaches every token and block (speculation). Fallback: decision 2's mixed form.
3. **Temporal consistency**: per-token error differs per frame and could appear as flicker in still
   regions (speculation); a PNG contact sheet can miss it.
4. **Text context outliers**: T5-family activations have large outliers; per-token i8 errs more on
   those rows (speculation). Zero padding rows stay exactly 0 (ADR 0025 decision 2).
5. **Row blocks**: a8 / `s16` × row blocks have the Uint32 identity check; at 81 frames `s16` changes
   only the block count (24 → 12), not the VRAM peak.
6. **VRAM and late release on the B570**: see decision 4's ordering.
7. **No gain on Metal**: handled by ADR 0128 decision 5's dp4a warning.
8. **Timing**: a8 adds dispatches (Anima 3,087 → 3,311 per step), so the per-submit limit is measured
   per preset.

## Acceptance

| Step | Content                                                                                                     | Green condition                                                                                                                                                                                                                                                                                                       |
| ---: | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    1 | i8 series export (`wan2.1-t2v-1.3b-i8-dyn`, transformer only, bias / norm upstream f32) and goldens at S = 192 and S = 14,040 | The series binds i8 and not f16 / i4; 307 i8 weights, 700,480 scales; the f16 series and existing goldens byte-identical (safetensors compared by tensor); `uv run pytest` green in tools/exporter and tools/export-recipes |
|    2 | Distribution presets, card, NOTICE, local mirror, the three `WAN_SESSION_POLICY` keys, admission test, census rows | `verify_dist` green; three presets in the manifest; download measured; swapped storage (f16 series in the i8 role or the reverse) red at build; the `f16` entry and container byte-identical; admission fault injection red; GPU-free checks green |
|    3 | r check of `f16+dit8` (S = 192, 14,040); per-knob step 1 relRMS at 33 / 81 frames                            | r band per S from six decision cases × 5, acceptance inside, fault injections (the four of `f16` + at least one i8-specific) outside — opt-in since 2026-10-17; per-knob values recorded                                                                                                                               |
|    4 | Self A/B check (33 / 81 frames), census, determinism, fault injections, per-submit time, VRAM, pass time     | Ceiling = measured relRMS × 2 per case; full bundle inside, at least one bit different, no non-finite; census exactly 307 linears and ≥ 1 attention; two runs bit-identical; all fault injections red; max per-submit GPU time ≤ 1 s in measured mode; 81 frames first                                                 |
|    5 | sha256 rows (2 steps × 33 frames × 2 presets; 81 frames in the 50-step opt-in)                               | `KARUME_REFERENCE=write` creates the rows and a rerun matches; `f16+dit8` rows are reference rows, practical rows are practical rows (ADR 0110 decision 7); existing `f16` rows unchanged                                                                                                                              |
|    6 | Visual A/B (12 practical videos × 33 frames × 50 steps paired with the `f16` 12) and the default ruling      | The maintainer's ruling; an unperformed review is never recorded as done (ADR 0058 addendum); non-finite count and time recorded                                                                                                                                                                                      |
|    7 | Docs sync                                                                                                   | quantization table 3, perf-ledger, limitations, ADR 0118 pointers, glossary, ACTIVE_DESIGN, backlog and CHANGELOG match                                                                                                                                                                                               |

All seven steps are done (addenda below).

## Rulings (2026-10-03, the maintainer)

1. Visual A/B material is handed over as PNG frame sequences and contact sheets first; a synchronized
   player page or ffmpeg on demand.
2. **Attention a8 error: confirm visually first**; if rejected, switch the practical preset to
   `f16+dit8-a8`.
3. The default preset is ruled after step 6 (decision 6).

## Open questions

- **An i4 (w4a8) preset**: Anima's i4 preset dropped a8 after its visual review (ADR 0076 decision 6);
  whether Wan gets one, and with a8 or not, is open.
- **Gains in Chrome**: download, VRAM and a8 speed of the practical preset in Chrome (ADR 0118 step 9
  ran only the precomputed path with `f16`).
- Closed: the expected values (replaced by measurements in steps 2–4), attention a8's N dependence
  (step 3), the mixed fallback (not needed — ruling 2026-10-04).

## Addendum (2026-10-03) — results of steps 1 and 2

Step 1 (`6cb822a9`): i8 container 1,426,806,948 B (9 parts), 307 i8 / 521 f32 tensors; growth (S = 768)
is excluded from the i8 goldens because decision 3 does not list that S. Step 2 (`fff846a6` /
`c36e543e`): three presets; an i8 preset downloads 1.72 GB (as expected); two build checks were added
(`rope_base` byte-identical; the embedding asset fits both transformers' context inputs).

## Addendum (2026-10-03) — test implementation of steps 3 to 5, notes on decisions 4 and 5

- **Tests done** (`006504a4`–`cce2ec43`): the reference preset's r check switches
  `e2e_wan_dit_test.ts` by series root (i8 at S = 192, 9 cases; S = 14,040, 8 cases). The self A/B
  check is `e2e_wan_ab_test.ts`. sha256 rows `<preset>-2step-seed-boxing-cats-seed42` (row class derived
  from the manifest's `session`) plus 50-step opt-in rows for the practical preset at 81 / 33 frames.
- **Decision 4 — five fault injections**: ① empty `session` → floor; ② different seed → determinism;
  ③ `attentionCompute` removed → census; ④ census table count ± 1 → census; ⑤ a different seed's latent
  against the reference preset → ceiling (a proxy for collapse; ADR 0110's "scale × 2" example has no
  entry through the pipeline A/B and is implemented in the DiT r check as `to_v` per-channel scale × 2).
- **Decision 5 — no per-layer differences**: they would need an i8 per-layer probe series; step 3 keeps
  only the step 1 relRMS per knob. A per-layer attribution is filed separately when a problem needs it.

## Addendum (2026-10-03) — results of steps 3, 4 and 5: attention a8 error grows strongly with row length

- **Step 3, r check of `f16+dit8`** (`bb41b26c`–): band 54 at S = 14,040 (worst 10.7 × 5) and 30 at
  S = 192 (5.97 × 5); acceptance inside (3.71 / 4.64, 3.82 / 3.43); five fault injections outside (`to_v`
  scale × 2 at r 1.66e+4–4.45e+4). Max per-submit GPU time 270.4 ms at S = 14,040.
- **Step 3, per-knob step 1 relRMS (33 frames, against the reference preset)**: linear a8 only
  **2.9991e-2** → + attention a8 **1.04e-1** → + s16 **1.0676e-1**. Most of the error is attention a8.
- **Step 4, self A/B check**: practical preset step 1 relRMS **1.0676e-1 at 33 frames / 2.0956e-1 at 81
  frames** (maxAbs 0.284 / 0.540); ceilings 2.2e-1 / 4.2e-1. Floor, census (307 linears in i8a8,
  attention 480 / 1,560) and determinism green; five fault injections red; a different seed's latent at
  relRMS 1.38.
- **Step 4, speed and VRAM** (B570, DiT alone, normal mode): one forward **68.5 → 34.0 s at S = 32,760
  and 17.1 → 8.5 s at S = 14,040 (−50%)**; max per-submit 381.5 / 157.2 ms; allocation 6.17 → 5.46 GiB
  (81 frames), 5.15 → 4.02 GiB (33 frames).
- **Step 5, sha256 rows**: `f16+dit8-2step-seed-boxing-cats-seed42` (reference row) and
  `f16+dit8-a8-attn8-s16-2step-seed-boxing-cats-seed42` (practical row) on the B570, reproduced.
- **Risk 1 materialized**: Anima's practical step 1 relRMS is 1.2–1.4e-2 (ADR 0110); Wan's is an order
  of magnitude larger and nearly doubles from N = 14,040 to 32,760, caused by attention a8. Step 1
  relRMS is not quality, so the decision went to the visual A/B: (a) proceed and judge visually
  (recommended, −50% time); (b) if rejected, `f16+dit8-a8` (relRMS 3.0e-2); (c) a separate 81-frame
  preset without attention a8 (rejected by decision 1's cost argument).

## Addendum (2026-10-03) — step 6 material

12 practical-preset videos (seeds 42–45 × boxing-cats / ferret / cat-dog-baking, 50 steps, 832×480, 33
frames) on the B570, 951–952 s each against 1,815 s for `f16`; no non-finite values; paired with the
`f16` videos of ADR 0118 step 7.

## Rulings (2026-10-04, the maintainer)

1. **The practical preset's quality passes.** In the 12 pairs (`f16` vs `f16+dit8-a8-attn8-s16`) there
   is no clear degradation; the bundle (linear a8 + attention a8 + f16 S storage) is fixed and option
   (b) is not taken.
2. The default preset was left open at first (material: no a8 gain on Metal; the `f16` preset took 46
   minutes for 81 frames × 50 steps in Chrome on an RTX 5070 Ti — ADR 0118 addendum 2026-10-04).
3. Next: the two 50-step practical sha256 rows (addendum below).
4. **Default preset = the practical preset `f16+dit8-a8-attn8-s16`** (same day, ruling decision 6). The
   maintainer's policy: the default is "the most practical preset now" (light, small quality loss); a
   later switch to "the preset closest to the original weights" stays possible. `f16` remains as an
   explicitly selected reference-side preset; existing sha256 rows, bands and case ids keep their
   values by naming `f16` explicitly. ADR 0128 (rewritten 2026-10-17) generalized this ruling into a
   per-model procedure; the default stays the practical preset and devices without dp4a get a runtime
   warning (its decision 5).

## Addendum (2026-10-04) — switching the default preset

- `WAN_DEFAULT_QUANT` in `tools/export-recipes/wan/distribution.py` is the practical preset; the card's
  `(default)` and the Usage `quant` derive from the manifest; the card's quality line states the visual
  review result.
- **Tests name the preset explicitly in every case**: `e2e_wan_pipeline_test.ts` has `F16_QUANT`, and
  `loadPipeline` / `rowClassOf` / `FULL_CASES` require the preset (omission stopped by the type). Cases
  without a preset name in their ids run with `quant: "f16"`.
- **Values kept**: the sha256 rows, case ids, bands (`LATENT_RATIO_BANDS` / `FRAME_RATIO_BAND`) and the
  fault floors in `fixtures/references/wan.json` and the tests did not change by one character.
- A host test (`packages/models/tests/wan_pipeline_test.ts`) checks that building from the mirror's
  `karume.json` without `quant` resolves to the practical preset. The follow-up work (rebuilding the
  mirror, host checks, the GPU lane) was done; examples and gpu-lab were removed later
  (`bb35fe2a`, ADR 0127).

## Addendum (2026-10-04) — two 50-step sha256 rows of the practical preset

`KARUME_WAN_FULL_PIPELINE=1 KARUME_REFERENCE=write` on the B570 matched the `f16` rows and wrote
`f16+dit8-a8-attn8-s16-50step-boxing-cats-seed42` (33 frames) and `…-seed42-81f`. Wall time at 81
frames 3,703 s against 7,178 s for `f16` (DiT stage × 0.49; DiT stage VRAM peak 6.18 GiB against
6.87 GiB); at 33 frames 988 s against 1,835 s. The card states these values (`card.py:117`).
