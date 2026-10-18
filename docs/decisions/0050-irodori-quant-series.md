# 0050: Irodori quantization series — phase 1 (f16) layout and checks

- Status: accepted (2026-08-12, phase 1 = f16; addenda: phases 2–3 on 2026-08-12, the `w4`
  preset on 2026-08-24, the 0.5.0 renaming on 2026-08-25); partly superseded by
  [0074](0074-quant-seat-naming.md) and [0128](0128-default-preset-original-weights.md) (preset
  names and the default, decisions 2 / 7 / 10), [0110](0110-practical-tier-numerics-contract.md)
  (decision 9 on 2026-10-17, the scope of decision 13) and [0129](0129-test-layout-and-lanes.md)
  (decision 4, 2026-10-17); decision 5 re-graded 2026-10-18
- Date: 2026-08-12
- Related: [0006](0006-quantization.md) (fake-quant method) / [0018](0018-f16-weight-execution.md) /
  [0027](0027-sbv2-f16-series.md) (the earlier f16 series) /
  [0029](0029-sbv2-i8-series-and-quant-quality.md) (series × storage dtype detection limits) /
  [0048](0048-irodori-host-port.md) / [0049](0049-irodori-codec-integration.md) / recon =
  [research/2026-08-12-irodori-quant-recon.md](../research/2026-08-12-irodori-quant-recon.md)

## Context

The Irodori-TTS v4 distribution was a single f32 series (8 graphs, 3.44 GB). The quantization
machinery (f16 / i8 storage execution in the runtime, the dist quant table, session wiring in
models) was complete from SBV2 / Anima; only the exporter's dtype axis and the quality checks were
missing. The recon measured that 99.8–99.9% of the weight bytes of all 8 graphs sit in weight slots
(eligible for compression). The maintainer ruled on three phases (2026-08-12): phase 1 f16 →
phase 2 i8 + a quality script → phase 3 w8a8.

## Decision (phase 1)

1. **f16 is a separate series** (`outputs/series/irodori-v4-small-f16/` / `dacvae-32dim-f16/` —
   `default_out_root` in `tools/export-recipes/irodori/export.py:2936`). Fake-quant is applied
   **after the effective weights are fixed and before the reference / golden are captured**:
   right after `load_*` in the DiT export and `pipeline_ref`; in the DACVAE export after
   `fold_weight_norm` and `lift_snake_alphas`, with check 3 (bit equality across
   `remove_weight_norm`) settled on the unrounded reference and the golden re-captured after
   rounding. **Snake alphas are not rounded** (lifted constants outside the weight slots; the
   golden also uses f32 alphas).
2. The distribution gets one file per role and storage dtype (as Anima / SBV2; today
   `<role>/model.<dtype>.krm` — `tools/export-recipes/irodori/distribution.py:197-201`).
   Presets f32 / f16, no mixing; defaultQuant stays f32. (Superseded: the preset set grew in
   decisions 7 and 10, the names follow ADR 0074, and the default is chosen by the procedure of
   ADR 0128 decision 1 — today `i8-a8`, `IRODORI_DEFAULT_QUANT` in `distribution.py:362`.)
3. **Series × storage dtype is pinned from both sides.** The existing presence check
   (`assert_storage`) **lets f16 assets into the f32 preset**, because compressed containers still
   hold bias / norm / graph constants in F32. `assert_storage_absent` with a forbidden table closes
   that direction; the presence check closes the other. The graph-declaration check
   (`assert_irodori_graphs`) runs on every dtype series.
4. **The latent check is parameterised by series**; each series derives its tolerance from its own
   measurement (ADR 0027's form; no reuse). An underived series is **red with the measurement
   printed, not SKIP**; S / forwards must match exactly for every series. Measured on 2026-08-12:
   f16 z maxAbs 2.1321e-4 (full) / 1.8516e-4 (no-ref) → `F16_Z_ATOL = 1.5e-3`
   (`packages/e2e/tests/e2e_irodori_latent_test.ts:124`). (Superseded by ADR 0129 decision 4 on
   2026-10-17: the latent comparison runs only with `KARUME_UPSTREAM_PARITY=1`
   (`e2e_irodori_latent_test.ts:306-309`); the rule applies inside that opt-in run.)
5. **The WAV sha256 check covers f32** (`QUANT = "f32"`, `e2e_irodori_wav_test.ts:118`); rows for
   other presets MAY be added. (Re-graded 2026-10-18 to MAY: f32-only was the check design of the
   time; sbv2 keeps rows for three presets (`e2e_sbv2_wav_test.ts`), and regressions of the default
   preset are caught by the self A/B check (ADR 0110 decision 5).) The f16 verdict is the
   maintainer's listening check on WAV pairs with the same text and seed. The upstream comparison
   (`EULER_REFERENCE_ATOL = 1e-3`, `irodori/pipeline_ref.py:143`) is **common to all series**:
   rounding hits the graph path and the upstream path once each, so it cancels.
6. **i8 / w8a8 are later phases**, decided on S drift (an i8 duration graph may change utterance
   length) after `measure_quant` measures it. (Resolved by the addendum of 2026-08-12.)

## Consequences

- f16: 1,721,312,568 of 3,438,182,144 bytes (**50.1%**), DiT resident 1.46 → 0.73 GB; listening
  check 2026-08-12 (plain and voice clone): "almost the same audio" — f16 accepted.
- The dist step needs every dtype series of the presets (no f32-only assembly —
  `distribution.py:557-567`). Compute stays f32 (ADR 0018); f16 only saves load time and VRAM.

## Addendum (2026-08-12) — phases 2 and 3: the w8 / w8a8 presets and the default

`measure_quant` (torch CPU simulation, 9+1 configurations × 2 cases) and real-GPU runs settled
decision 6: S did not drift in any configuration, the degradation is dominated by the DiT, and the
maintainer accepted all four presets by listening (details in commit `4d7253bc`).

7. **Four presets**: f32 / f16 / **w8 = all i8, no mixing** / **w8a8 = the w8 bytes** with an
   activation-quantized linear session. The preset → (storage dtype, session) mapping lives in one
   place, `IRODORI_QUANT_SEATS` (`distribution.py:309`); today the session is
   `{"linearCompute": "a8"}` and reaches only the DiT sessions (`dit` / `dit_context`). Default =
   `w8a8`. (Superseded: names per ADR 0074 — `w8` → `i8`, `w8a8` → `i8-a8`; default per ADR 0128
   decision 1.)
8. **The f32 preset forbids every compressed dtype**: i8 assets also hold bias / scale in F32, the
   i8 version of the phase-1 hole. Today f32 roles forbid f16 / i8 / i4 and i8 roles forbid i4
   (`IRODORI_STORAGE_FORBIDDEN`, `distribution.py:234`).
9. The w8a8 check cannot be a numeric parity net (ADR 0025 decision 6): exact S / forwards /
   latentDim against the w8 golden, a band `zBand` [0.1, 6] and a key census. (Superseded by ADR
   0110 decision 5 on 2026-10-17: `e2e_irodori_w8a8_test.ts` was removed (`80a8100b`); the census
   moved to the self A/B check `packages/e2e/tests/e2e_irodori_ab_test.ts`, which compares `i8-a8`
   with `i8` on the same device. The w8 latent tolerance `W8_Z_ATOL = 1e-2` remains.)

## Addendum (2026-08-23 / 24) — `w4` preset (i4 DiT only, GPTQ-calibrated)

The three quality iterations (R1–R3) and their measurements are in
[research/2026-08-24-gptq-expansion-quality.md](../research/2026-08-24-gptq-expansion-quality.md).
The maintainer approved two changes on 2026-08-23: "no mixing" in decision 7 is relaxed (the S
drift behind it disappears with GPTQ —
[research 2026-08-20 §6](../research/2026-08-20-gptq-awq-calibrated-rounding.md)); and "no RTN
preset first" (2026-08-20) is withdrawn in favour of widening GPTQ (a k-means preset, perf-ledger
Q-2, can be added later).

10. **Five presets**: the four above + **`w4` = i4 g32 for the DiT only**. DiT weights: the 168
    in-block linears other than adaLN are GPTQ i4 (gptq-rtn grid, 12 calibration cases × 40 steps);
    149 are i8 (144 adaLN — `attention_adaln` / `mlp_adaln` — plus 5 outside the blocks: `in_proj`,
    `out_proj`, `cond_module.{0,2,4}`); bias / norm stay f32. Since the DiT split (ADR
    [0114](0114-irodori-dit-context-split.md)) the 168 are 96 in `dit` and 72 in `dit_context`.
    The other 7 roles share the w8 bytes; the session is `{}` (w4a8 is unmeasured for irodori and
    was rejected on quality for anima, ADR 0076 decision 6). The default stays (w4's warm synthesis
    is the slowest, +38%; it exists for size and bandwidth). (Superseded for the default as in
    decision 7.)
11. **The three-way i4 split comes from listening attribution** (research §1): RTN i4 on the 5
    outside-block linears and a calibration corpus that did not generalise caused the muffling
    (→ i8; 12 cases, kept apart from the evaluation inputs); adaLN is sensitive and i4 shifts
    prosody (→ i8, +13.1 MiB). One implementation, `irodori.calib.is_adaln` (`calib.py:187`),
    drives both the calibration include and the storage split; **a missing adaLN fails loudly**,
    so an upstream rename cannot silently break the ruled layout.
12. **The golden is baked from the shipped bytes**: `pipeline_ref --dtype i4` reads the shipped
    container back (`restore_dit_from_i4_series`, with provenance, shape, count and preset checks).
    The w4 latent tolerance is `W4_Z_ATOL = 6e-3` (5.2 × the R2 measurement 1.1522e-3).
    `calib_provenance.json` is checked unconditionally at dist plan time (`method == "gptq"` —
    keeps plain-RTN series out; `assert_irodori_calib_provenance`, `distribution.py:875`).
13. **Simulation does not transfer to the shipped rig** (research §2): the measurement rig's A/B
    proves effects inside that rig only; the shipped rig re-solves GPTQ rounding and re-draws the
    utterance. Delicate properties (prosody) are therefore judged on the shipped bytes, not on the
    simulation. (Superseded in scope by ADR 0110 decision 8: a human quality verdict is required
    only when a preset becomes `defaultQuant`; opt-in safety is the machine checks' job.) Whether
    adaLN i8 helps on the shipped rig was left unverified (on/off A/B waits for demand — backlog).
14. **Listening verdict (2026-08-24): R3 fixes the muffling; shippable as opt-in.** Published on HF
    (commit `67e9584c`, fragmentation 29.4 MiB per range).

## Addendum (2026-08-25) — 0.5.0: pin updated, `w4` renamed `i8+dit4`

The 0.5.0 release pinned the renamed revision, so `w4` no longer needs `revision: "main"` (today
`IRODORI_SOURCES` in `packages/models/src/irodori/config.ts`, ADR
[0092](0092-distribution-repos-and-sources.md)). Names follow ADR 0074: `w4` → `i8+dit4`, `w8` →
`i8`, `w8a8` → `i8-a8`; the decisions above keep the names of their time.
