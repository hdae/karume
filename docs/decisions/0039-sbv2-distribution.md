# 0039: SBV2 distribution form (`sbv2/1`) and the future multi-speaker design

- Status: accepted (2026-08-07 — the maintainer accepted the listening check: "up to w8a8 the voices
  sound almost the same; f16 shows no degradation at all"); partly superseded by
  [0041](0041-manifest-v2.md) (manifest vocabulary and multi-model packaging: decisions 2 and 7,
  2026-08-08), [0069](0069-packed-w4-storage.md) (2026-08-17) and
  [0128](0128-default-preset-original-weights.md) decision 4 (2026-10-17) (decision 2),
  [0074](0074-quant-seat-naming.md) (2026-08-25) and 0128 decision 1 (2026-10-17) (decision 5), and
  [0079](0079-sbv2-two-layer-input.md) (decision 6, 2026-08-25). Decisions 1, 3 and 4 are in force.
- Date: 2026-08-07
- Related: ADR [0038](0038-manifest-v1.md) (manifest v1 — this ADR settled its SBV2 example) /
  [0013](0013-sbv2-chain-export.md) (emit targets of the chain) /
  [0026](0026-w8a8-deberta-deployment.md) (why DeBERTa i8 was accepted) /
  [0027](0027-sbv2-f16-series.md) and [0029](0029-sbv2-i8-series-and-quant-quality.md) (f16 / i8
  series) / [0037](0037-karume-monorepo.md) (parent decision on the distribution form) /
  [0008](0008-public-api.md) (thin public surface)

## Context

The TypeScript SBV2 implementation moved into this repository and the real-weight end-to-end tests
passed on a real GPU. ADR 0038's SBV2 manifest example said it was a sample "to be settled from the
measured series at export"; this ADR settles it.

The weights were `FN/FN4/` of Hugging Face `rufflet17/voice_models` (Style-Bert-VITS2 JP-Extra,
`2.6.1-JP-Extra`, one speaker, four styles, 44.1 kHz). Its `state_dict` matches the checkpoint used in
the earlier experiment tensor for tensor (1,165 tensors, same shapes), so the `style-bert-vits2==2.5.0`
pin stayed. The style set differs (4 against 7); that difference motivates decision 3.

Since then the published SBV2 distribution is `karume-sbv2-jvnv` (licence `cc-by-sa-4.0` with
`LICENSE.md` / `NOTICE.md`, ADR 0092 decision 7). `karume-sbv2-fn` is not published on Hugging Face
and has no permanent local mirror (rulings of 2026-08-20 and 2026-08-30; `docs/backlog.md`, parked
item "karume-sbv2-fn"; `docs/assets-layout.md`).

## Decision

### 1. Only the components needed at run time (3 graphs + 4 assets)

ADR 0038's example listed six emit targets (`dp` / `front` / `flow` / `dec` / `voice` plus
`text_encoder`), but only three are needed at run time: `voice` fuses flow and dec, and `front`
contains enc_p, dp and sdp (ADR 0013). `dp` / `flow` / `dec` serve only golden verification and are
not distributed; shipping them would waste flow 155 MB and dec 59 MB.

The components are `text_encoder`, `front`, `voice` (graphs) and `tokenizer`, `symbols`,
`style_vectors`, `speaker_embeddings` (assets). The sizes recorded on 2026-08-07 (FN4, 504 MiB in
total) predate the layer trim of ADR [0045](0045-deberta-layer-trim.md) (2026-08-11: the text encoder
drops its last two layers, WAV sha256 unchanged) and the `krm` container.

### 2. `text_encoder` is i8 only (no variants)

(Superseded by ADR 0041 decision 3 (2026-08-08), ADR 0069 (2026-08-17) and ADR 0128 decision 4
(2026-10-17): weights always carry a dtype key and the `{file}` form is gone; the `i8+bert4` and
`i4` presets carry the text encoder in i4; an original-weights preset is best effort and is not
ruled out by size.) The original rule: `export_deberta.py` had only f32 and i8, the f32 encoder
(1.32 GB) was not distributed, and i8 was accepted by ADR 0026 with a listening check (hidden[-3]
SNR 20.7 to 23.3 dB), so there was no reason for f16. The `f16` preset was therefore asymmetric
(generator nets f16, text encoder i8), which is why ADR 0038 made `weights` a per-component record.
A `{file}`-form component MUST NOT appear in a preset's `weights`.

### 3. `styles` / `speakers` are derived from the assets (MUST — never hard-code the tables)

`pipelineConfig.styles` / `speakers` copy `data.style2id` / `data.spk2id` from `config.json`. The
default knobs come from `style_bert_vits2.constants` (`DEFAULT_STYLE`, `DEFAULT_SDP_RATIO`, ...)
(`tools/export-recipes/sbv2/distribution.py`).

**The tables MUST NOT be hard-coded.** When the checkpoint changes, the style names and their order
change too (FN4 has 4 styles, the earlier checkpoint 7). A copied table keeps the shapes valid and
silently produces the voice of another style. Assembly also checks `defaults.style ∈ keys(styles)`
and `defaults.speaker ∈ keys(speakers)`; otherwise a distribution with a non-existent default fails
only at start-up.

### 4. Styles and speakers: ship the tables, pick the row at run time

`style_vectors` (`[styles, 256]`) and `speaker_embeddings` (`[speakers, gin_channels]`) are shipped as
tables, and the pipeline resolves name → row → vector. Styles are mixed as
`mean + (picked - mean) * weight`, where **`mean` is row 0** (the same formula as
`TTSModel.__get_style_vector`; `packages/models/src/sbv2/style.ts`).

The speaker embedding was missing from the first distribution: `front` and `voice` both take
`g[1,512,1]` as a graph input, but the demo path passed a `g` baked into an asset, so the gap
appeared only when the distribution was read through hub. Baking one selected speaker would leave
`pipelineConfig.speakers` mapping names to rows that cannot be selected, so speakers ship as a table
like `style_vectors`.

Checks (assembly stops with `DistError`):

- rows of `style_vectors` == `data.num_styles` == `len(data.style2id)`
- rows of `speaker_embeddings` == `data.n_speakers` == `len(data.spk2id)`, columns ==
  `model.gin_channels`
- the values of `styles` / `speakers` are a permutation of `0..rows-1` (also checked when
  `pipelineConfig` is parsed, `packages/models/src/sbv2/config.ts`)

**An ID is the row number, so a mismatch loads and runs and only produces another style's or
speaker's voice.** Matching the row counts is the only way to detect it, hence these three checks.

### 5. Presets `f16` / `w8` / `w8a8`, default `w8`

(Superseded by ADR 0074 (preset names, 0.5.0, 2026-08-25) and ADR 0128 decision 1 (how the default
preset is chosen, 2026-10-17); the current default is `i8+bert4`, `SBV2_DEFAULT_QUANT` in
`tools/export-recipes/sbv2/distribution.py`.) The original rule: only `w8a8` set
`session.linearCompute = "i8a8"`. Conv1d dominates SBV2 and linear is close to zero GFLOP (ADR 0025
decision 5), so `w8a8` was offered as a choice, not for speed. `w8` was the default because it was the
smallest download (400 MB) and listening allowed it.

### 6. The Japanese dictionary is not part of the distribution (provisional)

(Superseded by ADR 0079 on 2026-08-25: the yomi dependency and the dictionary fetch were removed from
models together with the `dictionary` injection option; the caller analyses the text with yomi.) The
original rule: the `@hdae/yomi` dictionary is tied to the yomi version, not to the model, so mixing it
into the distribution would force re-uploads on yomi updates and duplicate it per speaker. As a test
version, `packages/models` called `fetchDictionaryBytes` directly — the only path where models reached
the network without hub — with a `Sbv2PipelineOptions.dictionary` injection option. Declaring
`@hdae/yomi` in models also installed yomi for users who wanted only Anima.

### 7. No multi-speaker support now (options recorded)

(Superseded by ADR 0041 decisions 1 and 2 on 2026-08-08: one root manifest holds `models` plus
`defaultModel`, and the sub-manifest option was not adopted; `karume-sbv2-jvnv` holds F1 / F2 / M1 /
M2 with `shared/`.) The original analysis: speakers are a separate axis from the dtype labels, and
putting them into labels would make presets a speakers × dtypes product that hits the manifest v1
limits (`presets ≤ 32`, `components ≤ 64`) at ten speakers. Options were (a) sub-manifests inside the
repository with a shared `text_encoder` (63% of the distribution) and (b) expanding the speaker table,
which still needs per-speaker `front` / `voice` weights.

## Consequences

- Listening and the measured numbers disagree: `w8a8` has LSD 4.81 dB against f32, yet the maintainer
  could not tell it apart in normal playback. Numbers alone cannot choose a distribution preset.

## Addendum (2026-08-07) — verification and measurements

- The WAV from the distribution with `w8` matched the step-1 path that reads `outputs/series/` bit for
  bit (FN4); torch parity on f32 matched `w_ceil` exactly with waveform maxAbs 5.16e-5. The current
  WAV regression check is the per-environment row of jvnv F1 (`packages/e2e/tests/e2e_sbv2_wav_test.ts`).
- Quantization measured with torch simulation: SNR against f32 was f16 35.87 dB, w8 9.01 dB, w8a8
  7.83 dB; the degradation comes mainly from `front` (as in ADR 0029).
- GPU timing on 2026-08-10 refined "conv1d 86–90%": conv family 86.5% overall, 96.7% within
  front+voice; no effect on decision 5 ([research](../research/2026-08-10-op-timing-stats.md) §3).

## Addendum (2026-08-20) — default preset changes

- The quant table gained `w8-bert4` and `w4` (ADR 0069); the default moved to `w4` and, on the same
  day at the Hugging Face publishing check, to `w8-bert4` for quality
  ([research](../research/2026-08-20-gptq-awq-calibrated-rounding.md) §7). Renamed in 0.5.0 by ADR 0074
  (`w8-bert4` → `i8+bert4`).

## Addendum (2026-08-07 to 2026-09-24) — open items closed

- Licence: the open question about ShareAlike on the FN4 distribution is moot; the published jvnv
  distribution is `cc-by-sa-4.0` (ADR 0092 decision 7) and FN is not published (Context).
- The duplicated `defaults` in `symbols.json` and its consistency check were removed on 2026-09-24
  (review W-RC5-1); `pipelineConfig.defaults` is the only source (`sbv2_knob_defaults` in
  `tools/export-recipes/sbv2/distribution.py`; `defaults` is optional in
  `packages/models/src/sbv2/text/symbols.ts`).
- The DeBERTa end-to-end test was ported (2026-08-16); the export is launched with `-m` from
  `tools/export-recipes/deberta/export.py` (ADR 0065), and having no `karume` CLI entry is by design.
