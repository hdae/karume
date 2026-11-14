# 0151: Irodori v4-Large — the T5Gemma 2 text backbone, the Gemma tokenizer and a Gemma-licensed distribution

- Status: proposed (2026-11-14 — the recipe exports the seven text-side graphs, the tokenizer asset and the full-loop
  goldens, writes the f16 / i8 / i4 series and assembles the distribution; the TS pipeline reads the tokenizer asset;
  the local mirror is built and the real-GPU e2e passes on it. The listening, the licence review and the publication
  are pending — see Pending; the default preset is `i8-a8` by the maintainer's ruling of 2026-11-14, and the licence
  wording is unreviewed)
- Date: 2026-11-14
- Related: [0084](0084-gemma-tokenizer-chat.md) (the Gemma tokenizer asset; its addendum of 2026-11-14 adds the third
  post_processor shape) / [0149](0149-irodori-rare-token-split.md) (the rare-token split table; its addendum of
  2026-11-14) / [0150](0150-irodori-meanflow.md) decision 1 and [0149](0149-irodori-rare-token-split.md) decision 5
  (containers taken as published, borrowed across repositories) / [0038](0038-manifest-v1.md) §7 and its addendum of
  2026-10-28 (cross-repository references) / [0109](0109-manifest-v5-container.md) decision 3 /
  [0010](0010-symbolic-constant-folding.md) (Tmax constants and `sym_prefix_slice`) /
  [0048](0048-irodori-host-port.md) (the Irodori host port and the static scheme of the text side) /
  [0050](0050-irodori-quant-series.md) (the quant series; decision 6: an i8 duration graph may move the length) /
  [0114](0114-irodori-dit-context-split.md) (`dit_context` / `dit`) /
  [0128](0128-default-preset-original-weights.md) and [0147](0147-numerics-and-kernel-adoption.md) decision 3 (the
  default preset per model, chosen by the maintainer) / [0075](0075-quant-presentation.md) (preset labels) /
  [0122](0122-umt5-upstream-and-compatible-encoders.md) decision 6 (the pre-publication licence check) /
  [container-v1](../container-v1.md) §2.3 (`provenance.license`) / [0139](0139-tests-on-distributions-and-test-assets.md)
  (where goldens live) / [0107](0107-model-input-error.md) (input-caused failures)

## Context

Upstream published [Aratako/Irodori-TTS-v4-Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large) (revision
`2e0c55428ce97268a507f1feeb2478f8d9148e8b`, licence tag `gemma`): a rectified-flow checkpoint (`rf_velocity`, 40 Euler
steps with CFG, like v4-small) with the tensor names and structure of the small models' DiT, speaker encoder, duration
predictor and projectors, at other dimensions — DiT width 2048, 24 blocks (48 condition K/V tensors), 32 heads; text,
caption and speaker width 1280; text up to 256 tokens, caption up to 512; vocabulary 262,144. The export and the TS
pipeline read every one of these from the checkpoint's config and the manifest.

What is new for the stack:

- **A second text-backbone family.** The text and caption encoder is T5Gemma 2's text encoder
  (`google/t5gemma-2-1b-1b`; hidden 1152, 26 layers: 22 sliding and 4 full), where the small models use ModernBERT. Its
  sliding layers mask a real bidirectional window — 256 positions to the left and 257 to the right of each query, which
  starts to cut at T = 257 — and its full layers use linear-scaled RoPE. The text side never reaches the window (at most
  256 tokens with the BOS); a caption of more than 256 tokens does. The token embedding overwrites the end-of-image row
  with a separately trained vector through a boolean index assignment.
- **A Gemma BPE tokenizer** (T5Gemma 2's, shipped in upstream's `tokenizer/`), where the small models use a Unigram
  one. Its post_processor prepends `<bos>` and appends nothing, a shape neither gemma4 nor EmbeddingGemma has.
- **A 3.3 B-parameter checkpoint**: the f32 containers total 13.16 GB, the f32 DiT alone 6.2 GB.
- **The Gemma Terms of Use.** Upstream tags the model `license: gemma`, because its text and caption encoder is derived
  from `google/t5gemma-2-1b-1b`; the small models are MIT.

## Decision

1. **The backbone graph runs at the real length with the window folded into a constant.** The `backbone` graph takes
   `input_ids [1,T]` only (T up to 512) and no mask, under the static scheme the small models' text side already uses
   (B = 1, the caller packs the sequence; `_static_scheme_evidence` measures the equivalence with the padded, masked call
   for this backbone too). The sliding layers' band is built from the model's own window function
   (`t5gemma2_masks` in `tools/export-recipes/irodori/export.py`) and folded into one `[1,1,512,512]` constant that the
   22 sliding layers slice to `[T,T]` (`sym_prefix_slice` — ADR 0010, the mechanism of ModernBERT's local band); the 4
   full layers take no mask. No new IR op and no runtime change. Three export-time checks keep the window honest, none
   of which the golden alone would give:

   - **The window reaches the output** (`_sliding_window_evidence`): at T 40 / 256 / 257 / 300 / 512 (real text),
     upstream's windowed call differs from full attention in the sliding layers wherever transformers' mask cuts
     something, so a graph without the window could not pass.
   - **The wrapper matches upstream on both sides of the window** (the same function): the export wrapper equals
     upstream's windowed call at atol 0 at every one of those lengths, below and beyond T = 257.
   - **The folded constant is transformers' own mask** (`_t5gemma2_window_form`): in the exported IR, layer by layer,
     the full layers have no mask and the sliding layers slice one constant whose leading T×T block is the mask
     transformers builds (0 / −inf; `upstream_sliding_mask`, through transformers' vmap path, independently of
     `t5gemma2_masks`) at those lengths and both ends of the range.

   The golden cases of the backbone (`T5GEMMA2_GOLDEN_CASES`) reach T 388 and 512 on the caption side, and the
   full-loop goldens add the case `caption-window` (a 459-token caption the window cuts); the full-loop run fails before
   writing unless some case is cut by transformers' mask, and records the names in `meta.json` (`windowCutCases`, a key
   written for T5Gemma2 only).

   Alternatives not taken: dropping the window (exact up to T = 256, wrong for captions beyond it), and a run-time
   mask input (the Irodori text side has no run-time mask — the "Irodori text side" entry of
   [limitations](../limitations.md); the folded constant costs one `[512,512]` table). The
   family is recognized by the embedded config's `model_type`, the test upstream makes, never by the directory name; the
   small models' containers and goldens are byte-identical.

2. **The tokenizer is chosen by the asset's format, behind one interface.** The recipe reads the kind from upstream's
   `tokenizer.json` (`model.type` — `tokenizer_kind` in `tools/export-recipes/irodori/tokenizer_ref.py`) and writes the
   BPE kind in the shared Gemma format (`karume-gemma-tokenizer/1`, compiled by
   `tools/export-recipes/_shared/gemma_tokenizer.py`, `spec.postProcessor: "bos"`), with its own encode / batch / caption
   goldens checked against upstream's tokenizer class (including a caption beyond 256 tokens and one cut at 512) and a
   TS parity fixture (`packages/models/tests/fixtures/irodori-text/v4-large-parity.json`; the small models'
   `parity.json` is unchanged). On the TS side, `parseIrodoriTextTokenizer`
   (`packages/models/src/irodori/text/asset.ts`) builds the tokenizer the asset declares: no `format` key is the
   Unigram asset of the small models, `karume-gemma-tokenizer/1` is the Gemma asset (`IrodoriGemmaTokenizer`), and any
   other format fails. Both sit behind `IrodoriTextTokenizer` (`text-tokenizer.ts`: BOS id, pad id,
   `hasRareTokenSplit`, `encode`), and BOS, truncation and right padding are one shared function (`encodePaddedIds`),
   as upstream wraps both tokenizers in one `PretrainedTextTokenizer`: the body is encoded without special tokens, so
   the asset's post_processor never runs on this path. The pad id is the added token `<pad>` (the Gemma format has no
   `padId` field; the recipe checks it against the checkpoint config and upstream's wrapper).

   **The Gemma kind has no rare-token split**: upstream applies the split only to the Unigram tokenizer of
   `sbintuitions/modernbert-ja-310m`, so the asset carries no table, `hasRareTokenSplit` is `false`, an omitted
   `splitRareTokens` is off and an explicit `true` is the `ModelInputError` of ADR 0149 decision 3.

   Alternative not taken: a `pipelineConfig` key or the model id naming the tokenizer kind — a second declaration that
   could disagree with the asset it describes.

3. **The preset list is a per-model fact; v4-Large has no `f32` preset.** The recipe's list
   (`IRODORI_MODEL_PRESETS` in `tools/export-recipes/irodori/distribution.py`) gives v4-Large `f16` / `i8` / `i8-a8` /
   `i8+dit4`; the build reads and lays out only the series those presets select (`irodori_dtype_roles`: every graph in
   f16 and i8, the DiT pair in i4). The f32 series stays the recipe's reference series — the goldens are baked from it —
   and is never published: the f32 DiT alone is 6.2 GB. The family's labels "Half size" / "Quarter size" are sizes
   relative to `f32`, so this model has its own (`IRODORI_MODEL_SEAT_LABELS`: "Largest (f16)", "Smaller (int8)"), and
   its preset descriptions state storage facts only (`IRODORI_MODEL_SEAT_DESCRIPTIONS`), since nothing has been
   listened to or timed for it. **The default is `i8-a8`** (`IRODORI_MODEL_DEFAULT_QUANTS`), by the maintainer's
   ruling of 2026-11-14 on the listening material and the measurement in Consequences (ADR 0147 decision 3): the same
   download as `i8`, and the fastest preset measured (about 1.5–1.7 times `i8` on the RTX 3080 Ti). The default was
   `i8`, provisionally, until that ruling. Alternative not taken: the family's list with `f32` (a
   13.16 GB download that only repeats the reference series).

4. **A separate recipe Pipeline for the licence; the containers carry the provenance licence `gemma`.** v4-Large is
   built with `dist.py --pipeline irodori-large` (`LARGE_PIPELINE`), repository `karume-irodori-v4-large`, model id
   `v4-large`. The root files of a recipe Pipeline are one fixed set (`root_files`), and this line's are another set:
   `LICENSE.md` holds the Gemma Terms of Use and the Gemma Prohibited Use Policy verbatim (copies of upstream's files in
   `tools/export-recipes/irodori/licenses/`, digests pinned by tests), then the codec's MIT and Apache 2.0 texts;
   `NOTICE.md` opens with upstream's `NOTICE` line and lists the changes. The two Pipelines refuse each other's models.
   The plan, references and card template are the family's, and the manifest's pipeline is `irodori/1` as for the small
   models, so the TS pipeline is the same. Every container the model exports carries `license: gemma` in its provenance
   (`provenance_of`, picked by the backbone family; the small models keep `mit`), and the build fails unless each does
   (`assert_irodori_container_licenses`): a component can be taken out of a distribution on its own, and then the
   container alone states its terms. Alternative not taken: one Pipeline with per-model root files, which the core's
   Pipeline does not have.

5. **The codec pair is borrowed from the published v4-small; the tokenizer is the model's own.** The codec containers
   (f16 and i8) are the published `karume-irodori-v4-small` containers, declared in `IRODORI_PUBLISHED` and referenced
   across repositories; the build checks their bytes against the part-0 pins as for every published container (ADR 0149
   decision 5), and `--resolve-refs hf` checks them in v4-small's main. The seven text-side graphs come from v4-Large's
   own series, and the tokenizer asset is its own Gemma-format asset, read from the plain series
   (`outputs/series/irodori-v4-large/tokenizer/tokenizer.json`). Alternatives not taken: a fresh codec export in this
   repository (a duplicate of bytes already on the Hub, and a fresh export differs from the published containers in
   part 0 — ADR 0150 decision 1), and a tokenizer reference (v4-small's Unigram asset is another tokenizer).

6. **The f16 series keeps the backbone's non-persistent buffers at f32.** The T5Gemma 2 backbone has five buffers that
   are not in the checkpoint and that transformers computes in f32 at init: `embed_scale` and four RoPE `inv_freq`
   (`t5gemma2_f32_buffers`). Rounded to f16, the full layers' linear-scaled frequencies fall into f16's subnormal range
   and lose up to 15%, and the folded RoPE tables stop being upstream's values, an error upstream does not have. They are
   kept at f32 through the rounding. Measured on the backbone alone against f32 (2026-11-14): 61.3 → 68.6 dB on
   `caption-window` (61.2 → 68.6 dB on `caption-long`), almost all of it from the frequencies. **ModernBERT's buffers in
   the published small models are rounded the old way and are not changed here**: picking them too would move the
   published bytes of v4-small and v4.1-small. Alternative not taken: rounding every buffer as before (the measured
   loss above).

7. **The end-of-image row selection is a `where` with an f32 comparison.** Patch 5 of
   `tools/export-recipes/irodori/patch.py` (`_where_eoi_scaled_embedding_forward`) writes
   `T5Gemma2TextScaledWordEmbedding.forward`'s boolean `index_put` as
   `where(ids == eoi, eoi_embedding, scaled rows)`, the equality as `(id >= eoi) & (id <= eoi)` on the ids cast to f32,
   because the IR's scalar comparisons take f32 and its vocabulary has no `eq`. `where` copies, so the patched forward is
   bit-identical to upstream's. The cast is exact only for ids up to 2^24 (`F32_EXACT_INTEGER_MAX`), so a larger
   vocabulary fails at export; 262,144 = 2^18. `_eoi_evidence` checks that a golden case (`caption-eoi`) runs the
   selection and that it is not inert. Alternative not taken: an integer `eq` op in the IR, a vocabulary addition for
   one row selection that the existing ops express exactly.

## Consequences

- No byte of the small models moves: their containers, io goldens, full-loop goldens, tokenizer asset and parity fixture
  are byte-identical, the three small mirrors rebuild byte-identical, and the gemma4 tokenizer assets are byte-identical
  with the `bos` shape added.
- The TS side needed no change for the dimensions (the pipeline reads them from the manifest and the graphs); the
  changes are the tokenizer selection (decision 2) and the `bos` shape of the shared Gemma reader.
- Sizes (container totals of the series, 2026-11-14): f16 6.58 GB, i8 3.30 GB (backbone 1.01 GB, DiT 1.55 GB); the f32
  series, 13.16 GB, is reference only. The token embedding `[262144,1152]` is one initializer (f16 604 MB, i8 302 MB),
  the largest binding; the build derives the manifest's `requiredLimits` from it, and the value exists once the mirror
  is built.
- Measured on the recipe's full loop (2026-11-14): f16 against the f32 golden latent at 43.9, 59.5 and 56.0 dB on the
  three cases; i8 changes the predicted length of two cases by one frame, through the i8 duration graph (the known axis
  of ADR 0050 decision 6).
- The i4 calibration corpus is checked to be disjoint from the model's own evaluation table (`calib_cases.py`), which
  for T5Gemma2 includes `caption-window`.
- The i4 series (2026-11-14): GPTQ over 12 calibration cases × 40 steps took 89 minutes on the CPU; 336 linears inside
  the blocks are i4 (192 in `dit`, 144 in `dit_context`) and 293 stay i8. Download sizes of the presets (the mirror's
  card): `f16` 6.31 GiB, `i8` and `i8-a8` 3.17 GiB, `i8+dit4` 2.56 GiB.
- Real-GPU e2e on the mirror (2026-11-14, `node-nvidia-geforce-rtx-3080-ti`; `e2e_irodori_v4_large_test.ts`,
  `e2e_irodori_v4_large_wav_test.ts`, `e2e_irodori_ab_test.ts`): the seven f16 graphs differ from their torch goldens by
  4.8e-7 to 1.2e-4 at worst; the full-loop latent differs from each series' own golden by 8.2e-4 (f16), 3.2e-4 (i8) and
  1.2e-3 (i8+dit4) at \|z\| max about 5, with equal frame and forward counts in all nine cases; `i8-a8` differs from
  `i8` on the step-1 latent by relRMS 2.76e-3 and 4.28e-3, and its dispatch census equals the census table (485 / 144
  i8a8 GEMMs per run of `dit` / `dit_context`, no f32 linear). WAV sha256 rows are recorded for three cases on `f16`,
  one of them passing a 459-token caption through the sliding window.
- Node measurement of the presets (2026-11-14, RTX 3080 Ti, Node + Dawn; no reference, no caption, 60 forwards; the
  median wall time of one `generate` call over seeds 42–45, after one warm-up generation; the VRAM figure is the
  whole-GPU used memory sampled every 0.5 s over the run, 33 MiB of it idle):

  | Preset    | 3.7 s of audio | 8.8 s  | 19.5 s | VRAM peak |
  | --------- | -------------- | ------ | ------ | --------- |
  | `f16`     | 10.8 s         | 14.3 s | 23.4 s | 4,001 MiB |
  | `i8`      | 8.9 s          | 12.1 s | 20.9 s | 2,944 MiB |
  | `i8-a8`   | 6.1 s          | 8.0 s  | 12.0 s | 3,034 MiB |
  | `i8+dit4` | 8.4 s          | 11.5 s | 20.5 s | 2,138 MiB |

  Every preset predicts the same frame counts on these three texts. The final latents against `f16`'s (same text and
  seed, four seeds pooled): `i8` 17.8–18.5 dB, `i8-a8` 14.2–22.7 dB, `i8+dit4` 3.5–12.2 dB, at equal latent energy.
  That figure is the divergence of 40 guided steps, not an audio-quality measure; the listening decides.
- The `dit` io goldens store the condition K/V once (2026-11-14): the 48 inputs `[1,1519,32,64]` are byte-identical in
  the seven cases and equal to the outputs of the one `dit_context` golden, so each `dit` io leaves them out and names
  that golden and the output each comes from in its `__metadata__` (`inputs_from`). The recipe fails the export unless
  every omitted input is byte-identical to that golden as stored, and the e2e check feeds them from it, with the same
  measured differences. The per-graph io goldens of the f16 series go from 4.83 GB to 0.65 GB (`dit` 4.18 GB →
  0.30 MB, `dit_context` 0.61 GB). The small models keep the copy in their `dit` io: their goldens are in the
  test-asset dataset in that form.
- The distribution's legal texts follow the recipe's reading of Gemma Terms of Use §3.1, recorded as Unverified in
  `tools/export-recipes/irodori/THIRD_PARTY_NOTICES.md`; the maintainer reviews the wording before any upload.

## Pending

- The Playground page in a browser (built, not run).
- The maintainer's review of the licence wording, then the publication: the HF upload built with `--resolve-refs hf`
  against v4-small's main, the goldens in the test-asset dataset with the pin moved, the
  `irodori.SOURCES["irodori-v4-large"]` entry, and the Playground mount reading its repository name from that entry
  (`docs/release-runbook.md` §0).
