# Third-party notices — Wan2.1 / Wan2.2 / umT5-XXL

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/wan/`) is repo-only: it is never published to PyPI and none of
it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to the recipe
that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer. A field that names a review date records that the human review was
completed on that date.

## Upstream sources

- **Weights** — [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers),
  pinned in `sources.py` (`SOURCES`). `transformer` and `vae` (with the `scheduler` config) are
  converted into `karume-wan2.1`, which is built locally and not published. The umT5-XXL
  `text_encoder` is run locally to produce the precomputed text embeddings (`text_embeds.py`). The
  `tokenizer` is converted into one JSON table together with the prompt-cleaning tables
  (`umt5_tokenizer.py`). `karume-wan2.2` redistributes both the converted tokenizer and the
  precomputed embeddings.
- **Weights** — [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers),
  pinned in `sources.py` (`SOURCES`). `transformer` and `vae` (with the `scheduler` config) are
  converted and redistributed in `karume-wan2.2` (`distribution.py`, `WAN22`). Its `text_encoder`
  and `tokenizer` folders are not fetched.
- **umT5-XXL encoder weights and tokenizer** — [google/umt5-xxl](https://huggingface.co/google/umt5-xxl),
  pinned in `sources.py` (`UMT5_SOURCES`, with the SHA-256 of every shard and tokenizer file it
  reads). The encoder is converted to int8 (`umt5_export.py`) and the tokenizer into one JSON table
  without prompt cleaning (`umt5_xxl_tokenizer.py`); both are redistributed in the separate
  repository `karume-umt5-xxl` (`umt5_distribution.py`), whose encoder `karume-wan2.2` references.
- **Model implementation** — the `diffusers` package (`WanTransformer3DModel`, `AutoencoderKLWan`,
  `WanPipeline`, `UniPCMultistepScheduler`), pinned `diffusers==0.39.0`. `dit_patch.py`,
  `vae_patch.py`, `vae_tiling.py` and `few_step_ref.py` carry functions adapted from it, each under
  a "Third-party code notice" comment that names the upstream function and copies its copyright
  line.
- **Text encoder and tokenizer** — `UMT5EncoderModel` / `AutoTokenizer` from `transformers`
  (pinned `transformers==5.14.1`), imported only.
- **Prompt normalization** — `ftfy` (pinned `ftfy==6.3.1`), imported through diffusers'
  `prompt_clean`. `prompt_clean.py` evaluates ftfy's character tables and translates its
  `BADNESS_RE` pattern into JavaScript; the results are part of the tokenizer asset that
  `karume-wan2.1` ships.
- **Fixed prompts** — `prompts.py` quotes four prompt texts verbatim from the official Wan2.1
  repository (the t2v-1.3B example in `README.md` and `sample_neg_prompt` in
  `wan/configs/shared_config.py`) and from the Diffusers documentation and source (the Wan
  text-to-video example and the `WanPipeline.__call__` docstring example). Each entry keeps its
  commit-pinned source URL.
- **Technical report** — Wan: <https://arxiv.org/abs/2503.20314>.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### Wan-AI/Wan2.1-T2V-1.3B-Diffusers

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>                                                                                                                                                                                                                                                                                                                                                                                      |
| Revision used            | `0fad780a534b6463e45facd96134c9f345acfa5b` (`sources.py`; the containers' `provenance.upstreamRevision` carries the same value and `distribution.py` checks it)                                                                                                                                                                                                                                                                                |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form (f16-rounded weights; the transformer additionally as per-output-channel int8 weights — ADR 0120). The `text_encoder` folder itself is not re-distributed (`karume-umt5-xxl` converts the bit-identical encoder of `google/umt5-xxl` instead — see the next block); its outputs for the fixed prompts are re-distributed, and so is the tokenizer, converted into one JSON table. |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Weights license          | `apache-2.0` — `sources.py` records it from the model card front matter (HF API, 2026-10-02), and `export_vae.py` reads it from the downloaded snapshot's `README.md`; the repository has no LICENSE file. Reviewed by a human against the revision used on 2026-10-15.                                                                                                                                                                        |
| Attribution requirements | Apache 2.0 §4: the distribution bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) statement of changes — container re-expression, f16 rounding, the int8 transformer, the transformer and VAE rewrites, the text encoder referenced from `karume-umt5-xxl`, the precomputed text embeddings, the converted tokenizer).                                                                       |
| Published in             | `karume-wan2.1` is not published. `karume-wan2.2` redistributes two assets made from this checkpoint: the converted tokenizer (its files are identical to the `tokenizer` folder of Wan2.2 TI2V 5B — next block) with the prompt-cleaning tables, and the precomputed text embeddings computed with this checkpoint's umT5-XXL in bfloat16. Its `NOTICE.md` lists both.                                                                        |

### Wan-AI/Wan2.2-TI2V-5B-Diffusers

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Revision used            | `b8fff7315c768468a5333511427288870b2e9635` (`sources.py`; the containers' `provenance.upstreamRevision` carries the same value and `distribution.py` checks it)                                                                                                                                                                                                                                                                                                                                                                                     |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form in `karume-wan2.2`: the transformer only as per-output-channel int8 weights (its other parameters at the source float32 values; no float16 or float32 copy), the VAE decoder and the part of the VAE encoder that encodes one conditioning image rounded to float16. The `text_encoder` and `tokenizer` folders are not used: the text encoder is referenced from `karume-umt5-xxl`, and the tokenizer and the precomputed text embeddings come from Wan2.1 T2V 1.3B (previous block). |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Weights license          | `apache-2.0` — `sources.py` records it from the model card front matter (HF API), and the export reads it from the downloaded snapshot's `README.md`; the repository has no LICENSE file. Reviewed by a human against the revision used on 2026-10-15.                                                                                                                                                                                                                                                                                              |
| Attribution requirements | Apache 2.0 §4: `karume-wan2.2` bundles `LICENSE.md` (verbatim) and `NOTICE.md` (§4(b) — container re-expression, the int8 transformer, the f16 VAE decoder and image-to-video VAE encoder, the transformer and VAE rewrites, the text encoder referenced from `karume-umt5-xxl`, the precomputed text embeddings and the converted tokenizer of Wan2.1 T2V 1.3B).                                                                                                                                                                                   |

### google/umt5-xxl (the umT5-XXL encoder and tokenizer) — `karume-umt5-xxl`

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/umt5-xxl> (`UMT5ForConditionalGeneration`, float32, pickle `.bin` in 6 shards). Only the encoder is read (`shared.weight` and `encoder.*`, shards 1–3); the decoder and `lm_head` are not. Its encoder weights are bit-identical to the float32 `text_encoder` folder of Wan2.1 T2V 1.3B Diffusers above (checked 2026-10-04).                                                                                                                                                          |
| Revision used            | `66cb9e7e85526fe440a945569e42c72fb6cbc0ad` (`sources.py` `UMT5_SOURCES`; the container's `provenance.upstreamRevision` and the tokenizer asset's `source.revision` carry the same value and `umt5_distribution.py` checks both). Each shard's SHA-256 is checked against the table before it is unpickled, and so is each tokenizer file's (`tokenizer.json`, `tokenizer_config.json`) before it is read.                                                                                                              |
| Form of copy             | Re-distributed in converted storage form in its own repository: linear and vocabulary-embedding weights as per-output-channel (per-row) int8, RMSNorm weights and relative-position tables at the source float32 values; the tokenizer as one JSON table (vocabulary and scores, added tokens, the whitespace set, the maximum length — format `karume-umt5-tokenizer/1`, no prompt-cleaning tables). `karume-wan2.2` references the encoder by repository, commit, size and SHA-256 instead of storing a second copy. |
| Code license             | n/a (weights and tokenizer data only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Weights license          | `apache-2.0` — `sources.py` records it from the card data and the model card front matter of `google/umt5-xxl` (HF API, 2026-10-04); the repository has no LICENSE or NOTICE file. Reviewed by a human against the revision used on 2026-10-15.                                                                                                                                                                                                                                                                        |
| Attribution requirements | Apache 2.0 §4: `karume-umt5-xxl` bundles `LICENSE.md` (verbatim) and `NOTICE.md` (§4(b) — the encoder only, container re-expression, int8 weights, `gelu_new` replaced by the equivalent `GELU(approximate="tanh")`, the valid-token graph with the bucket indices as an input, the tokenizer converted into one JSON table and the inputs the host rejects).                                                                                                                                                          |

### diffusers (model implementation)

| Item                     | Value                                                                                                                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/diffusers>                                                                                                                                                                                                |
| Revision used            | `diffusers==0.39.0` (pinned in `pyproject.toml`, group `wan`)                                                                                                                                                                             |
| Form of copy             | Imported classes, plus adapted functions in `dit_patch.py`, `vae_patch.py`, `vae_tiling.py` (`blend_v` / `blend_h` verbatim) and `few_step_ref.py`, each marked with a "Third-party code notice".                                         |
| Code license             | Apache-2.0 — the installed wheel's metadata (`diffusers 0.39.0`, checked 2026-10-03); the adapted files carry "Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved." Unverified against the repository's `LICENSE`. |
| Weights license          | n/a                                                                                                                                                                                                                                       |
| Attribution requirements | Apache 2.0 §4(a)/(b) attaches to the adapted functions, which live only in this repo-only directory and are attributed in place. Nothing from `diffusers` code enters the published distribution.                                         |

### transformers / ftfy (imported only)

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers> / <https://github.com/rspeer/python-ftfy>                                                                                                                                                                                                                                                                                                                             |
| Revision used            | `transformers==5.14.1` / `ftfy==6.3.1` (pinned in `pyproject.toml`, group `wan`)                                                                                                                                                                                                                                                                                                                                    |
| Form of copy             | Import-time dependencies. No code is copied; the Wan tokenizer asset carries ftfy-derived data — character tables evaluated from ftfy 6.3.1, the `regex` package and the Unicode 16.0.0 character database, and a JavaScript translation of ftfy's `BADNESS_RE` pattern (`prompt_clean.py`).                                                                                                                        |
| Code license             | Apache-2.0 for both — the installed wheels' metadata (checked 2026-10-03). Unverified against the repositories' `LICENSE` files.                                                                                                                                                                                                                                                                                    |
| Weights license          | n/a                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Attribution requirements | None for this directory (no code copy). The ftfy-, `regex`- and Unicode-derived tables and the translated pattern in the Wan tokenizer asset (redistributed in `karume-wan2.2`) are listed in its `NOTICE.md`; the human review of these derived tables was completed on 2026-10-15. Nothing from `transformers` enters the distribution (the `karume-umt5-xxl` tokenizer asset carries no prompt-cleaning tables). |

### Fixed prompt texts

| Item                     | Value                                                                                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream repository      | <https://github.com/Wan-Video/Wan2.1> (README, `wan/configs/shared_config.py`) and <https://github.com/huggingface/diffusers> (documentation, `pipeline_wan.py`)               |
| Revision used            | Commit-pinned URLs in `prompts.py` (`9737cba9c1c3c4d04b33fcad41c111989865d315` for Wan2.1, `a3608b512ed7248499a44c61d954965ed9bdae4d` = the `v0.39.0` tag for diffusers)       |
| Form of copy             | Four short texts quoted verbatim; they are also stored in the text-embedding asset's metadata and printed in the distribution's model card.                                    |
| Code license             | Apache-2.0 for both repositories (Wan2.1: README license section; diffusers: as above). The human review of the quoted texts under these licenses was completed on 2026-10-15. |
| Weights license          | n/a                                                                                                                                                                            |
| Attribution requirements | The model card links each text to its commit-pinned source. The human review of the quoted texts was completed on 2026-10-15.                                                  |
