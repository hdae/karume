# Third-party notices — Wan2.1

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/wan/`) is repo-only: it is never published to PyPI and none of
it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to the recipe
that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers),
  pinned in `sources.py` (`SOURCES`). Only `transformer`, `vae` and `scheduler` are converted and
  redistributed; the umT5-XXL `text_encoder` is run locally to produce the precomputed text
  embeddings (`text_embeds.py`) and is not redistributed.
- **Model implementation** — the `diffusers` package (`WanTransformer3DModel`, `AutoencoderKLWan`,
  `WanPipeline`, `UniPCMultistepScheduler`), pinned `diffusers==0.39.0`. `dit_patch.py`,
  `vae_patch.py`, `vae_tiling.py` and `few_step_ref.py` carry functions adapted from it, each under
  a "Third-party code notice" comment that names the upstream function and copies its copyright
  line.
- **Text encoder and tokenizer** — `UMT5EncoderModel` / `AutoTokenizer` from `transformers`
  (pinned `transformers==5.14.1`), imported only.
- **Prompt normalization** — `ftfy` (pinned `ftfy==6.3.1`), imported only, through diffusers'
  `prompt_clean`.
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

| Item                     | Value                                                                                                                                                                                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>                                                                                                                                                                                                                                   |
| Revision used            | `0fad780a534b6463e45facd96134c9f345acfa5b` (`sources.py`; the containers' `provenance.upstreamRevision` carries the same value and `distribution.py` checks it)                                                                                                                             |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form (f16-rounded weights; the transformer additionally as per-output-channel int8 weights — ADR 0120). The text encoder is not re-distributed; its outputs for the fixed prompts are.                                              |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                          |
| Weights license          | `apache-2.0` — `sources.py` records it from the model card front matter (HF API, 2026-10-02), and `export_vae.py` reads it from the downloaded snapshot's `README.md`. Unverified by a human review against the revision used.                                                              |
| Attribution requirements | Apache 2.0 §4: the distribution bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) statement of changes — container re-expression, f16 rounding, the int8 transformer, the transformer and VAE rewrites, the precomputed text embeddings). |

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

| Item                     | Value                                                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers> / <https://github.com/rspeer/python-ftfy>                                                                   |
| Revision used            | `transformers==5.14.1` / `ftfy==6.3.1` (pinned in `pyproject.toml`, group `wan`)                                                                          |
| Form of copy             | Import-time dependencies; nothing is copied.                                                                                                              |
| Code license             | Apache-2.0 for both — the installed wheels' metadata (checked 2026-10-03). Unverified against the repositories' `LICENSE` files.                          |
| Weights license          | n/a                                                                                                                                                       |
| Attribution requirements | None for this directory (no copy). Nothing from either package enters the published distribution; the normalized prompt strings are data, not their code. |

### Fixed prompt texts

| Item                     | Value                                                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream repository      | <https://github.com/Wan-Video/Wan2.1> (README, `wan/configs/shared_config.py`) and <https://github.com/huggingface/diffusers> (documentation, `pipeline_wan.py`)         |
| Revision used            | Commit-pinned URLs in `prompts.py` (`9737cba9c1c3c4d04b33fcad41c111989865d315` for Wan2.1, `a3608b512ed7248499a44c61d954965ed9bdae4d` = the `v0.39.0` tag for diffusers) |
| Form of copy             | Four short texts quoted verbatim; they are also stored in the text-embedding asset's metadata and printed in the distribution's model card.                              |
| Code license             | Apache-2.0 for both repositories (Wan2.1: README license section; diffusers: as above). Unverified by a human review.                                                    |
| Weights license          | n/a                                                                                                                                                                      |
| Attribution requirements | The model card links each text to its commit-pinned source. Whether short example prompts need more than that is part of the release review.                             |
