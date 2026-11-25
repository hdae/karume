# Third-party notices — EmbeddingGemma 2

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/embeddinggemma2/`) is repo-only: it is never published to PyPI
and none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights, configs and tokenizer** — `google/embeddinggemma-2` on Hugging Face
  (<https://huggingface.co/google/embeddinggemma-2>), fetched at one commit into
  `inputs/embeddinggemma2/embeddinggemma-2/`; `sources.py` pins the commit and the SHA-256 of every
  file the scripts read. Of `model.safetensors` the text tower's tensors (`language_model.*`) and
  the vision tower's with its projector (`vision_tower.*`, `embed_vision.*` — `export_vision.py`)
  and the audio tower's with its projector (`audio_tower.*`, `embed_audio.*` — `export_audio.py`)
  are read.
- **Model implementation** — the `transformers` package (`EmbeddingGemma2TextModel`, its config and,
  for comparison only, the bidirectional sliding mask function of `masking_utils`; for the image
  path `Gemma4VisionModel`, `EmbeddingGemma2MultimodalEmbedder` and the image processor
  `Gemma4ImageProcessor`, which makes the goldens' patch rows; for the audio path
  `Gemma4AudioModel`, the same embedder and the feature extractor `Gemma4AudioFeatureExtractor`,
  which makes the goldens' log-mel frames), added per run with
  `uv run --with 'transformers==5.23.0'`. Nothing is monkeypatched and nothing is copied: the
  recipe registers its own attention function through transformers' public `AttentionInterface`
  (it calls torch's SDPA) and passes its own masks through the forward's `attention_mask` argument.
  The audio export runs a run-time copy of the audio tower's layers in which the attention and the
  clipped linears are replaced by modules of the recipe's own (the attention as a window of 12 keys,
  the clips with scalar bounds); upstream's classes and the tower that makes the goldens are left as
  they are.
- **SentenceTransformer pipeline** — the `sentence-transformers` package, imported by the opt-in
  `parity.py` and `multimodal_reference.py` (`uv run --with 'sentence-transformers==6.1.0'`), as a
  measurement reference; the second writes its outputs on synthetic images of our own (the token ids
  and the vectors) to a tracked test fixture.
- **Prompt prefixes** — read verbatim at run time from the snapshot's
  `config_sentence_transformers.json`.

NOTE: this recipe has a distribution recipe (`distribution.py` / `card.py` →
`models/karume-embeddinggemma-2`, published as `hdae/karume-embeddinggemma-2`, first in 0.24.0).
The distribution carries the Apache-2.0 text (`LICENSE.md`) and a modification notice
(`NOTICE.md` — upstream ships no `NOTICE` file), in the form of the Gemma 4 distribution. Moving
the pin to another commit needs this review again.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/embeddinggemma-2

| Item                     | Value                                                                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/embeddinggemma-2>                                                                                                           |
| Revision used            | `f9972ac22e696a105c482202646979e33e4b5a1e` (fetched 2026-11-21 into `inputs/embeddinggemma2/embeddinggemma-2`; pinned with per-file SHA-256 in `sources.py`) |
| Form of copy             | Loaded and converted (the text tower) into local series — f32 (upcast), bf16 (the checkpoint's values) and i8 (the token table and every linear rounded to per-row int8) — and the bf16 and i8 ones assembled into the local mirror `karume-embeddinggemma-2` with a compiled tokenizer and the prompt prefixes; the vision tower and its projector converted into local series (f32 upcast, bf16 and i8, its two position tables split from one, its RoPE folded into two tables), the bf16 and i8 ones assembled into the mirror as the weights role `vision`; the audio tower and its projector converted into local series (f32 upcast, bf16 and i8, its chunked attention as a window of 12 keys, its relative position keys and per-dimension query scale folded into constants), the bf16 and i8 ones assembled into the mirror as the weights role `audio`. Model outputs on synthetic inputs (the soft tokens and vectors of the goldens; the 768-wide vectors and the log-mel frames of `packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.*`, tracked). The mirror is re-distributed as `hdae/karume-embeddinggemma-2` (first published in 0.24.0). |
| Code license             | n/a (weights, configs and tokenizer only)                                                                                                                  |
| Weights license          | Apache-2.0 as declared by the model card's front matter (`license: apache-2.0`; the card body links its "License: Apache 2.0" line to <https://ai.google.dev/gemma/docs/gemma_4_license>) at the revision used; not gated; the repository has no LICENSE or NOTICE file. The card's "Ethics & Safety" section also states "Deployments must adhere to the Gemma Prohibited Use Policy" (<https://ai.google.dev/gemma/prohibited_use_policy>). Read against the revision used on 2026-11-25; the maintainer decided on 2026-11-25 to publish under Apache-2.0 with that sentence quoted. |
| Attribution requirements | Apache 2.0 §4: `karume-embeddinggemma-2` bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) — the three towers extracted and re-expressed, the `bf16` and `i8` forms, the compiled tokenizer). No upstream NOTICE file (§4(d) does not apply). The card and `NOTICE.md` quote the upstream card's sentence on the Gemma Prohibited Use Policy verbatim with its link and do not reproduce the policy (maintainer decision 2026-11-25). |

### transformers (model implementation)

| Item                     | Value                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>                                               |
| Revision used            | `transformers==5.23.0` (per-run `--with`, not in `pyproject.toml`)                          |
| Form of copy             | Imported and used as-is; no patch layer, no copy. An attention function of our own is registered through the public extension point; the audio export replaces two kinds of submodules in a run-time copy of the audio tower's layers with modules of its own. |
| Code license             | Apache-2.0 — the installed wheel's own `LICENSE` (`transformers-5.23.0.dist-info/licenses/LICENSE`: "Copyright 2018- The Hugging Face team. All rights reserved." and the Apache 2.0 text), checked 2026-11-25. |
| Weights license          | n/a                                                                                         |
| Attribution requirements | None attach here: no `transformers` code is copied into this directory or redistributed (import-time dependency only), and Apache 2.0 §4 starts at redistribution. Nothing from `transformers` enters `karume-embeddinggemma-2`. |

### sentence-transformers (opt-in parity measurement)

| Item                     | Value                                                                |
| ------------------------ | -------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/sentence-transformers>                |
| Revision used            | `sentence-transformers==6.1.0` (per-run `--with`, `parity.py` and `multimodal_reference.py`) |
| Form of copy             | Imported and used as-is to measure against; nothing enters a series (its outputs on synthetic inputs are the fixture of `multimodal_reference.py`). |
| Code license             | Apache-2.0 — the installed wheel's metadata (`License-Expression: Apache-2.0`) and its own `LICENSE` (`sentence_transformers-6.1.0.dist-info/licenses/`, which also holds a `NOTICE.txt`), checked 2026-11-25. |
| Weights license          | n/a                                                                  |
| Attribution requirements | None attach here: no `sentence-transformers` code is copied or redistributed (an opt-in measurement dependency), and Apache 2.0 §4 starts at redistribution. The tracked fixture holds only model outputs on synthetic inputs of our own. |
