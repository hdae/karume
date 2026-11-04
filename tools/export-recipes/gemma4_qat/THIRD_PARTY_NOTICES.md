# Third-party notices — Gemma 4 mobile QAT

Provenance record for the pre-release review (ADR 0065 stage 6) of this recipe directory and of
the distribution `karume-gemma4-qat-mobile` it builds (`distribution.py` — first published as
`hdae/karume-gemma4-qat`, renamed on 2026-11-04 when that name went to the q4_0 QAT line).
**It is not a license determination.**

This directory (`tools/export-recipes/gemma4_qat/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). It reuses this repository's Gemma 4
wrapper (`../gemma4/`, whose own record is `../gemma4/THIRD_PARTY_NOTICES.md`) and copies no
upstream implementation source.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a review scheduled before release (ADR 0065
decision 7; the per-publication check of `docs/release-runbook.md` §1). A field that says
"Checked … on <date>" records a check against the pinned revision on that date. Every block below
was reviewed on 2026-11-01 for the first publication of `hdae/karume-gemma4-qat`, and the
maintainer approved publishing it.

## Upstream sources

- **Weights** — `google/gemma-4-E2B-it-qat-mobile-transformers` and
  `google/gemma-4-E4B-it-qat-mobile-transformers` on Hugging Face, Google DeepMind's mobile
  quantization-aware-trained forms of `google/gemma-4-E2B-it` / `google/gemma-4-E4B-it` (the
  regular checkpoints' own record is `../gemma4/THIRD_PARTY_NOTICES.md`). Read from
  `inputs/gemma4-qat/<checkpoint>/`; the revisions are pinned in `config.py` (`SOURCES`, with the
  SHA-256 of the files `reference.json` fingerprints), and `distribution.py`
  (`assert_pinned_source`) refuses a series whose recorded fingerprint differs from the pin. Only
  the text decoder, its per-layer embeddings (PLE) and the tokenizer are converted.
- **Tokenizer** — each checkpoint's `tokenizer.json`, compiled at export time into a Karume
  tokenizer asset (`../_shared/gemma_tokenizer.py`, format `karume-gemma-tokenizer/1`).
- **Configuration** — `config.json` and `generation_config.json` are read for the RoPE
  parameters, the position limit and the recommended sampler, which `karume.json` declares as
  values; neither file is redistributed.
- **Chat format** — not redistributed. The chat template is rendered by the shared Gemma reader
  (`packages/models/src/gemma/text/chat.ts`, five spellings and three branches of the upstream
  `chat_template.jinja`, no template code), the same as for regular Gemma 4.
- **Model implementation** — `transformers` (`Gemma4ForConditionalGeneration`,
  `transformers.integrations.gemma_quant`), pinned `transformers==5.14.1` (`pyproject.toml`,
  group `gemma4-qat`), imported only.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/gemma-4-E2B-it-qat-mobile-transformers / google/gemma-4-E4B-it-qat-mobile-transformers

| Item                     | Value |
| ------------------------ | ----- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers> and <https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers> (Google DeepMind; `base_model` `google/gemma-4-E2B-it` / `google/gemma-4-E4B-it`, relation `quantized`, per the HF API tags) |
| Revision used            | E2B `dd693ff40353f057ca5f07e945ad867f4afbf2ec`, E4B `8e3ac78f1a4a97c19f5f194ab362a2206f71d961` (`config.py` `SOURCES`). Observed, not chosen: the commit every file of `inputs/gemma4-qat/<checkpoint>/` was fetched at on 2026-09-11 (`.cache/huggingface/download/*.metadata`). The `model.safetensors` SHA-256 recorded in each series' `reference.json` (E2B `efab4290…a9a4`, E4B `391946da…b1d4`) equals the LFS oid the HF API reports at that commit; `tokenizer.json` (`cc8d3a0c…fe0f`, the same in both, and the same as `google/gemma-4-E2B-it`'s before Google's glitch-token fixes of 2026-11-03 — the regular line now ships the fixed one; this line keeps the checkpoints' own) likewise; the local `README.md`, `config.json`, `generation_config.json`, `tokenizer_config.json` and `chat_template.jinja` are byte-identical to the files served raw at that commit. Both commits are still `main` on 2026-11-01 (last modified 2026-07-20; not gated). |
| Form of copy             | Re-distributed in converted storage form in `karume-gemma4-qat-mobile` (`e2b/model/`, `e4b/model/`): the text decoder only, re-expressed in the Karume container format in states form. The fixed int2 / int4 / int8 integers and scales of the quantized linear layers, of the token embedding (shared with the output head) and of the PLE are kept without requantization (`audit.py` compares every payload and scale with the input bytes); the static activation ranges (SRQ) keep the checkpoint's scales; int4 row scales are repeated per storage group; the output head's SRQ (scales 0 upstream, the identity) is left out; the remaining parameters are stored in float32, widened exactly. PLE is a host-read container asset, RoPE inputs are host-generated, and the exit is narrowed to the selected rows. The vision and audio towers and the key/value-cache scales are not converted (counted in `reference.json` `upstreamUnused`). The tokenizer is re-distributed as `e2b/tokenizer/tokenizer.json` / `e4b/tokenizer/tokenizer.json`, whose `source` names the repository and the checkpoint fingerprint. |
| Code license             | n/a (weights, tokenizer data and configuration only) |
| Weights license          | **Apache 2.0** — the model cards' front matter at the revisions used (`license: apache-2.0`, `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`; HF API `cardData.license` at both revisions, 2026-11-01) and the card's header line "License: Apache 2.0" (linked to the same page); that page is the Apache License 2.0 text with no appendix and no copyright line (fetched 2026-11-01, "Last updated 2026-04-01 UTC"). Neither repository has a `LICENSE` or `NOTICE` file (raw `LICENSE` at the revisions: 404; HF API file list). The cards state no further condition on use or redistribution; they end with a BibTeX citation, an academic citation format rather than a redistribution condition (as for regular Gemma 4). Checked against the revisions used on 2026-11-01; reviewed on 2026-11-01 for the first publication; maintainer approved publishing. |
| Attribution requirements | Apache 2.0 §4: `karume-gemma4-qat-mobile` bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) — the converted text decoder, the kept fixed integers and SRQ scales, the float32 parameters, PLE as container assets, host-built RoPE, the narrowed exit). No upstream NOTICE file or copyright line exists, so none is to be retained (§4(c)/(d)). The model card names each model's upstream at its commit with the licence link, credits Google DeepMind, and links the upstream cards for the usage guidance this project does not curate. No non-endorsement sentence: neither the licence nor the upstream cards ask for one (maintainer decision 2026-11-01). Reviewed on 2026-11-01 for the first publication; maintainer approved publishing. |
| Published in             | `karume-gemma4-qat-mobile` (first uploaded 2026-11-01 as `hdae/karume-gemma4-qat`; renamed 2026-11-04, not yet uploaded under the new name): `e2b/model/`, `e2b/tokenizer/`, `e4b/model/`, `e4b/tokenizer/`, and the attribution in its `README.md` and `NOTICE.md`. |

### transformers (model implementation)

| Item                     | Value |
| ------------------------ | ----- |
| Upstream repository      | <https://github.com/huggingface/transformers> |
| Revision used            | `transformers==5.14.1` (pinned in `pyproject.toml`, group `gemma4-qat`) |
| Form of copy             | Imported and used as-is: the checkpoint is loaded through `Gemma4ForConditionalGeneration`, and the packed integers and scales are read from the loaded `QuantizedLinear` / `QuantizedEmbedding` modules. `checkpoint.py` `TraceLinear` / `TraceEmbedding` are shape-only stand-ins written here; `TraceLinear.forward` has the same three steps as `QuantizedLinear.forward` (SRQ, linear, SRQ — scale 0 meaning the identity), which is the definition of the quantization scheme rather than copied statements. Listed so the inventory does not depend on a judgement of how short a copy may be. |
| Code license             | Apache-2.0 — the installed wheel's own `LICENSE` (`transformers-5.14.1.dist-info/licenses/LICENSE`: "Copyright 2018- The Hugging Face team. All rights reserved." and the Apache 2.0 text), checked 2026-11-01. |
| Weights license          | n/a |
| Attribution requirements | None attach here: no `transformers` code is copied into this directory or redistributed (import-time dependency only), and Apache 2.0 §4 starts at redistribution. Nothing from `transformers` enters `karume-gemma4-qat-mobile`. Reviewed on 2026-11-01 for the first publication; maintainer approved publishing. |

See [ADR 0097](../../../docs/decisions/0097-gemma4-qat-integration.md) for the integration and the
[research record](../../../docs/research/2026-09-10-codex-mtp-optimization.md) for the
measurements. Release validation remains separate from the local experimental distribution.
