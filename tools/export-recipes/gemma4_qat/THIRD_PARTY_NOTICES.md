# Third-party notices — Gemma 4 mobile QAT

Provenance record of this recipe directory and of the distribution it builds
(`karume-gemma4-qat-mobile` — `distribution.py`; first published as `hdae/karume-gemma4-qat`,
renamed on 2026-11-04 when that name went to the q4_0 QAT line): for each upstream, where it comes
from, the revision used, its license and what was checked. **It is not a license determination.**

This directory (`tools/export-recipes/gemma4_qat/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). It reuses this repository's Gemma 4
wrapper (`../gemma4/`, whose own record is `../gemma4/THIRD_PARTY_NOTICES.md`) and copies no
upstream implementation source.

`Unverified` marks a field nobody has checked against the upstream revision actually used. This
record is updated at each publication of a distribution built from this recipe, by the license
check of `docs/release-runbook.md` §1 ("License check"); a field that names a date records the
check made on that date, and moving a pin needs the check again. Every block below was reviewed on
2026-11-01 for the first publication of this line (then under the name `hdae/karume-gemma4-qat`,
which since 2026-11-04 holds the q4_0 QAT line of `../gemma4/`), and the maintainer approved
publishing it, except the tokenizer block, added on 2026-11-04 when the line moved to the regular
`-it` tokenizers.

## Upstream sources

- **Weights** — `google/gemma-4-E2B-it-qat-mobile-transformers` and
  `google/gemma-4-E4B-it-qat-mobile-transformers` on Hugging Face, Google DeepMind's mobile
  quantization-aware-trained forms of `google/gemma-4-E2B-it` / `google/gemma-4-E4B-it` (the
  regular checkpoints' own record is `../gemma4/THIRD_PARTY_NOTICES.md`). Read from
  `inputs/gemma4-qat/<checkpoint>/` (or the Hugging Face cache snapshot at the pin); the revisions
  are pinned in `config.py` (`SOURCES`, with the SHA-256 of the files `reference.json`
  fingerprints), and `distribution.py` (`assert_pinned_source`) refuses a series whose recorded
  fingerprint differs from the pin. Only the text decoder and its per-layer embeddings (PLE) are
  converted.
- **Tokenizer** — `tokenizer.json` of the regular `google/gemma-4-E2B-it` /
  `google/gemma-4-E4B-it` at the glitch-token fixes of 2026-11-03 (E2B
  `4adf15748de9e1a0097b6274cee9c15836c41c37`, E4B `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` —
  `config.py` `SOURCES`, the same pins as the q4_0 QAT line, `../gemma4/sources.py`
  `REGULAR_TOKENIZER_PINS`), read from the Hugging Face cache and compiled at export time into a
  Karume tokenizer asset (`../_shared/gemma_tokenizer.py`, format `karume-gemma-tokenizer/1`). The
  mobile repositories' own `tokenizer.json` is the one from before those fixes and is not read
  (maintainer decision 2026-11-04). Its SHA-256 is pinned and checked like the weights'.
- **Configuration** — `config.json` and `generation_config.json` are read for the RoPE
  parameters, the position limit and the recommended sampler, which `karume.json` declares as
  values; neither file is redistributed.
- **Chat format** — not redistributed. The chat template is rendered by the shared Gemma reader
  (`packages/models/src/gemma/text/chat.ts`, five spellings and three branches of the upstream
  `chat_template.jinja`, no template code), the same as for regular Gemma 4.
- **Model implementation** — `transformers` (`Gemma4ForConditionalGeneration`,
  `transformers.integrations.gemma_quant`), pinned `transformers==5.14.1` (`pyproject.toml`,
  group `gemma4-qat`), imported only.

## Upstream inventory

One block per upstream above. A block is completed before a distribution that carries its upstream
is published, and checked again when its pin moves.

### google/gemma-4-E2B-it-qat-mobile-transformers / google/gemma-4-E4B-it-qat-mobile-transformers

| Item                     | Value |
| ------------------------ | ----- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers> and <https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers> (Google DeepMind; `base_model` `google/gemma-4-E2B-it` / `google/gemma-4-E4B-it`, relation `quantized`, per the HF API tags) |
| Revision used            | E2B `dd693ff40353f057ca5f07e945ad867f4afbf2ec`, E4B `8e3ac78f1a4a97c19f5f194ab362a2206f71d961` (`config.py` `SOURCES`). Observed, not chosen: the commit every file of `inputs/gemma4-qat/<checkpoint>/` was fetched at on 2026-09-11 (`.cache/huggingface/download/*.metadata`). The `model.safetensors` SHA-256 recorded in each series' `reference.json` (E2B `efab4290…a9a4`, E4B `391946da…b1d4`) equals the LFS oid the HF API reports at that commit; `tokenizer.json` (`cc8d3a0c…fe0f`, the same in both, and the same as `google/gemma-4-E2B-it`'s before the glitch-token fixes of 2026-11-03) likewise, but it is no longer read — since 2026-11-04 the tokenizer comes from the regular `-it` repositories (block below); the local `README.md`, `config.json`, `generation_config.json`, `tokenizer_config.json` and `chat_template.jinja` are byte-identical to the files served raw at that commit. Both commits are still `main` on 2026-11-01 (last modified 2026-07-20; not gated). |
| Form of copy             | Re-distributed in converted storage form in `karume-gemma4-qat-mobile` (`e2b/model/`, `e4b/model/`): the text decoder only, re-expressed in the Karume container format in states form. The fixed int2 / int4 / int8 integers and scales of the quantized linear layers, of the token embedding (shared with the output head) and of the PLE are kept without requantization (`audit.py` compares every payload and scale with the input bytes); the static activation ranges (SRQ) keep the checkpoint's scales; int4 row scales are repeated per storage group; the output head's SRQ (scales 0 upstream, the identity) is left out; the remaining parameters are stored in float32, widened exactly. PLE is a host-read container asset, RoPE inputs are host-generated, and the exit is narrowed to the selected rows. The vision and audio towers and the key/value-cache scales are not converted (counted in `reference.json` `upstreamUnused`). The tokenizer is not taken from these repositories (block below). |
| Code license             | n/a (weights, tokenizer data and configuration only) |
| Weights license          | **Apache 2.0** — the model cards' front matter at the revisions used (`license: apache-2.0`, `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`; HF API `cardData.license` at both revisions, 2026-11-01) and the card's header line "License: Apache 2.0" (linked to the same page); that page is the Apache License 2.0 text with no appendix and no copyright line (fetched 2026-11-01, "Last updated 2026-04-01 UTC"). Neither repository has a `LICENSE` or `NOTICE` file (raw `LICENSE` at the revisions: 404; HF API file list). The cards state no further condition on use or redistribution; they end with a BibTeX citation, an academic citation format rather than a redistribution condition (as for regular Gemma 4). Checked against the revisions used on 2026-11-01; reviewed on 2026-11-01 for the first publication; maintainer approved publishing. |
| Attribution requirements | Apache 2.0 §4: `karume-gemma4-qat-mobile` bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) — the converted text decoder, the kept fixed integers and SRQ scales, the float32 parameters, PLE as container assets, host-built RoPE, the narrowed exit). No upstream NOTICE file or copyright line exists, so none is to be retained (§4(c)/(d)). The model card names each model's upstream at its commit with the licence link, credits Google DeepMind, and links the upstream cards for the usage guidance this project does not curate. Non-endorsement: the card carries under its title the sentence every Karume card carries, "It is not affiliated with or endorsed by Google." (`karume.modelcard`, since 2026-11-04); `NOTICE.md` carries none, since neither the licence nor the upstream cards ask for one (maintainer decision 2026-11-01). Reviewed on 2026-11-01 for the first publication; maintainer approved publishing. |
| Published in             | `karume-gemma4-qat-mobile` (first uploaded 2026-11-01 as `hdae/karume-gemma4-qat`; renamed and uploaded under the new name 2026-11-04; uploaded again 2026-11-20 with 0.23.0): `e2b/model/`, `e4b/model/`, and the attribution in its `README.md` and `NOTICE.md`. |

### google/gemma-4-E2B-it / google/gemma-4-E4B-it (tokenizer)

| Item                     | Value |
| ------------------------ | ----- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-E2B-it> and <https://huggingface.co/google/gemma-4-E4B-it> (Google DeepMind; the base models of the mobile checkpoints) |
| Revision used            | E2B `4adf15748de9e1a0097b6274cee9c15836c41c37`, E4B `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` (`config.py` `SOURCES`, shared with the q4_0 QAT line — `../gemma4/sources.py` `REGULAR_TOKENIZER_PINS`): the commits of 2026-11-03 that changed `tokenizer.json` alone, "fix: Remediate 233 glitch tokens in tokenizer.json" (E2B, pull request #51) and "fix: Remediate 219 glitch tokens in tokenizer.json" (E4B, pull request #50) — E2B drops 267 BPE merges and renames id 249788 to `<glitch_pruned_249788>`, E4B drops 253. `tokenizer.json` SHA-256: E2B `ab26b2ad…6ad2`, E4B `91d9cf86…6bc` (the LFS blob names in the Hugging Face cache); the export checks both against the pin and records them in `reference.json`. |
| Form of copy             | `tokenizer.json` only, compiled into a Karume tokenizer asset and re-distributed as `e2b/tokenizer/tokenizer.json` / `e4b/tokenizer/tokenizer.json`, whose `source` names this repository and the checkpoint fingerprint (both sources' files and revisions). The other files of these repositories are not read for this line. |
| Code license             | n/a (tokenizer data only) |
| Weights license          | **Apache 2.0** — HF API at both commits, 2026-11-04: `cardData.license` `apache-2.0`, `license_link` `https://ai.google.dev/gemma/docs/gemma_4_license` (the same link as the mobile cards'), not gated (the regular line's record, `../gemma4/THIRD_PARTY_NOTICES.md`). |
| Attribution requirements | Apache 2.0 §4, handled with the weights above: `NOTICE.md` names the tokenizer of each base model, and the card names each tokenizer's repository at its commit with the licence link. Maintainer approved shipping this tokenizer in the line on 2026-11-04; uploaded 2026-11-04. |
| Published in             | `karume-gemma4-qat-mobile`: `e2b/tokenizer/`, `e4b/tokenizer/` (uploaded 2026-11-04). |

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
measurements. Publication follows `docs/release-runbook.md`; the pinned revision is
`SOURCES["gemma4-qat-mobile"]` of `@karume/models/gemma4-qat-mobile` (`QAT_MOBILE_SOURCES` in
`packages/models/src/gemma/config.ts`).
