# Third-party notices — Gemma 4 E2B

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/gemma4/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — `google/gemma-4-E2B-it` on Hugging Face
  (<https://huggingface.co/google/gemma-4-E2B-it>), downloaded into
  `inputs/gemma4/gemma-4-E2B-it/`. Only the **text decoder** keys are read; the vision and audio
  towers are never loaded.
- **12B weights** — `google/gemma-4-12B-it-qat-q4_0-unquantized` on Hugging Face
  (<https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized>), Google's
  quantization-aware-trained checkpoint of `google/gemma-4-12B-it`, read from the HF cache at the
  commit `sources.py` pins. Only the text decoder keys are read; its 4-bit grid is stored as is
  (`export_unified.py`).
- **12B tokenizer, chat template and generation defaults** — `google/gemma-4-12B-it` on Hugging
  Face (<https://huggingface.co/google/gemma-4-12B-it>), read from the HF cache at the commit
  `sources.py` pins. Its weights are no longer read: the QAT repository above ships E2B's
  `tokenizer.json` (its other tokenizer files are byte-identical to 12B-it's), so the tokenizer
  comes from the base model.
- **MTP drafter weights** — `google/gemma-4-E2B-it-assistant` on Hugging Face
  (<https://huggingface.co/google/gemma-4-E2B-it-assistant>), loaded from the HF cache by
  `export_drafter.py` and re-distributed as `e2b/drafter/`.
- **Model implementation** — the `transformers` package (`Gemma4ForCausalLM`), added ad hoc with
  `uv run --with 'transformers==5.14.1'`. Nothing is monkeypatched and nothing is copied: the
  recipe renames the text keys onto the text-only class and exports through `torch.export`.
- **Tokenizer** — read verbatim at run time from the checkpoint's `tokenizer.json` and compiled
  into a Karume tokenizer asset (`tokenizer.py`); the golden prompts are this repository's own
  text.

NOTE: this recipe **does** have a distribution seat (`distribution.py` / `card.py` →
`models/karume-gemma4`). The verbatim Apache 2.0 text is vendored once for every family that
needs it as `../_shared/licenses/apache_license_2_0.txt` and shipped as the distribution's
`LICENSE.md`, alongside a `NOTICE.md` that lists the modifications required by Apache 2.0 §4(b).

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/gemma-4-E2B-it

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-E2B-it>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Revision used            | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` (fetched 2026-08-19 into `inputs/gemma4/gemma-4-E2B-it`; the commit is recorded in that directory's `.cache/huggingface/download/*.metadata`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Form of copy             | Loaded, not copied. The text decoder is re-distributed in converted, quantized storage form (`models/karume-gemma4`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Weights license          | **Apache 2.0** — the snapshot's own `README.md` frontmatter says `license: apache-2.0` and its `license_link` page carries the plain Apache 2.0 text. Checked against this revision on 2026-09-01. (An earlier claim that the Gemma Terms of Use applied was Gemma 3 knowledge carried over; it was retracted.) Re-checked 2026-09-25: `license: apache-2.0` on the HF API at this revision (`refs/main` in the HF cache is the same commit — local fetch record, 2026-09-25), and the model page states no additional terms; the README ends with a BibTeX citation, an academic citation format rather than a redistribution condition (interview 2026-09-25). |
| Attribution requirements | Apache 2.0 §4: the distribution bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) statement of changes — text-decoder extraction, container re-expression, int4/int8 quantization, PLE as host-read container assets, host-built RoPE). The card carries the Apache 2.0 attribution and links to the upstream model card for the usage guidance this project does not curate.                                                                                                                                                                                                                                  |

### google/gemma-4-12B-it

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-12B-it>                                                                                                                                                                                                                                                                                                                                              |
| Revision used            | `43c9d014b231474fa7212dbb48a762566c2b255d` — the hub-cache snapshot `sources.py` pins as 12B's tokenizer source (`HUB_PINS["12b"].tokenizer`, fetched 2026-11-02), with the SHA-256 of `tokenizer.json` at that revision; the chat template, `tokenizer_config.json` and `generation_config.json` are read from the same snapshot                                                           |
| Form of copy             | Loaded, not copied. Only the tokenizer, the chat template and the generation defaults are read: `tokenizer.json` is compiled into the Karume tokenizer asset (`12b/tokenizer/` in `models/karume-gemma4`), the chat template is rendered for the parity fixture, and the sampler and suppressed ids enter the manifest. The weights are not read (they come from the QAT checkpoint below). |
| Code license             | n/a (tokenizer and configuration only)                                                                                                                                                                                                                                                                                                                                                      |
| Weights license          | **Apache 2.0** — the snapshot's own `README.md` frontmatter says `license: apache-2.0` (`license_link` as for E2B), read at this revision on 2026-11-02. Unverified by a human review.                                                                                                                                                                                                      |
| Attribution requirements | Apache 2.0 §4, handled as for `google/gemma-4-E2B-it`: `NOTICE.md` names this checkpoint and its changes (no per-layer embeddings to move); the card names it as the base model and the tokenizer source.                                                                                                                                                                                   |

### google/gemma-4-12B-it-qat-q4_0-unquantized (12B weights)

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized>                                                                                                                                                                                                                                                                                                                          |
| Revision used            | `b6ed86275a6a5735884e208bfed95b445a684ca2` — the hub-cache snapshot `sources.py` pins as 12B's weights source (`HUB_PINS["12b"].weights`, fetched 2026-11-02), with the SHA-256 of `model.safetensors` and `config.json` at that revision                                                                                                                                                    |
| Form of copy             | Loaded, not copied. Only the text decoder (`model.language_model.*`) is read; the encoder-free vision and audio embedders are never loaded. Re-distributed in converted storage form (`12b/model/` in `models/karume-gemma4`): the linears' own 4-bit grid (codes and one step per group of 32) is stored as is, the embedding is rounded to int8. Its `tokenizer.json` (E2B's) is not read. |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                           |
| Weights license          | **Apache 2.0** — the snapshot's own `README.md` frontmatter says `license: apache-2.0` (`license_link` as for E2B) and names `google/gemma-4-12B-it` as `base_model`, read at this revision on 2026-11-03. Unverified by a human review.                                                                                                                                                     |
| Attribution requirements | Apache 2.0 §4: the card names this repository as the source of the 12B weights; `NOTICE.md` does not name it yet (it names `google/gemma-4-12B-it`) — open for the human review.                                                                                                                                                                                                             |

### google/gemma-4-E2B-it-assistant (MTP drafter)

| Item                     | Value                                                                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-E2B-it-assistant>                                                                                                                          |
| Revision used            | `2d874ef7d29f9a30599a1e4b3c1cbc9595f005df` — the HF cache snapshot, the same as `drafter.dir` in the export's `reference.json` (local fetch record, 2026-09-25)                   |
| Form of copy             | Loaded, not copied. Re-distributed in converted, quantized storage form (`e2b/drafter/` container in `models/karume-gemma4`).                                                     |
| Code license             | n/a (weights only)                                                                                                                                                                |
| Weights license          | **Apache 2.0** — `license: apache-2.0` on the HF API at this revision (HF API at revision, 2026-09-25; the user confirmed the model page the same day — interview 2026-09-25).    |
| Attribution requirements | Apache 2.0 §4, handled as for `google/gemma-4-E2B-it`: the distribution's `LICENSE.md` / `NOTICE.md` bundle the license text and the statement of changes (interview 2026-09-25). |

### transformers (model implementation)

| Item                     | Value                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream repository      | <https://github.com/huggingface/transformers>                                                                                                                      |
| Revision used            | `transformers==5.14.1` (ad hoc `--with`, not in `pyproject.toml`)                                                                                                  |
| Form of copy             | Imported and used as-is; key renaming happens on the loaded state dict, no patch, no copy.                                                                         |
| Code license             | Apache-2.0 — read from the installed wheel's own `LICENSE` (`transformers 5.14.1`, checked 2026-09-05; "Copyright 2018- The Hugging Face team").                   |
| Weights license          | n/a                                                                                                                                                                |
| Attribution requirements | None attach here: no `transformers` code is copied into this directory or redistributed (import-time dependency only), and Apache 2.0 §4 starts at redistribution. |
