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
- **Model implementation** — the `transformers` package (`Gemma4ForCausalLM`), added ad hoc with
  `uv run --with 'transformers==5.14.1'`. Nothing is monkeypatched and nothing is copied: the
  recipe renames the text keys onto the text-only class and exports through `torch.export`.
- **Tokenizer** — read verbatim at run time from the checkpoint's `tokenizer.json` and compiled
  into a Karume tokenizer asset (`tokenizer.py`); the golden prompts are this repository's own
  text.

NOTE: this recipe **does** have a distribution seat (`distribution.py` / `card.py` →
`models/karume-gemma4-e2b`). The verbatim Apache 2.0 text is vendored once for every family that
needs it as `../_shared/licenses/apache_license_2_0.txt` and shipped as the distribution's
`LICENSE.md`, alongside a `NOTICE.md` that lists the modifications required by Apache 2.0 §4(b).

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/gemma-4-E2B-it

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/gemma-4-E2B-it>                                                                                                                                                                                                                                                                                                                                                            |
| Revision used            | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` (fetched 2026-08-19 into `inputs/gemma4/gemma-4-E2B-it`; the commit is recorded in that directory's `.cache/huggingface/download/*.metadata`)                                                                                                                                                                                                                  |
| Form of copy             | Loaded, not copied. The text decoder is re-distributed in converted, quantized storage form (`models/karume-gemma4-e2b`).                                                                                                                                                                                                                                                                                 |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                        |
| Weights license          | **Apache 2.0** — the snapshot's own `README.md` frontmatter says `license: apache-2.0` and its `license_link` page carries the plain Apache 2.0 text. Checked against this revision on 2026-09-01. (An earlier claim that the Gemma Terms of Use applied was Gemma 3 knowledge carried over; it was retracted.)                                                                                           |
| Attribution requirements | Apache 2.0 §4: the distribution bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) statement of changes — text-decoder extraction, container re-expression, int4/int8 quantization, PLE sidecar, host-built RoPE). The card carries the Apache 2.0 attribution and links to the upstream model card for the usage guidance this project does not curate. |

### transformers (model implementation)

| Item                     | Value                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| Upstream repository      | <https://github.com/huggingface/transformers>                                              |
| Revision used            | `transformers==5.14.1` (ad hoc `--with`, not in `pyproject.toml`)                          |
| Form of copy             | Imported and used as-is; key renaming happens on the loaded state dict, no patch, no copy. |
| Code license             | Unverified                                                                                 |
| Weights license          | n/a                                                                                        |
| Attribution requirements | Unverified                                                                                 |
