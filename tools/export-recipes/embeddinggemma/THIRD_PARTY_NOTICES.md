# Third-party notices — EmbeddingGemma

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/embeddinggemma/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in *is* that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — `google/embeddinggemma-300m` on Hugging Face
  (<https://huggingface.co/google/embeddinggemma-300m>), placed by hand under
  `inputs/embeddinggemma/google-300m/`.
- **Model implementation** — the `transformers` package (`Gemma3TextModel`), added ad hoc with
  `uv run --with 'transformers==5.14.1'`. This recipe has no patch layer: nothing is monkeypatched
  and nothing is copied.
- **Prompt prefixes** — read verbatim at run time from the checkpoint's
  `config_sentence_transformers.json`; this directory keeps no copy of them.

NOTE: this recipe has no distribution seat yet (no `card.py` / `distribution.py`), so no weights
derived from it are published today. The inventory below still has to be settled before that
changes.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/embeddinggemma-300m

| Item | Value |
| ---- | ----- |
| Upstream repository | <https://huggingface.co/google/embeddinggemma-300m> |
| Revision used | Unverified |
| Form of copy | Loaded, not copied. Not distributed from this repository today. |
| Code license | n/a (weights only) |
| Weights license | Unverified |
| Attribution requirements | Unverified |

### transformers (model implementation)

| Item | Value |
| ---- | ----- |
| Upstream repository | <https://github.com/huggingface/transformers> |
| Revision used | `transformers==5.14.1` (ad hoc `--with`, not in `pyproject.toml`) |
| Form of copy | Imported and used as-is; no patch layer, no copy. |
| Code license | Unverified |
| Weights license | n/a |
| Attribution requirements | Unverified |
