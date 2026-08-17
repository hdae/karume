# Third-party notices — MiniCPM5-1B causal LM

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/minicpm5/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — `openbmb/MiniCPM5-1B` on Hugging Face
  (<https://huggingface.co/openbmb/MiniCPM5-1B>), placed by hand under
  `inputs/minicpm5/MiniCPM5-1B/`.
- **Model implementation** — the `transformers` package (`LlamaForCausalLM`), added ad hoc with
  `uv run --with 'transformers==5.14.1'`. This recipe has no patch layer: nothing is
  monkeypatched and nothing is copied. It only _adds_ one entry to the public attention registry
  (`AttentionInterface.register`), which keeps the grouped-query shape instead of materializing
  `repeat_kv`; the upstream `sdpa` entry is left untouched.
- **Tokenizer** — read verbatim at run time from the checkpoint's `tokenizer.json`; this
  directory keeps no copy of it, and the golden prompts are this repository's own text.

NOTE: this recipe has no distribution seat (no `card.py` / `distribution.py`), so no weights
derived from it are published today. The inventory below still has to be settled before that
changes.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### openbmb/MiniCPM5-1B

| Item                     | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/openbmb/MiniCPM5-1B>                                                       |
| Revision used            | `4e9de7a0778dc1c362e983e6858f0e77542cbdca` (fetched 2026-08-17 into `inputs/minicpm5/MiniCPM5-1B`) |
| Form of copy             | Loaded, not copied. Not distributed from this repository today.                                    |
| Code license             | n/a (weights only)                                                                                 |
| Weights license          | Unverified (the model card states apache-2.0; not checked against the revision used)               |
| Attribution requirements | Unverified                                                                                         |

### transformers (model implementation)

| Item                     | Value                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>                                          |
| Revision used            | `transformers==5.14.1` (ad hoc `--with`, not in `pyproject.toml`)                      |
| Form of copy             | Imported and used as-is; one added registry entry, no patch of upstream code, no copy. |
| Code license             | Unverified                                                                             |
| Weights license          | n/a                                                                                    |
| Attribution requirements | Unverified                                                                             |
