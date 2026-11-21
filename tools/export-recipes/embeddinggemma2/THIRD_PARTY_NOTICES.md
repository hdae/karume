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
  file the scripts read. Only the text tower's tensors (`language_model.*` of `model.safetensors`)
  are read.
- **Model implementation** — the `transformers` package (`EmbeddingGemma2TextModel`, its config and,
  for comparison only, the bidirectional sliding mask function of `masking_utils`), added per run
  with `uv run --with 'transformers==5.23.0'`. Nothing is monkeypatched and nothing is copied: the
  recipe registers its own attention function through transformers' public `AttentionInterface`
  (it calls torch's SDPA) and passes its own masks through the forward's `attention_mask` argument.
- **SentenceTransformer pipeline** — the `sentence-transformers` package, imported by the opt-in
  `parity.py` only (`uv run --with 'sentence-transformers==6.1.0'`), as a measurement reference.
- **Prompt prefixes** — read verbatim at run time from the snapshot's
  `config_sentence_transformers.json`.

NOTE: this recipe has a distribution recipe (`distribution.py` / `card.py` —
`karume-embeddinggemma-2`), built as a local mirror only; no weights derived from it are published
today. The mirror carries the Apache-2.0 text (`LICENSE.md`) and a modification notice
(`NOTICE.md` — upstream ships no `NOTICE` file), in the form of the Gemma 4 distribution. The
inventory below still has to be settled before anything is published. Moving the pin to another
commit needs this review again.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/embeddinggemma-2

| Item                     | Value                                                                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/embeddinggemma-2>                                                                                                           |
| Revision used            | `f9972ac22e696a105c482202646979e33e4b5a1e` (fetched 2026-11-21 into `inputs/embeddinggemma2/embeddinggemma-2`; pinned with per-file SHA-256 in `sources.py`) |
| Form of copy             | Loaded and converted (the text tower) into local series — f32 (upcast) and bf16 (the checkpoint's values) — and the bf16 one assembled into the local mirror `karume-embeddinggemma-2` with a compiled tokenizer and the prompt prefixes. Not distributed from this repository today. |
| Code license             | n/a (weights, configs and tokenizer only)                                                                                                                  |
| Weights license          | Apache-2.0 as declared by the model card's front matter (`license: apache-2.0`) at the revision used; not gated. Unverified by the release review           |
| Attribution requirements | Unverified                                                                                                                                                 |

### transformers (model implementation)

| Item                     | Value                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>                                               |
| Revision used            | `transformers==5.23.0` (per-run `--with`, not in `pyproject.toml`)                          |
| Form of copy             | Imported and used as-is; no patch layer, no copy. An attention function of our own is registered through the public extension point. |
| Code license             | Unverified                                                                                  |
| Weights license          | n/a                                                                                         |
| Attribution requirements | Unverified                                                                                  |

### sentence-transformers (opt-in parity measurement)

| Item                     | Value                                                                |
| ------------------------ | -------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/sentence-transformers>                |
| Revision used            | `sentence-transformers==6.1.0` (per-run `--with`, `parity.py` only)  |
| Form of copy             | Imported and used as-is to measure against; nothing enters a series. |
| Code license             | Unverified                                                           |
| Weights license          | n/a                                                                  |
| Attribution requirements | Unverified                                                           |
