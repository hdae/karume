# Third-party notices — DeBERTa-v2 text encoder

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/deberta/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in *is* that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [ku-nlp/deberta-v2-large-japanese-char-wwm](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm).
  This recipe exports the `text_encoder` seat of the SBV2 distribution (see `../sbv2/`).
- **Model implementation** — the `transformers` package. `patch.py` replaces
  `DisentangledSelfAttention.disentangled_attention_bias` with an equivalent that takes the
  relative-position tables as graph inputs; the package itself is not modified or copied.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### ku-nlp/deberta-v2-large-japanese-char-wwm

| Item | Value |
| ---- | ----- |
| Upstream repository | <https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm> |
| Revision used | Unverified |
| Form of copy | Loaded, not copied. Re-distributed in converted storage form. |
| Code license | n/a (weights only) |
| Weights license | `../sbv2/card.py` records `cc-by-sa-4.0` (checked 2026-08-07). Unverified against the revision used. |
| Attribution requirements | Unverified |

### transformers (model implementation)

| Item | Value |
| ---- | ----- |
| Upstream repository | <https://github.com/huggingface/transformers> |
| Revision used | `transformers==5.14.1` (via the `sbv2` dependency group) |
| Form of copy | Monkeypatch of imported classes; no copy in this directory. |
| Code license | Unverified |
| Weights license | n/a |
| Attribution requirements | Unverified |
