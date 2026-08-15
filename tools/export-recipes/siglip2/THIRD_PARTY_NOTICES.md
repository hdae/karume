# Third-party notices — SigLIP2 vision tower

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/siglip2/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224)
  and [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384).
  Only the vision tower is exported and distributed; the text tower is not.
- **Model implementation** — the `transformers` package (`Siglip*` classes), pinned
  `transformers==5.14.1`. `patch.py` replaces class attributes and self-reports that its q/k/v
  split follows `nn.MultiheadAttention`'s `_in_projection_packed` verbatim; nothing is copied.
- **Preprocessing reference** — `preprocess.py` treats `AutoImageProcessor` as the source of truth
  and generates parity fixtures from it rather than copying its code.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/siglip2-base-patch16-224 and google/siglip2-so400m-patch14-384

| Item                     | Value                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/siglip2-base-patch16-224> / <https://huggingface.co/google/siglip2-so400m-patch14-384> |
| Revision used            | Unverified                                                                                                            |
| Form of copy             | Loaded, not copied. The vision tower is re-distributed in converted storage form.                                     |
| Code license             | n/a (weights only)                                                                                                    |
| Weights license          | `card.py` records `apache-2.0` (checked on the model page, 2026-08-13). Unverified against the revision used.         |
| Attribution requirements | Unverified                                                                                                            |

### transformers (model implementation)

| Item                     | Value                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>                                                                         |
| Revision used            | `transformers==5.14.1` (pinned in `pyproject.toml`)                                                                   |
| Form of copy             | Monkeypatch of imported classes; `patch.py` follows `_in_projection_packed` verbatim for the packed-projection split. |
| Code license             | Unverified                                                                                                            |
| Weights license          | n/a                                                                                                                   |
| Attribution requirements | Unverified                                                                                                            |
