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
  `transformers==5.14.1`. `patch.py` replaces class attributes; nothing is copied.
- **Packed-projection split** — PyTorch, not transformers. `patch.py` self-reports that the order
  in which it splits `nn.MultiheadAttention`'s packed `in_proj_weight` into q / k / v follows
  `torch.nn.functional._in_projection_packed` verbatim. `nn.MultiheadAttention` and that helper
  are PyTorch's, so the attribution belongs to PyTorch.
- **Preprocessing reference** — `preprocess.py` treats `AutoImageProcessor` as the source of truth
  and generates parity fixtures from it rather than copying its code.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### google/siglip2-base-patch16-224 and google/siglip2-so400m-patch14-384

| Item                     | Value                                                                                                                                                                                                                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/google/siglip2-base-patch16-224> / <https://huggingface.co/google/siglip2-so400m-patch14-384>                                                                                                                                                                                |
| Revision used            | Unverified                                                                                                                                                                                                                                                                                           |
| Form of copy             | Loaded, not copied. The vision tower is re-distributed in converted storage form.                                                                                                                                                                                                                    |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                   |
| Weights license          | `card.py` records `apache-2.0` (checked on the model page, 2026-08-13). Unverified against the revision used.                                                                                                                                                                                        |
| Attribution requirements | Apache 2.0 §4: the distribution bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) statement of changes — vision-tower extraction, container re-expression, the two bit-exact shape rewrites, the non-bit-exact MAP head rewrite, no quantization). |

### transformers (model implementation)

| Item                     | Value                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream repository      | <https://github.com/huggingface/transformers>                                                                                                                      |
| Revision used            | `transformers==5.14.1` (pinned in `pyproject.toml`)                                                                                                                |
| Form of copy             | Monkeypatch of imported classes. Nothing is copied.                                                                                                                |
| Code license             | Apache-2.0 — read from the installed wheel's own `LICENSE` (`transformers 5.14.1`, checked 2026-09-05; "Copyright 2018- The Hugging Face team").                   |
| Weights license          | n/a                                                                                                                                                                |
| Attribution requirements | None attach here: no `transformers` code is copied into this directory or redistributed (import-time dependency only), and Apache 2.0 §4 starts at redistribution. |

### PyTorch (packed-projection split)

| Item                     | Value                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/pytorch/pytorch>                                                                                                                                                          |
| Revision used            | The `torch` version resolved by `uv.lock` (`torch>=2.13.0`, a base dependency of the core).                                                                                                   |
| Form of copy             | `patch.py` follows `torch.nn.functional._in_projection_packed`'s q / k / v split order verbatim.                                                                                              |
| Code license             | BSD-3-Clause — read from the installed wheel's own `LICENSE` (`torch 2.13.0`, checked 2026-09-05; that file lists the copyright holders, Facebook / Deepmind / NYU / NEC / IDIAP among them). |
| Weights license          | n/a                                                                                                                                                                                           |
| Attribution requirements | None attach here: what `patch.py` follows is the split _order_, not copied source, and no `torch` code is redistributed. BSD-3 clauses 1 / 2 start at redistribution of source or binary.     |
