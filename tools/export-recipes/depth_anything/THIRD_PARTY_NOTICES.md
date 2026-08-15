# Third-party notices — Depth Anything V2

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/depth_anything/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
  `card.py` notes that **only the Small checkpoint is Apache-2.0** in this family; adding a larger
  size to the table without redoing the license check would mis-attribute it.
- **Model implementation** — the `transformers` package (`DepthAnythingForDepthEstimation` +
  `Dinov2Backbone`), pinned `transformers==5.14.1`. `patch.py` replaces class attributes and one
  module type; nothing is copied. `export.py` self-reports a verbatim port of `DPTImageProcessor`'s
  normalization for the real-image goldens.
- **Paper** — Depth Anything V2, <https://arxiv.org/abs/2406.09414>.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### depth-anything/Depth-Anything-V2-Small-hf

| Item                     | Value                                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf>                                              |
| Revision used            | Unverified                                                                                                      |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                                   |
| Code license             | n/a (weights only)                                                                                              |
| Weights license          | `card.py` records `apache-2.0` (checked on the HF model API, 2026-08-14). Unverified against the revision used. |
| Attribution requirements | Unverified                                                                                                      |

### transformers (model implementation)

| Item                     | Value                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>                                                  |
| Revision used            | `transformers==5.14.1` (pinned in `pyproject.toml`)                                            |
| Form of copy             | Monkeypatch of imported classes; `export.py` ports `DPTImageProcessor` normalization verbatim. |
| Code license             | Unverified                                                                                     |
| Weights license          | n/a                                                                                            |
| Attribution requirements | Unverified                                                                                     |
