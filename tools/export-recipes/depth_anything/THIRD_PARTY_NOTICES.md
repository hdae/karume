# Third-party notices — Depth Anything V2

Provenance record of this recipe directory and of the distribution it builds
(`karume-depth-anything-v2`): for each upstream, where it comes from, the revision used, its license
and what was checked. **It is not a license determination.**

This directory (`tools/export-recipes/depth_anything/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used. This
record is updated at each publication of a distribution built from this recipe, by the license
check of `docs/release-runbook.md` §1 ("License check"); a field that names a date records the
check made on that date, and moving a pin needs the check again.

## Upstream sources

- **Weights** — [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
  `card.py` notes that **only the Small checkpoint is Apache-2.0** in this family; adding a larger
  size to the table without redoing the license check would mis-attribute it.
- **Model implementation** — the `transformers` package (`DepthAnythingForDepthEstimation` +
  `Dinov2Backbone`), pinned `transformers==5.14.1`. `patch.py` replaces class attributes and one
  module type; nothing is copied. `export.py` self-reports a verbatim port of `DPTImageProcessor`'s
  normalization for the synthetic goldens.
- **Paper** — Depth Anything V2, <https://arxiv.org/abs/2406.09414>.

## Upstream inventory

One block per upstream above. A block is completed before a distribution that carries its upstream
is published, and checked again when its pin moves.

### depth-anything/Depth-Anything-V2-Small-hf

| Item                     | Value                                                                                                                                                                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf>                                                                                                                                                                                                                  |
| Revision used            | `5426e4f0f36572d16453bbda7a8389317b1bef99` (fetched 2026-08-16 into `inputs/depth-anything/Depth-Anything-V2-Small-hf`)                                                                                                                                                             |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                                                                                                                                                                                                       |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                  |
| Weights license          | Apache-2.0. `card.py` records `apache-2.0` (checked on the HF model API, 2026-08-14); the model card front matter at the revision used (`5426e4f0`) says `license: apache-2.0` (checked 2026-11-26). Base and Large are CC BY-NC 4.0 and are not shipped (`card.py`).                |
| Attribution requirements | Apache 2.0 §4: the distribution bundles `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (§4(b) statement of changes — container re-expression, the two bit-exact rewrites, the non-bit-exact transposed-convolution rewrite, no quantization). |

### transformers (model implementation)

| Item                     | Value                                                                                                                                                                                                                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>                                                                                                                                                                                                                                                               |
| Revision used            | `transformers==5.14.1` (pinned in `pyproject.toml`)                                                                                                                                                                                                                                                         |
| Form of copy             | Monkeypatch of imported classes; `export.py` ports `DPTImageProcessor` normalization verbatim for the synthetic goldens.                                                                                                                                                                                    |
| Code license             | Apache-2.0 — read from the installed wheel's own `LICENSE` (`transformers 5.14.1`, checked 2026-09-05; "Copyright 2018- The Hugging Face team").                                                                                                                                                            |
| Weights license          | n/a                                                                                                                                                                                                                                                                                                         |
| Attribution requirements | Apache 2.0 §4(a)/(b) attaches to the ported normalization, which lives only in this repo-only directory and is attributed here and in `export.py`. Nothing from `transformers` enters the published distribution (import-time dependency), so the distribution's `LICENSE.md` / `NOTICE.md` are unaffected. |
