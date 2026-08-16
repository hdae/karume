# Third-party notices — BiRefNet family

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/birefnet/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [ZhengPeng7/BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR) and
  [egeorcun/lucida](https://huggingface.co/egeorcun/lucida) (a fine-tune of the former).
- **Model implementation** — the `birefnet.py` and `handler.py` modules that those repositories
  ship and `transformers` loads through `trust_remote_code`. `patch.py` replaces attributes on the
  loaded dynamic module and self-reports a verbatim port of `BasicLayer.forward`'s mask
  construction; `export.py` self-reports a verbatim port of `handler.py`'s `ImagePreprocessor`.
- **Paper** — BiRefNet, <https://arxiv.org/abs/2401.03407>.
- **Training data** — the upstream cards list sets distributed for research purposes (DIS5K,
  COD10K, HRSOD, P3M-10k) and, for `lucida`,
  [ToonOut](https://huggingface.co/datasets/joelseytre/toonout) (CC-BY-4.0). The model card
  reproduces this; nobody has checked it against the revisions used.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### ZhengPeng7/BiRefNet_HR

| Item                     | Value                                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/ZhengPeng7/BiRefNet_HR>                                                                                                                                       |
| Revision used            | `a7a562f6fd16021180f2f4348f4de003a2d3d1e1` (fetched 2026-08-16 into `inputs/birefnet/BiRefNet_HR`)                                                                                    |
| Form of copy             | Weights loaded and re-distributed in converted storage form; the bundled `birefnet.py` / `handler.py` are executed via `trust_remote_code` and partially ported verbatim (see above). |
| Code license             | Unverified (bundled remote code)                                                                                                                                                      |
| Weights license          | `card.py` records `mit` (checked on the HF model API, 2026-08-13). Unverified against the revision used.                                                                              |
| Attribution requirements | Unverified                                                                                                                                                                            |

### egeorcun/lucida

| Item                     | Value                                                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/egeorcun/lucida>                                                                                                     |
| Revision used            | `6cbedc9722652dc9a3df91dd871f0c4f3334e922` (fetched 2026-08-16 into `inputs/birefnet/lucida`)                                                |
| Form of copy             | Weights loaded and re-distributed in converted storage form.                                                                                 |
| Code license             | Unverified (bundled remote code)                                                                                                             |
| Weights license          | `card.py` records `mit` (checked on the HF model API, 2026-08-13), on top of BiRefNet_HR's own notice. Unverified against the revision used. |
| Attribution requirements | `card.py` carries the ToonOut (CC-BY-4.0) and upstream-copyright lines. Unverified.                                                          |
