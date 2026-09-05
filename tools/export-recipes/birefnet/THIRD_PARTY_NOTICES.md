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
  `BasicLayer` sits in the `models/backbones/swin_v1.py` section of the vendored `birefnet.py`,
  which carries **Swin Transformer**'s own copyright header — so that port's upstream is Microsoft,
  not the BiRefNet author.
- **Paper** — BiRefNet, <https://arxiv.org/abs/2401.03407>.
- **Training data** — the upstream cards list sets distributed for research purposes (DIS5K,
  COD10K, HRSOD, P3M-10k) and, for `lucida`,
  [ToonOut](https://huggingface.co/datasets/joelseytre/toonout) (CC-BY-4.0). The model card
  reproduces this; nobody has checked it against the revisions used.

## Upstream copyright notices

ADR [0092](../../../docs/decisions/0092-distribution-repos-and-sources.md) decision 5 keeps this
recipe in the repository and settles the licensing question in prose instead: the code below this
directory is **not** covered by the project's MIT license, and the families that self-report a
verbatim port carry the upstream notice here. Both ports are MIT, whose only condition is that the
copyright notice and the permission notice travel with the copy.

- `patch.py` — `BasicLayer.forward`'s shifted-window mask construction, ported verbatim from the
  `models/backbones/swin_v1.py` section of the vendored `birefnet.py`:

  ```
  Swin Transformer
  Copyright (c) 2021 Microsoft
  Licensed under The MIT License [see LICENSE for details]
  Written by Ze Liu, Yutong Lin, Yixuan Wei
  ```

- `export.py` — `handler.py`'s `ImagePreprocessor`, ported verbatim from the BiRefNet
  distribution. The upstream repositories declare `license: mit` on their model cards but ship no
  `LICENSE` file, so the notice this project carries is the one it also writes into the
  distribution's `LICENSE.md`:

  ```
  Copyright (c) 2024 ZhengPeng
  ```

  The year is not stated upstream; it is the BiRefNet paper's year, and the pre-release human
  review (ADR 0065 decision 7) owns the final call.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### ZhengPeng7/BiRefNet_HR

| Item                     | Value                                                                                                                                                                                                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/ZhengPeng7/BiRefNet_HR>                                                                                                                                                                                                                               |
| Revision used            | `a7a562f6fd16021180f2f4348f4de003a2d3d1e1` (fetched 2026-08-16 into `inputs/birefnet/BiRefNet_HR`)                                                                                                                                                                            |
| Form of copy             | Weights loaded and re-distributed in converted storage form; the bundled `birefnet.py` / `handler.py` are executed via `trust_remote_code` and partially ported verbatim (see above).                                                                                         |
| Code license             | MIT — the model card front matter of the revision used says `license: mit` (`README.md` of the fetched checkout, checked 2026-09-05); the bundled remote code carries no separate license file and falls under the repository license. Human review 2026-09-04 confirmed MIT. |
| Weights license          | `card.py` records `mit` (checked on the HF model API, 2026-08-13). Verified against the revision used: the fetched checkout's `README.md` front matter says `license: mit` (checked 2026-09-05; human review 2026-09-04).                                                     |
| Attribution requirements | MIT: the distribution bundles `LICENSE.md` (the MIT text from `../_shared/licenses/mit.txt` carrying `Copyright (c) 2024 ZhengPeng`) and a `NOTICE.md` listing the container re-expression, the layout-only rewrites and the absence of quantization.                         |

### egeorcun/lucida

| Item                     | Value                                                                                                                                                                                                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/egeorcun/lucida>                                                                                                                                                                                                                                      |
| Revision used            | `6cbedc9722652dc9a3df91dd871f0c4f3334e922` (fetched 2026-08-16 into `inputs/birefnet/lucida`)                                                                                                                                                                                 |
| Form of copy             | Weights loaded and re-distributed in converted storage form.                                                                                                                                                                                                                  |
| Code license             | MIT — the model card front matter of the revision used says `license: mit` (`README.md` of the fetched checkout, checked 2026-09-05); the bundled remote code carries no separate license file and falls under the repository license. Human review 2026-09-04 confirmed MIT. |
| Weights license          | `card.py` records `mit` (checked on the HF model API, 2026-08-13), on top of BiRefNet_HR's own notice. Verified against the revision used: the fetched checkout's `README.md` front matter says `license: mit` (checked 2026-09-05; human review 2026-09-04).                 |
| Attribution requirements | MIT: the distribution bundles `LICENSE.md` carrying **two** copyright lines (the fine-tune author and BiRefNet upstream) and a `NOTICE.md` listing the changes. `card.py` carries the ToonOut (CC-BY-4.0) and upstream-copyright lines.                                       |

### Questions the human review still has to answer

Neither row below is decided here; both are recorded so the pre-release review (ADR 0065
decision 7) has them in front of it.

1. **The copyright lines this project writes into the distribution's `LICENSE.md` are an
   inference, not a quotation.** Neither upstream repository ships a `LICENSE` file, so no
   rights holder and no year is stated anywhere upstream. `Copyright (c) 2024 ZhengPeng` takes
   the year from the BiRefNet paper, and `Copyright (c) 2026 egeorcun` takes the year the
   `lucida` card attaches to the current weights (`BIREFNET_COPYRIGHTS` in `distribution.py`).
   Confirm both holders and both years, or replace them with whatever the upstream authors
   state when asked.
2. **Whether the Swin Transformer notice (Copyright (c) 2021 Microsoft, MIT) also has to travel
   with the distribution repositories, not only with this recipe.** The verbatim port of
   `BasicLayer.forward`'s mask construction lives in `patch.py`, which is repo-only; what the
   distribution repositories carry is the exported graph and the weights, with **no Swin-derived
   source in them**. MIT's condition is about copies of the software, so the notice above is
   carried here. Confirm that the exported form does not itself count as a copy that has to
   carry it.
