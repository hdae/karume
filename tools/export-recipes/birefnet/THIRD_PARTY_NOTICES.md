# Third-party notices — BiRefNet family

Provenance record of this recipe directory and of the distributions it builds
(`karume-birefnet-hr`, `karume-lucida`): for each upstream, where it comes from, the revision used,
its license and what was checked. **It is not a license determination.**

This directory (`tools/export-recipes/birefnet/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used. This
record is updated at each publication of a distribution built from this recipe, by the license
check of `docs/release-runbook.md` §1 ("License check"); a field that names a date records the
check made on that date, and moving a pin needs the check again.

## Upstream sources

- **Weights** — [ZhengPeng7/BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR) and
  [egeorcun/lucida](https://huggingface.co/egeorcun/lucida) (a fine-tune of the former).
- **Model implementation** — the `birefnet.py` module that both repositories ship (and the
  `handler.py` that only `BiRefNet_HR` ships — Hub file lists at the revisions used, checked
  2026-11-26), which `transformers` loads through `trust_remote_code`. `patch.py` replaces
  attributes on the loaded dynamic module and self-reports a verbatim port of
  `BasicLayer.forward`'s mask construction; `export.py` self-reports a verbatim port of
  `handler.py`'s `ImagePreprocessor`. `BasicLayer` sits in the `models/backbones/swin_v1.py`
  section of the vendored `birefnet.py`, which carries **Swin Transformer**'s own copyright header
  — so that port's upstream is Microsoft, not the BiRefNet author.
- **Backbone** — Swin Transformer
  ([microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)): both checkpoints
  use its Swin-v1-L as their backbone (`swin_v1_l` in the vendored
  `birefnet.py`, byte-identical in both repositories at the revisions used — checked 2026-11-26),
  whose `models/backbones/swin_v1.py` section carries its code.
- **Paper** — BiRefNet, <https://arxiv.org/abs/2401.03407>.
- **Training data** — the upstream cards list sets distributed for research purposes (DIS5K,
  COD10K, HRSOD, P3M-10k) and, for `lucida`,
  [ToonOut](https://huggingface.co/datasets/joelseytre/toonout) (CC-BY-4.0). None of it is
  redistributed, and it attaches no condition to the MIT weights; the model card repeats it as a
  caution and points to the upstream. Checked against the revisions used on 2026-11-26 (the lucida
  card's `datasets` and caution line, and the ToonOut dataset card's `license: cc-by-4.0`).

## Upstream copyright notices

ADR [0092](../../../docs/decisions/0092-distribution-repos-and-sources.md) decision 5 keeps this
recipe in the repository and settles the licensing question in prose instead: the code below this
directory is **not** covered by the project's MIT license, and the families that self-report a
verbatim port carry the upstream notice here. Both ports are MIT, whose only condition is that the
copyright notice and the permission notice travel with the copy.

- `patch.py` — `BasicLayer.forward`'s shifted-window mask construction (`_build_attn_mask`),
  ported verbatim from the `models/backbones/swin_v1.py` section of the vendored `birefnet.py`;
  the function carries the in-place notice. The section's header:

  ```
  Swin Transformer
  Copyright (c) 2021 Microsoft
  Licensed under The MIT License [see LICENSE for details]
  Written by Ze Liu, Yutong Lin, Yixuan Wei
  ```

  The `LICENSE` that header points to is the one of <https://github.com/microsoft/Swin-Transformer>
  (checked on `main` at `f82860bfb5225915aca09c3227159ee9e1df874d`, 2026-11-26; the file last
  changed in `77409a45`, 2021-03-25). It is the MIT License with the copyright line
  `Copyright (c) Microsoft Corporation.`; its permission notice is the text of
  `../_shared/licenses/mit.txt` (the upstream file indents it by four spaces and ends without the
  final period).

- `export.py` — `handler.py`'s `ImagePreprocessor`, ported verbatim from the BiRefNet
  distribution. The Hugging Face repositories declare `license: mit` on their model cards but ship
  no `LICENSE` file; the notice below is the one in the `LICENSE` of the implementation repository
  the card links (<https://github.com/ZhengPeng7/BiRefNet>, checked 2026-11-26), and the one this
  project writes into the distribution's `LICENSE.md`:

  ```
  Copyright (c) 2024 ZhengPeng
  ```

## Upstream inventory

One block per upstream above. A block is completed before a distribution that carries its upstream
is published, and checked again when its pin moves.

### ZhengPeng7/BiRefNet_HR

| Item                     | Value                                                                                                                                                                                                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/ZhengPeng7/BiRefNet_HR>                                                                                                                                                                                                                               |
| Revision used            | `a7a562f6fd16021180f2f4348f4de003a2d3d1e1` (fetched 2026-08-16 into `inputs/birefnet/BiRefNet_HR`)                                                                                                                                                                            |
| Form of copy             | Weights loaded and re-distributed in converted storage form; the bundled `birefnet.py` / `handler.py` are executed via `trust_remote_code` and partially ported verbatim (see above).                                                                                         |
| Code license             | MIT — the model card front matter of the revision used says `license: mit` (`README.md` of the fetched checkout, checked 2026-09-05); the bundled remote code carries no separate license file and falls under the repository license. Human review 2026-09-04 confirmed MIT. |
| Weights license          | `card.py` records `mit` (checked on the HF model API, 2026-08-13). Verified against the revision used: the fetched checkout's `README.md` front matter says `license: mit` (checked 2026-09-05; human review 2026-09-04).                                                     |
| Attribution requirements | MIT: `distribution.py` writes the distribution's `LICENSE.md` (the MIT text from `../_shared/licenses/mit.txt` carrying `Copyright (c) 2024 ZhengPeng` and the two Swin Transformer lines — see that block) and a `NOTICE.md` listing the container re-expression, the bit-exact layout rewrites, the rewrites equivalent up to rounding (BatchNorm as an affine, the two-stage ASPP pooling, the commuted decoder-tail 1×1 convolution and upsample) and the absence of quantization. |

### egeorcun/lucida

| Item                     | Value                                                                                                                                                                                                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/egeorcun/lucida>                                                                                                                                                                                                                                      |
| Revision used            | `6cbedc9722652dc9a3df91dd871f0c4f3334e922` (fetched 2026-08-16 into `inputs/birefnet/lucida`)                                                                                                                                                                                 |
| Form of copy             | Weights loaded and re-distributed in converted storage form: `f32` as is, `f16` rounded to the nearest f16 (stated as quantization in `NOTICE.md`).                                                                                                                           |
| Code license             | MIT — the model card front matter of the revision used says `license: mit` (`README.md` of the fetched checkout, checked 2026-09-05); the bundled remote code carries no separate license file and falls under the repository license. Human review 2026-09-04 confirmed MIT. |
| Weights license          | `card.py` records `mit` (checked on the HF model API, 2026-08-13), on top of BiRefNet_HR's own notice. Verified against the revision used: the fetched checkout's `README.md` front matter says `license: mit` (checked 2026-09-05; human review 2026-09-04).                 |
| Attribution requirements | MIT: `distribution.py` writes the distribution's `LICENSE.md` carrying **four** copyright lines (the fine-tune author, BiRefNet upstream and the Swin Transformer backbone's two — see that block) and a `NOTICE.md` listing the changes. `card.py` carries the ToonOut (CC-BY-4.0) and upstream-copyright lines. |

### microsoft/Swin-Transformer

| Item                     | Value |
| ------------------------ | ----- |
| Upstream repository      | <https://github.com/microsoft/Swin-Transformer> |
| Revision used            | No checkout: its code reaches this recipe as the `models/backbones/swin_v1.py` section of the `birefnet.py` that both Hugging Face repositories ship at the revisions above (byte-identical in both). The `LICENSE` was read on `main` at `f82860bfb5225915aca09c3227159ee9e1df874d` (2026-11-26; the file last changed in `77409a45`, 2021-03-25). |
| Form of copy             | The backbone (Swin-v1-L) is executed via `trust_remote_code` and re-expressed in the exported graph, with its weights inside the BiRefNet checkpoints; `patch.py`'s `_build_attn_mask` is a verbatim port of its shifted-window mask generation, whose values the exported graph carries as a table of window kinds. |
| Code license             | MIT — the repository's `LICENSE` (copyright line `Copyright (c) Microsoft Corporation.`), checked 2026-11-26, and the section header in `birefnet.py` (copyright line `Copyright (c) 2021 Microsoft`, "Licensed under The MIT License"), checked 2026-11-26 and the copyright line again 2026-11-27. |
| Weights license          | n/a — no weights of this repository are loaded; the backbone's weights are part of the BiRefNet checkpoints above. |
| Attribution requirements | MIT: the notice travels with both distributions (maintainer rulings of 2026-11-26 and, for the section header's line, 2026-11-27). `distribution.py` appends two lines to the copyright lines of each `LICENSE.md`, after the BiRefNet lines: `Copyright (c) 2021 Microsoft` (`SWIN_TRANSFORMER_HEADER_COPYRIGHT`, the line of the section header the ported code carries), then `Copyright (c) Microsoft Corporation.` (`SWIN_TRANSFORMER_COPYRIGHT`, the line of the repository's `LICENSE`), above the one MIT permission notice; each `NOTICE.md` names the backbone and points at its copyright notice in `LICENSE.md`. In this recipe, `patch.py`'s `_build_attn_mask` carries the in-place notice. |

### Copyright lines of the distributions' `LICENSE.md`

Neither Hugging Face repository ships a `LICENSE` file at the revision used, but both cards link a
GitHub implementation repository whose `LICENSE` states the line (checked 2026-11-26):
`Copyright (c) 2024 ZhengPeng` (<https://github.com/ZhengPeng7/BiRefNet>) and
`Copyright (c) 2026 egeorcun` (<https://github.com/egeorcun/lucida>). `BIREFNET_COPYRIGHTS` in
`distribution.py` writes exactly those lines (lucida's `LICENSE.md` carries both, as its card says
"the original copyright notice is preserved"); each repository's list then ends with the two Swin
Transformer lines, once each, after its BiRefNet lines: the section header's
`Copyright (c) 2021 Microsoft`, then the `LICENSE`'s `Copyright (c) Microsoft Corporation.`.

### The Swin Transformer notice in the distributions

Settled by the maintainer on 2026-11-26: the Swin Transformer MIT notice (its copyright line and
permission notice, as in the repository's `LICENSE`) travels with `karume-birefnet-hr` and
`karume-lucida`, not only with this recipe. Settled on 2026-11-27: the copyright line of the
section header in the vendored `birefnet.py` (`Copyright (c) 2021 Microsoft`, the notice the
ported code itself carries) travels with them too, before the `LICENSE`'s line. The backbone is Swin Transformer throughout, and the
exported graph carries the values of its mask generation, which `patch.py` ports verbatim.
`distribution.py` writes it as described in the block above.
