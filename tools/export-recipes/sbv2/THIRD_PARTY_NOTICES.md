# Third-party notices — Style-Bert-VITS2 (SBV2)

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/sbv2/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Voice weights** — two families with different provenance, kept apart as card profiles:
  [rufflet17/voice_models](https://huggingface.co/rufflet17/voice_models) and
  [litagin/style_bert_vits2_jvnv](https://huggingface.co/litagin/style_bert_vits2_jvnv).
- **Text encoder weights** — [ku-nlp/deberta-v2-large-japanese-char-wwm](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm),
  re-distributed in every SBV2 repository (exported by `../deberta/`).
- **Model implementation** — the `style-bert-vits2` package (pinned `==2.5.0`; upstream project
  <https://github.com/litagin02/Style-Bert-VITS2>). `patch.py` replaces class attributes on
  imported classes and re-implements `transforms.py`'s spline in a branch-free equivalent form; the
  installed package is not modified and no file is copied here.
- **Corpus** — the JVNV profile cites <https://arxiv.org/abs/2310.06072> in the model card.

NOTE: elsewhere in this repository (`../vowel_detector/card.py`) Style-Bert-VITS2 is recorded as
**AGPL-3.0**. What that implies for a derived, branch-free re-implementation shipped in this
repository is exactly the question the release review has to answer; nothing here decides it.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### rufflet17/voice_models (FN voice family)

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/rufflet17/voice_models>                                                                                                                                                                                                                                                                                                                                                                                                            |
| Revision used            | `4fe4895a9c9750cce1da6a2e95e9bff1fc40bdd9` (HF cache `refs/main` — local fetch record, 2026-09-25)                                                                                                                                                                                                                                                                                                                                                         |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                                                                                                                                                                                                                                                                                                                                                                              |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Weights license          | None stated — the repository has no license metadata and no README (interview 2026-09-25); not redistributable until clarified. Separately, the author's Booth distribution page — <https://booth.pm/ja/items/6695672> (checked 2026-08-20) — allows commercial use, requires no credit and allows merging, but **does not address redistribution in writing**. Publication of the converted FN repository is **on hold** (2026-08-20 decision — backlog). |
| Attribution requirements | Unknown (no license) — the HF repository states no terms (interview 2026-09-25); the Booth page's "no credit required" (checked 2026-08-20) does not cover redistribution.                                                                                                                                                                                                                                                                                 |

### litagin/style_bert_vits2_jvnv (JVNV voice family)

| Item                     | Value                                                                                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/litagin/style_bert_vits2_jvnv>                                                                                                               |
| Revision used            | `205830ca1d49e666ddfbf2a755f0108e9cade4dd` (HF cache `refs/main` — local fetch record, 2026-09-25; the user confirmed it is the latest on HF — interview 2026-09-25) |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                                                                                        |
| Code license             | n/a (weights only)                                                                                                                                                   |
| Weights license          | `cc-by-sa-4.0`, unchanged (interview 2026-09-25). `card.py` records the same (checked 2026-08-09).                                                                   |
| Attribution requirements | CC BY-SA 4.0 attribution and the JVNV corpus citation, carried by the model card. The upstream gives no citation format (interview 2026-09-25).                      |

### ku-nlp/deberta-v2-large-japanese-char-wwm (text encoder)

| Item                     | Value                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm>                                                                                      |
| Revision used            | `547b0e8b044fba3f9b84d0ab9f990440bd130c8b` — the revision the export pins; the HF cache snapshot and `refs/main` agree (local fetch record, 2026-09-25) |
| Form of copy             | Re-distributed in every SBV2 repository (see `../deberta/`).                                                                                            |
| Code license             | n/a (weights only)                                                                                                                                      |
| Weights license          | `cc-by-sa-4.0`, unchanged (interview 2026-09-25). `card.py` records the same (checked 2026-08-07).                                                      |
| Attribution requirements | CC BY-SA 4.0 attribution, carried by the model card. The upstream gives no citation format (interview 2026-09-25).                                      |

### style-bert-vits2 (model implementation)

| Item                     | Value                                                                                                                                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/litagin02/Style-Bert-VITS2>                                                                                                                                                                                                                                       |
| Revision used            | `style-bert-vits2==2.5.0` (pinned in `pyproject.toml`)                                                                                                                                                                                                                                |
| Form of copy             | Monkeypatch of imported classes plus a branch-free re-implementation of `transforms.py`'s spline in `patch.py`. No file copied into this directory.                                                                                                                                   |
| Code license             | AGPL-3.0 — read from the installed wheel's own `LICENSE` and `License-Expression` metadata (`style-bert-vits2 2.5.0`, the pinned revision; checked 2026-09-05). The implication for the derived re-implementation in `patch.py` is still **undecided** (human review — release gate). |
| Weights license          | n/a                                                                                                                                                                                                                                                                                   |
| Attribution requirements | Unverified — they follow from the undecided question above (AGPL obligations attach only if `patch.py`'s branch-free re-implementation counts as a derived work; nothing is copied into this directory and no upstream source enters the published distribution).                     |
