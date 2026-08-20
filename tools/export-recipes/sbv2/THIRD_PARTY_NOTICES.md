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

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/rufflet17/voice_models>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Revision used            | Unverified                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Weights license          | The HF repository itself carries **no written terms** (re-checked 2026-08-20 via the HF API: no README / model card; the root holds only `.gitattributes`). The author's Booth distribution page for the FN set — <https://booth.pm/ja/items/6695672> (カセキメラ研究所; its mirror link points into this same HF repository's `FN/` subtree; checked 2026-08-20) — states verbatim: 「商用利用可能です / クレジット等不要です / マージも自由です」. **Redistribution is not addressed in writing anywhere found so far.** The page also states 「マージ品なので実在する人物の声にほぼ一致することはない（と思う）」. |
| Attribution requirements | None — the Booth page states 「クレジット等不要です」 explicitly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

### litagin/style_bert_vits2_jvnv (JVNV voice family)

| Item                     | Value                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/litagin/style_bert_vits2_jvnv>                                       |
| Revision used            | Unverified                                                                                   |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                |
| Code license             | n/a (weights only)                                                                           |
| Weights license          | `card.py` records `cc-by-sa-4.0` (checked 2026-08-09). Unverified against the revision used. |
| Attribution requirements | CC-BY-SA attribution and the JVNV corpus citation are carried by the model card. Unverified. |

### ku-nlp/deberta-v2-large-japanese-char-wwm (text encoder)

| Item                     | Value                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm>                           |
| Revision used            | Unverified                                                                                   |
| Form of copy             | Re-distributed in every SBV2 repository (see `../deberta/`).                                 |
| Code license             | n/a (weights only)                                                                           |
| Weights license          | `card.py` records `cc-by-sa-4.0` (checked 2026-08-07). Unverified against the revision used. |
| Attribution requirements | Unverified                                                                                   |

### style-bert-vits2 (model implementation)

| Item                     | Value                                                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/litagin02/Style-Bert-VITS2>                                                                                                                |
| Revision used            | `style-bert-vits2==2.5.0` (pinned in `pyproject.toml`)                                                                                                         |
| Form of copy             | Monkeypatch of imported classes plus a branch-free re-implementation of `transforms.py`'s spline in `patch.py`. No file copied into this directory.            |
| Code license             | Recorded as AGPL-3.0 in `../vowel_detector/card.py`. Unverified against the revision used, and the implication for the derived re-implementation is undecided. |
| Weights license          | n/a                                                                                                                                                            |
| Attribution requirements | Unverified                                                                                                                                                     |
