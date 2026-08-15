# Third-party notices — Vowel detector (lip-sync CRNN)

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/vowel_detector/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in *is* that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [hdae/vowel-detector](https://huggingface.co/hdae/vowel-detector), the ONNX
  distribution of the same CRNN, re-distributed here in converted storage form.
- **Model definition** — `export.py` self-reports that its `Crnn` is a verbatim copy of the
  upstream `training/src/vowel_detector/crnn.py`. This is the only recipe that keeps upstream model
  code in-tree, and it is deliberate: copying the definition is what keeps this recipe free of a
  dependency group.
- **Feature contract** — `distribution.py` reads the upstream `feature_config.json` verbatim.
- **Training-pipeline attribution** — the upstream `NOTICE.txt` names a distilled-from teacher
  ([reazon-research/japanese-hubert-base-k2](https://huggingface.co/reazon-research/japanese-hubert-base-k2),
  Apache-2.0, not distributed here), reading scripts
  ([ROHAN4600](https://github.com/mmorise/rohan4600), CC0-1.0; ITA corpus, public domain), Common
  Voice ja v25.0 (CC0-1.0), and speech synthesized with
  [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) (AGPL-3.0, used as a tool
  only). `card.py` carries these lines into the model card verbatim.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### hdae/vowel-detector (weights and model definition)

| Item | Value |
| ---- | ----- |
| Upstream repository | <https://huggingface.co/hdae/vowel-detector> |
| Revision used | Unverified |
| Form of copy | Weights loaded and re-distributed in converted storage form; the model definition is copied verbatim into `export.py`. |
| Code license | `card.py` records MIT from the upstream `LICENSE` / `NOTICE.txt` (© 2026 Spectopathy). Unverified against the revision used. |
| Weights license | Same MIT notice as above. Unverified against the revision used. |
| Attribution requirements | Upstream asks that the `NOTICE.txt` attributions travel with any distribution; `card.py` reproduces them. Whether the copied model definition also needs the MIT notice in-tree is Unverified. |
