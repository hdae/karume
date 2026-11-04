"""Export recipe for DeBERTa-v2 (the Japanese character BERT / text encoder of SBV2) (ADR 0065
decision 2).

Only what **cannot live in the general-purpose exporter core** goes here: the patch layer derived
from the upstream model ({@link deberta.patch}) and the export script ({@link deberta.export}). It
has no distribution recipe — this series ships as the `text_encoder` component of the SBV2
distribution, so the distribution table lives in `sbv2.distribution` (that side only picks up this
series' output **by path** and does not import this package).

The dependency direction is **recipe → core only** (enforced mechanically by
`tools/exporter/tests/test_architecture_boundary.py`).

MUST: **do not re-export** (do not let `from deberta import X` drag transformers in with it).
Callers import submodules by name.

Run the scripts from the export-recipes root (e.g. `uv run python -m deberta.export --dtype i8`).
"""
