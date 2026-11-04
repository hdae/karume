"""Export recipe for SBV2 (Style-Bert-VITS2 JP-Extra / text-to-speech) (ADR 0065 decision 2).

Only what **cannot live in the general-purpose exporter core** goes here: the patch layer derived
from the upstream model ({@link sbv2.patch}), the export scripts, the host assets and the demo
({@link sbv2.demo}), the quant measurements, the distribution recipe ({@link sbv2.distribution})
and the card template ({@link sbv2.card}).
The dependency direction is **recipe → core only** (enforced mechanically by
`tools/exporter/tests/test_architecture_boundary.py`).

The `text_encoder` of the distribution only picks up the output of the DeBERTa series (written by
the `deberta` recipe) **by path** and has no code import (the shared seat shares assets, it does
not couple the code).

MUST: **do not re-export** (do not let `from sbv2 import X` drag the whole family in with it).
Every script carries heavy upstream imports, so if `import sbv2` pulled in style_bert_vits2 /
transformers, even paths that only touch the distribution recipe would be dragged along.
Callers import submodules by name.

Run the scripts from the export-recipes root (e.g. `uv run python -m sbv2.export --dtype i8`).
"""
