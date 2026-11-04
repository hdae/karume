"""Export recipe for MiniCPM5-1B (causal LM, one-shot form) (ADR 0065 decision 2 — outside the
wheel, repository only).

Only **what cannot go into the general-purpose exporter core** belongs here: one export script
({@link minicpm5.export}). There is no separate patch layer — the upstream modeling code is not
replaced; adding one GQA-preserving attention implementation through transformers' public
extension point (`AttentionInterface.register`) is enough (it lives in the script —
`minicpm5.export.gqa_sdpa_attention`). There is no distribution recipe either: this series is an
asset for accepting ADR 0067 decision 1 (GQA as divisible broadcast) on real weights, and there is
no form to publish to HF for now.

Dependencies point **one way only, recipe → core**
(`tools/exporter/tests/test_architecture_boundary.py` enforces it mechanically).

MUST: **do not re-export** (do not make `from minicpm5 import X` pull transformers in
transitively). Callers import the submodules by name.

Run the scripts from the export-recipes root (for example
`uv run --with 'transformers==5.14.1' python -m minicpm5.export`).
"""
