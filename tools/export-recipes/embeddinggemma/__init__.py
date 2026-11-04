"""Export recipe for EmbeddingGemma-300m (sentence embeddings) (ADR 0065 decision 2 — outside the
wheel, repository only).

Only **what cannot live in the general-purpose exporter core** goes here: one export script
({@link embeddinggemma.export}) and the compile of the tokenizer asset
({@link embeddinggemma.tokenizer} — it runs on the same script family as gemma4; the
implementation is shared, **the asset is not**). There is no patch layer (the model exports without
replacing the upstream modeling code) and no distribution recipe — this series is an asset for
golden io and performance measurement, and there is no form to publish on HF for now.

The dependency runs **one way only, recipe → core** (enforced mechanically by
`tools/exporter/tests/test_architecture_boundary.py`).

MUST: **no re-exports** (do not make `from embeddinggemma import X` drag transformers in with it).
Callers import the submodules by name.

Run the scripts from the export-recipes root (for example
`uv run --with 'transformers==5.14.1' python -m embeddinggemma.export`).
"""
