"""Export recipe for the text tower of EmbeddingGemma 2 (sentence embeddings) (ADR 0065 decision 2
— outside the wheel, repository only).

Only **what cannot live in the general-purpose exporter core** goes here: the pin of the upstream
snapshot ({@link embeddinggemma2.sources}), the compile of the tokenizer asset
({@link embeddinggemma2.tokenizer} — the same machinery as gemma4 and the first EmbeddingGemma,
`_shared/gemma_tokenizer.py`), the export of the text tower's series
({@link embeddinggemma2.export}), the opt-in parity measurement against SentenceTransformer
({@link embeddinggemma2.parity}) and the distribution recipe with its model card
({@link embeddinggemma2.distribution} / {@link embeddinggemma2.card}), and the series of the
checkpoint's vision and audio towers ({@link embeddinggemma2.export_vision} /
{@link embeddinggemma2.export_audio}).

The dependency runs **one way only, recipe → core** (enforced mechanically by
`tools/exporter/tests/test_architecture_boundary.py`).

MUST: **no re-exports** (do not make `from embeddinggemma2 import X` drag transformers in with
it). Callers import the submodules by name.

Run the scripts from the export-recipes root with the transformers release this family is pinned
to, added per run (there is no dependency group — the NOTE in `pyproject.toml`):
`uv run --with 'transformers==5.23.0' python -m embeddinggemma2.tokenizer`.
"""
