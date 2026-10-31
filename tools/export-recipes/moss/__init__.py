"""MOSS-Image (text-to-image) export recipe (ADR 0138 / ADR 0065 decision 2 — outside the wheel,
repository only).

Only what cannot live in the generic exporter core goes here: the pin of the upstream checkpoint
and of its `trust_remote_code` pipeline file ({@link moss.sources}), the loaders of the upstream
components ({@link moss.pipeline_ref}), the fixed prompts and their precomputed text embeddings,
and the stage-0 reference records. Dependencies point one way only, recipe → core
(`tools/exporter/tests/test_architecture_boundary.py`).

MUST: nothing is re-exported (no `from moss import X` that drags the family's modules in). The
scripts carry heavy upstream imports; `import moss` must not pull diffusers into a path that only
reads the pin in {@link moss.sources}. Import submodules by name.

Scripts are launched from the export-recipes root, e.g.
`uv run --group moss --inexact python -m moss.sources --fetch`.
"""
