"""Export recipe for Depth Anything V2 (depth-estimation) (ADR 0065 decision 2 — outside the wheel,
repository only).

Only what **cannot live in the general-purpose exporter core** goes here: the patch layer derived
from the upstream model ({@link depth_anything.patch}), the export script, the distribution recipe
({@link depth_anything.distribution}) and the card template ({@link depth_anything.card}).
Dependencies point **one way only, recipe → core** (enforced mechanically by
`tools/exporter/tests/test_architecture_boundary.py`).

The package is named `depth_anything` rather than with a hyphen because of Python's identifier
rules — the pipeline name (`--pipeline depth-anything`) and the series and repository names keep
the hyphen as before, and {@link depth_anything.distribution} owns those spellings.

MUST: **do not re-export** (do not let `from depth_anything import X` drag the family's contents
in through the import chain). The script has heavy upstream imports, so if `import depth_anything`
pulled in transformers, every path that only touches the distribution recipe would be dragged in
too. Callers import submodules by name.

Run the script from the export-recipes root (e.g.
`uv run --group depth-anything python -m depth_anything.export`).
"""
