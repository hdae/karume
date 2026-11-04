"""Export recipe for the SigLIP2 vision tower (image-feature-extraction) (ADR 0065 decision 2).

Only what **cannot live in the general-purpose exporter core** goes here: the patch layer for the
upstream model ({@link siglip2.patch}), the export script, the reference implementation for
preprocessing parity ({@link siglip2.preprocess}), the distribution recipe
({@link siglip2.distribution}) and the card template ({@link siglip2.card}). Dependencies run
**one way only, recipe → core** (enforced mechanically by
`tools/exporter/tests/test_architecture_boundary.py`).

MUST: **do not re-export** (no `from siglip2 import X` that drags the family's contents in
transitively). The script carries heavy upstream imports, so if `import siglip2` pulled in
transformers, even paths that only touch the distribution recipe would be dragged along. Callers
import the submodule by name.

Run the script from the export-recipes root (e.g.
`uv run --group siglip2 python -m siglip2.export`).
"""
