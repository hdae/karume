"""Export recipe for the BiRefNet family (image-segmentation — BiRefNet_HR / Lucida) (ADR 0065
decision 2).

Only what **cannot live in the general-purpose exporter core** goes here: the patch layer for the
upstream model ({@link birefnet.patch}), the export script, the distribution recipe
({@link birefnet.distribution}) and the card template ({@link birefnet.card}). Dependencies run
**one way only, recipe → core** (enforced mechanically by
`tools/exporter/tests/test_architecture_boundary.py`).

Lucida is not a separate recipe — its structure is identical to BiRefNet_HR and only the origin of
the weights (and the attribution and repository name tied to it) differs, so the same recipe carries
it: as one axis of `--model-dir` in the export script, and as a pipeline entry in the distribution
(`--pipeline birefnet` / `lucida`, separate repositories — ADR 0092 decision 1).

MUST: **do not re-export** (no `from birefnet import X` that drags the family's contents in
transitively). The script carries heavy upstream imports, so if `import birefnet` pulled in
transformers / timm / kornia, even paths that only touch the distribution recipe would be dragged
along. Callers import the submodule by name.

Run the script from the export-recipes root (e.g.
`uv run --group birefnet python -m birefnet.export`).
"""
