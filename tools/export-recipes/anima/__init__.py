"""Export recipe for Anima (text-to-image) (ADR 0065 decision 2 — outside the wheel, repo-only).

Only **what cannot live in the general-purpose exporter core** goes here: the patch layer derived
from the upstream model ({@link anima.patch}), the tokenizer's reference implementation
({@link anima.text}), the scripts for the reference pipeline / tiling / rope tables, quant
measurement, the distribution recipe ({@link anima.distribution}) and the card template
({@link anima.card}). Dependencies run **one way only, recipe → core**
(`tools/exporter/tests/test_architecture_boundary.py` enforces it mechanically).

MUST: **do not re-export** (no shape in which `from anima import X` drags the whole family in on
import). Every script has heavy upstream imports, so if `import anima` pulled in diffusers /
transformers, even a path that only touches `anima.resolution` would be dragged along. Callers
import the submodules by name.

Scripts are started from the export-recipes root (e.g. `uv run python -m anima.export --dtype f16`).
"""
