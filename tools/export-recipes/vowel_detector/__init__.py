"""Export recipe of the vowel-detection CRNN (audio → vowel sequence for lip-sync; ADR 0065
decision 2).

Only **what cannot live in the general-purpose exporter core** goes here: the patch layer derived
from the upstream model ({@link vowel_detector.patch}), the export script, the distribution recipe
({@link vowel_detector.distribution}) and the card template ({@link vowel_detector.card}).
Dependencies point **one way only, recipe → core**
(`tools/exporter/tests/test_architecture_boundary.py` enforces it mechanically).

The package is named `vowel_detector` rather than with a hyphen because of Python's identifier
rules — the pipeline name (`--pipeline vowel-detector`) and the series and repository names keep
the hyphen as before, and {@link vowel_detector.distribution} holds those.

MUST: **no re-exports** (do not make `from vowel_detector import X` drag in the whole family's
contents). Callers import the submodules by name.

Run the scripts from the export-recipes root (e.g. `uv run python -m vowel_detector.export`).
"""
