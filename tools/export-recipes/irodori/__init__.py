"""Export recipe for Irodori-TTS (text-to-speech) (ADR 0065 decision 2 — outside the wheel,
repo-only).

Only **what cannot live in the general-purpose exporter core** goes here: the patch layer derived
from the upstream model ({@link irodori.patch}), the export scripts, the reference implementations
of the pipeline / tokenizer, quant measurement, the distribution recipe
({@link irodori.distribution}) and the card template ({@link irodori.card}). Dependencies run
**one way only, recipe → core** (`tools/exporter/tests/test_architecture_boundary.py` enforces it
mechanically).

The codec (DACVAE) is the {@link irodori.dacvae} subpackage — **a different upstream repository
under a different license**, so it is kept in one directory together with its provenance, but it
is not a separate recipe because `measure_quant` / `dacvae.host` import `irodori.export` and vice
versa (the distribution also ships both in one repository). It cannot sit at the top level either:
the upstream package is named `dacvae` itself, and `tools/export-recipes/dacvae/` would shadow it
on sys.path.

MUST: **do not re-export** (no shape in which `from irodori import X` drags the whole family in on
import). Every script has heavy upstream imports, so if `import irodori` pulled in transformers /
`irodori_tts`, even a path that only touches the distribution recipe would be dragged along.
Callers import the submodules by name.

Scripts are started from the export-recipes root (e.g.
`uv run python -m irodori.export --dtype i8`).
"""
