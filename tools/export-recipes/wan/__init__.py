"""The Wan export recipe — Wan2.2 TI2V 5B, text and image to video (ADR 0118 / ADR 0121 / ADR 0065
decision 2 — outside the wheel, repository only). The Wan2.1 T2V 1.3B scripts that remain are the
source of the text assets the Wan2.2 distribution ships and of helpers the `ti2v_*` scripts import.

Only **what cannot live in the general-purpose exporter core** goes here: the pins of the upstream
sources ({@link wan.sources}), the diffusers reference pipeline ({@link wan.pipeline_ref}), the
patches and export scripts of the DiT and the video VAE, the tiling reference, the text-embedding
generation and the distribution recipe. Dependencies point **one way only, recipe → core**
(`tools/exporter/tests/test_architecture_boundary.py` enforces it mechanically).

MUST: **do not re-export** (no shape where `from wan import X` drags the family's contents in with
it). Every script has heavy upstream imports, so if `import wan` pulled in diffusers, even the paths
that only read the pins of `wan.sources` would pay for it. Callers import the submodule by name.

Run the scripts from the export-recipes root, e.g.
`uv run --group wan python -m wan.sources --fetch`.
"""
