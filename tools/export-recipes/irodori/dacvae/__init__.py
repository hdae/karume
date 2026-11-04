"""Export recipe for DACVAE (the waveform ↔ latent codec of the Irodori chain).

**The upstream is a different repository with different weights from Irodori-TTS** (the
`facebookresearch/dacvae` implementation + the `Aratako/Semantic-DACVAE-Japanese-32dim` weights), so
its origin is kept in one directory where the provenance can be read (ADR 0065 decision 2 —
provenance lives with the code per family).

It sits under `irodori/` for two reasons:

1. **The code really is coupled** — {@link irodori.dacvae.host} imports `irodori.export`, and
   `irodori.measure_quant` imports {@link irodori.dacvae.export} (the distribution also ships both
   so that text to audio runs end to end from one repository — `irodori.distribution`).
2. **The upstream package is named `dacvae`** — a `tools/export-recipes/dacvae/` directory would
   shadow the upstream `dacvae` while the export-recipes root is on sys.path, and
   `from dacvae.model.dacvae import DACVAE` (how {@link irodori.dacvae.export} gets the
   implementation) would pick up itself. Nested, `irodori.dacvae` and `dacvae` are different keys
   and do not collide.

It has no distribution recipe (`Pipeline`) — the two codec graphs are shipped as the
`codec_decoder` / `codec_encoder` slots of the Irodori distribution, so `irodori.distribution` owns
the table.

MUST: **do not re-export** (no shape in which `from irodori.dacvae import X` drags the upstream
imports in). Callers import the submodules by name.

Scripts are started from the export-recipes root (e.g. `uv run python -m irodori.dacvae.export`).
"""
