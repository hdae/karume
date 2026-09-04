# SigLIP2 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstrings (`export.py` / `patch.py` / `preprocess.py` /
`distribution.py`); this file is the entry point only.

## What it emits

The **vision tower** of SigLIP2 as one graph — `SiglipVisionModel`'s `pooler_output` (`[B, hidden]`
through the MAP head), with batch static at 1 and no symbolic axis. Preprocessing is not on the
graph: the input is already-normalized `pixel_values f32 [1, 3, S, S]`. The series name comes from
the `--model-dir` directory name, and the goldens come in two groups (synthetic images by default,
plus real images with `--real-images`).

## Running

```sh
cd tools/export-recipes
uv run --group siglip2 python -m siglip2.export
uv run --group siglip2 python -m siglip2.export --verify              # eager equivalence vs the unpatched model
uv run --group siglip2-preprocess python -m siglip2.export --real-images
uv run --group siglip2-preprocess python -m siglip2.preprocess        # preprocessing parity fixture
uv run python dist.py --pipeline siglip2 \
    --model base --model so400m --out ../../models/karume-siglip2     # distribution
```

Both sizes live in **one** repository (`karume-siglip2`, default `base` — ADR
[0092](../../../docs/decisions/0092-distribution-repos-and-sources.md) decisions 1 and 8), so the
distribution command names both models and the output directory: the driver only derives a
directory name when a single model is built. The first `--model` becomes the manifest's
`defaultModel`.

`transformers` is pinned with `==` (the patch layer replaces class attributes of the modeling
code); the group is declared in [`../pyproject.toml`](../pyproject.toml).
