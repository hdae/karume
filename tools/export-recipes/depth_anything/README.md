# Depth Anything V2 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstrings (`export.py` / `patch.py` / `distribution.py`); this file is the
entry point only.

## What it emits

One graph — `predicted_depth` (`[B, H, W]`, relative depth, up to the head's final ReLU) of the
transformers port of Depth Anything V2 (DINOv2 backbone + DPT neck / head). Batch is static 1, there
is no symbolic axis, and the resolution is the single pretrained square point read from the weights
(`image_size=518`, patch 14). Preprocessing is not on the graph: the input is already-normalized
`pixel_values f32 [1, 3, S, S]` (ImageNet statistics, bicubic — **different from SigLIP2**).
Goldens come in two groups (synthetic images by default, plus real images with `--real-images`).

**Only Small may be distributed**: upstream licenses Base / Large as CC BY-NC 4.0 and Small as
Apache-2.0 (measured through the HF API on 2026-08-14). The script accepts any of them via
`--model-dir`.

## Running

```sh
cd tools/export-recipes
uv run --group depth-anything python -m depth_anything.export
uv run --group depth-anything python -m depth_anything.export --verify   # eager equivalence vs the unpatched model
uv run --group depth-anything-preprocess python -m depth_anything.export --real-images
uv run python dist.py --pipeline depth-anything                          # distribution
```

`transformers` is pinned with `==` (the patch layer replaces class attributes of the modeling
code); the groups are declared in [`../pyproject.toml`](../pyproject.toml).
