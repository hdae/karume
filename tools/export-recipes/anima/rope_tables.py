"""Rope table fixtures for **non-square** shapes (#23 — wiring for arbitrary and non-square
resolutions).

For the S-shaped DiT (`--dit-graph dyn` — ADR 0034) the host builds the rope cos / sin tables.
The host holds only the **per-axis base tables** (the container asset `rope_base`), and the
`[1,1,S,128]` table is built only by "laying out the base tables as `[t,h,w,t,h,w]`". At a square
resolution an **h ↔ w mix-up in this layout cannot be detected in principle** — Anima's
`rope_scale` has equal h and w (`[1.0, 4.0, 4.0]`), so `cos_h` and `cos_w` match byte for byte,
and with H'=W' the tables themselves hold the same values (detection limit 1 of ADR 0034).

**A non-square shape exposes a position mix-up** (with H'≠W', `[h position i]` and `[w position
i]` point to different rows). So the tables that upstream's `model.rope` produces are baked for
4 geometries and compared with the TS-side reconstruction (`ropeTables`) by **exact Uint32
equality**. What is baked is the output of `anima.patch.dit_rope_tables` itself (the formula is
not copied — the same discipline as the base tables).

Output (default `<repo>/outputs/series/anima-rope-nonsquare/`):

    rope.safetensors   `cos_<WxH>` / `sin_<WxH>` per geometry (each `[1,1,S,head_dim]`)
    rope.json          index of the geometries (latent dimensions, token grid, S, base-table rows)

    uv run python -m anima.rope_tables

**No weights are needed** (`CosmosRotaryPosEmbed` is pure computation with neither parameters nor
buffers), so the model is built from the config on the `meta` device — not a single byte of the
7.3GiB load happens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from _shared.paths import SERIES_ROOT

from .patch import dit_rope_tables
from .resolution import format_resolution

DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
DEFAULT_OUT = SERIES_ROOT / "anima-rope-nonsquare"
SPATIAL_COMPRESSION = 8

#: Geometries to bake (pixels, `(width, height)`). **16:9 and 3:4, each in both orientations: 4
#: patterns** (the user's instruction for #23).
#:
#: MUST: always include both members of each landscape / portrait pair. With only one, "an
#: implementation that swaps h and w" matches the other geometry's table and the numbers agree
#: with only the lookup key differing — baking the pair is what exposes the mix-up.
GEOMETRIES: tuple[tuple[int, int], ...] = (
    (1344, 768),
    (768, 1344),
    (1152, 896),
    (896, 1152),
)


def load_rope_model(repo: str) -> torch.nn.Module:
    """Only rope is needed, so the model is built from the config on the **meta device** (no
    weights are read).

    `CosmosRotaryPosEmbed` has neither parameters nor buffers, and `forward` builds the tables on
    the device of its input tensor (measured: upstream `transformer_cosmos.py`). So even with the
    body on meta, passing a CPU probe yields the tables as real CPU values.
    """
    from diffusers import CosmosTransformer3DModel

    config = CosmosTransformer3DModel.load_config(repo, subfolder="transformer")
    with torch.device("meta"):
        return CosmosTransformer3DModel.from_config(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if any(width == height for width, height in GEOMETRIES):
        raise SystemExit(
            "GEOMETRIES contains a square shape (a square cannot detect an h↔w mix-up)"
        )

    model = load_rope_model(args.repo)
    patch = tuple(int(size) for size in model.rope.patch_size)
    tensors: dict[str, torch.Tensor] = {}
    entries: list[dict[str, Any]] = []
    for width, height in GEOMETRIES:
        label = format_resolution(width, height)
        latent_height = height // SPATIAL_COMPRESSION
        latent_width = width // SPATIAL_COMPRESSION
        cos, sin = dit_rope_tables(model, latent_height, latent_width)
        tensors[f"cos_{label}"] = cos.contiguous()
        tensors[f"sin_{label}"] = sin.contiguous()
        entries.append(
            {
                "resolution": label,
                "width": width,
                "height": height,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "token_rows": latent_height // patch[1],
                "token_cols": latent_width // patch[2],
                "tokens": int(cos.shape[2]),
                "head_dim": int(cos.shape[3]),
            }
        )
        print(
            f"[rope] {label}: latent {latent_height}×{latent_width}"
            f" / S={cos.shape[2]} / head_dim={cos.shape[3]}",
            flush=True,
        )

    meta: dict[str, Any] = {
        "repo": args.repo,
        "spatial_compression": SPATIAL_COMPRESSION,
        "patch_size": list(patch),
        # Rows of the base tables (= the length of upstream's `seq = arange(max(max_size))`). The
        # source of truth for the host-side ceiling.
        "rope_base_rows": max(int(size) for size in model.rope.max_size),
        "geometries": entries,
        "tensors": {name: list(value.shape) for name, value in tensors.items()},
    }

    from safetensors.torch import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "rope.safetensors"))
    (args.out / "rope.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {len(tensors)} tensors → {args.out}", flush=True)


if __name__ == "__main__":
    main()
