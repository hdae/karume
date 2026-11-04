"""Build the reference fixture of the VAE decode's **fixed tiling** (#21, iteration T1).

Karume's VAE decoder graph is **structurally invariant to the resolution** (the
`model.safetensors` for 512px and for 1024px match exactly, down to the node list and the weight
bytes — measured in the safetensors form of the time;
`docs/research/2026-08-03-dynres-vae-tiling.md` §1.2), so the 512px asset can be used as is as a
**tile decoder** over 64×64 latents, with only the cutting / blending / pasting done on the host.
This is "the ground truth of the host-side numbers"; the TS implementation is
`packages/models/src/anima/tiling.ts`. The Deno-side E2E that compares against this fixture has
**not been ported to this repository yet** (the geometry match is guarded by the double freeze of
`MIRRORED_STARTS` on the Python side and `AXIS_STARTS` in
`packages/models/tests/anima_tiling_test.ts` on the TS side).

MUST: **do not call** `vae.enable_tiling()`. Upstream (`autoencoder_kl_qwenimage.tiled_decode`)
scans with `range(0, H, stride)`, so the last tile comes out short, which a fixed-shape tile
decoder cannot take. The scan is changed to "a rounded equal-spacing placement that snaps the last
tile's start to `extent − tile`" (the deliberate deviation the recon §4.2 announced) — so the
geometry matches the TS side only with **our own implementation**. **The blend formulas** are
verbatim ports of upstream's `blend_v` / `blend_h`; `anima/tests/test_tiling.py` pins their
equivalence by comparing against the real methods.

MUST: the reference is taken after fake-quantizing the weights to the same dtype as the asset
series (ADR 0006 — the same rule as `anima/pipeline_ref.py`). `--dtype f16` is the default because
the TS side opens `outputs/series/anima-f16/vae_decoder` (= the f16 series) as the tile decoder.

Output (default `<repo>/outputs/series/anima-tiling-f16-1024/`):

    tiling.safetensors   latents_denorm (input of the tiled decode) / image_tiled (output)
    tiling.json          tile geometry (start positions, blend width per pair) and metadata

    uv run python -m anima.tiling
    uv run python -m anima.tiling --dtype f16 --resolution 1024
    uv run python -m anima.tiling --resolution 1344x768 \
        --latents ../../outputs/series/anima-pipeline-turbo-f16-1344x768/pipeline.safetensors

`--resolution` is **WxH** (a square can be abbreviated). A non-square resolution has no static VAE
asset, so the tiled decode is the only path (#23 — the demo side requires `--vae-tiling` too).

By default the input latent is borrowed from `latents_denorm` of
`outputs/series/anima-pipeline-turbo-f16-1024/pipeline.safetensors` (denormalized = exactly the
VAE decoder's input). A real pipeline latent is used rather than randn because how the seams show
depends on the values. When the same fixture has `image` (the **untiled** torch decode), the
difference caused by tiling is recorded in the metadata as well (an observation, not a check —
tiling is an approximation, so the difference is not 0).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from _shared.paths import SERIES_ROOT
from karume.quantize import round_weights_to_f16

from .resolution import format_resolution, parse_resolution, resolution_meta

DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
#: The reference fixture the input latent is borrowed from
#: (`python -m anima.pipeline_ref --dtype f16 --resolution 1024`).
DEFAULT_LATENTS = SERIES_ROOT / "anima-pipeline-turbo-f16-1024" / "pipeline.safetensors"
#: The tile decoder's latent width (= the input `[1,16,64,64]` of the 512px asset).
TILE_LATENT = 64
#: The minimum latent overlap of adjacent tiles (= 64px in sample space — the TS side's
#: `MIN_TILE_OVERLAP_LATENT`).
MIN_OVERLAP_LATENT = 8
SPATIAL_COMPRESSION = 8


@dataclass(frozen=True)
class TileAxis:
    """The tile placement along one latent axis (the same semantics as the TS side's
    `TileAxis`).

    MUST: the difference of start positions (the spacing) is not a field — with rounded equal
    spacing it varies by up to 1 latent per pair. Derive it from `starts` where needed
    (`blend_at`).
    """

    extent: int
    tile: int
    starts: tuple[int, ...]

    def blend_at(self, scale: int, index: int) -> int:
        """The blend width of the pair `(index − 1, index)` (**sample space** — upstream's
        unit)."""
        if not 1 <= index < len(self.starts):
            raise ValueError(
                f"blend pair {index} is out of range ({len(self.starts)} start positions = "
                f"{len(self.starts) - 1} pairs)"
            )
        return (self.tile - (self.starts[index] - self.starts[index - 1])) * scale

    def blends(self, scale: int) -> list[int]:
        """The blend widths of all pairs (`count − 1` of them, since the first tile has no
        pair)."""
        return [self.blend_at(scale, index) for index in range(1, len(self.starts))]


@dataclass(frozen=True)
class TileGeometry:
    scale: int
    channels: int
    rows: TileAxis
    cols: TileAxis

    @property
    def tiles(self) -> int:
        return len(self.rows.starts) * len(self.cols.starts)


def plan_tile_axis(extent: int, tile: int, min_overlap: int) -> TileAxis:
    """The rounded equal-spacing snap placement along one axis (the same rule as the TS side's
    `planTileAxis`).

    The count is the smallest that satisfies "overlap ≥ `min_overlap`",
    `ceil(span / (tile − min_overlap)) + 1` (`span = extent − tile`), and the start positions are
    `round(i · span / (count − 1))`. The spacing need not divide evenly; the rounding error stays
    within ±0.5, so the spacing takes only two values.
    """
    if tile < 1:
        raise ValueError(f"tile width {tile} is below 1")
    if extent < tile:
        raise ValueError(f"the latent extent {extent} is smaller than the tile width {tile}")
    if not 0 <= min_overlap < tile:
        raise ValueError(f"the minimum overlap {min_overlap} is outside [0, {tile})")
    span = extent - tile
    if span == 0:
        # Degenerate: one tile. There is no pair, so no blending, and pasting is a plain copy.
        return TileAxis(extent=extent, tile=tile, starts=(0,))
    count = -(-span // (tile - min_overlap)) + 1
    # MUST: do not use the built-in `round` — Python rounds half to even and TS's `Math.round`
    # rounds half up, so for exactly the pairs that land on a half the two start positions would
    # silently differ by 1 latent. The integer floor((2·i·span + (count − 1)) / (2·(count − 1)))
    # is rounding half up itself.
    starts = tuple((2 * index * span + (count - 1)) // (2 * (count - 1)) for index in range(count))
    # The overlap's lower bound follows from the count formula, but that derivation rests on an
    # analysis of the rounding error that cannot be followed by eye. If it broke, the ramp would
    # no longer hide the seams (a silent error that shows only in the picture), so it fails
    # structurally.
    for index in range(1, count):
        overlap = tile - (starts[index] - starts[index - 1])
        if overlap < min_overlap:
            raise ValueError(
                f"the overlap of tiles {index - 1}/{index}, {overlap}, is below the floor"
                f" {min_overlap} (latent {extent} / tile {tile} / start positions {starts})"
            )
    return TileAxis(extent=extent, tile=tile, starts=starts)


def plan_tiling(
    latent_shape: tuple[int, ...],
    tile: int = TILE_LATENT,
    min_overlap: int = MIN_OVERLAP_LATENT,
    scale: int = SPATIAL_COMPRESSION,
) -> TileGeometry:
    """The tile geometry for a `[1,C,H,W]` latent."""
    if len(latent_shape) != 4 or latent_shape[0] != 1:
        raise ValueError(f"the latent shape {list(latent_shape)} is not [1,C,H,W]")
    return TileGeometry(
        scale=scale,
        channels=latent_shape[1],
        rows=plan_tile_axis(latent_shape[2], tile, min_overlap),
        cols=plan_tile_axis(latent_shape[3], tile, min_overlap),
    )


def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Linear-ramp blend with the tile above (verbatim port of `AutoencoderKLQwenImage.blend_v`;
    in place)."""
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
            :, :, :, y, :
        ] * (y / blend_extent)
    return b


def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Linear-ramp blend with the tile on the left (verbatim port of
    `AutoencoderKLQwenImage.blend_h`; in place)."""
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
            :, :, :, :, x
        ] * (x / blend_extent)
    return b


def tiled_decode(decode, z: torch.Tensor, geometry: TileGeometry) -> torch.Tensor:
    """Decode the rank-5 latent `[1,C,1,H,W]` tile by tile and blend the tiles together.

    `decode` is "rank-5 tile latent → rank-5 image tile". The execution path (TS) reuses one
    session; here the same `vae.decode` is called repeatedly.

    MUST: blend in the order **vertical (the tile above) → horizontal (the tile on the left)**,
    rewriting the tile array in place (same as upstream `tiled_decode` — what affects a neighbour
    is the **already blended** tile).
    MUST: take the blend width **per pair** (with rounded equal spacing it differs by up to 1
    latent between pairs).
    MUST: pasting assigns regions as "tile i owns `[starts[i], starts[i+1])`, the last one up to
    the end" (the snap version of upstream's "truncate to the spacing + crop the whole").
    Naively truncating to the spacing would sum to only `extent − tile` and lose the end.
    """
    rows_axis, cols_axis = geometry.rows, geometry.cols
    scale = geometry.scale
    tiles: list[list[torch.Tensor]] = []
    for top in rows_axis.starts:
        row: list[torch.Tensor] = []
        for left in cols_axis.starts:
            tile = z[:, :, :, top : top + rows_axis.tile, left : left + cols_axis.tile]
            row.append(decode(tile.contiguous()))
        tiles.append(row)

    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(tiles[i - 1][j], tile, rows_axis.blend_at(scale, i))
            if j > 0:
                tile = blend_h(row[j - 1], tile, cols_axis.blend_at(scale, j))
            row[j] = tile

    sample = tiles[0][0]
    out = sample.new_zeros(
        sample.shape[0],
        sample.shape[1],
        sample.shape[2],
        rows_axis.extent * scale,
        cols_axis.extent * scale,
    )
    for i, top in enumerate(rows_axis.starts):
        span_rows = (
            rows_axis.starts[i + 1] if i + 1 < len(rows_axis.starts) else rows_axis.extent
        ) - top
        for j, left in enumerate(cols_axis.starts):
            span_cols = (
                cols_axis.starts[j + 1] if j + 1 < len(cols_axis.starts) else cols_axis.extent
            ) - left
            out[
                :,
                :,
                :,
                top * scale : (top + span_rows) * scale,
                left * scale : (left + span_cols) * scale,
            ] = tiles[i][j][:, :, :, : span_rows * scale, : span_cols * scale]
    return out


def geometry_meta(geometry: TileGeometry) -> dict[str, Any]:
    """Write the geometry as JSON (the TS side compares its own computation with this table — it
    does not rely on the numbers alone)."""

    def axis(value: TileAxis) -> dict[str, Any]:
        return {
            "extent": value.extent,
            "tile": value.tile,
            "starts": list(value.starts),
            "blend_sample": value.blends(geometry.scale),
        }

    return {
        "scale": geometry.scale,
        "channels": geometry.channels,
        "rows": axis(geometry.rows),
        "cols": axis(geometry.cols),
        "tiles": geometry.tiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default outputs/series/anima-tiling-<dtype>-<resolution>/",
    )
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f16")
    parser.add_argument(
        "--latents",
        type=Path,
        default=DEFAULT_LATENTS,
        help="the pipeline.safetensors the input latent is borrowed from (key latents_denorm)",
    )
    parser.add_argument(
        "--resolution",
        default="1024",
        help="WxH (e.g. 1344x768). A square can be abbreviated — the same spelling as the demo's"
        " --resolution",
    )
    args = parser.parse_args()
    try:
        width, height = parse_resolution(args.resolution)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    from safetensors.torch import load_file, save_file

    if not args.latents.is_file():
        raise SystemExit(
            f"no input latent {args.latents} — first take the pipeline reference with `python -m"
            f" anima.pipeline_ref --dtype f16 --resolution {args.resolution} …`"
            " (the tiling section of tools/export-recipes/anima/README.md)"
        )
    source = load_file(str(args.latents))
    latents = source["latents_denorm"].to(torch.float32)
    # MUST: the axis order is `[..., H, W]`. Swapping them for a non-square resolution would
    # decode a transposed latent whose element count still matches, and only the reference would
    # become a different picture (a mix-up undetectable with a square).
    expected = [height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION]
    if list(latents.shape[2:]) != expected:
        raise SystemExit(
            f"the input latent shape {list(latents.shape)} does not match --resolution"
            f" {args.resolution} (latent {expected[0]}×{expected[1]})"
        )
    geometry = plan_tiling(tuple(latents.shape))
    print(
        f"[geometry] {geometry.rows.starts} × {geometry.cols.starts} / tile {TILE_LATENT} / "
        f"blend(sample) {geometry.rows.blends(geometry.scale)}"
        f"×{geometry.cols.blends(geometry.scale)} / {geometry.tiles} tiles",
        flush=True,
    )

    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(args.repo, subfolder="vae")
    vae.to(torch.float32).eval()
    # MUST: upstream's tiling is not used (its scan differs — see the module doc). The default is
    # False, but this keeps a future change of the config's default from silently taking the
    # reference with another geometry.
    if vae.use_tiling:
        raise SystemExit(
            "vae.use_tiling is True — upstream's tiling scans differently, so it is not used"
        )
    if args.dtype == "f16":
        report = round_weights_to_f16(vae)
        print(f"[fake-quant] vae (f16): {report.describe()}", flush=True)

    decoded = 0

    def decode(tile: torch.Tensor) -> torch.Tensor:
        nonlocal decoded
        with torch.no_grad():
            out = vae.decode(tile, return_dict=False)[0]
        decoded += 1
        print(f"[decode] tile {decoded}/{geometry.tiles} {list(out.shape)}", flush=True)
        return out

    image = tiled_decode(decode, latents.unsqueeze(2), geometry)[:, :, 0].contiguous()
    tensors = {"latents_denorm": latents.contiguous(), "image_tiled": image}

    meta: dict[str, Any] = {
        "repo": args.repo,
        "dtype": args.dtype,
        **resolution_meta(width, height),
        "latents_source": str(args.latents),
        "min_overlap_latent": MIN_OVERLAP_LATENT,
        "geometry": geometry_meta(geometry),
        "image_range": [float(image.min()), float(image.max())],
        "tensors": {name: list(value.shape) for name, value in tensors.items()},
    }
    # Observation (not a check): if the same fixture has an untiled torch decode, record the
    # difference. Tiling is an approximation that confines attention's receptive field to the
    # tile, so the difference is not 0.
    full = source.get("image")
    if full is not None and full.shape == image.shape:
        difference = (image - full.to(torch.float32)).abs()
        meta["vs_full_decode"] = {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
        }
        print(
            f"[observe] difference from the untiled decode: max {difference.max():.4e}", flush=True
        )

    args.out = (
        args.out or SERIES_ROOT / f"anima-tiling-{args.dtype}-{format_resolution(width, height)}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "tiling.safetensors"))
    (args.out / "tiling.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"fixture OK: {len(tensors)} tensors → {args.out}", flush=True)


if __name__ == "__main__":
    main()
