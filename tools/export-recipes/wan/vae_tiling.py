"""**Tiled decode** of Wan's video VAE (the Python-side source of truth of the geometry and the
reference fixtures — ADR 0118 / 0121).

The spatial extent of the chunk graphs (`wan/vae_patch.py`) is a fixed tile of `t×t` latents (the
input shape of the asset; the default is the series table `export_vae.VAE_SERIES` — 32 for 2.1, 16
for 2.2).
The host cuts the full frame (832×480 = 60×104 latents) into tiles, decodes them, blends the
overlaps and stitches them together (always tiled — ADR 0118 decision 2, the video version of
ADR 0033 / 0038 §4). This module is the **Python-side source of truth of the geometry and the
blending**, and the script for the reference fixtures read by the GPU comparison
(`packages/models/tests/e2e_wan_vae_tiles_test.ts` for 2.1,
`packages/models/tests/e2e_wan_ti2v_vae_tiles_test.ts` for 2.2).
The TS side is `packages/models/src/wan/vae-tiles.ts`.

## Geometry = rounded evenly spaced snapped placement (a deliberate deviation from upstream)

Upstream `AutoencoderKLWan.tiled_decode` scans with `range(0, H, stride)`, so the last tile is
shorter, which a fixed-shape chunk graph cannot take. The start positions **divide** the range
between 0 and `extent − tile` **evenly with rounding** (the same rule as Anima — ADR 0033 addendum
P-3, `plan_tile_axis` in `anima/tiling.py`). The count is the smallest that satisfies "the overlap
of each adjacent pair is at least the minimum". The source of truth for the minimum is the output's
{@link MIN_OVERLAP_PX}; the latent value is derived by dividing it by the spatial compression (graph
ratio × unpatchify factor) ({@link min_overlap_latent} — ADR 0121 decision 6). For Wan2.1,
{@link WAN21_MIN_OVERLAP_LATENT} = 8. 832×480 (60×104 latents, tile 32) gives 12 tiles at rows
0 / 14 / 28 × columns 0 / 24 / 48 / 72 (overlaps of 18 latents on rows and 8 latents on columns).

## Blend, paste, clamp

- The blend is upstream's `blend_v` / `blend_h` **verbatim** ({@link blend_v}). Vertical (the tile
  above) → horizontal (the tile to the left), in place, in sample space (the same as upstream
  `tiled_decode` — what affects the neighbour is the already blended tile). The same formula applies
  to every frame and every channel (upstream's `b[:, :, :, y, :]` takes the whole frame axis).
- Pasting is a **region assignment** (tile i owns `[starts[i], starts[i+1])`, only the last one up
  to the end) — the snapped version of upstream's "truncate to the stride width + crop the whole"
  (ADR 0033 decision 3. Truncating to the stride width would lose the end).
- `clamp(-1, 1)` comes after pasting (the same place as upstream). The reference is written
  **before** clamping (clamping hides the differences of saturated elements — the same judgement as
  stage 4).
- Wan2.2 (upstream config `patch_size` = 2) blends and pastes in the **patchify space** (12
  channels) of the chunk graph's output, and unpatchify → clamp come after that (the same order as
  upstream `tiled_decode` — the blend width is in pixels of patchify space too, and its unit
  matches upstream's 256/2 − 192/2 = 32). The reference is written before unpatchify and clamp
  ({@link unpatchify_frames} exists only for the tests and `frames_rgb`).

## Tiles outside, chunks inside

The cache is recreated for each tile (the same as `clear_cache` in upstream `tiled_decode` — with
chunks outside, one cache per tile would have to be held at once. Decision 2). The post-quant 1×1×1
conv is applied per tile × frame too (keeping the shape of upstream `tiled_decode`. The chunk graph
also applies it per frame inside the graph).

## Fixtures (`vae_tiles.<case>.safetensors` at the series root)

Wan2.1:

- `band` (sets the band): the tiled reference `frames` of a fixed-seed latent `[16,9,60,104]`
  (832×480, 33 frames) and the **non-tiled** reference `frames_full` (the chunk loop of upstream's
  non-tiled `_decode`, before clamping — for observing the approximation error of tiling. Not a
  check).
- `accept` (decides acceptance): another seed, **portrait** `[16,3,104,60]` (480×832, 9 frames).
  Both the tile positions and the chunk boundaries differ from band. Swapping rows and columns only
  shows up in the values for a shape whose height and width differ (on a square it is an
  involution).
- The meta carries the geometry (start positions, blend widths) — compared against the TS plan (on
  top of the double freeze of ADR 0033 addendum 9a, a comparison through the fixture is added).

Wan2.2 (`--model ti2v-5b` — {@link TI2V_FIXTURE_CASES}, tile 16, overlap 4 latents) has three:
`band` (832×480×81), `accept` (480×832×9, with the non-tiled reference) and `wide` (1280×704×5 — a
size whose blend width differs per pair). `frames` are in patchify space before clamping, and the
meta adds `patch_size`. Only `accept` also writes the RGB `frames_rgb [3, F, H, W]` with upstream's
unpatchify → clamp applied (to compare the end of the GPU VAE stage — the host's unpatchify — with
upstream's function itself). The 2.1 fixtures come from the same formulas, tensors and meta alike,
as before the 2.2 path was added.

The weights are rounded to f16-representable values before the reference is taken (ADR 0006 —
`export_vae.load_vae(round_f16=True)`).

A tile other than the model's own (`--tile 32` for Wan2.2 — ADR 0134 decision 6) writes
`vae_tiles.<case>.t<tile>.safetensors` beside the default-tile references ({@link fixture_name}:
the `.t<tile>` suffix appears only when the tile differs from the series tile, so the default names
never change). The plan keeps the minimum overlap of the model, and the meta carries `tile`. A case
whose latent is smaller than the tile on either axis cannot be fed to the fixed-shape chunk graph
(`plan_tile_axis` refuses it, as the TS side does): with the default case list it is skipped and
named in the summary; named with `--case` it is refused before the VAE is loaded. At tile 32 this
leaves Wan2.2 with `wide` only (`band` and `accept` are 30 latents on one axis).

    uv run --group wan --inexact python -m wan.vae_tiling                # band + accept
    uv run --group wan --inexact python -m wan.vae_tiling --case accept  # one case only
    uv run --group wan --inexact python -m wan.vae_tiling --model ti2v-5b  # Wan2.2
    uv run --group wan --inexact python -m wan.vae_tiling --model ti2v-5b --tile 32  # *.t32

MUST: touch diffusers only inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from _shared.paths import SERIES_ROOT
from wan import export_vae, vae_patch
from wan.sources import DEFAULT_MODEL

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: The minimum overlap of adjacent tiles in output pixels. The same as upstream's default blend
#: width `tile_sample_min − tile_sample_stride` = 256 − 192 = 64 px. The latent overlap is derived
#: from it by {@link min_overlap_latent} (no latent value is held per generation — ADR 0121
#: decision 6).
MIN_OVERLAP_PX = 64
#: The unpatchify factor of Wan2.1's VAE (upstream `config.patch_size` is None = 1 — no unpatchify).
WAN21_VAE_PATCH_SIZE = 1


def _is_positive_int(value: object) -> bool:
    # bool is a subtype of int, but the TS side's `Number.isInteger` does not treat booleans as
    # integers — rejected here too, to match. Requiring int as a type is stricter than TS (TS
    # accepts 8.0, but accepting it here would let a float through).
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def min_overlap_latent(scale: int, patch_size: int) -> int:
    """The minimum latent overlap = {@link MIN_OVERLAP_PX} ÷ the spatial compression (ADR 0121
    decision 6).

    Spatial compression = the spatial ratio `scale` between the graph's input and output × the
    host's unpatchify factor `patch_size`. Taking the compression from the graph ratio alone doubles
    the overlap for a generation with unpatchify (2.2's 8 × 2 = 16 becomes 8, so 8 latents =
    128 px), so the two come in as separate arguments.

    MUST: a compression that does not divide evenly fails without rounding (rounding would silently
    move the overlap away from 64 px).
    """
    if not (_is_positive_int(scale) and _is_positive_int(patch_size)):
        raise ValueError(
            "the spatial compression factors are not positive integers"
            f" (graph ratio {scale!r}, unpatchify {patch_size!r})"
        )
    compression = scale * patch_size
    if MIN_OVERLAP_PX % compression:
        raise ValueError(
            f"the overlap {MIN_OVERLAP_PX} px is not divisible by the spatial compression"
            f" {compression} (graph ratio {scale} × unpatchify {patch_size})"
        )
    return MIN_OVERLAP_PX // compression


#: Wan2.1's overlap (latents) = 64 ÷ (`export_vae.SPATIAL_SCALE` 8 × 1) = 8. The Python side has no
#: opened graph, so the graph ratio is the recipe's declared value (the same value as the frame side
#: `tile * SPATIAL_SCALE` of the exported chunk graph).
WAN21_MIN_OVERLAP_LATENT = min_overlap_latent(export_vae.SPATIAL_SCALE, WAN21_VAE_PATCH_SIZE)

#: File name of the reference fixtures (placed at the series root — `vae_tiles.<case>.safetensors`).
FIXTURE_PREFIX = "vae_tiles."
FIXTURE_SUFFIX = ".safetensors"

#: `--tile` must be a positive multiple of this (the exported chunk-graph pairs are 16 and 32 —
#: ADR 0121 decision 6 / ADR 0134 decision 1).
TILE_MULTIPLE = 16


def fixture_tile_suffix(tile: int, default_tile: int) -> str:
    """The tile part of a reference's file name: empty at the series tile, `.t<tile>` otherwise.

    MUST: the default tile writes no suffix — the existing references and every reader of them keep
    their names (ADR 0134 decision 6).
    """
    return "" if tile == default_tile else f".t{tile}"


def fixture_name(case_name: str, tile: int, default_tile: int) -> str:
    """`vae_tiles.<case>[.t<tile>].safetensors` ({@link fixture_tile_suffix})."""
    return f"{FIXTURE_PREFIX}{case_name}{fixture_tile_suffix(tile, default_tile)}{FIXTURE_SUFFIX}"


def default_tile_of(patch_size: int | None) -> int:
    """The series tile of the generation that `patch_size` names (`export_vae.VAE_SERIES`).

    `write_fixture` already receives `patch_size` as the generation's identity (it picks the
    `reference` string and the meta by it), so the default tile is looked up the same way instead of
    being passed alongside. MUST: an ambiguous or unknown generation fails loudly — a guessed
    default tile would put a non-default reference under a default name.
    """
    tiles = {
        series.tile for series in export_vae.VAE_SERIES.values() if series.patch_size == patch_size
    }
    if len(tiles) != 1:
        raise ValueError(
            f"patch_size {patch_size!r} does not name exactly one series tile in"
            f" export_vae.VAE_SERIES (found {sorted(tiles)})"
        )
    return next(iter(tiles))


@dataclass(frozen=True)
class TileAxis:
    """The tile placement along one latent axis (the same semantics as the TS side's
    `WanVaeTileAxis`).

    MUST: the difference of start positions (the spacing) is not held as a field — with rounded even
    spacing it varies by up to 1 latent per pair. It is computed from `starts` where needed
    ({@link blend_at}, {@link region}).
    """

    extent: int
    tile: int
    starts: tuple[int, ...]

    def blend_at(self, scale: int, index: int) -> int:
        """The blend width of the pair `(index − 1, index)` (in **sample space** — the same unit as
        upstream)."""
        if not 1 <= index < len(self.starts):
            raise ValueError(
                f"blend pair {index} is out of range ({len(self.starts)} starts = "
                f"{len(self.starts) - 1} pairs)"
            )
        return (self.tile - (self.starts[index] - self.starts[index - 1])) * scale

    def blends(self, scale: int) -> list[int]:
        """The blend widths of all pairs (the first tile has no pair, so `count − 1` of them)."""
        return [self.blend_at(scale, index) for index in range(1, len(self.starts))]

    def region(self, index: int) -> int:
        """The length (latents) of the region tile `index` owns = up to the start of the next tile,
        only the last one up to the end."""
        end = self.starts[index + 1] if index + 1 < len(self.starts) else self.extent
        return end - self.starts[index]


@dataclass(frozen=True)
class TilePlan:
    """The tile placement along the two axes of a latent `[C, F, H, W]`, and the latent ↔ sample
    scale."""

    scale: int
    rows: TileAxis
    cols: TileAxis

    @property
    def tiles(self) -> int:
        return len(self.rows.starts) * len(self.cols.starts)

    def meta(self) -> dict[str, str]:
        """The geometry put into the safetensors meta (strings only) — compared against the TS
        plan."""

        def joined(values: Sequence[int]) -> str:
            return ",".join(str(value) for value in values)

        return {
            "tile": str(self.rows.tile),
            "scale": str(self.scale),
            "rows_starts": joined(self.rows.starts),
            "cols_starts": joined(self.cols.starts),
            "rows_blend": joined(self.rows.blends(self.scale)),
            "cols_blend": joined(self.cols.blends(self.scale)),
        }


def plan_tile_axis(extent: int, tile: int, min_overlap: int) -> TileAxis:
    """The rounded evenly spaced snapped placement along one axis (the same rule as the TS side's
    `planWanVaeTileAxis`).

    The count is the smallest value that keeps "the overlap at least `min_overlap`",
    `ceil(span / (tile − min_overlap)) + 1` (`span = extent − tile`); the start positions are
    `round(i · span / (count − 1))` (0.5 rounds up).

    MUST: `min_overlap` has no default — its value differs per generation
    ({@link min_overlap_latent}), so if forgetting to pass it silently went through with the 2.1
    value, another generation's plan would quietly shift.
    """
    if tile < 1:
        raise ValueError(f"tile width {tile} is less than 1")
    if extent < tile:
        raise ValueError(f"the latent extent {extent} is smaller than the tile width {tile}")
    if not 0 <= min_overlap < tile:
        raise ValueError(f"the minimum overlap {min_overlap} is outside [0, {tile})")
    span = extent - tile
    if span == 0:
        # Degenerate: one tile. There is no pair, so no blending, and pasting is a plain copy.
        return TileAxis(extent=extent, tile=tile, starts=(0,))
    count = -(-span // (tile - min_overlap)) + 1
    # MUST: do not use the built-in `round` — Python rounds half to even while TS's `Math.round`
    # rounds 0.5 up, so only the pairs that land exactly on a half would silently shift their start
    # by 1 latent. The integer expression floor((2·i·span + (count − 1)) / (2·(count − 1))) is
    # exactly rounding 0.5 up.
    starts = tuple((2 * index * span + (count - 1)) // (2 * (count - 1)) for index in range(count))
    # The minimum overlap follows from the count formula, but the derivation relies on bounding the
    # rounding error and cannot be followed by eye. If it broke, the seams would no longer be hidden
    # by the ramp (a silent error that shows only in the picture), so fail structurally.
    for index in range(1, count):
        overlap = tile - (starts[index] - starts[index - 1])
        if overlap < min_overlap:
            raise ValueError(
                f"tiles {index - 1}/{index} overlap by {overlap}, below the minimum {min_overlap}"
                f" (latent {extent} / tile {tile} / starts {starts})"
            )
    return TileAxis(extent=extent, tile=tile, starts=starts)


def plan_tiles(
    height: int,
    width: int,
    tile: int,
    min_overlap: int,
    scale: int = export_vae.SPATIAL_SCALE,
) -> TilePlan:
    """The tile plan for a latent spatial extent `height × width` (each axis independent,
    `min_overlap` in latents)."""
    return TilePlan(
        scale=scale,
        rows=plan_tile_axis(height, tile, min_overlap),
        cols=plan_tile_axis(width, tile, min_overlap),
    )


# Third-party code notice. `blend_v`, `blend_h`, `decode_tiles` and `assemble_tiles` below are
# adapted from `AutoencoderKLWan.blend_v`, `AutoencoderKLWan.blend_h` and
# `AutoencoderKLWan.tiled_decode` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# `blend_v` / `blend_h` are verbatim (as module functions instead of methods). `decode_tiles` keeps
# the inner chunk loop verbatim and walks the snapped tile starts instead of
# `range(0, H, stride)`; `assemble_tiles` keeps the blend order and replaces the stride crop and the
# final crop by the region assignment; clamp and unpatchify are left to the consumer after
# assembly.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Linear ramp blend with the tile above (upstream's `blend_v` verbatim, in place)."""
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
            :, :, :, y, :
        ] * (y / blend_extent)
    return b


def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """Linear ramp blend with the tile to the left (upstream's `blend_h` verbatim, in place)."""
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
            :, :, :, :, x
        ] * (x / blend_extent)
    return b


def decode_tiles(
    vae: AutoencoderKLWan, latents: torch.Tensor, plan: TilePlan
) -> list[list[torch.Tensor]]:
    """Decodes each tile with the chunk loop (**tiles outside, chunks inside**, before clamping).

    `latents` is the denormalized latent `[1, z, F, H, W]`. Returns the tiles in row-major order,
    `[1, C, 1 + 4(F−1), s, s]` (s = tile side × scale, C the channels of the output space — 3 pixel
    channels for 2.1, 12 channels in patchify space for 2.2). The body is the inner loop of upstream
    `tiled_decode` verbatim; only the scan is the snapped placement (module doc).

    MUST: only the generations the chunk graph rewrite accepts (`vae_patch.assert_supported` — the
    reference exists to be compared with the chunk graphs).
    """
    vae_patch.assert_supported(vae)
    _, _, num_frames, height, width = latents.shape
    if (height, width) != (plan.rows.extent, plan.cols.extent):
        raise ValueError(
            f"the latent's spatial size {height}×{width} differs from the plan's"
            f" {plan.rows.extent}×{plan.cols.extent}"
        )
    tile_height, tile_width = plan.rows.tile, plan.cols.tile
    rows: list[list[torch.Tensor]] = []
    for i in plan.rows.starts:
        row: list[torch.Tensor] = []
        for j in plan.cols.starts:
            vae.clear_cache()
            frames: list[torch.Tensor] = []
            for k in range(num_frames):
                vae._conv_idx = [0]
                tile = latents[:, :, k : k + 1, i : i + tile_height, j : j + tile_width]
                tile = vae.post_quant_conv(tile)
                decoded = vae.decoder(
                    tile, feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=(k == 0)
                )
                frames.append(decoded)
            row.append(torch.cat(frames, dim=2))
        rows.append(row)
    vae.clear_cache()
    return rows


def assemble_tiles(tiles: list[list[torch.Tensor]], plan: TilePlan) -> torch.Tensor:
    """Blends the decoded tiles (row-major) and stitches them into `[1, C, F', H·s, W·s]`.

    MUST: the blend goes vertical (above) → horizontal (left) and rewrites the tiles **in place**
    (the same as upstream — what affects the neighbour is the already blended tile. In the region
    where four tiles overlap at a corner, the order of the coefficients changes).
    MUST: the blend width is looked up **per pair** (with rounded even spacing, pairs differ by up
    to 1 latent).
    MUST: pasting is the region assignment ({@link TileAxis.region}).
    """
    rows_axis, cols_axis = plan.rows, plan.cols
    if len(tiles) != len(rows_axis.starts) or any(
        len(row) != len(cols_axis.starts) for row in tiles
    ):
        raise ValueError(
            f"tiles {[len(row) for row in tiles]} differ from the plan's "
            f"{len(rows_axis.starts)}×{len(cols_axis.starts)}"
        )
    scale = plan.scale
    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            # blend the above tile and the left tile to the current tile
            if i > 0:
                tile = blend_v(tiles[i - 1][j], tile, rows_axis.blend_at(scale, i))
            if j > 0:
                tile = blend_h(row[j - 1], tile, cols_axis.blend_at(scale, j))
            row[j] = tile

    sample = tiles[0][0]
    out = sample.new_zeros(*sample.shape[:3], rows_axis.extent * scale, cols_axis.extent * scale)
    for i, top in enumerate(rows_axis.starts):
        span_rows = rows_axis.region(i) * scale
        for j, left in enumerate(cols_axis.starts):
            span_cols = cols_axis.region(j) * scale
            out[
                :,
                :,
                :,
                top * scale : top * scale + span_rows,
                left * scale : left * scale + span_cols,
            ] = tiles[i][j][:, :, :, :span_rows, :span_cols]
    return out


def tiled_decode_unclamped(
    vae: AutoencoderKLWan, latents: torch.Tensor, plan: TilePlan
) -> torch.Tensor:
    """The tiled decode reference (before clamping) = {@link decode_tiles} →
    {@link assemble_tiles}."""
    return assemble_tiles(decode_tiles(vae, latents, plan), plan)


def unpatchify_frames(frames: torch.Tensor, patch_size: int | None) -> torch.Tensor:
    """Output space `[1, C·p², F, h, w]` → pixels `[1, C, F, h·p, w·p]` (upstream's `unpatchify`).

    If `patch_size` is None (2.1 — a generation without unpatchify), `frames` is returned as is.
    Otherwise upstream's function itself is called (with a local copy, the claim about the
    reference's provenance would become a comparison with ourselves).
    Not applied to the fixture's `frames` (the reference is written before unpatchify — module
    doc).
    """
    if patch_size is None:
        return frames
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    return unpatchify(frames, patch_size=patch_size)


# ---- Reference fixtures -------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureCase:
    """One tiled decode fixture (the latent's seed, the number of chunks, the spatial extent)."""

    name: str
    seed: int
    chunks: int
    #: The latent's spatial extent (height × width).
    height: int
    width: int
    #: `band` = the case that sets the band, `accept` = the case that decides acceptance (ADR 0118
    #: addendum 2026-10-02).
    role: str
    #: Also take the non-tiled reference (to observe the approximation error of tiling — not a
    #: check).
    full: bool
    #: Also write the RGB `frames_rgb` with upstream's unpatchify → clamp applied (only for a
    #: generation that patchifies — compares the end of the GPU VAE stage with upstream's function
    #: itself).
    rgb: bool = False


#: The band case and the acceptance case differ in the latent (seed), the chunk boundaries (number
#: of chunks) and the tile positions (height and width).
FIXTURE_CASES = (
    FixtureCase("band", seed=20261012, chunks=9, height=60, width=104, role="band", full=True),
    FixtureCase("accept", seed=20261013, chunks=3, height=104, width=60, role="accept", full=False),
)

#: The Wan2.2 TI2V-5B cases (tile 16, 48 latent channels). Built the same way as 2.1; the seeds
#: differ from both 2.1's and stage 4's chunk sequences (`export_vae.TI2V_FIXTURE_CASES`):
#:
#: - `band`: 832×480×81 (30×52 latents, 21 chunks) — the one that sets the band (stage 5 of
#:   ADR 0121's acceptance table).
#: - `accept`: portrait 480×832×9 (52×30 latents, 3 chunks) — both the tile positions and the chunk
#:   boundaries differ from band. Has the non-tiled reference (observation) and the RGB
#:   `frames_rgb`.
#: - `wide`: 1280×704×5 (44×80 latents, 2 chunks) — the only size whose blend width differs per pair
#:   (rows 56 / 48 / 56, columns 40 / 48 / 40 / 40 / 48 / 40) (832×480 is uniform across pairs).
TI2V_FIXTURE_CASES = (
    FixtureCase("band", seed=20261051, chunks=21, height=30, width=52, role="band", full=False),
    FixtureCase(
        "accept", seed=20261052, chunks=3, height=52, width=30, role="accept", full=True, rgb=True
    ),
    FixtureCase("wide", seed=20261053, chunks=2, height=44, width=80, role="accept", full=False),
)

#: Model name (a key of `export_vae.VAE_SERIES`) → the case table.
FIXTURE_CASES_BY_MODEL: dict[str, tuple[FixtureCase, ...]] = {
    "t2v-1.3b": FIXTURE_CASES,
    "ti2v-5b": TI2V_FIXTURE_CASES,
}

#: The fixture meta `reference` (the reference's provenance). The 2.1 value is the same string as
#: before the 2.2 path was added.
FIXTURE_REFERENCE = "snapped-tile decode with upstream blend_v / blend_h before clamp (CPU f32)"
FIXTURE_REFERENCE_IN_PATCHIFY_SPACE = (
    "snapped-tile decode with upstream blend_v / blend_h in patchify space"
    " before unpatchify and clamp (CPU f32)"
)


def seeded_latents(vae: AutoencoderKLWan, case: FixtureCase) -> torch.Tensor:
    """A fixed-seed random latent, denormalized, `[1, z, F, H, W]` (the value range right before
    the decode of upstream `WanPipeline` — the same formula as `export_vae.fixture_latents`)."""
    generator = torch.Generator().manual_seed(case.seed)
    channels = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, channels, 1, 1, 1)
    noise = torch.randn(1, channels, case.chunks, case.height, case.width, generator=generator)
    return noise * std + mean


def _peak_rss_gib() -> float:
    """The peak RSS of this process (GiB — Linux's `ru_maxrss` is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def _difference(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    difference = (got - want).abs()
    want_max = float(want.abs().max())
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "reference_abs_max": want_max,
        "ratio": float(difference.max()) / want_max,
    }


def write_fixture(
    vae: AutoencoderKLWan,
    case: FixtureCase,
    tile: int,
    out_root: Path,
    *,
    min_overlap: int,
    patch_size: int | None,
) -> dict[str, Any]:
    """Writes one case (latent + tiled reference; the non-tiled reference too for `full`, RGB too
    for `rgb`). Returns a summary.

    MUST: `min_overlap` (latents) and `patch_size` have no defaults — if forgetting to pass them
    silently went through with the 2.1 values, another generation's fixture would quietly shift.
    `patch_size` fails unless it matches the upstream config (so the meta's `patch_size` and
    `reference` cannot disagree with the real model).
    """
    if vae.config.patch_size != patch_size:
        raise ValueError(
            f"patch_size {patch_size!r} differs from the upstream config's"
            f" {vae.config.patch_size!r}"
        )
    if case.rgb and patch_size is None:
        raise ValueError(
            f"frames_rgb of case {case.name} is only for a generation that patchifies"
            " (patch_size is None)"
        )
    # Resolved before the decode so an unknown generation fails before minutes of CPU work.
    path = out_root / fixture_name(case.name, tile, default_tile_of(patch_size))
    plan = plan_tiles(case.height, case.width, tile, min_overlap)
    latents = seeded_latents(vae, case)
    started = time.perf_counter()
    with torch.no_grad():
        frames = tiled_decode_unclamped(vae, latents, plan)[0]
    summary: dict[str, Any] = {
        "case": case.name,
        "role": case.role,
        "latents": list(latents.shape[1:]),
        "frames": list(frames.shape),
        "tiles": plan.tiles,
        "rows_starts": list(plan.rows.starts),
        "cols_starts": list(plan.cols.starts),
        "rows_blend": plan.rows.blends(plan.scale),
        "cols_blend": plan.cols.blends(plan.scale),
        "abs_max": float(frames.abs().max()),
        "tiled_seconds": round(time.perf_counter() - started, 1),
    }
    tensors = {"latents": latents[0].contiguous(), "frames": frames.contiguous()}
    metadata = {
        "seed": str(case.seed),
        "chunks": str(case.chunks),
        "role": case.role,
        "weights": "f16-rounded",
        "reference": (
            FIXTURE_REFERENCE if patch_size is None else FIXTURE_REFERENCE_IN_PATCHIFY_SPACE
        ),
        **plan.meta(),
    }
    if patch_size is not None:
        metadata["patch_size"] = str(patch_size)
    if case.rgb:
        # The same order as the end of upstream `_decode` (unpatchify → clamp). unpatchify is a
        # permutation, so the band in patchify space can be used as is.
        rgb = unpatchify_frames(frames[None], patch_size)[0].clamp(-1.0, 1.0)
        summary["frames_rgb"] = list(rgb.shape)
        tensors["frames_rgb"] = rgb.contiguous()
    if case.full:
        started = time.perf_counter()
        with torch.no_grad():
            full = vae_patch.reference_decode_unclamped(vae, latents)[0]
        observed = {
            "unclamped": _difference(frames, full),
            "clamped": _difference(frames.clamp(-1.0, 1.0), full.clamp(-1.0, 1.0)),
            "full_seconds": round(time.perf_counter() - started, 1),
        }
        summary["vs_full"] = observed
        tensors["frames_full"] = full.contiguous()
        metadata["frames_full"] = (
            "upstream non-tiled _decode chunk loop before clamp (CPU f32)"
            if patch_size is None
            else "upstream non-tiled _decode chunk loop before unpatchify and clamp (CPU f32)"
        )
    summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)

    staging = path.with_name(path.name + ".staging")
    out_root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(staging), metadata=metadata)
    staging.replace(path)
    summary["path"] = str(path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--model",
        choices=sorted(export_vae.VAE_SERIES),
        default=DEFAULT_MODEL,
        help=f"model (series, tile and cases come from the table — default {DEFAULT_MODEL})",
    )
    tiles = ", ".join(f"{model}: {series.tile}" for model, series in export_vae.VAE_SERIES.items())
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help=(
            f"latent tile side (default from the table — {tiles}; match the input shape of the"
            " exported chunk graph)."
            f" A positive multiple of {TILE_MULTIPLE}; a non-default tile writes"
            " vae_tiles.<case>.t<tile>.safetensors"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="series root (default: the model's series — the value in the table of decision 7)",
    )
    # choices is the union over all models (each model has its own table — `wide` is 2.2 only). A
    # name that is not in the selected model's table fails after parsing.
    names = list(
        dict.fromkeys(case.name for cases in FIXTURE_CASES_BY_MODEL.values() for case in cases)
    )
    parser.add_argument(
        "--case", action="append", choices=names, default=None, help="case to write (default: all)"
    )
    args = parser.parse_args(argv)
    series = export_vae.VAE_SERIES[args.model]
    cases = FIXTURE_CASES_BY_MODEL[args.model]
    unknown = sorted(set(args.case or ()) - {case.name for case in cases})
    if unknown:
        parser.error(
            f"--case {unknown} is not in the table of --model {args.model}"
            f" ({[case.name for case in cases]})"
        )
    tile = series.tile if args.tile is None else args.tile
    if tile <= 0:
        parser.error(f"--tile must be a positive integer ({tile})")
    out = args.out if args.out is not None else SERIES_ROOT / series.series
    min_overlap = min_overlap_latent(export_vae.SPATIAL_SCALE, series.patch_size or 1)
    if tile <= min_overlap:
        # The plan (`plan_tile_axis`) fails too, but only after the VAE is read — fail before.
        parser.error(f"--tile {tile} is not above the minimum overlap {min_overlap} (latents)")
    if tile % TILE_MULTIPLE:
        parser.error(f"--tile {tile} is not a multiple of {TILE_MULTIPLE}")
    selected = [case for case in cases if args.case is None or case.name in args.case]
    # A latent smaller than the tile cannot be fed to the fixed-shape chunk graph (the planner
    # refuses it). Checked here, before the VAE is read.
    too_small = [case for case in selected if min(case.height, case.width) < tile]
    if too_small and args.case is not None:
        parser.error(
            "--case "
            + ", ".join(f"{case.name} (latent {case.height}×{case.width})" for case in too_small)
            + f" is smaller than --tile {tile}"
        )
    selected = [case for case in selected if case not in too_small]
    if not selected:
        parser.error(f"no case of --model {args.model} is at least --tile {tile} on both axes")

    vae = export_vae.load_vae(args.model, round_f16=True)
    export_vae.assert_series_config(vae, series)
    if vae.use_tiling:
        # MUST: upstream's tiling scans differently (module doc). The reference is taken only with
        # our own geometry.
        raise SystemExit("vae.use_tiling is True — upstream's tiling is not used")
    summary = {
        "series": str(out),
        "tile": tile,
        "skipped": [
            {"case": case.name, "latent": [case.height, case.width], "reason": "smaller than tile"}
            for case in too_small
        ],
        "fixtures": [
            write_fixture(
                vae, case, tile, out, min_overlap=min_overlap, patch_size=series.patch_size
            )
            for case in selected
        ],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
