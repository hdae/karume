"""The contract of `wan/vae_tiling.py` (ADR 0118 decision 2, step 5 — the geometry and blending of
the tiled decode).

What is pinned here is the side where a break raises no exception and **only the values drift
quietly**:

- the start positions of the tile plan equal the TS side's (`packages/models/src/wan/vae-tiles.ts`)
  ({@link MIRRORED_STARTS} — frozen twice with the same values as `AXIS_STARTS` of the TS side's
  `packages/models/tests/wan_vae_tiles_test.ts`; the style of ADR 0033 addendum 9a)
- the invariants of the rounded evenly spaced snap placement (the last start is `extent − tile`,
  the overlap is at least the lower bound, the count is minimal, the spacing varies by at most one
  latent)
- the blend formula is **bit-identical** to the upstream (`AutoencoderKLWan.blend_v` / `blend_h`)
- the analytic solution of the paste (reconstructing the ramp, vertical → horizontal order at the
  corners, coverage at the ends)
- the tile reference in the degenerate case (one tile) is **bit-identical** to the upstream untiled
  chunk loop (real weights)
"""

from __future__ import annotations

import re

import pytest
import torch

from wan import vae_patch, vae_tiling

#: The real-size tile edge (latent — the default input shape of the chunk graph).
TILE = 32

#: Wan2.1's lower bound of the overlap (latent) — the value derived by the formula from the 64 px
#: output ({@link TestMinOverlapLatent}).
OVERLAP = vae_tiling.WAN21_MIN_OVERLAP_LATENT

#: Start positions at tile edge 32 and overlap lower bound 8 (the latent's full length → the
#: sequence of start positions). Frozen with **the same values** as the TS side (`AXIS_STARTS` of
#: `packages/models/tests/wan_vae_tiles_test.ts`) — the only path that compares the two
#: implementations by machine is the GPU comparison (the fixture's metadata), so this catches a
#: mirror drift on a machine without a GPU too.
#:
#: - 32: degenerate (one tile — the shape of the degenerate check)
#: - 60 / 104: the two edges of the accepted set (832×480 / 480×832 — ADR 0118 decision 7). Exactly
#:   the ADR's values (rows 0 / 14 / 28, columns 0 / 24 / 48 / 72)
#: - 57: the edge where `i·span/(count−1)` is exactly a half (span 25 / 3 tiles → 12.5). If the
#:   integer formula regresses to the built-in `round` (round half to even) it becomes (0, 12, 25) —
#:   the other edges have no halves, so this one row alone catches that regression (the same role as
#:   anima's 1872px)
MIRRORED_STARTS = {
    32: (0,),
    57: (0, 13, 25),
    60: (0, 14, 28),
    104: (0, 24, 48, 72),
}


class TestMinOverlapLatent:
    """The lower bound of the overlap is the 64 px output ÷ the spatial compression (the graph's
    ratio × the unpatchify factor — ADR 0121 decision 6).

    With only the 2.1 value (8), the formula would not turn red even if it ignored the unpatchify
    factor, so a row with factor 2 is also there to pin the formula at two points (a check of the
    formula, not of the 2.2 plan table).
    """

    @pytest.mark.parametrize(("scale", "patch_size", "expected"), [(8, 1, 8), (8, 2, 4)])
    def test_the_overlap_is_64_px_over_the_spatial_compression(
        self, scale: int, patch_size: int, expected: int
    ):
        assert vae_tiling.min_overlap_latent(scale, patch_size) == expected

    def test_wan21_overlap_is_eight_latents(self):
        """The same as the value the 2.1 plan table ({@link MIRRORED_STARTS}) was built from."""
        assert vae_tiling.WAN21_MIN_OVERLAP_LATENT == 8
        assert type(vae_tiling.WAN21_MIN_OVERLAP_LATENT) is int

    def test_a_compression_that_does_not_divide_64_px_is_rejected(self):
        """No rounding — rounding would silently move the overlap away from 64 px."""
        with pytest.raises(
            ValueError, match=r"spatial compression 24 \(graph ratio 8 × unpatchify 3\)"
        ):
            vae_tiling.min_overlap_latent(8, 3)

    @pytest.mark.parametrize(
        ("scale", "patch_size"), [(8.0, 1), (8, 1.0), (True, 1), (8, True), (0, 1), (8, 0), (-8, 1)]
    )
    def test_a_factor_that_is_not_a_positive_int_is_rejected(
        self, scale: object, patch_size: object
    ):
        """Like the TS side (`Number.isInteger`), booleans and values of 0 or less are refused, and
        int is also required as the type.

        TS treats 8.0 as an integer (JS has no distinction between float and int), but on the Python
        side letting 8.0 through would let a float flow on, so it is refused more strictly than on
        the TS side. The message names the breakdown (the graph's ratio, the unpatchify factor).
        """
        expected = rf"not positive integers \(graph ratio {re.escape(repr(scale))}, unpatchify "
        with pytest.raises(ValueError, match=expected):
            vae_tiling.min_overlap_latent(scale, patch_size)

    def test_the_tile_planners_require_the_overlap(self):
        """Forgetting to pass it is a TypeError — no silent planning with the 2.1 value."""
        with pytest.raises(TypeError):
            vae_tiling.plan_tile_axis(60, TILE)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            vae_tiling.plan_tiles(60, 104, TILE)  # type: ignore[call-arg]


class TestPlanTileAxis:
    @pytest.mark.parametrize("extent", sorted(MIRRORED_STARTS))
    def test_starts_match_the_ts_side(self, extent: int):
        assert vae_tiling.plan_tile_axis(extent, TILE, OVERLAP).starts == MIRRORED_STARTS[extent]

    def test_832x480_is_twelve_tiles(self):
        """832×480 (latent 60×104) is 3 × 4 = 12 tiles, with 1.97 times the untiled area (ADR 0118
        decision 2)."""
        plan = vae_tiling.plan_tiles(60, 104, TILE, OVERLAP)

        assert plan.tiles == 12
        assert plan.scale == 8
        # The overlap is 32 − 14 = 18 latents (144 px) for rows and 32 − 24 = 8 latents (64 px =
        # exactly the lower bound) for columns.
        assert plan.rows.blends(plan.scale) == [144, 144]
        assert plan.cols.blends(plan.scale) == [64, 64, 64]
        # The owned regions sum to the full length (covering up to the end).
        assert [plan.rows.region(i) for i in range(3)] == [14, 14, 32]
        assert [plan.cols.region(j) for j in range(4)] == [24, 24, 24, 32]
        assert round(plan.tiles * TILE * TILE / (60 * 104), 2) == 1.97

    def test_480x832_is_the_transpose(self):
        plan = vae_tiling.plan_tiles(104, 60, TILE, OVERLAP)

        assert plan.rows.starts == MIRRORED_STARTS[104]
        assert plan.cols.starts == MIRRORED_STARTS[60]

    @pytest.mark.parametrize("extent", [32, 33, 40, 56, 57, 60, 64, 80, 104, 128, 135, 256])
    def test_invariants_hold_for_every_extent(self, extent: int):
        """The invariants that make the placement one a fixed-shape chunk graph can consume (what
        follows from the rounded evenly spaced placement)."""
        axis = vae_tiling.plan_tile_axis(extent, TILE, OVERLAP)
        span = extent - TILE
        gaps = [second - first for first, second in zip(axis.starts, axis.starts[1:], strict=False)]

        assert axis.starts[0] == 0
        assert axis.starts[-1] == span, "the last tile snaps to the end"
        assert all(TILE - gap >= OVERLAP for gap in gaps)
        assert sum(axis.region(i) for i in range(len(axis.starts))) == extent
        # What rounded even spacing amounts to = the gaps differ by at most one latent (a regression
        # to "fixed stride + snap only the last" opens only the last pair wide and splits).
        assert not gaps or max(gaps) - min(gaps) <= 1, f"spread of the gaps {gaps}"

    @pytest.mark.parametrize("extent", [33, 40, 56, 57, 60, 64, 80, 104, 128, 135, 256])
    def test_tile_count_is_minimal(self, extent: int):
        """The count is the minimum constrained only by the overlap's lower bound (an implementation
        that errs on the safe side makes the tile count jump)."""
        span = extent - TILE

        assert len(vae_tiling.plan_tile_axis(extent, TILE, OVERLAP).starts) == (
            -(-span // (TILE - OVERLAP)) + 1
        )

    def test_extent_shorter_than_the_tile_is_rejected(self):
        """A fixed-shape graph cannot consume a short input — no silent zero padding."""
        with pytest.raises(ValueError, match="tile width"):
            vae_tiling.plan_tile_axis(31, TILE, OVERLAP)

    def test_overlap_at_or_above_the_tile_width_is_rejected(self):
        with pytest.raises(ValueError, match="minimum overlap"):
            vae_tiling.plan_tile_axis(60, TILE, TILE)

    def test_a_pair_outside_the_axis_is_rejected(self):
        with pytest.raises(ValueError, match="blend pair"):
            vae_tiling.plan_tile_axis(TILE, TILE, OVERLAP).blend_at(8, 1)

    def test_meta_carries_the_geometry(self):
        """The fixture's metadata (the fields compared with the TS plan)."""
        assert vae_tiling.plan_tiles(60, 104, TILE, OVERLAP).meta() == {
            "tile": "32",
            "scale": "8",
            "rows_starts": "0,14,28",
            "cols_starts": "0,24,48,72",
            "rows_blend": "144,144",
            "cols_blend": "64,64,64",
        }


class TestBlendIsomorphism:
    """Pin that it is a **verbatim** copy of the upstream `blend_v` / `blend_h` by comparing with
    the real thing.

    `importorskip` is taken inside each test (at module level it would skip the geometry cases too —
    the same judgment as `anima/tests/test_tiling.py`).
    """

    @pytest.mark.parametrize("blend", [1, 3, 8, 64])
    def test_blend_v_matches_upstream_bitwise(self, blend: int):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        upstream = wan_vae.AutoencoderKLWan.blend_v
        generator = torch.Generator().manual_seed(blend)
        a = torch.randn(1, 3, 5, 16, 8, generator=generator)
        b = torch.randn(1, 3, 5, 16, 8, generator=generator)

        mine = vae_tiling.blend_v(a.clone(), b.clone(), blend)
        theirs = upstream(None, a.clone(), b.clone(), blend)

        assert torch.equal(mine, theirs)

    @pytest.mark.parametrize("blend", [1, 3, 8, 64])
    def test_blend_h_matches_upstream_bitwise(self, blend: int):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        upstream = wan_vae.AutoencoderKLWan.blend_h
        generator = torch.Generator().manual_seed(100 + blend)
        a = torch.randn(1, 3, 5, 8, 16, generator=generator)
        b = torch.randn(1, 3, 5, 8, 16, generator=generator)

        mine = vae_tiling.blend_h(a.clone(), b.clone(), blend)
        theirs = upstream(None, a.clone(), b.clone(), blend)

        assert torch.equal(mine, theirs)


def _grid(plan: vae_tiling.TilePlan, make) -> list[list[torch.Tensor]]:
    """A tile array of the plan's shape (`make(i, j)` returns `[1, C, F, s, s]`)."""
    return [
        [make(i, j) for j in range(len(plan.cols.starts))] for i in range(len(plan.rows.starts))
    ]


def _ramp(plan: vae_tiling.TilePlan, axis: str):
    """A tile whose values are the positions inside the tile (row / column numbers) (the values in
    an overlap **differ** between the tiles above and below, left and right).

    With a construction where the same values overlap, a reversed blend direction would stay green
    (ADR 0033 finding 3).
    """
    side = plan.rows.tile * plan.scale
    index = torch.arange(side, dtype=torch.float32)
    plane = index.view(-1, 1).expand(side, side) if axis == "rows" else index.expand(side, side)

    def make(_i: int, _j: int) -> torch.Tensor:
        return plane.reshape(1, 1, 1, side, side).expand(1, 1, 2, side, side).clone()

    return make


class TestAssembleTiles:
    def test_single_tile_is_a_bitwise_identity(self):
        """The degenerate case (one tile) has no blend and its region is the whole — it becomes a
        plain copy of the tile."""
        plan = vae_tiling.plan_tiles(8, 8, tile=8, scale=1, min_overlap=2)
        tile = torch.randn(1, 3, 5, 8, 8)

        out = vae_tiling.assemble_tiles([[tile.clone()]], plan)

        assert torch.equal(out, tile)

    @pytest.mark.parametrize("axis", ["rows", "cols"])
    def test_linear_ramp_is_reconstructed_across_the_seams(self, axis: str):
        """With full length 16, tile 8, starts 0 / 4 / 8 (overlap 4), the slope 0..7 inside a tile
        becomes `0,1,2,3` → all 4 in the middle → `4,5,6,7` at the end after pasting (the linear
        ramp over the overlap width flattens the slope).

        A reversed blend direction, an off-by-one in the spacing, a dropped snap at the end and
        mixed-up owned regions all split here. It also checks that the same formula applies to every
        frame.
        """
        plan = vae_tiling.plan_tiles(16, 16, tile=8, scale=1, min_overlap=2)
        assert plan.rows.starts == (0, 4, 8)

        out = vae_tiling.assemble_tiles(_grid(plan, _ramp(plan, axis)), plan)

        expected = torch.tensor([0.0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7])
        for frame in range(2):
            line = out[0, 0, frame, :, 0] if axis == "rows" else out[0, 0, frame, 0, :]
            assert torch.equal(line, expected), f"frame {frame}"

    def test_corners_blend_vertically_first(self):
        """A corner (the region where four tiles overlap) is blended vertically, then horizontally
        (the same as the upstream).

        With full length 6, tile 4, starts 0 / 2 (blend width 2), and constant tiles a=1 (top
        left) / b=2 / c=3 / d=4 (bottom right). Follow the second row of the bottom-right tile
        (row 3 of the output) by hand:

        - The bottom-left tile is already blended vertically with the one above (a): row 1 =
          (a + c) / 2 = 2.
        - The top-right tile is already blended horizontally with the one to its left (a): row 3 =
          [1, 1.5, 2, 2].
        - The bottom-right tile (vertical first): row 1 = 0.5·row 3 of the top right + 0.5·d =
          [2.5, 2.75, 3, 3] → horizontally, column 0 = the bottom left's (1, 2) = 2, column 1 =
          0.5·2 + 0.5·2.75 = 2.375.

        Row 3 of the output = the bottom left's owned part [2, 2] + the bottom right's owned part
        [2, 2.375, 3, 3]. Swapping to horizontal → vertical order gives [2, 2, 1.5, 2.25, 3, 3].
        """
        plan = vae_tiling.plan_tiles(6, 6, tile=4, scale=1, min_overlap=2)
        assert plan.rows.starts == (0, 2)
        values = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}

        out = vae_tiling.assemble_tiles(
            _grid(plan, lambda i, j: torch.full((1, 1, 1, 4, 4), values[(i, j)])), plan
        )

        assert out[0, 0, 0, 3].tolist() == [2.0, 2.0, 2.0, 2.375, 3.0, 3.0]

    def test_the_far_edge_is_covered(self):
        """It covers up to the end (merely truncating to the stride width would leave the total
        short by `extent − tile`)."""
        plan = vae_tiling.plan_tiles(16, 26, tile=8, scale=2, min_overlap=2)

        out = vae_tiling.assemble_tiles(
            _grid(plan, lambda _i, _j: torch.full((1, 3, 2, 16, 16), 7.0)), plan
        )

        assert list(out.shape) == [1, 3, 2, 32, 52]
        assert torch.equal(out, torch.full_like(out, 7.0))

    def test_a_grid_that_does_not_match_the_plan_is_rejected(self):
        plan = vae_tiling.plan_tiles(16, 16, tile=8, scale=1, min_overlap=2)

        with pytest.raises(ValueError, match="differ from the plan's"):
            vae_tiling.assemble_tiles([[torch.zeros(1, 1, 1, 8, 8)]], plan)


class TestFixtureCases:
    def test_acceptance_differs_from_the_band_in_every_axis(self):
        """The acceptance case differs from the band's decision case in the latent (seed), the
        chunk boundaries and the tile positions (vertical and horizontal) (ADR 0118 addendum
        2026-10-02). A portrait shape brings a row / column mix-up out in the values."""
        cases = {case.role: case for case in vae_tiling.FIXTURE_CASES}
        band, accept = cases["band"], cases["accept"]

        assert (band.height, band.width, band.chunks) == (60, 104, 9)
        assert band.full and not accept.full
        assert accept.seed != band.seed
        assert accept.chunks != band.chunks
        assert (accept.height, accept.width) == (band.width, band.height)


@pytest.fixture(scope="module")
def vae(wan_snapshot):
    """The VAE of the pinned revision (CPU f32, no rounding — the identity claim does not depend on
    the weight values)."""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


class TestDegenerateTile:
    def test_one_tile_is_the_untiled_chunk_loop_bitwise(self, vae):
        """When the latent is exactly one tile (8×8, 3 chunks), the tile reference is
        **bit-identical** to the upstream untiled `_decode`'s chunk loop (before clamping) (the
        Python side of the degenerate case of ADR 0033 decision 4).

        It checks that the values do not change, including the difference in how the post-quant
        1×1×1 conv is applied (the upstream `tiled_decode` per frame, the untiled `_decode` once
        over all frames).
        """
        generator = torch.Generator().manual_seed(11)
        latents = torch.randn(1, 16, 3, 8, 8, generator=generator)
        plan = vae_tiling.plan_tiles(8, 8, tile=8, min_overlap=2)
        assert plan.tiles == 1

        with torch.no_grad():
            tiled = vae_tiling.tiled_decode_unclamped(vae, latents, plan)
            untiled = vae_patch.reference_decode_unclamped(vae, latents)

        assert torch.equal(tiled, untiled)
