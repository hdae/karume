"""The contracts of `anima/tiling.py`'s geometry and blending (the part that needs no real
weights).

Pinned here is the side where a breakage raises nothing and **only the numbers silently drift**:

- the invariants of the rounded equal-spacing snap placement (the last start is `extent − tile` /
  every overlap is at least the floor / the count follows the closed form / the spacing varies by
  at most 1 latent)
- the start positions match the TS side (`packages/models/src/anima/tiling.ts`)
- the blend formulas are **bit-identical** to upstream (`AutoencoderKLQwenImage.blend_v` /
  `blend_h`)
- the regions each tile pastes (truncating to the spacing loses the end — a consequence of the
  deliberate deviation from upstream)

Generating the reference fixture with real weights is manual (see the module doc).
"""

from __future__ import annotations

import pytest
import torch

from anima import tiling as anima_tiling

#: Measured start positions that match the TS side (`planTileAxis` in
#: `packages/models/src/anima/tiling.ts`) (side px → latent start positions). **There is no path
#: that compares the two implementations mechanically** (fixture generation is manual), so four
#: representative sides are frozen by value to detect the mirror drifting. The TS side freezes all
#: 97 sides of the accepted set with the same values (`AXIS_STARTS` in
#: `packages/models/tests/anima_tiling_test.ts`), so these four rows are a subset of that table.
#:
#: 1456px is the side that jumped to 60 tiles under the old rule and was rejected at the entry;
#: 1824px is P-3's measurement target. **1872px is the side where `i·span/(count−1)` lands exactly
#: on a half** (span 170 / 5 tiles → 42.5 and 127.5): if `plan_tile_axis`'s integer formula
#: regressed to the built-in `round` (half to even) it would give (0, 42, 85, 128, 170) — the
#: other three sides produce no halves, so this row alone catches that regression.
MIRRORED_STARTS = {
    1024: (0, 32, 64),
    1456: (0, 39, 79, 118),
    1824: (0, 55, 109, 164),
    1872: (0, 43, 85, 128, 170),
}


class TestPlanTileAxis:
    @pytest.mark.parametrize("side", sorted(MIRRORED_STARTS))
    def test_starts_match_the_ts_side(self, side):
        axis = anima_tiling.plan_tile_axis(side // anima_tiling.SPATIAL_COMPRESSION, 64, 8)

        assert axis.starts == MIRRORED_STARTS[side]

    def test_1024_latent_is_three_tiles_blended_over_256_sample_px(self):
        """1024px (latent 128) is 3 tiles with a latent overlap of 32 (= 256 in sample space) —
        the same placement as the old rule."""
        axis = anima_tiling.plan_tile_axis(128, 64, 8)

        assert axis.starts == (0, 32, 64)
        assert axis.blends(8) == [256, 256]

    def test_single_tile_degenerates_without_blending(self):
        """`extent == tile` (512px) is one tile with no blend pair — the premise of being
        bit-identical to the untiled path."""
        axis = anima_tiling.plan_tile_axis(64, 64, 8)

        assert axis.starts == (0,)
        assert axis.blends(8) == []

    @pytest.mark.parametrize(
        "extent", [64, 80, 96, 112, 128, 160, 182, 192, 228, 242, 256, 384, 512]
    )
    def test_invariants_hold_for_every_extent(self, extent):
        """The invariants of a placement a fixed-shape decoder can take (consequences of the
        rounded equal-spacing placement)."""
        axis = anima_tiling.plan_tile_axis(extent, 64, 8)
        span = extent - 64
        gaps = [second - first for first, second in zip(axis.starts, axis.starts[1:], strict=False)]

        assert axis.starts[0] == 0
        assert axis.starts[-1] == span, "the last tile snaps to the end"
        assert all(64 - gap >= 8 for gap in gaps), "every pair overlaps at least the floor"
        assert axis.blends(8) == [(64 - gap) * 8 for gap in gaps]
        # What rounded equal spacing means = the spacings (hence the blend widths) differ by at
        # most 1 latent. A regression to upstream's "fixed stride + snap only the last" would
        # open the last pair wide and break this.
        assert not gaps or max(gaps) - min(gaps) <= 1, f"spacing varies: {gaps}"

    @pytest.mark.parametrize("extent", [80, 96, 112, 128, 160, 182, 192, 228, 242, 256, 384, 512])
    def test_tile_count_is_minimal(self, extent):
        """The count is "the minimum constrained only by the overlap floor" =
        `ceil(span / (tile − floor)) + 1`.

        Without this, an over-safe implementation such as "always a spacing of 1" would stay
        green while the tile count soared (the redundancy costs time directly).
        """
        axis = anima_tiling.plan_tile_axis(extent, 64, 8)
        span = extent - 64

        assert len(axis.starts) == -(-span // (64 - 8)) + 1

    def test_minimum_overlap_is_honored(self):
        """Raising the overlap floor increases the count (the floor really takes effect)."""
        loose = anima_tiling.plan_tile_axis(128, 64, 8)
        tight = anima_tiling.plan_tile_axis(128, 64, 48)

        assert len(tight.starts) > len(loose.starts)
        assert all(tight.blend_at(1, index) >= 48 for index in range(1, len(tight.starts)))

    def test_extent_shorter_than_the_tile_is_rejected(self):
        """A fixed-shape decoder cannot take a short input — no silent zero padding."""
        with pytest.raises(ValueError, match="tile width"):
            anima_tiling.plan_tile_axis(32, 64, 8)

    def test_overlap_at_or_above_the_tile_width_is_rejected(self):
        with pytest.raises(ValueError, match="minimum overlap"):
            anima_tiling.plan_tile_axis(128, 64, 64)

    def test_a_pair_outside_the_axis_is_rejected(self):
        """Asking a degenerate axis (one tile) for a pair is a geometry mix-up — it does not
        silently return 0."""
        with pytest.raises(ValueError, match="blend pair"):
            anima_tiling.plan_tile_axis(64, 64, 8).blend_at(8, 1)


class TestPlanTiling:
    def test_axes_are_planned_independently(self):
        """The general form, valid without H = W (each axis independent)."""
        geometry = anima_tiling.plan_tiling((1, 16, 128, 64))

        assert len(geometry.rows.starts) == 3
        assert len(geometry.cols.starts) == 1
        assert geometry.tiles == 3
        assert geometry.channels == 16

    def test_rank_and_batch_are_checked(self):
        with pytest.raises(ValueError, match=r"\[1,C,H,W\]"):
            anima_tiling.plan_tiling((2, 16, 128, 128))


class TestGeometryMeta:
    def test_meta_carries_the_geometry_the_ts_side_compares_against(self):
        meta = anima_tiling.geometry_meta(anima_tiling.plan_tiling((1, 16, 128, 128)))

        assert meta["tiles"] == 9
        assert meta["scale"] == 8
        assert meta["rows"] == {
            "extent": 128,
            "tile": 64,
            "starts": [0, 32, 64],
            "blend_sample": [256, 256],
        }
        assert meta["cols"] == meta["rows"]


class TestBlendIsomorphism:
    """Pin, by comparing against the real ones, that these are **verbatim ports** of upstream's
    `blend_v` / `blend_h`.

    Since `enable_tiling` is not used (its scan differs), the formulas' identity can only be
    guaranteed here.

    `importorskip` is called **inside each test** — in the class body (= at module import)
    `Skipped(allow_module_level=True)` would skip the whole module, and even the geometry cases
    that need no diffusers would never run in CI (the default sync does not install diffusers).
    """

    @pytest.mark.parametrize("blend", [1, 3, 8])
    def test_blend_v_matches_upstream_bitwise(self, blend):
        qwenimage = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")
        upstream = qwenimage.AutoencoderKLQwenImage.blend_v
        a = torch.randn(1, 3, 1, 8, 8)
        b = torch.randn(1, 3, 1, 8, 8)

        mine = anima_tiling.blend_v(a.clone(), b.clone(), blend)
        theirs = upstream(None, a.clone(), b.clone(), blend)

        assert torch.equal(mine, theirs)

    @pytest.mark.parametrize("blend", [1, 3, 8])
    def test_blend_h_matches_upstream_bitwise(self, blend):
        qwenimage = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")
        upstream = qwenimage.AutoencoderKLQwenImage.blend_h
        a = torch.randn(1, 3, 1, 8, 8)
        b = torch.randn(1, 3, 1, 8, 8)

        mine = anima_tiling.blend_h(a.clone(), b.clone(), blend)
        theirs = upstream(None, a.clone(), b.clone(), blend)

        assert torch.equal(mine, theirs)


def _ramp_decoder(axis: str):
    """A synthetic decoder that depends on the position inside the tile (so the overlapping
    values **differ** per tile).

    Regardless of the latent's contents it returns `value = the row (column) number inside the
    tile`. That makes the values of an overlap region differ between the upper and lower (left and
    right) tiles, so **the blend direction** and **the region assignment** show up as numbers (with
    equal values overlapping, a reversed direction would stay green).
    """

    def decode(tile: torch.Tensor) -> torch.Tensor:
        _, _, frames, height, width = tile.shape
        index = torch.arange(height if axis == "rows" else width, dtype=torch.float32)
        plane = (
            index.view(-1, 1).expand(height, width)
            if axis == "rows"
            else index.view(1, -1).expand(height, width)
        )
        return plane.reshape(1, 1, 1, height, width).expand(1, 1, frames, height, width).clone()

    return decode


class TestTiledDecode:
    def test_single_tile_is_a_bitwise_identity(self):
        """The degenerate case (one tile) has no blending and one region covering everything — a
        plain copy of the decode output."""
        geometry = anima_tiling.plan_tiling((1, 2, 8, 8), tile=8, min_overlap=2, scale=1)
        z = torch.randn(1, 2, 1, 8, 8)
        decoded = torch.randn(1, 3, 1, 8, 8)

        out = anima_tiling.tiled_decode(lambda _tile: decoded.clone(), z, geometry)

        assert torch.equal(out, decoded)

    @pytest.mark.parametrize("axis", ["rows", "cols"])
    def test_linear_ramp_is_reconstructed_across_the_seams(self, axis):
        """It matches the closed-form result for a position-dependent decoder.

        With extent 16 / tile 8 / spacing 4 (overlap 4) the start positions are 0,4,8. When the
        values inside a tile are the slope `0..7`, the stitched result is `0,1,2,3` → all 4 in the
        middle → `4,5,6,7` at the end (the linear ramp over the overlap flattens the slope). A
        reversed blend direction, an off-by-one spacing, a missing snap of the last tile and a
        mixed-up region assignment all break here.
        """
        geometry = anima_tiling.plan_tiling((1, 2, 16, 16), tile=8, min_overlap=2, scale=1)
        assert geometry.rows.starts == (0, 4, 8)

        out = anima_tiling.tiled_decode(_ramp_decoder(axis), torch.zeros(1, 2, 1, 16, 16), geometry)

        expected = torch.tensor([0.0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7])
        line = out[0, 0, 0, :, 0] if axis == "rows" else out[0, 0, 0, 0, :]
        assert torch.equal(line, expected)

    def test_the_far_edge_is_covered(self):
        """It covers up to the far end (merely truncating to the spacing would sum to
        `extent − tile` and leave a gap)."""
        geometry = anima_tiling.plan_tiling((1, 2, 16, 16), tile=8, min_overlap=2, scale=1)

        out = anima_tiling.tiled_decode(
            lambda _tile: torch.full((1, 1, 1, 8, 8), 7.0), torch.zeros(1, 2, 1, 16, 16), geometry
        )

        assert list(out.shape) == [1, 1, 1, 16, 16]
        assert torch.equal(out, torch.full((1, 1, 1, 16, 16), 7.0))
