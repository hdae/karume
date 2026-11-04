"""Rebuilding rope **from the per-axis base tables** matches upstream's table even for a
non-square grid (H'≠W') (#23).

## Why look at a non-square grid

The host (TS) builds the `[1,1,S,head_dim]` rope table only by "laying out the per-axis base
tables as `[t,h,w,t,h,w]`" (ADR 0034). **On a square grid an h ↔ w mix-up in this layout is
undetectable in principle** — Anima's `rope_scale` has equal h and w, so `cos_h` and `cos_w`
match byte for byte, and with H'=W' the table itself holds the same values (detection limit 1 of
ADR 0034).

Here a rope with (1) an **H'≠W'** grid and (2) **different h and w frequencies** (a synthetic
configuration whose `rope_scale` has different h and w) pins that the rebuild from the base
tables matches upstream `CosmosRotaryPosEmbed`'s output under `torch.equal`. No real weights are
needed (rope is a pure computation with no parameters or buffers). The TS-side mirror
(`ropeTables`) is covered by `packages/models/tests/anima_dit_tokens_test.ts`, with the
`[t,h,w,t,h,w]` layout on synthetic base tables and the rejection at the ceiling (the comparison
with a baked fixture has **not been ported to this repository yet**).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anima.patch import dit_rope_base_tables, dit_rope_tables

transformer_cosmos = pytest.importorskip("diffusers.models.transformers.transformer_cosmos")
CosmosRotaryPosEmbed = transformer_cosmos.CosmosRotaryPosEmbed

#: The synthetic rope configuration. The point is **different h and w** in `rope_scale` (the real
#: Anima has `[1.0, 4.0, 4.0]`, equal h and w = a configuration where a mix-up is undetectable).
HIDDEN_SIZE = 48
MAX_SIZE = (16, 16, 16)
PATCH_SIZE = (1, 2, 2)
ROPE_SCALE = (2.0, 1.0, 3.0)
#: A non-square latent (token grid 6×4). **Different height and width** is this test's premise.
LATENT_HEIGHT = 12
LATENT_WIDTH = 8


class _RopeOnly(nn.Module):
    """`dit_rope_tables` / `dit_rope_base_tables` touch only `.rope`."""

    def __init__(self) -> None:
        super().__init__()
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=HIDDEN_SIZE,
            max_size=MAX_SIZE,
            patch_size=PATCH_SIZE,
            rope_scale=ROPE_SCALE,
        )


def rebuild_from_base(
    tables: dict[str, torch.Tensor], kind: str, rows: int, cols: int
) -> torch.Tensor:
    """Build `[S, head_dim]` from the base tables (the mirror of the TS side's `ropeTables`).

    One row is the block concatenation of `[t, h, w]` laid out twice (upstream's
    `cat([emb_t, emb_h, emb_w] * 2)`), and the token index is `h·W' + w`. t is always position 0
    for an image.
    """
    axis_t, axis_h, axis_w = (tables[f"{kind}_{axis}"] for axis in ("t", "h", "w"))
    lines = []
    for row in range(rows):
        for col in range(cols):
            blocks = [axis_t[0], axis_h[row], axis_w[col]]
            lines.append(torch.cat(blocks * 2))
    return torch.stack(lines)


@pytest.fixture(scope="module")
def model() -> nn.Module:
    return _RopeOnly()


class TestNonSquareRopeParity:
    def test_the_axis_tables_differ_so_a_swap_is_detectable(self, model):
        """In this synthetic configuration `cos_h != cos_w` (= a mix-up shows in the numbers).

        On the real model (Anima) these **match**, so the same check is vacuous — the very
        reason a non-square fixture comparison is needed (detection limit 1 of ADR 0034).
        """
        tables = dit_rope_base_tables(model)

        assert not torch.equal(tables["cos_h"], tables["cos_w"])

    def test_rebuilding_from_the_axis_tables_matches_upstream(self, model):
        """MUST: bit-identical (`torch.equal`). It copies the base tables, so there is no room for
        error."""
        tables = dit_rope_base_tables(model)
        cos, sin = dit_rope_tables(model, LATENT_HEIGHT, LATENT_WIDTH)
        rows = LATENT_HEIGHT // PATCH_SIZE[1]
        cols = LATENT_WIDTH // PATCH_SIZE[2]

        # The width of one row = `[t,h,w]` laid out twice = `hidden_size` (= attention's
        # head_dim).
        assert list(cos.shape) == [1, 1, rows * cols, HIDDEN_SIZE]
        assert torch.equal(rebuild_from_base(tables, "cos", rows, cols), cos[0, 0])
        assert torch.equal(rebuild_from_base(tables, "sin", rows, cols), sin[0, 0])

    def test_swapping_the_two_axes_breaks_the_match(self, model):
        """Vacuity check: injection shows "what the match above detects" (an h and w mix-up)."""
        tables = dit_rope_base_tables(model)
        swapped = dict(tables)
        swapped["cos_h"], swapped["cos_w"] = tables["cos_w"], tables["cos_h"]
        cos, _ = dit_rope_tables(model, LATENT_HEIGHT, LATENT_WIDTH)
        rows = LATENT_HEIGHT // PATCH_SIZE[1]
        cols = LATENT_WIDTH // PATCH_SIZE[2]

        assert not torch.equal(rebuild_from_base(swapped, "cos", rows, cols), cos[0, 0])

    def test_transposing_the_grid_breaks_the_match(self, model):
        """Also a vacuity check: swapping the grid's height and width gives another table (with
        the same element count)."""
        tables = dit_rope_base_tables(model)
        cos, _ = dit_rope_tables(model, LATENT_HEIGHT, LATENT_WIDTH)
        rows = LATENT_HEIGHT // PATCH_SIZE[1]
        cols = LATENT_WIDTH // PATCH_SIZE[2]

        transposed = rebuild_from_base(tables, "cos", cols, rows)

        assert transposed.shape == cos[0, 0].shape
        assert not torch.equal(transposed, cos[0, 0])
