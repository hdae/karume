"""rope の**軸別素表からの再構成**が非正方（H'≠W'）でも上流の表と一致すること（#23）。

## なぜ非正方で見るのか

ホスト（TS）は `[1,1,S,head_dim]` の rope 表を「軸別の素表を `[t,h,w,t,h,w]` に並べる」
だけで作る（ADR 0034）。**正方ではこの並べ方の h ↔ w 取り違えが原理的に検出できない** —
Anima の `rope_scale` は h と w が同値なので `cos_h` と `cos_w` がバイト単位で一致し、
H'=W' なら表そのものが同じ値になる（ADR 0034 の検出限界 1）。

ここでは①**H'≠W'** の格子で②**h と w の周波数が違う** rope（`rope_scale` の h と w を
別の値にした合成構成）を使い、素表からの再構成が上流 `CosmosRotaryPosEmbed` の出力と
`torch.equal` で一致することを固定する。実重みは要らない（rope はパラメータもバッファも
持たない純計算）。TS 側の鏡像（`ropeTables`）は
`packages/runtime/tests/e2e_anima_nonsquare_test.ts` が
焼いたフィクスチャとの Uint32 完全一致で押さえる。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anima.patch import dit_rope_base_tables, dit_rope_tables

transformer_cosmos = pytest.importorskip("diffusers.models.transformers.transformer_cosmos")
CosmosRotaryPosEmbed = transformer_cosmos.CosmosRotaryPosEmbed

#: 合成 rope の構成。`rope_scale` の **h と w を別の値**にするのが要点（Anima 実機は
#: `[1.0, 4.0, 4.0]` で h と w が同値 = 取り違えが検出できない構成）。
HIDDEN_SIZE = 48
MAX_SIZE = (16, 16, 16)
PATCH_SIZE = (1, 2, 2)
ROPE_SCALE = (2.0, 1.0, 3.0)
#: 非正方の latent（トークン格子 6×4）。**縦横が違う**ことがこのテストの前提。
LATENT_HEIGHT = 12
LATENT_WIDTH = 8


class _RopeOnly(nn.Module):
    """`dit_rope_tables` / `dit_rope_base_tables` が触るのは `.rope` だけ。"""

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
    """素表から `[S, head_dim]` を組む（TS 側 `ropeTables` の鏡像）。

    1 行は `[t, h, w]` を 2 回並べたブロック連結（上流の `cat([emb_t, emb_h, emb_w] * 2)`）で、
    トークン添字は `h·W' + w`。t は画像なので常に位置 0。
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
        """この合成構成では `cos_h != cos_w`（= 取り違えが数に出る）。

        実機（Anima）ではここが**一致する**ため、同じ検査が空虚になる — 非正方の
        フィクスチャ突合が要る理由そのもの（ADR 0034 の検出限界 1）。
        """
        tables = dit_rope_base_tables(model)

        assert not torch.equal(tables["cos_h"], tables["cos_w"])

    def test_rebuilding_from_the_axis_tables_matches_upstream(self, model):
        """MUST: ビット一致（`torch.equal`）。素表の写しなので誤差の余地が無い。"""
        tables = dit_rope_base_tables(model)
        cos, sin = dit_rope_tables(model, LATENT_HEIGHT, LATENT_WIDTH)
        rows = LATENT_HEIGHT // PATCH_SIZE[1]
        cols = LATENT_WIDTH // PATCH_SIZE[2]

        # 1 行の幅 = `[t,h,w]` を 2 回並べたもの = `hidden_size`（= attention の head_dim）。
        assert list(cos.shape) == [1, 1, rows * cols, HIDDEN_SIZE]
        assert torch.equal(rebuild_from_base(tables, "cos", rows, cols), cos[0, 0])
        assert torch.equal(rebuild_from_base(tables, "sin", rows, cols), sin[0, 0])

    def test_swapping_the_two_axes_breaks_the_match(self, model):
        """恒真化の門: 上の一致が「何を検出するのか」を注入で示す（h と w の取り違え）。"""
        tables = dit_rope_base_tables(model)
        swapped = dict(tables)
        swapped["cos_h"], swapped["cos_w"] = tables["cos_w"], tables["cos_h"]
        cos, _ = dit_rope_tables(model, LATENT_HEIGHT, LATENT_WIDTH)
        rows = LATENT_HEIGHT // PATCH_SIZE[1]
        cols = LATENT_WIDTH // PATCH_SIZE[2]

        assert not torch.equal(rebuild_from_base(swapped, "cos", rows, cols), cos[0, 0])

    def test_transposing_the_grid_breaks_the_match(self, model):
        """同じく恒真化の門: 格子の縦横を入れ替えると別の表になる（要素数は同じ）。"""
        tables = dit_rope_base_tables(model)
        cos, _ = dit_rope_tables(model, LATENT_HEIGHT, LATENT_WIDTH)
        rows = LATENT_HEIGHT // PATCH_SIZE[1]
        cols = LATENT_WIDTH // PATCH_SIZE[2]

        transposed = rebuild_from_base(tables, "cos", cols, rows)

        assert transposed.shape == cos[0, 0].shape
        assert not torch.equal(transposed, cos[0, 0])
