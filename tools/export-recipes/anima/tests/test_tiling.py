"""`anima/tiling.py` の幾何とブレンドの約束事（実重み不要分）。

ここで固定するのは、壊れても例外が出ず**数だけが静かにずれる**側:

- 丸め等間隔スナップ配置の不変条件（末尾が `extent − tile` / 重なりが下限以上 /
  本数が解析式どおり / 間隔のばらつきが高々 1 latent）
- 開始位置が TS 側（`packages/models/src/anima/tiling.ts`）と一致すること
- ブレンド式が上流（`AutoencoderKLQwenImage.blend_v` / `blend_h`）と**ビット一致**
- 貼り付けの担当領域（間隔ぶんで切り詰めると末端が欠ける — 上流からの意図的逸脱の帰結）

実重みでの参照フィクスチャ生成は手動（モジュール doc）。
"""

from __future__ import annotations

import pytest
import torch

from anima import tiling as anima_tiling

#: TS 側（`packages/models/src/anima/tiling.ts` の `planTileAxis`）と一致する開始位置の実測
#: （辺 px → latent の開始位置列）。**両実装の一致を機械で突き合わせる経路が無い**
#: （フィクスチャ生成は手動）ので、代表 4 辺を値で凍結して鏡像のずれを検出する。TS 側は
#: 受理集合の全 97 辺を同じ値で凍結している（`packages/models/tests/anima_tiling_test.ts` の
#: `AXIS_STARTS`）ので、この 4 行はその表の部分集合である。
#:
#: 1456px は旧規則で 60 本に跳ねて入口で拒否していた辺、1824px は P-3 の測定対象。**1872px は
#: `i·span/(本数−1)` がちょうど半分になる辺**（span 170 / 5 本 → 42.5 と 127.5）で、
#: `plan_tile_axis` の整数式が組み込み `round`（偶数丸め）へ退行すると (0, 42, 85, 128, 170) に
#: なる — 他の 3 辺は半端が出ないので、この 1 行だけがその退行を捕まえる。
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
        """1024px（latent 128）は 3 タイル・重なり latent 32（= sample 256）— 旧規則と同配置。"""
        axis = anima_tiling.plan_tile_axis(128, 64, 8)

        assert axis.starts == (0, 32, 64)
        assert axis.blends(8) == [256, 256]

    def test_single_tile_degenerates_without_blending(self):
        """`extent == tile`（512px）は 1 枚・ブレンド対なし — 非タイル経路とビット同一の前提。"""
        axis = anima_tiling.plan_tile_axis(64, 64, 8)

        assert axis.starts == (0,)
        assert axis.blends(8) == []

    @pytest.mark.parametrize(
        "extent", [64, 80, 96, 112, 128, 160, 182, 192, 228, 242, 256, 384, 512]
    )
    def test_invariants_hold_for_every_extent(self, extent):
        """固定形の decoder が食える配置であることの不変条件（丸め等間隔配置の帰結）。"""
        axis = anima_tiling.plan_tile_axis(extent, 64, 8)
        span = extent - 64
        gaps = [second - first for first, second in zip(axis.starts, axis.starts[1:], strict=False)]

        assert axis.starts[0] == 0
        assert axis.starts[-1] == span, "最後のタイルは末端へスナップする"
        assert all(64 - gap >= 8 for gap in gaps), "どの対も重なりが下限以上"
        assert axis.blends(8) == [(64 - gap) * 8 for gap in gaps]
        # 丸め等間隔の実体 = 間隔（したがってブレンド幅）の差は高々 1 latent。上流同型の
        # 「固定 stride + 末尾だけスナップ」に退行すると最後の対だけ大きく開いて割れる。
        assert not gaps or max(gaps) - min(gaps) <= 1, f"間隔のばらつき {gaps}"

    @pytest.mark.parametrize("extent", [80, 96, 112, 128, 160, 182, 192, 228, 242, 256, 384, 512])
    def test_tile_count_is_minimal(self, extent):
        """本数は「重なりの下限だけを制約にした最小」= `ceil(span / (tile − 下限)) + 1`。

        これが無いと「常に間隔 1」のような安全側の実装が緑のまま通り、タイル数が
        跳ね上がる（時間の冗長がそのまま効く）。
        """
        axis = anima_tiling.plan_tile_axis(extent, 64, 8)
        span = extent - 64

        assert len(axis.starts) == -(-span // (64 - 8)) + 1

    def test_minimum_overlap_is_honored(self):
        """重なりの下限を上げると本数が増える（下限が本当に効いている）。"""
        loose = anima_tiling.plan_tile_axis(128, 64, 8)
        tight = anima_tiling.plan_tile_axis(128, 64, 48)

        assert len(tight.starts) > len(loose.starts)
        assert all(tight.blend_at(1, index) >= 48 for index in range(1, len(tight.starts)))

    def test_extent_shorter_than_the_tile_is_rejected(self):
        """固定形の decoder は短い入力を食えない — 黙ってゼロ埋めしない。"""
        with pytest.raises(ValueError, match="タイル幅"):
            anima_tiling.plan_tile_axis(32, 64, 8)

    def test_overlap_at_or_above_the_tile_width_is_rejected(self):
        with pytest.raises(ValueError, match="最小の重なり"):
            anima_tiling.plan_tile_axis(128, 64, 64)

    def test_a_pair_outside_the_axis_is_rejected(self):
        """縮退（1 枚）に対を求めるのは幾何の取り違え — 0 を返して黙って素通ししない。"""
        with pytest.raises(ValueError, match="ブレンド対"):
            anima_tiling.plan_tile_axis(64, 64, 8).blend_at(8, 1)


class TestPlanTiling:
    def test_axes_are_planned_independently(self):
        """H = W でなくても成立する一般形（軸ごとに独立）。"""
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
    """上流の `blend_v` / `blend_h` の**逐語移植**であることを本物との突合で固定する。

    `enable_tiling` を使わない（走査形が違う）ぶん、式の同型はここでしか担保できない。
    """

    qwenimage = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")

    @pytest.mark.parametrize("blend", [1, 3, 8])
    def test_blend_v_matches_upstream_bitwise(self, blend):
        upstream = self.qwenimage.AutoencoderKLQwenImage.blend_v
        a = torch.randn(1, 3, 1, 8, 8)
        b = torch.randn(1, 3, 1, 8, 8)

        mine = anima_tiling.blend_v(a.clone(), b.clone(), blend)
        theirs = upstream(None, a.clone(), b.clone(), blend)

        assert torch.equal(mine, theirs)

    @pytest.mark.parametrize("blend", [1, 3, 8])
    def test_blend_h_matches_upstream_bitwise(self, blend):
        upstream = self.qwenimage.AutoencoderKLQwenImage.blend_h
        a = torch.randn(1, 3, 1, 8, 8)
        b = torch.randn(1, 3, 1, 8, 8)

        mine = anima_tiling.blend_h(a.clone(), b.clone(), blend)
        theirs = upstream(None, a.clone(), b.clone(), blend)

        assert torch.equal(mine, theirs)


def _ramp_decoder(axis: str):
    """タイル内の位置に依る合成 decoder（重なりの値がタイルごとに**違う**ものになる）。

    latent の中身に依らず `value = タイル内の行（列）番号` を返す。これで重なり領域の値が
    上下（左右）のタイルで別の数になり、**ブレンドの向き**と**担当領域の割り当て**が
    数値として現れる（同じ値が重なる作りだと向きを反転しても緑のまま通る）。
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
        """縮退（1 枚）はブレンド無し・領域が全体 — decode 出力の素の写しになる。"""
        geometry = anima_tiling.plan_tiling((1, 2, 8, 8), tile=8, min_overlap=2, scale=1)
        z = torch.randn(1, 2, 1, 8, 8)
        decoded = torch.randn(1, 3, 1, 8, 8)

        out = anima_tiling.tiled_decode(lambda _tile: decoded.clone(), z, geometry)

        assert torch.equal(out, decoded)

    @pytest.mark.parametrize("axis", ["rows", "cols"])
    def test_linear_ramp_is_reconstructed_across_the_seams(self, axis):
        """位置依存の decoder に対する解析解と一致する。

        extent 16 / tile 8 / 間隔 4（重なり 4）で開始位置は 0,4,8。タイル内の値が
        `0..7` の傾斜のとき、貼り合わせ後は `0,1,2,3` → 中間は全て 4 → 末尾が `4,5,6,7`
        になる（重なり幅の線形ランプが傾斜を平坦へ潰す）。ブレンドの向き反転・間隔の
        off-by-one・末端タイルのスナップ落とし・担当領域の取り違えは全てここで割れる。
        """
        geometry = anima_tiling.plan_tiling((1, 2, 16, 16), tile=8, min_overlap=2, scale=1)
        assert geometry.rows.starts == (0, 4, 8)

        out = anima_tiling.tiled_decode(_ramp_decoder(axis), torch.zeros(1, 2, 1, 16, 16), geometry)

        expected = torch.tensor([0.0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7])
        line = out[0, 0, 0, :, 0] if axis == "rows" else out[0, 0, 0, 0, :]
        assert torch.equal(line, expected)

    def test_the_far_edge_is_covered(self):
        """末端まで覆う（間隔ぶんで切り詰めるだけだと総和が `extent − tile` で欠ける）。"""
        geometry = anima_tiling.plan_tiling((1, 2, 16, 16), tile=8, min_overlap=2, scale=1)

        out = anima_tiling.tiled_decode(
            lambda _tile: torch.full((1, 1, 1, 8, 8), 7.0), torch.zeros(1, 2, 1, 16, 16), geometry
        )

        assert list(out.shape) == [1, 1, 1, 16, 16]
        assert torch.equal(out, torch.full((1, 1, 1, 16, 16), 7.0))
