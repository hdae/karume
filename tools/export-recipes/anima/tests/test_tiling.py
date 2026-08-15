"""`anima/tiling.py` の幾何とブレンドの約束事（実重み不要分）。

ここで固定するのは、壊れても例外が出ず**数だけが静かにずれる**側:

- 等間隔スナップ配置の不変条件（`(extent − tile) % stride == 0` / 末尾が `extent − tile` /
  重なりが下限以上 / 本数が最小）
- ブレンド式が上流（`AutoencoderKLQwenImage.blend_v` / `blend_h`）と**ビット一致**
- 貼り付けの担当領域（stride 幅で切り詰めると末端が欠ける — 上流からの意図的逸脱の帰結）

実重みでの参照フィクスチャ生成は手動（モジュール doc）。
"""

from __future__ import annotations

import pytest
import torch

from anima import tiling as anima_tiling


class TestPlanTileAxis:
    def test_1024_latent_is_three_tiles_of_stride_32(self):
        """1024px（latent 128）は 3 タイル・stride 32・重なり latent 32（= sample 256）。"""
        axis = anima_tiling.plan_tile_axis(128, 64, 8)

        assert axis.starts == (0, 32, 64)
        assert axis.stride == 32
        assert axis.blend(8) == 256

    def test_single_tile_degenerates_without_blending(self):
        """`extent == tile`（512px）は 1 枚・ブレンド幅 0 — 非タイル経路とビット同一の前提。"""
        axis = anima_tiling.plan_tile_axis(64, 64, 8)

        assert axis.starts == (0,)
        assert axis.stride == 64
        assert axis.blend(8) == 0

    @pytest.mark.parametrize("extent", [64, 80, 96, 112, 128, 160, 192, 256, 384, 512])
    def test_invariants_hold_for_every_extent(self, extent):
        """固定形の decoder が食える配置であることの不変条件（本数から stride を決める帰結）。"""
        axis = anima_tiling.plan_tile_axis(extent, 64, 8)

        assert axis.starts[0] == 0
        assert axis.starts[-1] == extent - 64, "最後のタイルは末端へスナップする"
        assert (extent - 64) % axis.stride == 0
        assert all(
            second - first == axis.stride
            for first, second in zip(axis.starts, axis.starts[1:], strict=False)
        ), "開始位置は等間隔"
        assert axis.tile - axis.stride >= 8 or len(axis.starts) == 1

    @pytest.mark.parametrize("extent", [80, 96, 112, 128, 160, 192, 256, 384, 512])
    def test_tile_count_is_minimal(self, extent):
        """本数は「重なりの下限を満たす最小」— 1 本減らすと下限を割るか割り切れない。

        これが無いと「常に stride 1」のような安全側の実装が緑のまま通り、タイル数が
        跳ね上がる（時間の冗長がそのまま効く）。
        """
        axis = anima_tiling.plan_tile_axis(extent, 64, 8)
        span = extent - 64

        for count in range(2, len(axis.starts)):
            assert span % (count - 1) != 0 or 64 - span // (count - 1) < 8, (
                f"{count} 本で足りるのに {len(axis.starts)} 本になっている"
            )

    def test_minimum_overlap_is_honored(self):
        """重なりの下限を上げると本数が増える（下限が本当に効いている）。"""
        loose = anima_tiling.plan_tile_axis(128, 64, 8)
        tight = anima_tiling.plan_tile_axis(128, 64, 48)

        assert len(tight.starts) > len(loose.starts)
        assert tight.tile - tight.stride >= 48

    def test_extent_shorter_than_the_tile_is_rejected(self):
        """固定形の decoder は短い入力を食えない — 黙ってゼロ埋めしない。"""
        with pytest.raises(ValueError, match="タイル幅"):
            anima_tiling.plan_tile_axis(32, 64, 8)

    def test_overlap_at_or_above_the_tile_width_is_rejected(self):
        with pytest.raises(ValueError, match="最小の重なり"):
            anima_tiling.plan_tile_axis(128, 64, 64)


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
            "stride": 32,
            "starts": [0, 32, 64],
            "blend_sample": 256,
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

        extent 16 / tile 8 / stride 4（重なり 4）で開始位置は 0,4,8。タイル内の値が
        `0..7` の傾斜のとき、貼り合わせ後は `0,1,2,3` → 中間は全て 4 → 末尾が `4,5,6,7`
        になる（重なり幅の線形ランプが傾斜を平坦へ潰す）。ブレンドの向き反転・stride の
        off-by-one・末端タイルのスナップ落とし・担当領域の取り違えは全てここで割れる。
        """
        geometry = anima_tiling.plan_tiling((1, 2, 16, 16), tile=8, min_overlap=2, scale=1)
        assert geometry.rows.starts == (0, 4, 8)

        out = anima_tiling.tiled_decode(_ramp_decoder(axis), torch.zeros(1, 2, 1, 16, 16), geometry)

        expected = torch.tensor([0.0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7])
        line = out[0, 0, 0, :, 0] if axis == "rows" else out[0, 0, 0, 0, :]
        assert torch.equal(line, expected)

    def test_the_far_edge_is_covered(self):
        """末端まで覆う（stride 幅で切り詰めるだけだと `n·stride < extent` で欠ける）。"""
        geometry = anima_tiling.plan_tiling((1, 2, 16, 16), tile=8, min_overlap=2, scale=1)

        out = anima_tiling.tiled_decode(
            lambda _tile: torch.full((1, 1, 1, 8, 8), 7.0), torch.zeros(1, 2, 1, 16, 16), geometry
        )

        assert list(out.shape) == [1, 1, 1, 16, 16]
        assert torch.equal(out, torch.full((1, 1, 1, 16, 16), 7.0))
