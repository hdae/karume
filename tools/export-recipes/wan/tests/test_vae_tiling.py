"""`wan/vae_tiling.py` の約束事（ADR 0118 決定 2 / 段 5 — タイル decode の幾何とブレンド）。

ここで固定するのは、壊れても例外が出ず**値だけが静かにずれる**側:

- タイル計画の開始位置が TS 側（`packages/models/src/wan/vae-tiles.ts`）と一致すること
  （{@link MIRRORED_STARTS} — TS 側 `packages/models/tests/wan_vae_tiles_test.ts` の
  `AXIS_STARTS` と同じ値の二重凍結。ADR 0033 追記 9a の流儀）
- 丸め等間隔スナップ配置の不変条件（末尾が `extent − tile` / 重なりが下限以上 / 本数が最小 /
  間隔のばらつきが高々 1 潜在）
- ブレンド式が上流（`AutoencoderKLWan.blend_v` / `blend_h`）と**ビット一致**
- 貼り付けの解析解（ランプの再構成・角での縦 → 横の順・末端の被覆）
- 縮退（タイル 1 枚）のタイル参照が上流の非タイル chunk ループと**ビット一致**（実重み）
"""

from __future__ import annotations

import re

import pytest
import torch

from wan import vae_patch, vae_tiling

#: 実寸のタイル辺（潜在 — chunk グラフの既定の入力形）。
TILE = 32

#: Wan2.1 の重なりの下限（潜在）— 出力 64 px から式で導いた値（{@link TestMinOverlapLatent}）。
OVERLAP = vae_tiling.WAN21_MIN_OVERLAP_LATENT

#: タイル辺 32・重なりの下限 8 の開始位置（潜在の全長 → 開始位置列）。TS 側
#: （`packages/models/tests/wan_vae_tiles_test.ts` の `AXIS_STARTS`）と**同じ値**で凍結する
#: — 両実装を機械で突き合わせる経路は GPU の照合（フィクスチャのメタ）だけなので、GPU の無い
#: 機でも鏡像のずれを捕まえる。
#:
#: - 32: 縮退（1 枚 — 縮退門の形）
#: - 60 / 104: 受理集合（832×480 / 480×832 — ADR 0118 決定 7）の 2 辺。ADR の値（行 0 / 14 / 28・
#:   列 0 / 24 / 48 / 72）そのもの
#: - 57: `i·span/(本数−1)` がちょうど半分になる辺（span 25 / 3 本 → 12.5）。整数式が組み込み
#:   `round`（偶数丸め）へ退行すると (0, 12, 25) になる — 他の辺は半端が出ないので、この 1 行
#:   だけがその退行を捕まえる（anima の 1872px と同じ役）
MIRRORED_STARTS = {
    32: (0,),
    57: (0, 13, 25),
    60: (0, 14, 28),
    104: (0, 24, 48, 72),
}


class TestMinOverlapLatent:
    """重なりの下限は出力 64 px ÷ 空間の圧縮（グラフの比 × unpatchify の倍率 — ADR 0121 決定 6）。

    2.1 の値（8）だけでは式が unpatchify の倍率を無視しても赤にならないので、倍率 2 の行も置いて
    式を 2 点で縛る（2.2 の計画表ではなく式の検証）。
    """

    @pytest.mark.parametrize(("scale", "patch_size", "expected"), [(8, 1, 8), (8, 2, 4)])
    def test_the_overlap_is_64_px_over_the_spatial_compression(
        self, scale: int, patch_size: int, expected: int
    ):
        assert vae_tiling.min_overlap_latent(scale, patch_size) == expected

    def test_wan21_overlap_is_eight_latents(self):
        """2.1 の計画表（{@link MIRRORED_STARTS}）を作った値と同じ。"""
        assert vae_tiling.WAN21_MIN_OVERLAP_LATENT == 8
        assert type(vae_tiling.WAN21_MIN_OVERLAP_LATENT) is int

    def test_a_compression_that_does_not_divide_64_px_is_rejected(self):
        """丸めない — 丸めると重なりが 64 px から黙ってずれる。"""
        with pytest.raises(ValueError, match=r"空間の圧縮 24（グラフの比 8 × unpatchify 3）"):
            vae_tiling.min_overlap_latent(8, 3)

    @pytest.mark.parametrize(
        ("scale", "patch_size"), [(8.0, 1), (8, 1.0), (True, 1), (8, True), (0, 1), (8, 0), (-8, 1)]
    )
    def test_a_factor_that_is_not_a_positive_int_is_rejected(
        self, scale: object, patch_size: object
    ):
        """TS 側（`Number.isInteger`）と同じく真偽値・0 以下を拒み、さらに型として int を要求する。

        TS は 8.0 を整数と見る（JS に float と int の区別が無い）が、Python 側は 8.0 を通すと
        float が流れるので TS より厳しく拒む。文言は内訳（グラフの比・unpatchify の倍率）を名指す。
        """
        expected = rf"正の整数でない（グラフの比 {re.escape(repr(scale))}・unpatchify "
        with pytest.raises(ValueError, match=expected):
            vae_tiling.min_overlap_latent(scale, patch_size)

    def test_the_tile_planners_require_the_overlap(self):
        """渡し忘れは TypeError — 2.1 の値で黙って計画しない。"""
        with pytest.raises(TypeError):
            vae_tiling.plan_tile_axis(60, TILE)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            vae_tiling.plan_tiles(60, 104, TILE)  # type: ignore[call-arg]


class TestPlanTileAxis:
    @pytest.mark.parametrize("extent", sorted(MIRRORED_STARTS))
    def test_starts_match_the_ts_side(self, extent: int):
        assert vae_tiling.plan_tile_axis(extent, TILE, OVERLAP).starts == MIRRORED_STARTS[extent]

    def test_832x480_is_twelve_tiles(self):
        """832×480（潜在 60×104）は 3 × 4 = 12 枚・面積は非タイルの 1.97 倍（ADR 0118 決定 2）。"""
        plan = vae_tiling.plan_tiles(60, 104, TILE, OVERLAP)

        assert plan.tiles == 12
        assert plan.scale == 8
        # 重なりは行 32 − 14 = 18 潜在（144 px）・列 32 − 24 = 8 潜在（64 px = 下限ちょうど）。
        assert plan.rows.blends(plan.scale) == [144, 144]
        assert plan.cols.blends(plan.scale) == [64, 64, 64]
        # 担当領域の和が全長（末端まで覆う）。
        assert [plan.rows.region(i) for i in range(3)] == [14, 14, 32]
        assert [plan.cols.region(j) for j in range(4)] == [24, 24, 24, 32]
        assert round(plan.tiles * TILE * TILE / (60 * 104), 2) == 1.97

    def test_480x832_is_the_transpose(self):
        plan = vae_tiling.plan_tiles(104, 60, TILE, OVERLAP)

        assert plan.rows.starts == MIRRORED_STARTS[104]
        assert plan.cols.starts == MIRRORED_STARTS[60]

    @pytest.mark.parametrize("extent", [32, 33, 40, 56, 57, 60, 64, 80, 104, 128, 135, 256])
    def test_invariants_hold_for_every_extent(self, extent: int):
        """固定形の chunk グラフが食える配置であることの不変条件（丸め等間隔配置の帰結）。"""
        axis = vae_tiling.plan_tile_axis(extent, TILE, OVERLAP)
        span = extent - TILE
        gaps = [second - first for first, second in zip(axis.starts, axis.starts[1:], strict=False)]

        assert axis.starts[0] == 0
        assert axis.starts[-1] == span, "最後のタイルは末端へスナップする"
        assert all(TILE - gap >= OVERLAP for gap in gaps)
        assert sum(axis.region(i) for i in range(len(axis.starts))) == extent
        # 丸め等間隔の実体 = 間隔の差は高々 1 潜在（「固定 stride + 末尾だけスナップ」への退行で
        # 最後の対だけ大きく開いて割れる）。
        assert not gaps or max(gaps) - min(gaps) <= 1, f"間隔のばらつき {gaps}"

    @pytest.mark.parametrize("extent", [33, 40, 56, 57, 60, 64, 80, 104, 128, 135, 256])
    def test_tile_count_is_minimal(self, extent: int):
        """本数は重なりの下限だけを制約にした最小（安全側に倒した実装はタイル数が跳ねる）。"""
        span = extent - TILE

        assert len(vae_tiling.plan_tile_axis(extent, TILE, OVERLAP).starts) == (
            -(-span // (TILE - OVERLAP)) + 1
        )

    def test_extent_shorter_than_the_tile_is_rejected(self):
        """固定形のグラフは短い入力を食えない — 黙ってゼロ埋めしない。"""
        with pytest.raises(ValueError, match="タイル幅"):
            vae_tiling.plan_tile_axis(31, TILE, OVERLAP)

    def test_overlap_at_or_above_the_tile_width_is_rejected(self):
        with pytest.raises(ValueError, match="最小の重なり"):
            vae_tiling.plan_tile_axis(60, TILE, TILE)

    def test_a_pair_outside_the_axis_is_rejected(self):
        with pytest.raises(ValueError, match="ブレンド対"):
            vae_tiling.plan_tile_axis(TILE, TILE, OVERLAP).blend_at(8, 1)

    def test_meta_carries_the_geometry(self):
        """フィクスチャのメタ（TS の計画と突き合わせる欄）。"""
        assert vae_tiling.plan_tiles(60, 104, TILE, OVERLAP).meta() == {
            "tile": "32",
            "scale": "8",
            "rows_starts": "0,14,28",
            "cols_starts": "0,24,48,72",
            "rows_blend": "144,144",
            "cols_blend": "64,64,64",
        }


class TestBlendIsomorphism:
    """上流の `blend_v` / `blend_h` の**逐語**であることを本物との突合で固定する。

    `importorskip` は各テストの中で受ける（モジュール直下に置くと幾何のケースまで skip する —
    `anima/tests/test_tiling.py` と同じ判断）。
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
    """計画の形のタイル配列（`make(i, j)` が `[1, C, F, s, s]` を返す）。"""
    return [
        [make(i, j) for j in range(len(plan.cols.starts))] for i in range(len(plan.rows.starts))
    ]


def _ramp(plan: vae_tiling.TilePlan, axis: str):
    """タイル内の位置（行 / 列番号）を値に持つタイル（重なりの値が上下・左右のタイルで**違う**）。

    同じ値が重なる作りだと、ブレンドの向きを反転しても緑のまま通る（ADR 0033 の知見 3）。
    """
    side = plan.rows.tile * plan.scale
    index = torch.arange(side, dtype=torch.float32)
    plane = index.view(-1, 1).expand(side, side) if axis == "rows" else index.expand(side, side)

    def make(_i: int, _j: int) -> torch.Tensor:
        return plane.reshape(1, 1, 1, side, side).expand(1, 1, 2, side, side).clone()

    return make


class TestAssembleTiles:
    def test_single_tile_is_a_bitwise_identity(self):
        """縮退（1 枚）はブレンド無し・領域が全体 — タイルの素の写しになる。"""
        plan = vae_tiling.plan_tiles(8, 8, tile=8, scale=1, min_overlap=2)
        tile = torch.randn(1, 3, 5, 8, 8)

        out = vae_tiling.assemble_tiles([[tile.clone()]], plan)

        assert torch.equal(out, tile)

    @pytest.mark.parametrize("axis", ["rows", "cols"])
    def test_linear_ramp_is_reconstructed_across_the_seams(self, axis: str):
        """全長 16・タイル 8・開始 0 / 4 / 8（重なり 4）で、タイル内の傾斜 0..7 は貼り合わせ後
        `0,1,2,3` → 中間は全て 4 → 末尾 `4,5,6,7` になる（重なり幅の線形ランプが傾斜を平坦へ潰す）。

        ブレンドの向き反転・間隔の off-by-one・末端のスナップ落とし・担当領域の取り違えは全て
        ここで割れる。全フレームに同じ式が掛かることも見る。
        """
        plan = vae_tiling.plan_tiles(16, 16, tile=8, scale=1, min_overlap=2)
        assert plan.rows.starts == (0, 4, 8)

        out = vae_tiling.assemble_tiles(_grid(plan, _ramp(plan, axis)), plan)

        expected = torch.tensor([0.0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7])
        for frame in range(2):
            line = out[0, 0, frame, :, 0] if axis == "rows" else out[0, 0, frame, 0, :]
            assert torch.equal(line, expected), f"フレーム {frame}"

    def test_corners_blend_vertically_first(self):
        """角（4 枚が重なる領域）は縦 → 横の順（上流と同じ）。

        全長 6・タイル 4・開始 0 / 2（ブレンド幅 2）で、定数タイル a=1（左上）/ b=2 / c=3 / d=4
        （右下）。右下タイルの 2 行目（出力の行 3）を手で追う:

        - 左下タイルは上（a）と縦ブレンド済み: 行 1 = (a + c) / 2 = 2。
        - 右上タイルは左（a）と横ブレンド済み: 行 3 = [1, 1.5, 2, 2]。
        - 右下タイル（縦が先）: 行 1 = 0.5·右上の行 3 + 0.5·d = [2.5, 2.75, 3, 3] → 横で
          列 0 = 左下の (1, 2) = 2・列 1 = 0.5·2 + 0.5·2.75 = 2.375。

        出力の行 3 = 左下の担当 [2, 2] + 右下の担当 [2, 2.375, 3, 3]。横 → 縦の順に入れ替えると
        [2, 2, 1.5, 2.25, 3, 3] になる。
        """
        plan = vae_tiling.plan_tiles(6, 6, tile=4, scale=1, min_overlap=2)
        assert plan.rows.starts == (0, 2)
        values = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}

        out = vae_tiling.assemble_tiles(
            _grid(plan, lambda i, j: torch.full((1, 1, 1, 4, 4), values[(i, j)])), plan
        )

        assert out[0, 0, 0, 3].tolist() == [2.0, 2.0, 2.0, 2.375, 3.0, 3.0]

    def test_the_far_edge_is_covered(self):
        """末端まで覆う（stride 幅で切り詰めるだけだと総和が `extent − tile` で欠ける）。"""
        plan = vae_tiling.plan_tiles(16, 26, tile=8, scale=2, min_overlap=2)

        out = vae_tiling.assemble_tiles(
            _grid(plan, lambda _i, _j: torch.full((1, 3, 2, 16, 16), 7.0)), plan
        )

        assert list(out.shape) == [1, 3, 2, 32, 52]
        assert torch.equal(out, torch.full_like(out, 7.0))

    def test_a_grid_that_does_not_match_the_plan_is_rejected(self):
        plan = vae_tiling.plan_tiles(16, 16, tile=8, scale=1, min_overlap=2)

        with pytest.raises(ValueError, match="計画"):
            vae_tiling.assemble_tiles([[torch.zeros(1, 1, 1, 8, 8)]], plan)


class TestFixtureCases:
    def test_acceptance_differs_from_the_band_in_every_axis(self):
        """受入れは帯の決定用と潜在（seed）・chunk 境界・タイルの位置（縦横）が全て違う
        （ADR 0118 追記 2026-10-02）。縦長は行と列の取り違えを値に出す。"""
        cases = {case.role: case for case in vae_tiling.FIXTURE_CASES}
        band, accept = cases["band"], cases["accept"]

        assert (band.height, band.width, band.chunks) == (60, 104, 9)
        assert band.full and not accept.full
        assert accept.seed != band.seed
        assert accept.chunks != band.chunks
        assert (accept.height, accept.width) == (band.width, band.height)


@pytest.fixture(scope="module")
def vae(wan_snapshot):
    """pin した revision の VAE（CPU f32・丸め無し — 恒等の主張は重みの値に依らない）。"""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


class TestDegenerateTile:
    def test_one_tile_is_the_untiled_chunk_loop_bitwise(self, vae):
        """潜在がタイル 1 枚ちょうど（8×8・3 chunk）なら、タイル参照は上流の非タイル `_decode` の
        chunk ループ（クランプ前）と**ビット一致**（ADR 0033 決定 4 の縮退の Python 側）。

        post-quant の 1×1×1 conv の掛け方（上流 `tiled_decode` はフレームごと・非タイル `_decode` は
        全フレームに 1 度）の違いも含めて値が変わらないことを見る。
        """
        generator = torch.Generator().manual_seed(11)
        latents = torch.randn(1, 16, 3, 8, 8, generator=generator)
        plan = vae_tiling.plan_tiles(8, 8, tile=8, min_overlap=2)
        assert plan.tiles == 1

        with torch.no_grad():
            tiled = vae_tiling.tiled_decode_unclamped(vae, latents, plan)
            untiled = vae_patch.reference_decode_unclamped(vae, latents)

        assert torch.equal(tiled, untiled)
