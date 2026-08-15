"""`anima.resolution` — 解像度の綴り（#23）。

参照台本の入口でしか使わない小さな関数だが、**壊れても実行時に例外が出ず数だけが静かに
ずれる**側の性質を持つ: `1344x768` を `768x1344` と読む誤りは、要素数も型も合ったまま
参照フィクスチャだけが転置された絵になる。往復の恒等と軸の順をここで固定する。
"""

from __future__ import annotations

import pytest

from anima.resolution import (
    GRANULARITY,
    format_resolution,
    parse_resolution,
    resolution_meta,
)


class TestParseResolution:
    def test_wxh_keeps_the_axis_order(self):
        """MUST: 先が幅・後が高さ（綴りの順）。入れ替えても要素数は合う。"""
        assert parse_resolution("1344x768") == (1344, 768)
        assert parse_resolution("768x1344") == (768, 1344)

    def test_a_bare_number_is_the_square_shorthand(self):
        assert parse_resolution("512") == (512, 512)

    def test_uppercase_x_is_accepted(self):
        assert parse_resolution("1024X1024") == (1024, 1024)

    @pytest.mark.parametrize("text", ["", "1344*768", "1344x", "x768", "1344x768x1", "-512", "1e3"])
    def test_a_malformed_spelling_is_refused(self, text):
        """綴りでない入力を既定へ縮退させない（打ち間違えが黙って別の形で走る）。"""
        with pytest.raises(ValueError, match="WxH"):
            parse_resolution(text)

    @pytest.mark.parametrize("text", ["1352x768", "768x1350", "0"])
    def test_a_side_off_the_grid_is_refused(self, text):
        """latent の各辺が patch 2 で割り切れない形は patchify が組めない。"""
        with pytest.raises(ValueError, match=str(GRANULARITY)):
            parse_resolution(text)


class TestFormatResolution:
    def test_a_square_uses_the_shorthand(self):
        """MUST: 正方は略記（既存フィクスチャのディレクトリ名が 1 文字も変わらないこと）。"""
        assert format_resolution(1024, 1024) == "1024"

    def test_a_non_square_spells_both_sides(self):
        assert format_resolution(1344, 768) == "1344x768"

    @pytest.mark.parametrize("text", ["512", "1024", "1344x768", "896x1152"])
    def test_the_round_trip_is_the_identity(self, text):
        assert format_resolution(*parse_resolution(text)) == text


class TestResolutionMeta:
    def test_a_square_keeps_the_int_field(self):
        """MUST: 正方の `resolution` は int のまま（既存の tiling.json を読むテストの欄）。"""
        assert resolution_meta(1024, 1024) == {"resolution": 1024, "width": 1024, "height": 1024}

    def test_a_non_square_spells_the_field(self):
        """一辺が存在しないので綴りを入れる。寸法の正本は width / height。"""
        assert resolution_meta(1344, 768) == {
            "resolution": "1344x768",
            "width": 1344,
            "height": 768,
        }
