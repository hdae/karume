"""記号次元の同一性判定（`karume.extents`）— 素の `==` との差だけを固定する。

素の比較が**ヒント値**で答えを出すことは torch の実装挙動なので、実際に export した
SymInt を材料に採る（定数を並べたテストでは、直したはずの経路が再び `==` に戻っても緑のまま
通ってしまう）。guard は shape env を書き換えるため、`==` を踏むケースは**毎回 export し直す**。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.export import Dim

from karume.extents import extent_key, extent_keys, same_extents


class TwoInputs(nn.Module):
    """先頭次元だけが動的な 2 入力（形の比較をしないので、export 時点では無関係な 2 記号）。"""

    def forward(self, a, b):
        return a.sum() + b.sum()


def _two_symbols(hint: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """**別のシンボル**で、**ヒント値が同じ**先頭次元を持つ 2 つの meta を採る。"""
    exported = torch.export.export(
        TwoInputs(),
        (torch.zeros(hint, 3), torch.zeros(hint, 5)),
        dynamic_shapes={"a": {0: Dim("A", min=2, max=64)}, "b": {0: Dim("B", min=2, max=64)}},
    )
    first, second = (node for node in exported.graph.nodes if node.op == "placeholder")
    return first.meta["val"], second.meta["val"]


class TestSymbolicExtents:
    def test_two_symbols_with_the_same_hint_are_not_the_same_extent(self) -> None:
        first, second = _two_symbols()
        assert extent_key(first.shape[0]) != extent_key(second.shape[0])
        assert extent_keys(first.shape[:1]) != extent_keys(second.shape[:1])

    def test_the_plain_comparison_this_replaces_answers_by_the_hint(self) -> None:
        """素の `==` は別シンボルでも `True` を返す（= 直前のテストの regress 検出器）。

        観測できるのは答えだけではない — `==` は shape env に等価 guard を積むので、**踏んだ
        後**は 2 つの記号が同じ式へ潰れる。export 済みの制約に後付けの特殊化が混ざるとは
        この書き換えのこと。
        """
        first, second = _two_symbols()
        assert [first.shape[0]] == [second.shape[0]]
        assert extent_key(first.shape[0]) == extent_key(second.shape[0])

    def test_a_symbol_matches_itself(self) -> None:
        first, _ = _two_symbols()
        assert extent_keys(first.shape) == extent_keys(first.shape)
        assert same_extents(first, first)

    def test_a_static_extent_is_keyed_by_its_value(self) -> None:
        first, second = _two_symbols()
        assert extent_key(first.shape[1]) == 3
        assert extent_key(second.shape[1]) == 5

    def test_shapes_that_share_a_hint_are_not_the_same_shape(self) -> None:
        """同じ rank・同じヒント値でも、記号が違えば `same_extents` は False。"""
        first, second = _two_symbols()
        assert not same_extents(first, second[:, :3])

    def test_a_meta_without_a_shape_is_never_the_same(self) -> None:
        first, _ = _two_symbols()
        assert not same_extents(first, None)
        assert not same_extents(None, first)
