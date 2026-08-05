"""融合 attention（ADR 0023）のエクスポータ側の門。

見るのは 4 つ:

1. **SDPA 保存はターゲット別**（既定の分解表は 11 op のまま）。
2. `_h_attention` が mask / causal / dropout≠0 / GQA / rank≠4 / D 不一致を**全件 fail loudly**。
3. attrs の `scale` が **torch math decomp の定数とビット一致**（半スケール契約）。
4. 保存したグラフの IR に `attention` ノードがちょうど 1 本出る。
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import export_and_convert, only_node
from torch import nn
from torch.export import Dim

from karume.convert import (
    ATTENTION_OP_PREFIX,
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    curated_decompositions,
)
from karume.ops import OP_CONTRACTS, assert_node_contract
from karume.shapes import declared_shape

#: D=128 の実測形（DiT の self / cross attention）。scale 省略時の半スケールの検証に使う。
DIT_DEPTH = 128
#: 設計 recon が実 IR から読み出したオラクル（`const.6953fe58410d6c34`）。
DIT_HALF_SCALE = 0.2973017692565918
#: D を記号にする動的軸（既定 scale が export 時に決まらないことの検証用）。
DYN_DEPTH = Dim("D", min=2, max=16)


class PlainAttention(nn.Module):
    """マスク無し・非因果・dropout 0 の SDPA（融合の対象そのもの）。"""

    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)


class CausalAttention(nn.Module):
    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


class MaskedAttention(nn.Module):
    def forward(self, q, k, v, mask):
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class DropoutAttention(nn.Module):
    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.5)


class GqaAttention(nn.Module):
    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)


class Rank3Attention(nn.Module):
    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v)


def _args(batch=2, heads=3, queries=5, keys=7, depth=4):
    return (
        torch.randn(batch, heads, queries, depth),
        torch.randn(batch, heads, keys, depth),
        torch.randn(batch, heads, keys, depth),
    )


def _preserved_attention(module: nn.Module, args: tuple[torch.Tensor, ...]):
    return export_and_convert(module, args, preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION)


class TestPreservedIsPerTarget:
    def test_the_default_table_does_not_preserve_sdpa(self):
        """MUST: 既定の分解表は 11 op のまま。

        グローバルに 12 op へ広げると、mask 付き SDPA を持つグラフ（Anima text_encoder の
        −inf 折り込み因果マスク）が `_h_attention` の fail loudly に当たり **export できなく
        なる**。ADR 0016 の safe-softmax ガード除去パスもその分解経路に掛かっている。
        """
        assert ATTENTION_OP_PREFIX not in PRESERVED_OP_PREFIXES
        assert (
            *PRESERVED_OP_PREFIXES,
            ATTENTION_OP_PREFIX,
        ) == PRESERVED_OP_PREFIXES_WITH_ATTENTION

    def test_the_default_table_still_decomposes_sdpa(self):
        """既定では従来どおり分解される（bmm → softmax → bmm の語彙で通る）。"""
        graph, _ = export_and_convert(PlainAttention(), _args())

        ops = [node.op for node in graph.nodes]
        assert "attention" not in ops
        assert ops.count("bmm") == 2
        assert ops.count("softmax") == 1

    def test_the_curated_table_keeps_sdpa_only_when_asked(self):
        table_default = curated_decompositions()
        table_with = curated_decompositions(PRESERVED_OP_PREFIXES_WITH_ATTENTION)
        preserved_names = {
            str(overload)
            for overload in table_default
            if str(overload).startswith(ATTENTION_OP_PREFIX)
        }
        assert preserved_names, "既定表に SDPA のエントリが無い（前提が崩れている）"
        assert not any(str(overload).startswith(ATTENTION_OP_PREFIX) for overload in table_with)

    def test_a_masked_attention_still_exports_under_the_default_table(self):
        """MUST: 保存しないターゲットは従来どおり通る（ADR 0016 の温存の実証）。

        text_encoder が実際に持つのはこの形（`attn_mask` に −inf を折り込んだバイアス）で、
        保存対象に含めると `_h_attention` の fail loudly で **export できなくなる**。
        """
        args = (*_args(), torch.zeros(2, 3, 5, 7))
        graph, _ = export_and_convert(MaskedAttention(), args)

        ops = [node.op for node in graph.nodes]
        assert "attention" not in ops
        assert ops.count("softmax") == 1


class TestPreservedProducesOneNode:
    def test_a_preserved_sdpa_becomes_a_single_attention_node(self):
        graph, _ = _preserved_attention(PlainAttention(), _args())

        assert [node.op for node in graph.nodes] == ["attention"]
        node = only_node(graph, "attention")
        assert len(node.ins) == 3
        # 契約（アリティ / attrs キー / 値域）を Python 側の表でも通す
        assert assert_node_contract(node, "test") is OP_CONTRACTS["attention"]

    def test_the_declared_shapes_are_head_first_rank4(self):
        graph, _ = _preserved_attention(PlainAttention(), _args())
        node = only_node(graph, "attention")

        shapes = [declared_shape(graph, name) for name in node.ins]
        assert shapes == [[2, 3, 5, 4], [2, 3, 7, 4], [2, 3, 7, 4]]
        assert declared_shape(graph, node.outs[0]) == [2, 3, 5, 4]

    def test_the_decomposition_debris_is_gone(self):
        """保存すると decomp 由来の恒等 expand / scale mul / kᵀ permute が IR ごと消える。"""
        graph, _ = _preserved_attention(PlainAttention(), _args())

        ops = [node.op for node in graph.nodes]
        for debris in ("expand", "mul", "permute", "bmm", "softmax", "reshape"):
            assert debris not in ops


class TestHalfScaleIsBitExact:
    def test_the_default_scale_matches_the_torch_math_decomp_constant(self):
        """MUST: `f32(√(1/√D))` が torch decomp の `mul` 定数と**ビット一致**する。

        オラクルは実 DiT IR の `const.6953fe58410d6c34` = 0.2973017692565918
        （= f32(128^-0.25)）。半スケールを全スケール（`1/√D`）に変えたり、`√` の位置を
        変えたりすると、ここが真っ先に赤くなる。
        """
        graph, _ = _preserved_attention(
            PlainAttention(), _args(batch=1, heads=2, queries=4, keys=4, depth=DIT_DEPTH)
        )
        scale = only_node(graph, "attention").attrs["scale"]

        assert scale == DIT_HALF_SCALE
        # f32 表現としても一致（JSON 往復と f32 丸めで ulp が動かない）
        assert torch.tensor(scale, dtype=torch.float32).item() == DIT_HALF_SCALE
        # 全スケール（内積後に 1 度掛ける形）の値とは**別物**であることを明示的に固定する
        assert scale != pytest.approx(1.0 / math.sqrt(DIT_DEPTH))
        assert scale * scale == pytest.approx(1.0 / math.sqrt(DIT_DEPTH), rel=1e-6)

    def test_the_torch_decomposition_emits_the_same_constant(self):
        """torch 側の decomp が実際に同じ定数を 2 本の `mul` に載せることを実測で押さえる。"""
        args = _args(batch=1, heads=2, queries=4, keys=4, depth=DIT_DEPTH)
        exported = torch.export.export(PlainAttention(), args, strict=False)
        decomposed = exported.run_decompositions(curated_decompositions())

        scalars = [
            node.args[1]
            for node in decomposed.graph.nodes
            if node.op == "call_function" and "mul" in str(node.target)
        ]
        assert len(scalars) == 2, f"scale の mul が 2 本でない: {scalars}"
        assert scalars[0] == scalars[1]
        assert torch.tensor(scalars[0], dtype=torch.float32).item() == DIT_HALF_SCALE

    def test_an_explicit_scale_is_carried_through_as_its_square_root(self):
        graph, _ = _preserved_attention(PlainAttention(scale=0.25), _args())
        scale = only_node(graph, "attention").attrs["scale"]

        assert scale == pytest.approx(0.5)


class TestUnsupportedFormsFailLoudly:
    """MUST: 語彙に無い形は**全件列挙して落とす**（黙って近似しない — 横断の不変条件）。"""

    def test_a_causal_attention_is_rejected(self):
        with pytest.raises(NotImplementedError, match="is_causal"):
            _preserved_attention(CausalAttention(), _args())

    def test_a_masked_attention_is_rejected(self):
        args = (*_args(), torch.zeros(2, 3, 5, 7))
        with pytest.raises(NotImplementedError, match="attn_mask"):
            export_and_convert(
                MaskedAttention(), args, preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION
            )

    def test_a_dropout_attention_is_rejected(self):
        with pytest.raises(NotImplementedError, match="dropout_p"):
            _preserved_attention(DropoutAttention(), _args())

    def test_a_gqa_attention_is_rejected(self):
        args = (
            torch.randn(2, 4, 5, 4),
            torch.randn(2, 2, 7, 4),
            torch.randn(2, 2, 7, 4),
        )
        with pytest.raises(NotImplementedError, match="enable_gqa"):
            _preserved_attention(GqaAttention(), args)

    def test_a_rank3_attention_is_rejected(self):
        args = (torch.randn(2, 5, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 4))
        with pytest.raises(NotImplementedError, match="rank-4"):
            _preserved_attention(Rank3Attention(), args)

    def test_a_mismatched_value_depth_is_rejected(self):
        """v の D だけ違う形（torch は許すが契約は 3 者とも同じ D）。"""
        args = (
            torch.randn(2, 3, 5, 4),
            torch.randn(2, 3, 7, 4),
            torch.randn(2, 3, 7, 6),
        )
        with pytest.raises(NotImplementedError, match="D"):
            _preserved_attention(PlainAttention(), args)

    def test_a_symbolic_depth_is_rejected(self):
        """D が記号次元だと `√scale_factor` を export 時に決められない（既定 scale の前提）。"""
        args = _args()
        with pytest.raises(NotImplementedError, match="記号次元"):
            export_and_convert(
                PlainAttention(),
                args,
                dynamic_shapes=({3: DYN_DEPTH}, {3: DYN_DEPTH}, {3: DYN_DEPTH}),
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )
