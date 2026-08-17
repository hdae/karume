"""融合 attention（ADR 0023 + 0067 決定 1）のエクスポータ側の門。

見るのは 6 つ:

1. **SDPA 保存はターゲット別**（既定の分解表は 11 op のまま）。
2. `_h_attention` が causal / dropout≠0 / rank≠4 / D 不一致と、**契約外の mask**
   （bool / rank≠4 / `[B,1,M,N]` / `[1,H,M,N]` / M・N 不一致）を**全件 fail loudly**。
3. attrs の `scale` が **torch math decomp の定数とビット一致**（半スケール契約）。
4. 保存したグラフの IR に `attention` ノードがちょうど 1 本出る。
5. `enable_gqa=True` が**保存ノードとして通る**（GQA / MQA — ADR 0067 決定 1。形の妥当性は
   shapes 層が見る）。
6. bool マスクの加算型 f32 への畳み込み（`normalize._additive_attn_mask`）が、
   **分解経路が焼く定数とバイト一致**する。
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
    convert,
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


class ConstMaskAttention(nn.Module):
    """マスクを**持ち上げ定数**（buffer でも parameter でもない素の属性）で持つ形。

    定数畳み込みの葉として適格なのはこの形だけで、Gemma3 の帯マスク（T にだけ依存する
    定数）が IR で `sym_prefix_slice` に落ちる経路と同じ土俵に乗る。
    """

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.mask = mask

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)


class DropoutAttention(nn.Module):
    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.5)


class GqaAttention(nn.Module):
    """`enable_gqa=True` の SDPA（KV head を q head へ整除 broadcast する形）。"""

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)


class Rank3Attention(nn.Module):
    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v)


def _args(batch=2, heads=3, queries=5, keys=7, depth=4, kv_heads=None):
    """q / k / v の実引数（`kv_heads` 省略時は H = Hkv の従来形）。"""
    kv = heads if kv_heads is None else kv_heads
    return (
        torch.randn(batch, heads, queries, depth),
        torch.randn(batch, kv, keys, depth),
        torch.randn(batch, kv, keys, depth),
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

    def test_a_dropout_attention_is_rejected(self):
        with pytest.raises(NotImplementedError, match="dropout_p"):
            _preserved_attention(DropoutAttention(), _args())

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


class TestGqaIsAcceptedAsDivisibleBroadcast:
    """`enable_gqa=True` は保存ノードとして通る（ADR 0067 決定 1 の整除 broadcast）。

    ハンドラは `enable_gqa` を**読まない** — 形の妥当性（`H % Hkv == 0` かつ `H ≥ Hkv`・
    k / v 間の Hkv 一致）を見るのは shapes 層の 1 箇所（convert の出口の
    `assert_graph_shapes` が全ノードを必ず通す）で、二重に検査すると受理集合が 2 箇所へ
    分かれる。**非整除の形はここでは踏めない** — torch 自身が「Number of heads in key and
    value must divide the number of heads in query」で落とすので、拒否側は test_shapes.py と
    適合表（op-contracts.json）が押さえる。

    分解経路（既定の表）の GQA は救わない（ADR 0067 決定 1 後半 — `bmm` のバッチ突合は
    意図的な検出線）。そちらは従来どおり repeat_kv が expand / reshape で実体化される。
    """

    def test_a_gqa_sdpa_becomes_a_single_attention_node(self):
        graph, _ = _preserved_attention(GqaAttention(), _args(heads=4, kv_heads=2))

        assert [node.op for node in graph.nodes] == ["attention"]
        node = only_node(graph, "attention")
        assert len(node.ins) == 3
        assert [declared_shape(graph, name) for name in node.ins] == [
            [2, 4, 5, 4],
            [2, 2, 7, 4],
            [2, 2, 7, 4],
        ]
        # 出力は **q 側の H**（kv 側へ縮まない）
        assert declared_shape(graph, node.outs[0]) == [2, 4, 5, 4]
        # 契約（アリティ / attrs キー / 値域）を Python 側の表でも通す
        assert assert_node_contract(node, "test") is OP_CONTRACTS["attention"]

    def test_the_mqa_form_carries_the_single_kv_head(self):
        """Hkv=1 も同じ整除式で表す — 別 op を作らない。

        形は適合表の MQA ケース（8:1・head_dim 256）と同じ比・同じ D で踏む。
        """
        graph, _ = _preserved_attention(
            GqaAttention(), _args(batch=1, heads=8, depth=256, kv_heads=1)
        )
        node = only_node(graph, "attention")

        assert [declared_shape(graph, name) for name in node.ins] == [
            [1, 8, 5, 256],
            [1, 1, 7, 256],
            [1, 1, 7, 256],
        ]
        assert declared_shape(graph, node.outs[0]) == [1, 8, 5, 256]

    def test_enable_gqa_with_equal_head_counts_emits_the_same_node(self):
        """r=1 の形は `enable_gqa` の有無で IR が変わらない（比の欄を作らない趣旨）。"""
        args = _args()
        gqa, _ = _preserved_attention(GqaAttention(), args)
        plain, _ = _preserved_attention(PlainAttention(), args)

        assert [node.op for node in gqa.nodes] == ["attention"]
        assert only_node(gqa, "attention") == only_node(plain, "attention")
        assert gqa.values == plain.values


class TestAdditiveMaskIsAccepted:
    """契約の mask（f32 加算型・rank-4・ちょうど `[1,1,M,N]`）だけが 4 本目に載る。"""

    def test_the_mask_becomes_a_fourth_input(self):
        args = (*_args(), torch.zeros(1, 1, 5, 7))
        graph, _ = _preserved_attention(MaskedAttention(), args)

        node = only_node(graph, "attention")
        assert len(node.ins) == 4
        assert declared_shape(graph, node.ins[3]) == [1, 1, 5, 7]
        assert declared_shape(graph, node.outs[0]) == [2, 3, 5, 4]
        # 契約（アリティ 3 か 4 / attrs キー / 値域）を Python 側の表でも通す
        assert assert_node_contract(node, "test") is OP_CONTRACTS["attention"]

    def test_the_maskless_form_keeps_arity_three(self):
        """MUST: mask 無しの経路は 1 バイトも変わらない（省略可能スロットの意味）。"""
        graph, _ = _preserved_attention(PlainAttention(), _args())

        assert len(only_node(graph, "attention").ins) == 3

    def test_a_symbolic_mask_length_is_matched_symbolically(self):
        """M / N が記号でも、q / k と**同じ記号**であることを突き合わせる。"""
        seq = Dim("T", min=2, max=16)
        args = (
            torch.randn(2, 3, 5, 4),
            torch.randn(2, 3, 5, 4),
            torch.randn(2, 3, 5, 4),
            torch.zeros(1, 1, 5, 5),
        )
        graph, _ = export_and_convert(
            MaskedAttention(),
            args,
            dynamic_shapes=({2: seq}, {2: seq}, {2: seq}, {2: seq, 3: seq}),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        node = only_node(graph, "attention")
        assert declared_shape(graph, node.ins[3]) == [1, 1, "T", "T"]


class TestNonContractMasksFailLoudly:
    """MUST: 契約外の mask は**全件列挙して落とす**（欄の不存在が「語彙に無い」を表す）。"""

    @pytest.mark.parametrize(
        ("shape", "why"),
        [
            ((2, 3, 5, 7), "B と H の両方を持つ形"),
            ((2, 1, 5, 7), "[B,1,M,N]（バッチ別マスク）"),
            ((1, 3, 5, 7), "[1,H,M,N]（ヘッド別マスク）"),
        ],
        ids=("bh", "batch", "head"),
    )
    def test_a_leading_axis_other_than_one_is_rejected(self, shape, why):
        args = (*_args(), torch.zeros(shape))
        with pytest.raises(NotImplementedError, match="先頭 2 軸"):
            _preserved_attention(MaskedAttention(), args)
        assert why  # 失敗時に形の意図が読めるようにパラメータへ残す

    def test_a_broadcast_query_axis_is_rejected(self):
        """`[1,1,1,N]`（全 query 行に同じマスク）は torch では通るが契約は M 一致を要求する。"""
        args = (*_args(), torch.zeros(1, 1, 1, 7))
        with pytest.raises(NotImplementedError, match="M / N"):
            _preserved_attention(MaskedAttention(), args)

    def test_a_rank3_mask_is_rejected(self):
        args = (*_args(), torch.zeros(1, 5, 7))
        with pytest.raises(NotImplementedError, match=r"rank 3 の attn_mask"):
            _preserved_attention(MaskedAttention(), args)

    def test_a_bool_mask_is_rejected_by_the_handler_itself(self):
        """正規化パスを外すと bool のまま届き、ハンドラの dtype 門が落とす。

        MUST: 門を正規化パスだけに置かない — パスが（形が違う等で）発火しなかったとき、
        bool マスクが黙って initializer 化の失敗まで滑り落ちる。
        """
        args = (*_args(), torch.ones(1, 1, 5, 7, dtype=torch.bool))
        exported = torch.export.export(MaskedAttention(), args, strict=False)
        decomposed = exported.run_decompositions(
            curated_decompositions(PRESERVED_OP_PREFIXES_WITH_ATTENTION)
        )

        with pytest.raises(NotImplementedError, match=r"dtype torch\.bool"):
            convert(decomposed)


class TestBoolMaskFoldsToTheDecompositionConstant:
    """bool → 加算型 f32 の畳み込みが torch 自身の変換と**同じ定数**であることの実測。

    ADR 0023 のビット同一（融合経路と分解経路が同じ値を出す）は、マスク側でも
    「同じ 0 / −inf の f32 定数が同じ位置で足される」ことに依存している。
    """

    #: 帯マスク（対角の周りだけ True）— 全 True だと定数が縮退して比較が恒真になる。
    MASK = torch.tril(torch.ones(5, 5, dtype=torch.bool), diagonal=1)[None, None]

    def _folded_masks(self, preserved) -> list[torch.Tensor]:
        graph, tensors = export_and_convert(
            ConstMaskAttention(self.MASK),
            _args(batch=2, heads=3, queries=5, keys=5),
            preserved=preserved,
        )
        return [
            tensors[init.tensor]
            for name, init in graph.initializers.items()
            if graph.values[name].shape == [1, 1, 5, 5]
        ]

    def test_the_preserved_path_bakes_the_same_bytes_as_the_decomposed_path(self):
        decomposed = self._folded_masks(PRESERVED_OP_PREFIXES)
        preserved = self._folded_masks(PRESERVED_OP_PREFIXES_WITH_ATTENTION)

        assert len(decomposed) == 1, f"分解経路の mask 定数が {len(decomposed)} 本"
        assert len(preserved) == 1, f"保存経路の mask 定数が {len(preserved)} 本"
        assert decomposed[0].dtype is torch.float32
        assert torch.equal(decomposed[0], preserved[0])
        # 恒真化の防止: 定数が実際に 0 と −inf の 2 値を持つ（全 0 なら比較に意味が無い）
        assert set(preserved[0].unique().tolist()) == {0.0, float("-inf")}

    def test_the_preserved_path_carries_the_mask_as_an_initializer(self):
        graph, _ = export_and_convert(
            ConstMaskAttention(self.MASK),
            _args(batch=2, heads=3, queries=5, keys=5),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        node = only_node(graph, "attention")
        assert len(node.ins) == 4
        assert node.ins[3] in graph.initializers
