"""正規化パスの振る舞い固定。

各パスは「IR 語彙を増やさない同値書き換え」なので、①書き換え前は語彙外で落ちること
②書き換え後は狙った 1 ノードになること の両方を見る（片側だけでは、パスが無くても
通る恒真テストになりうる）。
"""

from __future__ import annotations

import pytest
import torch
from conftest import export_and_convert, node_ops, only_node
from torch import nn

from karume.convert import UnsupportedAtenOpsError, convert, curated_decompositions
from karume.normalize import normalize_graph
from karume.ops import STRIDED_RANK

aten = torch.ops.aten


def decompose(module: nn.Module, args: tuple):
    ep = torch.export.export(module, args, strict=False)
    return ep.run_decompositions(curated_decompositions())


class Square(nn.Module):
    def forward(self, x):
        return x**2


class Cube(nn.Module):
    def forward(self, x):
        return x**3


class NoOpCast(nn.Module):
    def forward(self, x):
        return x.to(torch.float32) * x


class TestPow2ToMul:
    def test_squared_input_becomes_a_single_mul(self):
        graph, _ = export_and_convert(Square(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["mul"]
        assert only_node(graph, "mul").ins == ["x", "x"]

    def test_the_pass_reports_its_firing_count(self):
        decomposed = decompose(Square(), (torch.randn(3, 4),))

        assert normalize_graph(decomposed)["pow2->mul"] == 1

    def test_without_the_pass_pow_is_outside_the_vocabulary(self):
        """パスが載荷であることの確認 — 書き換えなければ未対応 op で落ちる。"""
        decomposed = decompose(Square(), (torch.randn(3, 4),))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.pow.Tensor_Scalar" in err.value.ops

    def test_other_exponents_are_left_alone(self):
        """同値でない書き換えはしない（3 乗は mul 1 回に落ちない）。"""
        decomposed = decompose(Cube(), (torch.randn(3, 4),))
        normalize_graph(decomposed)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.pow.Tensor_Scalar" in err.value.ops

    def test_a_promoting_float_exponent_is_left_alone(self):
        """`2.0` は `2.0 == 2` で素通りするが、整数テンソルでは型昇格を伴う（同値でない）。

        `pow(i64, 2.0)` は f32 を返す一方 `mul(i64, i64)` は i64 のまま — 書き換えると
        黙って別の dtype の値になる。触らなければ未対応 op として全件列挙で落ちる。
        """
        decomposed = decompose(FloatSquare(), (torch.arange(6).reshape(3, 2),))
        stats = normalize_graph(decomposed)

        assert "pow2->mul" not in stats
        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.pow.Tensor_Scalar" in err.value.ops

    def test_a_float_exponent_is_left_alone_even_without_promotion(self):
        """受理するのは**整数リテラルの 2**だけ（f32 では昇格が無くても書き換えない）。

        `2.0 == 2` に頼ると「昇格が無ければ同値」という判断を dtype 検査 1 本に預けることに
        なる。指数の表記そのものを門にしておけば、受理条件が「見た形」で読める。
        """
        decomposed = decompose(FloatSquare(), (torch.randn(3, 2),))
        stats = normalize_graph(decomposed)

        assert "pow2->mul" not in stats


class FloatSquare(nn.Module):
    """型昇格を伴う 2 乗（整数テンソル × float の指数）。"""

    def forward(self, x):
        return x**2.0


class Repeated(nn.Module):
    """恒等 repeat（引数が全て 1）— 実測 4 本と同じ形（recon §4-1）。"""

    def forward(self, x):
        return x.repeat(1, 1)


class Tripled(nn.Module):
    """本物の repeat（IR 語彙に無い）— 恒等除去が誤爆しないことの対照。"""

    def forward(self, x):
        return x.repeat(3, 1)


class RankRaisingRepeat(nn.Module):
    """引数が全て 1 でも本数が rank を超えれば rank が上がる（恒等ではない）。"""

    def forward(self, x):
        return x.repeat(1, 1, 1)


class ZeroAdd(nn.Module):
    def forward(self, x):
        return x + 0


class TestDropIdentityRepeat:
    def test_all_ones_repeat_disappears(self):
        graph, _ = export_and_convert(Repeated(), (torch.randn(3, 4),))

        assert node_ops(graph) == []
        assert graph.outputs == ["x"]

    def test_the_pass_reports_its_firing_count(self):
        decomposed = decompose(Repeated(), (torch.randn(3, 4),))

        assert normalize_graph(decomposed)["drop_identity_repeat"] == 1

    @pytest.mark.parametrize(
        "module", [Tripled(), RankRaisingRepeat()], ids=["real-repeat", "rank-raising"]
    )
    def test_non_identity_repeats_stay_in_the_enumeration(self, module):
        """B > 1 の repeat と rank を上げる形は書き換えない（黙って近似しない）。"""
        decomposed = decompose(module, (torch.randn(3, 4),))
        normalize_graph(decomposed)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.repeat.default" in err.value.ops


class TestDropIdentityAdd:
    def test_adding_zero_disappears(self):
        graph, _ = export_and_convert(ZeroAdd(), (torch.randn(3, 4),))

        assert node_ops(graph) == []

    def test_it_runs_before_scalar_promotion(self):
        """順序が逆だと `+0` が rank-1 定数になり、恒等除去の対象でなくなる。"""
        decomposed = decompose(ZeroAdd(), (torch.randn(3, 4),))
        stats = normalize_graph(decomposed)

        assert stats["drop_identity_add"] == 1
        assert "promote_scalar_operand" not in stats

    def test_a_non_zero_addend_is_promoted_instead(self):
        class OneAdd(nn.Module):
            def forward(self, x):
                return x + 1.0

        graph, _ = export_and_convert(OneAdd(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["add"]


class TestPromoteScalarOperands:
    """スカラ被演算子は rank-1 定数へ（IR 語彙も attrs も増やさない — recon §4-4）。"""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda x: x - 0.25, ["sub"]),
            (lambda x: x * 3.0, ["mul"]),
            (lambda x: x / 2.0, ["div"]),
        ],
        ids=["sub", "mul", "div"],
    )
    def test_each_binary_op_keeps_its_operand_order(self, build, expected):
        class Scaled(nn.Module):
            def forward(self, x):
                return build(x)

        graph, _ = export_and_convert(Scaled(), (torch.randn(3, 4),))

        assert node_ops(graph) == expected
        assert only_node(graph, expected[0]).ins[0] == "x"

    def test_a_scalar_on_the_left_stays_on_the_left(self):
        """`1 − mask` は非可換 — 定数を右に置くと符号が黙って反転する（mask 経路の実測形）。"""

        class OneMinus(nn.Module):
            def forward(self, mask):
                return 1 - mask

        graph, tensors = export_and_convert(OneMinus(), (torch.ones(3, 4, dtype=torch.int64),))

        node = only_node(graph, "sub")
        assert node.ins[1] == "mask"
        constant = tensors[graph.initializers[node.ins[0]].tensor]
        assert torch.equal(constant, torch.tensor([1], dtype=torch.int32))

    def test_the_constant_takes_the_tensor_dtype(self):
        """i32 側は i32 initializer（二項 op の「入力 dtype は同型」契約 — ADR 0010）。"""

        class Offset(nn.Module):
            def forward(self, mask):
                return mask - 1

        graph, _ = export_and_convert(Offset(), (torch.ones(3, 4, dtype=torch.int64),))

        constant = only_node(graph, "sub").ins[1]
        assert graph.values[constant].dtype == "i32"
        assert graph.initializers[constant].storage.dtype == "i32"

    def test_a_rank0_tensor_operand_is_left_alone(self):
        """`[] × [1] → [1]` は rank が上がる — 昇格すると宣言と実体が黙って食い違う。

        実測でも lifted スカラ定数の `sqrt(c * 3)` がこの形（畳み込みで消える）。
        """

        class ScalarTensor(nn.Module):
            def forward(self, x):
                scalar = x.sum(dim=-1).amax(dim=-1)  # rank-0 テンソル
                return x + (scalar * 3.0)

        decomposed = decompose(ScalarTensor(), (torch.randn(3, 4),))
        stats = normalize_graph(decomposed)

        assert "promote_scalar_operand" not in stats

    def test_alpha_is_left_alone(self):
        """係数付き加算は IR 語彙に無い — 昇格で alpha を黙って落とさない。

        alpha が生き残るかは torch の分解次第なので、残った場合だけ「書き換えていない」
        ことを見る（畳まれてしまう版では前提が成立しないので skip）。
        """

        class Alpha(nn.Module):
            def forward(self, x, y):
                return torch.add(x, y, alpha=2)

        decomposed = decompose(Alpha(), (torch.randn(3, 4), torch.randn(4)))
        graph = decomposed.graph_module.graph
        node = next((n for n in graph.nodes if n.target is aten.add.Tensor), None)
        if node is None or node.kwargs.get("alpha", 1) == 1:
            pytest.skip("この torch では alpha が分解で畳まれる")
        node.args = (node.args[0], 1.0)

        stats = normalize_graph(decomposed)

        assert "promote_scalar_operand" not in stats


class MaskedByEqZero(nn.Module):
    """実測形（`scores.masked_fill(attn_mask == 0, -1e4)`）を縮めたもの — recon §2。

    attn_mask はグラフ入力由来なので定数畳み込みできず、`eq` が実行系に残る。
    """

    def forward(self, scores, mask):
        return scores.masked_fill(mask == 0, -1e4)


class EqNonZero(nn.Module):
    def forward(self, x):
        return x == 2


class TestEqZeroToNotBool:
    """`eq(x, 0)` → `bitwise_not(cast(x, bool))`（ADR 0015 — 新 op を作らない裁定）。

    同値性の根拠は cast 規約「x → bool は x != 0」で、`not (x != 0)` は `x == 0` そのもの。
    """

    def test_the_measured_masked_fill_graph_lowers_without_a_new_op(self):
        graph, _ = export_and_convert(
            MaskedByEqZero(),
            (torch.randn(2, 3), torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.int64)),
        )

        assert node_ops(graph) == ["cast", "bitwise_not", "masked_fill"]
        assert only_node(graph, "cast").attrs == {"to": "bool"}

    def test_the_pass_reports_its_firing_count(self):
        decomposed = decompose(
            MaskedByEqZero(),
            (torch.randn(2, 3), torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.int64)),
        )

        assert normalize_graph(decomposed)["eq_zero->bitwise_not_cast"] == 1

    def test_without_the_pass_eq_is_outside_the_vocabulary(self):
        """パスが載荷であることの確認 — 書き換えなければ未対応 op で落ちる。"""
        decomposed = decompose(
            MaskedByEqZero(),
            (torch.randn(2, 3), torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.int64)),
        )

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.eq.Scalar" in err.value.ops

    def test_a_non_zero_comparand_is_left_alone(self):
        """0 以外との比較には成り立たない書き換えなので、触らず列挙へ回す。"""
        decomposed = decompose(EqNonZero(), (torch.randn(3),))

        stats = normalize_graph(decomposed)

        assert "eq_zero->bitwise_not_cast" not in stats
        with pytest.raises(UnsupportedAtenOpsError):
            convert(decomposed)


class TestDropMetadataAsserts:
    """`_assert_tensor_metadata` は IR に出ない副作用ノード。

    eliminate_dead_code はこれを impure と見なして残すため、消さないと置き換え済みの
    部分木が assert 経由で生き残る。
    """

    def test_assert_nodes_are_removed(self):
        decomposed = decompose(NoOpCast(), (torch.randn(3, 4),))
        graph = decomposed.graph_module.graph
        before = [
            node for node in graph.nodes if node.target is aten._assert_tensor_metadata.default
        ]
        assert before, "前提: 分解後に assert ノードが在ること"

        stats = normalize_graph(decomposed)

        assert stats["drop_metadata_assert"] == len(before)
        assert not [
            node for node in graph.nodes if node.target is aten._assert_tensor_metadata.default
        ]


class Split(nn.Module):
    """ConvFlow / ResidualCoupling の `torch.split(x, [half]*2, 1)`（recon §2）の縮小形。"""

    def forward(self, x):
        first, second = torch.split(x, [2, 3], 1)
        return first * 2.0, second


class TestSplitToSlices:
    """split_with_sizes + getitem → slice の列（IR に多出力 op は無い — ADR 0014）。"""

    def test_each_getitem_becomes_its_own_slice(self):
        graph, _ = export_and_convert(Split(), (torch.randn(1, 5, 4),))

        assert node_ops(graph) == ["slice", "slice", "mul"]
        # 区間は sizes の累積そのもの（境界を 1 つずらすと区間が重なる / 隙間ができる）
        assert graph.nodes[0].attrs == {"dim": 1, "start": 0, "end": 2}
        assert graph.nodes[1].attrs == {"dim": 1, "start": 2, "end": 5}
        assert graph.nodes[0].ins == ["x"] and graph.nodes[1].ins == ["x"]

    def test_the_pass_reports_its_firing_count(self):
        decomposed = decompose(Split(), (torch.randn(1, 5, 4),))

        # 取り出し口の本数ぶん（getitem 1 本 = slice 1 本）
        assert normalize_graph(decomposed)["split_with_sizes->slice"] == 2

    def test_without_the_pass_split_is_outside_the_vocabulary(self):
        """パスが載荷であることの確認 — 書き換えなければ未対応 op で落ちる。"""
        decomposed = decompose(Split(), (torch.randn(1, 5, 4),))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        # split 自身の meta はテンソルでない（タプル）ので列挙に出るのは取り出し口の getitem。
        assert [name for names in err.value.ops.values() for name in names] == [
            "getitem",
            "getitem_1",
        ]


# ---- M1-P4 波 2（ADR 0016）— rms fold / ガード除去 / rank 下げ / select ------


class HandWrittenRmsNorm(nn.Module):
    """Qwen3 / DiT と同じ手書き分解形（preserve では畳めず FX 照合が要る）。"""

    EPS = 1e-6

    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, hidden))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.EPS) * self.weight


class BroadcastRmsNorm(HandWrittenRmsNorm):
    """weight が最終次元長の rank1 でない形（契約と別の意味論になるので畳まない）。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, 3).reshape(3, 1))


class TestFoldRmsNorm:
    def test_the_decomposed_form_becomes_one_rms_norm_node(self):
        graph, _ = export_and_convert(HandWrittenRmsNorm(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["rms_norm"]
        assert only_node(graph, "rms_norm").attrs == {"eps": HandWrittenRmsNorm.EPS}

    def test_it_runs_before_the_pow2_rewrite(self):
        """順序 MUST（ADR 0016）— `pow2->mul` が先に走るとパターンが 1 つ減る。

        rms fold が先なら `pow(x,2)` は rms_norm ごと消えるので、`pow2->mul` の発火は 0 本。
        逆順だと mul(x,x) が残り、統計に 1 本立つ（= 順序が入れ替わったことが数に出る）。
        """
        decomposed = decompose(HandWrittenRmsNorm(), (torch.randn(3, 4),))

        stats = normalize_graph(decomposed)

        assert stats["rms_norm"] == 1
        assert "pow2->mul" not in stats

    def test_without_the_pass_the_decomposition_is_outside_the_vocabulary(self):
        decomposed = decompose(HandWrittenRmsNorm(), (torch.randn(3, 4),))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.mean.dim" in err.value.ops
        assert "aten.rsqrt.default" in err.value.ops

    def test_a_broadcast_weight_is_left_in_the_enumeration(self):
        """契約は「正規化長の正本 = weight の長さ」— broadcast 形は畳まず落とす。"""
        with pytest.raises(UnsupportedAtenOpsError) as err:
            export_and_convert(BroadcastRmsNorm(), (torch.randn(3, 3),))

        assert "aten.mean.dim" in err.value.ops


class CausalAttention(nn.Module):
    """SDPA の因果マスク経路（safe-softmax ガードが立つ最小形）。"""

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


class FullyMaskedAttention(nn.Module):
    """全要素 -inf のマスク（ガードが**本当に効く**形 — 除去すると NaN が下流へ流れる）。"""

    def forward(self, q, k, v):
        bias = torch.full((1, 1, 4, 4), float("-inf"))
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)


class BufferMaskedAttention(nn.Module):
    """マスクを `register_buffer` で持つ SDPA。

    buffer は torch.export で **placeholder に持ち上がる**ので、「placeholder は有限」と
    決め打つ判定はこの形を素通りさせる（-inf 行があってもガードを消す = NaN が下流へ流れる）。
    """

    def __init__(self, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("bias", bias)

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.bias)


ATTENTION_ARGS = (torch.randn(1, 1, 4, 3), torch.randn(1, 1, 4, 3), torch.randn(1, 1, 4, 3))


class TestDropSafeSoftmaxGuard:
    def test_a_causal_mask_lets_the_guard_be_removed(self):
        args = (torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3))
        decomposed = decompose(CausalAttention(), args)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:masked-rows-nonempty"] == 1
        assert not any(
            node.target in (aten.any.dim, aten.full_like.default, aten.logical_not.default)
            for node in decomposed.graph_module.graph.nodes
            if node.op == "call_function"
        )

    def test_without_the_pass_the_guard_ops_are_outside_the_vocabulary(self):
        args = (torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3))
        decomposed = decompose(CausalAttention(), args)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.any.dim" in err.value.ops
        assert "aten.full_like.default" in err.value.ops

    def test_an_all_masked_row_refuses_the_removal(self):
        """不活性の証明が立たない形は fail loudly（消すと NaN が下流へ流れる）。"""
        decomposed = decompose(FullyMaskedAttention(), ATTENTION_ARGS)

        with pytest.raises(NotImplementedError, match="不活性でない"):
            normalize_graph(decomposed)

    def test_an_all_masked_row_in_a_buffer_refuses_the_removal(self):
        """-inf 源が **placeholder（buffer）** でも実値で見る。

        「placeholder は有限」と決め打つ判定はこの形を素通りさせ、ガードが消えて NaN が
        下流へ流れる（グラフ内 `torch.full` の形だけを見ていると空振りする穴）。
        """
        module = BufferMaskedAttention(torch.full((1, 1, 4, 4), float("-inf")))
        decomposed = decompose(module, ATTENTION_ARGS)

        with pytest.raises(NotImplementedError, match="不活性でない"):
            normalize_graph(decomposed)

    def test_a_finite_buffer_mask_still_lets_the_guard_go(self):
        """実値が有限な buffer マスクは従来どおり除去できる（判定が過剰にならないこと）。"""
        decomposed = decompose(BufferMaskedAttention(torch.zeros(1, 1, 4, 4)), ATTENTION_ARGS)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:no-neg-inf"] == 1


class RepeatKv(nn.Module):
    """GQA の kv 複製（`unsqueeze → expand → view` が rank5 を作る唯一の形）。"""

    def forward(self, x):
        batch, groups, length, head = x.shape
        wide = x[:, :, None, :, :].expand(batch, groups, 3, length, head)
        return wide.reshape(batch, groups * 3, length, head)


class RopeUnbind(nn.Module):
    """`use_real_unbind_dim=-2` の RoPE 分割（rank5 の幅 1 slice + squeeze）。"""

    def forward(self, x):
        batch, heads, length, dim = x.shape
        split = x.reshape(batch, heads, length, 2, dim // 2)
        first = split[..., 0:1, :].squeeze(-2)
        second = split[..., 1:2, :].squeeze(-2)
        return torch.cat([-second, first], dim=-1)


class Rank5NotUnbind(nn.Module):
    """最終次元を割る rank5 の view だが、消費者が unbind ではない形（reshape → mul → reshape）。

    `_lower_split_unbind` は形の入口（最終次元を (A,B) に割る view）だけで掴むので、この形も
    一度は掴む。unbind でないと分かった時点で**書き換えずスキップ**するのが正しい。
    """

    def forward(self, x):
        batch, heads, length, dim = x.shape
        split = x.reshape(batch, heads, length, 2, dim // 2)
        return (split * 2.0).reshape(batch, heads, length, dim)


class Patchify(nn.Module):
    """patchify（rank8 の reshape → permute → reshape）。"""

    def forward(self, x):
        batch, channels, height, width = x.shape
        patches = x.reshape(batch, channels, 1, 1, height // 2, 2, width // 2, 2)
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return patches.reshape(batch, (height // 2) * (width // 2), channels * 4)


class TestRankLowering:
    @pytest.mark.parametrize(
        ("module", "args", "key"),
        [
            pytest.param(
                RepeatKv(), (torch.randn(1, 2, 4, 3),), "unit_expand->rank3", id="unit expand"
            ),
            pytest.param(
                RopeUnbind(), (torch.randn(1, 2, 4, 6),), "split_unbind->slice", id="split unbind"
            ),
            pytest.param(
                Patchify(), (torch.randn(1, 3, 4, 6),), "reshape_permute", id="reshape permute"
            ),
        ],
    )
    def test_the_rewritten_graph_returns_the_same_numbers(self, module, args, key):
        """rank 下げは同値書き換え — 書き換え後のグラフを実行して eager と突き合わせる。"""
        decomposed = decompose(module, args)

        stats = normalize_graph(decomposed)
        with torch.no_grad():
            got = decomposed.module()(*args)
            expected = module(*args)

        assert stats[key] >= 1
        assert torch.equal(got, expected)

    @pytest.mark.parametrize(
        ("module", "args"),
        [
            pytest.param(RepeatKv(), (torch.randn(1, 2, 4, 3),), id="unit expand"),
            pytest.param(RopeUnbind(), (torch.randn(1, 2, 4, 6),), id="split unbind"),
            pytest.param(Patchify(), (torch.randn(1, 3, 4, 6),), id="reshape permute"),
        ],
    )
    def test_every_value_fits_the_strided_rank_after_the_pass(self, module, args):
        """狙いは rank ≤ STRIDED_RANK に収めること（収まらないと convert が落ちる）。"""
        graph, _ = export_and_convert(module, args)

        assert max(len(value.shape) for value in graph.values.values()) <= STRIDED_RANK

    def test_a_non_unbind_consumer_is_skipped_not_rejected(self):
        """unbind でない rank5 の view は**素通し**（照合完了前にグラフは 1 ノードも変わらない）。

        ここで落とすと、他のパスで rank ≤ 4 に落ちうる形まで正規化の途中で殺す。本当に
        落ちない形が残れば convert 側の門（strided 族の rank 上限）が全件列挙で落とす。
        """
        args = (torch.randn(1, 2, 4, 6),)
        decomposed = decompose(Rank5NotUnbind(), args)

        stats = normalize_graph(decomposed)

        assert "split_unbind->slice" not in stats
        with torch.no_grad():
            assert torch.equal(decomposed.module()(*args), Rank5NotUnbind()(*args))
        # 変換もできる（rank 上限が効くのは strided コピー族だけで、elementwise は rank 自由）
        graph, _ = export_and_convert(Rank5NotUnbind(), args)
        assert node_ops(graph) == ["reshape", "mul", "reshape"]

    def test_a_rank4_graph_is_left_untouched(self):
        """発火条件は rank > STRIDED_RANK — 既存グラフ（rank ≤ 4）へ誤爆しない。"""

        class Rank4(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1, 3).reshape(1, 4, 6)

        decomposed = decompose(Rank4(), (torch.randn(1, 2, 3, 4),))

        stats = normalize_graph(decomposed)

        assert not any(
            key.startswith(("unit_expand", "split_unbind", "reshape_permute")) for key in stats
        )


class SelectUnitAxis(nn.Module):
    def forward(self, x):
        return torch.select(x, 1, 0) * 2.0


class TestSelectToSqueeze:
    def test_selecting_a_unit_axis_becomes_a_reshape(self):
        graph, _ = export_and_convert(SelectUnitAxis(), (torch.randn(2, 1, 3),))

        assert node_ops(graph) == ["reshape", "mul"]
        assert graph.values[graph.nodes[0].outs[0]].shape == [2, 3]

    def test_selecting_a_longer_axis_stays_in_the_enumeration(self):
        """長さ 2 以上の軸の select は squeeze と同値でない — 書き換えず全件列挙に回す。"""

        class SelectRow(nn.Module):
            def forward(self, x):
                return torch.select(x, 1, 0) * 2.0

        with pytest.raises(UnsupportedAtenOpsError) as err:
            export_and_convert(SelectRow(), (torch.randn(2, 3, 4),))

        assert "aten.select.int" in err.value.ops

    def test_selecting_a_symbolic_axis_is_rejected(self, dyn_t):
        """記号次元の軸はヒント値で長さを採ると無言の誤値になる — ここで止める。"""

        class SelectSymbolic(nn.Module):
            def forward(self, x):
                return torch.select(x, 0, 0) * 2.0

        decomposed = torch.export.export(
            SelectSymbolic(), (torch.randn(4, 3),), dynamic_shapes=({0: dyn_t},), strict=False
        ).run_decompositions(curated_decompositions())

        with pytest.raises(NotImplementedError, match="記号次元の軸を切る select"):
            normalize_graph(decomposed)
