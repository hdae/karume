"""aten op 対応表（aten_handlers.py）の振る舞い固定。

個別の aten op と融合 op が IR の語彙へ狙いどおりに落ちること、attrs が torch の意味論を
境界で明示化していること、契約に無い形が fail loudly になることを固定する。
エンジン側（convert.py）の契約面は test_convert.py。
"""

from __future__ import annotations

import pytest
import torch
from conftest import SYM_MAX, node_ops, only_node
from torch import nn
from torch.export import Dim
from torchvision.ops import deform_conv2d

from karume.aten_handlers import ATEN_HANDLERS
from karume.convert import UnsupportedAtenOpsError
from karume.ops import EMITTABLE_OPS


def _gelu_tanh_reference(x: torch.Tensor) -> torch.Tensor:
    """`gelu_tanh` の参照実装（TS 側 packages/runtime/src/reference/ops.ts の鏡像）。

    0.7978845608028654 = √(2/π)。erf 形と違いこの式そのものが定義なので、torch と別式に
    書き換える余地は無い（突合が見るのは近似の外れではなく**式の取り違え**）。
    """
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class TestGelu:
    """gelu は分解を止めて 1 ノードで運ぶ（core 分解は M0 語彙の外に散る）。

    近似種別は attrs ではなく **op 名**で分ける（契約が attrs 空 — 同じ op 名のまま数値だけ
    変わる分岐を契約の外に作らない）。
    """

    def test_exact_gelu_stays_a_single_node(self, convert_module):
        class Gelu(nn.Module):
            def forward(self, x):
                return nn.functional.gelu(x)

        graph, _ = convert_module(Gelu(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["gelu"]

    def test_tanh_approximation_becomes_its_own_op(self, convert_module):
        class GeluTanh(nn.Module):
            def forward(self, x):
                return nn.functional.gelu(x, approximate="tanh")

        graph, _ = convert_module(GeluTanh(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["gelu_tanh"]

    def test_reference_matches_torch_on_random_and_edge_values(self):
        generator = torch.Generator().manual_seed(20260811)
        random = torch.randn(4096, generator=generator) * 3.0
        # ±0 / ±大値（x³ が f32 で inf へ飛ぶ域を含む）/ ±小値（x³ が 0 へ落ちる域）。
        edges = torch.tensor(
            [0.0, -0.0, 3.4e38, -3.4e38, 1e20, -1e20, 1e-20, -1e-20, 20.0, -20.0],
            dtype=torch.float32,
        )
        for name, x in (("randn", random), ("edges", edges)):
            expected = nn.functional.gelu(x, approximate="tanh")
            torch.testing.assert_close(_gelu_tanh_reference(x), expected, msg=name)

    def test_two_gelus_are_numerically_distinguishable(self):
        # 同じ op へ畳めないことの根拠（畳むと 1e-3 級の誤差が黙って入る）。
        x = torch.linspace(-3.0, 3.0, 101)
        gap = (nn.functional.gelu(x) - _gelu_tanh_reference(x)).abs().max()
        assert gap > 1e-4


class TestSin:
    """`sin` は畳み込みと発行の 2 経路を**同時に**持つ（ADR 0043 の第 1 層）。

    定数部分木（RoPE の位置表）は従来どおり FOLDABLE_OPS が initializer へ畳み、実行時値を
    取る形だけが IR ノードとして残る。片方を消すともう片方が黙って劣化する — 定数を畳まなく
    なれば実行時の dispatch が増え、発行できなければ Snake 活性が export で落ちる。
    """

    def test_runtime_valued_sin_becomes_a_node(self, convert_module):
        class Sin(nn.Module):
            def forward(self, x):
                return torch.sin(x)

        graph, _ = convert_module(Sin(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["sin"]

    def test_constant_sin_is_still_folded_into_an_initializer(self, convert_module):
        """定数入力の `sin`（RoPE 表の形）はノードを残さず焼かれる。"""

        class ConstantTable(nn.Module):
            def forward(self, x):
                return x + torch.sin(torch.arange(4, dtype=torch.float32))

        graph, tensors = convert_module(ConstantTable(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["add"]
        folded = only_node(graph, "add").ins[1]
        torch.testing.assert_close(
            tensors[graph.initializers[folded].tensor],
            torch.sin(torch.arange(4, dtype=torch.float32)),
        )

    def test_snake_activation_converts_with_a_single_sin(self, convert_module):
        """DACVAE の Snake `x + sin²(αx)/(α+1e-9)`（この op を足した動機そのもの）。

        α は学習パラメータ = initializer なので `α+1e-9` も実行時ノードになる（パラメータ
        経由の畳み込みは不適格 — `_classify_foldable`）。`sin` は実行時値 `αx` を取るため
        1 ノードとして残り、`pow(2)` は分解で `mul` になる。
        NOTE: 係数を `(α+1e-9).reciprocal()` と書くと `aten.reciprocal.default` に当たる
        （FOLDABLE 専用で emit 経路が無い）。同値な除算形で書いてある。
        """

        class Snake(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = nn.Parameter(torch.full((1, 5, 1), 0.7))

            def forward(self, x):
                return x + torch.sin(self.alpha * x).pow(2) / (self.alpha + 1e-9)

        graph, _ = convert_module(Snake(), (torch.randn(1, 5, 8),))

        assert node_ops(graph) == ["mul", "sin", "mul", "add", "div", "add"]
        assert set(node_ops(graph)) <= EMITTABLE_OPS


class TestRowReduce:
    """reduce は 1 軸・keepdim 無しのみ。軸は attrs `dim` に**非負で**載る。"""

    def test_last_dim_reductions_convert(self, convert_module):
        class Reduce(nn.Module):
            def forward(self, x):
                return x.sum(dim=-1), x.amax(dim=-1), x.amin(dim=-1)

        graph, _ = convert_module(Reduce(), (torch.randn(3, 5),))

        assert node_ops(graph) == ["sum", "amax", "amin"]
        assert [graph.values[name].shape for name in graph.outputs] == [[3], [3], [3]]
        # 負の軸表記は境界で正規化する（IR の attrs は非負のみ）
        assert [node.attrs["dim"] for node in graph.nodes] == [1, 1, 1]

    def test_a_non_last_axis_is_carried_in_attrs(self, convert_module):
        """チャネル軸の縮約（VAE の L2 正規化）は permute 無しでそのまま書ける。"""

        class Reduce(nn.Module):
            def forward(self, x):
                return torch.sum(x * x, dim=1)

        graph, _ = convert_module(Reduce(), (torch.randn(2, 6, 3, 4),))

        assert "permute" not in node_ops(graph)
        assert only_node(graph, "sum").attrs == {"dim": 1}
        assert [graph.values[name].shape for name in graph.outputs] == [[2, 3, 4]]

    @pytest.mark.parametrize(
        ("build", "why"),
        [
            (lambda x: x.sum(dim=(0, 1)), "の sum は未対応（1 軸ずつ）"),
            (lambda x: x.sum(dim=-1, keepdim=True), "keepdim=True の sum は未対応"),
            (lambda x: x.amax(dim=(0, 1)), "の amax は未対応（1 軸ずつ）"),
            (lambda x: x.amin(dim=-1, keepdim=True), "keepdim=True の amin は未対応"),
        ],
        ids=["sum-multi-axis", "sum-keepdim", "amax-multi-axis", "amin-keepdim"],
    )
    def test_other_reductions_are_rejected(self, convert_module, build, why):
        class Reduce(nn.Module):
            def forward(self, x):
                return build(x)

        with pytest.raises(NotImplementedError, match=why):
            convert_module(Reduce(), (torch.randn(3, 5),))


class TestArgmax:
    """argmax は**最終次元 + keepdim=True** の固定形のみ（ADR 0068 決定 2）。

    reduce 族と違い attrs を持たない（軸は最終次元固定・rank 保存）ので、絞りは全て
    ハンドラの受理判定が担う。出力は torch の i64 → 境界で **i32** へ正規化される
    （ADR 0009）。
    """

    def test_last_dim_with_keepdim_becomes_one_node(self, convert_module):
        class Argmax(nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=-1, keepdim=True)

        graph, _ = convert_module(Argmax(), (torch.randn(3, 5),))

        assert node_ops(graph) == ["argmax"]
        # attrs 空（`dim` / `keepdim` の欄が無いことがそのまま「他の形は語彙に無い」の宣言）
        assert only_node(graph, "argmax").attrs == {}
        # rank 保存 = 最終次元を 1 に潰した形・dtype は i32
        assert [graph.values[name].shape for name in graph.outputs] == [[3, 1]]
        assert [graph.values[name].dtype for name in graph.outputs] == ["i32"]

    def test_a_symbolic_leading_dimension_survives(self, convert_module, dyn_t):
        """decode の出口形（`[B, T, V]` の最終次元 = 語彙軸）。先行次元は記号のまま残る。"""

        class Argmax(nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=2, keepdim=True)

        graph, _ = convert_module(Argmax(), (torch.randn(1, 4, 7),), dynamic_shapes=({1: dyn_t},))

        assert node_ops(graph) == ["argmax"]
        assert [graph.values[name].shape for name in graph.outputs] == [[1, "T", 1]]

    @pytest.mark.parametrize(
        ("build", "why"),
        [
            (lambda x: torch.argmax(x, dim=-1), "keepdim=False の argmax は未対応"),
            (lambda x: torch.argmax(x), "全次元 argmax は未対応"),
            (lambda x: torch.argmax(x, dim=0, keepdim=True), "最終次元以外（dim=0 / rank=2）"),
        ],
        ids=["no-keepdim", "all-dims", "non-last-axis"],
    )
    def test_other_forms_are_rejected(self, convert_module, build, why):
        class Argmax(nn.Module):
            def forward(self, x):
                return build(x)

        with pytest.raises(NotImplementedError, match=why):
            convert_module(Argmax(), (torch.randn(3, 5),))

    def test_kwarg_forms_read_the_same_slots(self):
        """`dim` / `keepdim` が kwargs で来ても同じ欄として読む（reduce 族と同じ規律）。

        位置引数を素で添字すると `IndexError` になり、ADR 0005 の「未対応は診断つきで落とす」
        規律の外へ落ちる（診断が「エクスポータが壊れた」に化ける）。
        """
        target = torch.ops.aten.argmax.default
        node = _reduce_node(target, kwargs={"dim": -1, "keepdim": True})

        assert ATEN_HANDLERS[target](node) == ("argmax", 1, {}, ())

        with pytest.raises(NotImplementedError, match="keepdim=False の argmax は未対応"):
            ATEN_HANDLERS[target](_reduce_node(target, kwargs={"dim": -1}))
        with pytest.raises(NotImplementedError, match="全次元 argmax は未対応"):
            ATEN_HANDLERS[target](_reduce_node(target, kwargs={"keepdim": True}))


def _reduce_node(target, args=(), kwargs=None, shape=(3, 5)):
    """reduce ハンドラへ直接食わせる FX ノードを 1 本組む。

    torch 2.13 の分解は reduce の `dim` を必ず位置引数で出すので、**kwargs 形は export
    からは作れない**（overload の呼び分け次第で来うる形）。ハンドラは純関数なので、
    グラフを組む代わりにノード 1 本を手で組んで直接呼ぶ。
    """
    graph = torch.fx.Graph()
    src = graph.placeholder("x")
    src.meta["val"] = torch.zeros(shape)
    return graph.call_function(target, (src, *args), dict(kwargs or {}))


class TestReduceArgumentsInKwargs:
    """reduce 族は `dim` / `keepdim` / `dtype` が kwargs で来ても同じ欄として読む。

    位置引数を素で添字すると `IndexError` になり、ADR 0005 の「未対応は診断つきで落とす」
    規律の外へ落ちる（診断が「エクスポータが壊れた」に化ける）。
    """

    @pytest.mark.parametrize(
        ("target", "op"),
        [
            (torch.ops.aten.sum.dim_IntList, "sum"),
            (torch.ops.aten.amax.default, "amax"),
            (torch.ops.aten.amin.default, "amin"),
        ],
    )
    def test_a_kwarg_dim_is_read_as_the_same_slot(self, target, op):
        node = _reduce_node(target, kwargs={"dim": [-1]})

        assert ATEN_HANDLERS[target](node) == (op, 1, {"dim": 1}, ())

    @pytest.mark.parametrize(
        ("target", "kwargs", "why"),
        [
            (torch.ops.aten.sum.dim_IntList, {"dim": None}, "全次元 sum は未対応"),
            (torch.ops.aten.sum.dim_IntList, {"dim": [0, 1]}, r"複数軸 \[0, 1\] の sum は未対応"),
            (torch.ops.aten.sum.dim_IntList, {"dim": [-1], "keepdim": True}, "keepdim=True の sum"),
            (
                torch.ops.aten.sum.dim_IntList,
                {"dim": [-1], "dtype": torch.float32},
                "dtype 指定付きの sum は未対応",
            ),
            (torch.ops.aten.amax.default, {}, "全次元 amax は未対応"),
            (torch.ops.aten.amax.default, {"dim": [0, 1]}, r"複数軸 \[0, 1\] の amax は未対応"),
            (torch.ops.aten.amin.default, {"dim": [-1], "keepdim": True}, "keepdim=True の amin"),
        ],
        ids=[
            "sum-all-dims",
            "sum-multi-axis",
            "sum-keepdim",
            "sum-dtype",
            "amax-dim-omitted",
            "amax-multi-axis",
            "amin-keepdim",
        ],
    )
    def test_unsupported_kwarg_forms_keep_their_diagnostic(self, target, kwargs, why):
        node = _reduce_node(target, kwargs=kwargs)

        with pytest.raises(NotImplementedError, match=why):
            ATEN_HANDLERS[target](node)


class TestCastAndBitwiseNot:
    """aten._to_copy → cast / aten.bitwise_not → bitwise_not（ADR 0009）。"""

    def test_to_copy_becomes_cast_with_the_target_dtype(self, convert_module):
        class Truncate(nn.Module):
            def forward(self, x):
                return x.to(torch.int64)

        graph, _ = convert_module(Truncate(), (torch.randn(3, 4),))

        node = only_node(graph, "cast")
        # i64 は境界正規化で i32 として宣言される（torch dtype をそのまま名前にしない）
        assert node.attrs == {"to": "i32"}
        assert graph.values[node.outs[0]].dtype == "i32"

    def test_casts_with_different_targets_are_not_merged_by_cse(self, convert_module):
        """CSE キーに attrs が載っていないと、同じ入力の片方が黙って消える。"""

        class TwoCasts(nn.Module):
            def forward(self, x):
                return x.to(torch.bool), x.to(torch.int64)

        graph, _ = convert_module(TwoCasts(), (torch.randn(3, 4),))

        targets = sorted(node.attrs["to"] for node in graph.nodes if node.op == "cast")
        assert targets == ["bool", "i32"]

    def test_bitwise_not_on_bool_is_emitted(self, convert_module):
        class Invert(nn.Module):
            def forward(self, mask):
                return torch.bitwise_not(mask.to(torch.bool))

        graph, _ = convert_module(Invert(), (torch.ones(3, 4, dtype=torch.int64),))

        node = only_node(graph, "bitwise_not")
        assert graph.values[node.outs[0]].dtype == "bool"

    def test_integer_bitwise_not_is_rejected(self, convert_module):
        """整数の ~x は bool の否定と意味が違う — 黙って読み替えない。"""

        class InvertInts(nn.Module):
            def forward(self, x):
                return torch.bitwise_not(x)

        with pytest.raises(NotImplementedError, match="bitwise_not は未対応"):
            convert_module(InvertInts(), (torch.ones(3, 4, dtype=torch.int64),))


class TestBmmAndGather:
    """bmm（rank-3 専業）と gather（最終次元固定）— ADR 0012。"""

    def test_bmm_keeps_the_batch_axis(self, convert_module, dyn_t):
        class Batched(nn.Module):
            def forward(self, x, y):
                return torch.bmm(x, y)

        # B / M / K / N を全て違う長さにする（軸の取り違えを shape 宣言から読める形）
        graph, _ = convert_module(
            Batched(),
            (torch.randn(2, 6, 3), torch.randn(2, 3, 5)),
            ({1: dyn_t}, None),
        )

        node = only_node(graph, "bmm")
        assert node.attrs == {}
        assert graph.values[node.outs[0]].shape == [2, "T", 5]

    def test_rank2_matmul_stays_matmul(self, convert_module):
        """rank-2 は matmul の担当（bmm と兼用にしない — 契約表）。"""

        class Flat(nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        graph, _ = convert_module(Flat(), (torch.randn(6, 3), torch.randn(3, 5)))

        assert node_ops(graph) == ["matmul"]

    def test_rank4_matmul_is_rejected(self, convert_module):
        """rank-4 の matmul は aten.bmm へ reshape 付きで落ちるか、語彙外として落ちる。

        どちらにせよ「rank-4 のまま bmm に化ける」ことは無い（rank-3 だけを受理する）。
        """

        class Deep(nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        graph, _ = convert_module(Deep(), (torch.randn(2, 3, 6, 4), torch.randn(2, 3, 4, 5)))

        for node in graph.nodes:
            if node.op != "bmm":
                continue
            assert len(graph.values[node.outs[0]].shape) == 3

    def test_gather_on_the_last_dim_is_emitted(self, convert_module, dyn_t):
        class Picking(nn.Module):
            def forward(self, src, index):
                return torch.gather(src, -1, index)

        index = torch.tensor([[0, 2, 1], [2, 2, 0], [1, 0, 2]] * 2)
        graph, _ = convert_module(
            Picking(),
            (torch.randn(6, 3), index),
            ({0: dyn_t}, {0: dyn_t}),
        )

        node = only_node(graph, "gather")
        assert node.attrs == {}
        # src は f32、index は i32（i64 からの境界正規化）— スロットで dtype が違う
        assert [graph.inputs[0].dtype, graph.inputs[1].dtype] == ["f32", "i32"]
        assert graph.values[node.outs[0]].dtype == "f32"

    def test_gather_on_a_non_last_dim_is_rejected(self, convert_module):
        class ByRow(nn.Module):
            def forward(self, src, index):
                return torch.gather(src, 0, index)

        with pytest.raises(NotImplementedError, match="最終次元以外"):
            convert_module(
                ByRow(),
                (torch.randn(3, 3), torch.tensor([[0, 1, 2], [2, 1, 0], [1, 1, 1]])),
            )

    def test_gather_with_a_shorter_leading_dim_is_rejected(self, convert_module):
        """torch は `index.size(d) <= src.size(d)` を許すが IR 契約は完全一致
        （packages/runtime/src/ops.ts）。

        rank だけ見て通すと export は緑のままブラウザ側の契約検査で落ちる（`row = i / J` の
        行対応が成り立たない形）。
        """

        class Narrow(nn.Module):
            def forward(self, src, index):
                return torch.gather(src, -1, index)

        with pytest.raises(NotImplementedError, match="先行次元"):
            convert_module(
                Narrow(),
                (torch.randn(4, 3), torch.tensor([[0, 1, 2], [2, 1, 0]])),
            )

    def test_gather_with_sparse_grad_is_rejected(self, convert_module):
        class Sparse(nn.Module):
            def forward(self, src, index):
                return torch.gather(src, -1, index, sparse_grad=True)

        with pytest.raises(NotImplementedError, match="sparse_grad"):
            convert_module(
                Sparse(),
                (torch.randn(3, 3), torch.tensor([[0, 1, 2], [2, 1, 0], [1, 1, 1]])),
            )


class TestMathOps:
    """M1-P3 波3 の数理 op（sdp の spline / dec の leaky_relu — recon §2）。

    固定するのは 3 つ: ① 実測形が狙った 1 ノードへ落ちること ② attrs が torch の意味論を
    **境界で明示化**していること ③ 契約に無い形（片側 clamp・整数の論理積・最終次元以外の
    cumsum・スカラ被演算子）が fail loudly になること。
    """

    def test_the_inside_test_lowers_to_compare_and_bitwise_and(self, convert_module):
        """spline の `inside = (x >= -b) & (x <= b)`（recon §2）。"""

        class Inside(nn.Module):
            def forward(self, x):
                return (x >= -1.0) & (x <= 1.0)

        graph, _ = convert_module(Inside(), (torch.randn(4),))

        assert node_ops(graph) == ["ge_scalar", "le_scalar", "bitwise_and"]
        assert only_node(graph, "ge_scalar").attrs == {"value": -1.0}
        assert only_node(graph, "le_scalar").attrs == {"value": 1.0}
        assert graph.values[only_node(graph, "bitwise_and").outs[0]].dtype == "bool"

    def test_searchsorted_lowers_to_ge_tensor_and_a_bool_sum(self, convert_module):
        """`sum(x[…,None] >= bl, dim=-1)`（recon §2）— bool の行 sum は i32 のカウント。"""

        class Searchsorted(nn.Module):
            def forward(self, x, bounds):
                return torch.sum(x.unsqueeze(-1) >= bounds, dim=-1)

        graph, _ = convert_module(Searchsorted(), (torch.randn(4), torch.rand(4, 3)))

        assert node_ops(graph) == ["reshape", "ge", "sum"]
        assert graph.values[only_node(graph, "ge").outs[0]].dtype == "bool"
        assert graph.values[only_node(graph, "sum").outs[0]].dtype == "i32"

    def test_gt_scalar_is_emitted_with_its_comparand(self, convert_module):
        class Gate(nn.Module):
            def forward(self, x):
                return x > 0.25

        graph, _ = convert_module(Gate(), (torch.randn(4),))

        assert only_node(graph, "gt_scalar").attrs == {"value": 0.25}

    def test_integer_comparison_is_rejected(self, convert_module):
        """契約は f32 専業（整数の比較は実測に無い — 黙って f32 に読み替えない）。"""

        class IntGate(nn.Module):
            def forward(self, x):
                return x >= 1

        with pytest.raises(NotImplementedError, match="ge_scalar は未対応"):
            convert_module(IntGate(), (torch.ones(4, dtype=torch.int64),))

    def test_clamp_carries_both_bounds(self, convert_module):
        class Clamped(nn.Module):
            def forward(self, x):
                return torch.clamp(x, -0.75, 0.5)

        graph, _ = convert_module(Clamped(), (torch.randn(4),))

        assert only_node(graph, "clamp").attrs == {"min": -0.75, "max": 0.5}

    def test_a_lower_bound_only_clamp_becomes_clamp_min(self, convert_module):
        """下限だけの clamp は **別 op**（clamp_min）へ落ちる（ADR 0017）。

        欠けた上限を +有限最大値で補うのは「表現が無い形を黙って別の形で実行する」ことに
        なるので採らない — 語彙に片側 op を足したのが ADR 0017 の裁定。
        """

        class LowerBound(nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-1.0)

        graph, _ = convert_module(LowerBound(), (torch.randn(4),))

        assert node_ops(graph) == ["clamp_min"]
        assert only_node(graph, "clamp_min").attrs == {"min": -1.0}

    def test_an_upper_bound_only_clamp_is_rejected(self, convert_module):
        """上限だけの形は語彙に無い（clamp_max は未実装 — 需要が出たら同じ手筋で足す）。"""

        class UpperBound(nn.Module):
            def forward(self, x):
                return torch.clamp(x, max=1.0)

        with pytest.raises(NotImplementedError, match="上限だけの clamp は未対応"):
            convert_module(UpperBound(), (torch.randn(4),))

    def test_inverted_clamp_bounds_are_rejected(self, convert_module):
        """min > max は WGSL の clamp が未定義になる向き（契約表と同じ門）。"""

        class Inverted(nn.Module):
            def forward(self, x):
                return torch.clamp(x, 1.0, -1.0)

        with pytest.raises(NotImplementedError, match="clamp は未対応"):
            convert_module(Inverted(), (torch.randn(4),))

    def test_log1p_stays_a_single_node(self, convert_module):
        """softplus 分解形の `log1p(exp(scaled))`（recon §2）。"""

        class Softplus(nn.Module):
            def forward(self, x):
                return torch.log1p(torch.exp(x))

        graph, _ = convert_module(Softplus(), (torch.randn(4),))

        assert node_ops(graph) == ["exp", "log1p"]

    def test_where_takes_three_tensors(self, convert_module):
        class Pick(nn.Module):
            def forward(self, cond, a, b):
                return torch.where(cond > 0.0, a, b)

        graph, _ = convert_module(Pick(), (torch.randn(4), torch.randn(4), torch.randn(4)))

        node = only_node(graph, "where")
        assert node.attrs == {}
        assert len(node.ins) == 3
        assert graph.values[node.outs[0]].dtype == "f32"

    def test_a_scalar_branch_arrives_as_a_constant_tensor(self, convert_module):
        """契約は attrs 空のアリティ 3 固定でスカラを載せる欄が無いが、torch 側の分解が
        `scalar_tensor` を挟むため、実際には rank-0 の定数として届く（畳み込みで
        initializer 化される）。IR 語彙を増やさずに通る形なのでそのまま受理する。
        """

        class ScalarBranch(nn.Module):
            def forward(self, cond, a):
                return torch.where(cond > 0.0, a, 0.0)

        graph, _ = convert_module(ScalarBranch(), (torch.randn(4), torch.randn(4)))

        node = only_node(graph, "where")
        assert len(node.ins) == 3
        assert node.ins[2] in graph.initializers
        assert graph.values[node.ins[2]].shape == []

    def test_integer_bitwise_and_is_rejected(self, convert_module):
        """整数の & は bool の論理積と意味が違う（bitwise_not と同じ絞り方）。"""

        class IntAnd(nn.Module):
            def forward(self, x, y):
                return x & y

        with pytest.raises(NotImplementedError, match="bitwise_and は未対応"):
            convert_module(
                IntAnd(),
                (torch.ones(4, dtype=torch.int64), torch.ones(4, dtype=torch.int64)),
            )

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            # ups / ResBlock（LRELU_SLOPE=0.1）と最終段（位置引数を省略した torch 既定）
            pytest.param(lambda x: nn.functional.leaky_relu(x, 0.1), 0.1, id="明示 0.1"),
            pytest.param(lambda x: nn.functional.leaky_relu(x), 0.01, id="既定 0.01"),
        ],
    )
    def test_leaky_relu_materializes_the_slope(self, convert_module, build, expected):
        """MUST: torch 側の既定を**境界で attrs に明示化**する（ADR 0015 — 実測は 2 種混在）。"""

        class Leaky(nn.Module):
            def forward(self, x):
                return build(x)

        graph, _ = convert_module(Leaky(), (torch.randn(4),))

        assert node_ops(graph) == ["leaky_relu"]
        assert only_node(graph, "leaky_relu").attrs == {"negative_slope": expected}

    def test_cumsum_normalizes_the_negative_axis(self, convert_module, dyn_t):
        """spline の cumwidths（`torch.cumsum(widths, dim=-1)`）。"""

        class Cumulative(nn.Module):
            def forward(self, x):
                return torch.cumsum(x, dim=-1)

        graph, _ = convert_module(Cumulative(), (torch.rand(4, 3),), ({0: dyn_t},))

        assert only_node(graph, "cumsum").attrs == {"dim": 1}

    def test_cumsum_on_a_non_final_axis_is_rejected(self, convert_module):
        class Wrong(nn.Module):
            def forward(self, x):
                return torch.cumsum(x, dim=0)

        with pytest.raises(NotImplementedError, match="cumsum は未対応"):
            convert_module(Wrong(), (torch.rand(4, 3),))


class TestFusedOps:
    """融合 op（ADR 0012）— 保存リスト 9 op のうち実測に出る 6 本。

    ここで固定するのは 2 つ: ① 保存が効いて**高位 op が 1 ノードのまま残る**こと
    ② attrs の値域外・実測に無い形が fail loudly になること。
    """

    def test_linear_stays_a_single_node_with_bias(self, convert_module, dyn_t):
        """保存しないと addmm + view に散る（recon §5）。"""

        class Projected(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(4, 3)

            def forward(self, x):
                return self.dense(x)

        graph, _ = convert_module(Projected(), (torch.randn(5, 4),), ({0: dyn_t},))

        node = only_node(graph, "linear")
        assert node.attrs == {}
        assert len(node.ins) == 3
        assert node_ops(graph) == ["linear"]
        # 重みは [out, in] の転置レイアウトのまま運ぶ（転置ノードを挟まない）
        assert graph.values[node.ins[1]].shape == [3, 4]
        assert graph.values[node.ins[2]].shape == [3]
        assert graph.values[node.outs[0]].shape == ["T", 3]

    def test_a_bias_free_linear_gets_a_synthesized_zero_bias(self, convert_module):
        """bias 無しの linear は**ゼロ bias 合成**でアリティ 3 へ正規化される（ADR 0016）。

        conv の合成（ADR 0015）と同じ手筋で、契約・カーネル・shape 層に arity 分岐を
        持ち込まない。合成が恒等なのは `+0` が f32 で厳密恒等だから。
        """

        class NoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(4, 3, bias=False)

            def forward(self, x):
                return self.dense(x)

        graph, tensors = convert_module(NoBias(), (torch.randn(5, 4),))

        node = only_node(graph, "linear")
        assert len(node.ins) == 3
        assert graph.values[node.ins[2]].shape == [3]
        bias = tensors[graph.initializers[node.ins[2]].tensor]
        assert torch.equal(bias, torch.zeros(3))

    def test_layer_norm_stays_single_output_with_attrs(self, convert_module, dyn_t):
        """保存しないと native_layer_norm（3 出力）+ getitem になり単一出力前提と衝突する。"""

        class Normed(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(4, eps=1e-07)

            def forward(self, x):
                return self.norm(x)

        graph, _ = convert_module(Normed(), (torch.randn(5, 4),), ({0: dyn_t},))

        node = only_node(graph, "layer_norm")
        assert node.attrs == {"normalized_shape": [4], "eps": 1e-07}
        assert len(node.ins) == 3
        assert len(node.outs) == 1
        assert node_ops(graph) == ["layer_norm"]

    def test_an_affine_free_layer_norm_gets_ones_and_zeros(self, convert_module):
        """affine 無しの layer_norm は ones/zeros 合成でアリティ 3 へ正規化される（ADR 0016）。

        合成の順序（weight → bias）が入れ替わると `×0 +1` になって値が壊れるので、実体まで
        見る（長さだけ見ると ones と zeros の取り違えが素通りする）。
        """

        class NoAffine(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(4, elementwise_affine=False)

            def forward(self, x):
                return self.norm(x)

        graph, tensors = convert_module(NoAffine(), (torch.randn(5, 4),))

        node = only_node(graph, "layer_norm")
        assert len(node.ins) == 3
        weight = tensors[graph.initializers[node.ins[1]].tensor]
        bias = tensors[graph.initializers[node.ins[2]].tensor]
        assert torch.equal(weight, torch.ones(4))
        assert torch.equal(bias, torch.zeros(4))

    def test_a_weight_only_layer_norm_gets_a_zero_bias(self, convert_module):
        """weight だけの layer_norm は zeros 合成でアリティ 3 になる（`norm_bias=false` の形）。

        weight が**本物のスロットに残る**ことまで見る（合成が weight 側へ滑り込むと
        「学習済み weight を捨てて ones で正規化」という沈黙誤値になり、長さは合うので
        shape 検査では捕まらない）。
        """

        class WeightOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4))

            def forward(self, x):
                return nn.functional.layer_norm(x, [4], self.weight, None)

        module = WeightOnly()
        graph, tensors = convert_module(module, (torch.randn(5, 4),))

        node = only_node(graph, "layer_norm")
        assert len(node.ins) == 3
        weight = tensors[graph.initializers[node.ins[1]].tensor]
        bias = tensors[graph.initializers[node.ins[2]].tensor]
        assert torch.equal(weight, module.weight.detach())
        assert torch.equal(bias, torch.zeros(4))

    def test_a_bias_only_layer_norm_is_rejected(self, convert_module):
        """bias だけの形は落とす（合成は末尾へ足すので滑り込みが起きる）。

        `weight=None, bias=あり` を通すと bias が weight のスロットへ入る — 長さが同じ
        なので shape 検査も素通りする沈黙誤値になる。
        """

        class BiasOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(4))

            def forward(self, x):
                return nn.functional.layer_norm(x, [4], None, self.bias)

        with pytest.raises(NotImplementedError, match="bias だけ"):
            convert_module(BiasOnly(), (torch.randn(5, 4),))

    def test_multi_axis_layer_norm_is_rejected(self, convert_module):
        """実測は [H] のみ（行カーネルは最終次元の連続並びが前提）。"""

        class Wide(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm([3, 4])

            def forward(self, x):
                return self.norm(x)

        with pytest.raises(NotImplementedError, match="normalized_shape"):
            convert_module(Wide(), (torch.randn(5, 3, 4),))

    def test_softmax_normalizes_the_negative_axis(self, convert_module, dyn_t):
        class Attention(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)

        graph, _ = convert_module(Attention(), (torch.randn(5, 4),), ({0: dyn_t},))

        node = only_node(graph, "softmax")
        # 負の軸表記はエクスポータ境界で非負へ正規化する（IR の attrs は非負のみ）
        assert node.attrs == {"dim": 1}

    def test_softmax_on_a_non_last_dim_is_rejected(self, convert_module):
        class ByColumn(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=0)

        with pytest.raises(NotImplementedError, match="最終次元以外"):
            convert_module(ByColumn(), (torch.randn(5, 4),))

    def test_a_runtime_bool_mask_emits_safe_softmax(self, convert_module):
        """ADR 0044 の連鎖: bool 入力 → `where(mask,0,-inf)` → add → **safe_softmax**。

        マスクが実行時値だとガードの不活性証明が原理的に立たないので、正規化パスが
        safe_softmax へ構成的に置換する。IR 境界のマスク dtype は bool（ADR 0009）。
        """

        class Masked(nn.Module):
            def forward(self, q, k, v, mask):
                return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        graph, _ = convert_module(
            Masked(),
            (
                torch.randn(1, 1, 4, 3),
                torch.randn(1, 1, 4, 3),
                torch.randn(1, 1, 4, 3),
                torch.tensor([[[[True, True, False, False]]]]),
            ),
        )

        node = only_node(graph, "safe_softmax")
        assert node.attrs == {"dim": 3}
        assert "softmax" not in node_ops(graph)
        # 0 / -inf は f32 バイナリ initializer に畳まれ、where 1 本で加算型へ変換される
        assert node_ops(graph).count("where") == 1
        assert [spec.dtype for spec in graph.inputs] == ["f32", "f32", "f32", "bool"]

    def test_a_static_finite_mask_still_emits_plain_softmax(self, convert_module):
        """静的マスクの経路は不変（ガードは従来どおり除去され、safe_softmax にならない）。"""

        class StaticMasked(nn.Module):
            def __init__(self):
                super().__init__()
                bias = torch.zeros(1, 1, 4, 4)
                bias[..., 0] = -1e4
                self.register_buffer("bias", bias)

            def forward(self, q, k, v):
                return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.bias)

        graph, _ = convert_module(
            StaticMasked(),
            (torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3)),
        )

        assert only_node(graph, "softmax").attrs == {"dim": 3}
        assert "safe_softmax" not in node_ops(graph)

    def test_embedding_carries_the_inert_padding_idx(self, convert_module, dyn_t):
        """padding_idx は受理して attrs に載せるが forward には効かない（ADR 0012）。"""

        class Table(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Embedding(5, 3, padding_idx=0)

            def forward(self, index):
                return self.table(index)

        graph, _ = convert_module(Table(), (torch.tensor([0, 1, 2, 3]),), ({0: dyn_t},))

        node = only_node(graph, "embedding")
        assert node.attrs == {"padding_idx": 0}
        # weight は f32 / index は i32（i64 からの境界正規化）— スロットで dtype が違う
        assert graph.values[node.ins[0]].dtype == "f32"
        assert graph.inputs[0].dtype == "i32"
        assert graph.values[node.outs[0]].shape == ["T", 3]

    def test_embedding_without_padding_idx_uses_the_sentinel(self, convert_module, dyn_t):
        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Embedding(5, 3)

            def forward(self, index):
                return self.table(index)

        graph, _ = convert_module(Plain(), (torch.tensor([0, 1, 2, 3]),), ({0: dyn_t},))

        assert only_node(graph, "embedding").attrs == {"padding_idx": -1}

    def test_embedding_with_a_scalar_index_is_rejected(self, convert_module):
        """torch はスカラ添字で `[H]` を返すが、契約の出力は `[…index, H]`（rank ≥ 1）。"""

        class Scalar(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Embedding(5, 3)

            def forward(self, index):
                return self.table(index)

        with pytest.raises(NotImplementedError, match="スカラ添字"):
            convert_module(Scalar(), (torch.tensor(2),))

    def test_sparse_embedding_is_rejected(self, convert_module):
        class Sparse(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Embedding(5, 3, sparse=True)

            def forward(self, index):
                return self.table(index)

        with pytest.raises(NotImplementedError, match="sparse=True"):
            convert_module(Sparse(), (torch.tensor([0, 1, 2, 3]),))

    def test_masked_fill_carries_the_scalar_value(self, convert_module, dyn_t):
        class Masked(nn.Module):
            def forward(self, x, mask):
                return x.masked_fill(mask.to(torch.bool), -3.4028234663852886e38)

        graph, _ = convert_module(
            Masked(),
            (torch.randn(1, 2, 4, 4), torch.ones(1, 1, 4, 4, dtype=torch.int64)),
            ({2: dyn_t, 3: dyn_t}, {2: dyn_t, 3: dyn_t}),
        )

        node = only_node(graph, "masked_fill")
        assert node.attrs == {"value": -3.4028234663852886e38}
        # mask は bool（cast 経由）で、出力は x と同形（右詰め broadcast は mask 側だけ）
        assert graph.values[node.ins[1]].dtype == "bool"
        assert graph.values[node.outs[0]].shape == [1, 2, "T", "T"]

    def test_masked_fill_value_survives_the_json_round_trip(self, convert_module, dyn_t):
        """ADR 0012: −3.4028234663852886e+38 は JSON 往復で **ulp 不変**でなければならない。

        グラフ JSON は safetensors の `__metadata__` に文字列として載るので、往復で 1 ulp
        でも動けば GPU に載る埋め値が torch と食い違う。
        """
        import json as _json
        import struct

        class Masked(nn.Module):
            def forward(self, x, mask):
                return x.masked_fill(mask.to(torch.bool), -3.4028234663852886e38)

        graph, _ = convert_module(
            Masked(),
            (torch.randn(2, 4), torch.ones(2, 4, dtype=torch.int64)),
            ({0: dyn_t}, {0: dyn_t}),
        )

        value = only_node(graph, "masked_fill").attrs["value"]
        restored = _json.loads(graph.to_json())
        round_tripped = next(
            n["attrs"]["value"] for n in restored["nodes"] if n["op"] == "masked_fill"
        )
        assert round_tripped == value == -3.4028234663852886e38
        # f32 へ丸めても同じビット列（= f32 の最小有限値ちょうど）
        assert struct.unpack("<f", struct.pack("<f", round_tripped))[0] == round_tripped

    def test_infinite_fill_value_is_rejected(self, convert_module):
        """`-inf` 埋めは attention で実在する書き方だが、IR v1 は非有限値を受理しない。

        黙って -3.4e38 に置き換えると softmax の下流で「本当に 0 になる」保証が崩れるので、
        fail loudly にして呼び出し側に書き換えさせる。
        """

        class Infinite(nn.Module):
            def forward(self, x, mask):
                return x.masked_fill(mask.to(torch.bool), float("-inf"))

        with pytest.raises(NotImplementedError, match="非有限"):
            convert_module(Infinite(), (torch.randn(2, 4), torch.ones(2, 4, dtype=torch.int64)))

    def test_conv1d_carries_all_four_attrs(self, convert_module, dyn_t):
        class Convolved(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Convolved(), (torch.randn(1, 4, 6),), ({2: dyn_t},))

        node = only_node(graph, "conv1d")
        # MUST: 既定値でも 4 つとも明示する（既定値補完に頼らない — ADR 0012 / 0015）
        assert node.attrs == {"stride": 1, "padding": 1, "dilation": 1, "groups": 1}
        assert len(node.ins) == 3
        assert graph.values[node.ins[1]].shape == [4, 4, 3]
        assert graph.values[node.outs[0]].shape == [1, 4, "T"]

    def test_conv1d_carries_groups_and_dilation(self, convert_module, dyn_t):
        """sdp の DDSConv 形（depthwise groups=C・dilation 3・k 5）が attrs で運ばれる。"""

        class Depthwise(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(6, 6, kernel_size=5, padding=6, dilation=3, groups=6)

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Depthwise(), (torch.randn(1, 6, 8),), ({2: dyn_t},))

        node = only_node(graph, "conv1d")
        assert node.attrs == {"stride": 1, "padding": 6, "dilation": 3, "groups": 6}
        # 重みの第 2 軸は Cin/groups = 1（Cin のままだと shape 層が落とす）
        assert graph.values[node.ins[1]].shape == [6, 1, 5]
        assert graph.values[node.outs[0]].shape == [1, 6, "T"]

    def test_a_bias_free_conv1d_gets_a_synthesized_zero_bias(self, convert_module, dyn_t):
        """bias 無しは落とさず**ゼロ bias 合成でアリティ 3 へ正規化**する（ADR 0015）。"""

        class Post(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(4, 3, kernel_size=7, padding=3, bias=False)

            def forward(self, x):
                return self.conv(x)

        graph, tensors = convert_module(Post(), (torch.randn(1, 4, 8),), ({2: dyn_t},))

        node = only_node(graph, "conv1d")
        assert len(node.ins) == 3
        bias_name = node.ins[2]
        # 合成 bias は通常の定数と同じ経路・命名（const_<digest>）で運ばれる
        assert bias_name.startswith("const_")
        assert bias_name in graph.initializers
        assert graph.values[bias_name].shape == [3]
        assert graph.values[bias_name].dtype == "f32"
        synthesized = tensors[graph.initializers[bias_name].tensor]
        assert torch.equal(synthesized, torch.zeros(3))

    def test_two_bias_free_convs_share_one_synthesized_initializer(self, convert_module):
        """同じ長さのゼロ bias は digest で 1 本に畳まれる（_add_const の重複排除）。"""

        class TwoPosts(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Conv1d(4, 3, kernel_size=3, padding=1, bias=False)
                self.second = nn.Conv1d(3, 3, kernel_size=3, padding=1, bias=False)

            def forward(self, x):
                return self.second(self.first(x))

        graph, _ = convert_module(TwoPosts(), (torch.randn(1, 4, 8),))

        convs = [node for node in graph.nodes if node.op == "conv1d"]
        assert len(convs) == 2
        assert convs[0].ins[2] == convs[1].ins[2]
        assert len([name for name in graph.initializers if name.startswith("const_")]) == 1

    def test_conv_transpose1d_carries_stride_and_padding(self, convert_module, dyn_t):
        """dec の ups 形（up 8 / k 16 / pad 4）。重みは [Cin, Cout, K] で出力長は L·stride。"""

        class Up(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.ConvTranspose1d(5, 3, kernel_size=16, stride=8, padding=4)

            def forward(self, x):
                return self.up(x)

        graph, _ = convert_module(Up(), (torch.randn(1, 5, 4),), ({2: dyn_t},))

        node = only_node(graph, "conv_transpose1d")
        assert node.attrs == {"stride": 8, "padding": 4}
        assert len(node.ins) == 3
        # MUST: [Cin, Cout, K]（conv1d の [Cout, Cin/groups, K] と転置）
        assert graph.values[node.ins[1]].shape == [5, 3, 16]
        assert graph.values[node.outs[0]].shape == [1, 3, "8T"]

    def test_a_bias_free_conv_transpose1d_also_gets_a_zero_bias(self, convert_module):
        """ゼロ bias 合成は Cout を**重みの第 2 軸**から取る（第 1 軸だと長さがずれる）。"""

        class Up(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.ConvTranspose1d(5, 3, kernel_size=2, stride=2, bias=False)

            def forward(self, x):
                return self.up(x)

        graph, _ = convert_module(Up(), (torch.randn(1, 5, 4),))

        node = only_node(graph, "conv_transpose1d")
        assert graph.values[node.ins[2]].shape == [3]

    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"output_padding": 1}, "output_padding"),
            ({"groups": 3}, "groups"),
            ({"dilation": 2}, "dilation"),
        ],
    )
    def test_conv_transpose1d_outside_the_kernel_contract_is_rejected(
        self, convert_module, kwargs, why
    ):
        """output_padding / groups / dilation は attrs に欄が無い（実測どおり 0 / 1 / 1）。"""

        class Up(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.ConvTranspose1d(3, 3, kernel_size=3, stride=2, **kwargs)

            def forward(self, x):
                return self.up(x)

        with pytest.raises(NotImplementedError, match=why):
            convert_module(Up(), (torch.randn(1, 3, 4),))


class TestLayoutOps:
    """レイアウト op（ADR 0011）— view / squeeze / unsqueeze → reshape、permute、expand。"""

    def test_view_unsqueeze_and_squeeze_all_normalize_to_reshape(self, convert_module, dyn_t):
        class Reshapes(nn.Module):
            def forward(self, x):
                viewed = x.view(-1, 12)
                lifted = viewed.unsqueeze(1)
                return torch.squeeze(lifted, dim=[1])

        graph, _ = convert_module(Reshapes(), (torch.randn(6, 3, 4),), ({0: dyn_t},))

        assert node_ops(graph) == ["reshape", "reshape", "reshape"]
        # 目標形は attrs ではなく出力の宣言 shape（attrs は空）。`-1` は解決済みの形で入る。
        assert [node.attrs for node in graph.nodes] == [{}, {}, {}]
        assert graph.values[graph.nodes[0].outs[0]].shape == ["T", 12]
        assert graph.values[graph.nodes[1].outs[0]].shape == ["T", 1, 12]
        assert graph.values[graph.nodes[2].outs[0]].shape == ["T", 12]

    def test_reshapes_to_different_shapes_are_not_merged_by_cse(self, convert_module):
        """reshape は attrs を持たない — CSE キーに出力 shape が無いと片方が黙って消える。"""

        class TwoViews(nn.Module):
            def forward(self, x):
                return x.view(2, 6), x.view(3, 4)

        graph, _ = convert_module(TwoViews(), (torch.randn(12),))

        shapes = sorted(graph.values[node.outs[0]].shape for node in graph.nodes)
        assert shapes == [[2, 6], [3, 4]]

    def test_permute_carries_normalized_dims(self, convert_module, dyn_t):
        class Heads(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1, 3)

        graph, _ = convert_module(Heads(), (torch.randn(1, 6, 3, 4),), ({1: dyn_t},))

        node = only_node(graph, "permute")
        assert node.attrs == {"dims": [0, 2, 1, 3]}
        assert graph.values[node.outs[0]].shape == [1, 3, "T", 4]

    def test_negative_permute_axes_are_normalized(self, convert_module):
        class Transposed(nn.Module):
            def forward(self, x):
                return x.permute(0, -1, -2)

        graph, _ = convert_module(Transposed(), (torch.randn(2, 3, 4),))

        assert only_node(graph, "permute").attrs == {"dims": [0, 2, 1]}

    def test_permute_beyond_the_strided_rank_is_rejected(self, convert_module):
        """rank > STRIDED_RANK の permute は依然として落ちる（門の持ち主が変わっただけ）。

        NOTE: ADR 0016 の rank 下げパス（`normalize._lower_reshape_permute`）が rank≥5 の
        permute を先に掴むようになったため、実際に落とすのは**そちら**（rank4 の端点が
        見つからない連鎖）。`_h_permute` の STRIDED_RANK 門は normalize を通さず convert を
        直接呼ぶ経路の second line として残る。
        """

        class Deep(nn.Module):
            def forward(self, x):
                return x.permute(0, 1, 2, 4, 3)

        with pytest.raises(NotImplementedError, match="端点が見つからない reshape/permute 連鎖"):
            convert_module(Deep(), (torch.randn(2, 2, 2, 2, 2),))

    def test_expand_takes_its_target_from_the_declared_shape(self, convert_module, dyn_t):
        class Spread(nn.Module):
            def forward(self, mask):
                return mask.to(torch.bool).expand(-1, 5)

        graph, _ = convert_module(Spread(), (torch.ones(6, 1, dtype=torch.int64),), ({0: dyn_t},))

        node = only_node(graph, "expand")
        assert node.attrs == {}
        assert graph.values[node.outs[0]].dtype == "bool"
        assert graph.values[node.outs[0]].shape == ["T", 5]

    def test_expand_of_int_indices_stays_i32(self, convert_module, dyn_t):
        class SpreadIndex(nn.Module):
            def forward(self, index):
                return index.expand(3, -1)

        graph, _ = convert_module(
            SpreadIndex(), (torch.zeros(1, 6, dtype=torch.int64),), ({1: dyn_t},)
        )

        node = only_node(graph, "expand")
        assert graph.values[node.outs[0]].dtype == "i32"
        assert graph.values[node.outs[0]].shape == [3, "T"]


class TestLayoutOps2:
    """レイアウト第 2 群（ADR 0014）— slice / cat / pad / flip。

    受理側は「実測形が 1 ノードに落ちる」ことを、拒否側は「表現の無い形が境界で落ちる」ことを
    見る（契約表に欄が無い軸を黙って既定で実行しない）。
    """

    def test_a_static_channel_slice_becomes_one_node(self, convert_module, dyn_t):
        class Split(nn.Module):
            def forward(self, x):
                return x[:, 3:], x[:, :3]

        graph, _ = convert_module(Split(), (torch.randn(1, 6, 5),), ({2: dyn_t},))

        assert node_ops(graph) == ["slice", "slice"]
        # 負の添字・省略はエクスポータ境界で軸長へ詰める（IR の attrs は非負のみ）
        assert graph.nodes[0].attrs == {"dim": 1, "start": 3, "end": 6}
        assert graph.nodes[1].attrs == {"dim": 1, "start": 0, "end": 3}
        assert graph.values[graph.nodes[0].outs[0]].shape == [1, 3, "T"]

    def test_negative_indices_are_normalized_at_the_boundary(self, convert_module):
        class Trim(nn.Module):
            def forward(self, x):
                return x[..., :-1]

        graph, _ = convert_module(Trim(), (torch.randn(2, 5),))

        assert only_node(graph, "slice").attrs == {"dim": 1, "start": 0, "end": 4}

    def test_a_strided_slice_is_rejected(self, convert_module):
        class EverySecond(nn.Module):
            def forward(self, x):
                return x[:, ::2]

        with pytest.raises(NotImplementedError, match="step"):
            convert_module(EverySecond(), (torch.randn(2, 6),))

    def test_a_symbolic_axis_slice_is_rejected(self, convert_module, dyn_t):
        """記号軸の切り出しは sym_prefix_slice の担当（重複させない — ADR 0014）。"""

        class Head(nn.Module):
            def forward(self, x):
                return x[:2] * 2.0

        with pytest.raises(NotImplementedError, match="記号次元"):
            convert_module(Head(), (torch.randn(6, 4),), ({0: dyn_t},))

    def test_cat_carries_every_operand_as_an_input(self, convert_module, dyn_t):
        class Join(nn.Module):
            def forward(self, a, b, c):
                return torch.cat([a, b, c], 1)

        graph, _ = convert_module(
            Join(),
            (torch.randn(1, 2, 4), torch.randn(1, 3, 4), torch.randn(1, 1, 4)),
            ({2: dyn_t}, {2: dyn_t}, {2: dyn_t}),
        )

        node = only_node(graph, "cat")
        # テンソルの**リスト**引数がそのままアリティになる（可変アリティ）
        assert node.ins == ["a", "b", "c"]
        assert node.attrs == {"dim": 1}
        assert graph.values[node.outs[0]].shape == [1, 6, "T"]

    def test_a_symbolic_cat_axis_sums_into_the_dimension_language(self, convert_module, dyn_t):
        """ADR 0046 — 連結軸は〈定数〉/〈同一シンボルの一次式〉なら記号でよい。

        ADR 0014 の「連結軸は静的」を改訂した。DiT の joint attention は
        `cat([self | context], dim=1)` で self 側の軸長が記号（`S+1519`）— 記号 1 本 + 定数は
        正準文法 `coeff·sym+offset` にそのまま載るので、拒否理由（和が載らない）が成り立たない。
        """

        class JoinTime(nn.Module):
            def forward(self, a, b):
                return torch.cat([a, b], 0)

        graph, _ = convert_module(
            JoinTime(), (torch.randn(6, 4), torch.randn(3, 4)), ({0: dyn_t}, None)
        )
        assert graph.values[only_node(graph, "cat").outs[0]].shape == ["T+3", 4]

        graph, _ = convert_module(
            JoinTime(), (torch.randn(6, 4), torch.randn(6, 4)), ({0: dyn_t}, {0: dyn_t})
        )
        assert graph.values[only_node(graph, "cat").outs[0]].shape == ["2T", 4]

    def test_mixing_different_symbols_on_the_cat_axis_is_rejected(self, convert_module, dyn_t):
        """異シンボルの和は 1 次元 1 シンボルの次元言語に載らない（ADR 0046 の唯一の拒否）。"""

        class JoinTime(nn.Module):
            def forward(self, a, b):
                return torch.cat([a, b], 0)

        other = Dim("U", min=2, max=SYM_MAX)
        with pytest.raises(NotImplementedError, match="異なるシンボル"):
            convert_module(
                JoinTime(),
                (torch.randn(6, 4), torch.randn(5, 4)),
                ({0: dyn_t}, {0: other}),
                symbol_names=("T", "U"),
            )

    def test_last_dim_zero_padding_becomes_one_node(self, convert_module, dyn_t):
        class Window(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [4, 4])

        graph, _ = convert_module(Window(), (torch.randn(1, 2, 6),), ({2: dyn_t},))

        node = only_node(graph, "pad")
        assert node.attrs == {"left": 4, "right": 4}
        # 記号長の最終次元は T+2w（次元言語のオフセット付き形）
        assert graph.values[node.outs[0]].shape == [1, 2, "T+8"]

    def test_an_asymmetric_pad_keeps_the_two_sides_apart(self, convert_module):
        """左右幅が違う pad の対応を **torch オラクル**で固定する（ADR 0014）。

        `[2, 1]` と `[1, 2]` は**出力長が同じ**なので、左右を取り違えても shape 検査も
        次元言語も素通りする（対称幅の既存ケースでは原理的に検出できない）。attrs の値
        だけを書き写したテストは「エクスポータが今そう出している」ことの追認にしかならない
        ので、torch の eager 出力で「先頭に 2 個・末尾に 1 個の 0」を確定させ、その並びと
        attrs の対応を 1 本のテストで結ぶ。
        """

        class Lopsided(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [2, 1])

        x = torch.tensor([[1.0, 2.0, 3.0]])
        graph, _ = convert_module(Lopsided(), (x,))

        assert only_node(graph, "pad").attrs == {"left": 2, "right": 1}
        assert graph.values[only_node(graph, "pad").outs[0]].shape == [1, 6]
        # torch 側の意味: 第 1 要素が**先頭**の幅。取り違えれば 0 の並びが変わる。
        assert torch.equal(Lopsided()(x), torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]]))

    def test_a_nonzero_pad_value_is_rejected(self, convert_module):
        class Filled(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [1, 1], value=1.0)

        with pytest.raises(NotImplementedError, match="定数 0"):
            convert_module(Filled(), (torch.randn(2, 3),))

    def test_padding_more_than_the_last_dim_is_rejected(self, convert_module):
        class Wide(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [1, 1, 2, 2])

        with pytest.raises(NotImplementedError, match="最終次元"):
            convert_module(Wide(), (torch.randn(2, 3),))

    def test_a_negative_pad_is_rejected(self, convert_module):
        """負幅は切り詰め（slice の意味）— 同じ形を 2 つの op で書けるようにしない。"""

        class Cropped(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [-1, 0])

        with pytest.raises(NotImplementedError, match="負の pad"):
            convert_module(Cropped(), (torch.randn(2, 3),))

    def test_flip_on_a_static_axis_becomes_one_node(self, convert_module, dyn_t):
        class Reverse(nn.Module):
            def forward(self, x):
                return torch.flip(x, [1])

        graph, _ = convert_module(Reverse(), (torch.randn(1, 6, 4),), ({2: dyn_t},))

        node = only_node(graph, "flip")
        assert node.attrs == {"dim": 1}
        assert graph.values[node.outs[0]].shape == [1, 6, "T"]

    def test_a_multi_axis_flip_is_rejected(self, convert_module):
        """IR の flip は 1 軸専業（黙って 1 軸に潰すと反転しない軸が出る）。"""

        class ReverseBoth(nn.Module):
            def forward(self, x):
                return torch.flip(x, [0, 1])

        with pytest.raises(NotImplementedError, match="軸 1 本"):
            convert_module(ReverseBoth(), (torch.randn(3, 4),))


class TestRmsNorm:
    """ADR 0017 — アリティ 2・eps 必須。正規化長の正本は weight（attrs に載せない）。"""

    def test_nn_rms_norm_stays_a_single_node_with_eps(self, convert_module, dyn_t):
        class Normed(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(4, eps=1e-6)

            def forward(self, x):
                return self.norm(x)

        graph, _ = convert_module(Normed(), (torch.randn(5, 4),), ({0: dyn_t},))

        node = only_node(graph, "rms_norm")
        assert node_ops(graph) == ["rms_norm"]
        assert node.attrs == {"eps": 1e-6}
        assert len(node.ins) == 2
        assert graph.values[node.ins[1]].shape == [4]

    def test_an_affine_free_rms_norm_gets_a_synthesized_ones_weight(self, convert_module):
        """weight 無し形は ones 合成でアリティ 2 へ正規化される（`×1` の厳密恒等）。"""

        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.RMSNorm(4, eps=1e-6, elementwise_affine=False)

            def forward(self, x):
                return self.norm(x)

        graph, tensors = convert_module(Plain(), (torch.randn(5, 4),))

        node = only_node(graph, "rms_norm")
        assert len(node.ins) == 2
        assert torch.equal(tensors[graph.initializers[node.ins[1]].tensor], torch.ones(4))

    def test_a_missing_eps_takes_the_torch_default(self, convert_module):
        """torch は eps 省略時に `finfo(dtype).eps` を使う — 境界で明示化する。"""

        class Defaulted(nn.Module):
            def forward(self, x, weight):
                return torch.rms_norm(x, [4], weight)

        graph, _ = convert_module(Defaulted(), (torch.randn(5, 4), torch.rand(4)))

        assert only_node(graph, "rms_norm").attrs == {"eps": float(torch.finfo(torch.float32).eps)}

    def test_a_multi_axis_normalized_shape_is_rejected(self, convert_module):
        """行カーネルは最終次元 1 本が前提（複数軸は別の意味論）。"""

        class MultiAxis(nn.Module):
            def forward(self, x, weight):
                return torch.rms_norm(x, [3, 4], weight)

        with pytest.raises(NotImplementedError, match="最終次元 1 本"):
            convert_module(MultiAxis(), (torch.randn(5, 3, 4), torch.rand(3, 4)))


class TestConv2d:
    """ADR 0017 — 空間 attrs は H/W の 2 成分・4 つとも宣言必須・重みは [Cout,Cin/g,Kh,Kw]。"""

    def test_asymmetric_attrs_are_carried_as_pairs(self, convert_module):
        """MUST: Kh≠Kw・stride/padding の H/W 非対称・Cin≠Cout をまとめて踏む形。"""

        class Wide(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(4, 6, kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Wide(), (torch.randn(1, 4, 6, 5),))

        node = only_node(graph, "conv2d")
        assert node.attrs == {
            "stride": [1, 2],
            "padding": [1, 0],
            "dilation": [1, 1],
            "groups": 1,
        }
        assert graph.values[node.ins[1]].shape == [6, 4, 3, 1]
        assert graph.values[node.outs[0]].shape == [1, 6, 6, 3]

    def test_a_scalar_spatial_argument_is_normalized_to_two_components(self, convert_module):
        """torch は `1` / `[1]` / `[1,1]` を同じ意味で受ける — 吸収は境界の側で済ませる。"""

        class Square(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1, dilation=1)

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Square(), (torch.randn(1, 2, 8, 8),))

        assert only_node(graph, "conv2d").attrs["stride"] == [2, 2]
        assert only_node(graph, "conv2d").attrs["padding"] == [1, 1]

    def test_a_bias_free_conv2d_gets_a_synthesized_zero_bias(self, convert_module):
        class NoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    6, 3, kernel_size=(1, 3), padding=(0, 1), groups=3, bias=False
                )

            def forward(self, x):
                return self.conv(x)

        graph, tensors = convert_module(NoBias(), (torch.randn(1, 6, 4, 4),))

        node = only_node(graph, "conv2d")
        assert node.attrs["groups"] == 3
        assert len(node.ins) == 3
        assert torch.equal(tensors[graph.initializers[node.ins[2]].tensor], torch.zeros(3))

    def test_dilation_is_carried_per_axis(self, convert_module):
        class Dilated(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 2, kernel_size=(2, 2), padding=(2, 1), dilation=(2, 1))

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Dilated(), (torch.randn(1, 3, 6, 3),))

        node = only_node(graph, "conv2d")
        assert node.attrs["dilation"] == [2, 1]
        assert graph.values[node.outs[0]].shape == [1, 2, 8, 4]


class TestUpsampleBilinear2d:
    """第 1 層 — align_corners=True 専業。受理は実測形（size 指定・rank 4・f32）だけ。"""

    def test_a_size_specified_interpolate_becomes_one_node(self, convert_module):
        """`F.interpolate(size=…, mode="bilinear")` は 1 ノードに落ちる（分解されない）。"""

        class Resize(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, size=(7, 9), mode="bilinear", align_corners=True
                )

        graph, _ = convert_module(Resize(), (torch.randn(1, 3, 4, 5),))

        assert node_ops(graph) == ["upsample_bilinear2d"]
        node = only_node(graph, "upsample_bilinear2d")
        assert node.attrs == {"output_size": [7, 9]}
        assert graph.values[node.outs[0]].shape == [1, 3, 7, 9]

    def test_shrinking_uses_the_same_op(self, convert_module):
        """縮小も同じ op（torch も同一 op で、antialias が無いのは仕様どおり）。"""

        class Shrink(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, size=(2, 3), mode="bilinear", align_corners=True
                )

        graph, _ = convert_module(Shrink(), (torch.randn(1, 3, 8, 10),))

        assert only_node(graph, "upsample_bilinear2d").attrs == {"output_size": [2, 3]}

    def test_align_corners_false_is_rejected(self, convert_module):
        """MUST: 座標式も端の扱いも別物なので、黙って同じ op で実行しない。"""

        class HalfPixel(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, size=(7, 9), mode="bilinear", align_corners=False
                )

        with pytest.raises(NotImplementedError) as err:
            convert_module(HalfPixel(), (torch.randn(1, 3, 4, 5),))

        assert "align_corners" in str(err.value)

    def test_a_scale_factor_is_rejected(self, convert_module):
        """MUST: 倍率指定は出力長の丸め規約がもう 1 つ増える（同じ形に 2 通りの IR）。"""

        class Doubled(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, scale_factor=2.0, mode="bilinear", align_corners=True
                )

        with pytest.raises(NotImplementedError) as err:
            convert_module(Doubled(), (torch.randn(1, 3, 4, 5),))

        assert "scale_factors" in str(err.value)

    def test_a_non_f32_input_is_rejected(self, convert_module):
        """契約は f32 専業。同じ overload で f16 / f64 が来るので入口で落とす。"""

        class Resize(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, size=(7, 9), mode="bilinear", align_corners=True
                )

        with pytest.raises(NotImplementedError) as err:
            convert_module(Resize(), (torch.randn(1, 3, 4, 5, dtype=torch.float64),))

        assert "f32" in str(err.value)

    def test_a_non_bilinear_mode_is_not_in_the_vocabulary(self, convert_module):
        """nearest は別の aten op（`upsample_nearest2d.vec`）で、対応表に無いので落ちる。

        MUST: mode を attrs に載せていないので、ここが「別の補間が黙って双線形で実行される」
        経路の唯一の門になる。
        """

        class Nearest(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(x, size=(7, 9), mode="nearest")

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Nearest(), (torch.randn(1, 3, 4, 5),))

        assert "upsample_nearest2d" in str(err.value)


class TestGruScan:
    """第 2 層 — GRU 隠れ側スキャン（ADR 0056）。受理は実測形（アリティ 4・f32）だけ。"""

    HIDDEN = 5

    @classmethod
    def _module(cls, *, reverse=False, hidden=None):
        width = cls.HIDDEN if hidden is None else hidden

        class Scan(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3 * width, width) * 0.1)
                self.bias = nn.Parameter(torch.randn(3 * width) * 0.1)

            def forward(self, gates, state):
                op = torch.ops.karume.gru_scan_reverse if reverse else torch.ops.karume.gru_scan
                return op(gates, state, self.weight, self.bias)

        return Scan()

    @classmethod
    def _args(cls, length=6):
        return (torch.randn(length, 1, 3 * cls.HIDDEN), torch.randn(1, cls.HIDDEN))

    @pytest.mark.parametrize("reverse", [False, True])
    def test_a_scan_stays_one_node_with_a_symbolic_time_axis(self, convert_module, dyn_t, reverse):
        """MUST: 時間軸は記号のまま 1 ノードで残る（この op の存在理由そのもの）。

        `aten.gru.input` は分解が時間方向へ完全展開されるため長さが specialize されるが、
        `karume::` の custom op は本体が `register_fake` の裏にあるのでトレースが入らず、
        `run_decompositions(curated_decompositions())` を通しても展開されない。
        """
        graph, _ = convert_module(
            self._module(reverse=reverse), self._args(), dynamic_shapes=({0: dyn_t}, None)
        )

        name = "gru_scan_reverse" if reverse else "gru_scan"
        assert node_ops(graph) == [name]
        node = only_node(graph, name)
        # attrs 空契約（走査方向は op 名が持つ — bool attr は語彙に無い）
        assert node.attrs == {}
        # gi / h0 / w_hh / b_hh の 4 本（custom op の位置引数順）
        assert len(node.ins) == 4
        assert graph.values[node.outs[0]].shape == ["T", 1, self.HIDDEN]

    def test_both_directions_are_distinct_ops(self, convert_module, dyn_t):
        """MUST: 方向は別 op 名。同じ名前へ畳むと走査順が IR から読めなくなる。"""

        class Bidirectional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3 * 5, 5) * 0.1)
                self.bias = nn.Parameter(torch.randn(3 * 5) * 0.1)

            def forward(self, gates, state):
                forward = torch.ops.karume.gru_scan(gates, state, self.weight, self.bias)
                backward = torch.ops.karume.gru_scan_reverse(gates, state, self.weight, self.bias)
                return torch.cat([forward, backward], dim=2)

        graph, _ = convert_module(Bidirectional(), self._args(), dynamic_shapes=({0: dyn_t}, None))

        assert node_ops(graph) == ["gru_scan", "gru_scan_reverse", "cat"]
        # 層間の結合は既存 cat（dim 2 は静的軸なので ADR 0046 の緩和すら要らない）
        assert only_node(graph, "cat").attrs == {"dim": 2}

    def test_a_non_f32_input_is_rejected(self, convert_module):
        """契約は f32 専業（同じ op で f64 が来る）。"""
        args = tuple(tensor.to(torch.float64) for tensor in self._args())

        with pytest.raises(NotImplementedError) as err:
            convert_module(self._module().to(torch.float64), args)

        assert "f32" in str(err.value)

    def test_a_batch_first_layout_is_rejected(self, convert_module):
        """MUST: `gi` は `[T,N,3H]`（time-first）固定。

        `batch_first` の欄が無いので、時間軸と特徴軸を畳んだ rank-2 の `gi` は**ここで**
        落ちる（通すと `[N,3H]` を「T = N の系列」として黙って走査する）。
        """

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(15, 5) * 0.1)
                self.bias = nn.Parameter(torch.randn(15) * 0.1)

            def forward(self, gates, state):
                return torch.ops.karume.gru_scan(gates, state, self.weight, self.bias)

        with pytest.raises(NotImplementedError) as err:
            convert_module(Flat(), (torch.randn(1, 15), torch.randn(1, 5)))

        assert "rank" in str(err.value)


class TestDeformConv2d:
    """第 1' 層 — DCNv2 専業（ADR 0055）。受理は実測形（BiRefNet の 20 箇所）だけ。"""

    @staticmethod
    def _module(padding=(1, 0), *, bias=True):
        class Deform(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 3, 3, 2))
                self.bias = nn.Parameter(torch.randn(5)) if bias else None

            def forward(self, x, offset, mask):
                return deform_conv2d(
                    input=x,
                    offset=offset,
                    weight=self.weight,
                    bias=self.bias,
                    padding=padding,
                    mask=mask,
                )

        return Deform()

    @staticmethod
    def _args():
        # padding (1, 0) → Hout = 4 + 2 - 2 = 4 / Wout = 5 + 0 - 1 = 4
        return (torch.randn(1, 3, 4, 5), torch.randn(1, 12, 4, 4), torch.randn(1, 6, 4, 4))

    def test_a_modulated_deform_conv_becomes_one_node(self, convert_module):
        """`torchvision::deform_conv2d` は curated decomp 後も 1 ノードで残る。"""
        graph, _ = convert_module(self._module(), self._args())

        assert node_ops(graph) == ["deform_conv2d"]
        node = only_node(graph, "deform_conv2d")
        assert node.attrs == {"padding": [1, 0]}
        # 入力は x / weight / offset / mask / bias の 5 本（torchvision の位置引数順）
        assert len(node.ins) == 5
        assert graph.values[node.outs[0]].shape == [1, 5, 4, 4]

    def test_a_missing_bias_is_folded_into_an_initializer(self, convert_module):
        """bias 無しは torchvision のラッパが `aten.full` を挿し、第 0 層が initializer にする。

        MUST: エクスポータ側でゼロ bias を合成しない（合成経路を二重に持たない — ADR 0055）。
        """
        graph, tensors = convert_module(self._module(bias=False), self._args())

        assert node_ops(graph) == ["deform_conv2d"]
        node = only_node(graph, "deform_conv2d")
        assert node.ins[4] in graph.initializers
        assert tensors[graph.initializers[node.ins[4]].tensor].tolist() == [0.0] * 5

    def test_an_unmodulated_deform_conv_is_rejected(self, convert_module):
        """MUST: use_mask=False（DCNv1）は落とす。

        torchvision は mask=None のとき **[1,1] のダミーテンソル**を渡してくるので、
        素通しするとダミーが modulator として掛かる沈黙誤値になる。
        """

        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 3, 3, 2))

            def forward(self, x, offset):
                return deform_conv2d(input=x, offset=offset, weight=self.weight, padding=(1, 0))

        x, offset, _mask = self._args()
        with pytest.raises(NotImplementedError) as err:
            convert_module(Plain(), (x, offset))

        assert "use_mask" in str(err.value)

    @pytest.mark.parametrize(
        ("kwargs", "shapes", "what"),
        [
            ({"stride": (2, 2)}, (1, 12, 2, 2), "stride_h"),
            ({"dilation": (2, 2)}, (1, 12, 2, 3), "dilation_h"),
        ],
    )
    def test_values_without_a_field_are_rejected(self, convert_module, kwargs, shapes, what):
        """MUST: attrs に欄が無い（= 1 固定）値は、既定値補完ではなく**実値**を見て落とす。"""

        class Strided(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 3, 3, 2))

            def forward(self, x, offset, mask):
                return deform_conv2d(
                    input=x,
                    offset=offset,
                    weight=self.weight,
                    padding=(1, 0),
                    mask=mask,
                    **kwargs,
                )

        offset = torch.randn(*shapes)
        mask = torch.randn(shapes[0], shapes[1] // 2, *shapes[2:])
        with pytest.raises(NotImplementedError) as err:
            convert_module(Strided(), (torch.randn(1, 3, 4, 5), offset, mask))

        assert what in str(err.value)

    def test_offset_groups_greater_than_one_is_rejected(self, convert_module):
        """MUST: offset_groups は offset のチャネル数から導かれる（欄が無い = 1 固定）。"""

        class Grouped(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 3, 3, 2))

            def forward(self, x, offset, mask):
                return deform_conv2d(
                    input=x, offset=offset, weight=self.weight, padding=(1, 0), mask=mask
                )

        # offset_groups = 3（offset のチャネルが 3 * 2 * Kh * Kw）
        with pytest.raises(NotImplementedError) as err:
            convert_module(
                Grouped(),
                (torch.randn(1, 3, 4, 5), torch.randn(1, 36, 4, 4), torch.randn(1, 18, 4, 4)),
            )

        assert "offset_groups" in str(err.value)

    def test_a_non_f32_input_is_rejected(self, convert_module):
        """契約は f32 専業（同じ op で f64 が来る）。"""
        args = tuple(tensor.to(torch.float64) for tensor in self._args())
        module = self._module().to(torch.float64)

        with pytest.raises(NotImplementedError) as err:
            convert_module(module, args)

        assert "f32" in str(err.value)
