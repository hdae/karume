"""コンバータの振る舞い固定。

黙って壊れた IR を書き出す経路（入力欠落・prefix 非可換な焼き込み・語彙外 op の近似）を
実際に踏むモジュールで再現し、fail loudly か正しい表現かのどちらかになることを固定する。
"""

from __future__ import annotations

import pytest
import torch
from conftest import SYM_MAX, node_ops, only_node
from torch import nn
from torch.export import Dim

from karume.convert import (
    Converter,
    UnsupportedAtenOpsError,
    _bitwise_equal,
    normalize_boundary_tensor,
)
from karume.ops import EMITTABLE_OPS
from karume.pipeline import export_to_file


class Scale(nn.Module):
    def forward(self, x):
        return x + x


class TestTensorInputs:
    """テンソル入力は位置引数から組む（all_input_nodes の重複排除に依らない）。"""

    def test_same_tensor_on_both_operands_yields_two_inputs(self, convert_module, dyn_t):
        class Square(nn.Module):
            def forward(self, x):
                return x * x

        graph, _ = convert_module(Square(), (torch.randn(6, 4),), ({0: dyn_t},))

        assert only_node(graph, "mul").ins == ["x", "x"]

    def test_symint_used_as_a_value_is_rejected_with_the_op_and_args(self, convert_module, dyn_t):
        class ScaleByLength(nn.Module):
            def forward(self, x):
                return x * x.shape[0]

        with pytest.raises(NotImplementedError) as err:
            convert_module(ScaleByLength(), (torch.randn(6, 4),), ({0: dyn_t},))

        message = str(err.value)
        assert "aten.mul.Tensor" in message
        assert "SymInt" in message
        assert "args=" in message

    def test_scalar_operand_becomes_a_rank1_constant(self, convert_module):
        """スカラ被演算子は正規化パスが rank-1 定数へ昇格させる（ADR 0010 / recon §4-4）。"""

        class Doubling(nn.Module):
            def forward(self, x):
                return x * 2.0

        graph, tensors = convert_module(Doubling(), (torch.randn(3, 4),))

        folded = only_node(graph, "mul").ins[1]
        assert graph.values[folded].shape == [1]
        assert torch.equal(tensors[graph.initializers[folded].tensor], torch.tensor([2.0]))

    def test_type_promoting_scalar_operand_is_still_rejected(self, convert_module):
        """昇格させるのは dtype が変わらない形だけ — 型昇格を含む形は fail loudly。

        `i32 テンソル × f32 スカラ` は cast を含むので、rank-1 定数に落とすと二項 op の
        「入力 dtype は同型」契約を黙って破る。
        """

        class Scaled(nn.Module):
            def forward(self, mask):
                return mask * 0.5

        with pytest.raises(NotImplementedError, match="スカラ被演算子は未対応"):
            convert_module(Scaled(), (torch.ones(3, 4, dtype=torch.int64),))


class TestUnsupportedOps:
    """語彙外の aten op は全件列挙して落とす（部分近似も 1 件打ち切りもしない）。"""

    def test_every_unsupported_op_is_listed_at_once(self, convert_module):
        """NOTE: 2 本目の見本は `sin` → `atan` に差し替えた（sin が IR 語彙に入り、
        「語彙外」の見本でなくなったため）。検査そのもの（2 件を 1 度に列挙する）は不変。
        """

        class Trigonometry(nn.Module):
            def forward(self, x):
                return torch.cos(x) + torch.atan(x)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Trigonometry(), (torch.randn(3, 4),))

        assert set(err.value.ops) == {"aten.cos.default", "aten.atan.default"}
        assert "aten.cos.default" in str(err.value)
        assert "aten.atan.default" in str(err.value)

    def test_op_outside_the_fold_allowlist_stays_in_the_enumeration(self, convert_module, dyn_t):
        """畳み込み allowlist に無い op は記号依存でも畳まない（黙って近似しない）。

        NOTE: 見本の op は `flip` → `sin` → `floor` と差し替えてきた（ADR 0014 で flip が実行系の
        語彙に、ADR 0016 で sin が畳み allowlist に入り、それぞれ「allowlist にも対応表にも
        無い op」の見本でなくなったため）。検査そのもの（記号依存でも畳まず全件列挙へ回す）は
        変えていない。
        """

        class Stepped(nn.Module):
            def forward(self, x):
                steps = torch.floor(torch.arange(x.shape[0], dtype=torch.float32) * 0.5)
                return x + steps.unsqueeze(-1)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Stepped(), (torch.randn(6, 4),), ({0: dyn_t},))

        assert "aten.floor.default" in err.value.ops


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


class TestConstantFolding:
    """定数部分木は initializer に焼く（記号に依存しないものだけ）。"""

    def test_constant_subtree_becomes_an_initializer(self, convert_module):
        class ScaleByOnes(nn.Module):
            def forward(self, x):
                return x * torch.ones(4)

        graph, tensors = convert_module(ScaleByOnes(), (torch.randn(3, 4),))

        folded = only_node(graph, "mul").ins[1]
        assert graph.initializers[folded].tensor in tensors
        assert torch.equal(tensors[graph.initializers[folded].tensor], torch.ones(4))

    def test_identical_constants_are_deduplicated(self, convert_module):
        class TwoOnes(nn.Module):
            def forward(self, x):
                return x * torch.ones(4), x + torch.ones(4)

        graph, tensors = convert_module(TwoOnes(), (torch.randn(3, 4),))

        assert len(tensors) == 1
        assert only_node(graph, "mul").ins[1] == only_node(graph, "add").ins[1]

    def test_integer_constant_becomes_an_i32_initializer(self, convert_module):
        """整数定数は i32 initializer（生の int32 格納 — ADR 0010）。i64 は境界で i32 へ。"""

        class IntegerIndex(nn.Module):
            def forward(self, src):
                return torch.gather(src, -1, torch.arange(4).unsqueeze(0).expand(3, 4))

        graph, tensors = convert_module(IntegerIndex(), (torch.randn(3, 4),))

        index = only_node(graph, "gather").ins[1]
        # expand は畳み込み frontier で止まる（実体化が膨らむ）ので、定数はその入力側。
        folded = only_node(graph, "expand").ins[0]
        assert index == only_node(graph, "expand").outs[0]
        assert graph.values[folded].dtype == "i32"
        assert graph.initializers[folded].storage.dtype == "i32"
        assert tensors[graph.initializers[folded].tensor].dtype is torch.int32

    def test_bool_constant_is_rejected(self, convert_module):
        """bool の initializer は IR v1 の語彙に無い（意味論は f32 / i32 のみ）。"""

        class Masked(nn.Module):
            def forward(self, x):
                return x.masked_fill(torch.arange(4) < 2, 0.0)

        with pytest.raises(NotImplementedError, match="の定数は IR v1 の initializer にできない"):
            convert_module(Masked(), (torch.randn(3, 4),))


class TestSymbolicConstantFolding:
    """記号依存の部分木は Tmax で焼き、実行時は sym_prefix_slice で切り出す（ADR 0010）。"""

    def test_symbol_dependent_constant_is_baked_at_tmax_and_prefix_sliced(
        self, convert_module, dyn_t
    ):
        class PositionIndex(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(-1)

        graph, tensors = convert_module(PositionIndex(), (torch.randn(6, 4),), ({0: dyn_t},))

        sliced = only_node(graph, "sym_prefix_slice")
        assert sliced.attrs == {"sym": "T", "slices": [{"dim": 0, "coeff": 1, "offset": 0}]}
        # 焼いた定数は **Tmax = dyn_t の上限**の長さで、実行時は先頭 T 要素だけを読む。
        baked = tensors[graph.initializers[sliced.ins[0]].tensor]
        assert torch.equal(baked, torch.arange(SYM_MAX, dtype=torch.float32).unsqueeze(-1))
        assert graph.values[sliced.outs[0]].shape == ["T", 1]

    def test_prefix_slice_beyond_the_strided_rank_is_rejected(self, convert_module, dyn_t):
        """prefix スライスは strided 実体化コピーで実行する（新カーネルは無い — ADR 0010）。

        rank 上限を export 時点で見ないと「export は緑・ブラウザだけ落ちる」になる。
        """

        class Wide(nn.Module):
            def forward(self, x):
                length = x.shape[0]
                table = torch.arange(length, dtype=torch.float32)
                wide = table.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,1,T,1]
                return x.reshape(1, 1, 1, length, 1) + wide

        with pytest.raises(NotImplementedError, match="strided カーネル"):
            convert_module(Wide(), (torch.randn(6),), ({0: dyn_t},))

    def test_a_static_subtree_stays_a_plain_constant(self, convert_module, dyn_t):
        """記号次元が無ければ sym_prefix_slice は出ない（無駄な 1 dispatch を足さない）。"""

        class StaticOffset(nn.Module):
            def forward(self, x):
                return x + torch.arange(4, dtype=torch.float32)

        graph, _ = convert_module(StaticOffset(), (torch.randn(6, 4),), ({0: dyn_t},))

        assert node_ops(graph) == ["add"]

    def test_a_subtree_that_uses_the_symbol_as_a_value_is_rejected(self, convert_module, dyn_t):
        """`arange(T) / T` は allowlist だけ見れば適格 — 消費位置の検査が止める。

        NOTE: 期待する診断を「2 点評価の非可換」から差し替えた。除数の T は extent でなく
        値位置なので div が畳み込みから外れ、テンソル入力 1 本の形として arity 検査が先に
        落とす（拒否されること自体は変わらない）。
        """

        class NormalizedPositions(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32) / x.shape[0]

        with pytest.raises(NotImplementedError, match=r"SymInt .* を値として使う形"):
            convert_module(NormalizedPositions(), (torch.randn(6),), ({0: dyn_t},))

    def test_a_constant_filled_with_the_symbol_is_rejected(self, convert_module, dyn_t):
        """`full((T,), T)` — shape も値も T 依存で、prefix の値だけが食い違う形。

        NOTE: 期待する診断を「2 点評価の非可換」から差し替えた。`fill_value` の T は値位置
        なので full が畳み込みから外れ、実行系の語彙に無い op として全件列挙へ回る。
        """

        class FilledWithLength(nn.Module):
            def forward(self, x):
                return x + torch.full((x.shape[0],), x.shape[0]).to(torch.float32)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(FilledWithLength(), (torch.randn(6),), ({0: dyn_t},))

        assert "aten.full.default" in err.value.ops

    def test_a_symbol_promoted_to_tensor_data_is_rejected(self, convert_module, dyn_t):
        """`scalar_tensor(T)` でテンソルデータへ昇格したシンボルは畳まない（2 点評価の穴）。

        `(T−Tmax)(T−(Tmax−1))` は 2 つの評価点で**両方 0** になるので実測はすり抜ける —
        焼けば T=2 で 182 倍が黙って消える。消費位置（値 vs extent）で落とすため
        scalar_tensor は畳み込みから外れ、実行系の語彙に無い op として拒否される。
        """

        class SymbolAsData(nn.Module):
            def forward(self, x):
                length = torch.scalar_tensor(x.shape[0], dtype=torch.float32)
                return x * ((length - SYM_MAX) * (length - (SYM_MAX - 1)))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(SymbolAsData(), (torch.randn(6),), ({0: dyn_t},))

        assert "aten.scalar_tensor.default" in err.value.ops

    def test_a_symbol_free_fill_value_still_folds_under_a_symbolic_extent(
        self, convert_module, dyn_t
    ):
        """落とすのは消費**位置** — 形が T 依存でも、値が T に依らなければ従来どおり畳む。"""

        class ConstantRow(nn.Module):
            def forward(self, x):
                return x + torch.full((x.shape[0],), 3.0)

        graph, tensors = convert_module(ConstantRow(), (torch.randn(6),), ({0: dyn_t},))

        sliced = only_node(graph, "sym_prefix_slice")
        baked = tensors[graph.initializers[sliced.ins[0]].tensor]
        assert torch.equal(baked, torch.full((SYM_MAX,), 3.0))

    def test_an_unbounded_symbol_is_rejected(self, convert_module):
        """Tmax が無ければ焼く長さを決められない（既定値で進めない）。"""

        class PositionIndex(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32)

        with pytest.raises(NotImplementedError, match="有限でない"):
            convert_module(PositionIndex(), (torch.randn(6),), ({0: Dim.DYNAMIC},))

    def test_a_symbol_range_too_narrow_for_two_points_is_rejected(self, convert_module):
        """第 2 評価点が 0/1 特殊化の回避線を割る値域は、検査が恒真化するので受理しない。"""

        class PositionIndex(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32)

        with pytest.raises(ValueError, match="2 点評価ができない"):
            convert_module(PositionIndex(), (torch.randn(2),), ({0: Dim("T", min=1, max=2)},))


class TestFoldProbeComparison:
    """2 点評価の比較はビット厳密（数値等価では符号ビットの差が素通りする）。

    NOTE: 比較関数を直に呼ぶ。symbol-as-data を拒否した後、export 経由でこの差を作るには
    「値が T に依るのに畳める部分木」が要る — つまり黙って通る形そのもので、テストから
    合成できない（合成できたらそれが未修正のバグ）。
    """

    def test_negative_zero_is_not_equal_to_positive_zero(self):
        minus = torch.tensor([-0.0])
        plus = torch.tensor([0.0])

        assert torch.equal(minus, plus)  # 数値等価は素通りする — 強化した理由そのもの
        assert not _bitwise_equal(minus, plus)

    def test_identical_bits_are_equal(self):
        assert _bitwise_equal(torch.tensor([[1.5, -0.0]]), torch.tensor([[1.5, -0.0]]))

    def test_a_shape_or_dtype_difference_is_not_equal(self):
        assert not _bitwise_equal(torch.tensor([1.0]), torch.tensor([[1.0]]))
        assert not _bitwise_equal(torch.tensor([1.0]), torch.tensor([1], dtype=torch.int32))


class TestCommonSubexpression:
    def test_identical_nodes_collapse_to_one(self, convert_module):
        """export は同一部分式を残す — IR 側で 1 本に畳む。"""

        class Twice(nn.Module):
            def forward(self, x, y):
                return (x + y) + (x + y)

        graph, _ = convert_module(Twice(), (torch.randn(3, 4), torch.randn(3, 4)))

        assert node_ops(graph) == ["add", "add"]
        outer = graph.nodes[1]
        assert outer.ins == [graph.nodes[0].outs[0]] * 2


class TestConstantDedup:
    """定数の重複排除は full SHA-256 で突合する（名前の短縮形は表示だけ）。"""

    def _converter(self) -> Converter:
        ep = torch.export.export(Scale(), (torch.randn(3, 4),), strict=False)
        return Converter(ep)

    def test_the_dedup_key_is_the_full_digest_not_the_shortened_name(self):
        """64bit 前置だけを突合キーにすると、衝突した 2 定数を実体比較なしで畳む。"""
        converter = self._converter()

        name = converter._add_const(torch.zeros(3), "テスト定数")

        (digest,) = converter._const_by_digest
        assert len(digest) == 64
        # 名前・テンソルキーの綴りは従来どおり先頭 16 hex。
        assert name == f"const_{digest[:16]}"
        assert converter.graph.initializers[name].tensor == f"const.{digest[:16]}"

    def test_a_short_name_collision_between_distinct_constants_fails_loudly(self):
        """full が違うのに短縮名が一致した形（本物の衝突は到達不能）を作って踏む。

        黙って通すと後から来た定数が先の宣言を上書きし、先の定数を参照していたノードが
        別の値を読む（沈黙誤グラフ）。
        """
        converter = self._converter()
        converter._add_const(torch.zeros(3), "テスト定数")
        # 突合キーだけ落として「別の定数が同じ短縮名に落ちた」状況にする。
        converter._const_by_digest.clear()

        with pytest.raises(AssertionError, match="衝突"):
            converter._add_const(torch.zeros(3), "テスト定数")


class TestDeclarations:
    """宣言はちょうど 1 箇所（入力は inputs[]、それ以外は values{}）。"""

    def test_inputs_are_not_declared_in_values(self, convert_module, dyn_t):
        graph, _ = convert_module(Scale(), (torch.randn(6, 4),), ({0: dyn_t},))

        assert [spec.name for spec in graph.inputs] == ["x"]
        assert "x" not in graph.values

    def test_every_node_output_is_declared(self, convert_module, dyn_t):
        graph, _ = convert_module(Scale(), (torch.randn(6, 4),), ({0: dyn_t},))

        for node in graph.nodes:
            for out in node.outs:
                assert out in graph.values

    def test_required_ops_are_derived_from_the_nodes(self, convert_module):
        class Mixed(nn.Module):
            def forward(self, x):
                return torch.relu(x) + torch.tanh(x)

        graph, _ = convert_module(Mixed(), (torch.randn(3, 4),))

        assert graph.required_ops == ["add", "relu", "tanh"]
        assert set(graph.required_ops) <= EMITTABLE_OPS


class TestInitializerProvenance:
    """パラメータ・バッファ・lifted 定数はいずれも FQN のテンソルキーで運ぶ。"""

    def test_parameters_buffers_and_lifted_constants_all_become_initializers(self, convert_module):
        class WithState(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4))
                self.register_buffer("shift", torch.randn(4))
                self.gain = torch.randn(4)  # 素の属性 → lifted tensor constant

            def forward(self, x):
                return x * self.weight + self.shift - self.gain

        graph, tensors = convert_module(WithState(), (torch.randn(3, 4),))

        assert set(tensors) == {"weight", "shift", "gain"}
        assert {init.tensor for init in graph.initializers.values()} == set(tensors)

    def test_non_f32_initializer_is_rejected(self, convert_module):
        class BoolBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mask", torch.zeros(4, dtype=torch.bool))

            def forward(self, x):
                return x + self.mask

        with pytest.raises(NotImplementedError, match="は IR v1 の initializer にできない"):
            convert_module(BoolBuffer(), (torch.randn(3, 4),))


class TestDtypes:
    def test_dtype_outside_the_semantic_vocabulary_is_rejected(self, convert_module):
        with pytest.raises(NotImplementedError, match="意味論 dtype 語彙に無い"):
            convert_module(Scale(), (torch.randn(3, 4, dtype=torch.float64),))


class TestSymbols:
    """IR に載る名前は呼び出し側が与え、user 入力の出現順で割り当てる。"""

    def test_static_graph_declares_no_symbols(self, convert_module):
        graph, _ = convert_module(Scale(), (torch.randn(3, 4),))

        assert graph.symbols == []
        assert graph.inputs[0].shape == [3, 4]

    def test_symbolic_dim_is_written_in_the_canonical_spelling(self, convert_module, dyn_t):
        graph, _ = convert_module(Scale(), (torch.randn(6, 4),), ({0: dyn_t},))

        assert graph.symbols == ["T"]
        assert graph.inputs[0].shape == ["T", 4]
        assert graph.values[graph.outputs[0]].shape == ["T", 4]

    def test_symbols_follow_the_input_order(self, convert_module):
        class CrossScores(nn.Module):
            def forward(self, target, source):
                return target @ source

        graph, _ = convert_module(
            CrossScores(),
            (torch.randn(6, 4), torch.randn(4, 5)),
            ({0: Dim("A", min=2, max=16)}, {1: Dim("B", min=2, max=16)}),
            symbol_names=("Ttgt", "Tsrc"),
        )

        assert graph.symbols == ["Ttgt", "Tsrc"]
        assert [spec.shape for spec in graph.inputs] == [["Ttgt", 4], [4, "Tsrc"]]
        assert graph.values[graph.outputs[0]].shape == ["Ttgt", "Tsrc"]

    def test_missing_symbol_name_is_rejected(self, convert_module):
        class CrossScores(nn.Module):
            def forward(self, target, source):
                return target @ source

        with pytest.raises(NotImplementedError) as err:
            convert_module(
                CrossScores(),
                (torch.randn(6, 4), torch.randn(4, 5)),
                ({0: Dim("A", min=2, max=16)}, {1: Dim("B", min=2, max=16)}),
            )

        assert "IR シンボル名が足りない" in str(err.value)
        assert "symbol_names" in str(err.value)

    def test_duplicate_symbol_names_are_rejected(self, convert_module):
        with pytest.raises(ValueError, match="symbol_names に重複"):
            convert_module(Scale(), (torch.randn(3, 4),), symbol_names=("T", "T"))

    @pytest.mark.parametrize("name", ["2T", "T+1", "", "T-1"])
    def test_symbol_name_outside_the_dim_grammar_is_rejected(self, convert_module, name):
        """名前が次元表記の文法に載らないと、書けてもランタイムが読めない次元になる。"""
        with pytest.raises(ValueError, match="次元表記の文法に載らない"):
            convert_module(Scale(), (torch.randn(3, 4),), symbol_names=(name,))


class TestOutputs:
    """出力列は graph_signature の user 出力だけで構成する。"""

    def test_multiple_user_outputs_keep_their_order(self, convert_module):
        class TwoOutputs(nn.Module):
            def forward(self, x):
                return torch.relu(x), torch.tanh(x)

        graph, _ = convert_module(TwoOutputs(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["relu", "tanh"]
        assert graph.outputs == [node.outs[0] for node in graph.nodes]

    def test_buffer_mutation_is_rejected(self, convert_module):
        """IR v1 は状態更新を表現できない — 黙って捨てず未対応として落とす。"""

        class Accumulating(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("total", torch.zeros(4))

            def forward(self, x):
                self.total.add_(x.sum(dim=-1))
                return torch.relu(x)

        with pytest.raises(NotImplementedError) as err:
            convert_module(Accumulating(), (torch.randn(4, 3),))

        assert "BUFFER_MUTATION" in str(err.value)


class TestDtypeBoundary:
    """torch 既定の i64 を IR の i32 へ落とす境界（ADR 0009）。"""

    def test_int64_input_is_declared_as_i32(self, convert_module, dyn_t):
        class MaskProduct(nn.Module):
            def forward(self, col, row):
                return col * row

        graph, _ = convert_module(
            MaskProduct(),
            (torch.ones(6, 1, dtype=torch.int64), torch.ones(1, 6, dtype=torch.int64)),
            ({0: dyn_t}, {1: dyn_t}),
        )

        assert [spec.dtype for spec in graph.inputs] == ["i32", "i32"]
        assert graph.values[only_node(graph, "mul").outs[0]].dtype == "i32"

    def test_boundary_tensor_normalization_maps_each_dtype(self):
        normalized = normalize_boundary_tensor(torch.tensor([1, -2], dtype=torch.int64), "t")
        assert normalized.dtype is torch.int32
        assert normalized.tolist() == [1, -2]

        # bool は u32 の 0/1（GPU 格納と同じ規約）
        flags = normalize_boundary_tensor(torch.tensor([True, False]), "t")
        assert flags.dtype is torch.uint32
        assert flags.tolist() == [1, 0]

        floats = torch.tensor([1.5, -2.5])
        assert normalize_boundary_tensor(floats, "t").dtype is torch.float32

    def test_out_of_range_int64_fails_loudly(self):
        """黙って切り詰めると添字が静かにずれる — 値域外は必ず落とす。"""
        too_big = torch.tensor([2**31], dtype=torch.int64)

        with pytest.raises(ValueError, match="i32 に収まらない"):
            normalize_boundary_tensor(too_big, "t")

        with pytest.raises(ValueError, match="i32 に収まらない"):
            normalize_boundary_tensor(torch.tensor([-(2**31) - 1], dtype=torch.int64), "t")

        # 境界値ちょうどは通る
        assert normalize_boundary_tensor(
            torch.tensor([2**31 - 1, -(2**31)], dtype=torch.int64), "t"
        ).tolist() == [2**31 - 1, -(2**31)]

    def test_unsupported_dtype_fails_loudly(self):
        with pytest.raises(NotImplementedError, match="意味論 dtype に落とせない"):
            normalize_boundary_tensor(torch.zeros(2, dtype=torch.float64), "t")


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


class TestCloneAlias:
    """aten.clone は恒等（IR の値は常に連続 — ADR 0011）。IR には出さず別名にする。"""

    def test_contiguous_clone_after_permute_is_absorbed(self, convert_module, dyn_t):
        class Materialize(nn.Module):
            def forward(self, x):
                return x.permute(1, 0, 2).contiguous()

        graph, _ = convert_module(Materialize(), (torch.randn(6, 3, 4),), ({0: dyn_t},))

        # clone は 1 ノードも残らず、permute の出力がそのままグラフ出力になる
        assert node_ops(graph) == ["permute"]
        assert graph.outputs == [graph.nodes[0].outs[0]]

    def test_non_contiguous_memory_format_is_rejected(self, convert_module):
        class ChannelsLast(nn.Module):
            def forward(self, x):
                return torch.clone(x, memory_format=torch.channels_last)

        with pytest.raises(NotImplementedError, match="memory_format"):
            convert_module(ChannelsLast(), (torch.randn(2, 3, 4, 5),))


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


class SymbolicCatJoin(nn.Module):
    """記号軸 cat（self | context）+ **連結後の長さを持つ入力**（recon の DiT 形の縮小版）。

    `gate` の宣言が派生次元 `T+3` になるのが肝で、この形は `Dim("T") + 3` を dynamic_shapes に
    書いたときだけ現れる（ADR 0046 の実測で初めて出た）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 2, bias=False)

    def forward(self, x, context, gate):
        return self.proj(torch.cat([x, context], 1) * gate)


class TestSymbolicCatEndToEnd:
    """記号軸 cat のグラフが export → 書き出し → verify_model まで通る（ADR 0046）。"""

    def test_a_derived_input_dimension_survives_the_range_constraint_scan(self, tmp_path, dyn_t):
        """MUST: 派生次元は range_constraints に **sympy の `Add`** としても並ぶ。

        `Dim("T") + 3` を入力 shape に宣言すると `ep.range_constraints` のキーに `s27 + 3` が
        入る。`.name` を素で読む走査は `AttributeError: 'Add' object has no attribute 'name'`
        で落ち、記号軸 cat のグラフは**変換に入る前に**死ぬ（ADR 0046 の同時修正）。
        派生次元の値域は素のシンボルの値域から決まる冗長情報なので、除いて失うものは無い。
        """
        graph = export_to_file(
            SymbolicCatJoin(),
            (torch.randn(1, 6, 4), torch.randn(1, 3, 4), torch.randn(1, 9, 1)),
            tmp_path / "model.safetensors",
            dynamic_shapes=({1: dyn_t}, None, {1: dyn_t + 3}),
        )

        # 派生次元は入力の宣言としてそのまま残る（束縛源は素の `T` の位置だけ）
        assert [spec.shape for spec in graph.inputs] == [[1, "T", 4], [1, 3, 4], [1, "T+3", 1]]
        assert graph.values[only_node(graph, "cat").outs[0]].shape == [1, "T+3", 4]


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
