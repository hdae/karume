"""コンバータの振る舞い固定。

黙って壊れた IR を書き出す経路（入力欠落・prefix 非可換な焼き込み・語彙外 op の近似）を
実際に踏むモジュールで再現し、fail loudly か正しい表現かのどちらかになることを固定する。
個別 aten op の対応表（aten_handlers.py）側の検証は test_aten_handlers.py。
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
    convert,
    curated_decompositions,
    normalize_boundary_tensor,
)
from karume.normalize import normalize_graph
from karume.ops import EMITTABLE_OPS, NON_EMITTABLE_OPS, TOPK_OP
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

    def test_topk_is_declared_non_emittable_and_really_cannot_be_exported(self, convert_module):
        """`topk` は契約表にあるが **torch から出せない**（ADR 0068 追記）。

        多出力 aten のタプル meta + `operator.getitem` スロット結線は新機構で、sampling の
        実需まで先送りした（裁定済み）。`NON_EMITTABLE_OPS` に席があるだけでは「ハンドラを
        足したのに golden 被覆の門から外れたまま」を防げないので、**落ちること自体**を門に
        する（ハンドラと getitem 結線を入れた日にここが赤くなり、そのとき
        `NON_EMITTABLE_OPS` から外して golden を足す）。

        実測（2026-08-17）: 止まるのは `aten.topk.default` ではなく **`operator.getitem`**。
        torch.export はタプル返しをそのまま返す形でも getitem ノードを挟むので、aten
        ハンドラだけを足しても道は開かない（結線が新機構だという ADR 追記の裏付け）。
        """

        class Topk(nn.Module):
            def forward(self, x):
                values, indices = torch.topk(x, 2, dim=-1)
                return values, indices.float()

        assert TOPK_OP in NON_EMITTABLE_OPS
        assert TOPK_OP not in EMITTABLE_OPS
        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Topk(), (torch.randn(3, 5),))

        assert [key for key in err.value.ops if "getitem" in key], err.value.ops
        assert "getitem" in str(err.value)


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


class TestDuplicateOutputs:
    """同じ IR 値を 2 つの出力へ割り当てる形は `convert()` の時点で落とす。

    出力名は集合（docs/ir-v1.md）で、受理側 `verify.parse_ir_graph` は重複を拒否する。
    検出を受理側まで遅らせると、検証を挟まない `emit.write_model` の直呼びが
    「書けたが読めない」配布形をそのまま据える。
    """

    def test_returning_the_same_value_twice_fails_loudly(self, convert_module):
        """同一 Node の 2 度返し（`_user_outputs` は出力引数を位置でそのまま返す）。"""

        class SameTwice(nn.Module):
            def forward(self, x):
                y = x + 1
                return y, y

        with pytest.raises(NotImplementedError) as err:
            convert_module(SameTwice(), (torch.randn(3, 4),))

        assert "出力" in str(err.value)
        assert "重複" in str(err.value)

    def test_two_structurally_identical_outputs_fail_loudly(self, convert_module):
        """CSE が畳む経路 — 別々の部分木でも `(op, ins, attrs, shape)` が同じなら 1 本になる。"""

        class SameShape(nn.Module):
            def forward(self, x):
                return x + 1, x + 1

        with pytest.raises(NotImplementedError) as err:
            convert_module(SameShape(), (torch.randn(3, 4),))

        assert "重複" in str(err.value)

    def test_distinct_outputs_still_convert(self, convert_module):
        """回帰の逆側 — 値が違えば 2 出力はそのまま通る（門の恒真化を防ぐ検出器）。"""

        class TwoValues(nn.Module):
            def forward(self, x):
                return x + 1, x + 2

        graph, _ = convert_module(TwoValues(), (torch.randn(3, 4),))

        assert len(set(graph.outputs)) == len(graph.outputs) == 2


class TestOutputSpecAgreement:
    """`output_specs` と output 引数の対応は**位置**でしか有効でない（normalize は
    `graph_signature` を更新しないので `output_specs[].arg.name` は実在しない名前を指しうる）。

    本数が食い違ったまま `zip(strict=True)` へ渡すと出力列が黙ってずれるので、その前に落とす。
    """

    def test_a_signature_with_fewer_output_specs_fails_loudly(self):
        class TwoOutputs(nn.Module):
            def forward(self, x):
                return torch.relu(x), torch.tanh(x)

        exported = torch.export.export(TwoOutputs(), (torch.randn(3, 4),), strict=False)
        decomposed = exported.run_decompositions(curated_decompositions())
        normalize_graph(decomposed)
        # user 出力を 1 件削った偽物（specs だけが減り、output 引数は 2 本のまま）。
        decomposed.graph_signature.output_specs = list(decomposed.graph_signature.output_specs)[:-1]

        with pytest.raises(AssertionError, match="本数不一致"):
            convert(decomposed)


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
