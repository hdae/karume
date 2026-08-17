"""エクスポータ側 shape 層の規則
（packages/runtime/tests/fixtures/op-contracts.json では書けない領域）。

適合表（test_ops_conformance.py）が押さえるのは TS 側と**結論が一致しなければならない**
面だけ。ここが押さえるのはその外側 — 束縛前の宣言 shape でしか起きない判定（記号次元の
一致・要素数の因数分解・記号長 conv1d）と、グラフ全体の突合（宣言 ≠ 計算で fail loudly）。
"""

from __future__ import annotations

import pytest

from karume.ir import IrGraph, IrInput, IrNode, IrValue
from karume.ops import (
    _UNARY_ATTRS,
    SCALAR_PARAM_ATTRS,
    OpContractError,
    resolve_op_contract,
)
from karume.shapes import assert_graph_shapes, compute_output_shape


def shape_of(op, ins, **kwargs):
    return compute_output_shape(resolve_op_contract(op), ins, "t", **kwargs)


class TestSymbolicDimensions:
    """記号次元は「同じ式か」でしか比較しない（束縛次第の判定はランタイム側の層）。"""

    def test_matching_symbols_propagate(self):
        assert shape_of("linear", [[1, "T", 192], [192, 192], [192]]) == [1, "T", 192]

    def test_different_symbolic_extents_are_not_silently_unified(self):
        # `T` と `2T` は束縛次第で一致しうるが、宣言だけでは同じ長さと言えない。
        with pytest.raises(OpContractError, match="縮約次元"):
            shape_of("matmul", [["T", "2T"], ["T", 4]])

    def test_a_symbolic_and_a_constant_extent_cannot_be_broadcast(self):
        with pytest.raises(OpContractError, match="broadcast 可否"):
            shape_of("add", [["T", 4], [3, 4]])

    def test_a_symbolic_extent_broadcasts_against_length_one(self):
        assert shape_of("add", [["T", 4], [1, 4]]) == ["T", 4]

    def test_a_symbolic_reduction_axis_is_not_treated_as_empty(self):
        """長さ 0 かどうかは束縛次第 — 記号のまま拒否すると実在のグラフを落とす。"""
        assert shape_of("amax", [[3, "T"]], attrs={"dim": 1}) == [3]

    def test_a_constant_zero_reduction_axis_is_still_rejected(self):
        with pytest.raises(OpContractError, match="長さ 0"):
            shape_of("amax", [[3, 0]], attrs={"dim": 1})

    def test_a_symbolic_axis_can_itself_be_the_reduced_one(self):
        assert shape_of("amin", [["T", 5]], attrs={"dim": 0}) == [5]


class TestReduceAxis:
    """reduce の軸は attrs `dim`（宣言必須・rank 外は fail loudly）。"""

    def test_a_non_last_axis_is_removed_from_the_shape(self):
        assert shape_of("sum", [[2, 384, 5, 7]], attrs={"dim": 1}) == [2, 5, 7]

    def test_the_first_axis_can_be_reduced(self):
        assert shape_of("sum", [[4, 6]], attrs={"dim": 0}) == [6]

    def test_a_missing_axis_is_rejected(self):
        with pytest.raises(OpContractError, match=r"attrs\.dim"):
            shape_of("sum", [[6, 10]])

    def test_an_out_of_rank_axis_is_rejected(self):
        with pytest.raises(OpContractError, match="範囲外"):
            shape_of("sum", [[6, 10]], attrs={"dim": 2})

    def test_a_zero_length_axis_that_is_not_reduced_is_accepted(self):
        assert shape_of("amax", [[0, 3]], attrs={"dim": 1}) == [0]


class TestScalarAttrInvariants:
    """キーを跨ぐ attrs の不変条件は shape 層でしか見られない（スキーマはキー単位の検査）。"""

    def test_clamp_accepts_bounds_in_order(self):
        assert shape_of("clamp", [[2, 3]], attrs={"min": -1, "max": 1}) == [2, 3]

    def test_clamp_rejects_inverted_bounds(self):
        # WGSL の clamp は lo > hi で未定義（TS 側 computeOutputShape と同じ門）。
        with pytest.raises(OpContractError, match="min"):
            shape_of("clamp", [[2, 3]], attrs={"min": 1, "max": -1})

    def test_a_missing_scalar_attr_is_rejected_by_the_shape_layer_too(self):
        with pytest.raises(OpContractError):
            shape_of("leaky_relu", [[4]], attrs={})

    def test_the_params_table_and_the_attrs_schema_cover_the_same_unary_ops(self):
        """スカラ attr を宣言した単項 op は必ず params 表にも載る（逆も）。

        attrs スキーマ（`_UNARY_ATTRS`）と params のレイアウト（`SCALAR_PARAM_ATTRS`）は
        別々に書かれた 2 表で、片方だけに op を足しても**例外は出ない**:
        params 表に無い側は `scalar_param_values` が空リストを返して attr の値が黙って
        カーネルへ届かず、attrs スキーマに無い側は宣言そのものが契約外 attrs 扱いになる。
        """
        assert set(SCALAR_PARAM_ATTRS) == set(_UNARY_ATTRS)

    def test_symbolic_shapes_pass_through_unary_ops(self):
        assert shape_of("leaky_relu", [[1, "T", 4]], attrs={"negative_slope": 0.1}) == [
            1,
            "T",
            4,
        ]


class TestWhereAndCumsum:
    """波3 の 2 kind（三項 broadcast / 最終次元固定の前縁和）。"""

    def test_where_broadcasts_all_three_operands(self):
        assert shape_of("where", [["T", 1], [1, 5], [1, 1]]) == ["T", 5]

    def test_where_rejects_operands_that_cannot_broadcast(self):
        with pytest.raises(OpContractError, match="broadcast"):
            shape_of("where", [[2], [3], [1]])

    def test_cumsum_keeps_the_symbolic_shape(self):
        assert shape_of("cumsum", [[1, 3, "T"]], attrs={"dim": 2}) == [1, 3, "T"]

    def test_cumsum_rejects_a_non_final_axis(self):
        with pytest.raises(OpContractError, match="最終次元"):
            shape_of("cumsum", [[2, 5]], attrs={"dim": 0})


class TestReshapeElementCount:
    """要素数は一次式の積。原始一次式の多重集合で比較する（`2T+2` = `2·(T+1)`）。"""

    def test_symbolic_element_counts_match_after_factorisation(self):
        assert shape_of("reshape", [["2T+2"]], declared=[2, "T+1"]) == [2, "T+1"]

    def test_a_missing_symbolic_factor_is_rejected(self):
        with pytest.raises(OpContractError, match="要素数"):
            shape_of("reshape", [[1, "T", 1024]], declared=[1024])

    def test_a_different_coefficient_is_rejected(self):
        with pytest.raises(OpContractError, match="要素数"):
            shape_of("reshape", [["2T"]], declared=["T", 2, 2])

    def test_a_zero_extent_collapses_the_element_count_on_both_sides(self):
        # 数値 shape での `0 × 何か = 0` と結論を揃える（片側だけ記号でも 0 は 0）。
        assert shape_of("reshape", [[0, "T"]], declared=[0]) == [0]


#: conv1d の attrs は 4 つとも宣言必須（既定値補完なし — ADR 0015）。
_CONV = {"stride": 1, "padding": 0, "dilation": 1, "groups": 1}


class TestSymbolicConv1d:
    def test_stride_one_keeps_the_symbolic_length(self):
        assert shape_of(
            "conv1d",
            [[1, 192, "P"], [192, 192, 3], [192]],
            attrs={**_CONV, "padding": 1},
        ) == [1, 192, "P"]

    def test_a_stride_that_divides_the_coefficient_stays_canonical(self):
        assert shape_of("conv1d", [[1, 4, "2T"], [6, 4, 2], [6]], attrs={**_CONV, "stride": 2}) == [
            1,
            6,
            "T",
        ]

    def test_a_stride_that_does_not_divide_the_coefficient_fails_loudly(self):
        # floor((T + …) / 2) は 1 次元 1 シンボルの一次式にならない — 黙って近似しない。
        with pytest.raises(OpContractError, match="一次式にならない"):
            shape_of("conv1d", [[1, 4, "T"], [6, 4, 2], [6]], attrs={**_CONV, "stride": 2})

    def test_a_negative_offset_fails_loudly(self):
        # K が大きいと `T - 4` 形になり、非負オフセットの正準文法に載らない。
        with pytest.raises(OpContractError, match="正準文法"):
            shape_of("conv1d", [[1, 4, "T"], [6, 4, 6], [6]], attrs=_CONV)

    def test_a_symbolic_kernel_length_fails_loudly(self):
        with pytest.raises(OpContractError, match="カーネル長"):
            shape_of("conv1d", [[1, 4, 8], [6, 4, "T"], [6]], attrs=_CONV)

    def test_dilation_widens_the_kernel_span(self):
        # 張りは D·(K−1)+1 = 7 なので、padding 3 で長さが保たれる（DDSConv と同じ関係）。
        assert shape_of(
            "conv1d",
            [[1, 6, "T"], [6, 1, 4], [6]],
            attrs={**_CONV, "padding": 3, "dilation": 2, "groups": 6},
        ) == [1, 6, "T"]

    def test_a_symbolic_channel_axis_fails_loudly(self):
        # groups の割り切りを判定できない（実測にチャネル記号は無い — 黙って通さない）。
        with pytest.raises(OpContractError, match="記号"):
            shape_of("conv1d", [[1, "T", 8], ["T", "T", 3], ["T"]], attrs=_CONV)


class TestSymbolicConvTranspose1d:
    def test_the_symbolic_length_is_multiplied_by_the_stride(self):
        assert shape_of(
            "conv_transpose1d",
            [[1, 3, "T"], [3, 2, 16], [2]],
            attrs={"stride": 8, "padding": 4},
        ) == [1, 2, "8T"]

    def test_both_the_coefficient_and_the_offset_scale(self):
        assert shape_of(
            "conv_transpose1d",
            [[1, 3, "2T+1"], [3, 2, 8], [2]],
            attrs={"stride": 2, "padding": 3},
        ) == [1, 2, "4T+2"]

    def test_the_general_output_length_is_rejected(self):
        # 2P != K − S の形は出力長が L·stride にならない — 一般形は見送り（ADR 0015）。
        with pytest.raises(OpContractError, match="2·padding"):
            shape_of(
                "conv_transpose1d",
                [[1, 3, "T"], [3, 2, 5], [2]],
                attrs={"stride": 2, "padding": 1},
            )

    def test_a_transposed_weight_layout_is_rejected(self):
        # [Cout, Cin, K] と読む取り違えは**非対称チャネル**でのみ赤くなる。
        with pytest.raises(OpContractError, match=r"\[Cin, Cout, K\]"):
            shape_of(
                "conv_transpose1d",
                [[1, 5, "T"], [3, 5, 2], [5]],
                attrs={"stride": 2, "padding": 0},
            )

    def test_a_symbolic_kernel_length_fails_loudly(self):
        with pytest.raises(OpContractError, match="カーネル長"):
            shape_of(
                "conv_transpose1d",
                [[1, 3, 8], [3, 2, "T"], [2]],
                attrs={"stride": 2, "padding": 0},
            )


class TestSymPrefixSlice:
    def test_the_prefix_length_comes_from_the_attrs(self):
        attrs = {"sym": "T", "slices": [{"dim": 1, "coeff": 2, "offset": 1}]}
        assert shape_of("sym_prefix_slice", [[4, 512]], attrs=attrs) == [4, "2T+1"]

    def test_a_symbolic_input_extent_is_rejected(self):
        """入力は Tmax で焼いた静的形（ADR 0010）— 記号入力は prefix の意味を壊す。"""
        attrs = {"sym": "T", "slices": [{"dim": 0, "coeff": 1, "offset": 0}]}

        with pytest.raises(OpContractError, match="記号"):
            shape_of("sym_prefix_slice", [["T", 4]], attrs=attrs)


#: shape 層は attrs の scale も必ず引く（値域の検査を通すための最小の attrs）。
_ATTENTION = {"scale": 0.5}


class TestAttentionGqaBroadcast:
    """GQA の整除 broadcast（ADR 0067 決定 1）— 適合表に書けない 2 面。

    受理 / 拒否の集合そのものは適合表（op-contracts.json の GQA 9 ケース）が TS 側と
    突き合わせる。表では押さえられないのは:

    1. **`Hkv = 0`** — 両実装とも剰余より**先**に `Hkv ≥ 1` を見るが、その順序が崩れたときの
       壊れ方が違う: TS は `H % 0` が NaN で条件が真になり契約エラーへ落ちるのに対し、Python の
       `%` は ZeroDivisionError を投げる（表が書けるのは「throws」までで例外の型は書けない）。
    2. **記号次元の H / Hkv** — 束縛前の宣言 shape でしか現れない（数値へ解決した時点で
       記号だった事実が消える）。

    整数形の拒否 4 件は表と重なるが、条件の枝が 1 箇所に並ぶと「順序が崩れたときどの枝が
    壊れたか」がここで読めるので、GQA の枝の回帰として同じクラスに置く。
    """

    def _shape_of(self, heads, kv_heads, value_heads=None):
        return shape_of(
            "attention",
            [
                [2, heads, 5, 4],
                [2, kv_heads, 7, 4],
                [2, kv_heads if value_heads is None else value_heads, 7, 4],
            ],
            attrs=_ATTENTION,
        )

    def test_a_divisible_head_count_broadcasts_to_the_query_heads(self):
        assert self._shape_of(4, 2) == [2, 4, 5, 4]

    def test_a_single_kv_head_is_the_mqa_form(self):
        assert self._shape_of(8, 1) == [2, 8, 5, 4]

    def test_equal_head_counts_stay_accepted(self):
        assert self._shape_of(3, 3) == [2, 3, 5, 4]

    def test_a_zero_kv_head_count_raises_the_contract_error_not_a_division_error(self):
        """MUST: 剰余より先に `Hkv == 0` を見る（ZeroDivisionError は契約エラーではない）。"""
        with pytest.raises(OpContractError, match="正の整数倍"):
            self._shape_of(4, 0)

    @pytest.mark.parametrize(
        ("heads", "kv_heads", "why"),
        [
            (0, 2, "H = 0（`0 % Hkv == 0` は整除を満たすので H ≥ Hkv を別条件で見る）"),
            (0, 0, "(H,Hkv) = (0,0)（H = Hkv は等値で整除枝を短絡するので Hkv ≥ 1 を先に見る）"),
            (4, 3, "H が Hkv の整数倍でない"),
            (2, 4, "Hkv > H（broadcast の向きが逆 — q 側を増やす形は語彙に無い）"),
        ],
        ids=("zero-h", "zero-both", "indivisible", "reversed"),
    )
    def test_a_head_count_that_is_not_a_positive_multiple_is_rejected(self, heads, kv_heads, why):
        with pytest.raises(OpContractError, match="正の整数倍"):
            self._shape_of(heads, kv_heads)
        assert why  # 失敗時に形の意図が読めるようにパラメータへ残す

    def test_a_kv_head_mismatch_between_k_and_v_is_rejected(self):
        """GQA でも k / v 間の Hkv は完全一致（緩めたのは q との関係だけ）。"""
        with pytest.raises(OpContractError, match=r"Hkv（k / v の軸 1）"):
            self._shape_of(4, 2, value_heads=4)

    def test_a_symbolic_head_count_that_matches_structurally_is_accepted(self):
        """同じ式なら r=1 として通る（記号のまま「同じ長さか」は判定できる）。"""
        assert self._shape_of("T", "T") == [2, "T", 5, 4]

    @pytest.mark.parametrize(
        ("heads", "kv_heads"),
        [("T", 2), (4, "T"), ("2T", "T")],
        ids=("symbolic-h", "symbolic-hkv", "both-symbolic"),
    )
    def test_head_counts_that_need_a_binding_to_divide_are_rejected(self, heads, kv_heads):
        """束縛次第で整除しうる形は黙って通さない（broadcast 可否と同じ規律）。"""
        with pytest.raises(OpContractError, match="宣言だけでは"):
            self._shape_of(heads, kv_heads)


def _graph() -> IrGraph:
    """入力 → linear → relu の最小グラフ（宣言はすべて正しい）。"""
    return IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 7])],
        outputs=["y"],
        values={
            "w": IrValue(dtype="f32", shape=[3, 7]),
            "b": IrValue(dtype="f32", shape=[3]),
            "h": IrValue(dtype="f32", shape=["T", 3]),
            "y": IrValue(dtype="f32", shape=["T", 3]),
        },
        nodes=[
            IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={}),
            IrNode(op="relu", ins=["h"], outs=["y"], attrs={}),
        ],
    )


class TestLayoutAxisIsStatic:
    """レイアウト第 2 群の対象軸の記号規則。

    `slice` / `flip` は**静的軸のみ**（ADR 0014）。`cat` の連結軸だけは ADR 0046 が
    「〈定数〉または〈**同一**シンボルの一次式〉」まで緩め、拒否は**異シンボルの混在**に
    差し替わった。

    束縛前の宣言 shape でしか判定できない**拒否**規則なので適合表には書けない（数値へ
    解決した時点で「T と U が別シンボル」という事実が消える）。TS 側は同じ規則を plan.ts が
    持つ — 層は違うが受理集合は同じ。受理側は適合表の shapes 節が両側で踏む。
    """

    def test_a_static_axis_slice_keeps_the_other_symbolic_dimensions(self):
        assert shape_of("slice", [[1, 6, "T"]], attrs={"dim": 1, "start": 3, "end": 6}) == [
            1,
            3,
            "T",
        ]

    def test_a_symbolic_slice_axis_is_rejected(self):
        with pytest.raises(OpContractError, match="記号次元"):
            shape_of("slice", [["T", 4]], attrs={"dim": 0, "start": 0, "end": 2})

    def test_a_static_axis_cat_keeps_the_other_symbolic_dimensions(self):
        assert shape_of("cat", [[1, 3, "T"], [1, 5, "T"]], attrs={"dim": 1}) == [1, 8, "T"]

    def test_a_symbolic_cat_axis_sums_with_a_constant(self):
        """ADR 0046 — `S`+定数 は正準文法にそのまま載る（DiT の KV 連結と同型）。"""
        assert shape_of("cat", [["T", 4], [3, 4]], attrs={"dim": 0}) == ["T+3", 4]

    def test_the_same_symbol_on_both_sides_accumulates_the_coefficient(self):
        assert shape_of("cat", [["T", 4], ["T", 4]], attrs={"dim": 0}) == ["2T", 4]
        assert shape_of("cat", [["2T+1", 4], [4, 4], ["T", 4]], attrs={"dim": 0}) == ["3T+5", 4]

    def test_mixing_different_symbols_on_the_cat_axis_is_rejected(self):
        """異シンボルの和は 1 次元 1 シンボルの次元言語に載らない（ADR 0046 の唯一の拒否）。"""
        with pytest.raises(OpContractError, match="異なるシンボル"):
            shape_of("cat", [["T", 4], ["U", 4]], attrs={"dim": 0})

    def test_a_mismatched_symbolic_dimension_off_the_cat_axis_is_rejected(self):
        with pytest.raises(OpContractError, match="軸 2 で違う"):
            shape_of("cat", [[1, 3, "T"], [1, 5, "2T"]], attrs={"dim": 1})

    def test_a_symbolic_flip_axis_is_rejected(self):
        with pytest.raises(OpContractError, match="記号次元"):
            shape_of("flip", [[1, "T"]], attrs={"dim": 1})

    def test_pad_extends_a_symbolic_last_dimension(self):
        """pad の最終次元は記号でよい（実測の T+2w = T+8 — 次元言語のオフセット付き形）。"""
        assert shape_of("pad", [[1, 2, "T"]], attrs={"left": 4, "right": 4}) == [1, 2, "T+8"]

    def test_pad_keeps_the_coefficient_of_a_symbolic_last_dimension(self):
        assert shape_of("pad", [["2T+1"]], attrs={"left": 1, "right": 2}) == ["2T+4"]


class TestGraphAgreement:
    def test_a_consistent_graph_passes(self):
        # 例外が出ないことが仕様（返り値は無い）。
        assert assert_graph_shapes(_graph()) is None

    def test_a_declared_output_that_contradicts_the_contract_is_rejected(self):
        """宣言（torch の meta 由来）を正としない — 契約から計算した shape と突き合わせる。"""
        graph = _graph()
        graph.values["h"] = IrValue(dtype="f32", shape=["T", 4])

        with pytest.raises(OpContractError, match="計算 shape"):
            assert_graph_shapes(graph)

    def test_a_symbol_that_drops_out_of_the_declaration_is_rejected(self):
        graph = _graph()
        graph.values["y"] = IrValue(dtype="f32", shape=[6, 3])

        with pytest.raises(OpContractError, match="計算 shape"):
            assert_graph_shapes(graph)

    def test_an_undeclared_input_value_is_rejected(self):
        graph = _graph()
        del graph.values["w"]

        with pytest.raises(OpContractError, match="宣言が無い"):
            assert_graph_shapes(graph)
