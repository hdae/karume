"""Rules of the exporter's shape layer
(the area packages/runtime/tests/fixtures/op-contracts.json cannot express).

The conformance table (test_ops_conformance.py) covers only the surface where the conclusion
**must agree** with the TS side. This file covers what lies outside it — judgements that arise
only on declared shapes before binding (equality of symbolic dimensions, factorization of element
counts, symbolic-length conv1d), and the whole-graph agreement check (declared ≠ computed fails
loudly).
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


def shapes_of(op, ins, **kwargs):
    """Shapes in ascending output-slot order (ADR 0068 decision 1 — only topk returns two)."""
    return compute_output_shape(resolve_op_contract(op), ins, "t", **kwargs)


def shape_of(op, ins, **kwargs):
    """Output shape of a single-output op (multi-output ops use shapes_of).

    MUST: check the count before returning slot 0. Silently returning only slot 0 of a
    multi-output op would allow tests whose second shape nobody checks.
    """
    outs = shapes_of(op, ins, **kwargs)
    assert len(outs) == 1, f"op '{op}' has {len(outs)} outputs (receive them with shapes_of)"
    return outs[0]


class TestSymbolicDimensions:
    """Symbolic dimensions compare only as "the same expression" (binding-dependent judgements
    belong to the runtime layer)."""

    def test_matching_symbols_propagate(self):
        assert shape_of("linear", [[1, "T", 192], [192, 192], [192]]) == [1, "T", 192]

    def test_different_symbolic_extents_are_not_silently_unified(self):
        # `T` and `2T` may agree depending on the binding, but the declaration alone cannot
        # call them the same length.
        with pytest.raises(OpContractError, match="contraction dimension"):
            shape_of("matmul", [["T", "2T"], ["T", 4]])

    def test_a_symbolic_and_a_constant_extent_cannot_be_broadcast(self):
        with pytest.raises(OpContractError, match="whether they broadcast"):
            shape_of("add", [["T", 4], [3, 4]])

    def test_a_symbolic_extent_broadcasts_against_length_one(self):
        assert shape_of("add", [["T", 4], [1, 4]]) == ["T", 4]

    def test_a_symbolic_reduction_axis_is_not_treated_as_empty(self):
        """Whether the length is 0 depends on the binding — rejecting the symbol would fail real
        graphs."""
        assert shape_of("amax", [[3, "T"]], attrs={"dim": 1}) == [3]

    def test_a_constant_zero_reduction_axis_is_still_rejected(self):
        with pytest.raises(OpContractError, match="length 0"):
            shape_of("amax", [[3, 0]], attrs={"dim": 1})

    def test_a_symbolic_axis_can_itself_be_the_reduced_one(self):
        assert shape_of("amin", [["T", 5]], attrs={"dim": 0}) == [5]


class TestReduceAxis:
    """The reduce axis is attrs `dim` (declaration required; out of rank fails loudly)."""

    def test_a_non_last_axis_is_removed_from_the_shape(self):
        assert shape_of("sum", [[2, 384, 5, 7]], attrs={"dim": 1}) == [2, 5, 7]

    def test_the_first_axis_can_be_reduced(self):
        assert shape_of("sum", [[4, 6]], attrs={"dim": 0}) == [6]

    def test_a_missing_axis_is_rejected(self):
        with pytest.raises(OpContractError, match=r"attrs\.dim"):
            shape_of("sum", [[6, 10]])

    def test_an_out_of_rank_axis_is_rejected(self):
        with pytest.raises(OpContractError, match="out of range"):
            shape_of("sum", [[6, 10]], attrs={"dim": 2})

    def test_a_zero_length_axis_that_is_not_reduced_is_accepted(self):
        assert shape_of("amax", [[0, 3]], attrs={"dim": 1}) == [0]


class TestScalarAttrInvariants:
    """Invariants across attrs keys show only in the shape layer (the schema checks per key)."""

    def test_clamp_accepts_bounds_in_order(self):
        assert shape_of("clamp", [[2, 3]], attrs={"min": -1, "max": 1}) == [2, 3]

    def test_clamp_rejects_inverted_bounds(self):
        # WGSL clamp is undefined for lo > hi (the same check as computeOutputShape on the TS side).
        with pytest.raises(OpContractError, match="min"):
            shape_of("clamp", [[2, 3]], attrs={"min": 1, "max": -1})

    def test_a_missing_scalar_attr_is_rejected_by_the_shape_layer_too(self):
        with pytest.raises(OpContractError):
            shape_of("leaky_relu", [[4]], attrs={})

    def test_the_params_table_and_the_attrs_schema_cover_the_same_unary_ops(self):
        """A unary op declaring a scalar attr is always in the params table too (and vice versa).

        The attrs schema (`_UNARY_ATTRS`) and the params layout (`SCALAR_PARAM_ATTRS`) are two
        tables written separately, and adding an op to only one of them **raises nothing**:
        missing from the params table, `scalar_param_values` returns an empty list and the attr
        value silently never reaches the kernel; missing from the attrs schema, the declaration
        itself is treated as attrs outside the contract.
        """
        assert set(SCALAR_PARAM_ATTRS) == set(_UNARY_ATTRS)

    def test_symbolic_shapes_pass_through_unary_ops(self):
        assert shape_of("leaky_relu", [[1, "T", 4]], attrs={"negative_slope": 0.1}) == [
            1,
            "T",
            4,
        ]


class TestWhereAndCumsum:
    """The two kinds of phase 3 (ternary broadcast / prefix sum over the fixed last dimension)."""

    def test_where_broadcasts_all_three_operands(self):
        assert shape_of("where", [["T", 1], [1, 5], [1, 1]]) == ["T", 5]

    def test_where_rejects_operands_that_cannot_broadcast(self):
        with pytest.raises(OpContractError, match="broadcast"):
            shape_of("where", [[2], [3], [1]])

    def test_cumsum_keeps_the_symbolic_shape(self):
        assert shape_of("cumsum", [[1, 3, "T"]], attrs={"dim": 2}) == [1, 3, "T"]

    def test_cumsum_rejects_a_non_final_axis(self):
        with pytest.raises(OpContractError, match="last dimension"):
            shape_of("cumsum", [[2, 5]], attrs={"dim": 0})


class TestReshapeElementCount:
    """An element count is a product of linear expressions, compared as a multiset of primitive
    linear expressions (`2T+2` = `2·(T+1)`)."""

    def test_symbolic_element_counts_match_after_factorisation(self):
        assert shape_of("reshape", [["2T+2"]], declared=[2, "T+1"]) == [2, "T+1"]

    def test_a_missing_symbolic_factor_is_rejected(self):
        with pytest.raises(OpContractError, match="element count"):
            shape_of("reshape", [[1, "T", 1024]], declared=[1024])

    def test_a_different_coefficient_is_rejected(self):
        with pytest.raises(OpContractError, match="element count"):
            shape_of("reshape", [["2T"]], declared=["T", 2, 2])

    def test_a_zero_extent_collapses_the_element_count_on_both_sides(self):
        # Agrees with `0 × anything = 0` on numeric shapes (0 is 0 even if only one side is
        # symbolic).
        assert shape_of("reshape", [[0, "T"]], declared=[0]) == [0]


#: All four conv1d attrs must be declared (no default filling — ADR 0015).
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
        # floor((T + …) / 2) is not a one-symbol linear expression per dimension — no silent
        # approximation.
        with pytest.raises(OpContractError, match="does not give a linear output length"):
            shape_of("conv1d", [[1, 4, "T"], [6, 4, 2], [6]], attrs={**_CONV, "stride": 2})

    def test_a_negative_offset_fails_loudly(self):
        # A large K gives the `T - 4` form, which the canonical grammar of non-negative offsets
        # cannot express.
        with pytest.raises(OpContractError, match="canonical grammar"):
            shape_of("conv1d", [[1, 4, "T"], [6, 4, 6], [6]], attrs=_CONV)

    def test_a_symbolic_kernel_length_fails_loudly(self):
        with pytest.raises(OpContractError, match="kernel length"):
            shape_of("conv1d", [[1, 4, 8], [6, 4, "T"], [6]], attrs=_CONV)

    def test_dilation_widens_the_kernel_span(self):
        # The span is D·(K−1)+1 = 7, so padding 3 keeps the length (the same relation as DDSConv).
        assert shape_of(
            "conv1d",
            [[1, 6, "T"], [6, 1, 4], [6]],
            attrs={**_CONV, "padding": 3, "dilation": 2, "groups": 6},
        ) == [1, 6, "T"]

    def test_a_symbolic_channel_axis_fails_loudly(self):
        # Divisibility by groups cannot be decided (no observed model has a channel symbol — not
        # let through silently).
        with pytest.raises(OpContractError, match="symbolic"):
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
        # With 2P != K − S the output length is not L·stride — the general form is deferred
        # (ADR 0015).
        with pytest.raises(OpContractError, match="2·padding"):
            shape_of(
                "conv_transpose1d",
                [[1, 3, "T"], [3, 2, 5], [2]],
                attrs={"stride": 2, "padding": 1},
            )

    def test_a_transposed_weight_layout_is_rejected(self):
        # Misreading it as [Cout, Cin, K] turns red only with **asymmetric channels**.
        with pytest.raises(OpContractError, match=r"\[Cin, Cout, K\]"):
            shape_of(
                "conv_transpose1d",
                [[1, 5, "T"], [3, 5, 2], [5]],
                attrs={"stride": 2, "padding": 0},
            )

    def test_a_symbolic_kernel_length_fails_loudly(self):
        with pytest.raises(OpContractError, match="kernel length"):
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
        """The input is a static shape fixed at Tmax (ADR 0010) — a symbolic input breaks the
        meaning of the prefix."""
        attrs = {"sym": "T", "slices": [{"dim": 0, "coeff": 1, "offset": 0}]}

        with pytest.raises(OpContractError, match="symbolic"):
            shape_of("sym_prefix_slice", [["T", 4]], attrs=attrs)


#: The shape layer always reads the attrs scale too (the minimal attrs that pass the range
#: check).
_ATTENTION = {"scale": 0.5}


class TestAttentionGqaBroadcast:
    """GQA divisible broadcast (ADR 0067 decision 1) — the two surfaces the conformance table
    cannot express.

    The accept / reject sets themselves are checked against the TS side by the conformance table
    (the 9 GQA cases in op-contracts.json). What the table cannot cover:

    1. **`Hkv = 0`** — both implementations check `Hkv ≥ 1` **before** the remainder, but they
       break differently when that order is lost: in TS `H % 0` is NaN, the condition is true and
       it falls into the contract error, whereas Python's `%` raises ZeroDivisionError (the table
       can say "throws" but not the exception type).
    2. **Symbolic H / Hkv** — they appear only on declared shapes before binding (the fact that
       they were symbols disappears once resolved to numbers).

    The 4 integer-shape rejections overlap the table, but with the condition branches side by
    side one can read here "which branch broke when the order was lost", so they sit in the same
    class as regressions of the GQA branches.
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
        """MUST: check `Hkv == 0` before the remainder (ZeroDivisionError is no contract error)."""
        with pytest.raises(OpContractError, match="positive integer multiple"):
            self._shape_of(4, 0)

    @pytest.mark.parametrize(
        ("heads", "kv_heads", "why"),
        [
            (0, 2, "H = 0 (`0 % Hkv == 0` divides, so H ≥ Hkv is checked separately)"),
            (0, 0, "(H,Hkv) = (0,0) (H = Hkv skips the divisibility branch: Hkv ≥ 1 first)"),
            (4, 3, "H is not an integer multiple of Hkv"),
            (2, 4, "Hkv > H (reversed broadcast — widening the q side is not in the vocabulary)"),
        ],
        ids=("zero-h", "zero-both", "indivisible", "reversed"),
    )
    def test_a_head_count_that_is_not_a_positive_multiple_is_rejected(self, heads, kv_heads, why):
        with pytest.raises(OpContractError, match="positive integer multiple"):
            self._shape_of(heads, kv_heads)
        assert why  # kept as a parameter so a failure shows the intent of the shape

    def test_a_kv_head_mismatch_between_k_and_v_is_rejected(self):
        """With GQA too, Hkv matches exactly between k and v (only the q relation was relaxed)."""
        with pytest.raises(OpContractError, match=r"Hkv \(axis 1 of k / v\)"):
            self._shape_of(4, 2, value_heads=4)

    def test_a_symbolic_head_count_that_matches_structurally_is_accepted(self):
        """The same expression passes as r=1 ("the same length" can be decided on symbols)."""
        assert self._shape_of("T", "T") == [2, "T", 5, 4]

    @pytest.mark.parametrize(
        ("heads", "kv_heads"),
        [("T", 2), (4, "T"), ("2T", "T")],
        ids=("symbolic-h", "symbolic-hkv", "both-symbolic"),
    )
    def test_head_counts_that_need_a_binding_to_divide_are_rejected(self, heads, kv_heads):
        """Shapes that divide only depending on the binding are not let through silently (the same
        rule as broadcast eligibility)."""
        with pytest.raises(OpContractError, match="the declaration alone"):
            self._shape_of(heads, kv_heads)


def _graph() -> IrGraph:
    """Minimal input → linear → relu graph (every declaration is correct)."""
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
    """Symbol rules for the target axis of layout group 2.

    `slice` / `flip` take **static axes only** (ADR 0014). Only the concatenation axis of `cat` was
    relaxed by ADR 0046 to "a constant or a linear expression of **one** symbol", and the rejection
    became **mixing different symbols**.

    These are **rejection** rules decidable only on declared shapes before binding, so the
    conformance table cannot express them (the fact that "T and U are different symbols"
    disappears once resolved to numbers). On the TS side plan.ts holds the same rules — a
    different layer, the same accepted set. The accepting side is taken on both sides by the
    shapes section of the conformance table.
    """

    def test_a_static_axis_slice_keeps_the_other_symbolic_dimensions(self):
        assert shape_of("slice", [[1, 6, "T"]], attrs={"dim": 1, "start": 3, "end": 6}) == [
            1,
            3,
            "T",
        ]

    def test_a_symbolic_slice_axis_is_rejected(self):
        with pytest.raises(OpContractError, match="symbolic dimension"):
            shape_of("slice", [["T", 4]], attrs={"dim": 0, "start": 0, "end": 2})

    def test_a_static_axis_cat_keeps_the_other_symbolic_dimensions(self):
        assert shape_of("cat", [[1, 3, "T"], [1, 5, "T"]], attrs={"dim": 1}) == [1, 8, "T"]

    def test_a_symbolic_cat_axis_sums_with_a_constant(self):
        """ADR 0046 — `S` + constant fits the canonical grammar as is (the same form as DiT's KV
        concatenation)."""
        assert shape_of("cat", [["T", 4], [3, 4]], attrs={"dim": 0}) == ["T+3", 4]

    def test_the_same_symbol_on_both_sides_accumulates_the_coefficient(self):
        assert shape_of("cat", [["T", 4], ["T", 4]], attrs={"dim": 0}) == ["2T", 4]
        assert shape_of("cat", [["2T+1", 4], [4, 4], ["T", 4]], attrs={"dim": 0}) == ["3T+5", 4]

    def test_mixing_different_symbols_on_the_cat_axis_is_rejected(self):
        """A sum of different symbols does not fit the one-symbol-per-dimension language (the only
        rejection of ADR 0046)."""
        with pytest.raises(OpContractError, match="different symbols"):
            shape_of("cat", [["T", 4], ["U", 4]], attrs={"dim": 0})

    def test_a_mismatched_symbolic_dimension_off_the_cat_axis_is_rejected(self):
        with pytest.raises(OpContractError, match=r"differs from .* on axis 2"):
            shape_of("cat", [[1, 3, "T"], [1, 5, "2T"]], attrs={"dim": 1})

    def test_a_symbolic_flip_axis_is_rejected(self):
        with pytest.raises(OpContractError, match="symbolic dimension"):
            shape_of("flip", [[1, "T"]], attrs={"dim": 1})

    def test_pad_extends_a_symbolic_last_dimension(self):
        """The last dimension of pad may be symbolic (the observed T+2w = T+8 — the offset form of
        the dimension language)."""
        assert shape_of("pad", [[1, 2, "T"]], attrs={"left": 4, "right": 4}) == [1, 2, "T+8"]

    def test_pad_keeps_the_coefficient_of_a_symbolic_last_dimension(self):
        assert shape_of("pad", [["2T+1"]], attrs={"left": 1, "right": 2}) == ["2T+4"]


class TestTopk:
    """topk is **the only multi-output op** (ADR 0068 decision 3) — both outputs are `[..., k]`.

    Of the accepting region (`1 <= k <= last dimension`), the attrs schema checks the lower bound
    and the shape rule the upper. With a symbolic last dimension the upper-bound check is
    **deferred** (it depends on the binding — the runtime layer decides).
    """

    def test_both_outputs_take_the_last_dimension_to_k(self):
        assert shapes_of("topk", [[6, 10]], attrs={"k": 3}) == [[6, 3], [6, 3]]

    def test_k_equal_to_the_last_dimension_is_the_upper_end(self):
        assert shapes_of("topk", [[4]], attrs={"k": 4}) == [[4], [4]]

    def test_leading_symbolic_dimensions_pass_through(self):
        assert shapes_of("topk", [[1, "T", 5]], attrs={"k": 2}) == [[1, "T", 2], [1, "T", 2]]

    def test_a_symbolic_last_dimension_defers_the_upper_bound(self):
        # k <= dim cannot be asserted on a symbol (the runtime decides after binding).
        assert shapes_of("topk", [[2, "T"]], attrs={"k": 9}) == [[2, 9], [2, 9]]

    def test_the_single_output_helper_refuses_a_multi_output_op(self):
        # Does not silently return only slot 0 of a multi-output op (rules out an unchecked
        # second output).
        with pytest.raises(AssertionError, match="has 2 outputs"):
            shape_of("topk", [[6, 10]], attrs={"k": 2})

    @pytest.mark.parametrize(
        ("attrs", "why"),
        [
            ({"k": 0}, "k=0 is not in the vocabulary (torch accepts it but defines no values)"),
            ({"k": 11}, "k > last dimension"),
            ({}, "k must be declared (static-k)"),
            ({"k": "T"}, "a symbolic k does not fit the static-shape premise"),
            ({"k": 1.5}, "non-integer k"),
            ({"k": True}, "bool derives from int but is not a k"),
        ],
    )
    def test_values_outside_the_accepting_region_are_rejected(self, attrs, why):
        with pytest.raises(OpContractError):
            shapes_of("topk", [[6, 10]], attrs=attrs)

    def test_a_zero_length_last_dimension_is_rejected(self):
        with pytest.raises(OpContractError):
            shapes_of("topk", [[3, 0]], attrs={"k": 1})

    def test_a_scalar_input_is_rejected(self):
        with pytest.raises(OpContractError, match="rank 1 or more"):
            shapes_of("topk", [[]], attrs={"k": 1})


def _topk_graph() -> IrGraph:
    """Minimal graph with topk's two outputs (a multi-output node through `assert_graph_shapes`).

    The exporter does not emit topk (no aten handler — ADR 0068 addendum 2), but the shape check
    that mirrors the contract must be complete for multi-output nodes too.
    """
    return IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 8])],
        outputs=["v", "i"],
        values={
            "v": IrValue(dtype="f32", shape=["T", 3]),
            "i": IrValue(dtype="i32", shape=["T", 3]),
        },
        nodes=[IrNode(op="topk", ins=["x"], outs=["v", "i"], attrs={"k": 3})],
    )


class TestMultiOutputGraphAgreement:
    """Shape check of multi-output nodes (ADR 0068 decision 1 — each slot vs. its declaration)."""

    def test_a_consistent_multi_output_graph_passes(self):
        assert assert_graph_shapes(_topk_graph()) is None

    def test_a_contradicting_second_output_is_rejected(self):
        """Only **slot 1** contradicts the contract (checking slot 0 alone would let it through)."""
        graph = _topk_graph()
        graph.values["i"] = IrValue(dtype="i32", shape=["T", 2])

        with pytest.raises(OpContractError, match="output 'i' computed shape"):
            assert_graph_shapes(graph)

    def test_a_missing_output_slot_is_rejected(self):
        """A graph cut down to one output fails the count check (the list-length check)."""
        graph = _topk_graph()
        graph.outputs = ["v"]
        # IrNode is frozen, so it is rebuilt by replacement (the declaration also cut to one).
        graph.nodes[0] = IrNode(op="topk", ins=["x"], outs=["v"], attrs={"k": 3})
        del graph.values["i"]

        with pytest.raises(OpContractError, match="output shape computation gave 2"):
            assert_graph_shapes(graph)


class TestGraphAgreement:
    def test_a_consistent_graph_passes(self):
        # Raising nothing is the specification (there is no return value).
        assert assert_graph_shapes(_graph()) is None

    def test_a_declared_output_that_contradicts_the_contract_is_rejected(self):
        """The declaration (from torch's meta) is not trusted — it is checked against the shape
        computed from the contract."""
        graph = _graph()
        graph.values["h"] = IrValue(dtype="f32", shape=["T", 4])

        with pytest.raises(OpContractError, match="computed shape"):
            assert_graph_shapes(graph)

    def test_a_symbol_that_drops_out_of_the_declaration_is_rejected(self):
        graph = _graph()
        graph.values["y"] = IrValue(dtype="f32", shape=[6, 3])

        with pytest.raises(OpContractError, match="computed shape"):
            assert_graph_shapes(graph)

    def test_an_undeclared_input_value_is_rejected(self):
        graph = _graph()
        del graph.values["w"]

        with pytest.raises(OpContractError, match="no dtype/shape declaration"):
            assert_graph_shapes(graph)
