"""Pins the converter's behaviour.

Paths that would silently write a broken IR (a missing input, baking that does not commute with the
prefix, approximating an op outside the vocabulary) are reproduced with modules that really take
them, pinning that each ends either in fail loudly or in a correct representation. The checks of
the per-aten-op table (aten_handlers.py) are in test_aten_handlers.py.
"""

from __future__ import annotations

import pytest
import torch
from conftest import SYM_MAX, node_ops, only_node
from torch import nn
from torch.export import Dim

from karume.container import Provenance
from karume.convert import (
    Converter,
    ModuleOrigin,
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
    """Tensor inputs are built from the positional arguments (not relying on the deduplication of
    all_input_nodes)."""

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
        """A scalar operand is promoted to a rank-1 constant by the normalization pass (ADR 0010 /
        recon §4-4)."""

        class Doubling(nn.Module):
            def forward(self, x):
                return x * 2.0

        graph, tensors = convert_module(Doubling(), (torch.randn(3, 4),))

        folded = only_node(graph, "mul").ins[1]
        assert graph.values[folded].shape == [1]
        assert torch.equal(tensors[graph.initializers[folded].tensor], torch.tensor([2.0]))

    def test_type_promoting_scalar_operand_is_still_rejected(self, convert_module):
        """Only forms that keep the dtype are promoted — a form with type promotion fails loudly.

        `i32 tensor × f32 scalar` includes a cast, so turning it into a rank-1 constant would
        silently break the binary ops' contract that "the input dtypes are the same".
        """

        class Scaled(nn.Module):
            def forward(self, mask):
                return mask * 0.5

        with pytest.raises(NotImplementedError, match="scalar operands are not supported"):
            convert_module(Scaled(), (torch.ones(3, 4, dtype=torch.int64),))


class TestUnsupportedOps:
    """aten ops outside the vocabulary are all enumerated and rejected (neither partial
    approximation nor stopping at the first one)."""

    def test_every_unsupported_op_is_listed_at_once(self, convert_module):
        """NOTE: the second example was replaced, `sin` → `atan` (sin entered the IR vocabulary and
        was no longer an example "outside the vocabulary"). The check itself (listing 2 at once)
        is unchanged.
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
        """An op not in the folding allowlist is not folded even when it depends on a symbol (no
        silent approximation).

        NOTE: the example op has been replaced `flip` → `sin` → `floor` (ADR 0014 put flip into the
        runtime's vocabulary and ADR 0016 put sin into the folding allowlist, so each stopped
        being an example of "an op in neither the allowlist nor the table"). The check itself (not
        folding even when symbol-dependent, sending it to the full enumeration) is unchanged.
        """

        class Stepped(nn.Module):
            def forward(self, x):
                steps = torch.floor(torch.arange(x.shape[0], dtype=torch.float32) * 0.5)
                return x + steps.unsqueeze(-1)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Stepped(), (torch.randn(6, 4),), ({0: dyn_t},))

        assert "aten.floor.default" in err.value.ops

    def test_topk_is_declared_non_emittable_and_really_cannot_be_exported(self, convert_module):
        """`topk` is in the contract table but **cannot be emitted from torch** (ADR 0068 addendum).

        Tuple meta of multi-output aten ops + wiring `operator.getitem` slots is a new mechanism,
        deferred until sampling really needs it (ruled). An entry in `NON_EMITTABLE_OPS` alone
        cannot prevent "a handler was added but it stays outside the golden-coverage check", so
        **the failure itself** is the check (this turns red on the day the handler and the getitem
        wiring go in; then it is removed from `NON_EMITTABLE_OPS` and goldens are added).

        Observed (2026-08-17): what stops is not `aten.topk.default` but **`operator.getitem`**.
        torch.export inserts getitem nodes even when the tuple is returned as is, so adding only an
        aten handler does not open the way (supporting the ADR addendum's point that the wiring
        is a new mechanism).
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


class _AtanActivation(nn.Module):
    """A leaf module that emits an op inside Core ATen (tagged `torch.Tag.core`) but not in the IR
    vocabulary."""

    def forward(self, x):
        return torch.atan(x)


class _AtanBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = _AtanActivation()

    def forward(self, x):
        return self.act(x) + x


class _StackedAtanBlocks(nn.Module):
    """The same unsupported op appears several times from nested modules (a miniature of the
    repeated layers of a ported model)."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_AtanBlock(), _AtanBlock()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return torch.lgamma(x)


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class TestUnsupportedOpDiagnostics:
    """The unsupported-op error carries, per op, what a port needs to decide (P4-006).

    Three things: Core ATen membership (the one axis that decides whether adding it to the
    vocabulary needs an ADR — docs/op-vocabulary.md "Procedure for adding an op"), the originating
    module (where to rewrite on the recipe side) and the occurrence count. With all three in one
    error, nothing has to be looked up again per op.
    """

    def test_core_aten_op_carries_its_count_and_innermost_module(self, convert_module):
        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(_StackedAtanBlocks(), (torch.randn(3, 4),))

        report = err.value.reports["aten.atan.default"]
        assert report.category == "core_aten"
        assert report.count == 2
        assert report.origins == (
            ModuleOrigin(
                module_type=_qualified_name(_AtanActivation),
                first_path="blocks.0.act",
                count=2,
            ),
        )
        assert report.untracked == 0
        message = str(err.value)
        assert "aten.atan.default ×2 [Core ATen]" in message
        assert "_AtanActivation ×2 (blocks.0.act)" in message

    def test_op_outside_core_aten_is_told_apart_in_the_same_error(self, convert_module):
        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(_StackedAtanBlocks(), (torch.randn(3, 4),))

        assert set(err.value.reports) == {"aten.atan.default", "aten.lgamma.default"}
        report = err.value.reports["aten.lgamma.default"]
        assert report.category == "non_core_aten"
        assert report.count == 1
        assert report.origins == (
            ModuleOrigin(module_type=_qualified_name(_StackedAtanBlocks), first_path="", count=1),
        )
        message = str(err.value)
        assert "aten.lgamma.default ×1 [outside Core ATen]" in message
        assert "_StackedAtanBlocks ×1 (<root>)" in message

    def test_target_outside_aten_is_not_classified_as_core_aten(self, convert_module):
        class Topk(nn.Module):
            def forward(self, x):
                values, indices = torch.topk(x, 2, dim=-1)
                return values, indices.float()

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Topk(), (torch.randn(3, 5),))

        getitem_reports = [
            report for target, report in err.value.reports.items() if "getitem" in target
        ]
        assert [report.category for report in getitem_reports] == ["non_aten"]
        assert "[outside aten]" in str(err.value)

    def test_node_without_module_stack_is_counted_as_untracked(self):
        """Nodes created by normalize have no `nn_module_stack` — the origin is counted as unknown,
        not made up."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        atan = graph.call_function(torch.ops.aten.atan.default, (x,))
        graph.output(atan)

        error = UnsupportedAtenOpsError([atan])

        report = error.reports["aten.atan.default"]
        assert report.origins == ()
        assert report.untracked == 1
        assert error.ops == {"aten.atan.default": [atan.name]}
        assert "unknown module ×1" in str(error)


class TestConstantFolding:
    """Constant subtrees are baked into initializers (only those that do not depend on a
    symbol)."""

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
        """An integer constant becomes an i32 initializer (raw int32 storage — ADR 0010). i64 goes
        to i32 at the boundary."""

        class IntegerIndex(nn.Module):
            def forward(self, src):
                return torch.gather(src, -1, torch.arange(4).unsqueeze(0).expand(3, 4))

        graph, tensors = convert_module(IntegerIndex(), (torch.randn(3, 4),))

        index = only_node(graph, "gather").ins[1]
        # expand stops at the folding frontier (materializing it would inflate), so the constant
        # is on its input side.
        folded = only_node(graph, "expand").ins[0]
        assert index == only_node(graph, "expand").outs[0]
        assert graph.values[folded].dtype == "i32"
        assert graph.initializers[folded].storage.dtype == "i32"
        assert tensors[graph.initializers[folded].tensor].dtype is torch.int32

    def test_bool_constant_is_rejected(self, convert_module):
        """A bool initializer is not in the IR v1 vocabulary (semantics f32 / i32 only)."""

        class Masked(nn.Module):
            def forward(self, x):
                return x.masked_fill(torch.arange(4) < 2, 0.0)

        with pytest.raises(
            NotImplementedError, match=r"a constant of dtype .* cannot be an IR v1 initializer"
        ):
            convert_module(Masked(), (torch.randn(3, 4),))


class _LastPositionByTensorIndex(nn.Module):
    """`arange(T)[[-1]] >= Tmax−2` — true at Tmax and Tmax−1, false at T=5."""

    def forward(self, x):
        positions = torch.arange(x.shape[0])
        return x + (positions[torch.tensor([-1])] >= SYM_MAX - 2).to(torch.float32)


class _LastPositionBySymbolicIndex(nn.Module):
    """The same hole with an index tensor of symbolic length — indexing with `full((T,), -1)` makes
    every element T−1."""

    def forward(self, x):
        positions = torch.arange(x.shape[0])
        last = torch.full((x.shape[0],), -1)
        return x + (positions[last] >= SYM_MAX - 2).to(torch.float32)


class _LastPositionsByNegativeSlice(nn.Module):
    """`arange(T)[-2:] >= Tmax−3` — both elements true at Tmax and Tmax−1, false at T=5."""

    def forward(self, x):
        positions = torch.arange(x.shape[0])
        return x + (positions[-2:] >= SYM_MAX - 3).to(torch.float32)


class _PositionAcrossConcatenation(nn.Module):
    """`cat([r, r, r, r])[7:8] >= 5` — position 7 is the first copy's 7 at T=15 or 16, the second
    copy's 2 at T=5.

    Four copies are joined so that even at T's lower bound 2 the total length 4T >= 8, which makes
    `[7:8]` a static single element (a slice output of symbolic length would fail on the shape
    side before baking and would not be a counterexample).
    """

    def forward(self, x):
        positions = torch.arange(x.shape[0])
        return x + (torch.cat([positions] * 4)[7:8] >= 5).to(torch.float32)


class _PositionAcrossRepetition(nn.Module):
    """`r.repeat(4)[7:8] >= 5` — the values equal the 4-way cat (the second round's positions
    shift with T)."""

    def forward(self, x):
        positions = torch.arange(x.shape[0])
        return x + (positions.repeat(4)[7:8] >= 5).to(torch.float32)


def _assert_slips_past_two_point_evaluation(module: nn.Module, trailing: tuple[int, ...]) -> None:
    """The counterexample's premise: it agrees at the 2 evaluation points (Tmax and Tmax−1) and
    disagrees outside them at T=5.

    = a shape that the 2-point evaluation does not stop. If the premise broke, the tests using the
    counterexample would protect nothing.
    """
    assert torch.equal(module(torch.zeros(SYM_MAX, *trailing)), torch.ones(SYM_MAX, *trailing))
    assert torch.equal(
        module(torch.zeros(SYM_MAX - 1, *trailing)), torch.ones(SYM_MAX - 1, *trailing)
    )
    assert torch.equal(module(torch.zeros(5, *trailing)), torch.zeros(5, *trailing))


class TestSymbolicConstantFolding:
    """Symbol-dependent subtrees are baked at Tmax and cut out with sym_prefix_slice at run time
    (ADR 0010)."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_symbolic_reshape_is_applied_after_slicing(self, convert_module, dyn_t, axis):
        """A view whose row boundaries move with the length is not made constant, even when it
        happens to agree near the upper bound."""

        class ReshapedTable(nn.Module):
            def forward(self, x):
                length = x.shape[axis]
                table = (torch.arange(6 * length, dtype=torch.float32) < 25).float()
                return x + table.reshape(x.shape)

        shape = [2, 3]
        shape.insert(axis, 6)
        module = ReshapedTable()
        graph, tensors = convert_module(module, (torch.zeros(shape),), ({axis: dyn_t},))
        # Only the table before reshaping by the symbolic length is baked. It differs from the
        # prefix of the table reshaped at the upper bound.
        sliced = only_node(graph, "sym_prefix_slice")
        reshaped = only_node(graph, "reshape")
        assert reshaped.ins == sliced.outs
        assert only_node(graph, "add").ins[1] == reshaped.outs[0]
        baked = tensors[graph.initializers[sliced.ins[0]].tensor]
        for length in range(2, SYM_MAX + 1):
            actual_shape = [2, 3]
            actual_shape.insert(axis, length)
            x = torch.zeros(actual_shape)
            actual = x + baked[: 6 * length].reshape(actual_shape)
            assert torch.equal(actual, module(x)), f"axis={axis}, T={length}"

    def test_symbol_dependent_constant_is_baked_at_tmax_and_prefix_sliced(
        self, convert_module, dyn_t
    ):
        class PositionIndex(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(-1)

        graph, tensors = convert_module(PositionIndex(), (torch.randn(6, 4),), ({0: dyn_t},))

        sliced = only_node(graph, "sym_prefix_slice")
        assert sliced.attrs == {"sym": "T", "slices": [{"dim": 0, "coeff": 1, "offset": 0}]}
        # The baked constant has the length **Tmax = the upper bound of dyn_t**; at run time only
        # the first T elements are read.
        baked = tensors[graph.initializers[sliced.ins[0]].tensor]
        assert torch.equal(baked, torch.arange(SYM_MAX, dtype=torch.float32).unsqueeze(-1))
        assert graph.values[sliced.outs[0]].shape == ["T", 1]

    def test_prefix_slice_beyond_the_strided_rank_is_rejected(self, convert_module, dyn_t):
        """A prefix slice runs as a strided materializing copy (no new kernel — ADR 0010).

        Without checking the rank limit at export time, it would be "export green, fails only in
        the browser".
        """

        class Wide(nn.Module):
            def forward(self, x):
                length = x.shape[0]
                table = torch.arange(length, dtype=torch.float32)
                wide = table.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,1,T,1]
                return x.reshape(1, 1, 1, length, 1) + wide

        with pytest.raises(NotImplementedError, match="strided kernel"):
            convert_module(Wide(), (torch.randn(6),), ({0: dyn_t},))

    def test_a_static_subtree_stays_a_plain_constant(self, convert_module, dyn_t):
        """Without a symbolic dimension no sym_prefix_slice is emitted (no wasted dispatch is
        added)."""

        class StaticOffset(nn.Module):
            def forward(self, x):
                return x + torch.arange(4, dtype=torch.float32)

        graph, _ = convert_module(StaticOffset(), (torch.randn(6, 4),), ({0: dyn_t},))

        assert node_ops(graph) == ["add"]

    def test_a_subtree_that_uses_the_symbol_as_a_value_is_rejected(self, convert_module, dyn_t):
        """`arange(T) / T` qualifies by the allowlist alone — the consumption-position check stops
        it.

        NOTE: the expected diagnostic was changed from "does not commute under the 2-point
        evaluation". The divisor T is in a value position, not an extent, so div drops out of the
        folding, and the arity check fails it first as a form with a single tensor input (that it
        is rejected does not change).
        """

        class NormalizedPositions(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32) / x.shape[0]

        with pytest.raises(NotImplementedError, match=r"a form that uses SymInt .* as a value"):
            convert_module(NormalizedPositions(), (torch.randn(6),), ({0: dyn_t},))

    def test_a_constant_filled_with_the_symbol_is_rejected(self, convert_module, dyn_t):
        """`full((T,), T)` — both the shape and the values depend on T, and only the prefix's values
        disagree.

        NOTE: the expected diagnostic was changed from "does not commute under the 2-point
        evaluation". The T of `fill_value` is in a value position, so full drops out of the
        folding and goes to the full enumeration as an op outside the runtime's vocabulary.
        """

        class FilledWithLength(nn.Module):
            def forward(self, x):
                return x + torch.full((x.shape[0],), x.shape[0]).to(torch.float32)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(FilledWithLength(), (torch.randn(6),), ({0: dyn_t},))

        assert "aten.full.default" in err.value.ops

    def test_a_symbol_promoted_to_tensor_data_is_rejected(self, convert_module, dyn_t):
        """A symbol promoted to tensor data through `scalar_tensor(T)` is not folded (the hole in
        the 2-point evaluation).

        `(T−Tmax)(T−(Tmax−1))` is **0 at both** evaluation points, so the observation slips
        through — baking it would silently lose a factor of 182 at T=2. Since the rejection is by
        consumption position (value vs extent), scalar_tensor drops out of the folding and is
        rejected as an op outside the runtime's vocabulary.
        """

        class SymbolAsData(nn.Module):
            def forward(self, x):
                length = torch.scalar_tensor(x.shape[0], dtype=torch.float32)
                return x * ((length - SYM_MAX) * (length - (SYM_MAX - 1)))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(SymbolAsData(), (torch.randn(6),), ({0: dyn_t},))

        assert "aten.scalar_tensor.default" in err.value.ops

    @pytest.mark.parametrize(
        ("module", "shape", "error", "match"),
        [
            # `r[tensor([-1])]` is index.Tensor — indexing the symbolic axis with an index tensor.
            (_LastPositionByTensorIndex(), (6,), UnsupportedAtenOpsError, "aten.index.Tensor"),
            # Not the index's shape: indexing the symbolic axis with an index tensor is itself
            # excluded (being negative is a property of the values).
            (_LastPositionBySymbolicIndex(), (6,), UnsupportedAtenOpsError, "aten.index.Tensor"),
            # `r[-2:]` is a slice with a negative start — the excluded slice fails in the table,
            # which does not accept a symbolic axis.
            (_LastPositionsByNegativeSlice(), (6, 2), NotImplementedError, "symbolic dimension"),
            # cat along the symbolic axis — which input position 7 falls into is decided by T.
            # With cat excluded, the slice after it also fails in the table, which does not
            # accept a symbolic axis.
            (_PositionAcrossConcatenation(), (6,), NotImplementedError, "symbolic dimension"),
            # repeat over the symbolic axis — excluded for the same reason as cat; repeat fails as
            # outside the vocabulary.
            (_PositionAcrossRepetition(), (6,), UnsupportedAtenOpsError, "aten.repeat.default"),
        ],
        ids=["tensor-index", "symbolic-tensor-index", "negative-slice", "cat", "repeat"],
    )
    def test_a_position_that_depends_on_the_symbolic_length_is_not_folded(
        self, convert_module, dyn_t, module, shape, error, match
    ):
        """A form in which a position on the symbolic axis is decided by the length T is not folded
        (the hole in the 2-point evaluation — ADR 0010 addendum).

        Indexing `arange(T)` counted from the end, or joining / repeating along the symbolic axis,
        turns the length T into a value (T−1, or "which round it falls in"). The symbol does not
        reach a value position, so it passes the consumption-position check, and when the later
        comparison is true at both Tmax and Tmax−1 it passes the 2-point evaluation too — baking it
        would silently turn a value that should be 0 at T=5 into 1. The exclusion is structural,
        without looking at the index values, so the subtree drops to the normal lowering and fails
        explicitly.
        """
        _assert_slips_past_two_point_evaluation(module, shape[1:])

        with pytest.raises(error, match=match):
            convert_module(module, (torch.randn(*shape),), ({0: dyn_t},))

    def test_a_concatenation_along_the_symbolic_axis_runs_at_runtime(self, convert_module, dyn_t):
        """cat along the symbolic axis is not folded — the second input's positions shift by the
        first input's length T."""

        class Doubled(nn.Module):
            def forward(self, x):
                positions = torch.arange(x.shape[0], dtype=torch.float32)
                return torch.cat([x, x]) + torch.cat([positions, positions])

        graph, _ = convert_module(Doubled(), (torch.randn(6),), ({0: dyn_t},))

        assert node_ops(graph).count("cat") == 2

    def test_a_contraction_over_the_symbolic_axis_runs_at_runtime(self, convert_module, dyn_t):
        """A bmm that contracts over the symbolic axis is not folded — the result is a sum over T
        terms and depends on the total length.

        `sum(arange(T)) >= (Tmax−1)(Tmax−2)/2` is true at Tmax and Tmax−1 and false at T=5 —
        baked, it would pass the 2-point evaluation and silently become 1. The excluded bmm is
        lowered as a run-time op.
        """

        class SumOfPositions(nn.Module):
            def forward(self, x):
                positions = torch.arange(x.shape[0], dtype=torch.float32)
                ones = torch.ones(x.shape[0])
                total = torch.bmm(positions.view(1, 1, -1), ones.view(1, -1, 1))
                threshold = (SYM_MAX - 1) * (SYM_MAX - 2) // 2
                return x + (total.view(1) >= threshold).to(torch.float32)

        _assert_slips_past_two_point_evaluation(SumOfPositions(), ())

        graph, _ = convert_module(SumOfPositions(), (torch.randn(6),), ({0: dyn_t},))

        assert "bmm" in node_ops(graph)

    def test_a_boolean_mask_index_is_not_folded(self, convert_module, dyn_t):
        """An index with a bool mask is not folded, with or without a symbolic axis (the
        conservative side).

        A rank-2 mask occupies 2 axes of the input with a single index, so the symbolic axis (the
        second axis) hides outside the correspondence "index k ↔ input axis k".
        Even without this exclusion, the output length of mask indexing is an unbacked symbol
        decided by values, and it failed explicitly on the baked result's shape (not a silently
        wrong value). What this test pins is how it fails — the excluded index goes into the full
        enumeration of unsupported ops as an op outside the vocabulary.
        """

        class Masked(nn.Module):
            def forward(self, x):
                grid = torch.arange(x.shape[0]).view(1, -1) + torch.zeros(2, 1, dtype=torch.long)
                return torch.cat([x, grid[grid >= 0].to(torch.float32)])

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Masked(), (torch.randn(6),), ({0: dyn_t},))

        assert "aten.index.Tensor" in err.value.ops

    def test_a_non_negative_slice_start_on_the_symbolic_axis_still_folds(
        self, convert_module, dyn_t
    ):
        """A slice counted from the start commutes with the prefix — only a negative start is
        excluded."""

        class Shifted(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0] + 1, dtype=torch.float32)[1:]

        graph, tensors = convert_module(Shifted(), (torch.randn(6),), ({0: dyn_t},))

        sliced = only_node(graph, "sym_prefix_slice")
        baked = tensors[graph.initializers[sliced.ins[0]].tensor]
        assert torch.equal(baked, torch.arange(1, SYM_MAX + 1, dtype=torch.float32))

    def test_a_symbol_free_fill_value_still_folds_under_a_symbolic_extent(
        self, convert_module, dyn_t
    ):
        """The rejection is by consumption **position** — even when the shape depends on T, values
        independent of T are folded as before."""

        class ConstantRow(nn.Module):
            def forward(self, x):
                return x + torch.full((x.shape[0],), 3.0)

        graph, tensors = convert_module(ConstantRow(), (torch.randn(6),), ({0: dyn_t},))

        sliced = only_node(graph, "sym_prefix_slice")
        baked = tensors[graph.initializers[sliced.ins[0]].tensor]
        assert torch.equal(baked, torch.full((SYM_MAX,), 3.0))

    def test_an_unbounded_symbol_is_rejected(self, convert_module):
        """Without Tmax the length to bake cannot be decided (no proceeding with a default)."""

        class PositionIndex(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32)

        with pytest.raises(NotImplementedError, match="is not finite"):
            convert_module(PositionIndex(), (torch.randn(6),), ({0: Dim.DYNAMIC},))

    def test_a_symbol_range_too_narrow_for_two_points_is_rejected(self, convert_module):
        """A range in which the second evaluation point falls below the line that avoids 0/1
        specialization is not accepted, since the check would become tautological."""

        class PositionIndex(nn.Module):
            def forward(self, x):
                return x + torch.arange(x.shape[0], dtype=torch.float32)

        with pytest.raises(ValueError, match="does not allow folding's 2-point evaluation"):
            convert_module(PositionIndex(), (torch.randn(2),), ({0: Dim("T", min=1, max=2)},))


class TestFoldProbeComparison:
    """The comparison of the 2-point evaluation is bit-exact (numerical equality lets a sign-bit
    difference through).

    NOTE: the comparison function is called directly. After symbol-as-data was rejected, producing
    this difference through export would need "a foldable subtree whose values depend on T" —
    exactly the shape that slips through silently, which a test cannot synthesize (if it could,
    that would be an unfixed bug).
    """

    def test_negative_zero_is_not_equal_to_positive_zero(self):
        minus = torch.tensor([-0.0])
        plus = torch.tensor([0.0])

        # Numerical equality lets it through — the very reason it was tightened.
        assert torch.equal(minus, plus)
        assert not _bitwise_equal(minus, plus)

    def test_identical_bits_are_equal(self):
        assert _bitwise_equal(torch.tensor([[1.5, -0.0]]), torch.tensor([[1.5, -0.0]]))

    def test_a_shape_or_dtype_difference_is_not_equal(self):
        assert not _bitwise_equal(torch.tensor([1.0]), torch.tensor([[1.0]]))
        assert not _bitwise_equal(torch.tensor([1.0]), torch.tensor([1], dtype=torch.int32))


class TestCommonSubexpression:
    def test_identical_nodes_collapse_to_one(self, convert_module):
        """export keeps identical subexpressions — the IR side folds them into one."""

        class Twice(nn.Module):
            def forward(self, x, y):
                return (x + y) + (x + y)

        graph, _ = convert_module(Twice(), (torch.randn(3, 4), torch.randn(3, 4)))

        assert node_ops(graph) == ["add", "add"]
        outer = graph.nodes[1]
        assert outer.ins == [graph.nodes[0].outs[0]] * 2


class TestConstantDedup:
    """Constant deduplication matches on the full SHA-256 (the shortened name is for display
    only)."""

    def _converter(self) -> Converter:
        ep = torch.export.export(Scale(), (torch.randn(3, 4),), strict=False)
        return Converter(ep)

    def test_the_dedup_key_is_the_full_digest_not_the_shortened_name(self):
        """Matching on a 64-bit prefix alone would fold 2 colliding constants without comparing
        their contents."""
        converter = self._converter()

        name = converter._add_const(torch.zeros(3), "test constant")

        (digest,) = converter._const_by_digest
        assert len(digest) == 64
        # The name and the tensor key are spelled with the first 16 hex digits, as before.
        assert name == f"const_{digest[:16]}"
        assert converter.graph.initializers[name].tensor == f"const.{digest[:16]}"

    def test_a_short_name_collision_between_distinct_constants_fails_loudly(self):
        """Builds and exercises a case where the full digests differ but the short names match (a
        real collision is unreachable).

        Letting it through would make the later constant overwrite the earlier declaration, and
        nodes that referenced the earlier constant would read another value (a silently wrong
        graph).
        """
        converter = self._converter()
        converter._add_const(torch.zeros(3), "test constant")
        # Drop only the match keys to create the situation "another constant landed on the same
        # short name".
        converter._const_by_digest.clear()

        with pytest.raises(AssertionError, match="collided"):
            converter._add_const(torch.zeros(3), "test constant")


class TestDeclarations:
    """Each value is declared in exactly one place (inputs in inputs[], everything else in
    values{})."""

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
    """Parameters, buffers and lifted constants are all carried by their FQN tensor keys."""

    def test_parameters_buffers_and_lifted_constants_all_become_initializers(self, convert_module):
        class WithState(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4))
                self.register_buffer("shift", torch.randn(4))
                self.gain = torch.randn(4)  # plain attribute → lifted tensor constant

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

        with pytest.raises(NotImplementedError, match="cannot be an IR v1 initializer"):
            convert_module(BoolBuffer(), (torch.randn(3, 4),))


class TestDtypes:
    def test_dtype_outside_the_semantic_vocabulary_is_rejected(self, convert_module):
        with pytest.raises(
            NotImplementedError, match="is not in IR v1's semantic dtype vocabulary"
        ):
            convert_module(Scale(), (torch.randn(3, 4, dtype=torch.float64),))


class TestSymbols:
    """The names that go into the IR are given by the caller and assigned in the order the user
    inputs appear."""

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

        assert "not enough IR symbol names" in str(err.value)
        assert "symbol_names" in str(err.value)

    def test_duplicate_symbol_names_are_rejected(self, convert_module):
        with pytest.raises(ValueError, match="symbol_names has duplicates"):
            convert_module(Scale(), (torch.randn(3, 4),), symbol_names=("T", "T"))

    @pytest.mark.parametrize("name", ["2T", "T+1", "", "T-1"])
    def test_symbol_name_outside_the_dim_grammar_is_rejected(self, convert_module, name):
        """A name outside the grammar of the dimension notation would give a dimension that can be
        written but that the runtime cannot read."""
        with pytest.raises(ValueError, match="is not in the grammar of the dimension notation"):
            convert_module(Scale(), (torch.randn(3, 4),), symbol_names=(name,))


class TestOutputs:
    """The output list is built from the user outputs of graph_signature only."""

    def test_multiple_user_outputs_keep_their_order(self, convert_module):
        class TwoOutputs(nn.Module):
            def forward(self, x):
                return torch.relu(x), torch.tanh(x)

        graph, _ = convert_module(TwoOutputs(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["relu", "tanh"]
        assert graph.outputs == [node.outs[0] for node in graph.nodes]

    def test_buffer_mutation_is_rejected(self, convert_module):
        """IR v1 cannot express a state update — it fails as unsupported instead of being silently
        dropped."""

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
    """Assigning the same IR value to 2 outputs fails already at `convert()`.

    The output names are a set (docs/ir-v2.md), and the accepting side `verify.parse_ir_graph`
    rejects duplicates. Delaying the detection to the accepting side would let a direct call of
    `emit.write_model`, which runs no verification, place a distribution that "was written but
    cannot be read".
    """

    def test_returning_the_same_value_twice_fails_loudly(self, convert_module):
        """The same Node returned twice (`_user_outputs` returns the output arguments by position as
        is)."""

        class SameTwice(nn.Module):
            def forward(self, x):
                y = x + 1
                return y, y

        with pytest.raises(NotImplementedError) as err:
            convert_module(SameTwice(), (torch.randn(3, 4),))

        assert "outputs" in str(err.value)
        assert "duplicates" in str(err.value)

    def test_two_structurally_identical_outputs_fail_loudly(self, convert_module):
        """The path CSE folds — separate subtrees become one when `(op, ins, attrs, shape)`
        match."""

        class SameShape(nn.Module):
            def forward(self, x):
                return x + 1, x + 1

        with pytest.raises(NotImplementedError) as err:
            convert_module(SameShape(), (torch.randn(3, 4),))

        assert "duplicates" in str(err.value)

    def test_distinct_outputs_still_convert(self, convert_module):
        """The other side of the regression — with different values, the 2 outputs pass as they are
        (a detector against a tautological check)."""

        class TwoValues(nn.Module):
            def forward(self, x):
                return x + 1, x + 2

        graph, _ = convert_module(TwoValues(), (torch.randn(3, 4),))

        assert len(set(graph.outputs)) == len(graph.outputs) == 2


class TestOutputSpecAgreement:
    """The correspondence between `output_specs` and the output arguments holds **by position**
    only (normalize does not update `graph_signature`, so `output_specs[].arg.name` can name
    something that does not exist).

    Passing mismatched counts to `zip(strict=True)` would silently shift the outputs, so it fails
    before that.
    """

    def test_a_signature_with_fewer_output_specs_fails_loudly(self):
        class TwoOutputs(nn.Module):
            def forward(self, x):
                return torch.relu(x), torch.tanh(x)

        exported = torch.export.export(TwoOutputs(), (torch.randn(3, 4),), strict=False)
        decomposed = exported.run_decompositions(curated_decompositions())
        normalize_graph(decomposed)
        # A fake with one user output removed (only the specs shrink; the output arguments stay
        # 2).
        decomposed.graph_signature.output_specs = list(decomposed.graph_signature.output_specs)[:-1]

        with pytest.raises(AssertionError, match="differ in count"):
            convert(decomposed)


class TestDtypeBoundary:
    """The boundary that drops torch's default i64 to the IR's i32 (ADR 0009)."""

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

        # bool becomes u32 0/1 (the same convention as GPU storage)
        flags = normalize_boundary_tensor(torch.tensor([True, False]), "t")
        assert flags.dtype is torch.uint32
        assert flags.tolist() == [1, 0]

        floats = torch.tensor([1.5, -2.5])
        assert normalize_boundary_tensor(floats, "t").dtype is torch.float32

    def test_out_of_range_int64_fails_loudly(self):
        """Silent truncation would shift indices silently — out of range always fails."""
        too_big = torch.tensor([2**31], dtype=torch.int64)

        with pytest.raises(ValueError, match="does not fit in i32"):
            normalize_boundary_tensor(too_big, "t")

        with pytest.raises(ValueError, match="does not fit in i32"):
            normalize_boundary_tensor(torch.tensor([-(2**31) - 1], dtype=torch.int64), "t")

        # The exact boundary values pass
        assert normalize_boundary_tensor(
            torch.tensor([2**31 - 1, -(2**31)], dtype=torch.int64), "t"
        ).tolist() == [2**31 - 1, -(2**31)]

    def test_unsupported_dtype_fails_loudly(self):
        with pytest.raises(NotImplementedError, match="cannot be dropped to a semantic dtype"):
            normalize_boundary_tensor(torch.zeros(2, dtype=torch.float64), "t")


class TestCloneAlias:
    """aten.clone is the identity (IR values are always contiguous — ADR 0011). It is not emitted
    into the IR but becomes an alias."""

    def test_contiguous_clone_after_permute_is_absorbed(self, convert_module, dyn_t):
        class Materialize(nn.Module):
            def forward(self, x):
                return x.permute(1, 0, 2).contiguous()

        graph, _ = convert_module(Materialize(), (torch.randn(6, 3, 4),), ({0: dyn_t},))

        # No clone node remains; the permute output becomes the graph output as is
        assert node_ops(graph) == ["permute"]
        assert graph.outputs == [graph.nodes[0].outs[0]]

    def test_non_contiguous_memory_format_is_rejected(self, convert_module):
        class ChannelsLast(nn.Module):
            def forward(self, x):
                return torch.clone(x, memory_format=torch.channels_last)

        with pytest.raises(NotImplementedError, match="memory_format"):
            convert_module(ChannelsLast(), (torch.randn(2, 3, 4, 5),))


class SymbolicCatJoin(nn.Module):
    """cat on the symbolic axis (self | context) + **an input with the concatenated length** (a
    scaled-down version of the DiT shape from the recon).

    The point is that `gate` is declared with the derived dimension `T+3`; this shape appears only
    when `Dim("T") + 3` is written in dynamic_shapes (it first showed up in the observations of
    ADR 0046).
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 2, bias=False)

    def forward(self, x, context, gate):
        return self.proj(torch.cat([x, context], 1) * gate)


class TestSymbolicCatEndToEnd:
    """A graph with cat on the symbolic axis passes export → write → verify_container (ADR 0046)."""

    def test_a_derived_input_dimension_survives_the_range_constraint_scan(self, tmp_path, dyn_t):
        """MUST: derived dimensions also appear in range_constraints as **a sympy `Add`**.

        Declaring `Dim("T") + 3` in an input shape puts `s27 + 3` into the keys of
        `ep.range_constraints`. A scan that reads `.name` bare fails with `AttributeError: 'Add'
        object has no attribute 'name'`, and a graph with cat on the symbolic axis dies **before
        conversion starts** (fixed together with ADR 0046). The range of a derived dimension is
        redundant information decided by the bare symbol's range, so nothing is lost by leaving it
        out.
        """
        graph = export_to_file(
            SymbolicCatJoin(),
            (torch.randn(1, 6, 4), torch.randn(1, 3, 4), torch.randn(1, 9, 1)),
            tmp_path / "model.krm",
            provenance=Provenance(license="mit"),
            graph_name="cat",
            dynamic_shapes=({1: dyn_t}, None, {1: dyn_t + 3}),
        )

        # The derived dimension stays in the input declaration as is (the binding source is only
        # the position of the bare `T`)
        assert [spec.shape for spec in graph.inputs] == [[1, "T", 4], [1, 3, 4], [1, "T+3", 1]]
        assert graph.values[only_node(graph, "cat").outs[0]].shape == [1, "T+3", 4]
