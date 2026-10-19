"""Pins the behaviour of the aten op table (aten_handlers.py).

Pins that individual aten ops and fused ops lower to the intended IR vocabulary, that attrs make
torch semantics explicit at the boundary, and that shapes outside the contract fail loudly.
The contract surface of the engine (convert.py) is in test_convert.py.
"""

from __future__ import annotations

import pytest
import torch
from conftest import SYM_MAX, node_ops, only_node
from torch import nn
from torch.export import Dim
from torchvision.ops import deform_conv2d

from karume.aten_handlers import ATEN_HANDLERS
from karume.convert import UnsupportedAtenOpsError, convert, curated_decompositions
from karume.normalize import normalize_graph
from karume.ops import EMITTABLE_OPS


def _gelu_tanh_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation of `gelu_tanh` (mirror of the TS
    packages/runtime/src/reference/ops.ts).

    0.7978845608028654 = √(2/π). Unlike the erf form, this formula is the definition itself, so
    there is no room to rewrite it differently from torch (the comparison looks for **a mixed-up
    formula**, not an approximation error).
    """
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class TestGelu:
    """gelu stops decomposition and travels as one node (the core decomposition scatters outside
    the M0 vocabulary).

    The approximation kind is told apart by **op name**, not attrs (the contract has empty attrs —
    no branch outside the contract where only the numbers change under the same op name).
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
        # ±0 / ±large values (including the range where x³ overflows to inf in f32) / ±small
        # values (the range where x³ underflows to 0).
        edges = torch.tensor(
            [0.0, -0.0, 3.4e38, -3.4e38, 1e20, -1e20, 1e-20, -1e-20, 20.0, -20.0],
            dtype=torch.float32,
        )
        for name, x in (("randn", random), ("edges", edges)):
            expected = nn.functional.gelu(x, approximate="tanh")
            torch.testing.assert_close(_gelu_tanh_reference(x), expected, msg=name)

    def test_two_gelus_are_numerically_distinguishable(self):
        # Why they cannot fold into one op (folding would silently add errors of order 1e-3).
        x = torch.linspace(-3.0, 3.0, 101)
        gap = (nn.functional.gelu(x) - _gelu_tanh_reference(x)).abs().max()
        assert gap > 1e-4


class TestSin:
    """`sin` has both paths, folding and emission, **at once** (layer 1 of ADR 0043 — now the Core
    ATen layer, ADR 0059).

    Constant subtrees (RoPE position tables) are folded into initializers by FOLDABLE_OPS as
    before, and only forms taking runtime values stay as IR nodes. Removing either silently
    degrades the other — without constant folding the runtime dispatches grow, and without
    emission the Snake activation fails at export.
    """

    def test_runtime_valued_sin_becomes_a_node(self, convert_module):
        class Sin(nn.Module):
            def forward(self, x):
                return torch.sin(x)

        graph, _ = convert_module(Sin(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["sin"]

    def test_constant_sin_is_still_folded_into_an_initializer(self, convert_module):
        """`sin` on a constant input (the RoPE table form) is folded away, leaving no node."""

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
        """DACVAE's Snake `x + sin²(αx)/(α+1e-9)` (the very reason this op was added).

        α is a learned parameter = an initializer, so `α+1e-9` is also a runtime node (folding
        through a parameter is not eligible — `_classify_foldable`). `sin` takes the runtime
        value `αx`, so it stays as one node, and `pow(2)` decomposes into `mul`.
        NOTE: writing the coefficient as `(α+1e-9).reciprocal()` hits `aten.reciprocal.default`
        (FOLDABLE only, with no emit path). It is written in the equivalent division form.
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
    """reduce takes one axis without keepdim only. The axis goes into attrs `dim` **as
    non-negative**."""

    def test_last_dim_reductions_convert(self, convert_module):
        class Reduce(nn.Module):
            def forward(self, x):
                return x.sum(dim=-1), x.amax(dim=-1), x.amin(dim=-1)

        graph, _ = convert_module(Reduce(), (torch.randn(3, 5),))

        assert node_ops(graph) == ["sum", "amax", "amin"]
        assert [graph.values[name].shape for name in graph.outputs] == [[3], [3], [3]]
        # Negative axis notation is normalized at the boundary (IR attrs are non-negative only)
        assert [node.attrs["dim"] for node in graph.nodes] == [1, 1, 1]

    def test_a_non_last_axis_is_carried_in_attrs(self, convert_module):
        """A reduction over the channel axis (the VAE's L2 normalization) is written as is, without
        a permute."""

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
            (lambda x: x.sum(dim=(0, 1)), r"are not supported for sum \(one axis at a time\)"),
            (lambda x: x.sum(dim=-1, keepdim=True), "sum with keepdim=True is not supported"),
            (lambda x: x.amax(dim=(0, 1)), r"are not supported for amax \(one axis at a time\)"),
            (lambda x: x.amin(dim=-1, keepdim=True), "amin with keepdim=True is not supported"),
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
    """argmax takes only the fixed form **last dimension + keepdim=True** (ADR 0068 decision 2).

    Unlike the reduce family it has no attrs (the axis is fixed to the last dimension, the rank is
    kept), so all narrowing is done by the handler's acceptance check. The output is torch's
    i64 → normalized to **i32** at the boundary (ADR 0009).
    """

    def test_last_dim_with_keepdim_becomes_one_node(self, convert_module):
        class Argmax(nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=-1, keepdim=True)

        graph, _ = convert_module(Argmax(), (torch.randn(3, 5),))

        assert node_ops(graph) == ["argmax"]
        # Empty attrs (the absence of `dim` / `keepdim` fields itself declares "no other form is in
        # the vocabulary")
        assert only_node(graph, "argmax").attrs == {}
        # Rank kept = the last dimension collapsed to 1; the dtype is i32
        assert [graph.values[name].shape for name in graph.outputs] == [[3, 1]]
        assert [graph.values[name].dtype for name in graph.outputs] == ["i32"]

    def test_a_symbolic_leading_dimension_survives(self, convert_module, dyn_t):
        """The decode exit shape (the last dimension of `[B, T, V]` = the vocabulary axis). Leading
        dimensions stay symbolic."""

        class Argmax(nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=2, keepdim=True)

        graph, _ = convert_module(Argmax(), (torch.randn(1, 4, 7),), dynamic_shapes=({1: dyn_t},))

        assert node_ops(graph) == ["argmax"]
        assert [graph.values[name].shape for name in graph.outputs] == [[1, "T", 1]]

    @pytest.mark.parametrize(
        ("build", "why"),
        [
            (lambda x: torch.argmax(x, dim=-1), "argmax with keepdim=False is not supported"),
            (lambda x: torch.argmax(x), "argmax over all dimensions is not supported"),
            (
                lambda x: torch.argmax(x, dim=0, keepdim=True),
                r"other than the last \(dim=0 / rank=2\)",
            ),
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
        """`dim` / `keepdim` arriving as kwargs are read as the same fields (the same rule as the
        reduce family).

        Plain indexing of positional arguments would raise `IndexError` and fall outside ADR
        0005's rule "unsupported input fails with a diagnosis" (the diagnosis would turn into
        "the exporter is broken").
        """
        target = torch.ops.aten.argmax.default
        node = _reduce_node(target, kwargs={"dim": -1, "keepdim": True})

        assert ATEN_HANDLERS[target](node) == ("argmax", 1, {}, ())

        with pytest.raises(NotImplementedError, match="argmax with keepdim=False is not supported"):
            ATEN_HANDLERS[target](_reduce_node(target, kwargs={"dim": -1}))
        with pytest.raises(
            NotImplementedError, match="argmax over all dimensions is not supported"
        ):
            ATEN_HANDLERS[target](_reduce_node(target, kwargs={"keepdim": True}))


def _reduce_node(target, args=(), kwargs=None, shape=(3, 5)):
    """Build one FX node to feed straight into a reduce handler.

    torch 2.13's decomposition always emits the reduce `dim` as a positional argument, so **the
    kwargs form cannot be produced by export** (it can arrive depending on how the overload is
    called). Handlers are pure functions, so instead of building a graph, one node is built by
    hand and the handler is called directly.
    """
    graph = torch.fx.Graph()
    src = graph.placeholder("x")
    src.meta["val"] = torch.zeros(shape)
    return graph.call_function(target, (src, *args), dict(kwargs or {}))


class TestReduceArgumentsInKwargs:
    """The reduce family reads `dim` / `keepdim` / `dtype` arriving as kwargs as the same fields.

    Plain indexing of positional arguments would raise `IndexError` and fall outside ADR 0005's
    rule "unsupported input fails with a diagnosis" (the diagnosis would turn into "the exporter
    is broken").
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
            (
                torch.ops.aten.sum.dim_IntList,
                {"dim": None},
                "sum over all dimensions is not supported",
            ),
            (
                torch.ops.aten.sum.dim_IntList,
                {"dim": [0, 1]},
                r"multiple axes \[0, 1\] are not supported for sum",
            ),
            (
                torch.ops.aten.sum.dim_IntList,
                {"dim": [-1], "keepdim": True},
                "sum with keepdim=True",
            ),
            (
                torch.ops.aten.sum.dim_IntList,
                {"dim": [-1], "dtype": torch.float32},
                "sum with a dtype is not supported",
            ),
            (torch.ops.aten.amax.default, {}, "amax over all dimensions is not supported"),
            (
                torch.ops.aten.amax.default,
                {"dim": [0, 1]},
                r"multiple axes \[0, 1\] are not supported for amax",
            ),
            (torch.ops.aten.amin.default, {"dim": [-1], "keepdim": True}, "amin with keepdim=True"),
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
    """aten._to_copy → cast / aten.bitwise_not → bitwise_not (ADR 0009)."""

    def test_to_copy_becomes_cast_with_the_target_dtype(self, convert_module):
        class Truncate(nn.Module):
            def forward(self, x):
                return x.to(torch.int64)

        graph, _ = convert_module(Truncate(), (torch.randn(3, 4),))

        node = only_node(graph, "cast")
        # i64 is declared as i32 by the boundary normalization (the torch dtype is not used as the
        # name as is)
        assert node.attrs == {"to": "i32"}
        assert graph.values[node.outs[0]].dtype == "i32"

    def test_casts_with_different_targets_are_not_merged_by_cse(self, convert_module):
        """Without attrs in the CSE key, one of two same-input nodes silently disappears."""

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
        """Integer ~x means something other than bool negation — not reinterpreted silently."""

        class InvertInts(nn.Module):
            def forward(self, x):
                return torch.bitwise_not(x)

        with pytest.raises(NotImplementedError, match=r"bitwise_not of dtype .* is not supported"):
            convert_module(InvertInts(), (torch.ones(3, 4, dtype=torch.int64),))


class TestBmmAndGather:
    """bmm (rank-3 only) and gather (last dimension fixed) — ADR 0012."""

    def test_bmm_keeps_the_batch_axis(self, convert_module, dyn_t):
        class Batched(nn.Module):
            def forward(self, x, y):
                return torch.bmm(x, y)

        # B / M / K / N all get different lengths (an axis mix-up shows in the declared shapes)
        graph, _ = convert_module(
            Batched(),
            (torch.randn(2, 6, 3), torch.randn(2, 3, 5)),
            ({1: dyn_t}, None),
        )

        node = only_node(graph, "bmm")
        assert node.attrs == {}
        assert graph.values[node.outs[0]].shape == [2, "T", 5]

    def test_rank2_matmul_stays_matmul(self, convert_module):
        """rank-2 belongs to matmul (not shared with bmm — the contract table)."""

        class Flat(nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        graph, _ = convert_module(Flat(), (torch.randn(6, 3), torch.randn(3, 5)))

        assert node_ops(graph) == ["matmul"]

    def test_rank4_matmul_is_rejected(self, convert_module):
        """A rank-4 matmul either lowers to aten.bmm with reshapes or fails as outside the
        vocabulary.

        Either way it never "turns into a bmm while still rank-4" (only rank-3 is accepted).
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
        # src is f32, index is i32 (boundary normalization from i64) — the dtype differs per slot
        assert [graph.inputs[0].dtype, graph.inputs[1].dtype] == ["f32", "i32"]
        assert graph.values[node.outs[0]].dtype == "f32"

    def test_gather_on_a_non_last_dim_is_rejected(self, convert_module):
        class ByRow(nn.Module):
            def forward(self, src, index):
                return torch.gather(src, 0, index)

        with pytest.raises(NotImplementedError, match="other than the last"):
            convert_module(
                ByRow(),
                (torch.randn(3, 3), torch.tensor([[0, 1, 2], [2, 1, 0], [1, 1, 1]])),
            )

    def test_gather_with_a_shorter_leading_dim_is_rejected(self, convert_module):
        """torch allows `index.size(d) <= src.size(d)`, but the IR contract requires an exact match
        (packages/runtime/src/ops.ts).

        Passing on rank alone would keep export green and fail the contract check in the browser
        (a shape where the `row = i / J` row correspondence does not hold).
        """

        class Narrow(nn.Module):
            def forward(self, src, index):
                return torch.gather(src, -1, index)

        with pytest.raises(NotImplementedError, match="leading dimension"):
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
    """Math ops of M1-P3 phase 3 (sdp's spline / dec's leaky_relu — recon §2).

    Three things are pinned: ① observed shapes lower to the intended single node; ② attrs make
    torch semantics **explicit at the boundary**; ③ shapes outside the contract (one-sided clamp,
    integer logical and, cumsum on a non-last dimension, scalar operands) fail loudly.
    """

    def test_the_inside_test_lowers_to_compare_and_bitwise_and(self, convert_module):
        """The spline's `inside = (x >= -b) & (x <= b)` (recon §2)."""

        class Inside(nn.Module):
            def forward(self, x):
                return (x >= -1.0) & (x <= 1.0)

        graph, _ = convert_module(Inside(), (torch.randn(4),))

        assert node_ops(graph) == ["ge_scalar", "le_scalar", "bitwise_and"]
        assert only_node(graph, "ge_scalar").attrs == {"value": -1.0}
        assert only_node(graph, "le_scalar").attrs == {"value": 1.0}
        assert graph.values[only_node(graph, "bitwise_and").outs[0]].dtype == "bool"

    def test_searchsorted_lowers_to_ge_tensor_and_a_bool_sum(self, convert_module):
        """`sum(x[…,None] >= bl, dim=-1)` (recon §2) — a row sum of bool is an i32 count."""

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
        """The contract is f32 only (no observed integer comparison — not silently read as f32)."""

        class IntGate(nn.Module):
            def forward(self, x):
                return x >= 1

        with pytest.raises(NotImplementedError, match="is not supported for ge_scalar"):
            convert_module(IntGate(), (torch.ones(4, dtype=torch.int64),))

    def test_clamp_carries_both_bounds(self, convert_module):
        class Clamped(nn.Module):
            def forward(self, x):
                return torch.clamp(x, -0.75, 0.5)

        graph, _ = convert_module(Clamped(), (torch.randn(4),))

        assert only_node(graph, "clamp").attrs == {"min": -0.75, "max": 0.5}

    def test_a_lower_bound_only_clamp_becomes_clamp_min(self, convert_module):
        """A lower-bound-only clamp lowers to **a separate op** (clamp_min) (ADR 0017).

        Filling the missing upper bound with the largest finite value would "silently run a
        shape that has no representation as another shape", so it is not done — adding a
        one-sided op to the vocabulary was ADR 0017's decision.
        """

        class LowerBound(nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-1.0)

        graph, _ = convert_module(LowerBound(), (torch.randn(4),))

        assert node_ops(graph) == ["clamp_min"]
        assert only_node(graph, "clamp_min").attrs == {"min": -1.0}

    def test_an_upper_bound_only_clamp_is_rejected(self, convert_module):
        """An upper-bound-only form is not in the vocabulary (clamp_max is not implemented — added
        the same way when needed)."""

        class UpperBound(nn.Module):
            def forward(self, x):
                return torch.clamp(x, max=1.0)

        with pytest.raises(
            NotImplementedError, match="clamp with only an upper bound is not supported"
        ):
            convert_module(UpperBound(), (torch.randn(4),))

    def test_inverted_clamp_bounds_are_rejected(self, convert_module):
        """min > max is the direction where WGSL clamp is undefined (the same check as the contract
        table)."""

        class Inverted(nn.Module):
            def forward(self, x):
                return torch.clamp(x, 1.0, -1.0)

        with pytest.raises(NotImplementedError, match=r"clamp with .* is not supported"):
            convert_module(Inverted(), (torch.randn(4),))

    def test_log1p_stays_a_single_node(self, convert_module):
        """`log1p(exp(scaled))` of the decomposed softplus (recon §2)."""

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
        """The contract is arity 3 with empty attrs and has no field for a scalar, but torch's
        decomposition inserts `scalar_tensor`, so it actually arrives as a rank-0 constant
        (folded into an initializer). It passes without growing the IR vocabulary, so it is
        accepted as is.
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
        """Integer & means something other than bool logical and (narrowed like bitwise_not)."""

        class IntAnd(nn.Module):
            def forward(self, x, y):
                return x & y

        with pytest.raises(NotImplementedError, match=r"bitwise_and of dtype .* is not supported"):
            convert_module(
                IntAnd(),
                (torch.ones(4, dtype=torch.int64), torch.ones(4, dtype=torch.int64)),
            )

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            # ups / ResBlock (LRELU_SLOPE=0.1) and the last stage (torch default, positional
            # argument omitted)
            pytest.param(lambda x: nn.functional.leaky_relu(x, 0.1), 0.1, id="explicit 0.1"),
            pytest.param(lambda x: nn.functional.leaky_relu(x), 0.01, id="default 0.01"),
        ],
    )
    def test_leaky_relu_materializes_the_slope(self, convert_module, build, expected):
        """MUST: make torch's default **explicit in attrs at the boundary** (ADR 0015 — both forms
        are observed)."""

        class Leaky(nn.Module):
            def forward(self, x):
                return build(x)

        graph, _ = convert_module(Leaky(), (torch.randn(4),))

        assert node_ops(graph) == ["leaky_relu"]
        assert only_node(graph, "leaky_relu").attrs == {"negative_slope": expected}

    def test_cumsum_normalizes_the_negative_axis(self, convert_module, dyn_t):
        """The spline's cumwidths (`torch.cumsum(widths, dim=-1)`)."""

        class Cumulative(nn.Module):
            def forward(self, x):
                return torch.cumsum(x, dim=-1)

        graph, _ = convert_module(Cumulative(), (torch.rand(4, 3),), ({0: dyn_t},))

        assert only_node(graph, "cumsum").attrs == {"dim": 1}

    def test_cumsum_on_a_non_final_axis_is_rejected(self, convert_module):
        class Wrong(nn.Module):
            def forward(self, x):
                return torch.cumsum(x, dim=0)

        with pytest.raises(NotImplementedError, match=r"cumsum on .* is not supported"):
            convert_module(Wrong(), (torch.rand(4, 3),))


class TestFusedOps:
    """Fused ops (ADR 0012) — the 6 of the 9 preserved ops that appear in observed models.

    Two things are pinned here: ① preservation works and **the high-level op stays one node**;
    ② attrs outside their range and shapes not observed fail loudly.
    """

    def test_linear_stays_a_single_node_with_bias(self, convert_module, dyn_t):
        """Without preservation it scatters into addmm + view (recon §5)."""

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
        # The weight travels in its transposed [out, in] layout (no transpose node in between)
        assert graph.values[node.ins[1]].shape == [3, 4]
        assert graph.values[node.ins[2]].shape == [3]
        assert graph.values[node.outs[0]].shape == ["T", 3]

    def test_a_bias_free_linear_gets_a_synthesized_zero_bias(self, convert_module):
        """A bias-free linear is normalized to arity 3 by **synthesizing a zero bias** (ADR 0016).

        The same approach as the conv synthesis (ADR 0015): no arity branch is brought into the
        contract, the kernels or the shape layer. The synthesis is an identity because `+0` is
        an exact identity in f32.
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
        """Without preservation it becomes native_layer_norm (3 outputs) + getitem, which conflicts
        with the single-output premise."""

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
        """An affine-free layer_norm gets arity 3 by synthesizing ones/zeros (ADR 0016).

        Swapping the synthesis order (weight → bias) would give `×0 +1` and break the values, so
        the contents are checked (checking lengths alone would let a ones/zeros mix-up through).
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
        """A weight-only layer_norm gets arity 3 by synthesizing zeros (the `norm_bias=false` form).

        Also checks that the weight **stays in the real slot** (if the synthesis slipped into the
        weight side, it would become a silent wrong value, "discard the learned weight and
        normalize with ones", and the lengths match, so the shape check would not catch it).
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
        """A bias-only form fails (the synthesis appends at the end, so the bias would slip in).

        Letting `weight=None, bias=present` through puts the bias into the weight slot — the
        lengths are equal, so it becomes a silent wrong value that passes the shape check too.
        """

        class BiasOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(4))

            def forward(self, x):
                return nn.functional.layer_norm(x, [4], None, self.bias)

        with pytest.raises(NotImplementedError, match="only a bias"):
            convert_module(BiasOnly(), (torch.randn(5, 4),))

    def test_multi_axis_layer_norm_is_rejected(self, convert_module):
        """Only [H] is observed (the row kernel assumes a contiguous last dimension)."""

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
        # Negative axis notation is normalized to non-negative at the exporter boundary (IR attrs
        # are non-negative only)
        assert node.attrs == {"dim": 1}

    def test_softmax_on_a_non_last_dim_is_rejected(self, convert_module):
        class ByColumn(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=0)

        with pytest.raises(NotImplementedError, match="other than the last"):
            convert_module(ByColumn(), (torch.randn(5, 4),))

    def test_a_runtime_bool_mask_emits_safe_softmax(self, convert_module):
        """ADR 0044's chain: bool input → `where(mask,0,-inf)` → add → **safe_softmax**.

        With a runtime mask, the guard's inactivity cannot be proven in principle, so the
        normalization pass replaces it with safe_softmax by construction. The mask dtype at the
        IR boundary is bool (ADR 0009).
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
        # 0 / -inf are folded into f32 binary initializers, and a single where converts to the
        # additive form
        assert node_ops(graph).count("where") == 1
        assert [spec.dtype for spec in graph.inputs] == ["f32", "f32", "f32", "bool"]

    def test_a_static_finite_mask_still_emits_plain_softmax(self, convert_module):
        """The static-mask path is unchanged (the guard is removed as before; no safe_softmax)."""

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
        """padding_idx is accepted and put into attrs, but has no effect on forward (ADR 0012)."""

        class Table(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Embedding(5, 3, padding_idx=0)

            def forward(self, index):
                return self.table(index)

        graph, _ = convert_module(Table(), (torch.tensor([0, 1, 2, 3]),), ({0: dyn_t},))

        node = only_node(graph, "embedding")
        assert node.attrs == {"padding_idx": 0}
        # weight is f32 / index is i32 (boundary normalization from i64) — the dtype differs per
        # slot
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
        """torch returns `[H]` for a scalar index, but the contract output is `[…index, H]`
        (rank ≥ 1)."""

        class Scalar(nn.Module):
            def __init__(self):
                super().__init__()
                self.table = nn.Embedding(5, 3)

            def forward(self, index):
                return self.table(index)

        with pytest.raises(NotImplementedError, match="scalar index"):
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
        # mask is bool (through a cast), and the output has the shape of x (right-aligned broadcast
        # on the mask side only)
        assert graph.values[node.ins[1]].dtype == "bool"
        assert graph.values[node.outs[0]].shape == [1, 2, "T", "T"]

    def test_masked_fill_value_survives_the_json_round_trip(self, convert_module, dyn_t):
        """ADR 0012: −3.4028234663852886e+38 must be **ulp-stable** through the JSON round trip.

        The graph JSON is carried as text (the container's graph descriptor), so a move of even
        1 ulp in the round trip would make the fill value on the GPU disagree with torch.
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
        # The same bits after rounding to f32 (= exactly the smallest finite f32)
        assert struct.unpack("<f", struct.pack("<f", round_tripped))[0] == round_tripped

    def test_infinite_fill_value_is_rejected(self, convert_module):
        """Filling with `-inf` is a real idiom in attention, but IR v1 does not accept non-finite
        values.

        Silently replacing it with -3.4e38 would break the guarantee downstream of softmax that
        it "really becomes 0", so it fails loudly and the caller rewrites it.
        """

        class Infinite(nn.Module):
            def forward(self, x, mask):
                return x.masked_fill(mask.to(torch.bool), float("-inf"))

        with pytest.raises(NotImplementedError, match="non-finite"):
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
        # MUST: all four are explicit even at their defaults (no default filling — ADR 0012 / 0015)
        assert node.attrs == {"stride": 1, "padding": 1, "dilation": 1, "groups": 1}
        assert len(node.ins) == 3
        assert graph.values[node.ins[1]].shape == [4, 4, 3]
        assert graph.values[node.outs[0]].shape == [1, 4, "T"]

    def test_conv1d_carries_groups_and_dilation(self, convert_module, dyn_t):
        """sdp's DDSConv form (depthwise groups=C, dilation 3, k 5) travels in attrs."""

        class Depthwise(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(6, 6, kernel_size=5, padding=6, dilation=3, groups=6)

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Depthwise(), (torch.randn(1, 6, 8),), ({2: dyn_t},))

        node = only_node(graph, "conv1d")
        assert node.attrs == {"stride": 1, "padding": 6, "dilation": 3, "groups": 6}
        # The weight's second axis is Cin/groups = 1 (left as Cin, the shape layer would fail it)
        assert graph.values[node.ins[1]].shape == [6, 1, 5]
        assert graph.values[node.outs[0]].shape == [1, 6, "T"]

    def test_a_bias_free_conv1d_gets_a_synthesized_zero_bias(self, convert_module, dyn_t):
        """No bias does not fail; it is **normalized to arity 3 by synthesizing a zero bias**
        (ADR 0015)."""

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
        # The synthesized bias travels on the same path and naming (const_<digest>) as ordinary
        # constants
        assert bias_name.startswith("const_")
        assert bias_name in graph.initializers
        assert graph.values[bias_name].shape == [3]
        assert graph.values[bias_name].dtype == "f32"
        synthesized = tensors[graph.initializers[bias_name].tensor]
        assert torch.equal(synthesized, torch.zeros(3))

    def test_two_bias_free_convs_share_one_synthesized_initializer(self, convert_module):
        """Zero biases of the same length fold into one by digest (the dedup of _add_const)."""

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
        """dec's ups form (up 8 / k 16 / pad 4). The weight is [Cin, Cout, K] and the output length
        is L·stride."""

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
        # MUST: [Cin, Cout, K] (transposed relative to conv1d's [Cout, Cin/groups, K])
        assert graph.values[node.ins[1]].shape == [5, 3, 16]
        assert graph.values[node.outs[0]].shape == [1, 3, "8T"]

    def test_a_bias_free_conv_transpose1d_also_gets_a_zero_bias(self, convert_module):
        """The zero-bias synthesis takes Cout from **the weight's second axis** (the first axis
        would give the wrong length)."""

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
        """output_padding / groups / dilation have no attrs field (0 / 1 / 1, as observed)."""

        class Up(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.ConvTranspose1d(3, 3, kernel_size=3, stride=2, **kwargs)

            def forward(self, x):
                return self.up(x)

        with pytest.raises(NotImplementedError, match=why):
            convert_module(Up(), (torch.randn(1, 3, 4),))


class TestLayoutOps:
    """Layout ops (ADR 0011) — view / squeeze / unsqueeze → reshape, permute, expand."""

    def test_view_unsqueeze_and_squeeze_all_normalize_to_reshape(self, convert_module, dyn_t):
        class Reshapes(nn.Module):
            def forward(self, x):
                viewed = x.view(-1, 12)
                lifted = viewed.unsqueeze(1)
                return torch.squeeze(lifted, dim=[1])

        graph, _ = convert_module(Reshapes(), (torch.randn(6, 3, 4),), ({0: dyn_t},))

        assert node_ops(graph) == ["reshape", "reshape", "reshape"]
        # The target shape is the output's declared shape, not attrs (attrs are empty). `-1` arrives
        # resolved.
        assert [node.attrs for node in graph.nodes] == [{}, {}, {}]
        assert graph.values[graph.nodes[0].outs[0]].shape == ["T", 12]
        assert graph.values[graph.nodes[1].outs[0]].shape == ["T", 1, 12]
        assert graph.values[graph.nodes[2].outs[0]].shape == ["T", 12]

    def test_reshapes_to_different_shapes_are_not_merged_by_cse(self, convert_module):
        """reshape has no attrs — without the output shape in the CSE key, one of two silently
        disappears."""

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
        """A permute with rank > STRIDED_RANK still fails (only the owner of the check changed).

        NOTE: ADR 0016's rank-lowering pass (`normalize._lower_reshape_permute`) now catches
        rank≥5 permutes first, so **that pass** is what actually fails it (a chain whose rank-4
        endpoints cannot be found). The STRIDED_RANK check of `_h_permute` remains as the second
        line for paths that call convert directly without normalize.
        """

        class Deep(nn.Module):
            def forward(self, x):
                return x.permute(0, 1, 2, 4, 3)

        with pytest.raises(NotImplementedError, match="reshape/permute chain without end points"):
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
    """Layout group 2 (ADR 0014) — slice / cat / pad / flip.

    The accepting side checks "observed shapes lower to one node", the rejecting side "shapes
    with no representation fail at the boundary" (an axis with no field in the contract table is
    not silently run with a default).
    """

    def test_a_static_channel_slice_becomes_one_node(self, convert_module, dyn_t):
        class Split(nn.Module):
            def forward(self, x):
                return x[:, 3:], x[:, :3]

        graph, _ = convert_module(Split(), (torch.randn(1, 6, 5),), ({2: dyn_t},))

        assert node_ops(graph) == ["slice", "slice"]
        # Negative and omitted indices are resolved against the axis length at the exporter boundary
        # (IR attrs are non-negative only)
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
        """Slicing a symbolic axis belongs to sym_prefix_slice (no duplication — ADR 0014)."""

        class Head(nn.Module):
            def forward(self, x):
                return x[:2] * 2.0

        with pytest.raises(NotImplementedError, match="symbolic dimension"):
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
        # The **list** argument of tensors becomes the arity as is (variable arity)
        assert node.ins == ["a", "b", "c"]
        assert node.attrs == {"dim": 1}
        assert graph.values[node.outs[0]].shape == [1, 6, "T"]

    def test_a_symbolic_cat_axis_sums_into_the_dimension_language(self, convert_module, dyn_t):
        """ADR 0046 — the concatenation axis may be symbolic if it is a constant / a linear
        expression of one symbol.

        It revised ADR 0014's "the concatenation axis is static". DiT's joint attention is
        `cat([self | context], dim=1)` with a symbolic axis length on the self side (`S+1519`) —
        one symbol + a constant fits the canonical grammar `coeff·sym+offset` as is, so the
        reason for rejection (the sum does not fit) does not hold.
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
        """A sum of different symbols does not fit the one-symbol-per-dimension language (the only
        rejection of ADR 0046)."""

        class JoinTime(nn.Module):
            def forward(self, a, b):
                return torch.cat([a, b], 0)

        other = Dim("U", min=2, max=SYM_MAX)
        with pytest.raises(NotImplementedError, match="different symbols"):
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
        # The symbolic last dimension is T+2w (the offset form of the dimension language)
        assert graph.values[node.outs[0]].shape == [1, 2, "T+8"]

    def test_an_asymmetric_pad_keeps_the_two_sides_apart(self, convert_module):
        """Pins the mapping of a pad with different left and right widths with **the torch oracle**
        (ADR 0014).

        `[2, 1]` and `[1, 2]` give **the same output length**, so swapping left and right passes
        both the shape check and the dimension language (the existing symmetric-width cases
        cannot detect it in principle). A test that copies only the attrs values would merely
        confirm "what the exporter emits today", so torch's eager output establishes "two zeros
        at the start, one at the end", and one test ties that order to the attrs.
        """

        class Lopsided(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [2, 1])

        x = torch.tensor([[1.0, 2.0, 3.0]])
        graph, _ = convert_module(Lopsided(), (x,))

        assert only_node(graph, "pad").attrs == {"left": 2, "right": 1}
        assert graph.values[only_node(graph, "pad").outs[0]].shape == [1, 6]
        # torch's meaning: the first element is the width at the **start**. A mix-up changes the
        # order of the zeros.
        assert torch.equal(Lopsided()(x), torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]]))

    def test_a_nonzero_pad_value_is_rejected(self, convert_module):
        class Filled(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [1, 1], value=1.0)

        with pytest.raises(NotImplementedError, match="constant 0"):
            convert_module(Filled(), (torch.randn(2, 3),))

    def test_padding_more_than_the_last_dim_is_rejected(self, convert_module):
        class Wide(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [1, 1, 2, 2])

        with pytest.raises(NotImplementedError, match="last dimension"):
            convert_module(Wide(), (torch.randn(2, 3),))

    def test_a_negative_pad_is_rejected(self, convert_module):
        """A negative width truncates (the meaning of slice) — the same shape must not be writable
        with two ops."""

        class Cropped(nn.Module):
            def forward(self, x):
                return nn.functional.pad(x, [-1, 0])

        with pytest.raises(NotImplementedError, match="negative pad"):
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
        """IR flip takes one axis only (silently collapsing to one axis would leave an axis
        unflipped)."""

        class ReverseBoth(nn.Module):
            def forward(self, x):
                return torch.flip(x, [0, 1])

        with pytest.raises(NotImplementedError, match="one axis only"):
            convert_module(ReverseBoth(), (torch.randn(3, 4),))


class TestRmsNorm:
    """ADR 0017 — arity 2, eps required. The weight is the source of truth for the normalized
    length (not put into attrs)."""

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
        """The weight-less form is normalized to arity 2 by synthesizing ones (`×1`, an exact
        identity)."""

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
        """torch uses `finfo(dtype).eps` when eps is omitted — made explicit at the boundary."""

        class Defaulted(nn.Module):
            def forward(self, x, weight):
                return torch.rms_norm(x, [4], weight)

        graph, _ = convert_module(Defaulted(), (torch.randn(5, 4), torch.rand(4)))

        assert only_node(graph, "rms_norm").attrs == {"eps": float(torch.finfo(torch.float32).eps)}

    def test_a_multi_axis_normalized_shape_is_rejected(self, convert_module):
        """The row kernel assumes the last dimension alone (several axes are other semantics)."""

        class MultiAxis(nn.Module):
            def forward(self, x, weight):
                return torch.rms_norm(x, [3, 4], weight)

        with pytest.raises(NotImplementedError, match="single last dimension"):
            convert_module(MultiAxis(), (torch.randn(5, 3, 4), torch.rand(3, 4)))


class TestConv2d:
    """ADR 0017 — spatial attrs have 2 components H/W, all four must be declared, the weight is
    [Cout,Cin/g,Kh,Kw]."""

    def test_asymmetric_attrs_are_carried_as_pairs(self, convert_module):
        """MUST: a shape that takes Kh≠Kw, H/W-asymmetric stride/padding and Cin≠Cout at once."""

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
        """torch takes `1` / `[1]` / `[1,1]` with the same meaning — the boundary absorbs this."""

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


class TestConv3d:
    """ADR 0118 decision 1 — unbatched rank-4 input only, spatial attrs have 3 components T/H/W,
    all four must be declared, the weight is [Cout,Cin/g,Kt,Kh,Kw]."""

    def test_an_unbatched_conv3d_is_preserved_as_one_node_with_triples(self, convert_module):
        """MUST: stop the decomposition of `aten.conv3d` and keep one node (no scattering into the
        generic `aten.convolution`).

        A shape that takes Kt≠Kh≠Kw, stride / padding asymmetric on all 3 axes and Cin≠Cout at
        once. x is the unbatched `[Cin, T, H, W]`, and the output is rank 4 too,
        `[Cout, Tout, Hout, Wout]` (no batch axis added).
        """

        class Wide(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(
                    4, 6, kernel_size=(3, 2, 1), stride=(1, 2, 1), padding=(0, 1, 0)
                )

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Wide(), (torch.randn(4, 5, 6, 5),))

        assert node_ops(graph) == ["conv3d"]
        node = only_node(graph, "conv3d")
        assert node.attrs == {
            "stride": [1, 2, 1],
            "padding": [0, 1, 0],
            "dilation": [1, 1, 1],
            "groups": 1,
        }
        assert graph.inputs[0].shape == [4, 5, 6, 5]
        assert graph.values[node.ins[1]].shape == [6, 4, 3, 2, 1]
        assert graph.values[node.outs[0]].shape == [6, 3, 4, 5]

    def test_a_scalar_spatial_argument_is_normalized_to_three_components(self, convert_module):
        """`F.conv3d(..., stride=2, padding=1)` reaches aten as scalars; the boundary absorbs it."""

        class Functional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 2, 3, 3, 3))
                self.bias = nn.Parameter(torch.randn(3))

            def forward(self, x):
                return nn.functional.conv3d(x, self.weight, self.bias, stride=2, padding=1)

        graph, _ = convert_module(Functional(), (torch.randn(2, 5, 6, 7),))

        node = only_node(graph, "conv3d")
        assert node.attrs["stride"] == [2, 2, 2]
        assert node.attrs["padding"] == [1, 1, 1]
        assert node.attrs["dilation"] == [1, 1, 1]
        assert graph.values[node.outs[0]].shape == [3, 3, 3, 4]

    def test_a_bias_free_conv3d_gets_a_synthesized_zero_bias(self, convert_module):
        class NoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(6, 3, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)

            def forward(self, x):
                return self.conv(x)

        graph, tensors = convert_module(NoBias(), (torch.randn(6, 2, 4, 4),))

        node = only_node(graph, "conv3d")
        assert len(node.ins) == 3
        assert torch.equal(tensors[graph.initializers[node.ins[2]].tensor], torch.zeros(3))

    def test_dilation_and_groups_are_carried(self, convert_module):
        """The semantics cover groups in general (correct in the IR even outside the GPU subset —
        the check is at plan time)."""

        class Dilated(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(
                    4, 2, kernel_size=(2, 2, 2), padding=(1, 2, 1), dilation=(1, 2, 1), groups=2
                )

            def forward(self, x):
                return self.conv(x)

        graph, _ = convert_module(Dilated(), (torch.randn(4, 3, 4, 5),))

        node = only_node(graph, "conv3d")
        assert node.attrs["dilation"] == [1, 2, 1]
        assert node.attrs["groups"] == 2
        assert graph.values[node.ins[1]].shape == [2, 2, 2, 2, 2]
        assert graph.values[node.outs[0]].shape == [2, 4, 6, 6]

    def test_a_batched_rank5_input_fails_loudly(self, convert_module):
        """MUST: batched rank 5 is outside the contract — it fails, naming the recipe-side way to
        drop B = 1."""

        class Batched(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(2, 3, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        with pytest.raises(NotImplementedError, match="batched rank 5"):
            convert_module(Batched(), (torch.randn(1, 2, 3, 4, 4),))


class TestUpsampleBilinear2d:
    """Core ATen layer (the old layer 1) — align_corners=True only. Only the observed form (size
    given, rank 4, f32) is accepted."""

    def test_a_size_specified_interpolate_becomes_one_node(self, convert_module):
        """`F.interpolate(size=…, mode="bilinear")` lowers to one node (not decomposed)."""

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
        """Downscaling is the same op (also one op in torch; no antialias, by specification)."""

        class Shrink(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, size=(2, 3), mode="bilinear", align_corners=True
                )

        graph, _ = convert_module(Shrink(), (torch.randn(1, 3, 8, 10),))

        assert only_node(graph, "upsample_bilinear2d").attrs == {"output_size": [2, 3]}

    def test_align_corners_false_is_rejected(self, convert_module):
        """MUST: the coordinate formula and the edge handling differ, so it is not silently run with
        the same op."""

        class HalfPixel(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, size=(7, 9), mode="bilinear", align_corners=False
                )

        with pytest.raises(NotImplementedError) as err:
            convert_module(HalfPixel(), (torch.randn(1, 3, 4, 5),))

        assert "align_corners" in str(err.value)

    def test_a_scale_factor_is_rejected(self, convert_module):
        """MUST: a scale factor adds another rounding rule for the output length (two IRs for the
        same shape)."""

        class Doubled(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, scale_factor=2.0, mode="bilinear", align_corners=True
                )

        with pytest.raises(NotImplementedError) as err:
            convert_module(Doubled(), (torch.randn(1, 3, 4, 5),))

        assert "scale_factors" in str(err.value)

    def test_a_non_f32_input_is_rejected(self, convert_module):
        """The contract is f32 only. f16 / f64 arrive through the same overload, so they fail at the
        entry."""

        class Resize(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(
                    x, size=(7, 9), mode="bilinear", align_corners=True
                )

        with pytest.raises(NotImplementedError) as err:
            convert_module(Resize(), (torch.randn(1, 3, 4, 5, dtype=torch.float64),))

        assert "f32" in str(err.value)

    def test_a_non_bilinear_mode_is_not_in_the_vocabulary(self, convert_module):
        """nearest is another aten op (`upsample_nearest2d.vec`), not in the table, so it fails.

        MUST: mode is not in attrs, so this is the only check on the path where "another
        interpolation silently runs as bilinear".
        """

        class Nearest(nn.Module):
            def forward(self, x):
                return nn.functional.interpolate(x, size=(7, 9), mode="nearest")

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert_module(Nearest(), (torch.randn(1, 3, 4, 5),))

        assert "upsample_nearest2d" in str(err.value)


class TestGruScan:
    """Extended composite layer (the old layer 2) — GRU hidden-side scan (ADR 0056). Only the
    observed form (arity 4, f32) is accepted."""

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
        """MUST: the time axis stays symbolic in one node (the very reason this op exists).

        `aten.gru.input` decomposes into a full unroll over time, so its length gets
        specialized, but the body of a `karume::` custom op sits behind `register_fake`, so the
        trace does not enter it and it is not unrolled even through
        `run_decompositions(curated_decompositions())`.
        """
        graph, _ = convert_module(
            self._module(reverse=reverse), self._args(), dynamic_shapes=({0: dyn_t}, None)
        )

        name = "gru_scan_reverse" if reverse else "gru_scan"
        assert node_ops(graph) == [name]
        node = only_node(graph, name)
        # Empty-attrs contract (the op name carries the scan direction — no bool attr in the
        # vocabulary)
        assert node.attrs == {}
        # The four gi / h0 / w_hh / b_hh (in the custom op's positional order)
        assert len(node.ins) == 4
        assert graph.values[node.outs[0]].shape == ["T", 1, self.HIDDEN]

    def test_both_directions_are_distinct_ops(self, convert_module, dyn_t):
        """MUST: the direction is a separate op name. Folding into one name hides the scan order
        from the IR."""

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
        # Layers are joined by the existing cat (dim 2 is a static axis, so not even ADR 0046's
        # relaxation is needed)
        assert only_node(graph, "cat").attrs == {"dim": 2}

    def test_a_non_f32_input_is_rejected(self, convert_module):
        """The contract is f32 only (f64 arrives through the same op)."""
        args = tuple(tensor.to(torch.float64) for tensor in self._args())

        with pytest.raises(NotImplementedError) as err:
            convert_module(self._module().to(torch.float64), args)

        assert "f32" in str(err.value)

    def test_a_batch_first_layout_is_rejected(self, convert_module):
        """MUST: `gi` is fixed to `[T,N,3H]` (time-first).

        There is no `batch_first` field, so a rank-2 `gi` with the time and feature axes folded
        fails **here** (let through, `[N,3H]` would be silently scanned as "a sequence with
        T = N").
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
    """Extended atomic layer (the old layer 1') — DCNv2 only (ADR 0055). Only the observed form
    (BiRefNet's 20 sites) is accepted."""

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
        """`torchvision::deform_conv2d` stays one node after the curated decomposition."""
        graph, _ = convert_module(self._module(), self._args())

        assert node_ops(graph) == ["deform_conv2d"]
        node = only_node(graph, "deform_conv2d")
        assert node.attrs == {"padding": [1, 0]}
        # The five inputs x / weight / offset / mask / bias (in torchvision's positional order)
        assert len(node.ins) == 5
        assert graph.values[node.outs[0]].shape == [1, 5, 4, 4]

    def test_a_missing_bias_is_folded_into_an_initializer(self, convert_module):
        """Without bias, the torchvision wrapper inserts `aten.full`, and the export-eliminated
        layer turns it into an initializer.

        MUST: the exporter does not synthesize a zero bias (no second synthesis path — ADR 0055).
        """
        graph, tensors = convert_module(self._module(bias=False), self._args())

        assert node_ops(graph) == ["deform_conv2d"]
        node = only_node(graph, "deform_conv2d")
        assert node.ins[4] in graph.initializers
        assert tensors[graph.initializers[node.ins[4]].tensor].tolist() == [0.0] * 5

    def test_an_unmodulated_deform_conv_is_rejected(self, convert_module):
        """MUST: use_mask=False (DCNv1) fails.

        With mask=None, torchvision passes **a [1,1] dummy tensor**, so letting it through would
        apply the dummy as the modulator — a silent wrong value.
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
        """MUST: values with no attrs field (= fixed at 1) fail on **the actual value**, not on a
        filled-in default."""

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
        """MUST: offset_groups derives from the offset's channel count (no field = fixed at 1)."""

        class Grouped(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 3, 3, 2))

            def forward(self, x, offset, mask):
                return deform_conv2d(
                    input=x, offset=offset, weight=self.weight, padding=(1, 0), mask=mask
                )

        # offset_groups = 3 (the offset has 3 * 2 * Kh * Kw channels)
        with pytest.raises(NotImplementedError) as err:
            convert_module(
                Grouped(),
                (torch.randn(1, 3, 4, 5), torch.randn(1, 36, 4, 4), torch.randn(1, 18, 4, 4)),
            )

        assert "offset_groups" in str(err.value)

    def test_a_non_f32_input_is_rejected(self, convert_module):
        """The contract is f32 only (f64 arrives through the same op)."""
        args = tuple(tensor.to(torch.float64) for tensor in self._args())
        module = self._module().to(torch.float64)

        with pytest.raises(NotImplementedError) as err:
            convert_module(module, args)

        assert "f32" in str(err.value)


class TestCoefficientAdd:
    """Addition with a coefficient (`alpha != 1`) is not in the IR vocabulary — conversion fails
    loudly.

    The normalization side leaves it alone (the three alpha guards of `normalize`), so this is the
    last check that stops IR with a silently dropped coefficient from being written. Whether
    alpha is folded by the decomposition depends on the torch version, so it is **injected**
    into the FX node's kwargs to stay independent of the torch version.
    """

    @staticmethod
    def _decomposed_add(alpha: int):
        class Alpha(nn.Module):
            def forward(self, x, y):
                return torch.add(x, y, alpha=alpha)

        decomposed = torch.export.export(
            Alpha(), (torch.randn(3, 4), torch.randn(3, 4)), strict=False
        ).run_decompositions(curated_decompositions())
        node = next(
            node
            for node in decomposed.graph_module.graph.nodes
            if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor
        )
        node.kwargs = {**node.kwargs, "alpha": alpha}
        normalize_graph(decomposed)
        return decomposed

    def test_a_coefficient_is_rejected_instead_of_being_dropped(self):
        with pytest.raises(NotImplementedError, match="alpha != 1"):
            convert(self._decomposed_add(2))

    def test_the_same_add_converts_when_the_coefficient_is_one(self):
        """Counterpart — `alpha == 1` is a plain add (detects a tautological rejection above)."""
        graph, _ = convert(self._decomposed_add(1))

        assert node_ops(graph) == ["add"]
