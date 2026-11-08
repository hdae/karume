"""Pins the behavior of the normalization passes.

Each pass is "an equivalent rewrite that adds no IR vocabulary", so both ① that the form fails as
outside the vocabulary before the rewrite and ② that it becomes the intended single node after
the rewrite are checked (either side alone could be a tautological test that passes even without
the pass).
"""

from __future__ import annotations

import pytest
import torch
from conftest import export_and_convert, node_ops, only_node
from torch import nn
from torch.export import Dim

from karume import normalize
from karume.convert import UnsupportedAtenOpsError, convert, curated_decompositions
from karume.normalize import SAFE_SOFTMAX_META, normalize_graph
from karume.ops import STRIDED_RANK

aten = torch.ops.aten


def decompose(module: nn.Module, args: tuple, dynamic_shapes=None):
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
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
        """Confirms the pass is load-bearing — without the rewrite it fails on an unsupported op."""
        decomposed = decompose(Square(), (torch.randn(3, 4),))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.pow.Tensor_Scalar" in err.value.ops

    def test_other_exponents_are_left_alone(self):
        """No non-equivalent rewrite (a cube does not become a single mul)."""
        decomposed = decompose(Cube(), (torch.randn(3, 4),))
        normalize_graph(decomposed)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.pow.Tensor_Scalar" in err.value.ops

    def test_a_promoting_float_exponent_is_left_alone(self):
        """`2.0` slips through as `2.0 == 2`, but on an integer tensor it involves type promotion
        (not equivalent).

        `pow(i64, 2.0)` returns f32 while `mul(i64, i64)` stays i64 — a rewrite would silently
        give values of a different dtype. Left alone, it fails as an unsupported op in the full
        enumeration.
        """
        decomposed = decompose(FloatSquare(), (torch.arange(6).reshape(3, 2),))
        stats = normalize_graph(decomposed)

        assert "pow2->mul" not in stats
        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.pow.Tensor_Scalar" in err.value.ops

    def test_a_float_exponent_is_left_alone_even_without_promotion(self):
        """Only **the integer literal 2** is accepted (no rewrite in f32 even without promotion).

        Relying on `2.0 == 2` would leave the judgment "equivalent if there is no promotion" to a
        single dtype check. Making the spelling of the exponent itself the check keeps the
        acceptance condition readable as "the form as written".
        """
        decomposed = decompose(FloatSquare(), (torch.randn(3, 2),))
        stats = normalize_graph(decomposed)

        assert "pow2->mul" not in stats


class FloatSquare(nn.Module):
    """A square involving type promotion (integer tensor × float exponent)."""

    def forward(self, x):
        return x**2.0


class Repeated(nn.Module):
    """An identity repeat (all arguments 1) — the same form as the 4 observed ones (recon §4-1)."""

    def forward(self, x):
        return x.repeat(1, 1)


class Tripled(nn.Module):
    """A real repeat (not in the IR vocabulary) — the counterpart showing identity removal does not
    misfire."""

    def forward(self, x):
        return x.repeat(3, 1)


class RankRaisingRepeat(nn.Module):
    """Even with all arguments 1, more arguments than the rank raise the rank (not an identity)."""

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
        """A repeat with B > 1 and a form that raises the rank are not rewritten (no silent
        approximation)."""
        decomposed = decompose(module, (torch.randn(3, 4),))
        normalize_graph(decomposed)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.repeat.default" in err.value.ops


class Alpha(nn.Module):
    """Addition with a coefficient (`x + alpha·y`) — not in the IR vocabulary, so normalization
    leaves it alone and conversion fails on it."""

    def forward(self, x, y):
        return torch.add(x, y, alpha=2)


#: The subject of {@link Alpha} (`_rewrite_add` swaps the second operand for a scalar).
ALPHA_ARGS = (torch.randn(3, 4), torch.randn(4))


def _rewrite_add(decomposed, *, addend: float, alpha: int):
    """Rewrites the `aten.add.Tensor` of the decomposed graph into "tensor + scalar (coefficient
    alpha)".

    Whether decomposition folds alpha depends on the torch version, so **both values are injected**
    to stay independent of it (an extension of the technique the test already uses to rewrite
    args). `addend` changes with the guard to pin — promotion (`_promote_scalar_operands`) fires
    only on a non-zero value, identity removal (`_drop_identity_add`) only on 0.0.
    """
    node = next(
        node
        for node in decomposed.graph_module.graph.nodes
        if node.op == "call_function" and node.target is aten.add.Tensor
    )
    node.args = (node.args[0], addend)
    node.kwargs = {**node.kwargs, "alpha": alpha}
    return node


class TestDropIdentityAdd:
    def test_adding_zero_disappears(self):
        graph, _ = export_and_convert(ZeroAdd(), (torch.randn(3, 4),))

        assert node_ops(graph) == []

    def test_it_runs_before_scalar_promotion(self):
        """In the reverse order `+0` would become a rank-1 constant and no longer be removed as an
        identity."""
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

    def test_a_coefficient_add_is_not_dropped_as_identity(self):
        """`x + 2·0` is an identity too, but it is left alone here so as not to write **IR that
        drops alpha**.

        Touching it would next lead to "IR that silently dropped the coefficient", so identity
        removal applies only to the form without alpha (the counterpart is the one test below).
        """
        decomposed = decompose(Alpha(), ALPHA_ARGS)
        _rewrite_add(decomposed, addend=0.0, alpha=2)

        stats = normalize_graph(decomposed)

        assert "drop_identity_add" not in stats

    def test_the_same_zero_addend_is_dropped_without_alpha(self):
        """Counterpart — with `alpha == 1` the same form disappears (keeps the assert above from
        being tautological)."""
        decomposed = decompose(Alpha(), ALPHA_ARGS)
        _rewrite_add(decomposed, addend=0.0, alpha=1)

        stats = normalize_graph(decomposed)

        assert stats["drop_identity_add"] == 1


class TestPromoteScalarOperands:
    """Scalar operands become rank-1 constants (adding neither IR vocabulary nor attrs — recon
    §4-4)."""

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
        """`1 − mask` is not commutative — putting the constant on the right silently flips the sign
        (the observed form of the mask path)."""

        class OneMinus(nn.Module):
            def forward(self, mask):
                return 1 - mask

        graph, tensors = export_and_convert(OneMinus(), (torch.ones(3, 4, dtype=torch.int64),))

        node = only_node(graph, "sub")
        assert node.ins[1] == "mask"
        constant = tensors[graph.initializers[node.ins[0]].tensor]
        assert torch.equal(constant, torch.tensor([1], dtype=torch.int32))

    def test_the_constant_takes_the_tensor_dtype(self):
        """The i32 side gets an i32 initializer (the binary-op contract "input dtypes are the same"
        — ADR 0010)."""

        class Offset(nn.Module):
            def forward(self, mask):
                return mask - 1

        graph, _ = export_and_convert(Offset(), (torch.ones(3, 4, dtype=torch.int64),))

        constant = only_node(graph, "sub").ins[1]
        assert graph.values[constant].dtype == "i32"
        assert graph.initializers[constant].storage.dtype == "i32"

    def test_a_rank0_tensor_operand_is_left_alone(self):
        """`[] × [1] → [1]` raises the rank — promoting would silently make the declaration and the
        data disagree.

        The observed `sqrt(c * 3)` of a lifted scalar constant has this form too (it disappears in
        folding).
        """

        class ScalarTensor(nn.Module):
            def forward(self, x):
                scalar = x.sum(dim=-1).amax(dim=-1)  # rank-0 tensor
                return x + (scalar * 3.0)

        decomposed = decompose(ScalarTensor(), (torch.randn(3, 4),))
        stats = normalize_graph(decomposed)

        assert "promote_scalar_operand" not in stats

    def test_alpha_is_left_alone(self):
        """Addition with a coefficient is not in the IR vocabulary — promotion does not silently
        drop alpha.

        Whether decomposition folds alpha depends on the torch version, so it is **injected** to
        exercise it deterministically (an extension of the technique the test already uses to
        rewrite args). The counterpart is the same subject without alpha — that one is promoted,
        so this assert is not tautological.
        """
        decomposed = decompose(Alpha(), ALPHA_ARGS)
        _rewrite_add(decomposed, addend=1.0, alpha=2)

        stats = normalize_graph(decomposed)

        assert "promote_scalar_operand" not in stats

    def test_the_same_addend_is_promoted_without_alpha(self):
        """Counterpart — with `alpha == 1` the same form is promoted (keeps the assert above from
        being tautological)."""
        decomposed = decompose(Alpha(), ALPHA_ARGS)
        _rewrite_add(decomposed, addend=1.0, alpha=1)

        stats = normalize_graph(decomposed)

        assert stats["promote_scalar_operand"] == 1


class MaskedByEqZero(nn.Module):
    """A reduced version of the observed form (`scores.masked_fill(attn_mask == 0, -1e4)`) — recon
    §2.

    attn_mask comes from a graph input, so it cannot be constant-folded and `eq` stays in the
    executed graph.
    """

    def forward(self, scores, mask):
        return scores.masked_fill(mask == 0, -1e4)


class EqNonZero(nn.Module):
    def forward(self, x):
        return x == 2


class TestEqZeroToNotBool:
    """`eq(x, 0)` → `bitwise_not(cast(x, bool))` (ADR 0015 — the ruling not to add a new op).

    The equivalence rests on the cast convention "x → bool is x != 0": `not (x != 0)` is exactly
    `x == 0`.
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
        """Confirms the pass is load-bearing — without the rewrite it fails on an unsupported op."""
        decomposed = decompose(
            MaskedByEqZero(),
            (torch.randn(2, 3), torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.int64)),
        )

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.eq.Scalar" in err.value.ops

    def test_a_non_zero_comparand_is_left_alone(self):
        """The rewrite does not hold for comparisons with anything other than 0, so the form is left
        alone and goes to the enumeration."""
        decomposed = decompose(EqNonZero(), (torch.randn(3),))

        stats = normalize_graph(decomposed)

        assert "eq_zero->bitwise_not_cast" not in stats
        with pytest.raises(UnsupportedAtenOpsError):
            convert(decomposed)


class TestDropMetadataAsserts:
    """`_assert_tensor_metadata` is a side-effect node that does not appear in the IR.

    eliminate_dead_code deems it impure and keeps it, so unless it is removed, already-replaced
    subtrees survive through the assert.
    """

    def test_assert_nodes_are_removed(self):
        decomposed = decompose(NoOpCast(), (torch.randn(3, 4),))
        graph = decomposed.graph_module.graph
        before = [
            node for node in graph.nodes if node.target is aten._assert_tensor_metadata.default
        ]
        assert before, "precondition: assert nodes exist after decomposition"

        stats = normalize_graph(decomposed)

        assert stats["drop_metadata_assert"] == len(before)
        assert not [
            node for node in graph.nodes if node.target is aten._assert_tensor_metadata.default
        ]


class Split(nn.Module):
    """A reduced form of ConvFlow / ResidualCoupling's `torch.split(x, [half]*2, 1)` (recon §2)."""

    def forward(self, x):
        first, second = torch.split(x, [2, 3], 1)
        return first * 2.0, second


class TestSplitToSlices:
    """split_with_sizes + getitem → a sequence of slices (the IR has no multi-output op — ADR
    0014)."""

    def test_each_getitem_becomes_its_own_slice(self):
        graph, _ = export_and_convert(Split(), (torch.randn(1, 5, 4),))

        assert node_ops(graph) == ["slice", "slice", "mul"]
        # The ranges are exactly the cumulative sizes (shifting a boundary by one makes ranges
        # overlap / leaves a gap)
        assert graph.nodes[0].attrs == {"dim": 1, "start": 0, "end": 2}
        assert graph.nodes[1].attrs == {"dim": 1, "start": 2, "end": 5}
        assert graph.nodes[0].ins == ["x"] and graph.nodes[1].ins == ["x"]

    def test_the_pass_reports_its_firing_count(self):
        decomposed = decompose(Split(), (torch.randn(1, 5, 4),))

        # As many as the extraction points (one getitem = one slice)
        assert normalize_graph(decomposed)["split_with_sizes->slice"] == 2

    def test_without_the_pass_split_is_outside_the_vocabulary(self):
        """Confirms the pass is load-bearing — without the rewrite it fails on an unsupported op."""
        decomposed = decompose(Split(), (torch.randn(1, 5, 4),))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        # split's own meta is not a tensor (a tuple), so what shows in the enumeration is the
        # getitem extraction points.
        assert [name for names in err.value.ops.values() for name in names] == [
            "getitem",
            "getitem_1",
        ]


# ---- M1-P4 phase 2 (ADR 0016) — rms fold / guard removal / rank lowering / select ------


class HandWrittenRmsNorm(nn.Module):
    """The same hand-written decomposition as Qwen3 / DiT (preserve cannot fold it; it needs FX
    matching)."""

    EPS = 1e-6

    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, hidden))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.EPS) * self.weight


class BroadcastRmsNorm(HandWrittenRmsNorm):
    """A form whose weight is not rank 1 of the last-dimension length (different semantics from the
    contract, so not folded)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, 3).reshape(3, 1))


class PowRmsNorm(HandWrittenRmsNorm):
    """The Gemma4 form — `pow(ms, -0.5)` instead of rsqrt (the form modeling_gemma4.py writes
    because of torch/JAX compiler differences; `_pow_neg_half_to_rsqrt` normalizes it in an
    earlier step)."""

    def forward(self, x):
        return x * (x.pow(2).mean(-1, keepdim=True) + self.EPS).pow(-0.5) * self.weight


class WeightlessRmsNorm(nn.Module):
    """Gemma4 `v_norm` (RMSNorm with_scale=False) — the form without weight."""

    EPS = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.EPS)


class TestFoldRmsNorm:
    def test_the_decomposed_form_becomes_one_rms_norm_node(self):
        graph, _ = export_and_convert(HandWrittenRmsNorm(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["rms_norm"]
        assert only_node(graph, "rms_norm").attrs == {"eps": HandWrittenRmsNorm.EPS}

    def test_it_runs_before_the_pow2_rewrite(self):
        """Order MUST (ADR 0016) — if `pow2->mul` ran first, the pattern would lose one node.

        With rms fold first, `pow(x,2)` disappears together with the rms_norm, so `pow2->mul`
        fires 0 times. In the reverse order mul(x,x) remains and the stats show 1 (= a swapped
        order shows in the count).
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

    def test_the_pow_form_becomes_one_rms_norm_node(self):
        """Gemma4's pow(ms, -0.5) form also folds into one node via `_pow_neg_half_to_rsqrt`."""
        graph, _ = export_and_convert(PowRmsNorm(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["rms_norm"]
        assert only_node(graph, "rms_norm").attrs == {"eps": PowRmsNorm.EPS}

    def test_a_stray_pow_neg_half_surfaces_as_rsqrt_in_the_enumeration(self):
        """A pow(-0.5) that does not fold into RMSNorm becomes rsqrt and **fails as an unsupported
        op** (rsqrt is not in the IR vocabulary — pins that it does not pass silently)."""

        class StrayPow(nn.Module):
            def forward(self, x):
                return (x * x + 1.0).pow(-0.5)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            export_and_convert(StrayPow(), (torch.randn(3, 4),))

        assert "aten.rsqrt.default" in err.value.ops

    def test_the_weightless_form_becomes_rms_norm_with_synthesized_ones(self):
        """The form without weight (Gemma4 v_norm) becomes one rms_norm node + synthesized ones (×1,
        an exact identity)."""
        graph, tensors = export_and_convert(WeightlessRmsNorm(), (torch.randn(3, 4),))

        assert node_ops(graph) == ["rms_norm"]
        node = only_node(graph, "rms_norm")
        assert node.attrs == {"eps": WeightlessRmsNorm.EPS}
        weight_key = graph.initializers[node.ins[1]].tensor
        assert torch.equal(tensors[weight_key], torch.ones(4))

    def test_a_coefficient_on_the_eps_add_blocks_the_fold(self):
        """A form with a coefficient on `add(ms, eps)` is not folded (eps is not silently read as
        `alpha·eps`).

        Removing the alpha guard of `_eps_operand` would make the attrs of the folded rms_norm
        disagree with the actual eps — a form where only the values quietly drift, so not folding
        is pinned (the counterpart is the test above that runs the same subject without alpha).
        """
        decomposed = decompose(HandWrittenRmsNorm(), (torch.randn(3, 4),))
        eps_add = next(
            node
            for node in decomposed.graph_module.graph.nodes
            if node.op == "call_function"
            and node.target in (aten.add.Tensor, aten.add.Scalar)
            and node.args[1] == HandWrittenRmsNorm.EPS
        )
        eps_add.kwargs = {**eps_add.kwargs, "alpha": 2}

        stats = normalize_graph(decomposed)

        assert "rms_norm" not in stats

    def test_a_weighted_norm_never_splits_into_the_weightless_form(self):
        """The second sweep (without weight) does not grab the inner mul of a norm with weight
        first.

        If it did, the norm would split into rms_norm_weightless + a bare mul and the weight fusion
        would disappear — the breakdown of the stats is the detector.
        """
        decomposed = decompose(HandWrittenRmsNorm(), (torch.randn(3, 4),))

        stats = normalize_graph(decomposed)

        assert stats["rms_norm"] == 1
        assert "rms_norm_weightless" not in stats

    def test_a_broadcast_weight_folds_weightless_and_keeps_its_own_mul(self):
        """A broadcast weight does **not go into** the rms_norm (the contract = the weight's length
        is the source of truth for the normalized length). Only the normalization part folds without
        weight, and the broadcast multiplication stays as a bare mul — semantics identical to torch,
        pinning that a weight outside the contract does not get into the rms_norm."""
        graph, tensors = export_and_convert(BroadcastRmsNorm(), (torch.randn(3, 3),))

        assert node_ops(graph) == ["rms_norm", "mul"]
        node = only_node(graph, "rms_norm")
        weight_key = graph.initializers[node.ins[1]].tensor
        assert torch.equal(tensors[weight_key], torch.ones(3))


class CausalAttention(nn.Module):
    """The causal-mask path of SDPA (the smallest form that raises the safe-softmax guard)."""

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


class FullyMaskedAttention(nn.Module):
    """A mask that is -inf everywhere (a form where the guard **really matters** — removing it lets
    NaN flow downstream)."""

    def forward(self, q, k, v):
        bias = torch.full((1, 1, 4, 4), float("-inf"))
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)


class LiteralMaskedAttention(nn.Module):
    """A mask built from a literal in the graph (a -inf source that does not write `-inf`
    directly).

    `negate=True` negates a `+inf` literal into -inf (the `-mask * inf` kind).
    """

    def __init__(self, literal: float, *, negate: bool = False) -> None:
        super().__init__()
        self.literal = literal
        self.negate = negate

    def forward(self, q, k, v):
        bias = torch.full((1, 1, 4, 4), self.literal)
        return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=-bias if self.negate else bias
        )


class BufferMaskedAttention(nn.Module):
    """SDPA holding its mask via `register_buffer`.

    torch.export **lifts a buffer to a placeholder**, so a verdict that assumes "placeholders are
    finite" lets this form through (it removes the guard even with a -inf row = NaN flows
    downstream).
    """

    def __init__(self, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("bias", bias)

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.bias)


class RuntimeMaskedAttention(nn.Module):
    """SDPA whose mask is **a run-time input** (the main subject of ADR 0044).

    The actual values are not known at export time, so inactivity can in principle not be proven.
    """

    def forward(self, q, k, v, mask):
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class NegInfScoreAttention(nn.Module):
    """A form with a -inf source on the score **side** too (the other unprovable kind — evaluating
    the mask does not settle it).

    The mask is finite and leaves no row empty, but the buffer on the q side contains -inf, which
    breaks the premise of the proof that "-inf enters only from the one additive mask".
    """

    def __init__(self) -> None:
        super().__init__()
        bias = torch.zeros(1, 1, 4, 4)
        bias[..., 0] = float("-inf")
        self.register_buffer("bias", bias)
        offset = torch.zeros(1, 1, 4, 3)
        offset[..., 0, 0] = float("-inf")
        self.register_buffer("offset", offset)

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(
            q + self.offset, k, v, attn_mask=self.bias
        )


class LogScoreAttention(nn.Module):
    """A form with `log` (an op that **produces** -inf) on the score side. The mask itself is
    finite.

    `log(0) = -inf` has no -inf literal among its arguments, so a verdict that looks only at the
    constant sweep does not count it as a source — the dependency cone is wrongly judged "finite"
    and the guard is **removed**.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bias", torch.zeros(1, 1, 4, 4))

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(
            torch.log(q.abs()), k, v, attn_mask=self.bias
        )


class DividedScoreAttention(nn.Module):
    """A form whose divisor on the score side is **a tensor** (cannot be shown non-zero) — `div(x,
    0) = ±inf`."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bias", torch.zeros(1, 1, 4, 4))

    def forward(self, q, k, v, divisor):
        return nn.functional.scaled_dot_product_attention(q / divisor, k, v, attn_mask=self.bias)


class GatheredMaskAttention(nn.Module):
    """A form that builds the mask with `index_select` from a buffer (the kind where evaluating the
    inactivity verdict raises).

    The evaluation resolves the buffer (placeholder) to its **actual values** (last axis 4) while
    substituting probe values (5 / 9) for the symbolic length, so when the two meet in one
    expression torch raises `IndexError`. That is a failure of the verdict, not a fault of the
    exporter, so it must not take the whole export down.
    """

    def __init__(self) -> None:
        super().__init__()
        bias = torch.zeros(1, 1, 1, 4)
        bias[..., 0] = float("-inf")
        self.register_buffer("bias", bias)

    def forward(self, q, k, v):
        index = torch.arange(k.shape[-2], device=k.device)
        return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=torch.index_select(self.bias, -1, index)
        )


ATTENTION_ARGS = (torch.randn(1, 1, 4, 3), torch.randn(1, 1, 4, 3), torch.randn(1, 1, 4, 3))
RUNTIME_MASK_ARGS = (*ATTENTION_ARGS, torch.tensor([[[[True, True, False, False]]]]))
#: The axis spec for exporting `GatheredMaskAttention` with a symbolic length (the sequence axis of
#: q / k / v).
SYMBOLIC_LENGTH_SHAPES = ({2: Dim("T", max=16)},) * 3


def _guard_ops_gone(decomposed) -> bool:
    return not any(
        node.target in (aten.any.dim, aten.full_like.default, aten.logical_not.default)
        for node in decomposed.graph_module.graph.nodes
        if node.op == "call_function"
    )


def _softmax_nodes(decomposed) -> list:
    return [
        node
        for node in decomposed.graph_module.graph.nodes
        if node.op == "call_function" and node.target is aten.softmax.int
    ]


class TestDropSafeSoftmaxGuard:
    def test_a_symbolic_mask_keeps_the_guard_outside_the_probe_lengths(self):
        """Non-empty rows at T=5/9 do not rule out fully masked rows at T=2..4."""

        class ShortMaskedRows(nn.Module):
            def forward(self, x):
                mask = torch.where(torch.arange(x.shape[1]) >= 4, 0.0, float("-inf"))
                scores = x + mask
                probabilities = torch.softmax(scores, dim=-1)
                empty = torch.logical_not(
                    torch.any(torch.logical_not(scores == float("-inf")), dim=-1, keepdim=True)
                )
                return torch.where(empty, torch.full_like(probabilities, 0), probabilities)

        graph, _ = export_and_convert(
            ShortMaskedRows(),
            (torch.zeros(2, 6),),
            ({1: Dim("T", min=2, max=16)},),
        )
        assert "softmax" not in node_ops(graph)
        assert len([node for node in graph.nodes if node.op == "safe_softmax"]) == 1

    def test_a_causal_mask_lets_the_guard_be_removed(self):
        args = (torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3))
        decomposed = decompose(CausalAttention(), args)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:masked-rows-nonempty"] == 1
        assert _guard_ops_gone(decomposed)
        # A proven form **stays a plain softmax** (does not drop to safe_softmax = existing assets
        # unchanged)
        assert not any(node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed))

    def test_without_the_pass_the_guard_ops_are_outside_the_vocabulary(self):
        args = (torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3))
        decomposed = decompose(CausalAttention(), args)

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.any.dim" in err.value.ops
        assert "aten.full_like.default" in err.value.ops

    def test_a_runtime_mask_rewrites_the_guard_to_safe_softmax(self):
        """A run-time mask can in principle not be proven → constructive replacement with
        safe_softmax (ADR 0044)."""
        decomposed = decompose(RuntimeMaskedAttention(), RUNTIME_MASK_ARGS)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:rewritten-safe"] == 1
        assert _guard_ops_gone(decomposed)
        assert [node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed)] == [True]

    def test_an_all_masked_row_rewrites_the_guard_to_safe_softmax(self):
        """A static mask with an actual all -inf row also drops to safe_softmax.

        Under ADR 0016 this failed loudly (removing the guard lets NaN flow downstream). ADR 0044
        added an op with **exactly the same semantics** as the guard, so the guard is replaced
        rather than removed — nothing is removed, so no NaN flows.
        """
        decomposed = decompose(FullyMaskedAttention(), ATTENTION_ARGS)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:rewritten-safe"] == 1
        assert [node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed)] == [True]

    def test_an_all_masked_row_in_a_buffer_rewrites_the_guard_to_safe_softmax(self):
        """A -inf source that is **a placeholder (buffer)** is also judged by its actual values, and
        since it cannot be proven it goes to replacement.

        A verdict that assumes "placeholders are finite" lets this form through, the guard is
        removed and NaN flows downstream (a hole that a check looking only at in-graph
        `torch.full` forms misses).
        """
        module = BufferMaskedAttention(torch.full((1, 1, 4, 4), float("-inf")))
        decomposed = decompose(module, ATTENTION_ARGS)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:rewritten-safe"] == 1
        assert [node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed)] == [True]

    @pytest.mark.parametrize(
        "module",
        [
            pytest.param(LiteralMaskedAttention(float("inf"), negate=True), id="negated-pos-inf"),
            pytest.param(LiteralMaskedAttention(-1e39), id="f32-overflowing-literal"),
        ],
    )
    def test_a_literal_that_is_neg_inf_only_in_f32_or_after_negation_is_a_source(self, module):
        """A negated `+inf` and a literal that becomes -inf when rounded to f32 also count as -inf
        sources.

        A verdict that looks only at the double comparison `value == -inf` wrongly calls both
        "finite" and **removes** the guard despite a row that is -inf everywhere (the IR turns a
        row torch returns as 0 into NaN).
        """
        decomposed = decompose(module, ATTENTION_ARGS)

        stats = normalize_graph(decomposed)

        assert "softmax_guard:no-neg-inf" not in stats
        assert [node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed)] == [True]

    def test_the_most_negative_finite_f32_literal_is_not_a_source(self):
        """`finfo(f32).min` is finite in f32 — it is not a source even after rounding (the detector
        of over-detection)."""
        decomposed = decompose(
            LiteralMaskedAttention(torch.finfo(torch.float32).min), ATTENTION_ARGS
        )

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:no-neg-inf"] == 1
        assert not any(node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed))

    def test_a_neg_inf_source_on_the_score_side_rewrites_to_safe_softmax(self):
        """A form with a -inf source on the score side (the other kind that evaluating the mask does
        not settle) also goes to replacement."""
        decomposed = decompose(NegInfScoreAttention(), ATTENTION_ARGS)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:rewritten-safe"] == 1
        assert [node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed)] == [True]

    def test_a_probe_that_raises_rewrites_the_guard_instead_of_killing_the_export(self):
        """A form whose evaluation fails with a torch exception also goes to replacement (the hole
        ADR 0044 was meant to close).

        Catching only `NotImplementedError` would send this form alone back to "unprovable =
        cannot export". The stats key is counted separately from forms the structure could not
        prove (`rewritten-safe`) — so that no path silently absorbs exceptions.
        """
        decomposed = decompose(
            GatheredMaskAttention(), ATTENTION_ARGS, dynamic_shapes=SYMBOLIC_LENGTH_SHAPES
        )

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:rewritten-unproven"] == 1
        assert "softmax_guard:rewritten-safe" not in stats
        assert _guard_ops_gone(decomposed)
        assert [node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed)] == [True]

    def test_the_probe_failure_keeps_the_original_exception_as_its_cause(self):
        """The normalized verdict always chains the cause (keeps a real implementation bug
        traceable)."""
        decomposed = decompose(
            GatheredMaskAttention(), ATTENTION_ARGS, dynamic_shapes=SYMBOLIC_LENGTH_SHAPES
        )
        guard = next(
            node
            for node in decomposed.graph_module.graph.nodes
            if node.op == "call_function" and node.target is aten.where.self
        )
        _, src = normalize._guard_parts(guard)

        with pytest.raises(
            NotImplementedError, match="cannot decide whether the safe-softmax guard is inactive"
        ) as err:
            normalize._assert_guard_inactive(src, normalize._Placeholders(decomposed))

        assert isinstance(err.value.__cause__, IndexError)

    def test_a_finite_buffer_mask_still_lets_the_guard_go(self):
        """A buffer mask with finite actual values can still be removed as before (the verdict does
        not over-detect)."""
        decomposed = decompose(BufferMaskedAttention(torch.zeros(1, 1, 4, 4)), ATTENTION_ARGS)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:no-neg-inf"] == 1
        assert not any(node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed))

    @pytest.mark.parametrize(
        ("module", "args"),
        [
            pytest.param(LogScoreAttention(), ATTENTION_ARGS, id="log"),
            pytest.param(
                DividedScoreAttention(),
                (*ATTENTION_ARGS, torch.randn(1, 1, 4, 3)),
                id="div-by-tensor",
            ),
        ],
    )
    def test_an_op_that_produces_neg_inf_keeps_the_guard(self, module, args):
        """Ops that **produce** -inf (`log` / a `div` whose denominator cannot be shown non-zero)
        count as sources.

        They have no -inf literal among their arguments, so the constant sweep cannot pick them
        up. Leaving them out of the sources would judge the dependency cone "finite", **remove**
        the guard and let the NaN of an all -inf row flow downstream (ADR 0044 decision 3 = an all
        -inf row into a plain softmax violates the contract). The safe side = branch ②.
        """
        decomposed = decompose(module, args)

        stats = normalize_graph(decomposed)

        assert "softmax_guard:no-neg-inf" not in stats
        assert [node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed)] == [True]

    def test_a_division_by_a_nonzero_literal_is_not_a_neg_inf_source(self):
        """A division by a non-zero literal denominator is not a source (`scores / sqrt(d)` is the
        ordinary path).

        Counting by op name alone, without looking at the divisor's value, would send practically
        every attention to branch ② (= exactly the proposal to drop branch ①). This test and
        {@link test_a_finite_buffer_mask_still_lets_the_guard_go} are the detectors of
        over-detection.
        """

        class ScaledScoreAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("bias", torch.zeros(1, 1, 4, 4))

            def forward(self, q, k, v):
                return nn.functional.scaled_dot_product_attention(
                    q / 8.0, k, v, attn_mask=self.bias
                )

        decomposed = decompose(ScaledScoreAttention(), ATTENTION_ARGS)

        stats = normalize_graph(decomposed)

        assert stats["softmax_guard:no-neg-inf"] == 1
        assert not any(node.meta.get(SAFE_SOFTMAX_META) for node in _softmax_nodes(decomposed))


class IdentityPermute(nn.Module):
    """A lone permute that does not change the axis order."""

    def forward(self, x):
        return x.permute(0, 1, 2, 3)


class InversePermuteChain(nn.Module):
    """An identity chain applying the same transpose twice."""

    def forward(self, x):
        first = x.permute(0, 2, 1, 3)
        return first.permute(0, 2, 1, 3)


class NonIdentityPermuteChain(nn.Module):
    """Adjacent permutes that still need one materialization after composition."""

    def forward(self, x):
        first = x.permute(0, 1, 3, 2)
        return first.permute(0, 2, 1, 3)


class ThreeStagePermuteChain(nn.Module):
    """A three-stage chain (one sweep in topological order folds it into one node)."""

    def forward(self, x):
        return x.permute(0, 1, 3, 2).permute(0, 2, 1, 3).permute(1, 0, 2, 3)


class SharedIntermediatePermute(nn.Module):
    """A chain whose intermediate permute has another consumer (after composition the intermediate
    stays and the meaning does not change)."""

    def forward(self, x):
        first = x.permute(0, 1, 3, 2)
        return first.permute(0, 2, 1, 3), first.sin()


class TestComposePermuteChains:
    def test_single_identity_permute_disappears(self):
        graph, _ = export_and_convert(IdentityPermute(), (torch.randn(2, 3, 4, 5),))

        assert node_ops(graph) == []
        assert graph.outputs == ["x"]

    def test_single_identity_permute_is_reported(self):
        decomposed = decompose(IdentityPermute(), (torch.randn(2, 3, 4, 5),))

        stats = normalize_graph(decomposed)

        assert stats["permute:identity"] == 1

    def test_inverse_pair_disappears(self):
        graph, _ = export_and_convert(InversePermuteChain(), (torch.randn(2, 3, 4, 5),))

        assert node_ops(graph) == []
        assert graph.outputs == ["x"]

    def test_identity_composition_is_reported(self):
        decomposed = decompose(InversePermuteChain(), (torch.randn(2, 3, 4, 5),))

        stats = normalize_graph(decomposed)

        assert stats["permute_chain:identity"] == 1

    def test_non_identity_pair_becomes_one_permute(self):
        graph, _ = export_and_convert(NonIdentityPermuteChain(), (torch.randn(2, 3, 4, 5),))

        assert node_ops(graph) == ["permute"]
        assert only_node(graph, "permute").attrs == {"dims": [0, 3, 1, 2]}

    def test_three_stage_chain_folds_in_one_pass(self):
        """Three or more stages become one node in one pass (they fold from upstream in topological
        order)."""
        args = (torch.randn(2, 3, 4, 5),)
        decomposed = decompose(ThreeStagePermuteChain(), args)

        stats = normalize_graph(decomposed)

        assert stats["permute_chain:composed"] == 2, "two compositions happen within one pass"
        graph, _ = export_and_convert(ThreeStagePermuteChain(), args)
        assert node_ops(graph) == ["permute"]
        # p1=[0,1,3,2] → p2=[0,2,1,3] → p3=[1,0,2,3] composes to [3,0,1,2]
        assert only_node(graph, "permute").attrs == {"dims": [3, 0, 1, 2]}

    def test_shared_intermediate_keeps_its_own_consumer(self):
        """Another consumer of the intermediate does not change the meaning (only the downstream
        input is rewired).

        Mixing up the composition order (`q[p[i]]`) makes this form drift **in values only, with
        the shapes still matching**.
        """
        module = SharedIntermediatePermute()
        args = (torch.randn(2, 3, 4, 5),)
        expected = module(*args)
        decomposed = decompose(module, args)

        stats = normalize_graph(decomposed)

        assert stats["permute_chain:composed"] == 1
        actual = decomposed.module()(*args)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        permutes = [
            node
            for node in decomposed.graph_module.graph.nodes
            if node.op == "call_function" and node.target is aten.permute.default
        ]
        assert len(permutes) == 2, "the intermediate permute survives because its consumer remains"
        assert [node.args[1] for node in permutes] == [[0, 1, 3, 2], [0, 3, 1, 2]]
        assert permutes[1].args[0] is permutes[0].args[0], "the downstream reads base directly"


class RepeatKv(nn.Module):
    """GQA's kv duplication (the only form where `unsqueeze → expand → view` creates rank 5)."""

    def forward(self, x):
        batch, groups, length, head = x.shape
        wide = x[:, :, None, :, :].expand(batch, groups, 3, length, head)
        return wide.reshape(batch, groups * 3, length, head)


class UnitRepeatKv(nn.Module):
    """kv duplication with a count of 1 (no GQA = an MHA configuration) — the whole chain is an
    identity."""

    def forward(self, x):
        batch, groups, length, head = x.shape
        wide = x[:, :, None, :, :].expand(batch, groups, 1, length, head)
        return wide.reshape(batch, groups, length, head)


class SymbolicRepeatKv(nn.Module):
    """kv duplication whose count is **a symbolic dimension** (the length of the duplication axis is
    known only at run time)."""

    def forward(self, x, reps):
        batch, groups, length, head = x.shape
        count = reps.shape[0]
        wide = x[:, :, None, :, :].expand(batch, groups, count, length, head)
        return wide.reshape(batch, groups * count, length, head)


class RopeUnbind(nn.Module):
    """The RoPE split of `use_real_unbind_dim=-2` (a width-1 slice of rank 5 + squeeze)."""

    def forward(self, x):
        batch, heads, length, dim = x.shape
        split = x.reshape(batch, heads, length, 2, dim // 2)
        first = split[..., 0:1, :].squeeze(-2)
        second = split[..., 1:2, :].squeeze(-2)
        return torch.cat([-second, first], dim=-1)


class SymbolicRopeUnbind(nn.Module):
    """A rank-5 slice whose extraction position is **symbolic** (the range ends arrive as FX
    Nodes)."""

    def forward(self, x, idx):
        batch, heads, length, dim = x.shape
        split = x.reshape(batch, heads, length, 2, dim // 2)
        start = idx.shape[0] - 2
        first = split[..., start : start + 1, :].squeeze(-2)
        second = split[..., 0:1, :].squeeze(-2)
        return torch.cat([-second, first], dim=-1)


class Rank5NotUnbind(nn.Module):
    """A rank-5 view that splits the last dimension, but whose consumer is not an unbind (reshape →
    mul → reshape).

    `_lower_split_unbind` grabs on the entry form alone (a view splitting the last dimension into
    (A,B)), so it grabs this form once too. The correct behavior is to **skip without rewriting**
    as soon as it turns out not to be an unbind.
    """

    def forward(self, x):
        batch, heads, length, dim = x.shape
        split = x.reshape(batch, heads, length, 2, dim // 2)
        return (split * 2.0).reshape(batch, heads, length, dim)


class RopeUnbindSqueezing(nn.Module):
    """The same split as RoPE (a rank-5 view splitting the last dimension into (2, D/2) → a width-1
    slice of the split axis), followed by a `squeeze` whose dropped axis can be chosen.

    `axis=3` is a real unbind dropping the split axis; `axis=0` drops another length-1 axis
    (batch), and since the width-1 axis is in a different position, it is not an unbind.
    """

    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x):
        batch, length, heads, dim = x.shape
        split = x.reshape(batch, length, heads, 2, dim // 2)
        first = split[:, :, :, 0:1, :].squeeze(self.axis)
        second = split[:, :, :, 1:2, :].squeeze(self.axis)
        return torch.cat([-second, first], dim=-1)


class Patchify(nn.Module):
    """patchify (rank-8 reshape → permute → reshape)."""

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
        """Rank lowering is an equivalent rewrite — runs the rewritten graph and compares it with
        eager."""
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
        """The aim is to fit within rank ≤ STRIDED_RANK (otherwise convert fails)."""
        graph, _ = export_and_convert(module, args)

        assert max(len(value.shape) for value in graph.values.values()) <= STRIDED_RANK

    def test_a_non_unbind_consumer_is_skipped_not_rejected(self):
        """A rank-5 view that is not an unbind **passes through** (not a single node of the graph
        changes before the match completes).

        Failing here would kill, partway through normalization, even forms that other passes can
        bring down to rank ≤ 4. If a form that really cannot be lowered remains, the check on the
        convert side (the rank limit of the strided family) fails it in the full enumeration.
        """
        args = (torch.randn(1, 2, 4, 6),)
        decomposed = decompose(Rank5NotUnbind(), args)

        stats = normalize_graph(decomposed)

        assert "split_unbind->slice" not in stats
        with torch.no_grad():
            assert torch.equal(decomposed.module()(*args), Rank5NotUnbind()(*args))
        # It also converts (the rank limit applies only to the strided copy family; elementwise
        # is free of rank)
        graph, _ = export_and_convert(Rank5NotUnbind(), args)
        assert node_ops(graph) == ["reshape", "mul", "reshape"]

    @pytest.mark.parametrize(
        ("axis", "expected"),
        [pytest.param(3, 2, id="split axis"), pytest.param(0, 0, id="batch axis")],
    )
    def test_only_a_squeeze_of_the_split_axis_is_folded(self, axis, expected):
        """Only a squeeze that drops **the split axis** is folded — a squeeze dropping another
        length-1 axis passes through.

        Matching on op kinds alone would also grab the form dropping the batch axis, and the
        replacing slice (`[1,T,H,64]`) would disagree with the original squeeze's output
        (`[T,H,1,64]`).
        """
        module = RopeUnbindSqueezing(axis)
        args = (torch.randn(1, 3, 2, 8),)
        decomposed = decompose(module, args)

        stats = normalize_graph(decomposed)

        assert stats.get("split_unbind->slice", 0) == expected
        with torch.no_grad():
            assert torch.equal(decomposed.module()(*args), module(*args))

    def test_a_unit_replication_collapses_the_whole_chain(self):
        """A chain with a duplication count of 1 (no GQA = an MHA configuration) disappears as a
        whole identity.

        Getting it wrong lets "a graph that dropped one axis" through the shape checks (the last
        line of defense is only the `extent_keys` match check).
        """
        args = (torch.randn(1, 2, 4, 3),)
        decomposed = decompose(UnitRepeatKv(), args)

        stats = normalize_graph(decomposed)

        assert stats["unit_expand->identity"] == 1
        with torch.no_grad():
            assert torch.equal(decomposed.module()(*args), UnitRepeatKv()(*args))
        graph, _ = export_and_convert(UnitRepeatKv(), args)
        # The whole chain disappears, so zero nodes remain (the output is the input itself).
        assert node_ops(graph) == []
        assert graph.outputs == ["x"]

    def test_a_symbolic_replication_count_is_rejected(self, dyn_t):
        """Rank 5 with a symbolic duplication count cannot be represented — fails instead of
        specializing to the hint value."""
        args = (torch.randn(1, 2, 4, 3), torch.randn(3))
        decomposed = decompose(SymbolicRepeatKv(), args, dynamic_shapes=(None, {0: dyn_t}))

        with pytest.raises(NotImplementedError, match="duplication along a symbolic dimension"):
            normalize_graph(decomposed)

    def test_a_symbolic_length_still_lowers_the_gqa_form(self, dyn_t):
        """A real repeat_kv lowers to rank 3 even when axes other than the duplication axis are
        symbolic.

        The accepting side of the reason shapes are compared with `extent_keys` (structural
        equality of expressions) — a plain list `!=` resolves SymInt's `__eq__` by the hint value,
        and the evaluation itself adds an after-the-fact guard substitution to the exported
        ShapeEnv (the substitution reaches `convert._dim`).
        """
        decomposed = decompose(RepeatKv(), (torch.randn(1, 2, 4, 3),), dynamic_shapes=({2: dyn_t},))

        stats = normalize_graph(decomposed)
        graph, _ = convert(decomposed)

        assert stats["unit_expand->rank3"] == 1
        assert node_ops(graph) == ["reshape", "expand", "reshape"]
        assert graph.values[graph.outputs[0]].shape == [1, 6, "T", 3]

    def test_a_symbolic_slice_bound_is_skipped_not_crashed(self):
        """A rank-5 slice with symbolic range ends is **skipped untouched** (this function's MUST).

        The ends of `aten.slice.Tensor(…, SymInt? start, SymInt? end, …)` arrive as FX Nodes when
        symbolic. Applying a plain `int()` gives a bare `TypeError` crash instead of the
        diagnostic (the full enumeration of unsupported ops) — specializing to the hint value is
        equally not allowed, so the grabbed form is not treated as an unbind and is left to the
        other passes and convert's rank check.
        """
        args = (torch.randn(1, 2, 4, 6), torch.randn(2))
        decomposed = decompose(
            SymbolicRopeUnbind(), args, dynamic_shapes=(None, {0: Dim("I", min=2, max=3)})
        )

        stats = normalize_graph(decomposed)

        assert "split_unbind->slice" not in stats

    def test_a_rank4_graph_is_left_untouched(self):
        """The firing condition is rank > STRIDED_RANK — no misfire on existing graphs (rank ≤
        4)."""

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

    def test_selecting_a_longer_axis_becomes_slice_and_reshape(self):
        """A select on a static axis of length 2 or more becomes `slice(width 1) + squeeze` (first
        seen in Gemma4's per-layer PLE slices; the existing reshape normalization handles the
        squeeze)."""

        class SelectRow(nn.Module):
            def forward(self, x):
                return torch.select(x, 1, 2) * 2.0

        graph, _ = export_and_convert(SelectRow(), (torch.randn(2, 3, 4),))

        assert node_ops(graph) == ["slice", "reshape", "mul"]
        assert only_node(graph, "slice").attrs == {"dim": 1, "start": 2, "end": 3}
        assert graph.values[graph.nodes[1].outs[0]].shape == [2, 4]

    def test_a_negative_select_index_is_normalized(self):
        """A negative index becomes a positive width-1 range per torch's select semantics (-size ≤
        index < size)."""

        class SelectLast(nn.Module):
            def forward(self, x):
                return torch.select(x, 1, -1) * 2.0

        graph, _ = export_and_convert(SelectLast(), (torch.randn(2, 3, 4),))

        assert only_node(graph, "slice").attrs == {"dim": 1, "start": 2, "end": 3}

    def test_selecting_a_symbolic_axis_is_rejected(self, dyn_t):
        """Taking the length of a symbolic axis from its hint value gives a silent wrong value —
        stopped here."""

        class SelectSymbolic(nn.Module):
            def forward(self, x):
                return torch.select(x, 0, 0) * 2.0

        decomposed = torch.export.export(
            SelectSymbolic(), (torch.randn(4, 3),), dynamic_shapes=({0: dyn_t},), strict=False
        ).run_decompositions(curated_decompositions())

        with pytest.raises(NotImplementedError, match="select on an axis of symbolic dimension"):
            normalize_graph(decomposed)

    def test_selecting_with_a_symbolic_index_is_rejected(self):
        """A select with a symbolic **index** also fails (even with a static axis length, the path
        to a silent wrong value is the same).

        Taking the index of `aten.select.int(self, int dim, SymInt index)` from the hint value
        bakes `slice(hint, hint+1)`, which silently reads a different row under another binding. A
        symbolic value arrives not as a raw `SymInt` but as **an FX Node** (`sub` here), so a check
        that looks only for `torch.SymInt` lets it through.
        """

        class SelectSymbolicIndex(nn.Module):
            def forward(self, x, idx):
                return torch.select(x, 1, idx.shape[0] - 1) * 2.0

        decomposed = torch.export.export(
            SelectSymbolicIndex(),
            (torch.randn(2, 8, 4), torch.randn(3)),
            dynamic_shapes=(None, {0: Dim("I", min=2, max=8)}),
            strict=False,
        ).run_decompositions(curated_decompositions())

        with pytest.raises(NotImplementedError, match="select with a symbolic index"):
            normalize_graph(decomposed)


class KeptSum(nn.Module):
    """The MoE router's renormalisation `w / w.sum(-1, keepdim=True)` (ADR 0145 decision 5)."""

    def forward(self, w):
        return w / w.sum(dim=-1, keepdim=True)


class TestSumKeepdimToUnsqueeze:
    def test_a_one_axis_keepdim_sum_becomes_sum_and_reshape(self, dyn_t):
        """The reduced value is `[T]` and the kept one `[T, 1]` — the leading axis stays
        symbolic."""
        graph, _ = export_and_convert(KeptSum(), (torch.rand(6, 4),), ({0: dyn_t},))

        assert node_ops(graph) == ["sum", "reshape", "div"]
        reduced = only_node(graph, "sum")
        assert reduced.attrs == {"dim": 1}
        assert graph.values[reduced.outs[0]].shape == ["T"]
        assert graph.values[only_node(graph, "reshape").outs[0]].shape == ["T", 1]

    def test_a_middle_axis_keeps_its_place(self):
        """The unit axis goes back where the reduced one was (`[2, 3, 4]` over axis 1 → `[2, 1,
        4]`)."""

        class MiddleSum(nn.Module):
            def forward(self, x):
                return x.sum(dim=-2, keepdim=True) * x

        graph, _ = export_and_convert(MiddleSum(), (torch.rand(2, 3, 4),))

        assert only_node(graph, "sum").attrs == {"dim": 1}
        assert graph.values[only_node(graph, "reshape").outs[0]].shape == [2, 1, 4]

    def test_the_rewrite_computes_the_same_values(self):
        """The rewritten FX graph run eagerly equals the kept sum (the IR reshape takes its shape
        from the meta, so a unit axis put back in the wrong place shows only here)."""
        w = torch.rand(3, 4)
        decomposed = decompose(KeptSum(), (w,))
        normalize_graph(decomposed)

        torch.testing.assert_close(decomposed.module()(w), KeptSum()(w))

    def test_the_pass_reports_its_firing_count(self):
        decomposed = decompose(KeptSum(), (torch.rand(6, 4),))

        assert normalize_graph(decomposed)["sum_keepdim->sum+unsqueeze"] == 1

    def test_without_the_pass_the_handler_rejects_keepdim(self):
        """Confirms the pass is load-bearing — the reduce contract has no keepdim."""
        decomposed = decompose(KeptSum(), (torch.rand(6, 4),))

        with pytest.raises(NotImplementedError, match="sum with keepdim=True is not supported"):
            convert(decomposed)

    def test_a_sum_without_keepdim_is_untouched(self):
        class PlainSum(nn.Module):
            def forward(self, x):
                return x.sum(dim=-1)

        graph, _ = export_and_convert(PlainSum(), (torch.rand(6, 4),))

        assert node_ops(graph) == ["sum"]

    @pytest.mark.parametrize(
        ("build", "why"),
        [
            (lambda x: x.sum(dim=(0, 1), keepdim=True), "multiple axes"),
            (
                lambda x: x.sum(dim=-1, keepdim=True, dtype=torch.float32),
                "sum with keepdim=True is not supported",
            ),
        ],
        ids=["multi-axis", "dtype"],
    )
    def test_other_keepdim_forms_still_reach_the_handler_refusal(self, build, why):
        class Other(nn.Module):
            def forward(self, x):
                return build(x)

        with pytest.raises(NotImplementedError, match=why):
            export_and_convert(Other(), (torch.rand(6, 4),))


class PerExpertScale(nn.Module):
    """The MoE router's `per_expert_scale[top_k_index]` (ADR 0145 decision 5): a rank-1 parameter
    indexed by a run-time index."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1.0, 7.0))

    def forward(self, w, index):
        return w * self.scale[index]


class TestIndexTableToEmbedding:
    def test_a_rank1_parameter_indexed_at_run_time_becomes_an_embedding(self, dyn_t):
        """`[E]` → reshape `[E, 1]` → embedding `[T, k, 1]` → reshape `[T, k]` (the index's shape,
        symbolic leading axis kept); the i64 index is declared i32 (ADR 0009)."""
        graph, _ = export_and_convert(
            PerExpertScale(),
            (torch.rand(6, 2), torch.tensor([[0, 5]] * 6)),
            ({0: dyn_t}, {0: dyn_t}),
        )

        assert node_ops(graph) == ["reshape", "embedding", "reshape", "mul"]
        rows, picked, back = graph.nodes[:3]
        assert rows.ins == ["p_scale"]
        assert graph.values[rows.outs[0]].shape == [6, 1]
        assert picked.ins == [rows.outs[0], "index"]
        assert graph.values[picked.outs[0]].shape == ["T", 2, 1]
        assert graph.values[back.outs[0]].shape == ["T", 2]
        assert next(spec for spec in graph.inputs if spec.name == "index").dtype == "i32"

    def test_the_rewrite_computes_the_same_values(self):
        """The rewritten FX graph run eagerly equals `table[index]` (a swapped table / index or a
        wrong unit axis would not)."""
        module = PerExpertScale()
        w, index = torch.rand(3, 2), torch.tensor([[4, 1], [0, 5], [2, 2]])
        decomposed = decompose(module, (w, index))
        normalize_graph(decomposed)

        torch.testing.assert_close(decomposed.module()(w, index), module(w, index))

    def test_the_pass_reports_its_firing_count(self):
        decomposed = decompose(PerExpertScale(), (torch.rand(3, 2), torch.tensor([[4, 1]] * 3)))

        assert normalize_graph(decomposed)["index_table->embedding"] == 1

    def test_without_the_pass_index_is_outside_the_vocabulary(self):
        decomposed = decompose(PerExpertScale(), (torch.rand(3, 2), torch.tensor([[4, 1]] * 3)))

        with pytest.raises(UnsupportedAtenOpsError) as err:
            convert(decomposed)

        assert "aten.index.Tensor" in err.value.ops

    def test_a_buffer_table_is_rewritten_too(self):
        class BufferTable(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.arange(1.0, 7.0))

            def forward(self, index):
                return self.table[index]

        graph, _ = export_and_convert(BufferTable(), (torch.tensor([3, 1]),))

        assert node_ops(graph) == ["reshape", "embedding", "reshape"]


class _RuntimeTable(nn.Module):
    def forward(self, table, index):
        return table[index]


class _TwoIndices(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.rand(6, 3))

    def forward(self, rows, cols):
        return self.table[rows, cols]


class _NoneSlot(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.rand(3, 6))

    def forward(self, index):
        return self.table[:, index]


class _Rank2Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.rand(6, 3))

    def forward(self, index):
        return self.table[index]


class _MaskIndex(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.rand(6))

    def forward(self, mask):
        return self.table[mask].sum(dim=0, keepdim=True)


class _LiftedConstantTable(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.arange(1.0, 7.0)

    def forward(self, index):
        return self.table[index]


class TestIndexTableOtherForms:
    """Every other `aten.index.Tensor` form is left alone and stays in the full enumeration of
    unsupported ops (the rewrite is not widened — ADR 0145 decision 5)."""

    @pytest.mark.parametrize(
        ("module", "args"),
        [
            (_RuntimeTable(), (torch.rand(6), torch.tensor([3, 1]))),
            (_TwoIndices(), (torch.tensor([3, 1]), torch.tensor([0, 2]))),
            (_NoneSlot(), (torch.tensor([3, 1]),)),
            (_Rank2Table(), (torch.tensor([3, 1]),)),
            (_MaskIndex(), (torch.tensor([True, False, True, True, False, True]),)),
            (_LiftedConstantTable(), (torch.tensor([3, 1]),)),
        ],
        ids=["runtime-table", "two-indices", "none-slot", "rank2-table", "mask", "lifted-constant"],
    )
    def test_the_form_is_refused_as_an_unsupported_op(self, module, args):
        with pytest.raises(UnsupportedAtenOpsError) as err:
            export_and_convert(module, args)

        assert "aten.index.Tensor" in err.value.ops
