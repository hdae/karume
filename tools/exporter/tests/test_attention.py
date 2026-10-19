"""The exporter-side checks of fused attention (ADR 0023 + 0067 decision 1).

Six things are checked:

1. **Preserving SDPA is per target** (the default decomposition table keeps its 12 ops).
2. `_h_attention` **fails loudly on every case** of causal / dropout≠0 / rank≠4 / D mismatch and
   of **masks outside the contract** (bool / rank≠4 / `[B,1,M,N]` / `[1,H,M,N]` / M or N
   mismatch).
3. The `scale` in attrs **matches the torch math decomp constant bit for bit** (the half-scale
   contract).
4. The IR of a preserved graph has exactly one `attention` node.
5. `enable_gqa=True` **passes as a preserved node** (GQA / MQA — ADR 0067 decision 1; the shapes
   layer checks the shape validity).
6. Folding a bool mask into an additive f32 mask (`normalize._additive_attn_mask`) **matches the
   constant the decomposition path bakes byte for byte**.
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

#: The observed shape with D=128 (DiT self / cross attention). Used to check the half scale when
#: scale is omitted.
DIT_DEPTH = 128
#: The oracle the design recon read out of the real IR (`const.6953fe58410d6c34`).
DIT_HALF_SCALE = 0.2973017692565918
#: A dynamic axis that makes D symbolic (to check that the default scale cannot be decided at
#: export time).
DYN_DEPTH = Dim("D", min=2, max=16)


class PlainAttention(nn.Module):
    """SDPA without a mask, non-causal, dropout 0 (exactly what fusion targets)."""

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
    """Holds the mask as **a lifted constant** (a plain attribute, neither a buffer nor a
    parameter).

    Only this shape qualifies as a leaf of constant folding, and it stands on the same ground as
    the path by which Gemma3's band mask (a constant that depends only on T) becomes
    `sym_prefix_slice` in the IR.
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
    """SDPA with `enable_gqa=True` (KV heads broadcast to the q heads by an exact division)."""

    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)


class Rank3Attention(nn.Module):
    def forward(self, q, k, v):
        return nn.functional.scaled_dot_product_attention(q, k, v)


def _args(batch=2, heads=3, queries=5, keys=7, depth=4, kv_heads=None):
    """The q / k / v arguments (with `kv_heads` omitted, the classic form H = Hkv)."""
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
        """MUST: the default decomposition table keeps its 12 ops.

        Widening it globally to 13 ops would make a graph with a masked SDPA (the causal mask with
        −inf folded in, of the Anima text_encoder) hit `_h_attention`'s fail loudly and **become
        impossible to export**. ADR 0016's safe-softmax guard removal pass also depends on that
        decomposition path.
        """
        assert ATTENTION_OP_PREFIX not in PRESERVED_OP_PREFIXES
        assert (
            *PRESERVED_OP_PREFIXES,
            ATTENTION_OP_PREFIX,
        ) == PRESERVED_OP_PREFIXES_WITH_ATTENTION

    def test_the_default_table_still_decomposes_sdpa(self):
        """By default it is decomposed as before (passes with the vocabulary bmm → softmax →
        bmm)."""
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
        assert preserved_names, "the default table has no SDPA entry (the premise is broken)"
        assert not any(str(overload).startswith(ATTENTION_OP_PREFIX) for overload in table_with)

    def test_a_masked_attention_still_exports_under_the_default_table(self):
        """MUST: a target that does not preserve passes as before (the proof that ADR 0016 is kept).

        This is the shape the text_encoder really has (a bias with −inf folded into `attn_mask`),
        and including it in the preserved set would **make export impossible** through
        `_h_attention`'s fail loudly.
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
        # Run the contract (arity / attrs keys / value ranges) through the Python-side table too
        assert assert_node_contract(node, "test") is OP_CONTRACTS["attention"]

    def test_the_declared_shapes_are_head_first_rank4(self):
        graph, _ = _preserved_attention(PlainAttention(), _args())
        node = only_node(graph, "attention")

        shapes = [declared_shape(graph, name) for name in node.ins]
        assert shapes == [[2, 3, 5, 4], [2, 3, 7, 4], [2, 3, 7, 4]]
        assert declared_shape(graph, node.outs[0]) == [2, 3, 5, 4]

    def test_the_decomposition_debris_is_gone(self):
        """Preserving removes the identity expand / scale mul / kᵀ permute that come from the decomp
        from the IR altogether."""
        graph, _ = _preserved_attention(PlainAttention(), _args())

        ops = [node.op for node in graph.nodes]
        for debris in ("expand", "mul", "permute", "bmm", "softmax", "reshape"):
            assert debris not in ops


class TestHalfScaleIsBitExact:
    def test_the_default_scale_matches_the_torch_math_decomp_constant(self):
        """MUST: `f32(√(1/√D))` **matches the `mul` constant of the torch decomp bit for bit**.

        The oracle is `const.6953fe58410d6c34` = 0.2973017692565918 of the real DiT IR
        (= f32(128^-0.25)). Changing the half scale into the full scale (`1/√D`), or moving the
        `√`, turns this red first.
        """
        graph, _ = _preserved_attention(
            PlainAttention(), _args(batch=1, heads=2, queries=4, keys=4, depth=DIT_DEPTH)
        )
        scale = only_node(graph, "attention").attrs["scale"]

        assert scale == DIT_HALF_SCALE
        # Also equal as an f32 value (the JSON round trip and f32 rounding do not move an ulp)
        assert torch.tensor(scale, dtype=torch.float32).item() == DIT_HALF_SCALE
        # Pin explicitly that it is **different** from the full-scale value (applied once after
        # the dot product)
        assert scale != pytest.approx(1.0 / math.sqrt(DIT_DEPTH))
        assert scale * scale == pytest.approx(1.0 / math.sqrt(DIT_DEPTH), rel=1e-6)

    def test_the_torch_decomposition_emits_the_same_constant(self):
        """Confirms by observation that the torch decomp really puts the same constant on its 2
        `mul`s."""
        args = _args(batch=1, heads=2, queries=4, keys=4, depth=DIT_DEPTH)
        exported = torch.export.export(PlainAttention(), args, strict=False)
        decomposed = exported.run_decompositions(curated_decompositions())

        scalars = [
            node.args[1]
            for node in decomposed.graph.nodes
            if node.op == "call_function" and "mul" in str(node.target)
        ]
        assert len(scalars) == 2, f"the scale muls are not 2: {scalars}"
        assert scalars[0] == scalars[1]
        assert torch.tensor(scalars[0], dtype=torch.float32).item() == DIT_HALF_SCALE

    def test_an_explicit_scale_is_carried_through_as_its_square_root(self):
        graph, _ = _preserved_attention(PlainAttention(scale=0.25), _args())
        scale = only_node(graph, "attention").attrs["scale"]

        assert scale == pytest.approx(0.5)


class TestUnsupportedFormsFailLoudly:
    """MUST: shapes outside the vocabulary **are all enumerated and rejected** (no silent
    approximation — a cross-cutting invariant)."""

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
        """Only v has a different D (torch allows it, but the contract requires the same D for all
        3)."""
        args = (
            torch.randn(2, 3, 5, 4),
            torch.randn(2, 3, 7, 4),
            torch.randn(2, 3, 7, 6),
        )
        with pytest.raises(NotImplementedError, match="D"):
            _preserved_attention(PlainAttention(), args)

    def test_a_symbolic_depth_is_rejected(self):
        """With D a symbolic dimension, `√scale_factor` cannot be decided at export time (a premise
        of the default scale)."""
        args = _args()
        with pytest.raises(NotImplementedError, match="symbolic dimension"):
            export_and_convert(
                PlainAttention(),
                args,
                dynamic_shapes=({3: DYN_DEPTH}, {3: DYN_DEPTH}, {3: DYN_DEPTH}),
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )


class TestGqaIsAcceptedAsDivisibleBroadcast:
    """`enable_gqa=True` passes as a preserved node (the divisible broadcast of ADR 0067 decision
    1).

    The handler **does not read** `enable_gqa` — the shape validity (`H % Hkv == 0` and
    `H ≥ Hkv ≥ 1`, Hkv equal between k / v) is checked in one place, the shapes layer (the
    `assert_graph_shapes` at the exit of convert always runs every node), and checking twice would
    split the accepted set into 2 places. **A non-divisible shape cannot be exercised here** —
    torch itself fails with "Number of heads in key and value must divide the number of heads in
    query", so the rejecting side is covered by test_shapes.py and the conformance table
    (op-contracts.json).

    GQA on the decomposition path (the default table) is not rescued (second half of ADR 0067
    decision 1 — the batch match of `bmm` is a deliberate detection line). There repeat_kv is
    materialised by expand / reshape as before.
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
        # The output has **the q-side H** (it does not shrink to the kv side)
        assert declared_shape(graph, node.outs[0]) == [2, 4, 5, 4]
        # Run the contract (arity / attrs keys / value ranges) through the Python-side table too
        assert assert_node_contract(node, "test") is OP_CONTRACTS["attention"]

    def test_the_mqa_form_carries_the_single_kv_head(self):
        """Hkv=1 is expressed by the same division — no separate op.

        The shape is exercised with the same ratio and the same D as the MQA case of the
        conformance table (8:1, head_dim 256).
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
        """With r=1 the IR does not change with or without `enable_gqa` (the intent of not having a
        ratio field)."""
        args = _args()
        gqa, _ = _preserved_attention(GqaAttention(), args)
        plain, _ = _preserved_attention(PlainAttention(), args)

        assert [node.op for node in gqa.nodes] == ["attention"]
        assert only_node(gqa, "attention") == only_node(plain, "attention")
        assert gqa.values == plain.values


class TestAdditiveMaskIsAccepted:
    """Only the contract's mask (additive f32, rank-4, exactly `[1,1,M,N]`) becomes the fourth
    input."""

    def test_the_mask_becomes_a_fourth_input(self):
        args = (*_args(), torch.zeros(1, 1, 5, 7))
        graph, _ = _preserved_attention(MaskedAttention(), args)

        node = only_node(graph, "attention")
        assert len(node.ins) == 4
        assert declared_shape(graph, node.ins[3]) == [1, 1, 5, 7]
        assert declared_shape(graph, node.outs[0]) == [2, 3, 5, 4]
        # Run the contract (arity 3 or 4 / attrs keys / value ranges) through the Python-side
        # table too
        assert assert_node_contract(node, "test") is OP_CONTRACTS["attention"]

    def test_the_maskless_form_keeps_arity_three(self):
        """MUST: the path without a mask does not change by a single byte (the meaning of an
        optional slot)."""
        graph, _ = _preserved_attention(PlainAttention(), _args())

        assert len(only_node(graph, "attention").ins) == 3

    def test_a_symbolic_mask_length_is_matched_symbolically(self):
        """Even when M / N are symbolic, they are matched against q / k as **the same symbol**."""
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
    """MUST: masks outside the contract **are all enumerated and rejected** (the absence of a field
    means "not in the vocabulary")."""

    @pytest.mark.parametrize(
        ("shape", "why"),
        [
            ((2, 3, 5, 7), "has both B and H"),
            ((2, 1, 5, 7), "[B,1,M,N] (per-batch mask)"),
            ((1, 3, 5, 7), "[1,H,M,N] (per-head mask)"),
        ],
        ids=("bh", "batch", "head"),
    )
    def test_a_leading_axis_other_than_one_is_rejected(self, shape, why):
        args = (*_args(), torch.zeros(shape))
        with pytest.raises(NotImplementedError, match="leading 2 axes"):
            _preserved_attention(MaskedAttention(), args)
        assert why  # kept as a parameter so that the shape's intent is readable on failure

    def test_a_broadcast_query_axis_is_rejected(self):
        """`[1,1,1,N]` (the same mask for every query row) passes in torch, but the contract
        requires M to match."""
        args = (*_args(), torch.zeros(1, 1, 1, 7))
        with pytest.raises(NotImplementedError, match="M / N"):
            _preserved_attention(MaskedAttention(), args)

    def test_a_rank3_mask_is_rejected(self):
        args = (*_args(), torch.zeros(1, 5, 7))
        with pytest.raises(NotImplementedError, match=r"attn_mask of rank 3"):
            _preserved_attention(MaskedAttention(), args)

    def test_a_bool_mask_is_rejected_by_the_handler_itself(self):
        """Without the normalization pass the mask arrives as bool, and the handler's dtype check
        rejects it.

        MUST: the check is not placed in the normalization pass only — when the pass does not
        fire (a different shape or the like), the bool mask would silently slide down to a failure
        at initializer conversion.
        """
        args = (*_args(), torch.ones(1, 1, 5, 7, dtype=torch.bool))
        exported = torch.export.export(MaskedAttention(), args, strict=False)
        decomposed = exported.run_decompositions(
            curated_decompositions(PRESERVED_OP_PREFIXES_WITH_ATTENTION)
        )

        with pytest.raises(NotImplementedError, match=r"dtype torch\.bool"):
            convert(decomposed)


class TestBoolMaskFoldsToTheDecompositionConstant:
    """Observation that folding bool → additive f32 gives **the same constant** as torch's own
    conversion.

    ADR 0023's bit identity (the fused path and the decomposition path give the same values)
    depends, on the mask side too, on "the same 0 / −inf f32 constants being added at the same
    positions".
    """

    #: A band mask (True only around the diagonal) — an all-True mask would degenerate the
    #: constant and make the comparison tautological.
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

        assert len(decomposed) == 1, f"the decomposition path has {len(decomposed)} mask constants"
        assert len(preserved) == 1, f"the preserved path has {len(preserved)} mask constants"
        assert decomposed[0].dtype is torch.float32
        assert torch.equal(decomposed[0], preserved[0])
        # Guard against a tautological test: the constant really holds the 2 values 0 and −inf
        # (with all zeros the comparison would be meaningless)
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
