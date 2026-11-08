"""`karume::expert_linear`, its handler, its shape rule and its weights (extended composite layer —
ADR 0145 decisions 2, 3 and 5).

Pinned here: the eager body is upstream's per-expert math for one projection (and the expert FFN
built from two of them follows upstream's experts module), the op exports as one IR node with a
symbolic token axis in both forms of x and survives the write → read back → verify round trip, the
handler narrows arity, rank and dtype, and emit writes a rank-3 i4 / i4f16 fixed weight for an
`expert_linear` (and only for one).

Expected bytes are written out by hand (deriving them from the code under test would make the checks
tautological).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest
import torch
from conftest import node_ops, only_node
from torch import nn

from karume.container import Provenance, container_parts, read_container, write_model_container
from karume.custom_ops import EXPERT_LINEAR_UPSTREAM, expert_linear, expert_linear_body
from karume.emit import EmitError, FixedQuantizedWeight, pack_int4, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.ops import OpContractError
from karume.pipeline import export_to_file
from karume.verify import (
    ContainerError,
    assert_ir_accepted,
    assert_runtime_support,
    verify_container,
)

PROVENANCE = Provenance(license="apache-2.0", writer="karume")

#: E, k, in, out — all distinct, so a swapped axis shows in a shape.
EXPERTS, PICKS, IN, OUT = 6, 2, 64, 48


def _inputs(
    tokens: int = 5, seed: int = 20261108
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [T, in], index [T, k] (distinct ids within a token, repeated across tokens) and the
    weight [E, out, in]."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, IN, generator=generator)
    index = torch.tensor([[(t + 1) % EXPERTS, (t + 4) % EXPERTS] for t in range(tokens)])
    weight = torch.randn(EXPERTS, OUT, IN, generator=generator) * 0.1
    return x, index, weight


class Linear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.ops.karume.expert_linear(x, index, self.weight)


class TestEagerBody:
    def test_each_pick_is_the_linear_of_its_experts_weight(self):
        """y[t, j] = weight[index[t, j]] · x[t] — against F.linear row by row (the hand form)."""
        x, index, weight = _inputs()

        y = expert_linear(x, index, weight)

        assert y.shape == (5, PICKS, OUT)
        for t in range(5):
            for j in range(PICKS):
                expected = nn.functional.linear(x[t], weight[index[t, j]])
                torch.testing.assert_close(y[t, j], expected, rtol=0, atol=1e-6)

    def test_the_per_pick_form_reads_one_row_per_pick(self):
        _, index, weight = _inputs()
        x = torch.randn(5, PICKS, IN, generator=torch.Generator().manual_seed(7))

        y = expert_linear(x, index, weight)

        for t in range(5):
            for j in range(PICKS):
                expected = nn.functional.linear(x[t, j], weight[index[t, j]])
                torch.testing.assert_close(y[t, j], expected, rtol=0, atol=1e-6)

    def test_two_of_them_make_upstreams_expert_ffn(self):
        """gate_up → gelu_tanh(gate) · up → down → × routing weight → sum over k equals upstream's
        `Gemma4TextExperts.forward` (which adds in expert order; this sums in pick order, so the
        comparison is numeric — ADR 0145 decision 2)."""
        modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
        config = modeling.Gemma4TextConfig(
            hidden_size=IN,
            num_experts=EXPERTS,
            moe_intermediate_size=OUT // 2,
            hidden_activation="gelu_pytorch_tanh",
        )
        experts = modeling.Gemma4TextExperts(config)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            experts.gate_up_proj.copy_(torch.randn(EXPERTS, OUT, IN, generator=generator) * 0.1)
            experts.down_proj.copy_(torch.randn(EXPERTS, IN, OUT // 2, generator=generator) * 0.1)
        x, index, _ = _inputs()
        routing = torch.rand(5, PICKS, generator=generator)

        hidden = expert_linear(x, index, experts.gate_up_proj.detach())
        gate, up = hidden.chunk(2, dim=-1)
        mixed = nn.functional.gelu(gate, approximate="tanh") * up
        y = (expert_linear(mixed, index, experts.down_proj.detach()) * routing[..., None]).sum(1)

        with torch.no_grad():
            expected = experts(x, index, routing)
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)

    def test_it_names_the_upstream_it_follows(self):
        assert EXPERT_LINEAR_UPSTREAM.startswith("transformers 5.14.1")

    @pytest.mark.parametrize("bad", [-1, EXPERTS], ids=["negative", "past-E"])
    def test_an_id_out_of_range_raises(self, bad: int):
        """ADR 0061's CPU rule — torch would wrap a negative id to another expert."""
        x, index, weight = _inputs()
        index[2, 1] = bad

        with pytest.raises(ValueError, match="outside the 6 experts"):
            expert_linear(x, index, weight)


class TestSharedBody:
    """`expert_linear_body` — the one per-pick summation of the op and of a reference that holds
    its experts in another form (ADR 0145 decision 5)."""

    def test_the_op_is_the_body_over_the_stacked_weight_bit_for_bit(self):
        x, index, weight = _inputs()

        got = expert_linear_body(x, index, weight.__getitem__, EXPERTS, OUT)

        assert torch.equal(got, expert_linear(x, index, weight))

    def test_a_getter_that_decodes_on_read_gives_the_op_over_the_decoded_weight(self):
        """A reference over another storage (a scale per expert, applied on read) sums exactly as
        the op over the decoded stacked weight."""
        x, index, weight = _inputs()
        scale = torch.linspace(0.5, 1.5, EXPERTS)

        got = expert_linear_body(x, index, lambda e: weight[e] * scale[e], EXPERTS, OUT)

        assert torch.equal(got, expert_linear(x, index, weight * scale[:, None, None]))

    def test_only_the_picked_experts_are_read_once_each_in_id_order(self):
        x, index, weight = _inputs()
        read: list[int] = []

        def weight_of(expert: int) -> torch.Tensor:
            read.append(expert)
            return weight[expert]

        expert_linear_body(x, index, weight_of, EXPERTS, OUT)

        assert read == sorted(set(index.flatten().tolist()))

    def test_an_id_out_of_range_raises_before_any_read(self):
        x, index, weight = _inputs()
        index[0, 0] = EXPERTS
        read: list[int] = []

        with pytest.raises(ValueError, match="outside the 6 experts"):
            expert_linear_body(x, index, lambda e: read.append(e) or weight[e], EXPERTS, OUT)
        assert read == []


class TestExport:
    def test_it_stays_one_node_with_a_symbolic_token_axis(self, convert_module, dyn_t):
        """MUST: T stays symbolic in one node, the index is declared i32 (ADR 0009), attrs are
        empty and the weight is an initializer."""
        x, index, weight = _inputs()
        graph, tensors = convert_module(
            Linear(weight), (x, index), dynamic_shapes=({0: dyn_t}, {0: dyn_t})
        )

        assert node_ops(graph) == ["expert_linear"]
        node = only_node(graph, "expert_linear")
        assert node.attrs == {}
        assert node.ins[:2] == [spec.name for spec in graph.inputs]
        assert [spec.dtype for spec in graph.inputs] == ["f32", "i32"]
        assert graph.values[node.ins[2]].shape == [EXPERTS, OUT, IN]
        assert graph.values[node.outs[0]].shape == ["T", PICKS, OUT]
        assert list(tensors) == ["weight"]

    def test_the_per_pick_form_exports_too(self, convert_module, dyn_t):
        _, index, weight = _inputs()
        x = torch.randn(5, PICKS, IN)
        graph, _ = convert_module(
            Linear(weight), (x, index), dynamic_shapes=({0: dyn_t}, {0: dyn_t})
        )

        node = only_node(graph, "expert_linear")
        assert graph.values[node.outs[0]].shape == ["T", PICKS, OUT]
        assert graph.inputs[0].shape == ["T", PICKS, IN]

    def test_it_round_trips_through_the_container(self, tmp_path: Path):
        final = tmp_path / "model.krm"
        x, index, weight = _inputs()

        export_to_file(
            Linear(weight),
            (x, index),
            final,
            provenance=Provenance(license="mit"),
            graph_name="experts",
            dynamic_shapes=({0: torch.export.Dim("T", min=2, max=64)},) * 2,
        )

        declaration = (
            verify_container(container_parts(final), blocks=True).graphs["experts"].declaration
        )
        assert declaration["requires"]["ops"] == ["expert_linear"]
        assert declaration["symbols"] == ["T"]
        assert declaration["values"][declaration["outputs"][0]]["shape"] == ["T", PICKS, OUT]

    def test_a_non_f32_x_is_rejected(self, convert_module):
        x, index, weight = _inputs()

        with pytest.raises(NotImplementedError, match="f32"):
            convert_module(Linear(weight.double()), (x.double(), index))

    def test_a_rank_2_weight_is_rejected(self, convert_module):
        """A rank-2 weight is linear's; the handler names the rank."""
        x, index, weight = _inputs()

        with pytest.raises(NotImplementedError, match=r"weight of rank 2"):
            convert_module(Linear(weight[0]), (x, index))

    def test_a_rank_1_index_is_rejected(self, convert_module):
        x, index, weight = _inputs()

        class Flat(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(weight)

            def forward(self, x, index):
                return torch.ops.karume.expert_linear(x, index.flatten(), self.weight)

        with pytest.raises(NotImplementedError, match=r"index of rank 1"):
            convert_module(Flat(), (x, index))

    def test_the_shape_rule_rejects_an_input_width_mismatch(self, convert_module):
        """The fake does not check shapes; the shape rule does (`shapes._expert_linear`)."""
        x, index, weight = _inputs()

        class Narrow(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(weight)

            def forward(self, x, index):
                return torch.ops.karume.expert_linear(x[:, :32], index, self.weight)

        with pytest.raises(OpContractError, match="input feature count"):
            convert_module(Narrow(), (x, index))


# ---------------------------------------------------------------------------
# Rank-3 fixed weights (ADR 0145 decisions 2 and 3)
# ---------------------------------------------------------------------------

#: A small expert table [E, out, in] = [2, 3, 32] with group 16: rows E·out = 6, 2 groups each.
SHAPE = [2, 3, 32]
ROWS, GROUPS = 6, 2


def fixed_expert_graph(extra: str | None = None) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """`y = expert_linear(x, index, w)` with `w [2, 3, 32]`; `extra` adds a second consumer of w
    (`add`, outside any weight slot)."""
    nodes = [IrNode(op="expert_linear", ins=["x", "index", "w"], outs=["y"], attrs={})]
    values = {
        "w": IrValue(dtype="f32", shape=SHAPE),
        "y": IrValue(dtype="f32", shape=[4, 2, 3]),
    }
    inputs = [
        IrInput(name="x", dtype="f32", shape=[4, 32]),
        IrInput(name="index", dtype="i32", shape=[4, 2]),
    ]
    outputs = ["y"]
    if extra == "add":
        inputs.append(IrInput(name="z", dtype="f32", shape=SHAPE))
        nodes.append(IrNode(op="add", ins=["z", "w"], outs=["s"], attrs={}))
        values["s"] = IrValue(dtype="f32", shape=SHAPE)
        outputs.append("s")
    graph = IrGraph(
        symbols=[],
        inputs=inputs,
        outputs=outputs,
        initializers={"w": IrInitializer(tensor="experts.w", storage=IrStorage(dtype="f32"))},
        values=values,
        nodes=nodes,
    )
    return graph, {"experts.w": torch.empty(*SHAPE, device="meta")}


def codes() -> torch.Tensor:
    """Packed codes `[E·out, in/2]`: row r holds q = (r % 7) + 1 everywhere."""
    q = torch.tensor([[(row % 7) + 1] * 32 for row in range(ROWS)], dtype=torch.int8)
    return pack_int4(q).reshape(ROWS, 16)


#: One F32 step per (row, group): 1.0 / 0.5 alternating (f16 0x3c00 / 0x3800).
STEPS = torch.tensor([[1.0, 0.5]] * ROWS, dtype=torch.float32)


def stored_bytes(model, key: str) -> bytes:
    return bytes(memoryview(model.tensors[key]))


class TestRank3FixedWeights:
    @pytest.mark.parametrize("dtype", ["i4", "i4f16"])
    def test_an_expert_weight_is_declared_with_its_group(self, dtype: str):
        graph, tensors = fixed_expert_graph()

        model = stored_model(
            graph, tensors, fixed_weights={"experts.w": FixedQuantizedWeight(dtype, codes(), STEPS)}
        )

        storage = model.graph.initializers["w"].storage
        assert (storage.dtype, storage.group_size) == (dtype, 16)
        # The payload is the packed codes as is (the bytes of [E·out, in]).
        assert stored_bytes(model, "experts.w") == codes().numpy().tobytes()

    def test_i4_stores_the_f32_steps_row_major(self):
        graph, tensors = fixed_expert_graph()

        model = stored_model(
            graph, tensors, fixed_weights={"experts.w": FixedQuantizedWeight("i4", codes(), STEPS)}
        )

        scale_key = model.graph.initializers["w"].storage.scale
        assert stored_bytes(model, scale_key) == struct.pack("<12f", *([1.0, 0.5] * ROWS))

    def test_i4f16_stores_the_f16_step_table(self):
        """Each step rounded to f16 once (ADR 0145 decision 3): 1.0 = 0x3c00, 0.5 = 0x3800, 12
        steps = 6 whole u32 words (no padding)."""
        graph, tensors = fixed_expert_graph()

        model = stored_model(
            graph,
            tensors,
            fixed_weights={"experts.w": FixedQuantizedWeight("i4f16", codes(), STEPS)},
        )

        scale_key = model.graph.initializers["w"].storage.scale
        assert stored_bytes(model, scale_key) == struct.pack("<12H", *([0x3C00, 0x3800] * ROWS))

    @pytest.mark.parametrize("dtype", ["i4", "i4f16"])
    def test_the_container_reads_back_and_passes_the_acceptance_rules(
        self, dtype: str, tmp_path: Path
    ):
        """The runtime support check admits i4f16 for expert_linear's weight slot (ADR 0145)."""
        graph, tensors = fixed_expert_graph()
        model = stored_model(
            graph, tensors, fixed_weights={"experts.w": FixedQuantizedWeight(dtype, codes(), STEPS)}
        )

        paths = write_model_container(
            tmp_path / "model.krm",
            model.graph,
            model.tensors,
            model.bindings,
            graph_name="g",
            provenance=PROVENANCE,
            single=True,
        )

        assert_ir_accepted(read_container(paths))
        assert_runtime_support(model.graph)

    @pytest.mark.parametrize("dtype", ["i2", "i8"])
    def test_an_expert_weight_refuses_i2_and_i8(self, dtype: str):
        graph, tensors = fixed_expert_graph()
        packed = torch.zeros((ROWS, 32 // (4 if dtype == "i2" else 1)), dtype=torch.uint8)
        if dtype == "i8":
            packed = packed.to(torch.int8)
        fixed = FixedQuantizedWeight(dtype, packed, torch.ones((ROWS, 1), dtype=torch.float32))

        with pytest.raises(EmitError, match=f"takes i4 / i4f16 only.*not {dtype}"):
            stored_model(graph, tensors, fixed_weights={"experts.w": fixed})

    @pytest.mark.parametrize("dtype", ["i4", "i4f16"])
    def test_a_rank_3_weight_that_another_op_reads_is_refused(self, dtype: str):
        """Read by an `add` too: not an expert-only weight, so the rank-2 rules apply (and i4f16
        keeps its refusal)."""
        graph, tensors = fixed_expert_graph(extra="add")

        with pytest.raises(EmitError, match="rank-2"):
            stored_model(
                graph,
                tensors,
                fixed_weights={"experts.w": FixedQuantizedWeight(dtype, codes(), STEPS)},
            )

    def test_a_rank_3_weight_of_linear_is_refused(self):
        """linear reads rank-2 weights only; a rank-3 fixed weight is an expert_linear's."""
        graph = IrGraph(
            symbols=[],
            inputs=[IrInput(name="x", dtype="f32", shape=[4, 32])],
            outputs=["y"],
            initializers={
                "w": IrInitializer(tensor="layer.w", storage=IrStorage(dtype="f32")),
                "b": IrInitializer(tensor="layer.b", storage=IrStorage(dtype="f32")),
            },
            values={
                "w": IrValue(dtype="f32", shape=SHAPE),
                "b": IrValue(dtype="f32", shape=[3]),
                "y": IrValue(dtype="f32", shape=[4, 3]),
            },
            nodes=[IrNode(op="linear", ins=["x", "w", "b"], outs=["y"], attrs={})],
        )
        tensors = {"layer.w": torch.empty(*SHAPE, device="meta"), "layer.b": torch.zeros(3)}

        with pytest.raises(EmitError, match=re.escape("must be a positive rank-2")):
            stored_model(
                graph,
                tensors,
                fixed_weights={"layer.w": FixedQuantizedWeight("i4", codes(), STEPS)},
            )

    def test_an_expert_weight_that_is_a_graph_output_is_refused(self):
        """It could not stay resident, and an expert weight never expands on the host."""
        graph, tensors = fixed_expert_graph()
        graph = IrGraph(
            symbols=graph.symbols,
            inputs=graph.inputs,
            outputs=[*graph.outputs, "w"],
            initializers=graph.initializers,
            values=graph.values,
            nodes=graph.nodes,
        )

        with pytest.raises(EmitError, match="also a graph output"):
            stored_model(
                graph,
                tensors,
                fixed_weights={"experts.w": FixedQuantizedWeight("i4", codes(), STEPS)},
            )


class TestRuntimeSupport:
    def test_i4f16_read_by_expert_linear_alone_is_supported(self):
        graph, _ = fixed_expert_graph()
        graph = IrGraph(
            symbols=graph.symbols,
            inputs=graph.inputs,
            outputs=graph.outputs,
            initializers={
                "w": IrInitializer(
                    tensor="experts.w", storage=IrStorage("i4f16", "experts.w.scale", 16)
                )
            },
            values=graph.values,
            nodes=graph.nodes,
        )

        assert_runtime_support(graph)

    def test_i4f16_also_read_by_another_op_is_not(self):
        graph, _ = fixed_expert_graph(extra="add")
        graph = IrGraph(
            symbols=graph.symbols,
            inputs=graph.inputs,
            outputs=graph.outputs,
            initializers={
                "w": IrInitializer(
                    tensor="experts.w", storage=IrStorage("i4f16", "experts.w.scale", 16)
                )
            },
            values=graph.values,
            nodes=graph.nodes,
        )

        with pytest.raises(ContainerError, match=r"unsupported storage dtype 'i4f16' \(1\): w"):
            assert_runtime_support(graph)


def half_storage_graph() -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """`y = expert_linear(x, index, w)` beside `z = linear(x, v, b)`: one expert weight and one
    weight an automatic f16 / bf16 storage takes, with real f32 data."""
    graph, _ = fixed_expert_graph()
    generator = torch.Generator().manual_seed(20261108)
    graph = IrGraph(
        symbols=graph.symbols,
        inputs=graph.inputs,
        outputs=[*graph.outputs, "z"],
        initializers={
            **graph.initializers,
            "v": IrInitializer(tensor="dense.v", storage=IrStorage(dtype="f32")),
            "b": IrInitializer(tensor="dense.b", storage=IrStorage(dtype="f32")),
        },
        values={
            **graph.values,
            "v": IrValue(dtype="f32", shape=[5, 32]),
            "b": IrValue(dtype="f32", shape=[5]),
            "z": IrValue(dtype="f32", shape=[4, 5]),
        },
        nodes=[*graph.nodes, IrNode(op="linear", ins=["x", "v", "b"], outs=["z"], attrs={})],
    )
    tensors = {
        "experts.w": torch.randn(*SHAPE, generator=generator),
        "dense.v": torch.randn(5, 32, generator=generator),
        "dense.b": torch.randn(5, generator=generator),
    }
    return graph, tensors


class TestAutomaticHalfStorage:
    """`expert_linear` refuses f16 / bf16 storage (ADR 0145 decision 2), so emit's automatic
    planning leaves its weight f32 — planned, the export would pass and the load fail."""

    @pytest.mark.parametrize("dtype", ["f16", "bf16"])
    def test_the_default_leaves_the_expert_weight_f32(self, dtype: str):
        graph, tensors = half_storage_graph()

        model = stored_model(graph, tensors, weight_dtype=dtype)

        storages = {name: model.graph.initializers[name].storage.dtype for name in ("w", "v")}
        assert storages == {"w": "f32", "v": dtype}
        assert_runtime_support(model.graph)

    @pytest.mark.parametrize("dtype", ["f16", "bf16"])
    def test_an_explicit_choice_on_the_expert_weight_fails_loudly(self, dtype: str):
        graph, tensors = half_storage_graph()

        with pytest.raises(EmitError, match="an expert_linear weight takes f32 / i4 / i4f16 only"):
            stored_model(graph, tensors, weight_dtype_overrides={"experts.w": dtype})
