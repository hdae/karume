"""Pins the behaviour of the IR surgery into the states form (`karume.states`).

The surgery's acceptance criterion is not "IR that can be written" but "**IR the runtime can
read**", so the main shapes go through the round trip (`to_json` → `parse_ir_graph` →
`assert_op_contracts` / `assert_runtime_support`). The surgery does not reimplement the rules but
leaves them to the existing checks, so passing those checks is itself the specification.

The subject is a small synthetic graph (no torch) — the real model has 894 nodes, and the surgery
touches only "attention and the folding leftovers of the mask", so only that shape is copied, at
minimal size.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from karume.container import Provenance
from karume.emit import stored_model
from karume.ir import IrGraph
from karume.ops import OpContractError
from karume.publish import publish_container
from karume.states import StateAttentionSpec, StatesFormError, StatesPlan, to_states_form
from karume.verify import (
    IrError,
    assert_ir_accepted,
    assert_op_contracts,
    assert_runtime_support,
    ir_graph_from_container,
    parse_ir_graph,
    verify_container,
)

#: Declared shape of q (`[B,H,M,D]` — H=8) and of k / v (`[B,Hkv,M,D]` — Hkv=2, GQA 4:1).
Q = [1, 8, "T", 16]
KV = [1, 2, "T", 16]

#: Tmax of the constant holding the additive causal mask (`sym_prefix_slice` cuts out the leading
#: T×T — ADR 0010).
TMAX = 64


def _input(name: str, shape: Sequence) -> dict:
    return {"name": name, "dtype": "f32", "shape": list(shape)}


def _value(shape: Sequence) -> dict:
    return {"dtype": "f32", "shape": list(shape)}


def _attention(q: str, k: str, v: str, out: str) -> dict:
    """Attention node in the classic form (mask as the 4th input)."""
    return {"op": "attention", "ins": [q, k, v, "mask"], "outs": [out], "attrs": {"scale": 0.25}}


#: Folding leftovers: Tmax² constant → cut out the bound T×T. In the states form causality becomes
#: a predicate computation, so they die (the main target of pruning).
MASK_NODE = {
    "op": "sym_prefix_slice",
    "ins": ["mask.table"],
    "outs": ["mask"],
    "attrs": {
        "sym": "T",
        "slices": [
            {"dim": 2, "coeff": 1, "offset": 0},
            {"dim": 3, "coeff": 1, "offset": 0},
        ],
    },
}


def source_dict(**overrides) -> dict:
    """One layer of chunk-local causal self-attention.

    The mask is a Tmax² constant + `sym_prefix_slice` (the same shape as the real model's folding
    result), and after attention sits one node consuming a **live** initializer (the counterpart
    showing pruning does not take live declarations with it).
    """
    graph = {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add", "attention", "sym_prefix_slice"]},
        "symbols": ["T"],
        "inputs": [_input("q", Q), _input("k", KV), _input("v", KV)],
        "outputs": ["o"],
        "initializers": {
            "mask.table": {"tensor": "mask.table", "storage": {"dtype": "f32"}},
            "bias": {"tensor": "attn.bias", "storage": {"dtype": "f32"}},
        },
        "values": {
            "mask.table": _value([1, 1, TMAX, TMAX]),
            "bias": _value([16]),
            "mask": _value([1, 1, "T", "T"]),
            "a": _value(Q),
            "o": _value(Q),
        },
        "nodes": [
            MASK_NODE,
            _attention("q", "k", "v", "a"),
            {"op": "add", "ins": ["a", "bias"], "outs": ["o"], "attrs": {}},
        ],
    }
    graph.update(overrides)
    if "requires" not in overrides:
        graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}
    return graph


def two_layer_dict(*, shared: bool = True) -> dict:
    """Graph with two attentions.

    `shared=True` is the wiring of a KV-sharing layer (layer 1 takes the owner layer's k/v value
    tensors directly as `ins` — ADR 0067 decision 4); `shared=False` gives each layer its own k/v.
    """
    extra = [_input("q1", Q)] if shared else [_input("q1", Q), _input("k1", KV), _input("v1", KV)]
    k1, v1 = ("k", "v") if shared else ("k1", "v1")
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add", "attention", "sym_prefix_slice"]},
        "symbols": ["T"],
        "inputs": [_input("q", Q), _input("k", KV), _input("v", KV), *extra],
        "outputs": ["o"],
        "initializers": {"mask.table": {"tensor": "mask.table", "storage": {"dtype": "f32"}}},
        "values": {
            "mask.table": _value([1, 1, TMAX, TMAX]),
            "mask": _value([1, 1, "T", "T"]),
            "a": _value(Q),
            "a1": _value(Q),
            "o": _value(Q),
        },
        "nodes": [
            MASK_NODE,
            _attention("q", "k", "v", "a"),
            _attention("q1", k1, v1, "a1"),
            {"op": "add", "ins": ["a", "a1"], "outs": ["o"], "attrs": {}},
        ],
    }


def source(**overrides) -> IrGraph:
    """The subject graph. **Built through the parser** — the surgery assumes valid IR as input."""
    return parse_ir_graph(json.dumps(source_dict(**overrides)))


def plan(*attentions: StateAttentionSpec, capacity_symbol: str = "C") -> StatesPlan:
    return StatesPlan(capacity_symbol=capacity_symbol, attentions=attentions)


#: Standard spec of the one-layer graph (full context, capacity symbol).
SPEC = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v")


def checked(graph: IrGraph) -> IrGraph:
    """Dump the surgery result to JSON and **run the parser and contract checks again** (the round
    trip is the acceptance criterion)."""
    parsed = parse_ir_graph(graph.to_json())
    assert_op_contracts(parsed)
    assert_runtime_support(parsed)
    return parsed


def ops_of(graph: IrGraph) -> list[str]:
    return [node.op for node in graph.nodes]


class TestASingleLayerSurgery:
    def test_the_mask_input_is_replaced_by_the_states_field(self):
        """Drops the mask, adds the `states` field (causal is a predicate — ADR 0067 decision 4)."""
        node = to_states_form(source(), plan(SPEC)).nodes[0]

        assert node.op == "attention"
        assert node.ins == ["q", "k", "v"]
        assert node.states == {"k": "l0.k", "v": "l0.v"}
        # Existing attrs pass through (the half-scale contract is not the surgery's concern).
        assert node.attrs == {"scale": 0.25}

    def test_an_append_is_inserted_after_the_reader(self):
        """One `state_append` per slot, right after the reader (ADR 0067 decision 5b, item 1).

        The later `add` stays **after** the append = the relative order of surviving nodes is kept.
        """
        graph = to_states_form(source(), plan(SPEC))

        assert ops_of(graph) == ["attention", "state_append", "state_append", "add"]
        assert [node.ins for node in graph.nodes[1:3]] == [["k"], ["v"]]
        assert [node.states for node in graph.nodes[1:3]] == [
            {"slot": "l0.k"},
            {"slot": "l0.v"},
        ]
        assert [node.outs for node in graph.nodes[1:3]] == [[], []]

    def test_the_slots_are_declared_with_the_capacity_symbol(self):
        """A slot derives `[B,Hkv,C,D]` from the ins' `[B,Hkv,M,D]` (only the capacity changes)."""
        graph = to_states_form(source(), plan(SPEC))

        assert list(graph.states) == ["l0.k", "l0.v"]
        assert [slot.shape for slot in graph.states.values()] == [[1, 2, "C", 16]] * 2
        assert {slot.dtype for slot in graph.states.values()} == {"f32"}
        assert graph.symbols == ["T", "C"]

    def test_the_folded_mask_constant_and_its_producer_are_pruned(self):
        """Nobody reads the Tmax² constant and `sym_prefix_slice` any more = not kept in the
        distribution."""
        graph = to_states_form(source(), plan(SPEC))

        assert "sym_prefix_slice" not in ops_of(graph)
        assert list(graph.initializers) == ["bias"]
        # Declarations of dead values go too (verify rejects orphaned declarations).
        assert list(graph.values) == ["bias", "a", "o"]

    def test_the_source_graph_is_left_untouched(self):
        """MUST be a pure function — without running the same checks on the graphs before and after
        the surgery, the origin of a failure cannot be isolated."""
        graph = source()
        before = graph.to_json()

        to_states_form(graph, plan(SPEC))

        assert graph.to_json() == before

    def test_the_result_passes_the_ir_gates(self):
        """Passes the states, order, shape and enumeration checks in the round trip."""
        graph = checked(to_states_form(source(), plan(SPEC)))

        assert graph.required_ops == ["add", "attention", "state_append"]


class TestSlidingWindow:
    def test_every_node_touching_the_slot_declares_the_window(self):
        """The logical col → physical row mapping MUST be the same for reads and writes (ADR 0067
        decision 4)."""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=32)

        graph = checked(to_states_form(source(), plan(spec)))

        assert [node.attrs.get("window") for node in graph.nodes[:3]] == [32, 32, 32]

    def test_an_integral_float_window_is_written_as_the_integer(self):
        """`32.0` is the runtime's window 32 (`Number.isSafeInteger(32.0)` is true), so the
        surgery accepts it and writes the integer (kept as float, the IR would carry `32.0`)."""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=32.0)

        graph = checked(to_states_form(source(), plan(spec)))

        windows = [node.attrs.get("window") for node in graph.nodes[:3]]
        assert windows == [32, 32, 32]
        assert all(type(window) is int for window in windows)

    def test_a_window_larger_than_the_capacity_is_rejected_by_the_slot_contract(self):
        """Consistency with the slot contract (`window ≤ C`) — a numeric capacity decides it from
        the declaration alone."""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=128, capacity=64)

        with pytest.raises(OpContractError, match="exceeds the slot capacity 64"):
            checked(to_states_form(source(), plan(spec)))


class TestSharedSlots:
    """KV-sharing layers (several attentions read one slot — the last 20 layers of Gemma 4 E2B)."""

    def test_the_append_follows_the_last_reader(self):
        """The append comes **after every reader** (a ring wrap cannot overwrite this step's
        readers' past rows — decision 5b)."""
        graph = checked(
            to_states_form(
                parse_ir_graph(json.dumps(two_layer_dict())),
                plan(SPEC, StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v")),
            )
        )

        assert ops_of(graph) == [
            "attention",
            "attention",
            "state_append",
            "state_append",
            "add",
        ]
        assert list(graph.states) == ["l0.k", "l0.v"]

    def test_readers_wiring_a_different_value_are_rejected(self):
        """A sharing layer wires the owner layer's k/v value tensors directly — another value is a
        wiring mistake."""
        graph = parse_ir_graph(json.dumps(two_layer_dict(shared=False)))

        with pytest.raises(
            StatesFormError, match=r"value written to slot '[^']*' differs between readers"
        ):
            to_states_form(
                graph, plan(SPEC, StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v"))
            )

    def test_readers_disagreeing_on_the_capacity_are_rejected(self):
        """A capacity that differs per reader would demand two physical shapes for the same slot."""
        graph = parse_ir_graph(json.dumps(two_layer_dict()))
        shared = StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v", capacity=128)

        with pytest.raises(
            StatesFormError, match=re.escape("derived shape differs between readers")
        ):
            to_states_form(graph, plan(SPEC, shared))

    def test_readers_disagreeing_on_the_window_are_rejected(self):
        """A window mismatch is a silent misread (one side reads a different physical row)."""
        graph = parse_ir_graph(json.dumps(two_layer_dict()))
        shared = StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v", window=32)

        with pytest.raises(StatesFormError, match=re.escape("window differs between readers")):
            to_states_form(graph, plan(SPEC, shared))


class TestCapacity:
    def test_a_numeric_capacity_replaces_the_symbol(self):
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", capacity=256)

        graph = checked(to_states_form(source(), plan(spec)))

        assert graph.states["l0.k"].shape == [1, 2, 256, 16]

    def test_the_symbol_is_absent_when_no_slot_uses_it(self):
        """Declaring an unused symbol rejects the whole graph (a symbol with no binding point)."""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", capacity=256)

        graph = checked(to_states_form(source(), plan(spec)))

        assert graph.symbols == ["T"]

    @pytest.mark.parametrize(
        ("capacity", "diagnosis"),
        [
            (0, "dimension 0 is not a positive integer"),
            (-1, "dimension -1 is not a non-negative integer"),
            (True, "dimension is neither a number nor a string"),
            (1.5, "dimension is neither a number nor a string"),
        ],
    )
    def test_a_capacity_outside_the_value_range_is_caught_only_by_the_parser(
        self, capacity, diagnosis
    ):
        """Unlike `window` (`_window` requires a positive integer), `capacity` has no range check in
        the surgery.

        Pins the current two steps explicitly — the surgery passes, and `parse_ir_graph` in the
        round trip fails (the diagnosis points at the IR shape, not at the plan field).
        """
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", capacity=capacity)

        graph = to_states_form(source(), plan(spec))

        assert graph.states["l0.k"].shape == [1, 2, capacity, 16]
        with pytest.raises(IrError, match=diagnosis):
            parse_ir_graph(graph.to_json())

    def test_a_mixed_plan_declares_the_symbol_once(self):
        """Capacity differing per layer (Gemma 4's mix of sliding / full).

        The symbol is added if at least one layer uses it (numeric-capacity layers do not).
        """
        layer1 = StateAttentionSpec(output="a1", k_slot="l1.k", v_slot="l1.v", capacity=TMAX)

        graph = checked(
            to_states_form(
                parse_ir_graph(json.dumps(two_layer_dict(shared=False))), plan(SPEC, layer1)
            )
        )

        assert graph.symbols == ["T", "C"]
        assert graph.states["l0.k"].shape == [1, 2, "C", 16]
        assert graph.states["l1.k"].shape == [1, 2, TMAX, 16]
        assert ops_of(graph) == [
            "attention",
            "state_append",
            "state_append",
            "attention",
            "state_append",
            "state_append",
            "add",
        ]


class TestRejectedSurgery:
    def test_an_unknown_target_is_rejected(self):
        with pytest.raises(StatesFormError, match="no node in the graph defines this value"):
            to_states_form(source(), plan(StateAttentionSpec("ghost", "l0.k", "l0.v")))

    def test_a_node_that_is_not_attention_is_rejected(self):
        with pytest.raises(StatesFormError, match=re.escape("op is 'sym_prefix_slice'")):
            to_states_form(source(), plan(StateAttentionSpec("mask", "l0.k", "l0.v")))

    def test_an_attention_without_a_mask_is_rejected(self):
        """A mask-less export expresses causality otherwise = the surgery's premise is broken."""
        nodes = [dict(node) for node in source_dict()["nodes"]]
        nodes[1] = {**nodes[1], "ins": ["q", "k", "v"]}

        with pytest.raises(StatesFormError, match="3 inputs"):
            to_states_form(source(nodes=nodes), plan(SPEC))

    def test_a_second_surgery_on_the_same_node_is_rejected(self):
        """A double surgery would "write this step's k/v twice" (two appends too)."""
        once = to_states_form(source(), plan(SPEC))

        with pytest.raises(StatesFormError, match="already in the states form"):
            to_states_form(once, plan(SPEC))

    def test_a_capacity_symbol_colliding_with_an_existing_symbol_is_rejected(self):
        """Using a symbol bound by an input as the capacity too would resolve value shapes by the
        capacity."""
        with pytest.raises(StatesFormError, match=re.escape("is already in graph.symbols")):
            to_states_form(source(), plan(SPEC, capacity_symbol="T"))

    @pytest.mark.parametrize("symbol", ["q", "o"])
    def test_a_capacity_symbol_colliding_with_a_value_name_is_rejected(self, symbol):
        """Detects the mix-up of passing a value name (input / node output) as the symbol field."""
        with pytest.raises(StatesFormError, match="same name as a value"):
            to_states_form(source(), plan(SPEC, capacity_symbol=symbol))

    @pytest.mark.parametrize("window", [0, -1, True, 1.5])
    def test_a_window_outside_the_value_range_is_rejected(self, window):
        """bool derives from int but is not a window (`True` would pass silently as width 1)."""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=window)

        with pytest.raises(StatesFormError, match="is not a positive integer"):
            to_states_form(source(), plan(spec))

    @pytest.mark.parametrize("window", [2**53, float(2**53), float("inf"), float("nan")])
    def test_a_window_the_runtime_rejects_is_rejected(self, window):
        """The upper bound is the safe integer, as in the runtime's `window` attr — a wider one
        would be an IR the runtime refuses."""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=window)

        with pytest.raises(StatesFormError, match="is not a positive integer"):
            to_states_form(source(), plan(spec))

    def test_an_input_orphaned_by_the_surgery_is_rejected(self):
        """An export that builds the mask from a graph input cannot go to the states form (a wiring
        mistake)."""
        graph = source_dict()
        nodes = [dict(node) for node in graph["nodes"]]
        nodes[1] = {**nodes[1], "ins": ["q", "k", "v", "m"]}

        with pytest.raises(StatesFormError, match=re.escape("inputs ['m'] unreachable")):
            to_states_form(
                source(inputs=[*graph["inputs"], _input("m", [1, 1, "T", "T"])], nodes=nodes),
                plan(SPEC),
            )


class TestTheOrderCheckIsEffective:
    """Fault injection showing the surgery's order does not "pass by accident" (ADR 0067 decision
    5b, item 2)."""

    def test_moving_an_append_before_its_reader_is_rejected(self):
        """Moving an append before its reader fails the order check = the check is effective."""
        graph = to_states_form(source(), plan(SPEC)).to_dict()
        nodes = graph["nodes"]
        # Swap the attention and the append of k (the data edges use only inputs, so SSA stays
        # valid — only the order check fails).
        graph["nodes"] = [nodes[1], nodes[0], *nodes[2:]]
        parsed = parse_ir_graph(json.dumps(graph))

        with pytest.raises(IrError, match="is followed by reader"):
            assert_op_contracts(parsed)


class TestTheStatesFormSurvivesTheContainer:
    """The surgered graph **can be rebuilt** from the container's two documents
    (`verify.ir_graph_from_container`).

    The container's graph document is IR v2 (initializer names are tensor keys, storage is in the
    binding table), so the `karume dist` and `karume verify` checks rebuild v1 before applying the
    rules. The `states` section and the nodes' `states` field carry over to v2 as is, but dropping
    them **fails the referential-integrity check** (declared slots nobody references), so the
    round trip itself is pinned.
    """

    def _published(self, tmp_path: Path) -> list[Path]:
        graph = to_states_form(source(), plan(SPEC))
        tensors = {}
        for name, initializer in graph.initializers.items():
            if initializer.is_shared:
                continue
            assert initializer.tensor is not None
            shape = [int(dim) for dim in graph.values[name].shape]
            dtype = torch.int32 if initializer.storage.dtype == "i32" else torch.float32
            tensors[initializer.tensor] = torch.zeros(shape, dtype=dtype)
        stored = stored_model(graph, tensors)
        result = publish_container(
            tmp_path / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="dec",
            provenance=Provenance(license="mit"),
        )
        return list(result.parts)

    def test_the_slots_and_their_references_round_trip(self, tmp_path: Path) -> None:
        read = verify_container(self._published(tmp_path)).read

        restored = ir_graph_from_container(read, "dec")

        assert sorted(restored.states) == ["l0.k", "l0.v"]
        referenced = {slot for node in restored.nodes for slot in node.states.values()}
        assert referenced == set(restored.states)

    def test_the_acceptance_gate_passes_on_the_container(self, tmp_path: Path) -> None:
        """Exactly the form that passes assembly (`karume dist`) and `karume verify`."""
        assert_ir_accepted(verify_container(self._published(tmp_path)).read)
