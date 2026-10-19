"""Pins the 3 declarations of a borrower graph (external slot / readonly attention / shared
initializer).

The source of truth is the stage 2 contract of ADR
[0096](../../../docs/decisions/0096-speculative-decoding.md); its TS mirror is
`packages/runtime/src/format/ir.ts` + `packages/runtime/src/runtime/plan.ts`. This file covers all
3 layers on the Python side — the declarations' round trip (`to_json` → `parse_ir_graph`), the
contract and shape checks, the surgery (`karume.states.to_external_states_form`) and the write
(shared declarations are excluded from storage).

The subject graphs are small synthetic graphs (no torch needed). The real drafter is unrolled as
4 layers × k steps, but the discipline of the 3 declarations is fully exercised with 1–2
attentions.

Fault injection (these 4 being rejected is what guarantees "the checks are effective" — if they
passed, **only the values** would change silently):

1. `state_append` into an external slot (the borrower pollutes the lender's ring)
2. a reader that is not readonly reads an external slot (reading with this step's k/v added)
3. `tensor` / `scale` on a shared initializer (the bytes come from 2 places)
4. external and owned slots mixed (the graph does not determine the allocation rule)
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
import torch

from karume.emit import EmitError, stored_model
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrState,
    IrStorage,
    IrValue,
)
from karume.ops import OpContractError
from karume.quantize import channel_scale, quantize_to_int8
from karume.states import (
    ExternalAttentionSpec,
    ExternalStatesPlan,
    StatesFormError,
    to_external_states_form,
)
from karume.verify import (
    IrError,
    assert_op_contracts,
    assert_runtime_support,
    parse_ir_graph,
)

#: The borrower's q (`[B,H,1,D]` — H=4; M MUST be exactly 1) and the shapes of the K / V inputs
#: placed for the trace. T is a placeholder that "may be a small fixed value" (the surgery drops it,
#: so it does not remain in the distribution).
Q_SLIDING = [1, 4, 1, 256]
KV_SLIDING = [1, 1, 8, 256]
Q_FULL = [1, 4, 1, 512]
KV_FULL = [1, 1, 8, 512]

#: The capacity of the lender's slots (sliding = window 512 + slack 8 as a number / full = a
#: symbol).
SLIDING_CAPACITY = 520
CAPACITY_SYMBOL = "C"
WINDOW = 512


def _input(name: str, shape: Sequence) -> dict:
    return {"name": name, "dtype": "f32", "shape": list(shape)}


def _value(shape: Sequence) -> dict:
    return {"dtype": "f32", "shape": list(shape)}


def source_dict() -> dict:
    """The borrower right after the trace (before the surgery) — 2 attentions in the classic form
    (sliding / full).

    The mask is **a baked constant** (bidirectional with q_len 1 = a zero row allowing every
    column). Once the surgery removes it from the node's inputs nobody reads it, so pruning
    removes it as is.
    """
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add", "attention"]},
        "symbols": [],
        "inputs": [
            _input("qs", Q_SLIDING),
            _input("qf", Q_FULL),
            _input("sliding_k", KV_SLIDING),
            _input("sliding_v", KV_SLIDING),
            _input("full_k", KV_FULL),
            _input("full_v", KV_FULL),
        ],
        "outputs": ["o", "af"],
        "initializers": {
            "mask_s": {"tensor": "drafter.mask_s", "storage": {"dtype": "f32"}},
            "mask_f": {"tensor": "drafter.mask_f", "storage": {"dtype": "f32"}},
            "bias": {"tensor": "drafter.bias", "storage": {"dtype": "f32"}},
        },
        "values": {
            "mask_s": _value([1, 1, 1, 8]),
            "mask_f": _value([1, 1, 1, 8]),
            "bias": _value([256]),
            "as": _value(Q_SLIDING),
            "af": _value(Q_FULL),
            "o": _value(Q_SLIDING),
        },
        "nodes": [
            {
                "op": "attention",
                "ins": ["qs", "sliding_k", "sliding_v", "mask_s"],
                "outs": ["as"],
                "attrs": {"scale": 1.0},
            },
            {
                "op": "attention",
                "ins": ["qf", "full_k", "full_v", "mask_f"],
                "outs": ["af"],
                "attrs": {"scale": 1.0},
            },
            {"op": "add", "ins": ["as", "bias"], "outs": ["o"], "attrs": {}},
        ],
    }


def source() -> IrGraph:
    """The subject graph. **Built through the parser** — the surgery presupposes a valid IR as
    input."""
    return parse_ir_graph(json.dumps(source_dict()))


SLIDING_SPEC = ExternalAttentionSpec(
    output="as",
    k_slot="l13.k",
    v_slot="l13.v",
    k_input="sliding_k",
    v_input="sliding_v",
    kv_heads=1,
    head_dim=256,
    capacity=SLIDING_CAPACITY,
    window=WINDOW,
)
FULL_SPEC = ExternalAttentionSpec(
    output="af",
    k_slot="l14.k",
    v_slot="l14.v",
    k_input="full_k",
    v_input="full_v",
    kv_heads=1,
    head_dim=512,
    capacity=CAPACITY_SYMBOL,
)


def plan(*attentions: ExternalAttentionSpec) -> ExternalStatesPlan:
    return ExternalStatesPlan(capacity_symbol=CAPACITY_SYMBOL, attentions=attentions)


def checked(graph: IrGraph) -> IrGraph:
    """Serialize the surgery result to JSON and **run it through the parser and the contract checks
    again** (the round trip is the acceptance condition)."""
    parsed = parse_ir_graph(graph.to_json())
    assert_op_contracts(parsed)
    assert_runtime_support(parsed)
    return parsed


class TestTheExternalSurgery:
    def test_the_reader_becomes_a_readonly_node_with_only_q(self):
        """The readonly form has exactly one input, q (neither this step's k/v nor a mask)."""
        graph = checked(to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC)))

        sliding, full = (node for node in graph.nodes if node.op == "attention")
        assert sliding.ins == ["qs"]
        assert sliding.attrs == {"scale": 1.0, "readonly": True, "window": WINDOW}
        assert sliding.states == {"k": "l13.k", "v": "l13.v"}
        # A full layer declares no `window` (the absence of the field is itself "the whole
        # context" — ADR 0067 decision 4).
        assert full.attrs == {"scale": 1.0, "readonly": True}

    def test_no_append_is_inserted(self):
        """The external storage belongs to the lender — the borrower writes nothing."""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert [node.op for node in graph.nodes] == ["attention", "attention", "add"]

    def test_the_slots_are_declared_external_from_the_plan(self):
        """The slot shape is the plan's `[1, Hkv, capacity, D]` (the placeholder T does not
        remain)."""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert graph.states["l13.k"] == IrState(
            dtype="f32", shape=[1, 1, SLIDING_CAPACITY, 256], external=True
        )
        assert graph.states["l14.v"] == IrState(
            dtype="f32", shape=[1, 1, CAPACITY_SYMBOL, 512], external=True
        )

    def test_the_traced_kv_inputs_are_dropped(self):
        """The K / V placeholders disappear together with their declarations (the caller no longer
        passes them)."""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert [spec.name for spec in graph.inputs] == ["qs", "qf"]

    def test_the_baked_masks_are_pruned(self):
        """Mask constants that nobody reads any more are pruned; live initializers remain."""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert sorted(graph.initializers) == ["bias"]
        assert "mask_s" not in graph.values

    def test_the_capacity_symbol_is_declared_only_when_used(self):
        """The capacity symbol is added to `symbols` only when a slot uses it."""
        with_symbol = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))
        assert with_symbol.symbols == [CAPACITY_SYMBOL]

    def test_the_input_declaration_order_is_preserved(self):
        """The remaining inputs keep their original order (the host binds by order)."""
        graph = to_external_states_form(source(), plan(FULL_SPEC, SLIDING_SPEC))

        assert [spec.name for spec in graph.inputs] == ["qs", "qf"]


class TestTheSurgeryRejectsMiswiring:
    def test_dropping_a_value_that_is_not_the_node_kv_is_rejected(self):
        """If the input to drop is not that node's k / v, it is a wiring error (another value would
        disappear)."""
        spec = ExternalAttentionSpec(
            output="as",
            k_slot="l13.k",
            v_slot="l13.v",
            k_input="qf",
            v_input="sliding_v",
            kv_heads=1,
            head_dim=256,
            capacity=SLIDING_CAPACITY,
            window=WINDOW,
        )

        with pytest.raises(StatesFormError, match=r"k given as 'qf'"):
            to_external_states_form(source(), plan(spec))

    def test_swapping_k_and_v_is_rejected(self):
        """A mix-up giving the original V as K and the original K as V — same shape, so if it passed
        only the values would silently differ."""
        spec = ExternalAttentionSpec(
            output="as",
            k_slot="l13.k",
            v_slot="l13.v",
            k_input="sliding_v",
            v_input="sliding_k",
            kv_heads=1,
            head_dim=256,
            capacity=SLIDING_CAPACITY,
            window=WINDOW,
        )

        with pytest.raises(StatesFormError, match=r"k given as 'sliding_v'"):
            to_external_states_form(source(), plan(spec))

    def test_naming_the_same_input_as_both_k_and_v_is_rejected(self):
        """Giving the same input as K and V (K goes into the V slot) also fails the per-index
        match."""
        spec = ExternalAttentionSpec(
            output="as",
            k_slot="l13.k",
            v_slot="l13.v",
            k_input="sliding_k",
            v_input="sliding_k",
            kv_heads=1,
            head_dim=256,
            capacity=SLIDING_CAPACITY,
            window=WINDOW,
        )

        with pytest.raises(StatesFormError, match=r"v given as 'sliding_k'"):
            to_external_states_form(source(), plan(spec))

    def test_an_already_operated_node_is_rejected(self):
        """A double surgery (already in the states form) is caught by the existing detection as
        is."""
        once = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        with pytest.raises(StatesFormError, match="already in the states form"):
            to_external_states_form(once, plan(SLIDING_SPEC))


class TestTheDeclarationsRoundTrip:
    def test_external_false_is_not_written(self):
        """The graph JSON of owned slots does not move by a single byte (the sha256 regression
        checks of existing assets do not move)."""
        slot = IrState(dtype="f32", shape=[1, 1, 8, 4])

        assert slot.to_dict() == {"dtype": "f32", "shape": [1, 1, 8, 4]}

    def test_a_shared_initializer_writes_no_tensor_key(self):
        """A shared declaration is `shared.tensor` only (it has no tensor key of its own)."""
        initializer = IrInitializer(
            shared=IrShared(tensor="model.lm_head.weight"), storage=IrStorage(dtype="i8")
        )

        assert initializer.to_dict() == {
            "shared": {"tensor": "model.lm_head.weight"},
            "storage": {"dtype": "i8"},
        }
        assert initializer.is_shared

    def test_the_operated_graph_survives_a_json_round_trip(self):
        """The round trip does not move the bytes (all 3 declarations have the same accepted set for
        reading and writing)."""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))
        text = graph.to_json()

        assert parse_ir_graph(text).to_json() == text


def _operated_json(**mutate) -> dict:
    """The JSON of the operated graph with one shared initializer added (the base for fault
    injection)."""
    graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC)).to_dict()
    graph["initializers"]["target_embed"] = {
        "shared": {"tensor": "model.lm_head.weight"},
        "storage": {"dtype": "i8"},
    }
    graph["values"]["target_embed"] = _value([16, 8])
    graph["nodes"] = [
        {
            "op": "embedding",
            "ins": ["target_embed", "tok"],
            "outs": ["e"],
            "attrs": {"padding_idx": -1},
        },
        *graph["nodes"],
    ]
    graph["inputs"] = [{"name": "tok", "dtype": "i32", "shape": [1, 1]}, *graph["inputs"]]
    graph["values"]["e"] = _value([1, 1, 8])
    graph["outputs"] = [*graph["outputs"], "e"]
    graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}
    graph.update(mutate)
    return graph


def _parse_and_check(graph: dict) -> IrGraph:
    parsed = parse_ir_graph(json.dumps(graph))
    assert_op_contracts(parsed)
    assert_runtime_support(parsed)
    return parsed


class TestTheGraphChecksAreEffective:
    """Fault injection — if any of the 4 passed, the IR would be "shapes and types match, only the
    values differ"."""

    def test_the_baseline_is_accepted(self):
        """The base itself passes (the control showing that the 4 below did not "fail from the
        start")."""
        assert _parse_and_check(_operated_json()).states["l13.k"].external is True

    def test_an_append_into_an_external_slot_is_rejected(self):
        graph = _operated_json()
        graph["nodes"] = [
            *graph["nodes"],
            {
                "op": "state_append",
                "ins": ["qs"],
                "outs": [],
                "attrs": {"window": WINDOW},
                "states": {"slot": "l13.k"},
            },
        ]
        graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}

        with pytest.raises(IrError, match="touches it in a form that is not readonly"):
            _parse_and_check(graph)

    def test_a_non_readonly_reader_of_an_external_slot_is_rejected(self):
        graph = _operated_json()
        for node in graph["nodes"]:
            if node.get("states", {}).get("k") == "l13.k":
                node["ins"] = ["qs", "sliding_k", "sliding_v"]
                node["attrs"] = {"scale": 1.0, "window": WINDOW}
        graph["inputs"] = [
            *graph["inputs"],
            _input("sliding_k", KV_SLIDING),
            _input("sliding_v", KV_SLIDING),
        ]

        with pytest.raises(IrError, match="touches it in a form that is not readonly"):
            _parse_and_check(graph)

    def test_a_readonly_reader_of_an_owned_slot_is_rejected(self):
        """The reverse — readonly reading an owned slot (a borrower without a lender)."""
        graph = _operated_json()
        for slot in graph["states"].values():
            slot.pop("external")

        with pytest.raises(IrError, match="owned slot"):
            _parse_and_check(graph)

    def test_mixing_external_and_owned_slots_is_rejected(self):
        graph = _operated_json()
        graph["states"]["l13.k"].pop("external")

        with pytest.raises(IrError, match="are mixed"):
            _parse_and_check(graph)

    def test_a_shared_initializer_with_a_tensor_key_is_rejected(self):
        graph = _operated_json()
        graph["initializers"]["target_embed"]["tensor"] = "drafter.embed"

        with pytest.raises(IrError, match="unknown key 'tensor'"):
            _parse_and_check(graph)

    @pytest.mark.parametrize("key", ["scale", "group_size"])
    def test_a_shared_initializer_with_a_companion_descriptor_is_rejected(self, key):
        """The companion tensor (scale) and the group step belong to the lender only (copying them
        gives 2 sources)."""
        graph = _operated_json()
        graph["initializers"]["target_embed"]["storage"][key] = (
            32 if key == "group_size" else "drafter.embed.scale"
        )

        with pytest.raises(IrError, match="a shared initializer cannot declare"):
            _parse_and_check(graph)

    def test_a_readonly_node_with_three_inputs_is_rejected(self):
        graph = _operated_json()
        for node in graph["nodes"]:
            if node.get("attrs", {}).get("readonly"):
                node["ins"] = ["qs", "qs", "qs"]

        with pytest.raises(OpContractError, match="readonly form takes exactly 1 input"):
            _parse_and_check(graph)

    def test_a_readonly_node_with_two_query_rows_is_rejected(self):
        """M = 1 MUST (stage 2 defines meaning only for "the one row at logical position P−1")."""
        graph = _operated_json()
        graph["inputs"] = [
            _input("qs", [1, 4, 2, 256]) if spec["name"] == "qs" else spec
            for spec in graph["inputs"]
        ]
        graph["values"]["as"] = _value([1, 4, 2, 256])
        graph["values"]["o"] = _value([1, 4, 2, 256])

        with pytest.raises(OpContractError, match=r"M \(axis 2\) of exactly 1"):
            _parse_and_check(graph)

    def test_a_slot_whose_head_dim_disagrees_with_q_is_rejected(self):
        """A slot mix-up (reading a slot of another kind) fails on the D axis."""
        graph = _operated_json()
        graph["states"]["l13.k"]["shape"] = [1, 1, SLIDING_CAPACITY, 512]
        graph["states"]["l13.v"]["shape"] = [1, 1, SLIDING_CAPACITY, 512]

        with pytest.raises(OpContractError, match="slot and q B / D mismatch"):
            _parse_and_check(graph)

    def test_a_window_wider_than_the_slot_capacity_is_rejected(self):
        graph = _operated_json()
        graph["states"]["l13.k"]["shape"] = [1, 1, 8, 256]
        graph["states"]["l13.v"]["shape"] = [1, 1, 8, 256]

        with pytest.raises(OpContractError, match="exceeds the slot capacity 8"):
            _parse_and_check(graph)


def _shared_container_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict]:
    """The minimal container with one shared initializer + one owned weight."""
    weight = (torch.arange(16, dtype=torch.float32) % 7 - 3).reshape(4, 4)
    scale = channel_scale(weight, 0)
    rounded = quantize_to_int8(weight, scale).to(torch.float32) * scale
    graph = IrGraph(
        inputs=[
            IrInput(name="x", dtype="f32", shape=[1, 4]),
            IrInput(name="tok", dtype="i32", shape=[1, 1]),
        ],
        outputs=["h", "e"],
        initializers={
            "target_embed": IrInitializer(
                shared=IrShared(tensor="model.lm_head.weight"), storage=IrStorage(dtype="i8")
            ),
            "w": IrInitializer(tensor="own.w", storage=IrStorage(dtype="f32")),
            "b": IrInitializer(tensor="own.b", storage=IrStorage(dtype="f32")),
        },
        values={
            "target_embed": IrValue(dtype="f32", shape=[16, 8]),
            "w": IrValue(dtype="f32", shape=[4, 4]),
            "b": IrValue(dtype="f32", shape=[4]),
            "h": IrValue(dtype="f32", shape=[1, 4]),
            "e": IrValue(dtype="f32", shape=[1, 1, 8]),
        },
        nodes=[
            IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={}),
            IrNode(
                op="embedding",
                ins=["target_embed", "tok"],
                outs=["e"],
                attrs={"padding_idx": -1},
            ),
        ],
    )
    tensors = {"own.w": rounded, "own.b": torch.zeros(4)}
    return graph, tensors, {"own.w": scale}


class TestTheContainerExcludesSharedDeclarations:
    def test_a_shared_declaration_needs_no_stored_tensor(self):
        """Exempt from the exact match of declarations and storage (the borrower's container holds
        no byte of it)."""
        graph, tensors, scales = _shared_container_graph()

        committed = stored_model(graph, tensors, weight_dtype="i8", weight_scales=scales).graph

        # The declared storage is written as is (neither planning nor conversion applies).
        assert committed.initializers["target_embed"].shared == IrShared(
            tensor="model.lm_head.weight"
        )
        assert committed.initializers["target_embed"].storage == IrStorage(dtype="i8")
        # Owned weights are planned to i8 as before (the control showing the exclusion is not
        # "let everything through").
        assert committed.initializers["w"].storage.dtype == "i8"

    def test_a_shared_tensor_key_present_in_the_container_is_surplus(self):
        """If the borrower holds bytes under the same name they are surplus (holding them twice,
        with the lender, is rejected)."""
        graph, tensors, scales = _shared_container_graph()
        tensors = {**tensors, "model.lm_head.weight": torch.zeros(16, 8)}

        with pytest.raises(EmitError, match="extra"):
            stored_model(graph, tensors, weight_dtype="i8", weight_scales=scales)


def dynamic_kv_source() -> IrGraph:
    """A borrower in which **only** the K / V placeholders bind the symbol T (q has a fixed shape of
    one row).

    The mask is a Tmax constant + `sym_prefix_slice` (the same shape as the leftovers of ADR 0010
    folding), and T is bound only by the K / V input shapes — when the surgery drops K / V, T loses
    its binding source.
    """
    return parse_ir_graph(
        json.dumps(
            {
                "format": "karume-ir",
                "version": 1,
                "requires": {"ops": ["attention", "sym_prefix_slice"]},
                "symbols": ["T"],
                "inputs": [
                    _input("qs", Q_SLIDING),
                    _input("sliding_k", [1, 1, "T", 256]),
                    _input("sliding_v", [1, 1, "T", 256]),
                ],
                "outputs": ["as"],
                "initializers": {
                    "mask.table": {"tensor": "drafter.mask", "storage": {"dtype": "f32"}},
                },
                "values": {
                    "mask.table": _value([1, 1, 1, 8]),
                    "mask": _value([1, 1, 1, "T"]),
                    "as": _value(Q_SLIDING),
                },
                "nodes": [
                    {
                        "op": "sym_prefix_slice",
                        "ins": ["mask.table"],
                        "outs": ["mask"],
                        "attrs": {"sym": "T", "slices": [{"dim": 3, "coeff": 1, "offset": 0}]},
                    },
                    {
                        "op": "attention",
                        "ins": ["qs", "sliding_k", "sliding_v", "mask"],
                        "outs": ["as"],
                        "attrs": {"scale": 1.0},
                    },
                ],
            }
        )
    )


class TestTheSymbolsAfterTheSurgery:
    """A symbol bound only by the dropped K / V disappears from `symbols` (the binding point of ADR
    0066 addendum (2026-08-17) item 7).

    Carrying the original `symbols` over as is would leave a symbol without a binding source, and
    the surgery would produce a graph that both the Python and the runtime parsers reject with
    "cannot be bound".
    """

    def test_a_symbol_bound_only_by_the_dropped_kv_is_removed(self):
        graph = checked(to_external_states_form(dynamic_kv_source(), plan(SLIDING_SPEC)))

        assert graph.symbols == []
        assert [spec.name for spec in graph.inputs] == ["qs"]

    def test_a_symbol_still_bound_by_a_kept_input_survives(self):
        """When q also carries T (a remaining input is the binding point), T stays (the control
        against dropping too much)."""
        source_graph = dynamic_kv_source()
        q_shape = [1, 4, "T", 256]
        source_graph.inputs[0] = IrInput(name="qs", dtype="f32", shape=q_shape)
        source_graph.values["as"] = IrValue(dtype="f32", shape=q_shape)

        graph = to_external_states_form(source_graph, plan(SLIDING_SPEC))

        assert graph.symbols == ["T"]
