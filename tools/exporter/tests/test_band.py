"""Pins the IR surgery that declares a band in place of an attention's mask (`karume.band`, ADR
0155), and the contract rules of `attrs.band` the table cannot express.

As for the states surgery, the acceptance criterion is "IR the runtime can read": the results go
through the round trip (`to_json` → `parse_ir_graph` → `assert_op_contracts` /
`assert_runtime_support`). The subject is a small synthetic graph (no torch) with the shape of a
sliding-window export: a folded `[1, 1, Tmax, Tmax]` band constant cut down by `sym_prefix_slice`
and shared by two attentions, plus one attention without a mask.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from karume.band import BandAttentionSpec, BandFormError, to_band_form
from karume.ir import IrGraph, IrNode
from karume.ops import OpContractError, assert_node_contract
from karume.verify import assert_op_contracts, assert_runtime_support, parse_ir_graph

#: q `[B,H,T,D]` (H = 4) and k / v `[B,Hkv,T,D]` (Hkv = 2 — GQA 2:1).
Q = [1, 4, "T", 8]
KV = [1, 2, "T", 8]
TMAX = 32


def _input(name: str, shape: Sequence) -> dict:
    return {"name": name, "dtype": "f32", "shape": list(shape)}


def _value(shape: Sequence) -> dict:
    return {"dtype": "f32", "shape": list(shape)}


def _attention(ins: Sequence[str], out: str) -> dict:
    return {"op": "attention", "ins": list(ins), "outs": [out], "attrs": {"scale": 0.5}}


#: The folding leftovers of the band mask (ADR 0010): the Tmax² constant cut down to T×T.
MASK_NODE = {
    "op": "sym_prefix_slice",
    "ins": ["band.table"],
    "outs": ["band"],
    "attrs": {
        "sym": "T",
        "slices": [
            {"dim": 2, "coeff": 1, "offset": 0},
            {"dim": 3, "coeff": 1, "offset": 0},
        ],
    },
}


def source_dict() -> dict:
    """Two sliding attentions sharing the band constant, one full attention without a mask, and a
    live initializer after them (pruning must not take it)."""
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add", "attention", "sym_prefix_slice"]},
        "symbols": ["T"],
        "inputs": [_input("q", Q), _input("k", KV), _input("v", KV)],
        "outputs": ["o"],
        "initializers": {
            "band.table": {"tensor": "band.table", "storage": {"dtype": "f32"}},
            "bias": {"tensor": "attn.bias", "storage": {"dtype": "f32"}},
        },
        "values": {
            "band.table": _value([1, 1, TMAX, TMAX]),
            "bias": _value([8]),
            "band": _value([1, 1, "T", "T"]),
            "s0": _value(Q),
            "s1": _value(Q),
            "f0": _value(Q),
            "a": _value(Q),
            "b": _value(Q),
            "o": _value(Q),
        },
        "nodes": [
            MASK_NODE,
            _attention(["q", "k", "v", "band"], "s0"),
            _attention(["q", "k", "v", "band"], "s1"),
            _attention(["q", "k", "v"], "f0"),
            {"op": "add", "ins": ["s0", "s1"], "outs": ["a"], "attrs": {}},
            {"op": "add", "ins": ["a", "f0"], "outs": ["b"], "attrs": {}},
            {"op": "add", "ins": ["b", "bias"], "outs": ["o"], "attrs": {}},
        ],
    }


def source() -> IrGraph:
    """The subject graph, built through the parser (the surgery assumes valid IR as input)."""
    return parse_ir_graph(json.dumps(source_dict()))


def checked(graph: IrGraph) -> IrGraph:
    """The round trip through the parser and the contract checks (the acceptance criterion)."""
    parsed = parse_ir_graph(graph.to_json())
    assert_op_contracts(parsed)
    assert_runtime_support(parsed)
    return parsed


SLIDING = (BandAttentionSpec(output="s0", band=4), BandAttentionSpec(output="s1", band=4))


class TestTheDeclaration:
    def test_the_mask_input_is_replaced_by_the_band_attr(self):
        graph = checked(to_band_form(source(), SLIDING))

        sliding = [node for node in graph.nodes if node.op == "attention" and node.outs != ["f0"]]
        assert [node.ins for node in sliding] == [["q", "k", "v"]] * 2
        assert [node.attrs for node in sliding] == [{"scale": 0.5, "band": 4}] * 2

    def test_an_attention_not_named_is_left_as_it_is(self):
        graph = checked(to_band_form(source(), SLIDING))

        full = next(node for node in graph.nodes if node.outs == ["f0"])
        assert full.ins == ["q", "k", "v"]
        assert full.attrs == {"scale": 0.5}

    def test_the_folded_band_constant_and_its_slice_are_pruned_once_no_attention_reads_them(self):
        graph = checked(to_band_form(source(), SLIDING))

        assert "sym_prefix_slice" not in [node.op for node in graph.nodes]
        assert list(graph.initializers) == ["bias"]
        assert "band" not in graph.values and "band.table" not in graph.values

    def test_a_mask_another_attention_still_reads_stays(self):
        """Naming one of the two readers keeps the constant (pruning follows liveness only)."""
        graph = checked(to_band_form(source(), SLIDING[:1]))

        assert "band.table" in graph.initializers
        reader = next(node for node in graph.nodes if node.outs == ["s1"])
        assert reader.ins == ["q", "k", "v", "band"]

    def test_the_order_symbols_and_inputs_are_kept(self):
        graph = checked(to_band_form(source(), SLIDING))

        assert [node.outs[0] for node in graph.nodes] == ["s0", "s1", "f0", "a", "b", "o"]
        assert graph.symbols == ["T"]
        assert [spec.name for spec in graph.inputs] == ["q", "k", "v"]

    def test_an_integral_float_band_is_written_as_the_integer(self):
        graph = checked(to_band_form(source(), (BandAttentionSpec(output="s0", band=4.0),)))

        band = next(node for node in graph.nodes if node.outs == ["s0"]).attrs["band"]
        assert band == 4 and type(band) is int

    def test_a_band_of_zero_is_a_declaration(self):
        graph = checked(to_band_form(source(), (BandAttentionSpec(output="s0", band=0),)))

        assert next(node for node in graph.nodes if node.outs == ["s0"]).attrs["band"] == 0

    def test_the_source_graph_is_left_untouched(self):
        graph = source()
        before = graph.to_json()

        to_band_form(graph, SLIDING)

        assert graph.to_json() == before

    def test_no_spec_returns_the_same_graph(self):
        graph = source()

        assert to_band_form(graph, ()).to_json() == graph.to_json()


class TestRefusals:
    def test_an_attention_without_a_mask(self):
        with pytest.raises(BandFormError, match="only the four including the mask"):
            to_band_form(source(), (BandAttentionSpec(output="f0", band=4),))

    def test_a_node_that_is_not_attention(self):
        with pytest.raises(BandFormError, match="only 'attention'"):
            to_band_form(source(), (BandAttentionSpec(output="a", band=4),))

    def test_a_value_no_node_defines(self):
        with pytest.raises(BandFormError, match="no node in the graph defines"):
            to_band_form(source(), (BandAttentionSpec(output="missing", band=4),))

    def test_a_node_named_twice(self):
        with pytest.raises(BandFormError, match="named twice"):
            to_band_form(source(), (SLIDING[0], SLIDING[0]))

    def test_a_double_surgery(self):
        once = to_band_form(source(), SLIDING)
        nodes = list(once.nodes)
        index = next(at for at, node in enumerate(nodes) if node.outs == ["s0"])
        nodes[index] = IrNode(
            op="attention", ins=["q", "k", "v", "x"], outs=["s0"], attrs=dict(nodes[index].attrs)
        )
        twice = IrGraph(
            symbols=once.symbols,
            inputs=once.inputs,
            outputs=once.outputs,
            initializers=once.initializers,
            values=once.values,
            nodes=nodes,
        )

        with pytest.raises(BandFormError, match="already declares band"):
            to_band_form(twice, (SLIDING[0],))

    @pytest.mark.parametrize("band", [-1, 1.5, True, "4", 2**53])
    def test_a_band_outside_the_contract_range(self, band):
        with pytest.raises(BandFormError, match="not a non-negative integer"):
            to_band_form(source(), (BandAttentionSpec(output="s0", band=band),))

    def test_a_mask_built_from_a_graph_input(self):
        """A run-time mask cannot become a band."""
        graph = _with_mask_input(source_dict())

        with pytest.raises(
            BandFormError, match=r"attention 's0': its mask depends on inputs \['mask'\]"
        ):
            to_band_form(parse_ir_graph(json.dumps(graph)), SLIDING)

    def test_a_mask_input_another_node_also_reads(self):
        """The refusal does not rest on the input being orphaned: a padding mask an unnamed
        attention shares is refused too."""
        graph = _with_mask_input(source_dict())
        graph["nodes"][2]["ins"] = ["q", "k", "v", "mask"]

        with pytest.raises(
            BandFormError, match=r"attention 's0': its mask depends on inputs \['mask'\]"
        ):
            to_band_form(parse_ir_graph(json.dumps(graph)), SLIDING[:1])

    def test_a_mask_derived_from_an_input_through_an_intermediate_node(self):
        graph = _with_mask_input(source_dict())
        graph["values"]["mask.sum"] = _value([1, 1, "T", "T"])
        graph["nodes"].insert(
            0, {"op": "add", "ins": ["mask", "mask"], "outs": ["mask.sum"], "attrs": {}}
        )
        graph["nodes"][1]["ins"][3] = "mask.sum"
        graph["nodes"][2]["ins"][3] = "mask.sum"

        with pytest.raises(
            BandFormError, match=r"attention 's0': its mask depends on inputs \['mask'\]"
        ):
            to_band_form(parse_ir_graph(json.dumps(graph)), SLIDING)

    def test_a_mask_built_from_constants_only_passes_the_walk(self):
        """The folded constant cut down by `sym_prefix_slice` (a symbol in attrs, no input) is the
        form the surgery exists for."""
        graph = checked(to_band_form(source(), SLIDING))

        assert [spec.name for spec in graph.inputs] == ["q", "k", "v"]

    def test_an_input_no_node_reads_after_the_surgery(self):
        """A generic wiring check, whatever made the input unread."""
        graph = source_dict()
        graph["inputs"].append(_input("unused", [1, "T"]))

        with pytest.raises(BandFormError, match=r"Inputs \['unused'\] are read by no node"):
            to_band_form(parse_ir_graph(json.dumps(graph)), SLIDING)


def _with_mask_input(graph: dict) -> dict:
    """`graph` (a `source_dict()`) with its two sliding attentions reading a graph input `mask` in
    place of the folded band constant."""
    graph["inputs"].append(_input("mask", [1, 1, "T", "T"]))
    graph["nodes"][1]["ins"][3] = "mask"
    graph["nodes"][2]["ins"][3] = "mask"
    graph["nodes"] = graph["nodes"][1:]
    del graph["values"]["band"], graph["values"]["band.table"]
    del graph["initializers"]["band.table"]
    graph["requires"] = {"ops": ["add", "attention"]}
    return graph


class TestTheContractOfBand:
    """The node-level rules of `attrs.band` (the mirror of the TS side's
    ops_conformance_test.ts)."""

    @staticmethod
    def _node(ins: Sequence[str], attrs: dict, states: dict | None = None) -> IrNode:
        return IrNode(op="attention", ins=list(ins), outs=["y"], attrs=attrs, states=states or {})

    def test_the_plain_form_takes_it(self):
        assert_node_contract(self._node(["q", "k", "v"], {"scale": 0.5, "band": 512}), "t")
        assert_node_contract(self._node(["q", "k", "v"], {"scale": 0.5, "band": 0}), "t")

    def test_never_beside_the_mask(self):
        with pytest.raises(OpContractError, match="cannot be declared together with the optional"):
            assert_node_contract(self._node(["q", "k", "v", "m"], {"scale": 0.5, "band": 4}), "t")

    def test_not_on_the_states_form(self):
        states = {"k": "kv.k", "v": "kv.v"}
        attrs = {"scale": 0.5, "band": 4}
        with pytest.raises(OpContractError, match="only on a node without a states field"):
            assert_node_contract(self._node(["q", "k", "v"], attrs, states), "t")

    def test_not_on_the_readonly_form(self):
        states = {"k": "kv.k", "v": "kv.v"}
        attrs = {"scale": 0.5, "readonly": True, "band": 4}
        with pytest.raises(OpContractError, match="only on a node without a states field"):
            assert_node_contract(self._node(["q"], attrs, states), "t")

    def test_the_range_applies_at_the_node_level(self):
        with pytest.raises(OpContractError, match=r"attrs\.band"):
            assert_node_contract(self._node(["q", "k", "v"], {"scale": 0.5, "band": -1}), "t")
