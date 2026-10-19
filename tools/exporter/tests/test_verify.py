"""Pins the IR's accepted set (the rules of a graph alone) and the safetensors layout rules.

If the exporter-side verification accepted a wider set than the runtime
(packages/runtime/src/format/ir.ts), the two would diverge as "export green, fails only in the
browser". Here each rule gets "a form that is accepted" and "a form that fails loudly" side by side.

NOTE: **the match against the container (`krm`) lives in `test_verify_container.py`** (the rules
that need declared shapes, piece sequences, the quantization steps, the scale lengths). The check
that read back the old distribution (safetensors shard sequences) was retired from here — only the
migration CLI (`karume.legacy`) reads it now, so it moved to `test_migrate.py` together with its
subject.
"""

from __future__ import annotations

import json
import re

import pytest

from karume.ops import OpContractError, permute_dims, state_window
from karume.verify import (
    ContainerError,
    IrError,
    assert_op_contracts,
    assert_reader_layout,
    assert_runtime_support,
    parse_ir_graph,
)


def base_graph() -> dict:
    """The minimal valid graph: one symbolic dimension, one initializer, one node."""
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add"]},
        "symbols": ["T"],
        "inputs": [{"name": "x", "dtype": "f32", "shape": ["T", 4]}],
        "outputs": ["y"],
        "initializers": {"w": {"tensor": "enc.w", "storage": {"dtype": "f32"}}},
        "values": {
            "w": {"dtype": "f32", "shape": [4]},
            "y": {"dtype": "f32", "shape": ["T", 4]},
        },
        "nodes": [{"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
    }


def parse(**overrides):
    return parse_ir_graph(json.dumps({**base_graph(), **overrides}))


class TestBaseline:
    def test_the_minimal_graph_is_accepted(self):
        graph = parse()

        assert graph.symbols == ["T"]
        assert graph.required_ops == ["add"]
        assert graph.values["y"].shape == ["T", 4]


class TestFormatAndVersion:
    def test_foreign_format_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.format")):
            parse(format="other-ir")

    def test_other_versions_are_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.version")):
            parse(version=2)

    def test_unknown_top_level_key_is_rejected(self):
        """Unreleased, so there is no forward-compatibility channel — unknown keys are not silently
        ignored."""
        with pytest.raises(IrError, match="unknown key"):
            parse(metadata={})

    def test_missing_top_level_key_is_rejected(self):
        graph = base_graph()
        del graph["symbols"]

        with pytest.raises(IrError, match="required key"):
            parse_ir_graph(json.dumps(graph))


class TestRequiredOps:
    def test_undeclared_op_is_rejected(self):
        with pytest.raises(IrError, match="undeclared"):
            parse(requires={"ops": []})

    def test_surplus_declaration_is_rejected(self):
        with pytest.raises(IrError, match="surplus"):
            parse(requires={"ops": ["add", "mul"]})

    def test_duplicate_declaration_is_rejected(self):
        with pytest.raises(IrError, match="is duplicated"):
            parse(requires={"ops": ["add", "add"]})


class TestSymbols:
    def test_symbol_absent_from_the_declaration_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("is not declared in graph.symbols")):
            parse(
                inputs=[{"name": "x", "dtype": "f32", "shape": ["U", 4]}],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["U", 4]},
                },
            )

    def test_symbol_never_appearing_in_an_input_is_rejected(self):
        """Binding comes from dimension positions of input shapes — a symbol that appears only in
        values cannot be bound."""
        with pytest.raises(IrError, match="appears in no dimension position"):
            parse(
                inputs=[{"name": "x", "dtype": "f32", "shape": [8, 4]}],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                },
            )

    def test_a_derived_dim_is_a_binding_source(self):
        """A derived form (`2T`) has a unique solution too, so it is a binding source (ADR 0057)."""
        graph = parse(
            inputs=[{"name": "x", "dtype": "f32", "shape": ["2T", 4]}],
            values={
                "w": {"dtype": "f32", "shape": [4]},
                "y": {"dtype": "f32", "shape": ["2T", 4]},
            },
        )

        assert graph.symbols == ["T"]

    def test_symbol_name_outside_the_grammar_is_rejected(self):
        with pytest.raises(IrError, match="symbol name"):
            parse(symbols=["2T"])

    def test_non_canonical_dim_spelling_is_rejected(self):
        with pytest.raises(IrError, match="canonical grammar"):
            parse(
                inputs=[{"name": "x", "dtype": "f32", "shape": ["T", 4]}],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["1T", 4]},
                },
            )

    def test_negative_dimension_is_rejected(self):
        with pytest.raises(IrError, match="is not a non-negative integer"):
            parse(
                values={
                    "w": {"dtype": "f32", "shape": [-4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                }
            )


class TestSsaAndTopologicalOrder:
    def test_double_definition_is_rejected(self):
        with pytest.raises(IrError, match="violates SSA single assignment"):
            parse(
                nodes=[
                    {"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {}},
                    {"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {}},
                ]
            )

    def test_forward_reference_is_rejected(self):
        with pytest.raises(IrError, match="forward reference"):
            parse(
                outputs=["y", "z"],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                    "z": {"dtype": "f32", "shape": ["T", 4]},
                },
                nodes=[
                    {"op": "add", "ins": ["x", "z"], "outs": ["y"], "attrs": {}},
                    {"op": "add", "ins": ["x", "w"], "outs": ["z"], "attrs": {}},
                ],
            )

    def test_undefined_output_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.outputs")):
            parse(outputs=["ghost"])

    def test_duplicate_output_is_rejected(self):
        with pytest.raises(IrError, match="is duplicated"):
            parse(outputs=["y", "y"])

    def test_empty_outs_is_rejected_by_the_contract_layer(self):
        """Zero outputs (ADR 0067 decision 5) — the parser gives no meaning to the number of `outs`.

        "Only ops whose contract declares an effect may have zero outputs" is enforced by the
        contract layer's output-count match. The rejection from the days when the parser looked
        at the count has been replaced by this one.
        """
        graph = parse(
            outputs=[],
            values={"w": {"dtype": "f32", "shape": [4]}},
            nodes=[{"op": "add", "ins": ["x", "w"], "outs": [], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="output count is 0"):
            assert_op_contracts(graph)


class TestDeclarationCompleteness:
    def test_input_declared_again_in_values_is_rejected(self):
        with pytest.raises(IrError, match="declared twice"):
            parse(
                values={
                    "x": {"dtype": "f32", "shape": ["T", 4]},
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                }
            )

    def test_orphan_declaration_is_rejected(self):
        with pytest.raises(IrError, match="a declaration no node defines"):
            parse(
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                    "ghost": {"dtype": "f32", "shape": [1]},
                }
            )

    def test_node_output_without_a_declaration_is_rejected(self):
        with pytest.raises(IrError, match="node output"):
            parse(values={"w": {"dtype": "f32", "shape": [4]}})

    def test_initializer_without_a_declaration_is_rejected(self):
        with pytest.raises(IrError, match="no dtype/shape declaration in values"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32"}},
                    "extra": {"tensor": "enc.extra", "storage": {"dtype": "f32"}},
                }
            )

    def test_i32_initializer_pairs_with_i32_storage(self):
        """Where symbol-dependent constants are baked (ADR 0010) — semantic i32 pairs with raw int32
        storage."""
        graph = parse(
            requires={"ops": ["mul"]},
            inputs=[{"name": "x", "dtype": "i32", "shape": ["T", 4]}],
            initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "i32"}}},
            values={
                "w": {"dtype": "i32", "shape": [4]},
                "y": {"dtype": "i32", "shape": ["T", 4]},
            },
            nodes=[{"op": "mul", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
        )

        assert graph.initializers["w"].storage.dtype == "i32"

    @pytest.mark.parametrize(
        ("semantic", "storage"),
        [("i32", "f32"), ("f32", "i32"), ("f32", "f16")],
        ids=["i32-as-f32", "f32-as-i32", "f32-as-f16-is-fine"],
    )
    def test_crossed_semantic_and_storage_dtypes_are_rejected(self, semantic, storage):
        """The f32 storage vocabulary (f32/f16/bf16/i8) and raw int32 do not cross.

        Allowing a cross would read an initializer declared i32 as f16 bits — a silently wrong
        value. Only f32 × f16 is a valid pair (accepted as a declaration; whether it runs is the
        concern of the support-table layer).
        """

        def build():
            return parse(
                requires={"ops": ["mul"]},
                inputs=[{"name": "x", "dtype": semantic, "shape": ["T", 4]}],
                initializers={"w": {"tensor": "enc.w", "storage": {"dtype": storage}}},
                values={
                    "w": {"dtype": semantic, "shape": [4]},
                    "y": {"dtype": semantic, "shape": ["T", 4]},
                },
                nodes=[{"op": "mul", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
            )

        if (semantic, storage) == ("f32", "f16"):
            assert build().initializers["w"].storage.dtype == "f16"
            return
        with pytest.raises(IrError, match="cannot pair with"):
            build()

    def test_bool_initializer_is_rejected(self):
        with pytest.raises(IrError, match="is outside the vocabulary"):
            parse(
                requires={"ops": ["bitwise_not"]},
                inputs=[{"name": "x", "dtype": "bool", "shape": ["T", 4]}],
                outputs=["y"],
                initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "f32"}}},
                values={
                    "w": {"dtype": "bool", "shape": [4]},
                    "y": {"dtype": "bool", "shape": ["T", 4]},
                },
                nodes=[{"op": "bitwise_not", "ins": ["x"], "outs": ["y"], "attrs": {}}],
            )

    def test_symbolic_initializer_shape_is_rejected(self):
        """An initializer not fixed before binding cannot be matched against the safetensors-side
        shape."""
        with pytest.raises(IrError, match="cannot use a symbolic dimension"):
            parse(
                values={
                    "w": {"dtype": "f32", "shape": ["T"]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                }
            )


class TestEmptyDeclarationNames:
    """An empty value name cannot be declared (mirror of the TS side,
    packages/runtime/src/format/ir.ts).

    The referencing side (input names, node inputs and outputs, graph outputs) accepts only
    non-empty strings, so a declaration with an empty name is a value that can never be referenced.
    `states` already had this check.
    """

    def test_an_empty_value_name_is_rejected(self):
        with pytest.raises(IrError, match=r"value name of graph\.values"):
            parse(
                values={
                    "": {"dtype": "f32", "shape": [1]},
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                }
            )

    def test_an_empty_initializer_name_is_rejected(self):
        with pytest.raises(IrError, match=r"initializer name of graph\.initializers"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32"}},
                    "": {"tensor": "enc.empty", "storage": {"dtype": "f32"}},
                }
            )

    def test_the_pair_that_neutralizes_both_orphan_checks_is_rejected(self):
        """An empty-named `values` entry and an empty-named `initializers` entry written **as a
        pair**.

        The orphan-declaration checks satisfy each other and do not fire (on the values side "an
        initializer defines it", on the initializer side "values has a declaration"), and the
        surplus-tensor check stays quiet because the initializer references it — without the
        empty-name check, one unreferenceable real tensor would sit in the distribution while
        `karume verify` stays green.
        """
        with pytest.raises(IrError, match="is not a non-empty string"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32"}},
                    "": {"tensor": "enc.empty", "storage": {"dtype": "f32"}},
                },
                values={
                    "": {"dtype": "f32", "shape": [2]},
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                },
            )


class TestStorageDescriptor:
    def test_scale_on_a_non_quantized_storage_is_rejected(self):
        with pytest.raises(IrError, match="cannot have scale / group_size"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32", "scale": "enc.s"}}
                }
            )

    def test_group_size_on_a_non_quantized_storage_is_rejected(self):
        with pytest.raises(IrError, match="cannot have scale / group_size"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "bf16", "group_size": 32}}
                }
            )

    def test_non_positive_group_size_is_rejected(self):
        with pytest.raises(IrError, match="is not a positive integer"):
            parse(
                initializers={
                    "w": {
                        "tensor": "enc.w",
                        "storage": {"dtype": "i8", "scale": "enc.s", "group_size": 0},
                    }
                }
            )

    def test_group_size_beyond_the_safe_integer_range_is_rejected(self):
        """The TS side reads JSON numbers as JS numbers, so 2^53 and above cannot be held as
        integers."""
        with pytest.raises(IrError, match="safe integer"):
            parse(
                initializers={
                    "w": {
                        "tensor": "enc.w",
                        "storage": {"dtype": "i8", "scale": "enc.s", "group_size": 2**53},
                    }
                }
            )

    def test_i8_without_a_scale_is_rejected(self):
        """MUST: i8 declares its scale explicitly
        (ADR 0019; mirror of the TS side, packages/runtime/src/format/ir.ts).

        Filling in a default of 1.0 would turn an omission into "weights dequantized with 1.0 for
        every channel", and both loading and running would pass (the difference is O(scale), yet
        no exception appears anywhere).
        """
        with pytest.raises(IrError, match=r"requires scale \(the scale tensor's key\)"):
            parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "i8"}}})

    def test_unknown_storage_key_is_rejected(self):
        with pytest.raises(IrError, match="unknown key"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32", "zero_point": 1}}
                }
            )

    def test_unknown_storage_dtype_is_rejected(self):
        with pytest.raises(IrError, match="is outside the vocabulary"):
            parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "i1"}}})


class TestGroupQuantizedStorage:
    """Declaration rules of group-quantized storage (i4 — ADR 0069 decision 2).

    scale and group_size are required, and the group length MUST be a power of 2 and 16 or more,
    with the quantized axis (last dimension) divisible by it (the constraint of never making a
    remainder group guarantees the byte alignment of row and group boundaries). Mirror of the TS
    side, `packages/runtime/src/format/ir.ts`.
    """

    def _i4(self, storage: dict, last_dim: int = 32) -> dict:
        return {
            "initializers": {"w": {"tensor": "enc.w", "storage": storage}},
            "values": {
                "w": {"dtype": "f32", "shape": [4, last_dim]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
            },
        }

    def test_a_well_formed_declaration_is_accepted(self):
        graph = parse(**self._i4({"dtype": "i4", "scale": "enc.s", "group_size": 32}))

        assert graph.initializers["w"].storage.group_size == 32
        assert graph.initializers["w"].storage.scale == "enc.s"

    def test_a_missing_group_size_is_rejected(self):
        with pytest.raises(IrError, match="requires group_size"):
            parse(**self._i4({"dtype": "i4", "scale": "enc.s"}))

    def test_a_missing_scale_is_rejected(self):
        with pytest.raises(IrError, match=r"requires scale \(the scale tensor's key\)"):
            parse(**self._i4({"dtype": "i4", "group_size": 32}))

    @pytest.mark.parametrize("group_size", [24, 8], ids=["not-a-power-of-two", "below-16"])
    def test_a_group_size_outside_the_accepted_set_is_rejected(self, group_size):
        with pytest.raises(IrError, match="is not a power of 2 that is 16 or more"):
            parse(**self._i4({"dtype": "i4", "scale": "enc.s", "group_size": group_size}))

    def test_a_quantization_axis_that_the_group_size_does_not_divide_is_rejected(self):
        with pytest.raises(IrError, match="is not divisible"):
            parse(**self._i4({"dtype": "i4", "scale": "enc.s", "group_size": 32}, last_dim=48))

    def test_a_shared_declaration_is_accepted_without_a_group_size(self):
        """A shared initializer cannot declare group_size (the lender is the source of truth for the
        step — ADR 0096 stage 2).

        Since the declaration side says so, if the declaration completeness check failed on "i4
        without group_size", a shared declaration lending an i4 embedding could never be written.
        """
        graph = parse(
            requires={"ops": ["embedding"]},
            inputs=[{"name": "ids", "dtype": "i32", "shape": ["T"]}],
            initializers={"x": {"shared": {"tensor": "t"}, "storage": {"dtype": "i4"}}},
            values={
                "x": {"dtype": "f32", "shape": [16, 32]},
                "y": {"dtype": "f32", "shape": ["T", 32]},
            },
            nodes=[
                {
                    "op": "embedding",
                    "ins": ["x", "ids"],
                    "outs": ["y"],
                    "attrs": {"padding_idx": -1},
                }
            ],
        )

        assert graph.initializers["x"].shared is not None
        assert graph.initializers["x"].shared.tensor == "t"


def declared_states(states: dict, *, referenced: list[str] | None = None) -> dict:
    """The overrides of a graph with `state_append` nodes added that reference the slots.

    Since reference completeness (ADR 0067 decisions 4 / 5 — a declared slot MUST be referenced by
    at least one node) went in, a graph that "declares but nobody references" is no longer
    accepted. Tests that look at the accepted set of the slot declaration itself move one point
    on top of this scaffold.

    Passing `referenced` narrows the referenced slots (to exercise the orphan-slot detection).
    """
    names = list(states) if referenced is None else referenced
    nodes = [
        *base_graph()["nodes"],
        *(
            {"op": "state_append", "ins": ["x"], "outs": [], "attrs": {}, "states": {"slot": name}}
            for name in names
        ),
    ]
    return {
        "requires": {"ops": ["add", "state_append"] if names else ["add"]},
        "states": states,
        "nodes": nodes,
    }


class TestStateSlots:
    """Named state slots (ADR 0066 decision 2) — the accepted set on the declaration side.

    References from nodes (ADR 0067's states field / `state_append`) belong to TestNodeStates.
    Here the shape is "satisfy the references with the minimal scaffold, then move one point of
    the declaration".
    """

    def test_a_graph_without_the_section_has_no_slots(self):
        """A graph without the section (= every existing model) is unaffected."""
        graph = parse()

        assert graph.states == {}
        # MUST: empty states are not written back — writing them would change the graph JSON of
        # existing models byte for byte.
        assert "states" not in graph.to_dict()

    def test_an_empty_section_is_accepted(self):
        graph = parse(states={})

        assert graph.states == {}
        assert "states" not in graph.to_dict()

    def test_slots_with_different_capacities_are_accepted(self):
        """Sliding layers and full layers are separate slots of different capacities (no premise of
        layers × uniform KV)."""
        graph = parse(
            **declared_states(
                {
                    "layer0.k": {"dtype": "f32", "shape": [1, 2, 512, 128]},
                    "layer0.v": {"dtype": "f32", "shape": [1, 2, 512, 128]},
                    "layer1.k": {"dtype": "f32", "shape": [1, 2, 131072, 128]},
                }
            )
        )

        assert list(graph.states) == ["layer0.k", "layer0.v", "layer1.k"]
        assert graph.states["layer0.k"].shape == [1, 2, 512, 128]
        assert graph.states["layer1.k"].shape == [1, 2, 131072, 128]

    @pytest.mark.parametrize("shape", [[8], [1, 2, 3, 4], ["T", 4]])
    def test_accepted_shapes(self, shape):
        """Both ends of rank 1..4 and a symbolic dimension (the binding source is a dimension
        position of an input shape or a states shape)."""
        graph = parse(**declared_states({"cache": {"dtype": "f32", "shape": shape}}))

        assert graph.states["cache"].shape == shape

    def test_the_section_round_trips_in_place(self):
        """A declaration section, so it is written back right after `values` and before `nodes` (the
        same order as the TS side)."""
        slot = {"dtype": "f32", "shape": [1, 2, 512, 128]}
        graph = parse(**declared_states({"layer0.k": slot}))
        keys = list(graph.to_dict())

        assert graph.to_dict()["states"] == {"layer0.k": slot}
        assert keys.index("states") == keys.index("values") + 1
        assert keys.index("nodes") == keys.index("states") + 1

    def test_a_slot_that_no_node_references_is_rejected(self):
        """Reference completeness (ADR 0067 decisions 4 / 5) — closes the same hole as orphan
        declarations in values.

        A slot nobody references silently leaves capacity that the GenerationContext allocates
        and nobody reads (tens of MiB for KV).
        """
        with pytest.raises(
            IrError, match=re.escape("graph.states['orphan']: a declaration no node references")
        ):
            parse(
                **declared_states(
                    {
                        "used": {"dtype": "f32", "shape": [4]},
                        "orphan": {"dtype": "f32", "shape": [4]},
                    },
                    referenced=["used"],
                )
            )

    def test_the_reserved_f16_storage_is_rejected_as_unsupported(self):
        """f16 is only a reserved option (ADR 0066 addendum of 2026-08-17, item 5) — it fails as
        "not supported", not as "outside the vocabulary"."""
        with pytest.raises(IrError, match="is not supported"):
            parse(states={"cache": {"dtype": "f16", "shape": [4]}})

    @pytest.mark.parametrize("dtype", ["i32", "bool", "bf16", "i8", "f64"])
    def test_other_dtypes_are_outside_the_vocabulary(self, dtype):
        with pytest.raises(IrError, match=re.escape("is outside the vocabulary (f32)")):
            parse(states={"cache": {"dtype": dtype, "shape": [4]}})

    @pytest.mark.parametrize("shape", [[1, 1, 1, 1, 1], []])
    def test_rank_outside_one_to_four_is_rejected(self, shape):
        with pytest.raises(IrError, match=re.escape("is outside 1..4")):
            parse(states={"cache": {"dtype": "f32", "shape": shape}})

    def test_zero_capacity_is_rejected(self):
        """A concrete shape including the capacity, so numeric dimensions are positive integers
        (unlike the non-negative ones of `values`)."""
        with pytest.raises(IrError, match="is not a positive integer"):
            parse(states={"cache": {"dtype": "f32", "shape": [1, 0, 4]}})

    def test_negative_dimension_is_rejected(self):
        with pytest.raises(IrError, match="is not a non-negative integer"):
            parse(states={"cache": {"dtype": "f32", "shape": [-4]}})

    def test_undeclared_symbol_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("is not declared in graph.symbols")):
            parse(states={"cache": {"dtype": "f32", "shape": ["S", 4]}})

    def test_non_canonical_dim_spelling_is_rejected(self):
        with pytest.raises(IrError, match="canonical grammar"):
            parse(states={"cache": {"dtype": "f32", "shape": ["1T", 4]}})

    @pytest.mark.parametrize(
        ("states", "message"),
        [
            ([], "graph.states: is not an object"),
            ({"cache": 4}, "graph.states['cache']: is not an object"),
            ({"cache": {"dtype": "f32", "shape": 4}}, "shape: is not an array"),
            ({"cache": {"shape": [4]}}, "required key 'dtype' is missing"),
            ({"cache": {"dtype": "f32", "shape": [4], "window": 512}}, "unknown key 'window'"),
            # An empty slot name is not accepted by the referencing fields (ADR 0067) = a
            # declaration that cannot be referenced.
            (
                {"": {"dtype": "f32", "shape": [4]}},
                "slot name of graph.states: is not a non-empty string",
            ),
        ],
    )
    def test_malformed_sections_are_rejected(self, states, message):
        with pytest.raises(IrError, match=re.escape(message)):
            parse(states=states)

    @pytest.mark.parametrize("name", ["x", "w", "y"])
    def test_a_slot_named_after_a_value_is_rejected(self, name):
        """Slot names are a namespace separate from values, but the same name is rejected as a
        detection line for mix-ups (the 3 roles input / initializer / node output).
        """
        with pytest.raises(
            IrError, match=re.escape(f"graph.states['{name}']: same name as a value")
        ):
            parse(states={name: {"dtype": "f32", "shape": [4]}})


def kv_graph(**overrides) -> dict:
    """The minimal graph of states-form attention + `state_append` (one decode step, one layer).

    `q[1,8,1,16]` / `k`, `v[1,2,1,16]` (GQA 4:1) and 2 slots of capacity 64. The ins are **only
    this step's chunk**; the past is held by the slots (ADR 0067 decision 4). The emission order
    is "every reader → append" (decision 5b).
    """
    graph = {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["attention", "state_append"]},
        "symbols": [],
        "inputs": [
            {"name": "q", "dtype": "f32", "shape": [1, 8, 1, 16]},
            {"name": "k", "dtype": "f32", "shape": [1, 2, 1, 16]},
            {"name": "v", "dtype": "f32", "shape": [1, 2, 1, 16]},
        ],
        "outputs": ["o"],
        "initializers": {},
        "values": {"o": {"dtype": "f32", "shape": [1, 8, 1, 16]}},
        "states": {
            "l0.k": {"dtype": "f32", "shape": [1, 2, 64, 16]},
            "l0.v": {"dtype": "f32", "shape": [1, 2, 64, 16]},
        },
        "nodes": [
            {
                "op": "attention",
                "ins": ["q", "k", "v"],
                "outs": ["o"],
                "attrs": {"scale": 0.25},
                "states": {"k": "l0.k", "v": "l0.v"},
            },
            {
                "op": "state_append",
                "ins": ["k"],
                "outs": [],
                "attrs": {},
                "states": {"slot": "l0.k"},
            },
            {
                "op": "state_append",
                "ins": ["v"],
                "outs": [],
                "attrs": {},
                "states": {"slot": "l0.v"},
            },
        ],
    }
    graph.update(overrides)
    # requires.ops ≡ the set of ops used (derived from the nodes unless given explicitly — so
    # that tests that replace nodes do not have to rewrite the declaration).
    if "requires" not in overrides:
        graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}
    return graph


def parse_kv(**overrides):
    return parse_ir_graph(json.dumps(kv_graph(**overrides)))


def check_kv(**overrides):
    """Run parse → the contract checks (arity / states field / order / shape)."""
    graph = parse_kv(**overrides)
    assert_op_contracts(graph)
    return graph


def kv_windows(*windows: int | None) -> list[dict]:
    """The node list with only the `window` of the 3 nodes (attention / append k / append v)
    replaced."""
    nodes = [dict(node) for node in kv_graph()["nodes"]]
    for node, window in zip(nodes, windows, strict=True):
        node["attrs"] = dict(node["attrs"])
        if window is not None:
            node["attrs"]["window"] = window
    return nodes


def plain_attention_graph(attrs: dict) -> dict:
    """The minimal graph of **classic-form** attention without a states field (the control for the
    contract that splits on the field's presence)."""
    graph = kv_graph()
    del graph["states"]
    graph["requires"] = {"ops": ["attention"]}
    graph["nodes"] = [{"op": "attention", "ins": ["q", "k", "v"], "outs": ["o"], "attrs": attrs}]
    return graph


class TestNodeStates:
    """The `states` field of nodes and `state_append` (ADR 0067 decisions 4 / 5 / 5b).

    The specification here is **the same accepted set and the same rejected set** as the TS side
    (parseNodeStates in format/ir.ts, assertStateField in ops/contracts.ts, assertStateOrder in
    runtime/plan.ts, assertStateSlotForm in ops/shapes.ts). Only the layers differ; if one side
    loosened, "export green, fails only in the browser" would come back on the state axis.
    """

    def test_a_decode_step_graph_is_accepted(self):
        graph = check_kv()

        assert graph.nodes[0].states == {"k": "l0.k", "v": "l0.v"}
        assert graph.nodes[1].states == {"slot": "l0.k"}
        assert assert_runtime_support(graph) is None

    def test_the_field_round_trips_only_when_present(self):
        """The presence of the field decides the form (ADR 0067 decision 4) — an empty field is not
        written back."""
        node = parse_kv().nodes[0].to_dict()

        assert list(node) == ["op", "ins", "outs", "attrs", "states"]
        assert node["states"] == {"k": "l0.k", "v": "l0.v"}
        assert "states" not in parse().nodes[0].to_dict()

    def test_a_layer_without_an_append_is_accepted(self):
        """A KV-sharing layer is expressed simply by "no `state_append` node" (the equivalent of
        ORT's kv_empty)."""
        graph = check_kv(nodes=kv_graph()["nodes"][:1])

        assert [node.op for node in graph.nodes] == ["attention"]

    def test_a_sliding_layer_is_accepted(self):
        """The optional attr `window` (absence of the field = the whole context). Reads and writes
        declare the same value."""
        graph = check_kv(nodes=kv_windows(64, 64, 64))

        assert graph.nodes[0].attrs["window"] == 64
        # The enumeration check (capability match) also looks at the **union** of required and
        # optional attrs — dropping it would reject a correct states-form graph as
        # "unimplemented attrs".
        assert assert_runtime_support(graph) is None

    def test_an_undeclared_slot_reference_is_rejected(self):
        """An undeclared slot is a name without storage — it would slip through to execution neither
        allocated nor bound."""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "states": {"k": "ghost", "v": "l0.v"}}

        with pytest.raises(
            IrError, match=re.escape("state slot 'ghost' is not declared in graph.states")
        ):
            parse_kv(nodes=nodes)

    def test_an_effect_op_with_outputs_is_rejected(self):
        """`state_append` **defines no value** (declaring zero outputs is the contract table's empty
        list itself)."""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[1] = {**nodes[1], "outs": ["z"]}
        values = {**kv_graph()["values"], "z": {"dtype": "f32", "shape": [1, 2, 1, 16]}}

        with pytest.raises(OpContractError, match="output count is 1"):
            check_kv(nodes=nodes, values=values)

    def test_the_same_slot_in_two_fields_is_rejected(self):
        """A mix-up writing the same slot for `k` and `v` is not caught by shape (same shape)."""
        nodes = [dict(node) for node in kv_graph()["nodes"][:2]]
        nodes[0] = {**nodes[0], "states": {"k": "l0.k", "v": "l0.k"}}

        with pytest.raises(
            OpContractError, match=r"references the same slot 'l0\.k' from several fields"
        ):
            check_kv(nodes=nodes, states={"l0.k": kv_graph()["states"]["l0.k"]})

    def test_a_states_form_attention_with_a_mask_is_rejected(self):
        """The states form is fixed causal and materializes no mask tensor (it takes no optional
        trailing input)."""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "ins": ["q", "k", "v", "m"]}
        inputs = [*kv_graph()["inputs"], {"name": "m", "dtype": "f32", "shape": [1, 1, 1, 64]}]

        with pytest.raises(OpContractError, match="states form takes exactly 3 inputs"):
            check_kv(nodes=nodes, inputs=inputs)

    @pytest.mark.parametrize("window", [0, -1, 1.5, "64", True])
    def test_window_outside_the_value_range_is_rejected(self, window):
        with pytest.raises(OpContractError, match="is not an integer of at least 1"):
            check_kv(nodes=kv_windows(window, window, window))

    @pytest.mark.parametrize("window", [8.0, 8e0, 64.0])
    def test_an_integral_float_window_is_accepted(self, window):
        """JSON `8.0` / `8e0` are treated as int (aligned with the TS side's accepted set).

        In TS, JSON.parse returns a single number type, so there is no "float with an integer
        value" distinction (`Number.isSafeInteger(8.0)` is true); if Python rejected floats
        outright, **only the exporter's verification would fail on a graph the runtime can read**
        (the accepted sets must not drift in either direction).
        """
        graph = check_kv(nodes=kv_windows(window, window, window))

        for node in graph.nodes:
            assert isinstance(node.attrs["window"], float), (
                "no float was passed (the check would pass vacuously)"
            )
            assert state_window(node.attrs, "t") == int(window)
            assert isinstance(state_window(node.attrs, "t"), int), "not normalized to int"

    @pytest.mark.parametrize(
        "window",
        [1.5, float(2**53), float("inf"), float("-inf"), float("nan")],
    )
    def test_a_non_integral_or_out_of_range_float_window_is_rejected(self, window):
        """A float that is not integral or is outside the safe range is rejected as before (the
        normalization does not widen the accepted set).

        Non-finite values are also stopped by the JSON reading (`parse_graph_json` rejecting
        non-standard literals), so this exercises the attrs layer itself through a direct call.
        """
        with pytest.raises(OpContractError, match="is not an integer of at least 1"):
            state_window({"window": window}, "t")

    def test_window_on_a_plain_attention_is_rejected(self):
        """The optional attrs are for the states form only — allowing them on the classic form would
        create an attr nobody reads."""
        with pytest.raises(OpContractError, match="only on a node with a states field"):
            assert_op_contracts(
                parse_ir_graph(json.dumps(plain_attention_graph({"scale": 0.25, "window": 8})))
            )

    def test_a_plain_attention_is_untouched(self):
        """A node without the field keeps the classic contract as is (existing assets and checks are
        unaffected)."""
        graph = parse_ir_graph(json.dumps(plain_attention_graph({"scale": 0.25})))

        assert assert_op_contracts(graph) is None
        assert graph.nodes[0].states == {}

    def test_a_reader_after_the_append_is_rejected(self):
        """A reader after the append would "read this step's k/v twice, as the past" (decision 5b
        ②)."""
        nodes = kv_graph()["nodes"]

        with pytest.raises(IrError, match="is followed by reader"):
            check_kv(nodes=[nodes[1], nodes[2], nodes[0]])

    def test_two_appends_to_one_slot_are_rejected(self):
        """Writing twice in one step advances the ring's position formula twice (decision 5b ①)."""
        nodes = kv_graph()["nodes"]

        with pytest.raises(IrError, match=re.escape("state_append appears 2 times")):
            check_kv(nodes=[nodes[0], nodes[1], nodes[2], nodes[1]])

    @pytest.mark.parametrize(
        ("windows", "why"),
        [
            ((64, 32, 64), "the read side and the write side differ in value"),
            (
                (None, 64, 64),
                "only the read side lacks the declaration (whole context mixed up with the ring)",
            ),
            ((64, None, 64), "only the write side lacks the declaration"),
        ],
    )
    def test_a_window_mismatch_on_one_slot_is_rejected(self, windows, why):
        """The mapping logical col → physical row MUST use the same formula for reads and writes
        (decision 4) — presence and value both match."""
        with pytest.raises(IrError, match=re.escape("attrs.window disagrees")):
            check_kv(nodes=kv_windows(*windows))

    def test_a_state_append_input_that_is_not_rank_four_is_rejected(self):
        """The declared input shape is fixed at `[B,Hkv,M,D]` (the same physical chunk dimensions as
        the attention ins).

        With rank 5, B / Hkv / D **happen to match** (axis 3 stays D), so only the rank check
        detects it.
        """
        inputs = [*kv_graph()["inputs"], {"name": "k5", "dtype": "f32", "shape": [1, 2, 1, 16, 1]}]
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[1] = {**nodes[1], "ins": ["k5"]}

        with pytest.raises(OpContractError, match=re.escape("must be rank-4 [B,Hkv,M,D]")):
            check_kv(inputs=inputs, nodes=nodes)

    def test_a_zero_length_chunk_is_rejected(self):
        """A softmax over an empty axis has no defined amax identity (the same reason as the classic
        form rejecting N=0)."""
        inputs = [
            {"name": "q", "dtype": "f32", "shape": [1, 8, 0, 16]},
            {"name": "k", "dtype": "f32", "shape": [1, 2, 0, 16]},
            {"name": "v", "dtype": "f32", "shape": [1, 2, 0, 16]},
        ]

        with pytest.raises(OpContractError, match="a chunk with M = 0"):
            check_kv(inputs=inputs, values={"o": {"dtype": "f32", "shape": [1, 8, 0, 16]}})

    def test_a_slot_that_is_not_rank_four_is_rejected(self):
        """The physical slot shape is fixed at `[B,Hkv,C,D]` (ADR 0067 decision 4 ②)."""
        slot = {"dtype": "f32", "shape": [1, 2, 64]}

        with pytest.raises(OpContractError, match=re.escape("must be rank-4 [B,Hkv,C,D]")):
            check_kv(states={"l0.k": slot, "l0.v": slot})

    def test_a_slot_whose_head_count_differs_from_the_input_is_rejected(self):
        """Referencing another layer's slot — with the same capacity it would be a silent
        misread."""
        slot = {"dtype": "f32", "shape": [1, 4, 64, 16]}

        with pytest.raises(OpContractError, match="B / Hkv / D mismatch"):
            check_kv(states={"l0.k": slot, "l0.v": slot})

    def test_slots_of_different_capacity_for_k_and_v_are_rejected(self):
        """A ring where only one side wraps first shifts the values silently."""
        with pytest.raises(OpContractError, match="differ in shape"):
            check_kv(
                states={
                    "l0.k": {"dtype": "f32", "shape": [1, 2, 64, 16]},
                    "l0.v": {"dtype": "f32", "shape": [1, 2, 32, 16]},
                }
            )

    def test_a_window_larger_than_the_capacity_is_rejected(self):
        """With a ring width above the capacity, a written row gets overwritten by itself within the
        same step."""
        with pytest.raises(OpContractError, match="exceeds the slot capacity 64"):
            check_kv(nodes=kv_windows(65, 65, 65))

    def test_a_symbolic_capacity_defers_the_window_bound(self):
        """With a symbolic capacity, `window <= C` is not decided (binding is the layer of
        `createGenerationContext`).

        The TS side handles only numbers after binding is resolved, so it can always decide — this
        one point is an asymmetry deferred on one side, the result of following the existing rule
        "do not pretend to have decided what the declaration alone cannot decide" (the same as for
        broadcastability and divisible broadcast).
        """
        slot = {"dtype": "f32", "shape": [1, 2, "C", 16]}
        graph = check_kv(
            symbols=["C"],
            nodes=kv_windows(9999, 9999, 9999),
            states={
                "l0.k": slot,
                "l0.v": slot,
            },
        )

        assert graph.symbols == ["C"]

    def test_a_states_only_symbol_is_bindable(self):
        """A symbol that appears only in states (the KV capacity) has a binding point too (ADR 0066
        addendum of 2026-08-17, item 7)."""
        slot = {"dtype": "f32", "shape": [1, 2, "C", 16]}

        assert check_kv(symbols=["C"], states={"l0.k": slot, "l0.v": slot}).symbols == ["C"]

    def test_a_states_only_symbol_in_a_value_shape_is_rejected(self):
        """Only bindings from inputs resolve value shapes — if it appeared there, it would always
        fail to bind at run time."""
        slot = {"dtype": "f32", "shape": [1, 2, "C", 16]}

        with pytest.raises(
            IrError, match=re.escape("states-only symbol 'C' appears in a value shape")
        ):
            parse_kv(
                symbols=["C"],
                states={"l0.k": slot, "l0.v": slot},
                values={"o": {"dtype": "f32", "shape": [1, 8, "C", 16]}},
            )

    def test_a_symbol_bound_nowhere_is_still_rejected(self):
        """The binding points are "input shapes ∪ states shapes" — a symbol in neither fails as
        before."""
        with pytest.raises(IrError, match="appears in no dimension position"):
            parse_kv(symbols=["C"])

    @pytest.mark.parametrize(
        ("states", "message"),
        [
            ([], "states: is not an object"),
            ({"k": 4}, "states['k']: is not a non-empty string"),
            ({"k": ""}, "states['k']: is not a non-empty string"),
            ({"": "l0.k"}, "key of graph.nodes[0].states: is not a non-empty string"),
        ],
    )
    def test_malformed_fields_are_rejected(self, states, message):
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "states": states}

        with pytest.raises(IrError, match=re.escape(message)):
            parse_kv(nodes=nodes)

    @pytest.mark.parametrize(
        ("states", "message"),
        [
            ({"k": "l0.k"}, "states field keys are ['k']"),
            (
                {"k": "l0.k", "v": "l0.v", "extra": "l0.k"},
                "states field keys are ['extra', 'k', 'v']",
            ),
        ],
    )
    def test_a_key_set_other_than_the_contract_is_rejected(self, states, message):
        """The key set exactly equals the contract's declaration — allowing a subset would let a
        forgotten `v` through."""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "states": states}

        with pytest.raises(OpContractError, match=re.escape(message)):
            check_kv(nodes=nodes)

    def test_a_state_append_without_the_field_is_rejected(self):
        """The states field of `state_append` is required (there is no write op without a
        destination)."""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        del nodes[1]["states"]

        with pytest.raises(OpContractError, match="states field is missing"):
            check_kv(nodes=nodes)

    def test_an_op_without_a_states_contract_is_rejected(self):
        """The field written on an op that cannot have one (enforcing the contract's `None` = cannot
        have it)."""
        nodes = [
            *kv_graph()["nodes"],
            {"op": "relu", "ins": ["o"], "outs": ["r"], "attrs": {}, "states": {"k": "l0.k"}},
        ]
        values = {**kv_graph()["values"], "r": {"dtype": "f32", "shape": [1, 8, 1, 16]}}

        with pytest.raises(OpContractError, match="has no states field"):
            check_kv(nodes=nodes, values=values)


class TestIntegralFloatDimensions:
    """Integral JSON floats (`1.0` / `1e0`) are accepted, as in TS.

    In TS, JSON.parse returns a single number type, so `4` and `4.0` cannot be told apart
    (`Number.isSafeInteger` is true for both). Rejecting floats outright here would diverge as "the
    runtime can read it, but the exporter's verification alone fails", so both sides pin the same
    accepted set.
    """

    @staticmethod
    def parse_raw_dim(raw: str, **overrides):
        """Inject a single dimension as raw JSON text (json.dumps would fold `1e0` into `1.0`)."""
        text = json.dumps({**base_graph(), **overrides})
        assert '"__DIM__"' in text
        return parse_ir_graph(text.replace('"__DIM__"', raw))

    @pytest.mark.parametrize("raw", ["4.0", "4e0", "0.4e1"])
    def test_integral_float_value_dims_are_accepted_as_int(self, raw):
        graph = self.parse_raw_dim(
            raw,
            values={
                "w": {"dtype": "f32", "shape": ["__DIM__"]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
            },
        )

        assert graph.values["w"].shape == [4]
        # The normalized result is int (kept as float, to_dict would write `4.0` back and break
        # the round trip).
        assert all(isinstance(dim, int) for dim in graph.values["w"].shape)

    @pytest.mark.parametrize("raw", ["4.0", "4e0"])
    def test_integral_float_state_dims_are_accepted_as_int(self, raw):
        graph = self.parse_raw_dim(
            raw, **declared_states({"cache": {"dtype": "f32", "shape": [1, "__DIM__"]}})
        )

        assert graph.states["cache"].shape == [1, 4]
        assert graph.to_dict()["states"]["cache"]["shape"] == [1, 4]

    @pytest.mark.parametrize("raw", ["1.5", "15e-1"])
    def test_non_integral_float_dims_are_rejected(self, raw):
        with pytest.raises(IrError, match="is neither a number nor a string"):
            self.parse_raw_dim(
                raw,
                values={
                    "w": {"dtype": "f32", "shape": ["__DIM__"]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                },
            )
        with pytest.raises(IrError, match="is neither a number nor a string"):
            self.parse_raw_dim(raw, states={"cache": {"dtype": "f32", "shape": [1, "__DIM__"]}})

    def test_integral_float_beyond_the_safe_range_is_rejected(self):
        """Fails at the same upper bound as `Number.isSafeInteger` (no value that only TS cannot
        read gets through)."""
        with pytest.raises(IrError, match="is not a non-negative integer"):
            self.parse_raw_dim(
                "1e300",
                values={
                    "w": {"dtype": "f32", "shape": ["__DIM__"]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                },
            )


class TestIntegralFloatGroupSize:
    """`group_size` accepts integral floats like the dimensions do (keeps the accepted sets
    aligned).

    On the TS side `Number.isSafeInteger(32.0)` is true, so an i4 graph written with `32.0` can be
    read. If only Python failed with "not a positive integer", **only the exporter's verification
    would fail on a distribution the runtime can read** (the same reason as the normalization in
    `_parse_shape`).
    """

    @staticmethod
    def parse_raw_group_size(raw: str):
        """Inject only `group_size` as raw JSON text (json.dumps would fold it into `32.0`)."""
        graph = {
            **base_graph(),
            "initializers": {
                "w": {
                    "tensor": "enc.w",
                    "storage": {"dtype": "i4", "scale": "enc.s", "group_size": "__GS__"},
                }
            },
            "values": {
                "w": {"dtype": "f32", "shape": [4, 32]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
            },
        }
        text = json.dumps(graph)
        assert '"__GS__"' in text
        return parse_ir_graph(text.replace('"__GS__"', raw))

    @pytest.mark.parametrize("raw", ["32.0", "3.2e1"])
    def test_an_integral_float_group_size_is_accepted_as_int(self, raw):
        group_size = self.parse_raw_group_size(raw).initializers["w"].storage.group_size

        assert group_size == 32
        # Kept as int (kept as float, to_dict would write `32.0` back and break the round trip).
        assert isinstance(group_size, int)

    def test_a_non_integral_group_size_is_still_rejected(self):
        with pytest.raises(IrError, match="is not a positive integer"):
            self.parse_raw_group_size("32.5")

    def test_an_integral_float_beyond_the_safe_range_is_still_rejected(self):
        """The same upper bound as `Number.isSafeInteger` (no value that only TS cannot read gets
        through)."""
        with pytest.raises(IrError, match="safe integer"):
            self.parse_raw_group_size("1e300")


class TestJsonLiterals:
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_standard_literals_are_rejected(self, literal):
        with pytest.raises(IrError, match="Non-standard literal"):
            parse_ir_graph('{"format":"karume-ir","version":' + literal + "}")

    def test_overflowing_exponent_is_rejected_at_the_value_level(self):
        """`1e999` is valid syntax but rounds to Infinity — it is rejected by value."""
        with pytest.raises(IrError, match="is not finite"):
            parse_ir_graph('{"format":"karume-ir","version":1e999}')

    def test_broken_json_is_reported_as_such(self):
        with pytest.raises(IrError, match="Cannot parse"):
            parse_ir_graph("{")


class TestRuntimeSupport:
    """The match against the runtime support table looks not only at op names but also at dtypes,
    attrs and storage dtypes."""

    def test_the_baseline_graph_is_executable(self):
        assert assert_runtime_support(parse()) is None

    def test_unsupported_ops_are_enumerated_at_once(self):
        """Ops not even in the vocabulary are enumerated **together** and fail (not one at a time).

        NOTE: the representative ops change with every vocabulary extension — conv_transpose1d
        entered the contract table with ADR 0015, conv2d with ADR 0017, conv3d with ADR 0118. This
        pins **the enumeration of several** with the multi-dimensional transposed convolutions
        (conv_transpose2d / conv_transpose3d).
        """
        graph = parse(
            requires={"ops": ["conv_transpose2d", "conv_transpose3d"]},
            outputs=["z"],
            values={
                "w": {"dtype": "f32", "shape": [4]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
                "z": {"dtype": "f32", "shape": ["T", 4]},
            },
            nodes=[
                {"op": "conv_transpose3d", "ins": ["x"], "outs": ["y"], "attrs": {}},
                {"op": "conv_transpose2d", "ins": ["y", "w"], "outs": ["z"], "attrs": {}},
            ],
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "unsupported ops" in message
        assert "conv_transpose3d" in message and "conv_transpose2d" in message

    def test_an_unbatched_conv3d_graph_is_executable(self):
        """conv3d (ADR 0118 decision 1) entered the vocabulary — an unbatched rank-4 input, a rank-5
        weight and 3-component [T, H, W] attrs pass both the support table and the contract (the
        old version rejected it here)."""
        graph = parse(
            requires={"ops": ["conv3d"]},
            symbols=[],
            inputs=[{"name": "x", "dtype": "f32", "shape": [3, 4, 6, 6]}],
            initializers={
                "w": {"tensor": "conv.weight", "storage": {"dtype": "f32"}},
                "b": {"tensor": "conv.bias", "storage": {"dtype": "f32"}},
            },
            values={
                "w": {"dtype": "f32", "shape": [5, 3, 3, 3, 3]},
                "b": {"dtype": "f32", "shape": [5]},
                "y": {"dtype": "f32", "shape": [5, 2, 6, 6]},
            },
            nodes=[
                {
                    "op": "conv3d",
                    "ins": ["x", "w", "b"],
                    "outs": ["y"],
                    "attrs": {
                        "stride": [1, 1, 1],
                        "padding": [0, 1, 1],
                        "dilation": [1, 1, 1],
                        "groups": 1,
                    },
                }
            ],
        )

        assert assert_runtime_support(graph) is None
        assert_op_contracts(graph)

    def test_every_semantic_dtype_can_be_transferred(self):
        """The transfer layer's axis accepts the whole semantic dtype vocabulary (ADR 0009 allowed
        i32 / bool).

        Inputs that no node consumes are still subject to the transfer layer's constraints (the
        executor transfers every graph.inputs), so the axis itself stays.
        """
        graph = parse(
            inputs=[
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "ids", "dtype": "i32", "shape": ["T"]},
                {"name": "mask", "dtype": "bool", "shape": ["T"]},
            ]
        )

        assert_runtime_support(graph)

    def test_a_storage_only_dtype_cannot_be_a_graph_input(self):
        """The other side — only semantic dtypes can be transferred; a storage dtype cannot be an
        input.

        `f16` is valid as an initializer's storage (f32-as-f16 in TestDeclarationCompleteness), so
        that "dtypes that can be stored" and "dtypes that can be transferred" are separate
        vocabularies can only be pinned from this side.
        """
        with pytest.raises(IrError, match="semantic dtype 'f16'"):
            parse(inputs=[{"name": "x", "dtype": "f16", "shape": ["T", 4]}])

    def test_an_op_that_does_not_accept_the_input_dtype_is_reported(self):
        """A dtype that can be transferred but that the op does not accept fails here (the reason
        there are 2 axes)."""
        graph = parse(inputs=[{"name": "x", "dtype": "i32", "shape": ["T", 4]}])

        with pytest.raises(ContainerError, match="unsupported semantic dtype"):
            assert_runtime_support(graph)

    def test_swapped_slots_of_a_per_slot_op_are_enumerated(self):
        """A form that slips through when matched only against the **union** of the accepted sets
        (same meaning as packages/runtime/src/ops/support.ts).

        The union for gather is {f32, i32}, so even with the value and the index swapped both
        dtypes are in the union. If the enumerating check did not look at slots, it would fall
        through to the contract checks and stop one at a time, breaking the intent "enumerate
        every unsupported case".
        """
        graph = parse(
            requires={"ops": ["gather"]},
            inputs=[
                {"name": "src", "dtype": "i32", "shape": ["T", 4]},
                {"name": "idx", "dtype": "f32", "shape": ["T", 3]},
            ],
            initializers={},
            values={"y": {"dtype": "f32", "shape": ["T", 3]}},
            nodes=[{"op": "gather", "ins": ["src", "idx"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "unsupported semantic dtypes (2)" in message
        assert "value 'src': i32" in message and "value 'idx': f32" in message

    def test_the_output_of_a_per_slot_op_is_checked_against_the_value_slot(self):
        """The output has the same type as slot 0 (the value side) — checking against the union
        would let gather's i32 output through."""
        graph = parse(
            requires={"ops": ["gather"]},
            inputs=[
                {"name": "src", "dtype": "f32", "shape": ["T", 4]},
                {"name": "idx", "dtype": "i32", "shape": ["T", 3]},
            ],
            initializers={},
            values={"y": {"dtype": "i32", "shape": ["T", 3]}},
            nodes=[{"op": "gather", "ins": ["src", "idx"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(ContainerError, match="value 'y': i32"):
            assert_runtime_support(graph)

    def test_swapped_output_slots_of_a_multi_output_op_are_enumerated(self):
        """The output declarations of a multi-output op (topk) **swapped** between slots
        (counterpart of the same-shaped test in packages/runtime/tests/ops_support_test.ts).

        With a regression that checks every output against slot 0's accepted set, the value side
        (f32) and the index side (i32) would both be judged against the same set, and only one of
        them would be enumerated (or both would slip through).
        """
        graph = parse(
            requires={"ops": ["topk"]},
            outputs=["v", "i"],
            initializers={},
            # slot 0 is the values (f32) and slot 1 the indices (i32), so **both** of these are
            # unsupported
            values={
                "v": {"dtype": "i32", "shape": ["T", 2]},
                "i": {"dtype": "f32", "shape": ["T", 2]},
            },
            nodes=[{"op": "topk", "ins": ["x"], "outs": ["v", "i"], "attrs": {"k": 2}}],
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "unsupported semantic dtypes (2)" in message
        assert "value 'v': i32" in message and "value 'i': f32" in message

    def test_bf16_storage_passes_the_capability_gate(self):
        """bf16 runs on the same path as f16 (ADR 0131; the same as RUNTIME_SUPPORT.storage on the
        TS side)."""
        graph = parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "bf16"}}})

        assert assert_runtime_support(graph) is None

    def test_unsupported_storage_dtype_is_reported(self, monkeypatch):
        """A storage not in the support table fails in the enumeration.

        Every storage in the IR vocabulary can run, so the table is narrowed to one without bf16
        to look at the check itself (the same shape as `M0_SUPPORT` in the TS side's
        ops_support_test).
        """
        monkeypatch.setattr(
            "karume.verify.M0_STORAGE_DTYPES", frozenset({"f32", "f16", "i8", "i4", "i2", "i32"})
        )
        graph = parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "bf16"}}})

        with pytest.raises(ContainerError, match="unsupported storage dtype"):
            assert_runtime_support(graph)

    def test_group_size_on_a_storage_other_than_i4_is_reported(self):
        """The only storage that accepts group quantization is i4 (ADR 0069 decision 2).

        A group_size on i8 has no execution path, and silently ignoring it would read per-group
        scales as per-channel — a silently wrong value. Mirror of assertRuntimeSupport on the TS
        side; without this check on the Python side there would be the asymmetry "verify green,
        fails only in the browser".
        """
        graph = parse(
            initializers={
                "w": {
                    "tensor": "enc.w",
                    "storage": {"dtype": "i8", "scale": "enc.s", "group_size": 32},
                }
            }
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "unsupported group quantization (1): w" in message
        assert "is i4 only" in message

    def test_unknown_attrs_on_a_known_op_are_reported(self):
        graph = parse(
            nodes=[{"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {"alpha": 2}}]
        )

        with pytest.raises(ContainerError, match="unimplemented attrs"):
            assert_runtime_support(graph)


class TestOpContracts:
    def test_arity_violation_is_rejected(self):
        graph = parse(nodes=[{"op": "add", "ins": ["x"], "outs": ["y"], "attrs": {}}])

        with pytest.raises(OpContractError, match="input count is 1"):
            assert_op_contracts(graph)

    def test_multiple_outputs_are_rejected(self):
        graph = parse(
            outputs=["y", "z"],
            values={
                "w": {"dtype": "f32", "shape": [4]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
                "z": {"dtype": "f32", "shape": ["T", 4]},
            },
            nodes=[{"op": "add", "ins": ["x", "w"], "outs": ["y", "z"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="output count is 2"):
            assert_op_contracts(graph)

    def _cast_graph(self, attrs: dict, out_dtype: str = "i32") -> dict:
        return {
            "requires": {"ops": ["cast"]},
            "initializers": {},
            "values": {"y": {"dtype": out_dtype, "shape": ["T", 4]}},
            "nodes": [{"op": "cast", "ins": ["x"], "outs": ["y"], "attrs": attrs}],
        }

    def test_cast_declares_its_target_dtype_in_attrs(self):
        graph = parse(**self._cast_graph({"to": "i32"}))

        assert_op_contracts(graph)

    def test_missing_required_attr_is_rejected(self):
        """Every declared key is required (a check that looks only for unknown keys lets this
        through — ADR 0012)."""
        graph = parse(**self._cast_graph({}))

        with pytest.raises(OpContractError, match="required attr 'to' is missing"):
            assert_op_contracts(graph)

    @pytest.mark.parametrize("value", ["i64", "float32", 3, None])
    def test_attr_value_outside_the_contract_is_rejected(self, value):
        graph = parse(**self._cast_graph({"to": value}))

        with pytest.raises(OpContractError, match="is not a semantic dtype"):
            assert_op_contracts(graph)

    def test_output_dtype_must_agree_with_the_contract(self):
        """A graph whose attrs.to disagrees with the declaration in values{} is not accepted."""
        graph = parse(**self._cast_graph({"to": "bool"}, out_dtype="i32"))

        with pytest.raises(OpContractError, match="differs from the contract's 'bool'"):
            assert_op_contracts(graph)

    def test_op_that_does_not_accept_the_input_dtype_is_rejected(self):
        graph = parse(
            requires={"ops": ["bitwise_not"]},
            initializers={},
            values={"y": {"dtype": "bool", "shape": ["T", 4]}},
            nodes=[{"op": "bitwise_not", "ins": ["x"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="cannot execute semantic dtype 'f32'"):
            assert_op_contracts(graph)


class TestSymPrefixSlice:
    """The symbolic prefix slice (ADR 0010) — the shape of its attrs and the 3 contracts that need
    the graph context."""

    def _graph(self, attrs: dict, *, const_shape=(16, 16), out_shape=("T", "T")) -> dict:
        return {
            "requires": {"ops": ["sym_prefix_slice"]},
            "inputs": [{"name": "x", "dtype": "i32", "shape": ["T"]}],
            "outputs": ["y"],
            "initializers": {"table": {"tensor": "enc.table", "storage": {"dtype": "i32"}}},
            "values": {
                "table": {"dtype": "i32", "shape": list(const_shape)},
                "y": {"dtype": "i32", "shape": list(out_shape)},
            },
            "nodes": [{"op": "sym_prefix_slice", "ins": ["table"], "outs": ["y"], "attrs": attrs}],
        }

    def _slices(self, *dims: int) -> list[dict]:
        return [{"dim": dim, "coeff": 1, "offset": 0} for dim in dims]

    def test_a_well_formed_prefix_slice_is_accepted(self):
        graph = parse(**self._graph({"sym": "T", "slices": self._slices(0, 1)}))

        assert_op_contracts(graph)

    def test_an_unbound_symbol_is_rejected(self):
        """A sym not in graph.symbols cannot be bound by the runtime, so the prefix length is
        undecided."""
        graph = parse(**self._graph({"sym": "U", "slices": self._slices(0, 1)}))

        with pytest.raises(IrError, match=re.escape("graph.symbols")):
            assert_op_contracts(graph)

    def test_a_dim_outside_the_input_rank_is_rejected(self):
        graph = parse(**self._graph({"sym": "T", "slices": self._slices(0, 2)}))

        with pytest.raises(IrError, match="outside the input rank 2"):
            assert_op_contracts(graph)

    def test_a_symbolic_input_shape_is_rejected(self):
        """The input is a static shape baked at Tmax — with a symbol, the read stride would shrink
        per run.

        Through an initializer it would fail earlier on parse_ir_graph's "cannot use a symbolic
        dimension", so this feeds **a graph input with a symbolic shape** to exercise this check
        alone.
        """
        graph = parse(
            requires={"ops": ["sym_prefix_slice"]},
            inputs=[{"name": "x", "dtype": "i32", "shape": ["T", 4]}],
            outputs=["y"],
            initializers={},
            values={"y": {"dtype": "i32", "shape": ["T", 2]}},
            nodes=[
                {
                    "op": "sym_prefix_slice",
                    "ins": ["x"],
                    "outs": ["y"],
                    "attrs": {"sym": "T", "slices": self._slices(0)},
                }
            ],
        )

        with pytest.raises(IrError, match="with a symbolic dimension"):
            assert_op_contracts(graph)

    def test_duplicate_dims_are_rejected(self):
        graph = parse(**self._graph({"sym": "T", "slices": self._slices(0, 0)}))

        with pytest.raises(OpContractError, match="has the same dim twice"):
            assert_op_contracts(graph)

    @pytest.mark.parametrize(
        "slices",
        [
            [{"dim": 0, "coeff": 0, "offset": 0}],
            [{"dim": 0, "coeff": 1, "offset": -1}],
            [{"dim": -1, "coeff": 1, "offset": 0}],
            [{"dim": 0, "coeff": 1}],
            [{"dim": 0, "coeff": 1, "offset": 0, "extra": 1}],
            [],
        ],
        ids=["coeff-0", "negative-offset", "negative-dim", "missing-key", "unknown-key", "empty"],
    )
    def test_malformed_slices_are_rejected(self, slices):
        graph = parse(**self._graph({"sym": "T", "slices": slices}))

        with pytest.raises(OpContractError):
            assert_op_contracts(graph)


class TestSlotDtypes:
    """Per-slot dtype contracts (an extension of ADR 0012) — gather's src=f32 / index=i32."""

    def _gather_graph(self, index_dtype: str = "i32", out_dtype: str = "f32") -> dict:
        return {
            "requires": {"ops": ["gather"]},
            "inputs": [
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "i", "dtype": index_dtype, "shape": ["T", 3]},
            ],
            "initializers": {},
            "values": {"y": {"dtype": out_dtype, "shape": ["T", 3]}},
            "nodes": [{"op": "gather", "ins": ["x", "i"], "outs": ["y"], "attrs": {}}],
        }

    def test_mixed_slot_dtypes_are_accepted(self):
        """A per-slot contract is exempt from "mixed input dtypes are rejected" (that is what
        per-slot means)."""
        graph = parse(**self._gather_graph())

        assert_op_contracts(graph)
        assert_runtime_support(graph)

    def test_index_slot_rejects_f32(self):
        graph = parse(**self._gather_graph(index_dtype="f32"))

        with pytest.raises(OpContractError, match="in slot 1"):
            assert_op_contracts(graph)

    def test_swapped_slots_are_rejected(self):
        """src and index passed the other way round (a match against the union alone lets it
        through)."""
        graph = parse(
            requires={"ops": ["gather"]},
            inputs=[
                {"name": "x", "dtype": "i32", "shape": ["T", 4]},
                {"name": "i", "dtype": "f32", "shape": ["T", 3]},
            ],
            initializers={},
            values={"y": {"dtype": "i32", "shape": ["T", 3]}},
            nodes=[{"op": "gather", "ins": ["x", "i"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="in slot 0"):
            assert_op_contracts(graph)

    def test_output_follows_the_value_slot(self):
        graph = parse(**self._gather_graph(out_dtype="i32"))

        with pytest.raises(OpContractError, match="differs from the contract's 'f32'"):
            assert_op_contracts(graph)

    def test_uniform_ops_still_reject_mixed_dtypes(self):
        """The uniform contract keeps rejecting mixes (it was not collapsed into per-slot)."""
        graph = parse(
            requires={"ops": ["mul"]},
            inputs=[
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "i", "dtype": "i32", "shape": ["T", 4]},
            ],
            initializers={},
            values={"y": {"dtype": "f32", "shape": ["T", 4]}},
            nodes=[{"op": "mul", "ins": ["x", "i"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="input dtypes are mixed"):
            assert_op_contracts(graph)


def write_raw_container(path, header: dict, data: bytes = b""):
    """A safetensors file with its header JSON built directly (for fault injection of invalid
    shapes that `save_file` cannot produce).

    The header length is padded with spaces to a multiple of 8 (the same convention as
    `legacy_writer._HEADER_ALIGN`). Without padding the data section would start off the 4-byte
    boundary, the alignment check would fail first, and **the index-range checks could not be
    exercised**.
    """
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + data)
    return path


class TestReaderIndexRange:
    """The header's indices have the same accepted set as the TS reader's `asIndex` (non-negative,
    at most 2^53−1).

    If verify accepted values that `packages/runtime/src/format/safetensors.ts` rejects, a file
    "green in the exporter, failing only in the browser's reader" would remain as a distribution.
    A check that looks only at the product of the element counts lets negative dimensions and
    values above the safe integer through in shapes where **the product happens to match**.
    """

    def test_a_negative_dimension_is_rejected(self, tmp_path):
        """`[0,-1]` has the product 0, equal to the byte length 0, so a product-only check lets it
        through."""
        header = {"t": {"dtype": "F32", "shape": [0, -1], "data_offsets": [0, 0]}}

        with pytest.raises(ContainerError, match="non-negative integer"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))

    def test_a_dimension_beyond_the_safe_integer_range_is_rejected(self, tmp_path):
        header = {"t": {"dtype": "F32", "shape": [0, 2**53], "data_offsets": [0, 0]}}

        with pytest.raises(ContainerError, match="non-negative integer"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))

    def test_an_element_count_beyond_the_safe_integer_range_is_rejected(self, tmp_path):
        """Each dimension is a safe integer but **the product** exceeds it (checked at the same
        stage as TS's elementCount)."""
        header = {"t": {"dtype": "F32", "shape": [2**27, 2**27], "data_offsets": [0, 0]}}

        with pytest.raises(ContainerError, match="element count exceeds the safe integer"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))

    def test_a_data_offset_beyond_the_safe_integer_range_is_rejected(self, tmp_path):
        """Fails on the index range before the size mismatch (TS runs asIndex first)."""
        header = {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 2**53]}}

        with pytest.raises(ContainerError, match="non-negative integer"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))


class TestReaderEntryStructure:
    """The structure of a single header entry also fails with the check's diagnostic (bare indexing
    would leak a KeyError / TypeError).

    Files the exporter wrote never reach this (the old writer `_save_ordered`, retired at step 3a,
    always wrote the 3 keys), but paths that read externally made safetensors do exist — the input
    of the migration CLI reading old shards (`karume.legacy` / `karume.migrate` — the superset for
    reading the old format, `allow_legacy_dtypes=True`) and the safetensors written by the dacvae
    conversion (`tools/export-recipes/irodori/dacvae/convert.py`) (the default mode = the same
    dtype set as the TS reader — no I4 / I2). `karume verify` accepts only containers.
    This keeps the subject of the diagnostic from turning from "an invalid file" into "the
    exporter broke".
    """

    @pytest.mark.parametrize(
        ("entry", "why"),
        [
            ({"shape": [2], "data_offsets": [0, 8]}, r"is missing \['dtype'\]"),
            ({"dtype": "F32", "data_offsets": [0, 8]}, r"is missing \['shape'\]"),
            ({"dtype": "F32", "shape": [2]}, r"is missing \['data_offsets'\]"),
            ("F32", "is not an object"),
            ({"dtype": "F32", "shape": 2, "data_offsets": [0, 8]}, "shape is not an array"),
        ],
        ids=["no-dtype", "no-shape", "no-offsets", "not-an-object", "shape-not-a-list"],
    )
    def test_a_malformed_entry_is_reported_as_a_container_error(self, tmp_path, entry, why):
        path = write_raw_container(tmp_path / "m.safetensors", {"t": entry}, b"\0" * 8)

        with pytest.raises(ContainerError, match=why):
            assert_reader_layout(path)


class TestReaderPackedFourBit:
    """packed 4-bit (ADR 0069 decision 2) — the size is in bits, the alignment is 4 bytes at the
    tensor's **start**.

    The `safetensors` library does not know `I4` (it is not in the dtype vocabulary of 0.8.0), so
    `assert_reader_layout`, which reads the header JSON directly, is the only check that can
    exercise I4. The TS-side reader (`packages/runtime/src/format/safetensors.ts`) rejects I4, so
    this pins the rules of the migration input mode that reads old shards
    (`allow_legacy_dtypes=True`).
    """

    def test_the_asset_gate_rejects_a_packed_tensor(self, tmp_path):
        """The asset check has the same dtype set as the TS reader — letting it through would allow
        writing an asset that first fails in the browser."""
        header = {"w": {"dtype": "I4", "shape": [3, 32], "data_offsets": [0, 48]}}

        with pytest.raises(ContainerError, match="dtype 'I4' unknown to the reader"):
            assert_reader_layout(
                write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 48)
            )

    def test_a_packed_tensor_is_half_its_element_count(self, tmp_path):
        header = {"w": {"dtype": "I4", "shape": [3, 32], "data_offsets": [0, 48]}}

        assert_reader_layout(
            write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 48),
            allow_legacy_dtypes=True,
        )

    def test_an_odd_element_count_is_rejected(self, tmp_path):
        """The total bit count does not land on a byte boundary (the last element sticks out by half
        a byte)."""
        header = {"w": {"dtype": "I4", "shape": [3], "data_offsets": [0, 2]}}

        with pytest.raises(ContainerError, match="does not land on a byte boundary"):
            assert_reader_layout(
                write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 2),
                allow_legacy_dtypes=True,
            )

    def test_a_size_that_is_not_half_the_element_count_is_rejected(self, tmp_path):
        header = {"w": {"dtype": "I4", "shape": [3, 32], "data_offsets": [0, 96]}}

        with pytest.raises(ContainerError, match="size mismatch"):
            assert_reader_layout(
                write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 96),
                allow_legacy_dtypes=True,
            )

    def test_a_tensor_start_off_the_four_byte_boundary_is_rejected(self, tmp_path):
        """I4 requires not element alignment (0.5 bytes) but a 4-byte alignment of its start for the
        u32 binding."""
        header = {
            "a": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]},
            "w": {"dtype": "I4", "shape": [4], "data_offsets": [2, 4]},
        }

        with pytest.raises(ContainerError, match="is not aligned"):
            assert_reader_layout(
                write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 4),
                allow_legacy_dtypes=True,
            )


class TestHeaderLengthBound:
    """The u64 header length is bounded by the actual file length before it goes to read (the same
    shape as `legacy.safetensors_header`).

    Without the bound a huge value would slip through to `read` and become an OverflowError /
    MemoryError, and the rule violation would not come out as the check's exception
    (ContainerError).
    """

    @pytest.mark.parametrize("header_length", [2**64 - 1, 2**40], ids=["u64-max", "1TiB"])
    def test_an_oversized_header_length_is_reported_as_a_container_error(
        self, tmp_path, header_length
    ):
        path = tmp_path / "m.safetensors"
        path.write_bytes(header_length.to_bytes(8, "little") + b'{"__metadata__":{}}')

        with pytest.raises(ContainerError, match="file length"):
            assert_reader_layout(path)

    def test_a_zero_header_length_is_rejected(self, tmp_path):
        path = tmp_path / "m.safetensors"
        path.write_bytes((0).to_bytes(8, "little"))

        with pytest.raises(ContainerError, match="file length"):
            assert_reader_layout(path)


class TestStridedRank:
    """The rank limit of the strided copy family (ADR 0011) —
    same meaning as assertStridedRank in packages/runtime/src/ops/shapes.ts.

    It depends only on the length of the declared shape, not on bindings, so it can be checked
    here even with symbolic dimensions.
    """

    def _permute_graph(self, rank: int) -> dict:
        source = ["T", *([2] * (rank - 1))]
        target = [*([2] * (rank - 1)), "T"]
        return {
            "requires": {"ops": ["permute"]},
            "inputs": [{"name": "x", "dtype": "f32", "shape": source}],
            "initializers": {},
            "values": {"y": {"dtype": "f32", "shape": target}},
            "nodes": [
                {
                    "op": "permute",
                    "ins": ["x"],
                    "outs": ["y"],
                    "attrs": {"dims": [*range(1, rank), 0]},
                }
            ],
        }

    def test_rank_at_the_limit_is_accepted(self):
        assert_op_contracts(parse(**self._permute_graph(4)))

    def test_rank_beyond_the_limit_is_rejected(self):
        with pytest.raises(OpContractError, match="the strided kernel limit"):
            assert_op_contracts(parse(**self._permute_graph(5)))

    def test_a_scalar_mask_is_rejected(self):
        """masked_fill's mask builds the strides of a right-aligned broadcast, so it needs rank 1 or
        more."""
        graph = parse(
            requires={"ops": ["masked_fill"]},
            inputs=[
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "m", "dtype": "bool", "shape": []},
            ],
            initializers={},
            values={"y": {"dtype": "f32", "shape": ["T", 4]}},
            nodes=[
                {"op": "masked_fill", "ins": ["x", "m"], "outs": ["y"], "attrs": {"value": 0.0}}
            ],
        )

        with pytest.raises(OpContractError, match="the strided kernel limit"):
            assert_op_contracts(graph)


class TestIntegralFloatPermuteDims:
    """permute's `dims` accepts integral floats like the other integer attrs (the same accepted set
    as TS).

    The TS-side check (`ops/attrs.ts`) uses `Number.isSafeInteger`, so `[0.0, 2e0, 1]` can be read.
    If only Python failed, **only the exporter's verification would fail on a graph the runtime
    can read**.
    """

    @pytest.mark.parametrize("dims", [[0.0, 2.0, 1.0], [0, 2e0, 1]])
    def test_integral_float_dims_are_accepted_as_int(self, dims):
        assert any(isinstance(dim, float) for dim in dims), (
            "no float was passed (the check would pass vacuously)"
        )

        normalized = permute_dims({"dims": dims}, "t")

        assert normalized == [0, 2, 1]
        assert all(type(dim) is int for dim in normalized), "not normalized to int"

    @pytest.mark.parametrize("dims", [[0, 1.5], [True, 0], [0.0, 0]])
    def test_non_integral_bool_or_duplicate_dims_are_still_rejected(self, dims):
        """The normalization does not widen the accepted set (`0.0` and `0` duplicated also fail as
        a duplicate)."""
        with pytest.raises(OpContractError, match="permute dims"):
            permute_dims({"dims": dims}, "t")
