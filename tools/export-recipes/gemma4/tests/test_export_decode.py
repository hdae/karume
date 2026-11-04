"""Script-level contract of `gemma4/export_decode.py` (the part that needs no real weights).

Emitting with real weights is manual (the same discipline as the tests of the 1-shot form). What
is pinned here is only the discipline whose breakage would give a **false PASS**:

- the RoPE hand-over inputs ({@link decode.RopeInputs}) return **exactly the values passed** per
  layer type, and both a carried-over binding and an unknown layer type fail loudly (the formula
  itself is compared with upstream by `test_rope.py`)
- the KV-sharing assignment (owning layer = the last layer of the same type before the sharing
  starts) matches upstream's rule, and `states_plan` creates one slot each for the owning layers
  only
- `assert_ir_form_decode` **actually detects** forms that "match numerically but broke silently"
  (leftovers of the previous form / a mixed-up sharing target / a misplaced window / the head_dim
  per layer type / a baked-in capacity / residue ops / a mixed-up exit / the count of storage
  dtypes)
- the greedy margin gate rejects steps at or below the floor (a margin guarantee that is not a
  tautology)
- `greedy_continuation` takes the continuation by **full re-forward** (from the start every step;
  the positions are arange)
- the band / causal masks are the 1-shot script's **as they are** (the window semantics are not
  held in two places)
- the branches of the variant that drives the 2 chunk series ({@link decode.ChunkVariant}) — the
  wrapper built, whether there is `last_row`, goldens, a provenance record, and the summary fields
  change correctly per series

Only the cases that need transformers SKIP via `importorskip` (the default-sync CI does not install
the model dependencies — the 2-job setup of ADR 0065).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared import decode_series as shared
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_token as token_only
from gemma4 import provenance
from gemma4.tests.test_export import (
    FULL_DEPTH,
    HEADS,
    HIDDEN,
    KV_HEADS,
    PLE_DIM,
    SLIDING_DEPTH,
    TINY_SYM_MAX,
    VOCAB,
    WINDOW,
)
from karume.artifacts import SUPERSEDED_SUFFIX
from karume.container import numbered_name
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_module
from karume.states import StatesFormError, StatesPlan, to_states_form

#: The tiny layer layout on the decode side. It is **the minimal form in which KV sharing holds** —
#: before the sharing starts (layer 3) there is one sliding and one full layer, each "the last layer
#: of its type", and the 2 sharing layers read one each. The 3 layers on the 1-shot side
#: (`test_export.LAYER_TYPES`) have no owning layer for the sharing layers' layer_type, so they
#: cannot be used for the decode surgery.
DECODE_LAYER_TYPES = (
    gx.SLIDING_ATTENTION,
    gx.SLIDING_ATTENTION,
    gx.FULL_ATTENTION,
    gx.SLIDING_ATTENTION,
    gx.FULL_ATTENTION,
)

#: Number of sharing layers (the last 2 = layer 3 / layer 4). The owning layers are sliding →
#: layer 1, full → layer 2.
KV_SHARED_LAYERS = 2

#: Number of owning layers (= the slots are twice this).
OWNER_LAYERS = len(DECODE_LAYER_TYPES) - KV_SHARED_LAYERS

DEPTHS = {gx.SLIDING_ATTENTION: SLIDING_DEPTH, gx.FULL_ATTENTION: FULL_DEPTH}

#: Position limit the tiny model declares (a number different from the real 131,072 — a hand copy
#: then fails).
TINY_MAX_POSITION = 256

#: Tiny RoPE declaration (sliding uses every frequency; full is `proportional` and only half
#: rotates).
TINY_ROPE_PARAMETERS = {
    gx.SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 10000.0},
    gx.FULL_ATTENTION: {
        "rope_type": "proportional",
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 0.25,
    },
}

TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    hidden_size=HIDDEN,
    head_dim=SLIDING_DEPTH,
    global_head_dim=FULL_DEPTH,
    num_hidden_layers=len(DECODE_LAYER_TYPES),
    num_kv_shared_layers=KV_SHARED_LAYERS,
    layer_types=list(DECODE_LAYER_TYPES),
    sliding_window=WINDOW,
    max_position_embeddings=TINY_MAX_POSITION,
    rope_parameters=dict(TINY_ROPE_PARAMETERS),
)

#: The tiny RoPE formula per layer type (used for the input widths of the synthetic IR and the
#: derived inputs of the wrapper calls).
TINY_ROPE_SPECS = decode.rope_specs(TINY_IR_CONFIG)

#: Storage breakdown of the positive case (i8 = the embeddings / i4 = the number of dummy linears).
STORAGE_COUNTS = {"i8": 4, "i4": 9}


# ---- RoPE hand-over inputs -----------------------------------------------


def _rope_of(layer_type: str, part: str, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Pick one of the 4 derived inputs by its spelling (the order is not assumed)."""
    return args[decode.ROPE_INPUTS.index(decode.rope_input_name(layer_type, part))]


class TestRopeInputNames:
    def test_the_names_and_order_are_the_forward_argument_names(self):
        """MUST: torch.export takes the wrapper's argument names as graph input names — a different
        spelling cannot be bound."""
        parameters = list(inspect.signature(decode.DecodeChunkWrapper.forward).parameters)

        assert parameters[1] == decode.INPUT_IDS
        assert tuple(parameters[2:]) == decode.ROPE_INPUTS

    def test_the_names_are_the_ones_the_distribution_recipe_expects(self):
        """The distribution recipe holds a copy on the side that does not import torch (the equality
        is checked here)."""
        from gemma4.distribution import GEMMA4_ROPE_INPUTS

        assert decode.ROPE_INPUTS == GEMMA4_ROPE_INPUTS

    def test_the_layer_type_order_is_the_config_order(self):
        """The order is the order of appearance in `config.layer_types` (sliding first)."""
        assert decode.unique_layer_types(TINY_IR_CONFIG) == decode.ROPE_LAYER_TYPES


class TestRopeSpecs:
    def test_it_derives_the_width_of_each_layer_type(self):
        assert TINY_ROPE_SPECS[gx.SLIDING_ATTENTION].head_dim == SLIDING_DEPTH
        assert TINY_ROPE_SPECS[gx.FULL_ATTENTION].head_dim == FULL_DEPTH
        # `proportional` rotates only 2 × int(0.25 × 32 // 2) = 8.
        assert TINY_ROPE_SPECS[gx.FULL_ATTENTION].rotary_dim == 8

    def test_a_width_that_disagrees_with_the_attention_head_dim_fails_loudly(self):
        """MUST: if rotary's head_dim source shifts, the widths of the table and of attention
        disagree.

        Upstream reads `global_head_dim` only for a full layer with `proportional`, so dropping
        rope_type to `default` makes only the table `head_dim` (16) wide — attention stays at
        `global_head_dim` (32), giving a form whose declaration can be written yet does not mesh.
        """
        parameters = {
            **TINY_ROPE_PARAMETERS,
            gx.FULL_ATTENTION: {"rope_type": "default", "rope_theta": 1000000.0},
        }
        derived = SimpleNamespace(**{**vars(TINY_IR_CONFIG), "rope_parameters": parameters})

        with pytest.raises(AssertionError, match="differs from the attention head_dim"):
            decode.rope_specs(derived)


class TestRopeArgs:
    def test_it_builds_one_pair_per_layer_type_with_the_declared_width(self):
        args = decode.rope_args(TINY_ROPE_SPECS, torch.arange(5).unsqueeze(0))

        assert len(args) == len(decode.ROPE_INPUTS)
        widths = ((gx.SLIDING_ATTENTION, SLIDING_DEPTH), (gx.FULL_ATTENTION, FULL_DEPTH))
        for layer_type, depth in widths:
            for part in decode.ROPE_PARTS:
                table = _rope_of(layer_type, part, args)
                assert tuple(table.shape) == (1, 5, depth)
                assert table.dtype is torch.float32

    def test_the_rows_follow_the_positions_it_was_given(self):
        """MUST: built from the position values (not the order) — decode's `pastLength + row` does
        not start at 0."""
        offset = decode.rope_args(TINY_ROPE_SPECS, torch.tensor([[3, 4]]))
        contiguous = decode.rope_args(TINY_ROPE_SPECS, torch.arange(5).unsqueeze(0))

        for part in decode.ROPE_PARTS:
            shifted = _rope_of(gx.SLIDING_ATTENTION, part, offset)
            whole = _rope_of(gx.SLIDING_ATTENTION, part, contiguous)
            assert torch.equal(shifted[0], whole[0, 3:5])

    def test_position_zero_is_the_identity_rotation(self):
        args = decode.rope_args(TINY_ROPE_SPECS, torch.zeros(1, 1, dtype=torch.int64))

        assert torch.equal(_rope_of(gx.FULL_ATTENTION, "cos", args)[0, 0], torch.ones(FULL_DEPTH))
        assert torch.equal(_rope_of(gx.FULL_ATTENTION, "sin", args)[0, 0], torch.zeros(FULL_DEPTH))

    def test_a_missing_layer_type_fails_loudly(self):
        with pytest.raises(AssertionError, match="no RoPE formula"):
            decode.rope_args(
                {gx.SLIDING_ATTENTION: TINY_ROPE_SPECS[gx.SLIDING_ATTENTION]},
                torch.arange(3).unsqueeze(0),
            )


class TestRopeInputs:
    def test_it_returns_the_bound_tables_untouched(self):
        """The hand-over does not touch the values (the very objects passed come back)."""
        receiver = decode.RopeInputs((gx.SLIDING_ATTENTION, gx.FULL_ATTENTION))
        sliding = (torch.zeros(1, 2, SLIDING_DEPTH), torch.ones(1, 2, SLIDING_DEPTH))
        full = (torch.full((1, 2, FULL_DEPTH), 3.0), torch.full((1, 2, FULL_DEPTH), 4.0))

        with receiver.bound({gx.SLIDING_ATTENTION: sliding, gx.FULL_ATTENTION: full}):
            got = receiver(torch.zeros(1, 1, HIDDEN), torch.zeros(1, 2), gx.FULL_ATTENTION)

        assert got[0] is full[0] and got[1] is full[1]

    def test_it_dispatches_on_the_layer_type(self):
        """MUST: running a full layer with the sliding pair (different width and frequencies) does
        not pass silently."""
        receiver = decode.RopeInputs((gx.SLIDING_ATTENTION, gx.FULL_ATTENTION))
        sliding = (torch.zeros(1, 2, SLIDING_DEPTH), torch.zeros(1, 2, SLIDING_DEPTH))
        full = (torch.ones(1, 2, FULL_DEPTH), torch.ones(1, 2, FULL_DEPTH))

        with receiver.bound({gx.SLIDING_ATTENTION: sliding, gx.FULL_ATTENTION: full}):
            narrow = receiver(torch.zeros(1), torch.zeros(1), gx.SLIDING_ATTENTION)

        assert narrow[0].shape[-1] == SLIDING_DEPTH

    def test_the_binding_is_released_after_the_call(self):
        """MUST: carried over, the next call would run with **the previous positions' tables**
        (still matching in shape and type)."""
        receiver = decode.RopeInputs((gx.SLIDING_ATTENTION,))
        tables = {gx.SLIDING_ATTENTION: (torch.zeros(1), torch.zeros(1))}

        with receiver.bound(tables):
            pass

        with pytest.raises(KeyError, match="are not bound"):
            receiver(torch.zeros(1), torch.zeros(1), gx.SLIDING_ATTENTION)

    def test_the_binding_is_released_even_when_the_body_raises(self):
        receiver = decode.RopeInputs((gx.SLIDING_ATTENTION,))
        tables = {gx.SLIDING_ATTENTION: (torch.zeros(1), torch.zeros(1))}

        with pytest.raises(RuntimeError), receiver.bound(tables):
            raise RuntimeError("fails midway")

        with pytest.raises(KeyError, match="are not bound"):
            receiver(torch.zeros(1), torch.zeros(1), gx.SLIDING_ATTENTION)

    def test_an_incomplete_binding_fails_loudly(self):
        """MUST: if even one declared layer type is missing, fail at binding time."""
        receiver = decode.RopeInputs((gx.SLIDING_ATTENTION, gx.FULL_ATTENTION))

        with (
            pytest.raises(KeyError, match=gx.FULL_ATTENTION),
            receiver.bound({gx.SLIDING_ATTENTION: (torch.zeros(1), torch.zeros(1))}),
        ):
            pass

    def test_an_unknown_layer_type_fails_loudly(self):
        receiver = decode.RopeInputs((gx.SLIDING_ATTENTION,))

        with (
            receiver.bound({gx.SLIDING_ATTENTION: (torch.zeros(1), torch.zeros(1))}),
            pytest.raises(KeyError, match=gx.FULL_ATTENTION),
        ):
            receiver(torch.zeros(1), torch.zeros(1), gx.FULL_ATTENTION)

    def test_it_carries_no_parameters_or_buffers(self):
        """MUST: the tables no longer go into the distribution (a remaining initializer would bring
        the position limit back into the asset)."""
        receiver = decode.RopeInputs((gx.SLIDING_ATTENTION, gx.FULL_ATTENTION))

        assert list(receiver.parameters()) == []
        assert list(receiver.buffers()) == []


class TestBoundRope:
    def test_it_pairs_each_layer_type_with_its_own_cos_and_sin(self):
        """MUST: a cos / sin mix-up matches in shape and type (the reason the spelling is kept in
        one place)."""
        sliding_cos, sliding_sin = torch.zeros(1), torch.ones(1)
        full_cos, full_sin = torch.full((1,), 2.0), torch.full((1,), 3.0)

        tables = decode.bound_rope(sliding_cos, sliding_sin, full_cos, full_sin)

        assert tables[gx.SLIDING_ATTENTION] == (sliding_cos, sliding_sin)
        assert tables[gx.FULL_ATTENTION] == (full_cos, full_sin)


class TestMaxPosition:
    def test_it_reads_the_model_declaration(self):
        assert decode.max_position(TINY_IR_CONFIG) == TINY_MAX_POSITION

    def test_a_non_positive_declaration_fails_loudly(self):
        derived = SimpleNamespace(**{**vars(TINY_IR_CONFIG), "max_position_embeddings": 0})

        with pytest.raises(AssertionError, match="max_position_embeddings"):
            decode.max_position(derived)


# ---- KV sharing assignment -----------------------------------------------


class TestKvOwnerLayers:
    def test_the_owner_is_the_last_layer_of_its_type_before_the_sharing_point(self):
        assert decode.first_shared_layer(TINY_IR_CONFIG) == OWNER_LAYERS
        assert decode.kv_owner_layers(TINY_IR_CONFIG) == {
            gx.SLIDING_ATTENTION: 1,
            gx.FULL_ATTENTION: 2,
        }

    def test_the_real_e2b_shape_maps_to_layers_13_and_14(self):
        """The assignment of the real E2B (35 layers / 20 sharing / a 7-cycle of 4 sliding + 1
        full)."""
        layer_types = [
            gx.FULL_ATTENTION if (index + 1) % 5 == 0 else gx.SLIDING_ATTENTION
            for index in range(35)
        ]
        config = SimpleNamespace(
            num_hidden_layers=35, num_kv_shared_layers=20, layer_types=layer_types
        )

        assert decode.kv_owner_layers(config) == {
            gx.SLIDING_ATTENTION: 13,
            gx.FULL_ATTENTION: 14,
        }
        assert decode.slot_layers(config)[15:] == tuple(
            14 if (index + 1) % 5 == 0 else 13 for index in range(15, 35)
        )

    def test_a_shared_layer_type_with_no_owner_fails_loudly(self):
        """MUST: a form that upstream would turn into a KeyError in `shared_kv_states` (the premise
        of the layer layout is broken)."""
        config = SimpleNamespace(
            num_hidden_layers=3,
            num_kv_shared_layers=1,
            layer_types=[gx.SLIDING_ATTENTION, gx.SLIDING_ATTENTION, gx.FULL_ATTENTION],
        )

        with pytest.raises(
            AssertionError, match="has no layer writing it before the sharing start"
        ):
            decode.kv_owner_layers(config)

    def test_the_layers_before_the_sharing_point_own_their_own_slots(self):
        assert decode.slot_layers(TINY_IR_CONFIG) == (0, 1, 2, 1, 2)


# ---- Synthetic IR (before and after the surgery) -------------------------


def _pre_surgery_graph(
    *,
    kv_heads: int = KV_HEADS,
    layer_types: tuple[str, ...] = DECODE_LAYER_TYPES,
    shared: int = KV_SHARED_LAYERS,
    storage_counts=STORAGE_COUNTS,
) -> IrGraph:
    """The **skeleton** of a decode chunk graph with the previous form of attention.

    What is checked is only attention / `state_append` / the states section, plus the inputs,
    outputs, residue ops and storage breakdown, so the side that makes q / k / v only has to be
    "live nodes with declared shapes" (the meaning of the ops does not matter).

    MUST: the layers are **chained in series** (layer i's q is made from layer i-1's output). An
    unchained layer disappears in the surgery's pruning (`to_states_form` drops nodes not reachable
    from the outputs), and the subject silently becomes a one-layer graph.
    MUST: the sharing layers' k / v point at **the same value names as the owning layers** (the
    shape of upstream's `shared_kv_states`). Different names make the surgery's source-match check
    fail — that is the evidence of the wiring in the real model.
    """
    boundary = len(layer_types) - shared
    owners: dict[str, int] = {
        layer_type: layer for layer, layer_type in enumerate(layer_types[:boundary])
    }
    graph = IrGraph(symbols=[decode.SEQ_SYMBOL])
    graph.inputs.append(IrInput(name=decode.INPUT_IDS, dtype="i32", shape=[1, "M"]))
    for layer_type in decode.ROPE_LAYER_TYPES:
        for part in decode.ROPE_PARTS:
            graph.inputs.append(
                IrInput(
                    name=decode.rope_input_name(layer_type, part),
                    dtype="f32",
                    shape=[1, "M", TINY_ROPE_SPECS[layer_type].head_dim],
                )
            )
    for name, shape in (
        ("tok.table", [VOCAB, HIDDEN]),
        ("mask.table", [1, 1, TINY_SYM_MAX, TINY_SYM_MAX]),
    ):
        graph.initializers[name] = IrInitializer(
            tensor=f"const.{name}", storage=IrStorage(dtype="f32")
        )
        graph.values[name] = IrValue(dtype="f32", shape=shape)

    graph.values["h"] = IrValue(dtype="f32", shape=[1, "M", HIDDEN])
    graph.values["mask"] = IrValue(dtype="f32", shape=[1, 1, "M", "M"])
    graph.nodes.append(
        IrNode(op="embedding", ins=["tok.table", decode.INPUT_IDS], outs=["h"], attrs={})
    )
    graph.nodes.append(
        IrNode(
            op=gx.SYM_PREFIX_SLICE_OP,
            ins=["mask.table"],
            outs=["mask"],
            attrs={"sym": "M", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    # MUST: make every one of the 4 derived inputs read (the surgery rejects inputs that became
    # unreachable — in the real model each layer's q / k read both the cos and the sin of its own
    # layer type).
    for layer, layer_type in enumerate(layer_types):
        depth = DEPTHS[layer_type]
        source = "h" if layer == 0 else f"attn{layer - 1}"
        cos = decode.rope_input_name(layer_type, "cos")
        sin = decode.rope_input_name(layer_type, "sin")
        graph.values[f"q{layer}"] = IrValue(dtype="f32", shape=[1, HEADS, "M", depth])
        graph.nodes.append(IrNode(op="mul", ins=[source, cos], outs=[f"q{layer}"], attrs={}))
        owner = layer if layer < boundary else owners[layer_type]
        if owner == layer:
            for slot, table in (("k", sin), ("v", cos)):
                name = f"{slot}{layer}"
                graph.values[name] = IrValue(dtype="f32", shape=[1, kv_heads, "M", depth])
                graph.nodes.append(IrNode(op="mul", ins=[source, table], outs=[name], attrs={}))
        graph.values[f"attn{layer}"] = IrValue(dtype="f32", shape=[1, HEADS, "M", depth])
        graph.nodes.append(
            IrNode(
                op=gx.ATTENTION_OP,
                ins=[f"q{layer}", f"k{owner}", f"v{owner}", "mask"],
                outs=[f"attn{layer}"],
                attrs={"scale": 1.0},
            )
        )
    graph.values["logits"] = IrValue(dtype="f32", shape=[1, "M", VOCAB])
    graph.values["token"] = IrValue(dtype="i32", shape=[1, "M", 1])
    last = f"attn{len(layer_types) - 1}"
    graph.nodes.append(IrNode(op="mul", ins=[last, last], outs=["logits"], attrs={}))
    graph.nodes.append(IrNode(op="argmax", ins=["logits"], outs=["token"], attrs={}))
    graph.outputs.extend(["logits", "token"])
    for dtype, count in storage_counts.items():
        for slot in range(count):
            name = f"weight_{dtype}_{slot}"
            graph.initializers[name] = IrInitializer(
                tensor=f"w.{dtype}.{slot}", storage=IrStorage(dtype=dtype)
            )
            graph.values[name] = IrValue(dtype="f32", shape=[2, 2])
            graph.nodes.append(IrNode(op="mul", ins=[name, "logits"], outs=[], attrs={}))
    return graph


def _surgical_graph(**kwargs) -> IrGraph:
    """The graph {@link _pre_surgery_graph} surgically turned into the states form for every
    attention."""
    graph = _pre_surgery_graph(**kwargs)
    return to_states_form(graph, decode.states_plan(graph, TINY_IR_CONFIG))


class TestStatesPlan:
    def test_the_slots_are_the_owner_layers_only(self):
        graph = _pre_surgery_graph()

        plan = decode.states_plan(graph, TINY_IR_CONFIG)

        assert plan.capacity_symbol == decode.CAPACITY_SYMBOL
        assert [(spec.k_slot, spec.v_slot) for spec in plan.attentions] == [
            ("l0.k", "l0.v"),
            ("l1.k", "l1.v"),
            ("l2.k", "l2.v"),
            # A sharing layer reads the owning layer's slots (it creates no slots of its own)
            ("l1.k", "l1.v"),
            ("l2.k", "l2.v"),
        ]
        # One-to-one with the attentions in nodes order (a mix-up reads another layer's KV)
        assert [spec.output for spec in plan.attentions] == [f"attn{layer}" for layer in range(5)]

    def test_the_window_follows_the_layer_type(self):
        plan = decode.states_plan(_pre_surgery_graph(), TINY_IR_CONFIG)

        assert [spec.window for spec in plan.attentions] == [WINDOW, WINDOW, None, WINDOW, None]
        # The capacity splits by layer type: sliding = window + slack, a number (slack so that
        # rejected rows do not overwrite the live window — SLIDING_SLACK_ROWS) / full = a symbol
        # (chosen at run time — ADR 0066 decision 3).
        slack = WINDOW + decode.SLIDING_SLACK_ROWS
        assert [spec.capacity for spec in plan.attentions] == [slack, slack, None, slack, None]

    def test_the_sliding_capacity_leaves_room_for_the_speculative_rows(self):
        """MUST: the slack goes onto the capacity, not onto the window (the `window` attrs stay)."""
        plan = decode.states_plan(_pre_surgery_graph(), TINY_IR_CONFIG)

        sliding = [spec for spec in plan.attentions if spec.window is not None]
        assert all(spec.capacity == spec.window + decode.SLIDING_SLACK_ROWS for spec in sliding)
        assert decode.SLIDING_SLACK_ROWS >= 1

    def test_a_layer_count_mismatch_fails_loudly(self):
        """MUST: a layer that was missed stays chunk-local causal (a silent wrong value)."""
        graph = _pre_surgery_graph(layer_types=DECODE_LAYER_TYPES[:3], shared=0)

        with pytest.raises(AssertionError, match=r"not matching the \d+ layers"):
            decode.states_plan(graph, TINY_IR_CONFIG)

    def test_a_node_order_that_is_not_layer_order_fails_loudly(self):
        """MUST: checks the premise itself that order of appearance = layer order (broken, the slot
        assignment shifts silently)."""
        shuffled = (
            gx.FULL_ATTENTION,
            gx.SLIDING_ATTENTION,
            gx.SLIDING_ATTENTION,
            gx.SLIDING_ATTENTION,
            gx.FULL_ATTENTION,
        )
        graph = _pre_surgery_graph(layer_types=shuffled)

        with pytest.raises(AssertionError, match="the nodes order is not the layer order"):
            decode.states_plan(graph, TINY_IR_CONFIG)


class TestAssertIrFormDecode:
    def test_the_states_form_passes(self):
        form = decode.assert_ir_form_decode(_surgical_graph(), TINY_IR_CONFIG, STORAGE_COUNTS)

        assert form["attention_nodes"] == len(DECODE_LAYER_TYPES)
        assert form["state_append_nodes"] == 2 * OWNER_LAYERS
        assert form["slots"] == 2 * OWNER_LAYERS
        assert form["kv_owners"] == {gx.SLIDING_ATTENTION: 1, gx.FULL_ATTENTION: 2}
        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["head_dim"] == {
            gx.SLIDING_ATTENTION: SLIDING_DEPTH,
            gx.FULL_ATTENTION: FULL_DEPTH,
        }
        assert form["storage"]["i4"] == STORAGE_COUNTS["i4"]

    def test_the_shared_layers_read_the_owner_slots(self):
        """The states field of a sharing layer is the owning layer's slot names (the main point of
        the form check)."""
        graph = _surgical_graph()

        attentions = [node for node in graph.nodes if node.op == gx.ATTENTION_OP]

        assert [node.states for node in attentions] == [
            {"k": "l0.k", "v": "l0.v"},
            {"k": "l1.k", "v": "l1.v"},
            {"k": "l2.k", "v": "l2.v"},
            {"k": "l1.k", "v": "l1.v"},
            {"k": "l2.k", "v": "l2.v"},
        ]

    def test_a_conventional_attention_left_behind_is_rejected(self):
        """MUST: a layer the surgery missed does not see the past (the numbers move, so goldens do
        not catch it)."""
        graph = _pre_surgery_graph()
        partial = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=decode.states_plan(graph, TINY_IR_CONFIG).attentions[:-1],
        )

        with pytest.raises(AssertionError, match="takes exactly q / k / v"):
            decode.assert_ir_form_decode(
                to_states_form(graph, partial), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_shared_layer_pointed_at_the_wrong_owner_is_rejected(self):
        """MUST: a mixed-up sharing target matches in shape and type (a silent wrong value reading
        another layer's past)."""
        graph = _pre_surgery_graph()
        specs = list(decode.states_plan(graph, TINY_IR_CONFIG).attentions)
        # Point the last full sharing layer (layer 4) at the slots of sliding owning layer 1.
        specs[4] = replace(specs[4], k_slot="l1.k", v_slot="l1.v", window=WINDOW)
        misrouted = StatesPlan(capacity_symbol=decode.CAPACITY_SYMBOL, attentions=tuple(specs))

        # The surgery itself fails with "the values written to the slot differ per reader" (the
        # evidence of the real wiring).
        with pytest.raises(StatesFormError, match="differs between readers"):
            to_states_form(graph, misrouted)

    def test_a_materialized_kv_shape_is_rejected(self):
        with pytest.raises(AssertionError, match="repeat_kv"):
            decode.assert_ir_form_decode(
                _surgical_graph(kv_heads=HEADS), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_head_dim_from_the_wrong_layer_type_is_rejected(self):
        """MUST: D is taken per layer type (fails if the full width reaches a sliding layer's q).

        The fault is injected through the graph's declared shapes — moving the config's `head_dim`
        would make the width gate of the RoPE derived inputs ({@link decode.assert_rope_inputs})
        fail first, and this check would not be reached.
        """
        graph = _surgical_graph()
        graph.values["q0"] = replace(graph.values["q0"], shape=[1, HEADS, "M", FULL_DEPTH])

        with pytest.raises(AssertionError, match="head_dim"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_window_on_a_full_layer_is_rejected(self):
        """A window on the full layers of the whole context silently discards the past outside the
        window."""
        graph = _surgical_graph()
        index, node = next(
            (index, node)
            for index, node in enumerate(graph.nodes)
            if node.op == gx.ATTENTION_OP and not node.attrs.get("window")
        )
        graph.nodes[index] = replace(node, attrs={**node.attrs, "window": WINDOW})

        with pytest.raises(AssertionError, match="window"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_missing_window_on_a_sliding_layer_is_rejected(self):
        """The reverse: without the window on the sliding layers the window has no effect (the
        whole context is read)."""
        graph = _surgical_graph()
        index, node = next(
            (index, node)
            for index, node in enumerate(graph.nodes)
            if node.op == gx.ATTENTION_OP and node.attrs.get("window")
        )
        graph.nodes[index] = replace(
            node, attrs={key: value for key, value in node.attrs.items() if key != "window"}
        )

        with pytest.raises(AssertionError, match="window"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_baked_full_capacity_is_rejected(self):
        """Baking the full capacity as a number makes the capacity unselectable when the context is
        created (ADR 0066 decision 3)."""
        graph = _pre_surgery_graph()
        baked = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=tuple(
                replace(spec, capacity=8) if spec.window is None else spec
                for spec in decode.states_plan(graph, TINY_IR_CONFIG).attentions
            ),
        )

        with pytest.raises(AssertionError, match="full capacity stays a symbol"):
            decode.assert_ir_form_decode(
                to_states_form(graph, baked), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_symbolic_sliding_capacity_is_rejected(self):
        """If the sliding capacity stays a symbol, dead rows beyond the slack come back (the side
        that passes when relaxed)."""
        graph = _pre_surgery_graph()
        symbolic = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=tuple(
                replace(spec, capacity=None)
                for spec in decode.states_plan(graph, TINY_IR_CONFIG).attentions
            ),
        )

        with pytest.raises(AssertionError, match=r"sliding is exactly window .* as a number"):
            decode.assert_ir_form_decode(
                to_states_form(graph, symbolic), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_sliding_capacity_without_the_slack_is_rejected(self):
        """A form that dropped the slack back to exactly the window (the side where the rejected
        rows of speculative verify overwrite the live window).

        Neither the `window` attrs nor the declaration move, so shape, type and IR verification all
        pass — only pinning the check to the same number as {@link decode.sliding_capacity} can
        catch it.
        """
        graph = _pre_surgery_graph()
        tight = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=tuple(
                replace(spec, capacity=spec.window) if spec.window is not None else spec
                for spec in decode.states_plan(graph, TINY_IR_CONFIG).attentions
            ),
        )

        with pytest.raises(AssertionError, match=r"sliding is exactly window .* as a number"):
            decode.assert_ir_form_decode(
                to_states_form(graph, tight), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_missing_state_append_is_rejected(self):
        graph = _surgical_graph()
        graph.nodes = [node for node in graph.nodes if node.states.get("slot") != "l2.v"]

        with pytest.raises(AssertionError, match="not one each of the"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _surgical_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "M", "M"]))

        with pytest.raises(AssertionError, match="graph inputs are"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    @pytest.mark.parametrize("op", decode.RESIDUE_OPS)
    def test_a_leftover_residue_op_is_rejected(self, op):
        """MUST: rejects a form where an unread Tmax constant / a RoPE left unfolded sits in the
        distribution."""
        graph = _surgical_graph()
        graph.nodes.append(IrNode(op=op, ins=["logits"], outs=["residue"], attrs={}))

        with pytest.raises(AssertionError, match="ops the surgery should have removed"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_an_exit_that_is_not_argmax_is_rejected(self):
        """Output 1 is the decode exit (ADR 0068 decision 4) — not to be confused with a form that
        outputs logits twice."""
        graph = _surgical_graph()
        graph.outputs[1] = "logits"

        with pytest.raises(AssertionError, match="output 1 comes from"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_single_output_graph_is_rejected(self):
        graph = _surgical_graph()
        graph.outputs.pop()

        with pytest.raises(
            AssertionError, match="the decode exit is the two outputs logits / token"
        ):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_storage_census_that_misses_a_compressed_weight_is_rejected(self):
        """MUST: a weight that falls outside eligibility **silently stays f32**, so the count
        catches it."""
        graph = _surgical_graph(storage_counts={"i8": STORAGE_COUNTS["i8"], "i4": 1})

        with pytest.raises(AssertionError, match="storage dtype counts"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_capacity_symbol_that_collides_with_an_input_symbol_fails_loudly(self):
        """A symbol mix-up (laying the capacity over a symbol from the inputs) is rejected by the
        surgery side."""
        graph = _pre_surgery_graph()
        colliding = StatesPlan(
            capacity_symbol=decode.SEQ_SYMBOL,
            attentions=decode.states_plan(graph, TINY_IR_CONFIG).attentions,
        )

        with pytest.raises(StatesFormError, match=r"is already in graph\.symbols"):
            to_states_form(graph, colliding)


# ---- Greedy margin gate / how continuations are taken --------------------


class TestGreedyMargins:
    def test_a_comfortable_margin_passes(self):
        shared.assert_greedy_margins({"a": [1.0, 0.5, 0.026], "b": [2.0]}, decode.MARGIN_FLOOR)

    def test_the_floor_stays_above_the_gate_precondition(self):
        """The producer-side floor > the consumer-side premise (2 × atol 1e-2 — Codex wave H finding
        H-04).

        Below it, a case "the script takes but that fails on the acceptance gate's margin premise"
        could be made, and the gate's "this fails only when the script and the asset disagree"
        would not hold. The source of truth for atol is PREFILL_ATOL (= 1e-2) of
        `e2e_gemma4_greedy_test.ts` — changing it means moving both.
        """
        assert decode.MARGIN_FLOOR > 2 * 1e-2

    def test_a_thin_margin_fails_loudly(self):
        """MUST: mixing steps without margin into the goldens splits the sequence on GPU
        deviations."""
        with pytest.raises(AssertionError, match=r"margin at or below the floor"):
            shared.assert_greedy_margins({"a": [1.0, 1e-3, 0.5]}, decode.MARGIN_FLOOR)

    def test_the_floor_itself_is_not_accepted(self):
        """The boundary is "exceeds" — exactly the floor does not pass."""
        with pytest.raises(AssertionError, match=r"margin at or below the floor"):
            shared.assert_greedy_margins({"a": [decode.MARGIN_FLOOR]}, decode.MARGIN_FLOOR)

    def test_every_offending_case_is_reported_at_once(self):
        """MUST: do not stop at the first one — otherwise one run does not gather the material for
        deciding which cases to drop."""
        with pytest.raises(AssertionError) as failure:
            shared.assert_greedy_margins(
                {"a": [1.0], "b": [1e-3], "c": [0.5, 0.0]}, decode.MARGIN_FLOOR
            )

        message = str(failure.value)
        assert "'b'" in message and "'c'" in message
        # Cases that passed do not appear in the diagnostic (only the names to drop are listed)
        assert "'a'" not in message


class _StubWrapper(nn.Module):
    """Subject chunk wrapper that records the call shapes (the logits are designed for the last
    position only).

    The last position's logits return a fixed row determined by "the current sequence length", so
    the expected token sequence and margin sequence can be computed by hand.
    """

    def __init__(self, rows: dict[int, list[float]]) -> None:
        super().__init__()
        self.rows = rows
        self.calls: list[tuple[list[int], list[int]]] = []

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        self.calls.append((input_ids[0].tolist(), position_ids[0].tolist()))
        length = int(input_ids.shape[1])
        logits = torch.zeros(1, length, len(self.rows[length]))
        logits[0, -1] = torch.tensor(self.rows[length])
        return logits, logits.argmax(-1, keepdim=True)


class TestGreedyContinuation:
    def test_it_re_forwards_the_whole_prefix_each_step(self):
        """MUST: full re-forward (the KV path is not used) — the expectation is not built with the
        mechanism under acceptance."""
        wrapper = _StubWrapper({2: [0.0, 3.0, 1.0], 3: [5.0, 0.0, 1.0], 4: [0.0, 0.0, 2.5]})

        tokens, margins = shared.greedy_continuation(wrapper, torch.tensor([[7, 8]]), 3)

        assert tokens == [1, 0, 2]
        assert margins == pytest.approx([2.0, 4.0, 2.5])
        assert wrapper.calls == [
            ([7, 8], [0, 1]),
            ([7, 8, 1], [0, 1, 2]),
            ([7, 8, 1, 0], [0, 1, 2, 3]),
        ]

    def test_the_positions_are_the_absolute_prefix(self):
        """The positions are `arange(len)` every step (the premise that the table-lookup RoPE can be
        looked up by real position)."""
        wrapper = _StubWrapper({3: [1.0, 0.0], 4: [0.0, 2.0]})

        shared.greedy_continuation(wrapper, torch.tensor([[4, 5, 6]]), 2)

        assert [positions for _, positions in wrapper.calls] == [[0, 1, 2], [0, 1, 2, 3]]


class TestCaseRoom:
    def test_a_prompt_that_fits_with_its_continuation_passes(self):
        shared.assert_case_room([("case", torch.zeros(1, 10, dtype=torch.int64))], 6, 16)

    def test_a_continuation_past_the_table_fails_loudly(self):
        """MUST: a position outside the table cannot be looked up (`F.embedding` does not silently
        return OOB)."""
        with pytest.raises(AssertionError, match="exceeds the RoPE table's"):
            shared.assert_case_room([("case", torch.zeros(1, 11, dtype=torch.int64))], 6, 16)

    def test_the_real_long_case_fits_the_shipped_capacity(self):
        """The long case (T=598) + the longest K fit **the distribution's default capacity** — a
        tripwire for the constants.

        The position limit became the model's declaration (131,072 for E2B), so the only
        constraint the goldens actually touch is "does it fit the default conversation capacity"
        (`gemma4.distribution.GEMMA4_CAPACITY`).
        """
        from gemma4.distribution import GEMMA4_CAPACITY

        assert 598 + max(decode.GREEDY_STEPS.values()) <= GEMMA4_CAPACITY


class TestGreedyCases:
    def test_every_greedy_case_is_a_golden_case(self):
        """MUST: do not hold case names only on the greedy side that the goldens lack (the prompt's
        origin disappears)."""
        assert set(decode.GREEDY_CASES) <= {name for name, _ in gx.GOLDEN_CASES}

    def test_every_greedy_case_has_a_first_token_expectation(self):
        """The condition for the first-continuation-token comparison (the cross-mechanism check
        against the 1-shot script) to apply to every adopted case."""
        assert set(decode.GREEDY_CASES) <= set(gx.GREEDY_EXPECTATIONS)

    def test_the_adopted_expectations_are_not_all_the_same_token(self):
        """The condition that dropping cases does not make `_sanity`'s constant-output detection
        line a tautology."""
        adopted = {gx.GREEDY_EXPECTATIONS[name] for name in decode.GREEDY_CASES}

        assert len(adopted) > 1


# ---- One pass with a tiny real model -------------------------------------


def _tiny_decode_config():
    """A tiny text config in which KV sharing holds ({@link DECODE_LAYER_TYPES})."""
    transformers = pytest.importorskip("transformers")
    return transformers.Gemma4TextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=len(DECODE_LAYER_TYPES),
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=SLIDING_DEPTH,
        global_head_dim=FULL_DEPTH,
        hidden_size_per_layer_input=PLE_DIM,
        layer_types=list(DECODE_LAYER_TYPES),
        num_kv_shared_layers=KV_SHARED_LAYERS,
        sliding_window=WINDOW,
        use_double_wide_mlp=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
    )


@pytest.fixture
def tiny_wrapper():
    """{@link decode.DecodeChunkWrapper} built from a tiny real model (no real weights are read)."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    gx.register_attention()
    config = _tiny_decode_config()
    config._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
    tables = nn.ModuleList(
        [
            nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
            for _ in DECODE_LAYER_TYPES
        ]
    )
    del model.model.embed_tokens_per_layer
    decode.swap_rope_inputs(model)
    return decode.DecodeChunkWrapper(model, tables).eval()


class TestExportedDecodeForm:
    """Actually take a tiny real model through export → surgery → container verification (needs
    transformers)."""

    @pytest.fixture
    def tiny_container(self, tiny_wrapper, tmp_path):
        """`(pre-surgery, verified, …)` after one pass in **the same order with the same storage
        settings** as `export_series`.

        MUST: exercise it including mixed quantization (default i8 + linear i4 + an explicit f32
        for the RoPE tables) — the storage plan applies to the graph **after** the surgery, so a
        pass written in f32 shows none of the decode-specific breakages such as "a setting for a
        pruned weight remains" or "the looked-up cos/sin become eligible for i8" (the latter was
        actually found here — 2026-08-19).
        """
        int8, int4, scales = gx.quantize_wrapper(tiny_wrapper)
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        specs = decode.rope_specs(tiny_wrapper.model.config)
        args, shapes = decode._export_args(decode.DECODE, ids, seq, specs)
        graph, tensors = export_module(
            tiny_wrapper,
            args,
            dynamic_shapes=shapes,
            symbol_names=(decode.SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        config = tiny_wrapper.model.config
        surgical = to_states_form(graph, decode.states_plan(graph, config))
        verified = decode._write_container(
            surgical,
            tensors,
            tmp_path / gx.MODEL_FILE,
            graph_name="tiny",
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides=dict.fromkeys(int4.scales, "i4"),
        )
        storage = {"i8": len(int8.scales), "i4": len(int4.scales)}
        return SimpleNamespace(pre=graph, verified=verified, config=config, storage=storage)

    def test_the_container_is_a_verified_states_form_graph(self, tiny_container):
        form = decode.assert_ir_form_decode(
            tiny_container.verified, tiny_container.config, tiny_container.storage
        )

        assert form["attention_nodes"] == len(DECODE_LAYER_TYPES)
        assert form["state_append_nodes"] == 2 * OWNER_LAYERS
        assert form["kv_owners"] == {gx.SLIDING_ATTENTION: 1, gx.FULL_ATTENTION: 2}
        assert [spec.name for spec in tiny_container.verified.inputs] == [
            decode.INPUT_IDS,
            *decode.ROPE_INPUTS,
        ]
        assert sorted(tiny_container.verified.symbols) == [
            decode.CAPACITY_SYMBOL,
            decode.SEQ_SYMBOL,
        ]
        # The surgery always prunes something (the mask constants of the 2 layer types) — 0 means
        # the folding did not take effect
        assert len(tiny_container.pre.initializers) > len(tiny_container.verified.initializers)

    def test_the_mask_residue_is_gone_but_the_pre_surgery_graph_had_it(self, tiny_container):
        """Gate against tautology: checks on the same asset that what gets pruned **actually
        existed** on the export side."""
        assert gx.SYM_PREFIX_SLICE_OP in tiny_container.pre.required_ops
        assert gx.SYM_PREFIX_SLICE_OP not in tiny_container.verified.required_ops

    def test_the_rope_arrives_as_graph_inputs_with_no_table_left(self, tiny_container):
        """Since the positions left the graph, RoPE is the input itself, and neither tables nor
        trigonometric functions remain.

        MUST: check down to the absence of `sin` (the only reachable residue of an unfolded RoPE —
        `cos` is not in the IR vocabulary, so if it remained the export itself would fail).
        """
        verified = tiny_container.verified

        assert [spec.name for spec in verified.inputs][1:] == list(decode.ROPE_INPUTS)
        assert "sin" not in verified.required_ops
        assert not [
            name for name, init in verified.initializers.items() if "rotary_emb." in init.tensor
        ]
        # The positional-table gather is gone, so `embedding` is only the main embedding + the PLE
        # split.
        readers = [node for node in verified.nodes if node.op == "embedding"]
        assert len(readers) == 1 + len(DECODE_LAYER_TYPES)

    def test_every_linear_weight_is_stored_as_int4(self, tiny_container):
        """MUST: if the surgery pruned a weight, the i4 count does not match (the detection line for
        a weight silently left in f32)."""
        storage = {
            name: init.storage.dtype for name, init in tiny_container.verified.initializers.items()
        }

        assert sum(dtype == "i4" for dtype in storage.values()) == tiny_container.storage["i4"]
        assert sum(dtype == "i8" for dtype in storage.values()) == tiny_container.storage["i8"]

    def test_the_masks_come_from_the_one_shot_recipe(self, tiny_wrapper, monkeypatch):
        """The window semantics are not rewritten on the decode side (the source of truth is the
        1-shot script's two masks).

        MUST: rebuilding the band would allow a form where "the 1-shot goldens and the decode
        goldens are taken with different windows" (each is consistent within itself, so it goes
        unnoticed without a comparison).
        """
        seen: list[tuple[str, int, int | None]] = []

        def spy(name, original):
            def wrapped(length, *rest):
                seen.append((name, int(length), *(int(value) for value in rest)))
                return original(length, *rest)

            return wrapped

        monkeypatch.setattr(gx, "additive_causal_mask", spy("causal", gx.additive_causal_mask))
        monkeypatch.setattr(gx, "additive_sliding_mask", spy("band", gx.additive_sliding_mask))
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)

        specs = decode.rope_specs(tiny_wrapper.model.config)
        with torch.no_grad():
            tiny_wrapper(ids, *decode.rope_args(specs, decode.positions_for(ids)))

        assert seen == [("causal", WINDOW + 3), ("band", WINDOW + 3, WINDOW)]

    def test_the_wrapper_returns_logits_and_the_greedy_token(self, tiny_wrapper):
        """MUST: the output order is `[logits, token]` (the runtime reads by slot number)."""
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)
        specs = decode.rope_specs(tiny_wrapper.model.config)

        with torch.no_grad():
            logits, token = tiny_wrapper(ids, *decode.rope_args(specs, decode.positions_for(ids)))

        assert logits.shape == (1, 5, VOCAB)
        assert token.shape == (1, 5, 1)
        assert torch.equal(token, logits.argmax(-1, keepdim=True))

    def test_the_shared_layers_reuse_the_owner_kv_tensors(self, tiny_wrapper):
        """The premise for the surgery's source-match check to work — upstream wires the same k/v
        values to the sharing layers.

        MUST: the check looks at **value names** of the exported graph (not eager tensor identity) —
        those value names are what the surgery compares.
        """
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        specs = decode.rope_specs(tiny_wrapper.model.config)
        args, shapes = decode._export_args(decode.DECODE, ids, seq, specs)
        graph, _ = export_module(
            tiny_wrapper,
            args,
            dynamic_shapes=shapes,
            symbol_names=(decode.SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        attentions = [node for node in graph.nodes if node.op == gx.ATTENTION_OP]
        sources = [tuple(node.ins[1:3]) for node in attentions]

        assert sources[3] == sources[1]
        assert sources[4] == sources[2]


class TestPublish:
    """The staging → final swap (`_publish`).

    What it protects is "**no regular asset can mix old and new**". With a form that writes file by
    file into final, a run that failed midway through the gates leaves "new model + old greedy",
    and the acceptance gate could go green on a rejected asset.
    """

    @staticmethod
    def _series(root, name: str, files: dict[str, str]):
        directory = root / name
        directory.mkdir()
        for filename, body in files.items():
            (directory / filename).write_text(body)
        return directory

    def test_a_fresh_target_is_created(self, tmp_path):
        staging = self._series(tmp_path, "staging", {"model": "new"})

        shared._publish(staging, tmp_path / "final")

        assert (tmp_path / "final" / "model").read_text() == "new"
        assert not staging.exists()

    def test_an_existing_target_is_replaced_wholesale(self, tmp_path):
        """A file only the old one has (e.g. the greedy of an excluded case) does not remain in
        final."""
        final = self._series(tmp_path, "final", {"model": "old", "greedy.stale": "old"})
        staging = self._series(tmp_path, "staging", {"model": "new"})

        shared._publish(staging, final)

        assert (final / "model").read_text() == "new"
        assert not (final / "greedy.stale").exists()
        assert not list(tmp_path.glob("final" + SUPERSEDED_SUFFIX))

    def test_a_failed_promotion_restores_the_old_series(self, tmp_path, monkeypatch):
        """Even if the promoting rename fails, final stays **the complete old asset** (neither
        absent nor mixed).

        NOTE: `os.replace` fails only for the one staging promotion (the backup and the restore go
        through the real one).
        """
        final = self._series(tmp_path, "final", {"model": "old"})
        staging = self._series(tmp_path, "staging", {"model": "new"})
        real_replace = os.replace

        def failing_replace(src, dst):
            if Path(src) == staging:
                raise OSError("promotion failed")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError, match="promotion failed"):
            shared._publish(staging, final)

        assert (final / "model").read_text() == "old"
        assert staging.exists()


# ---- The core of the variant drive ---------------------------------------


class TestChunkVariants:
    """The 4 fields the descriptor of the 2 chunk series ({@link decode.ChunkVariant}) declares."""

    def test_the_logits_series_declares_the_two_output_exit_with_goldens(self):
        assert decode.DECODE.wrapper is decode.DecodeChunkWrapper
        assert decode.DECODE.token_only is False
        assert decode.DECODE.goldens is True

    def test_the_token_only_series_declares_the_other_exit_without_goldens(self):
        """MUST: no goldens are taken because the acceptance gate reuses the logits series' greedy
        records."""
        assert token_only.VARIANT.wrapper is token_only.TokenOnlyChunkWrapper
        assert token_only.VARIANT.token_only is True
        assert token_only.VARIANT.goldens is False

    def test_the_two_series_land_in_different_directories(self):
        """Sharing the location would let the 2 assets with different exits overwrite each other."""
        assert decode.DECODE.out_dir != token_only.VARIANT.out_dir

    def test_both_wrappers_share_the_chunk_module_space(self):
        """MUST: the condition for reusing the quantization target predicates and the scale ledger
        (identity of the FQN space)."""
        assert issubclass(token_only.VARIANT.wrapper, decode.DECODE.wrapper)


class TestExportArgs:
    @pytest.fixture
    def seq(self):
        return Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)

    def test_the_logits_form_traces_ids_and_the_rope_inputs(self, seq):
        ids = torch.zeros(1, 5, dtype=torch.int64)

        args, shapes = decode._export_args(decode.DECODE, ids, seq, TINY_ROPE_SPECS)

        assert len(args) == 1 + len(decode.ROPE_INPUTS)
        # The example's positions are the `arange` of the full unpadded length (the RoPE rows
        # correspond directly to positions).
        assert torch.equal(
            _rope_of(gx.SLIDING_ATTENTION, "cos", args[1:]),
            decode.rope_args(TINY_ROPE_SPECS, torch.arange(5).unsqueeze(0))[0],
        )
        # The row count is the same symbol as the chunk row count (a separate symbol would allow
        # declaring a form that can be passed tables with a different row count).
        assert shapes == tuple({1: seq} for _ in args)

    def test_the_token_only_form_adds_the_last_row_as_a_static_input(self, seq):
        """MUST: `last_row` points at the last valid row T−1 and has no symbolic dimension (it is
        not tied to M)."""
        ids = torch.zeros(1, 5, dtype=torch.int64)

        args, shapes = decode._export_args(token_only.VARIANT, ids, seq, TINY_ROPE_SPECS)

        assert len(args) == 2 + len(decode.ROPE_INPUTS)
        assert args[-1].tolist() == [4]
        assert args[-1].dtype is torch.int64
        assert shapes[-1] is None


@pytest.fixture
def tiny_materials(monkeypatch):
    """Replace `one_shot.load_model_and_tables` with a tiny model (the stand-in that reads no real
    weights)."""
    transformers = pytest.importorskip("transformers")

    def materials(model_dir):
        torch.manual_seed(0)
        gx.register_attention()
        config = _tiny_decode_config()
        config._attn_implementation = gx.ATTENTION_NAME
        model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
        tables = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
                for _ in DECODE_LAYER_TYPES
            ]
        )
        return model, tables

    monkeypatch.setattr(gx, "load_model_and_tables", materials)


class TestLoadWrapper:
    @pytest.mark.parametrize(
        "variant",
        [decode.DECODE, token_only.VARIANT],
        ids=["logits", "token-only"],
    )
    def test_it_builds_the_variant_wrapper_on_a_host_rope(self, variant, tiny_materials):
        """The variant changes only the wrapper type (how the source tensors are read and the RoPE
        replacement are shared)."""
        wrapper = decode.load_wrapper(variant, Path("unused"))

        assert type(wrapper) is variant.wrapper
        assert isinstance(wrapper.model.model.rotary_emb, decode.RopeInputs)
        assert wrapper.model.model.rotary_emb.layer_types == decode.ROPE_LAYER_TYPES
        # The probe model's PLE table has been dropped (the condition for the quantization target
        # coverage — the same text as the 1-shot script).
        assert not hasattr(wrapper.model.model, "embed_tokens_per_layer")
        assert not wrapper.training


#: Case names used for one pass of the driver. Unless the names are in **both** the greedy adopted
#: set and the 1-shot expectation table, the series' gates (narrowing to `GREEDY_CASES` and the
#: comparison of the first continuation) take a different branch from the real one.
TINY_CASE_NAMES = ("capital-en", "capital-ja")

#: File names of the container the tiny series places. The distribution is **always split**
#: (container-v1 §8), so no single file comes out — the tiny model is a few KB, so it is the 3
#: parts "2 descriptors + const region + weights". The numbering is not baked in but taken from
#: {@link karume.container.numbered_name} (if the rule moves, this moves with it).
TINY_CONTAINER_PARTS = 3
TINY_CONTAINER_FILES = [
    numbered_name(gx.MODEL_FILE, index, TINY_CONTAINER_PARTS)
    for index in range(1, TINY_CONTAINER_PARTS + 1)
]

#: Summary fields per series (order included). If this changes, the shape of the run records
#: changes.
DECODE_SUMMARY_KEYS = [
    "dir",
    "nodes",
    "outputs",
    "initializers",
    "pruned_initializers",
    "model_bytes",
    "ops",
    "symbols",
    "io",
    "greedy",
    "greedy_steps",
    "case_lengths",
    "quantized",
    "form",
    "margin_min",
    "continuation",
    "sanity",
]
TOKEN_ONLY_SUMMARY_KEYS = [
    "dir",
    "nodes",
    "outputs",
    "initializers",
    "model_bytes",
    "ops",
    "symbols",
    "case_lengths",
    "quantized",
    "form",
    "reference",
    "sanity",
]


class _StubSeriesTokenizer:
    """Subject tokenizer with only the 3 surfaces needed for looking up the expectation table, the
    labels and decoding the continuation."""

    def encode(self, text: str, add_special_tokens: bool = True):
        """Matches only the calling convention of `tokenizers.Tokenizer.encode` (returns a single
        token)."""
        return SimpleNamespace(ids=[len(text) % VOCAB])

    def id_to_token(self, token: int) -> str:
        return f"<{token}>"

    def decode(self, tokens) -> str:
        return " ".join(f"<{token}>" for token in tokens)


@pytest.fixture
def tiny_series(monkeypatch, tiny_materials, tmp_path_factory):
    """The set of replacements for one pass of {@link decode.export_series} without reading real
    assets.

    What is replaced is **the origin of the source material** (model, cases, tokenizer) and only
    the 2 expectations a tiny random-weight model cannot satisfy — the continuation margin (the
    floor is pinned by `TestGreedyMargins`) and the last position's top-1 (pinned by
    `test_export.TestSanity`). The path quantization → export → surgery → write → form check →
    publication goes through the real thing.

    NOTE: the source-material directories (the checkpoint to fingerprint and the reused golden
    series) are created **outside** `tmp_path` — created inside, the check "neither the staging
    directory nor the backup directory remains" would count the source material too.
    """
    torch.manual_seed(1)
    cases = tuple(
        (name, torch.randint(0, VOCAB, (1, WINDOW + 3 - index), dtype=torch.int64))
        for index, name in enumerate(TINY_CASE_NAMES)
    )
    checkpoint = tmp_path_factory.mktemp("checkpoint")
    for name in provenance.FINGERPRINT_FILES:
        (checkpoint / name).write_bytes(name.encode())
    reference = tmp_path_factory.mktemp("reference-series")
    for name, ids in cases:
        save_file(
            {shared.PROMPT_KEY: ids[0].to(torch.int32).contiguous()},
            str(reference / f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}"),
        )
    monkeypatch.setattr(gx, "build_cases", lambda model_dir, sym_max, window: cases)
    monkeypatch.setattr(gx, "load_tokenizer", lambda model_dir: _StubSeriesTokenizer())
    monkeypatch.setattr(decode, "MARGIN_FLOOR", 0.0)
    seen: dict[str, dict[str, int]] = {}

    def record(greedy, expected, labels):
        seen["greedy"] = dict(greedy)
        return {"stub": "ok"}

    monkeypatch.setattr(gx, "_sanity", record)
    return SimpleNamespace(cases=cases, sanity=seen, checkpoint=checkpoint, reference=reference)


class TestExportSeries:
    """One pass of the variant drive (tiny model). What changes per series is the exit, the goldens,
    the unit placed and the summary fields."""

    def test_the_logits_series_publishes_the_container_with_its_goldens(
        self, tiny_series, tmp_path
    ):
        out_dir = tmp_path / "series"

        summary = decode.export_series(
            decode.DECODE,
            tmp_path / "unused",
            out_dir,
            sym_max=TINY_SYM_MAX,
            steps=1,
        )

        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [
                "greedy.capital-en.safetensors",
                "greedy.capital-ja.safetensors",
                "io.capital-en.safetensors",
                "io.capital-ja.safetensors",
                *TINY_CONTAINER_FILES,
            ]
        )
        assert summary["outputs"] == 2
        assert list(summary) == DECODE_SUMMARY_KEYS
        assert set(tiny_series.sanity["greedy"]) == set(TINY_CASE_NAMES)
        # Neither the staging directory nor the backup directory remains (the cleanup of the swap
        # is the core primitive's job).
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_the_token_only_series_publishes_the_container_with_its_provenance(
        self, tiny_series, tmp_path
    ):
        """A series that takes no goldens places "container + provenance record" (writes no io /
        greedy)."""
        out_dir = tmp_path / "series"

        summary = decode.export_series(
            token_only.VARIANT,
            tiny_series.checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            reference=tiny_series.reference,
        )

        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [*TINY_CONTAINER_FILES, "reference.json"]
        )
        assert summary["outputs"] == 1
        assert list(summary) == TOKEN_ONLY_SUMMARY_KEYS
        # sanity covers every case (there are no greedy records, so it is taken by full-length
        # forward).
        assert set(tiny_series.sanity["greedy"]) == set(TINY_CASE_NAMES)
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_the_provenance_record_binds_the_reference_goldens(self, tiny_series, tmp_path):
        """The record binds "the fingerprint of the source checkpoint" and "the digests of the
        reused goldens"."""
        out_dir = tmp_path / "series"

        decode.export_series(
            token_only.VARIANT,
            tiny_series.checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            reference=tiny_series.reference,
        )

        record = json.loads((out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8"))
        assert record["schema"] == provenance.SCHEMA
        assert record["series"] == out_dir.name
        assert record["checkpoint"]["dir"] == tiny_series.checkpoint.name
        assert sorted(record["checkpoint"]["files"]) == sorted(provenance.FINGERPRINT_FILES)
        assert record["reference"]["series"] == tiny_series.reference.name
        goldens = record["reference"]["goldens"]
        assert sorted(goldens) == sorted(
            f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in TINY_CASE_NAMES
        )
        for file, digest in goldens.items():
            raw = (tiny_series.reference / file).read_bytes()
            assert digest == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def test_a_missing_reference_golden_stops_the_token_only_series(self, tiny_series, tmp_path):
        """MUST: a combination whose reused series is incomplete fails before the staging directory
        is created (not a single asset is left)."""
        out_dir = tmp_path / "series"
        for path in tiny_series.reference.iterdir():
            path.unlink()

        with pytest.raises(AssertionError, match="reference golden"):
            decode.export_series(
                token_only.VARIANT,
                tiny_series.checkpoint,
                out_dir,
                sym_max=TINY_SYM_MAX,
                reference=tiny_series.reference,
            )

        assert list(tmp_path.iterdir()) == []

    def test_a_failing_gate_leaves_nothing_behind(self, tiny_series, tmp_path, monkeypatch):
        """MUST: nothing goes to final before the gates (a failed run leaves no asset that could
        pass acceptance)."""

        def refuse(greedy, expected, labels):
            raise AssertionError(
                "the top token at the last position differs from the expected continuation"
            )

        monkeypatch.setattr(gx, "_sanity", refuse)
        out_dir = tmp_path / "series"

        with pytest.raises(AssertionError, match="differs from the expected continuation"):
            decode.export_series(
                decode.DECODE,
                tmp_path / "unused",
                out_dir,
                sym_max=TINY_SYM_MAX,
                steps=1,
            )

        assert list(tmp_path.iterdir()) == []
