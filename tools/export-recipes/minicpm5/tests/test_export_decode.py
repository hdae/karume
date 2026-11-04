"""Script-level guarantees of `minicpm5/export_decode.py` (the part that needs no real weights).

Emitting with real weights is manual (the same rule as the one-shot form's tests). What is pinned
here is only the rules whose breakage gives a **false PASS**:

- the RoPE table lookup is **value-identical** to the original implementation (the very
  justification of the replacement), and an implementation the table cannot express (a forward
  that depends on more than the position) fails the comparison
- `states_plan` assigns layer-numbered slots in node order and fails on a count mismatch
- `assert_ir_form_decode` **actually detects** the forms that "agree numerically but are quietly
  broken" (a conventional form left over / repeat_kv materialized / a window attached / a baked-in
  capacity / residue ops / a mixed-up exit)
- the greedy margin check fails steps at or below the floor (a margin guarantee that is not
  vacuous)
- `greedy_continuation` takes the sequence by **full re-forward** (from the start every step, the
  positions are an arange)

Only the cases that need transformers SKIP via `importorskip` (the default-sync CI does not install
the model dependencies — the 2-job layout of ADR 0065).
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.export import Dim

from _shared import decode_series as shared
from karume.artifacts import SUPERSEDED_SUFFIX
from karume.container import Provenance
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_module
from karume.states import StateAttentionSpec, StatesFormError, StatesPlan, to_states_form
from minicpm5 import export as one_shot
from minicpm5 import export_decode as decode
from minicpm5.tests.test_export import (
    HEAD_DIM,
    HEADS,
    HIDDEN,
    KV_HEADS,
    LAYERS,
    TINY_SYM_MAX,
    VOCAB,
    _tiny_checkpoint,
)

#: Number of positions of the tiny RoPE table (the probes of
#: {@link decode.assert_rope_table_matches} hit 17 and `positions - 1`, so it needs 18 or more).
TINY_POSITIONS = 32

#: Tmax of the constant the additive causal mask is baked into (residue the surgery prunes —
#: ADR 0010).
TINY_TMAX = TINY_SYM_MAX

TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    num_hidden_layers=LAYERS,
)


class _StubRotary(nn.Module):
    """A subject with only the calling convention of `LlamaRotaryEmbedding`.

    The values are **a deterministic function of the position alone** (a form the table lookup can
    reproduce exactly). `x` is taken only for its dtype / device, and as in the upstream
    implementation its value is not read.
    """

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.scale = torch.arange(depth, dtype=torch.float32) * 0.017 + 0.31

    def angles(self, position_ids: torch.Tensor) -> torch.Tensor:
        return position_ids.unsqueeze(-1).to(torch.float32) * self.scale

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        del x
        angles = self.angles(position_ids)
        return angles.cos(), angles.sin()


class _LengthDependentRotary(_StubRotary):
    """A rotary that also depends on something **other than** the position (the sequence length) — a
    subject the table lookup cannot reproduce in principle."""

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        del x
        angles = self.angles(position_ids) + float(position_ids.shape[1])
        return angles.cos(), angles.sin()


@pytest.fixture
def probe() -> torch.Tensor:
    """A dummy input from which the rotary reads dtype / device."""
    return torch.zeros(1, 1, HIDDEN, dtype=torch.float32)


class TestRopeTable:
    def test_the_lookup_reproduces_the_reference(self, probe):
        rotary = _StubRotary(HEAD_DIM)

        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS)

        # No exception means torch.equal on every probe (values, shape, dtype and tuple structure
        # all match).
        decode.assert_rope_table_matches(rotary, table, probe, TINY_POSITIONS)
        assert table.cos_table.shape == (TINY_POSITIONS, HEAD_DIM)
        assert table.sin_table.shape == (TINY_POSITIONS, HEAD_DIM)

    def test_a_reference_that_is_not_a_function_of_position_is_rejected(self, probe):
        """MUST: silently lowering an implementation the table cannot express to a table shifts the
        values "plausibly"."""
        rotary = _LengthDependentRotary(HEAD_DIM)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS)

        with pytest.raises(AssertionError, match="the table lookup is misaligned"):
            decode.assert_rope_table_matches(rotary, table, probe, TINY_POSITIONS)

    def test_a_shifted_table_is_rejected(self, probe):
        """Fault injection: shifting the table by one row fails the comparison (= evidence that the
        green above is not vacuous)."""
        rotary = _StubRotary(HEAD_DIM)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS)
        shifted = decode.RopeTable(table.cos_table.roll(1, 0), table.sin_table)

        with pytest.raises(AssertionError, match="the table lookup is misaligned"):
            decode.assert_rope_table_matches(rotary, shifted, probe, TINY_POSITIONS)

    def test_the_swap_leaves_the_model_untouched_when_the_check_fails(self, probe):
        """MUST: a failed comparison does not leave the model half-changed (a table lookup with
        wrong values)."""
        model = SimpleNamespace(
            model=SimpleNamespace(rotary_emb=_LengthDependentRotary(HEAD_DIM)),
            config=SimpleNamespace(hidden_size=HIDDEN),
        )
        original = model.model.rotary_emb

        with pytest.raises(AssertionError):
            decode.swap_rope_table(model, TINY_POSITIONS)

        assert model.model.rotary_emb is original


def _pre_surgery_graph(kv_heads: int = KV_HEADS, layers: int = LAYERS) -> IrGraph:
    """The **skeleton** of a decode chunk graph with `layers` conventional-form attention nodes.

    The checks look only at attention / `state_append` / the states section and at the presence
    of inputs, outputs and residue ops, so what produces q / k / v only has to be "live nodes with
    declared shapes" (the meaning of the ops does not matter).

    MUST: the layers are **chained in series** (layer i's q/k/v are made from layer i-1's output).
    An unchained layer vanishes in the surgery's pruning (`to_states_form` drops nodes not
    reachable from the outputs), and the subject silently becomes a 1-layer graph.
    """
    graph = IrGraph(symbols=[decode.SEQ_SYMBOL])
    graph.inputs.append(IrInput(name=decode.INPUT_IDS, dtype="i32", shape=[1, "M"]))
    graph.inputs.append(IrInput(name=decode.POSITION_IDS, dtype="i32", shape=[1, "M"]))
    for name, shape in (
        ("tok.table", [VOCAB, HIDDEN]),
        ("rope.cos", [TINY_POSITIONS, HEAD_DIM]),
        ("mask.table", [1, 1, TINY_TMAX, TINY_TMAX]),
    ):
        graph.initializers[name] = IrInitializer(
            tensor=f"const.{name}", storage=IrStorage(dtype="f32")
        )
        graph.values[name] = IrValue(dtype="f32", shape=shape)

    graph.values["h"] = IrValue(dtype="f32", shape=[1, "M", HIDDEN])
    graph.values["cos"] = IrValue(dtype="f32", shape=[1, "M", HEAD_DIM])
    graph.values["mask"] = IrValue(dtype="f32", shape=[1, 1, "M", "M"])
    graph.nodes.append(
        IrNode(op="embedding", ins=["tok.table", decode.INPUT_IDS], outs=["h"], attrs={})
    )
    graph.nodes.append(
        IrNode(op="embedding", ins=["rope.cos", decode.POSITION_IDS], outs=["cos"], attrs={})
    )
    graph.nodes.append(
        IrNode(
            op=one_shot.SYM_PREFIX_SLICE_OP,
            ins=["mask.table"],
            outs=["mask"],
            attrs={"sym": "M", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    for layer in range(layers):
        source = "h" if layer == 0 else f"attn{layer - 1}"
        for slot, count in (("q", HEADS), ("k", kv_heads), ("v", kv_heads)):
            name = f"{slot}{layer}"
            graph.values[name] = IrValue(dtype="f32", shape=[1, count, "M", HEAD_DIM])
            graph.nodes.append(IrNode(op="mul", ins=[source, "cos"], outs=[name], attrs={}))
        graph.values[f"attn{layer}"] = IrValue(dtype="f32", shape=[1, HEADS, "M", HEAD_DIM])
        graph.nodes.append(
            IrNode(
                op=one_shot.ATTENTION_OP,
                ins=[f"q{layer}", f"k{layer}", f"v{layer}", "mask"],
                outs=[f"attn{layer}"],
                attrs={"scale": 0.5},
            )
        )
    graph.values["logits"] = IrValue(dtype="f32", shape=[1, "M", VOCAB])
    graph.values["token"] = IrValue(dtype="i32", shape=[1, "M", 1])
    graph.nodes.append(
        IrNode(op="mul", ins=[f"attn{layers - 1}", f"attn{layers - 1}"], outs=["logits"], attrs={})
    )
    graph.nodes.append(IrNode(op="argmax", ins=["logits"], outs=["token"], attrs={}))
    graph.outputs.extend(["logits", "token"])
    return graph


def _surgical_graph(**kwargs) -> IrGraph:
    """{@link _pre_surgery_graph} after surgery to the states form for every attention."""
    graph = _pre_surgery_graph(**kwargs)
    return to_states_form(graph, decode.states_plan(graph, kwargs.get("layers", LAYERS)))


class TestStatesPlan:
    def test_the_slots_are_named_per_layer_in_node_order(self):
        graph = _pre_surgery_graph()

        plan = decode.states_plan(graph, LAYERS)

        assert plan.capacity_symbol == decode.CAPACITY_SYMBOL
        assert [(spec.k_slot, spec.v_slot) for spec in plan.attentions] == [
            ("l0.k", "l0.v"),
            ("l1.k", "l1.v"),
        ]
        # One to one with the attention nodes in node order (a mix-up would write into another
        # layer's KV)
        assert [spec.output for spec in plan.attentions] == ["attn0", "attn1"]
        assert all(spec.window is None and spec.capacity is None for spec in plan.attentions)

    def test_a_layer_count_mismatch_fails_loudly(self):
        """MUST: a missed layer stays chunk-local causal (a silent wrong value)."""
        graph = _pre_surgery_graph(layers=1)

        with pytest.raises(AssertionError, match=r"does not match the \d+ layers"):
            decode.states_plan(graph, LAYERS)


class TestAssertIrFormDecode:
    def test_the_states_form_passes(self):
        form = decode.assert_ir_form_decode(_surgical_graph(), TINY_IR_CONFIG)

        assert form["attention_nodes"] == LAYERS
        assert form["state_append_nodes"] == 2 * LAYERS
        assert form["slots"] == 2 * LAYERS
        assert form["slot_shape"] == [1, KV_HEADS, decode.CAPACITY_SYMBOL, HEAD_DIM]
        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]

    def test_a_conventional_attention_left_behind_is_rejected(self):
        """MUST: a layer the surgery missed does not look at the past (the numbers move, so the
        golden does not catch it)."""
        graph = _pre_surgery_graph()
        partial = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=(StateAttentionSpec(output="attn0", k_slot="l0.k", v_slot="l0.v"),),
        )

        with pytest.raises(AssertionError, match="exactly 3: q / k / v"):
            decode.assert_ir_form_decode(to_states_form(graph, partial), TINY_IR_CONFIG)

    def test_a_materialized_kv_shape_is_rejected(self):
        with pytest.raises(AssertionError, match="repeat_kv"):
            decode.assert_ir_form_decode(_surgical_graph(kv_heads=HEADS), TINY_IR_CONFIG)

    def test_a_derived_head_dim_is_rejected(self):
        derived = SimpleNamespace(
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HIDDEN // HEADS,
            num_hidden_layers=LAYERS,
        )

        with pytest.raises(AssertionError, match=r"config\.head_dim"):
            decode.assert_ir_form_decode(_surgical_graph(), derived)

    def test_a_window_attribute_is_rejected(self):
        """A window attached to a full layer over the whole context silently cuts off the past
        outside the window."""
        graph = _surgical_graph()
        index, node = next(
            (index, node)
            for index, node in enumerate(graph.nodes)
            if node.op == one_shot.ATTENTION_OP
        )
        graph.nodes[index] = replace(node, attrs={**node.attrs, "window": 4})

        with pytest.raises(AssertionError, match="window"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_baked_capacity_is_rejected(self):
        """Baking the capacity as a number leaves no way to choose it in `createGenerationContext`
        (ADR 0066 decision 3)."""
        graph = _pre_surgery_graph()
        baked = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=tuple(
                StateAttentionSpec(
                    output=f"attn{layer}",
                    k_slot=decode.slot_name(layer, "k"),
                    v_slot=decode.slot_name(layer, "v"),
                    capacity=8,
                )
                for layer in range(LAYERS)
            ),
        )

        with pytest.raises(AssertionError, match="the capacity MUST stay a symbol"):
            decode.assert_ir_form_decode(to_states_form(graph, baked), TINY_IR_CONFIG)

    def test_a_missing_state_append_is_rejected(self):
        graph = _surgical_graph()
        graph.nodes = [node for node in graph.nodes if node.states.get("slot") != "l1.v"]

        with pytest.raises(AssertionError, match="not one per slot"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _surgical_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "M", "M"]))

        with pytest.raises(AssertionError, match="graph inputs"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    @pytest.mark.parametrize("op", decode.RESIDUE_OPS)
    def test_a_leftover_residue_op_is_rejected(self, op):
        """MUST: reject the form where a Tmax constant nobody reads / RoPE left unfolded sits in the
        distribution."""
        graph = _surgical_graph()
        graph.nodes.append(IrNode(op=op, ins=["logits"], outs=["residue"], attrs={}))

        with pytest.raises(AssertionError, match="ops that the surgery should have removed"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_an_exit_that_is_not_argmax_is_rejected(self):
        """Output 1 is the decode exit (ADR 0068 decision 4) — not to be confused with a form that
        emits the logits twice."""
        graph = _surgical_graph()
        graph.outputs[1] = "logits"

        with pytest.raises(AssertionError, match="output 1 comes from"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_single_output_graph_is_rejected(self):
        graph = _surgical_graph()
        graph.outputs.pop()

        with pytest.raises(AssertionError, match="the decode exit has 2: logits / token"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_capacity_symbol_that_collides_with_an_input_symbol_fails_loudly(self):
        """A symbol mix-up (putting the capacity on a symbol that comes from the inputs) is rejected
        by the surgery."""
        graph = _pre_surgery_graph()
        colliding = StatesPlan(
            capacity_symbol=decode.SEQ_SYMBOL,
            attentions=decode.states_plan(graph, LAYERS).attentions,
        )

        with pytest.raises(StatesFormError, match=r"is already in graph\.symbols"):
            to_states_form(graph, colliding)


class TestGreedyMargins:
    def test_a_comfortable_margin_passes(self):
        shared.assert_greedy_margins({"a": [1.0, 0.5, 0.011], "b": [2.0]}, decode.MARGIN_FLOOR)

    def test_the_floor_stays_above_the_gate_precondition(self):
        """The producer's floor > the consumer's precondition (2 × atol 1e-3 — the same check as the
        gemma4 side).

        Below it, a case that "the script takes but fails the acceptance check's margin
        precondition" could be produced, and the check's "this fails only when the script and the
        asset disagree" would not hold. The source of truth of atol is PREFILL_ATOL (= 1e-3) in
        `e2e_minicpm5_greedy_test.ts` — change both together.
        """
        assert decode.MARGIN_FLOOR > 2 * 1e-3

    def test_a_thin_margin_fails_loudly(self):
        """MUST: mixing a step without margin into the golden lets GPU deviation split the
        sequence."""
        with pytest.raises(AssertionError, match=r"margin at or below the floor"):
            shared.assert_greedy_margins({"a": [1.0, 1e-3, 0.5]}, decode.MARGIN_FLOOR)

    def test_the_floor_itself_is_not_accepted(self):
        """The boundary is "exceeds" — exactly the floor does not pass."""
        with pytest.raises(AssertionError, match=r"margin at or below the floor"):
            shared.assert_greedy_margins({"a": [decode.MARGIN_FLOOR]}, decode.MARGIN_FLOOR)

    def test_every_offending_case_is_reported_at_once(self):
        """MUST: do not stop at the first one — otherwise one real run would not gather what is
        needed to decide which cases to remove."""
        with pytest.raises(AssertionError) as failure:
            shared.assert_greedy_margins(
                {"a": [1.0], "b": [1e-3], "c": [0.5, 0.0]}, decode.MARGIN_FLOOR
            )

        message = str(failure.value)
        assert "'b'" in message and "'c'" in message
        # Cases that passed are not in the diagnostic (only the names of those to remove are listed)
        assert "'a'" not in message


class _StubWrapper(nn.Module):
    """A chunk-wrapper subject that records how it is called (only the logits at the last position
    are designed).

    The logits at the last position are a fixed row chosen by "the current sequence length", so
    the expected token sequence and the margin sequence can be computed by hand.
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
        """MUST: full re-forward (no KV path) — the expected values are not produced by the
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
        """The positions are `arange(len)` every step (the precondition for the table-lookup RoPE to
        look up real positions)."""
        wrapper = _StubWrapper({3: [1.0, 0.0], 4: [0.0, 2.0]})

        shared.greedy_continuation(wrapper, torch.tensor([[4, 5, 6]]), 2)

        assert [positions for _, positions in wrapper.calls] == [[0, 1, 2], [0, 1, 2, 3]]


class TestCaseRoom:
    def test_a_prompt_that_fits_with_its_continuation_passes(self):
        shared.assert_case_room([("case", torch.zeros(1, 10, dtype=torch.int64))], 6, 16)

    def test_a_continuation_past_the_table_fails_loudly(self):
        """MUST: positions outside the table cannot be looked up (`F.embedding` does not silently
        return out-of-bounds values)."""
        with pytest.raises(AssertionError, match="the RoPE table's 16 positions"):
            shared.assert_case_room([("case", torch.zeros(1, 11, dtype=torch.int64))], 6, 16)


class TestGreedyCases:
    def test_every_greedy_case_is_a_golden_case(self):
        """MUST: no case name only on the greedy side that is not in the golden (the prompt would
        lose its source)."""
        assert set(decode.GREEDY_CASES) <= {name for name, _ in one_shot.GOLDEN_CASES}

    def test_every_greedy_case_has_a_first_token_expectation(self):
        """The condition for the first-continuation-token comparison (the cross-mechanism check
        against the one-shot form's expectation table) to work on every adopted case."""
        assert set(decode.GREEDY_CASES) <= set(one_shot.GREEDY_EXPECTATIONS)

    def test_the_adopted_expectations_are_not_all_the_same_token(self):
        """The condition that keeps the constant-output detection line of `_sanity` from becoming
        vacuous after cases were removed."""
        adopted = {one_shot.GREEDY_EXPECTATIONS[name] for name in decode.GREEDY_CASES}

        assert len(adopted) > 1


def _export_tiny_decode(wrapper) -> tuple[IrGraph, dict]:
    ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
    seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
    return export_module(
        wrapper,
        (ids, decode.positions_for(ids)),
        dynamic_shapes=({1: seq}, {1: seq}),
        symbol_names=(decode.SEQ_SYMBOL,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


class TestPublish:
    """The staging → final swap (`_publish`).

    What it protects is that "**no canonical asset mixing old and new can be produced**" (closing
    a high finding of an independent review). Writing into final file by file would let a run that
    failed halfway through the checks leave "new model + old greedy", and the acceptance check
    could go green on an asset that was already rejected.
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
        """A file only the old series has (e.g. the greedy of an excluded case) does not remain in
        final."""
        final = self._series(tmp_path, "final", {"model": "old", "greedy.stale": "old"})
        staging = self._series(tmp_path, "staging", {"model": "new"})

        shared._publish(staging, final)

        assert (final / "model").read_text() == "new"
        assert not (final / "greedy.stale").exists()
        assert not list(tmp_path.glob("final" + SUPERSEDED_SUFFIX))

    def test_a_failed_promotion_restores_the_old_series(self, tmp_path, monkeypatch):
        """Even when the promoting rename fails, final stays **the complete old asset** (neither
        missing nor mixed).

        NOTE: `os.replace` fails only for the one staging promotion (moving aside and restoring go
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


class TestExportedDecodeForm:
    """Actually run a tiny real model through export → surgery → container verification (needs
    transformers)."""

    @pytest.fixture
    def tiny_container(self, tmp_path):
        wrapper = decode.load_wrapper(_tiny_checkpoint(tmp_path / "ckpt"), positions=TINY_POSITIONS)
        graph, tensors = _export_tiny_decode(wrapper)
        surgical = to_states_form(graph, decode.states_plan(graph, LAYERS))
        verified = decode._write_container(
            surgical,
            tensors,
            tmp_path / one_shot.MODEL_FILE,
            graph_name="tiny",
            provenance=Provenance(license="fixture"),
        )
        return graph, verified, wrapper

    def test_the_container_is_a_verified_states_form_graph(self, tiny_container):
        pre, verified, wrapper = tiny_container

        form = decode.assert_ir_form_decode(verified, wrapper.model.config)

        assert form["attention_nodes"] == LAYERS
        assert form["state_append_nodes"] == 2 * LAYERS
        assert [spec.name for spec in verified.inputs] == [decode.INPUT_IDS, decode.POSITION_IDS]
        assert sorted(verified.symbols) == [decode.CAPACITY_SYMBOL, decode.SEQ_SYMBOL]
        # The surgery always prunes something (the mask's Tmax² constant) — none means the folding
        # is not working
        assert len(pre.initializers) > len(verified.initializers)

    def test_the_mask_residue_is_gone_but_the_pre_surgery_graph_had_it(self, tiny_container):
        """The check against a vacuous pass: confirm on the same asset that what gets pruned
        **really existed** on the export side."""
        pre, verified, _ = tiny_container

        assert one_shot.SYM_PREFIX_SLICE_OP in pre.required_ops
        assert one_shot.SYM_PREFIX_SLICE_OP not in verified.required_ops

    def test_the_rope_is_a_table_lookup_not_folded_trig(self, tiny_container):
        """Now that positions are an input, RoPE is looked up with `embedding` rather than
        folded."""
        _, verified, _ = tiny_container

        readers = [
            node
            for node in verified.nodes
            if node.op == "embedding" and decode.POSITION_IDS in node.ins
        ]

        assert len(readers) == 2
        assert "sin" not in verified.required_ops

    def test_the_wrapper_returns_logits_and_the_greedy_token(self, tmp_path):
        """MUST: the output order is `[logits, token]` (the runtime reads by slot number)."""
        wrapper = decode.load_wrapper(_tiny_checkpoint(tmp_path / "ckpt"), positions=TINY_POSITIONS)
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)

        with torch.no_grad():
            logits, token = wrapper(ids, decode.positions_for(ids))

        assert logits.shape == (1, 5, VOCAB)
        assert token.shape == (1, 5, 1)
        assert torch.equal(token, logits.argmax(-1, keepdim=True))
