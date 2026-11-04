"""Script that writes the real-weight MiniCPM5-1B as a **states-form chunk graph** (IR v2 container
+ golden).

The **scaffold for accepting on real weights**
ADR [0066](../../../docs/decisions/0066-generation-context-state-slots.md) (GenerationContext and
named state slots) /
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) decisions 4–5b
(attention with state references and `state_append`) /
[0068](../../../docs/decisions/0068-decode-exit-multi-output.md) decision 4 (the decode exit). The
decode version of the one-shot form ({@link minicpm5.export}). The source of truth of the
model-independent checks (how greedy is taken, the margin check, the capacity check, swapping in
the published series) is {@link _shared.decode_series}, and the one-shot helpers are
{@link minicpm5.export}; both are imported and reused (the same rule is not written in two
places).

    uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode

## What goes into the graph

One **chunk wrapper** `(input_ids[1,M], position_ids[1,M]) → (logits[1,M,V], token[1,M,1])`.
`M` is the physical chunk dimension (ADR 0067 decision 4 — `chunkLength` for prefill / 1 for
decode), and the valid rows are the compact prefix of the first `queryLength` rows. `token` is
`logits.argmax(-1, keepdim=True)` (the decode exit of ADR 0068 decision 4 — torch's i64 is lowered
to i32 by the boundary normalization). **The output order is fixed as `[logits, token]`**, and the
runtime's slot numbers read this order.

## Positions are inputs, RoPE is a lookup (the only structural difference from one-shot)

The one-shot form has no `position_ids` — positions are always `0..T-1`, so RoPE's cos / sin fold
from `arange` constants into a Tmax table + `sym_prefix_slice` (ADR 0010). In decode the position
is `pastLength + row` and **is known only at run time**, so this folding does not hold.

So `model.model.rotary_emb` is replaced by an instance of {@link RopeTable} (a table-lookup
module) (not one line of the upstream modeling code is touched). The table holds as is the cos /
sin obtained by **calling the original `rotary_emb` once with `arange(POS_MAX)[None]`**, and the
forward looks them up with `F.embedding(position_ids, table)`. {@link swap_rope_table} checks value
identity at construction by comparing with `torch.equal` on three kinds of position sequences
(that the lookup is right can only be confirmed through the values).

MUST: `rope.assert_rope_lifted` is **not applied** — it is the check that demotes the `inv_freq`
buffer to a leaf of the folding, and this script replaces the whole rotary module, so `inv_freq`
disappears from the model. Instead, {@link assert_ir_form_decode} directly checks **the absence of
`sin`-family ops** (`cos` is not in the IR vocabulary, so a graph with RoPE left over fails the
export itself — the only residue that can arrive is on the `sin` side).

## The mask is there "only to get the trace through"

The fourth input of `attention` (the additive causal mask) uses the same
{@link minicpm5.export .additive_causal_mask} as the one-shot form, but in the states form causal
is a **predicate computation** (`col ≤ pastLength + row` — ADR 0067 decision 4), so the mask tensor
is not needed. The surgery (`karume.states.to_states_form`) drops the mask input and prunes the
Tmax² constant and `sym_prefix_slice`. {@link assert_ir_form_decode} checks that none of that
residue remains (if it did, "1MiB nobody reads" would sit in the distribution).

## Output layout

    outputs/series/minicpm5-1b-decode/model-NNNNN-of-NNNNN.krm  weights, constants + 2 documents
    outputs/series/minicpm5-1b-decode/io.<case>.safetensors     unpadded full-length I/O
    outputs/series/minicpm5-1b-decode/greedy.<case>.safetensors expected K-step greedy sequence

`io.*` uses the same key convention as the one-shot form (`input.<graph input name>` /
`output.<position>`). `greedy.*` is **the source of truth of the decode acceptance** and holds
`prompt` i32 `[T]` / `expected` i32 `[K]` / `margin` f32 `[K]`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim
from torch.nn import functional

from _shared.decode_series import _write_greedy, assert_case_room, positions_for
from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.ops import ARGMAX_OP, ATTENTION_OP, STATE_APPEND_OP
from karume.pipeline import export_module, publish_model
from karume.shapes import declared_shape
from karume.states import StateAttentionSpec, StatesPlan, to_states_form
from minicpm5 import export as one_shot

#: Default location of the artifacts (a directory separate from the one-shot series — a different
#: asset with a different graph shape).
DEFAULT_OUT_DIR = SERIES_ROOT / "minicpm5-1b-decode"

#: Where the official weights live (the same material as the one-shot form).
DEFAULT_MODEL_DIR = one_shot.DEFAULT_MODEL_DIR

#: Names of the graph inputs (the chunk wrapper's forward argument names). The spellings
#: {@link assert_ir_form_decode} uses to check "these two and nothing else".
INPUT_IDS = one_shot.INPUT_IDS
POSITION_IDS = "position_ids"

#: Symbol name of the physical chunk dimension (the name in the IR is decided by `symbol_names` —
#: spelled the same as the Dim name).
SEQ_SYMBOL = "M"

#: Capacity symbol of the state slots (ADR 0066 decision 2 / addendum 7 — bound at
#: `createGenerationContext`). **A states-only symbol that never appears in value shapes**, a
#: placeholder that keeps the capacity from being baked in at export time.
CAPACITY_SYMBOL = "C"

#: Number of positions in the RoPE table (= the upper bound of absolute positions that can be looked
#: up). Equal to `SYM_MAX` but **a different concept**, so it is declared under its own name —
#: SYM_MAX is the maximum number of rows in one chunk, this is the upper bound of absolute positions
#: across prompt + generation, and in decode `pastLength + M` cannot exceed it. The day will come
#: when only one of them moves.
ROPE_TABLE_POSITIONS = 512

#: Number of continuation steps of the greedy golden (exercises the exit of ADR 0068 decision 4 over
#: many steps).
GREEDY_STEPS = 16

#: Lower bound on each step's top1 − top2 logit gap. It guarantees that the deviation of GPU
#: execution (measured on the order of ~1e-4) cannot split the greedy sequence, and it is **not
#: vacuous** (a case with a continuation that is not strongly determined fails here).
#: A failing case is removed from {@link GREEDY_CASES} without lowering K — making a sequence with
#: no margin the golden would make a check that is "red even when the GPU side is correct".
#:
#: MUST: keep it **above** the precondition of the acceptance check (`e2e_minicpm5_greedy_test.ts`
#: — `minMargin > 2 × atol`, atol = `PREFILL_ATOL` 1e-3). Below it, a case that "the script takes
#: but the check's precondition fails" could be produced, and the check's "this fails only when the
#: script and the asset disagree" would not hold (the same rule as the gemma4 side — review G3-03 of
#: 2026-08-19).
MARGIN_FLOOR = 1e-2

#: The cases that get a greedy golden (a subset of {@link minicpm5.export.GOLDEN_CASES}).
#: The io golden is taken for all 4 cases (it is one step, so it needs no margin check).
#:
#: **`capital-ja` is excluded** (measured 2026-08-18 — at step 5 of K=16 the margin is
#: 0.00773 < 1e-2). The measured minimum margins of the 4 cases are capital-en 0.2111 (step 5) /
#: capital-ja 0.00773 (step 5) / context-en 0.03412 (step 3) / context-ja 0.02610 (step 15). K is
#: not lowered — a shorter sequence would not remove the property "the continuation is not
#: strongly determined" itself, and would only lower the detection power of the other cases.
GREEDY_CASES: tuple[str, ...] = ("capital-en", "context-en", "context-ja")

#: Residue ops that the surgery should remove. `sym_prefix_slice` is the mask's Tmax folding
#: (ADR 0010), and `sin` is what survives when RoPE fell neither into folding nor into the lookup.
#: NOTE: `cos` is not in the IR vocabulary (UNARY_OPS in ops.py has only `sin`), so a form with
#: cos left over fails the export and never gets here — only the 2 that can actually arrive are
#: checked.
RESIDUE_OPS = (one_shot.SYM_PREFIX_SLICE_OP, "sin")


def slot_name(layer: int, part: str) -> str:
    """State slot name with the layer number (`l0.k` / `l0.v` …).

    The spelling is kept in one place — if the surgery spec ({@link states_plan}) and the shape
    check ({@link assert_ir_form_decode}) built it separately, a spelling mismatch would pass in
    the form "the slot exists but nobody reads it".
    """
    return f"l{layer}.{part}"


class RopeTable(nn.Module):
    """A **position-table lookup** module with the same calling convention as
    `LlamaRotaryEmbedding`.

    Upstream calls `self.rotary_emb(hidden_states, position_ids=position_ids)` and receives the
    2-tuple `(cos, sin)` (each `[B, seq, head_dim]`) (modeling_llama.py:408). This module meets
    only that convention and replaces the trigonometric computation from `inv_freq` with **a
    gather from the table**.

    MUST: the tables are held as plain attributes (not buffers) — the same form as the demotion
    `rope.lift_rope_buffers` does for existing models, and torch.export picks them up as lifted
    tensor constants.
    MUST: no dtype cast in the return value. Upstream puts `cos.to(dtype=x.dtype)` last, but the
    table is built in the model's dtype ({@link swap_rope_table} compares them), so it is the
    identity, and writing an identity cast could leak `_to_copy` into the export.
    """

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        super().__init__()
        self.cos_table = cos
        self.sin_table = sin

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look the tables up with `position_ids[B,seq]` (`x` is an argument of the convention —
        its value is not read)."""
        del x
        return (
            functional.embedding(position_ids, self.cos_table),
            functional.embedding(position_ids, self.sin_table),
        )


def build_rope_table(rotary: nn.Module, probe: torch.Tensor, positions: int) -> RopeTable:
    """A {@link RopeTable} made from the cos / sin of calling the original rotary once with
    `arange(positions)`.

    MUST: the table is built from **the original implementation's output itself** (not rebuilt from
    `inv_freq`) — rebuilding would open a path that fails to copy `attention_scaling` or the
    per-rope_type initialization, and the value-identity comparison would come close to the vacuous
    "the same formula written twice and compared".
    """
    with torch.no_grad():
        cos, sin = rotary(probe, position_ids=torch.arange(positions).unsqueeze(0))
    return RopeTable(cos[0].detach().clone(), sin[0].detach().clone())


def assert_rope_table_matches(
    rotary: nn.Module, table: RopeTable, probe: torch.Tensor, positions: int
) -> None:
    """Check that the table lookup **matches the original implementation exactly** (tuple
    structure, shape, dtype and every element).

    MUST: not vacuous. The table is built from a single `arange`, so that the "index → row" lookup
    is right can only be confirmed through value identity. The three probes are ① contiguous
    (`0..4`) ② with an offset (`3..9` — decode's `pastLength + row`) ③ non-monotonic (confirming
    that the lookup goes **by value**, regardless of order).
    """
    probes = (
        torch.arange(5).unsqueeze(0),
        torch.arange(3, 10).unsqueeze(0),
        torch.tensor([[positions - 1, 0, 17, 2]], dtype=torch.int64),
    )
    for position_ids in probes:
        with torch.no_grad():
            reference = rotary(probe, position_ids=position_ids)
            actual = table(probe, position_ids=position_ids)
        where = f"position_ids={position_ids.tolist()}"
        if len(actual) != len(reference):
            raise AssertionError(
                f"{where}: {len(actual)} return values (the original returns {len(reference)})"
            )
        for index, (got, want) in enumerate(zip(actual, reference, strict=True)):
            if got.dtype is not want.dtype or got.shape != want.shape:
                raise AssertionError(
                    f"{where}: return value {index} is {got.dtype} {tuple(got.shape)} —"
                    f" the original gives {want.dtype} {tuple(want.shape)}"
                )
            if not torch.equal(got, want):
                raise AssertionError(
                    f"{where}: the values of return value {index} differ from the original"
                    f" (max difference {float((got - want).abs().max())}) — the table lookup is"
                    " misaligned"
                )


def swap_rope_table(model: nn.Module, positions: int) -> RopeTable:
    """Replace `model.model.rotary_emb` with the table lookup (after confirming value identity).

    The comparison runs **before** the replacement so that a failure does not leave the model in
    a half-changed state.
    """
    rotary = model.model.rotary_emb
    probe = torch.zeros(1, 1, int(model.config.hidden_size), dtype=torch.float32)
    table = build_rope_table(rotary, probe, positions)
    assert_rope_table_matches(rotary, table, probe, positions)
    model.model.rotary_emb = table
    return table


class DecodeChunkWrapper(nn.Module):
    """Export wrapper that fixes `LlamaForCausalLM` to the chunk form
    (`(input_ids, position_ids) → (logits, token)`).

    MUST: `attention_mask` is our own 4D additive mask (for the same reason as the one-shot form —
    omitting it sends transformers down the path that leaves things to `is_causal`, which fails at
    the mask check of `gqa_sdpa_attention`). It exists only to get the trace through, and the
    surgery drops it.
    MUST: `use_cache=False`. With True a `DynamicCache` appears and the form returns KV as outputs,
    in direct conflict with "state is not part of the graph's ordinary I/O" (ADR 0066 decision 1).
    MUST: the output order is `[logits, token]`. The runtime reads the two by slot number.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = one_shot.additive_causal_mask(input_ids.shape[1])
        logits = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits
        return logits, logits.argmax(-1, keepdim=True)


def load_wrapper(model_dir: Path, *, positions: int = ROPE_TABLE_POSITIONS) -> DecodeChunkWrapper:
    """Load the real weights in f32 and return an exportable chunk wrapper with RoPE replaced by
    the table lookup."""
    from transformers import LlamaForCausalLM

    one_shot.register_attention()
    model = LlamaForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation=one_shot.ATTENTION_NAME
    )
    model.eval()
    swap_rope_table(model, positions)
    return DecodeChunkWrapper(model).eval()


def states_plan(
    graph: IrGraph, layers: int, *, capacity_symbol: str = CAPACITY_SYMBOL
) -> StatesPlan:
    """The surgery spec that assigns layer-numbered k / v slots to the attention nodes one by one in
    node order.

    MiniCPM5-1B has no KV-sharing layers (all 24 layers compute their own k / v), so there are 2
    slots per layer = 48. No window is given (the full form over the whole context — sliding is a
    Gemma 4 matter).

    MUST: fail when the count disagrees with the config's layer count. A missed attention is "a
    layer left in mask form without surgery", and only that layer becomes chunk-local causal — a
    silent wrong value (the numbers move, but it does not look at the past).
    """
    outputs = [node.outs[0] for node in graph.nodes if node.op == ATTENTION_OP]
    if len(outputs) != layers:
        raise AssertionError(f"{len(outputs)} attention nodes (does not match the {layers} layers)")
    return StatesPlan(
        capacity_symbol=capacity_symbol,
        attentions=tuple(
            StateAttentionSpec(
                output=output,
                k_slot=slot_name(layer, "k"),
                v_slot=slot_name(layer, "v"),
            )
            for layer, output in enumerate(outputs)
        ),
    )


def assert_ir_form_decode(
    graph: IrGraph,
    config: Any,
    *,
    seq_symbol: str = SEQ_SYMBOL,
    capacity_symbol: str = CAPACITY_SYMBOL,
) -> dict[str, Any]:
    """Check the form of the states-form decode graph (every property that **breaks quietly while
    the numbers still agree**).

    Only the properties that comparing against the golden does not catch are listed:

    - if even one attention remains in the conventional form (4 ins with the mask), that layer
      alone does not look at the past
    - if `window` attrs are attached, a full layer runs as sliding (cutting off what is outside the
      window)
    - if a slot's capacity is a number instead of a symbol, the capacity cannot be chosen when the
      context is created (ADR 0066 decision 3)
    - if `sym_prefix_slice` / `sin` remain, a Tmax constant nobody reads or unfolded leftovers sit
      in the distribution
    - if a mask remains a graph input, it becomes something else: "the host builds T² and passes it
      every chunk"

    MUST: read `config.head_dim` (do not derive it from `hidden_size // num_attention_heads` —
    MiniCPM5-1B disagrees, 96 ≠ 128).
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    depth = int(config.head_dim)
    layers = int(config.num_hidden_layers)

    names = [spec.name for spec in graph.inputs]
    if names != [INPUT_IDS, POSITION_IDS]:
        raise AssertionError(
            f"graph inputs are {names} — not the 2 inputs `{INPUT_IDS}` / `{POSITION_IDS}`"
            " (the mask may have stayed an input instead of being folded)"
        )
    if len(graph.outputs) != 2:
        raise AssertionError(
            f"the IR has {len(graph.outputs)} outputs (the decode exit has 2: logits / token)"
        )
    producer = {out: node for node in graph.nodes for out in node.outs}
    token_source = producer.get(graph.outputs[1])
    if token_source is None or token_source.op != ARGMAX_OP:
        found = "not a node output" if token_source is None else token_source.op
        raise AssertionError(
            f"output 1 comes from {found} — not `{ARGMAX_OP}` (the decode exit of ADR 0068"
            " decision 4)"
        )

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(
            f"{len(attentions)} attention nodes (does not match the {layers} layers)"
        )
    for layer, node in enumerate(attentions):
        where = f"attention[{layer}]"
        if len(node.ins) != 3:
            raise AssertionError(
                f"{where}: {len(node.ins)} ins — the states form has exactly 3: q / k / v"
                " (4 means the conventional form with the mask remains)"
            )
        expected_states = {"k": slot_name(layer, "k"), "v": slot_name(layer, "v")}
        if node.states != expected_states:
            raise AssertionError(
                f"{where}: the states field is {node.states} (expected {expected_states})"
            )
        if "window" in node.attrs:
            raise AssertionError(
                f"{where}: attrs has window={node.attrs['window']} — it should be the full form"
                " over the whole context"
            )
        query, key, value = (declared_shape(graph, name) for name in node.ins)
        if query[1] != heads or key[1] != kv_heads or value[1] != kv_heads:
            raise AssertionError(
                f"{where}: the head axes are {[query[1], key[1], value[1]]} —"
                f" not the true GQA form {[heads, kv_heads, kv_heads]}"
                " (if k / v are widened to H, repeat_kv has materialized)"
            )
        if query[3] != depth or key[3] != depth or value[3] != depth:
            raise AssertionError(
                f"{where}: the D axes are {[query[3], key[3], value[3]]} —"
                f" not config.head_dim {depth}"
            )

    appends = [node for node in graph.nodes if node.op == STATE_APPEND_OP]
    written = sorted(node.states["slot"] for node in appends)
    expected_slots = sorted(
        slot_name(layer, part) for layer in range(layers) for part in ("k", "v")
    )
    if written != expected_slots:
        raise AssertionError(
            f"`{STATE_APPEND_OP}` writes to {len(written)} slots — not one per slot"
            f" (missing {sorted(set(expected_slots) - set(written))} /"
            f" extra {sorted(set(written) - set(expected_slots))})"
        )
    if sorted(graph.states) != expected_slots:
        raise AssertionError(
            f"the states declaration is {sorted(graph.states)} — not the"
            f" {len(expected_slots)} per-layer k / v slots"
        )
    slot_shape = [1, kv_heads, capacity_symbol, depth]
    for name in expected_slots:
        slot = graph.states[name]
        if slot.dtype != "f32" or list(slot.shape) != slot_shape:
            raise AssertionError(
                f"states['{name}'] is {slot.dtype} {list(slot.shape)} —"
                f" not f32 {slot_shape} (the capacity MUST stay a symbol)"
            )

    symbols = sorted(graph.symbols)
    if symbols != sorted({seq_symbol, capacity_symbol}):
        raise AssertionError(f"symbols are {symbols} — not {sorted({seq_symbol, capacity_symbol})}")
    residue = sorted(set(graph.required_ops) & set(RESIDUE_OPS))
    if residue:
        raise AssertionError(
            f"ops that the surgery should have removed remain: {residue}"
            " (residue of the mask's Tmax folding / RoPE left unfolded)"
        )
    return {
        "attention_nodes": len(attentions),
        "state_append_nodes": len(appends),
        "slots": len(expected_slots),
        "slot_shape": slot_shape,
        "heads": [heads, kv_heads, kv_heads],
        "head_dim": depth,
    }


def _write_container(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    graph_name: str,
    provenance: Provenance,
) -> IrGraph:
    """Write and verify the graph after surgery (the 3 publication steps are left to
    `pipeline.publish_model`).

    `export_to_file` runs export → write as one path with no gap for the surgery, so only the part
    from writing on is called through core's entry point. **Neither the rules nor the atomicity are
    reimplemented** — the checks of the states section, order and shapes, as well as the part split
    (ADR 0108) with its swap and cleanup, belong to core.

    Initializers killed by the pruning (the mask's Tmax² constant) are also dropped from the stored
    tensors — `stored_model` requires declarations and stored tensors to **match exactly**.
    """
    declared = {init.tensor for init in graph.initializers.values()}
    stored = {name: tensor for name, tensor in tensors.items() if name in declared}
    return publish_model(path, graph, stored, provenance=provenance, graph_name=graph_name)


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> list[str]:
    """Write each case's **unpadded full-length** inputs and outputs to `io.<case>.safetensors`.

    NOTE: this is a **reference table** of "the values torch produces with the same weights at the
    same positions", not the runtime's execution script. The decode series' graph runs in chunk
    form (`M = chunkLength`), so the Deno side splits the full length T into chunks and compares
    **only the valid rows** against the corresponding rows of `output.0` / `output.1` here. What is
    pinned to 0 on pad rows is **only the output of states-form attention** (ADR 0066 addendum 8);
    the later MLP / lm_head write meaningless values on pad rows too — the graph outputs of pad
    rows **MUST NOT be read** (the reference has no counterpart for them either).
    """
    written: list[str] = []
    for name, ids in cases:
        args = {INPUT_IDS: ids, POSITION_IDS: positions_for(ids)}
        with torch.no_grad():
            outputs = wrapper(args[INPUT_IDS], args[POSITION_IDS])
        tensors = {
            f"{one_shot.INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name}: input '{declared.name}'"
            )
            for declared in graph.inputs
        }
        for slot, output in enumerate(outputs):
            tensors[f"{one_shot.OUTPUT_PREFIX}{slot}"] = normalize_boundary_tensor(
                output.detach().contiguous(), f"{name}: output {slot}"
            )
        path = out_dir / f"{one_shot.IO_PREFIX}{name}{one_shot.IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    positions: int = ROPE_TABLE_POSITIONS,
    steps: int = GREEDY_STEPS,
) -> dict[str, Any]:
    """Write the states-form IR container, the io golden and the greedy golden, and return a
    summary.

    MUST: the artifacts are written to a staging directory and put in place only after passing
    **every check** (the shape check, the margin check, and the sanity check against the one-shot
    form's expectation table). Placing them in final before the checks would let a failed run leave
    "an asset that can pass the acceptance check" (the rules of the swap and of cleanup belong to
    the core primitive {@link karume.artifacts.staged_publication}).
    """
    # The provenance is read before the weights (the same function as the one-shot script —
    # `minicpm5.export.upstream_provenance`).
    provenance = one_shot.upstream_provenance(model_dir)
    wrapper = load_wrapper(model_dir, positions=positions)
    cases = one_shot.build_cases(model_dir, sym_max)
    greedy_cases = tuple(case for case in cases if case[0] in GREEDY_CASES)
    # The io only runs each prompt once as a whole (no positions for a continuation), so it is
    # checked with steps=0.
    assert_case_room(cases, 0, positions)
    assert_case_room(greedy_cases, steps, positions)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with staged_publication(out_dir) as staging:
        # The writer creates the directory itself (the primitive does not create it — it only
        # passes a path).
        staging.mkdir()
        # The example input is the longest case (far from the 0/1 specialization of symbolic
        # dimensions). min=2 for the same reason, max is the evaluation point of the mask's Tmax
        # folding (the surgery prunes it so it does not remain in the distribution, but the trace
        # goes through).
        _, example_ids = max(cases, key=lambda case: case[1].shape[1])
        seq = Dim(SEQ_SYMBOL, min=2, max=sym_max)
        print("[export] torch.export → conversion", file=sys.stderr, flush=True)
        graph, tensors = export_module(
            wrapper,
            (example_ids, positions_for(example_ids)),
            dynamic_shapes=({1: seq}, {1: seq}),
            symbol_names=(SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        config = wrapper.model.config
        print("[export] surgery to the states form → write", file=sys.stderr, flush=True)
        surgical = to_states_form(graph, states_plan(graph, int(config.num_hidden_layers)))
        # The graph name is the **part name** (the same spelling as the one-shot script —
        # `minicpm5.export.GRAPH_NAME`). Not derived from the directory name (container-v1 §2.1).
        verified = _write_container(
            surgical,
            tensors,
            staging / one_shot.MODEL_FILE,
            graph_name=one_shot.GRAPH_NAME,
            provenance=provenance,
        )
        form = assert_ir_form_decode(verified, config)

        print("[io] full-length forward", file=sys.stderr, flush=True)
        io_written = _write_io(wrapper, verified, cases, staging)
        greedy_written, tokens, margins = _write_greedy(
            wrapper, greedy_cases, staging, steps=steps, floor=MARGIN_FLOOR
        )

        # The first continuation token is compared with the one-shot form's expectation table
        # (`one_shot.GREEDY_EXPECTATIONS`) (a cross-mechanism comparison — the one-shot and decode
        # scripts are separate, so the same top-1 with the same weights and the same prompt
        # cross-validates the two).
        # MUST: evaluated before publication (when it fails, the staging directory goes with it —
        # no mixed asset is left).
        tokenizer = one_shot.load_tokenizer(model_dir)
        first = {name: continuation[0] for name, continuation in tokens.items()}
        expected = {
            name: token
            for name, token in one_shot.expected_token_ids(tokenizer).items()
            if name in first
        }
        labels = {
            token: tokenizer.id_to_token(token)
            for token in set(first.values()) | set(expected.values())
        }
        sanity = one_shot._sanity(first, expected, labels)
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "pruned_initializers": len(graph.initializers) - len(verified.initializers),
        "model_bytes": sum(
            p.stat().st_size for p in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "io": io_written,
        "greedy": greedy_written,
        "greedy_steps": steps,
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "form": form,
        "margin_min": {name: min(values) for name, values in sorted(margins.items())},
        "continuation": {
            name: tokenizer.decode(continuation) for name, continuation in sorted(tokens.items())
        },
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sym-max", type=int, default=one_shot.SYM_MAX)
    parser.add_argument("--positions", type=int, default=ROPE_TABLE_POSITIONS)
    parser.add_argument("--steps", type=int, default=GREEDY_STEPS)
    args = parser.parse_args(argv)
    summary = export_series(
        args.model_dir,
        args.out,
        sym_max=args.sym_max,
        positions=args.positions,
        steps=args.steps,
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
