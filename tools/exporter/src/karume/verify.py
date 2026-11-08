"""Checking every IR acceptance rule on the Python side too (docs/ir-v2.md), and the merge with
the container.

The TS sources of truth are packages/runtime/src/format/ir.ts (the rules of a graph alone —
`parseIrDeclarationValue`), packages/runtime/src/format/container/*.ts (the structure of the 2
documents, the codec ledger, the merge) and packages/runtime/src/ops/support.ts (the cross-check
against the runtime support table — `assertRuntimeSupport`). So that the exporter never emits a
file that "can be written but the runtime cannot read", the same rules are applied at the end of
the write path.

MUST: this is a fail-loudly check, not a place for approximation — neither unknown keys nor
non-canonical notation are silently ignored; they always raise (unreleased, so there is no
forward-compatibility channel).

    uv run karume verify ../../models/karume-irodori-v4.1-small/v4.1-small/dit/model.i8.krm

`parse_ir_graph` reads, while validating, the JSON of the exporter's internal structure (IR v1 —
`karume.ir`). It is used on 3 paths: checking the document rebuilt from the container's 2
documents ({@link ir_graph_from_container}), the migration CLI's path that reads the
`__metadata__` of old shards (container-v1 §12), and the public surface (`karume.__all__`).
Verifying a distribution (CLI) accepts only containers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from karume.container import (
    BLOCK_TAIL_ALIGN,
    BlockEncoding,
    ContainerFormatError,
    GraphDescriptor,
    ModelDescriptor,
    ReadContainer,
    assert_row_axis,
    codec_entry,
    container_parts,
    group_count,
    joined_scale_id,
    payload_bytes,
    per_channel_group_size,
    read_container,
    scale_bytes,
    scale_step_bytes,
)
from karume.dims import MAX_SAFE_INT, DimError, is_symbol_name, parse_dim, try_parse_dim
from karume.ir import (
    IR_FORMAT,
    IR_VERSION,
    MIN_GROUP_SIZE,
    IrDim,
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrState,
    IrStorage,
    IrValue,
)
from karume.ops import (
    ATTENTION_OP,
    IO_DTYPES,
    M0_STORAGE_DTYPES,
    OP_CONTRACTS,
    STATE_APPEND_OP,
    STRIDED_RANK_OPS,
    OpContractError,
    assert_node_contract,
    assert_strided_rank,
    attention_readonly,
    integral_float_to_int,
    resolve_node_dtypes,
    state_window,
    sym_prefix_slice_attrs,
)
from karume.shapes import assert_graph_shapes, declared_shape


class IrError(ValueError):
    """A violation of a rule decided by the graph JSON alone."""


class ContainerError(ValueError):
    """A violation of the safetensors layout rules, a missing capability, an inconsistency in the
    container → IR document conversion, and the wrapper of IR acceptance violations
    ({@link assert_ir_accepted}).

    The safetensors layout rules apply to the migration CLI's input (old shards) and to assets.
    Violations of the merge rules (2 documents + binding table) are raised not as this type but as
    `karume.container.ContainerFormatError` (the type of the same name as on the TS side) — callers
    that branch on type catch both.
    """


TOP_LEVEL_KEYS = (
    "format",
    "version",
    "requires",
    "symbols",
    "inputs",
    "outputs",
    "initializers",
    "values",
    "nodes",
)

#: Optional top-level sections. A graph without `states` (= every existing model) is unaffected
#: (ADR 0066 decision 2, "the exporter is unaffected until the first model that emits `states`").
OPTIONAL_TOP_LEVEL_KEYS = ("states",)

SEMANTIC_DTYPES = ("f32", "i32", "bool")
#: `i4f16` is the layout of `int4-sym-g-f16` (ADR 0145 decision 3): in the vocabulary, so a
#: container that carries it reads back, and refused by the runtime support check until an op reads
#: its f16 steps (`ops.M0_STORAGE_DTYPES`).
STORAGE_DTYPES = ("f32", "f16", "bf16", "i8", "i4", "i4f16", "i2", "i32")

#: Storage dtypes that may carry a scale / group_size descriptor (quantized storage). On the TS
#: side, these 4 words are the layouts of the codecs with `scale: "required"` in the codec ledger
#: (CODEC_LEDGER in `packages/runtime/src/format/container/codecs.ts`).
QUANTIZED_STORAGE_DTYPES = ("i8", "i4", "i4f16", "i2")

#: Storage dtypes of group quantization (a `group_size` that is a power of 2 ≥ 16 is required —
#: ADR 0069 decision 2): the layouts of the codecs with `grouping: "group"` in the codec ledger.
GROUP_STORAGE_DTYPES = ("i4", "i4f16")

#: dtype vocabulary of state slots. Only f32 for now (ADR 0066 decision 2).
STATE_DTYPES = ("f32",)
#: State slot dtypes that are only reserved (ADR 0066 addendum of 2026-08-17, item 5). The f16
#: storage of a full slot is the Session's opt-in (`SessionOptions.fullStateStorage`), and the graph
#: keeps declaring f32 under it. A declared f16 is **unsupported**, not "outside the vocabulary".
RESERVED_STATE_DTYPES = ("f16",)
#: Rank limit of state slots (ADR 0066 decision 2, "fixed rank (rank ≤ 4)"). The same number as the
#: strided kernels' limit (ops.STRIDED_RANK), but for a different reason, so the constant is not
#: shared.
MAX_STATE_RANK = 4

#: Semantic dtype of an initializer → allowed storage dtypes (docs/ir-v2.md "Values and types").
#: MUST: no crossing — an initializer declared `i32` would be read as an f16 bit pattern, giving
#: silently wrong values. There is no bool initializer in the vocabulary.
INITIALIZER_STORAGE = {"f32": ("f32", "f16", "bf16", "i8", "i4", "i4f16", "i2"), "i32": ("i32",)}


# ---- JSON layer --------------------------------------------------------------


def parse_graph_json(text: str) -> Any:
    """Read a graph JSON. Anything containing non-finite values is not accepted.

    Python's json accepts `Infinity` / `NaN` literals by default, but the browser's JSON.parse
    fails. In addition, `1e999` stays syntactically valid while rounding to Infinity, so it is
    rejected at the **value level**, not only by literal name (docs/ir-v2.md).
    """

    def reject_constant(literal: str) -> float:
        raise IrError(f"Non-standard literal in graph JSON: {literal}")

    def parse_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise IrError(f"Number '{raw}' in graph JSON is not finite")
        return value

    try:
        return json.loads(text, parse_constant=reject_constant, parse_float=parse_float)
    except IrError:
        raise
    except ValueError as cause:
        raise IrError(f"Cannot parse graph JSON: {cause}") from cause


# ---- Structure helpers -----------------------------------------------------------


def _as_object(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise IrError(f"{where}: is not an object")
    return value


def _as_array(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise IrError(f"{where}: is not an array")
    return value


def _as_nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or value == "":
        raise IrError(f"{where}: is not a non-empty string")
    return value


def _as_unique_strings(value: Any, where: str) -> list[str]:
    """A string array that allows no duplicates (a field treated as a set)."""
    items = [
        _as_nonempty_str(item, f"{where}[{index}]")
        for index, item in enumerate(_as_array(value, where))
    ]
    seen: set[str] = set()
    for item in items:
        if item in seen:
            raise IrError(f"{where}: '{item}' is duplicated")
        seen.add(item)
    return items


def _check_keys(
    obj: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], where: str
) -> None:
    for key in required:
        if key not in obj:
            raise IrError(f"{where}: required key '{key}' is missing")
    for key in obj:
        if key not in required and key not in optional:
            raise IrError(f"{where}: unknown key '{key}'")


def _as_semantic_dtype(value: Any, where: str) -> str:
    dtype = _as_nonempty_str(value, where)
    if dtype not in SEMANTIC_DTYPES:
        raise IrError(
            f"{where}: semantic dtype '{dtype}' is outside the vocabulary"
            f" ({' / '.join(SEMANTIC_DTYPES)})"
        )
    return dtype


def _as_storage_dtype(value: Any, where: str) -> str:
    dtype = _as_nonempty_str(value, where)
    if dtype not in STORAGE_DTYPES:
        raise IrError(
            f"{where}: storage dtype '{dtype}' is outside the vocabulary"
            f" ({' / '.join(STORAGE_DTYPES)})"
        )
    return dtype


def _parse_shape(value: Any, symbols: set[str], where: str) -> list[int | str]:
    shape: list[int | str] = []
    for index, dim in enumerate(_as_array(value, where)):
        at = f"{where}[{index}]"
        # bool derives from int but is not a dimension.
        if isinstance(dim, bool):
            raise IrError(f"{at}: dimension is neither a number nor a string")
        # Exceeding the safe range is caught by the int-side check below (`> MAX_SAFE_INT`) = the
        # same accepted set as TS `Number.isSafeInteger`.
        dim = integral_float_to_int(dim)
        if isinstance(dim, int):
            if dim < 0 or dim > MAX_SAFE_INT:
                raise IrError(f"{at}: dimension {dim} is not a non-negative integer")
            shape.append(dim)
            continue
        if not isinstance(dim, str):
            raise IrError(f"{at}: dimension is neither a number nor a string")
        expr = try_parse_dim(dim)
        if expr is None:
            raise IrError(
                f"{at}: dimension expression '{dim}' does not fit the canonical grammar"
                " coeff·sym+offset"
            )
        if expr.sym not in symbols:
            raise IrError(f"{at}: symbol '{expr.sym}' is not declared in graph.symbols")
        shape.append(dim)
    return shape


def _as_state_dtype(value: Any, where: str) -> str:
    dtype = _as_nonempty_str(value, where)
    if dtype in RESERVED_STATE_DTYPES:
        raise IrError(
            f"{where}: state slot dtype '{dtype}' is not supported"
            " (reserved by ADR 0066 addendum item 5 — needs an opt-in of the numeric contract)"
        )
    if dtype not in STATE_DTYPES:
        raise IrError(
            f"{where}: state slot dtype '{dtype}' is outside the vocabulary"
            f" ({' / '.join(STATE_DTYPES)})"
        )
    return dtype


def _parse_state(value: Any, symbols: set[str], where: str) -> IrState:
    """Declaration of one state slot (ADR 0066 decision 2; mirrors parseStateSlot on the TS side).

    MUST: the shape is **a concrete shape including the capacity**, so numeric dimensions are
    positive integers (unlike the non-negative ones of `values` — a slot of capacity 0 has no body
    that can be bound). The rank is 1..MAX_STATE_RANK (rank 0, with no capacity axis, cannot
    "include the capacity"). A symbolic dimension is fine if declared in `symbols`, and **the
    states shape is a binding point too** (the capacity `createGenerationContext` decides — ADR
    0066 addendum item 7; _check_symbol_bindability).

    The optional key `external` (ADR 0096 stage 2) declares "do not allocate the body yourself =
    look up the borrowed context's slot by name". MUST: only `true` is accepted — `false` means the
    same as the field being absent, and two spellings for one meaning would leave "which rule to
    read it by" undetermined by the declaration.
    """
    obj = _as_object(value, where)
    _check_keys(obj, ["dtype", "shape"], ["external"], where)
    dtype = _as_state_dtype(obj["dtype"], f"{where}.dtype")
    shape = _parse_shape(obj["shape"], symbols, f"{where}.shape")
    external = False
    if "external" in obj:
        if obj["external"] is not True:
            raise IrError(
                f"{where}.external: only true ({obj['external']!r} — false means the same as the"
                " field being absent)"
            )
        external = True
    if not 1 <= len(shape) <= MAX_STATE_RANK:
        raise IrError(
            f"{where}.shape: rank {len(shape)} is outside 1..{MAX_STATE_RANK}"
            " (MUST be a concrete shape of fixed rank including the capacity)"
        )
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim < 1:
            raise IrError(
                f"{where}.shape[{index}]: dimension {dim} is not a positive integer (no capacity"
                " can be taken)"
            )
    return IrState(dtype=dtype, shape=shape, external=external)


def _parse_storage(value: Any, where: str, *, shared: bool = False) -> IrStorage:
    """The storage descriptor.

    `shared` (the shared initializer of ADR 0096 stage 2) has **only the one key `dtype`**: the
    borrower's container holds not a single byte, so both the field describing the companion bytes
    (`scale`) and the group step (`group_size`) are **held only by the lender**. Copying them would
    grow the same fact in two places, and which one to dequantize with would be undetermined by
    the declaration. The i8 / i4 "scale required" and "i4 requires group_size" do not apply on
    this side either (applying them would make a shared declaration impossible to write in
    principle).
    """
    obj = _as_object(value, where)
    _check_keys(obj, ["dtype"], ["scale", "group_size"], where)
    dtype = _as_storage_dtype(obj["dtype"], f"{where}.dtype")
    has_scale = "scale" in obj
    has_group_size = "group_size" in obj
    if shared and (has_scale or has_group_size):
        extra = sorted(key for key in ("scale", "group_size") if key in obj)
        raise IrError(
            f"{where}: a shared initializer cannot declare {extra}"
            " (it is the side without bytes, so the lender's resident weight is the source of"
            " truth for the companion bytes and the group step)"
        )
    # scale / group_size describe quantized storage. On a non-quantized dtype they are an
    # exporter mix-up, so they are not accepted (silently ignoring them would give the storage two
    # meanings).
    if dtype not in QUANTIZED_STORAGE_DTYPES and (has_scale or has_group_size):
        raise IrError(f"{where}: storage dtype '{dtype}' cannot have scale / group_size")
    # MUST: i8 / i4 **declare** the scale explicitly
    # (ADR 0019 / 0069; on the TS side this mirrors `scale: "required"` in the codec ledger and
    # parseEncoding in packages/runtime/src/format/container/descriptor.ts).
    # Filling in a default of 1.0 would turn a forgotten scale into "weights dequantized with 1.0
    # on every channel", and both loading and running would pass (the difference shows up at
    # O(scale), yet no exception is raised anywhere).
    if dtype in QUANTIZED_STORAGE_DTYPES and not has_scale and not shared:
        raise IrError(f"{where}: storage dtype '{dtype}' requires scale (the scale tensor's key)")
    # MUST: i4 **declares** group_size explicitly (ADR 0069 decision 2). 4-bit storage without a
    # decided group length has no decided position to look up the scale again, and the decode
    # silently gives different values.
    if dtype == "i2" and has_group_size:
        raise IrError(f"{where}: i2 has a per-row scale only (group_size cannot be given)")
    if dtype in GROUP_STORAGE_DTYPES and not has_group_size and not shared:
        raise IrError(f"{where}: storage dtype '{dtype}' requires group_size (ADR 0069 decision 2)")
    scale = _as_nonempty_str(obj["scale"], f"{where}.scale") if has_scale else None
    group_size = None
    if has_group_size:
        raw = integral_float_to_int(obj["group_size"])
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise IrError(f"{where}.group_size: is not a positive integer")
        # The TS side reads it as a JSON number, so a value beyond 2^53−1 cannot be held as an
        # integer (parseEncoding in packages/runtime/src/format/container/descriptor.ts —
        # `groupSize` accepts safe integers only). Accepting it here would make only the runtime
        # fail.
        if raw > MAX_SAFE_INT:
            raise IrError(f"{where}.group_size: {raw} exceeds the safe integer 2^53−1")
        # MUST: the i4 group length is a power of 2 and 16 or more (ADR 0069 decision 2 — the same
        # constraint as ORT). This constraint keeps row and group boundaries always byte-aligned,
        # so the u32 binding holds without tail zero padding.
        if dtype in GROUP_STORAGE_DTYPES and (raw & (raw - 1) != 0 or raw < MIN_GROUP_SIZE):
            raise IrError(
                f"{where}.group_size: {raw} is not a power of 2 that is {MIN_GROUP_SIZE} or more"
                " (ADR 0069 decision 2)"
            )
        group_size = raw
    return IrStorage(dtype=dtype, scale=scale, group_size=group_size)


def _parse_node_states(
    obj: Mapping[str, Any], slots: Mapping[str, IrState], where: str
) -> dict[str, str]:
    """A node's `states` field (ADR 0067 decision 4; mirrors parseNodeStates on the TS side).

    This checks **only the 3 points decided by the graph alone**: that it is a plain object, that
    keys and values are non-empty strings, and that each value is a slot name declared in
    `graph.states`.

    MUST: a reference to an undeclared slot fails loudly. Passing it would let a node that "reads
    a name without a body" through Session construction, neither allocated nor bound, failing only
    at the execution step (a rule of the same layer as rejecting forward references on the value
    side).

    NOTE: the key set itself (exactly `{k,v}` / exactly `{slot}`) belongs to the op contract
    (ops.assert_node_contract) — the parser does not know the op vocabulary.
    """
    states: dict[str, str] = {}
    if "states" not in obj:
        return states
    for key, value in _as_object(obj["states"], f"{where}.states").items():
        _as_nonempty_str(key, f"key of {where}.states")
        slot = _as_nonempty_str(value, f"{where}.states['{key}']")
        if slot not in slots:
            raise IrError(
                f"{where}.states['{key}']: state slot '{slot}' is not declared in graph.states"
            )
        states[key] = slot
    return states


# ---- Rules of a graph alone -----------------------------------------------------


def parse_ir_graph(text: str) -> IrGraph:
    """Read a graph JSON (the IR v1 structure) while validating it.

    On the TS side, the rules of a graph alone are held by parseIrDeclarationValue in
    packages/runtime/src/format/ir.ts, and the storage rules (scale required, groupSize and so on)
    by parseEncoding in packages/runtime/src/format/container/descriptor.ts and the binding
    (bindGraphs) in packages/runtime/src/format/container/bind.ts (this equals those combined).
    mergedGraph in bind.ts only copies the merged result into an IrGraph and checks nothing.
    """
    root = _as_object(parse_graph_json(text), "graph")
    _check_keys(root, TOP_LEVEL_KEYS, OPTIONAL_TOP_LEVEL_KEYS, "graph")

    if root["format"] != IR_FORMAT:
        raise IrError(f"graph.format is not '{IR_FORMAT}': {root['format']!r}")
    if root["version"] != IR_VERSION or isinstance(root["version"], bool):
        raise IrError(f"graph.version is not {IR_VERSION}: {root['version']!r}")

    requires = _as_object(root["requires"], "graph.requires")
    _check_keys(requires, ["ops"], [], "graph.requires")
    required_ops = _as_unique_strings(requires["ops"], "graph.requires.ops")

    symbols = _as_unique_strings(root["symbols"], "graph.symbols")
    for symbol in symbols:
        if not is_symbol_name(symbol):
            raise IrError(f"graph.symbols: symbol name '{symbol}' is invalid")
    symbol_set = set(symbols)

    inputs: list[IrInput] = []
    for index, raw in enumerate(_as_array(root["inputs"], "graph.inputs")):
        where = f"graph.inputs[{index}]"
        obj = _as_object(raw, where)
        _check_keys(obj, ["name", "dtype", "shape"], [], where)
        inputs.append(
            IrInput(
                name=_as_nonempty_str(obj["name"], f"{where}.name"),
                dtype=_as_semantic_dtype(obj["dtype"], f"{where}.dtype"),
                shape=_parse_shape(obj["shape"], symbol_set, f"{where}.shape"),
            )
        )

    outputs = _as_unique_strings(root["outputs"], "graph.outputs")

    values: dict[str, IrValue] = {}
    for name, raw in _as_object(root["values"], "graph.values").items():
        # An empty value name is rejected (mirrors packages/runtime/src/format/ir.ts on the TS
        # side) — every referencing side (input names, node inputs / outputs, graph outputs)
        # accepts only non-empty strings, so passing it would give "a value that can be declared
        # but in principle never referenced". When empty-named values and empty-named
        # initializers are written **as a pair**, they satisfy each other's orphan-declaration
        # check and it does not fire, so loosening this would let an unreferenceable real tensor
        # sit in the distribution.
        _as_nonempty_str(name, "value name of graph.values")
        where = f"graph.values['{name}']"
        obj = _as_object(raw, where)
        _check_keys(obj, ["dtype", "shape"], [], where)
        values[name] = IrValue(
            dtype=_as_semantic_dtype(obj["dtype"], f"{where}.dtype"),
            shape=_parse_shape(obj["shape"], symbol_set, f"{where}.shape"),
        )

    initializers: dict[str, IrInitializer] = {}
    for name, raw in _as_object(root["initializers"], "graph.initializers").items():
        # An empty initializer name is rejected for the same reason (the counterpart of values
        # above — blocking only one of them lets "the empty-named pair" through as is).
        _as_nonempty_str(name, "initializer name of graph.initializers")
        where = f"graph.initializers['{name}']"
        obj = _as_object(raw, where)
        # A shared initializer (ADR 0096 stage 2) has `shared` instead of `tensor`. They are
        # **exclusive**; with both written, "does it hold bytes or borrow them" is undetermined by
        # the declaration (switching the required key set of `_check_keys` is itself what
        # enforces the exclusion).
        if "shared" in obj:
            _check_keys(obj, ["shared", "storage"], [], where)
            shared_obj = _as_object(obj["shared"], f"{where}.shared")
            _check_keys(shared_obj, ["tensor"], [], f"{where}.shared")
            initializers[name] = IrInitializer(
                shared=IrShared(
                    tensor=_as_nonempty_str(shared_obj["tensor"], f"{where}.shared.tensor")
                ),
                storage=_parse_storage(obj["storage"], f"{where}.storage", shared=True),
            )
            continue
        _check_keys(obj, ["tensor", "storage"], [], where)
        initializers[name] = IrInitializer(
            tensor=_as_nonempty_str(obj["tensor"], f"{where}.tensor"),
            storage=_parse_storage(obj["storage"], f"{where}.storage"),
        )

    # Omission is treated as an empty slot set (a graph without the section is unaffected — ADR
    # 0066 decision 2).
    states: dict[str, IrState] = {}
    for name, raw in _as_object(root.get("states", {}), "graph.states").items():
        # An empty slot name is rejected — the referencing fields (ADR 0067) accept only non-empty
        # strings, so passing it would give "a slot that can be declared but in principle never
        # referenced" (for values the orphan-declaration check closes the same hole).
        _as_nonempty_str(name, "slot name of graph.states")
        states[name] = _parse_state(raw, symbol_set, f"graph.states['{name}']")

    nodes: list[IrNode] = []
    for index, raw in enumerate(_as_array(root["nodes"], "graph.nodes")):
        where = f"graph.nodes[{index}]"
        obj = _as_object(raw, where)
        _check_keys(obj, ["op", "ins", "outs", "attrs"], ["states"], where)
        # NOTE: the number of `outs` is not checked here (0 = an effect op that defines no value
        # joined the vocabulary — ADR 0067 decision 5). "Only ops whose contract declares an
        # effect may have 0" is enforced in the contract layer (the output count cross-check of
        # assert_node_contract); the parser gives the count no meaning.
        nodes.append(
            IrNode(
                op=_as_nonempty_str(obj["op"], f"{where}.op"),
                ins=[
                    _as_nonempty_str(item, f"{where}.ins[{i}]")
                    for i, item in enumerate(_as_array(obj["ins"], f"{where}.ins"))
                ],
                outs=[
                    _as_nonempty_str(out, f"{where}.outs[{i}]")
                    for i, out in enumerate(_as_array(obj["outs"], f"{where}.outs"))
                ],
                attrs=_as_object(obj["attrs"], f"{where}.attrs"),
                states=_parse_node_states(obj, states, where),
            )
        )

    _check_symbol_bindability(symbols, inputs, states, values)
    defined = _check_definitions(inputs, initializers, nodes, outputs)
    _check_declarations(inputs, initializers, values, nodes, defined)
    _check_state_slots(states, values, defined, nodes)
    _check_required_ops(required_ops, nodes)

    return IrGraph(
        symbols=symbols,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        values=values,
        states=states,
        nodes=nodes,
    )


def _symbols_in(shape: Sequence[IrDim]) -> set[str]:
    """Symbol names that appear in a shape (only occurrences in dimension positions — no
    inference back from element counts)."""
    return {parse_dim(dim).sym for dim in shape if isinstance(dim, str)}


def _check_symbol_bindability(
    symbols: Sequence[str],
    inputs: Sequence[IrInput],
    states: Mapping[str, IrState],
    values: Mapping[str, IrValue],
) -> None:
    """A declared symbol MUST **have a binding point**. There are two binding points (ADR 0066
    addendum item 7):

    1. **A dimension position of an input shape** (solved from each run's actual sizes —
       `bindSymbols` in `runtime/plan.ts` on the TS side). A derived form (`2T` / `T+8`) is fine
       too — a linear expression of one symbol per dimension has a unique solution from the actual
       size (ADR 0057).
    2. **A dimension position of a states shape** (the KV capacity
       `createGenerationContext(spec.bindings)` decides — a value the user decides when creating
       the context, so baking it as an export-time constant contradicts ADR 0066 decision 3
       "static physical storage").

    MUST: **a states-only symbol (a symbol that appears only in states) must not appear in a value
    shape** (addendum item 7). Only input-derived bindings resolve ordinary value shapes, so its
    appearance would always be unbindable at run time — reject it at declaration time.
    """
    from_inputs: set[str] = set()
    for spec in inputs:
        from_inputs |= _symbols_in(spec.shape)
    from_states: set[str] = set()
    for slot in states.values():
        from_states |= _symbols_in(slot.shape)
    for symbol in symbols:
        if symbol not in from_inputs and symbol not in from_states:
            raise IrError(
                f"graph.symbols: '{symbol}' appears in no dimension position of an input shape /"
                " states shape — it cannot be bound"
            )
    for name, value in values.items():
        for symbol in sorted(_symbols_in(value.shape)):
            if symbol not in from_states or symbol in from_inputs:
                continue
            raise IrError(
                f"graph.values['{name}']: states-only symbol '{symbol}' appears in a value shape"
                " (only input-derived bindings resolve value shapes — ADR 0066 addendum item 7)"
            )


def _check_definitions(
    inputs: Sequence[IrInput],
    initializers: Mapping[str, IrInitializer],
    nodes: Sequence[IrNode],
    outputs: Sequence[str],
) -> set[str]:
    """SSA single assignment + topological order (forward references rejected) + the check that
    outputs are defined."""
    defined: set[str] = set()

    def define(name: str, where: str) -> None:
        if name in defined:
            raise IrError(
                f"{where}: value '{name}' is defined twice (violates SSA single assignment)"
            )
        defined.add(name)

    for spec in inputs:
        define(spec.name, "graph.inputs")
    for name in initializers:
        define(name, "graph.initializers")
    for index, node in enumerate(nodes):
        where = f"graph.nodes[{index}] ({node.op})"
        for ref in node.ins:
            # nodes MUST be stored in topological order — allowing forward references would leave
            # the execution order to implicit dependency analysis.
            if ref not in defined:
                raise IrError(
                    f"{where}: input '{ref}' is undefined (forward reference or undeclared)"
                )
        for out in node.outs:
            define(out, where)
    for output in outputs:
        if output not in defined:
            raise IrError(f"graph.outputs: '{output}' is undefined")
    return defined


def _check_group_quantized_shape(name: str, initializer: IrInitializer, value: IrValue) -> None:
    """Consistency of the declared shape and the group length of group-quantized storage (i4)
    (ADR 0069 decision 2; on the TS side, the group step check of the merge layer
    packages/runtime/src/format/container/bind.ts).

    The quantized axis is **the storage row** (the flattening of everything but the leading
    dimension — the in axis of linear `[O,I]`, the D axis of embedding `[V,D]`, the receptive
    field `Cin·K` of conv1d `[Cout,Cin,K]`), and its row length MUST be divisible by `group_size`
    (for a rank 2 weight this equals "the last dimension" — the rank-independent rule of ADR 0069
    decision 3). Allowing a fractional group would shorten only the last group's scale range,
    shift row boundaries off word boundaries, and make the decode of flat indices silently give
    different values (a design that removes the alignment problem itself by never creating
    fractions).
    """
    # The value range (a power of 2 and 16 or more) is guaranteed by _parse_storage. Presence is
    # optional only in the type, so restate it instead of silently skipping (the same style as
    # checkGroupQuantizedShape on the TS side).
    group_size = initializer.storage.group_size
    if group_size is None:
        raise IrError(f"graph.initializers['{name}']: storage i4 but no group_size")
    if len(value.shape) < 2:
        raise IrError(
            f"graph.values['{name}']: an initializer with storage i4 has no quantized axis"
            f" (rank {len(value.shape)} — needs rank 2 or more, a row axis and a quantized axis)"
        )
    row_length = 1
    for dim in value.shape[1:]:
        if not isinstance(dim, int):
            raise IrError(
                f"graph.values['{name}']: the shape of storage i4 cannot use a symbolic dimension"
            )
        row_length *= dim
    if row_length % group_size != 0:
        raise IrError(
            f"graph.values['{name}']: row length {row_length} of storage i4 is not divisible by"
            f" group_size {group_size} (ADR 0069 decision 2)"
        )


def _check_declarations(
    inputs: Sequence[IrInput],
    initializers: Mapping[str, IrInitializer],
    values: Mapping[str, IrValue],
    nodes: Sequence[IrNode],
    defined: set[str],
) -> None:
    """Completeness of declarations: inputs are declared by inputs[], initializers and node
    outputs by values{}, each exactly once. An orphan declaration (values nobody defines) also
    fails loudly.
    """
    input_names = {spec.name for spec in inputs}
    for name in values:
        if name in input_names:
            raise IrError(
                f"graph.values['{name}']: the input is already declared in inputs[] (declared"
                " twice)"
            )
        if name not in defined:
            raise IrError(f"graph.values['{name}']: a declaration no node defines")
    for name in initializers:
        if name not in values:
            raise IrError(f"graph.initializers['{name}']: no dtype/shape declaration in values")
        # Only INITIALIZER_STORAGE decides the pairing of semantics and storage (whether a
        # storage dtype can run is a separate layer — the support table cross-check).
        allowed = INITIALIZER_STORAGE.get(values[name].dtype)
        if allowed is None:
            raise IrError(
                f"graph.values['{name}']: the initializer's semantic dtype '{values[name].dtype}'"
                " is outside the vocabulary (f32 / i32 only)"
            )
        storage_dtype = initializers[name].storage.dtype
        if storage_dtype not in allowed:
            raise IrError(
                f"graph.initializers['{name}']: semantic dtype '{values[name].dtype}' cannot pair"
                f" with storage dtype '{storage_dtype}' ({' / '.join(allowed)} only)"
            )
        # An initializer that is not fixed before binding cannot be checked against the block
        # length of the binding table.
        if any(not isinstance(dim, int) for dim in values[name].shape):
            raise IrError(
                f"graph.values['{name}']: an initializer's shape cannot use a symbolic dimension"
            )
        if storage_dtype == "i2":
            shape = values[name].shape
            if len(shape) != 2 or any(dim <= 0 for dim in shape) or shape[1] % 16:
                raise IrError(
                    f"graph.values['{name}']: i2 requires a positive rank 2 with a row length that"
                    " is a multiple of 16"
                )
        # A shared initializer cannot declare group_size (_parse_storage) — the lender's
        # resident weight is the source of truth for the group step, and the merge layer checks
        # the lender's container (ADR 0096 stage 2).
        if storage_dtype in GROUP_STORAGE_DTYPES and not initializers[name].is_shared:
            _check_group_quantized_shape(name, initializers[name], values[name])
    for node in nodes:
        for out in node.outs:
            if out not in values:
                raise IrError(f"graph.values: no dtype/shape declaration for node output '{out}'")


def _check_state_slots(
    states: Mapping[str, IrState],
    values: Mapping[str, IrValue],
    defined: set[str],
    nodes: Sequence[IrNode],
) -> None:
    """Check state slot names (mirrors checkStateSlots on the TS side).

    A slot is not a value (it is not referenced in `ins` / `outs`; nodes reference it by name in a
    separate field — ADR 0066 decision 1, 0067 decision 4), so it is **a namespace separate from
    values**, but **a shared name is rejected**: the same name in separate namespaces only makes
    "a value name written where a slot name belongs / or the reverse" undetectable, and adds no
    expressive power. The same style as the rule that keeps scale tensor keys from colliding with
    other initializers' bodies (ADR 0019).

    **Reference completeness** (ADR 0067 decisions 4 / 5): a declared slot MUST be referenced from
    the `states` field of at least one node. A slot nobody references is the same hole as an
    orphan declaration in values: the GenerationContext silently keeps capacity it allocates but
    nobody reads (tens of MiB for KV).

    MUST: put the collision check **first** (a slot with the same name as a value is a mix-up
    whether or not it is referenced, so do not let it turn into the different diagnosis "not
    referenced").
    """
    for name in states:
        if name in defined or name in values:
            raise IrError(
                f"graph.states['{name}']: same name as a value"
                " (state slots are separate from the value namespace — the mix-up is rejected)"
            )
    referenced = {slot for node in nodes for slot in node.states.values()}
    for name in states:
        if name not in referenced:
            raise IrError(f"graph.states['{name}']: a declaration no node references")


def _check_required_ops(required_ops: Sequence[str], nodes: Sequence[IrNode]) -> None:
    """requires.ops ≡ the set of ops actually used in nodes (the precondition of the runtime
    cross-check)."""
    used = {node.op for node in nodes}
    declared = set(required_ops)
    missing = sorted(used - declared)
    extra = sorted(declared - used)
    if missing or extra:
        raise IrError(
            f"graph.requires.ops does not match the set of ops used: undeclared {missing} /"
            f" surplus {extra}"
        )


# ---- Cross-check against the runtime support table ---------------------------------------------


def _declared_dtype(graph: IrGraph, name: str) -> str:
    for spec in graph.inputs:
        if spec.name == name:
            return spec.dtype
    return graph.values[name].dtype


def _assert_sym_prefix_slice(graph: IrGraph, node: IrNode, where: str) -> None:
    """The contract of sym_prefix_slice that **needs graph context** (ADR 0010 / the same as
    packages/runtime/src/runtime/plan.ts).

    1. `sym` is in graph.symbols (otherwise the runtime cannot bind it and the prefix length is
       undetermined)
    2. `dim` is within the input rank
    3. The input's declared shape is **a static shape without symbols** (= the Tmax shape)

    MUST: 3 cannot be told apart from the numeric shape after binding (they agree on a run with T =
    Tmax). It can only be detected here, where the declared shape is visible.
    """
    sym, slices = sym_prefix_slice_attrs(node.attrs, where)
    if sym not in graph.symbols:
        raise IrError(
            f"{where}: sym '{sym}' of sym_prefix_slice is not in graph.symbols {graph.symbols}"
        )
    source = declared_shape(graph, node.ins[0])
    if any(not isinstance(dim, int) for dim in source):
        raise IrError(
            f"{where}: sym_prefix_slice input '{node.ins[0]}' has a declared shape {source} with"
            " a symbolic dimension (the input must be a static shape baked at Tmax)"
        )
    for entry in slices:
        if entry["dim"] >= len(source):
            raise IrError(
                f"{where}: dim {entry['dim']} of sym_prefix_slice is outside the input rank"
                f" {len(source)}"
            )


def _show_window(touch: tuple[int, str, int | None]) -> str:
    """A diagnostic fragment of `_assert_state_order` (`(nodes index, op name, window)` → display
    form)."""
    index, op, window = touch
    return f"nodes[{index}] ({op}) has {'no declaration' if window is None else window}"


def _assert_state_order(graph: IrGraph) -> None:
    """The order of state effects (item ② of ADR 0067 decision 5b; mirrors assertStateOrder in
    `runtime/plan.ts` on the TS side).

    State references **draw no tensor data edges**, so the DAG's topological order does not
    decide their order; the contract is the `nodes` **array order** itself. It does not depend on
    bindings, so 3 points are checked per slot:

    1. **At most one** `state_append` per slot (writing twice in one step advances the ring's
       position formula twice, and the past a reader sees changes partway through the step)
    2. If there is an append, it is **the last node that touches the slot** (a reader after the
       append would "read this step's k/v twice, as the past")
    3. The `window` of every node touching the same slot **agrees in presence and value** (the
       mapping logical col → physical row MUST be the same formula for reads and writes — ADR
       0067 decision 4; a different formula on the read side alone gives silent misreads)

    External slots (ADR 0096 stage 2) carry **3 more points** (the same discipline as
    `validateGraphContracts` / `assertStateOrder` in `plan.ts` on the TS side):

    4. **A mix** of external and owned slots **is rejected** (in a graph with even one external
       slot, every slot is external). A borrowing context has no owned slots, so with a mixed
       declaration "which allocation rule creates it" is undetermined by the graph
    5. **No** `state_append` to an external slot (the body is the lender's — a borrower writing to
       it would dirty the physical ring without advancing the lender's logical length)
    6. The readers of an external slot are **readonly attention only** / conversely, the slots
       readonly reads are **external only**. An ordinary states-form attention reading an
       external slot would mean "add this step's k/v to the lender's slot and then read", which
       directly contradicts 5

    MUST: fail loudly. In all 3 points, an "order / declaration error" is the kind of breakage that
    shows up not as an exception but as **different values**, so it can only be stopped at write
    time.
    """
    external = {name for name, slot in graph.states.items() if slot.external}
    if external and external != set(graph.states):
        owned = sorted(set(graph.states) - external)
        raise IrError(
            f"State slots: external {sorted(external)} and owned {owned} are mixed"
            " (in a graph with even one external slot, every slot MUST be external — ADR 0096"
            " stage 2)"
        )
    touches: dict[str, list[tuple[int, str, int | None]]] = {}
    for index, node in enumerate(graph.nodes):
        if not node.states:
            continue
        where = f"nodes[{index}] ({node.op})"
        # assert_node_contract has already checked the value ranges of attrs (this only looks
        # them up again).
        window = state_window(node.attrs, where)
        readonly = node.op == ATTENTION_OP and attention_readonly(node.attrs)
        for slot in node.states.values():
            touches.setdefault(slot, []).append((index, node.op, window))
            if slot in external and not readonly:
                raise IrError(
                    f"State slot '{slot}' is external but {where} touches it in a form that is"
                    " not readonly (readers MUST be readonly attention only, with no writes —"
                    " ADR 0096 stage 2)"
                )
            if readonly and slot not in external:
                raise IrError(
                    f"{where}: readonly attention reads owned slot '{slot}'"
                    " (readonly can read external slots only — ADR 0096 stage 2)"
                )
    for slot, touched in touches.items():
        appends = [entry for entry in touched if entry[1] == STATE_APPEND_OP]
        if len(appends) > 1:
            listed = ", ".join(f"nodes[{index}]" for index, _, _ in appends)
            raise IrError(
                f"State slot '{slot}': {STATE_APPEND_OP} appears {len(appends)} times ({listed})"
                " — at most once per step (ADR 0067 decision 5b)"
            )
        last = touched[-1]
        if len(appends) == 1 and last[1] != STATE_APPEND_OP:
            raise IrError(
                f"State slot '{slot}': {STATE_APPEND_OP} (nodes[{appends[0][0]}]) is followed by"
                f" reader nodes[{last[0]}] ({last[1]})"
                " (the append MUST be the last node touching the slot — ADR 0067 decision 5b)"
            )
        first = touched[0]
        mismatch = next((entry for entry in touched if entry[2] != first[2]), None)
        if mismatch is not None:
            raise IrError(
                f"State slot '{slot}': attrs.window disagrees"
                f" ({_show_window(first)} / {_show_window(mismatch)})"
                " — the mapping logical col → physical row MUST be the same formula for reads and"
                " writes (ADR 0067 decision 4)"
            )


def assert_runtime_support(graph: IrGraph) -> None:
    """Cross-check whether the M0 runtime can execute the graph (the same as assertRuntimeSupport
    in packages/runtime/src/ops/support.ts).

    MUST: check not only op names but **down to semantic dtypes and attrs**. A cross-check of names
    alone produces "in the support table yet fails at run time" (ADR 0005). Unsupported items are
    **listed in full**.

    NOTE: the scope is the semantic contracts (OP_CONTRACTS) and the capability table (the mirror
    of support.ts). The runtime's plan-time subset checks — conv3d's `groups == 1` (ADR 0118
    decision 1), `assertStaticLayoutAxis`, `assertCatAxis` (`validateGraphContracts` in
    packages/runtime/src/runtime/plan.ts) — are not copied. support.ts itself does not have them,
    so as long as this claims to be its mirror, they are not added here either. Such a shape
    passes verify, and at Session construction the runtime fails loudly naming the op and attrs.
    """
    missing_ops: set[str] = set()
    # dtype violations are deduplicated per declaration (value name) — piling them up naively
    # would give more entries than "the number of declarations to fix", weakening the meaning of
    # the listing as an indicator.
    bad_dtypes: dict[str, str] = {}
    bad_attrs: list[str] = []
    # The transfer layer's axis. Constraints exist even for inputs no node consumes, so they are
    # checked separately from the node-driven cross-check (declaration order = inputs first).
    for spec in graph.inputs:
        if spec.dtype not in IO_DTYPES:
            bad_dtypes[spec.name] = spec.dtype
    for index, node in enumerate(graph.nodes):
        contract = OP_CONTRACTS.get(node.op)
        if contract is None:
            missing_ops.add(node.op)
            continue
        where = f"nodes[{index}] ({node.op})"
        # MUST: inputs are checked against **per-slot** accepted sets. Comparing against the union
        # alone would let a slot mix-up of gather / embedding / masked_fill (value and index
        # passed the other way round) slip through the listing check as "both are in the union",
        # only to be stopped one at a time when it reaches the contract check (breaking the
        # intent "list unsupported items in full").
        for slot, name in enumerate(node.ins):
            # More inputs than the contract (an arity violation) belong to the contract check.
            # This is the listing check, so slots with no counterpart are checked against the
            # union to pick up as many as possible.
            accept = contract.slot_accept(slot) if slot < contract.arity else contract.dtypes
            dtype = _declared_dtype(graph, name)
            if dtype not in accept:
                bad_dtypes[name] = dtype
        # MUST: outputs are checked against **the range of the contract table's mapping**, **per
        # output slot** (cast is decided by attrs.to, so the whole vocabulary). Substituting the
        # input-side accepted sets would make **a correct graph fail the listing check** for ops
        # that change dtype, like comparisons (f32 → bool) or a bool sum (→ i32).
        out_accept: tuple[frozenset[str], ...] = (
            (frozenset(SEMANTIC_DTYPES),)
            if contract.kind == "cast"
            else tuple(frozenset(slot.values()) for slot in contract.output_dtypes)
        )
        for slot, name in enumerate(node.outs):
            # More outputs than the contract (an output count violation) belong to the contract
            # check, as on the input side. This is the listing check, so the extra ones are
            # checked against the union of all slots to pick up as many as possible.
            accept = out_accept[slot] if slot < len(out_accept) else frozenset().union(*out_accept)
            dtype = _declared_dtype(graph, name)
            if dtype not in accept:
                bad_dtypes[name] = dtype
        # MUST: check against **the union** of required and optional (ADR 0067's `window`).
        # Dropping the optional ones would reject a correct states-form graph for a missing
        # capability, as "unimplemented attrs" (the same projection as the union of
        # RUNTIME_SUPPORT.attrKeys on the TS side).
        unknown = sorted(
            key
            for key in node.attrs
            if key not in contract.attrs and key not in contract.optional_attrs
        )
        if unknown:
            bad_attrs.append(f"{where}: {', '.join(unknown)}")

    missing_storage: dict[str, list[str]] = {}
    # The only storage that accepts group quantization is **i4** (ADR 0069 decision 2). A
    # group_size on another storage dtype has no execution path, and silently ignoring it would
    # read per-group scales as per-channel, giving silently wrong values, so it fails as a missing
    # capability. On the TS side the merge layer (bind.ts) rejects this shape: for a codec whose
    # `grouping` in the codec ledger is "channel", groupSize MUST equal the row length (the only
    # codec with groups is int4-sym-g). A document built from a container
    # ({@link ir_graph_from_container}) gives group_size only for group codecs, so it never
    # reaches here; only the path that passes a v1 document directly does.
    group_quantized: list[str] = []
    for name, initializer in graph.initializers.items():
        dtype = initializer.storage.dtype
        if dtype not in M0_STORAGE_DTYPES:
            missing_storage.setdefault(dtype, []).append(name)
            continue
        if dtype != "i4" and initializer.storage.group_size is not None:
            group_quantized.append(name)

    if not (missing_ops or bad_dtypes or bad_attrs or missing_storage or group_quantized):
        return
    diagnostics: list[str] = []
    if missing_ops:
        diagnostics.append(
            f"unsupported ops ({len(missing_ops)}): {', '.join(sorted(missing_ops))}"
        )
    if bad_dtypes:
        listed = ", ".join(f"value '{name}': {dtype}" for name, dtype in bad_dtypes.items())
        diagnostics.append(f"unsupported semantic dtypes ({len(bad_dtypes)}): {listed}")
    if bad_attrs:
        diagnostics.append(f"unimplemented attrs ({len(bad_attrs)}): {'; '.join(bad_attrs)}")
    for dtype, users in sorted(missing_storage.items()):
        diagnostics.append(
            f"unsupported storage dtype '{dtype}' ({len(users)}): {', '.join(sorted(users))}"
        )
    if group_quantized:
        diagnostics.append(
            f"unsupported group quantization ({len(group_quantized)}):"
            f" {', '.join(sorted(group_quantized))}"
            " (group-quantized storage is i4 only — ADR 0069)"
        )
    raise ContainerError(f"Missing runtime capability — {' / '.join(diagnostics)}")


def assert_op_contracts(graph: IrGraph) -> None:
    """The contract check of every node (arity / declared output count / attrs schema / input and
    output dtype rules / shape) and the rules not decided by a node alone (the order of state
    effects — ADR 0067 decision 5b).

    NOTE: assert_runtime_support also looks at attrs and dtypes, but at a different layer — that
    one is "the check that lists missing capabilities to the model author at once", this one is
    the check "is an op in the support table built according to its contract". Both derive from
    the contract table in ops.py, so the rules never split.
    """
    for index, node in enumerate(graph.nodes):
        where = f"graph.nodes[{index}]"
        contract = assert_node_contract(node, where)
        resolve_node_dtypes(
            contract,
            node,
            [_declared_dtype(graph, name) for name in node.ins],
            [_declared_dtype(graph, name) for name in node.outs],
            where,
        )
        # Rank limit of the strided copy family (decided by the length of the declared shape
        # alone, independent of bindings).
        if contract.name in STRIDED_RANK_OPS:
            for slot, name in enumerate(node.ins):
                assert_strided_rank(
                    len(declared_shape(graph, name)), f"input {slot} '{name}'", where
                )
            out = node.outs[0]
            assert_strided_rank(len(declared_shape(graph, out)), f"output '{out}'", where)
        if contract.kind == "sym_prefix_slice":
            _assert_sym_prefix_slice(graph, node, where)
    # The order of state effects is a rule "not decided by a node alone", so it is checked once,
    # after the contract check of every node (the same order as validateGraphContracts on the TS
    # side).
    _assert_state_order(graph)
    # Output shapes are cross-checked once every node's declaration is in place (shapes.py
    # computes them from the contract rules independently, and fails when they disagree with the
    # declarations derived from torch's meta).
    assert_graph_shapes(graph)


# ---- safetensors layout (old shards, assets) ------------------------------

#: safetensors dtype → **bit** count of one element (the size table). The byte length is checked
#: by exact match with `numel × bits / 8`. A **superset** of DTYPE_BYTES in
#: `packages/runtime/src/format/safetensors.ts` on the TS side; the dialect dtypes `I4` / `I2`
#: exist only in this table (the TS reader does not accept them — the migration CLI that reads old
#: shards needs them). {@link _LEGACY_ONLY_DTYPES} limits their acceptance to migration input.
#: MUST: keep it separate from the alignment table (READER_DTYPE_ALIGN) (the three-way separation
#: of ADR 0069 decision 2) — `I4` packs 2 elements per byte, so "element size = alignment" does not
#: hold.
READER_DTYPE_BITS = {
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "I8": 8,
    "I4": 4,
    "I2": 2,
    "U8": 8,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "BOOL": 8,
}

#: safetensors dtype → the byte alignment required at **the start** of a tensor (the alignment
#: table). `I4` has no notion of element alignment and is 4 because the decode kernel binds it as
#: `array<u32>` (ADR 0069 decision 2).
READER_DTYPE_ALIGN = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I8": 1,
    "I4": 4,
    "I2": 4,
    "U8": 1,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "BOOL": 1,
}

#: Dialect dtypes the TS reader rejects (the packed 4-bit / 2-bit of old shards).
#: `assert_reader_layout` accepts them only in the migration CLI's input check
#: (`allow_legacy_dtypes=True`) — accepting them in the asset check would allow writing assets
#: where "the Python check is green but only the browser's reader fails".
_LEGACY_ONLY_DTYPES = frozenset({"I4", "I2"})

_HEADER_LENGTH_BYTES = 8


def _as_reader_index(value: Any, where: str, what: str) -> int:
    """The accepted set of the TS reader's `asIndex` (`packages/runtime/src/format/safetensors.ts`).

    MUST: check not only non-negativity but also **at most 2^53−1**. The TS side reads header JSON
    numbers as JS numbers, so a value beyond that cannot be held as an integer and is always
    rejected — if the Python side accepted it by looking only at the product of element counts, a
    file where "verify is green but only the browser's reader fails" would remain as a
    distribution. bool derives from int but is not an index.
    MUST: an integral float (`4.0`) is read as the integer — the TS reader sees one number type
    (the reason is in {@link integral_float_to_int}).
    """
    value = integral_float_to_int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_SAFE_INT:
        raise ContainerError(
            f"{where}: {what} {value!r} is not a non-negative integer (at most 2^53−1)"
        )
    return value


#: Keys a header entry always has (the fields the TS reader reads).
_READER_ENTRY_KEYS = ("dtype", "shape", "data_offsets")


def _as_reader_entry(value: Any, where: str) -> dict[str, Any]:
    """Check the structure of one header entry (an accepted set in the same style as
    `_as_reader_index`).

    MUST: do not index it raw. A missing one of the 3 keys, an entry that is not an object, or a
    shape that is not an array would leak as `KeyError` / `TypeError`, and the check's diagnosis
    would look like "the exporter broke" instead of "an invalid file" (the same reason the header
    length is bounded by the actual file length). `karume migrate` is a public CLI that consumes
    old shards made elsewhere, so the path to here does exist.
    """
    if not isinstance(value, dict):
        raise ContainerError(f"{where}: header entry is not an object: {value!r}")
    missing = [key for key in _READER_ENTRY_KEYS if key not in value]
    if missing:
        raise ContainerError(f"{where}: header entry is missing {missing}")
    if not isinstance(value["shape"], list):
        raise ContainerError(f"{where}: shape is not an array: {value['shape']!r}")
    return value


def _read_header(path: str | Path) -> tuple[dict[str, Any], int, int]:
    """Return the header JSON and `(absolute start of the data section, byte length of the data
    section)`.

    MUST: bound the declared length by the actual file length before passing it to read (the same
    defense as `legacy.safetensors_header`). Passing the u64 as is would leak a rule violation not
    as a ContainerError but as an OverflowError / MemoryError, and the check's diagnosis would look
    like "the exporter broke" instead of "an invalid file".
    """
    file = Path(path)
    file_size = file.stat().st_size
    with file.open("rb") as handle:
        raw_length = handle.read(_HEADER_LENGTH_BYTES)
        if len(raw_length) != _HEADER_LENGTH_BYTES:
            raise ContainerError(
                f"File is too short: {len(raw_length)} bytes (not even a header length)"
            )
        header_length = int.from_bytes(raw_length, "little")
        if header_length <= 0 or header_length > file_size - _HEADER_LENGTH_BYTES:
            raise ContainerError(
                f"Header length {header_length} contradicts the file length {file_size}"
            )
        header_bytes = handle.read(header_length)
        data_start = _HEADER_LENGTH_BYTES + header_length

    try:
        header = json.loads(header_bytes)
    except ValueError as cause:
        raise ContainerError(f"Cannot parse the safetensors header JSON: {cause}") from cause
    if not isinstance(header, dict):
        raise ContainerError("The safetensors header is not a top-level object")
    return header, data_start, file_size - data_start


def assert_reader_layout(path: str | Path, *, allow_legacy_dtypes: bool = False) -> None:
    """Check the safetensors layout rules (the same rules as `format/safetensors.ts` on the TS
    side).

    A file can be made that HF's `safe_open` reads but Karume cannot — the reader requires "the
    data section is covered without gaps" and "each tensor's **absolute** offset is aligned to its
    dtype's alignment unit", and the latter breaks when F32 / I32 / I4 is placed right after an F16
    with an odd element count (byte length ≡ 2 mod 4) (docs/limitations.md). It is used on 2
    paths: the input check before the migration CLI reads old shards (`karume.legacy`), and the
    check right after writing an asset safetensors (the recipes' asset writers). The dialect dtypes
    `I4` / `I2` are accepted only with `allow_legacy_dtypes=True` (the migration CLI's input
    check). The default is the same dtype set as the TS reader (the asset check).

    MUST: this check does not go through the `safetensors` reader (going through it would turn it
    into the mere claim "another reader could read it" rather than a reimplementation of the same
    rules). It reads the header JSON directly and mirrors the rules.
    """
    header, data_start, data_length = _read_header(path)

    declared = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        where = f"Tensor '{name}'"
        entry = _as_reader_entry(entry, where)
        dtype = entry["dtype"]
        if dtype not in READER_DTYPE_BITS:
            raise ContainerError(f"{where}: dtype '{dtype}' unknown to the reader")
        if dtype in _LEGACY_ONLY_DTYPES and not allow_legacy_dtypes:
            raise ContainerError(
                f"{where}: dtype '{dtype}' unknown to the reader"
                " (a dialect dtype — accepted only in migration input from old shards)"
            )
        offsets = entry["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ContainerError(
                f"{where}: data_offsets is not an array of 2 elements: {offsets!r}"
            )
        begin = _as_reader_index(offsets[0], where, "data_offsets")
        end = _as_reader_index(offsets[1], where, "data_offsets")
        count = 1
        for axis, dim in enumerate(entry["shape"]):
            count *= _as_reader_index(dim, where, f"dimension shape[{axis}]")
            # The TS side checks the product against safe integers at every step
            # (elementCount). Checking only the final product would pass an element count that
            # lost precision midway and happens to match the byte length.
            if count > MAX_SAFE_INT:
                raise ContainerError(f"{where}: element count exceeds the safe integer 2^53−1")
        bits = count * READER_DTYPE_BITS[dtype]
        # MUST: a total bit count that does not land on a byte boundary (an odd I4 element count)
        # fails loudly. The last element sticks out by half a byte, so the tensor's length is not
        # uniquely decided by the declaration.
        if bits % 8 != 0:
            raise ContainerError(
                f"{where}: {dtype} ({READER_DTYPE_BITS[dtype]} bits per element) element count"
                f" {count} is odd and does not land on a byte boundary"
            )
        if end - begin != bits // 8:
            raise ContainerError(
                f"{where}: size mismatch offsets={end - begin} "
                f"expected={bits // 8} ({dtype} {entry['shape']})"
            )
        declared.append((begin, end, name, dtype))

    cursor = 0
    for begin, end, name, dtype in sorted(declared):
        if begin != cursor:
            raise ContainerError(
                f"Tensor '{name}': the data section is not covered without gaps"
                f" (used end={cursor} / start of this tensor={begin})"
            )
        align = READER_DTYPE_ALIGN[dtype]
        if (data_start + begin) % align != 0:
            raise ContainerError(
                f"Tensor '{name}': absolute offset {data_start + begin} is not aligned to the"
                f" {dtype} alignment unit of {align} bytes"
                " (is a 4-byte type placed after an F16 with an odd element count? — "
                "the ordering convention is in karume/emit.py)"
            )
        cursor = end
    if cursor != data_length:
        raise ContainerError(
            f"{data_length - cursor} bytes of unused area at the end of the data section"
        )


# ---- Cross-check with the container format (krm / krg) -------------------------------------
#
# Mirrors `packages/runtime/src/format/container/bind.ts` (the merge layer) on the TS side. The
# rules decided by the descriptor alone (structure of the 2 documents, block layout, missing /
# surplus bindings) are already checked by the reader ({@link karume.container.read_container}),
# so what this applies is **only the rules that need the declared shape**:
#
# - the pairing of semantic dtype and codec (the encodings of `f32` / raw int32 for `i32` — a
#   crossing fails loudly)
# - payload length = the value decided by the declared shape and packing. The only difference
#   from the block length is padding (0 or more, less than 4)
# - the end of a piece sequence = `shape[0]`. Middle pieces have no padding
# - an initializer with `rowAxis != 0` cannot be split into pieces
# - the group step (per-channel: the row length / group codec: a power of 2 ≥ 16 that divides the
#   row length) and the scale length
# - the declared shape on the i2 path (`int2-off` / `ternary`) is a positive rank 2 with a row
#   length that is a multiple of 16


@dataclass(frozen=True)
class SupplyBlock:
    """The block of one body (one piece for a piece sequence)."""

    id: str
    part: int
    offset: int
    length: int
    #: The row range of the leading dimension this block carries (`[0, shape[0]]` for a whole
    #: tensor, `[0, 1]` for rank 0).
    rows: tuple[int, int]
    #: Byte length of the payload (the block length without tail padding).
    payload_bytes: int


@dataclass(frozen=True)
class ScaleSupply(SupplyBlock):
    """The companion scale, handed over as **one contiguous scale** whether or not it is split
    (mirrors TS `ScaleSupply`).

    Unsplit, it is the one declared block and `pieces` is empty. Split (container-v1 §5 rule ③ —
    ADR 0108 addendum 15), `pieces` holds the declared blocks in row order, piece k covering the
    rows of the body's piece k, and the fields above describe the **joined read**: `id` is
    {@link karume.container.joined_scale_id} of the pieces (not a declared block — the key under
    which {@link karume.container.ReadContainer.block} returns the pieces concatenated), `length`
    is the byte count that read returns, `offset` is the first piece's, and `part` is the part
    every piece is in.
    """

    pieces: tuple[SupplyBlock, ...] = ()


@dataclass(frozen=True)
class InitializerSupply:
    """The supply plan of one initializer (which blocks the body comes from)."""

    encoding: BlockEncoding
    #: The body's block sequence (one for a whole tensor).
    blocks: tuple[SupplyBlock, ...]
    #: The supplier. The const region (part 1, owned by the graph) or the weight side (model
    #: descriptor).
    origin: Literal["const", "model"]
    #: The companion scale (quantized codecs only) — one contiguous scale even when split.
    scale: ScaleSupply | None = None


@dataclass(frozen=True)
class BoundGraph:
    """The merge result of one graph (declaration + the supply plans of every non-shared
    initializer)."""

    declaration: Mapping[str, Any]
    supplies: Mapping[str, InitializerSupply]


def _located_block(
    graph: GraphDescriptor, model: ModelDescriptor | None
) -> Callable[[str, str], SupplyBlock]:
    """block id → location (part 1 for the const table, the declared part for the model
    table)."""
    const = {block.id: block for block in graph.const_blocks}
    data = {block.id: block for block in (model.blocks if model is not None else ())}

    def locate(block_id: str, where: str) -> SupplyBlock:
        found = const.get(block_id)
        part = 1
        if found is None:
            record = data.get(block_id)
            if record is None:
                raise ContainerFormatError(f"{where}: undeclared block '{block_id}'")
            found, part = record, record.part
        # rows / payload_bytes are filled in by the caller (the side that knows the declared
        # shape).
        return SupplyBlock(found.id, part, found.offset, found.length, (0, 0), 0)

    return locate


def _declared_tensor(
    declaration: Mapping[str, Any], name: str, where: str
) -> tuple[str, list[int]]:
    """An initializer's semantic dtype and **concrete shape** (bytes with a symbolic dimension
    cannot exist)."""
    values = declaration.get("values")
    if not isinstance(values, dict) or not isinstance(values.get(name), dict):
        raise ContainerFormatError(f"{where}: no dtype / shape declaration in `values`")
    value = values[name]
    dtype = value.get("dtype")
    if dtype not in SEMANTIC_DTYPES:
        raise ContainerFormatError(f"{where}: semantic dtype is outside the vocabulary: {dtype!r}")
    raw = value.get("shape")
    if not isinstance(raw, list):
        raise ContainerFormatError(f"{where}: the shape in `values` is not an array")
    shape: list[int] = []
    for raw_dim in raw:
        # Reading container JSON does not turn floats into ints — align with the same accepted
        # set as the graph's `_parse_shape` (the runtime reads `4.0` as dimension 4).
        dim = integral_float_to_int(raw_dim)
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ContainerFormatError(
                f"{where}: an initializer's shape cannot use a symbolic dimension ({dim!r})"
            )
        shape.append(dim)
    return dtype, shape


def _assert_padded(block: SupplyBlock, payload: int, where: str) -> None:
    """The relation of block length and payload length (container-v1 §4.1 — padding is 0 or more,
    less than 4)."""
    if not 0 <= block.length - payload < BLOCK_TAIL_ALIGN:
        raise ContainerFormatError(
            f"{where}: block '{block.id}' length {block.length} is not payload {payload} bytes +"
            f" padding (less than {BLOCK_TAIL_ALIGN})"
        )


def _is_i2_shape(shape: Sequence[int]) -> bool:
    """The logical shape of the i2 path, `[rows, width]` (mirrors
    `packages/runtime/src/format/i2.ts`)."""
    return len(shape) == 2 and all(dim > 0 for dim in shape) and shape[1] % 16 == 0


def _plan_pieces(
    where: str,
    shape: Sequence[int],
    payload: int,
    encoding: BlockEncoding,
    pieces: Sequence[tuple[str, tuple[int, int]]],
    locate: Callable[[str, str], SupplyBlock],
) -> list[SupplyBlock]:
    """The supply plan of a piece sequence (row range → byte range). A step after the reader has
    checked order and coverage."""
    if encoding.row_axis not in (None, 0):
        raise ContainerFormatError(
            f"{where}: an initializer with rowAxis {encoding.row_axis} cannot be split into pieces"
            " (rule ④)"
        )
    rows = shape[0] if shape else 1
    if not shape or rows == 0 or payload % rows != 0:
        raise ContainerFormatError(
            f"{where}: payload of {payload} bytes is not divisible by the leading dimension of"
            f" {rows} rows, so it cannot be split into pieces"
        )
    if pieces[-1][1][1] != rows:
        raise ContainerFormatError(
            f"{where}: the end of the piece sequence, row {pieces[-1][1][1]}, differs from the"
            f" declared shape's leading dimension of {rows} rows (rule ④)"
        )
    row_bytes = payload // rows
    planned: list[SupplyBlock] = []
    for index, (block_id, (begin, end)) in enumerate(pieces):
        by = f"{where} piece[{index}]"
        found = locate(block_id, by)
        piece_bytes = (end - begin) * row_bytes
        if index < len(pieces) - 1:
            # Padding cannot be put on a middle piece (it would overwrite the start of the next
            # piece).
            if found.length != piece_bytes:
                raise ContainerFormatError(
                    f"{by}: middle piece block length {found.length} differs from the byte count"
                    f" of the row range, {piece_bytes} (no padding allowed)"
                )
        else:
            _assert_padded(found, piece_bytes, by)
        planned.append(replace(found, rows=(begin, end), payload_bytes=piece_bytes))
    return planned


def _plan_supply(
    where: str,
    dtype: str,
    shape: Sequence[int],
    encoding: BlockEncoding,
    block: str | None,
    pieces: Sequence[tuple[str, tuple[int, int]]] | None,
    locate: Callable[[str, str], SupplyBlock],
    origin: Literal["const", "model"],
) -> InitializerSupply:
    """The supply plan of one initializer (declared shape × encoding × block table)."""
    entry = codec_entry(encoding.codec)
    if entry.layout not in INITIALIZER_STORAGE.get(dtype, ()):
        raise ContainerFormatError(
            f"{where}: semantic dtype '{dtype}' cannot pair with codec '{encoding.codec}'"
        )
    if entry.layout == "i2" and not _is_i2_shape(shape):
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' requires a declared shape of positive rank 2 with a"
            f" row length that is a multiple of 16 ([{','.join(str(dim) for dim in shape)}])"
        )
    numel = math.prod(shape)
    payload = payload_bytes(encoding.codec, numel, where)
    rows = shape[0] if shape else 1
    if pieces is None:
        if block is None:
            raise ContainerFormatError(f"{where}: the supply form is neither `block` nor `pieces`")
        found = locate(block, where)
        _assert_padded(found, payload, where)
        blocks = [replace(found, rows=(0, rows), payload_bytes=payload)]
    else:
        blocks = _plan_pieces(where, shape, payload, encoding, pieces, locate)
    if entry.scale == "forbidden":
        return InitializerSupply(encoding, tuple(blocks), origin)

    if encoding.row_axis is None or encoding.group_size is None or not encoding.scale_blocks:
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' is quantized, so rowAxis / groupSize / scale are"
            " required"
        )
    row_axis, group_size = encoding.row_axis, encoding.group_size
    assert_row_axis(encoding.codec, row_axis, where)
    if not shape or len(shape) <= row_axis:
        raise ContainerFormatError(
            f"{where}: for rowAxis {row_axis} the rank of the declared shape"
            f" [{','.join(str(dim) for dim in shape)}] is too small"
        )
    row_count = shape[row_axis]
    row_length = numel // row_count if row_count else 0
    if entry.grouping == "channel" and group_size != per_channel_group_size(row_length):
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' is per-channel, so groupSize MUST equal the row"
            f" length {per_channel_group_size(row_length)} (declared {group_size})"
        )
    if entry.grouping == "group":
        if group_size < MIN_GROUP_SIZE or group_size & (group_size - 1) != 0:
            raise ContainerFormatError(
                f"{where}: groupSize {group_size} is not a power of 2 that is {MIN_GROUP_SIZE} or"
                " more (ADR 0069 decision 2)"
            )
        if row_length % group_size != 0:
            raise ContainerFormatError(
                f"{where}: row length {row_length} (= numel / shape[{row_axis}]) is not divisible"
                f" by groupSize {group_size} (ADR 0069 decision 2)"
            )
    steps = group_count(row_length, group_size)
    payload_of_scale = scale_bytes(encoding.codec, row_count * steps)
    if encoding.scale_pieces is not None:
        scale = _plan_split_scale(
            where,
            pieces is not None,
            blocks,
            encoding.scale_pieces,
            row_count,
            steps * scale_step_bytes(encoding.codec),
            payload_of_scale,
            locate,
        )
        return InitializerSupply(encoding, tuple(blocks), origin, scale)
    assert encoding.scale_block is not None
    found = locate(encoding.scale_block, f"{where} scale")
    _assert_padded(found, payload_of_scale, f"{where} scale")
    return InitializerSupply(
        encoding,
        tuple(blocks),
        origin,
        ScaleSupply(
            found.id, found.part, found.offset, found.length, (0, row_count), payload_of_scale
        ),
    )


def _plan_split_scale(
    where: str,
    body_is_pieced: bool,
    body: Sequence[SupplyBlock],
    pieces: Sequence[tuple[str, tuple[int, int]]],
    row_count: int,
    row_bytes: int,
    payload: int,
    locate: Callable[[str, str], SupplyBlock],
) -> ScaleSupply:
    """The plan of a split scale (rule ③ — ADR 0108 addendum 15; mirrors TS `planSplitScale`):
    one scale block per body piece, piece k covering exactly the rows of body piece k, its length
    `rows · row_bytes` with `row_bytes = (row length / groupSize) · step bytes` (middle pieces
    exactly, the last with padding under 4). The reader has already checked that the list has 2 or
    more entries continuing from row 0."""
    by = f"{where} scale"
    # A split scale follows the body's pieces, so a body in one block has no boundaries to split
    # it on (the writer fails loudly instead — `karume.container._cut_scale`).
    if not body_is_pieced:
        raise ContainerFormatError(
            f"{by}: a split scale needs a piece list (the scale is split on the body's pieces)"
        )
    if len(pieces) != len(body):
        raise ContainerFormatError(
            f"{by}: {len(pieces)} scale pieces for {len(body)} body pieces (one scale piece per"
            " body piece — rule ③)"
        )
    # A piece list exists only for rowAxis 0 (rule ④), so the scale's rows are the body's leading
    # dimension. One scale row is its steps' bytes, not `payload / row_count`: an f16 payload
    # carries the word padding of the whole scale, which belongs to no row.
    planned: list[SupplyBlock] = []
    for index, ((block_id, rows), data) in enumerate(zip(pieces, body, strict=True)):
        piece_by = f"{by} piece[{index}]"
        if rows != data.rows:
            raise ContainerFormatError(
                f"{piece_by}: rows [{rows[0]}, {rows[1]}) differ from the rows"
                f" [{data.rows[0]}, {data.rows[1]}) of body piece {index} (rule ③)"
            )
        found = locate(block_id, piece_by)
        piece_bytes = (rows[1] - rows[0]) * row_bytes
        if index < len(pieces) - 1:
            # A middle piece carries no padding (§4.1 — the same rule as the body's pieces).
            if found.length != piece_bytes:
                raise ContainerFormatError(
                    f"{piece_by}: block '{found.id}' length {found.length} differs from the"
                    f" {piece_bytes} bytes of its rows (a middle piece has no padding)"
                )
        else:
            _assert_padded(found, piece_bytes, piece_by)
        planned.append(replace(found, rows=rows, payload_bytes=piece_bytes))
    return ScaleSupply(
        joined_scale_id([piece.id for piece in planned]),
        planned[0].part,
        planned[0].offset,
        sum(piece.length for piece in planned),
        (0, row_count),
        payload,
        tuple(planned),
    )


def bind_graphs(graph: GraphDescriptor, model: ModelDescriptor | None) -> dict[str, BoundGraph]:
    """Merge the graph descriptor and the binding table and decide the supply plan of each
    initializer (mirrors TS `bindGraphs`).

    When `model` is `None` (`krg`), only const supplies are filled — initializers that need weights
    remain declarations without a supply (a `krg` alone cannot build a Session).
    """
    locate = _located_block(graph, model)
    constants = {(entry.graph, entry.initializer): entry for entry in graph.constants}
    bound: dict[str, BoundGraph] = {}
    for graph_name, declaration in graph.graphs.items():
        initializers = declaration.get("initializers")
        if not isinstance(initializers, dict):
            raise ContainerFormatError(f"graph '{graph_name}': no `initializers` section")
        supplies: dict[str, InitializerSupply] = {}
        for name, init in initializers.items():
            if isinstance(init, dict) and init.get("shared") is True:
                continue
            where = f"graph '{graph_name}' initializer '{name}'"
            dtype, shape = _declared_tensor(declaration, name, where)
            constant = constants.get((graph_name, name))
            if constant is not None:
                supplies[name] = _plan_supply(
                    where, dtype, shape, constant.encoding, constant.block, None, locate, "const"
                )
                continue
            supply = (model.binding.get(graph_name, {}) if model is not None else {}).get(name)
            if supply is None:
                if model is None:
                    continue
                raise ContainerFormatError(f"{where}: no supply in the binding table")
            supplies[name] = _plan_supply(
                where, dtype, shape, supply.encoding, supply.block, supply.pieces, locate, "model"
            )
        bound[graph_name] = BoundGraph(declaration, supplies)
    return bound


#: Storage of an initializer without a supply (semantic dtype → raw storage). The only slots this
#: applies to are shared declarations (ADR 0096 stage 2) and `krg` (no binding table at all), and
#: in both **this container has no body**.
_PLAIN_STORAGE_FOR: Mapping[str, str] = {"f32": "f32", "i32": "i32"}


def _storage_document(supply: InitializerSupply) -> dict[str, Any]:
    """The binding table's `encoding` → the IR v1 `storage` descriptor (the inverse of
    {@link karume.container.Encoding}).

    In v1 `scale` is "the key of the scale tensor", but in the container the scale is a block, so
    the **block id** is spelled as is (diagnostics name the actual thing — the joined id for a
    split scale). Only group codecs emit `group_size` — old per-channel declarations have no
    `group_size`, and the container's `groupSize` (= the row length) is a copy derivable from the
    ledger, so putting it back would give "i8 with a group_size" (the shape the runtime support
    check rejects).
    """
    encoding = supply.encoding
    entry = codec_entry(encoding.codec)
    document: dict[str, Any] = {"dtype": entry.layout}
    if entry.scale == "required":
        assert supply.scale is not None
        document["scale"] = supply.scale.id
        if entry.grouping == "group":
            document["group_size"] = encoding.group_size
    return document


def ir_graph_from_container(read: ReadContainer, graph_name: str) -> IrGraph:
    """The container's 2 documents → IR v1 {@link IrGraph} (passing every check of
    `parse_ir_graph`).

    The graph descriptor holds **an IR v2 document** (docs/ir-v2.md — the initializer name is the
    tensor key, and storage has moved out to the binding table), so the difference from v1 is only
    2 points: `version` and `initializers`. What this returns is "the v1 document with those 2
    points filled in" passed through {@link parse_ir_graph}, so completeness of declarations, SSA,
    forward references, symbol bindability and the state slot rules are checked by **the same
    single path** (copying the checks for v2 would let only one of them go stale on the day a rule
    moves).

    The source of truth for storage is **the binding table** (`encoding.codec`); only slots without
    a supply (shared declarations and `krg`) get the raw storage of the semantic dtype — those
    slots hold not a single byte in this container, so the storage claim belongs to the lender's
    container (and the same checks applied to that container).
    """
    declaration = read.graph.graphs.get(graph_name)
    if declaration is None:
        raise ContainerError(
            f"The container has no graph '{graph_name}' (declared: {sorted(read.graph.graphs)})"
        )
    where = f"graph '{graph_name}'"
    supplies = bind_graphs(read.graph, read.model)[graph_name].supplies
    root = _as_object(declaration, where)
    values = _as_object(root.get("values"), f"{where}.values")
    declared = _as_object(root.get("initializers"), f"{where}.initializers")
    initializers: dict[str, Any] = {}
    for name, raw in declared.items():
        supply = supplies.get(name)
        if supply is not None:
            initializers[name] = {"tensor": name, "storage": _storage_document(supply)}
            continue
        semantic = _as_object(values.get(name), f"{where}.values['{name}']").get("dtype")
        plain = _PLAIN_STORAGE_FOR.get(semantic) if isinstance(semantic, str) else None
        if plain is None:
            raise ContainerError(
                f"{where} initializer '{name}': no supply, yet the semantic dtype is"
                f" {semantic!r} (it cannot have raw storage)"
            )
        shared = _as_object(raw, f"{where}.initializers['{name}']").get("shared") is True
        borrowed = {"shared": {"tensor": name}} if shared else {"tensor": name}
        initializers[name] = {**borrowed, "storage": {"dtype": plain}}
    document = {**root, "version": IR_VERSION, "initializers": initializers}
    return parse_ir_graph(json.dumps(document, separators=(",", ":"), allow_nan=False))


def assert_ir_accepted(read: ReadContainer) -> None:
    """Apply **the IR acceptance rules** (op vocabulary / runtime support / op contracts) to every
    graph the container declares.

    MUST: both the path that builds a distribution and the path that accepts one go through here
    (the input check of `karume dist`, and `karume verify`). The structural check
    ({@link verify_container}) only looks at "can it be opened as a container", so without this
    **a container declaring an op outside the vocabulary** would be put in place in a
    distribution and fail only at the caller's `createSession` (the very purpose the module doc
    states).
    """
    for name in sorted(read.graph.graphs):
        try:
            graph = ir_graph_from_container(read, name)
            assert_runtime_support(graph)
            assert_op_contracts(graph)
        except (ContainerError, ContainerFormatError, IrError, OpContractError, DimError) as cause:
            raise ContainerError(f"graph '{name}': {cause}") from cause


@dataclass(frozen=True)
class VerifiedContainer:
    """The result of {@link verify_container} (the opened reader, and the supply plans decided by
    the merge)."""

    read: ReadContainer
    graphs: dict[str, BoundGraph]


def verify_container(paths: Sequence[str | Path], *, blocks: bool = False) -> VerifiedContainer:
    """Open a container (the part sequence of a `krm` / one `krg`) and take it through the merge.

    `paths` is one file in single form, and in split form the part sequence **in order from part
    0**. By default every "rule decided by declarations" applies (the reader's structural check +
    the merge layer above), and **no actual bytes are read** (the three separate hash roles of
    §7). `blocks=True` takes every block again and compares down to sha256, and also applies the
    code check of `ternary` payloads (§6.3) — this MUST be the only path that "verifies a
    container" (the CLI goes through here too).
    """
    read = read_container([Path(path) for path in paths])
    if blocks:
        read.verify_blocks()
    bound = bind_graphs(read.graph, read.model)
    if blocks:
        _assert_ternary_payloads(read, bound)
    return VerifiedContainer(read, bound)


#: Bytes in which some 2-bit code is 0 (outside the `ternary` value range — container-v1 §6.3).
_TERNARY_CODE_ZERO_BYTES = frozenset(
    byte for byte in range(256) if any((byte >> shift) & 3 == 0 for shift in (0, 2, 4, 6))
)


def _assert_ternary_payloads(read: ReadContainer, bound: Mapping[str, BoundGraph]) -> None:
    """The extra condition of a `ternary` declaration: every 2-bit code of the payload is in
    `{1, 2, 3}` (container-v1 §6.3).

    Mirrors TS `assertTernaryCodes` (applied every time a block is read at load). Without it here,
    "verify is green but loading fails" would result. Only the payload is checked (tail padding
    excluded).
    """
    for graph_name, graph in bound.items():
        for name, supply in graph.supplies.items():
            if supply.encoding.codec != "ternary":
                continue
            for block in supply.blocks:
                payload = read.block(block.id)[: block.payload_bytes]
                offending = next(
                    (
                        index
                        for index, byte in enumerate(payload)
                        if byte in _TERNARY_CODE_ZERO_BYTES
                    ),
                    None,
                )
                if offending is not None:
                    raise ContainerFormatError(
                        f"graph '{graph_name}' initializer '{name}': the ternary payload has"
                        f" code 0 (outside the ternary value range) (byte {offending})"
                    )


# ---- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify containers (krm / krg) against every rule (2 documents + merge +"
        " sha256 of every block)"
    )
    parser.add_argument(
        "models",
        type=Path,
        nargs="+",
        help="containers to verify (several allowed; a part 0 file, a single-form file or a"
        " representative path — a split form is verified together as its numbered part sequence)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Verify the given containers one at a time.

    The arguments are **representative paths of containers** (a part 0 file may be passed too);
    a split container is resolved to its numbered part sequence and verified together
    ({@link karume.container.container_parts}) — verifying a single part alone could only say
    "the 2 documents are missing" and could see neither the merge nor the block sha256.

    What applies is the structural check of the 2 documents, the merge (`bind_graphs`), **the
    sha256 cross-check of every block taken again**, and the IR acceptance rules
    ({@link assert_ir_accepted} — **the same single path** as the assembly side's check).

    MUST: stop at the file that fails (do not verify the rest and collect at the end) — the
    exception spells out where the rule was broken, so raising it as is carries the most
    information.
    """
    args = build_parser().parse_args(argv)
    for path in args.models:
        print(_verify_container_line(path))


def _verify_container_line(path: Path) -> str:
    parts = container_parts(path)
    verified = verify_container(parts, blocks=True)
    read, bound = verified.read, verified.graphs
    assert_ir_accepted(read)
    assets = sorted(read.model.assets) if read.model is not None else []
    return (
        f"{path}: parts={len(parts)} blocks={len(read.block_ids)}"
        f" graphs={','.join(sorted(bound))}"
        f" initializers={sum(len(graph.supplies) for graph in bound.values())}"
        f" assets={','.join(assets) if assets else '(none)'}"
    )


if __name__ == "__main__":
    main()
