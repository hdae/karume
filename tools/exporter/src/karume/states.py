"""Graph surgery that rewrites a torch.export'ed graph into the **states form** (ADR 0067
decisions 4 / 5 / 5b).

torch.export cannot trace KV state. A recipe exports "chunk-local causal self-attention" (the
previous form, where attention takes a mask as its fourth input), and this module rewrites that IR
into the states form (the past in state slots, this step's k/v in `ins`). It is a
**model-independent mechanism**, so it lives in the generic core (Gemma 4 and MiniCPM5 go through
the same surgery).

The surgery is three operations:

1. **Reinterpretation** — drop the mask input of the target attention and attach the `states` field
   `{k, v}` (plus attrs `window` for sliding). causal becomes a predicate computation, so the mask
   tensor is no longer needed.
2. **Inserting the writes** — one `state_append` per slot, placed **right after the last reader of
   that slot** (the emission rule of ADR 0067 decision 5b = all readers → append).
3. **Pruning** — drop the nodes and declarations not reachable from `outputs` and `state_append`.
   The Tmax² mask constant and `sym_prefix_slice` (the result of the constant folding of ADR 0010)
   die here — kept, "an f32 of Tmax² nobody reads" would sit in the distribution.

MUST: a **pure function** (does not mutate the input graph). Unless the graphs before and after the
surgery can go through the same verification, the origin of a broken IR (export or surgery) cannot
be told apart.

NOTE: the check of the result's correctness is verify (states check, order check, shape check);
this module **only assembles a form that passes it**. The same rules are not copied here (no double
bookkeeping).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from karume.dims import MAX_SAFE_INT, parse_dim
from karume.ir import IrDim, IrGraph, IrNode, IrState
from karume.ops import ATTENTION_OP, STATE_APPEND_OP, integral_float_to_int
from karume.shapes import declared_shape

#: The state slot dtype (ADR 0066 decision 2). The source of truth for the vocabulary is
#: `verify.STATE_DTYPES`, where f16 is only reserved (addendum 5) — the surgery can create only f32
#: slots, which can actually run.
_SLOT_DTYPE = "f32"


class StatesFormError(ValueError):
    """The surgery into the states form fails (a mixed-up target, a wiring mistake, a wrong
    spec)."""


@dataclass(frozen=True)
class StateAttentionSpec:
    """The spec of one attention to rewrite into the states form.

    `output` is the target node's `outs[0]` — with SSA single assignment, the value name is itself
    the node's unique identifier.

    `window` is the sliding window width (`None` = the whole context — the optional attrs of ADR
    0067 decision 4), and `capacity` is a **numeric override** of the slot capacity (`None` uses
    the plan's capacity symbol). The capacity is a value `createGenerationContext` decides, so a
    symbol is the default; baking it as an export-time constant is a form that accepts "the
    capacity cannot be chosen at run time" (ADR 0066 decision 3).
    """

    output: str
    k_slot: str
    v_slot: str
    window: int | None = None
    capacity: int | None = None


@dataclass(frozen=True)
class StatesPlan:
    """The surgery spec for one graph (the capacity symbol + the target attentions in order)."""

    capacity_symbol: str
    attentions: tuple[StateAttentionSpec, ...]


@dataclass
class _Slot:
    """A state slot being registered.

    `source` is the input name of `state_append` (= the value the readers receive as k / v), and
    `last_reader` is the **nodes array index** of the last reader that touches this slot. KV-sharing
    layers (several attentions reading the same slot) are matched against the existing entry on the
    second and later registrations.
    """

    shape: list[IrDim]
    window: int | None
    source: str
    last_reader: int


def to_states_form(graph: IrGraph, plan: StatesPlan) -> IrGraph:
    """Returns a **new graph** with the previous-form attentions of the graph rewritten into the
    states form.

    The input `graph` is only read (MUST — module docstring). The returned graph has the states
    section + `state_append`, with the constant-folding leftovers of the mask pruned.
    """
    nodes = list(graph.nodes)
    slots: dict[str, _Slot] = {}
    for spec in plan.attentions:
        where = f"attention '{spec.output}'"
        index = _target_index(nodes, spec.output, where)
        node = nodes[index]
        _assert_convertible(node, where)
        window = _window(spec.window, where)
        capacity: IrDim = plan.capacity_symbol if spec.capacity is None else spec.capacity
        # Register in k / v order (appends at the same position are ordered this way —
        # _with_appends).
        for slot_name, source in ((spec.k_slot, node.ins[1]), (spec.v_slot, node.ins[2])):
            candidate = _Slot(
                shape=_slot_shape(graph, source, capacity, where),
                window=window,
                source=source,
                last_reader=index,
            )
            _register(slots, slot_name, candidate, where)
        attrs = dict(node.attrs)
        if window is not None:
            attrs["window"] = window
        nodes[index] = IrNode(
            op=node.op,
            # Drop the mask (fourth input), leaving the three q / k / v.
            ins=list(node.ins[:3]),
            outs=list(node.outs),
            attrs=attrs,
            states={"k": spec.k_slot, "v": spec.v_slot},
        )
    return _prune(graph, _with_appends(nodes, slots), slots, _symbols(graph, plan, slots))


@dataclass(frozen=True)
class ExternalAttentionSpec:
    """The spec of one attention to rewrite into the **external states form** (ADR 0096 stage 2).

    `output` is the target node's `outs[0]` (with SSA single assignment, the node's unique
    identifier). `k_input` / `v_input` are the **graph input names** of the K / V placed for
    tracing, and the surgery drops these two (the borrower has no k/v for this step — it reads only
    the past).

    MUST: the slot form **takes `kv_heads` / `head_dim` / `capacity` from the spec** (it is not
    derived from the declared shapes of the k/v inputs). The external slot's storage belongs to the
    lender context, and the borrower's declaration is correct only by "matching the lender", so it
    must not follow the shape of a throwaway placeholder put in for tracing (the day the
    placeholder's T is made smaller, the declaration would silently move).
    """

    output: str
    k_slot: str
    v_slot: str
    k_input: str
    v_input: str
    kv_heads: int
    head_dim: int
    #: The slot capacity (a number, or the plan's symbol).
    capacity: IrDim
    window: int | None = None


@dataclass(frozen=True)
class ExternalStatesPlan:
    """The external surgery spec for one graph (the capacity symbol + the target attentions in
    order)."""

    capacity_symbol: str
    attentions: tuple[ExternalAttentionSpec, ...]


def to_external_states_form(graph: IrGraph, plan: ExternalStatesPlan) -> IrGraph:
    """Returns a new graph with the previous-form attentions of the graph rewritten into the
    **external states form**.

    It differs from `to_states_form` in three points only, and the discipline (pure function, the
    roots of `_prune`, checks in verify) is the same:

    1. The node becomes the **readonly form** with `ins=[q]` (`attrs.readonly = True`). It has
       neither this step's k / v nor a mask — the borrower reads only the columns
       `[column_base, P)` of the lender's slots
    2. **No** `state_append` is inserted (the storage belongs to the lender). The slot declaration
       is `external: True`
    3. The dropped k / v graph inputs are exempt from `_prune`'s orphan-input rejection (two inputs
       dropped on purpose)

    MUST: the input `graph` is only read (pure function — module docstring).
    MUST: the mask input (fourth input) is dropped as before. The borrower's q is one row, in a
    bidirectional "look at every column" form, so the mask tensor carries no information (the
    column narrowing is held by window and the logical length).
    """
    nodes = list(graph.nodes)
    slots: dict[str, _Slot] = {}
    dropped: set[str] = set()
    for spec in plan.attentions:
        where = f"attention '{spec.output}'"
        index = _target_index(nodes, spec.output, where)
        node = nodes[index]
        _assert_convertible(node, where)
        window = _window(spec.window, where)
        # The indices are K = 1 / V = 2 (`_assert_dropped_input` matches each one separately —
        # looking at a set would let a K / V mix-up through).
        for slot_name, source, kv_index in (
            (spec.k_slot, spec.k_input, 1),
            (spec.v_slot, spec.v_input, 2),
        ):
            _assert_dropped_input(graph, node, source, kv_index, where)
            dropped.add(source)
            candidate = _Slot(
                shape=[1, spec.kv_heads, spec.capacity, spec.head_dim],
                window=window,
                source=source,
                last_reader=index,
            )
            _register(slots, slot_name, candidate, where)
        attrs = dict(node.attrs)
        attrs["readonly"] = True
        if window is not None:
            attrs["window"] = window
        nodes[index] = IrNode(
            op=node.op,
            # Keep only q (drop this step's k / v and the mask).
            ins=[node.ins[0]],
            outs=list(node.outs),
            attrs=attrs,
            states={"k": spec.k_slot, "v": spec.v_slot},
        )
    return _prune(
        graph,
        nodes,
        slots,
        _symbols(graph, plan, slots),
        external=True,
        droppable_inputs=dropped,
    )


def _assert_dropped_input(
    graph: IrGraph, node: IrNode, source: str, index: int, where: str
) -> None:
    """Checks that the k / v to drop is **the graph input that the node actually reads at that
    position**.

    `index` is the expected index in `ins` (K = 1 / V = 2). MUST: match **each** index separately
    (looking at the set `{ins[1], ins[2]}` lets through a mix-up that passes the original V as K and
    the original K as V; with equal shapes even the slot registration succeeds and the result is a
    silent wrong value — the borrower runs reading V from the lender's K slot).

    MUST: also check that it is a graph input. Naming something that is not an input "drops an
    intermediate value as if it were an input"; `_prune` removes it together with its readers, so
    only the output values change, silently.
    """
    role = {1: "k", 2: "v"}[index]
    if source != node.ins[index]:
        raise StatesFormError(
            f"{where}: {role} given as '{source}' is not this node's {role}"
            f" (ins[{index}] = '{node.ins[index]}')"
            " (k is exactly ins[1] / v is exactly ins[2])"
        )
    if not any(spec.name == source for spec in graph.inputs):
        raise StatesFormError(
            f"{where}: k / v to drop '{source}' is not a graph input"
            " (the external surgery can drop only inputs placed for tracing)"
        )


def _target_index(nodes: Sequence[IrNode], output: str, where: str) -> int:
    """The index of the node whose `outs[0]` is `output`."""
    for index, node in enumerate(nodes):
        if node.outs and node.outs[0] == output:
            return index
    raise StatesFormError(f"{where}: no node in the graph defines this value")


def _assert_convertible(node: IrNode, where: str) -> None:
    """The target is exactly a **previous-form attention** (four inputs including the mask, no
    states field).

    MUST: check all three points. A different op is a mixed-up spec, a states field means a double
    surgery (it would write this step's k/v twice), and three `ins` mean "an export without a mask"
    (causal is expressed in another form = the premise of the surgery is broken).

    MUST: put the states-field check **before** the arity check. A node that already went through
    the surgery has three `ins`, so in the reverse order a double surgery would turn into the same
    diagnostic as "an export without a mask".
    """
    if node.op != ATTENTION_OP:
        raise StatesFormError(
            f"{where}: op is '{node.op}' (only '{ATTENTION_OP}' can go through the surgery)"
        )
    if node.states:
        raise StatesFormError(f"{where}: already in the states form ({node.states})")
    if len(node.ins) != 4:
        raise StatesFormError(
            f"{where}: {len(node.ins)} inputs (only the four including the mask = q / k / v / mask)"
        )


def _window(window: int | None, where: str) -> int | None:
    """The sliding window width (`None` = the whole context).

    MUST: reject bool. Python's bool derives from int, so `True` would silently pass as a window
    width of 1 (the same discipline as `_assert_integer_attr` in the contract layer).
    MUST: accept the same set as the runtime's `window` attr (`Number.isSafeInteger`, at least 1):
    an integral float (`4.0`) is written as the integer, and a width beyond 2^53−1 is rejected
    (the reason is in {@link integral_float_to_int}).
    """
    if window is None:
        return None
    value = integral_float_to_int(window)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_SAFE_INT:
        raise StatesFormError(f"{where}: window {window!r} is not a positive integer")
    return value


def _slot_shape(graph: IrGraph, source: str, capacity: IrDim, where: str) -> list[IrDim]:
    """The declaration `[B,Hkv,M,D]` of this step's k / v → the concrete slot shape including the
    capacity, `[B,Hkv,C,D]`.

    MUST: B / Hkv / D are **derived from the declaration** (not taken as arguments). If the surgery
    held its own values, the shape agreement between the slot and ins (② of ADR 0067 decision 4)
    could break through a wrong spec.
    """
    shape = declared_shape(graph, source)
    if len(shape) != 4:
        raise StatesFormError(
            f"{where}: '{source}' declared shape {shape} is not rank-4 [B,Hkv,M,D]"
        )
    return [shape[0], shape[1], capacity, shape[3]]


def _register(slots: dict[str, _Slot], name: str, candidate: _Slot, where: str) -> None:
    """Registers a slot (KV-sharing layers bring several readers to the same name — ADR 0067
    decision 4).

    MUST: on the second and later registrations, require an **exact match** of the derived shape
    (= B / Hkv / D / capacity), `window` and the append's input name. By convention a sharing layer
    wires the owning layer's k/v value tensors into `ins` as they are, so a mismatch is itself a
    wiring mistake — letting it through would write different tensors into one slot.
    """
    slot = slots.get(name)
    if slot is None:
        slots[name] = candidate
        return
    if slot.shape != candidate.shape:
        raise StatesFormError(
            f"{where}: slot '{name}' derived shape differs between readers"
            f" ({slot.shape} / {candidate.shape})"
        )
    if slot.window != candidate.window:
        raise StatesFormError(
            f"{where}: slot '{name}' window differs between readers"
            f" ({slot.window} / {candidate.window})"
        )
    if slot.source != candidate.source:
        raise StatesFormError(
            f"{where}: the value written to slot '{name}' differs between readers"
            f" ('{slot.source}' / '{candidate.source}')"
            " — a KV-sharing layer wires the owning layer's k/v value tensors into ins as they are"
        )
    slot.last_reader = max(slot.last_reader, candidate.last_reader)


def _with_appends(nodes: Sequence[IrNode], slots: Mapping[str, _Slot]) -> list[IrNode]:
    """Inserts `state_append` **right after the last reader of the slot** (① of ADR 0067 decision
    5b).

    When several land at the same position, they are ordered by **slot registration order** (spec
    order × k → v) — the nodes array order is itself the contract for the order of state effects, so
    the choice is not left to the whims of a dict.
    """
    pending: dict[int, list[IrNode]] = {}
    for name, slot in slots.items():
        pending.setdefault(slot.last_reader, []).append(_append_node(name, slot))
    out: list[IrNode] = []
    for index, node in enumerate(nodes):
        out.append(node)
        out.extend(pending.get(index, ()))
    return out


def _append_node(name: str, slot: _Slot) -> IrNode:
    """The `state_append` for one slot (an effect op with zero outputs — ADR 0067 decision 5).

    MUST: declare the same `window` as the readers on a sliding slot. The logical col → physical row
    mapping MUST use the same formula for reads and writes (ADR 0067 decision 4), and dropping it on
    one side would be a silent misread (a mismatch in the declaration is rejected by
    verify._assert_state_order).
    """
    return IrNode(
        op=STATE_APPEND_OP,
        ins=[slot.source],
        outs=[],
        attrs={} if slot.window is None else {"window": slot.window},
        states={"slot": name},
    )


def _symbols(
    graph: IrGraph, plan: StatesPlan | ExternalStatesPlan, slots: Mapping[str, _Slot]
) -> list[str]:
    """The capacity symbol is added to `symbols` **only when some slot uses it**.

    MUST: a collision fails loudly. A collision with an existing symbol is the form "a symbol bound
    from the inputs is also used as the capacity" (value shapes would resolve with the capacity at
    context creation — the split of ADR 0066 addendum 7 breaks), and a collision with a value name
    is a mix-up that passed a tensor name into the symbol field.
    """
    symbol = plan.capacity_symbol
    if not any(slot.shape[2] == symbol for slot in slots.values()):
        return list(graph.symbols)
    if symbol in graph.symbols:
        raise StatesFormError(
            f"Capacity symbol '{symbol}' is already in graph.symbols"
            " (it would double the binding from the inputs)"
        )
    if symbol in graph.values or any(spec.name == symbol for spec in graph.inputs):
        raise StatesFormError(
            f"Capacity symbol '{symbol}' has the same name as a value"
            " (was a value name passed into the symbol field?)"
        )
    return [*graph.symbols, symbol]


def _prune(
    graph: IrGraph,
    nodes: Sequence[IrNode],
    slots: Mapping[str, _Slot],
    symbols: list[str],
    *,
    external: bool = False,
    droppable_inputs: Collection[str] = (),
) -> IrGraph:
    """Drops the nodes, initializers and `values` declarations not reachable from live values.

    The roots are `outputs` and **nodes without outputs** (effect ops that define no value =
    `state_append` — ADR 0067 decision 5). Without the effect ops as roots, reachability over data
    edges alone would remove them entirely. The nodes are in topological order
    (verify._check_definitions rejects forward references), so one reverse pass settles liveness,
    and the **relative order** of the remaining nodes is kept as is.

    MUST: `inputs` that became unreachable fail loudly. A surgery that orphans an input is a wiring
    mistake (an export that builds the mask from a graph input, for example); letting it through
    gives a graph where "the caller passes values as before but no node reads them" — it passes both
    the checks and execution, so nobody notices.

    `droppable_inputs` are **inputs dropped on purpose** (the k / v placeholders of the external
    surgery — ADR 0096 stage 2). They are exempt from the orphan rejection and are also removed
    from the `inputs` declaration itself. MUST: a form where someone still reads them after the drop
    fails loudly — removing only the declaration would leave "a node that reads an undefined name".
    """
    live_values = set(graph.outputs)
    live = [False] * len(nodes)
    for index in reversed(range(len(nodes))):
        node = nodes[index]
        if node.outs and not any(out in live_values for out in node.outs):
            continue
        live[index] = True
        live_values.update(node.ins)
    kept = [node for index, node in enumerate(nodes) if live[index]]

    dropped = set(droppable_inputs)
    still_read = sorted(name for name in dropped if name in live_values)
    if still_read:
        raise StatesFormError(
            f"Inputs {still_read} that should be dropped are still read by some node"
            " after the surgery"
            " (removing only the declaration would leave an undefined reference)"
        )
    orphans = [
        spec.name
        for spec in graph.inputs
        if spec.name not in live_values and spec.name not in dropped
    ]
    if orphans:
        raise StatesFormError(
            f"The surgery made inputs {orphans} unreachable"
            " (a form that builds the mask from an input cannot be lowered to the states form)"
        )

    defined = {out for node in kept for out in node.outs}
    initializers = {
        name: initializer for name, initializer in graph.initializers.items() if name in live_values
    }
    inputs = [spec for spec in graph.inputs if spec.name not in dropped]
    # MUST: re-derive the symbols from the remaining binding points (the dimension positions of
    # the input shapes and the states shapes — ADR 0066 addendum 7). Carrying over a symbol that
    # only the dropped K / V bound would make the surgery produce a graph the parser rejects with
    # "cannot be bound". The declaration order stays as in the original.
    bound = {
        parse_dim(dim).sym
        for shape in [*(spec.shape for spec in inputs), *(slot.shape for slot in slots.values())]
        for dim in shape
        if isinstance(dim, str)
    }
    return IrGraph(
        symbols=[symbol for symbol in symbols if symbol in bound],
        inputs=inputs,
        outputs=list(graph.outputs),
        initializers=initializers,
        values={
            name: value
            for name, value in graph.values.items()
            if name in defined or name in initializers
        },
        states={
            name: IrState(dtype=_SLOT_DTYPE, shape=slot.shape, external=external)
            for name, slot in slots.items()
        },
        nodes=kept,
    )
