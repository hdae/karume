"""Graph surgery that declares a **band** on exported attention nodes in place of their mask (ADR
0155).

A recipe exports the model as torch runs it: a sliding-window attention takes its band as an
additive mask (the fourth input — folded by ADR 0010 into a `[1, 1, Tmax, Tmax]` constant cut down
by `sym_prefix_slice`). This module then rewrites the attention nodes the recipe **names** into the
band form: the mask input is dropped and `attrs.band` is written, so the runtime checks the band
inside the score kernel from positions and no mask tensor exists. It is a model-independent
mechanism, so it lives in the generic core (the precedent is `karume.states.to_states_form`).

The surgery is two operations:

1. **Declaration** — on each named attention, drop the fourth input and write `attrs.band`.
2. **Pruning** — drop the nodes and declarations no live value reaches any more. The folded
   constant and its `sym_prefix_slice` die here once no attention reads the mask.

MUST: the band is **declared by the recipe**, never read off the mask. The surgery does not look at
the mask's values or at how it was built (a folded constant cannot be told from other constants
without pattern-matching it, and a matcher that silently stops matching becomes a wrong value).
That the declared band is the mask the model applies is the recipe's claim, backed by the recipe's
own checks of the exported graph against upstream.
MUST: a **pure function** (does not mutate the input graph — the same discipline as
`to_states_form`).
MUST: a mask that **depends on** a graph input (a run-time mask) fails loudly, whether or not other
nodes also read that input: before pruning, the surgery walks back from each dropped mask through
its producers (data edges only — it reads no values and matches no patterns), and a walk that
reaches a graph input is refused. A band cannot carry what a run-time value decides.

NOTE: the correctness of the result is checked by verify (the contract table refuses a band beside
a mask, on the states form, or out of range); this module only assembles a form that passes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from karume.dims import MAX_SAFE_INT
from karume.ir import IrGraph, IrNode
from karume.ops import ATTENTION_OP, integral_float_to_int


class BandFormError(ValueError):
    """The surgery into the band form fails (a mixed-up target, a wiring mistake, a wrong spec)."""


@dataclass(frozen=True)
class BandAttentionSpec:
    """One attention to rewrite into the band form.

    `output` is the target node's `outs[0]` (with SSA single assignment, the node's unique
    identifier). `band` is the declared width: key column `n` is visible to query row `m` iff
    `|m − n| ≤ band` (an integer ≥ 0 — the range of the contract's `band` attr).
    """

    output: str
    band: int


def to_band_form(graph: IrGraph, attentions: Sequence[BandAttentionSpec]) -> IrGraph:
    """Returns a **new graph** with the named attentions rewritten into the band form.

    The input `graph` is only read (MUST — module docstring). The returned graph has every named
    attention with `ins = [q, k, v]` and `attrs.band`, and every node and declaration no live value
    reaches is pruned (the folded constant and its `sym_prefix_slice` once no attention reads them).
    """
    nodes = list(graph.nodes)
    producers = {out: node for node in graph.nodes for out in node.outs}
    input_names = [spec.name for spec in graph.inputs]
    seen: set[str] = set()
    for spec in attentions:
        where = f"attention '{spec.output}'"
        if spec.output in seen:
            raise BandFormError(f"{where}: named twice")
        seen.add(spec.output)
        index = _target_index(nodes, spec.output, where)
        node = nodes[index]
        _assert_convertible(node, where)
        _assert_mask_reads_no_input(node.ins[3], producers, input_names, where)
        nodes[index] = IrNode(
            op=node.op,
            # Drop the mask (fourth input), leaving the three q / k / v.
            ins=list(node.ins[:3]),
            outs=list(node.outs),
            attrs={**node.attrs, "band": _band(spec.band, where)},
            states=dict(node.states),
        )
    return _prune(graph, nodes)


def _target_index(nodes: Sequence[IrNode], output: str, where: str) -> int:
    """The index of the node whose `outs[0]` is `output`."""
    for index, node in enumerate(nodes):
        if node.outs and node.outs[0] == output:
            return index
    raise BandFormError(f"{where}: no node in the graph defines this value")


def _assert_convertible(node: IrNode, where: str) -> None:
    """The target is exactly a **plain attention that takes a mask** (four inputs, no states field,
    no band yet).

    MUST: check every point. A different op is a mixed-up spec; a states field is the states form
    (its columns follow the causal rule and `window`); three `ins` mean an export without a mask
    (the declaration replaces a mask the model applies, so there must be one); a band already
    declared is a double surgery.
    """
    if node.op != ATTENTION_OP:
        raise BandFormError(
            f"{where}: op is '{node.op}' (only '{ATTENTION_OP}' can go through the surgery)"
        )
    if node.states:
        raise BandFormError(f"{where}: in the states form ({node.states}) — it takes no band")
    if "band" in node.attrs:
        raise BandFormError(f"{where}: already declares band {node.attrs['band']!r}")
    if len(node.ins) != 4:
        raise BandFormError(
            f"{where}: {len(node.ins)} inputs (only the four including the mask = q / k / v / mask"
            " — the band replaces the mask the export traced)"
        )


def _assert_mask_reads_no_input(
    mask: str, producers: dict[str, IrNode], input_names: Sequence[str], where: str
) -> None:
    """The dropped mask depends on no graph input (MUST — module docstring).

    Walks back from `mask` through the producers of the source graph. An initializer has no
    producer and ends the walk; a graph input reached by the walk is a run-time mask. The check
    does not depend on whether other nodes read the input, so a padding mask shared with an
    attention that is not named, or derived from the input through other nodes, is refused too.
    """
    reached: set[str] = set()
    visited: set[str] = set()
    pending = [mask]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in input_names:
            reached.add(name)
        producer = producers.get(name)
        if producer is not None:
            pending.extend(producer.ins)
    if reached:
        names = [name for name in input_names if name in reached]
        raise BandFormError(
            f"{where}: its mask depends on inputs {names}"
            " (a run-time mask, which a band cannot replace)"
        )


def _band(band: int, where: str) -> int:
    """The declared band width.

    MUST: reject bool (Python's bool derives from int, so `True` would pass as a band of 1).
    MUST: accept the same set as the runtime's `band` attr (`Number.isSafeInteger`, at least 0): an
    integral float (`512.0`) is written as the integer, and a width beyond 2^53−1 is rejected.
    """
    value = integral_float_to_int(band)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_INT:
        raise BandFormError(f"{where}: band {band!r} is not a non-negative integer")
    return value


def _prune(graph: IrGraph, nodes: Sequence[IrNode]) -> IrGraph:
    """Drops the nodes, initializers and `values` declarations no live value reaches.

    The roots are `outputs` and **nodes without outputs** (effect ops — `state_append`). The nodes
    are in topological order (verify rejects forward references), so one reverse pass settles
    liveness, and the **relative order** of the remaining nodes is kept.

    MUST: an input no node reads after the surgery fails loudly (a generic wiring check: the result
    would hold "an input the caller passes and no node reads"; a run-time mask is refused before
    pruning, by its own check).
    MUST: symbols and state slots are kept as they are — no input is dropped, so every binding
    point stays.
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
    orphans = [spec.name for spec in graph.inputs if spec.name not in live_values]
    if orphans:
        raise BandFormError(f"Inputs {orphans} are read by no node after the surgery")
    defined = {out for node in kept for out in node.outs}
    initializers = {
        name: initializer for name, initializer in graph.initializers.items() if name in live_values
    }
    return IrGraph(
        symbols=list(graph.symbols),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        initializers=initializers,
        values={
            name: value
            for name, value in graph.values.items()
            if name in defined or name in initializers
        },
        states=dict(graph.states),
        nodes=kept,
    )
