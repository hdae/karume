"""Derivation of the device limits a quant requires (the optional `requiredLimits` of the
manifest) — the pure-function side.

The demand of the weight tensors (the largest byte count of one resident buffer) comes from **the
container merge**, so `karume.dist.component_demand_bytes` owns it; this module owns only the size
of the state slots and the mapping demand → `requiredLimits`.


The reader is the hub — with a declaration it can report "this does not run on this device"
**before the download** (a quant without the declaration fails only at `createSession`, after
downloading GBs). The source of truth for the spelling is `REQUIRED_LIMIT_NAMES` in
`packages/hub/src/manifest.ts` (one-to-one with the runtime's `REQUIRED_LIMIT_KEYS`), and of those
this module bakes **only the two size limits** ({@link WEBGPU_DEFAULT_LIMITS}):

- `maxStorageBufferBindingSize` — the limit for passing one storage buffer as a binding
- `maxBufferSize` — the allocation limit of one buffer

The workgroup limits (`maxComputeWorkgroupStorageSize` / `maxComputeInvocationsPerWorkgroup` /
`maxComputeWorkgroupSize*`) are **not baked**: the required value is decided by the kernel design
(tile shape, the choice of `@workgroup_size`, row blocking), not by the bytes of the distribution,
and even for the same assets the need changes with run-time knobs (opt-in numerics variants).
Declaring them as export-time constants would make "a distribution that keeps claiming the lower
bound of an old kernel design", double bookkeeping where the declaration and the implementation
move independently.

MUST: the declaration guarantees only that the **resident part** (stored tensors and state slots)
fits within the defaults; it **does not cover the intermediate tensors (node outputs)** during
execution (ruling of 2026-09-04). "No field = runs on a default-spec device" is a claim about the
resident part, not a declaration that extends to intermediates. The actual demand after fusion
depends on the limits the device actually granted (row blocking sets its step by the allowed binding
width), so it is in principle not determined by the distribution, and baking the largest node output
**before** fusion would wrongly reject, before the download, "devices that run with row blocks" (a
declaration that cannot be met although execution can — the worst outcome). Forms whose
intermediates exceed the limits are caught by the run-time check at Session construction.

MAY (re-graded 2026-10-18 — ADR 0089 decision 3): the demand is taken on a **payload basis** (the
largest payload length of the stored tensors) — the size of weights that cannot stay resident and
are decoded to f32 for allocation (`expanded` of the runtime's `planWeightResidency` — numel × 4
after decoding f16 / i8 / i4) is not included. Whether a weight is decoded is decided by the graph
alone and does not depend on the device, so this is not an exclusion to avoid false rejection but a
**limitation** from not keeping a mirror of the eligibility check on the Python side (ADR 0089
decision 3). "No field = runs on a default-spec device" therefore makes no claim about expanded
weights — a distribution whose decoded size exceeds the guaranteed defaults fails, with no field
baked, at the limit check of Session construction (`assertWeightsWithinLimits`). As of 2026-09-24
the distributed assets have no expanded weights.

MUST: bake **only the fields above the guaranteed defaults**. This keeps the semantics "no
`requiredLimits` = runs on a default-spec device"; writing values within the defaults would add to
every distribution fields that "are declared but constrain nothing" (the boundary — exactly the
default value — is not baked either).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from karume.dims import MAX_SAFE_INT, DimError, DimExpr, eval_dim, try_parse_dim
from karume.ops import integral_float_to_int


class LimitsError(ValueError):
    """requiredLimits cannot be derived (a missing symbol binding, a broken declaration)."""


#: The **guaranteed defaults** of the WebGPU spec (the defaults of the supported limits — the
#: lower bound obtained without requesting anything from the adapter). Only demand above these
#: becomes `requiredLimits`.
#:
#: MUST: the order is the manifest key order as is (fixed in dictionary order — the field order does
#: not move when re-baked).
WEBGPU_DEFAULT_LIMITS: Mapping[str, int] = {
    "maxBufferSize": 268_435_456,
    "maxStorageBufferBindingSize": 134_217_728,
}

#: state slot dtype → bytes per element. The source of truth for the vocabulary is
#: `karume.verify.STATE_DTYPES` (f16 is only reserved; only f32 has an execution path) — this is a
#: table for counting sizes, so it holds **only the dtypes it knows** rather than a copy of the
#: vocabulary, and an unknown one fails loudly.
STATE_DTYPE_BYTES: Mapping[str, int] = {"f32": 4}

#: The `pipelineConfig` field that gives the bound value of the symbolic dimension (the KV
#: capacity) of the state shapes.
#:
#: MUST: read **the largest capacity the distribution allows** (`maxPosition`), not the default
#: capacity (`capacity`). The capacity is a run-time knob (the host chooses it at
#: `createGenerationContext`), so baking with the default gives the worst outcome: "the moment a
#: capacity larger than the default is chosen, `createSession` fails on a device that meets the
#: declaration". Baking with the upper bound always errs on the sufficient side.
STATE_CAPACITY_KEY = "maxPosition"


def required_limits(demand: int) -> dict[str, int]:
    """The largest byte count of one resident buffer → the `requiredLimits` to bake (only fields
    above the default, strict comparison).

    The binding demand and the buffer demand are the same value because weight tensors and state
    slots alike are placed as **1 buffer = 1 binding** (there is no split-and-bundle form). Because
    the defaults differ, in the band where the demand is 128MiB to 256MiB only
    `maxStorageBufferBindingSize` is baked — `maxBufferSize` then falls back to the spec default
    256MiB, so `maxStorageBufferBindingSize ≤ maxBufferSize` (a precondition of requestDevice) holds
    automatically.
    """
    if demand < 0:
        raise LimitsError(f"Demand {demand} bytes is negative")
    return {name: demand for name, default in WEBGPU_DEFAULT_LIMITS.items() if demand > default}


def state_bindings(
    states: Mapping[str, Any], pipeline_config: Mapping[str, Any], where: str
) -> dict[str, int]:
    """Symbolic dimensions of the state shapes → bound values (taken from the position-limit field
    of `pipelineConfig`).

    The binding point is `createGenerationContext(spec.bindings)` (ADR 0066 addendum 7), and the
    capacity passed there is a run-time knob the host chooses. Its **upper bound**, which the
    distribution carries, is the {@link STATE_CAPACITY_KEY} field of `pipelineConfig`, and
    requiredLimits is baked with that upper bound (the field's docstring).

    MUST: fail loudly when a binding cannot be obtained — silently leaving state out of the count
    bakes "a small value that looked only at the weights", the worst outcome of **a declaration that
    is not enough** (without a declaration it would at least fail at run time). Two or more symbols
    are treated the same way: which one is the capacity symbol cannot be decided from the
    distribution (a states-form shape has one symbol, the capacity — ADR 0066 decision 2).
    """
    found: set[str] = set()
    for slot in states.values():
        for dim in _slot_shape(slot, where):
            if isinstance(dim, str):
                found.add(_parse_symbol(dim, where).sym)
    symbols = sorted(found)
    if not symbols:
        return {}
    if len(symbols) > 1:
        raise LimitsError(
            f"{where}: state shapes have several symbols {symbols}"
            f" — only the one capacity set by pipelineConfig['{STATE_CAPACITY_KEY}'] can be bound"
        )
    # The same accepted set as the TS reader of the field (`Number.isSafeInteger`, at least 1):
    # an integral float (`640.0`) binds as the integer (the reason is in
    # {@link karume.ops.integral_float_to_int}).
    capacity = integral_float_to_int(pipeline_config.get(STATE_CAPACITY_KEY))
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or not 1 <= capacity <= MAX_SAFE_INT
    ):
        raise LimitsError(
            f"{where}: state shape symbol '{symbols[0]}' is bound by"
            f" pipelineConfig['{STATE_CAPACITY_KEY}'], which is not a positive integer"
            f" ({capacity!r})"
        )
    return {symbols[0]: capacity}


def max_state_slot_bytes(
    graph: Mapping[str, Any], pipeline_config: Mapping[str, Any], where: str
) -> int:
    """The byte count of **the largest single** state slot of the graph (ADR 0066 decision 2; 0
    without states).

    A slot is a resident buffer that `createGenerationContext` allocates whole for the capacity
    (1 slot = 1 buffer = 1 binding), so the size with the symbol bound to **the largest capacity the
    distribution allows** is the demand as is. In families with a large KV capacity **the state is
    larger than the largest tensor** (both stay one buffer — weights can be distributed across
    parts, but they become one again on the GPU side).
    """
    states = graph.get("states")
    if states is None:
        return 0
    if not isinstance(states, dict):
        raise LimitsError(f"{where}: graph.states is not an object")
    bindings = state_bindings(states, pipeline_config, where)
    largest = 0
    for name, slot in states.items():
        at = f"{where}: state slot '{name}'"
        dtype = slot.get("dtype") if isinstance(slot, Mapping) else None
        item_bytes = STATE_DTYPE_BYTES.get(dtype) if isinstance(dtype, str) else None
        if item_bytes is None:
            raise LimitsError(
                f"{at} dtype {dtype!r} cannot be sized (known: {sorted(STATE_DTYPE_BYTES)})"
            )
        elements = 1
        for dim in _slot_shape(slot, where):
            elements *= _dim_size(dim, bindings, at)
        largest = max(largest, elements * item_bytes)
    return largest


def _slot_shape(slot: Any, where: str) -> list[Any]:
    """The `shape` of a state slot declaration (looks at the form only — the range checks are in
    `karume.verify`)."""
    shape = slot.get("shape") if isinstance(slot, Mapping) else None
    if not isinstance(shape, list) or not shape:
        raise LimitsError(f"{where}: state slot shape is not a non-empty array ({shape!r})")
    return shape


def _parse_symbol(dim: str, where: str) -> DimExpr:
    """Decomposes a symbolic dimension `coeff·sym+offset` (non-canonical fails loudly)."""
    expr = try_parse_dim(dim)
    if expr is None:
        raise LimitsError(
            f"{where}: state shape dimension expression '{dim}'"
            " does not match the canonical grammar"
        )
    return expr


def _dim_size(dim: Any, bindings: Mapping[str, int], where: str) -> int:
    """Makes one dimension concrete (a number as is, a symbol through the binding table)."""
    if isinstance(dim, bool) or not isinstance(dim, int | str):
        raise LimitsError(f"{where}: shape dimension {dim!r} is neither an integer nor a symbol")
    if isinstance(dim, int):
        if dim < 1:
            raise LimitsError(f"{where}: shape dimension {dim} is not a positive integer")
        return dim
    try:
        return eval_dim(_parse_symbol(dim, where), bindings)
    except DimError as cause:
        raise LimitsError(f"{where}: dimension '{dim}' cannot be bound: {cause}") from cause
