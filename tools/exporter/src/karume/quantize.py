"""Weight fake-quant (rounding to values representable in the storage dtype) — ADR 0006 / 0018 /
0019 / 0069.

The semantics are "only the storage is quantized; compute is f32". The exporter rounds the weights
**first** and then captures the reference and the goldens, so the GPU side ("compressed storage +
f32 compute") and the torch side ("f32 compute on the rounded weights") compute **the same
numbers**, which separates quantization error from implementation error.

MUST: rounding is applied **before the reference and the goldens are captured** (ADR 0006). Applied
after, only the reference would be computed with the original weights, the E2E difference would be
"quantization error + implementation error" combined, and the tolerance would lose its meaning (and
since that only ever pushes toward loosening the tolerance, the check stays green while it loses
its power to detect).

MUST: rounding is applied to the **effective weights** — processing that rewrites weights, such as
baking in LoRA, removing weight_norm or the temporal slice of CausalConv3d, is done before rounding.
For f16 the reason is that "the composed value is no longer representable in f16"; for i8 it is
**even stricter**: when a discarded element affects amax, the per-channel scale itself shifts (the
values change in every element).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn


class QuantizeError(ValueError):
    """Rounding could not land on representable values / a quantization target is not what was
    expected."""


@dataclass(frozen=True)
class RoundReport:
    """Number of tensors and elements rounded (a count so that "f16 requested but 0 tensors" is
    never silent)."""

    parameters: int
    buffers: int
    elements: int

    def describe(self) -> str:
        return f"parameters {self.parameters} / buffers {self.buffers} / {self.elements:,} elements"


def round_weights_to_f16(model: torch.nn.Module) -> RoundReport:
    """Round parameters and f32 buffers to values representable in f16 (compute stays f32).

    The targets are "parameters + f32 buffers" because the initializers that can be stored as f16
    come only from there (convolution constants and folded constants are part of the graph
    semantics, and the golden is computed with their original values, so they are stored as f32
    — the eligibility check is on the emit.py side).

    MUST: fail loudly when a finite value saturates to non-finite. `Tensor.to(float16)` silently
    turns elements beyond the f16 range (|x| ≤ 65504) into ±inf, so without the check "a model with
    inf in its weights" and "a reference computed with the same inf" **agree**, and the output
    breaks while E2E stays green.
    """
    counts = {"parameter": 0, "buffer": 0}
    elements = 0
    with torch.no_grad():
        # named_parameters / named_buffers yield duplicates (shared weights) only once by default.
        # f16 rounding is idempotent, so hitting one twice would not change the values, but the
        # count is the number of actual tensors.
        targets = [("parameter", name, t) for name, t in model.named_parameters()]
        targets += [("buffer", name, t) for name, t in model.named_buffers()]
        for kind, name, tensor in targets:
            if tensor.dtype is not torch.float32:
                continue
            rounded = tensor.to(torch.float16).to(torch.float32)
            # Per element: elements that were non-finite to begin with are not the rounding's
            # fault, so they are not counted.
            saturated = torch.isfinite(tensor) & ~torch.isfinite(rounded)
            if saturated.any():
                overflow = int(saturated.sum())
                raise QuantizeError(
                    f"'{name}': rounding to f16 saturated {overflow} finite values to non-finite"
                    f" (the f16 range is |x| ≤ 65504; actual max |x| = "
                    f"{float(tensor.abs().max()):.4g})"
                )
            tensor.copy_(rounded)
            counts[kind] += 1
            elements += tensor.numel()
    return RoundReport(parameters=counts["parameter"], buffers=counts["buffer"], elements=elements)


def round_weights_to_bf16(model: torch.nn.Module) -> RoundReport:
    """Round parameters and the f32 buffers **the checkpoint carries** to values representable in
    bf16 (compute stays f32).

    The counterpart of {@link round_weights_to_f16} (same idempotence and counting). The rounding
    is the RNE of `Tensor.to(bfloat16)` (bf16 = the upper 16 bits of f32).

    MUST: non-persistent buffers (`persistent=False` — absent from `state_dict()`) are not rounded.
    They are not in the checkpoint but computed in f32 at init (such as the RoPE `inv_freq`), and
    upstream inference uses them in f32 as well. A bf16 asset is "the upstream bits + the same f32
    computed constants as upstream"; rounding them would put an error upstream does not have into
    the golden (observed: 1.4e-4 in Anima's text_conditioner — ADR 0131 decision 2). The f16
    version keeps rounding every buffer as before, so the existing series and goldens do not move.

    MUST: fail loudly when a finite value saturates to non-finite. bf16 has the same exponent range
    as f32 but a 7-bit mantissa, so its largest finite value is (2 − 2⁻⁷)·2¹²⁷ ≈ 3.39e38, and the
    top of the f32 range above it (|x| ≥ (2 − 2⁻⁸)·2¹²⁷) silently rounds up to ±inf under RNE —
    without the check, as for f16, "a model with inf in its weights" and "a reference computed
    with the same inf" agree.
    """
    counts = {"parameter": 0, "buffer": 0}
    elements = 0
    with torch.no_grad():
        # named_parameters / named_buffers yield duplicates (shared weights) only once by default.
        # bf16 rounding is idempotent, so hitting one twice would not change the values, but the
        # count is the number of actual tensors.
        # state_dict keys = the tensors the checkpoint carries (non-persistent buffers are absent).
        carried = model.state_dict().keys()
        targets = [("parameter", name, t) for name, t in model.named_parameters()]
        targets += [("buffer", name, t) for name, t in model.named_buffers() if name in carried]
        for kind, name, tensor in targets:
            if tensor.dtype is not torch.float32:
                continue
            rounded = tensor.to(torch.bfloat16).to(torch.float32)
            # Per element: elements that were non-finite to begin with are not the rounding's
            # fault, so they are not counted.
            saturated = torch.isfinite(tensor) & ~torch.isfinite(rounded)
            if saturated.any():
                overflow = int(saturated.sum())
                raise QuantizeError(
                    f"'{name}': rounding to bf16 saturated {overflow} finite values to non-finite"
                    f" (the largest finite bf16 value is ≈3.39e38; actual max |x| = "
                    f"{float(tensor.abs().max()):.4g})"
                )
            tensor.copy_(rounded)
            counts[kind] += 1
            elements += tensor.numel()
    return RoundReport(parameters=counts["parameter"], buffers=counts["buffer"], elements=elements)


# ---- i8 (per-channel symmetric) — ADR 0019 ---------------------------------

#: One side of the quantization range (**−128 is not used**). Closing it at ±127 puts the
#: largest-magnitude element on `q = ±127`, so the scale re-derived from the rounded weights is
#: `fl(fl(127·s)/127)`. A reachable scale `s = fl(amax/127)` is a **fixed point** of this map, so
#: fake-quant is **idempotent** (re-applying it is bit-identical — pinned by
#: `tests/test_quantize.py`). Reason: amax is an f32 that rounds to s as the nearest value of
#: `127·s`, so `fl(127·s)` is at least as close to `127·s` as amax is, and dividing back by 127
#: lands on s. Exact restoration of the amax element itself (`fl(127·s) == amax`) can be off by
#: 1 ulp in f32 — idempotence does not rest on it (ADR 0069 addendum 3). Allowing −128 causes
#: requantization in which only the scale moves.
INT8_MAX = 127

#: The **channel axis** of the per-channel scale (module type → axis index of the weight tensor —
#: ADR 0019). It is the output-channel axis; only `ConvTranspose1d` has the transposed weight
#: layout `[Cin, Cout, K]`, so its axis is 1.
#:
#: MUST: returns **the same axis** as `ops.WEIGHT_CHANNEL_AXES` (the same table keyed by op name).
#: Moving only one of them makes "the axis of the emitted scale" and "the axis the kernel reads"
#: disagree, and the values silently break. The correspondence is pinned by the cross-check test
#: in `tests/test_quantize.py` (with a module type → op name table).
QUANT_CHANNEL_AXES: Mapping[type[nn.Module], int] = MappingProxyType(
    {
        nn.Linear: 0,
        nn.Conv1d: 0,
        nn.Conv2d: 0,
        nn.Conv3d: 0,
        nn.ConvTranspose1d: 1,
        nn.Embedding: 0,
    }
)

#: The module types that can be quantization targets (derived from the keys of
#: {@link QUANT_CHANNEL_AXES}). The spelling for passing "the same 6 types as i8" to `op_types` —
#: not a separate list copied from the table (if they moved independently, "the types whose axis
#: can be looked up" and "the types that can be targets" would silently diverge).
QUANT_MODULE_TYPES: tuple[type[nn.Module], ...] = tuple(QUANT_CHANNEL_AXES)


@dataclass(frozen=True)
class Int8Report:
    """The scale ledger and counts of the quantized weights.

    The keys of `scales` are **in-model FQNs** (`<module>.weight`), the same space as the FQNs
    `convert.py` uses for an initializer's `tensor` (the key of the container's binding table). No
    matching by `id(tensor)` (ADR 0006) — parameter identity breaks easily under normalisation and
    folding, and when it breaks it just silently drops to "0 targets".
    """

    #: FQN → scale (keepdim shape of the same rank as the weight, F32).
    scales: Mapping[str, torch.Tensor]
    modules: int
    elements: int

    def describe(self) -> str:
        return f"modules {self.modules} / {self.elements:,} elements"


def _channel_axis(module: nn.Module) -> int | None:
    """The module's per-channel axis (None when it is not a target).

    An exact match is checked first, then isinstance — real models sometimes use thin subclasses
    of `nn.Linear`, and an exact type match alone would **silently drop them from the targets**
    (eligibility on the emit side is decided by the consuming op, so a dropped weight fails loudly
    as "eligible but has no scale", but whatever can be caught here is caught). A type that matches
    two or more entries has no single axis, so it fails.
    """
    exact = QUANT_CHANNEL_AXES.get(type(module))
    if exact is not None:
        return exact
    axes = {axis for cls, axis in QUANT_CHANNEL_AXES.items() if isinstance(module, cls)}
    if len(axes) > 1:
        raise QuantizeError(
            f"{type(module).__name__}: the per-channel axis is not unique"
            f" (candidates {sorted(axes)})"
        )
    return next(iter(axes), None)


def channel_scale(weight: torch.Tensor, axis: int) -> torch.Tensor:
    """Return `scale = clamp(amax / 127, f32 tiny)` in the keepdim shape of the same rank as weight.

    The lower clamp is needed for all-zero channels (`amax == 0`) — dividing as is gives NaN. After
    the clamp they land on `q = 0`, and `q·scale = 0` restores the original value exactly.
    """
    reduced = [dim for dim in range(weight.dim()) if dim != axis]
    amax = weight.abs().amax(dim=reduced, keepdim=True) if reduced else weight.abs()
    return torch.clamp(amax / INT8_MAX, min=torch.finfo(torch.float32).tiny)


def quantize_to_int8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return `q = clamp(round(w / scale), ±127)` as int8 (the scale is **not recomputed**).

    MUST: the caller passes the scale fake-quant used **as is**. Re-deriving it from amax here can
    move `scale` by 1 ulp through f32 rounding, and the stored values would no longer match the
    weights the golden was captured with (ADR 0019).
    """
    return torch.round(weight / scale).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)


def fake_quant_int8(model: nn.Module, include: Callable[[str], bool] | None = None) -> Int8Report:
    """Round the weights of modules with a weight slot to values representable in per-channel
    symmetric int8.

    The targets are only the `weight` of modules whose type is in {@link QUANT_CHANNEL_AXES} —
    neither biases nor norm-style weights are **touched** (bias always being f32 is the root fix
    of ADR 0006; the moment it became a quantization target, the prototype's demotion bug would
    return).

    `include` is a predicate on the **module FQN** (the name from `named_modules`; None = every
    target). It is the hook for splitting the targets in mixed storage (e.g. embedding in i8 and
    linear in i4). MUST: in mixed storage the i8 / i4 targets are split **exclusively** — passing
    a tied weight (lm_head = embed_tokens) through both rounds it twice, only the rounding of the
    side that ran later stays in the values, and the earlier scale ledger disagrees with the actual
    values.
    The default targets (include=None) include `nn.Conv3d` (ADR 0118 decision 1). When applying
    i8 to a model that lowers Conv3d to an op other than conv3d (using the weight on the host side,
    for example), narrow the targets with include — otherwise weights that never reach a conv3d
    weight slot are rounded too.

    MUST: the call order is "after the effective weights are settled, before the reference /
    goldens are captured" (module docstring).

    The returned scale ledger is passed to emit as is. When not a single weight could be quantized
    it fails loudly — so that "`--dtype i8` was requested but it effectively wrote f32" is never
    silent (ADR 0006; the same when `include` dropped every target).
    """
    scales: dict[str, torch.Tensor] = {}
    elements = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            # include is checked before _channel_axis — so that type ambiguity (several type
            # matches) of a non-target module does not fail.
            if include is not None and not include(name):
                continue
            axis = _channel_axis(module)
            if axis is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            fqn = f"{name}.weight" if name else "weight"
            if weight.dtype is not torch.float32:
                raise QuantizeError(
                    f"'{fqn}': dtype {weight.dtype} cannot be quantized (f32 semantics only)"
                )
            if axis >= weight.dim():
                raise QuantizeError(
                    f"'{fqn}': channel axis {axis} is not in a weight of rank {weight.dim()}"
                )
            scale = channel_scale(weight, axis)
            weight.copy_(quantize_to_int8(weight, scale).to(torch.float32) * scale)
            scales[fqn] = scale
            elements += weight.numel()
    if not scales:
        raise QuantizeError(
            "storage i8 was requested but there is no weight that can be quantized per channel"
            f" (target types: {', '.join(cls.__name__ for cls in QUANT_CHANNEL_AXES)})"
        )
    return Int8Report(scales=scales, modules=len(scales), elements=elements)


# ---- Target selection and the generalised "in axis" (shared by every rounding method) -------


def iter_quant_targets(
    model: nn.Module,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
    include: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor, int]]:
    """Enumerate the weights to quantize as `(FQN, weight, per-channel axis)` (whatever the
    rounding method).

    **The source of truth** for target selection — group quantization ({@link fake_quant_int4})
    and the measurement-only methods (`quant_methods`) both go through here. With two
    implementations, "the targets of the shipped weights" and "the targets whose quality was
    measured" would silently diverge, and the measured degradation and the produced asset would be
    different things.

    `op_types` is matched with `isinstance` (real models use thin subclasses of `nn.Linear` — a
    subclass can also be passed directly). The axis comes from {@link QUANT_CHANNEL_AXES}, and
    **types whose axis cannot be looked up are passed over** (if `op_types` misses entirely, the
    caller's "0 targets" diagnostic reports it — its wording differs per method, so it is not
    decided here).

    `include` is a predicate on the **module FQN** (None = every target). It is checked before the
    type for the same reason as in {@link fake_quant_int8} — so that type ambiguity of a non-target
    module does not fail.
    """
    for name, module in model.named_modules():
        if include is not None and not include(name):
            continue
        if not isinstance(module, op_types):
            continue
        axis = _channel_axis(module)
        if axis is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        fqn = f"{name}.weight" if name else "weight"
        if weight.dtype is not torch.float32:
            raise QuantizeError(
                f"'{fqn}': dtype {weight.dtype} cannot be quantized (f32 semantics only)"
            )
        if axis >= weight.dim():
            raise QuantizeError(
                f"'{fqn}': channel axis {axis} is not in a weight of rank {weight.dim()}"
            )
        yield fqn, weight, axis


def channel_rows(weight: torch.Tensor, axis: int) -> torch.Tensor:
    """Fold a weight into 2-D `[channel, in]` (the group axis = each op's generalised "in axis").

    - linear `[O,I]` / embedding `[V,D]` … identity (already in `[channel, in]` form)
    - conv1d `[Cout,Cin,K]` / conv2d `[Cout,Cin,Kh,Kw]` … the receptive field is flattened per
      output channel (`Cin·K` / `Cin·Kh·Kw`)
    - conv_transpose1d `[Cin,Cout,K]` … transposed layout, so axis 1 is moved to the front before
      flattening (the same orientation as the i8 per-channel axis — {@link QUANT_CHANNEL_AXES})

    NOTE: for a contiguous weight with axis 0 a **view** is returned (no copy), but
    `ConvTranspose1d` becomes non-contiguous through `movedim`, so `reshape` makes a **copy**.
    Writing back always goes through {@link restore_channel_rows} — writing into the flat form
    directly does not reach the original weight.
    """
    return weight.movedim(axis, 0).reshape(int(weight.shape[axis]), -1)


def restore_channel_rows(rows: torch.Tensor, weight: torch.Tensor, axis: int) -> torch.Tensor:
    """The inverse of {@link channel_rows} — returns `[channel, in]` to the shape of `weight`."""
    moved = list(weight.shape)
    moved.insert(0, moved.pop(axis))
    return rows.reshape(moved).movedim(0, axis)


def storage_rows(weight: torch.Tensor) -> torch.Tensor:
    """Rank-2 fold of the storage form (**first dimension = row, the rest flattened**) = the axis-0
    version of {@link channel_rows}.

    The group scale shape is decided by this fold whatever the rank (the rank independence of ADR
    0069 decision 3): `[weight.shape[0], (numel / weight.shape[0]) / group_size]`. For linear
    `[O,I]` and embedding `[V,D]` it is **the same shape** as "the weight's rank with only the last
    dimension replaced by the group count", and conv1d `[Cout,Cin,K]` → `[Cout, (Cin·K)/g]` is the
    only new shape (the phase J-5b supplement — ADR 0069 addendum 7).

    The row order is **exactly the weight's contiguous memory order** (the row-major bytes of
    `[O,Cin,K]` = the flat `[O, Cin·K]`), so the packed bytes are decided one way whatever the
    rank.

    MUST: a weight whose row axis is not first (`[Cin,Cout,K]` of `ConvTranspose1d`) is not passed
    through here — the shape passes, but the row order differs from the memory order. The i4
    eligible set (`emit.I4_WEIGHT_OPS`) not containing conv_transpose1d is that boundary, and emit
    fails loudly on an explicit request.
    """
    return channel_rows(weight, 0)


# ---- i4 (group symmetric along K) — ADR 0069 ---------------------------------

#: One side of the quantization range (**−8 is not used**). Closing it at ±7 puts the group's
#: amax element on `q = ±7`, and a reachable scale `s = fl(amax/7)` is a **fixed point** of
#: `fl(fl(7·s)/7)`, so fake-quant is **idempotent** (the 4-bit version of the `INT8_MAX`
#: argument — ADR 0069 decision 3, addendum 3). Exact restoration of the amax element can be off
#: by 1 ulp, so it is not relied on. Allowing −8 causes requantization in which only the scale
#: moves.
INT4_MAX = 7

#: Default group size (32, settled by the measurements of the Phase 0 sweep — ADR 0069 addendum
#: 1). It is a storage field, so it does not stop a size-first asset from choosing 64 / 128 (the
#: accepted set is a power of two and ≥ 16 — the range check is a storage-side rule held by
#: `verify.py`).
DEFAULT_GROUP_SIZE = 32


@dataclass(frozen=True)
class Int4Report:
    """The scale ledger and counts of weights rounded to group-symmetric int4 (the same container
    as `Int8Report`).

    The keys of `scales` are **in-model FQNs** (`<module>.weight`), as for i8. The values are
    group scales of the shape folded by {@link channel_rows} = F32 `[channel, group count]`.

    The shape follows the rank-2 rule of {@link storage_rows} whatever the rank (ADR 0069 decision
    3; a separate shape whose accepted set does not overlap the i8 keepdim broadcast shape):

    - linear `W[O,I]` → `[O, I/group_size]` / embedding `W[V,D]` → `[V, D/group_size]` …
      equivalent to "the weight's rank with only the last dimension replaced by the group count"
    - conv1d `W[Cout,Cin,K]` → `[Cout, (Cin·K)/group_size]` … rank 2 with the receptive field
      flattened (**shippable** — added to emit's eligible set in phase J-5b; `emit.I4_WEIGHT_OPS`)
    - conv2d / conv_transpose1d … the same rank 2, but not i4-eligible ops (measurement only)
    """

    #: FQN → scale (group shape `[channel, group count]`, F32).
    scales: Mapping[str, torch.Tensor]
    group_size: int
    modules: int
    elements: int

    def describe(self) -> str:
        return f"modules {self.modules} / {self.elements:,} elements / group {self.group_size}"


def grouped_view(weight: torch.Tensor, group_size: int, where: str) -> torch.Tensor:
    """Split the quantized axis (**last dimension = in axis**) into `[…, groups, group_size]`.

    For linear weights the last dimension is the in axis as is; other ops are folded into
    `[channel, in]` with {@link channel_rows} before being passed.

    MUST: a shape that does not divide fails loudly (ADR 0069 decision 2). Allowing a remainder
    group would make only the last group's scale cover a shorter range, row boundaries would drift
    from word boundaries, and flat-index decoding would silently produce other values — the
    storage design removes the alignment problem itself by the constraint of never making
    remainders.
    """
    in_axis = int(weight.shape[-1])
    if in_axis % group_size:
        raise QuantizeError(
            f"{where}: the quantized axis (last dimension) {in_axis} is not divisible by"
            f" group_size {group_size} (i4 makes no remainder group, MUST — ADR 0069 decision 2)"
        )
    return weight.reshape(*weight.shape[:-1], in_axis // group_size, group_size)


def group_size_of(weight: torch.Tensor, scale: torch.Tensor) -> int:
    """Derive the group size from a group-shaped scale (`[row, groups]` → `row length / groups`).

    The shape rule is the rank-2 form of {@link storage_rows} (ADR 0069 decision 3) — for rank-2
    weights (linear / embedding) it is equivalent to "the same rank with only the last dimension
    replaced by the group count", and a rank-3 conv1d `[Cout,Cin,K]` takes `[Cout, (Cin·K)/g]`.

    MUST: the group size comes **only from the scale passed in**. Taking it as a separate argument
    would let "the scale fake-quant used" and "the group size declared in storage" move
    independently, and a mismatch would be a silent wrong value because shape and type still agree
    (the same hole as the "no scale recomputation" of ADR 0069 decision 3). The storage-side
    declaration is derived from here too (`emit._plan_i4`).
    """
    if weight.dim() < 2 or scale.dim() != 2 or int(scale.shape[0]) != int(weight.shape[0]):
        raise QuantizeError(
            f"scale {list(scale.shape)} is not the group shape of weight {list(weight.shape)}"
            " (rank 2, rows = the weight's first dimension, last dimension = group count)"
        )
    groups = int(scale.shape[1])
    row_length = weight.numel() // int(weight.shape[0])
    if groups < 1 or row_length % groups:
        raise QuantizeError(
            f"the scale's group count {groups} does not divide the weight's row length {row_length}"
        )
    return row_length // groups


def group_scale(weight: torch.Tensor, group_size: int, where: str = "weight") -> torch.Tensor:
    """Return `scale = clamp(amax_group / 7, f32 tiny)` in group shape
    (`[row, row length/group_size]`).

    Rows are {@link storage_rows} (flattening all but the first dimension) — for rank-2 weights the
    last dimension itself; a rank-3 conv1d folds the receptive field `Cin·K` into one row. Dividing
    along the last dimension (K) would round groups that straddle receptive fields with different
    scales.

    The lower clamp is needed for all-zero groups (`amax == 0`) — dividing as is gives NaN. After
    the clamp they land on `q = 0`, and `q·scale = 0` restores the original value exactly (the
    4-bit version of the same text in `channel_scale`).
    """
    grouped = grouped_view(storage_rows(weight), group_size, where)
    amax = grouped.abs().amax(dim=-1)
    return torch.clamp(amax / INT4_MAX, min=torch.finfo(torch.float32).tiny)


def quantize_to_int4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return `q = clamp(round(w / scale), ±7)` in an int8 container, in the weight's logical
    shape.

    MUST: the caller passes the scale fake-quant used **as is** (same as `quantize_to_int8` —
    re-deriving it from amax here can move it by 1 ulp through f32 rounding). torch has no 4-bit
    container, so the values ride on int8. **Packing 2 elements into 1 byte is the storage side's
    job** (`emit.pack_int4`); this function does not know the pack order.

    Rank 3 and above (conv1d `[Cout,Cin,K]`) is folded into rows with {@link storage_rows} before
    being split into groups — dividing along the last dimension (K) would round groups that
    straddle receptive fields with different scales.
    """
    rows = storage_rows(weight)
    grouped = grouped_view(rows, group_size_of(weight, scale), "weight")
    quantized = torch.round(grouped / scale.unsqueeze(-1)).clamp_(-INT4_MAX, INT4_MAX)
    return quantized.reshape(weight.shape).to(torch.int8)


def dequantize_int4(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return `q·scale` as f32 in the weight's logical shape (the inverse of storage).

    The shared implementation that makes fake-quant (below) and emit's round-trip bit-identity
    check (ADR 0069 decision 4 ③) go through **the same path** — separate implementations would
    let "the formula used for rounding" and "the formula that verifies the storage" move
    independently.
    """
    rows = storage_rows(quantized.to(torch.float32))
    grouped = grouped_view(rows, group_size_of(quantized, scale), "quantized values")
    return (grouped * scale.unsqueeze(-1)).reshape(quantized.shape)


def fake_quant_int4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> Int4Report:
    """Round weights to values representable in group-symmetric int4 along the "in axis" (ADR
    0069).

    The default targets are **only the `weight` of `nn.Linear`** because the i4 execution path
    started limited to linear weight slots (ADR 0069 decision 5). `op_types` is **the explicit
    opt-in hook** and can be widened up to the i8 {@link QUANT_MODULE_TYPES} (all 6 types) — but
    shipping may widen only up to `emit.I4_WEIGHT_OPS` (linear / embedding / conv1d) (conv2d /
    conv3d / `ConvTranspose1d` are for quality measurement only). Biases and norm-style weights are
    left alone for every target.

    MUST: **only scales for `emit.I4_WEIGHT_OPS` (linear / embedding / conv1d) may be passed to
    emit** (eligibility in `emit._plan_i4` means the consuming op is in that set). The scale shape
    is decided one way by the rank-2 rule of {@link storage_rows} whatever the rank
    ({@link Int4Report}). conv2d and `ConvTranspose1d` have no i4 execution path, so emit fails
    loudly when given them — the widened targets serve the side that measures "how far the method
    recovers", not the shipping path (the measurement-only methods are in `quant_methods`).

    MUST: the group order of conv1d is **the weight's contiguous memory order**
    ({@link storage_rows}) — `ConvTranspose1d`, whose row axis is not first, can be used only for
    measurement (stored, it would disagree with the pack order).

    The group axis generalises each op's "in axis" ({@link channel_rows}) — the in axis for
    linear, the receptive field per output channel for conv (`Cin·K` / `Cin·Kh·Kw`), `Cin·K` per
    axis 1 of the transposed layout for `ConvTranspose1d`, and the D axis per vocabulary entry for
    embedding.

    `include` is a predicate on the **module FQN** (None = every target) — its meaning and the
    exclusivity MUST are the same as in {@link fake_quant_int8}. A tied lm_head (= embed_tokens,
    also consumed on the embedding side) is **i4-eligible** because both linear and embedding have
    an i4 decode path (`emit.I4_WEIGHT_OPS` — the embedding supplement of 2026-08-20 reversed
    "ineligible because linear only"). The scale shape agrees too, at `[V, D/g]`.

    MUST: the call order is "after the effective weights are settled, before the reference /
    goldens are captured" (module docstring).

    The returned scale ledger is passed to emit as is. When not a single weight could be quantized
    it fails loudly — so that "`--dtype i4` was requested but it effectively wrote f32" is never
    silent (ADR 0006; the same when `include` dropped every target).
    """
    scales: dict[str, torch.Tensor] = {}
    elements = 0
    with torch.no_grad():
        for fqn, weight, axis in iter_quant_targets(model, op_types, include):
            rows = channel_rows(weight, axis)
            scale = group_scale(rows, group_size, f"'{fqn}'")
            rounded = dequantize_int4(quantize_to_int4(rows, scale), scale)
            weight.copy_(restore_channel_rows(rounded, weight, axis))
            scales[fqn] = scale
            elements += weight.numel()
    if not scales:
        raise QuantizeError(
            "storage i4 was requested but there is no weight that can be group-quantized"
            f" (target types: {', '.join(cls.__name__ for cls in op_types)} — the default is"
            " nn.Linear only, ADR 0069 decision 5)"
        )
    return Int4Report(scales=scales, group_size=group_size, modules=len(scales), elements=elements)
