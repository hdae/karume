"""**Storage conversion** of the distribution, and assembly of the material handed to the
container (raw bytes + binding table).

The container itself (the physical format of `krm`, descriptors, the placement of blocks /
parts) belongs to `karume.container`, and the three publish steps (write → read back and verify
→ swap into place) belong to `karume.publish`. All this module decides is **which initializer is
written in which storage**, and that is closed inside the single path of {@link stored_model} (a
branch here would leave export green while only execution fails).

The storage branches are **f16 (ADR 0018), i8 (ADR 0019) and i4 (ADR 0069)**, plus the fixed
packed path ({@link _plan_fixed_weights} — i2 only through this path, ADR 0097).

## Eligibility (ADR 0018 / 0019 — **the same rule** for f16 and i8, narrower only for i4)

When writing with `weight_dtype="f16"` / `"i8"` / `"i4"`, only **eligible initializers** are
stored compressed:

1. The initializer is consumed **only** at weight slots of `ops.WEIGHT_SLOTS`
   (the same rule as `eligibleCompressedInitializers` in the runtime's
   `packages/runtime/src/runtime/plan.ts`.
   Biases, norm weights, mixed consumption and zero consumption are all ineligible — ADR 0006).
   **i4 is narrower still: only weight slots of {@link I4_WEIGHT_OPS} (`linear` / `embedding` /
   `conv1d`)** (ADR 0069 decision 5 and the embedding / conv1d addenda — these 3 are the only ops
   with an i4 decode path, and conv1d further needs `groups == 1` and a stored row length that
   divides). Even when generally eligible, conv2d / conv_transpose1d weights stay f32 — the same
   fallback as i8's ineligible ones, the counterpart of the runtime's eligible ∩ i4Eligible.
2. **The inverse conversion is bit-identical** (= fake-quant applied). For f16 the round trip
   `f32 → f16 → f32`, for i8 `q8.to(f32) · scale`, for i4 `dequant(unpack(pack(q4)))` (`scale` is
   **exactly** the value fake-quant used).

2 is a condition because storing unrounded values compressed gives "golden from the original
values, execution on rounded values", mixing quantization error with implementation error. An
eligible initializer that fails 2 **fails loudly** (a missed rounding, rounding in the wrong order,
or a folded constant that flowed into a weight slot — any of them, let through silently, would make
the E2E tolerance meaningless). Only the i4 check traces back **from the stored bytes**, because
mixing up pack / unpack is a silent wrong value with matching shape and type (ADR 0069 decision 4
③).

## companion scale (ADR 0019 / 0069)

The scale goes into the same container as **an independent tensor key**, and the IR side declares
that key explicitly with `storage.scale`. The key is built mechanically by `_scale_key`
(`karume.scale.<weight key>`), and **collisions with real tensors** are checked before writing (a
collision would mean "reading another weight as the scale": loading succeeds and only the values
break). The shape takes two forms by storage — for i8 the per-channel keepdim broadcast shape, for
i4 the group shape of **rank 2 (rows = the weight's first dimension, last dimension = the number of
groups)**, and the latter also declares `storage.group_size` (for rank-2 weights this equals "same
rank, only the last dimension is the number of groups"; for conv1d `[Cout,Cin,K]` it is
`[Cout,(Cin·K)/g]` — `quantize.storage_rows`). The scale is the value fake-quant used, received from
`quantize.fake_quant_int8` / `quantize.fake_quant_int4` and written as is — recomputing it from
amax here could move it by 1 ulp through f32 rounding, which breaks the correspondence with the
weights the golden was taken with.

## The three items handed to the container ({@link StoredModel})

Placement (how blocks are cut, how parts are packed, the order) is `karume.container`'s
responsibility, so this module hands over only three items: "the graph with the storage
declarations committed", "tensor key → **raw bytes after storage conversion**" and "tensor key →
{@link karume.container.Encoding}".

- The raw-bytes mapping is **lazy** ({@link _StoredTensors}) — only the one tensor fetched is
  converted and returned. Converting everything up front would keep it alive together with the
  caller's f32 set, and peak RAM would be the sum of both (at Irodori scale, f16 1.72GB / i8
  0.87GB on top of f32 3.44GB). The writer **fetches the data twice** (the pass that takes the
  sha256 and the pass that writes — container-v1 §4.1), so the same key MUST return **the same
  bytes every time**. The conversion is deterministic, so this holds.
- The binding table comes from the single function {@link karume.container.container_bindings}
  (**the same function** as the migration CLI, on the side that needs no torch). Deriving it in two
  paths would allow silent wrong values with matching shape and type, such as "declared i4 / data
  i8".
"""

from __future__ import annotations

from collections.abc import Buffer, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from typing import Literal

import torch

from karume.container import (
    CODEC_FOR_STORAGE,
    Encoding,
    bakeable_initializers,
    container_bindings,
    weight_channel_axes,
)
from karume.container import scale_bytes as codec_scale_bytes
from karume.ir import MIN_GROUP_SIZE, IrGraph, IrInitializer, IrNode, IrStorage
from karume.ops import (
    CONV1D_OP,
    EMBEDDING_OP,
    LINEAR_OP,
    WEIGHT_SLOTS,
    conv1d_attrs,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QuantizeError,
    dequantize_int4,
    group_size_of,
    quantize_to_int4,
    quantize_to_int8,
)

#: Storage dtypes that can be written (for weight slots). bf16 has the same skeleton as f16
#: (eligible = the round trip is bit-identical; the execution path is ADR 0131).
WEIGHT_DTYPES = ("f32", "f16", "bf16", "i8", "i4")

#: Ops **with an i4 execution path** (= the consumers of i4-eligible weight slots — linear of ADR
#: 0069 decision 5 and its embedding / conv1d addenda). The mirror of `I4_WEIGHT_OPS` in the
#: runtime's `packages/runtime/src/runtime/plan.ts`; when the two diverge the result is either
#: "export green, load falls back to CPU decode (the VRAM saving silently disappears)" or the
#: reverse, "export alone fails by requesting a storage it cannot declare".
#:
#: MUST: conv2d / conv3d / conv_transpose1d are not included. i4 on conv1d can be decoded only by
#: the igemm path (the `groups == 1` variant); passing i4 to the direct kernel (`groups > 1`) or to
#: the generator inputs of conv2d / conv3d / conv_transpose1d produces invalid WGSL
#: (`packages/runtime/src/kernels/weight-storage.ts`). The `groups == 1` narrowing is done by
#: {@link i4_eligible_initializers} from the node's attrs.
#: MUST: for conv_transpose1d, **the row axis of the weight `[Cin,Cout,K]` is not the first**, so
#: the packed bytes disagree with the row order (`quantize.storage_rows`) — adding a decode path
#: alone cannot make it eligible (the maintainer's ruling of 2026-08-20: no permuted pack).
I4_WEIGHT_OPS: frozenset[str] = frozenset({LINEAR_OP, EMBEDDING_OP, CONV1D_OP})

#: Offset of a packed nibble (stored value `u = q + 8`, range [1,15], 0 unused = 15 levels).
#: Should storage ever become asymmetric, this constant is merely reread as "the default = 8 when
#: `storage.zero_point` is omitted"; the pack format, pack order and range do not move
#: (reservation 2 of ADR 0069 decision 3).
INT4_OFFSET = 8


class EmitError(ValueError):
    """The declaration (graph JSON) and the set of stored tensors disagree / a storage
    precondition is broken."""


@dataclass(frozen=True)
class StorageBreakdown:
    """Tensor counts and bytes per storage dtype (the exporter side of ADR 0006's standing
    diagnostic)."""

    #: Initializers resident on the GPU in compressed form (= eligible).
    compressed_tensors: int
    compressed_bytes: int
    #: Initializers stored as f32 (biases / norm weights / constants / ineligible weights).
    plain_tensors: int
    plain_bytes: int
    #: companion scales of quantized storage (i8 / i4) (tensors separate from the weights). The
    #: runtime's `residentCompressedBytes` adds the scales too, so the diagnostic means the same on
    #: both sides.
    scale_bytes: int = 0

    def describe(self) -> str:
        scale = f", scale {self.scale_bytes:,} B" if self.scale_bytes else ""
        return (
            f"eligible {self.compressed_tensors} tensors / {self.compressed_bytes:,} B{scale}, "
            f"ineligible {self.plain_tensors} tensors / {self.plain_bytes:,} B"
        )


def eligible_compressed_initializers(graph: IrGraph) -> set[str]:
    """Initializers that **can** stay resident on the GPU in compressed storage
    (the mirror of `packages/runtime/src/runtime/plan.ts`).

    Eligible = "the initializer is consumed only at weight slots of ops listed in WEIGHT_SLOTS".
    MUST: a single consumption outside a weight slot makes it ineligible (that kernel reads it as
    f32, so uploading it compressed reinterprets the bits). Zero consumption is ineligible too
    (counting bytes that execution never uses as "GPU-resident compressed" would make the
    diagnostic diverge from reality).
    MUST: an initializer listed in `graph.outputs` is ineligible too (the runtime's readback
    copies from the weight buffer assuming 4 bytes / element of semantic f32, so keeping it
    resident compressed either fails validation or, at tiny sizes, silently returns reinterpreted
    bits).
    MUST: shared initializers (ADR 0096 stage 2 — declarations without bytes) are not put in the
    eligible set. This container holds not a single byte of them, so "store compressed or not" is
    not a question for them, and including them would make `_plan_weight_dtype` look up a tensor
    key that does not exist.
    """
    initializers = bakeable_initializers(graph)
    eligible: set[str] = set()
    disqualified: set[str] = set(graph.outputs)
    for node in graph.nodes:
        weight_slot = WEIGHT_SLOTS.get(node.op)
        for slot, name in enumerate(node.ins):
            if name not in initializers:
                continue
            if slot == weight_slot:
                eligible.add(name)
            else:
                disqualified.add(name)
    return eligible - disqualified


def _has_i4_kernel(node: IrNode) -> bool:
    """Whether this node **can decode** i4 at its weight slot (the narrowing of
    {@link I4_WEIGHT_OPS}).

    For conv1d, only the igemm variant (`groups == 1`) can decode i4; the direct kernel
    (`groups > 1`) stays without i4 support (the mirror of `i4Executable` in the runtime's
    `plan.ts`).

    MUST: `groups` is **not filled in with a default** (ADR 0012) — it goes through the contract
    table's `ops.conv1d_attrs`, which rejects a missing one. Silently assuming 1 here would make a
    depthwise weight resident as i4 and the direct kernel read the packed bytes as f32 (no
    exception is raised).
    """
    if node.op not in I4_WEIGHT_OPS:
        return False
    if node.op != CONV1D_OP:
        return True
    return conv1d_attrs(node.attrs, f"graph.nodes ({node.op})")[3] == 1


def _storage_row_length(graph: IrGraph, name: str) -> int | None:
    """The **stored row length** of an initializer (the row length of `quantize.storage_rows` =
    the product of all dimensions but the first).

    The group scale's shape is set by this row length regardless of rank (ADR 0069 decision 3), so
    the i4 eligibility check uses it as the dividend. Rank 1 or below, shapes with symbolic
    dimensions and undeclared names give None (they cannot have an i4 storage shape at all).
    """
    value = graph.values.get(name)
    if value is None or len(value.shape) < 2:
        return None
    length = 1
    for dim in value.shape[1:]:
        if not isinstance(dim, int):
            return None
        length *= dim
    return length


def _scale_group_size(
    graph: IrGraph, scales: Mapping[str, torch.Tensor], name: str, row_length: int
) -> int | None:
    """The group length declared by the weight's own group scale (None when it cannot be found).

    The source is the same as `quantize.group_size_of`: "**only the shape of the passed scale**"
    (`[rows, groups]` → row length / groups). If the eligibility check and the storage declaration
    had separate sources, the g fake-quant used and the g the check used could move independently —
    and a mismatch would stay silent, because shape and type still match.

    A scale missing from the ledger / not in group form returns None and defers to the caller's
    default ("eligible but no scale" fails loudly in `_plan_i4`, so it is not dropped from the
    eligible set here).
    """
    scale = scales.get(graph.initializers[name].tensor)
    if scale is None or scale.dim() != 2:
        return None
    groups = int(scale.shape[1])
    if groups < 1 or row_length % groups:
        return None
    return row_length // groups


def i4_eligible_initializers(
    graph: IrGraph,
    group_size: int = DEFAULT_GROUP_SIZE,
    scales: Mapping[str, torch.Tensor] | None = None,
) -> set[str]:
    """Initializers consumed at weight slots **only** by {@link I4_WEIGHT_OPS} and whose stored
    row length divides by the group length (the i4 eligible set — ADR 0069 decision 5 and its
    addenda).

    The kernels with an i4 decode path (`unpack4xU8` + group scale) are only linear / embedding /
    conv1d (`groups == 1`), so an initializer also consumed at any other weight slot (conv2d /
    conv3d / conv_transpose1d / conv1d with groups > 1) cannot be stored as i4 — that kernel reads
    the packed bytes as f32 (no exception is raised). Like `weight_channel_axes`, it is derived
    from **the consuming op** (the weight's shape alone cannot tell them apart).

    The row-length check ({@link _storage_row_length} % group length) is needed so that weights
    the no-partial-group MUST (ADR 0069 decision 2) cannot round are **dropped from the eligible
    set and quietly left f32**. Picking them up here would fail the whole export with "a weight
    fake-quant did not reach" (an ordinary graph with linear + a non-dividing conv1d could no
    longer be written as i4 — the same shape as I4-ELIG-01).

    MUST: the divisor is taken **per weight from its own group scale** (the `scales` ledger →
    {@link _scale_group_size}). Fixing the divisor to the shipped default group length
    (`quantize.DEFAULT_GROUP_SIZE`) would drop a weight with row length 48 rounded at g16 from the
    eligible set via `48 % 32 != 0`, and its values, still rounded to the i4 grid, would be **stored
    as f32** — only the quality loss, not a single byte smaller, with no exception and no
    diagnostic (the silence ADR 0006 names: "compression requested but effectively f32").
    `group_size` is the fallback default that applies only to names whose scale cannot be found.
    """
    bakeable = bakeable_initializers(graph)
    executable: set[str] = set()
    other: set[str] = set()
    for node in graph.nodes:
        slot = WEIGHT_SLOTS.get(node.op)
        if slot is None or slot >= len(node.ins):
            continue
        name = node.ins[slot]
        if name not in bakeable:
            continue
        (executable if _has_i4_kernel(node) else other).add(name)
    ledger = scales or {}
    eligible: set[str] = set()
    for name in executable - other:
        row_length = _storage_row_length(graph, name)
        if row_length is None:
            continue
        divisor = _scale_group_size(graph, ledger, name, row_length) or group_size
        if row_length % divisor == 0:
            eligible.add(name)
    return eligible


def _scale_key(tensor_key: str) -> str:
    """Tensor key of the companion scale (built mechanically from the weight key)."""
    return f"karume.scale.{tensor_key}"


def pack_int2(quantized: torch.Tensor) -> torch.Tensor:
    """Pack fixed integers in [-2,1] starting from the low 2 bits. Does not quantize or change the
    scale.

    The source of truth for the i2 byte order (ADR 0097): 4 elements consecutive in flat index go
    into one byte, element `4i` in the lowest 2 bits, stored value `u = q + 2` with offset 2. The
    writer (the gemma4_qat recipe) passes the upstream packed bytes as they are, so production code
    does not call this function. Mixing up the order would still only be a silent wrong value with
    matching shape and type, so the order lives here and `tests/test_i2_storage.py` pins it by byte
    value.
    """
    if quantized.dtype != torch.int8:
        raise EmitError("INT2 integers must be held in int8")
    flat = quantized.reshape(-1)
    if flat.numel() % 4:
        raise EmitError("INT2 element count must be a multiple of 4")
    if bool(((flat < -2) | (flat > 1)).any()):
        raise EmitError("INT2 integers must be in [-2,1]")
    u = (flat + 2).to(torch.uint8)
    return u[0::4] | (u[1::4] << 2) | (u[2::4] << 4) | (u[3::4] << 6)


def unpack_int2(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Unpack packed INT2 back to fixed integers. Requires the logical shape and the total byte
    count to agree."""
    count = 1
    for dim in shape:
        if type(dim) is not int or dim < 0:
            raise EmitError("INT2 shape must be non-negative integers")
        count *= dim
    if packed.dtype != torch.uint8 or packed.numel() * 4 != count:
        raise EmitError("INT2 bytes do not match the logical shape")
    flat = packed.reshape(-1)
    lanes = torch.stack(tuple((flat >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
    return (lanes.to(torch.int8) - 2).reshape(tuple(shape))


def pack_int4(quantized: torch.Tensor) -> torch.Tensor:
    """Pack `q ∈ [−7,7]` (held in int8, logical shape) into packed 4-bit (1-D uint8).

    MUST: **2 elements consecutive in flat index go into one byte**, element `2i` in the **low**
    nibble and `2i+1` in the **high** nibble (ADR 0069 decision 4 — the counterpart of the decode
    order of two `unpack4xU8` calls). The stored value is the unsigned nibble `u = q + 8` with
    offset 8. Mixing it up with llama.cpp Q4_0's split-half order would only be **a silent wrong
    value** with matching shape and type, so the source of truth for the order is here and
    `tests/test_emit.py` pins it by byte value.

    MUST: a shape with an odd element count fails loudly (the last element sticks out by half a
    byte, and the tensor length is not determined uniquely by the declaration) — the i4
    quantization axis divides by `group_size` (a power of two, at least 16), so this is not
    reached, but this is the source of truth for the pack order, so the precondition is not
    silently assumed.
    """
    flat = quantized.reshape(-1)
    if flat.numel() % 2:
        raise EmitError(
            f"packed 4-bit element count {flat.numel()} is odd (cannot pack 2 elements per byte)"
        )
    nibbles = (flat + INT4_OFFSET).to(torch.uint8)
    return nibbles[0::2] | (nibbles[1::2] << 4)


def unpack_int4(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Unpack packed 4-bit back to `q ∈ [−7,7]` (held in int8, logical shape `shape`) (the inverse
    of `pack_int4`).

    The same mapping as the runtime's decode (WGSL / CPU decode): the element at flat index `i`
    comes back as `u − 8` from the `i % 2 == 0 ? low : high` nibble of `packed[i / 2]`.
    """
    low = (packed & 0x0F).to(torch.int16) - INT4_OFFSET
    high = (packed >> 4).to(torch.int16) - INT4_OFFSET
    return torch.stack((low, high), dim=-1).reshape(tuple(shape)).to(torch.int8)


def _is_f16_exact(tensor: torch.Tensor) -> bool:
    """Whether the round trip f32 → f16 → f32 is bit-identical (= fake-quant applied)."""
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


def _is_bf16_exact(tensor: torch.Tensor) -> bool:
    """Whether the round trip f32 → bf16 → f32 is bit-identical (= fake-quant applied)."""
    return bool(torch.equal(tensor, tensor.to(torch.bfloat16).to(torch.float32)))


def _unrounded(name: str, tensor_key: str, detail: str) -> EmitError:
    return EmitError(
        f"initializer '{name}' ({tensor_key}) is eligible for a weight slot but {detail} —"
        " fake-quant was not applied, or was applied after the reference was taken"
        " (ADR 0006: rounding MUST precede taking the reference and the golden)"
    )


@dataclass(frozen=True)
class FixedQuantizedWeight:
    """Packed data and scale of fixed integers. The logical shape is carried by the matching
    f32/meta Tensor.

    `i4f16` is `int4-sym-g-f16` (ADR 0145 decision 3): the codes of `i4` and its F32 steps, stored
    as f16 steps ({@link f16_step_table}). Only `expert_linear` reads it, so no consumer admits it
    yet — linear / embedding / conv1d MUST NOT.
    """

    dtype: Literal["i2", "i4", "i4f16", "i8"]
    packed: torch.Tensor
    scale: torch.Tensor


#: The smallest positive normal f16, `2^-14` — the floor of an all-zero group's f16 step (ADR 0069
#: decision 3: the f32 floor `f32 tiny` rounds to a zero f16, which is refused).
F16_MIN_NORMAL = 2.0**-14


def f16_step_table(packed: torch.Tensor, steps: torch.Tensor, where: str) -> torch.Tensor:
    """The f16 step table `int4-sym-g-f16` stores for packed i4 codes and their F32 steps (ADR
    0145 decision 3, under ADR 0069 decision 3's conditions for an f16 scale).

    `packed` is the `[rows, width / 2]` uint8 codes ({@link pack_int4}'s order), `steps` the
    `[rows, groups]` F32 steps. Each step is rounded to f16 **once**, here (round to nearest even —
    the reference and the goldens use the rounded step, ADR 0006):

    - a group whose codes are all 0 takes `max(step, F16_MIN_NORMAL)` first (its values are 0 for
      any finite step, and `clamp(amax / 7, f32 tiny)` would round to a zero f16);
    - a rounded step that is zero or not finite fails loudly (the same family as the f16 emit's
      inf rejection);
    - the round trip traces back **from the table's bytes** (ADR 0069 decision 4 ③): read as the
      runtime reads them (u32 words, step `2i` in the low half, `2i+1` in the high half) and widened
      to f32, they decode the codes to the values of the rounded steps.

    Returns the steps flat in row-major order as F16, padded with one zero step to a whole u32
    word when their count is odd — exactly the payload `container.scale_bytes` declares.
    """
    if (
        packed.device.type != "cpu"
        or packed.dtype != torch.uint8
        or packed.ndim != 2
        or steps.device.type != "cpu"
        or steps.dtype != torch.float32
        or steps.ndim != 2
        or steps.shape[0] != packed.shape[0]
        or steps.shape[1] <= 0
    ):
        raise EmitError(
            f"{where}: f16 steps need CPU uint8 codes [rows, width/2] and F32 steps [rows, groups]"
        )
    rows, groups = (int(dim) for dim in steps.shape)
    width = int(packed.shape[1]) * 2
    if width % groups:
        raise EmitError(f"{where}: {groups} groups do not divide the row length {width}")
    codes = unpack_int4(packed, (rows, width))
    all_zero = (codes.reshape(rows, groups, width // groups) == 0).all(dim=-1)
    floored = torch.where(
        all_zero, torch.maximum(steps, torch.tensor(F16_MIN_NORMAL, dtype=torch.float32)), steps
    )
    rounded = floored.to(torch.float16)
    widened = rounded.to(torch.float32)
    refused = (widened == 0) | ~torch.isfinite(widened)
    if bool(refused.any()):
        row, group = (int(axis) for axis in refused.nonzero()[0])
        raise EmitError(
            f"{where}: {int(refused.sum())} steps round to a zero or non-finite f16 — the first is"
            f" group {group} of row {row} (F32 step {float(steps[row, group])!r})"
        )
    table = rounded.reshape(-1)
    if table.numel() % 2:
        table = torch.cat((table, torch.zeros(1, dtype=torch.float16)))
    # MUST: trace back from the bytes about to be written, read the way the kernel reads them —
    # comparing with `rounded` alone would let a swapped half or byte order slip past.
    words = table.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    halves = torch.stack((words & 0xFFFF, words >> 16), dim=-1).reshape(-1)[: rows * groups]
    # The 16-bit patterns as signed values, so the cast to int16 keeps the bits.
    bits = torch.where(halves >= 0x8000, halves - 0x10000, halves).to(torch.int16)
    read_back = bits.view(torch.float16).to(torch.float32).reshape(rows, groups)
    if not torch.equal(dequantize_int4(codes, read_back), dequantize_int4(codes, widened)):
        raise EmitError(f"{where}: the f16 step table does not decode back to its rounded steps")
    return table


@dataclass(frozen=True)
class _Conversion:
    """The compression conversion applied to one tensor right before storage (**not** applied at
    the planning step).

    `name` is the IR value name for diagnostic messages, `scale` the per-channel scale for i8
    (exactly the value fake-quant used — ADR 0019).
    """

    dtype: str
    name: str
    scale: torch.Tensor | None = None
    fixed_payload: torch.Tensor | None = None


@dataclass(frozen=True)
class _StoragePlan:
    """The plan for compressed storage. Does not touch a single byte of the weights."""

    #: IR value name → new initializer declaration (committed after every byte is written).
    declarations: dict[str, IrInitializer]
    #: Additional companion scales to store (container tensor key → tensor). Scales are small, so
    #: they may be materialised at the planning step (unlike the weights, holding them as a set
    #: does not show up in the peak).
    scales: dict[str, torch.Tensor]
    #: Container tensor key → the conversion applied right before writing.
    conversions: dict[str, _Conversion]


def _plan_i8(
    graph: IrGraph,
    reserved: Set[str],
    name: str,
    tensor: torch.Tensor,
    scales: Mapping[str, torch.Tensor],
    axes: Mapping[str, int],
) -> tuple[str, torch.Tensor, IrInitializer]:
    """Return the i8 storage plan (the scale's key and tensor, the new declaration) — does not
    touch the weight.

    `reserved` is "the stored tensor keys already taken" (the inputs + the scales planned before).
    The quantization itself and the bit-identity check of the inverse conversion are the heavy
    side that reads real data, so they go to `_convert_for_storage` (right before writing).
    """
    initializer = graph.initializers[name]
    key = initializer.tensor
    scale = scales.get(key)
    if scale is None:
        # Eligible but no scale = a weight fake-quant did not reach (a folded constant flowed
        # into a weight slot / a missed target module type). Falling back to f32 would only
        # silently shrink the VRAM saving, so it is not accepted when i8 is requested.
        raise _unrounded(name, key, "has no per-channel scale")
    if scale.dtype is not torch.float32:
        # companion scales are fixed to F32 (ADR 0019 / docs/ir-v2.md). The writer can
        # serialize F16 as is, and the equality check of the inverse conversion would pass for
        # "fake-quant applied with the same f16 scale", so it fails at the planning step (the
        # diagnostic moves forward from "verify after writing").
        raise EmitError(
            f"initializer '{name}' ({key}): scale dtype is {scale.dtype} — only F32 is accepted"
        )
    axis = axes.get(name)
    expected = [dim if index == axis else 1 for index, dim in enumerate(tensor.shape)]
    if list(scale.shape) != expected:
        raise EmitError(
            f"initializer '{name}' ({key}): scale {list(scale.shape)} is not the keepdim shape"
            f" of weight {list(tensor.shape)} along axis {axis} {expected}"
        )
    scale_key = _scale_key(key)
    if scale_key in reserved:
        raise EmitError(
            f"initializer '{name}': scale tensor key '{scale_key}' collides with a real tensor"
        )
    return (
        scale_key,
        scale.detach().contiguous(),
        IrInitializer(tensor=key, storage=IrStorage(dtype="i8", scale=scale_key)),
    )


def _plan_i4(
    graph: IrGraph,
    reserved: Set[str],
    name: str,
    tensor: torch.Tensor,
    scales: Mapping[str, torch.Tensor],
    i4_eligible: Set[str],
) -> tuple[str, torch.Tensor, IrInitializer]:
    """Return the i4 storage plan (the scale's key and tensor, the new declaration) — does not
    touch the weight.

    The same approach as `_plan_i8` (key generation, collision check, scale dtype check) with
    three differences: eligibility is **limited to weight slots of {@link I4_WEIGHT_OPS}** (ADR
    0069 decision 5), the scale is **in group form**, and the declaration carries
    `storage.group_size`. The group length is **taken from the shape of the passed scale**
    (`quantize.group_size_of` — accepting it as a separate argument would allow declaring a value
    other than the group fake-quant used, a silent wrong value with matching shape and type).

    Quantization, packing and the bit-identity check of the inverse conversion are the heavy side
    that reads real data, so they go to `_convert_for_storage` (right before writing).
    """
    initializer = graph.initializers[name]
    key = initializer.tensor
    if name not in i4_eligible:
        raise EmitError(
            f"initializer '{name}' ({key}): i4 storage is only for weight slots of"
            f" {' / '.join(sorted(I4_WEIGHT_OPS))}"
            " (ADR 0069 decision 5 and its addenda — conv1d also needs groups == 1 and a stored"
            " row length that divides by the group length)"
        )
    scale = scales.get(key)
    if scale is None:
        # Eligible but no scale = a weight fake-quant did not reach (same reason as `_plan_i8`).
        raise _unrounded(name, key, "has no group scale")
    if scale.dtype is not torch.float32:
        raise EmitError(
            f"initializer '{name}' ({key}): scale dtype is {scale.dtype} — only F32 is accepted"
        )
    try:
        group_size = group_size_of(tensor, scale)
    except QuantizeError as cause:
        raise EmitError(f"initializer '{name}' ({key}): {cause}") from cause
    # The accepted set (a power of two, at least 16) — the same rule as the declaration layer —
    # is enforced **before writing** (the same approach as `_plan_i8` checking the scale's keepdim
    # shape in two stages with verify). The check after writing is
    # `verify.ir_graph_from_container` (the side that rebuilds the IR from the binding table of
    # the container put in place).
    if group_size & (group_size - 1) or group_size < MIN_GROUP_SIZE:
        raise EmitError(
            f"initializer '{name}' ({key}): group_size {group_size} taken from the scale is not"
            f" a power of two of at least {MIN_GROUP_SIZE} (ADR 0069 decision 2)"
        )
    scale_key = _scale_key(key)
    if scale_key in reserved:
        raise EmitError(
            f"initializer '{name}': scale tensor key '{scale_key}' collides with a real tensor"
        )
    return (
        scale_key,
        scale.detach().contiguous(),
        IrInitializer(
            tensor=key,
            storage=IrStorage(dtype="i4", scale=scale_key, group_size=group_size),
        ),
    )


def _plan_fixed_weights(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    fixed_weights: Mapping[str, FixedQuantizedWeight],
) -> _StoragePlan:
    """Treat the fixed bytes as the only value; no re-quantization and no f32 materialisation
    (ADR 0097)."""
    bakeable = bakeable_initializers(graph)
    names = {graph.initializers[name].tensor: name for name in bakeable}
    if len(names) != len(bakeable):
        raise EmitError("Initializer names and tensor keys of fixed storage are not 1:1")
    unknown = sorted(set(fixed_weights) - set(names))
    if unknown:
        raise EmitError(f"Unknown fixed_weights keys: {unknown}")
    eligible = eligible_compressed_initializers(graph)
    plan = _StoragePlan(declarations={}, scales={}, conversions={})
    reserved = set(tensors)
    for key, fixed in fixed_weights.items():
        name = names[key]
        tensor = tensors[key]
        shape = list(tensor.shape)
        if (
            not tensor.is_meta
            or tensor.dtype != torch.float32
            or len(shape) != 2
            or any(dim <= 0 for dim in shape)
            or graph.values[name].dtype != "f32"
            or graph.values[name].shape != shape
            or graph.initializers[name].storage.dtype != "f32"
        ):
            raise EmitError(
                f"fixed_weights['{key}']: must be a positive rank-2 f32/meta of the declared shape"
            )
        consumers = [node for node in graph.nodes if name in node.ins]
        if fixed.dtype == "i4f16":
            # ADR 0145 decision 3: the f16 steps are read by `expert_linear` only, which no op
            # here is yet — linear / embedding / conv1d read a scale as F32.
            ops = " / ".join(sorted({node.op for node in consumers})) or "nothing"
            raise EmitError(
                f"fixed_weights['{key}']: int4-sym-g-f16 (f16 steps) is read by expert_linear"
                f" only — ADR 0145 (consumed by {ops})"
            )
        if name not in eligible or any(
            node.op not in (LINEAR_OP, EMBEDDING_OP) for node in consumers
        ):
            raise EmitError(f"fixed_weights['{key}']: only for linear/embedding weights")
        if fixed.dtype not in ("i2", "i4", "i8"):
            raise EmitError(f"fixed_weights['{key}']: dtype must be i2/i4/i8")
        rows, width = shape
        factor = {"i2": 4, "i4": 2, "i8": 1}[fixed.dtype]
        packed_dtype = torch.int8 if fixed.dtype == "i8" else torch.uint8
        packed = fixed.packed
        if (
            packed.device.type != "cpu"
            or packed.dtype != packed_dtype
            or not packed.is_contiguous()
            or width % factor
            or list(packed.shape) != [rows, width // factor]
        ):
            raise EmitError(
                f"fixed_weights['{key}']: packed has the wrong CPU dtype, shape or contiguity"
            )
        scale = fixed.scale
        if (
            scale.device.type != "cpu"
            or scale.dtype != torch.float32
            or scale.ndim != 2
            or scale.shape[0] != rows
            or scale.shape[1] <= 0
        ):
            raise EmitError(f"fixed_weights['{key}']: scale must be CPU F32 [rows,groups]")
        group_size = None
        if fixed.dtype == "i4":
            groups = scale.shape[1]
            group_size = width // groups
            if width % groups or group_size < 16 or group_size & (group_size - 1):
                raise EmitError(
                    f"fixed_weights['{key}']: I4 group must divide the row, be a power of two"
                    " and be at least 16"
                )
        elif list(scale.shape) != [rows, 1] or (fixed.dtype == "i2" and width % 16):
            raise EmitError(
                f"fixed_weights['{key}']: requires a row scale [rows,1] and I2 alignment to 16"
                " elements"
            )
        scale_key = _scale_key(key)
        if scale_key in reserved:
            raise EmitError(f"fixed_weights['{key}']: scale key collides: {scale_key}")
        scale = scale.detach().contiguous()
        reserved.add(scale_key)
        plan.scales[scale_key] = scale
        plan.declarations[name] = IrInitializer(
            tensor=key, storage=IrStorage(dtype=fixed.dtype, scale=scale_key, group_size=group_size)
        )
        plan.conversions[key] = _Conversion(
            dtype=fixed.dtype, name=name, scale=scale, fixed_payload=packed.detach()
        )
    return plan


def _plan_weight_dtype(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    weight_dtype: str,
    weight_scales: Mapping[str, torch.Tensor],
    weight_dtype_overrides: Mapping[str, str],
) -> _StoragePlan:
    """**Only plan** the compressed storage of eligible weights (neither declarations nor data are
    rewritten yet).

    MUST: no compressed tensors are created at this step. Converting everything up front would
    keep the compressed set alive **together with** the caller's f32 set, and peak RAM would be
    the sum of both (at Irodori scale, f16 1.72GB / i8 0.87GB on top of f32 3.44GB). Only checks
    that do not read real data run here (name ↔ tensor key 1:1, validity of the storage dtype,
    eligibility, presence and shape of the scale [i8 = keepdim / i4 = group], scale key
    collisions, "zero eligible"); the checks that read real data are done per tensor by
    `_convert_for_storage` right before writing.

    `weight_dtype_overrides` is an explicit **tensor key (FQN) → storage dtype** choice (mixed
    storage — first used for an LLM's "embedding as i8, linear as i4"). While the default
    `weight_dtype` is **quietly left f32** by the eligibility filter, an explicit choice **fails
    loudly when it cannot be met** (none of: unknown key, ineligible, non-f32 data, i4 on an
    i4-ineligible weight passes) — with the caller's intent written per tensor, there is no room to
    silently use another storage. An explicit `"f32"` serves as "excluded from the compressed
    default".
    """
    # MUST: the walk covers **only initializers whose data is written** (shared declarations
    # borrow the lender's bytes, so they are not subject to the storage dtype plan — ADR 0096
    # stage 2).
    bakeable = bakeable_initializers(graph)
    initializer_by_key = {graph.initializers[name].tensor: name for name in bakeable}
    if len(initializer_by_key) != len(bakeable):
        # MUST: initializer name ↔ tensor key is 1:1. If it collapses, the eligibility check only
        # sees the one name that won last, while the planning loop walks **every** eligible name —
        # the data is converted to packed per key, and the declaration of a name that was
        # ineligible stays f32, so only the values break while shape and type still match. The
        # current convert path (torch.export placeholder → FQN is injective,
        # `Converter._add_const` is unique by digest) does not reach this, but that 1:1 rests only
        # on one point of upstream implementation behaviour and is written down nowhere, so this
        # is the only check.
        names_by_key: dict[str, list[str]] = {}
        for name in sorted(bakeable):
            names_by_key.setdefault(graph.initializers[name].tensor, []).append(name)
        collided = {key: names for key, names in sorted(names_by_key.items()) if len(names) > 1}
        raise EmitError(
            f"Initializer names and tensor keys are not 1:1: {collided}"
            " (declarations are per name and data per key, so when only some of the names"
            " pointing at the same key get compressed storage, declaration and data diverge)"
        )
    if weight_dtype not in WEIGHT_DTYPES:
        raise EmitError(
            f"Storage dtype '{weight_dtype}' cannot be written ({' / '.join(WEIGHT_DTYPES)})"
        )
    for key, dtype in weight_dtype_overrides.items():
        if dtype not in WEIGHT_DTYPES:
            raise EmitError(
                f"weight_dtype_overrides['{key}']: storage dtype '{dtype}' cannot be written"
                f" ({' / '.join(WEIGHT_DTYPES)})"
            )
    plan = _StoragePlan(declarations={}, scales={}, conversions={})
    if weight_dtype == "f32" and not weight_dtype_overrides:
        return plan
    eligible = eligible_compressed_initializers(graph)
    unknown = sorted(set(weight_dtype_overrides) - set(initializer_by_key))
    if unknown:
        raise EmitError(
            f"weight_dtype_overrides keys {unknown} are not the tensor of any initializer"
            " (check the FQN spelling — ignoring them silently would quietly drop the requested"
            " compression)"
        )
    for key in sorted(weight_dtype_overrides):
        if weight_dtype_overrides[key] == "f32":
            continue
        name = initializer_by_key[key]
        if name not in eligible:
            raise EmitError(
                f"initializer '{name}' ({key}): explicitly requested"
                f" '{weight_dtype_overrides[key]}' but not eligible (consumed outside a weight slot"
                " — mixed consumption is outside the eligible set of ADR 0018/0019)"
            )
    requested = {weight_dtype, *weight_dtype_overrides.values()}
    axes = weight_channel_axes(graph) if "i8" in requested else {}
    # The divisor of the eligibility check is taken from the actual shipped g (= the shape of
    # `weight_scales`) — unless it has the same source as the value `_plan_i4` declares through
    # `group_size_of`, part of a g16 asset silently stays f32.
    i4_eligible = (
        i4_eligible_initializers(graph, scales=weight_scales) if "i4" in requested else set()
    )
    reserved = set(tensors)
    for name in sorted(eligible):
        key = graph.initializers[name].tensor
        dtype = weight_dtype_overrides.get(key, weight_dtype)
        if dtype == "f32":
            continue
        tensor = tensors[key]
        if tensor.dtype is not torch.float32:
            # i32 storage (symbol-dependent constants — ADR 0010) never reaches a weight slot,
            # but if it did, the semantics would differ altogether, so it is not silently stored
            # compressed. With an explicit choice it fails loudly.
            if key in weight_dtype_overrides:
                raise EmitError(
                    f"initializer '{name}' ({key}): explicitly requested '{dtype}' but the data is"
                    f" {tensor.dtype} (compressed storage is for f32 data only)"
                )
            continue
        # i4 eligibility is **limited to weight slots of {@link I4_WEIGHT_OPS}** (ADR 0069
        # decision 5) — with i4 as the default, ineligible weights (conv2d / conv3d /
        # conv_transpose1d / conv1d with groups > 1 / weights whose stored row length does not
        # divide by the group length) quietly stay f32. The same design as i8's ineligible
        # weights staying f32, and the counterpart of the runtime side (eligible ∩ i4Eligible in
        # `planWeightResidency` of weight-residency.ts). Without this exclusion, a graph mixing in
        # convs could not be exported ("the conv weight cannot be i4") (Codex phase F finding
        # I4-ELIG-01). An explicit i4 on an ineligible weight fails loudly at `_plan_i4`'s own
        # check.
        if dtype == "i4" and key not in weight_dtype_overrides and name not in i4_eligible:
            continue
        if dtype in ("f16", "bf16"):
            declaration = IrInitializer(tensor=key, storage=IrStorage(dtype=dtype))
            conversion = _Conversion(dtype=dtype, name=name)
        else:
            if dtype == "i8":
                scale_key, scale, declaration = _plan_i8(
                    graph, reserved, name, tensor, weight_scales, axes
                )
            else:
                scale_key, scale, declaration = _plan_i4(
                    graph, reserved, name, tensor, weight_scales, i4_eligible
                )
            plan.scales[scale_key] = scale
            reserved.add(scale_key)
            conversion = _Conversion(dtype=dtype, name=name, scale=scale)
        plan.conversions[key] = conversion
        plan.declarations[name] = declaration
    if weight_dtype != "f32" and not plan.declarations:
        # ADR 0006's "do not stay silent about 0 MB eligible under a compression request" is
        # enforced on the exporter side too. What is checked is "**whether not a single
        # compressed storage was planned**" — even with zero tensors in the default dtype, the
        # intent is met if explicit choices cover every eligible tensor with another dtype (a
        # mixed storage written out per tensor is not rejected here).
        explicit = sorted({dtype for dtype in weight_dtype_overrides.values() if dtype != "f32"})
        # The cause is not pinned to one (for i4, both "no consumer with a decode path" and
        # "stored row length does not divide by the group length" land in the same branch).
        if weight_dtype == "i4":
            cause = (
                "there may be no consumer with an i4 decode path (linear / embedding / conv1d"
                " with groups==1), or the stored row lengths may not divide by the group length"
            )
        else:
            cause = (
                f"check whether {weight_dtype} was requested for a graph without"
                " weight slots of fused ops"
            )
        raise EmitError(
            f"Not a single compressed storage was planned (default {weight_dtype} / explicit"
            f" {explicit or 'none'}) — {cause}"
        )
    return plan


def _convert_for_storage(key: str, tensor: torch.Tensor, conversion: _Conversion) -> torch.Tensor:
    """Conversion to compressed storage and the eligibility checks that read real data (the heavy
    side — keeps only one tensor alive).

    A failure here leaves not a single byte under the final name — writing goes to a temporary
    path (`.partial`), and the swap happens only for a run that passed verification (the three
    steps of `karume.publish`'s module doc).
    """
    if conversion.fixed_payload is not None:
        return conversion.fixed_payload
    if conversion.dtype == "f16":
        if not _is_f16_exact(tensor):
            raise _unrounded(conversion.name, key, "contains values f16 cannot represent")
        return tensor.to(torch.float16)
    if conversion.dtype == "bf16":
        if not _is_bf16_exact(tensor):
            raise _unrounded(conversion.name, key, "contains values bf16 cannot represent")
        return tensor.to(torch.bfloat16)
    if conversion.dtype == "i8":
        quantized = quantize_to_int8(tensor, conversion.scale)
        # MUST: the inverse conversion uses **exactly** the scale fake-quant used (no
        # recomputation — ADR 0019). A mismatch here comes down to one thing: the stored values
        # and the weights the golden was taken with are different.
        if not torch.equal(quantized.to(torch.float32) * conversion.scale, tensor):
            raise _unrounded(
                conversion.name,
                key,
                "is not bit-identical after the inverse i8 × per-channel scale",
            )
        return quantized
    packed = pack_int4(quantize_to_int4(tensor, conversion.scale))
    # MUST: the inverse conversion traces back **from the bytes about to be written** (ADR 0069
    # decision 4 ③). Comparing with the quantizer's return value would let a mixed-up pack /
    # unpack (a silent wrong value — shape and type match) slip past this check.
    restored = dequantize_int4(unpack_int4(packed, tensor.shape), conversion.scale)
    if not torch.equal(restored, tensor):
        raise _unrounded(
            conversion.name, key, "is not bit-identical after the inverse i4 × group scale"
        )
    return packed


#: Storage dtype → **bits** per element (the table that derives byte counts from the IR
#: declaration alone).
#:
#: The vocabulary follows verify's `STORAGE_DTYPES` (whether it can execute is owned by
#: `verify.assert_runtime_support` alone) — this layer **only counts bytes**, so it must not be
#: unable to count a readable graph. It is in bits because packed 4-bit (2 elements per byte —
#: ADR 0069 decision 2) cannot be expressed in bytes per element.
_STORAGE_BITS = {
    "f32": 32,
    "f16": 16,
    "bf16": 16,
    "i8": 8,
    "i4": 4,
    "i4f16": 4,
    "i2": 2,
    "i32": 32,
}
#: Storage dtypes not counted as compressed (= the plain 4-byte representation). Listing the
#: compressed side would mean fixing two places for every new storage dtype, so it is written **in
#: the negative**.
_PLAIN_STORAGE_DTYPES = ("f32", "i32")
#: Bytes per element of a companion scale (fixed to F32 — ADR 0019).
_SCALE_BYTES = 4


def storage_breakdown(graph: IrGraph) -> StorageBreakdown:
    """Count the breakdown from the storage declarations (derived from the graph every time,
    without carrying real tensors around).

    An initializer's shape is a static shape without symbols by the rules of IR v1 (verify.py), so
    declared shape × the storage dtype's bit width gives the actual byte count directly. The scale
    is derived from the declaration too (real tensors are not looked at) — for i8 "channel axis
    length × 4B", for i4 "number of groups = elements / group_size × 4B".
    """
    compressed_tensors = compressed_bytes = plain_tensors = plain_bytes = scale_bytes = 0
    axes = weight_channel_axes(graph)
    # Only **the bytes written into this container** are counted — shared declarations borrow
    # the lender's bytes and are excluded (the exclusion rule lives in one place,
    # {@link bakeable_initializers}).
    for name in sorted(bakeable_initializers(graph)):
        initializer = graph.initializers[name]
        shape = [int(dim) for dim in graph.values[name].shape]
        count = 1
        for dim in shape:
            count *= dim
        dtype = initializer.storage.dtype
        nbytes = count * _STORAGE_BITS[dtype] // 8
        if dtype not in _PLAIN_STORAGE_DTYPES:
            compressed_tensors += 1
            compressed_bytes += nbytes
            if dtype == "i8":
                scale_bytes += shape[axes[name]] * _SCALE_BYTES
            elif dtype == "i2":
                scale_bytes += shape[0] * _SCALE_BYTES
            elif dtype == "i4":
                # Divisibility is guaranteed for a readable graph (verify's
                # `_check_group_quantized_shape`). Presence is optional only in the type, so it is
                # restated rather than filled with a default — filling in 1 would silently give the
                # numbers of another model, "as many scales as weights".
                group_size = initializer.storage.group_size
                if group_size is None:
                    raise EmitError(f"initializer '{name}': i4 storage without group_size")
                scale_bytes += count // group_size * _SCALE_BYTES
            elif dtype == "i4f16":
                # f16 steps, padded to a whole u32 word (the payload the container declares).
                group_size = initializer.storage.group_size
                if group_size is None:
                    raise EmitError(f"initializer '{name}': i4f16 storage without group_size")
                scale_bytes += codec_scale_bytes(CODEC_FOR_STORAGE["i4f16"], count // group_size)
        else:
            plain_tensors += 1
            plain_bytes += nbytes
    return StorageBreakdown(
        compressed_tensors=compressed_tensors,
        compressed_bytes=compressed_bytes,
        plain_tensors=plain_tensors,
        plain_bytes=plain_bytes,
        scale_bytes=scale_bytes,
    )


# ---------------------------------------------------------------------------
# Material handed to the container
# ---------------------------------------------------------------------------


class _StoredTensors(Mapping[str, Buffer]):
    """A lazy mapping tensor key → **raw bytes after storage conversion** (converts only the one
    tensor fetched).

    MUST: the same key returns **the same bytes every time** (the writer fetches twice, in the
    sha256 pass and the write pass — container-v1 §4.1). The conversion is deterministic, so this
    holds, but the moment randomness or cache invalidation enters here, "a container whose declared
    sha256 differs from its actual bytes" appears.

    MUST: converted tensors are not carried over (one compressed tensor alive at a time — the
    writer's `container._Payloads` holds just one).
    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        conversions: Mapping[str, _Conversion],
    ) -> None:
        self._tensors = tensors
        self._conversions = conversions

    def __getitem__(self, key: str) -> Buffer:
        tensor = self._tensors[key]
        conversion = self._conversions.get(key)
        if conversion is not None:
            tensor = _convert_for_storage(key, tensor, conversion)
        elif tensor.is_meta:
            raise EmitError(f"Tensor '{key}': attempting to store a meta tensor (no data)")
        # Hand over through memoryview (`tobytes()` makes a copy of the tensor — a single DiT
        # weight tensor is hundreds of MB, and peak RAM would rise by that much).
        stored = tensor.contiguous()
        if stored.dtype is torch.bfloat16:
            # numpy has no bfloat16 — reinterpret as int16 of the same 2 bytes (the bits do not
            # change; the bytes are exactly bf16's little-endian 2 bytes / element).
            stored = stored.view(torch.int16)
        return memoryview(stored.numpy()).cast("B")

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)


@dataclass(frozen=True)
class StoredModel:
    """The three items handed to the container (the module doc's "The three items handed to the
    container")."""

    #: The graph with the storage declarations committed (the `graph` passed in is not changed by
    #: a single byte).
    graph: IrGraph
    #: Tensor key → raw bytes after storage conversion (lazy, one at a time).
    tensors: Mapping[str, Buffer]
    #: Tensor key → the container's storage declaration.
    bindings: Mapping[str, Encoding]


def stored_model(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    *,
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
    fixed_weights: Mapping[str, FixedQuantizedWeight] | None = None,
) -> StoredModel:
    """Plan the storage conversion and assemble the {@link StoredModel} handed to the container
    (writes not a single byte).

    When `weight_dtype` is `"f16"` / `"i8"` / `"i4"`, **only eligible weight slots** get compressed
    storage (declaration and data are decided on one path — deciding them separately would allow
    the silent misreading "declared f16 / data f32"). i4 eligibility is limited to weight slots of
    {@link I4_WEIGHT_OPS} (linear / embedding / conv1d), and conv1d further needs `groups == 1`
    and a dividing stored row length ({@link i4_eligible_initializers} — ADR 0069 decision 5 and
    its addenda).

    `fixed_weights` is the entry point that stores fixed packed values without re-quantizing them
    (ADR 0097). The matching `tensors` are f32/meta carrying only the logical shape, and the set of
    meta keys must match it exactly. Only for rank-2 linear / embedding weights; a mix of I2 / I4
    / I8 is kept. Specifying a real f32 value as well, or mixing with the automatic quantization's
    dtype / scale / override options, is rejected.

    `weight_dtype_overrides` (tensor key → storage dtype) is an explicit per-tensor choice that
    **takes precedence** over the default `weight_dtype` (mixed storage — the meaning and where
    fail-loudly applies are in `_plan_weight_dtype`'s docstring). When mixing dtypes that need a
    scale, pass the i8 / i4 ledgers **merged** in `weight_scales` (the key spaces are FQNs and do
    not overlap, so one Mapping is enough).

    MUST: no compressed tensors are created at the planning step (the checks that read real data
    are done per tensor by {@link _StoredTensors} when it is fetched). Converting everything up
    front would keep it alive together with the caller's f32 set, and peak RAM would be the sum of
    both.

    MUST: the commit stays **inside the view** of `dataclasses.replace`, and the `graph` passed in
    is not changed by a single byte. Rewriting the caller's declarations would make paths that
    reuse the same graph (rewrite with another dtype / count the breakdown) drag the previous
    compression declarations along.
    """
    # Shared initializers (ADR 0096 stage 2) have no data in this container, so they are left out
    # of the exact declaration / storage match (the declaration's storage is written as is —
    # neither plan nor conversion applies).
    declared = {graph.initializers[name].tensor for name in bakeable_initializers(graph)}
    stored = set(tensors)
    if declared != stored:
        raise EmitError(
            "Declared tensors and stored tensors do not match: "
            f"missing {sorted(declared - stored)} / extra {sorted(stored - declared)}"
        )
    fixed = {} if fixed_weights is None else fixed_weights
    meta_keys = {key for key, value in tensors.items() if value.is_meta}
    if meta_keys != set(fixed):
        raise EmitError("The key sets of f32/meta logical weights and fixed_weights do not match")
    if fixed and (weight_dtype != "f32" or weight_scales or weight_dtype_overrides):
        raise EmitError("fixed_weights cannot be combined with automatic quantization options")
    contiguous = {key: value.detach().contiguous() for key, value in tensors.items()}
    plan = (
        _plan_fixed_weights(graph, contiguous, fixed)
        if fixed
        else _plan_weight_dtype(
            graph, contiguous, weight_dtype, weight_scales or {}, weight_dtype_overrides or {}
        )
    )
    committed = replace(graph, initializers={**graph.initializers, **plan.declarations})
    source = {**contiguous, **plan.scales}
    return StoredModel(
        graph=committed,
        tensors=_StoredTensors(source, plan.conversions),
        bindings=container_bindings(committed),
    )
