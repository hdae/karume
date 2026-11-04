"""Writer of the container format (`krm` / `krg`), and a reader for its self-check —
docs/container-v1.md.

There are two sources of truth: the physical format, descriptor, binding table and codec ledger
are in `docs/container-v1.md`; the graph JSON and **canonical serialization** are in
`docs/ir-v2.md`. As an implementation, correctness means that **the TS reader**
(`packages/runtime/src/format/container/*.ts`) can open the output; this is its mirror
(`packages/runtime/tests/helpers/container-write.ts` is a test reference implementation of the same
rules).

The rules this layer keeps (each one leans toward "never approximate silently"):

- **Canonical serialization**: no whitespace, name maps in code point order, numbers spelled by
  ECMAScript `Number::toString`. The same input gives a **byte-identical** descriptor (the
  condition for deciding `krg` identity by content hash — container-v1 §9 / §13.5).
- **block**: start aligned to 64 B, length a multiple of 4 (tail padding is **baked by the
  writer**), at most the limit. Padding cannot be put on a middle piece (it would overwrite the
  start of the next piece), so cuts are made on a row step rounded to `4/gcd(rowBytes, 4)`.
- **part**: part 0 = header + 2 documents, part 1 = const region (declared even when empty), part 2
  onward = weights. No padding is inserted before a part of length 0 (inserting it would shift
  `krg` extraction by bytes).
- **Memory**: tensors are looked up one at a time and released one at a time. The whole is never
  held in memory.

MUST: writing has 3 steps: "① decide the layout ② stream the bytes one tensor at a time and take
the block / part sha256 ③ build and write the 2 documents". Part 0 holds the sha256 of the parts
after it. In split form part 0 is a separate file, so ② finishes writing the data parts and the
bytes are looked up only once. In single form "the document placed first depends on the contents
at the end" — either a temporary file goes in between or the bytes are looked up twice, and the
latter is taken (the bytes are **looked up twice, in ② and ③** — a temporary file would add one
write of the whole model).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

from karume.ir import IrGraph
from karume.ops import WEIGHT_CHANNEL_AXES, WEIGHT_SLOTS, integral_float_to_int

_MIB = 1024 * 1024

#: Fixed length of the header (§1): `[magic 4][u32 version][u64 graph descriptor length][u64 model
#: descriptor length]`.
HEADER_BYTES = 24

#: The container version this repository writes / reads (§1).
CONTAINER_VERSION = 1

#: Magic of the model container / graph container (4 ASCII characters — §1; only the magic
#: carries the kind).
MAGIC_MODEL = b"KRMC"
MAGIC_GRAPH = b"KRGC"

#: Alignment of a block start and of part concatenation boundaries (§4.1 / §8).
BLOCK_START_ALIGN = 64

#: Alignment of the block length (§4.1 — rule ② of the binding table).
BLOCK_TAIL_ALIGN = 4

#: Byte value of the padding (§4.1 — fixed at 0x00).
PAD_BYTE = b"\x00"

#: Upper limit of the block length (§4.1 — **at most** 32 MiB; an independent constant, not
#: derived from the part length).
BLOCK_MAX_BYTES = 32 * _MIB

#: Part lengths the writer can choose (part 2 onward — §4.2).
PART_LENGTH_CHOICES = (256 * _MIB, 512 * _MIB, 768 * _MIB, 1024 * _MIB)

#: Default part length (§4.2 — keeping 256 MiB was ruled at the step 3 acceptance; ADR 0108
#: addendum 5).
DEFAULT_PART_BYTES = PART_LENGTH_CHOICES[0]

#: Ceiling of the part length (every part including part 0 / 1 — §10).
PART_MAX_BYTES = 1024 * _MIB

#: Upper limit of the part count (§10 — inherits the old `MAX_SHARDS`).
MAX_PARTS = 1024

#: Upper limit of the block count of one container (the const table plus the model table — §10).
MAX_BLOCKS = 65_536

#: Upper limit of the graph count of one container (§10).
MAX_GRAPHS = 64

#: Upper limit of the byte length of the graph descriptor and of the model descriptor, each (§10).
MAX_DESCRIPTOR_BYTES = 32 * _MIB

#: Upper limit of the nesting depth of a descriptor / graph JSON (§10).
MAX_JSON_DEPTH = 64

# The 3 vocabularies below MUST be matched with `.fullmatch` — Python's `$` also matches just
# before a trailing newline, so `.match` would accept a spelling with a newline, and the accepted
# set would drift from the TS reader's (JS `$`).

#: Vocabulary of graph names (§2.1).
GRAPH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

#: Vocabulary of block ids (§0).
BLOCK_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

#: Spelling of a sha256 field (§0 — 64 lowercase hex characters).
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: Tensor key prefix that marks an initializer bound for the const region (the naming of
#: `convert._add_const`).
CONST_KEY_PREFIX = "const."

#: Block role (§2.2). `"const"` appears only on the const table side.
BlockRole = Literal["weight", "scale", "zero-point", "asset"]


class AssetInput(NamedTuple):
    """The entry for one asset (§2.2) — `(role, logical length, payload[, dedicated part])`.

    `role` is **the name of the interpreter on the models side** (`ple-index` / `rope-base` …);
    the runtime does not interpret it. `length` is the payload's byte count (logical length), and
    **the declaration comes first** — the block length is this rounded up to a multiple of
    {@link BLOCK_TAIL_ALIGN}, and the consumer does not strip trailing 0x00 by guessing (ADR 0109
    decision 4).

    `payload` is raw bytes, or a call that builds the bytes only when it is called. The latter
    MUST **return the same byte sequence every time it is called** (the writer calls it twice, once
    to take the sha256 and once to write, and rejects a length that differs from the declaration).

    `dedicated_part` marks **an asset that needs range reads** (PLE `values` / `scales`); its block
    MUST sit alone in a dedicated part (container-v1 §4.2) — whole-read assets (the index,
    `rope_base`) may share one part with each other.
    """

    role: str
    length: int
    payload: Buffer | Callable[[], Buffer]
    dedicated_part: bool = False


#: Version of the IR v2 graph JSON (docs/ir-v2.md).
IR_V2_FORMAT = "karume-ir"
IR_V2_VERSION = 2

#: Step for digesting a part on re-read (a part can be up to 1 GiB — the whole is never held in a
#: buffer).
_READ_CHUNK_BYTES = 8 * _MIB


class ContainerFormatError(Exception):
    """The one exception that collects violations of the container format (unsupported,
    inconsistent, corrupt)."""


# ---------------------------------------------------------------------------
# Canonical JSON (docs/ir-v2.md "Canonical serialization")
# ---------------------------------------------------------------------------

_REPR_PATTERN = re.compile(r"^(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$")

#: Boundaries at which ECMAScript `Number::toString` switches to exponent notation (steps 5 / 6 / 7
#: of §6.1.6.1.20).
_ECMA_EXPONENT_CEILING = 21
_ECMA_EXPONENT_FLOOR = -6

#: Upper limit of the absolute value of an integer that round-trips as a JSON number (an
#: ECMAScript Number is a double).
_MAX_SAFE_INTEGER = 2**53 - 1


def _format_double(value: float) -> str:
    """The spelling of ECMAScript `Number::toString` (docs/ir-v2.md "Canonical serialization").

    Python's `repr` gives the shortest round-trip digits but places the exponent differently
    (`repr(1e-6)` = `1e-06`, `repr(10000.0)` = `10000.0`). The digits are used as is, and **only
    the exponent placement** is changed to ECMAScript's rules (integral values get neither a
    decimal point nor an exponent / an exponent appears only when `|x| < 1e-6` or `|x| >= 1e21` /
    no zero padding of the exponent / `+` on a positive exponent).
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise ContainerFormatError(f"A non-finite number cannot be written to JSON: {value!r}")
    if value == 0.0:
        # -0.0 is "0" too (no two spellings).
        return "0"
    sign = "-" if value < 0 else ""
    matched = _REPR_PATTERN.match(repr(abs(value)))
    if matched is None:  # pragma: no cover - repr always has this form
        raise ContainerFormatError(f"Cannot read the repr of a float: {value!r}")
    integer_part = matched.group(1)
    fraction_part = matched.group(2) or ""
    exponent = matched.group(3)
    digits = integer_part + fraction_part
    # value = 0.<digits> × 10^point
    point = len(integer_part) + (int(exponent) if exponent else 0)
    without_leading = digits.lstrip("0")
    point -= len(digits) - len(without_leading)
    digits = without_leading.rstrip("0")
    count = len(digits)
    if count <= point <= _ECMA_EXPONENT_CEILING:
        return sign + digits + "0" * (point - count)
    if 0 < point <= _ECMA_EXPONENT_CEILING:
        return sign + digits[:point] + "." + digits[point:]
    if _ECMA_EXPONENT_FLOOR < point <= 0:
        return sign + "0." + "0" * (-point) + digits
    power = point - 1
    mantissa = digits if count == 1 else digits[0] + "." + digits[1:]
    return f"{sign}{mantissa}e{'+' if power >= 0 else '-'}{abs(power)}"


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        if -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            return str(value)
        # A value beyond the safe integers is rounded to a double in ECMAScript. Write the rounded
        # spelling (do not let only the Python side carry extra digits = reader and writer
        # disagreeing on the value).
        return _format_double(float(value))
    return _format_double(value)


def _write_json(value: Any, out: list[str], depth: int) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ContainerFormatError(f"JSON nesting exceeded the depth limit {MAX_JSON_DEPTH}")
    if value is None:
        out.append("null")
        return
    if value is True:
        out.append("true")
        return
    if value is False:
        out.append("false")
        return
    if isinstance(value, int | float):
        out.append(_format_number(value))
        return
    if isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=False))
        return
    if isinstance(value, list | tuple):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write_json(item, out, depth + 1)
        out.append("]")
        return
    if isinstance(value, Mapping):
        out.append("{")
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                raise ContainerFormatError(f"JSON key is not a string: {key!r}")
            if key == "__proto__":
                # The reader (TS) rejects this key (assigning it into a plain `{}` turns into a
                # [[Prototype]] set and creates no own property). Do not create what can be
                # written but not read.
                raise ContainerFormatError("JSON key '__proto__' cannot be written")
            if index:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _write_json(item, out, depth + 1)
        out.append("}")
        return
    raise ContainerFormatError(f"Value cannot be represented as JSON: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Canonical JSON string (no whitespace, `ensure_ascii=False`, NaN / Inf forbidden).

    **No reordering** — both the fixed-schema key order and the code point order of name maps are
    decided by the side that builds the value ({@link ir_v2_document} / descriptor assembly). This
    is a serializer that only "spells out the decided order"; mixing the two responsibilities would
    make it impossible to tell in a test "which order took effect".
    """
    out: list[str] = []
    _write_json(value, out, 0)
    return "".join(out)


def _canonical_free_form(value: Any) -> Any:
    """Canonicalize a free-form JSON value (a node's `attrs`) recursively — map keys in code point
    order."""
    if isinstance(value, Mapping):
        return {key: _canonical_free_form(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_canonical_free_form(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Codec ledger (§6.2 / §6.3 — equal to TS `format/container/codecs.ts`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodecPacking:
    """The source of truth for the declared byte length and alignment (the bit count is the
    derived value `block_bytes * 8 / block_elements`)."""

    block_elements: int
    block_bytes: int
    align_bytes: int


@dataclass(frozen=True)
class CodecEntry:
    #: Kind of decode path (derived from the codec — the consumer branches on it). The spelling is
    #: the same vocabulary as the old `storage.dtype`, and `ternary` shares the same i2 path as
    #: `int2-off` (§6.2 / ADR 0108 decision 13).
    layout: Literal["f32", "f16", "bf16", "i32", "i8", "i4", "i2"]
    packing: CodecPacking
    #: `"required"` = quantized (`rowAxis` / `groupSize` / `scale` required), `"forbidden"` = not
    #: quantized.
    scale: Literal["required", "forbidden"]
    #: `"channel"` = per-channel (`groupSize` is the row length), `"group"` = a group length that is
    #: a power of 2 ≥ 16.
    grouping: Literal["channel", "group"] | None
    #: The set of `rowAxis` values that can be written (empty when not quantized). Only int8-sym has
    #: `1` (the conv_transpose1d shape) — the i4 / i2 / ternary decode paths only have row-wise
    #: quantization (§6.3).
    row_axes: tuple[int, ...]


def _raw(layout: Literal["f32", "f16", "bf16", "i32"], block_bytes: int) -> CodecEntry:
    return CodecEntry(layout, CodecPacking(1, block_bytes, 4), "forbidden", None, ())


#: The first-edition ledger (§6.3). Registered names are **baked into assets**, so the values of
#: existing entries do not move.
#:
#: MUST: **both sides hold** the same table as the TS side
#: (`packages/runtime/src/format/container/codecs.ts`). It is a cross-package invariant rather
#: than a derivable fact, so the cross-check lives in a test (the ledger cross-check in
#: `tests/test_container.py`).
CODEC_LEDGER: Mapping[str, CodecEntry] = {
    "f32": _raw("f32", 4),
    "f16": _raw("f16", 2),
    "bf16": _raw("bf16", 2),
    "i32": _raw("i32", 4),
    "int8-sym": CodecEntry("i8", CodecPacking(1, 1, 4), "required", "channel", (0, 1)),
    "int4-sym-g": CodecEntry("i4", CodecPacking(8, 4, 4), "required", "group", (0,)),
    "int2-off": CodecEntry("i2", CodecPacking(16, 4, 4), "required", "channel", (0,)),
    "ternary": CodecEntry("i2", CodecPacking(16, 4, 4), "required", "channel", (0,)),
}

#: Lower limit of the group length of group quantization (ADR 0069 decision 2).
MIN_GROUP_SIZE = 16

#: Accepted set of scale dtypes (only f32 in the first edition — §6.1).
SCALE_DTYPES = ("f32",)

#: Old `storage.dtype` → registered ledger name (the "old `storage.dtype`" row of §6.3).
#:
#: MUST: nothing maps to `ternary` — even when the value range is a subset of `int2-off`, the claim
#: "this is ternary" is the quantizer's to make, and the old assets' i2 actually has `q = -2` in
#: around 6 % of elements (§6.3).
CODEC_FOR_STORAGE: Mapping[str, str] = {
    "f32": "f32",
    "f16": "f16",
    "bf16": "bf16",
    "i32": "i32",
    "i8": "int8-sym",
    "i4": "int4-sym-g",
    "i2": "int2-off",
}


def codec_entry(codec: str) -> CodecEntry:
    entry = CODEC_LEDGER.get(codec)
    if entry is None:
        raise ContainerFormatError(
            f"Codec ledger has no '{codec}' (ledger: {' / '.join(sorted(CODEC_LEDGER))})"
        )
    return entry


def assert_row_axis(codec: str, row_axis: int, where: str) -> None:
    """Whether `rowAxis` is an axis the codec's ledger entry allows (the writer and the merge layer
    go through the same single path).

    MUST: the general "0 or 1" is not enough — it would let `rowAxis: 1` be written for a
    square-shaped i4, making a container that only the runtime's `bindGraphs` rejects (the same
    accepted set as `rowAxes` of the TS ledger).
    """
    allowed = codec_entry(codec).row_axes
    if row_axis not in allowed:
        raise ContainerFormatError(
            f"{where}: rowAxis of codec '{codec}' must be {' / '.join(map(str, allowed))}"
            f" (declared {row_axis} — container-v1 §6.3)"
        )


def per_channel_group_size(row_length: int) -> int:
    """`groupSize` of a per-channel codec (= the row length).

    A zero-element degenerate shape (`in_features = 0` and the like) has row length 0, but
    `groupSize` MUST be 1 or more, so it is rounded to 1 ({@link group_count} brings the group count
    back to 1). Mirrors `codecs.ts` on the TS side.
    """
    return max(row_length, 1)


def group_count(row_length: int, group_size: int) -> int:
    """Group count of a scale, `row length / groupSize` (§6.1). A degenerate row length of 0 has
    group count 1."""
    return 1 if row_length == 0 else row_length // group_size


def payload_bytes(codec: str, numel: int, where: str) -> int:
    """Byte length of a payload (the formula of §6.1). A fractional packing block cannot be made."""
    packing = codec_entry(codec).packing
    if numel % packing.block_elements != 0:
        raise ContainerFormatError(
            f"{where}: element count {numel} is not divisible by the packing of codec '{codec}'"
            f" ({packing.block_elements} elements / block)"
        )
    return numel // packing.block_elements * packing.block_bytes


# ---------------------------------------------------------------------------
# IR v2 document (container-v1 §13.1 — the initializer name is the key of the body)
# ---------------------------------------------------------------------------


def _rename_map(graph: IrGraph) -> dict[str, str]:
    """IR v1 initializer name → v2 name (= tensor key).

    A v1 name is a torch.export placeholder name, and the upstream key was held separately by the
    `tensor` field. In v2 the name itself is the key of the body, so it is renamed to `tensor` (the
    lender's key for a shared declaration).
    """
    rename: dict[str, str] = {}
    for name, initializer in graph.initializers.items():
        if initializer.shared is not None:
            target = initializer.shared.tensor
        elif initializer.tensor is not None:
            target = initializer.tensor
        else:
            raise ContainerFormatError(
                f"initializer '{name}': neither `tensor` nor `shared` is present (invalid as IR v1)"
            )
        if not target:
            raise ContainerFormatError(f"initializer '{name}': the tensor key is an empty string")
        rename[name] = target
    return rename


def _assert_no_rename_collision(graph: IrGraph, rename: Mapping[str, str]) -> None:
    """Renaming does not overwrite a value name (overwriting would silently drop one
    declaration)."""
    occupied = {spec.name: "inputs" for spec in graph.inputs}
    for name in graph.values:
        if name not in rename:
            occupied[name] = "values"
    by_target: dict[str, str] = {}
    for name, target in rename.items():
        owner = occupied.get(target)
        if owner is not None:
            raise ContainerFormatError(
                f"initializer '{name}': rename target '{target}' collides with an existing value"
                f" name ({owner})"
            )
        previous = by_target.get(target)
        if previous is not None:
            raise ContainerFormatError(
                f"initializer '{name}' and '{previous}': rename target '{target}' collides"
            )
        by_target[target] = name


def ir_v2_document(graph: IrGraph) -> dict[str, Any]:
    """IR v1 {@link IrGraph} → IR v2 graph JSON in **canonical order** (docs/ir-v2.md).

    Three differences: ① initializers are renamed to tensor keys (also the `values` keys,
    `nodes[].ins` and `outputs`) ② `storage` is dropped, giving `{}` / `{"shared": true}`
    ③ `version` is 2.

    Keys follow the order of the spec's example; maps keyed by name (`initializers` / `values` /
    `states` / `attrs` / a node's `states`) and sets whose order carries no meaning
    (`requires.ops` / `symbols`) are in code point order; an empty `states`, a node's empty
    `states` and `external: false` are not written.
    """
    rename = _rename_map(graph)
    _assert_no_rename_collision(graph, rename)

    def renamed(name: str) -> str:
        return rename.get(name, name)

    document: dict[str, Any] = {
        "format": IR_V2_FORMAT,
        "version": IR_V2_VERSION,
        "requires": {"ops": sorted(graph.required_ops)},
        "symbols": sorted(graph.symbols),
        "inputs": [spec.to_dict() for spec in graph.inputs],
        "outputs": [renamed(name) for name in graph.outputs],
        "initializers": {
            rename[name]: ({"shared": True} if initializer.shared is not None else {})
            for name, initializer in sorted(
                graph.initializers.items(), key=lambda item: rename[item[0]]
            )
        },
        "values": {
            new: {"dtype": value.dtype, "shape": list(value.shape)}
            for new, value in sorted(
                ((renamed(name), value) for name, value in graph.values.items()),
                key=lambda item: item[0],
            )
        },
    }
    if graph.states:
        document["states"] = {name: graph.states[name].to_dict() for name in sorted(graph.states)}
    document["nodes"] = [
        {
            "op": node.op,
            "ins": [renamed(name) for name in node.ins],
            "outs": list(node.outs),
            "attrs": _canonical_free_form(node.attrs),
            **(
                {"states": {key: node.states[key] for key in sorted(node.states)}}
                if node.states
                else {}
            ),
        }
        for node in graph.nodes
    ]
    return document


# ---------------------------------------------------------------------------
# Descriptor types (§2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoding:
    """The storage the writer is told to use (per tensor key). The material for the descriptor's
    `encoding`.

    `scale_key` is **the tensor key of the companion scale** (looked up from `tensors` through the
    same entry); what goes into the descriptor is the id of the block that holds it.
    """

    codec: str
    group_size: int | None = None
    row_axis: int | None = None
    scale_key: str | None = None


#: A piece list (§5): `(block id, (begin row, end row))` in row order.
Pieces = tuple[tuple[str, tuple[int, int]], ...]


def joined_scale_id(block_ids: Sequence[str]) -> str:
    """The id under which a split scale is read whole: its pieces' block ids joined with `+`
    (ADR 0108 addendum 15).

    `+` is outside the block-id vocabulary (§0), so a joined id never names a declared block. The
    TS mirror is `joinedScaleId` in `packages/runtime/src/format/container/bind.ts`.
    """
    return "+".join(block_ids)


@dataclass(frozen=True)
class BlockEncoding:
    """The storage declaration that goes into the descriptor (§6.1). `packing` is a copy of the
    ledger, so it is not held as a field.

    A quantized codec's scale is either one block (`scale_block`) or, when one block cannot hold
    it, a piece list split on the data's pieces (`scale_pieces` — §5 rule ③, ADR 0108 addendum
    15); exactly one of the two is set.
    """

    codec: str
    row_axis: int | None = None
    group_size: int | None = None
    scale_block: str | None = None
    scale_pieces: Pieces | None = None

    @property
    def scale_blocks(self) -> tuple[str, ...]:
        """The scale's block ids (one, or the pieces of a split scale; empty without a scale)."""
        if self.scale_pieces is not None:
            return tuple(block for block, _ in self.scale_pieces)
        return () if self.scale_block is None else (self.scale_block,)

    def to_document(self) -> dict[str, Any]:
        entry = codec_entry(self.codec)
        packing = entry.packing
        document: dict[str, Any] = {
            "codec": self.codec,
            "packing": {
                "blockElements": packing.block_elements,
                "blockBytes": packing.block_bytes,
                "alignBytes": packing.align_bytes,
            },
        }
        quantized = entry.scale == "required"
        scale = self.scale_blocks or None
        present = (self.row_axis, self.group_size, scale)
        if not quantized:
            if any(field is not None for field in present):
                raise ContainerFormatError(
                    f"codec '{self.codec}' is not quantized, so"
                    " rowAxis / groupSize / scale cannot be written"
                )
            return document
        if any(field is None for field in present):
            raise ContainerFormatError(
                f"codec '{self.codec}' is quantized, so rowAxis / groupSize / scale are required"
            )
        if self.scale_block is not None and self.scale_pieces is not None:
            raise ContainerFormatError("A scale has exactly one of `block` and `pieces`")
        document["rowAxis"] = self.row_axis
        document["groupSize"] = self.group_size
        if self.scale_pieces is None:
            document["scale"] = {"block": self.scale_block, "dtype": SCALE_DTYPES[0]}
        else:
            document["scale"] = {
                "pieces": _pieces_document(self.scale_pieces, "scale"),
                "dtype": SCALE_DTYPES[0],
            }
        return document


def _pieces_document(pieces: Pieces, what: str) -> list[dict[str, Any]]:
    """A piece list → the descriptor's `pieces` (2 or more — one is written as `block`)."""
    if len(pieces) < 2:
        raise ContainerFormatError(
            f"The {what} `pieces` must have 2 or more entries (write one as `block`)"
        )
    return [{"block": block, "rows": [rows[0], rows[1]]} for block, rows in pieces]


@dataclass(frozen=True)
class ConstBlockRecord:
    """A block of the const region (§2.1 — offset is relative to the start of the const
    region)."""

    id: str
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class ConstantBinding:
    graph: str
    initializer: str
    block: str
    encoding: BlockEncoding


@dataclass(frozen=True)
class DataBlockRecord:
    """A block of part 2 onward (§2.2 — offset is relative to the start of the part it belongs
    to)."""

    id: str
    part: int
    offset: int
    length: int
    sha256: str
    role: BlockRole


@dataclass(frozen=True)
class PartRecord:
    index: int
    length: int
    sha256: str


@dataclass(frozen=True)
class AssetRecord:
    """One entry of the model descriptor's `assets` (§2.2 — name → block + role + logical length).

    `length` is the payload's byte count, and the length of `block` MUST be **this rounded up to a
    multiple of {@link BLOCK_TAIL_ALIGN}** (the declaration covers even the amount of padding).
    """

    block: str
    role: str
    length: int


@dataclass(frozen=True)
class WeightSupply:
    """The supply form of the binding table (§5). `block` (one whole block) and `pieces` (2 or
    more) are exclusive."""

    encoding: BlockEncoding
    block: str | None = None
    pieces: Pieces | None = None

    def to_document(self) -> dict[str, Any]:
        if (self.block is None) == (self.pieces is None):
            raise ContainerFormatError("A supply form has exactly one of `block` and `pieces`")
        if self.pieces is not None:
            supply: dict[str, Any] = {"pieces": _pieces_document(self.pieces, "supply")}
        else:
            supply = {"block": self.block}
        supply["encoding"] = self.encoding.to_document()
        return supply


@dataclass(frozen=True)
class Provenance:
    """Provenance (§2.3 — no licence text is included)."""

    license: str
    notice: str | None = None
    upstream_revision: str | None = None
    writer: str | None = None

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {"license": self.license}
        for key, value in (
            ("notice", self.notice),
            ("upstreamRevision", self.upstream_revision),
            ("writer", self.writer),
        ):
            if value is not None:
                document[key] = value
        return document


@dataclass(frozen=True)
class GraphDescriptor:
    """Graph descriptor (§2.1). **Byte-identical** in `krm` and `krg`."""

    graphs: Mapping[str, Mapping[str, Any]]
    const_length: int
    const_blocks: tuple[ConstBlockRecord, ...]
    constants: tuple[ConstantBinding, ...]

    def to_document(self) -> dict[str, Any]:
        ops = sorted({op for graph in self.graphs.values() for op in graph["requires"]["ops"]})
        return {
            "format": "karume-container",
            "version": CONTAINER_VERSION,
            "capabilities": {"ops": ops, "features": []},
            "graphs": {name: self.graphs[name] for name in sorted(self.graphs)},
            "const": {
                "length": self.const_length,
                "blocks": [
                    {
                        "id": block.id,
                        "offset": block.offset,
                        "length": block.length,
                        "sha256": block.sha256,
                    }
                    for block in sorted(self.const_blocks, key=lambda block: block.offset)
                ],
                "constants": [
                    {
                        "graph": entry.graph,
                        "initializer": entry.initializer,
                        "block": entry.block,
                        "encoding": entry.encoding.to_document(),
                    }
                    for entry in sorted(
                        self.constants, key=lambda entry: (entry.graph, entry.initializer)
                    )
                ],
            },
        }


@dataclass(frozen=True)
class ModelDescriptor:
    """Model descriptor (§2.2). Absent in `krg`."""

    parts: tuple[PartRecord, ...]
    blocks: tuple[DataBlockRecord, ...]
    binding: Mapping[str, Mapping[str, WeightSupply]]
    assets: Mapping[str, AssetRecord]
    provenance: Provenance

    def to_document(self) -> dict[str, Any]:
        codecs = sorted(
            {
                supply.encoding.codec
                for supplies in self.binding.values()
                for supply in supplies.values()
            }
        )
        return {
            "format": "karume-model",
            "version": CONTAINER_VERSION,
            "codecs": codecs,
            "parts": [
                {"index": part.index, "length": part.length, "sha256": part.sha256}
                for part in sorted(self.parts, key=lambda part: part.index)
            ],
            "blocks": [
                {
                    "id": block.id,
                    "part": block.part,
                    "offset": block.offset,
                    "length": block.length,
                    "sha256": block.sha256,
                    "role": block.role,
                }
                for block in sorted(self.blocks, key=lambda block: (block.part, block.offset))
            ],
            "binding": {
                graph: {
                    name: self.binding[graph][name].to_document()
                    for name in sorted(self.binding[graph])
                }
                for graph in sorted(self.binding)
            },
            "assets": {
                name: {
                    "block": self.assets[name].block,
                    "role": self.assets[name].role,
                    "length": self.assets[name].length,
                }
                for name in sorted(self.assets)
            },
            "provenance": self.provenance.to_document(),
        }


def serialize_graph_descriptor(descriptor: GraphDescriptor) -> bytes:
    return _encode_descriptor(descriptor.to_document(), "graph descriptor")


def serialize_model_descriptor(descriptor: ModelDescriptor) -> bytes:
    return _encode_descriptor(descriptor.to_document(), "model descriptor")


def _encode_descriptor(document: Mapping[str, Any], label: str) -> bytes:
    encoded = canonical_json(document).encode("utf-8")
    if len(encoded) > MAX_DESCRIPTOR_BYTES:
        raise ContainerFormatError(
            f"The {label} exceeds the limit of {MAX_DESCRIPTOR_BYTES} bytes: {len(encoded)}"
        )
    return encoded


# ---------------------------------------------------------------------------
# IR `storage` declarations → binding table (the mapping of §6.3)
# ---------------------------------------------------------------------------
#
# MUST: this mapping lives **on the side that does not need torch**. It is the single path both
# the writer (`karume.emit.stored_model`) and the migration CLI (`karume.migrate`) go through, so
# placing it in the storage conversion layer (torch-dependent) would pull torch into the migration
# CLI's import graph (a 1 GB-class dependency on a path that only reads old shards).


def bakeable_initializers(graph: IrGraph) -> set[str]:
    """Names of the initializers whose **bytes are written** into this container (the set without
    shared declarations).

    A shared initializer (ADR 0096 stage 2) only borrows the lender's bytes, so storage planning,
    eligibility checks and the cross-check of declarations against bytes all traverse this set
    (kept in one place — copying the exclusion everywhere would let only a newly added traversal
    hit `tensor = None`).
    """
    return {name for name, init in graph.initializers.items() if not init.is_shared}


def weight_channel_axes(graph: IrGraph) -> dict[str, int]:
    """Initializer consumed in a weight slot → per-channel axis (mirrors
    `packages/runtime/src/runtime/plan.ts`).

    The axis is taken from **the consuming op** (the weight shape alone cannot tell linear
    `[out,in]` from conv_transpose1d `[Cin,Cout,K]`). When ops with different axes consume the same
    initializer, it cannot be decided as one, so it fails loudly.
    """
    bakeable = bakeable_initializers(graph)
    axes: dict[str, int] = {}
    for node in graph.nodes:
        slot = WEIGHT_SLOTS.get(node.op)
        if slot is None or slot >= len(node.ins):
            continue
        name = node.ins[slot]
        if name not in bakeable:
            continue
        axis = WEIGHT_CHANNEL_AXES[node.op]
        if axes.setdefault(name, axis) != axis:
            raise ContainerFormatError(
                f"initializer '{name}': the per-channel axis differs between consuming ops"
                f" ({axes[name]} and {axis}) — one scale cannot represent it"
            )
    return axes


def concrete_shape(graph: IrGraph, name: str, where: str) -> list[int]:
    """Declared shape of an initializer (bytes with a symbolic dimension cannot exist).

    The migration CLI (`karume.migrate._assert_scale_layouts`) needs the same decision, so **the
    same single path** as the binding derivation ({@link container_bindings}) is on the public
    surface. Holding a copy would allow "the writer fails but migration passes" on the day the
    handling of symbolic dimensions moves on only one side.
    """
    value = graph.values.get(name)
    if value is None:
        raise ContainerFormatError(f"{where}: no declaration in `values`")
    shape: list[int] = []
    for dim in value.shape:
        if not isinstance(dim, int):
            raise ContainerFormatError(
                f"{where}: the initializer's shape has a symbolic dimension ({dim!r})"
            )
        shape.append(dim)
    return shape


def container_bindings(graph: IrGraph) -> dict[str, Encoding]:
    """IR `storage` declarations → container bindings (**tensor key** → {@link Encoding}).

    MUST: the writer ({@link karume.emit.stored_model}) and the migration CLI (`karume.migrate`)
    derive bindings from **this single path**. Deriving them on two paths could produce, from the
    same asset, silently wrong values whose shape and type both fit, such as "declared i4 / bytes
    i8" (the mapping rules of container-v1 §12).

    A shared initializer (a declaration without bytes) has no binding. `rowAxis` is taken from
    **the consuming op** ({@link weight_channel_axes}), and a per-channel `groupSize` is the row
    length (= numel / rows). The scale of a group codec (`int4-sym-g`) is baked with the leading
    dimension as rows, so a shape whose consuming op requires axis 1 cannot be mapped (fails
    loudly — silently declaring it as axis 0 would swap the values).
    """
    axes = weight_channel_axes(graph)
    bindings: dict[str, Encoding] = {}
    for name, initializer in graph.initializers.items():
        if initializer.is_shared:
            continue
        where = f"initializer '{name}'"
        key = initializer.tensor
        if key is None:
            raise ContainerFormatError(
                f"{where}: neither `tensor` nor `shared` is present (invalid as IR)"
            )
        storage = initializer.storage
        codec = CODEC_FOR_STORAGE.get(storage.dtype)
        if codec is None:
            raise ContainerFormatError(
                f"{where}: storage dtype '{storage.dtype}' has no mapping target in the codec"
                " ledger"
            )
        entry = codec_entry(codec)
        if entry.scale == "forbidden":
            bindings[key] = Encoding(codec)
            continue
        if storage.scale is None:
            raise ContainerFormatError(
                f"{where}: quantized storage '{storage.dtype}' but no scale is declared"
            )
        shape = concrete_shape(graph, name, where)
        row_axis = axes.get(name, 0)
        if len(shape) <= row_axis:
            raise ContainerFormatError(
                f"{where}: for rowAxis {row_axis} the declared shape {shape} is too shallow"
            )
        if entry.grouping == "group":
            if storage.group_size is None:
                raise ContainerFormatError(
                    f"{where}: group quantization but no group_size is declared"
                )
            if row_axis != 0:
                raise ContainerFormatError(
                    f"{where}: the consuming op's per-channel axis is {row_axis}, but a group"
                    " scale is baked with the leading dimension as rows (no shape to map to)"
                )
            group_size = storage.group_size
        else:
            row_count = shape[row_axis]
            group_size = per_channel_group_size(math.prod(shape) // row_count if row_count else 0)
        bindings[key] = Encoding(
            codec, group_size=group_size, row_axis=row_axis, scale_key=storage.scale
        )
    return bindings


# ---------------------------------------------------------------------------
# Header and part layout (§1 / §8 / §9)
# ---------------------------------------------------------------------------


def align_up(value: int, align: int) -> int:
    if value < 0:
        raise ContainerFormatError(f"Value to align is not non-negative: {value}")
    return -(-value // align) * align


def write_header(*, kind: Literal["model", "graph"], graph_length: int, model_length: int) -> bytes:
    if kind == "graph" and model_length != 0:
        raise ContainerFormatError("krg cannot have a model descriptor (its length MUST be 0)")
    if kind == "model" and model_length == 0:
        raise ContainerFormatError("krm always has a model descriptor (length 0 is the krg form)")
    if graph_length == 0:
        raise ContainerFormatError("The graph descriptor length is 0")
    magic = MAGIC_MODEL if kind == "model" else MAGIC_GRAPH
    return (
        magic
        + CONTAINER_VERSION.to_bytes(4, "little")
        + graph_length.to_bytes(8, "little")
        + model_length.to_bytes(8, "little")
    )


@dataclass(frozen=True)
class ContainerHeader:
    kind: Literal["model", "graph"]
    version: int
    graph_length: int
    model_length: int

    @property
    def part0_length(self) -> int:
        return HEADER_BYTES + self.graph_length + self.model_length


def read_header(raw: bytes) -> ContainerHeader:
    if len(raw) < HEADER_BYTES:
        raise ContainerFormatError(f"The header needs {HEADER_BYTES} bytes but has only {len(raw)}")
    magic = bytes(raw[0:4])
    if magic == MAGIC_MODEL:
        kind: Literal["model", "graph"] = "model"
    elif magic == MAGIC_GRAPH:
        kind = "graph"
    else:
        raise ContainerFormatError(
            f"Unknown magic: {magic.hex(' ')}"
            f" (a Karume container is {MAGIC_MODEL!r} / {MAGIC_GRAPH!r})"
        )
    version = int.from_bytes(raw[4:8], "little")
    if version != CONTAINER_VERSION:
        raise ContainerFormatError(
            f"Unsupported container version {version} (supported: {CONTAINER_VERSION})"
        )
    graph_length = int.from_bytes(raw[8:16], "little")
    model_length = int.from_bytes(raw[16:24], "little")
    if graph_length == 0:
        raise ContainerFormatError("The graph descriptor length is 0")
    for label, length in (("graph descriptor", graph_length), ("model descriptor", model_length)):
        if length > MAX_DESCRIPTOR_BYTES:
            raise ContainerFormatError(
                f"The {label} length {length} exceeds the limit {MAX_DESCRIPTOR_BYTES}"
            )
    if kind == "graph" and model_length != 0:
        raise ContainerFormatError(f"krg but the model descriptor length is {model_length}")
    if kind == "model" and model_length == 0:
        raise ContainerFormatError("krm but the model descriptor length is 0")
    return ContainerHeader(kind, version, graph_length, model_length)


#: Digit count of the sequence numbers (`<stem>-NNNNN-of-NNNNN<suffix>` — §8; the same spelling
#: convention as the old shards).
SEQUENCE_DIGITS = 5

#: Largest index the sequence numbers can express (derived from the digit count — a constraint
#: separate from the part count limit).
_MAX_SEQUENCE_INDEX = 10**SEQUENCE_DIGITS - 1

#: Inverse of the numbered file name (`<stem>-NNNNN-of-NNNNN` → `<stem>`). MUST: the sequence
#: numbers are ASCII digits only (here and in {@link _sequence_pattern}) — Python's `\d` also
#: matches Unicode digits (full-width U+FF11), and `int()` would read such a name as a part
#: {@link numbered_name} never writes.
_SEQUENCE_STEM = re.compile(rf"^(.+)-[0-9]{{{SEQUENCE_DIGITS}}}-of-[0-9]{{{SEQUENCE_DIGITS}}}$")


def numbered_name(name: str, index: int, total: int) -> str:
    """`<before the extension>-NNNNN-of-NNNNN<extension>` (`index` starts at 1 — §8).

    `name` may be a path fragment (only the last element is rewritten) — the distribution's
    relative path and the local real path come from the same spelling.
    """
    if not 1 <= index <= total <= min(MAX_PARTS, _MAX_SEQUENCE_INDEX):
        raise ContainerFormatError(
            f"Sequence number {index}/{total} is outside the range"
            f" 1..{min(MAX_PARTS, _MAX_SEQUENCE_INDEX)}"
        )
    parsed = Path(name)
    numbered = f"{index:0{SEQUENCE_DIGITS}d}-of-{total:0{SEQUENCE_DIGITS}d}"
    return str(parsed.with_name(f"{parsed.stem}-{numbered}{parsed.suffix}"))


def numbered_path(path: Path, index: int, total: int) -> Path:
    """The `Path` version of {@link numbered_name} (the parent directory stays as is)."""
    return path.with_name(numbered_name(path.name, index, total))


def base_path(path: Path) -> Path:
    """Numbered file name → **representative path** (unchanged when not numbered).

    The inverse of {@link numbered_name}. A fold so that a call pointing at a local file
    (`model.i8-00001-of-00003.krm`) is not silently treated as "a container of only one file".
    """
    name = Path(path.name)
    matched = _SEQUENCE_STEM.fullmatch(name.stem)
    return path if matched is None else path.with_name(f"{matched.group(1)}{name.suffix}")


def _sequence_pattern(path: Path) -> re.Pattern[str]:
    """A regular expression matching the numbered file names that belong to the same
    representative path as `path`.

    The stem / suffix are passed through `re.escape` — real paths contain dots and hyphens, so
    embedding them raw would pick up unrelated files (glob is not used for the same reason: a name
    containing `[` would silently be read differently).
    """
    name = Path(path.name)
    stem, suffix = re.escape(name.stem), re.escape(name.suffix)
    return re.compile(
        rf"^{stem}-([0-9]{{{SEQUENCE_DIGITS}}})-of-([0-9]{{{SEQUENCE_DIGITS}}}){suffix}$"
    )


def resolve_sequence(path: Path) -> tuple[Path, ...]:
    """Representative path → the existing numbered sequence (one element when not split).

    Always returned in index order. An unsplit file and a missing file both return `(path,)` (the
    diagnosis of absence belongs to the caller's check — doing it here first would split the
    spelling in two).

    MUST: an ambiguous situation fails loudly. Single form and numbered files **side by side**, a
    disagreeing `of`, missing / overflowing numbers — each leaves "which byte sequence to
    distribute" undetermined.
    """
    parent = path.parent
    if not parent.is_dir():
        return (path,)
    pattern = _sequence_pattern(path)
    found: dict[int, Path] = {}
    totals: set[int] = set()
    for entry in parent.iterdir():
        match = pattern.fullmatch(entry.name)
        if match is None or not entry.is_file():
            continue
        found[int(match.group(1))] = entry
        totals.add(int(match.group(2)))
    if not found:
        return (path,)
    if path.is_file():
        raise ContainerFormatError(
            f"{path}: single form and numbered files ({len(found)}) exist side by side"
            " — delete the leftovers of the previous write and retry"
        )
    if len(totals) != 1:
        raise ContainerFormatError(f"{path}: the sequence totals disagree: {sorted(totals)}")
    total = totals.pop()
    missing = sorted(set(range(1, total + 1)) - set(found))
    surplus = sorted(set(found) - set(range(1, total + 1)))
    if missing or surplus:
        raise ContainerFormatError(
            f"{path}: sequence 1..{total} is incomplete (missing {missing} / overflowing {surplus})"
        )
    return tuple(found[index] for index in range(1, total + 1))


def sequence_siblings(path: Path) -> tuple[Path, ...]:
    """Existing files that can be outputs of this representative path (the representative path +
    every numbered file).

    Used by cleanup (files a previous write left with a different split count) and by temporary
    file sweeping. Numbering consistency is not checked — **the more broken the leftovers, the more
    it matters to pick them up**, so they are picked up by the shape of the name alone.
    """
    siblings = [path] if path.is_file() else []
    parent = path.parent
    if parent.is_dir():
        pattern = _sequence_pattern(path)
        siblings.extend(
            entry
            for entry in sorted(parent.iterdir())
            if pattern.fullmatch(entry.name) and entry.is_file()
        )
    return tuple(siblings)


def container_paths(path: Path, part_count: int) -> list[Path]:
    """File names of the split form (from part 0; the numbering convention of §8)."""
    return [
        path.parent / numbered_name(path.name, index + 1, part_count) for index in range(part_count)
    ]


def container_parts(path: str | Path) -> tuple[Path, ...]:
    """The part sequence of one container (a part 0 file, a single-form file or the representative
    path may be passed).

    Split-form parts follow the numbering convention of §8, so folding and resolving borrow the
    single path of {@link base_path} / {@link resolve_sequence} (single form and numbered files
    side by side fail loudly there).
    """
    return resolve_sequence(base_path(Path(path)))


@dataclass(frozen=True)
class DocumentRef:
    """Expected values for one descriptor document (`container.descriptor` of manifest
    `karume/5`).

    The **byte length + sha256** of each of the 2 documents — a fact separate from the sha256 of
    the part 0 file (ADR 0109 decision 3). The sha256 of the `graph` document is used as is for
    `krg` identity (ADR 0108 decision 4).
    """

    length: int
    sha256: str

    def to_document(self) -> dict[str, Any]:
        return {"length": self.length, "sha256": self.sha256}


def read_descriptor_refs(path: Path) -> tuple[DocumentRef, DocumentRef]:
    """Take `(byte length, sha256)` of the 2 documents from a part 0 file (graph descriptor →
    model descriptor).

    Reads only the header + 2 documents (part 0 is at most 32 MiB × 2 + 24 B, so it fits whole).
    Structural validity is not checked — verification belongs to
    {@link karume.verify.verify_container}; this is only a reader for taking the expected values
    to bake into the manifest **from the placed file**.

    MUST: open it only once, and do not read the whole file for the 24 B header — the caller
    (`karume.dist._materialize_family`) passes through here **before** deciding "is it single
    form", so a whole read would let an assembly that put a single-form container into a weights
    slot load several GB into RAM before failing as a rule violation.
    """
    with path.open("rb") as handle:
        header = read_header(handle.read(HEADER_BYTES))
        if header.kind != "model":
            raise ContainerFormatError(f"{path}: not a krm (magic is {MAGIC_GRAPH!r})")
        graph_bytes = handle.read(header.graph_length)
        model_bytes = handle.read(header.model_length)
    for label, raw, expected in (
        ("graph descriptor", graph_bytes, header.graph_length),
        ("model descriptor", model_bytes, header.model_length),
    ):
        if len(raw) != expected:
            raise ContainerFormatError(
                f"{path}: the {label} is short of the declared {expected} bytes ({len(raw)} bytes)"
            )
    return (
        DocumentRef(header.graph_length, hashlib.sha256(graph_bytes).hexdigest()),
        DocumentRef(header.model_length, hashlib.sha256(model_bytes).hexdigest()),
    )


def derive_part_offsets(part0_length: int, part_lengths: Sequence[int]) -> list[int]:
    """Absolute offset at which each part sits in single form (§8).

    MUST: **no padding is inserted before a part of length 0** — inserting it would make a `krg`
    extracted from a `krm` with an empty const region differ in bytes from a `krg` written
    directly (§3 / §9).
    """
    offsets = [0]
    cursor = part0_length
    for length in part_lengths:
        start = cursor if length == 0 else align_up(cursor, BLOCK_START_ALIGN)
        offsets.append(start)
        cursor = start + length
    return offsets


# ---------------------------------------------------------------------------
# Layout (how blocks / parts are filled — §3 / §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Chunk:
    """Where the bytes of one block are (the tensor key, the byte range within it, and the length
    of the padding to bake)."""

    key: str
    #: Total payload bytes of that tensor (decided by the declaration — fails loudly if the actual
    #: bytes disagree).
    total: int
    begin: int
    end: int
    pad: int

    @property
    def length(self) -> int:
        return self.end - self.begin + self.pad


@dataclass(frozen=True)
class _Placed:
    id: str
    offset: int
    chunk: _Chunk


class _PartBuilder:
    """Packer for one part (offsets are relative to the start of the part)."""

    def __init__(self) -> None:
        self._cursor = 0
        self._blocks: list[_Placed] = []

    def probe(self, lengths: Sequence[int]) -> int:
        cursor = self._cursor
        for length in lengths:
            cursor = align_up(cursor, BLOCK_START_ALIGN) + length
        return cursor

    def push(self, block_id: str, chunk: _Chunk) -> None:
        offset = align_up(self._cursor, BLOCK_START_ALIGN)
        self._blocks.append(_Placed(block_id, offset, chunk))
        self._cursor = offset + chunk.length

    @property
    def is_empty(self) -> bool:
        return not self._blocks

    @property
    def length(self) -> int:
        return self._cursor

    @property
    def blocks(self) -> list[_Placed]:
        return self._blocks


def _stored_chunk(key: str, total: int, begin: int, end: int, tail_pad: bool, where: str) -> _Chunk:
    """The byte range that goes into a block (tail padding is baked by the writer — §4.1)."""
    length = end - begin
    if length == 0:
        raise ContainerFormatError(f"{where}: a block of length 0 is not created")
    if not tail_pad:
        # MUST: padding cannot be put on a middle piece (it would overwrite the start of the next
        # piece — rule ②).
        if length % BLOCK_TAIL_ALIGN != 0:
            raise ContainerFormatError(
                f"{where}: middle piece length {length} is not a multiple of {BLOCK_TAIL_ALIGN}"
            )
        return _Chunk(key, total, begin, end, 0)
    return _Chunk(key, total, begin, end, align_up(length, BLOCK_TAIL_ALIGN) - length)


def _cut_rows(
    payload: int, rows: int, block_bytes: int, where: str
) -> list[tuple[tuple[int, int], int, int, bool]]:
    """Cut a tensor at row boundaries into pieces within the block limit (one piece when no cut
    is needed)."""
    if align_up(payload, BLOCK_TAIL_ALIGN) <= block_bytes:
        return [((0, rows), 0, payload, True)]
    if rows <= 1:
        raise ContainerFormatError(f"{where}: one row exceeds the block limit, so it cannot be cut")
    if payload % rows != 0:
        raise ContainerFormatError(f"{where}: {payload} bytes is not divisible by {rows} rows")
    row_bytes = payload // rows
    step = BLOCK_TAIL_ALIGN // math.gcd(row_bytes, BLOCK_TAIL_ALIGN)
    capacity = block_bytes // row_bytes
    rows_per_piece = capacity - (capacity % step)
    if rows_per_piece <= 0:
        raise ContainerFormatError(
            f"{where}: with {row_bytes} bytes per row, the block limit {block_bytes} cannot hold"
            f" a bundle of rows aligned to {BLOCK_TAIL_ALIGN} bytes"
        )
    cuts: list[tuple[tuple[int, int], int, int, bool]] = []
    begin = 0
    while begin < rows:
        end = min(begin + rows_per_piece, rows)
        cuts.append(((begin, end), begin * row_bytes, end * row_bytes, end == rows))
        begin += rows_per_piece
    return cuts


def _cut_scale(
    scale_payload: int,
    rows: int,
    cuts: Sequence[tuple[tuple[int, int], int, int, bool]],
    block_bytes: int,
    where: str,
) -> list[tuple[tuple[int, int], int, int, bool]]:
    """The scale's blocks: one when it fits the block limit, otherwise one per data piece covering
    the same rows (rule ③ — ADR 0108 addendum 15). Fails loudly at planning time when no split
    fits.

    MUST: a scale that fits stays one block (only then is the container byte-identical to a writer
    without split scales — the descriptor's `scale` keeps `block`).
    """
    if align_up(scale_payload, BLOCK_TAIL_ALIGN) <= block_bytes:
        return [((0, rows), 0, scale_payload, True)]
    if len(cuts) == 1:
        raise ContainerFormatError(
            f"{where}: the scale of {scale_payload} bytes exceeds the block limit {block_bytes},"
            " and the data is one block, so there are no piece boundaries to split it on"
        )
    # Pieces exist only for rowAxis 0 (rule ④ — checked by the caller first), so the scale's rows
    # are the data's leading dimension.
    row_bytes = scale_payload // rows
    scale_cuts: list[tuple[tuple[int, int], int, int, bool]] = []
    for index, ((begin, end), _, _, last) in enumerate(cuts):
        length = (end - begin) * row_bytes
        if align_up(length, BLOCK_TAIL_ALIGN) > block_bytes:
            raise ContainerFormatError(
                f"{where} scale {index}: rows [{begin}, {end}) of the scale are {length} bytes,"
                f" over the block limit {block_bytes} (the data's pieces are too coarse for the"
                " scale)"
            )
        scale_cuts.append(((begin, end), begin * row_bytes, end * row_bytes, last))
    return scale_cuts


@dataclass(frozen=True)
class _TensorPlan:
    """The facts of one initializer that are "decided by the declaration" (decided without reading
    a single byte of the body)."""

    name: str
    encoding: Encoding
    payload: int
    rows: int
    scale_payload: int | None


def _numel(shape: Sequence[Any], where: str) -> int:
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ContainerFormatError(
                f"{where}: an initializer's shape cannot use a symbolic dimension ({dim!r})"
            )
        count *= dim
    return count


def _plan_tensor(
    name: str, document: Mapping[str, Any], encoding: Encoding, where: str
) -> _TensorPlan:
    value = document["values"].get(name)
    if value is None:
        raise ContainerFormatError(f"{where}: no dtype / shape declaration in `values`")
    shape = value["shape"]
    numel = _numel(shape, where)
    entry = codec_entry(encoding.codec)
    payload = payload_bytes(encoding.codec, numel, where)
    rows = shape[0] if shape else 1
    if entry.scale == "forbidden":
        if encoding.scale_key is not None or encoding.group_size is not None:
            raise ContainerFormatError(
                f"{where}: codec '{encoding.codec}' cannot have scale / groupSize"
            )
        return _TensorPlan(name, encoding, payload, rows, None)
    if encoding.scale_key is None or encoding.group_size is None:
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' requires scale and groupSize"
        )
    row_axis = encoding.row_axis or 0
    if row_axis not in (0, 1):
        raise ContainerFormatError(f"{where}: rowAxis must be 0 or 1 (declared {row_axis})")
    assert_row_axis(encoding.codec, row_axis, where)
    if len(shape) <= row_axis:
        raise ContainerFormatError(
            f"{where}: for rowAxis {row_axis} the rank of the declared shape {list(shape)} is"
            " too small"
        )
    row_count = shape[row_axis]
    row_length = numel // row_count if row_count else 0
    if entry.grouping == "channel" and encoding.group_size != per_channel_group_size(row_length):
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' is per-channel, so groupSize MUST equal the row"
            f" length {per_channel_group_size(row_length)} (declared {encoding.group_size})"
        )
    if entry.grouping == "group":
        size = encoding.group_size
        if size < MIN_GROUP_SIZE or size & (size - 1) != 0:
            raise ContainerFormatError(
                f"{where}: groupSize {size} is not a power of 2 that is {MIN_GROUP_SIZE} or more"
            )
        if row_length % size != 0:
            raise ContainerFormatError(
                f"{where}: row length {row_length} is not divisible by groupSize {size}"
            )
    scale_payload = row_count * group_count(row_length, encoding.group_size) * 4
    return _TensorPlan(name, encoding, payload, rows, scale_payload)


class _Serial:
    """Serial numbers of block ids (`w0001` / `s0002` … — the same spelling as the reference
    implementation)."""

    def __init__(self) -> None:
        self._value = 0

    def next(self, prefix: str) -> str:
        self._value += 1
        return f"{prefix}{self._value:04d}"


@dataclass(frozen=True)
class _ConstRegion:
    length: int
    blocks: list[_Placed]
    bindings: list[tuple[str, str, BlockEncoding]]


def _build_const_region(
    graph_name: str, plans: Sequence[_TensorPlan], block_bytes: int
) -> _ConstRegion:
    builder = _PartBuilder()
    serial = _Serial()
    bindings: list[tuple[str, str, BlockEncoding]] = []
    for plan in plans:
        where = f"const {graph_name}/{plan.name}"
        chunk = _stored_chunk(plan.name, plan.payload, 0, plan.payload, True, where)
        if chunk.length > block_bytes:
            # const has no piece-splitting mechanism (§3 — krg has no binding table, so a piece
            # sequence cannot be expressed).
            raise ContainerFormatError(
                f"{where}: const must fit in one block but is {chunk.length} bytes"
                f" (limit {block_bytes})"
            )
        block_id = serial.next("c")
        builder.push(block_id, chunk)
        scale_id: str | None = None
        if plan.scale_payload is not None:
            assert plan.encoding.scale_key is not None
            scale_chunk = _stored_chunk(
                plan.encoding.scale_key,
                plan.scale_payload,
                0,
                plan.scale_payload,
                True,
                f"{where} scale",
            )
            if scale_chunk.length > block_bytes:
                # A const is one block, so its scale has no piece boundaries to split on (rule ③).
                raise ContainerFormatError(
                    f"{where}: the scale of a const must fit in one block but is"
                    f" {scale_chunk.length} bytes (limit {block_bytes})"
                )
            scale_id = serial.next("cs")
            builder.push(scale_id, scale_chunk)
        bindings.append(
            (
                plan.name,
                block_id,
                BlockEncoding(
                    plan.encoding.codec,
                    None if scale_id is None else (plan.encoding.row_axis or 0),
                    None if scale_id is None else plan.encoding.group_size,
                    scale_id,
                ),
            )
        )
    return _ConstRegion(builder.length, builder.blocks, bindings)


@dataclass(frozen=True)
class _DataPlan:
    parts: list[_PartBuilder]
    roles: dict[str, BlockRole]
    supplies: dict[str, WeightSupply]


def _plan_data_parts(
    graph_name: str, plans: Sequence[_TensorPlan], part_bytes: int, block_bytes: int
) -> _DataPlan:
    parts = [_PartBuilder()]
    roles: dict[str, BlockRole] = {}
    supplies: dict[str, WeightSupply] = {}
    serial = _Serial()

    def next_id(prefix: str, role: BlockRole) -> str:
        block_id = serial.next(prefix)
        roles[block_id] = role
        return block_id

    def place_group(items: Sequence[tuple[str, _Chunk]], where: str) -> None:
        """Place a group that must "sit in the same part" together. Open a new part if it does not
        fit."""
        lengths = [chunk.length for _, chunk in items]
        if _PartBuilder().probe(lengths) > part_bytes:
            raise ContainerFormatError(
                f"{where}: a group that must sit in the same part exceeds the part length"
                f" {part_bytes}"
            )
        target = parts[-1]
        if target.probe(lengths) > part_bytes:
            target = _PartBuilder()
            parts.append(target)
        for block_id, chunk in items:
            target.push(block_id, chunk)

    for plan in plans:
        where = f"weight {graph_name}/{plan.name}"
        cuts = _cut_rows(plan.payload, plan.rows, block_bytes, where)
        if len(cuts) > 1 and (plan.encoding.row_axis or 0) != 0:
            # Rule ④: the scale's row ranges do not correspond to the pieces' row ranges, so it
            # cannot be split.
            raise ContainerFormatError(
                f"{where}: an initializer with rowAxis {plan.encoding.row_axis} cannot be split"
                " into pieces"
            )
        scale_cuts = (
            None
            if plan.scale_payload is None
            else _cut_scale(plan.scale_payload, plan.rows, cuts, block_bytes, where)
        )
        piece_ids = [next_id("w", "weight") for _ in cuts]
        head: list[tuple[str, _Chunk]] = [
            (
                piece_ids[0],
                _stored_chunk(
                    plan.name, plan.payload, cuts[0][1], cuts[0][2], cuts[0][3], f"{where} piece 0"
                ),
            )
        ]
        scale_ids: list[str] = []
        if scale_cuts is not None:
            assert plan.encoding.scale_key is not None and plan.scale_payload is not None
            for index, (_, begin, end, last) in enumerate(scale_cuts):
                scale_ids.append(next_id("s", "scale"))
                head.append(
                    (
                        scale_ids[-1],
                        _stored_chunk(
                            plan.encoding.scale_key,
                            plan.scale_payload,
                            begin,
                            end,
                            last,
                            f"{where} scale" if len(scale_cuts) == 1 else f"{where} scale {index}",
                        ),
                    )
                )
        # Rule ③: piece 1 and the scale (every piece of a split scale) share a part. The remaining
        # pieces can be placed independently.
        place_group(head, where)
        for index, (_, begin, end, last) in enumerate(cuts):
            if index == 0:
                continue
            place_group(
                [
                    (
                        piece_ids[index],
                        _stored_chunk(
                            plan.name, plan.payload, begin, end, last, f"{where} piece {index}"
                        ),
                    )
                ],
                f"{where} piece {index}",
            )
        quantized = scale_cuts is not None
        scale_pieces: Pieces | None = None
        if scale_cuts is not None and len(scale_cuts) > 1:
            scale_pieces = tuple(
                (scale_id, cut[0]) for scale_id, cut in zip(scale_ids, scale_cuts, strict=True)
            )
        encoding = BlockEncoding(
            plan.encoding.codec,
            (plan.encoding.row_axis or 0) if quantized else None,
            plan.encoding.group_size if quantized else None,
            scale_ids[0] if quantized and scale_pieces is None else None,
            scale_pieces,
        )
        supplies[plan.name] = (
            WeightSupply(encoding, block=piece_ids[0])
            if len(cuts) == 1
            else WeightSupply(
                encoding,
                pieces=tuple((piece_ids[index], cut[0]) for index, cut in enumerate(cuts)),
            )
        )
    return _DataPlan([part for part in parts if not part.is_empty], roles, supplies)


@dataclass(frozen=True)
class _AssetPlan:
    parts: list[_PartBuilder]
    #: Asset name → one entry of the model descriptor.
    bindings: dict[str, AssetRecord]


def _asset_view(name: str, asset: AssetInput) -> memoryview:
    """The bytes of one asset (a lazy call is resolved here). The caller releases it when done."""
    raw = asset.payload() if callable(asset.payload) else asset.payload
    try:
        return memoryview(raw).cast("B")
    except TypeError as cause:
        raise ContainerFormatError(f"Asset '{name}': payload is not a byte sequence") from cause


def _plan_asset_parts(
    assets: Mapping[str, AssetInput], part_bytes: int, block_bytes: int
) -> _AssetPlan:
    """Lay assets out in **parts not shared with weights** (§4.2 — range-read assets are not put
    behind a scan).

    The layout order is **the order the caller passed** (PLE comes in token order — with a
    scan-type source, adjacent tokens do not scatter into different parts). The spelling of the
    descriptor's `assets` is put in code point order by canonical serialization.

    Assets are not split into pieces (the row step is known only to the index of each role, and
    the descriptor holds no more than "name → block") — one that does not fit in one block fails
    loudly. A `dedicated_part` asset is placed alone in a dedicated part; other assets may share
    one part with each other.
    """
    parts: list[_PartBuilder] = []
    bindings: dict[str, AssetRecord] = {}
    current: _PartBuilder | None = None
    for serial, (name, asset) in enumerate(assets.items()):
        where = f"Asset '{name}'"
        if not name:
            raise ContainerFormatError("Asset name is an empty string")
        if not isinstance(asset.role, str) or not asset.role:
            raise ContainerFormatError(f"{where}: role is not a non-empty string ({asset.role!r})")
        if not isinstance(asset.length, int) or isinstance(asset.length, bool):
            raise ContainerFormatError(
                f"{where}: logical length is not an integer ({asset.length!r})"
            )
        chunk = _stored_chunk(name, asset.length, 0, asset.length, True, where)
        if chunk.length > block_bytes:
            raise ContainerFormatError(
                f"{where}: an asset must fit in one block but is {chunk.length} bytes"
                f" (limit {block_bytes})"
            )
        if chunk.length > part_bytes:
            raise ContainerFormatError(
                f"{where}: block of {chunk.length} bytes exceeds the part length {part_bytes}"
            )
        if asset.dedicated_part or current is None or current.probe([chunk.length]) > part_bytes:
            current = _PartBuilder()
            parts.append(current)
        # Block ids are decimal from 0 in layout order (`a.0` / `a.1` …).
        block_id = f"a.{serial}"
        current.push(block_id, chunk)
        bindings[name] = AssetRecord(block_id, asset.role, asset.length)
        if asset.dedicated_part:
            # Alone, MUST: the next asset does not share this part (§4.2).
            current = None
    return _AssetPlan(parts, bindings)


# ---------------------------------------------------------------------------
# Taking out the bytes (one tensor at a time; carried over only while consecutive)
# ---------------------------------------------------------------------------


class _Payloads:
    """The entry that looks up tensors **one at a time**. Carries one over only while the same key
    is consecutive.

    MUST: one tensor is carried over. A piece sequence is laid out consecutively within the same
    part, so this one avoids "looking up the same tensor again for every piece" while keeping the
    bytes alive at any one time to a single tensor.
    """

    def __init__(self, tensors: Mapping[str, Buffer]) -> None:
        self._tensors = tensors
        self._key: str | None = None
        self._view: memoryview | None = None

    def view(self, key: str, expected: int) -> memoryview:
        if key != self._key:
            self.release()
            try:
                raw = self._tensors[key]
            except KeyError:
                raise ContainerFormatError(f"The bytes of tensor '{key}' were not passed") from None
            view = memoryview(raw).cast("B")
            if view.nbytes != expected:
                raise ContainerFormatError(
                    f"Tensor '{key}': the bytes ({view.nbytes} bytes) differ from the"
                    f" {expected} bytes decided by the declaration"
                )
            self._view = view
            self._key = key
        assert self._view is not None
        return self._view

    def chunk(self, chunk: _Chunk) -> memoryview:
        return self.view(chunk.key, chunk.total)[chunk.begin : chunk.end]

    def release(self) -> None:
        self._view = None
        self._key = None


class _Sources(Mapping[str, Buffer]):
    """Bundle tensors and assets into one entry (either is looked up by a block's `key`).

    An asset name colliding with a tensor key would silently swap a block's bytes, so collisions
    are listed in full and rejected at bundling time.
    """

    def __init__(self, tensors: Mapping[str, Buffer], assets: Mapping[str, AssetInput]) -> None:
        clash = sorted(set(assets) & set(tensors))
        if clash:
            raise ContainerFormatError(f"Asset names collide with tensor keys: {', '.join(clash)}")
        self._tensors = tensors
        self._assets = dict(assets)

    def __getitem__(self, key: str) -> Buffer:
        asset = self._assets.get(key)
        if asset is None:
            return self._tensors[key]
        return _asset_view(key, asset)

    def __iter__(self) -> Iterator[str]:
        return iter([*self._tensors, *self._assets])

    def __len__(self) -> int:
        return len(self._tensors) + len(self._assets)


def _region_stream(
    blocks: Sequence[_Placed], payloads: _Payloads
) -> Iterator[tuple[str | None, bytes | memoryview]]:
    """Stream one region (= part) in order as a sequence of `(block id, bytes)`.

    MUST: **writing and sha256 share this one generator** (if §4.1's "take the digest including
    padding" were spelled separately on each side, the part sha256 and the actual bytes would
    drift silently). A gap that belongs to no block (padding for alignment) has id `None` and goes
    only into the part sha256.
    """
    cursor = 0
    for placed in blocks:
        if placed.offset > cursor:
            yield None, PAD_BYTE * (placed.offset - cursor)
        yield placed.id, payloads.chunk(placed.chunk)
        if placed.chunk.pad:
            yield placed.id, PAD_BYTE * placed.chunk.pad
        cursor = placed.offset + placed.chunk.length


def _emit_region(
    blocks: Sequence[_Placed], payloads: _Payloads, target: Path | None
) -> tuple[str, dict[str, str]]:
    """Stream a region in one pass and take the part sha256 and the sha256 of each block
    (including padding — §4.1). When `target` is given, write to it **in the same pass**.

    MUST: writing and the digest share the single {@link _region_stream} (if §4.1's "take the
    digest including padding" were spelled separately on each side, the part sha256 and the actual
    bytes would drift silently). The digests are taken while writing, so in split form the bytes
    are **looked up only once** (the storage conversion runs on every lookup).
    """
    region = hashlib.sha256()
    digests: dict[str, Any] = {}
    handle = None if target is None else target.open("wb")
    try:
        for block_id, chunk in _region_stream(blocks, payloads):
            if handle is not None:
                handle.write(chunk)
            region.update(chunk)
            if block_id is not None:
                digests.setdefault(block_id, hashlib.sha256()).update(chunk)
    finally:
        if handle is not None:
            handle.close()
    return region.hexdigest(), {key: digest.hexdigest() for key, digest in digests.items()}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _owned_initializers(document: Mapping[str, Any]) -> list[str]:
    """Initializers that hold their own bytes (shared ones are excluded, since their body is the
    lender's resident weight)."""
    return [
        name
        for name, declaration in document["initializers"].items()
        if not declaration.get("shared", False)
    ]


def _assert_binding_cover(
    expected: Sequence[str], bindings: Mapping[str, Encoding], graph_name: str, what: str
) -> None:
    """Reject missing / surplus bindings **listing every entry** (rule ⑤ — never silently produce a
    fattened / thinned distribution)."""
    missing = sorted(name for name in expected if name not in bindings)
    surplus = sorted(set(bindings) - set(expected))
    if missing or surplus:
        raise ContainerFormatError(
            f"The bindings of graph '{graph_name}' do not match {what}:"
            f" missing [{', '.join(missing)}] / surplus [{', '.join(surplus)}]"
        )


def _split_initializers(
    document: Mapping[str, Any], bindings: Mapping[str, Encoding], graph_name: str
) -> tuple[list[_TensorPlan], list[_TensorPlan]]:
    """Layout plan `(bound for const, bound for weights)` (`krm` — the bindings cover every
    initializer with its own bytes)."""
    owned = _owned_initializers(document)
    _assert_binding_cover(owned, bindings, graph_name, "the initializer declarations")
    consts: list[_TensorPlan] = []
    weights: list[_TensorPlan] = []
    for name in owned:
        plan = _plan_tensor(name, document, bindings[name], f"{graph_name}/{name}")
        (consts if name.startswith(CONST_KEY_PREFIX) else weights).append(plan)
    return consts, weights


def _const_initializers(
    document: Mapping[str, Any], bindings: Mapping[str, Encoding], graph_name: str
) -> list[_TensorPlan]:
    """Layout plan of the const region (`krg` — only const goes into the binding table).

    `krg` **has no weight binding table** (§9), so weight bindings that are passed in are rejected
    as surplus. The declarations of the initializers that need weights stay in the graph (the
    requirement is derived from the `values` declarations).
    """
    expected = [name for name in _owned_initializers(document) if name.startswith(CONST_KEY_PREFIX)]
    _assert_binding_cover(expected, bindings, graph_name, "the set of const supplies")
    return [
        _plan_tensor(name, document, bindings[name], f"{graph_name}/{name}") for name in expected
    ]


def _assert_graph_name(graph_name: str) -> None:
    if not GRAPH_NAME_PATTERN.fullmatch(graph_name):
        raise ContainerFormatError(
            f"Graph name '{graph_name}' is outside the vocabulary ({GRAPH_NAME_PATTERN.pattern})"
        )


def _assert_block_budget(count: int) -> None:
    if count > MAX_BLOCKS:
        raise ContainerFormatError(f"Block count {count} exceeds the limit {MAX_BLOCKS}")


def write_graph_container(
    path: Path,
    graph: IrGraph,
    const_tensors: Mapping[str, Buffer],
    bindings: Mapping[str, Encoding],
    *,
    graph_name: str,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> Path:
    """Write a `krg` (graph container) — `[header][graph descriptor][padding][const region]` (§9).

    `bindings` holds only const bindings (`krg` has no weight binding table). A `krg` written from
    the same graph and the same const is **byte-identical** to a `krg` extracted from a `krm` by
    byte copy.
    """
    _assert_graph_name(graph_name)
    if block_bytes > BLOCK_MAX_BYTES:
        raise ContainerFormatError(
            f"Block limit {block_bytes} exceeds the spec's {BLOCK_MAX_BYTES}"
        )
    document = ir_v2_document(graph)
    consts = _const_initializers(document, bindings, graph_name)
    region = _build_const_region(graph_name, consts, block_bytes)
    _assert_block_budget(len(region.blocks))
    payloads = _Payloads(const_tensors)
    _, block_hashes = _emit_region(region.blocks, payloads, None)
    payloads.release()
    descriptor = _graph_descriptor(graph_name, document, region, block_hashes)
    graph_bytes = serialize_graph_descriptor(descriptor)
    header = write_header(kind="graph", graph_length=len(graph_bytes), model_length=0)
    offsets = derive_part_offsets(HEADER_BYTES + len(graph_bytes), [region.length])
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = _Payloads(const_tensors)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(graph_bytes)
        handle.write(PAD_BYTE * (offsets[1] - HEADER_BYTES - len(graph_bytes)))
        for _, piece in _region_stream(region.blocks, payloads):
            handle.write(piece)
    payloads.release()
    return path


def _graph_descriptor(
    graph_name: str,
    document: Mapping[str, Any],
    region: _ConstRegion,
    block_hashes: Mapping[str, str],
) -> GraphDescriptor:
    return GraphDescriptor(
        graphs={graph_name: document},
        const_length=region.length,
        const_blocks=tuple(
            ConstBlockRecord(placed.id, placed.offset, placed.chunk.length, block_hashes[placed.id])
            for placed in region.blocks
        ),
        constants=tuple(
            ConstantBinding(graph_name, name, block_id, encoding)
            for name, block_id, encoding in region.bindings
        ),
    )


def write_model_container(
    path: Path,
    graph: IrGraph,
    tensors: Mapping[str, Buffer],
    bindings: Mapping[str, Encoding],
    *,
    graph_name: str,
    provenance: Provenance,
    assets: Mapping[str, AssetInput] = {},
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
    single: bool = False,
) -> list[Path]:
    """Write a `krm` (model container). Returns the written files in order (from part 0).

    `tensors` is the **tensor key → raw bytes** entry; the bytes are looked up one tensor at a
    time and released one at a time (with a lazy `Mapping`, the whole never lands in memory).
    `bindings` MUST cover every initializer that is not shared; keys starting with `const.` go to
    the const region (part 1), all others to the weights (part 2 onward).

    `assets` (asset name → {@link AssetInput}) are laid out in dedicated parts **after** the
    weight parts (§4.2 — range-read assets are not put behind the weight scan). The order is **the
    order passed**, and a `dedicated_part` asset becomes 1 block = 1 part.

    The file name is `path` itself in single form and `<stem>-NNNNN-of-NNNNN<suffix>` (from part 0)
    in split form.
    """
    _assert_graph_name(graph_name)
    if block_bytes > BLOCK_MAX_BYTES:
        raise ContainerFormatError(
            f"Block limit {block_bytes} exceeds the spec's {BLOCK_MAX_BYTES}"
        )
    if not 0 < part_bytes <= PART_MAX_BYTES:
        raise ContainerFormatError(
            f"Part length {part_bytes} is outside 1..{PART_MAX_BYTES} (the part length ceiling)"
        )
    document = ir_v2_document(graph)
    consts, weights = _split_initializers(document, bindings, graph_name)
    region = _build_const_region(graph_name, consts, block_bytes)
    data = _plan_data_parts(graph_name, weights, part_bytes, block_bytes)
    asset_plan = _plan_asset_parts(assets, part_bytes, block_bytes)
    builders = [*data.parts, *asset_plan.parts]
    roles: dict[str, BlockRole] = {
        **data.roles,
        **{record.block: "asset" for record in asset_plan.bindings.values()},
    }
    _assert_block_budget(len(region.blocks) + sum(len(part.blocks) for part in builders))
    if len(builders) + 2 > MAX_PARTS:
        raise ContainerFormatError(f"Part count {len(builders) + 2} exceeds the limit {MAX_PARTS}")

    # ② Stream the bytes one tensor at a time and take the block / part sha256.
    #
    # **Split form writes the data parts right here** (part 0 is a separate file, so it can be
    # written last) — the bytes are looked up only once, and the storage conversion runs only
    # once. In single form part 0 comes first and its contents depend on the sha256 of the parts
    # after it, so the bytes are **looked up twice** (a temporary file in between would add one
    # write of the whole model — the choice in the module doc).
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path] if single else container_paths(path, len(builders) + 2)
    payloads = _Payloads(_Sources(tensors, assets))
    region_hash, block_hashes = _emit_region(
        region.blocks, payloads, None if single else written[1]
    )
    parts = [PartRecord(1, region.length, region_hash)]
    data_blocks: list[DataBlockRecord] = []
    for index, builder in enumerate(builders):
        part_hash, hashes = _emit_region(
            builder.blocks, payloads, None if single else written[index + 2]
        )
        block_hashes.update(hashes)
        parts.append(PartRecord(index + 2, builder.length, part_hash))
        for placed in builder.blocks:
            data_blocks.append(
                DataBlockRecord(
                    placed.id,
                    index + 2,
                    placed.offset,
                    placed.chunk.length,
                    hashes[placed.id],
                    roles[placed.id],
                )
            )
    payloads.release()

    # ③ Build and write the 2 documents.
    graph_descriptor = _graph_descriptor(graph_name, document, region, block_hashes)
    model_descriptor = ModelDescriptor(
        parts=tuple(parts),
        blocks=tuple(data_blocks),
        binding={graph_name: data.supplies},
        assets=dict(asset_plan.bindings),
        provenance=provenance,
    )
    graph_bytes = serialize_graph_descriptor(graph_descriptor)
    model_bytes = serialize_model_descriptor(model_descriptor)
    header = write_header(
        kind="model", graph_length=len(graph_bytes), model_length=len(model_bytes)
    )
    part0_length = HEADER_BYTES + len(graph_bytes) + len(model_bytes)
    if part0_length > PART_MAX_BYTES:
        raise ContainerFormatError(
            f"Part 0 length {part0_length} exceeds the part length ceiling {PART_MAX_BYTES}"
        )
    if single:
        regions = [region.blocks, *(builder.blocks for builder in builders)]
        offsets = derive_part_offsets(part0_length, [part.length for part in parts])
        payloads = _Payloads(_Sources(tensors, assets))
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(graph_bytes)
            handle.write(model_bytes)
            for index, blocks in enumerate(regions):
                handle.write(PAD_BYTE * (offsets[index + 1] - handle.tell()))
                for _, piece in _region_stream(blocks, payloads):
                    handle.write(piece)
        payloads.release()
        return [path]

    # The data parts were put in place in ②. Only part 0 remains.
    with written[0].open("wb") as handle:
        handle.write(header)
        handle.write(graph_bytes)
        handle.write(model_bytes)
    return written


# ---------------------------------------------------------------------------
# Reader (for the self-check — the declaration checks of §2 / §5 + block sha256 verification)
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContainerFormatError(message)


def _require_object(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} is not an object")
    return value


def _require_keys(
    obj: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], where: str
) -> None:
    for key in required:
        _require(key in obj, f"{where}.{key} is missing")
    known = set(required) | set(optional)
    for key in obj:
        _require(key in known, f"{where}: unknown key '{key}'")


def _require_index(value: Any, where: str) -> int:
    # An integral float (`64.0`) is accepted, as TS `Number.isSafeInteger` does (the reason is in
    # the function of the same name in ops).
    value = integral_float_to_int(value)
    _require(
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_SAFE_INTEGER,
        f"{where} is not a non-negative safe integer: {value!r}",
    )
    return value


def _require_version(value: Any, expected: int, where: str) -> None:
    """Compare a version field as an integer.

    MUST: reject bool first. In Python `True == 1` is true, so `==` alone would pass
    `"version": true` as version 1, while the TS reader rejects it with `!== 1` (the accepted sets
    agree — ADR 0108 addendum 1, item 13).
    """
    normalized = integral_float_to_int(value)
    _require(
        type(normalized) is int and normalized == expected,
        f"{where} is not {expected}: {value!r}",
    )


def _require_string(value: Any, where: str) -> str:
    _require(isinstance(value, str) and value != "", f"{where} is not a non-empty string")
    return value


def _require_sha256(value: Any, where: str) -> str:
    _require(
        isinstance(value, str) and SHA256_HEX_PATTERN.fullmatch(value) is not None,
        f"{where} is not a sha256 (64 lowercase hex characters): {value!r}",
    )
    return value


def _require_block_id(value: Any, where: str) -> str:
    block_id = _require_string(value, where)
    _require(
        BLOCK_ID_PATTERN.fullmatch(block_id) is not None,
        f"{where}: block id '{block_id}' is outside the vocabulary ({BLOCK_ID_PATTERN.pattern})",
    )
    return block_id


def _parse_json_document(raw: bytes, where: str) -> Any:
    _require(
        len(raw) <= MAX_DESCRIPTOR_BYTES,
        f"{where} exceeds the limit of {MAX_DESCRIPTOR_BYTES} bytes: {len(raw)}",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as cause:
        raise ContainerFormatError(f"{where} is invalid as UTF-8: {cause}") from cause
    try:
        # MUST: block `parse_constant` (NaN / Infinity literals are forbidden by §0).
        parsed = json.loads(
            text,
            parse_constant=lambda literal: _require(
                False, f"{where}: non-finite number literal {literal}"
            ),
        )
    except json.JSONDecodeError as cause:
        raise ContainerFormatError(f"{where} is invalid as JSON: {cause}") from cause
    _assert_finite(parsed, where, 0)
    return parsed


def _assert_finite(value: Any, where: str, depth: int) -> None:
    """Reject numbers that, like `1e999`, are **syntactically valid but round to Infinity**
    (§0)."""
    _require(depth <= MAX_JSON_DEPTH, f"{where}: nesting exceeded the depth limit {MAX_JSON_DEPTH}")
    if isinstance(value, float):
        _require(
            math.isfinite(value), f"{where}: non-finite number (including overflow like 1e999)"
        )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{where}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            # MUST: `__proto__` is rejected by both the writer and the reader (§0). The TS reader
            # does not accept this name because assigning it into a plain `{}` turns into a
            # [[Prototype]] set — block it once for the whole JSON at the read entry, and do not
            # let it leak into maps where names can be placed freely (asset names, the binding
            # table).
            _require(key != "__proto__", f"{where}: key '__proto__' is not accepted")
            _assert_finite(item, f"{where}.{key}", depth + 1)


def _parse_encoding(value: Any, where: str) -> BlockEncoding:
    obj = _require_object(value, where)
    _require_keys(obj, ["codec", "packing"], ["rowAxis", "groupSize", "scale", "zeroPoint"], where)
    codec = _require_string(obj["codec"], f"{where}.codec")
    entry = codec_entry(codec)
    packing = _require_object(obj["packing"], f"{where}.packing")
    _require_keys(packing, ["blockElements", "blockBytes", "alignBytes"], [], f"{where}.packing")
    # Read as integers before comparing (blocks a bool `True` passing as 1 in tuple comparison).
    declared = tuple(
        _require_index(packing[key], f"{where}.packing.{key}")
        for key in ("blockElements", "blockBytes", "alignBytes")
    )
    expected = (entry.packing.block_elements, entry.packing.block_bytes, entry.packing.align_bytes)
    _require(
        declared == expected,
        f"{where}.packing: the ledger for codec '{codec}' has {expected} but the declaration has"
        f" {declared} (an asset written with another version of the ledger)",
    )
    quantized = entry.scale == "required"
    for key in ("rowAxis", "groupSize", "scale"):
        _require(
            (key in obj) == quantized,
            f"{where}.{key}: codec '{codec}' is"
            + (
                " quantized, so it is required"
                if quantized
                else " not quantized, so it cannot be written"
            ),
        )
    _require(
        "zeroPoint" not in obj,
        f"{where}.zeroPoint: the first-edition ledger cannot have zeroPoint",
    )
    if not quantized:
        return BlockEncoding(codec)
    row_axis = _require_index(obj["rowAxis"], f"{where}.rowAxis")
    _require(row_axis in (0, 1), f"{where}.rowAxis must be 0 or 1 (declared {row_axis})")
    scale = _require_object(obj["scale"], f"{where}.scale")
    _require_keys(scale, ["dtype"], ["block", "pieces"], f"{where}.scale")
    _require(
        scale["dtype"] in SCALE_DTYPES,
        f"{where}.scale.dtype is not in the accepted set ({' / '.join(SCALE_DTYPES)}):"
        f" {scale['dtype']!r}",
    )
    _require(
        ("block" in scale) != ("pieces" in scale),
        f"{where}.scale: write exactly one of 'block' (one block) and 'pieces' (split scale)",
    )
    group_size = _require_index(obj["groupSize"], f"{where}.groupSize")
    if "block" in scale:
        return BlockEncoding(
            codec, row_axis, group_size, _require_block_id(scale["block"], f"{where}.scale.block")
        )
    return BlockEncoding(
        codec,
        row_axis,
        group_size,
        scale_pieces=_parse_pieces(scale["pieces"], f"{where}.scale.pieces"),
    )


def _parse_pieces(value: Any, where: str) -> Pieces:
    """A piece list (§5): 2 or more `{block, rows: [begin, end)}`, each non-empty, continuing
    from row 0 without gaps. The one reader of both the data's `pieces` and a split scale's
    `scale.pieces`."""
    _require(isinstance(value, list), f"{where} is not an array")
    pieces: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    for index, raw_piece in enumerate(value):
        piece_where = f"{where}[{index}]"
        piece = _require_object(raw_piece, piece_where)
        _require_keys(piece, ["block", "rows"], [], piece_where)
        rows = piece["rows"]
        _require(
            isinstance(rows, list) and len(rows) == 2, f"{piece_where}.rows is not of length 2"
        )
        begin = _require_index(rows[0], f"{piece_where}.rows[0]")
        end = _require_index(rows[1], f"{piece_where}.rows[1]")
        _require(end > begin, f"{piece_where}.rows [{begin}, {end}) is an empty range")
        # Rule ④: cover the row range without gaps (the end = shape[0] is checked by the merge
        # layer, which knows the declared shape).
        _require(
            begin == cursor,
            f"{piece_where}: does not continue from row {cursor} (declared {begin})",
        )
        cursor = end
        pieces.append((_require_block_id(piece["block"], f"{piece_where}.block"), (begin, end)))
    _require(len(pieces) >= 2, f"{where} must have 2 or more entries")
    return tuple(pieces)


def _assert_block_shape(offset: int, length: int, where: str) -> None:
    _require(offset % BLOCK_START_ALIGN == 0, f"{where}: offset {offset} is not a multiple of 64")
    _require(length != 0, f"{where}: length is 0")
    _require(length % BLOCK_TAIL_ALIGN == 0, f"{where}: length {length} is not a multiple of 4")
    _require(
        length <= BLOCK_MAX_BYTES,
        f"{where}: length {length} exceeds the block limit {BLOCK_MAX_BYTES}",
    )


def _assert_no_overlap(
    blocks: Sequence[tuple[str, int, int]], region_length: int, where: str
) -> None:
    cursor = 0
    for block_id, offset, length in sorted(blocks, key=lambda block: block[1]):
        _require(offset >= cursor, f"{where}: block '{block_id}' overlaps the preceding block")
        cursor = offset + length
        _require(
            cursor <= region_length,
            f"{where}: block '{block_id}' runs past the region length {region_length} (end"
            f" {cursor})",
        )


def _parse_graph_descriptor(raw: bytes) -> GraphDescriptor:
    where = "graphDescriptor"
    root = _require_object(_parse_json_document(raw, where), where)
    _require_keys(root, ["format", "version", "capabilities", "graphs", "const"], [], where)
    _require(root["format"] == "karume-container", f"{where}.format is not 'karume-container'")
    _require_version(root["version"], CONTAINER_VERSION, f"{where}.version")

    capabilities = _require_object(root["capabilities"], f"{where}.capabilities")
    _require_keys(capabilities, ["ops", "features"], [], f"{where}.capabilities")
    _require(
        capabilities["features"] == [],
        f"{where}.capabilities.features: the first edition allows only an empty array",
    )

    graphs = _require_object(root["graphs"], f"{where}.graphs")
    _require(len(graphs) > 0, f"{where}.graphs is empty")
    _require(
        len(graphs) <= MAX_GRAPHS,
        f"{where}.graphs count {len(graphs)} exceeds the limit {MAX_GRAPHS}",
    )
    union: set[str] = set()
    for name, declaration in graphs.items():
        _require(
            GRAPH_NAME_PATTERN.fullmatch(name) is not None,
            f"{where}.graphs: graph name '{name}' is outside the vocabulary",
        )
        union |= set(_graph_required_ops(declaration, f"{where}.graphs['{name}']"))
    _require(
        capabilities["ops"] == sorted(union),
        f"{where}.capabilities.ops does not match the union of requires.ops of graphs",
    )

    region = _require_object(root["const"], f"{where}.const")
    _require_keys(region, ["length", "blocks", "constants"], [], f"{where}.const")
    length = _require_index(region["length"], f"{where}.const.length")
    blocks: list[ConstBlockRecord] = []
    previous = -1
    for index, raw_block in enumerate(region["blocks"]):
        block_where = f"{where}.const.blocks[{index}]"
        obj = _require_object(raw_block, block_where)
        _require_keys(obj, ["id", "offset", "length", "sha256"], [], block_where)
        record = ConstBlockRecord(
            _require_block_id(obj["id"], f"{block_where}.id"),
            _require_index(obj["offset"], f"{block_where}.offset"),
            _require_index(obj["length"], f"{block_where}.length"),
            _require_sha256(obj["sha256"], f"{block_where}.sha256"),
        )
        _assert_block_shape(record.offset, record.length, block_where)
        _require(
            record.offset > previous, f"{block_where}: offset {record.offset} is not ascending"
        )
        previous = record.offset
        blocks.append(record)
    _assert_no_overlap(
        [(block.id, block.offset, block.length) for block in blocks], length, f"{where}.const"
    )

    constants: list[ConstantBinding] = []
    for index, raw_entry in enumerate(region["constants"]):
        entry_where = f"{where}.const.constants[{index}]"
        obj = _require_object(raw_entry, entry_where)
        _require_keys(obj, ["graph", "initializer", "block", "encoding"], [], entry_where)
        constants.append(
            ConstantBinding(
                _require_string(obj["graph"], f"{entry_where}.graph"),
                _require_string(obj["initializer"], f"{entry_where}.initializer"),
                _require_block_id(obj["block"], f"{entry_where}.block"),
                _parse_encoding(obj["encoding"], f"{entry_where}.encoding"),
            )
        )
    descriptor = GraphDescriptor(graphs, length, tuple(blocks), tuple(constants))
    _validate_graph_descriptor(descriptor)
    return descriptor


def _graph_required_ops(declaration: Any, where: str) -> list[str]:
    obj = _require_object(declaration, where)
    _require(obj.get("format") == IR_V2_FORMAT, f"{where}.format is not '{IR_V2_FORMAT}'")
    _require_version(obj.get("version"), IR_V2_VERSION, f"{where}.version")
    requires = _require_object(obj.get("requires"), f"{where}.requires")
    # The same accepted set as the TS reader (`checkKeys(requires, ["ops"], [])` in `format/ir.ts`).
    _require_keys(requires, ["ops"], [], f"{where}.requires")
    ops = requires["ops"]
    _require(isinstance(ops, list), f"{where}.requires.ops is not an array: {ops!r}")
    return [_require_string(op, f"{where}.requires.ops[]") for op in ops]


def _graph_initializers(declaration: Mapping[str, Any], where: str) -> dict[str, bool]:
    """Initializer name → whether it is shared (the only information needed to decide the set the
    binding table is cross-checked against)."""
    out: dict[str, bool] = {}
    declared = _require_object(declaration.get("initializers"), f"{where}.initializers")
    for name, value in declared.items():
        obj = _require_object(value, f"{where}.initializers['{name}']")
        _require_keys(obj, [], ["shared"], f"{where}.initializers['{name}']")
        if "shared" in obj:
            _require(
                obj["shared"] is True,
                f"{where}.initializers['{name}'].shared: only true can be written",
            )
        out[name] = obj.get("shared", False) is True
    return out


def _validate_graph_descriptor(descriptor: GraphDescriptor) -> None:
    where = "graphDescriptor.const"
    _require(
        len(descriptor.const_blocks) <= MAX_BLOCKS,
        f"{where}.blocks count exceeds the limit {MAX_BLOCKS}",
    )
    ids = {block.id for block in descriptor.const_blocks}
    _require(len(ids) == len(descriptor.const_blocks), f"{where}.blocks: duplicate block id")
    referenced: set[str] = set()

    def claim(block_id: str, by: str) -> None:
        _require(block_id in ids, f"{by}: block '{block_id}' is not in the const table")
        _require(
            block_id not in referenced,
            f"{by}: block '{block_id}' is bound twice (1 block ≤ 1 binding)",
        )
        referenced.add(block_id)

    seen: set[tuple[str, str]] = set()
    for entry in descriptor.constants:
        entry_where = f"{where}.constants['{entry.graph}/{entry.initializer}']"
        declaration = descriptor.graphs.get(entry.graph)
        _require(declaration is not None, f"{entry_where}: undeclared graph '{entry.graph}'")
        assert declaration is not None
        initializers = _graph_initializers(declaration, f"graphDescriptor.graphs['{entry.graph}']")
        _require(
            entry.initializer in initializers,
            f"{entry_where}: the graph has no initializer '{entry.initializer}'",
        )
        _require(
            not initializers[entry.initializer],
            f"{entry_where}: a shared declaration cannot be supplied by const",
        )
        key = (entry.graph, entry.initializer)
        _require(key not in seen, f"{entry_where}: duplicate (graph, initializer)")
        seen.add(key)
        claim(entry.block, entry_where)
        # A const is one block (it has no piece list — §3), so there are no piece boundaries to
        # split its scale on (rule ③).
        _require(
            entry.encoding.scale_pieces is None,
            f"{entry_where}.encoding.scale: a const cannot have a split scale (a const has no"
            " pieces — rule ③)",
        )
        if entry.encoding.scale_block is not None:
            claim(entry.encoding.scale_block, f"{entry_where}.encoding.scale")
    for block_id in sorted(ids - referenced):
        raise ContainerFormatError(
            f"{where}: block '{block_id}' is not referenced from anywhere (surplus)"
        )


def _parse_supply(value: Any, where: str) -> WeightSupply:
    obj = _require_object(value, where)
    _require_keys(obj, ["encoding"], ["block", "pieces"], where)
    encoding = _parse_encoding(obj["encoding"], f"{where}.encoding")
    has_block = "block" in obj
    has_pieces = "pieces" in obj
    _require(has_block != has_pieces, f"{where}: write exactly one of 'block' and 'pieces'")
    if has_block:
        return WeightSupply(encoding, block=_require_block_id(obj["block"], f"{where}.block"))
    return WeightSupply(encoding, pieces=_parse_pieces(obj["pieces"], f"{where}.pieces"))


def _parse_model_descriptor(raw: bytes) -> ModelDescriptor:
    where = "modelDescriptor"
    root = _require_object(_parse_json_document(raw, where), where)
    _require_keys(
        root,
        ["format", "version", "codecs", "parts", "blocks", "binding", "assets", "provenance"],
        [],
        where,
    )
    _require(root["format"] == "karume-model", f"{where}.format is not 'karume-model'")
    _require_version(root["version"], CONTAINER_VERSION, f"{where}.version")

    parts: list[PartRecord] = []
    for index, raw_part in enumerate(root["parts"]):
        part_where = f"{where}.parts[{index}]"
        obj = _require_object(raw_part, part_where)
        _require_keys(obj, ["index", "length", "sha256"], [], part_where)
        record = PartRecord(
            _require_index(obj["index"], f"{part_where}.index"),
            _require_index(obj["length"], f"{part_where}.length"),
            _require_sha256(obj["sha256"], f"{part_where}.sha256"),
        )
        _require(record.index == index + 1, f"{part_where}.index is not {index + 1}")
        _require(
            record.length <= PART_MAX_BYTES,
            f"{part_where}.length {record.length} exceeds the part length ceiling",
        )
        parts.append(record)
    _require(len(parts) > 0, f"{where}.parts is empty (part 1 is declared even with length 0)")
    _require(len(parts) <= MAX_PARTS, f"{where}.parts count exceeds the limit {MAX_PARTS}")

    blocks: list[DataBlockRecord] = []
    for index, raw_block in enumerate(root["blocks"]):
        block_where = f"{where}.blocks[{index}]"
        obj = _require_object(raw_block, block_where)
        _require_keys(obj, ["id", "part", "offset", "length", "sha256", "role"], [], block_where)
        role = _require_string(obj["role"], f"{block_where}.role")
        _require(
            role in ("weight", "scale", "zero-point", "asset"),
            f"{block_where}.role is outside the vocabulary: {role}",
        )
        record = DataBlockRecord(
            _require_block_id(obj["id"], f"{block_where}.id"),
            _require_index(obj["part"], f"{block_where}.part"),
            _require_index(obj["offset"], f"{block_where}.offset"),
            _require_index(obj["length"], f"{block_where}.length"),
            _require_sha256(obj["sha256"], f"{block_where}.sha256"),
            role,  # type: ignore[arg-type]
        )
        _require(record.part >= 2, f"{block_where}.part is less than 2")
        _require(record.part <= len(parts), f"{block_where}.part: undeclared part {record.part}")
        _assert_block_shape(record.offset, record.length, block_where)
        blocks.append(record)
    for part in parts:
        _assert_no_overlap(
            [
                (block.id, block.offset, block.length)
                for block in blocks
                if block.part == part.index
            ],
            part.length,
            f"{where} part {part.index}",
        )

    binding: dict[str, dict[str, WeightSupply]] = {}
    for graph_name, raw_supplies in _require_object(root["binding"], f"{where}.binding").items():
        supplies = _require_object(raw_supplies, f"{where}.binding['{graph_name}']")
        binding[graph_name] = {
            name: _parse_supply(value, f"{where}.binding['{graph_name}']['{name}']")
            for name, value in supplies.items()
        }

    assets: dict[str, AssetRecord] = {}
    for name, raw_asset in _require_object(root["assets"], f"{where}.assets").items():
        asset_where = f"{where}.assets['{name}']"
        _require_string(name, f"key of {where}.assets")
        obj = _require_object(raw_asset, asset_where)
        _require_keys(obj, ["block", "role", "length"], [], asset_where)
        assets[name] = AssetRecord(
            _require_block_id(obj["block"], f"{asset_where}.block"),
            _require_string(obj["role"], f"{asset_where}.role"),
            _require_index(obj["length"], f"{asset_where}.length"),
        )

    prov = _require_object(root["provenance"], f"{where}.provenance")
    _require_keys(
        prov, ["license"], ["notice", "upstreamRevision", "writer"], f"{where}.provenance"
    )
    # The 3 optional fields, when present, also require a non-empty string, as the TS reader does
    # (the accepted sets agree).
    optional = {
        key: _require_string(prov[key], f"{where}.provenance.{key}") if key in prov else None
        for key in ("notice", "upstreamRevision", "writer")
    }
    provenance = Provenance(
        _require_string(prov["license"], f"{where}.provenance.license"),
        optional["notice"],
        optional["upstreamRevision"],
        optional["writer"],
    )

    descriptor = ModelDescriptor(tuple(parts), tuple(blocks), binding, assets, provenance)
    _validate_model_descriptor(descriptor, root["codecs"])
    return descriptor


def _validate_model_descriptor(descriptor: ModelDescriptor, codecs: Any) -> None:
    where = "modelDescriptor"
    _require(
        len(descriptor.blocks) <= MAX_BLOCKS,
        f"{where}.blocks count exceeds the limit {MAX_BLOCKS}",
    )
    by_id = {block.id: block for block in descriptor.blocks}
    _require(len(by_id) == len(descriptor.blocks), f"{where}.blocks: duplicate block id")
    referenced: set[str] = set()

    def claim(block_id: str, role: BlockRole, by: str) -> DataBlockRecord:
        block = by_id.get(block_id)
        _require(block is not None, f"{by}: block '{block_id}' is not in the table")
        assert block is not None
        _require(block.role == role, f"{by}: the role of block '{block_id}' is '{block.role}'")
        _require(
            block_id not in referenced,
            f"{by}: block '{block_id}' is bound twice (1 block ≤ 1 binding)",
        )
        referenced.add(block_id)
        return block

    used: set[str] = set()
    for graph_name, supplies in descriptor.binding.items():
        for name, supply in supplies.items():
            entry_where = f"{where}.binding['{graph_name}']['{name}']"
            used.add(supply.encoding.codec)
            ids = (
                [supply.block]
                if supply.block is not None
                else [block for block, _ in supply.pieces or ()]
            )
            first = claim(ids[0], "weight", entry_where).part
            previous = first
            for block_id in ids[1:]:
                part = claim(block_id, "weight", entry_where).part
                # Rule ④: the parts of the pieces MUST be non-decreasing in index order. The
                # building side does its bookkeeping assuming piece 1 is in the first part (TS
                # rejects it in `bindGraphs` — keep the accepted sets aligned).
                _require(
                    part >= previous,
                    f"{entry_where}: part {part} comes before part {previous} of the preceding"
                    " piece (the parts of the pieces MUST be non-decreasing in index order —"
                    " rule ④)",
                )
                previous = part
            split = supply.encoding.scale_pieces is not None
            for index, scale_block in enumerate(supply.encoding.scale_blocks):
                # Rule ③: the companion scale is in the same part as the body (piece 1 for a
                # piece sequence) — every block of a split scale too (construction reads the
                # whole scale with piece 1).
                scale_where = f"{entry_where}.encoding.scale" + (
                    f".pieces[{index}]" if split else ""
                )
                scale_part = claim(scale_block, "scale", scale_where).part
                _require(
                    scale_part == first,
                    f"{scale_where}: scale is in part {scale_part}, the start of"
                    f" the body in part {first} (MUST be the same part)",
                )
    for name, asset in descriptor.assets.items():
        asset_where = f"{where}.assets['{name}']"
        block = claim(asset.block, "asset", asset_where)
        # The declaration covers even the amount of padding (the consumer does not strip
        # trailing 0x00 by guessing — §2.2).
        _require(
            align_up(asset.length, BLOCK_TAIL_ALIGN) == block.length,
            f"{asset_where}.length: logical length {asset.length} rounded up to a multiple of"
            f" {BLOCK_TAIL_ALIGN} differs from the length of block '{asset.block}',"
            f" {block.length}",
        )
    for block_id in sorted(set(by_id) - referenced):
        raise ContainerFormatError(
            f"{where}: block '{block_id}' is not referenced from anywhere (surplus)"
        )
    declared = [_require_string(codec, f"{where}.codecs[]") for codec in codecs]
    for codec in declared:
        codec_entry(codec)
    _require(
        declared == sorted(used),
        f"{where}.codecs {declared} does not match the codec set of the binding table"
        f" {sorted(used)}",
    )


def _validate_against_graph(model: ModelDescriptor, graph: GraphDescriptor) -> None:
    """Cross-check between the 2 documents (§2 / rule ⑤ of §5)."""
    # The block count limit applies to the whole container (each document can be within the limit
    # while the sum exceeds it — the same place as TS).
    _require(
        len(graph.const_blocks) + len(model.blocks) <= MAX_BLOCKS,
        f"Block count (const {len(graph.const_blocks)} + weights {len(model.blocks)}) exceeds"
        f" the limit {MAX_BLOCKS}",
    )
    _require(
        model.parts[0].length == graph.const_length,
        f"Part 1 length {model.parts[0].length} differs from const.length {graph.const_length}",
    )
    const_ids = {block.id for block in graph.const_blocks}
    for block in model.blocks:
        _require(
            block.id not in const_ids,
            f"Block id '{block.id}' collides between the const table and the model table (MUST"
            " be disjoint)",
        )
    supplied = {(entry.graph, entry.initializer) for entry in graph.constants}
    for graph_name in model.binding:
        _require(
            graph_name in graph.graphs,
            f"modelDescriptor.binding: undeclared graph '{graph_name}'",
        )
    for graph_name, declaration in graph.graphs.items():
        initializers = _graph_initializers(declaration, f"graphDescriptor.graphs['{graph_name}']")
        expected = sorted(
            name
            for name, shared in initializers.items()
            if not shared and (graph_name, name) not in supplied
        )
        bound = model.binding.get(graph_name, {})
        missing = [name for name in expected if name not in bound]
        surplus = sorted(set(bound) - set(expected))
        _require(
            not missing and not surplus,
            f"The binding table of graph '{graph_name}' does not match the initializer"
            f" declarations: missing [{', '.join(missing)}] / surplus [{', '.join(surplus)}]",
        )


@dataclass(frozen=True)
class _PartSlice:
    path: Path
    offset: int
    length: int


class ReadContainer:
    """An opened container (the reader for the self-check). A block is taken when needed, and its
    sha256 is verified every time it is taken."""

    def __init__(
        self,
        header: ContainerHeader,
        graph_bytes: bytes,
        model_bytes: bytes,
        graph: GraphDescriptor,
        model: ModelDescriptor | None,
        slices: Sequence[_PartSlice],
    ) -> None:
        self.header = header
        self.graph_descriptor_bytes = graph_bytes
        self.model_descriptor_bytes = model_bytes
        self.graph = graph
        self.model = model
        self._slices = list(slices)
        self._located: dict[str, tuple[int, int, int, str]] = {}
        for block in graph.const_blocks:
            self._located[block.id] = (1, block.offset, block.length, block.sha256)
        for block in model.blocks if model is not None else ():
            self._located[block.id] = (block.part, block.offset, block.length, block.sha256)
        #: Split scales, read whole under their joined id (ADR 0108 addendum 15) → their pieces.
        self._joined: dict[str, tuple[str, ...]] = {}
        for supplies in model.binding.values() if model is not None else ():
            for supply in supplies.values():
                if supply.encoding.scale_pieces is not None:
                    pieces = supply.encoding.scale_blocks
                    self._joined[joined_scale_id(pieces)] = pieces

    @property
    def block_ids(self) -> list[str]:
        """The declared blocks (joined scale ids are not blocks)."""
        return list(self._located)

    def _read(self, part: int, offset: int, length: int) -> bytes:
        _require(0 <= part < len(self._slices), f"part {part} does not exist")
        target = self._slices[part]
        _require(
            offset + length <= target.length,
            f"part {part}: {offset}..{offset + length} runs past the part length {target.length}",
        )
        with target.path.open("rb") as handle:
            handle.seek(target.offset + offset)
            raw = handle.read(length)
        _require(len(raw) == length, f"part {part}: could not read {length} bytes")
        return raw

    def block(self, block_id: str) -> bytes:
        """Take one block, compare it against the declared sha256 and return it (the cold path of
        §7).

        The joined id of a split scale (`InitializerSupply.scale.id` of `karume.verify` — ADR 0108
        addendum 15) returns its pieces concatenated in row order, each verified like a block.
        """
        pieces = self._joined.get(block_id)
        if pieces is not None:
            return b"".join(self.block(piece) for piece in pieces)
        found = self._located.get(block_id)
        _require(found is not None, f"Undeclared block '{block_id}'")
        assert found is not None
        part, offset, length, expected = found
        raw = self._read(part, offset, length)
        actual = hashlib.sha256(raw).hexdigest()
        _require(
            actual == expected,
            f"The sha256 of block '{block_id}' differs from the declaration (declared {expected}"
            f" / actual {actual})",
        )
        return raw

    def verify_blocks(self) -> dict[str, str]:
        """Take every block again and verify its sha256 (returns block id → declared sha256)."""
        for block_id in self._located:
            self.block(block_id)
        return {block_id: found[3] for block_id, found in self._located.items()}

    def verify_parts(self) -> None:
        """Compare the sha256 of each whole part against the declaration (for cross-checks at
        publication and repacking — §7).

        Unlike a block, a part can be up to 1 GiB, so it is **digested in steps** (the whole is
        never held in a buffer).
        """
        _require(self.model is not None, "krg declares no parts")
        assert self.model is not None
        for part in self.model.parts:
            digest = hashlib.sha256()
            for offset in range(0, part.length, _READ_CHUNK_BYTES):
                digest.update(
                    self._read(part.index, offset, min(_READ_CHUNK_BYTES, part.length - offset))
                )
            actual = digest.hexdigest()
            _require(
                actual == part.sha256,
                f"The sha256 of part {part.index} differs from the declaration (declared"
                f" {part.sha256} / actual {actual})",
            )

    def extract_graph(self) -> bytes:
        """Build `[header'][graph descriptor][padding][const region]` (the byte-copy extraction of
        §9)."""
        length = self.graph.const_length
        region = b"" if length == 0 else self._read(1, 0, length)
        if self.model is not None:
            actual = hashlib.sha256(region).hexdigest()
            expected = self.model.parts[0].sha256
            _require(
                actual == expected,
                f"The sha256 of the const region differs from the declaration (declared"
                f" {expected} / actual {actual})",
            )
        header = write_header(
            kind="graph", graph_length=len(self.graph_descriptor_bytes), model_length=0
        )
        offsets = derive_part_offsets(HEADER_BYTES + len(self.graph_descriptor_bytes), [length])
        body = header + self.graph_descriptor_bytes
        return body + PAD_BYTE * (offsets[1] - len(body)) + region


def read_container(paths: Sequence[Path]) -> ReadContainer:
    """Re-read what was written (header, parse of the 2 documents, the declaration checks of §2 /
    §5).

    `paths` is one file in single form, and in split form the part sequence **in order from part
    0**.
    """
    _require(len(paths) > 0, "No container file was passed")
    first = Path(paths[0])
    with first.open("rb") as handle:
        head = handle.read(HEADER_BYTES)
    header = read_header(head)
    part0 = header.part0_length
    with first.open("rb") as handle:
        handle.seek(HEADER_BYTES)
        graph_bytes = handle.read(header.graph_length)
        model_bytes = handle.read(header.model_length)
    _require(
        len(graph_bytes) == header.graph_length,
        "The graph descriptor is shorter than its declared length",
    )
    _require(
        len(model_bytes) == header.model_length,
        "The model descriptor is shorter than its declared length",
    )

    graph = _parse_graph_descriptor(graph_bytes)
    model: ModelDescriptor | None = None
    if header.kind == "model":
        model = _parse_model_descriptor(model_bytes)
        _validate_against_graph(model, graph)
    part_lengths = [graph.const_length] if model is None else [part.length for part in model.parts]

    if len(paths) == 1:
        offsets = derive_part_offsets(part0, part_lengths)
        total = offsets[-1] + part_lengths[-1] if part_lengths else part0
        size = first.stat().st_size
        _require(
            size == total,
            f"Single-form length {size} differs from {total} derived from the declarations",
        )
        slices = [_PartSlice(first, 0, part0)] + [
            _PartSlice(first, offsets[index + 1], length)
            for index, length in enumerate(part_lengths)
        ]
    else:
        _require(
            len(paths) == len(part_lengths) + 1,
            f"{len(paths)} parts but the declaration has {len(part_lengths) + 1}",
        )
        slices = [_PartSlice(first, 0, part0)]
        for index, length in enumerate(part_lengths):
            path = Path(paths[index + 1])
            _require(
                path.stat().st_size == length,
                f"Part {index + 1} length {path.stat().st_size} differs from the declared {length}",
            )
            slices.append(_PartSlice(path, 0, length))
        _require(
            first.stat().st_size == part0,
            f"Part 0 length {first.stat().st_size} differs from the declared {part0}",
        )
    return ReadContainer(header, graph_bytes, model_bytes, graph, model, slices)
