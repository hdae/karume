"""The **read-only** layer for the old distribution (shard sequences of the safetensors dialect),
for the migration CLI only.

The distribution moved to `krm` (container-v1), so no code **writes** the safetensors dialect any
more. The public repositories were re-uploaded at 0.13.0, but the old form remains in the old HF
revisions (what pinned readers point at), so only the path by which `karume migrate` reads the old
form stays here (container-v1 §12 "Migration CLI contract": "Reading the old format lives **only
in this CLI**").

What this reads are sequences that include distributions under the old rules (single file / fat
graph shard / tail slack / piece key `<parent>#NNNNN-of-NNNNN`), and the current checks are not
applied (applying them would make migration impossible). A split tensor is **folded into its one
parent** while reading, so the caller always sees only "all bytes of the parent".

MUST: do not bring in torch. This handles **only byte sequences and declarations**; raising
tensors into torch and carrying them around leaves room for silently wrong values through a dtype
round trip (the f16 / i4 carrier types).

MUST: do not use this to **write** anything new. The source of truth for the numbered file name
convention (`<stem>-NNNNN-of-NNNNN`) is the container side ({@link
karume.container.numbered_name}); this only borrows it to resolve sequences of the old form.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from karume.container import resolve_sequence
from karume.verify import assert_reader_layout

#: The `__metadata__` key that carried the graph JSON in the old distribution (safetensors
#: dialect); only the migration CLI's `karume.legacy` / `karume.migrate` read it.
IR_METADATA_KEY = "karume_ir"

#: The safetensors dtype the old writer wrote for each storage dtype (IR v1 `storage.dtype`), one
#: to one.
#:
#: MUST: the shape is written as the **logical shape** (packed i4 / i2 is not the carrier shape
#: either — ADR 0069 decision 2). Migration checks the old tensor declarations against this table
#: and the declared IR shape before writing — looking at the byte length alone would silently
#: write a different type of the same length (reading an I32 1 as f32 gives about 1.4e-45) or a
#: different shape with only the same element count into the new container. The PLE sidecar's
#: `values` are looked up in the same table.
#:
#: There is no `bf16` row — the old writer could not write bf16 (storage output was only f32 / f16 /
#: i8 / i4, and the torch → safetensors table has no bfloat16 either). Storage not in the table
#: fails loudly in migration (adding a row by guess would silently accept input the old writer could
#: never have produced).
LEGACY_STORED_DTYPE: Mapping[str, str] = {
    "f32": "F32",
    "f16": "F16",
    "i32": "I32",
    "i8": "I8",
    "i4": "I4",
    "i2": "I2",
}

#: The safetensors dtype of the old writer's scales (companion scale, PLE `scales`).
LEGACY_SCALE_DTYPE = "F32"

#: Read unit for raw bytes. Its only requirement is not reading several GB at once; the value
#: itself is a plain I/O unit.
_CHUNK_BYTES = 1 << 20

#: Byte count of the safetensors header length field.
_HEADER_LENGTH_BYTES = 8

#: Digit count of the piece key's sequence numbers (old shard spec v3, ADR 0090 decision 1 — the
#: same spelling as the regular expression the `packages/runtime` reader had until ADR 0108 step 3d
#: removed it).
_PIECE_DIGITS = 5

#: Spelling of a piece key (`<parent>#NNNNN-of-NNNNN`). The parent name is taken greedily — parent
#: names containing `#` never appear in the old writer's output, so only the trailing sequence
#: number marks the split. MUST: ASCII digits only — Python's `\d` also matches Unicode digits
#: (full-width U+FF11), which the old writer never wrote and `int()` would still convert.
_PIECE_PATTERN = re.compile(rf"^(.+)#([0-9]{{{_PIECE_DIGITS}}})-of-([0-9]{{{_PIECE_DIGITS}}})$")


class LegacyFormatError(ValueError):
    """The old distribution cannot be read (inconsistent header, inconsistent piece sequence,
    duplicate tensor)."""


def parse_piece_key(key: str) -> tuple[str, int, int] | None:
    """Piece key → `(parent name, index, count)` (None when it is not a piece).

    MUST: check `count >= 2` and `1 <= index <= count` **here**. Passing an out-of-range spelling
    (`#00003-of-00002`, a piece sequence of only one) as a piece makes it the initial value of the
    sequence's progress state, and the violation gets attributed to "the order of the piece
    sequence". In fact the key itself is the old distribution's error, so carrying it as one tensor
    rather than reading it as a piece is more decisive for whoever fixes it.
    """
    match = _PIECE_PATTERN.fullmatch(key)
    if match is None:
        return None
    index, count = int(match.group(2)), int(match.group(3))
    if count < 2 or not 1 <= index <= count:
        return None
    return match.group(1), index, count


def resolve_shards(path: Path) -> tuple[Path, ...]:
    """Representative path of an old component → the existing shard sequence (read order =
    sequence-number order).

    The numbering convention is the same as for new parts (container-v1 §8), so resolution borrows
    the single path of {@link karume.container.resolve_sequence}.
    """
    return resolve_sequence(path)


@dataclass(frozen=True)
class StoredEntry:
    """Declaration of one tensor in an old container (safetensors dtype, **logical** shape, byte
    length)."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class SourceTensor:
    """One tensor on the input side — its **declaration as the parent** and where the bytes are.

    A split tensor has its pieces scattered over separate files, so the location is not one point
    but a **sequence of ranges** (`(file, absolute offset from the start of the file, byte
    length)`, in piece index order). A whole tensor is a sequence of one element; the later paths
    (sha256, copy, row-range slicing) look only at the sequence of ranges, so they do not branch.
    """

    entry: StoredEntry
    segments: tuple[tuple[Path, int, int], ...]


def _read_header(path: Path) -> tuple[Mapping[str, Any], int]:
    """Return the header JSON and the **absolute start of the data section**.

    NOTE: the layout rules (known dtype, matching declared byte length, no gaps, alignment) belong
    to {@link karume.verify.assert_reader_layout}, which the caller runs **first**. This is only
    the read after that has passed, so plain indexing is fine.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(_HEADER_LENGTH_BYTES), "little")
        # Bound the declared length by the actual file length (reading an invalid 8 bytes as is
        # would be a huge allocation).
        if header_length <= 0 or header_length > size - _HEADER_LENGTH_BYTES:
            raise LegacyFormatError(
                f"{path}: header length {header_length} contradicts the file length {size}"
            )
        header = json.loads(handle.read(header_length))
    if not isinstance(header, dict):
        raise LegacyFormatError(f"{path}: the safetensors header is not a top-level object")
    return header, _HEADER_LENGTH_BYTES + header_length


def safetensors_header(path: Path) -> Mapping[str, Any]:
    """Read only the header JSON (without scanning a payload of several GB)."""
    return _read_header(path)[0]


def _read_metadata(path: Path, header: Mapping[str, Any]) -> dict[str, str]:
    """Take `__metadata__` out as string → string (the only entry that carries it **verbatim**)."""
    raw = header.get("__metadata__", {})
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
    ):
        raise LegacyFormatError(f"{path}: __metadata__ is not a string → string map")
    return dict(raw)


@dataclass(frozen=True)
class _Fragment:
    """One piece being read (before folding — the material for checking the sequence numbers and
    the order of the shards that hold them)."""

    shard: int
    index: int
    count: int
    entry: StoredEntry
    segment: tuple[Path, int, int]


def _join_pieces(name: str, fragments: Sequence[_Fragment]) -> SourceTensor:
    """Fold a piece sequence into its one parent (all of old reader contract 5 is checked here).

    `fragments` is in **read order** (shard number order). The folded declaration is the parent's
    dtype, the whole shape (leading dimension = sum of each piece's rows) and the total byte
    length; the location is the pieces' sequence of ranges.
    """
    shards = [fragment.shard for fragment in fragments]
    repeated = sorted({shard for shard in shards if shards.count(shard) > 1})
    if repeated:
        raise LegacyFormatError(
            f"Tensor '{name}': shard{repeated} holds two pieces of the same parent"
            " (one piece per consecutive shard)"
        )
    # `parse_piece_key` already checked count >= 2 and the index range (an out-of-range spelling
    # is not read as a piece and is carried as one tensor under its name as is).
    count = fragments[0].count
    if len(fragments) != count:
        raise LegacyFormatError(
            f"Tensor '{name}': {len(fragments)} pieces do not match the declared total {count}"
        )
    head = fragments[0].entry
    rows = 0
    previous: int | None = None
    for position, fragment in enumerate(fragments, start=1):
        entry = fragment.entry
        if fragment.count != count:
            raise LegacyFormatError(
                f"Tensor '{name}': the piece totals disagree, {count} and {fragment.count}"
            )
        if fragment.index != position:
            raise LegacyFormatError(
                f"Tensor '{name}': piece {position} in shard order has index {fragment.index}"
                " (index rises from 1 in shard order)"
            )
        if previous is not None and fragment.shard != previous + 1:
            raise LegacyFormatError(
                f"Tensor '{name}': piece {fragment.index} is in shard[{fragment.shard}], the"
                f" previous piece in shard[{previous}] (one piece per consecutive shard)"
            )
        if entry.dtype != head.dtype:
            raise LegacyFormatError(
                f"Tensor '{name}': the dtype of piece {fragment.index} is {entry.dtype}"
                f" (piece 1 is {head.dtype})"
            )
        if entry.shape[1:] != head.shape[1:]:
            raise LegacyFormatError(
                f"Tensor '{name}': piece {fragment.index} has remaining dimensions"
                f" {list(entry.shape[1:])}, which differ from piece 1's {list(head.shape[1:])}"
            )
        if not entry.shape or entry.shape[0] < 1:
            raise LegacyFormatError(f"Tensor '{name}': piece {fragment.index} has fewer than 1 row")
        if position < count and entry.nbytes % 4:
            raise LegacyFormatError(
                f"Tensor '{name}': non-final piece {fragment.index} is {entry.nbytes} bytes, not"
                " a multiple of 4 (the reader cannot write it at an offset into the parent buffer)"
            )
        rows += entry.shape[0]
        previous = fragment.shard
    return SourceTensor(
        entry=StoredEntry(
            name=name,
            dtype=head.dtype,
            shape=(rows, *head.shape[1:]),
            nbytes=sum(fragment.entry.nbytes for fragment in fragments),
        ),
        segments=tuple(fragment.segment for fragment in fragments),
    )


def read_component(paths: Sequence[Path]) -> tuple[dict[str, str], dict[str, SourceTensor]]:
    """Read the files of an old component and return `(__metadata__ of the first shard, name →
    source tensor)`.

    Accepted are sequences that **include distributions under the old rules** (single file / fat
    graph shard / tail slack). Even so, "only the first shard has `karume_ir`" and "no two tensors
    share a name" were invariants under the old rules too, so they are checked here (migrating
    something that breaks them leaves it undetermined which byte sequence to distribute).

    A split tensor (piece key) is **folded into its one parent** ({@link _join_pieces} checks old
    reader contract 5). Folding is decided by shape alone — the input's `karume_ir` is not parsed
    yet, so "is the parent name declared" is not checked here; the migration side's cross-check
    (`migrate._assert_tensor_cover`) rejects it as surplus.
    """
    metadata: dict[str, str] = {}
    stored: dict[str, SourceTensor] = {}
    fragments: dict[str, list[_Fragment]] = {}
    owner: dict[str, Path] = {}
    for index, path in enumerate(paths):
        if not path.is_file():
            raise LegacyFormatError(f"Migration input does not exist: {path}")
        assert_reader_layout(path, allow_legacy_dtypes=True)
        header, data_start = _read_header(path)
        raw = _read_metadata(path, header)
        if index == 0:
            metadata = raw
        elif raw:
            raise LegacyFormatError(
                f"{path}: shard[{index}] has __metadata__"
                f" ({sorted(raw)} — only the first shard carries metadata)"
            )
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in owner:
                raise LegacyFormatError(
                    f"Tensor '{name}' is duplicated in {owner[name].name} and {path.name}"
                )
            owner[name] = path
            begin, end = spec["data_offsets"]
            entry = StoredEntry(
                name=name, dtype=spec["dtype"], shape=tuple(spec["shape"]), nbytes=end - begin
            )
            segment = (path, data_start + begin, end - begin)
            parsed = parse_piece_key(name)
            if parsed is None:
                stored[name] = SourceTensor(entry=entry, segments=(segment,))
                continue
            parent, piece_index, piece_count = parsed
            fragments.setdefault(parent, []).append(
                _Fragment(
                    shard=index,
                    index=piece_index,
                    count=piece_count,
                    entry=replace(entry, name=parent),
                    segment=segment,
                )
            )
    for parent in sorted(fragments):
        if parent in stored:
            raise LegacyFormatError(
                f"Tensor '{parent}' is in the container both whole and as pieces"
                " (a tensor MUST be one or the other)"
            )
        stored[parent] = _join_pieces(parent, fragments[parent])
    if IR_METADATA_KEY not in metadata:
        raise LegacyFormatError(
            f"{paths[0]}: __metadata__.{IR_METADATA_KEY} is missing (not an old Karume"
            " distribution)"
        )
    return metadata, stored


def _range_chunks(source: SourceTensor, begin: int, end: int) -> Iterator[bytes]:
    """Stream the parent's **byte range** `[begin, end)` one read unit at a time (no whole read).

    The range may cross the sequence of ranges (= the order of the pieces) — the old form's cuts
    are unrelated to the block cuts after migration, so the reader sees only "the parent's
    contiguous byte sequence".
    """
    cursor = 0
    for path, offset, length in source.segments:
        stop = cursor + length
        if stop <= begin or cursor >= end:
            cursor = stop
            continue
        lower, upper = max(begin, cursor), min(end, stop)
        with path.open("rb") as handle:
            handle.seek(offset + lower - cursor)
            remaining = upper - lower
            while remaining:
                chunk = handle.read(min(_CHUNK_BYTES, remaining))
                if not chunk:
                    raise LegacyFormatError(
                        f"{path}: the data section ran out partway through the declaration of"
                        f" tensor '{source.entry.name}' ({source.entry.nbytes} bytes)"
                    )
                remaining -= len(chunk)
                yield chunk
        cursor = stop


def payload_chunks(source: SourceTensor) -> Iterator[bytes]:
    """Stream one tensor (all bytes of the parent) one read unit at a time."""
    return _range_chunks(source, 0, source.entry.nbytes)
