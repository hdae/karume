"""CLI that compares rewritten Wan outputs with an existing series (ADR 0121 step 4 — checking
that the 2.1 series is unchanged).

After a recipe change, the 2.1 outputs (the VAE chunk graphs, fixtures and so on) are rewritten
into a scratch directory and compared with the existing series, only for the items named with
`--require`. Because this is a check, what the design avoids is saying "match" when the two differ
(a false match):

- There are two ways to compare.
  - `*.safetensors`: the set of tensor names, and per tensor the dtype / shape / raw bytes (a bit
    match — neither NaN payloads nor -0.0 are folded), plus equality of the `__metadata__` dict
    (key order is ignored). In safetensors 0.8.0 the key order of `__metadata__` changes from
    process to process, so the files cannot be compared byte for byte (ADR 0118 addendum
    "byte-identical regeneration"). The header is read by our own code ({@link read_header}), which
    also checks that the tensors fill the data area with no gaps — bytes outside the tensors are
    never silently missed. The comparison looks up the contents by name, and deliberately ignores
    the order of the tensors in the data area (data_offsets) and the whitespace padding of the
    header (if the contents read back are the same, the output is the same). `"__metadata__": null`
    is treated the same as an absent key (safetensors itself reads null as "absent").
  - Anything else (krm parts and so on): size and sha256.
- For a directory item, the set of relative paths on both sides (recursive, directories included)
  must be the same before each file is compared. An item missing on either side is a mismatch (not
  skipped). A directory item with no file at all to compare is also a mismatch — a match that
  compared nothing is a sign that the preparation of both sides is broken.
- Symbolic links are not followed: a link inside an item (the item itself, an intermediate
  component, the contents of a directory) is named as a mismatch. The writers of a series do not
  create links, so one is unexpected — following it and comparing would give a match whenever both
  sides point at the same target, whatever its contents. The roots (`--written` / `--existing`) are
  paths the user gave explicitly, so they may be followed. An item that is neither a regular file
  nor a directory (a FIFO and so on) is a mismatch too. The kind is read with lstat and the file is
  then opened by path, so a concurrent writer that swaps in a link in between is not considered (no
  writer runs on a root while it is being compared).
- If both sides are the same file (the same inode — the same root passed twice, a hard link), that
  is a mismatch. A comparison with itself always matches, so it is no check. A copy, however, is a
  different file and does match — the check only means something when the written side is the
  result of running the writers into an empty scratch directory (the procedure in `wan/README.md`).

Usage (from `tools/export-recipes`):

    uv run --group wan --inexact python -m wan.series_check --written <dir> --existing <dir> \\
      --require vae_decoder_first --require vae_chunks.band.safetensors

Writes one JSON object to standard output (the items compared, the result per file, the mismatches
and their reasons). The exit code is 0 only when there is no mismatch, 1 when there is a mismatch,
2 for an argument error (no `--require`, a name outside the root or naming the root itself, a name
containing NUL, a duplicate, a name nested inside another `--require`, a root that is not a
directory), and 3 for an unreadable file (permissions and so on — OSError; standard output stays
empty and the reason goes to standard error).

This is a separate entry point from the `--check` of export_dit (`series_mismatches` — only the
bytes directly under a component directory) (the design of ADR 0121 step 4 — export_dit itself is
not changed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

#: The kind of an item (from lstat — links are not followed).
FILE = "file"
DIR = "dir"
SYMLINK = "symlink"
OTHER = "other"
MISSING = "missing"

SIDE_LABELS = {"written": "written side", "existing": "existing side"}

SAFETENSORS_SUFFIX = ".safetensors"

#: Upper bound of the safetensors header length (100 MB, the same as safetensors itself — a broken
#: length does not cause a huge read).
MAX_HEADER_BYTES = 100_000_000

#: dtype → bytes per element (safetensors dtype names). A dtype not listed here (sub-byte F4 and so
#: on) is treated as unreadable and fails loudly — the element count is never guessed.
ITEM_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

#: The unit in which tensor bytes are read (16 MiB — a multiple of the element size of every dtype.
#: Even for a file of hundreds of MB, only 2 × this size is held at once).
CHUNK_BYTES = 1 << 24


class SafetensorsFormatError(ValueError):
    """Not readable as safetensors (the header, a dtype or the data-area layout breaks the
    format)."""


@dataclass(frozen=True)
class TensorEntry:
    """One tensor of the header (`begin` / `end` are byte offsets from the start of the data
    area)."""

    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


@dataclass(frozen=True)
class SafetensorsHeader:
    """A safetensors header (`data_start` is the position of the data area in the file)."""

    data_start: int
    tensors: dict[str, TensorEntry]
    #: None for a file without `__metadata__` (or with null) — kept distinct from an empty dict.
    metadata: dict[str, str] | None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    counts = Counter(key for key, _ in pairs)
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    if duplicated:
        raise SafetensorsFormatError(f"duplicate keys in the header {duplicated}")
    return dict(pairs)


def _tensor_entry(name: str, raw: Any) -> TensorEntry:
    if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data_offsets"}:
        raise SafetensorsFormatError(
            f"entry of tensor '{name}' is not dtype / shape / data_offsets"
        )
    dtype, shape, offsets = raw["dtype"], raw["shape"], raw["data_offsets"]
    if dtype not in ITEM_BYTES:
        raise SafetensorsFormatError(f"tensor '{name}' has unknown dtype {dtype!r}")
    if not isinstance(shape, list) or not all(type(dim) is int and dim >= 0 for dim in shape):
        raise SafetensorsFormatError(
            f"tensor '{name}' has shape {shape!r}, not a list of non-negative ints"
        )
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(type(offset) is int for offset in offsets)
        or not 0 <= offsets[0] <= offsets[1]
    ):
        raise SafetensorsFormatError(f"tensor '{name}' has broken data_offsets {offsets!r}")
    begin, end = offsets
    expected = math.prod(shape) * ITEM_BYTES[dtype]
    if end - begin != expected:
        raise SafetensorsFormatError(
            f"tensor '{name}' has {end - begin} bytes, not the shape {shape} × {dtype} = {expected}"
        )
    return TensorEntry(dtype, tuple(shape), begin, end)


def read_header(path: Path) -> SafetensorsHeader:
    """Read a safetensors header and check the format (the tensor bytes are not read).

    Requires the tensors to fill the data area from its start with no gaps or overlaps and to end at
    the end of the file (the same direction as the checks of safetensors itself). Bytes outside the
    tensors would be missed by a comparison of tensors and metadata alone, so they fail loudly.
    """
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafetensorsFormatError(f"file has only {len(prefix)} bytes (no header length)")
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_HEADER_BYTES or 8 + length > size:
            raise SafetensorsFormatError(
                f"header length {length} does not fit in a {size}-byte file"
            )
        raw = stream.read(length)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafetensorsFormatError(f"header is not UTF-8 JSON: {error}") from error
    if not isinstance(header, dict):
        raise SafetensorsFormatError("header is not a JSON object")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict) and all(isinstance(value, str) for value in metadata.values())
    ):
        raise SafetensorsFormatError("__metadata__ is not a string → string object")
    tensors = {name: _tensor_entry(name, raw_entry) for name, raw_entry in header.items()}
    position = 0
    for name, entry in sorted(tensors.items(), key=lambda item: (item[1].begin, item[1].end)):
        if entry.begin != position:
            raise SafetensorsFormatError(
                f"tensor '{name}' at position {entry.begin} follows data ending at {position}"
                " (gap or overlap in the data area)"
            )
        position = entry.end
    data_bytes = size - 8 - length
    if position != data_bytes:
        raise SafetensorsFormatError(
            f"of the {data_bytes} bytes of the data area, tensors cover only {position}"
        )
    return SafetensorsHeader(8 + length, tensors, metadata)


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    data = stream.read(length)
    if len(data) != length:
        raise SafetensorsFormatError(f"expected to read {length} bytes, got only {len(data)}")
    return data


def _diff_tensor_bytes(
    written: BinaryIO, existing: BinaryIO, length: int, item_bytes: int
) -> tuple[int | None, int]:
    """Compare `length` bytes of two aligned streams (the position of the first differing byte and
    the number of differing elements). `(None, 0)` when all are equal."""
    first: int | None = None
    differing = 0
    for position in range(0, length, CHUNK_BYTES):
        size = min(CHUNK_BYTES, length - position)
        left = _read_exact(written, size)
        right = _read_exact(existing, size)
        if left == right:
            continue
        unequal = np.frombuffer(left, dtype=np.uint8) != np.frombuffer(right, dtype=np.uint8)
        if first is None:
            first = position + int(np.flatnonzero(unequal)[0])
        differing += int(unequal.reshape(-1, item_bytes).any(axis=1).sum())
    return first, differing


def _metadata_mismatches(
    path: str, written: dict[str, str] | None, existing: dict[str, str] | None
) -> list[dict[str, Any]]:
    if written is None or existing is None:
        if written is existing:
            return []
        side = "written" if written is None else "existing"
        return [{"path": path, "reason": f"__metadata__ missing on the {SIDE_LABELS[side]}"}]
    mismatches: list[dict[str, Any]] = []
    for key in sorted(written.keys() | existing.keys()):
        if key not in existing:
            mismatches.append(
                {
                    "path": path,
                    "reason": "metadata key missing on the existing side",
                    "key": key,
                    "written": written[key],
                }
            )
        elif key not in written:
            mismatches.append(
                {
                    "path": path,
                    "reason": "metadata key missing on the written side",
                    "key": key,
                    "existing": existing[key],
                }
            )
        elif written[key] != existing[key]:
            mismatches.append(
                {
                    "path": path,
                    "reason": "metadata value differs",
                    "key": key,
                    "written": written[key],
                    "existing": existing[key],
                }
            )
    return mismatches


def compare_safetensors(
    written: Path, existing: Path, path: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare two safetensors files by tensors (name, dtype, shape, raw bytes) and metadata (dict
    equality)."""
    headers: dict[str, SafetensorsHeader] = {}
    mismatches: list[dict[str, Any]] = []
    for side, file in (("written", written), ("existing", existing)):
        try:
            headers[side] = read_header(file)
        except SafetensorsFormatError as error:
            mismatches.append(
                {
                    "path": path,
                    "reason": f"{SIDE_LABELS[side]} is not readable as safetensors: {error}",
                }
            )
    result: dict[str, Any] = {"path": path, "compare": "safetensors"}
    if mismatches:
        return result | {"match": False}, mismatches
    left, right = headers["written"], headers["existing"]
    for name in sorted(left.tensors.keys() - right.tensors.keys()):
        mismatches.append(
            {"path": path, "reason": "tensor missing on the existing side", "tensor": name}
        )
    for name in sorted(right.tensors.keys() - left.tensors.keys()):
        mismatches.append(
            {"path": path, "reason": "tensor missing on the written side", "tensor": name}
        )
    common = sorted(left.tensors.keys() & right.tensors.keys())
    with written.open("rb") as written_stream, existing.open("rb") as existing_stream:
        for name in common:
            mine, theirs = left.tensors[name], right.tensors[name]
            if mine.dtype != theirs.dtype:
                mismatches.append(
                    {
                        "path": path,
                        "reason": "dtype differs",
                        "tensor": name,
                        "written": mine.dtype,
                        "existing": theirs.dtype,
                    }
                )
                continue
            if mine.shape != theirs.shape:
                mismatches.append(
                    {
                        "path": path,
                        "reason": "shape differs",
                        "tensor": name,
                        "written": list(mine.shape),
                        "existing": list(theirs.shape),
                    }
                )
                continue
            written_stream.seek(left.data_start + mine.begin)
            existing_stream.seek(right.data_start + theirs.begin)
            item_bytes = ITEM_BYTES[mine.dtype]
            first, differing = _diff_tensor_bytes(
                written_stream, existing_stream, mine.end - mine.begin, item_bytes
            )
            if first is not None:
                mismatches.append(
                    {
                        "path": path,
                        "reason": "value bits differ",
                        "tensor": name,
                        "dtype": mine.dtype,
                        "shape": list(mine.shape),
                        "first_differing_byte_in_tensor": first,
                        "first_differing_element": first // item_bytes,
                        "differing_elements": differing,
                        "elements": math.prod(mine.shape),
                    }
                )
    mismatches += _metadata_mismatches(path, left.metadata, right.metadata)
    result |= {
        "match": not mismatches,
        "tensors": len(common),
        "tensor_bytes": sum(entry.end - entry.begin for entry in left.tensors.values()),
    }
    return result, mismatches


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_bytes(
    written: Path, existing: Path, path: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare two files other than safetensors by size and sha256."""
    sides = {}
    for side, file in (("written", written), ("existing", existing)):
        sides[side] = {"size": file.stat().st_size, "sha256": _sha256(file)}
    result: dict[str, Any] = {"path": path, "compare": "sha256", **sides}
    if sides["written"] == sides["existing"]:
        return result | {"match": True}, []
    reason = (
        "size differs"
        if sides["written"]["size"] != sides["existing"]["size"]
        else "sha256 differs"
    )
    return result | {"match": False}, [{"path": path, "reason": reason, **sides}]


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return SYMLINK
    if stat.S_ISDIR(mode):
        return DIR
    if stat.S_ISREG(mode):
        return FILE
    return OTHER


def _lookup(root: Path, name: PurePosixPath) -> tuple[str, PurePosixPath]:
    """Look up the kind of `root / name` without following links, intermediate components included
    (the kind and the position where the lookup stopped).

    If an intermediate component is a link, a FIFO or the like, the lookup stops there and returns
    its kind. If an intermediate component is a regular file, the item does not exist.
    """
    at = PurePosixPath()
    for index, part in enumerate(name.parts):
        at /= part
        try:
            kind = _kind((root / at).lstat().st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return MISSING, at
        last = index == len(name.parts) - 1
        if not last and kind != DIR:
            return (MISSING if kind == FILE else kind), at
    return kind, at


def _listing(directory: Path) -> dict[str, str]:
    """The contents of a directory (relative path → kind; recursive; links are listed as links, not
    followed)."""
    entries: dict[str, str] = {}
    pending = [PurePosixPath()]
    while pending:
        relative = pending.pop()
        with os.scandir(directory / relative) as scan:
            for entry in scan:
                child = relative / entry.name
                kind = _kind(entry.stat(follow_symlinks=False).st_mode)
                entries[str(child)] = kind
                if kind == DIR:
                    pending.append(child)
    return entries


def _unexpected_kind(path: str, side: str, kind: str) -> dict[str, Any]:
    label = SIDE_LABELS[side]
    if kind == MISSING:
        return {"path": path, "reason": f"missing on the {label}"}
    if kind == SYMLINK:
        return {"path": path, "reason": f"{label} is a symbolic link (not followed)"}
    return {"path": path, "reason": f"{label} is neither a regular file nor a directory"}


def _same_inode(written: Path, existing: Path) -> bool:
    left, right = written.lstat(), existing.lstat()
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def compare_file(
    written: Path, existing: Path, path: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare two regular files (`*.safetensors` by tensors and metadata, anything else by
    sha256)."""
    if _same_inode(written, existing):
        return {"path": path, "compare": "none", "match": False}, [
            {
                "path": path,
                "reason": "both sides are the same file (same inode — a self-comparison)",
            }
        ]
    if path.endswith(SAFETENSORS_SUFFIX):
        return compare_safetensors(written, existing, path)
    return compare_bytes(written, existing, path)


def _compare_tree(
    written: Path, existing: Path, name: PurePosixPath
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _same_inode(written, existing):
        return [], [
            {
                "path": str(name),
                "reason": "both sides are the same file (same inode — a self-comparison)",
            }
        ]
    left, right = _listing(written), _listing(existing)
    files: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for relative in sorted(left.keys() | right.keys()):
        path = str(name / relative)
        kinds = {"written": left.get(relative, MISSING), "existing": right.get(relative, MISSING)}
        unexpected = [
            _unexpected_kind(path, side, kind)
            for side, kind in kinds.items()
            if kind not in (FILE, DIR)
        ]
        if unexpected:
            mismatches += unexpected
        elif kinds["written"] != kinds["existing"]:
            mismatches.append(
                {
                    "path": path,
                    "reason": "kind differs",
                    "written": kinds["written"],
                    "existing": kinds["existing"],
                }
            )
        elif kinds["written"] == FILE:
            result, found = compare_file(written / relative, existing / relative, path)
            files.append(result)
            mismatches += found
    return files, mismatches


def compare(written: Path, existing: Path, required: Sequence[PurePosixPath]) -> dict[str, Any]:
    """Summary of comparing only the items in `required` (`match` True means zero mismatches)."""
    items: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for name in required:
        found: list[dict[str, Any]] = []
        kinds = {}
        for side, root in (("written", written), ("existing", existing)):
            kind, at = _lookup(root, name)
            kinds[side] = kind
            if kind not in (FILE, DIR):
                found.append(_unexpected_kind(str(name if kind == MISSING else at), side, kind))
        if not found and kinds["written"] != kinds["existing"]:
            found.append(
                {
                    "path": str(name),
                    "reason": "kind differs",
                    "written": kinds["written"],
                    "existing": kinds["existing"],
                }
            )
        compared: list[dict[str, Any]] = []
        if not found and kinds["written"] == FILE:
            result, found = compare_file(written / name, existing / name, str(name))
            compared = [result]
        elif not found:
            compared, found = _compare_tree(written / name, existing / name, name)
            if not compared and not found:
                found = [{"path": str(name), "reason": "required directory has no file to compare"}]
        files += compared
        items.append({"name": str(name), **kinds, "files": len(compared), "match": not found})
        mismatches += found
    return {
        "written": str(written),
        "existing": str(existing),
        "match": not mismatches,
        "items": items,
        "files": files,
        "mismatches": mismatches,
    }


def require_name(text: str) -> PurePosixPath:
    """A `--require` value (a relative path inside the root — rejects absolute paths, `..`, the root
    itself and NUL)."""
    name = PurePosixPath(text)
    if name.is_absolute() or ".." in name.parts or not name.parts or "\0" in text:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a relative path inside the root"
            " (absolute paths, '..', the root itself and NUL are not compared)"
        )
    return name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--written", type=Path, required=True, help="root of the rewritten side")
    parser.add_argument("--existing", type=Path, required=True, help="root of the existing series")
    parser.add_argument(
        "--require",
        type=require_name,
        action="append",
        required=True,
        help="item to compare (path relative to the root, a file or a directory — repeatable)",
    )
    args = parser.parse_args(argv)
    for flag, root in (("--written", args.written), ("--existing", args.existing)):
        if not root.is_dir():
            parser.error(f"{flag} {root} is not a directory")
    counts = Counter(args.require)
    duplicated = sorted(str(name) for name, count in counts.items() if count > 1)
    if duplicated:
        parser.error(f"--require is duplicated: {duplicated}")
    # A nested item only compares the same files twice and inflates the counts, so it is rejected
    # as an argument error.
    nested = sorted(
        f"{inner} is inside {outer}"
        for inner in counts
        for outer in counts
        if outer in inner.parents
    )
    if nested:
        parser.error(f"--require is nested: {nested}")
    try:
        summary = compare(args.written, args.existing, args.require)
    except OSError as error:
        print(f"series_check: cannot read a file (comparison incomplete): {error}", file=sys.stderr)
        return 3
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0 if summary["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
