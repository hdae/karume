"""Minimal safetensors writer **for tests only** — synthesizes the old distribution to migrate.

The distribution moved to `krm`, so the product has no code at all that writes the safetensors
dialect (container-v1 §12: only `karume.legacy` **reads** the old format, and only this file
**writes** it). To build subjects for `karume migrate`, it copies just the conventions the old
writer kept:

- The first shard holds only `karume_ir`, and its data section is empty (ADR 0081).
- The real data sits in the following weight shards, whose `__metadata__` is empty.
- The data section is ordered by **descending alignment unit** (F32 / I32 / I4 / I2 → F16 → I8),
  by name ascending within a group. F16 entries with an odd element count go further back
  (ADR 0063 — the Karume reader requires every absolute offset to sit on the dtype's alignment
  unit).
- i4 / i2 keep **the logical shape**, and only the byte length follows from the bit width
  (ADR 0069 decision 2).

MUST: the inputs (graph, stored tensors, scales) come from **the same single source** as
{@link ir_fixtures.fixture_spec}. With split inputs, the byte identity of "the container migrated
from the old form" and "the container written directly" would break on the difference of inputs,
not on a migration problem.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from ir_fixtures import Shape, fill_spec, fixture_spec

from karume.container import codec_entry, numbered_path
from karume.emit import StoredModel, stored_model
from karume.ir import IrGraph
from karume.legacy import IR_METADATA_KEY
from karume.ple import ple_row_bytes

#: Storage dtype → (safetensors dtype name, bits per element). A copy of the old writer's table.
_STORED_DTYPE: Mapping[str, tuple[str, int]] = {
    "f32": ("F32", 32),
    "f16": ("F16", 16),
    "bf16": ("BF16", 16),
    "i32": ("I32", 32),
    "i8": ("I8", 8),
    "i4": ("I4", 4),
    "i2": ("I2", 2),
}

#: First sort key (descending alignment unit).
_DTYPE_GROUP = {"F32": 0, "I32": 1, "I4": 2, "I2": 2, "BF16": 3, "F16": 3, "I8": 4}

#: The header length is padded with spaces to a multiple of this (puts the start of the data
#: section on a 4-byte boundary).
_HEADER_ALIGN = 8


@dataclass(frozen=True)
class Entry:
    """Declaration of one tensor in the data section (**logical** shape, stored byte length)."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def write_safetensors(entries: Sequence[Entry], metadata: Mapping[str, str]) -> bytes:
    """Assemble `[u64 LE header length][header JSON][data section]` **in declaration order**."""
    header: dict[str, object] = {"__metadata__": dict(metadata)}
    offset = 0
    for entry in entries:
        header[entry.name] = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.nbytes],
        }
        offset += entry.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % _HEADER_ALIGN)
    return len(blob).to_bytes(8, "little") + blob + b"".join(entry.payload for entry in entries)


def order(entries: Sequence[Entry]) -> list[Entry]:
    """Data-section order (descending alignment unit → odd F16 last → name ascending)."""
    return sorted(
        entries,
        key=lambda entry: (
            _DTYPE_GROUP[entry.dtype],
            1 if entry.dtype == "F16" and entry.nbytes % 4 else 0,
            entry.name,
        ),
    )


def _scale_shape(shape: Sequence[int], encoding) -> tuple[int, ...]:
    """Scale shape the old distribution had (group form / per-channel keepdim form)."""
    if codec_entry(encoding.codec).grouping == "group":
        return (shape[0], math.prod(shape[1:]) // encoding.group_size)
    return tuple(dim if axis == encoding.row_axis else 1 for axis, dim in enumerate(shape))


def legacy_entries(stored: StoredModel) -> list[Entry]:
    """Inputs after the storage conversion → old-distribution data-section declarations (sorted).

    It takes **the very return value** of {@link karume.emit.stored_model} — building two
    `StoredModel`s from the same inputs would run the storage conversion twice, and "which graph
    was written" would differ per caller.
    """
    committed = stored.graph
    entries: list[Entry] = []
    for name, initializer in committed.initializers.items():
        if initializer.is_shared:
            continue
        key = initializer.tensor
        shape = tuple(int(dim) for dim in committed.values[name].shape)
        label, _ = _STORED_DTYPE[initializer.storage.dtype]
        entries.append(Entry(key, label, shape, bytes(stored.tensors[key])))
        scale_key = initializer.storage.scale
        if scale_key is not None:
            payload = bytes(stored.tensors[scale_key])
            entries.append(
                Entry(scale_key, "F32", _scale_shape(shape, stored.bindings[key]), payload)
            )
    return order(entries)


def legacy_shards(
    *,
    mark: str = "fixture",
    storage: str = "f32",
    inputs: Sequence[tuple[str, Shape]] = (),
    outputs: Sequence[Shape] = ([1],),
    weights: Sequence[str] = ("weight",),
    baked: tuple[str, int] | None = None,
    groups: int = 1,
) -> list[bytes]:
    """Shard bytes of one old-distribution component (in reading order — the graph shard first).

    `groups` is the number of weight shards (not a count that splits the data section evenly:
    **the declarations are dealt to that many shards in order**). The option exists to take the
    paths where the migration resolves "a list across directories" and "several shards".
    """
    graph, tensors, scales, overrides = fixture_spec(mark, storage, inputs, outputs, weights, baked)
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )
    return _shards(stored.graph, legacy_entries(stored), groups)


def piece_entries(entry: Entry, rows: Sequence[tuple[int, int]]) -> list[Entry]:
    """Split one parent into pieces over the row ranges `rows` (old shard spec v3 — ADR 0090).

    A piece key is `<parent name>#NNNNN-of-NNNNN` (the index starts at 1, the total is
    `len(rows)`), its shape is `(rows, *remaining dims)`, and its payload is the matching span of
    the parent's row-major bytes.
    """
    row_bytes = entry.nbytes // entry.shape[0]
    count = len(rows)
    return [
        Entry(
            f"{entry.name}#{index:05d}-of-{count:05d}",
            entry.dtype,
            (stop - start, *entry.shape[1:]),
            entry.payload[start * row_bytes : stop * row_bytes],
        )
        for index, (start, stop) in enumerate(rows, start=1)
    ]


def legacy_piece_shards(
    *, parent: str, pieces: Sequence[tuple[int, int]], mark: str = "fixture", storage: str = "f32"
) -> list[bytes]:
    """Shards from the {@link legacy_shards} inputs with the tensor `parent` split into pieces.

    It copies the old writer's layout (old reader contract 5): piece 1 goes to the first weight
    shard with the other tensors, and pieces 2 onwards go **one each into consecutive following
    shards**.
    """
    graph, tensors, scales, overrides = fixture_spec(mark, storage)
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )
    entries = legacy_entries(stored)
    target = next(entry for entry in entries if entry.name == parent)
    split = piece_entries(target, pieces)
    rest = [entry for entry in entries if entry.name != parent]
    return [
        write_safetensors([], {IR_METADATA_KEY: stored.graph.to_json()}),
        write_safetensors(order([*rest, split[0]]), {}),
        *(write_safetensors([piece], {}) for piece in split[1:]),
    ]


def legacy_fill_shards(count: int, *, mark: str) -> list[bytes]:
    """Old distribution of `count` shards (the graph shard first, then one tensor each)."""
    if count < 2:
        raise ValueError(
            f"shard count {count} must be at least 2 (the first is a graph shard with no tensors)"
        )
    graph, tensors = fill_spec(count - 1, mark=mark)
    stored = stored_model(graph, tensors)
    return _shards(stored.graph, legacy_entries(stored), count - 1)


def _shards(committed: IrGraph, entries: Sequence[Entry], groups: int) -> list[bytes]:
    """Graph shard + weight shards (the declarations are dealt to `groups` shards in order)."""
    shards = [write_safetensors([], {IR_METADATA_KEY: committed.to_json()})]
    if not entries:
        return shards
    per = -(-len(entries) // groups)
    for index in range(0, len(entries), per):
        shards.append(write_safetensors(entries[index : index + per], {}))
    return shards


def stage_shards(directory: Path, name: str, shards: Sequence[bytes]) -> Path:
    """Place the old distribution as numbered files; return the component's representative path."""
    directory.mkdir(parents=True, exist_ok=True)
    representative = directory / name
    for index, blob in enumerate(shards, start=1):
        numbered_path(representative, index, len(shards)).write_bytes(blob)
    return representative


#: The one old `extras` kind that exists (the key of `karume.migrate.EXTRA_ASSETS`) — the source of
#: the container asset `rope_base`.
ROPE_BASE_NAME = "rope_base"

#: Index file name of the old PLE sidecar (`karume.ple.PLE_INDEX_ASSET` + `.json`).
PLE_INDEX_FILE = "ple.json"

#: Shard name of the old PLE sidecar (only one is placed — the migration discards the meaning of
#: the boundaries).
PLE_SHARD_FILE = "ple-00001.safetensors"

#: Key the old PLE shard holds in `__metadata__` (a copy of `karume.migrate.PLE_METADATA_KEY`).
_PLE_METADATA_KEY = "karume_ple"


def legacy_rope_base(directory: Path, *, values: Sequence[float] = (0.5, 1.5, 2.5, 3.5)) -> bytes:
    """Place the old `extras` file `rope_base.safetensors` and return its **bytes**.

    A container asset is the field for "bytes that are not weights but ship in the same
    container", so the migration folds this file in as a payload **without changing a single
    byte** (`karume.migrate._extra_assets`). The direct writer puts the same bytes into an
    `AssetInput`, so the part lists of both paths MUST be byte-identical.
    """
    payload = write_safetensors(
        [
            Entry(
                "rope.base",
                "F32",
                (len(values),),
                torch.tensor(values, dtype=torch.float32).numpy().tobytes(),
            )
        ],
        {},
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{ROPE_BASE_NAME}.safetensors").write_bytes(payload)
    return payload


@dataclass(frozen=True)
class LegacyPle:
    """The old PLE sidecar placed (schema 2) plus what the direct writer needs for its assets."""

    tokens: int
    layers: int
    dim: int
    storage: str
    embed_scale: float
    #: Row-major payloads concatenated in token order (`values` / `scales`).
    payloads: Mapping[str, bytes]


def legacy_ple_sidecar(
    directory: Path,
    *,
    tokens: int = 8,
    layers: int = 2,
    dim: int = 8,
    storage: str = "i8",
    embed_scale: float = 0.125,
) -> LegacyPle:
    """Place the old PLE sidecar (`ple.json` schema 2 + one shard) and return its inputs.

    The migration CLI (`karume.migrate._ple_assets`) reads these two files and folds them into
    index schema 3 and the `ple.values.<k>` / `ple.scales.<k>` blocks. The direct writer passes
    the same payloads to `karume.ple.ple_assets`, so the part lists of both paths MUST be
    byte-identical.
    """
    row_bytes = ple_row_bytes(storage, layers, dim)
    payloads = {
        key: bytes((index * 7 + 3) % 251 for index in range(tokens * stride))
        for key, stride in row_bytes.items()
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PLE_INDEX_FILE).write_text(
        json.dumps(
            {
                "schema": 2,
                "storage": storage,
                "tokens": tokens,
                "layers": layers,
                "dim": dim,
                "embedScale": embed_scale,
                "shards": [{"file": PLE_SHARD_FILE, "start": 0, "stop": tokens}],
            }
        )
    )
    (directory / PLE_SHARD_FILE).write_bytes(
        write_safetensors(
            [
                # The old writer and old reader (gemma's ple.ts) use `[rows, layers, dim]`
                # (the logical shape).
                Entry(
                    "values",
                    _PLE_SHARD_DTYPE[storage],
                    (tokens, layers, dim),
                    payloads["values"],
                ),
                Entry("scales", "F32", (tokens, layers), payloads["scales"]),
            ],
            {
                _PLE_METADATA_KEY: json.dumps(
                    {
                        "schema": 2,
                        "storage": storage,
                        "tokens": tokens,
                        "layers": layers,
                        "dim": dim,
                        "start": 0,
                        "stop": tokens,
                    }
                )
            },
        )
    )
    return LegacyPle(tokens, layers, dim, storage, embed_scale, payloads)


#: PLE storage → the shard's safetensors dtype name (only `values` can be packed).
_PLE_SHARD_DTYPE: Mapping[str, str] = {"i8": "I8", "i4": "I4", "i2": "I2"}
