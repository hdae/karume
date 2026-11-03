"""Pure functions that build PLE (per-layer embedding) into **container assets** (ADR 0085 / 0109
decision 4).

PLE is a table with one row per vocabulary entry (`values` and `scales`), not a weight — the host
range-reads it per token and sends it to the GPU (ADR 0085). What the sidecar (`ple.json` +
`ple-NNNNN.safetensors`) used to hold moved into the container's `assets`:

- `ple.values.<k>` / `ple.scales.<k>` (roles `ple-values` / `ple-scales`) — a block list cut in
  **multiples of a row**, each at most the block limit. They need range reads, so **1 block =
  1 part** ("placed alone in a dedicated part" of container-v1 §4.2).
- `ple_index` (role `ple-index`, schema 3) — an index that holds only the row byte size and the
  token range of each block. Readers translate token → (block, row offset) from it.

The scales come in two forms. Schema 3 holds one f32 per (token, layer) row. Schema 4
({@link PLE_GROUPED_INDEX_SCHEMA}) is schema 3 plus a required `groupSize`: one f32 per group of
`groupSize` consecutive values of a (token, layer) row, so a `scales` row is
`layers * dim / groupSize` f32 in (layer, group) order. It is the form of a table whose groups do
not share a step — Gemma 4's q4_0 QAT checkpoints store the table on a 4-bit grid with one step
per 32 values, and schema 3 cannot hold those values exactly. `group_size=None` writes schema 3.

MUST: assets are built through **this one path** (the migration CLI and the recipes alike). If two
paths spelled it, "a container migrated from the old sidecar" and "a container written directly by
a recipe" would silently disagree on the block cuts, and distributions with different bytes would
come out of the same weights.

MUST: the returned order is **the physical layout order as is** (token order — so neighbouring
tokens do not scatter across parts for a scan-type source). Canonical serialisation spells the
descriptor's `assets` in code point order, so the order matters only for the layout in the file.
"""

from __future__ import annotations

from collections.abc import Buffer, Callable, Mapping
from typing import Any

from karume.container import BLOCK_MAX_BYTES, BLOCK_TAIL_ALIGN, AssetInput, canonical_json

#: Name of the PLE index that lived in the old manifest's assets. It stays the asset name in the
#: new container.
PLE_INDEX_ASSET = "ple_index"

#: Version of the PLE index (the form that points at block lists — replaces schema 1 / 2 of the
#: old sidecar).
PLE_INDEX_SCHEMA = 3

#: Version of the PLE index with one scale per group of `groupSize` values (module docstring).
PLE_GROUPED_INDEX_SCHEMA = 4

#: Role of the PLE asset (the interpreter name on the models side — the runtime does not
#: interpret it).
PLE_INDEX_ROLE = "ple-index"

#: Row kind → role. The order is the layout order as is, so `values` comes first (the primary
#: target of token scans).
PLE_ROLES: Mapping[str, str] = {"values": "ple-values", "scales": "ple-scales"}

#: PLE storage → elements packed into 1 byte (mirror of `packages/models/src/gemma/ple-index.ts`).
PLE_PACK_FACTOR: Mapping[str, int] = {"i8": 1, "i2": 4, "i4": 2}

#: Bytes per PLE scale (f32 — same as above).
PLE_SCALE_BYTES = 4


class PleError(ValueError):
    """PLE dimensions cannot become assets (not divisible by the storage's pack factor, or not a
    single row fits in a block)."""


def ple_row_bytes(
    storage: str, layers: int, dim: int, group_size: int | None = None
) -> dict[str, int]:
    """Row kind → bytes per row (`values` is packed; `scales` is one f32 per layer, or one per
    group of `group_size` values with a group size — schema 4).

    MUST: both are multiples of 4 (blocks are cut in **multiples of a row**, so unless the row is a
    multiple of 4 the block tail cannot be 4-byte aligned without padding — container-v1 §4.1).
    MUST: a group size divides `dim` — a group never crosses a (token, layer) row.
    """
    factor = PLE_PACK_FACTOR.get(storage)
    if factor is None:
        raise PleError(f"unknown PLE storage '{storage}' ({' / '.join(sorted(PLE_PACK_FACTOR))})")
    if dim % factor != 0:
        raise PleError(
            f"dim {dim} is not divisible by the storage '{storage}' pack factor {factor}"
        )
    groups = 1
    if group_size is not None:
        if group_size <= 0 or dim % group_size != 0:
            raise PleError(f"group size {group_size} does not divide dim {dim}")
        groups = dim // group_size
    row_bytes = {"values": layers * dim // factor, "scales": layers * groups * PLE_SCALE_BYTES}
    for key, stride in row_bytes.items():
        if stride % BLOCK_TAIL_ALIGN != 0:
            raise PleError(
                f"one {key} row of {stride} bytes is not a multiple of {BLOCK_TAIL_ALIGN}"
                " (cannot cut blocks in multiples of a row)"
            )
    return row_bytes


def ple_block_ranges(
    *,
    storage: str,
    tokens: int,
    layers: int,
    dim: int,
    group_size: int | None = None,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Row kind → token range `[start, stop)` of each block (**the same cut** as
    {@link ple_assets}).

    The cuts depend only on the dimensions (not one byte of payload is needed), so a caller that
    wants the ranges before building the data calls this — a recipe picks "probe tokens that
    straddle a block boundary" before the data exists.
    """
    ranges: dict[str, tuple[tuple[int, int], ...]] = {}
    for key, stride in ple_row_bytes(storage, layers, dim, group_size).items():
        per_block = block_bytes // stride
        if per_block < 1:
            raise PleError(f"one {key} row of {stride} bytes exceeds the block limit {block_bytes}")
        ranges[key] = tuple(
            (start, min(start + per_block, tokens)) for start in range(0, tokens, per_block)
        )
    return ranges


def ple_assets(
    *,
    storage: str,
    tokens: int,
    layers: int,
    dim: int,
    embed_scale: float,
    read_values: Callable[[int, int], Buffer],
    read_scales: Callable[[int, int], Buffer],
    group_size: int | None = None,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> dict[str, AssetInput]:
    """PLE into container assets (index schema 3 — schema 4 with `group_size` — + the `values` /
    `scales` block lists).

    `read_values` / `read_scales` are calls that return **the byte range `[begin, end)` of the
    concatenated row-major payload** (each call MUST return the same bytes — the writer calls
    twice). The migration CLI passes range reads over the old shard list; a recipe passes slices
    of an in-memory tensor. The old shard boundaries carry no meaning (everything is concatenated
    in token order and cut again).

    The returned order is the physical layout order as is (`values` block list → `scales` block
    list → index).
    """
    row_bytes = ple_row_bytes(storage, layers, dim, group_size)
    cut = ple_block_ranges(
        storage=storage,
        tokens=tokens,
        layers=layers,
        dim=dim,
        group_size=group_size,
        block_bytes=block_bytes,
    )
    readers = {"values": read_values, "scales": read_scales}
    assets: dict[str, AssetInput] = {}
    document: dict[str, Any] = {
        "schema": PLE_INDEX_SCHEMA if group_size is None else PLE_GROUPED_INDEX_SCHEMA,
        "storage": storage,
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": embed_scale,
    }
    if group_size is not None:
        document["groupSize"] = group_size
    for key, role in PLE_ROLES.items():
        stride = row_bytes[key]
        read = readers[key]
        blocks: list[dict[str, Any]] = []
        for start, stop in cut[key]:
            name = f"ple.{key}.{len(blocks)}"
            blocks.append({"asset": name, "start": start, "stop": stop})
            # The block needs range reads, so it is placed alone in a dedicated part
            # (container-v1 §4.2).
            assets[name] = AssetInput(
                role,
                (stop - start) * stride,
                _slice(read, start * stride, stop * stride),
                dedicated_part=True,
            )
        document[key] = {"rowBytes": stride, "blocks": blocks}
    encoded = canonical_json(document).encode("utf-8")
    assets[PLE_INDEX_ASSET] = AssetInput(PLE_INDEX_ROLE, len(encoded), encoded)
    return assets


def _slice(read: Callable[[int, int], Buffer], begin: int, end: int) -> Callable[[], Buffer]:
    """Lazy read of one block (builds nothing until called)."""

    def pull() -> Buffer:
        return read(begin, end)

    return pull
