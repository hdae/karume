"""The PLE writer (`karume.ple`): the index forms and the block lists it builds.

The index documents of schema 3 (one f32 scale per row) and schema 4 (one per `groupSize` values)
are pinned byte for byte on a small synthetic table (6 tokens × 2 layers × 4 values) — the forms
the distributions carry, not their bytes: nothing here reads a published index. Schema 5 is the
float table: `values` only, the checkpoint's bf16 bits, no `scales` (ADR 0085 addenda 2026-11-04).
"""

from __future__ import annotations

import json

import pytest

from karume.ple import (
    PLE_FLOAT_INDEX_SCHEMA,
    PLE_INDEX_ASSET,
    PleError,
    ple_assets,
    ple_block_ranges,
    ple_row_bytes,
)

#: 6 tokens × 2 layers × 4 values, blocks of 16 B (two rows of i8 values per block).
TOKENS, LAYERS, DIM, BLOCK = 6, 2, 4, 16

SCHEMA_3_INDEX = (
    '{"schema":3,"storage":"i8","tokens":6,"layers":2,"dim":4,"embedScale":2,'
    '"values":{"rowBytes":8,"blocks":[{"asset":"ple.values.0","start":0,"stop":2},'
    '{"asset":"ple.values.1","start":2,"stop":4},{"asset":"ple.values.2","start":4,"stop":6}]},'
    '"scales":{"rowBytes":8,"blocks":[{"asset":"ple.scales.0","start":0,"stop":2},'
    '{"asset":"ple.scales.1","start":2,"stop":4},{"asset":"ple.scales.2","start":4,"stop":6}]}}'
)

SCHEMA_4_INDEX = (
    '{"schema":4,"storage":"i8","tokens":6,"layers":2,"dim":4,"embedScale":2,"groupSize":2,'
    '"values":{"rowBytes":8,"blocks":[{"asset":"ple.values.0","start":0,"stop":2},'
    '{"asset":"ple.values.1","start":2,"stop":4},{"asset":"ple.values.2","start":4,"stop":6}]},'
    '"scales":{"rowBytes":16,"blocks":[{"asset":"ple.scales.0","start":0,"stop":1},'
    '{"asset":"ple.scales.1","start":1,"stop":2},{"asset":"ple.scales.2","start":2,"stop":3},'
    '{"asset":"ple.scales.3","start":3,"stop":4},{"asset":"ple.scales.4","start":4,"stop":5},'
    '{"asset":"ple.scales.5","start":5,"stop":6}]}}'
)

#: The bf16 table: one row is 2 layers × 4 values × 2 B = 16 B, so one row per block.
SCHEMA_5_INDEX = (
    '{"schema":5,"storage":"bf16","tokens":6,"layers":2,"dim":4,"embedScale":2,'
    '"values":{"rowBytes":16,"blocks":[{"asset":"ple.values.0","start":0,"stop":1},'
    '{"asset":"ple.values.1","start":1,"stop":2},{"asset":"ple.values.2","start":2,"stop":3},'
    '{"asset":"ple.values.3","start":3,"stop":4},{"asset":"ple.values.4","start":4,"stop":5},'
    '{"asset":"ple.values.5","start":5,"stop":6}]}}'
)


def _payload(row_bytes: int) -> bytes:
    """A distinct byte per position of the concatenated payload (a mis-cut block is visible)."""
    return bytes(position % 251 for position in range(TOKENS * row_bytes))


def _reader(payload: bytes):
    return lambda begin, end: payload[begin:end]


def _index(assets) -> str:
    return bytes(assets[PLE_INDEX_ASSET].payload).decode()


class TestQuantizedIndexesKeepTheirBytes:
    def test_schema_3(self) -> None:
        assets = ple_assets(
            storage="i8",
            tokens=TOKENS,
            layers=LAYERS,
            dim=DIM,
            embed_scale=2.0,
            read_values=_reader(_payload(8)),
            read_scales=_reader(_payload(8)),
            block_bytes=BLOCK,
        )
        assert _index(assets) == SCHEMA_3_INDEX

    def test_schema_4(self) -> None:
        assets = ple_assets(
            storage="i8",
            tokens=TOKENS,
            layers=LAYERS,
            dim=DIM,
            embed_scale=2.0,
            read_values=_reader(_payload(8)),
            read_scales=_reader(_payload(16)),
            group_size=2,
            block_bytes=BLOCK,
        )
        assert _index(assets) == SCHEMA_4_INDEX

    def test_a_quantized_table_needs_its_scale_reader(self) -> None:
        with pytest.raises(PleError, match="storage 'i8' needs a scale reader"):
            ple_assets(
                storage="i8",
                tokens=TOKENS,
                layers=LAYERS,
                dim=DIM,
                embed_scale=2.0,
                read_values=_reader(_payload(8)),
                block_bytes=BLOCK,
            )


class TestFloatTable:
    def test_a_bf16_row_is_two_bytes_per_value_and_has_no_scales(self) -> None:
        assert ple_row_bytes("bf16", 35, 256) == {"values": 35 * 256 * 2}
        assert set(ple_block_ranges(storage="bf16", tokens=TOKENS, layers=LAYERS, dim=DIM)) == {
            "values"
        }

    def test_it_writes_schema_5_and_the_values_blocks_alone(self) -> None:
        payload = _payload(16)
        assets = ple_assets(
            storage="bf16",
            tokens=TOKENS,
            layers=LAYERS,
            dim=DIM,
            embed_scale=2.0,
            read_values=_reader(payload),
            block_bytes=BLOCK,
        )

        assert _index(assets) == SCHEMA_5_INDEX
        index = json.loads(_index(assets))
        assert index["schema"] == PLE_FLOAT_INDEX_SCHEMA
        assert "scales" not in index
        assert "groupSize" not in index
        blocks = {name: asset for name, asset in assets.items() if name != PLE_INDEX_ASSET}
        assert {asset.role for asset in blocks.values()} == {"ple-values"}
        # The blocks concatenate back to the payload: the bf16 bits are stored as they are.
        joined = b"".join(bytes(blocks[f"ple.values.{k}"].payload()) for k in range(len(blocks)))
        assert joined == payload

    def test_a_scale_reader_for_a_float_table_fails(self) -> None:
        with pytest.raises(PleError, match="storage 'bf16' has no scales to read"):
            ple_assets(
                storage="bf16",
                tokens=TOKENS,
                layers=LAYERS,
                dim=DIM,
                embed_scale=2.0,
                read_values=_reader(_payload(16)),
                read_scales=_reader(_payload(8)),
                block_bytes=BLOCK,
            )

    def test_a_group_size_for_a_float_table_fails(self) -> None:
        with pytest.raises(PleError, match="has no scales, so no group size"):
            ple_row_bytes("bf16", LAYERS, DIM, group_size=2)

    def test_an_unknown_storage_names_the_float_one_too(self) -> None:
        with pytest.raises(PleError, match="bf16 / i2 / i4 / i8"):
            ple_row_bytes("f16", LAYERS, DIM)

    def test_an_odd_row_cannot_be_cut_into_aligned_blocks(self) -> None:
        # 1 layer × 1 value × 2 B = 2 B per row, not a multiple of 4.
        with pytest.raises(PleError, match="not a multiple of 4"):
            ple_row_bytes("bf16", 1, 1)
