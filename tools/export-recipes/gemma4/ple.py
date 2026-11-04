"""Gemma 4's PLE (Per-Layer Embeddings) — the source of truth for splitting the single table into
L per-layer tables.

The count L is `num_hidden_layers` (E2B 35 / E4B 42), and every function derives it from the config
and the table's shape. The numbers below are written for E2B (L = 35) — for E4B the table has
10752 columns and is 11.3GB in f32.

`embed_tokens_per_layer` is a single `[262144, 35×256]` table, 9.4GB in f32 — **on its own it
exceeds WebGPU's single-buffer limit** (2.19GiB even when stored as i8). So it is split into 35
per-layer `nn.Embedding`s (each `[262144, 256]`, a column slice), reassembled into `[1,T,35,256]`
with `stack`, and passed upstream as `per_layer_inputs=`.

All five scripts ({@link gemma4.export} for the 1-shot form, {@link gemma4.export_decode} /
{@link gemma4.export_token} / {@link gemma4.export_product} for the chunk series, and
{@link gemma4.export_drafter} for the MTP drafter) **go through this one module** — if the split
rule grew in two places, a fix to only one of them would stay invisible to everyone until the
golden comparison.

## Reading by row block ({@link load_per_layer_tables})

MUST: never hold the whole single table in f32. The f32 `[262144, 8960]` is 9.4GB, and alive at
the same time as the split copies it doubles the peak RAM (the whole model is 18.5GB in f32, so
that alone makes it not fit). Read it by row block ({@link PLE_ROW_BLOCK}) and distribute each
block to the 35 tables as it is read.

## Distribution to the 35 tables ({@link per_layer_inputs})

Upstream `get_per_layer_inputs` reshapes a single `[B,T,L*D]` lookup, so layer `i` reads the
columns `[i*D, (i+1)*D)`. The split tables are those column slices, so `stack(dim=2)` gives the
same order. The scale ({@link per_layer_scale} = `hidden_size_per_layer_input ** 0.5` = 16.0) is a
power of two, so the f32 multiplication is exact — which is also the condition for the bit-exact
check below to hold.

## The bit-exact check with torch.equal ({@link assert_per_layer_split})

MUST: check once before the export that the split is **bit-identical** to upstream
`get_per_layer_inputs`. A wrong column assignment still matches in shape, type and dtype, so
`allclose` would pass "a close but different table", which again stays invisible to everyone until
the golden comparison. The reference side of the check is "the model's own
`embed_tokens_per_layer` loaded with only a few rows of the checkpoint" — the form that avoids
holding the 9.4GB f32 table in memory twice ({@link PLE_PROBE_ROWS} rows; the rows are the
scattered points of {@link probe_rows}).

NOTE: the spelling of the checkpoint's keys (the `model.language_model.` prefix) is not owned
here — renaming keys is knowledge of the script ({@link gemma4.export.renamed_state}), so the key
of the single table comes in as an argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

#: Number of rows used by the PLE split equivalence check (the rows are chosen by
#: {@link probe_rows} — scattered points at block boundaries, both ends and the middle). The model's
#: own `embed_tokens_per_layer` is allocated with only this many rows (by replacing
#: `config.vocab_size_per_layer_input` — upstream reads this field only from
#: `Gemma4TextModel.__init__` and `resize_token_embeddings`), so the 262144-row f32 table never
#: sits in memory twice. The 35 split tables are built directly from safetensors by
#: {@link load_per_layer_tables}.
PLE_PROBE_ROWS = 8

#: Row block for distributing the table to the 35 tables without reading it whole
#: (8192×8960×4B = 293MB in f32).
PLE_ROW_BLOCK = 8192


def per_layer_scale(config: Any) -> float:
    """The embed_scale applied to the PLE lookup (equal to upstream
    `Gemma4TextScaledWordEmbedding`).

    `hidden_size_per_layer_input` is 256, so the value is 16.0 — a power of two, so the f32
    multiplication is exact, which is also the condition for the bit-exact check of
    {@link assert_per_layer_split} to hold.
    """
    return float(config.hidden_size_per_layer_input) ** 0.5


def per_layer_inputs(
    tables: Sequence[nn.Module], input_ids: torch.Tensor, scale: float
) -> torch.Tensor:
    """Assemble the 35 per-layer lookups into `[B,T,L,D]` (equal to upstream
    `get_per_layer_inputs`).

    Upstream reshapes a single `[B,T,L*D]` lookup, so layer `i` reads the columns
    `[i*D, (i+1)*D)`. The split tables are those column slices, so `stack(dim=2)` gives the same
    order.
    """
    return torch.stack([table(input_ids) * scale for table in tables], dim=2)


def per_layer_table_shape(model_file: Path, key: str) -> tuple[int, int]:
    """`[rows, columns]` of the single PLE table (the data is not read)."""
    with safe_open(str(model_file), framework="pt") as handle:
        rows, packed = handle.get_slice(key).get_shape()
    return int(rows), int(packed)


def load_per_layer_tables(model_file: Path, key: str, layers: int, dim: int) -> nn.ModuleList:
    """Read the single PLE table split into 35 per-layer `nn.Embedding`s.

    MUST: never hold the whole single table in f32 (the reason is in the module docstring). Read it
    by row block ({@link PLE_ROW_BLOCK}) and distribute each block to the 35 tables as it is read.
    """
    rows, packed = per_layer_table_shape(model_file, key)
    if packed != layers * dim:
        raise ValueError(
            f"PLE table has {packed} columns, not {layers} layers × {dim} = {layers * dim}"
        )
    tables = [torch.empty(rows, dim, dtype=torch.float32) for _ in range(layers)]
    with safe_open(str(model_file), framework="pt") as handle:
        packed_table = handle.get_slice(key)
        for start in range(0, rows, PLE_ROW_BLOCK):
            stop = min(start + PLE_ROW_BLOCK, rows)
            block = packed_table[start:stop, :].to(torch.float32)
            for index, table in enumerate(tables):
                table[start:stop] = block[:, index * dim : (index + 1) * dim]
    return nn.ModuleList([nn.Embedding.from_pretrained(table, freeze=True) for table in tables])


def probe_rows(rows: int) -> tuple[int, ...]:
    """{@link PLE_PROBE_ROWS} **scattered** rows used by the PLE split equivalence check.

    Eight consecutive rows would touch only one row block ({@link PLE_ROW_BLOCK}), so a mix-up or
    corruption confined to another block would not show in the gate (Codex wave H finding H-05).
    Takes the first row, both sides of the first block boundary, a middle pair, the first row of
    the last block and the last pair, and fills any shortfall in order from the start so that
    there are **always exactly** {@link PLE_PROBE_ROWS} rows (a tiny table has no block boundaries
    but the same count — the probe model's row count = this count is baked into its config, so it
    is not made variable).
    """
    if rows < PLE_PROBE_ROWS:
        raise ValueError(f"PLE table has {rows} rows, fewer than the {PLE_PROBE_ROWS} probe rows")
    candidates = [
        0,
        PLE_ROW_BLOCK - 1,
        PLE_ROW_BLOCK,
        rows // 2,
        rows // 2 + 1,
        rows - PLE_ROW_BLOCK,
        rows - 2,
        rows - 1,
    ]
    picked: list[int] = []
    for row in candidates:
        if 0 <= row < rows and row not in picked:
            picked.append(row)
    filler = 1
    while len(picked) < PLE_PROBE_ROWS:
        if filler not in picked:
            picked.append(filler)
        filler += 1
    return tuple(picked[:PLE_PROBE_ROWS])


def assert_per_layer_split(model: nn.Module, tables: nn.ModuleList, probe: Sequence[int]) -> None:
    """Check that the 35-way split is **bit-identical** to upstream `get_per_layer_inputs`.

    The reference side is the model's own `embed_tokens_per_layer` (a model loaded with only the
    checkpoint's probe rows — the scattered points of {@link probe_rows} — in that order), i.e. the
    output itself after upstream's scale (`hidden_size_per_layer_input ** 0.5`) and reshape. The
    split side looks up the 35 tables by the **original row numbers** and reassembles them with
    {@link per_layer_inputs}.

    MUST: compare with `torch.equal` (bit-exact) — a wrong column assignment still matches in
    shape, type and dtype, so `allclose` would pass "a close but different table".
    """
    local = torch.arange(len(probe), dtype=torch.int64).unsqueeze(0)
    original = torch.tensor([list(probe)], dtype=torch.int64)
    with torch.no_grad():
        reference = model.model.get_per_layer_inputs(local, None)
        rebuilt = per_layer_inputs(tables, original, per_layer_scale(model.config))
    if tuple(reference.shape) != tuple(rebuilt.shape):
        raise AssertionError(
            f"PLE split shape {tuple(rebuilt.shape)} differs from upstream {tuple(reference.shape)}"
        )
    if not torch.equal(reference, rebuilt):
        worst = float((reference - rebuilt).abs().max())
        raise AssertionError(
            f"the {len(tables)}-way PLE split is not bit-identical to upstream get_per_layer_inputs"
            f" (max abs difference {worst})"
            " — the column assignment or the scale is wrong"
        )
