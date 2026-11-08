"""Export the real-weight Gemma 4 E2B / E4B (`--model`) as the **product graph** (the PLE out of
the graph, the selected rows' logits / hidden out) — the regular line's `model` container.

ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) decision 6 (the exit is the
selected rows' logits; sampling stays on the host) and ADR
[0085](../../../docs/decisions/0085-ple-host-gather.md) (the PLE gathered on the host) in one
export (plan α — ADR 0083 Consequences).

    uv run --with 'transformers==5.14.1' python -m gemma4.export_product              # E2B
    uv run --with 'transformers==5.14.1' python -m gemma4.export_product --model e4b  # E4B

## Two more inputs and two outputs than the chunk series

The chunk series' path (how the material is read, the host-supplied RoPE, the KV-sharing surgery,
the order of the gates) is {@link gemma4.export_decode}'s core, imported as it is (one discipline,
written once). Three differences:

- one more input, **`per_layer_inputs[1,M,L,256]` f32** (L = the layer count — E2B 35 / E4B 42).
  The PLE lookup is a pure row lookup over `input_ids` **alone**, so it leaves the graph and
  becomes an ordinary graph input the host supplies (ADR 0085 decision 6 — the runtime contract
  does not change by a character).
- one more input, **`last_row[R]` i32** (the row select of the token-only series —
  {@link gemma4.export_decode.TOKEN_ONLY_LAST_ROW}). The row count is the symbol
  {@link ROW_SYMBOL} because a speculative-decoding verify run scores several rows at once.
  **Ordinary prefill / decode bind R = 1**, and under that binding the values and the token
  sequence are bit-identical to the single-row exit.
- the exits are **`logits[1,R,V]` (output 0) and the post-final-norm hidden `[1,R,H]`
  (output 1)** — the selected **rows** only, no argmax: the token-only series'
  `TokenOnlyChunkWrapper` without its argmax, next to the hidden the drafter takes. Sampling and
  the RNG stay on the host (ADR 0083 decision 6's MUST); a prefill's read-back drops from the
  32 MiB of `[1,M,V]` to 1 MiB. MUST: the output order is **logits → hidden** (the runtime reads
  the slots by number).

## Storage: the checkpoint's bf16 values

1. **The linears and the embedding are bf16** (ADR 0131 — the tied head reads the embedding). The
   checkpoint is BF16, so nothing is rounded: the loader's F32 values are the checkpoint's bits,
   and the writer stores a weight as bf16 only when the round trip is exact (it fails loudly
   otherwise). The form check counts every linear and the embedding as bf16, and no i8 / i4.
2. **The PLE table is bf16 too** (index schema 5 — `karume.ple.PLE_FLOAT_INDEX_SCHEMA`, ADR 0085
   addendum 2026-11-04): the `values` blocks hold the checkpoint's bf16 bits and there is no
   `scales` table ({@link bf16_ple_tables} fails loudly on a value bf16 cannot hold).

## PLE (token-major, vocabulary-range blocks, container assets)

The 35 tables taken out of the graph are laid out **token-major** (`[token][layer][256]`) and cut
into blocks by **vocabulary range** (ADR 0085 decisions 1 / 2): one token's PLE is one contiguous
read, so reading single rows from a cache later (that ADR's alternative b) needs no re-export and
no re-upload. They live as **assets of the model container** (`ple_index` / `ple.values.<k>` —
ADR 0109 decision 4); the cut and the index spelling are core's {@link karume.ple.ple_assets}
alone (the migration CLI's one path — two spellings would move the cut silently).

MUST: the relayout is **bit-identical** to the 35-table path ({@link assert_ple_assets}). A table
shifted by one layer keeps its shape, type and dtype, so it takes a `torch.equal` gate, for the
same reason as `ple.py`'s split check. `ple.probe.safetensors` (the scattered tokens'
`per_layer_inputs` computed by **torch on the 35-table path** — what the embedding op produced
while the PLE was in the graph) is what the TS loader's output is held against.

## Goldens

The series takes its own (the q4_0 QAT series' rule — `gemma4.export_product_qat`): greedy
continuations of `gemma4.export_decode.GREEDY_CASES` by full re-forward of the stored model,
`gemma4.export_decode.GREEDY_STEPS` steps (`--steps`) and the margin floor
`gemma4.export_decode.MARGIN_FLOOR`, taken before the trace. Nothing is rounded, so they are the
upstream checkpoint's own continuations (computed in f32). The provenance record
({@link gemma4.provenance}) binds them to the container.

## Output layout

    outputs/series/gemma4-<model>-product-bf16/model.krm                  the container + PLE assets
    outputs/series/gemma4-<model>-product-bf16/ple.probe.safetensors      the PLE probe reference
    outputs/series/gemma4-<model>-product-bf16/greedy.<case>.safetensors  the greedy goldens
    outputs/series/gemma4-<model>-product-bf16/reference.json             provenance (own goldens)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Buffer, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.export import Dim
from torch.nn import functional

from _shared.container_read import open_container, read_asset
from _shared.decode_series import (
    _write_greedy,
    assert_case_room,
    positions_for,
)
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import ple, provenance
from gemma4.distribution import (
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_PRODUCT_BF16_SUFFIX,
    GEMMA4_PRODUCT_SUFFIX,
    GEMMA4_ROLE,
    gemma4_ple_dim,
)
from karume.artifacts import staged_publication
from karume.container import BLOCK_MAX_BYTES, AssetInput, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph
from karume.ops import ARGMAX_OP, EMBEDDING_OP, RESHAPE_OP
from karume.pipeline import export_module
from karume.ple import (
    PLE_FLOAT_INDEX_SCHEMA,
    PLE_GROUPED_INDEX_SCHEMA,
    PLE_INDEX_ASSET,
    PLE_INDEX_SCHEMA,
    PLE_PACK_FACTOR,
    ple_assets,
    ple_block_ranges,
    ple_row_bytes,
)
from karume.quantize import iter_quant_targets
from karume.shapes import declared_shape
from karume.states import to_states_form

#: The default model's (E2B) output directory (apart from the chunk series — other inputs, other
#: exits). The suffix is the distribution recipe's `GEMMA4_PRODUCT_BF16_SUFFIX` (the reader's word).
DEFAULT_OUT_DIR = one_shot.series_dir(GEMMA4_DEFAULT_MODEL, GEMMA4_PRODUCT_BF16_SUFFIX)

#: Graph input name of the host-supplied PLE (the wrapper's forward argument name itself —
#: torch.export takes it as the graph input name). The source of truth for the spelling is here,
#: referenced by {@link assert_ir_form_product} and the acceptance gate on the TS side.
PER_LAYER_INPUTS = "per_layer_inputs"

#: Graph input names of an image's rows on a model without per-layer embeddings (12B — the arguments
#: of `gemma4.export_unified.UnifiedProductWrapper`): the embedder's vectors and which rows take
#: them. The bool mask has width 1 — the IR's `where` broadcasts right-aligned.
SOFT_EMBEDS = "soft_embeds"
SOFT_MASK = "soft_mask"
WHERE_OP = "where"

#: Symbol name of the row selection (`last_row[R]` / the row axis of the two exits). The number of
#: rows a speculative-decoding verify run scores at once; ordinary prefill / decode bind R = 1 (the
#: same one-row exit as before).
ROW_SYMBOL = "R"

#: Trace upper bound of the symbol R. The rows a verify run scores at once = k draft rows + the one
#: bonus row confirmed just before, and the upper bound of k is the sliding ring's slack
#: ({@link decode.SLIDING_SLACK_ROWS} — the upper bound of rows that may be rejected), so
#: R ≤ slack + 1. Decided apart from the slack, the design "rejected rows do not overwrite the live
#: window" does not hold.
#:
#: NOTE: the IR's `symbols` is only a list of names with no bounds (`docs/ir-v2.md`), so this number
#: affects only **the range torch guards**. The runtime-side bound is held by the context's
#: `slidingSlack` (`queryLength ≤ slidingSlack + 1` of a deferred run).
ROW_SYM_MAX = decode.SLIDING_SLACK_ROWS + 1

#: Row count of the example input. Unless it is **2 or more**, torch.export may specialize R to 1,
#: and the IR cannot confirm it is alive as a symbol. The upper side ({@link ROW_SYM_MAX}) is held
#: by the Dim declaration.
EXAMPLE_ROWS = 2

#: File name of the dequantization bit-identity reference (**not part of the distribution** — an
#: acceptance golden that stays in the series).
PLE_PROBE_FILE = "ple.probe.safetensors"

#: The storage of the linears, the embedding and the PLE table (module docstring, "Storage").
BF16_STORAGE = "bf16"


#: Row kinds (also the index's field names — the spelling of `karume.ple.PLE_ROLES`).
PLE_VALUES_KEY = "values"
PLE_SCALES_KEY = "scales"

#: Tensor keys of `ple.probe.safetensors`.
PROBE_TOKENS_KEY = "tokens"
PROBE_INPUTS_KEY = "per_layer_inputs"


class ProductChunkWrapper(decode.DecodeChunkWrapper):
    """`(input_ids, 4 RoPE inputs, per_layer_inputs, last_row[R]) → (logits[1,R,V], hidden[1,R,H])`.

    MUST: a **subclass** of `DecodeChunkWrapper` (identity of the module FQN space — the
    quantization target predicates `is_int8_module` / `is_int4_module` and the condition for reusing
    the scale ledger's keys; the same reason as in the docstring of
    `export_decode.DecodeChunkWrapper`).
    MUST: `self.per_layer` (the 35 PLE tables) is held only as the source of the stored table and
    of the reference, never read by the forward — the PLE is a host-supplied graph input, so the
    module is dropped before the export ({@link export_series}). Exported with it, the graph would
    carry an unused 9.4GB state_dict.
    MUST: the row selection is `F.embedding` (the same text as the token-only form — passing a
    tensor to upstream's `logits_to_keep` lowers to `aten.index.Tensor` = advanced indexing outside
    the IR vocabulary). The three lm_head and softcap lines are verbatim isomorphic to
    {@link gemma4.export_token.TokenOnlyChunkWrapper}; **the only difference is that no argmax is
    placed** (ADR 0083 decision 6 — sampling stays on the host).
    MUST: the return order is fixed as **(logits, hidden)** — the runtime reads by output slot
    number, so swapping them would hand another tensor to sampling with shape and dtype still
    matching.
    MUST: the second output is the row-selected hidden itself **after the final norm and before
    lm_head** (`rowed`). Being the value before lm_head is the drafter side's input condition, and
    returning another intermediate value would still match the shape `[1,R,H]`.
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        last_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        tables = decode.bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        with self.model.model.rotary_emb.bound(tables):
            hidden = self.model.model(
                inputs_embeds=embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=mask,
                position_ids=None,
                use_cache=False,
            ).last_hidden_state
        # Lift to [1,R,H] after the row selection, then run lm_head (the same text as the
        # token-only form — the structural check of the row-selected lm_head looking at "the
        # selected rows" assumes this form).
        rowed = functional.embedding(last_row, hidden[0]).unsqueeze(0)
        logits = self.model.lm_head(rowed)
        cap = float(self.model.config.final_logit_softcapping)
        return torch.tanh(logits / cap) * cap, rowed


#: The variant passed to {@link decode.load_wrapper}. **Only the `wrapper` field is read** (how the
#: source tensors are read and the RoPE replacement do not depend on the wrapper type). The other 3
#: fields are branches for {@link decode.export_series}, which this script does not go through —
#: the product form's differences (two more inputs, logits exit, PLE assets) do not fit the 4 fields
#: of `ChunkVariant`, so this module drives the series.
_LOAD_VARIANT = decode.ChunkVariant(
    suffix=GEMMA4_PRODUCT_SUFFIX, wrapper=ProductChunkWrapper, token_only=False, goldens=False
)


def load_wrapper(model_dir: Path) -> ProductChunkWrapper:
    """Read the real weights in f32 and return the product wrapper with RoPE replaced by the
    hand-over inputs.

    How the source tensors are read (including the 3 equivalence checks) and the RoPE replacement
    are identical across the 3 chunk series, so this goes through {@link decode.load_wrapper}
    unchanged (the same discipline is not written in two places).
    """
    # The return value is exactly the type the variant's `wrapper` field builds
    # ({@link _LOAD_VARIANT}).
    return decode.load_wrapper(_LOAD_VARIANT, model_dir)


# ---- PLE assets ------------------------------------------------------------


def ple_token_bytes(layers: int, dim: int, storage: str) -> int:
    """One token's PLE bytes in `storage` (`karume.ple.ple_row_bytes` — the values, plus one f32
    scale per layer for an integer storage).

    The bf16 table has no scales: 35 × 256 × 2 = **17,920 bytes per token** for E2B. That is the
    point of token-major: one token's PLE is **one contiguous read** of this length (ADR 0085
    decision 1).
    """
    return sum(ple_row_bytes(storage, layers, dim).values())


def ple_table_rows(tables: Sequence[torch.nn.Module], vocab_size: int) -> int:
    """Row count of the 35-way PLE split (= the token rows of the PLE assets).

    MUST: do not take it from `config.vocab_size_per_layer_input` — that is the value after
    {@link gemma4.export.load_model_and_tables} replaced it with **the probe model's row count**
    (`ple.PLE_PROBE_ROWS` = 8), so reading it would write an 8-row PLE asset that still matches in
    shape and type (this actually happened once). The source of truth for the row count is only
    "the split tables actually loaded".

    MUST: check that it matches the main embedding's vocab row count (the **writer-side** half of
    ADR 0085 decision 5) — the host loader matches the id spaces assuming the two agree, so shipping
    them disagreeing makes the reader refuse, or, the moment the refusal is relaxed, read "a valid
    row of another token".
    """
    rows = {int(table.weight.shape[0]) for table in tables}
    if len(rows) != 1:
        raise AssertionError(f"the PLE split tables do not share one row count: {sorted(rows)}")
    found = rows.pop()
    if found != vocab_size:
        raise AssertionError(
            f"the PLE split tables have {found} rows, not the main embedding's vocab rows"
            f" {vocab_size} (the id space of the host gather would differ from the main embedding"
            " — ADR 0085 decision 5)"
        )
    return found


def ple_probe_tokens(tokens: int, ranges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """**Scattered** token ids for the dequantization bit-identity reference (both sides of each
    block boundary + both ends + the middle).

    N consecutive ids would touch only one block, so a mix-up confined to another block (an
    off-by-one in a range, a scale shifted by a layer) would not show in the gate (the block version
    of the same reason as {@link gemma4.ple.probe_rows}).
    """
    picked: list[int] = []
    for start, stop in ranges:
        picked.extend((start, start + 1, stop - 1))
    picked.extend((0, 1, tokens // 2, tokens - 2, tokens - 1))
    return tuple(sorted({token for token in picked if 0 <= token < tokens}))


def bf16_ple_tables(tables: Sequence[torch.nn.Module]) -> list[torch.Tensor]:
    """The PLE tables as their bf16 bits (`[V, D]` int16 views — the writer's form of bf16, ADR
    0131 decision 2), each table's f32 released as it is converted.

    MUST: every value round-trips `f32 → bf16 → f32` exactly — the checkpoint is BF16, so the
    loaded F32 values are bf16 values; one that is not means the loader changed a value, and storing
    it rounded would make the table no longer the original weights (fail loudly, as the writer does
    for the bf16 linears).
    MUST: release one table at a time — the 35 f32 tables are 9.4GB, and holding them next to the
    bf16 values raises the peak by gigabytes.
    """
    values: list[torch.Tensor] = []
    for index, table in enumerate(tables):
        weight = table.weight.data
        rounded = weight.to(torch.bfloat16)
        if not torch.equal(rounded.to(torch.float32), weight):
            worst = float((rounded.to(torch.float32) - weight).abs().max())
            raise AssertionError(
                f"PLE table {index} is not exact in bf16 (largest change {worst}) — the loaded"
                " values are not the checkpoint's bf16 values"
            )
        values.append(rounded.view(torch.int16))
        # Release the f32 (the caller drops the slot itself with `del wrapper.per_layer`).
        table.weight.data = torch.empty(0)
    return values


#: Approximate size of one write when spilling to the temporary file (bytes). It is how much of the
#: transpose's intermediate sits in RAM, so it is taken two orders of magnitude smaller than the
#: data of the 35 tables (about 2.35 GB for E2B).
_SPILL_CHUNK_BYTES = 64 * 1024 * 1024


def spill_payload(payload: Buffer, path: Path) -> None:
    """Spill the concatenated payload to `path` once (shared with the QAT side
    {@link gemma4_qat.ple}).

    MUST: the target is **a temporary file in the staging directory**, and the caller deletes it
    before the swap (it is not part of the distribution).
    """
    with path.open("wb") as handle:
        handle.write(payload)


def _spill_tables(
    tables: Sequence[torch.Tensor], stride: int, rows: int, path: Path
) -> Callable[[int, int], bytes]:
    """Spill the per-layer tables to `path` once as **a payload concatenated token-major**, and
    return a range reader over it.

    `tables` is held **per layer** (table-major), so each chunk of rows is stacked from the 35
    tables and written while being transposed to token-major (only one chunk of the transpose's
    intermediate is alive at a time).

    MUST: the target is **a temporary file in the staging directory**, and the caller releases the
    original data right after the spill (`values.clear()` / `row_scales.clear()`) — if the lazy
    reader kept the 35 tables in memory, the i8 data (about 2.35 GB for E2B) would stay resident
    throughout torch.export and the conversion (the script's RAM budget is the README's 24GB). It is
    deleted before the swap, so it never appears in the distribution.

    MUST: the same range returns **the same bytes** (the writer reads twice: once to take the sha256
    and once to write) — nobody rewrites the file after the spill.
    """
    per_chunk = max(1, _SPILL_CHUNK_BYTES // stride)
    with path.open("wb") as handle:
        for start in range(0, rows, per_chunk):
            stop = min(start + per_chunk, rows)
            chunk = torch.stack([table[start:stop] for table in tables], dim=1).contiguous()
            handle.write(memoryview(chunk.numpy()).cast("B"))
    written = path.stat().st_size
    if written != rows * stride:
        raise AssertionError(f"{path}: {written} bytes — not {rows} rows × {stride} bytes")

    def read(begin: int, end: int) -> bytes:
        if begin % stride or end % stride:
            raise AssertionError(
                f"PLE range [{begin}, {end}) is not a multiple of the {stride}-byte row"
            )
        with path.open("rb") as handle:
            handle.seek(begin)
            raw = handle.read(end - begin)
        if len(raw) != end - begin:
            raise AssertionError(f"{path}: PLE range [{begin}, {end}) ran out partway")
        return raw

    return read


@dataclass(frozen=True)
class PleContainerAssets:
    """Return value of {@link ple_container_assets} — the assets for the container, and the
    temporary files that may be deleted."""

    #: The assets passed as they are to `publish_model(assets=…)` (the index + the `values` /
    #: `scales` block sequences).
    assets: dict[str, AssetInput]
    #: Temporary files in the staging directory that hold the token-major payload (not part of the
    #: distribution).
    spills: tuple[Path, ...]

    def discard(self) -> None:
        """Delete the temporary files (MUST be called before the staging directory is placed as a
        whole — they must not mix into the distribution)."""
        for path in self.spills:
            path.unlink(missing_ok=True)


def ple_container_assets(
    values: Sequence[torch.Tensor],
    row_scales: Sequence[torch.Tensor] | None,
    *,
    tokens: int,
    layers: int,
    dim: int,
    embed_scale: float,
    spill_dir: Path,
    storage: str,
    group_size: int | None = None,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> PleContainerAssets:
    """Build the PLE as **assets of the model container** (the index + the `values` / `scales`
    block sequences).

    Both how it is cut and the index's spelling are owned by the core {@link karume.ple.ple_assets}
    (the same single path as the migration CLI). All this adds is the reader for "per-layer tables →
    the concatenated token-major payload".

    A bf16 `storage` (this series) takes `values` as each layer's bf16 bits (`[tokens, dim]`
    int16 — {@link bf16_ple_tables}) and `row_scales=None`, and builds the schema-5 index (no
    `scales`). An integer storage takes `values` as each layer's stored bytes
    (`[tokens, dim / pack factor]` — int8 for `i8`, packed uint8 for `i4` / `i2`) and `row_scales`
    as each layer's scales: `[tokens]` (schema 3, one scale per row) or, with a group size,
    `[tokens, dim / group_size]` (schema 4 — the q4_0 QAT line, {@link gemma4.export_product_qat}).

    MUST: the data is spilled **once** to a temporary file in `spill_dir` (the staging directory)
    (the MUST of {@link _spill_tables}) — the caller releases `values` / `row_scales` right after
    this call and does not carry the i8 data through the export.
    """
    row_bytes = ple_row_bytes(storage, layers, dim, group_size)
    # The row kinds the storage has (a bf16 table has no scales — `ple_assets` refuses a mismatch).
    tables = {PLE_VALUES_KEY: values, PLE_SCALES_KEY: row_scales}
    spills = {key: spill_dir / f".ple.{key}.spill" for key in row_bytes}
    readers = {}
    for key, path in spills.items():
        layered = tables[key]
        if layered is None:
            raise ValueError(f"storage '{storage}' has {key} rows, but none were given")
        readers[key] = _spill_tables(layered, row_bytes[key], tokens, path)
    return PleContainerAssets(
        ple_assets(
            storage=storage,
            tokens=tokens,
            layers=layers,
            dim=dim,
            embed_scale=embed_scale,
            read_values=readers[PLE_VALUES_KEY],
            read_scales=readers.get(PLE_SCALES_KEY),
            group_size=group_size,
            block_bytes=block_bytes,
        ),
        tuple(spills.values()),
    )


def _locate_block(blocks: Sequence[Mapping[str, Any]], token: int) -> Mapping[str, Any] | None:
    """The block containing the token (the index is an ascending partition without gaps, so a
    linear scan suffices — the probe is a few scattered points)."""
    for block in blocks:
        if int(block["start"]) <= token < int(block["stop"]):
            return block
    return None


def _unpack_ple_values(packed: torch.Tensor, storage: str) -> torch.Tensor:
    """Unpack the raw bytes (u8) of `values` into the quantized values q (i8) — the last axis grows
    by `× the pack factor`.

    The definition is the same as the codec ledger of container-v1 §6.3 (`int8-sym` / `int4-sym-g` /
    `int2-off` of the TS reader): i8 is `u = q`, i4 is `u = q + 8` (the **low** nibble of a byte
    first), i2 is `u = q + 2` (in order **from the low 2 bits**). It is also the same definition as
    the unpacking of upstream `QuantizedEmbedding`, and the gate holds only this one (getting the
    packing order wrong would compare "two things wrong in the same direction" with the reference
    side, so the reference side is kept as the upstream module's output).
    """
    if storage == "i8":
        return packed.view(torch.int8)
    bits = 8 // PLE_PACK_FACTOR[storage]
    mask = (1 << bits) - 1
    offset = 1 << (bits - 1)
    fields = [((packed >> shift) & mask).to(torch.int8) - offset for shift in range(0, 8, bits)]
    return torch.stack(fields, dim=-1).flatten(-2)


def assert_ple_assets(
    container: Path, index: Mapping[str, Any], probe: Sequence[int], reference: torch.Tensor
) -> None:
    """Rebuild the probe tokens from the assets of the placed container and check that they are
    **bit-identical** to the 35-table path.

    The reference side (`reference`) is the `[1,P,35,256]` that {@link gemma4.ple.per_layer_inputs}
    built from the 35 fake-quantized tables itself — that is, **the values the embedding op would
    have produced had the PLE stayed in the graph** (i8 storage + dequantization with a per-row
    scale restores the fake-quant values exactly — the ±127 argument of ADR 0019). The relayout side
    **reads the written bytes back** and assembles them in the same order as the host
    (`f32(q) * scale` → `* embed_scale`).

    Any storage (the index's `storage`) of i8 / i4 / i2 goes through the same gate — the reference
    of QAT's fixed packed PLE (ADR 0097 addendum 4) is the output of upstream `QuantizedEmbedding`,
    and packed values are unpacked by {@link _unpack_ple_values}. The bytes per row are read from
    the index's `rowBytes` as the source of truth, and a disagreement with the value derived from
    the storage and dimensions (`karume.ple.ple_row_bytes`) fails.

    A schema-4 index (`groupSize` — the q4_0 QAT line, {@link gemma4.export_product_qat}) holds
    one scale per group of `groupSize` values of a (token, layer) row, in (layer, group) order, and
    goes through the same check: each group's values times its own scale. MUST: `groupSize` is
    present exactly in a schema-4 index — a schema-3 reader would apply a group's scale to the
    whole row. A schema-5 index (a bf16 table — the bf16 series) has `values` only: each row's bf16
    bits widened to f32 times `embed_scale`. MUST: it spells neither `scales` nor `groupSize`.

    MUST: compare with `torch.equal` (bit-exact) — shifting the scale mapping by one layer, or a
    block's range by one row, yields **a valid row of another token** with shape, type and dtype
    still matching.
    MUST: read back (do not compare the in-memory arrays) — this gate is the last place that catches
    a form whose transpose is right but whose written dtype / shape / order is wrong.
    MUST: of each block read, **take only the probe rows and drop the rest** (do not keep whole
    blocks) — the probe touches both ends of every block, so keeping them would load the whole PLE
    (about 2.35 GB for E2B) into RAM once more while this gate runs. The container is also opened
    only once (`verify_container` is not run per block).
    MUST: do **not read the same block twice** — the probe touches both ends and the middle of a
    block, so re-reading per row would decode the same tens of MB of asset up to 3 times. Collect
    the needed rows first and read each block once (what stays resident is only the row buffers,
    unchanged).
    """
    layers = int(index["layers"])
    dim = int(index["dim"])
    embed_scale = float(index["embedScale"])
    storage = str(index["storage"])
    schema = index["schema"]
    floating = schema == PLE_FLOAT_INDEX_SCHEMA
    if floating and "groupSize" not in index and PLE_SCALES_KEY not in index:
        group_size = None
    elif schema == PLE_GROUPED_INDEX_SCHEMA:
        group_size = int(index["groupSize"])
    elif schema == PLE_INDEX_SCHEMA and "groupSize" not in index:
        group_size = None
    else:
        raise AssertionError(
            f"PLE index schema {schema!r} with groupSize {index.get('groupSize')!r}"
            f" (schema {PLE_INDEX_SCHEMA} has none, schema {PLE_GROUPED_INDEX_SCHEMA} needs one,"
            f" schema {PLE_FLOAT_INDEX_SCHEMA} has neither it nor scales)"
        )
    derived = ple_row_bytes(storage, layers, dim, group_size)
    if floating != (PLE_SCALES_KEY not in derived):
        raise AssertionError(f"PLE index schema {schema!r} with storage '{storage}'")
    row_bytes = {key: int(index[key]["rowBytes"]) for key in derived}
    if row_bytes != derived:
        raise AssertionError(
            f"the index's rowBytes {row_bytes} for storage '{storage}', layers {layers}, dim {dim}"
            f", groupSize {group_size} differ from the derived {derived}"
        )
    # One scale per row is one group of `dim` values.
    group = dim if group_size is None else group_size
    expected_shape = (1, len(probe), layers, dim)
    if tuple(reference.shape) != expected_shape:
        raise AssertionError(f"reference {tuple(reference.shape)} is not {expected_shape}")

    rebuilt = torch.zeros(expected_shape, dtype=torch.float32)
    opened = open_container(container)

    def rows_of(
        wanted: Sequence[tuple[Mapping[str, Any], int]], stride: int
    ) -> dict[tuple[str, int], torch.Tensor]:
        """Collect the needed `(block, token)` with **one read per block** (the rest is dropped).

        Rows are returned as raw u8 bytes (`stride` bytes) — interpreting them (unpacking,
        reading as f32) is the caller's job.
        """
        by_asset: dict[str, tuple[Mapping[str, Any], set[int]]] = {}
        for block, token in wanted:
            by_asset.setdefault(str(block["asset"]), (block, set()))[1].add(token)
        picked: dict[tuple[str, int], torch.Tensor] = {}
        for name, (block, tokens) in by_asset.items():
            raw = bytearray(read_asset(opened, name))
            start = int(block["start"])
            rows = int(block["stop"]) - start
            expected = rows * stride
            if len(raw) != expected:
                raise AssertionError(
                    f"asset '{name}' is {len(raw)} bytes — the expectation built from the index's"
                    f" range [{block['start']}, {block['stop']}) is {expected}"
                )
            view = torch.frombuffer(raw, dtype=torch.uint8).reshape(rows, stride)
            for token in sorted(tokens):
                picked[(name, token)] = view[token - start].clone()
            # Release the whole block's buffer before reading the next block (`view` shares `raw`).
            del view, raw
        return picked

    located: list[tuple[int, int, Mapping[str, Any], Mapping[str, Any] | None]] = []
    for position, token in enumerate(probe):
        blocks = {key: _locate_block(index[key]["blocks"], token) for key in derived}
        values_block = blocks[PLE_VALUES_KEY]
        scales_block = blocks.get(PLE_SCALES_KEY)
        if values_block is None or (not floating and scales_block is None):
            continue
        located.append((position, token, values_block, scales_block))
    if len(located) != len(probe):
        raise AssertionError(
            f"of the {len(probe)} probe tokens only {len(located)} fall in a block's range"
            " (the index's [start, stop) does not cover the vocab)"
        )

    values = rows_of(
        [(block, token) for _p, token, block, _s in located], row_bytes[PLE_VALUES_KEY]
    )
    scales = (
        {}
        if floating
        else rows_of(
            [(block, token) for _p, token, _v, block in located if block is not None],
            row_bytes[PLE_SCALES_KEY],
        )
    )
    for position, token, values_block, scales_block in located:
        packed = values[(str(values_block["asset"]), token)]
        if scales_block is None:
            # bf16: the row's bits widened to f32 (exact), then the embed scale.
            widened = packed.view(torch.bfloat16).to(torch.float32).reshape(layers, dim)
            rebuilt[0, position] = widened * embed_scale
            continue
        quantized = (
            _unpack_ple_values(packed, storage)
            .reshape(layers, dim // group, group)
            .to(torch.float32)
        )
        scale = (
            scales[(str(scales_block["asset"]), token)]
            .view(torch.float32)
            .reshape(layers, dim // group, 1)
        )
        rebuilt[0, position] = (quantized * scale * embed_scale).reshape(layers, dim)
    if not torch.equal(rebuilt, reference):
        worst = float((rebuilt - reference).abs().max())
        raise AssertionError(
            "the PLE relayout is not bit-identical to the 35-table path"
            f" (max abs difference {worst}) — the values and the per-row scales are mismatched,"
            " or the token range is shifted"
        )


# ---- Form checks -----------------------------------------------------------


def assert_soft_merge(graph: IrGraph, config: Any, seq_symbol: str = decode.SEQ_SYMBOL) -> None:
    """The image rows' merge of a model without per-layer embeddings (12B): the declared inputs,
    and one `where(soft_mask, soft_embeds, <the scaled embedding of input_ids>)` as their only
    consumer.

    MUST: checked in the graph — image vectors that pass through `embed_scale`, or a mask left
    unused, keep every shape and every text-only golden.
    """
    expected = {
        SOFT_EMBEDS: ("f32", [1, seq_symbol, int(config.hidden_size)]),
        SOFT_MASK: ("bool", [1, seq_symbol, 1]),
    }
    for spec in graph.inputs:
        if spec.name in expected and (spec.dtype, list(spec.shape)) != expected[spec.name]:
            raise AssertionError(
                f"'{spec.name}' is {spec.dtype} {list(spec.shape)} — not"
                f" {expected[spec.name][0]} {expected[spec.name][1]}"
            )
    merges = [node for node in graph.nodes if node.op == WHERE_OP]
    if len(merges) != 1 or merges[0].ins[:2] != [SOFT_MASK, SOFT_EMBEDS]:
        raise AssertionError(
            f"`{WHERE_OP}` nodes are {[node.ins for node in merges]} — not one"
            f" `{WHERE_OP}({SOFT_MASK}, {SOFT_EMBEDS}, embedding)`: the image vectors must reach it"
            " unscaled"
        )
    consumers = [
        name
        for node in graph.nodes
        if node is not merges[0]
        for name in node.ins
        if name in expected
    ]
    if consumers:
        raise AssertionError(f"{sorted(set(consumers))} have consumers besides the merge")
    producers = {out: node for node in graph.nodes for out in node.outs}
    frontier, seen = [merges[0].ins[2]], set[str]()
    while frontier:
        value = frontier.pop()
        if value in seen:
            continue
        seen.add(value)
        node = producers.get(value)
        if node is not None:
            frontier.extend(node.ins)
    if decode.INPUT_IDS not in seen or SOFT_EMBEDS in seen:
        raise AssertionError(
            f"the merge's text operand does not come from '{decode.INPUT_IDS}' alone"
        )


def assert_ir_form_product(
    graph: IrGraph,
    config: Any,
    storage_expectation: Mapping[str, int],
    ple_rows: int | None,
    *,
    seq_symbol: str = decode.SEQ_SYMBOL,
    capacity_symbol: str = decode.CAPACITY_SYMBOL,
    row_symbol: str = ROW_SYMBOL,
) -> dict[str, Any]:
    """Check the product graph's form (every property that **breaks silently while the numbers
    still match**).

    The states / layer types / residue / storage sections are delegated to
    {@link decode.assert_ir_form_common} (the body the 3 forms share — step 1b spelled them
    independently "because the inputs / exits are exclusive", but bracketing only the shared part
    into a function needs no parameterization of inputs / exits). What this function spells is only
    the inputs and exits specific to the product form.

    The gates specific to the product form (evidence of the PLE moved out and of the selected rows'
    logits / hidden exits):

    - **`per_layer_inputs` is present** among the graph inputs. If not, the PLE is still in the
      graph (2,240MiB resident again, and nobody reads the host gather's input).
    - **There is no** `embedding` that looks up the PLE table's shape
      (`[ple_rows, hidden_size_per_layer_input]`). A form where the input is added yet the table
      remains (wiring that passes both) can be written.
      MUST: `ple_rows` **comes in as an argument** — `config.vocab_size_per_layer_input` is the
      value after {@link gemma4.export.load_model_and_tables} replaced it with the probe model's 8
      rows, so taking it from the config would make this check effectively a no-op (this actually
      happened once).
    - The `embedding` nodes are exactly 2: **the main embedding + the row selection**. Counting
      catches, by number, a form where only part of the PLE remains (one of the 35 left unpruned).
      RoPE became a host-supplied input, so no `embedding` looks up a table any more. A model with
      the expert block adds exactly one router lookup per layer ({@link router_scale_lookups}).
    - The exit is **not argmax**. Even one `argmax` left removes the room for sampling (ADR 0083
      decision 6 — the GPU side goes up to the selected rows' logits).
    - There are **two outputs, in the order logits → hidden**. The runtime reads by slot number,
      and swapped, both are still f32 `[1, R, *]` with matching axes. The order is pinned by two
      things: the width comparison and the structural check that **output 0's ancestors reach
      lm_head** (the latter remains even if the widths happen to match).
    - The declared output shapes are `[1, R, vocab_size]` / `[1, R, hidden_size]` (the selected
      **rows** only). A regression to all-row logits still gives the same token sequence, so only
      a structural check can pin it (what ADR 0068 achieves).
    - The declared shape of `last_row` is `[R]`. It is the only input that can bind R, so a
      regression to a static `[1]` could write a form where "the symbol exists but has no binding
      point".

    `ple_rows=None` is the form of a model without per-layer embeddings (12B —
    `gemma4.export_unified`): the same contract minus PLE, so `per_layer_inputs` MUST be absent
    from the inputs and no PLE table can be looked for; the image rows' inputs are there instead
    ({@link assert_soft_merge}); every other rule holds unchanged. MUST: `None` exactly when the
    config declares no PLE (`hidden_size_per_layer_input` 0) — a caller cannot switch the PLE
    checks off for a model that has one.
    """
    decode.assert_layer_type_count(config)
    layers = int(config.num_hidden_layers)
    ple_dim = gemma4_ple_dim(config, "config")
    if (ple_rows is None) != (ple_dim == 0):
        raise AssertionError(
            f"ple_rows {ple_rows} for a config with hidden_size_per_layer_input {ple_dim}"
            " (None exactly for a model without per-layer embeddings)"
        )

    expected_inputs = [
        decode.INPUT_IDS,
        *([SOFT_EMBEDS, SOFT_MASK] if ple_rows is None else []),
        *decode.ROPE_INPUTS,
        *([PER_LAYER_INPUTS] if ple_rows is not None else []),
        decode.TOKEN_ONLY_LAST_ROW,
    ]
    names = [spec.name for spec in graph.inputs]
    if names != expected_inputs:
        raise AssertionError(
            f"graph inputs are {names}, not {expected_inputs}"
            " (the PLE may still be in the graph / mask or position_ids may be left as inputs"
            " instead of being folded)"
        )
    decode.assert_rope_inputs(graph, config, seq_symbol=seq_symbol)
    last_row_spec = next(spec for spec in graph.inputs if spec.name == decode.TOKEN_ONLY_LAST_ROW)
    if list(last_row_spec.shape) != [row_symbol]:
        raise AssertionError(
            f"declared shape of '{decode.TOKEN_ONLY_LAST_ROW}' is {list(last_row_spec.shape)} —"
            f" not [{row_symbol}] (the only binding point of the symbol {row_symbol})"
        )
    embeddings = [node for node in graph.nodes if node.op == EMBEDDING_OP]
    if ple_rows is None:
        assert_soft_merge(graph, config, seq_symbol)
    else:
        per_layer_spec = next(spec for spec in graph.inputs if spec.name == PER_LAYER_INPUTS)
        expected_shape = [1, seq_symbol, layers, ple_dim]
        if per_layer_spec.dtype != "f32" or list(per_layer_spec.shape) != expected_shape:
            raise AssertionError(
                f"'{PER_LAYER_INPUTS}' is {per_layer_spec.dtype} {list(per_layer_spec.shape)} —"
                f" not f32 {expected_shape}"
            )

        # A PLE table can only exist in a form consumed by **the embedding's weight slot**
        # (`WEIGHT_SLOTS[embedding] = 0`), so that slot is checked by name. Looking only at the
        # initializers' shapes would hit a same-shaped tensor in another slot (e.g. `mlp.down_proj`
        # of a tiny model).
        residents = sorted(
            node.ins[0]
            for node in embeddings
            if list(declared_shape(graph, node.ins[0])) == [ple_rows, ple_dim]
        )
        if residents:
            raise AssertionError(
                f"PLE table shape [{ple_rows}, {ple_dim}] is still looked up by `{EMBEDDING_OP}`"
                f" × {len(residents)}: {residents[:4]}"
                " (not fully moved out to the host gather)"
            )
    # One main embedding (the same data as the tied lm_head) + one row selection of the last row
    # (+ one router lookup per layer of a mixture of experts). Even one PLE left adds to this, and
    # the RoPE table lookup returning adds to it too (both are signs of a regression).
    routed = router_scale_lookups(graph, config, embeddings)
    expected_embeddings = 2 + len(routed)
    if len(embeddings) != expected_embeddings:
        raise AssertionError(
            f"`{EMBEDDING_OP}` × {len(embeddings)} — not 1 main embedding + 1 row selection"
            f" + {len(routed)} router lookups = {expected_embeddings}"
        )

    if len(graph.outputs) != 2:
        raise AssertionError(
            f"{len(graph.outputs)} IR outputs"
            " (the product exit is the 2 outputs logits + hidden — ADR 0083 decision 6 / the"
            " groundwork of speculative verify)"
        )
    if ARGMAX_OP in graph.required_ops:
        raise AssertionError(
            f"`{ARGMAX_OP}` is still in the graph"
            " (the product exit is the selected rows' logits; sampling / argmax are on the host"
            " — ADR 0083 decision 6)"
        )
    logits_shape = list(declared_shape(graph, graph.outputs[0]))
    hidden_shape = list(declared_shape(graph, graph.outputs[1]))
    expected_logits = [1, row_symbol, int(config.vocab_size)]
    expected_hidden = [1, row_symbol, int(config.hidden_size)]
    # MUST: check the order too (they differ only in width and are both f32 [1,R,*], so a swap does
    # not fail on the shape).
    for slot, (found, expected, what) in enumerate(
        ((logits_shape, expected_logits, "logits"), (hidden_shape, expected_hidden, "hidden"))
    ):
        if found != expected:
            raise AssertionError(
                f"declared shape of output {slot} is {found} — not {expected}"
                f" (the selected rows' {what})"
            )

    # The row-selected lm_head is pinned by **the same structural check** as the token-only form
    # (through the one on the decode side).
    producer = {out: node for node in graph.nodes for out in node.outs}
    logits_source = producer.get(graph.outputs[0])
    if logits_source is None:
        raise AssertionError(f"output 0 ('{graph.outputs[0]}') is not a node output")
    decode.assert_row_selected_lm_head(graph, producer, logits_source, rows=row_symbol)

    form = decode.assert_ir_form_common(
        graph,
        config,
        storage_expectation,
        seq_symbol=seq_symbol,
        capacity_symbol=capacity_symbol,
        extra_symbols=(row_symbol,),
    )
    return {
        **form,
        "embedding_nodes": len(embeddings),
        **({"router_lookups": len(routed)} if routed else {}),
        "logits": logits_shape,
        "hidden": hidden_shape,
    }


#: The tensor-name suffix of the MoE router's per-expert scale (upstream's
#: `Gemma4TextRouter.per_expert_scale`, rank 1 `[E]`).
ROUTER_SCALE_SUFFIX = ".router.per_expert_scale"


def router_scale_lookups(graph: IrGraph, config: Any, embeddings: Sequence[Any]) -> list[str]:
    """The `embedding` nodes that are a mixture-of-experts router's `per_expert_scale[index]`,
    by the per-expert-scale tensor each reads (ADR 0145 decision 5: the exporter writes that index
    as an `embedding` over the table reshaped to `[E, 1]`).

    MUST: a model with the expert block (`enable_moe_block`) has exactly one per layer, each over
    its own layer's `[num_experts, 1]` table reshaped from the parameter — not a constant folded
    from it (a shape-only trace would ship its uninitialised bytes); a model without one has none.
    Any other `embedding` stays counted as one that should not be there.
    """
    producers = {out: node for node in graph.nodes for out in node.outs}
    found: list[tuple[str, list[Any]]] = []
    for node in embeddings:
        source = producers.get(node.ins[0])
        if source is None or source.op != RESHAPE_OP or source.ins[0] not in graph.initializers:
            continue
        tensor = graph.initializers[source.ins[0]].tensor
        if tensor.endswith(ROUTER_SCALE_SUFFIX):
            found.append((tensor, list(declared_shape(graph, node.ins[0]))))
    tables = [tensor for tensor, _ in found]
    if not getattr(config, "enable_moe_block", False):
        if tables:
            raise AssertionError(f"router lookups {tables[:4]} in a model without the expert block")
        return []
    layers = int(config.num_hidden_layers)
    if len(tables) != layers or len(set(tables)) != layers:
        raise AssertionError(
            f"{len(tables)} router lookups over {len(set(tables))} tables — not one per layer"
            f" ({layers})"
        )
    experts = int(config.num_experts)
    wrong = {tensor: shape for tensor, shape in found if shape != [experts, 1]}
    if wrong:
        raise AssertionError(f"router lookups read tables not of shape [{experts}, 1]: {wrong}")
    return tables


# ---- Series ----------------------------------------------------------------


def trace(
    wrapper: ProductChunkWrapper,
    example_ids: torch.Tensor,
    per_layer_inputs: torch.Tensor,
    sym_max: int,
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export the product wrapper with the symbolic `M` / `R` and apply the states surgery.

    The one trace of the product graph: this series and the q4_0 QAT line
    ({@link gemma4.export_product_qat}) both call it, so the two graphs cannot drift apart.
    `per_layer_inputs` is the example's `[1,M,L,256]` (the 35-table path's — the values only feed
    the trace). MUST: the wrapper no longer holds `per_layer` (its 35 tables are unused by the
    forward, and a held one is a 9.4GB state_dict the export carries).
    """
    if hasattr(wrapper, "per_layer"):
        raise AssertionError("the product wrapper still holds the PLE tables (drop them first)")
    config = wrapper.model.config
    example_rope = decode.rope_args(decode.rope_specs(config), positions_for(example_ids))
    seq = Dim(decode.SEQ_SYMBOL, min=2, max=sym_max)
    # min=1 is a MUST — ordinary prefill / decode run with R = 1, so specializing it would break the
    # existing path. torch.export keeps a Dim that includes size 1 as a symbol (measured on 2.13).
    rows = Dim(ROW_SYMBOL, min=1, max=ROW_SYM_MAX)
    graph, tensors = export_module(
        wrapper,
        (
            example_ids,
            *example_rope,
            per_layer_inputs,
            decode.last_rows_for(example_ids, EXAMPLE_ROWS),
        ),
        dynamic_shapes=(*({1: seq} for _ in range(2 + len(example_rope))), {0: rows}),
        # The assignment follows the order of the user input placeholders
        # (`karume.convert._assign_input_symbols`), so they are listed as `input_ids[1,M]` →
        # `last_row[R]`.
        symbol_names=(decode.SEQ_SYMBOL, ROW_SYMBOL),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    return to_states_form(graph, decode.states_plan(graph, config)), tensors


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    steps: int = decode.GREEDY_STEPS[GEMMA4_DEFAULT_MODEL],
) -> dict[str, Any]:
    """Write the product container (the PLE as its assets), the PLE probe, the series' own greedy
    goldens of `steps` steps and the provenance record, and return the summary.

    MUST: the outputs are written into the staging directory and placed only after passing **every
    gate** (PLE bit-identity, the form check, the sanity check against the 1-shot expectation
    table, the golden margin). Placing them in the final location before the gates would let a
    failed run leave "assets that can pass the acceptance gate" behind (the discipline of the swap
    and the cleanup is the core primitive {@link karume.artifacts.staged_publication}).
    MUST: the PLE order is "build the reference and each case's inputs on the 35-table path → take
    the goldens (a full re-forward over the same path) → turn the tables into the stored values
    (the checkpoint's bf16 values unchanged — {@link bf16_ple_tables}) → drop `per_layer` →
    export". The other way round, the reference is built on the stored path, and the bit-identity
    gate compares two things wrong in the same direction.
    """
    # Function-scope imports: both modules import this one.
    from gemma4 import export_product_qat as qat
    from gemma4 import export_unified as unified

    wrapper = load_wrapper(model_dir)
    config = wrapper.model.config
    specs = decode.rope_specs(config)
    cases = one_shot.build_cases(model_dir, sym_max, wrapper.sliding_window)
    greedy_cases = tuple(case for case in cases if case[0] in decode.GREEDY_CASES)
    limit = decode.max_position(config)
    assert_case_room(cases, 0, limit)
    assert_case_room(greedy_cases, steps, limit)
    suppress = unified.suppress_tokens(model_dir, config)
    view = qat.GreedyView(wrapper, specs, suppress)
    # Nothing is rounded: the linears and the embedding keep the checkpoint's bf16 values (the
    # writer stores a weight as bf16 only when the round trip is exact, else fails — ADR 0131
    # decision 2), and so does the PLE table ({@link bf16_ple_tables}).
    # Every linear but the tied head, plus the embedding (the head reads the same tensor).
    bf16_weights = sum(1 for _ in iter_quant_targets(wrapper, include=one_shot.is_int4_module)) + 1

    layers = int(config.num_hidden_layers)
    dim = int(config.hidden_size_per_layer_input)
    # MUST: the source of truth for the row count is the loaded split tables (the config field has
    # been replaced with the probe model's 8 rows — {@link ple_table_rows}).
    tokens = ple_table_rows(wrapper.per_layer, int(config.vocab_size))
    embed_scale = float(wrapper.per_layer_scale)
    # The block cuts are determined by the dimensions alone (no data is needed), so the probe can be
    # chosen before the tables are converted.
    ranges = ple_block_ranges(storage=BF16_STORAGE, tokens=tokens, layers=layers, dim=dim)
    probe = ple_probe_tokens(tokens, ranges[PLE_VALUES_KEY])

    # Before the PLE leaves the graph, build ① the bit-identity reference and ② each case's graph
    # inputs once, on **the 35-table path** ({@link gemma4.ple.per_layer_inputs}, which the 3
    # scripts go through).
    print(
        "[ple] building the reference and the case inputs on the 35-table path",
        file=sys.stderr,
        flush=True,
    )
    with torch.no_grad():
        probe_reference = ple.per_layer_inputs(
            wrapper.per_layer, torch.tensor([list(probe)], dtype=torch.int64), embed_scale
        )
        case_inputs = {
            name: ple.per_layer_inputs(wrapper.per_layer, ids, embed_scale) for name, ids in cases
        }

    example_name, example_ids = max(cases, key=lambda case: case[1].shape[1])

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory itself (the primitive does not create it — it only
        # hands out a path).
        staged.mkdir()
        # The goldens of the stored model (the checkpoint's values throughout), by full re-forward
        # over the 35-table path — so before the tables go.
        print(
            f"[goldens] {[name for name, _ in greedy_cases]}, {steps} steps;"
            f" suppressed ids {list(suppress)}",
            file=sys.stderr,
            flush=True,
        )
        written, continuations, margins = _write_greedy(
            view, greedy_cases, staged, steps=steps, floor=decode.MARGIN_FLOOR
        )
        del view

        print("[ple] 35 tables → bf16 values", file=sys.stderr, flush=True)
        values = bf16_ple_tables(wrapper.per_layer)
        # The data was released above. Drop the attribute itself so the unused 35 tables are not
        # carried into the export.
        del wrapper.per_layer

        ple_assets_built = ple_container_assets(
            values,
            None,
            tokens=tokens,
            layers=layers,
            dim=dim,
            embed_scale=embed_scale,
            spill_dir=staged,
            storage=BF16_STORAGE,
        )
        assets = ple_assets_built.assets
        # MUST: the bf16 bodies are released **here** (from now on only the model's RAM goes into
        # the export) — the assets read from the working seat's temporary file, so there is no
        # reason to keep the 35 tables alive.
        values.clear()
        save_file(
            {
                PROBE_TOKENS_KEY: torch.tensor(list(probe), dtype=torch.int32).contiguous(),
                PROBE_INPUTS_KEY: probe_reference.contiguous(),
            },
            str(staged / PLE_PROBE_FILE),
        )

        print("[export] torch.export → conversion", file=sys.stderr, flush=True)
        try:
            surgical, tensors = trace(wrapper, example_ids, case_inputs[example_name], sym_max)
            verified = decode._write_container(
                surgical,
                tensors,
                staged / one_shot.MODEL_FILE,
                # The graph name is **the component name** (= the weights key of karume.json). It is
                # not derived from the directory name — neither the series name
                # (`gemma4-e2b-product-bf16`) nor the staging directory's name matches the
                # component name (container-v1 §2.1).
                graph_name=GEMMA4_ROLE,
                weight_dtype=BF16_STORAGE,
                weight_scales={},
                weight_dtype_overrides={},
                assets=assets,
            )
        finally:
            # MUST: delete the temporary files before the swap (the staging directory is placed as
            # a whole, so leftovers would mix into the distribution).
            ple_assets_built.discard()
        index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
        print(
            "[ple] rebuilding the probe from the placed container's assets",
            file=sys.stderr,
            flush=True,
        )
        assert_ple_assets(staged / one_shot.MODEL_FILE, index, probe, probe_reference)
        # Every linear and the embedding are bf16, and nothing is i8 / i4 (the PLE tables left the
        # graph).
        storage_expectation = {BF16_STORAGE: bf16_weights, "i8": 0, "i4": 0}
        form = assert_ir_form_product(verified, config, storage_expectation, tokens)

        print("[sanity] full-length forward", file=sys.stderr, flush=True)
        first: dict[str, int] = {}
        for name, ids in cases:
            with torch.no_grad():
                # Look up the same R rows as the example input and check the top-1 of the **last
                # row** (the sanity check also exercises the R > 1 row selection once — only the
                # continuation of the prompt's last row is compared with the expectation table).
                logits, _hidden = wrapper(
                    ids,
                    *decode.rope_args(specs, positions_for(ids)),
                    case_inputs[name],
                    decode.last_rows_for(ids, EXAMPLE_ROWS),
                )
            first[name] = int(logits[0, -1].argmax())

        # Compare the first continuation token with the 1-shot script's expectation table (a
        # cross-mechanism comparison — the scripts are different, so agreement of the top-1 with
        # the same weights and the same prompt is a cross-check).
        # MUST: evaluate before publishing (on failure the staging directory goes with it).
        tokenizer = one_shot.load_tokenizer(model_dir)
        expected = {
            name: token
            for name, token in one_shot.expected_token_ids(tokenizer).items()
            if name in first
        }
        labels = {
            token: tokenizer.id_to_token(token)
            for token in set(first.values()) | set(expected.values())
        }
        sanity = one_shot._sanity(first, expected, labels)

        # The provenance record goes into **the same directory** as the container (one swap, so a
        # pair of new container + old record cannot arise). The series name it states is the name
        # after placement — not the staging directory's name.
        print("[provenance] checkpoint fingerprint → reference.json", file=sys.stderr, flush=True)
        # The series' own goldens, as the q4_0 QAT series records them.
        own = provenance.assert_reference_goldens(staged, greedy_cases)
        reference_goldens = {**own, "series": out_dir.name, "suppressTokens": list(suppress)}
        record = provenance.build_record(out_dir, model_dir, reference_goldens)
        provenance.write_record(staged, record)

    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "model_bytes": sum(
            path.stat().st_size for path in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        # PLE is an asset inside the container, so its byte count comes from the row count the
        # index states × one row (the container's parts also hold the graph's weights, so it cannot
        # be cut out of the file sizes).
        "ple_bytes": tokens * ple_token_bytes(layers, dim, BF16_STORAGE),
        "ple_blocks": {
            key: len(index[key]["blocks"])
            for key in (PLE_VALUES_KEY, PLE_SCALES_KEY)
            if key in index
        },
        "ple_probe_tokens": list(probe),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "quantized": {BF16_STORAGE: bf16_weights},
        "form": form,
        "reference": record["reference"],
        "sanity": sanity,
        "greedy": written,
        "greedy_steps": steps,
        "margin_min": {name: min(values) for name, values in sorted(margins.items())},
        "continuation": {
            name: tokenizer.decode(continuation)
            for name, continuation in sorted(continuations.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = one_shot.series_parser(
        __doc__.split("\n\n")[0], partial(one_shot.series_dir, suffix=GEMMA4_PRODUCT_BF16_SUFFIX)
    )
    # The greedy steps (default: the model's `gemma4.export_decode.GREEDY_STEPS`).
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)
    # The default that depends on the model is filled here; the rest by `resolve_series_args` (the
    # model defaults of `series_parser`).
    if args.steps is None:
        args.steps = decode.GREEDY_STEPS[args.model]
    options = one_shot.resolve_series_args(args)
    model_dir = options.pop("model_dir")
    summary = export_series(model_dir, options.pop("out"), **options)
    print(json.dumps({"model_dir": str(model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
