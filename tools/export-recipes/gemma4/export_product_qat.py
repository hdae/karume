"""Writes Gemma 4 E2B / E4B (`--model`) as the product graph from Google's q4_0 QAT checkpoints,
every 2-D text weight imported from the checkpoint's own 4-bit grid, with its greedy goldens.

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m gemma4.export_product_qat              # E2B
    uv run --with 'transformers==5.14.1' python -m gemma4.export_product_qat --model e4b  # E4B

## The regular product series' graph, from other weights

The graph is the regular product series' (`gemma4.export_product`): the same wrapper
(`ProductChunkWrapper` — PLE supplied by the host, `logits[1,R,V]` then `hidden[1,R,H]`), the same
loader with its three equivalence checks ({@link load_wrapper}), the same trace
(`gemma4.export_product.trace`), the same form check (`assert_ir_form_product`) and the same PLE
probe check (`assert_ple_assets`). What differs is where the weights come from and how they are
stored:

1. **Two pinned sources** (`gemma4.sources.QAT_PINS`): the weights and their config are the hub
   cache's snapshot of `google/gemma-4-E{2,4}B-it-qat-q4_0-unquantized`, the tokenizer and the
   generation defaults the regular `google/gemma-4-E{2,4}B-it` at the revision of the regular
   line's hand-placed checkpoint. Both are checked against their pins before the load, and the
   provenance record's full digests once more before it is written (12B's rule —
   `gemma4.export_unified`).
2. **No k / v residues**: the QAT checkpoints carry nothing for the KV-shared layers' k / v, so the
   loader expects none (`kv_residues=False`) instead of expecting some to drop.
3. **Every 2-D text weight is imported from its q4_0 grid instead of rounded** ({@link
   import_wrapper}): every linear (including `per_layer_input_gate`, `per_layer_projection` and
   `per_layer_model_projection`), the embedding (and the tied head — one tensor) and the PLE table.
   Measured 2026-11-03 on both pinned snapshots: every group of 32 along the last axis of each of
   them recovers ({@link GROUP_SIZE}). The recovery is 12B's (`gemma4.export_unified.grid_import`:
   codes in [−8, 7] and one F32 step per group; a group that does not recover fails the export
   with the tensor and the group). The wrapper's F32 values are replaced by
   `dequantize_int4(codes, steps)` — the values the container's bytes decode to — so the sanity,
   the goldens and the PLE probe run on the stored model (ADR 0006). The linears and the embedding
   are written as fixed packed i4 (`fixed_weights` — ADR 0097's entry; nibble `q + 8`, the runtime
   decodes `u − 8`); no i8 is left.
4. **The PLE asset is schema 4** (`karume.ple` — {@link PLE_STORAGE} values, one F32 step per
   group of {@link GROUP_SIZE}). Within one (token, layer) row the eight groups of the table do not
   share a step (measured: the spread of the steps within a row has a median of 45%), so schema
   3's one scale per row cannot hold the grid. The reference does not depend on this choice: the
   asset holds the codes and steps of the grid, and the probe check reads it back bit for bit
   against the reference's per-layer inputs.
5. **Own goldens**: there is no QAT decode series to borrow `greedy.<case>.safetensors` from (the
   regular product's `--reference`), so the series takes its own, as 12B does: greedy
   continuations of {@link QAT_GREEDY_CASES} by full re-forward of the reference
   (`_shared.decode_series._write_greedy`), the margin floor of `gemma4.export_decode.MARGIN_FLOOR`
   and the step counts of `gemma4.export_decode.GREEDY_STEPS`, with the ids
   `generation_config.json` suppresses masked first (none for E2B / E4B). A case whose margin falls
   under the floor is dropped from {@link QAT_GREEDY_CASES}; the steps are never cut. The goldens
   are taken before the long trace and write, so a failing margin stops the run early.

The first token of every case is checked against `gemma4.export.GREEDY_EXPECTATIONS` (the regular
series' table) on the stored model, with the row count of the trace's example (R > 1), before the
goldens are taken.

## Output layout

    outputs/series/gemma4-<model>-qat-product/model.krm                  the container + PLE assets
    outputs/series/gemma4-<model>-qat-product/ple.probe.safetensors      the PLE read-back reference
    outputs/series/gemma4-<model>-qat-product/greedy.<case>.safetensors  the greedy goldens
    outputs/series/gemma4-<model>-qat-product/reference.json             provenance (both sources,
                                                                         own goldens)

The TypeScript reader (`packages/models/src/gemma/ple-index.ts`) reads schema 3 and 4; the q4_0
QAT line's distribution (`gemma4.distribution.QAT_Q4_0_PIPELINE`) takes this series and requires
schema 4. The regular and the mobile QAT series stay schema 3.

The embedding's group scale is `262144 × hidden / 32` F32 — 48 MiB for E2B, 80 MiB for E4B — over
the 32 MiB block limit, so the writer splits it on the embedding's pieces (container-v1 §5 rule ③,
ADR 0108 addendum 15). The linears' scales and the schema-4 PLE (an asset cut in whole rows) fit
in one block.

MUST: transformers is imported inside functions (the loader's rule).
"""

from __future__ import annotations

import gc
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from _shared.decode_series import _write_greedy, assert_case_room, positions_for
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_unified as unified
from gemma4 import ple, provenance, sources
from gemma4.card import GEMMA4_LICENSE
from gemma4.distribution import GEMMA4_DEFAULT_MODEL, GEMMA4_QAT_PRODUCT_SUFFIX, GEMMA4_ROLE
from gemma4.sources import ModelPins
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.dist import NOTICE_FILENAME
from karume.emit import FixedQuantizedWeight, pack_int4, unpack_int4
from karume.ir import IrGraph
from karume.pipeline import publish_model
from karume.ple import PLE_INDEX_ASSET, ple_block_ranges
from karume.quantize import DEFAULT_GROUP_SIZE, dequantize_int4
from qwen3_5.export import MEMORY_FLOOR_GIB, MemoryWatch

#: The q4_0 grid's group: 32 consecutive values along the last axis share a step.
GROUP_SIZE = DEFAULT_GROUP_SIZE

#: The PLE asset's storage (the grid's codes, packed two to a byte).
PLE_STORAGE = "i4"

#: Model → the cases whose greedy continuations become goldens (the margin floor's rule: a case
#: under the floor is dropped, the steps are never cut). Measured on the first export of each
#: model (2026-11-03): E2B's `context-en` is a near tie at its 10th of 16 steps (margin 0.0209,
#: under `decode.MARGIN_FLOOR`), so E2B keeps `capital-en` / `capital-ja` (lowest margins 0.085 /
#: 0.054 in 16 steps). E4B keeps all three (lowest 0.052 / 0.136 / 0.087 in 10 steps). NOTE:
#: neither of E2B's cases is longer than the sliding window, so E2B's goldens no longer reach past
#: it; the first-token sanity still runs `context-en` (T = 598) on the stored model.
QAT_GREEDY_CASES: Mapping[str, tuple[str, ...]] = {
    "e2b": tuple(case for case in decode.GREEDY_CASES if case != "context-en"),
    "e4b": decode.GREEDY_CASES,
}


class QatExportError(RuntimeError):
    """The checkpoint or the model is not what the q4_0 QAT line expects."""


def load_wrapper(model_dir: Path) -> product.ProductChunkWrapper:
    """`gemma4.export_product.load_wrapper` for a checkpoint without the KV-shared layers' k / v
    residues: the same loader (`gemma4.export.load_model_and_tables` — its three equivalence
    checks) told to expect none, the same RoPE swap (`gemma4.export_decode.load_wrapper`'s two
    steps) and the same wrapper type, whose module paths the grid import and the trace read."""
    model, tables = one_shot.load_model_and_tables(model_dir, kv_residues=False)
    del model.model.embed_tokens_per_layer
    decode.swap_rope_inputs(model)
    return product.ProductChunkWrapper(model, tables).eval()


# ---------------------------------------------------------------------------
# The grid import
# ---------------------------------------------------------------------------


def import_grid(weight: torch.Tensor, where: str) -> FixedQuantizedWeight:
    """Import the q4_0 grid of a rank-2 F32 weight held in memory, and replace its values **in
    place** by the grid's `dequantize_int4(codes, steps)`.

    Returns packed i4 (`[rows, cols / 2]` uint8 — `karume.emit.pack_int4`'s nibble order, `q + 8`)
    and the steps (`[rows, cols / GROUP_SIZE]` F32). A block of rows at a time
    (`gemma4.export_unified.CHUNK_BYTES` of F32 — `grid_import` is 12B's, per block; a group never
    crosses a row, so the blocks give the whole tensor's codes and steps), so no whole-tensor
    temporary is held besides the result. MUST: a group that does not recover fails loudly with
    `where` (the tensor) and the group (`grid_import`). MUST: the packed bytes are checked to decode
    to the values written back (the writer's round-trip rule for quantized storage — ADR 0069
    decision 4 ③, as `gemma4.export_unified.quantize_int4_rows`).
    """
    if weight.dim() != 2 or weight.dtype != torch.float32:
        raise QatExportError(f"{where}: {weight.dtype} rank {weight.dim()} (F32 rank 2 only)")
    rows, cols = weight.shape
    if cols % GROUP_SIZE:
        raise QatExportError(
            f"{where}: row length {cols} does not split into groups of {GROUP_SIZE}"
        )
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    steps = torch.empty((rows, cols // GROUP_SIZE), dtype=torch.float32)
    block = max(1, unified.CHUNK_BYTES // (cols * 4))
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        codes, block_steps = unified.grid_import(weight[start:stop], GROUP_SIZE, where, start)
        values = dequantize_int4(codes, block_steps)
        block_packed = pack_int4(codes).reshape(stop - start, cols // 2)
        if not torch.equal(
            dequantize_int4(unpack_int4(block_packed, codes.shape), block_steps), values
        ):
            raise QatExportError(f"{where} rows {start}..{stop}: packed i4 does not round-trip")
        weight[start:stop] = values
        packed[start:stop] = block_packed
        steps[start:stop] = block_steps
    return FixedQuantizedWeight(dtype="i4", packed=packed, scale=steps)


@dataclass
class GridImport:
    """{@link import_wrapper}'s result: the imported grid of every 2-D text weight."""

    #: Wrapper key → packed i4 + steps, for every linear and the tied embedding (under
    #: `gemma4.export_unified.EMBED_KEY` — the trace keeps one of the pair's two names).
    fixed: dict[str, FixedQuantizedWeight]
    #: Per PLE layer, the table's packed codes (`[tokens, dim / 2]`) and steps
    #: (`[tokens, dim / GROUP_SIZE]`) — the schema-4 asset's payload. A list so the caller can
    #: release it once the asset is spilled.
    ple: list[FixedQuantizedWeight] = field(default_factory=list)


def ple_where(layer: int, dim: int) -> str:
    """How an error names PLE layer `layer`'s table (the checkpoint's one table, and the columns
    the layer reads — `gemma4.ple`)."""
    columns = f"{layer * dim}..{(layer + 1) * dim}"
    return f"'{one_shot.PLE_CHECKPOINT_KEY}' layer {layer} (columns {columns})"


def import_wrapper(wrapper: product.ProductChunkWrapper) -> GridImport:
    """Import the grid of every 2-D text weight of `wrapper` in place ({@link import_grid}): the
    linears (`gemma4.export_unified.int4_targets` — every linear but the tied head), the embedding
    (the head reads the same tensor) and the PLE tables.

    MUST: every rank-2 parameter of the wrapper is one of those three classes — a new 2-D weight in
    upstream's model fails here instead of being stored as F32 next to the grid.
    """
    linears = unified.int4_targets(wrapper)
    fixed: dict[str, FixedQuantizedWeight] = {}
    for key in sorted(linears):
        module = wrapper.get_submodule(key.removesuffix(".weight"))
        fixed[key] = import_grid(module.weight.data, f"'{key}'")
    embedding = wrapper.get_submodule(one_shot.EMBED_TOKENS_MODULE).weight
    if embedding is not wrapper.get_submodule(one_shot.LM_HEAD_MODULE).weight:
        raise QatExportError("the head is not tied to the embedding")
    fixed[unified.EMBED_KEY] = import_grid(embedding.data, f"'{unified.EMBED_KEY}'")
    dim = int(wrapper.model.config.hidden_size_per_layer_input)
    tables = [
        import_grid(table.weight.data, ple_where(layer, dim))
        for layer, table in enumerate(wrapper.per_layer)
    ]
    imported = {
        *fixed,
        *(f"{one_shot.PER_LAYER_PREFIX}{layer}.weight" for layer in range(len(tables))),
    }
    planes = {name for name, parameter in wrapper.named_parameters() if parameter.dim() == 2}
    if planes != imported:
        raise QatExportError(
            f"2-D weights outside the grid import: {sorted(planes - imported)[:4]}"
            f" / imported but not parameters: {sorted(imported - planes)[:4]}"
        )
    return GridImport(fixed=fixed, ple=tables)


def container_material(
    graph: IrGraph, tensors: Mapping[str, torch.Tensor], fixed: Mapping[str, FixedQuantizedWeight]
) -> tuple[dict[str, torch.Tensor], dict[str, FixedQuantizedWeight]]:
    """The writer's tensors (meta for the fixed weights, the trace's F32 for the rest) and its
    fixed weights keyed by the names the graph declares (the tied embedding under whichever of the
    pair's two names the trace kept).

    MUST: every fixed weight is declared by the graph — a linear the trace dropped would otherwise
    leave its grid unwritten without a word.
    """
    declared = sorted({init.tensor for init in graph.initializers.values()})
    embedding = [key for key in declared if key in (unified.EMBED_KEY, unified.TIED_HEAD_KEY)]
    if len(embedding) != 1:
        raise QatExportError(f"the graph declares the tied embedding as {embedding}")
    named = {
        (embedding[0] if key == unified.EMBED_KEY else key): value for key, value in fixed.items()
    }
    missing = sorted(set(named) - set(declared))
    if missing:
        raise QatExportError(f"imported weights the graph does not declare: {missing[:4]}")
    material = {
        key: (torch.empty(tensors[key].shape, device="meta") if key in named else tensors[key])
        for key in declared
    }
    return material, named


# ---------------------------------------------------------------------------
# The reference's face for the goldens
# ---------------------------------------------------------------------------


class GreedyView(nn.Module):
    """`(input_ids, position_ids) → (logits of the last row, hidden)` — the face
    `_shared.decode_series.greedy_continuation` calls, over the product wrapper with the host's PLE
    lookup done by the 35-table path (`gemma4.ple.per_layer_inputs` over the wrapper's imported
    tables — what the graph's `per_layer_inputs` input receives).

    The logits come back with the suppressed ids at −inf (`gemma4.export_unified.suppressed_logits`
    — what the host's selection sees through `pipelineConfig.suppressTokens`).
    """

    def __init__(
        self,
        wrapper: product.ProductChunkWrapper,
        specs: Mapping[str, Any],
        suppress: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.specs = dict(specs)
        self.suppress = tuple(suppress)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_layer = ple.per_layer_inputs(
            self.wrapper.per_layer, input_ids, self.wrapper.per_layer_scale
        )
        logits, hidden = self.wrapper(
            input_ids,
            *decode.rope_args(self.specs, position_ids),
            per_layer,
            decode.last_row_for(input_ids),
        )
        return unified.suppressed_logits(logits, self.suppress), hidden


def first_tokens(
    wrapper: product.ProductChunkWrapper,
    cases: Sequence[tuple[str, torch.Tensor]],
    case_inputs: Mapping[str, torch.Tensor],
    suppress: Sequence[int],
) -> dict[str, int]:
    """The first continuation token of each case: the last of `EXAMPLE_ROWS` selected rows (the
    regular series' sanity — R > 1 is exercised once), suppressed ids masked."""
    specs = decode.rope_specs(wrapper.model.config)
    first: dict[str, int] = {}
    for name, ids in cases:
        with torch.no_grad():
            logits, _hidden = wrapper(
                ids,
                *decode.rope_args(specs, positions_for(ids)),
                case_inputs[name],
                decode.last_rows_for(ids, product.EXAMPLE_ROWS),
            )
        first[name] = int(unified.suppressed_logits(logits, suppress)[0, -1].argmax())
    return first


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    sym_max: int = one_shot.SYM_MAX,
    steps: int = decode.GREEDY_STEPS[GEMMA4_DEFAULT_MODEL],
    greedy_cases: Sequence[str] = decode.GREEDY_CASES,
    checkpoint_label: str | None = None,
    floor: float = MEMORY_FLOOR_GIB,
) -> dict[str, Any]:
    """Write the container (PLE assets included), the PLE probe, the greedy goldens and the
    provenance record through one staged seat, and return the summary.

    `model_dir` holds the QAT weights and their config, `tokenizer_dir` the tokenizer and the
    generation defaults; `pins` names both sources (`None` for a stand-in checkpoint — the record
    then carries no `sources` and the container no upstream revision). `checkpoint_label` is how
    the record names the checkpoint (`gemma4.sources.ModelPins.label` for the pinned snapshots).

    MUST: every gate (first tokens, margins, the PLE read-back, the form, the pinned fingerprint)
    passes before the seat is published — a failed run leaves nothing a gate could later accept.
    MUST: the grid import precedes every reference value (the sanity, the goldens, the probe and
    the per-layer inputs of the trace — ADR 0006), and the PLE probe and the per-layer inputs are
    taken by the 35-table path before the tables are dropped (the regular series' order).
    """
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    unknown = sorted(set(greedy_cases) - {name for name, _ in one_shot.GOLDEN_CASES})
    if unknown or not greedy_cases:
        raise QatExportError(f"greedy cases {list(greedy_cases)}: unknown {unknown}")
    with MemoryWatch(floor) as monitor:
        with monitor.stage("load"):
            wrapper = load_wrapper(model_dir)
        config = wrapper.model.config
        with monitor.stage("grid") as record:
            print("[grid] import every 2-D text weight", file=sys.stderr, flush=True)
            grid = import_wrapper(wrapper)
            imported = len(grid.fixed)
            record.details["fixed"] = imported
            record.details["ple_tables"] = len(grid.ple)
        specs = decode.rope_specs(config)
        cases = one_shot.build_cases(tokenizer_dir, sym_max, wrapper.sliding_window)
        golden = tuple(case for case in cases if case[0] in greedy_cases)
        limit = decode.max_position(config)
        assert_case_room(cases, 0, limit)
        assert_case_room(golden, steps, limit)
        suppress = unified.suppress_tokens(tokenizer_dir, config)

        layers = int(config.num_hidden_layers)
        dim = int(config.hidden_size_per_layer_input)
        tokens = product.ple_table_rows(wrapper.per_layer, int(config.vocab_size))
        embed_scale = float(wrapper.per_layer_scale)
        ranges = ple_block_ranges(
            storage=PLE_STORAGE, tokens=tokens, layers=layers, dim=dim, group_size=GROUP_SIZE
        )
        probe = product.ple_probe_tokens(tokens, ranges[product.PLE_VALUES_KEY])
        with torch.no_grad():
            probe_reference = ple.per_layer_inputs(
                wrapper.per_layer, torch.tensor([list(probe)], dtype=torch.int64), embed_scale
            )
            case_inputs = {
                name: ple.per_layer_inputs(wrapper.per_layer, ids, embed_scale)
                for name, ids in cases
            }

        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            with monitor.stage("sanity"):
                print("[sanity] first tokens of the stored model", file=sys.stderr, flush=True)
                first = first_tokens(wrapper, cases, case_inputs, suppress)
                tokenizer = one_shot.load_tokenizer(tokenizer_dir)
                expected = one_shot.expected_token_ids(tokenizer)
                labels = {
                    token: tokenizer.id_to_token(token)
                    for token in set(first.values()) | set(expected.values())
                }
                sanity = one_shot._sanity(first, expected, labels)
            with monitor.stage("goldens"):
                print(
                    f"[goldens] {[name for name, _ in golden]}, {steps} steps;"
                    f" suppressed ids {list(suppress)}",
                    file=sys.stderr,
                    flush=True,
                )
                view = GreedyView(wrapper, specs, suppress)
                written, continuations, margins = _write_greedy(
                    view, golden, staged, steps=steps, floor=decode.MARGIN_FLOOR
                )
                del view

            with monitor.stage("ple"):
                print("[ple] schema-4 assets from the imported tables", file=sys.stderr, flush=True)
                built = product.ple_container_assets(
                    [table.packed for table in grid.ple],
                    [table.scale for table in grid.ple],
                    tokens=tokens,
                    layers=layers,
                    dim=dim,
                    embed_scale=embed_scale,
                    spill_dir=staged,
                    storage=PLE_STORAGE,
                    group_size=GROUP_SIZE,
                )
                # The payload is in the spill files now: neither the packed tables nor the F32
                # tables are needed past this point (the trace never reads `per_layer`).
                grid.ple.clear()
                del wrapper.per_layer
                gc.collect()
                save_file(
                    {
                        product.PROBE_TOKENS_KEY: torch.tensor(list(probe), dtype=torch.int32),
                        product.PROBE_INPUTS_KEY: probe_reference.contiguous(),
                    },
                    str(staged / product.PLE_PROBE_FILE),
                )
            try:
                example_name, example_ids = max(cases, key=lambda case: case[1].shape[1])
                with monitor.stage("trace") as record:
                    print("[export] torch.export → states form", file=sys.stderr, flush=True)
                    graph, tensors = product.trace(
                        wrapper, example_ids, case_inputs[example_name], sym_max
                    )
                    record.details["nodes"] = len(graph.nodes)
                material, fixed = container_material(graph, tensors, grid.fixed)
                del tensors, wrapper, case_inputs
                grid.fixed.clear()
                gc.collect()
                with monitor.stage("write"):
                    print("[export] write the container", file=sys.stderr, flush=True)
                    verified = publish_model(
                        staged / one_shot.MODEL_FILE,
                        graph,
                        material,
                        provenance=Provenance(
                            license=GEMMA4_LICENSE,
                            notice=NOTICE_FILENAME,
                            upstream_revision=None if pins is None else pins.weights.revision,
                        ),
                        graph_name=GEMMA4_ROLE,
                        weight_dtype="f32",
                        fixed_weights=fixed,
                        assets=built.assets,
                    )
                del material, fixed
            finally:
                # MUST: the spill files go before the seat is published (they would ship).
                built.discard()
            index = json.loads(bytes(built.assets[PLE_INDEX_ASSET].payload))
            with monitor.stage("check"):
                print(
                    "[ple] read the probe back from the written assets", file=sys.stderr, flush=True
                )
                product.assert_ple_assets(
                    staged / one_shot.MODEL_FILE, index, probe, probe_reference
                )
                # Every linear and the tied embedding are i4. MUST: no i8 is counted as well — an
                # embedding rounded to i8 leaves the graph's shapes and ops unchanged.
                storage = {"i4": imported, "i8": 0}
                form = product.assert_ir_form_product(verified, config, storage, tokens)
            with monitor.stage("provenance"):
                own = provenance.assert_reference_goldens(staged, golden)
                record_ = provenance.build_record(
                    out_dir,
                    model_dir,
                    {**own, "series": out_dir.name, "suppressTokens": list(suppress)},
                    checkpoint_label=checkpoint_label,
                    tokenizer_dir=tokenizer_dir,
                    pins=pins,
                )
                if pins is not None:
                    provenance.assert_pinned_fingerprint(record_["checkpoint"], pins)
                provenance.write_record(staged, record_)
    parts = container_parts(out_dir / one_shot.MODEL_FILE)
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "parts": [path.stat().st_size for path in parts],
        "model_bytes": sum(path.stat().st_size for path in parts),
        "ple_bytes": tokens
        * sum(
            int(index[key]["rowBytes"]) for key in (product.PLE_VALUES_KEY, product.PLE_SCALES_KEY)
        ),
        "ple_blocks": {
            key: len(index[key]["blocks"])
            for key in (product.PLE_VALUES_KEY, product.PLE_SCALES_KEY)
        },
        "ple_probe_tokens": list(probe),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "quantized": storage,
        "form": form,
        "sanity": sanity,
        "greedy": written,
        "greedy_steps": steps,
        "margin_min": {name: min(values) for name, values in sorted(margins.items())},
        "continuation": {
            name: tokenizer.decode(continuation)
            for name, continuation in sorted(continuations.items())
        },
        "suppress_tokens": list(suppress),
        "reference": record_["reference"],
        "stages": [stage.to_dict() for stage in monitor.records],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = one_shot.series_parser(
        __doc__.split("\n\n")[0], partial(one_shot.series_dir, suffix=GEMMA4_QAT_PRODUCT_SUFFIX)
    )
    # The weights' default is the QAT line's pinned snapshot, not the regular line's hand-placed
    # checkpoint (`series_parser` registers that one).
    one_shot.add_model_default(parser, "model_dir", sources.qat_checkpoint_dir)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="the tokenizer and generation defaults (default: the model's pinned tokenizer source)",
    )
    one_shot.add_model_default(parser, "tokenizer_dir", sources.qat_tokenizer_dir)
    parser.add_argument("--steps", type=int, default=None)
    one_shot.add_model_default(parser, "steps", decode.GREEDY_STEPS.__getitem__)
    parser.add_argument("--memory-floor", type=float, default=MEMORY_FLOOR_GIB)
    args = parser.parse_args(argv)
    model = args.model
    options = one_shot.resolve_series_args(args)
    pins = sources.qat_pins(model)
    model_dir = options.pop("model_dir")
    summary = export_series(
        model_dir,
        options.pop("out"),
        tokenizer_dir=options["tokenizer_dir"],
        pins=pins,
        sym_max=options["sym_max"],
        steps=options["steps"],
        greedy_cases=QAT_GREEDY_CASES[model],
        checkpoint_label=pins.label,
        floor=options["memory_floor"],
    )
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "tokenizer_dir": str(options["tokenizer_dir"]),
                **summary,
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
