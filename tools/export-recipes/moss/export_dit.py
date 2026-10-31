"""Export of the MOSS-Image DiT as an i8 series — the graphs `dit`, `dit_skip22`, `dit_text` and
the assets `rope_base` and `training_grid` (ADR 0138 decisions 2–4, 6 and 11, stages 1
and 2).

    uv run --group moss --inexact python -m moss.export_dit inspect  # traces + structure only
    uv run --group moss --inexact python -m moss.export_dit write    # the series, behind both gates
    uv run --group moss --inexact python -m moss.export_dit eager    # the eager gate on the series
    uv run --group moss --inexact python -m moss.export_dit eager --without-series  # before one

`write` reads the stage-0 host record (`moss.host_fixture`; default
`outputs/bench/karume-moss-image/<today>_stage0-reference/host.safetensors`, another one by
`--host-record`) for gate 3.

Each command prints a JSON summary and writes it to `--out` (default
`outputs/bench/karume-moss-image/<date>_dit-export/<command>.json`); a failed gate exits non-zero.

## The series (`outputs/series/moss-image-sft-i8-dyn/` — docs/assets-layout.md)

One part directory per graph, each with its container `model.krm` (graph name = part name):

| Part         | Symbol | Weights                                             | Assets            |
| ------------ | ------ | --------------------------------------------------- | ----------------- |
| `dit`        | S      | 258 linears i8 (per-row scale), norms, biases f32   | `rope_base`,      |
|              |        |                                                     | `training_grid`   |
| `dit_skip22` | S      | none — every initializer declared shared with `dit` | —                 |
| `dit_text`   | T      | 11 linears i8, norms and biases f32                 | —                 |

`dit_text` runs with f32 compute in every preset (ADR 0138 decision 3 — a session setting of the
presets, stage 5; the weights are i8 like the rest).

The graphs and their inputs are {@link moss.dit_patch}. Both assets are safetensors bytes of
upstream outputs, which the host only reorders or indexes (ADR 0034 decision 3, ADR 0138
decision 11):

- `rope_base` (role `rope-base`): the per-axis sin / cos base tables
  ({@link moss.dit_patch.rope_base_tables}).
- `training_grid` (role `training-grid`, {@link training_grid_tables}): the 1,000-point training
  grid of `MOSSImageScheduler` — `sigmas` and `timesteps` `[1000]` f32 (descending, the grid every
  requested schedule snaps to) and `time_sinusoid` `[1000, freq_dim]` f32 (row i = what
  `MOSSImageTransformer.forward` feeds its time MLP at `timesteps[i]`, the graph input
  `timesteps_proj`). A timestep is always one of these 1,000 values (upstream snaps σ to the grid
  and the timestep is σ × 1000), so the host indexes the table by the training-grid index and
  never evaluates the sinusoid. The values come from upstream code exactly as `moss.host_fixture`
  takes them (its own functions are called — one path), and gate 3 checks them against that
  record.

## Export (the ADR 0121 decision 7 form — no exporter-core change)

The graphs are traced on a weightless (meta) upstream model built from `transformer/config.json`.
The 269 linear weights are read from the pinned bf16 checkpoint in row chunks, widened to f32
(exact) and quantized to per-row symmetric i8 the way the exporter does ({@link quantize_rows}),
and passed as `fixed_weights`; the other weights (RMSNorm weights, biases) are the checkpoint values
in f32. `dit_skip22` is traced the same way and every initializer is replaced by a shared
declaration of the same tensor key in the written `dit` (ADR 0096 decision 2, the ADR 0134 form).
Everything is written into a staging directory and moved into place only when both gates hold.

## Gate 1 — structure ({@link inspect_structure})

- inputs, output and the one symbol as declared; `is_image` bool `[1,S,1]`;
- no S- or T-dependent constant baked in: no initializer of rank ≥ 3 (ADR 0034 decision 2's form —
  the weights are rank ≤ 2), no initializer with a dimension equal to the trace's symbol maximum, no
  `sym_prefix_slice` (the exporter's form of a folded symbol-dependent constant — ADR 0010);
- the census ({@link EXPECTED}): linear nodes, `where`, attention, `div` (two per block — `situ`'s
  `g / 4` and `h / 25` stay divisions, ADR 0138 decision 6);
- storage: i8 exactly on the quantization targets, f32 on the rest; `dit_skip22` all shared, and its
  keys are `dit`'s minus block 22's.

## Gate 2 — eager equivalence, bit-identical (MUST — ADR 0138 decision 11)

One process, one set of f32 fake-quant weights: upstream's `MOSSImageTransformer` read in f32 and
rounded in place to the i8 grid (`karume.quantize.fake_quant_int8` — the same rounding as the
export), shared by upstream and the patched graphs (so the weights sit at the same addresses for
both — the MKL placement note of ADR 0121 decision 7). Per case ({@link CASES}):

1. Upstream `MOSSImageTransformer.forward` itself (positive: all blocks; negative:
   `skip_layers=(22,)`), with only `video_embedder` and `final_linear_video` called on zero-padded S
   rows ({@link moss.dit_patch.padded_rows}). Forward hooks keep what upstream fed and produced on
   the way: the time sinusoid, the RoPE table, the `text_embedder` output, the final norm's input.
2. `dit_text` (eager) on the embedding and the RoPE rows of the text ≡ upstream's `text_embedder`
   output.
3. The RoPE table assembled from `rope_base` ≡ upstream's table.
4. `dit` / `dit_skip22` (eager) on the inputs built from 2 and 3 and the hooked sinusoid; its first
   N rows ≡ upstream's sample.

Observations (recorded, not gated): upstream's unmodified forward against the M-aligned one, and
whether the final norm splits by row count (its image rows on N rows vs on S rows). With the
written series, the gate also checks that every i8 weight of the containers dequantizes to the
fake-quant weight the comparison ran with (packed × scale in f32, bit for bit).

## Gate 3 — the host tables ({@link host_tables_gate})

The `training_grid` asset read back from the written `dit` container is bit-identical to the
stage-0 host record (`host.safetensors` of `moss.host_fixture`: `training.sigmas`,
`training.timesteps`, `time_sinusoid` — computed there at two torch thread counts). The record is
required: without it `write` fails loudly.

RAM: `write` runs gate 2 in the same process, so it peaks with gate 2's f32 DiT (20.3 GB) plus the
comparison's work space — measured 28.54 GiB peak RSS (`write.json`, 2026-10-27; `eager
--without-series` alone 27.93 GiB). `inspect` only traces on the meta device and is light.

MUST: diffusers is imported inside the functions (`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load, save
from torch import nn
from torch.export import Dim

from _shared.container_read import open_container, read_asset
from _shared.paths import BENCH_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import AssetInput, Provenance, codec_entry, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.dist import NOTICE_FILENAME
from karume.emit import FixedQuantizedWeight, storage_breakdown
from karume.ir import IrGraph, IrInitializer, IrShared, IrStorage
from karume.ops import SYM_PREFIX_SLICE_OP
from karume.pipeline import export_module, publish_model
from karume.quantize import (
    QUANT_MODULE_TYPES,
    channel_scale,
    fake_quant_int8,
    iter_quant_targets,
    quantize_to_int8,
)
from karume.verify import ir_graph_from_container, verify_container
from moss import dit_patch, host_fixture, pipeline_ref, records, sources, text_embeds
from moss.dit_patch import DIT, DIT_SKIP22, DIT_TEXT, SKIP_BLOCK
from moss.prompts import NEGATIVE, POSITIVE
from moss.records import bits_equal

#: The i8 series (the `-dyn` suffix: symbolic-length graphs — ADR 0077's convention).
SERIES_NAME = "moss-image-sft-i8-dyn"
SERIES = SERIES_ROOT / SERIES_NAME

#: Container graph names = part directory names (container-v1 §2.1; `tests/test_graph_names.py`).
GRAPHS = (DIT, DIT_SKIP22, DIT_TEXT)
MODEL_FILE = "model.krm"

#: The asset of the `dit` container (ADR 0034 decision 3; the role the TS side reads it by).
ROPE_BASE_ASSET = "rope_base"
ROPE_BASE_ROLE = "rope-base"

#: The second asset of the `dit` container: the training grid's tables (module docstring;
#: ADR 0138 decision 11).
TRAINING_GRID_ASSET = "training_grid"
TRAINING_GRID_ROLE = "training-grid"

#: The tables of `training_grid` → their names in the stage-0 host record (gate 3).
TRAINING_GRID_RECORD_KEYS: Mapping[str, str] = {
    "sigmas": "training.sigmas",
    "timesteps": "training.timesteps",
    "time_sinusoid": "time_sinusoid",
}

#: torch.export's bounds of the symbols. They bound the trace only — IR symbols carry no bounds and
#: the accepted sizes are the host's (ADR 0121 decision 3's note). S: above the area a 4 GiB binding
#: would admit (2³² B / (192·4 B) / 256 px ≈ 21,845 image tokens) plus 4,096 text tokens. T: above
#: upstream's `max_sequence_length` (4,096).
#: MUST: neither equals a static extent of the graphs (4,096 is `text_dim`, 8,192 `ffn_dim`) — the
#: structure gate reads "a dimension equal to the symbol maximum" as a baked constant
#: ({@link symbol_constants}).
SYM_MAX = {"S": 32_768, "T": 4_608}
SYM_MIN = 2

#: The trace's example sizes (values do not matter; S = 266 and T = 10 avoid torch.export's 0 / 1
#: specialization and are the negative branch's 256² shape).
TRACE_LATENT = (16, 16)
TRACE_TEXT = 10

#: Rows per checkpoint read when quantizing (the largest DiT weight, `shared_block_modulation`
#: `[18432, 3072]`, takes two reads).
CHUNK_ROWS = 16_384

#: The census each graph must show (linear nodes are i8 weights except where a graph borrows them;
#: `div` is two per block — `situ`).
EXPECTED: Mapping[str, Mapping[str, int]] = {
    DIT: {"linear": 258, "where": 1, "attention": 42, "div": 84},
    DIT_SKIP22: {"linear": 252, "where": 1, "attention": 41, "div": 82},
    DIT_TEXT: {"linear": 11, "where": 0, "attention": 2, "div": 4},
}

#: The output names of the bench directory (docs/assets-layout.md).
BENCH_PURPOSE = "dit-export"


class DitExportError(RuntimeError):
    """A graph, a weight or a comparison left the form step 1 accepts."""


def default_out() -> Path:
    return BENCH_ROOT / records.BENCH_NAME / f"{date.today().isoformat()}_{BENCH_PURPOSE}"


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


def load_config(snapshot: Path) -> dict[str, Any]:
    """`MOSSImageTransformer`'s constructor arguments from `transformer/config.json`."""
    raw = json.loads((snapshot / "transformer" / "config.json").read_text(encoding="utf-8"))
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def meta_model(module: ModuleType, config: Mapping[str, Any]) -> nn.Module:
    """The upstream DiT without weights (parameters on the meta device; the derived `rope.bands`
    stays on the CPU — `_RoPE` builds it there explicitly)."""
    with torch.device("meta"):
        return module.MOSSImageTransformer(**config).eval()


def wrap(name: str, model: nn.Module) -> nn.Module:
    """The patched module of graph `name` ({@link moss.dit_patch})."""
    if name == DIT:
        return dit_patch.MossDit(model)
    if name == DIT_SKIP22:
        return dit_patch.MossDit(model, skip_layers=frozenset({SKIP_BLOCK}))
    if name == DIT_TEXT:
        return dit_patch.MossDitText(model)
    raise DitExportError(f"unknown graph {name!r} ({', '.join(GRAPHS)})")


def symbol_of(name: str) -> str:
    return "T" if name == DIT_TEXT else "S"


def example_inputs(
    name: str, config: Mapping[str, Any], *, device: str = "meta"
) -> tuple[torch.Tensor, ...]:
    """The trace's example inputs in the graph's input order (values do not reach the trace)."""
    dim, head_dim = int(config["dim"]), int(config["dim"]) // int(config["num_heads"])
    with torch.device(device):
        if name == DIT_TEXT:
            return (
                torch.empty(1, TRACE_TEXT, int(config["text_dim"])),
                torch.empty(1, 1, TRACE_TEXT, head_dim),
                torch.empty(1, 1, TRACE_TEXT, head_dim),
            )
        tokens = TRACE_LATENT[0] * TRACE_LATENT[1] + TRACE_TEXT
        return (
            torch.empty(1, tokens, int(config["latent_channels"])),
            torch.empty(1, tokens, dim),
            torch.empty(1, tokens, 1, dtype=torch.bool),
            torch.empty(1, 1, tokens, head_dim),
            torch.empty(1, 1, tokens, head_dim),
            torch.empty(1, int(config["freq_dim"])),
        )


def dynamic_shapes(name: str, sym_max: int) -> tuple[Any, ...]:
    """One symbol per graph; the RoPE tables share it (a separate symbol would accept tables of
    another length — the Wan / Anima rule)."""
    symbol = Dim(symbol_of(name), min=SYM_MIN, max=sym_max)
    if name == DIT_TEXT:
        return ({1: symbol}, {2: symbol}, {2: symbol})
    return ({1: symbol}, {1: symbol}, {1: symbol}, {2: symbol}, {2: symbol}, None)


@dataclass(frozen=True)
class Traced:
    """One traced graph: the IR, its tensors (weights on meta) and the quantization targets."""

    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    #: FQN of every linear weight (the i8 targets — upstream's state-dict keys).
    targets: tuple[str, ...]


def trace(
    name: str,
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    sym_max: int | None = None,
    device: str = "meta",
) -> Traced:
    """Export graph `name` of `model` (meta for the series; a real tiny model in the tests)."""
    wrapper = wrap(name, model)
    symbol = symbol_of(name)
    graph, tensors = export_module(
        wrapper,
        example_inputs(name, config, device=device),
        dynamic_shapes=dynamic_shapes(name, SYM_MAX[symbol] if sym_max is None else sym_max),
        symbol_names=(symbol,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    targets = tuple(
        sorted(fqn for fqn, _, _ in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES))
    )
    return Traced(graph=graph, tensors=tensors, targets=targets)


# ---------------------------------------------------------------------------
# Gate 1 — structure
# ---------------------------------------------------------------------------


def census(graph: IrGraph) -> dict[str, int]:
    """Node count per op."""
    return dict(sorted(Counter(node.op for node in graph.nodes).items()))


def symbol_constants(graph: IrGraph, sym_max: int) -> list[str]:
    """What looks like a baked symbol-dependent constant (module docstring "Gate 1")."""
    found = [
        f"initializer {name} {graph.values[name].shape} (rank ≥ 3)"
        for name in graph.initializers
        if len(graph.values[name].shape) >= 3
    ]
    found += [
        f"initializer {name} {graph.values[name].shape} (a dimension equals the symbol maximum)"
        for name in graph.initializers
        if sym_max in graph.values[name].shape
    ]
    found += [
        f"node {node.outs} is {SYM_PREFIX_SLICE_OP}"
        for node in graph.nodes
        if node.op == SYM_PREFIX_SLICE_OP
    ]
    return found


def declared_inputs(name: str, config: Mapping[str, Any]) -> list[list[Any]]:
    """`[name, shape, dtype]` of the graph inputs as ADR 0138 decisions 2 and 3 declare them."""
    dim, head_dim = int(config["dim"]), int(config["dim"]) // int(config["num_heads"])
    if name == DIT_TEXT:
        rope = [1, 1, "T", head_dim]
        return [
            ["text_embeds", [1, "T", int(config["text_dim"])], "f32"],
            ["rope_cos", rope, "f32"],
            ["rope_sin", rope, "f32"],
        ]
    rope = [1, 1, "S", head_dim]
    return [
        ["latent_tokens", [1, "S", int(config["latent_channels"])], "f32"],
        ["text_context", [1, "S", dim], "f32"],
        ["is_image", [1, "S", 1], "bool"],
        ["rope_cos", rope, "f32"],
        ["rope_sin", rope, "f32"],
        ["timesteps_proj", [1, int(config["freq_dim"])], "f32"],
    ]


def declared_output(name: str, config: Mapping[str, Any]) -> list[Any]:
    if name == DIT_TEXT:
        return [1, "T", int(config["dim"])]
    return [1, "S", int(config["latent_channels"])]


def inspect_structure(
    name: str,
    traced: Traced,
    config: Mapping[str, Any],
    *,
    sym_max: int | None = None,
    expected: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Gate 1's storage-independent items for one graph (`failures` empty = passed).

    `expected` defaults to {@link EXPECTED} for the real configuration; the tests pass the counts of
    their tiny models.
    """
    graph, symbol = traced.graph, symbol_of(name)
    ops = census(graph)
    inputs = [[entry.name, list(entry.shape), entry.dtype] for entry in graph.inputs]
    outputs = [list(graph.values[output].shape) for output in graph.outputs]
    failures: list[str] = []
    if list(graph.symbols) != [symbol]:
        failures.append(f"symbols {graph.symbols} (expected [{symbol!r}])")
    if inputs != declared_inputs(name, config):
        failures.append(f"inputs {inputs}")
    if outputs != [declared_output(name, config)]:
        failures.append(f"outputs {outputs}")
    failures += symbol_constants(graph, SYM_MAX[symbol] if sym_max is None else sym_max)
    counts = EXPECTED[name] if expected is None else expected
    for op, count in counts.items():
        if ops.get(op, 0) != count:
            failures.append(f"{ops.get(op, 0)} {op} nodes (expected {count})")
    if ops.get("linear", 0) != len(traced.targets):
        failures.append(f"{ops.get('linear', 0)} linear nodes for {len(traced.targets)} weights")
    return {
        "ops": ops,
        "symbols": list(graph.symbols),
        "inputs": inputs,
        "outputs": outputs,
        "linear_weights": len(traced.targets),
        "initializers": len(graph.initializers),
        "failures": failures,
    }


def inspect_storage(graph: IrGraph, targets: Sequence[str]) -> dict[str, Any]:
    """i8 exactly on `targets`, f32 on every other own initializer (a committed graph)."""
    storage = {
        initializer.tensor: initializer.storage.dtype
        for initializer in graph.initializers.values()
        if initializer.tensor is not None
    }
    i8 = sorted(key for key, dtype in storage.items() if dtype == "i8")
    other = sorted(key for key, dtype in storage.items() if dtype not in ("i8", "f32"))
    failures = []
    if i8 != sorted(targets):
        failures.append(f"i8 on {len(i8)} tensors, {len(targets)} targets")
    if other:
        failures.append(f"storage neither i8 nor f32: {other[:5]}")
    return {"i8_weights": len(i8), "f32_tensors": len(storage) - len(i8), "failures": failures}


def inspect_shared(graph: IrGraph, lender: IrGraph) -> dict[str, Any]:
    """`dit_skip22`: every initializer shared, and the keys are the lender's minus block 22's."""
    own = sorted(name for name, init in graph.initializers.items() if not init.is_shared)
    borrowed = {init.shared.tensor for init in graph.initializers.values() if init.shared}
    block = f"layers.{SKIP_BLOCK}."
    expected = {key for key in lender.initializers if not key.startswith(block)}
    failures = []
    if own:
        failures.append(f"initializers with bytes: {own[:5]}")
    if borrowed != expected:
        failures.append(
            f"shared keys differ from the lender's minus {block}*:"
            f" extra {sorted(borrowed - expected)[:5]}, missing {sorted(expected - borrowed)[:5]}"
        )
    return {"shared": len(borrowed), "owned": len(own), "failures": failures}


# ---------------------------------------------------------------------------
# Weights (the pinned bf16 checkpoint)
# ---------------------------------------------------------------------------

#: The diffusers spelling of the sharded checkpoint's index.
CHECKPOINT_INDEX = "diffusion_pytorch_model.safetensors.index.json"


class Checkpoint:
    """The DiT checkpoint read one tensor (or one row range) at a time, widened to f32.

    MUST: every tensor a graph reads is BF16 (the pinned checkpoint's dtype); another dtype fails
    loudly — the i8 rounding starts from these values.
    """

    def __init__(self, directory: Path) -> None:
        index = json.loads((directory / CHECKPOINT_INDEX).read_text(encoding="utf-8"))
        self._files = {key: directory / file for key, file in index["weight_map"].items()}

    def names(self) -> frozenset[str]:
        return frozenset(self._files)

    @contextmanager
    def _slice(self, key: str) -> Iterator[Any]:
        with safe_open(str(self._files[key]), framework="pt") as handle:
            view = handle.get_slice(key)
            if view.get_dtype() != "BF16":
                raise DitExportError(f"checkpoint '{key}' is {view.get_dtype()} (BF16 expected)")
            yield view

    def shape(self, key: str) -> list[int]:
        with self._slice(key) as view:
            return list(view.get_shape())

    def read(self, key: str) -> torch.Tensor:
        with self._slice(key) as view:
            return view[:].to(torch.float32)

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        with self._slice(key) as view:
            return view[start:stop].to(torch.float32)


def quantize_rows(
    checkpoint: Checkpoint, key: str, chunk_rows: int = CHUNK_ROWS
) -> FixedQuantizedWeight:
    """Per-row symmetric i8 of a `[rows, cols]` weight, read in row chunks.

    The steps are the exporter's two-stage form (the Wan precedent `wan.umt5_export.quantize_rows`):
    `fake_quant_int8`'s `quantize(w, s)·s`, then the stored `quantize(q·s, s)`, with the inverse
    checked. The scale is a row's amax, so chunks give the values of the whole tensor.
    """
    shape = checkpoint.shape(key)
    if len(shape) != 2:
        raise DitExportError(f"'{key}' has rank {len(shape)} (per-row i8 takes rank 2)")
    rows, cols = shape
    packed = torch.empty((rows, cols), dtype=torch.int8)
    scale = torch.empty((rows, 1), dtype=torch.float32)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        weight = checkpoint.read_rows(key, start, stop)
        chunk_scale = channel_scale(weight, 0)
        rounded = quantize_to_int8(weight, chunk_scale).to(torch.float32) * chunk_scale
        del weight
        chunk = quantize_to_int8(rounded, chunk_scale)
        if not torch.equal(chunk.to(torch.float32) * chunk_scale, rounded):
            raise DitExportError(f"'{key}' rows {start}–{stop}: i8 × scale does not invert")
        packed[start:stop] = chunk
        scale[start:stop] = chunk_scale
    return FixedQuantizedWeight(dtype="i8", packed=packed, scale=scale)


@dataclass(frozen=True)
class Material:
    """What `publish_model` takes for one graph (targets on meta, their bytes in `fixed`)."""

    tensors: dict[str, torch.Tensor]
    fixed: dict[str, FixedQuantizedWeight]


def materialize(traced: Traced, checkpoint: Checkpoint) -> Material:
    """The plain weights from the checkpoint in f32 and the targets as per-row i8."""
    weights = sorted(key for key, value in traced.tensors.items() if value.is_meta)
    missing = sorted(set(weights) - checkpoint.names())
    if missing:
        raise DitExportError(f"graph weights missing from the checkpoint: {missing[:5]}")
    unknown = sorted(set(traced.targets) - set(weights))
    if unknown:
        raise DitExportError(f"quantization targets that are no graph weight: {unknown[:5]}")
    tensors = dict(traced.tensors)
    fixed: dict[str, FixedQuantizedWeight] = {}
    for key in weights:
        if key in traced.targets:
            fixed[key] = quantize_rows(checkpoint, key)
            if list(fixed[key].packed.shape) != list(tensors[key].shape):
                raise DitExportError(f"'{key}': checkpoint shape differs from the graph")
            continue
        value = checkpoint.read(key)
        if list(value.shape) != list(tensors[key].shape):
            raise DitExportError(f"'{key}': checkpoint shape {list(value.shape)} differs")
        tensors[key] = value
    return Material(tensors=tensors, fixed=fixed)


def share_lender_weights(graph: IrGraph, lender: IrGraph) -> IrGraph:
    """Every initializer of `graph` → a shared declaration of the same tensor key in `lender` (the
    lender read back from its container — initializers keyed by tensor key), with the lender's
    storage (ADR 0134 decision 1's form)."""
    initializers: dict[str, IrInitializer] = {}
    for name, initializer in graph.initializers.items():
        key = initializer.tensor
        lent = lender.initializers.get(key) if key is not None else None
        if lent is None or lent.is_shared:
            raise DitExportError(f"initializer '{name}' ({key}) has no tensor in the lender")
        mine, theirs = graph.values[name], lender.values[key]
        if (mine.dtype, list(mine.shape)) != (theirs.dtype, list(theirs.shape)):
            raise DitExportError(
                f"'{key}': {mine.dtype} {list(mine.shape)} vs the lender's"
                f" {theirs.dtype} {list(theirs.shape)}"
            )
        initializers[name] = IrInitializer(
            shared=IrShared(tensor=key), storage=IrStorage(dtype=lent.storage.dtype)
        )
    return IrGraph(
        symbols=list(graph.symbols),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        initializers=initializers,
        values=dict(graph.values),
        states=dict(graph.states),
        nodes=list(graph.nodes),
    )


def provenance() -> Provenance:
    """The upstream licence and revision (`moss.sources` is the source of truth)."""
    return Provenance(
        license=sources.SOURCE.license,
        notice=NOTICE_FILENAME,
        upstream_revision=sources.SOURCE.revision,
    )


def rope_base_asset(module: ModuleType, config: Mapping[str, Any]) -> AssetInput:
    """The asset `rope_base` from a fresh upstream `_RoPE` (the loaded DiT's `rope` is the same
    function — it rebuilds its bands on load)."""
    head_dim = int(config["dim"]) // int(config["num_heads"])
    rope = module._RoPE(head_dim, float(config["rope_base"]))
    payload = save(dit_patch.rope_base_tables(rope))
    return AssetInput(ROPE_BASE_ROLE, len(payload), payload)


def training_grid_tables(
    module: ModuleType, config: Mapping[str, Any], scheduler_config: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """The tables of the asset `training_grid` (module docstring), from upstream code through
    `moss.host_fixture`'s own functions: the training grid of upstream `set_timesteps` and, per
    grid timestep, the input of `time_embedding[0]` of an upstream `MOSSImageTransformer` with the
    DiT's `freq_dim` (the sinusoid depends on `freq_dim` and the timestep only)."""
    sigmas, timesteps = host_fixture.training_grid(module, scheduler_config)
    model = host_fixture.tiny_transformer(module, int(config["freq_dim"]))
    sinusoid = torch.cat([host_fixture.time_sinusoid(model, t) for t in timesteps])
    return {"sigmas": sigmas, "timesteps": timesteps, "time_sinusoid": sinusoid}


def training_grid_asset(
    module: ModuleType, config: Mapping[str, Any], scheduler_config: Mapping[str, Any]
) -> AssetInput:
    """The asset `training_grid` ({@link training_grid_tables})."""
    payload = save(training_grid_tables(module, config, scheduler_config))
    return AssetInput(TRAINING_GRID_ROLE, len(payload), payload)


def host_tables_gate(container: Path, record: Path) -> dict[str, Any]:
    """Gate 3: the `training_grid` asset of a written `dit` container ≡ the stage-0 host record
    (bit for bit; `failures` empty = passed). A missing record fails loudly."""
    if not record.is_file():
        raise DitExportError(
            f"no stage-0 host record at {record} — run `python -m moss.host_fixture` or pass"
            " `--host-record`"
        )
    expected, _metadata = records.read_record(record, key=host_fixture.METADATA_KEY)
    tables = load(read_asset(container, TRAINING_GRID_ASSET))
    failures = []
    if set(tables) != set(TRAINING_GRID_RECORD_KEYS):
        failures.append(f"training_grid tables {sorted(tables)}")
    for name, key in TRAINING_GRID_RECORD_KEYS.items():
        if key not in expected:
            failures.append(f"the host record has no '{key}'")
        elif name in tables and not bits_equal(tables[name], expected[key]):
            failures.append(f"training_grid '{name}' differs from the host record's '{key}'")
    return {
        "record": str(record),
        "record_sha256": sources.sha256_of(record),
        "tables": {name: list(value.shape) for name, value in sorted(tables.items())},
        "failures": failures,
    }


def write_container(
    path: Path,
    graph: IrGraph,
    material: Material,
    *,
    name: str,
    assets: Mapping[str, AssetInput] = {},
) -> IrGraph:
    """One container (graph name = part name); returns the committed graph."""
    return publish_model(
        path,
        graph,
        material.tensors,
        provenance=provenance(),
        graph_name=name,
        fixed_weights=material.fixed,
        assets=assets,
    )


def read_graph(path: Path, name: str) -> IrGraph:
    """A written container's graph (initializers keyed by tensor key)."""
    return ir_graph_from_container(verify_container(container_parts(path)).read, name)


# ---------------------------------------------------------------------------
# Gate 2 — eager equivalence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EagerCase:
    """One comparison: image size (pixels), the step of the 2-step schedule, prompt, branch."""

    height: int
    width: int
    step: int
    prompt: str
    branch: str

    @property
    def name(self) -> str:
        return f"{self.height}x{self.width}-step{self.step}-{self.branch}-{self.prompt}"

    @property
    def graph(self) -> str:
        return DIT if self.branch == POSITIVE else DIT_SKIP22


#: The cases: the stage-0 sizes (256² and the non-square 256×384, so H and W cannot be swapped
#: unnoticed), both steps of the 2-step schedule, every positive prompt (T = 63 / 134 / 62) and the
#: empty negative (T = 10) with the skip of block 22. The step picks the timestep only: a step-1
#: case's latent is fresh noise from seed `SEED + 1`, not the latent after step 0 (the check is
#: functional equivalence of one forward, not a trajectory).
CASES: tuple[EagerCase, ...] = (
    EagerCase(256, 256, 0, "fridge-note", POSITIVE),
    EagerCase(256, 256, 0, "negative", NEGATIVE),
    EagerCase(256, 256, 1, "guide-whiteboard", POSITIVE),
    EagerCase(256, 256, 1, "negative", NEGATIVE),
    EagerCase(256, 384, 0, "guide-fridge-note", POSITIVE),
    EagerCase(256, 384, 0, "negative", NEGATIVE),
    EagerCase(256, 384, 1, "fridge-note", POSITIVE),
    EagerCase(256, 384, 1, "negative", NEGATIVE),
)

#: The seed of the cases' latents (step i draws seed + i; the stage-0 reference's seed).
SEED = 20261027

#: The steps of the schedule the cases take their timesteps from (the stage-0 reference's).
STEPS = 2


class Hooks:
    """Forward hooks on upstream modules that keep one value each per call (fails loudly on a
    second value — one forward per `Hooks`)."""

    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _keep(self, name: str, value: torch.Tensor) -> None:
        if name in self.values:
            raise DitExportError(f"{name} was seen twice in one forward")
        self.values[name] = value.detach().clone()

    def input_of(self, module: nn.Module, name: str) -> None:
        self._handles.append(
            module.register_forward_pre_hook(lambda _m, args: self._keep(name, args[0]))
        )

    def output_of(self, module: nn.Module, name: str) -> None:
        self._handles.append(
            module.register_forward_hook(lambda _m, _args, output: self._keep(name, output))
        )

    def __enter__(self) -> Hooks:
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    differs = actual.ne(expected)
    return {
        "bit_identical": bits_equal(actual, expected),
        "differing_elements": int(differs.sum()),
        "max_abs_diff": float((actual - expected).abs().max()),
        "max_abs": float(expected.abs().max()),
    }


@torch.no_grad()
def eager_case(
    model: nn.Module,
    graphs: Mapping[str, nn.Module],
    rope_base: Mapping[str, torch.Tensor],
    embeds: torch.Tensor,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    branch: str,
    observe: bool = True,
) -> dict[str, Any]:
    """One comparison (module docstring "Gate 2"); `failures` empty = bit-identical.

    `graphs` holds the patched modules by graph name (built on `model`, sharing its weights);
    `embeds` is `[T, text_dim]`, `latents` `[1, C, H, W]`, `timestep` upstream's 0-d f32 value.
    """
    height, width = int(latents.shape[2]), int(latents.shape[3])
    image, text = height * width, int(embeds.shape[0])
    skip = (SKIP_BLOCK,) if branch == NEGATIVE else ()
    context = embeds[None]
    with Hooks() as hooks, dit_patch.padded_rows(model, image + text):
        hooks.input_of(model.time_embedding[0], "time_sinusoid")
        hooks.output_of(model.rope, "rope")
        hooks.output_of(model.text_embedder, "text_context")
        hooks.input_of(model.final_norm_video, "final_norm_input")
        aligned = model(latents, timestep, context, skip_layers=skip).sample
    upstream = hooks.values
    rope = dit_patch.rope_table(rope_base, height, width, text)
    text_cos, text_sin = dit_patch.split_rope(rope[:, image:])
    refined = graphs[DIT_TEXT](context, text_cos, text_sin)
    inputs = dit_patch.dit_inputs(latents, refined, rope, upstream["time_sinusoid"])
    graph = graphs[DIT if branch == POSITIVE else DIT_SKIP22]
    with Hooks() as patched_hooks:
        patched_hooks.input_of(graph.final_norm_video, "final_norm_input")
        output = graph(*inputs)
    patched = dit_patch.head_rows(output, height, width)
    report: dict[str, Any] = {
        "N": image,
        "T": text,
        "S": image + text,
        "rope_base": _difference(rope, upstream["rope"]),
        "dit_text": _difference(refined, upstream["text_context"]),
        "dit": _difference(patched, aligned),
    }
    failures = [
        f"{key}: not bit-identical ({report[key]['differing_elements']} elements,"
        f" max {report[key]['max_abs_diff']:.3e})"
        for key in ("rope_base", "dit_text", "dit")
        if not report[key]["bit_identical"]
    ]
    if observe:
        norm = graph.final_norm_video
        rows = patched_hooks.values["final_norm_input"]
        report["observation"] = {
            "final_norm_input_image_rows": _difference(
                rows[:, :image], upstream["final_norm_input"]
            ),
            "final_norm_on_n_vs_s_rows": _difference(
                norm(rows)[:, :image], norm(rows[:, :image].contiguous())
            ),
            "unaligned_upstream": _difference(
                model(latents, timestep, context, skip_layers=skip).sample, aligned
            ),
        }
    report["failures"] = failures
    return report


def fake_quant(model: nn.Module, graphs: Mapping[str, nn.Module]) -> dict[str, Any]:
    """Round every linear weight of `model` in place to the i8 grid and check it reached exactly
    the graphs' quantization targets."""
    report = fake_quant_int8(model)
    # The graphs' FQNs are the model's (they hold upstream's modules under upstream's names).
    targets = {
        fqn
        for graph in graphs.values()
        for fqn, _, _ in iter_quant_targets(graph, op_types=QUANT_MODULE_TYPES)
    }
    if set(report.scales) != targets:
        raise DitExportError(
            f"fake-quant rounded {len(report.scales)} weights, the graphs have {len(targets)}:"
            f" {sorted(set(report.scales) ^ targets)[:5]}"
        )
    return {"rounded": report.describe()}


def container_weights(path: Path, model: nn.Module) -> dict[str, Any]:
    """Every i8 weight of a written container dequantizes (packed × scale in f32) to the model's
    weight under the same key, and every f32 tensor equals it — bit for bit."""
    verified = open_container(path)
    parameters = dict(model.named_parameters())
    read = verified.read
    differing: list[str] = []
    counts: Counter[str] = Counter()
    for bound in verified.graphs.values():
        values = bound.declaration["values"]
        for key, supply in bound.supplies.items():
            if key not in parameters:
                counts["constants"] += 1
                continue
            layout = codec_entry(supply.encoding.codec).layout
            payload = b"".join(read.block(b.id)[: b.payload_bytes] for b in supply.blocks)
            shape = tuple(values[key]["shape"])
            if layout == "i8":
                assert supply.scale is not None
                raw = read.block(supply.scale.id)[: supply.scale.payload_bytes]
                scale = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(-1, 1)
                packed = torch.frombuffer(bytearray(payload), dtype=torch.int8).reshape(shape)
                value = packed.to(torch.float32) * scale
            elif layout == "f32":
                value = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(shape)
            else:
                raise DitExportError(f"{path}: '{key}' stored as {layout}")
            counts[layout] += 1
            if not bits_equal(value, parameters[key].detach()):
                differing.append(key)
    return {"counts": dict(counts), "differing": differing}


def eager_gate(
    snapshot: Path,
    module: ModuleType,
    *,
    containers: Mapping[str, Path] | None = None,
    cases: Sequence[EagerCase] = CASES,
    clock: records.Clock,
) -> dict[str, Any]:
    """Gate 2 on the real weights (module docstring); `failures` empty = passed.

    Each case runs one forward at the timestep of its step (`timesteps[case.step]` of the
    {@link STEPS}-step schedule) on fresh noise from seed `SEED + case.step` — for step 1 that is
    not the latent after step 0. The step picks the timestep only; the check is the functional
    equivalence of one forward, not a trajectory (the trajectory is `moss.reference`'s).
    """
    model = pipeline_ref.load_transformer(snapshot, module, torch.float32)
    clock.lap("load transformer (f32)")
    graphs = {name: wrap(name, model) for name in GRAPHS}
    quantized = fake_quant(model, graphs)
    clock.lap("fake-quant i8")
    head_dim = int(model.config["dim"]) // int(model.config["num_heads"])
    rope_base = dit_patch.rope_base_tables(module._RoPE(head_dim, float(model.config["rope_base"])))
    embeds, _metadata = text_embeds.read_asset(
        SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME
    )
    scheduler = pipeline_ref.load_scheduler(snapshot, module)
    scheduler.set_timesteps(STEPS)
    timesteps = scheduler.timesteps.clone()
    reports: list[dict[str, Any]] = []
    for case in cases:
        latents = pipeline_ref.seeded_latents(SEED + case.step, case.height, case.width)
        report = eager_case(
            model,
            graphs,
            rope_base,
            embeds[case.prompt],
            latents,
            timesteps[case.step],
            branch=case.branch,
        )
        report = {"case": case.name, "graph": case.graph, **report}
        reports.append(report)
        clock.lap(f"eager {case.name}")
        print(f"[eager] {json.dumps(report, ensure_ascii=False)}", flush=True)
    failures = {r["case"]: r["failures"] for r in reports if r["failures"]}
    checked: dict[str, Any] = {}
    if containers is not None:
        for name, path in containers.items():
            checked[name] = container_weights(path, model)
            if checked[name]["differing"]:
                failures[f"container {name}"] = checked[name]["differing"][:5]
            clock.lap(f"container weights {name}")
    return {
        "fake_quant": quantized,
        "timesteps": [float(t) for t in timesteps],
        "cases": reports,
        "container_weights": checked,
        "failures": failures,
        "environment": records.environment(),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def inspect_only() -> dict[str, Any]:
    """Gate 1's storage-independent items on the meta traces (config only, no weights)."""
    snapshot, module = pipeline_ref.upstream()
    config = load_config(snapshot)
    model = meta_model(module, config)
    graphs = {}
    for name in GRAPHS:
        graphs[name] = inspect_structure(name, trace(name, model, config), config)
    failures = {name: g["failures"] for name, g in graphs.items() if g["failures"]}
    return {"graphs": graphs, "failures": failures, "passed": not failures}


def container_paths(directory: Path) -> dict[str, Path]:
    """The container of each graph under a series directory (part directory = graph name)."""
    return {name: directory / name / MODEL_FILE for name in GRAPHS}


def write_graphs(
    directory: Path,
    module: ModuleType,
    config: Mapping[str, Any],
    scheduler_config: Mapping[str, Any],
    checkpoint: Checkpoint,
    *,
    clock: records.Clock,
    sym_max: int | None = None,
    expected: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Trace the three graphs on a meta model of `config`, write `dit` (with the assets `rope_base`
    and `training_grid` — the latter from `scheduler_config`) and `dit_text` with the checkpoint's
    weights and `dit_skip22` borrowing every weight of `dit`, into `directory`
    ({@link container_paths}); returns gate 1's report per graph (`failures` empty = passed).

    `sym_max` / `expected` default to the real configuration's ({@link SYM_MAX} / {@link EXPECTED});
    the tests pass their tiny model's.
    """
    model = meta_model(module, config)
    paths = container_paths(directory)
    graphs: dict[str, dict[str, Any]] = {}

    def structure_of(name: str, traced: Traced) -> dict[str, Any]:
        counts = None if expected is None else expected[name]
        return inspect_structure(name, traced, config, sym_max=sym_max, expected=counts)

    for name in (DIT, DIT_TEXT):
        traced = trace(name, model, config, sym_max=sym_max)
        structure = structure_of(name, traced)
        clock.lap(f"trace {name}")
        material = materialize(traced, checkpoint)
        clock.lap(f"quantize {name}")
        paths[name].parent.mkdir()
        # Asset order = layout order (`rope_base` first keeps its block id `a.0`).
        assets = (
            {
                ROPE_BASE_ASSET: rope_base_asset(module, config),
                TRAINING_GRID_ASSET: training_grid_asset(module, config, scheduler_config),
            }
            if name == DIT
            else {}
        )
        graph = write_container(paths[name], traced.graph, material, name=name, assets=assets)
        del material
        gc.collect()
        storage = inspect_storage(graph, traced.targets)
        breakdown = storage_breakdown(graph)
        graphs[name] = {
            **structure,
            **storage,
            "compressed_bytes": breakdown.compressed_bytes,
            "plain_bytes": breakdown.plain_bytes,
            "failures": structure["failures"] + storage["failures"],
        }
        clock.lap(f"write {name}")
    traced = trace(DIT_SKIP22, model, config, sym_max=sym_max)
    structure = structure_of(DIT_SKIP22, traced)
    lender = read_graph(paths[DIT], DIT)
    paths[DIT_SKIP22].parent.mkdir()
    write_container(
        paths[DIT_SKIP22],
        share_lender_weights(traced.graph, lender),
        Material(tensors={}, fixed={}),
        name=DIT_SKIP22,
    )
    shared = inspect_shared(read_graph(paths[DIT_SKIP22], DIT_SKIP22), lender)
    graphs[DIT_SKIP22] = {
        **structure,
        **shared,
        "failures": structure["failures"] + shared["failures"],
    }
    clock.lap(f"write {DIT_SKIP22}")
    return graphs


def default_host_record() -> Path:
    """The stage-0 host record of today's reference directory (`moss.host_fixture`)."""
    return records.default_reference_dir() / host_fixture.RECORD_NAME


def write_series(host_record: Path) -> dict[str, Any]:
    """Export the three graphs into a staging directory, run the three gates, then move the series
    into place (module docstring). A failed gate discards the staging directory."""
    clock = records.Clock()
    snapshot, module = pipeline_ref.upstream()
    sources.check_snapshot(snapshot, ("transformer",))
    sources.verify_weights(snapshot)
    clock.lap("verify the pinned checkpoint")
    config = load_config(snapshot)
    checkpoint = Checkpoint(snapshot / "transformer")
    summary: dict[str, Any] = {"series": str(SERIES)}
    with staged_publication(SERIES) as staged:
        staged.mkdir(parents=True)
        summary["graphs"] = write_graphs(
            staged,
            module,
            config,
            host_fixture.scheduler_config(snapshot, module),
            checkpoint,
            clock=clock,
        )
        failures = {n: g["failures"] for n, g in summary["graphs"].items() if g["failures"]}
        if failures:
            raise DitExportError(f"gate 1 (structure) failed: {failures}")
        paths = container_paths(staged)
        summary["host_tables"] = host_tables_gate(paths[DIT], host_record)
        if summary["host_tables"]["failures"]:
            failed = summary["host_tables"]["failures"]
            raise DitExportError(f"gate 3 (host tables) failed: {failed}")
        clock.lap("host tables")
        gate = eager_gate(
            snapshot, module, containers={n: paths[n] for n in (DIT, DIT_TEXT)}, clock=clock
        )
        summary["eager"] = gate
        if gate["failures"]:
            raise DitExportError(f"gate 2 (eager equivalence) failed: {gate['failures']}")
    summary["seconds"] = {**clock.phases, "total": clock.total()}
    summary["peak_rss_gib"] = round(records.peak_rss_gib(), 2)
    summary["passed"] = True
    return summary


def eager_only(*, series: bool = True) -> dict[str, Any]:
    """Gate 2 against the written series (its containers' weights included). `series=False` runs
    the comparisons alone, before any series exists (`--without-series` — the container check is
    then reported as not run)."""
    clock = records.Clock()
    snapshot, module = pipeline_ref.upstream()
    containers = None
    if series:
        paths = container_paths(SERIES)
        containers = {name: paths[name] for name in (DIT, DIT_TEXT)}
        missing = [
            str(p) for p in containers.values() if not all(x.is_file() for x in container_parts(p))
        ]
        if missing:
            raise DitExportError(f"no written series ({missing}) — run `write` first")
    gate = eager_gate(snapshot, module, containers=containers, clock=clock)
    if containers is None:
        gate["container_weights"] = "not run (--without-series)"
    return {
        **gate,
        "seconds": {**clock.phases, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
        "passed": not gate["failures"],
    }


COMMANDS: Mapping[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "inspect": lambda _args: inspect_only(),
    "write": lambda args: write_series(args.host_record or default_host_record()),
    "eager": lambda args: eager_only(series=not args.without_series),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--out", type=Path, default=None, help="where the summary goes")
    parser.add_argument(
        "--without-series", action="store_true", help="eager: no container check (no series yet)"
    )
    parser.add_argument(
        "--host-record", type=Path, default=None, help="write: the stage-0 host record (gate 3)"
    )
    args = parser.parse_args(argv)
    if args.without_series and args.command != "eager":
        parser.error("--without-series applies to eager only")
    if args.host_record is not None and args.command != "write":
        parser.error("--host-record applies to write only")
    summary = COMMANDS[args.command](args)
    document = {"command": args.command, "result": summary, "versions": records.versions()}
    text = json.dumps(document, ensure_ascii=False, indent=1)
    print(text)
    out = default_out() if args.out is None else args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.command}.json").write_text(text + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
