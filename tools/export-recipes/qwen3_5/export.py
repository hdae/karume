"""Writes the Qwen3.5 text tower (`last_hidden_state` after the final norm, prefill, symbolic `T`
in [1, 4096], B = 1, no cache) as a Karume IR v2 container, in one storage series.

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' --inexact python -m qwen3_5.export plan
    uv run --with 'transformers==5.14.1' --inexact python -m qwen3_5.export write --storage i8
    uv run --with 'transformers==5.14.1' --inexact python -m qwen3_5.export write \\
        --storage bf16 --layers 4 --out ../../outputs/diag/<dir>      # a truncated check

The graph and its rewrites are {@link qwen3_5.patch}; the checkpoint and its pin are
{@link qwen3_5.checkpoint}. Every subcommand prints per-stage wall time and peak RSS as JSON and
runs under a `MemAvailable` floor ({@link MemoryWatch}): below it the process exits instead of
pushing the machine into swap.

## The export's shape (the umT5 form — `wan.umt5_export`, ADR 0119 decision 6)

Reading the whole checkpoint as F32 and rounding before export would hold 31.7 GB of F32 (7.94 B
parameters). Instead:

1. **The trace runs on a weightless (meta) upstream** ({@link meta_tower} → {@link trace}). The
   export follows shapes only, so the graph is the real-weight export's (pytest checks the JSON).
2. **i8: the linear and embedding weights are read from the checkpoint a block of rows at a time**
   and turned into packed values + scales by the exporter's own i8 procedure, then handed over as
   `fixed_weights` (ADR 0097's entry; {@link quantize_rows}). The per-channel scale closes per row
   (axis 0 for both), so the blocks give the bits a whole-tensor pass gives (pytest checks the
   container bytes against the naive form: F32 + `fake_quant_int8` + `weight_dtype="i8"`).
3. **What is not quantized** — the norms (the zero-centred ones offset by 1 — patch rewrite 4),
   the GDN `conv1d` / `A_log` / `dt_bias` / gated-norm weights, the exporter's own constants — is
   the checkpoint's value widened to F32.

`bf16` and `f32` read every weight as F32 (the writer converts to BF16 per tensor at write time:
BF16 storage of a BF16 checkpoint is the checkpoint's own values — ADR 0128 decision 4's
"original weights"). Their peak RSS is the F32 set and then some: the 4-layer check (7.49 GiB of
F32) peaked at 13.15 GiB while writing (2026-10-31), so the full model (29.6 GiB of F32) would need
about 52 GiB — more than a 64 GB machine shared with other work can promise. The full `bf16`
series needs a fixed-BF16 entry in the writer (BF16 payloads handed over like ADR 0097's fixed
integers, no F32 set) before it can be written here.

## Series

`outputs/series/qwen3.5-9b-text-<source>-<storage>-dyn/text_encoder/model.krm` ({@link
series_dir}): the model is the text tower of Qwen3.5-9B (no `lm_head`), the source is the pinned
checkpoint's label ({@link qwen3_5.checkpoint.SOURCES}), and `-dyn` is the symbolic length. The
container's graph name {@link GRAPH_NAME} is the component name a distribution's `weights` will use
(container-v1 §2.1; `tests/test_graph_names.py`). A truncated export (`--layers`) MUST name its own
`--out` — it never lands in a series.

MUST: transformers is imported inside functions (it is not in the default sync).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.emit import FixedQuantizedWeight
from karume.ir import IrGraph
from karume.pipeline import export_module, publish_model
from karume.quantize import channel_scale, iter_quant_targets, quantize_to_int8
from qwen3_5 import patch
from qwen3_5.checkpoint import DEFAULT_SOURCE, SOURCES, Checkpoint, TextTowerSource

#: The model part of series names (the text tower of Qwen3.5-9B).
SERIES_MODEL = "qwen3.5-9b-text"

#: Storage series this writer produces (`i8` = per-channel i8 linear + embedding; `bf16` = the
#: checkpoint's own values; `f32` = the same values widened — a check form, 2× the bytes).
STORAGES = ("i8", "bf16", "f32")

#: Where a series holds the container, and the container's graph name (= the component name of a
#: distribution's `weights` — a convention, not derived from the directory).
COMPONENT_DIR = "text_encoder"
MODEL_FILE = "model.krm"
GRAPH_NAME = "text_encoder"

#: The trace's example length. `T` is symbolic, so any value works that specialises nothing: not
#: 0 / 1 and none of the model's own sizes (heads 4 / 16 / 32, head dims 128 / 256, rotary 64,
#: conv kernel 4).
TRACE_LENGTH = 28

#: Rows read at once while quantizing (`[rows, 4096]` F32 = 256 MiB).
CHUNK_ROWS = 16_384

#: The `MemAvailable` floor (GiB) below which a run stops itself ({@link MemoryWatch}).
MEMORY_FLOOR_GIB = 6.0

#: Module types whose weight is quantized in the `i8` series (the `fixed_weights` entry takes only
#: linear / embedding weights; the GDN depthwise `conv1d` — 32 K elements a layer — stays F32).
QUANT_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Embedding)

_KIB_PER_GIB = 1 << 20


class Qwen35ExportError(RuntimeError):
    """The checkpoint, the graph or the quantization targets are not what the recipe expects."""


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _meminfo_gib(field_name: str) -> float:
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            key, _, rest = line.partition(":")
            if key == field_name:
                return int(rest.split()[0]) / _KIB_PER_GIB
    raise Qwen35ExportError(f"/proc/meminfo has no {field_name}")


def _status_gib(field_name: str) -> float:
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            key, _, rest = line.partition(":")
            if key == field_name:
                return int(rest.split()[0]) / _KIB_PER_GIB
    raise Qwen35ExportError(f"/proc/self/status has no {field_name}")


@dataclass
class StageRecord:
    """One stage's wall time, true peak RSS (`VmHWM`, reset at the stage's start) and the lowest
    `MemAvailable` sampled (GiB)."""

    name: str
    seconds: float = 0.0
    hwm: float = 0.0
    min_available: float = float("inf")
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": round(self.seconds, 1),
            "hwm_gib": round(self.hwm, 2),
            "min_available_gib": round(self.min_available, 2),
            **self.details,
        }


class MemoryWatch:
    """Samples `MemAvailable` twice a second; below `floor` it prints why and exits the process.

    The machine is shared (another project's training run holds memory too), so running out is
    not this process's failure alone: stopping early is the recoverable outcome, swap or the OOM
    killer is not. `os._exit` from the sampler thread ends the process even while the main thread
    is inside a long torch call.
    """

    def __init__(self, floor: float = MEMORY_FLOOR_GIB, interval: float = 0.5) -> None:
        self.floor = floor
        self.interval = interval
        self.records: list[StageRecord] = []
        self._current: StageRecord | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mem-watch", daemon=True)

    def __enter__(self) -> MemoryWatch:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            available = _meminfo_gib("MemAvailable")
            with self._lock:
                if self._current is not None:
                    self._current.min_available = min(self._current.min_available, available)
                stage = self._current.name if self._current is not None else "-"
            if available < self.floor:
                print(
                    f"[memory] MemAvailable {available:.2f} GiB < floor {self.floor} GiB in stage"
                    f" {stage!r} — exiting",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(70)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        record = StageRecord(name)
        with open("/proc/self/clear_refs", "w", encoding="ascii") as stream:
            stream.write("5")  # reset VmHWM to the current RSS
        with self._lock:
            self._current = record
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.seconds = time.perf_counter() - started
            with self._lock:
                self._current = None
            record.hwm = max(
                _status_gib("VmHWM"),
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _KIB_PER_GIB,
            )
            self.records.append(record)
            print(f"[stage] {json.dumps(record.to_dict())}", flush=True)


@contextmanager
def _unmeasured(name: str) -> Iterator[StageRecord]:
    yield StageRecord(name)


# ---------------------------------------------------------------------------
# Trace (weightless upstream)
# ---------------------------------------------------------------------------


def text_config(directory: Path, layers: int | None = None) -> Any:
    """The checkpoint's `text_config` (a `Qwen3_5Config` checkpoint) or its own config (a text
    model's), set to {@link patch.ATTENTION_NAME}; `layers` keeps the first `layers` layers."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(directory)
    config = getattr(config, "text_config", config)
    if layers is not None:
        if not 1 <= layers <= config.num_hidden_layers:
            raise Qwen35ExportError(f"--layers {layers} outside 1..{config.num_hidden_layers}")
        config.num_hidden_layers = layers
        config.layer_types = list(config.layer_types[:layers])
    patch.register_attention()
    config._attn_implementation = patch.ATTENTION_NAME
    patch.check_supported(config)
    return config


def meta_tower(config: Any) -> patch.Qwen35TextTower:
    """The wrapped upstream `Qwen3_5TextModel` built from `config` on the meta device (eval, f32),
    with the rewrites applied."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    with torch.device("meta"):
        model = Qwen3_5TextModel(config).float()
    return patch.prepare_text_tower(model.eval())


def trace(tower: patch.Qwen35TextTower) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export with the symbolic `T` (the same graph for meta and real weights) and check its
    form ({@link assert_ir_form}).

    The example inputs live on the weights' device: the RoPE rows meet activations directly (no
    weight op in between, unlike `input_ids`), and a CPU row times a meta activation is a device
    mismatch. Their values never matter — the trace follows shapes.
    """
    device = tower.embed_tokens.weight.device
    graph, tensors = export_module(
        tower,
        tuple(value.to(device) for value in patch.example_inputs(TRACE_LENGTH, tower.config)),
        dynamic_shapes=patch.dynamic_shapes(),
        symbol_names=(patch.SYMBOL,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    assert_ir_form(graph, tower)
    return graph, tensors


def _producers(graph: IrGraph) -> dict[str, Any]:
    return {out: node for node in graph.nodes for out in node.outs}


def assert_ir_form(graph: IrGraph, tower: patch.Qwen35TextTower) -> None:
    """The exported graph has the form the rewrites promise; anything else fails loudly.

    - inputs are exactly {@link patch.INPUT_NAMES}, with the one symbol `T`;
    - one `gated_delta_scan` per GDN layer and one `attention` per full-attention layer (no
      decomposed recurrence or attention);
    - attention keeps the k / v heads (GQA, not `repeat_kv`) and takes a mask produced by a
      run-time `where` — no `sym_prefix_slice` anywhere (no Tmax-folded constant);
    - every `rms_norm` takes its weight straight from an initializer (no run-time `1 + w`).
    """
    config = tower.config
    names = [entry.name for entry in graph.inputs]
    if names != list(patch.INPUT_NAMES) or graph.symbols != [patch.SYMBOL]:
        raise Qwen35ExportError(f"graph inputs {names} / symbols {graph.symbols}")
    counts = Counter(node.op for node in graph.nodes)
    expected = {
        "gated_delta_scan": tower.layer_types.count(patch.LINEAR_ATTENTION),
        "attention": tower.layer_types.count(patch.FULL_ATTENTION),
        "sym_prefix_slice": 0,
    }
    found = {op: counts.get(op, 0) for op in expected}
    if found != expected:
        raise Qwen35ExportError(f"op counts {found}, expected {expected}")
    producers = _producers(graph)
    for node in graph.nodes:
        if node.op == "attention":
            if len(node.ins) != 4:
                raise Qwen35ExportError(f"attention with {len(node.ins)} inputs (q, k, v, mask)")
            heads = graph.values[node.ins[1]].shape[1]
            if heads != config.num_key_value_heads:
                raise Qwen35ExportError(f"attention k has {heads} heads (GQA lost)")
            source = producers.get(node.ins[3])
            if source is None or source.op != "where":
                raise Qwen35ExportError(f"attention mask comes from {source and source.op}")
        if node.op == "rms_norm" and node.ins[1] not in graph.initializers:
            raise Qwen35ExportError(f"rms_norm weight '{node.ins[1]}' is computed at run time")


# ---------------------------------------------------------------------------
# Quantization (a block of rows at a time)
# ---------------------------------------------------------------------------


def quantize_rows(
    checkpoint: Checkpoint, key: str, chunk_rows: int = CHUNK_ROWS
) -> FixedQuantizedWeight:
    """Per-row i8 (`[rows, cols]`, axis 0) from blocks of rows (`wan.umt5_export.quantize_rows`'s
    procedure).

    One block takes the naive form's two steps: `fake_quant_int8` replaces the weight by
    `quantize(w, s)·s`, and emit stores `quantize(q·s, s)` after checking the round trip. Storing
    the first `q` directly would split from the naive form wherever `round((q·s)/s) ≠ q`, so the
    second quantization's values are stored, with the same check. No operation mixes rows (the
    scale is the row's amax), so blocks give the whole-tensor bits.
    """
    shape = checkpoint.shape(key)
    if len(shape) != 2:
        raise Qwen35ExportError(f"'{key}' has rank {len(shape)} (per-row i8 is rank 2 only)")
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
            raise Qwen35ExportError(f"'{key}' rows {start}..{stop}: i8 × scale does not round-trip")
        packed[start:stop] = chunk
        scale[start:stop] = chunk_scale
    return FixedQuantizedWeight(dtype="i8", packed=packed, scale=scale)


def quant_targets(tower: patch.Qwen35TextTower) -> dict[str, int]:
    """Tensor key → channel axis of the weights the `i8` series quantizes."""
    return {fqn: axis for fqn, _, axis in iter_quant_targets(tower, op_types=QUANT_TYPES)}


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TowerExport:
    """What {@link write_container} hands to `publish_model`."""

    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    #: The i8 series' packed weights (empty for `bf16` / `f32`).
    fixed: Mapping[str, FixedQuantizedWeight]
    #: `publish_model`'s `weight_dtype` (`f32` with `fixed`, else the storage itself).
    weight_dtype: str
    #: Tensor keys stored as `1 + w` (patch rewrite 4).
    unit_offset_keys: tuple[str, ...]


def checkpoint_value(
    checkpoint: Checkpoint, key: str, unit_offset_keys: frozenset[str]
) -> torch.Tensor:
    """A weight as the graph consumes it: the checkpoint's F32 value, offset by 1 for the
    zero-centred norms (upstream's `1.0 + w.float()`)."""
    value = checkpoint.read(key)
    return patch.unit_offset(value) if key in unit_offset_keys else value


def assert_checkpoint_keys(weights: Sequence[str], available: frozenset[str], layers: int) -> None:
    """Every graph weight is in the checkpoint, and every checkpoint key of the text tower is a
    graph weight — except the layers a truncated export leaves out."""
    missing = sorted(set(weights) - available)
    if missing:
        raise Qwen35ExportError(
            f"{len(missing)} graph weights are not in the checkpoint: {missing[:4]}"
        )
    unused = sorted(
        key
        for key in available - set(weights)
        if not (key.startswith("layers.") and int(key.split(".")[1]) >= layers)
    )
    if unused:
        raise Qwen35ExportError(
            f"{len(unused)} checkpoint keys the graph does not consume: {unused[:4]}"
        )


def prepare(
    checkpoint: Checkpoint,
    config: Any,
    storage: str,
    *,
    monitor: MemoryWatch | None = None,
    chunk_rows: int = CHUNK_ROWS,
) -> TowerExport:
    """Trace → the unquantized weights → the i8 tables (each stage measured by `monitor`)."""
    if storage not in STORAGES:
        raise Qwen35ExportError(f"storage {storage!r} — one of {STORAGES}")
    stage = monitor.stage if monitor is not None else _unmeasured
    with stage("trace") as record:
        tower = meta_tower(config)
        graph, tensors = trace(tower)
        targets = quant_targets(tower) if storage == "i8" else {}
        record.details["nodes"] = len(graph.nodes)
        record.details["ops"] = dict(Counter(node.op for node in graph.nodes).most_common())
    weights = sorted(key for key, value in tensors.items() if value.is_meta)
    assert_checkpoint_keys(weights, checkpoint.names(), config.num_hidden_layers)
    offsets = frozenset(tower.unit_offset_keys)
    if not offsets <= set(weights) or not set(targets) <= set(weights):
        raise Qwen35ExportError("offset norms or quantization targets are not graph weights")
    if offsets & set(targets):
        raise Qwen35ExportError("a quantization target is an offset norm")
    materialized = dict(tensors)
    with stage("plain") as record:
        plain = [key for key in weights if key not in targets]
        for key in plain:
            value = checkpoint_value(checkpoint, key, offsets)
            if list(value.shape) != list(tensors[key].shape):
                raise Qwen35ExportError(
                    f"'{key}': checkpoint shape {list(value.shape)},"
                    f" graph {list(tensors[key].shape)}"
                )
            materialized[key] = value
        record.details["tensors"] = len(plain)
    fixed: dict[str, FixedQuantizedWeight] = {}
    with stage("quantize") as record:
        for key, axis in sorted(targets.items()):
            if axis != 0:
                raise Qwen35ExportError(f"'{key}' has channel axis {axis} (per-row scales only)")
            weight = quantize_rows(checkpoint, key, chunk_rows)
            if list(weight.packed.shape) != list(tensors[key].shape):
                raise Qwen35ExportError(f"'{key}': checkpoint shape differs from the graph")
            fixed[key] = weight
        record.details["tensors"] = len(fixed)
    return TowerExport(
        graph=graph,
        tensors=materialized,
        fixed=fixed,
        weight_dtype="f32" if storage == "i8" else storage,
        unit_offset_keys=tower.unit_offset_keys,
    )


def provenance(source: TextTowerSource) -> Provenance:
    """The container's provenance: the pinned repository's licence identifier and revision."""
    return Provenance(license=source.license, upstream_revision=source.revision)


def write_container(export: TowerExport, path: Path, provenance_: Provenance) -> IrGraph:
    """Write the material to a container (`path` is the representative path of the parts)."""
    return publish_model(
        path,
        export.graph,
        dict(export.tensors),
        provenance=provenance_,
        graph_name=GRAPH_NAME,
        weight_dtype=export.weight_dtype,
        fixed_weights=export.fixed or None,
    )


def series_dir(source: TextTowerSource, storage: str) -> Path:
    """`outputs/series/qwen3.5-9b-text-<source>-<storage>-dyn`."""
    return SERIES_ROOT / f"{SERIES_MODEL}-{source.name}-{storage}-dyn"


def storage_counts(graph: IrGraph) -> dict[str, int]:
    """Initializers per storage dtype (`graph` with its storage committed)."""
    return dict(sorted(Counter(init.storage.dtype for init in graph.initializers.values()).items()))


def write(
    source: TextTowerSource,
    storage: str,
    *,
    layers: int | None = None,
    out: Path | None = None,
    model_dir: Path | None = None,
    floor: float = MEMORY_FLOOR_GIB,
) -> dict[str, Any]:
    """Write one container (the series' unless `out` is given) through a staged seat.

    MUST: a truncated export (`layers`) names its own `out` — a series holds the whole model.
    """
    if layers is not None and out is None:
        raise Qwen35ExportError("--layers needs --out (a truncated model is not a series)")
    target = out if out is not None else series_dir(source, storage) / COMPONENT_DIR
    checkpoint = Checkpoint.pinned(source, model_dir)
    config = text_config(checkpoint.directory, layers)
    with MemoryWatch(floor) as monitor:
        export = prepare(checkpoint, config, storage, monitor=monitor)
        with monitor.stage("write"):
            target.parent.mkdir(parents=True, exist_ok=True)
            with staged_publication(target) as staged:
                staged.mkdir()
                graph = write_container(export, staged / MODEL_FILE, provenance(source))
    parts = container_parts(target / MODEL_FILE)
    return {
        "container": str(target / MODEL_FILE),
        "storage": storage,
        "layers": config.num_hidden_layers,
        "parts": len(parts),
        "bytes": sum(part.stat().st_size for part in parts),
        "storage_counts": storage_counts(graph),
        "unit_offset_norms": len(export.unit_offset_keys),
        "stages": [record.to_dict() for record in monitor.records],
    }


def plan(source: TextTowerSource, *, layers: int | None, model_dir: Path | None) -> dict[str, Any]:
    """Trace only (no weight is read): the graph's op counts and inputs."""
    checkpoint = Checkpoint.pinned(source, model_dir)
    config = text_config(checkpoint.directory, layers)
    tower = meta_tower(config)
    graph, tensors = trace(tower)
    weights = sorted(key for key, value in tensors.items() if value.is_meta)
    assert_checkpoint_keys(weights, checkpoint.names(), config.num_hidden_layers)
    return {
        "layers": config.num_hidden_layers,
        "inputs": [[entry.name, entry.dtype, entry.shape] for entry in graph.inputs],
        "nodes": len(graph.nodes),
        "ops": dict(Counter(node.op for node in graph.nodes).most_common()),
        "weights": len(weights),
        "i8_targets": len(quant_targets(tower)),
        "unit_offset_norms": len(tower.unit_offset_keys),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qwen3_5.export", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "write"):
        command = commands.add_parser(name)
        command.add_argument("--source", choices=sorted(SOURCES), default=DEFAULT_SOURCE)
        command.add_argument("--layers", type=int, default=None, help="keep the first N layers")
        command.add_argument("--model-dir", type=Path, default=None, help="checkpoint directory")
        if name == "write":
            command.add_argument("--storage", choices=STORAGES, default="i8")
            command.add_argument("--out", type=Path, default=None, help="component directory")
            command.add_argument("--memory-floor", type=float, default=MEMORY_FLOOR_GIB)
    args = parser.parse_args(argv)
    source = SOURCES[args.source]
    if args.command == "plan":
        result = plan(source, layers=args.layers, model_dir=args.model_dir)
    else:
        result = write(
            source,
            args.storage,
            layers=args.layers,
            out=args.out,
            model_dir=args.model_dir,
            floor=args.memory_floor,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
