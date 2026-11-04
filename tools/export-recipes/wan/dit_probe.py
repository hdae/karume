"""Measure host RAM of the Wan2.2 TI2V-5B DiT export and reference (ADR 0121 step 0, decision 7).

    uv run --group wan --inexact python -m wan.dit_probe prepare --trace-only  # no weights needed
    uv run --group wan --inexact python -m wan.dit_probe prepare     # up to the i8 export material
    uv run --group wan --inexact python -m wan.dit_probe eager       # the eager-equivalence check
    uv run --group wan --inexact python -m wan.dit_probe reference   # layer-by-layer, S = 192
    uv run --group wan --inexact python -m wan.dit_probe reference --latent 21,30,52  # 8,190
    uv run --group wan --inexact python -m wan.dit_probe reference --latent 21,30,52 \
        --text boxing-cats --timestep 500 --growth   # isolating the f32 vs f64 difference (step 1)

Every subcommand prints the wall time and RSS per stage (`wan.umt5_export.MemoryMonitor` — samples
of `/proc` every second and VmHWM per stage) as JSON, and writes the same to `<command>.json` under
`--out` (default {@link default_out} — `outputs/bench/<distribution name>/<date>_dit-probe/`).
Nothing is written to the series (`outputs/series/`). `reference` compares with the `eager` result
in the same place, so pass the same `--out` to both when they run on different days.

The DiT patch of `prepare` / `eager` stays in step 0's form, the same as Wan2.1 (one time input —
`dit_patch.WanDitTokens`), and the 5B configuration passes the check of that premise (`image_dim` /
`added_kv_proj_dim` are None). The product's I2V-capable graph (2 time inputs, a condition mask —
ADR 0121 decision 3) is written by step 1's `wan.ti2v_export_dit`, which passes its own trace
(`tracer=`) to `prepare`. That graph only adds 3 linear nodes (310 in step 1's IR inspection), so
the shape of the RAM can be read from the values measured here.

## prepare (meta trace + i8 per chunk of rows + `fixed_weights` — the form of ADR 0119 step 10b)

1. **trace**: wrap a meta upstream without weights (config only) in the S-shaped wrapper and export
   it with meta example inputs (S = 192). The upper bound of the S symbol is {@link SYM_MAX}
   (1280×704×121 — ADR 0121 decision 3).
2. **plain**: read the weights that are not quantized (bias, norm, `scale_shift_table`) from the
   checkpoint as F32.
3. **quantize**: read the 307 linears (including the patch embedding's Linear) from the checkpoint
   one chunk of rows at a time, and turn them into packed i8 + per-row scales with the same
   procedure as the exporter (`umt5_export.quantize_rows` — the patch embedding is a Conv3d
   `[3072,48,1,2,2]` upstream, so it is read in 2-D with the output channels as rows).
4. **store**: commit the storage declaration with `karume.emit.stored_model(..., fixed_weights=...)`
   and apply the checks `publish_model` applies before writing (`assert_runtime_support` /
   `assert_op_contracts`). No container file is written — the writing entry point (`publish_model`
   stating `graph_name=`) is step 1's job, which adds a row to the cross-family check (the table in
   `tests/test_graph_names.py`).

## eager (the RAM of the eager-equivalence check — ADR 0121 decision 4's "full f32 18.63 GiB + small
activations")

Read the full upstream in f32 with `from_pretrained`, apply the i8 fake-quant through the S-shaped
wrapper, run the plain upstream forward (the reference) and the patched eager on one S = 192 case,
and apply the `export_dit.eager_failures` check (the same function as `wan.export_dit`). In
addition, confirm that the f32 of the layer-by-layer reference (`wan.dit_reference` — the same
in-memory weights) is bit-identical to the upstream f32, and take the layer-by-layer f64 too.

The RAM peak comes not in the fake-quant stage but in the layer-by-layer stage: while the full f32
is held, the layer-by-layer f64 adds the parts outside the blocks and one block in f64
({@link eager_memory} — for 5B, 18.63 + 0.67 + 1.22 GiB plus the base amount
{@link EAGER_BASE_BYTES}, about 21.1 GiB; an estimate read from the 1.3B measurement (peak
6.76 GiB = f32 5.29 + f64 outside the blocks and 2 blocks 0.88 + 0.59) as the value after moving to
one block at a time). Before the `load` and `layerwise` stages the script reads MemAvailable from
`/proc/meminfo`, and if it is less than the remaining estimate plus the headroom
{@link MEMORY_HEADROOM_BYTES}, it stops and states the amount needed (on a machine where a GPU job
shares host RAM, so that hours are not lost by falling into swap midway).

## reference (the RAM and time of the layer-by-layer reference — from the checkpoint, since there is
no container yet)

Take the layer-by-layer f32 / f64 with a reader that loads tensors from the checkpoint one at a time
and applies the i8 fake-quant (`dit_reference.CheckpointDitWeights`). The f32 runs at the same
misaligned addresses as upstream's mmap (`dit_reference.upstream_alignment`). If the `eager` result
is in `--out`, the f32 is compared bitwise with the upstream f32 reference and the f64 with eager's
layer-by-layer f64 (only for the same case).

### Isolating the CPU f32 vs f64 difference (ADR 0121 step 0's open investigation → step 1)

With synthetic inputs at S = 8,190 the ratio between the f32 and f64 outputs was 1.2e-1 (5.3e-4 at
S = 192). The options for isolating the cause:

- `--text <name>`: replace the text input, synthetic random `[valid length, 4096]`, with one of
  Wan2.1's precomputed embedding assets (the output of the same umT5 — `wan.text_embeds`). The
  latent is the same random draw of the same seed (the latent is drawn first), so only the text
  differs.
- `--timestep <t>`: a t other than 999.
- `--growth`: the per-block difference between f32 and f64 (largest absolute difference, ratio,
  relRMS, the token of the largest difference). The f32 run writes the block outputs one by one
  under `--out` (100 MB each at S = 8,190), and the f64 run reads them one by one, compares and
  deletes them (both are never held in RAM). Rows are appended to `growth.<case>.jsonl`, one row per
  block (blocks already done stay readable if the run stops midway).
- `--blocks <N>`: run only the first N blocks (f64 at S = 8,190 takes about 30 s per block — to see
  the layer where the growth starts within a time budget that cannot cover all 30 layers). The
  output is the value after the N-th block with the head applied (not a reference).
- `--case F,H,W:t:text[:N]` (repeatable): run the combinations of the 4 above in sequence in one
  execution (the weight reader and the process start-up happen once — the summary is
  `reference.<case>.json` per case).

MUST: import diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.export import Dim

from _shared.paths import BENCH_ROOT, SERIES_ROOT
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.emit import FixedQuantizedWeight, storage_breakdown, stored_model
from karume.ir import IrGraph
from karume.pipeline import export_module
from karume.quantize import QUANT_MODULE_TYPES, iter_quant_targets
from karume.verify import assert_op_contracts, assert_runtime_support
from wan import dit_patch, dit_reference, export_dit
from wan.pipeline_ref import pad_text_embeds
from wan.sources import SOURCES, local_snapshot
from wan.umt5_export import CHUNK_ROWS, Checkpoint, MemoryMonitor, quantize_rows

#: The default model (Wan2.2 TI2V-5B — ADR 0121).
DEFAULT_MODEL = "ti2v-5b"

#: Model → the distribution name of the default output location (`outputs/bench/<distribution
#: name>/` — `docs/assets-layout.md`).
BENCH_NAMES: dict[str, str] = {"t2v-1.3b": "karume-wan2.1", "ti2v-5b": "karume-wan2.2"}

#: The base amount added to the estimate of the eager peak (the process base, activations, the
#: fake-quant temporaries — from the 1.3B measurement, the 0.59 GiB left after subtracting the full
#: f32 and the layer-by-layer f64 from the peak, rounded).
EAGER_BASE_BYTES = int(0.6 * 2**30)

#: The headroom added to the estimate in the MemAvailable check (fluctuation of other processes,
#: error of the estimate).
MEMORY_HEADROOM_BYTES = 1 * 2**30

#: The upper bound of `Dim("S")` (1280×704×121 = 27,280 — ADR 0121 decision 3) and the lower bound
#: (the same as Wan2.1).
SYM_MAX = 27_280
SYM_MIN = export_dit.DIT_SYM_MIN

#: The case to measure (the head of `export_dit.CASES` = the band-deciding S = 192, t = 999).
PROBE_CASE = export_dit.CASES[0]

#: The spelling of the eager / reference output files (`<command>.<case>.safetensors`).
EAGER_PREFIX = "eager."
REFERENCE_PREFIX = "reference."
#: The output tensor keys (the upstream f32 reference, the layer-by-layer f32 / f64 — the f64 is
#: kept as F64 without rounding to f32).
UPSTREAM_KEY = "upstream.f32"
LAYERWISE_F32_KEY = "layerwise.f32"
LAYERWISE_F64_KEY = "layerwise.f64"


class DitProbeError(RuntimeError):
    """The checkpoint, the graph or the quantization targets differ from the assumptions, or host
    RAM is insufficient."""


def default_out(model: str) -> Path:
    """The default location of the outputs (the summary JSON and the eager / reference output
    tensors)."""
    return BENCH_ROOT / BENCH_NAMES[model] / f"{date.today().isoformat()}_dit-probe"


_GIB = 2**30


def eager_memory(config: Mapping[str, Any]) -> dict[str, int]:
    """The RAM estimate per eager stage (bytes — the module doc "eager").

    `f32`: the full upstream f32 (held from `load` on). `layerwise_f64`: the parts outside the
    blocks and one block that the layer-by-layer f64 adds. `peak`: both plus the base amount.
    """
    writer = dit_reference.LayerwiseDit(config)
    outer = sum(math.prod(shape) for shape in writer.outer_shapes.values())
    block = sum(math.prod(shape) for shape in writer.block_shapes.values())
    f32 = (outer + writer.layers * block) * 4
    layerwise_f64 = (outer + block) * 8
    return {
        "f32": f32,
        "layerwise_f64": layerwise_f64,
        "peak": f32 + layerwise_f64 + EAGER_BASE_BYTES,
    }


def mem_available() -> int:
    """MemAvailable from `/proc/meminfo` (bytes)."""
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, _, value = line.partition(":")
        if name == "MemAvailable":
            return int(value.split()[0]) * 1024
    raise DitProbeError("/proc/meminfo has no MemAvailable")


def require_available(stage: str, need: int) -> None:
    """Stop and state the amount needed if MemAvailable is less than `need` + headroom (call it
    before a heavy stage)."""
    available = mem_available()
    wanted = need + MEMORY_HEADROOM_BYTES
    if available < wanted:
        raise DitProbeError(
            f"MemAvailable before {stage} is {available / _GIB:.2f} GiB, less than the required"
            f" {wanted / _GIB:.2f} GiB (estimate {need / _GIB:.2f} + headroom"
            f" {MEMORY_HEADROOM_BYTES / _GIB:.2f}) — run it after other heavy work has finished"
        )


def transformer_dir(model: str) -> Path:
    """Where the DiT of the pinned revision lives (`local_snapshot` fails loudly if it has not been
    fetched)."""
    return local_snapshot(model) / dit_reference.TRANSFORMER_SUBFOLDER


def meta_wrapper(config: Mapping[str, Any]) -> dit_patch.WanDitTokens:
    """Wrap a meta upstream built from the config alone in the S-shaped wrapper (no weights are
    read)."""
    return dit_patch.WanDitTokens(dit_reference.meta_model(config))


def meta_inputs(config: Mapping[str, Any], spec: export_dit.CaseSpec) -> tuple[torch.Tensor, ...]:
    """The example inputs of the trace (meta — values do not affect the trace; the weights are meta,
    so the inputs are meta too)."""
    patch = [int(size) for size in config["patch_size"]]
    frames, height, width = spec.latent_shape
    tokens = (frames // patch[0]) * (height // patch[1]) * (width // patch[2])
    width_in = int(config["in_channels"]) * math.prod(patch)
    head_dim = int(config["attention_head_dim"])
    with torch.device("meta"):
        return (
            torch.empty(1, tokens, width_in),
            torch.empty(1, int(config["freq_dim"])),
            torch.empty(1, 512, int(config["text_dim"])),
            torch.empty(1, tokens, 1, head_dim),
            torch.empty(1, tokens, 1, head_dim),
        )


def dynamic_shapes(sym_max: int = SYM_MAX) -> tuple[Any, ...]:
    """The same form as `export_dit.dynamic_shapes`, with only the upper bound raised to 5B's
    accepted maximum."""
    tokens = Dim("S", min=SYM_MIN, max=sym_max)
    return ({1: tokens}, None, None, {1: tokens}, {1: tokens})


def trace(
    config: Mapping[str, Any], sym_max: int = SYM_MAX
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, int]]:
    """Run the S-shaped export on the meta upstream (graph, stored tensors (weights are meta),
    quantization targets)."""
    wrapper = meta_wrapper(config)
    graph, tensors = export_module(
        wrapper,
        meta_inputs(config, PROBE_CASE),
        dynamic_shapes=dynamic_shapes(sym_max),
        symbol_names=("S",),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    declared = [entry.name for entry in graph.inputs]
    if declared != list(export_dit.INPUT_NAMES):
        raise DitProbeError(f"the graph input names do not match the declaration: {declared}")
    targets = {
        fqn: axis for fqn, _, axis in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES)
    }
    return graph, tensors, targets


class _FlatRows:
    """A reader that presents the checkpoint's weights of rank 3 or more (the patch embedding's
    Conv3d) in 2-D with the output channels as rows (`quantize_rows` takes rank 2 only). The
    per-row scales are the same as for the Linear-form weight (the row amax)."""

    def __init__(self, checkpoint: Checkpoint) -> None:
        self._checkpoint = checkpoint

    def shape(self, key: str) -> list[int]:
        shape = self._checkpoint.shape(key)
        return [shape[0], math.prod(shape[1:])]

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        rows = self._checkpoint.read_rows(key, start, stop)
        return rows.reshape(rows.shape[0], -1)


@dataclass(frozen=True)
class DitExport:
    """The container material (in a form that can be passed to `publish_model` as is — quantization
    targets are meta, the real values in `fixed`)."""

    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    fixed: Mapping[str, FixedQuantizedWeight]
    plain: tuple[str, ...]


#: The trace entry point (config, upper bound of S → graph, stored tensors, quantization targets).
Tracer = Callable[[Mapping[str, Any], int], tuple[IrGraph, dict[str, torch.Tensor], dict[str, int]]]


def prepare(
    directory: Path,
    *,
    stage: Any,
    sym_max: int = SYM_MAX,
    chunk_rows: int = CHUNK_ROWS,
    trace_only: bool = False,
    tracer: Tracer | None = None,
) -> tuple[DitExport, dict[str, Any]]:
    """trace → plain → quantize (up to trace with `trace_only`). Returns the material and the
    summary fields.

    `tracer` replaces the trace (the default is the 2.1-form {@link trace}; the I2V-capable graph
    uses `wan.ti2v_export_dit.trace` — the rest of the material building (plain / quantize) goes
    through the same single path).
    """
    config = dit_reference.load_config(directory)
    with stage("trace") as record:
        graph, tensors, targets = (trace if tracer is None else tracer)(config, sym_max)
        record.details["nodes"] = len(graph.nodes)
    weights = sorted(key for key, value in tensors.items() if value.is_meta)
    unknown = sorted(set(targets) - set(weights))
    if unknown or any(axis != 0 for axis in targets.values()):
        raise DitProbeError(
            f"quantization targets missing from the graph's weights / not on axis 0: {unknown}"
        )
    summary: dict[str, Any] = {
        "nodes": len(graph.nodes),
        "ops": dict(sorted(Counter(node.op for node in graph.nodes).items())),
        "symbols": list(graph.symbols),
        "inputs": [[entry.name, list(entry.shape), entry.dtype] for entry in graph.inputs],
        "outputs": [list(graph.values[name].shape) for name in graph.outputs],
        "weights": len(weights),
        "quant_targets": len(targets),
    }
    plain = tuple(key for key in weights if key not in targets)
    if trace_only:
        return DitExport(graph=graph, tensors=tensors, fixed={}, plain=plain), summary
    checkpoint = Checkpoint(
        directory, index=dit_reference.DIFFUSERS_INDEX, single=dit_reference.DIFFUSERS_SINGLE
    )
    missing = sorted(set(weights) - checkpoint.names())
    if missing:
        raise DitProbeError(f"graph weights missing from the checkpoint: {missing[:5]}")
    materialized = dict(tensors)
    with stage("plain") as record:
        for key in plain:
            value = checkpoint.read(key)
            if list(value.shape) != list(tensors[key].shape):
                raise DitProbeError(
                    f"'{key}': the checkpoint shape {list(value.shape)} differs from the graph"
                )
            materialized[key] = value
        record.details["tensors"] = len(plain)
    fixed: dict[str, FixedQuantizedWeight] = {}
    rows_view = _FlatRows(checkpoint)
    with stage("quantize") as record:
        for key in sorted(targets):
            weight = quantize_rows(rows_view, key, chunk_rows)  # type: ignore[arg-type]
            if list(weight.packed.shape) != list(tensors[key].shape):
                raise DitProbeError(f"'{key}': the checkpoint shape differs from the graph")
            fixed[key] = weight
        record.details["tensors"] = len(fixed)
    summary["packed_bytes"] = sum(weight.packed.numel() for weight in fixed.values())
    return DitExport(graph=graph, tensors=materialized, fixed=fixed, plain=plain), summary


def store(export: DitExport) -> dict[str, Any]:
    """Commit the storage declaration and apply the pre-write checks (no container is written)."""
    stored = stored_model(export.graph, dict(export.tensors), fixed_weights=export.fixed).graph
    assert_runtime_support(stored)
    assert_op_contracts(stored)
    breakdown = storage_breakdown(stored)
    return {
        "storage": dict(
            sorted(Counter(item.storage.dtype for item in stored.initializers.values()).items())
        ),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
    }


def run_prepare(model: str, *, trace_only: bool, sym_max: int) -> dict[str, Any]:
    with MemoryMonitor() as monitor:
        export, summary = prepare(
            transformer_dir(model), stage=monitor.stage, sym_max=sym_max, trace_only=trace_only
        )
        if not trace_only:
            with monitor.stage("store") as record:
                summary.update(store(export))
                record.details["initializers"] = len(export.graph.initializers)
    return {**summary, "stages": [record.to_dict() for record in monitor.records]}


def _probe_spec(
    latent: tuple[int, int, int] | None, timestep: int | None = None
) -> export_dit.CaseSpec:
    """The case to measure (default {@link PROBE_CASE}; passing `latent` / `timestep` gives another
    grid or another time with the same valid length / seed)."""
    if latent is None and timestep is None:
        return PROBE_CASE
    return export_dit.CaseSpec(
        "probe",
        PROBE_CASE.latent_shape if latent is None else latent,
        PROBE_CASE.timestep if timestep is None else timestep,
        PROBE_CASE.text_length,
        PROBE_CASE.seed,
    )


def _case_name(spec: export_dit.CaseSpec, config: Mapping[str, Any]) -> str:
    patch_t, patch_h, patch_w = (int(size) for size in config["patch_size"])
    return spec.name((patch_t, patch_h, patch_w))


def run_eager(model: str, out: Path) -> dict[str, Any]:
    """The RAM of the eager-equivalence check and the layer-by-layer f32 / f64 (the same in-memory
    weights)."""
    spec = PROBE_CASE
    estimate = eager_memory(dit_reference.load_config(transformer_dir(model)))
    require_available("load", estimate["peak"])
    with MemoryMonitor() as monitor:
        with monitor.stage("load") as record:
            upstream = export_dit.load_transformer(model)
            record.details["parameters"] = sum(p.numel() for p in upstream.parameters())
        with monitor.stage("fake-quant") as record:
            wrapper = dit_patch.WanDitTokens(upstream)
            rounded, _ = export_dit.fake_quant("i8", upstream, wrapper)
            record.details["rounded"] = rounded
        with monitor.stage("reference"):
            case = export_dit.build_case(upstream, spec)
        with monitor.stage("eager") as record:
            report, _ = export_dit.eager_report(wrapper, upstream, case)
            failures = export_dit.eager_failures(report)
            record.details["failures"] = len(failures)
        require_available("layerwise", estimate["layerwise_f64"])
        with monitor.stage("layerwise") as record:
            writer = dit_reference.LayerwiseDit(upstream.config)
            source = dit_reference.ModuleWeights(upstream)
            latents, timestep, text = export_dit.case_inputs(upstream, spec)
            f32 = writer.forward(source, torch.float32, latents, timestep, text)
            f64 = writer.forward(source, torch.float64, latents, timestep, text)
            record.details["f32_seconds"] = round(f32.seconds, 1)
            record.details["f64_seconds"] = round(f64.seconds, 1)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {
        UPSTREAM_KEY: case.reference.contiguous(),
        LAYERWISE_F32_KEY: f32.output.contiguous(),
        LAYERWISE_F64_KEY: f64.output.contiguous(),
    }
    save_file(tensors, str(out / f"{EAGER_PREFIX}{case.name}.safetensors"))
    return {
        "case": case.name,
        "eager": report,
        "eager_failures": failures,
        "layerwise_f32_equals_upstream": bool(torch.equal(f32.output, case.reference)),
        "upstream_f32_vs_layerwise_f64": export_dit._ratio(case.reference, f64.output),
        "memory_estimate_gib": {key: round(value / _GIB, 2) for key, value in estimate.items()},
        "stages": [record.to_dict() for record in monitor.records],
    }


#: The default text input (synthetic random — `export_dit.case_inputs`). Any other value is a tensor
#: name in the embeddings asset.
SYNTHETIC_TEXT = "synthetic"

#: Wan2.1's precomputed embeddings asset (the output of the same umT5 — the place of
#: `wan.text_embeds`).
TEXT_EMBEDS_ASSET = SERIES_ROOT / "wan2.1-t2v-1.3b-text-embeds" / "text_embeds.safetensors"

#: The spelling of the per-block difference row file (`growth.<case>.jsonl`) and the temporary place
#: of the f32 block outputs.
GROWTH_PREFIX = "growth."
GROWTH_SPILL = "growth-spill"


def probe_inputs(
    model: Any, spec: export_dit.CaseSpec, text: str = SYNTHETIC_TEXT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The probe's inputs (latent, timestep, context `[1,512,4096]`).

    If `text` is synthetic, exactly `export_dit.case_inputs`. If it is a name in the embeddings
    asset, the latent is the same random draw of the same seed (`case_inputs` draws the latent
    first) and only the context becomes that embedding — only the text differs.
    """
    latents, timestep, synthetic = export_dit.case_inputs(model, spec)
    if text == SYNTHETIC_TEXT:
        return latents, timestep, synthetic
    from wan.text_embeds import read_asset

    tensors, _ = read_asset(TEXT_EMBEDS_ASSET)
    if text not in tensors:
        raise DitProbeError(f"the embeddings asset has no '{text}' ({sorted(tensors)})")
    return latents, timestep, pad_text_embeds(tensors[text])


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """A summary of the f32 vs f64 difference (ratio = largest absolute difference ÷ the reference's
    largest absolute value, relRMS, the token of the largest difference).

    `expected` is f64. The token is on the S axis of `[1,S,dim]` (for the output `[1,C,F,H,W]`, the
    flat position).
    """
    reference = expected.double()
    difference = (actual.double() - reference).abs()
    worst = int(difference.flatten().argmax())
    token = worst // int(reference.shape[-1]) if reference.dim() == 3 else worst
    return {
        "max_abs_ref": float(reference.abs().max()),
        "rms_ref": float(reference.square().mean().sqrt()),
        "max_abs_diff": float(difference.max()),
        "ratio": float(difference.max() / reference.abs().max()),
        "rel_rms": float(difference.square().mean().sqrt() / reference.square().mean().sqrt()),
        "worst_position": token,
    }


class _GrowthRecorder:
    """The per-block difference between f32 and f64 (the module doc's "Isolating").

    In the f32 run, {@link spill} writes the block outputs one by one; in the f64 run,
    {@link compare} reads them one by one, compares, deletes them and appends a row to `rows_path`.
    """

    def __init__(self, spill: Path, rows_path: Path) -> None:
        self.spill_dir = spill
        self.rows_path = rows_path
        self.rows: list[dict[str, Any]] = []
        spill.mkdir(parents=True, exist_ok=True)
        rows_path.write_text("", encoding="utf-8")

    def _path(self, index: int) -> Path:
        return self.spill_dir / f"f32.block.{index:02d}.safetensors"

    def spill(self, index: int, output: torch.Tensor) -> None:
        save_file({"h": output.contiguous()}, str(self._path(index)))

    def compare(self, index: int, output: torch.Tensor) -> None:
        path = self._path(index)
        with safe_open(str(path), framework="pt") as handle:
            mine = handle.get_tensor("h")
        row = {"block": index, **_difference(mine, output)}
        path.unlink()
        self.rows.append(row)
        with self.rows_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def close(self) -> None:
        """Delete the temporary place after the comparison (the case directory and the parent once
        it is empty)."""
        for directory in (self.spill_dir, self.spill_dir.parent):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()


def run_reference(
    model: str,
    out: Path,
    latent: tuple[int, int, int] | None,
    *,
    timestep: int | None = None,
    text: str = SYNTHETIC_TEXT,
    growth: bool = False,
    blocks: int | None = None,
) -> dict[str, Any]:
    """The RAM and time of the layer-by-layer f32 / f64 read from the checkpoint one tensor at a
    time with the i8 fake-quant applied (`text` / `timestep` / `growth` / `blocks` are the
    isolation options — the module doc)."""
    directory = transformer_dir(model)
    config = dit_reference.load_config(directory)
    spec = _probe_spec(latent, timestep)
    name = _case_name(spec, config)
    if text != SYNTHETIC_TEXT:
        name = f"{name}-{text}"
    if blocks is not None:
        name = f"{name}-b{blocks:02d}"
    out.mkdir(parents=True, exist_ok=True)
    recorder = (
        _GrowthRecorder(out / GROWTH_SPILL / name, out / f"{GROWTH_PREFIX}{name}.jsonl")
        if growth
        else None
    )
    with MemoryMonitor() as monitor:
        with monitor.stage("setup") as record:
            writer = dit_reference.LayerwiseDit(
                config, alignment=dit_reference.upstream_alignment(directory)
            )
            source = dit_reference.CheckpointDitWeights(
                directory, "i8", dit_reference.quant_keys(config)
            )
            latents, timesteps, embeds = probe_inputs(writer, spec, text)
            record.details["tokens"] = int(latents[0, 0].numel() // math.prod(writer.patch_size))
        results: dict[str, dit_reference.DitReference] = {}
        for label, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            on_output = None
            if recorder is not None:
                on_output = recorder.spill if label == "f32" else recorder.compare
            with monitor.stage(f"layerwise-{label}") as record:
                results[label] = writer.forward(
                    source,
                    dtype,
                    latents,
                    timesteps,
                    embeds,
                    on_block=_progress(name, label, writer.layers if blocks is None else blocks),
                    on_output=on_output,
                    blocks=blocks,
                )
                record.details["seconds"] = round(results[label].seconds, 1)
    save_file(
        {
            LAYERWISE_F32_KEY: results["f32"].output.contiguous(),
            LAYERWISE_F64_KEY: results["f64"].output.contiguous(),
        },
        str(out / f"{REFERENCE_PREFIX}{name}.safetensors"),
    )
    summary: dict[str, Any] = {
        "case": name,
        "latent": list(spec.latent_shape),
        "timestep": spec.timestep,
        "text": text,
        "blocks": writer.layers if blocks is None else blocks,
        "text_rows": int((embeds[0].abs().sum(dim=-1) != 0).sum()),
        "f32_vs_f64": export_dit._ratio(results["f32"].output, results["f64"].output),
        "output": _difference(results["f32"].output, results["f64"].output),
        "stages": [record.to_dict() for record in monitor.records],
    }
    if recorder is not None:
        recorder.close()
        summary["growth"] = recorder.rows
    eager = out / f"{EAGER_PREFIX}{name}.safetensors"
    # Show in the summary whether the counterpart to compare with exists (without it there is no
    # comparison field — making "not compared" visible).
    summary["eager_result"] = str(eager) if eager.is_file() else None
    if eager.is_file():
        with safe_open(str(eager), framework="pt") as handle:
            summary["f32_equals_eager_upstream"] = bool(
                torch.equal(results["f32"].output, handle.get_tensor(UPSTREAM_KEY))
            )
            summary["f64_equals_eager_layerwise"] = bool(
                torch.equal(results["f64"].output, handle.get_tensor(LAYERWISE_F64_KEY))
            )
    return summary


def _progress(name: str, label: str, layers: int) -> Any:
    def report(index: int, seconds: float) -> None:
        print(f"[block] {name} {label} {index + 1}/{layers} {seconds:.1f} s", flush=True)

    return report


def _latent(text: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in text.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--latent is F,H,W (the latent grid — e.g. 21,30,52)")
    return parts[0], parts[1], parts[2]


@dataclass(frozen=True)
class ProbeRun:
    """One `reference` case (one `--case` — grid, time, text, number of blocks to run)."""

    latent: tuple[int, int, int] | None
    timestep: int | None
    text: str
    blocks: int | None


def _probe_run(text: str) -> ProbeRun:
    """`--case F,H,W:t:text[:N]` (e.g. `4,30,52:999:boxing-cats`, `9,44,80:999:synthetic:6`)."""
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("--case is F,H,W:timestep:text[:blocks]")
    return ProbeRun(
        latent=_latent(parts[0]),
        timestep=int(parts[1]),
        text=parts[2],
        blocks=int(parts[3]) if len(parts) == 4 else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("prepare", "eager", "reference"))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument(
        "--out",
        type=Path,
        help="output location (default outputs/bench/<distribution name>/<date>_dit-probe)",
    )
    parser.add_argument("--trace-only", action="store_true", help="prepare: trace only")
    parser.add_argument(
        "--sym-max", type=int, default=SYM_MAX, help="prepare: upper bound of the S symbol"
    )
    parser.add_argument("--latent", type=_latent, help="reference: latent grid F,H,W")
    parser.add_argument("--timestep", type=int, help="reference: timestep (default 999)")
    parser.add_argument(
        "--text",
        default=SYNTHETIC_TEXT,
        help="reference: the text input (synthetic = synthetic random, otherwise a prompt name"
        " in the embeddings asset)",
    )
    parser.add_argument(
        "--growth",
        action="store_true",
        help="reference: record the per-block difference between f32 and f64",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        help="reference: run only the first N blocks (truncation for the error growth)",
    )
    parser.add_argument(
        "--case",
        type=_probe_run,
        action="append",
        default=[],
        help="reference: run F,H,W:timestep:text[:blocks] in sequence in one execution"
        " (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.trace_only and args.command != "prepare":
        parser.error("--trace-only applies only to prepare")
    reference_only = (
        args.latent is not None
        or args.timestep is not None
        or args.text != SYNTHETIC_TEXT
        or args.growth
        or args.blocks is not None
    )
    if (reference_only or args.case) and args.command != "reference":
        parser.error(
            "--latent / --timestep / --text / --growth / --blocks / --case apply only to reference"
        )
    if args.case and _single_flags(args):
        parser.error("--case is not combined with --latent / --timestep / --text / --blocks")
    if args.out is None:
        args.out = default_out(args.model)
    if args.command == "reference":
        runs = args.case or [ProbeRun(args.latent, args.timestep, args.text, args.blocks)]
        failed = False
        for run in runs:
            failed = _write_document(args, _reference(args, run)) or failed
        return 1 if failed else 0
    return 1 if _write_document(args, _summary(args)) else 0


def _single_flags(args: argparse.Namespace) -> bool:
    """Whether any single-case option (`--latent` / `--timestep` / `--text` / `--blocks`) is
    given."""
    return (
        args.latent is not None
        or args.timestep is not None
        or args.text != SYNTHETIC_TEXT
        or args.blocks is not None
    )


def _reference(args: argparse.Namespace, run: ProbeRun) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    summary = run_reference(
        args.model,
        args.out,
        run.latent,
        timestep=run.timestep,
        text=run.text,
        growth=args.growth,
        blocks=run.blocks,
    )
    return summary, started


def _summary(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if args.command == "prepare":
        return run_prepare(args.model, trace_only=args.trace_only, sym_max=args.sym_max), started
    return run_eager(args.model, args.out), started


def _write_document(args: argparse.Namespace, result: tuple[dict[str, Any], float]) -> bool:
    """Write the summary to `<command>.json` under `--out` (per case for reference) and return
    whether it failed."""
    summary, started = result
    document = {
        "command": args.command,
        "model": args.model,
        "result": summary,
        "seconds": round(time.perf_counter() - started, 1),
        "torch_threads": torch.get_num_threads(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # reference is written per case (runs with a different grid, time or text do not overwrite each
    # other).
    suffix = "-trace" if args.trace_only else ""
    if args.command == "reference":
        suffix = f".{summary['case']}"
    (args.out / f"{args.command}{suffix}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(document, ensure_ascii=False, indent=1))
    failed = (
        bool(summary.get("eager_failures")) or summary.get("layerwise_f32_equals_upstream") is False
    )
    failed = failed or False in (
        summary.get("f32_equals_eager_upstream"),
        summary.get("f64_equals_eager_layerwise"),
    )
    return failed


if __name__ == "__main__":
    sys.exit(main())
