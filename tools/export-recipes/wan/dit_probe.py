"""Wan2.2 TI2V-5B の DiT の export と参照のホスト RAM を測る台本（ADR 0121 段 0 — 決定 7）。

    uv run --group wan --inexact python -m wan.dit_probe prepare --trace-only  # 重み不要
    uv run --group wan --inexact python -m wan.dit_probe prepare     # i8 export の材料まで
    uv run --group wan --inexact python -m wan.dit_probe eager       # eager 同値の門
    uv run --group wan --inexact python -m wan.dit_probe reference   # 層逐次の参照（S = 192）
    uv run --group wan --inexact python -m wan.dit_probe reference --latent 21,30,52  # 8,190

どのサブコマンドも段ごとの壁時間と RSS（`wan.umt5_export.MemoryMonitor` — `/proc` の 1 秒ごとの
標本と段ごとの VmHWM）を JSON で出し、同じものを `--out`（既定 {@link default_out} —
`outputs/bench/<配布名>/<日付>_dit-probe/`）の `<command>.json` に書く。系列
（`outputs/series/`）には何も書かない。`reference` は同じ置き場の `eager` の結果と
突き合わせるので、日をまたぐときは両方に同じ `--out` を渡す。

段 0 の時点の DiT のパッチは Wan2.1 と同じ形（時刻入力 1 本 — `dit_patch.WanDitTokens`）で、5B の
構成はその前提の門（`image_dim` / `added_kv_proj_dim` が None）を通る。I2V 対応のグラフ（時刻入力
2 本・条件マスク — ADR 0121 決定 3）は段 1 で、グラフは linear のノードが 3 本増えるだけの見込み
なので、RAM の形はここで測った値で読める。

## prepare（meta trace + 行の塊ごとの i8 + `fixed_weights` — ADR 0119 段 10b の形）

1. **trace**: 重みを持たない meta の上流（config だけ）を S 形のラッパで包み、meta の例示入力
   （S = 192）で export する。S の記号の上限は {@link SYM_MAX}（1280×704×121 — ADR 0121 決定 3）。
2. **plain**: 量子化しない重み（bias・norm・`scale_shift_table`）を checkpoint の F32 のまま読む。
3. **quantize**: linear 307 本（patch 埋め込みの Linear を含む）を checkpoint から行の塊ごとに読み、
   exporter と同じ手順で i8 の packed + 行ごとの scale にする（`umt5_export.quantize_rows` —
   patch 埋め込みは上流では Conv3d `[3072,48,1,2,2]` なので、出力チャネルを行にした 2 次元で読む）。
4. **store**: `karume.emit.stored_model(..., fixed_weights=...)` で格納宣言を commit し、
   `publish_model` が書き出しの前に掛ける検査（`assert_runtime_support` / `assert_op_contracts`）を
   掛ける。容器のファイルは書かない — 書く口（`graph_name=` を名乗る `publish_model`）は全 family
   横断の門（`tests/test_graph_names.py` の表）に行を足す段 1 の仕事。

## eager（eager 同値の門の RAM — ADR 0121 決定 4 の「f32 全量 18.63 GiB + 小さい活性」）

上流を `from_pretrained` で f32 全量読み、S 形のラッパ経由で i8 の fake-quant を掛け、S = 192 の
1 ケースで上流の素の forward（参照）とパッチ後の eager を回して `export_dit.eager_failures` の門に
掛ける（`wan.export_dit` と同じ関数）。加えて層逐次の参照（`wan.dit_reference` — 同じ in-memory の
重み）の f32 が上流の f32 とビット一致することを見て、層逐次の f64 も採る。

RAM の山は fake-quant の段ではなく層逐次の段に来る: f32 の全量を持ったまま、層逐次の f64 が
ブロックの外と 1 ブロックを f64 で足す（{@link eager_memory} — 5B で 18.63 + 0.67 + 1.22 GiB に
基礎分 {@link EAGER_BASE_BYTES} を足して約 21.1 GiB。1.3B の実測〈山 6.76 GiB = f32 5.29 + f64 の
外と 2 ブロック 0.88 + 0.59〉から、1 ブロックずつにした後の値として読んだ見込み）。台本は
`load` と `layerwise` の段の前に `/proc/meminfo` の MemAvailable を読み、残りの見込みに余白
{@link MEMORY_HEADROOM_BYTES} を足した量に足りなければ、必要量を言って止まる（GPU のジョブと
ホスト RAM を分け合う機で、途中で swap に落ちて数時間を失わない）。

## reference（層逐次の参照の RAM と所要 — 容器がまだ無いので checkpoint から）

checkpoint から 1 本ずつ読んで i8 の fake-quant を掛ける読み口
（`dit_reference.CheckpointDitWeights`）で、層逐次の f32 / f64 を採る。f32 は上流の mmap と
同じずれの番地で回す（`dit_reference.upstream_alignment`）。`eager` の結果が `--out` にあれば、
f32 は上流の f32 の参照と、f64 は eager の層逐次の f64 とビットで比べる（同じケースのときだけ）。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.export import Dim

from _shared.paths import BENCH_ROOT
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.emit import FixedQuantizedWeight, storage_breakdown, stored_model
from karume.ir import IrGraph
from karume.pipeline import export_module
from karume.quantize import QUANT_MODULE_TYPES, iter_quant_targets
from karume.verify import assert_op_contracts, assert_runtime_support
from wan import dit_patch, dit_reference, export_dit
from wan.sources import SOURCES, local_snapshot
from wan.umt5_export import CHUNK_ROWS, Checkpoint, MemoryMonitor, quantize_rows

#: 既定のモデル（Wan2.2 TI2V-5B — ADR 0121）。
DEFAULT_MODEL = "ti2v-5b"

#: モデル → 出力の既定の置き場の配布名（`outputs/bench/<配布名>/` — `docs/assets-layout.md`）。
BENCH_NAMES: dict[str, str] = {"t2v-1.3b": "karume-wan2.1", "ti2v-5b": "karume-wan2.2"}

#: eager の山の見積りに足す基礎分（プロセスの基礎・活性・fake-quant の一時 — 1.3B の実測で、山から
#: f32 の全量と層逐次の f64 を引いた残り 0.59 GiB を丸めた値）。
EAGER_BASE_BYTES = int(0.6 * 2**30)

#: MemAvailable の検査で見込みに足す余白（ほかのプロセスの揺れ・見込みの誤差）。
MEMORY_HEADROOM_BYTES = 1 * 2**30

#: `Dim("S")` の上限（1280×704×121 = 27,280 — ADR 0121 決定 3）と下限（Wan2.1 と同じ）。
SYM_MAX = 27_280
SYM_MIN = export_dit.DIT_SYM_MIN

#: 測るケース（`export_dit.CASES` の先頭 = 帯の決定の S = 192・t = 999）。
PROBE_CASE = export_dit.CASES[0]

#: eager / reference の出力ファイルの綴り（`<command>.<case>.safetensors`）。
EAGER_PREFIX = "eager."
REFERENCE_PREFIX = "reference."
#: 出力テンソルのキー（上流の f32 の参照・層逐次の f32 / f64 — f64 は f32 へ丸めずに F64 で持つ）。
UPSTREAM_KEY = "upstream.f32"
LAYERWISE_F32_KEY = "layerwise.f32"
LAYERWISE_F64_KEY = "layerwise.f64"


class DitProbeError(RuntimeError):
    """checkpoint・グラフ・量子化の対象が想定から外れた・ホスト RAM が足りない。"""


def default_out(model: str) -> Path:
    """出力（要約の JSON と、eager / reference の出力テンソル）の既定の置き場。"""
    return BENCH_ROOT / BENCH_NAMES[model] / f"{date.today().isoformat()}_dit-probe"


_GIB = 2**30


def eager_memory(config: Mapping[str, Any]) -> dict[str, int]:
    """eager の段ごとの RAM の見込み（バイト — モジュール doc「eager」）。

    `f32`: 上流の f32 の全量（`load` から持ち続ける）。`layerwise_f64`: 層逐次の f64 が足す
    ブロックの外と 1 ブロック。`peak`: 両方に基礎分を足した山。
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
    """`/proc/meminfo` の MemAvailable（バイト）。"""
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, _, value = line.partition(":")
        if name == "MemAvailable":
            return int(value.split()[0]) * 1024
    raise DitProbeError("/proc/meminfo に MemAvailable が無い")


def require_available(stage: str, need: int) -> None:
    """MemAvailable が `need` + 余白に足りなければ、必要量を言って止まる（重い段の前に呼ぶ）。"""
    available = mem_available()
    wanted = need + MEMORY_HEADROOM_BYTES
    if available < wanted:
        raise DitProbeError(
            f"{stage} の前の MemAvailable {available / _GIB:.2f} GiB が必要量"
            f" {wanted / _GIB:.2f} GiB（見込み {need / _GIB:.2f} + 余白"
            f" {MEMORY_HEADROOM_BYTES / _GIB:.2f}）に足りない — ほかの重い作業が退いてから回す"
        )


def transformer_dir(model: str) -> Path:
    """pin した revision の DiT の置き場（取得済みでなければ `local_snapshot` が fail loudly）。"""
    return local_snapshot(model) / dit_reference.TRANSFORMER_SUBFOLDER


def meta_wrapper(config: Mapping[str, Any]) -> dit_patch.WanDitTokens:
    """config だけから組んだ meta の上流を S 形のラッパで包む（重みは読まない）。"""
    return dit_patch.WanDitTokens(dit_reference.meta_model(config))


def meta_inputs(config: Mapping[str, Any], spec: export_dit.CaseSpec) -> tuple[torch.Tensor, ...]:
    """trace の例示入力（meta — 値は trace に効かない。重みが meta なので入力も meta に揃える）。"""
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
    """`export_dit.dynamic_shapes` と同じ形で、上限だけを 5B の受理の最大にする。"""
    tokens = Dim("S", min=SYM_MIN, max=sym_max)
    return ({1: tokens}, None, None, {1: tokens}, {1: tokens})


def trace(
    config: Mapping[str, Any], sym_max: int = SYM_MAX
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, int]]:
    """meta の上流で S 形の export を回す（グラフ・格納テンソル〈重みは meta〉・量子化の対象）。"""
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
        raise DitProbeError(f"グラフ入力名が宣言と不一致: {declared}")
    targets = {
        fqn: axis for fqn, _, axis in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES)
    }
    return graph, tensors, targets


class _FlatRows:
    """checkpoint の rank 3 以上の重み（patch 埋め込みの Conv3d）を、出力チャネルを行にした 2 次元で
    見せる読み口（`quantize_rows` は rank 2 だけを受ける）。行ごとの scale は Linear 化した重みと
    同じ（行の amax）。"""

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
    """容器の材料（`publish_model` へそのまま渡せる形 — 量子化の対象は meta・実体は `fixed`）。"""

    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    fixed: Mapping[str, FixedQuantizedWeight]
    plain: tuple[str, ...]


def prepare(
    directory: Path,
    *,
    stage: Any,
    sym_max: int = SYM_MAX,
    chunk_rows: int = CHUNK_ROWS,
    trace_only: bool = False,
) -> tuple[DitExport, dict[str, Any]]:
    """trace → plain → quantize（`trace_only` なら trace まで）。戻りは材料と要約の欄。"""
    config = dit_reference.load_config(directory)
    with stage("trace") as record:
        graph, tensors, targets = trace(config, sym_max)
        record.details["nodes"] = len(graph.nodes)
    weights = sorted(key for key, value in tensors.items() if value.is_meta)
    unknown = sorted(set(targets) - set(weights))
    if unknown or any(axis != 0 for axis in targets.values()):
        raise DitProbeError(f"量子化の対象がグラフの重みに無い / 軸 0 でない: {unknown}")
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
        raise DitProbeError(f"グラフの重みが checkpoint に無い: {missing[:5]}")
    materialized = dict(tensors)
    with stage("plain") as record:
        for key in plain:
            value = checkpoint.read(key)
            if list(value.shape) != list(tensors[key].shape):
                raise DitProbeError(f"'{key}': checkpoint の形 {list(value.shape)} がグラフと違う")
            materialized[key] = value
        record.details["tensors"] = len(plain)
    fixed: dict[str, FixedQuantizedWeight] = {}
    rows_view = _FlatRows(checkpoint)
    with stage("quantize") as record:
        for key in sorted(targets):
            weight = quantize_rows(rows_view, key, chunk_rows)  # type: ignore[arg-type]
            if list(weight.packed.shape) != list(tensors[key].shape):
                raise DitProbeError(f"'{key}': checkpoint の形がグラフと違う")
            fixed[key] = weight
        record.details["tensors"] = len(fixed)
    summary["packed_bytes"] = sum(weight.packed.numel() for weight in fixed.values())
    return DitExport(graph=graph, tensors=materialized, fixed=fixed, plain=plain), summary


def store(export: DitExport) -> dict[str, Any]:
    """格納宣言を commit し、書き出しの前の検査を掛ける（容器は書かない）。"""
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


def _probe_spec(latent: tuple[int, int, int] | None) -> export_dit.CaseSpec:
    """測るケース（既定は {@link PROBE_CASE}・`latent` を渡すと同じ timestep / 有効長 / seed の
    別の格子）。"""
    if latent is None:
        return PROBE_CASE
    return export_dit.CaseSpec(
        "probe", latent, PROBE_CASE.timestep, PROBE_CASE.text_length, PROBE_CASE.seed
    )


def _case_name(spec: export_dit.CaseSpec, config: Mapping[str, Any]) -> str:
    patch_t, patch_h, patch_w = (int(size) for size in config["patch_size"])
    return spec.name((patch_t, patch_h, patch_w))


def run_eager(model: str, out: Path) -> dict[str, Any]:
    """eager 同値の門と層逐次の f32 / f64（同じ in-memory の重み）の RAM。"""
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


def run_reference(model: str, out: Path, latent: tuple[int, int, int] | None) -> dict[str, Any]:
    """checkpoint から 1 本ずつ i8 の fake-quant を掛けて読む層逐次の f32 / f64 の RAM と所要。"""
    directory = transformer_dir(model)
    config = dit_reference.load_config(directory)
    spec = _probe_spec(latent)
    name = _case_name(spec, config)
    with MemoryMonitor() as monitor:
        with monitor.stage("setup") as record:
            writer = dit_reference.LayerwiseDit(
                config, alignment=dit_reference.upstream_alignment(directory)
            )
            source = dit_reference.CheckpointDitWeights(
                directory, "i8", dit_reference.quant_keys(config)
            )
            latents, timestep, text = export_dit.case_inputs(writer, spec)
            record.details["tokens"] = int(latents[0, 0].numel() // math.prod(writer.patch_size))
        results: dict[str, dit_reference.DitReference] = {}
        for label, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            with monitor.stage(f"layerwise-{label}") as record:
                results[label] = writer.forward(
                    source,
                    dtype,
                    latents,
                    timestep,
                    text,
                    on_block=_progress(name, label, writer.layers),
                )
                record.details["seconds"] = round(results[label].seconds, 1)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            LAYERWISE_F32_KEY: results["f32"].output.contiguous(),
            LAYERWISE_F64_KEY: results["f64"].output.contiguous(),
        },
        str(out / f"{REFERENCE_PREFIX}{name}.safetensors"),
    )
    summary: dict[str, Any] = {
        "case": name,
        "f32_vs_f64": export_dit._ratio(results["f32"].output, results["f64"].output),
        "stages": [record.to_dict() for record in monitor.records],
    }
    eager = out / f"{EAGER_PREFIX}{name}.safetensors"
    # 突き合わせの相手の有無を要約に出す（無いと比較の欄が出ない — 「比べていない」を見える形に）。
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
        raise argparse.ArgumentTypeError("--latent は F,H,W（潜在の格子 — 例 21,30,52）")
    return parts[0], parts[1], parts[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("prepare", "eager", "reference"))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument(
        "--out", type=Path, help="出力の置き場（既定 outputs/bench/<配布名>/<日付>_dit-probe）"
    )
    parser.add_argument("--trace-only", action="store_true", help="prepare: trace だけ")
    parser.add_argument("--sym-max", type=int, default=SYM_MAX, help="prepare: S の記号の上限")
    parser.add_argument("--latent", type=_latent, help="reference: 潜在の格子 F,H,W")
    args = parser.parse_args(argv)
    if args.trace_only and args.command != "prepare":
        parser.error("--trace-only は prepare にだけ掛かる")
    if args.latent is not None and args.command != "reference":
        parser.error("--latent は reference にだけ掛かる")
    if args.out is None:
        args.out = default_out(args.model)
    started = time.perf_counter()
    if args.command == "prepare":
        summary = run_prepare(args.model, trace_only=args.trace_only, sym_max=args.sym_max)
    elif args.command == "eager":
        summary = run_eager(args.model, args.out)
    else:
        summary = run_reference(args.model, args.out, args.latent)
    document = {
        "command": args.command,
        "model": args.model,
        "result": summary,
        "seconds": round(time.perf_counter() - started, 1),
        "torch_threads": torch.get_num_threads(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "-trace" if args.trace_only else ""
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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
