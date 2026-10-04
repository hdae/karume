"""Wan2.2 TI2V-5B の DiT の export と参照のホスト RAM を測る台本（ADR 0121 段 0 — 決定 7）。

    uv run --group wan --inexact python -m wan.dit_probe prepare --trace-only  # 重み不要
    uv run --group wan --inexact python -m wan.dit_probe prepare     # i8 export の材料まで
    uv run --group wan --inexact python -m wan.dit_probe eager       # eager 同値の門
    uv run --group wan --inexact python -m wan.dit_probe reference   # 層逐次の参照（S = 192）
    uv run --group wan --inexact python -m wan.dit_probe reference --latent 21,30,52  # 8,190
    uv run --group wan --inexact python -m wan.dit_probe reference --latent 21,30,52 \
        --text boxing-cats --timestep 500 --growth   # f32 と f64 の差の切り分け（段 1）

どのサブコマンドも段ごとの壁時間と RSS（`wan.umt5_export.MemoryMonitor` — `/proc` の 1 秒ごとの
標本と段ごとの VmHWM）を JSON で出し、同じものを `--out`（既定 {@link default_out} —
`outputs/bench/<配布名>/<日付>_dit-probe/`）の `<command>.json` に書く。系列
（`outputs/series/`）には何も書かない。`reference` は同じ置き場の `eager` の結果と
突き合わせるので、日をまたぐときは両方に同じ `--out` を渡す。

`prepare` / `eager` の DiT のパッチは段 0 のまま Wan2.1 と同じ形（時刻入力 1 本 —
`dit_patch.WanDitTokens`）で、5B の構成はその前提の門（`image_dim` / `added_kv_proj_dim` が None）を
通る。製品の I2V 対応のグラフ（時刻入力 2 本・条件マスク — ADR 0121 決定 3）は段 1 の
`wan.ti2v_export_dit` が `prepare` に自前の trace（`tracer=`）を渡して書く。グラフは linear の
ノードが 3 本増えるだけ（段 1 の IR の検査で 310）なので、RAM の形はここで測った値で読める。

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

### CPU f32 と f64 の差の切り分け（ADR 0121 段 0 の要調査 → 段 1）

S = 8,190 の合成の入力で f32 と f64 の出力の比が 1.2e-1 だった（S = 192 は 5.3e-4）。原因を
切り分ける口:

- `--text <名前>`: text の入力を合成の乱数 `[有効長, 4096]` から、Wan2.1 の事前計算の埋め込み
  資産（同じ umT5 の出力 — `wan.text_embeds`）の 1 本に替える。潜在は同じ seed の同じ乱数（乱数は
  潜在を先に引く）なので、差は text だけ。
- `--timestep <t>`: t = 999 以外。
- `--growth`: ブロックごとの f32 と f64 の差（最大絶対差・比・relRMS・最大差のトークン）。f32 の
  回でブロックの出力を `--out` の下へ 1 本ずつ書き（S = 8,190 で 1 本 100 MB）、f64 の回で 1 本
  ずつ読んで比べて消す（両方を RAM に持たない）。行は `growth.<case>.jsonl` に 1 ブロック 1 行で
  足していく（途中で止めても済んだブロックは読める）。
- `--blocks <N>`: 先頭の N ブロックだけを回す（S = 8,190 の f64 は 1 ブロック 30 s 級 — 全 30 層を
  回せない時間の枠で、伸びの始まる層を見る）。出力は N ブロック目の後に head を掛けた値（参照では
  ない）。
- `--case F,H,W:t:text[:N]`（繰り返し可）: 上の 4 つの組を 1 回の実行で順に回す（重みの読み口と
  プロセスの起動を 1 回で済ませる — 要約はケースごとの `reference.<case>.json`）。

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


#: trace の口（config・S の上限 → グラフ・格納テンソル・量子化の対象）。
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
    """trace → plain → quantize（`trace_only` なら trace まで）。戻りは材料と要約の欄。

    `tracer` は trace の差し替え（既定は 2.1 形の {@link trace}。I2V 対応のグラフは
    `wan.ti2v_export_dit.trace` — 材料作りの残り〈plain / quantize〉は同じ 1 本を通す）。
    """
    config = dit_reference.load_config(directory)
    with stage("trace") as record:
        graph, tensors, targets = (trace if tracer is None else tracer)(config, sym_max)
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


def _probe_spec(
    latent: tuple[int, int, int] | None, timestep: int | None = None
) -> export_dit.CaseSpec:
    """測るケース（既定は {@link PROBE_CASE}・`latent` / `timestep` を渡すと同じ有効長 / seed の
    別の格子・別の時刻）。"""
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


#: text の入力の既定（合成の乱数 — `export_dit.case_inputs`）。ほかの値は埋め込み資産のテンソル名。
SYNTHETIC_TEXT = "synthetic"

#: Wan2.1 の事前計算の埋め込み資産（同じ umT5 の出力 — `wan.text_embeds` の席）。
TEXT_EMBEDS_ASSET = SERIES_ROOT / "wan2.1-t2v-1.3b-text-embeds" / "text_embeds.safetensors"

#: ブロックごとの差の行のファイルの綴り（`growth.<case>.jsonl`）と、f32 のブロック出力の一時置き場。
GROWTH_PREFIX = "growth."
GROWTH_SPILL = "growth-spill"


def probe_inputs(
    model: Any, spec: export_dit.CaseSpec, text: str = SYNTHETIC_TEXT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """下見の入力（潜在・timestep・文脈 `[1,512,4096]`）。

    `text` が合成なら `export_dit.case_inputs` そのもの。埋め込み資産の名前なら、潜在は同じ seed の
    同じ乱数（`case_inputs` は潜在を先に引く）で、文脈だけをその埋め込みにする — 差は text だけ。
    """
    latents, timestep, synthetic = export_dit.case_inputs(model, spec)
    if text == SYNTHETIC_TEXT:
        return latents, timestep, synthetic
    from wan.text_embeds import read_asset

    tensors, _ = read_asset(TEXT_EMBEDS_ASSET)
    if text not in tensors:
        raise DitProbeError(f"埋め込み資産に '{text}' が無い（{sorted(tensors)}）")
    return latents, timestep, pad_text_embeds(tensors[text])


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """f32 と f64 の差の要約（比 = 最大絶対差 ÷ 参照の最大絶対値・relRMS・最大差のトークン）。

    `expected` は f64。トークンは `[1,S,dim]` の S 軸（出力 `[1,C,F,H,W]` は平坦の位置）。
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
    """ブロックごとの f32 と f64 の差（モジュール doc「切り分け」）。

    f32 の回の {@link spill} がブロックの出力を 1 本ずつ書き、f64 の回の {@link compare} が 1 本
    ずつ読んで比べて消し、行を `rows_path` へ足す。
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
        """比べ終えた一時置き場（ケースのディレクトリと、空になった親）を消す。"""
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
    """checkpoint から 1 本ずつ i8 の fake-quant を掛けて読む層逐次の f32 / f64 の RAM と所要
    （`text` / `timestep` / `growth` / `blocks` は切り分けの口 — モジュール doc）。"""
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


@dataclass(frozen=True)
class ProbeRun:
    """`reference` の 1 ケースの指定（`--case` の 1 本 — 格子・時刻・text・回すブロックの数）。"""

    latent: tuple[int, int, int] | None
    timestep: int | None
    text: str
    blocks: int | None


def _probe_run(text: str) -> ProbeRun:
    """`--case F,H,W:t:text[:N]`（例 `4,30,52:999:boxing-cats`・`9,44,80:999:synthetic:6`）。"""
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("--case は F,H,W:timestep:text[:blocks]")
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
        "--out", type=Path, help="出力の置き場（既定 outputs/bench/<配布名>/<日付>_dit-probe）"
    )
    parser.add_argument("--trace-only", action="store_true", help="prepare: trace だけ")
    parser.add_argument("--sym-max", type=int, default=SYM_MAX, help="prepare: S の記号の上限")
    parser.add_argument("--latent", type=_latent, help="reference: 潜在の格子 F,H,W")
    parser.add_argument("--timestep", type=int, help="reference: timestep（既定 999）")
    parser.add_argument(
        "--text",
        default=SYNTHETIC_TEXT,
        help="reference: text の入力（synthetic = 合成の乱数・ほかは埋め込み資産のプロンプト名）",
    )
    parser.add_argument(
        "--growth", action="store_true", help="reference: ブロックごとの f32 と f64 の差を記録する"
    )
    parser.add_argument(
        "--blocks", type=int, help="reference: 先頭の N ブロックだけ回す（誤差の伸びの切り詰め）"
    )
    parser.add_argument(
        "--case",
        type=_probe_run,
        action="append",
        default=[],
        help="reference: F,H,W:timestep:text[:blocks] を 1 回の実行で順に回す（繰り返し可）",
    )
    args = parser.parse_args(argv)
    if args.trace_only and args.command != "prepare":
        parser.error("--trace-only は prepare にだけ掛かる")
    reference_only = (
        args.latent is not None
        or args.timestep is not None
        or args.text != SYNTHETIC_TEXT
        or args.growth
        or args.blocks is not None
    )
    if (reference_only or args.case) and args.command != "reference":
        parser.error(
            "--latent / --timestep / --text / --growth / --blocks / --case は reference に"
            "だけ掛かる"
        )
    if args.case and _single_flags(args):
        parser.error("--case と --latent / --timestep / --text / --blocks は併用しない")
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
    """1 ケースの指定（`--latent` / `--timestep` / `--text` / `--blocks`）が 1 つでもあるか。"""
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
    """要約を `--out` の `<command>.json`（reference はケースごと）へ書き、失敗の判定を返す。"""
    summary, started = result
    document = {
        "command": args.command,
        "model": args.model,
        "result": summary,
        "seconds": round(time.perf_counter() - started, 1),
        "torch_threads": torch.get_num_threads(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # reference はケースごとに書く（格子・時刻・text を替えた実行どうしで上書きしない）。
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
