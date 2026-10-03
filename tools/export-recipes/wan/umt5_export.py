"""Wan2.1 の umT5（`UMT5EncoderModel`）を i8 の S 形容器の材料にする書き手の骨格（ADR 0119 段 10b —
決定 5 / 6。段 10c で育てる）。

    uv run --group wan --inexact python -m wan.umt5_export prepare       # 材料まで（書かない）
    uv run --group wan --inexact python -m wan.umt5_export check-mask --dtype bf16 --out <席>
    uv run --group wan --inexact python -m wan.umt5_export check-mask --dtype f32 --out <席>
    uv run --group wan --inexact python -m wan.umt5_export compare-mask --out <席>

どのサブコマンドも段ごとの壁時間と RSS（{@link MemoryMonitor}）を JSON で出す。数値の記録は
research `2026-10-03-umt5-export-ram`。

## export の形（決定 6 — 段 10b の実測で決めた形）

全重みを f32 で読み、丸めてから export する素直な形は 31 GiB 機に収まらない見込み: f32 の全重みが
21.2 GiB あり、語彙埋め込み（`[256384, 4096]` f32 = 3.91 GiB）の fake-quant と emit の i8 変換が、
それぞれ一時に 2.0 倍・2.25 倍（実測）を上乗せする（research の §2）。そこで次の 3 つに分ける:

1. **trace は重みを持たない（meta）上流のモデルで回す**（{@link meta_text_encoder} →
   {@link trace}）。export が辿るのは形だけなので、グラフは実重みの export と JSON で同一になる
   （pytest が小模型で縛る）。
2. **量子化の対象（linear と語彙埋め込み）は checkpoint から行の塊ごとに読み**、exporter の i8 と
   同じ手順で packed と scale にして `fixed_weights`（ADR 0097 の入口）で渡す
   （{@link quantize_rows}）。per-channel の scale は行ごとに閉じる（linear も埋め込みも軸 0）ので、
   塊に割っても全体で一度に回した値とビット一致する。
3. **量子化しない重み**（RMSNorm の重み・相対位置の表 24 本）は checkpoint の f32 をそのまま渡す。
   `fixed_weights` を使う回の既定の格納は f32 なので、表の F32 の明示（`storage_overrides`）は
   要らない（fixed と自動量子化の指定は emit が混在を拒む）。

容器のバイトは素直な形（f32 で `fake_quant_i8` → `weight_dtype="i8"` + 表の F32 明示）と一致する
（pytest が小模型で、容器の全 part のバイト一致で縛る）。exporter core の変更は要らない。

## 容器を書く口がここに無い理由

容器は `publish_model(..., graph_name=<部品名>)` で書くが、recipe の台本が `graph_name=` を名乗ると
全 family 横断の門（`tests/test_graph_names.py`）が配布形の部品名（`wan.distribution.WAN_WEIGHTS`
のキー）との一致を求める。umT5 の部品名と配布の構成は段 10d（ADR 0119 裁定 1）なので、名乗る
綴りがまだ無い。段 10b の容器は recipe の外の driver が {@link prepare} の戻りを
`publish_model(..., fixed_weights=..., graph_name="text_encoder")` へ渡して書いた（置き場は
{@link SERIES} / {@link COMPONENT_DIR} — 部品名は下見と同じ仮置き）。部品名が決まった段で、書く
関数をここへ足し、門の表（`ENTRIES`）と `WAN_WEIGHTS` に行を足す。

MUST: transformers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import json
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
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.paths import SERIES_ROOT
from karume.container import Provenance
from karume.dist import NOTICE_FILENAME
from karume.emit import FixedQuantizedWeight
from karume.ir import IrGraph
from karume.pipeline import export_module
from karume.quantize import QUANT_MODULE_TYPES, channel_scale, iter_quant_targets, quantize_to_int8
from wan import umt5_patch
from wan.sources import DEFAULT_MODEL, SOURCES, text_snapshot

#: i8 の系列（綴りは配布の規約 `<名>-<格納>-dyn` — DiT の `wan2.1-t2v-1.3b-i8-dyn` に倣う）。
SERIES_NAME = "wan2.1-umt5-i8-dyn"
SERIES = SERIES_ROOT / SERIES_NAME

#: 系列の中の置き場（下見と同じ仮置き — 配布形の部品名は段 10d の裁定）。
COMPONENT_DIR = "text_encoder"
MODEL_FILE = "model.krm"

#: 上流の部品のディレクトリ（snapshot の下）。
UPSTREAM_SUBFOLDER = "text_encoder"

#: trace の例示入力の有効長（下見の `umt5_probe.EXPORT_LENGTHS[0]` と同じ — L は記号次元なので
#: 値は形の特殊化を踏まない数なら何でもよい: 0 / 1 と heads 64・d_kv 64・バケット 32 を避ける）。
TRACE_LENGTH = 28

#: 量子化で一度に読む行数（`[rows, 4096]` f32 で 256 MiB）。一時の確保は塊の数倍で、語彙埋め込み
#: 全体（3.91 GiB）を一度に回す形の 2.25 倍（約 8.8 GiB）を避ける。
CHUNK_ROWS = 16_384

#: checkpoint の索引（分割形）と単一形のファイル名（transformers の `save_pretrained` の綴り）。
CHECKPOINT_INDEX = "model.safetensors.index.json"
CHECKPOINT_SINGLE = "model.safetensors"

#: KiB → GiB（`/proc` の値と Linux の `ru_maxrss` は KiB）。
_KIB_PER_GIB = 1 << 20


class Umt5ExportError(RuntimeError):
    """checkpoint・グラフ・量子化の対象が想定（決定 5）から外れた。"""


# ---------------------------------------------------------------------------
# RSS の計測
# ---------------------------------------------------------------------------

_STATUS_FIELDS = ("VmRSS", "VmHWM", "RssAnon", "RssFile")
_MEMINFO_FIELDS = ("MemAvailable", "SwapFree")


def _read_kib(path: str, fields: Sequence[str]) -> dict[str, int]:
    """`/proc` の `Key: <n> kB` 形式のファイルから欄を読む（無い欄は fail loudly）。"""
    values: dict[str, int] = {}
    with open(path, encoding="ascii") as stream:
        for line in stream:
            key, _, rest = line.partition(":")
            if key in fields:
                values[key] = int(rest.split()[0])
    missing = sorted(set(fields) - set(values))
    if missing:
        raise Umt5ExportError(f"{path} に欄 {missing} が無い")
    return values


def _reset_peak() -> None:
    """VmHWM を今の RSS へ戻す（`clear_refs` の 5 — 段ごとのピークを取るため）。"""
    with open("/proc/self/clear_refs", "w", encoding="ascii") as stream:
        stream.write("5")


@dataclass
class StageRecord:
    """1 段の壁時間とメモリ（GiB）。

    `peak_rss` / `peak_anon` / `peak_file` は 1 秒ごとの標本（と段の終わりの 1 点）の最大で、標本の
    間の山は取りこぼしうる。取りこぼさないのは `hwm`（段の始めに戻した VmHWM — 段の中の真の山）。
    `ru_maxrss` は段の終わりの `getrusage` で、Linux ではプロセスの mm の hiwater を読むので
    `clear_refs` の戻しで一緒に戻る（実測 — 段ごとに下がる）。プロセス全体の山は各段の `hwm` の
    最大。
    """

    name: str
    seconds: float = 0.0
    peak_rss: float = 0.0
    peak_anon: float = 0.0
    peak_file: float = 0.0
    hwm: float = 0.0
    ru_maxrss: float = 0.0
    min_available: float = float("inf")
    min_swap_free: float = float("inf")
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": round(self.seconds, 1),
            **{
                key: round(getattr(self, key), 2)
                for key in (
                    "peak_rss",
                    "peak_anon",
                    "peak_file",
                    "hwm",
                    "ru_maxrss",
                    "min_available",
                    "min_swap_free",
                )
            },
            **self.details,
        }


class MemoryMonitor:
    """別スレッドで `/proc/self/status` と `/proc/meminfo` を 1 秒ごとに読み、段ごとの山を記録する。

    `with MemoryMonitor() as monitor:` の中で `with monitor.stage("load"):` を重ねずに並べる。
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.records: list[StageRecord] = []
        self._current: StageRecord | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)

    def __enter__(self) -> MemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        status = _read_kib("/proc/self/status", _STATUS_FIELDS)
        meminfo = _read_kib("/proc/meminfo", _MEMINFO_FIELDS)
        with self._lock:
            record = self._current
            if record is None:
                return
            record.peak_rss = max(record.peak_rss, status["VmRSS"] / _KIB_PER_GIB)
            record.peak_anon = max(record.peak_anon, status["RssAnon"] / _KIB_PER_GIB)
            record.peak_file = max(record.peak_file, status["RssFile"] / _KIB_PER_GIB)
            record.min_available = min(record.min_available, meminfo["MemAvailable"] / _KIB_PER_GIB)
            record.min_swap_free = min(record.min_swap_free, meminfo["SwapFree"] / _KIB_PER_GIB)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        """段 1 つを測る（VmHWM を戻してから始め、終わりに 1 点を足して締める）。"""
        record = StageRecord(name)
        _reset_peak()
        with self._lock:
            self._current = record
        self._sample()
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.seconds = time.perf_counter() - started
            self._sample()
            with self._lock:
                self._current = None
            record.hwm = _read_kib("/proc/self/status", ("VmHWM",))["VmHWM"] / _KIB_PER_GIB
            record.ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _KIB_PER_GIB
            self.records.append(record)
            print(f"[stage] {json.dumps(record.to_dict(), ensure_ascii=False)}", flush=True)


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------


class Checkpoint:
    """上流の safetensors（分割形か単一形）からテンソルを 1 本ずつ・行の塊ごとに読む。

    全部を一度に載せないための口（決定 6）。MUST: 格納は F32 だけを受ける — 上流の pin した
    checkpoint は F32（`total_size` 22,723,641,344 B）で、別の dtype なら丸めの出発点が変わる。
    """

    def __init__(self, directory: Path) -> None:
        index = directory / CHECKPOINT_INDEX
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            self._files = {key: directory / name for key, name in weight_map.items()}
        elif (directory / CHECKPOINT_SINGLE).is_file():
            single = directory / CHECKPOINT_SINGLE
            with safe_open(str(single), framework="pt") as handle:
                self._files = dict.fromkeys(handle.keys(), single)
        else:
            raise Umt5ExportError(
                f"{directory} に {CHECKPOINT_INDEX} も {CHECKPOINT_SINGLE} も無い"
            )

    def names(self) -> frozenset[str]:
        return frozenset(self._files)

    def shape(self, key: str) -> list[int]:
        with safe_open(str(self._files[key]), framework="pt") as handle:
            view = handle.get_slice(key)
            self._assert_f32(key, view.get_dtype())
            return list(view.get_shape())

    def read(self, key: str) -> torch.Tensor:
        """1 本を丸ごと（小さい重み — RMSNorm と相対位置の表）。"""
        with safe_open(str(self._files[key]), framework="pt") as handle:
            tensor = handle.get_tensor(key)
        self._assert_f32(key, str(tensor.dtype))
        return tensor

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        """行 `[start, stop)` だけ（先頭の軸の切り出し — 残りの軸は丸ごと）。"""
        with safe_open(str(self._files[key]), framework="pt") as handle:
            view = handle.get_slice(key)
            self._assert_f32(key, view.get_dtype())
            return view[start:stop]

    @staticmethod
    def _assert_f32(key: str, dtype: str) -> None:
        if dtype not in ("F32", "torch.float32"):
            raise Umt5ExportError(f"checkpoint の '{key}' が {dtype}（F32 だけを受ける）")


def checkpoint_keys(
    model: nn.Module, wrapper_keys: Sequence[str], available: frozenset[str]
) -> dict[str, str]:
    """ラッパのテンソルキー → checkpoint のキー。

    ラッパの `encoder` は上流の `encoder` そのものなので名前は同じ空間にある。tied な重み
    （`encoder.embed_tokens.weight` = `shared.weight`）は checkpoint に片方の名前でしか無いので、
    同じ Parameter を指す別名の中から checkpoint に在る 1 つを選ぶ。MUST: 候補が 0 本・2 本以上なら
    fail loudly（黙って別の重みを読まない）。
    """
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    by_name = {name: names for names in aliases.values() for name in names}
    mapping: dict[str, str] = {}
    for key in wrapper_keys:
        candidates = sorted(set(by_name.get(key, [key])) & available)
        if len(candidates) != 1:
            raise Umt5ExportError(f"'{key}' の checkpoint のキーが 1 つに決まらない: {candidates}")
        mapping[key] = candidates[0]
    return mapping


# ---------------------------------------------------------------------------
# trace（重みを持たない上流）
# ---------------------------------------------------------------------------


def upstream_dir(model: str = DEFAULT_MODEL) -> Path:
    """pin した revision の umT5 の置き場（取得済みでなければ `text_snapshot` が fail loudly）。"""
    return text_snapshot(model) / UPSTREAM_SUBFOLDER


def meta_text_encoder(directory: Path) -> nn.Module:
    """config だけから組んだ上流の `UMT5EncoderModel`（重みは meta・eval・f32）。"""
    from transformers import UMT5Config, UMT5EncoderModel

    config = UMT5Config.from_pretrained(directory)
    with torch.device("meta"):
        model = UMT5EncoderModel(config)
    return model.eval()


def trace_inputs(
    wrapper: umt5_patch.Umt5EncoderTokens, length: int = TRACE_LENGTH
) -> tuple[torch.Tensor, ...]:
    """trace の例示入力（token id `[1, L]` とバケット表 `[L, L]`）。

    値は trace に効かない（記号次元の形だけを辿る）。token id は語彙の内の決まった並び、表は
    0 層目の上流の式（重みを使わない）で作る。
    """
    config = wrapper.encoder.config
    ids = (torch.arange(length, dtype=torch.long) % (config.vocab_size - 3) + 3).unsqueeze(0)
    ids[0, -1] = config.eos_token_id
    attention = umt5_patch.bucket_attention(config)
    return ids, umt5_patch.relative_position_buckets(length, attention)


def trace(wrapper: umt5_patch.Umt5EncoderTokens) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """有効長 `L` の S 形で export する（重みが meta でも実体でも同じグラフ）。"""
    graph, tensors = export_module(
        wrapper,
        trace_inputs(wrapper),
        dynamic_shapes=umt5_patch.dynamic_shapes(),
        symbol_names=(umt5_patch.SYMBOL,),
    )
    declared = [entry.name for entry in graph.inputs]
    if declared != list(umt5_patch.INPUT_NAMES):
        raise Umt5ExportError(f"グラフ入力名が宣言と不一致: {declared}")
    return graph, tensors


# ---------------------------------------------------------------------------
# 量子化（行の塊ごと）
# ---------------------------------------------------------------------------


def quantize_rows(
    checkpoint: Checkpoint, key: str, chunk_rows: int = CHUNK_ROWS
) -> FixedQuantizedWeight:
    """行ごとの scale の i8（`[rows, cols]` の軸 0）を、行の塊ごとに読んで作る。

    1 塊の手順は素直な形の 2 段と同じ: `fake_quant_int8` が重みを `quantize(w, s)·s` に置き換え、
    emit が `quantize(q·s, s)` を焼いて逆変換の一致を検査する。ここも 2 回目の量子化の値を焼き、
    同じ一致を検査する — 1 回目の `q` を直に焼くと、`round((q·s)/s) ≠ q` の要素があった場合に
    素直な形と割れる（等しいことは前提にしない）。行の間で値が混ざる演算は無い（scale は行の
    amax）ので、塊に割っても全体で回した値とビット一致する。
    """
    shape = checkpoint.shape(key)
    if len(shape) != 2:
        raise Umt5ExportError(f"'{key}' が rank {len(shape)}（行ごとの i8 は rank 2 だけ）")
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
            raise Umt5ExportError(f"'{key}' の行 {start}〜{stop}: i8 × scale の逆変換が一致しない")
        packed[start:stop] = chunk
        scale[start:stop] = chunk_scale
    return FixedQuantizedWeight(dtype="i8", packed=packed, scale=scale)


# ---------------------------------------------------------------------------
# 材料の組み立て
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Umt5Export:
    """容器の材料（`publish_model` へそのまま渡せる形）。

    `tensors` の量子化の対象は meta（論理形だけ）で、実体は `fixed`（packed + scale）にある。
    量子化しない重みは checkpoint の f32 の実体。
    """

    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]
    fixed: Mapping[str, FixedQuantizedWeight]
    #: 量子化しない重みのテンソルキー（RMSNorm と相対位置の表 — 種類は {@link storage_kind}）。
    plain: tuple[str, ...]


def quant_targets(wrapper: umt5_patch.Umt5EncoderTokens) -> dict[str, int]:
    """i8 にする重み（テンソルキー → チャネル軸）。選択は下見と同じ（相対位置の表だけを外す）。"""
    include = umt5_patch.quant_include(wrapper)
    return {
        fqn: axis
        for fqn, _, axis in iter_quant_targets(
            wrapper, op_types=QUANT_MODULE_TYPES, include=include
        )
    }


def prepare(
    directory: Path,
    *,
    monitor: MemoryMonitor | None = None,
    chunk_rows: int = CHUNK_ROWS,
) -> Umt5Export:
    """trace → 量子化しない重みの読み込み → 量子化の対象の i8 化（段ごとに `monitor` で測る）。"""
    stage = monitor.stage if monitor is not None else _unmeasured
    with stage("trace") as record:
        model = meta_text_encoder(directory)
        wrapper = umt5_patch.Umt5EncoderTokens(model)
        graph, tensors = trace(wrapper)
        targets = quant_targets(wrapper)
        record.details["nodes"] = len(graph.nodes)
    checkpoint = Checkpoint(directory)
    weights = sorted(key for key, value in tensors.items() if value.is_meta)
    mapping = checkpoint_keys(model, weights, checkpoint.names())
    unknown = sorted(set(targets) - set(weights))
    if unknown:
        raise Umt5ExportError(f"量子化の対象がグラフの重みに無い: {unknown}")
    materialized = dict(tensors)
    with stage("plain") as record:
        plain = tuple(key for key in weights if key not in targets)
        for key in plain:
            value = checkpoint.read(mapping[key])
            if list(value.shape) != list(tensors[key].shape):
                raise Umt5ExportError(
                    f"'{key}': checkpoint の形 {list(value.shape)} がグラフと違う"
                )
            materialized[key] = value
        record.details["tensors"] = len(plain)
    fixed: dict[str, FixedQuantizedWeight] = {}
    with stage("quantize") as record:
        for key, axis in sorted(targets.items()):
            if axis != 0:
                raise Umt5ExportError(f"'{key}' のチャネル軸が {axis}（行ごとの scale だけを作る）")
            weight = quantize_rows(checkpoint, mapping[key], chunk_rows)
            if list(weight.packed.shape) != list(tensors[key].shape):
                raise Umt5ExportError(f"'{key}': checkpoint の形がグラフと違う")
            fixed[key] = weight
        record.details["tensors"] = len(fixed)
    return Umt5Export(graph=graph, tensors=materialized, fixed=fixed, plain=plain)


@contextmanager
def _unmeasured(name: str) -> Iterator[StageRecord]:
    yield StageRecord(name)


def provenance(model: str = DEFAULT_MODEL) -> Provenance:
    """容器へ焼く出所（上流のライセンス識別子と pin した revision — `wan.sources` が正本）。"""
    source = SOURCES[model]
    return Provenance(
        license=source.license, notice=NOTICE_FILENAME, upstream_revision=source.revision
    )


def storage_kind(key: str) -> str:
    """テンソルキーの種類（相対位置の表・語彙埋め込み・norm・linear・定数）— 検収の表の行。"""
    if key.endswith(f".{umt5_patch.RELATIVE_BIAS_ATTRIBUTE}.weight"):
        return umt5_patch.RELATIVE_BIAS_ATTRIBUTE
    if key == "encoder.embed_tokens.weight":
        return "embed_tokens"
    if key.endswith("layer_norm.weight"):
        return "norm"
    if key.endswith(".weight"):
        return "linear"
    return "constant"


def storage_by_kind(graph: IrGraph) -> dict[str, dict[str, int]]:
    """格納宣言を種類 × 格納 dtype で数える（`graph` は格納宣言を commit したもの）。"""
    counts: dict[str, Counter[str]] = {}
    for initializer in graph.initializers.values():
        kind = storage_kind(initializer.tensor or "")
        counts.setdefault(kind, Counter())[initializer.storage.dtype] += 1
    return {kind: dict(sorted(counter.items())) for kind, counter in sorted(counts.items())}


# ---------------------------------------------------------------------------
# 計測のサブコマンド
# ---------------------------------------------------------------------------


#: 決定 4 の裏付けで回す dtype（bf16 = 上流と資産の経路・f32 = 丸めの床を見る対照）。
MASK_DTYPES: Mapping[str, torch.dtype] = {"bf16": torch.bfloat16, "f32": torch.float32}

#: 出力を書くファイルの綴り（`mask-<dtype>.safetensors`・テンソルは `<形>.<プロンプト名>`）。
MASK_FILE = "mask-{dtype}.safetensors"
MASK_FORMS = ("valid", "padded")


def load_upstream(model: str, dtype: torch.dtype) -> tuple[Any, Any]:
    """pin した revision のトークナイザと上流の umT5（`dtype`・CPU・eval）。

    資産の経路（`wan.text_embeds.load_text_encoder` — bf16）と同じ呼び方で、dtype だけを選ぶ。
    f32 は素直な形の第 1 段（全重みを f32 で読む）そのもので、その RSS が research の起点になる。
    """
    from transformers import AutoTokenizer, UMT5EncoderModel

    snapshot = text_snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        snapshot, subfolder=UPSTREAM_SUBFOLDER, dtype=dtype
    ).eval()
    if encoder.dtype != dtype:
        raise Umt5ExportError(f"umT5 が {encoder.dtype} で読まれた（{dtype} を指定）")
    return tokenizer, encoder


def check_mask(
    model: str = DEFAULT_MODEL, dtype: str = "bf16", out: Path | None = None
) -> dict[str, Any]:
    """決定 4 の実モデルでの裏付け: 上流で「有効長だけ」と「512 + マスク」が一致するか。

    固定 4 プロンプトで上流の forward を 2 通りに回し、事前計算の資産（上流の
    `_get_t5_prompt_embeds` の bf16 の出力を f32 にしたもの）とも突き合わせる。bf16 では「512 +
    マスク」と資産の一致が、資産の経路をここで再現できていることの対照になる。f32 は丸めの床を
    見る対照（資産との差 = bf16 の経路の誤差）。`out` を渡すと両方の出力を {@link MASK_FILE} に
    書く（dtype をまたぐ比較は {@link compare_mask}）。
    """
    from wan.prompts import FIXED_PROMPTS
    from wan.text_embeds import ASSET_NAME, normalize, read_asset
    from wan.text_embeds import SERIES_NAME as EMBEDS_SERIES

    asset, _ = read_asset(SERIES_ROOT / EMBEDS_SERIES / ASSET_NAME)
    rows: list[dict[str, Any]] = []
    outputs: dict[str, torch.Tensor] = {}
    with MemoryMonitor() as monitor:
        with monitor.stage(f"load-{dtype}") as record:
            tokenizer, encoder = load_upstream(model, MASK_DTYPES[dtype])
            record.details["parameters"] = sum(p.numel() for p in encoder.parameters())
        for prompt in FIXED_PROMPTS:
            with monitor.stage(f"encode-{prompt.name}") as record:
                ids = torch.tensor(
                    [tokenizer([normalize(prompt.text)], add_special_tokens=True).input_ids[0]],
                    dtype=torch.long,
                )
                began = time.perf_counter()
                valid = umt5_patch.valid_output(encoder, ids)[0].float()
                valid_seconds = time.perf_counter() - began
                padded = umt5_patch.padded_output(encoder, ids)[0].float()
                expected = asset[prompt.name]
                row = {
                    "name": prompt.name,
                    "length": int(ids.shape[1]),
                    "valid_seconds": round(valid_seconds, 1),
                    "padded_seconds": round(time.perf_counter() - began - valid_seconds, 1),
                    "valid_vs_padded": dict(umt5_patch.compare(valid, padded)),
                    "padded_vs_asset": dict(umt5_patch.compare(padded, expected)),
                    "valid_vs_asset": dict(umt5_patch.compare(valid, expected)),
                }
                record.details["length"] = row["length"]
                rows.append(row)
                outputs[f"valid.{prompt.name}"] = valid.contiguous()
                outputs[f"padded.{prompt.name}"] = padded.contiguous()
                print(f"[mask] {json.dumps(row, ensure_ascii=False)}", flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        save_file(outputs, str(out / MASK_FILE.format(dtype=dtype)))
    return {"prompts": rows, "stages": [record.to_dict() for record in monitor.records]}


def compare_mask(directory: Path) -> list[dict[str, Any]]:
    """{@link check_mask} の bf16 と f32 の出力を突き合わせる（bf16 の 2 形の差を f32 と比べる）。

    `valid_error` / `padded_error` は bf16 の各形の f32 の同じ形に対する差で、bf16 の経路そのものの
    誤差。`valid_vs_padded` の差がこの誤差と同じ桁なら、2 形の差は丸めの順序の差で、どちらも
    同じ精度の bf16 の近似と言える。
    """
    files = {
        dtype: load_file(str(directory / MASK_FILE.format(dtype=dtype))) for dtype in MASK_DTYPES
    }
    names = sorted({key.split(".", 1)[1] for key in files["bf16"]})
    rows = []
    for name in names:
        bf16 = {form: files["bf16"][f"{form}.{name}"] for form in MASK_FORMS}
        f32 = {form: files["f32"][f"{form}.{name}"] for form in MASK_FORMS}
        rows.append(
            {
                "name": name,
                "length": int(bf16["valid"].shape[0]),
                "bf16_valid_vs_padded": dict(umt5_patch.compare(bf16["valid"], bf16["padded"])),
                "f32_valid_vs_padded": dict(umt5_patch.compare(f32["valid"], f32["padded"])),
                "valid_error": dict(umt5_patch.compare(bf16["valid"], f32["valid"])),
                "padded_error": dict(umt5_patch.compare(bf16["padded"], f32["padded"])),
            }
        )
    return rows


def prepare_summary(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """{@link prepare} を測り、材料の要約を返す（容器は書かない — モジュール doc）。"""
    with MemoryMonitor() as monitor:
        export = prepare(upstream_dir(model), monitor=monitor)
    return {
        "nodes": len(export.graph.nodes),
        "fixed": len(export.fixed),
        "plain": len(export.plain),
        "packed_bytes": sum(w.packed.numel() for w in export.fixed.values()),
        "stages": [record.to_dict() for record in monitor.records],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("prepare", "check-mask", "compare-mask"))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--dtype", default="bf16", choices=sorted(MASK_DTYPES), help="check-mask")
    parser.add_argument("--out", type=Path, help="check-mask が書く / compare-mask が読む席")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.command == "prepare":
        summary: Any = prepare_summary(args.model)
    elif args.command == "check-mask":
        summary = check_mask(args.model, args.dtype, args.out)
    else:
        if args.out is None:
            parser.error("compare-mask は --out（check-mask を 2 回書いた席）が要る")
        summary = compare_mask(args.out)
    print(
        json.dumps(
            {
                "command": args.command,
                "result": summary,
                "seconds": round(time.perf_counter() - started, 1),
                "torch_threads": torch.get_num_threads(),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
