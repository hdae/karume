"""Wan2.1 が使う umT5-XXL の encoder（`UMT5EncoderModel`）を i8 の S 形容器の材料にする書き手
（ADR 0119 段 10b — 決定 5 / 6。段 10c で育てる）。

上流は本家 `google/umt5-xxl` の pin（`wan.sources.UMT5_SOURCES` — pickle の分割形を sha256 の照合の
後に読む・ADR 0122 決定 1 / 2）。軸は 2 本（決定 3）: `--upstream`（umT5 の上流 — 重み・config・
出所）と `--model`（Wan の text 側の snapshot — `reference` のケースの id 列と `check-mask`）。

    uv run --group wan --inexact python -m wan.sources --umt5 xxl --fetch  # 本家の encoder の shard
    uv run --group wan --inexact python -m wan.umt5_export prepare       # 材料まで（書かない）
    uv run --group wan --inexact python -m wan.umt5_export write --check # 書いて系列と照合
    uv run --group wan --inexact python -m wan.umt5_export write         # 系列の容器を書き直す
    uv run --group wan --inexact python -m wan.umt5_export check-mask --dtype bf16 --out <席>
    uv run --group wan --inexact python -m wan.umt5_export check-mask --dtype f32 --out <席>
    uv run --group wan --inexact python -m wan.umt5_export compare-mask --out <席>
    uv run --group wan --inexact python -m wan.umt5_export reference     # 層逐次の参照 → golden
    uv run --group wan --inexact python -m wan.umt5_export write --intake inputs/umt5/<名前>
    uv run --group wan --inexact python -m wan.umt5_export reference --intake inputs/umt5/<名前>

`--intake` は umT5 の上流の軸を第三者の互換 encoder の取り込み（`wan.umt5_intake` — ADR 0122
決定 5）に替える。重み・config・出所は取り込みの記録 `intake.json` から引き、系列は
`umt5-xxl-<名前>-i8-dyn`。記録が名乗る元の dtype（BF16 / F32）だけを受け、BF16 は読みの時点で F32 へ
広げる（{@link Checkpoint}）。上流の表から読む経路（`--upstream`）は F32 だけのまま。ライセンス
未宣言の取り込みの `write` は `--allow-undeclared-license` の明示が要る（決定 6 — 容器の
ライセンス欄に未宣言の印を焼く）。

どのサブコマンドも段ごとの壁時間と RSS（{@link MemoryMonitor}）を JSON で出す。数値の記録は
research `2026-10-03-umt5-export-ram`。`reference`（段 10c）は書いた容器から重みを読んで CPU の
層逐次の参照を採り、golden を同じ席に書く（容器は書かない — 中身は {@link wan.umt5_reference}）。
受入れのケースは量子化なしの参照（checkpoint の F32 を 1 層ずつ — {@link checkpoint_weights}）も
採る。

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

## 容器を書く口

容器は {@link write_container} が `publish_model(..., graph_name=GRAPH_NAME)` で書く。部品名
{@link GRAPH_NAME} は umT5 の配布形（`wan.umt5_distribution.UMT5_WEIGHTS`）と Wan の配布形
（`wan.distribution.WAN_WEIGHTS` — 越境参照）の weights のキーで、全 family 横断の門
（`tests/test_graph_names.py`）が両方との一致を見る（ADR 0119 追記「段 10d の設計」D）。

`write --check` は
{@link write_container} の書いた容器を作業席に置き、系列の容器と全 part の sha256 で突き合わせる
（{@link assert_same_container} — 系列は置き換えない）。golden（`reference.*`）のメタは part 0 の
sha256 を持つので、`write` で容器を書き直したら `reference` で golden も書き直す。

MUST: transformers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import tempfile
import threading
import time
import weakref
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.dist import NOTICE_FILENAME
from karume.emit import FixedQuantizedWeight
from karume.ir import IrGraph
from karume.pipeline import export_module, publish_model
from karume.quantize import QUANT_MODULE_TYPES, channel_scale, iter_quant_targets, quantize_to_int8
from wan import umt5_patch, umt5_reference
from wan.sources import (
    DEFAULT_MODEL,
    SOURCES,
    UMT5_BIN_INDEX,
    UMT5_SOURCES,
    WAN21_MODELS,
    WanSourceError,
    pinned_umt5_shards,
    read_safetensors_header,
    text_snapshot,
    umt5_snapshot,
)
from wan.umt5_distribution import UMT5_DEFAULT_MODEL, intake_series_name, storage_kind
from wan.umt5_intake import (
    ALLOW_UNDECLARED_LICENSE_FLAG,
    INTAKE_DTYPES,
    Umt5Intake,
    Umt5IntakeError,
    assert_license_intent,
    assert_only_recorded_weights,
    load_intake,
)

#: i8 の系列（綴りは配布の規約 `<名>-<格納>-dyn`。名は出所 — 本家 `google/umt5-xxl` の encoder・
#: ADR 0122 決定 4）。
SERIES_NAME = "umt5-xxl-i8-dyn"
SERIES = SERIES_ROOT / SERIES_NAME

#: 系列の中の置き場（綴りは部品名 {@link GRAPH_NAME} と同じ — 規約であって導出ではない）。
COMPONENT_DIR = "text_encoder"
MODEL_FILE = "model.krm"

#: 容器のグラフ名 = 配布形の部品名 = weights のキー（container-v1 §2.1 — 配布側の綴りは
#: `wan.umt5_distribution.UMT5_ROLE`。一致は `tests/test_graph_names.py` の門）。
GRAPH_NAME = "text_encoder"

#: Wan の snapshot の中の umT5 のディレクトリ（`check-mask` だけが読む — 事前計算資産の経路を
#: 再現する対照で、容器の上流ではない — ADR 0122 決定 3）。
WAN_TEXT_ENCODER_SUBFOLDER = "text_encoder"

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


def _close_descriptors(descriptors: dict[Path, int]) -> None:
    for descriptor in descriptors.values():
        os.close(descriptor)
    descriptors.clear()


class PinnedBinShards:
    """pin した本家の pickle 分割形（`pytorch_model-0000N-of-0000M.bin` + 索引）の読み口
    （ADR 0122 決定 2）。

    組み立ての時点で、索引から導いた encoder の shard の集合を表と突き合わせ（`wan.sources.
    pinned_umt5_shards`）、表の各 shard を開いて記述子を握り、握った記述子から sha256 を照合する —
    1 本でも違えば 1 本も unpickle せずに落ちる。

    MUST: 読むときも握った記述子を `/proc/self/fd/<n>` で `torch.load` に渡す（path を開き直さ
    ない）。`/proc/self/fd/<n>` を開くと記述子の指す inode そのものが開くので、照合の後に path の
    先が差し替わっても（HF のキャッシュは別名に書いて rename する）、読むのは照合した内容になる。
    照合のたびに 10 GB 級の shard を読み直す形（開き直すなら再照合）は採らない。届かないのは、
    同じ inode をその場で書き換える者（書き込み権を持つ者）だけ — mmap で読む限り塞げない
    （塞ぐには全体を anon に写すことになり、RSS の山を戻す）。

    shard は読むたびに `torch.load(mmap=True, weights_only=True)` で開き、state はその場で捨てる
    （mmap は返した view が生きている間だけ残る）。state を握り続けると、触ったページが
    file-backed の RSS に乗ったまま残る（全 shard を握った形の実測: quantize の山 17.62 GiB のうち
    file が 11.6 GiB）。開く費用は 1 回 10〜20 ms（unpickle は shard の中の pickle だけ — 重みの
    ページは触らない）。
    """

    def __init__(self, directory: Path, shards: Mapping[str, str]) -> None:
        try:
            self.files = {
                key: directory / shard
                for key, shard in pinned_umt5_shards(directory / UMT5_BIN_INDEX, shards).items()
            }
        except (OSError, WanSourceError) as cause:
            raise Umt5ExportError(f"{directory}: {cause}") from cause
        self._descriptors: dict[Path, int] = {}
        # 記述子は持ち主が消えたときに閉じる（参照が切れた時点で決まって走る — 呼び手に close を
        # 求めない）。
        self._release = weakref.finalize(self, _close_descriptors, self._descriptors)
        try:
            for name, expected in sorted(shards.items()):
                path = directory / name
                if not path.is_file():
                    raise Umt5ExportError(f"{path} が無い（表の shard — 開く前に落とす）")
                self._descriptors[path] = os.open(path, os.O_RDONLY)
                with open(self._descriptors[path], "rb", closefd=False) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise Umt5ExportError(
                        f"{path} の sha256 {actual} が表の {expected} と違う — unpickle の前に"
                        "落とす（wan.sources.UMT5_SOURCES）"
                    )
        except BaseException:
            self._release()
            raise

    def tensors(self, keys: Sequence[str]) -> list[torch.Tensor]:
        """`keys` のテンソル（mmap の上の view・並びは `keys` と同じ — 呼び手は写してから捨てる）。

        shard ごとに 1 回だけ開くので、同じ shard の tied な別名は同じ storage を共有したまま返る
        （{@link Checkpoint.identical} の storage の照合）。
        """
        states: dict[Path, Mapping[str, torch.Tensor]] = {}
        for key in keys:
            path = self.files[key]
            if path not in states:
                states[path] = self._load(path)
        return [states[self.files[key]][key] for key in keys]

    def _load(self, path: Path) -> Mapping[str, torch.Tensor]:
        handle = f"/proc/self/fd/{self._descriptors[path]}"
        try:
            return torch.load(handle, map_location="cpu", mmap=True, weights_only=True)
        except RuntimeError as cause:
            # zip 形式でない（旧い torch.save の）pickle は mmap で開けない。zip 形式で落ちた
            # 例外は包まずにそのまま上げる（別の原因を名指しし損ねない）。
            if zipfile.is_zipfile(handle):
                raise
            raise Umt5ExportError(
                f"{path} が zip 形式の torch.save でない（mmap で開けない）"
            ) from cause


class PinnedSafetensors:
    """取り込み（ADR 0122 決定 5）の safetensors 1 本の読み口 — 記録が名指すファイルだけを開く。

    組み立ての時点でファイルを開いて記述子を握り、握った記述子から sha256 を記録の値と照合する。
    読み手（{@link Checkpoint}）は path を開き直さず {@link handle}（`/proc/self/fd/<n>`）を
    `safe_open` に渡すので、照合の後に path の先が差し替わっても、読むのは照合した内容になる
    （本家の pickle の読み口 {@link PinnedBinShards} と同じ考え方 — 届かないのは同じ inode を
    その場で書き換える者だけ）。

    MUST: 索引（`model.safetensors.index.json`）も同じ席の別のファイルも見ない — 索引の
    `weight_map` を先に引く読み口に渡すと、照合したファイルとは別のファイルが黙って読まれる。
    """

    def __init__(self, path: Path, sha256: str) -> None:
        if not path.is_file():
            raise Umt5ExportError(f"{path} が無い（intake.json が名指すファイル）")
        self.path = path
        self._descriptor = os.open(path, os.O_RDONLY)
        # 記述子は持ち主が消えたときに閉じる（{@link PinnedBinShards} と同じ）。
        self._release = weakref.finalize(self, os.close, self._descriptor)
        try:
            with open(self._descriptor, "rb", closefd=False) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != sha256:
                raise Umt5ExportError(
                    f"{path} の sha256 {actual} が intake.json の {sha256} と違う — 取り込み直す"
                )
        except BaseException:
            self._release()
            raise

    @property
    def handle(self) -> Path:
        """握った記述子の指す inode を開く path（開くたびに新しいファイル記述になる — 照合で
        進めた位置を引き継がない）。"""
        return Path(f"/proc/self/fd/{self._descriptor}")


#: 受ける格納 → その綴り（safetensors のヘッダ / torch の dtype の文字列）。
_ACCEPTED_DTYPES: Mapping[str, tuple[str, ...]] = {
    "F32": ("F32", "torch.float32"),
    "BF16": ("BF16", "torch.bfloat16"),
}


class Checkpoint:
    """上流の checkpoint からテンソルを 1 本ずつ・行の塊ごとに読む。

    全部を一度に載せないための口（決定 6）。MUST: 既定の格納は F32 だけを受ける — 上流の pin した
    checkpoint は F32（`total_size` 22,723,641,344 B）で、別の dtype なら丸めの出発点が変わる。

    `pinned` は取り込み（ADR 0122 決定 5）の読み口 {@link PinnedSafetensors} — 照合した記述子
    越しにそのファイルだけを読む（索引も同じ席の別のファイルも見ない）。

    `dtype="BF16"` は取り込み（ADR 0122 決定 5 — `intake.json` が元の dtype を名乗る）だけが渡す。
    受けるのは名乗った dtype だけで（全テンソルがその dtype でなければ落ちる）、読みの時点で F32 へ
    広げる（無損失 — bf16 の値は f32 で全部表せる）。読み手が見るのは常に F32 で、量子化はその
    決定的な関数なので、容器は「同じ値を F32 で持つ checkpoint」から書いた容器とバイト同一になる。

    既定は safetensors（分割形か単一形）。`index` / `single` はファイル名の綴り（既定は
    transformers の `save_pretrained`。diffusers の部品は `diffusion_pytorch_model…` — DiT の
    層逐次の参照 `wan.dit_reference` が渡す）。

    `shards`（ファイル名 → sha256）を渡したときだけ、pickle の分割形を {@link PinnedBinShards} で
    読む（ADR 0122 決定 2）。MUST: 渡すのは上流の表の行（`wan.sources.UMT5_SOURCES`）だけ —
    ディレクトリに `pytorch_model.bin.index.json` があるだけでは開かない（索引の有無で読み口を
    選ぶと、pin していない pickle も unpickle する口になる）。pickle の分割形は F32 だけ。
    """

    def __init__(
        self,
        directory: Path,
        *,
        index: str = CHECKPOINT_INDEX,
        single: str = CHECKPOINT_SINGLE,
        shards: Mapping[str, str] | None = None,
        pinned: PinnedSafetensors | None = None,
        dtype: str = "F32",
    ) -> None:
        if dtype not in _ACCEPTED_DTYPES:
            raise Umt5ExportError(f"dtype {dtype!r} は受けない（{sorted(_ACCEPTED_DTYPES)}）")
        if shards is not None and dtype != "F32":
            raise Umt5ExportError("pickle の分割形（本家の pin の行）は F32 だけを受ける")
        if shards is not None and pinned is not None:
            raise Umt5ExportError("shards と pinned は併用しない（読み口は 1 つ）")
        self._dtype = dtype
        self._bin: PinnedBinShards | None = None
        # 握った記述子を読み口の寿命だけ生かす（読みは記述子の handle 越し）。
        self._pinned_file = pinned
        if shards is not None:
            self._bin = PinnedBinShards(directory, shards)
            self._files = self._bin.files
        elif pinned is not None:
            # 索引の分岐を通らない（{@link PinnedSafetensors} の MUST）。
            with safe_open(str(pinned.handle), framework="pt") as handle:
                self._files = dict.fromkeys(handle.keys(), pinned.handle)
        elif (directory / index).is_file():
            weight_map = json.loads((directory / index).read_text(encoding="utf-8"))["weight_map"]
            self._files = {key: directory / name for key, name in weight_map.items()}
        elif (directory / single).is_file():
            path = directory / single
            with safe_open(str(path), framework="pt") as handle:
                self._files = dict.fromkeys(handle.keys(), path)
        else:
            raise Umt5ExportError(f"{directory} に {index} も {single} も無い")

    def names(self) -> frozenset[str]:
        return frozenset(self._files)

    def _pinned(self, *keys: str) -> list[torch.Tensor]:
        """pickle の分割形のテンソル（F32 を見てから — mmap の上の view。MUST: 呼び手は写すか
        形を見るだけにして、view を返さない — 握ると shard の mmap が閉じない）。"""
        assert self._bin is not None
        tensors = self._bin.tensors(keys)
        for key, tensor in zip(keys, tensors, strict=True):
            self._assert_dtype(key, str(tensor.dtype))
        return tensors

    def _same_storage(self, keys: Sequence[str]) -> bool:
        """pickle の分割形で `keys` が全部同じ storage（data pointer・offset・形・stride）か。"""
        first, *others = self._pinned(*keys)
        return all(
            tensor.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
            and tensor.storage_offset() == first.storage_offset()
            and tensor.shape == first.shape
            and tensor.stride() == first.stride()
            for tensor in others
        )

    def shape(self, key: str) -> list[int]:
        if self._bin is not None:
            (tensor,) = self._pinned(key)
            return list(tensor.shape)
        with safe_open(str(self._files[key]), framework="pt") as handle:
            view = handle.get_slice(key)
            self._assert_dtype(key, view.get_dtype())
            return list(view.get_shape())

    def read(self, key: str) -> torch.Tensor:
        """1 本を丸ごと（小さい重み — RMSNorm と相対位置の表）。"""
        if self._bin is not None:
            # 写す — mmap の上の view を返すと、容器や参照が shard の mmap を握り続ける。
            (tensor,) = self._pinned(key)
            return tensor.clone(memory_format=torch.contiguous_format)
        with safe_open(str(self._files[key]), framework="pt") as handle:
            tensor = handle.get_tensor(key)
        self._assert_dtype(key, str(tensor.dtype))
        return tensor.to(torch.float32)

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        """行 `[start, stop)` だけ（先頭の軸の切り出し — 残りの軸は丸ごと）。"""
        if self._bin is not None:
            (tensor,) = self._pinned(key)
            return tensor[start:stop].clone(memory_format=torch.contiguous_format)
        with safe_open(str(self._files[key]), framework="pt") as handle:
            view = handle.get_slice(key)
            self._assert_dtype(key, view.get_dtype())
            return view[start:stop].to(torch.float32)

    def identical(self, keys: Sequence[str], chunk_rows: int = CHUNK_ROWS) -> bool:
        """`keys` のテンソルが全部同じ値か（tied な別名の照合 — {@link checkpoint_keys}）。

        pickle の分割形で全部が同じ storage（data pointer・offset・形・stride）なら読まずに真。
        それ以外は形を見てから、行の塊ごとに F32 のビット列を比べる（`-0.0` と `0.0`・NaN の
        取り違えも拾う — 値の `==` ではなくビット）。
        """
        if self._bin is not None and self._same_storage(keys):
            return True
        shapes = [self.shape(key) for key in keys]
        if any(shape != shapes[0] for shape in shapes[1:]):
            return False
        for start in range(0, shapes[0][0], chunk_rows):
            stop = min(start + chunk_rows, shapes[0][0])
            first_rows = self.read_rows(keys[0], start, stop).view(torch.int32)
            for other in keys[1:]:
                if not torch.equal(
                    first_rows, self.read_rows(other, start, stop).view(torch.int32)
                ):
                    return False
        return True

    def _assert_dtype(self, key: str, dtype: str) -> None:
        """`key` の格納が受ける dtype（既定は F32・取り込みは `intake.json` の名乗り）か。

        読み手へ返す値は {@link read} / {@link read_rows} が F32 へ広げる（`to(float32)` は F32 なら
        同じ値のまま・BF16 なら無損失）。
        """
        if dtype not in _ACCEPTED_DTYPES[self._dtype]:
            if self._dtype == "F32":
                raise Umt5ExportError(f"checkpoint の '{key}' が {dtype}（F32 だけを受ける）")
            raise Umt5ExportError(
                f"checkpoint の '{key}' が {dtype}（intake.json が名乗る {self._dtype} だけを"
                "受ける）"
            )


def checkpoint_keys(
    model: nn.Module, wrapper_keys: Sequence[str], checkpoint: Checkpoint
) -> dict[str, str]:
    """ラッパのテンソルキー → checkpoint のキー。

    ラッパの `encoder` は上流の `encoder` そのものなので名前は同じ空間にある。tied な重み
    （`encoder.embed_tokens.weight` = `shared.weight`）は、同じ Parameter を指す別名の中から
    checkpoint に在るものを選ぶ。Wan の `save_pretrained` の形は片方の名前しか持たず、本家の索引は
    両方を持つ（ADR 0122 決定 2）。

    - 候補が 1 本: それを使う。
    - 候補が 2 本以上: 全候補が同じ値（{@link Checkpoint.identical} — 同じ storage かビット一致）
      なら、transformers の tied の宣言の先（`_tied_weights_keys` の値 — UMT5 では `shared.weight`）
      を選ぶ。

    MUST: 候補が 0 本・宣言の先が 1 つに決まらない・別名の値が 1 要素でも違う、は fail loudly
    （黙って別の重みを読まない）。
    """
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    by_name = {name: names for names in aliases.values() for name in names}
    tied = getattr(model, "_tied_weights_keys", None)
    targets = set(tied.values()) if isinstance(tied, Mapping) else set()
    available = checkpoint.names()
    mapping: dict[str, str] = {}
    for key in wrapper_keys:
        candidates = sorted(set(by_name.get(key, [key])) & available)
        if len(candidates) > 1:
            chosen = [name for name in candidates if name in targets]
            if len(chosen) != 1:
                raise Umt5ExportError(
                    f"'{key}' の checkpoint のキーが 1 つに決まらない: {candidates}（tied の宣言の"
                    f"先 {sorted(targets)} が候補に 1 つだけ在る形でない）"
                )
            if not checkpoint.identical(candidates):
                raise Umt5ExportError(
                    f"'{key}' の tied な別名 {candidates} の値が checkpoint の中で違う"
                    " — どちらを読んでも別の重みになる"
                )
            candidates = chosen
        if len(candidates) != 1:
            raise Umt5ExportError(f"'{key}' の checkpoint のキーが 1 つに決まらない: {candidates}")
        mapping[key] = candidates[0]
    return mapping


def _opened_checkpoint(
    directory: Path, shards: Mapping[str, str] | None, checkpoint: Checkpoint | None
) -> Checkpoint:
    """組み立て済みの読み口か、`directory`（と `shards`）から開いた読み口。"""
    if checkpoint is None:
        return Checkpoint(directory, shards=shards)
    if shards is not None:
        raise Umt5ExportError("checkpoint と shards は併用しない（読み口は 1 つ）")
    return checkpoint


def assert_encoder_keys(model: nn.Module, names: frozenset[str]) -> None:
    """checkpoint のキー集合が encoder の形（Wan の `text_encoder` と同じ — `shared.weight` と
    `encoder.*`）と過不足なく同じことを見る（取り込み — ADR 0122 決定 5）。

    期待値は config から組んだ meta の上流の parameter 名で、tied な別名の対
    （`shared.weight` / `encoder.embed_tokens.weight`）は 1 本と数える — 片方だけでも両方でも
    受ける（両方なら値の照合は {@link checkpoint_keys} の規則）。xxl の config では 242 本
    （対の両方を持つ本家の形は 243 本）。

    MUST: 欠け・余りは fail loudly — 旧い綴り（`blocks.N.attn.q` 等）や decoder を含むファイルは
    変換表を持たずに拒む（読む分だけを拾うと、別の構造の checkpoint の一部を黙って読む）。
    """
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    expected = {name for group in aliases.values() for name in group}
    missing = sorted(
        " / ".join(sorted(group)) for group in aliases.values() if not set(group) & names
    )
    extra = sorted(names - expected)
    if missing or extra:
        raise Umt5ExportError(
            f"checkpoint のキー集合が encoder の形（{len(aliases)} 本 — tied な別名の対は 1 本）と"
            f"違う: 欠け {len(missing)} 本 {missing[:4]} / 余り {len(extra)} 本 {extra[:4]}"
            "（旧い綴りや decoder を含むファイルは受けない — 変換表は持たない）"
        )


def _pinned_intake_file(directory: Path, file: str, sha256: str) -> PinnedSafetensors:
    """取り込み先 `directory` の記録のファイル `file` を握って照合した読み口（記録の外の重みの
    ファイルが同じ席に在れば、開く前に落とす — `wan.umt5_intake.assert_only_recorded_weights`）。"""
    try:
        assert_only_recorded_weights(directory, file)
    except Umt5IntakeError as cause:
        raise Umt5ExportError(str(cause)) from cause
    return PinnedSafetensors(directory / file, sha256)


def inspect_intake_checkpoint(directory: Path, file: str, sha256: str) -> str:
    """取り込んだ safetensors（`directory/file` — config は `directory`）を検査し、元の dtype を返す
    （`wan.umt5_intake` の手順 4 — ADR 0122 決定 5）。

    ファイルは書き手と同じ読み口（{@link PinnedSafetensors} — 握った記述子で `sha256` を照合し、
    ヘッダも重みも同じ記述子から読む）で開く。全テンソルが同じ dtype（`wan.umt5_intake.
    INTAKE_DTYPES` の BF16 か F32）・キー集合（{@link assert_encoder_keys}）・形（config から組んだ
    meta の上流と同じ）・tied な別名の対の値（{@link checkpoint_keys} — 両方を持つならビット一致で
    `shared.weight`）を見る。どれが違っても fail loudly。
    """
    path = directory / file
    pinned = _pinned_intake_file(directory, file, sha256)
    header = read_safetensors_header(pinned.handle)
    dtypes = sorted({str(entry["dtype"]) for entry in header.values()})
    if len(dtypes) != 1:
        raise Umt5ExportError(
            f"{path}: dtype が混在している（{dtypes}）— 全テンソルが同じ dtype だけを受ける"
        )
    (dtype,) = dtypes
    if dtype not in INTAKE_DTYPES:
        raise Umt5ExportError(
            f"{path}: dtype {dtype} は受けない（{list(INTAKE_DTYPES)} だけ — FP8 などは情報を"
            "失った派生）"
        )
    model = meta_text_encoder(directory)
    assert_encoder_keys(model, frozenset(header))
    shapes = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    wrong = {
        name: (entry["shape"], shapes[name])
        for name, entry in header.items()
        if entry["shape"] != shapes[name]
    }
    if wrong:
        raise Umt5ExportError(
            f"{path}: 形が config から組んだ encoder と違う（ファイル, 期待）: {wrong}"
        )
    checkpoint = Checkpoint(directory, pinned=pinned, dtype=dtype)
    wrapper = umt5_patch.Umt5EncoderTokens(model)
    checkpoint_keys(model, [name for name, _ in wrapper.named_parameters()], checkpoint)
    return dtype


def intake_checkpoint(intake: Umt5Intake) -> Checkpoint:
    """取り込み（`intake.json`）の safetensors の読み口（ADR 0122 決定 5）。

    記録が名指すファイルを握った記述子で sha256 照合し（{@link PinnedSafetensors} — 照合した
    内容と読む内容が同じ。取り込み先に記録の外の重みのファイルがあれば開く前に落とす）、記録が
    名乗る dtype だけを受ける読み口を組み、キー集合（{@link assert_encoder_keys}）を見る。形と
    tied な別名の値は {@link prepare} が読むときに見る。
    """
    pinned = _pinned_intake_file(intake.directory, intake.file.name, intake.file.sha256)
    checkpoint = Checkpoint(intake.directory, pinned=pinned, dtype=intake.dtype)
    assert_encoder_keys(meta_text_encoder(intake.directory), checkpoint.names())
    return checkpoint


def checkpoint_weights(
    directory: Path,
    shards: Mapping[str, str] | None = None,
    *,
    checkpoint: Checkpoint | None = None,
) -> umt5_reference.CheckpointWeights:
    """量子化しない重みの読み口（品質の記録の基準 — 段 10c）。

    容器のテンソルキー（ラッパの parameter 名）→ checkpoint のキーの対応は {@link prepare} と同じ
    {@link checkpoint_keys} で組む（上流は meta — 重みは読まない）。`shards` / `checkpoint` は
    {@link prepare} と同じ。
    """
    model = meta_text_encoder(directory)
    wrapper = umt5_patch.Umt5EncoderTokens(model)
    checkpoint = _opened_checkpoint(directory, shards, checkpoint)
    keys = [name for name, _ in wrapper.named_parameters()]
    return umt5_reference.CheckpointWeights(checkpoint, checkpoint_keys(model, keys, checkpoint))


# ---------------------------------------------------------------------------
# trace（重みを持たない上流）
# ---------------------------------------------------------------------------


def upstream_dir(upstream: str = UMT5_DEFAULT_MODEL) -> Path:
    """umT5 の上流の pin（`wan.sources.UMT5_SOURCES`）の置き場 — 本家はリポ直下に encoder を置く
    （取得済みでなければ `umt5_snapshot` が fail loudly）。"""
    return umt5_snapshot(upstream)


def upstream_shards(upstream: str = UMT5_DEFAULT_MODEL) -> Mapping[str, str]:
    """上流の表の行の shard の pin（{@link Checkpoint} の `shards` — pickle を開く唯一の口）。"""
    return UMT5_SOURCES[upstream].shards


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
    shards: Mapping[str, str] | None = None,
    checkpoint: Checkpoint | None = None,
    monitor: MemoryMonitor | None = None,
    chunk_rows: int = CHUNK_ROWS,
) -> Umt5Export:
    """trace → 量子化しない重みの読み込み → 量子化の対象の i8 化（段ごとに `monitor` で測る）。

    `directory` は config の置き場（グラフは config だけから組む）。重みの読み口は既定で
    `directory` の safetensors で、`shards` は {@link Checkpoint} と同じ（上流の表の行から組む
    ときだけ — {@link upstream_shards}）。`checkpoint` は組み立て済みの読み口（取り込み —
    {@link intake_checkpoint}）で、`shards` とは併用しない。
    """
    stage = monitor.stage if monitor is not None else _unmeasured
    with stage("trace") as record:
        model = meta_text_encoder(directory)
        wrapper = umt5_patch.Umt5EncoderTokens(model)
        graph, tensors = trace(wrapper)
        targets = quant_targets(wrapper)
        record.details["nodes"] = len(graph.nodes)
    checkpoint = _opened_checkpoint(directory, shards, checkpoint)
    weights = sorted(key for key, value in tensors.items() if value.is_meta)
    mapping = checkpoint_keys(model, weights, checkpoint)
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


def provenance(upstream: str = UMT5_DEFAULT_MODEL) -> Provenance:
    """容器へ焼く出所（umT5 の上流のライセンス識別子と pin した revision —
    `wan.sources.UMT5_SOURCES` が正本）。"""
    source = UMT5_SOURCES[upstream].source
    return Provenance(
        license=source.license, notice=NOTICE_FILENAME, upstream_revision=source.revision
    )


def intake_provenance(intake: Umt5Intake) -> Provenance:
    """取り込みの容器へ焼く出所（`intake.json` のライセンス — 未宣言なら印 — と pin した revision。
    ADR 0122 決定 5 / 6）。容器の provenance には dtype の欄が無いので、元の dtype は焼かない
    （記録と実験用ミラーのカードが名乗る）。"""
    return Provenance(
        license=intake.license, notice=NOTICE_FILENAME, upstream_revision=intake.revision
    )


def write_container(
    export: Umt5Export, path: Path, provenance_: Provenance | None = None
) -> IrGraph:
    """材料（{@link prepare} の戻り）を容器に書く（`path` は代表 path — 分割形の part 列になる）。

    格納は材料のまま（量子化の対象は `fixed` の packed + scale・残りは checkpoint の f32）で、
    グラフ名は部品名 {@link GRAPH_NAME}。出所は `provenance_`（省略時は本家の行 — {@link
    provenance}）。戻りは格納宣言を commit したグラフ（検収の表の入力 — {@link storage_by_kind}）。
    """
    return publish_model(
        path,
        export.graph,
        dict(export.tensors),
        provenance=provenance_ if provenance_ is not None else provenance(),
        graph_name=GRAPH_NAME,
        fixed_weights=export.fixed,
    )


def part_digests(path: Path) -> list[str]:
    """容器（代表 path）の part 列の sha256（part 0 から添字順）。"""
    digests = []
    for part in container_parts(path):
        with part.open("rb") as stream:
            digests.append(hashlib.file_digest(stream, "sha256").hexdigest())
    return digests


def assert_same_container(written: Path, existing: Path) -> None:
    """書いた容器が既存の容器と part 列ごとバイト同一であることを sha256 で見る。

    MUST: 違えば fail loudly（どの part が違うかを名指しする）。系列の容器は golden（part 0 の
    sha256 をメタに持つ）と配布形の両方の根なので、書き手が黙って別のバイト列を作る形を
    「同じ容器のつもり」で据えない。
    """
    actual = part_digests(written)
    expected = part_digests(existing)
    if len(actual) != len(expected):
        raise Umt5ExportError(
            f"part の本数が違う（書いた {len(actual)} 本 / 既存 {len(expected)} 本 — {existing}）"
        )
    differing = [index for index, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b]
    if differing:
        raise Umt5ExportError(
            f"part {differing} の sha256 が既存の容器と違う（{existing}）— 書き手か材料が"
            " 段 10b の容器と別のバイト列を作っている"
        )


@dataclass(frozen=True)
class EncoderInput:
    """umT5 の上流の軸 1 本ぶんの入力（ADR 0122 決定 3 — 本家の行 `--upstream` か取り込み
    `--intake`）。書き手の各サブコマンドはこれだけを見る（2 つの軸で経路を割らない）。"""

    #: config の置き場（グラフは config だけから組む）。
    directory: Path
    #: 重みの読み口を開く（本家の行は pickle の shard の sha256 の照合・取り込みはファイルの
    #: sha256 の照合とキー集合の門を含む）。
    open: Callable[[], Checkpoint]
    provenance: Provenance
    #: 系列の親（`<系列>/text_encoder/model.krm`）。
    series: Path
    #: golden のメタに残す軸の記録（{@link reference_axes} の `encoder`）。
    axis: Mapping[str, str]


def upstream_input(upstream: str = UMT5_DEFAULT_MODEL) -> EncoderInput:
    """本家の行（`wan.sources.UMT5_SOURCES`）の入力（取得済みでなければ fail loudly）。"""
    directory = upstream_dir(upstream)
    return EncoderInput(
        directory=directory,
        open=lambda: Checkpoint(directory, shards=upstream_shards(upstream)),
        provenance=provenance(upstream),
        series=SERIES,
        axis=upstream_axis(upstream),
    )


def intake_input(intake: Umt5Intake) -> EncoderInput:
    """取り込み（`intake.json`）の入力。系列は `umt5-xxl-<名前>-i8-dyn`（ADR 0122 決定 5 —
    綴りは `wan.umt5_distribution.intake_series_name`）。"""
    return EncoderInput(
        directory=intake.directory,
        open=lambda: intake_checkpoint(intake),
        provenance=intake_provenance(intake),
        series=SERIES_ROOT / intake_series_name(intake.name),
        axis={
            "intake": intake.name,
            "repo": intake.repo,
            "revision": intake.revision,
            "file": intake.file.name,
            "dtype": intake.dtype,
        },
    )


def write_series(upstream: str = UMT5_DEFAULT_MODEL, *, check: bool) -> dict[str, Any]:
    """本家の行から系列の容器を書く（{@link write_encoder}）。"""
    return write_encoder(upstream_input(upstream), check=check)


def write_intake_series(
    directory: Path, *, check: bool, allow_undeclared_license: bool
) -> dict[str, Any]:
    """取り込みから系列の容器を書く（{@link write_encoder}）。

    MUST: ライセンス未宣言の取り込みは、明示（`--allow-undeclared-license`）が無ければ 1 バイトも
    書く前に落とす（ADR 0122 決定 6）。明示があれば容器のライセンス欄に未宣言の印を焼く。
    """
    intake = load_intake(directory)
    try:
        assert_license_intent(intake, allowed=allow_undeclared_license)
    except Umt5IntakeError as cause:
        raise Umt5ExportError(str(cause)) from cause
    return write_encoder(intake_input(intake), check=check)


def write_encoder(source: EncoderInput, *, check: bool) -> dict[str, Any]:
    """系列の容器を書く（`check` なら作業席に書いて既存と照合するだけで、系列は置き換えない）。

    書き直す回は系列の部品ディレクトリを丸ごと差し替える（`staged_publication` — golden の
    `reference.*` も消えるので、続けて `reference` で書き直す）。
    """
    target = source.series / COMPONENT_DIR
    with MemoryMonitor() as monitor:
        export = prepare(source.directory, checkpoint=source.open(), monitor=monitor)
        with monitor.stage("write"):
            if check:
                with tempfile.TemporaryDirectory(dir=source.series, prefix=".check-") as scratch:
                    written = Path(scratch) / MODEL_FILE
                    graph = write_container(export, written, source.provenance)
                    assert_same_container(written, target / MODEL_FILE)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with staged_publication(target) as staged:
                    staged.mkdir()
                    graph = write_container(export, staged / MODEL_FILE, source.provenance)
    return {
        "check": check,
        "container": str(target / MODEL_FILE),
        "parts": len(container_parts(target / MODEL_FILE)),
        "storage_by_kind": storage_by_kind(graph),
        "stages": [record.to_dict() for record in monitor.records],
    }


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
    """Wan の pin した revision のトークナイザと umT5（`dtype`・CPU・eval — Wan の text 側の
    snapshot の軸・ADR 0122 決定 3）。

    資産の経路（`wan.text_embeds.load_text_encoder` — bf16）と同じ呼び方で、dtype だけを選ぶ。
    f32 は素直な形の第 1 段（全重みを f32 で読む）そのもので、その RSS が research の起点になる。
    """
    from transformers import AutoTokenizer, UMT5EncoderModel

    snapshot = text_snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        snapshot, subfolder=WAN_TEXT_ENCODER_SUBFOLDER, dtype=dtype
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


def prepare_summary(upstream: str = UMT5_DEFAULT_MODEL) -> dict[str, Any]:
    """本家の行の {@link prepare_encoder}。"""
    return prepare_encoder(upstream_input(upstream))


def prepare_intake_summary(directory: Path) -> dict[str, Any]:
    """取り込みの {@link prepare_encoder}（容器を書かないので未宣言の門は掛けない）。"""
    return prepare_encoder(intake_input(load_intake(directory)))


def prepare_encoder(source: EncoderInput) -> dict[str, Any]:
    """{@link prepare} を測り、材料の要約を返す（容器は書かない — モジュール doc）。"""
    with MemoryMonitor() as monitor:
        export = prepare(source.directory, checkpoint=source.open(), monitor=monitor)
    return {
        "nodes": len(export.graph.nodes),
        "fixed": len(export.fixed),
        "plain": len(export.plain),
        "packed_bytes": sum(w.packed.numel() for w in export.fixed.values()),
        "stages": [record.to_dict() for record in monitor.records],
    }


def reference_axes(upstream: str, wan_model: str) -> dict[str, dict[str, str]]:
    """golden のメタに残す 2 本の軸（ADR 0122 決定 3 — `reference` は両方を混ぜて使う唯一の
    サブコマンド）。

    - `encoder`: umT5 の上流（config と量子化しない重み — `wan.sources.UMT5_SOURCES`）
    - `cases`: ケースの id 列を採った Wan の text 側の snapshot（トークナイザ —
      `wan.sources.SOURCES`）
    """
    return _axes(upstream_axis(upstream), wan_model)


def upstream_axis(upstream: str) -> dict[str, str]:
    """本家の行の軸の記録（{@link EncoderInput} の `axis` — 取得済みでなくても組める）。"""
    encoder = UMT5_SOURCES[upstream].source
    return {"model": upstream, "repo": encoder.repo, "revision": encoder.revision}


def _axes(encoder: Mapping[str, str], wan_model: str) -> dict[str, dict[str, str]]:
    cases = SOURCES[wan_model]
    return {
        "encoder": dict(encoder),
        "cases": {
            "model": wan_model,
            "repo": cases.repo,
            "revision": cases.revision,
            "subfolder": "tokenizer",
        },
    }


def reference_summary(
    upstream: str = UMT5_DEFAULT_MODEL, wan_model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """本家の行の {@link reference_encoder}。"""
    return reference_encoder(upstream_input(upstream), wan_model)


def reference_intake_summary(directory: Path, wan_model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """取り込みの {@link reference_encoder}（量子化しない参照は取り込みの重みを F32 へ
    広げた値）。"""
    return reference_encoder(intake_input(load_intake(directory)), wan_model)


def reference_encoder(source: EncoderInput, wan_model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """書いた i8 系列の容器から層逐次の CPU 参照（f64 / f32）を採り、golden を同じ席に
    書く（段 10c — 容器は書かない）。

    ケースの id 列は Wan のトークナイザの経路（10a の fixture と同じ — 軸 `wan_model`）で採る。
    config と、受入れ（固定 4 本）の量子化しない重み（checkpoint の F32 — {@link
    checkpoint_weights}）は umT5 の上流の軸（`source` — 本家の行か取り込み）から読む。bf16 の
    事前計算資産との差も要約に出す（記録だけ）。どちらの軸の値を使ったかは golden のメタ
    （{@link reference_axes}・取り込みは {@link intake_input} の `axis`）。
    """
    from transformers import UMT5Config

    from wan.text_embeds import ASSET_NAME, read_asset
    from wan.text_embeds import SERIES_NAME as EMBEDS_SERIES

    component = source.series / COMPONENT_DIR
    directory = source.directory
    with MemoryMonitor() as monitor:
        with monitor.stage("cases") as record:
            cases = umt5_reference.reference_cases(umt5_reference.upstream_encoder(wan_model))
            record.details["lengths"] = {case.name: len(case.ids) for case in cases}
        embeddings, _ = read_asset(SERIES_ROOT / EMBEDS_SERIES / ASSET_NAME)
        result = umt5_reference.write_references(
            container=component / MODEL_FILE,
            out_dir=component,
            config=UMT5Config.from_pretrained(directory),
            cases=cases,
            stage=monitor.stage,
            unquantized=checkpoint_weights(directory, checkpoint=source.open()),
            axes=_axes(source.axis, wan_model),
            embeddings=embeddings,
        )
    return {**result, "stages": [record.to_dict() for record in monitor.records]}


#: サブコマンドごとに効く引数（argparse の dest）。MUST: これ以外の引数を明示したら拒む — 黙って
#: 無視すると、効かない軸（例 `write --model`）を指定したつもりの回が別の入力で走る。
COMMAND_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "prepare": frozenset({"upstream", "intake"}),
    "write": frozenset({"upstream", "intake", "check", "allow_undeclared_license"}),
    "check-mask": frozenset({"model", "dtype", "out"}),
    "compare-mask": frozenset({"out"}),
    "reference": frozenset({"upstream", "intake", "model"}),
}


def _flag(dest: str) -> str:
    """argparse の dest → 引数の綴り（`allow_undeclared_license` →
    `--allow-undeclared-license`）。"""
    return "--" + dest.replace("_", "-")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=tuple(COMMAND_ARGUMENTS))
    # 選択引数の既定は None（明示と省略を見分けて、効かない引数の明示を拒むため）— 実の既定値は
    # 下で埋める。
    parser.add_argument(
        "--upstream",
        choices=sorted(UMT5_SOURCES),
        help=f"umT5 の上流（prepare / write の重み・config・出所、reference の config と"
        f"量子化しない重み — 既定 {UMT5_DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--model",
        choices=WAN21_MODELS,
        help="Wan の text 側の snapshot（reference のケースの id 列・check-mask — 兄弟の台本"
        f" text_embeds / umt5_tokenizer / umt5_host_fixture の --model と同じ軸 — 既定"
        f" {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=None,
        help="write: 作業席に書いて系列の容器と sha256 で照合する（系列は置き換えない）",
    )
    parser.add_argument("--dtype", choices=sorted(MASK_DTYPES), help="check-mask（既定 bf16）")
    parser.add_argument("--out", type=Path, help="check-mask が書く / compare-mask が読む席")
    parser.add_argument(
        "--intake",
        type=Path,
        help="umT5 の上流の軸を取り込み（inputs/umt5/<名前>/ — wan.umt5_intake）にする（prepare /"
        " write / reference — --upstream とは併用しない。系列は umt5-xxl-<名前>-i8-dyn）",
    )
    parser.add_argument(
        ALLOW_UNDECLARED_LICENSE_FLAG,
        dest="allow_undeclared_license",
        action="store_true",
        default=None,
        help="write --intake: ライセンス未宣言の取り込みから、未宣言の印を焼いた容器を書く（手元の"
        "実験用だけ — ADR 0122 決定 6）",
    )
    args = parser.parse_args(argv)
    accepted = COMMAND_ARGUMENTS[args.command]
    for name in sorted(frozenset().union(*COMMAND_ARGUMENTS.values())):
        if getattr(args, name) is not None and name not in accepted:
            parser.error(
                f"{_flag(name)} は {args.command} に効かない（効くのは"
                f" {', '.join(_flag(other) for other in sorted(accepted))}）"
            )
    if args.intake is not None and args.upstream is not None:
        parser.error("--intake と --upstream は同じ軸（umT5 の上流）— どちらか 1 つ")
    if args.allow_undeclared_license is not None and args.intake is None:
        parser.error(f"{ALLOW_UNDECLARED_LICENSE_FLAG} は --intake の write にだけ効く")
    upstream = args.upstream if args.upstream is not None else UMT5_DEFAULT_MODEL
    model = args.model if args.model is not None else DEFAULT_MODEL
    started = time.perf_counter()
    if args.command == "prepare":
        summary: Any = (
            prepare_summary(upstream)
            if args.intake is None
            else prepare_intake_summary(args.intake)
        )
    elif args.command == "write":
        if args.intake is None:
            summary = write_series(upstream, check=args.check is True)
        else:
            summary = write_intake_series(
                args.intake,
                check=args.check is True,
                allow_undeclared_license=args.allow_undeclared_license is True,
            )
    elif args.command == "check-mask":
        summary = check_mask(model, args.dtype if args.dtype is not None else "bf16", args.out)
    elif args.command == "reference":
        summary = (
            reference_summary(upstream, model)
            if args.intake is None
            else reference_intake_summary(args.intake, model)
        )
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
