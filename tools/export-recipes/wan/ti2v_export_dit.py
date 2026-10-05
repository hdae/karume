"""Wan2.2 TI2V-5B の DiT を I2V 対応の S 形の IR（`krm`・i8 系列）と小さい S の golden へ
書き出す台本（ADR 0121 段 1 — 決定 2 / 3 / 4 / 5 / 7）。

    uv run --group wan --inexact python -m wan.ti2v_export_dit inspect        # trace と構造の検査
    uv run --group wan --inexact python -m wan.ti2v_export_dit write          # 容器 + golden
    uv run --group wan --inexact python -m wan.ti2v_export_dit write-full     # 実寸の golden を足す
    uv run --group wan --inexact python -m wan.ti2v_export_dit observe-quant  # 量子化の観測
    uv run --group wan --inexact python -m wan.ti2v_export_dit eager-full     # 全量の eager 同値

どのサブコマンドも要約の JSON を標準出力と `--out`（既定
`outputs/bench/karume-wan2.2/<日付>_dit-export/` — 系列とは別）の `<command>.json` に書き、
欄 `passed` が偽なら非 0 で終わる。

## グラフ（決定 3）

`wan.dit_patch.WanDitTokensTi2v`: 2.1 の 5 入力（トークン・`timesteps_proj`・文脈・RoPE の
2 本）の後ろに `timesteps_proj_condition [1,256]` と `condition_mask [1,S,1]`（bool）を足した
7 入力。時刻の MLP を M = 1 で 2 回回し、変調の各成分を消費の直前の `where` で選ぶ。
`Dim("S")` の上限は 27,280（1280×704×121 — 受理集合は TS が絞る）。

## export の形（決定 7 — ADR 0119 段 10b の形・exporter core の変更なし）

trace は重みを持たない meta の上流で回し（{@link trace}）、linear 307 本（patch 埋め込みの
Linear を含む）は checkpoint から行の塊ごとに i8 にして `fixed_weights` で渡す。bias・norm・
`scale_shift_table` は checkpoint の f32 のまま（`wan.dit_probe.prepare` の材料作りを
{@link trace} で回す — 段 0 の実測の山 5.84 GiB）。容器には資産 `rope_base`（2.1 と同じ席）が
入る。

## IR の検査（{@link inspect_graph} — 段 1 の検収）

- linear のノードが「量子化の対象 + 時刻の MLP の 3 本」（5B で 310 — 決定 2 の dispatch の
  本数）
- i8 の重みが量子化の対象とちょうど同じ（5B で 307）で、残りの重み（bias・norm・
  `scale_shift_table`）は f32
- `where` がブロックあたり 6 本 + head の 2 本で、どれも出力 `[1,S,dim]` の消費者が**直後の
  1 ノードだけ**（attention をまたいで生きる `where` の出力が無い — {@link where_placement}）
- グラフ入力が 7 本の宣言どおりで、`condition_mask` が bool `[1,S,1]`

## golden（段 1 — S = 192 の T2V の形と I2V の形）

ケースは {@link CASES}（決定用 6 本 = T2V の形 3 + I2V の形 3・受入れ 4 本 = T2V 2 + I2V 2 —
段 2 の r 門の本数）。文脈は Wan2.1 の事前計算の埋め込み資産の固定 4 プロンプト（合成の乱数は 5B の
CPU f32 と f64 の差を増幅する — {@link Ti2vCase} の doc）、潜在は seed からの乱数。重みは
**書いたばかりの容器**から層逐次で読む
（`dit_reference.ContainerDitWeights` — i8 は packed × scale を f32 で）。f32 は上流の
`from_pretrained` と同じ番地のずれで回す（f32 の GEMM の最終ビットが番地に依る —
`wan.dit_reference` のモジュール doc）。

- `reference.<case>`: `latents` / `timestep`（i32・生成側）/ `condition_timestep`（i32・条件側 —
  I2V は 0・T2V は生成側と同じ）/ `output`（CPU f32）/ `output.f64`（活性も f64・f32 へ丸めた値）。
  参照の時刻の表し方は決定 4: T2V は上流の 1 次元 timestep の forward、I2V は参照ラッパ（時刻の
  MLP を値ごとに M = 1 — `dit_patch.reference_dit` の `condition`）
- `io.<case>`: グラフ入力 7 本（`input.*` — 条件マスクは u32 の 0 / 1）とパッチ後の eager の出力
  `output.0`（同じ層逐次の重みで回した S 形のラッパ）

eager 同値の門（`export_dit.eager_failures` と同じ判定）: T2V の形は上流の 1 次元 timestep の
経路と、I2V の形は参照ラッパと trunk がビット一致。1 本でも外れたら作業席の中で止める（系列は
1 バイトも変わらない）。両側が同じ層逐次の重みを読むので、読み込みに共通の誤りはこの門では見えない
（下の `eager-full` が独立した重みで見る）。diffusers の 2 次元 timestep の経路（M = S の linear）
との差は観測として決定用ケースの要約に並べる（門ではない — 決定 4）。

## 実寸の golden（段 2 — `write-full`）

ケースは {@link FULL_CASES}（数値の門の 2 つの形 — 潜在 `[48,21,30,52]`〈832×480・81 フレーム・
S = 8,190〉と `[48,9,44,80]`〈1280×704・33 フレーム・S = 7,920〉。ADR 0121 追記「裁定 1 の確定」の
受理の上限から採った形で、832×480 は追記（2026-10-05）「受理寸法を公式の 2 寸法へ」で受理の外に
なった。DiT のグラフは寸法に依らないので、受理寸法と無関係な数値の門の形として使う）。決定用
6 本（形ごとに 3 本・T2V 3 + I2V 3）と受入れ 8 本（形ごとに T2V 2 + I2V 2）。1 ケースの中身は
S = 192 の golden と同じ（{@link golden_case} — f32 / f64 の参照・パッチ後の eager・eager 同値の
門）で、重みは**据えた容器**から層逐次で読む（容器は書き直さない）。M = S の経路の観測は採らない
（{@link OBSERVED_ROLES} は S = 192 の決定用だけ — 実寸で 1 ケース 1 forward 延びる）。

- **足し方**: 1 ケースずつ、系列の直下の作業ディレクトリ（`.write-full-*` — 部品ディレクトリの外）へ
  書いてから、2 本のファイルを部品ディレクトリへ `os.replace` で移す（同じファイルシステムの中の
  rename — 書きかけのファイルが部品ディレクトリに現れない）。門に落ちたケースは何も移さない。
- **再開**: 2 本とも揃ったケースは飛ばす。飛ばす前に、据わっている `input.*`（7 本）と
  `latents` / `timestep` / `condition_timestep` を今のケースの表から組み直した値とビットで比べ、
  違えば止める（{@link stored_case_state} — 表を変えた後の古い golden を「済み」に数えない）。
  1 本だけあるケース（移す途中で落ちた）は書き直す。前の実行が残した作業ディレクトリは始めに消す。
- **進捗**: ケースごとに `[write-full]` の行（済み・残り・残りの見込み — 見込みは済んだケースの
  平均、無ければ段 0 の実測から {@link FULL_CASE_SECONDS}）と、参照の f32 / f64 のブロックごとの
  `[block]` の行（`dit_reference.PROGRESS_TOKENS` 以上の S）を出す。RAM の山は段 0 の実測で
  約 5 GiB（{@link FULL_PEAK_BYTES} — 書く前に MemAvailable を確かめる）。
- MUST: `write` は部品ディレクトリを作業席ごと据え替えるので、`write` の後は実寸の golden が消える。
  `write` を回し直したら `write-full` も回し直す。

## 独立した重みでの eager 同値（`eager-full` — 決定 4 の形）

`write` の門は、パッチ後のグラフも参照も**同じ容器から読む層逐次の仕組み**で回す（RAM 8 GiB の枠に
収めるため）。だから読み込みに共通の誤り（容器の読み口・層逐次の列の組み方）は両側で相殺され、門を
すり抜けうる。`eager-full`（{@link eager_full}）はその穴を、決定 4 が書く形で 1 回だけ塞ぐ:

- 上流を `from_pretrained` で f32 の全量 1 本として読み、i8 の fake-quant をパッチ後のラッパ経由で
  掛ける（2.1 の `export_dit` と同じ — 容器も層逐次も通らない）
- 各ケースで、パッチ後のグラフ（{@link dit_patch.WanDitTokensTi2v} — T2V / I2V の形）の eager を
  上流（T2V は 1 次元 timestep の forward・I2V は参照ラッパ）と比べる（`export_dit.eager_failures`
  と同じ判定 — trunk のビット一致）
- 据えた golden とも突き合わせる: 上流の参照 = `reference.<case>` の `output`、パッチ後の出力 =
  `io.<case>` の `output.0`（どちらもビット一致 — golden は容器 + 上流の番地のずれで採った値で、
  層逐次の f32 が上流の f32 eager とビット一致する決定 7 の主張に依る。NOTE: MKL の振る舞いで仕様の
  保証ではない — `wan.dit_reference` のモジュール doc）。グラフ入力 7 本も golden と同じであること
  （同じケースを組めていること）を見る。

RAM の山は約 21 GiB（f32 全量 18.63 GiB + 基礎分）。上流を読む前に MemAvailable を読み、見込み
（`dit_probe.eager_memory` の f32 + 基礎分）に余白を足した量に足りなければ止まる
（`dit_probe.require_available`）。据えた golden が揃っていなければ、それも読む前に止まる。

## 量子化の観測（決定 5 — `observe-quant`）

量子化なしの CPU f32 参照に対する、重みだけ i8 の CPU f32 参照の差（max abs ÷ max ref）を
S = 192 の決定用ケースで 5B と 1.3B の両方について、同じプロンプトの割り当てで採る（読み口は
checkpoint の F32 と、1 本ずつ i8 の fake-quant を掛けた値 —
`dit_reference.CheckpointDitWeights`）。5B の最悪が 1.3B の最悪の 2 倍（目安）を超えたら
`escalate` が真になる（段 2 の前に裁定へ上げる）。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from safetensors.torch import save, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import BENCH_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import AssetInput, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_module, publish_model
from karume.quantize import QUANT_MODULE_TYPES, iter_quant_targets
from wan import dit_patch, dit_probe, dit_reference, export_dit
from wan.sources import local_snapshot
from wan.umt5_export import MemoryMonitor

#: 既定のモデル（Wan2.2 TI2V-5B — ADR 0121）。
DEFAULT_MODEL = "ti2v-5b"

#: i8 系列（決定 7 の系列名 — 接尾辞 `-dyn` は ADR 0077 の慣例）。
SERIES_NAME = "wan2.2-ti2v-5b-i8-dyn"
SERIES = SERIES_ROOT / SERIES_NAME

#: 容器のグラフ名 = 配布形の部品名 = weights のキー（container-v1 §2.1 — 2.1 の DiT と同じ席。
#: 一致は `tests/test_graph_names.py` の門）。系列の中の置き場も同じ綴り
#: （規約であって導出ではない）。
TARGET = "transformer"
MODEL_FILE = "model.krm"

#: グラフ入力の名前（ラッパの forward の引数名がそのまま IR の入力名になる —
#: 先頭 5 本は 2.1 と同じ）。
INPUT_NAMES = (*export_dit.INPUT_NAMES, "timesteps_proj_condition", "condition_mask")

#: `Dim("S")` の上限（1280×704×121 = 27,280 — 決定 3）と下限（2.1 と同じ）。
SYM_MAX = 27_280
SYM_MIN = export_dit.DIT_SYM_MIN

#: 時刻の MLP の linear の本数（`time_embedder` の 2 本 + `time_proj`）— M = 1 で 2 回回すので、
#: linear のノードはこの本数だけ量子化の対象より多い（決定 2）。
TIME_MLP_LINEARS = 3

#: 変調の `where` の本数（ブロックあたり 6 成分・head の 2 成分 — 決定 3）。
WHERE_PER_BLOCK = 6
WHERE_HEAD = 2

#: 5B の期待値（決定 2 — i8 の重みの本数 / 1 パスの linear の dispatch の本数）。
EXPECTED_5B = {"i8_weights": 307, "linear_nodes": 310}

#: golden の乱数の seed の起点（決定用。受入れは別系統の seed を直に持つ — 1.3B の
#: `export_dit.SEED` + 0〜25 とも受入れの 777001〜とも重ならない値）。
SEED = 20262000

#: 置き場の配布名（`outputs/bench/<配布名>/` — `docs/assets-layout.md`）。
BENCH_NAME = "karume-wan2.2"


class Ti2vExportError(RuntimeError):
    """グラフ・重み・golden が段 1 の検収の形から外れた。"""


Form = Literal["t2v", "i2v"]


@dataclass(frozen=True)
class Ti2vCase:
    """golden 1 ケースの条件（T2V / I2V・潜在の形・生成側の timestep・文脈のプロンプト・seed）。

    文脈は Wan2.1 の事前計算の埋め込み資産（同じ umT5 の出力 — ADR 0121 決定 9 が 2.2 にも
    そのまま写す資産）のプロンプト 1 本（`dit_probe.probe_inputs` — 有効長の後ろをゼロで 512 行
    まで埋める）。潜在は seed からの乱数。

    なぜ合成の乱数の文脈（2.1 の golden の作り方）を使わないか（段 1 の切り分け — 結果は
    `outputs/bench/karume-wan2.2/2026-10-04_dit-f32-f64-diagnosis/`）: N(0,1) の乱数は実の umT5
    の出力（std 約 0.08）の約 12 倍の大きさの分布の外で、5B ではそれだけで CPU f32 と f64 の差が
    層ごとに増幅される。S = 1,170・t = 999 の出力の比は合成 1.38e-2 / 実プロンプト 1.24e-6、
    S = 192 でも合成は 5.3e-4（1.3B の同じ作り方は 2e-6〜1.6e-4）。r 門の分母（CPU f32 の f64 に
    対する比）がこの増幅で飽和すると、微妙な故障注入が帯に紛れる。
    """

    role: str
    form: Form
    latent_shape: tuple[int, int, int]
    timestep: int
    text: str
    seed: int

    def name(self, patch_size: tuple[int, int, int]) -> str:
        return f"{self.role}-{self.form}-s{self.tokens(patch_size):05d}-t{self.timestep:04d}"

    def tokens(self, patch_size: tuple[int, int, int]) -> int:
        frames, height, width = self.latent_shape
        return (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])

    def spec(self) -> export_dit.CaseSpec:
        """潜在の乱数を引く形（`export_dit.case_inputs` — 潜在を先に引くので、合成の text の有効長
        〈ここでは 1〉は潜在の値に効かない。文脈は `probe_inputs` がプロンプトに替える）。"""
        return export_dit.CaseSpec(
            self.role, self.latent_shape, self.timestep, 1, self.seed, blocks=False
        )


#: golden のケース（S = 192・潜在 `[48,F,H,W]`）。決定用は timestep を 999〜113 に散らし、格子は正方
#: （3·8·8）と非正方（2·8·12 / 1·8·24 / 2·12·8 / 6·4·8 / 4·4·12）を混ぜる（h と w の取り違えは
#: 正方では隠れる）。I2V の形は F' ≥ 2（先頭の潜在フレームだけが条件 — F' = 1 だと全トークンが条件に
#: なる）。プロンプトは固定 4 本（正 3 本 + negative — CFG の 2 本目の文脈）を形と役割に散らす。
#: MUST: `accept`（受入れ）の結果を見て `band`（決定用）を足し引きしない（帯の決定と受入れの独立 —
#: 2.1 の `export_dit.CASES` と同じ規律）。
CASES: tuple[Ti2vCase, ...] = (
    Ti2vCase("band", "t2v", (3, 16, 16), 999, "boxing-cats", SEED + 0),
    Ti2vCase("band", "t2v", (2, 16, 24), 500, "ferret", SEED + 1),
    Ti2vCase("band", "t2v", (1, 16, 48), 250, "cat-dog-baking", SEED + 2),
    Ti2vCase("band", "i2v", (3, 16, 16), 999, "negative", SEED + 3),
    Ti2vCase("band", "i2v", (2, 24, 16), 750, "boxing-cats", SEED + 4),
    Ti2vCase("band", "i2v", (6, 8, 16), 113, "ferret", SEED + 5),
    Ti2vCase("accept", "t2v", (2, 16, 24), 600, "cat-dog-baking", 778001),
    Ti2vCase("accept", "t2v", (4, 8, 24), 30, "negative", 778002),
    Ti2vCase("accept", "i2v", (3, 16, 16), 400, "ferret", 778003),
    Ti2vCase("accept", "i2v", (4, 8, 24), 900, "cat-dog-baking", 778004),
)

#: trace の例示入力の格子（先頭のケース — 値は trace に効かない・S = 192 は 0 / 1 特殊化を
#: 踏まない）。
TRACE_LATENT = CASES[0].latent_shape

#: 実寸の 2 つの形（数値の門の形 — ADR 0121 追記「裁定 1 の確定」の受理の上限から採った）:
#: 832×480・81 フレーム = 潜在 `[48,21,30,52]`〈S = 21·15·26 = 8,190〉と 1280×704・33 フレーム =
#: `[48,9,44,80]`〈S = 9·22·40 = 7,920〉。832×480 は追記（2026-10-05）で受理の外になったが、DiT の
#: グラフは寸法に依らないので、受理寸法と無関係な数値の門の形として使う。
FULL_LATENTS: tuple[tuple[int, int, int], ...] = ((21, 30, 52), (9, 44, 80))

#: 実寸の golden のケース（段 2 の r 門 — モジュール doc「実寸の golden」）。決定用（`full-band`）は
#: 形ごとに 3 本で、T2V 3 本（t 999 / 113 / 750）と I2V 3 本（t 500 / 999 / 250）に分け、どちらの形
#: にも T2V と I2V を混ぜる。受入れ（`full-accept`）は形ごとに T2V 2 本 + I2V 2 本で、seed はどの
#: ケースとも別（778011〜）。文脈は固定 4 プロンプトを形と役割に散らす（{@link Ti2vCase} — 合成の
#: 乱数は使わない）。
#: MUST: `full-accept` の結果を見て `full-band` を足し引きしない（{@link CASES} と同じ規律）。
FULL_CASES: tuple[Ti2vCase, ...] = (
    Ti2vCase("full-band", "t2v", (21, 30, 52), 999, "boxing-cats", SEED + 10),
    Ti2vCase("full-band", "i2v", (21, 30, 52), 500, "ferret", SEED + 11),
    Ti2vCase("full-band", "t2v", (21, 30, 52), 113, "negative", SEED + 12),
    Ti2vCase("full-band", "i2v", (9, 44, 80), 999, "cat-dog-baking", SEED + 13),
    Ti2vCase("full-band", "t2v", (9, 44, 80), 750, "ferret", SEED + 14),
    Ti2vCase("full-band", "i2v", (9, 44, 80), 250, "boxing-cats", SEED + 15),
    Ti2vCase("full-accept", "t2v", (21, 30, 52), 999, "cat-dog-baking", 778011),
    Ti2vCase("full-accept", "t2v", (21, 30, 52), 600, "boxing-cats", 778012),
    Ti2vCase("full-accept", "i2v", (21, 30, 52), 900, "negative", 778013),
    Ti2vCase("full-accept", "i2v", (21, 30, 52), 400, "ferret", 778014),
    Ti2vCase("full-accept", "t2v", (9, 44, 80), 999, "negative", 778015),
    Ti2vCase("full-accept", "t2v", (9, 44, 80), 300, "ferret", 778016),
    Ti2vCase("full-accept", "i2v", (9, 44, 80), 800, "boxing-cats", 778017),
    Ti2vCase("full-accept", "i2v", (9, 44, 80), 50, "cat-dog-baking", 778018),
)

#: 実寸の 1 ケースの所要の見込み（秒 — 進捗の初期値）。段 0 の実測（S = 8,190・層逐次）の
#: f32 233 s + f64 510 s に、パッチ後の eager（f32・同じ層逐次）を f32 と同じ 233 s と見た和。
FULL_CASE_SECONDS = 233 + 510 + 233

#: 実寸の 1 ケースの RAM の山の見込み（段 0 の実測 5.04 GiB — 層逐次の参照・S = 8,190）。書く前に
#: MemAvailable がこれ + 余白（`dit_probe.MEMORY_HEADROOM_BYTES`）に足りなければ止まる。
FULL_PEAK_BYTES = int(5.1 * 2**30)

#: `write-full` の作業ディレクトリの接頭辞（系列の直下 — 部品ディレクトリの外）。
FULL_STAGING_PREFIX = ".write-full-"


def default_out() -> Path:
    """要約の既定の置き場（系列とは別 — `outputs/bench/karume-wan2.2/<日付>_dit-export/`）。"""
    return BENCH_ROOT / BENCH_NAME / f"{date.today().isoformat()}_dit-export"


def transformer_dir(model: str = DEFAULT_MODEL) -> Path:
    """pin した revision の DiT の置き場（取得済みでなければ `local_snapshot` が fail loudly）。"""
    return local_snapshot(model) / dit_reference.TRANSFORMER_SUBFOLDER


# ---------------------------------------------------------------------------
# trace（重みを持たない上流）
# ---------------------------------------------------------------------------


def meta_inputs(config: Mapping[str, Any], latent_shape: tuple[int, int, int]) -> tuple[Any, ...]:
    """trace の例示入力 7 本（meta — 重みが meta なので入力も meta に揃える）。"""
    patch = [int(size) for size in config["patch_size"]]
    frames, height, width = latent_shape
    tokens = (frames // patch[0]) * (height // patch[1]) * (width // patch[2])
    width_in = int(config["in_channels"]) * math.prod(patch)
    head_dim = int(config["attention_head_dim"])
    freq = int(config["freq_dim"])
    with torch.device("meta"):
        return (
            torch.empty(1, tokens, width_in),
            torch.empty(1, freq),
            torch.empty(1, 512, int(config["text_dim"])),
            torch.empty(1, tokens, 1, head_dim),
            torch.empty(1, tokens, 1, head_dim),
            torch.empty(1, freq),
            torch.empty(1, tokens, 1, dtype=torch.bool),
        )


def dynamic_shapes(sym_max: int = SYM_MAX) -> tuple[Any, ...]:
    """入力ごとの記号次元。RoPE の表と条件マスクも同じ `S`（別シンボルにすると長さのずれた形が
    受理されて沈黙誤値になる — 2.1 と同じ判断）。"""
    tokens = Dim("S", min=SYM_MIN, max=sym_max)
    return ({1: tokens}, None, None, {1: tokens}, {1: tokens}, None, {1: tokens})


def trace(
    config: Mapping[str, Any], sym_max: int = SYM_MAX
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, int]]:
    """meta の上流で I2V 対応の S 形の export を回す（`dit_probe.prepare` の tracer）。

    戻りはグラフ・格納テンソル（重みは meta）・量子化の対象（テンソルキー → チャネル軸）。
    """
    wrapper = dit_patch.WanDitTokensTi2v(dit_reference.meta_model(config))
    graph, tensors = export_module(
        wrapper,
        meta_inputs(config, TRACE_LATENT),
        dynamic_shapes=dynamic_shapes(sym_max),
        symbol_names=("S",),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    declared = [entry.name for entry in graph.inputs]
    if declared != list(INPUT_NAMES):
        raise Ti2vExportError(f"グラフ入力名が宣言と不一致: {declared}")
    targets = {
        fqn: axis for fqn, _, axis in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES)
    }
    return graph, tensors, targets


# ---------------------------------------------------------------------------
# IR の検査
# ---------------------------------------------------------------------------


def where_placement(graph: IrGraph) -> dict[str, Any]:
    """`where` の置き場所（決定 3）: 出力の消費者が直後の 1 ノードだけか。

    消費者が直後の 1 本なら、`where` の出力 `[1,S,dim]` はそのノードの間だけ生き、attention を
    またがない。上流の順（ブロックの先頭で 6 成分をまとめて作る）で書くと、ここで名指しされる。
    """
    consumers: dict[str, list[int]] = {}
    for position, node in enumerate(graph.nodes):
        for name in node.ins:
            consumers.setdefault(name, []).append(position)
    count = 0
    violations: list[str] = []
    for position, node in enumerate(graph.nodes):
        if node.op != "where":
            continue
        count += 1
        (output,) = node.outs
        users = consumers.get(output, [])
        if users != [position + 1]:
            crossed = sorted(
                {
                    graph.nodes[between].op
                    for between in range(position + 1, max(users, default=position) + 1)
                    if graph.nodes[between].op == "attention"
                }
            )
            violations.append(
                f"where #{position}（{output}）の消費者がノード {users}（直後の 1 本でない"
                f"{' — attention をまたぐ' if crossed else ''}）"
            )
        shape = graph.values[output].shape
        if len(shape) != 3 or shape[1] != "S":
            violations.append(f"where #{position}（{output}）の出力が {shape}（[1,S,dim] でない）")
    return {"count": count, "violations": violations}


def inspect_structure(graph: IrGraph, quant_targets: Sequence[str], layers: int) -> dict[str, Any]:
    """格納に依らない検査（linear の本数・`where` の本数と置き場所・入力の宣言）— 戻りの
    `failures` が空なら合格。"""
    ops = Counter(node.op for node in graph.nodes)
    placement = where_placement(graph)
    inputs = [[entry.name, list(entry.shape), entry.dtype] for entry in graph.inputs]
    failures: list[str] = []
    if ops["linear"] != len(quant_targets) + TIME_MLP_LINEARS:
        failures.append(
            f"linear のノードが {ops['linear']} 本（量子化の対象 {len(quant_targets)} +"
            f" 時刻の MLP {TIME_MLP_LINEARS} のはず）"
        )
    if placement["count"] != WHERE_PER_BLOCK * layers + WHERE_HEAD:
        failures.append(
            f"where が {placement['count']} 本"
            f"（{WHERE_PER_BLOCK} × {layers} + {WHERE_HEAD} のはず）"
        )
    failures.extend(placement["violations"])
    if [name for name, _, _ in inputs] != list(INPUT_NAMES):
        failures.append(f"グラフ入力が宣言と違う: {inputs}")
    elif inputs[-1][1:] != [[1, "S", 1], "bool"]:
        failures.append(f"条件マスクが bool [1,S,1] でない: {inputs[-1]}")
    return {
        "ops": dict(sorted(ops.items())),
        "linear_nodes": ops["linear"],
        "where": placement["count"],
        "symbols": list(graph.symbols),
        "inputs": inputs,
        "outputs": [list(graph.values[name].shape) for name in graph.outputs],
        "failures": failures,
    }


def inspect_storage(
    graph: IrGraph, quant_targets: Sequence[str], plain: Sequence[str]
) -> dict[str, Any]:
    """格納宣言を commit したグラフの格納の検査（i8 = 量子化の対象ちょうど・残りの重みは f32）。"""
    storage = {
        initializer.tensor: initializer.storage.dtype
        for initializer in graph.initializers.values()
        if initializer.tensor is not None
    }
    i8 = sorted(key for key, dtype in storage.items() if dtype == "i8")
    not_f32 = sorted(key for key in plain if storage.get(key) != "f32")
    failures: list[str] = []
    if i8 != sorted(quant_targets):
        failures.append(
            f"i8 の重みが量子化の対象と違う（i8 {len(i8)} 本 / 対象 {len(quant_targets)}）"
        )
    if not_f32:
        failures.append(f"f32 のはずの重みが f32 でない: {not_f32[:5]}")
    return {"i8_weights": len(i8), "f32_weights": len(plain) - len(not_f32), "failures": failures}


def inspect_graph(
    graph: IrGraph, quant_targets: Sequence[str], plain: Sequence[str], layers: int
) -> dict[str, Any]:
    """格納宣言を commit したグラフの検査（モジュール doc「IR の検査」— 構造 + 格納）。"""
    structure = inspect_structure(graph, quant_targets, layers)
    storage = inspect_storage(graph, quant_targets, plain)
    return {**structure, **storage, "failures": structure["failures"] + storage["failures"]}


def expected_counts(model: str, inspection: Mapping[str, Any]) -> list[str]:
    """5B の本数の期待値（決定 2 — {@link EXPECTED_5B}）との食い違い（5B 以外と、検査に
    無い欄は見ない）。"""
    if model != DEFAULT_MODEL:
        return []
    return [
        f"{key} が {inspection[key]}（決定 2 の期待値 {value}）"
        for key, value in EXPECTED_5B.items()
        if key in inspection and inspection[key] != value
    ]


# ---------------------------------------------------------------------------
# 容器
# ---------------------------------------------------------------------------


def rope_base_asset(writer: dit_reference.LayerwiseDit) -> dict[str, AssetInput]:
    """資産 `rope_base`（上流の RoPE の表から切り出した軸別素表 — 2.1 と同じ席・同じ形式）。"""
    payload = save(dit_patch.dit_rope_base_tables(writer.rope))
    return {
        export_dit.ROPE_BASE_ASSET: AssetInput(export_dit.ROPE_BASE_ROLE, len(payload), payload)
    }


def write_container(
    export: dit_probe.DitExport,
    path: Path,
    assets: Mapping[str, AssetInput],
    model: str = DEFAULT_MODEL,
) -> IrGraph:
    """材料（`dit_probe.prepare` の戻り）を容器に書く（`path` は代表 path — 分割形の part 列）。

    格納は材料のまま（量子化の対象は `fixed` の packed + scale・残りは checkpoint の f32）。戻りは
    格納宣言を commit したグラフ（{@link inspect_graph} の入力）。
    """
    return publish_model(
        path,
        export.graph,
        dict(export.tensors),
        provenance=export_dit.provenance(model),
        graph_name=TARGET,
        fixed_weights=export.fixed,
        assets=assets,
    )


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseInputs:
    """1 ケースの上流の入力・参照の時刻・グラフ入力 7 本。"""

    latents: torch.Tensor
    timestep: torch.Tensor
    condition_timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    #: 参照の時刻の表し方（T2V は None = 上流の 1 次元 timestep・I2V は参照ラッパ — 決定 4）。
    condition: dit_patch.TimestepCondition | None
    graph_inputs: tuple[torch.Tensor, ...]


def case_inputs(writer: dit_reference.LayerwiseDit, case: Ti2vCase) -> CaseInputs:
    """1 ケースの入力（乱数と文脈は `dit_probe.probe_inputs` の 1 か所で作る — f32 / f64 / eager が
    同じ値）。"""
    patch = writer.patch_size
    latents, timestep, encoder_hidden_states = dit_probe.probe_inputs(
        writer, case.spec(), case.text
    )
    conditioned = case.form == "i2v"
    condition_timestep = torch.zeros_like(timestep) if conditioned else timestep.clone()
    mask = dit_patch.dit_condition_mask(case.latent_shape, patch, conditioned=conditioned)
    rope_cos, rope_sin = dit_patch.dit_rope_tables(writer.rope, case.latent_shape)
    graph_inputs = (
        dit_patch.dit_patchify(latents, patch),
        writer.timesteps_proj(timestep),
        encoder_hidden_states,
        rope_cos,
        rope_sin,
        writer.timesteps_proj(condition_timestep),
        mask,
    )
    return CaseInputs(
        latents=latents,
        timestep=timestep,
        condition_timestep=condition_timestep,
        encoder_hidden_states=encoder_hidden_states,
        condition=dit_patch.TimestepCondition(condition_timestep, mask[..., 0])
        if conditioned
        else None,
        graph_inputs=graph_inputs,
    )


def patched_eager(
    writer: dit_reference.LayerwiseDit,
    source: dit_reference.WeightSource,
    inputs: CaseInputs,
    latent_shape: tuple[int, int, int],
    reference: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    """パッチ後のグラフ（{@link dit_patch.WanDitTokensTi2v}）の eager を同じ層逐次の重みで回し、
    参照との差の要約（`export_dit.eager_report` と同じ欄 — 判定は `export_dit.eager_failures`）と
    グラフ出力 `[1,S,pt·ph·pw·C]` を返す。

    trunk は patch 埋め込みを上流の conv3d の出力に差し替えた経路（ビット一致が主張）。Linear 化した
    patch 埋め込みの出力が conv3d とビット一致すれば trunk = full なので列を 1 回だけ回す。
    """
    patch = writer.patch_size

    def body(model: nn.Module) -> dict[str, Any]:
        wrapper = dit_patch.WanDitTokensTi2v(model, install_processors=False)
        tokens = inputs.graph_inputs[0]
        with dit_patch.flash_attention_only():
            hidden = model.patch_embedding(inputs.latents).flatten(2).transpose(1, 2)
            embedded = wrapper.patch_embedding(tokens)
            bound = export_dit.patch_embedding_error_bound(tokens, wrapper.patch_embedding)
            started = time.perf_counter()
            tokens_out = wrapper(*inputs.graph_inputs)
            seconds = time.perf_counter() - started
            full = dit_patch.dit_unpatchify(tokens_out, latent_shape, patch)
            trunk_from_full = torch.equal(embedded, hidden)
            trunk = (
                full
                if trunk_from_full
                else dit_patch.dit_unpatchify(
                    wrapper.forward_hidden(hidden, *inputs.graph_inputs[1:]), latent_shape, patch
                )
            )
        return {
            "tokens_out": tokens_out,
            "report": {
                "reference_max_abs": float(reference.abs().max()),
                "trunk_bit_exact": torch.equal(trunk, reference),
                "trunk_max_abs_diff": float((trunk - reference).abs().max()),
                "trunk_from_full": trunk_from_full,
                "patch_embedding_max_abs_diff": float((embedded - hidden).abs().max()),
                "patch_embedding_max_abs": float(hidden.abs().max()),
                "patch_embedding_within_bound": bool(((embedded - hidden).abs() <= bound).all()),
                "output_finite": bool(tokens_out.isfinite().all()),
                "full_bit_exact": torch.equal(full, reference),
                "full_max_abs_diff": float((full - reference).abs().max()),
                "patched_seconds": round(seconds, 1),
            },
        }

    result, _, _ = writer.run(
        source,
        torch.float32,
        body,
        prepare_block=lambda block: dit_patch.install_real_pair_processors([block]),
    )
    return result["report"], result["tokens_out"]


def per_token_timestep(inputs: CaseInputs) -> torch.Tensor:
    """diffusers の 2 次元 timestep `[1,S]`（観測の相手 — `WanPipeline(expand_timesteps=True)`
    は T2V でも全トークンに t を並べ、`WanImageToVideoPipeline` は条件フレームのトークンを 0 に
    する）。"""
    mask = inputs.graph_inputs[-1][..., 0]
    return torch.where(mask, inputs.condition_timestep, inputs.timestep).to(torch.int64)


def golden_case(
    writer: dit_reference.LayerwiseDit,
    source: dit_reference.WeightSource,
    case: Ti2vCase,
    out_dir: Path,
) -> dict[str, Any]:
    """1 ケースの参照（f32 / f64）・パッチ後の eager・観測を採り、門に掛けてから golden を書く。

    S が `dit_reference.PROGRESS_TOKENS` 以上のケース（実寸 — 1 forward が分単位）は、参照の
    ブロックごとの `[block]` の行と、パッチ後の eager の開始の行を出す（値は変わらない）。
    """
    patch = writer.patch_size
    name = case.name(patch)
    inputs = case_inputs(writer, case)
    common = (inputs.latents, inputs.timestep, inputs.encoder_hidden_states)
    long_case = case.tokens(patch) >= dit_reference.PROGRESS_TOKENS

    def progress(label: str) -> Any:
        return dit_reference._block_progress(name, label, writer.layers) if long_case else None

    reference32 = writer.forward(
        source, torch.float32, *common, condition=inputs.condition, on_block=progress("f32")
    )
    reference64 = writer.forward(
        source, torch.float64, *common, condition=inputs.condition, on_block=progress("f64")
    )
    if long_case:
        print(f"[golden] {name}: パッチ後の eager（f32・同じ層逐次の重み）", flush=True)
    report, tokens_out = patched_eager(
        writer, source, inputs, case.latent_shape, reference32.output
    )
    report.update(
        {
            "case": name,
            "form": case.form,
            "reference_f32_seconds": round(reference32.seconds, 1),
            "reference_f64_seconds": round(reference64.seconds, 1),
            "reference_f32_vs_f64_ratio": export_dit._ratio(reference32.output, reference64.output),
        }
    )
    if case.role in OBSERVED_ROLES:
        observed = writer.forward(
            source, torch.float32, inputs.latents, per_token_timestep(inputs), common[2]
        )
        report["per_token_timestep_vs_reference_ratio"] = export_dit._ratio(
            observed.output, reference32.output
        )
        report["per_token_timestep_bit_exact"] = torch.equal(observed.output, reference32.output)
    failures = export_dit.eager_failures(report)
    print(f"[eager] {json.dumps(report, ensure_ascii=False)}", flush=True)
    if failures:
        raise export_dit.EagerEquivalenceError(f"{name}: {failures}")
    io = {
        f"{export_dit.INPUT_PREFIX}{key}": normalize_boundary_tensor(
            value, f"{name} の入力 '{key}'"
        )
        for key, value in zip(INPUT_NAMES, inputs.graph_inputs, strict=True)
    }
    io[f"{export_dit.OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
        tokens_out.contiguous(), f"{name} の出力"
    )
    reference = export_dit.reference_tensors(
        name,
        latents=inputs.latents,
        timestep=inputs.timestep,
        output=reference32.output,
        blocks=[],
        output_f64=reference64.output,
    )
    reference[CONDITION_TIMESTEP_KEY] = normalize_boundary_tensor(
        inputs.condition_timestep, f"{name} の条件側の timestep"
    )
    save_file(io, str(out_dir / f"{export_dit.IO_PREFIX}{name}{export_dit.CASE_SUFFIX}"))
    save_file(
        reference, str(out_dir / f"{export_dit.REFERENCE_PREFIX}{name}{export_dit.CASE_SUFFIX}")
    )
    return report


#: diffusers の 2 次元 timestep の経路との差（観測 — 門ではない）を採る役割（決定用 6 本 —
#: T2V / I2V の形の両方を含む。受入れまで採ると 1 ケース 1 forward ずつ書き出しが延びる）。
OBSERVED_ROLES = frozenset({"band"})

#: `reference.<case>` の条件側の timestep のキー（I2V は 0・T2V は生成側と同じ — TS のホストテストが
#: `timesteps_proj_condition` の突き合わせに使う）。
CONDITION_TIMESTEP_KEY = "condition_timestep"


def write_series(model: str = DEFAULT_MODEL, *, names: Sequence[str] = ()) -> dict[str, Any]:
    """容器と golden を作業席に書き、門（IR の検査・eager 同値）を全部通ったときだけ系列へ据える。

    `names` はケース名で絞る（開発用 — 絞った系列は据えない: 席の中で止める）。
    """
    directory = transformer_dir(model)
    writer = dit_reference.LayerwiseDit(
        dit_reference.load_config(directory),
        alignment=dit_reference.upstream_alignment(directory),
    )
    known = {case.name(writer.patch_size) for case in CASES}
    if set(names) - known:
        raise Ti2vExportError(f"知らないケース名がある: {sorted(set(names) - known)}")
    chosen = [case for case in CASES if not names or case.name(writer.patch_size) in names]
    target = SERIES / TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    eager: list[dict[str, Any]] = []
    with MemoryMonitor() as monitor:
        export, prepared = dit_probe.prepare(
            directory, stage=monitor.stage, sym_max=SYM_MAX, tracer=trace
        )
        quant_targets = sorted(export.fixed)
        plain = export.plain
        with staged_publication(target) as staged:
            staged.mkdir()
            with monitor.stage("write") as record:
                graph = write_container(export, staged / MODEL_FILE, rope_base_asset(writer), model)
                record.details["initializers"] = len(graph.initializers)
            # golden は容器から読むので、材料（i8 の packed 約 4.65 GiB）を先に手放す。
            del export
            gc.collect()
            inspection = inspect_graph(graph, quant_targets, plain, writer.layers)
            failures = inspection["failures"] + expected_counts(model, inspection)
            if failures:
                raise Ti2vExportError(f"IR の検査に落ちた: {failures}")
            source = dit_reference.ContainerDitWeights(staged / MODEL_FILE)
            for case in chosen:
                with monitor.stage(f"golden {case.name(writer.patch_size)}"):
                    eager.append(golden_case(writer, source, case, staged))
            if names:
                raise Ti2vExportError(
                    f"ケースを絞った回（{len(chosen)} / {len(CASES)} 本）は系列を据えない"
                    "（全ケースが揃った系列だけを据える — 各ケースの結果は [eager] の行に出ている）"
                )
    breakdown = storage_breakdown(graph)
    return {
        "series": str(target),
        "prepare": prepared,
        "inspection": inspection,
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "text_embeds": {
            "path": str(dit_probe.TEXT_EMBEDS_ASSET),
            "sha256": dit_reference.file_sha256(dit_probe.TEXT_EMBEDS_ASSET),
        },
        "eager": eager,
        "worst_reference_f32_vs_f64_ratio": max(r["reference_f32_vs_f64_ratio"] for r in eager),
        "stages": [record.to_dict() for record in monitor.records],
        "passed": True,
    }


# ---------------------------------------------------------------------------
# 実寸の golden（段 2 — 据えた容器に golden を足す）
# ---------------------------------------------------------------------------

#: 据わっている `reference.<case>` の、ケースの表から組み直せる入力のキー（再開の照合）。
_STORED_REFERENCE_INPUTS = ("latents", "timestep", CONDITION_TIMESTEP_KEY)

#: 据わっているケースのファイルが持つべき出力のキー（`io` / `reference`）。
_STORED_IO_OUTPUT = f"{export_dit.OUTPUT_PREFIX}0"
_STORED_REFERENCE_OUTPUTS = ("output", export_dit.REFERENCE_F64_KEY)

#: {@link stored_case_state} の戻り。
StoredState = Literal["absent", "partial", "present"]


def stored_case_state(
    writer: dit_reference.LayerwiseDit, case: Ti2vCase, target: Path
) -> StoredState:
    """部品ディレクトリにある 1 ケースの golden の状態（モジュール doc「実寸の golden」の再開）。

    2 本とも揃っていれば、据わっている入力（`io` の `input.*` 7 本と `reference` の
    {@link _STORED_REFERENCE_INPUTS}）を今のケースの表から組み直した値とビットで比べ、出力のキーが
    揃っていることも見る。MUST: 食い違えば fail loudly — 表を変えた後の古い golden を「済み」に
    数えると、帯の決定用と受入れが黙って別の入力になる（消すのは人の判断）。
    """
    name = case.name(writer.patch_size)
    io_name, reference_name = export_dit.case_file_names(name)
    present = [(target / file).is_file() for file in (io_name, reference_name)]
    if not any(present):
        return "absent"
    if not all(present):
        return "partial"
    inputs = case_inputs(writer, case)
    expected = {
        "latents": inputs.latents,
        "timestep": normalize_boundary_tensor(inputs.timestep, f"{name} の timestep"),
        CONDITION_TIMESTEP_KEY: normalize_boundary_tensor(
            inputs.condition_timestep, f"{name} の条件側の timestep"
        ),
    }
    differing: list[str] = []
    with safe_open(str(target / io_name), "pt") as handle:
        keys = set(handle.keys())
        for key, value in zip(INPUT_NAMES, inputs.graph_inputs, strict=True):
            stored = f"{export_dit.INPUT_PREFIX}{key}"
            built = normalize_boundary_tensor(value, f"{name} の入力 '{key}'")
            if stored not in keys or not torch.equal(handle.get_tensor(stored), built):
                differing.append(stored)
        if _STORED_IO_OUTPUT not in keys:
            differing.append(f"{_STORED_IO_OUTPUT}（無い）")
    with safe_open(str(target / reference_name), "pt") as handle:
        keys = set(handle.keys())
        for key in _STORED_REFERENCE_INPUTS:
            if key not in keys or not torch.equal(handle.get_tensor(key), expected[key]):
                differing.append(f"reference.{key}")
        differing += [
            f"reference.{key}（無い）" for key in _STORED_REFERENCE_OUTPUTS if key not in keys
        ]
    if differing:
        raise Ti2vExportError(
            f"{target} の {name} の golden がケースの表と食い違う（{differing}）— 表を変えたなら、"
            f"古い {io_name} / {reference_name} を消してから回し直す"
        )
    return "present"


def _stored_ratio(target: Path, name: str) -> float:
    """据わっているケースの CPU f32 の参照の f64 に対する比（要約用）。

    据えた `output.f64`（f32 へ丸めた値）から採るので、TS の r 門の正規化の分母と同じ値になる
    （`golden_case` の要約の比は丸める前の f64 が相手で、分母の小さいケースでは数 % 違う）。
    """
    _, reference_name = export_dit.case_file_names(name)
    with safe_open(str(target / reference_name), "pt") as handle:
        return export_dit._ratio(
            handle.get_tensor("output"), handle.get_tensor(export_dit.REFERENCE_F64_KEY)
        )


def _append_log(log: Path, row: Mapping[str, Any]) -> None:
    """1 ケースの結果の行を 1 回の write で改行まで書いてディスクへ落とす（記録 — 再開の目印は
    部品ディレクトリのファイルの方）。"""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_full(
    model: str = DEFAULT_MODEL, *, names: Sequence[str] = (), log: Path | None = None
) -> dict[str, Any]:
    """据えた系列に実寸の golden を足す（モジュール doc「実寸の golden」— 容器は書き直さない）。

    `names` はケース名で絞る（絞っても書いたケースは据える — 1 ケースずつ足す形なので、長い実行を
    分けられる）。`log` を渡すと書いたケースごとに 1 行の JSON を足す。
    """
    directory = transformer_dir(model)
    target = SERIES / TARGET
    container = target / MODEL_FILE
    if not all(part.is_file() for part in container_parts(container)):
        raise Ti2vExportError(f"据えた容器が無い（{container}）。先に write で系列を据える")
    writer = dit_reference.LayerwiseDit(
        dit_reference.load_config(directory),
        alignment=dit_reference.upstream_alignment(directory),
    )
    patch = writer.patch_size
    known = {case.name(patch) for case in FULL_CASES}
    if set(names) - known:
        raise Ti2vExportError(f"知らないケース名がある: {sorted(set(names) - known)}")
    chosen = [case for case in FULL_CASES if not names or case.name(patch) in names]
    for stale in sorted(SERIES.glob(f"{FULL_STAGING_PREFIX}*")):
        print(f"[write-full] 前の実行の作業ディレクトリ {stale} を消す", flush=True)
        shutil.rmtree(stale)
    states = {case.name(patch): stored_case_state(writer, case, target) for case in chosen}
    pending = [case for case in chosen if states[case.name(patch)] != "present"]
    print(
        f"[write-full] {target}: {len(chosen)} ケースのうち済み {len(chosen) - len(pending)}・"
        f"これから {len(pending)}（1 ケースの見込み {FULL_CASE_SECONDS} s・"
        f"環境 {json.dumps(dit_reference.numeric_environment(), ensure_ascii=False)}）",
        flush=True,
    )
    if pending:
        dit_probe.require_available("write-full", FULL_PEAK_BYTES)
    source = dit_reference.ContainerDitWeights(container)
    written: dict[str, dict[str, Any]] = {}
    elapsed: list[float] = []
    with MemoryMonitor() as monitor:
        for position, case in enumerate(pending, start=1):
            name = case.name(patch)
            each = statistics.mean(elapsed) if elapsed else FULL_CASE_SECONDS
            print(
                f"[write-full] {position}/{len(pending)} {name}（{states[name]}）: 開始 —"
                f" 残りの見込み {each * (len(pending) - position + 1) / 60:.0f} 分",
                flush=True,
            )
            started = time.perf_counter()
            with (
                monitor.stage(f"golden {name}") as record,
                tempfile.TemporaryDirectory(dir=SERIES, prefix=FULL_STAGING_PREFIX) as scratch,
            ):
                report = golden_case(writer, source, case, Path(scratch))
                # 門を通ったケースだけを部品ディレクトリへ移す（同じファイルシステムの rename）。
                for file in export_dit.case_file_names(name):
                    os.replace(Path(scratch) / file, target / file)
            elapsed.append(time.perf_counter() - started)
            row = {
                **report,
                "role": case.role,
                "seconds": round(elapsed[-1], 1),
                "stage": record.to_dict(),
            }
            written[name] = row
            if log is not None:
                _append_log(log, row)
            print(
                f"[write-full] {position}/{len(pending)} {name}: 据えた {elapsed[-1]:.0f} s",
                flush=True,
            )
    cases = []
    for case in chosen:
        name = case.name(patch)
        row = written.get(name)
        cases.append(
            {
                "case": name,
                "role": case.role,
                "form": case.form,
                "status": "written" if row is not None else "present",
                "reference_f32_vs_f64_ratio": _stored_ratio(target, name),
                **({} if row is None else {"seconds": row["seconds"]}),
            }
        )
    return {
        "series": str(target),
        "cases": cases,
        "eager": list(written.values()),
        "text_embeds": {
            "path": str(dit_probe.TEXT_EMBEDS_ASSET),
            "sha256": dit_reference.file_sha256(dit_probe.TEXT_EMBEDS_ASSET),
        },
        "min_reference_f32_vs_f64_ratio": min(c["reference_f32_vs_f64_ratio"] for c in cases),
        "stages": [record.to_dict() for record in monitor.records],
        "environment": dit_reference.numeric_environment(),
        "passed": True,
    }


# ---------------------------------------------------------------------------
# 独立した重みでの eager 同値（決定 4 の形）
# ---------------------------------------------------------------------------


def golden_agreement(
    name: str,
    inputs: CaseInputs,
    reference: torch.Tensor,
    tokens_out: torch.Tensor,
    golden_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """in-memory の上流の参照・パッチ後の出力と、据えた golden の突き合わせ（要約の欄と食い違い）。

    グラフ入力 7 本も golden の `input.*` とビットで比べる（同じケースを組めていなければ、出力の
    比較は意味を持たない）。
    """
    io_name, reference_name = export_dit.case_file_names(name)
    with safe_open(str(golden_dir / io_name), "pt") as handle:
        stored_inputs = {
            key: handle.get_tensor(f"{export_dit.INPUT_PREFIX}{key}") for key in INPUT_NAMES
        }
        stored_output = handle.get_tensor(f"{export_dit.OUTPUT_PREFIX}0")
    with safe_open(str(golden_dir / reference_name), "pt") as handle:
        stored_reference = handle.get_tensor("output")
    differing = [
        key
        for key, value in zip(INPUT_NAMES, inputs.graph_inputs, strict=True)
        if not torch.equal(
            normalize_boundary_tensor(value, f"{name} の入力 '{key}'"), stored_inputs[key]
        )
    ]
    record = {
        "golden_inputs_bit_exact": not differing,
        "golden_reference_bit_exact": torch.equal(reference, stored_reference),
        "golden_reference_ratio": export_dit._ratio(reference, stored_reference),
        "golden_output_bit_exact": torch.equal(tokens_out, stored_output),
        "golden_output_ratio": export_dit._ratio(tokens_out, stored_output),
    }
    failures = []
    if differing:
        failures.append(
            f"グラフ入力 {differing} が golden の io と違う（同じケースを組めていない）"
        )
    if not record["golden_reference_bit_exact"]:
        failures.append(
            "上流の f32 の参照が golden の reference の output とビット一致しない"
            f"（比 {record['golden_reference_ratio']:.3e}）"
        )
    if not record["golden_output_bit_exact"]:
        failures.append(
            "パッチ後のグラフの出力が golden の io の output.0 とビット一致しない"
            f"（比 {record['golden_output_ratio']:.3e}）"
        )
    return record, failures


def full_eager_case(
    upstream: nn.Module,
    wrapper: dit_patch.WanDitTokensTi2v,
    writer: dit_reference.LayerwiseDit,
    case: Ti2vCase,
    golden_dir: Path,
) -> dict[str, Any]:
    """1 ケースの独立した重みでの eager 同値（モジュール doc「独立した重みでの eager 同値」）。

    `writer` は入力を組むだけ（RoPE の表と時刻の sinusoid — 重みは読まない）。参照とパッチ後の
    eager は f32 の全量の `upstream`（fake-quant 済み）と、それを包む `wrapper` で回す。戻りの欄
    `failures` が空なら合格。
    """
    name = case.name(writer.patch_size)
    inputs = case_inputs(writer, case)
    with torch.no_grad():
        started = time.perf_counter()
        reference = dit_patch.reference_dit(
            upstream,
            inputs.latents,
            inputs.timestep,
            inputs.encoder_hidden_states,
            inputs.condition,
        )
        seconds = time.perf_counter() - started
    built = export_dit.Case(
        name=name,
        spec=case.spec(),
        inputs=inputs.graph_inputs,
        latents=inputs.latents,
        timestep=inputs.timestep,
        reference=reference,
        reference_blocks=[],
        reference_seconds=seconds,
    )
    report, tokens_out = export_dit.eager_report(wrapper, upstream, built)
    golden, mismatches = golden_agreement(name, inputs, reference, tokens_out, golden_dir)
    report.update({"form": case.form, **golden})
    report["failures"] = export_dit.eager_failures(report) + mismatches
    print(f"[eager-full] {json.dumps(report, ensure_ascii=False)}", flush=True)
    return report


def eager_full(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """f32 の fake-quant の全量 1 本で、パッチ後のグラフを上流と比べ、据えた golden とも突き合わせる
    （モジュール doc「独立した重みでの eager 同値」— 全ケース）。

    MUST: 上流を読む（約 21 GiB）前に、据えた golden が揃っていることと MemAvailable を確かめる。
    """
    directory = transformer_dir(model)
    config = dit_reference.load_config(directory)
    writer = dit_reference.LayerwiseDit(config)
    target = SERIES / TARGET
    missing = [
        file
        for case in CASES
        for file in export_dit.case_file_names(case.name(writer.patch_size))
        if not (target / file).is_file()
    ]
    if missing:
        raise Ti2vExportError(
            f"据えた golden が揃っていない（{target} — 欠け {len(missing)} 本: {missing[:3]}）。"
            "先に write で系列を据える"
        )
    estimate = dit_probe.eager_memory(config)["f32"] + dit_probe.EAGER_BASE_BYTES
    dit_probe.require_available("load", estimate)
    reports: list[dict[str, Any]] = []
    with MemoryMonitor() as monitor:
        with monitor.stage("load") as record:
            upstream = export_dit.load_transformer(model)
            record.details["parameters"] = sum(p.numel() for p in upstream.parameters())
        with monitor.stage("fake-quant") as record:
            wrapper = dit_patch.WanDitTokensTi2v(upstream)
            record.details["rounded"] = export_dit.fake_quant_i8(upstream, wrapper).describe()
        for case in CASES:
            with monitor.stage(f"eager {case.name(writer.patch_size)}"):
                reports.append(full_eager_case(upstream, wrapper, writer, case, target))
    failures = {report["case"]: report["failures"] for report in reports if report["failures"]}
    return {
        "series": str(target),
        "memory_estimate_gib": round(estimate / 2**30, 2),
        "cases": reports,
        "failures": failures,
        "stages": [record.to_dict() for record in monitor.records],
        "passed": not failures,
    }


def inspect_only(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """trace と構造の検査だけ（重みは読まない — config だけ）。

    格納の検査（i8 の本数・f32 の残り）は `write` が書いた容器のグラフで見る。ここが見るのは
    linear の本数・`where` の置き場所・入力の宣言（どれも格納に依らない）と、量子化の対象の本数。
    """
    config = dit_reference.load_config(transformer_dir(model))
    graph, _, targets = trace(config)
    inspection = inspect_structure(graph, sorted(targets), int(config["num_layers"]))
    failures = inspection["failures"] + expected_counts(
        model, {**inspection, "i8_weights": len(targets)}
    )
    return {
        **inspection,
        "quant_targets": len(targets),
        "failures": failures,
        "passed": not failures,
    }


# ---------------------------------------------------------------------------
# 量子化の観測（決定 5）
# ---------------------------------------------------------------------------

#: 観測のモデル → そのモデルの決定用ケース（S = 192）。1.3B は i8 系列の決定用 6 本（T2V だけ）。
OBSERVE_MODELS = ("t2v-1.3b", DEFAULT_MODEL)

#: 5B の観測値が 1.3B の観測値の何倍を超えたら裁定に上げるか（決定 5 の目安）。
ESCALATE_FACTOR = 2.0


def _observe_cases(
    model: str, patch: tuple[int, int, int]
) -> list[tuple[str, export_dit.CaseSpec, Form, str]]:
    """観測のケース（名前・潜在の形・T2V / I2V の形・文脈のプロンプト）。

    5B は {@link CASES} の決定用 6 本。1.3B は i8 系列の決定用 6 本（T2V だけ）に、5B の決定用と
    同じ順でプロンプトを割り当てる（文脈の作り方を 2 つのモデルで揃える — 合成の乱数の文脈は
    5B だけを増幅する・{@link Ti2vCase}）。
    """
    band = [case for case in CASES if case.role == "band"]
    if model == DEFAULT_MODEL:
        return [(case.name(patch), case.spec(), case.form, case.text) for case in band]
    specs = [spec for spec in export_dit.series_cases("i8", full=False) if spec.role == "band"]
    return [
        (spec.name(patch), spec, "t2v", case.text) for spec, case in zip(specs, band, strict=True)
    ]


def observe_quant(models: Sequence[str] = OBSERVE_MODELS) -> dict[str, Any]:
    """量子化なしの f32 参照に対する重みだけ i8 の f32 参照の差（モジュール doc「量子化の観測」）。

    5B と 1.3B の決定用ケースで採り、5B の最悪が 1.3B の最悪の何倍かを並べる。
    """
    results: dict[str, Any] = {}
    for model in models:
        directory = transformer_dir(model)
        config = dit_reference.load_config(directory)
        writer = dit_reference.LayerwiseDit(
            config, alignment=dit_reference.upstream_alignment(directory)
        )
        unquantized = dit_reference.CheckpointDitWeights(directory, "none")
        quantized = dit_reference.CheckpointDitWeights(
            directory, "i8", dit_reference.quant_keys(config)
        )
        rows = []
        for name, spec, form, prompt in _observe_cases(model, writer.patch_size):
            latents, timestep, text = dit_probe.probe_inputs(writer, spec, prompt)
            condition = None
            if form == "i2v":
                mask = dit_patch.dit_condition_mask(
                    spec.latent_shape, writer.patch_size, conditioned=True
                )[..., 0]
                condition = dit_patch.TimestepCondition(torch.zeros_like(timestep), mask)
            plain = writer.forward(
                unquantized, torch.float32, latents, timestep, text, condition=condition
            )
            rounded = writer.forward(
                quantized, torch.float32, latents, timestep, text, condition=condition
            )
            ratio = export_dit._ratio(rounded.output, plain.output)
            rows.append({"case": name, "form": form, "text": prompt, "ratio": ratio})
            print(f"[observe] {model} {name}: {ratio:.3e}", flush=True)
        ratios = [row["ratio"] for row in rows]
        results[model] = {
            "cases": rows,
            "worst": max(ratios),
            "median": statistics.median(ratios),
        }
    summary: dict[str, Any] = {"models": results}
    if set(OBSERVE_MODELS) <= set(results):
        factor = results[DEFAULT_MODEL]["worst"] / results["t2v-1.3b"]["worst"]
        summary["worst_factor_5b_over_1_3b"] = factor
        summary["escalate"] = factor > ESCALATE_FACTOR
    summary["passed"] = True
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", choices=("inspect", "write", "write-full", "observe-quant", "eager-full")
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="ケース名で絞る（write: 系列は据えない・write-full: 書いたケースは据える）",
    )
    parser.add_argument("--out", type=Path, default=None, help="要約の置き場")
    args = parser.parse_args(argv)
    if args.case and args.command not in ("write", "write-full"):
        parser.error("--case は write / write-full にだけ掛かる")
    out = default_out() if args.out is None else args.out
    started = time.perf_counter()
    if args.command == "inspect":
        summary = inspect_only()
    elif args.command == "write":
        summary = write_series(names=args.case)
    elif args.command == "write-full":
        summary = write_full(names=args.case, log=out / "write-full.jsonl")
    elif args.command == "eager-full":
        summary = eager_full()
    else:
        summary = observe_quant()
    document = {
        "command": args.command,
        "result": summary,
        "seconds": round(time.perf_counter() - started, 1),
        "environment": dit_reference.numeric_environment(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.command}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(document, ensure_ascii=False, indent=1))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
