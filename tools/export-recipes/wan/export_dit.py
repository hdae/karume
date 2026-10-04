"""Wan2.1 の DiT を S 形の IR（`krm`）+ golden へ書き出す台本（ADR 0118 決定 3 / 8・段 2）。

起点は pin した revision の diffusers 版（`wan.sources.local_snapshot`）。グラフは
{@link wan.dit_patch.WanDitTokens}（入口は patchify の後・出口は unpatchify の前・次元はトークン長
`S` の 1 シンボル）。格納は f16 席（重みを f16 表現可能値へ丸めてから参照と golden を採り、
適格な重みスロットだけを f16 で格納する — ADR 0006 / 0018）。容器には重みに加えて**資産
`rope_base`**（ホストが RoPE の表を組むための軸別素表 — 役割 `rope-base`・ADR 0109 決定 4）が入る。

    uv run --group wan --inexact python -m wan.export_dit            # 製品のグラフ + golden
    uv run --group wan --inexact python -m wan.export_dit --layers   # 計測用（層別の出口）
    uv run --group wan --inexact python -m wan.export_dit --verify   # パッチ前後の eager 同値だけ
    uv run --group wan --inexact python -m wan.export_dit --dtype i8            # i8 系列（下の節）
    uv run --group wan --inexact python -m wan.export_dit --dtype i8 --no-full  # 同・実寸を除く
    uv run --group wan --inexact python -m wan.export_dit --dtype i8 --check    # 書き直して照合

出力（既定 `outputs/series/wan2.1-t2v-1.3b-f16-dyn/transformer/`・`--dtype i8` は
`outputs/series/wan2.1-t2v-1.3b-i8-dyn/transformer/`）:

    model-NNNNN-of-NNNNN.krm      重み・定数 + 2 文書の記述 + 資産 `rope_base`
    io.<case>.safetensors         グラフの入力（`input.*`）とパッチ後の torch CPU の出力
                                  （`output.0`）
    reference.<case>.safetensors  上流の素の diffusers（CPU f32・同じ f16 丸めの重み）の
                                  `latents` / `timestep`（i32）/ `output`（`[1,16,F,H,W]`）/
                                  `block.NN`（各ブロックの出力 `[1,S,1536]` —
                                  `CaseSpec.blocks` のケースだけ）。実寸のケースは加えて
                                  `output.f64`（活性も f64 で回した上流の出力を f32 へ丸めた値
                                  — 1 ブロックずつ f64 にする層逐次の
                                  `wan.dit_reference.LayerwiseDit`・ADR 0121 決定 7）

`--layers` は層別の出口を足した計測用のグラフ（{@link wan.dit_patch.WanDitTokensLayers}）を
`…-f16-dyn-probe/transformer/` へ書く。golden は書かない（入力は製品の系列の `io.*`、期待値は同じ
`reference.*` の `block.NN` を使う — 入力と重みが同じなので）。

## ケース（決定 8 と追記 2026-10-02 の「帯の決定と受入れを分ける」）

| 役割 | 潜在 `[1,16,F,H,W]` | S | timestep | 有効長 | seed |
| --- | --- | --: | --: | --: | --: |
| `band`（帯の決定） | `[1,16,3,16,16]` | 192 | 999 | 24 | `SEED` + 0 |
| `band` | `[1,16,2,16,24]` | 192 | 750 | 50 | `SEED` + 3 |
| `band` | `[1,16,3,16,16]` | 192 | 500 | 12 | `SEED` + 4 |
| `band` | `[1,16,1,16,48]` | 192 | 250 | 80 | `SEED` + 5 |
| `band` | `[1,16,2,16,24]` | 192 | 600 | 30 | `SEED` + 8 |
| `band` | `[1,16,3,16,16]` | 192 | 113 | 60 | `SEED` + 6 |
| `accept`（受入れ） | `[1,16,2,16,24]` | 192 | 600 | 45 | 777001 |
| `accept` | `[1,16,4,8,16]` | 128 | 30 | 9 | 777002 |
| `accept` | `[1,16,2,24,16]` | 192 | 400 | 70 | 777003 |
| `growth`（S に対する伸び） | `[1,16,3,32,32]` | 768 | 999 | 24 | `SEED` + 2 |
| `growth` | `[1,16,3,32,32]` | 768 | 500 | 37 | `SEED` + 7 |
| `full-band`（実寸の帯の決定） | `[1,16,9,60,104]` | 14,040 | 999 | 40 | `SEED` + 10 |
| `full-band`（名前の接尾辞 `-2`） | `[1,16,9,60,104]` | 14,040 | 999 | 112 | `SEED` + 11 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 750 | 23 | `SEED` + 12 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 500 | 64 | `SEED` + 9 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 250 | 91 | `SEED` + 13 |
| `full-band` | `[1,16,9,60,104]` | 14,040 | 113 | 7 | `SEED` + 14 |
| `full-accept`（実寸の受入れ） | `[1,16,9,60,104]` | 14,040 | 999 | 28 | 777006 |
| `full-accept` | `[1,16,9,60,104]` | 14,040 | 600 | 77 | 777007 |
| `full-band`（81 フレームの帯の決定） | `[1,16,21,60,104]` | 32,760 | 999 | 40 | `SEED` + 20 |
| `full-band`（名前の接尾辞 `-2`） | `[1,16,21,60,104]` | 32,760 | 999 | 112 | `SEED` + 21 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 750 | 23 | `SEED` + 22 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 500 | 64 | `SEED` + 23 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 250 | 91 | `SEED` + 24 |
| `full-band` | `[1,16,21,60,104]` | 32,760 | 113 | 7 | `SEED` + 25 |
| `full-accept`（81 フレームの受入れ） | `[1,16,21,60,104]` | 32,760 | 999 | 28 | 777008 |
| `full-accept` | `[1,16,21,60,104]` | 32,760 | 600 | 77 | 777009 |

帯の指標は「最大絶対差 ÷ 参照の最大絶対値」（決定 8 の目安の形）で、帯 = `band` の 6 ケースの
最悪の比 × 5（TS 側 `e2e_wan_dit_test.ts` の `DIT_RATIO_BAND`）。`accept` の 3 ケースは `band` と
seed が違い、timestep も 1 本（600）を除いて違う（600 の 1 本は、旧帯 `atol` 3.2e-4 を超えた検証の
未見ケース — seed 777001・有効長 45 — をそのまま再現したもの）。

MUST: `accept` の結果を見て `band` のケースを足し引きしない（帯の決定と受入れの独立が崩れる）。
受入れが帯を外れたら、帯を広げずに原因を調べる。

S = 14,040 の `full-band` / `full-accept` は実寸（832×480・33 フレーム — ADR 0118 段 3）の帯を、
S = 192 の帯とは**独立に**導くためのケース（決定 8 の外挿の規律）。帯 = `full-band` 6 ケースの
最悪の正規化した比 × 5（TS 側の `DIT_FULL_NORMALIZED_BAND`）で、`full-accept` 2 ケースで受け
入れる。決定用は timestep を 999（生成の最初のステップ — seed 2 本）/ 750 / 500 / 250 / 113 に
散らす。受入れは 999 と 600 で、seed（777006 / 777007）は段 2 のどのケースとも、決定用とも別。
受入れの seed は指標を正規化した比に変えたときに新しくした（旧 777004 / 777005 は指標を決める
前に結果を見ていたので、受入れの独立が崩れていた）。
MUST: `full-accept` の結果を見て `full-band` のケースも指標も変えない。

S = 32,760（832×480・81 フレーム = `Dim("S")` の上限ちょうど — ADR 0118 段 8）の 8 本は、帯を
S = 14,040 の帯とも**独立に**導く（決定 8 — S ごとに帯を導き、S = 14,040 の帯を持ち込まない）。
timestep と有効長は S = 14,040 の 8 本と同じ並びで、seed だけを新しくする（決定用 `SEED` + 20〜25・
受入れ 777008 / 777009 — どのケースとも別）。帯の形（決定用 6 本の最悪の正規化した比 × 5・受入れ
2 本）と上の MUST は S = 14,040 と同じ。

実寸の参照は 2 本: **f64 の参照**（`output.f64` — 活性も f64・重みは同じ f16 丸め）を正とし、CPU
f32 の参照（`output`）は正規化の分母に使う。指標は「GPU の f64 に対する比 ÷ CPU f32 の参照の f64
に対する比」（GPU の誤差が CPU f32 の何倍か）。比そのものは入力で 240 倍動く（丸めを増幅する入力
では CPU f32 の参照も同じだけ f64 から離れる）ので、CPU f32 の誤差で割って入力による増幅を打ち
消す。所要は f64 が f32 の約 2.7 倍（`[case]` 行に出す — S = 14,040 の 1 ケースで f64 約 370 s・
f32 約 136 s）。参照（f32 / f64）とパッチ後の eager の attention は CPU の flash 経路に固定する
（`dit_patch.flash_attention_only` — MATH へ落ちると S = 32,760 でスコア行列 1 枚が 51.5 GB に
なり OOM）。

各ブロックの出力（`block.NN` — S = 14,040 の 1 ケースで 2.6 GB）は S = 192 / 768 の全ケースと、
実寸では決定用 `full-band-s14040-t0999` と受入れ `full-accept-s14040-t0999` の 2 本だけが持つ
（`CaseSpec.blocks` — 層ごとの記録はこの 2 本で足り、残りは最終出力だけで判定できる）。
S = 32,760 のケースは 1 本も持たない: 突き合わせる相手の層別の出口（`--layers` の probe）が、
30 ブロック分の readback staging（約 6 GB）で B570 に載らない。

決定用を 1 ケースにしないのは、誤差が入力で 1 桁近く動くため（実測: t = 999 の 1 ケースで決めた帯を
t = 500 の未見ケースが 1.6 倍超えた。中ほどの timestep は CPU の eager でも入力の 1e-5 級の揺れを
約 4 倍に増幅する — 移植の誤りではなく入力の感度）。決定用は timestep を参照の設定の列
（999 → 60）の全域に散らし、seed と格子も変える。最大絶対差でなく比を見るのは、12,288 要素の裾の
1 要素で決まる絶対値が参照の振れ幅ごと動くため。

格子は正方（F'·H'·W' = 3·8·8）と非正方（2·8·12 / 1·8·24 / 2·12·8 / 4·4·8）を混ぜる。h と w を
取り違えたホスト実装は正方の格子では値が一致してしまうので、非正方で初めて割れる。テキスト文脈は
固定 seed の乱数 `[有効長, 4096]` を上流と同じ形（後ろをゼロで 512 行まで埋める —
`wan.pipeline_ref.pad_text_embeds`）にしたもの。

MUST: 参照は**上流の素の forward**（`dit_patch.reference_dit_layers`）で採る。attn1 の processor の
差し替えは素の経路を変えない（`dit_patch` のモジュール docstring）ので、参照を採る順序の門は要らな
い。
MUST: 重みの丸め（f16 系列の f16 丸め・i8 系列の fake-quant）は参照より**前**（ADR 0006 / 0019）。
丸めは S 形のラッパに掛け、共有している上流の部品にも同じ値が届く（patch 埋め込みは conv の重みの
view — `dit_patch.patch_embedding_linear`）。RoPE の表（`model.rope` のバッファ）は丸めない — 製品は
f32 の素表を焼き、参照もその値で回す。
MUST: 書き手の eager 同値の門（{@link eager_failures}）は**丸めた後**の重みで見る。参照とパッチ後の
eager は同じ丸め済みの重みを共有して回る。パッチの同値（trunk のビット一致・patch 埋め込みの
上界）は重みの値に依らない主張（RoPE の書き換えは重みを読まない）なので、丸めた後に見ても門の
意味は変わらない。加えて、参照が丸め済みの重みで採られたことも同じ門が縛る — 丸めが参照側へ
届いていなければ trunk がビット一致しない。丸める前（素の f32 の重み）の同値は
`--verify --no-f16` と pytest の実重みのテスト（`wan/tests/test_dit_patch.py` の
`TestRealWeights`）が見る。

## i8 系列（ADR 0120 決定 2 / 3）

`--dtype i8` は参照席 `f16+dit8` と実用席 `f16+dit8-a8-attn8-s16` の重みの系列を書く。対象は
transformer だけで、VAE は f16 系列のまま（決定 2）。

- 重み: linear 307 本（patch 埋め込みの Linear を含む）の `weight` を per-channel symmetric i8 の
  RTN（最近接丸め）で丸める（`karume.quantize.fake_quant_int8` を S 形のラッパへ —
  {@link fake_quant_i8}）。格納は i8 + 出力チャネルごとの f32 scale。校正は無い（重みは RTN で、
  活性は実行時の per-token の動的量子化）。
- bias・norm の weight・`scale_shift_table` は**上流の f32 の値のまま**。f16 系列の
  {@link round_to_f16} は掛けない（決定 2 — i8 の席に f16 の丸めを混ぜない。計算と格納の形は同じで
  値だけが違い、混ぜると配布の改変告知が増える）。
- ケース: f16 系列の S = 192 の組（`band` / `accept` — S = 128 の 1 本を含む）と S = 14,040 の組
  （`full-band` / `full-accept`）を、同じ seed・timestep のまま i8 の重みで採り直す
  （{@link I8_CASE_ROLES} / {@link I8_FULL_LATENTS}）。`growth`（S = 768）と S = 32,760 は採らない
  （決定 3 — S = 32,760 での伸びは f16 系列の門が見ている）。
- 参照: 正規化比 r の門を S = 192 でも持つので、**全ケースが f64 の参照**（`output.f64`）を持つ
  （f16 系列は実寸のケースだけ — {@link wants_float64}）。
- `--no-full` は S = 14,040 の 8 本を除いて書く（段の分割用 — S = 14,040 は CPU で約 1.5 時間）。
  系列は作業席ごと据え替わるので、揃った系列は `--no-full` 無しの実走で書き直す（ケースの乱数は
  seed から派生するので、S = 192 の golden は同じ値で書き直される）。

## 既存の系列との照合（`--check` — ADR 0121 決定 10）

`--check` は同じ書き手で部品ディレクトリを**系列の中の一時ディレクトリ**（`<系列>/.check-*/` —
umT5 の `write --check` と同じ置き方）へ書き直し、既存の部品ディレクトリと**全ファイルの sha256**
（容器の全 part と `io.*` / `reference.*`）で突き合わせる（{@link series_mismatches}）。系列は
置き換えない。違えば違うファイルを名指しして非 0 で終わる（一時ディレクトリは消える）。

守るもの: Wan2.1 と共有する書き手（`dit_patch` / `dit_reference` / この台本）を Wan2.2 のために
変えたとき、1.3B の容器と golden が 1 バイトも動いていないこと。書き手の門（eager 同値）は書き直した
側の内部の整合しか見ないので、既存の系列との突き合わせが別に要る。

- `--dtype i8 --no-full --check` は実寸（S = 14,040）の 8 ケースを書かず、既存のそのファイルは
  照合から外す（{@link check_absent} — 外したケースのファイルが既存に無ければそれも名指す）。容器は
  全 part を見る。
- `--layers --check` は計測用の系列（容器だけ）を照合する。
- 実寸の golden の値の再現は `wan.dit_reference compare`（層逐次の参照で採り直してテンソルごとに
  比べる）が別に見る。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import AssetInput, Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.emit import storage_breakdown
from karume.pipeline import export_to_file
from karume.quantize import (
    QUANT_MODULE_TYPES,
    Int8Report,
    channel_scale,
    fake_quant_int8,
    iter_quant_targets,
    quantize_to_int8,
    round_weights_to_f16,
)
from wan import dit_patch
from wan.pipeline_ref import TEXT_DIM, assert_no_mps, pad_text_embeds
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, local_snapshot

#: 系列（ADR 0118 決定 7 — 接尾辞 `-dyn` は ADR 0077 の慣例）と、計測用の層別出口の系列。
SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-f16-dyn"
PROBE_SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-f16-dyn-probe"
#: i8 系列（ADR 0120 決定 2 — transformer だけ。綴りは anima の `<model>-i8-dyn` に倣う）。
I8_SERIES = SERIES_ROOT / "wan2.1-t2v-1.3b-i8-dyn"

#: 書き出せる格納 dtype（= `export_to_file` の `weight_dtype`）。
DTYPES = ("f16", "i8")

#: 部品名（= 容器のグラフ名 = 配布 manifest の weights のキー — container-v1 §2.1）。
TARGET = "transformer"
MODEL_FILE = "model.krm"

#: 資産の名前と役割（Anima の S 形 DiT と同じ綴り — ADR 0109 決定 4）。
ROPE_BASE_ASSET = "rope_base"
ROPE_BASE_ROLE = "rope-base"

IO_PREFIX = "io."
REFERENCE_PREFIX = "reference."
#: 実寸のケースの `reference.<case>` に入れる f64 の参照のキー（TS 側の `REFERENCE_F64_KEY`）。
REFERENCE_F64_KEY = "output.f64"
CASE_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."
#: `--check` の一時ディレクトリの接頭辞（系列の中 — umT5 の `write --check` と同じ綴り）。
CHECK_PREFIX = ".check-"

#: グラフ入力の名前（ラッパの forward の引数名がそのまま IR の入力名になる）。
INPUT_NAMES = ("tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin")

#: `Dim("S")` の上限 = 81 フレーム（潜在 `[16,21,60,104]` → 21·30·52）。グラフ側にトークン長の制約は
#: 無い（RoPE の表を入力へ出したので S 依存の焼き込みが無い）ので、ここは受理集合の最大を宣言する。
DIT_SYM_MAX = 32_760
#: 同下限（0 / 1 特殊化を避ける線 — Anima の `DIT_SYM_MIN` と同じ）。
DIT_SYM_MIN = 2

#: 決定用と伸びの記録のケースの乱数はここから派生させる（グローバル seed に依存しない — 再生成で
#: バイト一致させる）。受入れのケースは別系統の seed（777001〜）を直に持つ。
SEED = 20261002


@dataclass(frozen=True)
class CaseSpec:
    """golden 1 ケースの条件（潜在の形・timestep・テキストの有効長・乱数の seed）。"""

    role: str
    latent_shape: tuple[int, int, int]
    timestep: int
    text_length: int
    seed: int
    #: 役割・S・timestep が同じケースを名前で分ける接尾辞（実寸の t = 999 の 2 本目）。
    variant: str = ""
    #: 各ブロックの出力（`block.NN`）を golden に持つか（層ごとの記録の相手 — 実寸は 2 本だけ）。
    blocks: bool = True

    def name(self, patch_size: tuple[int, int, int]) -> str:
        frames, height, width = self.latent_shape
        tokens = (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])
        suffix = f"-{self.variant}" if self.variant else ""
        return f"{self.role}-s{tokens:05d}-t{self.timestep:04d}{suffix}"

    @property
    def full_size(self) -> bool:
        """実寸（ADR 0118 段 3 の 33 フレーム・段 8 の 81 フレーム）のケース — f32 に加えて f64 の
        参照も採る。"""
        return self.role.startswith("full-")


#: ケースの表（モジュール docstring の表と同じ）。先頭が export の例示入力（小さい方 —
#: `torch.export` はトレースで 1 回 forward を回す）。
#: MUST: `band`（帯の決定）と `accept`（受入れ）は固定する — `accept` の結果を見て `band` を
#: 変えない（モジュール docstring）。実寸の `full-band` / `full-accept` も同じ。
CASES: tuple[CaseSpec, ...] = (
    CaseSpec("band", (3, 16, 16), 999, 24, SEED + 0),
    CaseSpec("band", (2, 16, 24), 750, 50, SEED + 3),
    CaseSpec("band", (3, 16, 16), 500, 12, SEED + 4),
    CaseSpec("band", (1, 16, 48), 250, 80, SEED + 5),
    CaseSpec("band", (2, 16, 24), 600, 30, SEED + 8),
    CaseSpec("band", (3, 16, 16), 113, 60, SEED + 6),
    CaseSpec("accept", (2, 16, 24), 600, 45, 777001),
    CaseSpec("accept", (4, 8, 16), 30, 9, 777002),
    CaseSpec("accept", (2, 24, 16), 400, 70, 777003),
    CaseSpec("growth", (3, 32, 32), 999, 24, SEED + 2),
    CaseSpec("growth", (3, 32, 32), 500, 37, SEED + 7),
    CaseSpec("full-band", (9, 60, 104), 999, 40, SEED + 10),
    CaseSpec("full-band", (9, 60, 104), 999, 112, SEED + 11, variant="2", blocks=False),
    CaseSpec("full-band", (9, 60, 104), 750, 23, SEED + 12, blocks=False),
    CaseSpec("full-band", (9, 60, 104), 500, 64, SEED + 9, blocks=False),
    CaseSpec("full-band", (9, 60, 104), 250, 91, SEED + 13, blocks=False),
    CaseSpec("full-band", (9, 60, 104), 113, 7, SEED + 14, blocks=False),
    CaseSpec("full-accept", (9, 60, 104), 999, 28, 777006),
    CaseSpec("full-accept", (9, 60, 104), 600, 77, 777007, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 999, 40, SEED + 20, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 999, 112, SEED + 21, variant="2", blocks=False),
    CaseSpec("full-band", (21, 60, 104), 750, 23, SEED + 22, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 500, 64, SEED + 23, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 250, 91, SEED + 24, blocks=False),
    CaseSpec("full-band", (21, 60, 104), 113, 7, SEED + 25, blocks=False),
    CaseSpec("full-accept", (21, 60, 104), 999, 28, 777008, blocks=False),
    CaseSpec("full-accept", (21, 60, 104), 600, 77, 777009, blocks=False),
)

#: i8 系列が採るケース（ADR 0120 決定 3 — 参照席 `f16+dit8` の r 門は S = 192 と S = 14,040）:
#: 実寸でない役割のうち S = 192 の組（`band` / `accept`）と、実寸のうち 33 フレームの格子
#: （S = 14,040）。{@link CASES} の行をそのまま使う（seed・timestep・有効長を f16 系列と揃える）。
I8_CASE_ROLES = ("band", "accept")
I8_FULL_LATENTS: tuple[tuple[int, int, int], ...] = ((9, 60, 104),)


def series_dir(dtype: str) -> Path:
    """格納 dtype → 系列のディレクトリ。"""
    return {"f16": SERIES, "i8": I8_SERIES}[dtype]


def series_cases(dtype: str, *, full: bool = True) -> tuple[CaseSpec, ...]:
    """格納 dtype の系列が採るケース（先頭は {@link CASES} と同じ例示入力）。`full=False` は実寸の
    ケースを除く（`--no-full`）。"""
    if dtype == "f16":
        cases = CASES
    else:
        cases = tuple(
            spec
            for spec in CASES
            if (
                spec.latent_shape in I8_FULL_LATENTS
                if spec.full_size
                else spec.role in I8_CASE_ROLES
            )
        )
    return cases if full else tuple(spec for spec in cases if not spec.full_size)


def wants_float64(dtype: str, spec: CaseSpec) -> bool:
    """そのケースが f64 の参照（`output.f64`）を持つか。

    f16 系列は実寸のケースだけ（S = 192 の帯は最大絶対差の比 — ADR 0118 決定 8）、i8 系列は
    全ケース（r 門を S = 192 でも持つ — ADR 0120 決定 3）。
    """
    return dtype == "i8" or spec.full_size


@dataclass(frozen=True)
class Case:
    """組み上がった 1 ケース（グラフ入力・上流の参照・パッチ後の eager 出力）。"""

    name: str
    spec: CaseSpec
    inputs: tuple[torch.Tensor, ...]
    latents: torch.Tensor
    timestep: torch.Tensor
    reference: torch.Tensor
    #: 各ブロックの出力（`spec.blocks` でないケースは空）。
    reference_blocks: list[torch.Tensor]
    #: 上流の素の forward（参照）の所要（秒・CPU f32 — 実寸のケースの所要の記録）。
    reference_seconds: float


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def load_transformer(model_name: str = DEFAULT_MODEL) -> nn.Module:
    """pin した revision の DiT を CPU f32 で読む（MPS の見える機では止める — 決定 3）。"""
    assert_no_mps()
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        local_snapshot(model_name), subfolder="transformer", torch_dtype=torch.float32
    )
    return model.eval()


def unrounded_weights(dtype: str, model: nn.Module) -> list[str]:
    """系列の格納 dtype の丸めの不動点に無い上流の重みの名前（空なら丸め済み）。

    - `f16`: 全パラメータが f16 の表現可能値（f16 → f32 の往復でビット不変）
    - `i8`: 上流の重みスロット（linear と patch 埋め込み）が per-channel の i8 の表現可能値（もう
      1 度丸めてビット不変 — `karume.quantize.INT8_MAX` の注記）

    丸めの書き手（{@link round_to_f16} / {@link fake_quant_i8}）と、丸め済みのモデルを受ける側
    （{@link float64_references}）が同じ判定を通る。
    """
    with torch.no_grad():
        if dtype == "f16":
            return [
                name
                for name, parameter in model.named_parameters()
                if not torch.equal(parameter, parameter.to(torch.float16).to(torch.float32))
            ]
        if dtype == "i8":
            missed = []
            for name, weight, axis in iter_quant_targets(model, op_types=QUANT_MODULE_TYPES):
                scale = channel_scale(weight, axis)
                rounded = quantize_to_int8(weight, scale).to(torch.float32) * scale
                if not torch.equal(rounded, weight):
                    missed.append(name)
            return missed
    raise ValueError(f"格納 dtype {dtype!r} の丸めは知らない（{' / '.join(DTYPES)}）")


def round_to_f16(model: nn.Module, wrapper: nn.Module) -> str:
    """S 形のラッパ経由で重みを f16 表現可能値へ丸め、上流の全パラメータに届いたことを確かめる。

    ラッパが持つのは上流の部品への参照と、conv の重みの view の Linear だけ（`rope` は持たない）。
    だから丸めは上流のパラメータ全部に届き、`rope` のバッファ（RoPE の表）には届かない。
    MUST: 両方を確かめる — 片方でも外れると「参照だけ別の重み」か「参照の表だけ f16」になり、差に
    量子化誤差が混ざって帯の意味が消える。
    """
    rope_before = {name: buffer.clone() for name, buffer in model.rope.named_buffers()}
    report = round_weights_to_f16(wrapper)
    missed = unrounded_weights("f16", model)
    if missed:
        raise AssertionError(f"f16 の丸めが上流のパラメータに届いていない: {missed[:5]}")
    moved = [
        name
        for name, buffer in model.rope.named_buffers()
        if not torch.equal(buffer, rope_before[name])
    ]
    if moved:
        raise AssertionError(f"RoPE の表（rope のバッファ）が丸められた: {moved}")
    return report.describe()


def fake_quant_i8(model: nn.Module, wrapper: nn.Module) -> Int8Report:
    """S 形のラッパ経由で linear の重みを per-channel symmetric i8 の表現可能値へ丸め（RTN —
    ADR 0019）、上流の重みスロット全部に届いたことを確かめる（ADR 0120 決定 2 の i8 系列）。

    ラッパの linear は上流の Linear そのものと、conv の重みの view の patch 埋め込みだけ。だから
    丸めは上流の全 Linear と patch 埋め込みの Conv3d に届く。bias・norm の weight・
    `scale_shift_table`・`rope` のバッファは `fake_quant_int8` が触らない（上流の f32 の値のまま —
    決定 2）。
    MUST: 上流側に届いたことを確かめる — 届かない重みが 1 本でもあると「参照だけ別の重み」になり、
    差に量子化誤差が混ざって帯の意味が消える（{@link round_to_f16} と同じ理由）。確かめ方は冪等性:
    i8 の表現可能値は per-channel の量子化の不動点（`karume.quantize.INT8_MAX` の注記）なので、
    もう 1 度丸めてビット不変なら届いている。本数も上流の重みスロットの本数と一致させる。
    """
    report = fake_quant_int8(wrapper)
    upstream = list(iter_quant_targets(model, op_types=QUANT_MODULE_TYPES))
    missed = unrounded_weights("i8", model)
    if missed:
        raise AssertionError(f"i8 の丸めが上流の重みに届いていない: {missed[:5]}")
    if len(upstream) != report.modules:
        raise AssertionError(
            f"ラッパで丸めた重み {report.modules} 本が上流の重みスロット {len(upstream)} 本と違う"
        )
    return report


def fake_quant(
    dtype: str, model: nn.Module, wrapper: nn.Module
) -> tuple[str, Mapping[str, torch.Tensor] | None]:
    """系列の格納 dtype の表現可能値へ重みを丸める（参照の採取より前 MUST — ADR 0006）。

    戻りは要約の 1 行と、i8 の scale 台帳（emit へそのまま渡す — f16 は None）。
    """
    if dtype == "i8":
        report = fake_quant_i8(model, wrapper)
        return f"i8 per-channel へ丸めた — {report.describe()}", report.scales
    if dtype == "f16":
        return f"f16 表現可能値へ丸めた — {round_to_f16(model, wrapper)}", None
    raise ValueError(f"格納 dtype {dtype!r} の系列は書けない（{' / '.join(DTYPES)}）")


def case_inputs(
    model: nn.Module, spec: CaseSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1 ケースの上流の入力（潜在 `[1,16,F,H,W]`・timestep・テキスト文脈 `[1,512,4096]`）。

    f32 と f64 の参照が同じ値を受けるよう、乱数はここ 1 か所で引く。
    """
    generator = _generator(spec.seed)
    latents = torch.randn(1, model.config.in_channels, *spec.latent_shape, generator=generator)
    text = torch.randn(spec.text_length, TEXT_DIM, generator=generator)
    # 上流のパイプラインは scheduler の int64 の timestep を `expand(batch)` して渡す。
    timestep = torch.tensor([spec.timestep], dtype=torch.int64)
    return latents, timestep, pad_text_embeds(text)


def build_case(model: nn.Module, spec: CaseSpec) -> Case:
    """1 ケースの入力を作り、上流の素の forward で参照（最終出力と、`spec.blocks` なら各ブロックの
    出力）を採る。"""
    patch_size = tuple(int(size) for size in model.config.patch_size)
    latents, timestep, encoder_hidden_states = case_inputs(model, spec)
    with torch.no_grad():
        started = time.perf_counter()
        if spec.blocks:
            reference, blocks = dit_patch.reference_dit_layers(
                model, latents, timestep, encoder_hidden_states
            )
        else:
            reference = dit_patch.reference_dit(model, latents, timestep, encoder_hidden_states)
            blocks = []
        reference_seconds = time.perf_counter() - started
        print(f"[case] {spec.name(patch_size)}: 上流の参照 {reference_seconds:.1f} s", flush=True)
        rope_cos, rope_sin = dit_patch.dit_rope_tables(model.rope, spec.latent_shape)
        inputs = (
            dit_patch.dit_patchify(latents, patch_size),
            dit_patch.dit_timesteps_proj(model, timestep),
            encoder_hidden_states,
            rope_cos,
            rope_sin,
        )
    return Case(
        name=spec.name(patch_size),
        spec=spec,
        inputs=inputs,
        latents=latents,
        timestep=timestep,
        reference=reference,
        reference_blocks=blocks,
        reference_seconds=reference_seconds,
    )


@dataclass(frozen=True)
class Float64Reference:
    """実寸の 1 ケースの f64 の参照（最終出力 `[1,16,F,H,W]` は f64 のまま — 書くときに丸める）。"""

    output: torch.Tensor
    #: 所要（秒・CPU f64）。
    seconds: float


def float64_references(
    model: nn.Module, specs: Sequence[CaseSpec], dtype: str
) -> dict[str, Float64Reference]:
    """f64 の参照を持つケース（{@link wants_float64}）の f64 の参照（上流の素の forward を活性も
    f64 で — 層逐次の `wan.dit_reference.LayerwiseDit`・ADR 0121 決定 7）。

    `model` は系列の格納 dtype の丸め（{@link fake_quant}）を掛けた後の f32 のモデル。重みはその
    f32 の値を 1 ブロックずつ f64 へ広げ、入力も {@link case_inputs} の同じ値を広げる。丸めは f32 の
    うちに掛ける（f64 で丸めると scale と丸めの値が f32 の参照と食い違う）。モデル全体を f64 に
    しない — 同時に持つ f64 はブロックの外と 1 ブロックだけ（5B では f64 の全量 37.25 GiB が
    31 GiB 機に載らない。1.3B もこの 1 本の経路を通す — 全体 f64 の経路は残さない）。
    MUST: `model` が丸め済みであることをここで確かめる（{@link unrounded_weights}）— 呼ぶ順が
    崩れて丸める前のモデルが渡ると、丸めていない重みの f64 golden が黙って書かれる。
    """
    from wan import dit_reference

    chosen = [spec for spec in specs if wants_float64(dtype, spec)]
    if not chosen:
        return {}
    unrounded = unrounded_weights(dtype, model)
    if unrounded:
        raise AssertionError(
            f"f64 の参照に {dtype} の丸めの前の重みが渡された（fake_quant の後に呼ぶ）:"
            f" {unrounded[:5]}"
        )
    writer = dit_reference.LayerwiseDit(model.config)
    source = dit_reference.ModuleWeights(model)
    references: dict[str, Float64Reference] = {}
    for spec in chosen:
        latents, timestep, encoder_hidden_states = case_inputs(model, spec)
        result = writer.forward(source, torch.float64, latents, timestep, encoder_hidden_states)
        name = spec.name(writer.patch_size)
        print(f"[case] {name}: 上流の f64 参照（層逐次）{result.seconds:.1f} s", flush=True)
        references[name] = Float64Reference(output=result.output, seconds=result.seconds)
    return references


def dynamic_shapes() -> tuple[Any, ...]:
    """入力ごとの記号次元。RoPE の表も同じ `S` で宣言する（別シンボルにすると「表と本体の長さが
    ずれた」形が受理されて沈黙誤値になる — Anima と同じ）。"""
    tokens = Dim("S", min=DIT_SYM_MIN, max=DIT_SYM_MAX)
    return ({1: tokens}, None, None, {1: tokens}, {1: tokens})


def provenance(model_name: str) -> Provenance:
    """容器へ焼く出所（上流のライセンス識別子と pin した revision — `wan.sources` が正本）。"""
    source = SOURCES[model_name]
    return Provenance(
        license=source.license, notice=NOTICE_FILENAME, upstream_revision=source.revision
    )


def rope_base_asset(model: nn.Module) -> dict[str, AssetInput]:
    """資産 `rope_base`（軸別素表の safetensors のバイト列そのもの — Anima と同じ席）。"""
    payload = save(dit_patch.dit_rope_base_tables(model.rope))
    return {ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), payload)}


class EagerEquivalenceError(AssertionError):
    """パッチ後の eager が ADR 0118 の主張（trunk のビット一致・Linear 化の差の上界・有限）から
    外れた。"""


def patch_embedding_error_bound(tokens: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """patch 埋め込みの Linear 化と上流の conv3d の差の、縮約順の違いだけで生じうる上界
    （要素ごと）。

    `2·γ_{K+1}·(|x|·|W|ᵀ + |b|)`（`γ_n = n·u / (1 − n·u)`・`u = 2⁻²⁴`・K = 窓の要素数で、bias の
    加算が項を 1 つ増やす）。2 つの縮約順のどちらも真値から `γ_{K+1}·Σ|項|` 以内に収まるので、
    差はその 2 倍以内。Linear 化はビット一致を期待しない（ADR 0118 — 差を記録する）が、並びの
    取り違えのような実装の誤りは値そのものの大きさで出て、この上界を超える。
    """
    weight = linear.weight
    magnitude = tokens.abs() @ weight.abs().T
    terms = tokens.shape[-1]
    if linear.bias is not None:
        magnitude = magnitude + linear.bias.abs()
        terms += 1
    unit = 2.0**-24
    return 2 * (terms * unit / (1 - terms * unit)) * magnitude


#: 要約の数値の欄のうち、有限でなければならないもの（`reference_f32_vs_f64_ratio` は実寸の
#: ケースだけが持つ — 下の {@link eager_failures}）。
_FINITE_FIELDS = (
    "reference_max_abs",
    "trunk_max_abs_diff",
    "patch_embedding_max_abs_diff",
    "full_max_abs_diff",
)


def eager_failures(report: Mapping[str, Any]) -> list[str]:
    """eager の要約 1 本が ADR 0118 の主張から外れた点（空なら合格）。

    MUST: emit は 1 つでもあれば公開しない（{@link emit}）・`--verify` は非 0 で終わる。見るのは
    trunk のビット一致（パッチの eager 同値 MUST）・patch 埋め込みの差が縮約順の上界の内
    （{@link patch_embedding_error_bound}）・パッチ後の出力と数値の欄が有限であること。
    """
    failures = []
    if report.get("trunk_bit_exact") is not True:
        failures.append("trunk（patch 埋め込み以外の書き換え）が上流とビット一致しない")
    if report.get("patch_embedding_within_bound") is not True:
        failures.append("patch 埋め込みの Linear 化と conv3d の差が縮約順の上界を超える")
    if report.get("output_finite") is not True:
        failures.append("パッチ後のグラフ出力に非有限値がある")
    fields = [*_FINITE_FIELDS]
    if "reference_f32_vs_f64_ratio" in report:
        fields.append("reference_f32_vs_f64_ratio")
    for field in fields:
        value = report.get(field)
        if not isinstance(value, float) or not math.isfinite(value):
            failures.append(f"{field} が有限の数でない（{value!r}）")
    return failures


def eager_report(
    wrapper: dit_patch.WanDitTokens, model: nn.Module, case: Case
) -> tuple[dict[str, Any], torch.Tensor]:
    """パッチ後の eager（ホストの unpatchify まで）と上流の素の出力の差（ケース 1 本）と、パッチ後の
    グラフ出力 `[1,S,pt·ph·pw·C]`（golden の `output.0` — 書き手が回し直さずに使う）。

    `trunk` は patch 埋め込みを上流の conv3d の出力に差し替えた経路（RoPE の書き換えを含む本体だけの
    比較 — ビット一致が主張）、`full` は Linear 化した patch 埋め込みを含む製品の経路
    （差を記録する — 判定は patch 埋め込みの差が {@link patch_embedding_error_bound} の内か）。
    判定の欄の読み方は {@link eager_failures}。

    patch 埋め込みの出力が上流の conv3d とビット一致するなら、trunk は full と同じ入力で同じ計算を
    する（`forward` は `forward_hidden(patch_embedding(tokens), …)`）ので回し直さない
    （`trunk_from_full`）。実寸（S = 14,040）では 1 forward が CPU で分単位なので、ケースあたりの
    forward をこの 1 本に絞る。

    attention は参照と同じく {@link dit_patch.flash_attention_only} の下で回す（S = 32,760 で MATH
    へ落ちると OOM）。
    """
    patch_size = wrapper.patch_size
    with torch.no_grad(), dit_patch.flash_attention_only():
        hidden = model.patch_embedding(case.latents).flatten(2).transpose(1, 2)
        embedded = wrapper.patch_embedding(case.inputs[0])
        bound = patch_embedding_error_bound(case.inputs[0], wrapper.patch_embedding)
        started = time.perf_counter()
        tokens_out = wrapper(*case.inputs)
        patched_seconds = time.perf_counter() - started
        full = dit_patch.dit_unpatchify(tokens_out, case.spec.latent_shape, patch_size)
        trunk_from_full = torch.equal(embedded, hidden)
        trunk = (
            full
            if trunk_from_full
            else dit_patch.dit_unpatchify(
                wrapper.forward_hidden(hidden, *case.inputs[1:]),
                case.spec.latent_shape,
                patch_size,
            )
        )
    report = {
        "case": case.name,
        "reference_max_abs": float(case.reference.abs().max()),
        "trunk_bit_exact": torch.equal(trunk, case.reference),
        "trunk_max_abs_diff": float((trunk - case.reference).abs().max()),
        "trunk_from_full": trunk_from_full,
        "patch_embedding_max_abs_diff": float((embedded - hidden).abs().max()),
        "patch_embedding_max_abs": float(hidden.abs().max()),
        "patch_embedding_within_bound": bool(((embedded - hidden).abs() <= bound).all()),
        "output_finite": bool(tokens_out.isfinite().all()),
        "full_bit_exact": torch.equal(full, case.reference),
        "full_max_abs_diff": float((full - case.reference).abs().max()),
        "reference_seconds": round(case.reference_seconds, 1),
        "patched_seconds": round(patched_seconds, 1),
    }
    return report, tokens_out


def _ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """帯の指標（最大絶対差 ÷ 参照の最大絶対値 — TS 側の `ratioOf` と同じ形）を f64 で。"""
    expected = expected.double()
    return float((actual.double() - expected).abs().max() / expected.abs().max())


def reference_tensors(
    name: str,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    output: torch.Tensor | None,
    blocks: Sequence[torch.Tensor],
    output_f64: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """`reference.<case>` の中身（`latents` / `timestep`〈i32〉/ `output` / `block.NN` /
    `output.f64`）。

    書き手（{@link _write_case_files}）と層逐次の参照の突き合わせ（`wan.dit_reference compare` —
    `output` を採らない `--f64-only` は `output=None`）が同じ組み方を通る — 2 か所で綴ると、
    突き合わせが書き手と違う形を「一致」と言いうる。
    """
    reference = {
        "latents": latents.contiguous(),
        "timestep": normalize_boundary_tensor(timestep, f"{name} の timestep"),
    }
    if output is not None:
        reference["output"] = output.contiguous()
    reference.update(
        {f"block.{index:02d}": block.contiguous() for index, block in enumerate(blocks)}
    )
    if output_f64 is not None:
        # 格納は f32 へ丸めた値（TS の safetensors は F64 を読まない）。丸めの差は要素ごとに
        # 2⁻²⁴·|x| 以下で、比にして 6e-8 以下。正規化の分母（CPU f32 の参照の f64 に対する比）を
        # 動かす割合は分母の最小で決まる: 決定用（`full-band`）だけなら S = 14,040 の最小 3.72e-6 で
        # 2% 未満・S = 32,760 の最小 2.07e-6 で約 2.9%、受入れを含めると最小 1.83e-6
        # （`full-accept-s14040-t0600`）で約 3.3%。帯（最悪 × 5）の判定には効かない（実測の表は
        # TS 側 `e2e_wan_dit_test.ts` の `REFERENCE_F64_KEY` と `DIT_FULL_NORMALIZED_BAND`）。
        reference[REFERENCE_F64_KEY] = output_f64.to(torch.float32).contiguous()
    return reference


def case_file_names(name: str) -> tuple[str, str]:
    """ケース 1 本の golden のファイル名（`io.<case>` と `reference.<case>`）。

    書き手（{@link _write_case_files}）と照合の除外（{@link check_absent}）が同じ綴りを通る。
    """
    return f"{IO_PREFIX}{name}{CASE_SUFFIX}", f"{REFERENCE_PREFIX}{name}{CASE_SUFFIX}"


def _write_case_files(
    case: Case, output: torch.Tensor, reference_f64: Float64Reference | None, out_dir: Path
) -> list[str]:
    """`io.<case>`（グラフ入力 + パッチ後の eager 出力 `output`）と `reference.<case>`（上流の値）を
    書く。実寸のケースは `reference.<case>` に f64 の参照 `output.f64` も入れる。"""
    io = {
        f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{case.name} の入力 '{key}'")
        for key, value in zip(INPUT_NAMES, case.inputs, strict=True)
    }
    io[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
        output.detach().contiguous(), f"{case.name} の出力"
    )
    reference = reference_tensors(
        case.name,
        latents=case.latents,
        timestep=case.timestep,
        output=case.reference,
        blocks=case.reference_blocks,
        output_f64=None if reference_f64 is None else reference_f64.output,
    )
    io_name, reference_name = case_file_names(case.name)
    save_file(io, str(out_dir / io_name))
    save_file(reference, str(out_dir / reference_name))
    return [io_name, reference_name]


class SeriesMismatchError(AssertionError):
    """`--check` で書き直した部品ディレクトリが既存の系列とバイトで違った（または照合の相手が
    無い）。"""


def file_digests(directory: Path) -> dict[str, str]:
    """部品ディレクトリ直下のファイル名 → sha256（ファイルでない項目は fail loudly — 照合が
    見ないものを残さない）。"""
    digests: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            raise SeriesMismatchError(f"部品ディレクトリにファイルでない項目がある: {path}")
        with path.open("rb") as stream:
            digests[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return digests


def series_mismatches(
    written: Path, existing: Path, *, absent: Collection[str] = ()
) -> dict[str, str]:
    """書き直した部品ディレクトリと既存の部品ディレクトリの食い違い（ファイル名 → 理由。空なら
    全ファイルがバイト一致）。

    `absent` は今回は書かない既存のファイル（`--no-full` が除く実寸のケース —
    {@link check_absent}）。書いていれば・既存に無ければ、それも食い違いとして名指す
    （除外が黙って空振りしない）。
    """
    actual = file_digests(written)
    expected = file_digests(existing)
    mismatches: dict[str, str] = {}
    for name in sorted(actual.keys() | expected.keys() | set(absent)):
        if name in absent:
            if name in actual:
                mismatches[name] = "照合から外したケースのファイルを書いた"
            elif name not in expected:
                mismatches[name] = "照合から外したケースのファイルが既存に無い"
        elif name not in expected:
            mismatches[name] = "既存に無い"
        elif name not in actual:
            mismatches[name] = "書いた側に無い"
        elif actual[name] != expected[name]:
            mismatches[name] = "sha256 が違う"
    return mismatches


def check_absent(
    dtype: str, patch_size: tuple[int, int, int], *, full: bool, layers: bool
) -> frozenset[str]:
    """`--check` で照合から外す既存のファイル（`--no-full` が書かない実寸のケースの golden）。

    計測用の系列（`--layers`）は golden を持たないので外すものは無い。
    """
    if full or layers:
        return frozenset()
    dropped = [spec for spec in series_cases(dtype) if spec.full_size]
    return frozenset(name for spec in dropped for name in case_file_names(spec.name(patch_size)))


@contextmanager
def _write_root(series: Path, *, check: bool) -> Iterator[Path]:
    """部品ディレクトリの親（`check` なら系列の中の一時ディレクトリ — 抜けるときに消える）。

    作業席（`staged_publication`）はこの下に開くので、`check` の回の書き込みは系列の部品
    ディレクトリに 1 バイトも届かない。
    """
    if not check:
        yield series
        return
    with tempfile.TemporaryDirectory(dir=series, prefix=CHECK_PREFIX) as scratch:
        yield Path(scratch)


def emit(args: argparse.Namespace) -> dict[str, Any]:
    """製品のグラフ（または `--layers` の計測用グラフ）を書き、要約を返す。

    ケースは 1 本ずつ組んで書いて捨てる（実寸のケースは各ブロックの出力を持つものだけで
    2.6 GB あり、全ケースを同時に持つとメモリに載らない）。グラフの export は全ケースの eager の後
    （例示入力は先頭）。

    MUST: eager の要約が 1 本でも {@link eager_failures} に掛かれば、作業席の中で止める — 系列の
    既存の final は 1 バイトも変わらない。止めずに据えると、壊れたラッパの golden（`io.*` も同じ
    ラッパから採る）が TS 側の突合に届くのは公開の後になる。

    MUST: `args.check` の回は部品ディレクトリを系列の中の一時ディレクトリへ書き、既存の部品
    ディレクトリと全ファイルの sha256 で突き合わせるだけ（{@link series_mismatches} — 系列は置き換え
    ない）。照合の相手が無ければ、上流を読む前に止める。
    """
    started = time.perf_counter()
    series = PROBE_SERIES if args.layers else series_dir(args.dtype)
    existing = series / TARGET
    if args.check and not existing.is_dir():
        raise SeriesMismatchError(f"照合の相手の部品ディレクトリが無い（{existing}）")
    # 計測用のグラフは golden を書かないので、例示入力（先頭のケース）だけを組む。
    specs = CASES[:1] if args.layers else series_cases(args.dtype, full=not args.no_full)
    model = load_transformer(args.model)
    wrapper_class = dit_patch.WanDitTokensLayers if args.layers else dit_patch.WanDitTokens
    wrapper = wrapper_class(model)
    rounded, scales = fake_quant(args.dtype, model, wrapper)
    print(f"[fake-quant] {TARGET}: {rounded}", flush=True)
    references_f64 = float64_references(model, specs, args.dtype)
    first = build_case(model, specs[0])
    eager: list[dict[str, Any]] = []
    written: list[str] = []

    with _write_root(series, check=args.check) as root:
        out_dir = root / TARGET
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            if not args.layers:
                for spec in specs:
                    case = first if spec is specs[0] else build_case(model, spec)
                    report, output = eager_report(wrapper, model, case)
                    reference_f64 = references_f64.get(case.name)
                    if reference_f64 is not None:
                        report["reference_f64_seconds"] = round(reference_f64.seconds, 1)
                        report["reference_f32_vs_f64_ratio"] = _ratio(
                            case.reference, reference_f64.output
                        )
                    print(f"[eager] {json.dumps(report, ensure_ascii=False)}", flush=True)
                    failures = eager_failures(report)
                    if failures:
                        raise EagerEquivalenceError(f"{case.name}: {failures}")
                    eager.append(report)
                    written += _write_case_files(case, output, reference_f64, staged)
                    del case, output
            graph = export_to_file(
                wrapper,
                first.inputs,
                staged / MODEL_FILE,
                provenance=provenance(args.model),
                graph_name=TARGET,
                assets=rope_base_asset(model),
                dynamic_shapes=dynamic_shapes(),
                symbol_names=("S",),
                weight_dtype=args.dtype,
                weight_scales=scales,
                preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            )
            declared = [entry.name for entry in graph.inputs]
            if not args.layers and declared != list(INPUT_NAMES):
                raise AssertionError(
                    f"グラフ入力名が宣言と不一致: {declared} vs {list(INPUT_NAMES)}"
                )
        model_bytes = sum(part.stat().st_size for part in container_parts(out_dir / MODEL_FILE))
        check: dict[str, Any] | None = None
        if args.check:
            patch_size = tuple(int(size) for size in model.config.patch_size)
            absent = check_absent(args.dtype, patch_size, full=not args.no_full, layers=args.layers)
            mismatches = series_mismatches(out_dir, existing, absent=absent)
            if mismatches:
                raise SeriesMismatchError(
                    f"書き直した {out_dir.name} が既存の系列（{existing}）と違う:"
                    f" {json.dumps(mismatches, ensure_ascii=False)}"
                )
            check = {
                "existing": str(existing),
                "identical_files": sorted(file_digests(out_dir)),
                "skipped_existing": sorted(absent),
            }
    breakdown = storage_breakdown(graph)
    return {
        "target": TARGET,
        "dtype": args.dtype,
        "dir": str(existing),
        "layers": args.layers,
        "check": check,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": model_bytes,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "inputs": [[entry.name, list(entry.shape)] for entry in graph.inputs],
        "output_shapes": [list(graph.values[name].shape) for name in graph.outputs],
        "io": written,
        "eager": eager,
        "seconds": round(time.perf_counter() - started, 1),
    }


def verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    """パッチ前後の eager 同値だけを実重みで実測する（容器は書かない — 判定は {@link main} が
    {@link eager_failures} で掛ける）。"""
    model = load_transformer(args.model)
    wrapper = dit_patch.WanDitTokens(model)
    if not args.no_f16:
        fake_quant(args.dtype, model, wrapper)
    return [
        eager_report(wrapper, model, build_case(model, spec))[0]
        for spec in series_cases(args.dtype, full=not args.no_full)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument(
        "--dtype",
        default="f16",
        choices=DTYPES,
        help="系列の格納（i8 は transformer だけの …-i8-dyn/ — ADR 0120 決定 2）",
    )
    parser.add_argument(
        "--layers",
        action="store_true",
        help="層別の出口を足した計測用のグラフを …-probe/ へ書く（golden は書かない）",
    )
    parser.add_argument(
        "--verify", action="store_true", help="パッチ前後の eager 同値だけを測る（容器は書かない）"
    )
    parser.add_argument(
        "--no-f16",
        action="store_true",
        help="--verify で f16 の丸めを掛けない（素の f32 の重みで測る）",
    )
    parser.add_argument(
        "--no-full",
        action="store_true",
        help="i8 系列の実寸（S = 14,040）のケースを採らない（段の分割用 — 後で無しで書き直す）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="系列の中の一時ディレクトリへ書き直し、既存の部品ディレクトリと全ファイルの sha256 で"
        "突き合わせるだけ（系列は置き換えない）",
    )
    args = parser.parse_args(argv)
    if args.no_f16 and (not args.verify or args.dtype != "f16"):
        parser.error("--no-f16 は f16 の --verify とだけ使う（素の f32 の重みで同値だけを測る口）")
    if args.no_full and args.dtype != "i8":
        # f16 系列は 27 ケースの揃った golden を据えている（ADR 0118 段 3 / 8）。欠けた golden で
        # 据え替える口を作らない。
        parser.error("--no-full は --dtype i8 とだけ使う")
    if args.layers and args.dtype != "f16":
        parser.error("--layers は f16 系列だけ（計測用の層別出口 …-f16-dyn-probe/）")
    if args.verify and args.layers:
        parser.error("--verify と --layers は併用しない")
    if args.verify and args.check:
        parser.error("--verify と --check は併用しない（--verify は容器を書かない）")
    if args.verify:
        reports = verify(args)
        print(json.dumps(reports, indent=1, ensure_ascii=False))
        failures = {
            report["case"]: found for report in reports if (found := eager_failures(report))
        }
        if failures:
            print(
                f"eager 同値が崩れた: {json.dumps(failures, ensure_ascii=False)}", file=sys.stderr
            )
            return 1
        return 0
    print(json.dumps(emit(args), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
