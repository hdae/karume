"""SBV2 **生成ネット**（net_g）の量子化パターン別 音質劣化を torch CPU で実測する。

`measure_quant_anima.py`（perf-c Q0）と同じ思想の品質ゲート — 実行カーネルを作る前に、
「その量子化で音がどこまで劣化するか」だけを torch の fake-quant で先に答える。GPU コードは
0 行で、ここで測るのは**量子化そのものの質**（ADR 0006 の fake-quant 方法論では E2E は
実装誤差しか測らないため、品質は別軸で測る必要がある）。

14 構成を**同一発話・同一乱数**（`outputs/examples/karume-sbv2-jvnv/sbv2-dump/dump.safetensors`
の離散入力・ノイズ列）で走らせる。既存 5 構成は**生成ネット側だけ**を振る
（**BERT は f32 固定**）:

    (1) f32    基準。`sbv2.demo reference` と同じ経路（既存 reference.wav とビット一致）
    (2) f16    適格重みを f16 表現可能値へ丸め（`quantize.round_weights_to_f16`）。活性は f32
               = 実装済みの **f16 格納系列**（ADR 0027）の対応物
    (3) w8     重みを per-channel symmetric i8 へ fake-quant（`quantize.fake_quant_int8`）
               = ADR 0019 の w8a32（i8 格納・f32 計算）の対応物
    (4) w8a16  (3) + **対象 op の入力活性**を f16 へ丸め
    (5) w8a8   (3) + **対象 op の入力活性**を per-位置 symmetric i8 へ fake-quant

w4 の 6 構成（方式スクリーニングの勝者 4 種・group 長 g は**既定 32 /`--w4-group-size` で
可変** — ADR 0069 追記 5 / 波 J-3 の g 軸）。丸めは全て core（`karume.quantize` /
`karume.quant_methods`）の共有で、台本ローカルの実装は持たない:

    (6) w4-rtn      net_g の**全役割**（conv1d / conv_transpose1d / linear / embedding）を
                    RTN group symmetric i4（`fake_quant_int4` の `op_types` opt-in）
    (7) w4-nf4      同じ全役割を NF4 の固定表 × group absmax scale
    (8) w4-mxfp4    同じ全役割を FP4 e2m1 表 × group の 2 のべき scale（OCP MX）
    (9) w4-kmeans   同じ全役割を k-means codebook（層をまたぐ共有表 × group 正規化）
    (10) bert-w4-rtn  **BERT（DeBERTa）の linear だけ**を RTN i4（group 長 g）・net_g は f32 固定
    (11) bert-w4-nf4  同上を NF4

BERT 側の 2 本が**配布の本命** — `shared/text_encoder` は配布 karume-sbv2-jvnv の 3 割弱を
占め、しかも linear = 今日の i4 実行経路そのもの（ADR 0069 決定 5）なので、聴感が通れば
そのまま配布候補になる。net_g 側の w4 は逆に**格納形が無い**（conv 系の scale は受容野を
平坦化した rank 2 で emit へ構造的に渡せない — 追記 5）ので、「非 linear の w4 は品質が
どこまで戻るか」を GPU コード 0 行で先に答える測定列。net_g の **linear 限定**の形は品質を
測らない — 対象が 6 本 / 適格要素の 1% 未満で、**利得が無いこと自体が採否の根拠**になる
ため、サイズ試算だけを report へ出す。

量子化軸（各 op の in 軸 — conv は受容野 `Cin·K`）が group 長で割り切れない重みは対象から
**除外**する（i4 は端数 group を作らない MUST — ADR 0069 決定 2 で core が fail loudly する）。
除外一覧・役割別の対象規模・サイズ試算は `report.json` の `w4` 節。

## 校正付き丸め（{@link CALIB_CONFIGS}・波 J-2）

上の w4 は重みだけを見て丸める（RTN 系）。**校正付き丸め**（GPTQ — core の
`karume.quant_calib`）は「その層に実際に流れる活性」から**同じ格子の中で**丸め先を選び直す
方式で、BERT 側に 3 構成を足す（net_g は f32 固定・格子は上の方式グリッドと 1 バイトも
変わらない）:

    (12) bert-gptq-rtn     BERT の linear を GPTQ × RTN i4 g32（`bert-w4-rtn` の校正版）
    (13) bert-gptq-nf4     同上を NF4 の固定表（`bert-w4-nf4` の校正版）
    (14) bert-gptq-kmeans  同上を k-means 表（core の `kmeans_shared` = **層ごと** 1 枚）

校正入力は {@link deberta.calib_texts.CALIB_TEXTS} の 48 文を既存の DeBERTa tokenizer 経路で採り、
先頭 encoder layer への呼び出しを捕まえて stage 列へ流す（{@link capture_stage_batches}）。
stage は**特徴を採る層まで**なので、対象は `bert:linear` の census の部分集合になる
（{@link CALIB_TARGET}）。`--calib-limit` で縮小実行でき、縮小した事実は表と `report.json` の
両方へ残る。

    uv run --group sbv2 python -m sbv2.measure_quant

出力は `outputs/bench/karume-sbv2-jvnv/<日付>_quant-sim/`（`<config>.wav` 14 本 +
`report.json`）。`--configs` で主要構成を名前で絞れる（`f32` は SNR の基準なので常に走る）。
w4 の group 長は
`--w4-group-size`（2 冪かつ 16 以上・既定 32）で振れる — 適格判定・丸め・方式名の 3 つへ
同じ g が流れる（校正付き構成の格子は 32 のまま = {@link CALIB_GROUP_SIZE}）。

## 活性量子化の粒度と適用点

適用点は **対象 op の入力**（出力ではない）。対象は重みスロットを持つ op のうち活性が連続値
であるもの、すなわち `conv1d` / `conv_transpose1d` / `linear`。`embedding` は入力が整数 ID
なので対象外（重み側だけが量子化される）。

**MUST: 差し替えは op 粒度**（`torch.nn.functional` の関数を差し替える）— モジュールの
`forward_pre_hook` では足りない。`sbv2.patch._patched_ffn_forward` は `self.conv_1(x)` では
なく `functional.conv1d(x, self.conv_1.weight, …)` を直に呼ぶので、`nn.Conv1d` へ掛けた
フックは 1 度も発火しない（この台本の初版が踏んだ穴 — enc_p 12 本 + flow 48 本の FFN conv が
丸ごと素通りしていた。しかも「量子化しているつもりで劣化が軽く出る」向きに沈黙する）。
Karume 側も IR とカーネルは op 粒度で活性を食うので、op 粒度が意味論的にも正しい。

粒度は ADR 0025 の **per-token ±127** を conv1d へ移したもの:

    linear           [..., k]     行 = 最終軸（ADR 0025 そのもの）
    conv1d           [B, Cin, T]  行 = **時刻 t ごとに Cin 軸**（`dim=-2` で amax）
    conv_transpose1d [B, Cin, T]  同上（重みの `[Cin,Cout,K]` 転置は重み側の話で活性には
                                  効かない）

数値式は `karume.act_quant.quantize_rows` の**再利用**（写経しない）— 軸を最終軸へ
転置して同じ関数へ通す。したがって `s = clamp(rowmax|x|/127, tiny)` / `q = clamp(round(x/s),
±127)` / `x̂ = q·s`（偶数丸め・全ゼロ行は厳密 0）は 1 箇所の正本を共有する。

**なぜ「時刻ごと」なのか**（将来のカーネルが実装しうる粒度であること）: conv1d を
implicit GEMM で組むと縮約軸は `Cin·K`（タップ込み）になり、im2col の 1 列は K 個の時刻から
値を集めるので「列 1 本 = scale 1 個」にするには窓全体の amax が要る。一方 conv1d は
**タップごとの 1x1 conv（= GEMM）の shift 加算**にも分解でき、その形なら 1 タップの縮約軸は
`Cin` だけなので「時刻ごとに 1 scale」が ADR 0025 の per-token とそのまま同型になり、
`(Σ q_a·q_w)·(s_a·s_w)` の整数縮約が成立する（タップ間は f32 で足す）。ここで測る粒度は
その分解形に対応する。窓 amax 方式（列 1 本 1 scale）は同じ時刻集合の max を取るぶん
**必ずこちらより粗い**ので、この計測は w8a8 conv1d の**楽観側の上界**として読むこと。

NOTE: `k % 4 == 0` のような i8 詰めの整列条件は模さない（パディングで満たせる実装詳細で、
品質の問いには効かない）。

## 時間グリッドの固定（SNR を意味のある量にするための MUST）

front の出力 `logw` は duration を決めるので、量子化で `ceil` が 1 つでも飛ぶと**フレーム数
そのものが変わり**、波形長が構成間で食い違って SNR が定義できなくなる。そこで全構成とも
展開には **dump 側の `w_ceil`** を使う（`sbv2.demo.run_reference` が zp_noise の形を理由に
そうしているのと同じ選択）。構成ごとの「自前 w_ceil なら何フレームになったか」は
`report.json` の `w_ceil` 欄に診断として残す — こちらが割れていれば「SNR は同じ時間グリッド
上での比較であって、実運用では発話長も変わる」と読む必要がある。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import struct
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch
from safetensors.torch import load_file
from torch import nn

from _shared.paths import BENCH_ROOT, DIST_ROOT, EXAMPLES_ROOT
from deberta.calib_texts import CALIB_TEXTS
from karume.act_quant import quantize_rows
from karume.ir import MIN_GROUP_SIZE
from karume.quant_calib import (
    CalibMethod,
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quant_methods import (
    DEFAULT_CODEBOOK_LEVELS,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QUANT_MODULE_TYPES,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
    round_weights_to_f16,
)

from . import demo, export, patch

#: dump の既定入力（`examples/sbv2/dump.ts` の既定出力先 — 綴りは向こうが正本）。
DEFAULT_DUMP_DIR = EXAMPLES_ROOT / "karume-sbv2-jvnv" / "sbv2-dump"
DEFAULT_DUMP = DEFAULT_DUMP_DIR / "dump.safetensors"
#: f32 構成の恒真化を防ぐ突合先（`sbv2.demo reference` が同じ dump から書いた WAV）。
DEFAULT_REFERENCE_WAV = DEFAULT_DUMP_DIR / "reference.wav"
#: 測定結果の置き場。ベンチ生成物なので資産（`sbv2.demo.DEFAULT_DEMO_DIR`）とも系列とも
#: 分離する — 掃除（`rm -rf outputs/bench`）がそれらを巻き込まないため（docs/assets-layout.md）。
DEFAULT_OUT = BENCH_ROOT / "karume-sbv2-jvnv" / f"{date.today().isoformat()}_quant-sim"
#: サイズ縮小率の分母になる配布形（`karume dist` の生成物 — 実バイトを読むだけで書かない）。
DEFAULT_DIST_DIR = DIST_ROOT / "karume-sbv2-jvnv"


# ---- w4 の語彙（方式・役割）— ADR 0069 追記 5 --------------------------------

# w4 の group 長 g は `--w4-group-size`（既定 = core の格納既定 `DEFAULT_GROUP_SIZE`）で振る —
# 波 J-3 の g 軸。方式の差だけを見たいときは既定のまま走らせる（2 軸を同時に振ると方式の差と
# g の差が混ざる — ADR 0069 追記 5 の位置づけ）。
#
# MUST: g は module 直下の定数ではなく**引数で流す**。適格判定（`census_w4_targets`）・丸め
# （`w4_methods` が `fake_quant_*` へ渡す `group_size`）・方式名（`rtn_method_name`）の 3 つが
# 1 個の値を共有すること — 片方だけ動かすと「g16 と書いてある g32 の測定」が黙って出る
# （丸めだけ動けば端数 group で core が fail loudly するが、適格判定だけ動くと数字は出たまま
# 格子が食い違う）。校正付き構成の格子はこの軸に**乗らない**（`CALIB_GROUP_SIZE`）。

#: k-means の共有表 1 枚のビット数（16 準位 × f32）。品質だけ見ると**表の代金**が見えない
#: ので、サイズ試算では表のコストを式へ明示的に載せる。
CODEBOOK_BITS = DEFAULT_CODEBOOK_LEVELS * 32


@dataclass(frozen=True)
class TargetCounts:
    """w4 対象集合の計数（サイズ試算の入力）。

    group 長を計数と同じ器に持つのは、**「数えた g」と「投影した g」が黙って割れない**ため —
    bpw の投影は group 数から出るので、g を別引数で持ち回ると片方だけ古い値になれる。
    """

    modules: int
    channels: int
    elements: int
    group_size: int

    def __add__(self, other: TargetCounts) -> TargetCounts:
        if self.group_size != other.group_size:
            raise AssertionError(
                f"group 長の違う計数は足せない（{self.group_size} と {other.group_size}）"
            )
        return TargetCounts(
            modules=self.modules + other.modules,
            channels=self.channels + other.channels,
            elements=self.elements + other.elements,
            group_size=self.group_size,
        )

    @property
    def groups(self) -> int:
        """group の総数（group ごとに scale 1 個が要る方式の試算に使う）。

        `elements = Σ チャネル数 × in 軸`で、対象は in 軸が group 長で割り切れるものだけ
        （割り切れない重みは {@link census_w4_targets} が除外する）なので端数は出ない。
        """
        return self.elements // self.group_size


class QuantReport(Protocol):
    """core の `fake_quant_*` が返す計数レポートの共通面（`Int4Report` / `MethodReport`）。"""

    modules: int
    elements: int

    def describe(self) -> str: ...


@dataclass(frozen=True)
class W4Method:
    """w4 の丸め方式 1 種 — 名前・当て方・サイズ試算の式を 1 行に束ねる。

    3 つを別表へ散らすと「品質を測った方式」と「サイズを試算した方式」が黙って割れる
    （方式を 1 種足したときに片方だけ更新される形になる）。丸めの実装は全て core の共有で、
    ここは呼び分けだけを持つ — 台本ローカルに丸めを書くと、測った式と出荷する式が独立に
    動きはじめる。
    """

    #: 表と JSON に出る方式名（RTN だけ g を焼く — {@link rtn_method_name}）。表の**鍵**は
    #: g に依らない方式の種なので、g を振っても構成表 {@link CONFIGS} の指す先は動かない。
    name: str
    #: 丸めを model へ in-place で当てる（`(model, op_types, include)`）。group 長は
    #: {@link w4_methods} が閉じ込むので、呼ぶ側が g を持ち回らなくても格子は 1 個に決まる。
    apply: Callable[[nn.Module, tuple[type[nn.Module], ...], Callable[[str], bool]], QuantReport]
    #: 量子化対象集合の投影ビット数。
    projected_bits: Callable[[TargetCounts], float]
    #: `projected_bits` の式（**出力へそのまま載せる** — 投影の前提を表から追えるように）。
    formula: str


#: RTN（= 唯一の格納形 `i4`）の**方式の種** = {@link w4_methods} の鍵。方式列の**基準**で、
#: 他の 3 種はこれとの差で読む。
RTN_KIND = "rtn"


def rtn_method_name(group_size: int) -> str:
    """RTN の方式名（**g を名前に焼く**）。

    MUST: 既定の 32 で従来と同じ `rtn-i4-g32` になること — 過去の研究記録（ADR 0069 追記 5 の
    方式スクリーニング）がこの綴りで数値を残しており、既定の実行の突合先を動かさないため。
    他の 3 種は名前に g を持たない（同じ理由で綴りを動かさない）ので、g は `w4` 節の
    `group_size` と各方式の式（{@link W4Method.formula}）から読む。
    """
    return f"rtn-i4-g{group_size}"


def w4_methods(group_size: int) -> Mapping[str, W4Method]:
    """方式 4 種（スクリーニングの勝者）を group 長 `group_size` で束ねた表。

    鍵は**方式の種**（g に依らない）で、値だけが g を担ぐ — 構成表 {@link CONFIGS} が指す先を
    g で動かさないため。表を g から**作る**のは、丸めへ渡す `group_size`・表に出る式・方式名を
    同じ 1 個の g から導くため（別々に持つと「g16 と書いてある g32 の測定」が作れる）。

    落選済みで載せないもの: FP4 と k-means の per_tensor / per_channel（安いファミリ 2 本の
    実測で落選 — ADR 0069 追記 5）。
    """
    return MappingProxyType(
        {
            RTN_KIND: W4Method(
                rtn_method_name(group_size),
                lambda model, op_types, include: fake_quant_int4(
                    model, group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups,
                f"4bit + g{group_size} f32 scale = {4 + 32 / group_size} bpw",
            ),
            "nf4": W4Method(
                "nf4",
                lambda model, op_types, include: fake_quant_nf4(
                    model, group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups,
                f"4bit + g{group_size} f32 scale = {4 + 32 / group_size} bpw"
                "（準位表は固定値なので模型に載らない）",
            ),
            "mxfp4": W4Method(
                "mxfp4",
                lambda model, op_types, include: fake_quant_mxfp4(
                    model, group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 8 * counts.groups,
                f"4bit + g{group_size} E8M0 scale = {4 + 8 / group_size} bpw",
            ),
            "kmeans:shared": W4Method(
                "kmeans:shared",
                lambda model, op_types, include: fake_quant_kmeans(
                    model, "shared", group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups + CODEBOOK_BITS,
                f"4bit + g{group_size} f32 scale + 表 {DEFAULT_CODEBOOK_LEVELS}×f32 を全体で 1 枚",
            ),
        }
    )


#: w4 の対象**役割** → `op_types`。`all` は i8 と同じ 5 op 種の表（`QUANT_MODULE_TYPES`）を
#: そのまま渡す形で、net_g に居るのはうち 4 種（conv1d / conv_transpose1d / linear /
#: embedding）— 表を写した別リストにはしない（独立に動くと「軸を引ける型」と「対象にする型」
#: が黙って割れる）。`linear` は**今日の配布対応形**（i4 の実行経路は linear 限定 —
#: ADR 0069 決定 5）。
W4_ROLES: Mapping[str, tuple[type[nn.Module], ...]] = MappingProxyType(
    {"all": QUANT_MODULE_TYPES, "linear": (nn.Linear,)}
)


# ---- 校正付き丸めの語彙（波 J-2）--------------------------------------------

#: 校正付き丸めの group 長 = **32 に釘付け**（`--w4-group-size` の g 軸には乗らない）。校正は
#: **格子を 1 バイトも変えない**（同じ格子の中で丸め先を選び直すだけ — `karume.quant_calib` の
#: モジュール docstring）方式の比較なので、振るべきは方式であって g ではない。方式グリッド側の
#: g を借りずに独立した定数にしてあるのは、g 軸を振ったときに校正の格子が**黙って一緒に動く**
#: のを防ぐため — 食い違いは `w4.group_size` と `w4.calib.group_size` が別々に出ることで読める。
CALIB_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: 校正付き構成の対象名（`w4` 節の対象列）。`bert:linear` の census とは**集合が違う** —
#: 校正の stage は特徴を採る層までしか作らない（{@link bert_stages}）ので、それ以降の層の
#: linear は入らない。代わりに**配布形（22 層 variant）の linear 集合と一致する**ので、
#: {@link project_distribution} の縮小試算がそのままこの集合の話になる。
CALIB_TARGET = "bert:linear:calib"


def group_scale_bits(counts: TargetCounts) -> float:
    """4bit ペイロード + group ごとの f32 scale（RTN / NF4 の格納そのもの）。"""
    return 4 * counts.elements + 32 * counts.groups


def layer_table_bits(counts: TargetCounts) -> float:
    """上に**層ごと 1 枚**の k-means 表を足した式。

    core の `kmeans_shared` は「group absmax で正規化した値空間に 1 枚の表」を張るが、その
    射程は**層内**に閉じる（stage 逐次の駆動と全層を跨ぐ表が噛み合わない —
    {@link karume.quant_calib.GridSpec} の NOTE）。方式グリッドの `kmeans:shared`（全体で
    1 枚 = {@link CODEBOOK_BITS} を 1 回だけ）と式が違うのはこのため。
    """
    return group_scale_bits(counts) + CODEBOOK_BITS * counts.modules


@dataclass(frozen=True)
class CalibConfig:
    """校正付き丸め 1 本ぶんの指定（方式 × 格納グリッド）。

    丸めの実装は core（`karume.quant_calib`）の共有で、ここは呼び分けと投影式だけを持つ —
    {@link W4Method} と同じ理由（台本ローカルに丸めを書くと、測った式と出荷する式が独立に
    動きはじめる）。
    """

    name: str
    method: CalibMethod
    grid: GridSpec
    #: 量子化対象集合の投影ビット数（{@link W4Method.projected_bits} と同じ器）。
    projected_bits: Callable[[TargetCounts], float]
    #: `projected_bits` の式（**出力へそのまま載せる**）。
    formula: str

    @property
    def label(self) -> str:
        """表と JSON に出る方式名（`<方式>/<格納グリッド>`）。"""
        return f"{self.method}/{self.grid.kind}"


#: 校正グリッド 3 本 = GPTQ × 格納グリッド 3 種（core の `GRID_KINDS` 全部）。既存の
#: `bert-w4-rtn` / `bert-w4-nf4` の**校正版**で、格子は 1 バイトも変わらない。
#:
#: AWQ 系を置かないのは、丸めた重みが `W_eff = Q(W')/s` の形になり、等価倍率 `s` を隣接演算へ
#: fold するか companion テンソルとして配るまで**単独で格納できない**から
#: （`karume.quant_calib` のモジュール docstring）。ここは配布候補の**聴感を WAV で確かめる**
#: 席なので、格納の当てが無い列は聴く WAV を増やすだけになる（格納形が無いことを測る席は
#: net_g 側の w4 4 本 — 立場が逆）。
CALIB_CONFIGS: Mapping[str, CalibConfig] = MappingProxyType(
    {
        config.name: config
        for config in (
            CalibConfig(
                "gptq-rtn",
                "gptq",
                GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
                group_scale_bits,
                "4bit + g32 f32 scale = 5.0 bpw（格納は RTN i4 そのもの）",
            ),
            CalibConfig(
                "gptq-nf4",
                "gptq",
                GridSpec(kind="nf4", group_size=CALIB_GROUP_SIZE),
                group_scale_bits,
                "4bit + g32 f32 scale = 5.0 bpw（格子は NF4 の固定表）",
            ),
            CalibConfig(
                "gptq-kmeans",
                "gptq",
                GridSpec(kind="kmeans_shared", group_size=CALIB_GROUP_SIZE),
                layer_table_bits,
                f"4bit + g32 f32 scale + 表 {DEFAULT_CODEBOOK_LEVELS}×f32 を**層ごと** 1 枚",
            ),
        )
    }
)


@dataclass(frozen=True)
class Recipe:
    """1 構成の量子化レシピ。

    `scope` は丸めを当てる**サブグラフ**（`front` = enc_p + dp + sdp / `voice` = flow + dec）。
    net_g を振る構成は両方に当てる（= 生成ネット全体）。片側だけに当てる形は診断用（下）で、
    BERT だけを振る構成は空（net_g は f32 のまま）。
    """

    weight: str | None
    act: str | None
    scope: tuple[str, ...] = ("front", "voice")
    #: w4 の方式の種（`weight == "w4"` のときだけ意味を持つ — {@link w4_methods} の鍵）。
    method: str | None = None
    #: w4 の対象役割（同上 — {@link W4_ROLES} の鍵）。
    roles: str = "all"
    #: BERT（DeBERTa）の linear へ当てる w4 方式。**None = f32 固定**（既存 5 構成の契約）。
    bert_method: str | None = None
    #: BERT の linear へ当てる**校正付き**丸め（{@link CALIB_CONFIGS} の鍵）。
    #: `bert_method` との併用は構成の書き間違い（{@link bert_variant} が落とす）。
    bert_calib: str | None = None


#: 主要 14 構成（WAV を `--out` 直下へ書き、聴き比べの対象になる）。
CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        "f32": Recipe(None, None),
        "f16": Recipe("f16", None),
        "w8": Recipe("i8", None),
        "w8a16": Recipe("i8", "f16"),
        "w8a8": Recipe("i8", "i8"),
        # net_g の全役割 w4（格納形は無い測定列 — モジュール docstring）。
        "w4-rtn": Recipe("w4", None, method=RTN_KIND),
        "w4-nf4": Recipe("w4", None, method="nf4"),
        "w4-mxfp4": Recipe("w4", None, method="mxfp4"),
        "w4-kmeans": Recipe("w4", None, method="kmeans:shared"),
        # BERT の linear だけ（net_g は f32 固定 — 既存 5 構成と直交する分離軸）。
        "bert-w4-rtn": Recipe(None, None, scope=(), bert_method=RTN_KIND),
        "bert-w4-nf4": Recipe(None, None, scope=(), bert_method="nf4"),
        # 上 2 本の校正版（GPTQ・net_g は f32 固定・格子は同じ — 波 J-2）。
        "bert-gptq-rtn": Recipe(None, None, scope=(), bert_calib="gptq-rtn"),
        "bert-gptq-nf4": Recipe(None, None, scope=(), bert_calib="gptq-nf4"),
        "bert-gptq-kmeans": Recipe(None, None, scope=(), bert_calib="gptq-kmeans"),
    }
)

#: 診断構成（`--out/diagnostics/` へ書く）。劣化を front 側と voice 側へ**直交分解**する。
#:
#: 先行実験の既録値（`docs/research/2026-08-03-demo-w8-perf-recon.md`: int8 per-channel の
#: voice SNR 12.8〜24.2dB）は **voice 単体**の測定なので、全体構成の数とは直接比較できない。
#: 桁合わせの相手はこの `w8-voice-only` の方。
DIAGNOSTICS: Mapping[str, Recipe] = MappingProxyType(
    {
        "w8-front-only": Recipe("i8", None, ("front",)),
        "w8-voice-only": Recipe("i8", None, ("voice",)),
        "w8a8-front-only": Recipe("i8", "i8", ("front",)),
        "w8a8-voice-only": Recipe("i8", "i8", ("voice",)),
    }
)

#: 構成名 → レシピの全表（`run_config` はここから引く）。
RECIPES: Mapping[str, Recipe] = MappingProxyType({**CONFIGS, **DIAGNOSTICS})

#: 活性量子化の対象 **op**（`torch.nn.functional` の名前）→ 入力テンソルの「行」の軸。
#:
#: MUST: 適用点は**モジュールではなく op**。モジュールの `forward_pre_hook` で足りると
#: 考えるのは誤りで、`sbv2.patch._patched_ffn_forward` は `self.conv_1(x)` ではなく
#: `functional.conv1d(x, self.conv_1.weight, …)` を直に呼ぶため、`nn.Conv1d` へ掛けたフックは
#: **1 度も発火しない**（実測: enc_p 12 本 + flow 48 本の FFN conv が丸ごと素通りしていた）。
#: Karume 側も IR/カーネルは op 粒度で活性を食うので、op 粒度が意味論的にも正しい。
#:
#: `embedding` は入力が整数 ID なので載せない（重みだけが量子化される）。
ACT_ROW_AXIS: Mapping[str, int] = MappingProxyType(
    {
        "conv1d": -2,
        "conv_transpose1d": -2,
        "linear": -1,
    }
)

#: 差し替え前の op 実体（復元検査の正本 — import 時に 1 度だけ捕まえる）。
PRISTINE_OPS: Mapping[str, Any] = MappingProxyType(
    {op: getattr(torch.nn.functional, op) for op in ACT_ROW_AXIS}
)

#: 層名の連番を潰して役割でまとめる（`flow.flows.0.enc.in_layers.3|conv1d` →
#: `flow.flows.*.enc.in_layers.*|conv1d`）。キーは `<モジュール FQN>|<op>` の形。
_INDEX = re.compile(r"\.\d+(?=[.|]|$)")


def layer_group(name: str) -> str:
    return _INDEX.sub(".*", name)


# ---- 活性の fake-quant フック ------------------------------------------------


@dataclass
class ActStat:
    """1 呼び出し地点（モジュール名 × op）ぶんの活性量子化誤差（通算の二乗和）。"""

    op: str
    axis: int
    calls: int = 0
    err2: float = 0.0
    ref2: float = 0.0
    saturated: int = 0

    def rel_rms(self) -> float:
        return math.sqrt(self.err2 / self.ref2) if self.ref2 > 0 else 0.0


def round_to_f16(x: torch.Tensor) -> torch.Tensor:
    """活性を f16 表現可能値へ丸める（計算は f32 のまま）。

    重み側 `round_weights_to_f16` は飽和を fail loudly にするが、活性は実行時にしか値が
    決まらないので**数えて報告する**（飽和が起きること自体が w8a16 の可否に対する答えで、
    途中で落として残り 4 構成の測定を捨てる方が損）。
    """
    return x.to(torch.float16).to(torch.float32)


def quantize_activation(x: torch.Tensor, axis: int) -> torch.Tensor:
    """`axis` を行とみなした per-行 symmetric i8 の fake-quant。

    数値式は `act_quant.quantize_rows`（最終軸を行とする ADR 0025 の正本）を**そのまま**
    使い、軸だけ転置で合わせる — 式を書き写すと片方だけが仕様から外れる。
    """
    if axis in (-1, x.dim() - 1):
        return quantize_rows(x)
    return quantize_rows(x.transpose(axis, -1)).transpose(axis, -1).contiguous()


@dataclass
class ActQuant:
    """対象 op の**入力**を丸め、呼び出し地点ごとの誤差を集計する。

    仕組みは 2 段:

    1. `torch.nn.functional.{conv1d, conv_transpose1d, linear}` を**差し替える**（丸めの本体）。
       op 粒度なので、モジュール経由の呼び出しもパッチ層が直接呼ぶ呼び出しも同じ 1 箇所で
       捕まる（{@link ACT_ROW_AXIS} の MUST）。
    2. `roots` の配下 **全モジュール**へ push/pop フックを掛けて名前スタックを持つ（丸めない
       — 誤差を「どの層で起きたか」へ帰属させるためだけ）。

    スタックが空 = `roots` の外なので丸めない。これが `scope` の実装で、front だけを量子化する
    診断構成では voice の呼び出しがそのまま素通りする（グローバル差し替えのままスコープを
    切る唯一の手段）。

    `mode` は `"f16"`（w8a16）か `"i8"`（w8a8）。差し替えは `detach` で必ず戻す。
    """

    roots: tuple[tuple[str, nn.Module], ...]
    mode: str
    disable: bool = False
    stats: dict[str, ActStat] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    patched: dict[str, Any] = field(default_factory=dict)
    skipped_calls: int = 0

    def attach(self) -> int:
        """名前スタックのフックを掛け、対象 op を差し替える。返り値は差し替えた op 数。

        MUST: 本数を返して呼び出し側が 0 本を検出できる形にする（`act_quant.attach_act_quant`
        と同じ流儀 — 0 本のまま走ると「w8a8 のつもりで w8 の数を採った」に気づけない）。
        """
        for tag, root in self.roots:
            for name, module in root.named_modules():
                qualified = f"{tag}.{name}" if name else tag
                self.handles.append(module.register_forward_pre_hook(self._make_push(qualified)))
                self.handles.append(module.register_forward_hook(self._pop))
        if self.disable:
            # 故障注入（`--inject drop-act-quant`）: 帰属の足場だけ作り、丸めは掛けない。
            return 0
        for op, axis in ACT_ROW_AXIS.items():
            original = getattr(torch.nn.functional, op)
            self.patched[op] = original
            setattr(torch.nn.functional, op, self._make_wrapper(op, axis, original))
        return len(self.patched)

    def detach(self) -> None:
        """差し替えとフックを必ず戻す（構成間のリーク禁止）。"""
        for op, original in self.patched.items():
            setattr(torch.nn.functional, op, original)
        self.patched.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stack.clear()

    def _make_push(self, name: str):
        def push(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            self.stack.append(name)

        return push

    def _pop(self, _module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        self.stack.pop()

    def _make_wrapper(self, op: str, axis: int, original: Any):
        def wrapper(*args: Any, **kwargs: Any):
            if not self.stack:
                # roots の外（= scope 外）。素通りさせる。
                return original(*args, **kwargs)
            if not args or not isinstance(args[0], torch.Tensor):
                # 入力がキーワードで来る呼び出しは丸めようがない。計数して可視化する。
                self.skipped_calls += 1
                return original(*args, **kwargs)
            x = args[0]
            key = f"{self.stack[-1]}|{op}"
            stat = self.stats.setdefault(key, ActStat(op=op, axis=axis))
            rounded = round_to_f16(x) if self.mode == "f16" else quantize_activation(x, axis)
            if self.mode == "f16":
                stat.saturated += int((torch.isfinite(x) & ~torch.isfinite(rounded)).sum())
            stat.calls += 1
            stat.err2 += float((rounded - x).square().sum())
            stat.ref2 += float(x.square().sum())
            return original(rounded, *args[1:], **kwargs)

        return wrapper


# ---- 指標 -------------------------------------------------------------------


def snr_db(value: torch.Tensor, reference: torch.Tensor) -> float:
    """`10·log10(Σref² / Σ(value−ref)²)`（構成間の音質比較の主指標）。"""
    err = float((value - reference).double().square().sum())
    ref = float(reference.double().square().sum())
    if err == 0.0:
        return math.inf
    if ref == 0.0:
        return -math.inf
    return 10.0 * math.log10(ref / err)


#: LSD の STFT 設定。hop は資産の `hopLength`（dec の総 upsample 率）に合わせる。
STFT_N_FFT = 2048
STFT_HOP = 512
#: 振幅の床（参照の最大振幅に対する相対値 = −100dB）。無音区間の対数が −inf へ落ちるのを
#: 防ぐためだけの下限で、両系列に**同じ床**を当てる（片側だけに当てると差が床に依存する）。
STFT_FLOOR = 1e-5


def log_spectral_distance(value: torch.Tensor, reference: torch.Tensor) -> float:
    """対数スペクトル距離 `sqrt(mean((20log10|X| − 20log10|Y|)²))` [dB]。

    波形 SNR は位相が少しずれるだけで崩れる（HiFi-GAN の励起の軌道が変われば、聴感上
    同じ音でも残差は信号と同オーダーになる）。実際この計測でも SNR 13〜16dB 帯では構成の
    順序が入れ替わる。LSD は振幅スペクトルだけを見るので**位相ずれに鈍く**、
    「音色・帯域の壊れ方」を測る側の指標として併記する。どちらか一方では判断しない。
    """
    window = torch.hann_window(STFT_N_FFT)
    spectra = [
        torch.stft(
            wave, n_fft=STFT_N_FFT, hop_length=STFT_HOP, window=window, return_complex=True
        ).abs()
        for wave in (value, reference)
    ]
    floor = float(spectra[1].max()) * STFT_FLOOR
    a, b = (20.0 * torch.log10(spectrum.clamp_min(floor)) for spectrum in spectra)
    return float((a - b).square().mean().sqrt())


def rel_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    denom = float(reference.double().square().sum())
    if denom == 0.0:
        return 0.0
    return math.sqrt(float((value - reference).double().square().sum()) / denom)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- w4 の対象集合（census）--------------------------------------------------


@dataclass(frozen=True)
class W4Census:
    """1 ルート × 1 役割の w4 対象集合 — 適格の計数と、group 長で割り切れず外した重み。

    対象選択の**正本は core**（`quantize.iter_quant_targets`）で、ここが足すのは group 長の
    割り切れ判定だけ。判定は {@link include} で core の述語へ畳んで返すので、「数えた対象」と
    「丸めた対象」は最後まで 1 つの集合のまま動く。
    """

    counts: TargetCounts
    #: 適格としたモジュール FQN（配布形のテンソル名との突合に使う — 配布形は torch の FQN を
    #: そのまま担ぐので、この集合が「今日の i4 適格」の正本になる）。
    eligible: tuple[str, ...]
    #: 除外したモジュール FQN（量子化軸が group 長 = `counts.group_size` で割り切れない）。
    excluded: tuple[str, ...]
    excluded_elements: int
    #: モジュール型名 → 適格の計数（役割別の対象規模を報告へ出すため）。
    by_role: Mapping[str, TargetCounts]

    def include(self) -> Callable[[str], bool]:
        """core の `include`（モジュール FQN の述語）— 除外集合の補集合。"""
        excluded = frozenset(self.excluded)
        return lambda name: name not in excluded


def census_w4_targets(
    root: nn.Module, op_types: tuple[type[nn.Module], ...], group_size: int
) -> W4Census:
    """`root` の w4 対象を役割別に数え、量子化軸が `group_size` で割り切れない重みを外す。

    MUST: 割り切れない重みは**構成ごと落とすのではなく対象から外す**（i4 は端数 group を
    作らない — ADR 0069 決定 2。core は割り切れなければ fail loudly するので、外さないと
    測定そのものが立たない）。実測（FN4・既定 g32）で外れるのは受容野が 32 の倍数にならない conv
    （sdp の分離 conv・入力 1 チャネルの 1x1・dec 末尾の 16 チャネル resblock）だけで、
    適格要素に対して 0.1% に満たない — 一覧と計数は `report.json` の `w4.census`
    （「黙って対象が痩せた」を読み手が検出できること）。
    """
    kinds = {name: type(module).__name__ for name, module in root.named_modules()}
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    eligible: list[str] = []
    excluded: list[str] = []
    excluded_elements = 0
    for fqn, weight, axis in iter_quant_targets(root, op_types):
        name = fqn[: -len(".weight")]
        if channel_rows(weight, axis).shape[-1] % group_size:
            excluded.append(name)
            excluded_elements += weight.numel()
            continue
        eligible.append(name)
        row = totals[kinds[name]]
        row[0] += 1
        row[1] += int(weight.shape[axis])
        row[2] += weight.numel()
    by_role = {
        kind: TargetCounts(*row, group_size=group_size) for kind, row in sorted(totals.items())
    }
    total = TargetCounts(0, 0, 0, group_size=group_size)
    for counts in by_role.values():
        total = total + counts
    return W4Census(
        counts=total,
        eligible=tuple(eligible),
        excluded=tuple(excluded),
        excluded_elements=excluded_elements,
        by_role=MappingProxyType(by_role),
    )


def net_g_census(model_dir: Path, group_size: int) -> Mapping[tuple[str, str], W4Census]:
    """net_g を 1 度だけ読んで、役割 × サブグラフの対象集合を採る（鍵は `(役割, タグ)`）。

    構成ごとに数え直さないのは `include` の由来を 1 箇所にするため — {@link run_config} は
    ここが返した集合の述語をそのまま core へ渡す。FQN 空間は読み直しても同じなので、構成ごとに
    読み直す net_g（丸めを積み重ねないための pristine 再読み込み）へそのまま効く。
    """
    net_g, _ = export.load_net_g(model_dir)
    patch.apply_all_patches()
    export.ensure_dec_plain(net_g)
    roots = {"front": patch.Sbv2Front(net_g), "voice": patch.Sbv2Voice(net_g)}
    census = {
        (roles, tag): census_w4_targets(root, op_types, group_size)
        for roles, op_types in W4_ROLES.items()
        for tag, root in roots.items()
    }
    del net_g, roots
    gc.collect()
    return MappingProxyType(census)


# ---- サイズ試算（**実測ではなく式による投影**）-------------------------------


@dataclass(frozen=True)
class SizeProjection:
    """方式 × 対象集合の投影サイズ（式は {@link W4Method.formula} が逐語で持つ）。"""

    counts: TargetCounts
    bits: float
    formula: str

    @property
    def bits_per_weight(self) -> float:
        return self.bits / self.counts.elements

    @property
    def projected_mib(self) -> float:
        return self.bits / 8 / 1024**2

    @property
    def f32_mib(self) -> float:
        """同じ対象集合を f32 で持ったときの MiB（投影の分母 — 圧縮率を表から読めるように）。"""
        return self.counts.elements * 4 / 1024**2


#: サイズ試算の対象集合 → 説明。`net_g:linear` は品質を測らない席（対象が全体の 1% 未満で、
#: **利得が無いこと自体が採否の根拠**）なので、ここにだけ現れる。
PROJECTION_TARGETS: Mapping[str, str] = MappingProxyType(
    {
        "net_g:all": "net_g の全役割（格納形は無い — conv 系の scale は emit へ渡せない）",
        "net_g:linear": "net_g の linear 限定（今日の配布対応形）",
        "bert:linear": "BERT（DeBERTa）の linear 限定（今日の配布対応形・配布の本命）",
    }
)


def build_projections(
    counts: Mapping[str, TargetCounts], methods: Mapping[str, W4Method]
) -> list[dict[str, Any]]:
    """対象集合 × 方式の全組み合わせのサイズ試算。

    どの行に品質測定が付くかは {@link CONFIGS} から**引く**（別表に書くと構成を足したときに
    片方だけ更新される）。突合は**方式の種**（= 表の鍵）で、行に出る `method` は g を焼いた
    方式名の方。
    """
    measured: dict[tuple[str, str], str] = {}
    for name, recipe in CONFIGS.items():
        if recipe.weight == "w4" and recipe.method is not None:
            measured[(f"net_g:{recipe.roles}", recipe.method)] = name
        elif recipe.bert_method is not None:
            measured[("bert:linear", recipe.bert_method)] = name
    rows: list[dict[str, Any]] = []
    for target, description in PROJECTION_TARGETS.items():
        for kind, method in methods.items():
            projection = SizeProjection(
                counts=counts[target],
                bits=method.projected_bits(counts[target]),
                formula=method.formula,
            )
            rows.append(
                {
                    "target": target,
                    "target_description": description,
                    "method": method.name,
                    "measured_by": measured.get((target, kind)),
                    "modules": projection.counts.modules,
                    "elements": projection.counts.elements,
                    "bits_per_weight": projection.bits_per_weight,
                    "projected_mib": projection.projected_mib,
                    "f32_mib": projection.f32_mib,
                    "formula": projection.formula,
                }
            )
    return rows


# ---- 配布形に対する縮小試算 --------------------------------------------------

#: 配布 safetensors の scale テンソルの接頭（`karume.scale.<重みのキー>`）。
DIST_SCALE_PREFIX = "karume.scale."

#: DeBERTa の配布グラフにだけ付くテンソル名の接頭（22 層 variant — 層番号は測定側の 24 層の
#: 部分集合なので、これを外せば丸めの対象集合と同じ FQN 空間になる）。
DIST_BERT_PREFIX = "model."


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """safetensors のヘッダ（先頭 8 バイト LE 長 + JSON）だけを読む。

    本体を読まないのは、配布形が GiB 級で、要るのが名前・dtype・`data_offsets` だけだから。
    """
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def project_distribution(
    dist_dir: Path, linear_fqns: frozenset[str], group_size: int
) -> dict[str, Any]:
    """配布形の**実ファイル**に対する「i8 → i4（group 長 `group_size`）」の縮小試算。

    対象は**今日の配布対応形 = linear の重みスロットだけ**（conv / embedding の i4 は格納形も
    実行経路も無い — ADR 0069 決定 5 / 追記 5）。配布形はテンソル名に torch の FQN をそのまま
    担ぐので、丸めの対象集合（`linear_fqns` = census が数えたのと同じ FQN）との突合で引ける。
    「rank 2 の I8」という形だけで引くと embedding が混ざる（`enc_p.emb` / `word_embeddings`）
    ので、突合しなかった rank 2 の I8 は本数を返して読み手が監査できる形にする。

    式（テンソル 1 本 `[O,I]` あたり・逐語）:

        i8 … `O·I` バイト + scale `[O,1]` の `4·O` バイト（どちらも**配布形の実バイト**）
        i4 … `O·I/2` バイト（nibble 詰め）+ scale `[O,I/g]` の `4·O·(I/g)` バイト

    分母は配布ディレクトリの実ファイル総バイト。`shared/` の下（話者間で 1 本の DeBERTa）と
    話者ごとの net_g を分けて出す — 前者は 1 本ぶん、後者は話者数ぶん効く。

    `group_size` は適格判定（`linear_fqns` を作った census）と**同じ g** MUST — 別の g を渡すと
    「census が外した重みがここでは割り切れる」形の食い違いが数字だけに現れる。
    """
    files = sorted(path for path in dist_dir.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    unmatched = 0
    for path in files:
        if path.suffix != ".safetensors":
            continue
        header = read_safetensors_header(path)
        bucket = "shared（DeBERTa text_encoder）" if "shared" in path.parts else "話者ごとの net_g"
        for key, entry in header.items():
            if entry["dtype"] != "I8" or len(entry["shape"]) != 2:
                continue
            if key not in linear_fqns and key.removeprefix(DIST_BERT_PREFIX) not in linear_fqns:
                unmatched += 1
                continue
            out_channels, in_axis = entry["shape"]
            if in_axis % group_size:
                raise AssertionError(
                    f"{path}: '{key}' の in 軸 {in_axis} が group {group_size} で割り切れない"
                )
            scale = header.get(DIST_SCALE_PREFIX + key)
            if scale is None:
                raise AssertionError(f"{path}: '{key}' に対応する scale テンソルが無い")
            row = groups[bucket]
            row[0] += 1
            row[1] += tensor_bytes(entry) + tensor_bytes(scale)
            row[2] += out_channels * in_axis // 2 + 4 * out_channels * (in_axis // group_size)
    delta = sum(row[1] - row[2] for row in groups.values())
    return {
        "root": str(dist_dir),
        "group_size": group_size,
        "files": len(files),
        "total_bytes": total_bytes,
        "groups": {
            name: {
                "tensors": row[0],
                "current_bytes": row[1],
                "projected_bytes": row[2],
                "delta_bytes": row[1] - row[2],
                "shrink_of_total": (row[1] - row[2]) / total_bytes,
            }
            for name, row in sorted(groups.items())
        },
        "rank2_i8_not_linear": unmatched,
        "delta_bytes": delta,
        "shrink_of_total": delta / total_bytes if total_bytes else 0.0,
        "formula": f"i8 = O·I + 4·O バイト / i4 = O·I/2 + 4·O·(I/{group_size}) バイト"
        "（linear の重みスロットだけ・分母は配布形の実ファイル総バイト）",
    }


def tensor_bytes(entry: Mapping[str, Any]) -> int:
    start, end = entry["data_offsets"]
    return int(end) - int(start)


# ---- 1 構成の実行 ------------------------------------------------------------


@dataclass
class ChainInputs:
    """全構成で共有する入力一式（f32 の BERT 特徴も含めて 1 度だけ作る）。"""

    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any]
    bert_feature: torch.Tensor
    #: BERT の w4 対象集合（linear 限定）— サイズ試算と丸めの `include` の正本。
    bert_census: W4Census
    g: torch.Tensor
    style_vec: torch.Tensor


def load_bert() -> Any:
    """参照用の DeBERTa（**切り詰めていない全 24 層**）を読む。"""
    from transformers import DebertaV2Model

    bert = DebertaV2Model.from_pretrained(
        demo.BERT_REPO, dtype=torch.float32, attn_implementation="eager"
    )
    bert.eval()
    return bert


def bert_feature_of(bert: nn.Module, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """BERT を走らせて音素へ tile 展開した特徴を返す（丸めの有無に依らない共通経路）。"""
    with torch.no_grad():
        hidden_states = bert(
            input_ids=tensors["input_ids"].to(torch.int64),
            attention_mask=tensors["attention_mask"].to(torch.int64),
            output_hidden_states=True,
        ).hidden_states
    # MUST: 参照は**切り詰めていない全 24 層**モデルなので位置は定数を直に使う
    # （`meta["bertHiddenFromEnd"]` は配布グラフ側の位置 — sbv2.demo の 2 定数の使い分け）。
    hidden = hidden_states[-demo.BERT_HIDDEN_FROM_END][0]
    word2ph = tensors["word2ph"].reshape(-1).tolist()
    return demo.tile_bert(hidden, word2ph).unsqueeze(0)


def prepare_inputs(dump_path: Path, assets_path: Path, group_size: int) -> ChainInputs:
    """dump / assets を読み、**f32 の BERT** を 1 回だけ走らせて特徴を作る。

    f32 の特徴を構成間で共有するのは契約（既存 5 構成 + net_g の w4 は BERT=f32 固定）
    そのもので、何度走らせても同じ数が出る計算を 1 回に畳んでいるだけ。BERT を振る構成は
    {@link quantized_bert_feature} が**素の重みから**別に作る。トークナイズの突合は
    `sbv2.demo.run_reference` と同じものを踏む（Deno 側の移植と食い違えば波形の手前で落ちる）。
    """
    meta = demo.dump_metadata(dump_path)
    tensors = load_file(str(dump_path))
    assets = load_file(str(assets_path))

    tokenizer = demo.load_bert_tokenizer()
    expected_ids = tokenizer(meta["bertText"])["input_ids"]
    dumped_ids = tensors["input_ids"].reshape(-1).tolist()
    if expected_ids != dumped_ids:
        raise AssertionError(
            f"DeBERTa トークナイズが dump と食い違う（python={expected_ids} / dump={dumped_ids}）"
        )

    bert = load_bert()
    # 対象集合は f32 の模型から採る（丸めても FQN と形は動かないので、読み直しは要らない）。
    bert_census = census_w4_targets(bert, W4_ROLES["linear"], group_size)
    bert_feature = bert_feature_of(bert, tensors)
    del bert
    gc.collect()

    return ChainInputs(
        tensors=tensors,
        meta=meta,
        bert_feature=bert_feature,
        bert_census=bert_census,
        g=assets["g"].to(torch.float32),
        style_vec=assets["style_vec"].to(torch.float32),
    )


def quantized_bert_feature(
    kind: str, inputs: ChainInputs, methods: Mapping[str, W4Method]
) -> tuple[torch.Tensor, str]:
    """BERT の **linear の重みだけ**を w4 方式で丸めてから特徴を採る（`(特徴, 計数)`）。

    MUST: 模型は**素の重みから読み直す**（方式を積み重ねない）— net_g 側が構成ごとに
    `load_net_g` を呼び直しているのと同じ pristine の採り方で、退避／復元の可変状態を持たない。
    対象を linear 限定にするのは i4 の実行経路がそこだけだから（ADR 0069 決定 5）。BERT の
    embedding（`word_embeddings` / `rel_embeddings`）と 1 本の conv は f32 のまま残る。
    """
    bert = load_bert()
    report = methods[kind].apply(bert, W4_ROLES["linear"], inputs.bert_census.include())
    feature = bert_feature_of(bert, inputs.tensors)
    del bert
    gc.collect()
    return feature, report.describe()


# ---- 校正付き丸めの駆動（Catcher + stage 分解）-------------------------------

#: 校正 stage のモデル内 FQN 接頭辞（`DebertaV2Model.encoder` = `DebertaV2Encoder` →
#: `.layer`）。stage 内の局所 FQN が `<層番号>.attention.self.query_proj.weight` になるように
#: {@link EncoderStage} が子の名前を層番号にしてあるので、接頭辞はここまでで足りる
#: （scale 台帳のキーを {@link karume.quantize.Int4Report} と同じ**モデル内 FQN** の空間へ
#: 揃えるため — core の `StageSpec` 契約）。
BERT_STAGE_PREFIX = "encoder.layer"

#: {@link EncoderStage.forward} が ConvLayer 用に受け取る keyword 名。`DebertaV2Encoder` は
#: 層へ渡す 4 次元 mask とは別に、**素の 2 次元 mask** を ConvLayer へ渡す。
INPUT_MASK_KWARG = "input_mask"


class _FirstStageReached(Exception):  # noqa: N818 — 異常ではなく打ち切りの合図なので Error と呼ばない
    """先頭 stage の入力が揃った合図（校正 forward を打ち切るための番兵）。"""


class EncoderStage(nn.Module):
    """encoder layer 1 枚を「hidden を位置引数で受ける」形へ包む stage ラッパ。

    包む理由は 2 つあり、どちらも `DebertaV2Encoder.forward`（transformers 5.14.1）の
    呼び出しの形に由来する:

    1. 層へ mask を**位置引数**で渡す（`layer_module(next_kv, attention_mask, …)`）。
       `calibrate_stages` は次 stage へ「選んだ出力を**唯一の位置引数**」として渡す駆動なので、
       mask は keyword で運ぶ形に直さないと 2 段目以降で落ちる。
    2. **先頭層だけ**は出力に ConvLayer が乗る（ループ内の `i == 0` — stage の入力 hidden を
       残差として混ぜ直す）。stage 分解でここを落とすと 2 段目以降が本物と違う hidden を
       見ることになり、校正が別の活性から丸め先を選ぶ。

    層 kwargs（`relative_pos` / `rel_embeddings` / `query_states` / `output_attentions`）は
    ループの**前**に 1 度だけ作られ**全層で同一**（実測で確認 — Gemma3 のような layer_type 別
    の切り替えは無い）なので、`calibrate_stages` の既定（stage 間で kwargs 不変）にそのまま
    乗る。

    子モジュールの名前を**層番号**にしてあるのは、stage 内の局所 FQN
    （`0.attention.self.query_proj.weight`）へ {@link BERT_STAGE_PREFIX} を足すだけでモデル内
    FQN へ戻すため。ConvLayer も子として登録されるが `nn.Linear` を 1 本も持たない
    （`Conv1d` + `LayerNorm`）ので、走査（`iter_quant_targets` の `nn.Linear` 限定）の対象集合は
    動かない。
    """

    def __init__(self, index: int, layer: nn.Module, conv: nn.Module | None) -> None:
        super().__init__()
        self.child = str(index)
        self.add_module(self.child, layer)
        self.conv = conv

    def forward(
        self, hidden: torch.Tensor, *, input_mask: torch.Tensor, **layer_kwargs: Any
    ) -> torch.Tensor:
        output = getattr(self, self.child)(hidden, **layer_kwargs)[0]
        if self.conv is None:
            return output
        return self.conv(hidden, output, input_mask)


def bert_stages(bert: nn.Module) -> tuple[StageSpec, ...]:
    """実行順の encoder layer を `(モデル内 FQN 接頭辞, stage)` で返す。

    **特徴を採る層までしか作らない** — 参照模型は切り詰めていない全 24 層だが、特徴は
    `hidden_states[-demo.BERT_HIDDEN_FROM_END]`（= 22 層目の出力）から採るので、それ以降の層は
    丸めても出力に 1bit も効かない（配布グラフが 22 層 variant なのと同じ境界 — ADR 0045
    決定 1/2 の「参照側と配布グラフ側で位置の定数を分ける」で、ここが参照側の位置）。したがって
    校正の対象集合は既存 `bert-w4-*` の census（全 24 層）の**部分集合**であり、**配布形の
    linear 集合と一致する**（{@link CALIB_TARGET}）。
    """
    encoder = bert.encoder
    if encoder.conv is None:
        raise AssertionError(
            "DeBERTa encoder に ConvLayer が無い（先頭層の残差混合が消える構成 —"
            " 模型の構成が台本の想定と違う）"
        )
    layers = list(encoder.layer)
    count = len(layers) - (demo.BERT_HIDDEN_FROM_END - 1)
    if count < 1:
        raise AssertionError(
            f"encoder が {len(layers)} 層しか無い"
            f"（特徴は末尾から {demo.BERT_HIDDEN_FROM_END} 番目の hidden から採る）"
        )
    return tuple(
        (BERT_STAGE_PREFIX, EncoderStage(index, layer, encoder.conv if index == 0 else None))
        for index, layer in enumerate(layers[:count])
    )


def calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """stage 内の量子化対象を fqn 引きの重みと計数で返す（校正の**走査** = 門の基準）。

    対象選択は core の `iter_quant_targets` の共有（{@link census_w4_targets} と同文 — 写した
    別実装にすると「数えた対象」と「丸めた対象」が黙って割れる）。ここで数えた本数と、校正が
    実際に丸めた本数を {@link assert_calib_covers_scan} が突き合わせる。

    net_g 側の census と違って割り切れない重みを**外さず fail loudly** にする — DeBERTa の
    linear は量子化軸が 1024 / 4096 の 2 種しか無く、除外が要るのは受容野が半端な conv を持つ
    net_g 側の話だから。外す道をここへ作ると「黙って対象が痩せた」が校正側にも生える。
    """
    weights: dict[str, torch.Tensor] = {}
    channels = 0
    elements = 0
    for prefix, stage in stages:
        for local, weight, axis in iter_quant_targets(stage, W4_ROLES["linear"]):
            fqn = f"{prefix}.{local}"
            span = int(channel_rows(weight, axis).shape[-1])
            if span % CALIB_GROUP_SIZE:
                raise AssertionError(
                    f"'{fqn}': 量子化軸 {span} が group {CALIB_GROUP_SIZE} で割り切れない"
                    "（i4 は端数 group を作らない — ADR 0069 決定 2）"
                )
            weights[fqn] = weight
            channels += int(weight.shape[axis])
            elements += int(weight.numel())
    if not weights:
        raise AssertionError("encoder stage に nn.Linear が 1 本も無い（模型の構成が想定と違う）")
    return weights, TargetCounts(
        modules=len(weights), channels=channels, elements=elements, group_size=CALIB_GROUP_SIZE
    )


def calib_corpus(limit: int | None) -> tuple[str, ...]:
    """校正に使う文（`limit` は**先頭 N 文**の上限 — `None` は全 48 文）。

    先頭から採るのは {@link deberta.calib_texts.CALIB_TEXTS} が朗読調 / 問いかけ / 数字読みを
    混ぜて並べてあるため（縮小実行でも役割の混合が保たれる）。
    """
    if limit is not None and limit < 1:
        raise ValueError(f"校正文数の上限は 1 以上（実測 {limit}）")
    return CALIB_TEXTS if limit is None else CALIB_TEXTS[:limit]


def assert_calib_disjoint(texts: Sequence[str], evaluated: Sequence[str]) -> None:
    """校正コーパスが評価文と**部分一致でも**重ならないことを見る。

    MUST: fail loudly。重なると SNR / LSD が「校正で見た文をそのまま出せたか」を測る数になり、
    校正の質ではなく漏れを測ることになる。評価文の正本は dump の `meta` なので、コーパス側に
    写しを持たずここで突き合わせる（写すと dump を録り直したときに片方だけ古くなる）。
    """
    hits = sorted(
        {text for text in texts for body in evaluated if body and (body in text or text in body)}
    )
    if hits:
        raise AssertionError(
            f"校正コーパスが評価文と重なっている: {hits[:3]}（評価文 {list(evaluated)}）"
        )


def calib_inputs(texts: Sequence[str]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """校正コーパスを**既存の tokenizer 経路**（{@link sbv2.demo.load_bert_tokenizer}）で
    `(input_ids, attention_mask)` の `[1,T]` へ落とす。

    トークナイザを写さないのは `prepare_inputs` の dump 突合と同文 — 別経路で引くと「校正で
    見た活性」と「評価で流れる活性」が黙って別のトークン分割になる。
    """
    tokenizer = demo.load_bert_tokenizer()
    encoded = [tokenizer(text) for text in texts]
    return tuple(
        (
            torch.tensor([entry["input_ids"]], dtype=torch.int64),
            torch.tensor([entry["attention_mask"]], dtype=torch.int64),
        )
        for entry in encoded
    )


def capture_stage_batches(
    bert: nn.Module, inputs: Sequence[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[StageBatch, ...]:
    """先頭 stage への hidden と付随引数を forward_pre_hook で捕まえる（Catcher）。

    校正入力は「embeddings を通った後の hidden と、encoder が組んだ 4 次元 mask / 相対位置 /
    `rel_embeddings`」で、自前で組み直すと transformers 側と黙って割れる。**先頭層の呼び出し
    そのもの**を捕まえるのが、前段を写さずに同じ入力を採る形。

    ConvLayer が要る 2 次元 mask は層へは渡らないので、**ConvLayer の呼び出し**からもう 1 本
    捕まえる（`attention_mask.dim() <= 2` の分岐を写すと片方だけ仕様から外れる、で同文）。
    番兵で打ち切るのは ConvLayer 側 — そこまでで走るのは先頭層 1 枚だけ。

    MUST: 揃わずに forward が完走したら fail loudly — stage の綴りが模型の構成と食い違って
    いる合図で、黙って進むと「校正入力ゼロ」の診断が core 側で出るだけになる。
    """
    encoder = bert.encoder
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    masks: list[torch.Tensor] = []

    def catch_layer(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((args, dict(kwargs)))

    def catch_conv(_module: nn.Module, args: tuple[Any, ...]) -> None:
        masks.append(args[2])
        raise _FirstStageReached

    handles = [
        encoder.layer[0].register_forward_pre_hook(catch_layer, with_kwargs=True),
        encoder.conv.register_forward_pre_hook(catch_conv),
    ]
    batches: list[StageBatch] = []
    try:
        for index, (ids, mask) in enumerate(inputs):
            captured.clear()
            masks.clear()
            try:
                with torch.no_grad():
                    bert(input_ids=ids, attention_mask=mask)
            except _FirstStageReached:
                pass
            if len(captured) != 1 or len(masks) != 1:
                raise AssertionError(
                    f"校正入力 {index} で先頭 stage の入力が揃わなかった"
                    f"（層 {len(captured)} 件 / ConvLayer {len(masks)} 件）"
                    "— stage の綴りが模型の構成と食い違っている"
                )
            args, kwargs = captured[0]
            if len(args) != 2:
                raise AssertionError(
                    f"先頭層が位置引数 {len(args)} 個で呼ばれた（hidden と mask の 2 個が想定 —"
                    " transformers 側の呼び出しの形が変わっている）"
                )
            hidden, attention_mask = args
            batches.append(
                (
                    (hidden.detach(),),
                    {"attention_mask": attention_mask, INPUT_MASK_KWARG: masks[0], **kwargs},
                )
            )
    finally:
        for handle in handles:
            handle.remove()
    return tuple(batches)


@dataclass(frozen=True)
class CalibRig:
    """校正付き構成 1 本ぶんの足場（stage 列・走査・先頭 stage への入力）。

    minicpm5 / EG のリグと違って**構成ごとに組み直す** — この台本は方式を積み重ねないために
    BERT を構成ごとに素の重みから読み直す（{@link quantized_bert_feature} と同じ pristine の
    採り方）ので、stage も先頭 stage の入力もその模型に紐づく。
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    counts: TargetCounts
    batches: tuple[StageBatch, ...]


def build_calib_rig(bert: nn.Module, meta: Mapping[str, Any], limit: int | None) -> CalibRig:
    """校正の足場を組む（stage 分解 → 走査 → 評価文との分離検査 → tokenize → Catcher）。"""
    stages = bert_stages(bert)
    scan, counts = calib_targets(stages)
    texts = calib_corpus(limit)
    assert_calib_disjoint(texts, (meta["text"], meta["bertText"]))
    return CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=capture_stage_batches(bert, calib_inputs(texts)),
    )


def apply_calib(config: CalibConfig, rig: CalibRig) -> CalibReport:
    """校正付き丸め 1 本を模型へ in-place で当てる（stage 逐次の駆動は core 側）。"""
    return calibrate_stages(rig.stages, rig.batches, method=config.method, spec=config.grid)


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """校正が丸めた層が stage の走査と**過不足なく**一致することを見る。

    MUST: fail loudly。stage の綴りや対象型が変わって encoder の一部が校正に載らなくなっても
    表には行が残り、しかも丸め漏れのぶん SNR / LSD は**良い側**に出る（素通りを数字から
    読めない）。既存 6 門（{@link run_gates}）と同じ立場の恒真化の遮断。
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise AssertionError(
            f"[{name}] 校正が丸めた {report.modules} 本が走査の {len(scan)} 本と一致しない"
            f"（丸め漏れ {missing[:3]} / 走査に無い {extra[:3]}）"
        )


def calibrated_bert_feature(
    name: str, inputs: ChainInputs, limit: int | None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """BERT の **encoder 内 linear** を校正付き丸めで丸めてから特徴を採る（`(特徴, 報告)`）。

    MUST: 模型は**素の重みから読み直す**（方式を積み重ねない — {@link quantized_bert_feature}
    と同文）。丸めそのものは core の `calibrate_stages` の共有で、台本が持つのは stage 分解と
    校正入力の採り方だけ。
    """
    config = CALIB_CONFIGS[name]
    bert = load_bert()
    rig = build_calib_rig(bert, inputs.meta, limit)
    report = apply_calib(config, rig)
    assert_calib_covers_scan(report, rig.scan, name)
    feature = bert_feature_of(bert, inputs.tensors)
    projection = SizeProjection(
        counts=rig.counts, bits=config.projected_bits(rig.counts), formula=config.formula
    )
    rows = [layer.tokens for layer in report.layers]
    detail = {
        "quant": report.describe(),
        "method": config.label,
        "target": CALIB_TARGET,
        "stages": report.stages,
        "calib_texts": len(rig.batches),
        # `--calib-limit` の縮小実行だったかの記録（`None` = 全量）— 縮小した数値を全量の
        # 数値として読まれないように、指標と同じファイルへ残す。
        "calib_limit": limit,
        # 層ごとに見た入力**行**数の幅（`H = Σ XᵀX` の重み）。`query_proj` / `key_proj` は
        # share_att_key の相対位置埋め込み（1 forward あたり 512 行）も食うので、素のトークン
        # 行数（= 下限側）より必ず多い側へ出る — 1 つの数で書くと読み手が取り違える。
        "calib_rows": {"min": min(rows), "max": max(rows)},
        "size": {
            "modules": projection.counts.modules,
            "elements": projection.counts.elements,
            "bits_per_weight": projection.bits_per_weight,
            "projected_mib": projection.projected_mib,
            "f32_mib": projection.f32_mib,
            "formula": projection.formula,
        },
    }
    del bert, rig
    gc.collect()
    return feature, detail


def bert_variant(recipe: Recipe, methods: Mapping[str, W4Method]) -> str | None:
    """BERT 特徴のキャッシュ鍵 = **表に出る方式名**（`None` = f32 固定）。

    素の方式（{@link w4_methods}）と校正付き（{@link CALIB_CONFIGS}）で名前空間は交わらない
    ので 1 本の鍵で足りる。両方を持つ Recipe は構成の書き間違いなので落とす。

    鍵に種ではなく**名前**を採るのは、この鍵がそのまま `w4.bert_quant` と
    `gates.bert_quant_effective` の欄名として出るから — g を振った 2 回の実行の欄名が同じだと
    読み手が取り違える。
    """
    if recipe.bert_method is not None and recipe.bert_calib is not None:
        raise AssertionError("BERT の丸めは素の方式か校正付きのどちらか一方")
    if recipe.bert_method is not None:
        return methods[recipe.bert_method].name
    return recipe.bert_calib


def bert_feature_for(
    recipe: Recipe, inputs: ChainInputs, calib_limit: int | None, methods: Mapping[str, W4Method]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """構成の BERT 契約に合った特徴と、その丸めの報告を作る（f32 固定の構成では呼ばない）。"""
    if recipe.bert_calib is not None:
        return calibrated_bert_feature(recipe.bert_calib, inputs, calib_limit)
    if recipe.bert_method is None:
        raise AssertionError("BERT の丸めを持たない構成で特徴を作ろうとしている")
    feature, described = quantized_bert_feature(recipe.bert_method, inputs, methods)
    return feature, {
        "quant": described,
        "method": methods[recipe.bert_method].name,
        "target": "bert:linear",
    }


def run_config(
    name: str,
    model_dir: Path,
    inputs: ChainInputs,
    *,
    bert_feature: torch.Tensor,
    census: Mapping[tuple[str, str], W4Census],
    methods: Mapping[str, W4Method],
    group_size: int,
    inject: str | None,
) -> dict[str, Any]:
    """1 構成ぶんのチェーンを走らせ、波形と診断を返す。

    net_g は構成ごとに読み直す（= 丸めを積み重ねない pristine の採り方）。`bert_feature` は
    構成の BERT 契約に合った特徴（f32 固定 or w4 で丸めたもの）を呼び出し側が渡す。

    MUST（順序 — `quantize` モジュールと ADR 0018/0019/0027）: 重みの丸めは
    **`apply_all_patches` と `ensure_dec_plain` の後**に当てる。remove_weight_norm より先に
    丸めると `weight_g`/`weight_v` を丸めることになり、そこから作られる実効重みは f16/i8 の
    格子に乗らない（i8 は捨てられる要素が amax に効いて per-channel scale ごとずれる。i4 は
    group ごとに同じことが起きる）。
    """
    started = time.perf_counter()
    recipe = RECIPES[name]
    tensors = inputs.tensors
    knobs = inputs.meta["knobs"]

    net_g, hps = export.load_net_g(model_dir)
    patch.apply_all_patches()
    export.ensure_dec_plain(net_g)
    front = patch.Sbv2Front(net_g)
    voice = patch.Sbv2Voice(net_g)
    # front（enc_p / dp / sdp）と voice（flow / dec）は互いに素なモジュール集合。
    subgraphs = (("front", front), ("voice", voice))
    scoped = [(tag, module) for tag, module in subgraphs if tag in recipe.scope]

    # --- 重みの丸め -----------------------------------------------------------
    weight_report: dict[str, str] = {}
    w4_targets: dict[str, dict[str, int]] = {}
    for tag, module in scoped:
        if recipe.weight == "f16":
            weight_report[tag] = round_weights_to_f16(module).describe()
        elif recipe.weight == "i8":
            weight_report[tag] = fake_quant_int8(module).describe()
        elif recipe.weight == "w4":
            if recipe.method is None:
                raise AssertionError(f"{name}: w4 構成なのに方式が指定されていない")
            scoped_census = census[(recipe.roles, tag)]
            report = methods[recipe.method].apply(
                module, W4_ROLES[recipe.roles], scoped_census.include()
            )
            weight_report[tag] = report.describe()
            w4_targets[tag] = {"modules": report.modules, "elements": report.elements}

    # --- 活性の丸め（対象 op の入力を差し替え）--------------------------------
    quant: ActQuant | None = None
    attached = 0
    if recipe.act is not None:
        quant = ActQuant(roots=tuple(scoped), mode=recipe.act, disable=(inject == "drop-act-quant"))
        attached = quant.attach()

    # --- front ---------------------------------------------------------------
    x_mask = tensors["x_mask"].to(torch.float32)
    with torch.no_grad():
        logw_sdp, logw_dp, m_p, logs_p = front(
            tensors["x"].to(torch.int64),
            x_mask,
            tensors["tone"].to(torch.int64),
            tensors["language"].to(torch.int64),
            bert_feature,
            inputs.style_vec,
            inputs.g,
            tensors["z_noise"].to(torch.float32),
        )

    # --- ホストグルー（デモ main.ts / demo.run_reference と同式）-----------
    logw = logw_sdp * knobs["sdpRatio"] + logw_dp * (1.0 - knobs["sdpRatio"])
    w = torch.exp(logw) * x_mask * knobs["lengthScale"]
    own_ceil = torch.ceil(w).to(torch.int64).reshape(-1)
    dumped_ceil = tensors["w_ceil"].reshape(-1).to(torch.int64)
    # MUST: 展開は **dump 側の w_ceil**。構成ごとに自前の w_ceil を使うと波形長が割れて
    # SNR が定義できなくなる（zp_noise の形も dump 側の Ty で固定されている）。
    expand_idx = torch.repeat_interleave(torch.arange(dumped_ceil.shape[0]), dumped_ceil)
    total_frames = int(expand_idx.shape[0])
    zp_noise = tensors["zp_noise"].to(torch.float32)
    z_p = (
        m_p[:, :, expand_idx] + zp_noise * torch.exp(logs_p[:, :, expand_idx]) * knobs["noiseScale"]
    )
    y_mask = torch.ones(1, 1, total_frames)
    idx_k, valid = patch.build_relattn_tables(total_frames, export.EXPECTED_WINDOW_SIZE)

    # --- voice（flow + dec 融合）--------------------------------------------
    with torch.no_grad():
        audio = voice(z_p, y_mask, inputs.g, idx_k, valid).reshape(-1)

    stats: dict[str, ActStat] = {}
    skipped = 0
    if quant is not None:
        quant.detach()
        stats = quant.stats
        skipped = quant.skipped_calls
    # 差し替えとフックの復元は構成間のリークに直結する（op の差し替えが残ると次の構成が
    # 黙って量子化される）。`is` で**素の実体**まで戻ったことを検査する — 構成を跨ぐ
    # 唯一の可変状態なので、丸めなしの構成（f32 / f16 / w8）でも毎回見る。
    leaked = [op for op in ACT_ROW_AXIS if getattr(torch.nn.functional, op) is not PRISTINE_OPS[op]]
    if leaked:
        raise AssertionError(f"{name}: 活性量子化の op 差し替えが残っている: {leaked}")

    diagnostics = {
        "config": name,
        "weight_mode": recipe.weight,
        "act_mode": recipe.act,
        "scope": list(recipe.scope),
        "w4": {
            "method": methods[recipe.method].name if recipe.method is not None else None,
            "roles": recipe.roles if recipe.weight == "w4" else None,
            "group_size": group_size if recipe.weight == "w4" else None,
            "targets": w4_targets,
        },
        "bert_method": (
            methods[recipe.bert_method].name if recipe.bert_method is not None else None
        ),
        "bert_calib": recipe.bert_calib,
        "version": hps.version,
        "frames": total_frames,
        "samples": int(audio.shape[0]),
        "seconds": round(audio.shape[0] / inputs.meta["samplingRate"], 3),
        "weight_quant": weight_report,
        "act_quant": {
            "patched_ops": attached,
            "call_sites": len(stats),
            "calls": sum(stat.calls for stat in stats.values()),
            "calls_by_op": {
                op: sum(stat.calls for stat in stats.values() if stat.op == op)
                for op in ACT_ROW_AXIS
            },
            "skipped_calls": skipped,
            "f16_saturated": sum(stat.saturated for stat in stats.values()),
        },
        "w_ceil": {
            "exact_vs_dump": bool(torch.equal(own_ceil, dumped_ceil)),
            "own_frames": int(own_ceil.sum()),
            "dump_frames": int(dumped_ceil.sum()),
            "max_abs_diff": int((own_ceil - dumped_ceil).abs().max()),
        },
        "elapsed": round(time.perf_counter() - started, 1),
    }
    front_outputs = {
        "logw": logw.detach().clone(),
        "m_p": m_p.detach().clone(),
        "logs_p": logs_p.detach().clone(),
    }

    del net_g, front, voice, z_p, idx_k, valid
    gc.collect()
    return {
        "audio": audio.detach().clone(),
        "front": front_outputs,
        "stats": stats,
        "diagnostics": diagnostics,
    }


# ---- レポート ---------------------------------------------------------------


def act_group_table(stats: dict[str, ActStat], limit: int) -> list[dict[str, Any]]:
    """層グループ別の活性量子化誤差（relRMS 降順の上位のみ）。"""
    groups: dict[str, list[ActStat]] = defaultdict(list)
    for name, stat in stats.items():
        groups[layer_group(name)].append(stat)
    rows = [
        {
            "group": name,
            "op": members[0].op,
            "call_sites": len(members),
            "rel_rms_max": max(stat.rel_rms() for stat in members),
            "rel_rms_mean": sum(stat.rel_rms() for stat in members) / len(members),
        }
        for name, members in groups.items()
    ]
    rows.sort(key=lambda row: -row["rel_rms_max"])
    return rows[:limit]


def build_report(
    args: argparse.Namespace,
    inputs: ChainInputs,
    results: dict[str, dict[str, Any]],
    wavs: dict[str, Path],
    gates: dict[str, Any],
    w4: dict[str, Any],
) -> dict[str, Any]:
    base = results["f32"]["audio"]

    def entry_for(name: str) -> dict[str, Any]:
        result = results[name]
        audio = result["audio"]
        entry = dict(result["diagnostics"])
        entry["wav"] = str(wavs[name])
        entry["wav_sha256"] = sha256(wavs[name].read_bytes())
        entry["maxAbs"] = float(audio.abs().max())
        entry["snr_db_vs_f32"] = snr_db(audio, base)
        entry["lsd_db_vs_f32"] = log_spectral_distance(audio, base)
        entry["rel_rms_vs_f32"] = rel_rms(audio, base)
        entry["maxAbs_diff_vs_f32"] = float((audio - base).abs().max())
        entry["bit_equal_to_f32"] = bool(torch.equal(audio, base))
        entry["front_snr_db_vs_f32"] = {
            key: snr_db(result["front"][key], results["f32"]["front"][key])
            for key in ("logw", "m_p", "logs_p")
        }
        if result["stats"]:
            entry["act_groups"] = act_group_table(result["stats"], args.top)
        return entry

    configs = [entry_for(name) for name in CONFIGS if name in results]
    diagnostics = [entry_for(name) for name in DIAGNOSTICS if name in results]

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/sbv2/measure_quant.py",
        "torch": torch.__version__,
        "dump": str(args.dump),
        "text": inputs.meta["text"],
        "knobs": inputs.meta["knobs"],
        "sampling_rate": inputs.meta["samplingRate"],
        "method": {
            "bert": "f32 固定（全構成共有 — 生成ネット側だけを振る）",
            "act_target_ops": sorted(ACT_ROW_AXIS),
            "act_apply_point": "対象 op の入力（torch.nn.functional の op 自体を差し替え —"
            " モジュールの forward_pre_hook では FFN パッチの直接呼び出しを取り逃す）",
            "act_row_axis": dict(ACT_ROW_AXIS),
            "act_i8_formula": "s = clamp(rowmax|x|/127, f32 tiny) / q = clamp(round(x/s), ±127)"
            " / x̂ = q·s（act_quant.quantize_rows を軸転置で再利用）",
            "time_grid": "全構成とも dump 側 w_ceil で展開（波形長を揃えて SNR を定義可能にする）",
            "metrics": "SNR は波形の残差（位相ずれに弱い）/ LSD は対数振幅スペクトルの距離"
            f"（n_fft={STFT_N_FFT} hop={STFT_HOP}・床 −100dB・位相ずれに鈍い）。両方で読む",
            "diagnostics": "劣化の front / voice 直交分解（先行実験の既録 voice SNR とは"
            " `w8-voice-only` が比較相手）",
            "w4": f"方式は g={args.w4_group_size}（`--w4-group-size`・既定 {DEFAULT_GROUP_SIZE} —"
            " ADR 0069 追記 5 の方式軸 / 波 J-3 の g 軸）。net_g は全役割・BERT は linear 限定で、"
            "丸めは core（karume.quantize / karume.quant_methods）の共有。サイズは実測ではなく"
            "**式による投影**で、格納形を持つのは RTN（`i4`）だけ",
            "bert_calib": "校正付き丸め（GPTQ）は core（karume.quant_calib）の共有で、**格子は"
            "方式グリッドと 1 バイトも変わらない**（変わるのは同じ格子の中でどの準位へ寄せるか）。"
            "校正入力は deberta/calib_texts.py の 48 文を既存の DeBERTa tokenizer 経路で採り、"
            "評価文とは部分一致まで分離する。stage は encoder の**特徴を採る層まで**で、"
            f"対象は `{CALIB_TARGET}`（`bert:linear` の census の部分集合）",
        },
        "gates": gates,
        "w4": w4,
        "configs": configs,
        "diagnostics": diagnostics,
    }


def format_table(entries: list[dict[str, Any]]) -> str:
    header = [
        "config",
        "SNR vs f32 (dB)",
        "LSD (dB)",
        "relRMS",
        "maxAbs",
        "max|Δ|",
        "自前 w_ceil",
        "秒",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for entry in entries:
        snr = entry["snr_db_vs_f32"]
        ceil = entry["w_ceil"]
        frames = f"{ceil['own_frames']} / {ceil['dump_frames']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    entry["config"],
                    "inf（基準）" if math.isinf(snr) else f"{snr:.2f}",
                    f"{entry['lsd_db_vs_f32']:.2f}",
                    f"{entry['rel_rms_vs_f32']:.4e}",
                    f"{entry['maxAbs']:.6f}",
                    f"{entry['maxAbs_diff_vs_f32']:.4e}",
                    ("一致" if ceil["exact_vs_dump"] else "割れ") + f"（{frames} frames）",
                    f"{entry['elapsed']:.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_projection_table(rows: list[dict[str, Any]]) -> str:
    header = ["対象集合", "方式", "本数", "要素数", "bpw", "投影 MiB", "f32 MiB", "品質測定", "式"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["target"],
                    row["method"],
                    str(row["modules"]),
                    f"{row['elements']:,}",
                    f"{row['bits_per_weight']:.3f}",
                    f"{row['projected_mib']:.1f}",
                    f"{row['f32_mib']:.1f}",
                    row["measured_by"] or "—",
                    row["formula"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_distribution(projection: dict[str, Any]) -> str:
    mib = 1024**2
    lines = [
        f"配布形 {projection['root']}（実ファイル {projection['files']} 本 /"
        f" {projection['total_bytes'] / mib:.1f} MiB）に対する"
        f" i8 → i4 g{projection['group_size']} の縮小試算:",
        f"  式: {projection['formula']}",
    ]
    for name, group in projection["groups"].items():
        lines.append(
            f"  {name}: {group['tensors']} テンソル"
            f" {group['current_bytes'] / mib:.1f} MiB → {group['projected_bytes'] / mib:.1f} MiB"
            f"（−{group['delta_bytes'] / mib:.1f} MiB = 配布全体の"
            f" −{group['shrink_of_total'] * 100:.2f}%）"
        )
    lines.append(
        f"  合計: −{projection['delta_bytes'] / mib:.1f} MiB ="
        f" 配布全体の −{projection['shrink_of_total'] * 100:.2f}%"
        f"（linear 以外の rank 2 I8 は {projection['rank2_i8_not_linear']} 本を対象外）"
    )
    return "\n".join(lines)


# ---- 検証ゲート -------------------------------------------------------------


def run_gates(
    results: dict[str, dict[str, Any]],
    wavs: dict[str, Path],
    reference_wav: Path,
    census: Mapping[tuple[str, str], W4Census],
    bert_reports: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """恒真化と構成生成バグの検出器（赤は最後にまとめて上げる）。

    ① f32 の WAV が既存 `reference.wav` とバイト一致（sim 経路が本物と同じ計算であること）
    ② 各構成が f32 と実際に違うこと（丸めが素通りしていない）
    ③ w8a16 / w8a8 が w8 と違うこと（**活性量子化の適用漏れの検出器** — 漏れると
       活性の丸めが消えて w8 と**ビット一致**する）。差し替え op 数と呼び出し地点数の
       0 検査も併せて置く（②③ は値の側から、こちらは仕掛けの側から同じ穴を見る）
    ④ w4 の丸め本数・要素数が census と一致すること（`include` が無視されると対象が
       黙って増え、除外したはずの重みで core が fail loudly する前に数だけずれる）
    ⑤ w4 構成どうしが互いに違うこと（**方式の呼び分け漏れの検出器** — 同じ方式を 2 度
       当てているとビット一致する）
    ⑥ BERT の丸めが特徴を実際に動かすこと（丸めが素通りすると f32 と特徴がビット一致し、
       ②の波形側だけでは「net_g が f32 だから同じ」と見分けが付かない）

    `--configs` で構成を絞ったときは、走らなかった構成に関わる検査を**飛ばす**（絞った側で
    赤にすると smoke が常に赤になる）。飛ばした事実は各欄が空になることで読める。
    """
    gates: dict[str, Any] = {}
    reference_ok = None
    if reference_wav.is_file():
        reference_ok = sha256(wavs["f32"].read_bytes()) == sha256(reference_wav.read_bytes())
    gates["f32_wav_bit_exact_vs_reference"] = {
        "reference": str(reference_wav),
        "checked": reference_ok is not None,
        "passed": reference_ok,
    }

    base = results["f32"]["audio"]
    differs = {
        name: not bool(torch.equal(results[name]["audio"], base))
        for name in CONFIGS
        if name != "f32" and name in results
    }
    gates["differs_from_f32"] = differs

    act_names = [name for name in ("w8a16", "w8a8") if name in results]
    act_effective: dict[str, bool] = {}
    if "w8" in results:
        w8 = results["w8"]["audio"]
        act_effective = {
            name: not bool(torch.equal(results[name]["audio"], w8)) for name in act_names
        }
    gates["act_quant_effective_vs_w8"] = act_effective

    patched_ops = {
        name: results[name]["diagnostics"]["act_quant"]["patched_ops"] for name in act_names
    }
    call_sites = {
        name: results[name]["diagnostics"]["act_quant"]["call_sites"] for name in act_names
    }
    gates["act_quant_patched_ops"] = patched_ops
    gates["act_quant_call_sites"] = call_sites

    w4_names = [name for name in CONFIGS if name in results and CONFIGS[name].weight == "w4"]
    w4_targets = {
        name: {
            tag: (
                results[name]["diagnostics"]["w4"]["targets"][tag]
                == {
                    "modules": census[(CONFIGS[name].roles, tag)].counts.modules,
                    "elements": census[(CONFIGS[name].roles, tag)].counts.elements,
                }
            )
            for tag in CONFIGS[name].scope
        }
        for name in w4_names
    }
    gates["w4_targets_match_census"] = w4_targets

    w4_distinct = {
        f"{left} vs {right}": not bool(torch.equal(results[left]["audio"], results[right]["audio"]))
        for left, right in combinations(w4_names, 2)
    }
    gates["w4_methods_distinct"] = w4_distinct

    bert_effective = {method: report["differs_from_f32"] for method, report in bert_reports.items()}
    gates["bert_quant_effective"] = bert_effective

    failures: list[str] = []
    if reference_ok is False:
        failures.append("f32 の WAV が reference.wav とバイト一致しない（sim 経路の恒真化検査）")
    failures += [
        f"{name} が f32 とビット一致してしまった" for name, ok in differs.items() if not ok
    ]
    failures += [
        f"{name} が w8 とビット一致してしまった（活性量子化の適用漏れ）"
        for name, ok in act_effective.items()
        if not ok
    ]
    failures += [
        f"{name} の活性量子化の差し替え op が 0 本" for name, n in patched_ops.items() if n == 0
    ]
    failures += [
        f"{name} の活性量子化の呼び出し地点が 0 件" for name, n in call_sites.items() if n == 0
    ]
    failures += [
        f"{name}/{tag} の w4 対象が census と食い違う"
        for name, tags in w4_targets.items()
        for tag, ok in tags.items()
        if not ok
    ]
    failures += [
        f"{pair} がビット一致してしまった（w4 方式の呼び分け漏れ）"
        for pair, ok in w4_distinct.items()
        if not ok
    ]
    failures += [
        f"BERT の {method} が f32 と同じ特徴を出した（丸めの素通り）"
        for method, ok in bert_effective.items()
        if not ok
    ]
    gates["failures"] = failures
    return gates


def selected_configs(spec: str | None) -> tuple[str, ...]:
    """`--configs` の指定を主要構成の部分集合へ解く（`f32` は SNR の基準なので常に入る）。"""
    if spec is None:
        return tuple(CONFIGS)
    names = [name for name in spec.split(",") if name]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"未知の構成 {unknown}（選べるのは {list(CONFIGS)}）")
    return tuple(name for name in CONFIGS if name == "f32" or name in names)


def build_w4_section(
    census: Mapping[tuple[str, str], W4Census],
    inputs: ChainInputs,
    bert_reports: Mapping[str, dict[str, Any]],
    dist_dir: Path,
    calib_limit: int | None,
    methods: Mapping[str, W4Method],
    group_size: int,
) -> dict[str, Any]:
    """report.json の `w4` 節 — 対象集合の census・サイズ試算・配布形への投影。

    品質を測らない席（net_g の linear 限定）もここには出る — **利得が無いこと自体が採否の
    根拠**なので、対象本数とサイズ試算だけは常に見えている必要がある。
    """
    net_g_counts = {
        roles: census[(roles, "front")].counts + census[(roles, "voice")].counts
        for roles in W4_ROLES
    }
    counts = {
        "net_g:all": net_g_counts["all"],
        "net_g:linear": net_g_counts["linear"],
        "bert:linear": inputs.bert_census.counts,
    }
    # 配布形のテンソル名と突き合わせる「今日の i4 適格」= linear の重みスロットの FQN 集合。
    linear_fqns = frozenset(
        f"{name}.weight"
        for source in (census[("linear", "front")], census[("linear", "voice")], inputs.bert_census)
        for name in source.eligible
    )
    section: dict[str, Any] = {
        "group_size": group_size,
        "methods": {method.name: method.formula for method in methods.values()},
        "census": {
            "net_g": {
                f"{roles}/{tag}": _census_entry(entry) for (roles, tag), entry in census.items()
            },
            "bert": _census_entry(inputs.bert_census),
        },
        "projections": build_projections(counts, methods),
        "bert_quant": dict(bert_reports),
        "calib": {
            "group_size": CALIB_GROUP_SIZE,
            "target": CALIB_TARGET,
            "target_description": "BERT（DeBERTa）の encoder 内 linear のうち**特徴を採る層"
            "まで** — `bert:linear` の census の部分集合（末尾の層は出力に効かない）",
            "texts": len(calib_corpus(calib_limit)),
            "limit": calib_limit,
            "methods": {name: config.formula for name, config in CALIB_CONFIGS.items()},
        },
    }
    if dist_dir.is_dir():
        section["distribution"] = project_distribution(dist_dir, linear_fqns, group_size)
    else:
        section["distribution"] = {"root": str(dist_dir), "checked": False}
    return section


def _census_entry(census: W4Census) -> dict[str, Any]:
    return {
        "modules": census.counts.modules,
        "channels": census.counts.channels,
        "elements": census.counts.elements,
        "groups": census.counts.groups,
        "by_role": {
            role: {"modules": c.modules, "channels": c.channels, "elements": c.elements}
            for role, c in census.by_role.items()
        },
        "excluded": list(census.excluded),
        "excluded_modules": len(census.excluded),
        "excluded_elements": census.excluded_elements,
    }


def parse_w4_group_size(raw: str) -> int:
    """`--w4-group-size` の受理（**2 冪かつ {@link MIN_GROUP_SIZE} 以上**）。

    値域は core の格納規則そのもの（ADR 0069 決定 2 — 行境界・group 境界が常にバイト整列する
    条件）を `karume.ir` から借りる。ここで落とすのは、外れた g では**測っても出荷できない**
    から — 実測が終わってから emit / verify が撥ねるのでは 1 本ぶん丸損する。
    """
    value = int(raw)
    if value & (value - 1) or value < MIN_GROUP_SIZE:
        raise argparse.ArgumentTypeError(
            f"group 長 {value} が 2 冪かつ {MIN_GROUP_SIZE} 以上でない（ADR 0069 決定 2）"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=export.DEFAULT_MODEL_DIR)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--assets", type=Path, default=demo.DEFAULT_DEMO_DIR / demo.STYLE_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference-wav", type=Path, default=DEFAULT_REFERENCE_WAV)
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST_DIR)
    parser.add_argument("--top", type=int, default=12, help="層グループ表に載せる本数")
    parser.add_argument(
        "--configs",
        default=None,
        help="主要構成のうち走らせるものをコンマ区切りで絞る（既定は全部）。"
        " f32 は SNR の基準なので常に走る",
    )
    parser.add_argument(
        "--w4-group-size",
        type=parse_w4_group_size,
        default=DEFAULT_GROUP_SIZE,
        help=f"w4 の group 長（2 冪かつ {MIN_GROUP_SIZE} 以上・既定 {DEFAULT_GROUP_SIZE}）。"
        " 適格判定・丸め・方式名の 3 つへ同じ g が流れる。校正付き構成の格子は 32 のまま",
    )
    parser.add_argument(
        "--calib-limit",
        type=int,
        default=None,
        help=f"校正コーパスの先頭 N 文だけを使う（既定は全 {len(CALIB_TEXTS)} 文）。"
        " 縮小 smoke 用のノブで、校正付き構成にだけ効く",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="front / voice 直交分解の診断構成を走らせない（主要構成だけ）",
    )
    parser.add_argument(
        "--inject",
        choices=("drop-act-quant",),
        default=None,
        help="故障注入（検出器の検出力を実証する）— 活性量子化の op 差し替えをせずに"
        " w8a16 / w8a8 を作る（構成生成バグの再現）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = selected_configs(args.configs)
    # MUST: g はここで 1 度だけ読み、以降は引数で流す（module 直下に持つと「適格判定は g16 /
    # 丸めは g32」の黙った割れを作れる — {@link rtn_method_name} の MUST）。
    methods = w4_methods(args.w4_group_size)

    inputs = prepare_inputs(args.dump, args.assets, args.w4_group_size)
    print(
        f"[inputs] text={inputs.meta['text']!r} bert={list(inputs.bert_feature.shape)}"
        f" bert_linear={inputs.bert_census.counts.modules}",
        flush=True,
    )
    census = net_g_census(args.model_dir, args.w4_group_size)
    for roles in W4_ROLES:
        front, voice = census[(roles, "front")], census[(roles, "voice")]
        print(
            f"[census] net_g {roles}: 適格 {front.counts.modules + voice.counts.modules} 本"
            f" / {front.counts.elements + voice.counts.elements:,} 要素"
            f"（除外 {len(front.excluded) + len(voice.excluded)} 本"
            f" / {front.excluded_elements + voice.excluded_elements:,} 要素）",
            flush=True,
        )

    plan: list[tuple[str, Path]] = [(name, args.out) for name in configs]
    if not args.no_diagnostics:
        diag_dir = args.out / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        plan += [(name, diag_dir) for name in DIAGNOSTICS]

    # BERT を振る構成の特徴（丸め方ごとに 1 度だけ作る — 素の重みから読み直すので積み重ならない）。
    features: dict[str | None, torch.Tensor] = {None: inputs.bert_feature}
    bert_reports: dict[str, dict[str, Any]] = {}

    results: dict[str, dict[str, Any]] = {}
    wavs: dict[str, Path] = {}
    for name, directory in plan:
        variant = bert_variant(RECIPES[name], methods)
        if variant is not None and variant not in features:
            feature, detail = bert_feature_for(RECIPES[name], inputs, args.calib_limit, methods)
            features[variant] = feature
            bert_reports[variant] = detail | {
                "differs_from_f32": not bool(torch.equal(feature, inputs.bert_feature)),
                "rel_rms_vs_f32": rel_rms(feature, inputs.bert_feature),
            }
            print(f"[bert:{variant}] {detail['quant']}", flush=True)
        results[name] = run_config(
            name,
            args.model_dir,
            inputs,
            bert_feature=features[variant],
            census=census,
            methods=methods,
            group_size=args.w4_group_size,
            inject=args.inject,
        )
        audio = results[name]["audio"]
        path = directory / f"{name}.wav"
        path.write_bytes(demo.wav_pcm16(audio.numpy(), inputs.meta["samplingRate"]))
        wavs[name] = path
        diagnostics = results[name]["diagnostics"]
        print(
            f"[{name}] {diagnostics['elapsed']:.0f}s frames={diagnostics['frames']}"
            f" act={diagnostics['act_quant']['calls']} calls → {path}",
            flush=True,
        )

    gates = run_gates(results, wavs, args.reference_wav, census, bert_reports)
    w4 = build_w4_section(
        census,
        inputs,
        bert_reports,
        args.dist_dir,
        args.calib_limit,
        methods,
        args.w4_group_size,
    )
    report = build_report(args, inputs, results, wavs, gates, w4)
    # dump 側の波形（実 GPU の Karume 出力）との突合も残す — f32 経路の二重の裏取り。
    dump_audio = inputs.tensors["audio"].reshape(-1).to(torch.float32)
    if dump_audio.shape == results["f32"]["audio"].shape:
        report["f32_vs_dump_audio"] = {
            "maxAbs": float((results["f32"]["audio"] - dump_audio).abs().max()),
            "snr_db": snr_db(dump_audio, results["f32"]["audio"]),
        }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print(format_table(report["configs"]))
    if report["diagnostics"]:
        print()
        print("診断（front / voice 直交分解）:")
        print(format_table(report["diagnostics"]))
    calib_notes = [
        f"> {variant}: {entry['quant']} / 投影 {entry['size']['bits_per_weight']:.3f} bpw"
        f"（{entry['size']['formula']}）"
        for variant, entry in bert_reports.items()
        if variant in CALIB_CONFIGS
    ]
    if calib_notes:
        print()
        print(
            f"校正付き構成の内訳（校正 {len(calib_corpus(args.calib_limit))} 文 /"
            f" 対象 {CALIB_TARGET}）:"
        )
        print("\n".join(calib_notes))
        if args.calib_limit is not None:
            print(
                f"> 校正は先頭 {args.calib_limit} 文のみの縮小実行"
                f"（全 {len(CALIB_TEXTS)} 文ではない — 数値を全量のものとして読まないこと）"
            )
    print()
    print("w4 サイズ試算（**実測ではなく式による投影** — 格納形を持つのは RTN だけ）:")
    print(format_projection_table(report["w4"]["projections"]))
    distribution = report["w4"]["distribution"]
    print()
    if distribution.get("checked") is False:
        print(f"配布形 {distribution['root']} が無いので縮小試算は省略")
    else:
        print(format_distribution(distribution))
    print()
    print(f"report → {args.out / 'report.json'}")
    if gates["failures"]:
        raise AssertionError("検証ゲートが赤: " + " / ".join(gates["failures"]))
    print("gates: all green")


if __name__ == "__main__":
    main()
