"""SBV2 **生成ネット**（net_g）の量子化パターン別 音質劣化を torch CPU で実測する。

`measure_quant_anima.py`（perf-c Q0）と同じ思想の品質ゲート — 実行カーネルを作る前に、
「その量子化で音がどこまで劣化するか」だけを torch の fake-quant で先に答える。GPU コードは
0 行で、ここで測るのは**量子化そのものの質**（ADR 0006 の fake-quant 方法論では E2E は
実装誤差しか測らないため、品質は別軸で測る必要がある）。

11 構成を**同一発話・同一乱数**（`outputs/demo/sbv2-dump/dump.safetensors` の離散入力・ノイズ列）で
走らせる。既存 5 構成は**生成ネット側だけ**を振る（**BERT は f32 固定**）:

    (1) f32    基準。`sbv2.demo reference` と同じ経路（既存 reference.wav とビット一致）
    (2) f16    適格重みを f16 表現可能値へ丸め（`quantize.round_weights_to_f16`）。活性は f32
               = 実装済みの **f16 格納系列**（ADR 0027）の対応物
    (3) w8     重みを per-channel symmetric i8 へ fake-quant（`quantize.fake_quant_int8`）
               = ADR 0019 の w8a32（i8 格納・f32 計算）の対応物
    (4) w8a16  (3) + **対象 op の入力活性**を f16 へ丸め
    (5) w8a8   (3) + **対象 op の入力活性**を per-位置 symmetric i8 へ fake-quant

w4 の 6 構成（方式スクリーニングの勝者 4 種・**g=32 固定** — ADR 0069 追記 5）。丸めは全て
core（`karume.quantize` / `karume.quant_methods`）の共有で、台本ローカルの実装は持たない:

    (6) w4-rtn      net_g の**全役割**（conv1d / conv_transpose1d / linear / embedding）を
                    RTN group symmetric i4（`fake_quant_int4` の `op_types` opt-in）
    (7) w4-nf4      同じ全役割を NF4 の固定表 × g32 absmax scale
    (8) w4-mxfp4    同じ全役割を FP4 e2m1 表 × g32 の 2 のべき scale（OCP MX）
    (9) w4-kmeans   同じ全役割を k-means codebook（層をまたぐ共有表 × g32 正規化）
    (10) bert-w4-rtn  **BERT（DeBERTa）の linear だけ**を RTN i4 g32・net_g は f32 固定
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

    uv run --group sbv2 python -m sbv2.measure_quant

出力は `outputs/demo/quant-sim/`（`<config>.wav` 11 本 + `report.json`）。`--configs` で
主要構成を名前で絞れる（`f32` は SNR の基準なので常に走る）。

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch
from safetensors.torch import load_file
from torch import nn

from _shared.paths import DIST_ROOT, OUTPUTS_ROOT
from karume.act_quant import quantize_rows
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

#: デモ・ベンチの生成物置き場。資産（`sbv2.demo.DEFAULT_DEMO_DIR`）と分離する —
#: 生成物の掃除（`rm -rf outputs/demo`）が資産や系列を巻き込まないため（docs/assets-layout.md）。
DEMO_OUT_ROOT = OUTPUTS_ROOT / "demo"
DEFAULT_DUMP = DEMO_OUT_ROOT / "sbv2-dump" / "dump.safetensors"
#: f32 構成の恒真化を防ぐ突合先（`sbv2.demo reference` が同じ dump から書いた WAV）。
DEFAULT_REFERENCE_WAV = DEMO_OUT_ROOT / "sbv2-dump" / "reference.wav"
DEFAULT_OUT = DEMO_OUT_ROOT / "quant-sim"
#: サイズ縮小率の分母になる配布形（`karume dist` の生成物 — 実バイトを読むだけで書かない）。
DEFAULT_DIST_DIR = DIST_ROOT / "karume-sbv2-jvnv"


# ---- w4 の語彙（方式・役割）— ADR 0069 追記 5 --------------------------------

#: w4 の group 長。**32 固定**で g 軸は振らない — g の答えは ADR 0069 追記 1（Phase 0）が
#: 出しており、ここで測りたいのは「同じ g で丸め方を変えたときの差」だけだから（2 軸を
#: 同時に振ると方式の差と g の差が混ざる。追記 5 の位置づけ）。
W4_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: k-means の共有表 1 枚のビット数（16 準位 × f32）。品質だけ見ると**表の代金**が見えない
#: ので、サイズ試算では表のコストを式へ明示的に載せる。
CODEBOOK_BITS = DEFAULT_CODEBOOK_LEVELS * 32


@dataclass(frozen=True)
class TargetCounts:
    """w4 対象集合の計数（サイズ試算の入力）。"""

    modules: int
    channels: int
    elements: int

    def __add__(self, other: TargetCounts) -> TargetCounts:
        return TargetCounts(
            modules=self.modules + other.modules,
            channels=self.channels + other.channels,
            elements=self.elements + other.elements,
        )

    @property
    def groups(self) -> int:
        """g32 group の総数（group ごとに scale 1 個が要る方式の試算に使う）。

        `elements = Σ チャネル数 × in 軸`で、対象は in 軸が group 長で割り切れるものだけ
        （割り切れない重みは {@link census_w4_targets} が除外する）なので端数は出ない。
        """
        return self.elements // W4_GROUP_SIZE


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

    name: str
    #: 丸めを model へ in-place で当てる（`(model, op_types, include)`）。
    apply: Callable[[nn.Module, tuple[type[nn.Module], ...], Callable[[str], bool]], QuantReport]
    #: 量子化対象集合の投影ビット数。
    projected_bits: Callable[[TargetCounts], float]
    #: `projected_bits` の式（**出力へそのまま載せる** — 投影の前提を表から追えるように）。
    formula: str


#: RTN（= 唯一の格納形 `i4`）の方式名。方式列の**基準**で、他の 3 種はこれとの差で読む。
RTN_METHOD = "rtn-i4-g32"

#: 方式 4 種（スクリーニングの勝者 — FP4 と k-means の per_tensor / per_channel は安い
#: ファミリ 2 本の実測で落選済み）。
W4_METHODS: Mapping[str, W4Method] = MappingProxyType(
    {
        method.name: method
        for method in (
            W4Method(
                RTN_METHOD,
                lambda model, op_types, include: fake_quant_int4(
                    model, W4_GROUP_SIZE, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups,
                "4bit + g32 f32 scale = 5.0 bpw",
            ),
            W4Method(
                "nf4",
                lambda model, op_types, include: fake_quant_nf4(
                    model, W4_GROUP_SIZE, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups,
                "4bit + g32 f32 scale = 5.0 bpw（準位表は固定値なので模型に載らない）",
            ),
            W4Method(
                "mxfp4",
                lambda model, op_types, include: fake_quant_mxfp4(
                    model, W4_GROUP_SIZE, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 8 * counts.groups,
                "4bit + g32 E8M0 scale = 4.25 bpw",
            ),
            W4Method(
                "kmeans:shared",
                lambda model, op_types, include: fake_quant_kmeans(
                    model, "shared", W4_GROUP_SIZE, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups + CODEBOOK_BITS,
                f"4bit + g32 f32 scale + 表 {DEFAULT_CODEBOOK_LEVELS}×f32 を全体で 1 枚",
            ),
        )
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
    #: w4 の方式名（`weight == "w4"` のときだけ意味を持つ — {@link W4_METHODS} の鍵）。
    method: str | None = None
    #: w4 の対象役割（同上 — {@link W4_ROLES} の鍵）。
    roles: str = "all"
    #: BERT（DeBERTa）の linear へ当てる w4 方式。**None = f32 固定**（既存 5 構成の契約）。
    bert_method: str | None = None


#: 主要 11 構成（WAV を `--out` 直下へ書き、聴き比べの対象になる）。
CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        "f32": Recipe(None, None),
        "f16": Recipe("f16", None),
        "w8": Recipe("i8", None),
        "w8a16": Recipe("i8", "f16"),
        "w8a8": Recipe("i8", "i8"),
        # net_g の全役割 w4（格納形は無い測定列 — モジュール docstring）。
        "w4-rtn": Recipe("w4", None, method=RTN_METHOD),
        "w4-nf4": Recipe("w4", None, method="nf4"),
        "w4-mxfp4": Recipe("w4", None, method="mxfp4"),
        "w4-kmeans": Recipe("w4", None, method="kmeans:shared"),
        # BERT の linear だけ（net_g は f32 固定 — 既存 5 構成と直交する分離軸）。
        "bert-w4-rtn": Recipe(None, None, scope=(), bert_method=RTN_METHOD),
        "bert-w4-nf4": Recipe(None, None, scope=(), bert_method="nf4"),
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
    #: 除外したモジュール FQN（量子化軸が {@link W4_GROUP_SIZE} で割り切れない）。
    excluded: tuple[str, ...]
    excluded_elements: int
    #: モジュール型名 → 適格の計数（役割別の対象規模を報告へ出すため）。
    by_role: Mapping[str, TargetCounts]

    def include(self) -> Callable[[str], bool]:
        """core の `include`（モジュール FQN の述語）— 除外集合の補集合。"""
        excluded = frozenset(self.excluded)
        return lambda name: name not in excluded


def census_w4_targets(root: nn.Module, op_types: tuple[type[nn.Module], ...]) -> W4Census:
    """`root` の w4 対象を役割別に数え、量子化軸が group 長で割り切れない重みを外す。

    MUST: 割り切れない重みは**構成ごと落とすのではなく対象から外す**（i4 は端数 group を
    作らない — ADR 0069 決定 2。core は割り切れなければ fail loudly するので、外さないと
    測定そのものが立たない）。実測（FN4）で外れるのは受容野が 32 の倍数にならない conv
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
        if channel_rows(weight, axis).shape[-1] % W4_GROUP_SIZE:
            excluded.append(name)
            excluded_elements += weight.numel()
            continue
        eligible.append(name)
        row = totals[kinds[name]]
        row[0] += 1
        row[1] += int(weight.shape[axis])
        row[2] += weight.numel()
    by_role = {kind: TargetCounts(*row) for kind, row in sorted(totals.items())}
    total = TargetCounts(0, 0, 0)
    for counts in by_role.values():
        total = total + counts
    return W4Census(
        counts=total,
        eligible=tuple(eligible),
        excluded=tuple(excluded),
        excluded_elements=excluded_elements,
        by_role=MappingProxyType(by_role),
    )


def net_g_census(model_dir: Path) -> Mapping[tuple[str, str], W4Census]:
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
        (roles, tag): census_w4_targets(root, op_types)
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


def build_projections(counts: Mapping[str, TargetCounts]) -> list[dict[str, Any]]:
    """対象集合 × 方式の全組み合わせのサイズ試算。

    どの行に品質測定が付くかは {@link CONFIGS} から**引く**（別表に書くと構成を足したときに
    片方だけ更新される）。
    """
    measured: dict[tuple[str, str], str] = {}
    for name, recipe in CONFIGS.items():
        if recipe.weight == "w4" and recipe.method is not None:
            measured[(f"net_g:{recipe.roles}", recipe.method)] = name
        elif recipe.bert_method is not None:
            measured[("bert:linear", recipe.bert_method)] = name
    rows: list[dict[str, Any]] = []
    for target, description in PROJECTION_TARGETS.items():
        for method in W4_METHODS.values():
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
                    "measured_by": measured.get((target, method.name)),
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


def project_distribution(dist_dir: Path, linear_fqns: frozenset[str]) -> dict[str, Any]:
    """配布形の**実ファイル**に対する「i8 → i4 g32」の縮小試算。

    対象は**今日の配布対応形 = linear の重みスロットだけ**（conv / embedding の i4 は格納形も
    実行経路も無い — ADR 0069 決定 5 / 追記 5）。配布形はテンソル名に torch の FQN をそのまま
    担ぐので、丸めの対象集合（`linear_fqns` = census が数えたのと同じ FQN）との突合で引ける。
    「rank 2 の I8」という形だけで引くと embedding が混ざる（`enc_p.emb` / `word_embeddings`）
    ので、突合しなかった rank 2 の I8 は本数を返して読み手が監査できる形にする。

    式（テンソル 1 本 `[O,I]` あたり・逐語）:

        i8 … `O·I` バイト + scale `[O,1]` の `4·O` バイト（どちらも**配布形の実バイト**）
        i4 … `O·I/2` バイト（nibble 詰め）+ scale `[O,I/32]` の `4·O·(I/32)` バイト

    分母は配布ディレクトリの実ファイル総バイト。`shared/` の下（話者間で 1 本の DeBERTa）と
    話者ごとの net_g を分けて出す — 前者は 1 本ぶん、後者は話者数ぶん効く。
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
            if in_axis % W4_GROUP_SIZE:
                raise AssertionError(
                    f"{path}: '{key}' の in 軸 {in_axis} が group {W4_GROUP_SIZE} で割り切れない"
                )
            scale = header.get(DIST_SCALE_PREFIX + key)
            if scale is None:
                raise AssertionError(f"{path}: '{key}' に対応する scale テンソルが無い")
            row = groups[bucket]
            row[0] += 1
            row[1] += tensor_bytes(entry) + tensor_bytes(scale)
            row[2] += out_channels * in_axis // 2 + 4 * out_channels * (in_axis // W4_GROUP_SIZE)
    delta = sum(row[1] - row[2] for row in groups.values())
    return {
        "root": str(dist_dir),
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
        "formula": "i8 = O·I + 4·O バイト / i4 = O·I/2 + 4·O·(I/32) バイト"
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


def prepare_inputs(dump_path: Path, assets_path: Path) -> ChainInputs:
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
    bert_census = census_w4_targets(bert, W4_ROLES["linear"])
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


def quantized_bert_feature(method_name: str, inputs: ChainInputs) -> tuple[torch.Tensor, str]:
    """BERT の **linear の重みだけ**を w4 方式で丸めてから特徴を採る（`(特徴, 計数)`）。

    MUST: 模型は**素の重みから読み直す**（方式を積み重ねない）— net_g 側が構成ごとに
    `load_net_g` を呼び直しているのと同じ pristine の採り方で、退避／復元の可変状態を持たない。
    対象を linear 限定にするのは i4 の実行経路がそこだけだから（ADR 0069 決定 5）。BERT の
    embedding（`word_embeddings` / `rel_embeddings`）と 1 本の conv は f32 のまま残る。
    """
    bert = load_bert()
    report = W4_METHODS[method_name].apply(bert, W4_ROLES["linear"], inputs.bert_census.include())
    feature = bert_feature_of(bert, inputs.tensors)
    del bert
    gc.collect()
    return feature, report.describe()


def run_config(
    name: str,
    model_dir: Path,
    inputs: ChainInputs,
    *,
    bert_feature: torch.Tensor,
    census: Mapping[tuple[str, str], W4Census],
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
            report = W4_METHODS[recipe.method].apply(
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
            "method": recipe.method,
            "roles": recipe.roles if recipe.weight == "w4" else None,
            "group_size": W4_GROUP_SIZE if recipe.weight == "w4" else None,
            "targets": w4_targets,
        },
        "bert_method": recipe.bert_method,
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
            "w4": "方式は **g=32 固定**（ADR 0069 追記 5）。net_g は全役割・BERT は linear 限定で、"
            "丸めは core（karume.quantize / karume.quant_methods）の共有。サイズは実測ではなく"
            "**式による投影**で、格納形を持つのは RTN（`i4`）だけ",
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
        f" {projection['total_bytes'] / mib:.1f} MiB）に対する i8 → i4 g32 の縮小試算:",
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
        "group_size": W4_GROUP_SIZE,
        "methods": {name: method.formula for name, method in W4_METHODS.items()},
        "census": {
            "net_g": {
                f"{roles}/{tag}": _census_entry(entry) for (roles, tag), entry in census.items()
            },
            "bert": _census_entry(inputs.bert_census),
        },
        "projections": build_projections(counts),
        "bert_quant": dict(bert_reports),
    }
    if dist_dir.is_dir():
        section["distribution"] = project_distribution(dist_dir, linear_fqns)
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


def main() -> None:
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = selected_configs(args.configs)

    inputs = prepare_inputs(args.dump, args.assets)
    print(
        f"[inputs] text={inputs.meta['text']!r} bert={list(inputs.bert_feature.shape)}"
        f" bert_linear={inputs.bert_census.counts.modules}",
        flush=True,
    )
    census = net_g_census(args.model_dir)
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

    # BERT を振る構成の特徴（方式ごとに 1 度だけ作る — 素の重みから読み直すので積み重ならない）。
    features: dict[str | None, torch.Tensor] = {None: inputs.bert_feature}
    bert_reports: dict[str, dict[str, Any]] = {}

    results: dict[str, dict[str, Any]] = {}
    wavs: dict[str, Path] = {}
    for name, directory in plan:
        method = RECIPES[name].bert_method
        if method is not None and method not in features:
            feature, described = quantized_bert_feature(method, inputs)
            features[method] = feature
            bert_reports[method] = {
                "quant": described,
                "differs_from_f32": not bool(torch.equal(feature, inputs.bert_feature)),
                "rel_rms_vs_f32": rel_rms(feature, inputs.bert_feature),
            }
            print(f"[bert:{method}] {described}", flush=True)
        results[name] = run_config(
            name,
            args.model_dir,
            inputs,
            bert_feature=features[method],
            census=census,
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
    w4 = build_w4_section(census, inputs, bert_reports, args.dist_dir)
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
