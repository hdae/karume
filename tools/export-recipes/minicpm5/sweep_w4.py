"""MiniCPM5-1B の w4 fake-quant sweep（ADR [0069](../../../docs/decisions/0069-packed-w4-storage.md)
決定 6 = Phase 0）と**量子化方式スクリーニング**の台本。

    uv run --with 'transformers==5.14.1' python -m minicpm5.sweep_w4

**runtime は 1 行も触らない**。ここでやるのは torch 側で重みを丸めて（fake-quant）品質の
劣化を測ることだけで、格納形（`storage.dtype: "i4"` / pack 順 / WGSL）は測定の結果を見て
から実装する。ADR 0058 決定 5 の分担で言えば「品質 = 人間レビュー + 助言的数値」の
数値側を作る側で、機械門はここでは baseline の再現性 1 本だけ（{@link
assert_baseline_reproduces}）。

## 何を測るか

**Phase 0 グリッド**（{@link SWEEP_CONFIGS}）= group_size {32, 64, 128} × 対称 / 非対称の
6 通り + baseline + `lm_head` 除外の 1 本。

**方式グリッド**（{@link METHOD_CONFIGS}）= 方式 7 種（{@link QUANT_METHODS} — RTN i4 /
FP4 / NF4 / MXFP4 / k-means の表 3 粒度）× 対象 2 形（linear 限定 169 本 / `embed_tokens`
込み 170 本）。**group_size は 32 固定**で g 軸は振らない — g の答えは Phase 0 が出して
いて、ここで測りたいのは「同じ g で丸め方を変えたときの差」だけだから（2 軸を同時に
振ると方式の差と g の差が混ざる）。丸めの実装は core（`karume.quant_methods` /
`karume.quantize`）の共有で、対象選択も `iter_quant_targets` の共有 — 写した別実装にすると
「測った対象・式」と「出荷する対象・式」が黙って割れる。

**校正グリッド**（{@link CALIB_CONFIGS}）= 校正付き丸め 5 本（GPTQ × 格納グリッド 3 種 /
AWQ / AWQ+GPTQ）。丸めは core の `karume.quant_calib` の共有で、**格子は方式グリッドと同じ**
（変わるのは同じ格子の中でどの準位へ寄せるか）。対象は **decoder 内の linear 限定の 1 形**
だけで、`embed_tokens` 込みの組も `lm_head` も作らない — 校正の駆動（`calibrate_stages`）が
stage 内の `nn.Linear` に閉じているため、既存の「方式 7 種 × 対象 2 形」の表とは対象集合が
違う（{@link CalibConfig}）。校正入力は {@link minicpm5.calib_texts.CALIB_TEXTS} の 48 文で、
先頭 decoder layer への呼び出しを捕まえて stage 列へ流す（{@link capture_stage_batches}）。

config ごとに 4 列を採る（先の 3 列は Phase 0 と同じ・4 列目は方式グリッドと校正グリッド）:

1. **weight 相対 RMSE**（族別）— 丸めそのものの大きさ。族内の全層をまとめた
   `‖w − fq‖₂ / ‖w‖₂` で、どの族が壊れやすいかを見る。
2. **teacher-forced 一致 / NLL** — 期待列を 1 つずつ与えたときの次トークン。発散しても列が
   ずれないので、config 間で**同じ位置**の劣化量を比べられる。
3. **自由走行 greedy の発散 step** — 実際の生成が期待列から離れるまでの長さ。人間が読む
   ときの「壊れ方」に一番近い列。
4. **サイズ試算**（{@link QuantMethod.formula}）— 方式ごとの実効 bpw と、量子化対象テンソル
   集合だけを投影した MiB。品質だけ見ると表のコストが見えない（k-means per-channel は
   表がチャネル数に比例する）ので、品質と同じ表で並べる。**実測ではなく式による投影**で、
   格納形を持たない方式（この台本の 7 種のうち RTN 以外全部）は書けもしない。

期待列は**波 E の資産が正本**（`outputs/series/minicpm5-1b-decode/greedy.<case>.safetensors`
— margin 門つきで採った 3 ケース）。ここで採り直さないのは、sweep が測りたいのが
「同じ期待列に対する量子化の劣化」だけだから。tokenizer は通さない（トークン id で完結する）。

## 量子化の形（ADR 0069 決定 3）

group 軸は**重みの最終次元 = linear の in 軸**。対称は `s = clamp(amax/7, f32 tiny)` /
`q = clamp(round(w/s), ±7)` で、amax 要素が厳密復元されるので fake-quant が冪等になる
（ADR 0019 の ±127 論証の 4bit 版）。非対称は**測定列**で、zero-point を丸めも格納もしない
連続値のまま使う — 「zero-point を足したときに届きうる品質の上界」を見て、対称の品質不足を
実装前に検知するための列（決定 3 の予約 3）。

## 出力

進捗は stderr へ即 flush（背景実行で追うため）。最後に stdout へ markdown 表 4 枚
（要約 / 品質 / 族別 wRMSE / サイズ試算）を出す — そのまま `docs/research/` へ転記する形。
`--json` を渡すと機械可読の JSON も書く（**config 1 本ごとに書き直す** — 数十分級の実行が
途中で落ちても、そこまでの測定値が残る）。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from _shared.decode_series import EXPECTED_KEY, GREEDY_PREFIX, GREEDY_SUFFIX, PROMPT_KEY
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
    fake_quant_fp4,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QuantizeError,
    fake_quant_int4,
    iter_quant_targets,
)
from minicpm5 import export as one_shot
from minicpm5 import export_decode as decode
from minicpm5.calib_texts import CALIB_TEXTS

#: 実重みの置き場（1-shot / decode 形と同じ素材）。
DEFAULT_MODEL_DIR = one_shot.DEFAULT_MODEL_DIR

#: 期待列（波 E の greedy golden）の置き場。
DEFAULT_DECODE_DIR = decode.DEFAULT_OUT_DIR

#: 対称量子化の片側幅（**−8 は使わない**）。±7 に閉じると group の amax 要素が `q = ±7` に
#: 乗って `q·s` で厳密に復元され、fake-quant が**冪等**になる（ADR 0069 決定 3 — ADR 0019 の
#: ±127 論証の 4bit 版）。−8 を許すと scale だけが動く再量子化が起きる。
INT4_MAX = 7

#: 非対称の unsigned nibble の値域（**0 は未使用**の 15 準位）。対称の `q ∈ [−7,7]` を
#: offset 8 で載せた先と同じ値域で、pack 形式・pack 順は zero-point の有無で変わらない
#: （ADR 0069 決定 3 の予約 2）。
UINT4_MIN = 1
UINT4_MAX = 15

#: zero-point 省略時の既定（ADR 0069 決定 3 の予約 2 — dequant は `(u − 8)·scale`）。
#: 非対称の縮退 group はこの既定へ落ちる（{@link asymmetric_components}）。
DEFAULT_ZERO_POINT = 8

#: 量子化対象の linear 族（fqn の末尾モジュール名）。MiniCPM5-1B は 24 層 ×
#: q/k/v/o/gate/up/down + `lm_head` の 169 本で、`tie_word_embeddings: false` なので
#: `lm_head` は独立した重み。族に割るのは「どこが壊れるか」を分離するため。
LINEAR_FAMILIES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

#: 語彙側の族名（`--only` 相当の除外指定 {@link SweepConfig.include_lm_head} が指す先）。
LM_HEAD = "lm_head"

#: 入力側の embedding の族名（方式グリッドの「非 linear 込み」で増える 1 本）。
#: `tie_word_embeddings: false` なので `lm_head` とは独立した重み。
EMBED_TOKENS = "embed_tokens"

#: 族別 wRMSE 表の列（方式グリッドは `embed_tokens` まで載りうる）。
METHOD_FAMILIES: tuple[str, ...] = (*LINEAR_FAMILIES, EMBED_TOKENS)

#: baseline の config 名（{@link select_configs} が常に先頭へ入れる）。
BASELINE_NAME = "baseline"


@dataclass(frozen=True)
class SweepConfig:
    """1 実行ぶんの指定。

    `group_size is None` は量子化しない baseline。`include_lm_head=False` は `lm_head` を
    対象から外す列で、語彙側 200M 本の寄与を block 側から分離するために置く。
    """

    name: str
    group_size: int | None = None
    asymmetric: bool = False
    include_lm_head: bool = True


#: sweep グリッド（この順で走る）。group_size は 2 冪かつ ≥ 16（ADR 0069 決定 2）。
SWEEP_CONFIGS: tuple[SweepConfig, ...] = (
    SweepConfig(BASELINE_NAME),
    SweepConfig("g32-sym", group_size=32),
    SweepConfig("g64-sym", group_size=64),
    SweepConfig("g128-sym", group_size=128),
    SweepConfig("g32-asym", group_size=32, asymmetric=True),
    SweepConfig("g64-asym", group_size=64, asymmetric=True),
    SweepConfig("g128-asym", group_size=128, asymmetric=True),
    SweepConfig("g128-sym-blocks", group_size=128, include_lm_head=False),
)


# ---- 方式グリッド（量子化方式スクリーニング）-------------------------------

#: 方式比較の group 長 = **32 固定**（core の既定 = Phase 0 の実測で確定した値）。g 軸を
#: 同時に振らないのは、方式の差と g の差が混ざると「どちらが効いたか」が言えなくなるため。
METHOD_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: k-means の表 1 枚のビット数（16 centroid × f32）。
CODEBOOK_BITS = DEFAULT_CODEBOOK_LEVELS * 32


@dataclass(frozen=True)
class TargetCounts:
    """量子化対象テンソル集合の計数（サイズ試算の入力）。

    `channels` は {@link karume.quantize.channel_rows} の行数の総和 = per-channel の表を
    張る単位の数で、`modules` は層数（per-tensor の表の枚数）。
    """

    modules: int
    channels: int
    elements: int

    @property
    def groups(self) -> int:
        """g32 group の総数（group ごとに scale 1 個が要る方式のサイズ試算に使う）。"""
        return self.elements // METHOD_GROUP_SIZE


@dataclass(frozen=True)
class QuantMethod:
    """丸め方式 1 種 — 名前・丸めの当て方・サイズ試算の式を 1 行に束ねる。

    3 つを別表に散らすと「品質を測った方式」と「サイズを試算した方式」が黙って割れる
    （方式を 1 種足したときに片方だけ更新される形になる）。
    """

    #: 表と JSON に出る方式名。
    name: str
    #: config 名の接頭（`<slug>-linear` / `<slug>-embed`）。
    slug: str
    #: 丸めを model へ in-place で当てる（`(model, op_types, group_size)`）。
    round_weights: Callable[[nn.Module, tuple[type[nn.Module], ...], int], object]
    #: 量子化対象集合の投影ビット数。
    projected_bits: Callable[[TargetCounts], float]
    #: `projected_bits` の式（**出力へそのまま載せる** — 投影の前提を表から追えるように）。
    formula: str


#: RTN（= 格納形 `i4`）の方式名。方式列の**基準** — 他の 6 種はこれとの差で読む。
RTN_METHOD = "rtn-g32-sym"

#: k-means 方式名の接頭（`kmeans:<粒度>` — core の {@link
#: karume.quant_methods.KMEANS_GRANULARITIES} と同じ綴り）。
KMEANS_PREFIX = "kmeans:"

#: 方式 7 種（この順で走る）。丸めは全て core の共有実装で、ここは呼び分けだけを持つ。
QUANT_METHODS: tuple[QuantMethod, ...] = (
    QuantMethod(
        RTN_METHOD,
        "rtn",
        lambda model, op_types, group: fake_quant_int4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups,
        "4bit + g32 f32 scale = 5.0 bpw",
    ),
    QuantMethod(
        "fp4",
        "fp4",
        lambda model, op_types, group: fake_quant_fp4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups,
        "4bit + g32 f32 scale = 5.0 bpw",
    ),
    QuantMethod(
        "nf4",
        "nf4",
        lambda model, op_types, group: fake_quant_nf4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups,
        "4bit + g32 f32 scale = 5.0 bpw",
    ),
    QuantMethod(
        "mxfp4",
        "mxfp4",
        lambda model, op_types, group: fake_quant_mxfp4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 8 * counts.groups,
        "4bit + g32 E8M0 scale = 4.25 bpw",
    ),
    QuantMethod(
        f"{KMEANS_PREFIX}per_tensor",
        "kmeans-tensor",
        lambda model, op_types, group: fake_quant_kmeans(
            model, "per_tensor", group, op_types=op_types
        ),
        lambda counts: 4 * counts.elements + CODEBOOK_BITS * counts.modules,
        "4bit + 表 16×f32 毎層（scale 無し）",
    ),
    QuantMethod(
        f"{KMEANS_PREFIX}per_channel",
        "kmeans-channel",
        lambda model, op_types, group: fake_quant_kmeans(
            model, "per_channel", group, op_types=op_types
        ),
        lambda counts: 4 * counts.elements + CODEBOOK_BITS * counts.channels,
        "4bit + 表 16×f32 毎チャネル（scale 無し）",
    ),
    QuantMethod(
        f"{KMEANS_PREFIX}shared",
        "kmeans-shared",
        lambda model, op_types, group: fake_quant_kmeans(model, "shared", group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups + CODEBOOK_BITS,
        "4bit + g32 f32 scale + 表 16×f32 を全体で 1 枚",
    ),
)


@dataclass(frozen=True)
class MethodConfig:
    """方式グリッドの 1 実行ぶん（方式 × 対象 2 形）。

    `include_embedding` は「非 linear 込み」の列 — `embed_tokens` を対象へ足す。i4 の実行
    経路は linear 限定（ADR 0069 決定 5）なので**出荷できる形ではない**が、語彙側を
    丸めたときの品質とサイズの取り分は測っておく価値がある（混成格納の判断材料）。
    """

    name: str
    method: QuantMethod
    include_embedding: bool


#: 方式グリッド = 7 方式 × 対象 2 形（linear 限定 / `embed_tokens` 込み）。
METHOD_CONFIGS: tuple[MethodConfig, ...] = tuple(
    MethodConfig(
        name=f"{method.slug}-{'embed' if include_embedding else 'linear'}",
        method=method,
        include_embedding=include_embedding,
    )
    for method in QUANT_METHODS
    for include_embedding in (False, True)
)


# ---- 校正グリッド（校正付き丸め）-------------------------------------------

#: 校正付き丸めの group 長 = 方式グリッドと同じ 32 固定。校正は**格子を 1 バイトも変えない**
#: （同じ格子の中で丸め先を選び直すだけ — `karume.quant_calib` のモジュール docstring）ので、
#: g 軸を振らない理由も {@link METHOD_GROUP_SIZE} と同文。
CALIB_GROUP_SIZE = METHOD_GROUP_SIZE


def group_scale_bits(counts: TargetCounts) -> float:
    """4bit ペイロード + group ごとの f32 scale（RTN / NF4 の格納そのもの）。"""
    return 4 * counts.elements + 32 * counts.groups


def layer_table_bits(counts: TargetCounts) -> float:
    """上に**層ごと 1 枚**の k-means 表を足した式。

    core の `kmeans_shared` は「group absmax で正規化した値空間に 1 枚の表」を張るが、
    その射程は**層内**に閉じる（stage 逐次の駆動と全層を跨ぐ表が噛み合わない —
    {@link karume.quant_calib.GridSpec} の NOTE）。方式グリッドの `kmeans:shared`
    （全体で 1 枚）と式が違うのはこのため。
    """
    return group_scale_bits(counts) + CODEBOOK_BITS * counts.modules


#: group absmax scale を持つ 4bit 格子の投影式（表を持たない方式で共有する文言）。
GROUP_SCALE_FORMULA = "4bit + g32 f32 scale = 5.0 bpw"

#: AWQ 系の投影式に付ける注記 — 等価倍率 `s` は `W_eff` に畳み込んだ形でしか持てず、
#: 格納するには fold か companion が要る（`karume.quant_calib` のモジュール docstring）。
AWQ_FORMULA_NOTE = "（等価倍率 s の fold / companion は未計上）"


@dataclass(frozen=True)
class CalibConfig:
    """校正付き丸め 1 本ぶんの指定（方式 × 格納グリッド）。

    対象は **decoder 内の `nn.Linear` 限定**（{@link karume.quant_calib.calibrate_stages} が
    stage 内の linear しか見ない）。`lm_head` と `embed_tokens` は stage の外なので入らず、
    方式グリッドの「対象 2 形」とは**対象集合が違う** — サイズ試算も校正が実際に丸めた
    集合の計数から出す（{@link calib_targets}）。
    """

    name: str
    method: CalibMethod
    grid: GridSpec
    #: 量子化対象集合の投影ビット数（{@link QuantMethod.projected_bits} と同じ器）。
    projected_bits: Callable[[TargetCounts], float]
    #: `projected_bits` の式（**出力へそのまま載せる**）。
    formula: str

    @property
    def label(self) -> str:
        """表と JSON に出る方式名（`<方式>/<格納グリッド>`）。"""
        return f"{self.method}/{self.grid.kind}"


#: 校正グリッド = 5 本（この順で走る）。丸めは全て core の共有実装
#: （`karume.quant_calib`）で、ここは呼び分けと投影式だけを持つ。**対象は linear 限定の
#: 1 形のみ** — `embed_tokens` 込みの組は作らない（`calibrate_stages` は `nn.Linear` 限定で、
#: 語彙表は stage の外）。
CALIB_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn",
        "gptq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        GROUP_SCALE_FORMULA,
    ),
    CalibConfig(
        "gptq-nf4",
        "gptq",
        GridSpec(kind="nf4", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        f"{GROUP_SCALE_FORMULA}（格子は NF4 の固定表）",
    ),
    CalibConfig(
        "gptq-kmeans",
        "gptq",
        GridSpec(kind="kmeans_shared", group_size=CALIB_GROUP_SIZE),
        layer_table_bits,
        "4bit + g32 f32 scale + 表 16×f32 を層ごと 1 枚",
    ),
    CalibConfig(
        "awq-rtn",
        "awq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        f"{GROUP_SCALE_FORMULA}{AWQ_FORMULA_NOTE}",
    ),
    CalibConfig(
        "awq-gptq-rtn",
        "awq+gptq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        f"{GROUP_SCALE_FORMULA}{AWQ_FORMULA_NOTE}",
    ),
)

#: `--only` が選べる config の型（Phase 0 / 方式 / 校正の 3 系統）。
AnyConfig = SweepConfig | MethodConfig | CalibConfig

#: `--only` が選べる全 config（宣言順 = 実行順）。
ALL_CONFIGS: tuple[AnyConfig, ...] = (*SWEEP_CONFIGS, *METHOD_CONFIGS, *CALIB_CONFIGS)


# ---- 量子化式（ADR 0069 決定 3）--------------------------------------------


def grouped_view(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """量子化軸（**最終次元 = linear の in 軸**）を `[…, groups, group_size]` へ割る。

    MUST: 割り切れない形は fail loudly。`i4` は in 軸が `group_size` で割り切れることを
    MUST とする（ADR 0069 決定 2 — 端数 group を作らない制約で行境界・group 境界のバイト
    整列を保証している）ので、端数を黙って作った形の品質を測ると「格納できない形の数値」を
    根拠に既定を決めることになる。
    """
    in_axis = int(weight.shape[-1])
    if in_axis % group_size:
        raise QuantizeError(
            f"in 軸 {in_axis} が group_size {group_size} で割り切れない"
            "（`i4` は量子化軸が group_size で割り切れることを MUST とする — ADR 0069 決定 2）"
        )
    return weight.reshape(*weight.shape[:-1], in_axis // group_size, group_size)


def symmetric_components(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """group 対称量子化の `(q, scale)` を group 形（`[…, groups, *]`）で返す。

    `scale = clamp(amax / 7, f32 tiny)` / `q = clamp(round(w / scale), ±7)`。下限 clamp が
    要るのは全ゼロ group（`amax == 0`）— そのまま割ると NaN になる。clamp 後は `q = 0` に
    落ちて `q·scale = 0` が厳密に元値へ戻る（`quantize.channel_scale` と同文の 4bit 版）。

    fq（{@link fake_quant_symmetric}）と別に成分を返すのは、値域（**−8 が出ないこと**）と
    group ごとに scale が分かれていることをテストが直接縛れるようにするため。
    """
    grouped = grouped_view(weight, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / INT4_MAX, min=torch.finfo(torch.float32).tiny)
    return torch.round(grouped / scale).clamp_(-INT4_MAX, INT4_MAX), scale


def fake_quant_symmetric(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """対称 15 準位の fake-quant（`q·scale` を weight と同じ形で返す）。

    MUST: **冪等**（再適用でビット不変）。amax 要素が `q = ±7` に乗って厳密復元されることの
    帰結で、崩れると「丸め済みのはずの重みが再量子化のたびに動く」= 格納形と参照の対応が
    切れる（ADR 0019 / 0069 決定 3）。`tests/test_sweep_w4.py` が固定する。
    """
    q, scale = symmetric_components(weight, group_size)
    return (q * scale).reshape(weight.shape)


def asymmetric_components(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """非対称（**連続** zero-point）の `(u, scale, zero)` を group 形で返す — 測定専用。

    `scale = clamp((max − min) / 14, f32 tiny)` / `zero = 1 − min/scale` /
    `u = clamp(round(w/scale + zero), 1, 15)`。zero を丸めも格納もしないので、これは
    「zero-point 欄を足したときに届きうる品質の**上界**」であって格納形ではない
    （ADR 0069 決定 3 の予約 3 — 対称の品質不足を実装前に検知するための測定列）。

    MUST: 縮退 group（`max == min`）は**対称式へ落とす**（`scale = clamp(|c|/7, tiny)` /
    `zero = 8`）。min-max 式では `scale` が下限 clamp に張り付いて `w/scale` が発散し、
    定数 group が clamp で ~0 へ潰れる（0 でない定数 group が黙って 0 になる沈黙誤値）。
    """
    grouped = grouped_view(weight, group_size)
    tiny = torch.finfo(torch.float32).tiny
    upper = grouped.amax(dim=-1, keepdim=True)
    lower = grouped.amin(dim=-1, keepdim=True)
    degenerate = upper == lower
    span = torch.clamp((upper - lower) / (UINT4_MAX - UINT4_MIN), min=tiny)
    scale = torch.where(degenerate, torch.clamp(upper.abs() / INT4_MAX, min=tiny), span)
    zero = torch.where(
        degenerate,
        torch.full_like(span, float(DEFAULT_ZERO_POINT)),
        UINT4_MIN - lower / span,
    )
    u = torch.round(grouped / scale + zero).clamp_(UINT4_MIN, UINT4_MAX)
    return u, scale, zero


def fake_quant_asymmetric(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """非対称 15 準位の fake-quant（`(u − zero)·scale` を weight と同じ形で返す）。

    NOTE: 対称と違い**冪等は保証しない** — min / max 要素の復元は f32 の丸めの分だけ緩く、
    再適用で `scale` が 1ulp 動きうる。測定列なので冪等は要求しない（格納形は対称のみで
    始める — ADR 0069 決定 3）。
    """
    u, scale, zero = asymmetric_components(weight, group_size)
    return ((u - zero) * scale).reshape(weight.shape)


# ---- 量子化対象 -------------------------------------------------------------


def family_of(fqn: str) -> str:
    """重みの fqn（`….q_proj.weight`）→ 族名（末尾のモジュール名）。"""
    return fqn.split(".")[-2]


def linear_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """全 `nn.Linear` の `weight` を fqn 引きで集める（bias / norm / embedding は触らない）。

    fqn は `named_modules` で引く（`quantize.fake_quant_int8` と同じ流儀 — `id(tensor)` で
    突き合わせない。パラメータ同一性は正規化で簡単に崩れ、崩れても黙って対象が減るだけ）。

    MUST: 族が {@link LINEAR_FAMILIES} に無い linear は fail loudly。transformers 側の構成が
    変わって対象が増減したことを黙って測り続けると、族別の表が別のモデルの数値になる。
    """
    weights: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        fqn = f"{name}.weight"
        if family_of(fqn) not in LINEAR_FAMILIES:
            raise QuantizeError(
                f"'{fqn}': 未知の linear 族（対象は {list(LINEAR_FAMILIES)}）"
                "— 模型の構成が sweep の想定と違う"
            )
        weights[fqn] = module.weight
    if not weights:
        raise QuantizeError("nn.Linear が 1 本も無い（模型の構成が sweep の想定と違う）")
    return weights


def method_op_types(include_embedding: bool) -> tuple[type[nn.Module], ...]:
    """方式グリッドの対象 2 形 → `iter_quant_targets` の `op_types`。

    非 linear 込みの列で足すのを `nn.Embedding` だけに絞るのは、MiniCPM5-1B に他の量子化
    可能型（conv 系）が 1 本も無いから — i8 の 5 種を丸ごと渡しても対象は同じで、「広い型を
    渡した」という見かけだけが増える。
    """
    return (nn.Linear, nn.Embedding) if include_embedding else (nn.Linear,)


def collect_targets(
    model: nn.Module, op_types: tuple[type[nn.Module], ...]
) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """量子化対象を fqn 引きの重みと計数で返す（対象選択は core の共有）。

    MUST: 族が {@link METHOD_FAMILIES} に無い対象は fail loudly（{@link linear_weights} と
    同文）。対象が黙って増減すると、族別の表もサイズ試算も別のモデルの数値になる。
    """
    weights: dict[str, torch.Tensor] = {}
    channels = 0
    elements = 0
    for fqn, weight, axis in iter_quant_targets(model, op_types):
        if family_of(fqn) not in METHOD_FAMILIES:
            raise QuantizeError(
                f"'{fqn}': 未知の族（対象は {list(METHOD_FAMILIES)}）"
                "— 模型の構成が sweep の想定と違う"
            )
        weights[fqn] = weight
        channels += int(weight.shape[axis])
        elements += int(weight.numel())
    if not weights:
        raise QuantizeError(
            f"量子化対象が 1 本も無い（対象の型: {', '.join(cls.__name__ for cls in op_types)}）"
        )
    return weights, TargetCounts(modules=len(weights), channels=channels, elements=elements)


# ---- サイズ試算 -------------------------------------------------------------


@dataclass(frozen=True)
class SizeProjection:
    """方式 × 対象集合の**投影**サイズ（実測ではない — 式は {@link QuantMethod.formula}）。"""

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


def project_size(method: QuantMethod, counts: TargetCounts) -> SizeProjection:
    """方式の式を対象集合の計数へ当てる。"""
    return SizeProjection(counts=counts, bits=method.projected_bits(counts), formula=method.formula)


# ---- 方式の適用 -------------------------------------------------------------


def apply_method(model: nn.Module, config: MethodConfig, shared_stride: int) -> None:
    """方式グリッド 1 本の丸めを model へ in-place で当てる。

    `shared_stride > 1` のときだけ `kmeans:shared` の**表の fit** が等間隔部分標本になる
    （core の `fit_stride` — 適用は常に全量・他の方式には効かない）。全量 fit は 1B 級で
    実メモリを超えるための逃げ道で、使った事実は JSON と markdown の脚注へそのまま出る
    （{@link karume.quant_methods.fake_quant_kmeans} の MUST）。
    """
    op_types = method_op_types(config.include_embedding)
    if config.method.name == f"{KMEANS_PREFIX}shared" and shared_stride > 1:
        fake_quant_kmeans(
            model, "shared", METHOD_GROUP_SIZE, op_types=op_types, fit_stride=shared_stride
        )
        return
    config.method.round_weights(model, op_types, METHOD_GROUP_SIZE)


# ---- 校正付き丸めの駆動（Catcher + stage 分解）-------------------------------

#: decoder 層のモデル内 FQN 接頭辞（`CausalLmWrapper.model` = `LlamaForCausalLM` →
#: `.model` = `LlamaModel` → `.layers`）。接頭辞を `calibrate_stages` へ渡すのは、scale 台帳の
#: キーを {@link karume.quantize.Int4Report} と同じ**モデル内 FQN** の空間へ揃えるため
#: （core の `StageSpec` 契約）。
DECODER_PREFIX = "model.model.layers"


class _FirstStageReached(Exception):  # noqa: N818 — 異常ではなく打ち切りの合図なので Error と呼ばない
    """先頭 stage へ入力が届いた合図（校正 forward を打ち切るための番兵）。"""


def decoder_stages(wrapper: nn.Module) -> tuple[StageSpec, ...]:
    """実行順の decoder layer を `(モデル内 FQN 接頭辞, モジュール)` で返す。

    最終段の `norm` と `lm_head` は**入れない** — 校正の対象は decoder 内の linear だけで、
    `lm_head` を stage として足すと「その入力が最終 norm を通っていない」形になる。方式
    グリッドの 169 本と対象集合が違うのはこのため（{@link CalibConfig}）。
    """
    layers = wrapper.model.model.layers
    return tuple((f"{DECODER_PREFIX}.{index}", layer) for index, layer in enumerate(layers))


def calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """stage 内の量子化対象を fqn 引きの重みと計数で返す（校正の**走査** = 門の基準）。

    対象選択は core の `iter_quant_targets` の共有（{@link collect_targets} と同文 — 写した
    別実装にすると「測った対象」と「丸めた対象」が黙って割れる）。ここで数えた本数と、
    校正が実際に丸めた本数を {@link assert_calib_covers_scan} が突き合わせる。
    """
    weights: dict[str, torch.Tensor] = {}
    channels = 0
    elements = 0
    for prefix, stage in stages:
        for local, weight, axis in iter_quant_targets(stage, (nn.Linear,)):
            fqn = f"{prefix}.{local}"
            if family_of(fqn) not in LINEAR_FAMILIES:
                raise QuantizeError(
                    f"'{fqn}': 未知の linear 族（対象は {list(LINEAR_FAMILIES)}）"
                    "— 模型の構成が sweep の想定と違う"
                )
            weights[fqn] = weight
            channels += int(weight.shape[axis])
            elements += int(weight.numel())
    if not weights:
        raise QuantizeError("decoder stage に nn.Linear が 1 本も無い（模型の構成が想定と違う）")
    return weights, TargetCounts(modules=len(weights), channels=channels, elements=elements)


def calib_inputs(model_dir: Path, limit: int | None) -> tuple[torch.Tensor, ...]:
    """校正コーパスを**既存の tokenizer 経路**（{@link minicpm5.export.load_tokenizer}）で
    `[1,T]` の id 列へ落とす。

    `limit` は先頭 N 文の上限（縮小 smoke 用 — `None` は全 48 文）。並びの先頭から採るのは
    {@link minicpm5.calib_texts.CALIB_TEXTS} が言語とスタイルを混ぜて並べてあるため。
    """
    if limit is not None and limit < 1:
        raise ValueError(f"校正文数の上限は 1 以上（実測 {limit}）")
    texts = CALIB_TEXTS if limit is None else CALIB_TEXTS[:limit]
    tokenizer = one_shot.load_tokenizer(model_dir)
    return tuple(torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64) for text in texts)


def capture_stage_batches(
    wrapper: nn.Module, first: nn.Module, inputs: Sequence[torch.Tensor]
) -> tuple[StageBatch, ...]:
    """先頭 decoder layer への `(args, kwargs)` を forward_pre_hook で捕まえる（Catcher）。

    校正入力は「embedding・RoPE 表・加算 causal mask を通った後の hidden とその付随引数」で、
    自前で組み直すと transformers 側の前段（と `export.py` のラッパが渡す mask の形）と黙って
    割れる。**先頭 stage の呼び出しそのもの**を捕まえ、番兵例外で forward を打ち切るのが、
    前段を写さずに同じ入力を採る形。

    kwargs（mask / position_ids / position_embeddings …）は Llama では**全層で同一**なので、
    `calibrate_stages` の既定（stage 間で不変として運ぶ）にそのまま乗る。

    MUST: 捕まえずに forward が完走したら fail loudly — stage の綴りが模型の構成と食い違って
    いる合図で、黙って進むと「校正入力ゼロ」の診断が core 側で出るだけになる。
    """
    captured: list[StageBatch] = []

    def catcher(_module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        captured.append((tuple(args), dict(kwargs)))
        raise _FirstStageReached

    handle = first.register_forward_pre_hook(catcher, with_kwargs=True)
    try:
        for index, ids in enumerate(inputs):
            try:
                with torch.no_grad():
                    wrapper(ids)
            except _FirstStageReached:
                continue
            raise AssertionError(
                f"校正入力 {index} で先頭 decoder layer が呼ばれずに forward が完走した"
                "（stage の綴りが模型の構成と食い違っている）"
            )
    finally:
        handle.remove()
    return tuple(captured)


@dataclass(frozen=True)
class CalibRig:
    """校正付き構成が共有する足場（stage 列・走査・先頭 stage への入力）。

    config ごとに作り直さない — 先頭 stage の入力は `embed_tokens` を通っただけの hidden で、
    校正が丸めるのは decoder 内の linear だけなので、pristine 復元を挟んでも動かない
    （**丸めを 1 本も当てる前**に 1 回だけ採る）。
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    counts: TargetCounts
    batches: tuple[StageBatch, ...]


def build_calib_rig(wrapper: nn.Module, model_dir: Path, limit: int | None) -> CalibRig:
    """校正の足場を組む（stage 分解 → 走査 → tokenize → Catcher）。"""
    stages = decoder_stages(wrapper)
    scan, counts = calib_targets(stages)
    batches = capture_stage_batches(wrapper, stages[0][1], calib_inputs(model_dir, limit))
    return CalibRig(stages=stages, scan=scan, counts=counts, batches=batches)


def apply_calib(config: CalibConfig, rig: CalibRig, shared_stride: int) -> CalibReport:
    """校正付き丸め 1 本を model へ in-place で当てる（stage 逐次の駆動は core 側）。

    `shared_stride > 1` のときだけ `kmeans_shared` の**表の fit** が等間隔部分標本になる
    （{@link apply_method} と同じ逃げ道・同じノブ）。使った事実は
    {@link karume.quant_calib.CalibReport.describe} が拾って出力へ出る。
    """
    spec = config.grid
    if spec.kind == "kmeans_shared" and shared_stride > 1:
        spec = replace(spec, fit_stride=shared_stride)
    return calibrate_stages(rig.stages, rig.batches, method=config.method, spec=spec)


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """校正が丸めた層が stage の走査と**過不足なく**一致することを見る。

    MUST: fail loudly。stage の綴りや対象型が変わって decoder の一部が校正に載らなくなっても
    表には行が残り、しかも丸め漏れのぶん品質は**良い側**に出る（素通りを数字から読めない）。
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise AssertionError(
            f"[{name}] 校正が丸めた {report.modules} 本が走査の {len(scan)} 本と一致しない"
            f"（丸め漏れ {missing[:3]} / 走査に無い {extra[:3]}）"
        )


# ---- 期待列（波 E の資産）---------------------------------------------------


@dataclass(frozen=True)
class GreedyCase:
    """波 E の greedy 期待列 1 ケース（`prompt[1,T]` / `expected[K]` — どちらも i64）。"""

    name: str
    prompt: torch.Tensor
    expected: torch.Tensor

    @property
    def steps(self) -> int:
        return int(self.expected.numel())


def load_cases(decode_dir: Path) -> tuple[GreedyCase, ...]:
    """`greedy.<case>.safetensors` を glob で読む（tokenizer は通さない）。

    資産が無ければ**実行手順つきで**落とす — 60 分級の実行の入口で、直せる形の失敗を
    「ファイルが無い」だけで返さない。
    """
    paths = sorted(decode_dir.glob(f"{GREEDY_PREFIX}*{GREEDY_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"greedy 期待列が {decode_dir} に 1 件も無い — 先に波 E の decode 資産を作ること:\n"
            "  cd tools/export-recipes && "
            "uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode"
        )
    cases: list[GreedyCase] = []
    for path in paths:
        name = path.name[len(GREEDY_PREFIX) : -len(GREEDY_SUFFIX)]
        tensors = load_file(str(path))
        cases.append(
            GreedyCase(
                name=name,
                prompt=tensors[PROMPT_KEY].to(torch.int64).unsqueeze(0),
                expected=tensors[EXPECTED_KEY].to(torch.int64),
            )
        )
    return tuple(cases)


# ---- 測定 -------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """1 config × 1 ケースの測定値。"""

    steps: int
    matches: int
    nll: float
    prefix: int


@dataclass(frozen=True)
class ConfigResult:
    """1 config の測定値（族別 wRMSE は量子化した族だけが載る）。

    `overall_rmse` は対象集合を丸ごと 1 本にした `‖w − fq‖₂ / ‖w‖₂`（族平均ではない —
    族ごとの要素数が 20 倍違うので平均だと重み付けが恣意的になる）。`size` は方式グリッドと
    校正グリッドだけが持つ（Phase 0 の config は格納形が i4 の 1 通りで、試算しても同じ数が
    並ぶ）。
    """

    name: str
    weight_rmse: Mapping[str, float]
    cases: Mapping[str, CaseResult]
    seconds: float = 0.0
    overall_rmse: float | None = None
    #: 表と JSON に出る方式名（Phase 0 の config は方式軸を持たないので `None`）。
    method: str | None = None
    size: SizeProjection | None = None
    #: 校正付き構成の 1 行要約（{@link karume.quant_calib.CalibReport.describe} —
    #: `fit_stride` を使った事実や AWQ の α の範囲もここに出る）。
    calib: str | None = None


def teacher_forced(wrapper: nn.Module, case: GreedyCase) -> tuple[int, float]:
    """`cat(prompt, expected)` を 1 forward して `(一致数, NLL 平均)` を返す。

    読むのは位置 `T−1 … T+K−2` の K 行 —「prompt の末尾から expected を 1 つずつ与えたときの
    次トークン」で、自由走行と違い**発散しても列がずれない**ので config 間で同じ位置の劣化を
    比べられる。NLL は同じ K 行の `−log softmax(logits)[expected]` の平均で、1 位が変わる
    手前の劣化（順位は保つが確信が薄れる）まで拾う。
    """
    ids = torch.cat([case.prompt, case.expected.unsqueeze(0)], dim=1)
    with torch.no_grad():
        logits = wrapper(ids)
    start = int(case.prompt.shape[1]) - 1
    rows = logits[0, start : start + case.steps]
    matches = int((rows.argmax(dim=-1) == case.expected).sum())
    log_probs = torch.log_softmax(rows.to(torch.float32), dim=-1)
    nll = float(-log_probs.gather(1, case.expected.unsqueeze(1)).mean())
    return matches, nll


def greedy_prefix(wrapper: nn.Module, case: GreedyCase) -> int:
    """自由走行 greedy で expected と一致し続けた先頭 step 数（全一致なら K）。

    MUST: full re-forward（毎 step 先頭から標準 causal で計算 — `export_decode
    .greedy_continuation` と同じ流儀）。KV cache 経路で採ると、量子化の劣化と cache 実装の
    影響が混ざる。発散した時点で打ち切る — 以後の列は「別の文」なので比べる意味が無く、
    劣化した config ほど速く終わる。
    """
    current = case.prompt
    for step in range(case.steps):
        with torch.no_grad():
            logits = wrapper(current)
        token = int(logits[0, -1].argmax())
        if token != int(case.expected[step]):
            return step
        current = torch.cat([current, torch.tensor([[token]], dtype=torch.int64)], dim=1)
    return case.steps


def restore_pristine(
    weights: Mapping[str, torch.Tensor], pristine: Mapping[str, torch.Tensor]
) -> None:
    """全対象を pristine（素の f32）へ戻す。

    MUST: **config ごとに毎回**戻す。戻さずに当てると測っているのが「RTN の上の NF4」に
    なり、config 間の比較そのものが意味を失う（方式は積み重ねない）。戻す範囲は
    Phase 0 と方式グリッドの**和集合**（linear + `embed_tokens`）— 片方のグリッドが触った
    重みがもう片方の測定に残らないように。
    """
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])


def apply_config(weights: Mapping[str, torch.Tensor], config: SweepConfig) -> list[str]:
    """Phase 0 の fake-quant を当て、量子化した fqn を返す（復元は呼び出し側）。"""
    if config.group_size is None:
        return []
    quantize = fake_quant_asymmetric if config.asymmetric else fake_quant_symmetric
    touched: list[str] = []
    with torch.no_grad():
        for fqn, weight in weights.items():
            if family_of(fqn) == LM_HEAD and not config.include_lm_head:
                continue
            weight.copy_(quantize(weight, config.group_size))
            touched.append(fqn)
    return touched


def measure_rmse(
    weights: Mapping[str, torch.Tensor],
    pristine: Mapping[str, torch.Tensor],
    touched: Sequence[str],
) -> tuple[dict[str, float], float | None]:
    """丸め後の重みと pristine から族別 + 全体の相対 RMSE を採る。

    相対 RMSE は族内の全層をまとめた `‖w − fq‖₂ / ‖w‖₂`（要素数が約分されるので層の大きさで
    重み付けされない）。総和は f64 で採る — 族によっては 2 億要素を足すので f32 累算では
    桁落ちが数値の意味を食う。
    """
    errors: dict[str, float] = defaultdict(float)
    norms: dict[str, float] = defaultdict(float)
    with torch.no_grad():
        for fqn in touched:
            family = family_of(fqn)
            difference = weights[fqn] - pristine[fqn]
            errors[family] += float(difference.pow(2).sum(dtype=torch.float64))
            norms[family] += float(pristine[fqn].pow(2).sum(dtype=torch.float64))
    if not errors:
        return {}, None
    families = {family: math.sqrt(errors[family] / norms[family]) for family in errors}
    return families, math.sqrt(sum(errors.values()) / sum(norms.values()))


def assert_baseline_reproduces(
    cases: Sequence[GreedyCase], measured: Mapping[str, CaseResult]
) -> None:
    """baseline（f32 のまま）が期待列を teacher-forced で**完全に**再現することを見る。

    MUST: fail loudly。ここが割れるのは量子化の話ではなく**ハーネスが壊れている**合図
    （重みの取り違え・位置のずれ・期待列と模型の不一致）で、以後の全 config の数値の意味が
    消える。部分再実行（`--only`）でも baseline を必ず先に走らせるのはこの門のため。
    """
    wrong = {
        case.name: f"{measured[case.name].matches}/{case.steps}"
        for case in cases
        if measured[case.name].matches != case.steps
    }
    if wrong:
        raise AssertionError(
            f"baseline の teacher-forced 一致が完全でない {wrong}"
            "（量子化以前の問題 — 模型・期待列・位置の対応を疑う）"
        )


def select_configs(only: Sequence[str]) -> tuple[AnyConfig, ...]:
    """`--only` で選んだ config を宣言順で返す（**baseline は常に先頭で走る**）。

    baseline を外せないのは、部分再実行でも sanity 門（{@link assert_baseline_reproduces}）を
    必ず通すため — 60 分級の実行を刻んで回すときほど、「壊れた模型で測り続ける」経路を
    残さないことが効く。
    """
    if not only:
        return ALL_CONFIGS
    chosen = {BASELINE_NAME, *only}
    return tuple(config for config in ALL_CONFIGS if config.name in chosen)


def run_sweep(
    model_dir: Path,
    configs: Sequence[AnyConfig],
    *,
    decode_dir: Path = DEFAULT_DECODE_DIR,
    kmeans_shared_stride: int = 0,
    calib_limit: int | None = None,
    json_path: Path | None = None,
) -> list[ConfigResult]:
    """模型を 1 回だけ読み、config を順に当てて測る。

    pristine クローン（CPU・linear 169 本で ~3.5GB・`embed_tokens` 込みで ~4.3GB）を先に
    採るので、config ごとの丸めは常に元の重みから始まる。退避の範囲を**常に和集合**に
    するのは、Phase 0 の config が embedding を戻さない形を作らないため。

    校正付き構成の足場（{@link CalibRig}）は**最初に要求されたときに 1 回だけ**組む —
    pristine 復元は config ごとの先頭で済んでいるので、そこで採る先頭 stage の入力は素の
    重みのものになる。校正 config を 1 本も選ばなければ tokenizer も Catcher も走らない。

    進捗は 1 ケース / 1 config ごとに stderr へ即 flush し、`json_path` があれば config
    1 本ごとに書き直す（k-means の数十分級の実行が途中で落ちても測定値が残る）。
    """
    cases = load_cases(decode_dir)
    lengths = " ".join(f"{case.name}(T={case.prompt.shape[1]},K={case.steps})" for case in cases)
    print(f"[sweep] 期待列 {len(cases)} ケース: {lengths}", file=sys.stderr, flush=True)

    wrapper = one_shot.load_wrapper(model_dir)
    linear = linear_weights(wrapper)
    union, counts = collect_targets(wrapper, method_op_types(include_embedding=True))
    print(
        f"[sweep] 量子化対象 linear {len(linear)} 本 / 和集合 {counts.modules} 本"
        f"（{counts.elements:,} 要素）を pristine 退避",
        file=sys.stderr,
        flush=True,
    )
    pristine = {fqn: weight.detach().clone() for fqn, weight in union.items()}

    results: list[ConfigResult] = []
    rig: CalibRig | None = None
    for config in configs:
        # 前の config の一時領域（k-means は f64 で対象と同オーダー）を次の丸めの前に手放す。
        # 参照循環に載った GB 級のテンソルが世代 GC 待ちで残ると、次の config が実メモリを
        # 踏み抜く（refcount だけでは足りない場所があるので明示する）。
        gc.collect()
        started = time.perf_counter()
        restore_pristine(union, pristine)
        method: str | None = None
        size: SizeProjection | None = None
        calib: str | None = None
        if isinstance(config, SweepConfig):
            touched = apply_config(linear, config)
        elif isinstance(config, MethodConfig):
            method = config.method.name
            targets, target_counts = collect_targets(
                wrapper, method_op_types(config.include_embedding)
            )
            apply_method(wrapper, config, kmeans_shared_stride)
            touched = list(targets)
            size = project_size(config.method, target_counts)
        else:
            if rig is None:
                rig = build_calib_rig(wrapper, model_dir, calib_limit)
                print(
                    f"[sweep] 校正 {len(rig.batches)} 文 / stage {len(rig.stages)} 段 /"
                    f" 対象 linear {rig.counts.modules} 本"
                    "（decoder 内のみ — lm_head / embed_tokens は含まない）",
                    file=sys.stderr,
                    flush=True,
                )
            method = config.label
            report = apply_calib(config, rig, kmeans_shared_stride)
            assert_calib_covers_scan(report, rig.scan, config.name)
            touched = list(rig.scan)
            calib = report.describe()
            size = SizeProjection(
                counts=rig.counts,
                bits=config.projected_bits(rig.counts),
                formula=config.formula,
            )
            print(f"[{config.name}] {calib}", file=sys.stderr, flush=True)
        weight_rmse, overall = measure_rmse(union, pristine, touched)
        rounded = time.perf_counter() - started
        overall_text = "-" if overall is None else f"{overall:.4g}"
        print(
            f"[{config.name}] 丸め {len(touched)} 本 {rounded:.1f}s wRMSE {overall_text}",
            file=sys.stderr,
            flush=True,
        )
        measured: dict[str, CaseResult] = {}
        for case in cases:
            matches, nll = teacher_forced(wrapper, case)
            prefix = greedy_prefix(wrapper, case)
            measured[case.name] = CaseResult(
                steps=case.steps, matches=matches, nll=nll, prefix=prefix
            )
            print(
                f"[{config.name}] {case.name} teacher {matches}/{case.steps}"
                f" nll {nll:.4g} greedy {prefix}/{case.steps}",
                file=sys.stderr,
                flush=True,
            )
        if config.name == BASELINE_NAME:
            assert_baseline_reproduces(cases, measured)
        results.append(
            ConfigResult(
                name=config.name,
                weight_rmse=weight_rmse,
                cases=measured,
                seconds=time.perf_counter() - started,
                overall_rmse=overall,
                method=method,
                size=size,
                calib=calib,
            )
        )
        print(
            f"[{config.name}] 完了 {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if json_path is not None:
            json_path.write_text(
                results_json(results, cases, kmeans_shared_stride, calib_limit), encoding="utf-8"
            )
    return results


# ---- 表 ---------------------------------------------------------------------


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def quality_table(results: Sequence[ConfigResult]) -> str:
    """config × ケース別［teacher 一致 / NLL / 発散 step］の表。"""
    names = list(results[0].cases)
    header = ["config"]
    for name in names:
        header += [f"{name} 一致", f"{name} NLL", f"{name} 発散"]
    rows = []
    for result in results:
        row = [result.name]
        for name in names:
            case = result.cases[name]
            row += [
                f"{case.matches}/{case.steps}",
                f"{case.nll:.4g}",
                f"{case.prefix}/{case.steps}",
            ]
        rows.append(row)
    return _markdown(header, rows)


def rmse_table(results: Sequence[ConfigResult]) -> str:
    """config × 族別 weight 相対 RMSE の表（量子化しなかった族は `-`）。"""
    rows = [
        [result.name]
        + [
            f"{result.weight_rmse[family]:.4g}" if family in result.weight_rmse else "-"
            for family in METHOD_FAMILIES
        ]
        for result in results
    ]
    return _markdown(["config", *METHOD_FAMILIES], rows)


def summary_table(results: Sequence[ConfigResult]) -> str:
    """方式 × 対象 2 形を 1 行 1 config で読む要約表（ケース合算）。

    合算は「3 ケース × 16 位置 = 48 点の一致数」「NLL の 3 ケース単純和」「自由走行の
    接頭辞長の 3 ケース和」— ケース別の内訳は {@link quality_table} に残る。
    """
    header = ["config", "方式", "対象", "bpw", "投影 MiB", "wRMSE", "teacher", "NLL 和", "greedy"]
    rows = []
    for result in results:
        cases = list(result.cases.values())
        total = sum(case.steps for case in cases)
        rows.append(
            [
                result.name,
                result.method or "-",
                "-" if result.size is None else f"{result.size.counts.modules} 本",
                "-" if result.size is None else f"{result.size.bits_per_weight:.4g}",
                "-" if result.size is None else f"{result.size.projected_mib:.1f}",
                "-" if result.overall_rmse is None else f"{result.overall_rmse:.4g}",
                f"{sum(case.matches for case in cases)}/{total}",
                f"{sum(case.nll for case in cases):.4g}",
                f"{sum(case.prefix for case in cases)}/{total}",
            ]
        )
    return _markdown(header, rows)


def size_table(results: Sequence[ConfigResult]) -> str:
    """方式 × 対象集合のサイズ試算（**式による投影**・実測ではない）。"""
    header = ["config", "本数", "チャネル", "要素", "式", "bpw", "投影 MiB", "f32 MiB"]
    rows = [
        [
            result.name,
            f"{result.size.counts.modules}",
            f"{result.size.counts.channels:,}",
            f"{result.size.counts.elements:,}",
            result.size.formula,
            f"{result.size.bits_per_weight:.4g}",
            f"{result.size.projected_mib:.1f}",
            f"{result.size.f32_mib:.1f}",
        ]
        for result in results
        if result.size is not None
    ]
    return _markdown(header, rows)


def results_json(
    results: Sequence[ConfigResult],
    cases: Sequence[GreedyCase],
    kmeans_shared_stride: int,
    calib_limit: int | None = None,
) -> str:
    """機械可読の測定値（markdown 表の全列 + 表に出さない計数）。

    `calib_limit` は**校正が縮小実行だったか**の記録（`None` = 全量）— 縮小した数値を
    全量の数値として読まれないように、測定値と同じファイルへ残す。
    """
    payload = {
        "group_size": METHOD_GROUP_SIZE,
        "kmeans_shared_stride": kmeans_shared_stride,
        "calib_limit": calib_limit,
        "cases": [
            {"name": case.name, "prompt_length": int(case.prompt.shape[1]), "steps": case.steps}
            for case in cases
        ],
        "configs": [
            {
                "name": result.name,
                "method": result.method,
                "calib": result.calib,
                "seconds": round(result.seconds, 3),
                "weight_rmse": dict(result.weight_rmse),
                "overall_rmse": result.overall_rmse,
                "cases": {
                    name: {
                        "steps": case.steps,
                        "matches": case.matches,
                        "nll": case.nll,
                        "prefix": case.prefix,
                    }
                    for name, case in result.cases.items()
                },
                "size": None
                if result.size is None
                else {
                    "modules": result.size.counts.modules,
                    "channels": result.size.counts.channels,
                    "elements": result.size.counts.elements,
                    "formula": result.size.formula,
                    "bits_per_weight": result.size.bits_per_weight,
                    "projected_mib": result.size.projected_mib,
                    "f32_mib": result.size.f32_mib,
                },
            }
            for result in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--decode-dir", type=Path, default=DEFAULT_DECODE_DIR)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=[config.name for config in ALL_CONFIGS],
        help="この config だけ走らせる（複数可・部分再実行用）。baseline は常に先行する。",
    )
    parser.add_argument("--json", type=Path, default=None, help="機械可読の測定値の書き出し先。")
    parser.add_argument(
        "--kmeans-shared-stride",
        type=int,
        default=0,
        help="kmeans:shared の表の fit だけを等間隔 stride の部分標本で採る（0 / 1 = 全量）。"
        "全量が実メモリに載らない機体のための逃げ道で、丸めは常に全量へ当たる。",
    )
    parser.add_argument(
        "--calib-limit",
        type=int,
        default=None,
        help=f"校正コーパスの先頭 N 文だけを使う（既定は全 {len(CALIB_TEXTS)} 文）。"
        "縮小 smoke 用のノブで、校正付き構成にだけ効く。",
    )
    args = parser.parse_args(argv)
    results = run_sweep(
        args.model_dir,
        select_configs(args.only),
        decode_dir=args.decode_dir,
        kmeans_shared_stride=args.kmeans_shared_stride,
        calib_limit=args.calib_limit,
        json_path=args.json,
    )
    if args.kmeans_shared_stride > 1:
        print(f"> kmeans:shared の表は 1/{args.kmeans_shared_stride} 部分標本で fit（適用は全量）")
        print()
    if args.calib_limit is not None:
        print(
            f"> 校正コーパスは先頭 {args.calib_limit} 文のみ（全 {len(CALIB_TEXTS)} 文の縮小実行）"
        )
        print()
    print(summary_table(results))
    print()
    print(quality_table(results))
    print()
    print(rmse_table(results))
    print()
    print(size_table(results))
    notes = [f"> {result.name}: {result.calib}" for result in results if result.calib]
    if notes:
        print()
        print("校正付き構成の内訳（`CalibReport.describe`）:")
        print("\n".join(notes))


if __name__ == "__main__":
    main()
