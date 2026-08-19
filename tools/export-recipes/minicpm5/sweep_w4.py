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

config ごとに 4 列を採る（先の 3 列は Phase 0 と同じ・4 列目は方式グリッドのみ）:

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
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from _shared.decode_series import EXPECTED_KEY, GREEDY_PREFIX, GREEDY_SUFFIX, PROMPT_KEY
from karume.quant_methods import (
    DEFAULT_CODEBOOK_ITERATIONS,
    DEFAULT_CODEBOOK_LEVELS,
    fake_quant_fp4,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
    fit_codebook,
    group_absmax_scale,
    round_groups_to_levels,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QuantizeError,
    channel_rows,
    fake_quant_int4,
    group_size_of,
    iter_quant_targets,
    restore_channel_rows,
)
from minicpm5 import export as one_shot
from minicpm5 import export_decode as decode

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

#: `--only` が選べる全 config（宣言順 = 実行順）。
ALL_CONFIGS: tuple[SweepConfig | MethodConfig, ...] = (*SWEEP_CONFIGS, *METHOD_CONFIGS)


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


def fake_quant_kmeans_shared_strided(
    model: nn.Module, op_types: tuple[type[nn.Module], ...], group_size: int, stride: int
) -> int:
    """`kmeans:shared` の**表の fit だけ**を等間隔 stride の部分標本で採る版（適用は全量）。

    core の {@link karume.quant_methods.fake_quant_kmeans}(`shared`) は全対象の正規化値を
    1 本へ連結してから Lloyd を回すので、1B 級では f32 の連結 4GB に加えて `fit_codebook` の
    f64 一時領域（値・重み・添字）が 20GB 級になり、機体の実メモリで落ちる。標本を
    `flat[::stride]`（乱数を使わない = 同一入力 → ビット同一）に落とすと fit の一時領域だけが
    1/stride になり、**丸めそのものは全量**へ当たる。

    MUST: 使ったら出力へ明記する（`--kmeans-shared-stride` は JSON と markdown の脚注へ
    そのまま出る）— 部分標本で採った表と全量で採った表は別物になりうるので、数値を読む側が
    区別できないと困る。

    戻り値は丸めた本数（core の `MethodReport.modules` 相当）。
    """
    samples: list[torch.Tensor] = []
    with torch.no_grad():
        for fqn, weight, axis in iter_quant_targets(model, op_types):
            rows = channel_rows(weight, axis)
            scale = group_absmax_scale(rows, group_size, 1.0, f"'{fqn}'")
            grouped = grouped_view(rows, group_size_of(rows, scale))
            samples.append((grouped / scale.unsqueeze(-1)).reshape(-1)[::stride].clone())
        table = fit_codebook(
            torch.cat(samples).reshape(1, -1), DEFAULT_CODEBOOK_LEVELS, DEFAULT_CODEBOOK_ITERATIONS
        ).reshape(-1)
        del samples
        modules = 0
        for fqn, weight, axis in iter_quant_targets(model, op_types):
            rows = channel_rows(weight, axis)
            scale = group_absmax_scale(rows, group_size, 1.0, f"'{fqn}'")
            rounded = round_groups_to_levels(rows, table, scale, f"'{fqn}'")
            weight.copy_(restore_channel_rows(rounded, weight, axis))
            modules += 1
    return modules


def apply_method(model: nn.Module, config: MethodConfig, shared_stride: int) -> None:
    """方式グリッド 1 本の丸めを model へ in-place で当てる。

    `shared_stride > 1` のときだけ `kmeans:shared` が部分標本 fit の版
    （{@link fake_quant_kmeans_shared_strided}）へ替わる — 他の方式には効かない。
    """
    op_types = method_op_types(config.include_embedding)
    if config.method.name == f"{KMEANS_PREFIX}shared" and shared_stride > 1:
        fake_quant_kmeans_shared_strided(model, op_types, METHOD_GROUP_SIZE, shared_stride)
        return
    config.method.round_weights(model, op_types, METHOD_GROUP_SIZE)


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
    族ごとの要素数が 20 倍違うので平均だと重み付けが恣意的になる）。`size` は方式グリッド
    だけが持つ（Phase 0 の config は格納形が i4 の 1 通りで、試算しても同じ数が並ぶ）。
    """

    name: str
    weight_rmse: Mapping[str, float]
    cases: Mapping[str, CaseResult]
    seconds: float = 0.0
    overall_rmse: float | None = None
    method: QuantMethod | None = None
    size: SizeProjection | None = None


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


def select_configs(only: Sequence[str]) -> tuple[SweepConfig | MethodConfig, ...]:
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
    configs: Sequence[SweepConfig | MethodConfig],
    *,
    decode_dir: Path = DEFAULT_DECODE_DIR,
    kmeans_shared_stride: int = 0,
    json_path: Path | None = None,
) -> list[ConfigResult]:
    """模型を 1 回だけ読み、config を順に当てて測る。

    pristine クローン（CPU・linear 169 本で ~3.5GB・`embed_tokens` 込みで ~4.3GB）を先に
    採るので、config ごとの丸めは常に元の重みから始まる。退避の範囲を**常に和集合**に
    するのは、Phase 0 の config が embedding を戻さない形を作らないため。

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
    for config in configs:
        # 前の config の一時領域（k-means は f64 で対象と同オーダー）を次の丸めの前に手放す。
        # 参照循環に載った GB 級のテンソルが世代 GC 待ちで残ると、次の config が実メモリを
        # 踏み抜く（refcount だけでは足りない場所があるので明示する）。
        gc.collect()
        started = time.perf_counter()
        restore_pristine(union, pristine)
        method: QuantMethod | None = None
        size: SizeProjection | None = None
        if isinstance(config, SweepConfig):
            touched = apply_config(linear, config)
        else:
            method = config.method
            targets, target_counts = collect_targets(
                wrapper, method_op_types(config.include_embedding)
            )
            apply_method(wrapper, config, kmeans_shared_stride)
            touched = list(targets)
            size = project_size(method, target_counts)
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
            )
        )
        print(
            f"[{config.name}] 完了 {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if json_path is not None:
            json_path.write_text(
                results_json(results, cases, kmeans_shared_stride), encoding="utf-8"
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
                result.method.name if result.method else "-",
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
    results: Sequence[ConfigResult], cases: Sequence[GreedyCase], kmeans_shared_stride: int
) -> str:
    """機械可読の測定値（markdown 表の全列 + 表に出さない計数）。"""
    payload = {
        "group_size": METHOD_GROUP_SIZE,
        "kmeans_shared_stride": kmeans_shared_stride,
        "cases": [
            {"name": case.name, "prompt_length": int(case.prompt.shape[1]), "steps": case.steps}
            for case in cases
        ],
        "configs": [
            {
                "name": result.name,
                "method": result.method.name if result.method else None,
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
    args = parser.parse_args(argv)
    results = run_sweep(
        args.model_dir,
        select_configs(args.only),
        decode_dir=args.decode_dir,
        kmeans_shared_stride=args.kmeans_shared_stride,
        json_path=args.json,
    )
    if args.kmeans_shared_stride > 1:
        print(f"> kmeans:shared の表は 1/{args.kmeans_shared_stride} 部分標本で fit（適用は全量）")
        print()
    print(summary_table(results))
    print()
    print(quality_table(results))
    print()
    print(rmse_table(results))
    print()
    print(size_table(results))


if __name__ == "__main__":
    main()
