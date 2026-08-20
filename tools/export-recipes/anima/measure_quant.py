"""w8 / w8a8 / attention a8 の**量子化品質**を torch CPU 上で測る（Q0 の品質ゲート）。

GPU コードは 0 行。ADR 0006 の fake-quant 方法論では E2E は**実装誤差しか測らない**ので、
量子化そのものの質は別軸で測る必要がある（ADR 0019 の品質ゲート①）。この台本はその軸を
Anima DiT で埋める。

10 構成を**同一プロセス・同一 seed**で走らせて直交分解する（{@link CONFIGS}）:

    (a) f32               基準（LoRA 焼き込み済みの素の重み）
    (b) w8                重みのみ per-channel symmetric i8 で fake-quant（`fake_quant_int8`）
    (c) w8a8              (b) + **適格 linear の入力活性**を per-token symmetric i8 で fake-quant
    (d) w8a8-qk           (c) + attention の **q / k のみ** i8
    (e) w8a8-pv           (c) + attention の **P̃ / V のみ** i8
    (f) w8a8-qkpv         (c) + attention の q/k/P̃/V を i8（attention a8 の本命）
    (g) w8-qkpv           (b) + attention のみ i8（linear 活性は f32 運用のときの単独寄与）
    (i) w8a8-qkpv-s16     (f) + **attention スコア S を f16 で格納**（案γ 波 1 の実運用形）
    (j) w8a8-s16          (c) + S の f16 格納のみ（S 丸め単独の寄与を分離する）
    (h) w8a8-qkpv-statsq  (f) の softmax 分母を**量子化後総和** `Σ round(127·P̃)/127` に置換

記号は**外部参照の安定のため据え置き**（ADR 0030 と ACTIVE_DESIGN が「構成 (h)」を名指しで
参照している）。実行順は上の並びどおりで、(h) は末尾のまま — (h) は波 Q0 で**不採用確定**なので、
`--until w8a8-qkpv-s16` が案γ の門に必要な最小の前方接頭辞になる（{@link CONFIGS} のコメント）。

(c) の活性量子化は `docs/research/2026-08-03-dp4a-w8a8-design.md` §4.2 の数値仕様の鏡像:

    s = clamp(rowmax(|x|) / 127, f32 tiny)      … 行 = 最終軸（per-token）
    q = clamp(round(x / s), -127, +127)         … **±127 に閉じる**（−128 不使用）
    x̂ = q · s

`torch.round` は偶数丸め。全ゼロ行は `s = tiny` → `q = 0` → `x̂ = 0` で厳密。
適格判定は同 doc §4.1 と同じ「DiT 内の `nn.Linear`（bias 有無を問わない）で `k % 4 == 0`」。

(d)〜(h) の attention 量子化の粒度は `docs/research/2026-08-04-attention-a8-design.md` §5.1 の
粒度 MUST 表（= ランタイムが実装できる粒度と厳密一致）。詳細は {@link attention_sim}。
(i)(j) の S f16 格納は `docs/research/2026-08-04-intermediate-f16-design.md` §3.4② / §7 波 1 の
実装形と粒度厳密一致（{@link round_scores_f16}）。

MUST: (a)→(b)→… の順に**同じモデルインスタンス**を使う（fake_quant_int8 は in-place で
f32 重みを潰すので順序が逆にできない）。7.29GiB のロードが 1 回で済む副次効果もある。

出力（`--out`、既定はこのファイルの隣ではなくスクラッチ側を明示指定する運用）:

    report.md              数値表（step 別 relRMS / PSNR / 層別 relRMS / P̃ 分布 / 門の判定）
    attn.json              attention 量子化の集計（機械可読 — 門の判定値もここ）
    image_<config>.png     最終画像（PIL が無ければ image_*.npy へフォールバック）
    dit_out.safetensors    全構成の step 別 latent（生データ — 追加の分析用）
    layers.csv             層別の生表（454 本ぶん — report.md には要約だけ載る）

    uv run python -m anima.measure_quant --out /path/to/q0

`--w4-screen` は**独立のモード**（10 構成は 1 本も走らない — {@link run_w4_screen}）。
w4 の丸め方式（rtn-i4-g32 / nf4 / mxfp4 / kmeans:shared）と**校正付き丸め**（gptq-rtn /
gptq-nf4 / gptq-kmeans — 波 J-2）を横並びにする側で、上の 10 構成の**積み上げ意味論とは
別経路**: あちらは同じインスタンスへ in-place で積み上げるが、方式比較は構成ごとに pristine
（素の f32 重み）へ戻してから当てる（積み重ね禁止 MUST）。記号・基準・出力ファイル名も
共有しない。校正付きの対象は **DiT block 列の linear 限定**で 4 方式とは集合が違う
（{@link CALIB_TARGET}）。

NOTE: 非 DiT（text_encoder / text_conditioner / VAE）は全構成とも **f32 のまま**にする。
比較軸を DiT の量子化 1 本に絞るため（資産系列 `outputs/series/anima-i8/` は他 3 つが f16 だが、
その差は全構成に同じ形で乗るので相対比較では相殺される）。attention の差し替えも
**DiT の denoise ループの間だけ**掛かる（text encode / VAE decode は素通し — 復元は
`PRISTINE_SDPA` 検査で機械確認する）。

NOTE: レポート生成（`report.md` / `attn.json` / `layers.csv` の組み立て）は
`anima/measure_report.py` に分けてある — 数値仕様とは変更頻度が違うため。
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from karume.act_quant import (
    attach_act_quant,
    detach_act_quant,
    is_eligible,
    quantize_rows,
    quantize_rows_parts,
)
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
    INT8_MAX,
    QUANT_MODULE_TYPES,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
)

from .measure_report import build_report, build_summary, write_layer_csv
from .pipeline_ref import (
    LATENT_CHANNELS,
    PROMPT,
    SEED,
    SHIFT,
    SPATIAL_COMPRESSION,
    denormalize_latents,
    encode_text,
    reference_steps,
    sigma_schedule,
    timesteps_proj_table,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
DEFAULT_LORA = REPO_ROOT / "inputs" / "anima" / "anima-turbo-lora-v0.2.safetensors"

#: turbo 運用値（CFG=1 は uncond 分岐を計算しない — anima/pipeline_ref.py の MUST と同じ形）。
GUIDANCE = 1.0

#: 差し替え前の SDPA 実体（復元検査の正本 — import 時に 1 度だけ捕まえる）。
PRISTINE_SDPA = F.scaled_dot_product_attention

TINY = torch.finfo(torch.float32).tiny


@dataclass(frozen=True)
class Config:
    """1 構成のレシピ。

    `weight_i8` は「この構成に入る**手前**で重みを i8 へ落とす」フラグ（`fake_quant_int8` は
    in-place なので、表の順序がそのまま適用順になる）。`attn` は attention の量子化範囲。
    """

    name: str
    weight_i8: bool
    linear_act: bool
    #: `None` / `"qk"` / `"pv"` / `"qkpv"`。
    attn: str | None = None
    #: softmax の分母を `Σ round(127·P̃)/127`（量子化後総和）にする（設計 doc §4.3）。
    stats_quantized: bool = False
    #: attention スコア S を **f16 で格納**する（案γ 波 1 — {@link round_scores_f16}）。
    score_f16: bool = False
    #: P̃ の行 peak/rms 分布を採る（量子化していない素の P̃ を見る構成でだけ true）。
    measure_pt: bool = False


#: 実行順（重みの丸めは in-place なので **順序が意味を持つ** MUST）。
#:
#: s16 の 2 本は `w8-qkpv` の直後・`w8a8-qkpv-statsq` の手前に置く。重みは (b) の 1 回きりで
#: in-place に i8 へ落ちるので、(b) 以降はどこに挿しても**重み状態は同じ**（挿入位置が決めるのは
#: `--until` の前方接頭辞に何が入るかだけ）。門の対象 `w8a8-qkpv-s16` を先に置いてあるので
#: `--until w8a8-qkpv-s16` が門に必要な最小の接頭辞になる: 集計に要る (a)(b)(c) と、
#: 「S f16 が attn-a8 へ上乗せする劣化」の直接の対照 (f) までが入り、**波 Q0 で不採用が確定した
#: (h) statsq だけが外れる**。S 丸め単独の寄与まで見るなら `--until w8a8-s16`。
CONFIGS: tuple[Config, ...] = (
    Config("f32", weight_i8=False, linear_act=False),
    Config("w8", weight_i8=True, linear_act=False),
    Config("w8a8", weight_i8=False, linear_act=True, measure_pt=True),
    Config("w8a8-qk", weight_i8=False, linear_act=True, attn="qk"),
    Config("w8a8-pv", weight_i8=False, linear_act=True, attn="pv"),
    Config("w8a8-qkpv", weight_i8=False, linear_act=True, attn="qkpv"),
    Config("w8-qkpv", weight_i8=False, linear_act=False, attn="qkpv"),
    Config("w8a8-qkpv-s16", weight_i8=False, linear_act=True, attn="qkpv", score_f16=True),
    Config("w8a8-s16", weight_i8=False, linear_act=True, score_f16=True),
    Config("w8a8-qkpv-statsq", weight_i8=False, linear_act=True, attn="qkpv", stats_quantized=True),
)


def attn_config_names(configs: tuple[Config, ...]) -> tuple[str, ...]:
    """**attention を触る**構成（門の判定対象）。

    a8 量子化（`attn`）と S の f16 格納（`score_f16`）はどちらも attention 側の変更なので、
    門・②の PSNR 対・⑤の層別表は同じ集合を見る。
    """
    return tuple(config.name for config in configs if config.attn or config.score_f16)


CONFIG_NAMES = tuple(config.name for config in CONFIGS)
#: 相対劣化の基準（f32 比 PSNR は w8a8 で既に飽和しているため — 設計 doc §5.3）。
BASELINE = "w8a8"
ATTN_CONFIGS = attn_config_names(CONFIGS)


# ---- 活性の per-token fake-quant -------------------------------------------
#
# 数値仕様（`quantize_rows`）と適格判定（`is_eligible`）の正本は
# `karume.act_quant`（`anima/pipeline_ref.py` の鏡像フィクスチャと共有 — 2 箇所に
# 書き写すと片方だけが仕様から外れる）。ここはその上に**層別の誤差集計**を足したフック。


@dataclass
class LayerStat:
    """1 層ぶんの誤差集計（全 step・全呼び出しを通算した二乗和で持つ）。"""

    in_features: int
    out_features: int
    calls: int = 0
    #: 入力活性の量子化誤差（Σ(x̂−x)² と Σx²）。
    in_err2: float = 0.0
    in_ref2: float = 0.0
    #: 出力の差（w8a8 − w8）。`--measure-step` の step だけ実測する（linear がもう 1 回走る）。
    out_err2: float = 0.0
    out_ref2: float = 0.0
    out_calls: int = 0
    #: 行ごとの amax/rms（外れ値の指標 — 大きいほど i8 の格子が無駄になる）。
    peak_sum: float = 0.0
    peak_rows: int = 0
    peak_max: float = 0.0

    def in_rel_rms(self) -> float:
        return math.sqrt(self.in_err2 / self.in_ref2) if self.in_ref2 > 0 else 0.0

    def out_rel_rms(self) -> float:
        return math.sqrt(self.out_err2 / self.out_ref2) if self.out_ref2 > 0 else 0.0

    def peak_mean(self) -> float:
        return self.peak_sum / self.peak_rows if self.peak_rows else 0.0


@dataclass
class ActQuantHooks:
    """適格 linear の入力を per-token i8 へ落とし、層別の誤差を集計するフック束。

    `measure_step` の step でだけ「量子化しなかった場合の出力」を追加で計算して層別の
    出力 relRMS を採る（w8 の重みは既に当たっているので、この差は**活性量子化だけ**の寄与）。
    """

    model: nn.Module
    measure_step: int
    stats: dict[str, LayerStat] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    step: int = -1
    _pending: dict[str, torch.Tensor] = field(default_factory=dict)

    def attach(self) -> None:
        self.handles.append(self.model.register_forward_pre_hook(self._on_model))
        for name, module in self.model.named_modules():
            if not is_eligible(module):
                continue
            self.stats[name] = LayerStat(module.in_features, module.out_features)
            self.handles.append(
                module.register_forward_pre_hook(self._make_pre(name), with_kwargs=False)
            )
            self.handles.append(module.register_forward_hook(self._make_post(name)))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _on_model(self, _module: nn.Module, _args: tuple[object, ...]) -> None:
        # CFG=1 なので DiT の呼び出し 1 回 = 1 step。
        self.step += 1

    def _make_pre(self, name: str):
        def pre(_module: nn.Module, args: tuple[torch.Tensor, ...]):
            if not args:
                # 位置引数で呼ばれない linear は活性量子化の対象にできない（本数が減れば
                # `[eligible]` の計数と `stat.calls == 0` で見える）。
                return None
            x = args[0]
            xq = quantize_rows(x)
            stat = self.stats[name]
            stat.calls += 1
            diff = xq - x
            stat.in_err2 += float(diff.square().sum())
            stat.in_ref2 += float(x.square().sum())
            flat = x.reshape(-1, x.shape[-1])
            peak = flat.abs().amax(dim=-1) / flat.square().mean(dim=-1).sqrt().clamp_min(
                torch.finfo(torch.float32).tiny
            )
            stat.peak_sum += float(peak.sum())
            stat.peak_rows += peak.numel()
            stat.peak_max = max(stat.peak_max, float(peak.amax()))
            if self.step == self.measure_step:
                self._pending[name] = x
            return (xq, *args[1:])

        return pre

    def _make_post(self, name: str):
        def post(module: nn.Module, _args: tuple[object, ...], output: torch.Tensor):
            x = self._pending.pop(name, None)
            if x is None:
                return None
            reference = F.linear(x, module.weight, module.bias)
            stat = self.stats[name]
            stat.out_err2 += float((output - reference).square().sum())
            stat.out_ref2 += float(reference.square().sum())
            stat.out_calls += 1
            return None

        return post


# ---- attention の a8 シム ----------------------------------------------------
#
# 設計 = `docs/research/2026-08-04-attention-a8-design.md`。粒度は同 §5.1 の MUST 表で、
# **ランタイムが実装できる粒度と厳密一致**させる（緩めると「実装不能な楽観上界」になる）:
#
#   q     [B·H·M, D] の行ごと amax（縮約軸 D が最内連続なので `quantize_rows` そのもの）
#   k     [B·H·N, D] の行ごと amax（同上）
#   P̃     **scale 固定 1/127**（`q_P = round(127·exp(S−m))`）。行内 max が構造的に 1.0 なので
#         per-token amax を取っても 1/127 にしかならない = 適応の余地がゼロ（§2.2）。
#         **amax を取り直してはならない**（「取れる」実装だと誤読されるため式にも書かない）。
#   V     `(b,h,d)` ごとに **N 全体**の amax（縮約軸 n に依存しない scale = 括り出せる形）。
#         per-token（行 n ごと）はランタイムに実装不能な楽観上界なので**禁止**（§2.3）。
#   S     **f16 格納**（`score_f16` 構成 — 案γ 設計 doc §3.4② / §7 波 1）。①QK が書く
#         「半スケール適用済みの生スコア」を f16 へ落として戻し、**丸めはその 1 回だけ**。
#         ② 行 max `m` と分母 `l = Σexp(S−m)`・③ の `exp(S−m)` と `qP = round(127·exp(S−m))` は
#         **全て同じ丸め済み S** から f32 で計算する（ランタイムの ②③ が同じ f16 バッファを
#         読んで `f32()` へ広げるのと厳密に同じ粒度 — attention.ts の `read()` と
#         attention-i8a8.ts の A タイル充填）。
#
# 半スケール（√scale_factor）は **dequant 側**へ 2 回掛ける（量子化の前に掛けると丸めが 1 段
# ずれ、`quantize_rows` を無改変で使えなくもなる — §2.1 の裁定）。
#
# 整数縮約は float64 で厳密に行う（|acc| ≤ D·127² も N·127² も 2^53 に対し桁で余裕があるので
# 加算順に依らず厳密）。目的は品質測定なので f32 の ULP 級の丸め差までは再現しない。

#: P̃ 行 peak/rms のヒストグラム（分位点を有界メモリで採る）。peak/rms ≤ √N なので log₂ 上限 8
#: は N ≤ 65,536 まで足りる。分解能は 0.01 log₂ ≈ 0.7%。
PT_HIST_BINS = 800
PT_HIST_LOG_MAX = 8.0


@dataclass
class AttnStat:
    """1 attention ノードぶんの集計（全 step 通算）。"""

    calls: int = 0
    #: P̃ の行 peak/rms（= 1/rms — peak は構造的に 1.0）のヒストグラムと生の最大値。
    pt_hist: torch.Tensor = field(
        default_factory=lambda: torch.zeros(PT_HIST_BINS, dtype=torch.int64)
    )
    pt_rows: int = 0
    pt_max: float = 0.0
    pt_sum: float = 0.0
    #: S / O が「attention を量子化しなかった場合」から動いた量（`measure_step` のみ実測）。
    s_err2: float = 0.0
    s_ref2: float = 0.0
    o_err2: float = 0.0
    o_ref2: float = 0.0
    measured: int = 0

    def s_rel_rms(self) -> float:
        return math.sqrt(self.s_err2 / self.s_ref2) if self.s_ref2 > 0 else 0.0

    def o_rel_rms(self) -> float:
        return math.sqrt(self.o_err2 / self.o_ref2) if self.o_ref2 > 0 else 0.0

    def pt_mean(self) -> float:
        return self.pt_sum / self.pt_rows if self.pt_rows else 0.0

    def pt_quantile(self, q: float) -> float:
        return hist_quantile(self.pt_hist, q)


def hist_add(counts: torch.Tensor, values: torch.Tensor) -> None:
    """log₂ 等間隔ビンへ加算（値は 1 以上 — peak/rms は定義から 1 を下回らない）。"""
    scaled = torch.log2(values.reshape(-1).clamp_min(1.0)) * (PT_HIST_BINS / PT_HIST_LOG_MAX)
    index = scaled.to(torch.int64).clamp_(0, PT_HIST_BINS - 1)
    counts += torch.bincount(index, minlength=PT_HIST_BINS)


def hist_quantile(counts: torch.Tensor, q: float) -> float:
    """ヒストグラムの分位点（ビン内は log₂ 上で線形補間）。"""
    total = int(counts.sum())
    if total == 0:
        return 0.0
    cumulative = counts.cumsum(0)
    target = q * total
    index = int(torch.searchsorted(cumulative, torch.tensor(target)).clamp_(0, PT_HIST_BINS - 1))
    below = float(cumulative[index - 1]) if index else 0.0
    width = PT_HIST_LOG_MAX / PT_HIST_BINS
    fraction = (target - below) / float(counts[index]) if counts[index] > 0 else 0.0
    return float(2.0 ** ((index + min(max(fraction, 0.0), 1.0)) * width))


def peak_over_rms(pt: torch.Tensor) -> torch.Tensor:
    """P̃ の行ごとの `peak/rms`。peak は `exp(0) = 1` で構造的に 1.0 なので `1/rms`。"""
    mean_square = pt.square().sum(dim=-1, dtype=torch.float64) / pt.shape[-1]
    return mean_square.clamp_min(TINY).rsqrt()


@dataclass
class ScoreF16Probe:
    """S の f16 丸めが**実際に発火した**ことの機械検査（恒真化の防止）。

    「丸めを掛けたつもりで恒等だった」は数値にも所要時間にも出ない（ADR 0024 追補 /
    0030 決定 5 と同型の沈黙）。丸め**前後で 1 要素も変わらない呼び出しがゼロ**であることを
    毎回見て、集計を門（{@link build_summary}）へ出す。
    """

    calls: int = 0
    fired: int = 0
    elements: int = 0
    #: 素の S の |最大値|（f16 の上限 65,504 に対する余裕 — 波 1 の格納形が飽和しない裏取り）。
    abs_max: float = 0.0

    def observe(self, before: torch.Tensor, after: torch.Tensor) -> None:
        self.calls += 1
        self.elements += before.numel()
        # `abs()` の巨大な一時を作らずに |max| を採る（S は 1024px で 2.7 億要素 / ノード）。
        self.abs_max = max(self.abs_max, float(before.max()), -float(before.min()))
        # `torch.equal` は短絡評価なので bool の一時テンソルを作らない。
        if not torch.equal(after, before):
            self.fired += 1


def round_scores_f16(scores: torch.Tensor, probe: ScoreF16Probe | None = None) -> torch.Tensor:
    """S を f16 へ落として f32 へ戻す（案γ 波 1 の「S の格納形」の鏡像）。

    MUST: 呼び出しは attention 1 回につき**ここ 1 箇所だけ**。②行統計も ③PV も、戻り値の
    丸め済み S から計算する（ランタイムは同じ f16 バッファを 2 度読むので、2 度丸めた形は
    実装できない上界になる）。
    """
    rounded = scores.half().float()
    if probe is not None:
        probe.observe(scores, rounded)
    return rounded


def flatten_heads(x: torch.Tensor) -> torch.Tensor:
    """`[B,H,S,D]` → `[B·H,S,D]`（scale の行/列束縛はこの平坦形の上で定義される）。"""
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])


def attention_scores(
    query: torch.Tensor, key: torch.Tensor, scale: float, *, quant_qk: bool
) -> torch.Tensor:
    """`S = (q·√sf)·(k·√sf)ᵀ` を `[B·H,M,N]` で返す。"""
    q, k = flatten_heads(query), flatten_heads(key)
    if not quant_qk:
        return torch.matmul(q, k.transpose(-1, -2)) * scale
    # 半スケールは dequant 側（`qs·√sf` と `ks·√sf`）— 量子化の前には掛けない MUST。
    half = math.sqrt(scale)
    qq, qs = quantize_rows_parts(q)
    kq, ks = quantize_rows_parts(k)
    accumulator = qq.double() @ kq.double().transpose(-1, -2)
    row_scale = (qs.double() * half) * (ks.double() * half).transpose(-1, -2)
    return (accumulator * row_scale).to(torch.float32)


def attention_sim(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    quant_qk: bool,
    quant_pv: bool,
    stats_quantized: bool,
    score_f16: bool,
    scale: float,
    probe: ScoreF16Probe | None = None,
) -> torch.Tensor:
    """`[B,H,M,D]` レイアウトの SDPA を、指定範囲だけ i8 粒度で置き換えて評価する。"""
    v = flatten_heads(value)
    scores = attention_scores(query, key, scale, quant_qk=quant_qk)
    if score_f16:
        # 格納の丸めはここ 1 回だけ（以降の m / Σ / exp / qP は全て丸め済み S から）。
        scores = round_scores_f16(scores, probe)
    exponent = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    if not quant_pv:
        probabilities = exponent / exponent.sum(dim=-1, keepdim=True)
        return torch.matmul(probabilities, v).reshape(query.shape)

    # P̃ の scale は 1/127 固定（clamp 不要 — 上限 127 は `exp(0)=1` から構造的）。
    quantized_pt = torch.round(exponent * INT8_MAX)
    if stats_quantized:
        total = quantized_pt.sum(dim=-1, keepdim=True, dtype=torch.float32) / INT8_MAX
    else:
        total = exponent.sum(dim=-1, keepdim=True)
    # V は (b,h,d) ごとに N 全体の amax（縮約軸 n に依存しない = 括り出せる）。
    value_scale = torch.clamp(v.abs().amax(dim=-2, keepdim=True) / INT8_MAX, min=TINY)
    quantized_v = torch.round(v / value_scale).clamp_(-INT8_MAX, INT8_MAX)
    accumulator = quantized_pt.double() @ quantized_v.double()
    row_inv = (1.0 / total).double() * (1.0 / INT8_MAX)
    return (accumulator * row_inv * value_scale.double()).to(torch.float32).reshape(query.shape)


@dataclass
class AttnSim:
    """`F.scaled_dot_product_attention` を **op 粒度**で差し替える（モジュールフック禁止）。

    差し替えが op 粒度でなければならない理由は ADR 0029 の教訓そのもの: diffusers の
    `CosmosAttnProcessor2_*` は `attn.processor` の中から
    `attention_dispatch._native_attention` 経由で `F.scaled_dot_product_attention` を
    **キーワード引数で**直に呼ぶ（`nn.Module` として存在しない）。モジュールフックでは
    1 度も発火しないまま「量子化しているつもりで劣化が軽く出る」向きに沈黙する。

    帰属のためだけに DiT 配下の全モジュールへ push/pop フックを掛けて名前スタックを持つ
    （`measure_quant_sbv2.ActQuant` と同じ仕掛け）。スタック先頭が attention モジュール名になる。

    スコープは **DiT のみ**: `attach` / `detach` を denoise ループの前後だけで呼ぶので、
    text encode（`run_dit` の冒頭）と VAE decode（`decode_all`）は素通しになる。
    """

    model: nn.Module
    mode: str | None
    stats_quantized: bool
    #: S を f16 で格納する（案γ 波 1 — a8 量子化とは独立の軸）。
    score_f16: bool
    measure_pt: bool
    measure_step: int
    #: 故障注入（`--inject drop-attn-quant`）: 帰属の足場だけ作り、量子化も S の丸めも掛けない。
    disable: bool = False
    #: S f16 丸めの発火検査（恒真化の防止 — 構成を通した集計）。
    s16: ScoreF16Probe = field(default_factory=ScoreF16Probe)
    stats: dict[str, AttnStat] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    step: int = -1
    calls: int = 0
    skipped: int = 0
    patched: bool = False

    def attach(self) -> None:
        self.handles.append(self.model.register_forward_pre_hook(self._on_model))
        for name, module in self.model.named_modules():
            self.handles.append(module.register_forward_pre_hook(self._make_push(name)))
            self.handles.append(module.register_forward_hook(self._pop))
        F.scaled_dot_product_attention = self._wrapper
        self.patched = True

    def detach(self) -> None:
        if self.patched:
            F.scaled_dot_product_attention = PRISTINE_SDPA
            self.patched = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stack.clear()

    def _on_model(self, _module: nn.Module, _args: tuple[object, ...]) -> None:
        # CFG=1 なので DiT の呼び出し 1 回 = 1 step。
        self.step += 1

    def _make_push(self, name: str):
        def push(_module: nn.Module, _args: tuple[object, ...]) -> None:
            self.stack.append(name)

        return push

    def _pop(self, _module: nn.Module, _args: tuple[object, ...], _output: object) -> None:
        self.stack.pop()

    def _wrapper(self, *args: object, **kwargs: object) -> torch.Tensor:
        query = kwargs.get("query", args[0] if len(args) > 0 else None)
        key = kwargs.get("key", args[1] if len(args) > 1 else None)
        value = kwargs.get("value", args[2] if len(args) > 2 else None)
        if not isinstance(query, torch.Tensor) or query.dim() != 4:
            # 想定外の呼ばれ方は黙って素通しにせず数える（0 でなければ報告に出る）。
            self.skipped += 1
            return PRISTINE_SDPA(*args, **kwargs)
        for name in ("attn_mask", "is_causal", "dropout_p", "enable_gqa"):
            given = kwargs.get(name)
            if given not in (None, False, 0.0, 0):
                raise NotImplementedError(f"attention シムが想定しない SDPA 引数: {name}={given}")
        scale = kwargs.get("scale")
        scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)

        node = self.stack[-1] if self.stack else "<root>"
        stat = self.stats.setdefault(node, AttnStat())
        stat.calls += 1
        self.calls += 1

        if self.measure_pt:
            # (c) の走行中に採る = 量子化器が見る**素の P̃**（attention は f32 のまま）。
            scores = attention_scores(query, key, scale, quant_qk=False)
            ratio = peak_over_rms(torch.exp(scores - scores.amax(dim=-1, keepdim=True)))
            hist_add(stat.pt_hist, ratio)
            stat.pt_rows += ratio.numel()
            stat.pt_sum += float(ratio.sum())
            stat.pt_max = max(stat.pt_max, float(ratio.amax()))
            del scores, ratio

        if self.disable or (self.mode is None and not self.score_f16):
            return PRISTINE_SDPA(*args, **kwargs)

        quant_qk = self.mode is not None and "qk" in self.mode
        quant_pv = self.mode is not None and "pv" in self.mode
        output = attention_sim(
            query,
            key,
            value,
            quant_qk=quant_qk,
            quant_pv=quant_pv,
            stats_quantized=self.stats_quantized,
            score_f16=self.score_f16,
            scale=scale,
            probe=self.s16,
        )
        if self.step == self.measure_step:
            # 「attention を量子化しなかった場合」との差 = attention 量子化だけの寄与
            # （linear の活性量子化は q/k/v の中に既に乗っている）。
            reference = attention_sim(
                query,
                key,
                value,
                quant_qk=False,
                quant_pv=False,
                stats_quantized=False,
                score_f16=False,
                scale=scale,
            )
            scores_ref = attention_scores(query, key, scale, quant_qk=False)
            scores_value = (
                attention_scores(query, key, scale, quant_qk=True) if quant_qk else scores_ref
            )
            if self.score_f16:
                # S 側の誤差にも格納の丸めを含める（probe は渡さない — 本走行で計上済み）。
                scores_value = round_scores_f16(scores_value)
            stat.s_err2 += float((scores_value - scores_ref).square().sum())
            stat.s_ref2 += float(scores_ref.square().sum())
            stat.o_err2 += float((output - reference).square().sum())
            stat.o_ref2 += float(reference.square().sum())
            stat.measured += 1
        return output


def count_attention_nodes(model: nn.Module) -> list[str]:
    """DiT 内の attention モジュール名（発火計数の期待値の出どころ）。"""
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "to_q") and hasattr(module, "to_out")
    ]


# ---- 指標 -------------------------------------------------------------------


def rel_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    """‖v − r‖₂ / ‖r‖₂。"""
    denom = float(reference.square().sum())
    if denom == 0.0:
        return 0.0
    return math.sqrt(float((value - reference).square().sum()) / denom)


def psnr(value: torch.Tensor, reference: torch.Tensor, data_range: float) -> float:
    mse = float((value - reference).square().mean())
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(data_range * data_range / mse)


def to_uint8(image: torch.Tensor) -> torch.Tensor:
    """[-1,1] の [1,3,H,W] を [H,W,3] の uint8 へ（PNG と目視 PSNR の共通経路）。"""
    scaled = (image[0].permute(1, 2, 0).clamp(-1.0, 1.0) + 1.0) * 127.5
    return scaled.round().clamp(0.0, 255.0).to(torch.uint8)


# ---- 段 ---------------------------------------------------------------------


@dataclass
class DitRun:
    """`run_dit` の成果（構成横断の生データと診断）。"""

    latents: dict[str, torch.Tensor]
    #: (c) w8a8 の適格 linear 層別統計（既存の ③ 表の出どころ）。
    linear_stats: dict[str, LayerStat]
    #: 構成名 → attention ノード別統計。
    attn_stats: dict[str, dict[str, AttnStat]]
    #: 構成名 → 発火計数などの診断。
    diagnostics: dict[str, dict[str, object]]
    attention_nodes: list[str]


def run_dit(args: argparse.Namespace) -> DitRun:
    """全構成の DiT を同一インスタンスで順に走らせ、step 別 latent と診断を返す。"""
    sigmas = sigma_schedule(args.steps, SHIFT)
    print(f"[sigmas] {sigmas[0]:.4f} … {sigmas[-2]:.4f} → 0 ({args.steps} steps)", flush=True)

    text = encode_text(args.repo, args.max_sequence_length, "f32")
    embeds = text["encoder_hidden_states"]
    print(f"[text] encoder_hidden_states={list(embeds.shape)}", flush=True)

    latent = args.resolution // SPATIAL_COMPRESSION
    latents = torch.randn(
        (1, LATENT_CHANNELS, 1, latent, latent),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float32,
    )

    _proj, model = timesteps_proj_table(args.repo, sigmas, "f32", args.lora, args.lora_scale)
    eligible = sum(1 for _, m in model.named_modules() if is_eligible(m))
    linears = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    print(f"[eligible] linear {linears} 本中 k%4==0 が {eligible} 本", flush=True)

    attention_nodes = count_attention_nodes(model)
    print(f"[attention] ノード {len(attention_nodes)} 本 / step", flush=True)

    out: dict[str, torch.Tensor] = {}
    attn_stats: dict[str, dict[str, AttnStat]] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    linear_stats: dict[str, LayerStat] = {}

    weights_i8 = False
    for config in CONFIGS:
        if config.weight_i8:
            report = fake_quant_int8(model)
            print(f"[fake-quant] transformer (i8): {report.describe()}", flush=True)
            weights_i8 = True

        # 層別統計を採るのは基準構成 (c) だけ（他は計測コストを払う理由が無い）。
        hooks: ActQuantHooks | None = None
        handles: list[object] = []
        attached = 0
        if config.linear_act and config.name == BASELINE:
            hooks = ActQuantHooks(model, measure_step=args.measure_step)
            hooks.attach()
            attached = len(hooks.stats)
        elif config.linear_act:
            handles, attached = attach_act_quant(model)

        sim: AttnSim | None = None
        if config.attn or config.score_f16 or config.measure_pt:
            sim = AttnSim(
                model=model,
                mode=config.attn,
                stats_quantized=config.stats_quantized,
                score_f16=config.score_f16,
                measure_pt=config.measure_pt,
                measure_step=args.measure_step,
                disable=(args.inject == "drop-attn-quant"),
            )
            sim.attach()

        started = time.perf_counter()
        result = reference_steps(
            model,
            latents,
            embeds,
            embeds,
            sigmas,
            # `reference_steps` は Anima family 移行（c84dd07）から (width, height) を受ける —
            # int のままだと展開で TypeError（known-issues 2026-08-19 解消分・正方形前提）。
            (args.resolution, args.resolution),
            args.steps,
            GUIDANCE,
        )
        for key, value in result.items():
            if key.startswith("latents_"):
                out[f"{config.name}/{key}"] = value
        elapsed = time.perf_counter() - started

        if hooks is not None:
            hooks.detach()
            linear_stats = hooks.stats
        detach_act_quant(handles)
        calls = skipped = 0
        s16 = ScoreF16Probe()
        if sim is not None:
            sim.detach()
            attn_stats[config.name] = sim.stats
            calls, skipped = sim.calls, sim.skipped
            s16 = sim.s16
        # 差し替えは構成を跨ぐ唯一のグローバル可変状態。素の実体まで戻ったことを **毎構成**
        # `is` で見る（残ると次の構成が黙って量子化され、text/VAE 段まで汚染する）。
        if F.scaled_dot_product_attention is not PRISTINE_SDPA:
            raise AssertionError(f"{config.name}: SDPA の差し替えが残っている")

        diagnostics[config.name] = {
            "weight_i8": weights_i8,
            "linear_act_i8": config.linear_act,
            "linear_act_sites": attached,
            "attn": config.attn,
            "stats_quantized": config.stats_quantized,
            "score_f16": config.score_f16,
            "s16_rounded_calls": s16.calls,
            "s16_fired_calls": s16.fired,
            "s16_elements": s16.elements,
            "s16_abs_max": s16.abs_max,
            "attn_calls": calls,
            "attn_calls_expected": len(attention_nodes) * args.steps if sim is not None else 0,
            "attn_nodes_seen": len(attn_stats.get(config.name, {})),
            "attn_skipped_calls": skipped,
            "elapsed": round(elapsed, 1),
        }
        s16_note = f" s16={s16.fired}/{s16.calls} fired" if config.score_f16 else ""
        print(
            f"[{config.name}] {elapsed:.1f}s linear-act={attached} attn={calls} calls{s16_note}",
            flush=True,
        )

    del model
    gc.collect()
    return DitRun(out, linear_stats, attn_stats, diagnostics, attention_nodes)


def decode_all(
    repo: str, latents: dict[str, torch.Tensor], last: str, tags: Sequence[str]
) -> dict[str, torch.Tensor]:
    """全構成の最終 latent を**同じ f32 VAE**で decode する（VAE のロードは 1 回）。

    構成名を引数で受けるのは、`--w4-screen` が別の表を持つため（10 構成のモジュール
    グローバルをこちらから読むと、モードごとに別の集合を渡せない）。
    """
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(repo, subfolder="vae")
    vae.to(torch.float32).eval()
    images: dict[str, torch.Tensor] = {}
    for tag in tags:
        started = time.perf_counter()
        z = denormalize_latents(vae, latents[f"{tag}/{last}"])
        with torch.no_grad():
            images[tag] = vae.decode(z, return_dict=False)[0][:, :, 0].contiguous()
        print(f"[decode] {tag} ({time.perf_counter() - started:.1f}s)", flush=True)
    del vae
    gc.collect()
    return images


def save_images(images: dict[str, torch.Tensor], out: Path) -> str:
    """PNG で保存（PIL が無ければ npy へフォールバックし、その旨を返す）。"""
    arrays = {tag: to_uint8(image) for tag, image in images.items()}
    try:
        from PIL import Image
    except ImportError:
        import numpy as np

        for tag, array in arrays.items():
            np.save(out / f"image_{tag}.npy", array.numpy())
        return "PIL が無いため image_*.npy（uint8 [H,W,3]）で保存"
    for tag, array in arrays.items():
        Image.fromarray(array.numpy()).save(out / f"image_{tag}.png")
    return "image_*.png"


# ---- w4 方式スクリーニング（`--w4-screen` — 独立モード）----------------------
#
# 上の 10 構成とは**別経路**（MUST）。あちらは (a)→(b)→… を同じインスタンスへ in-place で
# 積み上げる意味論（重みの i8 の上に活性 i8 を足す）だが、こちらは**方式間の比較**なので、
# 構成ごとに pristine（素の f32 重み）へ戻してから当てる。戻さずに当てると測っているのが
# 「NF4 の上の MXFP4」になり、方式比較そのものが成立しない（EG / MiniCPM5 の台本と同じ規律）。
# 記号・基準・出力ファイル名も上の表とは共有しない。
#
# 持ち込む 4 方式は、安いファミリ 2 本のスクリーニングで残った勝者（ADR 0069 追記 5）:
#
#     rtn-i4-g32     配布形 `i4` の格納そのもの（比較の基準）
#     nf4            正規分布の分位点格子（fp4 は同コストで nf4 に支配されたので落選）
#     mxfp4          FP4 表 × group の 2 のべき scale（4.25 bpw の側）
#     kmeans:shared  層をまたいで 1 枚の 16 centroid 表（per_tensor は崩壊・per_channel は被支配）
#
# 丸めの実装は全て core（`karume.quantize` / `karume.quant_methods`）の共有で、対象選択も
# `iter_quant_targets` の共有 — 写した別実装にすると「測った対象・式」と「出荷する対象・式」が
# 黙って割れる。スコープは 10 構成と同じ **DiT のみ**（text_encoder / text_conditioner / VAE は
# 全構成 f32 のまま）。

#: 方式比較の group 長 = **32 固定**（ADR 0069 追記 1 で確定した既定 = core の既定）。
#: g 軸を同時に振らないのは、方式の差と g の差が混ざると「どちらが効いたか」が言えなくなるため
#: （ADR 0069 追記 5 ③ / 2026-08-19 ユーザー裁定）。
W4_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: 丸めの対象型。**DiT に実在する量子化可能型は `nn.Linear` だけ**（patchify も
#: `CosmosPatchEmbed.proj` = `nn.Linear`・埋め込み表も conv も無い）なので、i4 の実行経路
#: （ADR 0069 決定 5 の linear 限定）と対象が一致する = 全構成が**配布対応形**になる。
#: 非 linear が現れたら {@link scan_w4_targets} が fail loudly で落とす。
W4_OP_TYPES: tuple[type[nn.Module], ...] = (nn.Linear,)

#: 基準構成の綴り（丸めなし — 全ての relRMS / PSNR の分母）。
W4_BASE_CONFIG = "f32"

#: `--w4-screen` の生成物（10 構成の `report.md` / `attn.json` とは別名 — 同じ `--out` へ
#: 出しても取り違えないように）。
W4_REPORT_FILE = "w4_report.md"
W4_JSON_FILE = "w4_screen.json"
W4_LATENTS_FILE = "w4_latents.safetensors"

#: `kmeans:shared` の**表の fit** に使う標本数の予算。shared は全対象の正規化値を 1 本へ
#: 連結してから Lloyd 法を回すので、f32 の連結 + f64 の作業領域 + int64 の割り当て添字で
#: 標本 1 要素あたり ~28B 要る。DiT は 19.6 億要素あり、全量 fit は連結だけで 7.3GiB・
#: 作業領域込みで 50GiB 級 = 実メモリ（31GB）に載らない。2,000 万要素なら ~0.6GiB に収まり、
#: 16 centroid の 1 次元 k-means には桁で足りる（適用は常に全量 — core の `fit_stride`）。
W4_KMEANS_FIT_BUDGET = 20_000_000

# サイズ試算の bit 幅（格納規則の逐語）。出典は `docs/ir-v1.md` の `i4` 格納形（scale は
# **F32**・group ごと 1 個）と OCP Microscaling Formats v1.0（MX の共有 scale は E8M0 =
# 指数 1 バイト）。k-means は格納形を持たない測定専用方式なので、**表のコストを込みで**数える。

#: 4bit 格子のペイロード（全方式共通 — 比較しているのは「格子の張り方」であって bit 数ではない）。
W4_PAYLOAD_BITS = 4.0
#: group scale の bit 幅（`i4` の格納は F32 の group scale が MUST — `docs/ir-v1.md`）。
W4_F32_SCALE_BITS = 32.0
#: MXFP4 の共有 scale は E8M0（指数 1 バイト）。
W4_MX_SCALE_BITS = 8.0
#: codebook 1 エントリの bit 幅（centroid を F32 で持つ）。
W4_CODEBOOK_ENTRY_BITS = 32.0

W4_F32_BITS = 32.0
BITS_PER_BYTE = 8
MIB = 1024 * 1024


@dataclass(frozen=True)
class W4Targets:
    """w4 の対象集合の素性（サイズ試算の入力と、除外の台帳）。

    `groups` は g32 の group 数の総和で、{@link karume.quantize.channel_rows} の平坦形
    （= 丸めが group を割る軸）から数える — 別の軸で数えると試算だけが別の形の数になる。
    """

    modules: int
    elements: int
    groups: int
    #: 量子化軸が `W4_GROUP_SIZE` で割り切れず**対象から外した**重み（FQN → 量子化軸長）。
    excluded: Mapping[str, int]

    def include(self, name: str) -> bool:
        """`fake_quant_*` の `include` 述語（**モジュール FQN** で呼ばれる）。

        FQN の綴りは {@link karume.quantize.iter_quant_targets} と同じ形で作る（片方だけ
        別の綴りにすると、除外したつもりのモジュールが黙って対象に残る）。
        """
        return (f"{name}.weight" if name else "weight") not in self.excluded


def scan_w4_targets(model: nn.Module) -> W4Targets:
    """対象集合の素性と、量子化軸が g32 で割り切れない除外一覧を 1 度に採る。

    MUST: 非 linear の量子化可能型（conv 系 / embedding）が現れたら fail loudly。この台本は
    「対象 = linear 限定 = i4 の実行経路と一致 = 配布対応形」を前提に表を作っており、上流の
    diffusers が patchify を conv へ替えれば前提が黙って崩れる（表は出るが、出せない形の
    数値になる）。増えたときは**非 linear 込みの列を足すかどうか**をまず決める。

    MUST: 割り切れない対象は `include` で外し、除外一覧を出力へ載せる（ADR 0069 決定 2 —
    端数 group は格納できない形なので、その bpw も品質も出さない）。黙って全体を諦めない。
    """
    wide = {fqn for fqn, _weight, _axis in iter_quant_targets(model, QUANT_MODULE_TYPES)}
    targets = list(iter_quant_targets(model, W4_OP_TYPES))
    extra = sorted(wide - {fqn for fqn, _weight, _axis in targets})
    if extra:
        raise AssertionError(
            f"DiT に非 linear の量子化可能な重みがある: {extra}"
            "（この台本は linear 限定 = i4 の実行経路と一致する形を測る前提 — "
            "非 linear 込みの列を足すかどうかを先に決めること）"
        )
    modules = elements = groups = 0
    excluded: dict[str, int] = {}
    for fqn, weight, axis in targets:
        rows = channel_rows(weight, axis)
        span = int(rows.shape[1])
        if span % W4_GROUP_SIZE:
            excluded[fqn] = span
            continue
        modules += 1
        elements += int(weight.numel())
        groups += int(rows.shape[0]) * (span // W4_GROUP_SIZE)
    if not modules:
        raise AssertionError(
            f"g{W4_GROUP_SIZE} で量子化できる重みが 1 本も無い（除外 {sorted(excluded)}）"
        )
    return W4Targets(modules=modules, elements=elements, groups=groups, excluded=excluded)


def w4_fit_stride(elements: int) -> int:
    """対象要素数から `fake_quant_kmeans(fit_stride=…)` の等間隔 stride を決める。

    予算（{@link W4_KMEANS_FIT_BUDGET}）に収まるなら 1 = 全量 fit。**奇数へ切り上げる**のは
    group 長が 2 冪だから — 偶数 stride は group 内の偶数 lane しか踏まない（`gcd > 1`）が、
    奇数なら常に互いに素で標本が group 内の全 lane を等しく踏む。
    """
    stride = -(-elements // W4_KMEANS_FIT_BUDGET)
    return stride + 1 if stride % 2 == 0 else stride


class W4RoundReport(Protocol):
    """`fake_quant_*` の戻り値のうち**この台本が読む面**だけ（i4 と測定専用方式で型が違う）。"""

    @property
    def modules(self) -> int: ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class W4Method:
    """1 方式のレシピ（丸めの当て方とサイズ試算の式を 1 行に束ねる）。

    2 つを別表に散らすと「品質を測った方式」と「サイズを試算した方式」が黙って割れる
    （方式を 1 つ足したときに片方だけ更新される形になる）。`apply` は**素の f32 重みへ**
    当てる前提で、積み重ねの禁止は呼び出し側の {@link restore_w4} が担保する。
    """

    name: str
    #: `(model, include 述語, kmeans の fit_stride) -> 丸めの計数`。
    apply: Callable[[nn.Module, Callable[[str], bool], int], W4RoundReport]
    #: 対象集合の投影ビット数。
    bits: Callable[[W4Targets], float]
    #: `bits` の式（**出力へそのまま載せる** — 投影の前提を表から追えるように）。
    formula: str


def _w4_group_scaled_bits(scale_bits: float) -> Callable[[W4Targets], float]:
    """group scale を持つ方式の総 bit（`4·N + scale_bits·G`）。"""

    def bits(targets: W4Targets) -> float:
        return W4_PAYLOAD_BITS * targets.elements + scale_bits * targets.groups

    return bits


W4_METHODS: tuple[W4Method, ...] = (
    W4Method(
        "rtn-i4-g32",
        lambda model, include, _stride: fake_quant_int4(
            model, W4_GROUP_SIZE, include=include, op_types=W4_OP_TYPES
        ),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G（G = N/32 の group ごとに F32 scale 1 個 — 配布形 `i4` の格納そのもの）",
    ),
    W4Method(
        "nf4",
        lambda model, include, _stride: fake_quant_nf4(
            model, W4_GROUP_SIZE, include=include, op_types=W4_OP_TYPES
        ),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G（i4 と同じ group absmax scale・格子が正規分布の分位点）",
    ),
    W4Method(
        "mxfp4",
        lambda model, include, _stride: fake_quant_mxfp4(
            model, W4_GROUP_SIZE, include=include, op_types=W4_OP_TYPES
        ),
        _w4_group_scaled_bits(W4_MX_SCALE_BITS),
        "4·N + 8·G（共有 scale が E8M0 の 2 のべき — OCP MX v1.0）",
    ),
    W4Method(
        "kmeans:shared",
        lambda model, include, stride: fake_quant_kmeans(
            model,
            "shared",
            W4_GROUP_SIZE,
            include=include,
            op_types=W4_OP_TYPES,
            fit_stride=stride,
        ),
        lambda targets: (
            W4_PAYLOAD_BITS * targets.elements
            + W4_F32_SCALE_BITS * targets.groups
            + W4_CODEBOOK_ENTRY_BITS * DEFAULT_CODEBOOK_LEVELS
        ),
        "4·N + 32·G + 32·16（表はモデル全体で 1 枚・group absmax scale つき）",
    ),
)

W4_METHOD_NAMES = tuple(method.name for method in W4_METHODS)


def select_w4_methods(only: Sequence[str]) -> tuple[W4Method, ...]:
    """`--w4-only` で選んだ方式を宣言順で返す（基準 `f32` は常に走る — 比較の分母）。"""
    if not only:
        return W4_METHODS
    chosen = set(only)
    return tuple(method for method in W4_METHODS if method.name in chosen)


def w4_size_projection(method: W4Method, targets: W4Targets) -> dict[str, Any]:
    """方式 × 対象集合の実効 bpw と投影 MiB（対象テンソルだけの合計 — 模型全体ではない）。"""
    bits = method.bits(targets)
    baseline_bits = W4_F32_BITS * targets.elements
    return {
        "modules": targets.modules,
        "elements": targets.elements,
        "groups": targets.groups,
        "bitsPerWeight": bits / targets.elements,
        "projectedMiB": bits / BITS_PER_BYTE / MIB,
        "f32MiB": baseline_bits / BITS_PER_BYTE / MIB,
        "ratio": bits / baseline_bits,
        "formula": method.formula,
    }


def restore_w4(weights: Mapping[str, torch.Tensor], pristine: Mapping[str, torch.Tensor]) -> None:
    """全対象を pristine の f32 重みへ戻す（**方式を積み重ねない** MUST の実体）。"""
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])


# ---- 校正付き丸め（`--w4-screen` の校正列・波 J-2）---------------------------
#
# 上の 4 方式は重みだけを見て丸める（RTN 系）。**校正付き丸め**（GPTQ — core の
# `karume.quant_calib`）は「その層に実際に流れる活性」から**同じ格子の中で**丸め先を選び直す
# 側で、格納グリッド 3 種を同じ列へ足す。積み重ね禁止も pristine 復元も 4 方式と同文。
#
# 対象は **DiT block 列の `nn.Linear` 限定**（{@link CALIB_TARGET}）で、4 方式の対象（DiT 全体の
# linear）とは**集合が違う** — 校正の駆動（`calibrate_stages`）は stage 内の `nn.Linear` に
# 閉じるので、block の外（`patch_embed.proj` / `norm_out` / `proj_out` / `time_embed`）が入らない。
# サイズ試算も品質もその集合で読む（表と JSON に対象を明記する）。
#
# 校正入力は**参照 step 列を横断**して採る（`--steps` ぶん・固定 PROMPT / 固定 SEED）。拡散
# モデルの活性は sigma で分布が動くので、1 step だけ見ると後半 step の分布が校正から漏れる。

#: 校正付き構成の対象名（表と JSON の「対象」列）。
CALIB_TARGET = "dit:blocks-linear"

#: DiT block 列の**モデル内 FQN 接頭辞**（scale 台帳のキーを `Int4Report` と同じ FQN 空間へ
#: 揃えるために stage へ渡す — core の `StageSpec` の契約）。
CALIB_BLOCK_PREFIX = "transformer_blocks"


@dataclass(frozen=True)
class CalibConfig:
    """校正付き丸め 1 本ぶんの指定（方式 × 格納グリッド）。

    `bits` / `formula` は {@link W4Method} と同じ器（{@link w4_size_projection} が両方を受ける）
    — 校正は**格子を 1 バイトも変えない**ので、格納形の式は方式側と共有できる。
    """

    name: str
    method: CalibMethod
    grid: GridSpec
    bits: Callable[[W4Targets], float]
    formula: str


def _calib_codebook_bits(targets: W4Targets) -> float:
    """`kmeans_shared` の総 bit — 表は**層ごとに 1 枚**（core の `GridSpec` の NOTE）。"""
    return (
        W4_PAYLOAD_BITS * targets.elements
        + W4_F32_SCALE_BITS * targets.groups
        + W4_CODEBOOK_ENTRY_BITS * DEFAULT_CODEBOOK_LEVELS * targets.modules
    )


#: 校正付き構成 3 本（この順で表に並ぶ）。AWQ を置かないのは、等価倍率 `s` が単独で格納できず
#: （fold か companion が要る）出口の無い列になるため（core の `quant_calib` の MUST）。
CALIB_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn",
        "gptq",
        GridSpec(kind="rtn", group_size=W4_GROUP_SIZE),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G（格納は配布形 `i4` そのもの — 校正は丸め先だけを変える）",
    ),
    CalibConfig(
        "gptq-nf4",
        "gptq",
        GridSpec(kind="nf4", group_size=W4_GROUP_SIZE),
        _w4_group_scaled_bits(W4_F32_SCALE_BITS),
        "4·N + 32·G（格子は NF4 の固定表）",
    ),
    CalibConfig(
        "gptq-kmeans",
        "gptq",
        GridSpec(kind="kmeans_shared", group_size=W4_GROUP_SIZE),
        _calib_codebook_bits,
        "4·N + 32·G + 32·16·(層数)（表の射程は**層内** — 全体 1 枚の kmeans:shared とは別式）",
    ),
)

CALIB_NAMES: tuple[str, ...] = tuple(config.name for config in CALIB_CONFIGS)


def select_calib(only: Sequence[str]) -> tuple[CalibConfig, ...]:
    """`--w4-only` で選んだ校正付き構成を宣言順で返す（{@link select_w4_methods} と同じ流儀）。"""
    if not only:
        return CALIB_CONFIGS
    chosen = set(only)
    return tuple(config for config in CALIB_CONFIGS if config.name in chosen)


def calib_stages(model: nn.Module) -> tuple[StageSpec, ...]:
    """実行順の DiT block を `(モデル内 FQN 接頭辞, stage)` で返す。

    block を**そのまま** stage にできるのは `CosmosTransformerBlock.forward` が
    「hidden を位置引数・残りを keyword で受けて hidden を返す」形をしているから
    （包み直すと写しが上流とずれうるので、包まないのが最善）。接頭辞に block 番号まで
    入れてあるので、stage 内の局所 FQN（`attn1.to_q.weight`）はそのままモデル内 FQN へ戻る。
    """
    return tuple(
        (f"{CALIB_BLOCK_PREFIX}.{index}", block)
        for index, block in enumerate(model.transformer_blocks)
    )


def scan_calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], W4Targets]:
    """stage 内の校正対象を fqn 引きの重みと素性で返す（**走査** = 過不足一致門の基準）。

    MUST: 非 linear の量子化可能型が現れたら fail loudly（{@link scan_w4_targets} と同文）。

    MUST: g32 非整列は**除外せず fail loudly** — 校正は stage を丸ごと駆動する形なので、
    途中の 1 本だけ外すと「走査の本数 = 丸めた本数」の門が張れなくなる（4 方式側は層ごとに
    独立なので除外一覧を出す運用でよい）。
    """
    weights: dict[str, torch.Tensor] = {}
    modules = elements = groups = 0
    for prefix, stage in stages:
        wide = {fqn for fqn, _weight, _axis in iter_quant_targets(stage, QUANT_MODULE_TYPES)}
        targets = list(iter_quant_targets(stage, W4_OP_TYPES))
        extra = sorted(wide - {fqn for fqn, _weight, _axis in targets})
        if extra:
            raise AssertionError(
                f"{prefix} に非 linear の量子化可能な重みがある: {extra}"
                "（校正は nn.Linear 限定 — 非 linear 込みの列を足すかどうかを先に決めること）"
            )
        for fqn, weight, axis in targets:
            rows = channel_rows(weight, axis)
            span = int(rows.shape[1])
            if span % W4_GROUP_SIZE:
                raise AssertionError(
                    f"{prefix}.{fqn}: 量子化軸 {span} が g{W4_GROUP_SIZE} で割り切れない"
                    "（校正は stage 単位で駆動するので 1 本だけ外す逃げ道が無い）"
                )
            weights[f"{prefix}.{fqn}"] = weight
            modules += 1
            elements += int(weight.numel())
            groups += int(rows.shape[0]) * (span // W4_GROUP_SIZE)
    if not modules:
        raise AssertionError(f"校正対象 '{CALIB_TARGET}' に量子化できる重みが 1 本も無い")
    return weights, W4Targets(modules=modules, elements=elements, groups=groups, excluded={})


def capture_stage_batches(
    model: nn.Module, run_reference: Callable[[], dict[str, torch.Tensor]], limit: int
) -> tuple[dict[str, torch.Tensor], tuple[StageBatch, ...]]:
    """参照 denoise を 1 周回しつつ、先頭 block への `(args, kwargs)` を step ごとに捕まえる。

    CFG=1（{@link GUIDANCE}）なので 1 step = DiT 1 forward = 1 バッチ。付随引数は
    `CosmosTransformer3DModel.forward` が block ループの**前**に 1 回作って全 block へ同じ
    ものを渡すので、stage 間で不変にできる（`advance_kwargs` は要らない）。

    位置引数の名前は `block.forward` の**シグネチャから**引く — 上流が引数の順序を変えても
    綴りが黙って入れ替わらないようにするため（写した名前の並びを持たない）。

    MUST: 呼び出し側は**丸めを 1 本も当てる前**（基準 `f32` の周回）にだけ使う。
    """
    blocks = list(model.transformer_blocks)
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    batches: list[StageBatch] = []

    def pre(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if len(batches) >= limit:
            return
        batches.append(
            (
                (args[0].detach(),),
                {**dict(zip(names[1:], args[1:], strict=False)), **kwargs},
            )
        )

    handle = blocks[0].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        result = run_reference()
    finally:
        handle.remove()
    if not batches:
        raise AssertionError(
            "校正入力を 1 step も捕まえられなかった"
            "（DiT の block ループが台本の想定と食い違っている）"
        )
    return result, tuple(batches)


@dataclass(frozen=True)
class CalibRig:
    """校正付き構成が共有する足場（stage 列・走査・先頭 stage への入力）。

    構成ごとに作り直さない — 捕捉は基準 `f32` の周回で 1 回だけ（**丸めを 1 本も当てる前**）。
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    targets: W4Targets
    batches: tuple[StageBatch, ...]


def apply_calib(config: CalibConfig, rig: CalibRig) -> CalibReport:
    """校正付き丸め 1 本を DiT へ in-place で当てる（stage 逐次の駆動は core 側）。"""
    report = calibrate_stages(rig.stages, rig.batches, method=config.method, spec=config.grid)
    assert_calib_covers_scan(report, rig.scan, config.name)
    return report


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """校正が丸めた層が stage の走査と**過不足なく**一致することを見る。

    MUST: fail loudly。stage の綴りや対象型が変わって block の一部が校正に載らなくなっても
    表には行が残り、しかも丸め漏れのぶん PSNR は**良い側**に出る（素通りを数字から読めない）。
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise AssertionError(
            f"[{name}] 校正が丸めた {report.modules} 本が走査の {len(scan)} 本と一致しない"
            f"（丸め漏れ {missing[:3]} / 走査に無い {extra[:3]}）"
        )


def run_w4_screen(args: argparse.Namespace) -> dict[str, Any]:
    """方式ごとに pristine から丸め直して DiT を走らせ、品質とサイズ試算を集める。

    模型のロードは 1 回きり（7.29GiB）。pristine の退避は対象の重みだけ（除外分は 1 度も
    触らないので採らない）で、丸めの一時領域と合わせたピークが実メモリを決める。

    校正付き構成（{@link CALIB_CONFIGS}）は 4 方式の**後**に走る（対象集合が違うので表の
    並びで混ざらない）。足場は基準 `f32` の周回で 1 回だけ組む — 校正入力は**丸めを 1 本も
    当てる前**に採る必要があるため。
    """
    methods = select_w4_methods(args.w4_only)
    calib_configs = select_calib(args.w4_only)
    sigmas = sigma_schedule(args.steps, SHIFT)
    print(f"[sigmas] {sigmas[0]:.4f} … {sigmas[-2]:.4f} → 0 ({args.steps} steps)", flush=True)

    text = encode_text(args.repo, args.max_sequence_length, "f32")
    embeds = text["encoder_hidden_states"]
    print(f"[text] encoder_hidden_states={list(embeds.shape)}", flush=True)

    latent = args.resolution // SPATIAL_COMPRESSION
    latents_init = torch.randn(
        (1, LATENT_CHANNELS, 1, latent, latent),
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float32,
    )

    _proj, model = timesteps_proj_table(args.repo, sigmas, "f32", args.lora, args.lora_scale)
    targets = scan_w4_targets(model)
    stride = w4_fit_stride(targets.elements)
    print(
        f"[w4] 対象 {targets.modules} 本 / {targets.elements:,} 要素"
        f" / group {targets.groups:,}（g{W4_GROUP_SIZE}）",
        flush=True,
    )
    for fqn, span in sorted(targets.excluded.items()):
        print(f"[w4] 除外 {fqn}: 量子化軸 {span} が {W4_GROUP_SIZE} で割り切れない", flush=True)
    if stride > 1:
        print(
            f"[w4] kmeans:shared の表は 1/{stride} の等間隔部分標本で fit"
            f"（≈ {targets.elements // stride:,} 要素・適用は全量）",
            flush=True,
        )

    weights = {
        fqn: weight
        for fqn, weight, _axis in iter_quant_targets(model, W4_OP_TYPES)
        if fqn not in targets.excluded
    }
    pristine = {fqn: weight.detach().clone() for fqn, weight in weights.items()}

    rig: CalibRig | None = None
    if calib_configs:
        stages = calib_stages(model)
        calib_scan, calib_targets = scan_calib_targets(stages)
        print(
            f"[calib] stage {len(stages)} 段 / 対象 linear {calib_targets.modules} 本"
            f" / {calib_targets.elements:,} 要素（{CALIB_TARGET} — block の外は含まない）",
            flush=True,
        )

    # `model` を**既定引数で束ねる**のは、下の `del` が同名の局所を外すため（素のクロージャ
    # だと、消えた後に呼ばれたとき NameError になる形が残る）。
    def run_reference(model: nn.Module = model) -> dict[str, torch.Tensor]:
        return reference_steps(
            model,
            latents_init,
            embeds,
            embeds,
            sigmas,
            (args.resolution, args.resolution),
            args.steps,
            GUIDANCE,
        )

    latents: dict[str, torch.Tensor] = {}
    configs: dict[str, dict[str, Any]] = {}
    sizes: dict[str, dict[str, Any]] = {}
    for method in (None, *methods):
        name = W4_BASE_CONFIG if method is None else method.name
        started = time.perf_counter()
        if method is None:
            report = "丸めなし（基準）"
        else:
            # MUST: 当てる前に必ず戻す（方式を積み重ねない — 節の冒頭）。
            restore_w4(weights, pristine)
            rounded = method.apply(model, targets.include, stride)
            if rounded.modules != targets.modules:
                raise AssertionError(
                    f"{name}: 丸めた本数 {rounded.modules} が対象 {targets.modules} と違う"
                    "（include 述語か op_types が効いていない）"
                )
            report = rounded.describe()
            sizes[name] = w4_size_projection(method, targets)
        if method is None and calib_configs:
            # MUST: 捕捉は**丸めを 1 本も当てる前**（基準 f32 の周回）に 1 回だけ。
            result, batches = capture_stage_batches(model, run_reference, args.steps)
            rig = CalibRig(stages=stages, scan=calib_scan, targets=calib_targets, batches=batches)
            print(f"[calib] 校正バッチ {len(batches)} 本（step 横断）を捕捉", flush=True)
        else:
            result = run_reference()
        for key, value in result.items():
            if key.startswith("latents_"):
                latents[f"{name}/{key}"] = value
        elapsed = time.perf_counter() - started
        configs[name] = {"method": name, "quantReport": report, "elapsed": round(elapsed, 1)}
        print(f"[{name}] {report} ({elapsed:.1f}s)", flush=True)

    for config in calib_configs:
        if rig is None:
            raise AssertionError("校正の足場が組まれていない（基準 f32 の周回が走っていない）")
        started = time.perf_counter()
        # MUST: 校正付きも積み重ねない — 当てる前に pristine へ戻す（4 方式と同文）。
        restore_w4(weights, pristine)
        calib_report = apply_calib(config, rig)
        result = run_reference()
        for key, value in result.items():
            if key.startswith("latents_"):
                latents[f"{config.name}/{key}"] = value
        elapsed = time.perf_counter() - started
        configs[config.name] = {
            "method": f"{calib_report.method}/{calib_report.grid}",
            "target": CALIB_TARGET,
            "quantReport": calib_report.describe(),
            "calibModules": calib_report.modules,
            "calibBatches": len(rig.batches),
            "elapsed": round(elapsed, 1),
        }
        sizes[config.name] = w4_size_projection(config, rig.targets)
        print(f"[{config.name}] {calib_report.describe()} ({elapsed:.1f}s)", flush=True)

    calib_meta = (
        None
        if rig is None
        else {
            "name": CALIB_TARGET,
            "stages": len(rig.stages),
            "modules": rig.targets.modules,
            "elements": rig.targets.elements,
            "groups": rig.targets.groups,
            "batches": len(rig.batches),
        }
    )
    # 丸め済みの重みと pristine（対象と同サイズ）を VAE decode の前に手放す。
    # `rig` は stage 経由で block を、batches で校正入力を掴んだままなので一緒に落とす。
    del model, weights, pristine, rig
    if calib_configs:
        del stages, calib_scan, calib_targets
    gc.collect()

    last = f"latents_step{args.steps:04d}"
    tags = tuple(configs)
    images = decode_all(args.repo, latents, last, tags)
    keys = [f"latents_step{index:04d}" for index in range(1, args.steps + 1)]
    failures: list[str] = []
    base_image = images[W4_BASE_CONFIG]
    u8_base = to_uint8(base_image).to(torch.float32)
    # 基準行そのものは測らない（自分との比較は relRMS 0 / PSNR ∞ で情報が無く、∞ は JSON の
    # 値としても不正）。表も門も丸めた構成だけを見る。
    for name, entry in ((name, configs[name]) for name in configs if name != W4_BASE_CONFIG):
        u8 = to_uint8(images[name]).to(torch.float32)
        entry["latentRelRms"] = [
            rel_rms(latents[f"{name}/{key}"], latents[f"{W4_BASE_CONFIG}/{key}"]) for key in keys
        ]
        entry["psnrF32"] = psnr(images[name], base_image, 2.0)
        entry["psnrUint8"] = psnr(u8, u8_base, 255.0)
        entry["imageRelRms"] = rel_rms(images[name], base_image)
        entry["uint8MaxDiff"] = float((u8 - u8_base).abs().max())
        entry["moved"] = not torch.equal(
            latents[f"{name}/{last}"], latents[f"{W4_BASE_CONFIG}/{last}"]
        )
        entry["size"] = sizes[name]
        # 恒真化の遮断: 素通りは常に「品質が良い」側の嘘になる（丸めが 1 本も当たっていない
        # 場合だけ基準とビット一致し、しかも数値は完璧に見える）。
        if not entry["moved"]:
            failures.append(f"{name}: 最終 latent が `{W4_BASE_CONFIG}` とビット一致（素通り）")

    image_note = save_images(images, args.out)
    if not args.no_latents:
        from safetensors.torch import save_file

        save_file(latents, str(args.out / W4_LATENTS_FILE))
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/anima/measure_quant.py --w4-screen",
        "torch": torch.__version__,
        "repo": args.repo,
        "prompt": PROMPT,
        "steps": args.steps,
        "resolution": args.resolution,
        "guidanceScale": GUIDANCE,
        "seed": SEED,
        "lora": str(args.lora),
        "groupSize": W4_GROUP_SIZE,
        "codebookLevels": DEFAULT_CODEBOOK_LEVELS,
        "kmeansFitStride": stride,
        "kmeansFitSamplesApprox": targets.elements // stride,
        "targets": {
            "opTypes": [cls.__name__ for cls in W4_OP_TYPES],
            "modules": targets.modules,
            "elements": targets.elements,
            "groups": targets.groups,
        },
        # 校正列の対象は上の 4 方式と**別集合**（block の外が入らない）— 表の bpw と品質を
        # 読むときに取り違えないよう、素性を並べて出す。
        "calibTargets": calib_meta,
        "excluded": [
            {"weight": fqn, "quantAxis": span} for fqn, span in sorted(targets.excluded.items())
        ],
        "imageNote": image_note,
        "configs": configs,
        "gates": {
            "moved": {
                name: entry["moved"] for name, entry in configs.items() if name != W4_BASE_CONFIG
            },
            "failures": failures,
        },
    }


def _w4_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def w4_report_markdown(payload: Mapping[str, Any]) -> str:
    """`w4_report.md` の本文（**payload だけ**から作る — 表と JSON が割れないように）。"""
    configs: dict[str, dict[str, Any]] = payload["configs"]
    names = list(configs)
    quantized = [name for name in names if name != W4_BASE_CONFIG]
    lines: list[str] = []
    lines.append(
        f"# w4 方式スクリーニング — Anima DiT の丸め方式 {len(quantized)} 種（torch CPU 実測）"
    )
    lines.append("")
    lines.append(
        f"- 日付: {payload['generated']} / 計測: `{payload['script']}`"
        f"（torch {payload['torch']}・CPU f32）"
    )
    lines.append(
        f"- 条件: Anima turbo {payload['steps']} step・{payload['resolution']}px"
        f"・CFG={payload['guidanceScale']:g}・LoRA 焼き込み・seed {payload['seed']}"
    )
    lines.append(f"- プロンプト: `{payload['prompt']}`")
    lines.append(
        f"- 対象: DiT の `{'` / `'.join(payload['targets']['opTypes'])}`"
        f" {payload['targets']['modules']} 本 / {payload['targets']['elements']:,} 要素"
        f"（g={payload['groupSize']} 固定・group {payload['targets']['groups']:,} 個）。"
        "**非 DiT（text_encoder / text_conditioner / VAE）は全構成とも f32 のまま**。"
    )
    lines.append(
        "- MUST: 構成ごとに pristine（素の f32 重み）へ戻してから当てる — **方式は積み重ねない**。"
        "同じ台本の 10 構成（w8 / w8a8 / attention a8）の積み上げ意味論とは**別経路**。"
    )
    lines.append(
        "- 丸めは全て core の共有実装（`karume.quantize.fake_quant_int4` / "
        "`karume.quant_methods` の nf4 / mxfp4 / kmeans）で、対象選択も "
        "`iter_quant_targets` の共有（測った対象と出荷する対象を割らないため）。"
    )
    excluded = payload["excluded"]
    if excluded:
        lines.append(
            "- 除外（量子化軸が g"
            f"{payload['groupSize']} で割り切れないので対象から外した — ADR 0069 決定 2）: "
            + " / ".join(f"`{item['weight']}`（軸 {item['quantAxis']}）" for item in excluded)
        )
    calib = payload.get("calibTargets")
    if calib:
        lines.append(
            f"- **校正付き構成**（`{'` / `'.join(CALIB_NAMES)}`）の対象は上とは別集合 —"
            f" `{calib['name']}` = DiT block 列の `nn.Linear` {calib['modules']} 本 /"
            f" {calib['elements']:,} 要素（block の外の `patch_embed` / `norm_out` /"
            " `proj_out` / `time_embed` は入らない）。校正入力は参照 step 列を横断した"
            f" {calib['batches']} バッチ（stage {calib['stages']} 段）。**bpw も品質も"
            "この集合で読む**（4 方式の行と直接は比較できない）。"
        )
    if payload["kmeansFitStride"] > 1:
        lines.append(
            f"- `kmeans:shared` の**表の fit だけ** 1/{payload['kmeansFitStride']} の等間隔"
            f"部分標本（≈ {payload['kmeansFitSamplesApprox']:,} 要素）で採った — **適用は全量**。"
            "全量 fit は作業領域が実メモリに載らないため（部分標本の表と全量の表は別物に"
            "なりうるので、数値を読む側が区別できるようここに明記する）。"
        )
    lines.append("")

    lines.append(f"## ① step ごとの latent relRMS（基準 = `{W4_BASE_CONFIG}`）")
    lines.append("")
    lines.append("`relRMS(v, r) = ‖v − r‖₂ / ‖r‖₂`。")
    lines.append("")
    rows = [
        [str(index + 1), *(f"{configs[name]['latentRelRms'][index]:.4e}" for name in quantized)]
        for index in range(payload["steps"])
    ]
    lines.append(_w4_table(["step", *(f"`{name}`" for name in quantized)], rows))
    lines.append("")

    lines.append(f"## ② VAE decode 後の最終画像（基準 = `{W4_BASE_CONFIG}`）")
    lines.append("")
    lines.append(
        "PSNR は `[-1,1]` の生 tensor（data range 2.0）と、PNG と同じ uint8 量子化後"
        "（data range 255）の 2 系列。relRMS は生 tensor。"
    )
    lines.append("")
    rows = [
        [
            f"`{name}`",
            f"{configs[name]['psnrF32']:.2f}",
            f"{configs[name]['psnrUint8']:.2f}",
            f"{configs[name]['imageRelRms']:.4e}",
            f"{configs[name]['uint8MaxDiff']:.0f}",
            f"{configs[name]['elapsed']:.0f}s",
        ]
        for name in quantized
    ]
    lines.append(
        _w4_table(
            ["構成", "PSNR f32 (dB)", "PSNR uint8 (dB)", "relRMS", "uint8 最大差 (/255)", "所要"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"画像: {payload['imageNote']}（{' / '.join(f'`image_{name}`' for name in names)}）"
    )
    lines.append("")

    lines.append("## ③ サイズ試算（対象テンソル集合のみ・**式による投影**）")
    lines.append("")
    lines.append(
        f"N = 対象要素数 {payload['targets']['elements']:,} / G = N/{payload['groupSize']} = "
        f"{payload['targets']['groups']:,}。実測ではなく格納規則の式で、格納形を持たない方式"
        "（rtn 以外）は書けもしない — 品質と同じ表で並べるのは、表のコストが見えないまま"
        "品質だけで選ばないため。"
    )
    lines.append("")
    rows = [
        [
            f"`{name}`",
            f"{configs[name]['size']['bitsPerWeight']:.3f}",
            f"{configs[name]['size']['projectedMiB']:.1f}",
            f"{configs[name]['size']['f32MiB']:.1f}",
            f"{configs[name]['size']['ratio'] * 100:.1f}%",
            configs[name]["size"]["formula"],
        ]
        for name in quantized
    ]
    lines.append(_w4_table(["構成", "bpw", "投影 MiB", "f32 MiB", "比", "式"], rows))
    lines.append("")

    lines.append("## ④ 門")
    lines.append("")
    lines.append(
        "丸めが素通りしていないこと（各構成の最終 latent が基準と**ビット一致しない**）— "
        "一致するのは丸めが 1 本も当たっていない場合だけで、しかも品質は「完璧」側に出る。"
    )
    lines.append("")
    rows = [
        [
            f"`{name}`",
            "OK" if configs[name]["moved"] else "**NG（素通り）**",
            configs[name]["quantReport"],
        ]
        for name in quantized
    ]
    lines.append(_w4_table(["構成", "基準と異なる", "丸めの計数"], rows))
    lines.append("")
    return "\n".join(lines) + "\n"


def main_w4(args: argparse.Namespace) -> None:
    """`--w4-screen` の入口（10 構成は 1 本も走らない）。"""
    args.out.mkdir(parents=True, exist_ok=True)
    payload = run_w4_screen(args)
    (args.out / W4_REPORT_FILE).write_text(w4_report_markdown(payload), encoding="utf-8")
    (args.out / W4_JSON_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print(f"report OK → {args.out / W4_REPORT_FILE}", flush=True)
    failures = payload["gates"]["failures"]
    if failures:
        raise AssertionError("検証ゲートが赤: " + " / ".join(failures))
    print("gates: all green", flush=True)


def main() -> None:
    # --until が派生テーブルごと前方部分へ差し替えるため（宣言は使用より前が文法上の要請）。
    global CONFIGS, CONFIG_NAMES, ATTN_CONFIGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, required=True, help="レポートと画像の出力先")
    parser.add_argument("--steps", type=int, default=10, help="turbo の step 数")
    parser.add_argument("--resolution", type=int, default=512, help="正方形の一辺（px）")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA, help="DiT へ焼き込む LoRA")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--measure-step",
        type=int,
        default=1,
        help="層別の出力 relRMS を採る step（0 起点）",
    )
    parser.add_argument(
        "--inject",
        choices=("drop-attn-quant",),
        default=None,
        help="故障注入（検出器の検出力を実証する）— attention の SDPA 差し替えは掛けるが"
        " 量子化も S の f16 丸めも一切適用しない（(d)〜(j) が基準構成と一致する向きへ倒れ、"
        "`*-s16` は丸めの発火計数も 0 に落ちることを実測する）",
    )
    parser.add_argument(
        "--no-latents",
        action="store_true",
        help="dit_out.safetensors（構成 × step の生 latent）を書かない",
    )
    parser.add_argument(
        "--until",
        choices=CONFIG_NAMES,
        default=None,
        help="この構成までの**前方部分**だけ走らせる（P̃ 分布の確認だけなら `w8a8` で足りる）。"
        " 重みの i8 丸めは in-place で順序依存なので任意の部分集合は許さない — 前方接頭辞のみ。"
        f" 基準 `{BASELINE}` より手前で切る指定は集計が成立しないため拒否する",
    )
    parser.add_argument(
        "--w4-screen",
        action="store_true",
        help="**10 構成の代わりに** w4 の丸め方式スクリーニングを走らせる"
        f"（{' / '.join(W4_METHOD_NAMES)} + 校正付き {' / '.join(CALIB_NAMES)}"
        f" + 基準 f32・g{W4_GROUP_SIZE} 固定）。"
        "構成ごとに素の f32 重みへ戻してから当てる別経路で、生成物も別名",
    )
    parser.add_argument(
        "--w4-only",
        action="append",
        default=[],
        choices=(*W4_METHOD_NAMES, *CALIB_NAMES),
        help="`--w4-screen` でこの方式 / 校正付き構成だけ走らせる（複数可・部分再実行用）。"
        "基準 f32 は常に走る",
    )
    args = parser.parse_args()
    if args.w4_only and not args.w4_screen:
        parser.error("--w4-only は --w4-screen と併用する")
    if args.w4_screen:
        # 積み上げ経路のノブ（前方接頭辞と attention の故障注入）は方式スクリーニングに
        # 意味を持たない — 黙って無視せず拒否する。
        if args.until is not None or args.inject is not None:
            parser.error("--w4-screen は --until / --inject と併用できない（別経路）")
        main_w4(args)
        return
    if args.until is not None:
        cutoff = CONFIG_NAMES.index(args.until)
        if cutoff < CONFIG_NAMES.index(BASELINE):
            parser.error(f"--until は {BASELINE} 以降の構成のみ（基準が無いと集計不能）")
        # 派生テーブルごと前方部分へ差し替える（以降の全段が同じ正本を読む）。
        CONFIGS = CONFIGS[: cutoff + 1]
        CONFIG_NAMES = tuple(config.name for config in CONFIGS)
        ATTN_CONFIGS = attn_config_names(CONFIGS)
    args.out.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    run = run_dit(args)
    latents, stats = run.latents, run.linear_stats
    if not args.no_latents:
        save_file(latents, str(args.out / "dit_out.safetensors"))
    write_layer_csv(stats, args.out / "layers.csv")

    images = decode_all(args.repo, latents, f"latents_step{args.steps:04d}", CONFIG_NAMES)
    image_note = save_images(images, args.out)
    # --until が書き換えた派生表と engine 側の指標は**全て引数で渡す**（レポート側から
    # このモジュールを import すると循環するので、逆辺は型注釈だけに留めてある）。
    summary = build_summary(
        args,
        run,
        latents,
        images,
        config_names=CONFIG_NAMES,
        attn_configs=ATTN_CONFIGS,
        baseline=BASELINE,
        pt_hist_bins=PT_HIST_BINS,
        rel_rms=rel_rms,
        psnr=psnr,
        hist_quantile=hist_quantile,
    )
    report = build_report(
        args,
        run,
        latents,
        images,
        image_note,
        summary,
        config_names=CONFIG_NAMES,
        attn_configs=ATTN_CONFIGS,
        baseline=BASELINE,
        guidance=GUIDANCE,
        pt_hist_bins=PT_HIST_BINS,
        rel_rms=rel_rms,
        psnr=psnr,
        to_uint8=to_uint8,
        hist_quantile=hist_quantile,
    )
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "attn.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "steps": args.steps,
                "resolution": args.resolution,
                "guidance_scale": GUIDANCE,
                "seed": SEED,
                "lora": str(args.lora),
                "torch": torch.__version__,
                "eligible_linears": len(stats),
                "attention_nodes": len(run.attention_nodes),
                "configs": list(CONFIG_NAMES),
                "inject": args.inject,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    for key, value in summary["gates"].items():
        print(f"[gate] {key}: {value}", flush=True)
    print(f"report OK → {args.out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
