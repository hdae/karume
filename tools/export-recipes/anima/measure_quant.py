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
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

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
from karume.quantize import INT8_MAX, fake_quant_int8

from .measure_report import build_report, build_summary, write_layer_csv
from .pipeline_ref import (
    LATENT_CHANNELS,
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
            model, latents, embeds, embeds, sigmas, args.resolution, args.steps, GUIDANCE
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


def decode_all(repo: str, latents: dict[str, torch.Tensor], last: str) -> dict[str, torch.Tensor]:
    """全構成の最終 latent を**同じ f32 VAE**で decode する（VAE のロードは 1 回）。"""
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(repo, subfolder="vae")
    vae.to(torch.float32).eval()
    images: dict[str, torch.Tensor] = {}
    for tag in CONFIG_NAMES:
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


def main() -> None:
    # --until が派生テーブルごと前方部分へ差し替えるため（宣言は使用より前が文法上の要請）。
    global CONFIGS, CONFIG_NAMES, ATTN_CONFIGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, required=True, help="レポートと画像の出力先")
    parser.add_argument("--steps", type=int, default=10, help="turbo の step 数")
    parser.add_argument("--resolution", type=int, default=512)
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
    args = parser.parse_args()
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

    images = decode_all(args.repo, latents, f"latents_step{args.steps:04d}")
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
