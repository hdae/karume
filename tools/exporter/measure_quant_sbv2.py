"""SBV2 **生成ネット**（net_g）の量子化パターン別 音質劣化を torch CPU で実測する。

`measure_quant_anima.py`（perf-c Q0）と同じ思想の品質ゲート — 実行カーネルを作る前に、
「その量子化で音がどこまで劣化するか」だけを torch の fake-quant で先に答える。GPU コードは
0 行で、ここで測るのは**量子化そのものの質**（ADR 0006 の fake-quant 方法論では E2E は
実装誤差しか測らないため、品質は別軸で測る必要がある）。

5 構成を**同一発話・同一乱数**（`models/sbv2-demo/out/dump.safetensors` の離散入力・ノイズ列）で
走らせる。**BERT は全構成 f32 固定**で、振るのは生成ネット側だけ:

    (1) f32    基準。`sbv2_demo.py reference` と同じ経路（既存 reference.wav とビット一致）
    (2) f16    適格重みを f16 表現可能値へ丸め（`quantize.round_weights_to_f16`）。活性は f32
               = 実装済みの **f16 格納系列**（ADR 0027）の対応物
    (3) w8     重みを per-channel symmetric i8 へ fake-quant（`quantize.fake_quant_int8`）
               = ADR 0019 の w8a32（i8 格納・f32 計算）の対応物
    (4) w8a16  (3) + **対象 op の入力活性**を f16 へ丸め
    (5) w8a8   (3) + **対象 op の入力活性**を per-位置 symmetric i8 へ fake-quant

    uv run --group sbv2 python measure_quant_sbv2.py

出力は `models/sbv2-demo/quant-sim/`（`<config>.wav` 5 本 + `report.json`）。

## 活性量子化の粒度と適用点

適用点は **対象 op の入力**（出力ではない）。対象は重みスロットを持つ op のうち活性が連続値
であるもの、すなわち `conv1d` / `conv_transpose1d` / `linear`。`embedding` は入力が整数 ID
なので対象外（重み側だけが量子化される）。

**MUST: 差し替えは op 粒度**（`torch.nn.functional` の関数を差し替える）— モジュールの
`forward_pre_hook` では足りない。`patch_sbv2._patched_ffn_forward` は `self.conv_1(x)` では
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
展開には **dump 側の `w_ceil`** を使う（`sbv2_demo.run_reference` が zp_noise の形を理由に
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
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

import export_sbv2
import sbv2_demo
from karume import patch_sbv2
from karume.act_quant import quantize_rows
from karume.quantize import fake_quant_int8, round_weights_to_f16

DEFAULT_DUMP = sbv2_demo.DEFAULT_DEMO_DIR / "out" / "dump.safetensors"
#: f32 構成の恒真化を防ぐ突合先（`sbv2_demo.py reference` が同じ dump から書いた WAV）。
DEFAULT_REFERENCE_WAV = sbv2_demo.DEFAULT_DEMO_DIR / "out" / "reference.wav"
DEFAULT_OUT = sbv2_demo.DEFAULT_DEMO_DIR / "quant-sim"


@dataclass(frozen=True)
class Recipe:
    """1 構成の量子化レシピ。

    `scope` は丸めを当てる**サブグラフ**（`front` = enc_p + dp + sdp / `voice` = flow + dec）。
    主要 5 構成は両方に当てる（= 生成ネット全体）。片側だけに当てる形は診断用（下）。
    """

    weight: str | None
    act: str | None
    scope: tuple[str, ...] = ("front", "voice")


#: 主要 5 構成（WAV を `--out` 直下へ書き、聴き比べの対象になる）。
CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        "f32": Recipe(None, None),
        "f16": Recipe("f16", None),
        "w8": Recipe("i8", None),
        "w8a16": Recipe("i8", "f16"),
        "w8a8": Recipe("i8", "i8"),
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
#: 考えるのは誤りで、`patch_sbv2._patched_ffn_forward` は `self.conv_1(x)` ではなく
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


# ---- 1 構成の実行 ------------------------------------------------------------


@dataclass
class ChainInputs:
    """全構成で共有する入力一式（BERT 特徴も含めて 1 度だけ作る）。"""

    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any]
    bert_feature: torch.Tensor
    g: torch.Tensor
    style_vec: torch.Tensor


def prepare_inputs(dump_path: Path, assets_path: Path) -> ChainInputs:
    """dump / assets を読み、**f32 の BERT** を 1 回だけ走らせて特徴を作る。

    BERT を構成間で共有するのは契約（BERT=f32 固定）そのもので、5 回走らせても同じ数が
    出る計算を 1 回に畳んでいるだけ。トークナイズの突合は `sbv2_demo.run_reference` と同じ
    ものを踏む（Deno 側の移植と食い違えば波形の手前で落ちる）。
    """
    meta = sbv2_demo.dump_metadata(dump_path)
    tensors = load_file(str(dump_path))
    assets = load_file(str(assets_path))

    tokenizer = sbv2_demo.load_bert_tokenizer()
    expected_ids = tokenizer(meta["bertText"])["input_ids"]
    dumped_ids = tensors["input_ids"].reshape(-1).tolist()
    if expected_ids != dumped_ids:
        raise AssertionError(
            f"DeBERTa トークナイズが dump と食い違う（python={expected_ids} / dump={dumped_ids}）"
        )

    from transformers import DebertaV2Model

    bert = DebertaV2Model.from_pretrained(
        sbv2_demo.BERT_REPO, dtype=torch.float32, attn_implementation="eager"
    )
    bert.eval()
    with torch.no_grad():
        hidden_states = bert(
            input_ids=tensors["input_ids"].to(torch.int64),
            attention_mask=tensors["attention_mask"].to(torch.int64),
            output_hidden_states=True,
        ).hidden_states
    hidden = hidden_states[-meta["bertHiddenFromEnd"]][0]
    word2ph = tensors["word2ph"].reshape(-1).tolist()
    bert_feature = sbv2_demo.tile_bert(hidden, word2ph).unsqueeze(0)
    del bert, hidden_states
    gc.collect()

    return ChainInputs(
        tensors=tensors,
        meta=meta,
        bert_feature=bert_feature,
        g=assets["g"].to(torch.float32),
        style_vec=assets["style_vec"].to(torch.float32),
    )


def run_config(
    name: str, model_dir: Path, inputs: ChainInputs, *, inject: str | None
) -> dict[str, Any]:
    """1 構成ぶんのチェーンを走らせ、波形と診断を返す。

    MUST（順序 — `quantize` モジュールと ADR 0018/0019/0027）: 重みの丸めは
    **`apply_all_patches` と `ensure_dec_plain` の後**に当てる。remove_weight_norm より先に
    丸めると `weight_g`/`weight_v` を丸めることになり、そこから作られる実効重みは f16/i8 の
    格子に乗らない（i8 は捨てられる要素が amax に効いて per-channel scale ごとずれる）。
    """
    started = time.perf_counter()
    recipe = RECIPES[name]
    tensors = inputs.tensors
    knobs = inputs.meta["knobs"]

    net_g, hps = export_sbv2.load_net_g(model_dir)
    patch_sbv2.apply_all_patches()
    export_sbv2.ensure_dec_plain(net_g)
    front = patch_sbv2.Sbv2Front(net_g)
    voice = patch_sbv2.Sbv2Voice(net_g)
    # front（enc_p / dp / sdp）と voice（flow / dec）は互いに素なモジュール集合。
    subgraphs = (("front", front), ("voice", voice))
    scoped = [(tag, module) for tag, module in subgraphs if tag in recipe.scope]

    # --- 重みの丸め -----------------------------------------------------------
    weight_report: dict[str, str] = {}
    for tag, module in scoped:
        if recipe.weight == "f16":
            weight_report[tag] = round_weights_to_f16(module).describe()
        elif recipe.weight == "i8":
            weight_report[tag] = fake_quant_int8(module).describe()

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
            inputs.bert_feature,
            inputs.style_vec,
            inputs.g,
            tensors["z_noise"].to(torch.float32),
        )

    # --- ホストグルー（デモ main.ts / sbv2_demo.run_reference と同式）-----------
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
    idx_k, valid = patch_sbv2.build_relattn_tables(total_frames, export_sbv2.EXPECTED_WINDOW_SIZE)

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

    configs = [entry_for(name) for name in CONFIGS]
    diagnostics = [entry_for(name) for name in DIAGNOSTICS if name in results]

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/exporter/measure_quant_sbv2.py",
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
        },
        "gates": gates,
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


# ---- 検証ゲート -------------------------------------------------------------


def run_gates(
    results: dict[str, dict[str, Any]], wavs: dict[str, Path], reference_wav: Path
) -> dict[str, Any]:
    """恒真化と構成生成バグの検出器（赤は最後にまとめて上げる）。

    ① f32 の WAV が既存 `reference.wav` とバイト一致（sim 経路が本物と同じ計算であること）
    ② 各構成が f32 と実際に違うこと（丸めが素通りしていない）
    ③ w8a16 / w8a8 が w8 と違うこと（**活性量子化の適用漏れの検出器** — 漏れると
       活性の丸めが消えて w8 と**ビット一致**する）。差し替え op 数と呼び出し地点数の
       0 検査も併せて置く（②③ は値の側から、こちらは仕掛けの側から同じ穴を見る）
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
        if name != "f32"
    }
    gates["differs_from_f32"] = differs

    w8 = results["w8"]["audio"]
    act_effective = {
        name: not bool(torch.equal(results[name]["audio"], w8)) for name in ("w8a16", "w8a8")
    }
    gates["act_quant_effective_vs_w8"] = act_effective

    patched_ops = {
        name: results[name]["diagnostics"]["act_quant"]["patched_ops"] for name in ("w8a16", "w8a8")
    }
    call_sites = {
        name: results[name]["diagnostics"]["act_quant"]["call_sites"] for name in ("w8a16", "w8a8")
    }
    gates["act_quant_patched_ops"] = patched_ops
    gates["act_quant_call_sites"] = call_sites

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
    gates["failures"] = failures
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=export_sbv2.DEFAULT_MODEL_DIR)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument(
        "--assets", type=Path, default=sbv2_demo.DEFAULT_DEMO_DIR / sbv2_demo.STYLE_FILE
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference-wav", type=Path, default=DEFAULT_REFERENCE_WAV)
    parser.add_argument("--top", type=int, default=12, help="層グループ表に載せる本数")
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="front / voice 直交分解の診断構成を走らせない（主要 5 構成だけ）",
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

    inputs = prepare_inputs(args.dump, args.assets)
    print(
        f"[inputs] text={inputs.meta['text']!r} bert={list(inputs.bert_feature.shape)}",
        flush=True,
    )

    plan: list[tuple[str, Path]] = [(name, args.out) for name in CONFIGS]
    if not args.no_diagnostics:
        diag_dir = args.out / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        plan += [(name, diag_dir) for name in DIAGNOSTICS]

    results: dict[str, dict[str, Any]] = {}
    wavs: dict[str, Path] = {}
    for name, directory in plan:
        results[name] = run_config(name, args.model_dir, inputs, inject=args.inject)
        audio = results[name]["audio"]
        path = directory / f"{name}.wav"
        path.write_bytes(sbv2_demo.wav_pcm16(audio.numpy(), inputs.meta["samplingRate"]))
        wavs[name] = path
        diagnostics = results[name]["diagnostics"]
        print(
            f"[{name}] {diagnostics['elapsed']:.0f}s frames={diagnostics['frames']}"
            f" act={diagnostics['act_quant']['calls']} calls → {path}",
            flush=True,
        )

    gates = run_gates(results, wavs, args.reference_wav)
    report = build_report(args, inputs, results, wavs, gates)
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
    print(f"report → {args.out / 'report.json'}")
    if gates["failures"]:
        raise AssertionError("検証ゲートが赤: " + " / ".join(gates["failures"]))
    print("gates: all green")


if __name__ == "__main__":
    main()
