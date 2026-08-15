"""Irodori-TTS v4 の量子化構成別 品質劣化を torch CPU で実測する（波 2 の測定台本）。

`measure_quant_sbv2.py` / `measure_quant_anima.py` と同じ思想の品質ゲート — GPU コードは 0 行で、
ここで測るのは**量子化そのものの質**（ADR 0006 の fake-quant 方法論では E2E は実装誤差しか
測らないため、品質は別軸で測る必要がある）。答えを出したい問いは 1 つ:

    **S（フレーム数）は動くか** — duration を i8 にすると発話長が動きうる（SBV2 w8 の
    w_ceil 198→196 と同じ軸・ADR 0029）。latent 門の「S / forwards 完全一致」が壊れるなら、
    配布形の w8 席は混成（duration だけ据え置き）にする必要がある（ADR 0050 決定 6）。

    cd tools/exporter
    uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' \
        python -m irodori.measure_quant --config f32
    uv run ... python -m irodori.measure_quant --config i8-all
    uv run ... python -m irodori.measure_quant          # 実測はせずレポートだけ組み直す

**1 実行 = 1 構成**（`--config`）。full-loop は DiT を 1 ケースあたり 240 forward 回す torch CPU
実行なので、9 構成を 1 プロセスに詰めると数時間の不可分な塊になる。構成ごとに独立に呼べる形に
して、`--out` の下に per-config の JSON / WAV / z を残し、最後の実行が全部を突き合わせて
`report.json` と Markdown 表を出す（基準 `f32` を先に回すこと — 他構成はその成果物を読む）。

## 構成（`--config` の綴り）

| 構成             | 重み                 | 何を答える                                      |
| ---------------- | -------------------- | ----------------------------------------------- |
| `f32`            | 丸めなし（**基準**） | 恒真化防止（full-loop golden の z とバイト一致）|
| `f16`            | 全役割 f16           | ADR 0050 波 1 の系列の対応物                    |
| `i8-all`         | 全役割 i8            | 配布 `w8` 席（配布 25.2%）                      |
| `i8-mixed`       | duration 以外 i8     | 混成表の候補（ADR 0050 決定 6 の (i) 案）       |
| `w8a8`           | 全役割 i8 + 活性 i8  | 配布 `w8a8` 席（DiT の Linear だけ活性も i8）   |
| `i8-<役割>-only` | その役割だけ i8      | **直交分解 5 本**（境界 = `load_*` と 1:1）     |

`w8a8` は `i8-all` と**同じ丸めの重み**に、DiT の `nn.Linear` 入力へ per-token i8 の
fake-quant（`karume.act_quant.quantize_rows` — ランタイム `quantize-rows.ts` の数値鏡像）を
モジュールフックで足したもの。フックで安全なのは `patch.py` が `functional` で呼ぶのが
`rms_norm` / `silu` だけで、量子化対象の Linear が実モジュールのまま残るため（SBV2 では
`functional.conv1d` 直呼びのせいでフックが沈黙で取りこぼした — ADR 0029 の検出限界）。
**適用本数は毎回出力し、0 本なら fail loudly**（「w8a8 のつもりで w8 の数を採った」に気づけない
形を作らない）。活性量子化を DiT に限るのは配布形の席と揃えるため（`session` は `dit` の
Session にだけ降りる — models 側 `pipeline.ts`）。

直交分解の 5 本は `dit` / `backbone` / `speaker` / `duration` / `codec`。**5 本の和は `i8-all`
にならない** — projector 2 本（各 6.83MB）は単独の軸に立てていない（桁が 2 つ小さく、単独で
量子化する動機が無い — 量子化 recon の sizeBreakdown）。`i8-all` / `i8-mixed` には入る。

## 時間グリッドの固定（SNR / LSD を意味のある量にするための MUST）

duration の出力は S を決めるので、量子化で銀行家丸めが 1 つ飛ぶだけで **latent 長そのものが
変わり**、波形長が構成間で食い違って SNR が定義できなくなる。そこで基準以外の構成は
**`f32` の S を外から固定して**回す（`irodori.pipeline_ref.run_case(frames_override=…)` —
`measure_quant_sbv2.py` が dump 側 `w_ceil` で時間グリッドを固定するのと同じ選択）。

**構成が自分で予測した S** は `predictedS` として別に残す。これが**この台本の主成果**で、
品質指標は「同じ時間グリッド上での比較」として読む — 実運用では発話長も動く。

## 直交性の門（恒真化の遮断）

役割の効き方は構造的に決まっているので、**「変わること」だけでなく「変わらないこと」も**
門にする（片側だけだと、丸めが素通りしても scope が漏れても緑になる）:

- `duration` は **S にしか効かない**（`log_frames` → 銀行家丸め → S。他の段は読まない）
  ⇒ グリッドを固定した `i8-duration-only` の z と波形は基準と**ビット一致**するのが正しい
- `codec` は **波形にしか効かない**（latent の後段）⇒ z はビット一致・波形は変わる
- `speaker` は**参照のあるケースだけ** z に効く（参照なしはホストのゼロ供給短絡で encoder が
  走らない — ADR 0048）⇒ `i8-speaker-only` は full で変わり・no-ref でビット一致、が緑の条件
- それ以外（backbone / projector / dit）は z が変わる
- `w8a8` は**重みだけの `i8-all` とも z が違う**（活性シムの素通り検出 — 素通りしたら
  z は `i8-all` とビット一致してしまい、基準との差だけ見ていると「効いている」と誤読する）

## 出力

    <--out>/<config>.json               構成 1 本ぶんの実測（S / forwards / 秒 / 品質指標）
    <--out>/<config>.<case>.safetensors z と波形（次の構成が基準として読む f32 の正本）
    <--out>/<config>.<case>.wav         聴き比べ用（16bit PCM・最終裁定は聴感 = ユーザー）
    <--out>/report.json                 全構成の突き合わせ（Markdown 表は stdout）

NOTE: 末尾トリム（`find_flattening_point`）と秒指定の切り出しは**通さない** — 構成ごとに
切る位置が動くと波形長が揃わず SNR が定義できない。実運用の長さは `predictedSeconds` が持つ。

NOTE: 指標の実装（`snr_db` / `log_spectral_distance` / `rel_rms`）は `measure_quant_sbv2.py`
からの**複製**。共有モジュール化は exporter の io 共通化波（Gemma 直前）の領分で、そこまでは
台本ごとに閉じた形を保つ（2026-08-12 網羅レビューの D3 裁定）。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from karume import act_quant
from karume.paths import OUTPUTS_ROOT, SERIES_ROOT

from . import export as ex
from . import patch
from . import pipeline_ref as ip
from .dacvae import export as dv

#: デモ・ベンチの生成物置き場（資産と分離する — `rm -rf outputs/demo` が系列を巻き込まない）。
DEFAULT_OUT = OUTPUTS_ROOT / "demo" / "irodori-quant-sim"

#: 恒真化防止の突合先（`irodori/pipeline_ref.py` が f32 系列へ書く full-loop golden）。
DEFAULT_GOLDEN_DIR = SERIES_ROOT / "irodori-v4-small" / "pipeline"

#: 基準構成の綴り（他構成はこの成果物を読んで比較する）。
BASE_CONFIG = "f32"

#: 役割 = グラフ境界（`irodori.export.load_*` と 1:1）+ コーデック。
ROLE_CODEC = "codec"
ROLES: tuple[str, ...] = (
    ex.TARGET_BACKBONE,
    ex.TARGET_TEXT_PROJ,
    ex.TARGET_CAPTION_PROJ,
    ex.TARGET_SPEAKER,
    ex.TARGET_DURATION,
    ex.TARGET_DIT,
    ROLE_CODEC,
)

#: latent（z）を動かしうる役割。`duration` は S にしか効かず、`codec` は latent の後段なので、
#: どちらもグリッドを固定した比較では z をビット単位で動かさない（モジュール docstring の
#: 「直交性の門」）。
LATENT_ROLES: frozenset[str] = frozenset(ROLES) - {ex.TARGET_DURATION, ROLE_CODEC}

#: **ケースに参照があるときだけ** z に効く役割。参照なしのケースでは speaker encoder が
#: 走らない（ホストのゼロ供給短絡 — ADR 0048）ので、`speaker` を丸めても z は不変が正しい
#: （実測 2026-08-12: `i8-speaker-only` の no-ref は z も波形も基準とビット一致）。
REFERENCE_ONLY_ROLES: frozenset[str] = frozenset({ex.TARGET_SPEAKER})

#: ケース名 → 参照音声の有無（直交性の門がケース条件で期待を割るための表）。
CASE_HAS_REFERENCE: Mapping[str, bool] = MappingProxyType(
    {case.name: case.reference is not None for case in ip.PIPELINE_CASES}
)


@dataclass(frozen=True)
class Recipe:
    """1 構成の量子化レシピ（`weight` が `None` なら丸めない）。"""

    weight: str | None
    roles: tuple[str, ...]
    #: DiT の `nn.Linear` 入力へ per-token i8 の fake-quant を掛けるか（w8a8 の活性側）。
    act_quant: bool = False


#: 活性シムの比較相手（**同じ重みで活性だけ素の**構成）。素通り検出はこの 1 本との差で見る。
WEIGHT_ONLY_BASE = "i8-all"

#: 主要 5 構成（聴き比べと配布形の裁定に載る側）。
CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        BASE_CONFIG: Recipe(None, ()),
        "f16": Recipe("f16", ROLES),
        WEIGHT_ONLY_BASE: Recipe("i8", ROLES),
        # ADR 0050 決定 6 の (i) 案 — duration だけ f32 据え置き。
        "i8-mixed": Recipe("i8", tuple(role for role in ROLES if role != ex.TARGET_DURATION)),
        # 配布形の `w8a8` 席 — 重みは `i8-all` と 1 バイトも変わらず、DiT の活性だけが i8。
        "w8a8": Recipe("i8", ROLES, act_quant=True),
    }
)

#: 直交分解（グラフ境界ごとに 1 本ずつ i8 にする — 劣化の帰属先を割る）。
DECOMPOSED_ROLES: tuple[str, ...] = (
    ex.TARGET_DIT,
    ex.TARGET_BACKBONE,
    ex.TARGET_SPEAKER,
    ex.TARGET_DURATION,
    ROLE_CODEC,
)
DIAGNOSTICS: Mapping[str, Recipe] = MappingProxyType(
    {f"i8-{role}-only": Recipe("i8", (role,)) for role in DECOMPOSED_ROLES}
)

#: 構成名 → レシピの全表。
RECIPES: Mapping[str, Recipe] = MappingProxyType({**CONFIGS, **DIAGNOSTICS})

#: LSD の STFT 設定（48kHz の解析グリッド — hop 512 ≈ 10.7ms）。
STFT_N_FFT = 2048
STFT_HOP = 512
#: 振幅の床（参照の最大振幅に対する相対値 = −100dB）。無音区間の対数が −inf へ落ちるのを
#: 防ぐためだけの下限で、両系列に**同じ床**を当てる（片側だけに当てると差が床に依存する）。
STFT_FLOOR = 1e-5

CASE_SUFFIX = ".safetensors"
REPORT_FILE = "report.json"


# ---- 指標（`measure_quant_sbv2.py` からの複製 — モジュール docstring の NOTE）----------


def snr_db(value: torch.Tensor, reference: torch.Tensor) -> float:
    """`10·log10(Σref² / Σ(value−ref)²)`（構成間の音質比較の主指標）。"""
    err = float((value - reference).double().square().sum())
    ref = float(reference.double().square().sum())
    if err == 0.0:
        return math.inf
    if ref == 0.0:
        return -math.inf
    return 10.0 * math.log10(ref / err)


def log_spectral_distance(value: torch.Tensor, reference: torch.Tensor) -> float:
    """対数スペクトル距離 `sqrt(mean((20log10|X| − 20log10|Y|)²))` [dB]。

    波形 SNR は位相が少しずれるだけで崩れる（flow-matching の軌道が変われば、聴感上同じ音でも
    残差が信号と同オーダーになる）。LSD は振幅スペクトルだけを見るので**位相ずれに鈍く**、
    「音色・帯域の壊れ方」を測る側の指標として併記する。どちらか一方では判断しない。
    """
    window = torch.hann_window(STFT_N_FFT)
    spectra = [
        torch.stft(
            wave_data, n_fft=STFT_N_FFT, hop_length=STFT_HOP, window=window, return_complex=True
        ).abs()
        for wave_data in (value, reference)
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


def write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """モノラル 16bit PCM で書く（`decodeWav` の逆 — 書きは ×32767 が Karume の規約）。"""
    clipped = audio.detach().reshape(-1).to(torch.float32).clamp(-1.0, 1.0)
    pcm = torch.floor(clipped * 32767.0 + 0.5).to(torch.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.numpy().tobytes())


# ---- 構成の組み立て ---------------------------------------------------------


def role_modules(modules: Mapping[str, nn.Module], roles: Sequence[str]) -> dict[str, nn.Module]:
    """役割 → その役割の丸め対象モジュール（`irodori.export.fake_quant` へ渡す形）。

    MUST: 3 つの norm を**消費側の役割**へ束ねる（`speaker_norm` は speaker グラフ・
    `text_norm` は duration グラフ・`caption_norm` は caption-proj グラフの内側 — 所在は
    ADR 0048 決定 1）。全役割を選ぶと `irodori.export.export_series` /
    `irodori.pipeline_ref.emit` が丸める 9 本と**同じ集合**になり、f16 構成が配布系列の
    対応物であることが集合として保証される。
    """
    grouped: dict[str, dict[str, nn.Module]] = {
        ex.TARGET_BACKBONE: {ex.TARGET_BACKBONE: modules[ex.TARGET_BACKBONE]},
        ex.TARGET_TEXT_PROJ: {ex.TARGET_TEXT_PROJ: modules[ex.TARGET_TEXT_PROJ]},
        ex.TARGET_CAPTION_PROJ: {
            ex.TARGET_CAPTION_PROJ: modules[ex.TARGET_CAPTION_PROJ],
            "caption_norm": modules["caption_norm"],
        },
        ex.TARGET_SPEAKER: {
            ex.TARGET_SPEAKER: modules[ex.TARGET_SPEAKER],
            "speaker_norm": modules["speaker_norm"],
        },
        ex.TARGET_DURATION: {
            ex.TARGET_DURATION: modules[ex.TARGET_DURATION],
            "text_norm": modules["text_norm"],
        },
        ex.TARGET_DIT: {ex.TARGET_DIT: modules[ex.TARGET_DIT]},
    }
    picked: dict[str, nn.Module] = {}
    for role in roles:
        picked.update(grouped.get(role, {}))
    return picked


class Loaded(NamedTuple):
    """`load_modules` の戻り（丸めを当てる前の素の一式）。"""

    source: Any
    modules: dict[str, nn.Module]
    config: Any
    text_config: Mapping[str, Any]
    model_config: Mapping[str, Any]


def load_modules(model_dir: Path, source_dir: Path) -> Loaded:
    """`irodori.pipeline_ref.emit` と同じ 9 本を組む（丸めはまだ当てない）。"""
    source = ex.IrodoriSource(source_dir)
    text_config, model_config = ex.read_configs(model_dir)
    state = load_file(str(model_dir / ex.MODEL_FILE))
    backbone = ex.load_backbone(source, state, text_config)
    hidden_size = int(backbone.hidden_size)
    config = source.model_config(model_config)
    modules = {
        ex.TARGET_BACKBONE: backbone,
        ex.TARGET_TEXT_PROJ: ex.load_projector(
            source,
            state,
            model_config,
            ex.TEXT_PROJ_PREFIX,
            hidden_size,
            int(model_config["text_dim"]),
        ),
        ex.TARGET_CAPTION_PROJ: ex.load_projector(
            source,
            state,
            model_config,
            ex.CAPTION_PROJ_PREFIX,
            hidden_size,
            int(model_config["caption_dim"]),
        ),
        ex.TARGET_SPEAKER: ex.load_speaker_encoder(source, state, config),
        ex.TARGET_DURATION: ex.load_duration_predictor(source, state, config),
        ex.TARGET_DIT: ex.load_dit(source, state, config, text_config),
        "speaker_norm": ex.load_rms_norm(
            source, state, ex.SPEAKER_NORM_PREFIX, int(config.speaker_dim), float(config.norm_eps)
        ),
        "text_norm": ex.load_rms_norm(
            source, state, ex.TEXT_NORM_PREFIX, int(config.text_dim), float(config.norm_eps)
        ),
        "caption_norm": ex.load_rms_norm(
            source,
            state,
            ex.CAPTION_NORM_PREFIX,
            int(config.caption_dim_resolved),
            float(config.norm_eps),
        ),
    }
    return Loaded(source, modules, config, text_config, model_config)


def build_graphs(
    modules: Mapping[str, nn.Module], config: Any, model_config: Mapping[str, Any]
) -> ip.HostGraphs:
    """`irodori.pipeline_ref.emit` と同じ 6 本のグラフラッパ。

    MUST: **パッチ適用後**でしか正しく動かない（実数形 RoPE 表を渡すため）。
    """
    speaker_max = ex.speaker_sym_max(model_config)
    return ip.HostGraphs(
        backbone=ex.BackboneGraph(modules[ex.TARGET_BACKBONE]),
        text_proj=ex.ProjectorGraph(modules[ex.TARGET_TEXT_PROJ]),
        caption_proj=ex.CaptionProjectorGraph(
            modules[ex.TARGET_CAPTION_PROJ], modules["caption_norm"]
        ),
        speaker=ex.SpeakerGraph(modules[ex.TARGET_SPEAKER], modules["speaker_norm"], speaker_max),
        duration=ex.DurationGraph(modules[ex.TARGET_DURATION], modules["text_norm"]),
        dit=ex.DitGraph(modules[ex.TARGET_DIT], ex.dit_sym_max(config)),
    )


def build_decoder(
    source_dir: Path, model_dir: Path, weight: str | None, quantize: bool
) -> tuple[nn.Module, int, str | None]:
    """コーデックの decode 経路（`DecoderGraph`）とサンプリング周波数・丸めの要約を返す。

    MUST（順序）: 丸めは `fold_weight_norm` / `lift_snake_alphas` の**後**（`irodori.dacvae.export`
    の `_fake_quant` の順序 MUST — ここは同じ前処理を同じ順で通してから同じ関数を呼ぶ）。
    切り詰めた `in_proj` は decode 経路に出てこないので組まない。
    """
    source = dv.DacvaeSource(source_dir)
    model = dv.load_codec(source, model_dir)
    dv.bypass_watermark(model.decoder)
    dv.fold_weight_norm(model)
    dv.lift_snake_alphas(source, model)
    report = dv._fake_quant(weight if quantize and weight is not None else "f32", model).report
    return (
        dv.DecoderGraph(model.quantizer.out_proj, model.decoder),
        int(model.sample_rate),
        report,
    )


# ---- 1 構成の実行 ------------------------------------------------------------


def golden_z(golden_dir: Path, case: str) -> torch.Tensor:
    """full-loop golden の最終 z（`f32` 構成の恒真化を遮断する唯一の突合先）。"""
    path = golden_dir / f"{ip.CASE_PREFIX}{case}{ip.CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"full-loop golden が無い: {path}"
            "（`uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref` で作る）"
        )
    return load_file(str(path))["z"].to(torch.float32)


def base_payload(out_dir: Path, case: str) -> dict[str, torch.Tensor]:
    """基準構成（`f32`）の z と波形。"""
    path = out_dir / f"{BASE_CONFIG}.{case}{CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"基準構成の成果物が無い: {path}"
            f"（先に `--config {BASE_CONFIG}` を回す — 比較の基準は f32）"
        )
    return load_file(str(path))


def weight_only_path(out_dir: Path, case: str) -> Path:
    """活性シムの比較相手（{@link WEIGHT_ONLY_BASE} — 同じ重み・活性は素）の成果物。

    存在検査だけを分けてあるのは、full-loop を回す**前**に確かめるため（回し切ってから
    「比較相手が無い」で落ちると数十分ぶんの計算が捨てになる）。
    """
    path = out_dir / f"{WEIGHT_ONLY_BASE}.{case}{CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"活性シムの比較相手が無い: {path}"
            f"（先に `--config {WEIGHT_ONLY_BASE}` を回す — 素通り検出は重みだけの構成との差）"
        )
    return path


def weight_only_payload(out_dir: Path, case: str) -> dict[str, torch.Tensor]:
    """{@link weight_only_path} の z と波形。"""
    return load_file(str(weight_only_path(out_dir, case)))


class LatentResult(NamedTuple):
    """{@link latent_stage} の戻り。`act_quant_linears` は活性シムを掛けた `nn.Linear` の本数。"""

    metas: dict[str, dict[str, Any]]
    latents: dict[str, torch.Tensor]
    reports: dict[str, str]
    act_quant_linears: int


def attach_dit_act_quant(
    recipe: Recipe, modules: Mapping[str, nn.Module]
) -> tuple[list[object], int]:
    """w8a8 構成で DiT の `nn.Linear` 入力へ per-token i8 の fake-quant を掛ける。

    MUST: **0 本は fail loudly** — 掛からないまま回すと `i8-all` と同じ数が `w8a8` の名前で
    レポートに載り、しかも品質は「良い」側に出る（活性量子化は誤差を増やすので、素通りは
    常に緑寄りの嘘になる）。掛ける先を DiT に限るのは配布形の席と揃えるため（`session` は
    `dit` の Session にだけ降りる）。
    """
    if not recipe.act_quant:
        return [], 0
    handles, attached = act_quant.attach_act_quant(modules[ex.TARGET_DIT])
    if attached == 0:
        act_quant.detach_act_quant(handles)
        raise SystemExit(
            "活性量子化シムを掛けた nn.Linear が 0 本"
            f"（`{ex.TARGET_DIT}` の Linear が適格条件 k % {act_quant.PACK_ALIGN} == 0 を"
            "満たしていない、またはモジュール構成が変わった）"
        )
    return handles, attached


def latent_stage(
    name: str,
    recipe: Recipe,
    args: argparse.Namespace,
    base_frames: Mapping[str, int] | None,
) -> LatentResult:
    """テキスト → latent（構成ぶんの丸めを当てて full-loop を回す）。"""
    from tokenizers import Tokenizer

    source, modules, config, text_config, model_config = load_modules(
        args.model_dir, args.source_dir
    )
    # MUST: 丸めは load の直後・golden の採取より前（`irodori.export.fake_quant` の順序 MUST）。
    # 対象が空になるのは `i8-codec-only`（latent 側の役割を 1 つも含まない）だけで、そのときは
    # 丸めそのものを呼ばない — `fake_quant` の「0 本は fail loudly」は正しい規律なので、
    # ここで例外にせず**呼ばない**ことで満たす。
    scoped = role_modules(modules, recipe.roles)
    quantized = (
        ex.fake_quant(recipe.weight, scoped)
        if recipe.weight is not None and scoped
        else ex.FakeQuantResult({}, {})
    )
    patch.apply_patches()
    # 活性シムは重みの丸めの**後**（重みは実行前に決まり、活性は実行時に決まる — 両者は独立
    # だが、報告の順序を「重み → 活性」で揃えると素通りの切り分けが 1 本の出力で済む）。
    handles, act_quant_linears = attach_dit_act_quant(recipe, modules)
    if act_quant_linears:
        print(f"[{name}] 活性量子化シム: nn.Linear {act_quant_linears} 本（dit）", flush=True)
    graphs = build_graphs(modules, config, model_config)
    speaker_max = ex.speaker_sym_max(model_config)
    caps = {
        "text": int(model_config["max_text_len"]),
        "speaker": speaker_max + 1,
        "caption": int(model_config["max_caption_len"]),
    }
    tokenizer = Tokenizer.from_file(str(args.model_dir / ex.TOKENIZER_FILE))

    metas: dict[str, dict[str, Any]] = {}
    latents: dict[str, torch.Tensor] = {}
    for case in ip.PIPELINE_CASES:
        started = time.perf_counter()
        tensors, meta = ip.run_case(
            case,
            graphs,
            source,
            modules[ex.TARGET_DURATION],
            tokenizer,
            text_config,
            model_config,
            config,
            caps,
            frames_override=None if base_frames is None else base_frames[case.name],
        )
        meta["elapsed"] = round(time.perf_counter() - started, 1)
        metas[case.name] = meta
        latents[case.name] = tensors["z"].detach().clone()
        print(
            f"[{name}/{case.name}] S={meta['S']} predicted={meta.get('predictedS', meta['S'])}"
            f" forwards={meta['forwards']} {meta['elapsed']:.0f}s",
            flush=True,
        )
    act_quant.detach_act_quant(handles)
    del graphs, modules, source
    gc.collect()
    return LatentResult(metas, latents, dict(quantized.reports), act_quant_linears)


def run_config(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """1 構成ぶんを走らせ、per-config の JSON を書いて返す。"""
    started = time.perf_counter()
    recipe = RECIPES[name]
    base_frames = None
    goldens = None
    if name == BASE_CONFIG:
        # MUST: 突合先の有無は**走らせる前**に確かめる（full-loop を回し切った後に
        # 「golden が無い」で落ちると、数十分ぶんの計算が捨てになる）。
        goldens = {case.name: golden_z(args.golden_dir, case.name) for case in ip.PIPELINE_CASES}
    else:
        base_json = args.out / f"{BASE_CONFIG}.json"
        if not base_json.is_file():
            raise SystemExit(
                f"基準構成の成果物が無い: {base_json}"
                f"（先に `--config {BASE_CONFIG}` を回す — 比較の基準は f32）"
            )
        base_meta = json.loads(base_json.read_text(encoding="utf-8"))
        base_frames = {case: entry["S"] for case, entry in base_meta["cases"].items()}
    if recipe.act_quant:
        # MUST: ここも**走らせる前**に確かめる（活性シムの素通り検出は重みだけの構成との差で
        # しか出せないので、無ければこの構成を回しても答えが出ない）。
        for case in ip.PIPELINE_CASES:
            weight_only_path(args.out, case.name)

    metas, latents, reports, act_quant_linears = latent_stage(name, recipe, args, base_frames)

    decoder, sample_rate, codec_report = build_decoder(
        args.codec_source_dir,
        args.codec_model_dir,
        recipe.weight,
        ROLE_CODEC in recipe.roles,
    )
    audios: dict[str, torch.Tensor] = {}
    for case, latent in latents.items():
        with torch.no_grad():
            audios[case] = decoder(latent).reshape(-1).detach().clone()
        print(f"[{name}/{case}] decoded {audios[case].shape[0]} samples", flush=True)
    del decoder
    gc.collect()

    args.out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for case, meta in metas.items():
        audio, latent = audios[case], latents[case]
        path = args.out / f"{name}.{case}.wav"
        write_wav(path, audio, sample_rate)
        save_file(
            {"z": latent.contiguous(), "audio": audio.contiguous()},
            str(args.out / f"{name}.{case}{CASE_SUFFIX}"),
        )
        entry: dict[str, Any] = {
            "S": meta["S"],
            "predictedS": meta.get("predictedS", meta["S"]),
            "forwards": meta["forwards"],
            "predictedFrames": meta["duration"]["predictedFrames"],
            "predictedSeconds": round(meta.get("predictedS", meta["S"]) / ex.CODEC_FRAME_RATE, 3),
            "audioSeconds": round(int(audio.shape[0]) / sample_rate, 3),
            "samples": int(audio.shape[0]),
            "zAbsMax": meta["zAbsMax"],
            "cfgEffectMaxAbs": meta["cfgEffectMaxAbs"],
            "elapsed": meta["elapsed"],
            "wav": str(path),
            "wavSha256": sha256(path.read_bytes()),
        }
        if name != BASE_CONFIG:
            base = base_payload(args.out, case)
            base_z, base_audio = base["z"].to(torch.float32), base["audio"].to(torch.float32)
            if latent.shape != base_z.shape or audio.shape != base_audio.shape:
                raise SystemExit(
                    f"{case}: 基準と形が違う（z {tuple(latent.shape)} vs"
                    f" {tuple(base_z.shape)} / 波形 {tuple(audio.shape)} vs"
                    f" {tuple(base_audio.shape)}）— 基準を焼き直したなら全構成を回し直す"
                )
            entry["vsBase"] = {
                "zBitEqual": bool(torch.equal(latent, base_z)),
                "zRelRms": rel_rms(latent, base_z),
                "zMaxAbsDiff": float((latent - base_z).abs().max()),
                "wavBitEqual": bool(torch.equal(audio, base_audio)),
                "wavSnrDb": snr_db(audio, base_audio),
                "wavLsdDb": log_spectral_distance(audio, base_audio),
                "wavRelRms": rel_rms(audio, base_audio),
                "wavMaxAbsDiff": float((audio - base_audio).abs().max()),
            }
        if recipe.act_quant:
            # 素通り検出の観測（評定は {@link run_gates} が持つ）— **重みが同じ構成との差**を
            # 見るので、ここが 0 なら活性シムが 1 本も効いていない。
            weight_only = weight_only_payload(args.out, case)
            other_z = weight_only["z"].to(torch.float32)
            other_audio = weight_only["audio"].to(torch.float32)
            if latent.shape != other_z.shape or audio.shape != other_audio.shape:
                raise SystemExit(
                    f"{case}: {WEIGHT_ONLY_BASE} と形が違う（z {tuple(latent.shape)} vs"
                    f" {tuple(other_z.shape)} / 波形 {tuple(audio.shape)} vs"
                    f" {tuple(other_audio.shape)}）— 時間グリッドの固定が効いていない"
                )
            entry["vsWeightOnly"] = {
                "config": WEIGHT_ONLY_BASE,
                "zBitEqual": bool(torch.equal(latent, other_z)),
                "zRelRms": rel_rms(latent, other_z),
                "zMaxAbsDiff": float((latent - other_z).abs().max()),
                "wavSnrDb": snr_db(audio, other_audio),
                "wavLsdDb": log_spectral_distance(audio, other_audio),
            }
        entries[case] = entry

    payload = {
        "config": name,
        "weight": recipe.weight,
        "roles": list(recipe.roles),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "sampleRate": sample_rate,
        "frameRate": ex.CODEC_FRAME_RATE,
        "fakeQuant": reports,
        "codecFakeQuant": codec_report,
        "actQuantLinears": act_quant_linears,
        "cases": entries,
        "gates": run_gates(name, recipe, latents, entries, goldens),
        "elapsed": round(time.perf_counter() - started, 1),
    }
    (args.out / f"{name}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


# ---- 検証ゲート -------------------------------------------------------------


def run_gates(
    name: str,
    recipe: Recipe,
    latents: Mapping[str, torch.Tensor],
    entries: Mapping[str, Mapping[str, Any]],
    goldens: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any]:
    """恒真化と scope 漏れの検出器（赤は最後にまとめて上げる）。

    ① `f32` の z が full-loop golden と**バイト一致**（この台本の経路が本物と同じ計算である
       ことの実証 — 外れたら測定値は全部意味を失う）
    ② 各構成の z / 波形が**変わるべきところだけ変わる**（モジュール docstring の直交性の門）。
       「変わること」だけを門にすると丸めの素通りしか捕まらず、「変わらないこと」だけを
       門にすると scope 漏れしか捕まらない — 両方を期待値表から出す
    ③ 活性シムを持つ構成は**重みだけの構成とも z が違う**（素通り検出）。基準との差だけでは
       重みの丸めで説明が付いてしまい、活性シムが 1 本も効いていなくても緑になる
    """
    gates: dict[str, Any] = {}
    failures: list[str] = []
    if goldens is not None:
        matched = {
            case: bool(torch.equal(latent, goldens[case])) for case, latent in latents.items()
        }
        gates["z_bit_exact_vs_golden"] = {"cases": matched}
        failures += [
            f"{case}: f32 の z が full-loop golden とバイト一致しない（経路の恒真化検査）"
            for case, ok in matched.items()
            if not ok
        ]
    else:
        observed = {
            case: {
                "zChanged": not entry["vsBase"]["zBitEqual"],
                "wavChanged": not entry["vsBase"]["wavBitEqual"],
            }
            for case, entry in entries.items()
        }
        expectations: dict[str, dict[str, bool]] = {}
        for case in observed:
            # 期待は**ケース条件**で割る — 参照なしのケースでは speaker 経路が走らない
            # （{@link REFERENCE_ONLY_ROLES}）ので、その分の役割を期待から外す。
            effective = set(recipe.roles) & LATENT_ROLES
            if not CASE_HAS_REFERENCE[case]:
                effective -= REFERENCE_ONLY_ROLES
            expect_z = bool(effective)
            expectations[case] = {
                "zChanged": expect_z,
                "wavChanged": expect_z or ROLE_CODEC in recipe.roles,
            }
        gates["orthogonality"] = {"expected": expectations, "observed": observed}
        for case, seen in observed.items():
            expect_z = expectations[case]["zChanged"]
            expect_wav = expectations[case]["wavChanged"]
            if seen["zChanged"] != expect_z:
                failures.append(
                    f"{case}: z が「{'変わる' if expect_z else '変わらない'}」期待に反した"
                    f"（役割 {list(recipe.roles)} — 丸めの素通りか scope 漏れ）"
                )
            if seen["wavChanged"] != expect_wav:
                failures.append(
                    f"{case}: 波形が「{'変わる' if expect_wav else '変わらない'}」期待に反した"
                    f"（役割 {list(recipe.roles)}）"
                )
        if recipe.act_quant:
            moved = {
                case: not entry["vsWeightOnly"]["zBitEqual"] for case, entry in entries.items()
            }
            gates["act_quant_changes_the_latent"] = {"versus": WEIGHT_ONLY_BASE, "cases": moved}
            failures += [
                f"{case}: z が {WEIGHT_ONLY_BASE} とビット一致した（活性シムが素通りしている"
                " — 重みだけの構成と同じ数を w8a8 の名前で測っている）"
                for case, ok in moved.items()
                if not ok
            ]
    if recipe.weight is not None:
        gates["quantized_roles"] = list(recipe.roles)
    gates["failures"] = failures
    return gates


# ---- レポート ---------------------------------------------------------------


def collect(out_dir: Path) -> dict[str, dict[str, Any]]:
    """`--out` に残っている per-config の JSON を構成順に集める。

    MUST: 直交性の門は**ここで評定し直す**（保存 JSON の `gates` を再掲しない）— 門の期待は
    コードが正本で、測定後に期待側の誤りを直したとき、保存済みの評定が古い期待のまま
    レポートへ蘇るため（実例: speaker のケース条件 — no-ref はゼロ短絡で不変が正しい）。
    基準 `f32` の golden ビット一致だけはテンソル比較（測定時にしかできない）なので保存値を
    使う。観測（`vsBase` / `vsWeightOnly` のビット等値）は測定の事実で、評定と違い保存してよい側。
    """
    found: dict[str, dict[str, Any]] = {}
    for name in RECIPES:
        path = out_dir / f"{name}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if name != BASE_CONFIG:
                payload["gates"] = run_gates(name, RECIPES[name], {}, payload["cases"], None)
            found[name] = payload
    return found


def drift_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """**最上段の表** — S が動いたか（混成表の裁定はこの 1 枚で決まる）。"""
    header = ["config", "case", "predicted S", "基準 S", "S ドリフト", "forwards", "発話長 (s)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    base = collected.get(BASE_CONFIG, {}).get("cases", {})
    for name, payload in collected.items():
        for case, entry in payload["cases"].items():
            reference = base.get(case, {}).get("predictedS")
            drift = "—" if reference is None else f"{entry['predictedS'] - reference:+d}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        case,
                        str(entry["predictedS"]),
                        "—" if reference is None else str(reference),
                        "一致" if drift == "+0" else drift,
                        str(entry["forwards"]),
                        f"{entry['predictedSeconds']:.2f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def quality_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """品質（**基準と同じ時間グリッド上**での比較）。"""
    header = ["config", "case", "latent relRMS", "latent max|Δ|", "SNR (dB)", "LSD (dB)", "秒"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for name, payload in collected.items():
        for case, entry in payload["cases"].items():
            versus = entry.get("vsBase")
            if versus is None:
                row = ["0（基準）", "0", "inf（基準）", "0（基準）"]
            else:
                snr = versus["wavSnrDb"]
                row = [
                    f"{versus['zRelRms']:.4e}",
                    f"{versus['zMaxAbsDiff']:.4e}",
                    "inf" if math.isinf(snr) else f"{snr:.2f}",
                    f"{versus['wavLsdDb']:.2f}",
                ]
            lines.append("| " + " | ".join([name, case, *row, f"{entry['elapsed']:.0f}"]) + " |")
    return "\n".join(lines)


def build_report(collected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/exporter/irodori/measure_quant.py",
        "torch": torch.__version__,
        "base": BASE_CONFIG,
        "method": {
            "time_grid": "基準以外は f32 の S を外から固定（波形長を揃えて SNR を定義可能に"
            "する）。構成自身の予測は predictedS",
            "roles": list(ROLES),
            "latent_roles": sorted(LATENT_ROLES),
            "metrics": "SNR は波形の残差（位相ずれに弱い）/ LSD は対数振幅スペクトルの距離"
            f"（n_fft={STFT_N_FFT} hop={STFT_HOP}・床 −100dB・位相ずれに鈍い）。両方で読む",
            "trim": "末尾トリムと秒切り出しは通さない（構成間で長さを揃えるため）",
            "act_quant": f"w8a8 は DiT の nn.Linear 入力へ per-token i8 の fake-quant を掛ける"
            f"（karume.act_quant — ランタイム quantize-rows.ts の鏡像）。素通り検出は"
            f" {WEIGHT_ONLY_BASE} との z ビット等値",
            "verdict": "最終裁定は聴感（ユーザー）— WAV は同一テキスト・同一 seed で並ぶ",
        },
        "configs": dict(collected),
        "failures": {
            name: payload["gates"]["failures"]
            for name, payload in collected.items()
            if payload["gates"]["failures"]
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--codec-model-dir", type=Path, default=dv.DEFAULT_MODEL_DIR)
    parser.add_argument("--codec-source-dir", type=Path, default=dv.DEFAULT_SOURCE_DIR)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--config",
        choices=tuple(RECIPES),
        default=None,
        help="走らせる構成（1 実行 = 1 構成）。省略すると実測はせず、"
        f"--out に残っている成果物から {REPORT_FILE} と表を組み直すだけ",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        payload = run_config(args.config, args)
        print(f"[{args.config}] {payload['elapsed']:.0f}s → {args.out / f'{args.config}.json'}")

    collected = collect(args.out)
    if not collected:
        raise SystemExit(f"{args.out} に構成の成果物が 1 つも無い（--config で走らせる）")
    report = build_report(collected)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print("S ドリフト（混成表の裁定はこの表）:")
    print(drift_table(collected))
    print()
    print("品質（基準と同じ時間グリッド上）:")
    print(quality_table(collected))
    print()
    print(f"report → {args.out / REPORT_FILE}")
    if report["failures"]:
        raise AssertionError(
            "検証ゲートが赤: "
            + " / ".join(
                f"{name}: {'・'.join(items)}" for name, items in report["failures"].items()
            )
        )
    print("gates: all green")


if __name__ == "__main__":
    main()
