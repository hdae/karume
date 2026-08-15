r"""DACVAE を挟む**ホスト側前処理**の数の正（参照音声の正規化・reflect pad・末尾トリム）。

`irodori/dacvae/export.py` がグラフ（G6 decoder / G7 encoder）を出すのに対し、こちらが出すのは
グラフの**外側**に残る段の参照値。W4 のホスト実装（TS の codec 統合）が突き合わせる材料で、
値は上流実装（`irodori_tts.codec` / `irodori_tts.inference_runtime`）を**呼んで**採る。

    cd tools/export-recipes
    uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' \
        python -m irodori.dacvae.host

`irodori_tts` を import するので transformers のピンが要る（`irodori/export.py` と同じ理由）。
`descript-audiotools` は LUFS 測定の実経路（ITU-R BS.1770-4 の K-weighting IIR）、`einops` は
`dacvae/__init__.py` が引くためだけに要る。

出力（既定 `outputs/series/dacvae-32dim/host/`・`.gitignore` 配下）:

    meta.json                ケースごとの実測スカラー（LUFS・利得・長さ）と常設門の実測値
    case.<name>.safetensors  生波形 / 正規化後 / reflect pad 後 / latent
    trim.safetensors         末尾トリムの入力 latent（実 z 2 本 + 末尾無音の合成 1 本）

## ホストが担う段（このファイルが写している唯一のもの）

1. **WAV の読み出し**（int16 PCM → f32）。規約は **/32768**（`_wav_scale_evidence` が
   int16 の両端で毎回実測する — この 1 本で TS 側のデコーダの割り算が決まる）
2. **LUFS 正規化**（`normalize_db = -16` のとき）→ `ensure_max_of_audio`。
   利得は `exp((target − ref_db) · GAIN_FACTOR)`、続く peak 利得は `peak > 1` のときだけ `1/peak`
3. **peak 安全スケール**（`normalize_db = None` かつ `ensure_max = True` のとき）。
   上流はこちらだけ `1.0 / float(peak)` を掛ける別経路（`codec.py` の `encode_waveform`）
4. **reflect pad**（`DACVAE._pad` — hop 1920 の倍数へ右詰め）
5. **末尾トリム**（`find_flattening_point` — 生成 latent の末尾が平坦かつ 0 近傍になる位置）

MUST: 2〜5 は**式を写さず上流を呼ぶ**（`_normalize_loudness` / `_pad` /
`find_flattening_point`）。写しているのは 1 と、正規化を「LUFS 利得 × peak 利得」へ**分解**した
表現だけで、後者は `_decomposition_evidence` が「記録したスカラーで正規化後の波形が
ビット一致で再現できる」ことを毎回実測する（TS 側は分解された形しか使えない — audiotools の
IIR を移植せずに済ませるには ref_db を golden から受け取るしかない）。

## 何を門にするか

emit の前に全て実測し、1 つでも外れたら**何も書かない**（ADR 0005 の fail loudly）:

- **WAV の正規化規約**が /32768 ちょうどであること（int16 の下端 −32768 が厳密に −1.0）
- **分解の再現**（上記）と、**上流 `_normalize_loudness` とのビット一致**
- **latent が上流 `encode_waveform` とビット一致**すること（前処理 → グラフの全経路）
- 各ケースが**狙った枝を実際に踏んでいる**こと（恒真化の遮断）:
  0.5 秒未満のケースが本当に 0.5 秒未満 / peak > 1 のケースで peak 利得が 1 未満 /
  ちょうど hop の倍数のケースで pad が 0 サンプル・端数のケースで正
- **末尾トリム**が実 latent では切らず（point == 長さ）、末尾を 0 にした合成 latent では
  ちょうどその位置を返すこと
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file

from _shared.paths import SERIES_ROOT
from karume.convert import normalize_boundary_tensor

from .. import export as ir
from . import export as ex

META_FILE = "meta.json"
CASE_PREFIX = "case."
CASE_SUFFIX = ".safetensors"
TRIM_FILE = "trim.safetensors"

#: `ensure_max` の既定（`inference_runtime.SamplingRequest.ref_ensure_max`）。**噛むのは
#: `normalize_db = None` のときだけ**で、LUFS 正規化側は audiotools の
#: `ensure_max_of_audio()` が内側で同じことをする（`codec.py` のコメントどおり）。
ENSURE_MAX = True

#: ホスト前処理の golden ケース。
#: `(名前, 長さ〈秒・None = 全長〉, 追加サンプル, 振幅倍率, 目標 LUFS〈None = 正規化しない〉)`。
#:
#: 参照音声は 48kHz mono 7.6 秒で、長さが hop 1920 のちょうど倍数（190 フレーム）。そのため
#: 既定ケースだけでは reflect pad が 0 サンプルの恒等になるので、端数（+777 サンプル）の
#: ケースを別に置く。4 本目は 0.5 秒未満で、audiotools が**測定のためにゼロ pad する**枝
#: （`LoudnessMixin.loudness`）を踏む。3 本目は振幅 3 倍で peak > 1 を作る。
HOST_CASES: tuple[tuple[str, float | None, int, float, float | None], ...] = (
    ("ref-default", None, 0, 1.0, ex.REFERENCE_NORMALIZE_DB),
    ("ref-plain", 3.0, 0, 1.0, None),
    ("ref-hot", 3.0, 0, 3.0, None),
    ("ref-short", 0.4, 0, 1.0, ex.REFERENCE_NORMALIZE_DB),
    ("ref-odd", 3.0, 777, 1.0, ex.REFERENCE_NORMALIZE_DB),
)

#: 末尾トリムの golden ケース `(名前, 供給元, 末尾を 0 にするフレーム数)`。
#:
#: 実 latent 2 本は**切らない**側（`find_flattening_point` が長さそのものを返す）の実測で、
#: 合成 1 本は末尾 40 フレームを 0 にして「ちょうど 121 を返す」ことを見る。前者だけだと
#: 「常に長さを返す実装」でも通ってしまう（恒真化の遮断）。
TRIM_CASES: tuple[tuple[str, str, int], ...] = (
    ("z-full", "full", 0),
    ("z-no-ref", "no-ref", 0),
    ("z-silent-tail", "full", 40),
)

#: int16 PCM の正規化規約を測る値（両端と 0 近傍）。−32768 が厳密に −1.0 へ写れば /32768。
WAV_SCALE_PROBE = (-32768, -32767, -1, 0, 1, 16384, 32767)
INT16_DIVISOR = 32768


class IrodoriHostSource(NamedTuple):
    """`irodori_tts` から借りるホスト側の 2 つ（`sys.path` 追加で import する）。

    MUST: import は**実装 clone から**行う（写しを台本に持たない）。`irodori_tts/__init__.py`
    が transformers を引くので、ピン付きの `--with` が無い環境では import 時に落ちる。
    """

    codec_cls: Any
    find_flattening_point: Any

    @classmethod
    def load(cls, source_dir: Path) -> IrodoriHostSource:
        codec_py = source_dir / "irodori_tts" / "codec.py"
        if not codec_py.is_file():
            raise SystemExit(
                f"Irodori の実装が見つからない: {codec_py}"
                "（`git clone https://github.com/Aratako/Irodori-TTS` の展開先を"
                " --irodori-dir に指定する）"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from irodori_tts.codec import DACVAECodec
        from irodori_tts.inference_runtime import find_flattening_point

        return cls(codec_cls=DACVAECodec, find_flattening_point=find_flattening_point)


def _wav_scale_evidence(sample_rate: int) -> dict[str, Any]:
    """WAV リーダの int16 → f32 規約を合成ファイルで実測する（門）。

    MUST: 主張ではなく実測にする — /32767 の実装と /32768 の実装は、実音声では相対差
    3e-5 しか出ず、golden を突き合わせるだけでは tolerance に埋もれて見えない。
    """
    import numpy as np
    import soundfile as sf

    values = np.array(WAV_SCALE_PROBE, dtype=np.int16)
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "int16-probe.wav"
        sf.write(str(path), values, sample_rate, subtype="PCM_16")
        decoded, decoded_rate = ex.read_wav(path)
    if decoded_rate != sample_rate:
        raise AssertionError(f"probe の sr が {decoded_rate}（書いた値 {sample_rate} と違う）")
    expected = torch.tensor([value / INT16_DIVISOR for value in WAV_SCALE_PROBE])
    if not torch.equal(decoded, expected.to(decoded.dtype)):
        raise AssertionError(
            f"WAV の int16 正規化が /{INT16_DIVISOR} でない"
            f"（{WAV_SCALE_PROBE} → {[float(v) for v in decoded]}）"
        )
    return {
        "loader": "soundfile",
        "int16Divisor": INT16_DIVISOR,
        "probeInt16": list(WAV_SCALE_PROBE),
        "probeFloat": [float(value) for value in decoded],
    }


def _slice_reference(
    wav: torch.Tensor, sample_rate: int, seconds: float | None, extra: int, scale: float
) -> torch.Tensor:
    """参照音声から 1 ケースぶんの生波形を切り出す（長さと振幅だけを変える）。"""
    length = wav.numel() if seconds is None else int(seconds * sample_rate) + extra
    if not 0 < length <= wav.numel():
        raise SystemExit(f"切り出し長 {length} が参照音声の {wav.numel()} サンプルに収まらない")
    return (wav[:length] * scale).contiguous()


def _normalize_decomposed(
    raw: torch.Tensor, sample_rate: int, target_db: float | None
) -> tuple[torch.Tensor, dict[str, float]]:
    """正規化を「LUFS 利得 × peak 利得」へ分解して返す（`(正規化後, スカラー)`）。

    `target_db` が `None` のときは上流 `encode_waveform` の peak 安全スケールだけの経路
    （`1.0 / float(peak)`）。それ以外は audiotools の `normalize` → `ensure_max_of_audio` で、
    利得は `exp((target − ref_db) · GAIN_FACTOR)`。**分解が正しいこと自体は
    `_decomposition_evidence` が実測する**（ここは記録のための式）。
    """
    from audiotools import AudioSignal

    signal = AudioSignal(raw.unsqueeze(0).unsqueeze(0), int(sample_rate))
    reference_db = float(signal.loudness())
    if target_db is None:
        loudness_gain = 1.0
    else:
        delta = torch.tensor(target_db - reference_db)
        loudness_gain = float(torch.exp(delta * AudioSignal.GAIN_FACTOR))
    scaled = raw * loudness_gain
    peak = float(scaled.abs().max())
    peak_gain = (1.0 / peak) if peak > 1.0 else 1.0
    return (scaled * peak_gain).contiguous(), {
        "refDb": reference_db,
        "loudnessGain": loudness_gain,
        "peakBeforeScale": peak,
        "peakGain": peak_gain,
    }


def _decomposition_evidence(
    raw: torch.Tensor,
    decomposed: torch.Tensor,
    scalars: Mapping[str, float],
    where: str,
) -> float:
    """記録したスカラーだけで正規化後の波形が再現できることを実測する（門）。

    TS 側は audiotools の IIR を持たない（`refDb` を golden から受け取る）ので、**この 2 つの
    掛け算が正規化の全て**であることがホスト移植の前提そのものになる。
    """
    rebuilt = raw * scalars["loudnessGain"] * scalars["peakGain"]
    if not torch.equal(rebuilt, decomposed):
        raise AssertionError(
            f"{where}: 利得 2 本で正規化後の波形を再現できない"
            f"（最大絶対差 {float((rebuilt - decomposed).abs().max())}）"
        )
    return 0.0


def _upstream_normalize_evidence(
    codec_cls: Any, raw: torch.Tensor, sample_rate: int, target_db: float, normalized: torch.Tensor
) -> float:
    """上流 `DACVAECodec._normalize_loudness` とのビット一致（門）。"""
    upstream = codec_cls._normalize_loudness(raw, sample_rate=sample_rate, target_db=target_db)
    if not torch.equal(upstream, normalized):
        raise AssertionError(
            "LUFS 正規化が上流と一致しない"
            f"（最大絶対差 {float((upstream - normalized).abs().max())}）"
        )
    return 0.0


def _branch_evidence(cases: Mapping[str, Mapping[str, Any]], hop: int) -> dict[str, Any]:
    """各ケースが狙った枝を実際に踏んでいることを実測する（恒真化の遮断）。

    MUST: ケース表を増やしても「どれも既定経路を通っただけ」で緑になる形にしない。
    見るのは ① 0.5 秒未満のケースが本当に短い ② peak > 1 のケースで peak 利得が 1 未満・
    peak ≤ 1 のケースで厳密に 1 ③ hop の倍数のケースで pad が 0・端数のケースで正、の 3 点。
    """
    short = cases["ref-short"]
    if short["seconds"] >= 0.5:
        raise AssertionError(
            f"ref-short が {short['seconds']} 秒 — audiotools のゼロ pad 測定枝を踏んでいない"
        )
    if short["refDb"] == cases["ref-default"]["refDb"]:
        raise AssertionError("ref-short と ref-default の LUFS が同値 — 測定が長さに依っていない")
    if not cases["ref-hot"]["peakGain"] < 1.0:
        raise AssertionError(
            f"ref-hot の peak 利得が {cases['ref-hot']['peakGain']} — peak > 1 の枝を踏んでいない"
        )
    if cases["ref-plain"]["peakGain"] != 1.0:
        raise AssertionError(
            f"ref-plain の peak 利得が {cases['ref-plain']['peakGain']} — peak ≤ 1 で縮んでいる"
        )
    if cases["ref-default"]["padSamples"] != 0:
        raise AssertionError(
            f"ref-default の pad が {cases['ref-default']['padSamples']} サンプル"
            f" — 参照音声の長さが hop {hop} の倍数でなくなっている"
        )
    if not 0 < cases["ref-odd"]["padSamples"] < hop:
        raise AssertionError(
            f"ref-odd の pad が {cases['ref-odd']['padSamples']} サンプル — 端数の枝を踏んでいない"
        )
    return {
        "shortSeconds": short["seconds"],
        "hotPeakGain": cases["ref-hot"]["peakGain"],
        "oddPadSamples": cases["ref-odd"]["padSamples"],
    }


def _trim_evidence(
    find_flattening_point: Any, latents: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, int]]:
    """末尾トリムの `{z → 位置}` を上流関数で採り、恒真でないことを確かめる（門）。

    実 latent は**切らない**（位置 = 長さ）、末尾を 0 にした合成は**ちょうどその位置**を返す。
    片方だけだと「常に長さを返す」「常に 0 を返す」実装が通ってしまう。
    """
    points: dict[str, dict[str, int]] = {}
    for name, source, silence in TRIM_CASES:
        latent = latents[name]
        frames = int(latent.shape[1])
        point = int(find_flattening_point(latent[0]))
        expected = frames - silence
        if point != expected:
            raise AssertionError(
                f"{name}: find_flattening_point が {point}（期待 {expected}）"
                " — 末尾トリムの前提（実 latent は切らない / 無音は切る）が崩れている"
            )
        points[name] = {"source": source, "frames": frames, "silentTail": silence, "point": point}
    return points


def _trim_latents(latent_dir: Path) -> dict[str, torch.Tensor]:
    """{@link TRIM_CASES} の入力 latent を組む（実 z の末尾を 0 にした合成を含む）。"""
    sources: dict[str, torch.Tensor] = {}
    for name in {source for _n, source, _s in TRIM_CASES}:
        path = latent_dir / f"{ex.LATENT_CASE_PREFIX}{name}{CASE_SUFFIX}"
        if not path.is_file():
            raise SystemExit(
                f"実 latent が無い: {path}"
                "（`uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref` で作る）"
            )
        sources[name] = load_file(str(path))[ex.LATENT_KEY].to(torch.float32)
    latents: dict[str, torch.Tensor] = {}
    for name, source, silence in TRIM_CASES:
        latent = sources[source].clone()
        if silence:
            if silence >= int(latent.shape[1]):
                raise SystemExit(f"{name}: 無音 {silence} フレームが latent 長以上")
            latent[:, -silence:] = 0.0
        latents[name] = latent.contiguous()
    return latents


def emit(
    model_dir: Path,
    source_dir: Path,
    out_dir: Path,
    *,
    latent_dir: Path = ex.DEFAULT_LATENT_DIR,
    reference_wav: Path = ex.DEFAULT_REFERENCE_WAV,
    irodori_dir: Path = ir.DEFAULT_SOURCE_DIR,
) -> dict[str, Any]:
    """ホスト前処理の golden を書き、要約を返す（門に落ちたら 1 バイトも書かない）。"""
    upstream = IrodoriHostSource.load(irodori_dir)
    DACVAECodec = upstream.codec_cls  # noqa: N806 — 上流のクラス名をそのまま使う

    source = ex.DacvaeSource(source_dir)
    model = ex.load_codec(source, model_dir)
    ex.bypass_watermark(model.decoder)
    sample_rate = int(model.sample_rate)
    hop = ex.hop_length(model)
    wav, wav_rate = ex.read_wav(reference_wav)
    if wav_rate != sample_rate:
        raise SystemExit(
            f"{reference_wav.name}: {wav_rate}Hz はコーデックの {sample_rate}Hz と違う"
            "（リサンプルはこの台本では通さない）"
        )
    wav_scale = _wav_scale_evidence(sample_rate)

    # 上流の正本経路（`encode_waveform`）は **weight_norm を畳む前**に通す — 畳んだ後で採ると
    # 「畳んだ後どうし」の比較になり、latent の突合が恒真化する
    # （`irodori.dacvae.export` と同じ規律）。
    codec = DACVAECodec(
        model=model,
        sample_rate=sample_rate,
        latent_dim=int(model.quantizer.in_proj.out_channels) // 2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=ex.REFERENCE_NORMALIZE_DB,
    )
    raws = {
        name: _slice_reference(wav, sample_rate, seconds, extra, scale)
        for name, seconds, extra, scale, _db in HOST_CASES
    }
    upstream_latents = {
        name: codec.encode_waveform(
            raws[name].reshape(1, 1, -1), sample_rate, normalize_db=target_db, ensure_max=ENSURE_MAX
        ).clone()
        for name, _sec, _extra, _scale, target_db in HOST_CASES
    }

    cases: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, torch.Tensor]] = {}
    normalize_evidence: dict[str, float] = {}
    decomposition_evidence: dict[str, float] = {}
    for name, _seconds, _extra, scale, target_db in HOST_CASES:
        raw = raws[name]
        normalized, scalars = _normalize_decomposed(raw, sample_rate, target_db)
        decomposition_evidence[name] = _decomposition_evidence(raw, normalized, scalars, name)
        if target_db is not None:
            normalize_evidence[name] = _upstream_normalize_evidence(
                DACVAECodec, raw, sample_rate, target_db, normalized
            )
        with torch.no_grad():
            padded = model._pad(normalized.reshape(1, 1, -1))
        frames = int(padded.shape[-1]) // hop
        payloads[name] = {
            "raw": raw,
            "normalized": normalized,
            # MUST: clone する — pad が 0 サンプルのケース（`ref-default`）では `_pad` が入力を
            # そのまま返すので、記憶域を共有したまま safetensors へ渡すと保存が落ちる。
            "padded": padded.reshape(-1).clone(),
            "latent": upstream_latents[name],
        }
        cases[name] = {
            "samples": int(raw.numel()),
            "seconds": float(raw.numel()) / sample_rate,
            "amplitudeScale": scale,
            "normalizeDb": target_db,
            "ensureMax": ENSURE_MAX,
            **scalars,
            "paddedSamples": int(padded.shape[-1]),
            "padSamples": int(padded.shape[-1]) - int(raw.numel()),
            "frames": frames,
        }
    branch = _branch_evidence(cases, hop)

    # ---- 前処理を当てて、ホスト鎖（正規化 → pad → グラフ）が上流とビット一致することを実測 ----
    ex.fold_weight_norm(model)
    ex.lift_snake_alphas(source, model)
    encoder_graph = ex.EncoderGraph(model.encoder, ex.truncated_in_proj(model.quantizer.in_proj))
    chain_evidence: dict[str, float] = {}
    for name in cases:
        framed = payloads[name]["padded"].reshape(1, cases[name]["frames"], hop)
        with torch.no_grad():
            chain = encoder_graph(framed)
        if not torch.equal(chain, upstream_latents[name]):
            raise AssertionError(
                f"{name}: ホスト鎖の latent が上流 encode_waveform と一致しない"
                f"（最大絶対差 {float((chain - upstream_latents[name]).abs().max())}）"
            )
        chain_evidence[name] = 0.0

    trim_latents = _trim_latents(latent_dir)
    trim = _trim_evidence(upstream.find_flattening_point, trim_latents)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for name, tensors in payloads.items():
        path = out_dir / f"{CASE_PREFIX}{name}{CASE_SUFFIX}"
        save_file(
            {
                key: normalize_boundary_tensor(value.contiguous(), f"{name} の '{key}'")
                for key, value in tensors.items()
            },
            str(path),
        )
        written[path.name] = path.stat().st_size
    save_file(
        {name: value.contiguous() for name, value in trim_latents.items()},
        str(out_dir / TRIM_FILE),
    )
    written[TRIM_FILE] = (out_dir / TRIM_FILE).stat().st_size

    meta_payload = {
        "sourceRepo": ex.SOURCE_REPO,
        "sourceCommit": ex.SOURCE_COMMIT,
        "referenceWav": reference_wav.name,
        "sampleRate": sample_rate,
        "hopLength": hop,
        "frameRate": ex.frame_rate(model),
        "gainFactor": _gain_factor(),
        "wavScale": wav_scale,
        "cases": cases,
        "trim": trim,
        "branchEvidence": branch,
        "upstreamNormalizeMaxAbs": normalize_evidence,
        "decompositionMaxAbs": decomposition_evidence,
        "upstreamLatentMaxAbs": chain_evidence,
    }
    (out_dir / META_FILE).write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    written[META_FILE] = (out_dir / META_FILE).stat().st_size
    return {"dir": str(out_dir), "bytes": written, **meta_payload}


def _gain_factor() -> float:
    """audiotools の利得換算係数（`ln(10)/20`）。**定数を写さずライブラリから読む**。"""
    from audiotools import AudioSignal

    return float(AudioSignal.GAIN_FACTOR)


def default_out_dir(model_dir: Path) -> Path:
    """既定の置き場（`outputs/series/<実重みのディレクトリ名>/host/`）。"""
    return SERIES_ROOT / model_dir.name / "host"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--irodori-dir", type=Path, default=ir.DEFAULT_SOURCE_DIR)
    parser.add_argument("--latent-dir", type=Path, default=ex.DEFAULT_LATENT_DIR)
    parser.add_argument("--reference-wav", type=Path, default=ex.DEFAULT_REFERENCE_WAV)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = default_out_dir(args.model_dir) if args.out is None else args.out
    summary = emit(
        args.model_dir,
        args.source_dir,
        out_dir,
        latent_dir=args.latent_dir,
        reference_wav=args.reference_wav,
        irodori_dir=args.irodori_dir,
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
