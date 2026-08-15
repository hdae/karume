r"""Irodori-TTS v4 の**ホスト側アルゴリズム**の数の正（full-loop latent golden）。

`irodori/export.py` がグラフを、`irodori/tokenizer_ref.py` がテキスト前処理の資産を出すのに対し、
こちらが出すのは「グラフを 6 本回して latent を作るまで」の**最終値**。W3 のホスト実装
（TS のパイプライン）が突き合わせる統合門の参照値そのもので、値は
**export したグラフ基準**（= `irodori/export.py` の Graph ラッパ = eager 同値実装）で採る。

    cd tools/exporter
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype f16
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i8

`--dtype f16` / `--dtype i8` は**別系列**（`outputs/series/irodori-v4-small-{f16,i8}/pipeline/`）
へ書く。丸めは `irodori.export.fake_quant` を load 直後に当てるので、golden も上流突合の参照も
**同じ丸めた重み**で計算される（ADR 0018 / 0019 / 0027 / 0050 — 系列ごとに golden を焼き直す形）。

出力（既定 `outputs/series/irodori-v4-small{,-f16,-i8}/pipeline/`・`.gitignore` 配下）:

    meta.json                入力テキスト / caption / S / seed / CFG scale / forward 数 /
                             t スケジュール / 供給源、および常設門の実測値
    case.<name>.safetensors  参照 latent・初期ノイズ・token id・最終 z
    t-embed.safetensors      40 step で実際に使う t と、そのホスト式 `t_embed` 表 `[40,512]`

## ホストが担う段（このファイルが写している唯一のもの）

1. 前処理 → tokenize → BOS 前置（body は `max_*_len - 1` で切る）。前処理は種別で違い、
   text は `normalize_text` → strip・caption は **strip のみ**（上流 `_synthesize` の綴り）
2. `backbone` → `text-proj` / `caption-proj`（**列を詰めたまま**呼ぶ = 静的方式）
3. 参照 latent を `speaker_patch_size` で patch → `speaker` → 平均トークン前置
4. `duration` の 5 入力を組む（`speaker_vec` = 平均トークン / `caption_vec` =
   **`caption-proj` の第 2 出力**の masked mean）
5. S 決定: `expm1(log_frames)` → 銀行家丸め → `[min_frames, max_frames]` へ clamp
6. 条件 state を Tmax へ右 pad ・区間マスクを組む（ADR 0047）
7. Euler 40 step + CFG independent（`t ∈ [0.5, 1.0]` の step だけ・uncond は
   **該当区間のマスクを全 False にするだけ** — ADR 0047 決定 1）

`t_embed` はホスト生成（`cos` が IR の語彙に無い — ADR 0043）。**式は写さず上流の
`get_timestep_embedding` を呼ぶ**。平均トークン前置・latent の patch・caption の masked mean も
同じ理由で上流の関数／メソッドを呼ぶ。写しているのは 5 と 7（上流では `_synthesize` /
`sample_euler_rf_cfg` の内側に埋まっていて関数として取り出せない）だけで、その 2 つは
**上流の正本経路との突合**で守る（下の門）。

## 何を門にするか

emit の前に全て実測し、1 つでも外れたら**何も書かない**（ADR 0005 の fail loudly）:

- **S の決定が上流と一致**すること（上流 `predict_duration_log_frames` から独立に S を出す）
- **最終 z が上流の `sample_euler_rf_cfg` と一致**すること（{@link EULER_REFERENCE_ATOL}）。
  グラフ経路は「条件 KV を毎 forward 再計算・列を詰めて backbone を呼ぶ・uncond をマスク還元」
  の 3 点で上流と実装が違うので**ビット一致はしない**が、40 step 積み上げても差が値域の
  1/1000 に収まることを毎回実測する。ここが崩れたら CFG の合成式・t スケジュール・
  マスクの区間割りのどれかが違っている
- CFG が**実際に効いている**こと（cond のみで回した z との差 — 恒真化の遮断）
- 初期ノイズが上流と**ビット一致**すること（同じ seed・同じ生成順）
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from karume.convert import normalize_boundary_tensor

from . import export as ex
from . import patch

META_FILE = "meta.json"
CASE_PREFIX = "case."
CASE_SUFFIX = ".safetensors"
T_EMBED_FILE = "t-embed.safetensors"

#: Euler の step 数（上流 `SamplingRequest.num_steps` の既定）。
NUM_STEPS = 40

#: t スケジュールの初期倍率（`rf.sample_euler_rf_cfg` の `init_scale`）。
INIT_SCALE = 0.999

#: CFG を掛ける t の範囲（上流 `SamplingRequest.cfg_min_t` / `cfg_max_t` の既定）。
CFG_MIN_T, CFG_MAX_T = 0.5, 1.0

#: CFG の強さ（同 `cfg_scale_text` / `cfg_scale_speaker` / `cfg_scale_caption` の既定）。
#: 並びは ADR 0047 の uncond 変種と同じ（`DIT_UNCOND_VARIANTS`）。
CFG_SCALES: Mapping[str, float] = {"text": 3.0, "speaker": 5.0, "caption": 3.0}

#: S の clamp 範囲を決める秒数（上流 `SamplingRequest.min_seconds` / `max_seconds` の既定）。
#: フレーム数への換算は `CODEC_FRAME_RATE`（= sample_rate / hop_length）で行う。
MIN_SECONDS, MAX_SECONDS = 0.5, 30.0

#: 最終 z が上流 `sample_euler_rf_cfg` と一致すると見なす最大絶対差。
#:
#: 実測（下の常設門が毎回出す・40 step）は **full = 1.03e-4 / no-ref = 1.80e-5** で、z の
#: 値域は \|z\| 上端 5.10 / 4.33。閾値 1e-3 は実測最悪の約 9.7 倍で、値域の 1/5000。
#: 差の出どころは 3 つとも**構造的**: ①グラフ経路は条件 KV を毎 forward 再計算する
#: （上流はキャッシュ）②backbone を詰めた列で呼ぶ（上流は 256 へ pad —
#: `_static_scheme_evidence` が 1e-5 台と実測）③uncond をマスク還元で表す（上流は
#: state 0 + マスク 0）。いずれも数学的には恒等で、残るのは f32 の縮約順序差だけ。
#: 40 step の Euler がそれを増幅した結果がこの桁で（1 step では 1.4e-5 / 4.5e-6 と実測）、
#: **式の取り違え**（CFG の符号・スケール・t スケジュール・マスクの区間割り）は値域と
#: 同じ O(1) で出る — 実際、CFG の有無だけで z は 5.56 / 3.23 動く（`cfgEffectMaxAbs`）。
#:
#: **圧縮系列（`--dtype f16` / `i8`）でも同じ閾値を掛ける**: 丸めはグラフ経路と上流経路の**両方**へ
#: 同じ 1 回だけ当たる（`fake_quant` が丸めた `dit` を上流 `sample_euler_rf_cfg` もそのまま
#: 使う）ので、ここで測る差は f32 系列と同じ「実装差だけ」であり、量子化誤差は両辺で相殺する。
#: 桁が変わりうるのは丸めが条件数の悪い領域を踏んだ場合だけで、それは**実測で決着させる**
#: （閾値を系列ごとに割るのは、実測が実際に外れてからにする — 先回りして緩めると、外れた
#: ことが分からなくなる）。f16 の実測値はまだ無い。
EULER_REFERENCE_ATOL = 1e-3

#: CFG が実際に効いていることを見る下限（cond のみで回した z との最大絶対差）。
#: MUST: 恒真にしない — スケールを 0 にしたり uncond のマスクを間違えて cond と同じにすると、
#: 上流との突合は**両方が同じ間違いをする**限り通ってしまう（上流も同じ入力で回すため）。
CFG_EFFECT_MIN = 1e-2


class ReferenceSpec(NamedTuple):
    """参照 latent の供給元（**合成のまま維持** — 第 4 波で実音声 latent は採れるように
    なった〈`irodori/dacvae/host.py`〉が、この golden を差し替えると突合値が全て動くので既存を守る。
    実音声の値域は speaker 単体門の実 latent ケース〈`irodori.export.SPEAKER_REAL_CASES`〉が
    受け持つ）。

    `frames` は patch 前のフレーム数で、`speaker_patch_size` で割り切れる値にする
    （割り切れない端は上流 `patch_sequence_with_mask` が捨てるので、golden の入力と
    実際に使われた列がずれる）。
    """

    frames: int
    seed: int


class PipelineCase(NamedTuple):
    """full-loop の 1 ケース。`reference` が `None` なら参照なし（`no_ref`）経路。"""

    name: str
    why: str
    text: str
    caption: str
    reference: ReferenceSpec | None
    seed: int


#: full-loop の 2 ケース。
#:
#: - `full` — 実テキスト + 参照 latent + caption あり。CFG は 3 本とも有効（1 step あたり
#:   4 forward）
#: - `no-ref` — 参照なし（speaker はホストがゼロを置く・speaker CFG off）+ caption 空文字
#:   （caption CFG off・マスク全 0）。1 step あたり 2 forward
PIPELINE_CASES: tuple[PipelineCase, ...] = (
    PipelineCase(
        name="full",
        why="参照あり + caption あり（CFG 3 本・1 step 4 forward）",
        text="今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
        caption="若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
        reference=ReferenceSpec(frames=124, seed=201),
        seed=1234,
    ),
    PipelineCase(
        name="no-ref",
        why="参照なし（speaker ゼロ短絡・CFG off）+ caption 空（マスク全 0・CFG off）",
        text="本日はお越しいただき、誠にありがとうございます。",
        caption="",
        reference=None,
        seed=1235,
    ),
)


def t_schedule(num_steps: int) -> torch.Tensor:
    """`[num_steps+1]` の t 列（上流 `sample_euler_rf_cfg` の linear スケジュールそのもの）。

    MUST: 上流と同じ式（`(1 - linspace(0,1,n+1)) * 0.999`）で作る。TS 側は閉形式
    `0.999 * (1 - i/n)` で作る想定なので、**両者がビット一致するか**を
    {@link closed_form_matches} が毎回実測する（外れたら TS 側の式を合わせる）。
    """
    return (1.0 - torch.linspace(0.0, 1.0, num_steps + 1)) * INIT_SCALE


def closed_form_matches(schedule: torch.Tensor, num_steps: int) -> float:
    """閉形式 `0.999 * (1 - i/n)` と上流スケジュールの最大絶対差（0 ならビット一致）。

    実測: `num_steps=40` では **5.96e-8**（ビット一致ではない — `linspace` は
    `start + i*step` を f32 で積むので、閉形式の `1 - i/n` と最終 ulp が割れる点がある）。
    t は `t_embed` と刻み幅 `t_next - t` にしか入らず、この差が最終 z に効く量は
    {@link EULER_REFERENCE_ATOL} の 3 桁下。**値は meta に載せる** — ホスト実装が
    どちらの式を使うかの判断材料になる。
    """
    closed = torch.tensor(
        [INIT_SCALE * (1.0 - index / num_steps) for index in range(num_steps + 1)],
        dtype=schedule.dtype,
    )
    return float((closed - schedule).abs().max())


def banker_round_frames(frames: float) -> int:
    """`int(round(x))` — Python の**銀行家丸め**（`inference_runtime._synthesize`）。

    MUST: 「四捨五入」と読み替えない。0.5 ちょうどのとき偶数側へ丸まるので、TS 側で
    `Math.round`（常に上へ）を使うと 1 フレームずれる。上流の綴りは `int(round(x))` で、
    `round` が既に int を返すので `int(...)` は形を揃えるためだけの写し。
    """
    return round(frames)


def sequence_length(log_frames: torch.Tensor, frame_rate: int) -> tuple[int, dict[str, Any]]:
    """duration の出力（log frames）から latent 長 S を決める。

    上流 `inference_runtime._synthesize` の 5 行（`expm1` → 平均 → `duration_scale` 倍 →
    銀行家丸め → `[min_frames, max_frames]` clamp）と同じ。`duration_scale` は既定 1.0。
    `latent_patch_size` は 1 なので patched 長 = latent 長（値は config から確かめる）。
    """
    frames = float(torch.expm1(log_frames).float().mean())
    min_frames = max(1, math.ceil(MIN_SECONDS * frame_rate))
    max_frames = max(1, math.floor(MAX_SECONDS * frame_rate))
    steps = max(min_frames, min(max_frames, banker_round_frames(frames)))
    return steps, {
        "predictedFrames": round(frames, 4),
        "minFrames": min_frames,
        "maxFrames": max_frames,
    }


def _pack_body(tokenizer: Any, body: str, bos_id: int, max_length: int) -> torch.Tensor:
    """前処理済みの本文 → `[1,T]` の**詰めた** token 列（BOS + `max_length-1` で切ったもの）。

    上流 `PretrainedTextTokenizer.batch_encode` は同じ列を右 pad して返す。静的方式では
    ホストが pad を消して呼ぶので（`irodori/export.py` のモジュール docstring）、ここでは
    pad しない列を作る。
    """
    ids = list(tokenizer.encode(body, add_special_tokens=False).ids)[: max_length - 1]
    return torch.tensor([[bos_id, *ids]], dtype=torch.int64)


def _packed_ids(
    tokenizer: Any, text: str, bos_id: int, max_length: int, normalize_text: Any
) -> torch.Tensor:
    """**text 側**の token 列（`normalize_text` → `strip` → 詰めた列）。"""
    normalized = normalize_text(text).strip()
    if not normalized:
        raise SystemExit("正規化後の本文が空")
    return _pack_body(tokenizer, normalized, bos_id, max_length)


def _packed_caption_ids(tokenizer: Any, caption: str, bos_id: int, max_length: int) -> torch.Tensor:
    """**caption 側**の token 列（`strip` のみ → 詰めた列）。

    MUST: `normalize_text` を掛けない。上流 `inference_runtime._synthesize` が caption に
    掛けるのは `str(...).strip()` だけ（`normalize_text` は text 専用）で、正規化を足すと
    外側括弧の剥がし・NFKC・記号削除のぶんだけ conditioning が黙って別物になる。
    """
    stripped = caption.strip()
    if not stripped:
        raise SystemExit("strip 後の caption が空")
    return _pack_body(tokenizer, stripped, bos_id, max_length)


class HostGraphs(NamedTuple):
    """ホストが回す 5 本のグラフ（`irodori/export.py` の eager 同値ラッパそのもの）。"""

    backbone: nn.Module
    text_proj: nn.Module
    caption_proj: nn.Module
    speaker: nn.Module
    duration: nn.Module
    dit: nn.Module


def _segment_masks(
    length: int, used: Mapping[str, int], caps: Mapping[str, int], uncond: str | None
) -> torch.Tensor:
    """`[1,1,1,S+1519]` の連結マスク（self / text / speaker / caption の順）。

    `uncond` を指定した区間だけ全 False にする = ADR 0047 決定 1 の uncond。
    """
    segments = [torch.ones((1, length), dtype=torch.bool)]
    for name in ex.DIT_UNCOND_VARIANTS:
        segments.append(ex._segment_mask(caps[name], 0 if uncond == name else used[name]))
    return torch.cat(segments, dim=1)[:, None, None, :]


def _euler(
    graphs: HostGraphs,
    source: ex.IrodoriSource,
    noise: torch.Tensor,
    states: Sequence[torch.Tensor],
    used: Mapping[str, int],
    caps: Mapping[str, int],
    enabled: Sequence[str],
    embed_dim: int,
    schedule: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Euler 40 step + CFG independent。戻りは `(最終 z, forward 数)`。

    上流 `rf.sample_euler_rf_cfg` の写しは 3 点だけ（残りは全て上流の関数を呼ぶ）:
    `v = v_cond + Σ scale_k (v_cond - v_k)` / `x += v (t_next - t)` /
    CFG を掛ける区間 `cfg_min_t ≤ t ≤ cfg_max_t`。突合は {@link EULER_REFERENCE_ATOL}。
    """
    x_t = noise
    masks = {
        name: _segment_masks(int(noise.shape[1]), used, caps, name) for name in (None, *enabled)
    }
    forwards = 0
    for index in range(len(schedule) - 1):
        t, t_next = schedule[index], schedule[index + 1]
        with torch.no_grad():
            t_embed = source.timestep_embedding(t.reshape(1), embed_dim).to(x_t.dtype)
            v_cond = graphs.dit(x_t, t_embed, masks[None], *states)
            forwards += 1
            v = v_cond
            if enabled and CFG_MIN_T <= float(t) <= CFG_MAX_T:
                for name in enabled:
                    v_uncond = graphs.dit(x_t, t_embed, masks[name], *states)
                    forwards += 1
                    v = v + CFG_SCALES[name] * (v_cond - v_uncond)
            x_t = x_t + v * (t_next - t)
    return x_t, forwards


def run_case(
    case: PipelineCase,
    graphs: HostGraphs,
    source: ex.IrodoriSource,
    duration_predictor: nn.Module,
    tokenizer: Any,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    config: Any,
    caps: Mapping[str, int],
    *,
    frames_override: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """1 ケースぶんのホスト経路を回して `(保存テンソル, meta)` を返す。

    `frames_override` は **S を外から固定する**ための注入口（既定 `None` = duration の予測
    どおり）。golden の emit は必ず `None` で呼ぶ（S も golden の一部）。使うのは
    `irodori/measure_quant.py` で、量子化構成の間で波形長を揃えないと SNR / LSD が定義
    できないため（`measure_quant_sbv2.py` が dump 側 `w_ceil` で時間グリッドを固定するのと
    同じ理由）。**予測そのもの**は `meta["predictedS"]` に残す — S ドリフトはこの台本が測る
    量ではなく、測る側が読む診断値。
    """
    bos_id = int(text_config["bos_token_id"])
    embed_dim = int(config.timestep_embed_dim)
    latent_dim = int(config.patched_latent_dim)
    if int(config.latent_patch_size) != 1:
        raise SystemExit(
            f"latent_patch_size が {config.latent_patch_size} — S = latent 長の前提が崩れる"
        )

    with torch.no_grad():
        # ---- ①〜② テキスト系（列を詰めたまま backbone → projector）----
        text_ids = _packed_ids(tokenizer, case.text, bos_id, caps["text"], source.normalize_text)
        text_state = graphs.text_proj(graphs.backbone(text_ids))
        caption_ids = None
        caption_length = 0
        caption_state = torch.zeros((1, 1, int(model_config["caption_dim"])))
        caption_pooled = None
        if case.caption:
            caption_ids = _packed_caption_ids(tokenizer, case.caption, bos_id, caps["caption"])
            caption_state, caption_normed = graphs.caption_proj(graphs.backbone(caption_ids))
            caption_length = int(caption_state.shape[1])
            caption_pooled = duration_predictor._caption_vec(
                batch_size=1,
                device=caption_normed.device,
                dtype=caption_normed.dtype,
                caption_state=caption_normed,
                caption_mask=torch.ones(caption_normed.shape[:2], dtype=torch.bool),
                has_caption=torch.ones((1,), dtype=torch.bool),
            )

        # ---- ③ speaker（参照なしはホストがゼロを置く — `_no_reference_evidence` が根拠）----
        reference_latent = torch.zeros((1, int(config.speaker_patch_size), int(config.latent_dim)))
        if case.reference is not None:
            generator = torch.Generator().manual_seed(case.reference.seed)
            reference_latent = torch.randn(
                1, case.reference.frames, int(config.latent_dim), generator=generator
            )
        reference_mask = torch.full(
            reference_latent.shape[:2], case.reference is not None, dtype=torch.bool
        )
        from irodori_tts.model import patch_sequence_with_mask

        patched, patched_mask = patch_sequence_with_mask(
            reference_latent, reference_mask, int(config.speaker_patch_size)
        )
        if case.reference is None:
            speaker_state = torch.zeros((1, int(patched.shape[1]) + 1, int(config.speaker_dim)))
        else:
            speaker_state, _mask = source.prepend_masked_mean_token(
                graphs.speaker(patched), patched_mask
            )

        # ---- ④ duration の 5 入力 ----
        has_speaker = case.reference is not None
        speaker_vec = (
            speaker_state[:, 0] if has_speaker else torch.zeros((1, int(config.speaker_dim)))
        )
        caption_vec = (
            caption_pooled
            if caption_pooled is not None
            else torch.zeros((1, int(config.caption_dim_resolved)))
        )
        log_frames = graphs.duration(
            text_state,
            speaker_vec,
            torch.tensor([[has_speaker]], dtype=torch.bool),
            caption_vec,
            torch.tensor([[caption_pooled is not None]], dtype=torch.bool),
        )

    # ---- ⑤ S 決定 ----
    predicted, duration_meta = sequence_length(log_frames, ex.CODEC_FRAME_RATE)
    steps = predicted if frames_override is None else frames_override

    # ---- ⑥ 条件の右 pad と区間長 ----
    # caption が空のときは state をゼロのまま置き、マスクを全 0 にする（上流
    # `_synthesize` の `caption_mask.zero_()` と同値 — 区間の寄与は exp(−inf)=0 で厳密に 0）。
    used = {
        "text": int(text_state.shape[1]),
        "speaker": int(speaker_state.shape[1]) if has_speaker else 0,
        "caption": caption_length,
    }
    states = (
        ex._right_pad(text_state, caps["text"], f"{case.name} の text 条件"),
        ex._right_pad(speaker_state, caps["speaker"], f"{case.name} の speaker 条件"),
        ex._right_pad(caption_state, caps["caption"], f"{case.name} の caption 条件"),
    )

    # ---- ⑦ Euler + CFG ----
    # 有効な CFG は上流 `resolve_cfg_scales` / `sample_euler_rf_cfg` の条件どおり:
    # text は常に / speaker は参照ありのときだけ / caption は非空のときだけ。
    enabled = ["text"]
    if has_speaker:
        enabled.append("speaker")
    if used["caption"] > 0:
        enabled.append("caption")
    schedule = t_schedule(NUM_STEPS)
    noise = torch.randn(
        (1, steps, latent_dim),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(case.seed),
    )
    latent, forwards = _euler(
        graphs, source, noise, states, used, caps, enabled, embed_dim, schedule
    )
    cond_only, _forwards = _euler(
        graphs, source, noise, states, used, caps, (), embed_dim, schedule
    )
    cfg_effect = float((latent - cond_only).abs().max())
    if cfg_effect < CFG_EFFECT_MIN:
        raise SystemExit(
            f"{case.name}: CFG 有無の差が {cfg_effect} — CFG が効いていない疑い"
            "（スケール 0 / uncond マスクが cond と同じ）"
        )

    # MUST: id 列は保存時に IR の実表現（i64 → i32）へ落とす — TS 側が同じ境界規約
    # （ADR 0009）で読むため。ここでは int64 のまま持ち、上流突合にもこの列を渡す。
    tensors = {
        "noise": noise,
        "z": latent.detach().contiguous(),
        "reference_latent": reference_latent,
        "text_ids": text_ids,
    }
    if caption_ids is not None:
        tensors["caption_ids"] = caption_ids
    meta = {
        "why": case.why,
        "text": case.text,
        "caption": case.caption,
        "seed": case.seed,
        "S": steps,
        "duration": {**duration_meta, "logFrames": round(float(log_frames), 6)},
        "tokens": {"text": int(text_ids.shape[1]), "caption": used["caption"]},
        "segments": dict(used),
        "cfg": {name: CFG_SCALES[name] for name in enabled},
        "forwards": forwards,
        "cfgEffectMaxAbs": float(f"{cfg_effect:.4e}"),
        "reference": (
            None
            if case.reference is None
            else {
                "kind": "synthetic-standard-normal",
                "why": "合成のまま維持（golden の安定のため — 実音声の値域は speaker 単体門の"
                " SPEAKER_REAL_CASES が受け持つ。ReferenceSpec の docstring 参照）",
                "frames": case.reference.frames,
                "seed": case.reference.seed,
                "patchedTokens": int(patched.shape[1]),
            }
        ),
        "zAbsMax": float(f"{float(latent.abs().max()):.6f}"),
    }
    if frames_override is not None:
        # MUST: 固定したときだけ足す（golden の meta.json を系列間で動かさない）。
        meta["predictedS"] = predicted
    return tensors, meta


def _right_pad_ids(
    ids: torch.Tensor | None, length: int, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[1,length]` へ右詰め pad した token 列とマスク（上流 `batch_encode` の最終形）。

    `ids` が `None` なら全 pad + マスク全 0（上流 `caption_mask.zero_()` と同じ形）。caption は
    {@link upstream_caption_condition} が上流の入口から作るので、今の呼び出し元は text だけ。
    """
    padded = torch.full((1, length), pad_id, dtype=torch.int64)
    mask = torch.zeros((1, length), dtype=torch.bool)
    if ids is not None:
        used = int(ids.shape[1])
        padded[0, :used] = ids[0]
        mask[0, :used] = True
    return padded, mask


def upstream_caption_tokenizer(model_dir: Path, add_bos: bool) -> Any:
    """上流 `_synthesize` が caption に使うのと**同じ**トークナイザ（`caption_tokenizer`）。

    `add_bos` は上流 `ModelConfig.caption_add_bos_resolved` から採る（直書きしない）。
    """
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from transformers import AutoTokenizer

    source = str((model_dir / ex.TOKENIZER_FILE).parent)
    return PretrainedTextTokenizer(AutoTokenizer.from_pretrained(source), add_bos=add_bos)


def upstream_caption_condition(
    caption_tokenizer: Any, caption: str, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """caption の条件入力 `[1,max_caption_len]`（token 列 + マスク）を**上流の入口から**作る。

    MUST: ホストが作った `caption_ids` を上流ループへ渡し直さない — 渡し直すと前処理の
    取り違えが両辺へ同じだけ乗って突合が恒真化し、caption の綴りが割れても z が一致する。
    ここは上流 `inference_runtime._synthesize` の caption 経路（`str(...).strip()` →
    `caption_tokenizer.batch_encode` → 空なら `caption_mask.zero_()`）をそのまま呼ぶ。
    """
    caption_text = str(caption).strip()
    ids, mask = caption_tokenizer.batch_encode([caption_text], max_length=max_length)
    if caption_text == "":
        mask.zero_()
    return ids, mask


def upstream_latent(
    model: nn.Module,
    case: PipelineCase,
    text_ids: torch.Tensor,
    caption_padded: torch.Tensor,
    caption_mask: torch.Tensor,
    reference_latent: torch.Tensor,
    steps: int,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """上流の正本経路（`rf.sample_euler_rf_cfg`）で同じ条件を回す。戻りは `(z, 初期ノイズ)`。

    MUST: 入力は上流の呼び方に揃える — text の token 列は `max_text_len` へ右 pad（静的方式で
    ホストが消した pad を戻す）、caption は {@link upstream_caption_condition} が上流の入口で
    作った列をそのまま受ける、参照なしは `no_ref` と同じゼロ latent + 全 0 マスク。
    """
    from irodori_tts.rf import sample_euler_rf_cfg

    pad_id = int(text_config["pad_token_id"])
    text_padded, text_mask = _right_pad_ids(text_ids, int(model_config["max_text_len"]), pad_id)
    reference_mask = torch.full(
        reference_latent.shape[:2], case.reference is not None, dtype=torch.bool
    )
    noise = torch.randn(
        (1, steps, int(config.patched_latent_dim)),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(case.seed),
    )
    latent = sample_euler_rf_cfg(
        model=model,
        text_input_ids=text_padded,
        text_mask=text_mask,
        ref_latent=reference_latent,
        ref_mask=reference_mask,
        sequence_length=steps,
        caption_input_ids=caption_padded,
        caption_mask=caption_mask,
        num_steps=NUM_STEPS,
        cfg_scale_text=CFG_SCALES["text"],
        cfg_scale_caption=CFG_SCALES["caption"],
        cfg_scale_speaker=CFG_SCALES["speaker"] if case.reference is not None else 0.0,
        cfg_guidance_mode="independent",
        cfg_min_t=CFG_MIN_T,
        cfg_max_t=CFG_MAX_T,
        seed=case.seed,
    )
    return latent, noise


def upstream_sequence_length(
    model: nn.Module,
    text_ids: torch.Tensor,
    caption_padded: torch.Tensor,
    caption_mask: torch.Tensor,
    reference_latent: torch.Tensor,
    has_speaker: bool,
    text_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    config: Any,
) -> int:
    """上流の `encode_conditions` + `predict_duration_log_frames` から S を独立に出す。"""
    reference_mask = torch.full(reference_latent.shape[:2], has_speaker, dtype=torch.bool)
    pad_id = int(text_config["pad_token_id"])
    text_padded, text_mask = _right_pad_ids(text_ids, int(model_config["max_text_len"]), pad_id)
    with torch.no_grad():
        (
            text_state,
            text_state_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_state_mask,
        ) = model.encode_conditions(
            text_input_ids=text_padded,
            text_mask=text_mask,
            ref_latent=reference_latent,
            ref_mask=reference_mask,
            caption_input_ids=caption_padded,
            caption_mask=caption_mask,
        )
        log_frames = model.predict_duration_log_frames(
            text_state=text_state,
            text_mask=text_state_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_state_mask,
            duration_features=torch.zeros((1, int(config.duration_aux_dim))),
            has_speaker=torch.tensor([has_speaker], dtype=torch.bool),
            has_caption=torch.tensor([bool(caption_mask.any())], dtype=torch.bool),
        )
    steps, _meta = sequence_length(log_frames, ex.CODEC_FRAME_RATE)
    return steps


def t_embed_table(source: ex.IrodoriSource, schedule: torch.Tensor, dim: int) -> torch.Tensor:
    """40 step で**実際に使う** t に対する `[40, dim]` の `t_embed` 表（W3 の host unit test 用）。

    MUST: 式は写さず上流 `get_timestep_embedding` を呼ぶ（θ の割り方が変われば写した式は
    黙って古いまま通る）。
    """
    with torch.no_grad():
        rows = [source.timestep_embedding(t.reshape(1), dim) for t in schedule[:-1]]
    return torch.cat(rows, dim=0).contiguous()


def emit(model_dir: Path, source_dir: Path, out_dir: Path, dtype: str = "f32") -> dict[str, Any]:
    """full-loop golden を書き、要約を返す（検証に落ちたら 1 バイトも書かない）。"""
    from tokenizers import Tokenizer

    source = ex.IrodoriSource(source_dir)
    text_config, model_config = ex.read_configs(model_dir)
    state = load_file(str(model_dir / ex.MODEL_FILE))
    backbone = ex.load_backbone(source, state, text_config)
    hidden_size = int(backbone.hidden_size)
    text_projector = ex.load_projector(
        source, state, model_config, ex.TEXT_PROJ_PREFIX, hidden_size, int(model_config["text_dim"])
    )
    caption_projector = ex.load_projector(
        source,
        state,
        model_config,
        ex.CAPTION_PROJ_PREFIX,
        hidden_size,
        int(model_config["caption_dim"]),
    )
    config = source.model_config(model_config)
    speaker_encoder = ex.load_speaker_encoder(source, state, config)
    speaker_norm = ex.load_rms_norm(
        source, state, ex.SPEAKER_NORM_PREFIX, int(config.speaker_dim), float(config.norm_eps)
    )
    text_norm = ex.load_rms_norm(
        source, state, ex.TEXT_NORM_PREFIX, int(config.text_dim), float(config.norm_eps)
    )
    caption_norm = ex.load_rms_norm(
        source,
        state,
        ex.CAPTION_NORM_PREFIX,
        int(config.caption_dim_resolved),
        float(config.norm_eps),
    )
    duration_predictor = ex.load_duration_predictor(source, state, config)
    dit = ex.load_dit(source, state, config, text_config)

    # MUST: 丸めは load の直後・golden の採取より前（`irodori.export.fake_quant` の順序 MUST）。
    # `dit` は `TextToLatentRFDiT` **丸ごと**（backbone / projector / speaker / duration の
    # コピーを内側に持つ）なので、上流突合（`sample_euler_rf_cfg` / `predict_duration_log_frames`）
    # もここで丸めた重みで回る = グラフ経路と上流経路の両辺が同じ丸めを受ける。
    quantized = ex.fake_quant(
        dtype,
        {
            ex.TARGET_BACKBONE: backbone,
            ex.TARGET_TEXT_PROJ: text_projector,
            ex.TARGET_CAPTION_PROJ: caption_projector,
            ex.TARGET_SPEAKER: speaker_encoder,
            ex.TARGET_DURATION: duration_predictor,
            ex.TARGET_DIT: dit,
            "speaker_norm": speaker_norm,
            "text_norm": text_norm,
            "caption_norm": caption_norm,
        },
    )

    # MUST: グラフラッパは**パッチ後**でしか正しく動かない（実数形 RoPE 表を渡すため）。
    # このファイルは「パッチ前の参照」を採らない（採る側は irodori/export.py）ので、
    # ここで当てて構わない。
    patch.apply_patches()
    speaker_max = ex.speaker_sym_max(model_config)
    graphs = HostGraphs(
        backbone=ex.BackboneGraph(backbone),
        text_proj=ex.ProjectorGraph(text_projector),
        caption_proj=ex.CaptionProjectorGraph(caption_projector, caption_norm),
        speaker=ex.SpeakerGraph(speaker_encoder, speaker_norm, speaker_max),
        duration=ex.DurationGraph(duration_predictor, text_norm),
        dit=ex.DitGraph(dit, ex.dit_sym_max(config)),
    )
    caps = {
        "text": int(model_config["max_text_len"]),
        "speaker": speaker_max + 1,
        "caption": int(model_config["max_caption_len"]),
    }
    tokenizer = Tokenizer.from_file(str(model_dir / ex.TOKENIZER_FILE))
    caption_tokenizer = upstream_caption_tokenizer(model_dir, bool(config.caption_add_bos_resolved))

    schedule = t_schedule(NUM_STEPS)
    cases: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, torch.Tensor]] = {}
    for case in PIPELINE_CASES:
        tensors, meta = run_case(
            case,
            graphs,
            source,
            duration_predictor,
            tokenizer,
            text_config,
            model_config,
            config,
            caps,
        )
        # ---- 常設門: S と最終 z を上流の正本経路と突き合わせる ----
        text_ids = tensors["text_ids"]
        # MUST: caption の条件入力は上流の入口から独立に作る（ホストの列を渡し直さない —
        # {@link upstream_caption_condition}）。両者が同じ列になることは直下で実測する。
        caption_padded, caption_mask = upstream_caption_condition(
            caption_tokenizer, case.caption, caps["caption"]
        )
        host_caption_ids = tensors.get("caption_ids")
        upstream_used = int(caption_mask.sum())
        host_used = 0 if host_caption_ids is None else int(host_caption_ids.shape[1])
        if host_used != upstream_used or (
            host_caption_ids is not None
            and not torch.equal(host_caption_ids[0], caption_padded[0, :upstream_used])
        ):
            raise SystemExit(
                f"{case.name}: caption の token 列がホスト（{host_used} token）と"
                f"上流（{upstream_used} token）で食い違う — 前処理の綴りが割れている"
            )
        upstream_steps = upstream_sequence_length(
            dit,
            text_ids,
            caption_padded,
            caption_mask,
            tensors["reference_latent"],
            case.reference is not None,
            text_config,
            model_config,
            config,
        )
        if upstream_steps != meta["S"]:
            raise SystemExit(
                f"{case.name}: S がホスト {meta['S']} / 上流 {upstream_steps} で食い違う"
                "（expm1 → 銀行家丸め → clamp のどこかが違う）"
            )
        reference, upstream_noise = upstream_latent(
            dit,
            case,
            text_ids,
            caption_padded,
            caption_mask,
            tensors["reference_latent"],
            meta["S"],
            text_config,
            model_config,
            config,
        )
        if not torch.equal(upstream_noise, tensors["noise"]):
            raise SystemExit(f"{case.name}: 初期ノイズが上流と一致しない（seed の使い方が違う）")
        worst = float((reference - tensors["z"]).abs().max())
        if worst > EULER_REFERENCE_ATOL:
            raise SystemExit(
                f"{case.name}: 最終 z が上流と {worst} 違う（許容 {EULER_REFERENCE_ATOL}）"
                " — CFG 合成式 / t スケジュール / マスク区間のどれかが崩れている"
            )
        meta["upstreamMaxAbs"] = float(f"{worst:.4e}")
        cases[case.name] = meta
        payloads[case.name] = tensors

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
    table = t_embed_table(source, schedule, int(config.timestep_embed_dim))
    save_file(
        {"t": schedule[:-1].contiguous(), "t_embed": table},
        str(out_dir / T_EMBED_FILE),
    )
    written[T_EMBED_FILE] = (out_dir / T_EMBED_FILE).stat().st_size

    meta_payload = {
        "dtype": dtype,
        "fakeQuant": quantized.reports,
        "steps": NUM_STEPS,
        "initScale": INIT_SCALE,
        "cfgRange": [CFG_MIN_T, CFG_MAX_T],
        "cfgScales": dict(CFG_SCALES),
        "frameRate": ex.CODEC_FRAME_RATE,
        "tSchedule": [float(value) for value in schedule.tolist()],
        "tScheduleClosedFormMaxAbs": closed_form_matches(schedule, NUM_STEPS),
        "tEmbedDim": int(config.timestep_embed_dim),
        "caps": caps,
        "cases": cases,
    }
    (out_dir / META_FILE).write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    written[META_FILE] = (out_dir / META_FILE).stat().st_size
    return {"dir": str(out_dir), "bytes": written, **meta_payload}


def default_out_dir(model_dir: Path, dtype: str = "f32") -> Path:
    """既定の置き場（`outputs/series/irodori-<実重みのディレクトリ名>{,-f16,-i8}/pipeline/`）。

    系列 root の綴りは `irodori.export.default_out_root` と同一 —
    `pipeline/` はその下の 1 段（グラフのターゲットと並ぶ席）。
    """
    return ex.default_out_root(model_dir, dtype) / "pipeline"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dtype",
        choices=ex.WEIGHT_DTYPES,
        default="f32",
        help="重みの格納 dtype（f16 / i8 は golden を fake-quant 後の重みで焼き直す"
        " — ADR 0018 / 0019 / 0027 / 0050）",
    )
    args = parser.parse_args(argv)
    out_dir = default_out_dir(args.model_dir, args.dtype) if args.out is None else args.out
    summary = emit(args.model_dir, args.source_dir, out_dir, args.dtype)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
