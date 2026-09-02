r"""Irodori-TTS v4 の**ホスト側アルゴリズム**の数の正（full-loop latent golden）。

`irodori/export.py` がグラフを、`irodori/tokenizer_ref.py` がテキスト前処理の資産を出すのに対し、
こちらが出すのは「グラフを 6 本回して latent を作るまで」の**最終値**。W3 のホスト実装
（TS のパイプライン）が突き合わせる統合門の参照値そのもので、値は
**export したグラフ基準**（= `irodori/export.py` の Graph ラッパ = eager 同値実装）で採る。

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype f16
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i8
    uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref --dtype i4

`--dtype f16` / `--dtype i8` / `--dtype i4` は**別系列**
（`outputs/series/irodori-v4-small-{f16,i8,i4}/pipeline/`）へ書く。丸めは
`irodori.export.fake_quant` を load 直後に当てるので、golden も上流突合の参照も**同じ丸めた
重み**で計算される（ADR 0018 / 0019 / 0027 / 0050 — 系列ごとに golden を焼き直す形）。

`--dtype i4`（配布の quant 席 `w4`）だけは丸めが 2 段になる: 段 1 は全役割を **i8 席と同一の
fake-quant**、段 2 は **`--dtype i4` で export 済みの `dit` コンテナを読み戻して**ラッパ所有
パラメタを上書きする（{@link restore_dit_from_i4_series}）。したがってこの系列は
**export を先に走らせてある**ことが前提。

出力（既定 `outputs/series/irodori-v4-small{,-f16,-i8,-i4}/pipeline/`・`.gitignore` 配下）:

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
- **最終 z が上流の `sample_euler_rf_cfg` と一致**すること（2 段判定）。
  グラフ経路は「条件 KV を毎 forward 再計算・列を詰めて backbone を呼ぶ・uncond をマスク還元」
  の 3 点で上流と実装が違うので**ビット一致はしない**。1 段目は固定閾値
  {@link EULER_REFERENCE_ATOL}。誤差の蓄積・増幅のされ方はモデル × 入力 × 丸めで桁ごと
  動く（v4.1 f16 で実測 37,107 倍 — 定数コメントの追記 2026-09-01）ので、超過時は増幅率を
  実測し {@link euler_reference_within_sensitivity} で正規化判定する。それでも落ちたら
  CFG の合成式・t スケジュール・マスクの区間割りのどれかが違っている
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

from _shared.calib_provenance import calib_complaint
from karume.convert import normalize_boundary_tensor
from karume.dist import ir_graph, safetensors_header
from karume.emit import unpack_int4
from karume.quantize import dequantize_int4
from karume.shards import parse_piece_key, resolve_shards
from karume.verify import READER_DTYPE_BITS

from . import export as ex
from . import patch
from .distribution import CALIB_PROVENANCE_FILE, CALIB_SHIPPABLE_METHOD, irodori_calib_floor

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
#: **圧縮系列（`--dtype f16` / `i8` / `i4`）でも同じ閾値を掛ける**: 丸めはグラフ経路と上流経路の
#: **両方**へ同じ 1 回だけ当たる（`fake_quant` が丸めた `dit` を上流 `sample_euler_rf_cfg` も
#: そのまま使う。i4 の読み戻しも同じ `dit` の Parameter を上書きするので同様）ので、ここで測る
#: 差は f32 系列と同じ「実装差だけ」であり、量子化誤差は両辺で相殺する。
#: 桁が変わりうるのは丸めが条件数の悪い領域を踏んだ場合だけで、それは**実測で決着させる**
#: （閾値を系列ごとに割るのは、実測が実際に外れてからにする — 先回りして緩めると、外れた
#: ことが分からなくなる）。
#:
#: 追記（2026-09-01）: v4.1-small の f16 で実際に外れた（full = 1.33e-2）。実測で決着させた
#: 結果は「実装差の種は従来水準のまま、40 step の反復がそれを 37,107 倍に蓄積・増幅する
#: 入力だった」（v4-small f16 は 353 倍 —
#: docs/research/2026-09-01-irodori-v41-euler-sensitivity.md）。
#: 蓄積後の値はモデル × 入力 × 丸めで桁ごと動くため、この定数は**1 段目（fast path）**に
#: 格下げし、超過時は増幅率を実測して正規化する 2 段目（下の
#: {@link euler_reference_within_sensitivity}）で合否を決める。
EULER_REFERENCE_ATOL = 1e-3

#: 2 段目: 増幅率で正規化した実装差の上限。実測（2026-09-01・v4 / v4.1 × f32 / f16 / i8 の
#: 10 セル）の worst / amp は 1.6e-8〜2.2e-6 に収まる — ここはその上限の約 2 倍。
EULER_NOISE_PER_AMP = 5e-6

#: 2 段目: 増幅率と無関係に掛ける絶対上限。式の取り違え（CFG の符号・スケール・t スケジュール・
#: マスク区間）は O(1) で出る（`cfgEffectMaxAbs` 実測 3〜5）ので、その遥か下で止める。
EULER_REFERENCE_ABS_CEILING = 5e-2

#: 増幅率実測の摂動幅。実装差の種（f32 の縮約順序差 ~1e-7）より十分大きく、
#: 線形応答が読める大きさ。
SENSITIVITY_EPS = 1e-6


def euler_reference_within_sensitivity(worst: float, amp: float) -> bool:
    """2 段目の合否 — 増幅率 `amp` で正規化した実装差が種の水準か、かつ絶対上限内か。

    `worst` は上流との最終 z の最大絶対差、`amp` は初期 noise への微小摂動が最終 z へ届く
    倍率（{@link run_case} の `sensitivity_probe`）。1 段目（{@link EULER_REFERENCE_ATOL}）を
    超えた場合にだけ呼ばれる。
    """
    return worst <= amp * EULER_NOISE_PER_AMP and worst <= EULER_REFERENCE_ABS_CEILING


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
    sensitivity_probe: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """1 ケースぶんのホスト経路を回して `(保存テンソル, meta)` を返す。

    `frames_override` は **S を外から固定する**ための注入口（既定 `None` = duration の予測
    どおり）。golden の emit は必ず `None` で呼ぶ（S も golden の一部）。使うのは
    `irodori/measure_quant.py` で、量子化構成の間で波形長を揃えないと SNR / LSD が定義
    できないため（`measure_quant_sbv2.py` が dump 側 `w_ceil` で時間グリッドを固定するのと
    同じ理由）。**予測そのもの**は `meta["predictedS"]` に残す — S ドリフトはこの台本が測る
    量ではなく、測る側が読む診断値。

    `sensitivity_probe=True` は Euler ループをもう 1 周（noise + {@link SENSITIVITY_EPS}）
    走らせ、増幅率を `meta["sensitivityAmp"]` に載せる。emit の 2 段目
    （{@link euler_reference_within_sensitivity}）専用 — 常時は測らない（コスト倍化）。
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
    sensitivity_amp = None
    if sensitivity_probe:
        perturbed, _fw = _euler(
            graphs,
            source,
            noise + SENSITIVITY_EPS,
            states,
            used,
            caps,
            enabled,
            embed_dim,
            schedule,
        )
        sensitivity_amp = float((perturbed - latent).abs().max()) / SENSITIVITY_EPS

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
        **(
            {"sensitivityAmp": float(f"{sensitivity_amp:.4e}")}
            if sensitivity_amp is not None
            else {}
        ),
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


#: i4 系列の読み戻しで受け付ける格納 dtype → `(safetensors の dtype 名, 生バイトを載せる器)`。
#:
#: i4 は packed 4bit（1 バイトに 2 要素 — ADR 0069 決定 2）なので器は uint8 で、論理形へ戻すのは
#: `karume.emit.unpack_int4`。i8 が並ぶのは **block 外の 5 本**（`in_proj` / `out_proj` /
#: `cond_module.{0,2,4}`）と **adaLN 144 本**（`attention_adaln` / `mlp_adaln`）の計 149 本
#: （どちらも聴感裁定 2026-08-23 で i4 から外した。`irodori.export._fake_quant_i4`）。
#: ここに無い格納（f16 / bf16）が `dit` のコンテナに現れたら、w4 席の混成が想定と違う形で
#: 出荷されている（{@link restore_dit_from_i4_series} が落とす）。
_RESTORE_STORAGE: Mapping[str, tuple[str, torch.dtype]] = {
    "f32": ("F32", torch.float32),
    "i8": ("I8", torch.int8),
    "i4": ("I4", torch.uint8),
}

#: 逆変換に scale が要る格納 dtype（宣言に scale が無ければ読み戻せない = 即エラー）。
_SCALED_STORAGE = frozenset({"i8", "i4"})

#: 持ち上げ定数のテンソルキーの接頭辞（`karume.convert` が `const.<digest16>` で振る）。
#:
#: コンテナに並ぶのは**ラッパ所有パラメタと持ち上げ定数の 2 種だけ**なので、上書き対象の席は
#: これを除いた残りで決まる。綴りが core 側で動いたら、定数が「モジュールに無いパラメタ」として
#: 形の門に掛かる — 黙って通る側には倒れない。
_LIFTED_CONST_PREFIX = "const."

#: safetensors のヘッダ長を書く先頭バイト数（データ節の開始位置 = これ + ヘッダ長）。
_HEADER_LENGTH_BYTES = 8


class RestoredDit(NamedTuple):
    """i4 系列から読み戻した `dit` の記録（meta.json の `i4Source` に載る）。"""

    #: 読んだコンテナ（= 配布へ入るバイトそのもの）。
    container: Path
    #: その系列の校正条件（`calib_provenance.json` — 方式・格子・ケース数・step 数）。
    calib: Mapping[str, Any]
    #: i4 格納だったパラメタの本数（コンテナが正 — 期待値をコードに焼かない）。
    int4: int
    #: i8 格納だったパラメタの本数（block 外の `in_proj` / `out_proj` / `cond_module` と
    #: block 内の adaLN — 実重みでは 149 本）。
    int8: int
    #: f32 格納のまま読み戻したパラメタの本数（bias / norm）。
    plain: int
    #: 読み戻しで**値が動いた** i4 パラメタの本数（席の効き門の実測値）。
    changed: int


def _shippable_calib(series_dir: Path) -> Mapping[str, Any]:
    """i4 系列の校正条件を読み、**配布して良い条件**（GPTQ × 下限以上の予算）であることを
    確かめて返す。

    MUST: `--no-calib`（素の RTN）や `--calib-steps 1`（smoke 予算）の生成物から golden を
    焼かない。校正の方式も予算も格納形を 1 バイトも変えない（格子は RTN i4 g32 のまま）ので、
    コンテナのどこを見ても判別できず、出るのは音の劣化だけ — 配布の組み立てが
    `irodori.distribution.assert_irodori_calib_provenance` で張っているのと**同じ判定**
    （`_shared.calib_provenance` が正本）を、golden の焼き直しにも張る（golden だけが smoke 用の
    丸めで採られていると、配布資産との突合が「両辺が違う重み」のまま緑になる）。
    """
    path = series_dir / CALIB_PROVENANCE_FILE
    if not path.is_file():
        raise SystemExit(
            f"i4 系列の校正条件の記録が無い: {path}"
            "（`python -m irodori.export --dtype i4` で書かれる）"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as cause:
        raise SystemExit(f"校正条件の記録を解析できない: {path} — {cause}") from cause
    complaint = calib_complaint(
        record, method=CALIB_SHIPPABLE_METHOD, at_least=irodori_calib_floor()
    )
    if complaint is not None:
        raise SystemExit(
            f"i4 系列の校正条件が配布の条件を満たしていない: {path} は{complaint}"
            " — `--no-calib` / smoke 予算の生成物から golden を焼かない"
        )
    return record


def _stored_parameters(graph: Mapping[str, Any], where: Path) -> dict[str, tuple[str, str | None]]:
    """IR の initializer 宣言から「パラメタ席のテンソルキー → `(格納 dtype, scale キー)`」を引く。

    scale のキーを綴りから組み立てず**宣言から引く**のは、格納の正本が IR だから
    （`karume.emit` は `karume.scale.<重みキー>` で振るが、それは書き手側の実装詳細で、
    読み手が写経すると 2 箇所で独立に動ける）。i4 / i8 なのに scale の宣言が無い席は即エラー
    （逆変換の足場が無い = 読み戻せない）。
    """
    initializers = graph.get("initializers")
    if not isinstance(initializers, dict):
        raise SystemExit(f"{where}: IR メタデータに initializers が無い")
    stored: dict[str, tuple[str, str | None]] = {}
    for name, raw in sorted(initializers.items()):
        storage = raw.get("storage") if isinstance(raw, dict) else None
        key = raw.get("tensor") if isinstance(raw, dict) else None
        if not isinstance(key, str) or not isinstance(storage, dict):
            raise SystemExit(f"{where}: initializer '{name}' の宣言が読めない")
        if key.startswith(_LIFTED_CONST_PREFIX):
            continue
        dtype = storage.get("dtype")
        if dtype not in _RESTORE_STORAGE:
            raise SystemExit(
                f"{where}: '{key}' の格納 {dtype!r} は読み戻せない"
                f"（w4 席の dit に並ぶのは {' / '.join(sorted(_RESTORE_STORAGE))} だけ）"
            )
        scale = storage.get("scale")
        if dtype in _SCALED_STORAGE and not isinstance(scale, str):
            raise SystemExit(f"{where}: {dtype} 格納の '{key}' に scale の宣言が無い")
        if key in stored:
            raise SystemExit(f"{where}: テンソルキー '{key}' を 2 つの initializer が指している")
        stored[key] = (dtype, scale if dtype in _SCALED_STORAGE else None)
    return stored


#: テンソル 1 本の在処（収容 shard・データ節内の `[begin, end)`）。piece 列は index 順に並ぶ。
_Parts = dict[str, tuple[tuple[Path, int, int], ...]]


def _entry_offsets(container: Path, key: str, entry: Any) -> tuple[int, int]:
    """ヘッダ項目の `data_offsets`（形が違えば即エラー）。"""
    offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise SystemExit(f"{container}: '{key}' のヘッダ項目が読めない")
    return int(offsets[0]), int(offsets[1])


def _fold_piece_entries(
    container: Path, name: str, found: list[tuple[int, int, Path, Any]]
) -> tuple[dict[str, Any], tuple[tuple[Path, int, int], ...]]:
    """piece 列を親 1 本の宣言と区間列へ畳む（読み手契約 5 — ADR 0090 決定 1）。

    宣言は親の dtype と全体 shape（先頭次元 = 各 piece の行数の和）。ここで畳まないと i4 席の
    集合突合が piece キーで数えられ、「I4 格納のテンソルが 1 本も無い」と誤って落ちる。
    """
    ordered = sorted(found)
    count = ordered[0][1]
    if [index for index, *_rest in ordered] != list(range(1, count + 1)):
        raise SystemExit(
            f"{container}: '{name}' の piece 連番 1..{count} が揃っていない"
            f"（現物 {[index for index, *_rest in ordered]}）"
        )
    head = ordered[0][3]
    rows = 0
    parts: list[tuple[Path, int, int]] = []
    for index, _count, shard, entry in ordered:
        if entry.get("dtype") != head.get("dtype"):
            raise SystemExit(
                f"{container}: '{name}' の piece {index} が {entry.get('dtype')!r}"
                f"（piece 1 は {head.get('dtype')!r}）— dtype は親と同一 MUST"
            )
        shape = entry.get("shape")
        if not isinstance(shape, list) or not shape or shape[1:] != head["shape"][1:]:
            raise SystemExit(
                f"{container}: '{name}' の piece {index} の shape {shape} が"
                f" piece 1 の {head['shape']} と先頭次元以外で違う"
            )
        rows += int(shape[0])
        begin, end = _entry_offsets(container, name, entry)
        parts.append((shard, begin, end))
    return {"dtype": head["dtype"], "shape": [rows, *head["shape"][1:]]}, tuple(parts)


def _component_headers(container: Path) -> tuple[dict[str, Any], _Parts]:
    """コンポーネント全 shard のヘッダを 1 枚へ畳み、テンソルキー → 在処の区間列も返す。

    MUST: 代表 path 1 本だけを見ない — 配布形は常に「グラフ shard（データ節 0 本）+ weight
    shard 列」（ADR 0081）なので、先頭を読むだけでは I4 のテンソルが 1 本も見えず、
    「i4 系列ではない」と誤って落ちる。`__metadata__` は畳んだ表に入れない（IR の取り出しは
    `karume.dist.ir_graph` の側の仕事で、あちらがグラフ shard を名指しで読む）。

    分割テンソル（`<親名>#NNNNN-of-NNNNN` — ADR 0090）も**親 1 本へ畳む**。畳んだ宣言は親の
    dtype と全体 shape で、在処は piece の index 順に並ぶ区間列になる（配布形の shard 順の
    整合そのものは `karume verify` が持つ — ここは読むために要る整合だけを見る）。
    """
    merged: dict[str, Any] = {}
    parts: _Parts = {}
    pieces: dict[str, list[tuple[int, int, Path, Any]]] = {}
    owner: dict[str, Path] = {}
    for shard in resolve_shards(container):
        for key, entry in safetensors_header(shard).items():
            if key == "__metadata__":
                continue
            if key in owner:
                raise SystemExit(
                    f"{container}: テンソル '{key}' が {owner[key].name} と {shard.name} に"
                    "重複している（shard 跨ぎの重複は配布形の不変条件違反）"
                )
            owner[key] = shard
            parsed = parse_piece_key(key)
            if parsed is None:
                merged[key] = entry
                parts[key] = ((shard, *_entry_offsets(container, key, entry)),)
                continue
            name, index, count = parsed
            pieces.setdefault(name, []).append((index, count, shard, entry))
    for name, found in pieces.items():
        if name in merged:
            raise SystemExit(
                f"{container}: テンソル '{name}' が丸ごとと piece の両方でコンテナに居る"
                "（1 テンソルはどちらか一方 MUST）"
            )
        merged[name], parts[name] = _fold_piece_entries(container, name, found)
    return merged, parts


def _read_stored(
    container: Path,
    header: Mapping[str, Any],
    parts: Mapping[str, tuple[tuple[Path, int, int], ...]],
    expected: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    """コンテナの生バイトを論理形の torch テンソルへ読む（I4 は nibble 展開まで・I8 は素の器）。

    MUST: `safetensors` のリーダを通さない — ライブラリ（0.8.0）の dtype 語彙に `I4` が無く、
    packed 4bit を含むコンテナは開く時点で落ちる（`karume.verify` が自前リーダを持つのと同じ
    理由）。展開そのものは**書き下ろさず** core（`karume.emit.unpack_int4`）を呼ぶ。

    `expected` は「テンソルキー → IR が宣言した格納 dtype」、`parts` は
    {@link _component_headers} が引いた在処の区間列（分割テンソルなら piece の index 順）。
    データ節のオフセットは **shard ごとに独立**（ADR 0081）なので、shard 単位でまとめて開いて
    から席を引き、最後に親ごとへ連結する（行は連続メモリ順なので、バイト列の連結がそのまま
    親の実体になる）。ヘッダの dtype が宣言と食い違う / 宣言した形と実バイト長が合わない、は
    どちらも即エラー（宣言と実体の 2 面を突き合わせる）。
    """
    by_shard: dict[Path, list[tuple[str, int, int, int]]] = {}
    for key in sorted(expected):
        found = parts.get(key)
        if found is None or not isinstance(header.get(key), dict):
            raise SystemExit(f"{container}: テンソル '{key}' がコンテナに無い")
        for position, (shard, begin, end) in enumerate(found):
            by_shard.setdefault(shard, []).append((key, position, begin, end))

    raw: dict[tuple[str, int], bytes] = {}
    for shard, wanted in by_shard.items():
        with shard.open("rb") as stream:
            head = stream.read(_HEADER_LENGTH_BYTES)
            data_start = _HEADER_LENGTH_BYTES + int.from_bytes(head, "little")
            for key, position, begin, end in wanted:
                stream.seek(data_start + begin)
                chunk = stream.read(end - begin)
                if len(chunk) != end - begin:
                    raise SystemExit(f"{shard}: '{key}' のデータ節がファイル末尾で切れている")
                raw[(key, position)] = chunk

    values: dict[str, torch.Tensor] = {}
    for key in sorted(expected):
        dtype = expected[key]
        name, container_dtype = _RESTORE_STORAGE[dtype]
        entry = header[key]
        if entry.get("dtype") != name:
            raise SystemExit(
                f"{container}: '{key}' は IR の宣言が {dtype} なのにヘッダは"
                f" {entry.get('dtype')!r}（宣言と実体が割れている）"
            )
        shape = entry.get("shape")
        if not isinstance(shape, list):
            raise SystemExit(f"{container}: '{key}' のヘッダ項目が読めない")
        blob = b"".join(raw[(key, position)] for position in range(len(parts[key])))
        bits = math.prod(int(dim) for dim in shape) * READER_DTYPE_BITS[name]
        if bits % 8 or len(blob) != bits // 8:
            raise SystemExit(
                f"{container}: '{key}' の宣言 {shape} × {name} と実バイト {len(blob)} が食い違う"
            )
        flat = torch.frombuffer(bytearray(blob), dtype=container_dtype)
        values[key] = unpack_int4(flat, shape) if dtype == "i4" else flat.reshape(shape)
    return values


def _dequantize_stored(dtype: str, raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """出荷バイト（i4 packed / i8）を scale で f32 の論理形へ戻す。

    i4 は core の `karume.quantize.dequantize_int4` をそのまま呼ぶ（`karume.emit` が格納前に
    「書くバイトから戻してビット一致」を実測しているのと**同じ経路** — ADR 0069 決定 4 ③）。

    NOTE: i8 側に相当する関数は core に無い（`karume.quantize` は `quantize_to_int8` だけを
    持ち、逆変換は `karume.emit._convert_for_storage` がビット一致検査の中で `quantized *
    scale` と直に書いている）。ここも同じ 1 式で、per-channel scale は keepdim 形なので
    ブロードキャストがそのまま軸に乗る。
    """
    if dtype == "i4":
        return dequantize_int4(raw, scale)
    return raw.to(torch.float32) * scale


def restore_dit_from_i4_series(wrapper: nn.Module, series_dir: Path) -> RestoredDit:
    """i4 系列の**出荷バイト**で、`dit` のラッパ所有パラメタを丸ごと上書きする（丸めの段 2）。

    段 1（{@link emit}）は全役割を **w8 席と同一の fake-quant**（i8）で丸める — quant 席 `w4` は
    他 7 役に w8 の i8 バイトを共有させる混成席（`irodori.distribution.IRODORI_QUANT_SEATS`）
    なので、条件エンコーダ側の丸めは w8 golden と同一が正しい。ここはその上に `dit` だけを
    重ねる段で、**校正をもう 1 度走らせない**: GPTQ の丸め先は捕捉した活性に依るので「2 回の
    校正が同じ丸めを出す」ことはどこも保証していない。golden が見た重みは配布バイトそのもの、
    を機械で言い切れる唯一の形が「エクスポート済みの系列を読み戻す」。

    上書きは i4 格納だけでなく**ラッパ所有パラメタの全部**（i8 格納の 149 本も、f32
    格納の bias / norm も、コンテナの値で書く）。元値と一致するはずの側もコンテナから書くことで、
    「golden が見た重み = 配布バイト」に例外席を作らない。

    門（どれも fail loudly・1 つでも外れたら golden を 1 バイトも書かせない）:

    - **provenance**: 系列が GPTQ 校正付きで丸められたこと（{@link _shippable_calib}）
    - **形**: 上書き対象の FQN がラッパ所有パラメタと過不足なく一致すること（コンテナに在るのに
      モジュールに無い / 逆、どちらも即エラー）。ずれたまま通すと、上書きされなかった重みだけが
      i8 の値で golden に載る
    - **本数**: ヘッダの I4 テンソルの集合と、IR の宣言から i4 として上書きした集合が一致する
      こと（本数はコンテナが正 — 期待値を焼かない）。**I8 は数えない** — 段 1 と同じ丸めなので
      「i4 系列を読んでいる」の証拠にならず、i4 側の集合一致だけがそれを言える
    - **席の効き**: 上書きで値が動いた i4 パラメタが 1 本も無い、を落とす（読み戻しが効いて
      いないのに w8 golden を w4 golden と呼ぶ事故は、数値も形も合うので他のどの門にも掛からない）
    """
    container = series_dir / ex.MODEL_FILE
    if not all(shard.is_file() for shard in resolve_shards(container)):
        raise SystemExit(
            f"i4 系列のコンテナが無い: {container}"
            "（`python -m irodori.export --dtype i4` を先に走らせる）"
        )
    calib = _shippable_calib(series_dir)
    # ヘッダは 2 度読む（オフセット表と IR メタデータ）— IR の取り出しは core の `ir_graph` に
    # 任せて綴りを写経しない。読むのはどちらもヘッダだけで、数 GB のデータ節は舐めない。
    header, parts = _component_headers(container)
    stored = _stored_parameters(ir_graph(container), container)
    owned = dict(wrapper.named_parameters())
    absent = sorted(set(owned) - set(stored))
    extra = sorted(set(stored) - set(owned))
    if absent or extra:
        raise SystemExit(
            f"{container}: 上書き対象がラッパ所有パラメタと一致しない —"
            f" コンテナに無い {absent[:3]} / モジュールに無い {extra[:3]}"
            "（DitGraph の構成と export した系列のどちらかが動いている）"
        )
    int4_keys = frozenset(key for key, (dtype, _scale) in stored.items() if dtype == "i4")
    packed_keys = {
        key
        for key, entry in header.items()
        if isinstance(entry, dict) and entry.get("dtype") == "I4"
    }
    if packed_keys != int4_keys:
        raise SystemExit(
            f"{container}: I4 格納のテンソル {len(packed_keys)} 本に対し、i4 として読み戻す宣言は"
            f" {len(int4_keys)} 本（過不足: {sorted(packed_keys ^ int4_keys)[:3]}）"
        )
    if not int4_keys:
        raise SystemExit(f"{container}: I4 格納のテンソルが 1 本も無い（i4 系列ではない）")
    expected = {key: dtype for key, (dtype, _scale) in stored.items()}
    expected.update({scale: "f32" for _dtype, scale in stored.values() if scale is not None})
    values = _read_stored(container, header, parts, expected)

    # 1 本ずつ「戻して → 比べて → 書いて → 捨てる」。全部を f32 で持ってから書くと、`dit` の
    # f32 一式（1.4GB 級）と同じ大きさの複製がもう 1 つ同時に生きる（`karume.emit` が格納側で
    # 同じ理由の逐次化をしているのと対）。門に落ちた時点で例外なので、途中まで上書きされた
    # モジュールが golden を書くことはない。
    changed = 0
    with torch.no_grad():
        for key, (dtype, scale_key) in sorted(stored.items()):
            raw = values.pop(key)
            value = raw if scale_key is None else _dequantize_stored(dtype, raw, values[scale_key])
            parameter = owned[key]
            if tuple(value.shape) != tuple(parameter.shape):
                raise SystemExit(
                    f"{container}: '{key}' の形が コンテナ {list(value.shape)} /"
                    f" モジュール {list(parameter.shape)} で食い違う"
                )
            # MUST: 効き門に数えるのは **i4 の席だけ**。i8 の 149 本は段 1 の i8 丸めと
            # 同じ scale・同じ格子なので値が動かないのが正常で、数に入れると「i4 が 1 本も
            # 効いていない」を i8 の一致が埋め合わせて隠す。
            if dtype == "i4" and not torch.equal(parameter.detach(), value):
                changed += 1
            parameter.copy_(value)
            del raw, value
    if changed == 0:
        raise SystemExit(
            f"{container}: 読み戻した i4 {len(int4_keys)} 本が段 1（i8 丸め）の値と全て同じ"
            " — i4 の読み戻しが効いていない（w8 golden を w4 golden と呼ぶ事故）"
        )
    int8 = sum(1 for dtype, _scale in stored.values() if dtype == "i8")
    plain = len(stored) - len(int4_keys) - int8
    print(
        f"[fake-quant] {ex.TARGET_DIT}: i4 系列の出荷バイトで上書きした —"
        f" i4 {len(int4_keys)} 本（うち段 1 と値が違うもの {changed} 本）/"
        f" i8 {int8} 本 / f32 {plain} 本・校正 {calib}",
        flush=True,
    )
    return RestoredDit(
        container=container,
        calib=calib,
        int4=len(int4_keys),
        int8=int8,
        plain=plain,
        changed=changed,
    )


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
    #
    # MUST（i4 = 配布の quant 席 `w4`）: 丸めを 2 段に割る。段 1 はここで **i8 席と同一の
    # fake-quant**（w4 席は他 7 役に w8 の i8 バイトを共有させるので、条件エンコーダ側の丸めは
    # w8 golden と同一が正しい）、段 2 は {@link restore_dit_from_i4_series} が `dit` のラッパ
    # 所有パラメタだけを出荷バイトで上書きする。`irodori.export.fake_quant` の i4 経路を呼んで
    # 校正をここでもう 1 度走らせない理由は段 2 の docstring。
    quantized = ex.fake_quant(
        "i8" if dtype == "i4" else dtype,
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
    restored = (
        restore_dit_from_i4_series(
            # ラッパ所有パラメタは `dit` の Parameter そのもの（`DitGraph` は張り替えずに
            # 同じ属性名で抱える）なので、ラッパ経由の上書きが `dit` 側にも通る。
            ex.DitGraph(dit, ex.dit_sym_max(config)),
            ex.default_out_root(model_dir, dtype) / ex.TARGET_DIT,
        )
        if dtype == "i4"
        else None
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
            # ---- 2 段目: 増幅率を実測して正規化判定（定数コメントの追記 2026-09-01） ----
            # run_case は決定的なので再実行 + 摂動走行で増幅率だけを足す。z の同一性は
            # 判定の前提（違えば「同じ軌道の感度」を測れていない）なので明示的に確かめる。
            probed_tensors, probed_meta = run_case(
                case,
                graphs,
                source,
                duration_predictor,
                tokenizer,
                text_config,
                model_config,
                config,
                caps,
                sensitivity_probe=True,
            )
            if not torch.equal(probed_tensors["z"], tensors["z"]):
                raise SystemExit(
                    f"{case.name}: 再実行の z が一致しない — ホスト経路が非決定的で、"
                    "増幅率の実測が成立しない"
                )
            amp = float(probed_meta["sensitivityAmp"])
            meta["sensitivityAmp"] = probed_meta["sensitivityAmp"]
            if not euler_reference_within_sensitivity(worst, amp):
                raise SystemExit(
                    f"{case.name}: 最終 z が上流と {worst} 違う（1 段目の許容"
                    f" {EULER_REFERENCE_ATOL} を超過）。増幅率 {amp:.4g} で正規化しても"
                    f" {worst / amp:.4g} > {EULER_NOISE_PER_AMP}、または絶対上限"
                    f" {EULER_REFERENCE_ABS_CEILING} 超え — 誤差の蓄積では説明できない差で、"
                    " CFG 合成式 / t スケジュール / マスク区間のどれかが崩れている疑い"
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

    meta_payload: dict[str, Any] = {
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
    if restored is not None:
        # MUST: 既存キーを 1 つも動かさない（deno 側の latent 門が読む）。i4 のときだけ足す
        # 1 本で、`fakeQuant` が段 1（i8）しか語らないぶんの出所をここが受け持つ。
        meta_payload["i4Source"] = {
            "container": str(restored.container),
            "calib": restored.calib,
            "int4Tensors": restored.int4,
            "int8Tensors": restored.int8,
            "f32Tensors": restored.plain,
            "changedByRestore": restored.changed,
        }
    (out_dir / META_FILE).write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    written[META_FILE] = (out_dir / META_FILE).stat().st_size
    return {"dir": str(out_dir), "bytes": written, **meta_payload}


def default_out_dir(model_dir: Path, dtype: str = "f32") -> Path:
    """既定の置き場（`outputs/series/irodori-<実重みのディレクトリ名>{,-f16,-i8,-i4}/pipeline/`）。

    系列 root の綴りは `irodori.export.default_out_root` と同一 —
    `pipeline/` はその下の 1 段（グラフのターゲットと並ぶ席）。
    """
    return ex.default_out_root(model_dir, dtype) / "pipeline"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dtype",
        choices=ex.WEIGHT_DTYPES,
        default="f32",
        help="重みの格納 dtype（f16 / i8 は golden を fake-quant 後の重みで焼き直す"
        " — ADR 0018 / 0019 / 0027 / 0050。i4 は i8 席と同じ丸めの上に、export 済みの"
        " i4 系列（--model-dir から導く）の出荷バイトで dit を上書きする — ADR 0069）",
    )
    args = parser.parse_args(argv)
    out_dir = default_out_dir(args.model_dir, args.dtype) if args.out is None else args.out
    summary = emit(args.model_dir, args.source_dir, out_dir, args.dtype)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
