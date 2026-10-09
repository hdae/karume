"""Wan2.2 TI2V-5B の I2V の 2 ステップの通しの CPU 参照（ADR 0121 段 9b）。

条件画像 1 枚から 1280×704・9 フレーム（潜在 `[48,3,44,80]`・S = 2,640・条件のトークン P = 880）を
2 ステップ（CFG あり・guidance 5.0・shift 5.0）回した潜在と、それをタイル decode → unpatchify →
クランプした RGB のフレームを、diffusers の `WanImageToVideoPipeline` の `expand_timesteps` の分岐
（Wan2.2 5B の I2V）を CPU f32 で回して作る。GPU の通し（`WanTi2vPipeline` の I2V）はこの値と帯で
照合する。T2V の `wan/ti2v_few_step_ref.py` は変えずに部品を流用する（重みの読み手・scheduler・
プロンプトの行の引き方・タイル decode・2 相の順）。

## 条件づけ（diffusers の `expand_timesteps` の分岐そのもの）

- 条件の潜在: 上流の非タイル encode の mu を `(mu − mean)·f32(1/f32(std))` で正規化した値
  （`prepare_latents` — `[48,1,44,80]`）。
- 各 step: DiT の入力だけ先頭の潜在フレームを条件で置き換える（`(1 − m)·cond + m·latents`・m は
  先頭フレームが 0）。スケジューラは置き換えない潜在で進む（callback が受けるのもこの値）。
- ループの後に 1 回置き換えてから VAE へ（`latents_final`）。
- 時刻: 条件のトークン（先頭 P 個）は t = 0、残りは生成側の t。

公式（Wan2.2 の `textimage2video.py` の i2v）は初めと各 step の後にスケジューラの状態そのものを
置き換える。DiT の入力は 2 つの形でビット一致し、UniPC（thresholding なし）は要素ごとに進むので、
最終出力もビット一致する（{@link sample} が 2 つの形を 1 本の経路で書く。
`wan/tests/test_ti2v_i2v_few_step_ref.py` が pin の UniPC と擬似モデルで常設し、実走では記録した
DiT の出力から公式の形を再生して要約に書く — 観測であって門ではない）。

## 時刻の形（ADR 0121 決定 4）

上流の forward は `[1,S]` の timestep を受けると時刻の MLP を S 行（M = S）で回す。参照は Karume の
グラフと同じ意味論 — 値ごとに M = 1 で 2 回（生成側 t・条件側 0）— で採る。DiT の
`condition_embedder` を呼び出しの間だけ {@link _PerStepTokenwiseEmbedder} に差し替え、各 forward の
平坦の timestep から生成側の t を読んで DiT の参照ラッパ（`dit_patch._TokenwiseTimeEmbedder` —
DiT の golden と同じもの）へ委ねる。決定用の 2 本は差し替えずに（diffusers の pipeline そのもの・
M = S）も回して観測する（門ではない）。

## 入力

- 条件画像: `inputs/wan-i2v/<name>-832x480.png`（git 追跡外・sha256 は
  `export_vae_encoder.IMAGES` — 段 9a の golden の来歴と同じ）を公式の前処理
  （`i2v_preprocess_ref.crop_resize` — 覆う側の倍率で LANCZOS → 中央クロップ）で切った exact-size の
  PIL を渡す。上流の `VideoProcessor` はこの寸法の画像の画素を変えない（Pillow は同寸法の resize を
  写しで返す）— pipeline が encoder へ渡す [-1, 1] の画像が公式の `to_signed_unit` とビット一致する
  ことを確かめる（外れたら fail loudly）。
- テキスト文脈・初期ノイズ・重み・scheduler: T2V の参照と同じ（`ti2v_few_step_ref` のモジュール
  doc）。プロンプトは画像と同じ名前の固定プロンプト。negative は明示する（diffusers の既定 `""` にも
  公式の既定 `sample_neg_prompt` にも預けない — T2V と同じ扱い）。
- pipeline: pin の snapshot から `WanImageToVideoPipeline.from_pretrained` で組む
  （`expand_timesteps` は model_index の値 — 読めた値が真であることを確かめる）。重い部品は全て
  渡すので、snapshot から読むのは config だけ（repo id + `local_files_only` は未取得のファイルの
  ために `IncompleteSnapshotError` になるので、取得済みの snapshot のパスを渡す）。

## 組み方（単一プロセス・2 相 — T2V の参照と同じ）

1. 第 1 相（DiT）: 全ケースを 2 ステップ回す（VAE の encode は pipeline の中）。DiT の出力は
   forward の hook で記録し、cond / uncond の振り分けは hook が受けたテキスト文脈の値で決める。
   決定用の 2 本は M = S の経路でも回す。
2. 第 2 相（VAE）: DiT を手放し（参照が残っていたら fail loudly）、置き換えた潜在を段 5 の 2.2 の
   経路（`vae_tiling` — patchify 空間でブレンド → 上流の unpatchify → clamp）でフレームにする。

## 出力（i8 の DiT の系列の根の `pipeline_steps_i2v.<case>.safetensors`）

テンソル（f32 — 画像の 2 本だけ u8）:

- `latents_init` `[48,3,44,80]` — 注入した初期ノイズ（先頭フレームもノイズのまま — 置き換えは
  pipeline の中）。
- `condition_latents` `[48,1,44,80]` — 条件の潜在（正規化の後）。
- `noise_cond.<i>` / `noise_uncond.<i>` — 各 step の DiT の出力。
- `latents.<i>` — 各 step の後のスケジューラの潜在（置き換える前 — diffusers の callback と同じ
  値）。
- `latents_final` — ループの後に先頭フレームを置き換えた潜在（VAE へ渡す値）。
- `frames` `[3,9,704,1280]` — unpatchify とクランプの後。
- `latents_mean` / `latents_std` `[48]`。
- `source` `[480,832,3]` u8 — 元画像の RGB8（来歴の sha256 の PNG の画素 — GPU のテストが PNG の
  decode にも別の資産にも頼らずに前処理の鎖を回せる）・`rgb8` `[704,1280,3]` u8 — 切った RGB8。
- 決定用だけ `observed_m_s.latents.<i>` / `observed_m_s.latents_final`（M = S の観測）。

メタ: T2V の参照の欄（`expand_timesteps` は `"true"`）に加えて、`timestep_mlp`
（{@link TIMESTEP_MLP}）・`condition_tokens`（P）・`condition_timestep`（`"0"`）・`image`・
`image_sha256`（PNG）・`source_rgb8_sha256` / `rgb8_sha256`（u8 の生バイト列）・`width`・`height`・
`fit`（`"crop"`）・`resample`（`"LANCZOS"`）・`version_<lib>`（値を決めるライブラリの版）。

## ケース（決定用 2 本 + 受入れ 1 本 — 3 本とも同じ形）

役割は段 9a の encoder の golden と同じ割り振り（決定用 = boxing-cats / cat-dog-baking・受入れ =
ferret）。MUST: `accept` の結果を見て決定用のケースも帯も変えない。受入れの形を決定用と変えない。

    uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref                       # 全部
    uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref --case accept-ferret  # 1 本

## RAM と所要（実測 2026-10-09・Ryzen 5 7600・torch のスレッド 6）

- RAM の門の見込みは T2V の参照の見込み（`ti2v_few_step_ref.memory_estimate` 23.7 GiB）に
  I2V の上乗せ 0.5 GiB を足した約 24.2 GiB（{@link memory_estimate} — `require_available` が
  余白 1 GiB を足す）。プロセスの RSS の山は 24.07 GiB で、T2V の見込みを 0.37 GiB 上回った
  （上乗せはその実測から）。S は T2V より短いが山は下がらなかった。
- 所要は 3 ケースで 41 分 48 秒: 第 1 相 約 16 分（M = 1 が 1 ケース 179〜202 s・M = S の観測が
  190〜202 s）+ タイル decode 28 枚 × 3 chunk が 1 ケース 503〜506 s。
- 第 1 相の結果はメモリにだけ持つ（第 2 相の途中で落ちたら第 1 相から回し直す）。

MUST: diffusers / Pillow は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared.paths import SERIES_ROOT
from wan import (
    dit_patch,
    dit_probe,
    export_vae,
    export_vae_encoder,
    i2v_preprocess_ref,
    pipeline_ref,
    text_embeds,
    ti2v_few_step_ref,
    vae_tiling,
)
from wan.sources import local_snapshot

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
    from PIL import Image

#: 上流のモデル（`wan.sources.SOURCES` のキー）。
MODEL = ti2v_few_step_ref.MODEL

#: 少ステップの通しのステップ数・shift・CFG の強さ（T2V の参照と同じ値）。
STEPS = ti2v_few_step_ref.STEPS
FLOW_SHIFT = ti2v_few_step_ref.FLOW_SHIFT
GUIDANCE_SCALE = ti2v_few_step_ref.GUIDANCE_SCALE

#: 潜在 `[C, F, H, W]`（1280×704・9 フレーム — 条件 1 + 生成 2 の潜在フレームで、置き換えの
#: フレームの添字のずれ〈例: 1 つおきに置き換える〉まで観測点に出る）。
LATENT_SHAPE = (48, 3, 44, 80)

#: 縦横の合わせ方（公式の前処理 — `i2v_preprocess_ref.crop_resize`）と、その resample。
FIT: i2v_preprocess_ref.Fit = "crop"
RESAMPLE = "LANCZOS"

#: 条件のトークンの時刻（上流の `first_frame_mask` が先頭の潜在フレームを 0 にする）。
CONDITION_TIMESTEP = 0

#: 系列の根（i8 の DiT が参照席を特徴づけるので、T2V の参照と同じ i8 の DiT の系列に置く）。
SERIES = ti2v_few_step_ref.SERIES

#: 参照のファイル名（系列の根に置く — `pipeline_steps_i2v.<case>.safetensors`。T2V の
#: `pipeline_steps.<case>` と同じ根に並ぶ。配布形は許可リスト〈`distribution.wan_placements`〉なので
#: ここに足しても配布へは入らない）。
FIXTURE_PREFIX = "pipeline_steps_i2v."
FIXTURE_SUFFIX = ".safetensors"

#: メタの `weights`（T2V の参照と同じ — DiT は i8 の RTN の fake-quant・VAE は f16 表現可能値）。
WEIGHTS = ti2v_few_step_ref.WEIGHTS

#: メタの `timestep_mlp`（時刻の MLP の回し方 — 決定 4: 値ごとに M = 1・生成側 t と条件側 0）。
TIMESTEP_MLP = "per-value-m1"

#: メタの `reference`（参照の素性）。
REFERENCE = (
    "diffusers WanImageToVideoPipeline (expand_timesteps=True read from the pinned model_index) CPU"
    " f32, umT5 embeddings injected, the time MLP run once per value at M=1 (generation t,"
    " condition 0; ADR 0121 decision 4), the condition image cropped by the official Wan2.2"
    " preprocessing; then the snapped-tile decode of the final replaced latents with upstream"
    " blend_v / blend_h in patchify space, upstream unpatchify and clamp"
)

#: 決定用のメタの `observed_m_s`（観測の素性 — 門ではない）。
OBSERVED_M_S = (
    "scheduler latents after each step and the final replaced latents of the plain diffusers"
    " WanImageToVideoPipeline (timestep [1,S], the time MLP over all S tokens); observation only,"
    " not a gate"
)


@dataclass(frozen=True)
class FixtureCase(ti2v_few_step_ref.FixtureCase):
    """I2V の少ステップの通し 1 本（T2V のケースに条件画像の名前を足したもの）。"""

    #: 条件画像の名前（`export_vae_encoder.IMAGES` の `name`）。
    image: str


#: 決定用 2 本（帯を決める）+ 受入れ 1 本。プロンプトは画像と同じ名前の固定プロンプト。seed は
#: T2V の参照（20262101〜03）とも 2.1 の参照とも違う値。
#: MUST: `accept` の結果を見て決定用のケースも帯も変えない。
FIXTURE_CASES = (
    FixtureCase(
        "band-boxing-cats", role="band", seed=20262111, prompt="boxing-cats", image="boxing-cats"
    ),
    FixtureCase(
        "band-cat-dog-baking",
        role="band",
        seed=20262112,
        prompt="cat-dog-baking",
        image="cat-dog-baking",
    ),
    FixtureCase("accept-ferret", role="accept", seed=20262113, prompt="ferret", image="ferret"),
)


def fixture_path(out_root: Path, case: FixtureCase) -> Path:
    return out_root / f"{FIXTURE_PREFIX}{case.name}{FIXTURE_SUFFIX}"


#: I2V の山が T2V の見込み（`ti2v_few_step_ref.memory_estimate`）を上回った分の上乗せ。
#: 実測（2026-10-09・3 ケース）の山 24.07 GiB は T2V の見込み 23.70 GiB を 0.37 GiB 上回った —
#: 切り上げて 0.5 GiB。出所は未特定（推測: pipeline の中の VAE の encode と M = S の観測の活性）。
I2V_EXCESS_BYTES = int(0.5 * 2**30)


def memory_estimate(model: str = MODEL) -> int:
    """I2V の参照の RAM の見込み（バイト）= T2V の見込み + {@link I2V_EXCESS_BYTES}（5B で約
    24.2 GiB — 実測の山 24.07 GiB を覆う。`require_available` が余白 1 GiB を足す）。

    MUST: 実測の山を下回る見込みにしない — 余白 1 GiB が見込みの誤差に食われ、他の作業と並べたときに
    42 分の走行の途中で OOM かスワップに入る。
    """
    return ti2v_few_step_ref.memory_estimate(model) + I2V_EXCESS_BYTES


# ---- 条件画像 ------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceImage:
    """条件画像の元 1 枚（来歴の名前と PNG の sha256・開いた RGB の画像）。"""

    name: str
    sha256: str
    image: Image.Image


@dataclass(frozen=True)
class ConditionImage:
    """1 ケースの条件画像（元画像と、公式の前処理で切った exact-size の画像）。"""

    source: SourceImage
    cropped: Image.Image

    @property
    def source_rgb8(self) -> torch.Tensor:
        return i2v_preprocess_ref.rgb8(self.source.image)

    @property
    def rgb8(self) -> torch.Tensor:
        return i2v_preprocess_ref.rgb8(self.cropped)

    @property
    def signed(self) -> torch.Tensor:
        """公式の [-1, 1]（`to_signed_unit`）の `[1, 3, H, W]` — pipeline が encoder へ渡す画像の
        照合の相手。"""
        return i2v_preprocess_ref.to_signed_unit(self.cropped)[:, :, 0]


def condition_image(source: SourceImage, width: int, height: int) -> ConditionImage:
    """元画像を公式の前処理（`crop`）で `width × height` に切る。"""
    return ConditionImage(
        source=source, cropped=i2v_preprocess_ref.preprocess(source.image, width, height, FIT)
    )


def load_sources(cases: Sequence[FixtureCase]) -> dict[str, SourceImage]:
    """ケースが使う条件画像を開く（sha256 と寸法が来歴と違えば fail loudly — 重い読み込みの前に
    呼ぶ）。"""
    images = {image.name: image for image in export_vae_encoder.IMAGES}
    sources: dict[str, SourceImage] = {}
    for name in sorted({case.image for case in cases}):
        image = images.get(name)
        if image is None:
            raise ValueError(f"条件画像 {name!r} が来歴の表に無い（{sorted(images)}）")
        sources[name] = SourceImage(name, image.sha256, export_vae_encoder.load_image(image))
    return sources


def _bytes_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


# ---- 条件づけ（置き換えの式・2 つの形）-------------------------------------------------


# Third-party code notice. `first_frame_mask`, `replace_first_frame` and `sample` below reproduce
# the conditioning of the Wan2.2 TI2V-5B image-to-video loop in two upstreams:
#
# - the `expand_timesteps` branch of `WanImageToVideoPipeline` in huggingface/diffusers
#   (`src/diffusers/pipelines/wan/pipeline_wan_i2v.py`, `diffusers==0.39.0`: the first-frame mask
#   of `prepare_latents`, lines 461-466; the model-input replacement, the CFG and the scheduler
#   step of `__call__`, lines 761, 792 and 795; the replacement after the loop, line 817).
#   License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
#   copyright notice,
#   copied verbatim from the header of that file:
#
#     Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# - `WanTI2V.i2v` in Wan-Video/Wan2.2 (`wan/textimage2video.py`, commit
#   1ea34ff48f87168174e12956e200b1d908b1c5ff: the replacement before the loop, line 551, and after
#   each step, line 598). License: Apache License, Version 2.0. Upstream copyright notice, copied
#   verbatim from the header of that file:
#
#     Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
def first_frame_mask(latent_shape: Sequence[int]) -> torch.Tensor:
    """上流の `first_frame_mask` `[1, 1, F, H, W]`（f32 — 先頭の潜在フレームが 0・残りが 1）。"""
    _, frames, height, width = latent_shape
    mask = torch.ones(1, 1, frames, height, width, dtype=torch.float32)
    mask[:, :, 0] = 0
    return mask


def replace_first_frame(
    latents: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """先頭の潜在フレームを条件で置き換える `(1 − m)·cond + m·latents`（diffusers と公式の同じ式）。

    `condition` は `[1, C, 1, H, W]`（フレーム軸で放送）。式のまま書く — 0·cond は ±0 を足すだけで
    先頭以外のフレームは `latents` とビット一致するが、cond の ±Inf は 0·Inf = NaN として全
    フレームへ広がる（だから条件の潜在は有限であることを先に確かめる）。
    """
    return (1 - mask) * condition + mask * latents


#: 置き換えの形（{@link sample}）。
Form = Literal["diffusers", "official"]
FORMS: tuple[Form, ...] = ("diffusers", "official")

#: 1 step の速度（`index`・timestep・DiT の入力 → CFG の後の DiT の出力）。
Velocity = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Sampled:
    """{@link sample} の戻り（`[1, C, F, H, W]` の列）。"""

    #: 各 step の DiT の入力。
    model_inputs: list[torch.Tensor]
    #: 各 step の scheduler の出力（その形の置き換えの前）。
    states: list[torch.Tensor]
    #: VAE へ渡す潜在（先頭フレームが条件）。
    final: torch.Tensor


def sample(
    scheduler: UniPCMultistepScheduler,
    latents_init: torch.Tensor,
    condition: torch.Tensor,
    mask: torch.Tensor,
    velocity: Velocity,
    *,
    form: Form,
) -> Sampled:
    """I2V の denoise のループを 2 つの置き換えの形のどちらかで回す（`scheduler` は
    `set_timesteps` 済みの新しいもの）。

    - `diffusers`（`WanImageToVideoPipeline` の `expand_timesteps` の分岐）: DiT の入力だけを
      置き換え、scheduler は置き換えない潜在で進め、ループの後に 1 回置き換える。
    - `official`（公式 Wan2.2 の i2v）: 初めと各 step の後に scheduler の状態そのものを置き換える。

    DiT の入力は 2 つの形で同じ値になる（先頭フレームはどちらも条件・残りは同じ scheduler の
    出力）。scheduler の状態は先頭フレームの領域だけが違い、最後の置き換えでその違いが消える。
    """
    if form not in FORMS:
        raise ValueError(f"置き換えの形 {form!r} は未知（{', '.join(FORMS)}）")
    official = form == "official"
    latents = replace_first_frame(latents_init, condition, mask) if official else latents_init
    model_inputs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for index, timestep in enumerate(scheduler.timesteps):
        model_input = latents if official else replace_first_frame(latents, condition, mask)
        model_inputs.append(model_input)
        stepped = scheduler.step(
            velocity(index, timestep, model_input), timestep, latents, return_dict=False
        )[0]
        states.append(stepped)
        latents = replace_first_frame(stepped, condition, mask) if official else stepped
    final = latents if official else replace_first_frame(latents, condition, mask)
    return Sampled(model_inputs=model_inputs, states=states, final=final)


def fresh_scheduler(config: Mapping[str, Any], steps: int) -> UniPCMultistepScheduler:
    """同じ config の新しい UniPC を `steps` で `set_timesteps` したもの（再生用 — pipeline と
    状態を共有しない）。"""
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_config(config)
    scheduler.set_timesteps(steps)
    scheduler.set_begin_index(0)
    return scheduler


def recorded_velocity(
    noise_cond: Sequence[torch.Tensor], noise_uncond: Sequence[torch.Tensor], guidance: float
) -> Velocity:
    """記録した DiT の出力（`[C, F, H, W]`）から上流と同じ CFG を組む速度（DiT の入力は読まない —
    2 つの形で DiT の入力が同じであることは呼び手が `model_inputs` で確かめる）。"""

    def velocity(index: int, _timestep: torch.Tensor, _model_input: torch.Tensor) -> torch.Tensor:
        uncond = noise_uncond[index][None]
        return uncond + guidance * (noise_cond[index][None] - uncond)

    return velocity


# ---- 時刻の形（決定 4 の参照ラッパを pipeline の各 forward に当てる）------------------------


def condition_token_mask(transformer: nn.Module, latent_shape: Sequence[int]) -> torch.Tensor:
    """条件のトークンのマスク `[1, S]`（bool — 先頭の潜在フレームの P = H'·W' トークンが真 —
    グラフ入力 `condition_mask` と同じ `dit_patch.dit_condition_mask`）。"""
    _, frames, height, width = latent_shape
    patch = tuple(transformer.config.patch_size)
    return dit_patch.dit_condition_mask((frames, height, width), patch, conditioned=True)[..., 0]


class _PerStepTokenwiseEmbedder(nn.Module):
    """各 forward の平坦の timestep `[S]` から生成側の t を読み、DiT の参照ラッパ
    （`dit_patch._TokenwiseTimeEmbedder` — 時刻の MLP を値ごとに M = 1 で 2 回）へ委ねる差し替え。

    pipeline は step ごとに t を変えるが、参照ラッパは生成側の t を構築時に受ける — ここが forward
    ごとにそれを組む。生成側の t は int64 で渡す（T2V の参照の M = 1 の経路の `t.expand(1)` と同じ
    dtype — sinusoidal は `.float()` するので値も同じ）。

    MUST: 生成側のトークンの timestep が 1 値でなければ・整数でなければ fail loudly。条件側が 0 で
    あることとマスクとの対応は参照ラッパが確かめる（`where(mask, 0, t)` との一致）。
    """

    def __init__(self, embedder: nn.Module, mask: torch.Tensor, seen: list[int]) -> None:
        super().__init__()
        self.embedder = embedder
        self.mask = mask
        self.seen = seen

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        tokens = self.mask.shape[1]
        if tuple(timestep.shape) != (tokens,) or timestep_seq_len != tokens:
            raise AssertionError(
                f"上流が渡した timestep の形 {tuple(timestep.shape)}・seq_len {timestep_seq_len} が"
                f" トークンごとの ({tokens},) でない（expand_timesteps の分岐のはず）"
            )
        values = timestep[~self.mask[0]].unique()
        if values.numel() != 1:
            raise AssertionError(
                f"生成側のトークンの timestep が {values.numel()} 値（1 値のはず）"
            )
        value = float(values[0])
        if not value.is_integer():
            raise AssertionError(f"生成側の timestep {value} が整数でない")
        self.seen.append(int(value))
        condition = dit_patch.TimestepCondition(
            torch.tensor([CONDITION_TIMESTEP], dtype=torch.int64), self.mask
        )
        embedder = dit_patch._TokenwiseTimeEmbedder(
            self.embedder, torch.tensor([int(value)], dtype=torch.int64), condition
        )
        return embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len
        )


@contextmanager
def tokenwise_time_embedding(transformer: nn.Module, mask: torch.Tensor) -> Iterator[list[int]]:
    """この間だけ DiT の `condition_embedder` を {@link _PerStepTokenwiseEmbedder} に差し替える
    （戻り = forward ごとの生成側の t の列）。抜けるときは例外でも元へ戻す。"""
    original = transformer.condition_embedder
    seen: list[int] = []
    transformer.condition_embedder = _PerStepTokenwiseEmbedder(original, mask, seen)
    try:
        yield seen
    finally:
        transformer.condition_embedder = original


# ---- pipeline -------------------------------------------------------------------------


def build_pipeline(
    snapshot: Path,
    transformer: nn.Module,
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
) -> WanImageToVideoPipeline:
    """pin の snapshot から umT5 無しの `WanImageToVideoPipeline` を組む（重い部品は全て渡す）。

    `expand_timesteps` は model_index の値（pin は真）— 偽なら Wan2.1 の I2V（チャネル連結）の分岐に
    なるので止まる。2 段の DiT（`boundary_ratio`）と画像 encoder（Wan2.1 の I2V の CLIP）は
    使わない。
    """
    pipeline_ref.assert_no_mps()
    from diffusers import WanImageToVideoPipeline

    pipeline = WanImageToVideoPipeline.from_pretrained(
        snapshot,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
    )
    if pipeline.config.expand_timesteps is not True:
        raise AssertionError(
            f"model_index の expand_timesteps が {pipeline.config.expand_timesteps}（Wan2.2 5B の"
            " I2V の分岐は真）"
        )
    if pipeline.config.boundary_ratio is not None or pipeline.transformer_2 is not None:
        raise AssertionError("2 段の DiT（boundary_ratio / transformer_2）は使わない")
    if pipeline.image_encoder is not None or transformer.config.image_dim is not None:
        raise AssertionError("画像 encoder（Wan2.1 の I2V の CLIP の経路）は使わない")
    if (
        pipeline.transformer is not transformer
        or pipeline.vae is not vae
        or pipeline.scheduler is not scheduler
    ):
        raise AssertionError("pipeline が渡した部品と別のものを持っている")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@contextmanager
def captured_condition(
    pipeline: WanImageToVideoPipeline, image: ConditionImage, mask: torch.Tensor
) -> Iterator[dict[str, torch.Tensor]]:
    """この間の `prepare_latents` の戻りの条件の潜在を写し取る（戻り = `{"condition": …}`）。

    MUST（どれも fail loudly）: pipeline が encoder へ渡す [-1, 1] の画像が公式の前処理とビット
    一致すること・上流の `first_frame_mask` が {@link first_frame_mask} と同じこと・条件の潜在が
    有限であること（cond の ±Inf は置き換えの式で全フレームへ NaN として広がる）。
    """
    captured: dict[str, torch.Tensor] = {}
    prepare = pipeline.prepare_latents
    expected_image = image.signed

    def capture(sample: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if not torch.equal(sample, expected_image):
            raise AssertionError(
                f"{image.source.name}: pipeline の前処理の画像が公式の [-1, 1] とビット一致しない"
            )
        latents, condition, upstream_mask = prepare(sample, *args, **kwargs)
        if not torch.equal(upstream_mask, mask):
            raise AssertionError(f"上流の first_frame_mask {tuple(upstream_mask.shape)} が違う")
        if not bool(condition.isfinite().all()):
            raise AssertionError(f"{image.source.name}: 条件の潜在に非有限値がある")
        captured["condition"] = condition.detach().clone()
        return latents, condition, upstream_mask

    pipeline.prepare_latents = capture
    try:
        yield captured
    finally:
        del pipeline.prepare_latents


def _call(
    pipeline: WanImageToVideoPipeline,
    embeds: Mapping[str, torch.Tensor],
    names: ti2v_few_step_ref.PromptNames,
    latents_init: torch.Tensor,
    image: ConditionImage,
    record_step: Callable[..., dict[str, Any]],
) -> torch.Tensor:
    num_frames, height, width = ti2v_few_step_ref.video_size(pipeline, latents_init.shape[1:])
    output = pipeline(
        image=image.cropped,
        prompt_embeds=pipeline_ref.pad_text_embeds(embeds[names.positive]),
        negative_prompt_embeds=pipeline_ref.pad_text_embeds(embeds[names.negative]),
        latents=latents_init.clone(),
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        output_type="latent",
        return_dict=False,
        callback_on_step_end=record_step,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    result: torch.Tensor = output[0]
    return result


@dataclass
class Denoised:
    """第 1 相の 1 ケースぶん（第 2 相と書き出しへ渡す — 潜在は `[C, F, H, W]`）。"""

    case: FixtureCase
    names: ti2v_few_step_ref.PromptNames
    image: ConditionImage
    latents_init: torch.Tensor
    #: 条件の潜在 `[C, 1, H, W]`（正規化の後）。
    condition: torch.Tensor
    #: 条件のトークンの数 P（DiT の patch で数えた先頭の潜在フレームのトークン）。
    condition_tokens: int
    #: 各 step の DiT の入力 `[1, C, F, H, W]`（書き出さない — 公式の形の再生の照合用）。
    model_inputs: list[torch.Tensor]
    noise_cond: list[torch.Tensor]
    noise_uncond: list[torch.Tensor]
    #: 各 step の後の scheduler の潜在（置き換える前）。
    latents: list[torch.Tensor]
    #: ループの後に置き換えた潜在（VAE へ渡す値）。
    latents_final: torch.Tensor
    timesteps: list[int]
    sigmas: list[float]
    seconds: dict[str, float]
    #: M = S の経路の各 step の潜在・置き換えた潜在・条件の潜在が M = 1 と同じか（決定用だけ）。
    observed: list[torch.Tensor] | None = None
    observed_final: torch.Tensor | None = None
    observed_condition_equal: bool | None = None


def denoise_case(
    pipeline: WanImageToVideoPipeline,
    case: FixtureCase,
    embeds: Mapping[str, torch.Tensor],
    names: ti2v_few_step_ref.PromptNames,
    image: ConditionImage,
) -> Denoised:
    """決定 4 の時刻の形（値ごとに M = 1）で 1 ケースを 2 ステップ回す。"""
    generator = torch.Generator().manual_seed(case.seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)
    tokens = condition_token_mask(pipeline.transformer, LATENT_SHAPE)
    mask = first_frame_mask(LATENT_SHAPE)

    # 上流へ渡す文脈と同じ値（`_call` が同じ関数で埋める）— hook はこれと値で突き合わせる。
    contexts = {
        "cond": pipeline_ref.pad_text_embeds(embeds[names.positive]),
        "uncond": pipeline_ref.pad_text_embeds(embeds[names.negative]),
    }
    forwards: list[tuple[int, str, torch.Tensor, torch.Tensor]] = []

    def forward_recorder(captured: Mapping[str, torch.Tensor]) -> Callable[..., None]:
        def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], output: Any) -> None:
            timestep = kwargs["timestep"]
            if tuple(timestep.shape) != (1, tokens.shape[1]):
                raise AssertionError(
                    f"{case.name}: timestep の形 {tuple(timestep.shape)} が"
                    f" (1, {tokens.shape[1]}) でない（expand_timesteps の分岐のはず）"
                )
            hidden = kwargs["hidden_states"]
            # 置き換えが DiT の入力に届いていること（先頭フレーム = 条件の潜在）。
            if not torch.equal(hidden[:, :, :1], captured["condition"]):
                raise AssertionError(f"{case.name}: DiT の入力の先頭フレームが条件の潜在でない")
            context = kwargs["encoder_hidden_states"]
            labels = [label for label, value in contexts.items() if torch.equal(context, value)]
            if len(labels) != 1:
                raise AssertionError(
                    f"{case.name}: DiT の文脈が cond / uncond の埋め込みの {len(labels)} 本と"
                    "一致する（1 本のはず）— 振り分けを決められない"
                )
            generation = int(timestep[0][~tokens[0]].max())
            forwards.append(
                (generation, labels[0], output[0].detach().clone(), hidden.detach().clone())
            )

        return record_forward

    steps: list[torch.Tensor] = []
    started = time.perf_counter()
    with (
        torch.no_grad(),
        dit_patch.flash_attention_only(),
        captured_condition(pipeline, image, mask) as captured,
        tokenwise_time_embedding(pipeline.transformer, tokens) as seen,
    ):
        handle = pipeline.transformer.register_forward_hook(
            forward_recorder(captured), with_kwargs=True
        )
        try:
            final = _call(
                pipeline,
                embeds,
                names,
                latents_init,
                image,
                ti2v_few_step_ref._step_recorder(steps),
            )
        finally:
            handle.remove()
    seconds = time.perf_counter() - started
    timesteps = [int(value) for value in pipeline.scheduler.timesteps]
    sigmas = [float(value) for value in pipeline.scheduler.sigmas]
    expected = [t for t in timesteps for _ in range(2)]
    if len(forwards) != 2 * STEPS or len(steps) != STEPS:
        raise AssertionError(
            f"forward {len(forwards)} 回・step {len(steps)} 回（CFG あり 2 ステップ）"
        )
    if [t for t, _, _, _ in forwards] != expected or seen != expected:
        raise AssertionError(
            f"forward の timestep {[t for t, _, _, _ in forwards]}・時刻の MLP の t {seen} が"
            f" {timesteps} と合わない"
        )
    outputs: dict[tuple[int, str], torch.Tensor] = {}
    model_inputs: dict[tuple[int, str], torch.Tensor] = {}
    for index, (_, label, output, hidden) in enumerate(forwards):
        outputs[(index // 2, label)] = output
        model_inputs[(index // 2, label)] = hidden
    if sorted(outputs) != [(step, label) for step in range(STEPS) for label in sorted(contexts)]:
        raise AssertionError(
            "ステップごとの cond / uncond が 1 本ずつでない"
            f"（{[label for _, label, _, _ in forwards]}）"
        )
    for step in range(STEPS):
        if not torch.equal(model_inputs[(step, "cond")], model_inputs[(step, "uncond")]):
            raise AssertionError(f"{case.name}: step {step} の cond / uncond の DiT の入力が違う")
    return Denoised(
        case=case,
        names=names,
        image=image,
        latents_init=latents_init[0],
        condition=captured["condition"][0],
        condition_tokens=int(tokens.sum()),
        model_inputs=[model_inputs[(step, "cond")] for step in range(STEPS)],
        noise_cond=[outputs[(step, "cond")][0] for step in range(STEPS)],
        noise_uncond=[outputs[(step, "uncond")][0] for step in range(STEPS)],
        latents=[state[0] for state in steps],
        latents_final=final[0],
        timesteps=timesteps,
        sigmas=sigmas,
        seconds={"denoise": round(seconds, 1)},
    )


def observe_m_s(
    pipeline: WanImageToVideoPipeline,
    denoised: Denoised,
    embeds: Mapping[str, torch.Tensor],
) -> None:
    """diffusers の pipeline そのもの（時刻の MLP を S 行で回す M = S の経路）で同じ要求を回し、
    `denoised` の観測の欄を埋める（観測 — 門ではない）。

    hook は M = 1 とは別のもの: 時刻の差し替えが無いこと、timestep `[1,S]` が条件のトークンで 0・
    生成側で 1 値であることを確かめ、M = 1 と同じ時刻の列かを見る。
    """
    case = denoised.case
    transformer = pipeline.transformer
    if isinstance(transformer.condition_embedder, _PerStepTokenwiseEmbedder):
        raise AssertionError(f"{case.name}: M = S の観測に時刻の差し替えが残っている")
    tokens = condition_token_mask(transformer, LATENT_SHAPE)
    seen: list[int] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], _output: Any) -> None:
        timestep = kwargs["timestep"]
        if tuple(timestep.shape) != (1, tokens.shape[1]):
            raise AssertionError(
                f"{case.name}: M = S の経路の timestep の形 {tuple(timestep.shape)} が"
                f" (1, {tokens.shape[1]}) でない"
            )
        values = timestep[0][~tokens[0]].unique()
        if values.numel() != 1 or bool(timestep[0][tokens[0]].any()):
            raise AssertionError(
                f"{case.name}: M = S の timestep が条件のトークンで 0・生成側で 1 値でない"
            )
        seen.append(int(values[0]))

    steps: list[torch.Tensor] = []
    handle = transformer.register_forward_hook(record_forward, with_kwargs=True)
    started = time.perf_counter()
    try:
        with (
            torch.no_grad(),
            dit_patch.flash_attention_only(),
            captured_condition(pipeline, denoised.image, first_frame_mask(LATENT_SHAPE)) as cond,
        ):
            final = _call(
                pipeline,
                embeds,
                denoised.names,
                denoised.latents_init[None],
                denoised.image,
                ti2v_few_step_ref._step_recorder(steps),
            )
    finally:
        handle.remove()
    denoised.seconds["observe_m_s"] = round(time.perf_counter() - started, 1)
    if seen != [t for t in denoised.timesteps for _ in range(2)] or len(steps) != STEPS:
        raise AssertionError(
            f"{case.name}: M = S の forward の timestep {seen}・step {len(steps)} 回が"
            f" M = 1 の {denoised.timesteps} と合わない"
        )
    denoised.observed = [state[0] for state in steps]
    denoised.observed_final = final[0]
    denoised.observed_condition_equal = torch.equal(cond["condition"][0], denoised.condition)


def denoise_phase(
    load_transformer: Callable[[], nn.Module],
    snapshot: Path,
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    cases: Sequence[FixtureCase],
    embeds: Mapping[str, torch.Tensor],
    names: Mapping[str, ti2v_few_step_ref.PromptNames],
    sources: Mapping[str, SourceImage],
) -> tuple[list[Denoised], weakref.ref[nn.Module]]:
    """第 1 相: DiT を読み、全ケースを回す。DiT への参照はこの関数の中だけに置く（戻ると手放す）。

    戻りの弱参照は、第 2 相の前に DiT が回収されたことを確かめるため
    （`ti2v_few_step_ref.assert_released`）。
    """
    transformer = load_transformer()
    released = weakref.ref(transformer)
    pipeline = build_pipeline(snapshot, transformer, vae, scheduler)
    _, height, width = ti2v_few_step_ref.video_size(pipeline, LATENT_SHAPE)
    results: list[Denoised] = []
    for case in cases:
        image = condition_image(sources[case.image], width, height)
        denoised = denoise_case(pipeline, case, embeds, names[case.name], image)
        if case.observes_m_s:
            observe_m_s(pipeline, denoised, embeds)
        print(f"[denoise] {case.name}: {json.dumps(denoised.seconds)}", flush=True)
        results.append(denoised)
    return results, released


# ---- 再生と観測 ------------------------------------------------------------------------


def replay(denoised: Denoised, scheduler_config: Mapping[str, Any]) -> dict[str, Any]:
    """記録した DiT の出力から 2 つの形を再生する。

    MUST（fail loudly）: diffusers の形が pipeline の記録（DiT の入力・各 step の潜在・置き換えた
    潜在）をビット一致で作り直すこと — 記録した cond / uncond と CFG・置き換えの式の自己整合。
    公式の形は観測（DiT の入力が diffusers の形と同じ値なら、記録した出力で公式の形を正しく
    再生できる — その前提と、最終出力の一致を要約に書く）。
    """
    condition = denoised.condition[None]
    mask = first_frame_mask(LATENT_SHAPE)
    velocity = recorded_velocity(denoised.noise_cond, denoised.noise_uncond, GUIDANCE_SCALE)
    runs = {
        form: sample(
            fresh_scheduler(scheduler_config, STEPS),
            denoised.latents_init[None],
            condition,
            mask,
            velocity,
            form=form,
        )
        for form in FORMS
    }
    upstream = runs["diffusers"]
    name = denoised.case.name
    for index in range(STEPS):
        if not torch.equal(upstream.model_inputs[index], denoised.model_inputs[index]):
            raise AssertionError(f"{name}: 再生した step {index} の DiT の入力が記録と違う")
        if not torch.equal(upstream.states[index][0], denoised.latents[index]):
            raise AssertionError(f"{name}: 再生した step {index} の潜在が記録と違う")
    if not torch.equal(upstream.final[0], denoised.latents_final):
        raise AssertionError(f"{name}: 再生した置き換えた潜在が記録と違う")
    official = runs["official"]
    return {
        "model_inputs_bit_exact": all(
            torch.equal(got, want)
            for got, want in zip(official.model_inputs, denoised.model_inputs, strict=True)
        ),
        "final_bit_exact": torch.equal(official.final[0], denoised.latents_final),
        "final_max_abs": float((official.final[0] - denoised.latents_final).abs().max()),
        "states_first_frame_max_abs": [
            float((got[0, :, 0] - want[:, 0]).abs().max())
            for got, want in zip(official.states, denoised.latents, strict=True)
        ],
    }


#: 段 9a の encoder の golden と突き合わせるテンソル（golden のキー → この参照の値の取り出し）。
_ENCODER_GOLDEN_KEYS: Mapping[str, Callable[[Denoised], torch.Tensor]] = {
    "source": lambda denoised: denoised.image.source_rgb8,
    "rgb8": lambda denoised: denoised.image.rgb8,
    "latent": lambda denoised: denoised.condition,
}


def check_encoder_golden(denoised: Denoised, root: Path | None) -> dict[str, Any] | str:
    """元画像・切った画像・条件の潜在を段 9a の encoder の golden（同じ画像・寸法の `source` /
    `rgb8` / `latent`）と突き合わせる。

    MUST（fail loudly）: golden があればビット一致すること。外れたまま書くと、TS の通しの e2e の
    帯と条件の潜在の門（9a の encoder の帯を当てる）が、9a の golden とずれた参照に対して導かれる。
    golden の無い機では観測として `missing <path>` を返す（golden は別の台本
    `wan.export_vae_encoder` が書く — 無くても参照は作れる）。
    """
    if root is None:
        return "not-checked"
    width, height = denoised.image.cropped.size
    golden = export_vae_encoder.GoldenCase(
        denoised.case.image, width, height, FIT, denoised.case.role
    )
    path = export_vae_encoder.golden_path(root, golden)
    if not path.is_file():
        return f"missing {path}"
    with safe_open(str(path), framework="pt") as handle:
        expected = {key: handle.get_tensor(key) for key in _ENCODER_GOLDEN_KEYS}
    mismatched = {
        key: _golden_difference(expected[key], read(denoised))
        for key, read in _ENCODER_GOLDEN_KEYS.items()
        if not torch.equal(expected[key], read(denoised))
    }
    if mismatched:
        raise AssertionError(
            f"{denoised.case.name}: 段 9a の encoder の golden {path} とビット一致しない"
            f"（{mismatched}）— 参照を書かずに止まる"
        )
    return {"bit_exact": True, "keys": sorted(_ENCODER_GOLDEN_KEYS), "path": str(path)}


def _golden_difference(expected: torch.Tensor, got: torch.Tensor) -> str:
    """golden との食い違いの要約（形・dtype が違えばそれを、同じなら最大絶対差を言う）。"""
    if expected.shape != got.shape or expected.dtype != got.dtype:
        return (
            f"golden {tuple(expected.shape)} {expected.dtype} / 参照 {tuple(got.shape)} {got.dtype}"
        )
    return f"max_abs {float((expected.double() - got.double()).abs().max())}"


def first_frame_difference(frames: torch.Tensor, image: ConditionImage) -> dict[str, float]:
    """出力の先頭フレームと条件画像（[-1, 1]）の差（観測 — VAE の往復の誤差の大きさ）。"""
    difference = (frames[:, 0] - image.signed[0]).abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}


# ---- 書き出し --------------------------------------------------------------------------


def case_tensors(
    denoised: Denoised, frames: torch.Tensor, vae: AutoencoderKLWan
) -> dict[str, torch.Tensor]:
    """書き出すテンソル（画像の 2 本は u8・他は f32・非有限があれば fail loudly）。"""
    tensors: dict[str, torch.Tensor] = {
        "latents_init": denoised.latents_init,
        "condition_latents": denoised.condition,
    }
    for index in range(STEPS):
        tensors[f"noise_cond.{index}"] = denoised.noise_cond[index]
        tensors[f"noise_uncond.{index}"] = denoised.noise_uncond[index]
        tensors[f"latents.{index}"] = denoised.latents[index]
    tensors["latents_final"] = denoised.latents_final
    if denoised.observed is not None and denoised.observed_final is not None:
        for index, latents in enumerate(denoised.observed):
            tensors[f"observed_m_s.latents.{index}"] = latents
        tensors["observed_m_s.latents_final"] = denoised.observed_final
    tensors["frames"] = frames
    tensors["latents_mean"] = torch.tensor(vae.config.latents_mean, dtype=torch.float32)
    tensors["latents_std"] = torch.tensor(vae.config.latents_std, dtype=torch.float32)
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise AssertionError(f"{denoised.case.name}: {name} が f32 でない（{tensor.dtype}）")
        if not bool(tensor.isfinite().all()):
            raise AssertionError(f"{denoised.case.name}: {name} に非有限値がある")
    tensors["source"] = denoised.image.source_rgb8
    tensors["rgb8"] = denoised.image.rgb8
    return tensors


def case_metadata(
    denoised: Denoised,
    plan: vae_tiling.TilePlan,
    patch_size: int,
    flow_shift: float,
    common: Mapping[str, str],
) -> dict[str, str]:
    """書き出すメタ（safetensors のメタは文字列だけ）。"""
    case = denoised.case
    image = denoised.image
    width, height = image.cropped.size
    metadata = {
        "role": case.role,
        "seed": str(case.seed),
        "prompt": denoised.names.positive,
        "negative": denoised.names.negative,
        "steps": str(STEPS),
        "guidance_scale": str(GUIDANCE_SCALE),
        "flow_shift": str(flow_shift),
        "expand_timesteps": "true",
        "timestep_mlp": TIMESTEP_MLP,
        "condition_tokens": str(denoised.condition_tokens),
        "condition_timestep": str(CONDITION_TIMESTEP),
        "timesteps": json.dumps(denoised.timesteps),
        "sigmas": json.dumps(denoised.sigmas),
        "weights": WEIGHTS,
        "reference": REFERENCE,
        "image": image.source.name,
        "image_sha256": image.source.sha256,
        "source_rgb8_sha256": _bytes_sha256(image.source_rgb8),
        "rgb8_sha256": _bytes_sha256(image.rgb8),
        "width": str(width),
        "height": str(height),
        "fit": FIT,
        "resample": RESAMPLE,
        **plan.meta(),
        "patch_size": str(patch_size),
        **common,
    }
    if denoised.observed is not None:
        metadata["observed_m_s"] = OBSERVED_M_S
    return metadata


def common_metadata(embeds_sha256: str, model: str = MODEL) -> dict[str, str]:
    """全ケースに共通のメタ（T2V の参照の欄 + 値を決めるライブラリの版 — LANCZOS は Pillow の版で
    決まる）。"""
    versions = export_vae_encoder.library_versions()
    return {
        **ti2v_few_step_ref.common_metadata(embeds_sha256, model),
        **{f"version_{name}": value for name, value in versions.items()},
    }


def generate(
    cases: Sequence[FixtureCase],
    *,
    load_transformer: Callable[[], nn.Module],
    snapshot: Path,
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    embeds: Mapping[str, torch.Tensor],
    names: Mapping[str, ti2v_few_step_ref.PromptNames],
    sources: Mapping[str, SourceImage],
    tile: int,
    out_root: Path,
    common: Mapping[str, str],
    encoder_golden_root: Path | None = None,
) -> list[dict[str, Any]]:
    """2 相で全ケースを回して書く（staging → 置換）。要約を返す。

    MUST: 第 2 相は DiT を手放してから（`ti2v_few_step_ref.assert_released`）— 第 1 相の山に VAE の
    decode を重ねない。
    """
    patch_size = vae.config.patch_size
    if patch_size is None:
        raise ValueError("VAE の patch_size が None — Wan2.2（patchify する世代）の VAE を渡す")
    denoised_cases, released = denoise_phase(
        load_transformer, snapshot, vae, scheduler, cases, embeds, names, sources
    )
    ti2v_few_step_ref.assert_released(released)
    # 9a の golden との照合は第 2 相（1 ケース約 8 分の decode）の前に全ケースぶん — 外れたら
    # 1 本も書かない。
    goldens = {
        denoised.case.name: check_encoder_golden(denoised, encoder_golden_root)
        for denoised in denoised_cases
    }
    decoder = ti2v_few_step_ref.build_pipeline(None, vae, scheduler, expand_timesteps=False)
    flow_shift = float(scheduler.config.flow_shift)
    summaries: list[dict[str, Any]] = []
    for denoised in denoised_cases:
        case = denoised.case
        official = replay(denoised, scheduler.config)
        final = denoised.latents_final[None]
        plan = vae_tiling.plan_tiles(
            final.shape[-2],
            final.shape[-1],
            tile,
            vae_tiling.min_overlap_latent(export_vae.SPATIAL_SCALE, patch_size),
        )
        started = time.perf_counter()
        frames = ti2v_few_step_ref.decode_frames(decoder, final, plan, patch_size)
        denoised.seconds["decode"] = round(time.perf_counter() - started, 1)
        expected = (3, *ti2v_few_step_ref.video_size(decoder, final.shape[1:]))
        if tuple(frames.shape) != expected:
            raise AssertionError(
                f"{case.name}: frames の形 {tuple(frames.shape)} が {expected} でない"
            )
        tensors = case_tensors(denoised, frames, vae)
        metadata = case_metadata(denoised, plan, patch_size, flow_shift, common)
        path = fixture_path(out_root, case)
        staging = path.with_name(path.name + ".staging")
        out_root.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: tensor.contiguous() for name, tensor in tensors.items()},
            str(staging),
            metadata=metadata,
        )
        staging.replace(path)
        summary: dict[str, Any] = {
            "case": case.name,
            "role": case.role,
            "seed": case.seed,
            "prompt": case.prompt,
            "image": case.image,
            "timesteps": denoised.timesteps,
            "sigmas": denoised.sigmas,
            "condition_abs_max": float(denoised.condition.abs().max()),
            "latents_abs_max": float(denoised.latents_final.abs().max()),
            "frames": list(frames.shape),
            "tiles": plan.tiles,
            "official_form": official,
            "first_frame_vs_image": first_frame_difference(frames, denoised.image),
            "condition_vs_encoder_golden": goldens[case.name],
            "seconds": denoised.seconds,
            "path": str(path),
            "bytes": path.stat().st_size,
            "peak_rss_gib": round(ti2v_few_step_ref._peak_rss_gib(), 2),
        }
        if denoised.observed is not None and denoised.observed_final is not None:
            summary["observed_m_s_vs_m_1"] = {
                "condition_equal": denoised.observed_condition_equal,
                "latents": [
                    ti2v_few_step_ref._difference(observed, latents)
                    for observed, latents in zip(denoised.observed, denoised.latents, strict=True)
                ],
                "latents_final": ti2v_few_step_ref._difference(
                    denoised.observed_final, denoised.latents_final
                ),
            }
        print(f"[write] {json.dumps(summary, ensure_ascii=False)}", flush=True)
        summaries.append(summary)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out", type=Path, default=SERIES, help="系列の根（既定は i8 の DiT の系列）"
    )
    parser.add_argument(
        "--embeds",
        type=Path,
        default=SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME,
        help="テキスト埋め込み資産（wan.text_embeds の出力）",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in FIXTURE_CASES],
        default=None,
        help="書くケース（既定は全部）",
    )
    args = parser.parse_args(argv)
    if not args.embeds.is_file():
        parser.error(f"{args.embeds} が無い — 先に `python -m wan.text_embeds` で作る")

    embeds, asset_metadata = text_embeds.read_asset(args.embeds)
    selected = [case for case in FIXTURE_CASES if args.case is None or case.name in args.case]
    # 重い読み込みの前に、資産の行と役割・条件画像の来歴を全ケースぶん確かめる。
    names = {case.name: ti2v_few_step_ref.prompt_names(asset_metadata, case) for case in selected}
    sources = load_sources(selected)
    embeds_sha256 = hashlib.sha256(args.embeds.read_bytes()).hexdigest()

    estimate = memory_estimate(MODEL)
    dit_probe.require_available("load", estimate)
    vae = ti2v_few_step_ref.load_rounded_vae(MODEL)
    snapshot = local_snapshot(MODEL)
    scheduler = ti2v_few_step_ref.load_scheduler(snapshot)
    summary = {
        "series": str(args.out),
        "memory_estimate_gib": round(estimate / 2**30, 2),
        "fixtures": generate(
            selected,
            load_transformer=lambda: ti2v_few_step_ref.load_quantized_transformer(MODEL),
            snapshot=snapshot,
            vae=vae,
            scheduler=scheduler,
            embeds=embeds,
            names=names,
            sources=sources,
            tile=ti2v_few_step_ref.VAE_SERIES.tile,
            out_root=args.out,
            common=common_metadata(embeds_sha256),
            encoder_golden_root=SERIES_ROOT / export_vae.TI2V_SERIES_NAME,
        ),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
