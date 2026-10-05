"""Wan2.2 TI2V-5B の 2 ステップの通しの CPU 参照（ADR 0121 段 6 — T2V）。

1280×704・17 フレーム（潜在 `[48,5,44,80]`・S = 4,400）を 2 ステップ（CFG あり・guidance 5.0・
shift 5.0）回した潜在（VAE の前）と、それをタイル decode → unpatchify → クランプした RGB の
フレームを、diffusers の `WanPipeline` を CPU f32 で素のまま回して作る。GPU の通し
（`WanTi2vPipeline`）はこの値と帯で照合する。2.1 の `wan/few_step_ref.py` の 2.2 版で、2.1 の台本と
`pipeline_ref.py` は変えない（`pipeline_ref.video_size` が 16 ch と縮尺 8 を書き込んでいる）—
使うのは `pipeline_ref.pad_text_embeds` と `few_step_ref.denormalize_latents` だけ。

## 入力

- テキスト文脈: 埋め込み資産（`wan.text_embeds` — 2.1 と同じバイトの資産・ADR 0121 決定 9）。
  プロンプトは資産のメタの `name` で引き、役割（positive / negative）を確かめる。negative は
  `name` が `negative` の行。資産の sha256 をメタに書く（埋め込みを作り直したら参照も作り直す）。
- 初期ノイズ: torch CPU の `randn`（seed 固定・`[1,48,5,44,80]`）を上流の `latents=` に注入する。
- 重み: DiT は f32 の全量を読んでから per-channel i8 の表現可能値へ丸める（RTN の fake-quant —
  `ti2v_export_dit.eager_full` と同じ手順なので、系列の i8 と同じ値になる）。VAE は
  `export_vae.load_vae(round_f16=True)` の f16 表現可能値。
- scheduler: pin の config を `flow_shift=5.0` を明示して読み、読んだ値を確かめる（ADR 0121
  決定 8 — pin の値も 5.0 だが、参照の前提を config の既定に預けない）。

## 組み方（単一プロセス・2 相）

1. 第 1 相（DiT）: `WanPipeline(expand_timesteps=False)` で全ケースを 2 ステップ回す。時刻は
   1 次元の timestep の経路（時刻の MLP は M = 1・`[B,6,C]` の変調）で、決定 4 の T2V の形と
   同じ。DiT の出力は forward の hook で記録し、cond / uncond の振り分けは hook が受けたテキスト
   文脈の値で決める（`few_step_ref` と同じ — 呼び順に依らない）。
2. 観測（門ではない）: 決定用の 2 本は `expand_timesteps=True`（上流の既定・timestep が `[1,S]`
   の M = S の経路）でも 2 ステップ回し、各ステップの潜在を `observed_m_s.latents.<i>` に書く。
   hook は別のもの（timestep が `[1,S]` で値が 1 つであることを確かめる — `few_step_ref` の
   `int()` は `[S]` を受けない）。
3. 第 2 相（VAE）: DiT を手放し（参照が残っていたら fail loudly）、段 5 の 2.2 の経路
   （`vae_tiling` — patchify 空間でブレンド → 上流の unpatchify → clamp）でフレームを作る。

## 出力（i8 の DiT の系列の根の `pipeline_steps.<case>.safetensors`）

- テンソル: `latents_init`・`noise_cond.<i>`・`noise_uncond.<i>`・`latents.<i>`
  （`[48,5,44,80]`）・`frames [3,17,704,1280]`（unpatchify とクランプの後）・`latents_mean` /
  `latents_std`（f32・48 値）、決定用だけ `observed_m_s.latents.<i>`。
- メタ: role / seed / prompt / negative / `text_embeds_sha256` / steps / `guidance_scale` /
  `flow_shift`（`"5.0"`）/ timesteps / sigmas / weights / reference / source / タイル計画
  （`vae_tiling.TilePlan.meta` + `patch_size`）/ `expand_timesteps`（`"false"`）/ `environment`
  （torch の版・スレッド数・MKL / OpenMP の環境変数 — f32 の最終ビットは MKL の振る舞いに依る・
  `dit_reference.numeric_environment`）、決定用だけ `observed_m_s`（観測の説明）。

## ケース（決定用 2 本 + 受入れ 1 本 — 3 本とも同じ形）

MUST: `accept` の結果を見て決定用のケースも帯も変えない。受入れの形を決定用と変えない（形を
変えると帯の外へ出る理由が混ざる）。

    uv run --group wan --inexact python -m wan.ti2v_few_step_ref                          # 全部
    uv run --group wan --inexact python -m wan.ti2v_few_step_ref --case accept-ferret     # 1 本だけ

## RAM と所要（推測）

- RAM の山は第 1 相で約 24 GiB（{@link memory_estimate} の見込み 23.7 GiB = DiT の f32 の全量
  18.63 + 基礎分 0.6 + 実測と式の差 0.35 + S = 4,400 の活性 1.5 + VAE の f32 2.63）。読み込みの前に
  MemAvailable を見て、見込み + 余白 1 GiB（約 24.7 GiB）に足りなければ止まる。見込みが外れれば
  門を通っても山で足りなくなりうる（門は保証ではない）。第 2 相は約 6 GiB。
- 所要は 3 ケースで約 3 時間（DiT の forward 20 回 × 約 2 分 + タイル decode 28 枚 × 5 chunk
  × 3 ケース）。
- 第 1 相の結果はメモリにだけ持つ。DiT の手放しの検査で止まる・第 2 相の途中で落ちると、
  第 1 相（約 40 分）を回し直す（書き終えたケースのファイルは残る）。
- 空きが足りないときの戻り道は層逐次の書き手（`dit_reference.LayerwiseDit` — 山約 5 GiB）だが、
  `WanPipeline.transformer` に差し替えられる部品ではない（独自の `forward` を持つ）ので、
  diffusers の呼び方へ合わせる adapter を新しく書く必要がある（ここには無い）。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import sys
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file
from torch import nn

from _shared.paths import SERIES_ROOT
from wan import (
    dit_patch,
    dit_probe,
    dit_reference,
    export_dit,
    export_vae,
    few_step_ref,
    pipeline_ref,
    prompts,
    text_embeds,
    ti2v_export_dit,
    vae_tiling,
)
from wan.sources import SOURCES, count_parameters, local_snapshot

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

#: 上流のモデル（`wan.sources.SOURCES` のキー）。
MODEL = "ti2v-5b"

#: 少ステップの通しのステップ数。
STEPS = 2

#: 潜在 `[C, F, H, W]`（1280×704・17 フレーム — 公式の対応寸法・利用者の裁定 2026-10-05）。
LATENT_SHAPE = (48, 5, 44, 80)

#: UniPC の shift（pin の scheduler の `flow_shift` = 公式の 720p の値 — 明示して読み、確かめる）。
FLOW_SHIFT = 5.0

#: CFG の強さ（2.1 の参照と同じ値）。
GUIDANCE_SCALE = pipeline_ref.GUIDANCE_SCALE

#: 系列の根（i8 の DiT が参照席を特徴づけるので、i8 の DiT の系列に置く）。
SERIES = ti2v_export_dit.SERIES

#: VAE の系列（タイルと patchify の倍率の正本）。
VAE_SERIES = export_vae.VAE_SERIES[MODEL]

#: メタの `weights`（DiT は i8 の RTN の fake-quant・VAE は f16 表現可能値）。
WEIGHTS = "dit-i8-rtn-fake-quant+vae-f16-rounded"

#: メタの `reference`（参照の素性）。
REFERENCE = (
    "diffusers WanPipeline(expand_timesteps=False) CPU f32 (umT5 embeddings injected), then the"
    " snapped-tile decode with upstream blend_v / blend_h in patchify space, upstream unpatchify"
    " and clamp"
)

#: 決定用のメタの `observed_m_s`（観測の素性 — 門ではない）。
OBSERVED_M_S = (
    "latents after each step with expand_timesteps=True (timestep [1,S], the upstream default);"
    " observation only, not a gate"
)


@dataclass(frozen=True)
class FixtureCase:
    """少ステップの通し 1 本（初期ノイズの seed と正のプロンプト）。"""

    name: str
    #: `band` = 帯を決めるケース・`accept` = 受け入れを判定するケース。
    role: str
    seed: int
    #: 埋め込み資産の正のプロンプトの名前（資産のメタの `name`）。
    prompt: str

    @property
    def observes_m_s(self) -> bool:
        """M = S の経路の観測も採るか（決定用だけ）。"""
        return self.role == "band"


#: 決定用 2 本（帯を決める）+ 受入れ 1 本。seed は 2.1 の参照（20261030〜33）とも、DiT の golden
#: （20262000 台の前半）とも違う値。
#: MUST: `accept` の結果を見て決定用のケースも帯も変えない。
FIXTURE_CASES = (
    FixtureCase("band-boxing-cats", role="band", seed=20262101, prompt="boxing-cats"),
    FixtureCase("band-cat-dog-baking", role="band", seed=20262102, prompt="cat-dog-baking"),
    FixtureCase("accept-ferret", role="accept", seed=20262103, prompt="ferret"),
)

#: negative の行の名前（資産のメタの `name`）。
NEGATIVE_NAME = "negative"


@dataclass(frozen=True)
class PromptNames:
    """1 ケースが資産から引く正 / negative の名前。"""

    positive: str
    negative: str


def prompt_names(asset_metadata: Mapping[str, Any], case: FixtureCase) -> PromptNames:
    """資産のメタの `name` で正 / negative の行を引き、役割を確かめる（外れたら fail loudly）。"""
    entries = {entry["name"]: entry for entry in asset_metadata["prompts"]}
    for name, role in ((case.prompt, prompts.POSITIVE), (NEGATIVE_NAME, prompts.NEGATIVE)):
        entry = entries.get(name)
        if entry is None:
            raise ValueError(
                f"{case.name}: 埋め込み資産に {name!r} の行が無い（{sorted(entries)}）"
            )
        if entry["role"] != role:
            raise ValueError(
                f"{case.name}: 資産の {name!r} の役割が {entry['role']!r}（{role} のはず）"
            )
    return PromptNames(positive=case.prompt, negative=NEGATIVE_NAME)


def video_size(pipeline: WanPipeline, latent_shape: Sequence[int]) -> tuple[int, int, int]:
    """潜在 `[C, F, H, W]` → 上流の `(num_frames, height, width)`。

    縮尺は上流が vae の config から取った値（`vae_scale_factor_*` — 2.2 は 4 / 16）。上流の
    `prepare_latents` は注入した潜在の形を見ないので、チャネルは DiT と VAE の両方と突き合わせる。
    """
    channels, frames, height, width = latent_shape
    if channels != pipeline.vae.config.z_dim:
        raise ValueError(
            f"潜在のチャネル {channels} が VAE の z_dim {pipeline.vae.config.z_dim} と違う"
        )
    if pipeline.transformer is not None and channels != pipeline.transformer.config.in_channels:
        raise ValueError(
            f"潜在のチャネル {channels} が DiT の in_channels"
            f" {pipeline.transformer.config.in_channels} と違う"
        )
    temporal = pipeline.vae_scale_factor_temporal
    spatial = pipeline.vae_scale_factor_spatial
    return temporal * (frames - 1) + 1, spatial * height, spatial * width


def token_count(transformer: nn.Module, latent_shape: Sequence[int]) -> int:
    """DiT のトークン数 S = F/pt · H/ph · W/pw（M = S の経路の timestep の長さ）。"""
    patch_t, patch_h, patch_w = transformer.config.patch_size
    _, frames, height, width = latent_shape
    return (frames // patch_t) * (height // patch_h) * (width // patch_w)


def load_quantized_transformer(model: str = MODEL) -> nn.Module:
    """DiT を f32 の全量で読み、i8 の表現可能値へ丸める（`ti2v_export_dit.eager_full` と同じ
    手順）。"""
    upstream = export_dit.load_transformer(model)
    report = export_dit.fake_quant_i8(upstream, dit_patch.WanDitTokensTi2v(upstream))
    print(f"[fake-quant] transformer: {report.describe()}", flush=True)
    return upstream


def load_rounded_vae(model: str = MODEL) -> AutoencoderKLWan:
    """VAE を f16 表現可能値で読み、系列の前提（patch_size・空間の圧縮）と照合する。"""
    vae = export_vae.load_vae(model, round_f16=True)
    export_vae.assert_series_config(vae, VAE_SERIES)
    if vae.use_tiling:
        # MUST: 上流のタイル化は走査形が違う（`vae_tiling` のモジュール doc）。
        raise SystemExit("vae.use_tiling が True — 上流のタイル化は使わない")
    return vae


def load_scheduler(snapshot: Path) -> UniPCMultistepScheduler:
    """pin の scheduler を `flow_shift` を明示して読み、読んだ値を確かめる（決定 8）。"""
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(
        snapshot, subfolder="scheduler", flow_shift=FLOW_SHIFT
    )
    if scheduler.config.flow_shift != FLOW_SHIFT:
        raise AssertionError(
            f"scheduler の flow_shift {scheduler.config.flow_shift} が {FLOW_SHIFT} でない"
        )
    # `flow_shift` が σ に効くのは flow の σ で、動的 shift でないときだけ（動的 shift は
    # `set_timesteps` で `mu` の式に替わり、`flow_shift` を読まない）。
    if (
        scheduler.config.use_dynamic_shifting is not False
        or scheduler.config.use_flow_sigmas is not True
    ):
        raise AssertionError(
            "scheduler の use_dynamic_shifting"
            f" {scheduler.config.use_dynamic_shifting}・use_flow_sigmas"
            f" {scheduler.config.use_flow_sigmas} — flow_shift の式が効く形（False / True）でない"
        )
    return scheduler


def build_pipeline(
    transformer: nn.Module | None,
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    *,
    expand_timesteps: bool,
) -> WanPipeline:
    """umT5 無しの `WanPipeline` を組む（vae は外さない — 外すと空間の縮尺が既定の 8 に落ちる）。"""
    pipeline_ref.assert_no_mps()
    from diffusers import WanPipeline

    pipeline = WanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
        expand_timesteps=expand_timesteps,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


#: f32 の全量 + fake-quant の実測の山が式（f32 の全量 + 基礎分）を上回った分（ADR 0121 の追記
#: 「f32 の全量での eager 同値」の eager-full — 山 19.57 GiB・式 18.63 + 0.6 = 19.23 GiB）。段 0 の
#: 20.89 GiB は層逐次の f64（1.89 GiB）を含むので、ここの比較の相手ではない。
MEASURED_EXCESS_BYTES = int(0.35 * 2**30)

#: S = 4,400 の 1 forward の活性の山の見込み（推測 — 実測なし）。M = S の観測が最大で、
#: `[1,S,6,3072]` の変調（全体と各ブロックの和の 2 本 × 0.30 GiB）+ FFN の中間と GELU の出力
#: （2 本 × `[1,S,14336]` 0.23 GiB）+ `[1,S,3072]` 6 本ほど（0.30 GiB）≈ 1.37 GiB を切り上げた値。
ACTIVATION_BYTES = int(1.5 * 2**30)


def memory_estimate(model: str = MODEL) -> int:
    """第 1 相の RAM の見込み（バイト）= DiT の f32 の全量 + 基礎分 + 実測と式の差 + 活性 +
    VAE の f32（encoder を含む）。5B で約 23.7 GiB（`require_available` が余白 1 GiB を足す）。
    """
    config = dit_reference.load_config(dit_probe.transformer_dir(model))
    vae_bytes = count_parameters(local_snapshot(model) / "vae") * 4
    return (
        dit_probe.eager_memory(config)["f32"]
        + dit_probe.EAGER_BASE_BYTES
        + MEASURED_EXCESS_BYTES
        + ACTIVATION_BYTES
        + vae_bytes
    )


def _peak_rss_gib() -> float:
    """このプロセスの RSS の最大（GiB — Linux の `ru_maxrss` は KiB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


@dataclass
class Denoised:
    """第 1 相の 1 ケースぶん（第 2 相と書き出しへ渡す）。"""

    case: FixtureCase
    names: PromptNames
    latents_init: torch.Tensor
    noise_cond: list[torch.Tensor]
    noise_uncond: list[torch.Tensor]
    latents: list[torch.Tensor]
    #: M = S の経路の各ステップの潜在（決定用だけ — 受入れは None）。
    observed: list[torch.Tensor] | None
    timesteps: list[int]
    sigmas: list[float]
    seconds: dict[str, float]


def _call(
    pipeline: WanPipeline,
    embeds: Mapping[str, torch.Tensor],
    names: PromptNames,
    latents_init: torch.Tensor,
    record_step: Callable[..., dict[str, Any]],
) -> torch.Tensor:
    num_frames, height, width = video_size(pipeline, latents_init.shape[1:])
    output = pipeline(
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


def _step_recorder(steps: list[torch.Tensor]) -> Callable[..., dict[str, Any]]:
    def record_step(_pipe: Any, _index: int, _t: Any, callback_kwargs: dict[str, Any]) -> dict:
        steps.append(callback_kwargs["latents"].detach().clone())
        return {}

    return record_step


def denoise_case(
    pipeline: WanPipeline,
    case: FixtureCase,
    embeds: Mapping[str, torch.Tensor],
    names: PromptNames,
) -> Denoised:
    """M = 1 の経路（`expand_timesteps=False`）で 1 ケースを 2 ステップ回す。"""
    generator = torch.Generator().manual_seed(case.seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)

    # 上流へ渡す文脈と同じ値（`_call` が同じ関数で埋める）— hook はこれと値で突き合わせる。
    contexts = {
        "cond": pipeline_ref.pad_text_embeds(embeds[names.positive]),
        "uncond": pipeline_ref.pad_text_embeds(embeds[names.negative]),
    }
    forwards: list[tuple[int, str, torch.Tensor]] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], output: Any) -> None:
        # 上流は `return_dict=False` でタプルを返す。timestep は `t.expand(1)`（1 次元の int64）。
        timestep = kwargs["timestep"]
        if timestep.dim() != 1:
            raise AssertionError(
                f"{case.name}: timestep の形 {tuple(timestep.shape)} が 1 次元でない"
                "（M = 1 の経路のはず）"
            )
        context = kwargs["encoder_hidden_states"]
        labels = [label for label, value in contexts.items() if torch.equal(context, value)]
        if len(labels) != 1:
            raise AssertionError(
                f"{case.name}: DiT の文脈が cond / uncond の埋め込みの {len(labels)} 本と一致する"
                "（1 本のはず）— 振り分けを決められない"
            )
        forwards.append((int(timestep[0]), labels[0], output[0].detach().clone()))

    steps: list[torch.Tensor] = []
    handle = pipeline.transformer.register_forward_hook(record_forward, with_kwargs=True)
    started = time.perf_counter()
    try:
        # attention は `export_dit` の参照と同じく CPU の flash 経路に固定する（MATH へ黙って
        # 落ちる経路を残さない — `few_step_ref` と同じ）。
        with torch.no_grad(), dit_patch.flash_attention_only():
            final = _call(pipeline, embeds, names, latents_init, _step_recorder(steps))
    finally:
        handle.remove()
    seconds = time.perf_counter() - started
    timesteps = [int(value) for value in pipeline.scheduler.timesteps]
    sigmas = [float(value) for value in pipeline.scheduler.sigmas]
    if len(forwards) != 2 * STEPS or len(steps) != STEPS:
        raise AssertionError(
            f"forward {len(forwards)} 回・step {len(steps)} 回（CFG あり 2 ステップ）"
        )
    if [t for t, _, _ in forwards] != [t for t in timesteps for _ in range(2)]:
        raise AssertionError(
            f"forward の timestep {[t for t, _, _ in forwards]} が {timesteps} と合わない"
        )
    outputs: dict[tuple[int, str], torch.Tensor] = {}
    for index, (_, label, output) in enumerate(forwards):
        outputs[(index // 2, label)] = output
    if sorted(outputs) != [(step, label) for step in range(STEPS) for label in sorted(contexts)]:
        raise AssertionError(
            "ステップごとの cond / uncond が 1 本ずつでない"
            f"（{[label for _, label, _ in forwards]}）"
        )
    if not torch.equal(final, steps[-1]):
        raise AssertionError("最終の潜在が最後の step の記録と違う")
    return Denoised(
        case=case,
        names=names,
        latents_init=latents_init[0],
        noise_cond=[outputs[(index, "cond")][0] for index in range(STEPS)],
        noise_uncond=[outputs[(index, "uncond")][0] for index in range(STEPS)],
        latents=[step[0] for step in steps],
        observed=None,
        timesteps=timesteps,
        sigmas=sigmas,
        seconds={"denoise": round(seconds, 1)},
    )


def observe_m_s(
    pipeline: WanPipeline, denoised: Denoised, embeds: Mapping[str, torch.Tensor]
) -> list[torch.Tensor]:
    """M = S の経路（`expand_timesteps=True`）で同じ初期ノイズを 2 ステップ回し、各ステップの潜在を
    返す（観測 — 門ではない）。

    hook は M = 1 とは別のもの: 上流はここで timestep を `[1,S]`（全トークン同じ値）で渡すので、
    形と値が 1 つであることを確かめ、M = 1 と同じ時刻の列かを見る。
    """
    case = denoised.case
    tokens = token_count(pipeline.transformer, LATENT_SHAPE)
    seen: list[int] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], _output: Any) -> None:
        timestep = kwargs["timestep"]
        if tuple(timestep.shape) != (1, tokens):
            raise AssertionError(
                f"{case.name}: M = S の経路の timestep の形 {tuple(timestep.shape)} が"
                f" (1, {tokens}) でない"
            )
        values = timestep.unique()
        if values.numel() != 1:
            raise AssertionError(
                f"{case.name}: M = S の timestep が {values.numel()} 値（T2V は 1 値）"
            )
        seen.append(int(values[0]))

    steps: list[torch.Tensor] = []
    handle = pipeline.transformer.register_forward_hook(record_forward, with_kwargs=True)
    started = time.perf_counter()
    try:
        with torch.no_grad(), dit_patch.flash_attention_only():
            final = _call(
                pipeline, embeds, denoised.names, denoised.latents_init[None], _step_recorder(steps)
            )
    finally:
        handle.remove()
    denoised.seconds["observe_m_s"] = round(time.perf_counter() - started, 1)
    if seen != [t for t in denoised.timesteps for _ in range(2)] or len(steps) != STEPS:
        raise AssertionError(
            f"{case.name}: M = S の forward の timestep {seen}・step {len(steps)} 回が"
            f" M = 1 の {denoised.timesteps} と合わない"
        )
    if not torch.equal(final, steps[-1]):
        raise AssertionError(f"{case.name}: M = S の最終の潜在が最後の step の記録と違う")
    return [step[0] for step in steps]


def denoise_phase(
    load_transformer: Callable[[], nn.Module],
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    cases: Sequence[FixtureCase],
    embeds: Mapping[str, torch.Tensor],
    names: Mapping[str, PromptNames],
) -> tuple[list[Denoised], weakref.ref[nn.Module]]:
    """第 1 相: DiT を読み、全ケースを回す。DiT への参照はこの関数の中だけに置く（戻ると手放す）。

    戻りの弱参照は、第 2 相の前に DiT が回収されたことを確かめるため（{@link assert_released}）。
    """
    transformer = load_transformer()
    released = weakref.ref(transformer)
    pipeline_m_1 = build_pipeline(transformer, vae, scheduler, expand_timesteps=False)
    pipeline_m_s = build_pipeline(transformer, vae, scheduler, expand_timesteps=True)
    results: list[Denoised] = []
    for case in cases:
        denoised = denoise_case(pipeline_m_1, case, embeds, names[case.name])
        if case.observes_m_s:
            denoised.observed = observe_m_s(pipeline_m_s, denoised, embeds)
        print(f"[denoise] {case.name}: {json.dumps(denoised.seconds)}", flush=True)
        results.append(denoised)
    return results, released


def assert_released(released: weakref.ref[nn.Module]) -> None:
    """DiT が回収されたことを確かめる（第 2 相の RAM の見込みは DiT が居ない前提）。"""
    gc.collect()
    if released() is not None:
        raise AssertionError("DiT への参照が残っている — 第 2 相（VAE）の前に手放す")


def decode_frames(
    pipeline: WanPipeline, latents: torch.Tensor, plan: vae_tiling.TilePlan, patch_size: int
) -> torch.Tensor:
    """最終の潜在 `[1, z, F, H, W]` → RGB `[3, F', H·16, W·16]`（unpatchify とクランプの後）。

    上流と同じ逆正規化 → 段 5 のタイル decode（patchify 空間・クランプ前）→ 上流の unpatchify →
    `clamp(-1, 1)`（上流 `_decode` の末尾と同じ順）。
    """
    with torch.no_grad():
        patchified = vae_tiling.tiled_decode_unclamped(
            pipeline.vae, few_step_ref.denormalize_latents(pipeline, latents), plan
        )
    rgb: torch.Tensor = vae_tiling.unpatchify_frames(patchified, patch_size).clamp(-1.0, 1.0)[0]
    return rgb


def _difference(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    max_abs = float((got - want).abs().max())
    return {"max_abs": max_abs, "ratio": max_abs / float(want.abs().max())}


def case_tensors(
    denoised: Denoised, frames: torch.Tensor, vae: AutoencoderKLWan
) -> dict[str, torch.Tensor]:
    """書き出すテンソル（全て f32・非有限があれば fail loudly）。"""
    tensors: dict[str, torch.Tensor] = {"latents_init": denoised.latents_init}
    for index in range(STEPS):
        tensors[f"noise_cond.{index}"] = denoised.noise_cond[index]
        tensors[f"noise_uncond.{index}"] = denoised.noise_uncond[index]
        tensors[f"latents.{index}"] = denoised.latents[index]
    if denoised.observed is not None:
        for index, latents in enumerate(denoised.observed):
            tensors[f"observed_m_s.latents.{index}"] = latents
    tensors["frames"] = frames
    tensors["latents_mean"] = torch.tensor(vae.config.latents_mean, dtype=torch.float32)
    tensors["latents_std"] = torch.tensor(vae.config.latents_std, dtype=torch.float32)
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise AssertionError(f"{denoised.case.name}: {name} が f32 でない（{tensor.dtype}）")
        if not bool(tensor.isfinite().all()):
            raise AssertionError(f"{denoised.case.name}: {name} に非有限値がある")
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
    metadata = {
        "role": case.role,
        "seed": str(case.seed),
        "prompt": denoised.names.positive,
        "negative": denoised.names.negative,
        "steps": str(STEPS),
        "guidance_scale": str(GUIDANCE_SCALE),
        "flow_shift": str(flow_shift),
        "expand_timesteps": "false",
        "timesteps": json.dumps(denoised.timesteps),
        "sigmas": json.dumps(denoised.sigmas),
        "weights": WEIGHTS,
        "reference": REFERENCE,
        **plan.meta(),
        "patch_size": str(patch_size),
        **common,
    }
    if denoised.observed is not None:
        metadata["observed_m_s"] = OBSERVED_M_S
    return metadata


def common_metadata(embeds_sha256: str, model: str = MODEL) -> dict[str, str]:
    """全ケースに共通のメタ（埋め込みの sha256・出所の pin・f32 の最終ビットが依る実行環境）。"""
    source = SOURCES[model]
    return {
        "text_embeds_sha256": embeds_sha256,
        "source": f"{source.repo}@{source.revision}",
        "environment": json.dumps(dit_reference.numeric_environment(), sort_keys=True),
    }


def generate(
    cases: Sequence[FixtureCase],
    *,
    load_transformer: Callable[[], nn.Module],
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    embeds: Mapping[str, torch.Tensor],
    names: Mapping[str, PromptNames],
    tile: int,
    out_root: Path,
    common: Mapping[str, str],
) -> list[dict[str, Any]]:
    """2 相で全ケースを回して書く（staging → 置換）。要約を返す。

    MUST: 第 2 相は DiT を手放してから（{@link assert_released}）— 第 1 相の山に VAE の decode を
    重ねない。
    """
    patch_size = vae.config.patch_size
    if patch_size is None:
        raise ValueError("VAE の patch_size が None — Wan2.2（patchify する世代）の VAE を渡す")
    denoised_cases, released = denoise_phase(load_transformer, vae, scheduler, cases, embeds, names)
    assert_released(released)
    decoder = build_pipeline(None, vae, scheduler, expand_timesteps=False)
    flow_shift = float(scheduler.config.flow_shift)
    summaries: list[dict[str, Any]] = []
    for denoised in denoised_cases:
        case = denoised.case
        final = denoised.latents[-1][None]
        plan = vae_tiling.plan_tiles(
            final.shape[-2],
            final.shape[-1],
            tile,
            vae_tiling.min_overlap_latent(export_vae.SPATIAL_SCALE, patch_size),
        )
        started = time.perf_counter()
        frames = decode_frames(decoder, final, plan, patch_size)
        denoised.seconds["decode"] = round(time.perf_counter() - started, 1)
        expected = (3, *video_size(decoder, final.shape[1:]))
        if tuple(frames.shape) != expected:
            raise AssertionError(
                f"{case.name}: frames の形 {tuple(frames.shape)} が {expected} でない"
            )
        tensors = case_tensors(denoised, frames, vae)
        metadata = case_metadata(denoised, plan, patch_size, flow_shift, common)
        path = out_root / f"{few_step_ref.FIXTURE_PREFIX}{case.name}{few_step_ref.FIXTURE_SUFFIX}"
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
            "timesteps": denoised.timesteps,
            "sigmas": denoised.sigmas,
            "latents_abs_max": float(final.abs().max()),
            "frames": list(frames.shape),
            "tiles": plan.tiles,
            "seconds": denoised.seconds,
            "path": str(path),
            "bytes": path.stat().st_size,
            "peak_rss_gib": round(_peak_rss_gib(), 2),
        }
        if denoised.observed is not None:
            summary["observed_m_s_vs_m_1"] = [
                _difference(observed, latents)
                for observed, latents in zip(denoised.observed, denoised.latents, strict=True)
            ]
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
    # 重い読み込みの前に、資産の行と役割を全ケースぶん確かめる。
    names = {case.name: prompt_names(asset_metadata, case) for case in selected}
    embeds_sha256 = hashlib.sha256(args.embeds.read_bytes()).hexdigest()

    estimate = memory_estimate(MODEL)
    dit_probe.require_available("load", estimate)
    vae = load_rounded_vae(MODEL)
    scheduler = load_scheduler(local_snapshot(MODEL))
    summary = {
        "series": str(args.out),
        "memory_estimate_gib": round(estimate / 2**30, 2),
        "fixtures": generate(
            selected,
            load_transformer=lambda: load_quantized_transformer(MODEL),
            vae=vae,
            scheduler=scheduler,
            embeds=embeds,
            names=names,
            tile=VAE_SERIES.tile,
            out_root=args.out,
            common=common_metadata(embeds_sha256),
        ),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
