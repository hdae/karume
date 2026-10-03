"""Wan2.1 の少ステップの通しの参照（ADR 0118 決定 5 / 8・検収 段 6）。

832×480・33 フレーム（潜在 `[16,9,60,104]`）を 2 ステップ（CFG あり・guide 5.0・shift 3.0）回した
潜在（VAE の前）と、それをタイル decode したフレーム（クランプ前）を、diffusers の `WanPipeline` を
CPU f32 で素のまま回して作る（`wan/pipeline_ref.py` の上に乗る）。GPU の通し
（`packages/models/src/wan/` の結線）はこの値と帯で照合する。

## 入力

- テキスト文脈: 埋め込み資産（`wan.text_embeds` — `outputs/series/wan2.1-t2v-1.3b-text-embeds/`）の
  正 1 本 + negative。資産の sha256 をメタに書く（埋め込みを作り直したら参照も作り直す）。
- 初期ノイズ: torch CPU の `randn`（seed 固定・`[1,16,9,60,104]`）を上流の `latents=` に注入する
  （決定 5 — 製品のホスト生成器とは別の列。TS はこの値を fixture から読む）。
- 重み: DiT と VAE を f16 表現可能値へ丸めてから回す（決定 8・ADR 0006）。DiT は RoPE の
  バッファを丸めない経路（`export_dit.round_to_f16`）、VAE は `export_vae.load_vae(round_f16=True)`
  と同じ丸め。

## 出力（系列の根の `pipeline_steps.<case>.safetensors`）

- `latents_init` `[16,9,60,104]`: 注入した初期ノイズ。
- `noise_cond.<i>` / `noise_uncond.<i>`: ステップ `i` の DiT の出力（forward の hook で記録。上流の
  計算には触らない）。cond / uncond の振り分けは呼び出しの順ではなく、hook が受けたテキスト文脈の
  値（正 / negative の埋め込みのどちらと一致するか）で決める — 上流の呼び順が変わっても入れ替わった
  まま書かない。GPU との差を DiT とホストの UniPC / CFG に帰属させる用。
- `latents.<i>`: ステップ `i` の後の潜在（`callback_on_step_end` で記録）。最後が VAE の入力。
- `frames` `[3,33,480,832]`: 最終の潜在を上流と同じ逆正規化（`pipeline_wan.py` の
  `latents / latents_std + latents_mean` の逐語）→ 段 5 のタイル decode の参照
  （`vae_tiling.tiled_decode_unclamped` — タイル 32・12 枚）。クランプ前。

## ケース（帯の決定用と受入れ用を分ける — ADR 0118 決定 8・追記 2026-10-02）

決定用（role `band`）2 本と受入れ（role `accept`）1 本は seed もプロンプトも互いに違う。
MUST: `accept` の結果を見て決定用のケースも帯も変えない。

    uv run --group wan --inexact python -m wan.few_step_ref                               # 全部
    uv run --group wan --inexact python -m wan.few_step_ref --case accept-ferret          # 1 本だけ

所要の目安（6 コアの CPU）: DiT の forward 4 回 + タイル decode で 1 ケース約 25 分。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from _shared.paths import SERIES_ROOT
from karume.quantize import round_weights_to_f16
from wan import dit_patch, export_vae, pipeline_ref, prompts, text_embeds, vae_tiling
from wan.export_dit import round_to_f16
from wan.sources import DEFAULT_MODEL, SOURCES

if TYPE_CHECKING:
    from diffusers import WanPipeline

#: 少ステップの通しのステップ数（決定 8）。
STEPS = 2

#: 潜在 `[C, F, H, W]`（832×480・33 フレーム — 最初の到達目標）。
LATENT_SHAPE = (16, 9, 60, 104)

#: 参照のファイル名（系列の根に置く — `pipeline_steps.<case>.safetensors`）。
FIXTURE_PREFIX = "pipeline_steps."
FIXTURE_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class FixtureCase:
    """少ステップの通し 1 本（初期ノイズの seed と正のプロンプト）。"""

    name: str
    #: `band` = 帯を決めるケース・`accept` = 受け入れを判定するケース。
    role: str
    seed: int
    #: 埋め込み資産の正のプロンプトの名前（{@link wan.prompts.FIXED_PROMPTS}）。
    prompt: str


#: 決定用 2 本（帯を決める）+ 受入れ 1 本（決定 8 の形 — 段 3 の `band` / `accept` と同じ分け方）。
#: MUST: `accept` の結果を見て決定用のケースも帯も変えない。seed 20261031 は前の受入れに使った値で、
#: 帯の決定に影響した可能性があるので使わない（受入れは未見の seed で見る）。
FIXTURE_CASES = (
    FixtureCase("band-boxing-cats", role="band", seed=20261030, prompt="boxing-cats"),
    FixtureCase("band-cat-dog-baking", role="band", seed=20261032, prompt="cat-dog-baking"),
    FixtureCase("accept-ferret", role="accept", seed=20261033, prompt="ferret"),
)


def load_rounded_pipeline(model: str = DEFAULT_MODEL) -> WanPipeline:
    """上流の `WanPipeline`（umT5 無し・CPU f32）の DiT と VAE を f16 表現可能値へ丸める。"""
    pipeline = pipeline_ref.load_pipeline(model)
    transformer = pipeline.transformer.eval()
    rounded = round_to_f16(transformer, dit_patch.WanDitTokens(transformer))
    print(f"[fake-quant] transformer: {rounded}", flush=True)
    print(f"[fake-quant] vae: {round_weights_to_f16(pipeline.vae.eval()).describe()}", flush=True)
    return pipeline


# Third-party code notice. `denormalize_latents` below is adapted from `WanPipeline.__call__` in
# huggingface/diffusers (`src/diffusers/pipelines/wan/pipeline_wan.py`, `diffusers==0.39.0`):
# the de-normalization before `vae.decode` is verbatim.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def denormalize_latents(pipeline: WanPipeline, latents: torch.Tensor) -> torch.Tensor:
    """上流の decode の直前の逆正規化（`latents / (1 / std) + mean` — 演算の順も上流のまま）。"""
    latents = latents.to(pipeline.vae.dtype)
    latents_mean = (
        torch.tensor(pipeline.vae.config.latents_mean)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
        1, pipeline.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    return latents / latents_std + latents_mean


def _peak_rss_gib() -> float:
    """このプロセスの RSS の最大（GiB — Linux の `ru_maxrss` は KiB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def run_case(
    pipeline: WanPipeline,
    case: FixtureCase,
    embeds: dict[str, torch.Tensor],
    *,
    tile: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """1 ケースを回し、テンソルと要約を返す。"""
    prompt = text_embeds.prompt_by_name(case.prompt)
    if prompt.role != prompts.POSITIVE:
        raise ValueError(f"{case.name}: {case.prompt} は正のプロンプトでない")
    generator = torch.Generator().manual_seed(case.seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)

    # 上流へ渡す文脈と同じ値（`pipeline_ref.run` が同じ関数で埋める）— hook はこれと値で
    # 突き合わせる。
    contexts = {
        "cond": pipeline_ref.pad_text_embeds(embeds[case.prompt]),
        "uncond": pipeline_ref.pad_text_embeds(embeds["negative"]),
    }
    forwards: list[tuple[int, str, torch.Tensor]] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], output: Any) -> None:
        # 上流は `return_dict=False` でタプルを返す。timestep は `t.expand(1)`（int64）。
        context = kwargs["encoder_hidden_states"]
        labels = [label for label, value in contexts.items() if torch.equal(context, value)]
        if len(labels) != 1:
            raise AssertionError(
                f"{case.name}: DiT の文脈が cond / uncond の埋め込みの {len(labels)} 本と一致する"
                "（1 本のはず）— 振り分けを決められない"
            )
        forwards.append((int(kwargs["timestep"][0]), labels[0], output[0].detach().clone()))

    steps: list[torch.Tensor] = []

    def record_step(_pipe: Any, _index: int, _t: Any, callback_kwargs: dict[str, Any]) -> dict:
        steps.append(callback_kwargs["latents"].detach().clone())
        return {}

    handle = pipeline.transformer.register_forward_hook(record_forward, with_kwargs=True)
    started = time.perf_counter()
    try:
        # attention は `export_dit` の参照と同じく CPU の flash 経路に固定する — 既定の選択が
        # torch の更新で MATH へ変わったとき、2 種の参照の前提が黙って割れないように（落ちる形で
        # 止める）。
        with torch.no_grad(), dit_patch.flash_attention_only():
            final = pipeline_ref.run(
                pipeline,
                prompt_embeds=embeds[case.prompt],
                negative_prompt_embeds=embeds["negative"],
                latents=latents_init.clone(),
                num_inference_steps=STEPS,
                output_type="latent",
                callback_on_step_end=record_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
    finally:
        handle.remove()
    denoise_seconds = time.perf_counter() - started
    timesteps = [int(value) for value in pipeline.scheduler.timesteps]
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

    plan = vae_tiling.plan_tiles(LATENT_SHAPE[2], LATENT_SHAPE[3], tile)
    started = time.perf_counter()
    with torch.no_grad():
        frames = vae_tiling.tiled_decode_unclamped(
            pipeline.vae, denormalize_latents(pipeline, final), plan
        )[0]
    decode_seconds = time.perf_counter() - started

    tensors: dict[str, torch.Tensor] = {"latents_init": latents_init[0]}
    for index in range(STEPS):
        tensors[f"noise_cond.{index}"] = outputs[(index, "cond")][0]
        tensors[f"noise_uncond.{index}"] = outputs[(index, "uncond")][0]
        tensors[f"latents.{index}"] = steps[index][0]
    tensors["frames"] = frames
    for name, tensor in tensors.items():
        if not bool(tensor.isfinite().all()):
            raise AssertionError(f"{case.name}: {name} に非有限値がある")
    summary = {
        "case": case.name,
        "role": case.role,
        "seed": case.seed,
        "prompt": case.prompt,
        "timesteps": timesteps,
        "sigmas": [float(value) for value in pipeline.scheduler.sigmas],
        "latents_abs_max": float(final.abs().max()),
        "frames": list(frames.shape),
        "frames_abs_max": float(frames.abs().max()),
        "denoise_seconds": round(denoise_seconds, 1),
        "decode_seconds": round(decode_seconds, 1),
    }
    return tensors, summary


def write_case(
    pipeline: WanPipeline,
    case: FixtureCase,
    embeds: dict[str, torch.Tensor],
    embeds_sha256: str,
    *,
    tile: int,
    model: str,
    out_root: Path,
) -> dict[str, Any]:
    """1 ケースを回して書く（staging → 置換）。要約を返す。"""
    tensors, summary = run_case(pipeline, case, embeds, tile=tile)
    plan = vae_tiling.plan_tiles(LATENT_SHAPE[2], LATENT_SHAPE[3], tile)
    source = SOURCES[model]
    metadata = {
        "role": case.role,
        "seed": str(case.seed),
        "prompt": case.prompt,
        "negative": "negative",
        "text_embeds_sha256": embeds_sha256,
        "steps": str(STEPS),
        "guidance_scale": str(pipeline_ref.GUIDANCE_SCALE),
        "flow_shift": str(pipeline.scheduler.config.flow_shift),
        "timesteps": json.dumps(summary["timesteps"]),
        "sigmas": json.dumps(summary["sigmas"]),
        "weights": "f16-rounded",
        "reference": (
            "diffusers WanPipeline CPU f32 (umT5 embeddings injected), then the snapped-tile"
            " decode with upstream blend_v / blend_h before clamp"
        ),
        "source": f"{source.repo}@{source.revision}",
        **plan.meta(),
    }
    path = out_root / f"{FIXTURE_PREFIX}{case.name}{FIXTURE_SUFFIX}"
    staging = path.with_name(path.name + ".staging")
    out_root.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        str(staging),
        metadata=metadata,
    )
    staging.replace(path)
    summary["path"] = str(path)
    summary["bytes"] = path.stat().st_size
    summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--tile", type=int, default=export_vae.DEFAULT_TILE)
    parser.add_argument("--out", type=Path, default=SERIES_ROOT / export_vae.SERIES_NAME)
    parser.add_argument(
        "--embeds",
        type=Path,
        default=SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME,
        help="テキスト埋め込み資産（wan.text_embeds の出力）",
    )
    names = [case.name for case in FIXTURE_CASES]
    parser.add_argument(
        "--case", action="append", choices=names, default=None, help="書くケース（既定は全部）"
    )
    args = parser.parse_args(argv)
    if not args.embeds.is_file():
        parser.error(f"{args.embeds} が無い — 先に `python -m wan.text_embeds` で作る")

    embeds, _ = text_embeds.read_asset(args.embeds)
    embeds_sha256 = hashlib.sha256(args.embeds.read_bytes()).hexdigest()
    pipeline = load_rounded_pipeline(args.model)
    if pipeline.vae.use_tiling:
        # MUST: 上流のタイル化は走査形が違う（vae_tiling のモジュール doc）。
        raise SystemExit("vae.use_tiling が True — 上流のタイル化は使わない")
    selected = [case for case in FIXTURE_CASES if args.case is None or case.name in args.case]
    summary = {
        "series": str(args.out),
        "fixtures": [
            write_case(
                pipeline,
                case,
                embeds,
                embeds_sha256,
                tile=args.tile,
                model=args.model,
                out_root=args.out,
            )
            for case in selected
        ],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
