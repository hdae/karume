"""Reference for the few-step end-to-end run of Wan2.1 (ADR 0118 decisions 5 / 8, acceptance of
step 6).

Runs diffusers' `WanPipeline` unmodified on CPU f32 for 832×480, 33 frames (latent
`[16,9,60,104]`), 2 steps (with CFG, guide 5.0, shift 3.0), and records the latent (before the VAE)
and the frames from its tiled decode (before the clamp) (it builds on `wan/pipeline_ref.py`). The
GPU end-to-end run (the wiring in `packages/models/src/wan/`) is checked against these values within
a band.

## Inputs

- Text context: one positive prompt + the negative from the embeddings asset (`wan.text_embeds` —
  `outputs/series/wan2.1-t2v-1.3b-text-embeds/`). The asset's sha256 goes into the metadata (when
  the embeddings are rebuilt, the reference is rebuilt too).
- Initial noise: a torch CPU `randn` (fixed seed, `[1,16,9,60,104]`) injected through upstream's
  `latents=` (decision 5 — a different sequence from the product's host generator; TS reads these
  values from the fixture).
- Weights: the DiT and the VAE are rounded to f16-representable values before the run (decision 8,
  ADR 0006). The DiT takes the path that does not round the RoPE buffers
  (`export_dit.round_to_f16`); the VAE gets the same rounding as
  `export_vae.load_vae(round_f16=True)`.

## Output (`pipeline_steps.<case>.safetensors` at the series root)

- `latents_init` `[16,9,60,104]`: the injected initial noise.
- `noise_cond.<i>` / `noise_uncond.<i>`: the DiT output at step `i` (recorded by a forward hook;
  upstream's computation is not touched). cond and uncond are told apart not by call order but by
  the value of the text context the hook received (whether it equals the positive or the negative
  embeddings) — if upstream's call order changes, they are not written swapped. Used to attribute a
  GPU difference to the DiT versus the host UniPC / CFG.
- `latents.<i>`: the latent after step `i` (recorded with `callback_on_step_end`). The last one is
  the VAE input.
- `frames` `[3,33,480,832]`: the final latent, de-normalized as upstream does (verbatim
  `latents / latents_std + latents_mean` from `pipeline_wan.py`) → the reference of the step-5
  tiled decode (`vae_tiling.tiled_decode_unclamped` — tile 32, 12 tiles). Before the clamp.

## Cases (band cases and acceptance cases are separate — ADR 0118 decision 8, addendum 2026-10-02)

The 2 band cases (role `band`) and the 1 acceptance case (role `accept`) all differ in seed and
prompt.
MUST: do not change the band cases or the band after looking at the `accept` result.

    uv run --group wan --inexact python -m wan.few_step_ref                               # all
    uv run --group wan --inexact python -m wan.few_step_ref --case accept-ferret          # one

Rough cost (6-core CPU): 4 DiT forwards + the tiled decode, about 25 minutes per case.

MUST: import diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
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
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS

if TYPE_CHECKING:
    from diffusers import WanPipeline

#: The step count of the few-step end-to-end run (decision 8).
STEPS = 2

#: The latent `[C, F, H, W]` (832×480, 33 frames — the first target).
LATENT_SHAPE = (16, 9, 60, 104)

#: The reference file name (at the series root — `pipeline_steps.<case>.safetensors`).
FIXTURE_PREFIX = "pipeline_steps."
FIXTURE_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class FixtureCase:
    """One few-step end-to-end run (seed of the initial noise and the positive prompt)."""

    name: str
    #: `band` = a case that sets the band, `accept` = a case that judges acceptance.
    role: str
    seed: int
    #: The name of the positive prompt in the embeddings asset ({@link wan.prompts.FIXED_PROMPTS}).
    prompt: str


#: 2 band cases (set the band) + 1 acceptance case (the shape of decision 8 — the same split as the
#: `band` / `accept` of step 3).
#: MUST: do not change the band cases or the band after looking at the `accept` result. Seed
#: 20261031 was used for an earlier acceptance and may have influenced the band, so it is not used
#: (acceptance is judged on an unseen seed).
FIXTURE_CASES = (
    FixtureCase("band-boxing-cats", role="band", seed=20261030, prompt="boxing-cats"),
    FixtureCase("band-cat-dog-baking", role="band", seed=20261032, prompt="cat-dog-baking"),
    FixtureCase("accept-ferret", role="accept", seed=20261033, prompt="ferret"),
)


def load_rounded_pipeline(model: str = DEFAULT_MODEL) -> WanPipeline:
    """Rounds the DiT and the VAE of upstream `WanPipeline` (no umT5, CPU f32) to f16-representable
    values."""
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
    """The de-normalization right before upstream's decode (`latents / (1 / std) + mean` — the
    order of operations is also upstream's)."""
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
    """The peak RSS of this process (GiB — Linux `ru_maxrss` is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def run_case(
    pipeline: WanPipeline,
    case: FixtureCase,
    embeds: dict[str, torch.Tensor],
    *,
    tile: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Runs one case and returns the tensors and a summary."""
    prompt = text_embeds.prompt_by_name(case.prompt)
    if prompt.role != prompts.POSITIVE:
        raise ValueError(f"{case.name}: {case.prompt} is not a positive prompt")
    generator = torch.Generator().manual_seed(case.seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)

    # The same values as the context passed upstream (`pipeline_ref.run` pads them with the same
    # function) — the hook matches against these by value.
    contexts = {
        "cond": pipeline_ref.pad_text_embeds(embeds[case.prompt]),
        "uncond": pipeline_ref.pad_text_embeds(embeds["negative"]),
    }
    forwards: list[tuple[int, str, torch.Tensor]] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], output: Any) -> None:
        # Upstream returns a tuple with `return_dict=False`. The timestep is `t.expand(1)` (int64).
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
        # Pin attention to the CPU flash path, as the `export_dit` reference does — so that if a
        # torch update switches the default choice to MATH, the premise shared by the two kinds of
        # reference does not break silently (it stops with an error instead).
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
            f"{len(forwards)} forwards, {len(steps)} steps (expected: CFG with 2 steps)"
        )
    if [t for t, _, _ in forwards] != [t for t in timesteps for _ in range(2)]:
        raise AssertionError(
            f"forward timesteps {[t for t, _, _ in forwards]} do not match {timesteps}"
        )
    outputs: dict[tuple[int, str], torch.Tensor] = {}
    for index, (_, label, output) in enumerate(forwards):
        outputs[(index // 2, label)] = output
    if sorted(outputs) != [(step, label) for step in range(STEPS) for label in sorted(contexts)]:
        raise AssertionError(
            f"not exactly one cond and one uncond per step ({[label for _, label, _ in forwards]})"
        )
    if not torch.equal(final, steps[-1]):
        raise AssertionError("the final latent differs from the record of the last step")

    plan = vae_tiling.plan_tiles(
        LATENT_SHAPE[2], LATENT_SHAPE[3], tile, vae_tiling.WAN21_MIN_OVERLAP_LATENT
    )
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
            raise AssertionError(f"{case.name}: {name} has non-finite values")
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
    """Runs and writes one case (staging → replace). Returns a summary."""
    tensors, summary = run_case(pipeline, case, embeds, tile=tile)
    plan = vae_tiling.plan_tiles(
        LATENT_SHAPE[2], LATENT_SHAPE[3], tile, vae_tiling.WAN21_MIN_OVERLAP_LATENT
    )
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
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument("--tile", type=int, default=export_vae.DEFAULT_TILE)
    parser.add_argument("--out", type=Path, default=SERIES_ROOT / export_vae.SERIES_NAME)
    parser.add_argument(
        "--embeds",
        type=Path,
        default=SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME,
        help="the text-embeddings asset (the output of wan.text_embeds)",
    )
    names = [case.name for case in FIXTURE_CASES]
    parser.add_argument(
        "--case", action="append", choices=names, default=None, help="cases to write (default: all)"
    )
    args = parser.parse_args(argv)
    if not args.embeds.is_file():
        parser.error(f"{args.embeds} is missing — build it first with `python -m wan.text_embeds`")

    embeds, _ = text_embeds.read_asset(args.embeds)
    embeds_sha256 = hashlib.sha256(args.embeds.read_bytes()).hexdigest()
    pipeline = load_rounded_pipeline(args.model)
    if pipeline.vae.use_tiling:
        # MUST: upstream's tiling scans in a different order (module doc of vae_tiling).
        raise SystemExit("vae.use_tiling is True — upstream's tiling is not used")
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
