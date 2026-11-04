"""The reference pipeline of Wan2.1 (runs diffusers' `WanPipeline` on CPU f32 — ADR 0118
decisions 4 / 5).

The numerical source of truth is `WanPipeline` of diffusers 0.39.0 (`pipeline_wan.py`). This module
is the shared part that builds and runs it **unmodified**; the scripts that make the reference
outputs of the DiT (steps 2 / 3) and the VAE (steps 4 / 5) build on it. The patch layer is not
applied (the same discipline as `anima/pipeline_ref.py` — if the patches also ran on the reference
side, a patch bug would land identically in the reference and in the code under test and pass with
a difference of 0).

How it is built:

- **CPU f32** (both DiT and VAE with `torch_dtype=torch.float32`). Upstream's bf16 autocast is not
  used (WebGPU has no bf16 — decision 3).
- **`text_encoder=None` / `tokenizer=None`**. umT5-XXL (about 22.7 GB in f32) is not loaded — it
  is not kept next to the DiT in the dev machine's 31 GiB of host RAM (decision 4). The text
  context is passed to `prompt_embeds` / `negative_prompt_embeds` as **valid-length embeddings
  `[L_valid, 4096]`** and is injected here in upstream's shape (`[1, 512, 4096]` zero-padded after
  the valid length — `_get_t5_prompt_embeds`).
- **UniPC uses the scheduler config as is** (`flow_shift` 3.0 — decision 5. If only the side that
  makes the reference rewrote values, the property that `from_pretrained` alone reproduces the
  reference would be lost).
- The initial noise is injected from outside (`latents=` — decision 5; the caller makes it with a
  torch CPU `randn`).

MUST: do not build on a machine where MPS is visible ({@link assert_no_mps}). diffusers'
`WanRotaryPosEmbed` builds the RoPE table in float32 only on machines where MPS is available and in
float64 everywhere else (`transformer_wan.py:375`). It is pinned to the float64 side so that the
reference and the base tables do not depend on the machine (decision 3).

MUST: import diffusers / transformers inside functions (the `wan` group is not in the default sync
— `tests/test_optional_group_imports.py`).

    uv run --group wan python -m wan.pipeline_ref --smoke   # loading plus a 1-step smoke run
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING, Any

import torch

from wan.sources import DEFAULT_MODEL, WAN21_MODELS, local_snapshot

if TYPE_CHECKING:
    from diffusers import WanPipeline

#: The length of upstream's text context (the default `max_sequence_length` of
#: `WanPipeline.__call__`).
MAX_SEQUENCE_LENGTH = 512

#: The hidden size of umT5-XXL (`text_dim` in the transformer config).
TEXT_DIM = 4096

#: The temporal / spatial downsampling factors of the VAE (`scale_factor_temporal` /
#: `scale_factor_spatial` in the vae config).
TEMPORAL_SCALE = 4
SPATIAL_SCALE = 8

#: The settings for making reference outputs (decision 5). shift comes from the scheduler config,
#: so it is not here.
GUIDANCE_SCALE = 5.0
NUM_INFERENCE_STEPS = 50


class MpsVisibleError(RuntimeError):
    """A reference was about to be built on a machine where MPS is visible (the RoPE table would be
    float32 — decision 3)."""


def assert_no_mps() -> None:
    """Checks the condition under which the RoPE table is built in float64 (no MPS)."""
    if torch.backends.mps.is_available():
        raise MpsVisibleError(
            "MPS is visible — diffusers' WanRotaryPosEmbed builds the RoPE table in float32, and"
            " the reference differs from a float64 machine (ADR 0118 decision 3). Run on a"
            " machine without MPS"
        )


def pad_text_embeds(embeds: torch.Tensor) -> torch.Tensor:
    """Valid-length embeddings `[L_valid, 4096]` → upstream's `[1, 512, 4096]` (zero-padded at the
    end).

    Upstream's `_get_t5_prompt_embeds` cuts the umT5 output at the valid length and then pads it to
    512 with `new_zeros`. This builds it in the same order (the cut values → concatenated zeros).
    Anything other than f32 fails loudly (no rounding the reference does not have — decision 4).
    """
    if embeds.dtype != torch.float32:
        raise ValueError(f"pass the embeddings in f32 (got {embeds.dtype})")
    if embeds.dim() != 2 or embeds.shape[1] != TEXT_DIM:
        raise ValueError(
            f"pass the embeddings as [L_valid, {TEXT_DIM}] (got {tuple(embeds.shape)})"
        )
    length = embeds.shape[0]
    if not 0 < length <= MAX_SEQUENCE_LENGTH:
        raise ValueError(f"valid length {length} is outside 1..{MAX_SEQUENCE_LENGTH}")
    padding = embeds.new_zeros(MAX_SEQUENCE_LENGTH - length, TEXT_DIM)
    return torch.cat([embeds, padding]).unsqueeze(0)


def video_size(latents: torch.Tensor) -> tuple[int, int, int]:
    """Latent `[1, 16, F, H, W]` → upstream's `(num_frames, height, width)` (`4(F−1)+1`, `8H`,
    `8W`)."""
    if latents.dim() != 5 or latents.shape[0] != 1 or latents.shape[1] != 16:
        raise ValueError(f"pass the latent as [1, 16, F, H, W] (got {tuple(latents.shape)})")
    frames, height, width = latents.shape[2:]
    return TEMPORAL_SCALE * (frames - 1) + 1, SPATIAL_SCALE * height, SPATIAL_SCALE * width


def load_pipeline(model: str = DEFAULT_MODEL) -> WanPipeline:
    """Builds `WanPipeline` on CPU f32 with the DiT, VAE and scheduler of the pinned revision (no
    umT5)."""
    assert_no_mps()
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )

    snapshot = local_snapshot(model)
    transformer = WanTransformer3DModel.from_pretrained(
        snapshot, subfolder="transformer", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler.from_pretrained(snapshot, subfolder="scheduler")
    pipeline = WanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def run(
    pipeline: WanPipeline,
    *,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    output_type: str = "latent",
    **call_kwargs: Any,
) -> torch.Tensor:
    """Injects the embeddings and the noise, runs `WanPipeline.__call__` and returns the output (by
    default the final latent).

    `negative_prompt_embeds` is required because without it upstream tries to embed an empty string
    with umT5 and fails with `text_encoder=None` (decision 4 passes the official negative
    explicitly). The video size is derived from the latent's shape (passing it separately would make
    it possible to build a shape that disagrees with the latent). `call_kwargs` such as
    `callback_on_step_end` go to upstream unchanged.
    """
    num_frames, height, width = video_size(latents)
    output = pipeline(
        prompt_embeds=pad_text_embeds(prompt_embeds),
        negative_prompt_embeds=pad_text_embeds(negative_prompt_embeds),
        latents=latents,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type=output_type,
        return_dict=False,
        **call_kwargs,
    )
    result: torch.Tensor = output[0]
    return result


def _smoke(model: str) -> int:
    """Checks that loading and 1 step (with CFG, latent `[1,16,3,16,16]` = S 192) + decode run.

    The embeddings and the noise are fixed-seed random numbers (umT5 is not loaded). No numbers are
    compared — making the reference outputs is the job of the step 2 / 4 scripts. The decode goes
    through the upstream path as is (de-normalization → `vae.decode` → post-processing).
    """
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randn(16, TEXT_DIM, generator=generator)
    negative = torch.randn(24, TEXT_DIM, generator=generator)
    latents = torch.randn(1, 16, 3, 16, 16, generator=generator)

    started = time.perf_counter()
    pipeline = load_pipeline(model)
    loaded = time.perf_counter()
    video = run(
        pipeline,
        prompt_embeds=prompt,
        negative_prompt_embeds=negative,
        latents=latents,
        num_inference_steps=1,
        output_type="pt",
    )
    finished = time.perf_counter()

    finite = bool(video.isfinite().all())
    print(f"load: {loaded - started:.1f} s")
    print(f"1 step (CFG) + vae decode: {finished - loaded:.1f} s  video {tuple(video.shape)}")
    print(f"finite: {finite}  range [{video.min().item():.4f}, {video.max().item():.4f}]")
    print(f"timesteps: {pipeline.scheduler.timesteps.tolist()}")
    return 0 if finite else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument("--smoke", action="store_true", help="loading plus a 1-step smoke run")
    args = parser.parse_args(argv)
    if not args.smoke:
        parser.error(
            "only --smoke for now (making the reference outputs is the job of the step 2 / 4"
            " scripts)"
        )
    with torch.no_grad():
        return _smoke(args.model)


if __name__ == "__main__":
    sys.exit(main())
