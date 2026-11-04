"""CPU reference of the two-step end-to-end I2V run of Wan2.2 TI2V-5B (ADR 0121 step 9b).

From one condition image it builds, at 1280×704 and 9 frames (latent `[48,3,44,80]`, S = 2,640,
condition tokens P = 880), the latents of 2 steps (with CFG, guidance 5.0, shift 5.0) and the RGB
frames obtained from them by tiled decode → unpatchify → clamp, by running the `expand_timesteps`
branch of diffusers' `WanImageToVideoPipeline` (the I2V of Wan2.2 5B) on CPU in f32. The GPU
end-to-end run (the I2V of `wan.Ti2vPipeline`) is checked against these values within a band. The
parts of the T2V `wan/ti2v_few_step_ref.py` are reused unchanged (the weight reader, the scheduler,
how the prompt rows are looked up, the tiled decode, the two-phase order).

## Conditioning (exactly the `expand_timesteps` branch of diffusers)

- Condition latent: the mu of the upstream non-tiled encode normalized as
  `(mu − mean)·f32(1/f32(std))` (`prepare_latents` — `[48,1,44,80]`).
- Each step: only the DiT input has its first latent frame replaced by the condition
  (`(1 − m)·cond + m·latents`, with m 0 on the first frame). The scheduler advances on the
  unreplaced latents (which is also what the callback receives).
- After the loop it is replaced once more before going to the VAE (`latents_final`).
- Timesteps: the condition tokens (the first P) get t = 0, the rest the generation-side t.

The official code (the i2v of Wan2.2's `textimage2video.py`) replaces the scheduler state itself
at the start and after every step. The DiT input is bit-identical between the two forms, and UniPC
(without thresholding) advances element by element, so the final output is bit-identical too
({@link sample} writes both forms as one path; `wan/tests/test_ti2v_i2v_few_step_ref.py` keeps this
in place with the pinned UniPC and a mock model, and a real run replays the official form from the
recorded DiT outputs and puts it in the summary — an observation, not a check).

## The timestep form (ADR 0121 decision 4)

Given a `[1,S]` timestep, the upstream forward runs the time MLP over S rows (M = S). The reference
is taken with the same semantics as Karume's graph — twice with M = 1, once per value (the
generation-side t and the condition side's 0). For the duration of the call, the DiT's
`condition_embedder` is replaced with {@link _PerStepTokenwiseEmbedder}, which reads the
generation-side t from each forward's flat timestep and delegates to the DiT reference wrapper
(`dit_patch._TokenwiseTimeEmbedder` — the same one as the DiT golden). The two deciding cases are
also run without the replacement (the diffusers pipeline as is, M = S) as an observation (not a
check).

## Inputs

- Condition image: `inputs/wan-i2v/<name>-832x480.png` (git-untracked; the sha256 is in
  `export_vae_encoder.IMAGES` — the same provenance as the step 9a goldens), cropped by the
  official preprocessing (`i2v_preprocess_ref.crop_resize` — LANCZOS at the covering scale →
  center crop), passed as an exact-size PIL image. The upstream `VideoProcessor` does not change
  the pixels of an image of this size (Pillow returns a copy for a same-size resize) — it is
  checked that the [-1, 1] image the pipeline passes to the encoder is bit-identical to the
  official `to_signed_unit` (fails loudly otherwise).
- Text context, initial noise, weights, scheduler: the same as the T2V reference (the module doc of
  `ti2v_few_step_ref`). The prompt is the fixed prompt with the same name as the image. The
  negative is given explicitly (left neither to diffusers' default `""` nor to the official
  default `sample_neg_prompt` — the same treatment as T2V).
- pipeline: built from the pinned snapshot with `WanImageToVideoPipeline.from_pretrained`
  (`expand_timesteps` is the model_index value — it is checked that the value read is true). All
  heavy components are passed in, so only the configs are read from the snapshot (repo id +
  `local_files_only` raises `IncompleteSnapshotError` for files not fetched, so the path of the
  fetched snapshot is passed).

## How it is put together (single process, two phases — the same as the T2V reference)

1. Phase 1 (DiT): every case runs 2 steps (the VAE encode happens inside the pipeline). The DiT
   outputs are recorded by a forward hook, and cond / uncond are told apart by the value of the
   text context the hook received. The two deciding cases also run on the M = S path.
2. Phase 2 (VAE): the DiT is released (fails loudly if a reference remains), and the replaced
   latents are turned into frames through the 2.2 path of step 5 (`vae_tiling` — blend in patchify
   space → upstream unpatchify → clamp).

## Output (`pipeline_steps_i2v.<case>.safetensors` at the root of the i8 DiT series)

Tensors (f32 — only the two image ones are u8):

- `latents_init` `[48,3,44,80]` — the injected initial noise (the first frame is still noise too —
  the replacement happens inside the pipeline).
- `condition_latents` `[48,1,44,80]` — the condition latent (after normalization).
- `noise_cond.<i>` / `noise_uncond.<i>` — the DiT outputs of each step.
- `latents.<i>` — the scheduler latents after each step (before replacement — the same values as
  diffusers' callback).
- `latents_final` — the latents with the first frame replaced after the loop (what goes to the
  VAE).
- `frames` `[3,9,704,1280]` — after unpatchify and clamp.
- `latents_mean` / `latents_std` `[48]`.
- `source` `[480,832,3]` u8 — the source image's RGB8 (the pixels of the PNG with the provenance
  sha256 — so the GPU test can run the preprocessing chain without relying on PNG decoding or
  another asset), `rgb8` `[704,1280,3]` u8 — the cropped RGB8.
- Deciding cases only: `observed_m_s.latents.<i>` / `observed_m_s.latents_final` (the M = S
  observation).

Meta: the fields of the T2V reference (`expand_timesteps` is `"true"`) plus `timestep_mlp`
({@link TIMESTEP_MLP}), `condition_tokens` (P), `condition_timestep` (`"0"`), `image`,
`image_sha256` (PNG), `source_rgb8_sha256` / `rgb8_sha256` (the raw u8 bytes), `width`, `height`,
`fit` (`"crop"`), `resample` (`"LANCZOS"`) and `version_<lib>` (the versions of the libraries that
determine the values).

## Cases (2 deciding + 1 acceptance — all three the same form)

The roles are assigned as for the step 9a encoder goldens (deciding = boxing-cats / cat-dog-baking,
acceptance = ferret). MUST: the `accept` results never change the deciding cases or the band. The
acceptance case keeps the same form as the deciding ones.

    uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref                       # all
    uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref --case accept-ferret  # one

## RAM and run time (measured 2026-10-09, Ryzen 5 7600, 6 torch threads)

- The RAM check's estimate is the T2V reference's estimate (`ti2v_few_step_ref.memory_estimate`
  23.7 GiB) plus the I2V excess of 0.5 GiB, about 24.2 GiB ({@link memory_estimate} —
  `require_available` adds a 1 GiB margin). The process's RSS peak was 24.07 GiB, 0.37 GiB above
  the T2V estimate (the excess comes from that measurement). S is shorter than for T2V, but the peak
  did not drop.
- The run time is 41 min 48 s for the 3 cases: phase 1 about 16 min (M = 1 takes 179 to 202 s per
  case, the M = S observation 190 to 202 s) + the tiled decode of 28 tiles × 3 chunks, 503 to
  506 s per case.
- The phase 1 results are kept in memory only (if phase 2 fails midway, rerun from phase 1).

MUST: diffusers / Pillow are imported inside functions (the `wan` group is not in the default sync
— `tests/test_optional_group_imports.py`).
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

#: The upstream model (a key of `wan.sources.SOURCES`).
MODEL = ti2v_few_step_ref.MODEL

#: The step count, shift and CFG strength of the few-step run (the same values as the T2V
#: reference).
STEPS = ti2v_few_step_ref.STEPS
FLOW_SHIFT = ti2v_few_step_ref.FLOW_SHIFT
GUIDANCE_SCALE = ti2v_few_step_ref.GUIDANCE_SCALE

#: Latent `[C, F, H, W]` (1280×704, 9 frames — 1 condition + 2 generated latent frames, so even an
#: off-by-one in the replaced frame's index (e.g. replacing every other frame) shows at the
#: observation points).
LATENT_SHAPE = (48, 3, 44, 80)

#: How width and height are fitted (the official preprocessing — `i2v_preprocess_ref.crop_resize`)
#: and its resample.
FIT: i2v_preprocess_ref.Fit = "crop"
RESAMPLE = "LANCZOS"

#: The timestep of the condition tokens (the upstream `first_frame_mask` sets the first latent frame
#: to 0).
CONDITION_TIMESTEP = 0

#: The series root (the i8 DiT characterizes the reference preset, so it goes in the same i8 DiT
#: series as the T2V reference).
SERIES = ti2v_few_step_ref.SERIES

#: The reference's file name (at the series root — `pipeline_steps_i2v.<case>.safetensors`, next to
#: the T2V `pipeline_steps.<case>` at the same root; the distribution uses an allow-list
#: (`distribution.wan_placements`), so adding it here does not put it into the distribution).
FIXTURE_PREFIX = "pipeline_steps_i2v."
FIXTURE_SUFFIX = ".safetensors"

#: The meta's `weights` (the same as the T2V reference — the DiT is i8 RTN fake-quant, the VAE f16
#: representable values).
WEIGHTS = ti2v_few_step_ref.WEIGHTS

#: The meta's `timestep_mlp` (how the time MLP is run — decision 4: M = 1 per value, generation t
#: and condition 0).
TIMESTEP_MLP = "per-value-m1"

#: The meta's `reference` (what the reference is).
REFERENCE = (
    "diffusers WanImageToVideoPipeline (expand_timesteps=True read from the pinned model_index) CPU"
    " f32, umT5 embeddings injected, the time MLP run once per value at M=1 (generation t,"
    " condition 0; ADR 0121 decision 4), the condition image cropped by the official Wan2.2"
    " preprocessing; then the snapped-tile decode of the final replaced latents with upstream"
    " blend_v / blend_h in patchify space, upstream unpatchify and clamp"
)

#: The deciding cases' meta `observed_m_s` (what the observation is — not a check).
OBSERVED_M_S = (
    "scheduler latents after each step and the final replaced latents of the plain diffusers"
    " WanImageToVideoPipeline (timestep [1,S], the time MLP over all S tokens); observation only,"
    " not a gate"
)


@dataclass(frozen=True)
class FixtureCase(ti2v_few_step_ref.FixtureCase):
    """One few-step I2V run (a T2V case plus the name of the condition image)."""

    #: The condition image's name (`name` of `export_vae_encoder.IMAGES`).
    image: str


#: 2 deciding cases (they decide the band) + 1 acceptance case. The prompt is the fixed prompt with
#: the same name as the image. The seeds differ from both the T2V reference (20262101 to 03) and
#: the 2.1 reference.
#: MUST: the `accept` results never change the deciding cases or the band.
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


#: The excess by which the I2V peak exceeded the T2V estimate (`ti2v_few_step_ref.memory_estimate`).
#: The measured peak (2026-10-09, 3 cases) of 24.07 GiB exceeded the T2V estimate of 23.70 GiB by
#: 0.37 GiB — rounded up to 0.5 GiB. The source is not identified (speculation: the VAE encode
#: inside the pipeline and the activations of the M = S observation).
I2V_EXCESS_BYTES = int(0.5 * 2**30)


def memory_estimate(model: str = MODEL) -> int:
    """The RAM estimate of the I2V reference (bytes) = the T2V estimate + {@link I2V_EXCESS_BYTES}
    (about 24.2 GiB for 5B — covers the measured peak of 24.07 GiB; `require_available` adds a
    1 GiB margin).

    MUST: never estimate below the measured peak — the 1 GiB margin would be eaten by the
    estimate's error, and run alongside other work the 42-minute run would hit OOM or swap midway.
    """
    return ti2v_few_step_ref.memory_estimate(model) + I2V_EXCESS_BYTES


# ---- Condition image ------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceImage:
    """One source condition image (the provenance name, the PNG's sha256 and the opened RGB
    image)."""

    name: str
    sha256: str
    image: Image.Image


@dataclass(frozen=True)
class ConditionImage:
    """The condition image of one case (the source image and the exact-size image cropped by the
    official preprocessing)."""

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
        """The official [-1, 1] (`to_signed_unit`) `[1, 3, H, W]` — what the image the pipeline
        passes to the encoder is checked against."""
        return i2v_preprocess_ref.to_signed_unit(self.cropped)[:, :, 0]


def condition_image(source: SourceImage, width: int, height: int) -> ConditionImage:
    """Crop the source image to `width × height` with the official preprocessing (`crop`)."""
    return ConditionImage(
        source=source, cropped=i2v_preprocess_ref.preprocess(source.image, width, height, FIT)
    )


def load_sources(cases: Sequence[FixtureCase]) -> dict[str, SourceImage]:
    """Open the condition images the cases use (fails loudly if the sha256 or the size differs
    from the provenance — call before the heavy loading)."""
    images = {image.name: image for image in export_vae_encoder.IMAGES}
    sources: dict[str, SourceImage] = {}
    for name in sorted({case.image for case in cases}):
        image = images.get(name)
        if image is None:
            raise ValueError(
                f"condition image {name!r} is not in the provenance table ({sorted(images)})"
            )
        sources[name] = SourceImage(name, image.sha256, export_vae_encoder.load_image(image))
    return sources


def _bytes_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


# ---- Conditioning (the replacement formula, two forms) -------------------------------------


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
    """The upstream `first_frame_mask` `[1, 1, F, H, W]` (f32 — 0 on the first latent frame, 1 on
    the rest)."""
    _, frames, height, width = latent_shape
    mask = torch.ones(1, 1, frames, height, width, dtype=torch.float32)
    mask[:, :, 0] = 0
    return mask


def replace_first_frame(
    latents: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Replace the first latent frame with the condition, `(1 − m)·cond + m·latents` (the same
    formula in diffusers and the official code).

    `condition` is `[1, C, 1, H, W]` (broadcast over the frame axis). Written as the formula —
    0·cond only adds ±0, so the frames other than the first are bit-identical to `latents`, but a
    ±Inf in cond spreads to every frame as 0·Inf = NaN (which is why the condition latent is
    checked to be finite first).
    """
    return (1 - mask) * condition + mask * latents


#: The replacement form ({@link sample}).
Form = Literal["diffusers", "official"]
FORMS: tuple[Form, ...] = ("diffusers", "official")

#: The velocity of one step (`index`, timestep, DiT input → DiT output after CFG).
Velocity = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Sampled:
    """What {@link sample} returns (sequences of `[1, C, F, H, W]`)."""

    #: The DiT input of each step.
    model_inputs: list[torch.Tensor]
    #: The scheduler output of each step (before that form's replacement).
    states: list[torch.Tensor]
    #: The latents passed to the VAE (the first frame is the condition).
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
    """Run the I2V denoising loop in one of the two replacement forms (`scheduler` is a new one
    with `set_timesteps` already applied).

    - `diffusers` (the `expand_timesteps` branch of `WanImageToVideoPipeline`): replace only the
      DiT input, advance the scheduler on the unreplaced latents, and replace once after the loop.
    - `official` (the i2v of the official Wan2.2): replace the scheduler state itself at the start
      and after every step.

    The DiT input has the same values in both forms (the first frame is the condition in both, the
    rest the same scheduler output). The scheduler states differ only in the first frame's region,
    and the final replacement removes that difference.
    """
    if form not in FORMS:
        raise ValueError(f"unknown replacement form {form!r} ({', '.join(FORMS)})")
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
    """A new UniPC with the same config, `set_timesteps` with `steps` (for replay — it shares no
    state with the pipeline)."""
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_config(config)
    scheduler.set_timesteps(steps)
    scheduler.set_begin_index(0)
    return scheduler


def recorded_velocity(
    noise_cond: Sequence[torch.Tensor], noise_uncond: Sequence[torch.Tensor], guidance: float
) -> Velocity:
    """A velocity that builds the same CFG as upstream from the recorded DiT outputs
    (`[C, F, H, W]`) (it does not read the DiT input — the caller checks through `model_inputs` that
    the DiT input is the same in both forms)."""

    def velocity(index: int, _timestep: torch.Tensor, _model_input: torch.Tensor) -> torch.Tensor:
        uncond = noise_uncond[index][None]
        return uncond + guidance * (noise_cond[index][None] - uncond)

    return velocity


# ---- The timestep form (applying decision 4's reference wrapper to each pipeline forward) ----


def condition_token_mask(transformer: nn.Module, latent_shape: Sequence[int]) -> torch.Tensor:
    """The condition-token mask `[1, S]` (bool — true for the P = H'·W' tokens of the first latent
    frame — `dit_patch.dit_condition_mask`, the same as the graph input `condition_mask`)."""
    _, frames, height, width = latent_shape
    patch = tuple(transformer.config.patch_size)
    return dit_patch.dit_condition_mask((frames, height, width), patch, conditioned=True)[..., 0]


class _PerStepTokenwiseEmbedder(nn.Module):
    """A replacement that reads the generation-side t from each forward's flat timestep `[S]` and
    delegates to the DiT reference wrapper (`dit_patch._TokenwiseTimeEmbedder` — the time MLP twice
    with M = 1, once per value).

    The pipeline changes t every step, but the reference wrapper takes the generation-side t at
    construction — this builds one per forward. The generation-side t is passed as int64 (the same
    dtype as `t.expand(1)` on the M = 1 path of the T2V reference — the sinusoidal embedding calls
    `.float()`, so the values are the same too).

    MUST: fail loudly unless the timestep of the generation-side tokens is a single value and an
    integer. That the condition side is 0 and matches the mask is checked by the reference wrapper
    (agreement with `where(mask, 0, t)`).
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
                f"the timestep shape {tuple(timestep.shape)} and seq_len {timestep_seq_len}"
                f" upstream passed are not the per-token ({tokens},) (it should be the"
                " expand_timesteps branch)"
            )
        values = timestep[~self.mask[0]].unique()
        if values.numel() != 1:
            raise AssertionError(
                f"the timestep of the generation-side tokens takes {values.numel()} values"
                " (it should be one)"
            )
        value = float(values[0])
        if not value.is_integer():
            raise AssertionError(f"the generation-side timestep {value} is not an integer")
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
    """Replace the DiT's `condition_embedder` with {@link _PerStepTokenwiseEmbedder} for the
    duration (yields the list of generation-side t per forward). It is restored on exit, even on
    an exception."""
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
    """Build a `WanImageToVideoPipeline` without umT5 from the pinned snapshot (every heavy
    component is passed in).

    `expand_timesteps` is the model_index value (true at the pin) — false would select the branch
    of Wan2.1's I2V (channel concatenation), so it stops. The two-stage DiT (`boundary_ratio`) and
    the image encoder (the CLIP of Wan2.1's I2V) are not used.
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
            f"expand_timesteps of model_index is {pipeline.config.expand_timesteps} (true for the"
            " I2V branch of Wan2.2 5B)"
        )
    if pipeline.config.boundary_ratio is not None or pipeline.transformer_2 is not None:
        raise AssertionError("the two-stage DiT (boundary_ratio / transformer_2) is not used")
    if pipeline.image_encoder is not None or transformer.config.image_dim is not None:
        raise AssertionError("the image encoder (the CLIP path of Wan2.1's I2V) is not used")
    if (
        pipeline.transformer is not transformer
        or pipeline.vae is not vae
        or pipeline.scheduler is not scheduler
    ):
        raise AssertionError("the pipeline holds components other than the ones passed")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@contextmanager
def captured_condition(
    pipeline: WanImageToVideoPipeline, image: ConditionImage, mask: torch.Tensor
) -> Iterator[dict[str, torch.Tensor]]:
    """Copy the condition latent `prepare_latents` returns during the block (yields
    `{"condition": …}`).

    MUST (each fails loudly): the [-1, 1] image the pipeline passes to the encoder is bit-identical
    to the official preprocessing, the upstream `first_frame_mask` is the same as
    {@link first_frame_mask}, and the condition latent is finite (a ±Inf in cond spreads to every
    frame as NaN through the replacement formula).
    """
    captured: dict[str, torch.Tensor] = {}
    prepare = pipeline.prepare_latents
    expected_image = image.signed

    def capture(sample: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if not torch.equal(sample, expected_image):
            raise AssertionError(
                f"{image.source.name}: the pipeline's preprocessed image is not bit-identical to"
                " the official [-1, 1]"
            )
        latents, condition, upstream_mask = prepare(sample, *args, **kwargs)
        if not torch.equal(upstream_mask, mask):
            raise AssertionError(
                f"the upstream first_frame_mask {tuple(upstream_mask.shape)} differs"
            )
        if not bool(condition.isfinite().all()):
            raise AssertionError(f"{image.source.name}: the condition latent has non-finite values")
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
    """One case of phase 1 (passed to phase 2 and to the writer — latents are `[C, F, H, W]`)."""

    case: FixtureCase
    names: ti2v_few_step_ref.PromptNames
    image: ConditionImage
    latents_init: torch.Tensor
    #: The condition latent `[C, 1, H, W]` (after normalization).
    condition: torch.Tensor
    #: The number of condition tokens P (the tokens of the first latent frame, counted with the
    #: DiT's patch).
    condition_tokens: int
    #: The DiT input `[1, C, F, H, W]` of each step (not written — for checking the replay of the
    #: official form).
    model_inputs: list[torch.Tensor]
    noise_cond: list[torch.Tensor]
    noise_uncond: list[torch.Tensor]
    #: The scheduler latents after each step (before replacement).
    latents: list[torch.Tensor]
    #: The latents replaced after the loop (what goes to the VAE).
    latents_final: torch.Tensor
    timesteps: list[int]
    sigmas: list[float]
    seconds: dict[str, float]
    #: The per-step latents and the replaced latents of the M = S path, and whether its condition
    #: latent is the same as for M = 1 (deciding cases only).
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
    """Run one case for 2 steps with decision 4's timestep form (M = 1 per value)."""
    generator = torch.Generator().manual_seed(case.seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)
    tokens = condition_token_mask(pipeline.transformer, LATENT_SHAPE)
    mask = first_frame_mask(LATENT_SHAPE)

    # The same values as the contexts passed upstream (`_call` fills them with the same function) —
    # the hook compares against these by value.
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
                    f"{case.name}: the timestep shape {tuple(timestep.shape)} is not"
                    f" (1, {tokens.shape[1]}) (it should be the expand_timesteps branch)"
                )
            hidden = kwargs["hidden_states"]
            # The replacement reaches the DiT input (first frame = the condition latent).
            if not torch.equal(hidden[:, :, :1], captured["condition"]):
                raise AssertionError(
                    f"{case.name}: the first frame of the DiT input is not the condition latent"
                )
            context = kwargs["encoder_hidden_states"]
            labels = [label for label, value in contexts.items() if torch.equal(context, value)]
            if len(labels) != 1:
                raise AssertionError(
                    f"{case.name}: the DiT context matches {len(labels)} of the cond / uncond"
                    " embeddings (it should be one) — cannot tell them apart"
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
            f"{len(forwards)} forwards, {len(steps)} steps (2 steps with CFG expected)"
        )
    if [t for t, _, _, _ in forwards] != expected or seen != expected:
        raise AssertionError(
            f"the forward timesteps {[t for t, _, _, _ in forwards]} and the time-MLP t {seen} do"
            f" not match {timesteps}"
        )
    outputs: dict[tuple[int, str], torch.Tensor] = {}
    model_inputs: dict[tuple[int, str], torch.Tensor] = {}
    for index, (_, label, output, hidden) in enumerate(forwards):
        outputs[(index // 2, label)] = output
        model_inputs[(index // 2, label)] = hidden
    if sorted(outputs) != [(step, label) for step in range(STEPS) for label in sorted(contexts)]:
        raise AssertionError(
            "not exactly one cond and one uncond per step"
            f" ({[label for _, label, _, _ in forwards]})"
        )
    for step in range(STEPS):
        if not torch.equal(model_inputs[(step, "cond")], model_inputs[(step, "uncond")]):
            raise AssertionError(f"{case.name}: the cond / uncond DiT inputs of step {step} differ")
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
    """Run the same request through the diffusers pipeline as is (the M = S path, the time MLP over
    S rows) and fill the observation fields of `denoised` (an observation — not a check).

    The hook is a different one from M = 1: it checks that there is no timestep replacement, that
    the timestep `[1,S]` is 0 on the condition tokens and a single value on the generation side, and
    whether the timestep sequence is the same as for M = 1.
    """
    case = denoised.case
    transformer = pipeline.transformer
    if isinstance(transformer.condition_embedder, _PerStepTokenwiseEmbedder):
        raise AssertionError(
            f"{case.name}: the timestep replacement is still in place for the M = S observation"
        )
    tokens = condition_token_mask(transformer, LATENT_SHAPE)
    seen: list[int] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], _output: Any) -> None:
        timestep = kwargs["timestep"]
        if tuple(timestep.shape) != (1, tokens.shape[1]):
            raise AssertionError(
                f"{case.name}: the timestep shape {tuple(timestep.shape)} on the M = S path is"
                f" not (1, {tokens.shape[1]})"
            )
        values = timestep[0][~tokens[0]].unique()
        if values.numel() != 1 or bool(timestep[0][tokens[0]].any()):
            raise AssertionError(
                f"{case.name}: the M = S timestep is not 0 on the condition tokens and a single"
                " value on the generation side"
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
            f"{case.name}: the M = S forward timesteps {seen} and {len(steps)} steps do not match"
            f" {denoised.timesteps} of M = 1"
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
    """Phase 1: load the DiT and run every case. References to the DiT live only inside this
    function (released on return).

    The returned weak reference is for checking that the DiT was collected before phase 2
    (`ti2v_few_step_ref.assert_released`).
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


# ---- Replay and observation ------------------------------------------------------------------


def replay(denoised: Denoised, scheduler_config: Mapping[str, Any]) -> dict[str, Any]:
    """Replay both forms from the recorded DiT outputs.

    MUST (fail loudly): the diffusers form reproduces the pipeline's record (the DiT inputs, the
    per-step latents, the replaced latents) bit for bit — the self-consistency of the recorded
    cond / uncond, the CFG and the replacement formula. The official form is an observation (if the
    DiT inputs have the same values as in the diffusers form, the official form can be replayed
    correctly from the recorded outputs — that premise and the agreement of the final output go
    into the summary).
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
            raise AssertionError(
                f"{name}: the replayed DiT input of step {index} differs from the record"
            )
        if not torch.equal(upstream.states[index][0], denoised.latents[index]):
            raise AssertionError(
                f"{name}: the replayed latents of step {index} differ from the record"
            )
    if not torch.equal(upstream.final[0], denoised.latents_final):
        raise AssertionError(f"{name}: the replayed replaced latents differ from the record")
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


#: The tensors checked against the step 9a encoder golden (golden key → how this reference's value
#: is taken).
_ENCODER_GOLDEN_KEYS: Mapping[str, Callable[[Denoised], torch.Tensor]] = {
    "source": lambda denoised: denoised.image.source_rgb8,
    "rgb8": lambda denoised: denoised.image.rgb8,
    "latent": lambda denoised: denoised.condition,
}


def check_encoder_golden(denoised: Denoised, root: Path | None) -> dict[str, Any] | str:
    """Check the source image, the cropped image and the condition latent against the step 9a
    encoder golden (`source` / `rgb8` / `latent` of the same image and size).

    MUST (fail loudly): bit-identical whenever the golden exists. Writing despite a mismatch would
    derive the band of the TS end-to-end e2e and the condition-latent check (which applies the 9a
    encoder band) against a reference that drifted from the 9a golden. On a machine without the
    golden it returns `missing <path>` as an observation (the golden is written by another script,
    `wan.export_vae_encoder` — the reference can be built without it).
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
            f"{denoised.case.name}: not bit-identical to the step 9a encoder golden {path}"
            f" ({mismatched}) — stopping without writing the reference"
        )
    return {"bit_exact": True, "keys": sorted(_ENCODER_GOLDEN_KEYS), "path": str(path)}


def _golden_difference(expected: torch.Tensor, got: torch.Tensor) -> str:
    """A summary of the mismatch with the golden (the shape and dtype if they differ, else the max
    absolute difference)."""
    if expected.shape != got.shape or expected.dtype != got.dtype:
        return (
            f"golden {tuple(expected.shape)} {expected.dtype} / reference {tuple(got.shape)}"
            f" {got.dtype}"
        )
    return f"max_abs {float((expected.double() - got.double()).abs().max())}"


def first_frame_difference(frames: torch.Tensor, image: ConditionImage) -> dict[str, float]:
    """The difference between the output's first frame and the condition image ([-1, 1]) (an
    observation — the size of the VAE round-trip error)."""
    difference = (frames[:, 0] - image.signed[0]).abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}


# ---- Writing ---------------------------------------------------------------------------


def case_tensors(
    denoised: Denoised, frames: torch.Tensor, vae: AutoencoderKLWan
) -> dict[str, torch.Tensor]:
    """The tensors to write (the two image ones u8, the rest f32; fails loudly on non-finite
    values)."""
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
            raise AssertionError(f"{denoised.case.name}: {name} is not f32 ({tensor.dtype})")
        if not bool(tensor.isfinite().all()):
            raise AssertionError(f"{denoised.case.name}: {name} has non-finite values")
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
    """The meta to write (safetensors meta holds strings only)."""
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
    """The meta common to every case (the fields of the T2V reference + the versions of the
    libraries that determine the values — LANCZOS depends on the Pillow version)."""
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
    """Run and write every case in two phases (staging → replace). Returns the summaries.

    MUST: phase 2 starts only after the DiT is released (`ti2v_few_step_ref.assert_released`) —
    the VAE decode is never stacked on the phase 1 peak.
    """
    patch_size = vae.config.patch_size
    if patch_size is None:
        raise ValueError(
            "the VAE's patch_size is None — pass a Wan2.2 VAE (the generation that patchifies)"
        )
    denoised_cases, released = denoise_phase(
        load_transformer, snapshot, vae, scheduler, cases, embeds, names, sources
    )
    ti2v_few_step_ref.assert_released(released)
    # The check against the 9a golden runs for every case before phase 2 (about 8 minutes of
    # decode per case) — on a mismatch nothing is written.
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
                f"{case.name}: the frames shape {tuple(frames.shape)} is not {expected}"
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
        "--out", type=Path, default=SERIES, help="the series root (default: the i8 DiT series)"
    )
    parser.add_argument(
        "--embeds",
        type=Path,
        default=SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME,
        help="the text-embedding asset (the output of wan.text_embeds)",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in FIXTURE_CASES],
        default=None,
        help="the cases to write (default: all)",
    )
    args = parser.parse_args(argv)
    if not args.embeds.is_file():
        parser.error(f"{args.embeds} is missing — build it first with `python -m wan.text_embeds`")

    embeds, asset_metadata = text_embeds.read_asset(args.embeds)
    selected = [case for case in FIXTURE_CASES if args.case is None or case.name in args.case]
    # Before the heavy loading, check the asset rows and roles and the condition images'
    # provenance for every case.
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
