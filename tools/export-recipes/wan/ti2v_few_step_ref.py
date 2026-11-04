"""The two-step end-to-end CPU reference of Wan2.2 TI2V-5B (ADR 0121 step 6 — T2V).

It runs diffusers' `WanPipeline` unmodified on the CPU in f32 for two steps (with CFG, guidance 5.0,
shift 5.0) at 1280×704 with 17 frames (latent `[48,5,44,80]`, S = 4,400) and produces the latents
(before the VAE) and the RGB frames from tiled decode → unpatchify → clamp of them. The GPU end to
end (`wan.Ti2vPipeline`) is compared with these values within a band. It is the 2.2 version of
2.1's `wan/few_step_ref.py`; the 2.1 script and `pipeline_ref.py` do not change
(`pipeline_ref.video_size` hard-codes 16 ch and scale 8) — only `pipeline_ref.pad_text_embeds` and
`few_step_ref.denormalize_latents` are used.

## Inputs

- Text context: the embedding asset (`wan.text_embeds` — the same bytes as 2.1's asset, ADR 0121
  decision 9). Prompts are looked up by `name` in the asset's metadata and their role (positive /
  negative) is checked. The negative is the row whose `name` is `negative`. The asset's sha256 is
  written into the metadata (when the embeddings are rebuilt, the reference is rebuilt too).
- Initial noise: torch CPU `randn` (fixed seed, `[1,48,5,44,80]`) injected into the upstream's
  `latents=`.
- Weights: the DiT is loaded whole in f32 and then rounded to per-channel i8 representable values
  (RTN fake-quant — the same procedure as `ti2v_export_dit.eager_full`, so the values equal the
  series' i8). The VAE is the f16 representable values of `export_vae.load_vae(round_f16=True)`.
- scheduler: the pinned config is loaded with `flow_shift=5.0` given explicitly, and the loaded
  value is checked (ADR 0121 decision 8 — the pinned value is also 5.0, but the reference's premise
  is not left to the config's default).

## How it is put together (single process, two phases)

1. Phase 1 (DiT): run every case for two steps with `WanPipeline(expand_timesteps=False)`. Time
   takes the one-dimensional timestep path (the time MLP with M = 1, modulation `[B,6,C]`), the same
   as decision 4's T2V form. The DiT outputs are recorded with a forward hook, and the cond / uncond
   split is decided by the value of the text context the hook received (the same as
   `few_step_ref` — independent of the call order).
2. Observation (not a gate): the two decision cases also run two steps with `expand_timesteps=True`
   (the upstream default, the M = S path where the timestep is `[1,S]`), and the latents of each
   step are written to `observed_m_s.latents.<i>`. The hook is a different one (it checks that the
   timestep is `[1,S]` with a single value — `int()` of `few_step_ref` does not take `[S]`).
3. Phase 2 (VAE): release the DiT (fail loudly if a reference remains), and make the frames through
   step 5's 2.2 path (`vae_tiling` — blend in patchify space → upstream unpatchify → clamp).

## Outputs (`pipeline_steps.<case>.safetensors` at the root of the i8 DiT series)

- Tensors: `latents_init`, `noise_cond.<i>`, `noise_uncond.<i>`, `latents.<i>` (`[48,5,44,80]`),
  `frames [3,17,704,1280]` (after unpatchify and clamping), `latents_mean` / `latents_std` (f32, 48
  values), and `observed_m_s.latents.<i>` for the decision cases only.
- Metadata: role / seed / prompt / negative / `text_embeds_sha256` / steps / `guidance_scale` /
  `flow_shift` (`"5.0"`) / timesteps / sigmas / weights / reference / source / the tile plan
  (`vae_tiling.TilePlan.meta` + `patch_size`) / `expand_timesteps` (`"false"`) / `environment`
  (torch version, thread count, MKL / OpenMP environment variables — the last bits of f32 depend on
  MKL's behavior; `dit_reference.numeric_environment`), and `observed_m_s` (a description of the
  observation) for the decision cases only.

## Cases (2 decision + 1 acceptance — all three the same shape)

MUST: do not change the decision cases or the band after seeing the result of `accept`. Do not give
the acceptance case a different shape from the decision cases (a different shape mixes in other
reasons for leaving the band).

    uv run --group wan --inexact python -m wan.ti2v_few_step_ref                          # all
    uv run --group wan --inexact python -m wan.ti2v_few_step_ref --case accept-ferret     # one only

## RAM and run time (guesses)

- The RAM peak is about 24 GiB in phase 1 (the {@link memory_estimate} estimate of 23.7 GiB = the
  DiT's whole f32 18.63 + base 0.6 + the gap between measurement and formula 0.35 + activations at
  S = 4,400 1.5 + the VAE's f32 2.63). Before loading it checks MemAvailable and stops if it falls
  short of the estimate + a 1 GiB margin (about 24.7 GiB). If the estimate is wrong, the peak can
  still run short after passing the check (the check is not a guarantee). Phase 2 is about 6 GiB.
- The run time is about 3 hours for the 3 cases (20 DiT forwards × about 2 minutes + tiled decode of
  28 tiles × 5 chunks × 3 cases).
- Phase 1's results are held only in memory. If the DiT release check stops it or it fails partway
  through phase 2, phase 1 (about 40 minutes) is run again (the files of cases already written
  remain).
- The fallback when free memory is short is the layer-by-layer writer (`dit_reference.LayerwiseDit`
  — peak about 5 GiB), but it is not a component that can be swapped into `WanPipeline.transformer`
  (it has its own `forward`), so a new adapter that matches the diffusers call would have to be
  written (there is none here).

MUST: diffusers is imported inside functions (the `wan` group is not part of the default sync —
`tests/test_optional_group_imports.py`).
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

#: The upstream model (a key of `wan.sources.SOURCES`).
MODEL = "ti2v-5b"

#: The step count of the few-step end to end.
STEPS = 2

#: The latent `[C, F, H, W]` (1280×704, 17 frames — the official supported size, the maintainer's
#: ruling of 2026-10-05).
LATENT_SHAPE = (48, 5, 44, 80)

#: UniPC's shift (the pinned scheduler's `flow_shift` = the official 720p value — loaded explicitly
#: and checked).
FLOW_SHIFT = 5.0

#: The CFG strength (the same value as the 2.1 reference).
GUIDANCE_SCALE = pipeline_ref.GUIDANCE_SCALE

#: The series root (the i8 DiT characterizes the reference preset, so it goes in the i8 DiT
#: series).
SERIES = ti2v_export_dit.SERIES

#: The VAE series (the source of truth of the tile and the patchify factor).
VAE_SERIES = export_vae.VAE_SERIES[MODEL]

#: The metadata's `weights` (the DiT is i8 RTN fake-quant, the VAE is f16 representable values).
WEIGHTS = "dit-i8-rtn-fake-quant+vae-f16-rounded"

#: The metadata's `reference` (the reference's origin).
REFERENCE = (
    "diffusers WanPipeline(expand_timesteps=False) CPU f32 (umT5 embeddings injected), then the"
    " snapped-tile decode with upstream blend_v / blend_h in patchify space, upstream unpatchify"
    " and clamp"
)

#: The decision cases' metadata `observed_m_s` (the observation's origin — not a gate).
OBSERVED_M_S = (
    "latents after each step with expand_timesteps=True (timestep [1,S], the upstream default);"
    " observation only, not a gate"
)


@dataclass(frozen=True)
class FixtureCase:
    """One few-step end-to-end run (the seed of the initial noise and the positive prompt)."""

    name: str
    #: `band` = a case that decides the band, `accept` = a case that judges acceptance.
    role: str
    seed: int
    #: The name of the positive prompt in the embedding asset (`name` in the asset's metadata).
    prompt: str

    @property
    def observes_m_s(self) -> bool:
        """Whether the observation of the M = S path is also taken (decision cases only)."""
        return self.role == "band"


#: 2 decision cases (they decide the band) + 1 acceptance case. The seeds differ from both the 2.1
#: reference (20261030–33) and the DiT golden (the low part of the 20262000s).
#: MUST: do not change the decision cases or the band after seeing the result of `accept`.
FIXTURE_CASES = (
    FixtureCase("band-boxing-cats", role="band", seed=20262101, prompt="boxing-cats"),
    FixtureCase("band-cat-dog-baking", role="band", seed=20262102, prompt="cat-dog-baking"),
    FixtureCase("accept-ferret", role="accept", seed=20262103, prompt="ferret"),
)

#: The name of the negative row (`name` in the asset's metadata).
NEGATIVE_NAME = "negative"


@dataclass(frozen=True)
class PromptNames:
    """The positive / negative names one case looks up in the asset."""

    positive: str
    negative: str


def prompt_names(asset_metadata: Mapping[str, Any], case: FixtureCase) -> PromptNames:
    """Look up the positive / negative rows by `name` in the asset's metadata and check their roles
    (fail loudly on a mismatch)."""
    entries = {entry["name"]: entry for entry in asset_metadata["prompts"]}
    for name, role in ((case.prompt, prompts.POSITIVE), (NEGATIVE_NAME, prompts.NEGATIVE)):
        entry = entries.get(name)
        if entry is None:
            raise ValueError(
                f"{case.name}: the embedding asset has no row {name!r} ({sorted(entries)})"
            )
        if entry["role"] != role:
            raise ValueError(
                f"{case.name}: the role of {name!r} in the asset is {entry['role']!r} (expected"
                f" {role})"
            )
    return PromptNames(positive=case.prompt, negative=NEGATIVE_NAME)


def video_size(pipeline: WanPipeline, latent_shape: Sequence[int]) -> tuple[int, int, int]:
    """Latent `[C, F, H, W]` → the upstream's `(num_frames, height, width)`.

    The scale factors are the values the upstream took from the vae's config
    (`vae_scale_factor_*` — 4 / 16 for 2.2). The upstream `prepare_latents` does not look at the
    shape of the injected latents, so the channels are checked against both the DiT and the VAE.
    """
    channels, frames, height, width = latent_shape
    if channels != pipeline.vae.config.z_dim:
        raise ValueError(
            f"latent channels {channels} differ from the VAE's z_dim {pipeline.vae.config.z_dim}"
        )
    if pipeline.transformer is not None and channels != pipeline.transformer.config.in_channels:
        raise ValueError(
            f"latent channels {channels} differ from the DiT's in_channels"
            f" {pipeline.transformer.config.in_channels}"
        )
    temporal = pipeline.vae_scale_factor_temporal
    spatial = pipeline.vae_scale_factor_spatial
    return temporal * (frames - 1) + 1, spatial * height, spatial * width


def token_count(transformer: nn.Module, latent_shape: Sequence[int]) -> int:
    """The DiT's token count S = F/pt · H/ph · W/pw (the timestep length of the M = S path)."""
    patch_t, patch_h, patch_w = transformer.config.patch_size
    _, frames, height, width = latent_shape
    return (frames // patch_t) * (height // patch_h) * (width // patch_w)


def load_quantized_transformer(model: str = MODEL) -> nn.Module:
    """Load the DiT whole in f32 and round it to i8 representable values (the same procedure as
    `ti2v_export_dit.eager_full`)."""
    upstream = export_dit.load_transformer(model)
    report = export_dit.fake_quant_i8(upstream, dit_patch.WanDitTokensTi2v(upstream))
    print(f"[fake-quant] transformer: {report.describe()}", flush=True)
    return upstream


def load_rounded_vae(model: str = MODEL) -> AutoencoderKLWan:
    """Load the VAE as f16 representable values and check it against the series' premises
    (patch_size, spatial compression)."""
    vae = export_vae.load_vae(model, round_f16=True)
    export_vae.assert_series_config(vae, VAE_SERIES)
    if vae.use_tiling:
        # MUST: the upstream tiling scans differently (the module doc of `vae_tiling`).
        raise SystemExit("vae.use_tiling is True — the upstream tiling is not used")
    return vae


def load_scheduler(snapshot: Path) -> UniPCMultistepScheduler:
    """Load the pinned scheduler with `flow_shift` given explicitly and check the loaded value
    (decision 8)."""
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(
        snapshot, subfolder="scheduler", flow_shift=FLOW_SHIFT
    )
    if scheduler.config.flow_shift != FLOW_SHIFT:
        raise AssertionError(
            f"scheduler flow_shift {scheduler.config.flow_shift} is not {FLOW_SHIFT}"
        )
    # `flow_shift` acts on σ only with flow σ and without dynamic shift (with dynamic shift,
    # `set_timesteps` switches to the `mu` formula and does not read `flow_shift`).
    if (
        scheduler.config.use_dynamic_shifting is not False
        or scheduler.config.use_flow_sigmas is not True
    ):
        raise AssertionError(
            "scheduler use_dynamic_shifting"
            f" {scheduler.config.use_dynamic_shifting}, use_flow_sigmas"
            f" {scheduler.config.use_flow_sigmas} — not the form where the flow_shift formula"
            " applies (False / True)"
        )
    return scheduler


def build_pipeline(
    transformer: nn.Module | None,
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    *,
    expand_timesteps: bool,
) -> WanPipeline:
    """Build a `WanPipeline` without umT5 (the vae is kept — without it the spatial scale falls
    back to the default 8)."""
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


#: How much the measured peak of whole f32 + fake-quant exceeded the formula (whole f32 + base)
#: (eager-full of the ADR 0121 addendum "eager equivalence with the whole f32" — peak 19.57 GiB,
#: formula 18.63 + 0.6 = 19.23 GiB). Step 0's 20.89 GiB includes the layer-by-layer f64 (1.89 GiB),
#: so it is not the counterpart of this comparison.
MEASURED_EXCESS_BYTES = int(0.35 * 2**30)

#: The estimated activation peak of one forward at S = 4,400 (a guess — not measured). The M = S
#: observation is the largest: the `[1,S,6,3072]` modulation (2 of them, the whole and the
#: per-block sum, × 0.30 GiB) + the FFN intermediate and the GELU output (2 × `[1,S,14336]`
#: 0.23 GiB) + about 6 `[1,S,3072]` (0.30 GiB) ≈ 1.37 GiB, rounded up.
ACTIVATION_BYTES = int(1.5 * 2**30)


def memory_estimate(model: str = MODEL) -> int:
    """Phase 1's RAM estimate (bytes) = the DiT's whole f32 + base + the gap between measurement
    and formula + activations + the VAE's f32 (encoder included). About 23.7 GiB for the 5B
    (`require_available` adds a 1 GiB margin).
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
    """This process's peak RSS (GiB — Linux's `ru_maxrss` is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


@dataclass
class Denoised:
    """One case of phase 1 (handed to phase 2 and to the writer)."""

    case: FixtureCase
    names: PromptNames
    latents_init: torch.Tensor
    noise_cond: list[torch.Tensor]
    noise_uncond: list[torch.Tensor]
    latents: list[torch.Tensor]
    #: The latents of each step of the M = S path (decision cases only — None for acceptance).
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
    """Run one case for two steps on the M = 1 path (`expand_timesteps=False`)."""
    generator = torch.Generator().manual_seed(case.seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)

    # The same values as the context passed to the upstream (`_call` fills them with the same
    # function) — the hook compares against these by value.
    contexts = {
        "cond": pipeline_ref.pad_text_embeds(embeds[names.positive]),
        "uncond": pipeline_ref.pad_text_embeds(embeds[names.negative]),
    }
    forwards: list[tuple[int, str, torch.Tensor]] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], output: Any) -> None:
        # The upstream returns a tuple with `return_dict=False`. The timestep is `t.expand(1)`
        # (one-dimensional int64).
        timestep = kwargs["timestep"]
        if timestep.dim() != 1:
            raise AssertionError(
                f"{case.name}: timestep shape {tuple(timestep.shape)} is not one-dimensional"
                " (expected the M = 1 path)"
            )
        context = kwargs["encoder_hidden_states"]
        labels = [label for label, value in contexts.items() if torch.equal(context, value)]
        if len(labels) != 1:
            raise AssertionError(
                f"{case.name}: the DiT's context matches {len(labels)} of the cond / uncond"
                " embeddings (expected 1) — the split cannot be decided"
            )
        forwards.append((int(timestep[0]), labels[0], output[0].detach().clone()))

    steps: list[torch.Tensor] = []
    handle = pipeline.transformer.register_forward_hook(record_forward, with_kwargs=True)
    started = time.perf_counter()
    try:
        # attention is pinned to the CPU flash path, as in `export_dit`'s reference (leaving no
        # path that silently falls back to MATH — the same as `few_step_ref`).
        with torch.no_grad(), dit_patch.flash_attention_only():
            final = _call(pipeline, embeds, names, latents_init, _step_recorder(steps))
    finally:
        handle.remove()
    seconds = time.perf_counter() - started
    timesteps = [int(value) for value in pipeline.scheduler.timesteps]
    sigmas = [float(value) for value in pipeline.scheduler.sigmas]
    if len(forwards) != 2 * STEPS or len(steps) != STEPS:
        raise AssertionError(
            f"{len(forwards)} forwards, {len(steps)} steps (expected 2 steps with CFG)"
        )
    if [t for t, _, _ in forwards] != [t for t in timesteps for _ in range(2)]:
        raise AssertionError(
            f"the forwards' timesteps {[t for t, _, _ in forwards]} do not match {timesteps}"
        )
    outputs: dict[tuple[int, str], torch.Tensor] = {}
    for index, (_, label, output) in enumerate(forwards):
        outputs[(index // 2, label)] = output
    if sorted(outputs) != [(step, label) for step in range(STEPS) for label in sorted(contexts)]:
        raise AssertionError(
            f"not exactly one cond / uncond per step ({[label for _, label, _ in forwards]})"
        )
    if not torch.equal(final, steps[-1]):
        raise AssertionError("the final latents differ from the record of the last step")
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
    """Run the same initial noise for two steps on the M = S path (`expand_timesteps=True`) and
    return the latents of each step (an observation — not a gate).

    The hook differs from the M = 1 one: here the upstream passes the timestep as `[1,S]` (the same
    value for every token), so it checks the shape and that there is a single value, and whether
    the sequence of times is the same as with M = 1.
    """
    case = denoised.case
    tokens = token_count(pipeline.transformer, LATENT_SHAPE)
    seen: list[int] = []

    def record_forward(_module: Any, _args: Any, kwargs: dict[str, Any], _output: Any) -> None:
        timestep = kwargs["timestep"]
        if tuple(timestep.shape) != (1, tokens):
            raise AssertionError(
                f"{case.name}: the timestep shape of the M = S path {tuple(timestep.shape)} is"
                f" not (1, {tokens})"
            )
        values = timestep.unique()
        if values.numel() != 1:
            raise AssertionError(
                f"{case.name}: the M = S timestep has {values.numel()} values (T2V has 1)"
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
            f"{case.name}: the M = S forwards' timesteps {seen} and {len(steps)} steps do not"
            f" match M = 1's {denoised.timesteps}"
        )
    if not torch.equal(final, steps[-1]):
        raise AssertionError(
            f"{case.name}: the final M = S latents differ from the record of the last step"
        )
    return [step[0] for step in steps]


def denoise_phase(
    load_transformer: Callable[[], nn.Module],
    vae: AutoencoderKLWan,
    scheduler: UniPCMultistepScheduler,
    cases: Sequence[FixtureCase],
    embeds: Mapping[str, torch.Tensor],
    names: Mapping[str, PromptNames],
) -> tuple[list[Denoised], weakref.ref[nn.Module]]:
    """Phase 1: load the DiT and run every case. The reference to the DiT lives only inside this
    function (it is released on return).

    The returned weak reference is for checking that the DiT was collected before phase 2
    ({@link assert_released}).
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
    """Check that the DiT was collected (phase 2's RAM estimate assumes the DiT is gone)."""
    gc.collect()
    if released() is not None:
        raise AssertionError("a reference to the DiT remains — release it before phase 2 (VAE)")


def decode_frames(
    pipeline: WanPipeline, latents: torch.Tensor, plan: vae_tiling.TilePlan, patch_size: int
) -> torch.Tensor:
    """Final latents `[1, z, F, H, W]` → RGB `[3, F', H·16, W·16]` (after unpatchify and clamping).

    The same denormalization as the upstream → step 5's tiled decode (patchify space, before
    clamping) → the upstream unpatchify → `clamp(-1, 1)` (the same order as the end of the upstream
    `_decode`).
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
    """The tensors to write (all f32; fail loudly on any non-finite value)."""
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
            raise AssertionError(f"{denoised.case.name}: {name} is not f32 ({tensor.dtype})")
        if not bool(tensor.isfinite().all()):
            raise AssertionError(f"{denoised.case.name}: {name} has non-finite values")
    return tensors


def case_metadata(
    denoised: Denoised,
    plan: vae_tiling.TilePlan,
    patch_size: int,
    flow_shift: float,
    common: Mapping[str, str],
) -> dict[str, str]:
    """The metadata to write (safetensors metadata is strings only)."""
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
    """The metadata common to every case (the embeddings' sha256, the source pin, the execution
    environment the last bits of f32 depend on)."""
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
    """Run every case in two phases and write them (staging → replace). Returns the summaries.

    MUST: phase 2 starts only after the DiT is released ({@link assert_released}) — the VAE decode
    is not stacked on phase 1's peak.
    """
    patch_size = vae.config.patch_size
    if patch_size is None:
        raise ValueError(
            "the VAE's patch_size is None — pass a Wan2.2 VAE (the patchifying generation)"
        )
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
                f"{case.name}: frames shape {tuple(frames.shape)} is not {expected}"
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
        "--out", type=Path, default=SERIES, help="the series root (default: the i8 DiT series)"
    )
    parser.add_argument(
        "--embeds",
        type=Path,
        default=SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME,
        help="the text embedding asset (the output of wan.text_embeds)",
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
        parser.error(
            f"{args.embeds} does not exist — build it first with `python -m wan.text_embeds`"
        )

    embeds, asset_metadata = text_embeds.read_asset(args.embeds)
    selected = [case for case in FIXTURE_CASES if args.case is None or case.name in args.case]
    # Before the heavy loading, check the asset's rows and roles for every case.
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
