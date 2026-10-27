"""The stage-0 DiT reference of MOSS-Image: upstream `pipe(...)` run on CPU with the DiT in f32,
recorded block by block through forward hooks, and re-assembled from the hooks (plan step 0 item 6
and the step's check).

## The run

- Upstream `MOSSImagePipeline.__call__` itself, at 256² (N = 256) and 256×384 (height × width,
  N = 384 — a non-square size so H and W cannot be swapped unnoticed), 2 steps, CFG 3 with the
  negative branch skipping block 22 (upstream's defaults), the fixed-prompt embeddings from
  `moss.text_embeds` (one positive + the empty negative) and seeded initial latents injected as
  `latents=`. `output_type="pt"`, so the VAE decode runs too.
- The DiT is read in f32 (bf16 → f32 is exact; the arithmetic is f32 — upstream's own bf16 run is
  not a baseline, as for Wan). The VAE is f32 as upstream runs it.

## What the hooks keep (per size, in `dit-<H>x<W>.safetensors`)

- per step i: `step<i>.latents_in`, `.timestep`, `.video_tokens` (the patchified latents — the
  input of `video_embedder`), `.video_embedded`, `.time_sinusoid` (the input of the time MLP),
  `.time_embedding`, `.modulation`, `.head_modulation`, `.prediction` (after CFG), `.latents_out`;
  and per branch b ∈ {positive, negative}: `step<i>.<b>.block<j>` (the output of every block that
  ran, `[1, S, 3072]`) and `step<i>.<b>.sample` (the DiT's output).
- per branch: `<b>.text_context` (the `text_embedder` output — what `dit_text` must produce) and
  `<b>.rope` (the RoPE table of that branch's S); both are the same at every step (checked).
- the decode: `vae.z`, `vae.pre_attention`, `vae.post_attention`, `vae.decoded` (`moss.pipeline_ref.
  VaeRecorder`) and `image` (what `pipe(...)` returned, `[1, 3, H, W]` in [0, 1]).
- the inputs: `latents.init`, `text_embeds.positive`, `text_embeds.negative`, and the scheduler's
  `sigmas` / `timesteps`.

## The check (MUST — the step's gate; the record is not written unless every item holds)

Re-assembled from the hooks, in the same process, bit for bit against upstream. Recording
({@link record_case}) and checking ({@link check_recording}, under `torch.no_grad()`) are separate
steps, so the gate itself is tested on a tiny pipeline with perturbed recordings
(`moss/tests/test_reference.py`).

1. Each step has one positive and one negative forward at the same timestep and latents, the
   branch told apart by the text embedding it received (not by call order); the positive ran all
   42 blocks and the negative all but block 22.
2. The chain (structural): block 0's recorded input is `cat(video_embedded, text_context)`, and
   every block's recorded input is the recorded output of the block that ran before it (block 23
   takes block 21's in the negative). This shows the records are wired in upstream's order; it does
   not recompute any block — the block arithmetic is verified at step 1, against these records.
3. The head: the last block's output through upstream's head modules gives the recorded `sample`.
4. The trajectory: CFG and the Euler update written here (the host functions of plan step 4), fed
   the recorded samples and the scheduler's sigmas, give every step's input latents and finally the
   decode input `vae.z` (through `moss.pipeline_ref.denormalize` / `unpatchify`).
5. The image: `vae.decode` of the re-assembled `z`, clamped and post-processed as upstream does,
   equals the image `pipe(...)` returned.
6. The host functions: the recorded `rope` and `time_sinusoid` equal `moss.host_fixture`'s functions
   (a fresh `_RoPE`, the tiny transformer), which tie that fixture to the loaded DiT.

    uv run --group moss --inexact python -m moss.reference                 # both sizes
    uv run --group moss --inexact python -m moss.reference --size 256x256  # one

Output: `dit-<H>x<W>.safetensors` and `reference.json` (the run summary: the gate, time, peak RAM)
in the stage-0 reference directory (`moss.records.default_reference_dir`). The record's metadata
names its input by sha256 and repository-relative path only ({@link repository_name}); absolute
paths go to `reference.json`.

MUST: diffusers is imported inside the functions (`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from _shared.paths import REPO_ROOT, SERIES_ROOT
from moss import host_fixture, pipeline_ref, records, sources, text_embeds
from moss.prompts import NEGATIVE, POSITIVE, prompt_by_name
from moss.records import bits_equal

#: `(height, width)` in pixels.
SIZES: tuple[tuple[int, int], ...] = ((256, 256), (256, 384))

#: Requested steps (both survive the snapping: σ = 1 and one value near 0.024).
STEPS = 2

#: The seed of the injected initial latents.
SEED = 20261027

#: The positive prompt of the references (the model card's example).
PROMPT = "fridge-note"

#: The single metadata key of the records.
METADATA_KEY = "karume.moss.dit_reference"

#: The embedding asset the references read.
TEXT_EMBEDS = SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME


class DitReferenceError(RuntimeError):
    """The hook records do not re-assemble into upstream's result."""


@dataclass
class Forward:
    """One DiT forward as the hooks saw it."""

    branch: str
    skip_layers: frozenset[int]
    latents_in: torch.Tensor
    timestep: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    #: The block indices that ran, in order.
    ran: list[int] = field(default_factory=list)
    #: The input each block that ran received (for the chain check; not written to the record).
    inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    sample: torch.Tensor | None = None


class DitRecorder:
    """Forward hooks on the DiT that keep one {@link Forward} per call."""

    def __init__(self, transformer: Any, positive: torch.Tensor, negative: torch.Tensor) -> None:
        self.forwards: list[Forward] = []
        self._embeds = {POSITIVE: positive, NEGATIVE: negative}
        keep = self._keep_output
        modules: list[tuple[Any, str]] = [
            (transformer.video_embedder, "video_embedded"),
            (transformer.text_embedder, "text_context"),
            (transformer.rope, "rope"),
            (transformer.time_embedding, "time_embedding"),
            (transformer.shared_block_modulation, "modulation"),
            (transformer.video_head_modulation, "head_modulation"),
        ]
        self._handles = [
            transformer.register_forward_pre_hook(self._begin, with_kwargs=True),
            transformer.register_forward_hook(self._end),
            transformer.video_embedder.register_forward_pre_hook(
                lambda _m, args: self._keep("video_tokens", args[0])
            ),
            transformer.time_embedding[0].register_forward_pre_hook(
                lambda _m, args: self._keep("time_sinusoid", args[0])
            ),
            *(module.register_forward_hook(keep(name)) for module, name in modules),
            *(
                layer.register_forward_hook(self._block(index))
                for index, layer in enumerate(transformer.layers)
            ),
        ]

    @property
    def _current(self) -> Forward:
        if not self.forwards or self.forwards[-1].sample is not None:
            raise DitReferenceError("a DiT submodule ran outside a recorded forward")
        return self.forwards[-1]

    def _begin(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        latents, timestep, context = args[0], args[1], args[2]
        matches = [b for b, e in self._embeds.items() if bits_equal(context[0], e)]
        if len(matches) != 1:
            raise DitReferenceError(f"the text context matches {matches} (one branch expected)")
        skip = kwargs.get("skip_layers")
        self.forwards.append(
            Forward(
                branch=matches[0],
                skip_layers=frozenset(() if skip is None else skip),
                latents_in=latents.detach().clone(),
                timestep=torch.as_tensor(timestep).detach().clone(),
            )
        )

    def _end(self, _module: Any, _args: Any, output: Any) -> None:
        self._current.sample = output.sample.detach().clone()

    def _keep(self, name: str, tensor: torch.Tensor) -> None:
        current = self._current
        if name in current.tensors:
            raise DitReferenceError(f"{name} ran twice in one forward")
        current.tensors[name] = tensor.detach().clone()

    def _keep_output(self, name: str) -> Any:
        return lambda _module, _args, output: self._keep(name, output)

    def _block(self, index: int) -> Any:
        def hook(_module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
            current = self._current
            if index in current.inputs:
                raise DitReferenceError(f"block {index} ran twice in one forward")
            current.inputs[index] = args[0].detach().clone()
            current.ran.append(index)
            self._keep(f"block{index}", output)

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


@dataclass
class Recording:
    """One `pipe(...)` call: its inputs, the scheduler's grid and what the hooks kept."""

    height: int
    width: int
    latents_init: torch.Tensor
    #: The positive and negative `[T, text_dim]` embeddings the call received.
    embeds: dict[str, torch.Tensor]
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    forwards: list[Forward]
    #: What `moss.pipeline_ref.VaeRecorder` kept (`vae.z`, …, `vae.decoded`).
    vae: dict[str, torch.Tensor]
    #: What `pipe(...)` returned (`output_type="pt"`).
    image: torch.Tensor


# Third-party code notice. `head` and `cfg` below are adapted from `pipeline_moss_image.py` of
# OpenMOSS-Team/MOSS-Image-SFT (revision pinned in `moss.sources`), licensed under the Apache
# License, Version 2.0: `head` is the four statements after the block loop of
# `MOSSImageTransformer.forward`, with upstream's modules and in upstream's order (batch of one);
# `cfg` is the CFG line of `MOSSImagePipeline.__call__`. `euler` is the update line of diffusers'
# `FlowMatchEulerDiscreteScheduler.step` (diffusers 0.39.0, Apache License, Version 2.0).


def head(
    transformer: Any, last: torch.Tensor, head_modulation: torch.Tensor, h: int, w: int
) -> torch.Tensor:
    """The DiT's output from the last block's output (`[1, S, 3072]` → `[1, 128, h, w]`)."""
    modulation = head_modulation.view(1, 2, -1)
    x = transformer.final_norm_video(last[:, : h * w].to(transformer.final_norm_video.weight.dtype))
    x = x * (1 + modulation[0:1, 1, None]) + modulation[0:1, 0, None]
    x = transformer.final_linear_video(x.to(transformer.final_linear_video.weight.dtype))
    return x.transpose(1, 2).reshape(1, -1, h, w)


def cfg(positive: torch.Tensor, negative: torch.Tensor, scale: float) -> torch.Tensor:
    """The CFG combination in f32, in upstream's form (`uncond + scale · (cond − uncond)`)."""
    return negative + scale * (positive - negative)


def euler(
    latents: torch.Tensor, prediction: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor
) -> torch.Tensor:
    """One flow-matching Euler step as `FlowMatchEulerDiscreteScheduler.step` computes it
    (`sample + (σ_next − σ) · v` in f32, cast back to the prediction's dtype)."""
    return (latents.to(torch.float32) + (sigma_next - sigma) * prediction).to(prediction.dtype)


def check(condition: bool, failures: list[str], message: str) -> None:
    """Collect a failed gate item (all items are evaluated before failing)."""
    if not condition:
        failures.append(message)


def record_case(
    pipe: Any,
    embeds: dict[str, torch.Tensor],
    latents_init: torch.Tensor,
    *,
    height: int,
    width: int,
    steps: int = STEPS,
    skip_layers: Sequence[int] = pipeline_ref.UNCOND_SKIP_LAYERS,
    guidance_scale: float = pipeline_ref.GUIDANCE_SCALE,
) -> Recording:
    """Run upstream `pipe(...)` once with the hooks (the recording half of the gate)."""
    recorder = DitRecorder(pipe.transformer, embeds[POSITIVE], embeds[NEGATIVE])
    vae_recorder = pipeline_ref.VaeRecorder(pipe.vae)
    try:
        image = pipe(
            prompt_embeds=embeds[POSITIVE][None],
            negative_prompt_embeds=embeds[NEGATIVE][None],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            uncond_skip_layers=tuple(skip_layers),
            latents=latents_init.clone(),
            output_type="pt",
        ).images
    finally:
        recorder.close()
        vae_recorder.close()
    return Recording(
        height=height,
        width=width,
        latents_init=latents_init,
        embeds=dict(embeds),
        sigmas=pipe.scheduler.sigmas.clone(),
        timesteps=pipe.scheduler.timesteps.clone(),
        forwards=recorder.forwards,
        vae=vae_recorder.tensors,
        image=image,
    )


def gate_items(layers: int, skip_layers: Sequence[int]) -> list[str]:
    """What the gate checks, in the words the run summary and the record carry."""
    return [
        f"branches by text context; positive ran all {layers} blocks,"
        f" negative skipped {sorted(skip_layers)}",
        "chain (structural): every block's recorded input == the recorded output of the block that"
        " ran before it (block 0: cat(video_embedded, text_context)); the block arithmetic is not"
        " recomputed here - it is verified at step 1",
        "head: last block output -> upstream head modules == recorded sample",
        "trajectory: CFG + Euler re-assembled == every step's latents in and the decode input",
        "image: decode(z) -> clamp -> postprocess == pipe(...) images",
        "host functions: rope == fresh _RoPE, time sinusoid == tiny transformer",
    ]


@torch.no_grad()
def check_recording(
    pipe: Any,
    module: ModuleType,
    recording: Recording,
    *,
    skip_layers: Sequence[int] = pipeline_ref.UNCOND_SKIP_LAYERS,
    guidance_scale: float = pipeline_ref.GUIDANCE_SCALE,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Re-assemble `recording` with upstream's modules (the checking half of the gate, without
    autograd); return the record's tensors and the gate summary.

    Every item is evaluated; any failure raises {@link DitReferenceError} listing all of them.
    """
    transformer, vae = pipe.transformer, pipe.vae
    height, width = recording.height, recording.width
    h, w = height // pipeline_ref.PIXELS_PER_CELL, width // pipeline_ref.PIXELS_PER_CELL
    layers = len(transformer.layers)
    embeds, forwards = recording.embeds, recording.forwards
    sigmas, timesteps = recording.sigmas, recording.timesteps
    failures: list[str] = []
    steps = int(timesteps.shape[0])
    check(len(forwards) == 2 * steps, failures, f"{len(forwards)} forwards for {steps} steps")
    tensors: dict[str, torch.Tensor] = {
        "latents.init": recording.latents_init,
        "text_embeds.positive": embeds[POSITIVE],
        "text_embeds.negative": embeds[NEGATIVE],
        "sigmas": sigmas,
        "timesteps": timesteps,
    }
    config = transformer.config
    tiny = host_fixture.tiny_transformer(module, int(config["freq_dim"]))
    skipped = frozenset(skip_layers)
    latents = recording.latents_init
    for step in range(steps):
        pair = {f.branch: f for f in forwards[2 * step : 2 * step + 2]}
        if set(pair) != {POSITIVE, NEGATIVE}:
            failures.append(f"step {step}: branches {sorted(pair)}")
            break
        positive, negative = pair[POSITIVE], pair[NEGATIVE]
        for branch, forward, skip in (
            (POSITIVE, positive, frozenset[int]()),
            (NEGATIVE, negative, skipped),
        ):
            where = f"step {step} {branch}"
            check(
                forward.skip_layers == skip,
                failures,
                f"{where}: skip_layers {sorted(forward.skip_layers)}",
            )
            expected_ran = [i for i in range(layers) if i not in skip]
            check(forward.ran == expected_ran, failures, f"{where}: ran {forward.ran}")
            if not forward.ran or forward.sample is None:
                failures.append(f"{where}: no block ran or no sample was returned")
                continue
            previous = torch.cat(
                (forward.tensors["video_embedded"], forward.tensors["text_context"]), 1
            )
            broken = []
            for j in forward.ran:
                if not bits_equal(forward.inputs[j], previous):
                    broken.append(j)
                previous = forward.tensors[f"block{j}"]
            check(not broken, failures, f"{where}: chain broken at blocks {broken}")
            check(bits_equal(forward.timestep, timesteps[step]), failures, f"{where}: timestep")
            check(bits_equal(forward.latents_in, latents), failures, f"{where}: latents in")
            last = forward.tensors[f"block{forward.ran[-1]}"]
            rebuilt = head(transformer, last, forward.tensors["head_modulation"], h, w)
            check(bits_equal(rebuilt, forward.sample), failures, f"{where}: head")
            for j in forward.ran:
                tensors[f"step{step}.{branch}.block{j}"] = forward.tensors[f"block{j}"]
            tensors[f"step{step}.{branch}.sample"] = forward.sample
            for name in ("text_context", "rope"):
                key = f"{branch}.{name}"
                if key in tensors:
                    check(
                        bits_equal(tensors[key], forward.tensors[name]),
                        failures,
                        f"{where}: {name} moved",
                    )
                else:
                    tensors[key] = forward.tensors[name]
        shared = (
            "video_tokens",
            "video_embedded",
            "time_sinusoid",
            "time_embedding",
            "modulation",
            "head_modulation",
        )
        for name in shared:
            same = bits_equal(positive.tensors[name], negative.tensors[name])
            check(same, failures, f"step {step}: {name} differs between the branches")
            tensors[f"step{step}.{name}"] = positive.tensors[name]
        sinusoid = host_fixture.time_sinusoid(tiny, timesteps[step])
        check(
            bits_equal(sinusoid, positive.tensors["time_sinusoid"]),
            failures,
            f"step {step}: sinusoid ≠ host_fixture",
        )
        if positive.sample is None or negative.sample is None:
            break
        prediction = cfg(positive.sample.float(), negative.sample.float(), guidance_scale)
        tensors[f"step{step}.latents_in"] = latents
        tensors[f"step{step}.timestep"] = timesteps[step].reshape(1)
        tensors[f"step{step}.prediction"] = prediction
        latents = euler(latents, prediction, sigmas[step], sigmas[step + 1])
        tensors[f"step{step}.latents_out"] = latents
    for branch in (POSITIVE, NEGATIVE):
        key = f"{branch}.rope"
        if key not in tensors:
            failures.append(f"{branch}: no rope was recorded")
            continue
        length = int(embeds[branch].shape[0])
        table = host_fixture.rope_table(module, config, h, w, length)
        check(bits_equal(table, tensors[key]), failures, f"{branch}: rope ≠ host_fixture")

    z = pipeline_ref.unpatchify(pipeline_ref.denormalize(vae, latents))
    recorded = recording.vae
    check(bits_equal(z, recorded["vae.z"]), failures, "the re-assembled decode input ≠ upstream's")
    decoded = vae.decode(z, return_dict=False)[0]
    check(bits_equal(decoded, recorded["vae.decoded"]), failures, "decode ≠ upstream's")
    rebuilt_image = pipe.image_processor.postprocess(decoded.clamp(-1, 1), output_type="pt")
    check(
        bits_equal(rebuilt_image, recording.image), failures, "the re-assembled image ≠ pipe(...)"
    )
    if failures:
        raise DitReferenceError(f"{height}x{width}: the gate failed: {failures}")
    tensors.update(recorded)
    tensors["image"] = recording.image
    gate = {
        "steps": steps,
        "forwards": len(forwards),
        "blocks_recorded": sum(len(f.ran) for f in forwards),
        "items": gate_items(layers, sorted(skipped)),
        "passed": True,
    }
    return tensors, gate


def run_case(
    pipe: Any,
    module: ModuleType,
    embeds: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run `pipe(...)` once with the hooks, re-assemble, and return the record and the gate."""
    h, w = height // pipeline_ref.PIXELS_PER_CELL, width // pipeline_ref.PIXELS_PER_CELL
    clock = records.Clock()
    latents_init = pipeline_ref.seeded_latents(SEED, height, width)
    recording = record_case(pipe, embeds, latents_init, height=height, width=width)
    pipe_seconds = clock.lap(f"pipe {height}x{width}")
    tensors, gate = check_recording(pipe, module, recording)
    check_seconds = clock.lap(f"re-assemble {height}x{width}")
    summary = {
        "size": [height, width],
        "latent": [h, w],
        "N": h * w,
        "S": {
            POSITIVE: h * w + int(embeds[POSITIVE].shape[0]),
            NEGATIVE: h * w + int(embeds[NEGATIVE].shape[0]),
        },
        "gate": gate,
        "seconds": {"pipe": pipe_seconds, "re_assemble": check_seconds},
    }
    return tensors, summary


def repository_name(path: Path) -> str:
    """`path` relative to the repository root (as `/`-separated text), or its file name when it lies
    outside the repository — a record names its inputs this way and identifies them by sha256, so
    the record's bytes do not depend on where the checkout lives."""
    try:
        return path.absolute().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def load_embeds(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    """The positive ({@link PROMPT}) and negative embeddings, the asset metadata and its sha256."""
    tensors, metadata = text_embeds.read_asset(path)
    negative = [e["name"] for e in metadata["prompts"] if e["role"] == NEGATIVE]
    if prompt_by_name(PROMPT).role != POSITIVE or len(negative) != 1:
        raise DitReferenceError(f"the asset does not hold {PROMPT!r} and one negative")
    return (
        {POSITIVE: tensors[PROMPT], NEGATIVE: tensors[negative[0]]},
        metadata,
        sources.sha256_of(path),
    )


def generate(out: Path, sizes: Sequence[tuple[int, int]], embeds_path: Path) -> dict[str, Any]:
    """Load once, run every size, write one record per size; return the run summary."""
    clock = records.Clock()
    snapshot, module = pipeline_ref.upstream()
    embeds, embeds_metadata, embeds_sha256 = load_embeds(embeds_path)
    transformer = pipeline_ref.load_transformer(snapshot, module, torch.float32)
    clock.lap("load transformer (f32)")
    vae = pipeline_ref.load_vae(snapshot)
    scheduler = pipeline_ref.load_scheduler(snapshot, module)
    pipe = pipeline_ref.build_pipeline(
        module, transformer=transformer, scheduler=scheduler, vae=vae
    )
    clock.lap("load vae and scheduler")
    cases = []
    for height, width in sizes:
        tensors, summary = run_case(pipe, module, embeds, height=height, width=width)
        clock.lap(f"case {height}x{width}")
        # The timings stay out of the record (a rerun that computes the same values writes the
        # same bytes); they are in the run summary.
        facts = {key: value for key, value in summary.items() if key != "seconds"}
        metadata = {
            "source": {
                "repo": sources.SOURCE.repo,
                "revision": sources.SOURCE.revision,
                "pipeline_code_sha256": sources.PIPELINE_CODE_SHA256,
            },
            "text_embeds": {
                "path": repository_name(embeds_path),
                "sha256": embeds_sha256,
                "positive": PROMPT,
                "tokens": {
                    POSITIVE: int(embeds[POSITIVE].shape[0]),
                    NEGATIVE: int(embeds[NEGATIVE].shape[0]),
                },
                "versions": embeds_metadata["run"]["versions"],
            },
            "dit_dtype": "float32",
            "vae_dtype": "float32",
            "seed": SEED,
            "noise": "torch.randn(generator=torch.Generator().manual_seed(seed), dtype=float32)",
            "guidance_scale": pipeline_ref.GUIDANCE_SCALE,
            "uncond_skip_layers": list(pipeline_ref.UNCOND_SKIP_LAYERS),
            "requested_steps": STEPS,
            "versions": records.versions(),
            "torch_threads": torch.get_num_threads(),
            **facts,
        }
        path = out / f"dit-{height}x{width}.safetensors"
        records.write_record(path, tensors, metadata, key=METADATA_KEY)
        write_seconds = clock.lap(f"write {path.name}")
        cases.append(
            {
                **summary,
                "path": str(path),
                "bytes": path.stat().st_size,
                "write_seconds": write_seconds,
            }
        )
        del tensors
    return {
        "cases": cases,
        "environment": records.environment(),
        "versions": records.versions(),
        "seconds": {**clock.phases, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
    }


def parse_size(text: str) -> tuple[int, int]:
    """`<height>x<width>` in pixels."""
    height, width = (int(part) for part in text.split("x"))
    return height, width


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--size", type=parse_size, action="append", help="<height>x<width>")
    parser.add_argument("--out", type=Path, default=None, help="default: the stage-0 reference dir")
    parser.add_argument("--text-embeds", type=Path, default=TEXT_EMBEDS)
    args = parser.parse_args(argv)
    out = args.out or records.default_reference_dir()
    summary = generate(out, args.size or SIZES, args.text_embeds)
    text = json.dumps(summary, indent=1)
    print(text)
    (out / "reference.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
