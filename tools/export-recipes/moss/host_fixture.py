"""Fixed values of the MOSS-Image host functions, taken from upstream's own code (stage 0 and
decision 11 of ADR NNNN).

The TS host functions (stages 2 and 4) must reproduce these as Uint32-identical values; nothing
here transcribes upstream's formulas — each value is produced by upstream code:

- `rope.<H>x<W>.t<T>` `[1, H·W + T, 128]`: `_RoPE.forward(H, W, T)` of a fresh `_RoPE(head_dim,
  rope_base)` from the pinned pipeline code (the loaded DiT's `rope` is the same function — it
  rebuilds its bands on load; `moss.reference` checks the two against each other). The cases cover
  odd and even T, H and W, a non-square size whose transpose is another case's size, the minimal
  1×1 with T = 1 and 1024² with the empty negative (S = 4,106).
- `training.sigmas` / `training.timesteps` `[1000]`: the 1000-point training grid of
  `MOSSImageScheduler.set_timesteps`, in its descending order. It is the `training` tensor upstream
  snaps to; it is read out by running upstream `set_timesteps(1000)` on a scheduler whose `shift`
  equals its `training_shift`, where the requested grid is the training grid itself (each point
  snaps to itself). Every requested sequence below is checked to be made of members of this table.
- `steps.<n>.sigmas` `[k + 1]` (with the terminal 0) / `steps.<n>.timesteps` `[k]` /
  `steps.<n>.grid` `[k]` (the indices into the training grid): upstream `set_timesteps(n)` with the
  pinned scheduler config, for n = 1–200, 500 and 1000. `k` (the steps that actually run) can be
  smaller than `n` — duplicates after snapping are dropped.
- `steps.<n>.requested` `[n]` f32 (the requested grid before snapping — upstream's
  `grid(n, shift)` of `torch.linspace`) / `steps.<n>.nearest` `[n]` int32 (the training-grid index
  each requested point snapped to, before the duplicates are dropped): taken during that same
  `set_timesteps(n)` call from upstream's own `torch.searchsorted` argument and `torch.where`
  result ({@link step_sequence} wraps both for the call; only calls made directly by
  `MOSSImageScheduler.set_timesteps` are kept). The TS port of `torch.linspace` and of the grid
  formula is checked against `requested` at Uint32 and its snapping against `nearest` (stage
  4); the snapped values alone would hide a 1-ULP slip anywhere but near a midpoint.
- `shift.<i>.steps.<n>.{requested,nearest,grid,sigmas,timesteps}`: the same five, from upstream
  `set_timesteps(n)` on the pinned config with `shift` replaced by `SHIFTS[i]` (the metadata's
  `shift_cases`), for the counts {@link SHIFT_STEP_COUNTS} (two of them above the 1,000 training
  points). The shift enters the requested grid three times (`lower`, the f32 scale and `shift − 1`),
  and the pinned config holds only one value of it.
- `cfg.cond` / `cfg.uncond` `[1, C, H, W]` and `cfg.<i>.combined`: one step of upstream
  `MOSSImagePipeline.__call__` at `guidance_scale = CFG_SCALES[i]` (the metadata's `cfg`) on a tiny
  DiT — the positive sample, the negative sample (one block skipped, so the two differ) and the
  prediction upstream hands to `scheduler.step`, i.e. its own CFG line `uncond + guidance_scale ·
  (prediction − uncond)`. The scales are mostly not representable in f32, so the record fixes how
  the Python float meets the f32 tensors. The samples are the same at every scale (checked).
- `time_sinusoid` `[1000, 256]`: row i is what `MOSSImageTransformer.forward` feeds its time MLP
  for the timestep `training.timesteps[i]` (the float64 sinusoid cast to f32), taken with a forward
  pre-hook on `time_embedding[0]` of a tiny `MOSSImageTransformer` (only `freq_dim` reaches that
  value; the timestep is passed as the pipeline passes it, a 0-d f32 tensor, one call per timestep).

Every value is computed twice, with torch at one thread and at the default thread count, and the
two runs must agree bit for bit before anything is written; the record names the torch version,
both thread counts, torch's CPU capability (the ISA its kernels were dispatched to) and the
MKL / OpenMP environment.

What the thread comparison reaches: ATen splits an elementwise kernel across threads only from its
grain size up (`at::internal::GRAIN_SIZE` = 32,768 elements in torch 2.13's `TensorIterator.h` — an
implementation fact, not a guarantee). The RoPE phase tensor has S · 64 elements, so only the
64×64 case (S = 4,106: 262,784 elements per `sin` / `cos`) is computed on several threads; every
other RoPE case (S ≤ 395), the 1000-point grids, the snapping and the 256-wide sinusoid rows are
below the grain size and run on one thread either way, so for them the comparison shows that the
thread count does not change the code path rather than that a parallel path agrees.

    uv run --group moss --inexact python -m moss.host_fixture

Output: `host.safetensors` and `host.json` (the run summary) in the stage-0 reference directory
(`moss.records.default_reference_dir`).

MUST: diffusers is imported inside the functions (`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from moss import pipeline_ref, records, sources

#: `(height, width, text_length)` in latent cells and tokens.
ROPE_CASES: tuple[tuple[int, int, int], ...] = (
    (16, 16, 10),
    (16, 24, 11),
    (24, 16, 10),
    (15, 17, 12),
    (17, 14, 13),
    (1, 1, 1),
    (64, 64, 10),
)

#: The requested step counts whose sequences are recorded.
STEP_COUNTS: tuple[int, ...] = (*range(1, 201), 500, 1000)

#: Shifts other than the pinned config's whose sequences are recorded too — around 1 (where
#: `shift − 1` changes sign), small, large and not representable in f32.
SHIFTS: tuple[float, ...] = (0.1, 0.35, 0.9999, 1.0, 1.0001, 2.2, 3.0, 3.7, 7.25, 12.0, 50.0)

#: The step counts recorded for each of {@link SHIFTS}.
SHIFT_STEP_COUNTS: tuple[int, ...] = (2, 7, 30, 50, 77, 199, 1024, 5000)

#: The guidance scales of the CFG record (upstream accepts any finite scale ≥ 0; 1 runs no CFG).
CFG_SCALES: tuple[float, ...] = (0.0, 0.1, 0.35, 1.7, 2.9, 3.0, 3.3, 4.0001, 7.5)

#: The CFG record's tiny DiT depth, the negative's skipped block, the latent grid (cells per side)
#: and the seed of its inputs.
CFG_LAYERS = 4
CFG_SKIP_LAYERS: tuple[int, ...] = (2,)
CFG_CELLS = 64
CFG_SEED = 3

#: The single metadata key of `host.safetensors`.
METADATA_KEY = "karume.moss.host_fixture"

#: The record's file name.
RECORD_NAME = "host.safetensors"


class HostFixtureError(ValueError):
    """A fixed value broke an invariant it must hold."""


def transformer_config(snapshot: Path) -> dict[str, Any]:
    """`transformer/config.json` of the snapshot."""
    return json.loads((snapshot / "transformer" / "config.json").read_text(encoding="utf-8"))


def scheduler_config(snapshot: Path, module: ModuleType) -> dict[str, Any]:
    """The pinned scheduler config as `MOSSImageScheduler` reads it."""
    return dict(module.MOSSImageScheduler.load_config(snapshot / "scheduler"))


def rope_table(
    module: ModuleType, config: Mapping[str, Any], height: int, width: int, text_length: int
) -> torch.Tensor:
    """`_RoPE(head_dim, rope_base).forward(height, width, text_length)` (`[1, S, 128]`)."""
    rope = module._RoPE(config["dim"] // config["num_heads"], config["rope_base"])
    return rope(height, width, text_length, torch.device("cpu"))


def tiny_transformer(module: ModuleType, freq_dim: int, *, num_layers: int = 1) -> Any:
    """A `MOSSImageTransformer` that keeps `freq_dim` and shrinks everything else (the sinusoid of
    `forward` depends on `freq_dim` and the timestep only). With `num_layers` > 1 a skip index can
    apply to it (the gate test of `moss.reference` runs a whole pipeline on it)."""
    torch.manual_seed(0)
    return module.MOSSImageTransformer(
        dim=8,
        ffn_dim=8,
        num_heads=1,
        num_layers=num_layers,
        text_dim=8,
        freq_dim=freq_dim,
        latent_channels=4,
        rope_base=10000.0,
    ).eval()


def time_sinusoid(model: Any, timestep: torch.Tensor) -> torch.Tensor:
    """The input of `time_embedding[0]` when `model` runs at `timestep` (`[1, freq_dim]` f32)."""
    seen: list[torch.Tensor] = []
    hook = model.time_embedding[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    try:
        channels = model.config.latent_channels
        with torch.no_grad():
            model(
                torch.zeros(1, channels, 1, 1),
                timestep,
                torch.zeros(1, 1, model.config.text_dim),
            )
    finally:
        hook.remove()
    if len(seen) != 1 or seen[0].dtype != torch.float32:
        raise HostFixtureError("the time MLP did not see one f32 input")
    return seen[0]


def training_grid(
    module: ModuleType, config: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The training grid (sigmas and timesteps, descending) through upstream `set_timesteps`."""
    total = int(config["num_train_timesteps"])
    shift = float(config["training_shift"])
    scheduler = module.MOSSImageScheduler(
        num_train_timesteps=total, shift=shift, training_shift=shift
    )
    scheduler.set_timesteps(total)
    sigmas, timesteps = scheduler.sigmas[:-1].clone(), scheduler.timesteps.clone()
    if sigmas.shape != (total,) or not bool((sigmas[1:] < sigmas[:-1]).all()):
        raise HostFixtureError(f"the training grid is not {total} strictly decreasing values")
    if float(sigmas[0]) != 1.0 or float(scheduler.sigmas[-1]) != 0.0:
        raise HostFixtureError("the training grid does not start at 1 or end with the terminal 0")
    return sigmas, timesteps


@dataclass(frozen=True)
class StepSequence:
    """What one upstream `set_timesteps(steps)` call produced and the two values it went through."""

    #: `[k + 1]` with the terminal 0 (after snapping and dropping duplicates).
    sigmas: torch.Tensor
    #: `[k]`.
    timesteps: torch.Tensor
    #: `[steps]` f32: the requested grid before snapping (the second argument of `searchsorted`).
    requested: torch.Tensor
    #: `[total]` f32 ascending: the training grid upstream snapped to (the first argument).
    training_ascending: torch.Tensor
    #: `[steps]` int64: the ascending-grid index each requested point snapped to (`where`'s result).
    nearest_ascending: torch.Tensor


def step_sequence(module: ModuleType, scheduler: Any, steps: int) -> StepSequence:
    """Upstream `set_timesteps(steps)`, with its `torch.searchsorted` and `torch.where` wrapped for
    the call so their inputs and output are read, not re-derived.

    Only the calls whose caller is `MOSSImageScheduler.set_timesteps` itself are kept (diffusers'
    `set_timesteps`, which it calls next, is another frame); each must happen exactly once.
    """
    code = module.MOSSImageScheduler.set_timesteps.__code__
    seen: dict[str, list[tuple[tuple[Any, ...], torch.Tensor]]] = {"searchsorted": [], "where": []}
    originals: dict[str, Callable[..., torch.Tensor]] = {
        name: getattr(torch, name) for name in seen
    }

    def wrap(name: str) -> Callable[..., torch.Tensor]:
        original = originals[name]

        def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
            result = original(*args, **kwargs)
            if sys._getframe(1).f_code is code:
                seen[name].append((args, result))
            return result

        return wrapper

    for name in seen:
        setattr(torch, name, wrap(name))
    try:
        scheduler.set_timesteps(steps)
    finally:
        for name, original in originals.items():
            setattr(torch, name, original)
    calls = {name: len(found) for name, found in seen.items()}
    if calls != {"searchsorted": 1, "where": 1}:
        raise HostFixtureError(f"{steps} steps: set_timesteps made the calls {calls} (one each)")
    (training, requested), _upper = seen["searchsorted"][0]
    _where_args, nearest = seen["where"][0]
    if requested.shape != (steps,) or requested.dtype != torch.float32:
        raise HostFixtureError(
            f"{steps} steps: requested {requested.dtype} {tuple(requested.shape)}"
        )
    if nearest.shape != (steps,) or nearest.dtype != torch.int64:
        raise HostFixtureError(f"{steps} steps: nearest {nearest.dtype} {tuple(nearest.shape)}")
    return StepSequence(
        sigmas=scheduler.sigmas.clone(),
        timesteps=scheduler.timesteps.clone(),
        requested=requested.clone(),
        training_ascending=training.clone(),
        nearest_ascending=nearest.clone(),
    )


def shifted_scheduler(snapshot: Path, module: ModuleType, shift: float) -> Any:
    """`MOSSImageScheduler` of the pinned config with `shift` replaced (everything else kept)."""
    scheduler = module.MOSSImageScheduler.from_config(
        scheduler_config(snapshot, module), shift=shift
    )
    if float(scheduler.config.shift) != shift:
        raise HostFixtureError(f"the scheduler took shift {scheduler.config.shift}, not {shift}")
    return scheduler


def guidance_step(
    pipe: Any, scale: float, inputs: Mapping[str, torch.Tensor]
) -> list[torch.Tensor]:
    """One step of upstream `pipe(...)` at guidance `scale`: `[positive sample, negative sample,
    the prediction handed to scheduler.step]` (module docstring). The samples are read with a
    forward hook on the transformer, the prediction from the argument of `scheduler.step` (an
    instance attribute shadowing the method for the call); the call must make exactly two forwards
    and one step."""
    scheduler, transformer = pipe.scheduler, pipe.transformer
    samples: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    step = scheduler.step

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        predictions.append(args[0].detach().clone())
        return step(*args, **kwargs)

    hook = transformer.register_forward_hook(
        lambda _module, _args, output: samples.append(output.sample.detach().clone())
    )
    scheduler.step = recording_step
    try:
        with torch.no_grad():
            pipe(
                prompt_embeds=inputs["positive"],
                negative_prompt_embeds=inputs["negative"],
                height=CFG_CELLS * 16,
                width=CFG_CELLS * 16,
                num_inference_steps=1,
                guidance_scale=scale,
                uncond_skip_layers=CFG_SKIP_LAYERS,
                latents=inputs["latents"].clone(),
                output_type="latent",
            )
    finally:
        hook.remove()
        del scheduler.step
    if len(samples) != 2 or len(predictions) != 1:
        raise HostFixtureError(
            f"guidance {scale}: {len(samples)} forwards and {len(predictions)} steps (2 and 1)"
        )
    found = [*samples, *predictions]
    if any(tensor.dtype != torch.float32 for tensor in found):
        raise HostFixtureError(f"guidance {scale}: a sample or the prediction is not f32")
    return found


def guidance_samples(module: ModuleType, config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """{@link guidance_step} per scale of {@link CFG_SCALES} on a tiny DiT with fixed inputs
    (module docstring): `cfg.cond`, `cfg.uncond` and each scale's `cfg.<i>.combined`. Every call
    must give the same two samples, and the skipped block must change the negative one."""
    transformer = tiny_transformer(module, 256, num_layers=CFG_LAYERS)
    scheduler = module.MOSSImageScheduler.from_config(config)
    pipe = pipeline_ref.build_pipeline(module, transformer=transformer, scheduler=scheduler)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator().manual_seed(CFG_SEED)
    text_dim = transformer.config.text_dim
    channels = transformer.config.latent_channels
    inputs = {
        "positive": torch.randn(1, 3, text_dim, generator=generator),
        "negative": torch.randn(1, 2, text_dim, generator=generator),
        "latents": torch.randn(1, channels, CFG_CELLS, CFG_CELLS, generator=generator),
    }
    tensors: dict[str, torch.Tensor] = {}
    for index, scale in enumerate(CFG_SCALES):
        cond, uncond, combined = guidance_step(pipe, scale, inputs)
        if index == 0:
            if records.bits_equal(cond, uncond):
                raise HostFixtureError("the skipped block did not change the negative sample")
            tensors["cfg.cond"], tensors["cfg.uncond"] = cond, uncond
        elif not (
            records.bits_equal(cond, tensors["cfg.cond"])
            and records.bits_equal(uncond, tensors["cfg.uncond"])
        ):
            raise HostFixtureError(f"guidance {scale}: the samples moved between the calls")
        tensors[f"cfg.{index}.combined"] = combined
    return tensors


def sequence_tensors(
    steps: int, sequence: StepSequence, sigmas: torch.Tensor, timesteps: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The five recorded tensors of one `set_timesteps(steps)` call, after the checks that tie
    them to the training grid (module docstring)."""
    total = sigmas.shape[0]
    seq_sigmas, seq_timesteps = sequence.sigmas, sequence.timesteps
    if float(seq_sigmas[-1]) != 0.0 or seq_sigmas.shape[0] != seq_timesteps.shape[0] + 1:
        raise HostFixtureError(f"{steps} steps: the sigmas do not end with the terminal 0")
    index = grid_indices(sigmas, seq_sigmas[:-1])
    if not records.bits_equal(timesteps[index.long()], seq_timesteps):
        raise HostFixtureError(f"{steps} steps: a timestep is not the training grid's")
    # The snapping went through this table and these indices
    # (descending index = total − 1 − ascending index).
    if not records.bits_equal(sequence.training_ascending.flip(0), sigmas):
        raise HostFixtureError(f"{steps} steps: upstream snapped to another training grid")
    nearest = (total - 1 - sequence.nearest_ascending).to(torch.int32)
    if not records.bits_equal(torch.unique_consecutive(nearest), index):
        raise HostFixtureError(f"{steps} steps: the snapped indices do not give the sequence")
    return {
        "sigmas": seq_sigmas,
        "timesteps": seq_timesteps,
        "grid": index,
        "requested": sequence.requested,
        "nearest": nearest,
    }


def grid_indices(table: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """The index of each value in `table` (bit-identical members only; anything else fails)."""
    table_bits = table.contiguous().view(torch.int32)
    value_bits = values.contiguous().view(torch.int32)
    matches = (value_bits[:, None] == table_bits[None, :]).nonzero()
    if matches.shape[0] != values.shape[0] or not bool(
        (matches[:, 0] == torch.arange(values.shape[0])).all()
    ):
        raise HostFixtureError("a value is not exactly one member of the table")
    return matches[:, 1].to(torch.int32)


def compute(snapshot: Path, module: ModuleType) -> dict[str, torch.Tensor]:
    """Every fixed value (one pass at the current thread count)."""
    tensors: dict[str, torch.Tensor] = {}
    dit = transformer_config(snapshot)
    for height, width, length in ROPE_CASES:
        tensors[f"rope.{height}x{width}.t{length}"] = rope_table(module, dit, height, width, length)
    config = scheduler_config(snapshot, module)
    sigmas, timesteps = training_grid(module, config)
    tensors["training.sigmas"], tensors["training.timesteps"] = sigmas, timesteps
    scheduler = pipeline_ref.load_scheduler(snapshot, module)
    for steps in STEP_COUNTS:
        sequence = step_sequence(module, scheduler, steps)
        for name, tensor in sequence_tensors(steps, sequence, sigmas, timesteps).items():
            tensors[f"steps.{steps}.{name}"] = tensor
    for shift_index, shift in enumerate(SHIFTS):
        shifted = shifted_scheduler(snapshot, module, shift)
        for steps in SHIFT_STEP_COUNTS:
            sequence = step_sequence(module, shifted, steps)
            for name, tensor in sequence_tensors(steps, sequence, sigmas, timesteps).items():
                tensors[f"shift.{shift_index}.steps.{steps}.{name}"] = tensor
    tensors.update(guidance_samples(module, config))
    model = tiny_transformer(module, int(dit["freq_dim"]))
    tensors["time_sinusoid"] = torch.cat([time_sinusoid(model, t) for t in timesteps])
    return tensors


def disagreements(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> list[str]:
    """The names whose tensors differ in bits (or exist on one side only)."""
    names = sorted(set(left) | set(right))
    return [
        n
        for n in names
        if n not in left or n not in right or not records.bits_equal(left[n], right[n])
    ]


def generate(out: Path) -> dict[str, Any]:
    """Compute the fixed values at two thread counts, compare, write; return the run summary."""
    clock = records.Clock()
    snapshot, module = pipeline_ref.upstream()
    default_threads = torch.get_num_threads()
    thread_counts = (1, default_threads) if default_threads != 1 else (1, 2)
    runs: list[dict[str, torch.Tensor]] = []
    environments: list[dict[str, Any]] = []
    for threads in thread_counts:
        torch.set_num_threads(threads)
        runs.append(compute(snapshot, module))
        environments.append(records.environment())
        clock.lap(f"compute at {threads} thread(s)")
    torch.set_num_threads(default_threads)
    differing = disagreements(runs[0], runs[1])
    if differing:
        raise HostFixtureError(f"the thread counts {thread_counts} disagree on {differing[:10]}")
    # What steers the CPU kernels goes into the record (the CPU model and host name do not); the
    # environment is not changed between the runs, so both must report the same.
    steering = {key: environments[-1][key] for key in ("cpu_capability", "threading_env")}
    if any({key: e[key] for key in steering} != steering for e in environments):
        raise HostFixtureError(f"the CPU capability or threading environment moved: {environments}")
    tensors = runs[-1]
    steps_taken = {str(n): int(tensors[f"steps.{n}.timesteps"].shape[0]) for n in STEP_COUNTS}
    metadata = {
        "source": {
            "repo": sources.SOURCE.repo,
            "revision": sources.SOURCE.revision,
            "pipeline_code_sha256": sources.PIPELINE_CODE_SHA256,
        },
        "rope_cases": [list(case) for case in ROPE_CASES],
        "step_counts": list(STEP_COUNTS),
        "steps_taken": steps_taken,
        "shift_cases": {"shifts": list(SHIFTS), "step_counts": list(SHIFT_STEP_COUNTS)},
        "cfg": {
            "scales": list(CFG_SCALES),
            "layers": CFG_LAYERS,
            "skip_layers": list(CFG_SKIP_LAYERS),
            "cells": CFG_CELLS,
            "seed": CFG_SEED,
        },
        "scheduler_config": scheduler_config(snapshot, module),
        "versions": records.versions(),
        "thread_counts": list(thread_counts),
        "threads_agree": True,
        **steering,
    }
    path = out / RECORD_NAME
    records.write_record(path, tensors, metadata, key=METADATA_KEY)
    clock.lap("write")
    shortened = {n: k for n, k in steps_taken.items() if int(n) != k}
    return {
        "path": str(path),
        "tensors": len(tensors),
        "sha256": sources.sha256_of(path),
        "thread_counts": list(thread_counts),
        "threads_agree": True,
        "environments": environments,
        "steps_shortened_by_snapping": shortened,
        "seconds": {**clock.phases, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=None, help="default: the stage-0 reference dir")
    args = parser.parse_args(argv)
    out = args.out or records.default_reference_dir()
    summary = generate(out)
    text = json.dumps(summary, indent=1)
    print(text)
    (out / "host.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
