"""Reference fixture for the host UniPC and CFG of Wan2.1 (ADR 0118 decision 5, step 6).

Bakes the UniPC (flow matching, shift 3.0, bh2, order 2, 50 steps) and the CFG combination that the
TS host (`packages/models/src/wan/`) implements by **driving the real classes** of diffusers 0.39.0.
No real weights are needed — what is fed in is synthetic model output with a fixed seed (a small
latent `[1,16,3,16,16]`, f32). The scheduler config is the pinned revision's
`scheduler/scheduler_config.json` as is (the side that makes the reference does not rewrite values —
decision 5).

## Contents

- **σ and timestep sequences** (for each step count in {@link SCHEDULE_STEPS} — 50 is the reference
  setting, 2 the few-step end-to-end run): `scheduler.sigmas` after `set_timesteps` (f32, including
  the trailing 0) and `scheduler.timesteps` (int64). In addition, the **f64 σ sequence** goes into
  the JSON (TS computes in f64 and drops to f32 at the end — decision 5). The f64 sequence is
  rebuilt with the same numpy expressions as the flow branch of upstream `set_timesteps`
  (`scheduling_unipc_multistep.py:428-450`), and is written only after checking that its values
  dropped to f32 are bit-identical to `scheduler.sigmas` and that truncating `σ × 1000` to int64
  matches `scheduler.timesteps` exactly (upstream does not expose its f64 intermediates, so the
  values that cannot be extracted are rebuilt and pinned by the values that can).
- **Trajectory**: the latent after each step when the initial latent and 50 model outputs (both a
  fixed-seed `randn`) are run in the same shape as the upstream loop
  (`for t in scheduler.timesteps: latents = scheduler.step(out, t, latents)`). The order of each
  step (`this_order`) and whether the corrector ran go into the JSON.
- **CFG**: cond / uncond (fixed seed) and the upstream combination `noise_uncond + guidance_scale *
  (noise_pred - noise_uncond)` (verbatim from `pipeline_wan.py:632`, f32, guide 5.0).

The TS reader has no F64 (`packages/runtime/src/format/safetensors.ts`), so the f64 sequences go
into the JSON (Python's `repr` is a decimal form that round-trips the value).

    uv run --group wan --inexact python -m wan.scheduler_ref

Output (default `packages/models/tests/fixtures/wan-scheduler/`):

    unipc.safetensors  sigmas / timesteps (50 steps), latents_init, model_outputs,
                       trajectory, cfg.*
    unipc.json         source, config, σ (f64 / f32) and timestep sequences (2 / 50 steps), orders,
                       seeds

MUST: import diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from _shared.paths import REPO_ROOT
from wan.pipeline_ref import GUIDANCE_SCALE, NUM_INFERENCE_STEPS
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, local_snapshot

DEFAULT_OUT = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-scheduler"

#: The fixture latent `[C, F, H, W]` (the batch axis is dropped to match the host's shape — upstream
#: receives `[1, …]`).
LATENT_SHAPE = (16, 3, 16, 16)

#: The step counts whose σ / timestep sequences are baked (50 = the reference setting, 2 = the
#: few-step end-to-end run — decision 8).
SCHEDULE_STEPS = (2, NUM_INFERENCE_STEPS)

#: Seeds of the trajectory and CFG random numbers (the values mean nothing — only being fixed
#: matters).
TRAJECTORY_SEED = 20261020
CFG_SEED = 20261021

#: The amount upstream subtracts from σ[0] (`scheduling_unipc_multistep.py:437-440`).
SIGMA_EPS = 1e-6


class SchedulerFixtureError(AssertionError):
    """A rebuilt f64 sequence does not match the upstream f32 / int64 sequence."""


def load_scheduler(model: str = DEFAULT_MODEL) -> Any:
    """Builds `UniPCMultistepScheduler` from the scheduler config of the pinned revision."""
    from diffusers import UniPCMultistepScheduler

    return UniPCMultistepScheduler.from_pretrained(local_snapshot(model), subfolder="scheduler")


def sigmas_float64(config: Any, steps: int) -> np.ndarray:
    """The f64 σ sequence of the flow branch of upstream `set_timesteps` (before the trailing 0 is
    appended; length `steps`).

    The expressions are the same numpy calls as the upstream lines
    (`scheduling_unipc_multistep.py:430-440`). The dynamic-shift and `shift_terminal` branches are
    not taken with the Wan2.1 config, so fail loudly if either is set.
    """
    if not config.use_flow_sigmas or config.use_dynamic_shifting or config.shift_terminal:
        raise SchedulerFixtureError("only the static-shift flow branch is supported")
    sigmas = np.linspace(1, 1 / config.num_train_timesteps, steps + 1)[:-1]
    sigmas = config.flow_shift * sigmas / (1 + (config.flow_shift - 1) * sigmas)
    if np.fabs(sigmas[0] - 1) < SIGMA_EPS:
        sigmas[0] -= SIGMA_EPS
    return sigmas


def schedule(scheduler: Any, steps: int) -> dict[str, Any]:
    """The sequences of `set_timesteps(steps)` (f32 σ, int64 timesteps) and the pinned f64 σ
    sequence."""
    scheduler.set_timesteps(steps)
    sigmas32 = scheduler.sigmas.numpy()
    timesteps = scheduler.timesteps.numpy()
    if scheduler.sigmas.dtype != torch.float32 or scheduler.timesteps.dtype != torch.int64:
        raise SchedulerFixtureError(
            f"unexpected dtype of the upstream sequences: {scheduler.sigmas.dtype} /"
            f" {scheduler.timesteps.dtype}"
        )
    sigmas64 = sigmas_float64(scheduler.config, steps)
    with_zero = np.concatenate([sigmas64, [0.0]])
    if not np.array_equal(with_zero.astype(np.float32).view(np.uint32), sigmas32.view(np.uint32)):
        raise SchedulerFixtureError(
            f"steps={steps}: the f64 sequence dropped to f32 differs from upstream"
        )
    truncated = (sigmas64 * scheduler.config.num_train_timesteps).astype(np.int64)
    if not np.array_equal(truncated, timesteps):
        raise SchedulerFixtureError(
            f"steps={steps}: truncating σ × 1000 differs from the upstream timesteps"
        )
    return {
        "steps": steps,
        "sigmas_f64": [float(value) for value in with_zero],
        "sigmas_f32": [float(value) for value in sigmas32],
        "timesteps": [int(value) for value in timesteps],
    }


def trajectory(scheduler: Any, seed: int) -> dict[str, Any]:
    """Feeds synthetic model outputs to `step` in the same shape as the upstream loop and collects
    the latent after each step."""
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    generator = torch.Generator().manual_seed(seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)
    model_outputs = torch.randn(NUM_INFERENCE_STEPS, 1, *LATENT_SHAPE, generator=generator)
    latents = latents_init.clone()
    states: list[torch.Tensor] = []
    orders: list[int] = []
    correctors: list[bool] = []
    for index, t in enumerate(scheduler.timesteps):
        # Record the same condition as the check inside `step` (`use_corrector`), from the state
        # before the call.
        correctors.append(
            index > 0
            and index - 1 not in scheduler.disable_corrector
            and scheduler.last_sample is not None
        )
        latents = scheduler.step(model_outputs[index], t, latents, return_dict=False)[0]
        states.append(latents)
        orders.append(int(scheduler.this_order))
    stacked = torch.cat(states)
    if not bool(stacked.isfinite().all()):
        raise SchedulerFixtureError("the trajectory has non-finite values")
    return {
        "latents_init": latents_init[0],
        "model_outputs": model_outputs[:, 0],
        "trajectory": stacked,
        "orders": orders,
        "correctors": correctors,
    }


def classifier_free_guidance(seed: int, guidance_scale: float) -> dict[str, torch.Tensor]:
    """The CFG combination (the expression of upstream `pipeline_wan.py:632`, verbatim — upstream
    has no function for it, so the expression is copied)."""
    generator = torch.Generator().manual_seed(seed)
    noise_pred = torch.randn(1, *LATENT_SHAPE, generator=generator)
    noise_uncond = torch.randn(1, *LATENT_SHAPE, generator=generator)
    current_guidance_scale = guidance_scale
    combined = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
    return {"cfg.cond": noise_pred[0], "cfg.uncond": noise_uncond[0], "cfg.out": combined[0]}


def build(model: str = DEFAULT_MODEL) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Builds the fixture tensors and metadata (JSON) without writing them."""
    scheduler = load_scheduler(model)
    schedules = {str(steps): schedule(scheduler, steps) for steps in SCHEDULE_STEPS}
    # Rebuild after taking the sequences — `set_timesteps` resets the step state (model_outputs
    # and so on).
    run = trajectory(load_scheduler(model), TRAJECTORY_SEED)
    reference = schedules[str(NUM_INFERENCE_STEPS)]
    tensors = {
        "sigmas": torch.tensor(reference["sigmas_f32"], dtype=torch.float32),
        "timesteps": torch.tensor(reference["timesteps"], dtype=torch.int64),
        "latents_init": run["latents_init"],
        "model_outputs": run["model_outputs"],
        "trajectory": run["trajectory"],
        **classifier_free_guidance(CFG_SEED, GUIDANCE_SCALE),
    }
    source = SOURCES[model]
    config = {
        key: value for key, value in dict(scheduler.config).items() if not key.startswith("_")
    }
    meta = {
        "_doc": [
            "Fixture for checking the Wan2.1 host UniPC and CFG (packages/models/src/wan/).",
            "Generated by tools/export-recipes/wan/scheduler_ref.py (drives the"
            " UniPCMultistepScheduler of diffusers 0.39.0 — the formulas are not copied).",
            "sigmas_f64 rebuilds the upstream f64 intermediates with the same numpy expressions"
            " (checked: dropped to f32 they are bit-identical to sigmas_f32, and truncating"
            " × 1000 matches timesteps).",
            "trajectory[i] is the latent after step i. model_outputs[i] is that step's input.",
        ],
        "source": {"repo": source.repo, "revision": source.revision},
        "scheduler_config": config,
        "latent_shape": list(LATENT_SHAPE),
        "trajectory_seed": TRAJECTORY_SEED,
        "cfg_seed": CFG_SEED,
        "guidance_scale": GUIDANCE_SCALE,
        "orders": run["orders"],
        "correctors": run["correctors"],
        "schedules": schedules,
    }
    return tensors, meta


def write(out: Path, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Writes the fixture and returns a summary."""
    tensors, meta = build(model)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        str(out / "unipc.safetensors"),
    )
    # Two-space indent (the indent of the repository formatter, Oxfmt — it also folds short arrays
    # onto one line, so run `vp fmt` on the written file; the fixture is tracked, so a rewrite must
    # not produce a diff).
    (out / "unipc.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    reference = meta["schedules"][str(NUM_INFERENCE_STEPS)]
    return {
        "out": str(out),
        "bytes": (out / "unipc.safetensors").stat().st_size,
        "sigma0_f64": reference["sigmas_f64"][0],
        "sigma0_f32": reference["sigmas_f32"][0],
        "timesteps_head": reference["timesteps"][:5],
        "timesteps_2_steps": meta["schedules"]["2"]["timesteps"],
        "orders": "".join(str(order) for order in meta["orders"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps(write(args.out, args.model), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
