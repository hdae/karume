"""A small rig that produces the golden of DPM++ 2M (`DPMSolverMultistepScheduler`).

The family-independent TS module (`packages/models/src/generation/dpm-solver-multistep.ts`) is a
1:1 copy of diffusers 0.39.0's `DPMSolverMultistepScheduler`, so this **drives the real class** and
prints the coefficients and the state update of each step. The comparison target is
`packages/models/tests/anima_dpm_solver_multistep_test.ts` (its inline expected values come from
here).

No real weights are needed at all — what is fed in is **synthetic model output** (fixed seed,
small tensors, f32), so `uv run --no-sync python -m anima.dpmsolver_ref` runs in seconds.

## MUST: plug in the Euler path's sigma schedule (not diffusers' built-in flow schedule)

diffusers 0.39.0's `set_timesteps` has no `sigmas=` argument (measured — it takes only the 4
arguments `num_inference_steps` / `device` / `mu` / `timesteps`), so the only path that builds a
flow schedule is `use_flow_sigmas=True` + `flow_shift`. That built-in path is

    alphas = linspace(1, 1/num_train_timesteps, N+1);  sigmas = 1 − alphas
    sigmas = flip(shift·s / (1 + (shift−1)·s))[:-1]  ⧺ [0]

so the grid points of the schedule become `shift(0.999·k/N)` (k = N…1) and are **not
byte-identical** to karume's Euler path ({@link sigma_schedule} = `shift(k/N)`). Measured
(shift=3): for every one of steps=2/4/8/32 the max abs difference is 3.752112e-04 and every element
differs in its bits. The causes are two: ① `1/num_train_timesteps` is the end point of the
linspace, so the first sigma is 0.999666 instead of 1; ② the rounding to f32 happens **after** the
shift transform (karume rounds before).

The selectable sampler is required to run **on the same schedule as the existing Euler path**, so
after `set_timesteps` the `sigmas` / `timesteps` are replaced with karume's schedule before running
`step()`. Only the tables are replaced; the update formulas of `convert_model_output` /
`dpm_solver_first_order_update` / `multistep_dpm_solver_second_order_update` / `step` run as the
real class has them.

## The ±inf at the ends of the schedule is expected (do not clamp it)

karume's schedule starts at exactly 1 and ends at exactly 0, so `_sigma_to_alpha_sigma_t` (on the
flow path `alpha = 1 − sigma`) returns `alpha_s = 0` at step 0 and `sigma_t = 0` at the last step,
making `lambda = log(alpha) − log(sigma)` ∓inf and `h` +inf. The formulas do not break down and
reduce to

    step 0      exp(−h) = 0 → x ← sigma_t·x + alpha_t·x0 (equivalent to one Euler step)
    step 1      lambda_s1 = −inf → r0 = inf → D1 = 0 (the second-order term drops out naturally)
    last step   sigma_t/sigma_s0 = 0, coefficient = −1 → x ← x0 (bit-identical)

diffusers' built-in flow schedule starts at 0.999666 and never hits inf — **a property specific to
this path**, so the TS copy also passes ∓inf through in the same order (special-casing it in
advance changes the values).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from .pipeline_ref import NUM_TRAIN_TIMESTEPS, SHIFT, sigma_schedule

#: Seed of the synthetic model output. The value has no meaning (only being fixed matters).
DEFAULT_SEED = 20260824


def build_scheduler(sigmas: np.ndarray, shift: float) -> Any:
    """A `DPMSolverMultistepScheduler` with karume's schedule plugged in (configured as DPM++ 2M).

    The configuration is DPM++ 2M itself: `algorithm_type="dpmsolver++"` × `solver_order=2` ×
    `solver_type="midpoint"`. It is flow matching, so `prediction_type="flow_prediction"`
    (`x0 = x − sigma·v`), and `final_sigmas_type="zero"` pairs with the schedule's terminal 0.

    MUST: call `set_begin_index(0)`. The default `_init_step_index` finds the step by searching
    `self.timesteps` for the timestep value, so if the replaced schedule has **two timesteps that
    round to the same integer** (which happens as steps grow), the first step is off by one.
    """
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        solver_order=2,
        prediction_type="flow_prediction",
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        use_flow_sigmas=True,
        flow_shift=shift,
        final_sigmas_type="zero",
    )
    scheduler.set_timesteps(len(sigmas) - 1)
    scheduler.sigmas = torch.from_numpy(sigmas)
    scheduler.timesteps = torch.from_numpy(sigmas[:-1] * NUM_TRAIN_TIMESTEPS).to(dtype=torch.int64)
    scheduler.set_begin_index(0)
    return scheduler


def native_flow_sigmas(steps: int, shift: float) -> np.ndarray:
    """diffusers' built-in flow schedule (`use_flow_sigmas=True`) — for recording the difference."""
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=shift,
        final_sigmas_type="zero",
    )
    scheduler.set_timesteps(steps)
    return scheduler.sigmas.numpy()


def step_coefficients(scheduler: Any, index: int, steps: int) -> dict[str, Any]:
    """The intermediate schedule of that step (coefficients compared 1:1 with the TS copy).

    `sample_scale` = `sigma_t/sigma_s0`, `d0_scale` = `alpha_t·(exp(−h) − 1)`, `d1_scale` = half of
    it (midpoint), `inv_r0` = `1/r0` (the coefficient of the second-order term).

    With `solver_order=2` the branch structure of `step()` never reaches `lower_order_second`, so
    the only steps that fall back to first order are **step 0** (`lower_order_nums < 1`) and **the
    last step** (`final_sigmas_type="zero"` ⇒ `lower_order_final` is always true). The
    second-order coefficients are **not emitted** for those two — at the last step both `h` and
    `h_0` are +inf and `r0 = inf/inf = NaN`, and printing it would mix "an unused NaN" into the
    golden.
    """
    sigmas = scheduler.sigmas
    sigma_s0, sigma_t = sigmas[index], sigmas[index + 1]
    alpha_t, alpha_s0 = 1 - sigma_t, 1 - sigma_s0
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0
    d0_scale = alpha_t * (torch.exp(-h) - 1.0)
    coefficients: dict[str, Any] = {
        "order": 1 if index in (0, steps - 1) else 2,
        "lambda_t": float(lambda_t),
        "lambda_s0": float(lambda_s0),
        "h": float(h),
        "sample_scale": float(sigma_t / sigma_s0),
        "d0_scale": float(d0_scale),
    }
    if coefficients["order"] == 2:
        sigma_s1 = sigmas[index - 1]
        lambda_s1 = torch.log(1 - sigma_s1) - torch.log(sigma_s1)
        h_0 = lambda_s0 - lambda_s1
        coefficients["d1_scale"] = float(0.5 * d0_scale)
        coefficients["inv_r0"] = float(1.0 / (h_0 / h))
    return coefficients


def run(steps: int, shift: float, elements: int, seed: int) -> dict[str, Any]:
    """Feeds synthetic model outputs to the real class; collects coefficients and step updates."""
    sigmas = sigma_schedule(steps, shift)
    scheduler = build_scheduler(sigmas, shift)
    native = native_flow_sigmas(steps, shift)

    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn((elements,), generator=generator, dtype=torch.float32)
    sample = initial
    records: list[dict[str, Any]] = []
    for index in range(steps):
        velocity = torch.randn((elements,), generator=generator, dtype=torch.float32)
        coefficients = step_coefficients(scheduler, index, steps)
        updated = scheduler.step(velocity, scheduler.timesteps[index], sample).prev_sample
        records.append(
            {
                "index": index,
                "velocity": [float(value) for value in velocity],
                "coefficients": coefficients,
                # `model_outputs[-1]` = that step's x0 prediction (the output of
                # `convert_model_output`).
                "x0": [float(value) for value in scheduler.model_outputs[-1]],
                "sample": [float(value) for value in updated],
            }
        )
        sample = updated

    return {
        "steps": steps,
        "shift": shift,
        "seed": seed,
        "sigmas": [float(value) for value in sigmas],
        "ladder": {
            "native_flow_sigmas": [float(value) for value in native],
            "max_abs_diff": float(np.max(np.abs(native - sigmas))),
            "bitwise_equal": bool(
                np.array_equal(native.view(np.uint32), sigmas.view(np.uint32)),
            ),
        },
        "initial_sample": [float(value) for value in initial],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--shift", type=float, default=SHIFT)
    parser.add_argument("--elements", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(json.dumps(run(args.steps, args.shift, args.elements, args.seed), indent=2))


if __name__ == "__main__":
    main()
