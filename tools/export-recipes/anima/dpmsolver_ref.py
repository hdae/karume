"""DPM++ 2M（`DPMSolverMultistepScheduler`）の golden を出す小さいリグ。

TS 側の family 非依存モジュール（`packages/models/src/generation/dpm-solver-multistep.ts`）は
diffusers 0.39.0 の `DPMSolverMultistepScheduler` の 1:1 写しなので、その**実クラスを駆動して**
係数と各 step の状態更新を印字する。突き合わせ先は
`packages/models/tests/dpm_solver_multistep_test.ts`（inline 期待値の由来がここ）。

実重みは 1 バイトも要らない — 食わせるのは**合成 model 出力**（シード固定・小テンソル・f32）
なので、`uv run --no-sync python -m anima.dpmsolver_ref` が数秒で回る。

## MUST: 梯子は既存 Euler 経路のものを差し込む（diffusers 内蔵の flow 梯子は使わない）

diffusers 0.39.0 の `set_timesteps` は `sigmas=` 引数を持たない（実測 — 受けるのは
`num_inference_steps` / `device` / `mu` / `timesteps` の 4 つだけ）ので、flow の梯子を組む
経路は `use_flow_sigmas=True` + `flow_shift` の 1 本しかない。その内蔵経路は

    alphas = linspace(1, 1/num_train_timesteps, N+1);  sigmas = 1 − alphas
    sigmas = flip(shift·s / (1 + (shift−1)·s))[:-1]  ⧺ [0]

なので梯子の格子点が `shift(0.999·k/N)`（k = N…1）になり、karume の Euler 経路
（{@link sigma_schedule} = `shift(k/N)`）と**バイト同一にならない**。実測（shift=3）:
steps=2/4/8/32 のいずれでも最大絶対差 3.752112e-04・全要素がビット不一致。原因は
①`1/num_train_timesteps` を linspace の終端に置くので先頭 sigma が 1 でなく 0.999666 になる
②f32 への丸めが shift 変換の**後**（karume は前）の 2 点。

選択サンプラは **既存 Euler 経路と同じ梯子の上**で回すのが要件なので、`set_timesteps` の後に
`sigmas` / `timesteps` を karume の梯子で差し替えてから `step()` を回す。差し替えるのは表だけで、
`convert_model_output` / `dpm_solver_first_order_update` /
`multistep_dpm_solver_second_order_update` / `step` の更新式は実クラスのものがそのまま走る。

## 梯子の端で出る ±inf は正常（潰さない）

karume の梯子は先頭が厳密に 1・末尾が厳密に 0 なので、`_sigma_to_alpha_sigma_t`（flow 経路は
`alpha = 1 − sigma`）が step 0 で `alpha_s = 0`・最終 step で `sigma_t = 0` を返し、
`lambda = log(alpha) − log(sigma)` が ∓inf、`h` が +inf になる。式は破綻せず

    step 0      exp(−h) = 0 → x ← sigma_t·x + alpha_t·x0（Euler 1 段と同値）
    step 1      lambda_s1 = −inf → r0 = inf → D1 = 0（2 次項が自然に落ちる）
    最終 step   sigma_t/sigma_s0 = 0・係数 = −1 → x ← x0（ビット同一）

に落ちる。diffusers 内蔵の flow 梯子は先頭が 0.999666 なので inf を踏まない — **この経路
固有の性質**なので、TS 側の写しでも同じ順序で ∓inf を通す（前もって特別扱いすると値が変わる）。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from .pipeline_ref import NUM_TRAIN_TIMESTEPS, SHIFT, sigma_schedule

#: 合成 model 出力のシード。値に意味は無い（固定されていることだけが要件）。
DEFAULT_SEED = 20260824


def build_scheduler(sigmas: np.ndarray, shift: float) -> Any:
    """karume の梯子を差し込んだ `DPMSolverMultistepScheduler`（DPM++ 2M の構成）。

    構成は DPM++ 2M そのもの: `algorithm_type="dpmsolver++"` × `solver_order=2` ×
    `solver_type="midpoint"`。flow matching なので `prediction_type="flow_prediction"`
    （`x0 = x − sigma·v`）で、`final_sigmas_type="zero"` は梱包の終端 0 と対。

    MUST: `set_begin_index(0)` を立てる。既定の `_init_step_index` は timestep 値を
    `self.timesteps` から検索して step を決めるので、差し替えた梯子に**同じ整数へ丸まる
    timestep が 2 つ**あると（steps を増やすと起きる）先頭の step が 1 つずれる。
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
    """diffusers 内蔵の flow 梯子（`use_flow_sigmas=True`）— 差分の記録用。"""
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
    """その step の中間スケジュール（TS 側の写しと 1 対 1 で突き合わせる係数）。

    `sample_scale` = `sigma_t/sigma_s0`・`d0_scale` = `alpha_t·(exp(−h) − 1)`・
    `d1_scale` = その半分（midpoint）・`inv_r0` = `1/r0`（2 次項の係数）。

    `solver_order=2` では `step()` の分岐構造上 `lower_order_second` に到達しないので、1 次へ
    落ちるのは **step 0**（`lower_order_nums < 1`）と**最終 step**（`final_sigmas_type="zero"`
    ⇒ `lower_order_final` が常に真）の 2 つだけ。2 次項の係数はその 2 つでは**出さない** —
    最終 step は `h` も `h_0` も +inf で `r0 = inf/inf = NaN` になり、印字すると「使われない
    NaN」が golden に混ざる。
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
    """合成 model 出力の系列を実クラスへ食わせ、係数と各 step の更新出力を集める。"""
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
                # `model_outputs[-1]` = その step の x0 予測（`convert_model_output` の出力）。
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
