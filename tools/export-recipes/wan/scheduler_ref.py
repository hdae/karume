"""Wan2.1 のホストの UniPC と CFG の参照 fixture（ADR 0118 決定 5・段 6）。

TS のホスト（`packages/models/src/wan/`）が持つ UniPC（flow matching・shift 3.0・bh2・order 2・
50 ステップ）と CFG の合成を、diffusers 0.39.0 の**実クラスを駆動して**焼く。実重みは要らない —
食わせるのは seed 固定の合成のモデル出力（小形の潜在 `[1,16,3,16,16]`・f32）。scheduler の
config は pin した revision の `scheduler/scheduler_config.json` をそのまま使う（参照を作る側で
値を書き換えない — 決定 5）。

## 中身

- **σ 列と timestep 列**（{@link SCHEDULE_STEPS} の各ステップ数 — 50 は参照の設定・2 は少ステップの
  通し）: `set_timesteps` の後の `scheduler.sigmas`（f32・最後の 0 を含む）と `scheduler.timesteps`
  （int64）。加えて **f64 の σ 列**を JSON に置く（TS は f64 で計算して最後に f32 へ落とす —
  決定 5）。f64 の列は上流の `set_timesteps` の flow の分岐
  （`scheduling_unipc_multistep.py:428-450`）と同じ numpy の式で作り直し、f32 へ落とした値が
  `scheduler.sigmas` とビット一致・`σ × 1000` の int64 への切り捨てが `scheduler.timesteps` と
  完全一致することを確かめてから書く（上流は f64 の中間を外へ出さないので、取り出せない値を
  作り直して、出せる値で縛る）。
- **軌跡**: 初期の潜在と 50 本のモデル出力（どちらも seed 固定の `randn`）を、上流のループと同じ
  形（`for t in scheduler.timesteps: latents = scheduler.step(out, t, latents)`）で回した各ステップ
  後の潜在。ステップごとの次数（`this_order`）と corrector の有無を JSON に置く。
- **CFG**: cond / uncond（seed 固定）と、上流の合成 `noise_uncond + guidance_scale * (noise_pred -
  noise_uncond)`（`pipeline_wan.py:632` の逐語・f32・guide 5.0）。

TS の読み手は F64 を持たない（`packages/runtime/src/format/safetensors.ts`）ので、f64 の列は JSON
（Python の `repr` は往復で値を保つ 10 進）に置く。

    uv run --group wan --inexact python -m wan.scheduler_ref

出力（既定 `packages/models/tests/fixtures/wan-scheduler/`）:

    unipc.safetensors  sigmas / timesteps（50 ステップ）・latents_init・model_outputs・
                       trajectory・cfg.*
    unipc.json         出所・config・σ（f64 / f32）と timestep の列（2 / 50 ステップ）・次数・seed

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
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

#: fixture の潜在 `[C, F, H, W]`（バッチ軸はホストの形に合わせて落とす — 上流へは `[1, …]` で
#: 渡す）。
LATENT_SHAPE = (16, 3, 16, 16)

#: σ / timestep の列を焼くステップ数（50 = 参照の設定・2 = 少ステップの通し — 決定 8）。
SCHEDULE_STEPS = (2, NUM_INFERENCE_STEPS)

#: 軌跡と CFG の乱数の seed（値に意味は無い — 固定されていることだけが要件）。
TRAJECTORY_SEED = 20261020
CFG_SEED = 20261021

#: 上流が σ[0] から引く量（`scheduling_unipc_multistep.py:437-440`）。
SIGMA_EPS = 1e-6


class SchedulerFixtureError(AssertionError):
    """作り直した f64 の列が上流の f32 / int64 の列と合わない。"""


def load_scheduler(model: str = DEFAULT_MODEL) -> Any:
    """pin した revision の scheduler の config で `UniPCMultistepScheduler` を組む。"""
    from diffusers import UniPCMultistepScheduler

    return UniPCMultistepScheduler.from_pretrained(local_snapshot(model), subfolder="scheduler")


def sigmas_float64(config: Any, steps: int) -> np.ndarray:
    """上流 `set_timesteps` の flow の分岐の f64 の σ 列（最後の 0 を足す前・長さ `steps`）。

    式は上流の行（`scheduling_unipc_multistep.py:430-440`）と同じ numpy の呼び出し。動的 shift と
    `shift_terminal` の分岐は Wan2.1 の config では通らないので、立っていたら fail loudly。
    """
    if not config.use_flow_sigmas or config.use_dynamic_shifting or config.shift_terminal:
        raise SchedulerFixtureError("flow の静的 shift の分岐以外は対象外")
    sigmas = np.linspace(1, 1 / config.num_train_timesteps, steps + 1)[:-1]
    sigmas = config.flow_shift * sigmas / (1 + (config.flow_shift - 1) * sigmas)
    if np.fabs(sigmas[0] - 1) < SIGMA_EPS:
        sigmas[0] -= SIGMA_EPS
    return sigmas


def schedule(scheduler: Any, steps: int) -> dict[str, Any]:
    """`set_timesteps(steps)` の列（f32 の σ・int64 の timestep）と、縛った f64 の σ 列。"""
    scheduler.set_timesteps(steps)
    sigmas32 = scheduler.sigmas.numpy()
    timesteps = scheduler.timesteps.numpy()
    if scheduler.sigmas.dtype != torch.float32 or scheduler.timesteps.dtype != torch.int64:
        raise SchedulerFixtureError(
            f"上流の列の dtype が想定外: {scheduler.sigmas.dtype} / {scheduler.timesteps.dtype}"
        )
    sigmas64 = sigmas_float64(scheduler.config, steps)
    with_zero = np.concatenate([sigmas64, [0.0]])
    if not np.array_equal(with_zero.astype(np.float32).view(np.uint32), sigmas32.view(np.uint32)):
        raise SchedulerFixtureError(f"steps={steps}: f64 の列を f32 へ落とした値が上流と違う")
    truncated = (sigmas64 * scheduler.config.num_train_timesteps).astype(np.int64)
    if not np.array_equal(truncated, timesteps):
        raise SchedulerFixtureError(f"steps={steps}: σ × 1000 の切り捨てが上流の timestep と違う")
    return {
        "steps": steps,
        "sigmas_f64": [float(value) for value in with_zero],
        "sigmas_f32": [float(value) for value in sigmas32],
        "timesteps": [int(value) for value in timesteps],
    }


def trajectory(scheduler: Any, seed: int) -> dict[str, Any]:
    """合成のモデル出力を上流のループと同じ形で `step` に食わせ、各ステップ後の潜在を集める。"""
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    generator = torch.Generator().manual_seed(seed)
    latents_init = torch.randn(1, *LATENT_SHAPE, generator=generator)
    model_outputs = torch.randn(NUM_INFERENCE_STEPS, 1, *LATENT_SHAPE, generator=generator)
    latents = latents_init.clone()
    states: list[torch.Tensor] = []
    orders: list[int] = []
    correctors: list[bool] = []
    for index, t in enumerate(scheduler.timesteps):
        # `step` の中の判定（`use_corrector`）と同じ条件を、呼ぶ前の状態で記録する。
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
        raise SchedulerFixtureError("軌跡に非有限値がある")
    return {
        "latents_init": latents_init[0],
        "model_outputs": model_outputs[:, 0],
        "trajectory": stacked,
        "orders": orders,
        "correctors": correctors,
    }


def classifier_free_guidance(seed: int, guidance_scale: float) -> dict[str, torch.Tensor]:
    """CFG の合成（上流 `pipeline_wan.py:632` の式を逐語 — 上流に関数が無いので式を写す）。"""
    generator = torch.Generator().manual_seed(seed)
    noise_pred = torch.randn(1, *LATENT_SHAPE, generator=generator)
    noise_uncond = torch.randn(1, *LATENT_SHAPE, generator=generator)
    current_guidance_scale = guidance_scale
    combined = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
    return {"cfg.cond": noise_pred[0], "cfg.uncond": noise_uncond[0], "cfg.out": combined[0]}


def build(model: str = DEFAULT_MODEL) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """fixture のテンソルとメタ（JSON）を組む（書かない）。"""
    scheduler = load_scheduler(model)
    schedules = {str(steps): schedule(scheduler, steps) for steps in SCHEDULE_STEPS}
    # 列を取った後で作り直す — `set_timesteps` が step の状態（model_outputs 等）を初期化する。
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
            "Wan2.1 のホストの UniPC と CFG（packages/models/src/wan/）の突き合わせ用 fixture。",
            "生成: tools/export-recipes/wan/scheduler_ref.py（diffusers 0.39.0 の"
            " UniPCMultistepScheduler を駆動 — 式は写さない）。",
            "sigmas_f64 は上流の f64 の中間を同じ numpy の式で作り直したもの（f32 へ落とすと"
            " sigmas_f32 とビット一致・× 1000 の切り捨てが timesteps と一致することを"
            "確かめてある）。",
            "trajectory[i] は i 番目の step の後の潜在。model_outputs[i] がその step の入力。",
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
    """fixture を書き、要約を返す。"""
    tensors, meta = build(model)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        str(out / "unipc.safetensors"),
    )
    # 2 字下げ（`deno fmt` の JSON の形 — 追跡対象の fixture なので書き直しで差分を出さない）。
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
