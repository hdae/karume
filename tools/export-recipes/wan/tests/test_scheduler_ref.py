"""`wan/scheduler_ref.py` の約束事 — UniPC の σ / timestep の列の性質と fixture の往復。

列の性質（最初の timestep が 999・単調）は上流の `set_timesteps` が作る値で見る。scheduler の
config は pin した snapshot のものを使う（{@link wan_snapshot} — 無ければ SKIP）。
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from wan import scheduler_ref
from wan.pipeline_ref import NUM_INFERENCE_STEPS


@pytest.fixture(scope="module")
def built(wan_snapshot: Path):
    pytest.importorskip("diffusers")
    return scheduler_ref.build()


@pytest.fixture
def scheduler(wan_snapshot: Path):
    pytest.importorskip("diffusers")
    return scheduler_ref.load_scheduler()


class TestSchedule:
    def test_the_reference_schedule_starts_at_999_and_decreases(self, scheduler):
        schedule = scheduler_ref.schedule(scheduler, NUM_INFERENCE_STEPS)
        timesteps = schedule["timesteps"]
        sigmas = schedule["sigmas_f32"]

        assert len(timesteps) == NUM_INFERENCE_STEPS
        assert len(sigmas) == NUM_INFERENCE_STEPS + 1
        assert timesteps[0] == 999
        assert all(a > b for a, b in pairwise(timesteps))
        assert all(a > b for a, b in pairwise(sigmas))
        assert sigmas[-1] == 0.0

    def test_the_first_sigma_carries_the_1e6_correction(self, scheduler):
        """shift 後の σ[0] は厳密に 1 で、上流は 1e-6 を引く — それが最初の timestep を 999 に
        する。"""
        sigmas = scheduler_ref.sigmas_float64(scheduler.config, NUM_INFERENCE_STEPS)

        assert sigmas[0] == 1.0 - scheduler_ref.SIGMA_EPS
        assert int(np.int64(sigmas[0] * 1000)) == 999
        assert int(np.int64(1.0 * 1000)) == 1000

    def test_the_shift_comes_from_the_pinned_config(self, scheduler):
        assert scheduler.config.flow_shift == 3.0

    def test_the_two_step_schedule_of_the_few_step_run(self, scheduler):
        schedule = scheduler_ref.schedule(scheduler, 2)

        assert schedule["timesteps"] == [999, 750]
        assert schedule["sigmas_f32"][-1] == 0.0

    def test_a_rebuilt_float64_column_that_drifts_is_caught(
        self, scheduler, monkeypatch: pytest.MonkeyPatch
    ):
        """作り直した f64 の列を縛る検査そのものの故障注入（1 つずらした列は通らない）。"""
        honest = scheduler_ref.sigmas_float64

        def shifted(config, steps):
            values = honest(config, steps)
            return np.concatenate([values[1:], values[-1:] * 0.5])

        monkeypatch.setattr(scheduler_ref, "sigmas_float64", shifted)

        with pytest.raises(scheduler_ref.SchedulerFixtureError):
            scheduler_ref.schedule(scheduler, NUM_INFERENCE_STEPS)


class TestFixture:
    def test_it_is_deterministic(self, built, wan_snapshot: Path):
        tensors, meta = built
        again_tensors, again_meta = scheduler_ref.build()

        assert again_meta == meta
        for name, tensor in tensors.items():
            assert torch.equal(again_tensors[name], tensor), name

    def test_the_orders_warm_up_and_lower_at_the_end(self, built):
        """bh2・order 2・`lower_order_final`: 最初と最後が 1 次・間は 2 次。corrector は 2 本目
        から。"""
        _, meta = built

        assert meta["orders"] == [1] + [2] * (NUM_INFERENCE_STEPS - 2) + [1]
        assert meta["correctors"] == [False] + [True] * (NUM_INFERENCE_STEPS - 1)

    def test_the_tensors_have_the_documented_shapes(self, built):
        tensors, _ = built
        shape = scheduler_ref.LATENT_SHAPE

        assert tensors["sigmas"].dtype == torch.float32
        assert tensors["timesteps"].dtype == torch.int64
        assert tensors["latents_init"].shape == shape
        assert tensors["model_outputs"].shape == (NUM_INFERENCE_STEPS, *shape)
        assert tensors["trajectory"].shape == (NUM_INFERENCE_STEPS, *shape)
        for name in ("cfg.cond", "cfg.uncond", "cfg.out"):
            assert tensors[name].shape == shape

    def test_it_round_trips_through_the_files(self, built, tmp_path: Path):
        from safetensors.torch import load_file

        tensors, meta = built

        scheduler_ref.write(tmp_path)

        loaded = load_file(str(tmp_path / "unipc.safetensors"))
        assert json.loads((tmp_path / "unipc.json").read_text(encoding="utf-8")) == meta
        assert sorted(loaded) == sorted(tensors)
        for name, tensor in tensors.items():
            assert torch.equal(loaded[name], tensor), name

    def test_the_checked_in_fixture_is_current(self, built):
        """追跡対象の fixture が今の台本の出力と同じ（台本だけ直して書き直し忘れを掴む）。"""
        from safetensors.torch import load_file

        directory = scheduler_ref.DEFAULT_OUT
        if not (directory / "unipc.safetensors").is_file():
            pytest.skip(f"{directory} に fixture が無い — `python -m wan.scheduler_ref` で作る")
        tensors, meta = built

        loaded = load_file(str(directory / "unipc.safetensors"))
        assert json.loads((directory / "unipc.json").read_text(encoding="utf-8")) == meta
        for name, tensor in tensors.items():
            assert torch.equal(loaded[name], tensor), name
