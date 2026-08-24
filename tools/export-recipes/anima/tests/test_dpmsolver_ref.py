"""DPM++ 2M の golden リグが**本当に karume の梯子の上で実クラスを回している**ことの門。

このリグの出力は `packages/models/tests/dpm_solver_multistep_test.ts` に inline されている
ので、リグが黙って別の梯子・別の構成へ滑ると **TS 側だけが赤くなり原因が読めない**。差し替えが
効いていること（と、差し替えが必要であること = 内蔵 flow 梯子とは別物であること）をここで固定
する。数値の正本は TS 側ではなくこのリグ。
"""

from __future__ import annotations

import numpy as np
import pytest

from anima.dpmsolver_ref import build_scheduler, native_flow_sigmas, run
from anima.pipeline_ref import SHIFT, sigma_schedule

pytest.importorskip("diffusers")

STEPS = 8


class TestLadderSubstitution:
    def test_the_scheduler_runs_on_the_karume_ladder_bit_for_bit(self) -> None:
        """MUST: ビット一致。梯子は写しではなく `sigma_schedule` の値そのものを差し込む。"""
        sigmas = sigma_schedule(STEPS, SHIFT)

        scheduler = build_scheduler(sigmas, SHIFT)

        assert np.array_equal(
            scheduler.sigmas.numpy().view(np.uint32),
            sigmas.view(np.uint32),
        )

    def test_the_builtin_flow_ladder_is_a_different_one(self) -> None:
        """恒真化の門: 差し替えが要る理由（内蔵 flow 梯子と別物）を数で示す。

        一致するようになったら差し替えは不要になる — その時はリグの MUST 節ごと畳む。
        """
        sigmas = sigma_schedule(STEPS, SHIFT)

        native = native_flow_sigmas(STEPS, SHIFT)

        assert native.shape == sigmas.shape
        assert not np.array_equal(native, sigmas)
        assert float(np.max(np.abs(native - sigmas))) == pytest.approx(3.752112e-04, rel=1e-4)


class TestSeries:
    def test_the_last_step_returns_the_x0_prediction_unchanged(self) -> None:
        """終端 σ=0 の 1 次落ち（係数が `(0, −1)`）— TS 側の同名テストと対。"""
        series = run(STEPS, SHIFT, elements=4, seed=1)

        last = series["records"][-1]
        assert last["coefficients"]["order"] == 1
        assert last["sample"] == last["x0"]

    def test_only_the_first_and_last_steps_drop_to_first_order(self) -> None:
        """`solver_order=2` では `lower_order_second` に到達しない（step() の分岐構造）。"""
        series = run(STEPS, SHIFT, elements=2, seed=1)

        orders = [record["coefficients"]["order"] for record in series["records"]]
        assert orders == [1, 2, 2, 2, 2, 2, 2, 1]

    def test_the_series_is_reproducible_from_the_seed(self) -> None:
        """golden を inline する前提 — 同じ seed なら同じ数（環境差は別途 torch が担保）。"""
        first = run(4, SHIFT, elements=3, seed=99)
        second = run(4, SHIFT, elements=3, seed=99)

        assert first == second
