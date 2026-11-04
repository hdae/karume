"""Check that the DPM++ 2M golden rig **really runs the real class on karume's sigma ladder**.

This rig's output is inlined in `packages/models/tests/anima_dpm_solver_multistep_test.ts`, so if
the rig silently slides onto another ladder or another configuration, **only the TS side turns red
and the cause cannot be read**. This pins that the substitution takes effect (and that it is
needed = the ladder differs from the built-in flow ladder). The numeric source of truth is this
rig, not the TS side.
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
        """MUST: bit-identical. The ladder plugs in the values of `sigma_schedule` themselves, not a
        copy."""
        sigmas = sigma_schedule(STEPS, SHIFT)

        scheduler = build_scheduler(sigmas, SHIFT)

        assert np.array_equal(
            scheduler.sigmas.numpy().view(np.uint32),
            sigmas.view(np.uint32),
        )

    def test_the_builtin_flow_ladder_is_a_different_one(self) -> None:
        """Guard against a vacuous test: show in numbers why the substitution is needed (it differs
        from the built-in flow ladder).

        Once they match, the substitution is no longer needed — then fold the rig's MUST section
        away with it.
        """
        sigmas = sigma_schedule(STEPS, SHIFT)

        native = native_flow_sigmas(STEPS, SHIFT)

        assert native.shape == sigmas.shape
        assert not np.array_equal(native, sigmas)
        assert float(np.max(np.abs(native - sigmas))) == pytest.approx(3.752112e-04, rel=1e-4)


class TestSeries:
    def test_the_last_step_returns_the_x0_prediction_unchanged(self) -> None:
        """The drop to first order at the terminal σ=0 (coefficients `(0, −1)`) — pairs with the TS
        test of the same name."""
        series = run(STEPS, SHIFT, elements=4, seed=1)

        last = series["records"][-1]
        assert last["coefficients"]["order"] == 1
        assert last["sample"] == last["x0"]

    def test_only_the_first_and_last_steps_drop_to_first_order(self) -> None:
        """With `solver_order=2`, `lower_order_second` is never reached (the branch structure of
        step())."""
        series = run(STEPS, SHIFT, elements=2, seed=1)

        orders = [record["coefficients"]["order"] for record in series["records"]]
        assert orders == [1, 2, 2, 2, 2, 2, 2, 1]

    def test_the_series_is_reproducible_from_the_seed(self) -> None:
        """The premise of inlining the golden — the same seed gives the same numbers (torch covers
        environment differences separately)."""
        first = run(4, SHIFT, elements=3, seed=99)
        second = run(4, SHIFT, elements=3, seed=99)

        assert first == second
