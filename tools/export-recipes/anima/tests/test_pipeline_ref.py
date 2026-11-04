"""The contracts of the part of `anima/pipeline_ref.py` that **needs no real weights**.

Only the side where a breakage raises nothing at run time and **only the numbers silently drift**
is pinned here:

- the shape of the sigma schedule (shift really takes effect / terminal 0 / strictly decreasing /
  f32)
- the channel axis of denormalization (axis 1 of rank 5 — a mix-up still matches the element
  count, so it slips past shape checks)
- the prompt constants stay inside the conditioner's accepted set (`Dim("Ttgt", min=2)`)

The emit / fixture generation that needs real weights is manual (see the README).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anima import pipeline_ref as anima_pipeline


class TestSigmaSchedule:
    def test_shape_and_terminal_zero(self):
        """`step()` reads `sigmas[i+1]`, so one extra terminal 0 is needed."""
        sigmas = anima_pipeline.sigma_schedule(32, anima_pipeline.SHIFT)

        assert sigmas.dtype == np.float32
        assert len(sigmas) == 33
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas[-1] == 0.0

    def test_strictly_decreasing(self):
        sigmas = anima_pipeline.sigma_schedule(32, anima_pipeline.SHIFT)

        assert np.all(np.diff(sigmas) < 0)

    def test_shift_bends_the_schedule_upward(self):
        """shift>1 **pushes sigma up** (`shift·s/(1+(shift−1)s) ≥ s`).

        Errors such as swapping multiplication and division / applying shift to the denominator
        only flip this direction and fail. The end points (s=1 and the terminal 0) do not move
        with shift, so only the interior is checked.
        """
        plain = anima_pipeline.sigma_schedule(32, 1.0)
        shifted = anima_pipeline.sigma_schedule(32, 3.0)

        assert np.all(shifted[1:-1] > plain[1:-1])
        assert shifted[0] == plain[0]

    def test_two_steps_is_the_smallest_usable_schedule(self):
        """`linspace` divides by zero with num=1. This exercises the lower bound of 2."""
        sigmas = anima_pipeline.sigma_schedule(2, anima_pipeline.SHIFT)

        assert len(sigmas) == 3
        assert np.all(np.isfinite(sigmas))


class _FakeVaeConfig:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.latents_mean = mean
        self.latents_std = std
        self.z_dim = len(mean)


class _FakeVae:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.config = _FakeVaeConfig(mean, std)


class TestDenormalizeLatents:
    def test_channel_axis_is_one(self):
        """MUST: the per-channel constants apply to the channel axis (axis 1 of rank 5).

        A mixed-up axis still matches the element count (the broadcast goes through), so it is
        detected only by handing out **a different value per channel**.
        """
        vae = _FakeVae([0.0, 10.0], [1.0, 1.0])
        latents = torch.zeros(1, 2, 3, 3)

        out = anima_pipeline.denormalize_latents(vae, latents)

        assert list(out.shape) == [1, 2, 1, 3, 3]
        assert torch.all(out[:, 0] == 0.0)
        assert torch.all(out[:, 1] == 10.0)

    def test_std_scales_before_the_mean_is_added(self):
        """`z/(1/std) + mean` — an implementation that adds mean first also scales mean and
        drifts."""
        vae = _FakeVae([1.0], [4.0])
        latents = torch.full((1, 1, 1, 1), 2.0)

        out = anima_pipeline.denormalize_latents(vae, latents)

        assert out.item() == pytest.approx(9.0)


class TestPromptConstants:
    def test_neither_prompt_is_empty(self):
        """An empty prompt's T5 id sequence has length 1, outside the conditioner's
        `Dim(min=2)`."""
        assert anima_pipeline.PROMPT.strip() != ""
        assert anima_pipeline.NEGATIVE_PROMPT.strip() != ""


class TestComponentDtype:
    """The per-component storage dtype of each series (ADR 0019 — one to one with the asset
    series).

    If they drift, "only the fixture's reference holds another model's numbers", and the
    difference of the end-to-end chain turns into a mix of quantization error and implementation
    error (which only ever pushes toward loosening the tolerance).
    """

    LABELS = ("text_encoder", "text_conditioner", "transformer", "vae")

    @pytest.mark.parametrize("label", LABELS)
    def test_f32_series_rounds_nothing(self, label):
        assert anima_pipeline.component_dtype("f32", label) == "f32"

    @pytest.mark.parametrize("label", LABELS)
    def test_f16_series_rounds_every_component(self, label):
        assert anima_pipeline.component_dtype("f16", label) == "f16"

    @pytest.mark.parametrize("label", LABELS)
    def test_bf16_series_rounds_every_component_to_bf16(self, label):
        """The faithful preset is bf16 for all four roles (ADR 0131 decision 3 — the assets are
        `anima-bf16/`)."""
        assert anima_pipeline.component_dtype("bf16", label) == "bf16"

    def test_i8_series_quantizes_only_the_transformer(self):
        """The assets are `outputs/series/anima-i8/transformer` + the other three of
        `outputs/series/anima-f16/`."""
        assert anima_pipeline.component_dtype("i8", "transformer") == "i8"
        for label in ("text_encoder", "text_conditioner", "vae"):
            assert anima_pipeline.component_dtype("i8", label) == "f16", label

    def test_every_series_has_its_own_output_root(self):
        roots = anima_pipeline.DEFAULT_OUTS

        assert set(roots) == set(anima_pipeline.COMPONENT_DTYPES)
        assert len(set(roots.values())) == len(roots)

    def test_the_cli_only_accepts_series_that_have_a_seat_in_every_table(self):
        """The accepted set = the key set of the three tables. If they drift, `--dtype` falls
        through to a bare KeyError.

        Wiring `export.py`'s storage dtypes (including series without a reference path, such as
        i4) straight into argparse's choices would run as far as loading the text_encoder and
        then die without a single diagnostic.
        """
        accepted = set(anima_pipeline.REF_DTYPES)

        assert accepted == set(anima_pipeline.COMPONENT_DTYPES)
        assert accepted == set(anima_pipeline.DEFAULT_OUTS)
        assert accepted == set(anima_pipeline._FALLBACK_DTYPE)
