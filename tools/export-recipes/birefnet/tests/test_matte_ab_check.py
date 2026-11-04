"""The contract of `birefnet/matte_ab_check.py` — it checks the A/B tool's α against the same
contract as the golden.

The tool itself (Deno, GPU) is not run. A `work/` of the same shape the tool writes (manifest,
logits, α per procedure) is built with the upstream functions, and the test checks that an α that
follows the contract passes and that an α off the contract by even one pixel fails by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from birefnet import matte_ab_check, matte_fixture

pytest.importorskip("PIL")

#: The size of the logit map, and the target size the tool resizes to (an upscale).
SOURCE = (24, 20)
TARGET = (37, 31)


def _logits() -> np.ndarray:
    """Logits with an edge (a disc — saturated at the centre, background outside)."""
    height, width = SOURCE[1], SOURCE[0]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(xx - 11.3, yy - 9.6)
    return (1.7 * (7.2 - distance)).astype(np.float32)


def _alphas(logits: np.ndarray) -> dict[str, np.ndarray]:
    """The α the contract expects (the upstream functions themselves)."""
    width, height = TARGET
    return {
        # The tool (TS) also rounds ties when it writes, so the stand-in does not refuse ties
        # either (without ties the result is the same).
        matte_ab_check.UPSTREAM: matte_fixture.hr_alpha(
            logits, width, height, allow_ties=True
        ).copy(),
        matte_ab_check.CURRENT: matte_fixture.round_half_up(
            matte_fixture.lucida_scaled(logits, width, height)
        ),
    }


def _write_work(
    directory: Path,
    logits: np.ndarray,
    alphas: dict[str, np.ndarray],
    graph: tuple[int, int] = SOURCE,
) -> Path:
    from safetensors.numpy import save_file

    save_file({"add_214": logits[None, None]}, directory / "subject.logits-1024.safetensors")
    save_file(alphas, directory / "subject.alpha.safetensors")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "graph": {"width": graph[0], "height": graph[1]},
                "entries": [
                    {
                        "name": "subject",
                        "width": TARGET[0],
                        "height": TARGET[1],
                        "logits": "subject.logits-1024.safetensors",
                        "alpha": "subject.alpha.safetensors",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return directory


class TestTheCheck:
    def test_alphas_made_by_the_upstream_functions_pass(self, tmp_path):
        logits = _logits()
        report = matte_ab_check.check_work(_write_work(tmp_path, logits, _alphas(logits)))

        assert matte_ab_check.violations(report) == []
        assert report["subject"][matte_ab_check.UPSTREAM] == {
            "mismatches": 0,
            "excludedTiePixels": 0,
            "ok": True,
        }

    def test_one_pixel_off_in_the_upstream_procedure_is_named(self, tmp_path):
        logits = _logits()
        alphas = _alphas(logits)
        alphas[matte_ab_check.UPSTREAM][15, 18] ^= 1

        report = matte_ab_check.check_work(_write_work(tmp_path, logits, alphas))

        assert matte_ab_check.violations(report) == [f"subject/{matte_ab_check.UPSTREAM}"]
        assert report["subject"][matte_ab_check.UPSTREAM]["mismatches"] == 1

    def test_a_step_away_from_a_rounding_boundary_is_outside_epsilon(self, tmp_path):
        logits = _logits()
        alphas = _alphas(logits)
        width, height = TARGET
        scaled = matte_fixture.lucida_scaled(logits, width, height).astype(np.float64)
        # Move the pixel farthest from a rounding boundary (largest |255α − (k + 0.5)|) by one
        # level — ε cannot excuse it.
        distance = np.abs(scaled - (np.floor(scaled) + 0.5))
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        current = alphas[matte_ab_check.CURRENT]
        current[y, x] = current[y, x] + 1 if current[y, x] < 255 else current[y, x] - 1

        report = matte_ab_check.check_work(_write_work(tmp_path, logits, alphas))

        assert matte_ab_check.violations(report) == [f"subject/{matte_ab_check.CURRENT}"]
        assert report["subject"][matte_ab_check.CURRENT]["outsideEpsilon"] == 1

    def test_a_missing_procedure_fails_loudly(self, tmp_path):
        logits = _logits()
        alphas = _alphas(logits)
        del alphas[matte_ab_check.CURRENT]

        with pytest.raises(ValueError, match="no α for the procedures"):
            matte_ab_check.check_work(_write_work(tmp_path, logits, alphas))

    def test_logits_whose_shape_differs_from_the_manifest_graph_fail_loudly(self, tmp_path):
        logits = _logits()
        work = _write_work(tmp_path, logits, _alphas(logits), graph=(SOURCE[1], SOURCE[0]))

        with pytest.raises(
            ValueError, match=r"logit shape \[1, 1, 20, 24\] differs from the manifest's graph"
        ):
            matte_ab_check.check_work(work)


#: Where the tied sample goes (row, column — inside the saturated disc). Logit 0 → 255σ = exactly
#: 127.5.
TIE_AT = (10, 12)


def _logits_with_a_tie() -> np.ndarray:
    logits = _logits()
    logits[TIE_AT] = 0.0
    return logits


class TestTies:
    """A tied sample does not skip the whole image: only the pixels the tie reaches are excluded,
    and the rest are compared for zero mismatches."""

    def test_the_image_is_checked_with_only_the_reach_of_the_tie_excluded(self, tmp_path):
        logits = _logits_with_a_tie()
        report = matte_ab_check.check_work(_write_work(tmp_path, logits, _alphas(logits)))

        result = report["subject"][matte_ab_check.UPSTREAM]
        assert matte_ab_check.violations(report) == []
        assert result["mismatches"] == 0
        # One sample reaches only a few pixels per axis: the support width (4 input samples)
        # divided by the scale factor.
        assert 0 < result["excludedTiePixels"] < TARGET[0] * TARGET[1] // 4

    @pytest.mark.parametrize(
        "target",
        [TARGET, (13, 11), (SOURCE[0], 31)],
        ids=["upscale", "downscale", "same-width"],
    )
    def test_the_excluded_pixels_cover_every_pixel_that_depends_on_the_tie(self, target):
        # The difference between two images with the tied sample's 8-bit value set to 0 and to
        # 255 (every pixel that depends on that sample) lies inside the excluded pixels. A
        # one-level rounding difference does not show the small weights at the edge of the
        # support in 8 bits, so a support copy that is too narrow would slip through; hence the
        # widest swing. The surroundings are a mid-value plane — with saturated surroundings the
        # negative lobes' difference clamps to 0 / 255 and disappears.
        logits = np.full((SOURCE[1], SOURCE[0]), 0.3, dtype=np.float32)
        logits[TIE_AT] = 0.0
        width, height = target
        ties = matte_ab_check.tie_samples(logits)
        assert ties.sum() == 1

        def forced(value: int):
            def quantize(scaled: np.ndarray) -> np.ndarray:
                plane = matte_fixture.round_half_up(scaled)
                plane[ties] = value
                return plane

            return matte_fixture.hr_alpha(logits, width, height, quantize=quantize)

        changed = forced(0) != forced(255)
        excluded = matte_ab_check.tie_reach(ties, width, height)

        assert changed.any()
        assert not (changed & ~excluded).any()

    def test_a_mismatch_outside_the_reach_of_the_tie_is_still_named(self, tmp_path):
        logits = _logits_with_a_tie()
        alphas = _alphas(logits)
        width, height = TARGET
        excluded = matte_ab_check.tie_reach(matte_ab_check.tie_samples(logits), width, height)
        inside = tuple(np.argwhere(excluded)[0])
        outside = tuple(np.argwhere(~excluded)[0])
        alphas[matte_ab_check.UPSTREAM][inside] ^= 1
        alphas[matte_ab_check.UPSTREAM][outside] ^= 1

        report = matte_ab_check.check_work(_write_work(tmp_path, logits, alphas))

        assert matte_ab_check.violations(report) == [f"subject/{matte_ab_check.UPSTREAM}"]
        assert report["subject"][matte_ab_check.UPSTREAM]["mismatches"] == 1

    def test_the_reach_of_an_axis_of_the_same_size_is_the_sample_itself(self):
        np.testing.assert_array_equal(matte_ab_check.bicubic_reach(5, 5), np.eye(5, dtype=bool))
