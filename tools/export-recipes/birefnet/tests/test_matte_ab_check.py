"""`birefnet/matte_ab_check.py` の約束事 — A/B の道具の α を、golden と同じ契約で検算すること。

道具そのもの（Deno・GPU）は回さない。道具が書くのと同じ形の `work/`（manifest・logit・手順ごとの
α）を上流の関数で作り、契約どおりの α は通り、1 画素でも契約から外れた α は名指しで落ちることを
見る。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from birefnet import matte_ab_check, matte_fixture

pytest.importorskip("PIL")

#: logit の地図の寸法と、道具が当てる目標寸法（拡大）。
SOURCE = (24, 20)
TARGET = (37, 31)


def _logits() -> np.ndarray:
    """輪郭を持つ logit（円盤 — 中心は飽和・外側は背景）。"""
    height, width = SOURCE[1], SOURCE[0]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(xx - 11.3, yy - 9.6)
    return (1.7 * (7.2 - distance)).astype(np.float32)


def _alphas(logits: np.ndarray) -> dict[str, np.ndarray]:
    """契約どおりの α（上流の関数そのもの）。"""
    width, height = TARGET
    return {
        matte_ab_check.UPSTREAM: matte_fixture.hr_alpha(logits, width, height).copy(),
        matte_ab_check.CURRENT: matte_fixture.round_half_up(
            matte_fixture.lucida_scaled(logits, width, height)
        ),
    }


def _write_work(directory: Path, logits: np.ndarray, alphas: dict[str, np.ndarray]) -> Path:
    from safetensors.numpy import save_file

    save_file({"add_214": logits[None, None]}, directory / "subject.logits-1024.safetensors")
    save_file(alphas, directory / "subject.alpha.safetensors")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "graph": {"width": SOURCE[0], "height": SOURCE[1]},
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
        assert report["subject"][matte_ab_check.UPSTREAM] == {"mismatches": 0, "ok": True}

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
        # 丸めの境目から最も遠い画素（|255α − (k + 0.5)| が最大）を 1 段ずらす — ε では救えない。
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

        with pytest.raises(ValueError, match="手順の α が無い"):
            matte_ab_check.check_work(_write_work(tmp_path, logits, alphas))
