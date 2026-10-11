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
        # 道具（TS）は同点も丸めて書くので、代役も同点を拒まない（同点が無ければ結果は同じ）。
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

    def test_logits_whose_shape_differs_from_the_manifest_graph_fail_loudly(self, tmp_path):
        logits = _logits()
        work = _write_work(tmp_path, logits, _alphas(logits), graph=(SOURCE[1], SOURCE[0]))

        with pytest.raises(ValueError, match=r"logit の形 \[1, 1, 20, 24\] が manifest の graph"):
            matte_ab_check.check_work(work)


#: 同点の標本を置く位置（行, 列 — 円盤の飽和した内側）。logit 0 → 255σ = 127.5 ちょうど。
TIE_AT = (10, 12)


def _logits_with_a_tie() -> np.ndarray:
    logits = _logits()
    logits[TIE_AT] = 0.0
    return logits


class TestTies:
    """同点の標本があっても画像ごと飛ばさず、同点が届く画素だけを外して 0 不一致で比べる。"""

    def test_the_image_is_checked_with_only_the_reach_of_the_tie_excluded(self, tmp_path):
        logits = _logits_with_a_tie()
        report = matte_ab_check.check_work(_write_work(tmp_path, logits, _alphas(logits)))

        result = report["subject"][matte_ab_check.UPSTREAM]
        assert matte_ab_check.violations(report) == []
        assert result["mismatches"] == 0
        # 1 標本が届くのは各軸で台の幅（入力 4 標本）を倍率で割った数画素だけ。
        assert 0 < result["excludedTiePixels"] < TARGET[0] * TARGET[1] // 4

    @pytest.mark.parametrize(
        "target",
        [TARGET, (13, 11), (SOURCE[0], 31)],
        ids=["upscale", "downscale", "same-width"],
    )
    def test_the_excluded_pixels_cover_every_pixel_that_depends_on_the_tie(self, target):
        # 同点の標本の 8bit 値を 0 と 255 に振った 2 枚の差（その標本に依る画素の全体）が、外す
        # 画素の内側に収まる。丸めの 1 段の差では台の端の小さい重みが 8bit に出ず、台の写しが
        # 狭すぎても見逃すので、振り幅を最大にする。周りは中間値の平面にする — 飽和した周りでは
        # 負のローブの差が 0 / 255 に張り付いて消える。
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
