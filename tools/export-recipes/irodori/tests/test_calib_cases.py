"""`irodori/calib_cases.py` の校正コーパスと**評価入力からの分離**。

本文そのものは判断の対象にしない（品質は実測が答える）。ここで固定するのは、崩れると
測定・裁定の**意味が変わる**側だけ:

- 重複が無く、話体が散っている（1 つの型へ寄せると丸めの偏りがその型に張り付く）
- 評価入力（`irodori.pipeline_ref.PIPELINE_CASES`）から **text / caption / seed のどれでも**
  分離している
- 分離検査そのものが**素通りしない**（フォールト注入で落ちること）
- 校正経路の入口（{@link irodori.calib_cases.calibration_cases}）が検査を通ること
"""

from __future__ import annotations

import pytest

from irodori import calib_cases
from irodori.calib_cases import (
    CALIB_CASES,
    assert_calib_disjoint,
    calibration_cases,
    evaluation_cases,
)
from irodori.pipeline_ref import PIPELINE_CASES


class TestCorpus:
    def test_it_holds_distinct_cases(self):
        assert len({case.name for case in CALIB_CASES}) == len(CALIB_CASES)
        assert len({case.text for case in CALIB_CASES}) == len(CALIB_CASES)
        assert len({case.caption for case in CALIB_CASES}) == len(CALIB_CASES)

    def test_every_seed_is_distinct(self):
        """初期ノイズと参照 latent の種が被ると、その 2 ケースは同じ軌道を見る。"""
        seeds = [seed for case in CALIB_CASES for seed in calib_cases._seeds(case)]

        assert len(set(seeds)) == len(seeds)

    def test_every_case_drives_all_three_conditions(self):
        """条件 3 本（text / speaker / caption）とも有効 — 交差 attention の全区間が動く。"""
        for case in CALIB_CASES:
            assert case.reference is not None, case.name
            assert case.caption, case.name
            assert case.text, case.name


class TestEvaluationSources:
    def test_it_gathers_the_pipeline_reference_cases(self):
        """評価入力の正本は `pipeline_ref.PIPELINE_CASES` 1 箇所（写しを持たない）。"""
        assert evaluation_cases() == PIPELINE_CASES


class TestDisjointness:
    def test_the_shipped_corpus_is_disjoint_from_every_evaluation_input(self):
        assert_calib_disjoint(calibration_cases(), evaluation_cases())

    def test_an_identical_text_is_caught(self):
        clashing = CALIB_CASES[0]._replace(text=PIPELINE_CASES[0].text)

        with pytest.raises(AssertionError, match="本文が評価入力と重なっている"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_a_text_contained_in_an_evaluation_input_is_caught(self):
        """部分一致まで見る — 評価文の前半だけを校正に使うのも漏れ。"""
        body = PIPELINE_CASES[0].text
        clashing = CALIB_CASES[0]._replace(text=body[: len(body) // 2])

        with pytest.raises(AssertionError, match="本文が評価入力と重なっている"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_a_shared_caption_is_caught(self):
        clashing = CALIB_CASES[0]._replace(caption=PIPELINE_CASES[0].caption)

        with pytest.raises(AssertionError, match="本文が評価入力と重なっている"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_an_empty_caption_is_not_treated_as_a_match(self):
        """評価側の `no-ref` は caption 空 — 空文字を「部分一致」に数えると全件が重なる。"""
        assert any(case.caption == "" for case in PIPELINE_CASES), "空 caption の評価ケースが無い"

        assert_calib_disjoint(CALIB_CASES, PIPELINE_CASES)

    def test_a_shared_noise_seed_is_caught(self):
        clashing = CALIB_CASES[0]._replace(seed=PIPELINE_CASES[0].seed)

        with pytest.raises(AssertionError, match="乱数種が評価入力と重なっている"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_a_shared_reference_seed_is_caught(self):
        """`seed` を変えても参照 latent の種が同じなら、speaker 区間の活性は共有される。"""
        reference = PIPELINE_CASES[0].reference
        assert reference is not None, "参照ありの評価ケースが無い"
        clashing = CALIB_CASES[0]._replace(reference=reference)

        with pytest.raises(AssertionError, match="乱数種が評価入力と重なっている"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_the_selection_path_runs_the_check(self, monkeypatch):
        """MUST: 校正経路の入口が検査を通ること（定数だけ緑でも意味が無い）。"""
        monkeypatch.setattr(calib_cases, "CALIB_CASES", (PIPELINE_CASES[0],))

        with pytest.raises(AssertionError, match="評価入力と重なっている"):
            calibration_cases()

    def test_an_empty_corpus_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(calib_cases, "CALIB_CASES", ())

        with pytest.raises(AssertionError, match="校正ケースが 1 件も無い"):
            calibration_cases()
