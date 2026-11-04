"""The calibration corpus of `irodori/calib_cases.py` and its **separation from the evaluation
inputs**.

The texts themselves are not judged here (measurement answers the quality question). What is
pinned is only what would **change the meaning** of the measurements and rulings if it broke:

- no duplicates, and the speaking styles are spread (concentrating on one style makes the
  rounding bias stick to that style)
- separated from the evaluation inputs (`irodori.pipeline_ref.PIPELINE_CASES`) **in every one of
  text / caption / seed**
- the separation check itself **does not pass vacuously** (it fails under fault injection)
- the entry point of the calibration path ({@link irodori.calib_cases.calibration_cases}) runs
  the check
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
        """If the seeds of the initial noise and the reference latent collide, the two cases see the
        same trajectory."""
        seeds = [seed for case in CALIB_CASES for seed in calib_cases._seeds(case)]

        assert len(set(seeds)) == len(seeds)

    def test_every_case_drives_all_three_conditions(self):
        """All three conditions (text / speaker / caption) are active — every segment of the
        cross-attention is exercised."""
        for case in CALIB_CASES:
            assert case.reference is not None, case.name
            assert case.caption, case.name
            assert case.text, case.name


class TestEvaluationSources:
    def test_it_gathers_the_pipeline_reference_cases(self):
        """The evaluation inputs have one source of truth, `pipeline_ref.PIPELINE_CASES` (no
        copy)."""
        assert evaluation_cases() == PIPELINE_CASES


class TestDisjointness:
    def test_the_shipped_corpus_is_disjoint_from_every_evaluation_input(self):
        assert_calib_disjoint(calibration_cases(), evaluation_cases())

    def test_an_identical_text_is_caught(self):
        clashing = CALIB_CASES[0]._replace(text=PIPELINE_CASES[0].text)

        with pytest.raises(AssertionError, match="bodies overlap the evaluation inputs"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_a_text_contained_in_an_evaluation_input_is_caught(self):
        """Substring matches count too — calibrating on just the first half of an evaluation text
        is also a leak."""
        body = PIPELINE_CASES[0].text
        clashing = CALIB_CASES[0]._replace(text=body[: len(body) // 2])

        with pytest.raises(AssertionError, match="bodies overlap the evaluation inputs"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_a_shared_caption_is_caught(self):
        clashing = CALIB_CASES[0]._replace(caption=PIPELINE_CASES[0].caption)

        with pytest.raises(AssertionError, match="bodies overlap the evaluation inputs"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_an_empty_caption_is_not_treated_as_a_match(self):
        """The evaluation side's `no-ref` has an empty caption — counting the empty string as a
        "substring match" would make every case overlap."""
        assert any(case.caption == "" for case in PIPELINE_CASES), (
            "no evaluation case with an empty caption"
        )

        assert_calib_disjoint(CALIB_CASES, PIPELINE_CASES)

    def test_a_shared_noise_seed_is_caught(self):
        clashing = CALIB_CASES[0]._replace(seed=PIPELINE_CASES[0].seed)

        with pytest.raises(AssertionError, match="seeds overlap the evaluation inputs"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_a_shared_reference_seed_is_caught(self):
        """Even with a different `seed`, the same reference-latent seed shares the activations of
        the speaker segment."""
        reference = PIPELINE_CASES[0].reference
        assert reference is not None, "no evaluation case with a reference"
        clashing = CALIB_CASES[0]._replace(reference=reference)

        with pytest.raises(AssertionError, match="seeds overlap the evaluation inputs"):
            assert_calib_disjoint((clashing,), PIPELINE_CASES)

    def test_the_selection_path_runs_the_check(self, monkeypatch):
        """MUST: the entry point of the calibration path runs the check (a green constant alone
        means nothing)."""
        monkeypatch.setattr(calib_cases, "CALIB_CASES", (PIPELINE_CASES[0],))

        with pytest.raises(AssertionError, match="overlap the evaluation inputs"):
            calibration_cases()

    def test_an_empty_corpus_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(calib_cases, "CALIB_CASES", ())

        with pytest.raises(AssertionError, match="no calibration cases"):
            calibration_cases()
