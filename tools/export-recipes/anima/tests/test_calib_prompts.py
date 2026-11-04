"""The calibration corpus of `anima/calib_prompts.py` and its **separation from the evaluation
inputs**.

The wording itself is not judged (measurement answers for quality). What is pinned here is only
what **changes the meaning** of measurements and rulings when it breaks:

- no duplicates, and the default count fits the corpus (a reduced run does not become "the same
  prompt N times")
- cutting off the head keeps the mix of subjects / styles (`--calib-prompts N` takes the first N)
- separated **even by substring** from the 3 evaluation inputs (`pipeline_ref` / the TS E2E / the
  4 cases of the visual evaluation)
- the separation check itself **does not pass vacuously** (it fails under fault injection)
- fail loudly when extracting the evaluation inputs yields nothing (a separation check that
  silently loosens is the greater danger)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anima import calib_prompts
from anima.calib_prompts import (
    CALIB_PROMPTS,
    DEFAULT_CALIB_PROMPTS,
    assert_calib_disjoint,
    calibration_prompts,
    evaluation_prompts,
)
from anima.pipeline_ref import PROMPT

#: Tags that name a style (used to judge that styles are mixed into the head of the corpus).
STYLE_TAGS = ("chibi", "monochrome", "watercolor", "film grain", "sketch")


class TestCorpus:
    def test_it_holds_distinct_prompts(self):
        assert len(set(CALIB_PROMPTS)) == len(CALIB_PROMPTS)

    def test_the_default_count_fits_the_corpus(self):
        """If the default exceeded the limit, a run that does not touch `--calib-prompts` would fail
        outright."""
        assert 1 <= DEFAULT_CALIB_PROMPTS <= len(CALIB_PROMPTS)

    def test_the_default_selection_keeps_the_mix(self):
        """Only the first N are used, so the default count alone must spread over subjects and
        styles."""
        head = CALIB_PROMPTS[:DEFAULT_CALIB_PROMPTS]

        assert any("no humans" in prompt for prompt in head), "no prompt without people"
        assert any("1girl" in prompt or "1boy" in prompt for prompt in head), (
            "no single-person prompt"
        )
        assert any(prompt.startswith(("2girls", "3girls", "4girls")) for prompt in head), (
            "no multi-person prompt"
        )
        assert any(tag in prompt for prompt in head for tag in STYLE_TAGS), (
            "no prompt that names a style"
        )

    def test_an_out_of_range_count_is_refused(self):
        with pytest.raises(ValueError, match="is out of range"):
            calibration_prompts(0)
        with pytest.raises(ValueError, match="is out of range"):
            calibration_prompts(len(CALIB_PROMPTS) + 1)


class TestEvaluationSources:
    def test_it_gathers_all_three_sources(self):
        """The 2 fixed prompts of `pipeline_ref` + the 1 of the TS E2E + the 4 cases of the visual
        evaluation."""
        evaluated = evaluation_prompts()

        assert PROMPT in evaluated
        assert len(evaluated) >= 7
        assert any(prompt.startswith("score_9") for prompt in evaluated), (
            "eval-images composition missing"
        )

    def test_concatenated_string_literals_are_joined(self):
        """Join a `"…" + "…"` concatenation back into one (taking only fragments would make the
        separation check work only halfway)."""
        joined = calib_prompts._declaration('const X = "ab" +\n  "cd";\n', Path("x.ts"), "X")

        assert joined == "abcd"

    def test_a_missing_source_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(calib_prompts, "E2E_SOURCE", Path("packages/models/tests/gone.ts"))

        with pytest.raises(AssertionError, match="The source of truth of the evaluation input"):
            evaluation_prompts()

    def test_a_renamed_declaration_fails_loudly(self, monkeypatch):
        """Fail when a changed spelling makes the extraction empty (a separation check that silently
        loosens is the greater danger)."""
        monkeypatch.setattr(calib_prompts, "E2E_SOURCE", Path("README.md"))

        with pytest.raises(AssertionError, match="const PROMPT"):
            evaluation_prompts()


class TestDisjointness:
    def test_the_shipped_corpus_is_disjoint_from_every_evaluation_input(self):
        """Neither the default selection nor the whole corpus overlaps any evaluation input."""
        evaluated = evaluation_prompts()

        assert_calib_disjoint(calibration_prompts(DEFAULT_CALIB_PROMPTS), evaluated)
        assert_calib_disjoint(CALIB_PROMPTS, evaluated)

    def test_an_identical_prompt_is_caught(self):
        with pytest.raises(AssertionError, match="overlaps the evaluation inputs"):
            assert_calib_disjoint((PROMPT,), evaluation_prompts())

    def test_a_prompt_contained_in_an_evaluation_input_is_caught(self):
        """Substrings count — calibrating on only the first half of an evaluation prompt is a leak
        too."""
        with pytest.raises(AssertionError, match="overlaps the evaluation inputs"):
            assert_calib_disjoint((PROMPT[: len(PROMPT) // 2],), evaluation_prompts())

    def test_a_prompt_that_contains_an_evaluation_input_is_caught(self):
        with pytest.raises(AssertionError, match="overlaps the evaluation inputs"):
            assert_calib_disjoint((f"{PROMPT}, extra tag",), evaluation_prompts())

    def test_the_selection_path_runs_the_check(self, monkeypatch):
        """MUST: the `--calib-prompts` path goes through the check (green on the constant alone
        means nothing)."""
        monkeypatch.setattr(calib_prompts, "CALIB_PROMPTS", (PROMPT,))

        with pytest.raises(AssertionError, match="overlaps the evaluation inputs"):
            calibration_prompts(1)
