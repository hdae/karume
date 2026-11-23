"""Properties of the calibration corpus in `embeddinggemma2/calib_texts.py`.

The wording itself is not under judgment (measurement answers for quality). What is pinned here is
only the side that **changes the meaning of the measurements** when it breaks:

- **Not a single text overlaps the evaluation texts** — with an overlap the cosine becomes "did the
  model embed a text the calibration already saw", a number that measures leakage, not the quality
  of the calibration
- Exactly 48 texts with no duplicates (no text is measured twice)
- Every prompt kind is either a key of `export.load_prompts` or `None` (with a misspelling only the
  leading token sequence of the calibration inputs drifts from real use)
- The first few texts already hold the mix of roles (query / document / plain)
"""

from __future__ import annotations

from embeddinggemma2 import export as eg
from embeddinggemma2.calib_texts import CALIB_TEXTS

#: The keys of `prompts` in `config_sentence_transformers.json` (spellings pinned without reading
#: the real asset — the actual lookup in `measure_quant.tokenize` fails on a dict access).
PROMPT_KEYS = {"query", "document"}


class TestCorpus:
    def test_it_holds_forty_eight_distinct_texts(self):
        assert len(CALIB_TEXTS) == 48
        assert len({body for _prompt, body in CALIB_TEXTS}) == 48

    def test_no_text_overlaps_the_golden_cases(self):
        """MUST: calibration and evaluation stay apart (module docstring)."""
        golden = {body for _name, _prompt, body in eg.GOLDEN_CASES}

        assert not golden & {body for _prompt, body in CALIB_TEXTS}

    def test_no_text_quotes_a_golden_case(self):
        """A partial match is still an overlap (checked one direction at a time with `in`)."""
        bodies = [body for _prompt, body in CALIB_TEXTS]
        for _name, _prompt, golden in eg.GOLDEN_CASES:
            assert not any(golden in body or body in golden for body in bodies)

    def test_every_prompt_key_is_spelled_like_the_official_ones(self):
        assert {prompt for prompt, _body in CALIB_TEXTS} <= PROMPT_KEYS | {None}

    def test_all_three_roles_appear_in_the_first_few_texts(self):
        """An order in which the first three texts already have query / document / plain."""
        head = {prompt for prompt, _body in CALIB_TEXTS[:3]}

        assert head == PROMPT_KEYS | {None}

    def test_the_two_query_roles_are_balanced(self):
        """Retrieval uses query and document **asymmetrically** — do not round on the activations of
        only one of them."""
        prompts = [prompt for prompt, _body in CALIB_TEXTS]

        assert prompts.count("query") == prompts.count("document") == 20
