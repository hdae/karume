"""Properties of the calibration corpus in `minicpm5/calib_texts.py`.

The wording itself is not judged here (measurement answers the quality question). What is pinned
is only what **changes the meaning of the measurements** when it breaks:

- **Not one text overlaps the evaluation texts** — with an overlap, the teacher / greedy agreement
  becomes "whether the run reproduced sentences seen in calibration", a number that measures
  leakage rather than the quality of the calibration
- Exactly 48 texts without duplicates (a reduced `--calib-limit` run does not become "the same text
  N times")
- Cutting the head keeps the mix of languages and styles (the input of a reduced smoke run does not
  become English only)
"""

from __future__ import annotations

from minicpm5 import export as one_shot
from minicpm5.calib_texts import CALIB_TEXTS

#: How code fragments start (prose never starts with these spellings).
CODE_PREFIXES = ("def ", "SELECT", "impl ", "$ ")


def has_japanese(text: str) -> bool:
    """Whether the text contains at least one kana or kanji character (for the language ratio — no
    tokenizer involved)."""
    return any("぀" <= char <= "ヿ" or "一" <= char <= "鿿" for char in text)


def is_code(text: str) -> bool:
    """Whether the text is a code fragment (it has a newline = several lines, or starts like a
    declaration, a query or a shell command)."""
    return "\n" in text or text.startswith(CODE_PREFIXES)


class TestCorpus:
    def test_it_holds_forty_eight_distinct_texts(self):
        assert len(CALIB_TEXTS) == 48
        assert len(set(CALIB_TEXTS)) == 48

    def test_no_text_overlaps_the_golden_cases(self):
        """MUST: calibration and evaluation stay apart (module docstring)."""
        golden = {body for _name, body in one_shot.GOLDEN_CASES}

        assert not golden & set(CALIB_TEXTS)

    def test_no_text_quotes_a_golden_case(self):
        """A partial match is still an overlap (checked with `in` in each direction)."""
        for _name, body in one_shot.GOLDEN_CASES:
            assert not any(body in text or text in body for text in CALIB_TEXTS)

    def test_both_languages_appear_in_the_first_few_texts(self):
        """An order in which a reduced run (`--calib-limit 4`) still has both English and
        Japanese."""
        head = CALIB_TEXTS[:4]

        assert any(has_japanese(text) for text in head)
        assert any(not has_japanese(text) for text in head)

    def test_a_code_fragment_appears_in_the_first_few_texts(self):
        """The third of the three kinds the docstring declares — reordering does not push code to
        the fifth text or later."""
        assert any(is_code(text) for text in CALIB_TEXTS[:4])

    def test_the_code_fragments_are_well_represented(self):
        """With a single code fragment, only that language's symbols would represent the
        activations."""
        assert sum(1 for text in CALIB_TEXTS if is_code(text)) == 6

    def test_the_two_languages_are_both_well_represented(self):
        """Leaning towards either English or Japanese pins the activation bias to that language's
        token distribution."""
        japanese = sum(1 for text in CALIB_TEXTS if has_japanese(text))

        assert 10 <= japanese <= len(CALIB_TEXTS) - 10
