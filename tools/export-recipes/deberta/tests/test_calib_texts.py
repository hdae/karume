"""Properties of the calibration corpus in `deberta/calib_texts.py`.

The wording itself is not judged (measurement answers for quality). What is pinned here is only
what **changes the meaning of the measurements** when it breaks:

- exactly 48 sentences without duplicates (a reduced `--calib-limit` run does not become "the same
  sentence N times")
- the lengths are spread (GPTQ's `H = Σ XᵀX` is weighted by token count, so with uniform lengths
  only the activations of that length dominate)
- the mix of roles survives truncation from the start (the input of a reduced smoke run is not
  narration only)
- not one sentence overlaps the script's own golden sentences
  ({@link deberta.export.GOLDEN_SENTENCES})

Only the last one **can be checked statically** (the golden sentences are constants of the script;
no assets are read). Keeping it disjoint from the evaluation sentences at run time is a separate
matter: there **the assets (the dump) are the source of truth**, so the run-time
{@link sbv2.measure_quant.assert_calib_disjoint} checks it, not this file (a copy on the corpus
side would leave only one side stale when the dump is re-recorded). `test_measure_quant.py` tests
the detection power of that check itself.
"""

from __future__ import annotations

from deberta import export as export_deberta
from deberta.calib_texts import CALIB_TEXTS

#: Kanji numerals used to detect the number-reading role (not passed through the tokenizer — only
#: the presence of the characters is checked).
NUMERALS = "〇一二三四五六七八九十百千万"


def is_question(text: str) -> bool:
    """Whether the sentence is a question (conversational)."""
    return text.endswith("？")


def has_numeral(text: str) -> bool:
    """Whether the sentence contains a number reading."""
    return any(char in NUMERALS for char in text)


class TestCorpus:
    def test_it_holds_forty_eight_distinct_texts(self):
        assert len(CALIB_TEXTS) == 48
        assert len(set(CALIB_TEXTS)) == 48

    def test_no_text_is_a_golden_sentence(self):
        """If a calibration sentence overlaps a golden sentence, the rounding targets are picked
        from "the very sentence being measured"."""
        assert not set(export_deberta.GOLDEN_SENTENCES) & set(CALIB_TEXTS)

    def test_no_text_quotes_a_golden_sentence(self):
        """A partial match is still an overlap (checked with `in` in each direction)."""
        for golden in export_deberta.GOLDEN_SENTENCES:
            assert not any(golden in text or text in golden for text in CALIB_TEXTS)

    def test_the_lengths_are_spread(self):
        """Both short and long sentences are needed (with char-wwm the character count is about the
        token count)."""
        lengths = [len(text) for text in CALIB_TEXTS]

        assert min(lengths) <= 12
        assert max(lengths) >= 40

    def test_every_role_appears_in_the_first_few_texts(self):
        """An order in which even a reduced run (`--calib-limit 4`) covers narration / question /
        number reading."""
        head = CALIB_TEXTS[:4]

        assert any(is_question(text) for text in head)
        assert any(has_numeral(text) for text in head)
        assert any(not is_question(text) and not has_numeral(text) for text in head)

    def test_both_roles_are_well_represented(self):
        """Leaning towards either one pins the bias of the activations to that role's vocabulary
        distribution."""
        questions = sum(1 for text in CALIB_TEXTS if is_question(text))
        numerals = sum(1 for text in CALIB_TEXTS if has_numeral(text))

        assert 8 <= questions <= len(CALIB_TEXTS) - 8
        assert 8 <= numerals <= len(CALIB_TEXTS) - 8
