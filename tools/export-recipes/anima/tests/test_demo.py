"""The rules of `anima/demo.py` / `anima.text` / `_shared.qwen2_tokenizer`.

When this layer breaks, **only the ids silently become something else** (the shapes still match
while the image changes). Two kinds of things are pinned here:

- **Re-checks against the committed fixture** (run when the HF assets are present, SKIP
  otherwise): both `AutoTokenizer` (the source of truth) and the Python reference implementation
  (= the mirror of the TS implementation) reproduce the fixture's ids. The point of this is
  **noticing a change in upstream's tokenizer.json without regenerating**; it runs at a different
  time than the emit-side check (`build_cases`).
- **The parts that do not depend on assets** (always run) — looking up the baked tables, splitting
  out added tokens, the pre-token scan, and the script and the fixture having the same set of
  cases.

The exhaustive checks (the pre-token scan over all 1,112,064 code points × 11 contexts; the
normalization folding over every cp + 200,000 random samples) are **checks that always run on
regeneration** (`anima_demo.build_assets`); they take minutes, so they are not in pytest.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from _shared import qwen2_tokenizer as q2
from anima import demo as anima_demo
from anima import pipeline_ref as anima_pipeline
from anima import text as at

#: Read the location the script writes to as is (the location's spelling is not derived
#: independently in 2 places — keeping our own `parents[N]` of a different depth would make one
#: of them point to a different root the day the module moves one level).
FIXTURE_PATH = anima_demo.DEFAULT_FIXTURE_PATH
FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

TABLES = at.SpmTables.from_json(FIXTURE["t5"]["normalizer"])
CLASSES: dict[str, q2.Ranges] = FIXTURE["qwen"]["classes"]
CASE_FOLD: dict[int, int] = {cp: target for cp, target in FIXTURE["qwen"]["caseFold"]}
NFC_SEGMENTS: q2.Ranges = FIXTURE["qwen"]["nfcSegments"]


class TestFixtureShape:
    def test_the_fixture_is_not_empty(self):
        """If a mix-up / failed generation leaves cases empty, the TS side's loop runs 0 times."""
        assert len(FIXTURE["cases"]) == len(anima_demo.PROMPT_CASES)
        assert FIXTURE["maxLength"] == anima_demo.MAX_LENGTH

    def test_the_committed_cases_match_the_script(self):
        """The fixture is not stale relative to the script (both the id and the text are
        checked)."""
        want = [(name, text) for name, text, _ in anima_demo.PROMPT_CASES]
        got = [(case["id"], case["text"]) for case in FIXTURE["cases"]]

        assert got == want

    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in FIXTURE["cases"]]

        assert len(set(ids)) == len(ids)

    def test_the_t5_search_width_comes_from_the_whole_vocabulary(self):
        """MUST: `minScore` / `maxTokenLength` are values of the **whole** vocabulary (not derived
        from the subset).

        Computing them from the subset (124 entries, measured) changes the unknown-node weight and
        the search width, and gives a different segmentation. Values from the whole vocabulary
        exceed the subset's minimum / longest **strictly** (allowing equality would let the
        mistake "emit computes them from the subset" pass too).
        """
        subset = FIXTURE["t5"]["vocab"]

        assert FIXTURE["t5"]["minScore"] < min(score for _, _, score in subset)
        assert FIXTURE["t5"]["maxTokenLength"] > max(len(token) for token, _, _ in subset)

    def test_the_pipeline_prompts_are_covered_by_the_cases(self):
        """The 2 prompts of `anima_pipeline` are what the Deno-side cross-check looks up in the
        vocabulary subset.

        If this breaks, that side fails in the hard-to-read form "a token not in the
        vocabulary".
        """
        texts = {case["text"] for case in FIXTURE["cases"]}

        assert anima_pipeline.PROMPT in texts
        assert anima_pipeline.NEGATIVE_PROMPT in texts


class TestNormalizerTables:
    """Looking up the baked tables (Precompiled's shortest prefix wins).

    NOTE: this is not a comparison with an independent source of truth but a regression check
    against committed frozen values (regenerating the tables and the cases moves both sides at
    once, so after the regeneration procedure this class is always green).
    """

    def test_every_case_reproduces_the_recorded_normalization(self):
        for case in FIXTURE["cases"]:
            assert at.normalize_with_tables(TABLES, case["text"]) == case["t5Normalized"], case[
                "id"
            ]

    def test_the_shortest_matching_prefix_wins_and_the_rest_is_dropped(self):
        # A + U+0301 has a rule, so Á. Adding one more keeps Á (the 3rd character is dropped).
        assert at.normalize_with_tables(TABLES, "Á") == "Á"
        assert at.normalize_with_tables(TABLES, "Á́") == "Á"
        # The 2cp prefix Â wins, not the 3cp rule Ấ (a longest-match implementation fails
        # here).
        assert at.normalize_with_tables(TABLES, "Ấ") == "Â"

    def test_clusters_of_six_bytes_or_more_skip_the_whole_replacement(self):
        text = "Ấ́"  # 1 + 2 + 2 + 2 = 7 bytes

        assert at.normalize_with_tables(TABLES, text) == text

    def test_a_cluster_of_exactly_six_bytes_still_skips_the_whole_replacement(self):
        """The boundary itself (turning `< 6` into `<= 6` silently drops the combining characters).

        U+00A0(2) + U+0300(2) + U+0300(2) = exactly 6 bytes. With only the 7-byte case above,
        this one-character rewrite would slip through (the same holds on the TS side — measured
        by fault injection).
        """
        text = "\u00a0\u0300\u0300"

        assert at.normalize_with_tables(TABLES, text) == " \u0300\u0300"

    def test_crlf_is_one_cluster_but_two_newlines_are_two(self):
        assert at.normalize_with_tables(TABLES, "\r\n") == " "
        assert at.normalize_with_tables(TABLES, "\n\n") == "  "


class TestRanges:
    def test_adjacent_codepoints_fold_into_one_interval(self):
        assert q2.to_ranges([5, 6, 7, 10]) == [[5, 7], [10, 10]]

    def test_lookup_matches_the_source_set(self):
        source = {1, 2, 3, 9, 0x10FFFF}
        ranges = q2.to_ranges(source)

        for cp in [0, 1, 2, 3, 4, 8, 9, 10, 0x10FFFE, 0x10FFFF]:
            assert q2.in_ranges(ranges, cp) == (cp in source), hex(cp)


class TestAddedTokens:
    def test_leftmost_longest(self):
        added = ["<|endoftext|>", "<|end"]

        assert q2.split_added_tokens("a<|endoftext|>b", added) == [
            ("a", False),
            ("<|endoftext|>", True),
            ("b", False),
        ]

    def test_a_partial_match_is_not_split_out(self):
        """`<|endoftext|x` matches only partway — splitting it out would change the ids."""
        assert q2.split_added_tokens("<|endoftext|x", ["<|endoftext|>"]) == [
            ("<|endoftext|x", False)
        ]

    def test_an_empty_added_token_is_rejected(self):
        """The empty string matches at every position and the scan never advances (a seed of an
        infinite loop)."""
        with pytest.raises(ValueError, match="empty string"):
            q2.split_added_tokens("abc", ["", "<|a"])


class TestNfcSegments:
    """The ranges where the source of truth's NFC and plain `unicodedata` disagree (the segment
    table)."""

    def test_the_recorded_cases_are_reproduced_by_the_segment_table(self):
        """The fixture's `[input, source-of-truth output]` pairs are reproduced by the segment table
        (the same pairs as the TS implementation)."""
        cases = FIXTURE["qwen"]["nfcCases"]

        assert len(cases) >= 200
        for text, want in cases:
            assert q2.nfc_with_segments(text, NFC_SEGMENTS) == want, [hex(ord(c)) for c in text]

    def test_the_plain_nfc_disagrees_with_the_source_of_truth(self):
        """Measured: the source of truth treats U+089A as combining class 0, so it does not reorder.

        Dropping that range from the segment table returns to plain NFC (= different ids).
        """
        text = "\u0818\u089a"

        assert unicodedata.normalize("NFC", text) == "\u089a\u0818"
        assert q2.nfc_with_segments(text, NFC_SEGMENTS) == text

        dropped = [r for r in NFC_SEGMENTS if not r[0] <= 0x089A <= r[1]]

        assert q2.nfc_with_segments(text, dropped) == "\u089a\u0818"


class TestByteEncoder:
    def test_the_table_is_a_bijection_over_256_bytes(self):
        table = q2.bytes_to_unicode()

        assert len(table) == 256
        assert len(set(table.values())) == 256

    def test_no_mapped_character_is_whitespace(self):
        """A merges key is a single string `"left right"` — whitespace mixed in would break the
        split."""
        table = q2.bytes_to_unicode()

        assert not any(ch.isspace() for ch in table.values())


class TestQwenPreTokenize:
    def test_digits_split_one_by_one(self):
        """`\\p{N}` has no `+` (`2024` becomes 4 tokens)."""
        assert q2.qwen_pre_tokenize("2024", CLASSES, CASE_FOLD) == ["2", "0", "2", "4"]

    def test_a_leading_space_joins_the_following_word(self):
        assert q2.qwen_pre_tokenize("a cat", CLASSES, CASE_FOLD) == ["a", " cat"]

    def test_trailing_spaces_keep_the_last_one_for_the_next_word(self):
        """`\\s+(?!\\S)` — if a non-space follows, the last character is left for the next
        piece."""
        assert q2.qwen_pre_tokenize("a   b", CLASSES, CASE_FOLD) == ["a", "  ", " b"]

    def test_the_scan_never_stalls(self):
        """A position that cannot advance raises (no implementation that silently advances one
        character)."""
        for text in ("", " ", "\r\n", "\u00a0", "\U0001f408", "'", "'s", "'S"):
            q2.qwen_pre_tokenize(text, CLASSES, CASE_FOLD)

    def test_the_case_fold_table_is_not_ascii_only(self):
        """Measured: the source of truth's `(?i:)` is simple case folding — U+017F (ſ) is treated
        as s.

        Settling for ASCII case swapping leaves the `'\u017f` of `it\u017fs` unsplit as one piece,
        and the split points differ from the source of truth's `'\u017f` + `s`. Dropping U+017F
        from the table makes this fail.
        """
        assert CASE_FOLD[0x017F] == ord("s")
        assert CASE_FOLD[ord("S")] == ord("s")
        assert q2.qwen_pre_tokenize("it'\u017fs", CLASSES, CASE_FOLD) == ["it", "'\u017f", "s"]

        # Cutting the table down to ASCII (= the most natural misconception in a port) changes
        # the split points.
        ascii_only = {cp: t for cp, t in CASE_FOLD.items() if cp < 0x80}

        assert q2.qwen_pre_tokenize("it'\u017fs", CLASSES, ascii_only) == ["it", "'\u017fs"]


# ---- comparisons that run only when the real tokenizer.json is present -----------

REPO = anima_demo.DEFAULT_REPO


def _cached(filename: str) -> Path | None:
    """A file in the HF cache. None if it is not there (**does not download**)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(REPO, filename, local_files_only=True))
    except Exception:
        return None


QWEN_JSON = _cached("tokenizer/tokenizer.json")
T5_JSON = _cached("t5_tokenizer/tokenizer.json")
needs_tokenizers = pytest.mark.skipif(
    QWEN_JSON is None or T5_JSON is None,
    reason=f"{REPO}'s tokenizer.json is not in the HF cache (regenerate: anima/demo.py in the"
    " README)",
)


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """Build the reference implementation from the fixture's tables + the real tokenizer.json's
    vocabulary.

    The tables themselves are **not regenerated** (scanning every code point takes minutes).
    What this wants to see is "whether the committed tables and the real vocabulary reproduce
    the source of truth's ids"; the equivalence of the table folding is checked by
    `anima_demo.build_assets` on every emit.
    """
    assert QWEN_JSON is not None and T5_JSON is not None
    source = at.AnimaTokenizerSource(QWEN_JSON, T5_JSON)
    qwen_raw, t5_raw = source.qwen_json(), source.t5_json()
    return {
        "qwen": {"addedTokens": FIXTURE["qwen"]["addedTokens"]},
        "t5": {"unkId": t5_raw["model"]["unk_id"], "eosId": FIXTURE["t5"]["eosId"]},
        "classes": CLASSES,
        "caseFold": CASE_FOLD,
        "nfcSegments": NFC_SEGMENTS,
        "tables": TABLES,
        "qwenVocab": qwen_raw["model"]["vocab"],
        "qwenMerges": [(left, right) for left, right in qwen_raw["model"]["merges"]],
        "t5Vocab": [(token, float(score)) for token, score in t5_raw["model"]["vocab"]],
    }


@needs_tokenizers
class TestReferenceParity:
    def test_the_reference_implementation_reproduces_the_fixture(self, built):
        """The mirror of the TS implementation produces the fixture's ids (confirms what TS should
        copy)."""
        qwen_ref, t5_ref = anima_demo.references(
            {**built, "t5": {**built["t5"], "addedTokens": FIXTURE["t5"]["addedTokens"]}}
        )
        for case in FIXTURE["cases"]:
            assert qwen_ref.encode(case["text"], anima_demo.MAX_LENGTH) == case["qwenIds"], case[
                "id"
            ]
            assert t5_ref.encode(case["text"], anima_demo.MAX_LENGTH) == case["t5Ids"], case["id"]

    def test_the_search_width_is_recomputed_from_the_whole_vocabulary(self, built):
        """Recompute the search width from the **whole vocabulary** of the real tokenizer.json and
        compare for equality.

        The inequality check on the fixture's subset (`TestFixtureShape`) covers only a necessary
        condition, and also passes "a value computed from part of the whole (the top N and so
        on)". Only when the assets are present can it be recomputed from the source of truth
        itself.
        """
        vocab = built["t5Vocab"]

        assert FIXTURE["t5"]["minScore"] == min(score for _, score in vocab)
        assert FIXTURE["t5"]["maxTokenLength"] == max(len(token) for token, _ in vocab)

    def test_the_fixture_still_matches_the_upstream_tokenizers(self):
        """The source of truth (AutoTokenizer) has not moved. If it has, the fixture needs
        regenerating."""
        transformers = pytest.importorskip("transformers")
        # NOTE: `local_files_only=True` cannot be used — resolving an **optional** file such as
        # `special_tokens_map.json` fails with OSError (measured: it cannot be built even with
        # both tokenizer.json and tokenizer_config.json in the cache). Without a network: SKIP.
        try:
            truth = {
                "qwen": transformers.AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer"),
                "t5": transformers.AutoTokenizer.from_pretrained(REPO, subfolder="t5_tokenizer"),
            }
        except OSError as cause:
            pytest.skip(f"cannot build AutoTokenizer (offline environment): {cause}")

        for case in FIXTURE["cases"]:
            assert anima_demo.reference_ids(truth["qwen"], case["text"]) == case["qwenIds"], case[
                "id"
            ]
            assert anima_demo.reference_ids(truth["t5"], case["text"]) == case["t5Ids"], case["id"]
