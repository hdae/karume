"""Script-level contracts of `irodori/tokenizer_ref.py` (the part that needs no real assets).

Emitting from the real assets is manual (the same discipline as `irodori/export.py`). Pinned here
is only the discipline whose breakage would become a **silent id list mismatch**:

- The checks on the shape of upstream's `tokenizer.json` (model type / unk_id / byte_fallback /
  normalizer / pre_tokenizer / the added-token flags / the base id of the byte tokens / the
  premise of the line split)
- `bosId` / `padId` agree with **both** the checkpoint config and the added tokens
- The expected byte_fallback id list is computed independently from UTF-8
- The fail-loudly checks of `build_encode_cases` / `build_batch_cases` / `TokenizerPair.encode`
  (emoji in and out of the vocabulary, agreement of the two id-list paths, the mask's valid
  length, truncation firing)
- No case body in the case tables holds a raw invisible character
- The rule-coverage check of `normalize_text` really fails when a case is removed
- The NFKC diff table holds no identity entry and is the same when built twice
- The vocabulary **subset** of the parity fixture covers the lattice's match set (fault
  injection into the extraction rule + `minScore` / `maxTokenLength` taken from the whole)
- The caption cases go through strip only and are **sensitive** to normalization (the check
  fails if they become insensitive)
- The Gemma BPE kind (v4-Large): which kind a `tokenizer.json` gets, its own default fixture,
  the conditioning config (BOS on text and caption, one tokenizer for both), no rare-token split,
  BOS / pad agreement, the roles of its encode cases, the caption window (longer than
  `max_text_len`, cut at `max_caption_len`) and the shape of its fixture
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import unicodedata
from pathlib import Path

import pytest
import torch

from irodori import tokenizer_ref as tok

TEXT_CONFIG = {"bos_token_id": 1, "pad_token_id": 3}


#: The same first 15 entries as the real one (the order that puts the byte tokens' base id at 15).
SPECIALS = (
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<sep>",
    "<mask>",
    "<cls>",
    "<|system|>",
    "<|assistant|>",
    "<|user|>",
    "<|available_tools|>",
    "<|tool_calls|>",
    "<|tool_results|>",
    "<|code|>",
    "<|file|>",
)


def _raw(**overrides) -> dict:
    """A minimal upstream `tokenizer.json` (vocabulary = 15 special + 256 byte + 2 regular)."""
    vocab: list[list] = [[content, 0.0] for content in SPECIALS]
    vocab += [[f"<0x{value:02X}>", 0.0] for value in range(256)]
    vocab += [["▁こんにちは", -1.5], ["世界", -2.5]]
    raw = {
        "added_tokens": [
            {
                "id": index,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for index, content in enumerate(SPECIALS)
        ],
        "normalizer": None,
        "pre_tokenizer": dict(tok.EXPECTED_PRE_TOKENIZER),
        "model": {"type": "Unigram", "unk_id": 0, "byte_fallback": True, "vocab": vocab},
    }
    raw.update(overrides)
    return raw


class TestUpstreamShape:
    """MUST: do not leave "it should match the premises" as a claim — if it does not, only the id
    list silently breaks."""

    def test_the_expected_shape_reports_the_byte_base_id(self):
        assert tok.check_upstream_shape(_raw()) == {
            "vocab": 273,
            "addedTokens": 15,
            "byteBaseId": tok.EXPECTED_BYTE_BASE_ID,
        }

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda raw: raw["model"].update(type="BPE"), "Unigram"),
            (lambda raw: raw["model"].update(unk_id=2), "unk_id"),
            (lambda raw: raw["model"].update(byte_fallback=False), "byte_fallback"),
            (lambda raw: raw.update(normalizer={"type": "Precompiled"}), "normalizer"),
            (lambda raw: raw["pre_tokenizer"].update(prepend_scheme="always"), "pre_tokenizer"),
            (lambda raw: raw["added_tokens"][0].update(normalized=True), "non-default flags"),
            (lambda raw: raw["model"]["vocab"].append(["改\n行", 0.0]), "contains a newline"),
            (lambda raw: raw["model"]["vocab"].append(["空 白", 0.0]), "tab / space"),
            (lambda raw: raw["model"]["vocab"].insert(0, ["<extra>", 0.0]), "byte value"),
            (lambda raw: raw["model"]["vocab"].__delitem__(20), "expected 256"),
        ],
    )
    def test_a_broken_shape_fails_loudly(self, mutate, message):
        raw = _raw()
        mutate(raw)

        with pytest.raises(SystemExit, match=message):
            tok.check_upstream_shape(raw)


class TestAsset:
    def test_the_vocab_text_has_one_line_per_score(self):
        asset = tok.build_asset(_raw(), TEXT_CONFIG)

        lines = asset["vocabText"].split("\n")
        assert lines == [
            *SPECIALS,
            *(f"<0x{value:02X}>" for value in range(256)),
            "▁こんにちは",
            "世界",
        ]
        assert len(asset["scores"]) == len(lines)
        assert asset["addedTokens"][:2] == [["<unk>", 0], ["<s>", 1]]
        assert [entry[1] for entry in asset["addedTokens"]] == sorted(
            entry[1] for entry in asset["addedTokens"]
        )

    def test_a_bos_id_that_disagrees_with_the_added_vocab_fails_loudly(self):
        """MUST: do not trust the config alone — prepending another id passes as a value."""
        with pytest.raises(SystemExit, match="bos"):
            tok.build_asset(_raw(), {"bos_token_id": 5, "pad_token_id": 3})

    def test_a_pad_id_that_disagrees_with_the_added_vocab_fails_loudly(self):
        with pytest.raises(SystemExit, match="pad"):
            tok.build_asset(_raw(), {"bos_token_id": 1, "pad_token_id": 7})


class TestByteFallbackIds:
    """MUST: the expectation is built from UTF-8 without looking at the vocabulary (keeps the
    test from becoming tautological)."""

    def test_a_four_byte_emoji_expands_to_four_ids(self):
        ids = tok.byte_fallback_ids("\U0001f442")

        assert ids == [tok.EXPECTED_BYTE_BASE_ID + byte for byte in (0xF0, 0x9F, 0x91, 0x82)]

    def test_ascii_stays_one_id_per_byte(self):
        assert tok.byte_fallback_ids("ab") == [
            tok.EXPECTED_BYTE_BASE_ID + 0x61,
            tok.EXPECTED_BYTE_BASE_ID + 0x62,
        ]


class TestRuleCoverage:
    """MUST: do not leave "every rule is exercised" as a claim (the detector for holes in the
    golden)."""

    @pytest.fixture(scope="class")
    @staticmethod
    def cases():
        pytest.importorskip("irodori_tts")
        from irodori_tts.text_normalization import normalize_text

        return tok.build_normalize_cases(normalize_text)

    def test_the_full_case_list_covers_every_rule(self, cases):
        coverage = tok.check_rule_coverage(cases)

        assert coverage["rules"] == 14
        assert coverage["brackets"] >= 3

    def test_dropping_the_case_that_fires_a_rule_fails_loudly(self, cases):
        thinned = [case for case in cases if case["name"] != "wave-dash"]

        with pytest.raises(SystemExit, match="does not fire"):
            tok.check_rule_coverage(thinned)

    def test_a_case_list_without_brackets_fails_loudly(self, cases):
        thinned = [case for case in cases if not case["why"].startswith("bracket stripping")]

        with pytest.raises(SystemExit, match="coverage is thin"):
            tok.check_rule_coverage(thinned)

    def test_the_normalized_field_is_not_stripped(self, cases):
        """MUST: strip is the pipeline stage's job (do not wipe out the leading-space case)."""
        by_name = {case["name"]: case for case in cases}
        assert by_name["ideographic-space"]["normalized"] == "前後ろ"
        # Also check one case that does not copy the raw text (normalization really applies).
        assert by_name["question-bang"]["normalized"] == "本当?そうだ!"


class TestNfkcDiffTable:
    @pytest.fixture(scope="class")
    @staticmethod
    def table():
        return tok.nfkc_diff_table()

    def test_every_entry_is_a_real_difference(self, table):
        for key, value in table.items():
            source = chr(int(key))
            assert value != source
            assert unicodedata.normalize("NFKC", source) == value

    def test_known_compatibility_characters_are_present(self, table):
        # Full-width A / half-width kana ｶ / the ligature ㈱ — typical characters NFKC changes.
        assert table[str(0xFF21)] == "A"
        assert table[str(0xFF76)] == "カ"
        assert table[str(0x3231)] == "(株)"

    def test_identity_codepoints_are_absent(self, table):
        for cp in (0x0041, 0x3042, 0x4E16, 0x1F600):
            assert str(cp) not in table

    def test_the_table_is_deterministic(self, table):
        assert tok.nfkc_diff_table() == table
        assert list(tok.nfkc_diff_table()) == list(table)


#: The token with the lowest score and the greatest length in the **full** vocabulary, which no
#: case contains. Material to tell an implementation that derives `minScore` / `maxTokenLength`
#: from the subset (= changes the weight of unknown nodes and the search width) from the correct
#: one that takes them from the whole.
UNREACHABLE_TOKEN = "宇宙のはて" * 5

#: One case with a half-width space (it matches `▁こんにちは` only after the Metaspace
#: replacement).
PARITY_CASES = [
    {"name": "greeting", "why": "", "raw": " こんにちは世界", "normalized": " こんにちは世界"}
]


def _raw_with_unreachable_token() -> dict:
    raw = _raw()
    raw["model"]["vocab"].append([UNREACHABLE_TOKEN, -9.5])
    return raw


class _FakePair:
    """A stand-in for only the surface `build_caption_cases` touches (`wrapper.batch_encode`).

    The ids are code points (`[BOS, *body]`, right-padded), with no vocabulary and no Viterbi —
    all this needs to see is "whether only strip is applied on the caption side".
    """

    class _Wrapper:
        def batch_encode(self, texts, max_length: int):
            body = [ord(char) for char in texts[0]][: max_length - 1]
            ids = torch.full((1, max_length), 3, dtype=torch.int64)
            mask = torch.zeros((1, max_length), dtype=torch.bool)
            ids[0, : len(body) + 1] = torch.tensor([1, *body], dtype=torch.int64)
            mask[0, : len(body) + 1] = True
            return ids, mask

    def __init__(self) -> None:
        self.wrapper = _FakePair._Wrapper()


#: A minimal config of what `build_caption_cases` reads.
CAPTION_MODEL_CONFIG = {"max_caption_len": 32}


def _fake_normalize(body: str) -> str:
    """A stand-in for only the stages of `normalize_text` that affect the caption cases.

    The real one lives upstream and needs the real assets (`TestRuleCoverage` covers it with
    importorskip). All this needs is the property "normalization moves the id list", so it copies
    three stages: symbol removal, stripping the outer brackets and NFKC.
    """
    stripped = body.strip()
    for char in "①;　":
        stripped = stripped.replace(char, "")
    stripped = stripped.removeprefix("「").removesuffix("」")
    return unicodedata.normalize("NFKC", stripped)


class TestCaptionCases:
    """MUST: the caption side gets strip only (upstream's `_synthesize` does not apply
    `normalize_text`)."""

    def test_the_packed_ids_come_from_the_stripped_text(self):
        cases = tok.build_caption_cases(_FakePair(), _fake_normalize, CAPTION_MODEL_CONFIG)
        by_name = {case["name"]: case for case in cases}
        brackets = by_name["caption-brackets"]

        assert brackets["stripped"] == "「若く元気な女性の声」"
        assert brackets["idsPacked"] == [1, *(ord(char) for char in brackets["stripped"])]
        assert brackets["normalizedIdsPacked"] == [1, *(ord(char) for char in "若く元気な女性の声")]
        assert brackets["normalizeSensitive"] is True

    def test_a_table_that_no_longer_reacts_to_normalization_fails_loudly(self):
        """Against tautology: if normalization became the identity, the two paths would agree and
        the check would lose its power."""
        with pytest.raises(SystemExit, match="caption cases are sensitive to normalization"):
            tok.build_caption_cases(_FakePair(), lambda body: body, CAPTION_MODEL_CONFIG)


class _FakeRawTokenizer:
    """Only the surface of `tokenizers.Tokenizer` that `build_encode_cases` touches
    (`get_vocab`)."""

    def __init__(self, vocab) -> None:
        self._vocab = tuple(vocab)

    def get_vocab(self) -> dict[str, int]:
        return {content: index for index, content in enumerate(self._vocab)}


class _FakeEncodePair:
    """A stand-in for only the surface `build_encode_cases` touches (`raw.get_vocab` /
    `encode`).

    A character in the vocabulary is one id (its code point), one that is not expands to
    byte_fallback — it copies only the **branch** of the real Unigram, with no Viterbi.
    `overrides` replaces the id list of a given case, for fault injection into the byte_fallback
    check.
    """

    def __init__(self, *, vocab=(tok.IN_VOCAB_EMOJI,), overrides=None) -> None:
        self._vocab = set(vocab)
        self.raw = _FakeRawTokenizer(sorted(self._vocab))
        self._overrides = dict(overrides or {})

    def encode(self, text: str, where: str) -> list[int]:
        if where in self._overrides:
            return list(self._overrides[where])
        ids: list[int] = []
        for char in text:
            if char in self._vocab:
                ids.append(ord(char))
            else:
                ids.extend(tok.byte_fallback_ids(char))
        return ids


class TestEncodeCases:
    """MUST: the script checks that cases b / c are "in / outside the vocabulary" before taking
    the golden."""

    def test_the_byte_fallback_case_expands_to_the_utf8_bytes(self):
        cases = tok.build_encode_cases(_FakeEncodePair(), lambda text: text)

        by_name = {case["name"]: case for case in cases}
        assert by_name["emoji-byte-fallback"]["ids"] == tok.byte_fallback_ids(
            tok.OUT_OF_VOCAB_EMOJI
        )
        assert len(by_name["emoji-byte-fallback"]["ids"]) == 4
        # The in-vocabulary side stays one id (b and c are not the same test).
        assert by_name["emoji-in-vocab"]["ids"].count(ord(tok.IN_VOCAB_EMOJI)) == 1
        assert [case["name"] for case in cases] == [name for name, _why, _text in tok.ENCODE_CASES]

    def test_an_in_vocab_emoji_that_is_absent_from_the_vocab_fails_loudly(self):
        with pytest.raises(SystemExit, match="is not in the vocabulary"):
            tok.build_encode_cases(_FakeEncodePair(vocab=()), lambda text: text)

    def test_an_out_of_vocab_emoji_that_is_present_in_the_vocab_fails_loudly(self):
        pair = _FakeEncodePair(vocab=(tok.IN_VOCAB_EMOJI, tok.OUT_OF_VOCAB_EMOJI))

        with pytest.raises(SystemExit, match="do not take byte_fallback"):
            tok.build_encode_cases(pair, lambda text: text)

    def test_a_case_that_did_not_take_the_byte_fallback_path_fails_loudly(self):
        """Against tautology: the expectation is built from UTF-8 without the vocabulary, so a
        shifted id list fails."""
        pair = _FakeEncodePair(overrides={"emoji-byte-fallback": [7]})

        with pytest.raises(SystemExit, match="did not go through byte_fallback"):
            tok.build_encode_cases(pair, lambda text: text)


class _FakeEncoding:
    """A stand-in for only the return value of `tokenizers` (it has `.ids`)."""

    def __init__(self, ids) -> None:
        self.ids = list(ids)


class _FakeRawEncoder:
    def __init__(self, ids) -> None:
        self._ids = list(ids)

    def encode(self, text: str, add_special_tokens: bool) -> _FakeEncoding:
        return _FakeEncoding(self._ids)


class _FakeHfEncoder:
    def __init__(self, ids) -> None:
        self._ids = list(ids)

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return list(self._ids)


def _encode_pair(raw_ids, hf_ids) -> tok.TokenizerPair:
    """Build a `TokenizerPair` without the upstream imports (`__init__` needs the real assets'
    clone)."""
    pair = object.__new__(tok.TokenizerPair)
    pair.raw = _FakeRawEncoder(raw_ids)
    pair.hf = _FakeHfEncoder(hf_ids)
    return pair


class TestTokenizerPairEncode:
    """MUST: the golden is taken only after the two paths (raw tokenizers / through
    transformers) agree."""

    def test_two_paths_that_agree_return_that_id_list(self):
        assert tok.TokenizerPair.encode(_encode_pair([4, 5, 6], [4, 5, 6]), "あ", "ja-plain") == [
            4,
            5,
            6,
        ]

    def test_two_paths_that_disagree_fail_loudly(self):
        pair = _encode_pair([4, 5, 6], [4, 5])

        with pytest.raises(SystemExit, match="tokenizers and transformers"):
            tok.TokenizerPair.encode(pair, "あ", "ja-plain")


class _FakeBatchPair(_FakePair):
    """A stand-in for the surface `build_batch_cases` touches (`wrapper.batch_encode` +
    `encode`).

    One id per character, so the length from `encode` and the wrapper's valid length reach the
    same value along **separate paths** — the valid-length check does not become a copy of the
    implementation.
    """

    def encode(self, text: str, where: str) -> list[int]:
        return [ord(char) for char in text]


class _FakeShortMaskPair(_FakeBatchPair):
    """Returns a mask whose valid length alone is one short (fault injection into the
    valid-length check)."""

    class _Wrapper(_FakePair._Wrapper):
        def batch_encode(self, texts, max_length: int):
            ids, mask = super().batch_encode(texts, max_length)
            mask[0, int(mask[0].sum()) - 1] = False
            return ids, mask

    def __init__(self) -> None:
        super().__init__()
        self.wrapper = _FakeShortMaskPair._Wrapper()


#: A minimal config of what `build_batch_cases` reads (`max_text_len` is a width the long case
#: always overflows).
BATCH_MODEL_CONFIG = {"max_text_len": 64, "max_caption_len": 128}


class TestBatchCases:
    """MUST: the golden always shows the truncation order (cut the body, then the BOS)."""

    def test_the_truncating_case_fills_the_window_and_keeps_the_bos_first(self):
        cases = tok.build_batch_cases(_FakeBatchPair(), lambda text: text, BATCH_MODEL_CONFIG)

        by_name = {case["name"]: case for case in cases}
        truncated = by_name["batch-truncated"]
        assert truncated["maxLength"] == 64
        assert sum(truncated["mask"]) == 64
        assert truncated["idsPadded"][0] == 1
        # The side that does not overflow is exactly BOS + body (the padding is outside the mask).
        short = by_name["batch-short"]
        assert sum(short["mask"]) == len(short["normalized"]) + 1
        assert by_name["batch-caption"]["maxLength"] == 128

    def test_a_mask_whose_effective_length_disagrees_with_the_body_fails_loudly(self):
        with pytest.raises(SystemExit, match="mask's valid length"):
            tok.build_batch_cases(_FakeShortMaskPair(), lambda text: text, BATCH_MODEL_CONFIG)

    def test_a_case_list_where_truncation_never_fires_fails_loudly(self):
        roomy = {"max_text_len": 4096, "max_caption_len": 4096}

        with pytest.raises(SystemExit, match="no case makes truncation fire"):
            tok.build_batch_cases(_FakeBatchPair(), lambda text: text, roomy)


class TestParityFixture:
    """Checks that the TS-side Viterbi builds the same lattice from a vocabulary **subset** as
    from the full vocabulary."""

    def test_the_lattice_text_is_the_metaspace_replaced_normalized_text(self):
        assert tok.lattice_texts(PARITY_CASES, [], []) == ["▁こんにちは世界"]

    def test_the_subset_holds_every_token_the_lattice_can_reach(self):
        raw = _raw_with_unreachable_token()

        subset = tok.vocab_subset(
            raw["model"]["vocab"],
            tok.lattice_texts(PARITY_CASES, [], []),
            {content: index for index, content in enumerate(SPECIALS)},
        )

        tokens = {row[0] for row in subset}
        assert "▁こんにちは" in tokens
        assert "世界" in tokens
        # A token no case contains is left out (that is what makes it a subset).
        assert UNREACHABLE_TOKEN not in tokens
        # The 256 bytes + the added tokens are always included.
        assert len([row for row in subset if row[0].startswith("<0x")]) == 256
        assert set(SPECIALS) <= tokens
        # Ascending id (the premise of a byte-deterministic emit).
        assert [row[1] for row in subset] == sorted(row[1] for row in subset)

    def test_skipping_the_metaspace_replacement_loses_a_token(self):
        """MUST: extraction runs on the string **after** the ▁ replacement — running it on the
        plain normalized text thins the lattice."""
        raw = _raw_with_unreachable_token()

        subset = tok.vocab_subset(raw["model"]["vocab"], [PARITY_CASES[0]["normalized"]], {})

        assert "▁こんにちは" not in {row[0] for row in subset}

    def test_a_subset_that_lost_a_reachable_token_fails_loudly(self):
        """A check in the **opposite direction** from the extraction, so it catches what the
        extraction misses."""
        raw = _raw_with_unreachable_token()
        texts = tok.lattice_texts(PARITY_CASES, [], [])
        full = tok.vocab_subset(raw["model"]["vocab"], texts, {})
        thinned = [row for row in full if row[0] != "世界"]

        assert tok.check_subset_covers_lattice(raw["model"]["vocab"], texts, full) == 2
        with pytest.raises(SystemExit, match="does not cover"):
            tok.check_subset_covers_lattice(raw["model"]["vocab"], texts, thinned)

    def test_the_global_min_score_and_width_are_not_the_subset_values(self):
        """MUST: the weight of unknown nodes and the search width come from the **full**
        vocabulary (not derived from the subset)."""
        raw = _raw_with_unreachable_token()

        fixture = tok.build_parity_fixture(
            tok.build_asset(raw, TEXT_CONFIG), raw["model"]["vocab"], PARITY_CASES, [], [], [], {}
        )

        assert fixture["asset"]["minScore"] == -9.5
        assert min(row[2] for row in fixture["asset"]["vocab"]) == -2.5
        assert fixture["asset"]["maxTokenLength"] == len(UNREACHABLE_TOKEN)
        assert max(len(row[0]) for row in fixture["asset"]["vocab"]) < len(UNREACHABLE_TOKEN)

    def test_the_fixture_is_deterministic(self):
        raw = _raw_with_unreachable_token()
        asset = tok.build_asset(raw, TEXT_CONFIG)

        def build() -> str:
            fixture = tok.build_parity_fixture(
                asset, raw["model"]["vocab"], PARITY_CASES, [], [], [], {"65": "A"}
            )
            return json.dumps(fixture, ensure_ascii=False)

        assert build() == build()


#: The case tables whose body literals are collected (each laid out as
#: `(name, reason, body, ...)`).
CASE_TABLE_NAMES = ("ENCODE_CASES", "BATCH_CASES", "CAPTION_CASES", "NORMALIZE_CASES")


def _case_body_literals(names: tuple[str, ...] = CASE_TABLE_NAMES) -> dict[str, str]:
    """Collect the **bodies** of the case tables as spelled in the script, not as values.

    The value cannot tell "a raw U+3099" from "U+3099 written as `\\u3099`" (the whole point of
    this MUST), so `ast` gives the position of each body and the source range there is read.
    """
    source = Path(tok.__file__).read_text(encoding="utf-8")
    literals: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name not in names:
            continue
        for case in node.value.elts:
            literals[case.elts[0].value] = ast.get_source_segment(source, case.elts[2])
    return literals


def _invisible_codepoints(text: str) -> list[str]:
    """The invisible characters that appear **raw** in `text` (combining marks Mn / format
    characters Cf / U+3000)."""
    return [
        f"U+{ord(char):04X}"
        for char in text
        if char == "\u3000" or unicodedata.category(char) in ("Mn", "Cf")
    ]


class TestCaseTableLiterals:
    r"""MUST (the docstring of `ENCODE_CASES`): invisible characters are always written as
    `\uXXXX`.

    Written raw, one NFC normalization by an editor turns `か + U+3099` into the precomposed `が`,
    and one trailing-whitespace cleanup removes the final U+3000. **Both stay green once the
    golden is regenerated** (nobody can see that the case lost what it tests), so a check that
    reads the tables' literals statically lives here. Only the bodies are judged — the why field
    holds `U+3099` as explanatory text.
    """

    def test_no_case_body_is_written_with_a_raw_invisible_character(self):
        literals = _case_body_literals()

        # All four tables were really collected (an empty scan target would make the check vanish).
        assert len(literals) == sum(
            len(table)
            for table in (
                tok.ENCODE_CASES,
                tok.BATCH_CASES,
                tok.CAPTION_CASES,
                tok.NORMALIZE_CASES,
            )
        )
        assert {name: _invisible_codepoints(text) for name, text in literals.items()} == {
            name: [] for name in literals
        }

    def test_the_escaped_cases_still_carry_the_characters_they_name(self):
        """Against tautology: only the notation changed; the value has not moved by a single
        bit."""
        by_name = {name: text for name, _why, text in tok.ENCODE_CASES}
        dakuten = by_name["combining-dakuten"]
        # `か` + the combining mark, not the precomposed `が` (the case keeps what it tests).
        assert "\u304c" not in dakuten
        composed = "\u304c\u3063\u3053\u3044\u3044\u3058\u3083\u3093\u3051\u3093"
        assert unicodedata.normalize("NFKC", dakuten) == composed
        caption = {name: text for name, _why, text in tok.CAPTION_CASES}
        assert caption["caption-surrounding-space"].count("\u3000") == 3


class TestSplitCaseTable:
    """The split cases follow the discipline of the other case tables."""

    def test_no_split_case_body_is_written_with_a_raw_invisible_character(self):
        literals = _case_body_literals(("SPLIT_CASES",))

        assert len(literals) == len(tok.SPLIT_CASES)
        assert {name: _invisible_codepoints(text) for name, text in literals.items()} == {
            name: [] for name in literals
        }

    def test_the_split_cases_have_unique_names_and_the_batch_cases_exist(self):
        names = [name for name, _why, _text in tok.SPLIT_CASES]

        assert len(names) == len(set(names))
        assert set(names) >= tok.SPLIT_BATCH_CASES


class TestCli:
    def test_the_default_out_dir_is_derived_from_the_weight_directory(self, tmp_path):
        out = tok.default_out_dir(tmp_path / "v4-small")

        assert out.name == "tokenizer"
        assert out.parent.name == "irodori-v4-small"

    def test_the_case_tables_have_unique_names(self):
        """A duplicate name silently drops one entry from the golden's comparison table."""
        for table in (tok.ENCODE_CASES, tok.NORMALIZE_CASES, tok.CAPTION_CASES):
            names = [name for name, _why, _text in table]
            assert len(names) == len(set(names))
        batch = [name for name, _why, _text, _key in tok.BATCH_CASES]
        assert len(batch) == len(set(batch))

    def test_the_expected_pre_tokenizer_is_not_mutated_by_callers(self):
        """Relies on `_raw` passing a `dict(...)` copy (a shared reference would leak the fault
        injection)."""
        snapshot = copy.deepcopy(tok.EXPECTED_PRE_TOKENIZER)
        raw = _raw()
        raw["pre_tokenizer"]["prepend_scheme"] = "always"

        assert snapshot == tok.EXPECTED_PRE_TOKENIZER


#: The ids of the synthetic coverage vocabulary below (after the 15 specials + 256 bytes).
_KEEP, _HIRAGANA, _DISABLED, _FORCED, _PIECE_A, _PIECE_B, _FORCED_3 = (
    271,
    272,
    273,
    274,
    275,
    276,
    277,
)
_PIECES_3, _DISABLED_NEXT, _SHORT, _SHORT_B, _SHORT_C = (278, 279, 280), 281, 282, 283, 284
_USER = SPECIALS.index("<|user|>")


def _coverage_vocab() -> list[list]:
    """15 specials + 256 bytes + the tokens each split role needs (scores on upstream's scale)."""
    vocab = [list(row) for row in _raw()["model"]["vocab"][: 15 + 256]]
    vocab += [
        [tok.SPLIT_KEEP_TOKEN, -14.0],
        [tok.SPLIT_HIRAGANA_TOKEN, -14.0],
        ["disabled", -14.0],
        ["freeship", -10.0],
        ["free", -8.0],
        ["ship", -8.0],
        ["seveneleven", -10.0],
        ["se", -9.0],
        ["ven", -9.0],
        ["eleven", -9.0],
        ["nextword", -14.0],
        ["short", -8.0],
        ["shortb", -8.0],
        ["shortc", -8.0],
    ]
    return vocab


_COVERAGE_TABLE = {
    "disabledScore": -1e9,
    "disabledIds": [_DISABLED, _DISABLED_NEXT],
    "forcedSplits": [[_FORCED, [_PIECE_A, _PIECE_B]], [_FORCED_3, list(_PIECES_3)]],
}
_FORCE_SPLITS = {"freeship": ("free", "ship"), "seveneleven": ("seven", "eleven")}
_EMOJI_BYTES = tok.byte_fallback_ids(tok.OUT_OF_VOCAB_EMOJI)


def _coverage_cases() -> list[dict]:
    """Measured-looking id lists in which every role of `SPLIT_CASES` happens."""
    rows = {
        "split-commit-examples": ([_DISABLED, _FORCED], [_SHORT, _PIECE_A, _PIECE_B]),
        "split-forced": ([_FORCED, _FORCED_3], [_PIECE_A, _PIECE_B, *_PIECES_3]),
        "split-forced-solved-piece": ([_FORCED_3], list(_PIECES_3)),
        "split-keep": ([_KEEP], [_KEEP]),
        "split-hiragana-only": ([_HIRAGANA, _DISABLED_NEXT], [_HIRAGANA, _SHORT_B, _SHORT_C]),
        "split-added-token-emoji": (
            [_USER, _DISABLED, *_EMOJI_BYTES],
            [_USER, _SHORT, *_EMOJI_BYTES],
        ),
        "split-truncates": ([_SHORT] * 10, [_SHORT] * 20),
        "split-long-passage": ([_SHORT] * 20, [_SHORT] * 22),
    }
    return [{"name": name, "ids": ids, "idsSplit": split} for name, (ids, split) in rows.items()]


def _check_coverage(cases) -> dict:
    return tok.check_split_coverage(
        cases, _COVERAGE_TABLE, _coverage_vocab(), _FORCE_SPLITS, -13.0, 16
    )


class TestDeriveRareTokenSplit:
    """MUST: the table is what the splitter **did** — every replaced id and nothing else."""

    @staticmethod
    def _vocabs():
        original = [["a", -1.0], ["bcde", -14.0], ["f", -2.0], ["ghij", -15.0]]
        split = [["a", -1.0], ["bcde", -1e9], ["f", -2.0], ["ghij", -1e9]]
        return original, split

    def test_the_replaced_ids_and_the_forced_map_become_the_table(self):
        original, split = self._vocabs()

        table = tok.derive_rare_token_split(original, split, {2: [0], 0: [2, 2]})

        assert table == {
            "disabledScore": -1e9,
            "disabledIds": [1, 3],
            "forcedSplits": [[0, [2, 2]], [2, [0]]],
        }

    @pytest.mark.parametrize(
        ("mutate", "forced", "message"),
        [
            (lambda split: split.__setitem__(0, ["x", -1.0]), {}, "instead of"),
            (lambda split: split.__setitem__(3, ["ghij", -2e9]), {}, "2 different values"),
            (
                lambda split: (
                    split.__setitem__(1, ["bcde", -14.0]) or split.__setitem__(3, ["ghij", -15.0])
                ),
                {},
                "replaced no score",
            ),
            (lambda split: split.pop(), {}, "entries"),
            (lambda split: None, {1: [0]}, "also disabled"),
            (lambda split: None, {0: []}, "no pieces"),
            (lambda split: None, {0: [4]}, "leaves the vocabulary"),
        ],
    )
    def test_a_splitter_of_another_shape_fails_loudly(self, mutate, forced, message):
        original, split = self._vocabs()
        mutate(split)

        with pytest.raises(SystemExit, match=message):
            tok.derive_rare_token_split(original, split, forced)

    def test_the_split_min_score_is_the_replaced_vocabulary_minimum(self):
        """MUST: the split Unigram derives its unknown-piece weight from its **own** minimum."""
        original, split = self._vocabs()
        table = tok.derive_rare_token_split(original, split, {})

        assert tok.split_min_score(original, table) == -1e9
        # Not the original minimum, and not the minimum of what is left after removing them.
        assert min(score for _token, score in original) == -15.0


class TestAssetSplitKey:
    def test_the_table_is_appended_as_the_last_key(self):
        table = {"disabledScore": -1e9, "disabledIds": [271], "forcedSplits": []}

        plain = tok.build_asset(_raw(), TEXT_CONFIG)
        with_split = tok.build_asset(_raw(), TEXT_CONFIG, table)

        assert tok.RARE_TOKEN_SPLIT_KEY not in plain
        assert list(with_split) == [*plain, tok.RARE_TOKEN_SPLIT_KEY]
        assert with_split[tok.RARE_TOKEN_SPLIT_KEY] == table
        # The bytes of the asset without the table are a prefix of the asset with it.
        plain_text = json.dumps(plain, ensure_ascii=False)
        split_text = json.dumps(with_split, ensure_ascii=False)
        assert split_text.startswith(plain_text[:-1] + ", ")

    def test_the_fixture_cuts_the_disabled_ids_to_the_subset(self):
        raw = _raw_with_unreachable_token()
        unreachable = len(raw["model"]["vocab"]) - 1
        world = [token for token, _score in raw["model"]["vocab"]].index("世界")
        table = {
            "disabledScore": -1e9,
            "disabledIds": [world, unreachable],
            "forcedSplits": [[world, [world - 1]]],
        }

        fixture = tok.build_parity_fixture(
            tok.build_asset(raw, TEXT_CONFIG, table),
            raw["model"]["vocab"],
            PARITY_CASES,
            [],
            [],
            [],
            {},
            [{"normalized": "世界"}],
        )

        split = fixture["asset"][tok.RARE_TOKEN_SPLIT_KEY]
        assert split["disabledIds"] == [world]
        assert split["disabledTotal"] == 2
        assert split["minScore"] == -1e9
        assert split["forcedSplits"] == table["forcedSplits"]
        assert fixture["encode"]["split"] == [{"normalized": "世界"}]

    def test_a_fixture_without_the_table_has_no_split_sections(self):
        raw = _raw_with_unreachable_token()

        fixture = tok.build_parity_fixture(
            tok.build_asset(raw, TEXT_CONFIG), raw["model"]["vocab"], PARITY_CASES, [], [], [], {}
        )

        assert tok.RARE_TOKEN_SPLIT_KEY not in fixture["asset"]
        assert "split" not in fixture["encode"]


def _unigram_tokenizer(vocab):
    """A real `tokenizers` Unigram over `vocab` with the pre-tokenizer of the real asset."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(
        models.Unigram([tuple(row) for row in vocab], unk_id=0, byte_fallback=False)
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=tok.METASPACE, prepend_scheme="never", split=False
    )
    return tokenizer


class TestTableReproducesSplitter:
    """The table alone (ids + one score + the forced map) reproduces upstream's splitter."""

    @pytest.fixture(scope="class")
    @staticmethod
    def splitter():
        token_split = pytest.importorskip("irodori_tts.token_split")
        vocab = [
            ["<unk>", 0.0],
            ["浦和レッズ", -14.0],
            ["浦和", -8.0],
            ["レッ", -9.0],
            ["ズ", -9.0],
            ["送料無料", -10.0],
            ["送料", -8.0],
            ["無料", -8.0],
            ["の", -3.0],
        ]
        return _unigram_tokenizer(vocab), token_split.RareTokenSplitter(_unigram_tokenizer(vocab))

    def test_the_rebuilt_tokenizer_agrees_with_the_splitter(self, splitter):
        tokenizer, rare = splitter
        raw = json.loads(tokenizer.to_str())
        split_vocab = json.loads(rare.tokenizer.to_str())["model"]["vocab"]
        table = tok.derive_rare_token_split(raw["model"]["vocab"], split_vocab, rare.forced)

        assert table["disabledIds"] == [1]
        assert table["forcedSplits"] == [[5, [6, 7]]]
        texts = ["浦和レッズの送料無料", "浦和レッズ", "送料無料"]
        assert tok.check_table_reproduces_splitter(raw, table, rare.encode, texts) == 3
        assert rare.encode("浦和レッズの送料無料") == [2, 3, 4, 8, 6, 7]

    def test_a_table_missing_a_disabled_id_fails_loudly(self, splitter):
        """Against tautology: the rebuilt side never sees the rule, so a short table shows."""
        tokenizer, rare = splitter
        raw = json.loads(tokenizer.to_str())
        table = {"disabledScore": -1e9, "disabledIds": [], "forcedSplits": [[5, [6, 7]]]}

        with pytest.raises(SystemExit, match="does not reproduce"):
            tok.check_table_reproduces_splitter(raw, table, rare.encode, ["浦和レッズ"])


class TestSplitCoverage:
    """MUST (the docstring of `SPLIT_CASES`): every case still exercises the role it names."""

    def test_cases_that_exercise_every_role_pass(self):
        assert _check_coverage(_coverage_cases()) == {"changed": 7, "cases": 8}

    @pytest.mark.parametrize(
        ("name", "field", "ids", "message"),
        [
            ("split-commit-examples", "ids", [_SHORT, _FORCED], "no disabled token"),
            ("split-commit-examples", "idsSplit", [_DISABLED], "survived the split"),
            ("split-forced", "ids", [_FORCED], "fewer than two"),
            ("split-forced-solved-piece", "ids", [_FORCED], "had to be solved"),
            ("split-keep", "idsSplit", [_SHORT], "not one token on both paths"),
            ("split-hiragana-only", "idsSplit", [_HIRAGANA, _DISABLED_NEXT], "nothing next to"),
            ("split-added-token-emoji", "idsSplit", [_SHORT, *_EMOJI_BYTES], "added token is lost"),
            ("split-truncates", "ids", [_SHORT] * 16, "split side only"),
            ("split-long-passage", "ids", [_SHORT] * 15, "truncation does not fire unsplit"),
        ],
    )
    def test_a_case_that_lost_its_role_fails_loudly(self, name, field, ids, message):
        cases = _coverage_cases()
        next(case for case in cases if case["name"] == name)[field] = ids

        with pytest.raises(SystemExit, match=message):
            _check_coverage(cases)

    def test_a_keep_token_above_the_threshold_fails_loudly(self):
        """The KEEP case only shows KEEP_TOKENS when the token would otherwise be disabled."""
        with pytest.raises(SystemExit, match="above the threshold"):
            tok.check_split_coverage(
                _coverage_cases(), _COVERAGE_TABLE, _coverage_vocab(), _FORCE_SPLITS, -20.0, 16
            )


# --- the Gemma BPE kind (Irodori-TTS v4-Large) ------------------------------------------------

#: The added tokens of the synthetic Gemma tokenizer `(spelling, special)`, ids 0.. in order —
#: the real one's specials, its non-special HTML and newline entries, and a turn marker.
_GEMMA_ADDED = (
    ("<pad>", True),
    ("<eos>", True),
    ("<bos>", True),
    ("<unk>", True),
    ("<start_of_turn>", True),
    ("\n", False),
    ("\n\n", False),
    ("<b>", False),
    ("</b>", False),
)

#: The plain vocabulary after the 256 byte tokens.
_GEMMA_PIECES = ("a", "b", "ab", tok.METASPACE, tok.METASPACE * 2, tok.METASPACE * 3)


def _gemma_raw(pieces=(), post_single=None) -> dict:
    """A minimal T5Gemma 2-shaped `tokenizer.json` (BPE, the shared generator's accepted set,
    the bos-only post_processor unless `post_single` replaces it)."""
    from _shared.gemma_tokenizer import (
        EXPECTED_DECODER,
        EXPECTED_NORMALIZER,
        EXPECTED_PRE_TOKENIZER,
        byte_token,
    )

    tokens = [
        *(content for content, _special in _GEMMA_ADDED),
        *(byte_token(byte) for byte in range(256)),
        *_GEMMA_PIECES,
        tok.IN_VOCAB_EMOJI,
        *pieces,
    ]
    bos = {"SpecialToken": {"id": "<bos>", "type_id": 0}}
    sequence = {"Sequence": {"id": "A", "type_id": 0}}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": index,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": special,
            }
            for index, (content, special) in enumerate(_GEMMA_ADDED)
        ],
        "normalizer": EXPECTED_NORMALIZER,
        "pre_tokenizer": EXPECTED_PRE_TOKENIZER,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [bos, sequence] if post_single is None else post_single,
            "pair": [bos, sequence],
            "special_tokens": {},
        },
        "decoder": EXPECTED_DECODER,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": {token: index for index, token in enumerate(tokens)},
            "merges": [
                ["a", "b"],
                [tok.METASPACE, tok.METASPACE],
                [tok.METASPACE * 2, tok.METASPACE],
            ],
        },
    }


def _gemma_compiled(pieces=()):
    from _shared.gemma_tokenizer import compile_tokenizer

    return compile_tokenizer(_gemma_raw(pieces))


#: The checkpoint configs of the two kinds (the measured values that matter here).
LARGE_TEXT_CONFIG = {"bos_token_id": 2, "pad_token_id": 0}
LARGE_MODEL_CONFIG = {
    "text_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "text_add_bos": True,
    "use_caption_condition": True,
    "caption_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "caption_add_bos": True,
}
SPLIT_REPO = "sbintuitions/modernbert-ja-310m"


class TestTokenizerKind:
    """MUST: the kind comes from upstream's `tokenizer.json` itself (`model.type`)."""

    def test_a_unigram_tokenizer_gets_the_unigram_kind(self):
        assert tok.tokenizer_kind(_raw()) == tok.UNIGRAM_KIND

    def test_a_bpe_tokenizer_gets_the_gemma_kind(self):
        assert tok.tokenizer_kind(_gemma_raw()) == tok.GEMMA_KIND

    @pytest.mark.parametrize(
        "raw", [{"model": {"type": "WordPiece"}}, {"model": {}}, {}, {"model": "BPE"}]
    )
    def test_any_other_tokenizer_fails_loudly(self, raw):
        with pytest.raises(SystemExit, match="neither Unigram nor BPE"):
            tok.tokenizer_kind(raw)

    @pytest.mark.parametrize(
        ("directory", "kind"),
        [("v4-small", tok.UNIGRAM_KIND), ("v4-large", tok.GEMMA_KIND)],
    )
    def test_the_real_model_directories_get_their_kind(self, directory, kind):
        model_dir = tok.REPO_ROOT / "inputs" / "irodori" / directory
        if not (model_dir / tok.TOKENIZER_FILE).is_file():
            pytest.skip(f"{model_dir} is not on this machine")

        assert tok.tokenizer_kind(tok._tokenizer_json(model_dir)) == kind


class TestDefaultFixturePath:
    def test_the_unigram_kind_keeps_the_shared_fixture(self, tmp_path):
        for name in ("v4-small", "v4.1-small-mf"):
            assert tok.default_fixture_path(tmp_path / name, tok.UNIGRAM_KIND) == (
                tok.DEFAULT_FIXTURE_PATH
            )

    def test_the_gemma_kind_writes_its_own_fixture_next_to_it(self, tmp_path):
        path = tok.default_fixture_path(tmp_path / "v4-large", tok.GEMMA_KIND)

        assert path == tok.DEFAULT_FIXTURE_PATH.with_name("v4-large-parity.json")
        assert path != tok.DEFAULT_FIXTURE_PATH


class TestConditioningConfig:
    """MUST: the goldens prepend the BOS and tokenize the caption with the text tokenizer."""

    def test_the_v4_large_config_passes(self):
        tok.check_conditioning_config(LARGE_MODEL_CONFIG)

    def test_a_null_caption_repo_and_bos_fall_back_to_the_text_values(self):
        tok.check_conditioning_config(
            {**LARGE_MODEL_CONFIG, "caption_tokenizer_repo": None, "caption_add_bos": None}
        )

    def test_a_model_without_caption_conditioning_skips_the_caption_checks(self):
        tok.check_conditioning_config(
            {**LARGE_MODEL_CONFIG, "use_caption_condition": False, "caption_add_bos": False}
        )

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"text_add_bos": False}, "text_add_bos"),
            ({"text_add_bos": None}, "text_add_bos"),
            ({"caption_tokenizer_repo": SPLIT_REPO}, "not the text tokenizer"),
            ({"caption_add_bos": False}, "caption_add_bos"),
        ],
    )
    def test_a_config_the_goldens_do_not_describe_fails_loudly(self, override, message):
        with pytest.raises(SystemExit, match=message):
            tok.check_conditioning_config({**LARGE_MODEL_CONFIG, **override})


class TestGemmaRareTokenSplit:
    """The Gemma kind has no rare-token split (upstream's splitter is for the Unigram tokenizer
    of one repository only)."""

    def test_the_t5gemma_repository_has_no_split(self):
        tok.check_gemma_has_no_split(LARGE_MODEL_CONFIG, SPLIT_REPO)

    def test_a_bpe_tokenizer_under_the_split_repository_fails_loudly(self):
        with pytest.raises(SystemExit, match="needs a Unigram"):
            tok.check_gemma_has_no_split({"text_tokenizer_repo": SPLIT_REPO}, SPLIT_REPO)

    def test_the_split_repository_is_upstreams_constant(self):
        token_split = pytest.importorskip("irodori_tts.token_split")

        assert token_split.SPLIT_TOKENIZER_REPO == SPLIT_REPO

    def test_the_gemma_asset_and_fixture_carry_no_split_table(self):
        from _shared.gemma_tokenizer import asset_payload

        compiled = _gemma_compiled()
        asset = asset_payload(compiled, source={})
        fixture = tok.build_gemma_parity_fixture(compiled, {}, {}, [], [], [])

        assert tok.RARE_TOKEN_SPLIT_KEY not in asset
        assert tok.RARE_TOKEN_SPLIT_KEY not in fixture["asset"]
        assert "split" not in fixture["encode"]


class TestGemmaPostProcessor:
    """The shared generator's accepted set gains T5Gemma 2's bos-only post_processor."""

    def test_a_bos_only_post_processor_is_its_own_shape(self):
        from _shared.gemma_tokenizer import POST_BOS, spec_of

        compiled = _gemma_compiled()

        assert compiled.post_processor == POST_BOS == "bos"
        assert spec_of(compiled)["postProcessor"] == "bos"

    def test_a_bos_without_the_sequence_is_still_rejected(self):
        from _shared.gemma_tokenizer import UnsupportedTokenizerError, compile_tokenizer

        bos = {"SpecialToken": {"id": "<bos>", "type_id": 0}}
        with pytest.raises(UnsupportedTokenizerError, match="unknown shape"):
            compile_tokenizer(_gemma_raw(post_single=[bos]))


class TestGemmaSpecialIds:
    """MUST: the BOS / pad ids agree across the config, the asset and upstream's wrapper."""

    def test_agreeing_ids_are_returned(self):
        assert tok.check_gemma_special_ids(_gemma_compiled(), LARGE_TEXT_CONFIG, 2, 0) == {
            "bosId": 2,
            "padId": 0,
        }

    @pytest.mark.parametrize(
        ("text_config", "bos", "pad", "message"),
        [
            ({"bos_token_id": 1, "pad_token_id": 0}, 2, 0, "bos"),
            (LARGE_TEXT_CONFIG, 1, 0, "bos"),
            ({"bos_token_id": 2, "pad_token_id": 3}, 2, 0, "pad"),
            (LARGE_TEXT_CONFIG, 2, None, "pad"),
        ],
    )
    def test_a_disagreeing_id_fails_loudly(self, text_config, bos, pad, message):
        with pytest.raises(SystemExit, match=message):
            tok.check_gemma_special_ids(_gemma_compiled(), text_config, bos, pad)

    def test_a_tokenizer_without_the_pad_token_fails_loudly(self):
        compiled = _gemma_compiled()
        without_pad = dataclasses.replace(
            compiled,
            added_tokens=[row for row in compiled.added_tokens if row[0] != tok.GEMMA_PAD_TOKEN],
        )

        with pytest.raises(SystemExit, match="pad"):
            tok.check_gemma_special_ids(without_pad, LARGE_TEXT_CONFIG, 2, 0)


def _gemma_role_cases(compiled) -> list[dict]:
    """Gemma encode cases whose ids play every role (built from the compiled tables)."""
    vocab = {token: index for index, token in enumerate(compiled.vocab)}
    added = dict(compiled.added_tokens)

    def utf8(text: str) -> list[int]:
        return [compiled.byte_ids[byte] for byte in text.encode("utf-8")]

    ids = {
        "emoji-in-vocab": [vocab["a"], vocab[tok.IN_VOCAB_EMOJI]],
        "byte-fallback-kanji": [*utf8(tok.GEMMA_BYTE_FALLBACK_KANJI), vocab["a"]],
        "byte-fallback-emoji-pair": utf8(tok.GEMMA_BYTE_FALLBACK_EMOJI * 2),
        "added-html": [added["<b>"], vocab["ab"], added["</b>"]],
        "special-in-text": [added["<start_of_turn>"], vocab["a"], added["<bos>"]],
        "added-lookalike": [vocab["a"], vocab["b"]],
        "newlines": [vocab["a"], added["\n\n"], vocab["b"], added["\n"]],
        "double-space": [vocab[tok.METASPACE * 2], vocab[tok.METASPACE * 3]],
    }
    return [
        {
            "name": name,
            "normalized": text,
            "ids": ids.get(name, []),
        }
        for name, _why, text in tok.GEMMA_ENCODE_CASES
    ]


class TestGemmaEncodeRoles:
    """MUST (the docstring of `GEMMA_ENCODE_CASES`): each case still exercises its boundary."""

    def test_cases_that_play_every_role_pass(self):
        compiled = _gemma_compiled()

        assert tok.check_gemma_encode_roles(_gemma_role_cases(compiled), compiled) == {
            "cases": len(tok.GEMMA_ENCODE_CASES),
            "byteFallbackIds": 8,
        }

    def test_the_byte_expectation_uses_the_byte_table_not_a_fixed_base(self):
        """The Unigram base 15 does not hold here (the real Gemma base is 238)."""
        compiled = _gemma_compiled()
        cases = _gemma_role_cases(compiled)
        pair_case = next(case for case in cases if case["name"] == "byte-fallback-emoji-pair")
        pair_case["ids"] = tok.byte_fallback_ids(tok.GEMMA_BYTE_FALLBACK_EMOJI * 2)

        with pytest.raises(SystemExit, match="byte-fallback-emoji-pair"):
            tok.check_gemma_encode_roles(cases, compiled)

    @pytest.mark.parametrize(
        ("name", "ids", "message"),
        [
            ("emoji-in-vocab", [], "emoji-in-vocab: not one token"),
            ("byte-fallback-kanji", [], "byte-fallback-kanji: no byte_fallback run"),
            ("added-html", [7], "added-html: not cut out"),
            ("special-in-text", [4], "special-in-text: not cut out"),
            ("added-lookalike", [7], "added-lookalike: an added token was cut out"),
            ("newlines", [5], "newlines: not cut out"),
            ("double-space", [], "double-space: no metaspace-run token"),
        ],
    )
    def test_a_case_that_lost_its_role_fails_loudly(self, name, ids, message):
        compiled = _gemma_compiled()
        cases = _gemma_role_cases(compiled)
        next(case for case in cases if case["name"] == name)["ids"] = ids

        with pytest.raises(SystemExit, match=message):
            tok.check_gemma_encode_roles(cases, compiled)

    def test_a_byte_fallback_character_that_entered_the_vocabulary_fails_loudly(self):
        compiled = _gemma_compiled(pieces=(tok.GEMMA_BYTE_FALLBACK_KANJI,))

        with pytest.raises(SystemExit, match="is in the vocabulary"):
            tok.check_gemma_encode_roles(_gemma_role_cases(compiled), compiled)


def _gemma_caption_config() -> dict:
    """Windows that put the two long cases (code-point ids) on each side of their boundary."""
    length = len(tok.LONG_CAPTION)
    return {"max_text_len": length, "max_caption_len": 2 * length + 10}


def _drop_tabs(body: str) -> str:
    return _fake_normalize(body).replace("\t", "")


class TestGemmaCaptionCases:
    """The caption window: BOS first, longer than max_text_len, cut at max_caption_len."""

    @pytest.fixture
    @staticmethod
    def cases():
        return tok.build_caption_cases(
            _FakeBatchPair(),
            _drop_tabs,
            _gemma_caption_config(),
            (*tok.CAPTION_CASES, *tok.GEMMA_CAPTION_CASES),
        )

    def test_every_caption_starts_with_the_bos_and_keeps_the_raw_spacing(self, cases):
        by_name = {case["name"]: case for case in cases}

        assert all(case["idsPacked"][0] == 1 for case in cases)
        tabs = by_name["caption-newlines-tabs"]
        assert tabs["idsPacked"][1:] == [ord(char) for char in tabs["stripped"]]
        assert "\t" in tabs["stripped"] and "\n\n" in tabs["stripped"]
        assert tabs["normalizeSensitive"] is True

    def test_the_long_captions_hold_their_roles(self, cases):
        config = _gemma_caption_config()
        lengths = tok.check_caption_lengths(cases, _FakeBatchPair(), config)

        assert lengths == {
            "overTextLength": 2 * len(tok.LONG_CAPTION) + 1,
            "truncatedBody": 3 * len(tok.LONG_CAPTION),
        }
        truncated = next(case for case in cases if case["name"] == "caption-truncated")
        # BOS, then the body cut at max_caption_len - 1 (truncation after the BOS).
        assert truncated["idsPacked"] == [
            1,
            *(ord(char) for char in truncated["stripped"][: config["max_caption_len"] - 1]),
        ]

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"max_text_len": 10_000}, "caption-over-text-length"),
            ({"max_caption_len": 10_000}, "caption-truncated"),
        ],
    )
    def test_windows_that_take_the_roles_away_fail_loudly(self, override, message):
        config = {**_gemma_caption_config(), **override}
        cases = tok.build_caption_cases(
            _FakeBatchPair(), _drop_tabs, config, (*tok.CAPTION_CASES, *tok.GEMMA_CAPTION_CASES)
        )

        with pytest.raises(SystemExit, match=message):
            tok.check_caption_lengths(cases, _FakeBatchPair(), config)

    def test_the_default_case_table_is_unchanged(self):
        """The small models' goldens keep their caption cases."""
        cases = tok.build_caption_cases(_FakePair(), _fake_normalize, CAPTION_MODEL_CONFIG)

        assert [case["name"] for case in cases] == [name for name, _w, _t in tok.CAPTION_CASES]


class TestGemmaParityFixture:
    def test_the_fixture_has_the_gemma_text_shape_and_keeps_the_pad_id(self):
        compiled = _gemma_compiled()
        vocab = {token: index for index, token in enumerate(compiled.vocab)}
        encode = [{"name": "ab", "normalized": "ab", "ids": [vocab["ab"]]}]
        batch = [{"normalized": "ab", "idsPadded": [2, vocab["ab"], 0, 0]}]

        fixture = tok.build_gemma_parity_fixture(
            compiled, {"path": "p"}, {"bosId": 2, "padId": 0}, encode, batch, []
        )

        assert fixture["spec"]["postProcessor"] == "bos"
        assert set(fixture["asset"]) == {
            "vocab",
            "merges",
            "byteIds",
            "addedTokens",
            "specialIds",
            "unkId",
            "bosId",
            "eosId",
        }
        # `<pad>` is not in any text, but the padded ids hold it, so the subset keeps it.
        assert [tok.GEMMA_PAD_TOKEN, 0] in fixture["asset"]["addedTokens"]
        assert fixture["conditioning"] == {"bosId": 2, "padId": 0}
        assert fixture["encode"] == {"cases": encode, "batch": batch, "caption": []}
        assert fixture["stats"]["vocabTotal"] == len(compiled.vocab)


class TestGemmaCaseTables:
    """The Gemma case tables follow the discipline of the other tables."""

    def test_no_case_body_is_written_with_a_raw_invisible_character(self):
        literals = _case_body_literals(("GEMMA_ENCODE_CASES", "GEMMA_CAPTION_CASES"))

        assert len(literals) == len(tok.GEMMA_ENCODE_CASES) + len(tok.GEMMA_CAPTION_CASES)
        assert {name: _invisible_codepoints(text) for name, text in literals.items()} == {
            name: [] for name in literals
        }

    def test_the_names_are_unique_with_the_shared_caption_cases(self):
        encode = [name for name, _why, _text in tok.GEMMA_ENCODE_CASES]
        caption = [name for name, _why, _text in (*tok.CAPTION_CASES, *tok.GEMMA_CAPTION_CASES)]

        assert len(encode) == len(set(encode))
        assert len(caption) == len(set(caption))
