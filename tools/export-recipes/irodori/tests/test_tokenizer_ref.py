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
"""

from __future__ import annotations

import ast
import copy
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


def _case_body_literals() -> dict[str, str]:
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
        if name not in CASE_TABLE_NAMES:
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
