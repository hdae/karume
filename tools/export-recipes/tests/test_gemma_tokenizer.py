"""The compile of the Gemma tokenizers (`_shared/gemma_tokenizer.py`) — the accepted set and how
it is folded.

Only what **would not surface until run time if let through** is pinned here:

- anything outside the accepted set **fails at compile time** (ADR 0084 decision 1 — not
  scattering failures over run time is the very reason for compile-to-asset). A normalizer /
  pre_tokenizer / decoder / model flag that differs in even one field is a different tokenizer;
  let through, only the id sequence changes, silently
- byteIds are looked up as **256 explicit entries** and a missing one fails (no `base + byte`
  numbering is assumed)
- a duplicated merge pair or a side not in the vocabulary fails (upstream is a HashMap, so the
  later entry wins silently)
- the subset extraction ({@link subset_keys}) covers every pair that can appear while scanning

The real assets (2 × a 32 MB `tokenizer.json`) exist only locally, so this runs on a **synthetic
minimal asset**. The synthetic asset is in a form `tokenizers.Tokenizer.from_str` accepts as is,
so the same script runs all the way through the path that captures the expected values
({@link build_fixture}).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from gemma_synthetic import PIECES
from gemma_synthetic import tokenizer_json as _tokenizer_json

from _shared import gemma_tokenizer as shared_tokenizer
from _shared.gemma_tokenizer import (
    ASSET_FORMAT,
    POST_BOS_EOS,
    POST_NONE,
    SHARED_ENCODE_CASES,
    UnsupportedTokenizerError,
    asset_payload,
    build_fixture,
    byte_token,
    compile_tokenizer,
    subset_keys,
)


class TestAcceptance:
    def test_the_real_shape_compiles(self) -> None:
        compiled = compile_tokenizer(_tokenizer_json())

        assert compiled.post_processor == POST_NONE
        assert compiled.unk_id == 3
        assert len(compiled.byte_ids) == 256
        assert compiled.merges == [(260, 261), (264, 262), (263, 260), (266, 261)]

    def test_a_wrapping_post_processor_is_recognised(self) -> None:
        """The EmbeddingGemma shape (`<bos>` … `<eos>`) — accepted by the same implementation as
        gemma4."""
        wrapped = _tokenizer_json(
            post_processor={
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 0}},
                ],
                "pair": [],
                "special_tokens": {},
            }
        )

        assert compile_tokenizer(wrapped).post_processor == POST_BOS_EOS

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model.type", "Unigram"),
            ("model.dropout", 0.1),
            ("model.byte_fallback", False),
            ("model.fuse_unk", False),
            ("model.ignore_merges", True),
            ("model.continuing_subword_prefix", "##"),
            ("model.end_of_word_suffix", "</w>"),
            ("model.unk_token", "<pad>"),
            ("truncation", {"max_length": 512}),
            ("padding", {"strategy": "BatchLongest"}),
        ],
    )
    def test_a_changed_flag_is_rejected(self, field: str, value: Any) -> None:
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(_tokenizer_json(**{field: value}))

    def test_a_different_normalizer_is_rejected(self) -> None:
        """A change in how the metaspace is inserted changes the whole id sequence (it raises no
        exception)."""
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(
                _tokenizer_json(
                    normalizer={"type": "Replace", "pattern": {"String": " "}, "content": "_"}
                )
            )

    def test_a_different_pre_tokenizer_is_rejected(self) -> None:
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(_tokenizer_json(pre_tokenizer={"type": "Metaspace"}))

    def test_a_reordered_decoder_chain_is_rejected(self) -> None:
        """Swapping the order of `ByteFallback` and `Replace` changes how a byte run is folded."""
        reordered = {
            "type": "Sequence",
            "decoders": [
                {"type": "ByteFallback"},
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "Fuse"},
            ],
        }
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(_tokenizer_json(decoder=reordered))

    def test_an_unknown_post_processor_is_rejected(self) -> None:
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(
                _tokenizer_json(
                    post_processor={
                        "type": "TemplateProcessing",
                        "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}],
                        "pair": [],
                        "special_tokens": {},
                    }
                )
            )

    def test_an_added_token_flag_that_changes_splitting_is_rejected(self) -> None:
        raw = _tokenizer_json()
        raw["added_tokens"][0]["lstrip"] = True
        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(raw)


class TestByteFallbackTable:
    def test_a_missing_byte_token_is_rejected(self) -> None:
        """The `base + byte` numbering is **a measured fact of this asset**, not a guarantee of the
        schema."""
        raw = _tokenizer_json()
        removed = raw["model"]["vocab"].pop(byte_token(0x7F))
        raw["model"]["vocab"]["<0x7f>"] = removed  # the lower-case spelling is not upstream's

        with pytest.raises(UnsupportedTokenizerError, match="0x7F"):
            compile_tokenizer(raw)

    def test_a_byte_shaped_token_beyond_the_256_is_rejected(self) -> None:
        """Closes the case where decode's byte test (an id set) and upstream's test (the shape of
        the spelling) disagree."""
        raw = _tokenizer_json()
        raw["model"]["vocab"]["<0xZZ>"] = len(raw["model"]["vocab"])

        with pytest.raises(UnsupportedTokenizerError, match="256"):
            compile_tokenizer(raw)


class TestMerges:
    def test_a_duplicated_pair_is_rejected(self) -> None:
        """Upstream is a HashMap, so a duplicated pair silently lets the later entry win and only
        the split rule becomes something else."""
        raw = _tokenizer_json()
        raw["model"]["merges"].append(["a", "b"])

        with pytest.raises(UnsupportedTokenizerError, match="duplicated"):
            compile_tokenizer(raw)

    def test_a_pair_whose_join_is_not_in_the_vocabulary_is_rejected(self) -> None:
        raw = _tokenizer_json()
        raw["model"]["merges"].append(["c", "a"])

        with pytest.raises(UnsupportedTokenizerError, match="not in the vocabulary"):
            compile_tokenizer(raw)

    def test_the_old_string_form_is_rejected(self) -> None:
        raw = _tokenizer_json()
        raw["model"]["merges"] = ["a b"]

        with pytest.raises(UnsupportedTokenizerError):
            compile_tokenizer(raw)


class TestAssetPayload:
    def test_the_payload_carries_the_shape_the_runtime_matches_on(self) -> None:
        compiled = compile_tokenizer(_tokenizer_json())
        payload = asset_payload(compiled, source={"path": "synthetic"})

        assert payload["format"] == ASSET_FORMAT
        assert payload["spec"]["postProcessor"] == POST_NONE
        assert payload["mergesCount"] == len(compiled.merges)
        assert payload["mergesText"].count("\n") == len(compiled.merges) - 1
        assert payload["vocab"][payload["byteIds"][0xFF]] == "<0xFF>"

    def test_the_vocabulary_is_written_in_id_order(self) -> None:
        compiled = compile_tokenizer(_tokenizer_json())
        payload = asset_payload(compiled, source={})

        assert payload["vocab"][0] == "<pad>"
        assert payload["vocab"][-1] == PIECES[-1]


class TestSubsetKeys:
    def test_every_substring_up_to_the_limit_is_covered(self) -> None:
        keys = subset_keys(["abc"], max_token_chars=3)

        assert keys == {"a", "b", "c", "ab", "bc", "abc"}

    def test_spaces_are_folded_to_metaspace_before_slicing(self) -> None:
        """The symbols are substrings of the **normalized** string — slicing the raw string would
        miss `▁a`."""
        assert "▁a" in subset_keys([" a"], max_token_chars=2)

    def test_longer_substrings_than_the_limit_are_not_needed(self) -> None:
        keys = subset_keys(["abcd"], max_token_chars=2)

        assert "abc" not in keys
        assert "cd" in keys


class TestFixture:
    def test_expectations_come_from_the_upstream_tokenizer(self, tmp_path: Any) -> None:
        """The path of the expected values does not go through compile (if it did, the match would
        become vacuous — ADR 0084 decision 7)."""
        # The upstream reference `tokenizers` is a transitive dependency of a family group
        # (transformers) and is absent from the default-sync CI — guarded by a SKIP like the other
        # family-dependent tests (the convention of `test_optional_group_imports.py`).
        pytest.importorskip("tokenizers")
        from _shared.gemma_tokenizer import EncodeCase

        raw = _tokenizer_json()
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        compiled = compile_tokenizer(raw)

        fixture = build_fixture(
            tokenizer_json=path,
            compiled=compiled,
            encode_cases=[EncodeCase("abc", "two merge steps apply", "abc")],
            decode_cases=[],
            source={},
        )

        [case] = fixture["encode"]
        # "abc" folds into one token in 2 steps: ("a","b") → ("ab","c").
        assert case["ids"] == [compiled.vocab.index("abc")]
        # The subset holds every symbol and pair that appears while scanning.
        assert [row for row in fixture["asset"]["merges"] if row[2] == 0]
        assert fixture["decode"][0]["name"] == "roundtrip-abc"
        assert fixture["decode"][0]["text"] == "abc"


class TestEmit:
    """`emit` writes the asset and the fixture **only once both are ready** (no path where only one
    of them gets newer)."""

    def test_a_failing_fixture_capture_leaves_no_asset_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(json.dumps(_tokenizer_json()), encoding="utf-8")
        asset_path = tmp_path / "out" / "tokenizer.asset.json"
        fixture_path = tmp_path / "out" / "fixture.json"

        def _failing_capture(**_: Any) -> dict[str, Any]:
            raise RuntimeError("upstream capture failed")

        # The source is written relative to the repository, so the synthetic asset's directory
        # stands in for the root.
        monkeypatch.setattr(shared_tokenizer, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(shared_tokenizer, "build_fixture", _failing_capture)

        with pytest.raises(RuntimeError, match="upstream capture failed"):
            shared_tokenizer.emit(
                tokenizer_json=tokenizer_path,
                asset_path=asset_path,
                fixture_path=fixture_path,
            )

        assert not asset_path.exists()
        assert not fixture_path.exists()


#: Variation selectors (`Mn` but combining class 0 — zero width, and visually indistinguishable
#: from the emoji next to them).
#: The source that spells `SHARED_ENCODE_CASES` (the check looks at the **spelling**, not the
#: run-time string — `\u200d` becomes a raw U+200D at run time, so `case.text` cannot tell how it
#: was written).
SHARED_CASES_SOURCE = Path(shared_tokenizer.__file__)

VARIATION_SELECTORS = frozenset(
    [*range(0xFE00, 0xFE10), *range(0xE0100, 0xE01F0)],
)


def _must_be_escaped(char: str) -> bool:
    """Whether a character, written raw into the source, "could vanish in an edit unnoticed".

    Split into 3 classes (not `Cf`/`Mn` as a whole — a mark that is `Mn` but **visible as a
    glyph**, like Thai `U+0E31`, may be written raw):

    1. `Cf` (format controls — ZWJ / ZWNJ and the like; zero width)
    2. combining characters with a non-zero combining class (`A` + `U+0301` is **visually
       identical** to `U+00C1`, and which one is written cannot be told by eye = the case's claim
       becomes unverifiable)
    3. variation selectors (a relative of 1; `Mn` with combining class 0, so 1 and 2 miss them)
    """
    return (
        unicodedata.category(char) == "Cf"
        or unicodedata.combining(char) != 0
        or ord(char) in VARIATION_SELECTORS
    )


class TestCaseSpelling:
    """MUST: invisible characters are written as `\\uXXXX` (the note on `SHARED_ENCODE_CASES`).

    Written raw, losing one character during an edit **does not turn anything red**: the expected
    values are always recaptured from upstream `tokenizers`, so the fixture is rebuilt with the
    weakened case, and a case named `zwj` keeps running green without a single ZWJ in it (only the
    case name and `why` still claim the original point).
    """

    def test_the_source_spells_invisible_characters_as_escapes(self) -> None:
        source = SHARED_CASES_SOURCE.read_text(encoding="utf-8")
        raw = sorted({f"U+{ord(char):04X}" for char in source if _must_be_escaped(char)})

        assert raw == [], f"raw invisible characters in the source: {raw}"

    @pytest.mark.parametrize(
        ("name", "codepoint", "count"),
        [
            ("zwj", 0x200D, 2),
            ("emoji", 0xFE0F, 1),
            ("combining", 0x0301, 2),
            ("combining", 0x0302, 1),
        ],
    )
    def test_the_cases_still_carry_the_boundary_they_name(
        self, name: str, codepoint: int, count: int
    ) -> None:
        """Rewriting into escapes did not remove the boundary (the counterpart that keeps the check
        above from being vacuous)."""
        case = next(entry for entry in SHARED_ENCODE_CASES if entry.name == name)

        assert case.text.count(chr(codepoint)) == count
