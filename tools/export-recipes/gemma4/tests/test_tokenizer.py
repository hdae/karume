"""Script-level contract of the tokenizer asset script (`gemma4/tokenizer.py`) (the part that
needs no real asset).

The real asset (the 32MB `tokenizer.json`) exists only locally, so what is checked here are rules
that hold **on constants and paths alone** (neither transformers nor torch is needed). The
counterpart for `chat.py` is `TestScopeGate` of {@link gemma4.tests.test_chat}.

Only what turns into a **false PASS** when broken is pinned:

- names are unique together with the shared battery
  ({@link _shared.gemma_tokenizer.SHARED_ENCODE_CASES}). `emit` only concatenates the two lists
  and does not check uniqueness, so a name collision puts two entries of the same name into the
  parity fixture. The TS side looks up one entry by name in places
  (`fixture.encode.find(…)` in `packages/models/tests/gemma4_tokenizer_test.ts`), and `find`
  silently returns the first — **it measures another case and goes green** (rather than failing
  as unreadable)
- the spelling of the output path matches the distribution recipe's series name (writer and reader
  build it from the same one word — the declaration in `gemma4/distribution.py`)
- the parity fixture's output path is an existing file tracked in git (if it disappears, the TS
  side loses what it reads)
"""

from __future__ import annotations

import json

import pytest

from _shared import gemma_tokenizer
from _shared.gemma_tokenizer import SHARED_ENCODE_CASES, emit
from gemma4 import sources, tokenizer
from gemma4.distribution import (
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_TOKENIZER_FILE,
    GEMMA4_TOKENIZER_SERIES,
    GEMMA4_TOKENIZER_SUFFIX,
    gemma4_series_name,
)


class TestScopeGate:
    def test_the_case_names_are_unique_against_the_shared_battery(self) -> None:
        """On a name collision the TS side's `find` silently measures another case (false PASS)."""
        names = [case.name for case in (*SHARED_ENCODE_CASES, *tokenizer.CASES)]

        assert sorted(names) == sorted(set(names))

    def test_every_shipped_case_carries_a_name_and_a_text(self) -> None:
        """The name is the fixture's index, the text is the very thing measured (empty, the check
        becomes vacuous).

        NOTE: the shared battery holds `empty` (the empty string) on purpose, but the
        family-specific additions are cases that "exercise this checkpoint's spellings", so there
        is no reason for their text to be empty.
        """
        for case in tokenizer.CASES:
            assert case.name
            assert case.text
            assert case.why


class TestOutputPaths:
    def test_the_asset_path_is_the_series_name_the_distribution_reads(self) -> None:
        """The writer (script) and the reader (distribution recipe) build the series name from the
        same one word."""
        expected = gemma4_series_name(GEMMA4_DEFAULT_MODEL, GEMMA4_TOKENIZER_SUFFIX)

        assert tokenizer.ASSET_PATH.parent.name == expected
        assert tokenizer.ASSET_PATH.name == GEMMA4_TOKENIZER_FILE

    def test_the_parity_fixture_is_a_tracked_file(self) -> None:
        """The fixture is tracked in git (the file the TS side's `gemma4_tokenizer_test.ts`
        reads)."""
        assert tokenizer.FIXTURE_PATH.is_file()


class TestPerModelSeries:
    @pytest.mark.parametrize("model", ["e4b", "12b"])
    def test_the_model_has_its_own_series_and_fixture(self, model: str) -> None:
        """The paths come from the same word as the distribution's own series for the model."""
        assert tokenizer.asset_path(model).parent.name == gemma4_series_name(
            model, GEMMA4_TOKENIZER_SUFFIX
        )
        assert tokenizer.fixture_path(model).name == f"gemma4-{model}-parity.json"
        assert GEMMA4_TOKENIZER_SERIES[model] == model

    def test_e2b_keeps_its_paths(self) -> None:
        assert tokenizer.asset_path("e2b") == tokenizer.ASSET_PATH
        assert tokenizer.fixture_path("e2b") == tokenizer.FIXTURE_PATH
        assert GEMMA4_TOKENIZER_SERIES["e2b"] == "e2b"

    @pytest.mark.parametrize("model", tokenizer.MODELS)
    def test_the_case_names_are_unique(self, model: str) -> None:
        names = [
            case.name
            for case in (*SHARED_ENCODE_CASES, *tokenizer.CASES, *tokenizer.EXTRA_CASES[model])
        ]

        assert sorted(names) == sorted(set(names))

    @pytest.mark.parametrize(
        ("model", "joined_by"),
        [("12b", "e2b"), ("12b", "e4b"), ("e4b", "e2b")],
        ids=["12b-vs-e2b", "12b-vs-e4b", "e4b-vs-e2b"],
    )
    def test_the_cases_run_into_the_merges_the_two_tokenizers_disagree_on(
        self, model: str, joined_by: str
    ) -> None:
        """Each case of `model` splits differently under the merges of `joined_by` — rebuilt from
        `model`'s own `tokenizer.json` with the merge list of `joined_by`'s compiled asset (the
        vocabulary, added tokens and pipeline of the files are the same), so the cases measure the
        difference they are there for. A 12B case is a merge 12B dropped (12B splits more); an E4B
        case is a merge E2B dropped (E4B splits less)."""
        tokenizers = pytest.importorskip("tokenizers")
        pytest.importorskip("huggingface_hub")
        upstream = sources.tokenizer_dir(model) / "tokenizer.json"
        other = tokenizer.asset_path(joined_by)
        if not upstream.is_file() or not other.is_file():
            pytest.skip(f"needs {upstream} and {other}")
        raw = json.loads(upstream.read_text(encoding="utf-8"))
        compiled = json.loads(other.read_text(encoding="utf-8"))
        vocab = compiled["vocab"]
        other_merges = [
            [vocab[int(left)], vocab[int(right)]]
            for left, right in (line.split(" ") for line in compiled["mergesText"].split("\n"))
        ]
        own = tokenizers.Tokenizer.from_str(json.dumps(raw))
        raw["model"]["merges"] = other_merges
        like_other = tokenizers.Tokenizer.from_str(json.dumps(raw))

        cases = [case for case in tokenizer.EXTRA_CASES[model] if "dropped-merge" in case.name]
        assert cases, model
        for case in cases:
            ours = own.encode(case.text, add_special_tokens=False).ids
            theirs = like_other.encode(case.text, add_special_tokens=False).ids
            assert ours != theirs, case.name
            if model == "12b":
                assert len(ours) > len(theirs), case.name
            else:
                assert len(ours) < len(theirs), case.name

    def test_e2b_s_case_is_the_renamed_entry(self) -> None:
        """E2B's fix renamed id 249788 (`gemma4.tokenizer.EXTRA_CASES`): the character no longer
        maps to it on E2B, but still does on E4B (needs both compiled assets)."""
        tokenizers = pytest.importorskip("tokenizers")
        (case,) = tokenizer.EXTRA_CASES["e2b"]
        upstream = {
            model: sources.tokenizer_dir(model) / "tokenizer.json" for model in ("e2b", "e4b")
        }
        if not all(path.is_file() for path in upstream.values()):
            pytest.skip(f"needs {' and '.join(map(str, upstream.values()))}")
        ids = {
            model: tokenizers.Tokenizer.from_file(str(path))
            .encode(case.text, add_special_tokens=False)
            .ids
            for model, path in upstream.items()
        }

        assert ids["e4b"] == [249788]
        assert 249788 not in ids["e2b"]


class TestEmitSourcePath:
    def test_an_outside_checkpoint_is_recorded_by_the_name_given(
        self, tmp_path, monkeypatch
    ) -> None:
        """A hub snapshot is outside the repository, so its path cannot be repository-relative;
        the fixture records the `<repo>@<revision>` spelling instead."""
        pytest.importorskip("tokenizers")
        from gemma_synthetic import tokenizer_json

        # The fixture is git-tracked in real use; here it lands in `tmp_path`, which the closing
        # hint must be able to spell relative to the "repository".
        monkeypatch.setattr(gemma_tokenizer, "REPO_ROOT", tmp_path)

        upstream = tmp_path / "snapshot" / "tokenizer.json"
        upstream.parent.mkdir()
        upstream.write_text(json.dumps(tokenizer_json()), encoding="utf-8")
        asset, fixture = tmp_path / "asset.json", tmp_path / "fixture.json"

        emit(
            tokenizer_json=upstream,
            asset_path=asset,
            fixture_path=fixture,
            source_path="google/example@" + "0" * 40 + "/tokenizer.json",
        )

        for path in (asset, fixture):
            source = json.loads(path.read_text(encoding="utf-8"))["source"]
            assert source["path"] == "google/example@" + "0" * 40 + "/tokenizer.json"
