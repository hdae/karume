"""Script-level contract of the tokenizer asset script (`embeddinggemma2/tokenizer.py`) (the part
that needs no real asset).

Only what turns into a **false PASS** when broken is pinned:

- the case names are unique together with the shared battery (the TS side looks a case up by
  name with `find`, which silently returns the first)
- every distinct prompt prefix has exactly one case, and the prompt check fails on a prefix
  without a case and on a case whose prefix upstream dropped
- the transformers check refuses any release but the pinned one (the synced environment's own
  release would otherwise produce the fixture without a trace)
- the git-tracked fixture holds exactly the cases the script declares (a case edited without
  regenerating the fixture would leave the TS side testing the old text)

The cases that read the pinned snapshot SKIP explicitly without it (and without tokenizers /
transformers at the pinned release).
"""

from __future__ import annotations

import json
from importlib import metadata

import pytest

from _shared.gemma_tokenizer import SHARED_ENCODE_CASES
from embeddinggemma2 import sources, tokenizer


def _snapshot_file(name: str):
    path = sources.SNAPSHOT_DIR / name
    if not path.is_file():
        pytest.skip(f"the pinned snapshot is not placed at {sources.SNAPSHOT_DIR}")
    return path


class TestCases:
    def test_the_case_names_are_unique_against_the_shared_battery(self) -> None:
        names = [case.name for case in (*SHARED_ENCODE_CASES, *tokenizer.CASES)]

        assert sorted(names) == sorted(set(names))

    def test_every_case_carries_a_name_a_reason_and_a_text(self) -> None:
        for case in tokenizer.CASES:
            assert case.name
            assert case.why
            assert case.text

    def test_every_prompt_prefix_has_exactly_one_case(self) -> None:
        prompted = [case for case in tokenizer.CASES if case.name.startswith("prompt-")]
        for prefix in tokenizer.PROMPT_PREFIXES:
            with_text = [
                case for case in prompted if case.text.startswith(prefix) and case.text != prefix
            ]
            assert len(with_text) == 1, prefix
        assert len(set(tokenizer.PROMPT_PREFIXES)) == len(tokenizer.PROMPT_PREFIXES)

    def test_the_long_case_is_one_of_the_cases(self) -> None:
        assert tokenizer.LONG_CASE in tokenizer.CASES


class TestAssertPrompts:
    def test_the_upstream_shape_passes(self) -> None:
        """Several prompt names share one prefix upstream (20 names, 8 prefixes)."""
        prompts = {f"name{index}": prefix for index, prefix in enumerate(tokenizer.PROMPT_PREFIXES)}
        prompts["alias"] = tokenizer.PROMPT_PREFIXES[0]

        tokenizer.assert_prompts({"prompts": prompts})

    def test_a_prefix_without_a_case_fails(self) -> None:
        prompts = dict(enumerate(tokenizer.PROMPT_PREFIXES))
        prompts[99] = "task: new task | query: "

        with pytest.raises(tokenizer.TokenizerScriptError, match="new task"):
            tokenizer.assert_prompts({"prompts": prompts})

    def test_a_prefix_upstream_dropped_fails(self) -> None:
        prompts = dict(enumerate(tokenizer.PROMPT_PREFIXES[1:]))

        with pytest.raises(tokenizer.TokenizerScriptError, match="no longer upstream"):
            tokenizer.assert_prompts({"prompts": prompts})

    def test_no_prompts_fails(self) -> None:
        with pytest.raises(tokenizer.TokenizerScriptError, match="no prompts"):
            tokenizer.assert_prompts({})

    def test_the_pinned_snapshot_s_prompts_pass(self) -> None:
        path = _snapshot_file("config_sentence_transformers.json")
        sources.assert_pinned(sources.SNAPSHOT_DIR, ["config_sentence_transformers.json"])

        tokenizer.assert_prompts(json.loads(path.read_text(encoding="utf-8")))


class TestTransformersVersion:
    def test_the_pinned_release_passes(self) -> None:
        tokenizer.assert_transformers_version("5.23.0")

    @pytest.mark.parametrize("version", ["5.14.1", "5.23.1", "5.24.0"])
    def test_any_other_release_fails_with_the_command(self, version: str) -> None:
        with pytest.raises(tokenizer.TokenizerScriptError, match="uv run --with"):
            tokenizer.assert_transformers_version(version)


class TestOutputs:
    def test_the_asset_goes_to_the_tokenizer_series(self) -> None:
        assert tokenizer.ASSET_PATH.parent.name == "embeddinggemma-2-tokenizer"
        assert tokenizer.ASSET_PATH.name == "tokenizer.json"

    def test_every_file_the_script_reads_is_pinned(self) -> None:
        assert set(tokenizer.READ_FILES) <= set(sources.PIN.sha256)

    def test_the_tracked_fixture_holds_the_declared_cases(self) -> None:
        fixture = json.loads(tokenizer.FIXTURE_PATH.read_text(encoding="utf-8"))

        assert [(row["name"], row["text"]) for row in fixture["encode"]] == [
            (case.name, case.text) for case in (*SHARED_ENCODE_CASES, *tokenizer.CASES)
        ]
        assert fixture["source"]["path"] == f"{sources.PIN.label}/tokenizer.json"
        assert fixture["source"]["sha256"] == sources.PIN.sha256["tokenizer.json"]
        assert fixture["spec"]["postProcessor"] == "bos-eos"


class TestPinnedSnapshot:
    def test_the_long_case_is_long_enough(self) -> None:
        tokenizers = pytest.importorskip("tokenizers")
        path = _snapshot_file("tokenizer.json")
        reference = tokenizers.Tokenizer.from_file(str(path))

        count = len(reference.encode(tokenizer.LONG_CASE.text, add_special_tokens=True).ids)

        assert count >= tokenizer.MIN_LONG_TOKENS

    def test_transformers_and_tokenizers_agree_on_every_case(self) -> None:
        """Needs the pinned transformers (`uv run --with 'transformers==5.23.0' pytest …`)."""
        pytest.importorskip("transformers")
        version = metadata.version("transformers")
        if version != tokenizer.REQUIRED_TRANSFORMERS:
            pytest.skip(
                f"transformers {version}, the check needs {tokenizer.REQUIRED_TRANSFORMERS}"
            )
        _snapshot_file("tokenizer.json")
        sources.assert_pinned(sources.SNAPSHOT_DIR, tokenizer.READ_FILES)

        tokenizer.cross_check(sources.SNAPSHOT_DIR, (*SHARED_ENCODE_CASES, *tokenizer.CASES))
