"""Contracts of `qwen3_5.tokenizer` (the Qwen3.5 tokenizer asset and its fixture).

A broken layer here changes **only the ids** — the shapes still match and the image changes. Two
kinds of checks:

- **Against the committed fixture** (always run): the Python mirror of the TS implementation
  reproduces every id sequence from the fixture's tables, the chat-prompt mirror reproduces every
  rendered string, and the fixture's case set is the script's.
- **Against upstream** (runs when the pinned tokenizer files are in the HF cache, SKIP otherwise):
  transformers still renders and tokenizes every case as committed — so an upstream drift is caught
  without regenerating.

The exhaustive checks of the folded Unicode tables (every code point × every probe) are the emit
gate of `qwen3_5.tokenizer` (minutes long), not pytest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anima import text as at
from qwen3_5 import tokenizer as qt
from qwen3_5 import tokenizer_cases as tc

FIXTURE: dict[str, Any] = json.loads(qt.FIXTURE_PATH.read_text(encoding="utf-8"))
ACCEPTED = [case for case in FIXTURE["promptCases"] if "rendered" in case]
REFUSED = [case for case in FIXTURE["promptCases"] if "reject" in case]


def fixture_reference() -> at.Qwen2Reference:
    """The mirror over the fixture's subset (exactly what the TS test reads)."""
    qwen = FIXTURE["qwen"]
    return at.Qwen2Reference(
        vocab=qwen["vocab"],
        merges={(left, right): rank for left, right, rank in qwen["merges"]},
        added=dict(qwen["addedTokens"]),
        classes=qwen["classes"],
        case_fold={cp: target for cp, target in qwen["caseFold"]},
        nfc_segments=qwen["nfcSegments"],
    )


class TestFixtureShape:
    def test_the_prompt_cases_are_the_scripts(self):
        want = [(name, prompt) for name, prompt, _ in qt.prompt_inputs()]
        got = [(case["id"], case["prompt"]) for case in FIXTURE["promptCases"]]

        assert got == want

    def test_the_text_cases_are_the_scripts(self):
        tables = {"nfcSegments": FIXTURE["qwen"]["nfcSegments"]}
        want = [(name, text) for name, text, _ in qt.text_inputs(tables)]
        got = [(case["id"], case["text"]) for case in FIXTURE["textCases"]]

        assert got == want
        assert len(got) == len(tc.TEXT_CASES) + qt.RANDOM_TEXT_CASES

    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in FIXTURE["promptCases"] + FIXTURE["textCases"]]

        assert len(set(ids)) == len(ids)

    def test_the_envelope_is_the_assets(self):
        assert FIXTURE["assetFormat"] == qt.ASSET_FORMAT
        assert FIXTURE["maxLength"] == qt.MAX_LENGTH
        assert FIXTURE["source"] == qt.source_record()
        assert len(FIXTURE["qwen"]["addedTokens"]) == 33

    def test_the_boundary_cases_sit_on_the_bound(self):
        tokens = {case["id"]: case["tokens"] for case in ACCEPTED}

        for name, want in qt.BOUNDARY_TOKENS.items():
            assert tokens[name] == want, name

    def test_the_empty_prompt_is_ten_tokens(self):
        empty = next(case for case in ACCEPTED if case["id"] == "empty")

        assert empty["ids"] == [248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198]

    def test_python_space_is_str_isspace(self):
        assert FIXTURE["pythonSpace"] == qt.python_space()
        assert len(FIXTURE["pythonSpace"]) == 29


class TestMirrors:
    def test_the_bpe_mirror_reproduces_every_case_from_the_fixture_tables(self):
        ref = fixture_reference()

        for case in ACCEPTED:
            assert ref.encode(case["rendered"], qt.MAX_LENGTH) == case["ids"], case["id"]
        for case in FIXTURE["textCases"]:
            assert ref.encode(case["text"]) == case["ids"], case["id"]

    def test_the_chat_mirror_reproduces_every_rendered_prompt(self):
        for case in ACCEPTED:
            assert qt.render_chat_prompt(case["prompt"]) == case["rendered"], case["id"]

    def test_the_chat_mirror_refuses_what_the_template_refuses(self):
        assert REFUSED, "no refused case"
        for case in REFUSED:
            with pytest.raises(qt.Qwen35TokenizerError):
                qt.render_chat_prompt(case["prompt"])


def _backend(**patch: Any) -> dict[str, Any]:
    """A backend JSON in the shape the engine reproduces (`check_backend_shape` accepts it)."""
    backend: dict[str, Any] = {
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": "x"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
        },
        "added_tokens": [
            {
                "content": "<x>",
                "lstrip": False,
                "rstrip": False,
                "single_word": False,
                "normalized": False,
            }
        ],
    }
    backend.update(patch)
    return backend


class TestBackendShape:
    def test_the_engines_shape_is_accepted(self):
        qt.check_backend_shape(_backend())

    @pytest.mark.parametrize(
        "patch",
        [
            {"normalizer": {"type": "NFKC"}},
            {"post_processor": {"type": "TemplateProcessing", "single": []}},
            {
                "model": {
                    **_backend()["model"],
                    "byte_fallback": True,
                }
            },
            {
                "added_tokens": [
                    {
                        "content": "<x>",
                        "lstrip": True,
                        "rstrip": False,
                        "single_word": False,
                        "normalized": False,
                    }
                ]
            },
        ],
    )
    def test_a_shape_the_engine_does_not_reproduce_fails(self, patch: dict[str, Any]):
        with pytest.raises(qt.Qwen35TokenizerError):
            qt.check_backend_shape(_backend(**patch))


@pytest.fixture(scope="module")
def upstream(tmp_path_factory: pytest.TempPathFactory) -> Any:
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    directory: Path = tmp_path_factory.mktemp("qwen3_5-tokenizer")
    try:
        qt.pinned_tokenizer_files(directory, fetch=False)
    except qt.Qwen35TokenizerError as cause:
        pytest.skip(f"the pinned tokenizer is not in the HF cache ({cause})")
    return qt.load_truth(directory)


class TestAgainstUpstream:
    def test_upstream_still_renders_and_tokenizes_every_case(self, upstream: Any):
        for case in ACCEPTED:
            assert qt.upstream_render(upstream, case["prompt"]) == case["rendered"], case["id"]
            assert qt.upstream_ids(upstream, case["rendered"], truncate=True) == case["ids"]
        for case in REFUSED:
            assert qt.upstream_render(upstream, case["prompt"]) is None, case["id"]
        for case in FIXTURE["textCases"]:
            assert qt.upstream_ids(upstream, case["text"], truncate=False) == case["ids"]

    def test_the_built_backend_has_the_engines_shape(self, upstream: Any):
        qt.check_backend_shape(qt.backend_json(upstream))
