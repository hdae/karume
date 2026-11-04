"""The contract of `wan/umt5_xxl_tokenizer.py` — the standalone tokenizer asset of `karume-umt5-xxl`
and the TS fixture.

The rejection rules themselves (the whitespace shapes, the rejection order) are exercised against a
synthetic source of truth. Tests that read the upstream tokenizer SKIP when the pinned tokenizer
files of the original model are not in the HF cache (fetch them with
`uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer --fetch`). Tests that compare with
the Wan tokenizer also need `tokenizer/` of Wan's pin (SKIP when absent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from real_assets import mirror_asset

from wan import prompt_clean as pc
from wan import umt5_tokenizer as ut
from wan import umt5_xxl_tokenizer as xt
from wan.distribution import WAN22, WAN_TOKENIZER_ROLE
from wan.sources import UMT5_SOURCES, WanSourceError, pinned_umt5_tokenizer
from wan.umt5_distribution import (
    UMT5_DEFAULT_MODEL,
    UMT5_REPO_NAME,
    UMT5_TOKENIZER_FIELDS,
    UMT5_TOKENIZER_ROLE,
)

#: Part of the `WhitespaceSplit` whitespace set (synthetic — tab through CR, U+0020, NBSP, the
#: ideographic space).
SPACE = pc.RangeSet([[9, 13], [32, 32], [160, 160], [12288, 12288]])

ADDED = ["</s>", "<s>"]


def fake_truth(
    transformers_ids: dict[str, list[int]], tokenizer_json_ids: dict[str, list[int]] | None = None
) -> ut.Truth:
    """A synthetic source of truth that answers from a string → id-sequence table (the two paths can
    be given separately)."""
    plain = tokenizer_json_ids if tokenizer_json_ids is not None else transformers_ids
    return ut.Truth(
        transformers=lambda text: transformers_ids[text],
        masked_length=lambda text: len(transformers_ids[text]),
        tokenizer_json=lambda text: plain[text],
        backend={},
        raw={},
        pre_tokenize=lambda text: ["▁" + word for word in text.split()],
    )


class TestTheWhitespaceRule:
    """Whitespace shapes that split the two paths without preprocessing (whitespace other than
    U+0020, a trailing U+0020)."""

    @pytest.mark.parametrize(
        "text", ["a\tcat", "a cat\nsits", "a\r\nb", "a cat", "猫　犬", "a cat ", "   "]
    )
    def test_other_whitespace_and_a_trailing_space_split_the_paths(self, text: str) -> None:
        assert xt.whitespace_split(SPACE, text)

    @pytest.mark.parametrize("text", ["a cat", "  a cat", "a   cat  sits", "a▁b", ""])
    def test_leading_and_inner_spaces_do_not(self, text: str) -> None:
        assert not xt.whitespace_split(SPACE, text)


class TestExpectation:
    def test_agreeing_paths_are_accepted_with_both_id_lists(self) -> None:
        truth = fake_truth({"a cat": [289, 6283, 1]})

        assert xt.expectation(truth, ADDED, SPACE, "a cat") == {
            "ids": [289, 6283, 1],
            "tokenizerJsonIds": [289, 6283, 1],
        }

    def test_whitespace_is_rejected_with_the_split_on_record(self) -> None:
        truth = fake_truth({"a ": [289, 1]}, {"a ": [289, 273, 1]})

        assert xt.expectation(truth, ADDED, SPACE, "a ") == {
            "reject": "whitespace",
            "split": {"transformers": [289, 1], "tokenizerJson": [289, 273, 1]},
        }

    def test_added_tokens_and_space_metaspace_come_before_whitespace(self) -> None:
        """The rejection order is the same as on the TS side (added tokens → ▁ right after
        whitespace → whitespace shape) — the source of truth is not consulted."""
        truth = fake_truth({})

        assert xt.expectation(truth, ADDED, SPACE, "a </s>\tb") == {"reject": "added-token"}
        assert xt.expectation(truth, ADDED, SPACE, "a ▁b\t") == {"reject": "metaspace"}

    def test_the_rest_is_judged_by_the_shared_wan_path(self) -> None:
        """Out-of-vocabulary (a 2 versus 3 split) and the check that fails any other split are the
        same single path as Wan's."""
        oov = fake_truth({"a 𠀋": [289, 273, 2, 1]}, {"a 𠀋": [289, 273, 3, 1]})
        broken = fake_truth({"x": [4, 5, 1]}, {"x": [4, 6, 1]})

        assert xt.expectation(oov, ADDED, SPACE, "a 𠀋") == {"reject": "oov"}
        with pytest.raises(ValueError, match="in a way other than out-of-vocabulary"):
            xt.expectation(broken, ADDED, SPACE, "x")


class TestTheSpelling:
    def test_the_default_model_is_the_distribution_model(self) -> None:
        from wan.umt5_distribution import UMT5_DEFAULT_MODEL

        assert xt.DEFAULT_MODEL == UMT5_DEFAULT_MODEL
        assert xt.DEFAULT_MODEL in UMT5_SOURCES

    def test_every_boundary_case_has_an_intended_outcome(self) -> None:
        assert [case_id for case_id, _, _ in xt.CASES] == list(xt.EXPECTED_OUTCOMES)

    def test_the_committed_fixture_is_small(self) -> None:
        assert (xt.FIXTURE_DIR / xt.CASES_FIXTURE).stat().st_size < 200_000


@pytest.fixture(scope="module")
def truth(tmp_path_factory: pytest.TempPathFactory) -> ut.Truth:
    """The source of truth of the original model's pinned tokenizer (from a directory holding only
    the two verified files — SKIP when absent)."""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("huggingface_hub")
    try:
        return xt.load_pinned_truth(xt.DEFAULT_MODEL, tmp_path_factory.mktemp("umt5-tokenizer"))
    except WanSourceError as error:
        pytest.skip(f"the original umT5 tokenizer is not available locally: {error}")


def _committed() -> dict[str, Any]:
    return json.loads((xt.FIXTURE_DIR / xt.CASES_FIXTURE).read_text(encoding="utf-8"))


class TestAgainstUpstream:
    def test_the_backend_has_the_shape_the_typescript_copies(self, truth: ut.Truth) -> None:
        """No normalizer, WhitespaceSplit → Metaspace, unk_id 2 — the same shape as Wan's (the TS
        T5 layer reproduces it as is)."""
        ut.check_upstream_shape(truth)
        assert truth.transformers("a </s> b") == [289, 1, 748, 1]
        assert truth.tokenizer_json("a </s> b") == [289, 273, 1, 748, 1]

    def test_the_pinned_files_are_the_only_inputs(self, tmp_path: Path) -> None:
        pytest.importorskip("huggingface_hub")
        try:
            pinned_umt5_tokenizer(xt.DEFAULT_MODEL, tmp_path)
        except WanSourceError as error:
            pytest.skip(f"the original umT5 tokenizer is not available locally: {error}")

        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
            UMT5_SOURCES[xt.DEFAULT_MODEL].tokenizer
        )

    def test_the_asset_is_the_t5_tables_without_prompt_cleaning(self, truth: ut.Truth) -> None:
        asset = xt.build_asset(truth, xt.DEFAULT_MODEL, {"tokenizers": "t", "transformers": "f"})

        assert list(asset) == [
            "format",
            "source",
            "versions",
            "maxLength",
            "unkId",
            "eosId",
            "addedTokens",
            "space",
            "vocabText",
            "scores",
        ]
        assert (
            set(asset) == UMT5_TOKENIZER_FIELDS
        )  # the set of fields the distribution gate requires
        assert asset["format"] == xt.ASSET_FORMAT == "karume-umt5-tokenizer/1"
        assert asset["source"] == {
            "repo": "google/umt5-xxl",
            "revision": UMT5_SOURCES[xt.DEFAULT_MODEL].source.revision,
            "files": dict(UMT5_SOURCES[xt.DEFAULT_MODEL].tokenizer),
        }
        assert asset["versions"] == {"tokenizers": "t", "transformers": "f"}
        assert asset["maxLength"] == ut.MAX_LENGTH == 512
        assert (asset["unkId"], asset["eosId"]) == (2, 1)
        assert len(asset["scores"]) == 256_300
        assert len(asset["addedTokens"]) == 304

    def test_the_t5_tables_match_the_wan_asset_up_to_score_spelling(self, truth: ut.Truth) -> None:
        """The same T5 tables as the Wan asset the published Wan2.2 mirror ships (only the scores
        differ, in their f64 spelling — by less than 1e-14)."""
        shipped = mirror_asset(WAN22.repo_name, WAN22.default_model, WAN_TOKENIZER_ROLE)
        wan = json.loads(shipped.read_text(encoding="utf-8"))
        asset = xt.build_asset(truth, xt.DEFAULT_MODEL, {"tokenizers": "t", "transformers": "f"})

        for key in ("maxLength", "unkId", "eosId", "addedTokens", "space", "vocabText"):
            assert asset[key] == wan[key], key
        assert max(abs(a - b) for a, b in zip(asset["scores"], wan["scores"], strict=True)) < 1e-14
        assert set(wan) - set(asset) == {"promptClean"}

    def test_the_series_asset_is_what_the_generator_makes(self, truth: ut.Truth) -> None:
        """The tokenizer asset the `karume-umt5-xxl` mirror ships is what the generator makes."""
        shipped = mirror_asset(UMT5_REPO_NAME, UMT5_DEFAULT_MODEL, UMT5_TOKENIZER_ROLE)

        rebuilt = json.dumps(
            xt.build_asset(truth, xt.DEFAULT_MODEL, xt.asset_versions()), ensure_ascii=False
        )

        assert rebuilt.encode("utf-8") == shipped.read_bytes()

    def test_the_committed_fixture_is_what_the_generator_makes(self, truth: ut.Truth) -> None:
        """Rebuild the committed fixture from upstream and check that it matches (the generator
        also checks that each boundary case lands where it aims)."""
        committed = _committed()
        asset = xt.build_asset(truth, xt.DEFAULT_MODEL, committed["versions"])

        regenerated = xt.build_fixture(truth, asset, xt.build_cases(truth))

        assert regenerated == committed

    def test_the_fixture_covers_every_outcome(self) -> None:
        outcomes = {xt.outcome(case) for case in _committed()["cases"]}

        assert outcomes == {
            "accepted",
            "added-token",
            "metaspace",
            "whitespace",
            "oov",
            "too-long",
            "too-short",
        }

    def test_the_ids_equal_the_wan_tokenizer_ids(self, truth: ut.Truth) -> None:
        """The original and the Wan tokenizers differ only in how the scores are spelled — the id
        sequences of the accepted cases are the same."""
        try:
            wan = ut.load_truth(ut.tokenizer_snapshot())
        except WanSourceError as error:
            pytest.skip(f"the Wan tokenizer is not available locally: {error}")
        accepted = [case for case in _committed()["cases"] if "ids" in case]

        assert accepted
        for case in accepted:
            assert wan.transformers(case["text"]) == case["ids"], case["id"]
            assert wan.tokenizer_json(case["text"]) == case["ids"], case["id"]
