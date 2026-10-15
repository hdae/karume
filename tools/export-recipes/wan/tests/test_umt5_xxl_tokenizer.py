"""`wan/umt5_xxl_tokenizer.py` の約束事 — `karume-umt5-xxl` の単体のトークナイザ資産と TS の
フィクスチャ。

拒否の判定そのもの（空白の形・拒否の順）は合成の正本で叩く。上流のトークナイザを読むテストは、本家の
pin したトークナイザのファイルが HF キャッシュに無ければ SKIP（取得は
`uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer --fetch`）。Wan のトークナイザと
突き合わせるテストは Wan の pin の `tokenizer/` も要る（無ければ SKIP）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _shared.paths import SERIES_ROOT
from wan import prompt_clean as pc
from wan import umt5_tokenizer as ut
from wan import umt5_xxl_tokenizer as xt
from wan.sources import UMT5_SOURCES, WanSourceError, pinned_umt5_tokenizer
from wan.umt5_distribution import UMT5_TOKENIZER_FIELDS

#: `WhitespaceSplit` の空白集合の一部（合成 — タブから CR・U+0020・NBSP・全角空白）。
SPACE = pc.RangeSet([[9, 13], [32, 32], [160, 160], [12288, 12288]])

ADDED = ["</s>", "<s>"]


def fake_truth(
    transformers_ids: dict[str, list[int]], tokenizer_json_ids: dict[str, list[int]] | None = None
) -> ut.Truth:
    """文字列 → id 列の表で答える合成の正本（2 経路を別々に与えられる）。"""
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
    """前処理が無いと 2 経路が割れる空白の形（U+0020 以外の空白・末尾の U+0020）。"""

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
        """拒否の順は TS と同じ（追加語彙 → 空白の直後の ▁ → 空白の形）— 正本には聞かない。"""
        truth = fake_truth({})

        assert xt.expectation(truth, ADDED, SPACE, "a </s>\tb") == {"reject": "added-token"}
        assert xt.expectation(truth, ADDED, SPACE, "a ▁b\t") == {"reject": "metaspace"}

    def test_the_rest_is_judged_by_the_shared_wan_path(self) -> None:
        """語彙外（2 と 3 の割れ）と、それ以外の割れ方で落とす門は Wan と同じ 1 本。"""
        oov = fake_truth({"a 𠀋": [289, 273, 2, 1]}, {"a 𠀋": [289, 273, 3, 1]})
        broken = fake_truth({"x": [4, 5, 1]}, {"x": [4, 6, 1]})

        assert xt.expectation(oov, ADDED, SPACE, "a 𠀋") == {"reject": "oov"}
        with pytest.raises(ValueError, match="割れ方"):
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
    """本家の pin したトークナイザの正本（照合済みの 2 本だけの置き場から — 無ければ SKIP）。"""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("huggingface_hub")
    try:
        return xt.load_pinned_truth(xt.DEFAULT_MODEL, tmp_path_factory.mktemp("umt5-tokenizer"))
    except WanSourceError as error:
        pytest.skip(f"本家 umT5 のトークナイザが手元に無い: {error}")


def _committed() -> dict[str, Any]:
    return json.loads((xt.FIXTURE_DIR / xt.CASES_FIXTURE).read_text(encoding="utf-8"))


class TestAgainstUpstream:
    def test_the_backend_has_the_shape_the_typescript_copies(self, truth: ut.Truth) -> None:
        """正規化なし・WhitespaceSplit → Metaspace・unk_id 2 — Wan と同じ形（TS の T5 層がそのまま
        再現する）。"""
        ut.check_upstream_shape(truth)
        assert truth.transformers("a </s> b") == [289, 1, 748, 1]
        assert truth.tokenizer_json("a </s> b") == [289, 273, 1, 748, 1]

    def test_the_pinned_files_are_the_only_inputs(self, tmp_path: Path) -> None:
        pytest.importorskip("huggingface_hub")
        try:
            pinned_umt5_tokenizer(xt.DEFAULT_MODEL, tmp_path)
        except WanSourceError as error:
            pytest.skip(f"本家 umT5 のトークナイザが手元に無い: {error}")

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
        assert set(asset) == UMT5_TOKENIZER_FIELDS  # 配布の門が要求する欄の集合
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
        """Wan の資産の T5 の表と同じ（スコアだけが f64 の表記の差 — 最大 1e-14 未満）。"""
        series = SERIES_ROOT / ut.SERIES_NAME / ut.ASSET_FILE
        if not series.is_file():
            pytest.skip(f"Wan のトークナイザ資産の系列が無い: {series}")
        wan = json.loads(series.read_text(encoding="utf-8"))
        asset = xt.build_asset(truth, xt.DEFAULT_MODEL, {"tokenizers": "t", "transformers": "f"})

        for key in ("maxLength", "unkId", "eosId", "addedTokens", "space", "vocabText"):
            assert asset[key] == wan[key], key
        assert max(abs(a - b) for a, b in zip(asset["scores"], wan["scores"], strict=True)) < 1e-14
        assert set(wan) - set(asset) == {"promptClean"}

    def test_the_series_asset_is_what_the_generator_makes(self, truth: ut.Truth) -> None:
        series = SERIES_ROOT / xt.SERIES_NAME / xt.ASSET_FILE
        if not series.is_file():
            pytest.skip(f"umT5 のトークナイザ資産の系列が無い: {series}")

        rebuilt = json.dumps(
            xt.build_asset(truth, xt.DEFAULT_MODEL, xt.asset_versions()), ensure_ascii=False
        )

        assert rebuilt.encode("utf-8") == series.read_bytes()

    def test_the_committed_fixture_is_what_the_generator_makes(self, truth: ut.Truth) -> None:
        """commit 済みのフィクスチャを上流から作り直して一致を見る（境界のケースの狙いの決着も
        生成器が見る）。"""
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
        """本家と Wan のトークナイザはスコアの表記だけが違う — 受理したケースの id 列は同じ。"""
        try:
            wan = ut.load_truth(ut.tokenizer_snapshot())
        except WanSourceError as error:
            pytest.skip(f"Wan のトークナイザが手元に無い: {error}")
        accepted = [case for case in _committed()["cases"] if "ids" in case]

        assert accepted
        for case in accepted:
            assert wan.transformers(case["text"]) == case["ids"], case["id"]
            assert wan.tokenizer_json(case["text"]) == case["ids"], case["id"]
