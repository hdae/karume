"""トークナイザ資産の台本（`gemma4/tokenizer.py`）の台本レベルの約束事（実資産不要分）。

実資産（32MB の `tokenizer.json`）はローカルにしか無いので、ここが見るのは**定数と path
だけ**で成立する規律（transformers も torch も要らない）。`chat.py` の同役は
{@link gemma4.tests.test_chat} の `TestScopeGate`。

固定するのは、壊れると**偽 PASS** になる側だけ:

- 名前が共有電池（{@link _shared.gemma_tokenizer.SHARED_ENCODE_CASES}）と合わせて一意で
  あること。`emit` は 2 つの列を連結するだけで一意性を見ないので、名前が衝突するとパリティ
  フィクスチャに同名 2 件が並ぶ。TS 側は名前で 1 件を引く場所があり
  （`packages/models/tests/gemma4_tokenizer_test.ts` の `fixture.encode.find(…)`）、`find` は
  黙って先頭を返す — **別のケースを測って緑になる**（落ちて読めなくなるのではない）
- 出力先の綴りが配布 recipe の系列名と一致すること（書き手と読み手が同じ 1 語から組む —
  `gemma4/distribution.py` の宣言）
- パリティフィクスチャの出力先が git 管理の実在ファイルであること（消えると TS 側が読む先を
  失う）
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
        """名前が衝突すると TS 側の `find` が別のケースを黙って測る（偽 PASS）。"""
        names = [case.name for case in (*SHARED_ENCODE_CASES, *tokenizer.CASES)]

        assert sorted(names) == sorted(set(names))

    def test_every_shipped_case_carries_a_name_and_a_text(self) -> None:
        """名前はフィクスチャの索引・本文は測る対象そのもの（空だと門が恒真になる）。

        NOTE: 共有電池は `empty`（空文字列）を意図して持つが、family 固有の追加分は
        「この checkpoint の綴りを踏む」ためのケースなので本文が空になる理由が無い。
        """
        for case in tokenizer.CASES:
            assert case.name
            assert case.text
            assert case.why


class TestOutputPaths:
    def test_the_asset_path_is_the_series_name_the_distribution_reads(self) -> None:
        """書き手（台本）と読み手（配布 recipe）が同じ 1 語から系列名を組む。"""
        expected = gemma4_series_name(GEMMA4_DEFAULT_MODEL, GEMMA4_TOKENIZER_SUFFIX)

        assert tokenizer.ASSET_PATH.parent.name == expected
        assert tokenizer.ASSET_PATH.name == GEMMA4_TOKENIZER_FILE

    def test_the_parity_fixture_is_a_tracked_file(self) -> None:
        """フィクスチャは git 管理（TS 側 `gemma4_tokenizer_test.ts` が読む現物）。"""
        assert tokenizer.FIXTURE_PATH.is_file()


class TestTwelveB:
    def test_12b_has_its_own_series_and_fixture(self) -> None:
        """12B's paths come from the same word as the distribution's own series for it."""
        assert tokenizer.asset_path("12b").parent.name == gemma4_series_name(
            "12b", GEMMA4_TOKENIZER_SUFFIX
        )
        assert tokenizer.asset_path("e2b") == tokenizer.ASSET_PATH
        assert tokenizer.fixture_path("e2b") == tokenizer.FIXTURE_PATH
        assert tokenizer.fixture_path("12b").name == "gemma4-12b-parity.json"
        assert GEMMA4_TOKENIZER_SERIES["12b"] == "12b"

    def test_the_12b_case_names_are_unique(self) -> None:
        names = [
            case.name
            for case in (*SHARED_ENCODE_CASES, *tokenizer.CASES, *tokenizer.EXTRA_CASES["12b"])
        ]

        assert sorted(names) == sorted(set(names))

    def test_the_12b_cases_run_into_the_merges_12b_dropped(self) -> None:
        """Each 12B case splits differently under E2B's merges — rebuilt from 12B's own
        `tokenizer.json` with the merge list of E2B's compiled asset (everything else in the two
        files is the same), so the cases measure the difference they are there for."""
        tokenizers = pytest.importorskip("tokenizers")
        upstream = sources.tokenizer_dir("12b") / "tokenizer.json"
        if not upstream.is_file() or not tokenizer.ASSET_PATH.is_file():
            pytest.skip(f"needs {upstream} and {tokenizer.ASSET_PATH}")
        raw = json.loads(upstream.read_text(encoding="utf-8"))
        e2b = json.loads(tokenizer.ASSET_PATH.read_text(encoding="utf-8"))
        vocab = e2b["vocab"]
        e2b_merges = [
            [vocab[int(left)], vocab[int(right)]]
            for left, right in (line.split(" ") for line in e2b["mergesText"].split("\n"))
        ]
        twelve_b = tokenizers.Tokenizer.from_str(json.dumps(raw))
        raw["model"]["merges"] = e2b_merges
        like_e2b = tokenizers.Tokenizer.from_str(json.dumps(raw))

        for case in tokenizer.EXTRA_CASES["12b"]:
            ours = twelve_b.encode(case.text, add_special_tokens=False).ids
            theirs = like_e2b.encode(case.text, add_special_tokens=False).ids
            assert ours != theirs, case.name
            assert len(ours) > len(theirs), case.name


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
