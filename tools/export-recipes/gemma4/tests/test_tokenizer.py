"""トークナイザ資産の台本（`gemma4/tokenizer.py`）の台本レベルの約束事（実資産不要分）。

実資産（32MB の `tokenizer.json`）はローカルにしか無いので、ここが見るのは**定数と path
だけ**で成立する規律（transformers も torch も要らない）。`chat.py` の同役は
{@link gemma4.tests.test_chat} の `TestScopeGate`。

固定するのは、壊れると**偽 PASS** になる側だけ:

- 名前が共有電池（{@link _shared.gemma_tokenizer.SHARED_ENCODE_CASES}）と合わせて一意で
  あること。`emit` は 2 つの列を連結するだけで一意性を見ないので、名前が衝突するとパリティ
  フィクスチャに同名 2 件が並ぶ。TS 側は名前で 1 件を引く場所があり
  （`packages/models/tests/gemma_tokenizer_test.ts` の `fixture.encode.find(…)`）、`find` は
  黙って先頭を返す — **別のケースを測って緑になる**（落ちて読めなくなるのではない）
- 出力先の綴りが配布 recipe の系列名と一致すること（書き手と読み手が同じ 1 語から組む —
  `gemma4/distribution.py` の宣言）
- パリティフィクスチャの出力先が git 管理の実在ファイルであること（消えると TS 側が読む先を
  失う）
"""

from __future__ import annotations

from _shared.gemma_tokenizer import SHARED_ENCODE_CASES
from gemma4 import tokenizer
from gemma4.distribution import (
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_TOKENIZER_FILE,
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
        """フィクスチャは git 管理（TS 側 `gemma_tokenizer_test.ts` が読む現物）。"""
        assert tokenizer.FIXTURE_PATH.is_file()
