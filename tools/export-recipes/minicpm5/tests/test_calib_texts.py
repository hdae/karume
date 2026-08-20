"""`minicpm5/calib_texts.py` の校正コーパスの性質。

文面そのものは判断の対象にしない（品質は実測が答える）。ここで固定するのは、崩れると
**測定値の意味が変わる**側だけ:

- **評価文と 1 文も重ならない** — 重なると teacher / greedy の一致が「校正で見た文を
  そのまま出せたか」になり、校正の質ではなく漏れを測る数になる
- 48 文ちょうどで重複が無い（`--calib-limit` の縮小実行が「同じ文を N 回」にならない）
- 先頭を切っても言語とスタイルの混合が保たれる（縮小 smoke の入力が英語だけにならない）
"""

from __future__ import annotations

from minicpm5 import export as one_shot
from minicpm5.calib_texts import CALIB_TEXTS


def has_japanese(text: str) -> bool:
    """かな・漢字を 1 文字でも含むか（言語比の判定 — トークナイザは通さない）。"""
    return any("぀" <= char <= "ヿ" or "一" <= char <= "鿿" for char in text)


class TestCorpus:
    def test_it_holds_forty_eight_distinct_texts(self):
        assert len(CALIB_TEXTS) == 48
        assert len(set(CALIB_TEXTS)) == 48

    def test_no_text_overlaps_the_golden_cases(self):
        """MUST: 校正と評価の分離（モジュール docstring）。"""
        golden = {body for _name, body in one_shot.GOLDEN_CASES}

        assert not golden & set(CALIB_TEXTS)

    def test_no_text_quotes_a_golden_case(self):
        """部分一致でも重なりは重なり（`in` で片方向ずつ見る）。"""
        for _name, body in one_shot.GOLDEN_CASES:
            assert not any(body in text or text in body for text in CALIB_TEXTS)

    def test_both_languages_appear_in_the_first_few_texts(self):
        """縮小実行（`--calib-limit 4`）でも英日が揃う並び。"""
        head = CALIB_TEXTS[:4]

        assert any(has_japanese(text) for text in head)
        assert any(not has_japanese(text) for text in head)

    def test_the_two_languages_are_both_well_represented(self):
        """英日どちらかへ寄ると、活性の偏りがその言語のトークン分布に張り付く。"""
        japanese = sum(1 for text in CALIB_TEXTS if has_japanese(text))

        assert 10 <= japanese <= len(CALIB_TEXTS) - 10
