"""`deberta/calib_texts.py` の校正コーパスの性質。

文面そのものは判断の対象にしない（品質は実測が答える）。ここで固定するのは、崩れると
**測定値の意味が変わる**側だけ:

- 48 文ちょうどで重複が無い（`--calib-limit` の縮小実行が「同じ文を N 回」にならない）
- 長さが散っている（GPTQ の `H = Σ XᵀX` はトークン数で重み付くので、長さが揃うと
  その長さの活性だけが支配する）
- 先頭を切っても役割の混合が保たれる（縮小 smoke の入力が朗読調だけにならない）
- 台本自身が持つ golden 文（{@link deberta.export.GOLDEN_SENTENCES}）と 1 文も重ならない

最後の 1 本だけは**静的に確かめられる**（golden 文は台本の定数で、資産を読まない）。実行時の
評価文との分離は別の話で、そちらは**資産（dump）が正本**なので、ここではなく実行時の
{@link sbv2.measure_quant.assert_calib_disjoint} が見る（写しをコーパス側へ置くと dump を
録り直したときに片方だけ古くなる）。その門自体の検出力は `test_measure_quant.py` が試す。
"""

from __future__ import annotations

from deberta import export as export_deberta
from deberta.calib_texts import CALIB_TEXTS

#: 数字読みの席の判定に使う漢数字（トークナイザは通さない — 文字の有無だけで見る）。
NUMERALS = "〇一二三四五六七八九十百千万"


def is_question(text: str) -> bool:
    """問いかけ（会話調）の文か。"""
    return text.endswith("？")


def has_numeral(text: str) -> bool:
    """数字読みを含む文か。"""
    return any(char in NUMERALS for char in text)


class TestCorpus:
    def test_it_holds_forty_eight_distinct_texts(self):
        assert len(CALIB_TEXTS) == 48
        assert len(set(CALIB_TEXTS)) == 48

    def test_no_text_is_a_golden_sentence(self):
        """校正文が golden 文と重なると、丸め先を「測る文そのもの」から選ぶことになる。"""
        assert not set(export_deberta.GOLDEN_SENTENCES) & set(CALIB_TEXTS)

    def test_no_text_quotes_a_golden_sentence(self):
        """部分一致でも重なりは重なり（`in` で片方向ずつ見る）。"""
        for golden in export_deberta.GOLDEN_SENTENCES:
            assert not any(golden in text or text in golden for text in CALIB_TEXTS)

    def test_the_lengths_are_spread(self):
        """短文と長文の両方が要る（char-wwm なので文字数がほぼトークン数）。"""
        lengths = [len(text) for text in CALIB_TEXTS]

        assert min(lengths) <= 12
        assert max(lengths) >= 40

    def test_every_role_appears_in_the_first_few_texts(self):
        """縮小実行（`--calib-limit 4`）でも朗読調 / 問いかけ / 数字読みが揃う並び。"""
        head = CALIB_TEXTS[:4]

        assert any(is_question(text) for text in head)
        assert any(has_numeral(text) for text in head)
        assert any(not is_question(text) and not has_numeral(text) for text in head)

    def test_both_roles_are_well_represented(self):
        """どちらかへ寄ると、活性の偏りがその役割の語彙分布に張り付く。"""
        questions = sum(1 for text in CALIB_TEXTS if is_question(text))
        numerals = sum(1 for text in CALIB_TEXTS if has_numeral(text))

        assert 8 <= questions <= len(CALIB_TEXTS) - 8
        assert 8 <= numerals <= len(CALIB_TEXTS) - 8
