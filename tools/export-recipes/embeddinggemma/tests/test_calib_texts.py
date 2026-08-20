"""`embeddinggemma/calib_texts.py` の校正コーパスの性質。

文面そのものは判断の対象にしない（品質は実測が答える）。ここで固定するのは、崩れると
**測定値の意味が変わる**側だけ:

- **評価文と 1 文も重ならない** — 重なると cosine が「校正で見た文をそのまま埋め込めたか」
  になり、校正の質ではなく漏れを測る数になる
- 48 文ちょうどで重複が無い（`--calib-limit` の縮小実行が「同じ文を N 回」にならない）
- プロンプト種別が `export.load_prompts` のキーか `None` のどちらか（綴りが外れると
  校正入力の先頭トークン列だけが実運用と食い違う）
- 先頭を切っても役割（query / document / 素）の混合が保たれる
"""

from __future__ import annotations

from embeddinggemma import export as eg
from embeddinggemma.calib_texts import CALIB_TEXTS

#: `config_sentence_transformers.json` の `prompts` が持つキー（実資産を読まずに固定する
#: 綴り — 実際の引きは `calib_inputs` が辞書アクセスで落とす）。
PROMPT_KEYS = {"query", "document"}


class TestCorpus:
    def test_it_holds_forty_eight_distinct_texts(self):
        assert len(CALIB_TEXTS) == 48
        assert len({body for _prompt, body in CALIB_TEXTS}) == 48

    def test_no_text_overlaps_the_golden_cases(self):
        """MUST: 校正と評価の分離（モジュール docstring）。"""
        golden = {body for _name, _prompt, body in eg.GOLDEN_CASES}

        assert not golden & {body for _prompt, body in CALIB_TEXTS}

    def test_no_text_quotes_a_golden_case(self):
        """部分一致でも重なりは重なり（`in` で片方向ずつ見る）。"""
        bodies = [body for _prompt, body in CALIB_TEXTS]
        for _name, _prompt, golden in eg.GOLDEN_CASES:
            assert not any(golden in body or body in golden for body in bodies)

    def test_every_prompt_key_is_spelled_like_the_official_ones(self):
        assert {prompt for prompt, _body in CALIB_TEXTS} <= PROMPT_KEYS | {None}

    def test_all_three_roles_appear_in_the_first_few_texts(self):
        """縮小実行（`--calib-limit 3`）でも query / document / 素が揃う並び。"""
        head = {prompt for prompt, _body in CALIB_TEXTS[:3]}

        assert head == PROMPT_KEYS | {None}

    def test_the_two_query_roles_are_balanced(self):
        """検索は query と document の**非対称**な使い方 — 片方だけの活性で丸めない。"""
        prompts = [prompt for prompt, _body in CALIB_TEXTS]

        assert prompts.count("query") == prompts.count("document") == 20
