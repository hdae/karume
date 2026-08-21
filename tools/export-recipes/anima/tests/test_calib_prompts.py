"""`anima/calib_prompts.py` の校正コーパスと**評価入力からの分離**。

文面そのものは判断の対象にしない（品質は実測が答える）。ここで固定するのは、崩れると
測定・裁定の**意味が変わる**側だけ:

- 重複が無く、既定本数がコーパスに収まる（縮小実行が「同じプロンプトを N 回」にならない）
- 先頭を切っても被写体 / 画風の混合が保たれる（`--calib-prompts N` は先頭 N 本）
- 評価入力 3 箇所（`pipeline_ref` / Deno E2E / eval-images）から**部分一致でも**分離している
- 分離検査そのものが**素通りしない**（フォールト注入で落ちること）
- 評価入力の抽出が空になったら fail loudly（黙って分離検査が緩む方が危ない）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anima import calib_prompts
from anima.calib_prompts import (
    CALIB_PROMPTS,
    DEFAULT_CALIB_PROMPTS,
    assert_calib_disjoint,
    calibration_prompts,
    evaluation_prompts,
)
from anima.pipeline_ref import PROMPT

#: 画風を名指しするタグ（先頭群に画風が混ざっていることの判定に使う）。
STYLE_TAGS = ("chibi", "monochrome", "watercolor", "film grain", "sketch")


class TestCorpus:
    def test_it_holds_distinct_prompts(self):
        assert len(set(CALIB_PROMPTS)) == len(CALIB_PROMPTS)

    def test_the_default_count_fits_the_corpus(self):
        """既定が上限を超えると `--calib-prompts` を触らない実行がそのまま落ちる。"""
        assert 1 <= DEFAULT_CALIB_PROMPTS <= len(CALIB_PROMPTS)

    def test_the_default_selection_keeps_the_mix(self):
        """先頭 N 本しか使わないので、既定の本数だけで被写体・画風が散っていること。"""
        head = CALIB_PROMPTS[:DEFAULT_CALIB_PROMPTS]

        assert any("no humans" in prompt for prompt in head), "人物なしが 1 本も無い"
        assert any("1girl" in prompt or "1boy" in prompt for prompt in head), "単独人物が無い"
        assert any(prompt.startswith(("2girls", "3girls", "4girls")) for prompt in head), (
            "複数人物が無い"
        )
        assert any(tag in prompt for prompt in head for tag in STYLE_TAGS), (
            "画風を名指しするプロンプトが無い"
        )

    def test_an_out_of_range_count_is_refused(self):
        with pytest.raises(ValueError, match="範囲外"):
            calibration_prompts(0)
        with pytest.raises(ValueError, match="範囲外"):
            calibration_prompts(len(CALIB_PROMPTS) + 1)


class TestEvaluationSources:
    def test_it_gathers_all_three_sources(self):
        """`pipeline_ref` の固定 2 本 + Deno E2E の 1 本 + eval-images の 4 ケース。"""
        evaluated = evaluation_prompts()

        assert PROMPT in evaluated
        assert len(evaluated) >= 7
        assert any(prompt.startswith("score_9") for prompt in evaluated), "eval-images の合成が無い"

    def test_concatenated_string_literals_are_joined(self):
        """`"…" + "…"` の連結を 1 本に戻す（断片だけ採ると分離検査が半分しか効かない）。"""
        joined = calib_prompts._declaration('const X = "ab" +\n  "cd";\n', Path("x.ts"), "X")

        assert joined == "abcd"

    def test_a_missing_source_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(calib_prompts, "E2E_SOURCE", Path("packages/models/tests/gone.ts"))

        with pytest.raises(AssertionError, match="評価入力の正本"):
            evaluation_prompts()

    def test_a_renamed_declaration_fails_loudly(self, monkeypatch):
        """綴りが変わって抽出が空になったら落とす（黙って分離検査が緩む方が危ない）。"""
        monkeypatch.setattr(calib_prompts, "E2E_SOURCE", Path("README.md"))

        with pytest.raises(AssertionError, match="const PROMPT"):
            evaluation_prompts()


class TestDisjointness:
    def test_the_shipped_corpus_is_disjoint_from_every_evaluation_input(self):
        """既定選択も全量も、評価入力と 1 本も重ならない。"""
        evaluated = evaluation_prompts()

        assert_calib_disjoint(calibration_prompts(DEFAULT_CALIB_PROMPTS), evaluated)
        assert_calib_disjoint(CALIB_PROMPTS, evaluated)

    def test_an_identical_prompt_is_caught(self):
        with pytest.raises(AssertionError, match="評価入力と重なっている"):
            assert_calib_disjoint((PROMPT,), evaluation_prompts())

    def test_a_prompt_contained_in_an_evaluation_input_is_caught(self):
        """部分一致まで見る — 評価プロンプトの前半だけを校正に使うのも漏れ。"""
        with pytest.raises(AssertionError, match="評価入力と重なっている"):
            assert_calib_disjoint((PROMPT[: len(PROMPT) // 2],), evaluation_prompts())

    def test_a_prompt_that_contains_an_evaluation_input_is_caught(self):
        with pytest.raises(AssertionError, match="評価入力と重なっている"):
            assert_calib_disjoint((f"{PROMPT}, extra tag",), evaluation_prompts())

    def test_the_selection_path_runs_the_check(self, monkeypatch):
        """MUST: `--calib-prompts` の経路が検査を通ること（定数だけ緑でも意味が無い）。"""
        monkeypatch.setattr(calib_prompts, "CALIB_PROMPTS", (PROMPT,))

        with pytest.raises(AssertionError, match="評価入力と重なっている"):
            calibration_prompts(1)
