"""i4 系列の校正条件の判定（`_shared/calib_provenance.py`）— 3 つの読み手が共有する規律。

固定するのは、**壊れても資産からは読めない**側だけ:

- 方式が配布可でなければ他の欄を見るまでもなく落ちること（診断の主語を予算にしない）
- 予算は**下限**で、モデル条件は**一致**で見ること（後者は「大きい方が良い」が成り立たない）
- **欄の不在は受理する**こと（後方互換 MUST — 欄を足しても既存系列に再 export を課さない）
- 欄が在るのに数として読めない場合は fail loudly すること（黙って受理する側に倒さない）

家族側の席（どの欄をどちらのモードで見るか）は `irodori/tests/test_distribution.py` /
`irodori/tests/test_pipeline_ref.py` / `anima/tests/test_distribution.py` が持つ。
"""

from __future__ import annotations

from _shared.calib_provenance import calib_complaint

#: 出荷済みの irodori w4 系列の記録と同じ形（`outputs/series/irodori-v4-small-i4/`）。
SHIPPED = {"method": "gptq", "grid": "rtn", "group_size": 32, "cases": 12, "steps": 40}

FLOOR = {"cases": 12, "steps": 40}


class TestMethod:
    def test_a_shippable_record_has_nothing_to_complain_about(self) -> None:
        assert calib_complaint(SHIPPED, method="gptq", at_least=FLOOR) is None

    def test_another_method_is_named_with_both_sides(self) -> None:
        complaint = calib_complaint({**SHIPPED, "method": "rtn"}, method="gptq", at_least=FLOOR)

        assert complaint is not None
        assert "'rtn'" in complaint
        assert "'gptq'" in complaint

    def test_a_record_that_is_not_a_mapping_falls_out_as_a_method_mismatch(self) -> None:
        """壊れた記録を「欄が無いだけ」として受理する側には倒さない。"""
        assert calib_complaint([1, 2, 3], method="gptq") is not None

    def test_the_method_is_judged_before_the_budget(self) -> None:
        """`--no-calib` は予算欄を 0 で書く — 主語が「予算不足」になると原因が読めない。"""
        record = {**SHIPPED, "method": "rtn", "cases": 0, "steps": 0}

        complaint = calib_complaint(record, method="gptq", at_least=FLOOR)

        assert complaint is not None
        assert "丸め方式" in complaint


class TestBudgetFloor:
    def test_a_budget_below_the_floor_is_named_with_the_field(self) -> None:
        complaint = calib_complaint({**SHIPPED, "steps": 1}, method="gptq", at_least=FLOOR)

        assert complaint is not None
        assert "'steps'" in complaint
        assert "1 < 40" in complaint

    def test_a_budget_above_the_floor_passes(self) -> None:
        """予算は「削られていないこと」だけを見る（増やした側は質が上がるだけ）。"""
        assert calib_complaint({**SHIPPED, "steps": 80}, method="gptq", at_least=FLOOR) is None

    def test_a_field_the_record_does_not_carry_is_skipped(self) -> None:
        """後方互換 MUST: 欄の追加が既存系列へ再 export（= 丸め時間）を課さない。"""
        legacy = {"method": "gptq", "grid": "rtn", "group_size": 32}

        assert calib_complaint(legacy, method="gptq", at_least=FLOOR) is None

    def test_a_field_that_is_not_a_number_fails_loudly(self) -> None:
        assert (
            calib_complaint({**SHIPPED, "steps": "40"}, method="gptq", at_least=FLOOR) is not None
        )

    def test_a_boolean_does_not_slip_through_as_one(self) -> None:
        """`True` は数として 1 に化ける — 記録が壊れている合図なので落とす。"""
        complaint = calib_complaint(
            {**SHIPPED, "steps": True}, method="gptq", at_least={"steps": 1}
        )

        assert complaint is not None


class TestExactConditions:
    """モデル別の条件（step / CFG）は一致で見る — 大きい方が良いが成り立たない軸。"""

    def test_the_conditions_of_the_model_pass(self) -> None:
        record = {"method": "gptq", "steps": 8, "guidance": 1.0}

        assert calib_complaint(record, method="gptq", exactly={"steps": 8, "guidance": 1.0}) is None

    def test_a_larger_budget_from_another_model_is_still_refused(self) -> None:
        """素版（20 step・CFG 4）の条件で焼いた資産を turbo の席に載せるのも誤り。"""
        record = {"method": "gptq", "steps": 20, "guidance": 4.0}

        complaint = calib_complaint(record, method="gptq", exactly={"steps": 8, "guidance": 1.0})

        assert complaint is not None
        assert "'steps'" in complaint

    def test_a_condition_the_record_does_not_carry_is_skipped(self) -> None:
        """`guidance` は 2026-08-23 に足した欄 — それ以前の系列は欄を持たない。"""
        record = {"method": "gptq", "steps": 8}

        assert calib_complaint(record, method="gptq", exactly={"steps": 8, "guidance": 1.0}) is None
