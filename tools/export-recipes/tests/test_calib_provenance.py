"""The check of the calibration conditions of an i4 series (`_shared/calib_provenance.py`) — the
rule its 3 readers share.

Only what **cannot be read from the asset once broken** is pinned here:

- a method that is not allowed for distribution fails before any other field is looked at (the
  diagnostic is not about the budget)
- budgets are checked against a **lower bound** and model conditions by **equality** (for the
  latter "bigger is better" does not hold)
- **a missing field is accepted** (backward compatibility MUST — adding a field does not force a
  re-export of existing series)
- a field that is present but cannot be read as a number fails loudly (it is never silently
  accepted)

The family side (which field is checked in which mode) lives in
`irodori/tests/test_distribution.py` / `irodori/tests/test_pipeline_ref.py` /
`anima/tests/test_distribution.py`.
"""

from __future__ import annotations

from _shared.calib_provenance import calib_complaint

#: The same shape as the record of the shipped irodori w4 series
#: (`outputs/series/irodori-v4-small-i4/`).
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
        """A broken record is never accepted as "just missing fields"."""
        assert calib_complaint([1, 2, 3], method="gptq") is not None

    def test_the_method_is_judged_before_the_budget(self) -> None:
        """`--no-calib` writes the budget fields as 0 — a diagnostic about "budget too small" would
        hide the cause."""
        record = {**SHIPPED, "method": "rtn", "cases": 0, "steps": 0}

        complaint = calib_complaint(record, method="gptq", at_least=FLOOR)

        assert complaint is not None
        assert "rounding method" in complaint


class TestBudgetFloor:
    def test_a_budget_below_the_floor_is_named_with_the_field(self) -> None:
        complaint = calib_complaint({**SHIPPED, "steps": 1}, method="gptq", at_least=FLOOR)

        assert complaint is not None
        assert "'steps'" in complaint
        assert "1 < 40" in complaint

    def test_a_budget_above_the_floor_passes(self) -> None:
        """The budget is only checked for "not cut" (a larger one only raises the quality)."""
        assert calib_complaint({**SHIPPED, "steps": 80}, method="gptq", at_least=FLOOR) is None

    def test_a_field_the_record_does_not_carry_is_skipped(self) -> None:
        """Backward compatibility MUST: adding a field does not force a re-export (= the rounding
        time) on existing series."""
        legacy = {"method": "gptq", "grid": "rtn", "group_size": 32}

        assert calib_complaint(legacy, method="gptq", at_least=FLOOR) is None

    def test_a_field_that_is_not_a_number_fails_loudly(self) -> None:
        assert (
            calib_complaint({**SHIPPED, "steps": "40"}, method="gptq", at_least=FLOOR) is not None
        )

    def test_a_boolean_does_not_slip_through_as_one(self) -> None:
        """`True` turns into 1 as a number — it signals a broken record, so it fails."""
        complaint = calib_complaint(
            {**SHIPPED, "steps": True}, method="gptq", at_least={"steps": 1}
        )

        assert complaint is not None


class TestExactConditions:
    """Per-model conditions (step / CFG) are checked by equality — an axis where "bigger is
    better" does not hold."""

    def test_the_conditions_of_the_model_pass(self) -> None:
        record = {"method": "gptq", "steps": 8, "guidance": 1.0}

        assert calib_complaint(record, method="gptq", exactly={"steps": 8, "guidance": 1.0}) is None

    def test_a_larger_budget_from_another_model_is_still_refused(self) -> None:
        """Putting an asset built under the base model's conditions (20 steps, CFG 4) into the
        turbo slot is wrong too."""
        record = {"method": "gptq", "steps": 20, "guidance": 4.0}

        complaint = calib_complaint(record, method="gptq", exactly={"steps": 8, "guidance": 1.0})

        assert complaint is not None
        assert "'steps'" in complaint

    def test_a_condition_the_record_does_not_carry_is_skipped(self) -> None:
        """`guidance` is a field added on 2026-08-23 — series from before that do not have it."""
        record = {"method": "gptq", "steps": 8}

        assert calib_complaint(record, method="gptq", exactly={"steps": 8, "guidance": 1.0}) is None
