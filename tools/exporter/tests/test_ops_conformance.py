"""Conformance-table tests of the op contracts (Python side).

The source of truth is not the implementation but
`packages/runtime/tests/fixtures/op-contracts.json` (from the repository root), and the TS
implementation (packages/runtime/tests/ops_conformance_test.ts) checks the same things against **the
same table**. The table is read rather than copied because the moment a copy appears, "a contract
that passes on one side only" is born (the same discipline as test_dims.py).

Only the contract surfaces where the two implementations can silently diverge are checked: the full
set of op names / arity / slot dtypes / the attrs key sets (both required and **optional** — ADR
0067 decision 4) / attrs value ranges / the contract of the `states` field (key set and whether it
is required — ADR 0067 decisions 4 / 5) / the output count and the per-output-slot dtype mapping
(ADR 0068 decision 1) / the output shape rules (including the rank limit) / the slots eligible for
low-precision storage (ADR 0018).
"""

from __future__ import annotations

import pytest
from conftest import OP_CONTRACT_TABLE, OP_CONTRACT_TABLE_PATH

from karume.dims import eval_dim, parse_dim
from karume.ops import (
    OP_CONTRACTS,
    STRIDED_RANK,
    WEIGHT_CHANNEL_AXES,
    WEIGHT_SLOTS,
    OpContractError,
    PerSlotDtypes,
    UniformDtypes,
    resolve_op_contract,
)
from karume.shapes import compute_output_shape

OPS = OP_CONTRACT_TABLE["ops"]
ATTR_VALUES = OP_CONTRACT_TABLE["attr_values"]
SHAPES = OP_CONTRACT_TABLE["shapes"]

#: Every attr with a value range (required + optional). Being optional is no reason for a looser
#: value range, so the attr_values section looks up both on the same footing (the same split as the
#: TS side's ops_conformance_test.ts).
ATTR_KEYS = {entry["op"]: [*entry["attrs"], *entry.get("optional_attrs", [])] for entry in OPS}

#: Ops that **create** an output dimension **from attrs** instead of propagating it from the inputs.
#: Staying symbolic even after the bindings are substituted is correct (prefix length
#: coeff·sym+offset — ADR 0010), so only the substitution test below treats them differently. The
#: layer that applies bindings is the TS side (plan.ts); the exporter side has no bindings.
DIM_CREATING_OPS = {"sym_prefix_slice"}


def _shape_id(case) -> str:
    ins = " ".join(str(shape) for shape in case["ins"])
    return f"{case['op']}({ins}){'!' if case.get('throws') else ''}"


def _resolve(shape, bindings) -> list[int]:
    """Turns the dimension language into numbers with the bindings (to confirm the conclusion
    matches the symbolic computation)."""
    return [dim if isinstance(dim, int) else eval_dim(parse_dim(dim), bindings) for dim in shape]


def _shape_of(shape, bindings):
    """With bindings=None the shape stays symbolic; otherwise it is resolved to numbers first."""
    return shape if bindings is None else _resolve(shape, bindings)


def _compute(case, bindings=None):
    declared = case.get("declared")
    state_shapes = case.get("state_shapes")
    return compute_output_shape(
        resolve_op_contract(case["op"]),
        [_shape_of(shape, bindings) for shape in case["ins"]],
        "t",
        declared=None if declared is None else _shape_of(declared, bindings),
        attrs=case.get("attrs"),
        # states / state_shapes are passed as a pair (ADR 0067 decision 4) — a case where the
        # table writes states alone is exactly the rejection case "no resolved slot shape".
        states=case.get("states"),
        state_shapes=None
        if state_shapes is None
        else {slot: _shape_of(shape, bindings) for slot, shape in state_shapes.items()},
    )


def _attr_check(entry):
    """The check function for one row of the attr_values section (it may come from either the
    required or the optional field)."""
    contract = resolve_op_contract(entry["op"])
    check = contract.attrs.get(entry["attr"]) or contract.optional_attrs.get(entry["attr"])
    assert check is not None, (
        f"{entry['op']}: the contract table has no check function for attr '{entry['attr']}'"
    )
    return check


class TestFixtureTable:
    def test_the_shared_case_table_exists_and_has_all_sections(self):
        assert OP_CONTRACT_TABLE_PATH.exists(), (
            f"the conformance case table is missing: {OP_CONTRACT_TABLE_PATH}"
        )
        assert {"strided_rank_max", "ops", "attr_values", "shapes"} <= set(OP_CONTRACT_TABLE)

    def test_the_op_set_matches_the_contract_table(self):
        assert sorted(entry["op"] for entry in OPS) == sorted(OP_CONTRACTS)

    def test_no_op_is_declared_twice(self):
        # A duplicate would silently ignore one of the declarations and loosen the comparison.
        assert len({entry["op"] for entry in OPS}) == len(OPS)

    def test_the_strided_rank_limit_matches_the_implementation(self):
        assert OP_CONTRACT_TABLE["strided_rank_max"] == STRIDED_RANK

    def test_every_value_range_entry_has_both_an_accepted_and_a_rejected_case(self):
        """A value-range case has at least one accepted and one rejected value (an empty list goes
        green with zero checks).

        The two tests of {@link TestAttrValues} only loop over the lists with `for`, so an entry
        with `accept: []` / `reject: []` passes without checking anything. The existence check
        (`test_every_declared_attr_has_a_value_range_case`) looks only at **whether** an entry
        exists, so a change that effectively disables the value-range check of an attr would go
        undetected on both the Python and the TS side.
        """
        empty = [
            f"{entry['op']}.{entry['attr']}"
            for entry in ATTR_VALUES
            if not entry["accept"] or not entry["reject"]
        ]

        assert empty == []


class TestDeclaredContract:
    """The ops section: arity / slot dtypes / attrs key sets match the contract table."""

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_arity_matches(self, entry):
        assert resolve_op_contract(entry["op"]).arity == entry["arity"]

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_variadic_matches(self, entry):
        """Variadic arity is the contract surface "how many inputs are accepted" itself.

        Making it variadic on one side only splits the count the exporter can write from the count
        the runtime accepts.
        """
        assert resolve_op_contract(entry["op"]).variadic == entry.get("variadic", False)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_max_arity_matches(self, entry):
        """The limit on optional trailing inputs is a contract surface too (widening it on one side
        only splits IR with a mask)."""
        assert resolve_op_contract(entry["op"]).max_arity == entry.get("max_arity")

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_attr_keys_match(self, entry):
        assert sorted(resolve_op_contract(entry["op"]).attr_keys) == sorted(entry["attrs"])

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_optional_attr_keys_match(self, entry):
        """Optional attrs (ADR 0067's `window`) are a contract surface themselves.

        Moving one side only leads straight to "the runtime rejects a form the exporter can
        write". They are not merged with the required fields into one set because that would erase
        the distinction that only "fields whose absence is itself a different declaration" go here
        (no default filling — ADR 0012).
        """
        contract = resolve_op_contract(entry["op"])

        assert sorted(contract.optional_attr_keys) == sorted(entry.get("optional_attrs", []))

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_states_field_contract_matches(self, entry):
        """The contract of the `states` field (ADR 0067 decisions 4 / 5 — the exact key set +
        whether the field is required).

        The comparison includes which ops can carry the field and which cannot (`None` = cannot
        carry it).
        """
        contract = resolve_op_contract(entry["op"])
        declared = entry.get("states")
        if declared is None:
            assert contract.states is None, entry["op"]
            return

        assert contract.states is not None, entry["op"]
        assert sorted(contract.states.keys) == sorted(declared["keys"])
        assert contract.states.required == declared["required"]

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_output_dtypes_match(self, entry):
        """The list of per-output-slot dtype mappings (when omitted: one output, the identity over
        the accepted set of slot 0).

        If comparisons (f32 → bool), sum of bool (→ i32) and where (bool → f32) are not aligned on
        both sides, "the exporter can write it but the runtime reads it as a different TypedArray"
        appears. The list length = the output count the contract declares (ADR 0068 decision 1) is
        compared with the same table. Only cast lacks this field, since its output is decided by
        attrs.to.
        """
        contract = resolve_op_contract(entry["op"])
        if entry["op"] == "cast":
            assert "out_dtypes" not in entry
            return
        slots = contract.slot_dtypes
        domain = slots.accept if isinstance(slots, UniformDtypes) else slots.slots[0]
        declared = entry.get("out_dtypes", [{}])

        assert contract.output_dtypes == tuple(
            {dtype: slot.get(dtype, dtype) for dtype in domain} for slot in declared
        )
        assert contract.output_count == len(declared)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_weight_slot_matches(self, entry):
        """The slots eligible for low-precision storage (ADR 0018).

        A split leads either to "the runtime deems a weight the exporter wrote in f16 ineligible
        and decodes it on the CPU" (the VRAM saving silently disappears) or the reverse, "rounding
        that does not match the f32 golden". Neither raises, so they are compared from the table
        side.
        """
        assert WEIGHT_SLOTS.get(entry["op"]) == entry.get("weight_slot")

    def test_the_weight_slot_table_has_no_extra_entries(self):
        # Also fails on an eligible slot that exists only in the implementation (= a demotion path
        # missing from the table).
        assert len([entry for entry in OPS if "weight_slot" in entry]) == len(WEIGHT_SLOTS)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_channel_axis_matches(self, entry):
        """The axis the i8 per-channel scale applies to (ADR 0019).

        A split gives "the kernel reads a scale the exporter built on axis 0 as axis 1", where
        **nothing raises and only the values break** (conv_transpose1d's [Cin,Cout,K] even passes
        the shape check when Cin == Cout). The same table also checks that it appears exactly
        together with the weight slot.
        """
        assert WEIGHT_CHANNEL_AXES.get(entry["op"]) == entry.get("channel_axis")
        assert ("channel_axis" in entry) == ("weight_slot" in entry), entry["op"]

    def test_the_channel_axis_table_has_no_extra_entries(self):
        assert len([entry for entry in OPS if "channel_axis" in entry]) == len(WEIGHT_CHANNEL_AXES)

    @pytest.mark.parametrize("entry", OPS, ids=lambda e: e["op"])
    def test_slot_dtypes_match(self, entry):
        slots = resolve_op_contract(entry["op"]).slot_dtypes
        declared = entry["dtypes"]
        if isinstance(slots, UniformDtypes):
            assert declared["kind"] == "uniform", entry["op"]
            assert sorted(slots.accept) == sorted(declared["accept"])
            return
        assert isinstance(slots, PerSlotDtypes)
        assert declared["kind"] == "perSlot", entry["op"]
        assert [sorted(accept) for accept in slots.slots] == [
            sorted(accept) for accept in declared["slots"]
        ]


class TestAttrValues:
    """The attr_values section: value ranges can only be expressed by check functions, so the
    verdicts are compared by passing actual values through."""

    @pytest.mark.parametrize("entry", ATTR_VALUES, ids=lambda e: f"{e['op']}.{e['attr']}")
    def test_the_attr_is_declared_in_the_ops_section(self, entry):
        assert entry["attr"] in ATTR_KEYS[entry["op"]]

    @pytest.mark.parametrize("entry", ATTR_VALUES, ids=lambda e: f"{e['op']}.{e['attr']}")
    def test_accepted_values_pass_the_schema(self, entry):
        check = _attr_check(entry)
        for value in entry["accept"]:
            check(value, f"{entry['op']}.{entry['attr']}")

    @pytest.mark.parametrize("entry", ATTR_VALUES, ids=lambda e: f"{e['op']}.{e['attr']}")
    def test_rejected_values_fail_loudly(self, entry):
        check = _attr_check(entry)
        for value in entry["reject"]:
            with pytest.raises(OpContractError):
                check(value, f"{entry['op']}.{entry['attr']}")

    def test_every_declared_attr_has_a_value_range_case(self):
        """No attr is left without a value range in the table (a missed entry leaves the range
        unchecked)."""
        covered = {f"{entry['op']}.{entry['attr']}" for entry in ATTR_VALUES}

        missing = [
            f"{op}.{attr}"
            for op, attrs in ATTR_KEYS.items()
            for attr in attrs
            if f"{op}.{attr}" not in covered
        ]

        assert missing == []


class TestOutputShapes:
    """The shapes section: the outputs computed from the declared shapes (symbols included) match
    the table."""

    @pytest.mark.parametrize("case", SHAPES, ids=_shape_id)
    def test_shapes_follow_the_table(self, case):
        if case.get("throws"):
            with pytest.raises(OpContractError):
                _compute(case)
            return

        # The table's outs is a per-output-slot list (ADR 0068 decision 1). compute_output_shape
        # returns a list too, so the lists are compared (this fails as soon as the counts differ).
        assert _compute(case) == case["outs"]

    @pytest.mark.parametrize("case", SHAPES, ids=_shape_id)
    def test_the_numeric_reading_agrees_with_the_symbolic_one(self, case):
        """Numeric shapes with the bindings applied reach the same conclusion (the TS side runs the
        same table over this domain).

        If the symbolic computation and the computation after applying the bindings disagree,
        "passes at export but fails on run-time shapes" comes back — closure under substitution
        is pinned here.
        """
        bindings = case.get("bindings", {})
        if case.get("throws"):
            with pytest.raises(OpContractError):
                _compute(case, bindings)
            return
        if case["op"] in DIM_CREATING_OPS:
            # Pins that it is **not** closed under substitution (even with numeric inputs, a
            # symbolic dimension from attrs comes out = the binding layer is not here). Passing it
            # through would loosen the split between the layers.
            assert _compute(case, bindings) == _compute(case)
            return

        assert _compute(case, bindings) == [_resolve(shape, bindings) for shape in case["outs"]]

    def test_every_op_is_exercised_by_an_accepting_case(self):
        """No op is left whose shape rule the table does not exercise."""
        accepted = {case["op"] for case in SHAPES if not case.get("throws")}

        assert accepted == {entry["op"] for entry in OPS}
