"""Contract of the rounding up (`ceil_significant`) that sets the collapse ceiling values.

The table of ceilings (`COLLAPSE_CEILINGS` in `packages/models/tests/helpers/collapse-ceilings.ts`)
holds "worst × 2 rounded up to two significant digits", so if this is off by one step, every row of
the table silently loosens / tightens.
"""

from __future__ import annotations

import pytest

from deberta.collapse_ceilings import ceil_significant


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0198, 0.020),  # a remainder goes up (output.2's worst 0.0099 × 2)
        (0.3240, 0.33),  # a remainder past the 2nd digit goes up too (output.24's worst 0.1620 × 2)
        (0.0995, 0.1),  # carries over and gains a digit
    ],
)
def test_rounds_up_to_two_significant_digits(value: float, expected: float) -> None:
    assert ceil_significant(value) == expected


@pytest.mark.parametrize("value", [0.02, 0.14, 0.28, 0.051])
def test_keeps_a_value_already_on_two_digits(value: float) -> None:
    # A binary fraction does not push it one step up (0.14 / 0.01 is 14.000…02 in f64, and a naive
    # round-up returns 0.15) — the reason to round in decimal. The table's values for output.12 / 22
    # have this form.
    assert ceil_significant(value) == value


def test_rejects_zero() -> None:
    # A ceiling of 0 means "any difference is red" and changes what the check means — a sample
    # whose worst is 0 breaks an assumption of the derivation.
    with pytest.raises(ValueError):
        ceil_significant(0.0)
