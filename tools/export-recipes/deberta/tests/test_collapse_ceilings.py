"""崩壊上限の値を決める切り上げ（`ceil_significant`）の約束事。

上限の表（`e2e_deberta_w8a8_test.ts` の `COLLAPSE_CEILINGS`）は「最悪 × 2 を有効数字 2 桁で
切り上げた値」なので、ここが 1 段ずれると表の全行が黙って緩む / 締まる。
"""

from __future__ import annotations

import pytest

from deberta.collapse_ceilings import ceil_significant


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0198, 0.020),  # 端数は上へ（output.2 の最悪 0.0099 × 2）
        (0.3240, 0.33),  # 2 桁目の端数も上へ（output.24 の最悪 0.1620 × 2）
        (0.0995, 0.1),  # 繰り上がって桁が増える
    ],
)
def test_rounds_up_to_two_significant_digits(value: float, expected: float) -> None:
    assert ceil_significant(value) == expected


@pytest.mark.parametrize("value", [0.02, 0.14, 0.28, 0.051])
def test_keeps_a_value_already_on_two_digits(value: float) -> None:
    # 2 進の端数で 1 段上へずれない（0.14 / 0.01 は f64 で 14.000…02 になり、素朴な切り上げは
    # 0.15 を返す）— 10 進で丸める理由。表の output.12 / 22 の値がこの形。
    assert ceil_significant(value) == value


def test_rejects_zero() -> None:
    # 上限 0 は「どんな差も赤」で門の意味が変わる — 最悪が 0 の標本は導出の前提の破れ。
    with pytest.raises(ValueError):
        ceil_significant(0.0)
