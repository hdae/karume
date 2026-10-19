"""Per-token i8 activation fake-quant (`karume.act_quant`) — the numeric spec mirroring the runtime.

The source of truth is `packages/runtime/src/kernels/quantize-rows.ts` and
`packages/runtime/src/reference/i8a8.ts`. What this file pins is "the torch side has the same
grid, the same rounding and the same NaN propagation as that spec"; the numbers of the E2E
reference fixtures (`anima/pipeline_ref.py --act-quant`) depend on it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from karume.act_quant import (
    PACK_ALIGN,
    attach_act_quant,
    detach_act_quant,
    is_eligible,
    quantize_rows,
)
from karume.quantize import QuantizeError


class KeywordOnlyCaller(nn.Module):
    """A model that calls an eligible linear with keyword arguments only (the pre-hook's `args` is
    empty)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(input=x)


def test_quantize_rows_keeps_shape_and_dtype() -> None:
    x = torch.randn(3, 5, 8, dtype=torch.float32)
    out = quantize_rows(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_zero_row_is_exactly_zero() -> None:
    """An all-zero row is exact via `s = tiny` → `q = 0` → `0 · tiny = 0` (what the lower clamp is
    for)."""
    x = torch.zeros(2, 6, dtype=torch.float32)
    assert torch.equal(quantize_rows(x), x)


def test_grid_is_closed_at_plus_minus_127() -> None:
    """MUST: **−128 is not used** (the element with the largest magnitude lands on ±127·s).

    An implementation that uses −128 maps the largest negative element to `-128·s`, giving
    `|x̂| > |x|`.
    """
    x = torch.tensor([[-4.0, 1.0, 2.5, 0.5]], dtype=torch.float32)
    out = quantize_rows(x)
    scale = 4.0 / 127.0
    assert out[0, 0].item() == -127 * scale
    # The restored values do not exceed the original largest magnitude (using −128 would)
    assert out.abs().amax().item() <= x.abs().amax().item()


def test_round_is_ties_to_even() -> None:
    """MUST: ties go to even (matching WGSL's `round`). A half-up implementation gives 0.5→1 /
    2.5→3."""
    # amax = 127, so s = 127/127 = exactly 1.0 → x̂ is round(x) itself
    x = torch.tensor([[127.0, 0.5, 1.5, -0.5, -1.5, 2.5, 3.5, -3.5]], dtype=torch.float32)
    out = quantize_rows(x)
    assert out.tolist() == [[127.0, 0.0, 2.0, -0.0, -2.0, 2.0, 4.0, -4.0]]


def test_scale_is_per_row_and_independent() -> None:
    """The scale is set per row (an outlier in one row does not crush the grid of the other
    rows)."""
    x = torch.tensor([[1.0, -1.0, 0.5, 0.25], [1000.0, 1.0, 0.5, 0.25]], dtype=torch.float32)
    out = quantize_rows(x)
    # Row 0 has s = 1/127, so 0.25 is 32 steps (the value is kept; only f32 rounding is tolerated)
    assert abs(out[0, 3].item() - 32 / 127) < 1e-7
    # Row 1 has s = 1000/127, so 0.25 collapses to 0 steps (evidence that rows are independent)
    assert out[1, 3].item() == 0.0
    assert abs(out[1, 0].item() - 1000.0) < 1e-3


def test_nan_propagates_to_the_whole_row() -> None:
    """One NaN in a row turns the whole row NaN (propagation through the scale — the same
    granularity as ADR 0020)."""
    x = torch.tensor([[1.0, 2.0, float("nan"), 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    out = quantize_rows(x)
    assert bool(out[0].isnan().all())
    assert bool(out[1].isfinite().all())


def test_is_eligible_requires_linear_and_pack_alignment() -> None:
    assert PACK_ALIGN == 4
    assert is_eligible(nn.Linear(8, 3))
    assert is_eligible(nn.Linear(8, 3, bias=False)), "bias or no bias does not matter"
    assert not is_eligible(nn.Linear(7, 3)), (
        "k % 4 != 0 does not sit on the word boundary of the i8 payload"
    )
    assert not is_eligible(nn.Conv1d(4, 4, 3))


def test_attach_counts_only_eligible_linears_and_changes_the_forward() -> None:
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 4), nn.Linear(4, 3))
    x = torch.randn(2, 8, dtype=torch.float32)
    with torch.no_grad():
        plain = model(x).clone()

    handles, attached = attach_act_quant(model)
    assert attached == 3, "all 3 have k % 4 == 0"
    with torch.no_grad():
        quantized = model(x).clone()
    # With the hooks in effect the output moves (zero hooks raise nothing either, so the move
    # itself is the detector)
    assert not torch.equal(plain, quantized)

    detach_act_quant(handles)
    assert handles == []
    with torch.no_grad():
        restored = model(x).clone()
    assert torch.equal(plain, restored), "detach did not restore the original forward"


def test_attach_skips_ineligible_linear() -> None:
    model = nn.Sequential(nn.Linear(7, 4), nn.Linear(4, 3))
    handles, attached = attach_act_quant(model)
    try:
        assert attached == 1
    finally:
        detach_act_quant(handles)


def test_a_linear_called_with_keyword_arguments_only_fails_loudly() -> None:
    """A call without positional arguments raises instead of passing through (rules out the case
    where only the attached count goes up).

    Passing through would record a reference with activation quantization missing on that layer
    alone, and neither the "attached count" diagnostic nor the tolerance check could tell that
    difference from quantization error. The diagnostic names the target's FQN.
    """
    model = KeywordOnlyCaller()
    handles, attached = attach_act_quant(model)
    try:
        assert attached == 1, (
            "it is counted as an eligible linear (the premise of the pass-through)"
        )
        with pytest.raises(QuantizeError, match="'fc'"), torch.no_grad():
            model(torch.randn(2, 8, dtype=torch.float32))
    finally:
        detach_act_quant(handles)


def test_a_non_tensor_first_positional_argument_fails_loudly() -> None:
    """A non-Tensor first positional argument also fails with karume's diagnostic (not an internal
    error from torch)."""
    model = nn.Sequential(nn.Linear(8, 4))
    handles, _ = attach_act_quant(model)
    try:
        with pytest.raises(QuantizeError, match="'0'"), torch.no_grad():
            model([0.0] * 8)
    finally:
        detach_act_quant(handles)


def test_first_layer_input_is_quantized() -> None:
    """The hook applies to the **input** (applying it to the output gives numbers shifted by one
    layer)."""
    layer = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    x = torch.tensor([[127.0, 0.4, 0.0, 0.0]], dtype=torch.float32)
    handles, _ = attach_act_quant(layer)
    try:
        with torch.no_grad():
            out = layer(x)
    finally:
        detach_act_quant(handles)
    # s = 1.0, so the input is rounded to [127, 0, 0, 0] and the output is 127
    assert out.item() == 127.0
