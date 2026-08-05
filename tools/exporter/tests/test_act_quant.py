"""活性 per-token i8 fake-quant（`karume.act_quant`）— ランタイム鏡像の数値仕様。

正本は `packages/runtime/src/kernels/quantize-rows.ts` と
`packages/runtime/src/reference/i8a8.ts`。ここが固定するのは
「torch 側がその仕様と同じ格子・同じ丸め・同じ NaN 伝播を持つ」ことで、E2E の参照
フィクスチャ（`anima_pipeline.py --act-quant`）の数値がここに掛かっている。
"""

from __future__ import annotations

import torch
from torch import nn

from karume.act_quant import (
    PACK_ALIGN,
    attach_act_quant,
    detach_act_quant,
    is_eligible,
    quantize_rows,
)


def test_quantize_rows_keeps_shape_and_dtype() -> None:
    x = torch.randn(3, 5, 8, dtype=torch.float32)
    out = quantize_rows(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_zero_row_is_exactly_zero() -> None:
    """全ゼロ行は `s = tiny` → `q = 0` → `0 · tiny = 0` で厳密（下限 clamp の意味）。"""
    x = torch.zeros(2, 6, dtype=torch.float32)
    assert torch.equal(quantize_rows(x), x)


def test_grid_is_closed_at_plus_minus_127() -> None:
    """MUST: **−128 を使わない**（絶対値最大の要素が ±127·s に乗る）。

    −128 を使う実装だと、負側の最大要素が `-128·s` になって `|x̂| > |x|` になる。
    """
    x = torch.tensor([[-4.0, 1.0, 2.5, 0.5]], dtype=torch.float32)
    out = quantize_rows(x)
    scale = 4.0 / 127.0
    assert out[0, 0].item() == -127 * scale
    # 復元値が元の絶対値最大を超えない（−128 を使うと超える）
    assert out.abs().amax().item() <= x.abs().amax().item()


def test_round_is_ties_to_even() -> None:
    """MUST: 同点は偶数側（WGSL の `round` と一致）。half-up 実装だと 0.5→1 / 2.5→3。"""
    # amax = 127 なので s = 127/127 = 1.0 ちょうど → x̂ は round(x) そのもの
    x = torch.tensor([[127.0, 0.5, 1.5, -0.5, -1.5, 2.5, 3.5, -3.5]], dtype=torch.float32)
    out = quantize_rows(x)
    assert out.tolist() == [[127.0, 0.0, 2.0, -0.0, -2.0, 2.0, 4.0, -4.0]]


def test_scale_is_per_row_and_independent() -> None:
    """行ごとに scale が決まる（1 行の外れ値が他の行の格子を潰さない）。"""
    x = torch.tensor([[1.0, -1.0, 0.5, 0.25], [1000.0, 1.0, 0.5, 0.25]], dtype=torch.float32)
    out = quantize_rows(x)
    # 行 0 は s = 1/127 なので 0.25 が 32 段（値を保つ。f32 の丸めぶんだけ許容する）
    assert abs(out[0, 3].item() - 32 / 127) < 1e-7
    # 行 1 は s = 1000/127 なので 0.25 は 0 段へ潰れる（行が独立している証拠）
    assert out[1, 3].item() == 0.0
    assert abs(out[1, 0].item() - 1000.0) < 1e-3


def test_nan_propagates_to_the_whole_row() -> None:
    """行内の 1 つの NaN が行全体を NaN にする（scale 経由の伝播 — ADR 0020 と同じ粒度）。"""
    x = torch.tensor([[1.0, 2.0, float("nan"), 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    out = quantize_rows(x)
    assert bool(out[0].isnan().all())
    assert bool(out[1].isfinite().all())


def test_is_eligible_requires_linear_and_pack_alignment() -> None:
    assert PACK_ALIGN == 4
    assert is_eligible(nn.Linear(8, 3))
    assert is_eligible(nn.Linear(8, 3, bias=False)), "bias の有無は問わない"
    assert not is_eligible(nn.Linear(7, 3)), "k % 4 != 0 は i8 ペイロードの語境界に乗らない"
    assert not is_eligible(nn.Conv1d(4, 4, 3))


def test_attach_counts_only_eligible_linears_and_changes_the_forward() -> None:
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 4), nn.Linear(4, 3))
    x = torch.randn(2, 8, dtype=torch.float32)
    with torch.no_grad():
        plain = model(x).clone()

    handles, attached = attach_act_quant(model)
    assert attached == 3, "3 本とも k % 4 == 0"
    with torch.no_grad():
        quantized = model(x).clone()
    # フックが効いていれば出力が動く（0 本でも例外にならないので、動くこと自体が検出器）
    assert not torch.equal(plain, quantized)

    detach_act_quant(handles)
    assert handles == []
    with torch.no_grad():
        restored = model(x).clone()
    assert torch.equal(plain, restored), "detach が元の forward を戻していない"


def test_attach_skips_ineligible_linear() -> None:
    model = nn.Sequential(nn.Linear(7, 4), nn.Linear(4, 3))
    handles, attached = attach_act_quant(model)
    try:
        assert attached == 1
    finally:
        detach_act_quant(handles)


def test_first_layer_input_is_quantized() -> None:
    """フックは**入力**に掛かる（出力に掛けると 1 層ぶんずれた数になる）。"""
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
    # s = 1.0 なので入力は round されて [127, 0, 0, 0] になり、出力は 127
    assert out.item() == 127.0
