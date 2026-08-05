"""fake-quant（格納表現可能値への丸め）— ADR 0006 の「丸めは参照より前」の実体。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from karume.ops import (
    CONV1D_OP,
    CONV2D_OP,
    CONV_TRANSPOSE1D_OP,
    EMBEDDING_OP,
    LINEAR_OP,
    WEIGHT_CHANNEL_AXES,
)
from karume.quantize import (
    INT8_MAX,
    QUANT_CHANNEL_AXES,
    QuantizeError,
    channel_scale,
    fake_quant_int8,
    quantize_to_int8,
    round_weights_to_f16,
)


def is_f16_exact(tensor: torch.Tensor) -> bool:
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.register_buffer("scale", torch.randn(3))
        # i64 バッファ（添字表など）は丸めの対象外 — 触ると値が壊れる。
        self.register_buffer("index", torch.arange(3, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


class TestRounding:
    def test_parameters_and_f32_buffers_become_f16_representable(self):
        model = Tiny()
        assert not is_f16_exact(model.linear.weight)

        round_weights_to_f16(model)

        assert is_f16_exact(model.linear.weight)
        assert is_f16_exact(model.linear.bias)
        assert is_f16_exact(model.scale)

    def test_non_f32_buffers_are_left_alone(self):
        """i32 / i64 のバッファは f16 の値域と無関係。丸めに巻き込むと添字が壊れる。"""
        model = Tiny()

        round_weights_to_f16(model)

        assert model.index.dtype is torch.int64
        assert torch.equal(model.index, torch.arange(3, dtype=torch.int64))

    def test_dtype_stays_f32(self):
        """意味論は f32 のまま（格納だけが f16 — ADR 0006）。"""
        model = Tiny()

        round_weights_to_f16(model)

        assert model.linear.weight.dtype is torch.float32

    def test_rounding_is_idempotent(self):
        model = Tiny()
        round_weights_to_f16(model)
        first = model.linear.weight.clone()

        round_weights_to_f16(model)

        assert torch.equal(model.linear.weight, first)

    def test_the_report_counts_what_was_rounded(self):
        model = Tiny()

        report = round_weights_to_f16(model)

        assert report.parameters == 2  # weight / bias
        assert report.buffers == 1  # scale のみ（index は f32 でない）
        assert report.elements == 4 * 3 + 3 + 3


class TestOverflow:
    def test_finite_values_that_saturate_to_infinity_fail_loudly(self):
        """f16 の値域（|x| ≤ 65504）超えは静かに ±inf になる。

        無検査だと「inf を含む重み」と「同じ inf で計算した参照」が一致してしまい、
        E2E は緑のまま出力だけが壊れる（差 0 になる方向の壊れ方）。
        """
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = 1e6

        with pytest.raises(QuantizeError, match="非有限へ飽和"):
            round_weights_to_f16(model)

    def test_already_infinite_values_are_not_reported_as_overflow(self):
        """元から非有限なら丸めのせいではない（limitations: 非有限は検査対象外）。"""
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = float("inf")

        round_weights_to_f16(model)

        assert torch.isinf(model.linear.weight[0, 0])


class Weighted(nn.Module):
    """`WEIGHT_SLOTS` の全 5 op 相当のモジュール（i8 の per-channel 軸を全種類踏む）。"""

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(5, 3)
        self.conv = nn.Conv1d(3, 5, kernel_size=3)
        self.image = nn.Conv2d(2, 3, kernel_size=(3, 1))
        self.up = nn.ConvTranspose1d(5, 2, kernel_size=3, stride=3)
        self.table = nn.Embedding(7, 5)
        # norm 系 weight と生 Parameter は量子化の対象外（bias も同様 — ADR 0006）
        self.norm = nn.LayerNorm(3)
        self.gain = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dense(x)) * self.gain


class TestInt8ChannelAxes:
    def test_the_module_table_mirrors_the_op_table(self):
        """モジュール型で引く表と op 名で引く表が**同じ軸**を返す（ADR 0019）。

        割れると「エクスポータが軸 0 で作った scale をカーネルが軸 1 として引く」になり、
        例外なしで値だけが壊れる。対応は 1 対 1 なので、ここで写像として突き合わせる。
        """
        expected = {
            nn.Linear: LINEAR_OP,
            nn.Conv1d: CONV1D_OP,
            nn.Conv2d: CONV2D_OP,
            nn.ConvTranspose1d: CONV_TRANSPOSE1D_OP,
            nn.Embedding: EMBEDDING_OP,
        }
        assert set(expected) == set(QUANT_CHANNEL_AXES)
        assert set(expected.values()) == set(WEIGHT_CHANNEL_AXES)
        for module_type, op in expected.items():
            assert QUANT_CHANNEL_AXES[module_type] == WEIGHT_CHANNEL_AXES[op], op

    def test_conv_transpose1d_uses_the_second_axis(self):
        """重みが `[Cin, Cout, K]` の転置レイアウト — ここだけ軸 1。"""
        assert QUANT_CHANNEL_AXES[nn.ConvTranspose1d] == 1

    def test_scale_is_a_keepdim_tensor_on_the_channel_axis(self):
        model = Weighted()

        report = fake_quant_int8(model)

        assert list(report.scales["dense.weight"].shape) == [3, 1]
        assert list(report.scales["conv.weight"].shape) == [5, 1, 1]
        assert list(report.scales["image.weight"].shape) == [3, 1, 1, 1]
        assert list(report.scales["up.weight"].shape) == [1, 2, 1]
        assert list(report.scales["table.weight"].shape) == [7, 1]

    def test_a_subclass_of_a_known_module_is_still_quantized(self):
        """実モデルは nn.Linear の薄い派生を使うことがある（黙って対象から外さない）。"""

        class Derived(nn.Linear):
            pass

        model = Derived(4, 3)

        report = fake_quant_int8(model)

        assert list(report.scales["weight"].shape) == [3, 1]


class TestInt8Quantization:
    def test_weights_are_exactly_representable_after_the_round(self):
        """`q·scale` で**ビット一致**して戻る（emit の逆変換ゲートが立つ条件）。"""
        model = Weighted()

        report = fake_quant_int8(model)

        for name, module in [
            ("dense", model.dense),
            ("conv", model.conv),
            ("image", model.image),
            ("up", model.up),
            ("table", model.table),
        ]:
            scale = report.scales[f"{name}.weight"]
            restored = quantize_to_int8(module.weight, scale).to(torch.float32) * scale
            assert torch.equal(restored, module.weight), name

    def test_minus_128_is_never_used(self):
        """−128 を捨てて ±127 に閉じる（最大絶対値要素が厳密復元されて冪等になる）。"""
        model = Weighted()

        report = fake_quant_int8(model)

        for name, scale in report.scales.items():
            weight = model.get_parameter(name)
            quantized = quantize_to_int8(weight, scale)
            assert int(quantized.min()) >= -INT8_MAX, name
            assert int(quantized.max()) <= INT8_MAX, name

    def test_reapplying_the_round_is_bit_identical(self):
        """冪等（ADR 0019）。scale を引き直しても同じ値になることが、二重適用や再 export で
        golden との対応が黙って壊れないことの担保になる。"""
        model = Weighted()
        first = fake_quant_int8(model)
        snapshot = {name: model.get_parameter(name).clone() for name in first.scales}

        second = fake_quant_int8(model)

        for name, before in snapshot.items():
            assert torch.equal(model.get_parameter(name), before), name
            assert torch.equal(second.scales[name], first.scales[name]), f"{name} の scale"

    def test_an_all_zero_channel_does_not_produce_nan(self):
        """`amax == 0` の行は下限 clamp（f32 tiny）で 0/0 を避ける — 値は 0 のまま。"""
        model = Weighted()
        with torch.no_grad():
            model.table.weight[2].zero_()

        report = fake_quant_int8(model)

        assert torch.isfinite(model.table.weight).all()
        assert torch.equal(model.table.weight[2], torch.zeros(5))
        assert float(report.scales["table.weight"][2]) == torch.finfo(torch.float32).tiny

    def test_the_scale_is_the_channel_amax_over_127(self):
        model = Weighted()
        weight = model.dense.weight.detach().clone()

        scale = channel_scale(weight, 0)

        assert torch.equal(scale, weight.abs().amax(dim=1, keepdim=True) / INT8_MAX)

    def test_dtype_stays_f32(self):
        """意味論は f32 のまま（格納だけが i8 — ADR 0006）。"""
        model = Weighted()

        fake_quant_int8(model)

        assert model.dense.weight.dtype is torch.float32

    def test_bias_and_norm_weights_are_left_alone(self):
        """MUST: bias を量子化の対象に載せない（プロトタイプの降格バグの根治形）。"""
        model = Weighted()
        bias = model.dense.bias.detach().clone()
        norm = model.norm.weight.detach().clone()
        gain = model.gain.detach().clone()

        report = fake_quant_int8(model)

        assert torch.equal(model.dense.bias, bias)
        assert torch.equal(model.norm.weight, norm)
        assert torch.equal(model.gain, gain)
        assert "dense.bias" not in report.scales
        assert "norm.weight" not in report.scales

    def test_the_report_counts_what_was_quantized(self):
        model = Weighted()

        report = fake_quant_int8(model)

        assert report.modules == 5
        assert report.elements == 3 * 5 + 5 * 3 * 3 + 3 * 2 * 3 * 1 + 5 * 2 * 3 + 7 * 5


class TestInt8FailsLoudly:
    def test_a_model_without_weight_slots_fails_loudly(self):
        """「i8 指定なのに対象 0 本」を沈黙させない（ADR 0006 の常設診断）。"""

        class NoWeights(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(3)

        with pytest.raises(QuantizeError, match="1 本も無い"):
            fake_quant_int8(NoWeights())

    def test_a_non_f32_weight_fails_loudly(self):
        model = nn.Linear(4, 3)
        model.weight = nn.Parameter(model.weight.to(torch.float16))

        with pytest.raises(QuantizeError, match="量子化できない"):
            fake_quant_int8(model)
