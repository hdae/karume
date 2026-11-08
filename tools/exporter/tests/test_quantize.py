"""fake-quant (rounding to values the storage can represent) — what ADR 0006's "rounding comes
before the reference" is in practice."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from karume.ops import (
    CONV1D_OP,
    CONV2D_OP,
    CONV3D_OP,
    CONV_TRANSPOSE1D_OP,
    EMBEDDING_OP,
    EXPERT_LINEAR_OP,
    LINEAR_OP,
    WEIGHT_CHANNEL_AXES,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    INT4_MAX,
    INT8_MAX,
    QUANT_CHANNEL_AXES,
    QUANT_MODULE_TYPES,
    QuantizeError,
    channel_rows,
    channel_scale,
    dequantize_int4,
    fake_quant_int4,
    fake_quant_int8,
    group_scale,
    group_size_of,
    quantize_to_int4,
    quantize_to_int8,
    round_weights_to_bf16,
    round_weights_to_f16,
)


def is_f16_exact(tensor: torch.Tensor) -> bool:
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.register_buffer("scale", torch.randn(3))
        # i64 buffers (index tables and the like) are not rounded — touching them breaks the values.
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
        """i32 / i64 buffers have nothing to do with the f16 range. Rounding them breaks the
        indices."""
        model = Tiny()

        round_weights_to_f16(model)

        assert model.index.dtype is torch.int64
        assert torch.equal(model.index, torch.arange(3, dtype=torch.int64))

    def test_dtype_stays_f32(self):
        """The semantics stay f32 (only the storage is f16 — ADR 0006)."""
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
        assert report.buffers == 1  # scale only (index is not f32)
        assert report.elements == 4 * 3 + 3 + 3


class TestOverflow:
    def test_finite_values_that_saturate_to_infinity_fail_loudly(self):
        """Values beyond the f16 range (|x| ≤ 65504) silently become ±inf.

        Without a check, "weights containing inf" and "a reference computed with the same inf"
        agree, and the E2E stays green while only the output breaks (a breakage toward a zero
        difference).
        """
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = 1e6

        with pytest.raises(QuantizeError, match=r"saturated .* to non-finite"):
            round_weights_to_f16(model)

    def test_already_infinite_values_are_not_reported_as_overflow(self):
        """A value that was already non-finite is not the rounding's fault (limitations: non-finite
        values are not checked)."""
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = float("inf")

        round_weights_to_f16(model)

        assert torch.isinf(model.linear.weight[0, 0])

    def test_a_saturating_value_next_to_an_existing_infinity_still_fails_loudly(self):
        """The verdict is per element — even with an existing inf in the same tensor, saturation of
        a finite value is not missed.

        The count includes only elements that were "finite before, non-finite after rounding"
        (existing infs are not counted).
        """
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = float("inf")
            model.linear.weight[0, 1] = 1e6

        with pytest.raises(QuantizeError, match="saturated 1 finite values to non-finite"):
            round_weights_to_f16(model)


def is_bf16_exact(tensor: torch.Tensor) -> bool:
    return bool(torch.equal(tensor, tensor.to(torch.bfloat16).to(torch.float32)))


class TestBf16Rounding:
    """bf16 fake-quant (the counterpart of the f16 version — same targets, idempotence and
    counting)."""

    def test_parameters_and_f32_buffers_become_bf16_representable(self):
        model = Tiny()
        assert not is_bf16_exact(model.linear.weight)

        round_weights_to_bf16(model)

        assert is_bf16_exact(model.linear.weight)
        assert is_bf16_exact(model.linear.bias)
        assert is_bf16_exact(model.scale)
        assert model.linear.weight.dtype is torch.float32

    def test_rounding_is_round_to_nearest_even(self):
        """RNE with a 7-bit mantissa — an exact midpoint goes to even, above the midpoint rounds up.

        Truncation (taking the upper 16 bits as is) would drop the second and third values to the
        lower side.
        """
        model = Tiny()
        with torch.no_grad():
            # 1 + 2**-8 is exactly midway between 1.0 and 1 + 2**-7 → to even (1.0).
            model.scale[0] = 1 + 2**-8
            # 1 + 3·2**-9 is above the midpoint → to 1 + 2**-7.
            model.scale[1] = 1 + 3 * 2**-9
            # 1 + 3·2**-8 is midway between 1 + 2**-7 and 1 + 2**-6 → to even (1 + 2**-6).
            model.scale[2] = 1 + 3 * 2**-8

        round_weights_to_bf16(model)

        assert model.scale.tolist() == [1.0, 1 + 2**-7, 1 + 2**-6]

    def test_non_f32_buffers_are_left_alone(self):
        model = Tiny()

        round_weights_to_bf16(model)

        assert model.index.dtype is torch.int64
        assert torch.equal(model.index, torch.arange(3, dtype=torch.int64))

    def test_non_persistent_buffers_keep_their_f32_values(self):
        """Computed values absent from the checkpoint (`persistent=False` buffers such as RoPE's
        inv_freq) are used in f32 upstream too, so they are not rounded. Persistent buffers
        (`scale`) are rounded.
        """
        model = Tiny()
        computed = torch.tensor([1 + 2**-9, 1 + 3 * 2**-9, 0.1])
        model.register_buffer("computed", computed.clone(), persistent=False)
        assert "computed" not in model.state_dict()

        report = round_weights_to_bf16(model)

        assert torch.equal(model.computed, computed)
        assert is_bf16_exact(model.scale)
        assert report.buffers == 1

    def test_rounding_is_idempotent(self):
        model = Tiny()
        round_weights_to_bf16(model)
        first = model.linear.weight.clone()

        round_weights_to_bf16(model)

        assert torch.equal(model.linear.weight, first)

    def test_the_report_counts_what_was_rounded(self):
        model = Tiny()

        report = round_weights_to_bf16(model)

        assert report.parameters == 2  # weight / bias
        assert report.buffers == 1  # scale only (index is not f32)
        assert report.elements == 4 * 3 + 3 + 3

    def test_values_beyond_the_f16_range_stay_finite(self):
        """bf16 has the same exponent range as f32 — 1e6, which saturates in f16, rounds and stays
        finite."""
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = 1e6

        round_weights_to_bf16(model)

        assert torch.isfinite(model.linear.weight).all()

    def test_finite_values_at_the_top_of_the_f32_range_fail_loudly(self):
        """f32's largest finite value exceeds bf16's (≈3.39e38) and silently becomes inf under
        RNE."""
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = torch.finfo(torch.float32).max

        with pytest.raises(QuantizeError, match="saturated 1 finite values to non-finite"):
            round_weights_to_bf16(model)

    def test_already_infinite_values_are_not_reported_as_overflow(self):
        """A value that was already non-finite is not the rounding's fault (the same handling as the
        f16 version)."""
        model = Tiny()
        with torch.no_grad():
            model.linear.weight[0, 0] = float("inf")

        round_weights_to_bf16(model)

        assert torch.isinf(model.linear.weight[0, 0])


class Weighted(nn.Module):
    """Modules equivalent to all 6 ops of `WEIGHT_SLOTS` (exercises every kind of i8 per-channel
    axis)."""

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(5, 3)
        self.conv = nn.Conv1d(3, 5, kernel_size=3)
        self.image = nn.Conv2d(2, 3, kernel_size=(3, 1))
        # conv3d (ADR 0118 decision 1) — weight [Cout, Cin, Kt, Kh, Kw], rank 5, axis 0
        self.volume = nn.Conv3d(2, 4, kernel_size=(2, 1, 3))
        self.up = nn.ConvTranspose1d(5, 2, kernel_size=3, stride=3)
        self.table = nn.Embedding(7, 5)
        # Norm weights and bare Parameters are not quantized (nor is bias — ADR 0006)
        self.norm = nn.LayerNorm(3)
        self.gain = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dense(x)) * self.gain


class TestInt8ChannelAxes:
    def test_the_module_table_mirrors_the_op_table(self):
        """The table looked up by module type and the table looked up by op name return **the same
        axis** (ADR 0019).

        A split gives "the kernel reads a scale the exporter built on axis 0 as axis 1", where
        only the values break without an exception. The correspondence is one to one, so they are
        compared as a mapping here.
        """
        expected = {
            nn.Linear: LINEAR_OP,
            nn.Conv1d: CONV1D_OP,
            nn.Conv2d: CONV2D_OP,
            nn.Conv3d: CONV3D_OP,
            nn.ConvTranspose1d: CONV_TRANSPOSE1D_OP,
            nn.Embedding: EMBEDDING_OP,
        }
        assert set(expected) == set(QUANT_CHANNEL_AXES)
        # expert_linear has a weight slot but no module type and no i8: it refuses i8 storage, and
        # its rank-3 weight is written only through fixed weights (ADR 0145 decision 2).
        assert set(expected.values()) == set(WEIGHT_CHANNEL_AXES) - {EXPERT_LINEAR_OP}
        for module_type, op in expected.items():
            assert QUANT_CHANNEL_AXES[module_type] == WEIGHT_CHANNEL_AXES[op], op

    def test_conv_transpose1d_uses_the_second_axis(self):
        """The weight has the transposed layout `[Cin, Cout, K]` — only here is it axis 1."""
        assert QUANT_CHANNEL_AXES[nn.ConvTranspose1d] == 1

    def test_scale_is_a_keepdim_tensor_on_the_channel_axis(self):
        model = Weighted()

        report = fake_quant_int8(model)

        assert list(report.scales["dense.weight"].shape) == [3, 1]
        assert list(report.scales["conv.weight"].shape) == [5, 1, 1]
        assert list(report.scales["image.weight"].shape) == [3, 1, 1, 1]
        assert list(report.scales["volume.weight"].shape) == [4, 1, 1, 1, 1]
        assert list(report.scales["up.weight"].shape) == [1, 2, 1]
        assert list(report.scales["table.weight"].shape) == [7, 1]

    def test_a_subclass_of_a_known_module_is_still_quantized(self):
        """Real models may use thin subclasses of nn.Linear (they are not silently left out)."""

        class Derived(nn.Linear):
            pass

        model = Derived(4, 3)

        report = fake_quant_int8(model)

        assert list(report.scales["weight"].shape) == [3, 1]


class TestInt8Quantization:
    def test_weights_are_exactly_representable_after_the_round(self):
        """Restores **bit-identically** via `q·scale` (the condition for emit's bit-identity check
        of the inverse conversion to pass)."""
        model = Weighted()

        report = fake_quant_int8(model)

        for name, module in [
            ("dense", model.dense),
            ("conv", model.conv),
            ("image", model.image),
            ("volume", model.volume),
            ("up", model.up),
            ("table", model.table),
        ]:
            scale = report.scales[f"{name}.weight"]
            restored = quantize_to_int8(module.weight, scale).to(torch.float32) * scale
            assert torch.equal(restored, module.weight), name

    def test_minus_128_is_never_used(self):
        """Drops −128 and closes at ±127 (the element with the largest magnitude lands on ±127,
        making the scale a fixed point)."""
        model = Weighted()

        report = fake_quant_int8(model)

        for name, scale in report.scales.items():
            weight = model.get_parameter(name)
            quantized = quantize_to_int8(weight, scale)
            assert int(quantized.min()) >= -INT8_MAX, name
            assert int(quantized.max()) <= INT8_MAX, name

    def test_reapplying_the_round_is_bit_identical(self):
        """Idempotent (ADR 0019). That recomputing the scale gives the same values is what
        guarantees that double application or a re-export does not silently break the correspondence
        with the golden."""
        model = Weighted()
        first = fake_quant_int8(model)
        snapshot = {name: model.get_parameter(name).clone() for name in first.scales}

        second = fake_quant_int8(model)

        for name, before in snapshot.items():
            assert torch.equal(model.get_parameter(name), before), name
            assert torch.equal(second.scales[name], first.scales[name]), f"scale of {name}"

    def test_an_all_zero_channel_does_not_produce_nan(self):
        """A row with `amax == 0` avoids 0/0 through the lower clamp (f32 tiny) — the values stay
        0."""
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
        """The semantics stay f32 (only the storage is i8 — ADR 0006)."""
        model = Weighted()

        fake_quant_int8(model)

        assert model.dense.weight.dtype is torch.float32

    def test_bias_and_norm_weights_are_left_alone(self):
        """MUST: bias is not a quantization target (the root fix of the prototype's demotion
        bug)."""
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

        assert report.modules == 6
        assert report.elements == (
            3 * 5 + 5 * 3 * 3 + 3 * 2 * 3 * 1 + 4 * 2 * 2 * 1 * 3 + 5 * 2 * 3 + 7 * 5
        )


class Grouped(nn.Module):
    """A model mixing i4 targets (`nn.Linear`) and non-targets (conv / embedding / norm / bare
    Parameter).

    Both linears have an in axis that is a multiple of 32 (the divisibility MUST of ADR 0069
    decision 2), and `wide` fits two groups in a row — which directly exercises separate scales
    per group.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(32, 3)
        self.wide = nn.Linear(64, 2)
        self.conv = nn.Conv1d(3, 5, kernel_size=3)
        self.table = nn.Embedding(7, 32)
        self.norm = nn.LayerNorm(3)
        self.gain = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dense(x)) * self.gain


class TestInt4GroupQuantization:
    def test_the_scale_is_the_group_amax_over_seven(self):
        weight = torch.randn(3, 32)

        scale = group_scale(weight, 16)

        grouped = weight.reshape(3, 2, 16)
        assert torch.equal(scale, grouped.abs().amax(dim=-1) / INT4_MAX)

    def test_the_scale_has_one_value_per_group(self):
        """The shape has **the weight's rank, with only the last dimension being the group count**
        (ADR 0069 decision 3).

        The point of this shape is that its accepted set does not intersect the i8 keepdim
        broadcast shape (`[3,1]`); mixing them up would read the per-group scales as one value per
        channel.
        """
        model = Grouped()

        report = fake_quant_int4(model, group_size=32)

        assert list(report.scales["dense.weight"].shape) == [3, 1]
        assert list(report.scales["wide.weight"].shape) == [2, 2]

    def test_the_default_group_size_is_32(self):
        """The default confirmed by the Phase 0 sweep measurements (ADR 0069 addendum 1)."""
        model = Grouped()

        report = fake_quant_int4(model)

        assert DEFAULT_GROUP_SIZE == 32
        assert report.group_size == 32
        assert list(report.scales["wide.weight"].shape) == [2, 2]

    def test_weights_are_exactly_representable_after_the_round(self):
        """Restores **bit-identically** via `q·scale` (the condition for emit's bit-identity check
        of the inverse conversion to pass)."""
        model = Grouped()

        report = fake_quant_int4(model, group_size=32)

        for name in ("dense.weight", "wide.weight"):
            weight = model.get_parameter(name)
            scale = report.scales[name]
            assert torch.equal(dequantize_int4(quantize_to_int4(weight, scale), scale), weight)

    def test_minus_eight_is_never_used(self):
        """Drops −8 and closes at ±7 (the element with the largest magnitude lands on ±7, making the
        scale a fixed point)."""
        model = Grouped()

        report = fake_quant_int4(model, group_size=32)

        for name, scale in report.scales.items():
            quantized = quantize_to_int4(model.get_parameter(name), scale)
            assert int(quantized.min()) >= -INT4_MAX, name
            assert int(quantized.max()) <= INT4_MAX, name

    def test_every_group_reaches_the_full_seven(self):
        """The element with the largest magnitude of each group always lands on `q = ±7` (a
        consequence of `scale = amax/7`).

        This is the mechanism of idempotence (the next test) itself. If some group did not reach
        ±7, its scale would shrink on requantization and its values would move.
        """
        weight = torch.randn(3, 32)
        scale = group_scale(weight, 16)

        quantized = quantize_to_int4(weight, scale)

        assert torch.equal(
            quantized.reshape(3, 2, 16).abs().amax(dim=-1), torch.full((3, 2), INT4_MAX)
        )

    def test_recomputing_the_scale_from_a_rounded_weight_is_a_fixed_point(self):
        """A scale recomputed from rounded weights is equal (why idempotence holds — the same
        property as i8).

        NOTE (observed; weaker than the ADR's wording): "**exact** restoration of the amax element"
        does not always hold in f32 — `fl(7·fl(amax/7))` can differ from amax by 1ulp (about 9.5%
        over 200,000 random groups of 16). Still, since `|q| = 7` is reached,
        `fl(amax(|q·s|)/7) = s` is a fixed point and **idempotence holds** (which is what ADR 0069
        decision 3 is after). A measurement of the same kind exists on the i8 side in
        `tests/test_emit.py`.
        """
        weight = torch.randn(3, 32)
        scale = group_scale(weight, 16)

        rounded = dequantize_int4(quantize_to_int4(weight, scale), scale)

        assert torch.equal(group_scale(rounded, 16), scale)

    def test_reapplying_the_round_is_bit_identical(self):
        """Idempotent (ADR 0069 decision 3 — the 4-bit version of ADR 0019's ±127 argument)."""
        model = Grouped()
        first = fake_quant_int4(model, group_size=32)
        snapshot = {name: model.get_parameter(name).clone() for name in first.scales}

        second = fake_quant_int4(model, group_size=32)

        for name, before in snapshot.items():
            assert torch.equal(model.get_parameter(name), before), name
            assert torch.equal(second.scales[name], first.scales[name]), f"scale of {name}"

    def test_an_all_zero_group_does_not_produce_nan(self):
        """A group with `amax == 0` avoids 0/0 through the lower clamp (f32 tiny) — the values stay
        0."""
        model = Grouped()
        with torch.no_grad():
            model.wide.weight[1, 32:].zero_()

        report = fake_quant_int4(model, group_size=32)

        assert torch.isfinite(model.wide.weight).all()
        assert torch.equal(model.wide.weight[1, 32:], torch.zeros(32))
        assert float(report.scales["wide.weight"][1, 1]) == torch.finfo(torch.float32).tiny

    def test_only_linear_weights_are_quantized(self):
        """Only the weights of `nn.Linear` are targets (ADR 0069 decision 5 — the execution path is
        linear only).

        The targets differ from all the kinds of i8's {@link QUANT_CHANNEL_AXES}. Including conv /
        embedding would produce "weights written in a storage that cannot be executed".
        """
        model = Grouped()
        untouched = {
            "conv.weight": model.conv.weight.detach().clone(),
            "table.weight": model.table.weight.detach().clone(),
            "norm.weight": model.norm.weight.detach().clone(),
            "dense.bias": model.dense.bias.detach().clone(),
        }

        report = fake_quant_int4(model, group_size=32)

        assert set(report.scales) == {"dense.weight", "wide.weight"}
        for name, before in untouched.items():
            assert torch.equal(model.get_parameter(name), before), name

    def test_dtype_stays_f32(self):
        """The semantics stay f32 (only the storage is i4 — ADR 0006)."""
        model = Grouped()

        fake_quant_int4(model, group_size=32)

        assert model.dense.weight.dtype is torch.float32

    def test_a_subclass_of_linear_is_still_quantized(self):
        """Real models may use thin subclasses of nn.Linear (they are not silently left out)."""

        class Derived(nn.Linear):
            pass

        report = fake_quant_int4(Derived(32, 3), group_size=32)

        assert list(report.scales["weight"].shape) == [3, 1]

    def test_the_report_counts_what_was_quantized(self):
        model = Grouped()

        report = fake_quant_int4(model, group_size=32)

        assert report.modules == 2
        assert report.elements == 3 * 32 + 2 * 64
        assert report.group_size == 32


#: Fixed weights for exercising groups along the "in axis" — 4 groups of 3 elements (= 2 channels
#: × 2 groups). The order splits the scale three ways, 1 / 2 / tiny, includes elements that get
#: rounded (3.25 → 3, 1.75 → 2, 6.5 → 3 steps), and **the rounding is unambiguous** (no .5 ties,
#: so it does not depend on round-to-even).
GROUP_VALUES: tuple[tuple[float, ...], ...] = (
    (7.0, 3.25, -2.0),
    (-7.0, 1.75, 0.0),
    (14.0, 6.5, -4.0),
    (0.0, 0.0, 0.0),
)

#: The expected values of {@link GROUP_VALUES} reduced to `q·scale` (scale = amax/7 = 1 / 1 / 2 /
#: tiny).
GROUP_ROUNDED: tuple[tuple[float, ...], ...] = (
    (7.0, 3.0, -2.0),
    (-7.0, 2.0, 0.0),
    (14.0, 6.0, -4.0),
    (0.0, 0.0, 0.0),
)

#: The expected group scales (channel × group). An all-zero group falls to the lower clamp.
GROUP_SCALES = ((1.0, 1.0), (2.0, torch.finfo(torch.float32).tiny))


def group_tensor(values: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    """The expected form folded to `[channel, in]` (channel c holds groups 2c and 2c+1)."""
    rows = [values[2 * channel] + values[2 * channel + 1] for channel in range(2)]
    return torch.tensor(rows, dtype=torch.float32)


def linear_with_groups() -> nn.Linear:
    module = nn.Linear(6, 2)
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES))
    return module


def embedding_with_groups() -> nn.Embedding:
    module = nn.Embedding(2, 6)
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES))
    return module


def conv1d_with_groups() -> nn.Conv1d:
    """`[Cout, Cin, K]` — `Cin·K`, flattened per output channel, is split into groups."""
    module = nn.Conv1d(2, 2, kernel_size=3)
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES).reshape(2, 2, 3))
    return module


def conv2d_with_groups() -> nn.Conv2d:
    """`[Cout, Cin, Kh, Kw]` — flattened row-major as `Cin·Kh·Kw` (groups split at Kh
    boundaries)."""
    module = nn.Conv2d(1, 2, kernel_size=(2, 3))
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES).reshape(2, 1, 2, 3))
    return module


def conv3d_with_groups() -> nn.Conv3d:
    """`[Cout, Cin, Kt, Kh, Kw]`, rank 5 — flattened row-major as `Cin·Kt·Kh·Kw` (groups split at
    Kt boundaries)."""
    module = nn.Conv3d(1, 2, kernel_size=(2, 1, 3))
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES).reshape(2, 1, 2, 1, 3))
    return module


def conv_transpose1d_with_groups() -> nn.ConvTranspose1d:
    """The transposed layout `[Cin, Cout, K]` — `Cin·K` is flattened per **axis 1** (Cout).

    The weight is placed with axes 0 and 1 swapped, so an implementation that reads axis 0 cuts
    different groups (= this fixture exposes an axis mix-up).
    """
    module = nn.ConvTranspose1d(2, 2, kernel_size=3)
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES).reshape(2, 2, 3).transpose(0, 1))
    return module


class WeightedGroups(nn.Module):
    """A model bundling every kind of `QUANT_MODULE_TYPES` (**the in axis after flattening is 6 for
    all**, so the same group_size passes)."""

    def __init__(self) -> None:
        super().__init__()
        self.dense = linear_with_groups()
        self.conv = conv1d_with_groups()
        self.image = conv2d_with_groups()
        self.volume = conv3d_with_groups()
        self.up = conv_transpose1d_with_groups()
        self.table = embedding_with_groups()


class TestInt4InAxisPerOpType:
    """The group axis = the generalized "in axis" of each op (the correctness of the axis when
    widened with `op_types`)."""

    @pytest.mark.parametrize(
        ("build", "op_type"),
        [
            (linear_with_groups, nn.Linear),
            (conv1d_with_groups, nn.Conv1d),
            (conv2d_with_groups, nn.Conv2d),
            (conv3d_with_groups, nn.Conv3d),
            (conv_transpose1d_with_groups, nn.ConvTranspose1d),
            (embedding_with_groups, nn.Embedding),
        ],
    )
    def test_each_op_type_groups_along_its_own_in_axis(self, build, op_type) -> None:
        """Every op kind splits "the receptive field per output channel" into groups (the expected
        values are written out).

        The expected values come from a path separate from the implementation (the written-out
        {@link GROUP_ROUNDED}) — calling the implementation's flattening on the expected side too
        would make both agree, and go green, with the axes mixed up.
        """
        module = build()

        report = fake_quant_int4(module, group_size=3, op_types=(op_type,))

        assert torch.equal(report.scales["weight"], torch.tensor(GROUP_SCALES))
        assert torch.equal(
            channel_rows(module.weight, QUANT_CHANNEL_AXES[op_type]),
            group_tensor(GROUP_ROUNDED),
        )

    def test_the_transposed_layout_is_written_back_in_place(self) -> None:
        """The flattened form of `ConvTranspose1d` is not a view of the weight (`movedim` →
        `reshape` copies).

        Writing back only into the flattened form would run the rounding while **not a single
        element of the weight changes** (a silence where only the report looks right). This checks
        directly on the original `[Cin, Cout, K]`.
        """
        module = conv_transpose1d_with_groups()

        fake_quant_int4(module, group_size=3, op_types=(nn.ConvTranspose1d,))

        expected = group_tensor(GROUP_ROUNDED).reshape(2, 2, 3).transpose(0, 1)
        assert torch.equal(module.weight, expected)

    def test_the_default_target_is_still_linear_only(self) -> None:
        """The default stays `nn.Linear` (`op_types` is an explicit opt-in — existing calls behave
        as before)."""
        model = Weighted()
        untouched = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name != "dense.weight"
        }

        report = fake_quant_int4(model, group_size=5)

        assert set(report.scales) == {"dense.weight"}
        for name, before in untouched.items():
            assert torch.equal(model.get_parameter(name), before), name

    def test_widening_to_all_types_quantizes_all_of_them(self) -> None:
        """Widening to `QUANT_MODULE_TYPES` (all the kinds, as for i8) rounds all of them with the
        same group."""
        model = WeightedGroups()
        # Pins that the model has every kind (prevents going green with a stale model when a type
        # is added)
        assert {type(module) for module in model.children()} == set(QUANT_MODULE_TYPES)

        report = fake_quant_int4(model, group_size=3, op_types=QUANT_MODULE_TYPES)

        assert set(report.scales) == {
            "dense.weight",
            "conv.weight",
            "image.weight",
            "volume.weight",
            "up.weight",
            "table.weight",
        }
        assert report.elements == 12 * 6
        for name, module in model.named_children():
            axis = QUANT_CHANNEL_AXES[type(module)]
            assert torch.equal(report.scales[f"{name}.weight"], torch.tensor(GROUP_SCALES)), name
            assert torch.equal(channel_rows(module.weight, axis), group_tensor(GROUP_ROUNDED)), name

    def test_a_group_size_that_does_not_divide_a_flattened_receptive_field_fails_loudly(
        self,
    ) -> None:
        """Divisibility is checked on **the in axis after flattening** (a conv whose `Cin·K` does
        not divide fails)."""
        model = nn.Conv1d(3, 5, kernel_size=3)  # Cin·K = 9

        with pytest.raises(QuantizeError, match="is not divisible by"):
            fake_quant_int4(model, group_size=32, op_types=(nn.Conv1d,))

    def test_a_type_without_a_channel_axis_is_not_a_target(self) -> None:
        """A type whose axis cannot be looked up does not become a target (= fails loudly as zero
        targets)."""
        with pytest.raises(QuantizeError, match="there is no weight"):
            fake_quant_int4(Weighted(), group_size=32, op_types=(nn.LayerNorm,))


#: A conv1d weight `[Cout=1, Cin=4, K=2]` — the flattened row is `[7,1,2,0, 14,3,0,0]` (`Cin·K` =
#: 8). Cut into groups of 4, amax is 7 and 14 = scales 1 and 2, and **the group boundaries differ
#: from cutting at K (=2)**, so a folding mix-up shows in the values.
CONV_ROW_WEIGHT: tuple[tuple[tuple[float, ...], ...], ...] = (
    ((7.0, 1.0), (2.0, 0.0), (14.0, 3.0), (0.0, 0.0)),
)


class TestInt4StorageRowsForRank3:
    """The storage's scale rule is **rank-independent** (ADR 0069 decision 3) — rank 2 as
    `[leading dimension, row length/g]`.

    conv1d `[Cout,Cin,K]` is the only new shape (phase J-5b). A row is `Cin·K` in **contiguous
    memory order**; cutting at the last dimension (K) would round groups that span receptive
    fields with a different scale (a silent wrong value with shape and byte length still
    matching).
    """

    def test_the_group_scale_of_a_conv1d_weight_is_rank2(self):
        report = fake_quant_int4(conv1d_with_groups(), group_size=3, op_types=(nn.Conv1d,))

        scale = report.scales["weight"]
        assert list(scale.shape) == [2, 2], "the weight is rank 3 but the scale is rank 2"

    def test_the_row_is_flattened_in_contiguous_memory_order(self):
        """The expected values are hand-written — calling the implementation's folding on the
        expected side would go green with the mix-up in place.

        The group scales of the flattened row `[7,1,2,0 | 14,3,0,0]` are 1 and 2, and `3.0` falls
        onto the grid of 2 as `round(3/2)·2 = 4`. An implementation that cuts at K (=2) fails
        because row length 8 does not split into groups of 4, and one that folds column-major puts
        `3.0` into the group with amax 7, where it does not move.
        """
        weight = torch.tensor(CONV_ROW_WEIGHT)

        scale = group_scale(weight, 4)

        assert torch.equal(scale, torch.tensor([[1.0, 2.0]]))
        assert torch.equal(
            dequantize_int4(quantize_to_int4(weight, scale), scale),
            torch.tensor((((7.0, 1.0), (2.0, 0.0), (14.0, 4.0), (0.0, 0.0)),)),
        )

    def test_the_round_trip_of_a_rank3_weight_is_bit_identical(self):
        """The condition for emit's bit-identity check of the inverse conversion (ADR 0069 decision
        4 ③) to pass for rank 3 too."""
        module = conv1d_with_groups()

        report = fake_quant_int4(module, group_size=3, op_types=(nn.Conv1d,))

        weight, scale = module.weight, report.scales["weight"]
        assert weight.dim() == 3
        assert torch.equal(dequantize_int4(quantize_to_int4(weight, scale), scale), weight)


class TestInt4GroupSize:
    """The group length is taken from **the shape of the passed scale** (taking it as a separate
    argument would split the declaration from the data)."""

    def test_the_group_size_comes_from_the_scale_shape(self):
        assert group_size_of(torch.zeros(3, 64), torch.zeros(3, 2)) == 32

    def test_the_group_size_of_a_rank3_weight_comes_from_the_flattened_row(self):
        """The row length of `[3,2,16]` is 32 (`Cin·K`) — not the last dimension 16."""
        assert group_size_of(torch.zeros(3, 2, 16), torch.zeros(3, 2)) == 16

    def test_a_scale_with_the_same_rank_as_a_rank3_weight_fails_loudly(self):
        """ "The weight's rank" is a rule that holds only for rank-2 weights (rank 3 has a rank-2
        scale)."""
        with pytest.raises(QuantizeError, match="group shape"):
            group_size_of(torch.zeros(3, 2, 16), torch.zeros(3, 2, 1))

    def test_a_scale_with_a_different_rank_fails_loudly(self):
        with pytest.raises(QuantizeError, match="group shape"):
            group_size_of(torch.zeros(3, 64), torch.zeros(2))

    def test_a_scale_whose_leading_axes_differ_fails_loudly(self):
        with pytest.raises(QuantizeError, match="group shape"):
            group_size_of(torch.zeros(3, 64), torch.zeros(4, 2))

    def test_a_group_count_that_does_not_divide_the_axis_fails_loudly(self):
        with pytest.raises(QuantizeError, match="does not divide"):
            group_size_of(torch.zeros(3, 64), torch.zeros(3, 5))


class TestInt4FailsLoudly:
    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        """The MUST of no partial groups (ADR 0069 decision 2) — row boundaries would drift from
        word boundaries."""
        model = nn.Linear(48, 3)

        with pytest.raises(QuantizeError, match="is not divisible by"):
            fake_quant_int4(model, group_size=32)

    def test_a_model_without_linear_fails_loudly(self):
        """ "i4 requested but zero targets" is not silent (ADR 0006's standing diagnostic)."""

        class NoLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.table = nn.Embedding(7, 32)

        with pytest.raises(QuantizeError, match="there is no weight"):
            fake_quant_int4(NoLinear())

    def test_a_non_f32_weight_fails_loudly(self):
        model = nn.Linear(32, 3)
        model.weight = nn.Parameter(model.weight.to(torch.float16))

        with pytest.raises(QuantizeError, match="cannot be quantized"):
            fake_quant_int4(model)


class TestInt8FailsLoudly:
    def test_a_model_without_weight_slots_fails_loudly(self):
        """ "i8 requested but zero targets" is not silent (ADR 0006's standing diagnostic)."""

        class NoWeights(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(3)

        with pytest.raises(QuantizeError, match="there is no weight"):
            fake_quant_int8(NoWeights())

    def test_a_non_f32_weight_fails_loudly(self):
        model = nn.Linear(4, 3)
        model.weight = nn.Parameter(model.weight.to(torch.float16))

        with pytest.raises(QuantizeError, match="cannot be quantized"):
            fake_quant_int8(model)


class TestIncludeFilter:
    """`include` (a predicate on module FQNs) — the hook that splits i8 / i4 targets exclusively in
    mixed storage."""

    def test_int8_only_touches_included_modules(self):
        model = Weighted()
        before = model.dense.weight.detach().clone()

        report = fake_quant_int8(model, include=lambda fqn: fqn == "table")

        assert sorted(report.scales) == ["table.weight"]
        assert torch.equal(model.dense.weight, before)

    def test_int4_only_touches_included_modules(self):
        model = Grouped()
        before = model.wide.weight.detach().clone()

        report = fake_quant_int4(model, include=lambda fqn: fqn == "dense")

        assert sorted(report.scales) == ["dense.weight"]
        assert torch.equal(model.wide.weight, before)

    def test_int8_still_fails_loudly_when_include_drops_everything(self):
        with pytest.raises(QuantizeError, match="there is no weight"):
            fake_quant_int8(Weighted(), include=lambda fqn: False)

    def test_int4_still_fails_loudly_when_include_drops_everything(self):
        with pytest.raises(QuantizeError, match="there is no weight"):
            fake_quant_int4(Grouped(), include=lambda fqn: False)

    def test_an_excluded_module_does_not_trip_the_type_ambiguity_check(self):
        """The include predicate runs before the type check — an ambiguous type on a non-target
        module does not fail."""
        # Builds, by swapping __class__, a type that is an isinstance hit for two types with
        # different axes (Linear=0 / ConvTranspose1d=1) at once (it cannot be built through the
        # regular __init__ chain — a model only for the type check).
        odd = nn.Linear(4, 3)
        odd.__class__ = type("Odd", (nn.Linear, nn.ConvTranspose1d), {})

        class Holder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.odd = odd
                self.table = nn.Embedding(7, 5)

        with pytest.raises(QuantizeError, match="is not unique"):
            fake_quant_int8(Holder())
        report = fake_quant_int8(Holder(), include=lambda fqn: fqn == "table")

        assert sorted(report.scales) == ["table.weight"]


class TestReachableScaleIsAFixedPoint:
    """The basis of idempotence itself: a reachable scale `s = fl(amax/D)` is a fixed point of
    `fl(fl(D·s)/D)`.

    Exact restoration of the amax element (`fl(D·s) == amax`) does not always hold, so
    idempotence does not rest on it (the comment of `INT8_MAX`; ADR 0069 addendum 3). f32 rounding
    is invariant under multiplication by powers of two within the normal range, so covering amax
    with every f32 in `[1, 2)` (2^23 values) covers most normal numbers. The binade of the smallest
    normal number `[2^−126, 2^−125)`, where the scale falls into subnormals, is covered separately.
    D is the top level of the table for i8 (127) / i4 (7) / FP4 (6) / NF4 (1).
    """

    ONE_TO_TWO = 0x3F800000  # the bit pattern of f32 1.0
    SMALLEST_NORMAL = 0x00800000  # the bit pattern of f32 2^−126

    @staticmethod
    def every_f32_in_one_binade(start: int = ONE_TO_TWO) -> torch.Tensor:
        """2^23 values from `start` (the bit pattern at the head of the binade) = every f32 of that
        binade."""
        return torch.arange(start, start + (1 << 23), dtype=torch.int32).view(torch.float32)

    @pytest.mark.parametrize("binade", [ONE_TO_TWO, SMALLEST_NORMAL])
    @pytest.mark.parametrize("max_level", [float(INT8_MAX), float(INT4_MAX), 6.0, 1.0])
    def test_scaling_back_and_forth_returns_the_same_scale(self, max_level: float, binade: int):
        amax = self.every_f32_in_one_binade(binade)
        scale = amax / max_level

        rescaled = (scale * max_level) / max_level

        assert torch.equal(rescaled, scale)

    @pytest.mark.parametrize("max_level", [float(INT8_MAX), float(INT4_MAX), 6.0])
    def test_the_amax_element_itself_is_not_always_restored_exactly(self, max_level: float):
        """Pins that explaining idempotence by "exact restoration of amax" is wrong (some amax
        values are off by 1ulp)."""
        amax = self.every_f32_in_one_binade()

        restored = (amax / max_level) * max_level

        assert bool((restored != amax).any())
