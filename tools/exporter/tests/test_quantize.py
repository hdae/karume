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


class Grouped(nn.Module):
    """i4 の対象（`nn.Linear`）と対象外（conv / embedding / norm / 生 Parameter）を混ぜた模型。

    linear の in 軸はどちらも 32 の倍数（ADR 0069 決定 2 の整除 MUST）で、`wide` は 1 行に
    group が 2 つ入る形 — group ごとに scale が分かれていることを直接踏める。
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
        """形は**重みと同 rank・最終次元だけ group 数**（ADR 0069 決定 3）。

        i8 の keepdim broadcast 形（`[3,1]`）と受理集合が交わらないのがこの形の眼目で、
        取り違えると group ごとの scale が 1 チャネル 1 値として読まれる。
        """
        model = Grouped()

        report = fake_quant_int4(model, group_size=32)

        assert list(report.scales["dense.weight"].shape) == [3, 1]
        assert list(report.scales["wide.weight"].shape) == [2, 2]

    def test_the_default_group_size_is_32(self):
        """Phase 0 sweep の実測で確定した既定（ADR 0069 追記 1）。"""
        model = Grouped()

        report = fake_quant_int4(model)

        assert DEFAULT_GROUP_SIZE == 32
        assert report.group_size == 32
        assert list(report.scales["wide.weight"].shape) == [2, 2]

    def test_weights_are_exactly_representable_after_the_round(self):
        """`q·scale` で**ビット一致**して戻る（emit の逆変換ゲートが立つ条件）。"""
        model = Grouped()

        report = fake_quant_int4(model, group_size=32)

        for name in ("dense.weight", "wide.weight"):
            weight = model.get_parameter(name)
            scale = report.scales[name]
            assert torch.equal(dequantize_int4(quantize_to_int4(weight, scale), scale), weight)

    def test_minus_eight_is_never_used(self):
        """−8 を捨てて ±7 に閉じる（最大絶対値要素が厳密復元されて冪等になる）。"""
        model = Grouped()

        report = fake_quant_int4(model, group_size=32)

        for name, scale in report.scales.items():
            quantized = quantize_to_int4(model.get_parameter(name), scale)
            assert int(quantized.min()) >= -INT4_MAX, name
            assert int(quantized.max()) <= INT4_MAX, name

    def test_every_group_reaches_the_full_seven(self):
        """各 group の最大絶対値要素は必ず `q = ±7` に乗る（`scale = amax/7` の帰結）。

        冪等（次のテスト）の機構そのもの。どこかの group が ±7 に届かないと、そこだけ
        再量子化で scale が縮んで値が動く。
        """
        weight = torch.randn(3, 32)
        scale = group_scale(weight, 16)

        quantized = quantize_to_int4(weight, scale)

        assert torch.equal(
            quantized.reshape(3, 2, 16).abs().amax(dim=-1), torch.full((3, 2), INT4_MAX)
        )

    def test_recomputing_the_scale_from_a_rounded_weight_is_a_fixed_point(self):
        """丸め済みの重みから引き直した scale は同値（冪等が成り立つ理由 — i8 と同じ性質）。

        NOTE（実測・ADR の文言より弱い）: 「amax 要素の**厳密**復元」は f32 では常には
        成り立たない — `fl(7·fl(amax/7))` は amax と 1ulp ずれることがある
        （group 16 の乱数 20 万 group で約 9.5%）。それでも `|q| = 7` に乗る事実から
        `fl(amax(|q·s|)/7) = s` は不動点になり、**冪等は保たれる**（ADR 0069 決定 3 が
        効かせたいのはこちら）。同型の実測は i8 側 `tests/test_emit.py` にもある。
        """
        weight = torch.randn(3, 32)
        scale = group_scale(weight, 16)

        rounded = dequantize_int4(quantize_to_int4(weight, scale), scale)

        assert torch.equal(group_scale(rounded, 16), scale)

    def test_reapplying_the_round_is_bit_identical(self):
        """冪等（ADR 0069 決定 3 — ADR 0019 の ±127 論証の 4bit 版）。"""
        model = Grouped()
        first = fake_quant_int4(model, group_size=32)
        snapshot = {name: model.get_parameter(name).clone() for name in first.scales}

        second = fake_quant_int4(model, group_size=32)

        for name, before in snapshot.items():
            assert torch.equal(model.get_parameter(name), before), name
            assert torch.equal(second.scales[name], first.scales[name]), f"{name} の scale"

    def test_an_all_zero_group_does_not_produce_nan(self):
        """`amax == 0` の group は下限 clamp（f32 tiny）で 0/0 を避ける — 値は 0 のまま。"""
        model = Grouped()
        with torch.no_grad():
            model.wide.weight[1, 32:].zero_()

        report = fake_quant_int4(model, group_size=32)

        assert torch.isfinite(model.wide.weight).all()
        assert torch.equal(model.wide.weight[1, 32:], torch.zeros(32))
        assert float(report.scales["wide.weight"][1, 1]) == torch.finfo(torch.float32).tiny

    def test_only_linear_weights_are_quantized(self):
        """対象は `nn.Linear` の weight だけ（ADR 0069 決定 5 — 実行経路が linear 限定）。

        i8 の {@link QUANT_CHANNEL_AXES} 全 5 種とは対象が違う。conv / embedding を巻き込むと
        「実行できない格納で書かれた重み」ができる。
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
        """意味論は f32 のまま（格納だけが i4 — ADR 0006）。"""
        model = Grouped()

        fake_quant_int4(model, group_size=32)

        assert model.dense.weight.dtype is torch.float32

    def test_a_subclass_of_linear_is_still_quantized(self):
        """実モデルは nn.Linear の薄い派生を使うことがある（黙って対象から外さない）。"""

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


#: 「in 軸」の group を踏むための固定重み — 3 要素の group を 4 つ（= 2 チャネル × 2 group）。
#: scale が 1 / 2 / tiny の 3 通りに割れる並びで、丸めが起きる要素（3.25 → 3・1.75 → 2・
#: 6.5 → 3 段）を含み、かつ**丸めに曖昧さが無い**（.5 の同点が無いので偶数丸めに依存しない）。
GROUP_VALUES: tuple[tuple[float, ...], ...] = (
    (7.0, 3.25, -2.0),
    (-7.0, 1.75, 0.0),
    (14.0, 6.5, -4.0),
    (0.0, 0.0, 0.0),
)

#: {@link GROUP_VALUES} を `q·scale` へ落とした期待値（scale = amax/7 = 1 / 1 / 2 / tiny）。
GROUP_ROUNDED: tuple[tuple[float, ...], ...] = (
    (7.0, 3.0, -2.0),
    (-7.0, 2.0, 0.0),
    (14.0, 6.0, -4.0),
    (0.0, 0.0, 0.0),
)

#: 期待する group scale（チャネル × group）。全ゼロ group は下限 clamp に落ちる。
GROUP_SCALES = ((1.0, 1.0), (2.0, torch.finfo(torch.float32).tiny))


def group_tensor(values: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    """`[チャネル, in]` へ畳んだ期待形（チャネル c は group 2c・2c+1 を持つ）。"""
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
    """`[Cout, Cin, K]` — 出力チャネルごとに `Cin·K` を平坦化した先が group に割れる。"""
    module = nn.Conv1d(2, 2, kernel_size=3)
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES).reshape(2, 2, 3))
    return module


def conv2d_with_groups() -> nn.Conv2d:
    """`[Cout, Cin, Kh, Kw]` — 平坦化は `Cin·Kh·Kw` の行優先（group は Kh 境界で割れる）。"""
    module = nn.Conv2d(1, 2, kernel_size=(2, 3))
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES).reshape(2, 1, 2, 3))
    return module


def conv_transpose1d_with_groups() -> nn.ConvTranspose1d:
    """`[Cin, Cout, K]` の転置レイアウト — **軸 1**（Cout）ごとに `Cin·K` を平坦化する。

    重みの軸 0 と 1 を入れ替えて置くので、軸 0 で読んだ実装は別の group を切ってしまう
    （= 軸の取り違えがこのフィクスチャで露見する）。
    """
    module = nn.ConvTranspose1d(2, 2, kernel_size=3)
    with torch.no_grad():
        module.weight.copy_(group_tensor(GROUP_VALUES).reshape(2, 2, 3).transpose(0, 1))
    return module


class WeightedGroups(nn.Module):
    """5 op 種を 1 つに束ねた模型（**平坦化後の in 軸が全部 6**なので同じ group_size で通る）。"""

    def __init__(self) -> None:
        super().__init__()
        self.dense = linear_with_groups()
        self.conv = conv1d_with_groups()
        self.image = conv2d_with_groups()
        self.up = conv_transpose1d_with_groups()
        self.table = embedding_with_groups()


class TestInt4InAxisPerOpType:
    """group の軸 = 各 op の「in 軸」一般化（`op_types` で広げたときの軸の正しさ）。"""

    @pytest.mark.parametrize(
        ("build", "op_type"),
        [
            (linear_with_groups, nn.Linear),
            (conv1d_with_groups, nn.Conv1d),
            (conv2d_with_groups, nn.Conv2d),
            (conv_transpose1d_with_groups, nn.ConvTranspose1d),
            (embedding_with_groups, nn.Embedding),
        ],
    )
    def test_each_op_type_groups_along_its_own_in_axis(self, build, op_type) -> None:
        """5 op 種とも「出力チャネルごとの受容野」を group に割る（期待値は直書き）。

        期待値は実装と別経路（{@link GROUP_ROUNDED} の直書き）で持つ — 実装の平坦化を
        期待値側でも呼ぶと、軸を取り違えたまま両者が一致して緑になる。
        """
        module = build()

        report = fake_quant_int4(module, group_size=3, op_types=(op_type,))

        assert torch.equal(report.scales["weight"], torch.tensor(GROUP_SCALES))
        assert torch.equal(
            channel_rows(module.weight, QUANT_CHANNEL_AXES[op_type]),
            group_tensor(GROUP_ROUNDED),
        )

    def test_the_transposed_layout_is_written_back_in_place(self) -> None:
        """`ConvTranspose1d` の平坦形は重みの view にならない（`movedim` → `reshape` がコピー）。

        書き戻しを平坦形へ書いて済ませると、丸めは走ったのに**重みは 1 要素も変わらない**
        （報告だけが正しく見える沈黙）。ここは元の `[Cin, Cout, K]` 上で直接確かめる。
        """
        module = conv_transpose1d_with_groups()

        fake_quant_int4(module, group_size=3, op_types=(nn.ConvTranspose1d,))

        expected = group_tensor(GROUP_ROUNDED).reshape(2, 2, 3).transpose(0, 1)
        assert torch.equal(module.weight, expected)

    def test_the_default_target_is_still_linear_only(self) -> None:
        """既定は `nn.Linear` のまま（`op_types` は明示 opt-in — 既存呼び出しの挙動は不変）。"""
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

    def test_widening_to_all_five_types_quantizes_all_of_them(self) -> None:
        """`QUANT_MODULE_TYPES`（i8 と同じ 5 種）まで広げると 5 本とも同じ group で丸まる。"""
        model = WeightedGroups()

        report = fake_quant_int4(model, group_size=3, op_types=QUANT_MODULE_TYPES)

        assert set(report.scales) == {
            "dense.weight",
            "conv.weight",
            "image.weight",
            "up.weight",
            "table.weight",
        }
        assert report.elements == 12 * 5
        for name, module in model.named_children():
            axis = QUANT_CHANNEL_AXES[type(module)]
            assert torch.equal(report.scales[f"{name}.weight"], torch.tensor(GROUP_SCALES)), name
            assert torch.equal(channel_rows(module.weight, axis), group_tensor(GROUP_ROUNDED)), name

    def test_a_group_size_that_does_not_divide_a_flattened_receptive_field_fails_loudly(
        self,
    ) -> None:
        """整除は**平坦化後の in 軸**で見る（conv の `Cin·K` が割り切れなければ落とす）。"""
        model = nn.Conv1d(3, 5, kernel_size=3)  # Cin·K = 9

        with pytest.raises(QuantizeError, match="割り切れない"):
            fake_quant_int4(model, group_size=32, op_types=(nn.Conv1d,))

    def test_a_type_without_a_channel_axis_is_not_a_target(self) -> None:
        """軸を引けない型を渡しても対象にはならない（= 対象 0 本として fail loudly）。"""
        with pytest.raises(QuantizeError, match="1 本も無い"):
            fake_quant_int4(Weighted(), group_size=32, op_types=(nn.LayerNorm,))


#: `[Cout=1, Cin=4, K=2]` の conv1d 重み — 平坦行は `[7,1,2,0, 14,3,0,0]`（`Cin·K` = 8）。
#: group 4 で切ると amax は 7 と 14 = scale 1 と 2 になり、**K（=2）で切った場合と group 境界が
#: 一致しない**ので、畳み方の取り違えが値で露見する。
CONV_ROW_WEIGHT: tuple[tuple[tuple[float, ...], ...], ...] = (
    ((7.0, 1.0), (2.0, 0.0), (14.0, 3.0), (0.0, 0.0)),
)


class TestInt4StorageRowsForRank3:
    """格納の scale 規則は **rank 非依存**（ADR 0069 決定 3）— `[先頭次元, 行長/g]` の rank2。

    conv1d `[Cout,Cin,K]` が唯一の新形（波 J-5b）。行は**連続メモリ順**の `Cin·K` で、
    最終次元（K）で切ると受容野をまたぐ group が別の scale で丸められる（形もバイト長も
    合ったままの沈黙誤値）。
    """

    def test_the_group_scale_of_a_conv1d_weight_is_rank2(self):
        report = fake_quant_int4(conv1d_with_groups(), group_size=3, op_types=(nn.Conv1d,))

        scale = report.scales["weight"]
        assert list(scale.shape) == [2, 2], "重みは rank3 でも scale は rank2"

    def test_the_row_is_flattened_in_contiguous_memory_order(self):
        """期待値は手書き — 実装の畳み込みを期待値側で呼ぶと取り違えたまま緑になる。

        平坦行 `[7,1,2,0 | 14,3,0,0]` の group scale は 1 と 2 で、`3.0` は 2 の格子へ
        `round(3/2)·2 = 4` に落ちる。K（=2）で切る実装なら行長 8 が group 4 で割れず落ち、
        列優先で畳む実装なら `3.0` が amax 7 の group に入って動かない。
        """
        weight = torch.tensor(CONV_ROW_WEIGHT)

        scale = group_scale(weight, 4)

        assert torch.equal(scale, torch.tensor([[1.0, 2.0]]))
        assert torch.equal(
            dequantize_int4(quantize_to_int4(weight, scale), scale),
            torch.tensor((((7.0, 1.0), (2.0, 0.0), (14.0, 4.0), (0.0, 0.0)),)),
        )

    def test_the_round_trip_of_a_rank3_weight_is_bit_identical(self):
        """emit の逆変換ビット一致門（ADR 0069 決定 4 ③）が rank3 でも立つ条件。"""
        module = conv1d_with_groups()

        report = fake_quant_int4(module, group_size=3, op_types=(nn.Conv1d,))

        weight, scale = module.weight, report.scales["weight"]
        assert weight.dim() == 3
        assert torch.equal(dequantize_int4(quantize_to_int4(weight, scale), scale), weight)


class TestInt4GroupSize:
    """group 長は**渡された scale の形**から引く（別引数で受けると宣言と実体が割れる）。"""

    def test_the_group_size_comes_from_the_scale_shape(self):
        assert group_size_of(torch.zeros(3, 64), torch.zeros(3, 2)) == 32

    def test_the_group_size_of_a_rank3_weight_comes_from_the_flattened_row(self):
        """`[3,2,16]` の行長は 32（`Cin·K`）— 最終次元 16 ではない。"""
        assert group_size_of(torch.zeros(3, 2, 16), torch.zeros(3, 2)) == 16

    def test_a_scale_with_the_same_rank_as_a_rank3_weight_fails_loudly(self):
        """「重みと同 rank」は rank2 の重みでしか成り立たない規則（rank3 は rank2 の scale）。"""
        with pytest.raises(QuantizeError, match="group 形"):
            group_size_of(torch.zeros(3, 2, 16), torch.zeros(3, 2, 1))

    def test_a_scale_with_a_different_rank_fails_loudly(self):
        with pytest.raises(QuantizeError, match="group 形"):
            group_size_of(torch.zeros(3, 64), torch.zeros(2))

    def test_a_scale_whose_leading_axes_differ_fails_loudly(self):
        with pytest.raises(QuantizeError, match="group 形"):
            group_size_of(torch.zeros(3, 64), torch.zeros(4, 2))

    def test_a_group_count_that_does_not_divide_the_axis_fails_loudly(self):
        with pytest.raises(QuantizeError, match="割り切らない"):
            group_size_of(torch.zeros(3, 64), torch.zeros(3, 5))


class TestInt4FailsLoudly:
    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        """端数 group を作らない MUST（ADR 0069 決定 2）— 行境界が語境界からずれる。"""
        model = nn.Linear(48, 3)

        with pytest.raises(QuantizeError, match="割り切れない"):
            fake_quant_int4(model, group_size=32)

    def test_a_model_without_linear_fails_loudly(self):
        """「i4 指定なのに対象 0 本」を沈黙させない（ADR 0006 の常設診断）。"""

        class NoLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.table = nn.Embedding(7, 32)

        with pytest.raises(QuantizeError, match="1 本も無い"):
            fake_quant_int4(NoLinear())

    def test_a_non_f32_weight_fails_loudly(self):
        model = nn.Linear(32, 3)
        model.weight = nn.Parameter(model.weight.to(torch.float16))

        with pytest.raises(QuantizeError, match="量子化できない"):
            fake_quant_int4(model)


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


class TestIncludeFilter:
    """`include`（モジュール FQN の述語）— 混成格納で i8 / i4 の対象を排他に割る口。"""

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
        with pytest.raises(QuantizeError, match="1 本も無い"):
            fake_quant_int8(Weighted(), include=lambda fqn: False)

    def test_int4_still_fails_loudly_when_include_drops_everything(self):
        with pytest.raises(QuantizeError, match="1 本も無い"):
            fake_quant_int4(Grouped(), include=lambda fqn: False)

    def test_an_excluded_module_does_not_trip_the_type_ambiguity_check(self):
        """include の判定は型検査より先 — 対象外モジュールの曖昧型で落とさない。"""
        # 軸の違う 2 型（Linear=0 / ConvTranspose1d=1）へ同時に isinstance ヒットする型を
        # __class__ 差し替えで作る（正規の __init__ 連鎖では合成できない — 型判定だけの模型）。
        odd = nn.Linear(4, 3)
        odd.__class__ = type("Odd", (nn.Linear, nn.ConvTranspose1d), {})

        class Holder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.odd = odd
                self.table = nn.Embedding(7, 5)

        with pytest.raises(QuantizeError, match="1 つに決まらない"):
            fake_quant_int8(Holder())
        report = fake_quant_int8(Holder(), include=lambda fqn: fqn == "table")

        assert sorted(report.scales) == ["table.weight"]
