"""Regression tests that the rewrites of `depth_anything.patch` are equivalent to the original
implementation (the part that needs no real weights).

End-to-end equivalence with the real weights at 518² is the job of the export script
(`depth_anything.export.py --verify`); this file pins, unit by unit, "which rewrite is
equivalent at which strength".

The claims come at 2 strengths (matching the docstring of `depth_anything.patch`):

- **Index equivalence** (`ConvTranspose2d` → 1×1 conv + pixel shuffle) is **`torch.equal` on
  integer-valued data**. With integer values the f32 multiply-adds do not round, so a different
  reduction order makes no difference — that is, "difference 0" directly claims that **the index
  mapping matches exactly** (Cin / Cout / H / W all get different lengths so that an axis mix-up
  does not slip through). Realistic data (random f32) moves by the reduction order, so it is
  checked with a tolerance of the rounding width.
- **Operation-sequence equivalence** (last fusion stage scale factor → explicit size / the check
  of the position-embedding interpolation) is **`torch.equal` = bit match**. Not a single
  operation is added or removed, so letting it pass on "the difference is small" would let a size
  mix-up slip through.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from depth_anything import patch as patch_depth_anything
from depth_anything.patch import SubPixelUpsample

#: The tolerance of the `ConvTranspose2d` decomposition on random f32 (the rounding width of the
#: different reduction order along Cin).
#:
#: MUST: do not state the measured end-to-end width with the real weights at 518² here — that is
#: spelled only in {@link depth_anything.measurements.CONVT_MAXDIFF} (the MUST of that file. If it
#: split from the number the card and `NOTICE.md` state, the prose would stay plausible when only
#: one moves). This is the bound on a **synthetic module**, with headroom above the width measured
#: with the real weights.
SUB_PIXEL_ATOL = 1e-5


def _integer_transposed_conv(in_channels: int, out_channels: int, scale: int) -> nn.ConvTranspose2d:
    """A `ConvTranspose2d(kernel == stride)` whose weights and bias are **integer-valued**.

    Integer values keep the f32 multiply-adds in the range where they do not round (they match
    bit for bit even with a different reduction order, so difference 0 means index equivalence
    itself).
    """
    module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)
    generator = torch.Generator().manual_seed(20260814 + scale)
    with torch.no_grad():
        module.weight.copy_(torch.randint(-4, 5, module.weight.shape, generator=generator).float())
        module.bias.copy_(torch.randint(-4, 5, module.bias.shape, generator=generator).float())
    return module.eval()


class TestSubPixelUpsample:
    """Equivalence of `ConvTranspose2d(kernel == stride)` → 1×1 conv + pixel shuffle."""

    @pytest.mark.parametrize("scale", [2, 3, 4])
    def test_integer_data_matches_the_transposed_convolution_bit_for_bit(self, scale: int) -> None:
        """On integer-valued data it is a **bit match** (= the index mapping matches exactly).

        Cin / Cout / H / W / batch all get different lengths — checked only with square shapes
        or Cin == Cout, a mix-up of the weight axes (swapping the first 2 axes of
        `[Cin, Cout, r, r]`) or of H/W would pass with difference 0.
        """
        source = _integer_transposed_conv(3, 5, scale)
        generator = torch.Generator().manual_seed(7)
        x = torch.randint(-3, 4, (2, 3, 4, 6), generator=generator).float()

        with torch.no_grad():
            expected = source(x)
            got = SubPixelUpsample(source)(x)

        assert tuple(got.shape) == tuple(expected.shape)
        assert torch.equal(got, expected)

    def test_random_float_data_stays_within_the_rounding_width(self) -> None:
        """On random f32 it moves by the reduction order (staying within the rounding width)."""
        source = nn.ConvTranspose2d(48, 48, kernel_size=4, stride=4).eval()
        generator = torch.Generator().manual_seed(11)
        x = torch.randn((1, 48, 37, 37), generator=generator)

        with torch.no_grad():
            expected = source(x)
            got = SubPixelUpsample(source)(x)

        assert not torch.equal(got, expected), (
            "if the rounding does not move, the tolerance loses its justification"
        )
        assert float((got - expected).abs().max()) < SUB_PIXEL_ATOL

    @pytest.mark.parametrize("scale", [2, 3])
    def test_the_shuffle_matches_pixel_shuffle(self, scale: int) -> None:
        """The rearrangement part is `F.pixel_shuffle` itself (splitting it into rank-4 keeps the
        mapping).

        With the 1×1 conv of {@link SubPixelUpsample} made the identity (identity matrix, bias 0),
        only the rearrangement remains. It is compared with an independent reference (ATen's
        `pixel_shuffle`).
        """
        channels = 3
        source = nn.ConvTranspose2d(channels * scale * scale, channels, scale, stride=scale)
        with torch.no_grad():
            source.weight.zero_()
            source.bias.zero_()
            # weight is [Cin, Cout, r, r] — read Cin = c·r² as (c, ki, kj) to build the identity
            # mapping.
            for out_channel in range(channels):
                for row in range(scale):
                    for column in range(scale):
                        index = (out_channel * scale + row) * scale + column
                        source.weight[index, out_channel, row, column] = 1.0

        generator = torch.Generator().manual_seed(13)
        x = torch.randn((2, channels * scale * scale, 4, 6), generator=generator)
        with torch.no_grad():
            got = SubPixelUpsample(source)(x)

        assert torch.equal(got, nn.functional.pixel_shuffle(x, scale))

    @pytest.mark.parametrize(
        ("kwargs", "why", "match"),
        [
            (
                {"kernel_size": 3, "stride": 2},
                "kernel != stride makes the windows overlap",
                r"kernel_size=\(3, 3\) and stride=\(2, 2\)",
            ),
            (
                {"kernel_size": 2, "stride": 2, "padding": 1},
                "padding drops the edges",
                r"padding=\(1, 1\)",
            ),
            (
                {"kernel_size": 2, "stride": 2, "output_padding": 1},
                "output_padding extends the edges",
                r"output_padding=\(1, 1\)",
            ),
            (
                {"kernel_size": 2, "stride": 2, "dilation": 2},
                "dilation spreads the window",
                r"dilation=\(2, 2\)",
            ),
            ({"kernel_size": 2, "stride": 2, "groups": 2}, "groups splits Cin", r"groups=2"),
            (
                {"kernel_size": 2, "stride": 2, "bias": False},
                "no bias is a form not seen in practice",
                r"without a bias",
            ),
        ],
    )
    def test_forms_outside_the_decomposition_are_rejected(
        self, kwargs: dict[str, object], why: str, match: str
    ) -> None:
        """Forms where the decomposition does not hold fail loudly (no silent drop to different
        numbers).

        The message is checked too because a rejection reason that states **the actual upstream
        value** decides the reader's next move. Checking only the type would let a regression
        slip through in which the check converts the value before reporting it (reporting
        `dilation=(2, 2)` as `dilation=(1, 1)`).
        """
        source = nn.ConvTranspose2d(4, 4, **kwargs)
        with pytest.raises(NotImplementedError, match=match):
            SubPixelUpsample(source)


class TestFeatureFusionStage:
    """Equivalence of the last fusion stage's `scale_factor=2` → `size=(2H, 2W)`."""

    @staticmethod
    def _stage(fusion_hidden_size: int) -> nn.Module:
        modeling = pytest.importorskip("transformers.models.depth_anything.modeling_depth_anything")
        configuration = pytest.importorskip(
            "transformers.models.depth_anything.configuration_depth_anything"
        )
        config = configuration.DepthAnythingConfig(
            fusion_hidden_size=fusion_hidden_size,
            neck_hidden_sizes=[fusion_hidden_size] * 4,
        )
        stage = modeling.DepthAnythingFeatureFusionStage(config).eval()
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for parameter in stage.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
        return stage

    @staticmethod
    def _features(channels: int) -> list[torch.Tensor]:
        """4 reassemble-like tensors (descending resolution, **non-square** to catch an H/W
        mix-up)."""
        generator = torch.Generator().manual_seed(19)
        return [
            torch.randn((1, channels, height, width), generator=generator)
            for height, width in ((40, 56), (20, 28), (10, 14), (5, 7))
        ]

    def test_the_rewritten_stage_is_bit_exact(self) -> None:
        """The replacement's output and the original's are a **bit match** (all 4 stages)."""
        stage = self._stage(6)
        features = self._features(6)

        with torch.no_grad():
            expected = stage(features)
            got = patch_depth_anything._feature_fusion_stage_forward(stage, features)

        assert len(got) == len(expected)
        for index, (left, right) in enumerate(zip(got, expected, strict=True)):
            assert tuple(left.shape) == tuple(right.shape), index
            assert torch.equal(left, right), index

    def test_the_last_stage_doubles_the_spatial_size(self) -> None:
        """Only the last stage doubles the input size (the same as `scale_factor=2` in the
        original)."""
        stage = self._stage(6)
        features = self._features(6)

        with torch.no_grad():
            got = patch_depth_anything._feature_fusion_stage_forward(stage, features)

        assert [tuple(item.shape[2:]) for item in got] == [(10, 14), (20, 28), (40, 56), (80, 112)]

    def test_the_residual_resize_branch_stays_dead(self) -> None:
        """The residual resize branch (`align_corners=False` in the original) is
        **structurally** dead.

        The output of stage `i` is resized to the size of `hidden_states[i+1]`, so it always has
        the same shape as the residual of stage `i+1`. The same holds when the input resolutions
        are distorted (the size is taken from the residual itself, so they move together) — the
        reason `patch_depth_anything._feature_fusion_stage_forward` has no guard.
        """
        stage = self._stage(6)
        features = self._features(6)
        features[1] = features[1][:, :, :-3, :-5]

        with torch.no_grad():
            got = patch_depth_anything._feature_fusion_stage_forward(stage, features)

        assert [tuple(item.shape[2:]) for item in got] == [(10, 14), (17, 23), (40, 56), (80, 112)]

    @pytest.mark.parametrize(("height", "width"), [(5, 7), (1, 3), (13, 13)])
    def test_size_and_scale_factor_agree_bit_for_bit(self, height: int, width: int) -> None:
        """Measure the premise of the rewrite directly (with `align_corners=True` the scale
        factor is not used).

        aten's `area_pixel_compute_scale` builds `(in−1)/(out−1)` from the input and output sizes
        when `align_corners=True`, so a scale factor and an explicit size give **the same
        coordinate formula**. If this breaks, the replacement's bit-match claim breaks with it.
        """
        generator = torch.Generator().manual_seed(23)
        x = torch.randn((1, 3, height, width), generator=generator)
        by_scale = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        by_size = nn.functional.interpolate(
            x, size=(height * 2, width * 2), mode="bilinear", align_corners=True
        )
        assert torch.equal(by_scale, by_size)


class TestInterpolatePosEncoding:
    """The check of the position-embedding interpolation (identity or fail)."""

    @staticmethod
    def _embeddings(num_positions: int, dim: int = 4) -> nn.Module:
        module = nn.Module()
        module.position_embeddings = nn.Parameter(torch.randn((1, num_positions + 1, dim)))
        return module

    def test_the_matching_shape_returns_the_table_itself(self) -> None:
        """When the patch count matches, the position embeddings are returned **as is** (the same
        as the original's branch)."""
        module = self._embeddings(9)
        embeddings = torch.zeros((1, 10, 4))

        got = patch_depth_anything._interpolate_pos_encoding(module, embeddings, 42, 42)

        assert got is module.position_embeddings

    @pytest.mark.parametrize(
        ("patches", "height", "width"),
        [(16, 42, 42), (9, 42, 56)],
    )
    def test_shapes_that_would_need_interpolation_are_rejected(
        self, patches: int, height: int, width: int
    ) -> None:
        """Shapes that need interpolation (where the original emits bicubic) fail — no op outside
        the vocabulary gets created."""
        module = self._embeddings(9)
        embeddings = torch.zeros((1, patches + 1, 4))

        with pytest.raises(NotImplementedError, match="bicubic"):
            patch_depth_anything._interpolate_pos_encoding(module, embeddings, height, width)


class TestAssertSupported:
    """The check of the configuration the replacements assume (only what deviates without a
    shape error)."""

    @staticmethod
    def _config(**overrides: object) -> object:
        configuration = pytest.importorskip(
            "transformers.models.depth_anything.configuration_depth_anything"
        )
        config = configuration.DepthAnythingConfig()
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_the_reference_configuration_passes(self) -> None:
        model = nn.Module()
        model.config = self._config()
        model.config.backbone_config.apply_layernorm = True
        model.config.backbone_config.reshape_hidden_states = False

        patch_depth_anything.assert_supported(model)

    @pytest.mark.parametrize(
        ("attribute", "value", "match"),
        [
            ("depth_estimation_type", "metric", "depth_estimation_type"),
            ("backbone_apply_layernorm", False, "apply_layernorm"),
            ("backbone_reshape_hidden_states", True, "reshape_hidden_states"),
        ],
    )
    def test_configurations_outside_the_patch_are_rejected(
        self, attribute: str, value: object, match: str
    ) -> None:
        model = nn.Module()
        model.config = self._config()
        model.config.backbone_config.apply_layernorm = True
        model.config.backbone_config.reshape_hidden_states = False
        if attribute.startswith("backbone_"):
            setattr(model.config.backbone_config, attribute.removeprefix("backbone_"), value)
        else:
            setattr(model.config, attribute, value)

        with pytest.raises(ValueError, match=match):
            patch_depth_anything.assert_supported(model)


class TestPretrainedResolution:
    """Reading the only input resolution at which no position-embedding interpolation happens
    (one square size)."""

    @staticmethod
    def _model(image_size: object) -> nn.Module:
        """A minimal skeleton holding only `config.backbone_config.image_size` (synthetic is
        enough)."""
        model = nn.Module()
        model.config = SimpleNamespace(backbone_config=SimpleNamespace(image_size=image_size))
        return model

    def test_a_square_image_size_is_returned_as_an_int(self) -> None:
        resolution = patch_depth_anything.pretrained_resolution(self._model(518))

        assert resolution == 518
        assert isinstance(resolution, int)

    @pytest.mark.parametrize("image_size", [[518, 392], (518, 392), [518, 518]])
    def test_a_pair_shaped_image_size_is_rejected(self, image_size: object) -> None:
        """MUST: non-square is a path the replacements do not have — a pair spelling fails even
        when its values are equal.

        Once it arrives as a list / tuple, the premise of "the one size without interpolation"
        moves to a different path (the upstream `interpolate_pos_encoding`), so it is not
        accepted even when the values are square.
        """
        with pytest.raises(ValueError, match="square only"):
            patch_depth_anything.pretrained_resolution(self._model(image_size))
