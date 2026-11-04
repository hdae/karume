"""The patch layer that takes Depth Anything V2 (single-image relative depth estimation) through
torch.export → IR (docs/ir-v2.md).

The target is the transformers port (`DepthAnythingForDepthEstimation` of
`depth-anything/Depth-Anything-V2-Small-hf` = DINOv2 backbone + DPT neck / head). Only
transformers class attributes and the `nn.ConvTranspose2d` instances inside the model are
replaced; not a single weight element is rewritten.

Every replacement is **tier 0** (no new op is added), and they fall into 2 groups:

- **Bit-identical (not a single operation added or removed)**
  ① In `DepthAnythingFeatureFusionStage.forward`, **only the last stage** gets `size=None`, and
     the fusion layer calls `F.interpolate(..., scale_factor=2, align_corners=True)`
     (the `scale_factors` form of `aten.upsample_bilinear2d.vec`, which the converter does not
     accept — it would give two rounding conventions for the output length;
     `aten_handlers._h_upsample_bilinear2d`). The shapes are static, so the output size is
     computed first and passed as `size=`. **With align_corners=True the scale factor is not
     used** (aten's `area_pixel_compute_scale` builds `(in−1)/(out−1)`), so the values are
     bit-identical.
  ② `Dinov2Embeddings.interpolate_pos_encoding` becomes **identity or fail**. The original
     returns the position embeddings as they are when `num_patches == num_positions and
     height == width`, and otherwise emits `aten.upsample_bicubic2d` (not in the vocabulary).
     Pinned to the pretraining resolution (518² = 37×37 patches) only the former path is taken,
     but it **silently falls back to bicubic when the premise breaks**, so the condition is
     inverted to fail loudly (the values are identical to the original).
- **Not bit-identical (equivalent formula, only the rounding differs) — measured values below**
  ③ The DPT reassemble `nn.ConvTranspose2d` (kernel == stride == 4 / 2, padding 0) becomes
     {@link SubPixelUpsample} (a 1×1 conv + two rank-4 rearrangements equivalent to a pixel
     shuffle). Karume's conv_transpose is 1d only, and limited to the form
     `2·padding == K − stride`, so this form **only goes through as a decomposition**. With
     kernel == stride the transposed convolution has no overlap and every output position is a
     single dot product, so it **exactly** equals a 1×1 conv whose weights are rearranged
     `[Cin, Cout, r, r] → [Cout·r·r, Cin, 1, 1]` composed with a pixel shuffle (the index
     equivalence is pinned by `depth_anything/tests/test_patch.py` with `torch.equal` on
     integer-valued data). On real f32 data the least significant bits move because the
     **reduction order** along Cin differs from ATen's transposed convolution (the measured
     upper bound is spelled only in {@link depth_anything.measurements.CONVT_MAXDIFF} — kept on
     **the side that does not load torch**, and the distribution side takes it from there. The
     RMS of the values ≈ 1.0).

The entry points are split to match the 2 groups ({@link apply_layout_patches} /
{@link apply_module_patches}; {@link apply} applies both) — if they could only be applied
together, the bit identity of ①② could not be measured on its own (the same discipline as
`birefnet.patch`).

MUST: no patch has a fallback. A model that breaks a premise (kernel == stride / the pretraining
resolution) fails on the spot instead of silently dropping to a different numeric path.

NOTE: ①② replace class attributes process-wide, so "the reference before patching" can be taken
only once per process ({@link patches_applied} is the check).
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn

#: The expected depth type (with `metric` the head becomes sigmoid × max_depth, a different graph
#: with a different meaning).
EXPECTED_DEPTH_ESTIMATION_TYPE = "relative"

#: The expected backbone (the DPT neck's input assumes the feature_maps of `Dinov2Backbone`).
EXPECTED_BACKBONE_TYPE = "dinov2"

#: The scale factor of the last fusion stage (`modifier = {"scale_factor": 2}` in the original).
LAST_FUSION_SCALE = 2

#: Whether the replacements were applied (the check that keeps the side taking the pre-patch
#: reference from becoming vacuous).
_APPLIED = False


def patches_applied() -> bool:
    """Whether this process applied the class-attribute replacements."""
    return _APPLIED


def assert_supported(model: nn.Module) -> None:
    """Check the configuration the replacements assume.

    Only what **drops to a different path without a shape error when it deviates** is checked
    (the same approach as `birefnet.patch.assert_supported`).
    """
    config = model.config
    if config.depth_estimation_type != EXPECTED_DEPTH_ESTIMATION_TYPE:
        raise ValueError(
            f"depth_estimation_type is {config.depth_estimation_type!r}"
            f" (anything other than {EXPECTED_DEPTH_ESTIMATION_TYPE!r} takes a different head"
            " path) — not covered by the replacements"
        )
    backbone_config = getattr(config, "backbone_config", None)
    if backbone_config is None or backbone_config.model_type != EXPECTED_BACKBONE_TYPE:
        raise ValueError(
            f"the backbone is not {EXPECTED_BACKBONE_TYPE!r} — not covered by the replacements"
        )
    if not backbone_config.apply_layernorm:
        raise ValueError(
            "apply_layernorm=False — the final LayerNorm is not applied to feature_maps"
            " (the values the DPT neck receives change silently)"
        )
    if backbone_config.reshape_hidden_states:
        raise ValueError(
            "reshape_hidden_states=True — feature_maps are already [B,C,H,W], so the"
            " reassemble's cls removal and reshape would be applied twice"
        )


def pretrained_resolution(model: nn.Module) -> int:
    """The only input resolution at which the position-embedding interpolation **does not
    happen** (= the backbone's `image_size`).

    The check of ② accepts only this one size (`num_patches == num_positions and height ==
    width`).
    """
    image_size = model.config.backbone_config.image_size
    if isinstance(image_size, (list, tuple)):
        raise ValueError(
            f"image_size={image_size!r} is a path the replacements do not have (square only)"
        )
    return int(image_size)


# ---- ① last fusion stage: scale factor → explicit size ---------------------


def _feature_fusion_stage_forward(
    self: nn.Module, hidden_states: Any, size: Any = None
) -> list[torch.Tensor]:
    """An equivalent implementation of `DepthAnythingFeatureFusionStage.forward` (the last stage
    is called with `size=` too).

    It differs from the original in one point only, and the values do not change: the last
    stage's `size=None` (= the fusion layer uses `scale_factor=2`) becomes `(2H, 2W)` computed
    from the input's spatial size. The coordinate formula of `align_corners=True` depends only on
    the input and output sizes, so this is **bit-identical** to passing the scale factor (that
    premise is measured directly by `depth_anything/tests/test_patch.py`).

    NOTE: the fusion layer's other resize branch (`align_corners=False` when the residual's shape
    differs — a variant not in the IR vocabulary) is **structurally dead** in this stage, so no
    guard is placed: the output of stage `i` is resized to the size of `hidden_states[i+1]`, and
    that `hidden_states[i+1]` is the residual of stage `i+1` as is (the neck's convs bring the
    channels of every stage to `fusion_hidden_size`). Should this ever break, the converter's
    `_h_upsample_bilinear2d` rejects `align_corners=False`.
    """
    hidden_states = hidden_states[::-1]

    fused_hidden_states: list[torch.Tensor] = []
    fused_hidden_state: torch.Tensor | None = None

    for index, (hidden_state, layer) in enumerate(zip(hidden_states, self.layers, strict=True)):
        if index != len(hidden_states) - 1:
            target_size = tuple(int(dim) for dim in hidden_states[index + 1].shape[2:])
        else:
            source = hidden_state if fused_hidden_state is None else fused_hidden_state
            target_size = tuple(int(dim) * LAST_FUSION_SCALE for dim in source.shape[2:])

        if fused_hidden_state is None:
            fused_hidden_state = layer(hidden_state, size=target_size)
        else:
            fused_hidden_state = layer(fused_hidden_state, hidden_state, size=target_size)

        fused_hidden_states.append(fused_hidden_state)

    return fused_hidden_states


# ---- ② position-embedding interpolation (identity or fail) ----------------


def _interpolate_pos_encoding(
    self: nn.Module, embeddings: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """An equivalent implementation of `Dinov2Embeddings.interpolate_pos_encoding` (shapes that
    need interpolation fail).

    The original has a branch that returns `self.position_embeddings` as is under the same
    condition (bit-identical). The `aten.upsample_bicubic2d` the original emits otherwise is not in
    the vocabulary, and exported as is it would go unnoticed until the converter, so it fails
    here.
    """
    num_patches = int(embeddings.shape[1]) - 1
    num_positions = int(self.position_embeddings.shape[1]) - 1
    if num_patches != num_positions or height != width:
        raise NotImplementedError(
            f"the patch count {num_patches} differs from the position embeddings'"
            f" {num_positions} (input {height}×{width}) — bicubic interpolation of the position"
            " embeddings is not in the vocabulary. The replacements cover only square input at"
            " the pretraining resolution"
        )
    return self.position_embeddings


# ---- ③ conv_transpose2d (kernel == stride) → 1×1 conv + pixel shuffle ------


class SubPixelUpsample(nn.Module):
    """An equivalent form of `nn.ConvTranspose2d(Cin, Cout, kernel_size=r, stride=r, padding=0)`.

    With kernel == stride the output windows do not overlap, so the transposed convolution

        y[b, co, i·r + ki, j·r + kj] = Σ_ci x[b, ci, i, j]·W[ci, co, ki, kj] + bias[co]

    decomposes into "one dot product + a rearrangement". A 1×1 conv with the weights rearranged
    into `[Cout·r·r, Cin, 1, 1]` does the dot product, and a pixel-shuffle-like rearrangement does
    the indexing. The bias does not depend on position, so it can be replicated over
    `(co, ki, kj)` and folded into the conv (adding **the same value**, not `+0`).

    Written directly, a pixel shuffle is a rank-6 view + permute, beyond the rank limit (4) of the
    IR's strided representation. Here it is split into **two rank-4 rearrangements** (W direction
    → H direction):

        [B, C·r·r, H, W] → [B·C·r, r, H, W] → permute(0,2,3,1) → [B·C, r, H, W·r]
                         → permute(0,2,1,3) → [B, C, H·r, W·r]

    MUST: a `ConvTranspose2d` whose `output_padding` / `groups` / `dilation` / `padding` are not
    the defaults is not accepted (the decomposition does not hold).
    """

    def __init__(self, source: nn.ConvTranspose2d) -> None:
        super().__init__()
        strides = set(source.stride)
        if len(strides) != 1 or set(source.kernel_size) != strides:
            raise NotImplementedError(
                f"a ConvTranspose2d whose kernel_size={source.kernel_size} and"
                f' stride={source.stride} are not "one value shared by H/W" is not replaced'
                " (overlapping windows cannot be decomposed into a 1×1 conv)"
            )
        (stride,) = strides
        # The expected set differs per field (dilation defaults to 1, the others to 0). Checking
        # converted values would make the message state the converted number instead of **the
        # actual upstream value**.
        for what, value, expected in (
            ("padding", source.padding, {0}),
            ("output_padding", source.output_padding, {0}),
            ("dilation", source.dilation, {1}),
        ):
            if set(value) != expected:
                raise NotImplementedError(f"a ConvTranspose2d with {what}={value} is not replaced")
        if source.groups != 1:
            raise NotImplementedError(
                f"a ConvTranspose2d with groups={source.groups} is not replaced"
            )
        if source.bias is None:
            raise NotImplementedError("a ConvTranspose2d without a bias is not replaced")

        weight = source.weight.detach()  # [Cin, Cout, r, r]
        in_channels, out_channels = weight.shape[0], weight.shape[1]
        self.scale = int(stride)
        self.out_channels = int(out_channels)
        self.register_buffer(
            "weight",
            weight.permute(1, 2, 3, 0)
            .reshape(out_channels * stride * stride, in_channels, 1, 1)
            .contiguous(),
        )
        self.register_buffer(
            "bias",
            source.bias.detach()
            .reshape(out_channels, 1)
            .expand(out_channels, stride * stride)
            .reshape(-1)
            .contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale, channels = self.scale, self.out_channels
        projected = nn.functional.conv2d(x, self.weight, self.bias)
        batch, _, height, width = projected.shape
        columns = projected.reshape(batch * channels * scale, scale, height, width)
        columns = columns.permute(0, 2, 3, 1)
        columns = columns.reshape(batch * channels, scale, height, width * scale)
        rows = columns.permute(0, 2, 1, 3)
        return rows.reshape(batch, channels, height * scale, width * scale)


def _replace_transposed_convs(model: nn.Module) -> int:
    """Replace `nn.ConvTranspose2d` with {@link SubPixelUpsample} (walking the instances)."""
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.ConvTranspose2d):
                setattr(parent, name, SubPixelUpsample(child))
                replaced += 1
    return replaced


# ---- applying ---------------------------------------------------------------


def apply_layout_patches(model: nn.Module) -> dict[str, int]:
    """Apply the bit-identical replacements ①② (two transformers class attributes).

    This step alone is **bit-identical** to the unpatched eager model, and the first point of
    `depth_anything.export.py --verify` measures that claim. That is why it is exposed separately
    from ③.
    """
    global _APPLIED
    assert_supported(model)
    neck = sys.modules[type(model.neck).__module__]
    embeddings = sys.modules[type(model.backbone.embeddings).__module__]
    neck.DepthAnythingFeatureFusionStage.forward = _feature_fusion_stage_forward
    embeddings.Dinov2Embeddings.interpolate_pos_encoding = _interpolate_pos_encoding
    _APPLIED = True
    return {"fusion_stage": 1, "interpolate_pos_encoding": 1}


def apply_module_patches(model: nn.Module) -> dict[str, int]:
    """Apply replacement ③, which moves the rounding (walking the `ConvTranspose2d` instances)."""
    return {"conv_transpose2d": _replace_transposed_convs(model)}


def apply(model: nn.Module) -> dict[str, int]:
    """Apply every replacement (the export path — ①②③). Returns the replacement counts."""
    return {**apply_layout_patches(model), **apply_module_patches(model)}
