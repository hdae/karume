"""Rewrites **chunk 0** (a single image) of Wan2.2's video VAE encoder into three graphs that can be
exported with the spatial lengths kept symbolic (ADR 0121 decision 11, stage 9a).

The I2V condition latents are produced only by chunk 0 of passing one image (one frame in time)
through upstream `AutoencoderKLWan._encode` (diffusers 0.39.0 `autoencoder_kl_wan.py`)
(`time_conv` does not run and the cache is empty). IR dimension expressions are "a linear
expression of one symbol", so the encoder is split into three graphs around the mid block's
attention (which builds the sequence length h·w on one axis):

- pre ({@link VaeEncoderPre}): conv_in, the four down blocks, mid's resnets[0].
  `[12,1,8h,8w]` → `[640,1,h,w]`.
- attn ({@link VaeEncoderAttention}): mid's attention (sequence layout). `[640,S]` → `[640,S]`.
- post ({@link VaeEncoderPost}): mid's resnets[1], norm_out, conv_out, quant_conv.
  `[640,1,h,w]` → `[48,1,h,w]`.

h, w are the latent height and width, S = h·w. `[640,1,h,w]` ↔ `[640,S]` is only a reinterpretation
of the bytes (the same order). The input of pre is the image after patchify (upstream's `patchify`
— channel = c·p² + width sub-index·p + height sub-index); the output of post is the first half of
quant_conv's output = mu (upstream's `DiagonalGaussianDistribution.mode()`).
The asset is independent of the resolution (ADR 0038 §4) — the check of accepted sizes lives on the
pipeline side.

There are five rewrites, and each only reads the weights of upstream's modules (no class attribute
is replaced — the same discipline as {@link wan.vae_patch}):

1. **Convolution over the time slice** ({@link LastSliceConv3d}) — in chunk 0 a CausalConv3d with
   time kernel 3 pads two zero frames at the start of time before convolving, so only the last time
   slice `weight[:, :, 2]` touches real data. The weight is folded to that one slice
   (`[o,i,1,kh,kw]`) and becomes a conv3d with time padding 0 (only the zero × weight terms vanish
   — the differences are only signed zeros and the reduction order. The GPU equivalence is pinned
   permanently in `gpu_conv3d_parity_test.ts` by identity check ② of ADR 0118 stage 1). Neither
   the causal time pad nor a zero tensor of symbolic shape is needed any more.
2. **Chunk-0 closed form of AvgDown3D** ({@link avg_down_shortcut}) — upstream uses a rank-8 view /
   permute and `mean(dim=2)`. In the pinned configuration ({@link SUPPORTED_SHORTCUTS}) it folds to
   a spatial 2×2 mean or the identity. A block with time factor 2 pads one zero frame before T = 1,
   so the even output channels are all zero (the mean of four zeros) and the odd ones are the
   spatial 2×2 mean of the source channel. The mean is `sum × 1/4` (`aten.mean.dim` is not in the
   exporter's vocabulary; 1/4 is a power of 2, so with the same summation order the result is
   bit-identical). The space-to-depth is a rank-4 decomposition that builds no product of symbols
   (`[C·H/2,2,W/2,2]` → `[C,4,H/2,W/2]`).
3. **Asymmetric ZeroPad2d** ({@link pad_right_bottom}) — IR's pad covers only the last dimension
   with constant 0, so the W side is padded directly and the H side is rotated to the last axis by
   a permute and then padded.
4. **Sequence layout of mid's attention** ({@link VaeEncoderAttention}) — `[C,S]` is laid out as the
   rows of `[S,C]`, and the RMS norm (images=True) and the 1×1 convs (to_qkv / proj) are written as
   a row L2 normalization and linear layers.
5. **Slicing out mu** ({@link VaeEncoderPost}) — the first `z` channels of the `2·z` channels of
   quant_conv's (1×1×1) output.

MUST: only Wan2.2's encoder (`is_residual=True`, patch 2) is accepted. A config whose structure
differs from the expected one fails loudly in {@link assert_supported} (the same practice as
{@link wan.vae_patch.assert_supported}).

MUST: import diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from _shared.vae_rank4 import l2_normalize
from wan.vae_patch import (
    CACHE_FRAMES,
    SAMPLE_CHANNELS,
    UnsupportedVaeError,
    pointwise_conv3d,
    rms_norm,
)

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: The accepted generation (upstream config's `(is_residual, patch_size)`) — Wan2.2 only. The 2.1
#: encoder (`WanResample` for down, no shortcut outside the residual) has no I2V, so it is not
#: written.
SUPPORTED_GENERATION: tuple[bool, int] = (True, 2)

#: The index of the time slice that touches chunk 0's real data in a CausalConv3d with time kernel
#: 3 (the first `CACHE_FRAMES` frames are zero padding — the front of time in upstream's
#: `_padding`).
LAST_SLICE = CACHE_FRAMES

#: The spatial down factor (ZeroPad2d((0,1,0,1)) + Conv2d stride 2, AvgDown3D's factor_s).
DOWN_SCALE = 2

#: The closed table of accepted AvgDown3D configurations: `(factor_t, factor_s, group_size)` → name
#: of the closed form.
#: - `(1, 2, 4)`: spatial 2×2 mean (output channels = input channels).
#: - `(2, 2, 4)`: interleaved with one zero frame at the start of time (output 2c = 0, 2c+1 = the
#:   spatial 2×2 mean of input c).
#: - `(1, 1, 1)`: identity (output = input).
#: A combination outside the table (group 2 / 8 and so on) fails loudly because the channel order
#: assumption has not been checked for it.
SUPPORTED_SHORTCUTS: dict[tuple[int, int, int], str] = {
    (1, 2, 4): "mean2x2",
    (2, 2, 4): "zero-interleaved-mean2x2",
    (1, 1, 1): "identity",
}

#: The down resample modes that run in chunk 0 (downsample3d's `time_conv` does not run in chunk 0
#: — upstream only stores the input in the cache in the first chunk).
DOWN_MODES = frozenset({"downsample2d", "downsample3d"})


# ---- Configuration check ------------------------------------------------------------


def assert_supported(vae: AutoencoderKLWan) -> None:
    """Checks that the encoder has the configuration the rewrite assumes (fails loudly otherwise).

    Only the points the rewrite assumes **structurally** are checked: the generation, the input
    channels (`3 × p²`), the down block type and number of resnets, that each block's shortcut is in
    the table {@link SUPPORTED_SHORTCUTS} and agrees with the residual path (whether it goes down,
    and its time factor), that mid is a single Res → Attention → Res layer, and that the outputs
    (conv_out and quant_conv) have `2·z` channels.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualDownBlock

    config = vae.config
    generation = (bool(config.is_residual), config.patch_size)
    if generation != SUPPORTED_GENERATION:
        raise UnsupportedVaeError(
            f"(is_residual, patch_size) = {generation} is unsupported (only Wan2.2's"
            f" {SUPPORTED_GENERATION} is accepted)"
        )
    encoder = vae.encoder
    channels = SAMPLE_CHANNELS * int(config.patch_size) ** 2
    if int(encoder.conv_in.in_channels) != channels:
        raise UnsupportedVaeError(
            f"the input channels {encoder.conv_in.in_channels} are not 3 × patch_size² = {channels}"
        )
    for index, block in enumerate(encoder.down_blocks):
        if not isinstance(block, WanResidualDownBlock):
            raise UnsupportedVaeError(
                f"down block {index} has type {type(block).__name__}, not WanResidualDownBlock"
            )
        if len(block.resnets) != int(config.num_res_blocks):
            raise UnsupportedVaeError(
                f"down block {index} has {len(block.resnets)} resnets, not the config's"
                f" num_res_blocks {config.num_res_blocks}"
            )
        _assert_shortcut_follows_the_downsampler(index, block)
    mid = encoder.mid_block
    if len(mid.attentions) != 1 or len(mid.resnets) != 2:
        raise UnsupportedVaeError("only a mid block of one Res → Attention → Res layer is written")
    moments = 2 * int(config.z_dim)
    if int(encoder.conv_out.out_channels) != moments:
        raise UnsupportedVaeError(
            f"the output {encoder.conv_out.out_channels} of conv_out is not 2 × z_dim = {moments}"
        )
    quant = vae.quant_conv
    if (int(quant.in_channels), int(quant.out_channels)) != (moments, moments):
        raise UnsupportedVaeError(
            f"quant_conv's {quant.in_channels} → {quant.out_channels} is not {moments} → {moments}"
        )


def _assert_shortcut_follows_the_downsampler(index: int, block: nn.Module) -> None:
    """The shortcut (AvgDown3D) is in the table and has the same factors as the residual path's
    down.

    The spatial factor must agree with whether there is a down (ZeroPad2d + stride-2 Conv2d), and
    the time factor with whether it is downsample3d — otherwise the shapes and orders of chunk 0's
    residual and shortcut do not match (upstream can let it through silently by broadcasting the
    addition).
    """
    shortcut, downsampler = block.avg_shortcut, block.downsampler
    key = (int(shortcut.factor_t), int(shortcut.factor_s), int(shortcut.group_size))
    if key not in SUPPORTED_SHORTCUTS:
        raise UnsupportedVaeError(
            f"down block {index}: the shortcut (factor_t, factor_s, group) = {key} is unsupported"
            f" (only {list(SUPPORTED_SHORTCUTS)} are accepted)"
        )
    if downsampler is None:
        expected = (1, 1)
    elif downsampler.mode in DOWN_MODES:
        expected = (2 if downsampler.mode == "downsample3d" else 1, DOWN_SCALE)
    else:
        raise UnsupportedVaeError(
            f"down block {index}: the resample mode {downsampler.mode} does not occur in the"
            " encoder"
        )
    if key[:2] != expected:
        raise UnsupportedVaeError(
            f"down block {index}: the shortcut scale factors (time {key[0]}, space {key[1]})"
            f" differ from the residual path's {expected}"
        )
    # The channel correspondence (the out = in · factor_t the closed form assumes) is not checked —
    # upstream's AvgDown3D derives group_size from `in · factor_t · factor_s² // out`, so once the
    # group is in the table (each of them factor_s²), out = in · factor_t follows.


# ---- Rewrite parts (they only read upstream modules) --------------------------------------


def _assert_foldable(conv: nn.Module) -> None:
    """Whether the CausalConv3d can be folded to the last time slice in chunk 0 (time kernel 3,
    causal padding 2 at the front of time, symmetric spatial padding, stride / dilation 1)."""
    pad_w, pad_w_after, pad_h, pad_h_after, pad_t, pad_t_after = conv._padding
    kernel_t = int(conv.weight.shape[2])
    if kernel_t != LAST_SLICE + 1 or pad_t != LAST_SLICE or pad_t_after != 0:
        raise UnsupportedVaeError(
            f"time kernel {kernel_t} with causal padding {conv._padding} cannot be folded"
            f" (only {LAST_SLICE} frames at the start of time and kernel {LAST_SLICE + 1})"
        )
    if pad_w != pad_w_after or pad_h != pad_h_after:
        raise UnsupportedVaeError(
            f"the asymmetric spatial padding {conv._padding} cannot be folded into conv3d"
        )
    if tuple(conv.stride) != (1, 1, 1) or tuple(conv.dilation) != (1, 1, 1):
        raise UnsupportedVaeError(
            f"stride {conv.stride} / dilation {conv.dilation} are unsupported (only 1)"
        )


class LastSliceConv3d(nn.Module):
    """A CausalConv3d with time kernel 3 folded to the shape of chunk 0 (one frame in time, the two
    frames before it zero padding).

    The weight is upstream's `weight[:, :, 2:]` (`[o, i, 1, kh, kw]` — a copy is held), the bias a
    copy of upstream's. The input is the unbatched `[Cin, 1, H, W]`; the spatial padding goes into
    conv3d's attrs (the same symmetric values as upstream). Compared with upstream's `F.pad` (two
    zero frames before time) → conv3d, only the terms of zero × the weights of two slices vanish
    (the value differences are only signed zeros and the reduction order).
    """

    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        _assert_foldable(conv)
        self.weight = nn.Parameter(
            conv.weight.detach()[:, :, LAST_SLICE:].clone(), requires_grad=False
        )
        self.bias = (
            None
            if conv.bias is None
            else nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        )
        pad_w, _, pad_h, _, _, _ = conv._padding
        self.padding = (0, int(pad_h), int(pad_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(
                f"the input must be one unbatched frame [C, 1, H, W] ({tuple(x.shape)})"
            )
        return functional.conv3d(x, self.weight, self.bias, padding=self.padding)


def pad_right_bottom(frames: torch.Tensor) -> torch.Tensor:
    """Writes `nn.ZeroPad2d((0, 1, 0, 1))` (one 0 to the right and one to the bottom of
    `[N, C, H, W]`) with last-dimension pads.

    IR's pad covers only the last dimension with constant 0, so the W side is padded directly,
    and the H side is rotated to the last axis by a permute, padded and rotated back (the padded
    value is +0, the same as upstream).
    """
    widened = functional.pad(frames, (0, 1))
    heightened = functional.pad(widened.permute(0, 1, 3, 2), (0, 1))
    return heightened.permute(0, 1, 3, 2)


def downsample(resample: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Chunk 0 of a down resample (downsample2d / downsample3d) (`x` is `[C, 1, H, W]`).

    In chunk 0, downsample3d's `time_conv` does not run (upstream only stores the input in the
    cache in the first chunk), so both modes become "per-frame ZeroPad2d((0,1,0,1)) → Conv2d stride
    2". The permute that puts frames on the batch axis is the same as upstream's
    `permute(0,2,1,3,4).reshape(b*t, …)`.
    """
    if resample.mode not in DOWN_MODES:
        raise UnsupportedVaeError(
            f"resample mode that does not occur in the encoder: {resample.mode}"
        )
    pad, conv = resample.resample
    if tuple(pad.padding) != (0, 1, 0, 1):
        raise UnsupportedVaeError(f"the down's ZeroPad2d {pad.padding} is not (0, 1, 0, 1)")
    frames = pad_right_bottom(x.permute(1, 0, 2, 3))
    return conv(frames).permute(1, 0, 2, 3)


def _mean2x2(x: torch.Tensor) -> torch.Tensor:
    """The spatial 2×2 mean `[C, 1, H/2, W/2]` of each channel of `[C, 1, H, W]` (H, W even).

    Reduces with `sum` in the group order of upstream's AvgDown3D (sub-index `sh·2 + sw`) and then
    multiplies by `1/4` (a power of 2, so bit-identical to `/4`). A rank-4 decomposition that builds
    no product of symbols on one axis:
    `[C·H/2, 2(sh), W/2, 2(sw)]` → `[C·H/2, sh, sw, W/2]` → `[C, H/2, 4, W/2]` →
    `[C, 4, H/2, W/2]`.
    """
    channels, frames, height, width = x.shape
    if frames != 1:
        raise ValueError(f"only the single frame of chunk 0 is accepted ({tuple(x.shape)})")
    rows = x.reshape(channels * (height // DOWN_SCALE), DOWN_SCALE, width // DOWN_SCALE, DOWN_SCALE)
    grouped = rows.permute(0, 1, 3, 2).reshape(
        channels, height // DOWN_SCALE, DOWN_SCALE * DOWN_SCALE, width // DOWN_SCALE
    )
    summed = grouped.permute(0, 2, 1, 3).sum(dim=1)
    mean = summed * (1.0 / (DOWN_SCALE * DOWN_SCALE))
    return mean.reshape(channels, 1, height // DOWN_SCALE, width // DOWN_SCALE)


def avg_down_shortcut(module: nn.Module, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """The chunk-0 (T = 1) closed form of `residual + AvgDown3D(x)` (`x` is the block input
    `[Cin,1,H,W]`, `residual` the output of the residual path `[Cout,1,H',W']`).

    - Identity (`(1,1,1)`): `residual + x`.
    - Spatial 2×2 mean (`(1,2,4)`): `residual + mean2x2(x)`.
    - One zero frame at the start of time (`(2,2,4)`): upstream averages the channel order
      `c·8 + ft·4 + sh·2 + sw` in groups of four, so output 2c is the mean of the time zeros (= +0)
      and 2c+1 is `mean2x2(x)[c]`. Adding +0 to the even channels is the identity (only the signed
      zero −0 + +0 = +0 differs), so it is left out: `residual` is reinterpreted as `[C,2,H',W']`
      and only the odd channels are added to (the slice / cat on the channel axis is on a concrete
      axis of length 2).
    """
    key = (int(module.factor_t), int(module.factor_s), int(module.group_size))
    form = SUPPORTED_SHORTCUTS.get(key)
    if form is None:
        raise UnsupportedVaeError(f"shortcut {key} is unsupported ({list(SUPPORTED_SHORTCUTS)})")
    if int(module.in_channels) != x.shape[0]:
        raise UnsupportedVaeError(
            f"the shortcut's input channels {module.in_channels} do not match the input"
            f" {tuple(x.shape)}"
        )
    if form == "identity":
        return residual + x
    mean = _mean2x2(x)
    if form == "mean2x2":
        return residual + mean
    channels, _, height, width = mean.shape
    pairs = residual.reshape(channels, int(module.factor_t), height, width)
    zero_half, data_half = pairs[:, :1], pairs[:, 1:]
    joined = torch.cat([zero_half, data_half + mean], dim=1)
    return joined.reshape(channels * int(module.factor_t), 1, height, width)


# Third-party code notice. `_ResidualBlock.forward`, `_DownBlock.forward` and
# `VaeEncoderPost.forward` below are adapted from `WanResidualBlock.forward`,
# `WanResidualDownBlock.forward` and the head of `WanEncoder3d.forward` /
# `AutoencoderKLWan._encode` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of operations is kept; the feat_cache bookkeeping is replaced by the chunk-0 fold
# (`LastSliceConv3d`), the AvgDown3D shortcut by its chunk-0 closed form and the defensive
# `x.clone()` is left out (it does not change values).
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
class _ResidualBlock(nn.Module):
    """Chunk 0 of `WanResidualBlock` (shortcut → norm1 → SiLU → conv1 → norm2 → SiLU → dropout →
    conv2 → add). conv1 / conv2 are copies folded into {@link LastSliceConv3d}; the rest are
    upstream's modules."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.conv1 = LastSliceConv3d(block.conv1)
        self.norm2 = block.norm2
        self.conv2 = LastSliceConv3d(block.conv2)
        self.conv_shortcut = block.conv_shortcut
        self.nonlinearity = block.nonlinearity
        self.dropout = block.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = isinstance(self.conv_shortcut, nn.Identity)
        shortcut = x if identity else pointwise_conv3d(self.conv_shortcut, x)
        hidden = self.nonlinearity(rms_norm(self.norm1, x, dim=0))
        hidden = self.conv1(hidden)
        hidden = self.dropout(self.nonlinearity(rms_norm(self.norm2, hidden, dim=0)))
        hidden = self.conv2(hidden)
        return hidden + shortcut


class _DownBlock(nn.Module):
    """Chunk 0 of `WanResidualDownBlock` (Res × n → down → add the shortcut)."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(_ResidualBlock(resnet) for resnet in block.resnets)
        self.downsampler = block.downsampler
        self.avg_shortcut = block.avg_shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for resnet in self.resnets:
            hidden = resnet(hidden)
        if self.downsampler is not None:
            hidden = downsample(self.downsampler, hidden)
        return avg_down_shortcut(self.avg_shortcut, x, hidden)


class VaeEncoderPre(nn.Module):
    """The pre graph: the image after patchify `[12, 1, 8h, 8w]` → the output of mid's resnets[0]
    `[640, 1, h, w]`.

    The down resamples hold upstream's modules, but only `resample` (ZeroPad2d + Conv2d) runs in
    chunk 0 — the weights of downsample3d's `time_conv` do not appear in the graph.
    """

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        assert_supported(vae)
        encoder = vae.encoder
        self.conv_in = LastSliceConv3d(encoder.conv_in)
        self.down_blocks = nn.ModuleList(_DownBlock(block) for block in encoder.down_blocks)
        self.mid_resnet = _ResidualBlock(encoder.mid_block.resnets[0])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_in(image)
        for block in self.down_blocks:
            hidden = block(hidden)
        return self.mid_resnet(hidden)


class _Linear(nn.Module):
    """A linear layer that folds a 1×1 Conv2d's weight `[o, i, 1, 1]` to `[o, i]` (applied to the
    rows of the sequence layout)."""

    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        if tuple(conv.kernel_size) != (1, 1) or tuple(conv.stride) != (1, 1):
            raise UnsupportedVaeError(
                "only a 1×1 Conv2d with stride 1 can be folded into a linear layer"
                f" ({conv.kernel_size}, {conv.stride})"
            )
        out_channels, in_channels = int(conv.out_channels), int(conv.in_channels)
        self.weight = nn.Parameter(
            conv.weight.detach().reshape(out_channels, in_channels).clone(), requires_grad=False
        )
        self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return functional.linear(rows, self.weight, self.bias)


# Third-party code notice. `VaeEncoderAttention.forward` below is adapted from
# `WanAttentionBlock.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of operations is kept; the image layout `[(b t), c, h, w]` is replaced by the sequence
# layout `[S, c]`, so the RMS norm becomes a row L2 normalization and the 1x1 convolutions become
# linear layers.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
class VaeEncoderAttention(nn.Module):
    """The attn graph: mid's single-head attention in the sequence layout (`[C, S]` → `[C, S]`,
    C = 640).

    The input is the bytes of pre's output `[C, 1, h, w]` read as `[C, S]` (S = h·w). After laying
    it out as rows `[S, C]`, it applies the RMS norm (images=True's gamma `(C,1,1)` folded to
    `[C]`), to_qkv (linear), SDPA (`[1, 1, S, C]` — B = 1, H = 1, D = C), proj (linear) and the
    residual addition, in that order (the same as upstream).
    """

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        assert_supported(vae)
        block = vae.encoder.mid_block.attentions[0]
        norm = block.norm
        if isinstance(norm.bias, torch.Tensor) or not norm.channel_first:
            raise UnsupportedVaeError(
                "an RMS_norm with a bias or channel_first=False is unsupported"
            )
        self.channels = int(block.dim)
        self.scale = float(norm.scale)
        self.gamma = nn.Parameter(
            norm.gamma.detach().reshape(self.channels).clone(), requires_grad=False
        )
        self.to_qkv = _Linear(block.to_qkv)
        self.proj = _Linear(block.proj)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2 or tokens.shape[0] != self.channels:
            raise ValueError(
                f"the input must be the sequence layout [{self.channels}, S]"
                f" ({tuple(tokens.shape)})"
            )
        rows = tokens.permute(1, 0)
        hidden = l2_normalize(rows, dim=1) * self.scale * self.gamma
        qkv = self.to_qkv(hidden).reshape(1, 1, rows.shape[0], 3 * self.channels)
        query, key, value = qkv.chunk(3, dim=-1)
        attended = functional.scaled_dot_product_attention(query, key, value)
        projected = self.proj(attended.reshape(rows.shape[0], self.channels))
        return projected.permute(1, 0) + tokens


class VaeEncoderPost(nn.Module):
    """The post graph: `[640, 1, h, w]` → mu `[z, 1, h, w]` (z = 48).

    mid's resnets[1] → norm_out → SiLU → conv_out (folded copy) → quant_conv (1×1×1 — upstream
    applies it at the end of the non-tiled `_encode`) → the first z channels (the mean side of
    `torch.chunk(parameters, 2, dim=1)` in upstream's `DiagonalGaussianDistribution` = `mode()`).
    The logvar second half does not appear in the graph.
    """

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        assert_supported(vae)
        encoder = vae.encoder
        self.mid_resnet = _ResidualBlock(encoder.mid_block.resnets[1])
        self.norm_out = encoder.norm_out
        self.nonlinearity = encoder.nonlinearity
        self.conv_out = LastSliceConv3d(encoder.conv_out)
        self.quant_conv = vae.quant_conv
        self.latent_channels = int(vae.config.z_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.mid_resnet(hidden)
        hidden = self.nonlinearity(rms_norm(self.norm_out, hidden, dim=0))
        moments = pointwise_conv3d(self.quant_conv, self.conv_out(hidden))
        return moments[: self.latent_channels]


def encoder_graphs(
    vae: AutoencoderKLWan,
) -> tuple[VaeEncoderPre, VaeEncoderAttention, VaeEncoderPost]:
    """The modules of the three graphs (eval). The weights are copies of the values at construction
    time (the f16 rounding is done before this)."""
    return (
        VaeEncoderPre(vae).eval(),
        VaeEncoderAttention(vae).eval(),
        VaeEncoderPost(vae).eval(),
    )


def as_sequence(hidden: torch.Tensor) -> torch.Tensor:
    """pre's output `[C, 1, h, w]` → attn's input `[C, S]` (the same bytes reinterpreted)."""
    channels, frames, height, width = hidden.shape
    if frames != 1:
        raise ValueError(f"only the single frame of chunk 0 is accepted ({tuple(hidden.shape)})")
    return hidden.reshape(channels, height * width)


def encode_chunk0(
    graphs: tuple[VaeEncoderPre, VaeEncoderAttention, VaeEncoderPost], image: torch.Tensor
) -> torch.Tensor:
    """Chains the three graphs eagerly (with the same reinterpretations as the host's stage
    assembly).

    `image` is `[12, 1, 8h, 8w]` after patchify. Returns mu `[z, 1, h, w]`.
    """
    pre, attention, post = graphs
    hidden = pre(image)
    channels, _, height, width = hidden.shape
    sequence = attention(as_sequence(hidden))
    return post(sequence.reshape(channels, 1, height, width))


# ---- Reference (upstream itself) ----------------------------------------------------------


def reference_mu(vae: AutoencoderKLWan, image: torch.Tensor) -> torch.Tensor:
    """mu of upstream's non-tiled encode (the I2V condition latents before normalization —
    `[1, z, 1, h, w]`).

    `image` is `[1, 3, 1, H, W]` ([-1, 1], before patchify — upstream's `_encode` patchifies).
    Calls `retrieve_latents(vae.encode(…), sample_mode="argmax")` as the I2V pipeline does
    (`pipeline_wan_i2v.py` — `latent_dist.mode()` = the first half of quant_conv's output).
    MUST: not taken on the tiled path (when `use_tiling` is set, `_encode` branches to
    `tiled_encode`) — the rewrite copies the non-tiled computation.
    """
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

    assert_supported(vae)
    if vae.use_tiling:
        raise UnsupportedVaeError(
            "the reference is taken with the non-tiled _encode (use_tiling is set)"
        )
    if image.dim() != 5 or image.shape[:3] != (1, SAMPLE_CHANNELS, 1):
        raise ValueError(f"pass the image as [1, 3, 1, H, W] ({tuple(image.shape)})")
    return retrieve_latents(vae.encode(image), sample_mode="argmax")


# Third-party code notice. `normalize_condition` below reproduces the normalization of the
# condition latents in `WanImageToVideoPipeline.prepare_latents` in
# huggingface/diffusers
# (`src/diffusers/pipelines/wan/pipeline_wan_i2v.py`, `diffusers==0.39.0`) verbatim.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def normalize_condition(vae: AutoencoderKLWan, mu: torch.Tensor) -> torch.Tensor:
    """Normalization of the condition latents `(mu − mean) · f32(1 / f32(std))` (`mu` is
    `[1, z, F, h, w]`).

    Upstream multiplies by `1.0 / torch.tensor(latents_std)` (an f32 division) — some of the 48
    values differ from f64 `1/std` rounded to f32. The TS side shares the existing form of
    `latents.ts` (`f32(1 / f32(std))`).
    """
    channels = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean).view(1, channels, 1, 1, 1).to(mu.dtype)
    inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, channels, 1, 1, 1).to(mu.dtype)
    return (mu - mean) * inverse_std
