"""Rewrites Wan's video VAE decoder (2.1 and 2.2) into **chunk graphs** (unbatched rank 4)
(ADR 0118, ADR 0121 decision 6).

The decode of upstream `AutoencoderKLWan` (diffusers 0.39.0 `autoencoder_kl_wan.py`) runs the
decoder one latent frame at a time and carries the causal cache (feat_cache) around as a Python list
with the sentinel `'Rep'`. This is restructured into two kinds of static graphs, "one latent frame +
cache tensors → frames + updated cache" (one for the first chunk = `first=True` of
{@link WanVaeChunkDecoder}, one for the following chunks = `first=False`). There are six rewrites,
each written as **a function that reads the weights of upstream's modules as they are**:

1. **Normalized cache + causal padding as a cat** ({@link causal_conv3d}) — the cache always has 2
   frames, starts as zeros, and is updated to the last 2 frames of `cat(cache, x)`. A CausalConv3d
   with time kernel 3 becomes `conv3d(cat(cache, x))` in every chunk (time padding 0, spatial
   padding in the attrs).
2. **RMS_norm in rank 4** ({@link rms_norm}) — `F.normalize` × √C × gamma is written as `sum` →
   `sqrt` → `clamp_min` → division (`linalg_vector_norm` is kept out of the vocabulary — ADR 0017).
3. **upsample3d's time interleave in rank 4** ({@link _shared.vae_rank4.interleave_frames}) —
   upstream's rank 6 (`reshape(b,2,c,t,h,w)` → `stack(dim=3)`) is written with rank-4 reshapes /
   permutes.
4. **Folding per-frame 2D processing into (b t)** ({@link resample} / {@link attention_block}) —
   `[C,T,H,W]` → permute → `[T,C,H,W]` puts the frames on the batch axis of conv2d / attention.
5. **f32 normalize**: diffusers runs normalize in f32 for fp16 / bf16 input
   (`autoencoder_kl_wan.py:202-210`). karume's activations are always f32 (only storage is
   compressed — ADR 0006), so this branch always takes the "stay f32" side and needs no rewrite
   ({@link rms_norm} rejects anything but f32).
6. **The shortcut outside the residual in rank 4** (Wan2.2's `DupUp3D` — {@link avg_shortcut} →
   {@link _shared.vae_rank4.dup_up_3d}) — upstream's rank-8 view / permute is written as a
   duplication and per-axis depth-to-space (each step rank 4). The slice that drops the first
   `ft − 1` frames of time in the first chunk lives on the side of the cache mechanism (here).

The accepted generations are closed to the two of {@link SUPPORTED_GENERATIONS}: Wan2.1
(`WanUpBlock`, output 3 pixel channels) and Wan2.2 (`WanResidualUpBlock`, output 12 channels in
patchify space). The 2.2 output is returned without unpatchify (unpatchify and clamp are on the host
— ADR 0121 decision 6). The 2.1 branch does not go through the 2.2 parts
({@link WanVaeChunkDecoder.forward}).

Why the normalized cache matches upstream in value (the derivation of decision 2): the zeros
upstream pads at the start of time with `F.pad` and the zeros placed in the cache here are the same
values, so the input tensor of each conv3d is element-wise identical to upstream's (for a conv at
latent resolution, upstream goes `[0,0,x0]` → `[0,x0,x1]` → `[x0,x1,x2]`, and here the cache goes
`[0,0]` → `[0,x0]` → `[x0,x1]`, giving the same order). `time_conv` does not run in the first chunk
(upstream's `'Rep'`), so its cache stays zero and chunk 2 reads `[0,0,x1]` — the same as upstream's
`'Rep'` path (a pad of two zeros). This equivalence is measured by {@link normalized_cache_decode}
with **upstream's code unchanged** and only the cache normalized (2.1 in `tests/test_vae_patch.py`,
2.2 in `tests/test_vae_patch_ti2v.py`).

Tensors are unbatched `[C, T, H, W]` (B = 1 is dropped — decision 2). Only the conv3d weights are
rank 5; all values fit in rank 4 or less (the limit of the strided copy family — ADR 0011 / 0014 /
0016).

MUST: no module is replaced (no monkeypatching of class attributes). The functions only **read**
upstream's modules, so the pre-patch reference ({@link reference_decode_unclamped}) can be taken at
any time — this shape makes unnecessary the "already applied" flag check that anima's VAE patch has
(taking the reference after patching would make the equivalence check vacuous).

MUST: import diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from _shared.vae_rank4 import (
    UPSAMPLE_SCALE,
    dup_up_3d,
    interleave_frames,
    l2_normalize,
    nearest_exact_2x,
)

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: The number of frames each CausalConv3d carries over to the next chunk (upstream's `CACHE_T` —
#: `autoencoder_kl_wan.py:31`).
CACHE_FRAMES = 2

#: The prefix of the cache input names (the graph inputs are `cache_00` … — the names `torch.export`
#: gives when it flattens a dict input).
CACHE_INPUT = "cache"

#: The closed table of generations the rewrite accepts: upstream config's
#: `(is_residual, patch_size)` → type name of the up block (the name in
#: `diffusers.models.autoencoders.autoencoder_kl_wan` — diffusers is imported inside functions).
#: Wan2.1 is `WanUpBlock`, Wan2.2 `WanResidualUpBlock` (it has the DupUp3D `avg_shortcut`, and a
#: single dim → dim upsampler). A combination outside the table (`(True, None)`, `(False, 2)`,
#: patch 4 and so on) fails loudly, because the assumptions about whether there is a shortcut and
#: about the output order have not been checked for it.
SUPPORTED_GENERATIONS: dict[tuple[bool, int | None], str] = {
    (False, None): "WanUpBlock",
    (True, 2): "WanResidualUpBlock",
}

#: The number of output pixel channels (RGB). A generation that patchifies returns `× patch_size²`
#: channels (patchify space — 12 for 2.2) as they are.
SAMPLE_CHANNELS = 3

#: upsampler mode → time factor (upsample3d doubles it with `time_conv`'s time interleave).
TIME_FACTORS = {"upsample3d": 2, "upsample2d": 1}


class UnsupportedVaeError(NotImplementedError):
    """Received a VAE outside the configuration the rewrite assumes (a decoder of
    {@link SUPPORTED_GENERATIONS})."""


# ---- Cache table ----------------------------------------------------------------------


@dataclass(frozen=True)
class CacheSlot:
    """One causal cache (corresponds to one CausalConv3d with time kernel 3).

    `index` follows upstream's `feat_idx` order (the order in which the decoder's forward goes
    through the CausalConv3d layers); the names and the order of the graph inputs / outputs follow
    it. `scale` is the spatial factor relative to the side of the latent tile (doubled at every
    nearest ×2 of an up block).
    """

    index: int
    channels: int
    scale: int
    #: upsample3d's `time_conv` (does not run in the first chunk — upstream's `'Rep'`).
    time_conv: bool

    @property
    def key(self) -> str:
        """The key of the dict input (the graph input name is `cache_<key>`)."""
        return f"{self.index:02d}"

    @property
    def name(self) -> str:
        """The graph input name (the spelling `torch.export` gives when it flattens the dict input
        `cache`)."""
        return f"{CACHE_INPUT}_{self.key}"

    def shape(self, tile: int) -> tuple[int, int, int, int]:
        """The shape `[C, 2, h, w]` for a latent tile side `tile`."""
        side = tile * self.scale
        return (self.channels, CACHE_FRAMES, side, side)


def assert_supported(vae: AutoencoderKLWan) -> None:
    """Checks that the decoder has the configuration the rewrite assumes (fails loudly otherwise).

    Only the points the rewrite assumes **structurally** are checked: the generation is in the table
    {@link SUPPORTED_GENERATIONS} and every up block has that generation's type, mid has one
    attention, and the number of output channels is `3 × patch_size²`. For Wan2.2 additionally, each
    up block has a shortcut exactly when it has an upsampler, and the shortcut's factors are the
    upsampler's (space ×2; time ×2 for upsample3d, ×1 for upsample2d) — the convention of 1 frame
    for the first chunk and 4 frames afterwards relies on the residual path and the shortcut having
    the same time factor.
    """
    from diffusers.models.autoencoders import autoencoder_kl_wan

    config = vae.config
    generation = (bool(config.is_residual), config.patch_size)
    if generation not in SUPPORTED_GENERATIONS:
        raise UnsupportedVaeError(
            f"(is_residual, patch_size) = {generation} is unsupported"
            f" (only {list(SUPPORTED_GENERATIONS)} are accepted)"
        )
    decoder = vae.decoder
    if len(decoder.mid_block.attentions) != 1 or len(decoder.mid_block.resnets) != 2:
        raise UnsupportedVaeError("only a mid block of one Res → Attention → Res layer is written")
    up_block_type = getattr(autoencoder_kl_wan, SUPPORTED_GENERATIONS[generation])
    for block in decoder.up_blocks:
        if not isinstance(block, up_block_type):
            raise UnsupportedVaeError(
                f"up block type {type(block).__name__} is not generation {generation}'s"
                f" {up_block_type.__name__}"
            )
        if config.is_residual:
            _assert_shortcut_follows_the_upsampler(block)
    channels = SAMPLE_CHANNELS * (config.patch_size or 1) ** 2
    if decoder.conv_out.out_channels != channels:
        raise UnsupportedVaeError(
            f"the output channels {decoder.conv_out.out_channels} are not"
            f" {SAMPLE_CHANNELS} × patch_size² = {channels}"
        )


def _assert_shortcut_follows_the_upsampler(block: nn.Module) -> None:
    """Whether the shortcut (DupUp3D) of a `WanResidualUpBlock` is present exactly when the
    upsampler is, with the same factors."""
    shortcut, upsampler = block.avg_shortcut, block.upsampler
    if (shortcut is None) != (upsampler is None):
        raise UnsupportedVaeError(
            "an up block's shortcut and upsampler disagree on presence (only one of them exists)"
        )
    if shortcut is None:
        return
    expected = (TIME_FACTORS.get(upsampler.mode), UPSAMPLE_SCALE)
    if (shortcut.factor_t, shortcut.factor_s) != expected:
        raise UnsupportedVaeError(
            f"the shortcut scale factors (time {shortcut.factor_t}, space {shortcut.factor_s})"
            f" differ from the upsampler ({upsampler.mode}), which has {expected}"
        )


def _upsampler_of(block: nn.Module) -> nn.Module | None:
    """The upsampler of an up block (`WanResidualUpBlock` has one `upsampler`, `WanUpBlock`
    `upsamplers[0]`; None if there is none)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualUpBlock, WanUpBlock

    if isinstance(block, WanResidualUpBlock):
        return block.upsampler
    if isinstance(block, WanUpBlock):
        return None if block.upsamplers is None else block.upsamplers[0]
    raise UnsupportedVaeError(f"up block type {type(block).__name__} is unsupported")


def cache_slots(vae: AutoencoderKLWan) -> tuple[CacheSlot, ...]:
    """Builds the causal cache table in the decoder's forward order (the same order as upstream's
    `feat_idx`).

    Derived by walking the real modules (the table is not held by hand): conv_in → mid's Res ×2 →
    each up block's Res ×3 and upsample3d's `time_conv` → head. `shortcut` and the post-quant 1×1×1
    have no cache (time padding 0 — decision 2). Wan2.2's shortcut (DupUp3D) only moves data and has
    no cache either.
    """
    assert_supported(vae)
    decoder = vae.decoder
    slots: list[CacheSlot] = []
    scale = 1

    def add(conv: nn.Module, *, time_conv: bool = False) -> None:
        _assert_causal(conv)
        slots.append(CacheSlot(len(slots), int(conv.in_channels), scale, time_conv))

    add(decoder.conv_in)
    for resnet in decoder.mid_block.resnets:
        add(resnet.conv1)
        add(resnet.conv2)
    for block in decoder.up_blocks:
        for resnet in block.resnets:
            add(resnet.conv1)
            add(resnet.conv2)
        upsampler = _upsampler_of(block)
        if upsampler is not None:
            if upsampler.mode == "upsample3d":
                add(upsampler.time_conv, time_conv=True)
            elif upsampler.mode != "upsample2d":
                raise UnsupportedVaeError(
                    f"resample mode that does not occur in the decoder: {upsampler.mode}"
                )
            scale *= UPSAMPLE_SCALE
    add(decoder.conv_out)
    return tuple(slots)


def graph_slots(slots: Sequence[CacheSlot], *, first: bool) -> tuple[CacheSlot, ...]:
    """The caches the graph has as inputs / outputs (first does not have the two `time_conv` ones —
    the table of decision 2)."""
    return tuple(slot for slot in slots if not (first and slot.time_conv))


def zero_caches(slots: Sequence[CacheSlot], tile: int, *, first: bool) -> dict[str, torch.Tensor]:
    """A dict with the graph's cache inputs set to zeros (key order = graph input order)."""
    return {slot.key: torch.zeros(slot.shape(tile)) for slot in graph_slots(slots, first=first)}


# ---- Rewrite parts (they only read upstream modules) --------------------------------------


def _assert_causal(conv: nn.Module) -> None:
    """Whether it is a CausalConv3d with time kernel 3, causal padding 2 and stride / dilation 1."""
    pad_w, pad_w_after, pad_h, pad_h_after, pad_t, pad_t_after = conv._padding
    kernel_t = int(conv.weight.shape[2])
    if pad_t != CACHE_FRAMES or pad_t_after != 0 or kernel_t != CACHE_FRAMES + 1:
        raise UnsupportedVaeError(
            f"causal padding {conv._padding} with time kernel {kernel_t} is unsupported"
            f" (only {CACHE_FRAMES} frames at the start of time and kernel {CACHE_FRAMES + 1})"
        )
    if pad_w != pad_w_after or pad_h != pad_h_after:
        raise UnsupportedVaeError(
            f"the asymmetric spatial padding {conv._padding} cannot be folded into conv3d's attrs"
        )
    if tuple(conv.stride) != (1, 1, 1) or tuple(conv.dilation) != (1, 1, 1):
        raise UnsupportedVaeError(
            f"stride {conv.stride} / dilation {conv.dilation} are unsupported (only 1)"
        )


def causal_conv3d(
    conv: nn.Module, x: torch.Tensor, cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs a CausalConv3d with time kernel 3 as "cat the cache at the start of time → conv3d with
    time padding 0".

    `x` is `[Cin, T, H, W]`, `cache` the last 2 frames of the previous chunk's input
    `[Cin, 2, H, W]` (zeros for the first chunk). Returns the output `[Cout, T, H, W]` and the last
    2 frames of `cat(cache, x)` to hand to the next chunk (upstream's update rule for `cache_x`
    with T = 1 / T ≥ 2 merged into one).
    The spatial padding is folded into conv3d's attrs (IR's pad covers only the last dimension with
    constant 0 — decision 1).
    """
    _assert_causal(conv)
    expected = (x.shape[0], CACHE_FRAMES, x.shape[2], x.shape[3])
    if tuple(cache.shape) != expected:
        raise ValueError(
            f"cache shape {tuple(cache.shape)} does not match the input {tuple(x.shape)}"
            f" ({expected})"
        )
    pad_w, _, pad_h, _, _, _ = conv._padding
    joined = torch.cat([cache, x], dim=1)
    out = functional.conv3d(joined, conv.weight, conv.bias, padding=(0, pad_h, pad_w))
    return out, joined[:, -CACHE_FRAMES:]


def pointwise_conv3d(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Runs a 1×1×1 CausalConv3d without a cache (post-quant, shortcut) as an unbatched conv3d."""
    if tuple(conv.weight.shape[2:]) != (1, 1, 1) or any(conv._padding):
        raise UnsupportedVaeError(
            f"anything other than 1×1×1 (kernel {tuple(conv.weight.shape[2:])}, padding"
            f" {conv._padding}) cannot run without a cache"
        )
    return functional.conv3d(x, conv.weight, conv.bias)


def rms_norm(norm: nn.Module, x: torch.Tensor, dim: int) -> torch.Tensor:
    """The rank-4 version of `WanRMS_norm` (channel axis `dim` — 0 for `[C,T,H,W]`, 1 for
    `[T,C,H,W]`).

    gamma applies right-aligned in upstream's stored shape (`images=False`'s `(C,1,1,1)` lines up
    with `[C,T,H,W]`, `images=True`'s `(C,1,1)` with the last 3 axes of `[T,C,H,W]`). Upstream's
    `+ self.bias` (Python's `0.0` when `bias=False`) is dropped — an addition that adds only a node
    without changing the value, with no observable difference other than signed zeros (the same
    judgement as anima's VAE patch).
    """
    if isinstance(norm.bias, torch.Tensor):
        raise UnsupportedVaeError("an RMS_norm with a learned bias is unsupported")
    if not norm.channel_first:
        raise UnsupportedVaeError("an RMS_norm with channel_first=False is unsupported")
    if x.dtype != torch.float32:
        # Upstream's fp16 / bf16 branch (only normalize runs in f32) is not written — activations
        # are always f32.
        raise UnsupportedVaeError(f"the input of RMS_norm must be f32 ({x.dtype})")
    return l2_normalize(x, dim) * norm.scale * norm.gamma


# Third-party code notice. `attention_block` below is adapted from `WanAttentionBlock.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# It is verbatim except that the time axis is folded by a rank-4 permute instead of the rank-5
# permute / reshape, and the RMS norm is the rank-4 rewrite (`rms_norm`).
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def attention_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Runs mid's single-head attention (`WanAttentionBlock`) on `[C,T,H,W]` as is.

    Upstream's `rearrange(b c t h w → (b t) c h w)` is copied as the permute
    `[C,T,H,W] → [T,C,H,W]`, which puts the frames on the attention's batch axis (`[T,1,H·W,C]` —
    B = T, H = 1, D = C). A chunk's mid is one latent frame, so T = 1. The body is upstream's
    forward with only the entry into and exit from the time axis removed (the same shape as anima's
    `_attention_block_forward`).
    """
    identity = x
    channels, frames, height, width = x.shape
    hidden = rms_norm(block.norm, x.permute(1, 0, 2, 3), dim=1)
    qkv = block.to_qkv(hidden)
    qkv = qkv.reshape(frames, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
    query, key, value = qkv.chunk(3, dim=-1)
    attended = functional.scaled_dot_product_attention(query, key, value)
    attended = attended.squeeze(1).permute(0, 2, 1).reshape(frames, channels, height, width)
    return block.proj(attended).permute(1, 0, 2, 3) + identity


class _CacheCursor:
    """Hands out the caches in the decoder's forward order and collects the updated caches in the
    same order (upstream's `feat_idx`)."""

    def __init__(
        self, slots: Sequence[CacheSlot], cache: Mapping[str, torch.Tensor], *, first: bool
    ) -> None:
        expected = [slot.key for slot in graph_slots(slots, first=first)]
        if list(cache) != expected:
            raise ValueError(
                f"cache keys {list(cache)} differ from {expected} (the order is checked too)"
            )
        self._slots: Iterator[CacheSlot] = iter(slots)
        self._cache = cache
        self._first = first
        self._updated: list[torch.Tensor] = []

    def causal(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        slot = next(self._slots)
        if slot.time_conv:
            raise AssertionError(
                f"slot {slot.key} is a time_conv (the forward order is out of step with the table)"
            )
        return self._apply(slot, conv, x)

    def time_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
        """Runs `time_conv`. In the first chunk it does not run and returns None (upstream's `'Rep'`
        path)."""
        slot = next(self._slots)
        if not slot.time_conv:
            raise AssertionError(
                f"slot {slot.key} is not a time_conv (the forward order is out of step with the"
                " table)"
            )
        return None if self._first else self._apply(slot, conv, x)

    def finish(self) -> list[torch.Tensor]:
        """The updated caches (in the same order as the input). Fails loudly unless the table is
        used up."""
        rest = list(self._slots)
        if rest:
            raise AssertionError(
                f"forward did not use up the caches: {[slot.key for slot in rest]}"
            )
        return self._updated

    def _apply(self, slot: CacheSlot, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out, updated = causal_conv3d(conv, x, self._cache[slot.key])
        self._updated.append(updated)
        return out


# Third-party code notice. `residual_block` and `resample` below are adapted from
# `WanResidualBlock.forward` and `WanResample.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of operations is kept; the feat_cache bookkeeping is replaced by the normalized
# cache (`causal_conv3d`) and the rank-6 interleave by `interleave_frames`.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def residual_block(block: nn.Module, x: torch.Tensor, cursor: _CacheCursor) -> torch.Tensor:
    """The rank-4 version of `WanResidualBlock` (shortcut → norm1 → SiLU → conv1 → norm2 → SiLU →
    conv2 → add).

    dropout is the identity in eval (`nn.Dropout` is passed through — the same module as upstream
    is called).
    """
    shortcut = x if isinstance(block.conv_shortcut, nn.Identity) else None
    if shortcut is None:
        shortcut = pointwise_conv3d(block.conv_shortcut, x)
    hidden = block.nonlinearity(rms_norm(block.norm1, x, dim=0))
    hidden = cursor.causal(block.conv1, hidden)
    hidden = block.dropout(block.nonlinearity(rms_norm(block.norm2, hidden, dim=0)))
    hidden = cursor.causal(block.conv2, hidden)
    return hidden + shortcut


def resample(module: nn.Module, x: torch.Tensor, cursor: _CacheCursor) -> torch.Tensor:
    """The rank-4 version of `WanResample` (upsample3d / upsample2d).

    upsample3d is `time_conv` (next only) → time interleave → per-frame nearest ×2 + Conv2d 3×3. The
    per-frame 2D processing puts the frames on the batch axis with the permute
    `[C,T,H,W] → [T,C,H,W]` (op for op the same as upstream's `permute(0,2,1,3,4).reshape(b*t, …)`).
    """
    if module.mode == "upsample3d":
        doubled = cursor.time_conv(module.time_conv, x)
        if doubled is not None:
            x = interleave_frames(doubled)
    elif module.mode != "upsample2d":
        raise UnsupportedVaeError(
            f"resample mode that does not occur in the decoder: {module.mode}"
        )
    upsample, conv = module.resample
    frames = conv(nearest_exact_2x(upsample, x.permute(1, 0, 2, 3)))
    return frames.permute(1, 0, 2, 3)


# Third-party code notice. `avg_shortcut` and `residual_up_block` below are adapted from
# `DupUp3D.forward` and `WanResidualUpBlock.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of operations and the first-chunk slice are kept; the rank-8 view / permute is replaced
# by the rank-4 `dup_up_3d`, the defensive `x.clone()` is left out (it does not change values) and
# the feat_cache bookkeeping is replaced by the normalized cache (`causal_conv3d`).
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def avg_shortcut(module: nn.Module, x: torch.Tensor, *, first_chunk: bool) -> torch.Tensor:
    """The rank-4 version of Wan2.2's shortcut `DupUp3D` (`x` is `[Cin, T, H, W]`).

    The body is {@link _shared.vae_rank4.dup_up_3d}. In the first chunk the first `ft − 1` frames of
    time are dropped (upstream's `x[:, :, ft-1:]` — matching the residual path's upsample3d, which
    does not run `time_conv` in the first chunk and stays at one frame). Upstream's slice for ft = 1
    is the identity, so it is not written.
    """
    if int(module.in_channels) != x.shape[0]:
        raise UnsupportedVaeError(
            f"the shortcut's input channels {module.in_channels} do not match the input"
            f" {tuple(x.shape)}"
        )
    factor_t = int(module.factor_t)
    out = dup_up_3d(
        x,
        out_channels=int(module.out_channels),
        factor_t=factor_t,
        factor_s=int(module.factor_s),
    )
    return out[:, factor_t - 1 :] if first_chunk and factor_t > 1 else out


def residual_up_block(
    block: nn.Module, x: torch.Tensor, cursor: _CacheCursor, *, first: bool
) -> torch.Tensor:
    """The rank-4 version of `WanResidualUpBlock` (Res ×3 → upsampler → add the shortcut).

    The shortcut is built from the block's input and added to the residual path's output (the same
    order as upstream's `x + avg_shortcut(x_copy, first_chunk)`).
    """
    shortcut_input = x
    for resnet in block.resnets:
        x = residual_block(resnet, x, cursor)
    if block.upsampler is not None:
        x = resample(block.upsampler, x, cursor)
    if block.avg_shortcut is not None:
        x = x + avg_shortcut(block.avg_shortcut, shortcut_input, first_chunk=first)
    return x


# Third-party code notice. `WanVaeChunkDecoder.forward` below is adapted from
# `WanDecoder3d.forward` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# The order of the blocks is kept; the feat_cache list and the `first_chunk` flag are replaced by
# graph inputs / outputs and two graph variants.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
class WanVaeChunkDecoder(nn.Module):
    """One VAE decode chunk graph (`first=True` for the first chunk, `False` for the following
    ones).

    The input is one latent frame `[z, 1, h, w]` (z is 16 for 2.1 and 48 for 2.2; the
    denormalization `z·std + mean` is on the host — decision 2) and the cache dict (keys are
    {@link CacheSlot.key}, in table order). The output is the frames **before clamping** (first
    gives `[C, 1, 8h, 8w]`, next `[C, 4, 8h, 8w]` — C is 3 pixel channels for 2.1 and 12 channels
    in patchify space for 2.2) and the updated cache (in the same order as the input).
    `clamp(-1, 1)` (and 2.2's unpatchify) are on the host (after the tile blending — the same place
    as upstream's `tiled_decode`).

    The post-quant 1×1×1 conv is applied per frame inside the graph (upstream's non-tiled decode
    applies it once to all frames, but for a 1×1×1 conv that equals per frame — `tiled_decode`
    applies it per frame).
    """

    def __init__(self, vae: AutoencoderKLWan, *, first: bool) -> None:
        super().__init__()
        self.slots = cache_slots(vae)
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.first = first
        #: Wan2.2's up blocks (`WanResidualUpBlock` — {@link assert_supported} has checked the
        #: types).
        self.residual = bool(vae.config.is_residual)

    @property
    def cache_slots(self) -> tuple[CacheSlot, ...]:
        """The caches this graph has as inputs / outputs (input order = output order)."""
        return graph_slots(self.slots, first=self.first)

    def forward(
        self, latent: torch.Tensor, cache: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        if latent.dim() != 4 or latent.shape[1] != 1:
            raise ValueError(
                f"pass the latent as one unbatched frame [C, 1, h, w] ({tuple(latent.shape)})"
            )
        decoder = self.decoder
        cursor = _CacheCursor(self.slots, cache, first=self.first)
        x = pointwise_conv3d(self.post_quant_conv, latent)
        x = cursor.causal(decoder.conv_in, x)
        mid = decoder.mid_block
        x = residual_block(mid.resnets[0], x, cursor)
        x = attention_block(mid.attentions[0], x)
        x = residual_block(mid.resnets[1], x, cursor)
        for block in decoder.up_blocks:
            if self.residual:
                x = residual_up_block(block, x, cursor, first=self.first)
                continue
            for resnet in block.resnets:
                x = residual_block(resnet, x, cursor)
            if block.upsamplers is not None:
                x = resample(block.upsamplers[0], x, cursor)
        x = decoder.nonlinearity(rms_norm(decoder.norm_out, x, dim=0))
        x = cursor.causal(decoder.conv_out, x)
        return (x, *cursor.finish())


# ---- Reference and chunk loop (eager) -----------------------------------------------------


# Third-party code notice. The loop of `reference_decode_unclamped` below reproduces the chunk loop
# of `AutoencoderKLWan._decode` in
# huggingface/diffusers
# (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`).
# verbatim; the unpatchify branch and the final clamp after it are left out.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def reference_decode_unclamped(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """The chunk loop of upstream's non-tiled `_decode` itself (without the unpatchify and `clamp`
    after it).

    `latents` is `[1, z, F, h, w]` (denormalized). Returns `[1, C, 1 + 4(F−1), 8h, 8w]` (C is 3
    pixel channels for 2.1 and 12 channels in patchify space for 2.2 — the same space as the chunk
    graph's output). Copies the chunk loop of upstream's `AutoencoderKLWan._decode`
    (`autoencoder_kl_wan.py:1187-1217`) verbatim and stops before unpatchify (absent in 2.1) and
    `clamp` — clamping hides the differences of saturated elements, so the comparison is done before
    clamping (the tests pin that applying unpatchify and clamp makes it bit-identical to upstream's
    `_decode`).
    """
    assert_supported(vae)
    vae.clear_cache()
    hidden = vae.post_quant_conv(latents)
    frames: list[torch.Tensor] = []
    for index in range(latents.shape[2]):
        vae._conv_idx = [0]
        frames.append(
            vae.decoder(
                hidden[:, :, index : index + 1],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=index == 0,
            )
        )
    vae.clear_cache()
    return torch.cat(frames, dim=2)


def normalized_cache_decode(
    vae: AutoencoderKLWan, latents: torch.Tensor, tile: int
) -> torch.Tensor:
    """A chunk loop with **only the cache normalization** applied (upstream's decoder code as is,
    rank 5).

    Calls upstream's decoder unchanged and only changes the initial value of feat_cache to "2 zero
    frames" (the two `time_conv` ones are `None` in the first chunk = upstream's `'Rep'` path, and
    are replaced by zeros before the second chunk). When upstream receives a 2-frame cache it sets
    the time pad to 0 and runs on the cat alone, so if this is bit-identical to upstream's `_decode`
    then "the cache normalization does not change values" holds (the measurement of decision 2's
    derivation; the spatial pad stays explicit as upstream has it).
    """
    slots = cache_slots(vae)
    side = latents.shape[3]
    if latents.shape[4] != side or side != tile:
        raise ValueError(
            f"the latent's spatial size {tuple(latents.shape[3:])} is not a square of tile {tile}"
        )

    def zero(slot: CacheSlot) -> torch.Tensor:
        return torch.zeros(1, *slot.shape(tile))

    hidden = vae.post_quant_conv(latents)
    feat: list[torch.Tensor | str | None] = [
        None if slot.time_conv else zero(slot) for slot in slots
    ]
    frames: list[torch.Tensor] = []
    for index in range(latents.shape[2]):
        frames.append(
            vae.decoder(
                hidden[:, :, index : index + 1],
                feat_cache=feat,
                feat_idx=[0],
                first_chunk=index == 0,
            )
        )
        if index == 0:
            if any(feat[slot.index] != "Rep" for slot in slots if slot.time_conv):
                raise AssertionError("time_conv did not take the 'Rep' path in the first chunk")
            feat = [zero(slot) if slot.time_conv else feat[slot.index] for slot in slots]
        for slot in slots:
            cached = feat[slot.index]
            if not isinstance(cached, torch.Tensor) or cached.shape[2] != CACHE_FRAMES:
                raise AssertionError(
                    f"after chunk {index}, cache {slot.key} does not have 2 frames"
                )
    return torch.cat(frames, dim=2)


def chunk_decode(
    first: WanVaeChunkDecoder, following: WanVaeChunkDecoder, latents: torch.Tensor
) -> torch.Tensor:
    """Runs the two rewritten chunk graphs eagerly (the host's chunk loop for one tile).

    `latents` is unbatched `[z, F, h, w]`. The cache is recreated as zeros at the start of the tile;
    after first, only the two `time_conv` ones are added, still zero, before handing it to next
    (first does not touch them — decision 2). Returns `[C, 1 + 4(F−1), 8h, 8w]` before clamping
    (C is the same as the output of {@link WanVaeChunkDecoder}).
    """
    if not first.first or following.first:
        raise ValueError("pass a first=True graph as first and a first=False graph as following")
    tile = latents.shape[2]
    zeros = zero_caches(first.slots, tile, first=False)
    cache = {slot.key: zeros[slot.key] for slot in first.cache_slots}
    frames: list[torch.Tensor] = []
    for index in range(latents.shape[1]):
        graph = first if index == 0 else following
        latent = latents[:, index : index + 1]
        out, *updated = graph(latent, cache)
        frames.append(out)
        cache = dict(zeros)
        cache.update(
            (slot.key, value) for slot, value in zip(graph.cache_slots, updated, strict=True)
        )
    return torch.cat(frames, dim=1)
