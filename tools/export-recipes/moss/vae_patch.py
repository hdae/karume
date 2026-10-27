"""The patch layer that turns the MOSS-Image VAE decoder (the FLUX.2 small decoder —
`AutoencoderKLFlux2` of the pinned diffusers) into three IR graphs with symbolic lengths (ADR NNNN
decision 8, plan step 3, plan question 17 a).

## The graphs

| Graph              | Module                    | Input → output                       | Symbols |
| ------------------ | ------------------------- | ------------------------------------ | ------- |
| `vae_decoder_pre`  | `MossVaeDecoderPre`       | `z [1,32,h,w]` → `[1,384,h,w]`       | h, w    |
| `vae_decoder_attn` | `MossVaeDecoderAttention` | `tokens [1,384,S]` → `[1,384,S]`     | S = h·w |
| `vae_decoder_post` | `MossVaeDecoderPost`      | `hidden [1,384,h,w]` → `[1,3,8h,8w]` | h, w    |

h, w are the height and width of the VAE latent (the decode input after unpatchify — 1/8 of the
image); `[1,384,h,w]` and `[1,384,S]` are the same bytes, so the host hands the tensors over by
reinterpretation (the Wan2.2 VAE encoder form — ADR 0121 addendum 2026-10-09). The split sits around
the mid attention because the IR writes one symbol per dimension and attention needs the sequence
`h·w` as one axis.

- pre: `post_quant_conv` (1×1, 32 → 32), `decoder.conv_in` (32 → 384) and the mid block's first
  resnet — `AutoencoderKLFlux2._decode`, `Decoder.forward` and `UNetMidBlock2D.forward` up to the
  attention.
- attn: the mid block's attention (single head, dim 384, GroupNorm 32) on the sequence form.
- post: the mid block's second resnet, the four `UpDecoderBlock2D` (three ×2 upsamples), then
  `conv_norm_out` (GroupNorm), `conv_act` (SiLU) and `conv_out` (96 → 3). The clamp to [-1, 1] is
  the host's (upstream clamps after `decode`, in the pipeline).

## What changes against upstream

Three things; everything else — `ResnetBlock2D`, `nn.GroupNorm` (a `group_norm` node of the
vocabulary), `nn.SiLU`, the convolutions, the linears and the attention's SDPA — is upstream's
module, called as it is.

1. **The split**: the statements of `_decode`, `Decoder.forward`, `UNetMidBlock2D.forward` and
   `UpDecoderBlock2D.forward` are called one by one across the three modules, in upstream's order.
2. **`Upsample2D.forward`** ({@link upsample_2d}): upstream calls
   `F.interpolate(scale_factor=2.0, mode="nearest")`, which the IR has no op for. Nearest ×2 maps
   output index o to `floor(o/2)`, nearest-exact to `floor((o+0.5)/2)`; for every integer o they are
   the same index, so it is the 2× repetition `_shared.vae_rank4.repeat_2x` — the reshape / expand
   form the runtime's fusion rule `upsample2x` recognises (data movement only, bit-identical).
3. **The mid attention on `[1,C,S]`** ({@link MossVaeDecoderAttention}): `AttnProcessor2_0`'s
   statements for a 4-D input, with the input already flat (upstream's `view(B,C,H·W)`) and the
   output left flat (upstream's `reshape(B,C,H,W)` of the transposed result becomes the transpose
   alone). The residual and the rescale are upstream's.

## Eager equivalence (the step's gate — `moss.export_vae`)

The three modules chained on the CPU in f32 ({@link decode_split}) are bit-identical to upstream's
untiled `vae.decode` ({@link decode_reference}) — same weights, same process.

MUST: the decoder's configuration is the one the rewrite assumes ({@link assert_supported} fails
loudly on any other — the `wan.vae_patch.assert_supported` discipline). No class attribute of
diffusers is patched.

MUST: diffusers is imported inside the functions (the `moss` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional

from _shared.vae_rank4 import UPSAMPLE_SCALE, repeat_2x

if TYPE_CHECKING:
    from diffusers import AutoencoderKLFlux2

#: The graph names (= the container graph names and the series part directories).
VAE_DECODER_PRE = "vae_decoder_pre"
VAE_DECODER_ATTN = "vae_decoder_attn"
VAE_DECODER_POST = "vae_decoder_post"
GRAPHS = (VAE_DECODER_PRE, VAE_DECODER_ATTN, VAE_DECODER_POST)

#: The graph inputs (the forward argument names become the IR input names).
INPUT_NAMES = {VAE_DECODER_PRE: "z", VAE_DECODER_ATTN: "tokens", VAE_DECODER_POST: "hidden"}

#: The `time_embedding_norm` values of the decoder's resnets (the mid block passes `"default"`, the
#: up blocks the decoder's `norm_type` `"group"`). Without a time embedding both run
#: `norm2` alone in `ResnetBlock2D.forward` (its `else` branch and its `"default"` branch with
#: `temb` None).
RESNET_TIME_NORMS = frozenset({"default", "group"})


class UnsupportedVaeError(ValueError):
    """The VAE is not the configuration this rewrite assumes."""


def _fail(condition: bool, why: str) -> None:
    if not condition:
        raise UnsupportedVaeError(why)


def assert_supported(vae: AutoencoderKLFlux2) -> None:
    """Checks the points the rewrite assumes structurally (fails loudly otherwise): untiled decode
    with `post_quant_conv`; the decoder with group norms, a mid block of resnet → attention →
    resnet without time embedding, `UpDecoderBlock2D` up blocks of plain `ResnetBlock2D`, and
    interpolating ×2 `Upsample2D` with a convolution; a single-head mid attention with GroupNorm,
    a residual connection, no rescale and `AttnProcessor2_0`."""
    from diffusers.models.attention_processor import AttnProcessor2_0
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
    from diffusers.models.upsampling import Upsample2D

    _fail(not vae.use_tiling, "the VAE has tiling enabled (upstream decodes untiled)")
    _fail(vae.post_quant_conv is not None, "the VAE has no post_quant_conv")
    decoder = vae.decoder
    _fail(isinstance(decoder.conv_norm_out, nn.GroupNorm), "conv_norm_out is not a GroupNorm")
    _fail(isinstance(decoder.conv_act, nn.SiLU), "conv_act is not SiLU")
    mid = decoder.mid_block
    _fail(len(mid.resnets) == 2 and len(mid.attentions) == 1, "the mid block is not res-attn-res")
    resnets = [*mid.resnets]
    for index, block in enumerate(decoder.up_blocks):
        _fail(type(block) is UpDecoderBlock2D, f"up block {index} is {type(block).__name__}")
        resnets += [*block.resnets]
        upsamplers = [] if block.upsamplers is None else [*block.upsamplers]
        for upsampler in upsamplers:
            _fail(type(upsampler) is Upsample2D, f"up block {index} upsampler {type(upsampler)}")
            _fail(
                upsampler.interpolate
                and upsampler.use_conv
                and not upsampler.use_conv_transpose
                and upsampler.norm is None
                and upsampler.name == "conv",
                f"up block {index}: an Upsample2D other than interpolate + conv",
            )
    for index, resnet in enumerate(resnets):
        _fail(type(resnet) is ResnetBlock2D, f"resnet {index} is {type(resnet).__name__}")
        _fail(
            resnet.upsample is None
            and resnet.downsample is None
            and resnet.time_emb_proj is None
            and resnet.time_embedding_norm in RESNET_TIME_NORMS,
            f"resnet {index}: up / down sampling or a time embedding",
        )
    attention = mid.attentions[0]
    _fail(attention.group_norm is not None, "the mid attention has no GroupNorm")
    _fail(attention.heads == 1, f"the mid attention has {attention.heads} heads")
    _fail(
        attention.spatial_norm is None
        and attention.norm_q is None
        and attention.norm_k is None
        and attention.residual_connection
        and attention.rescale_output_factor == 1.0,
        "the mid attention has a spatial norm, q / k norms, no residual or a rescale",
    )
    _fail(
        type(attention.processor) is AttnProcessor2_0,
        f"the mid attention's processor is {type(attention.processor).__name__}",
    )
    _fail(not decoder.gradient_checkpointing, "the decoder has gradient checkpointing on")


def spatial_scale(vae: AutoencoderKLFlux2) -> int:
    """Image pixels per VAE latent cell: 2 per ×2 upsample (three in the FLUX.2 decoder → 8)."""
    return UPSAMPLE_SCALE ** sum(
        0 if block.upsamplers is None else len(block.upsamplers) for block in vae.decoder.up_blocks
    )


def hidden_channels(vae: AutoencoderKLFlux2) -> int:
    """The mid block's width (`decoder_block_out_channels[-1]` — 384)."""
    return int(vae.decoder.conv_in.out_channels)


# Third-party code notice. `upsample_2d`, the three `forward` methods below and `decode_split` are
# adapted from diffusers 0.39.0 (`Upsample2D.forward` in `models/upsampling.py`,
# `AttnProcessor2_0.__call__` in `models/attention_processor.py`, `UNetMidBlock2D.forward` and
# `UpDecoderBlock2D.forward` in `models/unets/unet_2d_blocks.py`, `Decoder.forward` in
# `models/autoencoders/vae.py`, `AutoencoderKLFlux2._decode` in
# `models/autoencoders/autoencoder_kl_flux2.py`), licensed under the Apache License, Version 2.0.
# The statements are upstream's for a batch of one in f32, in upstream's order; what changed is
# listed in the module docstring.


def upsample_2d(upsampler: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`Upsample2D.forward` (interpolate + conv, `output_size=None`) with the nearest ×2 written as
    the 2× repetition (module docstring "What changes" 2). The f32 / batch-size branches of
    upstream are identities for a batch of one in f32."""
    if x.shape[1] != upsampler.channels:
        raise UnsupportedVaeError(f"upsample of {x.shape[1]} channels ({upsampler.channels})")
    return upsampler.conv(repeat_2x(x))


class MossVaeDecoderPre(nn.Module):
    """`vae_decoder_pre` — module docstring. Holds upstream's modules under upstream's names
    (tensor keys `post_quant_conv.…`, `decoder.conv_in.…`, `decoder.mid_block.resnets.0.…`)."""

    def __init__(self, vae: AutoencoderKLFlux2) -> None:
        super().__init__()
        assert_supported(vae)
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = _DecoderPart(conv_in=vae.decoder.conv_in, mid_resnet0=vae.decoder.mid_block)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        sample = self.post_quant_conv(z)
        sample = self.decoder.conv_in(sample)
        return self.decoder.mid_block.resnets[0](sample, None)


class MossVaeDecoderAttention(nn.Module):
    """`vae_decoder_attn` — `AttnProcessor2_0.__call__` on `[1,C,S]` (module docstring "What
    changes" 3). Tensor keys `decoder.mid_block.attentions.0.…`."""

    def __init__(self, vae: AutoencoderKLFlux2) -> None:
        super().__init__()
        assert_supported(vae)
        self.decoder = _DecoderPart(mid_attention=vae.decoder.mid_block)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn = self.decoder.mid_block.attentions[0]
        residual = tokens
        batch_size = tokens.shape[0]
        hidden_states = tokens.transpose(1, 2)
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2)
        hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class MossVaeDecoderPost(nn.Module):
    """`vae_decoder_post` — module docstring. Tensor keys `decoder.mid_block.resnets.1.…`,
    `decoder.up_blocks.…`, `decoder.conv_norm_out.…`, `decoder.conv_out.…`."""

    def __init__(self, vae: AutoencoderKLFlux2) -> None:
        super().__init__()
        assert_supported(vae)
        decoder = vae.decoder
        self.decoder = _DecoderPart(
            mid_resnet1=decoder.mid_block,
            up_blocks=decoder.up_blocks,
            conv_norm_out=decoder.conv_norm_out,
            conv_act=decoder.conv_act,
            conv_out=decoder.conv_out,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder
        sample = decoder.mid_block.resnets[1](hidden, None)
        for block in decoder.up_blocks:
            for resnet in block.resnets:
                sample = resnet(sample, temb=None)
            if block.upsamplers is not None:
                for upsampler in block.upsamplers:
                    sample = upsample_2d(upsampler, sample)
        sample = decoder.conv_norm_out(sample)
        sample = decoder.conv_act(sample)
        return decoder.conv_out(sample)


class _DecoderPart(nn.Module):
    """A `decoder` namespace holding only the upstream submodules one graph reads, under upstream's
    attribute names (so the tensor keys are upstream's state-dict keys). The mid block is held as a
    container of the pieces the graph reads: `mid_resnet0` / `mid_resnet1` keep `resnets.0` /
    `resnets.1`, `mid_attention` keeps `attentions.0`."""

    def __init__(
        self,
        *,
        conv_in: nn.Module | None = None,
        mid_resnet0: nn.Module | None = None,
        mid_attention: nn.Module | None = None,
        mid_resnet1: nn.Module | None = None,
        up_blocks: nn.Module | None = None,
        conv_norm_out: nn.Module | None = None,
        conv_act: nn.Module | None = None,
        conv_out: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if conv_in is not None:
            self.conv_in = conv_in
        mid = _MidPart()
        if mid_resnet0 is not None:
            mid.resnets = nn.ModuleList([mid_resnet0.resnets[0], nn.Identity()])
        if mid_resnet1 is not None:
            mid.resnets = nn.ModuleList([nn.Identity(), mid_resnet1.resnets[1]])
        if mid_attention is not None:
            mid.attentions = nn.ModuleList([mid_attention.attentions[0]])
        if any(part is not None for part in (mid_resnet0, mid_attention, mid_resnet1)):
            self.mid_block = mid
        if up_blocks is not None:
            self.up_blocks = up_blocks
        if conv_norm_out is not None:
            self.conv_norm_out = conv_norm_out
        if conv_act is not None:
            self.conv_act = conv_act
        if conv_out is not None:
            self.conv_out = conv_out


class _MidPart(nn.Module):
    """The mid block's pieces one graph reads (see {@link _DecoderPart})."""


def decoder_graphs(vae: AutoencoderKLFlux2) -> tuple[nn.Module, nn.Module, nn.Module]:
    """The three graph modules on `vae`'s weights (eval)."""
    return (
        MossVaeDecoderPre(vae).eval(),
        MossVaeDecoderAttention(vae).eval(),
        MossVaeDecoderPost(vae).eval(),
    )


def decode_split(graphs: tuple[nn.Module, nn.Module, nn.Module], z: torch.Tensor) -> torch.Tensor:
    """The three graphs chained as the host chains them: `[1,C,h,w]` ⇄ `[1,C,S]` by reshape (the
    bytes the host reinterprets)."""
    pre, attention, post = graphs
    hidden = pre(z)
    batch, channels, height, width = hidden.shape
    tokens = attention(hidden.reshape(batch, channels, height * width))
    return post(tokens.reshape(batch, channels, height, width))


def decode_reference(vae: AutoencoderKLFlux2, z: torch.Tensor) -> torch.Tensor:
    """Upstream's decode as `MOSSImagePipeline.__call__` runs it (untiled, before the clamp)."""
    assert_supported(vae)
    return vae.decode(z, return_dict=False)[0]


def describe(vae: AutoencoderKLFlux2) -> dict[str, Any]:
    """The decoder's widths and counts (the op inventory's denominators)."""
    decoder = vae.decoder
    norms = [m for m in decoder.modules() if isinstance(m, nn.GroupNorm)]
    convs = [m for m in decoder.modules() if isinstance(m, nn.Conv2d)]
    return {
        "hidden_channels": hidden_channels(vae),
        "up_blocks": [
            [int(block.resnets[0].in_channels), int(block.resnets[-1].out_channels)]
            for block in decoder.up_blocks
        ],
        "group_norms": len(norms),
        "groups": sorted({int(m.num_groups) for m in norms}),
        "conv2d": len(convs) + 1,
        "spatial_scale": spatial_scale(vae),
    }
