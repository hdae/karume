"""The layer that curates the four components of Anima (the diffusers version) for **IR quality**.

The targets are the Qwen3 text encoder / AnimaTextConditioner / CosmosDiT / VAE decoder (ADR 0016).
Replacements are attribute swaps on already-imported classes (monkeypatching) and wrappers; the
`diffusers` / `transformers` packages themselves are not touched.

**How the motivation differs from `patch_sbv2` (ADR 0016 / recon §5)**: patch_sbv2 existed to make
export possible at all (without removing the branches, `torch.export` fails on a data-dependent
guard), whereas here **all 4 of 4 export as they are** (measured in recon §2). The layer still
exists for the quality of the resulting IR, with three aims:

- **Add no new op** — CausalConv3d(T=1) becomes conv2d, nearest-exact ×2 becomes reshape/expand,
  channel L2 becomes a last-axis sum (conv3d / upsample / linalg_vector_norm stay out of the
  vocabulary)
- **Stay within rank ≤ 4** — drop the VAE's T=1 axis, move the patchify flatten ahead of proj
- **Bake no run-time knob into the graph** — promote the timestep embedding to a graph input, and
  fold the always-zero padding_mask into a constant channel

Patches / wrappers:

1. **Qwen3 wrapper** — does not pass `attention_mask`. The pipeline uses a single prompt
   (`padding="longest"`), so the mask is all ones and `causal & all_true == causal`; the additive
   bias is the same tensor and eager matches bit for bit.
2. **Conditioner wrapper** — does not pass the two all-ones masks (SDPA's additive bias is all
   zeros and `scores + 0` matches bit for bit). The trailing `min_sequence_length` padding and the
   output-mask multiply happen on the host (IR pad is last-axis only — the vocabulary is not
   widened).
3. **DiT wrapper** — promotes the timestep sin/cos embedding to a host-computed graph input, and
   folds `padding_mask` (always zero in the pipeline) into a zero constant channel at latent
   resolution. Input and output are rank4 with the T axis dropped.
4. **VAE decoder patch** — replaces the T=1 CausalConv3d with "a conv2d over the last temporal slice
   of the weight", removing rank5 entirely. RMS_norm / Resample / Upsample / AttentionBlock also
   become rank4 versions. `feat_cache` is never read at T=1, so passing one fails loudly.
5. **Lifting the RoPE buffer** — `inv_freq` is demoted to a plain attribute so that it becomes a
   leaf of constant folding.
6. **DiT token-shaped wrapper** (#21 phase T2) — an **additional series** that moves patchify /
   unpatchify / the rope tables to the host. The verbatim wrapper of 3 does not move a single line
   (the static series keeps its existing assets).

MUST: a patched module is **eager-equivalent** to the unpatched one (`anima/export.py --verify`
measures this with the real weights). A change that is not equivalent MUST NOT be placed here.

MUST: the VAE patch **replaces class attributes process-wide**, so the "pre-patch reference" can be
taken only once per process. {@link vae_patches_applied} answers whether it has been applied, and
the caller (`anima/export.py`) rejects an out-of-order call by failing loudly (guarding against a
vacuous pass = a false PASS — the same discipline as ADR 0013).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from _shared.vae_rank4 import l2_normalize, nearest_exact_2x
from karume.rope import assert_rope_lifted

#: Whether the VAE patch has been applied. Kept only to make the side effect of the process-wide
#: replacement visible (taking the "pre-patch reference" after the patch makes the equivalence check
#: vacuous — ADR 0013 / 0016).
_VAE_APPLIED = False


def vae_patches_applied() -> bool:
    """Whether this process has already applied the VAE decoder patch.

    The check with which the side that takes the reference values (the equivalence check) asserts
    "not applied yet".
    """
    return _VAE_APPLIED


# ---- ④ Patches that make the VAE decoder rank4 ------------------------------


def _causal_conv3d_to_conv2d(conv: nn.Conv3d) -> nn.Conv2d:
    """Build the `nn.Conv2d` equivalent to a T=1 CausalConv3d (the weight is its last temporal
    slice).

    Causal padding stacks `2·pad_t` zeros at the **start** of the time axis, so with T=1 and
    `2·pad_t == kt−1` the input's only frame lands exactly at the end of the window and the temporal
    convolution meets only the last slice of the weight. Spatial padding is folded into conv2d's
    padding argument (IR pad is last-axis only, so an explicit pad cannot stay in the graph —
    ADR 0017).

    MUST: the equivalence condition checks **both sides**. With `2·pad_t < kt−1` the window never
    reaches the input (output time length 0); with `2·pad_t > kt−1` the window also covers the
    leading zeros and **the output has two or more frames** — neither can be expressed as "one last
    slice of the weight". The temporal stride is limited to 1 as well (a form of the original that
    subsamples time does not appear in the decoder, and the last-slice equivalence would be a
    different claim).
    MUST: the weight slice is done by **replacing the module, not in the graph**. A `select` in FX
    would make the rank5 original weight (three times the storage) an initializer as it is.
    """
    pad_w, pad_h, pad_t2 = conv._padding[0], conv._padding[2], conv._padding[4]
    kernel_t, kernel_h, kernel_w = (int(size) for size in conv.weight.shape[2:])
    if pad_t2 != kernel_t - 1:
        raise NotImplementedError(
            f"因果パディングが等価条件を外れる（2·pad_t={pad_t2}, kt={kernel_t}"
            f" — 2·pad_t == kt−1 のみ）: T=1 等価が崩れる"
        )
    if int(conv.stride[0]) != 1:
        raise NotImplementedError(
            f"時間 stride={int(conv.stride[0])} の CausalConv3d は未対応"
            "（T=1 等価は stride 1 のみ）"
        )
    # Check directly that the output time length is exactly 1 (re-confirming what the two conditions
    # above imply).
    time_out = (1 + pad_t2 - kernel_t) // int(conv.stride[0]) + 1
    if time_out != 1:
        raise NotImplementedError(
            f"output time length {time_out} for a T=1 input — not conv2d-equivalent"
        )
    flat = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        (kernel_h, kernel_w),
        stride=(conv.stride[1], conv.stride[2]),
        padding=(pad_h, pad_w),
        bias=conv.bias is not None,
    )
    flat.weight = nn.Parameter(conv.weight.detach()[:, :, -1].clone())
    if conv.bias is not None:
        flat.bias = nn.Parameter(conv.bias.detach().clone())
    return flat


def _rms_norm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The rank4 version of `QwenImageRMS_norm`. gamma has been reshaped to `(1,C,1,1)` at patch
    time.

    The original's `+ self.bias` (Python's `0.0` when `bias=False`) is dropped — an addition that
    changes no value and only adds a node, with no observable difference except a signed zero. A
    configuration with a learned bias (none exists in the decoder) fails loudly.
    """
    if isinstance(self.bias, torch.Tensor):
        raise NotImplementedError("学習された bias 付きの RMS_norm は rank4 パッチの対象外")
    if not self.channel_first:
        raise NotImplementedError("RMS_norm with channel_first=False is outside the rank4 patch")
    return l2_normalize(x, dim=1) * self.scale * self.gamma


def _resample_forward(self: nn.Module, x: torch.Tensor, feat_cache=None, feat_idx=None):
    """The rank4 version of `QwenImageResample` (images only, T=1).

    The `time_conv` stage of `upsample3d` does not run in the original either when
    `feat_cache is None` (the equivalent of the first chunk at T=1). The `downsample` modes do not
    appear in the decoder.
    """
    if feat_cache is not None:
        raise NotImplementedError("Resample with feat_cache is outside the image-only patch")
    if self.mode not in ("none", "upsample2d", "upsample3d"):
        raise NotImplementedError(f"resample mode not present in the decoder: {self.mode}")
    return self.resample(x)


def _upsample_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Express `QwenImageUpsample`'s nearest-exact ×2 with reshape / expand (bit-identical — the
    source of truth is {@link _shared.vae_rank4.nearest_exact_2x}, shared with wan's chunk
    graph)."""
    return nearest_exact_2x(self, x)


def _attention_block_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The rank4 version of `QwenImageAttentionBlock` (the original minus moving the time axis in
    and out)."""
    identity = x
    batch_size, channels, height, width = x.shape
    normalized = self.norm(x)
    qkv = self.to_qkv(normalized)
    qkv = qkv.reshape(batch_size, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
    query, key, value = qkv.chunk(3, dim=-1)
    attended = functional.scaled_dot_product_attention(query, key, value)
    attended = attended.squeeze(1).permute(0, 2, 1).reshape(batch_size, channels, height, width)
    return self.proj(attended) + identity


def apply_vae_decoder_patch(vae: nn.Module) -> None:
    """Replace the VAE decode path with a rank4, T=1-only one (idempotent).

    Class-attribute replacement (**process-wide**) + reshaping of modules / parameters (per
    instance). gamma is stored as `(C,1,1)` / `(C,1,1,1)`, which breaks right-aligned broadcasting
    against a rank4 input (with `images=False` axis 0 collides between C and B) — it is aligned to
    `(1,C,1,1)` at patch time so that forward has no branch.
    """
    global _VAE_APPLIED
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageAttentionBlock,
        QwenImageCausalConv3d,
        QwenImageResample,
        QwenImageRMS_norm,
        QwenImageUpsample,
    )

    QwenImageRMS_norm.forward = _rms_norm_forward
    QwenImageResample.forward = _resample_forward
    QwenImageUpsample.forward = _upsample_forward
    QwenImageAttentionBlock.forward = _attention_block_forward
    _VAE_APPLIED = True

    # Replace only the decode path (the encoder is not exported, so it is left alone).
    slots = [(vae, "post_quant_conv")]
    slots += [
        (parent, name)
        for parent in vae.decoder.modules()
        for name, child in parent.named_children()
        if isinstance(child, QwenImageCausalConv3d)
    ]
    for parent, name in slots:
        child = getattr(parent, name)
        if isinstance(child, QwenImageCausalConv3d):
            setattr(parent, name, _causal_conv3d_to_conv2d(child))
    for module in vae.decoder.modules():
        if not isinstance(module, QwenImageRMS_norm):
            continue
        target = (1, module.gamma.numel(), 1, 1)
        if tuple(module.gamma.shape) != target:
            module.gamma = nn.Parameter(module.gamma.detach().reshape(target))


class AnimaVaeDecoder(nn.Module):
    """VAE decode of one image (T=1). Input and output are both rank4 with the T axis dropped.

    Equivalent to the `num_frame=1` path of `AutoencoderKLQwenImage._decode` (the inverse
    normalisation with `latents_mean` / `latents_std` is on the host — run-time knobs are not baked
    into the graph).
    """

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.vae.post_quant_conv(latents)
        return torch.clamp(self.vae.decoder(hidden), min=-1.0, max=1.0)


def reference_vae_decode(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """The reference output of diffusers' own decode **before the patch** (rank5, with feat_cache).

    It calls `vae.decode` as is, so `--verify` measures "conv3d → conv2d equivalence" and
    "feat_cache has no effect on the result at T=1" **in one comparison at once** (checking one of
    them through a separate path would leave room for the other's premise to break silently).
    """
    return vae.decode(latents.unsqueeze(2), return_dict=False)[0].squeeze(2)


# ---- ① Qwen3 text encoder -----------------------------------------------------


def disable_packed_sequence_detection() -> None:
    """Pin `transformers`' packed sequence detection to "nothing detected" (process-wide).

    MUST: this is a **process-wide replacement** (the same approach as the VAE patch). Without a
    mask, `Qwen3Model` looks for a packed sequence (several sentences concatenated, with the
    position returning to 0 midway) from `position_ids`. In eager it falls through to `None`
    (nothing detected) for the monotonic positions of one sentence, but during tracing that
    value-dependent early return cannot be taken, and an `index` that looks up `cumsum` by position
    and an `eq` mask stay in the graph. The exporter excludes "an index that looks up a symbolic
    axis with an index tensor" from constant folding (ADR 0010 addendum 2026-10-10), so export fails
    on out-of-vocabulary ops (measured 2026-10-17: `aten.index.Tensor` ×2, `aten.eq.Tensor`,
    `aten.ne.Scalar`). This recipe's text encoder takes a single prompt with `padding="longest"`
    (positions increase monotonically over 0..T−1), so the detection result is always None, as in
    eager — it is pinned to that value.
    """
    import transformers.masking_utils as masking_utils

    def no_packed_sequences(position_ids: torch.Tensor) -> None:
        del position_ids
        return None

    masking_utils.find_packed_sequence_indices = no_packed_sequences


class AnimaTextEncoder(nn.Module):
    """The `last_hidden_state` of `Qwen3Model`. `attention_mask` is not passed.

    The pipeline uses a single prompt with `padding="longest"`, so the mask is all ones and
    `causal & all_true == causal`. The additive bias is the same tensor, so eager matches bit for
    bit (`--verify text_encoder` measures it). Apply {@link disable_packed_sequence_detection}
    before export (a premise of not passing the mask).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        assert_rope_lifted(model, "text_encoder")
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, use_cache=False).last_hidden_state


# ---- ② Text conditioner -------------------------------------------------------


class AnimaConditioner(nn.Module):
    """The body of `AnimaTextConditioner` (no mask, before the 512 padding).

    With all-ones masks SDPA's additive bias is all zeros, so it matches the no-mask form bit for
    bit. The trailing `min_sequence_length` padding and the output-mask multiply are **on the host**
    (IR pad is last-axis only, and multi-axis pad is not added to the vocabulary — ADR 0016).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        assert_rope_lifted(model, "text_conditioner")
        self.model = model

    def forward(
        self, source_hidden_states: torch.Tensor, target_input_ids: torch.Tensor
    ) -> torch.Tensor:
        model = self.model
        hidden_states = model.embed(target_input_ids).to(dtype=source_hidden_states.dtype)
        hidden_states = model.in_proj(hidden_states)
        position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
        source_position_ids = torch.arange(source_hidden_states.shape[1]).unsqueeze(0)
        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        source_position_embeddings = model.rotary_emb(hidden_states, source_position_ids)
        for block in model.blocks:
            hidden_states = block(
                hidden_states,
                source_hidden_states,
                position_embeddings=position_embeddings,
                source_position_embeddings=source_position_embeddings,
            )
        return model.norm(model.out_proj(hidden_states))


def reference_conditioner(
    model: nn.Module, source_hidden_states: torch.Tensor, target_input_ids: torch.Tensor
) -> torch.Tensor:
    """The leading rows, **before** the 512 padding, of the original path given all-ones masks (the
    reference values)."""
    out = model(
        source_hidden_states,
        target_input_ids,
        torch.ones(1, target_input_ids.shape[1], dtype=torch.long),
        torch.ones(1, source_hidden_states.shape[1], dtype=torch.long),
    )
    return out[:, : target_input_ids.shape[1]]


# ---- ③ CosmosDiT -----------------------------------------------------------


class AnimaDit(nn.Module):
    """One denoise step of `CosmosTransformer3DModel` (one side of cond / uncond).

    It differs from the original forward in three points only:

    - the timestep → sin/cos embedding is computed on the host, and `timesteps_proj` is an input
      (run-time knobs and trigonometry are not baked into the graph)
    - `padding_mask` (`latents.new_zeros` in the pipeline) is folded into a zero constant channel at
      latent resolution (a nearest resize is the identity on a zero input)
    - the T axis (= 1) of input and output is dropped, and the patchify flatten is moved ahead of
      proj (view reordering only; the normalisation pass handles the reduction to rank ≤ 4)

    Everything else (rope / transformer_blocks / norm_out / unpatchify) is the original as is.
    """

    def __init__(self, model: nn.Module, latent_height: int, latent_width: int) -> None:
        super().__init__()
        if model.config.extra_pos_embed_type is not None:
            raise NotImplementedError(
                "configurations with a learnable pos embed are not supported (Anima has null)"
            )
        if model.config.img_context_dim_in or model.config.use_crossattn_projection:
            raise NotImplementedError(
                "configurations with img_context / crossattn_projection are not supported"
            )
        if not model.config.concat_padding_mask:
            raise NotImplementedError(
                "configurations with concat_padding_mask=False are not supported"
            )
        self.model = model
        # A plain attribute (not a parameter / buffer) — carried as a lifted tensor constant.
        self.padding_channel = torch.zeros(1, 1, latent_height, latent_width)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        patch_t, patch_h, patch_w = model.config.patch_size
        hidden_states = torch.cat([latents, self.padding_channel], dim=1).unsqueeze(2)
        batch, channels, frames, height, width = hidden_states.shape

        image_rotary_emb = model.rope(hidden_states, fps=None)

        patches = hidden_states.reshape(
            batch,
            channels,
            frames // patch_t,
            patch_t,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        hidden_states = model.patch_embed.proj(patches)

        temb = model.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = model.time_embed.norm(timesteps_proj)

        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                None,
                None,
                None,
            )

        hidden_states = model.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = model.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (patch_h, patch_w, patch_t, -1))
        hidden_states = hidden_states.unflatten(
            1, (frames // patch_t, height // patch_h, width // patch_w)
        )
        # NOTE: as the original's comment says, this permutation is not the reverse of patchify, but
        # it is correct.
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return hidden_states.squeeze(2)


class AnimaDitTokens(nn.Module):
    """The **token-shaped** form of `AnimaDit` (an S-shaped graph without a resolution — #21 phase
    T2).

    It differs from {@link AnimaDit} only at the entry and the exit; the 28-block body does not
    change by a single node:

    - the input is **`tokens [1,S,68]`** instead of `latents [1,16,H,W]` (`68 = 17·2·2` — the
      always-zero padding channel appended and then patchified, both on the host =
      {@link dit_patchify})
    - the rope cos / sin tables are **promoted to graph inputs** (the same move as SBV2's relative
      position tables — ADR 0013). This removes the constant folding of `model.rope`, and **not one
      S-dependent initializer remains**
    - the output is the tokens `[1,S,64]` **before** unpatchify (the inverse reordering is on the
      host too = {@link dit_unpatchify})

    As a result no H / W appears in the graph, and it can be written with linear expressions of the
    single token-length symbol `S` only (the dimension language is "a linear expression of one
    symbol per dimension" — docs/ir-v2.md).

    MUST: the rope tables are taken as `[1,1,S,head_dim]` and brought back to 2-D with
    `flatten(0, 2)`. Upstream's `apply_rotary_emb` adds axes with `cos[None,None]` on the premise
    that **the table it is given is 2-D**, so passing it as 4-D would inflate it to rank 6 and
    change what the broadcast means. The input is declared as `[1,1,S,head_dim]` so that the
    constants that were baked into the static graph (the same shape) can be compared byte for byte
    with the host implementation as they are.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if model.config.extra_pos_embed_type is not None:
            raise NotImplementedError(
                "configurations with a learnable pos embed are not supported (Anima has null)"
            )
        if model.config.img_context_dim_in or model.config.use_crossattn_projection:
            raise NotImplementedError(
                "configurations with img_context / crossattn_projection are not supported"
            )
        if not model.config.concat_padding_mask:
            raise NotImplementedError(
                "configurations with concat_padding_mask=False are not supported"
            )
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        hidden_states = model.patch_embed.proj(tokens)
        image_rotary_emb = (rope_cos.flatten(0, 2), rope_sin.flatten(0, 2))

        temb = model.time_embed.t_embedder(timesteps_proj)
        embedded_timestep = model.time_embed.norm(timesteps_proj)

        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                None,
                None,
                None,
            )

        hidden_states = model.norm_out(hidden_states, embedded_timestep, temb)
        return model.proj_out(hidden_states)


def dit_patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """`latents [1,C,H,W]` → `tokens [1,S,(C+1)·pt·ph·pw]` (the entry moved to the host).

    MUST: this is **the same reordering, verbatim,** as the entry of {@link AnimaDit.forward}
    (concatenating the padding channel + patchify). If it is off by even one axis, the S-shaped
    graph's output differs from the static graph's (the shapes match, so a shape check does not
    catch it). The TS mirror (`patchifyLatents` / `unpatchifyTokens`) is covered for the reordering
    alone by `packages/models/tests/anima_dit_tokens_test.ts`, and the end-to-end numbers by the PNG
    sha256 of `packages/e2e/tests/e2e_anima_test.ts` on a real GPU.
    """
    patch_t, patch_h, patch_w = patch_size
    batch, _, height, width = latents.shape
    padding_channel = latents.new_zeros(batch, 1, height, width)
    hidden_states = torch.cat([latents, padding_channel], dim=1).unsqueeze(2)
    batch, channels, frames, height, width = hidden_states.shape
    patches = hidden_states.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return patches.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


def dit_unpatchify(
    tokens: torch.Tensor,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """`tokens [1,S,pt·ph·pw·C]` → `latents [1,C,H,W]` (the exit moved to the host).

    MUST: verbatim the same as the exit of {@link AnimaDit.forward}. It is **not the reverse of
    patchify** (the order of the last axis is `(ph, pw, pt, C)` here against `(C, pt, ph, pw)` on
    the patchify side) — as the original's comment says; "fixing" it into the reverse breaks it
    silently.
    """
    patch_t, patch_h, patch_w = patch_size
    hidden_states = tokens.unflatten(2, (patch_h, patch_w, patch_t, -1))
    hidden_states = hidden_states.unflatten(
        1, (1, latent_height // patch_h, latent_width // patch_w)
    )
    hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return hidden_states.squeeze(2)


def dit_rope_tables(
    model: nn.Module, latent_height: int, latent_width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rope cos / sin tables `[1,1,S,head_dim]` passed to the S-shaped graph.

    The source of truth of the formula is **upstream's `CosmosRotaryPosEmbed` itself** (only called,
    never copied). The two leading axes are the ones `apply_rotary_emb` adds to a 2-D table, and
    match the shape of the constants that were baked into the static graph.
    """
    probe = torch.zeros(1, 1, 1, latent_height, latent_width)
    cos, sin = model.rope(probe, fps=None)
    return cos[None, None], sin[None, None]


#: Tensor keys of the host base tables (cos / sin per axis). For an image (F'=1) only one row of `t`
#: is used.
ROPE_BASE_KEYS = ("cos_t", "sin_t", "cos_h", "sin_h", "cos_w", "sin_w")


def dit_rope_base_tables(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract the **per-axis base tables** from which the host (TS) builds the rope tables, from
    the upstream formula.

    ## Why the tables are baked (why TS does not recompute the trigonometry)

    The f32 implementation of `torch.cos` / `torch.sin` **can be 1 ulp off the correctly rounded
    value** (measured: of all 8,192 position × frequency combinations, 472 for cos / 231 for sin
    differ by 1 ulp; e.g. `cos(1.0)` is 0x3f0a5141 in torch against the correctly rounded
    0x3f0a5140). JS's `Math.cos` rounds correctly in f64, so copying the formula into TS **always
    produces this difference**. As long as the static graph has torch's values baked in, **bit
    identity** with the S-shaped form can only hold by baking the tables (the same judgement as
    baking the tokenizer's Unicode classes into tables instead of reimplementing them in TS).

    ## Why the tables do not depend on the resolution

    Position → angle is independent per axis, and `freqs = cat([emb_t, emb_h, emb_w] * 2, dim=-1)`
    only lays out the per-axis blocks side by side. So the "axis × position" base tables are enough
    to build the table for any H' / W'. The row count is the length of upstream's
    `seq = arange(max(max_size))` = the **implementation ceiling** of the position table itself, 128
    for Anima (= latent 256 = 2048px). An H' / W' beyond it fails upstream as well, because the
    shape of `repeat` does not match.

    MUST: extract **from the output** of `model.rope` (do not copy the formula). A call with
    H'=rows / W'=1 pins the w block to position 0, so only the h block moves with the position.
    """
    rope = model.rope
    rows = max(int(size) for size in rope.max_size)
    widths = (rope.dim_t // 2, rope.dim_h // 2, rope.dim_w // 2)
    patch_t, patch_h, patch_w = rope.patch_size

    def probe(frames: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Call upstream's rope as is, with a shape that gives `pe_size = [frames, height,
        width]`."""
        sample = torch.zeros(1, 1, frames * patch_t, height * patch_h, width * patch_w)
        return rope(sample, fps=None)

    # One call per axis in which "only that axis moves, the other two are pinned to position 0".
    cos_t, sin_t = probe(rows, 1, 1)
    cos_h, sin_h = probe(1, rows, 1)
    cos_w, sin_w = probe(1, 1, rows)

    start_h = widths[0]
    start_w = widths[0] + widths[1]
    end_w = start_w + widths[2]
    tables = {
        "cos_t": cos_t[:, : widths[0]].contiguous(),
        "sin_t": sin_t[:, : widths[0]].contiguous(),
        "cos_h": cos_h[:, start_h:start_w].contiguous(),
        "sin_h": sin_h[:, start_h:start_w].contiguous(),
        "cos_w": cos_w[:, start_w:end_w].contiguous(),
        "sin_w": sin_w[:, start_w:end_w].contiguous(),
    }
    _assert_rope_base_shapes(tables, rows, widths)
    return tables


def _assert_rope_base_shapes(
    tables: dict[str, torch.Tensor], rows: int, widths: tuple[int, int, int]
) -> None:
    """Check that the extraction offsets of the base tables match upstream's block boundaries (the
    guard against a vacuous pass).

    An extraction one block off still has the right shape (especially in configurations whose h and
    w blocks are equally wide). So it checks that **only that axis moves with the position**: the
    row at position 0 has angle 0, so cos=1 / sin=0, and the row at position 1 moves away from it
    (unless the frequency is 0). Slicing another block would stay constant over the position and
    break the latter.
    """
    for axis, width in zip(("t", "h", "w"), widths, strict=True):
        for kind in ("cos", "sin"):
            table = tables[f"{kind}_{axis}"]
            if tuple(table.shape) != (rows, width):
                raise AssertionError(
                    f"rope base table {kind}_{axis} has shape {tuple(table.shape)},"
                    f" expected {(rows, width)}"
                )
        if not torch.equal(tables[f"cos_{axis}"][0], torch.ones(width)):
            raise AssertionError(
                f"rope base table cos_{axis} is not all ones at position 0 (misaligned slice)"
            )
        if not torch.equal(tables[f"sin_{axis}"][0], torch.zeros(width)):
            raise AssertionError(
                f"rope base table sin_{axis} is not all zeros at position 0 (misaligned slice)"
            )
        if torch.equal(tables[f"sin_{axis}"][1], torch.zeros(width)):
            raise AssertionError(
                f"rope 素表 sin_{axis} が位置 1 でも全 0（この軸で動いていない = 切り出しずれ）"
            )


def dit_timesteps_proj(model: nn.Module, timestep: torch.Tensor) -> torch.Tensor:
    """The timestep embedding computed on the host (the `time_proj` stage of `CosmosEmbedding`)."""
    return model.time_embed.time_proj(timestep).to(torch.float32)


def reference_dit(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """The reference output on diffusers' path **before the patch** (`padding_mask` is zero, as in
    the pipeline)."""
    pixels = latents.shape[-2] * 8
    return model(
        hidden_states=latents.unsqueeze(2),
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        padding_mask=latents.new_zeros(1, 1, pixels, pixels),
        return_dict=False,
    )[0].squeeze(2)
