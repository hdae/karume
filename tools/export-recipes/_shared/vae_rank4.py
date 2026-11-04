"""Leaf pure functions shared by the rank-4 rewrites of VAE decoders (ADR 0065 decision 2; ADR
0118 acceptance 4).

The decoder rewrites of anima (QwenImage VAE; an image-only patch with T = 1) and wan (Wan2.1 /
2.2 VAE; chunk graphs) must express the same computations without adding new ops to the
vocabulary:

- L2 normalization along channels ({@link l2_normalize} — `linalg_vector_norm` stays out of the
  vocabulary; ADR 0017)
- nearest-exact ×2 upsampling ({@link nearest_exact_2x} — upsample stays out of the vocabulary)
- temporal interleaving of the `time_conv` output ({@link interleave_frames} — fits rank 6 into
  rank 4; wan is its only user today)
- Wan2.2's shortcut outside the residual, `DupUp3D` ({@link dup_up_3d} — fits rank 8 into rank 4;
  wan is its only user today; ADR 0121 decision 6)

All of them are pure functions that read neither module weights nor family configuration, so with
a copy in each of the two families one could end up fixed in only one. The single source of truth
is here.

What is **not lifted here**: the cache mechanism, the configuration check
(`wan.vae_patch.assert_supported`), and the forwards ported from upstream code (RMS_norm /
Resample / AttentionBlock). Each family's upstream classes and assumptions differ.

MUST: the rewrites are **equivalent**. Pure data movement ({@link nearest_exact_2x} /
{@link interleave_frames} / {@link dup_up_3d}) is bit-identical to upstream; the one whose
reduction order changes ({@link l2_normalize}) matches `F.normalize` within rounding
(`anima/tests/test_patch.py`, `wan/tests/test_vae_patch.py`, `wan/tests/test_dup_up_3d.py`).

The dependency direction is recipe → `_shared` only (this module does not import families). It
depends only on torch and does not read diffusers (of the upsampling module it receives, it looks
only at `mode` and `scale_factor`).
"""

from __future__ import annotations

import torch
from torch import nn

#: The default eps of `F.normalize` (the norm is clamped from below to this value before dividing).
NORM_EPS = 1e-12

#: The factor of the nearest-exact upsampling. It matches reshape / expand exactly **only for an
#: integer factor**.
UPSAMPLE_SCALE = 2


def l2_normalize(x: torch.Tensor, dim: int, eps: float = NORM_EPS) -> torch.Tensor:
    """An equivalent implementation of `F.normalize(x, dim=dim)` (`sum` → `sqrt` → `clamp_min` →
    divide).

    `sum` carries the reduction axis in its attrs, so the channel-wise L2 is applied to that axis
    **without a permute** (anima used to go permute → sum over the last dimension → permute, and
    those two accounted for 99% of the VAE decoder's uncoalesced strided traffic —
    docs/research/2026-08-04-vae-axis-reduce-recon.md §2).

    MUST: keep the original `clamp_min(eps) → divide` order. Replacing it with `+eps` changes the
    numeric semantics (the property of returning 0 for a zero input is lost).
    """
    norm = torch.sqrt(torch.sum(x * x, dim=dim)).clamp(min=eps).unsqueeze(dim)
    return x / norm


def nearest_exact_2x(upsample: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Express nearest-exact ×2 (`[B,C,H,W]`) with reshape / expand (data movement only, so
    bit-identical).

    nearest-exact maps output index o to `floor((o+0.5)/scale)`. With the integer scale 2 this is
    exactly a twofold duplication of each input element (a non-integer factor does not map the
    same, so it fails loudly). The `x.float()` / `type_as` of the upstream upsamplers
    (`QwenImageUpsample` / `WanUpsample`) are identities on f32 activations.

    MUST: check `mode` too. The replacement matches only nearest-exact; `nearest` is a different
    mapping of the output index (to `floor(o/scale)`), and bilinear and the like interpolate
    differently altogether.
    """
    if upsample.mode != "nearest-exact":
        raise NotImplementedError(
            f"upsampling other than nearest-exact is unsupported: {upsample.mode}"
        )
    raw = upsample.scale_factor
    scale = (
        (float(raw), float(raw))
        if isinstance(raw, (int, float))
        else tuple(float(value) for value in raw)
    )
    if scale != (float(UPSAMPLE_SCALE), float(UPSAMPLE_SCALE)):
        raise NotImplementedError(
            f"nearest-exact other than ×{UPSAMPLE_SCALE} is unsupported: {scale}"
        )
    return repeat_2x(x)


def repeat_2x(x: torch.Tensor) -> torch.Tensor:
    """Each spatial element repeated 2×2 (`[B,C,H,W]` → `[B,C,2H,2W]` — data movement only, in the
    reshape / expand form the runtime's fusion rule `upsample2x` recognises).

    The common body of nearest-exact ×2 ({@link nearest_exact_2x}) and nearest ×2 (output index o →
    `floor(o/2)`, the same index as `floor((o+0.5)/2)` for every integer o — `moss.vae_patch`).
    MUST: the caller checks the scale and the mode first (this function checks no mapping).
    """
    batch, channels, height, width = x.shape
    wide = x.reshape(batch * channels * height, width, 1)
    wide = wide.expand(batch * channels * height, width, UPSAMPLE_SCALE)
    wide = wide.reshape(batch * channels, height, UPSAMPLE_SCALE * width)
    tall = wide.reshape(batch * channels, height, 1, UPSAMPLE_SCALE * width)
    tall = tall.expand(batch * channels, height, UPSAMPLE_SCALE, UPSAMPLE_SCALE * width)
    return tall.reshape(batch, channels, UPSAMPLE_SCALE * height, UPSAMPLE_SCALE * width)


def interleave_frames(x: torch.Tensor) -> torch.Tensor:
    """Reorder the doubled-channel output `[2C,T,H,W]` of `time_conv` alternately along frames
    → `[C,2T,H,W]`.

    `out[c, 2t+i] = x[i·C+c, t]` (the same order as upstream's `reshape(b,2,c,t,h,w)` →
    `stack(dim=3)` → `reshape` — diffusers 0.39.0 `autoencoder_kl_wan.py:299-301`). The
    intermediates stay within rank 4: `[2,C,T,H·W]` → permute `(1,2,0,3)` → `[C,T,2,H·W]` →
    `[C,2T,H,W]`. Data movement only.
    """
    doubled, frames, height, width = x.shape
    if doubled % 2 != 0:
        raise ValueError(f"the time_conv output channels {doubled} are not a multiple of 2")
    channels = doubled // 2
    paired = x.reshape(2, channels, frames, height * width).permute(1, 2, 0, 3)
    return paired.reshape(channels, 2 * frames, height, width)


def dup_up_3d(x: torch.Tensor, *, out_channels: int, factor_t: int, factor_s: int) -> torch.Tensor:
    """The rank-4 version of `DupUp3D.forward` (channel duplication → rank-8 view / permute).

    `x` is the unbatched `[Cin, T, H, W]`; the result is `[Cout, T·ft, H·fs, W·fs]`. The order is
    upstream's: with `j = ((o·ft + a)·fs + b)·fs + d` and `rep = Cout·ft·fs² / Cin`,
    `out[o, t·ft + a, h·fs + b, w·fs + d] = x[⌊j / rep⌋, t, h, w]` (upstream `repeat_interleave` →
    `view(B, Cout, ft, fs, fs, T, H, W)` → `permute(0, 1, 5, 2, 6, 3, 7, 4)` — diffusers 0.39.0
    `autoencoder_kl_wan.py:106-125`). The intermediates stay within rank 4: duplicate
    `[Cin, 1, T·H·W]` → expand `[Cin, rep, T·H·W]` → depth-to-space into W, H, T in turn, starting
    from the lowest digit of the channel index j (each stage reshape → permute → reshape). Data
    movement only, so bit-identical to upstream.

    The slice that drops the first `ft − 1` frames of the first chunk (upstream's `first_chunk`)
    is not here — it belongs to the cache mechanism, so the wan side
    (`wan.vae_patch.avg_shortcut`) holds it.
    """
    channels, frames, height, width = x.shape
    factor = factor_t * factor_s * factor_s
    if (out_channels * factor) % channels != 0:
        raise ValueError(
            f"output channels {out_channels} × factor {factor} is not divisible by the input"
            f" channels {channels}"
        )
    repeats = out_channels * factor // channels
    volume = frames * height * width
    # Duplication (upstream's repeat_interleave(dim=1)): channel index j = c·rep + r.
    dup = x.reshape(channels, 1, volume).expand(channels, repeats, volume)
    # The W digit: j = m·fs + d → [m, T·H, W, d] → [m, T, H, W·fs] (m < Cout·ft·fs).
    rows = out_channels * factor_t * factor_s
    y = dup.reshape(rows, factor_s, frames * height, width).permute(0, 2, 3, 1)
    y = y.reshape(rows, frames, height, width * factor_s)
    # The H digit: m = n·fs + b → [n, T·H, b, W·fs] → [n, T, H·fs, W·fs] (n < Cout·ft).
    rows = out_channels * factor_t
    y = y.reshape(rows, factor_s, frames * height, width * factor_s).permute(0, 2, 1, 3)
    y = y.reshape(rows, frames, height * factor_s, width * factor_s)
    if factor_t == 1:
        return y
    # The T digit: n = o·ft + a → [o, T, a, H·fs·W·fs] → [o, T·ft, H·fs, W·fs].
    plane = height * factor_s * width * factor_s
    y = y.reshape(out_channels, factor_t, frames, plane).permute(0, 2, 1, 3)
    return y.reshape(out_channels, frames * factor_t, height * factor_s, width * factor_s)
