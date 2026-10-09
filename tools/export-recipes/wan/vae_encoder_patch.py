"""Wan2.2 の動画 VAE encoder の **chunk 0**（画像 1 枚）を、空間長を記号のまま export できる
3 グラフへ書き直す（ADR 0121 決定 11・段 9a）。

I2V の条件の潜在は、画像 1 枚（時間 1 枚）を上流 `AutoencoderKLWan._encode`（diffusers 0.39.0
`autoencoder_kl_wan.py`）に通した chunk 0 だけで作られる（`time_conv` は走らず・cache は空）。
IR の次元式は「記号 1 つの一次式」なので、mid block の attention（系列長 h·w を 1 軸に作る）の
前後で 3 グラフに分ける:

- pre（{@link VaeEncoderPre}）: conv_in 〜 down block 4 本 〜 mid の resnets[0]。
  `[12,1,8h,8w]` → `[640,1,h,w]`。
- attn（{@link VaeEncoderAttention}）: mid の attention（系列形）。`[640,S]` → `[640,S]`。
- post（{@link VaeEncoderPost}）: mid の resnets[1]・norm_out・conv_out・quant_conv。
  `[640,1,h,w]` → `[48,1,h,w]`。

h, w は潜在の高さ・幅、S = h·w。`[640,1,h,w]` ↔ `[640,S]` はバイト列の読み替えだけ（同じ並び）。
pre の入力は patchify 後の画像（上流の `patchify` — チャネル = c·p² + 幅の副添字·p +
高さの副添字）、post の出力は quant_conv の出力の前半 = mu（上流の
`DiagonalGaussianDistribution.mode()`）。
資産は解像度から独立する（ADR 0038 §4）— 受理寸法の門はパイプラインの側に置く。

書き直しは 5 つで、どれも上流のモジュールの重みを読むだけ（クラス属性の差し替えはしない —
{@link wan.vae_patch} と同じ規律）:

1. **時間スライスへの畳み込み**（{@link LastSliceConv3d}）— 時間カーネル 3 の CausalConv3d は、
   chunk 0 では時間の先頭に 2 枚のゼロを詰めてから畳むので、実データに掛かるのは最後の時間
   スライス `weight[:, :, 2]` だけ。重みをその 1 枚（`[o,i,1,kh,kw]`）に畳み、時間 padding 0 の
   conv3d にする（ゼロ×重みの項が消えるだけ — 差は符号付きゼロと縮約の順だけ。GPU の同値は
   ADR 0118 段 1 の恒等門 ② が `gpu_conv3d_parity_test.ts` に常設）。時間の因果 pad も記号形の
   ゼロのテンソルも要らなくなる。
2. **AvgDown3D の chunk 0 の閉じた形**（{@link avg_down_shortcut}）— 上流は rank 8 の view /
   permute と `mean(dim=2)`。pin の構成（{@link SUPPORTED_SHORTCUTS}）では空間 2×2 の平均か恒等に
   畳める。時間倍率 2 のブロックは T = 1 の先頭にゼロを 1 枚詰めるので、出力チャネルの偶数番は
   全てゼロ（4 つのゼロの平均）・奇数番は元チャネルの空間 2×2 の平均になる。平均は `sum × 1/4`
   （`aten.mean.dim` は exporter の語彙に無い・1/4 は 2 の冪なので和の順が同じならビット一致）。
   space-to-depth は記号の積を作らない rank 4 の分解（`[C·H/2,2,W/2,2]` → `[C,4,H/2,W/2]`）。
3. **非対称の ZeroPad2d**（{@link pad_right_bottom}）— IR の pad は最終次元・定数 0 だけなので、
   W 側は pad、H 側は permute で最終軸へ回して pad する。
4. **mid の attention の系列形**（{@link VaeEncoderAttention}）— `[C,S]` を `[S,C]` の行に置き、
   RMS norm（images=True）と 1×1 conv（to_qkv / proj）を行の L2 正規化と linear で書く。
5. **mu の切り出し**（{@link VaeEncoderPost}）— quant_conv（1×1×1）の出力 `2·z` ch の前半 `z` ch。

MUST: 受けるのは Wan2.2 の encoder（`is_residual=True`・patch 2）だけ。構造が想定と違う config は
{@link assert_supported} が fail loudly で落とす（{@link wan.vae_patch.assert_supported} と
同じ流儀）。

MUST: diffusers は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
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

#: 受ける世代（上流 config の `(is_residual, patch_size)`）— Wan2.2 だけ。2.1 の encoder
#: （`WanResample` の down・残差の外側のショートカット無し）は I2V を持たないので書いていない。
SUPPORTED_GENERATION: tuple[bool, int] = (True, 2)

#: 時間カーネル 3 の CausalConv3d で、chunk 0 の実データに掛かる時間スライスの添字（先頭
#: `CACHE_FRAMES` 枚がゼロ詰め — 上流の `_padding` の時間の前側）。
LAST_SLICE = CACHE_FRAMES

#: 空間の down の倍率（ZeroPad2d((0,1,0,1)) + Conv2d stride 2・AvgDown3D の factor_s）。
DOWN_SCALE = 2

#: AvgDown3D の受ける構成の閉じた表: `(factor_t, factor_s, group_size)` → 閉じた形の名前。
#: - `(1, 2, 4)`: 空間 2×2 の平均（出力チャネル = 入力チャネル）。
#: - `(2, 2, 4)`: 時間の先頭のゼロ 1 枚と交互（出力 2c = 0・2c+1 = 入力 c の空間 2×2 の平均）。
#: - `(1, 1, 1)`: 恒等（出力 = 入力）。
#: 表に無い組（group 2 / 8 など）はチャネルの並びの前提を確かめていないので fail loudly。
SUPPORTED_SHORTCUTS: dict[tuple[int, int, int], str] = {
    (1, 2, 4): "mean2x2",
    (2, 2, 4): "zero-interleaved-mean2x2",
    (1, 1, 1): "identity",
}

#: chunk 0 で走る down の resample の mode（downsample3d の `time_conv` は chunk 0 では走らない —
#: 上流は最初の chunk で入力を cache に積むだけ）。
DOWN_MODES = frozenset({"downsample2d", "downsample3d"})


# ---- 構成の門 ----------------------------------------------------------------------


def assert_supported(vae: AutoencoderKLWan) -> None:
    """書き直しが前提とする encoder の構成かを確かめる（外れたら fail loudly）。

    見るのは書き直しが**構造として**仮定している点だけ: 世代・入口のチャネル（`3 × p²`）・down
    block の型と resnets の本数・各 block のショートカットが {@link SUPPORTED_SHORTCUTS} の表にあり
    残差の経路（down の有無と時間の倍率）と揃うこと・mid が Res → Attention → Res の 1 層で
    あること・出口（conv_out と quant_conv）が `2·z` ch であること。
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualDownBlock

    config = vae.config
    generation = (bool(config.is_residual), config.patch_size)
    if generation != SUPPORTED_GENERATION:
        raise UnsupportedVaeError(
            f"(is_residual, patch_size) = {generation} は未対応（受けるのは"
            f" {SUPPORTED_GENERATION} の Wan2.2 だけ）"
        )
    encoder = vae.encoder
    channels = SAMPLE_CHANNELS * int(config.patch_size) ** 2
    if int(encoder.conv_in.in_channels) != channels:
        raise UnsupportedVaeError(
            f"入口のチャネル {encoder.conv_in.in_channels} が 3 × patch_size² = {channels} でない"
        )
    for index, block in enumerate(encoder.down_blocks):
        if not isinstance(block, WanResidualDownBlock):
            raise UnsupportedVaeError(
                f"down block {index} の型 {type(block).__name__} は WanResidualDownBlock でない"
            )
        if len(block.resnets) != int(config.num_res_blocks):
            raise UnsupportedVaeError(
                f"down block {index} の resnets {len(block.resnets)} 本が config の"
                f" num_res_blocks {config.num_res_blocks} と違う"
            )
        _assert_shortcut_follows_the_downsampler(index, block)
    mid = encoder.mid_block
    if len(mid.attentions) != 1 or len(mid.resnets) != 2:
        raise UnsupportedVaeError("mid block は Res → Attention → Res の 1 層だけを書いている")
    moments = 2 * int(config.z_dim)
    if int(encoder.conv_out.out_channels) != moments:
        raise UnsupportedVaeError(
            f"conv_out の出口 {encoder.conv_out.out_channels} が 2 × z_dim = {moments} でない"
        )
    quant = vae.quant_conv
    if (int(quant.in_channels), int(quant.out_channels)) != (moments, moments):
        raise UnsupportedVaeError(
            f"quant_conv の {quant.in_channels} → {quant.out_channels} が {moments} → {moments}"
            " でない"
        )


def _assert_shortcut_follows_the_downsampler(index: int, block: nn.Module) -> None:
    """ショートカット（AvgDown3D）が表にあり、残差の経路の down と同じ倍率であること。

    空間の倍率は down の有無（ZeroPad2d + stride 2 の Conv2d）と、時間の倍率は downsample3d か
    どうかと揃っていなければならない — 揃わないと chunk 0 の残差とショートカットの形・並びが
    合わない（上流は加算の broadcast で黙って通しうる）。
    """
    shortcut, downsampler = block.avg_shortcut, block.downsampler
    key = (int(shortcut.factor_t), int(shortcut.factor_s), int(shortcut.group_size))
    if key not in SUPPORTED_SHORTCUTS:
        raise UnsupportedVaeError(
            f"down block {index} のショートカット (factor_t, factor_s, group) = {key} は未対応"
            f"（受けるのは {list(SUPPORTED_SHORTCUTS)} だけ）"
        )
    if downsampler is None:
        expected = (1, 1)
    elif downsampler.mode in DOWN_MODES:
        expected = (2 if downsampler.mode == "downsample3d" else 1, DOWN_SCALE)
    else:
        raise UnsupportedVaeError(
            f"down block {index} の resample mode {downsampler.mode} は encoder に現れない"
        )
    if key[:2] != expected:
        raise UnsupportedVaeError(
            f"down block {index} のショートカットの倍率 (時間 {key[0]}, 空間 {key[1]}) が"
            f" 残差の経路の {expected} と違う"
        )
    # チャネルの対応（閉じた形が仮定する out = in · factor_t）は検査しない — 上流の AvgDown3D は
    # group_size を `in · factor_t · factor_s² // out` から導くので、表の group（どれも
    # factor_s²）に載った時点で out = in · factor_t が従う。


# ---- 書き直しの部品（上流モジュールを読むだけ）------------------------------------


def _assert_foldable(conv: nn.Module) -> None:
    """chunk 0 で最後の時間スライスへ畳める CausalConv3d か（時間カーネル 3・時間の前側の因果
    padding 2・空間 padding 対称・stride / dilation 1）。"""
    pad_w, pad_w_after, pad_h, pad_h_after, pad_t, pad_t_after = conv._padding
    kernel_t = int(conv.weight.shape[2])
    if kernel_t != LAST_SLICE + 1 or pad_t != LAST_SLICE or pad_t_after != 0:
        raise UnsupportedVaeError(
            f"時間カーネル {kernel_t}・因果パディング {conv._padding} は畳めない"
            f"（時間の先頭に {LAST_SLICE} 枚・カーネル {LAST_SLICE + 1} だけ）"
        )
    if pad_w != pad_w_after or pad_h != pad_h_after:
        raise UnsupportedVaeError(f"空間の非対称 padding {conv._padding} は conv3d に畳めない")
    if tuple(conv.stride) != (1, 1, 1) or tuple(conv.dilation) != (1, 1, 1):
        raise UnsupportedVaeError(
            f"stride {conv.stride}・dilation {conv.dilation} は未対応（1 だけ）"
        )


class LastSliceConv3d(nn.Module):
    """時間カーネル 3 の CausalConv3d を chunk 0（時間 1 枚・先頭 2 枚はゼロ詰め）の形へ畳んだもの。

    重みは上流の `weight[:, :, 2:]`（`[o, i, 1, kh, kw]` — 写しを持つ）、bias は上流の写し。
    入力は unbatched の `[Cin, 1, H, W]`、空間 padding は conv3d の attrs（上流と同じ対称の値）。
    上流の `F.pad`（時間の前にゼロ 2 枚）→ conv3d と比べ、ゼロ × 重みの 2 スライス分の項が消える
    だけ（値の差は符号付きゼロと縮約の順だけ）。
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
            raise ValueError(f"入力は unbatched の 1 フレーム [C, 1, H, W]（{tuple(x.shape)}）")
        return functional.conv3d(x, self.weight, self.bias, padding=self.padding)


def pad_right_bottom(frames: torch.Tensor) -> torch.Tensor:
    """`nn.ZeroPad2d((0, 1, 0, 1))`（`[N, C, H, W]` の右と下に 0 を 1 つずつ）を最終次元の pad
    で書く。

    IR の pad は最終次元・定数 0 だけなので、W 側をそのまま pad し、H 側は permute で最終軸へ
    回して pad してから戻す（詰める値は上流と同じ +0）。
    """
    widened = functional.pad(frames, (0, 1))
    heightened = functional.pad(widened.permute(0, 1, 3, 2), (0, 1))
    return heightened.permute(0, 1, 3, 2)


def downsample(resample: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """down の resample（downsample2d / downsample3d）の chunk 0（`x` は `[C, 1, H, W]`）。

    chunk 0 では downsample3d の `time_conv` は走らない（上流は最初の chunk で入力を cache に
    積むだけ）ので、どちらの mode も「フレームごとの ZeroPad2d((0,1,0,1)) → Conv2d stride 2」に
    なる。フレームを batch に置く permute は上流の `permute(0,2,1,3,4).reshape(b*t, …)` と同じ。
    """
    if resample.mode not in DOWN_MODES:
        raise UnsupportedVaeError(f"encoder に現れない resample mode: {resample.mode}")
    pad, conv = resample.resample
    if tuple(pad.padding) != (0, 1, 0, 1):
        raise UnsupportedVaeError(f"down の ZeroPad2d {pad.padding} は (0, 1, 0, 1) でない")
    frames = pad_right_bottom(x.permute(1, 0, 2, 3))
    return conv(frames).permute(1, 0, 2, 3)


def _mean2x2(x: torch.Tensor) -> torch.Tensor:
    """`[C, 1, H, W]` の各チャネルの空間 2×2 の平均 `[C, 1, H/2, W/2]`（H, W は偶数）。

    上流 AvgDown3D のグループの並び（副添字 `sh·2 + sw`）のまま `sum` で縮約してから `× 1/4`
    （2 の冪なので `/4` とビット一致）。記号の積を 1 軸に作らない rank 4 の分解:
    `[C·H/2, 2(sh), W/2, 2(sw)]` → `[C·H/2, sh, sw, W/2]` → `[C, H/2, 4, W/2]` →
    `[C, 4, H/2, W/2]`。
    """
    channels, frames, height, width = x.shape
    if frames != 1:
        raise ValueError(f"chunk 0 の 1 フレームだけを受ける（{tuple(x.shape)}）")
    rows = x.reshape(channels * (height // DOWN_SCALE), DOWN_SCALE, width // DOWN_SCALE, DOWN_SCALE)
    grouped = rows.permute(0, 1, 3, 2).reshape(
        channels, height // DOWN_SCALE, DOWN_SCALE * DOWN_SCALE, width // DOWN_SCALE
    )
    summed = grouped.permute(0, 2, 1, 3).sum(dim=1)
    mean = summed * (1.0 / (DOWN_SCALE * DOWN_SCALE))
    return mean.reshape(channels, 1, height // DOWN_SCALE, width // DOWN_SCALE)


def avg_down_shortcut(module: nn.Module, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """`residual + AvgDown3D(x)` の chunk 0（T = 1）の閉じた形（`x` は block の入力 `[Cin,1,H,W]`・
    `residual` は残差の経路の出力 `[Cout,1,H',W']`）。

    - 恒等（`(1,1,1)`）: `residual + x`。
    - 空間 2×2 の平均（`(1,2,4)`）: `residual + mean2x2(x)`。
    - 時間の先頭にゼロ 1 枚（`(2,2,4)`）: 上流のチャネルの並び `c·8 + ft·4 + sh·2 + sw` を 4 個ずつ
      平均するので、出力 2c は時間のゼロの平均（= +0）・2c+1 は `mean2x2(x)[c]`。偶数番に +0 を
      足すのは恒等（符号付きゼロ −0 + +0 = +0 だけが違う）なので省き、`residual` を `[C,2,H',W']`
      に読み替えて奇数番にだけ足す（チャネル軸の slice / cat は具体の長さ 2 の軸）。
    """
    key = (int(module.factor_t), int(module.factor_s), int(module.group_size))
    form = SUPPORTED_SHORTCUTS.get(key)
    if form is None:
        raise UnsupportedVaeError(f"ショートカット {key} は未対応（{list(SUPPORTED_SHORTCUTS)}）")
    if int(module.in_channels) != x.shape[0]:
        raise UnsupportedVaeError(
            f"ショートカットの入力チャネル {module.in_channels} が入力 {tuple(x.shape)} と合わない"
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
    """`WanResidualBlock` の chunk 0（shortcut → norm1 → SiLU → conv1 → norm2 → SiLU → dropout →
    conv2 → 加算）。conv1 / conv2 は {@link LastSliceConv3d} に畳んだ写し、ほかは上流の
    モジュール。"""

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
    """`WanResidualDownBlock` の chunk 0（Res × n → down → ショートカットを加算）。"""

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
    """pre のグラフ: patchify 後の画像 `[12, 1, 8h, 8w]` → mid の resnets[0] の出力
    `[640, 1, h, w]`。

    down の resample は上流のモジュールを持つが、chunk 0 で走るのは `resample`（ZeroPad2d +
    Conv2d）だけ — downsample3d の `time_conv` の重みはグラフに現れない。
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
    """1×1 の Conv2d の重み `[o, i, 1, 1]` を `[o, i]` に畳んだ linear（系列形の行に掛ける）。"""

    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        if tuple(conv.kernel_size) != (1, 1) or tuple(conv.stride) != (1, 1):
            raise UnsupportedVaeError(
                "1×1・stride 1 の Conv2d だけを linear に畳める"
                f"（{conv.kernel_size}・{conv.stride}）"
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
    """attn のグラフ: mid の単一 head attention を系列形で（`[C, S]` → `[C, S]`・C = 640）。

    入力は pre の出力 `[C, 1, h, w]` のバイト列を `[C, S]`（S = h·w）と読んだもの。行 `[S, C]` に
    置いてから、RMS norm（images=True の gamma `(C,1,1)` を `[C]` に畳む）・to_qkv（linear）・
    SDPA（`[1, 1, S, C]` — B = 1・H = 1・D = C）・proj（linear）・残差の加算の順（上流と同じ）。
    """

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        assert_supported(vae)
        block = vae.encoder.mid_block.attentions[0]
        norm = block.norm
        if isinstance(norm.bias, torch.Tensor) or not norm.channel_first:
            raise UnsupportedVaeError("bias 付き・channel_first=False の RMS_norm は未対応")
        self.channels = int(block.dim)
        self.scale = float(norm.scale)
        self.gamma = nn.Parameter(
            norm.gamma.detach().reshape(self.channels).clone(), requires_grad=False
        )
        self.to_qkv = _Linear(block.to_qkv)
        self.proj = _Linear(block.proj)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2 or tokens.shape[0] != self.channels:
            raise ValueError(f"入力は系列形の [{self.channels}, S]（{tuple(tokens.shape)}）")
        rows = tokens.permute(1, 0)
        hidden = l2_normalize(rows, dim=1) * self.scale * self.gamma
        qkv = self.to_qkv(hidden).reshape(1, 1, rows.shape[0], 3 * self.channels)
        query, key, value = qkv.chunk(3, dim=-1)
        attended = functional.scaled_dot_product_attention(query, key, value)
        projected = self.proj(attended.reshape(rows.shape[0], self.channels))
        return projected.permute(1, 0) + tokens


class VaeEncoderPost(nn.Module):
    """post のグラフ: `[640, 1, h, w]` → mu `[z, 1, h, w]`（z = 48）。

    mid の resnets[1] → norm_out → SiLU → conv_out（畳んだ写し）→ quant_conv（1×1×1 — 上流は
    非タイルの `_encode` の最後に掛ける）→ 前半 z ch（上流の `DiagonalGaussianDistribution` の
    `torch.chunk(parameters, 2, dim=1)` の mean 側 = `mode()`）。logvar の後半はグラフに出さない。
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
    """3 グラフのモジュール（eval）。重みは構築の時点の値の写し（f16 の丸めはこの前に済ませる）。"""
    return (
        VaeEncoderPre(vae).eval(),
        VaeEncoderAttention(vae).eval(),
        VaeEncoderPost(vae).eval(),
    )


def as_sequence(hidden: torch.Tensor) -> torch.Tensor:
    """pre の出力 `[C, 1, h, w]` → attn の入力 `[C, S]`（同じバイト列の読み替え）。"""
    channels, frames, height, width = hidden.shape
    if frames != 1:
        raise ValueError(f"chunk 0 の 1 フレームだけを受ける（{tuple(hidden.shape)}）")
    return hidden.reshape(channels, height * width)


def encode_chunk0(
    graphs: tuple[VaeEncoderPre, VaeEncoderAttention, VaeEncoderPost], image: torch.Tensor
) -> torch.Tensor:
    """3 グラフを eager で繋ぐ（ホストの段の組み立てと同じ読み替え）。

    `image` は patchify 後の `[12, 1, 8h, 8w]`。戻りは mu `[z, 1, h, w]`。
    """
    pre, attention, post = graphs
    hidden = pre(image)
    channels, _, height, width = hidden.shape
    sequence = attention(as_sequence(hidden))
    return post(sequence.reshape(channels, 1, height, width))


# ---- 参照（上流そのもの）------------------------------------------------------------


def reference_mu(vae: AutoencoderKLWan, image: torch.Tensor) -> torch.Tensor:
    """上流の非タイル encode の mu（I2V の条件の潜在の正規化の前 — `[1, z, 1, h, w]`）。

    `image` は `[1, 3, 1, H, W]`（[-1, 1]・patchify の前 — 上流の `_encode` が patchify する）。
    I2V パイプラインと同じ `retrieve_latents(vae.encode(…), sample_mode="argmax")`
    （`pipeline_wan_i2v.py` — `latent_dist.mode()` = quant_conv の出力の前半）を呼ぶ。
    MUST: タイルの経路（`use_tiling` が立つと `_encode` は `tiled_encode` へ分岐する）では採らない —
    書き直しは非タイルの計算を写している。
    """
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

    assert_supported(vae)
    if vae.use_tiling:
        raise UnsupportedVaeError("参照は非タイルの _encode で採る（use_tiling が立っている）")
    if image.dim() != 5 or image.shape[:3] != (1, SAMPLE_CHANNELS, 1):
        raise ValueError(f"画像は [1, 3, 1, H, W] で渡す（{tuple(image.shape)}）")
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
    """条件の潜在の正規化 `(mu − mean) · f32(1 / f32(std))`（`mu` は `[1, z, F, h, w]`）。

    上流は `1.0 / torch.tensor(latents_std)`（f32 の割り算）を掛ける — f64 の `1/std` を f32 へ
    丸めた値とは 48 本のうち一部が違う。TS 側は `latents.ts` の既存の形（`f32(1 / f32(std))`）を
    共有する。
    """
    channels = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean).view(1, channels, 1, 1, 1).to(mu.dtype)
    inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, channels, 1, 1, 1).to(mu.dtype)
    return (mu - mean) * inverse_std
