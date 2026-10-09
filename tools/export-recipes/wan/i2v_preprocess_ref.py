"""Wan2.2 TI2V の I2V の画像の前処理の**参照**（ADR 0121 決定 11・段 9a）。

画像 1 枚（RGB8）→ 出力寸法の RGB8 → [-1, 1] → patchify → VAE encoder の入力、の鎖を上流と同じ
Pillow / torch の呼び出しで作る。TS 側の前処理（LANCZOS の 22 bit 固定小数点の写し・寸法の丸め・
クロップ）を**ビット一致**で縛る golden の正本（`wan.export_vae_encoder` が書く）。

縦横の合わせ方（fit）は 2 つ:

- `crop`（既定 — 公式 Wan2.2 の i2v・`textimage2video.py:462-477`）: `scale = max(ow/iw, oh/ih)` →
  寸法は Python の `round`（.5 は偶数側）→ LANCZOS → 中央クロップ（起点 `(rw − ow) // 2`）。
- `stretch`（diffusers の `WanImageToVideoPipeline` の `VideoProcessor.preprocess`）: `(ow, oh)` へ
  直接 LANCZOS（縦横比を保たない）。

[-1, 1] は公式の `TF.to_tensor(img).sub_(0.5).div_(0.5)`（= `f32(2·f32(x/255) − 1)` — 2 倍は
丸めと可換なので diffusers の `2·x − 1` とビット一致する）。patchify は上流 diffusers の
`patchify`（チャネル = c·p² + 幅の副添字·p + 高さの副添字 — 幅が先）。

出力寸法の選び方（受理集合から縦横比の比で最も近いもの）は TS のパイプラインの責務で、ここは
寸法を明示して受ける（golden は縦長の 704×1280 を横長の画像に明示して作る）。

MUST: Pillow / torchvision / diffusers は関数の中で import する（`wan` グループは既定の sync に
入らない — `tests/test_optional_group_imports.py`）。値は Pillow の版で決まる（LANCZOS の
実装）ので、golden は版を来歴に記録する。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from PIL import Image

#: 縦横の合わせ方（モジュールの doc の 2 つ）。
Fit = Literal["crop", "stretch"]
FITS: tuple[Fit, ...] = ("crop", "stretch")

#: diffusers の I2V パイプラインの `VideoProcessor` の空間の倍率（`vae_scale_factor_spatial` —
#: Wan2.2 の VAE の `scale_factor_spatial`）。resize の値には効かない（16 の倍数の検査に使う）。
VAE_SCALE_FACTOR_SPATIAL = 16


def load_rgb(path: Path) -> Image.Image:
    """画像を RGB で開く（公式 `generate.py:376` の `Image.open(...).convert("RGB")`）。"""
    from PIL import Image

    with Image.open(path) as opened:
        return opened.convert("RGB")


# Third-party code notice. `crop_resize` below reproduces the image preprocessing of
# `WanTI2V.i2v` in Wan-Video/Wan2.2 (`wan/textimage2video.py`, commit
# 1ea34ff48f87168174e12956e200b1d908b1c5ff, lines 462-477) verbatim, except that the output size is
# given instead of chosen by `best_output_size`.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream
# copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
def crop_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """公式の i2v の前処理: 覆う側の倍率で LANCZOS → 中央クロップ（fit = `crop`）。

    寸法 `round(iw · scale)` は Python の `round`（.5 は偶数側 — JS の `Math.round` とは割れる）。
    クロップの起点は floor の `//`。
    """
    from PIL import Image

    iw, ih = image.width, image.height
    scale = max(width / iw, height / ih)
    resized = image.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (resized.width - width) // 2
    y1 = (resized.height - height) // 2
    cropped = resized.crop((x1, y1, x1 + width, y1 + height))
    if cropped.size != (width, height):
        raise AssertionError(f"クロップの結果 {cropped.size} が ({width}, {height}) でない")
    return cropped


def stretch_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """diffusers の I2V の前処理の resize（fit = `stretch` — `(ow, oh)` へ直接 LANCZOS）。

    上流の `VideoProcessor.resize`（`resize_mode="default"`・`resample="lanczos"`・
    `reducing_gap=None` の既定）をそのまま呼ぶ — 写しで縛ると参照の素性の主張が恒真になる。
    """
    from diffusers.video_processor import VideoProcessor

    if width % VAE_SCALE_FACTOR_SPATIAL or height % VAE_SCALE_FACTOR_SPATIAL:
        raise ValueError(
            f"寸法 {width}×{height} は {VAE_SCALE_FACTOR_SPATIAL} の倍数でない（パイプラインの"
            " check_inputs が拒む）"
        )
    processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR_SPATIAL)
    return processor.resize(image, height=height, width=width)


def preprocess(image: Image.Image, width: int, height: int, fit: Fit) -> Image.Image:
    """fit に応じた resize（とクロップ）で出力寸法の RGB 画像を作る。"""
    if fit == "crop":
        return crop_resize(image, width, height)
    if fit == "stretch":
        return stretch_resize(image, width, height)
    raise ValueError(f"fit {fit!r} は未知（{', '.join(FITS)}）")


def rgb8(image: Image.Image) -> torch.Tensor:
    """RGB 画像の画素 `[H, W, 3]` uint8（行優先・画素あたり 3 バイト — TS の `Rgb8Image` の
    並び）。"""
    import numpy as np

    if image.mode != "RGB":
        raise ValueError(f"RGB の画像だけを受ける（mode {image.mode}）")
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())


def to_signed_unit(image: Image.Image) -> torch.Tensor:
    """公式の `TF.to_tensor(img).sub_(0.5).div_(0.5)` → `[1, 3, 1, H, W]` f32（VAE の encode の
    入力）。"""
    import torchvision.transforms.functional as tf

    return tf.to_tensor(image).sub_(0.5).div_(0.5).unsqueeze(1).unsqueeze(0)


def encoder_input(sample: torch.Tensor, patch_size: int) -> torch.Tensor:
    """`[1, 3, 1, H, W]` → 上流 `patchify` → unbatched の `[3·p², 1, H/p, W/p]`（pre のグラフの
    入力）。"""
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    if sample.dim() != 5 or sample.shape[:3] != (1, 3, 1):
        raise ValueError(f"[1, 3, 1, H, W] で渡す（{tuple(sample.shape)}）")
    return patchify(sample, patch_size=patch_size)[0].contiguous()
