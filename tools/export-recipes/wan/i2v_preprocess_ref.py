"""**Reference** for the image preprocessing of I2V in Wan2.2 TI2V (ADR 0121 decision 11, step 9a).

Builds the chain image (RGB8) → RGB8 at the output size → [-1, 1] → patchify → VAE encoder input
with the same Pillow / torch calls as upstream. It is the source of truth of the goldens
(`wan.export_vae_encoder` writes them) that pin the TS preprocessing (a copy of LANCZOS in 22-bit
fixed point, the size rounding, the crop) **bit-exactly**.

There are two ways to fit the aspect ratio (fit):

- `crop` (default — the official Wan2.2 i2v, `textimage2video.py:462-477`): `scale = max(ow/iw,
  oh/ih)` → the size with Python's `round` (.5 to even) → LANCZOS → center crop (origin
  `(rw − ow) // 2`).
- `stretch` (`VideoProcessor.preprocess` of diffusers' `WanImageToVideoPipeline`): LANCZOS straight
  to `(ow, oh)` (does not keep the aspect ratio).

[-1, 1] is the official `TF.to_tensor(img).sub_(0.5).div_(0.5)` (= `f32(2·f32(x/255) − 1)` —
doubling commutes with rounding, so it is bit-identical to diffusers' `2·x − 1`). patchify is
upstream diffusers' `patchify` (channel = c·p² + width sub-index·p + height sub-index — width
first).

Choosing the output size (the closest aspect ratio in the accepted set) is the job of the TS
pipeline; here the size is given explicitly (the goldens make the portrait 704×1280 from a landscape
image, given explicitly).

MUST: import Pillow / torchvision / diffusers inside functions (the `wan` group is not in the
default sync — `tests/test_optional_group_imports.py`). The values depend on the Pillow version (the
LANCZOS implementation), so the goldens record the version in their provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from PIL import Image

#: The ways to fit the aspect ratio (the two in the module doc).
Fit = Literal["crop", "stretch"]
FITS: tuple[Fit, ...] = ("crop", "stretch")

#: The spatial factor of the `VideoProcessor` of diffusers' I2V pipeline (`vae_scale_factor_spatial`
#: — the `scale_factor_spatial` of the Wan2.2 VAE). It does not affect the resize values (it is used
#: for the multiple-of-16 check).
VAE_SCALE_FACTOR_SPATIAL = 16


def load_rgb(path: Path) -> Image.Image:
    """Opens an image as RGB (`Image.open(...).convert("RGB")` of the official
    `generate.py:376`)."""
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
    """The official i2v preprocessing: LANCZOS at the covering scale → center crop (fit = `crop`).

    The size `round(iw · scale)` uses Python's `round` (.5 to even — it disagrees with JS
    `Math.round`). The crop origin is the floor `//`.
    """
    from PIL import Image

    iw, ih = image.width, image.height
    scale = max(width / iw, height / ih)
    resized = image.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (resized.width - width) // 2
    y1 = (resized.height - height) // 2
    cropped = resized.crop((x1, y1, x1 + width, y1 + height))
    if cropped.size != (width, height):
        raise AssertionError(f"the crop result {cropped.size} is not ({width}, {height})")
    return cropped


def stretch_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """The resize of diffusers' I2V preprocessing (fit = `stretch` — LANCZOS straight to
    `(ow, oh)`).

    Calls upstream `VideoProcessor.resize` as is (defaults `resize_mode="default"`,
    `resample="lanczos"`, `reducing_gap=None`) — pinning it with a copy would make the claim about
    the reference's provenance a tautology.
    """
    if width % VAE_SCALE_FACTOR_SPATIAL or height % VAE_SCALE_FACTOR_SPATIAL:
        raise ValueError(
            f"size {width}×{height} is not a multiple of {VAE_SCALE_FACTOR_SPATIAL} (the pipeline's"
            " check_inputs rejects it)"
        )
    from diffusers.video_processor import VideoProcessor

    processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR_SPATIAL)
    return processor.resize(image, height=height, width=width)


def preprocess(image: Image.Image, width: int, height: int, fit: Fit) -> Image.Image:
    """Makes the RGB image at the output size with the resize (and crop) of the given fit."""
    if fit == "crop":
        return crop_resize(image, width, height)
    if fit == "stretch":
        return stretch_resize(image, width, height)
    raise ValueError(f"unknown fit {fit!r} ({', '.join(FITS)})")


def rgb8(image: Image.Image) -> torch.Tensor:
    """The pixels of an RGB image, `[H, W, 3]` uint8 (row-major, 3 bytes per pixel — the layout of
    the TS `Rgb8Image`)."""
    import numpy as np

    if image.mode != "RGB":
        raise ValueError(f"only RGB images are accepted (mode {image.mode})")
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())


def to_signed_unit(image: Image.Image) -> torch.Tensor:
    """The official `TF.to_tensor(img).sub_(0.5).div_(0.5)` → `[1, 3, 1, H, W]` f32 (the input of
    the VAE encode)."""
    import torchvision.transforms.functional as tf

    return tf.to_tensor(image).sub_(0.5).div_(0.5).unsqueeze(1).unsqueeze(0)


def encoder_input(sample: torch.Tensor, patch_size: int) -> torch.Tensor:
    """`[1, 3, 1, H, W]` → upstream `patchify` → unbatched `[3·p², 1, H/p, W/p]` (the input of the
    pre graph)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    if sample.dim() != 5 or sample.shape[:3] != (1, 3, 1):
        raise ValueError(f"pass [1, 3, 1, H, W] (got {tuple(sample.shape)})")
    return patchify(sample, patch_size=patch_size)[0].contiguous()
