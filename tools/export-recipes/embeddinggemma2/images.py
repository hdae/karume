"""The synthetic images of the EmbeddingGemma 2 image path: the tower's goldens
({@link embeddinggemma2.export_vision}) and the whole-input reference vectors
({@link embeddinggemma2.multimodal_reference}) are taken on them.

Every image is computed from a formula (a seeded generator for the noise), so the bytes are the
same on every machine and a fixture can carry them as they are. They are neutral on purpose — a
gradient, flat shapes, noise and stripes; no person, character, text or brand. Do not add others.

Each covers one thing the resize rule does (`get_aspect_ratio_preserving_size` of upstream's
image processor at the snapshot's budget of 280 soft tokens):

| Image             | Width × height | Resized to | Soft tokens | What it covers                   |
| ----------------- | -------------- | ---------- | ----------- | -------------------------------- |
| `gradient-square` | 64 × 64        | 768 × 768  | 256         | a square (upscaled), smooth      |
| `shapes-wide`     | 128 × 64       | 1104 × 528 | 253         | a wide aspect, hard edges        |
| `noise-tall`      | 48 × 112       | 480 × 1200 | 250         | a tall aspect, noise             |
| `stripes-edge`    | 1190 × 17      | 6672 × 48  | 139         | the smallest count of the rule   |

The table's sizes and counts are pinned by the tests against upstream's own function. 139 is the
smallest count the rule produces: in exact arithmetic 1190 × 17 resizes to 96 × 6720 px (2 × 140
blocks), but the f64 `sqrt` lands both sides a hair under their multiples of 48 (1.9999999999999998
and 139.99999999999997), so both round down — 48 × 6672 px, 1 × 139 blocks. Without that rounding
the smallest is 140.

MUST: torch-free and transformers-free (numpy only) — the pure tests import it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

#: The seed of `noise-tall` (a fixed generator, so its bytes never move).
NOISE_SEED = 20261122


@dataclass(frozen=True)
class SyntheticImage:
    """One image: its name, its size and the formula that draws it."""

    name: str
    width: int
    height: int
    #: What the image covers (the module table).
    why: str
    draw: Callable[[int, int], np.ndarray]

    def rgb8(self) -> np.ndarray:
        """The pixels as row-major RGB8 `[height, width, 3]` uint8."""
        pixels = self.draw(self.width, self.height)
        if pixels.dtype != np.uint8 or pixels.shape != (self.height, self.width, 3):
            raise ValueError(
                f"{self.name}: the formula drew {pixels.dtype} {pixels.shape}, not uint8"
                f" {(self.height, self.width, 3)}"
            )
        return np.ascontiguousarray(pixels)


def _grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.arange(width), np.arange(height), indexing="xy")


def _gradient(width: int, height: int) -> np.ndarray:
    x, y = _grid(width, height)
    red = x * 255 // max(width - 1, 1)
    green = y * 255 // max(height - 1, 1)
    blue = 255 - (red + green) // 2
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _shapes(width: int, height: int) -> np.ndarray:
    x, y = _grid(width, height)
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[...] = (232, 228, 214)
    disc = (x - width // 4) ** 2 + (y - height // 2) ** 2 <= (height // 3) ** 2
    pixels[disc] = (40, 96, 168)
    block = (x >= width // 2) & (x < width - 8) & (y >= height // 4) & (y < height - 10)
    pixels[block] = (196, 72, 52)
    bar = (y >= height - 6) & (y < height - 3)
    pixels[bar] = (30, 30, 30)
    return pixels


def _noise(width: int, height: int) -> np.ndarray:
    generator = np.random.default_rng(NOISE_SEED)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _stripes(width: int, height: int) -> np.ndarray:
    x, y = _grid(width, height)
    band = (x // 7) % 3
    pixels = np.zeros((height, width, 3), dtype=np.int64)
    for channel in range(3):
        pixels[..., channel] = np.where(band == channel, 220, 40 + (x * 90) // width)
    pixels[..., 1] += y * 3
    return pixels.clip(0, 255).astype(np.uint8)


#: The images, in the order the goldens list them.
IMAGES: Mapping[str, SyntheticImage] = {
    image.name: image
    for image in (
        SyntheticImage("gradient-square", 64, 64, "a square (upscaled), smooth values", _gradient),
        SyntheticImage("shapes-wide", 128, 64, "a wide aspect, hard edges", _shapes),
        SyntheticImage("noise-tall", 48, 112, "a tall aspect, every pixel independent", _noise),
        SyntheticImage(
            "stripes-edge",
            1190,
            17,
            "the smallest soft-token count the resize rule produces (139, a rounding edge)",
            _stripes,
        ),
    )
}
