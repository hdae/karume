"""The synthetic images of the image path (`embeddinggemma2/images.py`).

Pinned here: each image is deterministic RGB8 of its stated size, and the module table's resized
sizes and soft-token counts are what upstream's resize rule gives — `stripes-edge` included, at the
smallest count the rule can produce at the 280-token budget (a scan over aspect ratios shows that
count is 139, a rounding edge of the rule). The cases that need upstream's function SKIP without
transformers at the pinned release.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np
import pytest

from embeddinggemma2 import export
from embeddinggemma2.images import IMAGES

#: The module table: name → (resized width, resized height, soft tokens).
TABLE = {
    "gradient-square": (768, 768, 256),
    "shapes-wide": (1104, 528, 253),
    "noise-tall": (480, 1200, 250),
    "stripes-edge": (6672, 48, 139),
}

#: The resize rule's constants at the snapshot's budget (patch 16, pooling 3, 280 soft tokens).
PATCH, POOLING, BUDGET = 16, 3, 280


def _resize_rule():
    pytest.importorskip("transformers")
    if metadata.version("transformers") != export.REQUIRED_TRANSFORMERS:
        pytest.skip(f"transformers {export.REQUIRED_TRANSFORMERS} is not installed")
    from transformers.models.gemma4.image_processing_gemma4 import (
        get_aspect_ratio_preserving_size,
    )

    def resized(height: int, width: int) -> tuple[int, int]:
        return get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=PATCH,
            max_patches=BUDGET * POOLING**2,
            pooling_kernel_size=POOLING,
        )

    return resized


class TestPixels:
    def test_each_image_is_rgb8_of_its_size(self) -> None:
        for image in IMAGES.values():
            pixels = image.rgb8()

            assert pixels.dtype == np.uint8
            assert pixels.shape == (image.height, image.width, 3)
            assert pixels.flags["C_CONTIGUOUS"]

    def test_each_image_is_the_same_bytes_every_time(self) -> None:
        for image in IMAGES.values():
            assert np.array_equal(image.rgb8(), image.rgb8())

    def test_no_image_is_flat(self) -> None:
        """A constant image would let a wrong patch order pass (every patch the same)."""
        for image in IMAGES.values():
            assert len(np.unique(image.rgb8().reshape(-1, 3), axis=0)) > 3

    def test_the_names_are_the_tables(self) -> None:
        assert list(IMAGES) == list(TABLE)


class TestResizeRule:
    def test_each_image_resizes_to_its_table_row(self) -> None:
        resized = _resize_rule()
        for name, (width, height, tokens) in TABLE.items():
            image = IMAGES[name]

            target_height, target_width = resized(image.height, image.width)

            assert (target_width, target_height) == (width, height), name
            assert (target_width // PATCH) * (target_height // PATCH) // POOLING**2 == tokens

    def test_the_smallest_count_the_rule_produces_is_139_and_stripes_edge_has_it(self) -> None:
        resized = _resize_rule()
        smallest = BUDGET
        for short in range(1, 31):
            for long in range(short, short * 290 + 1):
                for height, width in ((short, long), (long, short)):
                    target_height, target_width = resized(height, width)
                    count = (target_height // PATCH) * (target_width // PATCH) // POOLING**2
                    smallest = min(smallest, count)

        from embeddinggemma2 import export_vision

        assert smallest == export_vision.SOFT_TOKENS_MIN == TABLE["stripes-edge"][2] == 139
