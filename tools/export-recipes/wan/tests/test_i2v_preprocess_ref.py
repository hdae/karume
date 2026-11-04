"""The contract of `wan/i2v_preprocess_ref.py` — the reference of the I2V image preprocessing
(ADR 0121 decision 11, step 9a).

Runs on synthetic images only (no dependence on the test images, which are not tracked by git). What
it checks:

- crop (official): the scale of the covering side, Python's `round` (.5 goes to the even side) and
  the size and origin of the floor center crop are pinned with the 3 golden sizes and a size where
  the rounding splits (512×301).
- stretch (diffusers): upstream `VideoProcessor.resize` equals a direct Pillow LANCZOS resize.
- [-1, 1]: the official `to_tensor → sub(0.5) → div(0.5)` is bit-identical to
  `f32(2·f32(x/255) − 1)` for all 256 values (the formula the TS side copies).
- patchify: channel = c·4 + width sub-index·2 + height sub-index (width first).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wan import i2v_preprocess_ref

pytest.importorskip("PIL")


def _image(width: int, height: int, seed: int = 0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8))


def _lanczos(image, width: int, height: int):
    from PIL import Image

    return image.resize((width, height), Image.LANCZOS)


class TestCropResize:
    @pytest.mark.parametrize(
        ("size", "resized", "origin"),
        [
            ((1280, 704), (1280, 738), (0, 17)),
            ((704, 1280), (2219, 1280), (757, 0)),
            ((256, 160), (277, 160), (10, 0)),
        ],
        ids=["1280x704", "704x1280", "256x160"],
    )
    def test_the_golden_sizes_of_an_832x480_image(self, size, resized, origin):
        """832×480 → resize to the covering side → center crop (the 3 golden sizes)."""
        out_width, out_height = size
        source = _image(832, 480)
        x1, y1 = origin
        want = _lanczos(source, *resized).crop((x1, y1, x1 + out_width, y1 + out_height))

        got = i2v_preprocess_ref.crop_resize(source, out_width, out_height)

        assert got.size == (out_width, out_height)
        assert np.array_equal(np.asarray(got), np.asarray(want))

    def test_a_half_way_size_rounds_to_the_even_side(self):
        """512×301 → 1280×704: with scale 2.5, 301·2.5 = 752.5 → Python's round gives 752 (it
        splits from JS Math.round, which gives 753). The origin is (752 − 704) // 2 = 24."""
        source = _image(512, 301, seed=1)
        want = _lanczos(source, 1280, 752).crop((0, 24, 1280, 728))
        rounded_away = _lanczos(source, 1280, 753).crop((0, 24, 1280, 728))

        got = np.asarray(i2v_preprocess_ref.crop_resize(source, 1280, 704))

        assert np.array_equal(got, np.asarray(want))
        assert not np.array_equal(got, np.asarray(rounded_away))


class TestStretchResize:
    def test_the_diffusers_resize_is_a_direct_lanczos(self):
        pytest.importorskip("diffusers")
        source = _image(832, 480, seed=2)

        got = i2v_preprocess_ref.stretch_resize(source, 1280, 704)

        assert np.array_equal(np.asarray(got), np.asarray(_lanczos(source, 1280, 704)))

    def test_a_size_off_the_multiple_of_16_is_refused(self):
        with pytest.raises(ValueError, match="16"):
            i2v_preprocess_ref.stretch_resize(_image(64, 64), 100, 64)


class TestSignedUnit:
    def test_every_byte_maps_to_twice_the_f32_quotient_minus_one(self):
        pytest.importorskip("torchvision")
        from PIL import Image

        values = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)

        got = i2v_preprocess_ref.to_signed_unit(Image.fromarray(values))

        quotient = values.astype(np.float32) / np.float32(255)
        want = (np.float32(2) * quotient - np.float32(1)).transpose(2, 0, 1)
        assert got.shape == (1, 3, 1, 16, 16)
        assert np.array_equal(got[0, :, 0].numpy().view(np.int32), want.view(np.int32))


class TestEncoderInput:
    def test_the_channel_is_c_times_4_plus_width_digit_times_2_plus_height_digit(self):
        pytest.importorskip("diffusers")
        sample = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 1, 4, 6)

        patches = i2v_preprocess_ref.encoder_input(sample, 2)

        assert patches.shape == (12, 1, 2, 3)
        for channel in range(3):
            for column_digit in range(2):
                for row_digit in range(2):
                    index = channel * 4 + column_digit * 2 + row_digit
                    want = sample[0, channel, 0, row_digit::2, column_digit::2]
                    assert torch.equal(patches[index, 0], want)


class TestRgb8:
    def test_the_pixels_are_row_major_with_three_bytes_each(self):
        image = _image(5, 3, seed=3)

        got = i2v_preprocess_ref.rgb8(image)

        assert (got.dtype, tuple(got.shape)) == (torch.uint8, (3, 5, 3))
        assert np.array_equal(got.numpy(), np.asarray(image))

    def test_an_image_other_than_rgb_is_refused(self):
        with pytest.raises(ValueError, match="RGB"):
            i2v_preprocess_ref.rgb8(_image(4, 4).convert("L"))
