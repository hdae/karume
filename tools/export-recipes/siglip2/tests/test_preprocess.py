"""Script-level contracts of `siglip2/preprocess.py` (the part that needs no real weights).

The fixture emit is manual (the same discipline as `irodori/tokenizer_ref.py`). What is pinned
here is only the discipline whose breakage would give a **silent parity mismatch**:

- the `resample` check **rejects anything other than 2 (PIL BILINEAR)** — letting 3 (BICUBIC)
  through would make the TS side go green against "a filter other than the reference's" (measured
  to be off by up to 47/255)
- the checks of the preprocessing constants (mean / std / rescale) and `do_*` fail when a single
  field is changed
- `check_sibling_configs` catches differences **other than** `size` (the substance of the claim
  that this layer serves both 224 and 384)
- the "stored intermediate is the real path itself" check of `reference` fails when the resize is
  replaced
- the cases are distinct images and actually cover downscale / upscale / same size / mixed axes /
  1×1 (written so that removing a case fails)
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from siglip2 import preprocess as pre


def _write_config(root, name: str, **overrides) -> None:
    config = {
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "image_processor_type": "SiglipImageProcessor",
        "rescale_factor": 0.00392156862745098,
        "resample": 2,
        "size": {"height": 224, "width": 224},
    }
    config.update(overrides)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "preprocessor_config.json").write_text(json.dumps(config), encoding="utf-8")


class TestCases:
    def test_cases_cover_the_geometries_that_have_their_own_branch(self) -> None:
        cases = pre.build_cases()
        shapes = {
            case["name"]: (case["image"].shape[0], case["image"].shape[1], *case["out"])
            for case in cases
        }
        assert len(shapes) == len(cases), "duplicate case names"
        # Count downscale / upscale / same size / mixed axes / 1×1 from the dimensions, not the
        # names.
        shrink = [s for s in shapes.values() if s[0] > s[2] and s[1] > s[3]]
        grow = [s for s in shapes.values() if s[0] < s[2] and s[1] < s[3]]
        same = [s for s in shapes.values() if s[0] == s[2] and s[1] == s[3]]
        mixed = [s for s in shapes.values() if (s[0] > s[2]) != (s[1] > s[3])]
        single = [s for s in shapes.values() if s[0] == 1 and s[1] == 1]
        assert len(shrink) >= 3 and grow and same and mixed and single

    def test_images_are_pairwise_distinct(self) -> None:
        # Using the same image twice would shrink those two cases into a single check.
        blobs = [case["image"].tobytes() for case in pre.build_cases()]
        assert len(set(blobs)) == len(blobs)

    def test_images_are_rgb8_of_the_declared_shape(self) -> None:
        for case in pre.build_cases():
            image = case["image"]
            assert image.dtype == np.uint8
            assert image.ndim == 3 and image.shape[2] == 3

    def test_cases_are_reproducible(self) -> None:
        # The random seed is fixed, so two calls give the same pixels (otherwise the fixture
        # would differ on every regeneration and its diff would be unreadable).
        first = [case["image"].tobytes() for case in pre.build_cases()]
        second = [case["image"].tobytes() for case in pre.build_cases()]
        assert first == second


class TestSiblingConfigs:
    def test_size_may_differ(self, tmp_path) -> None:
        _write_config(tmp_path, "base", size={"height": 224, "width": 224})
        _write_config(tmp_path, "so400m", size={"height": 384, "width": 384})
        sizes = pre.check_sibling_configs(tmp_path)
        assert sizes == {
            "base": {"height": 224, "width": 224},
            "so400m": {"height": 384, "width": 384},
        }

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("resample", 3),
            ("image_mean", [0.485, 0.456, 0.406]),
            ("rescale_factor", 1.0),
            ("do_normalize", False),
        ],
    )
    def test_other_fields_may_not_differ(self, tmp_path, key, value) -> None:
        _write_config(tmp_path, "base")
        _write_config(tmp_path, "so400m", **{key: value})
        with pytest.raises(SystemExit, match=key):
            pre.check_sibling_configs(tmp_path)

    def test_no_checkpoint_is_an_error(self, tmp_path) -> None:
        with pytest.raises(SystemExit):
            pre.check_sibling_configs(tmp_path)


def _processor(**overrides):
    """A `SiglipImageProcessor` with overridden defaults (skipped where transformers is not
    installed)."""
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    processor = transformers.SiglipImageProcessor(
        resample=pre.EXPECTED_RESAMPLE,
        image_mean=list(pre.EXPECTED_MEAN),
        image_std=list(pre.EXPECTED_STD),
        rescale_factor=pre.EXPECTED_RESCALE,
        size={"height": 8, "width": 8},
    )
    for key, value in overrides.items():
        setattr(processor, key, value)
    return processor


class TestProcessorShape:
    def test_the_expected_shape_passes(self) -> None:
        shape = pre.check_processor_shape(_processor())
        assert shape["resample"] == 2

    def test_bicubic_is_rejected(self) -> None:
        # The crux. `resample` is a PIL constant, 2 = BILINEAR / 3 = BICUBIC, and the class
        # attribute defaults to BICUBIC. The moment the config's override is missed, this stops.
        with pytest.raises(SystemExit, match="resample"):
            pre.check_processor_shape(_processor(resample=3))

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("image_mean", [0.485, 0.456, 0.406], "mean/std"),
            ("image_std", [0.229, 0.224, 0.225], "mean/std"),
            ("rescale_factor", 1.0, "rescale_factor"),
            ("do_resize", False, "do_resize"),
            ("do_rescale", False, "do_rescale"),
            ("do_normalize", False, "do_normalize"),
            ("do_center_crop", True, "do_center_crop"),
            ("do_pad", True, "do_pad"),
        ],
    )
    def test_each_constant_is_gated(self, key, value, match) -> None:
        with pytest.raises(SystemExit, match=match):
            pre.check_processor_shape(_processor(**{key: value}))

    def test_the_pil_backend_is_rejected(self) -> None:
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("torchvision")
        with pytest.raises(SystemExit, match="TorchvisionBackend"):
            pre.check_processor_shape(transformers.SiglipImageProcessorPil())


class TestReference:
    IMAGE = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    def test_the_stored_intermediate_is_on_the_real_path(self) -> None:
        processor = _processor()
        resized, pixel_values = pre.reference(processor, self.IMAGE, 4, 6)
        assert resized.shape == (4, 6, 3)
        assert resized.dtype == np.uint8
        assert pixel_values.shape == (3, 4, 6)

    def test_an_unfused_upstream_fails_loudly(self, monkeypatch) -> None:
        # Inject the form in which upstream changed **how** rescale and normalize are folded
        # (multiply by 1/255 first, then subtract the mean/std on the `[0,1]` scale). The values
        # look the same, but there is one more f32 rounding, so the difference from the fused form
        # the script bakes shows here — if it did not, the TS side would "stay green although
        # upstream changed".
        backend = pytest.importorskip("transformers.image_processing_backends")

        def unfused(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
            scaled = images.to(dtype=torch.float32) * rescale_factor
            mean = torch.tensor(image_mean).view(1, 3, 1, 1)
            std = torch.tensor(image_std).view(1, 3, 1, 1)
            return (scaled - mean) / std

        monkeypatch.setattr(backend.TorchvisionBackend, "rescale_and_normalize", unfused)
        with pytest.raises(SystemExit, match="pixel_values"):
            pre.reference(_processor(), self.IMAGE, 4, 6)


class TestCommittedFixture:
    """Correspondence between the git-tracked fixture and `build_cases()` (turns a missed
    regeneration red).

    The Python side checks the geometric coverage of `build_cases()`, the Deno side
    (`packages/models/tests/image_preprocess_test.ts`) checks the committed JSON — without a
    comparison between them, **both stay green even after they diverge** (the form in which only
    one was edited and the regeneration was forgotten).
    """

    @staticmethod
    def _fixture() -> dict:
        # The file is tracked by git, so a missing file fails rather than skips.
        return json.loads(pre.DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_the_case_names_match_in_order(self) -> None:
        fixture = self._fixture()

        assert [case["name"] for case in fixture["cases"]] == [
            case["name"] for case in pre.build_cases()
        ]

    def test_every_case_carries_the_geometry_it_was_built_from(self) -> None:
        """If the input / output dimensions were off, the Deno side would compare against the
        reference at a different scale."""
        fixture = self._fixture()

        for baked, case in zip(fixture["cases"], pre.build_cases(), strict=True):
            image = case["image"]
            out_height, out_width = case["out"]
            assert (baked["height"], baked["width"]) == image.shape[:2], baked["name"]
            assert (baked["outHeight"], baked["outWidth"]) == (out_height, out_width), baked["name"]

    def test_the_baked_constants_are_the_expected_ones(self) -> None:
        constants = self._fixture()["constants"]

        assert tuple(constants["imageMean"]) == pre.EXPECTED_MEAN
        assert tuple(constants["imageStd"]) == pre.EXPECTED_STD
        assert constants["rescaleFactor"] == pre.EXPECTED_RESCALE
