"""Script-level contracts of `birefnet/export.py` (the part that needs no real weights).

Emitting with the real weights is manual (the same discipline as the existing tests of
`deberta/export.py` / `siglip2/export.py`). This file pins only the rules whose breakage would
give a **false PASS**:

- The graph output is **one matte from the last stage** (if multi-scale intermediate predictions
  get mixed in, the io positional convention shifts silently = a training-mode graph written out
  as an inference graph)
- The series name follows **both the model name and the resolution** (missing either one lets an
  asset of another resolution silently overwrite the same seat)
- The resolution step (a multiple of 64) fails **at the entry** (not as a reshape error halfway)
- `_sanity` checks the separation of the salient object by **order** (the value range and the
  agreement of identical inputs both hold trivially)
- The real-image golden (`--real-images`) **does not silently tolerate a missing image**, puts
  the source image's sha256 in `__metadata__`, and discriminates by **the order of the
  foreground ratios**
- `--verify` does not emit (in the same process the reference gets contaminated)
- The series-name spelling and the normalization constants **agree with the distribution recipe
  (`birefnet.distribution`)** (the upstream has no machine-readable preprocessing config and both
  sides hold declarations, so if they move independently the golden and the user's preprocessing
  drift apart)
- Storage-dtype series (ADR 0113): `--dtype` separates the series, it is exclusive with
  `--verify`, the rounding sits right after `load_model` and before `patch.apply` (moving it
  later moves ⑦'s α / β — fault injection), and neither the quality measurement nor the lossless
  gate holds trivially
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.paths import SERIES_ROOT
from _shared.upstream import UpstreamProvenanceError
from birefnet import export as bn
from karume.container import Provenance
from karume.pipeline import export_to_file

#: The provenance baked into the tiny export (the value is not checked — deriving the script's
#: provenance is `_shared.upstream`'s job).
TINY_PROVENANCE = Provenance(license="fixture")

#: The resolution of the tiny synthetic model (the smallest shape where `disc_mask` has a 2×2
#: inside).
TINY_SIZE = 8

CASES = (
    ("disc", torch.linspace(-1.0, 1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
    ("ramp", torch.linspace(1.0, -1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
)


class TinyMatte(nn.Module):
    """A minimal skeleton of `MatteLogits` (`[B,3,S,S] → [B,1,S,S]`, down to the argument
    name)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.conv(pixel_values)


class TwoOutputMatte(TinyMatte):
    """A shape with two outputs (to check that `_write_io` refuses it)."""

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        matte = super().forward(pixel_values)
        return matte, matte * 2.0


class ScaledPreds(nn.Module):
    """A skeleton of the BiRefNet side that returns `scaled_preds` (a list). Only the number of
    elements is set from outside."""

    def __init__(self, outputs: int) -> None:
        super().__init__()
        self.outputs = outputs
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        matte = self.conv(pixel_values)
        return [matte * float(index + 1) for index in range(self.outputs)]


def _disc_matte(inside: float, outside: float) -> torch.Tensor:
    """A matte with constants inside / outside the disc (only the order of separation moves)."""
    mask = bn.disc_mask(TINY_SIZE)
    return torch.where(mask, torch.tensor(inside), torch.tensor(outside)).reshape(
        1, 1, TINY_SIZE, TINY_SIZE
    )


def _mattes(disc: torch.Tensor) -> dict[str, torch.Tensor]:
    """Four cases; all but `disc` are filled with arbitrary values that differ from each other."""
    return {
        "disc": disc,
        "ramp": torch.full((1, 1, TINY_SIZE, TINY_SIZE), 0.25),
        "checker": torch.full((1, 1, TINY_SIZE, TINY_SIZE), -0.5),
        "noise": torch.full((1, 1, TINY_SIZE, TINY_SIZE), 1.5),
    }


def _ratio_matte(ratio: float, jitter: int) -> torch.Tensor:
    """A matte whose foreground ratio is exactly `ratio`.

    `jitter` shifts only the background value per case (the foreground ratio stays, but the
    tensors **differ** — so the check never stops at `_sanity`'s "every case differs" check
    before it reaches the discrimination).
    """
    pixels = TINY_SIZE * TINY_SIZE
    flat = torch.full((pixels,), -1.0 - 0.01 * jitter)
    flat[: round(ratio * pixels)] = 1.0
    return flat.reshape(1, 1, TINY_SIZE, TINY_SIZE)


def _real_mattes(
    person: tuple[float, float], scene: tuple[float, float]
) -> dict[str, torch.Tensor]:
    """Mattes for the 4 real-image cases (foreground ratios given per group)."""
    ratios = dict(
        zip((*bn.REAL_PERSON_CASES, *bn.REAL_SCENE_CASES), (*person, *scene), strict=True)
    )
    return {
        name: _ratio_matte(ratio, jitter=index + 1)
        for index, (name, ratio) in enumerate(ratios.items())
    }


def _real_array(index: int) -> torch.Tensor:
    """The pixels of one tiny real image (`[S, S, 3]` u8). Values vary by case and channel."""
    array = torch.zeros((TINY_SIZE, TINY_SIZE, 3), dtype=torch.uint8)
    for channel in range(3):
        array[:, :, channel] = 10 * index + 30 * channel + 7
    return array


def _write_real_images(root: Path) -> dict[str, bytes]:
    """Write tiny PNGs under the `REAL_CASES` names and return case name → raw bytes."""
    from PIL import Image

    written: dict[str, bytes] = {}
    for index, (name, file_name, _why) in enumerate(bn.REAL_CASES):
        path = root / file_name
        Image.fromarray(_real_array(index).numpy()).save(path)
        written[name] = path.read_bytes()
    return written


@pytest.fixture
def exported(tmp_path):
    """Export one tiny wrapper and return `(wrapper, graph, out_dir)`."""
    torch.manual_seed(0)
    wrapper = TinyMatte()
    graph = export_to_file(
        wrapper,
        (CASES[0][1],),
        tmp_path / bn.MODEL_FILE,
        provenance=TINY_PROVENANCE,
        graph_name="tiny",
        symbol_names=(),
    )
    return wrapper, graph, tmp_path


class TestSeriesLayout:
    def test_the_default_output_dir_is_a_series(self):
        """Series output goes under `outputs/series/` (not the distribution's `models/` —
        _shared.paths)."""
        assert (
            bn.default_out_dir(bn.DEFAULT_MODEL_DIR, bn.DEFAULT_RESOLUTION, "f16").parent
            == SERIES_ROOT
        )

    def test_each_resolution_gets_its_own_series(self):
        """MUST: the resolution goes into the name too — the baked constants change, so sharing a
        seat would erase the earlier asset."""
        low = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024, "f16")
        high = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 2048, "f16")

        assert low != high
        assert low.name.endswith("-1024-f16") and high.name.endswith("-2048-f16")

    def test_each_model_gets_its_own_series(self):
        other = bn.MODELS_ROOT / "BiRefNet_lite"

        assert bn.default_out_dir(other, 1024, "f16") != bn.default_out_dir(
            bn.DEFAULT_MODEL_DIR, 1024, "f16"
        )

    def test_the_distribution_looks_for_the_same_series_name(self):
        """MUST: the series name is spelled **the same by the writer and the reader**.

        The distribution recipe finds a series by "checkpoint → upstream repository name →
        series directory name + resolution". If only one side's formula moves, assembly fails
        with "no series" (safe in itself), but if it drifts to a spelling that **picks up another
        model's series**, nobody can notice.
        """
        from birefnet.card import BIREFNET_UPSTREAM
        from birefnet.distribution import (
            BIREFNET_MODELS,
            birefnet_default_quant,
            birefnet_series_name,
        )

        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            model_dir = bn.MODELS_ROOT / repo.split("/", 1)[1]
            dtype = birefnet_default_quant(checkpoint)
            for model in BIREFNET_MODELS:
                assert bn.default_out_dir(model_dir, int(model), dtype).name == (
                    birefnet_series_name(checkpoint, model, dtype)
                )


class TestResolution:
    @pytest.mark.parametrize("resolution", [64, 256, 1024, 2048])
    def test_multiples_of_the_step_are_accepted(self, resolution: int):
        bn.assert_resolution(resolution)

    @pytest.mark.parametrize("resolution", [0, -1024, 224, 1000, 1056])
    def test_anything_else_fails_loudly(self, resolution: int):
        """MUST: fail at the entry — passed on, the shapes merely stop matching inside Swin and
        no reason is left."""
        with pytest.raises(SystemExit, match="is not a multiple of 64"):
            bn.assert_resolution(resolution)


class TestMatteLogits:
    def test_it_returns_the_only_prediction(self):
        torch.manual_seed(0)
        wrapper = bn.MatteLogits(ScaledPreds(1)).eval()
        pixel_values = CASES[0][1]

        with torch.no_grad():
            assert torch.equal(wrapper(pixel_values), wrapper.model(pixel_values)[0])

    def test_more_than_one_prediction_fails_loudly(self):
        """MUST: intermediate predictions mean training mode — do not silently take `[-1]`."""
        wrapper = bn.MatteLogits(ScaledPreds(4)).eval()

        with pytest.raises(ValueError, match="a training-mode graph is not written out"):
            wrapper(CASES[0][1])


class TestPreprocessing:
    def test_normalize_matches_the_reference_transform(self):
        """MUST: the normalization is handler.py's verbatim — torchvision's implementation is the
        independent oracle."""
        from torchvision import transforms

        torch.manual_seed(1)
        image = torch.rand(1, 3, TINY_SIZE, TINY_SIZE)
        expected = transforms.Normalize(bn.IMAGENET_MEAN, bn.IMAGENET_STD)(image[0]).unsqueeze(0)

        assert torch.equal(bn.normalize_image(image), expected)

    def test_the_statistics_are_per_channel(self):
        """With the same constant on all 3 channels, a wrong preprocessing order would not show
        in the values."""
        assert len(set(bn.IMAGENET_MEAN)) == 3
        assert len(set(bn.IMAGENET_STD)) == 3

    def test_the_distribution_declares_the_same_constants(self):
        """MUST: the numbers the distribution declares equal the numbers the golden is baked
        with.

        The BiRefNet upstreams have no machine-readable source like `preprocessor_config.json`;
        the normalization constants are written only inside the bundled `handler.py`. So both the
        script (here) and the assembly (`birefnet.distribution`) have to hold declarations — if
        the two tables move independently, **a golden baked with one set of statistics while
        users preprocess with the other** can be produced silently.
        """
        from birefnet.distribution import BIREFNET_IMAGE_MEAN, BIREFNET_IMAGE_STD

        assert BIREFNET_IMAGE_MEAN == bn.IMAGENET_MEAN
        assert BIREFNET_IMAGE_STD == bn.IMAGENET_STD


class TestGoldenCases:
    def test_cases_have_the_configured_shape(self):
        for name, pixel_values in bn.build_cases(TINY_SIZE):
            assert tuple(pixel_values.shape) == (1, 3, TINY_SIZE, TINY_SIZE), name
            assert pixel_values.dtype is torch.float32, name

    def test_every_case_is_a_different_image(self):
        """MUST: no image is used twice — the sanity check that compares outputs would hold
        trivially."""
        cases = bn.build_cases(TINY_SIZE)

        for index, (name, pixel_values) in enumerate(cases):
            for other_name, other in cases[index + 1 :]:
                assert not torch.equal(pixel_values, other), (
                    f"{name} and {other_name} are identical"
                )

    def test_the_disc_case_actually_has_a_disc(self):
        """The basis of the discrimination — if inside and outside the disc had the same colour,
        {@link bn._sanity} would lose its meaning."""
        cases = dict(bn.build_cases(TINY_SIZE))
        image = cases[bn.DISC_CASE][0]
        mask = bn.disc_mask(TINY_SIZE)

        assert bool(mask.any()) and not bool(mask.all())
        for channel in range(3):
            plane = image[channel]
            assert float(plane[mask].std()) == 0.0
            assert float(plane[mask].mean()) != float(plane[~mask].mean())

    def test_the_sanity_case_is_one_of_the_cases(self):
        assert bn.DISC_CASE in {name for name, _ in bn.build_cases(TINY_SIZE)}


class TestWriteIo:
    def test_writes_one_file_per_case_with_the_declared_keys(self, exported):
        wrapper, graph, out_dir = exported

        written, mattes = bn._write_io(wrapper, graph, CASES, out_dir)

        assert written == [f"{bn.IO_PREFIX}{name}{bn.IO_SUFFIX}" for name, _ in CASES]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{bn.INPUT_PREFIX}{bn.INPUT_NAME}", f"{bn.OUTPUT_PREFIX}0"}
        assert tuple(mattes["disc"].shape) == (1, 1, TINY_SIZE, TINY_SIZE)

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path):
        """MUST: the output is one matte — a second one would silently shift the io positional
        convention."""
        torch.manual_seed(0)
        wrapper = TwoOutputMatte()
        graph = export_to_file(
            wrapper,
            (CASES[0][1],),
            tmp_path / bn.MODEL_FILE,
            provenance=TINY_PROVENANCE,
            graph_name="tiny",
            symbol_names=(),
        )

        with pytest.raises(AssertionError, match="the final matte is 1"):
            bn._write_io(wrapper, graph, CASES, tmp_path)

    def test_metadata_is_written_only_for_the_cases_that_have_it(self, exported):
        """Only the real-image cases carry `__metadata__` (identifying the source image)."""
        wrapper, graph, out_dir = exported
        metadata = {CASES[0][0]: {bn.SOURCE_IMAGE_KEY: "photo.png", bn.SOURCE_SHA256_KEY: "ab"}}

        written, _ = bn._write_io(wrapper, graph, CASES, out_dir, metadata)

        with safe_open(str(out_dir / written[0]), framework="pt") as handle:
            assert handle.metadata() == metadata[CASES[0][0]]
        with safe_open(str(out_dir / written[1]), framework="pt") as handle:
            assert handle.metadata() is None


class TestSanity:
    def test_passes_when_the_disc_is_brighter_than_its_surroundings(self):
        result = bn._sanity(_mattes(_disc_matte(inside=3.0, outside=-4.0)))

        assert result["disc_logit_mean"] == {"inside": 3.0, "outside": -4.0}
        covered = float(bn.disc_mask(TINY_SIZE).to(torch.float32).mean())
        assert result["foreground_ratio"][bn.DISC_CASE] == pytest.approx(covered)

    def test_fails_loudly_when_the_order_is_inverted(self):
        """MUST: a salient object darker than the background is a broken segmentation."""
        with pytest.raises(AssertionError, match="the salient object is not separated"):
            bn._sanity(_mattes(_disc_matte(inside=-4.0, outside=3.0)))

    def test_fails_loudly_when_the_matte_is_uniform(self):
        """An output collapsed to a constant (inside and outside the disc equal) fails too."""
        with pytest.raises(AssertionError, match="the salient object is not separated"):
            bn._sanity(_mattes(_disc_matte(inside=0.5, outside=0.5)))

    def test_fails_loudly_when_two_cases_share_an_output(self):
        """MUST: fail when the input does not reach the output (identical outputs side by side)."""
        mattes = _mattes(_disc_matte(inside=3.0, outside=-4.0))
        mattes["checker"] = mattes["ramp"].clone()

        with pytest.raises(AssertionError, match="the input has no effect"):
            bn._sanity(mattes)

    def test_it_reports_the_shape_mismatch_instead_of_broadcasting(self):
        """When the disc's shape and the output's shape disagree, do not broadcast silently (an
        H ≠ W matte)."""
        mattes = _mattes(torch.zeros(1, 1, 2, TINY_SIZE))

        with pytest.raises(AssertionError, match="differs from the disc's shape"):
            bn._sanity(mattes)


class TestRealImageCases:
    def test_the_two_groups_partition_the_cases(self):
        """MUST: fail when a group's names leave the case names (the discrimination would
        silently look at two other images)."""
        names = {name for name, _file, _why in bn.REAL_CASES}

        assert set(bn.REAL_PERSON_CASES).isdisjoint(bn.REAL_SCENE_CASES)
        assert set(bn.REAL_PERSON_CASES) | set(bn.REAL_SCENE_CASES) == names

    def test_every_case_reads_a_different_file(self):
        files = [file for _name, file, _why in bn.REAL_CASES]

        assert len(set(files)) == len(files)

    def test_it_normalizes_with_the_handler_statistics(self, tmp_path):
        """The preprocessing is handler.py's verbatim — the values are rebuilt independently and
        compared."""
        raw = _write_real_images(tmp_path)

        cases = bn.build_real_cases(TINY_SIZE, tmp_path)

        assert [name for name, _pixels, _md in cases] == list(raw)
        mean = torch.tensor(bn.IMAGENET_MEAN).reshape(3, 1, 1)
        std = torch.tensor(bn.IMAGENET_STD).reshape(3, 1, 1)
        for index, (name, pixel_values, _md) in enumerate(cases):
            assert tuple(pixel_values.shape) == (1, 3, TINY_SIZE, TINY_SIZE), name
            assert pixel_values.dtype is torch.float32, name
            # The expectation is rebuilt from the written pixels (`ToTensor` / `Normalize` are
            # not copied; only "reorder to planar, divide by 255, subtract the statistics" is
            # spelled out as a separate path).
            planar = _real_array(index).permute(2, 0, 1).to(torch.float32)
            assert torch.equal(pixel_values[0], ((planar / 255.0) - mean) / std), name

    def test_it_records_the_source_image_and_its_digest(self, tmp_path):
        """MUST: the fields that fail an environment whose images were re-baked without
        re-recording the golden, before the comparison."""
        raw = _write_real_images(tmp_path)

        cases = bn.build_real_cases(TINY_SIZE, tmp_path)

        files = {name: file for name, file, _why in bn.REAL_CASES}
        for name, _pixel_values, metadata in cases:
            assert metadata[bn.SOURCE_IMAGE_KEY] == files[name]
            assert metadata[bn.SOURCE_SHA256_KEY] == hashlib.sha256(raw[name]).hexdigest()

    def test_a_missing_image_fails_loudly(self, tmp_path):
        """MUST: do not silently write three images (`--real-images` is an explicit request)."""
        _write_real_images(tmp_path)
        (tmp_path / bn.REAL_CASES[-1][1]).unlink()

        with pytest.raises(SystemExit, match=r"real image .* is missing"):
            bn.build_real_cases(TINY_SIZE, tmp_path)


class TestRealSanity:
    def test_synthetic_only_emits_have_no_real_verdict(self):
        """The default emit (4 synthetic cases) passes the sanity check too (the real images are
        an extra group)."""
        assert "real_foreground" not in bn._sanity(_mattes(_disc_matte(inside=3.0, outside=-4.0)))

    def test_it_passes_when_the_salient_cases_have_more_foreground(self):
        mattes = {
            **_mattes(_disc_matte(inside=3.0, outside=-4.0)),
            **_real_mattes(person=(0.5, 0.125), scene=(0.0, 0.0625)),
        }

        result = bn._sanity(mattes)

        assert result["real_foreground"][bn.REAL_PERSON_CASES[0]] == 0.5
        assert result["real_foreground"][bn.REAL_SCENE_CASES[0]] == 0.0

    def test_it_fails_loudly_when_a_scene_has_more_foreground(self):
        """MUST: if an image without a salient object covers more than a person, the matte has
        not captured the meaning."""
        mattes = {
            **_mattes(_disc_matte(inside=3.0, outside=-4.0)),
            **_real_mattes(person=(0.5, 0.0625), scene=(0.0, 0.25)),
        }

        with pytest.raises(
            AssertionError, match="the order of the real images' foreground ratios is reversed"
        ):
            bn._sanity(mattes)

    def test_it_fails_loudly_when_every_case_covers_the_same_area(self):
        """MUST: fail an output whose area no longer depends on the input (the values differ but
        the order ties).

        Identical values would be caught first by the earlier "every case differs" check, so
        only **the area** is equalized here (checking that the foreground-ratio side has no hole
        left).
        """
        mattes = {
            **_mattes(_disc_matte(inside=3.0, outside=-4.0)),
            **_real_mattes(person=(0.5, 0.5), scene=(0.5, 0.5)),
        }

        with pytest.raises(
            AssertionError, match="the order of the real images' foreground ratios is reversed"
        ):
            bn._sanity(mattes)


class TestVerifyCli:
    def test_verify_does_not_emit(self, monkeypatch):
        """MUST: never combined with emit in one process (the class replacement contaminates the
        reference)."""
        seen: list[str] = []
        monkeypatch.setattr(
            bn,
            "verify_patches",
            lambda _dir, _resolution: (
                seen.append("verify")
                or [{"stage": "layout", "claim": "bit-exact", "bit_exact": True, "maxdiff": {}}]
            ),
        )
        monkeypatch.setattr(bn, "export_series", lambda *_a, **_kw: pytest.fail("--verify emitted"))

        bn.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            bn, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(bn, "verify_patches", lambda *_a: pytest.fail("emit ran --verify"))

        bn.main([])

        assert seen == ["emit"]

    def test_the_output_dir_follows_the_model_dir_and_the_resolution(self, monkeypatch, tmp_path):
        """MUST: without `--out` the series follows both axes (a fixed one would be overwritten)."""
        seen: list[tuple[Path, int]] = []
        monkeypatch.setattr(
            bn,
            "export_series",
            lambda _dir, out, resolution, **_kw: (
                seen.append((out, resolution)) or {"dir": str(out)}
            ),
        )

        bn.main(["--model-dir", str(tmp_path / "BiRefNet_lite"), "--resolution", "512"])

        assert seen == [(SERIES_ROOT / "birefnet-lite-512", 512)]

    @pytest.mark.parametrize(("argv", "expected"), [([], False), (["--real-images"], True)])
    def test_real_images_is_opt_in(self, monkeypatch, argv: list[str], expected: bool):
        """The real-image golden is written only on explicit request (the default is the 4
        synthetic cases)."""
        seen: list[bool] = []
        monkeypatch.setattr(
            bn,
            "export_series",
            lambda _dir, out, _resolution, real_images, **_kw: (
                seen.append(real_images) or {"dir": str(out)}
            ),
        )

        bn.main(argv)

        assert seen == [expected]


class _RecordingMatte(nn.Module):
    """A skeleton that only records the stage of each call (reference / stage 1 / stage 2 /
    stage 3).

    The output is decided by the input (identical at every stage), so the first-stage
    bit-identity assert of `verify_patches` passes — all this wants to see is **the order**.
    """

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls
        self.stage = "reference"

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        if not self.calls or self.calls[-1] != self.stage:
            self.calls.append(self.stage)
        return [pixel_values.mean(dim=1, keepdim=True)]


class TestVerifyOrder:
    """The ordering invariant of `verify_patches` (reference → stage 1 → stage 2 → stage 3)."""

    RESOLUTION = 64

    def test_a_patched_process_cannot_take_the_reference(self, monkeypatch):
        """MUST: a process with the patches applied does not take the reference (a zero
        difference that holds trivially)."""
        monkeypatch.setattr(bn.patch, "patches_applied", lambda: True)

        with pytest.raises(SystemExit, match="trivially true"):
            bn.verify_patches(Path("/nonexistent"), self.RESOLUTION)

    def test_the_reference_is_taken_once_before_any_patch(self, monkeypatch):
        """Catch the regression that re-takes the reference per stage (the second stage's
        reference would be a patched value)."""
        calls: list[str] = []
        recorder = _RecordingMatte(calls)
        monkeypatch.setattr(bn.patch, "patches_applied", lambda: False)
        monkeypatch.setattr(bn, "load_model", lambda _dir: recorder)

        def _apply_layout(model: nn.Module) -> dict[str, int]:
            calls.append("apply_layout")
            model.stage = "layout"
            return {}

        def _apply_modules(model: nn.Module) -> dict[str, int]:
            calls.append("apply_modules")
            model.stage = "modules"
            return {}

        def _apply_tail(model: nn.Module) -> dict[str, int]:
            calls.append("apply_tail")
            model.stage = "tail"
            return {}

        monkeypatch.setattr(bn.patch, "apply_layout_patches", _apply_layout)
        monkeypatch.setattr(bn.patch, "apply_module_patches", _apply_modules)
        monkeypatch.setattr(bn.patch, "apply_tail_patches", _apply_tail)
        monkeypatch.setattr(
            bn.patch, "prepare", lambda _wrapper, _sample: calls.append("prepare") or None
        )

        entries = bn.verify_patches(Path("/nonexistent"), self.RESOLUTION)

        assert calls == [
            "reference",
            "apply_layout",
            "prepare",
            "layout",
            "apply_modules",
            "modules",
            "apply_tail",
            "tail",
        ]
        assert [entry["stage"] for entry in entries] == ["layout", "modules", "tail"]


class TestProvenance:
    """The container's provenance is derived from the actual `--model-dir`
    (`_shared.upstream.snapshot_provenance`)."""

    def test_a_checkpoint_without_the_download_record_is_refused_before_the_weights(
        self, monkeypatch, tmp_path
    ) -> None:
        """A checkpoint that cannot state its provenance fails before the weights are read
        (nothing gets written)."""
        monkeypatch.setattr(
            bn,
            "load_wrapper",
            lambda *_a: pytest.fail("read the weights before the provenance check"),
        )
        out_dir = tmp_path / "series"

        with pytest.raises(UpstreamProvenanceError, match="is missing"):
            bn.export_series(tmp_path / "BiRefNet_HR", out_dir, bn.DEFAULT_RESOLUTION, dtype="f16")

        assert not out_dir.exists()


# ---- Storage-dtype series (ADR 0113) --------------------------------------------


class TestWeightDtypeSeries:
    def test_the_f32_series_keeps_its_existing_name(self):
        """f32 has no suffix — the names of existing series (shipped assets) do not move."""
        assert bn.default_out_dir(bn.MODELS_ROOT / "lucida", 1024, "f32").name == "lucida-1024"

    def test_each_storage_dtype_gets_its_own_series(self):
        """MUST: sharing a series would silently apply the f32 series' net (the per-series
        tolerance) to compressed assets."""
        plain = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024, "f32")
        half = bn.default_out_dir(bn.DEFAULT_MODEL_DIR, 1024, "f16")

        assert plain != half
        assert half.name == "birefnet-hr-1024-f16"

    def test_the_distribution_offers_exactly_the_dtypes_the_recipe_writes(self):
        """If the writer's (script's) and the reader's (distribution recipe's) dtype tables move
        independently, a seat that cannot be baked gets shipped."""
        from birefnet.distribution import BIREFNET_WEIGHT_DTYPES

        assert bn.WEIGHT_DTYPES == BIREFNET_WEIGHT_DTYPES

    def test_the_distribution_looks_for_the_same_series_name_per_dtype(self):
        """MUST: the writer and the reader spell the series name the same, down to the dtype
        suffix."""
        from birefnet.card import BIREFNET_UPSTREAM
        from birefnet.distribution import BIREFNET_MODELS, birefnet_series_name

        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            model_dir = bn.MODELS_ROOT / repo.split("/", 1)[1]
            for model in BIREFNET_MODELS:
                for dtype in bn.WEIGHT_DTYPES:
                    assert bn.default_out_dir(
                        model_dir, int(model), dtype
                    ).name == birefnet_series_name(checkpoint, model, dtype)


def _record_emit(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    """Replace `export_series` and collect the `(out_dir, dtype)` it receives."""
    seen: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        bn,
        "export_series",
        lambda _dir, out, _resolution, real_images, dtype: (
            seen.append((out, dtype)) or {"dir": str(out)}
        ),
    )
    return seen


def _downloaded(model_dir: Path, revision: str) -> Path:
    """A `--model-dir` holding only the download record of `hf download --local-dir` (the form
    `_shared.upstream.snapshot_revision` reads)."""
    from _shared.upstream import SNAPSHOT_RECORD_DIR, SNAPSHOT_RECORD_FILE

    record = model_dir / SNAPSHOT_RECORD_DIR / f"{SNAPSHOT_RECORD_FILE}.metadata"
    record.parent.mkdir(parents=True)
    record.write_text(f"{revision}\netag\n0\n", encoding="utf-8")
    return model_dir


class TestDtypeCli:
    def test_the_default_is_the_checkpoints_own_storage(self, monkeypatch):
        """The default is the upstream storage type (ADR 0113 addendum 2026-10-11) — f16 for
        HR, f32 for Lucida."""
        seen = _record_emit(monkeypatch)

        bn.main([])
        bn.main(["--model-dir", str(bn.MODELS_ROOT / "lucida")])

        assert seen == [
            (SERIES_ROOT / "birefnet-hr-1024-f16", "f16"),
            (SERIES_ROOT / "lucida-1024", "f32"),
        ]

    def test_hr_refuses_f32(self, monkeypatch, capsys):
        """HR's upstream is f16 only — an f32 series only widens the values exactly, adds no
        information, and is not shipped."""
        monkeypatch.setattr(bn, "export_series", lambda *_a, **_kw: pytest.fail("emitted"))

        with pytest.raises(SystemExit):
            bn.main(["--dtype", "f32"])

        assert "a dtype wider than the upstream checkpoint's storage" in capsys.readouterr().err

    def test_lucida_writes_both_dtypes(self, monkeypatch):
        """Lucida's upstream is f32 — both f32 (the default) and a rounded f16 can be written."""
        seen = _record_emit(monkeypatch)
        lucida = str(bn.MODELS_ROOT / "lucida")

        bn.main(["--model-dir", lucida, "--dtype", "f32"])
        bn.main(["--model-dir", lucida, "--dtype", "f16"])

        assert seen == [
            (SERIES_ROOT / "lucida-1024", "f32"),
            (SERIES_ROOT / "lucida-1024-f16", "f16"),
        ]

    def test_the_script_writes_exactly_the_seats_the_distribution_ships(self):
        """The writer and the shipper share one rule (no series is baked that cannot ship, and
        every shipped seat's series can be baked).

        The checkpoint is the one whose pinned value matches the revision in `--model-dir`'s
        download record; otherwise (also when the record is missing or unreadable) it is looked
        up from the directory name (`model_dir_checkpoint`).
        """
        from birefnet.card import BIREFNET_CHECKPOINTS
        from birefnet.distribution import birefnet_default_quant, birefnet_seat_dtypes

        for checkpoint, entry in BIREFNET_CHECKPOINTS.items():
            model_dir = bn.MODELS_ROOT / entry.repo.split("/", 1)[1]
            assert bn.export_dtypes(model_dir) == (
                birefnet_seat_dtypes(checkpoint),
                birefnet_default_quant(checkpoint),
            )

    def test_a_renamed_hr_directory_still_refuses_f32(self, monkeypatch, tmp_path, capsys):
        """MUST: the checkpoint is looked up from the download record's revision first — an HR
        with a renamed directory does not fall back to an experimental checkpoint and cannot
        write the retired f32 series.
        """
        from birefnet.card import BIREFNET_CHECKPOINTS
        from birefnet.distribution import birefnet_default_quant, birefnet_seat_dtypes

        model_dir = _downloaded(tmp_path / "hr-renamed", BIREFNET_CHECKPOINTS["hr"].revision)
        monkeypatch.setattr(bn, "export_series", lambda *_a, **_kw: pytest.fail("emitted"))

        assert bn.export_dtypes(model_dir) == (
            birefnet_seat_dtypes("hr"),
            birefnet_default_quant("hr"),
        )
        with pytest.raises(SystemExit):
            bn.main(["--model-dir", str(model_dir), "--dtype", "f32"])
        assert "a dtype wider than the upstream checkpoint's storage" in capsys.readouterr().err

    def test_a_revision_and_a_name_that_disagree_are_refused(self, tmp_path):
        """MUST: the content (revision) is HR, the name is Lucida — the disagreement is not
        silently resolved to either side."""
        from birefnet.card import BIREFNET_CHECKPOINTS

        model_dir = _downloaded(tmp_path / "lucida", BIREFNET_CHECKPOINTS["hr"].revision)

        with pytest.raises(SystemExit, match="disagree"):
            bn.export_dtypes(model_dir)

    def test_an_unpinned_revision_falls_back_to_the_directory_name(self, tmp_path):
        """A revision matching no pinned value (an upstream update, say) is looked up by the
        directory name."""
        from birefnet.distribution import birefnet_default_quant, birefnet_seat_dtypes

        model_dir = _downloaded(tmp_path / "BiRefNet_HR", "0" * 40)

        assert bn.export_dtypes(model_dir) == (
            birefnet_seat_dtypes("hr"),
            birefnet_default_quant("hr"),
        )

    def test_f16_is_written_to_its_own_series(self, monkeypatch):
        seen = _record_emit(monkeypatch)

        bn.main(["--dtype", "f16", "--resolution", "2048"])

        assert seen == [(SERIES_ROOT / "birefnet-hr-2048-f16", "f16")]

    @pytest.mark.parametrize("dtype", ["f32", "f16"])
    def test_dtype_and_verify_are_exclusive(self, monkeypatch, dtype: str):
        """MUST: `--verify`'s reference is the unpatched f32 eager model — an explicit `--dtype`
        is refused even when it is f32 (the same structural reason as ADR 0027 decision 3).
        Neither path runs.
        """
        monkeypatch.setattr(bn, "verify_patches", lambda *_a: pytest.fail("--verify ran"))
        monkeypatch.setattr(bn, "export_series", lambda *_a, **_kw: pytest.fail("emitted"))

        with pytest.raises(SystemExit):
            bn.main(["--verify", "--dtype", dtype])

    def test_i8_is_not_offered_yet(self, monkeypatch):
        """i8 is added in stage 2 together with a BiRefNet-specific i8 plan — a copied i8 would
        break silently."""
        monkeypatch.setattr(bn, "export_series", lambda *_a, **_kw: pytest.fail("emitted"))

        with pytest.raises(SystemExit):
            bn.main(["--dtype", "i8"])


class TinyBatchNormMatte(nn.Module):
    """A skeleton of what `load_model` returns (it returns a list) with one BatchNorm in between.

    The minimal shape for seeing how ⑦'s (`BatchNorm2d` → `ChannelAffine`) α / β move with the
    position of the rounding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        return [self.head(self.norm(self.conv(pixel_values)))]


def _f16_checkpoint_model() -> TinyBatchNormMatte:
    """An eval model like HR (every weight and BatchNorm statistic is representable in f16)."""
    torch.manual_seed(7)
    model = TinyBatchNormMatte().eval()
    with torch.no_grad():
        model.norm.running_mean.uniform_(-0.5, 0.5)
        model.norm.running_var.uniform_(0.3, 3.0)
        model.norm.weight.uniform_(0.5, 1.5)
        model.norm.bias.uniform_(-0.2, 0.2)
        for tensor in (*model.parameters(), model.norm.running_mean, model.norm.running_var):
            tensor.copy_(tensor.to(torch.float16).to(torch.float32))
    return model


def _affine(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """The α / β of the `ChannelAffine` that ⑦ added."""
    affines = [module for module in model.modules() if isinstance(module, bn.patch.ChannelAffine)]
    assert len(affines) == 1
    return affines[0].alpha.clone(), affines[0].beta.clone()


def _f32_checkpoint_model() -> TinyBatchNormMatte:
    """An eval model like Lucida (weights and BatchNorm statistics are f32 values not
    representable in f16).

    The seed is fixed, so the two loads (for the reference and for the rounding) return the same
    weights.
    """
    torch.manual_seed(11)
    model = TinyBatchNormMatte().eval()
    with torch.no_grad():
        model.norm.running_mean.uniform_(-0.5, 0.5)
        model.norm.running_var.uniform_(0.3, 3.0)
        model.norm.weight.uniform_(0.5, 1.5)
        model.norm.bias.uniform_(-0.2, 0.2)
    return model


@pytest.fixture
def tiny_load(monkeypatch: pytest.MonkeyPatch) -> list[str | tuple[str, str]]:
    """Replace what `load_wrapper` calls out to (loading weights, patching, baking) with tiny
    versions and collect the call order.

    Only patches ⑦⑧ are applied (`apply_module_patches` — the stage that applies to any module).
    ①–⑥ replace class attributes of the upstream dynamic module, so they do not apply to tiny.
    Rounding is recorded as `("round", dtype)` (so the call order tells the reference side from
    the rounding side).
    """
    calls: list[str | tuple[str, str]] = []

    def _load(_dir: Path) -> nn.Module:
        calls.append("load")
        return _f16_checkpoint_model()

    real_round = bn.round_to_storage

    def _round(model: nn.Module, dtype: str):
        calls.append(("round", dtype))
        return real_round(model, dtype)

    def _apply(model: nn.Module) -> dict[str, int]:
        calls.append("apply")
        return bn.patch.apply_module_patches(model)

    def _prepare(wrapper: nn.Module, sample: torch.Tensor) -> torch.Tensor:
        calls.append("prepare")
        with torch.no_grad():
            return wrapper(sample)

    monkeypatch.setattr(bn, "load_model", _load)
    monkeypatch.setattr(bn, "round_to_storage", _round)
    monkeypatch.setattr(bn.patch, "apply", _apply)
    monkeypatch.setattr(bn.patch, "prepare", _prepare)
    return calls


class TestRoundingPoint:
    """Rounding comes right after `load_model` and before `patch.apply` (the MUST of the module
    docstring)."""

    def test_the_weights_are_rounded_between_the_load_and_the_patches(self, tiny_load):
        bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f16")

        assert tiny_load == ["load", ("round", "f16"), "apply", "prepare"]

    def test_f32_does_not_round(self, tiny_load):
        _wrapper, rounding = bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f32")

        assert rounding is None

    def test_an_f16_checkpoint_comes_out_bit_identical(self, tiny_load):
        """Like HR (an f16 checkpoint): the rounding is the identity, and α / β and the outputs
        are bit-identical to the f32 series."""
        plain, _ = bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f32")
        half, rounding = bn.load_wrapper(Path("/nonexistent"), TINY_SIZE, "f16")
        pixel_values = CASES[0][1]

        assert rounding is not None and rounding.parameters > 0
        for got, expected in zip(_affine(half), _affine(plain), strict=True):
            assert bn._bit_exact(got, expected)
        with torch.no_grad():
            assert bn._bit_exact(half(pixel_values), plain(pixel_values))

    def test_rounding_after_the_patches_moves_alpha_and_beta(self):
        """Fault injection: rounding **after** apply moves even the derived constants α / β,
        stored in f32, to f16 values.

        α = w·rsqrt(var+eps), derived from BatchNorm statistics representable in f16, is in
        general not representable in f16, so rounding afterwards moves the value — storage stays
        f32, so VRAM does not shrink and only the numbers move. For an HR-like checkpoint this
        also checks that the difference fails the quality gate (lossless = bit-identical).
        """
        before = _f16_checkpoint_model()
        bn.round_to_storage(before, "f16")
        bn.patch.apply_module_patches(before)
        after = _f16_checkpoint_model()
        bn.patch.apply_module_patches(after)
        bn.round_weights_to_f16(after)

        alpha_before, beta_before = _affine(before)
        alpha_after, beta_after = _affine(after)
        assert not bn._bit_exact(alpha_before, alpha_after)
        assert not bn._bit_exact(beta_before, beta_after)

        pixel_values = CASES[0][1]
        with torch.no_grad():
            reference = {"disc": before(pixel_values)[0]}
            misplaced = {"disc": after(pixel_values)[0]}
        quality = bn.measure_quality("f16", {"F16": 5}, reference, misplaced)
        with pytest.raises(AssertionError, match="does not match f32 bit for bit"):
            bn.assert_quality_gate(quality)

    def test_an_unknown_dtype_is_refused_before_touching_the_weights(self):
        model = _f16_checkpoint_model()
        snapshot = {name: tensor.clone() for name, tensor in model.state_dict().items()}

        with pytest.raises(SystemExit, match="cannot write"):
            bn.round_to_storage(model, "i8")
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, snapshot[name]), name


class TestCheckpointCensus:
    def test_it_counts_the_stored_dtypes_from_the_header(self, tmp_path):
        save_file(
            {
                "a": torch.zeros(2, dtype=torch.float16),
                "b": torch.zeros(3, dtype=torch.float16),
                "c": torch.zeros(1, dtype=torch.int64),
            },
            str(tmp_path / bn.CHECKPOINT_FILE),
        )

        assert bn.checkpoint_dtypes(tmp_path) == {"F16": 2, "I64": 1}

    def test_a_missing_checkpoint_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="census"):
            bn.checkpoint_dtypes(tmp_path)

    @pytest.mark.parametrize(
        ("census", "lossless"),
        [
            ({"F16": 687, "I64": 67}, True),
            ({"F32": 687, "I64": 67}, False),
            ({"F16": 1, "F32": 1}, False),
            ({"F16": 1, "BF16": 1}, False),
            # An unknown spelling counts as floating point (leaning towards not claiming
            # lossless).
            ({"F16": 1, "F8_E8M0": 1}, False),
            ({"I64": 3}, False),
            ({}, False),
        ],
    )
    def test_f16_is_lossless_only_when_every_float_is_f16(self, census, lossless):
        assert bn.lossless_under("f16", census) is lossless

    def test_f32_never_rounds(self):
        assert bn.lossless_under("f32", {"F32": 1})

    def test_another_compressed_dtype_has_no_lossless_verdict(self):
        """The verdict "lossless when everything is F16" is for f16 only.

        Reused for stage 2's i8, it would become a gate demanding bit identity from HR's i8
        series.
        """
        with pytest.raises(ValueError, match="i8"):
            bn.lossless_under("i8", {"F16": 687, "I64": 67})

    @pytest.mark.parametrize("checkpoint", ["hr", "lucida"])
    def test_the_card_names_the_same_storage_as_the_real_header(self, checkpoint: str):
        """The card's `stored_dtype` (a fact not in the manifest) agrees with the header the
        export gate reads.

        SKIP on a machine without the real weights (`inputs/` is outside the repository). If they
        disagree, the quality gate treats a seat the card and NOTICE call "lossless" as lossy (or
        the other way round).
        """
        from birefnet.card import BIREFNET_CHECKPOINTS

        entry = BIREFNET_CHECKPOINTS[checkpoint]
        model_dir = bn.MODELS_ROOT / entry.repo.split("/", 1)[1]
        if not (model_dir / bn.CHECKPOINT_FILE).is_file():
            pytest.skip(f"{model_dir} has no real weights")

        census = bn.checkpoint_dtypes(model_dir)

        assert {name for name in census if name not in bn._NON_FLOAT_DTYPES} == {
            entry.stored_dtype.upper()
        }
        assert bn.lossless_under("f16", census) is (entry.stored_dtype == "f16")

    @pytest.mark.parametrize("checkpoint", ["hr", "lucida"])
    def test_the_pinned_revision_is_the_one_the_script_bakes(self, checkpoint: str):
        """The pinned revision of the attribution table (the value assembly checks the
        container's provenance against) agrees with the revision of the local download record the
        script bakes into the container.

        SKIP on a machine without the download record (`inputs/` is outside the repository). If
        they disagree, every locally baked series fails assembly's provenance gate (to take an
        upstream update, move the pinned value).
        """
        from _shared.upstream import SNAPSHOT_RECORD_DIR, SNAPSHOT_RECORD_FILE, snapshot_revision
        from birefnet.card import BIREFNET_CHECKPOINTS

        entry = BIREFNET_CHECKPOINTS[checkpoint]
        model_dir = bn.MODELS_ROOT / entry.repo.split("/", 1)[1]
        if not (model_dir / SNAPSHOT_RECORD_DIR / f"{SNAPSHOT_RECORD_FILE}.metadata").is_file():
            pytest.skip(f"{model_dir} has no download record")

        assert snapshot_revision(model_dir) == entry.revision


def _logits(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, len(values))


class TestQualityMeasure:
    def test_identical_outputs_measure_zero(self):
        reference = {"disc": _logits(-3.0, 0.5, 4.0)}

        quality = bn.measure_quality("f16", {"F32": 1}, reference, {"disc": reference["disc"]})

        case = quality["cases"]["disc"]
        assert case["bit_exact"] is True
        assert case["max_abs"] == 0.0
        assert case["mask_mismatch"] == 0.0
        assert case["alpha_max_lsb"] == 0

    def test_it_measures_the_logit_mask_and_alpha_differences(self):
        """Sign flips (boundary pixels) and steps of the 8-bit α are counted separately."""
        reference = {"disc": _logits(-0.01, 2.0, 0.0, 5.0)}
        rounded = {"disc": _logits(0.01, 2.0, 0.0, 5.5)}

        case = bn.measure_quality("f16", {"F32": 1}, reference, rounded)["cases"]["disc"]

        assert case["bit_exact"] is False
        assert case["max_abs"] == pytest.approx(0.5)
        # Only 1 of the 4 pixels (-0.01 → 0.01) flips in the mask.
        assert case["mask_mismatch"] == pytest.approx(0.25)
        # 8-bit α: the flipped pixel moves one step 127 → 128, and 5.0 → 5.5 one step 254 → 255.
        assert case["alpha_max_lsb"] == 1
        assert case["alpha_lsb_changed"] == pytest.approx(0.5)
        assert case["rel_rms"] > 0.0

    def test_signed_zero_is_not_bit_exact(self):
        """`torch.equal` holds `0.0 == -0.0` true — the lossless gate compares bits."""
        case = bn.measure_quality(
            "f16", {"F16": 1}, {"disc": _logits(0.0, 1.0)}, {"disc": _logits(-0.0, 1.0)}
        )["cases"]["disc"]

        assert case["max_abs"] == 0.0
        assert case["bit_exact"] is False

    def test_the_worst_row_takes_the_maximum_over_the_cases(self):
        reference = {"disc": _logits(1.0, 2.0), "ramp": _logits(-1.0, 3.0)}
        rounded = {"disc": _logits(1.0, 2.25), "ramp": _logits(-1.0, 3.0)}

        quality = bn.measure_quality("f16", {"F32": 1}, reference, rounded)

        assert quality["worst"]["max_abs"] == pytest.approx(0.25)
        assert quality["worst"]["bit_exact"] is False
        assert quality["cases"]["ramp"]["bit_exact"] is True

    def test_the_case_sets_must_match(self):
        """Silently dropping a case present on one side only would erase its degradation from the
        record."""
        with pytest.raises(AssertionError, match="cases disagree"):
            bn.measure_quality("f16", {"F32": 1}, {"disc": _logits(1.0)}, {"ramp": _logits(1.0)})

    @pytest.mark.parametrize(
        ("census", "gate"), [({"F16": 1, "I64": 1}, "bit-exact"), ({"F32": 1}, "recorded")]
    )
    def test_the_expectation_comes_from_the_checkpoint_not_from_the_result(self, census, gate):
        """The expectation (lossless or not) comes from the upstream storage — deriving it from
        the rounded result would make the gate hold trivially."""
        reference = {"disc": _logits(1.0)}

        quality = bn.measure_quality("f16", census, reference, {"disc": _logits(1.5)})

        assert quality["gate"] == gate
        assert quality["checkpoint_dtypes"] == census


class TestQualityGate:
    def test_a_lossless_series_passes_when_every_case_is_bit_exact(self):
        reference = {"disc": _logits(1.0, -2.0), "ramp": _logits(0.5)}

        bn.assert_quality_gate(bn.measure_quality("f16", {"F16": 1}, reference, dict(reference)))

    def test_a_lossless_series_fails_on_any_moved_case(self):
        """Like HR, if even one case moved it is not published (catches a regression in where
        the rounding sits)."""
        reference = {"disc": _logits(1.0, -2.0), "ramp": _logits(0.5)}
        rounded = {"disc": _logits(1.0, -2.0), "ramp": _logits(0.5000001)}

        with pytest.raises(AssertionError, match=r"bit for bit: \['ramp'\]"):
            bn.assert_quality_gate(bn.measure_quality("f16", {"F16": 1}, reference, rounded))

    def test_the_lossy_limit_is_half_an_8bit_alpha_step(self):
        """The provisional limit is 0.5 step of the 8-bit α (the kill criterion of deep-dive E1
        §6, ADR 0113 decision 5)."""
        assert pytest.approx(0.5 / 255) == bn.ALPHA_MAE_LIMIT

    def test_a_lossy_series_below_the_alpha_limit_passes(self):
        """Move by 0.004 near logit 0 (α slope 1/4) — α moves about 1e-3 (about half the
        limit)."""
        reference = {"disc": _logits(0.0, 0.0), "ramp": _logits(0.0)}
        rounded = {"disc": _logits(0.004, 0.004), "ramp": _logits(0.0)}
        quality = bn.measure_quality("f16", {"F32": 1}, reference, rounded)
        assert 0.0 < quality["worst"]["alpha_mae"] < bn.ALPHA_MAE_LIMIT

        bn.assert_quality_gate(quality)

    def test_a_lossy_series_over_the_alpha_limit_fails(self):
        """Move by 0.02 in the same shape — α moves about 5e-3, over the limit (≈ 1.96e-3)."""
        reference = {"disc": _logits(0.0, 0.0), "ramp": _logits(0.0)}
        rounded = {"disc": _logits(0.02, 0.02), "ramp": _logits(0.0)}
        quality = bn.measure_quality("f16", {"F32": 1}, reference, rounded)
        assert quality["worst"]["alpha_mae"] > bn.ALPHA_MAE_LIMIT

        with pytest.raises(AssertionError, match=r"provisional line .*: \['disc'\]"):
            bn.assert_quality_gate(quality)

    def test_a_lossy_series_that_did_not_move_at_all_fails(self):
        """Control: a rounding that should be lossy moving not a single bit means the harness is
        looking at the same weights."""
        reference = {"disc": _logits(1.0, -2.0)}

        with pytest.raises(AssertionError, match="trivially true"):
            bn.assert_quality_gate(
                bn.measure_quality("f16", {"F32": 1}, reference, dict(reference))
            )


class TestExportSeriesQuality:
    """The wiring of `export_series`: the reference is taken before the rounding, and the quality
    gate runs before publishing.

    The heavy stages (loading the real weights, the Swin patches) are replaced with tiny ones
    ({@link tiny_load}).
    """

    #: `export_series` checks the resolution step (64) at the entry, so tiny runs at 64² too.
    SIZE = 64

    @pytest.fixture
    def wired(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_load):
        """A real run of `export_series` with provenance, census and real images replaced by
        tiny ones."""
        monkeypatch.setattr(bn, "upstream_provenance", lambda _dir: TINY_PROVENANCE)
        monkeypatch.setattr(bn, "checkpoint_dtypes", lambda _dir: {"F16": 5, "I64": 1})
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_real_images(corpus)
        # tiny cannot separate the `disc` circle or the real images' foreground (the
        # discriminating sanity check belongs to the real model), so the order checks are
        # removed (that the real-image order check **is called** is checked by its own test).
        monkeypatch.setattr(bn, "_sanity", lambda mattes: {"cases": sorted(mattes)})
        monkeypatch.setattr(bn, "_real_sanity", lambda ratios: dict(ratios))
        return tmp_path, corpus

    def test_f16_writes_the_quality_record_next_to_the_golden(self, wired):
        tmp_path, corpus = wired
        out_dir = tmp_path / "series-f16"

        summary = bn.export_series(
            Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
        )

        quality = json.loads((out_dir / bn.QUALITY_FILE).read_text(encoding="utf-8"))
        # The quality measurement has 8 cases, 4 synthetic + 4 real images, even without
        # `--real-images`.
        expected = sorted(
            [name for name, _ in bn.build_cases(self.SIZE)]
            + [name for name, _file, _why in bn.REAL_CASES]
        )
        assert sorted(quality["cases"]) == expected
        assert quality["gate"] == "bit-exact"
        assert quality["worst"]["bit_exact"] is True
        # The golden holds no real images (no `real_images`).
        assert all(not name.startswith(f"{bn.IO_PREFIX}photo-") for name in summary["io"])
        assert summary["dtype"] == "f16"
        assert summary["quality"]["gate"] == "bit-exact"
        assert summary["storage"]["compressed_tensors"] > 0

    def test_the_reference_is_taken_before_the_weights_are_rounded(self, wired, tiny_load):
        tmp_path, corpus = wired

        bn.export_series(
            Path("/nonexistent"),
            tmp_path / "series-f16",
            self.SIZE,
            corpus_root=corpus,
            dtype="f16",
        )

        # The first (the reference) is an f32 wrapper that skips the rounding; the second is
        # the rounding side.
        assert tiny_load == [
            "load",
            ("round", "f32"),
            "apply",
            "prepare",
            "load",
            ("round", "f16"),
            "apply",
            "prepare",
        ]

    def test_a_failed_quality_gate_publishes_nothing(self, wired, monkeypatch):
        """MUST: the gate comes before publishing — a failed run leaves no "asset that would pass
        the acceptance gate"."""
        tmp_path, corpus = wired
        # Fault injection: only the reference is taken with other weights (a difference shows up
        # although it is HR-like).
        real_reference = bn.reference_outputs
        monkeypatch.setattr(
            bn,
            "reference_outputs",
            lambda *args: {name: out + 1.0 for name, out in real_reference(*args).items()},
        )
        out_dir = tmp_path / "series-f16"

        with pytest.raises(AssertionError, match="does not match f32 bit for bit"):
            bn.export_series(
                Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
            )

        assert not out_dir.exists()

    def test_a_lossy_checkpoint_is_recorded_and_published(self, wired, monkeypatch):
        """The lossy wiring: a tiny with an F32 census and weights not representable in f16
        (like Lucida).

        The gate is `recorded` (no bit identity expected), the difference is inside the limit, so
        it is published, and `quality.json` keeps the bit mismatch.
        """
        tmp_path, corpus = wired
        monkeypatch.setattr(bn, "checkpoint_dtypes", lambda _dir: {"F32": 5, "I64": 1})
        monkeypatch.setattr(bn, "load_model", lambda _dir: _f32_checkpoint_model())
        out_dir = tmp_path / "series-f16"

        summary = bn.export_series(
            Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
        )

        quality = json.loads((out_dir / bn.QUALITY_FILE).read_text(encoding="utf-8"))
        assert quality["gate"] == "recorded"
        assert quality["lossless_expected"] is False
        assert quality["worst"]["bit_exact"] is False
        assert 0.0 < quality["worst"]["alpha_mae"] < bn.ALPHA_MAE_LIMIT
        assert summary["quality"]["gate"] == "recorded"

    def test_the_real_images_are_checked_without_real_images(self, wired, monkeypatch):
        """A compressed series applies the foreground-ratio order check to the rounded outputs of
        the 4 real images even without `--real-images`."""
        tmp_path, corpus = wired
        seen: list[set[str]] = []

        def _record(ratios):
            seen.append(set(ratios))
            return dict(ratios)

        monkeypatch.setattr(bn, "_real_sanity", _record)

        out_dir = tmp_path / "series-f16"

        summary = bn.export_series(
            Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
        )

        assert seen == [{name for name, _file, _why in bn.REAL_CASES}]
        assert "real_foreground" in summary["sanity"]

    def test_a_broken_real_image_order_publishes_nothing(self, wired, monkeypatch):
        tmp_path, corpus = wired

        def _broken(_ratios):
            raise AssertionError("the real images' foreground ratios are in the wrong order")

        monkeypatch.setattr(bn, "_real_sanity", _broken)
        out_dir = tmp_path / "series-f16"

        with pytest.raises(AssertionError, match="wrong order"):
            bn.export_series(
                Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f16"
            )

        assert not out_dir.exists()

    def test_f32_writes_no_quality_record(self, wired):
        tmp_path, corpus = wired
        out_dir = tmp_path / "series"

        summary = bn.export_series(
            Path("/nonexistent"), out_dir, self.SIZE, corpus_root=corpus, dtype="f32"
        )

        assert not (out_dir / bn.QUALITY_FILE).exists()
        assert summary["quality"] is None
        assert summary["storage"]["compressed_tensors"] == 0
