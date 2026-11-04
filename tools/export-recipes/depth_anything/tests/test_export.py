"""Script-level contracts of `depth_anything/export.py` (the part that needs no real weights).

The parts that need real weights (the IR's contents, equivalence before and after the patches)
are covered by running the script and by `depth_anything/tests/test_patch.py`. This file pins the
contracts that **can break even without real weights**:

- the series spelling (`outputs/series/<model name>/`) follows `--model-dir`
- the nature of the 4 golden cases (normalized, different from each other, square)
- sanity ({@link depth_anything.export._sanity}) is **not vacuous** — it really rejects uniform
  outputs, input-independent outputs and outputs that do not follow the geometry
- the real-image sanity ({@link depth_anything.export._real_sanity}) is not vacuous either — it
  really rejects outputs where the near and far regions are equal / reversed
- identification of the preprocessing source of truth
  ({@link depth_anything.export.check_processor}) — starting with **resample 3 = BICUBIC**,
  shifting a single field makes it fail
- writing the golden io ({@link depth_anything.export._write_io}) — a **tiny synthetic weight
  set** is actually exported and run through the key convention and the "the output is one depth
  map" check (no real weights needed)
- `--verify` does not emit (in one process the class-attribute replacements contaminate the
  reference)
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from upstream_fixture import OTHER_REVISION, write_snapshot

from _shared.container_read import read_provenance
from _shared.paths import SERIES_ROOT
from _shared.upstream import UpstreamProvenanceError
from depth_anything import export as da
from karume.container import Provenance
from karume.dist import NOTICE_FILENAME
from karume.pipeline import export_to_file

#: The provenance baked into the tiny export (its value is not compared — deriving the script's
#: provenance is the job of `_shared.upstream`).
TINY_PROVENANCE = Provenance(license="fixture")

#: The resolution of the synthetic goldens (the real weights' 518 is heavy, so it is small here,
#: where only the nature of the cases is checked).
SMALL = 28

#: The resolution of the tiny synthetic model (it only runs one export, so the minimum is
#: enough).
TINY_SIZE = 6

#: The 2 cases fed to the tiny synthetic weights (`_write_io` looks only at the names and the
#: count).
TINY_CASES = (
    ("disc", torch.linspace(-1.0, 1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
    ("ramp", torch.linspace(1.0, -1.0, 3 * TINY_SIZE * TINY_SIZE).reshape(1, 3, TINY_SIZE, -1)),
)


class TinyDepth(nn.Module):
    """The minimal skeleton of {@link depth_anything.export.DepthMap} (`[B,3,S,S] → [B,S,S]`)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.conv(pixel_values).squeeze(1)


class TwoOutputDepth(TinyDepth):
    """A form with 2 outputs (to confirm that `_write_io` rejects it)."""

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth = super().forward(pixel_values)
        return depth, depth * 2.0


@pytest.fixture
def exported(tmp_path):
    """Export one tiny wrapper and return `(wrapper, graph, out_dir)`."""
    torch.manual_seed(0)
    wrapper = TinyDepth()
    graph = export_to_file(
        wrapper,
        (TINY_CASES[0][1],),
        tmp_path / da.MODEL_FILE,
        provenance=TINY_PROVENANCE,
        graph_name="tiny",
        symbol_names=(),
    )
    return wrapper, graph, tmp_path


def _depth_from(plane: torch.Tensor) -> torch.Tensor:
    """Turn an `[H, W]` plane into a depth map `[1, H, W]` (non-negative)."""
    return (plane - plane.min()).unsqueeze(0)


def _depths(ramp_weight: float, other_weight: float) -> dict[str, torch.Tensor]:
    """Synthetic outputs for the 4 cases in which only `ramp` correlates strongly with the
    diagonal ramp.

    Raising `other_weight` moves the largest correlation away from `ramp` (the axis that shows
    the sanity discrimination really works).
    """
    plane = da.ramp_plane(SMALL)
    generator = torch.Generator().manual_seed(3)
    noise = torch.rand((4, SMALL, SMALL), generator=generator)
    return {
        "disc": _depth_from(noise[0]),
        "ramp": _depth_from(ramp_weight * plane + noise[1]),
        "checker": _depth_from(other_weight * plane + noise[2]),
        "noise": _depth_from(noise[3]),
    }


def _real_depths(near: float, far: float) -> dict[str, torch.Tensor]:
    """Synthetic depths for the 4 real-image cases (maps with `near` in the near rectangle and
    `far` in the far one).

    Swapping `near` and `far` flips the discrimination, and making them equal makes it uniform —
    those 2 axes show that the sanity check is not vacuous. The background constant is shifted per
    case so that `_assert_distinct` (identical maps mean the input is not getting through) does
    not trip.
    """
    depths: dict[str, torch.Tensor] = {}
    for index, (name, regions) in enumerate(sorted(da.REAL_REGIONS.items())):
        depth = torch.full((1, SMALL, SMALL), 0.01 * (index + 1))
        for label, top, bottom, left, right in regions:
            value = near if label == regions[0][0] else far
            depth[
                :,
                int(top * SMALL) : int(bottom * SMALL),
                int(left * SMALL) : int(right * SMALL),
            ] = value
        depths[name] = depth
    return depths


class TestSeriesLayout:
    def test_the_series_name_is_the_model_directory_name(self) -> None:
        """The series name is the model's directory name (lower-cased) — the resolution is not an
        axis, so it is not spelled."""
        assert da.default_out_dir(da.DEFAULT_MODEL_DIR) == (
            SERIES_ROOT / "depth-anything-v2-small-hf"
        )

    def test_each_model_gets_its_own_series(self) -> None:
        """The size axis (Small / Base / Large) is separated by the name — a shared location would
        erase the earlier assets."""
        names = {
            (da.MODELS_ROOT / f"Depth-Anything-V2-{size}-hf").name: da.default_out_dir(
                da.MODELS_ROOT / f"Depth-Anything-V2-{size}-hf"
            )
            for size in ("Small", "Base", "Large")
        }

        assert len(set(names.values())) == 3
        assert names["Depth-Anything-V2-Base-hf"] == SERIES_ROOT / "depth-anything-v2-base-hf"

    def test_the_graph_input_name_matches_the_wrapper_signature(self) -> None:
        """`INPUT_NAME` is the wrapper's argument name itself (the premise that makes the
        comparison after export meaningful)."""
        parameters = list(inspect.signature(da.DepthMap.forward).parameters)
        assert parameters == ["self", da.INPUT_NAME]


class TestCases:
    def test_the_four_cases_are_normalized_images_of_the_requested_size(self) -> None:
        """All 4 cases are `[1, 3, S, S]` and return to `[0, 1]` images when denormalized."""
        mean = torch.tensor(da.IMAGENET_MEAN).reshape(1, 3, 1, 1)
        std = torch.tensor(da.IMAGENET_STD).reshape(1, 3, 1, 1)
        cases = da.build_cases(SMALL)

        assert [name for name, _ in cases] == ["disc", "ramp", "checker", "noise"]
        for name, pixel_values in cases:
            assert tuple(pixel_values.shape) == (1, 3, SMALL, SMALL), name
            raw = pixel_values * std + mean
            assert float(raw.min()) >= -1e-6, name
            assert float(raw.max()) <= 1.0 + 1e-6, name

    def test_the_cases_differ_from_each_other(self) -> None:
        """If the same image appeared twice, the output distinctness check (sanity) would become
        vacuous."""
        images = [pixel_values for _name, pixel_values in da.build_cases(SMALL)]
        for index, left in enumerate(images):
            for right in images[index + 1 :]:
                assert not torch.equal(left, right)

    def test_the_noise_case_does_not_depend_on_the_global_seed(self) -> None:
        """`noise` is made with its own Generator (the global seed does not move the golden)."""
        torch.manual_seed(1)
        first = da.build_cases(SMALL)[3][1]
        torch.manual_seed(2)
        assert torch.equal(first, da.build_cases(SMALL)[3][1])


class TestSanity:
    def test_a_geometry_following_output_passes(self) -> None:
        report = da._sanity(_depths(ramp_weight=4.0, other_weight=0.0))

        assert report["ramp_correlation"]["ramp"] > report["ramp_correlation"]["checker"]
        assert set(report["depth_range"]) == {"disc", "ramp", "checker", "noise"}

    def test_identical_outputs_are_rejected(self) -> None:
        """Rejects the form where the input has no effect (2 cases with identical outputs)."""
        depths = _depths(ramp_weight=4.0, other_weight=0.0)
        depths["checker"] = depths["noise"]

        with pytest.raises(AssertionError, match="the input has no effect"):
            da._sanity(depths)

    def test_a_uniform_output_is_rejected(self) -> None:
        """Rejects a uniformly collapsed output (min == max)."""
        depths = _depths(ramp_weight=4.0, other_weight=0.0)
        depths["disc"] = torch.full((1, SMALL, SMALL), 2.5)

        with pytest.raises(AssertionError, match="uniform"):
            da._sanity(depths)

    def test_a_negative_output_is_rejected(self) -> None:
        """Rejects negative values that contradict the ReLU at the end of the head (a miswired
        head)."""
        depths = _depths(ramp_weight=4.0, other_weight=0.0)
        depths["noise"] = depths["noise"] - 1.0

        with pytest.raises(AssertionError, match="ReLU"):
            da._sanity(depths)

    def test_an_output_that_does_not_follow_the_geometry_is_rejected(self) -> None:
        """Rejects the form where a case without a monotonic cue correlates more strongly (proof
        that the discrimination works)."""
        with pytest.raises(AssertionError, match="geometry"):
            da._sanity(_depths(ramp_weight=0.0, other_weight=4.0))


class TestWriteIo:
    """Writing goldens with tiny synthetic weights (the only export path that runs without real
    weights)."""

    def test_writes_one_file_per_case_with_the_declared_keys(self, exported) -> None:
        wrapper, graph, out_dir = exported

        written, depths = da._write_io(wrapper, graph, TINY_CASES, out_dir)

        assert written == [f"{da.IO_PREFIX}{name}{da.IO_SUFFIX}" for name, _ in TINY_CASES]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{da.INPUT_PREFIX}{da.INPUT_NAME}", f"{da.OUTPUT_PREFIX}0"}
        assert tuple(tensors[f"{da.INPUT_PREFIX}{da.INPUT_NAME}"].shape) == (
            1,
            3,
            TINY_SIZE,
            TINY_SIZE,
        )
        assert tuple(depths["disc"].shape) == (1, TINY_SIZE, TINY_SIZE)

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path) -> None:
        """MUST: the output is one depth map — a second one would silently shift io's positional
        convention."""
        torch.manual_seed(0)
        wrapper = TwoOutputDepth()
        graph = export_to_file(
            wrapper,
            (TINY_CASES[0][1],),
            tmp_path / da.MODEL_FILE,
            provenance=TINY_PROVENANCE,
            graph_name="tiny",
            symbol_names=(),
        )

        with pytest.raises(AssertionError, match="exactly one depth map"):
            da._write_io(wrapper, graph, TINY_CASES, tmp_path)


class TestVerifyCli:
    def test_verify_does_not_emit(self, monkeypatch) -> None:
        """MUST: not combined with emit in one process (the class replacements contaminate the
        reference)."""
        seen: list[str] = []
        monkeypatch.setattr(
            da,
            "verify_patches",
            lambda _dir: (
                seen.append("verify")
                or [{"stage": "layout", "claim": "bit-exact", "bit_exact": True, "maxdiff": {}}]
            ),
        )
        monkeypatch.setattr(da, "export_series", lambda *_a, **_kw: pytest.fail("--verify emitted"))

        da.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            da, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(da, "verify_patches", lambda *_a: pytest.fail("the emit ran --verify"))

        da.main([])

        assert seen == ["emit"]

    def test_the_output_dir_follows_the_model_dir(self, monkeypatch) -> None:
        """MUST: without `--out` the series follows the size axis (a fixed location would be
        overwritten)."""
        seen: list[Path] = []
        monkeypatch.setattr(
            da, "export_series", lambda _dir, out, **_kw: seen.append(out) or {"dir": str(out)}
        )

        da.main(["--model-dir", "/tmp/Depth-Anything-V2-Large-hf"])

        assert seen == [SERIES_ROOT / "depth-anything-v2-large-hf"]

    @pytest.mark.parametrize(("argv", "expected"), [([], False), (["--real-images"], True)])
    def test_real_images_reaches_the_series(self, monkeypatch, argv, expected) -> None:
        """`--real-images` reaches the emit (otherwise the real-image goldens would silently not
        be written)."""
        seen: list[bool] = []
        monkeypatch.setattr(
            da,
            "export_series",
            lambda _dir, _out, real_images=False: seen.append(real_images) or {"dir": "x"},
        )

        da.main(argv)

        assert seen == [expected]


class TestRealRegions:
    """The real-image discrimination (near region > far region, as the composition says)."""

    def test_every_real_case_has_a_near_far_pair(self) -> None:
        # A case missing its pair fails here by name, before it becomes a KeyError in
        # `_real_sanity`.
        assert set(da.REAL_REGIONS) == {name for name, _file, _why in da.REAL_CASES}
        for name, regions in da.REAL_REGIONS.items():
            assert len(regions) == 2, name
            for _label, top, bottom, left, right in regions:
                assert 0.0 <= top < bottom <= 1.0 and 0.0 <= left < right <= 1.0, name

    def test_the_near_region_is_lower_in_the_frame_than_the_far_region(self) -> None:
        # The claim of the composition itself (the near side is in the lower part of the frame).
        # Rewriting the rectangles so that the claim flips makes this fail.
        for name, (
            (_near, near_top, _nb, _nl, _nr),
            (_far, far_top, *_rest),
        ) in da.REAL_REGIONS.items():
            assert near_top > far_top, name

    def test_a_near_far_ordering_passes(self) -> None:
        report = da._real_sanity(_real_depths(near=3.0, far=0.5))

        assert set(report) == set(da.REAL_REGIONS)
        assert report["photo-corridor"]["floor"] > report["photo-corridor"]["vanishing-point"]

    def test_a_flipped_ordering_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="near/far"):
            da._real_sanity(_real_depths(near=0.5, far=3.0))

    def test_a_uniform_output_is_rejected(self) -> None:
        # A uniform map gives "near <= far", and a vacuous sanity check would let it through
        # here.
        with pytest.raises(AssertionError):
            da._real_sanity(_real_depths(near=1.0, far=1.0))

    def test_an_empty_region_fails_loudly(self) -> None:
        tiny = torch.zeros((1, 4, 4))
        with pytest.raises(AssertionError, match="became empty"):
            da._region_mean(tiny, ("sliver", 0.10, 0.12, 0.0, 1.0))


def _processor(**overrides):
    """A `DPTImageProcessor` with overridden defaults (skipped where transformers is missing)."""
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    processor = transformers.DPTImageProcessor(
        resample=da.EXPECTED_RESAMPLE,
        image_mean=list(da.IMAGENET_MEAN),
        image_std=list(da.IMAGENET_STD),
        rescale_factor=da.EXPECTED_RESCALE,
        size={"height": 518, "width": 518},
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
    )
    for key, value in overrides.items():
        setattr(processor, key, value)
    return processor


class TestCheckProcessor:
    def test_the_expected_shape_passes(self) -> None:
        shape = da.check_processor(_processor())

        assert shape["resample"] == 3
        assert shape["ensure_multiple_of"] == 14

    def test_bilinear_is_rejected(self) -> None:
        # The crux. `resample` is a PIL constant, 2 = BILINEAR / 3 = BICUBIC, and with SigLIP2's
        # 2 the TS side would stay green while matching "a different filter than the reference".
        with pytest.raises(SystemExit, match="resample"):
            da.check_processor(_processor(resample=2))

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("image_mean", [0.5, 0.5, 0.5], "mean/std"),
            ("image_std", [0.5, 0.5, 0.5], "mean/std"),
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
            da.check_processor(_processor(**{key: value}))

    def test_the_pil_backend_is_rejected(self) -> None:
        pytest.importorskip("torchvision")
        transformers = pytest.importorskip("transformers")
        with pytest.raises(SystemExit, match="TorchvisionBackend"):
            da.check_processor(transformers.DPTImageProcessorPil())


class TestRealCases:
    def test_a_missing_image_stops_before_the_heavy_imports(self, tmp_path) -> None:
        """MUST: `--real-images` is an explicit request — it does not silently write 3 images."""
        with pytest.raises(SystemExit, match="demo:eval-images"):
            da.build_real_cases(da.DEFAULT_MODEL_DIR, 518, tmp_path)


class TestRealCasesMetadata:
    """The `__metadata__` of real-image goldens (identifying the source image) and the input-shape
    check."""

    @staticmethod
    def _real_array(index: int) -> torch.Tensor:
        """The pixels of one tiny real image (`[S, S, 3]` u8). The values shift with the case and
        the channel."""
        array = torch.zeros((8, 8, 3), dtype=torch.uint8)
        for channel in range(3):
            array[:, :, channel] = 10 * index + 30 * channel + 7
        return array

    @classmethod
    def _write_real_images(cls, root: Path) -> dict[str, bytes]:
        """Write tiny PNGs under the `REAL_CASES` names and return case name → raw bytes."""
        image_module = pytest.importorskip("PIL.Image")
        written: dict[str, bytes] = {}
        for index, (name, file_name, _why) in enumerate(da.REAL_CASES):
            path = root / file_name
            image_module.fromarray(cls._real_array(index).numpy()).save(path)
            written[name] = path.read_bytes()
        return written

    @staticmethod
    def _install_processor(monkeypatch, processor) -> None:
        """Replace the `AutoImageProcessor` that `build_real_cases` uses with a synthetic
        processor."""
        transformers = pytest.importorskip("transformers")
        monkeypatch.setattr(
            transformers.AutoImageProcessor, "from_pretrained", staticmethod(lambda _dir: processor)
        )

    def test_it_records_the_source_image_and_its_digest(self, monkeypatch, tmp_path) -> None:
        """MUST: the fields that make an environment fail before the comparison when the images
        were baked again but the goldens were not retaken.

        The TS-side `packages/models/tests/e2e_depth_anything_real_test.ts` compares these 2
        fields, so missing fields would silently make the comparison itself vacuous.
        """
        raw = self._write_real_images(tmp_path)
        self._install_processor(monkeypatch, _processor())

        cases = da.build_real_cases(da.DEFAULT_MODEL_DIR, 518, tmp_path)

        files = {name: file for name, file, _why in da.REAL_CASES}
        assert [name for name, _pixels, _md in cases] == list(files)
        for name, pixel_values, metadata in cases:
            assert tuple(pixel_values.shape) == (1, 3, 518, 518), name
            assert metadata[da.SOURCE_IMAGE_KEY] == files[name], name
            assert metadata[da.SOURCE_SHA256_KEY] == hashlib.sha256(raw[name]).hexdigest(), name

    def test_a_processor_that_returns_another_size_fails_loudly(
        self, monkeypatch, tmp_path
    ) -> None:
        """If the preprocessing's output shape differs from the graph's input shape, it fails
        without writing a single image."""
        self._write_real_images(tmp_path)
        self._install_processor(monkeypatch, _processor(size={"height": 28, "width": 28}))

        with pytest.raises(AssertionError, match="keep_aspect_ratio"):
            da.build_real_cases(da.DEFAULT_MODEL_DIR, 518, tmp_path)


class TestSeriesNameAgreement:
    def test_the_distribution_looks_for_the_same_series_name(self) -> None:
        """MUST: the series name is spelled **the same on the writing and the reading side**.

        Today both sides agree just by lower-casing, but the day one of them gains a `_` → `-`
        replacement, nobody would notice the spelling drifting so that the assembly picks up
        **another model's series** (a port of the birefnet test of the same name).
        """
        from depth_anything.card import DEPTH_ANYTHING_UPSTREAM
        from depth_anything.distribution import depth_anything_series_name

        for model, repo in DEPTH_ANYTHING_UPSTREAM.items():
            model_dir = da.MODELS_ROOT / repo.split("/", 1)[1]
            assert da.default_out_dir(model_dir).name == depth_anything_series_name(model)


class _RecordingDepth(nn.Module):
    """A skeleton that only records the step of each call (reference / step 1 / step 2).

    The output is determined by the input (the same in every step), so the bit-match assert of
    `verify_patches`' first step passes — all this wants to see is the **order**.
    """

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls
        self.stage = "reference"

    def forward(self, pixel_values: torch.Tensor):
        if not self.calls or self.calls[-1] != self.stage:
            self.calls.append(self.stage)
        return SimpleNamespace(predicted_depth=pixel_values.sum(dim=1))


class TestVerifyOrder:
    """The order invariant of `verify_patches` (reference → step 1 → step 2)."""

    def test_a_patched_process_cannot_take_the_reference(self, monkeypatch) -> None:
        """MUST: no reference is taken in a process with patches applied (difference 0 would make
        the check vacuous)."""
        monkeypatch.setattr(da.patch, "patches_applied", lambda: True)

        with pytest.raises(SystemExit, match="vacuous"):
            da.verify_patches(Path("/nonexistent"))

    def test_the_reference_is_taken_once_before_any_patch(self, monkeypatch) -> None:
        """Catches the regression of retaking the reference per step (the second step's reference
        would be a patched value)."""
        calls: list[str] = []
        recorder = _RecordingDepth(calls)
        monkeypatch.setattr(da.patch, "patches_applied", lambda: False)
        monkeypatch.setattr(da, "load_model", lambda _dir: recorder)
        monkeypatch.setattr(da.patch, "pretrained_resolution", lambda _model: SMALL)

        def _apply_layout(model: nn.Module) -> None:
            calls.append("apply_layout")
            model.stage = "layout"

        def _apply_modules(model: nn.Module) -> None:
            calls.append("apply_modules")
            model.stage = "modules"

        monkeypatch.setattr(da.patch, "apply_layout_patches", _apply_layout)
        monkeypatch.setattr(da.patch, "apply_module_patches", _apply_modules)

        entries = da.verify_patches(Path("/nonexistent"))

        assert calls == ["reference", "apply_layout", "layout", "apply_modules", "modules"]
        assert [entry["stage"] for entry in entries] == ["layout", "modules"]


class TestStagedPublication:
    """MUST: put in place only after every check passes (a failed run disappears with its staging
    location)."""

    @staticmethod
    def _stage_tiny(monkeypatch, tmp_path: Path) -> Path:
        """Set things up so `export_series` runs once without real weights, and return the
        location of the fake checkpoint."""
        torch.manual_seed(0)
        wrapper = TinyDepth()
        monkeypatch.setattr(da, "load_wrapper", lambda _dir: (wrapper, TINY_SIZE))
        monkeypatch.setattr(da, "build_cases", lambda _resolution: TINY_CASES)
        monkeypatch.setattr(
            da,
            "build_real_cases",
            lambda *_a, **_kw: (
                (
                    "photo-portrait",
                    TINY_CASES[0][1],
                    {da.SOURCE_IMAGE_KEY: "portrait.png", da.SOURCE_SHA256_KEY: "0" * 64},
                ),
            ),
        )
        monkeypatch.setattr(da, "_sanity", lambda _depths: {"depth_range": {}})
        model_dir = tmp_path / "Depth-Anything-V2-Small-hf"
        write_snapshot(model_dir, license="apache-2.0")
        return model_dir

    def test_a_passing_run_leaves_the_series_in_place(self, monkeypatch, tmp_path) -> None:
        """The counterpart that shows the check is not vacuous (when the checks pass, the series
        is put in place) — without it the claims below would be vacuous."""
        model_dir = self._stage_tiny(monkeypatch, tmp_path)
        monkeypatch.setattr(da, "_real_sanity", lambda _depths: {"photo-portrait": {}})
        out_dir = tmp_path / "series"

        summary = da.export_series(model_dir, out_dir, real_images=True)

        assert out_dir.is_dir()
        assert "real_sanity" in summary

    def test_a_base_checkpoint_names_its_own_license(self, monkeypatch, tmp_path) -> None:
        """MUST: the container's provenance comes from the actual `--model-dir` — Base states
        CC BY-NC 4.0.

        Baking a constant (the Apache-2.0 of the attribution table) would make Base's container
        state Small's licence.
        """
        model_dir = self._stage_tiny(monkeypatch, tmp_path)
        write_snapshot(model_dir, license="cc-by-nc-4.0", revision=OTHER_REVISION)
        out_dir = tmp_path / "series"

        da.export_series(model_dir, out_dir)

        assert read_provenance(out_dir / da.MODEL_FILE) == Provenance(
            license="cc-by-nc-4.0", notice=NOTICE_FILENAME, upstream_revision=OTHER_REVISION
        )

    def test_a_checkpoint_without_the_download_record_is_refused_before_the_weights(
        self, monkeypatch, tmp_path
    ) -> None:
        """A checkpoint that cannot state its provenance fails before the weights are read
        (nothing is put in place)."""
        monkeypatch.setattr(
            da,
            "load_wrapper",
            lambda _dir: pytest.fail("read the weights before the provenance check"),
        )
        out_dir = tmp_path / "series"

        with pytest.raises(UpstreamProvenanceError, match="is missing"):
            da.export_series(tmp_path / "Depth-Anything-V2-Small-hf", out_dir)

        assert not out_dir.exists()

    def test_a_failing_real_image_gate_leaves_nothing_behind(self, monkeypatch, tmp_path) -> None:
        """When the real-image near/far check fails, `out_dir` does not exist (no evaluation after
        putting it in place)."""
        model_dir = self._stage_tiny(monkeypatch, tmp_path)

        def _reject(_depths):
            raise AssertionError("the near/far ordering is reversed")

        monkeypatch.setattr(da, "_real_sanity", _reject)
        out_dir = tmp_path / "series"

        with pytest.raises(AssertionError, match="near/far"):
            da.export_series(model_dir, out_dir, real_images=True)

        assert not out_dir.exists()
