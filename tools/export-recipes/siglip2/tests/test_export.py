"""Script-level contracts of `siglip2/export.py` (the part that needs no real weights).

The emit with real weights is manual (the same discipline as the existing tests of
`deberta/export.py` / `embeddinggemma/export.py`). What is pinned here is only the discipline whose
breakage would give a **false PASS**:

- the graph has **exactly one output, pooler_output** (a second one would silently shift the
  positional convention of io)
- the golden synthetic images **differ from one another** (using the same image twice makes the
  cosine-order sanity check vacuous)
- `_sanity` checks the **order** of cosines (both the norm and "the cosine of an input with itself"
  are vacuous)
- `--verify` does not emit (in one process the reference gets contaminated)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from upstream_fixture import FIXTURE_REVISION, write_snapshot

from _shared.container_read import read_provenance
from _shared.paths import SERIES_ROOT
from karume.container import Provenance
from karume.dist import NOTICE_FILENAME
from karume.pipeline import export_to_file
from siglip2 import export as sg

#: Provenance baked into the tiny export (the values are not compared — deriving the script's
#: provenance is the job of `_shared.upstream`).
TINY_PROVENANCE = Provenance(license="fixture")


class TinyVisionPooler(nn.Module):
    """Minimal skeleton of `VisionPooler` (`[B,3,S,S] → [B,H]`, down to the parameter name)."""

    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.head = nn.Linear(4, 4)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        return self.head(torch.sum(patches, dim=1))


class TwoOutputPooler(TinyVisionPooler):
    """A form with two outputs (to confirm that `_write_io` refuses it)."""

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = super().forward(pixel_values)
        return pooled, pooled * 2.0


class Config:
    """Only the fields of `SiglipVisionConfig` that `build_cases` reads."""

    num_channels = 3
    image_size = 32
    patch_size = 4


CASES = (
    ("case0", torch.linspace(-1.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)),
    ("case1", torch.linspace(1.0, -1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)),
)

#: Minimal outputs that pass the synthetic side of `_sanity` (a base for varying only the
#: real-image side).
SYNTHETIC_POOLED = {
    "ramp": torch.tensor([[1.0, 0.0]]),
    "ramp-dim": torch.tensor([[0.9, 0.1]]),
    "checker": torch.tensor([[0.0, 1.0]]),
    "noise": torch.tensor([[0.5, 0.5]]),
}


def _real_pooled(street: list[float]) -> dict[str, torch.Tensor]:
    """Outputs of the 4 real-image cases (only `photo-street` varies — whether it is closer to the
    person side or the scene side)."""
    return {
        "photo-portrait": torch.tensor([[1.0, 0.0]]),
        "photo-street": torch.tensor([street]),
        "photo-landscape": torch.tensor([[0.0, 1.0]]),
        "photo-corridor": torch.tensor([[0.1, 0.99]]),
    }


@pytest.fixture
def exported(tmp_path):
    """Export one tiny wrapper and return `(wrapper, graph, out_dir)`."""
    torch.manual_seed(0)
    wrapper = TinyVisionPooler()
    graph = export_to_file(
        wrapper,
        (CASES[0][1],),
        tmp_path / sg.MODEL_FILE,
        provenance=TINY_PROVENANCE,
        graph_name="tiny",
        symbol_names=(),
    )
    return wrapper, graph, tmp_path


class TestSeriesLayout:
    def test_the_default_output_dir_is_a_series(self):
        """Series output goes under `outputs/series/` (not the distribution's `models/` —
        _shared.paths)."""
        assert sg.default_out_dir(sg.DEFAULT_MODEL_DIR).parent == SERIES_ROOT

    def test_each_model_gets_its_own_series(self):
        """MUST: a separate series per model — a shared spelling would silently overwrite the
        earlier assets."""
        other = sg.MODELS_ROOT / "siglip2-so400m-patch14-384"

        assert sg.default_out_dir(other) != sg.default_out_dir(sg.DEFAULT_MODEL_DIR)
        assert sg.default_out_dir(other).name == other.name


class TestWriteIo:
    def test_writes_one_file_per_case_with_the_declared_keys(self, exported):
        wrapper, graph, out_dir = exported

        written, pooled = sg._write_io(wrapper, graph, CASES, out_dir)

        assert written == [f"{sg.IO_PREFIX}{name}{sg.IO_SUFFIX}" for name, _ in CASES]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{sg.INPUT_PREFIX}{sg.INPUT_NAME}", f"{sg.OUTPUT_PREFIX}0"}
        assert tuple(pooled["case0"].shape) == (1, 4)

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path):
        """MUST: exactly one output, pooler_output — a second one would silently shift the
        positional convention of io."""
        torch.manual_seed(0)
        wrapper = TwoOutputPooler()
        graph = export_to_file(
            wrapper,
            (CASES[0][1],),
            tmp_path / sg.MODEL_FILE,
            provenance=TINY_PROVENANCE,
            graph_name="tiny",
            symbol_names=(),
        )

        with pytest.raises(AssertionError, match="pooler_output is exactly one"):
            sg._write_io(wrapper, graph, CASES, tmp_path)


class TestGoldenCases:
    def test_cases_are_normalized_pixel_values_of_the_configured_shape(self):
        cases = sg.build_cases(Config())

        for name, pixel_values in cases:
            assert tuple(pixel_values.shape) == (1, 3, 32, 32), name
            assert pixel_values.dtype is torch.float32, name
            assert float(pixel_values.abs().max()) <= 1.0, name

    def test_every_case_is_a_different_image(self):
        """MUST: never use the same image twice — the cosine-order sanity check would become
        vacuous."""
        cases = sg.build_cases(Config())

        for index, (name, pixel_values) in enumerate(cases):
            for other_name, other in cases[index + 1 :]:
                assert not torch.equal(pixel_values, other), (
                    f"{name} and {other_name} are identical"
                )

    def test_the_sanity_pairs_name_existing_cases(self):
        names = {name for name, _ in sg.build_cases(Config())}

        assert set(sg.NEAR_PAIR) | set(sg.FAR_PAIR) <= names


class TestRealCases:
    def test_every_case_names_a_distinct_image(self):
        """MUST: neither case names nor file names repeat (using the same image twice makes the
        discrimination check vacuous)."""
        names = [name for name, _file, _why in sg.REAL_CASES]
        files = [file for _name, file, _why in sg.REAL_CASES]

        assert len(set(names)) == len(names)
        assert len(set(files)) == len(files)

    def test_case_names_do_not_collide_with_the_synthetic_ones(self):
        """The goldens share one directory, so a name collision would silently lose one of
        them."""
        synthetic = {name for name, _ in sg.build_cases(Config())}

        assert synthetic.isdisjoint({name for name, _file, _why in sg.REAL_CASES})

    def test_the_discrimination_groups_name_existing_cases(self):
        names = {name for name, _file, _why in sg.REAL_CASES}

        assert set(sg.REAL_PERSON_CASES) | set(sg.REAL_SCENE_CASES) <= names

    def test_a_missing_image_fails_loudly_before_anything_heavy(self, tmp_path):
        """MUST: do not silently skip a missing image (if the goldens were written with 3 images,
        the e2e side would notice the gap)."""
        with pytest.raises(SystemExit, match="demo:eval-images"):
            sg.build_real_cases(tmp_path, Config(), tmp_path)


class TestIoMetadata:
    def test_real_cases_carry_the_source_image_digest(self, exported):
        """Real-image goldens identify their source image (to catch the mistake of re-baking the
        images without re-taking the goldens)."""
        wrapper, graph, out_dir = exported
        digest = "f" * 64
        metadata = {"case0": {sg.SOURCE_IMAGE_KEY: "a.png", sg.SOURCE_SHA256_KEY: digest}}

        written, _pooled = sg._write_io(wrapper, graph, CASES, out_dir, metadata)

        with safe_open(str(out_dir / written[0]), framework="pt") as opened:
            assert opened.metadata() == {sg.SOURCE_IMAGE_KEY: "a.png", sg.SOURCE_SHA256_KEY: digest}
        with safe_open(str(out_dir / written[1]), framework="pt") as opened:
            assert opened.metadata() is None


class TestSanity:
    def test_passes_when_the_near_pair_is_closer(self):
        pooled = {
            "ramp": torch.tensor([[1.0, 0.0]]),
            "ramp-dim": torch.tensor([[0.9, 0.1]]),
            "checker": torch.tensor([[0.0, 1.0]]),
            "noise": torch.tensor([[0.5, 0.5]]),
        }

        result = sg._sanity(pooled)

        assert result["cosine"][f"{sg.NEAR_PAIR[0]}×{sg.NEAR_PAIR[1]}"] > 0.9

    def test_fails_loudly_when_the_order_is_inverted(self):
        """If the pair with similar structure is farther apart than the far pair, the embedding is
        broken."""
        pooled = {
            "ramp": torch.tensor([[1.0, 0.0]]),
            "ramp-dim": torch.tensor([[0.0, 1.0]]),
            "checker": torch.tensor([[0.9, 0.1]]),
            "noise": torch.tensor([[0.5, 0.5]]),
        }

        with pytest.raises(AssertionError, match="the cosine order contradicts the structure"):
            sg._sanity(pooled)

    def test_real_images_are_judged_only_when_they_are_present(self):
        """An emit with only the 4 synthetic cases (the default) also passes the sanity check (the
        real images are an additional group)."""
        assert "real_cosine" not in sg._sanity(dict(SYNTHETIC_POOLED))

    def test_real_images_pass_when_the_two_people_are_closest(self):
        pooled = {**SYNTHETIC_POOLED, **_real_pooled(street=[0.99, 0.1])}

        result = sg._sanity(pooled)

        assert result["real_cosine"][f"{sg.REAL_PERSON_CASES[0]}×{sg.REAL_PERSON_CASES[1]}"] > 0.9

    def test_real_images_fail_loudly_when_a_scene_is_closer(self):
        """MUST: if a person × scene pair is closer than person × person, the embedding does not
        capture the meaning."""
        pooled = {**SYNTHETIC_POOLED, **_real_pooled(street=[0.1, 0.99])}

        with pytest.raises(AssertionError, match="the real-image cosine order is inverted"):
            sg._sanity(pooled)

    def test_fails_loudly_when_every_case_collapses_to_one_point(self):
        """MUST: an embedding collapsed to one point also fails (the near and the far pair become
        equal)."""
        vector = torch.tensor([[0.6, 0.8]])
        pooled = dict.fromkeys(("ramp", "ramp-dim", "checker", "noise"), vector)

        with pytest.raises(AssertionError, match="the cosine order contradicts the structure"):
            sg._sanity(pooled)


class TestVerifyCli:
    def test_verify_does_not_emit(self, monkeypatch):
        """MUST: never combined with an emit in one process (the class replacement contaminates
        the reference)."""
        seen: list[str] = []
        monkeypatch.setattr(
            sg,
            "verify_patches",
            lambda _dir: (
                seen.append("verify")
                or [
                    {"stage": "shape-folds", "claim": "bit-exact", "bit_exact": True, "maxdiff": {}}
                ]
            ),
        )
        monkeypatch.setattr(sg, "export_series", lambda *_a, **_kw: pytest.fail("--verify emitted"))

        sg.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            sg, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(sg, "verify_patches", lambda _dir: pytest.fail("the emit ran --verify"))

        sg.main([])

        assert seen == ["emit"]

    def test_the_output_dir_follows_the_model_dir(self, monkeypatch):
        """MUST: without `--out` the series follows `--model-dir` (a fixed one would be
        overwritten)."""
        seen: list[Path] = []
        monkeypatch.setattr(
            sg, "export_series", lambda _dir, out, **_kw: seen.append(out) or {"dir": str(out)}
        )

        sg.main(["--model-dir", "/tmp/siglip2-so400m-patch14-384"])

        assert seen == [SERIES_ROOT / "siglip2-so400m-patch14-384"]


class _RecordingVision(nn.Module):
    """Skeleton that only records the step of each call (reference / step 1 / step 2).

    `verify_patches` touches only `model(pixel_values=…).pooler_output` and the two attributes
    `model.embeddings` / `model.head`. By default it returns the same value regardless of the step
    (the bit-identity of step 1 passes).
    """

    def __init__(self, calls: list[str], drift: float = 0.0) -> None:
        super().__init__()
        self.calls = calls
        self.stage = "reference"
        self.drift = drift
        self.config = Config()
        self.embeddings = nn.Identity()
        self.head = nn.Identity()

    def forward(self, pixel_values: torch.Tensor):
        if not self.calls or self.calls[-1] != self.stage:
            self.calls.append(self.stage)
        offset = 0.0 if self.stage == "reference" else self.drift
        return SimpleNamespace(pooler_output=pixel_values.flatten(1).sum(dim=1) + offset)


def _stage_verify(monkeypatch, model: _RecordingVision) -> list[str]:
    """Run `verify_patches` once without real weights (returns the record of steps)."""
    calls = model.calls
    monkeypatch.setattr(sg.patch, "patches_applied", lambda: False)
    monkeypatch.setattr(sg, "load_model", lambda _dir: model)
    monkeypatch.setattr(sg, "build_cases", lambda _config: CASES)

    def _apply_shape(_embeddings) -> None:
        calls.append("apply_shape")
        model.stage = "shape-folds"

    def _apply_map_head(_head) -> None:
        calls.append("apply_map_head")
        model.stage = "map-head"

    monkeypatch.setattr(sg.patch, "apply_shape_patches", _apply_shape)
    monkeypatch.setattr(sg.patch, "apply_map_head_patch", _apply_map_head)
    return calls


class TestVerifyOrder:
    """Actually exercise the two paths to a false PASS (a vacuous reference / letting a
    non-bit-identical result through)."""

    def test_a_patched_process_cannot_take_the_reference(self, monkeypatch):
        """MUST: never take the reference in an already patched process (the equivalence check
        would become vacuous with a difference of 0).

        It fails before the heavy `load_model`, so no real weights are needed.
        """
        monkeypatch.setattr(sg.patch, "_SHAPE_APPLIED", True)

        with pytest.raises(SystemExit, match="would become vacuous"):
            sg.verify_patches(Path("/nonexistent"))

    def test_the_reference_is_taken_once_before_any_patch(self, monkeypatch):
        """Catch the regression of re-taking the reference per step (the second step's reference
        would be a patched value)."""
        model = _RecordingVision([])
        calls = _stage_verify(monkeypatch, model)

        entries = sg.verify_patches(Path("/nonexistent"))

        assert calls == [
            "reference",
            "apply_shape",
            "shape-folds",
            "apply_map_head",
            "map-head",
        ]
        assert [entry["stage"] for entry in entries] == ["shape-folds", "map-head"]

    def test_a_shape_fold_that_is_not_bit_exact_stops_the_run(self, monkeypatch):
        """The claim of step 1 is **bit-identity** itself.

        Accepting "the difference is small" would let a dimension mix-up through.
        """
        model = _RecordingVision([], drift=1e-7)
        _stage_verify(monkeypatch, model)

        with pytest.raises(AssertionError, match="the shape folds are not bit-identical"):
            sg.verify_patches(Path("/nonexistent"))


class TestRealCaseShapeGate:
    """It fails when `preprocessor_config.json` and `config.json` are from different
    generations."""

    @staticmethod
    def _write_real_images(root: Path) -> None:
        image_module = pytest.importorskip("PIL.Image")
        array = np.zeros((8, 8, 3), dtype=np.uint8)
        for index, (_name, file_name, _why) in enumerate(sg.REAL_CASES):
            array[:, :, 0] = 10 * index + 7
            image_module.fromarray(array).save(root / file_name)

    def test_a_preprocessor_of_another_generation_fails_loudly(self, monkeypatch, tmp_path):
        """The preprocessing returns 8² while the vision config says 32² — the form in which only
        the golden is baked at another resolution."""
        self._write_real_images(tmp_path)
        monkeypatch.setattr(
            sg,
            "load_image_processor",
            lambda _dir: lambda **_kwargs: {"pixel_values": torch.zeros((1, 3, 8, 8))},
        )

        with pytest.raises(
            AssertionError, match=r"differs from \(1, 3, 32, 32\), which comes from the config"
        ):
            sg.build_real_cases(tmp_path, Config(), tmp_path)


class TestStagedPublication:
    """MUST: put in place only after every check (a failed run is removed with its staging
    directory)."""

    @staticmethod
    def _stage_tiny(monkeypatch, tmp_path: Path) -> Path:
        """Plug in a tiny wrapper and return the location of a fake checkpoint with a download
        record."""
        torch.manual_seed(0)
        wrapper = TinyVisionPooler()
        wrapper.model = SimpleNamespace(config=Config())
        monkeypatch.setattr(sg, "load_wrapper", lambda _dir: wrapper)
        monkeypatch.setattr(sg, "build_cases", lambda _config: CASES)
        model_dir = tmp_path / "siglip2-fixture"
        write_snapshot(model_dir, license="apache-2.0")
        return model_dir

    def test_a_passing_run_leaves_the_series_in_place(self, monkeypatch, tmp_path):
        """The counterpart showing it is not vacuous (when the checks pass, the series is put in
        place) — without it the claim below would be vacuous."""
        model_dir = self._stage_tiny(monkeypatch, tmp_path)
        monkeypatch.setattr(sg, "_sanity", lambda _pooled: {})
        out_dir = tmp_path / "series"

        summary = sg.export_series(model_dir, out_dir)

        assert out_dir.is_dir()
        assert summary["dir"] == str(out_dir)

    def test_the_container_names_the_checkpoint_it_was_baked_from(self, monkeypatch, tmp_path):
        """The container's provenance comes from the actual `--model-dir` (the README's license +
        the downloaded commit SHA)."""
        model_dir = self._stage_tiny(monkeypatch, tmp_path)
        write_snapshot(model_dir, license="cc-by-nc-4.0")
        monkeypatch.setattr(sg, "_sanity", lambda _pooled: {})
        out_dir = tmp_path / "series"

        sg.export_series(model_dir, out_dir)

        assert read_provenance(out_dir / sg.MODEL_FILE) == Provenance(
            license="cc-by-nc-4.0", notice=NOTICE_FILENAME, upstream_revision=FIXTURE_REVISION
        )

    def test_a_failing_sanity_leaves_nothing_behind(self, monkeypatch, tmp_path):
        model_dir = self._stage_tiny(monkeypatch, tmp_path)

        def _reject(_pooled):
            raise AssertionError("the discrimination order is broken")

        monkeypatch.setattr(sg, "_sanity", _reject)
        out_dir = tmp_path / "series"

        with pytest.raises(AssertionError, match="the discrimination order is broken"):
            sg.export_series(model_dir, out_dir)

        assert not out_dir.exists()
