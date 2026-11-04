"""The contract of the intake of a third-party umT5-compatible encoder (ADR 0122 step b —
decisions 5 / 6).

- Intake (`wan.umt5_intake`): the files are placed with their names and contents unchanged, the
  sha256 equals the API's value, and `intake.json` states the original dtype, the provenance of the
  config (the upstream's own / a copy of the original model's) and the license (a marker when
  undeclared). A mismatch in sha256, the key set, the values of the alias pair or the configuration
  fields, FP8, a mix, `.bin` and the old spelling fail loudly. The form with both members of the
  pair and the form with only one are accepted. A license value that does not identify the
  redistribution terms, such as `unknown`, becomes the undeclared marker. When the intake target is
  under `models/`, it fails before the API is called; when the intake target holds anything outside
  the files allowed there (the recorded file, the config, the record, the confirmation record) —
  an index, an upper-case extension, `.pkl`, a half-written `.part` included — it fails before the
  record is written. The record holds the sha256 of the placed config, and rewriting the config
  after the intake makes the readers (the writer, the reference, the assembly) fail before reading.
- Reader: only the files the record names are read, through a descriptor whose sha256 was checked
  (even with an index in place or a path swapped after the check, what is read is the checked
  content).
- Writer (`wan.umt5_export --intake`): the container written from a BF16 intake matches, in the
  sha256 of every part, the container written from safetensors widened to F32 with the same values
  and the same record (differing only in the dtype and in the sha256 / byte count of the files,
  which follow their contents) (widening at read time does not change the rounding starting point).
  A mismatch between the record's dtype and the file's dtype and a file swapped after the intake
  fail loudly. With an undeclared license, `write` without the explicit flag fails reading and
  writing nothing; with the flag, the marker goes into the container's license field.
- Assembly (`dist.py --pipeline umt5 --intake`): a model that came from an intake fails when the
  output is under `models/`, whether or not its license is declared (also through a symlink or a
  relative path with `..`). An undeclared license fails before writing without the explicit flag.
  The root files are derived from the record, without the original model's LICENSE and the
  "No retraining" NOTICE. It does not claim the name `karume-umt5-xxl`. When declared, the record
  and the container's provenance are compared.

The repository names are fictitious (ADR 0122 — no third-party repository name is written into a
tracked file). The weights are a small randomly initialized umT5 (`wan.umt5_probe.TINY_CONFIG`) —
neither real weights nor HF access are needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from container_series import write_component
from safetensors.torch import load_file, save_file
from upstream_fixture import FIXTURE_REVISION, OTHER_REVISION

import dist
from _shared.container_read import read_provenance
from _shared.licenses import apache_license_2_0
from karume.container import Provenance
from karume.dist import (
    LICENSE_FILENAME,
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
)
from wan import umt5_export as ue
from wan import umt5_intake as ui
from wan import umt5_probe as probe
from wan.sources import UMT5_CONFIG, UMT5_SOURCES
from wan.tests import umt5_fixture
from wan.umt5_distribution import (
    PIPELINE,
    UMT5_DEFAULT_MODEL,
    UMT5_REPO_NAME,
    intake_container,
    intake_pipeline,
    intake_repo_name,
)

#: A fictitious upstream (a spelling that matches no real repository).
REPO = "fixture-org/umt5-compatible-fixture"
FILE = "umt5-compatible-fixture_bf16.safetensors"
NAME = "fixture"

#: The original model's row (the basis for checking the configuration and for the copy).
BASE = UMT5_SOURCES[ui.BASE_UPSTREAM].source

EMBED = "encoder.embed_tokens.weight"
SHARED = "shared.weight"

#: The number of rows per block (divides none of vocabulary 384, d_ff 160, d_model 64).
CHUNK_ROWS = 7


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FakeHub:
    """A stand-in for the HF API and downloads (`wan.umt5_intake.Hub`).

    The upstream holds only `FIXTURE_REVISION` of {@link REPO}, and the original model holds only
    its config. `sha256` overrides the LFS value (fault injection of a mismatch), and `card` is
    `cardData` (license, base).
    """

    def __init__(
        self,
        files: Mapping[str, Path],
        base_config: Path,
        *,
        license: str | None = None,
        base_model: tuple[str, ...] | None = None,
        sha256: Mapping[str, str] | None = None,
        api_revision: str = FIXTURE_REVISION,
    ) -> None:
        self.files = dict(files)
        self.base_config = base_config
        self.license = license
        self.base_model = base_model
        self.sha256 = dict(sha256 or {})
        self.api_revision = api_revision
        self.downloads: list[tuple[str, str, str]] = []

    def revision(self, repo: str, revision: str) -> ui.HubRevision:
        assert (repo, revision) == (REPO, FIXTURE_REVISION)
        return ui.HubRevision(
            sha=self.api_revision,
            files={
                name: ui.HubFile(
                    size=path.stat().st_size,
                    sha256=None if name == UMT5_CONFIG else self.sha256.get(name, _sha256(path)),
                )
                for name, path in self.files.items()
            },
            license=self.license,
            base_model=self.base_model,
        )

    def download(self, repo: str, revision: str, filename: str) -> Path:
        self.downloads.append((repo, revision, filename))
        if (repo, revision) == (BASE.repo, BASE.revision):
            assert filename == UMT5_CONFIG
            return self.base_config
        assert (repo, revision) == (REPO, FIXTURE_REVISION)
        return self.files[filename]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`save_pretrained` of a small umT5 (config + a single F32 safetensors — only one of the tied
    weights)."""
    pytest.importorskip("transformers")
    directory = tmp_path_factory.mktemp("umt5-tiny")
    probe.tiny_model().save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def f32_tensors(tiny_dir: Path) -> dict[str, torch.Tensor]:
    return load_file(str(tiny_dir / ue.CHECKPOINT_SINGLE))


@pytest.fixture(scope="module")
def bf16_tensors(f32_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The shape of a fine-tuned version (the same keys, BF16 of the same shapes — f32 values
    rounded with RNE)."""
    return {key: value.to(torch.bfloat16) for key, value in f32_tensors.items()}


def _upstream(root: Path, tensors: Mapping[str, torch.Tensor], name: str = FILE) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(root / name))
    return root / name


def _intake(
    tmp_path: Path,
    tiny_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    name: str = NAME,
    config: Path | None = None,
    **hub: Any,
) -> ui.Umt5Intake:
    files = {FILE: _upstream(tmp_path / "upstream", tensors)}
    if config is not None:
        files[UMT5_CONFIG] = config
    return ui.intake(
        REPO,
        FIXTURE_REVISION,
        FILE,
        name,
        hub=FakeHub(files, tiny_dir / UMT5_CONFIG, **hub),
        out_root=tmp_path / "inputs" / "umt5",
    )


def _config(tmp_path: Path, tiny_dir: Path, **changes: Any) -> Path:
    """The config the upstream holds (only the differences from the original config, through
    `changes`)."""
    document = json.loads((tiny_dir / UMT5_CONFIG).read_text(encoding="utf-8"))
    document.update(changes)
    path = tmp_path / "upstream-config" / UMT5_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


class TestTheIntake:
    def test_it_places_the_file_unchanged_and_records_the_upstream(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        placed = tmp_path / "inputs" / "umt5" / NAME / FILE
        assert intake.checkpoint == placed
        assert placed.read_bytes() == (tmp_path / "upstream" / FILE).read_bytes()
        assert _sha256(placed) == intake.file.sha256
        assert not placed.with_name(f"{FILE}.part").exists()
        record = json.loads((placed.parent / ui.INTAKE_FILE).read_text(encoding="utf-8"))
        assert record["format"] == ui.INTAKE_FORMAT
        assert (record["repo"], record["revision"]) == (REPO, FIXTURE_REVISION)
        assert record["file"] == {
            "name": FILE,
            "size": placed.stat().st_size,
            "sha256": _sha256(placed),
        }
        assert record["dtype"] == "BF16"

    def test_an_undeclared_upstream_is_marked_and_its_configuration_is_assumed(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        """An upstream with neither cardData nor a config (the shape of this fine-tuned version):
        the undeclared marker, a null base, and a configuration copied from the original config."""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        assert intake.license == ui.UNDECLARED_LICENSE == "NOASSERTION"
        assert intake.undeclared
        assert intake.base_model is None
        assert intake.config == ui.ConfigOrigin(
            "base-copy", BASE.repo, BASE.revision, _sha256(tiny_dir / UMT5_CONFIG)
        )
        assert intake.config_assumed
        assert (intake.directory / UMT5_CONFIG).read_bytes() == (
            tiny_dir / UMT5_CONFIG
        ).read_bytes()

    def test_a_declared_upstream_with_its_own_config_is_recorded_as_such(
        self, tmp_path, tiny_dir, f32_tensors
    ):
        config = _config(tmp_path, tiny_dir, architectures=["UMT5EncoderModel"])

        intake = _intake(
            tmp_path,
            tiny_dir,
            f32_tensors,
            config=config,
            license="apache-2.0",
            base_model=(BASE.repo,),
        )

        assert (intake.license, intake.base_model, intake.dtype) == (
            "apache-2.0",
            (BASE.repo,),
            "F32",
        )
        assert intake.config.source == "upstream"
        assert not intake.config_assumed
        assert (intake.directory / UMT5_CONFIG).read_bytes() == config.read_bytes()
        assert intake.config.sha256 == _sha256(config)

    def test_rerunning_the_same_intake_is_safe(self, tmp_path, tiny_dir, bf16_tensors):
        first = _intake(tmp_path, tiny_dir, bf16_tensors)
        second = _intake(tmp_path, tiny_dir, bf16_tensors)

        assert second.file == first.file

    def test_a_sha256_that_differs_from_the_api_fails_and_records_nothing(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        with pytest.raises(ui.Umt5IntakeError, match="differs from the API values"):
            _intake(tmp_path, tiny_dir, bf16_tensors, sha256={FILE: "0" * 64})

        destination = tmp_path / "inputs" / "umt5" / NAME
        assert not (destination / FILE).exists()
        assert not (destination / ui.INTAKE_FILE).exists()

    def test_an_api_commit_other_than_the_pin_fails(self, tmp_path, tiny_dir, bf16_tensors):
        with pytest.raises(ui.Umt5IntakeError, match="the requested revision"):
            _intake(tmp_path, tiny_dir, bf16_tensors, api_revision=OTHER_REVISION)

    @pytest.mark.parametrize(
        ("file", "revision", "match"),
        [
            ("pytorch_model.bin", FIXTURE_REVISION, "is not safetensors"),
            ("sub/" + FILE, FIXTURE_REVISION, "is not a basename"),
            (FILE, "main", "40-digit"),
        ],
    )
    def test_a_pickle_a_nested_file_or_a_branch_fails_before_the_api(
        self, tmp_path, file, revision, match
    ):
        class Unreachable:
            def revision(self, *_: Any) -> Any:
                raise AssertionError("did not fail before calling the API")

            def download(self, *_: Any) -> Any:
                raise AssertionError("did not fail before the download")

        with pytest.raises(ui.Umt5IntakeError, match=match):
            ui.intake(REPO, revision, file, NAME, hub=Unreachable(), out_root=tmp_path)

    def test_a_repository_with_a_trailing_newline_fails_before_the_api(self, tmp_path):
        """Python's `$` also matches right before a trailing newline — a spelling the hub refuses
        must not get through."""

        class Unreachable:
            def revision(self, *_: Any) -> Any:
                raise AssertionError("did not fail before calling the API")

            def download(self, *_: Any) -> Any:
                raise AssertionError("did not fail before the download")

        with pytest.raises(ui.Umt5IntakeError, match="is not of the form '<owner>/<name>'"):
            ui.intake(
                f"{REPO}\n", FIXTURE_REVISION, FILE, NAME, hub=Unreachable(), out_root=tmp_path
            )

    def test_the_name_of_the_base_row_is_refused(self):
        with pytest.raises(ui.Umt5IntakeError, match="the original's row"):
            ui.assert_intake_name(UMT5_DEFAULT_MODEL)

    @pytest.mark.parametrize("license", ["unknown", "UNKNOWN", "", "  ", "NoAssertion", "none"])
    def test_a_license_that_names_no_terms_is_recorded_as_undeclared(
        self, tmp_path, tiny_dir, bf16_tensors, license
    ):
        """HF's `license: unknown` and the like count as no declaration (no container is written
        without the explicit flag)."""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors, license=license)

        assert intake.license == ui.UNDECLARED_LICENSE
        assert intake.undeclared

    def test_a_foreign_weight_file_in_the_destination_fails_and_records_nothing(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        destination = tmp_path / "inputs" / "umt5" / NAME
        destination.mkdir(parents=True)
        (destination / ue.CHECKPOINT_INDEX).write_text(
            json.dumps({"weight_map": {SHARED: "evil.safetensors"}}), encoding="utf-8"
        )

        with pytest.raises(ue.Umt5ExportError, match="files outside the record"):
            _intake(tmp_path, tiny_dir, bf16_tensors)
        assert not (destination / ui.INTAKE_FILE).exists()


class TestTheIntakeDestination:
    """When the intake target (`<name>` under `--out`) is under `models/`, it fails before the API
    is called."""

    class Unreachable:
        def revision(self, *_: Any) -> Any:
            raise AssertionError("did not fail before calling the API")

        def download(self, *_: Any) -> Any:
            raise AssertionError("did not fail before the download")

    @pytest.fixture
    def models(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(ui, "DIST_ROOT", tmp_path / "models")
        (tmp_path / "models").mkdir()
        return tmp_path / "models"

    def _refused(self, out_root: Path, models: Path) -> None:
        with pytest.raises(ui.Umt5IntakeError, match="are not placed in models/"):
            ui.intake(REPO, FIXTURE_REVISION, FILE, NAME, hub=self.Unreachable(), out_root=out_root)
        assert not (models / NAME).exists()

    def test_models_is_refused(self, models):
        self._refused(models, models)

    def test_a_symlink_into_models_is_refused(self, tmp_path, models):
        (tmp_path / "staging").symlink_to(models)

        self._refused(tmp_path / "staging", models)

    def test_a_relative_path_climbing_into_models_is_refused(self, tmp_path, models, monkeypatch):
        (tmp_path / "work").mkdir()
        monkeypatch.chdir(tmp_path / "work")

        self._refused(Path("..", "inputs", "..", "models"), models)

    def test_the_default_destination_is_outside_models(self):
        """The default location (`inputs/umt5/`) passes the check (the check is not one that fails
        everywhere — the pair)."""
        ui.assert_outside_distribution_root(ui.INTAKE_ROOT / NAME, root=ui.DIST_ROOT)


class TestTheCardLicense:
    """`license` / `license_name` of `cardData` → the identifier recorded (`None` becomes the
    undeclared marker)."""

    @pytest.mark.parametrize(
        ("license_id", "license_name", "expected"),
        [
            (None, None, None),
            ("unknown", None, None),
            ("Unknown", None, None),
            ("", None, None),
            ("other", None, None),
            ("other", "", None),
            ("other", "unknown", None),
            ("other", "vendor-license", "vendor-license"),
            ("apache-2.0", None, "apache-2.0"),
        ],
    )
    def test_values_that_name_no_terms_become_undeclared(self, license_id, license_name, expected):
        assert ui._card_license(license_id, license_name, REPO) == expected

    @pytest.mark.parametrize(("license_id", "license_name"), [(["mit"], None), ("other", 1)])
    def test_a_value_that_is_not_a_string_fails(self, license_id, license_name):
        with pytest.raises(ui.Umt5IntakeError, match="is not a string"):
            ui._card_license(license_id, license_name, REPO)


class TestTheCheckpointGate:
    """Step 4 of the intake (header, key set, shapes, alias pair) — any difference and no record is
    written."""

    def _refused(self, tmp_path, tiny_dir, tensors, match: str) -> None:
        with pytest.raises(ue.Umt5ExportError, match=match):
            _intake(tmp_path, tiny_dir, tensors)
        assert not (tmp_path / "inputs" / "umt5" / NAME / ui.INTAKE_FILE).exists()

    def test_a_missing_key_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = dict(bf16_tensors)
        del tensors["encoder.final_layer_norm.weight"]
        self._refused(tmp_path, tiny_dir, tensors, "key set differs")

    def test_a_decoder_key_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = {**bf16_tensors, "decoder.final_layer_norm.weight": torch.ones(64).bfloat16()}
        self._refused(tmp_path, tiny_dir, tensors, "extra 1")

    def test_the_old_spelling_fails(self, tmp_path, tiny_dir, bf16_tensors):
        """Wan's original spelling (`blocks.N.attn.q` and so on) is refused without a conversion
        table."""
        tensors = {
            key.replace("encoder.block.", "blocks.").replace(".layer.0.SelfAttention.", ".attn."): (
                value
            )
            for key, value in bf16_tensors.items()
        }
        self._refused(tmp_path, tiny_dir, tensors, "key set differs")

    def test_a_wrong_shape_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = {**bf16_tensors, "encoder.final_layer_norm.weight": torch.ones(65).bfloat16()}
        self._refused(tmp_path, tiny_dir, tensors, "shapes differ")

    def test_fp8_fails(self, tmp_path, tiny_dir, f32_tensors):
        tensors = {key: value.to(torch.float8_e4m3fn) for key, value in f32_tensors.items()}
        self._refused(tmp_path, tiny_dir, tensors, "dtype F8_E4M3 is not accepted")

    def test_mixed_bf16_and_f32_fails(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = {**bf16_tensors, SHARED: bf16_tensors[SHARED].float()}
        self._refused(tmp_path, tiny_dir, tensors, "mixed dtypes")

    def test_both_names_of_the_tied_pair_are_accepted_when_bit_identical(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        """The original model's form (both members of the pair — the 243-tensor side)."""
        tensors = {**bf16_tensors, EMBED: bf16_tensors[SHARED].clone()}

        assert _intake(tmp_path, tiny_dir, tensors).dtype == "BF16"

    def test_the_tied_pair_that_differs_in_one_element_fails(
        self, tmp_path, tiny_dir, bf16_tensors
    ):
        other = bf16_tensors[SHARED].clone()
        other[3, 5] += 1
        self._refused(
            tmp_path, tiny_dir, {**bf16_tensors, EMBED: other}, "differ within the checkpoint"
        )

    def test_only_the_other_name_of_the_pair_is_accepted(self, tmp_path, tiny_dir, bf16_tensors):
        tensors = dict(bf16_tensors)
        tensors[EMBED] = tensors.pop(SHARED)

        assert _intake(tmp_path, tiny_dir, tensors).dtype == "BF16"


class TestTheConfigGate:
    @pytest.mark.parametrize(
        "change",
        [
            {"relative_attention_max_distance": 64},
            {"relative_attention_num_buckets": 16},
            {"d_ff": 192},
        ],
    )
    def test_an_upstream_config_that_differs_from_the_base_fails(
        self, tmp_path, tiny_dir, bf16_tensors, change
    ):
        """The bucket configuration is not part of the graph description (the output alone could
        change while the container stays byte-identical), so when the upstream has a config it is
        compared with the original model's."""
        config = _config(tmp_path, tiny_dir, **change)

        with pytest.raises(ui.Umt5IntakeError, match="in the configuration fields"):
            _intake(tmp_path, tiny_dir, bf16_tensors, config=config)

    @staticmethod
    def _rewrite_config(intake: ui.Umt5Intake, **changes: Any) -> None:
        """Rewrite the config by hand after the intake (fault injection)."""
        path = intake.directory / UMT5_CONFIG
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(changes)
        path.write_text(json.dumps(document), encoding="utf-8")

    @pytest.mark.parametrize("config_source", ["base-copy", "upstream"])
    def test_a_config_rewritten_after_the_intake_is_refused(
        self, tmp_path, tiny_dir, bf16_tensors, config_source
    ):
        """The record holds the sha256 of the placed bytes and checks it on every read (a rewrite of
        the bucket configuration, which does not affect the graph, is also refused as a
        configuration not checked against the original model)."""
        config = None if config_source == "base-copy" else _config(tmp_path, tiny_dir)
        intake = _intake(tmp_path, tiny_dir, bf16_tensors, config=config)
        assert intake.config.source == config_source
        self._rewrite_config(intake, relative_attention_max_distance=64)

        with pytest.raises(ui.Umt5IntakeError, match="differs from the record's"):
            ui.load_intake(intake.directory)

    def test_a_missing_config_is_refused(self, tmp_path, tiny_dir, bf16_tensors):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        (intake.directory / UMT5_CONFIG).unlink()

        with pytest.raises(ui.Umt5IntakeError, match="is missing"):
            ui.load_intake(intake.directory)

    @pytest.mark.parametrize(
        ("command", "entry"),
        [
            (["write", ui.ALLOW_UNDECLARED_LICENSE_FLAG], "prepare"),
            (["prepare"], "prepare"),
            (["reference"], "reference_encoder"),
        ],
    )
    def test_the_writer_and_the_reference_refuse_a_rewritten_config_before_reading(
        self, tmp_path, tiny_dir, bf16_tensors, monkeypatch, command, entry
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        self._rewrite_config(intake, relative_attention_max_distance=64)
        monkeypatch.setattr(ue, "SERIES_ROOT", tmp_path / "series")

        def unreachable(*_: Any, **__: Any) -> Any:
            raise AssertionError("read with the rewritten config")

        monkeypatch.setattr(ue, entry, unreachable)

        with pytest.raises(ui.Umt5IntakeError, match="differs from the record's"):
            ue.main([command[0], "--intake", str(intake.directory), *command[1:]])
        assert not (tmp_path / "series").exists()


class TestTheRecord:
    def _document(self, tmp_path, tiny_dir, bf16_tensors) -> tuple[Path, dict[str, Any]]:
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        path = intake.directory / ui.INTAKE_FILE
        return path, json.loads(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"dtype": "F16"}, "dtype"),
            ({"format": "karume-umt5-intake/0"}, "format"),
            ({"format": "karume-umt5-intake/1"}, "redo"),
            (
                {
                    "config": {
                        "source": "base-copy",
                        "base": {"repo": BASE.repo, "revision": BASE.revision},
                    }
                },
                "config is not of the form",
            ),
            ({"extra": 1}, "the fields differ"),
            ({"revision": "main"}, "40-digit"),
            ({"license": ""}, "license is empty"),
            ({"license": "unknown"}, "does not identify the redistribution terms"),
            ({"license": "noassertion"}, "does not identify the redistribution terms"),
            ({"license": "Other"}, "does not identify the redistribution terms"),
        ],
    )
    def test_a_broken_record_is_refused(self, tmp_path, tiny_dir, bf16_tensors, change, match):
        path, document = self._document(tmp_path, tiny_dir, bf16_tensors)
        path.write_text(json.dumps({**document, **change}), encoding="utf-8")

        with pytest.raises(ui.Umt5IntakeError, match=match):
            ui.load_intake(path.parent)

    def test_the_base_row_key_is_the_distribution_model_name(self):
        assert ui.BASE_UPSTREAM == UMT5_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Writer (--intake)
# ---------------------------------------------------------------------------


def _widened_copy(intake: ui.Umt5Intake, root: Path, tensors: Mapping[str, torch.Tensor]) -> Path:
    """A copy of the same intake with the file replaced by F32 `tensors` and the record's dtype set
    to F32 (the file's sha256 / byte count change with its contents — the other fields are the
    same)."""
    shutil.copytree(intake.directory, root)
    save_file(dict(tensors), str(root / FILE))
    document = json.loads((root / ui.INTAKE_FILE).read_text(encoding="utf-8"))
    document["dtype"] = "F32"
    document["file"] = {
        "name": FILE,
        "size": (root / FILE).stat().st_size,
        "sha256": _sha256(root / FILE),
    }
    (root / ui.INTAKE_FILE).write_text(json.dumps(document), encoding="utf-8")
    return root


def _container(intake: ui.Umt5Intake, out: Path) -> list[str]:
    export = ue.prepare(
        intake.directory, checkpoint=ue.intake_checkpoint(intake), chunk_rows=CHUNK_ROWS
    )
    ue.write_container(export, out / ue.MODEL_FILE, ue.intake_provenance(intake))
    return ue.part_digests(out / ue.MODEL_FILE)


class TestTheWriter:
    def test_the_bf16_intake_writes_the_container_of_its_values_widened_to_f32(
        self, tmp_path, tiny_dir, bf16_tensors, f32_tensors
    ):
        """Widening at read time does not change the rounding starting point (the acceptance of
        ADR 0122 step b). The control: writing from F32 before rounding gives a different container
        (even for the same layer, i8 starting from bf16 has different bytes)."""
        bf16 = _intake(tmp_path, tiny_dir, bf16_tensors)
        widened = ui.load_intake(
            _widened_copy(
                bf16,
                tmp_path / "widened" / NAME,
                {key: value.float() for key, value in bf16_tensors.items()},
            )
        )
        unrounded = ui.load_intake(_widened_copy(bf16, tmp_path / "unrounded" / NAME, f32_tensors))

        digests = _container(bf16, tmp_path / "out-bf16")

        assert widened.dtype == "F32"
        assert _container(widened, tmp_path / "out-widened") == digests
        assert _container(unrounded, tmp_path / "out-unrounded") != digests

    def test_a_record_that_names_another_dtype_fails(self, tmp_path, tiny_dir, bf16_tensors):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        path = intake.directory / ui.INTAKE_FILE
        path.write_text(
            json.dumps({**json.loads(path.read_text(encoding="utf-8")), "dtype": "F32"}),
            encoding="utf-8",
        )

        with pytest.raises(ue.Umt5ExportError, match="only F32 is accepted"):
            ue.prepare(
                intake.directory,
                checkpoint=ue.intake_checkpoint(ui.load_intake(intake.directory)),
            )

    def test_an_f32_file_under_a_bf16_record_fails(self, tmp_path, tiny_dir, bf16_tensors):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        widened = _widened_copy(
            intake, tmp_path / "w" / NAME, {k: v.float() for k, v in bf16_tensors.items()}
        )
        path = widened / ui.INTAKE_FILE
        path.write_text(
            json.dumps({**json.loads(path.read_text(encoding="utf-8")), "dtype": "BF16"}),
            encoding="utf-8",
        )

        with pytest.raises(
            ue.Umt5ExportError, match=r"only the BF16 intake\.json declares is accepted"
        ):
            ue.prepare(widened, checkpoint=ue.intake_checkpoint(ui.load_intake(widened)))

    def test_a_file_replaced_after_the_intake_fails(self, tmp_path, tiny_dir, bf16_tensors):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        other = dict(bf16_tensors)
        other[SHARED] = other[SHARED] + 1
        save_file(other, str(intake.checkpoint))

        with pytest.raises(ue.Umt5ExportError, match="sha256"):
            ue.intake_checkpoint(intake)

    def test_the_reader_ignores_an_index_and_reads_only_the_recorded_file(
        self, tmp_path, tiny_dir, bf16_tensors, monkeypatch
    ):
        """Fault injection: put another file (values +1) and an index pointing at it in the intake
        target, and even with the outside-the-record weights check removed, the reader reads only
        the checked file (it does not take the index branch)."""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        evil = {key: value + 1 for key, value in bf16_tensors.items()}
        save_file(evil, str(intake.directory / "evil.safetensors"))
        (intake.directory / ue.CHECKPOINT_INDEX).write_text(
            json.dumps({"weight_map": dict.fromkeys(evil, "evil.safetensors")}), encoding="utf-8"
        )
        monkeypatch.setattr(ue, "assert_only_intake_files", lambda *_: None)

        checkpoint = ue.intake_checkpoint(intake)

        assert torch.equal(checkpoint.read(SHARED), bf16_tensors[SHARED].float())

    @pytest.mark.parametrize(
        "foreign",
        [
            ue.CHECKPOINT_INDEX,
            "evil.safetensors",
            "sub/x.bin",
            # Closed by an allow list (forms that a list of extensions misses).
            "EVIL.SAFETENSORS",
            "weights.pkl",
            "weights.npz",
            f"{FILE}.part",
            "Config.json",
            "notes.txt",
        ],
    )
    def test_a_foreign_weight_file_next_to_the_record_fails(
        self, tmp_path, tiny_dir, bf16_tensors, foreign
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        (intake.directory / foreign).parent.mkdir(parents=True, exist_ok=True)
        (intake.directory / foreign).write_bytes(b"{}")

        with pytest.raises(ue.Umt5ExportError, match="files outside the record"):
            ue.intake_checkpoint(intake)

    def test_a_symlink_under_an_allowed_name_fails(self, tmp_path, tiny_dir, bf16_tensors):
        """A symlink is outside even with an allowed name: under the confirmation record's name it
        could point at weights outside the record."""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"{}")
        (intake.directory / ui.LICENSE_REVIEW_FILE).symlink_to(outside)

        with pytest.raises(ue.Umt5ExportError, match="files outside the record"):
            ue.intake_checkpoint(intake)

    def test_the_review_note_is_allowed_next_to_the_record(self, tmp_path, tiny_dir, bf16_tensors):
        """The pair: with only the allowed files (the recorded file, the config, the record, the
        confirmation record) it reads."""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        (intake.directory / ui.LICENSE_REVIEW_FILE).write_text("reviewed\n", encoding="utf-8")

        assert ue.intake_checkpoint(intake).names()

    def test_a_file_swapped_after_the_check_is_not_read(self, tmp_path, tiny_dir, bf16_tensors):
        """The check and the read see the same content: even when the path is pointed at another
        file after the check (rename), what is read is the checked content the held descriptor
        points at."""
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)
        checkpoint = ue.intake_checkpoint(intake)
        swapped = tmp_path / "swapped.safetensors"
        save_file({key: value + 1 for key, value in bf16_tensors.items()}, str(swapped))
        os.replace(swapped, intake.checkpoint)

        assert torch.equal(checkpoint.read(SHARED), bf16_tensors[SHARED].float())
        assert torch.equal(checkpoint.read_rows(SHARED, 2, 5), bf16_tensors[SHARED][2:5].float())

    def test_the_upstream_table_path_still_takes_f32_only(self, tmp_path, bf16_tensors):
        save_file(dict(bf16_tensors), str(tmp_path / ue.CHECKPOINT_SINGLE))

        with pytest.raises(ue.Umt5ExportError, match="only F32 is accepted"):
            ue.Checkpoint(tmp_path).read(SHARED)


class TestTheWriteCommand:
    @pytest.fixture
    def series_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        root = tmp_path / "series"
        monkeypatch.setattr(ue, "SERIES_ROOT", root)
        return root

    def test_an_undeclared_intake_fails_before_reading_or_writing(
        self, tmp_path, tiny_dir, bf16_tensors, series_root, monkeypatch
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        def unreachable(*_: Any, **__: Any) -> Any:
            raise AssertionError("read the weights without the explicit flag")

        monkeypatch.setattr(ue, "prepare", unreachable)

        with pytest.raises(ue.Umt5ExportError, match=ui.ALLOW_UNDECLARED_LICENSE_FLAG):
            ue.main(["write", "--intake", str(intake.directory)])
        assert not series_root.exists()

    def test_an_unknown_license_needs_the_declaration_too(
        self, tmp_path, tiny_dir, bf16_tensors, series_root
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors, license="unknown")

        with pytest.raises(ue.Umt5ExportError, match=ui.ALLOW_UNDECLARED_LICENSE_FLAG):
            ue.main(["write", "--intake", str(intake.directory)])
        assert not series_root.exists()

    def test_the_declaration_writes_the_mark_into_the_container(
        self, tmp_path, tiny_dir, bf16_tensors, series_root, capsys
    ):
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        ue.main(["write", "--intake", str(intake.directory), ui.ALLOW_UNDECLARED_LICENSE_FLAG])

        container = series_root / f"umt5-xxl-{NAME}-i8-dyn" / ue.COMPONENT_DIR / ue.MODEL_FILE
        assert read_provenance(container) == Provenance(
            license=ui.UNDECLARED_LICENSE,
            notice=NOTICE_FILENAME,
            upstream_revision=FIXTURE_REVISION,
        )

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["write", ui.ALLOW_UNDECLARED_LICENSE_FLAG], "applies only to write with --intake"),
            (["write", "--intake", "seat", "--upstream", "xxl"], "are the same axis"),
            (["check-mask", "--intake", "seat"], "--intake does not apply to check-mask"),
            (
                ["reference", "--intake", "seat", ui.ALLOW_UNDECLARED_LICENSE_FLAG],
                "--allow-undeclared-license does not apply to reference",
            ),
        ],
    )
    def test_misplaced_arguments_are_refused(self, argv, message, capsys):
        with pytest.raises(SystemExit) as exited:
            ue.main(argv)

        assert exited.value.code == 2
        assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Assembly (dist.py --intake)
# ---------------------------------------------------------------------------


def _record(
    directory: Path,
    *,
    license: str = ui.UNDECLARED_LICENSE,
    config_source: str = "base-copy",
    base_model: list[str] | None = None,
) -> Path:
    """Put only the record the assembly reads and the config the record pins (the assembly does not
    read the weight files)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / UMT5_CONFIG).write_text("{}\n", encoding="utf-8")
    (directory / ui.INTAKE_FILE).write_text(
        json.dumps(
            {
                "format": ui.INTAKE_FORMAT,
                "name": directory.name,
                "repo": REPO,
                "revision": FIXTURE_REVISION,
                "file": {"name": FILE, "size": 1234, "sha256": "ab" * 32},
                "dtype": "BF16",
                "license": license,
                "base_model": base_model,
                "config": {
                    "source": config_source,
                    "base": {"repo": BASE.repo, "revision": BASE.revision},
                    "sha256": _sha256(directory / UMT5_CONFIG),
                },
                "fetched_at": "2026-10-04T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return directory


class TestTheMirror:
    @pytest.fixture
    def roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the default locations of `models/` and the local experimental mirror at a
        temporary directory."""
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "LOCAL_DIST_ROOT", tmp_path / "local-dist")
        (tmp_path / "models").mkdir()
        return tmp_path

    def _series(
        self, root: Path, *, license: str = ui.UNDECLARED_LICENSE, revision: str = FIXTURE_REVISION
    ) -> None:
        provenance = Provenance(license=license, notice=NOTICE_FILENAME, upstream_revision=revision)
        write_component(
            intake_container(root / "series", NAME),
            umt5_fixture.umt5_container(provenance=provenance),
        )

    def _argv(self, root: Path, *extra: str) -> list[str]:
        return [
            "--pipeline",
            "umt5",
            "--intake",
            str(root / "inputs" / "umt5" / NAME),
            "--series",
            str(root / "series"),
            *extra,
        ]

    def test_an_undeclared_intake_needs_the_declaration_before_anything_is_written(self, roots):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)

        with pytest.raises(DistError, match=ui.ALLOW_UNDECLARED_LICENSE_FLAG):
            dist.main(self._argv(roots))
        assert not (roots / "local-dist").exists()

    def test_the_undeclared_mirror_carries_no_license_and_says_so(self, roots, capsys):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)

        dist.main(self._argv(roots, ui.ALLOW_UNDECLARED_LICENSE_FLAG))

        out_dir = roots / "local-dist" / NAME
        assert not (out_dir / LICENSE_FILENAME).exists()
        notice = (out_dir / NOTICE_FILENAME).read_text(encoding="utf-8")
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "declares no license (recorded as `NOASSERTION`)" in notice
        assert f"`{REPO}` at commit `{FIXTURE_REVISION}`" in notice
        assert "bfloat16" in notice and "widened to float32 without loss" in notice
        for claim in ("No retraining", "Wan", "Google", "google/"):
            assert claim not in notice
        assert "license: NOASSERTION" in card
        assert "base_model" not in card.split("---")[1]
        assert "> **Local experimental conversion — do not redistribute.**" in card
        assert "**Configuration**: assumed" in card
        assert "The upstream file stores bfloat16 values." in card
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == [NAME]

    def test_a_declared_mirror_ships_the_verbatim_license(self, roots, capsys):
        _record(
            roots / "inputs" / "umt5" / NAME,
            license="apache-2.0",
            config_source="upstream",
            base_model=["fixture-org/base"],
        )
        self._series(roots, license="apache-2.0")

        dist.main(self._argv(roots))

        out_dir = roots / "local-dist" / NAME
        assert (out_dir / LICENSE_FILENAME).read_text(encoding="utf-8") == apache_license_2_0()
        notice = (out_dir / NOTICE_FILENAME).read_text(encoding="utf-8")
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "declares the license `apache-2.0`" in notice
        assert "It declares the base model `fixture-org/base`." in notice
        for claim in ("No retraining", "Wan", "Google", "google/"):
            assert claim not in notice
        assert "license: apache-2.0" in card and "base_model: fixture-org/base" in card
        assert "> **Local experimental conversion — not published.**" in card
        assert "**Configuration**: the upstream `config.json`" in card

    @pytest.mark.parametrize("license", [ui.UNDECLARED_LICENSE, "apache-2.0"])
    def test_models_is_refused_whatever_the_license(self, roots, license):
        _record(roots / "inputs" / "umt5" / NAME, license=license)
        self._series(roots, license=license)

        with pytest.raises(DistError, match="are not placed in models/"):
            dist.main(
                self._argv(
                    roots, "--out", str(roots / "models" / NAME), ui.ALLOW_UNDECLARED_LICENSE_FLAG
                )
            )
        assert not (roots / "models" / NAME).exists()

    def test_a_symlink_into_models_is_refused(self, roots):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)
        (roots / "staging").symlink_to(roots / "models")

        with pytest.raises(DistError, match="are not placed in models/"):
            dist.main(
                self._argv(
                    roots, "--out", str(roots / "staging" / NAME), ui.ALLOW_UNDECLARED_LICENSE_FLAG
                )
            )
        assert not (roots / "models" / NAME).exists()

    def test_a_relative_path_climbing_into_models_is_refused(self, roots, monkeypatch):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)
        (roots / "work").mkdir()
        monkeypatch.chdir(roots / "work")
        out = os.path.join("..", "local-dist", "..", "models", NAME)

        with pytest.raises(DistError, match="are not placed in models/"):
            dist.main(self._argv(roots, "--out", out, ui.ALLOW_UNDECLARED_LICENSE_FLAG))
        assert not (roots / "models" / NAME).exists()

    @pytest.mark.parametrize(
        ("license", "revision", "match"),
        [
            ("cc-by-nc-4.0", FIXTURE_REVISION, r"provenance\.license"),
            ("apache-2.0", OTHER_REVISION, "different revision"),
        ],
    )
    def test_the_container_provenance_is_checked_against_the_record(
        self, roots, license, revision, match
    ):
        _record(roots / "inputs" / "umt5" / NAME, license="apache-2.0")
        self._series(roots, license=license, revision=revision)

        with pytest.raises(DistError, match=match):
            dist.main(self._argv(roots))
        assert not (roots / "local-dist").exists()

    def test_a_marked_container_does_not_pass_as_a_declared_record(self, roots):
        """Even with the record rewritten to declared, a container baked with the marker fails the
        comparison (closed by the container's contents)."""
        _record(roots / "inputs" / "umt5" / NAME, license="apache-2.0")
        self._series(roots, license=ui.UNDECLARED_LICENSE)

        with pytest.raises(DistError, match="NOASSERTION"):
            dist.main(self._argv(roots))

    def test_a_declared_license_without_a_text_is_refused_before_writing(self, roots):
        _record(roots / "inputs" / "umt5" / NAME, license="mit")
        self._series(roots, license="mit")

        with pytest.raises(DistError, match="there is no text for the license"):
            dist.main(self._argv(roots))
        assert not (roots / "local-dist").exists()

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            (["--model", UMT5_DEFAULT_MODEL], "differs from the intake name"),
            (["--pipeline", "wan"], "--intake applies only to --pipeline umt5"),
        ],
    )
    def test_mixing_with_other_models_or_pipelines_is_refused(self, roots, extra, match):
        _record(roots / "inputs" / "umt5" / NAME)
        self._series(roots)

        with pytest.raises(DistError, match=match):
            dist.main([*self._argv(roots, ui.ALLOW_UNDECLARED_LICENSE_FLAG), *extra])

    def test_the_declaration_alone_is_refused(self, roots):
        with pytest.raises(DistError, match="applies only together with --intake"):
            dist.main(["--pipeline", "umt5", ui.ALLOW_UNDECLARED_LICENSE_FLAG])

    def test_it_never_names_the_base_repository(self, tmp_path):
        intake = ui.load_intake(_record(tmp_path / NAME))
        pipeline = intake_pipeline(intake, allow_undeclared_license=True)

        assert pipeline.repo_name(NAME) == intake_repo_name(NAME) != UMT5_REPO_NAME
        assert not intake_repo_name(NAME).startswith("karume-")
        assert pipeline.root_files != PIPELINE.root_files

    def test_the_base_pipeline_is_unchanged(self):
        assert dist.PIPELINES["umt5"] is PIPELINE
        assert set(PIPELINE.root_files) == {LICENSE_FILENAME, NOTICE_FILENAME}
        assert "No retraining and no fine-tuning" in PIPELINE.root_files[NOTICE_FILENAME]


class TestTheWholeChain:
    def test_intake_write_and_mirror(self, tmp_path, tiny_dir, bf16_tensors, monkeypatch, capsys):
        """Run intake → writer (explicit flag) → local experimental mirror (explicit flag) through
        the real writers."""
        monkeypatch.setattr(ue, "SERIES_ROOT", tmp_path / "series")
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(dist, "LOCAL_DIST_ROOT", tmp_path / "local-dist")
        intake = _intake(tmp_path, tiny_dir, bf16_tensors)

        ue.main(["write", "--intake", str(intake.directory), ui.ALLOW_UNDECLARED_LICENSE_FLAG])
        dist.main(
            [
                "--pipeline",
                "umt5",
                "--intake",
                str(intake.directory),
                "--series",
                str(tmp_path / "series"),
                ui.ALLOW_UNDECLARED_LICENSE_FLAG,
            ]
        )

        out_dir = tmp_path / "local-dist" / NAME
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["models"][NAME]["pipeline"] == "umt5-encoder/1"
        assert sorted(path.name for path in out_dir.iterdir() if path.is_file()) == sorted(
            [MODEL_CARD_FILENAME, NOTICE_FILENAME, MANIFEST_FILENAME]
        )
        # The intake records only the repository, so its account is the publisher named.
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").splitlines()
        assert (
            f"This is an unofficial conversion of {REPO} for Karume."
            " It is not affiliated with or endorsed by fixture-org."
        ) in card
