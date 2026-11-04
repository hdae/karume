"""The contract of the `--check` of `wan/export_dit.py` (the entry point that rewrites the component
directories with the writers into a temporary directory inside the series and matches every file
against the existing series by sha256 — ADR 0121 decision 10).

What it guards is the very entry point that confirms "when the writers shared with Wan2.1 change,
the 1.3B containers and goldens have not moved by a single byte":

- if the writers did not change, every file (all container parts, `io.*`, `reference.*`) matches,
  and the series does not change by a single byte (no temporary directory is left either)
- if the writers' output changes, it stops naming the differing files (fault injection: change the
  seed of one case → only that case's 2 files are named)
- `--no-full` leaves only the files of the full-size cases out of the matching (a left-out file
  missing on the existing side is named)
- with nothing to match against, it stops before reading the upstream

Runs only on a synthetic model (`test_dit_patch.TINY_DIT` — a small randomly initialised
`WanTransformer3DModel`). The check of the real 1.3B is `python -m wan.export_dit --dtype i8
--no-full --check` and the like (runbook).
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from wan import export_dit
from wan.tests.test_dit_patch import TINY_DIT, TINY_LATENT

#: The synthetic patch (patch_size of TINY_DIT).
PATCH = (1, 2, 2)

#: The cases of the synthetic series (the 2 roles the i8 series takes + 1 full-size role — the
#: full-size grid is selected by replacing {@link export_dit.I8_FULL_LATENTS} with the synthetic
#: grid).
CASES = (
    export_dit.CaseSpec("band", TINY_LATENT, 999, 5, 1),
    export_dit.CaseSpec("accept", TINY_LATENT, 500, 3, 2),
    export_dit.CaseSpec("full-band", TINY_LATENT, 250, 4, 3, blocks=False),
)


def _tiny(_model_name: str = "") -> nn.Module:
    """A synthetic DiT with the same weights every time (stands in for
    `export_dit.load_transformer` — the rewrite round gets the same values too)."""
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    torch.manual_seed(20261004)
    return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _point_at_tiny(patch: pytest.MonkeyPatch, root: Path) -> None:
    patch.setattr(export_dit, "CASES", CASES)
    patch.setattr(export_dit, "I8_FULL_LATENTS", (TINY_LATENT,))
    patch.setattr(export_dit, "SERIES", root / "f16")
    patch.setattr(export_dit, "I8_SERIES", root / "i8")
    patch.setattr(export_dit, "PROBE_SERIES", root / "probe")
    patch.setattr(export_dit, "load_transformer", _tiny)
    patch.setattr(export_dit, "TEXT_DIM", TINY_DIT["text_dim"])
    patch.setattr(export_dit, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))


def _digests(directory: Path) -> dict[str, str]:
    return export_dit.file_digests(directory)


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> Iterator[Path]:
    """The root of a series with the synthetic i8 series written (all cases, the full-size role
    included)."""
    root = tmp_path_factory.mktemp("export-dit-check")
    with pytest.MonkeyPatch.context() as patch:
        _point_at_tiny(patch, root)
        assert export_dit.main(["--dtype", "i8"]) == 0
    yield root / "i8"


@pytest.fixture
def series(written: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A per-test copy (tests that break or delete things do not touch the series of other
    tests)."""
    copy = tmp_path / "i8"
    shutil.copytree(written, copy)
    _point_at_tiny(monkeypatch, tmp_path)
    return copy


def _check(*flags: str) -> dict[str, Any]:
    """Run `--check` of the i8 series and return the summary (the arguments pass the checks of
    `main`)."""
    return export_dit.emit(_namespace("--dtype", "i8", "--check", *flags))


def _namespace(*argv: str) -> Any:
    """The arguments of `emit` after the same argument checks as `main` (to receive the summary as
    the return value)."""
    import argparse

    captured: dict[str, argparse.Namespace] = {}

    def capture(args: argparse.Namespace) -> dict[str, Any]:
        captured["args"] = args
        return {}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export_dit, "emit", capture)
        assert export_dit.main(list(argv)) == 0
    return captured["args"]


class TestAnUnchangedWriter:
    def test_every_file_matches_and_the_series_is_untouched(self, series: Path) -> None:
        before = _digests(series / export_dit.TARGET)

        summary = _check()

        names = [spec.name(PATCH) for spec in CASES]
        golden = {file for name in names for file in export_dit.case_file_names(name)}
        assert summary["check"]["skipped_existing"] == []
        assert golden <= set(summary["check"]["identical_files"])
        assert any(name.startswith("model-") for name in summary["check"]["identical_files"])
        assert set(summary["check"]["identical_files"]) == set(before)
        assert _digests(series / export_dit.TARGET) == before
        assert sorted(path.name for path in series.iterdir()) == [export_dit.TARGET]

    def test_no_full_skips_exactly_the_full_size_case_files(self, series: Path) -> None:
        summary = _check("--no-full")

        full = set(export_dit.case_file_names(CASES[2].name(PATCH)))
        assert set(summary["check"]["skipped_existing"]) == full
        assert not full & set(summary["check"]["identical_files"])


class TestAChangedWriter:
    def test_a_changed_case_names_exactly_its_two_files(
        self, series: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fault injection: change the seed of one acceptance case (only that case's writer output
        moves)."""
        changed = dataclasses.replace(CASES[1], seed=CASES[1].seed + 1)
        monkeypatch.setattr(export_dit, "CASES", (CASES[0], changed, CASES[2]))
        before = _digests(series / export_dit.TARGET)

        with pytest.raises(export_dit.SeriesMismatchError) as caught:
            _check()

        message = str(caught.value)
        io_name, reference_name = export_dit.case_file_names(CASES[1].name(PATCH))
        assert io_name in message
        assert reference_name in message
        assert export_dit.case_file_names(CASES[0].name(PATCH))[0] not in message
        assert _digests(series / export_dit.TARGET) == before
        assert sorted(path.name for path in series.iterdir()) == [export_dit.TARGET]

    def test_a_skipped_file_missing_from_the_series_is_named(self, series: Path) -> None:
        _, reference_name = export_dit.case_file_names(CASES[2].name(PATCH))
        (series / export_dit.TARGET / reference_name).unlink()

        with pytest.raises(export_dit.SeriesMismatchError, match=reference_name):
            _check("--no-full")


class TestNoExistingSeries:
    def test_it_stops_before_reading_the_upstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _point_at_tiny(monkeypatch, tmp_path)

        def never(_model_name: str) -> nn.Module:
            raise AssertionError("read the upstream although there was nothing to match against")

        monkeypatch.setattr(export_dit, "load_transformer", never)

        with pytest.raises(export_dit.SeriesMismatchError, match="to compare against"):
            _check()


class TestSeriesMismatches:
    def _directory(self, root: Path, files: dict[str, bytes]) -> Path:
        root.mkdir()
        for name, payload in files.items():
            (root / name).write_bytes(payload)
        return root

    def test_each_kind_of_difference_is_named(self, tmp_path: Path) -> None:
        written = self._directory(tmp_path / "w", {"a": b"1", "b": b"2", "extra": b"x"})
        existing = self._directory(tmp_path / "e", {"a": b"1", "b": b"3", "gone": b"y"})

        assert export_dit.series_mismatches(written, existing) == {
            "b": "sha256 differs",
            "extra": "missing from the existing directory",
            "gone": "missing from the written directory",
        }

    def test_skipped_files_must_exist_only_in_the_series(self, tmp_path: Path) -> None:
        written = self._directory(tmp_path / "w", {"a": b"1", "both": b"z"})
        existing = self._directory(tmp_path / "e", {"a": b"1", "skip": b"s", "both": b"z"})

        assert export_dit.series_mismatches(
            written, existing, absent={"skip", "both", "nowhere"}
        ) == {
            "both": "wrote the file of an excluded case",
            "nowhere": "the file of an excluded case is missing from the existing directory",
        }

    def test_a_directory_inside_the_component_fails_loudly(self, tmp_path: Path) -> None:
        written = self._directory(tmp_path / "w", {"a": b"1"})
        (written / "nested").mkdir()

        with pytest.raises(export_dit.SeriesMismatchError, match="not a file"):
            export_dit.file_digests(written)


class TestTheFlags:
    def test_check_and_verify_do_not_mix(self) -> None:
        with pytest.raises(SystemExit):
            export_dit.main(["--verify", "--check"])
