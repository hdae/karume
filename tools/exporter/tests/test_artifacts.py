"""Unit tests of the transactional publication of artifacts (`karume.artifacts`).

The invariant observed is one — **final is either swapped in or left as it was, nothing else**.
That no path really lets a half-built state appear at the canonical path is checked by injecting a
fault at each step (the write fails midway / the verification check rejects / the rename hits an
I/O fault).

A text tree of a few bytes is enough as the subject — this layer knows only paths and rename and
does not touch the format of the contents at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from karume.artifacts import (
    STAGING_SUFFIX,
    SUPERSEDED_SUFFIX,
    ArtifactSwapError,
    staged_publication,
    swap_into_place,
)


def _tree(root: Path, name: str, files: dict[str, str]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for filename, body in files.items():
        (directory / filename).write_text(body, encoding="utf-8")
    return directory


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }


def _siblings(final: Path) -> list[str]:
    """The paths left next to final (staging and superseded paths) — empty on success and on
    failure alike."""
    return sorted(path.name for path in final.parent.iterdir() if path.name != final.name)


class TestStagedPublication:
    """The single path staging path → body (write + verification check) → swap."""

    def test_a_successful_body_lands_the_staged_tree_at_the_final_path(
        self, tmp_path: Path
    ) -> None:
        """The swap is **whole** — a file that existed only in the old artifact is not carried
        over."""
        final = _tree(tmp_path, "final", {"model": "old", "stale": "old"})

        with staged_publication(final) as staging:
            staging.mkdir()
            (staging / "model").write_text("new", encoding="utf-8")

        assert _snapshot(final) == {"model": "new"}
        assert _siblings(final) == []

    def test_a_failure_while_writing_leaves_the_final_artifact_untouched(
        self, tmp_path: Path
    ) -> None:
        """Even when the write fails midway, the canonical path does not change by a single byte."""
        final = _tree(tmp_path, "final", {"model": "old"})
        before = _snapshot(final)

        with (
            pytest.raises(OSError, match="the write failed midway"),
            staged_publication(final) as staging,
        ):
            staging.mkdir()
            (staging / "model").write_text("half", encoding="utf-8")
            raise OSError("the write failed midway")

        assert _snapshot(final) == before
        assert _siblings(final) == []

    def test_a_rejecting_verification_hook_leaves_the_final_artifact_untouched(
        self, tmp_path: Path
    ) -> None:
        """The check is inside the body — even when it rejects after everything is written, the swap
        does not happen at all."""
        final = _tree(tmp_path, "final", {"model": "old"})
        before = _snapshot(final)

        def verify(tree: Path) -> None:
            raise AssertionError(f"verification of {tree.name} failed")

        with (
            pytest.raises(AssertionError, match=r"verification of .* failed"),
            staged_publication(final) as staging,
        ):
            staging.mkdir()
            (staging / "model").write_text("new", encoding="utf-8")
            verify(staging)

        assert _snapshot(final) == before
        assert _siblings(final) == []

    def test_a_working_directory_left_by_an_interrupted_run_is_discarded(
        self, tmp_path: Path
    ) -> None:
        """A path left by an interruption is not reused (no file the writer did not plan gets mixed
        in)."""
        final = _tree(tmp_path, "final", {"model": "old"})
        _tree(tmp_path, "final" + STAGING_SUFFIX, {"interrupted": "半端"})

        with staged_publication(final) as staging:
            staging.mkdir()
            (staging / "model").write_text("new", encoding="utf-8")

        assert _snapshot(final) == {"model": "new"}
        assert _siblings(final) == []

    def test_an_artifact_left_only_in_the_superseded_slot_is_restored(self, tmp_path: Path) -> None:
        """The previous run failed **between** the two renames — the last-known-good in the
        superseded path is moved back, not discarded.

        Even when the body fails, the last-known-good is back at the canonical path (had it been
        discarded, "neither final nor a superseded path" = the next step would stop with the local
        artifact lost entirely).
        """
        superseded = _tree(tmp_path, "final" + SUPERSEDED_SUFFIX, {"model": "old"})
        final = tmp_path / "final"

        with (
            pytest.raises(OSError, match="the write failed midway"),
            staged_publication(final) as staging,
        ):
            staging.mkdir()
            raise OSError("the write failed midway")

        assert _snapshot(final) == {"model": "old"}
        assert not superseded.exists()

    def test_a_single_file_artifact_is_swapped_with_one_atomic_rename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-file artifact does not go through the superseded path — there is no moment at
        all when final is absent."""
        final = tmp_path / "model.safetensors"
        final.write_text("old", encoding="utf-8")
        renames: list[tuple[str, str]] = []
        real = os.replace

        def recording(src: Path, dst: Path) -> None:
            renames.append((Path(src).name, Path(dst).name))
            real(src, dst)

        monkeypatch.setattr("karume.artifacts.os.replace", recording)

        with staged_publication(final) as staged:
            staged.write_text("new", encoding="utf-8")

        assert final.read_text(encoding="utf-8") == "new"
        assert renames == [(final.name + STAGING_SUFFIX, final.name)]
        assert _siblings(final) == []


class TestSwapIntoPlace:
    """The swap for callers that pass an already filled staging path (injects an I/O fault into
    the rename)."""

    def _fail_replace_from(
        self, monkeypatch: pytest.MonkeyPatch, source: Path, message: str
    ) -> None:
        """Make only the `os.replace` that moves `source` fail with an I/O fault (the others go
        through to the real one)."""
        real = os.replace

        def failing(src: Path, dst: Path) -> None:
            if Path(src) == source:
                raise OSError(message)
            real(src, dst)

        monkeypatch.setattr("karume.artifacts.os.replace", failing)

    def test_a_failing_promotion_puts_the_previous_artifact_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even when the promotion fails, final stays **the complete old asset** (neither absent nor
        mixed)."""
        final = _tree(tmp_path, "final", {"model": "old"})
        staging = _tree(tmp_path, "staging", {"model": "new"})
        self._fail_replace_from(monkeypatch, staging, "promotion failed")

        with pytest.raises(ArtifactSwapError, match="swap") as failure:
            swap_into_place(staging, final)

        # The cause (the I/O fault) is kept in the chain — a failed swap is not mistaken for a
        # failed write.
        assert isinstance(failure.value.__cause__, OSError)
        assert _snapshot(final) == {"model": "old"}
        # The staging path belongs to the caller (not removed), but no superseded path is left.
        assert _siblings(final) == ["staging"]

    def test_a_failing_evacuation_leaves_the_previous_artifact_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even when moving aside fails, os.replace is atomic, so final stays intact."""
        final = _tree(tmp_path, "final", {"model": "old"})
        staging = _tree(tmp_path, "staging", {"model": "new"})
        self._fail_replace_from(monkeypatch, final, "moving aside failed")

        with pytest.raises(ArtifactSwapError, match="aside") as failure:
            swap_into_place(staging, final)

        assert isinstance(failure.value.__cause__, OSError)
        assert _snapshot(final) == {"model": "old"}
        assert _siblings(final) == ["staging"]
