"""The upgrade of a `karume/5` repository to the current manifest format (`karume upgrade` —
`karume.migrate.upgrade_repository`, ADR 0153).

The repository is built in the test: assembled by `karume.dist` (which writes the current
format) and then given back the `karume/5` spelling, with a cross-repository asset reference added
— exactly the shape of a mirror that `karume/5` tooling assembled. No real asset is read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from ir_fixtures import ir_container

from karume import migrate
from karume.container import numbered_name
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NAME_MAP_FILENAME,
    NAME_MAP_FORMAT,
    Artifact,
    DistError,
    ModelPlan,
    WeightFiles,
    assemble_family,
    manifest_text,
    verify_dist,
)
from karume.migrate import MigrateError, upgrade_main, upgrade_repository

#: A cross-repository asset reference (no local file — only its shape is checked), kept as is.
BORROWED = {
    "repo": "hdae/karume-lender",
    "revision": "b" * 40,
    "path": "shared/tokenizer/tokenizer.json",
    "size": 3,
    "sha256": "c" * 64,
}

#: The `generator` of the repository before the upgrade (kept: the content does not change).
OLD_GENERATOR = "karume/0.13.0"


def _card(manifest: Mapping[str, Any], host_assets: object) -> str:
    """A card that states the format the way every recipe's card does."""
    return (
        "# Synthetic\n\n"
        f"- The distribution manifest is `{MANIFEST_FILENAME}` (`{manifest['format']}`).\n"
        "- Other text that must not move.\n"
    )


def _series(root: Path) -> Path:
    """A valid container series (numbered parts) and its representative path."""
    root.mkdir(parents=True, exist_ok=True)
    payloads = ir_container(mark="w", named="w")
    for index, payload in enumerate(payloads, start=1):
        (root / numbered_name("model.krm", index, len(payloads))).write_bytes(payload)
    return root / "model.krm"


def _repository(tmp_path: Path, *, card: bool = True) -> Path:
    """A `karume/5` repository: one model, its container, the legal text, a card stating the
    format and a cross-repository asset reference."""
    repo = tmp_path / "models" / "karume-old"
    plan = ModelPlan(
        name="A",
        pipeline="anima/1",
        artifacts={"w": Artifact(rel_path="w/model.krm", source=_series(tmp_path / "series"))},
        weights={"w": {"f16": WeightFiles(file="w")}},
        assets={},
        quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
        default_quant="f16",
        pipeline_config={},
    )
    assemble_family(
        [plan],
        repo,
        "A",
        render_card=_card if card else None,
        root_files={"LICENSE.md": "licence\n"},
    )
    manifest = json.loads((repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["format"] = "karume/5"
    # The exporter that converted the weights, not the one that runs the upgrade.
    manifest["generator"] = OLD_GENERATOR
    manifest["models"]["A"]["assets"]["tokenizer"] = dict(BORROWED)
    (repo / MANIFEST_FILENAME).write_text(manifest_text(manifest), encoding="utf-8")
    if card:
        text = (repo / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        (repo / MODEL_CARD_FILENAME).write_text(
            text.replace("(`karume/6`)", "(`karume/5`)"), encoding="utf-8"
        )
    return repo


def _name_map(path: Path, text: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    default = json.dumps({"format": NAME_MAP_FORMAT, "layouts": []}, indent=2) + "\n"
    path.write_text(default if text is None else text, encoding="utf-8")
    return path


def _manifest(repo: Path) -> dict[str, Any]:
    return json.loads((repo / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _snapshot(root: Path) -> dict[str, tuple[bytes, int, int]]:
    """Every file under `root`: its bytes, modification time and inode (a rewrite in place or a
    replacement both show)."""
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestTheUpgrade:
    def test_it_writes_the_current_format_and_keeps_the_content(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        before = _manifest(repo)

        result = upgrade_repository(repo)

        after = _manifest(repo)
        assert result.upgraded
        assert result.previous_format == "karume/5"
        assert not result.name_map
        assert after == {**before, "format": "karume/6"}
        # The key order is the old one (only the value of `format` moved).
        assert list(after) == list(before)
        assert after["models"]["A"]["assets"]["tokenizer"] == BORROWED
        assert after["generator"] == OLD_GENERATOR
        verify_dist(repo)

    def test_a_given_name_map_is_copied_and_referenced(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        source = _name_map(tmp_path / "recipe" / "name-map.json")
        before = _manifest(repo)

        result = upgrade_repository(repo, name_map=source)

        payload = source.read_bytes()
        assert result.name_map
        assert (repo / NAME_MAP_FILENAME).read_bytes() == payload
        assert _manifest(repo) == {
            **before,
            "format": "karume/6",
            "nameMap": {
                "path": NAME_MAP_FILENAME,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }
        assert verify_dist(repo)[NAME_MAP_FILENAME] == len(payload)

    def test_no_weights_file_is_touched(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        before = _snapshot(repo)

        upgrade_repository(repo, name_map=_name_map(tmp_path / "recipe" / "name-map.json"))

        after = _snapshot(repo)
        rewritten = {MANIFEST_FILENAME, MODEL_CARD_FILENAME}
        assert {path: after[path] for path in before if path not in rewritten} == {
            path: entry for path, entry in before.items() if path not in rewritten
        }
        assert set(after) - set(before) == {NAME_MAP_FILENAME}
        assert any(path.endswith(".krm") for path in before)

    def test_the_card_states_the_new_format_and_says_the_rest_as_before(
        self, tmp_path: Path
    ) -> None:
        repo = _repository(tmp_path)
        card = (repo / MODEL_CARD_FILENAME).read_text(encoding="utf-8")

        result = upgrade_repository(repo)

        assert result.card_statements == 1
        assert (repo / MODEL_CARD_FILENAME).read_text(encoding="utf-8") == card.replace(
            "`karume.json` (`karume/5`)", "`karume.json` (`karume/6`)"
        )

    def test_a_card_that_does_not_state_the_format_is_left_as_it_is(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        (repo / MODEL_CARD_FILENAME).write_text("# A card without the format\n", encoding="utf-8")
        card = _snapshot(repo)[MODEL_CARD_FILENAME]

        result = upgrade_repository(repo)

        assert result.card_statements == 0
        assert _snapshot(repo)[MODEL_CARD_FILENAME] == card

    def test_a_repository_without_a_card_gets_none(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path, card=False)

        result = upgrade_repository(repo)

        assert result.card_statements == 0
        assert not (repo / MODEL_CARD_FILENAME).exists()

    def test_it_writes_only_inside_the_directory_and_leaves_no_temporary_file(
        self, tmp_path: Path
    ) -> None:
        repo = _repository(tmp_path)
        source = _name_map(tmp_path / "recipe" / "name-map.json")
        outside = {
            path: entry
            for path, entry in _snapshot(tmp_path).items()
            if not path.startswith("models/karume-old/")
        }

        upgrade_repository(repo, name_map=source)

        assert {
            path: entry
            for path, entry in _snapshot(tmp_path).items()
            if not path.startswith("models/karume-old/")
        } == outside
        assert not [path for path in _snapshot(repo) if path.endswith(".partial")]


class TestASecondRun:
    def test_it_does_nothing_on_a_current_repository(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        source = _name_map(tmp_path / "recipe" / "name-map.json")
        upgrade_repository(repo, name_map=source)
        upgraded = _snapshot(repo)

        again = upgrade_repository(repo, name_map=source)
        without = upgrade_repository(repo)

        assert not again.upgraded
        assert not without.upgraded
        assert again.previous_format == "karume/6"
        assert _snapshot(repo) == upgraded

    def test_another_name_map_for_a_current_repository_fails(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        upgrade_repository(repo, name_map=_name_map(tmp_path / "recipe" / "name-map.json"))
        upgraded = _snapshot(repo)
        other = _name_map(
            tmp_path / "recipe" / "other.json", json.dumps({"format": NAME_MAP_FORMAT}) + "\n"
        )

        with pytest.raises(MigrateError, match="references another name map"):
            upgrade_repository(repo, name_map=other)
        assert _snapshot(repo) == upgraded

    def test_a_name_map_for_a_current_repository_without_one_fails(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        upgrade_repository(repo)
        upgraded = _snapshot(repo)

        with pytest.raises(MigrateError, match="references no name map"):
            upgrade_repository(repo, name_map=_name_map(tmp_path / "recipe" / "name-map.json"))
        assert _snapshot(repo) == upgraded


def _rewrite_manifest(mutate: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def apply(repo: Path) -> None:
        manifest = _manifest(repo)
        mutate(manifest)
        (repo / MANIFEST_FILENAME).write_text(manifest_text(manifest), encoding="utf-8")

    return apply


def _first_part(repo: Path) -> Path:
    return next(path for path in sorted(repo.rglob("*.krm")))


class TestABrokenInputLeavesTheRepositoryAsItWas:
    @pytest.mark.parametrize(
        ("fault", "error", "match"),
        [
            (
                _rewrite_manifest(lambda manifest: manifest.update(format="karume/4")),
                MigrateError,
                "only 'karume/5' can be upgraded",
            ),
            (
                _rewrite_manifest(lambda manifest: manifest.update(nameMap={})),
                MigrateError,
                r"top-level members \['nameMap'\]",
            ),
            (
                lambda repo: (repo / MANIFEST_FILENAME).write_text("{", encoding="utf-8"),
                MigrateError,
                "cannot be read as JSON",
            ),
            (
                _rewrite_manifest(lambda manifest: manifest.update(defaultModel="B")),
                DistError,
                "defaultModel 'B'",
            ),
            (lambda repo: _first_part(repo).unlink(), DistError, "references is missing"),
            (
                lambda repo: _first_part(repo).write_bytes(b"short"),
                DistError,
                "size differs from the manifest",
            ),
            (
                lambda repo: (repo / NAME_MAP_FILENAME).write_text("{}", encoding="utf-8"),
                DistError,
                r"does not declare are mixed in: name-map\.json",
            ),
        ],
    )
    def test_a_broken_repository(
        self,
        tmp_path: Path,
        fault: Callable[[Path], None],
        error: type[Exception],
        match: str,
    ) -> None:
        repo = _repository(tmp_path)
        fault(repo)
        before = _snapshot(repo)

        with pytest.raises(error, match=match):
            upgrade_repository(repo, name_map=_name_map(tmp_path / "recipe" / "name-map.json"))
        assert _snapshot(repo) == before

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("not json", "cannot be read as JSON"),
            ('{"format": "karume-name-map/2"}', "has the format 'karume-name-map/2'"),
        ],
    )
    def test_a_broken_name_map(self, tmp_path: Path, text: str, match: str) -> None:
        repo = _repository(tmp_path)
        before = _snapshot(repo)

        with pytest.raises(DistError, match=match):
            upgrade_repository(repo, name_map=_name_map(tmp_path / "recipe" / "map.json", text))
        assert _snapshot(repo) == before

    def test_a_missing_name_map(self, tmp_path: Path) -> None:
        repo = _repository(tmp_path)
        before = _snapshot(repo)

        with pytest.raises(DistError, match="does not exist"):
            upgrade_repository(repo, name_map=tmp_path / "recipe" / "absent.json")
        assert _snapshot(repo) == before

    def test_a_short_write_found_on_reading_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What is read back from a temporary file must be what was written; when it is not,
        nothing replaces the old files and no temporary file is left."""
        repo = _repository(tmp_path)
        source = _name_map(tmp_path / "recipe" / "name-map.json")
        before = _snapshot(repo)
        read_bytes = Path.read_bytes

        def short(path: Path) -> bytes:
            data = read_bytes(path)
            return data[:-1] if path.name.endswith(".partial") else data

        monkeypatch.setattr(Path, "read_bytes", short)
        with pytest.raises(MigrateError, match="the bytes read back differ"):
            upgrade_repository(repo, name_map=source)
        monkeypatch.undo()
        assert _snapshot(repo) == before

    def test_a_failure_while_replacing_leaves_the_old_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manifest is replaced last: when a replacement fails, the repository still declares
        what it declared, and no temporary file is left."""
        repo = _repository(tmp_path)
        source = _name_map(tmp_path / "recipe" / "name-map.json")
        manifest = (repo / MANIFEST_FILENAME).read_bytes()
        replace = migrate.os.replace
        calls: list[str] = []

        def failing_second(source_path: Path, final: Path) -> None:
            calls.append(Path(final).name)
            if len(calls) == 2:
                raise OSError("injected")
            replace(source_path, final)

        monkeypatch.setattr(migrate.os, "replace", failing_second)
        with pytest.raises(OSError, match="injected"):
            upgrade_repository(repo, name_map=source)
        monkeypatch.undo()

        assert calls == [NAME_MAP_FILENAME, MODEL_CARD_FILENAME]
        assert (repo / MANIFEST_FILENAME).read_bytes() == manifest
        assert not [path for path in _snapshot(repo) if path.endswith(".partial")]


class TestTheCommand:
    def test_it_says_what_it_wrote_then_that_nothing_is_to_do(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _repository(tmp_path)
        source = _name_map(tmp_path / "recipe" / "name-map.json")

        upgrade_main([str(repo), "--name-map", str(source)])
        upgrade_main([str(repo), "--name-map", str(source)])

        first, second = capsys.readouterr().out.splitlines()
        assert first == (
            f"{repo / MANIFEST_FILENAME}: karume/5 → karume/6, name map {NAME_MAP_FILENAME}"
            " referenced; card: 1 format statement(s) rewritten"
        )
        assert second == f"{repo / MANIFEST_FILENAME}: already karume/6 — nothing to do"
