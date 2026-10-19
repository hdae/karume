"""Dispatch of the `karume` subcommand CLI (`karume.cli`).

Argument parsing belongs to each command's own parser (the CLI keeps no copy of it), so what this
file pins is only **which main receives argv unchanged**.

NOTE: the `export-*` subcommands and the script loader (formerly `load_script`) are gone together
with their roster (ADR 0065 stages 3+4 complete — the scripts moved to
`tools/export-recipes/<family>/`), so their claims left this file **together with their subject**.
What remains is the three commands dist / migrate / verify.
"""

from __future__ import annotations

import pytest

from karume import cli, dist, migrate, verify


def _spy(monkeypatch: pytest.MonkeyPatch, target: object, name: str) -> list[list[str]]:
    """Replace `target.name` with a stub that only records the argv it is called with."""
    seen: list[list[str]] = []
    monkeypatch.setattr(target, name, lambda argv: seen.append(list(argv)))
    return seen


class TestDispatch:
    def test_it_forwards_the_rest_of_argv_to_dist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy(monkeypatch, dist, "main")
        cli.main(["dist", "--models", "/tmp/models", "--out", "/tmp/out"])
        assert seen == [["--models", "/tmp/models", "--out", "/tmp/out"]]

    def test_it_forwards_the_rest_of_argv_to_verify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy(monkeypatch, verify, "main")
        cli.main(["verify", "a/model.krm", "b/model.krm"])
        assert seen == [["a/model.krm", "b/model.krm"]]

    def test_it_forwards_the_rest_of_argv_to_migrate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _spy(monkeypatch, migrate, "main")
        cli.main(["migrate", "a/model.safetensors", "--out", "/tmp/out", "--license", "mit"])
        assert seen == [["a/model.safetensors", "--out", "/tmp/out", "--license", "mit"]]

    def test_it_passes_help_through_to_the_body_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`karume dist --help` is not handled by the CLI; the command prints its own usage (a
        consequence of keeping no copy)."""
        seen = _spy(monkeypatch, dist, "main")
        cli.main(["dist", "--help"])
        assert seen == [["--help"]]

    def test_it_takes_no_arguments_of_its_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CLI parses no word after the subcommand (it does not consume `--models` and the
        like)."""
        seen = _spy(monkeypatch, dist, "main")
        cli.main(["dist"])
        assert seen == [[]]


class TestUsage:
    def test_it_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit) as raised:
            cli.main([])
        assert raised.value.code == 2

    def test_it_rejects_an_unknown_subcommand(self) -> None:
        with pytest.raises(SystemExit) as raised:
            cli.main(["publish"])
        assert raised.value.code == 2

    def test_the_roster_is_dist_migrate_and_verify(self) -> None:
        """The roster is the three commands dist / migrate / verify.

        If `karume export-siglip2` remained, the wheel's CLI reading a script that is not in the
        wheel would come back as a path that "happens to work only in the working tree".
        `karume repack` (repacking into the shard layout) was retired at step 3a, when the
        distribution became `krm`.
        """
        assert sorted(cli.COMMANDS) == ["dist", "migrate", "verify"]
