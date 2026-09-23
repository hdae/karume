"""`karume` サブコマンド CLI（`karume.cli`）のディスパッチ。

引数の解釈は各本体の parser が持つ（CLI は写しを持たない）ので、ここで固定するのは
**どの main へ argv がそのまま渡るか**だけ。

NOTE: `export-*` サブコマンドと台本ローダ（旧 `load_script`）は名簿ごと消えた
（ADR 0065 段 3+4 完了 — 台本は `tools/export-recipes/<family>/` へ出た）ので、その主張は
このファイルから**被験体ごと**居なくなった。残るのは dist / migrate / verify の 3 コマンド。
"""

from __future__ import annotations

import pytest

from karume import cli, dist, migrate, verify


def _spy(monkeypatch: pytest.MonkeyPatch, target: object, name: str) -> list[list[str]]:
    """`target.name` を「呼ばれた argv を積むだけ」に差し替える。"""
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
        """`karume dist --help` は CLI が握らず本体の使い方を出す（写しを持たない帰結）。"""
        seen = _spy(monkeypatch, dist, "main")
        cli.main(["dist", "--help"])
        assert seen == [["--help"]]

    def test_it_takes_no_arguments_of_its_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """サブコマンドの後ろは 1 語も解釈しない（`--models` 等を CLI 側が食わない）。"""
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
        """名簿は dist / migrate / verify の 3 つ。

        `karume export-siglip2` が残っていると、wheel に無い台本を wheel の CLI が読む形が
        「たまたま作業ツリーでだけ動く」経路として復活する。`karume repack`（shard 仕様への
        詰め替え）は配布形が `krm` になった段 3a で退役した。
        """
        assert sorted(cli.COMMANDS) == ["dist", "migrate", "verify"]
