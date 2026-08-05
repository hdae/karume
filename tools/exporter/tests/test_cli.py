"""`karume` サブコマンド CLI（`karume.cli`）のディスパッチ。

引数の解釈は各本体の parser が持つ（CLI は写しを持たない）ので、ここで固定するのは
**どの main へ argv がそのまま渡るか**だけ。台本 `export_anima.py` は実重み依存が重いので、
読み込み関数を差し替えて「呼ばれ方」を見る。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from karume import cli, dist, verify


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
        cli.main(["verify", "a/model.safetensors", "b/model.safetensors"])
        assert seen == [["a/model.safetensors", "b/model.safetensors"]]

    def test_it_forwards_the_rest_of_argv_to_the_export_script(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []
        loaded: list[str] = []

        def load(name: str) -> SimpleNamespace:
            loaded.append(name)
            return SimpleNamespace(main=lambda argv: seen.append(list(argv)))

        monkeypatch.setattr(cli, "load_script", load)
        cli.main(["export", "--dtype", "f16", "--target", "transformer"])
        assert loaded == [cli.EXPORT_SCRIPT]
        assert seen == [["--dtype", "f16", "--target", "transformer"]]

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


class TestExportScript:
    def test_it_reaches_the_script_that_lives_outside_the_package(self) -> None:
        """台本はパッケージ外なので、パスの綴りが腐っても import エラーにならない。

        `main(argv)` を持つところまで見る — CLI が渡す argv の受け口が消えたら落とす。
        """
        module = cli.load_script(cli.EXPORT_SCRIPT)
        assert callable(module.main)
