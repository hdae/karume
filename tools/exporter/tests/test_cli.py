"""`karume` サブコマンド CLI（`karume.cli`）のディスパッチ。

引数の解釈は各本体の parser が持つ（CLI は写しを持たない）ので、ここで固定するのは
**どの main へ argv がそのまま渡るか**だけ。台本 `export_anima.py` は実重み依存が重いので、
読み込み関数を差し替えて「呼ばれ方」を見る。
"""

from __future__ import annotations

import inspect
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

    def test_it_picks_the_export_script_by_subcommand_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """台本の選択はサブコマンド名だけ — 素の `export` は Anima のまま動かさない。

        `--verify` × `--target` のような台本側の排他規則は、CLI が argv を 1 語も
        読まないことでだけ抜けなく効く（`--pipeline` を CLI が食う形にしない根拠）。
        """
        seen: list[list[str]] = []
        loaded: list[str] = []

        def load(name: str) -> SimpleNamespace:
            loaded.append(name)
            return SimpleNamespace(main=lambda argv: seen.append(list(argv)))

        monkeypatch.setattr(cli, "load_script", load)
        cli.main(["export-sbv2", "--verify", "front"])
        cli.main(["export-sbv2", "--help"])
        assert loaded == [cli.EXPORT_SBV2_SCRIPT, cli.EXPORT_SBV2_SCRIPT]
        assert seen == [["--verify", "front"], ["--help"]]
        assert cli.EXPORT_SBV2_SCRIPT != cli.EXPORT_SCRIPT

    def test_it_dispatches_embeddinggemma_to_its_own_script(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []
        loaded: list[str] = []

        def load(name: str) -> SimpleNamespace:
            loaded.append(name)
            return SimpleNamespace(main=lambda argv: seen.append(list(argv)))

        monkeypatch.setattr(cli, "load_script", load)
        cli.main(["export-embeddinggemma", "--sym-max", "512"])
        assert loaded == [cli.EXPORT_EMBEDDINGGEMMA_SCRIPT]
        assert seen == [["--sym-max", "512"]]

    @pytest.mark.parametrize(
        ("command", "script"),
        [
            ("export-irodori", "EXPORT_IRODORI_SCRIPT"),
            ("export-dacvae", "EXPORT_DACVAE_SCRIPT"),
            ("export-deberta", "EXPORT_DEBERTA_SCRIPT"),
        ],
    )
    def test_every_export_script_has_a_subcommand_of_its_own(
        self, monkeypatch: pytest.MonkeyPatch, command: str, script: str
    ) -> None:
        """台本は 1 本残らずサブコマンド名で綴れる（cli.py の DECIDED）。

        載っていない台本は `uv run python export_*.py` でしか呼べず、**同じ資産を作る道が
        2 通りある**状態になる（片方だけが規約の更新から取り残される）。
        """
        seen: list[list[str]] = []
        loaded: list[str] = []

        def load(name: str) -> SimpleNamespace:
            loaded.append(name)
            return SimpleNamespace(main=lambda argv: seen.append(list(argv)))

        monkeypatch.setattr(cli, "load_script", load)
        cli.main([command, "--out", "/tmp/series"])
        assert loaded == [getattr(cli, script)]
        assert seen == [["--out", "/tmp/series"]]

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

    @pytest.mark.parametrize(
        "script",
        [
            "EXPORT_SCRIPT",
            "EXPORT_SBV2_SCRIPT",
            "EXPORT_EMBEDDINGGEMMA_SCRIPT",
            "EXPORT_IRODORI_SCRIPT",
            "EXPORT_DACVAE_SCRIPT",
            "EXPORT_DEBERTA_SCRIPT",
        ],
    )
    def test_every_export_script_takes_argv(self, script: str) -> None:
        """台本は 1 本残らず `main(argv)` を受ける（`argv` 無しの main は CLI に載らない）。"""
        module = cli.load_script(getattr(cli, script))
        assert "argv" in inspect.signature(module.main).parameters
