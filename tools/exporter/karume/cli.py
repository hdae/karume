"""`karume` コマンド — export / dist / verify のサブコマンド式ディスパッチ。

    karume export --dtype f16 --dit-graph dyn --lora turbo.safetensors  # 台本 export_anima.py
    karume dist --models ../../models                                   # karume.dist
    karume verify ../../models/anima-turbo/transformer/model.f16.safetensors

MUST: CLI は**引数を解釈しない**。先頭の 1 語でディスパッチし、残りはそのまま対応する
`main(argv)` へ渡す（`--help` も素通しするので、使い方は各本体の parser が出す）。ここに
引数の写しを持つと、台本が持つ排他規則（`--verify` × `--target` のような**沈黙誤値の門**）が
CLI 側の写しでは抜けたまま通る形が生まれ、しかも片方だけ古くなっても誰も落ちない。

MUST: ディスパッチは遅延 import — `karume dist` / `karume verify` を、export 台本
（diffusers / モデル定義の重い import）を 1 つも読まずに起動できる形に保つ。

NOTE: `export_anima.py` は**パッケージ外のリポジトリ直下スクリプト**（台本）なので、import
経路ではなくパス指定で読む。wheel には入らない — 無ければ fail loudly で置き場を示す。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

#: 台本の置き場（karume/cli.py → karume → tools/exporter）。
_SCRIPT_DIR = Path(__file__).resolve().parents[1]

#: `karume export` が包む台本（ADR 0016 の emit ターゲット 4 本を書き出す側）。
EXPORT_SCRIPT = "export_anima"


def load_script(name: str) -> ModuleType:
    """リポジトリ直下の台本をパス指定で読み込む。

    MUST: 実行の**前**に `sys.modules` へ登録する（未登録のままだと台本の `@dataclass` が
    `sys.modules[cls.__module__]` を引けずに AttributeError で落ちる — 実測）。読み込み済みの
    名前はそのまま返す（import 経路で先に入っている実行文脈で二重ロードにしない）。
    """
    if (loaded := sys.modules.get(name)) is not None:
        return loaded
    path = _SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path) if path.is_file() else None
    if spec is None or spec.loader is None:
        raise SystemExit(
            f"export 台本が読めない: {path}"
            "（台本はパッケージ外のスクリプトで wheel に入らない — `karume export` は"
            "リポジトリの作業ツリーでだけ動く）"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_export(argv: Sequence[str]) -> None:
    return load_script(EXPORT_SCRIPT).main(argv)


def run_dist(argv: Sequence[str]) -> None:
    from karume import dist

    return dist.main(argv)


def run_verify(argv: Sequence[str]) -> None:
    from karume import verify

    return verify.main(argv)


#: サブコマンド名 → （ハンドラ, 一覧に出す 1 行）。順序がそのまま `--help` の並び。
COMMANDS: Mapping[str, tuple[Callable[[Sequence[str]], None], str]] = {
    "export": (run_export, "モデルを IR v1 + golden io へ書き出す（台本 export_anima.py）"),
    "dist": (run_dist, "配布ディレクトリを組み立てて karume.json / README.md を書く"),
    "verify": (run_verify, "配布形 safetensors を IR v1 の全規則で検証する"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karume", description="Karume exporter — torch.export 済みモデルを IR v1 へ落とす"
    )
    subcommands = parser.add_subparsers(
        dest="command", required=True, metavar=f"{{{','.join(COMMANDS)}}}"
    )
    for name, (_, summary) in COMMANDS.items():
        # add_help=False: `karume <cmd> --help` を本体の parser へ素通しするため。
        subcommands.add_parser(name, help=summary, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args, rest = build_parser().parse_known_args(argv)
    handler, _ = COMMANDS[args.command]
    handler(rest)


if __name__ == "__main__":
    main()
