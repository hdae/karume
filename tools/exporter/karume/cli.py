"""`karume` コマンド — dist / verify のサブコマンド式ディスパッチ。

    karume dist --pipeline siglip2   # 受理集合が空の core 単体では落ちる（下の NOTE）
    karume verify ../../models/anima-turbo/transformer/model.f16.safetensors

MUST: CLI は**引数を解釈しない**。先頭の 1 語でディスパッチし、残りはそのまま対応する
`main(argv)` へ渡す（`--help` も素通しするので、使い方は各本体の parser が出す）。ここに
引数の写しを持つと、本体が持つ排他規則（**沈黙誤値の門**）が CLI 側の写しでは抜けたまま通る
形が生まれ、しかも片方だけ古くなっても誰も落ちない。

NOTE: `export-*` サブコマンドは**この名簿から全部降りた** — export 台本は 1 本残らず
`tools/export-recipes/<family>/` へ出て、リポ直下にも wheel にも無い（ADR 0065 段 3+4）。
起動は `uv run python -m <family>.export`（export-recipes ルートから）。台本をパス指定で
読み込む機構（旧 `load_script`）も使い手が消えたので撤去した — 半端に残すと「wheel に無い
ものを wheel の CLI が読む」逆流の足場が残る。

NOTE: `karume dist` は core の受理集合（{@link karume.dist.PIPELINES} — 今は**空**）で走るので、
family を組むには repo driver（`tools/export-recipes/dist.py`）を使う。この席を残すのは
組み立てエンジン自体が core の責務だから（ADR 0065 決定 1）。

MUST: ディスパッチは遅延 import — `karume verify` を `karume.dist` の import 抜きで起動できる
形に保つ（逆も同じ）。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence


def run_dist(argv: Sequence[str]) -> None:
    from karume import dist

    return dist.main(argv)


def run_verify(argv: Sequence[str]) -> None:
    from karume import verify

    return verify.main(argv)


#: サブコマンド名 → （ハンドラ, 一覧に出す 1 行）。順序がそのまま `--help` の並び。
COMMANDS: Mapping[str, tuple[Callable[[Sequence[str]], None], str]] = {
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
