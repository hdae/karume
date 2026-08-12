"""`karume` コマンド — export / export-sbv2 / dist / verify のサブコマンド式ディスパッチ。

    karume export --dtype f16 --dit-graph dyn --lora turbo.safetensors  # 台本 export_anima.py
    karume export-sbv2 --dtype i8 --target front                        # 台本 export_sbv2.py
    karume dist --pipeline sbv2                                         # karume.dist
    karume verify ../../models/anima-turbo/transformer/model.f16.safetensors

MUST: CLI は**引数を解釈しない**。先頭の 1 語でディスパッチし、残りはそのまま対応する
`main(argv)` へ渡す（`--help` も素通しするので、使い方は各本体の parser が出す）。ここに
引数の写しを持つと、台本が持つ排他規則（`--verify` × `--target` のような**沈黙誤値の門**）が
CLI 側の写しでは抜けたまま通る形が生まれ、しかも片方だけ古くなっても誰も落ちない。

DECIDED: 台本の選択も**サブコマンド名**で綴る（`karume export-sbv2`）— `karume export
--pipeline sbv2` にすると CLI がフラグを 1 つ読んで素通しから外すことになり、上の MUST が
「1 つだけなら」で崩れる。素の `karume export` は Anima のまま据え置く（既存の綴りを動かすと
呼び出し側が黙って別の台本に届く）。

MUST: ディスパッチは遅延 import — `karume dist` / `karume verify` を、export 台本
（diffusers / style_bert_vits2 / モデル定義の重い import）を 1 つも読まずに起動できる形に
保つ。台本の読み込みは {@link load_script} の中でだけ起きる。

NOTE: 台本（`export_anima.py` / `export_sbv2.py`）は**パッケージ外のリポジトリ直下
スクリプト**なので、import 経路ではなくパス指定で読む。wheel には入らない — 無ければ
fail loudly で置き場を示す。
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

#: `karume export-sbv2` が包む台本（ADR 0013 の emit ターゲット 5 本を書き出す側）。
EXPORT_SBV2_SCRIPT = "export_sbv2"

#: `karume export-embeddinggemma` が包む台本（文埋め込み 1 系列を書き出す側）。
EXPORT_EMBEDDINGGEMMA_SCRIPT = "export_embeddinggemma"

#: `karume export-irodori` が包む台本（TTS のテキスト〜DiT 6 グラフを書き出す側）。
EXPORT_IRODORI_SCRIPT = "export_irodori"

#: `karume export-dacvae` が包む台本（Irodori のコーデック 2 グラフを書き出す側）。
EXPORT_DACVAE_SCRIPT = "export_dacvae"

#: `karume export-deberta` が包む台本（実重み DeBERTa-v2 の系列を書き出す側）。
EXPORT_DEBERTA_SCRIPT = "export_deberta"


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


def run_export_sbv2(argv: Sequence[str]) -> None:
    return load_script(EXPORT_SBV2_SCRIPT).main(argv)


def run_export_embeddinggemma(argv: Sequence[str]) -> None:
    return load_script(EXPORT_EMBEDDINGGEMMA_SCRIPT).main(argv)


def run_export_irodori(argv: Sequence[str]) -> None:
    return load_script(EXPORT_IRODORI_SCRIPT).main(argv)


def run_export_dacvae(argv: Sequence[str]) -> None:
    return load_script(EXPORT_DACVAE_SCRIPT).main(argv)


def run_export_deberta(argv: Sequence[str]) -> None:
    return load_script(EXPORT_DEBERTA_SCRIPT).main(argv)


def run_dist(argv: Sequence[str]) -> None:
    from karume import dist

    return dist.main(argv)


def run_verify(argv: Sequence[str]) -> None:
    from karume import verify

    return verify.main(argv)


#: サブコマンド名 → （ハンドラ, 一覧に出す 1 行）。順序がそのまま `--help` の並び。
COMMANDS: Mapping[str, tuple[Callable[[Sequence[str]], None], str]] = {
    "export": (run_export, "Anima を IR v1 + golden io へ書き出す（台本 export_anima.py）"),
    "export-sbv2": (
        run_export_sbv2,
        "SBV2 を IR v1 + golden io へ書き出す（台本 export_sbv2.py）",
    ),
    "export-embeddinggemma": (
        run_export_embeddinggemma,
        "EmbeddingGemma を IR v1 + golden io へ書き出す（台本 export_embeddinggemma.py）",
    ),
    "export-irodori": (
        run_export_irodori,
        "Irodori-TTS のテキスト〜DiT を IR v1 + golden io へ書き出す（台本 export_irodori.py）",
    ),
    "export-dacvae": (
        run_export_dacvae,
        "Irodori のコーデック（DACVAE）を IR v1 + golden io へ書き出す（台本 export_dacvae.py）",
    ),
    "export-deberta": (
        run_export_deberta,
        "実重み DeBERTa-v2 を IR v1 + golden io へ書き出す（台本 export_deberta.py）",
    ),
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
