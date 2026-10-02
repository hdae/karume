"""Wan2.1 の上流取得元（HF リポと revision の pin）と、部品の取得（ADR 0118 決定 5 / 7）。

`SOURCES` が上流取得元の表の正本。値の revision は HF API で解決した commit SHA（40 桁）で、
ADR 0092 決定 3（配布リポの revision を焼く流儀）を上流側にも当てたもの。recipe の台本は
ここを通して上流を読む — 台本ごとに `from_pretrained(repo)` を綴ると、既定ブランチの先頭が
動いた日に参照と export が黙って別の checkpoint を読む。

取得するのは DiT（`transformer`）・VAE（`vae`）・scheduler の config と、ライセンスの
front matter を持つ `README.md`・部品の索引 `model_index.json` だけ。umT5（`text_encoder`・
f32 で約 22.7 GB）とトークナイザは取らない — 段 6 のテキスト埋め込みの別プロセスが取る
（開発機のホスト RAM 31 GiB に umT5 と DiT を同居させない — ADR 決定 4）。

    uv run --group wan python -m wan.sources --fetch     # 取得（HF の既定キャッシュへ）
    uv run --group wan python -m wan.sources             # 取得済みの検査（パラメータ数）

MUST: `huggingface_hub` は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpstreamSource:
    """上流の HF リポ 1 本（repo + pin した revision + ライセンス識別子）。"""

    repo: str
    #: HF の commit SHA（40 桁の小文字 16 進）。ブランチ名やタグは置かない。
    revision: str
    #: 上流 `README.md` の front matter の `license:`（2026-10-02 に HF API で確認）。
    license: str


@dataclass(frozen=True)
class Component:
    """上流リポの部品 1 つ（`subfolder`）と、それを取得するか。"""

    subfolder: str
    fetch: bool
    #: 取得の要否の理由（台本と報告の読み手向け）。
    why: str


#: 上流取得元の表。キーは配布形のモデル名（`models/karume-wan2.1/` のモデル `t2v-1.3b`
#: — ADR 0118 決定 7）。Wan2.1 の 14B などを足すときは行を足す。
SOURCES: dict[str, UpstreamSource] = {
    "t2v-1.3b": UpstreamSource(
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        revision="0fad780a534b6463e45facd96134c9f345acfa5b",
        license="apache-2.0",
    ),
}

#: 既定のモデル（今は 1 本だけ）。
DEFAULT_MODEL = "t2v-1.3b"

#: 上流リポの部品の列挙。`fetch=False` の部品はこの recipe の取得では落とさない。
COMPONENTS: tuple[Component, ...] = (
    Component("transformer", fetch=True, why="DiT（段 2 / 3）— fp32 safetensors 2 分割・約 5.7 GB"),
    Component("vae", fetch=True, why="動画 VAE（段 4 / 5）— fp32 約 0.5 GB"),
    Component("scheduler", fetch=True, why="UniPC の config（shift 3.0 — 決定 5）"),
    Component("text_encoder", fetch=False, why="umT5-XXL f32 約 22.7 GB — 段 6 の別プロセスが取る"),
    Component("tokenizer", fetch=False, why="umT5 のトークナイザ — 段 6 の別プロセスが取る"),
)

#: 部品の外で取得するリポ直下のファイル（ライセンスの front matter と部品の索引）。
ROOT_FILES: tuple[str, ...] = ("README.md", "model_index.json")

#: 取得済みの checkpoint が持つべきパラメータ数（safetensors のヘッダの要素数の和）。
#: 値は調査（2026-10-02）の値で、pin した revision の実物でも一致を確かめた。VAE は encoder を含む。
EXPECTED_PARAMETERS: dict[str, int] = {
    "transformer": 1_418_996_800,
    "vae": 126_892_531,
}

#: safetensors のヘッダの dtype → 要素あたりバイト数（ヘッダの整合検査用）。
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1}


class WanSourceError(RuntimeError):
    """上流の checkpoint が pin した形で手元に無い・ヘッダが読めない。"""


def allow_patterns() -> list[str]:
    """`snapshot_download` へ渡す取得対象（取得する部品の配下 + リポ直下の 2 本）。"""
    return [*ROOT_FILES, *(f"{part.subfolder}/*" for part in COMPONENTS if part.fetch)]


def fetch(model: str = DEFAULT_MODEL) -> Path:
    """pin した revision の取得対象を HF の既定キャッシュへ落とし、snapshot の置き場を返す。"""
    from huggingface_hub import snapshot_download

    source = SOURCES[model]
    return Path(
        snapshot_download(source.repo, revision=source.revision, allow_patterns=allow_patterns())
    )


def local_snapshot(model: str = DEFAULT_MODEL) -> Path:
    """取得済みの snapshot のディレクトリ（ネットワークに出ない）。

    無ければ fail loudly — 台本が黙って取得を始めると、5.7 GB の取得が参照作成の途中に紛れる。
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = SOURCES[model]
    try:
        snapshot = Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=allow_patterns(),
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"{source.repo}@{source.revision} が HF キャッシュに無い — 先に"
            " `uv run --group wan python -m wan.sources --fetch` で取得する"
        ) from error
    missing = [part.subfolder for part in COMPONENTS if part.fetch]
    missing = [name for name in missing if not (snapshot / name).is_dir()]
    if missing:
        raise WanSourceError(f"{snapshot} に部品 {missing} が無い — `--fetch` で取り直す")
    return snapshot


def read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """safetensors のヘッダ（先頭 8 バイトの長さ + JSON）だけを読む（本体は読まない）。"""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return header


def count_parameters(component_dir: Path) -> int:
    """部品のディレクトリにある safetensors の全テンソルの要素数の和。

    ヘッダの `data_offsets` の幅と `dtype × shape` が食い違う行は fail loudly（ヘッダの読み違い
    で数が合ってしまう形を潰す）。
    """
    files = sorted(component_dir.glob("*.safetensors"))
    if not files:
        raise WanSourceError(f"{component_dir} に safetensors が無い")
    total = 0
    for file in files:
        for name, entry in read_safetensors_header(file).items():
            shape = entry["shape"]
            dtype = entry["dtype"]
            offsets = entry["data_offsets"]
            assert isinstance(shape, list) and isinstance(dtype, str) and isinstance(offsets, list)
            count = 1
            for size in shape:
                count *= int(size)
            if count * _DTYPE_BYTES[dtype] != int(offsets[1]) - int(offsets[0]):
                raise WanSourceError(f"{file} の {name}: dtype {dtype} × {shape} と幅が合わない")
            total += count
    return total


def check_snapshot(snapshot: Path) -> dict[str, int]:
    """取得済みの部品のパラメータ数を数え、{@link EXPECTED_PARAMETERS} と突き合わせる。"""
    counts = {name: count_parameters(snapshot / name) for name in EXPECTED_PARAMETERS}
    wrong = {
        name: (counts[name], expected)
        for name, expected in EXPECTED_PARAMETERS.items()
        if counts[name] != expected
    }
    if wrong:
        raise WanSourceError(f"パラメータ数が調査の値と違う（実物, 期待）: {wrong}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SOURCES))
    parser.add_argument("--fetch", action="store_true", help="pin した revision で取得する")
    args = parser.parse_args(argv)

    snapshot = fetch(args.model) if args.fetch else local_snapshot(args.model)
    counts = check_snapshot(snapshot)
    print(f"snapshot: {snapshot}")
    for name, count in counts.items():
        print(f"{name}: {count:,} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
