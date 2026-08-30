"""recipe のテストが shard 列（`ir_fixtures` の戻り）を系列 / 配布形として扱うための糊。

配布形は常に分割される（ADR 0081）ので、`ir_fixtures.ir_container` が返すのは 1 ファイルぶんの
バイト列ではなく **shard 列**（先頭 = グラフ shard・以降 = weight shard）になった。系列側へ
置くにも、組み立て後の現物と突き合わせるにも、代表 path から連番 path を導く同じ 1 本の綴りが
要る — それが `karume.shards` なので、ここが持つのはその薄い呼び出しだけである。

置き場がここ 1 箇所なのは、7 家族の `tests/test_distribution.py` が同じ 3 手を踏むため
（系列へ書く / 期待 path を並べる / 置かれた現物を読む）。連番の綴りを各 family へ写すと、
分割規則が動いた日にテストだけが古びる。`tools/export-recipes/conftest.py` がこのディレクトリを
sys.path へ張る（`ir_fixtures` と同じ経路）。

MUST: 正当なコンポーネントの代表 path 自身（`model.safetensors`）は**書かない** — 単一ファイルと
連番の同居は「どちらを配るか」が一意に決まらないので `karume.shards.resolve_shards` が
fail loudly する。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from karume.dist import WeightFiles
from karume.shards import resolve_shards, shard_name, shard_path, shard_siblings

#: `ir_fixtures.ir_container` の戻りが割れる shard 本数（グラフ shard + weight shard 1 本）。
#: 合成の資産は数 KB なので weight shard は 1 本に収まるが、**常時分割**（ADR 0081）なので
#: 1 本きりにはならない。ここが狂えば期待 path の突合（{@link placed_paths}）がそのまま落ちる。
CONTAINER_SHARDS = 2


def write_component(path: Path, payload: bytes | Sequence[bytes]) -> None:
    """代表 path が指すコンポーネントを系列へ置く（親ディレクトリごと作る）。

    形は 2 つあり、どちらを渡すかがそのまま「実物どおりの配布形か、門に落とすための偽資産か」の
    区別になる:

    - `Sequence[bytes]`（`ir_fixtures` の戻り）は **shard 連番**へ書く。読む順そのままで、
      先頭がグラフ shard。
    - `bytes` は代表 path **自身**へ 1 本で書く。これは分割の前段（計画の門）で止まることを
      見るための偽資産の席で、連番にする理由が無い（組み立てまで届く席には使えない —
      グラフ shard 空の門が名指しで落とす）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
        return
    total = len(payload)
    for index, shard in enumerate(payload, start=1):
        shard_path(path, index, total).write_bytes(shard)


def replace_component(path: Path, payload: bytes | Sequence[bytes]) -> None:
    """既に置かれているコンポーネントを**別の中身へ差し替える**（前の形の現物を消してから書く）。

    系列の 1 席だけを取り違えた資産へ差し替えて門を試す、という形の席。単一ファイルと連番の
    同居は `karume.shards.resolve_shards` が fail loudly するので、前の形（分割 / 単一）の
    現物を先に払わないと、**試したい門の手前で別の理由で落ちる**。
    """
    for stale in shard_siblings(path):
        stale.unlink()
    write_component(path, payload)


def read_component(path: Path) -> list[bytes]:
    """代表 path が指すコンポーネントの現物を、読む順の shard バイト列として返す。

    `write_component` の逆で、`ir_fixtures` の戻り（`list[bytes]`）とそのまま比較できる。
    """
    return [shard.read_bytes() for shard in resolve_shards(path)]


def shard_paths(rel_path: str, total: int = CONTAINER_SHARDS) -> list[str]:
    """代表の相対 path → 配布形に現れる連番 path 列。"""
    return [shard_name(rel_path, index, total) for index in range(1, total + 1)]


def placed_paths(
    output_paths: Mapping[str, str],
    weights: Mapping[str, Mapping[str, WeightFiles]],
) -> list[str]:
    """出力 path 表 → 配布形に現れる相対 path 列（**weights の席だけ**を連番へ展開する）。

    展開が掛かる席の判定は `karume.dist.expand_weight_shards` と同じ導出 — weights 宣言が
    指す役割名の集合で、assets / extras の席（1 ファイル参照）はそのまま残る。recipe 側で
    「どの役割が分割されるか」を手で列挙すると、席が増えた日に期待値だけが古びる。
    """
    expanded = {files.file for labels in weights.values() for files in labels.values()}
    return [
        rel
        for role, path in output_paths.items()
        for rel in (shard_paths(path) if role in expanded else [path])
    ]
