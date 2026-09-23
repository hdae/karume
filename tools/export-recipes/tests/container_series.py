"""recipe のテストが part 列（`ir_fixtures` の戻り）を系列 / 配布形として扱うための糊。

配布形は常に分割される（container-v1 §8）ので、`ir_fixtures.ir_container` が返すのは 1 ファイル
ぶんのバイト列ではなく **part 列**（part 0 = 2 文書・part 1 = const 領域・part 2 以降 = 重み）に
なる。系列側へ置くにも、組み立て後の現物と突き合わせるにも、代表 path から連番 path を導く
同じ 1 本の綴りが要る — それが `karume.container` の連番規約なので、ここが持つのはその薄い
呼び出しだけである。

置き場がここ 1 箇所なのは、7 家族の `tests/test_distribution.py` が同じ 3 手を踏むため
（系列へ書く / 期待 path を並べる / 置かれた現物を読む）。連番の綴りを各 family へ写すと、
分割規則が動いた日にテストだけが古びる。`tools/export-recipes/conftest.py` がこのディレクトリを
sys.path へ張る（`ir_fixtures` と同じ経路）。

MUST: 正当なコンポーネントの代表 path 自身（`model.krm`）は**書かない** — 単一ファイルと
連番の同居は「どちらを配るか」が一意に決まらないので `karume.container.container_parts` が
fail loudly する。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from karume.container import container_parts, numbered_name, numbered_path, sequence_siblings
from karume.dist import WeightFiles

#: `ir_fixtures.ir_container` の戻りが割れる part 本数（2 文書 + const 領域 + 重み 1 part）。
#: 合成の資産は数 KB なので重みは 1 part に収まるが、**常時分割**（container-v1 §8）なので
#: 1 本きりにはならない。ここが狂えば期待 path の突合（{@link placed_paths}）がそのまま落ちる。
CONTAINER_PARTS = 3


def write_component(path: Path, payload: bytes | Sequence[bytes]) -> None:
    """代表 path が指すコンポーネントを系列へ置く（親ディレクトリごと作る）。

    形は 2 つあり、どちらを渡すかがそのまま「実物どおりの配布形か、門に落とすための偽資産か」の
    区別になる:

    - `Sequence[bytes]`（`ir_fixtures` の戻り）は **part 連番**へ書く。読む順そのままで、
      先頭が part 0（ヘッダ + 2 文書）。
    - `bytes` は代表 path **自身**へ 1 本で書く。これは容器として開く前段（計画の門）で止まる
      ことを見るための偽資産の席で、連番にする理由が無い。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
        return
    total = len(payload)
    for index, part in enumerate(payload, start=1):
        numbered_path(path, index, total).write_bytes(part)


def replace_component(path: Path, payload: bytes | Sequence[bytes]) -> None:
    """既に置かれているコンポーネントを**別の中身へ差し替える**（前の形の現物を消してから書く）。

    系列の 1 席だけを取り違えた資産へ差し替えて門を試す、という形の席。単一ファイルと連番の
    同居は `karume.container.container_parts` が fail loudly するので、前の形（分割 / 単一）の
    現物を先に払わないと、**試したい門の手前で別の理由で落ちる**。
    """
    for stale in sequence_siblings(path):
        stale.unlink()
    write_component(path, payload)


def read_component(path: Path) -> list[bytes]:
    """代表 path が指すコンポーネントの現物を、読む順の part バイト列として返す。

    `write_component` の逆で、`ir_fixtures` の戻り（`list[bytes]`）とそのまま比較できる。
    """
    return [part.read_bytes() for part in container_parts(path)]


def part_paths(rel_path: str, total: int = CONTAINER_PARTS) -> list[str]:
    """代表の相対 path → 配布形に現れる連番 path 列。"""
    return [numbered_name(rel_path, index, total) for index in range(1, total + 1)]


def placed_paths(
    output_paths: Mapping[str, str],
    weights: Mapping[str, Mapping[str, WeightFiles]],
    totals: Mapping[str, int] = {},
) -> list[str]:
    """出力 path 表 → 配布形に現れる相対 path 列（**weights の席だけ**を連番へ展開する）。

    展開が掛かる席の判定は `karume.dist.expand_weight_parts` と同じ導出 — weights 宣言が
    指す役割名の集合で、assets の席（1 ファイル参照）はそのまま残る。recipe 側で「どの役割が
    分割されるか」を手で列挙すると、席が増えた日に期待値だけが古びる。

    `totals` は役割名 → part 本数の上書き（既定は {@link CONTAINER_PARTS}）— 資産を同梱する
    容器は専用 part がその分だけ増えるので、席ごとに本数が違う family がここで名乗る。
    """
    expanded = {files.file for labels in weights.values() for files in labels.values()}
    return [
        rel
        for role, path in output_paths.items()
        for rel in (
            part_paths(path, totals.get(role, CONTAINER_PARTS)) if role in expanded else [path]
        )
    ]
