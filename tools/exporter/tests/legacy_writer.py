"""**テスト専用**の最小 safetensors 書き手 — 移行 CLI の入力（旧配布形）を合成する。

配布形は `krm` へ移ったので、safetensors 方言を書くコードは製品側に 1 本も無い
（container-v1 §12: 旧形式を**読む**のは `karume.legacy` だけ・**書く**のはここだけ）。
`karume migrate` の被験体を作るために、旧書き手が守っていた規約だけをここで写す:

- 先頭 shard は `karume_ir` だけを持ち、データ節は空（ADR 0081）。
- 実データは後続の weight shard に載り、`__metadata__` は空。
- データ節の並びは**整列単位の降順**（F32 / I32 / I4 / I2 → F16 → I8）で、同群は名前昇順。
  要素数が奇数の F16 はさらに後ろへ寄せる（ADR 0063 — Karume のリーダは絶対 offset が
  dtype の整列単位に載ることを要求する）。
- i4 / i2 は **shape が論理形のまま**で、バイト長だけが bit 幅から決まる（ADR 0069 決定 2）。

MUST: 素材（グラフ・格納テンソル・scale）は {@link ir_fixtures.fixture_spec} と**同じ 1 本**
から取る。素材を割ると「旧形から移した容器」と「直接書いた容器」のバイト同一が、移行の
問題ではなく素材の違いで崩れる。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from ir_fixtures import Shape, fill_spec, fixture_spec

from karume.container import codec_entry, numbered_path
from karume.emit import StoredModel, stored_model
from karume.ir import IrGraph
from karume.legacy import IR_METADATA_KEY
from karume.ple import ple_row_bytes

#: 格納 dtype → (safetensors dtype 名, 1 要素の bit 数)。旧書き手の表の写し。
_STORED_DTYPE: Mapping[str, tuple[str, int]] = {
    "f32": ("F32", 32),
    "f16": ("F16", 16),
    "bf16": ("BF16", 16),
    "i32": ("I32", 32),
    "i8": ("I8", 8),
    "i4": ("I4", 4),
    "i2": ("I2", 2),
}

#: 並び順の第 1 キー（整列単位の降順）。
_DTYPE_GROUP = {"F32": 0, "I32": 1, "I4": 2, "I2": 2, "BF16": 3, "F16": 3, "I8": 4}

#: ヘッダ長はこの倍数へ空白で詰める（データ節先頭を 4 バイト境界へ載せる）。
_HEADER_ALIGN = 8


@dataclass(frozen=True)
class Entry:
    """データ節に並ぶテンソル 1 本の宣言（**論理** shape・格納後のバイト長）。"""

    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def write_safetensors(entries: Sequence[Entry], metadata: Mapping[str, str]) -> bytes:
    """`[u64 LE ヘッダ長][ヘッダ JSON][データ節]` を**宣言の順で**組む。"""
    header: dict[str, object] = {"__metadata__": dict(metadata)}
    offset = 0
    for entry in entries:
        header[entry.name] = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.nbytes],
        }
        offset += entry.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % _HEADER_ALIGN)
    return len(blob).to_bytes(8, "little") + blob + b"".join(entry.payload for entry in entries)


def order(entries: Sequence[Entry]) -> list[Entry]:
    """データ節に並ぶ順（整列単位の降順 → 奇数 F16 は後ろ → 名前昇順）。"""
    return sorted(
        entries,
        key=lambda entry: (
            _DTYPE_GROUP[entry.dtype],
            1 if entry.dtype == "F16" and entry.nbytes % 4 else 0,
            entry.name,
        ),
    )


def _scale_shape(shape: Sequence[int], encoding) -> tuple[int, ...]:
    """旧配布形が持っていた scale の形（group 形 / per-channel の keepdim 形）。"""
    if codec_entry(encoding.codec).grouping == "group":
        return (shape[0], math.prod(shape[1:]) // encoding.group_size)
    return tuple(dim if axis == encoding.row_axis else 1 for axis, dim in enumerate(shape))


def legacy_entries(stored: StoredModel) -> list[Entry]:
    """格納変換を済ませた素材 → 旧配布形のデータ節の宣言（並べ替え済み）。

    受けるのは {@link karume.emit.stored_model} の戻り**そのもの** — 同じ素材から 2 つの
    `StoredModel` を作ると、格納変換を 2 度回したうえに「どちらのグラフを焼いたか」が
    呼び手ごとに割れる。
    """
    committed = stored.graph
    entries: list[Entry] = []
    for name, initializer in committed.initializers.items():
        if initializer.is_shared:
            continue
        key = initializer.tensor
        shape = tuple(int(dim) for dim in committed.values[name].shape)
        label, _ = _STORED_DTYPE[initializer.storage.dtype]
        entries.append(Entry(key, label, shape, bytes(stored.tensors[key])))
        scale_key = initializer.storage.scale
        if scale_key is not None:
            payload = bytes(stored.tensors[scale_key])
            entries.append(
                Entry(scale_key, "F32", _scale_shape(shape, stored.bindings[key]), payload)
            )
    return order(entries)


def legacy_shards(
    *,
    mark: str = "fixture",
    storage: str = "f32",
    inputs: Sequence[tuple[str, Shape]] = (),
    outputs: Sequence[Shape] = ([1],),
    weights: Sequence[str] = ("weight",),
    baked: tuple[str, int] | None = None,
    groups: int = 1,
) -> list[bytes]:
    """旧配布形 1 コンポーネントの shard バイト列（読む順 — 先頭がグラフ shard）。

    `groups` は weight shard の本数（データ節を均等に割る本数ではなく、**宣言の並びを
    その本数へ順に配る**）。移行が「ディレクトリを跨ぐ列」や「複数 shard」を解くところを
    踏むための席。
    """
    graph, tensors, scales, overrides = fixture_spec(mark, storage, inputs, outputs, weights, baked)
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )
    return _shards(stored.graph, legacy_entries(stored), groups)


def piece_entries(entry: Entry, rows: Sequence[tuple[int, int]]) -> list[Entry]:
    """親 1 本を行範囲 `rows` の piece 列へ割る（旧 shard 仕様 v3 — ADR 0090）。

    piece のキーは `<親名>#NNNNN-of-NNNNN`（index は 1 始まり・総数は `len(rows)`）、shape は
    `(行数, *残り次元)` で、payload は親の行優先バイト列の該当区間。
    """
    row_bytes = entry.nbytes // entry.shape[0]
    count = len(rows)
    return [
        Entry(
            f"{entry.name}#{index:05d}-of-{count:05d}",
            entry.dtype,
            (stop - start, *entry.shape[1:]),
            entry.payload[start * row_bytes : stop * row_bytes],
        )
        for index, (start, stop) in enumerate(rows, start=1)
    ]


def legacy_piece_shards(
    *, parent: str, pieces: Sequence[tuple[int, int]], mark: str = "fixture", storage: str = "f32"
) -> list[bytes]:
    """{@link legacy_shards} と同じ素材で、テンソル `parent` を piece 列に割った shard 列。

    旧書き手の配置（旧読み手契約 5）を写す: piece 1 は他のテンソルと同じ先頭の weight shard に、
    piece 2 以降は**連続する後続の shard に 1 本ずつ**置く。
    """
    graph, tensors, scales, overrides = fixture_spec(mark, storage)
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )
    entries = legacy_entries(stored)
    target = next(entry for entry in entries if entry.name == parent)
    split = piece_entries(target, pieces)
    rest = [entry for entry in entries if entry.name != parent]
    return [
        write_safetensors([], {IR_METADATA_KEY: stored.graph.to_json()}),
        write_safetensors(order([*rest, split[0]]), {}),
        *(write_safetensors([piece], {}) for piece in split[1:]),
    ]


def legacy_fill_shards(count: int, *, mark: str) -> list[bytes]:
    """`count` 本の shard 列になる旧配布形（先頭がグラフ shard・以降 1 本 1 テンソル）。"""
    if count < 2:
        raise ValueError(f"shard 数 {count} は 2 以上（先頭はテンソルを持たないグラフ shard）")
    graph, tensors = fill_spec(count - 1, mark=mark)
    stored = stored_model(graph, tensors)
    return _shards(stored.graph, legacy_entries(stored), count - 1)


def _shards(committed: IrGraph, entries: Sequence[Entry], groups: int) -> list[bytes]:
    """グラフ shard + weight shard 列（宣言の並びを `groups` 本へ順に配る）。"""
    shards = [write_safetensors([], {IR_METADATA_KEY: committed.to_json()})]
    if not entries:
        return shards
    per = -(-len(entries) // groups)
    for index in range(0, len(entries), per):
        shards.append(write_safetensors(entries[index : index + per], {}))
    return shards


def stage_shards(directory: Path, name: str, shards: Sequence[bytes]) -> Path:
    """旧配布形を連番で置き、コンポーネントの代表 path を返す。"""
    directory.mkdir(parents=True, exist_ok=True)
    representative = directory / name
    for index, blob in enumerate(shards, start=1):
        numbered_path(representative, index, len(shards)).write_bytes(blob)
    return representative


#: 旧 `extras` の実在 1 種（`karume.migrate.EXTRA_ASSETS` の鍵）— 容器の資産 `rope_base` の素。
ROPE_BASE_NAME = "rope_base"

#: 旧 PLE sidecar の索引ファイル名（`karume.ple.PLE_INDEX_ASSET` + `.json`）。
PLE_INDEX_FILE = "ple.json"

#: 旧 PLE sidecar の shard 名（1 本だけ置く — 境界の意味は移行が捨てる）。
PLE_SHARD_FILE = "ple-00001.safetensors"

#: 旧 PLE shard が `__metadata__` に持つキー（`karume.migrate.PLE_METADATA_KEY` の写し）。
_PLE_METADATA_KEY = "karume_ple"


def legacy_rope_base(directory: Path, *, values: Sequence[float] = (0.5, 1.5, 2.5, 3.5)) -> bytes:
    """旧 `extras` の `rope_base.safetensors` を置き、その**バイト列**を返す。

    容器の資産は「重みではないが同じ容器で配るバイト列」の席なので、移行はこのファイルを
    **1 バイトも変えずに** payload として畳む（`karume.migrate._extra_assets`）。直接書く側は
    同じバイト列を `AssetInput` に載せるので、両経路の part 列はバイト同一になる MUST。
    """
    payload = write_safetensors(
        [
            Entry(
                "rope.base",
                "F32",
                (len(values),),
                torch.tensor(values, dtype=torch.float32).numpy().tobytes(),
            )
        ],
        {},
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{ROPE_BASE_NAME}.safetensors").write_bytes(payload)
    return payload


@dataclass(frozen=True)
class LegacyPle:
    """置いた旧 PLE sidecar（schema 2）と、直接書く側が同じ資産を組むのに要る事実。"""

    tokens: int
    layers: int
    dim: int
    storage: str
    embed_scale: float
    #: token 順に連結した行優先 payload（`values` / `scales`）。
    payloads: Mapping[str, bytes]


def legacy_ple_sidecar(
    directory: Path,
    *,
    tokens: int = 8,
    layers: int = 2,
    dim: int = 8,
    storage: str = "i8",
    embed_scale: float = 0.125,
) -> LegacyPle:
    """旧 PLE sidecar（`ple.json` schema 2 + shard 1 本）を置き、素材を返す。

    移行 CLI（`karume.migrate._ple_assets`）はこの 2 ファイルを読み、索引 schema 3 と
    `ple.values.<k>` / `ple.scales.<k>` の block 列へ畳む。直接書く側は同じ payload を
    `karume.ple.ple_assets` へ渡すので、両経路の part 列はバイト同一になる MUST。
    """
    row_bytes = ple_row_bytes(storage, layers, dim)
    payloads = {
        key: bytes((index * 7 + 3) % 251 for index in range(tokens * stride))
        for key, stride in row_bytes.items()
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PLE_INDEX_FILE).write_text(
        json.dumps(
            {
                "schema": 2,
                "storage": storage,
                "tokens": tokens,
                "layers": layers,
                "dim": dim,
                "embedScale": embed_scale,
                "shards": [{"file": PLE_SHARD_FILE, "start": 0, "stop": tokens}],
            }
        )
    )
    (directory / PLE_SHARD_FILE).write_bytes(
        write_safetensors(
            [
                Entry(
                    "values",
                    _PLE_SHARD_DTYPE[storage],
                    (tokens, dim * layers),
                    payloads["values"],
                ),
                Entry("scales", "F32", (tokens, layers), payloads["scales"]),
            ],
            {
                _PLE_METADATA_KEY: json.dumps(
                    {
                        "schema": 2,
                        "storage": storage,
                        "tokens": tokens,
                        "layers": layers,
                        "dim": dim,
                        "start": 0,
                        "stop": tokens,
                    }
                )
            },
        )
    )
    return LegacyPle(tokens, layers, dim, storage, embed_scale, payloads)


#: PLE の格納 → shard の safetensors dtype 名（`values` 側だけが packed になりうる）。
_PLE_SHARD_DTYPE: Mapping[str, str] = {"i8": "I8", "i4": "I4", "i2": "I2"}
