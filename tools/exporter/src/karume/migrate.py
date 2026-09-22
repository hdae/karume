"""旧配布形（safetensors の shard 列）→ コンテナ形式（`krm` / `krg`）へ移す（container-v1 §12）。

    # 部品 1 つ（段 1 の面）
    karume migrate ../../models/karume-depth-anything-v2/small/depth/model.f32.safetensors \
        --out /tmp/migrated --license apache-2.0
    # リポ丸ごと（段 2 の面 — `karume/5` の karume.json まで書く・ADR 0109 決定 8）
    karume migrate --manifest ../../models/karume-irodori/karume.json \
        --out /tmp/karume-irodori --license apache-2.0

**旧形式を読む処理はこの層にだけ置く**（§12）— 新しい読み手は `karume/5` と新コンテナしか読まず、
両読みは実装しない。ここが動かすのは**容器と宣言の形**だけである:

- **payload の生バイトは 1 バイトも変えない**（`repack.py` の不変条件 1 を継承）。末尾の詰め物は
  §4.1 のとおり**新たに焼かれる**ので、突き合わせるのは block 全体ではなく initializer ごとの
  payload（旧 shard の実体の sha256 = 新 block の payload 部の sha256）。
- **IR は v1 → v2 へ再 serialize する**（`repack.py` の不変条件 2「`karume_ir` の逐語同一」は
  退役）。改名と正準直列化は {@link karume.container.ir_v2_document}。
- **codec は台帳へ写す**（`i8 → int8-sym` / `i4 → int4-sym-g` / `i2 → int2-off`）。`ternary` へは
  写さない — 値域が部分集合でも「三値である」という主張は量子化器の側がするもの。
- **`rowAxis` は消費側 op から引く**（`emit.weight_channel_axes` の鏡像 — `conv_transpose1d`
  だけ 1）。旧 keepdim 形の scale が**その軸の形ちょうど**であることを現物の宣言で確かめてから
  焼く（バイト数だけ合わせると per-column の scale を per-channel として宣言できてしまう）。

MUST: 自己検査（書いたものを読み直して initializer ごとに sha256 を突き合わせる）を**通してから
据える** — 書き出しは一時 path（`.partial`）へ行い、検査が通った回だけ `os.replace` で本番名へ
移す（`repack.repack_component` と同じ規律）。**旧入力は読むだけ**で、消しも書き換えもしない。

リポ丸ごとモード（`--manifest`）が足すのは 4 つである（ADR 0109 決定 3 / 4 / 8）:

- shard 列を**旧 manifest の宣言から**組む（代表 path からの復元はしない — ディレクトリを跨ぐ
  列がこれで解ける）。同じ列を複数モデルが指すときは 1 度だけ変換する。
- 旧 `extras`（実在は `rope_base` 1 種）と **PLE sidecar** を容器の資産へ畳む。PLE は索引
  （`ple_index` — schema 3）と `values` / `scales` の block 列になり、manifest の `assets` からは
  消える。
- 越境参照（FileRef に `repo` / `revision`）の列は変換せず、`--cross-repo` が指す**変換済み
  ディレクトリ**の `karume/5` から `container` を引き写して `repo` / `revision` を付ける。
- 変換しなかった通常ファイル（README / LICENSE / tokenizer / …）をそのまま複写し、
  `karume/5` の `karume.json` を書く。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from karume.container import (
    BLOCK_MAX_BYTES,
    CODEC_FOR_STORAGE,
    DEFAULT_PART_BYTES,
    GRAPH_NAME_PATTERN,
    AssetInput,
    Encoding,
    Provenance,
    ReadContainer,
    canonical_json,
    codec_entry,
    container_paths,
    per_channel_group_size,
    read_container,
    write_model_container,
)
from karume.dist import (
    REPO_RE,
    REVISION_RE,
    assert_manifest_limits,
    file_ref,
    generator_tag,
    manifest_text,
    safetensors_header,
    sha256_file,
)
from karume.emit import EmitError, weight_channel_axes
from karume.ir import IR_METADATA_KEY, IrGraph
from karume.repack import SourceTensor, payload_chunks, read_component
from karume.shards import component_path, resolve_shards, shard_siblings
from karume.verify import BoundGraph, bind_graphs, parse_ir_graph

#: 新コンテナの拡張子（§1 — 種別は magic が持つが、ファイル名も分けておく）。
MODEL_SUFFIX = ".krm"
GRAPH_SUFFIX = ".krg"

#: 読む旧 manifest の形式識別子（両読みはしない — ADR 0109 決定 1）。
LEGACY_MANIFEST_FORMAT = "karume/4"

#: 書く manifest の形式識別子（ADR 0109 決定 1）。
MANIFEST_FORMAT = "karume/5"

#: manifest のファイル名（ADR 0041 §1 — リポジトリ直下の固定名）。
MANIFEST_FILE = "karume.json"

#: 旧 `extras` の名前 → 容器の `(資産名, 役割)`。実在は `rope_base` 1 種だけ（ADR 0109 決定 4）で、
#: 知らない名前は**黙って落とさず** fail loudly にする。
EXTRA_ASSETS: Mapping[str, tuple[str, str]] = {"rope_base": ("rope_base", "rope-base")}

#: PLE sidecar の持ち主（pipeline 名 → 畳み先の部品名 — ADR 0109 決定 4 / 0085）。
PLE_OWNER: Mapping[str, str] = {"gemma4": "model"}

#: 旧 manifest の assets に居る PLE 索引の名前。新しい容器でも同じ名前の資産になる。
PLE_INDEX_ASSET = "ple_index"

#: 新しい PLE 索引の版（block 列を指す形 — 旧 sidecar の schema 2 を置き換える）。
PLE_INDEX_SCHEMA = 3

#: PLE の資産の役割（models 側の解釈者名 — runtime は解釈しない）。
PLE_INDEX_ROLE = "ple-index"
PLE_ROLES: Mapping[str, str] = {"values": "ple-values", "scales": "ple-scales"}

#: 旧 PLE shard が持つメタデータのキー（索引との整合をここで突き合わせる）。
PLE_METADATA_KEY = "karume_ple"

#: 旧 PLE 索引の格納 → 1 バイトに詰まる要素数（`packages/models/src/gemma/ple-index.ts` の鏡像）。
PLE_PACK_FACTOR: Mapping[str, int] = {"i2": 4, "i4": 2}

#: PLE の scale 1 個ぶんのバイト数（f32 — 同上）。
PLE_SCALE_BYTES = 4


class MigrateError(ValueError):
    """移行の前提が破れた（写し先の無い格納・宣言と現物の食い違い・出力先の残骸）。"""


@dataclass(frozen=True)
class DocumentRef:
    """descriptor 1 文書ぶんの期待値（manifest `karume/5` の `container.descriptor`）。"""

    length: int
    sha256: str

    def to_document(self) -> dict[str, Any]:
        return {"length": self.length, "sha256": self.sha256}


@dataclass(frozen=True)
class MigrationResult:
    """1 コンポーネントの移行結果。"""

    #: 据えた `krm` の part 列（part 0 から。単一形なら 1 本）。
    parts: tuple[Path, ...]
    #: `--graph` で書いた `krg`（書かなければ `None`）。
    graph: Path | None
    #: 供給計画を組んだ initializer の本数。
    initializers: int
    #: 旧実体と sha256 を突き合わせた payload の本数（実体 + companion scale）。
    payloads: int
    #: 突き合わせた資産の本数（extras と PLE — 部品単位モードでは 0）。
    assets: int
    #: 2 文書（グラフ記述 / モデル記述）の `(バイト長, sha256)`。
    descriptor: tuple[DocumentRef, DocumentRef]


class _SourcePayloads(Mapping[str, bytes]):
    """テンソルキー → 生バイトの遅延写像（**引かれた 1 本だけ**を読む）。

    書き手（{@link karume.container.write_model_container}）は実体を 1 本ずつ引いて 1 本ずつ
    手放すので、この口を遅延にしておけば全量はメモリに載らない（数十 GB の系列でもピークは
    「最大のテンソル 1 本」）。
    """

    def __init__(self, sources: Mapping[str, SourceTensor]) -> None:
        self._sources = sources

    def __getitem__(self, key: str) -> bytes:
        return b"".join(payload_chunks(self._sources[key]))

    def __iter__(self) -> Iterator[str]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)


def _concrete_shape(graph: IrGraph, name: str, where: str) -> list[int]:
    """initializer の宣言 shape（記号次元を持つ実体は在りえない）。"""
    value = graph.values.get(name)
    if value is None:
        raise MigrateError(f"{where}: `values` に宣言が無い")
    shape: list[int] = []
    for dim in value.shape:
        if not isinstance(dim, int):
            raise MigrateError(f"{where}: initializer の shape に記号次元がある（{dim!r}）")
        shape.append(dim)
    return shape


def container_bindings(graph: IrGraph) -> dict[str, Encoding]:
    """IR v1 の `storage` → コンテナの束縛（**テンソルキー** → {@link Encoding}）。

    共有 initializer（バイトを持たない宣言）は束縛を持たない。`rowAxis` は消費側 op から引き
    （`emit.weight_channel_axes`）、per-channel の `groupSize` は行長（= numel / 行数）になる。
    group codec（`int4-sym-g`）の旧 scale は先頭次元を行として焼かれているので、消費 op が軸 1 を
    要求する形は写せない（fail loudly — 黙って軸 0 として宣言すると値が入れ替わる）。
    """
    try:
        axes = weight_channel_axes(graph)
    except EmitError as cause:
        raise MigrateError(str(cause)) from cause
    bindings: dict[str, Encoding] = {}
    for name, initializer in graph.initializers.items():
        if initializer.is_shared:
            continue
        where = f"initializer '{name}'"
        key = initializer.tensor
        if key is None:
            raise MigrateError(f"{where}: `tensor` も `shared` も無い（IR v1 として不正）")
        storage = initializer.storage
        codec = CODEC_FOR_STORAGE.get(storage.dtype)
        if codec is None:
            raise MigrateError(f"{where}: 旧格納 dtype '{storage.dtype}' の写し先が台帳に無い")
        entry = codec_entry(codec)
        if entry.scale == "forbidden":
            bindings[key] = Encoding(codec)
            continue
        if storage.scale is None:
            raise MigrateError(f"{where}: 量子化格納 '{storage.dtype}' なのに scale の宣言が無い")
        shape = _concrete_shape(graph, name, where)
        row_axis = axes.get(name, 0)
        if len(shape) <= row_axis:
            raise MigrateError(f"{where}: rowAxis {row_axis} に対して宣言 shape {shape} が浅い")
        if entry.grouping == "group":
            if storage.group_size is None:
                raise MigrateError(f"{where}: group 量子化なのに group_size の宣言が無い")
            if row_axis != 0:
                raise MigrateError(
                    f"{where}: 消費 op の per-channel 軸が {row_axis} だが、旧 group scale は"
                    "先頭次元を行として焼かれている（写せる形が無い）"
                )
            group_size = storage.group_size
        else:
            row_count = shape[row_axis]
            group_size = per_channel_group_size(math.prod(shape) // row_count if row_count else 0)
        bindings[key] = Encoding(
            codec, group_size=group_size, row_axis=row_axis, scale_key=storage.scale
        )
    return bindings


def _expected_scale_shape(shape: Sequence[int], encoding: Encoding) -> list[int]:
    """旧配布形が持っているはずの scale の形（`verify._assert_scale_tensor` の受理形）。"""
    assert encoding.group_size is not None and encoding.row_axis is not None
    if codec_entry(encoding.codec).grouping == "group":
        # group 形は rank2 ちょうど（行数 = 先頭次元・最終次元 = 行長 / group 長）。
        return [shape[0], math.prod(shape[1:]) // encoding.group_size]
    # per-channel は keepdim broadcast 形（`rowAxis` の軸だけが伸びる）。
    return [dim if axis == encoding.row_axis else 1 for axis, dim in enumerate(shape)]


def _assert_scale_layouts(
    graph: IrGraph, bindings: Mapping[str, Encoding], sources: Mapping[str, SourceTensor]
) -> None:
    """旧 scale の**形**が新しい宣言（`rowAxis` / `groupSize`）と一致することを見る。

    MUST: バイト数の一致だけでは足りない（書き手はそこまでしか見ない）— 正方の重みでは
    `[1, N]` の per-column scale が `rowAxis: 0` の per-channel として通り、チャネルの値が
    黙って入れ替わる。現物の**宣言 shape**まで突き合わせてから焼く。
    """
    for name, initializer in graph.initializers.items():
        if initializer.is_shared or initializer.tensor is None:
            continue
        encoding = bindings[initializer.tensor]
        if encoding.scale_key is None:
            continue
        where = f"initializer '{name}'"
        source = sources.get(encoding.scale_key)
        if source is None:
            raise MigrateError(f"{where}: scale テンソル '{encoding.scale_key}' がファイルに無い")
        shape = _concrete_shape(graph, name, where)
        expected = _expected_scale_shape(shape, encoding)
        if list(source.entry.shape) != expected:
            raise MigrateError(
                f"{where}: scale '{encoding.scale_key}' の形 {list(source.entry.shape)} が"
                f" codec '{encoding.codec}'・rowAxis {encoding.row_axis}・groupSize"
                f" {encoding.group_size} の形 {expected} と違う"
            )


def _assert_tensor_cover(
    bindings: Mapping[str, Encoding], sources: Mapping[str, SourceTensor]
) -> None:
    """旧コンテナのテンソルが 1 本残らず移ることを**全件列挙で**確かめる。

    MUST: 余剰も落とす（黙って置き去りにしない）— 束縛表は宣言から導くので、宣言に無い実体は
    新しいコンテナに居場所が無い。旧配布形のほうが誤っている可能性があり、どちらであれ
    「移したつもりでバイトが減った」を作らない。
    """
    referenced = set(bindings) | {
        encoding.scale_key for encoding in bindings.values() if encoding.scale_key is not None
    }
    missing = sorted(referenced - set(sources))
    surplus = sorted(set(sources) - referenced)
    if missing or surplus:
        raise MigrateError(
            "旧コンテナのテンソルと宣言が一致しない:"
            f" 不足 [{', '.join(missing)}] / 余剰 [{', '.join(surplus)}]"
        )


def _source_digest(source: SourceTensor) -> str:
    digest = hashlib.sha256()
    for chunk in payload_chunks(source):
        digest.update(chunk)
    return digest.hexdigest()


def _assert_payloads_match(
    read_back: ReadContainer,
    bound: BoundGraph,
    bindings: Mapping[str, Encoding],
    sources: Mapping[str, SourceTensor],
) -> int:
    """書いたコンテナの payload を旧実体と突き合わせる（§12 の不変条件 5）。

    取り直しは {@link karume.container.ReadContainer.block} 越しなので、block ごとの sha256 も
    同時に検証される。突き合わせるのは block 全体ではなく **payload 部**（末尾の詰め物は
    この移行で新たに焼かれるバイトで、旧配布形には無い）。
    """
    checked = 0
    for name, supply in sorted(bound.supplies.items()):
        digest = hashlib.sha256()
        for block in supply.blocks:
            digest.update(read_back.block(block.id)[: block.payload_bytes])
        _assert_digest(f"initializer '{name}'", digest.hexdigest(), sources[name])
        checked += 1
        scale_key = bindings[name].scale_key
        if supply.scale is not None and scale_key is not None:
            raw = read_back.block(supply.scale.id)[: supply.scale.payload_bytes]
            _assert_digest(
                f"initializer '{name}' の scale '{scale_key}'",
                hashlib.sha256(raw).hexdigest(),
                sources[scale_key],
            )
            checked += 1
    return checked


def _assert_digest(where: str, actual: str, source: SourceTensor) -> None:
    expected = _source_digest(source)
    if actual != expected:
        raise MigrateError(
            f"{where}: payload の sha256 が旧配布形と違う（旧 {expected} / 新 {actual}）"
        )


def _assert_assets_match(read_back: ReadContainer, assets: Mapping[str, AssetInput]) -> int:
    """書いた容器の資産を、渡した payload と突き合わせる（§12 の不変条件 5 を資産へ広げたもの）。

    突き合わせるのは **payload 部**で、末尾の詰め物が 0x00 であることも見る（詰め物は §4.1 の
    とおりこの移行で新たに焼かれるバイトなので、旧側には相手が無い）。
    """
    model = read_back.model
    if model is None:  # pragma: no cover - krm を読み直した直後なので在る
        raise MigrateError("krg には資産を載せられない")
    for name in sorted(assets):
        asset = assets[name]
        binding = model.assets.get(name)
        if binding is None:
            raise MigrateError(f"資産 '{name}' が書いた容器の宣言に無い")
        if binding.role != asset.role:
            raise MigrateError(
                f"資産 '{name}': 役割が '{binding.role}'（渡したのは '{asset.role}'）"
            )
        if binding.length != asset.length:
            raise MigrateError(
                f"資産 '{name}': 宣言の論理長が {binding.length}（渡したのは {asset.length}）"
            )
        raw = memoryview(read_back.block(binding.block))
        payload = asset.payload
        expected = memoryview(payload() if callable(payload) else payload).cast("B")
        if expected.nbytes != asset.length:
            raise MigrateError(
                f"資産 '{name}': 引き直した実体が {expected.nbytes} バイト"
                f"（宣言は {asset.length}）— 引かれるたびに同じバイト列を返す MUST"
            )
        if raw[: expected.nbytes] != expected:
            raise MigrateError(f"資産 '{name}': payload が渡したバイト列と違う")
        if bytes(raw[expected.nbytes :]) != b"\x00" * (raw.nbytes - expected.nbytes):
            raise MigrateError(f"資産 '{name}': 末尾の詰め物が 0x00 でない")
    return len(assets)


def _assert_no_leftovers(final: Path, graph_path: Path | None) -> None:
    """出力先に前回の成果物が残っていない（**消さずに止まる** — どれを配るかが決まらない）。"""
    existing = [str(path) for path in shard_siblings(final)]
    if graph_path is not None and graph_path.is_file():
        existing.append(str(graph_path))
    if existing:
        raise MigrateError(
            f"出力先に前回の成果物が残っている（消してからやり直す）: {', '.join(existing)}"
        )


def migrate_component(
    path: str | Path,
    out_dir: str | Path,
    *,
    provenance: Provenance,
    graph_name: str | None = None,
    single: bool = False,
    write_graph: bool = False,
    _part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> MigrationResult:
    """コンポーネント 1 つ（旧単一形 / 旧 shard 列）を `krm` へ移す。

    `path` は代表 path でも手元の現物（`...-00001-of-00002.safetensors`）でもよい
    （{@link karume.shards.component_path} が畳む）。`graph_name` の既定は**親ディレクトリ名**
    （配布形のコンポーネント名）で、コンテナの語彙（`[A-Za-z0-9._-]{1,64}`）から外れる場合は
    明示する。

    `_part_bytes` / `_block_bytes` は**テストからのみ触る**寸法の差し込み（合成の小さな資産で
    part またぎと piece 分割を踏むため）— 公開ノブではない。
    """
    source = component_path(Path(path))
    name = graph_name if graph_name is not None else source.parent.name
    if GRAPH_NAME_PATTERN.match(name) is None:
        raise MigrateError(
            f"グラフ名 '{name}' がコンテナの語彙（{GRAPH_NAME_PATTERN.pattern}）から外れる"
            " — `--graph-name` で明示する"
        )
    final = Path(out_dir) / f"{source.stem}{MODEL_SUFFIX}"
    return _migrate_shards(
        resolve_shards(source),
        final,
        provenance=provenance,
        graph_name=name,
        single=single,
        graph_path=final.with_suffix(GRAPH_SUFFIX) if write_graph else None,
        part_bytes=_part_bytes,
        block_bytes=_block_bytes,
    )


def _migrate_shards(
    shard_paths: Sequence[Path],
    final: Path,
    *,
    provenance: Provenance,
    graph_name: str,
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    graph_path: Path | None = None,
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> MigrationResult:
    """**明示の shard 列**を 1 本の `krm` へ移す（部品単位モードとリポ丸ごとモードの共通経路）。

    `shard_paths` は読む順（先頭がグラフ shard）。列を組むのは呼び手で、代表 path からの復元
    （{@link karume.shards.resolve_shards}）も旧 manifest の宣言もここへ来る前に済んでいる。
    """
    metadata, stored = read_component(shard_paths)
    graph = parse_ir_graph(metadata[IR_METADATA_KEY])
    bindings = container_bindings(graph)
    _assert_tensor_cover(bindings, stored)
    _assert_scale_layouts(graph, bindings, stored)

    _assert_no_leftovers(final, graph_path)
    staged = final.with_name(f"{final.stem}.{uuid4().hex}.partial{MODEL_SUFFIX}")
    replaced: list[Path] = []
    try:
        written = write_model_container(
            staged,
            graph,
            _SourcePayloads(stored),
            bindings,
            graph_name=graph_name,
            provenance=provenance,
            assets=assets,
            part_bytes=part_bytes,
            block_bytes=block_bytes,
            single=single,
        )
        read_back = read_container(written)
        bound = bind_graphs(read_back.graph, read_back.model)[graph_name]
        payloads = _assert_payloads_match(read_back, bound, bindings, stored)
        checked = _assert_assets_match(read_back, assets)
        documents = (
            DocumentRef(
                len(read_back.graph_descriptor_bytes),
                hashlib.sha256(read_back.graph_descriptor_bytes).hexdigest(),
            ),
            DocumentRef(
                len(read_back.model_descriptor_bytes),
                hashlib.sha256(read_back.model_descriptor_bytes).hexdigest(),
            ),
        )
        published = [final] if single else list(container_paths(final, len(written)))
        # MUST: `krg` は据え替えの**前**に抜く（読み手は一時 path の part を指している）。
        if graph_path is not None:
            graph_path.write_bytes(read_back.extract_graph())
        for staged_part, target in zip(written, published, strict=True):
            os.replace(staged_part, target)
            replaced.append(target)
    except BaseException:
        # 書き出しが途中で落ちた回は返り値が無いので、一時 path の**名前の形**から拾う。
        for leftover in shard_siblings(staged):
            leftover.unlink(missing_ok=True)
        # MUST: 据え替えの途中で落ちた回も**何も残さない**（半分だけ公開された容器を残すと、
        # 次の実行が「前回の成果物が残っている」で止まる — §12）。
        for target in replaced:
            target.unlink(missing_ok=True)
        # `krg` は抽出できた後に落ちた回だけ在る（前段の門が「先に在った」形を除いてある）。
        if graph_path is not None:
            graph_path.unlink(missing_ok=True)
        raise
    return MigrationResult(
        tuple(published), graph_path, len(bound.supplies), payloads, checked, documents
    )


# ---- 旧 manifest（`karume/4`）の読み取り -----------------------------------
#
# 旧形式を読むのはこの層だけ（§12）なので、hub の parse とは**別に**ここで全件検査する。
# 未知のキーは fail loudly — 黙って落とすと「移したつもりで席が 1 つ消えた」になる。


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MigrateError(message)


def _object(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} がオブジェクトでない")
    return value


def _keys(
    obj: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], where: str
) -> None:
    for key in required:
        _require(key in obj, f"{where}.{key} が無い")
    known = {*required, *optional}
    for key in obj:
        _require(key in known, f"{where}: 未知のキー '{key}'")


def _text(value: Any, where: str) -> str:
    _require(isinstance(value, str) and value != "", f"{where} が非空文字列でない")
    return value


def _count(value: Any, where: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{where} が非負整数でない: {value!r}",
    )
    return value


@dataclass(frozen=True)
class FileRef:
    """ADR 0038 §2 の 3 点セット（+ 越境の `repo` / `revision`）。"""

    path: str
    size: int
    sha256: str
    repo: str | None = None
    revision: str | None = None

    @property
    def cross(self) -> tuple[str, str] | None:
        """越境参照なら `(repo, revision)`・自リポ参照なら `None`。"""
        return None if self.repo is None else (self.repo, self.revision or "")

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {"path": self.path, "size": self.size, "sha256": self.sha256}
        if self.repo is not None:
            document["repo"] = self.repo
        if self.revision is not None:
            document["revision"] = self.revision
        return document


def _file_ref(value: Any, where: str) -> FileRef:
    obj = _object(value, where)
    _keys(obj, ["path", "size", "sha256"], ["repo", "revision"], where)
    ref = FileRef(
        _text(obj["path"], f"{where}.path"),
        _count(obj["size"], f"{where}.size"),
        _text(obj["sha256"], f"{where}.sha256"),
        None if "repo" not in obj else _text(obj["repo"], f"{where}.repo"),
        None if "revision" not in obj else _text(obj["revision"], f"{where}.revision"),
    )
    _require(
        (ref.repo is None) == (ref.revision is None),
        f"{where}: 越境参照は `repo` と `revision` の対で書く（片方だけは指し先が決まらない）",
    )
    return ref


@dataclass(frozen=True)
class WeightEntry:
    """1 (部品, dtype) ぶんのファイル群（`karume/4` の `{shards, extras?}`）。"""

    shards: tuple[FileRef, ...]
    extras: tuple[tuple[str, FileRef], ...]

    @property
    def refs(self) -> tuple[FileRef, ...]:
        return (*self.shards, *(ref for _, ref in self.extras))


@dataclass(frozen=True)
class LegacyModel:
    pipeline: Mapping[str, Any]
    weights: Mapping[str, Mapping[str, WeightEntry]]
    assets: Mapping[str, FileRef]
    quants: Mapping[str, Any]
    default_quant: str
    pipeline_config: Mapping[str, Any]


@dataclass(frozen=True)
class LegacyManifest:
    default_model: str
    models: Mapping[str, LegacyModel]


def _read_json(path: Path, where: str) -> Any:
    _require(path.is_file(), f"{where} が無い: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as cause:
        raise MigrateError(f"{where} が JSON として読めない: {path}") from cause


def read_legacy_manifest(path: Path) -> LegacyManifest:
    """旧 `karume/4` の `karume.json` を読む（major が違えば fail loudly — 両読みはしない）。"""
    where = MANIFEST_FILE
    root = _object(_read_json(path, where), where)
    # MUST: 版の判定を欄の検査より先に置く（別 major の manifest は「欄が違う」ではなく
    # 「その版は読まない」と言うのが直す側にとって決定的）。
    _require(
        root.get("format") == LEGACY_MANIFEST_FORMAT,
        f"{where}.format が {root.get('format')!r} — 移行できるのは"
        f" '{LEGACY_MANIFEST_FORMAT}' だけ（ADR 0109 決定 1）",
    )
    _keys(root, ["format", "generator", "defaultModel", "models"], [], where)
    models: dict[str, LegacyModel] = {}
    for name, raw in _object(root["models"], f"{where}.models").items():
        models[name] = _legacy_model(raw, f"{where}.models['{name}']")
    _require(len(models) > 0, f"{where}.models が空")
    default_model = _text(root["defaultModel"], f"{where}.defaultModel")
    _require(default_model in models, f"{where}.defaultModel '{default_model}' が models に無い")
    return LegacyManifest(default_model, models)


def _legacy_model(value: Any, where: str) -> LegacyModel:
    obj = _object(value, where)
    _keys(
        obj,
        ["pipeline", "weights", "assets", "quants", "defaultQuant", "pipelineConfig"],
        [],
        where,
    )
    pipeline = _object(obj["pipeline"], f"{where}.pipeline")
    _keys(pipeline, ["name", "major"], [], f"{where}.pipeline")
    _text(pipeline["name"], f"{where}.pipeline.name")
    weights: dict[str, dict[str, WeightEntry]] = {}
    for component, labels in _object(obj["weights"], f"{where}.weights").items():
        entries: dict[str, WeightEntry] = {}
        for dtype, raw in _object(labels, f"{where}.weights['{component}']").items():
            entries[dtype] = _weight_entry(raw, f"{where}.weights['{component}']['{dtype}']")
        _require(len(entries) > 0, f"{where}.weights['{component}'] が空")
        weights[component] = entries
    assets = {
        name: _file_ref(raw, f"{where}.assets['{name}']")
        for name, raw in _object(obj["assets"], f"{where}.assets").items()
    }
    quants = _object(obj["quants"], f"{where}.quants")
    default_quant = _text(obj["defaultQuant"], f"{where}.defaultQuant")
    _require(default_quant in quants, f"{where}.defaultQuant '{default_quant}' が quants に無い")
    return LegacyModel(
        pipeline,
        weights,
        assets,
        quants,
        default_quant,
        _object(obj["pipelineConfig"], f"{where}.pipelineConfig"),
    )


def _weight_entry(value: Any, where: str) -> WeightEntry:
    obj = _object(value, where)
    _keys(obj, ["shards"], ["extras"], where)
    raw_shards = obj["shards"]
    _require(isinstance(raw_shards, list) and raw_shards, f"{where}.shards が非空の配列でない")
    extras = _object(obj.get("extras", {}), f"{where}.extras")
    return WeightEntry(
        tuple(_file_ref(raw, f"{where}.shards[{i}]") for i, raw in enumerate(raw_shards)),
        tuple(
            (name, _file_ref(extras[name], f"{where}.extras['{name}']")) for name in sorted(extras)
        ),
    )


# ---- リポ丸ごとモード -------------------------------------------------------


@dataclass(frozen=True)
class CrossRepo:
    """`--cross-repo <owner/name>=<変換済みディレクトリ>@<40 桁 revision>` の 1 件。"""

    repo: str
    directory: Path
    revision: str


def parse_cross_repo(spec: str) -> CrossRepo:
    """`--cross-repo` の 1 語を読む（形が違えば fail loudly — 推測しない）。"""
    shape = "<owner/name>=<変換済みディレクトリ>@<40 桁 revision>"
    repo, sep, rest = spec.partition("=")
    _require(bool(sep) and bool(rest), f"--cross-repo '{spec}' が {shape} の形でない")
    directory, at, revision = rest.rpartition("@")
    _require(bool(at) and bool(directory), f"--cross-repo '{spec}' が {shape} の形でない")
    _require(
        REPO_RE.match(repo) is not None, f"--cross-repo '{spec}': repo が <owner>/<name> でない"
    )
    _require(
        REVISION_RE.match(revision) is not None,
        f"--cross-repo '{spec}': revision が 40 桁 hex 小文字でない（pin は完全な commit sha）",
    )
    return CrossRepo(repo, Path(directory), revision)


@dataclass(frozen=True)
class Container:
    """manifest `karume/5` の `container` 欄（ADR 0109 決定 3）。"""

    #: 2 文書（グラフ記述 / モデル記述）の期待値。
    descriptor: tuple[DocumentRef, DocumentRef]
    #: part 0 を含む全 part の FileRef（添字順）。
    parts: tuple[FileRef, ...]

    def to_document(self) -> dict[str, Any]:
        graph, model = self.descriptor
        return {
            "descriptor": {"graph": graph.to_document(), "model": model.to_document()},
            "parts": [ref.to_document() for ref in self.parts],
        }


@dataclass(frozen=True)
class _Seat:
    """manifest 上の 1 席（モデル, 部品, dtype ラベル）。"""

    model: str
    component: str
    dtype: str

    def __str__(self) -> str:
        return f"{self.model}.weights.{self.component}.{self.dtype}"


@dataclass(frozen=True)
class _Unit:
    """**1 度だけ**変換する単位。同じ内容を指す席（sbv2 の共有部品）はここで畳まれる。"""

    component: str
    shards: tuple[FileRef, ...]
    extras: tuple[tuple[str, FileRef], ...]
    #: PLE を畳む席だけが持つ（{@link _PleFold.refs} — 畳み方が違う席を同じ容器にしないための鍵）。
    ple: tuple[tuple[str, FileRef], ...]


@dataclass(frozen=True)
class _PleFold:
    """1 モデルぶんの PLE の畳み先（部品名・容器へ移す assets の並び・読んだ旧索引）。

    旧索引はここが**1 度だけ**読んだものを運ぶ（同じ事実を 2 度導くと、片方だけ検査が
    足されたときにずれる）。
    """

    component: str
    refs: tuple[tuple[str, FileRef], ...]
    index: Mapping[str, Any]

    @property
    def asset_names(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.refs)

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(ref.path for _, ref in self.refs)


@dataclass(frozen=True)
class ConvertedContainer:
    """変換した容器 1 本の報告（同じ容器を指す席が複数あっても 1 件）。"""

    rel_paths: tuple[str, ...]
    initializers: int
    payloads: int
    assets: int


@dataclass(frozen=True)
class RepositoryResult:
    """リポ丸ごとモードの結果。"""

    manifest: Path
    converted: tuple[ConvertedContainer, ...]
    #: 越境参照から引き写した席の数（変換していない）。
    crossed: int
    #: そのまま複写した通常ファイルの本数。
    copied: int


def migrate_repository(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    provenance: Provenance,
    cross_repos: Sequence[CrossRepo] = (),
    _part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> RepositoryResult:
    """旧 `karume/4` のリポを丸ごと `karume/5` + `krm` へ移す（ADR 0109 決定 8）。

    旧リポは**読むだけ**で、成果物は全部 `out_dir` の下に出る。`_part_bytes` / `_block_bytes`
    はテストからのみ触る寸法の差し込み（部品単位モードと同じ）。

    MUST: 書くのは**分割形だけ**（単一形の席が無い）— `karume/5` の `container.parts` は
    part 0 + part 1 の 2 要素以上 MUST で、単一形はその形を作れない（ADR 0109 決定 3）。
    """
    source = Path(manifest_path)
    repo = source.parent.resolve()
    out = Path(out_dir).resolve()
    _require(
        repo != out and repo not in out.parents and out not in repo.parents,
        f"出力先 {out} が入力リポ {repo} と入れ子になっている（旧入力は読むだけ MUST）",
    )
    legacy = read_legacy_manifest(source)
    crosses = {cross.repo: cross for cross in cross_repos}
    _require(
        len(crosses) == len(cross_repos),
        "--cross-repo に同じ repo が 2 度出ている（どちらを引くかが決まらない）",
    )
    seat_crosses = {seat: _entry_cross(entry, str(seat)) for seat, entry in _seats(legacy)}
    _assert_cross_repos_declared(seat_crosses, crosses)
    folds = {
        name: _plan_ple(repo, model, f"models['{name}']") for name, model in legacy.models.items()
    }

    containers: dict[_Seat, Container] = {}
    converted: dict[_Unit, Container] = {}
    reports: list[ConvertedContainer] = []
    folded: set[str] = set()
    crossed = 0
    for seat, entry in _seats(legacy):
        fold = folds[seat.model]
        cross = seat_crosses[seat]
        if cross is not None:
            containers[seat] = _crossed_container(crosses[cross[0]], entry, seat)
            crossed += 1
            continue
        folding = fold if fold is not None and fold.component == seat.component else None
        if folding is not None:
            folded.add(seat.model)
        unit = _Unit(
            seat.component, entry.shards, entry.extras, () if folding is None else folding.refs
        )
        found = converted.get(unit)
        if found is None:
            found, report = _convert_unit(
                unit,
                repo,
                out,
                str(seat),
                provenance=provenance,
                fold=folding,
                part_bytes=_part_bytes,
                block_bytes=_block_bytes,
            )
            converted[unit] = found
            reports.append(report)
        containers[seat] = found
    for name, fold in folds.items():
        # MUST: 畳み先が 1 つも無いまま assets から消すと PLE が黙って配布物から落ちる
        # （持ち主の部品が丸ごと越境参照のときに起きる）。
        _require(
            fold is None or name in folded,
            f"models['{name}']: PLE の持ち主の部品 '{fold.component if fold else ''}' に"
            "自リポで変換する席が 1 つも無い（畳み先が決まらない）",
        )

    skip = {MANIFEST_FILE}
    for unit in converted:
        skip.update(ref.path for ref in unit.shards)
        # MUST: 容器の資産になった extras も複写しない（宣言外の実体を配布物に残さない）。
        skip.update(ref.path for _, ref in unit.extras)
    for fold in folds.values():
        if fold is not None:
            skip.update(fold.paths)
    copied = _copy_plain_files(repo, out, skip)

    manifest = _manifest_document(legacy, containers, folds)
    assert_manifest_limits(manifest)
    out.mkdir(parents=True, exist_ok=True)
    written = out / MANIFEST_FILE
    _require(not written.exists(), f"出力先に {MANIFEST_FILE} が既に在る（消してからやり直す）")
    written.write_text(manifest_text(manifest), encoding="utf-8")
    return RepositoryResult(written, tuple(reports), crossed, copied)


def _seats(legacy: LegacyManifest) -> Iterator[tuple[_Seat, WeightEntry]]:
    """manifest 上の全席を宣言順に（席の走査はこの 1 本道だけを通る）。"""
    for model_name, model in legacy.models.items():
        for component, labels in model.weights.items():
            for dtype, entry in labels.items():
                yield _Seat(model_name, component, dtype), entry


def _entry_cross(entry: WeightEntry, where: str) -> tuple[str, str] | None:
    """1 席の越境参照（容器単位 MUST — 混在は fail loudly・ADR 0109 決定 3）。"""
    crosses = {ref.cross for ref in entry.refs}
    if len(crosses) != 1:
        shown = ", ".join(sorted("自リポ" if c is None else f"{c[0]}@{c[1]}" for c in crosses))
        raise MigrateError(f"{where}: 越境参照が混在している（容器単位 MUST）: {shown}")
    return crosses.pop()


def _assert_cross_repos_declared(
    seat_crosses: Mapping[_Seat, tuple[str, str] | None], crosses: Mapping[str, CrossRepo]
) -> None:
    """`--cross-repo` が足りない回は、**参照している repo と本数を全部列挙して**止まる。"""
    missing: Counter[str] = Counter()
    for cross in seat_crosses.values():
        if cross is not None and cross[0] not in crosses:
            missing[cross[0]] += 1
    if missing:
        shown = " / ".join(f"{repo}（{count} 席）" for repo, count in sorted(missing.items()))
        raise MigrateError(
            f"越境参照の repo が --cross-repo で渡されていない: {shown}"
            " — 先にその repo を同じモードで変換し、"
            "`--cross-repo <owner/name>=<出力ディレクトリ>@<40 桁 revision>` で渡す"
        )


def _crossed_container(cross: CrossRepo, entry: WeightEntry, seat: _Seat) -> Container:
    """変換済みディレクトリの `karume/5` から、同じ**容器**の `container` を引き写す。

    探す鍵は席の名前（モデル, 部品, dtype）ではなく、**旧 shard 列から導いた part 0 の置き場**
    である — 出力 path の規則は両リポで同じ（{@link _output_location}）一方、席の名前が貸し手と
    借り手で揃っている保証は無い（借り手が別名のモデルから借りる形が実在する）。
    """
    where = f"{cross.repo} の {MANIFEST_FILE}"
    root = _object(_read_json(cross.directory / MANIFEST_FILE, where), where)
    _require(
        root.get("format") == MANIFEST_FORMAT,
        f"{where}.format が '{root.get('format')}' — 引き写せるのは '{MANIFEST_FORMAT}' だけ",
    )
    directory, stem = _output_location(entry.shards, str(seat))
    wanted = f"{directory}/{stem}{MODEL_SUFFIX}"
    found: tuple[str, Mapping[str, Any]] | None = None
    candidates: set[str] = set()
    for label, raw in _crossed_entries(root, where):
        container = _object(raw, label)
        head = container.get("parts")
        _require(isinstance(head, list) and head, f"{label}.parts が非空の配列でない")
        key = component_path(Path(_file_ref(head[0], f"{label}.parts[0]").path)).as_posix()
        candidates.add(key)
        if key == wanted and found is None:
            found = (label, container)
    if found is None:
        raise MigrateError(
            f"{where}: 引き写す容器 '{wanted}' が無い"
            "（越境参照の引き写しは旧 shard 列から導いた part 0 の置き場で探す）"
            f" — 参照先が持つのは {', '.join(sorted(candidates)) if candidates else '（無し）'}"
        )
    return _crossed_parse(found[1], found[0], cross)


def _crossed_entries(root: Mapping[str, Any], where: str) -> Iterator[tuple[str, Any]]:
    """参照先 `karume/5` の全 (モデル, 部品, dtype) の `container` 欄（位置の綴りつき）。"""
    for model_name, raw_model in _object(root.get("models"), f"{where}.models").items():
        model_where = f"{where} の {model_name}"
        weights = _object(_object(raw_model, model_where).get("weights"), f"{model_where}.weights")
        for component, labels in weights.items():
            for dtype, raw_entry in _object(labels, f"{model_where}.weights.{component}").items():
                label = f"{model_where}.weights.{component}.{dtype}"
                obj = _object(raw_entry, label)
                _keys(obj, ["container"], [], label)
                yield f"{label}.container", obj["container"]


def _crossed_parse(container: Mapping[str, Any], where: str, cross: CrossRepo) -> Container:
    """引き写す `container` 欄 1 件を読む（欄の欠け・未知のキーは fail loudly）。"""
    _keys(container, ["descriptor", "parts"], [], where)
    descriptor = _object(container["descriptor"], f"{where}.descriptor")
    _keys(descriptor, ["graph", "model"], [], f"{where}.descriptor")
    documents = []
    for key in ("graph", "model"):
        document = _object(descriptor[key], f"{where}.descriptor.{key}")
        _keys(document, ["length", "sha256"], [], f"{where}.descriptor.{key}")
        documents.append(
            DocumentRef(
                _count(document["length"], f"{where}.descriptor.{key}.length"),
                _text(document["sha256"], f"{where}.descriptor.{key}.sha256"),
            )
        )
    parts: list[FileRef] = []
    for index, raw in enumerate(container["parts"]):
        ref = _file_ref(raw, f"{where}.parts[{index}]")
        _require(
            ref.cross is None,
            f"{where}.parts[{index}] が既に越境参照を持つ（2 段の引き写しは指し先が辿れない）",
        )
        parts.append(replace(ref, repo=cross.repo, revision=cross.revision))
    return Container((documents[0], documents[1]), tuple(parts))


def _convert_unit(
    unit: _Unit,
    repo: Path,
    out: Path,
    where: str,
    *,
    provenance: Provenance,
    fold: _PleFold | None,
    part_bytes: int,
    block_bytes: int,
) -> tuple[Container, ConvertedContainer]:
    """1 単位（部品 × dtype の shard 列）を `krm` へ変換し、manifest の `container` を組む。"""
    if GRAPH_NAME_PATTERN.match(unit.component) is None:
        raise MigrateError(
            f"{where}: 部品名 '{unit.component}' がコンテナのグラフ名の語彙"
            f"（{GRAPH_NAME_PATTERN.pattern}）から外れる"
        )
    directory, stem = _output_location(unit.shards, where)
    # 資産の並びがそのまま物理配置の順になる（PLE は token 順 — §4.2 の走査型取得元のため）。
    assets = _ple_assets(repo, fold, block_bytes, where)
    assets.update(_extra_assets(repo, unit.extras, where))
    result = _migrate_shards(
        [repo / ref.path for ref in unit.shards],
        out / directory / f"{stem}{MODEL_SUFFIX}",
        provenance=provenance,
        graph_name=unit.component,
        assets=assets,
        part_bytes=part_bytes,
        block_bytes=block_bytes,
    )
    parts = tuple(_published_ref(out, path) for path in result.parts)
    return (
        Container(result.descriptor, parts),
        ConvertedContainer(
            tuple(ref.path for ref in parts), result.initializers, result.payloads, result.assets
        ),
    )


def _output_location(shards: Sequence[FileRef], where: str) -> tuple[str, str]:
    """`(出力ディレクトリの相対 path, stem)` — 置き場は**重み shard の親**（推測しない）。"""
    stems = set()
    for ref in shards:
        name = PurePosixPath(ref.path)
        _require(
            name.suffix == ".safetensors",
            f"{where}: 旧 shard '{ref.path}' の拡張子が .safetensors でない",
        )
        stems.add(component_path(Path(name.name)).stem)
    _require(
        len(stems) == 1, f"{where}: shard 列のファイル名の stem が揃っていない: {sorted(stems)}"
    )
    # 列の 2 本目以降が重み shard。グラフ shard だけの列は先頭の親に置く。
    weights = shards[1:] or shards[:1]
    directories = {PurePosixPath(ref.path).parent.as_posix() for ref in weights}
    _require(
        len(directories) == 1,
        f"{where}: 重み shard の親ディレクトリが揃っていない（{sorted(directories)}）"
        " — 置き場を推測しない",
    )
    return directories.pop(), stems.pop()


def _published_ref(out: Path, path: Path) -> FileRef:
    """据えた part 1 本の FileRef（長さ 0 の part も `size: 0` で載せる — ADR 0109 決定 3）。"""
    return FileRef(**file_ref(out, path.relative_to(out).as_posix(), sha256_file(path)))


def _extra_assets(
    repo: Path, extras: Sequence[tuple[str, FileRef]], where: str
) -> dict[str, AssetInput]:
    """旧 `extras` を容器の資産へ（実在は `rope_base` 1 種 — 知らない名前は fail loudly）。"""
    assets: dict[str, AssetInput] = {}
    for name, ref in extras:
        mapped = EXTRA_ASSETS.get(name)
        if mapped is None:
            raise MigrateError(
                f"{where}: extras '{name}' の写し先が無い"
                f"（容器の資産へ移せるのは {' / '.join(sorted(EXTRA_ASSETS))} だけ）"
            )
        _require(ref.cross is None, f"{where}: extras '{name}' が越境参照（容器へ畳めない）")
        asset_name, role = mapped
        # 論理長は旧宣言の `size`（現物とずれていれば書き手が落とす — 推測しない）。
        assets[asset_name] = AssetInput(role, ref.size, _file_reader(repo / ref.path))
    return assets


def _file_reader(path: Path) -> Callable[[], bytes]:
    """ファイル 1 本を丸ごと読む遅延の呼び出し（引かれるまで開かない）。"""

    def read() -> bytes:
        return path.read_bytes()

    return read


def _copy_plain_files(repo: Path, out: Path, skip: frozenset[str] | set[str]) -> int:
    """変換しなかった通常ファイルをそのまま複写する（`.` で始まるディレクトリは入らない）。"""
    copied = 0
    for source in sorted(repo.rglob("*")):
        rel = source.relative_to(repo)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        if not source.is_file() or rel.as_posix() in skip:
            continue
        target = out / rel
        _require(
            not target.exists(),
            f"出力先に {rel.as_posix()} が既に在る（消してからやり直す）",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    return copied


def _manifest_document(
    legacy: LegacyManifest,
    containers: Mapping[_Seat, Container],
    folds: Mapping[str, _PleFold | None],
) -> dict[str, Any]:
    """`karume/5` の `karume.json`（旧値はそのまま・weights だけ `container` に置き換わる）。"""
    models: dict[str, Any] = {}
    for name, model in legacy.models.items():
        fold = folds[name]
        dropped = frozenset() if fold is None else fold.asset_names
        models[name] = {
            "pipeline": dict(model.pipeline),
            "weights": {
                component: {
                    dtype: {"container": containers[_Seat(name, component, dtype)].to_document()}
                    for dtype in labels
                }
                for component, labels in model.weights.items()
            },
            "assets": {
                asset: ref.to_document()
                for asset, ref in model.assets.items()
                if asset not in dropped
            },
            "quants": dict(model.quants),
            "defaultQuant": model.default_quant,
            "pipelineConfig": dict(model.pipeline_config),
        }
    return {
        "format": MANIFEST_FORMAT,
        "generator": generator_tag(),
        "defaultModel": legacy.default_model,
        "models": models,
    }


# ---- PLE sidecar → 容器の資産（索引 schema 3）-------------------------------


def _plan_ple(repo: Path, model: LegacyModel, where: str) -> _PleFold | None:
    """PLE の畳み先を決める（索引が無ければ `None`）。

    MUST: `ple_index` が在るのに持ち主の表に無い pipeline なら fail loudly — どの部品の容器へ
    畳むかが決まらないまま「assets から消えた」形を作らない。
    """
    index_ref = model.assets.get(PLE_INDEX_ASSET)
    if index_ref is None:
        return None
    pipeline = str(model.pipeline["name"])
    component = PLE_OWNER.get(pipeline)
    if component is None:
        raise MigrateError(
            f"{where}: assets に '{PLE_INDEX_ASSET}' が在るが pipeline '{pipeline}' は PLE の"
            f"持ち主でない（表: {', '.join(f'{k} → {v}' for k, v in sorted(PLE_OWNER.items()))}）"
        )
    _require(
        component in model.weights,
        f"{where}: PLE の持ち主の部品 '{component}' が weights に無い",
    )
    _require(index_ref.cross is None, f"{where}: '{PLE_INDEX_ASSET}' が越境参照（容器へ畳めない）")
    index = read_ple_index(repo / index_ref.path, f"{where} の {PLE_INDEX_ASSET}")
    refs: list[tuple[str, FileRef]] = [(PLE_INDEX_ASSET, index_ref)]
    for entry in index["shards"]:
        name = str(entry["file"])
        ref = model.assets.get(name)
        if ref is None:
            raise MigrateError(f"{where}: 索引が指す PLE shard '{name}' が assets に無い")
        _require(ref.cross is None, f"{where}: PLE shard '{name}' が越境参照（容器へ畳めない）")
        refs.append((name, ref))
    return _PleFold(component, tuple(refs), index)


def read_ple_index(path: Path, where: str) -> dict[str, Any]:
    """旧 PLE 索引（`ple.json`）を読む。

    受理集合は `packages/models/src/gemma/ple-index.ts` の鏡像で、容器へ畳めるのは packed 格納の
    schema 2 だけ（schema 1 の i8 形は実資産に無い）。
    """
    root = _object(_read_json(path, where), where)
    # MUST: 版の判定を欄の検査より先に置く（{@link read_legacy_manifest} と同じ理由 — 別の版は
    # 「欄が欠けている」ではなく「その版は畳めない」と言うのが直す側にとって決定的）。
    _require(
        root.get("schema") == 2,
        f"{where}.schema が {root.get('schema')!r} — 容器へ畳めるのは packed 格納の schema 2 だけ",
    )
    _keys(root, ["schema", "storage", "tokens", "layers", "dim", "embedScale", "shards"], [], where)
    _require(
        root["storage"] in PLE_PACK_FACTOR,
        f"{where}.storage が {root['storage']!r}"
        f"（{' / '.join(sorted(PLE_PACK_FACTOR))} のいずれか）",
    )
    for key in ("tokens", "layers", "dim"):
        _require(_count(root[key], f"{where}.{key}") > 0, f"{where}.{key} が 1 以上でない")
    scale = root["embedScale"]
    _require(
        isinstance(scale, int | float) and not isinstance(scale, bool) and scale > 0,
        f"{where}.embedScale が正の数でない: {scale!r}",
    )
    shards = root["shards"]
    _require(isinstance(shards, list) and shards, f"{where}.shards が非空の配列でない")
    cursor = 0
    for index, raw in enumerate(shards):
        entry = _object(raw, f"{where}.shards[{index}]")
        _keys(entry, ["file", "start", "stop"], [], f"{where}.shards[{index}]")
        _text(entry["file"], f"{where}.shards[{index}].file")
        start = _count(entry["start"], f"{where}.shards[{index}].start")
        stop = _count(entry["stop"], f"{where}.shards[{index}].stop")
        _require(
            start == cursor and stop > start,
            f"{where}.shards[{index}]: 範囲 [{start}, {stop}) が"
            f" {cursor} から続く空でない区間でない",
        )
        cursor = stop
    _require(cursor == root["tokens"], f"{where}: shard の合計 {cursor} 行が tokens と違う")
    return root


@dataclass(frozen=True)
class _StoredEntry:
    """旧 sidecar のテンソル 1 本（ヘッダだけから決まる事実）。"""

    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def _read_sidecar_header(path: Path) -> tuple[dict[str, str], dict[str, _StoredEntry], int]:
    """旧 sidecar の safetensors を**ヘッダだけ**読む（PLE shard は数百 MB になる）。"""
    header = safetensors_header(path)
    with path.open("rb") as handle:
        data_start = 8 + int.from_bytes(handle.read(8), "little")
    raw = header.get("__metadata__", {})
    _require(isinstance(raw, dict), f"{path}: __metadata__ がマップでない")
    entries: dict[str, _StoredEntry] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        entries[name] = _StoredEntry(spec["dtype"], tuple(spec["shape"]), begin, end)
    return {str(k): str(v) for k, v in raw.items()}, entries, data_start


def _read_segments(segments: Sequence[tuple[Path, int, int]], begin: int, end: int) -> bytes:
    """連結された区間列の `[begin, end)` を読む（PLE の行は shard 境界を跨ぐ）。"""
    out = bytearray()
    cursor = 0
    for path, offset, length in segments:
        stop = cursor + length
        if stop > begin and cursor < end:
            lower, upper = max(begin, cursor), min(end, stop)
            with path.open("rb") as handle:
                handle.seek(offset + lower - cursor)
                chunk = handle.read(upper - lower)
            if len(chunk) != upper - lower:
                raise MigrateError(f"{path}: PLE の区間 [{lower}, {upper}) が途中で尽きた")
            out += chunk
        cursor = stop
    if len(out) != end - begin:
        raise MigrateError(f"PLE の区間 [{begin}, {end}) が shard 列の長さ {cursor} に収まらない")
    return bytes(out)


def _segment_reader(
    segments: tuple[tuple[Path, int, int], ...], begin: int, end: int
) -> Callable[[], bytes]:
    def read() -> bytes:
        return _read_segments(segments, begin, end)

    return read


def _ple_assets(
    repo: Path, fold: _PleFold | None, block_bytes: int, where: str
) -> dict[str, AssetInput]:
    """PLE sidecar を容器の資産へ（索引 schema 3 + `values` / `scales` の block 列）。

    旧 shard の境界は意味を持たない（全 shard を token 順に連結して切り直す）。block は**行の
    倍数**で ≤ `block_bytes` に切るので、行バイト数が 4 の倍数であるかぎり詰め物は要らない。

    返す並びが**そのまま物理配置の順**（token 順 — 区間読みの block は 1 block = 1 part）。
    """
    if fold is None:
        return {}
    index = fold.index
    shard_refs = fold.refs[1:]  # 先頭は索引そのもの（畳み先では新しい索引に置き換わる）。
    tokens, layers, dim = index["tokens"], index["layers"], index["dim"]
    factor = PLE_PACK_FACTOR[index["storage"]]
    _require(
        dim % factor == 0,
        f"{where}: dim {dim} が格納 '{index['storage']}' の詰め数 {factor} で割り切れない",
    )
    row_bytes = {"values": layers * dim // factor, "scales": layers * PLE_SCALE_BYTES}
    segments: dict[str, list[tuple[Path, int, int]]] = {"values": [], "scales": []}
    for position, (name, ref) in enumerate(shard_refs):
        declared = index["shards"][position]
        _require(
            name == declared["file"],
            f"{where}: PLE shard の並びが索引と違う（{position} 番目は '{declared['file']}'）",
        )
        path = repo / ref.path
        rows = declared["stop"] - declared["start"]
        metadata, entries, data_start = _read_sidecar_header(path)
        _assert_ple_metadata(metadata, index, declared, f"{where} の '{name}'")
        _require(
            set(entries) == set(row_bytes),
            f"{where} の '{name}': テンソルが {sorted(entries)}"
            f"（{' / '.join(sorted(row_bytes))} の 2 本 MUST）",
        )
        for key, expected in row_bytes.items():
            entry = entries[key]
            _require(
                entry.shape[:1] == (rows,),
                f"{where} の '{name}'.{key}: 先頭次元 {entry.shape[:1]} が索引の {rows} 行と違う",
            )
            _require(
                entry.nbytes == rows * expected,
                f"{where} の '{name}'.{key}: {entry.nbytes} バイトが"
                f" {rows} 行 × {expected} バイトと違う",
            )
            _require(
                expected % 4 == 0,
                f"{where}: {key} の 1 行 {expected} バイトが 4 の倍数でない"
                "（行の倍数で block に切れない）",
            )
            segments[key].append((path, data_start + entry.begin, entry.nbytes))

    assets: dict[str, AssetInput] = {}
    document: dict[str, Any] = {
        "schema": PLE_INDEX_SCHEMA,
        "storage": index["storage"],
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": index["embedScale"],
    }
    for key, role in PLE_ROLES.items():
        stride = row_bytes[key]
        per_block = block_bytes // stride
        _require(
            per_block >= 1,
            f"{where}: {key} の 1 行 {stride} バイトが block 上限 {block_bytes} を超える",
        )
        placed = tuple(segments[key])
        blocks: list[dict[str, Any]] = []
        for start in range(0, tokens, per_block):
            stop = min(start + per_block, tokens)
            name = f"ple.{key}.{len(blocks)}"
            blocks.append({"asset": name, "start": start, "stop": stop})
            # 区間読みを要する block なので専用 part に単独で置く（container-v1 §4.2）。
            assets[name] = AssetInput(
                role,
                (stop - start) * stride,
                _segment_reader(placed, start * stride, stop * stride),
                dedicated_part=True,
            )
        document[key] = {"rowBytes": stride, "blocks": blocks}
    encoded = canonical_json(document).encode("utf-8")
    assets[PLE_INDEX_ASSET] = AssetInput(PLE_INDEX_ROLE, len(encoded), encoded)
    return assets


def _assert_ple_metadata(
    metadata: Mapping[str, str],
    index: Mapping[str, Any],
    declared: Mapping[str, Any],
    where: str,
) -> None:
    """shard の `karume_ple` が索引と食い違っていないこと（食い違いは沈黙誤値の種）。"""
    raw = metadata.get(PLE_METADATA_KEY)
    if raw is None:
        raise MigrateError(f"{where}: __metadata__.{PLE_METADATA_KEY} が無い")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as cause:
        raise MigrateError(f"{where}: {PLE_METADATA_KEY} が JSON として読めない") from cause
    found = _object(parsed, f"{where}.{PLE_METADATA_KEY}")
    expected = {key: index[key] for key in ("schema", "storage", "tokens", "layers", "dim")}
    expected.update({key: declared[key] for key in ("start", "stop")})
    for key, value in expected.items():
        _require(
            found.get(key) == value,
            f"{where}.{PLE_METADATA_KEY}.{key} が {found.get(key)!r} — 索引は {value!r}",
        )


# ---- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="旧配布形（safetensors）をコンテナ形式（krm / krg）へ移す（入力は読むだけ）"
    )
    parser.add_argument(
        "models",
        type=Path,
        nargs="*",
        help="移行するコンポーネントの代表 path（`...-00001-of-00002` の現物を渡してもよい）",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="リポ丸ごとモード: 旧 karume/4 の karume.json（位置引数とは併用できない）",
    )
    parser.add_argument(
        "--cross-repo",
        action="append",
        default=[],
        metavar="OWNER/NAME=DIR@REVISION",
        help="越境参照の引き写し元（同じモードで先に変換したディレクトリと新しい 40 桁 revision）",
    )
    parser.add_argument("--out", type=Path, required=True, help="書き出し先のディレクトリ")
    parser.add_argument(
        "--license", required=True, help="provenance.license（ライセンス識別子 — 本文は載せない）"
    )
    parser.add_argument("--notice", default=None, help="provenance.notice（NOTICE への参照）")
    parser.add_argument(
        "--upstream-revision", default=None, help="provenance.upstreamRevision（上流の revision）"
    )
    parser.add_argument(
        "--writer", default=None, help=f"provenance.writer（既定: {generator_tag()}）"
    )
    parser.add_argument(
        "--graph-name",
        default=None,
        help="グラフ名（既定は親ディレクトリ名 — 複数指定のときは渡せない）",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="単一形の krm を書く（既定は分割形・部品単位モードだけ）",
    )
    parser.add_argument("--graph", action="store_true", help="krg（グラフ容器）も書き出す")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """指定されたコンポーネント（またはリポ丸ごと）を移行する。

    MUST: 落ちたところで止める（残りを移して最後にまとめない）— 例外は破れた不変条件まで
    綴ってあるので、そのまま送出するのが最も情報量が多い。
    """
    args = build_parser().parse_args(argv)
    provenance = Provenance(
        license=args.license,
        notice=args.notice,
        upstream_revision=args.upstream_revision,
        writer=args.writer if args.writer is not None else generator_tag(),
    )
    if args.manifest is not None:
        _run_repository(args, provenance)
        return
    if not args.models:
        raise MigrateError("移行するコンポーネントの代表 path か --manifest のどちらかが要る")
    if args.cross_repo:
        raise MigrateError("--cross-repo はリポ丸ごとモード（--manifest）でだけ使える")
    if args.graph_name is not None and len(args.models) > 1:
        raise MigrateError(
            f"--graph-name は 1 コンポーネントにだけ渡せる（指定は {len(args.models)} 本）"
        )
    for path in args.models:
        result = migrate_component(
            path,
            args.out,
            provenance=provenance,
            graph_name=args.graph_name,
            single=args.single,
            write_graph=args.graph,
        )
        graph = f" + {result.graph.name}" if result.graph is not None else ""
        print(
            f"{path}: parts {len(result.parts)} ({result.parts[0].name} …){graph}"
            f" initializers={result.initializers} payloads={result.payloads}"
        )


def _run_repository(args: argparse.Namespace, provenance: Provenance) -> None:
    """リポ丸ごとモードの入口（併用できない指定はここで全部落とす）。"""
    if args.models:
        raise MigrateError(
            f"--manifest と位置引数は併用できない（位置引数 {len(args.models)} 本）"
            " — リポ丸ごとモードは shard 列を manifest の宣言から組む"
        )
    for name, value in (("--graph-name", args.graph_name), ("--graph", args.graph or None)):
        if value is not None:
            raise MigrateError(
                f"{name} は部品単位モードの席（リポ丸ごとモードのグラフ名は manifest の"
                " weights のキー・共有 krg は karume/5 に置かない — ADR 0109 決定 5）"
            )
    if args.single:
        raise MigrateError(
            "--single はリポ丸ごとモードでは使えない（karume/5 の container.parts は"
            " part 0 + part 1 の 2 要素以上 MUST・HF の公式配布は分割形だけ"
            " — ADR 0109 決定 3 / container-v1 §8）"
        )
    result = migrate_repository(
        args.manifest,
        args.out,
        provenance=provenance,
        cross_repos=[parse_cross_repo(spec) for spec in args.cross_repo],
    )
    for converted in result.converted:
        print(
            f"{converted.rel_paths[0]}: parts {len(converted.rel_paths)}"
            f" initializers={converted.initializers} payloads={converted.payloads}"
            f" assets={converted.assets}"
        )
    print(
        f"{result.manifest}: containers={len(result.converted)}"
        f" crossed={result.crossed} copied={result.copied}"
    )


if __name__ == "__main__":
    main()
