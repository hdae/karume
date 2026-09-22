"""旧配布形（safetensors の shard 列）→ コンテナ形式（`krm` / `krg`）へ移す（container-v1 §12）。

    karume migrate ../../models/karume-depth-anything-v2/small/depth/model.f32.safetensors \
        --out /tmp/migrated --license apache-2.0

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
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from karume.container import (
    BLOCK_MAX_BYTES,
    CODEC_FOR_STORAGE,
    DEFAULT_PART_BYTES,
    GRAPH_NAME_PATTERN,
    Encoding,
    Provenance,
    ReadContainer,
    codec_entry,
    container_paths,
    per_channel_group_size,
    read_container,
    write_model_container,
)
from karume.dist import generator_tag
from karume.emit import EmitError, weight_channel_axes
from karume.ir import IR_METADATA_KEY, IrGraph
from karume.repack import SourceTensor, payload_chunks, read_component
from karume.shards import component_path, resolve_shards, shard_siblings
from karume.verify import BoundGraph, bind_graphs, parse_ir_graph

#: 新コンテナの拡張子（§1 — 種別は magic が持つが、ファイル名も分けておく）。
MODEL_SUFFIX = ".krm"
GRAPH_SUFFIX = ".krg"


class MigrateError(ValueError):
    """移行の前提が破れた（写し先の無い格納・宣言と現物の食い違い・出力先の残骸）。"""


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
    sources = resolve_shards(source)
    metadata, stored = read_component(sources)
    graph = parse_ir_graph(metadata[IR_METADATA_KEY])
    name = graph_name if graph_name is not None else source.parent.name
    if GRAPH_NAME_PATTERN.match(name) is None:
        raise MigrateError(
            f"グラフ名 '{name}' がコンテナの語彙（{GRAPH_NAME_PATTERN.pattern}）から外れる"
            " — `--graph-name` で明示する"
        )
    bindings = container_bindings(graph)
    _assert_tensor_cover(bindings, stored)
    _assert_scale_layouts(graph, bindings, stored)

    final = Path(out_dir) / f"{source.stem}{MODEL_SUFFIX}"
    graph_path = final.with_suffix(GRAPH_SUFFIX) if write_graph else None
    _assert_no_leftovers(final, graph_path)
    staged = final.with_name(f"{final.stem}.{uuid4().hex}.partial{MODEL_SUFFIX}")
    try:
        written = write_model_container(
            staged,
            graph,
            _SourcePayloads(stored),
            bindings,
            graph_name=name,
            provenance=provenance,
            part_bytes=_part_bytes,
            block_bytes=_block_bytes,
            single=single,
        )
        read_back = read_container(written)
        bound = bind_graphs(read_back.graph, read_back.model)[name]
        payloads = _assert_payloads_match(read_back, bound, bindings, stored)
        published = [final] if single else list(container_paths(final, len(written)))
        # MUST: `krg` は据え替えの**前**に抜く（読み手は一時 path の part を指している）。
        if graph_path is not None:
            graph_path.write_bytes(read_back.extract_graph())
        for staged_part, target in zip(written, published, strict=True):
            os.replace(staged_part, target)
    except BaseException:
        # 書き出しが途中で落ちた回は返り値が無いので、一時 path の**名前の形**から拾う。
        for leftover in shard_siblings(staged):
            leftover.unlink(missing_ok=True)
        # `krg` は抽出できた後に落ちた回だけ在る（前段の門が「先に在った」形を除いてある）。
        if graph_path is not None:
            graph_path.unlink(missing_ok=True)
        raise
    return MigrationResult(tuple(published), graph_path, len(bound.supplies), payloads)


# ---- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="旧配布形（safetensors）をコンテナ形式（krm / krg）へ移す（入力は読むだけ）"
    )
    parser.add_argument(
        "models",
        type=Path,
        nargs="+",
        help="移行するコンポーネントの代表 path（`...-00001-of-00002` の現物を渡してもよい）",
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
    parser.add_argument("--single", action="store_true", help="単一形の krm を書く（既定は分割形）")
    parser.add_argument("--graph", action="store_true", help="krg（グラフ容器）も書き出す")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """指定されたコンポーネントを 1 つずつ移行する。

    MUST: 落ちたところで止める（残りを移して最後にまとめない）— 例外は破れた不変条件まで
    綴ってあるので、そのまま送出するのが最も情報量が多い。
    """
    args = build_parser().parse_args(argv)
    if args.graph_name is not None and len(args.models) > 1:
        raise MigrateError(
            f"--graph-name は 1 コンポーネントにだけ渡せる（指定は {len(args.models)} 本）"
        )
    provenance = Provenance(
        license=args.license,
        notice=args.notice,
        upstream_revision=args.upstream_revision,
        writer=args.writer if args.writer is not None else generator_tag(),
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


if __name__ == "__main__":
    main()
