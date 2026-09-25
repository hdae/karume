"""旧配布形（safetensors の shard 列）→ コンテナ形式（`krm` / `krg`）へ移す（container-v1 §12）。

    # 部品 1 つ（段 1 の面）
    karume migrate ../../models/karume-depth-anything-v2/small/depth/model.f32.safetensors \
        --out /tmp/migrated --license apache-2.0
    # リポ丸ごと（段 2 の面 — `karume/5` の karume.json まで書く・ADR 0109 決定 8）
    karume migrate --manifest ../../models/karume-irodori/karume.json \
        --out /tmp/karume-irodori --license apache-2.0

**旧形式を読む処理はこの層にだけ置く**（§12）— 新しい読み手は `karume/5` と新コンテナしか読まず、
両読みは実装しない。ここが動かすのは**容器と宣言の形**だけである:

- **payload の生バイトは 1 バイトも変えない**。末尾の詰め物は
  §4.1 のとおり**新たに焼かれる**ので、突き合わせるのは block 全体ではなく initializer ごとの
  payload（旧 shard の実体の sha256 = 新 block の payload 部の sha256）。
- **IR は v1 → v2 へ再 serialize する**（逐語同一は保たない）。改名と正準直列化は
  {@link karume.container.ir_v2_document}。
- **codec は台帳へ写す**（`i8 → int8-sym` / `i4 → int4-sym-g` / `i2 → int2-off`）。`ternary` へは
  写さない — 値域が部分集合でも「三値である」という主張は量子化器の側がするもの。
- **`rowAxis` は消費側 op から引く**（{@link karume.container.weight_channel_axes} —
  `conv_transpose1d` だけ 1）。旧 keepdim 形の scale が**その軸の形ちょうど**であることを
  現物の宣言で確かめてから焼く（バイト数だけ合わせると per-column の scale を per-channel として
  宣言できてしまう）。

MUST: 自己検査（書いたものを読み直して initializer ごとに sha256 を突き合わせる）を**通してから
据える** — 書き出しは一時 path（`.partial`）へ行い、検査が通った回だけ `os.replace` で本番名へ
移す。**旧入力は読むだけ**で、消しも書き換えもしない。

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
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    GRAPH_NAME_PATTERN,
    PART_LENGTH_CHOICES,
    SHA256_HEX_PATTERN,
    AssetInput,
    ContainerFormatError,
    DocumentRef,
    Encoding,
    Provenance,
    base_path,
    codec_entry,
    concrete_shape,
    container_bindings,
    sequence_siblings,
)
from karume.dist import (
    REPO_RE,
    REVISION_RE,
    assert_manifest_limits,
    file_ref,
    generator_tag,
    manifest_text,
    sha256_file,
)
from karume.ir import IrGraph
from karume.legacy import (
    IR_METADATA_KEY,
    SourceTensor,
    _read_metadata,
    payload_chunks,
    read_component,
    resolve_shards,
    safetensors_header,
)
from karume.ple import PLE_INDEX_ASSET, PLE_PACK_FACTOR, PleError, ple_assets, ple_row_bytes
from karume.publish import PublishError, PublishResult, publish_container
from karume.verify import assert_reader_layout, parse_ir_graph

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
PLE_OWNER: Mapping[str, str] = {"gemma4": "model", "gemma4-qat": "model"}

#: 旧 PLE shard が持つメタデータのキー（索引との整合をここで突き合わせる）。
PLE_METADATA_KEY = "karume_ple"

#: CLI の `--part-bytes` の単位（MiB — 選択集合がちょうど MiB の整数倍）。
_MIB = 1024 * 1024


class MigrateError(ValueError):
    """移行の前提が破れた（写し先の無い格納・宣言と現物の食い違い・出力先の残骸）。"""


def _assert_part_length(part_bytes: int) -> None:
    """part 長は**書き手の選択集合**の 1 つ MUST（container-v1 §4.2 / ADR 0108）。

    書き手（{@link karume.container.write_model_container}）が強制するのは天井だけなので、
    集合の検査はここで持つ — 天井の内側の半端な値（300 MiB など）を黙って通すと、part 長を
    比べるつもりの焼き直しが集合外の容器を配布物に混ぜる。
    """
    if part_bytes not in PART_LENGTH_CHOICES:
        choices = ", ".join(str(choice // _MIB) for choice in PART_LENGTH_CHOICES)
        raise MigrateError(
            f"part 長 {part_bytes} バイトが書き手の選択集合 {{{choices}}} MiB の外"
            " — container-v1 §4.2 / ADR 0108"
        )


#: 1 コンポーネントの移行結果（据えた part 列・2 文書・突き合わせた本数）。公開の 3 段は
#: `karume.publish` と共有なので、結果の器も同じものを使う。
MigrationResult = PublishResult


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
    """initializer の宣言 shape（{@link karume.container.concrete_shape} の翻訳層）。

    本体は書き手と**同じ 1 本**で、ここが足すのは例外の語彙の翻訳だけ（{@link _bindings} と
    同じ形）— 移行 CLI の呼び手は `MigrateError` だけを捕まえる。
    """
    try:
        return concrete_shape(graph, name, where)
    except ContainerFormatError as cause:
        raise MigrateError(str(cause)) from cause


def _bindings(graph: IrGraph) -> dict[str, Encoding]:
    """旧 IR の `storage` → コンテナの束縛。

    MUST: 導出は書き手と**同じ 1 本**（{@link karume.container.container_bindings}）を通る。
    移行側で別に綴ると、同じ資産から「旧形から移した容器」と「直接書いた容器」で別の束縛が
    出る。ここが足すのは例外の語彙の翻訳だけ。
    """
    try:
        return container_bindings(graph)
    except ContainerFormatError as cause:
        raise MigrateError(str(cause)) from cause


def _expected_scale_shape(shape: Sequence[int], encoding: Encoding) -> list[int]:
    """旧配布形が持っているはずの scale の形（旧 IR v1 の受理形 — keepdim 形は IR v2 で退役し、
    知っているのはこの移行 CLI だけ）。"""
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


def _assert_no_leftovers(final: Path, graph_path: Path | None) -> None:
    """出力先に前回の成果物が残っていない（**消さずに止まる** — どれを配るかが決まらない）。

    MUST: 移行だけがこの門を持つ（export の再実行は前回の出力を据え替えてよいが、移行は
    「旧入力は読むだけ・出力先は空から作る」が成立条件なので、残骸を黙って消さない）。
    """
    existing = [str(path) for path in sequence_siblings(final)]
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
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    write_graph: bool = False,
    part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> MigrationResult:
    """コンポーネント 1 つ（旧単一形 / 旧 shard 列）を `krm` へ移す。

    `path` は代表 path でも手元の現物（`...-00001-of-00002.safetensors`）でもよい
    （{@link karume.container.base_path} が畳む）。`graph_name` の既定は**親ディレクトリ名**
    （配布形のコンポーネント名）で、コンテナの語彙（`[A-Za-z0-9._-]{1,64}`）から外れる場合は
    明示する。

    `assets`（資産名 → {@link karume.container.AssetInput}）は旧 shard 列の**隣に散っている**
    バイト列（PLE sidecar・`rope_base`）を同じ容器へ畳む席。リポ丸ごとモードは旧 manifest の
    宣言から自分で組む（{@link _convert_unit}）ので、ここを使うのは manifest を持たない
    置き場（系列ディレクトリ）を移す呼び手 — 畳み方（どのファイルがどの資産名か）は family を
    知っている側にしか決められないので、core は受け取るだけにする（ADR 0065）。

    `part_bytes` は part 長（書き手の選択集合 {@link karume.container.PART_LENGTH_CHOICES} の
    1 つ — 集合外は `MigrateError`）。`_block_bytes` は**テストからのみ触る**寸法の差し込み
    （合成の小さな資産で piece 分割を踏むため）— 公開ノブではない。
    """
    _assert_part_length(part_bytes)
    source = base_path(Path(path))
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
        assets=assets,
        single=single,
        graph_path=final.with_suffix(GRAPH_SUFFIX) if write_graph else None,
        part_bytes=part_bytes,
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
    （{@link karume.legacy.resolve_shards}）も旧 manifest の宣言もここへ来る前に済んでいる。

    MUST: 書く → 読み直して検証 → 据え替えの 3 段は export の一本道と**同じ 1 本**
    （{@link karume.publish.publish_container}）を通る。ここが足すのは旧形の読み取りと、
    旧形にしか無い前提（scale の形・テンソルの過不足・出力先が空であること）の検査だけである。
    """
    metadata, stored = read_component(shard_paths)
    graph = parse_ir_graph(metadata[IR_METADATA_KEY])
    bindings = _bindings(graph)
    _assert_tensor_cover(bindings, stored)
    _assert_scale_layouts(graph, bindings, stored)
    _assert_no_leftovers(final, graph_path)
    try:
        return publish_container(
            final,
            graph,
            _SourcePayloads(stored),
            bindings,
            graph_name=graph_name,
            provenance=provenance,
            assets=assets,
            single=single,
            graph_path=graph_path,
            part_bytes=part_bytes,
            block_bytes=block_bytes,
        )
    except PublishError as cause:
        raise MigrateError(str(cause)) from cause


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


def _sha256(value: Any, where: str) -> str:
    _require(
        isinstance(value, str) and SHA256_HEX_PATTERN.match(value) is not None,
        f"{where} が sha256（小文字 16 進 64 文字）でない: {value!r}",
    )
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
        _sha256(obj["sha256"], f"{where}.sha256"),
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


#: `pipeline` の綴り `<name>/<major>`（hub の `parsePipeline` と同じ受理形 — 実物は文字列で、
#: `{name, major}` のオブジェクトではない）。
PIPELINE_RE = re.compile(r"^([A-Za-z0-9_-]+)/(\d+)$")


@dataclass(frozen=True)
class LegacyModel:
    #: `<name>/<major>` の文字列そのまま（karume/5 へもこの綴りで写す）。
    pipeline: str
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
    pipeline = _text(obj["pipeline"], f"{where}.pipeline")
    _require(
        PIPELINE_RE.match(pipeline) is not None,
        f"{where}.pipeline が '<name>/<major>' の形でない: {pipeline!r}",
    )
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
    part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> RepositoryResult:
    """旧 `karume/4` のリポを丸ごと `karume/5` + `krm` へ移す（ADR 0109 決定 8）。

    旧リポは**読むだけ**で、成果物は全部 `out_dir` の下に出る。`part_bytes`（書き手の選択集合 —
    集合外は `MigrateError`）と `_block_bytes`（テストからのみ触る寸法の差し込み）は部品単位
    モードと同じ。

    MUST: 書くのは**分割形だけ**（単一形の席が無い）— `karume/5` の `container.parts` は
    part 0 + part 1 の 2 要素以上 MUST で、単一形はその形を作れない（ADR 0109 決定 3）。
    """
    _assert_part_length(part_bytes)
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
    _assert_declared_files(repo, legacy)
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
                part_bytes=part_bytes,
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


def _assert_declared_files(repo: Path, legacy: LegacyManifest) -> None:
    """旧 manifest が宣言する自リポのファイルを、現物の size（stat）と sha256 に突き合わせる。

    MUST: 変換と複写より**前に**全部見る。移行の「生バイト同一」は手元のファイルとの比較なので、
    手元のミラーが旧 revision と食い違っていると（途中まで更新・壊れたファイル）、その中身が
    新しい sha256 で洗浄された `karume/5` になり、逐語で写す `assets` の FileRef は現物と違う
    sha256 を名乗る。越境参照は手元に実体が無いので見ない（引き写す側の綴りは読み取りで見る）。
    """
    checked: set[FileRef] = set()
    for model_name, model in legacy.models.items():
        entries = [entry for labels in model.weights.values() for entry in labels.values()]
        refs = [*(ref for entry in entries for ref in entry.refs), *model.assets.values()]
        for ref in refs:
            if ref.cross is not None or ref in checked:
                continue
            checked.add(ref)
            where = f"models['{model_name}']: 旧 manifest が宣言する '{ref.path}'"
            path = repo / ref.path
            _require(path.is_file(), f"{where} が手元に無い")
            size = path.stat().st_size
            _require(size == ref.size, f"{where} の長さ {size} が宣言 {ref.size} と違う")
            actual = sha256_file(path)
            _require(
                actual == ref.sha256,
                f"{where} の sha256 が宣言と違う（宣言 {ref.sha256} / 現物 {actual}）"
                " — 手元のミラーが旧 revision と食い違っている",
            )


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
        key = base_path(Path(_file_ref(head[0], f"{label}.parts[0]").path)).as_posix()
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
                _sha256(document["sha256"], f"{where}.descriptor.{key}.sha256"),
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
        stems.add(base_path(Path(name.name)).stem)
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
            "pipeline": model.pipeline,
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
    pipeline = model.pipeline.split("/", 1)[0]
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

    受理集合は `packages/models/src/gemma/ple-index.ts` の鏡像 — schema 1（I8・`storage` 欄なし。
    karume-gemma4 の実資産）と schema 2（packed の `i2` / `i4`・ADR 0097）。戻りの `storage` は
    schema 1 でも `"i8"` に正規化してある（新索引 schema 3 は格納を必ず綴る）。
    """
    raw_root = _object(_read_json(path, where), where)
    # MUST: 版の判定を欄の検査より先に置く（{@link read_legacy_manifest} と同じ理由 — 別の版は
    # 「欄が欠けている」ではなく「その版は畳めない」と言うのが直す側にとって決定的）。
    schema = raw_root.get("schema")
    _require(
        schema in (1, 2),
        f"{where}.schema が {schema!r} — 容器へ畳めるのは schema 1（i8）と 2（packed）だけ",
    )
    if schema == 1:
        _keys(raw_root, ["schema", "tokens", "layers", "dim", "embedScale", "shards"], [], where)
        root: dict[str, Any] = {**raw_root, "storage": "i8"}
    else:
        _keys(
            raw_root,
            ["schema", "storage", "tokens", "layers", "dim", "embedScale", "shards"],
            [],
            where,
        )
        root = dict(raw_root)
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
    """旧 sidecar の safetensors を**ヘッダだけ**読む（PLE shard は数百 MB になる）。

    MUST: 旧 shard の読み取り（{@link karume.legacy.read_component}）と同じ門を通す — レイアウト
    （既知 dtype・宣言長の一致・隙間なし・整列）を先に見て、`__metadata__` は文字列 → 文字列の
    マップとして**逐語**で受ける（型を強制変換すると壊れた sidecar が遠い場所の例外になる）。
    """
    assert_reader_layout(path, allow_legacy_dtypes=True)
    header = safetensors_header(path)
    with path.open("rb") as handle:
        data_start = 8 + int.from_bytes(handle.read(8), "little")
    metadata = _read_metadata(path, header)
    entries: dict[str, _StoredEntry] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        entries[name] = _StoredEntry(spec["dtype"], tuple(spec["shape"]), begin, end)
    return metadata, entries, data_start


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
    segments: tuple[tuple[Path, int, int], ...],
) -> Callable[[int, int], bytes]:
    """連結した区間列に対する「バイト範囲を読む」呼び出し（{@link ple_assets} の受け口）。"""

    def read(begin: int, end: int) -> bytes:
        return _read_segments(segments, begin, end)

    return read


def ple_sidecar_assets(
    index_path: Path, *, where: str | None = None, block_bytes: int = BLOCK_MAX_BYTES
) -> dict[str, AssetInput]:
    """旧 PLE sidecar（`ple.json` + **同じディレクトリに並ぶ** shard 列）を容器の資産へ畳む。

    manifest を持たない置き場（export 系列のディレクトリ）を移す呼び手の入口。畳み方は
    リポ丸ごとモードと**同じ 1 本**（{@link _fold_ple}）で、違うのは shard の在処を
    旧 manifest の `assets` から引くか索引が名乗るファイル名から引くかだけである。

    NOTE: この経路では畳み先の門のうち**並びと本数の 2 つは自明に真**（列を索引そのものから
    組むので）— 効くのは旧 manifest から在処を引く側だけである。したがって「索引に載って
    いない `ple-*.safetensors` が現物に在る」を見るのは**呼び手の責務**で、ここは通す。
    """
    at = str(index_path) if where is None else where
    index = read_ple_index(index_path, at)
    directory = index_path.parent
    shards = [(str(entry["file"]), directory / str(entry["file"])) for entry in index["shards"]]
    return _fold_ple(shards, index, block_bytes, at)


def _ple_assets(
    repo: Path, fold: _PleFold | None, block_bytes: int, where: str
) -> dict[str, AssetInput]:
    """旧 manifest が宣言した PLE sidecar を容器の資産へ畳む（在処は `assets` の FileRef）。"""
    if fold is None:
        return {}
    # 先頭は索引そのもの（畳み先では新しい索引に置き換わる）ので落とす。
    shards = [(name, repo / ref.path) for name, ref in fold.refs[1:]]
    return _fold_ple(shards, fold.index, block_bytes, where)


def _fold_ple(
    shards: Sequence[tuple[str, Path]],
    index: Mapping[str, Any],
    block_bytes: int,
    where: str,
) -> dict[str, AssetInput]:
    """PLE shard 列 → 容器の資産（組み立て自体は {@link ple_assets} の 1 本）。

    ここが持つのは**旧 sidecar の読み取り**だけ — 索引と現物の突合（行数・バイト長・shard の
    並び・`karume_ple` メタデータ）を済ませ、token 順に連結した区間列を読み口として渡す。旧
    shard の境界は意味を持たない（全 shard を token 順に連結して切り直す）。
    """
    tokens, layers, dim = index["tokens"], index["layers"], index["dim"]
    try:
        row_bytes = ple_row_bytes(index["storage"], layers, dim)
    except PleError as cause:
        raise MigrateError(f"{where}: {cause}") from cause
    segments: dict[str, list[tuple[Path, int, int]]] = {"values": [], "scales": []}
    _require(
        len(shards) == len(index["shards"]),
        f"{where}: PLE shard が {len(shards)} 本（索引は {len(index['shards'])} 本）",
    )
    for position, (name, path) in enumerate(shards):
        declared = index["shards"][position]
        _require(
            name == declared["file"],
            f"{where}: PLE shard の並びが索引と違う（{position} 番目は '{declared['file']}'）",
        )
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
            segments[key].append((path, data_start + entry.begin, entry.nbytes))
    try:
        return ple_assets(
            storage=index["storage"],
            tokens=tokens,
            layers=layers,
            dim=dim,
            embed_scale=index["embedScale"],
            read_values=_segment_reader(tuple(segments["values"])),
            read_scales=_segment_reader(tuple(segments["scales"])),
            block_bytes=block_bytes,
        )
    except PleError as cause:
        raise MigrateError(f"{where}: {cause}") from cause


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
    # schema 1 の shard メタデータは `storage` を持たない（索引側は i8 へ正規化済み）。
    keys = ("schema", "tokens", "layers", "dim") + (("storage",) if index["schema"] == 2 else ())
    expected = {key: index[key] for key in keys}
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
        "--writer",
        default=None,
        help="provenance.writer（既定: 書かない — 呼び手が明示したときだけ容器に載る）",
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
    choices = [choice // _MIB for choice in PART_LENGTH_CHOICES]
    parser.add_argument(
        "--part-bytes",
        type=int,
        choices=choices,
        default=DEFAULT_PART_BYTES // _MIB,
        help=f"part 長 — 書き手の選択集合 {{{','.join(map(str, choices))}}} MiB",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """指定されたコンポーネント（またはリポ丸ごと）を移行する。

    MUST: 落ちたところで止める（残りを移して最後にまとめない）— 例外は破れた不変条件まで
    綴ってあるので、そのまま送出するのが最も情報量が多い。

    MUST: `provenance.writer` は**既定で書かない**。生成器タグを既定で埋めると、移行した容器と
    recipe が直接 export した容器（`PROVENANCE` はどれも `writer` を綴らない）が永久に別バイト
    になり、「ミラーを焼き直したらモデル記述だけが動く」形になる。ツールの版を持つ席は
    `karume.json` の `generator` 欄 1 箇所で、容器側に写しを置く理由が無い（golden が
    `writer` を書かない理由と同じ — {@link karume.goldens.GOLDEN_PROVENANCE}）。
    """
    args = build_parser().parse_args(argv)
    provenance = Provenance(
        license=args.license,
        notice=args.notice,
        upstream_revision=args.upstream_revision,
        writer=args.writer,
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
            part_bytes=args.part_bytes * _MIB,
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
        part_bytes=args.part_bytes * _MIB,
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
