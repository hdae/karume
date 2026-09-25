"""配布形（`krm`）を据える 3 段 — 書く → 読み直して検証 → 据え替え。

配布形を作る経路は**この 1 本**を通る（export の一本道 `karume.pipeline` も移行 CLI
`karume.migrate` も）。3 段を呼び手が各自で綴っていた頃は、書き出し側の変更（分割数が変わる）が
「一時 path をそのまま検証する」写しの側で黙って壊れた。原子性の規律も後始末もここに置く。

MUST: 書き出しの直後に**読み直して**検証を通す — 「書けたが読めない」ファイルを配布物として
残さないための門（ADR 0005 の fail loudly 規律）。検証は 4 つで、1〜3 は読み直した容器と
呼び手が渡した材料の突合、4 は宣言の追加条件である:

1. **payload の一致**: initializer と companion scale の payload バイト列が、渡した実体と
   sha256 で一致する。突き合わせるのは block 全体ではなく **payload 部** — 末尾の詰め物は
   書き手が焼くバイトで、渡した側には相手が無い（container-v1 §4.1 / §12 の不変条件 5）。
2. **資産の一致**: 役割・論理長・payload・末尾の詰め物が 0x00 であること。
3. **2 文書**: グラフ記述 / モデル記述のバイト長と sha256（manifest `karume/5` の
   `container.descriptor` が名乗る値と同じ事実 — ADR 0109 決定 3）。
4. **`ternary` のコード**: payload の全 2 bit コードが `{1, 2, 3}`（container-v1 §6.3 —
   `karume verify` と同じ検査。TS の読み手はロード時に落とすので、据える前に落とす）。

MUST: 門を実効にするため、書き出しと検証は**同じディレクトリの一時ファイル**（`.partial`）に
対して行い、通ってはじめて `os.replace` で本番名へ据える（同一ディレクトリなので置換は原子的）。
据え替えの途中で落ちた回も**何も残さない** — 半分だけ公開された容器を残すと、次の実行が
「前回の成果物が残っている」で止まる（container-v1 §12）。

MUST: 据わった後に**前回の出力の残り**（part 数が変わった回の置き去り）を消す。据え替えは
ファイル単位でしか原子的にならないので、消さないと「前回の形と今回の形が混ざった容器」が
残り、`container_parts` の解決が fail loudly で止まる。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Buffer, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    AssetInput,
    DocumentRef,
    Encoding,
    Provenance,
    ReadContainer,
    container_paths,
    read_container,
    sequence_siblings,
    write_model_container,
)
from karume.ir import IrGraph
from karume.verify import BoundGraph, _assert_ternary_payloads, bind_graphs


class PublishError(ValueError):
    """据える前提が破れた（読み直した容器が渡した材料と食い違う）。"""


@dataclass(frozen=True)
class PublishResult:
    """1 コンポーネントを据えた結果。"""

    #: 据えた `krm` の part 列（part 0 から。単一形なら 1 本）。
    parts: tuple[Path, ...]
    #: 一緒に書いた `krg`（書かなければ `None`）。
    graph: Path | None
    #: 供給計画を組んだ initializer の本数。
    initializers: int
    #: 渡した実体と sha256 を突き合わせた payload の本数（実体 + companion scale）。
    payloads: int
    #: 突き合わせた資産の本数。
    assets: int
    #: 2 文書（グラフ記述 / モデル記述）の `(バイト長, sha256)`。
    descriptor: tuple[DocumentRef, DocumentRef]


def _digest(raw: Buffer) -> str:
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def _assert_payloads_match(
    read_back: ReadContainer,
    bound: BoundGraph,
    bindings: Mapping[str, Encoding],
    tensors: Mapping[str, Buffer],
) -> int:
    """書いた容器の payload を、渡した実体と突き合わせる（モジュール doc の検証 1）。

    取り直しは {@link karume.container.ReadContainer.block} 越しなので、block ごとの sha256 も
    同時に検証される。
    """
    checked = 0
    for name, supply in sorted(bound.supplies.items()):
        digest = hashlib.sha256()
        for block in supply.blocks:
            digest.update(read_back.block(block.id)[: block.payload_bytes])
        _assert_digest(f"initializer '{name}'", digest.hexdigest(), tensors, name)
        checked += 1
        scale_key = bindings[name].scale_key
        # MUST: 片方だけが `None` の食い違いを素通ししない — 渡した束縛と読み直した容器の
        # 束縛表がずれている形そのもので、通すと「宣言 i4 / 実体 i8」の相方（scale を
        # 持つはずの席が scale 無しで据わる）が門を抜ける。
        if (supply.scale is None) != (scale_key is None):
            raise PublishError(
                f"initializer '{name}': 渡した束縛は scale"
                f" {'を持つ' if scale_key is not None else 'を持たない'}のに、"
                f"書いた容器の供給は scale {'を持つ' if supply.scale is not None else 'を持たない'}"
            )
        if supply.scale is not None and scale_key is not None:
            raw = read_back.block(supply.scale.id)[: supply.scale.payload_bytes]
            _assert_digest(
                f"initializer '{name}' の scale '{scale_key}'",
                hashlib.sha256(raw).hexdigest(),
                tensors,
                scale_key,
            )
            checked += 1
    return checked


def _assert_digest(where: str, actual: str, tensors: Mapping[str, Buffer], key: str) -> None:
    expected = _digest(tensors[key])
    if actual != expected:
        raise PublishError(
            f"{where}: payload の sha256 が渡した実体と違う（渡した {expected} / 書いた {actual}）"
        )


def _assert_assets_match(read_back: ReadContainer, assets: Mapping[str, AssetInput]) -> int:
    """書いた容器の資産を、渡した payload と突き合わせる（モジュール doc の検証 2）。

    突き合わせるのは **payload 部**で、末尾の詰め物が 0x00 であることも見る（詰め物は §4.1 の
    とおり書き手が焼くバイトなので、渡した側には相手が無い）。
    """
    model = read_back.model
    if model is None:  # pragma: no cover - krm を読み直した直後なので在る
        raise PublishError("krg には資産を載せられない")
    for name in sorted(assets):
        asset = assets[name]
        binding = model.assets.get(name)
        if binding is None:
            raise PublishError(f"資産 '{name}' が書いた容器の宣言に無い")
        if binding.role != asset.role:
            raise PublishError(
                f"資産 '{name}': 役割が '{binding.role}'（渡したのは '{asset.role}'）"
            )
        if binding.length != asset.length:
            raise PublishError(
                f"資産 '{name}': 宣言の論理長が {binding.length}（渡したのは {asset.length}）"
            )
        raw = memoryview(read_back.block(binding.block))
        payload = asset.payload
        expected = memoryview(payload() if callable(payload) else payload).cast("B")
        if expected.nbytes != asset.length:
            raise PublishError(
                f"資産 '{name}': 引き直した実体が {expected.nbytes} バイト"
                f"（宣言は {asset.length}）— 引かれるたびに同じバイト列を返す MUST"
            )
        if raw[: expected.nbytes] != expected:
            raise PublishError(f"資産 '{name}': payload が渡したバイト列と違う")
        if bytes(raw[expected.nbytes :]) != b"\x00" * (raw.nbytes - expected.nbytes):
            raise PublishError(f"資産 '{name}': 末尾の詰め物が 0x00 でない")
    return len(assets)


def publish_container(
    final: Path,
    graph: IrGraph,
    tensors: Mapping[str, Buffer],
    bindings: Mapping[str, Encoding],
    *,
    graph_name: str,
    provenance: Provenance,
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    graph_path: Path | None = None,
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> PublishResult:
    """`krm` を書いて検証し、`final`（単一形）/ その連番（分割形）へ据える。

    `tensors` は**テンソルキー → 生バイト**の口で、実体は 1 本ずつ引いて 1 本ずつ手放す
    （`Mapping` を遅延にすれば全量はメモリに載らない）。実体は**複数回引かれる** — 分割形は
    書き出し 1 回 + 読み直し検証 1 回、単一形は書き手が sha256 を採る走査と書き出しの走査で
    2 回 + 読み直し検証 1 回 — ので、同じキーからは毎回同じバイト列が返る MUST。

    `part_bytes` / `block_bytes` は寸法の差し込み（テストが小さな資産で part またぎと piece
    分割を踏むための席）。既定値は、`part_bytes` が container-v1 §4.2 の既定（256 MiB）、
    `block_bytes` が §4.1 / §10 の block 上限（32 MiB）である。

    MUST: 呼び手は**作業席**（`karume.artifacts.staged_publication` の中）へ据える。据え替えの
    途中で落ちた回は「据えかけた part を消して何も残さない」で閉じるので、`final` が
    コミット済みの置き場（前回の成果物が在る場所）だと**前回の last-known-good が消える** —
    そこへ原子的に据えたい呼び手は、作業席へ書いてディレクトリごと swap する側を使う。
    """
    staged = final.with_name(f"{final.stem}.{uuid4().hex}.partial{final.suffix}")
    replaced: list[Path] = []
    published: list[Path] = []
    staged_graph: Path | None = None
    try:
        written = write_model_container(
            staged,
            graph,
            tensors,
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
        payloads = _assert_payloads_match(read_back, bound, bindings, tensors)
        # 宣言の追加条件（モジュール doc の検証 4）— `karume verify` と同じ 1 本を通す。
        _assert_ternary_payloads(read_back, {graph_name: bound})
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
        # `krm` と同じ 3 段（一時名 → `os.replace`）を通す — 最終名へ直接書くと、途中で落ちた
        # 回に**切り詰められた `krg`** が最終名に残る（プロセスが強制終了されれば例外経路の
        # 後始末も走らない）。
        if graph_path is not None:
            staged_graph = graph_path.with_name(f"{graph_path.name}.{uuid4().hex}.partial")
            staged_graph.write_bytes(read_back.extract_graph())
            os.replace(staged_graph, graph_path)
            # 据わった時点で一時名はもう無い（後始末の対象は「自分が書いた最終名」だけ）。
            staged_graph = None
            replaced.append(graph_path)
        for staged_part, target in zip(written, published, strict=True):
            os.replace(staged_part, target)
            replaced.append(target)
    except BaseException:
        # 書き出しが途中で落ちた回は返り値が無いので、一時 path の**名前の形**から拾う。
        for leftover in sequence_siblings(staged):
            leftover.unlink(missing_ok=True)
        if staged_graph is not None:
            staged_graph.unlink(missing_ok=True)
        # MUST: 消すのは**この呼び出しが据えた**現物だけ（`replaced`）— `graph_path` を
        # 無条件に消すと、書き出しが `krg` を抜く前に落ちた回に**前回の `krg`** が道連れになる。
        for target in replaced:
            target.unlink(missing_ok=True)
        raise
    for stale in sequence_siblings(final):
        if stale not in published:
            stale.unlink()
    return PublishResult(
        tuple(published), graph_path, len(bound.supplies), payloads, checked, documents
    )
