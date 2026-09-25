"""export → 正規化 → 変換 → 格納変換 → 公開の一本道。

エクスポート台本（モデルごとのスクリプト）が段の順序を各自で書くと、正規化の抜けや
検証漏れが台本ごとに散る。順序はここ 1 箇所で決める。

配布形は**コンテナ**（`krm` — container-v1）で、書き出しの 3 段（書く → 読み直して検証 →
据え替え）は `karume.publish` が持つ。ここが足すのは「どの名前で・どの出所で・どの資産を
同梱して据えるか」だけである。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    GRAPH_NAME_PATTERN,
    AssetInput,
    ContainerFormatError,
    Provenance,
)
from karume.convert import PRESERVED_OP_PREFIXES, convert, curated_decompositions
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IrGraph
from karume.normalize import normalize_graph
from karume.publish import publish_container
from karume.verify import assert_op_contracts, assert_runtime_support

#: コンテナ（モデル容器）の拡張子（container-v1 §1 — 種別は magic が持つが名前も分けておく）。
MODEL_SUFFIX = ".krm"


def export_module(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    *,
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """nn.Module を IR グラフ + 格納テンソルへ変換する。

    `preserved` は分解を止める高位 op の接頭辞集合（既定は 11 op）。**ターゲット別**に
    差し替えられるのは融合 attention（ADR 0023）のためで、SDPA 保存は
    `PRESERVED_OP_PREFIXES_WITH_ATTENTION` を渡したターゲットだけが得る — 表をグローバルに
    広げると mask 付き SDPA を持つグラフが `_h_attention` の fail loudly で export 不能になる。
    """
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    decomposed = ep.run_decompositions(curated_decompositions(preserved))
    normalize_graph(decomposed)
    return convert(decomposed, symbol_names=symbol_names)


def _assert_graph_name(graph_name: str) -> str:
    """容器のグラフ名が語彙（`[A-Za-z0-9._-]{1,64}`）に収まることを落とす。

    MUST: 既定を持たない（`provenance` と同じ扱い）— 呼び手は**作業席**（`<部品>.staging/`）へ
    書くので、親ディレクトリ名を既定にすると全系列のグラフ名が `<部品>.staging` になる。
    語彙には `.` が入るので fail loudly もせず、ランタイムが部品名で引いた時点で初めて
    「コンテナにグラフが無い」になる。部品名（= `karume.json` の weights のキー）を知って
    いるのは呼び手だけなので、呼び手が名乗る（container-v1 §2.1 — 移行 CLI が使う部品名と
    同じ綴り MUST）。
    """
    if GRAPH_NAME_PATTERN.match(graph_name) is None:
        raise ContainerFormatError(
            f"グラフ名 '{graph_name}' がコンテナの語彙（{GRAPH_NAME_PATTERN.pattern}）から外れる"
        )
    return graph_name


def _assert_model_path(path: str | Path) -> Path:
    """出力 path が容器の拡張子（`.krm`）で終わることを落とす。

    MUST: 他の拡張子を `.krm` へ黙って読み替えない — 配布形は `krm` の 1 つだけで
    （container-v1 §8）、別の拡張子を渡す呼び手は退役した形を書くつもりでいる。
    """
    final = Path(path)
    if final.suffix != MODEL_SUFFIX:
        raise ValueError(
            f"出力 path '{final}' の拡張子が {MODEL_SUFFIX} ではない（配布形は krm だけ）"
        )
    return final


def publish_model(
    path: str | Path,
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    provenance: Provenance,
    graph_name: str,
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
    fixed_weights: Mapping[str, FixedQuantizedWeight] | None = None,
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    _part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> IrGraph:
    """格納変換 → 公開の 3 段（変換済みのグラフを受ける入口）。

    MUST: 配布形を作る経路は**この 1 本**を通る。手術を挟む台本（decode 変種）が同じ段を各自で
    綴っていた頃は、書き出し側の変更が写しの側で黙って壊れた。

    `provenance` は**必須**（既定値で出所を偽らない — `license` を落とした配布形を作らない）。
    `graph_name` も**必須**で、綴りは**配布形の部品名 = `karume.json` の weights のキー**
    MUST（container-v1 §2.1）— ランタイムは `prepareContainer(opened,
    <weights キー>)` でグラフを名前で引き、移行 CLI も同じキーで焼く。**据え替え先の
    ディレクトリ名とは一致しないことがある**（系列直下に容器を置く family は
    `siglip2-so400m-patch14-384` のような系列名が、irodori は `caption-proj` のような
    ハイフン綴りがディレクトリ名になる — キーはそれぞれ `vision` / `caption_proj`）ので、
    **呼び手は部品名を定数で名乗る**（ディレクトリから導かない）。作業席の名前
    （`<部品>.staging`）を拾わせない側の門は {@link _assert_graph_name}。
    `assets`（資産名 → {@link karume.container.AssetInput}）は PLE 索引や `rope_base` のような
    「重みではないが同じ容器で配るバイト列」の席で、重みの part の**後ろ**の専用 part に載る
    （ADR 0109 決定 4）。

    `path` は `.krm` で終わる代表 path（他の拡張子は `ValueError` — {@link _assert_model_path}）。
    既定の分割形では part 0 から `<stem>-NNNNN-of-NNNNN.krm` の連番になり、`path` 自身は
    書かれない（HF の公式配布は分割形だけ — container-v1 §8）。`single=True` は手元用の単一形。

    戻すのは**格納宣言を commit したグラフ**（実際に焼いた `storage` を持つビュー）。渡された
    `graph` は 1 バイトも変えない。「書いたものが読めるか」は `karume.publish` の読み直し検証
    （payload / 資産 / 2 文書の突合）が済ませている。

    MUST: 書き出しの**前**に IR の受理規則（ランタイム支援 + op 契約）を掛ける — 台本は
    `to_states_form` のようなグラフ手術を挟むので、変換段の検査だけでは「書けるがランタイムが
    読めない」容器を止められない。掛ける相手は `stored_model` が commit したグラフ（実際に
    焼く格納を持つビュー）で、落ちた回は **1 バイトも据わらない**。
    """
    final = _assert_model_path(path)
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=weight_dtype,
        weight_scales=weight_scales,
        weight_dtype_overrides=weight_dtype_overrides,
        fixed_weights=fixed_weights,
    )
    assert_runtime_support(stored.graph)
    assert_op_contracts(stored.graph)
    publish_container(
        final,
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name=_assert_graph_name(graph_name),
        provenance=provenance,
        assets=assets,
        single=single,
        part_bytes=_part_bytes,
        block_bytes=_block_bytes,
    )
    return stored.graph


def export_to_file(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    path: str | Path,
    *,
    provenance: Provenance,
    graph_name: str,
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
    assets: Mapping[str, AssetInput] = {},
) -> IrGraph:
    """変換して `krm` として据え、格納宣言を commit したグラフを返す（export → 公開の一本道）。

    `weight_dtype` が `"f16"` / `"i8"` / `"i4"` のとき適格な重みスロットだけが圧縮格納になる
    （ADR 0018 / 0019 / 0069）。呼び出し側は**丸め（fake-quant）を参照・golden の採取より前に
    済ませておく** MUST — 掛け忘れは `emit.stored_model` が fail loudly で落とす。
    `weight_scales` は i8 / i4 の scale 台帳（`quantize.fake_quant_int8` / `fake_quant_int4` の
    戻り — 混成では両者を合流して渡す）。`weight_dtype_overrides`（テンソルキー → dtype）は
    1 本単位の明示指定で既定に優先する（混成格納 — 線引きは `emit._plan_weight_dtype`）。

    `provenance` と `graph_name`（= 配布形の部品名 = `karume.json` の weights のキー）は
    **必須** — 意味と理由は {@link publish_model}。書き出し以降（原子性・part 分割・後始末）も
    向こうが持つ。
    """
    # 出力 path の拡張子は export の前に落とす（重い export を回してから拒否しない）。
    _assert_model_path(path)
    graph, tensors = export_module(
        module,
        args,
        dynamic_shapes=dynamic_shapes,
        symbol_names=symbol_names,
        preserved=preserved,
    )
    return publish_model(
        path,
        graph,
        tensors,
        provenance=provenance,
        weight_dtype=weight_dtype,
        weight_scales=weight_scales,
        weight_dtype_overrides=weight_dtype_overrides,
        assets=assets,
        graph_name=graph_name,
    )
