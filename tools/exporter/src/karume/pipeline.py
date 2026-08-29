"""export → 正規化 → 変換 → 書き出し → 検証の一本道。

エクスポート台本（モデルごとのスクリプト）が段の順序を各自で書くと、正規化の抜けや
検証漏れが台本ごとに散る。順序はここ 1 箇所で決める。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from karume.convert import PRESERVED_OP_PREFIXES, convert, curated_decompositions
from karume.emit import write_model
from karume.ir import IrGraph
from karume.normalize import normalize_graph
from karume.shards import shard_path, shard_siblings
from karume.verify import verify_shards


def export_module(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    *,
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """nn.Module を IR v1 グラフ + 格納テンソルへ変換する。

    `preserved` は分解を止める高位 op の接頭辞集合（既定は 11 op）。**ターゲット別**に
    差し替えられるのは融合 attention（ADR 0023）のためで、SDPA 保存は
    `PRESERVED_OP_PREFIXES_WITH_ATTENTION` を渡したターゲットだけが得る — 表をグローバルに
    広げると mask 付き SDPA を持つグラフが `_h_attention` の fail loudly で export 不能になる。
    """
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    decomposed = ep.run_decompositions(curated_decompositions(preserved))
    normalize_graph(decomposed)
    return convert(decomposed, symbol_names=symbol_names)


def publish_model(
    path: str | Path,
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
) -> IrGraph:
    """書き出し → 検証 → 据え替えの 3 段（変換済みのグラフを受ける入口）。

    MUST: 配布形を作る経路は**この 1 本**を通る。手術を挟む台本（decode 変種の
    `_write_container`）が同じ 3 段を各自で綴っていた頃は、shard 分割のような書き出し側の
    変更が「一時 path をそのまま検証する」写しの側で黙って壊れた（分割されると `path` 自身は
    書かれない）。原子性の規律も後始末も 1 箇所に置く。

    MUST: 書き出しの直後に verify を通す — 「書けたが読めない」ファイルを配布物として
    残さないための門（ADR 0005 の fail loudly 規律）。門を実効にするため、書き出しと検証は
    **同じディレクトリの一時ファイル**に対して行い、verify が通ってはじめて `os.replace` で
    `path` へ差し替える（同一ディレクトリなので置換は原子的）。書き出しか検証が落ちたときは
    一時ファイル（分割時は連番ぶんも）を捨て、既存の `path` は 1 バイトも変えない —
    直接 truncate すると「再エクスポートに失敗した」だけで手元の正常な配布物が消える。

    データ節が `shards.SHARD_TAIL_LIMIT` を超えるコンポーネントは連番の shard 列として
    据わる（ADR 0070 決定 1）。**据え替えはファイル単位でしか原子的にならない**ので、
    中断した回は「前回の形と今回の形が混ざったコンポーネント」を残しうる — その現場は
    `shards.resolve_shards` が組み立て時に fail loudly で受ける（黙って一方を配らない）。
    据わった後に**前回の出力の残り**（分割数が変わった / 単一 ↔ 分割が入れ替わった回の
    置き去り）を消すのは、単一ファイルを truncate で上書きしていたのと同じ意味の後始末で、
    このコンポーネントの出力名以外には触れない。

    NOTE: この原子性はここの層のもの。`emit.write_model` を直接呼ぶ経路（検証を挟まない
    書き出し）は原子化の外で、渡された path をその場で truncate する。
    """
    final = Path(path)
    # 一意 suffix — 同じ final を狙う別プロセス / 別ターゲットの一時ファイルと衝突させない。
    staged = final.with_name(f"{final.name}.{uuid4().hex}.partial")
    try:
        written = write_model(
            staged,
            graph,
            tensors,
            weight_dtype=weight_dtype,
            weight_scales=weight_scales,
            weight_dtype_overrides=weight_dtype_overrides,
        )
        verified = verify_shards(written)
        total = len(written)
        published = [
            final if total == 1 else shard_path(final, index, total)
            for index in range(1, total + 1)
        ]
        for source, target in zip(written, published, strict=True):
            os.replace(source, target)
    except BaseException:
        # 書き出しが途中で落ちた回は返り値が無いので、一時 path の**名前の形**から拾う
        # （分割の途中まで書けた連番も残骸なので同じ席に居る）。
        for leftover in shard_siblings(staged):
            leftover.unlink(missing_ok=True)
        raise
    for stale in shard_siblings(final):
        if stale not in published:
            stale.unlink()
    return verified


def export_to_file(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    path: str | Path,
    *,
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
) -> IrGraph:
    """変換して書き出し、書いた配布形を IR v1 の全規則で検証して返す（export → 公開の一本道）。

    `weight_dtype` が `"f16"` / `"i8"` のとき適格な重みスロットだけが圧縮格納になる
    （ADR 0018 / 0019）。呼び出し側は**丸め（fake-quant）を参照・golden の採取より前に
    済ませておく** MUST — 掛け忘れは write_model が fail loudly で落とす（emit.py の適格判定）。
    `weight_scales` は i8 / i4 の scale 台帳（`quantize.fake_quant_int8` / `fake_quant_int4` の
    戻り — 混成では両者を合流して渡す）。`weight_dtype_overrides`（テンソルキー → dtype）は
    1 本単位の明示指定で既定に優先する（混成格納 — 線引きは `emit._plan_weight_dtype`）。

    書き出し以降（原子性・shard 分割・後始末）は {@link publish_model} が持つ。
    """
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
        weight_dtype=weight_dtype,
        weight_scales=weight_scales,
        weight_dtype_overrides=weight_dtype_overrides,
    )
