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
from karume.verify import verify_model


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


def export_to_file(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    path: str | Path,
    *,
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
) -> IrGraph:
    """変換して書き出し、書いたファイルを IR v1 の全規則で検証して返す。

    `weight_dtype` が `"f16"` / `"i8"` のとき適格な重みスロットだけが圧縮格納になる
    （ADR 0018 / 0019）。呼び出し側は**丸め（fake-quant）を参照・golden の採取より前に
    済ませておく** MUST — 掛け忘れは write_model が fail loudly で落とす（emit.py の適格判定）。
    `weight_scales` は i8 のときの per-channel scale 台帳（`quantize.fake_quant_int8` の戻り）。

    MUST: 書き出しの直後に verify_model を通す — 「書けたが読めない」ファイルを
    配布物として残さないための門（ADR 0005 の fail loudly 規律）。門を実効にするため、
    書き出しと検証は**同じディレクトリの一時ファイル**に対して行い、verify が通って
    はじめて `os.replace` で `path` へ差し替える（同一ディレクトリなので置換は原子的）。
    書き出しか検証が落ちたときは一時ファイルを捨て、既存の `path` は 1 バイトも変えない
    — 直接 truncate すると「再エクスポートに失敗した」だけで手元の正常な配布物が消える。

    NOTE: この原子性はここの層のもの。`emit.write_model` を直接呼ぶ経路（検証を挟まない
    書き出し）は原子化の外で、渡された path をその場で truncate する。
    """
    graph, tensors = export_module(
        module,
        args,
        dynamic_shapes=dynamic_shapes,
        symbol_names=symbol_names,
        preserved=preserved,
    )
    final = Path(path)
    # 一意 suffix — 同じ final を狙う別プロセス / 別ターゲットの一時ファイルと衝突させない。
    staged = final.with_name(f"{final.name}.{uuid4().hex}.partial")
    try:
        write_model(staged, graph, tensors, weight_dtype=weight_dtype, weight_scales=weight_scales)
        verified = verify_model(staged)
        os.replace(staged, final)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return verified
