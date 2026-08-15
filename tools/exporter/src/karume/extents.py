"""次元長の同一性判定 — 記号次元（`SymInt`）を guard 無しで突き合わせる 1 箇所。

MUST: SymInt を `==` で比べない（shape をリストごと `==` で比べるのも同じ）— torch は
**ヒント値**で真偽を決めるので、①別のシンボル同士でもヒントが同じなら `True` が返り
②評価そのものが guard を積んで export 済みの制約へ後付けの特殊化を混ぜる（torch 2.13 実測:
`[s97] == [s52]` は `True` を返し、以後 `s97` は `s52` へ置換される）。sympy の式は構造的等価で
比べられるので、記号のまま突き合わせる。同一と**証明できない**組は「違う」に倒す
（受理範囲を広げない側の誤り）。

MUST: `int()` も同じ理由で記号を含む次元へ掛けない（具体値へ黙って特殊化する）。ここが
`free_symbols` を見てから int 化しているのはそのため。

置き場が `karume.dims` でないのは、あちらが TS 側 `format/dims.ts` と対になる**次元表記の
文法**のモジュールで torch を 1 つも import しないため（`karume verify` / `karume dist` は
export 台本の重い import を 1 つも読まずに起動する — `cli.py` の MUST）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def extent_key(dim: Any) -> Any:
    """次元の同一性判定に使う鍵（記号は式そのもの、具体値は int）。"""
    if isinstance(dim, torch.SymInt):
        expr = dim.node.expr
        return expr if expr.free_symbols else int(dim)
    return int(dim)


def extent_keys(dims: Sequence[Any]) -> tuple[Any, ...]:
    """shape（またはその一部）を鍵の列へ。タプル同士の `==` が shape の同一性判定になる。"""
    return tuple(extent_key(dim) for dim in dims)


def same_extents(out: Any, src: Any) -> bool:
    """2 つの meta が同じ shape か（記号次元は式の構造で突き合わせる）。

    テンソルを運ばない meta（SymInt / None）は「違う」— shape を持たないものは同一性を
    主張できない。
    """
    if not isinstance(out, torch.Tensor) or not isinstance(src, torch.Tensor):
        return False
    if out.dim() != src.dim():
        return False
    return extent_keys(out.shape) == extent_keys(src.shape)
