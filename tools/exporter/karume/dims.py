"""次元言語: shape の記号次元を表す一次式 `coeff·sym+offset`（1 次元 1 シンボル）。

文法の正本は実装ではなく適合ケース表 `packages/runtime/tests/fixtures/dim-grammar.json`
（リポジトリ直下）で、TS 実装（packages/runtime/src/format/dims.ts）と本モジュールは
同じ表で検証する。片側だけ動かすと
「エクスポータは書けるがランタイムが読めない」次元表記が生まれるため、
文法を変えるときは表・TS・Python を同時に動かすこと。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

# 受理側（TS）の数値は IEEE754 double なので、これを超える次元は往復で値が変わる。
MAX_SAFE_INT = 2**53 - 1

_SYMBOL_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_SYMBOL_RE = re.compile(_SYMBOL_PATTERN)
_DIM_RE = re.compile(rf"(\d+)?({_SYMBOL_PATTERN})(?:\+(\d+))?")


class DimError(ValueError):
    """次元式が正準文法に適合しない / 束縛できない。"""


@dataclass(frozen=True)
class DimExpr:
    """`coeff·sym + offset`。coeff >= 1・offset >= 0 の非負一次式のみ。"""

    coeff: int
    sym: str
    offset: int


def is_symbol_name(name: str) -> bool:
    return _SYMBOL_RE.fullmatch(name) is not None


def _parse_part(text: str | None, minimum: int, omitted: int) -> int | None:
    """省略可能部（coeff=1 / offset=0）の明示表記と先頭ゼロは非正準として拒否する。

    同じ次元に 2 通り以上の表記を許すと、束縛表・shape 同一性判定が表記ゆれで割れる。
    """
    if text is None:
        return omitted
    if text.startswith("0"):
        return None
    value = int(text)
    if value < minimum or value > MAX_SAFE_INT:
        return None
    return value


def try_parse_dim(text: str) -> DimExpr | None:
    """正準表記なら分解、非正準・非該当は None（判定と分解を兼ねる版）。"""
    matched = _DIM_RE.fullmatch(text)
    if matched is None:
        return None
    coeff = _parse_part(matched.group(1), 2, 1)
    offset = _parse_part(matched.group(3), 1, 0)
    if coeff is None or offset is None:
        return None
    return DimExpr(coeff=coeff, sym=matched.group(2), offset=offset)


def parse_dim(text: str) -> DimExpr:
    expr = try_parse_dim(text)
    if expr is None:
        raise DimError(f"次元式 '{text}' が正準文法 coeff·sym+offset に適合しない")
    return expr


def format_dim(expr: DimExpr) -> str:
    """正準表記へ戻す。parse_dim との往復は文字列同一（適合ケース表で固定）。"""
    if not is_symbol_name(expr.sym):
        raise DimError(f"シンボル名 '{expr.sym}' が不正")
    if expr.coeff < 1 or expr.coeff > MAX_SAFE_INT:
        raise DimError(f"係数 {expr.coeff} が不正")
    if expr.offset < 0 or expr.offset > MAX_SAFE_INT:
        raise DimError(f"オフセット {expr.offset} が不正")
    head = "" if expr.coeff == 1 else str(expr.coeff)
    tail = "" if expr.offset == 0 else f"+{expr.offset}"
    return f"{head}{expr.sym}{tail}"


def eval_dim(expr: DimExpr, bindings: Mapping[str, int]) -> int:
    """束縛表を当てて具体次元にする。未束縛・非整数束縛は既定値で進めず落とす。"""
    if expr.sym not in bindings:
        raise DimError(f"シンボル '{expr.sym}' が束縛されていない")
    bound = bindings[expr.sym]
    # bool は int の派生だが次元ではない（True が 1 として黙って通るのを塞ぐ）。
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
        raise DimError(f"シンボル '{expr.sym}' の束縛 {bound!r} が非負整数でない")
    size = expr.coeff * bound + expr.offset
    if size > MAX_SAFE_INT:
        raise DimError(f"次元 {format_dim(expr)} の評価結果が安全整数を超える")
    return size
