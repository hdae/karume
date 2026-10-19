"""Dimension language: the linear expression `coeff·sym+offset` for a symbolic dimension of a shape
(one symbol per dimension).

The source of truth for the grammar is not the implementation but the conformance table
`packages/runtime/tests/fixtures/dim-grammar.json` (from the repository root); the TS implementation
(packages/runtime/src/format/dims.ts) and this module are checked against the same table. Moving
only one side creates dimension spellings that "the exporter can write but the runtime cannot
read", so when the grammar changes, move the table, TS and Python together.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

# Numbers on the accepting side (TS) are IEEE754 doubles, so a dimension above this changes value
# in a round trip.
MAX_SAFE_INT = 2**53 - 1

_SYMBOL_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_SYMBOL_RE = re.compile(_SYMBOL_PATTERN)
# Digits are ASCII only (`\d` also matches Unicode digits — Arabic-Indic, full-width — and the
# accepted set splits from TS's `\d`. `int()` converts Unicode digits too, so the round trip with
# format would break as well).
_DIM_RE = re.compile(rf"([0-9]+)?({_SYMBOL_PATTERN})(?:\+([0-9]+))?")


class DimError(ValueError):
    """A dimension expression does not match the canonical grammar / cannot be bound."""


@dataclass(frozen=True)
class DimExpr:
    """`coeff·sym + offset`. Only non-negative linear expressions (coeff >= 1, offset >= 0)."""

    coeff: int
    sym: str
    offset: int


def is_symbol_name(name: str) -> bool:
    return _SYMBOL_RE.fullmatch(name) is not None


def _parse_part(text: str | None, minimum: int, omitted: int) -> int | None:
    """Rejects explicit spellings of the omissible parts (coeff=1 / offset=0) and leading zeros as
    non-canonical.

    Allowing two or more spellings for the same dimension would split the binding table and the
    shape identity check over spelling variants.
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
    """Decomposes a canonical spelling; non-canonical or non-matching text gives None (the variant
    that both checks and decomposes)."""
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
        raise DimError(
            f"Dimension expression '{text}' does not match the canonical grammar coeff·sym+offset"
        )
    return expr


def format_dim(expr: DimExpr) -> str:
    """Back to the canonical spelling. A round trip with parse_dim is string-identical (pinned by
    the conformance table)."""
    if not is_symbol_name(expr.sym):
        raise DimError(f"Invalid symbol name '{expr.sym}'")
    if expr.coeff < 1 or expr.coeff > MAX_SAFE_INT:
        raise DimError(f"Invalid coefficient {expr.coeff}")
    if expr.offset < 0 or expr.offset > MAX_SAFE_INT:
        raise DimError(f"Invalid offset {expr.offset}")
    head = "" if expr.coeff == 1 else str(expr.coeff)
    tail = "" if expr.offset == 0 else f"+{expr.offset}"
    return f"{head}{expr.sym}{tail}"


def eval_dim(expr: DimExpr, bindings: Mapping[str, int]) -> int:
    """Applies the binding table to get a concrete dimension. An unbound symbol or a non-integer
    binding fails instead of going on with a default."""
    if expr.sym not in bindings:
        raise DimError(f"Symbol '{expr.sym}' is not bound")
    bound = bindings[expr.sym]
    # bool derives from int but is not a dimension (this stops True from silently passing as 1).
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
        raise DimError(
            f"Symbol '{expr.sym}' is bound to {bound!r}, which is not a non-negative integer"
        )
    size = expr.coeff * bound + expr.offset
    if size > MAX_SAFE_INT:
        raise DimError(f"Value of dimension {format_dim(expr)} exceeds the safe integer range")
    return size
