"""Identity check of dimension lengths — the one place that matches symbolic dimensions
(`SymInt`) without guards.

MUST: do not compare SymInts with `==` (the same goes for comparing whole shape lists with `==`) —
torch decides the truth value by the **hint value**, so ① two different symbols with the same hint
return `True`, and ② the evaluation itself adds a guard and mixes an after-the-fact specialisation
into the exported constraints (observed on torch 2.13: `[s97] == [s52]` returns `True`, and `s97`
is replaced by `s52` from then on). sympy expressions can be compared by structural equality, so
they are matched as symbols. A pair that **cannot be proven** identical counts as "different" (an
error on the side that does not widen the accepted range).

MUST: for the same reason, do not apply `int()` to a dimension that contains a symbol either (it
silently specialises to a concrete value). That is why this module converts to int only after
looking at `free_symbols`.

It does not live in `karume.dims` because that module is the **grammar of dimension spellings**,
the counterpart of `format/dims.ts` on the TS side, and imports nothing from torch (`karume verify`
/ `karume dist` start without reading any of the heavy imports of the export scripts — the MUST in
`cli.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def extent_key(dim: Any) -> Any:
    """The key used for the identity check of a dimension (a symbol is the expression itself, a
    concrete value is an int)."""
    if isinstance(dim, torch.SymInt):
        expr = dim.node.expr
        return expr if expr.free_symbols else int(dim)
    return int(dim)


def extent_keys(dims: Sequence[Any]) -> tuple[Any, ...]:
    """A shape (or part of one) to a tuple of keys. `==` between the tuples is the shape identity
    check."""
    return tuple(extent_key(dim) for dim in dims)


def same_extents(out: Any, src: Any) -> bool:
    """Whether two metas have the same shape (symbolic dimensions are matched by the structure of
    the expression).

    A meta that carries no tensor (SymInt / None) is "different" — something without a shape cannot
    claim identity.
    """
    if not isinstance(out, torch.Tensor) or not isinstance(src, torch.Tensor):
        return False
    if out.dim() != src.dim():
        return False
    return extent_keys(out.shape) == extent_keys(src.shape)
