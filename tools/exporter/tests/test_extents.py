"""Identity of symbolic dimensions (`karume.extents`) — pins only the difference from a plain `==`.

That the plain comparison answers by the **hint value** is torch implementation behavior, so the
material is SymInts from an actual export (a test that lines up constants would stay green even if
the fixed path went back to `==`). A guard rewrites the shape env, so a case that hits `==`
**exports again every time**.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.export import Dim

from karume.extents import extent_key, extent_keys, same_extents


class TwoInputs(nn.Module):
    """Two inputs whose leading dimension alone is dynamic (no shape comparison, so at export time
    they are two unrelated symbols)."""

    def forward(self, a, b):
        return a.sum() + b.sum()


def _two_symbols(hint: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Takes two metas whose leading dimensions are **different symbols** with **the same hint
    value**."""
    exported = torch.export.export(
        TwoInputs(),
        (torch.zeros(hint, 3), torch.zeros(hint, 5)),
        dynamic_shapes={"a": {0: Dim("A", min=2, max=64)}, "b": {0: Dim("B", min=2, max=64)}},
    )
    first, second = (node for node in exported.graph.nodes if node.op == "placeholder")
    return first.meta["val"], second.meta["val"]


class TestSymbolicExtents:
    def test_two_symbols_with_the_same_hint_are_not_the_same_extent(self) -> None:
        first, second = _two_symbols()
        assert extent_key(first.shape[0]) != extent_key(second.shape[0])
        assert extent_keys(first.shape[:1]) != extent_keys(second.shape[:1])

    def test_the_plain_comparison_this_replaces_answers_by_the_hint(self) -> None:
        """A plain `==` returns `True` even for different symbols (= the regression detector of the
        previous test).

        The answer is not the only thing observable — `==` adds an equality guard to the shape env,
        so **after hitting it** the two symbols collapse into the same expression. This rewrite is
        what "an after-the-fact specialization mixed into the exported constraints" means.
        """
        first, second = _two_symbols()
        assert [first.shape[0]] == [second.shape[0]]
        assert extent_key(first.shape[0]) == extent_key(second.shape[0])

    def test_a_symbol_matches_itself(self) -> None:
        first, _ = _two_symbols()
        assert extent_keys(first.shape) == extent_keys(first.shape)
        assert same_extents(first, first)

    def test_a_static_extent_is_keyed_by_its_value(self) -> None:
        first, second = _two_symbols()
        assert extent_key(first.shape[1]) == 3
        assert extent_key(second.shape[1]) == 5

    def test_shapes_that_share_a_hint_are_not_the_same_shape(self) -> None:
        """Even with the same rank and the same hint values, `same_extents` is False when the
        symbols differ."""
        first, second = _two_symbols()
        assert not same_extents(first, second[:, :3])

    def test_a_meta_without_a_shape_is_never_the_same(self) -> None:
        first, _ = _two_symbols()
        assert not same_extents(first, None)
        assert not same_extents(None, first)
