"""Shared helpers that export → normalize → convert synthetic modules.

No real model is used — each case only exports an nn.Module of a few dozen lines on the CPU, so
it needs neither a download nor a GPU and runs in a few seconds.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.export import Dim

from karume.convert import PRESERVED_OP_PREFIXES, convert, curated_decompositions
from karume.ir import IrGraph
from karume.normalize import normalize_graph

#: The repository root (tools/exporter/tests → tools/exporter → tools → repo).
REPO_ROOT = Path(__file__).resolve().parents[3]
#: The runtime package's test fixtures (the conformance case tables are shared with the TS
#: implementation).
RUNTIME_FIXTURES = REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures"
#: The conformance case table of the dimension grammar. The TS and Python implementations are
#: checked against **this one file** (a copy would put the sync back on manual discipline —
#: docs/ir-v2.md).
DIM_GRAMMAR_PATH = RUNTIME_FIXTURES / "dim-grammar.json"
DIM_GRAMMAR = json.loads(DIM_GRAMMAR_PATH.read_text(encoding="utf-8"))

#: The conformance case table of the op contracts. Likewise one file shared with the TS side
#: (packages/runtime/tests/ops_conformance_test.ts); both implementations **check their own
#: contract table against this table** (moving only one side turns both red).
OP_CONTRACT_TABLE_PATH = RUNTIME_FIXTURES / "op-contracts.json"
OP_CONTRACT_TABLE = json.loads(OP_CONTRACT_TABLE_PATH.read_text(encoding="utf-8"))

#: The upper bound of the dynamic axis in the synthetic tests.
SYM_MAX = 16


def export_and_convert(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """The same steps as `pipeline.export_module` (a mirror, down to the `preserved` parameter)."""
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    decomposed = ep.run_decompositions(curated_decompositions(preserved))
    normalize_graph(decomposed)
    return convert(decomposed, symbol_names=symbol_names)


@pytest.fixture(scope="session")
def convert_module() -> Callable[..., tuple[IrGraph, dict[str, torch.Tensor]]]:
    return export_and_convert


@pytest.fixture
def dyn_t() -> Dim:
    """A dynamic axis (min=2 avoids the 0/1 specialization). The IR-side name does not reach
    torch."""
    return Dim("T", min=2, max=SYM_MAX)


def node_ops(graph: IrGraph) -> list[str]:
    return [node.op for node in graph.nodes]


def only_node(graph: IrGraph, op: str):
    matched = [node for node in graph.nodes if node.op == op]
    assert len(matched) == 1, f"expected one {op} node, got {len(matched)}: {graph.nodes}"
    return matched[0]
