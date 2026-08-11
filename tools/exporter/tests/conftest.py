"""合成モジュールを export→normalize→convert する共通ヘルパ。

実モデルは使わない — 数十行の nn.Module を CPU で export するだけなので、
ダウンロードも GPU も要らず各ケースが数秒で回る。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.export import Dim

from karume.convert import PRESERVED_OP_PREFIXES, convert, curated_decompositions
from karume.ir import IrGraph
from karume.normalize import normalize_graph

#: リポジトリ直下（tools/exporter/tests → tools/exporter → tools → repo）。
REPO_ROOT = Path(__file__).resolve().parents[3]
#: ランタイムパッケージのテスト資産（適合ケース表は TS 実装と共有する）。
RUNTIME_FIXTURES = REPO_ROOT / "packages" / "runtime" / "tests" / "fixtures"
#: 次元文法の適合ケース表。TS 実装と Python 実装は**この 1 ファイル**で検証する
#: （コピーを作ると同期が人手の規律に戻る — docs/ir-v1.md）。
DIM_GRAMMAR_PATH = RUNTIME_FIXTURES / "dim-grammar.json"
DIM_GRAMMAR = json.loads(DIM_GRAMMAR_PATH.read_text(encoding="utf-8"))

#: op 契約の適合ケース表。同じく TS 側（packages/runtime/tests/ops_conformance_test.ts）と
#: 共有する 1 ファイルで、両実装が**自分の契約表をこの表へ突き合わせる**
#: （片側だけ動かすと両方赤になる）。
OP_CONTRACT_TABLE_PATH = RUNTIME_FIXTURES / "op-contracts.json"
OP_CONTRACT_TABLE = json.loads(OP_CONTRACT_TABLE_PATH.read_text(encoding="utf-8"))

#: Irodori のモデル実装 clone（`export_irodori.py` の `--source-dir` 既定）。**在れば**
#: import 可能にする — `patch_irodori` の Irodori 側パッチは `irodori_tts` を差し替えるので、
#: 同値テストにはこの clone が要る。git 追跡外なので無い環境ではテスト側が skip する
#: （`pytest.importorskip("irodori_tts")`）。
IRODORI_SOURCE_DIR = REPO_ROOT / "inputs" / "irodori" / "Irodori-TTS"
if IRODORI_SOURCE_DIR.is_dir() and str(IRODORI_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(IRODORI_SOURCE_DIR))

#: 合成テストの動的軸の上限。
SYM_MAX = 16


def export_and_convert(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """`pipeline.export_module` と同じ段（`preserved` の受け口まで含めて鏡像）。"""
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    decomposed = ep.run_decompositions(curated_decompositions(preserved))
    normalize_graph(decomposed)
    return convert(decomposed, symbol_names=symbol_names)


@pytest.fixture(scope="session")
def convert_module() -> Callable[..., tuple[IrGraph, dict[str, torch.Tensor]]]:
    return export_and_convert


@pytest.fixture
def dyn_t() -> Dim:
    """動的軸（min=2 で 0/1 特殊化を避ける）。IR 側の名前は torch に伝わらない。"""
    return Dim("T", min=2, max=SYM_MAX)


def node_ops(graph: IrGraph) -> list[str]:
    return [node.op for node in graph.nodes]


def only_node(graph: IrGraph, op: str):
    matched = [node for node in graph.nodes if node.op == op]
    assert len(matched) == 1, f"{op} ノードが {len(matched)} 件: {graph.nodes}"
    return matched[0]
