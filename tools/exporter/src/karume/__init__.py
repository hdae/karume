"""Karume exporter — torch.export 済みモデルを IR（docs/ir-v2.md）とコンテナ（`krm`）へ落とす。

公開面は `__all__` の 12 件が正本（列挙とここの説明は 1:1 — 機械門は
`tests/test_architecture_boundary.py` の TestThePublicSurface）:
IR の型と metadata キー（`IR_METADATA_KEY` / `IrGraph`）・変換（`convert` /
`curated_decompositions` / `normalize_graph`）・格納変換（`FixedQuantizedWeight` /
`stored_model`）・出所（`Provenance`）・検証（`parse_ir_graph` / `verify_container`）・
一本道（`export_module` / `export_to_file` / `publish_model`）。

配布形を作る経路は `publish_model` の 1 本を通す（格納変換 → 書く → 読み直して検証 →
据え替え — `pipeline.publish_model` と `publish.publish_container` の docstring）。
"""

from karume.container import Provenance
from karume.convert import convert, curated_decompositions
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IR_METADATA_KEY, IrGraph
from karume.normalize import normalize_graph
from karume.pipeline import export_module, export_to_file, publish_model
from karume.verify import parse_ir_graph, verify_container

# 既存の公開面テストは ASCII 昇順を要求する。RUF022 の「定数優先」と区別する。
__all__ = [  # noqa: RUF022
    "FixedQuantizedWeight",
    "IR_METADATA_KEY",
    "IrGraph",
    "Provenance",
    "convert",
    "curated_decompositions",
    "export_module",
    "export_to_file",
    "normalize_graph",
    "parse_ir_graph",
    "publish_model",
    "stored_model",
    "verify_container",
]
