"""Karume exporter — torch.export 済みモデルを IR v1（docs/ir-v1.md）へ落とす。

公開面は `__all__` の 11 件が正本（列挙とここの説明は 1:1 — 機械門は
`tests/test_architecture_boundary.py` の TestThePublicSurface）:
IR の型と metadata キー（`IR_METADATA_KEY` / `IrGraph`）・変換（`convert` /
`curated_decompositions` / `normalize_graph`）・書き出し（`write_model`）・
検証（`parse_ir_graph` / `verify_model`）・一本道（`export_module` / `export_to_file` /
`publish_model`）。

配布形を作る経路は `publish_model` の 1 本を通す（書き出し → 検証 → 据え替えの 3 段を
呼び手が綴り直さないための原語 — `pipeline.publish_model` の docstring）。
"""

from karume.convert import convert, curated_decompositions
from karume.emit import write_model
from karume.ir import IR_METADATA_KEY, IrGraph
from karume.normalize import normalize_graph
from karume.pipeline import export_module, export_to_file, publish_model
from karume.verify import parse_ir_graph, verify_model

__all__ = [
    "IR_METADATA_KEY",
    "IrGraph",
    "convert",
    "curated_decompositions",
    "export_module",
    "export_to_file",
    "normalize_graph",
    "parse_ir_graph",
    "publish_model",
    "verify_model",
    "write_model",
]
