"""Karume exporter — torch.export 済みモデルを IR v1（docs/ir-v1.md）へ落とす。

公開面はこの 4 つ: 変換（convert）・書き出し（write_model）・検証（verify_model）・
一本道（export_module / export_to_file）。
"""

from karume.convert import convert, curated_decompositions
from karume.emit import write_model
from karume.ir import IR_METADATA_KEY, IrGraph
from karume.normalize import normalize_graph
from karume.pipeline import export_module, export_to_file
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
    "verify_model",
    "write_model",
]
