"""IR v1（docs/ir-v1.md）のグラフ表現と JSON 直列化。

TS 側の型は packages/runtime/src/format/ir.ts。ここは「エクスポータが組み立てる側」の器で、
規則の検査は verify.py が受け持つ（組み立てと検査を同じ関数に混ぜると、
検査を通らない中間状態を作れなくなってテストが書けなくなる）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

IR_FORMAT = "karume-ir"
IR_VERSION = 1
#: グラフ JSON を載せる safetensors `__metadata__` のキー。
IR_METADATA_KEY = "karume_ir"

#: 非負整数、または `coeff·sym+offset` の正準表記（dims.py）。
IrDim = int | str


@dataclass(frozen=True)
class IrStorage:
    """格納 dtype。M0 ランタイムが実行できるのは f32 のみ。"""

    dtype: str
    #: 量子化格納の scale テンソルの safetensors キー（storage.dtype == "i8" のみ）。
    scale: str | None = None
    group_size: int | None = None

    def to_dict(self) -> dict:
        out: dict = {"dtype": self.dtype}
        if self.scale is not None:
            out["scale"] = self.scale
        if self.group_size is not None:
            out["group_size"] = self.group_size
        return out


@dataclass(frozen=True)
class IrInitializer:
    #: safetensors のテンソルキー。
    tensor: str
    storage: IrStorage

    def to_dict(self) -> dict:
        return {"tensor": self.tensor, "storage": self.storage.to_dict()}


@dataclass(frozen=True)
class IrValue:
    dtype: str
    shape: list[IrDim]

    def to_dict(self) -> dict:
        return {"dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class IrInput:
    name: str
    dtype: str
    shape: list[IrDim]

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class IrNode:
    op: str
    ins: list[str]
    outs: list[str]
    attrs: dict

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "ins": list(self.ins),
            "outs": list(self.outs),
            "attrs": dict(self.attrs),
        }


@dataclass
class IrGraph:
    symbols: list[str] = field(default_factory=list)
    inputs: list[IrInput] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    initializers: dict[str, IrInitializer] = field(default_factory=dict)
    values: dict[str, IrValue] = field(default_factory=dict)
    nodes: list[IrNode] = field(default_factory=list)

    @property
    def required_ops(self) -> list[str]:
        """nodes で実際に使われる op 集合。

        MUST: 独立フィールドとして持たない — requires.ops と nodes の一致は IR v1 の
        検査規則（docs/ir-v1.md）であり、二重管理すると「宣言だけ更新して実体が古い」
        グラフを作れてしまう。常に nodes から導出する。
        """
        return sorted({node.op for node in self.nodes})

    def to_dict(self) -> dict:
        return {
            "format": IR_FORMAT,
            "version": IR_VERSION,
            "requires": {"ops": self.required_ops},
            "symbols": list(self.symbols),
            "inputs": [spec.to_dict() for spec in self.inputs],
            "outputs": list(self.outputs),
            "initializers": {name: init.to_dict() for name, init in self.initializers.items()},
            "values": {name: value.to_dict() for name, value in self.values.items()},
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def to_json(self) -> str:
        """グラフ JSON 文字列。

        allow_nan=False は必須（docs/ir-v1.md）— NaN / Infinity は JSON の標準リテラルに
        無く、ブラウザの JSON.parse が落ちる。受理集合をランタイム側に揃えるため、
        書き出しの時点で失敗させる。
        """
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)
