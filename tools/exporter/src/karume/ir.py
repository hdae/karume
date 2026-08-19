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

#: group 量子化格納（`i4`）の group 長の下限（受理集合は **2 冪かつこれ以上** — ADR 0069
#: 決定 2・ORT と同制約）。ライタ（emit）と検証（verify）が同じ受理集合を見るための 1 本
#: （TS 側の正本は `packages/runtime/src/format/ir.ts`）。
MIN_GROUP_SIZE = 16


@dataclass(frozen=True)
class IrStorage:
    """格納 dtype。実行経路があるのは f32 / f16 / i8 / i4 と、記号依存定数の i32。"""

    dtype: str
    #: 量子化格納の scale テンソルの safetensors キー（storage.dtype == "i8" / "i4" のみ）。
    scale: str | None = None
    #: group 量子化の group 長（storage.dtype == "i4" では必須 — ADR 0069 決定 2）。
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
class IrState:
    """名前付き state スロット（ADR 0066 決定 2）。

    shape は容量込みの具体形（rank ≤ 4・数値次元は正整数）。値ではないので `values` に宣言を
    持たず、ノードの `ins` / `outs` からも参照されない（参照の欄は ADR 0067 の担当）。
    """

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
    #: state スロットの名前参照（ADR 0067 決定 4 — `ins` / `outs` と**別の欄**）。キーは op 契約が
    #: 固定する固定語（`attention` の `k` / `v`・`state_append` の `slot`）で、値は
    #: `graph.states` で宣言済みのスロット名。欄を持たないノードは空表。
    states: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "ins": list(self.ins),
            "outs": list(self.outs),
            "attrs": dict(self.attrs),
            # MUST: 空の states は**書かない**（IrGraph.to_dict の states 節と同じ理由）。
            # 常に出すと states を 1 本も持たない既存モデルのグラフ JSON がバイト単位で変わり、
            # 配布物の sha 門が全部動く。
            **({"states": dict(self.states)} if self.states else {}),
        }


@dataclass
class IrGraph:
    symbols: list[str] = field(default_factory=list)
    inputs: list[IrInput] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    initializers: dict[str, IrInitializer] = field(default_factory=dict)
    values: dict[str, IrValue] = field(default_factory=dict)
    #: state スロット宣言（ADR 0066 決定 2）。これを出すのは export 後の手術
    #: `karume.states.to_states_form`（ADR 0067）— `convert` は attention 形のまま出す。
    states: dict[str, IrState] = field(default_factory=dict)
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
            # MUST: 空の states は**書かない**。常に出すと states を 1 本も持たない既存モデルの
            # グラフ JSON がバイト単位で変わり、配布物の sha 門が全部動く（ADR 0066 決定 2 の
            # 「states を出す最初のモデルまで無風」）。
            **(
                {"states": {name: slot.to_dict() for name, slot in self.states.items()}}
                if self.states
                else {}
            ),
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def to_json(self) -> str:
        """グラフ JSON 文字列。

        allow_nan=False は必須（docs/ir-v1.md）— NaN / Infinity は JSON の標準リテラルに
        無く、ブラウザの JSON.parse が落ちる。受理集合をランタイム側に揃えるため、
        書き出しの時点で失敗させる。
        """
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)
