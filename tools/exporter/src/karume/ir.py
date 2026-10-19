"""Graph representation of IR v1 (docs/ir-v2.md) and its JSON serialization — the exporter's
internal holder. For the distribution, `karume.container.ir_v2_document` maps it to a v2 document
and writes that.

The TS-side types are in packages/runtime/src/format/ir.ts. This is the holder on "the side the
exporter builds"; checking the rules is verify.py's job (mixing building and checking in one
function would make intermediate states that fail the checks impossible to build, and with them
the tests).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

IR_FORMAT = "karume-ir"
IR_VERSION = 1

#: A non-negative integer, or the canonical `coeff·sym+offset` notation (dims.py).
IrDim = int | str

#: The minimum group length of group-quantized storage (`i4`) (the accepted set is **a power of
#: two and at least this** — ADR 0069 decision 2, the same constraint as ORT). One constant so that
#: the writer (emit) and the verifier (verify) see the same accepted set (the TS-side source of
#: truth is `MIN_GROUP_SIZE` in `packages/runtime/src/format/container/codecs.ts`).
MIN_GROUP_SIZE = 16


@dataclass(frozen=True)
class IrStorage:
    """Storage dtype. Execution paths exist for f32 / f16 / i8 / i4 / i2 and for i32 of
    symbol-dependent constants."""

    dtype: str
    #: Container tensor key of the scale tensor of quantized storage (only for storage.dtype ==
    #: "i8" / "i4" / "i2").
    scale: str | None = None
    #: Group length of group quantization (required when storage.dtype == "i4" — ADR 0069
    #: decision 2).
    group_size: int | None = None

    def to_dict(self) -> dict:
        out: dict = {"dtype": self.dtype}
        if self.scale is not None:
            out["scale"] = self.scale
        if self.group_size is not None:
            out["group_size"] = self.group_size
        return out


@dataclass(frozen=True)
class IrShared:
    """Target of a shared initializer (ADR 0096 stage 2 — the borrowing graph holds no bytes).

    `tensor` is **the tensor key in the lending container** (the actual key in the distribution).
    The borrowing Session looks up the lending graph's initializer with this key and consumes the
    already resident weight buffer as is.
    """

    tensor: str

    def to_dict(self) -> dict:
        return {"tensor": self.tensor}


@dataclass(frozen=True)
class IrInitializer:
    """Declaration of one initializer.

    MUST: `tensor` (own bytes) and `shared` (borrows the lender's bytes) are **exclusive — exactly
    one of them**. verify rejects both / neither (the building layer has no checks — module
    docstring). The shared side writes not a single byte into the container, so it has no
    `storage.scale` (the scale belongs to the lender's resident weight).
    """

    storage: IrStorage
    #: Container tensor key (with own bytes).
    tensor: str | None = None
    #: Reference to a tensor key of the lending container (when shared).
    shared: IrShared | None = None

    @property
    def is_shared(self) -> bool:
        """Whether this is a shared declaration without bytes (the exclusion condition of the
        walks — emit / verify)."""
        return self.shared is not None

    def to_dict(self) -> dict:
        if self.shared is not None:
            return {"shared": self.shared.to_dict(), "storage": self.storage.to_dict()}
        return {"tensor": self.tensor, "storage": self.storage.to_dict()}


@dataclass(frozen=True)
class IrValue:
    dtype: str
    shape: list[IrDim]

    def to_dict(self) -> dict:
        return {"dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class IrState:
    """A named state slot (ADR 0066 decision 2).

    shape is the concrete shape including capacity (rank ≤ 4, numeric dimensions are positive
    integers). It is not a value, so it has no declaration in `values` and is not referenced from
    a node's `ins` / `outs` (the referencing field is ADR 0067's).

    `external` is a slot that **does not allocate its own storage** (ADR 0096 stage 2) — the
    borrowing context binds it to the lender's slot of the same name. It cannot have a
    `state_append`; its only readers are readonly attentions.
    """

    dtype: str
    shape: list[IrDim]
    external: bool = False

    def to_dict(self) -> dict:
        # MUST: `external: false` is **not written** (the absence of the field is itself the
        # declaration of "own slot" — always writing it would change the graph JSON of existing
        # models byte for byte and move every sha256 regression check of the distributions).
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            **({"external": True} if self.external else {}),
        }


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
    #: Named references to state slots (ADR 0067 decision 4 — **a separate field** from `ins` /
    #: `outs`). The keys are fixed words set by the op contract (`k` / `v` of `attention`, `slot`
    #: of `state_append`) and the values are slot names declared in `graph.states`. A node
    #: without the field has an empty table.
    states: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "ins": list(self.ins),
            "outs": list(self.outs),
            "attrs": dict(self.attrs),
            # An empty states is not written — matches the rule of v2's canonical serialization
            # (`container.ir_v2_document`, docs/ir-v2.md "Canonical serialization") that does not
            # write empty fields. The v1 JSON is not part of the distribution.
            **({"states": dict(self.states)} if self.states else {}),
        }


@dataclass
class IrGraph:
    symbols: list[str] = field(default_factory=list)
    inputs: list[IrInput] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    initializers: dict[str, IrInitializer] = field(default_factory=dict)
    values: dict[str, IrValue] = field(default_factory=dict)
    #: State slot declarations (ADR 0066 decision 2). They are produced by the post-export
    #: rewrite `karume.states.to_states_form` (ADR 0067) — `convert` emits the attention form as
    #: is.
    states: dict[str, IrState] = field(default_factory=dict)
    nodes: list[IrNode] = field(default_factory=list)

    @property
    def required_ops(self) -> list[str]:
        """The set of ops actually used by nodes.

        MUST: not kept as an independent field — agreement of requires.ops with nodes is a check
        rule of IR v1 (docs/ir-v2.md), and keeping it twice would allow a graph whose
        "declaration was updated but whose content is stale". Always derived from nodes.
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
            # An empty states is not written — matches the rule of v2's canonical serialization
            # (`container.ir_v2_document`, docs/ir-v2.md "Canonical serialization") that does not
            # write empty fields. The v1 JSON is not part of the distribution.
            **(
                {"states": {name: slot.to_dict() for name, slot in self.states.items()}}
                if self.states
                else {}
            ),
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def to_json(self) -> str:
        """The graph JSON string.

        allow_nan=False is required (docs/ir-v2.md) — NaN / Infinity are not standard JSON
        literals and the browser's JSON.parse fails on them. To match the accepted set to the
        runtime side, writing fails at this point.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)
