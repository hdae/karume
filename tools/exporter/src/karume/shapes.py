"""The layer that keeps the output shape rules on the exporter side too
(equivalent to `computeOutputShape` in packages/runtime/src/ops/shapes.ts on the TS side).

Its purpose is to remove "export is green, and the first failure is at Session construction". It
does not take the declared shapes that torch's meta attached at face value: it matches them, on
every node, against **shapes computed independently from the rules of the contract table**, and a
mismatch fails loudly at export time (the diagnostic discipline of ADR 0005).

The only difference from the TS side is the domain:

- TS handles **numeric shapes after binding resolution** (concrete values are always settled
  before execution — ADR 0004).
- This module handles **declared shapes before binding**, so a dimension is an integer or the
  dimension language `coeff·sym+offset` (dims.py). For integer-only inputs it reaches the same
  conclusion as TS one-to-one (the `shapes` section of the conformance table
  packages/runtime/tests/fixtures/op-contracts.json matches both sides).

MUST: decisions that need bindings are not made here — rather than "silently loosening for symbolic
dimensions", the split is that **rules settled only after binding belong to the runtime-side
layer**. There are only two:

1. No reduction over an axis of length 0 (amax / amin / softmax) — whether a symbolic dimension
   becomes 0 depends on the binding.
2. `sym_prefix_slice` exceeding Tmax (prefix length ≤ constant dimension) — the upper bound is not
   in the IR.

The TS side (plan.ts → computeOutputShape) always checks both before execution, so the layers
leave no gap.

There is one split in the opposite direction: **the symbol rules for the target axis of layout
group 2** — `slice` / `flip` take "static axes only" (ADR 0014), `cat` takes "up to a linear
sum of the same symbol" (ADR 0046) — can be decided only here (the layer that looks at declared
shapes). The TS side handles only numeric shapes after binding resolution, so the same rules are
held by `validateGraphContracts` in plan.ts (the layer that looks at declared shapes) — a different
layer, but the accepted set is the same on both sides.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from karume.dims import MAX_SAFE_INT, DimError, DimExpr, format_dim, parse_dim
from karume.ir import IrDim, IrGraph
from karume.ops import (
    OpContract,
    OpContractError,
    arity_fits,
    assert_strided_rank,
    attention_readonly,
    attention_scale,
    axis_dim,
    conv1d_attrs,
    conv2d_attrs,
    conv3d_attrs,
    conv_transpose1d_attrs,
    cumsum_dim,
    deform_conv2d_attrs,
    describe_arity,
    layer_norm_attrs,
    pad_attrs,
    permute_dims,
    reduce_dim,
    resolve_op_contract,
    rms_norm_eps,
    scalar_param_values,
    slice_attrs,
    softmax_dim,
    state_window,
    static_quantize_scale,
    sym_prefix_slice_attrs,
    topk_k,
    upsample_bilinear2d_attrs,
)


@dataclass(frozen=True)
class Extent:
    """The length of one dimension. A constant has `sym=None` (coeff 0); a symbolic dimension is
    `coeff·sym+offset`.

    Equality is structural (the dataclass default) — the only way to decide "same length" while
    staying symbolic, and it does not confuse `T` with `2T`.
    """

    coeff: int
    sym: str | None
    offset: int

    @property
    def is_const(self) -> bool:
        return self.sym is None

    def is_value(self, value: int) -> bool:
        """Whether this is a constant dimension of exactly `value` (always False for a symbolic
        dimension — it depends on the binding, so nothing is asserted)."""
        return self.sym is None and self.offset == value

    def to_dim(self) -> IrDim:
        if self.sym is None:
            return self.offset
        return format_dim(DimExpr(coeff=self.coeff, sym=self.sym, offset=self.offset))


_ONE = Extent(coeff=0, sym=None, offset=1)


def _extent(dim: Any, where: str) -> Extent:
    # bool derives from int but is not a dimension (this stops True from silently passing as 1).
    if isinstance(dim, bool) or not isinstance(dim, (int, str)):
        raise OpContractError(
            f"{where}: dimension {dim!r} is neither an integer nor a dimension expression"
        )
    if isinstance(dim, int):
        if not 0 <= dim <= MAX_SAFE_INT:
            raise OpContractError(f"{where}: dimension {dim} is not a non-negative safe integer")
        return Extent(coeff=0, sym=None, offset=dim)
    try:
        expr = parse_dim(dim)
    except DimError as cause:
        raise OpContractError(f"{where}: {cause}") from cause
    return Extent(coeff=expr.coeff, sym=expr.sym, offset=expr.offset)


def extents(shape: Sequence[IrDim], where: str) -> list[Extent]:
    """Lowers a declared shape to a list of lengths (comparison and computation happen only on
    this representation)."""
    return [_extent(dim, f"{where} dimension {index}") for index, dim in enumerate(shape)]


def _show(shape: Sequence[Extent]) -> str:
    return ",".join(str(extent.to_dim()) for extent in shape)


def _at(shape: Sequence[Extent], index: int) -> Extent:
    """A right-aligned broadcast read (out of range is length 1).

    MUST: do not leave a negative index to Python's wrap-around — it would pick from the end, and
    the broadcast of operands of different ranks would silently align with a different axis.
    """
    return shape[index] if index >= 0 else _ONE


def broadcast_extents(a: Sequence[Extent], b: Sequence[Extent], where: str) -> list[Extent]:
    """Right-aligned broadcast as in torch (equivalent to `broadcastShapes` in
    packages/runtime/src/ops/shapes.ts).

    MUST: do not decide the result with max — the pair 0 and 1 (max gives 1) is 0 in torch.
    """
    rank = max(len(a), len(b))
    out: list[Extent] = []
    for index in range(rank):
        da = _at(a, len(a) - rank + index)
        db = _at(b, len(b) - rank + index)
        if da == db or db.is_value(1):
            out.append(da)
        elif da.is_value(1):
            out.append(db)
        elif da.is_const and db.is_const:
            raise OpContractError(
                f"{where}: shapes [{_show(a)}] and [{_show(b)}] cannot be broadcast right-aligned"
            )
        else:
            # Two symbols (or a symbol and a constant) that do not match. It may hold depending
            # on the binding, but the declaration alone cannot decide — letting it through silently
            # would let the exporter write "IR you cannot judge without running it".
            raise OpContractError(
                f"{where}: shapes [{_show(a)}] and [{_show(b)}] dimension {index}"
                f" ({da.to_dim()} and {db.to_dim()}): the declaration alone cannot decide whether"
                " they broadcast"
            )
    return out


def _numel_key(shape: Sequence[Extent]) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    """The key used for the identity check of element counts (the element-count check of
    reshape).

    An element count is a product of linear expressions, so comparing it symbolically needs a
    factorisation. Each dimension is split into "content (gcd of coefficient and offset) × primitive
    linear expression"; the content goes to the integer side and the primitive linear expression
    into a multiset — a linear expression is irreducible in the multivariate polynomial ring (a
    UFD), so this pair matching is equivalent to equality as polynomials (`2T+2` and `2·(T+1)` get
    the same key).

    MUST: when the integer side is 0, collapse the key to 0 too (aligns the conclusion with
    `0 × anything = 0` on numeric shapes).
    """
    scalar = 1
    factors: list[tuple[str, int, int]] = []
    for extent in shape:
        if extent.sym is None:
            scalar *= extent.offset
            continue
        content = math.gcd(extent.coeff, extent.offset)
        scalar *= content
        factors.append((extent.sym, extent.coeff // content, extent.offset // content))
    if scalar == 0:
        return (0, ())
    return (scalar, tuple(sorted(factors)))


def _require_declared(
    declared: Sequence[IrDim] | None, contract: OpContract, where: str
) -> list[Extent]:
    if declared is None:
        raise OpContractError(
            f"{where}: op '{contract.name}' output shape is the declared target shape"
            " (declared is required)"
        )
    return extents(declared, f"{where} declared shape")


@dataclass(frozen=True)
class _StateContext:
    """The node's `states` field and the declared shapes of the slots it points at (ADR 0067
    decision 4).

    The TS side carries the same thing in two fields of ShapeContext (`states` / `stateShapes`).
    """

    states: Mapping[str, str]
    shapes: Mapping[str, Sequence[IrDim]]

    @property
    def referenced(self) -> bool:
        """Whether this is the states form (the presence of the field decides the form — ADR 0067
        decision 4)."""
        return bool(self.states)

    def slot(self, key: str, where: str) -> tuple[str, list[Extent]]:
        """(name, declared shape) of the slot a key of the `states` field points at.

        MUST: a path without resolved shapes fails loudly. The slots are allocated by the
        GenerationContext (ADR 0066 decision 1), so a path without it = one-shot execution cannot
        run state-referencing nodes.
        """
        name = self.states.get(key)
        if name is None:
            raise OpContractError(f"{where}: states field has no '{key}'")
        shape = self.shapes.get(name)
        if shape is None:
            raise OpContractError(
                f"{where}: state slot '{name}' shape is not resolved"
                " (running a node that references state requires a GenerationContext"
                " — ADR 0066 decision 1)"
            )
        return name, extents(shape, f"{where} state slot '{name}'")


def compute_output_shape(
    contract: OpContract,
    input_shapes: Sequence[Sequence[IrDim]],
    where: str,
    *,
    declared: Sequence[IrDim] | None = None,
    attrs: Mapping[str, Any] | None = None,
    states: Mapping[str, str] | None = None,
    state_shapes: Mapping[str, Sequence[IrDim]] | None = None,
) -> list[list[IrDim]]:
    """Computes the **list of shapes in output slot order** from the declared shapes
    (equivalent to `computeOutputShape` in packages/runtime/src/ops/shapes.ts — ADR 0068
    decision 1).

    Only `topk` returns 2 (values + indices — ADR 0068 decision 3), only `state_append` returns
    **an empty list** (an effect op that defines no value — ADR 0067 decision 5), and everything
    else returns 1. `declared` is required only for ops whose "output's declared shape is the
    target shape" (reshape / expand — ADR 0011). `attrs` is required for permute / layer_norm /
    softmax / conv1d / sym_prefix_slice / topk.

    `states` is the node's `states` field (**being non-empty is itself what identifies "the states
    form"** — ADR 0067 decision 4), and `state_shapes` are the slot shapes from `graph.states`
    (concrete shapes including the capacity). The two MUST be passed together — a
    state-referencing node without resolved slot shapes fails loudly; computing it silently as the
    previous form would be "a different computation that does not read the past KV".
    """
    node_attrs: Mapping[str, Any] = attrs if attrs is not None else {}
    # A readonly-form attention (ADR 0096 stage 2) has exactly one input, q — the input-count
    # check is in the arm (the contract table's arity is the previous form's 3 to 4, and applying
    # it here would reject correct graphs).
    readonly = contract.kind == "attention" and bool(states) and attention_readonly(node_attrs)
    if not readonly and not arity_fits(contract, len(input_shapes)):
        raise OpContractError(
            f"{where}: op '{contract.name}' has {len(input_shapes)} input shapes"
            f" (the contract is {describe_arity(contract)})"
        )
    ins = [extents(shape, f"{where} input {index}") for index, shape in enumerate(input_shapes)]
    slots = _StateContext(states or {}, state_shapes or {})
    return [
        [extent.to_dim() for extent in slot]
        for slot in _compute(contract, ins, where, declared, node_attrs, slots)
    ]


def _sole(shape: list[Extent]) -> list[list[Extent]]:
    """The arm of a single-output op (each arm states explicitly that there is one output — ADR
    0068 decision 1)."""
    return [shape]


def _compute(
    contract: OpContract,
    ins: list[list[Extent]],
    where: str,
    declared: Sequence[IrDim] | None,
    attrs: Mapping[str, Any],
    slots: _StateContext,
) -> list[list[Extent]]:
    kind = contract.kind
    if kind == "unary":
        # MUST: check the ranges of the scalar attrs and **invariants that span keys** (clamp's
        # min <= max) here. This computation is the only common path every node goes through, and
        # the attrs schema can express only per-key checks (assert_graph_shapes calls this for
        # every node).
        scalar_param_values(contract, attrs, where)
        return _sole(list(ins[0]))
    if kind == "cast":
        return _sole(list(ins[0]))
    if kind == "binary":
        return _sole(broadcast_extents(ins[0], ins[1], f"{where} ({contract.name})"))
    if kind == "where":
        # As in torch, the three broadcast right-aligned (the condition widens by the same rule
        # as the values).
        label = f"{where} ({contract.name})"
        return _sole(broadcast_extents(broadcast_extents(ins[0], ins[1], label), ins[2], label))
    if kind == "cumsum":
        return _sole(_cumsum(ins, where, attrs))
    if kind == "matmul":
        return _sole(_matmul(ins, where))
    if kind == "bmm":
        return _sole(_bmm(ins, where))
    if kind == "gather":
        return _sole(_gather(ins, where))
    if kind == "row_reduce":
        return _sole(_row_reduce(contract, ins, where, attrs))
    if kind == "argmax":
        return _sole(_argmax(ins, where))
    # The only multi-output arm (values + indices — ADR 0068 decision 3). Not going through _sole
    # is itself the declaration "two outputs".
    if kind == "topk":
        return _topk(ins, where, attrs)
    if kind == "reshape":
        return _sole(_reshape(contract, ins, where, declared))
    if kind == "permute":
        return _sole(_permute(ins, where, attrs))
    if kind == "expand":
        return _sole(_expand(contract, ins, where, declared))
    if kind == "slice":
        return _sole(_slice(ins, where, attrs))
    if kind == "cat":
        return _sole(_cat(ins, where, attrs))
    if kind == "pad":
        return _sole(_pad(ins, where, attrs))
    if kind == "flip":
        return _sole(_flip(ins, where, attrs))
    if kind == "sym_prefix_slice":
        return _sole(_sym_prefix_slice(ins, where, attrs))
    if kind == "linear":
        return _sole(_linear(ins, where))
    if kind == "layer_norm":
        return _sole(_layer_norm(ins, where, attrs))
    if kind == "static_quantize":
        static_quantize_scale(attrs, where)
        return _sole(list(ins[0]))
    if kind == "rms_norm":
        return _sole(_rms_norm(ins, where, attrs))
    # safe_softmax has the same shape rule and attrs as softmax (only the empty-row values differ
    # — ADR 0044).
    if kind in ("softmax", "safe_softmax"):
        return _sole(_softmax(ins, where, attrs, contract.name))
    if kind == "attention":
        return _sole(_attention(ins, where, attrs, slots))
    # An effect op that defines no value (ADR 0067 decision 5). Returning **an empty list**
    # without going through `_sole` is itself the declaration "0 outputs" (the same style as the
    # multi-output arm building its list).
    if kind == "state_append":
        return _state_append(ins, where, attrs, slots)
    if kind == "embedding":
        return _sole(_embedding(ins, where))
    if kind == "masked_fill":
        return _sole(_masked_fill(ins, where))
    if kind == "conv1d":
        return _sole(_conv1d(ins, where, attrs))
    if kind == "conv2d":
        return _sole(_conv2d(ins, where, attrs))
    if kind == "conv3d":
        return _sole(_conv3d(ins, where, attrs))
    if kind == "conv_transpose1d":
        return _sole(_conv_transpose1d(ins, where, attrs))
    if kind == "deform_conv2d":
        return _sole(_deform_conv2d(ins, where, attrs))
    if kind == "upsample_bilinear2d":
        return _sole(_upsample_bilinear2d(ins, where, attrs))
    # The op name carries the scan direction, but the shape rule is identical for both
    # directions (the time order of the output does not change either).
    if kind == "gru_scan":
        return _sole(_gru_scan(contract, ins, where))
    if kind == "gated_delta_scan":
        return _sole(_gated_delta_scan(contract, ins, where))
    raise OpContractError(f"{where}: op kind '{kind}' has no shape rule")


def _matmul(ins: list[list[Extent]], where: str) -> list[Extent]:
    a, b = ins
    if len(a) != 2 or len(b) != 2:
        raise OpContractError(
            f"{where}: matmul accepts only rank-2 × rank-2 ([{_show(a)}] × [{_show(b)}])"
        )
    if a[1] != b[0]:
        raise OpContractError(
            f"{where}: matmul contraction dimension mismatch [{_show(a)}] × [{_show(b)}]"
        )
    return [a[0], b[1]]


def _bmm(ins: list[list[Extent]], where: str) -> list[Extent]:
    a, b = ins
    # MUST: rank-2 does not pass (it is matmul's job). Sharing would let the form without the batch
    # axis pass under the same op name, and a B mix-up would slip through the shape check.
    if len(a) != 3 or len(b) != 3:
        raise OpContractError(
            f"{where}: bmm accepts only rank-3 × rank-3 (rank-2 is matmul):"
            f" [{_show(a)}] × [{_show(b)}]"
        )
    if a[0] != b[0]:
        raise OpContractError(f"{where}: bmm batch dimension mismatch [{_show(a)}] × [{_show(b)}]")
    if a[2] != b[1]:
        raise OpContractError(
            f"{where}: bmm contraction dimension mismatch [{_show(a)}] × [{_show(b)}]"
        )
    return [a[0], a[1], b[2]]


def _gather(ins: list[list[Extent]], where: str) -> list[Extent]:
    src, index = ins
    # The contract is "last dimension fixed" — the leading dimensions of src and index match
    # exactly and only the last dimension is free (narrower than torch's general gather).
    if not src or len(index) != len(src):
        raise OpContractError(
            f"{where}: gather needs src and index of the same rank (1 or more):"
            f" [{_show(src)}] / [{_show(index)}]"
        )
    for axis in range(len(src) - 1):
        if src[axis] != index[axis]:
            raise OpContractError(
                f"{where}: gather leading dimension {axis} mismatch"
                f" [{_show(src)}] / [{_show(index)}]"
            )
    # The output has index's shape (values are taken from src). The range of the indices
    # depends on run-time data, so it is not checked.
    return list(index)


def _row_reduce(
    contract: OpContract, ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    shape = ins[0]
    if not shape:
        raise OpContractError(
            f"{where}: reduce input must be rank 1 or more (a scalar cannot be reduced)"
        )
    dim = reduce_dim(attrs, where)
    # MUST: negatives and out-of-rank values fail loudly (normalising negative axis notation is
    # the handlers' responsibility; filling in `% rank` here would silently absorb "a mismatch
    # between the declaration and the actual rank" into another axis).
    if dim >= len(shape):
        raise OpContractError(
            f"{where}: op '{contract.name}' attrs.dim={dim} is out of range for rank {len(shape)}"
        )
    # amax/amin over an empty axis has no definable identity (sum gives 0). Whether a symbolic
    # dimension becomes 0 depends on the binding, so only constant dimensions can be decided here
    # (the rest is the runtime-side layer).
    if shape[dim].is_value(0) and contract.name != "sum":
        raise OpContractError(f"{where}: op '{contract.name}' cannot reduce an axis of length 0")
    return shape[:dim] + shape[dim + 1 :]


def _argmax(ins: list[list[Extent]], where: str) -> list[Extent]:
    """The output shape of argmax (ADR 0068 decision 2) — **rank-preserving, collapsing the last
    dimension to 1**.

    Unlike the reduce family, the axis is not in attrs but fixed to the last dimension (the absence
    of the field declares "other axes are not in the vocabulary"), and there is no `keepdim` field
    either, so the output rank equals the input's.
    """
    shape = ins[0]
    if not shape:
        raise OpContractError(
            f"{where}: argmax input must be rank 1 or more (a scalar cannot be reduced)"
        )
    # An axis of length 0 has no "index of the maximum" (torch rejects it too). Whether a symbolic
    # dimension becomes 0 depends on the binding, so only constant dimensions can be decided here
    # (the rest is the runtime-side layer — the same split as amax).
    if shape[-1].is_value(0):
        raise OpContractError(f"{where}: argmax cannot reduce a last dimension of length 0")
    return [*shape[:-1], _ONE]


def _topk(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[list[Extent]]:
    """The output shapes of topk (ADR 0068 decision 3) — **both values and indices** are
    `[..., k]`.

    The axis is fixed to the last dimension (the same narrowing as argmax) and the rank is
    preserved. The accepted domain is `1 <= k <= last dimension`; the attrs schema (TOPK_ATTRS)
    checks the lower bound and this function the upper bound.
    """
    shape = ins[0]
    if not shape:
        raise OpContractError(
            f"{where}: topk input must be rank 1 or more (a scalar cannot be reduced)"
        )
    k = topk_k(attrs, where)
    last = shape[-1]
    # For a symbolic last dimension the comparison with k is **deferred** (it depends on the
    # binding, so nothing is asserted — the decision is in the runtime-side layer, the same split as
    # amax's length-0 check). Length 0 fails against k >= 1.
    if last.is_const and k > last.offset:
        raise OpContractError(
            f"{where}: topk attrs.k={k} exceeds the last dimension {last.offset}"
            f" (input [{_show(shape)}])"
        )
    out = [*shape[:-1], Extent(coeff=0, sym=None, offset=k)]
    return [out, list(out)]


def _reshape(
    contract: OpContract, ins: list[list[Extent]], where: str, declared: Sequence[IrDim] | None
) -> list[Extent]:
    target = _require_declared(declared, contract, where)
    source = ins[0]
    # The contract is only that element counts match (element order unchanged). Loosening it
    # would make the size of the aliased real buffer disagree with the declared shape, and readback
    # would read out of range.
    if _numel_key(source) != _numel_key(target):
        raise OpContractError(
            f"{where}: reshape element count mismatch [{_show(source)}] → [{_show(target)}]"
        )
    return target


def _permute(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    # The output rank equals the input's (it only reorders), so checking the input side is
    # enough.
    assert_strided_rank(len(source), "permute input", where)
    dims = permute_dims(attrs, where)
    if len(dims) != len(source):
        raise OpContractError(f"{where}: permute dims {dims} do not match input rank {len(source)}")
    out: list[Extent] = []
    for dim in dims:
        if dim >= len(source):
            raise OpContractError(
                f"{where}: permute dims has an axis outside input rank {len(source)}: {dim}"
            )
        out.append(source[dim])
    return out


def _expand(
    contract: OpContract, ins: list[list[Extent]], where: str, declared: Sequence[IrDim] | None
) -> list[Extent]:
    target = _require_declared(declared, contract, where)
    source = ins[0]
    assert_strided_rank(len(source), "expand input", where)
    assert_strided_rank(len(target), "expand output", where)
    if len(target) < len(source):
        raise OpContractError(
            f"{where}: expand cannot lower the rank [{_show(source)}] → [{_show(target)}]"
        )
    # Right-aligned; each input dimension either "matches the target" or "is length 1
    # (replicated with stride 0)".
    offset = len(target) - len(source)
    for index, extent in enumerate(source):
        if not extent.is_value(1) and extent != target[offset + index]:
            raise OpContractError(
                f"{where}: expand cannot expand dimension {index} ({extent.to_dim()}),"
                f" which is not of length 1, to {target[offset + index].to_dim()}"
            )
    return target


def _axis_extent(shape: list[Extent], dim: int, op: str, where: str) -> Extent:
    """Takes the length of the target axis (checks only that it is within the rank)."""
    if dim >= len(shape):
        raise OpContractError(f"{where}: {op} dim {dim} is outside input rank {len(shape)}")
    return shape[dim]


def _static_axis(shape: list[Extent], dim: int, op: str, where: str) -> Extent:
    """Takes the target axis of `slice` / `flip` (**static axes only** — ADR 0014).

    MUST: reject symbolic axes. The reason differs per op, but the conclusion is the same:
    - `slice` — slicing a symbolic axis is sym_prefix_slice's job (no duplication). In addition,
      the range check (end <= axis length) cannot be decided while symbolic.
    - `flip` — every observed case is a static axis (flow's 192ch / sdp's 2ch). Dynamic axes are not
      widened until an observed need appears.

    The concatenation axis of `cat` was loosened to "a linear sum of the same symbol" by ADR 0046
    (see `_cat`).

    NOTE: the TS side looks only at **numeric shapes after binding resolution**, so this decision is
    held by plan.ts (the layer that looks at declared shapes) — a different layer, but the accepted
    set is the same on both sides.
    """
    extent = _axis_extent(shape, dim, op, where)
    if not extent.is_const:
        raise OpContractError(
            f"{where}: {op} axis {dim} is a symbolic dimension {extent.to_dim()}"
            " (the target axis must be static — slicing a symbolic axis is sym_prefix_slice)"
        )
    return extent


def _slice(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    # Executed by reusing the strided read copy family (ADR 0014), so the rank limit is the
    # same.
    assert_strided_rank(len(source), "slice input", where)
    dim, start, end = slice_attrs(attrs, where)
    extent = _static_axis(source, dim, "slice", where)
    # MUST: out-of-range slices do not pass (on the GPU they would read the neighbouring row
    # without an exception).
    if end > extent.offset:
        raise OpContractError(f"{where}: slice end {end} exceeds axis {dim} length {extent.offset}")
    # MUST: an invariant that spans keys (the same split as clamp's min <= max).
    if start > end:
        raise OpContractError(f"{where}: slice start {start} exceeds end {end}")
    out = list(source)
    out[dim] = Extent(coeff=0, sym=None, offset=end - start)
    return out


def _cat(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """The concatenation axis is a constant or a linear expression in **the same** symbol (ADR
    0046 revises ADR 0014).

    MUST: reject a concatenation that mixes different symbols — the sum does not fit the dimension
    language (a linear expression with one symbol per dimension). With the same symbol,
    `Σ(coeff_i·sym + offset_i)` is the canonical form as is.
    """
    dim = axis_dim(attrs, where)
    first = ins[0]
    assert_strided_rank(len(first), "cat input", where)
    coeff = 0
    offset = 0
    sym: str | None = None
    for index, shape in enumerate(ins):
        if len(shape) != len(first):
            raise OpContractError(
                f"{where}: cat input {index} rank {len(shape)} differs from"
                f" input 0 rank {len(first)}"
            )
        extent = _axis_extent(shape, dim, "cat", where)
        if extent.sym is not None:
            if sym is not None and sym != extent.sym:
                raise OpContractError(
                    f"{where}: cat concatenation axis {dim} mixes different symbols"
                    f" ({sym} and {extent.sym}) — the sum does not fit the dimension language"
                    " coeff·sym+offset"
                )
            sym = extent.sym
        # MUST: every axis **other than** the concatenation axis must match (as in torch).
        # Loosening it would leave part of the output unwritten by any input, breaking the
        # full-write invariant.
        for axis, other in enumerate(shape):
            if axis != dim and other != first[axis]:
                raise OpContractError(
                    f"{where}: cat input {index} [{_show(shape)}] differs from"
                    f" input 0 [{_show(first)}] on axis {axis} (the concatenation axis is {dim})"
                )
        coeff += extent.coeff
        offset += extent.offset
    out = list(first)
    # Output axis length = sum of the input axis lengths. This rule itself guarantees "all inputs
    # together cover the whole output" (full-write — ADR 0014). With constants only it returns to
    # sym=None, coeff=0.
    out[dim] = Extent(coeff=coeff, sym=sym, offset=offset)
    return out


def _pad(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    if not source:
        raise OpContractError(
            f"{where}: pad input must be rank 1 or more (it needs a last dimension)"
        )
    left, right = pad_attrs(attrs, where)
    last = source[-1]
    out = list(source)
    # A symbolic last dimension can be padded too (the observed T+2w = T+8 — the dimension
    # language's form with an offset).
    out[-1] = Extent(coeff=last.coeff, sym=last.sym, offset=last.offset + left + right)
    return out


def _flip(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    if not source:
        raise OpContractError(
            f"{where}: flip input must be rank 1 or more (it needs an axis to reverse)"
        )
    _static_axis(source, axis_dim(attrs, where), "flip", where)
    # Reversal does not change the shape (identity shape rule).
    return list(source)


def _sym_prefix_slice(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    source = ins[0]
    # The output rank equals the input's (it only cuts the head of each axis).
    assert_strided_rank(len(source), "sym_prefix_slice input", where)
    sym, slices = sym_prefix_slice_attrs(attrs, where)
    out = list(source)
    for entry in slices:
        dim = entry["dim"]
        if dim >= len(source):
            raise OpContractError(
                f"{where}: sym_prefix_slice dim {dim} is outside input rank {len(source)}"
            )
        # MUST: the input is a static form baked at Tmax (ADR 0010). Allowing a symbolic input
        # dimension would change the read stride per run and break the meaning of the prefix.
        if not source[dim].is_const:
            raise OpContractError(
                f"{where}: sym_prefix_slice input [{_show(source)}] dimension {dim} is symbolic"
                " (the input must be a static form baked at Tmax)"
            )
        # The check prefix length coeff·sym+offset ≤ constant dimension (Tmax exceeded) needs
        # the binding, so the runtime-side layer holds it — the upper bound is not in the IR.
        out[dim] = Extent(coeff=entry["coeff"], sym=sym, offset=entry["offset"])
    return out


def _linear(ins: list[list[Extent]], where: str) -> list[Extent]:
    x, weight, bias = ins
    if len(x) < 1 or len(weight) != 2 or len(bias) != 1:
        raise OpContractError(
            f"{where}: linear expects x[…,in] × W[out,in] + b[out] (rank ≥ 1 / 2 / 1):"
            f" [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    out_features, in_features = weight
    if x[-1] != in_features:
        raise OpContractError(
            f"{where}: linear input feature count mismatch [{_show(x)}] × [{_show(weight)}]"
        )
    if bias[0] != out_features:
        raise OpContractError(
            f"{where}: linear bias length {bias[0].to_dim()} differs from"
            f" the output feature count {out_features.to_dim()}"
        )
    return [*x[:-1], out_features]


def _layer_norm(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    x, weight, bias = ins
    normalized_shape, _ = layer_norm_attrs(attrs, where)
    # The contract is "last dimension only" (attrs already checked). Here the actual shape's
    # tail is matched.
    normalized = Extent(coeff=0, sym=None, offset=normalized_shape[0])
    if len(x) < 1 or x[-1] != normalized:
        raise OpContractError(
            f"{where}: layer_norm normalized_shape {normalized_shape} differs from"
            f" the last dimension of input [{_show(x)}]"
        )
    for name, shape in (("weight", weight), ("bias", bias)):
        if len(shape) != 1 or shape[0] != normalized:
            raise OpContractError(
                f"{where}: layer_norm {name} [{_show(shape)}] differs from"
                f" normalized_shape {normalized_shape}"
            )
    return list(x)


def _rms_norm(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """The output shape of rms_norm (ADR 0017).

    MUST: the source of truth for the normalised length is **the weight's length** (no
    normalized_shape field in attrs). layer_norm holds the same fact twice, in attrs and in the
    weight, and checks their agreement here — rms_norm keeps the fact in one place and makes the
    check itself unnecessary.
    """
    x, weight = ins
    # MUST: eps is read here too (keeps a single path through the range check — the same role as
    # unary's scalar_param_values).
    rms_norm_eps(attrs, where)
    if len(x) < 1:
        raise OpContractError(
            f"{where}: rms_norm input must be rank 1 or more (it needs a last dimension)"
        )
    # An axis of length 0 cannot be reduced (a sum of squares 0 / element count 0 makes the mean
    # 0/0 — the same as softmax).
    if x[-1].is_value(0):
        raise OpContractError(f"{where}: rms_norm cannot normalise an axis of length 0")
    if len(weight) != 1 or weight[0] != x[-1]:
        raise OpContractError(
            f"{where}: rms_norm weight [{_show(weight)}] does not match"
            f" input [{_show(x)}]: expected rank-1 of the last dimension length {x[-1].to_dim()}"
        )
    return list(x)


def _softmax(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any], name: str
) -> list[Extent]:
    shape = ins[0]
    dim = softmax_dim(attrs, where)
    # MUST: a general dim is not accepted "to be implemented someday". Anything but the last
    # dimension breaks the row kernel's premise (a contiguous reduced axis), and letting it through
    # would silently fold a different axis.
    if len(shape) < 1 or dim != len(shape) - 1:
        raise OpContractError(
            f"{where}: {name} accepts only the last dimension"
            f" (attrs.dim={dim} / input [{_show(shape)}])"
        )
    if shape[-1].is_value(0):
        raise OpContractError(f"{where}: {name} cannot reduce an axis of length 0")
    return list(shape)


def _assert_state_slot_form(
    name: str,
    slot: list[Extent],
    ins: list[Extent],
    window: int | None,
    where: str,
) -> None:
    """The physical form of a state slot (①②③ of ADR 0067 decision 4; mirror of
    assertStateSlotForm on the TS side).

    A slot is **fixed at `[B, Hkv, C, D]`**, its B / Hkv / D match this step's chunk (`ins`), and
    for sliding (attrs `window`) `window ≤ C` must hold.

    MUST: fail here, as the **state extension** of the shape check that otherwise looks only at
    ordinary values. A slot mix-up (referencing another layer's slot) is an OOB access when the
    capacity differs, and a silent misread when the capacity is the same.
    NOTE: `window ≤ C` is decided **only when C is a constant** (a symbolic capacity depends on the
    binding at `createGenerationContext` — what the declaration alone cannot decide is not treated
    as decided). The TS side handles only numbers after binding resolution, so it can always
    decide = only this side defers.
    NOTE: `pastLength + queryLength ≤ C` for a full slot needs the logical lengths at run time, so
    it is on the context side (④ of ADR 0067 decision 4) — it cannot be checked here.
    """
    show = f"state slot '{name}' [{_show(slot)}] / input [{_show(ins)}]"
    if len(slot) != 4:
        raise OpContractError(f"{where}: {show} — the slot must be rank-4 [B,Hkv,C,D]")
    if slot[0] != ins[0] or slot[1] != ins[1] or slot[3] != ins[3]:
        raise OpContractError(f"{where}: {show} — slot and input B / Hkv / D mismatch")
    if window is not None and slot[2].is_const and window > slot[2].offset:
        raise OpContractError(
            f"{where}: {show} — attrs.window {window} exceeds the slot capacity {slot[2].offset}"
        )


def _state_append(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any], slots: _StateContext
) -> list[list[Extent]]:
    """The output shapes of `state_append` = **an empty list** (ADR 0067 decision 5 — an effect op
    that defines no value).

    The input is exactly rank-4 `[B,Hkv,M,D]` (the same physical chunk dimensions as attention's
    ins), and its agreement with the referenced slot is held to **the same detection line** as the
    reading attention (same formula for reads and writes, MUST).
    """
    x = ins[0]
    if len(x) != 4:
        raise OpContractError(
            f"{where}: state_append input must be rank-4 [B,Hkv,M,D]: [{_show(x)}]"
        )
    window = state_window(attrs, where)
    name, slot = slots.slot("slot", where)
    _assert_state_slot_form(name, slot, x, window, where)
    return []


def _attention_states(
    q: list[Extent],
    k: list[Extent],
    v: list[Extent],
    rest: list[list[Extent]],
    where: str,
    attrs: Mapping[str, Any],
    slots: _StateContext,
    show: str,
) -> list[Extent]:
    """The states-form attention branch (ADR 0067 decision 4 — entered from `_attention` after the
    head / D checks).

    MUST: no mask is taken (causal is fixed and computed by predicate — no mask tensor is
    materialised). The contract layer rejects it too, but direct calls from the conformance table
    come through here.
    MUST: M **matches exactly** across q / k / v (they share this step's chunk — unlike the free N
    of the previous form, a mismatch here is nothing but IR that "mixed in another step's k/v").
    MUST: the k / v slots have **the same shape** (pairing slots with different capacities makes a
    ring where one side wraps first, and the values silently drift).
    """
    if rest:
        raise OpContractError(
            f"{where}: states-form attention takes no mask (causal is fixed —"
            f" ADR 0067 decision 4) {show} + mask [{_show(rest[0])}]"
        )
    if k[2] != q[2] or v[2] != q[2]:
        raise OpContractError(
            f"{where}: states-form attention needs the same M (axis 2) in q / k / v {show}"
        )
    # softmax over an empty axis has no definable amax identity (the same reason as the previous
    # form's N=0 rejection). Whether a symbolic dimension becomes 0 depends on the binding, so only
    # constant dimensions can be decided here.
    if q[2].is_value(0):
        raise OpContractError(
            f"{where}: states-form attention cannot handle a chunk with M = 0 {show}"
        )
    window = state_window(attrs, where)
    k_name, k_slot = slots.slot("k", where)
    v_name, v_slot = slots.slot("v", where)
    if k_slot != v_slot:
        raise OpContractError(
            f"{where}: attention k / v slots differ in shape"
            f" ('{k_name}' [{_show(k_slot)}] / '{v_name}' [{_show(v_slot)}])"
        )
    _assert_state_slot_form(k_name, k_slot, k, window, where)
    _assert_state_slot_form(v_name, v_slot, v, window, where)
    return list(q)


def _assert_kv_head_broadcast(q_heads: Extent, kv_heads: Extent, where: str, show: str) -> None:
    """Divisible broadcast of `H` and `Hkv` (ADR 0067 decision 1 — the only accepted form for GQA
    / MQA).

    The three forms — previous / states / readonly — impose the same rule, so it is kept in one
    function (a copy would loosen on one side only — a slot's Hkv legitimately differs from q's, so
    the loosening would show only in the values).

    MUST: `Hkv ≥ 1` is checked **before** the equality short-circuit — `(H,Hkv) = (0,0)` is
    structurally equal, so it never reaches the divisibility branch and "IR that dropped the whole
    head axis" slips through. `H = 0` alone pairs with `Hkv ≥ 1`, so the `H ≥ Hkv` branch below
    rejects it (the mirror of computeOutputShape on the TS side). Symbolic dimensions are left to
    the "the declaration alone cannot decide" side (the branch below), so only a constant Hkv is
    checked.
    MUST: check `H ≥ Hkv` as a **separate condition** from divisibility — `H = 0` satisfies
    `0 % Hkv == 0`, so divisibility alone lets "IR that dropped H entirely" through. The broadcast
    always goes kv → q.
    MUST: `Hkv == 0` is rejected by the `Hkv ≥ 1` branch above **before** the remainder — Python's
    `%` raises ZeroDivisionError, so if the condition order broke it would no longer fail with the
    same contract error as the TS side (where `4 % 0` is NaN and the condition is true) (`%` can be
    written here because that order is kept).
    """
    if kv_heads.is_const and kv_heads.offset < 1:
        raise OpContractError(
            f"{where}: attention Hkv {kv_heads.offset} is not positive (H must be a positive"
            " integer multiple of Hkv — GQA needs H % Hkv == 0 and H ≥ Hkv ≥ 1;"
            f" ADR 0067 decision 1) {show}"
        )
    if q_heads == kv_heads:
        return
    # Symbolic dimensions fail as "the declaration alone cannot decide" (the same discipline as
    # broadcast_extents — letting them through silently would allow IR you cannot judge without
    # running it. No symbolic head count has been observed).
    if not q_heads.is_const or not kv_heads.is_const:
        raise OpContractError(
            f"{where}: attention H {q_heads.to_dim()} and Hkv {kv_heads.to_dim()}:"
            f" the declaration alone cannot decide whether they broadcast divisibly {show}"
        )
    if q_heads.offset < kv_heads.offset or q_heads.offset % kv_heads.offset != 0:
        raise OpContractError(
            f"{where}: attention H {q_heads.offset} is not a positive integer multiple of"
            f" Hkv {kv_heads.offset} (GQA needs H % Hkv == 0 and H ≥ Hkv — ADR 0067 decision 1)"
            f" {show}"
        )


def _assert_readonly_slot_form(
    name: str, slot: list[Extent], q: list[Extent], window: int | None, where: str
) -> None:
    """The physical form of the slots a readonly attention reads (ADR 0096 stage 2 — contract
    §1.2).

    The only difference from the states form is that the counterpart compared against is **q**,
    not this step's k / v; the axes checked are B (axis 0) and D (axis 3) + the divisible broadcast
    of the head axis + `window ≤ C`.

    MUST: do not turn this into "no k/v for this step, so no axes are checked" — a slot mix-up
    (referencing a slot of another layer or another kind) is an OOB access when the capacity
    differs and a silent misread when it is the same, so the same detection line as the states form
    is redrawn against q.
    """
    show = f"state slot '{name}' [{_show(slot)}] / q [{_show(q)}]"
    if len(slot) != 4:
        raise OpContractError(f"{where}: {show} — the slot must be rank-4 [B,Hkv,C,D]")
    if slot[0] != q[0] or slot[3] != q[3]:
        raise OpContractError(f"{where}: {show} — slot and q B / D mismatch")
    _assert_kv_head_broadcast(q[1], slot[1], where, show)
    if window is not None and slot[2].is_const and window > slot[2].offset:
        raise OpContractError(
            f"{where}: {show} — attrs.window {window} exceeds the slot capacity {slot[2].offset}"
        )


def _attention_readonly(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any], slots: _StateContext
) -> list[Extent]:
    """The output shape of a readonly-form attention (ADR 0096 stage 2; contract §1.2).

    Takes **only** `q[B,H,1,D]`; the past columns are held by the k / v slots → `[B,H,1,D]`.

    MUST: M (axis 2) is **exactly 1**. The readonly form has meaning only as "the one row at logical
    position P−1 reads the columns `[column_base, P)`" (stage 2 covers only the ①′ family), so
    letting M > 1 through would run without anyone deciding "which row sits at which position".
    MUST: the k / v slots have **the same shape** (pairing slots with different capacities makes a
    ring where one side wraps first, and the values silently drift — the same reason as the states
    form).
    """
    # MUST: scale is read here too (the same role as the common path of `_attention`).
    attention_scale(attrs, where)
    if len(ins) != 1:
        raise OpContractError(
            f"{where}: readonly-form attention takes exactly one input, q ({len(ins)} given —"
            " it takes neither this step's k/v nor a mask; ADR 0096 stage 2)"
        )
    q = ins[0]
    show = f"q [{_show(q)}]"
    if len(q) != 4:
        raise OpContractError(
            f"{where}: readonly-form attention accepts only rank-4 q[B,H,1,D]: {show}"
        )
    if not q[2].is_value(1):
        raise OpContractError(
            f"{where}: readonly-form attention needs M (axis 2) of exactly 1 {show}"
        )
    window = state_window(attrs, where)
    k_name, k_slot = slots.slot("k", where)
    v_name, v_slot = slots.slot("v", where)
    if k_slot != v_slot:
        raise OpContractError(
            f"{where}: attention k / v slots differ in shape"
            f" ('{k_name}' [{_show(k_slot)}] / '{v_name}' [{_show(v_slot)}])"
        )
    _assert_readonly_slot_form(k_name, k_slot, q, window, where)
    _assert_readonly_slot_form(v_name, v_slot, q, window, where)
    return list(q)


def _attention(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any], slots: _StateContext
) -> list[Extent]:
    """The output shape of attention (ADR 0023 + 0067 decision 1 / decision 4).

    `q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]` (+ an optional `mask[1,1,M,N]`) → `[B,H,M,D]`.

    MUST: B **matches exactly** across q / k / v (looking only at the product lets a mix-up
    through).
    MUST: H and Hkv are a **divisible broadcast** (`H % Hkv == 0` and `H ≥ Hkv ≥ 1` — ADR 0067
    decision 1). `r = H / Hkv` is a derived value with no attrs field. MQA with Hkv=1 uses the same
    formula. The equal Hkv of k and v, the same D for all three and the N=0 rejection **stay as
    detectors of mix-ups**.
    MUST: D is the same for all three (allowing a different length on the v side alone would let a
    mix-up escape the element count).
    MUST: the first two mask axes are **exactly 1** (dedicated to broadcasting to B·H). Letting
    `[B,1,M,N]` through would add forms that "pass silently as long as B matches" and make the
    kernel carry broadcast rules outside the contract — rejected by the absence of a field (widened
    when a need for run-time masks appears).

    **The states form** (ADR 0067 decision 4): a node with a `states` field reads "the past from
    the slots, this step's k/v from `ins`"; the head / D rules stay common to both forms and **they
    branch from there** — the previous form looks at the N of k/v (the full length, past + present),
    while the states form's ins are only this step's chunk, so **M agrees across the three** and
    the past is held by the slot capacity C.
    """
    # The readonly form (ADR 0096 stage 2) has no k / v for this step = **its only input is q**,
    # so unpacking q/k/v itself does not work. The branch MUST come before the unpacking.
    if slots.referenced and attention_readonly(attrs):
        return _attention_readonly(ins, where, attrs, slots)
    q, k, v, *rest = ins
    # MUST: scale is read here too (the same role as computeOutputShape on the TS side — this
    # computation is the only common path every node goes through, and the attrs schema can express
    # only per-key checks).
    attention_scale(attrs, where)
    show = f"[{_show(q)}] / [{_show(k)}] / [{_show(v)}]"
    if len(q) != 4 or len(k) != 4 or len(v) != 4:
        raise OpContractError(
            f"{where}: attention accepts only rank-4"
            f" q[B,H,M,D] / k[B,Hkv,N,D] / v[B,Hkv,N,D]: {show}"
        )
    if q[0] != k[0] or q[0] != v[0]:
        raise OpContractError(f"{where}: attention axis 0 (B) mismatch {show}")
    # MUST: the Hkv of k / v **matches exactly** (GQA loosens only the relation to q — ADR 0067
    # decision 1).
    if k[1] != v[1]:
        raise OpContractError(f"{where}: attention Hkv (axis 1 of k / v) mismatch {show}")
    _assert_kv_head_broadcast(q[1], k[1], where, show)
    if q[3] != k[3] or q[3] != v[3]:
        raise OpContractError(f"{where}: attention D (axis 3) mismatch {show}")
    if slots.referenced:
        return _attention_states(q, k, v, rest, where, attrs, slots, show)
    if k[2] != v[2]:
        raise OpContractError(f"{where}: attention N (axis 2 of k / v) mismatch {show}")
    # softmax over an empty axis has no definable amax identity. Whether a symbolic dimension
    # becomes 0 depends on the binding, so only constant dimensions can be decided here (the rest is
    # the runtime-side layer — module docstring).
    if k[2].is_value(0):
        raise OpContractError(f"{where}: attention cannot reduce an N of length 0 {show}")
    if rest:
        mask = rest[0]
        if len(mask) != 4 or not mask[0].is_value(1) or not mask[1].is_value(1):
            raise OpContractError(f"{where}: attention mask must be [1,1,M,N] ([{_show(mask)}])")
        if mask[2] != q[2] or mask[3] != k[2]:
            raise OpContractError(
                f"{where}: attention mask [{_show(mask)}] M / N do not match q / k {show}"
            )
    return list(q)


def _cumsum(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    shape = ins[0]
    dim = cumsum_dim(attrs, where)
    # MUST: nothing but the last dimension is accepted (the same reason as softmax — the row
    # kernel assumes the reduced axis is contiguous, and letting it through would silently fold a
    # different axis).
    if len(shape) < 1 or dim != len(shape) - 1:
        raise OpContractError(
            f"{where}: cumsum accepts only the last dimension"
            f" (attrs.dim={dim} / input [{_show(shape)}])"
        )
    # An axis of length 0 passes through (the identity of a prefix sum is 0 — unlike amax /
    # softmax, it can be defined).
    return list(shape)


def _embedding(ins: list[list[Extent]], where: str) -> list[Extent]:
    weight, index = ins
    if len(weight) != 2:
        raise OpContractError(f"{where}: embedding weight must be rank-2 [V,H]: [{_show(weight)}]")
    if len(index) < 1:
        raise OpContractError(
            f"{where}: embedding index must be rank 1 or more (there is no scalar index)"
        )
    # The index range 0 <= index < V depends on run-time data, so the shape contract does not
    # check it.
    return [*index, weight[1]]


def _masked_fill(ins: list[list[Extent]], where: str) -> list[Extent]:
    x, mask = ins
    assert_strided_rank(len(x), "masked_fill x", where)
    assert_strided_rank(len(mask), "masked_fill mask", where)
    # MUST: the output **always has x's shape** (the mask side is only read with right-aligned
    # broadcast). Using broadcast as is would also let through forms where the mask widens x, and
    # the fill value would leak into elements that do not exist.
    if len(mask) > len(x):
        raise OpContractError(
            f"{where}: masked_fill mask rank {len(mask)} exceeds x rank {len(x)}"
            " (mask broadcasts right-aligned only)"
        )
    offset = len(x) - len(mask)
    for index, extent in enumerate(mask):
        if not extent.is_value(1) and extent != x[offset + index]:
            raise OpContractError(
                f"{where}: masked_fill mask [{_show(mask)}] cannot be broadcast"
                f" right-aligned to x [{_show(x)}] (dimension {index})"
            )
    return list(x)


def _conv1d(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    x, weight, bias = ins
    stride, padding, dilation, groups = conv1d_attrs(attrs, where)
    if len(x) != 3 or len(weight) != 3 or len(bias) != 1:
        raise OpContractError(
            f"{where}: conv1d expects x[B,Cin,L] / W[Cout,Cin/groups,K] / b[Cout] (rank 3 / 3 / 1):"
            f" [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    channels_out, weight_in, kernel = weight
    # MUST: the channel axes are static (symbolic Cin/Cout has not been observed, and whether groups
    # divides them cannot be decided).
    if not channels_in.is_const or not channels_out.is_const:
        raise OpContractError(
            f"{where}: conv1d channel axes {channels_in.to_dim()} / {channels_out.to_dim()} are"
            " symbolic (cannot decide whether groups divides them)"
        )
    # MUST: the group split must divide on both sides; that is the contract (depthwise is
    # groups = Cin = Cout).
    if channels_in.offset % groups != 0 or channels_out.offset % groups != 0:
        raise OpContractError(
            f"{where}: conv1d groups {groups} does not divide Cin {channels_in.offset} /"
            f" Cout {channels_out.offset}"
        )
    if not weight_in.is_value(channels_in.offset // groups):
        raise OpContractError(
            f"{where}: conv1d weight should be [Cout, Cin/groups, K]"
            f" (Cin/groups = {channels_in.offset // groups}) but is [{_show(weight)}]"
            f" (x is [{_show(x)}] / groups {groups})"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: conv1d bias length {bias[0].to_dim()} differs from"
            f" the output channels {channels_out.to_dim()}"
        )
    if not kernel.is_const:
        raise OpContractError(
            f"{where}: conv1d kernel length {kernel.to_dim()} is symbolic"
            " (the output length cannot be determined)"
        )
    return [
        x[0],
        channels_out,
        _conv_length("conv1d", x[2], kernel.offset, stride, padding, dilation, where),
    ]


def _conv2d(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """The output shape of conv2d (ADR 0017).

    MUST: the weight is **[Cout, Cin/groups, Kh, Kw]**, and the **order** of Kh and Kw is part of
    the contract. A mix-up with a matching element count (Cout and Cin/groups swapped / Kh and Kw
    swapped) slips through on symmetric forms, so the conformance table has asymmetric forms with
    Cin ≠ Cout and Kh ≠ Kw (the lesson of conv_transpose1d).
    MUST: the two spatial axes apply the same general form **independently** (H and W are not
    folded into one).
    """
    x, weight, bias = ins
    stride, padding, dilation, groups = conv2d_attrs(attrs, where)
    if len(x) != 4 or len(weight) != 4 or len(bias) != 1:
        raise OpContractError(
            f"{where}: conv2d expects x[B,Cin,H,W] / W[Cout,Cin/groups,Kh,Kw] / b[Cout]"
            f" (rank 4 / 4 / 1): [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    channels_out, weight_in, kernel_h, kernel_w = weight
    # MUST: the channel axes are static (symbolic Cin/Cout has not been observed, and whether groups
    # divides them cannot be decided).
    if not channels_in.is_const or not channels_out.is_const:
        raise OpContractError(
            f"{where}: conv2d channel axes {channels_in.to_dim()} / {channels_out.to_dim()} are"
            " symbolic (cannot decide whether groups divides them)"
        )
    if channels_in.offset % groups != 0 or channels_out.offset % groups != 0:
        raise OpContractError(
            f"{where}: conv2d groups {groups} does not divide Cin {channels_in.offset} /"
            f" Cout {channels_out.offset}"
        )
    if not weight_in.is_value(channels_in.offset // groups):
        raise OpContractError(
            f"{where}: conv2d weight should be [Cout, Cin/groups, Kh, Kw]"
            f" (Cin/groups = {channels_in.offset // groups}) but is [{_show(weight)}]"
            f" (x is [{_show(x)}] / groups {groups})"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: conv2d bias length {bias[0].to_dim()} differs from"
            f" the output channels {channels_out.to_dim()}"
        )
    for name, kernel in (("Kh", kernel_h), ("Kw", kernel_w)):
        if not kernel.is_const:
            raise OpContractError(
                f"{where}: conv2d kernel length {name} {kernel.to_dim()} is symbolic"
                " (the output length cannot be determined)"
            )
    return [
        x[0],
        channels_out,
        _conv_length("conv2d H", x[2], kernel_h.offset, stride[0], padding[0], dilation[0], where),
        _conv_length("conv2d W", x[3], kernel_w.offset, stride[1], padding[1], dilation[1], where),
    ]


def _conv3d(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """The output shape of conv3d (ADR 0118 decision 1 — **unbatched**).

    MUST: x is rank 4 [Cin, T, H, W] only. Accepting batched rank 5 could let a mix-up that reads
    the leading axis as Cin slip through with a matching element count (the contract has no
    batched form).
    MUST: the weight is **[Cout, Cin/groups, Kt, Kh, Kw]**, and the **order** of Kt, Kh and Kw is
    part of the contract. The conformance table has asymmetric forms with Cin ≠ Cout and
    Kt ≠ Kh ≠ Kw (forms whose output shape changes under a mix-up).
    MUST: a kernel length of 0 fails before the output length formula (`dilation·(K−1)` turns
    negative and the output becomes longer than the input — the runtime's params layer rejects it
    as a positive integer, so if this check were missing the accepted set would split).
    MUST: the three axes apply the same general form **independently** ({@link _conv_length} is
    called per axis).
    """
    x, weight, bias = ins
    stride, padding, dilation, groups = conv3d_attrs(attrs, where)
    if len(x) != 4 or len(weight) != 5 or len(bias) != 1:
        raise OpContractError(
            f"{where}: conv3d expects x[Cin,T,H,W] / W[Cout,Cin/groups,Kt,Kh,Kw] / b[Cout]"
            f" (rank 4 / 5 / 1 — x is unbatched):"
            f" [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    channels_in = x[0]
    channels_out, weight_in, *kernels = weight
    # MUST: the channel axes are static (symbolic Cin/Cout has not been observed, and whether groups
    # divides them cannot be decided).
    if not channels_in.is_const or not channels_out.is_const:
        raise OpContractError(
            f"{where}: conv3d channel axes {channels_in.to_dim()} / {channels_out.to_dim()} are"
            " symbolic (cannot decide whether groups divides them)"
        )
    if channels_in.offset % groups != 0 or channels_out.offset % groups != 0:
        raise OpContractError(
            f"{where}: conv3d groups {groups} does not divide Cin {channels_in.offset} /"
            f" Cout {channels_out.offset}"
        )
    if not weight_in.is_value(channels_in.offset // groups):
        raise OpContractError(
            f"{where}: conv3d weight should be [Cout, Cin/groups, Kt, Kh, Kw]"
            f" (Cin/groups = {channels_in.offset // groups}) but is [{_show(weight)}]"
            f" (x is [{_show(x)}] / groups {groups})"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: conv3d bias length {bias[0].to_dim()} differs from"
            f" the output channels {channels_out.to_dim()}"
        )
    for name, kernel in zip(("Kt", "Kh", "Kw"), kernels, strict=True):
        if not kernel.is_const:
            raise OpContractError(
                f"{where}: conv3d kernel length {name} {kernel.to_dim()} is symbolic"
                " (the output length cannot be determined)"
            )
        if kernel.offset < 1:
            raise OpContractError(
                f"{where}: conv3d kernel length {name} must be a positive integer"
                f" (weight [{_show(weight)}])"
            )
    return [
        channels_out,
        *(
            _conv_length(
                f"conv3d {label}",
                x[1 + axis],
                kernels[axis].offset,
                stride[axis],
                padding[axis],
                dilation[axis],
                where,
            )
            for axis, label in enumerate(("T", "H", "W"))
        ),
    ]


def _conv_length(
    label: str,
    length: Extent,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    where: str,
) -> Extent:
    """The output length `floor((L + 2P - D*(K-1) - 1) / S) + 1` (the general form with dilation of
    ADR 0015).

    Every axis of conv1d / conv2d / conv3d follows **the same rule**, so one function is shared
    (`label` is the subject of the diagnostic). For a symbolic length the division stays linear:
    with `L = c·s + o` and `c % S == 0`, `floor((c·s + o') / S) = (c/S)·s + floor(o'/S)` holds
    exactly (c·s is a multiple of S). A form that does not divide does not fit the canonical
    grammar (a linear expression with one symbol per dimension) — it fails instead of being
    silently approximated (torch cannot build the shape expression either, for the same reason, so
    in practice it fails before reaching here).
    """
    span_end = dilation * (kernel - 1) + 1
    span = length.offset + 2 * padding - span_end
    if length.is_const:
        if span < 0:
            raise OpContractError(
                f"{where}: {label} input length {length.offset} (padding {padding}) is shorter"
                f" than the kernel span with dilation {dilation}: {span_end}"
            )
        return Extent(coeff=0, sym=None, offset=span // stride + 1)
    if length.coeff % stride != 0:
        raise OpContractError(
            f"{where}: {label} symbolic input length {length.to_dim()} with stride {stride}"
            " does not give a linear output length (it does not fit the canonical grammar)"
        )
    offset = span // stride + 1
    if offset < 0:
        raise OpContractError(
            f"{where}: {label} output length for {length.to_dim()} (K={kernel} / P={padding} /"
            f" S={stride} / D={dilation}) has a negative offset {offset} (it does not fit the"
            " canonical grammar)"
        )
    return Extent(coeff=length.coeff // stride, sym=length.sym, offset=offset)


def _conv_transpose1d(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    """The output shape of conv_transpose1d (ADR 0015).

    MUST: the weight is **[Cin, Cout, K]** (transposed relative to conv1d). A mix-up can produce a
    form with a matching element count that slips through the shape check, so the tests pin
    asymmetric channel counts.
    MUST: only forms whose output length is exactly `L·stride` (`2P == K - S`) are accepted. The
    general form `(L-1)·S - 2P + K` is deferred (all five observed cases have this form — widened
    when there is demand).
    """
    x, weight, bias = ins
    stride, padding = conv_transpose1d_attrs(attrs, where)
    if len(x) != 3 or len(weight) != 3 or len(bias) != 1:
        raise OpContractError(
            f"{where}: conv_transpose1d expects x[B,Cin,L] / W[Cin,Cout,K] / b[Cout]"
            f" (rank 3 / 3 / 1): [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    weight_in, channels_out, kernel = weight
    if weight_in != channels_in:
        raise OpContractError(
            f"{where}: conv_transpose1d weight should be [Cin, Cout, K]"
            f" (Cin = {channels_in.to_dim()}) but is [{_show(weight)}] (x is [{_show(x)}])"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: conv_transpose1d bias length {bias[0].to_dim()} differs from"
            f" the output channels {channels_out.to_dim()}"
        )
    if not kernel.is_const:
        raise OpContractError(
            f"{where}: conv_transpose1d kernel length {kernel.to_dim()} is symbolic"
            " (the output length cannot be determined)"
        )
    if 2 * padding != kernel.offset - stride:
        raise OpContractError(
            f"{where}: conv_transpose1d accepts only the form 2·padding == K - stride"
            f" (K {kernel.offset} / stride {stride} / padding {padding} —"
            " otherwise the output length is not L·stride)"
        )
    length = x[2]
    # Output length = L·stride. For a symbolic length both the coefficient and the offset are
    # multiplied by stride (it stays linear and fits the canonical grammar).
    if length.is_const:
        out_length = Extent(coeff=0, sym=None, offset=length.offset * stride)
    else:
        out_length = Extent(
            coeff=length.coeff * stride, sym=length.sym, offset=length.offset * stride
        )
    return [x[0], channels_out, out_length]


def _deform_conv2d(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """The output shape of deform_conv2d (extended atomic layer, the old layer 1'; ADR 0055).

    MUST: the weight is **[Cout, Cin, Kh, Kw]** (no groups field = fixed at 1, so the second axis is
    Cin itself). A mix-up can produce a form with a matching element count, so the conformance table
    has asymmetric forms with Cin != Cout and Kh != Kw (the same lesson as conv2d).
    MUST: the output spatial size is **derived** from x + weight + padding and only matched against
    offset / mask. Taking Hout from the offset side would let "IR where only offset has a different
    shape" slip through.
    """
    x, weight, offset, mask, bias = ins
    padding = deform_conv2d_attrs(attrs, where)
    if len(x) != 4 or len(weight) != 4 or len(offset) != 4 or len(mask) != 4 or len(bias) != 1:
        raise OpContractError(
            f"{where}: deform_conv2d expects x[B,Cin,H,W] / W[Cout,Cin,Kh,Kw] /"
            " offset[B,2*Kh*Kw,Hout,Wout] / mask[B,Kh*Kw,Hout,Wout] / b[Cout]"
            f" (rank 4/4/4/4/1): [{_show(x)}] / [{_show(weight)}] / [{_show(offset)}] /"
            f" [{_show(mask)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    channels_out, weight_in, kernel_h, kernel_w = weight
    if weight_in != channels_in:
        raise OpContractError(
            f"{where}: deform_conv2d weight should be [Cout, Cin, Kh, Kw]"
            f" (Cin = {channels_in.to_dim()}) but is [{_show(weight)}] (x is [{_show(x)}])"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: deform_conv2d bias length {bias[0].to_dim()} differs from"
            f" the output channels {channels_out.to_dim()}"
        )
    for name, kernel in (("Kh", kernel_h), ("Kw", kernel_w)):
        if not kernel.is_const:
            raise OpContractError(
                f"{where}: deform_conv2d kernel length {name} {kernel.to_dim()} is symbolic"
                " (the output length cannot be determined)"
            )
    # No stride / dilation field = fixed at 1, so the conv family's general form is shared with
    # those values.
    height_out = _conv_length("deform_conv2d H", x[2], kernel_h.offset, 1, padding[0], 1, where)
    width_out = _conv_length("deform_conv2d W", x[3], kernel_w.offset, 1, padding[1], 1, where)
    taps = kernel_h.offset * kernel_w.offset
    for name, shape, channels in (("offset", offset, 2 * taps), ("mask", mask, taps)):
        expected = [x[0], Extent(coeff=0, sym=None, offset=channels), height_out, width_out]
        if shape != expected:
            raise OpContractError(
                f"{where}: deform_conv2d {name} should be [{_show(expected)}]"
                f" (offset_groups = 1) but is [{_show(shape)}]"
            )
    return [x[0], channels_out, height_out, width_out]


def _gru_scan(contract: OpContract, ins: list[list[Extent]], where: str) -> list[Extent]:
    """The output shape of gru_scan / gru_scan_reverse (extended composite layer, the old layer 2;
    ADR 0056).

    `gi[T,N,3H]` / `h0[N,H]` / `W_hh[3H,H]` / `b_hh[3H]` -> `y[T,N,H]`. **The time axis T passes
    through as a symbol** (the reason this op exists — the decomposed form is unrolled T times and
    loses the symbol).

    MUST: the single source of truth for the hidden width is **h0's last dimension**. gi / W_hh /
    b_hh are only matched against it; taking the same fact from two places would allow a form where
    a 3H / H mix-up slips through.
    MUST: H is static (a symbolic H cannot build 3·H in the dimension language — the same narrowing
    as the channel axes of the conv family). Batch N and time T may be symbolic (they are only
    matched by structural equality and need no arithmetic).
    """
    gi, initial, weight, bias = ins
    if len(gi) != 3 or len(initial) != 2 or len(weight) != 2 or len(bias) != 1:
        raise OpContractError(
            f"{where}: {contract.name} expects gi[T,N,3H] / h0[N,H] / W_hh[3H,H] / b_hh[3H]"
            f" (rank 3/2/2/1): [{_show(gi)}] / [{_show(initial)}] / [{_show(weight)}] /"
            f" [{_show(bias)}]"
        )
    hidden = initial[1]
    if not hidden.is_const:
        raise OpContractError(
            f"{where}: {contract.name} hidden width {hidden.to_dim()} is symbolic"
            " (3·H cannot be built as a dimension expression)"
        )
    if gi[1] != initial[0]:
        raise OpContractError(
            f"{where}: {contract.name} batch differs between gi [{_show(gi)}] and"
            f" h0 [{_show(initial)}]"
        )
    gates = Extent(coeff=0, sym=None, offset=3 * hidden.offset)
    if gi[2] != gates:
        raise OpContractError(
            f"{where}: {contract.name} gi last dimension {gi[2].to_dim()} is not"
            f" 3·H = {gates.offset} (the gates are r / z / n, three in all)"
        )
    if weight != [gates, hidden]:
        raise OpContractError(
            f"{where}: {contract.name} W_hh should be [{_show([gates, hidden])}] but is"
            f" [{_show(weight)}]"
        )
    if bias[0] != gates:
        raise OpContractError(
            f"{where}: {contract.name} b_hh length {bias[0].to_dim()} differs from"
            f" 3·H = {gates.offset}"
        )
    return [gi[0], gi[1], hidden]


def _gated_delta_scan(contract: OpContract, ins: list[list[Extent]], where: str) -> list[Extent]:
    """The output shape of gated_delta_scan (extended composite layer — ADR 0135, the value form).

    `q[B,T,Hk,dk]` / `k[B,T,Hk,dk]` / `v[B,T,Hv,dv]` / `g[B,T,Hv]` / `beta[B,T,Hv]` ->
    `o[B,T,Hv,dv]`. **The length axis T passes through as a symbol** (the reason this op exists —
    upstream's Python loops specialise it).

    MUST: B and T come from q; k / v / g / beta are **only matched** against them (k structurally
    equal to q, v's B / T, g and beta exactly `[B,T,Hv]` with Hv from v).
    MUST: the head broadcast is divisible (`Hv % Hk == 0`, `Hv ≥ Hk ≥ 1`), and the head counts
    must be constant to decide it (a symbolic head count fails as "the declaration alone cannot
    decide" — the same discipline as attention's GQA rule). `Hk ≥ 1` comes first (Python's `%`
    would raise ZeroDivisionError instead of the contract error the TS side gives).
    MUST: a constant axis of length 0 is rejected (B, T, dk, dv ≥ 1 — the semantics start at T ≥ 1).
    A symbolic B / T depends on the binding, so the TS side checks it before execution (the module
    docstring's split). The head dims may be symbolic: nothing is computed from them.
    """
    q, k, v, g, beta = ins
    show = (
        f"q [{_show(q)}] / k [{_show(k)}] / v [{_show(v)}] / g [{_show(g)}] / beta [{_show(beta)}]"
    )
    if len(q) != 4 or len(k) != 4 or len(v) != 4 or len(g) != 3 or len(beta) != 3:
        raise OpContractError(
            f"{where}: {contract.name} expects q[B,T,Hk,dk] / k[B,T,Hk,dk] / v[B,T,Hv,dv] /"
            f" g[B,T,Hv] / beta[B,T,Hv] (rank 4/4/4/3/3): {show}"
        )
    batch, time, key_heads, key_dim = q
    value_heads, value_dim = v[2], v[3]
    if k != q:
        raise OpContractError(f"{where}: {contract.name} k must have q's shape [B,T,Hk,dk]: {show}")
    if v[0] != batch or v[1] != time:
        raise OpContractError(f"{where}: {contract.name} B / T of v differ from q: {show}")
    expected = [batch, time, value_heads]
    for name, shape in (("g", g), ("beta", beta)):
        if shape != expected:
            raise OpContractError(
                f"{where}: {contract.name} {name} should be [B,T,Hv] = [{_show(expected)}]: {show}"
            )
    if any(extent.is_value(0) for extent in (batch, time, key_dim, value_dim)):
        raise OpContractError(
            f"{where}: {contract.name} needs B, T, dk and dv of at least 1: {show}"
        )
    if key_heads.is_const and key_heads.offset < 1:
        raise OpContractError(
            f"{where}: {contract.name} Hv {value_heads.to_dim()} is not a positive integer multiple"
            f" of Hk {key_heads.to_dim()} (the head broadcast needs Hv % Hk == 0 and Hv ≥ Hk ≥ 1"
            f" — ADR 0135 decision 1): {show}"
        )
    if not key_heads.is_const or not value_heads.is_const:
        raise OpContractError(
            f"{where}: {contract.name} Hk {key_heads.to_dim()} and Hv {value_heads.to_dim()}:"
            f" the declaration alone cannot decide whether they broadcast divisibly: {show}"
        )
    if value_heads.offset < key_heads.offset or value_heads.offset % key_heads.offset != 0:
        raise OpContractError(
            f"{where}: {contract.name} Hv {value_heads.offset} is not a positive integer multiple"
            f" of Hk {key_heads.offset} (the head broadcast needs Hv % Hk == 0 and Hv ≥ Hk ≥ 1"
            f" — ADR 0135 decision 1): {show}"
        )
    return [batch, time, value_heads, value_dim]


def _upsample_bilinear2d(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    """The output shape of upsample_bilinear2d (Core ATen layer, the old layer 1;
    align_corners=True only).

    The output spatial size is set directly by attrs `output_size` (the input H/W only act as the
    denominator of the scale factor), so **the input spatial axes may be symbolic**. B / C pass
    through.

    MUST: spatial axes of length 0 do not pass. The scale is `(in - 1) / (out - 1)`, so in = 0 gives
    a negative scale and reads go outside the input (a symbolic length depends on the binding, so
    the TS side checks it before execution).
    """
    x = ins[0]
    output_size = upsample_bilinear2d_attrs(attrs, where)
    if len(x) != 4:
        raise OpContractError(
            f"{where}: upsample_bilinear2d accepts only x[B,C,H,W] (rank 4): [{_show(x)}]"
        )
    for name, axis in (("H", 2), ("W", 3)):
        if x[axis].is_value(0):
            raise OpContractError(
                f"{where}: upsample_bilinear2d cannot interpolate a spatial axis {name} of"
                f" length 0 ([{_show(x)}])"
            )
    return [
        x[0],
        x[1],
        Extent(coeff=0, sym=None, offset=output_size[0]),
        Extent(coeff=0, sym=None, offset=output_size[1]),
    ]


# ---- Matching the whole graph ---------------------------------------------


def declared_shape(graph: IrGraph, name: str) -> list[IrDim]:
    """Value name → declared shape (inputs from inputs[], everything else from values{})."""
    for spec in graph.inputs:
        if spec.name == name:
            return spec.shape
    value = graph.values.get(name)
    if value is None:
        raise OpContractError(f"Value '{name}' has no dtype/shape declaration")
    return value.shape


def assert_graph_shapes(graph: IrGraph) -> None:
    """Computes the output shapes of every node from the contracts and matches them against the
    declarations (from torch's meta).

    MUST: do not treat the declaration as correct. Trusting the shapes torch attached as they are
    would write out IR that disagrees with the contract rules (gather's matching leading
    dimensions, conv1d's output length, reshape's element count and so on) "as declared", and it
    would not surface until Session construction in the runtime.
    """
    # The concrete slot shapes including the capacity are handed out from the one declaration
    # (graph.states) — on the TS side this is the layer that passes the shapes the
    # GenerationContext resolved (ADR 0067 decision 4).
    state_shapes = {name: slot.shape for name, slot in graph.states.items()}
    for index, node in enumerate(graph.nodes):
        where = f"graph.nodes[{index}] ({node.op})"
        contract = resolve_op_contract(node.op)
        ins = [declared_shape(graph, name) for name in node.ins]
        declared_outputs = [declared_shape(graph, name) for name in node.outs]
        # NOTE: the two ops that require a target shape (reshape / expand) are single-output, so
        # `declared` gets slot 0 only (multi-output op shapes are derived from the inputs and
        # attrs — the same as plan.ts on the TS side). An effect op with zero outputs (ADR 0067
        # decision 5) has no slot 0, so None is passed.
        computed = compute_output_shape(
            contract,
            ins,
            where,
            declared=declared_outputs[0] if declared_outputs else None,
            attrs=node.attrs,
            states=node.states,
            state_shapes=state_shapes,
        )
        # MUST: check here that the list lengths match (if the counts split, the second output
        # would never be matched against anything).
        if len(computed) != len(declared_outputs):
            raise OpContractError(
                f"{where}: output shape computation gave {len(computed)}"
                f" (declared {len(declared_outputs)})"
            )
        for slot, declared in enumerate(declared_outputs):
            if extents(computed[slot], where) != extents(declared, where):
                raise OpContractError(
                    f"{where}: output '{node.outs[slot]}' computed shape"
                    f" [{','.join(str(dim) for dim in computed[slot])}] does not match"
                    f" declared [{','.join(str(dim) for dim in declared)}]"
                )
