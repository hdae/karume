"""aten op → {@link Emitted} table (the mapping onto the IR v1 op vocabulary).

The table itself of **step 2 (conversion by the aten → IR op table)** of the three steps the
module docstring of convert.py describes. The engine for graph traversal, constant folding and CSE
is in convert.py.

Only this side grows each time a model family is added; the engine side barely changes — the
purpose of the split is to separate that growth point. {@link Emitted} and the predicates both sides
use also live here (the import direction is one way only, convert → aten_handlers).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

# MUST: `torch.ops.torchvision.*` cannot be looked up without loading torchvision's C++ library
# (the handler keys cannot be written). Making it a soft dependency with a conditional import would
# split environments into "one that fails deform_conv2d as an unsupported op" and "one that passes",
# so it is imported plainly as a base dependency (ADR 0055 decision 7).
import torchvision  # noqa: F401  -- registering torch.ops.torchvision is the side effect
from torch.fx import Node

# MUST: `torch.ops.karume.*` cannot be looked up without importing the custom op registration
# module (the handler keys cannot be written). Registration is a process-wide global side effect,
# and the "zero side effects in every module" invariant does not apply to the Python exporter
# (ADR 0056 decision 6).
import karume.custom_ops  # noqa: F401  -- registering torch.ops.karume is the side effect
from karume.extents import extent_key
from karume.normalize import SAFE_SOFTMAX_META
from karume.ops import (
    EXPERT_LINEAR_OP,
    GATED_DELTA_SCAN_OP,
    GRU_SCAN_OP,
    GRU_SCAN_REVERSE_OP,
    STRIDED_RANK,
    static_quantize_scale,
)

aten = torch.ops.aten

#: Semantic dtypes (docs/ir-v2.md "Values and types"). A torch dtype not listed here fails loudly.
#:
#: MUST: torch's default integer int64 is **declared as i32** (the boundary normalization of ADR
#: 0009). WebGPU has no 64-bit integer buffers, so the conversion point into the world without
#: 64 bits is fixed at one place in the exporter. normalize_boundary_tensor holds the check on the
#: value side.
DTYPE_NAMES = {
    torch.float32: "f32",
    torch.int32: "i32",
    torch.int64: "i32",
    torch.bool: "bool",
}


class Emitted(NamedTuple):
    """The IR node specification a handler declares. ins is the expected number of tensor inputs.

    attrs grow as one set with the attrs schema of the op contract table (ops.py) — a key the
    contract table does not declare fails loudly in verify (ADR 0012).

    `synth_consts` are synthesis instructions **that normalize fused ops with optional slots to a
    fixed arity** (ADR 0015's conv zero bias / ADR 0016's linear zero bias, layer_norm's
    ones/zeros affine, rms_norm's ones weight). Each element is `(fill value, length)` and they are
    **appended in declaration order**. The handler does not call `_add_const` because creating
    initializers belongs to the Converter's state (deduplication, value-name numbering) — handlers
    stay pure functions.

    The synthesis is mathematically an identity because `+0` / `×1` are exact identities in f32
    (ADR 0015). The purpose is to keep arity branches out of both kernels and contracts, which is
    different from "silently adding forms not seen in measurements" (synthesized constants travel
    the same path with the same naming as ordinary initializers).
    """

    op: str
    ins: int
    attrs: Mapping[str, Any] = MappingProxyType({})
    synth_consts: tuple[tuple[float, int], ...] = ()


def _expect(cond: bool, node: Node, why: str) -> None:
    if not cond:
        raise NotImplementedError(f"{node.target}: {why} (args={node.args}, kwargs={node.kwargs})")


def _has_free_symbols(value: Any) -> bool:
    """Whether the value has a SymInt / symbolic shape (a structural check that does not trigger
    guard evaluation)."""
    if isinstance(value, torch.SymInt):
        return bool(value.node.expr.free_symbols)
    if isinstance(value, torch.Tensor):
        return any(_has_free_symbols(d) for d in value.shape)
    if isinstance(value, (list, tuple)):
        return any(_has_free_symbols(item) for item in value)
    return False


# ---- aten op table -------------------------------------------------------


def _simple(op: str, ins: int):
    def handler(node: Node) -> Emitted:
        return Emitted(op, ins)

    return handler


def _h_binary(op: str):
    """Binary elementwise (torch-style right-aligned broadcast)."""

    def handler(node: Node) -> Emitted:
        _expect(node.kwargs.get("alpha", 1) == 1, node, "alpha != 1 is not supported")
        _expect(
            all(isinstance(arg, Node) for arg in node.args[:2]),
            node,
            "scalar operands are not supported (binary op contracts have empty attrs — no field"
            " to carry a scalar)",
        )
        return Emitted(op, 2)

    return handler


#: `aten.gelu`'s `approximate` → IR op. The contract has empty attrs, so the approximation kind is
#: a **separate op** (changing the approximation under the same op name would create a "values
#: change silently" branch outside the contract).
_GELU_OPS = {"none": "gelu", "tanh": "gelu_tanh"}


def _h_gelu(node: Node) -> Emitted:
    approximate = node.kwargs.get("approximate", "none")
    _expect(
        approximate in _GELU_OPS, node, f"gelu with approximate={approximate!r} is not supported"
    )
    return Emitted(_GELU_OPS[approximate], 1)


def _h_sum(node: Node) -> Emitted:
    src = node.args[0].meta["val"]
    # Fields are read through `_arg_or_kwarg` — even when overload dispatch puts `dim` in kwargs,
    # the form does not fall to a bare index (IndexError) but joins the diagnostic discipline
    # below (ADR 0005) — MUST.
    dims = _arg_or_kwarg(node, 1, "dim", None)
    _expect(
        dims is not None,
        node,
        "sum over all dimensions is not supported (reduce takes one axis only)",
    )
    keepdim = bool(_arg_or_kwarg(node, 2, "keepdim", False))
    dtype = _arg_or_kwarg(node, 3, "dtype", None)
    # Only one axis is accepted (folding several axes into one node would make the reduction
    # order unreadable from the IR — the execution side only reduces one axis at a time).
    # keepdim / dtype remain unsupported as before.
    _expect(
        len(dims) == 1,
        node,
        f"multiple axes {list(dims)} are not supported for sum (one axis at a time)",
    )
    _expect(not keepdim, node, "sum with keepdim=True is not supported")
    _expect(dtype is None, node, "sum with a dtype is not supported")
    return Emitted("sum", 1, {"dim": _normalized_dims(node, src.dim(), dims)[0]})


#: kwargs `_to_copy` accepts. A form with anything other than dtype (layout / device /
#: memory_format) does more than "a dtype conversion", so it fails loudly instead of becoming a
#: cast.
_TO_COPY_KWARGS = frozenset({"dtype"})


def _h_to_copy(node: Node) -> Emitted:
    """aten._to_copy → cast (ADR 0009). The target comes from the output dtype in meta.

    MUST: meta["val"].dtype is the source of truth, not the dtype in kwargs — i64 is declared as
    i32 by the boundary normalization, so naming it after the torch dtype would disagree with the
    declaration.
    """
    extra = sorted(set(node.kwargs) - _TO_COPY_KWARGS)
    _expect(not extra, node, f"_to_copy with kwargs {extra} other than dtype is not supported")
    val = node.meta["val"]
    to = DTYPE_NAMES.get(val.dtype)
    if to is None:
        raise NotImplementedError(
            f"{node.target}: target dtype {val.dtype} is not in the IR v1 semantic dtype vocabulary"
        )
    return Emitted("cast", 1, {"to": to})


def _h_bitwise_not(node: Node) -> Emitted:
    """aten.bitwise_not → bitwise_not (negation of bool only).

    Integer bitwise_not (~x) means something different from negating a bool. The contract is bool
    only, so integer inputs fail instead of being silently read as negation.
    """
    src = node.args[0].meta["val"]
    _expect(
        src.dtype is torch.bool,
        node,
        f"bitwise_not of dtype {src.dtype} is not supported (bool only)",
    )
    return Emitted("bitwise_not", 1)


def _h_reshape(node: Node) -> Emitted:
    """view / squeeze(dims) / unsqueeze → reshape (ADR 0011).

    The target shape is not in attrs but is **the declared output shape** (emitted into
    `values{}` from `meta["val"]`), so the argument size list (which contains `-1` and SymInts)
    does not go into the IR. This only checks that it is "a meta operation that does not change
    element order" — the runtime's contract check verifies the element counts match.
    """
    _expect(not node.kwargs, node, f"forms with kwargs {sorted(node.kwargs)} are not supported")
    return Emitted("reshape", 1)


def _normalized_dims(node: Node, rank: int, dims: Sequence[Any]) -> list[int]:
    """Normalize torch's negative axis notation to non-negative (IR attrs are non-negative only —
    the contract table)."""
    out: list[int] = []
    for dim in dims:
        _expect(isinstance(dim, int), node, f"axis {dim!r} is not an integer")
        index = int(dim)
        _expect(-rank <= index < rank, node, f"axis {index} is out of range for rank {rank}")
        out.append(index % rank)
    return out


def _h_permute(node: Node) -> Emitted:
    """aten.permute → permute (attrs `dims`). The materializing copy is one strided kernel
    family."""
    _expect(not node.kwargs, node, f"permute with kwargs {sorted(node.kwargs)} is not supported")
    src = node.args[0].meta["val"]
    rank = src.dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"permute of rank {rank} cannot run on the strided kernel (rank ≤ {STRIDED_RANK})",
    )
    dims = _normalized_dims(node, rank, node.args[1])
    _expect(len(dims) == rank, node, f"dims {dims} differs from rank {rank}")
    return Emitted("permute", 1, {"dims": dims})


def _h_expand(node: Node) -> Emitted:
    """aten.expand → expand (ADR 0011). The target shape is the declared output shape; no attrs.

    The argument size list contains `-1` (keep) and SymInts, but the resolved shape is in
    `meta["val"]`, so that is the source of truth. The check "only dimensions of length 1 are
    expanded" belongs to the runtime's contract (on the numeric shape after binding resolution).
    """
    extra = sorted(set(node.kwargs) - {"implicit"})
    _expect(not extra, node, f"expand with kwargs {extra} is not supported")
    _expect(
        not node.kwargs.get("implicit", False), node, "expand with implicit=True is not supported"
    )
    rank = node.meta["val"].dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"expand of rank {rank} cannot run on the strided kernel (rank ≤ {STRIDED_RANK})",
    )
    return Emitted("expand", 1)


def _static_extent(node: Node, tensor: torch.Tensor, dim: int, op: str) -> int:
    """**The target axis of `slice` / `flip` is static** (ADR 0014) — a symbolic dimension fails
    loudly.

    Cutting a symbolic axis belongs to `sym_prefix_slice`, and `slice` is static only (no overlap).
    flip's axes are all static in measurements. Only `cat`'s concatenation axis was relaxed, in
    ADR 0046 (`_cat_axis`).
    """
    extent = tensor.shape[dim]
    _expect(
        not _has_free_symbols(extent),
        node,
        f"{op} axis {dim} is a symbolic dimension (the target axis must be static)",
    )
    return int(extent)


def _static_size_or_fail(node: Node, dim: Any, what: str) -> int:
    """A dimension used as the length of a synthesized constant must be static (a symbol fails
    loudly).

    Synthesizing a slot of symbolic length would need "an initializer whose length is decided at
    run time", a concept that does not exist in IR v1. In measurements it is a weight dimension and
    always static, but this stops if that breaks.
    """
    _expect(
        not _has_free_symbols(dim),
        node,
        f"{what} is a symbolic dimension (the length of the synthesized constant is undetermined)",
    )
    return int(dim)


def _h_slice(node: Node) -> Emitted:
    """aten.slice.Tensor → slice (attrs `dim` / `start` / `end`, **static axis, static range**).

    MUST: `step != 1` fails (IR slice is a contiguous window only — strided reading cannot be
    expressed by the strided family's single variable word).
    MUST: negative indices and omissions (`None` / int64 max) are **clamped to the axis length
    here** (IR attrs are non-negative only — the same boundary normalization as permute's dims).
    MUST: a slice on a symbolic axis fails (it belongs to sym_prefix_slice — no overlap).
    """
    extra = sorted(set(node.kwargs) - {"dim", "start", "end", "step"})
    _expect(not extra, node, f"slice with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    rank = src.dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"slice of rank {rank} cannot run on the strided kernel (rank ≤ {STRIDED_RANK})",
    )
    dim = _normalized_dims(node, rank, [_arg_or_kwarg(node, 1, "dim", 0)])[0]
    size = _static_extent(node, src, dim, "slice")
    step = _arg_or_kwarg(node, 4, "step", 1)
    _expect(step == 1, node, f"slice with step={step!r} is not supported (contiguous windows only)")
    bounds: list[int] = []
    for index, (key, default) in enumerate((("start", 0), ("end", size))):
        value = _arg_or_kwarg(node, 2 + index, key, None)
        if value is None:
            bounds.append(default)
            continue
        _expect(
            not _has_free_symbols(value) and isinstance(value, (int, torch.SymInt)),
            node,
            f"slice with {key}={value!r} is not supported (static indices only)",
        )
        # Negative indices count from the end; out of range is clamped to the axis length, as in
        # torch.
        bound = int(value)
        bounds.append(min(max(bound + size if bound < 0 else bound, 0), size))
    start, end = bounds
    # torch collapses start > end to an empty slice, but the IR has no representation for it (it
    # is not silently made length 0).
    _expect(start <= end, node, f"slice with start={start} > end={end} is not supported")
    return Emitted("slice", 1, {"dim": dim, "start": start, "end": end})


def _cat_axis(node: Node, tensors: Sequence[Node], dim: int) -> None:
    """cat's concatenation axis is limited to <a constant> or <a linear expression in **one**
    symbol> (ADR 0046).

    With one symbol, the sum `Σ(coeff_i·sym + offset_i)` fits the canonical grammar
    `coeff·sym+offset` as is (`S`+1519 → `S+1519`, `S`+`S` → `2S`).
    MUST: a concatenation that mixes different symbols fails — the sum does not fit the grammar of
    one symbol per dimension. torch.export also refuses to declare a sum of several symbols, so
    widening acceptance would not let in an unrepresentable form, but without stopping here the
    discovery is delayed until `_sym_parts`, which writes the output shape, and the cause is
    unreadable.
    """
    symbols: dict[str, int] = {}
    for index, item in enumerate(tensors):
        extent = item.meta["val"].shape[dim]
        if not _has_free_symbols(extent):
            continue
        for symbol in extent.node.expr.free_symbols:
            symbols.setdefault(symbol.name, index)
    _expect(
        len(symbols) <= 1,
        node,
        f"cat that mixes different symbols on concatenation axis {dim} is not supported"
        f" (symbols per input: {symbols}) — the sum does not fit the dimension language"
        " coeff·sym+offset",
    )


def _h_cat(node: Node) -> Emitted:
    """aten.cat.default → cat (attrs `dim`, **variable arity**).

    MUST: a cat with 1 input fails (an identity copy, below the contract's arity lower bound 2).
    MUST: at most one kind of symbol on the concatenation axis (ADR 0046 — `_cat_axis`).
    """
    _expect(not node.kwargs, node, f"cat with kwargs {sorted(node.kwargs)} is not supported")
    tensors = node.args[0]
    _expect(
        isinstance(tensors, (list, tuple)) and all(isinstance(item, Node) for item in tensors),
        node,
        "cat whose first argument is not a list of tensors is not supported",
    )
    _expect(len(tensors) >= 2, node, f"cat with {len(tensors)} inputs is not supported (2 or more)")
    first = tensors[0].meta["val"]
    rank = first.dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"cat of rank {rank} cannot run on the strided kernel (rank ≤ {STRIDED_RANK})",
    )
    dim = _normalized_dims(node, rank, [_arg_or_kwarg(node, 1, "dim", 0)])[0]
    _cat_axis(node, tensors, dim)
    return Emitted("cat", len(tensors), {"dim": dim})


def _h_constant_pad_nd(node: Node) -> Emitted:
    """aten.constant_pad_nd → pad (attrs `left` / `right`, **last dimension, constant 0 only**).

    MUST: a fill value other than 0 and a pad wider than the last dimension fail (no field in the
    contract = no representation).
    MUST: negative widths (truncation) fail too — that means slice, not pad, and letting it through
    would allow the same form in two ops and make the output length computation negative.
    """
    _expect(not node.kwargs, node, f"pad with kwargs {sorted(node.kwargs)} is not supported")
    pad = node.args[1]
    _expect(
        isinstance(pad, (list, tuple)) and len(pad) == 2,
        node,
        f"pad={pad!r} is not supported (one [left, right] pair on the last dimension only)",
    )
    _expect(not _has_free_symbols(pad), node, f"pad={pad!r} containing symbols is not supported")
    widths = [int(width) for width in pad]
    _expect(
        all(width >= 0 for width in widths),
        node,
        f"negative pad={pad!r} is not supported (truncation)",
    )
    value = _arg_or_kwarg(node, 2, "value", 0)
    _expect(
        isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0,
        node,
        f"constant_pad_nd with value={value!r} is not supported (constant 0 only)",
    )
    return Emitted("pad", 1, {"left": widths[0], "right": widths[1]})


def _h_flip(node: Node) -> Emitted:
    """aten.flip → flip (attrs `dim`, **one static axis**).

    MUST: a flip over several axes fails (IR flip is one axis only — the measurements in flow and
    sdp are both one axis). Silently collapsing several axes into one leaves axes unflipped.
    """
    _expect(not node.kwargs, node, f"flip with kwargs {sorted(node.kwargs)} is not supported")
    src = node.args[0].meta["val"]
    dims = node.args[1]
    _expect(
        isinstance(dims, (list, tuple)) and len(dims) == 1,
        node,
        f"flip with dims={dims!r} is not supported (one axis only)",
    )
    dim = _normalized_dims(node, src.dim(), dims)[0]
    _static_extent(node, src, dim, "flip")
    return Emitted("flip", 1, {"dim": dim})


def _h_bmm(node: Node) -> Emitted:
    """aten.bmm → bmm (rank-3 batched matmul).

    MUST: anything other than rank-3 fails. The runtime's bmm is rank-3 only, and rank-2 belongs to
    matmul (the contract table). Letting it through here gives "export is green, only the browser
    fails".
    """
    for position, arg in enumerate(node.args[:2]):
        rank = arg.meta["val"].dim()
        _expect(
            rank == 3,
            node,
            f"bmm with input {position} of rank {rank} is not supported (rank-3 only)",
        )
    return Emitted("bmm", 2)


def _h_gather(node: Node) -> Emitted:
    """aten.gather → gather (**last dimension fixed**, no attrs).

    MUST: forms whose dim is not the last dimension fail (the contract fixes the last dimension —
    the same narrowing as softmax in ADR 0012; general dims are widened once measurements require
    them).
    MUST: `sparse_grad=True` fails (only the default False is accepted). A sparse gradient
    representation does not change the meaning of forward, but widening acceptance "because it
    seemingly has no effect on forward" lets the next argument that does have an effect slip
    through for the same reason.
    MUST: **leading dimensions must match exactly between src and index** (the IR contract —
    equivalent to computeOutputShape in packages/runtime/src/ops/shapes.ts). torch allows
    `index.size(d) <= src.size(d)`, but the runtime's kernel reads with `row = i / J` on the premise
    "leading dimensions match = the row index is shared". Passing it after checking only the rank
    makes export green while only the browser fails with a contract violation (or rows silently
    shift).
    """
    src = node.args[0].meta["val"]
    rank = src.dim()
    dim = _normalized_dims(node, rank, [node.args[1]])[0]
    _expect(
        dim == rank - 1,
        node,
        f"gather on a dimension other than the last (dim={dim} / rank={rank}) is not supported",
    )
    extra = sorted(set(node.kwargs) - {"sparse_grad"})
    _expect(not extra, node, f"gather with kwargs {extra} is not supported")
    sparse_grad = node.args[3] if len(node.args) > 3 else node.kwargs.get("sparse_grad", False)
    _expect(not sparse_grad, node, "gather with sparse_grad=True is not supported")
    _expect(len(node.args) <= 4, node, f"gather with {len(node.args)} arguments is not supported")
    index = node.args[2].meta["val"]
    _expect(
        index.dim() == rank,
        node,
        f"gather whose index rank {index.dim()} differs from src rank {rank} is not supported",
    )
    for axis in range(rank - 1):
        _expect(
            extent_key(src.shape[axis]) == extent_key(index.shape[axis]),
            node,
            f"gather whose leading dimension {axis} does not match between src {list(src.shape)}"
            f" and index {list(index.shape)} is not supported (the IR contract requires an exact"
            " match — narrower than torch's index <= src)",
        )
    return Emitted("gather", 2)


def _h_where(node: Node) -> Emitted:
    """aten.where.self → where (ternary elementwise).

    MUST: only forms whose 3 arguments are all tensors are accepted. torch also allows scalar
    arguments (`where(cond, x, 0.0)`), but the contract has empty attrs and fixed arity 3, so there
    is no field to carry a scalar (the same narrowing as the binary ops' `_h_binary` — if needed,
    a scalar-promotion pass in normalize removes it).
    """
    _expect(not node.kwargs, node, f"where with kwargs {sorted(node.kwargs)} is not supported")
    _expect(
        len(node.args) == 3 and all(isinstance(arg, Node) for arg in node.args),
        node,
        "where with a scalar operand is not supported (the contract has empty attrs and fixed"
        " arity 3)",
    )
    return Emitted("where", 3)


def _h_clamp(node: Node) -> Emitted:
    """aten.clamp → clamp (both sides) / **clamp_min** (lower side only — ADR 0017).

    MUST: the missing side is not filled with ±the largest finite value. That would "silently run
    a form without a representation as a different form", so the lower-only form goes to a
    **separate op** (`clamp_min`, added to the vocabulary in ADR 0017). The upper-only form is not
    in the vocabulary and fails — when needed, add `clamp_max` the same way.
    MUST: `min <= max` is checked by the contract table (the shapes layer) — WGSL's clamp is
    undefined when inverted.
    """
    extra = sorted(set(node.kwargs) - {"min", "max"})
    _expect(not extra, node, f"clamp with kwargs {extra} is not supported")
    bounds = [_arg_or_kwarg(node, 1, "min", None), _arg_or_kwarg(node, 2, "max", None)]
    for name, value in zip(("min", "max"), bounds, strict=True):
        if value is None:
            continue
        _expect(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            node,
            f"clamp with {name}={value!r} is not supported (bounds are finite scalars)",
        )
        _expect(
            math.isfinite(float(value)),
            node,
            f"non-finite {name}={value!r} is not supported by IR v1",
        )
    raw_min, raw_max = bounds
    _expect(
        raw_min is not None,
        node,
        "clamp with only an upper bound is not supported (no clamp_max in the vocabulary)",
    )
    if raw_max is None:
        return Emitted("clamp_min", 1, {"min": float(raw_min)})
    minimum, maximum = float(raw_min), float(raw_max)
    _expect(minimum <= maximum, node, f"clamp with min={minimum} > max={maximum} is not supported")
    return Emitted("clamp", 1, {"min": minimum, "max": maximum})


def _h_leaky_relu(node: Node) -> Emitted:
    """aten.leaky_relu → leaky_relu (attrs `negative_slope`).

    MUST: torch's default (0.01) is **read here and put into attrs**. dec mixes 0.1 (ups /
    ResBlock) and 0.01 (the last stage, with the positional argument omitted), so leaving it out of
    the IR and relying on the runtime's default would silently get one of them wrong (ADR 0015).
    The "never fill in defaults" rule is about the IR contract side; making torch's semantics
    explicit at the boundary is the responsibility of the opposite side.
    """
    extra = sorted(set(node.kwargs) - {"negative_slope"})
    _expect(not extra, node, f"leaky_relu with kwargs {extra} is not supported")
    slope = _arg_or_kwarg(node, 1, "negative_slope", 0.01)
    _expect(
        isinstance(slope, (int, float)) and not isinstance(slope, bool),
        node,
        f"negative_slope={slope!r} is not a scalar",
    )
    number = float(slope)
    _expect(
        math.isfinite(number),
        node,
        f"non-finite negative_slope={slope!r} is not supported by IR v1",
    )
    return Emitted("leaky_relu", 1, {"negative_slope": number})


def _h_compare_scalar(op: str):
    """aten.ge/le/gt.Scalar → comparison op (attrs `value`, the output is bool)."""

    def handler(node: Node) -> Emitted:
        _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} are not supported for {op}")
        _expect(len(node.args) == 2, node, f"{len(node.args)} arguments are not supported for {op}")
        value = node.args[1]
        _expect(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            node,
            f"value={value!r} is not a scalar (the Tensor form is a separate op)",
        )
        number = float(value)
        _expect(
            math.isfinite(number), node, f"non-finite value={value!r} is not supported by IR v1"
        )
        src = node.args[0].meta["val"]
        _expect(
            src.dtype is torch.float32,
            node,
            f"dtype {src.dtype} is not supported for {op} (f32 only)",
        )
        return Emitted(op, 1, {"value": number})

    return handler


def _h_ge_tensor(node: Node) -> Emitted:
    """aten.ge.Tensor → ge (f32 × f32 → bool, right-aligned broadcast)."""
    _expect(not node.kwargs, node, f"ge with kwargs {sorted(node.kwargs)} is not supported")
    _expect(
        all(isinstance(arg, Node) for arg in node.args[:2]),
        node,
        "scalar operands are not supported (a scalar comparison is ge_scalar)",
    )
    return Emitted("ge", 2)


def _h_bitwise_and(node: Node) -> Emitted:
    """aten.bitwise_and.Tensor → bitwise_and (**logical AND of bools only**).

    Integer bitwise_and means something different from the logical AND of bools (the contract is
    bool only — the same narrowing as bitwise_not). Integer inputs fail instead of being silently
    read as a logical AND.
    """
    _expect(
        not node.kwargs, node, f"bitwise_and with kwargs {sorted(node.kwargs)} is not supported"
    )
    for position, arg in enumerate(node.args[:2]):
        _expect(
            isinstance(arg, Node),
            node,
            f"bitwise_and with a scalar input {position} is not supported",
        )
        _expect(
            arg.meta["val"].dtype is torch.bool,
            node,
            f"bitwise_and of dtype {arg.meta['val'].dtype} is not supported (bool only)",
        )
    return Emitted("bitwise_and", 2)


def _h_cumsum(node: Node) -> Emitted:
    """aten.cumsum → cumsum (attrs `dim`, **last dimension only**).

    MUST: anything other than the last dimension fails (the same narrowing as softmax / gather).
    The runtime scans sequentially with 1 invocation = 1 row and assumes the accumulation axis is
    contiguous.
    MUST: negative axis notation is normalized to non-negative at the boundary (IR attrs are
    non-negative only — the contract table).
    """
    extra = sorted(set(node.kwargs) - {"dtype"})
    _expect(not extra, node, f"cumsum with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    rank = src.dim()
    dim = _normalized_dims(node, rank, [node.args[1]])[0]
    _expect(
        dim == rank - 1,
        node,
        f"cumsum on a dimension other than the last (dim={dim} / rank={rank}) is not supported",
    )
    dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
    _expect(dtype is None, node, "cumsum with a dtype is not supported")
    return Emitted("cumsum", 1, {"dim": dim})


def _h_row_reduce(op: str):
    """aten.amax / aten.amin (one axis, no keepdim)."""

    def handler(node: Node) -> Emitted:
        src = node.args[0].meta["val"]
        # Read through `_arg_or_kwarg` like `_h_sum` — collapsing an omitted axis to `[]` would give
        # **a diagnostic that contradicts the facts**, "multiple axes [] are not supported for
        # {op}", so it is rejected as a reduction over all dimensions.
        dims = _arg_or_kwarg(node, 1, "dim", None)
        _expect(
            dims is not None,
            node,
            f"{op} over all dimensions is not supported (reduce takes one axis only)",
        )
        keepdim = bool(_arg_or_kwarg(node, 2, "keepdim", False))
        # Only one axis (the same narrowing as sum — the execution side only reduces one axis at a
        # time).
        _expect(
            len(dims) == 1,
            node,
            f"multiple axes {list(dims)} are not supported for {op} (one axis at a time)",
        )
        _expect(not keepdim, node, f"{op} with keepdim=True is not supported")
        return Emitted(op, 1, {"dim": _normalized_dims(node, src.dim(), dims)[0]})

    return handler


def _h_argmax(node: Node) -> Emitted:
    """aten.argmax → argmax (**only the fixed form: last dimension + keepdim=True** — ADR 0068
    decision 2).

    torch's schema is `argmax(Tensor self, int? dim=None, bool keepdim=False)`, and unlike the
    reduce family `dim` is **a single int** (optional). Negative axis notation is normalized to
    non-negative at the boundary (IR attrs are non-negative only — no attrs are emitted here, but
    it is used to match against the last dimension).

    MUST: every form that is not accepted fails loudly (no silent approximation):

    - `dim` omitted (argmax over all dimensions) — the output is a rank 0 scalar, a different thing
      from the rank-preserving contract.
    - `keepdim=False` — a form whose output rank drops by 1. The missing `keepdim` field in the
      contract itself declares "that form is not in the vocabulary" (the rule of ADR 0023 decision
      4). When needed, add it as a composition with reshape or as a separate op.
    - not the last dimension — the row kernel assumes the reduction axis is contiguous (the same
      narrowing as softmax / cumsum / gather).
    """
    src = node.args[0].meta["val"]
    rank = src.dim()
    # Fields are read through `_arg_or_kwarg` (even when overload dispatch puts it in kwargs, the
    # form does not fall to a bare index IndexError but joins the diagnostic discipline — the same
    # MUST as the reduce family).
    dim = _arg_or_kwarg(node, 1, "dim", None)
    _expect(
        dim is not None,
        node,
        "argmax over all dimensions is not supported (last dimension only)",
    )
    keepdim = bool(_arg_or_kwarg(node, 2, "keepdim", False))
    _expect(
        keepdim,
        node,
        "argmax with keepdim=False is not supported (only the rank-preserving fixed form)",
    )
    axis = _normalized_dims(node, rank, [dim])[0]
    _expect(
        axis == rank - 1,
        node,
        f"argmax on a dimension other than the last (dim={axis} / rank={rank}) is not supported",
    )
    return Emitted("argmax", 1)


# ---- Fused ops (ADR 0012 / ADR 0007's preserved list) ---------------------


def _arg_or_kwarg(node: Node, index: int, key: str, default: Any) -> Any:
    """Read the same field whether it comes as a positional argument or in kwargs (absorbs overload
    dispatch)."""
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(key, default)


def _h_linear(node: Node) -> Emitted:
    """aten.linear → linear (`x[…,in] × W[out,in] + b[out]`).

    MUST: no bias is **not rejected; a zero bias is synthesized** to normalize to arity 3 (ADR 0016
    — the same approach as conv). In Anima's measurements 698 of 711 linears have no bias, and an
    arity branch there would ripple through every stage: contract, kernel and shape layer. The
    synthesis is the exact identity `+0`.
    """
    _expect(not node.kwargs, node, f"linear with kwargs {sorted(node.kwargs)} is not supported")
    bias = node.args[2] if len(node.args) > 2 else None
    if isinstance(bias, Node):
        return Emitted("linear", 3)
    _expect(bias is None, node, f"linear bias={bias!r} is neither a tensor nor None")
    weight = node.args[1].meta["val"]
    _expect(
        weight.dim() == 2,
        node,
        f"linear weight of rank {weight.dim()} is not supported ([out,in])",
    )
    features_out = _static_size_or_fail(node, weight.shape[0], "linear output dimension")
    return Emitted("linear", 3, synth_consts=((0.0, features_out),))


def _h_layer_norm(node: Node) -> Emitted:
    """aten.layer_norm → layer_norm (attrs `normalized_shape` / `eps`).

    MUST: without preserving it, it becomes native_layer_norm (3 outputs) + getitem, and wiring
    the getitem of multi-output aten ops is deferred until sampling needs it (ADR 0068 addendum 2),
    so it cannot be taken (recon §5).
    MUST: the normalized axis is the single last dimension only. The affine is accepted in 3
    forms, "both", "neither" and "weight only", and missing slots are filled by synthesizing
    ones/zeros (ADR 0016). Other forms break the row kernel's premise.
    """
    _expect(not node.kwargs, node, f"layer_norm with kwargs {sorted(node.kwargs)} is not supported")
    src = node.args[0].meta["val"]
    normalized_shape = node.args[1]
    _expect(
        isinstance(normalized_shape, (list, tuple)) and len(normalized_shape) == 1,
        node,
        f"normalized_shape {normalized_shape!r} is not supported (the single last dimension only)",
    )
    extent = src.shape[-1]
    _expect(
        not _has_free_symbols(extent),
        node,
        "layer_norm whose normalized axis is a symbolic dimension is not supported (attrs hold"
        " static lengths only)",
    )
    dims = [int(normalized_shape[0])]
    _expect(
        dims[0] == int(extent),
        node,
        f"normalized_shape {dims} differs from the input's last dimension {int(extent)}",
    )
    weight = node.args[2] if len(node.args) > 2 else None
    bias = node.args[3] if len(node.args) > 3 else None
    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5
    _expect(
        math.isfinite(eps) and eps > 0,
        node,
        f"eps={eps} is not supported (finite positive numbers only)",
    )
    attrs = {"normalized_shape": dims, "eps": eps}
    if isinstance(weight, Node) and isinstance(bias, Node):
        return Emitted("layer_norm", 3, attrs)
    if isinstance(weight, Node):
        # ADR 0016: only the bias is synthesized as zeros to normalize to arity 3 (the exact
        # identity `+0`). Synthesis appends at the end, so weight stays in its real slot
        # (ModernBERT's `norm_bias=false`).
        return Emitted("layer_norm", 3, attrs, synth_consts=((0.0, dims[0]),))
    # MUST: the **bias-only** form fails. Synthesis "appends in order", so letting through a form
    # with bias but no weight would slide the bias into the weight's slot (the element counts
    # match, so it passes the shape check too — a silent wrong value). The reverse (weight only)
    # has no such risk.
    _expect(
        bias is None,
        node,
        f"layer_norm with only a bias and no weight (bias={bias!r}) is not supported",
    )
    # ADR 0016: normalize to arity 3 by synthesizing ones/zeros (the exact identity `×1 +0`).
    return Emitted("layer_norm", 3, attrs, synth_consts=((1.0, dims[0]), (0.0, dims[0])))


def _h_group_norm(node: Node) -> Emitted:
    """aten.group_norm → group_norm (attrs `num_groups` / `eps` — ADR 0138 decision 7).

    MUST: preserved (`convert.PRESERVED_OP_PREFIXES`). Decomposed it becomes native_group_norm
    (3 outputs) + getitem, with `HxW` the product of the spatial symbols — neither has an IR form.
    MUST: the input is f32 `[N,C,...]` with a static C divisible by num_groups (the shape rule
    checks it again on the IR). The affine is accepted in the forms "both", "neither" and "weight
    only"; missing slots are synthesized as ones / zeros of length C (the layer_norm move — ADR
    0016). The bias-only form fails (synthesis appends in order, so the bias would slide into the
    weight's slot — the element counts match, a silent wrong value). `cudnn_enabled` is a backend
    hint without effect on the values and is not carried.
    """
    _expect(not node.kwargs, node, f"group_norm with kwargs {sorted(node.kwargs)} is not supported")
    src = node.args[0].meta["val"]
    _expect(src.dtype == torch.float32, node, f"group_norm of dtype {src.dtype} is not supported")
    _expect(src.dim() >= 2, node, f"group_norm of rank {src.dim()} is not supported ([N,C,...])")
    channels = _static_size_or_fail(node, src.shape[1], "group_norm channel axis")
    groups = node.args[1]
    _expect(
        isinstance(groups, int) and not isinstance(groups, bool) and groups >= 1,
        node,
        f"group_norm num_groups={groups!r} is not a positive integer",
    )
    _expect(
        channels % groups == 0,
        node,
        f"group_norm num_groups {groups} does not divide the channels {channels}",
    )
    weight = node.args[2] if len(node.args) > 2 else None
    bias = node.args[3] if len(node.args) > 3 else None
    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5
    _expect(
        math.isfinite(eps) and eps > 0,
        node,
        f"eps={eps} is not supported (finite positive numbers only)",
    )
    attrs = {"num_groups": groups, "eps": eps}
    if isinstance(weight, Node) and isinstance(bias, Node):
        return Emitted("group_norm", 3, attrs)
    if isinstance(weight, Node):
        _expect(bias is None, node, f"group_norm bias={bias!r} is neither a tensor nor None")
        return Emitted("group_norm", 3, attrs, synth_consts=((0.0, channels),))
    _expect(weight is None, node, f"group_norm weight={weight!r} is neither a tensor nor None")
    _expect(
        bias is None,
        node,
        f"group_norm with only a bias and no weight (bias={bias!r}) is not supported",
    )
    return Emitted("group_norm", 3, attrs, synth_consts=((1.0, channels), (0.0, channels)))


def _h_rms_norm(node: Node) -> Emitted:
    """aten.rms_norm → rms_norm (attrs `eps`, **arity 2** — ADR 0017).

    There are two supply routes (ADR 0017): the form where `aten.rms_norm` from diffusers'
    `nn.RMSNorm` is kept as PRESERVED, and the form where `normalize._fold_rms_norm` folded a
    hand-written decomposition. The folded side comes with an explicit eps (only the former takes
    the branch without eps). Both routes have forms without weight — the folding side also folds
    the weight-less form of Gemma4's `v_norm` (with_scale=False) with weight=None.

    MUST: the normalized axis is the single last dimension only, and weight is rank1 of the last
    dimension's length (the contract — the source of truth for the normalized length is the
    weight's length). `normalized_shape` does not go into the IR (no duplicated state).
    MUST: when eps is omitted, **torch's default is made explicit at the boundary**
    (`finfo(f32).eps`). The same rule as leaky_relu's negative_slope — the IR side's "never fill in
    defaults" is about the runtime contract, and the exporter making torch's semantics explicit is
    the responsibility of the opposite side.
    """
    extra = sorted(set(node.kwargs) - {"weight", "eps"})
    _expect(not extra, node, f"rms_norm with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    normalized_shape = node.args[1]
    _expect(
        isinstance(normalized_shape, (list, tuple)) and len(normalized_shape) == 1,
        node,
        f"normalized_shape {normalized_shape!r} is not supported (the single last dimension only)",
    )
    extent = _static_size_or_fail(node, src.shape[-1], "rms_norm normalized axis")
    _expect(
        int(normalized_shape[0]) == extent,
        node,
        f"normalized_shape {list(normalized_shape)} differs from the input's last dimension"
        f" {extent}",
    )
    raw_eps = _arg_or_kwarg(node, 3, "eps", None)
    if raw_eps is None:
        eps = float(torch.finfo(torch.float32).eps)
    else:
        _expect(
            isinstance(raw_eps, (int, float)) and not isinstance(raw_eps, bool),
            node,
            f"eps={raw_eps!r} is not a scalar",
        )
        eps = float(raw_eps)
    _expect(
        math.isfinite(eps) and eps > 0,
        node,
        f"eps={eps} is not supported (finite positive numbers only)",
    )
    weight = _arg_or_kwarg(node, 2, "weight", None)
    if isinstance(weight, Node):
        weight_val = weight.meta["val"]
        _expect(
            list(weight_val.shape) == [extent],
            node,
            f"rms_norm whose weight shape {list(weight_val.shape)} is not [{extent}] is not"
            " supported",
        )
        return Emitted("rms_norm", 2, {"eps": eps})
    _expect(weight is None, node, f"rms_norm weight={weight!r} is neither a tensor nor None")
    # ADR 0017: the weight-less form is normalized to arity 2 by synthesizing ones (the exact
    # identity `×1`).
    return Emitted("rms_norm", 2, {"eps": eps}, synth_consts=((1.0, extent),))


def _h_softmax(node: Node) -> Emitted:
    """aten.softmax → softmax / safe_softmax (attrs `dim`, **last dimension only**).

    MUST: anything other than the last dimension fails (the same narrowing as gather — general dims
    once measurements require them). The runtime has one safe-softmax row kernel and assumes the
    reduction axis is contiguous.

    A softmax that `normalize._drop_safe_softmax_guard` replaced constructively, because it could
    not drop SDPA's safe-softmax guard by proof from actual values, carries the
    {@link SAFE_SOFTMAX_META} flag (ADR 0044). The flag travels in the node's meta because the FX
    graph layer has no aten op called "safe_softmax".
    """
    extra = sorted(set(node.kwargs) - {"dtype"})
    _expect(not extra, node, f"softmax with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    rank = src.dim()
    dim = _normalized_dims(node, rank, [node.args[1]])[0]
    _expect(
        dim == rank - 1,
        node,
        f"softmax on a dimension other than the last (dim={dim} / rank={rank}) is not supported",
    )
    dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
    _expect(dtype is None, node, "softmax with a dtype is not supported")
    op = "safe_softmax" if node.meta.get(SAFE_SOFTMAX_META) else "softmax"
    return Emitted(op, 1, {"dim": dim})


def _h_attention(node: Node) -> Emitted:
    """aten.scaled_dot_product_attention → attention (attrs `scale` — ADR 0023).

    The argument form is `(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    enable_gqa=False)`. **Accepted is "non-causal, dropout 0, rank-4" with either no mask or an
    additive f32 `[1,1,M,N]` mask only** (GQA is accepted as a divisible broadcast by ADR 0067
    decision 1 — see below). Everything else is listed in full and fails loudly
    (no silent approximation — a cross-cutting invariant). SDPA preservation is still enabled
    **per target** (`curated_decompositions(preserved=…)` — to keep ADR 0016's guard-removal pass).

    MUST: the mask is **additive f32** (`S' = S + mask`) with shape exactly `[1,1,M,N]`. Bool masks
    are not accepted here — `normalize._additive_attn_mask` converts them with the same ops and the
    same constants as torch's own decomposition (`where(mask, 0, -inf)`). The check also stays on
    the handler side so that a form for which the normalization pass did not fire (passed by
    kwargs, etc.) does not slip through silently.

    MUST: attrs `scale` is **`f32(√scale_factor)` (the half scale)**. It is **the same constant**
    as the form torch's math decomp writes as `q *= math.sqrt(scale_factor); k *=
    math.sqrt(scale_factor)`, and a drift of even 1 ulp here breaks bit identity with the
    decomposition path (the core of ADR 0023's design).
    Observed oracle: D=128 with scale omitted gives `0.2973017692565918` (= f32(128^-0.25)).
    MUST: rounded to f32 before it is stored. The IR's JSON literal then has **the same decimal
    representation** as the `mul` constant of torch's decomp (the value the exporter bakes as an
    f32 initializer).
    """
    extra = sorted(
        set(node.kwargs) - {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
    )
    _expect(not extra, node, f"scaled_dot_product_attention with kwargs {extra} is not supported")
    attn_mask = _arg_or_kwarg(node, 3, "attn_mask", None)
    _expect(
        attn_mask is None or isinstance(attn_mask, Node),
        node,
        f"attention whose attn_mask={attn_mask!r} is not a tensor value is not supported",
    )
    dropout_p = _arg_or_kwarg(node, 4, "dropout_p", 0.0)
    _expect(
        isinstance(dropout_p, (int, float)) and not isinstance(dropout_p, bool),
        node,
        f"dropout_p={dropout_p!r} is not a scalar",
    )
    _expect(
        float(dropout_p) == 0.0,
        node,
        f"attention with dropout_p={dropout_p} is not supported (inference only)",
    )
    is_causal = _arg_or_kwarg(node, 5, "is_causal", False)
    _expect(
        is_causal is False,
        node,
        "attention with is_causal=True is not supported (no field for a causal mask)",
    )
    # `enable_gqa` is **not read** (ADR 0067 decision 1 — the H match was relaxed to a divisible
    # broadcast, so the same preserved node is emitted for both True and False). The validity of
    # the shape (`H % Hkv == 0` and `H ≥ Hkv`, matching Hkv between k / v) is checked at one place
    # in the shapes layer, and `assert_graph_shapes` at convert's exit always runs every node
    # through it — checking twice here would split the accepted set into two places. torch itself
    # also fails a non-divisible enable_gqa with RuntimeError.
    shapes = []
    for index, name in enumerate(("q", "k", "v")):
        value = node.args[index].meta["val"]
        _expect(
            value.dim() == 4,
            node,
            f"attention with {name} of rank {value.dim()} is not supported (rank-4 head-first"
            " only)",
        )
        shapes.append(value.shape)
    depth = _static_size_or_fail(node, shapes[0][3], "attention D (axis 3 of q)")
    for index, name in enumerate(("k", "v"), start=1):
        other = _static_size_or_fail(node, shapes[index][3], f"attention D (axis 3 of {name})")
        _expect(
            other == depth,
            node,
            f"attention whose {name} D {other} differs from q D {depth} is not supported",
        )
    raw_scale = _arg_or_kwarg(node, 6, "scale", None)
    if raw_scale is None:
        # The same formula as torch's default (`_scaled_dot_product_attention_math`).
        scale_factor = 1.0 / math.sqrt(depth)
    else:
        _expect(
            isinstance(raw_scale, (int, float)) and not isinstance(raw_scale, bool),
            node,
            f"scale={raw_scale!r} is not a scalar",
        )
        scale_factor = float(raw_scale)
    _expect(
        math.isfinite(scale_factor) and scale_factor > 0,
        node,
        f"attention with scale={scale_factor} is not supported (finite positive numbers only — √"
        " is taken)",
    )
    half = float(torch.tensor(math.sqrt(scale_factor), dtype=torch.float32).item())
    if attn_mask is None:
        return Emitted("attention", 3, {"scale": half})
    mask = attn_mask.meta["val"]
    _expect(
        mask.dtype is torch.float32,
        node,
        f"attn_mask of dtype {mask.dtype} is not supported (additive f32 only —"
        " normalize._additive_attn_mask turns bool into where(mask, 0, -inf))",
    )
    _expect(
        mask.dim() == 4,
        node,
        f"attn_mask of rank {mask.dim()} is not supported ([1,1,M,N] only)",
    )
    _expect(
        extent_key(mask.shape[0]) == 1 and extent_key(mask.shape[1]) == 1,
        node,
        f"attn_mask whose leading 2 axes {list(mask.shape[:2])} are not 1 is not supported"
        " (broadcast over B·H only — [B,1,M,N] / [1,H,M,N] are not in the contract)",
    )
    _expect(
        extent_key(mask.shape[2]) == extent_key(shapes[0][2])
        and extent_key(mask.shape[3]) == extent_key(shapes[1][2]),
        node,
        f"attention whose attn_mask M / N {list(mask.shape[2:])} differ from q / k"
        f" {[shapes[0][2], shapes[1][2]]} is not supported",
    )
    return Emitted("attention", 4, {"scale": half})


def _h_embedding(node: Node) -> Emitted:
    """aten.embedding → embedding (weight f32[V,H] × index i32[…] → f32[…,H]).

    MUST: `padding_idx` goes into attrs but **has no effect on forward** (a field that keeps the
    padding row from being updated by gradients; the forward pass is the same as a plain row
    gather). Dropping it from the contract to ignore it would open a hole in the "unknown attrs
    fail loudly" rule, so it is accepted and carried (ADR 0012).
    MUST: `scale_grad_by_freq` / `sparse` are accepted only at their default (False). They are also
    gradient-side fields, but widening acceptance "because they seemingly have no effect on
    forward" lets the next argument that does have an effect slip through for the same reason (the
    same rule as gather's sparse_grad).
    MUST: indices are **rank ≥ 1** (the IR contract — packages/runtime/src/ops.ts). torch allows a
    scalar index and returns `[H]`, but the runtime's contract is `[…index, H]` and has no rank 0
    form.
    """
    extra = sorted(set(node.kwargs) - {"padding_idx", "scale_grad_by_freq", "sparse"})
    _expect(not extra, node, f"embedding with kwargs {extra} is not supported")
    index = node.args[1].meta["val"]
    _expect(
        index.dim() >= 1,
        node,
        "embedding with a scalar index (rank 0) is not supported (the contract's output is"
        " […index, H])",
    )
    padding_idx = _arg_or_kwarg(node, 2, "padding_idx", -1)
    _expect(
        isinstance(padding_idx, int) and not isinstance(padding_idx, bool) and padding_idx >= -1,
        node,
        f"padding_idx={padding_idx!r} is not supported (integers of at least -1 only)",
    )
    _expect(
        not _arg_or_kwarg(node, 3, "scale_grad_by_freq", False),
        node,
        "embedding with scale_grad_by_freq=True is not supported",
    )
    _expect(
        not _arg_or_kwarg(node, 4, "sparse", False),
        node,
        "embedding with sparse=True is not supported",
    )
    return Emitted("embedding", 2, {"padding_idx": int(padding_idx)})


def _h_masked_fill(node: Node) -> Emitted:
    """aten.masked_fill.Scalar → masked_fill (attrs `value`).

    MUST: a non-finite fill value fails (IR v1 rejects non-finite values both as JSON literals and
    at the value level). Filling attention with `-inf` is a real way of writing it, so this check
    is actually hit. The observed value -3.4028234663852886e+38 (the smallest finite f32) passes
    this check.

    NOTE: torch itself requires the mask to be bool in forward (an integer mask fails with
    `expected predicate to be bool`), so it is not checked here. On the IR side it is guaranteed by
    the contract table's per-slot dtypes (x=f32 / mask=bool).
    """
    _expect(
        not node.kwargs, node, f"masked_fill with kwargs {sorted(node.kwargs)} is not supported"
    )
    value = node.args[2]
    _expect(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        node,
        f"value={value!r} is not a scalar (the masked_fill.Tensor form is not supported)",
    )
    number = float(value)
    _expect(math.isfinite(number), node, f"non-finite value={value!r} is not supported by IR v1")
    return Emitted("masked_fill", 2, {"value": number})


def _single_spatial(node: Node, value: Any, what: str) -> int:
    """A spatial parameter of the conv family (normalizes either notation, `1` or `[1]`, to one
    integer)."""
    if isinstance(value, (list, tuple)):
        _expect(
            len(value) == 1,
            node,
            f"{what}={value!r} is not supported (conv1d has one spatial axis)",
        )
        value = value[0]
    _expect(
        isinstance(value, int) and not isinstance(value, bool),
        node,
        f"{what}={value!r} is not an integer",
    )
    return int(value)


def _conv_bias(node: Node, channels_out: int, what: str) -> tuple[tuple[float, int], ...]:
    """Look at the bias slot and return "the zero bias to synthesize" (empty when there is a bias).

    MUST: no bias is **not rejected; a zero bias is synthesized** to normalize to arity 3 (ADR
    0015). Measurements (dec's conv_post / the conv group of Anima's VAE) include convs without
    bias, and synthesis is a narrower surface than bringing arity branches into kernels and
    contracts.
    """
    bias = _arg_or_kwarg(node, 2, "bias", None)
    if isinstance(bias, Node):
        return ()
    _expect(bias is None, node, f"{what} bias={bias!r} is neither a tensor nor None")
    return ((0.0, channels_out),)


def _h_conv1d(node: Node) -> Emitted:
    """aten.conv1d → conv1d (attrs `stride` / `padding` / `dilation` / `groups`).

    MUST: all four are **explicit** in attrs (no reliance on filled-in defaults — ADR 0012 / 0015).
    The contract table also requires them, so omitting either side makes verify fail.
    MUST: the shape layer (shapes.py) checks that `groups` divides Cin / Cout and that the weight
    is `[Cout, Cin/groups, K]`. Only the value ranges are checked here.
    """
    extra = sorted(set(node.kwargs) - {"bias", "stride", "padding", "dilation", "groups"})
    _expect(not extra, node, f"conv1d with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    _expect(src.dim() == 3, node, f"conv1d of rank {src.dim()} is not supported ([B,Cin,L] only)")
    weight = node.args[1].meta["val"]
    _expect(weight.dim() == 3, node, f"conv1d weight of rank {weight.dim()} is not supported")
    stride = _single_spatial(node, _arg_or_kwarg(node, 3, "stride", 1), "stride")
    padding = _single_spatial(node, _arg_or_kwarg(node, 4, "padding", 0), "padding")
    dilation = _single_spatial(node, _arg_or_kwarg(node, 5, "dilation", 1), "dilation")
    groups = _arg_or_kwarg(node, 6, "groups", 1)
    _expect(
        isinstance(groups, int) and not isinstance(groups, bool),
        node,
        f"groups={groups!r} is not an integer",
    )
    _expect(stride >= 1, node, f"stride={stride} is not supported (positive integers only)")
    _expect(padding >= 0, node, f"padding={padding} is not supported (non-negative integers only)")
    _expect(dilation >= 1, node, f"dilation={dilation} is not supported (positive integers only)")
    _expect(groups >= 1, node, f"groups={groups} is not supported (positive integers only)")
    return Emitted(
        "conv1d",
        3,
        {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups},
        synth_consts=_conv_bias(
            node, _static_size_or_fail(node, weight.shape[0], "conv1d output channels"), "conv1d"
        ),
    )


def _pair_spatial(node: Node, value: Any, what: str) -> list[int]:
    """Normalize a spatial parameter of conv2d to 2 components, `[H, W]` (ADR 0017).

    torch accepts the 3 notations `1` / `[1]` / `[1,1]` with the same meaning
    (`nn.Conv2d(kernel_size=3)` holds `stride=1` as a scalar). **IR attrs always have 2
    components**, so the notation is absorbed here at the boundary (no "a scalar applies to both
    axes" rule on the runtime side).
    """
    if isinstance(value, (list, tuple)):
        items = list(value)
        _expect(
            len(items) in (1, 2),
            node,
            f"{what}={value!r} is not supported (conv2d has 2 spatial axes)",
        )
        if len(items) == 1:
            items = [items[0], items[0]]
    else:
        items = [value, value]
    for item in items:
        _expect(
            isinstance(item, int) and not isinstance(item, bool),
            node,
            f"{what}={value!r} is not an integer",
        )
    return [int(item) for item in items]


def _h_conv2d(node: Node) -> Emitted:
    """aten.conv2d → conv2d (attrs `stride` / `padding` / `dilation` / `groups` — ADR 0017).

    MUST: all four are **explicit** in attrs (no reliance on filled-in defaults — ADR 0012 / 0015).
    The 3 spatial ones have 2 components, H/W; only `groups` is a scalar.
    MUST: the shape layer (shapes.py) checks that `groups` divides Cin / Cout and the consistency
    of the weight `[Cout, Cin/groups, Kh, Kw]`. Only notation normalization and value ranges are
    checked here.
    """
    extra = sorted(set(node.kwargs) - {"bias", "stride", "padding", "dilation", "groups"})
    _expect(not extra, node, f"conv2d with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    _expect(src.dim() == 4, node, f"conv2d of rank {src.dim()} is not supported ([B,Cin,H,W] only)")
    weight = node.args[1].meta["val"]
    _expect(weight.dim() == 4, node, f"conv2d weight of rank {weight.dim()} is not supported")
    stride = _pair_spatial(node, _arg_or_kwarg(node, 3, "stride", 1), "stride")
    padding = _pair_spatial(node, _arg_or_kwarg(node, 4, "padding", 0), "padding")
    dilation = _pair_spatial(node, _arg_or_kwarg(node, 5, "dilation", 1), "dilation")
    groups = _arg_or_kwarg(node, 6, "groups", 1)
    _expect(
        isinstance(groups, int) and not isinstance(groups, bool),
        node,
        f"groups={groups!r} is not an integer",
    )
    _expect(
        all(value >= 1 for value in stride),
        node,
        f"stride={stride} is not supported (positive integers only)",
    )
    _expect(
        all(value >= 0 for value in padding),
        node,
        f"padding={padding} is not supported (non-negative only)",
    )
    _expect(
        all(value >= 1 for value in dilation),
        node,
        f"dilation={dilation} is not supported (positive integers only)",
    )
    _expect(groups >= 1, node, f"groups={groups} is not supported (positive integers only)")
    return Emitted(
        "conv2d",
        3,
        {"stride": stride, "padding": padding, "dilation": dilation, "groups": int(groups)},
        synth_consts=_conv_bias(
            node, _static_size_or_fail(node, weight.shape[0], "conv2d output channels"), "conv2d"
        ),
    )


def _triple_spatial(node: Node, value: Any, what: str) -> list[int]:
    """Normalize a spatial parameter of conv3d to 3 components, `[T, H, W]` (ADR 0118 decision 1).

    torch accepts the 3 notations `1` / `[1]` / `[1,1,1]` with the same meaning
    (`nn.Conv3d(kernel_size=3, padding=1)` lowers to aten as `[1, 1, 1]`, but
    `F.conv3d(x, w, stride=2)` arrives as a scalar). **IR attrs always have 3 components**, so the
    notation is absorbed here at the boundary (the 3D version of {@link _pair_spatial} — no "a
    scalar applies to every axis" rule on the runtime side).
    """
    if isinstance(value, (list, tuple)):
        items = list(value)
        _expect(
            len(items) in (1, 3),
            node,
            f"{what}={value!r} is not supported (conv3d has 3 spatial axes)",
        )
        if len(items) == 1:
            items = [items[0], items[0], items[0]]
    else:
        items = [value, value, value]
    for item in items:
        _expect(
            isinstance(item, int) and not isinstance(item, bool),
            node,
            f"{what}={value!r} is not an integer",
        )
    return [int(item) for item in items]


def _h_conv3d(node: Node) -> Emitted:
    """aten.conv3d → conv3d (attrs `stride` / `padding` / `dilation` / `groups` — ADR 0118).

    MUST: only an **unbatched rank 4 input** (`[Cin, T, H, W]`) and a rank 5 weight are accepted.
    A batched rank 5 input is not in the contract (the surrounding cat / slice hit the strided
    family's rank limit of 4), so it fails loudly and names how the recipe should write it.
    MUST: all four are **explicit** in attrs (no reliance on filled-in defaults — ADR 0012 / 0015).
    The 3 spatial ones have 3 components, T/H/W; only `groups` is a scalar. The shape layer
    (shapes.py) checks the divisibility of groups and the consistency of the weight
    `[Cout, Cin/groups, Kt, Kh, Kw]`.
    """
    extra = sorted(set(node.kwargs) - {"bias", "stride", "padding", "dilation", "groups"})
    _expect(not extra, node, f"conv3d with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    _expect(
        src.dim() != 5,
        node,
        "conv3d with a batched rank 5 input [B,Cin,T,H,W] is not supported — drop B = 1 in the"
        " recipe and call F.conv3d with the unbatched [Cin,T,H,W] (ADR 0118 decision 1)",
    )
    _expect(
        src.dim() == 4,
        node,
        f"conv3d of rank {src.dim()} is not supported (unbatched [Cin,T,H,W] only)",
    )
    weight = node.args[1].meta["val"]
    _expect(weight.dim() == 5, node, f"conv3d weight of rank {weight.dim()} is not supported")
    stride = _triple_spatial(node, _arg_or_kwarg(node, 3, "stride", 1), "stride")
    padding = _triple_spatial(node, _arg_or_kwarg(node, 4, "padding", 0), "padding")
    dilation = _triple_spatial(node, _arg_or_kwarg(node, 5, "dilation", 1), "dilation")
    groups = _arg_or_kwarg(node, 6, "groups", 1)
    _expect(
        isinstance(groups, int) and not isinstance(groups, bool),
        node,
        f"groups={groups!r} is not an integer",
    )
    _expect(
        all(value >= 1 for value in stride),
        node,
        f"stride={stride} is not supported (positive integers only)",
    )
    _expect(
        all(value >= 0 for value in padding),
        node,
        f"padding={padding} is not supported (non-negative only)",
    )
    _expect(
        all(value >= 1 for value in dilation),
        node,
        f"dilation={dilation} is not supported (positive integers only)",
    )
    _expect(groups >= 1, node, f"groups={groups} is not supported (positive integers only)")
    return Emitted(
        "conv3d",
        3,
        {"stride": stride, "padding": padding, "dilation": dilation, "groups": int(groups)},
        synth_consts=_conv_bias(
            node, _static_size_or_fail(node, weight.shape[0], "conv3d output channels"), "conv3d"
        ),
    )


def _h_deform_conv2d(node: Node) -> Emitted:
    """torchvision.deform_conv2d → deform_conv2d (attrs `padding` only — ADR 0055).

    The schema is `(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias,
    SymInt stride_h, SymInt stride_w, SymInt pad_h, SymInt pad_w, SymInt dilation_h,
    SymInt dilation_w, SymInt groups, SymInt offset_groups, bool use_mask)`, and the 5 positional
    tensors become the 5 IR inputs as is (`_tensor_args` is in positional order).

    It is a custom op in the `torchvision::` namespace, so it is not registered in the Core ATen
    decomposition table and stays one node after `run_decompositions(curated_decompositions())`
    (observed).

    MUST: `use_mask=False` (DCNv1) fails. The contract is DCNv2 only and requires mask as a slot,
    and torchvision passes **a `[1,1]` dummy tensor** when `mask=None` — passing it through would
    be a silent wrong value where the dummy applies as the modulator (no guarantee the shape check
    catches it).
    MUST: `stride` / `dilation` / `groups` / `offset_groups` other than 1 fail. The contract has
    no field for them, so this is the only check (it looks at the **actual values**, not filled-in
    defaults).
    MUST: narrowed to rank 4 and f32 (the contract is f32 only).
    NOTE: for no bias, torchvision's Python wrapper inserts `aten.full([Cout], 0)`, so the
    exporter's zero-bias synthesis (the conv family's `_conv_bias`) is not needed — no second
    synthesis path.
    """
    _expect(
        not node.kwargs, node, f"deform_conv2d with kwargs {sorted(node.kwargs)} is not supported"
    )
    _expect(
        len(node.args) == 14,
        node,
        f"deform_conv2d with {len(node.args)} arguments is not supported (the schema has 14)",
    )
    (
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        use_mask,
    ) = node.args[5:]
    # MUST: use_mask is checked **before the tensor shapes**. DCNv1's mask slot holds a [1,1]
    # dummy, so swapping the order would make the diagnostic "a rank 2 mask" and hide that
    # "DCNv1 is not in the vocabulary" (ADR 0005's diagnostic discipline).
    _expect(
        use_mask is True,
        node,
        f"deform_conv2d with use_mask={use_mask!r} is not supported"
        " (the contract is DCNv2 only with a required mask — there is no way to omit the slot)",
    )
    src, weight, offset, mask = (node.args[index].meta["val"] for index in range(4))
    for name, value in (("input", src), ("weight", weight), ("offset", offset), ("mask", mask)):
        _expect(
            value.dim() == 4, node, f"rank {value.dim()} is not supported for deform_conv2d {name}"
        )
        _expect(
            value.dtype == torch.float32,
            node,
            f"dtype {value.dtype} is not supported for deform_conv2d {name} (f32 only)",
        )

    # The schema has SymInts, so first check that it is **a static plain int** (a symbolic
    # stride / padding cannot go into attrs).
    def static_int(what: str, value: Any) -> int:
        _expect(
            isinstance(value, int) and not isinstance(value, bool),
            node,
            f"{what}={value!r} is not a static integer",
        )
        return int(value)

    for what, value in (
        ("stride_h", stride_h),
        ("stride_w", stride_w),
        ("dilation_h", dilation_h),
        ("dilation_w", dilation_w),
        ("groups", groups),
        ("offset_groups", offset_groups),
    ):
        _expect(
            static_int(what, value) == 1,
            node,
            f"deform_conv2d with {what}={value!r} is not supported (no field in attrs = fixed"
            " to 1)",
        )
    padding = [static_int("pad_h", pad_h), static_int("pad_w", pad_w)]
    _expect(
        all(value >= 0 for value in padding),
        node,
        f"padding={padding} is not supported (non-negative only)",
    )
    return Emitted("deform_conv2d", 5, {"padding": padding})


def _gru_scan_emitted(node: Node, name: str) -> Emitted:
    """karume.gru_scan{,_reverse} → the IR op of the same name (empty attrs — ADR 0056).

    The schema is `(Tensor gi, Tensor h0, Tensor w_hh, Tensor b_hh) -> Tensor`, and the 4
    positional tensors become the 4 IR inputs as is (`_tensor_args` is in positional order). It is
    a custom op in the `karume::` namespace, so it is not registered in the Core ATen decomposition
    table and stays one node after `run_decompositions` (observed — the docstring of
    `karume/custom_ops.py`).

    MUST: the scan direction uses **the op name** as the IR op name as is (not moved into attrs).
    Folding the names into one and splitting by attrs would mean adding a bool check function to
    the contract table (every existing check function explicitly rejects `bool` as derived from
    int — `_assert_integer_attr` in ops.py). The name is not derived from the overload but **passed
    by the dispatching caller** (one-to-one with the `ATEN_HANDLERS` keys, independent of the
    spelling of torch's internal representation).
    MUST: narrowed to rank and f32 (the contract is f32 only). Multiple layers / bidirectional /
    consuming `h_n` are expressed in the form **before reaching this op** — the caller lines up
    nodes per layer and direction (ADR 0056 decision 7), so all that can be detected here is "the
    form of one node is outside the contract".
    """
    _expect(not node.kwargs, node, f"gru_scan with kwargs {sorted(node.kwargs)} is not supported")
    _expect(
        len(node.args) == 4,
        node,
        f"gru_scan with {len(node.args)} arguments is not supported (the schema has 4)",
    )
    gi, initial, weight, bias = (node.args[index].meta["val"] for index in range(4))
    for slot, value, rank in (
        ("gi", gi, 3),
        ("h0", initial, 2),
        ("w_hh", weight, 2),
        ("b_hh", bias, 1),
    ):
        _expect(
            value.dim() == rank,
            node,
            f"rank {value.dim()} is not supported for gru_scan {slot} (rank {rank} only)",
        )
        _expect(
            value.dtype == torch.float32,
            node,
            f"dtype {value.dtype} is not supported for gru_scan {slot} (f32 only)",
        )
    return Emitted(name, 4, {})


def _h_gru_scan(node: Node) -> Emitted:
    return _gru_scan_emitted(node, GRU_SCAN_OP)


def _h_gru_scan_reverse(node: Node) -> Emitted:
    return _gru_scan_emitted(node, GRU_SCAN_REVERSE_OP)


def _h_gated_delta_scan(node: Node) -> Emitted:
    """karume.gated_delta_scan → the IR op of the same name (empty attrs — ADR 0135 decision 5).

    The schema is `(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta) -> Tensor`, and the 5
    positional tensors become the 5 IR inputs as is. It is a custom op in the `karume::`
    namespace, so it stays one node after `run_decompositions` with **the length axis T symbolic**
    (its body sits behind `register_fake` — the docstring of `karume/custom_ops.py`).

    MUST: narrowed to arity 5, rank 4 / 4 / 4 / 3 / 3 and f32 (the contract is uniform f32; a bf16
    model's casts are upstream's and stay outside the op). The shape relations (B / T / head
    broadcast) are the shape rule's job (`shapes._gated_delta_scan`), so they are not repeated
    here.
    """
    _expect(
        not node.kwargs,
        node,
        f"gated_delta_scan with kwargs {sorted(node.kwargs)} is not supported",
    )
    _expect(
        len(node.args) == 5,
        node,
        f"gated_delta_scan with {len(node.args)} arguments is not supported (the schema has 5)",
    )
    for slot, arg, rank in zip(
        ("q", "k", "v", "g", "beta"), node.args, (4, 4, 4, 3, 3), strict=True
    ):
        value = arg.meta["val"]
        _expect(
            value.dim() == rank,
            node,
            f"rank {value.dim()} is not supported for gated_delta_scan {slot} (rank {rank} only)",
        )
        _expect(
            value.dtype == torch.float32,
            node,
            f"dtype {value.dtype} is not supported for gated_delta_scan {slot} (f32 only)",
        )
    return Emitted(GATED_DELTA_SCAN_OP, 5, {})


def _h_expert_linear(node: Node) -> Emitted:
    """karume.expert_linear → the IR op of the same name (empty attrs — ADR 0145 decision 5).

    The schema is `(Tensor x, Tensor index, Tensor weight) -> Tensor`, and the 3 positional tensors
    become the 3 IR inputs as is. It is a custom op in the `karume::` namespace, so it stays one
    node after `run_decompositions` with **the token axis T symbolic** (its body sits behind
    `register_fake`).

    MUST: narrowed to x f32 of rank 2 (`[T,in]`, shared by the picks) or 3 (`[T,k,in]`, one row
    per pick), index of rank 2 (torch's int64 is declared i32 at the boundary — ADR 0009; an int32
    index is accepted too) and weight f32 of rank 3 (`[E,out,in]`). The shape relations (T, k,
    in) are the shape rule's job (`shapes._expert_linear`), and the index's range is the runtime's
    plan-time producer rule (a `topk` over E — ADR 0145 decision 2).
    """
    _expect(
        not node.kwargs,
        node,
        f"expert_linear with kwargs {sorted(node.kwargs)} is not supported",
    )
    _expect(
        len(node.args) == 3,
        node,
        f"expert_linear with {len(node.args)} arguments is not supported (the schema has 3)",
    )
    x, index, weight = (arg.meta["val"] for arg in node.args)
    _expect(
        x.dim() in (2, 3) and x.dtype == torch.float32,
        node,
        f"expert_linear x of rank {x.dim()} / {x.dtype} is not supported"
        " (f32 [T,in] or [T,k,in] only)",
    )
    _expect(
        index.dim() == 2 and index.dtype in (torch.int64, torch.int32),
        node,
        f"expert_linear index of rank {index.dim()} / {index.dtype} is not supported"
        " (an integer [T,k] only)",
    )
    _expect(
        weight.dim() == 3 and weight.dtype == torch.float32,
        node,
        f"expert_linear weight of rank {weight.dim()} / {weight.dtype} is not supported"
        " (f32 [E,out,in] only)",
    )
    return Emitted(EXPERT_LINEAR_OP, 3, {})


def _h_upsample_bilinear2d(node: Node) -> Emitted:
    """aten.upsample_bilinear2d.vec → upsample_bilinear2d (attrs `output_size` only).

    The signature is `(Tensor input, SymInt[]? output_size, bool align_corners,
    float[]? scale_factors)`. All measurements (the BiRefNet family / upstream Depth Anything V2)
    are `F.interpolate(..., size=(H,W), mode="bilinear", align_corners=True)`, and no active path
    specifies `scale_factors`.

    MUST: `align_corners=False` fails. Both the coordinate formula (`scale·(i+0.5) − 0.5`) and the
    edge handling differ, and accepting it would **change the numbers under the same op name** (the
    same reason as gelu / gelu_tanh — when needed, add it as a separate op). The contract has no
    field for it, so this is the only check.
    MUST: specifying `scale_factors` fails. Deriving the output length from a factor adds another
    rounding rule (`floor(in·factor)`) and gives a second IR alongside the `output_size` form.
    MUST: narrowed to rank 4 and f32. The 3D / 5D versions are separate aten ops and do not come
    here, but f16 / f64 come through the same overload (the contract is f32 only).
    """
    _expect(not node.kwargs, node, f"upsample with kwargs {sorted(node.kwargs)} is not supported")
    src = node.args[0].meta["val"]
    _expect(
        src.dim() == 4,
        node,
        f"upsample_bilinear2d of rank {src.dim()} is not supported ([B,C,H,W] only)",
    )
    _expect(
        src.dtype == torch.float32,
        node,
        f"upsample_bilinear2d of dtype {src.dtype} is not supported (f32 only)",
    )
    output_size = node.args[1]
    align_corners = node.args[2]
    scale_factors = node.args[3] if len(node.args) > 3 else None
    _expect(
        align_corners is True,
        node,
        f"upsample_bilinear2d with align_corners={align_corners!r} is not supported"
        " (the contract is align_corners=True only — no field)",
    )
    _expect(
        scale_factors is None,
        node,
        f"upsample_bilinear2d with scale_factors={scale_factors!r} is not supported"
        " (the output length is given by size only)",
    )
    _expect(
        isinstance(output_size, (list, tuple)) and len(output_size) == 2,
        node,
        f"output_size={output_size!r} is not supported (2 components, [Hout, Wout], only)",
    )
    _expect(
        not _has_free_symbols(output_size),
        node,
        f"output_size={output_size!r} containing symbols is not supported (attrs hold static"
        " integers only)",
    )
    sizes = [int(size) for size in output_size]
    _expect(
        all(size >= 1 for size in sizes),
        node,
        f"output_size={sizes} is not supported (positive integers only)",
    )
    return Emitted("upsample_bilinear2d", 1, {"output_size": sizes})


def _h_conv_transpose1d(node: Node) -> Emitted:
    """aten.conv_transpose1d → conv_transpose1d (attrs `stride` / `padding` — ADR 0015).

    MUST: trailing positional defaults are **omitted** in the preserved form (ups3/4 really have
    the 4-argument form `(input, weight, bias, [stride])` — recon §4). `_arg_or_kwarg` fills the
    missing positions with defaults, so there is no IndexError, but the job here is **not to pass
    the filled-in values through as is**: `output_padding` / `groups` / `dilation` have no field in
    the contract (fixed to 0 / 1 / 1 as measured), so anything other than the default fails.
    MUST: `stride >= 1` (stride 0 is a division by zero in the kernel and hangs the GPU — recon
    §4).
    The shape layer checks that the weight is `[Cin, Cout, K]` and that the output length
    `L·stride` holds.
    """
    extra = sorted(
        set(node.kwargs) - {"bias", "stride", "padding", "output_padding", "groups", "dilation"}
    )
    _expect(not extra, node, f"conv_transpose1d with kwargs {extra} is not supported")
    src = node.args[0].meta["val"]
    _expect(
        src.dim() == 3,
        node,
        f"conv_transpose1d of rank {src.dim()} is not supported ([B,Cin,L] only)",
    )
    weight = node.args[1].meta["val"]
    _expect(
        weight.dim() == 3, node, f"conv_transpose1d weight of rank {weight.dim()} is not supported"
    )
    stride = _single_spatial(node, _arg_or_kwarg(node, 3, "stride", 1), "stride")
    padding = _single_spatial(node, _arg_or_kwarg(node, 4, "padding", 0), "padding")
    output_padding = _single_spatial(
        node, _arg_or_kwarg(node, 5, "output_padding", 0), "output_padding"
    )
    groups = _arg_or_kwarg(node, 6, "groups", 1)
    dilation = _single_spatial(node, _arg_or_kwarg(node, 7, "dilation", 1), "dilation")
    _expect(
        stride >= 1,
        node,
        f"stride={stride} is not supported (positive integers only — stride 0 hangs)",
    )
    _expect(padding >= 0, node, f"padding={padding} is not supported (non-negative integers only)")
    _expect(
        output_padding == 0,
        node,
        f"conv_transpose1d with output_padding={output_padding} is not supported (no field in"
        " attrs)",
    )
    _expect(
        groups == 1,
        node,
        f"conv_transpose1d with groups={groups} is not supported (no field in attrs)",
    )
    _expect(
        dilation == 1,
        node,
        f"conv_transpose1d with dilation={dilation} is not supported (no field in attrs)",
    )
    # The weight is [Cin, Cout, K] — Cout is the second axis (not the first axis as in conv1d)
    return Emitted(
        "conv_transpose1d",
        3,
        {"stride": stride, "padding": padding},
        synth_consts=_conv_bias(
            node,
            _static_size_or_fail(node, weight.shape[1], "conv_transpose1d output channels"),
            "conv_transpose1d",
        ),
    )


def _h_static_quantize(node: Node) -> Emitted:
    _expect(
        len(node.args) == 2 and not node.kwargs,
        node,
        "static_quantize requires an input and a fixed scale",
    )
    scale = static_quantize_scale({"scale": node.args[1]}, "static_quantize")
    return Emitted("static_quantize", 1, {"scale": scale})


ATEN_HANDLERS = {
    torch.ops.karume.static_quantize.default: _h_static_quantize,
    aten.neg.default: _simple("neg", 1),
    aten.abs.default: _simple("abs", 1),
    aten.exp.default: _simple("exp", 1),
    aten.log.default: _simple("log", 1),
    aten.sqrt.default: _simple("sqrt", 1),
    # Inside constant subtrees FOLDABLE_OPS folds it (RoPE tables). Only the form that takes
    # run-time values comes here — DACVAE's Snake activation `x + (α+1e-9)⁻¹·sin²(αx)` was the first
    # (layer 1 of ADR 0043 — now the Core ATen layer, ADR 0059).
    aten.sin.default: _simple("sin", 1),
    aten.tanh.default: _simple("tanh", 1),
    aten.sigmoid.default: _simple("sigmoid", 1),
    aten.relu.default: _simple("relu", 1),
    aten.gelu.default: _h_gelu,
    aten.add.Tensor: _h_binary("add"),
    aten.sub.Tensor: _h_binary("sub"),
    aten.mul.Tensor: _h_binary("mul"),
    aten.div.Tensor: _h_binary("div"),
    aten.mm.default: _simple("matmul", 2),
    aten.bmm.default: _h_bmm,
    aten.gather.default: _h_gather,
    aten.sum.dim_IntList: _h_sum,
    aten.amax.default: _h_row_reduce("amax"),
    aten.amin.default: _h_row_reduce("amin"),
    # The exit of greedy decoding (ADR 0068 decision 2). `aten.argmax.default` is inside Core ATen
    # (`torch.Tag.core`, observed 2026-08-17) and not in default_decompositions either, so it stays
    # one node after the curated decomp.
    aten.argmax.default: _h_argmax,
    aten._to_copy.default: _h_to_copy,
    aten.bitwise_not.default: _h_bitwise_not,
    # Math ops of phase 3 (sdp's spline / dec's leaky_relu — recon §2)
    aten.log1p.default: _simple("log1p", 1),
    aten.where.self: _h_where,
    aten.clamp.default: _h_clamp,
    aten.leaky_relu.default: _h_leaky_relu,
    aten.ge.Scalar: _h_compare_scalar("ge_scalar"),
    aten.le.Scalar: _h_compare_scalar("le_scalar"),
    aten.gt.Scalar: _h_compare_scalar("gt_scalar"),
    aten.ge.Tensor: _h_ge_tensor,
    aten.bitwise_and.Tensor: _h_bitwise_and,
    aten.cumsum.default: _h_cumsum,
    # Layout (ADR 0011) — the 3 forms that do not change element order are normalized to one reshape
    aten.view.default: _h_reshape,
    aten.unsqueeze.default: _h_reshape,
    aten.squeeze.dims: _h_reshape,
    aten.permute.default: _h_permute,
    aten.expand.default: _h_expand,
    # Layout ops, second group, of phase 4 (ADR 0014). slice / cat are also in FOLDABLE_OPS — inside
    # constant subtrees they are folded, and only the forms left on the execution side come to this
    # handler (the two coexist).
    aten.slice.Tensor: _h_slice,
    aten.cat.default: _h_cat,
    aten.constant_pad_nd.default: _h_constant_pad_nd,
    aten.flip.default: _h_flip,
    # Fused ops (ADR 0012) — high-level ops whose decomposition is stopped go to one kernel
    aten.linear.default: _h_linear,
    aten.layer_norm.default: _h_layer_norm,
    aten.softmax.int: _h_softmax,
    aten.embedding.default: _h_embedding,
    aten.masked_fill.Scalar: _h_masked_fill,
    aten.conv1d.default: _h_conv1d,
    aten.conv_transpose1d.default: _h_conv_transpose1d,
    # M1-P4 phase 2 (ADR 0016 / 0017) — Anima's 4 components.
    # NOTE: `squeeze.dim` / `transpose.int` do not appear even once in measurements after the
    # curated decomp (the measured result for the open item of recon §8). `_select_to_squeeze`
    # emits `squeeze.dims`, which the existing reshape normalization takes, so they are not added
    # here (add one line when they show up).
    aten.rms_norm.default: _h_rms_norm,
    aten.conv2d.default: _h_conv2d,
    # 3D convolution (extended composite layer — ADR 0118 decision 1). An unbatched `F.conv3d` stays
    # `.default` with its rank 4 input (observed with torch 2.13.0). The `.padding` overload for a
    # string padding is not in the measurements.
    aten.conv3d.default: _h_conv3d,
    # Bilinear resample (layer 1 — now the Core ATen layer — a common premise of the BiRefNet
    # family / Depth Anything V2). `F.interpolate(size=…, mode="bilinear")` lowers only to the
    # `.vec` overload, and `.default` (the form taking scales_h / scales_w separately) does not
    # appear in measurements (add one line when it shows up).
    aten.upsample_bilinear2d.vec: _h_upsample_bilinear2d,
    # Group normalisation (Core ATen layer — ADR 0138 decision 7): the preserved composite
    # `aten.group_norm` (`native_group_norm` + getitem is not wired — the layer_norm case).
    aten.group_norm.default: _h_group_norm,
    # DCNv2 (layer 1' — now the extended atomic layer — the front-line blocker of the BiRefNet
    # family; ADR 0055). A custom op in the `torchvision::` namespace, so it is not registered in
    # the Core ATen decomposition table and stays one node after the curated decomp.
    torch.ops.torchvision.deform_conv2d.default: _h_deform_conv2d,
    # GRU hidden-side scan, 2 directions (layer 2 — now the extended composite layer — ADR 0056).
    # Our own custom op in the `karume::` namespace; it is not in the decomposition table, so it
    # stays one node after the curated decomp = **the time axis T stays symbolic**.
    torch.ops.karume.gru_scan.default: _h_gru_scan,
    torch.ops.karume.gru_scan_reverse.default: _h_gru_scan_reverse,
    # The gated delta rule scan (extended composite layer — ADR 0135, the value form). Our own
    # custom op; it stays one node after the curated decomp = **the length axis T stays symbolic**.
    torch.ops.karume.gated_delta_scan.default: _h_gated_delta_scan,
    # The routed expert linear (extended composite layer — ADR 0145). Our own custom op; it stays
    # one node after the curated decomp = **the token axis T stays symbolic**.
    torch.ops.karume.expert_linear.default: _h_expert_linear,
    # perf-a (ADR 0023) — only targets that preserve SDPA come to this handler
    # (`curated_decompositions(preserved=…)` chooses per target).
    aten.scaled_dot_product_attention.default: _h_attention,
}
