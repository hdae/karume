"""torch.export'ed ExportedProgram → IR v1 graph (docs/ir-v2.md).

The input is an ExportedProgram run through the curated decomp table (functionalized, preserved
ops still high-level). Processing has 3 steps:

1. Constant folding: a subtree that depends only on constants and shape symbols is **evaluated
   for real at each symbol's maximum (Tmax)** and baked into an initializer; at run time the
   leading part is cut out with a symbolic prefix slice (`sym_prefix_slice`) (ADR 0010). The main
   purpose is to bake DeBERTa's relative-position bucket table (which contains log / ceil) at
   export time — it removes by construction the bug class "1 ulp difference at a bucket boundary →
   gather index off by 1".
2. Conversion through the aten → IR op table. Unsupported aten ops fail loudly **with all of them
   listed** (failing one at a time leaves whoever fills in the vocabulary unable to tell how many
   are missing — ADR 0005's diagnostic discipline). The table itself (the handlers) is
   {@link karume.aten_handlers} — this is the engine side.
3. CSE of pure ops (identical (op, ins, attrs, shape) collapse into one).

MUST: the folding allowlist ({@link FOLDABLE_OPS}) holds only "ops that commute with a prefix
slice" — an op for which `f(T)[0:T,…] == f(Tmax)[0:T,…]` does not hold (a dimension reduction, a
normalization that uses T as a value, etc.) must not be added. However, **the allowlist alone
cannot protect this invariant** (composing allowlisted ops with sym_size can bring T into a
subtree as a "value"), so it is protected in 2 steps:

1. Look at where the symbol is consumed — only extent (length / shape) positions are allowed via
   {@link SYMBOL_EXTENT_ARGS}, and a subtree reached at a value position is removed from foldable
   (symbol-as-data is rejected). A removed op goes down the normal lowering, and if it is outside
   the vocabulary the whole export is rejected as an unsupported op. Forms that reference a
   position on a symbolic axis (an index that looks up a symbolic axis with an index tensor —
   negative or not; a slice with a negative start; cat / repeat / reduction along a symbolic axis)
   and index with a bool / uint8 mask are likewise removed by structure
   ({@link _references_symbolic_position}). Excluding the mask index does not close a silent wrong
   value; it routes a form that already failed explicitly on an unbacked output length onto the
   path where it fails as an op outside the vocabulary.
2. On every fold, **evaluate at 2 points, Tmax and another point, and check the prefix match by
   observation** (_check_prefix_commutes).

The 2-point evaluation is a breakwater, not a proof — there are counterexamples that agree at the
2 points by accident, such as `(scalar_tensor(T)−Tmax)(scalar_tensor(T)−(Tmax−1))`, and on its own
it lets a silent wrong value through. Step 1 is the main protection.

There can be several symbols. The caller gives the IR names through symbol_names, assigned in
order of appearance of the user input placeholders (= forward's argument order) — torch.export's
internal symbol names (`s97` etc.) change from graph to graph, so they are not carried into the
IR. Only one symbol may appear in one dimension (the `s20+s97` form fails loudly).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import OutputKind
from torch.fx import Node

# Bindings of the shared pieces moved out to the table side ({@link karume.aten_handlers}).
# Imports go one way only, convert → aten_handlers (a reverse edge would pull the table into the
# engine).
from karume.aten_handlers import (
    ATEN_HANDLERS,
    DTYPE_NAMES,
    Emitted,
    _expect,
    _has_free_symbols,
    aten,
)
from karume.dims import MAX_SAFE_INT, DimExpr, format_dim, is_symbol_name
from karume.extents import extent_keys, same_extents
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrStorage,
    IrValue,
)
from karume.ops import EMITTABLE_OPS, STRIDED_RANK
from karume.shapes import assert_graph_shapes

#: Value range of i32 (used by the boundary-normalization check).
I32_MIN = -(2**31)
I32_MAX = 2**31 - 1

#: torch dtype of a baked constant → IR storage dtype (docs/ir-v2.md "Values and types").
#:
#: f32 passes through the "encoding of f32 values" vocabulary; `i32` is raw int32 (ADR 0010's
#: explicit exception). i64 is dropped to i32 by normalize_boundary_tensor before it gets here. A
#: constant of a dtype not listed here (bool etc.) has no initializer vocabulary, so it fails
#: loudly.
_CONST_STORAGE = {torch.float32: "f32", torch.int32: "i32"}


def normalize_boundary_tensor(tensor: torch.Tensor, where: str) -> torch.Tensor:
    """Drop a boundary tensor (input / output data) to the representation of IR v1's semantic dtype.

    - i64 → i32 (**out of range fails loudly** — silent truncation shifts indices quietly)
    - bool → u32 0 / 1 (the same convention as GPU storage — ADR 0009)
    - f32 / i32 stay as they are

    MUST: this is the single conversion point. Writing `.to(torch.int32)` at a call site drops the
    range check on that path.
    """
    dtype = tensor.dtype
    if dtype is torch.float32 or dtype is torch.int32:
        return tensor.contiguous()
    if dtype is torch.bool:
        return tensor.to(torch.uint32).contiguous()
    if dtype is torch.int64:
        if tensor.numel() > 0:
            low = int(tensor.min())
            high = int(tensor.max())
            if low < I32_MIN or high > I32_MAX:
                raise ValueError(
                    f"{where}: i64 value range [{low}, {high}] does not fit in i32"
                    f" ([{I32_MIN}, {I32_MAX}]) — IR v1 has no i64 (ADR 0009)"
                )
        return tensor.to(torch.int32).contiguous()
    raise NotImplementedError(
        f"{where}: torch dtype {dtype} cannot be dropped to a semantic dtype of IR v1"
        f" ({', '.join(sorted(set(DTYPE_NAMES.values())))})"
    )


def _check_clone(node: Node) -> None:
    """aten.clone → identity (not emitted to the IR; it becomes an alias of its input).

    Grounds: IR v1 values are **always contiguous** (permute / expand are emitted as contiguous
    strided copies, and reshape does not change the element order — ADR 0011). A contiguous copy of
    a contiguous value is an identity, so the 24 observed (recon §2: ① materialization right after
    a permute, ②③ from dropout / functionalization) are all absorbed here.

    MUST: forms whose `memory_format` is not contiguous are rejected. channels_last and the like
    are not "making contiguous" but a reorder into another layout; collapsing them into an identity
    shifts the element order silently.
    """
    extra = sorted(set(node.kwargs) - {"memory_format"})
    _expect(not extra, node, f"clone with kwargs {extra} is not supported")
    memory_format = node.kwargs.get("memory_format", torch.contiguous_format)
    _expect(
        memory_format in (torch.contiguous_format, torch.preserve_format),
        node,
        f"clone with memory_format={memory_format} is not supported"
        " (IR values are always contiguous — ADR 0011)",
    )


#: No-ops not emitted to the IR (the value becomes an alias of the input). The value is "the check
#: function of the acceptance condition"; an op that may be aliased unconditionally maps to None.
SKIP_OPS: dict[Any, Callable[[Node], None] | None] = {
    aten.alias.default: None,
    aten._assert_tensor_metadata.default: None,
    aten.clone.default: _check_clone,
}

#: The 2 evaluation points of folding. MAX puts every symbol at its upper bound (Tmax), PROBE puts
#: every symbol at **Tmax − 1**. The commutativity check switches both points together and
#: compares.
#:
#: MUST: the second point differs from Tmax and is **at least 2** (torch.export's 0/1
#: specialization presupposes `Dim(min=2)`, and a point at 1 or below may evaluate "a special form
#: that holds only at that length"). If the points coincide, the 2-point evaluation becomes
#: tautological and the check silently means nothing — _sym_range is the guard.
FOLD_POINT_MAX = "max"
FOLD_POINT_PROBE = "probe"

#: Smallest value allowed as the second point of the 2-point evaluation (the line that avoids 0/1
#: specialization).
MIN_PROBE_VALUE = 2

#: Allowlist of constant folding.
#:
#: Why each op commutes with a prefix slice:
#:
#: - `arange(T)` is the first T elements of `arange(Tmax)`
#: - elementwise / comparison / selection (where) act per element (broadcast included — the
#:   corresponding dimensions are kept)
#: - slice / squeeze / unsqueeze / permute re-map indices. A symbolic view is allowed only to add or
#:   remove unit axes; splitting or merging axes is left to run time
#: - `full` / `scalar_tensor` / `clone` / `_to_copy` do not touch the values themselves
#:
#: Of the non-commuting uses, "forms that consume a symbol as a value" are removed from foldable by
#: {@link SYMBOL_EXTENT_ARGS}, and shape-dependent views and references to positions on a symbolic
#: axis ({@link _references_symbolic_position}) are excluded by structure. The 2-point evaluation
#: of _check_prefix_commutes is an additional counterexample check and does not prove
#: commutativity for an arbitrary op. Three other reasons for exclusion:
#: ① ops whose materialization explodes (expand; stopped at the frontier — except a no-op expand,
#: which `_classify_foldable` lets through) ② non-deterministic ops (RNG)
#: ③ ops that obviously do not commute with a prefix, such as reductions and flips.
#: Growing the vocabulary is an explicit act — an op not in the table is not folded, and if it is
#: consumed it fails as an unsupported op.
FOLDABLE_OPS = {
    aten.arange.default,
    aten.arange.start_step,
    aten.full.default,
    aten.scalar_tensor.default,
    aten.clone.default,
    aten._to_copy.default,
    aten.add.Tensor,
    aten.sub.Tensor,
    aten.mul.Tensor,
    aten.div.Tensor,
    aten.neg.default,
    aten.abs.default,
    aten.exp.default,
    aten.log.default,
    aten.sqrt.default,
    aten.cat.default,
    aten.permute.default,
    aten.slice.Tensor,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.unsqueeze.default,
    aten.view.default,
    # Relative-position bucket table (the foldable group of recon §2 — ADR 0010). All are
    # per-element maps, and the value at row t / column j does not depend on T, so they commute
    # with a prefix.
    aten.sign.default,
    aten.ceil.default,
    aten.clamp.default,
    aten.lt.Scalar,
    aten.gt.Scalar,
    aten.le.Scalar,
    aten.bitwise_and.Tensor,
    aten.where.self,
    # RoPE and constant generation (ADR 0016's FOLDABLE extension). **Only ops observed on the
    # folding frontier** are added (of the ADR's candidate table, only add.Scalar / mul.Scalar are
    # not added: a normalization pass collapses them first, so they never reach the frontier).
    #
    # - `sin` / `cos` — the RoPE position table itself. Collapsing it while it is still constant
    #   removes one run-time dispatch outright, so `sin` stays here even after it entered the IR
    #   vocabulary (`cos` is not in the vocabulary, so not folding = failing, as before). Only a
    #   `sin` that takes a run-time value goes down to `_simple`
    # - `reciprocal` / `pow.Scalar` — the inv_freq formula (`1 / theta**(i/D)`)
    # - `repeat` — the step that spreads DiT's rope table over the patch grid (leaves are constants
    #   only; output on the order of 1024×128)
    # - `bmm` — `inv_freq[…] @ position_ids[…]` of the conditioner / Qwen3 ([1,D/2,1] × [1,1,T])
    aten.sin.default,
    aten.cos.default,
    aten.reciprocal.default,
    aten.pow.Scalar,
    aten.repeat.default,
    aten.bmm.default,
    # Causal mask generation (transformers' `create_causal_mask` — leaves are arange and constants
    # only). Without folding this, the i64 index arithmetic stays in the IR as is (out of reach of
    # ADR 0009's boundary normalization).
    aten.cumsum.default,
    aten.index.Tensor,
    aten.eq.Tensor,
    aten.le.Tensor,
    aten.ne.Scalar,
    # Composition of the sliding-window and causal masks (Gemma3's `create_masks_for_generate` —
    # overlays 2 band masks with `|` and tests outside the window with `>`). Both are per-element
    # comparisons / logical ors, and the value at row t / column j does not depend on T, so they
    # commute with a prefix (observed to pass the 2-point evaluation as well).
    aten.bitwise_or.Tensor,
    aten.gt.Tensor,
    # Bidirectional (non-causal) mask generation (transformers' `bidirectional_mask_function`
    # uses `q_idx >= 0` as an all-true seed). One comparison, and the value at row t / column j
    # does not depend on T, so it commutes with a prefix.
    aten.ge.Scalar,
}

#: Names of the arguments of foldable ops **where a symbol may appear** (= extent = length /
#: shape).
#:
#: MUST: a node in which a symbol reaches an argument not listed here has consumed the symbol **as
#: a value** (promoted it to tensor data) and is removed from foldable. At an extent position T
#: moves only the **length** of the baked constant and prefix isomorphism holds; at a value
#: position the element **contents** become a constant that is right only at Tmax, and the 2-point
#: evaluation can agree by accident (`(T−Tmax)(T−(Tmax−1))` built from `scalar_tensor(T)` is a
#: counterexample — once baked, the value changes silently for any T other than Tmax and Tmax−1).
#:
#: - `arange` — `end` only. start / step decide the values; T moves only the length
#: - `full` / `view` / `expand` / `repeat` — the shape (size / repeats) is an extent, `fill_value`
#:   is a value
#: - `slice` — `end` only (a T-dependent `start` cuts out something other than the leading part =
#:   does not commute with a prefix)
#:
#: Ops not in the table (`scalar_tensor` / binary elementwise / scalar comparisons / an axis `dim`,
#: etc.) have no extent argument — a symbol reaching them is consumption as a value. A position of
#: unknown class falls on the excluded side.
SYMBOL_EXTENT_ARGS: dict[Any, frozenset[str]] = {
    aten.arange.default: frozenset({"end"}),
    aten.arange.start_step: frozenset({"end"}),
    aten.full.default: frozenset({"size"}),
    aten.view.default: frozenset({"size"}),
    aten.expand.default: frozenset({"size"}),
    aten.repeat.default: frozenset({"repeats"}),
    aten.slice.Tensor: frozenset({"end"}),
}

#: High-level ops whose decomposition is stopped (fusion kept) = **ADR 0007's 9 ops** (extended
#: from gelu alone in ADR 0012).
#:
#: Only ops that, once decomposed, no longer map 1:1 to a kernel are taken out:
#:
#: - `linear` — scatters into addmm + view (16 observed / 2 layers). One fused kernel is right.
#: - `layer_norm` — becomes native_layer_norm (**3 outputs**) + getitem, and getitem wiring of
#:   multi-output aten is deferred until sampling needs it (ADR 0068 addendum 2), so it cannot be
#:   accepted. Preserving it is effectively mandatory.
#: - `softmax` — only becomes `aten._softmax`, nothing gained. Keeps the one safe-softmax kernel.
#: - `gelu` — scatters into a composition of the erf / tanh approximation (preserved since M0).
#: - `conv1d` / `conv2d` / `conv3d` / `conv_transpose1d` — become the generic `aten.convolution`
#:   form.
#: - `embedding` — not decomposed by the core decomposition anyway, but listed in the preserved
#:   list to make the intent explicit.
#: - `masked_fill` — scatters into where + scalar_tensor. Keeps the form with the fill value in
#:   attrs.
#: - `leaky_relu` — added in ADR 0015 (9 ops → 10 ops). The decomposed form is
#:   `gt_scalar + mul + where`; intermediate buffers grow 1.5–2×, breaking the premise of dec's
#:   memory estimate.
#:
#: - `rms_norm` — added in ADR 0017 (10 ops → 11 ops). The path that preserves `aten.rms_norm`
#:   from diffusers' `nn.RMSNorm`; hand-written decomposed forms (Qwen3 / DiT) cannot be folded by
#:   preservation, so `normalize._fold_rms_norm` takes them (2 supply routes — ADR 0017).
#: - `group_norm` — added in ADR 0138 decision 7 (12 ops → 13 ops). Decomposed, `nn.GroupNorm`
#:   becomes `native_group_norm` (**3 outputs**) + getitem (the layer_norm case) with its `HxW`
#:   argument the product of the spatial symbols (`sym_size · sym_size` — no IR form); preserving
#:   `aten.group_norm` keeps the one node with its `num_groups` / `eps` and nothing symbolic.
#:   No export changes by it: before, any GroupNorm failed as an unsupported `native_group_norm`.
#:
#: NOTE: conv_transpose1d got its kernel, contract table and handler in ADR 0015, conv2d in ADR
#: 0017 (dec's 5 ups / Anima VAE decoder's 37, respectively). conv3d is ADR 0118 decision 1
#: (extended composite layer, outside Core ATen — the causal conv of the video VAE); an unbatched
#: `F.conv3d` stays in the trace as `aten.conv3d.default` with a rank-4 input (observed with torch
#: 2.13.0 — no normalization pass needed).
PRESERVED_OP_PREFIXES = (
    "aten.conv1d.",
    "aten.conv2d.",
    "aten.conv3d.",
    "aten.conv_transpose1d.",
    "aten.embedding.",
    "aten.gelu.",
    "aten.group_norm.",
    "aten.layer_norm.",
    "aten.leaky_relu.",
    "aten.linear.",
    "aten.masked_fill.",
    "aten.rms_norm.",
    "aten.softmax.",
)

#: Prefix that preserves fused attention (ADR 0023).
#:
#: MUST: **not** part of the default {@link PRESERVED_OP_PREFIXES}. The table is global, so adding
#: it would also preserve SDPA with masks outside the contract (a bool mask passed through kwargs,
#: `[B,1,M,N]`, etc.), and `_h_attention`'s fail-loudly **would make export impossible**. It is
#: enabled by a **per-target** choice through `curated_decompositions(preserved=…)` (so that
#: ADR 0016's safe-softmax guard removal pass keeps working as it is).
ATTENTION_OP_PREFIX = "aten.scaled_dot_product_attention."

#: The preserved set that keeps SDPA (only for targets with no mask or only an additive
#: `[1,1,M,N]` mask — DiT / VAE decoder / EmbeddingGemma).
PRESERVED_OP_PREFIXES_WITH_ATTENTION = (*PRESERVED_OP_PREFIXES, ATTENTION_OP_PREFIX)


#: Core ATen membership of an unsupported op. Whether adding it to the vocabulary needs an ADR is
#: decided by this one test (docs/op-vocabulary.md "Procedure for adding an op" — the entry test =
#: `torch.Tag.core in op.tags`). `non_aten` is outside the aten namespace (`operator.getitem`,
#: custom ops, etc.) and is not subject to the test.
AtenCategory = Literal["core_aten", "non_core_aten", "non_aten"]

_CATEGORY_LABELS: dict[AtenCategory, str] = {
    "core_aten": "Core ATen",
    "non_core_aten": "outside Core ATen",
    "non_aten": "outside aten",
}


def aten_category(target: object) -> AtenCategory:
    """Decide the Core ATen membership of an FX node's target from the tags of the torch in use.

    MUST: no hand-written list — the Core ATen set moves with the torch version, so a list goes
    stale silently. The source of truth for the test is the `Tag.core` on torch's schema.
    """
    if isinstance(target, torch._ops.OpOverload) and target.namespace == "aten":
        return "core_aten" if torch.Tag.core in target.tags else "non_core_aten"
    return "non_aten"


@dataclass(frozen=True)
class ModuleOrigin:
    """The module that produced an unsupported op (per innermost class of `nn_module_stack`).

    Grouping by class is because, in a model that repeats the same layer (stacked blocks), listing
    by path makes the line count grow with the depth until it is unreadable. Where to rewrite is
    decided by "which class", and the first path is enough to find it.
    """

    #: Fully qualified name of the class (the `nn_module_stack` value as is).
    module_type: str
    #: Module path of the class's first node (`""` = the root's forward).
    first_path: str
    count: int


@dataclass(frozen=True)
class UnsupportedOpReport:
    """Decision material for one unsupported op (Core ATen membership, source modules, count)."""

    target: str
    category: AtenCategory
    nodes: tuple[str, ...]
    #: Source modules (in order of first appearance).
    origins: tuple[ModuleOrigin, ...]
    #: Number of nodes without `nn_module_stack` (nodes created by normalize, etc. — no source is
    #: made up).
    untracked: int

    @property
    def count(self) -> int:
        return len(self.nodes)


def _innermost_module(node: Node) -> tuple[str, str] | None:
    """(path, class name) of the innermost module that produced the node; None without a stack."""
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return None
    # torch.export stacks the modules in a dict from outermost to innermost (the last is innermost).
    path, module_type = list(stack.values())[-1]
    return str(path), str(module_type)


def _report(target: str, nodes: Sequence[Node]) -> UnsupportedOpReport:
    origins: dict[str, ModuleOrigin] = {}
    untracked = 0
    for node in nodes:
        innermost = _innermost_module(node)
        if innermost is None:
            untracked += 1
            continue
        path, module_type = innermost
        seen = origins.get(module_type)
        origins[module_type] = (
            ModuleOrigin(module_type=module_type, first_path=path, count=1)
            if seen is None
            else replace(seen, count=seen.count + 1)
        )
    return UnsupportedOpReport(
        target=target,
        category=aten_category(nodes[0].target),
        nodes=tuple(node.name for node in nodes),
        origins=tuple(origins.values()),
        untracked=untracked,
    )


def _describe(report: UnsupportedOpReport) -> str:
    sources = [
        f"{origin.module_type.rsplit('.', 1)[-1]} ×{origin.count} ({origin.first_path or '<root>'})"
        for origin in report.origins
    ]
    if report.untracked:
        sources.append(f"unknown module ×{report.untracked}")
    return (
        f"  - {report.target} ×{report.count} [{_CATEGORY_LABELS[report.category]}]"
        f" from: {', '.join(sources)} / nodes: {', '.join(report.nodes)}"
    )


class UnsupportedAtenOpsError(NotImplementedError):
    """aten ops with no table entry in the IR vocabulary (all of them listed).

    Each op carries its Core ATen membership, source modules and count — so that a single error is
    enough to decide whether to add it to the vocabulary (whether an ADR is needed depends on Core
    ATen membership) or to avoid it by rewriting the module on the recipe side.
    """

    def __init__(self, nodes: Sequence[Node]) -> None:
        grouped: dict[str, list[Node]] = {}
        for node in nodes:
            grouped.setdefault(str(node.target), []).append(node)
        self.reports = {target: _report(target, grouped[target]) for target in sorted(grouped)}
        self.ops = {target: list(report.nodes) for target, report in self.reports.items()}
        listed = "\n".join(_describe(report) for report in self.reports.values())
        super().__init__(
            f"unsupported aten ops ({len(self.reports)} kinds):\n{listed}\n"
            "Classes: [Core ATen] = tagged torch.Tag.core (adding it to the vocabulary needs only"
            " a ledger NOTE; but an addition that breaks an execution-model invariant such as"
            " single output requires an ADR — ADR 0064 axis B)"
            " / [outside Core ATen] = adding it to the vocabulary requires an ADR (consider a"
            " decomposition or a rewrite on the recipe side first)"
            " / [outside aten] = operator.getitem, custom ops, etc. (not subject to the Core ATen"
            " test). "
            'The order of the test and its exceptions: docs/op-vocabulary.md "Procedure for adding'
            ' an op"'
        )


def _carries_symbol(value: Any) -> bool:
    """Whether an argument **carries a symbol as a value** (a SymInt itself / a node carrying one).

    Tensor arguments are False — propagation through tensors works per subtree by "removing the
    consuming node from foldable", so only the entry points where a symbol newly comes in are
    looked at here.
    """
    if isinstance(value, Node):
        val = value.meta.get("val")
        return not isinstance(val, torch.Tensor) and _has_free_symbols(val)
    if isinstance(value, (list, tuple)):
        return any(_carries_symbol(item) for item in value)
    if isinstance(value, torch.Tensor):
        return False
    return _has_free_symbols(value)


def _uses_symbol_as_data(node: Node) -> bool:
    """Whether a node consumes a symbol at a non-extent argument position (= a value).

    See {@link SYMBOL_EXTENT_ARGS}. Argument names are looked up in the op's schema — so the test
    is the same whether the argument comes positionally or as a kwarg.
    """
    allowed = SYMBOL_EXTENT_ARGS.get(node.target, frozenset())
    names = [argument.name for argument in node.target._schema.arguments]
    named = [*zip(names, node.args, strict=False), *node.kwargs.items()]
    return any(name not in allowed and _carries_symbol(value) for name, value in named)


def _schema_args(node: Node) -> dict[str, Any]:
    """op schema argument name → value (one table whether positional, kwarg or default)."""
    arguments = node.target._schema.arguments
    bound = {
        argument.name: argument.default_value
        for argument in arguments
        if argument.has_default_value()
    }
    bound.update(zip([argument.name for argument in arguments], node.args, strict=False))
    bound.update(node.kwargs)
    return bound


#: Ops whose form {@link _references_symbolic_position} checks (all within {@link FOLDABLE_OPS}).
_POSITION_SENSITIVE_OPS = frozenset(
    {
        aten.index.Tensor,
        aten.slice.Tensor,
        aten.cat.default,
        aten.repeat.default,
        aten.bmm.default,
    }
)


#: Index dtypes torch interprets as a mask (uint8 is deprecated but treated as a mask like bool).
_MASK_INDEX_DTYPES = frozenset({torch.bool, torch.uint8})


def _references_symbolic_position(node: Node) -> bool:
    """Whether a form references a **position** on a symbolic axis (structurally not commuting).

    MUST: even when the symbol does not reach a value position (it passes
    {@link _uses_symbol_as_data}), a form that "counts from the end" or "reduces the whole length"
    of a symbolic-length axis turns the length T into element values. `arange(T)[[-1]]` becomes
    T−1 and `arange(T)[-2:]` becomes `[T−2, T−1]`; if a later comparison happens to agree at the 2
    points Tmax and Tmax−1, it slips past the 2-point evaluation as a silent wrong value. So as not
    to rely on the values by chance, the following forms are removed from folding without looking
    at the index values (ADR 0010's addendum of 2026-10-10):

    - `index` — looks up a symbolic axis with an index tensor (a negative index wraps by the
      length; whether an index is non-negative is a property of the values and is not decided by
      structure). A bool / uint8 mask index is removed whether or not there is a symbolic axis — a
      mask of rank 2 or more spans several axes, so the "index k ↔ input axis k" correspondence
      breaks and a symbolic axis can hide behind the mask's second or later axis (falls on the
      conservative side). The output length of a mask index is, however, an unbacked symbol decided
      by the mask's values, and even without this exclusion it already failed explicitly on the
      baked result's shape ({@link Converter._sym_parts}'s "symbol that does not appear in an input
      shape"). This is not a rule that closes a silent wrong value; it aligns how the failure shows
      up with an op outside the vocabulary (the full list of unsupported ops)
    - `slice` — cuts a symbolic axis with a negative start (counting from the end). A non-negative
      start is a position from the front, so it commutes with a prefix and is folded
    - `cat` — concatenates along a symbolic axis (positions of later inputs shift by the length T
      of earlier ones)
    - `repeat` — repeats a symbolic axis 2 or more times (as with cat — positions of the second
      round shift by T)
    - `bmm` — contracts over a symbolic axis (a sum of T terms, depends on the whole length)
    """
    target = node.target
    if target not in _POSITION_SENSITIVE_OPS:
        return False
    args = _schema_args(node)
    if target is aten.cat.default:
        out = node.meta["val"]
        return _has_free_symbols(out.shape[args["dim"] % out.dim()])
    source = args["self"].meta["val"]
    if target is aten.index.Tensor:
        indices = args["indices"]
        if any(
            index is not None and index.meta["val"].dtype in _MASK_INDEX_DTYPES for index in indices
        ):
            return True
        return any(
            index is not None and _has_free_symbols(source.shape[axis])
            for axis, index in enumerate(indices)
        )
    if target is aten.slice.Tensor:
        start = args["start"]
        return (
            _has_free_symbols(source.shape[args["dim"] % source.dim()])
            and isinstance(start, int)
            and start < 0
        )
    if target is aten.repeat.default:
        # repeats map to the input axes right-aligned (the leading extras are new axes).
        counts = args["repeats"][len(args["repeats"]) - source.dim() :]
        return any(
            _has_free_symbols(extent) and not (isinstance(count, int) and count == 1)
            for extent, count in zip(source.shape, counts, strict=True)
        )
    # bmm: contraction axis k of [b, n, k] × [b, k, m].
    return _has_free_symbols(source.shape[2])


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare 2 tensors **bit-exactly**, dtype and shape included.

    MUST: doing folding's 2-point comparison with numeric equality (`torch.equal`) makes
    `-0.0 == 0.0` true and lets through a form that bakes a constant differing only in the sign bit
    (initializers carry signed zero as is, so the sign changes silently at run time). Floats are
    compared through an integer view of the contiguous bytes, and NaN passes only when the payload
    matches too.
    """
    if left.dtype is not right.dtype or left.shape != right.shape:
        return False
    if left.is_floating_point():
        left = left.contiguous().flatten().view(torch.uint8)
        right = right.contiguous().flatten().view(torch.uint8)
    return torch.equal(left, right)


def _holds_node(value: Any) -> bool:
    if isinstance(value, Node):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_node(item) for item in value)
    return False


class Converter:
    def __init__(self, ep: ExportedProgram, *, symbol_names: Sequence[str] = ("T",)) -> None:
        self.ep = ep
        self.symbol_names = tuple(symbol_names)
        if len(set(self.symbol_names)) != len(self.symbol_names):
            raise ValueError(f"symbol_names has duplicates: {self.symbol_names}")
        for name in self.symbol_names:
            # A name outside the grammar of the dimension notation would give a dimension that can
            # be written but not read by the runtime.
            if not is_symbol_name(name):
                raise ValueError(
                    f"IR symbol name {name!r} is not in the grammar of the dimension notation"
                )
        self.gm = ep.graph_module
        signature = ep.graph_signature
        self.placeholder_fqn: dict[str, str] = {
            **signature.inputs_to_parameters,
            **signature.inputs_to_buffers,
            **signature.inputs_to_lifted_tensor_constants,
        }
        self.user_inputs = set(signature.user_inputs)
        self.lifted_constants = set(signature.inputs_to_lifted_tensor_constants)
        self.state: dict[str, torch.Tensor] = {
            **dict(ep.named_parameters()),
            **dict(ep.named_buffers()),
            **ep.constants,
        }
        self.graph = IrGraph()
        self.tensors: dict[str, torch.Tensor] = {}
        # FX node name → IR value name
        self.value_of: dict[str, str] = {}
        # Deferred aliases of SKIP ops
        self.alias_of: dict[str, Node] = {}
        # torch internal symbol name → IR symbol name
        self.sym_names: dict[str, str] = {}
        # IR symbol name → (lower bound, upper bound). The upper bound is Tmax (folding's first
        # evaluation point).
        #
        # MUST: the single source is the ExportedProgram's range_constraints (= exactly the
        # constraints that dynamic_shapes' `Dim(min=…, max=…)` registered with torch). Taking Tmax
        # from the caller as a separate argument would duplicate dynamic_shapes, and when the two
        # disagree "a constant baked shorter than declared, out of range at run time" passes
        # silently.
        self.sym_ranges: dict[str, tuple[Any, Any]] = {}
        # MUST: only **plain symbols** are picked up as keys. Writing a derived dimension (a
        # `Dim("S") + 1519` declared in an input shape — ADR 0046) into dynamic_shapes puts a sympy
        # `Add` (`s27 + 1519`) among the range_constraints keys, which has no `.name` and breaks
        # the whole scan. A derived dimension's range is redundant information decided by the
        # plain symbol's range, so nothing is lost by leaving it out.
        self._internal_ranges: dict[str, tuple[Any, Any]] = {
            symbol.name: (value_range.lower, value_range.upper)
            for symbol, value_range in ep.range_constraints.items()
            # sympy is not imported directly (a transitive dependency of torch, not a declared
            # dependency) — being a plain symbol is checked with the predicate sympy itself puts on
            # every expression.
            if getattr(symbol, "is_Symbol", False)
        }
        # Naming-done flag (set by _assign_input_symbols — any later new naming is rejected)
        self.symbols_assigned = False
        # Evaluation point → (node name → evaluation result). Split per point for the 2-point
        # evaluation.
        self._fold_cache: dict[str, dict[str, Any]] = {}
        self._const_by_digest: dict[str, str] = {}
        self._cse: dict[str, str] = {}

    # ---- Symbols -------------------------------------------------------

    def _sym_parts(self, value: torch.SymInt) -> DimExpr:
        """Decompose a SymInt into the linear expression coeff·sym+offset.

        Only integers with coefficient >= 1 and offset >= 0 are accepted. Quadratic expressions and
        expressions that mix 2 or more symbols in one dimension (`s20+s97`) are not in the
        canonical grammar, so they fail loudly.
        """
        expr = value.node.expr
        symbols = list(expr.free_symbols)
        if len(symbols) != 1:
            reason = (
                "shape expression mixing several symbols in one dimension"
                if len(symbols) > 1
                else "uninterpretable shape expression"
            )
            raise NotImplementedError(f"{reason} is not supported by IR v1: {expr}")
        symbol = symbols[0]
        coeff = expr.coeff(symbol, 1)
        offset = expr.coeff(symbol, 0)
        if (
            not getattr(coeff, "is_Integer", False)
            or not getattr(offset, "is_Integer", False)
            or coeff * symbol + offset != expr
            or not 1 <= int(coeff) <= MAX_SAFE_INT
            or not 0 <= int(offset) <= MAX_SAFE_INT
        ):
            raise NotImplementedError(f"shape expression is not supported by IR v1: {expr}")
        return DimExpr(coeff=int(coeff), sym=self._sym_name(symbol.name), offset=int(offset))

    def _sym_name(self, internal: str) -> str:
        """Look up the IR name of a torch internal symbol (only _assign_input_symbols names)."""
        name = self.sym_names.get(internal)
        if name is not None:
            return name
        if self.symbols_assigned:
            # The runtime cannot bind a symbol that does not appear in an input shape (binding
            # comes only from dimension positions of input shapes — docs/ir-v2.md). Putting it in
            # the IR would make the graph unrunnable.
            raise NotImplementedError(
                f"symbol {internal}, which does not appear in an input shape, appeared in a value"
                f" (from inputs={self.sym_names}) — the runtime cannot bind it, so it cannot be"
                " put in the IR"
            )
        if len(self.sym_names) >= len(self.symbol_names):
            raise NotImplementedError(
                f"not enough IR symbol names: no name to assign to internal symbol {internal}"
                f" (symbol_names={list(self.symbol_names)}, assigned={self.sym_names})"
                " — add a name to convert(symbol_names=...)"
            )
        name = self.symbol_names[len(self.sym_names)]
        self.sym_names[internal] = name
        # The upper bound is needed only at folding's evaluation points, so it is only registered
        # here, not checked (a graph without symbol-dependent constants can be exported without
        # Dim(max=…)).
        found = self._internal_ranges.get(internal)
        if found is not None:
            self.sym_ranges[name] = found
        return name

    def _sym_range(self, name: str) -> tuple[int, int]:
        """(lower bound, Tmax) of a symbol. A range unusable for 2-point evaluation fails loudly.

        MUST: Tmax must be finite (an unset `Dim(max=…)` is `int_oo`). Going on with a default
        while it is infinite would decide the length of the baked constant arbitrarily.
        MUST: the second evaluation point `Tmax − 1` must be at least the lower bound and at least
        {@link MIN_PROBE_VALUE}. Otherwise the 2-point evaluation collapses into one point and the
        commutativity check becomes tautological (= the check disappears silently).
        """
        found = self.sym_ranges.get(name)
        if found is None:
            raise NotImplementedError(
                f"the range of symbol {name} is not in the ExportedProgram's range_constraints"
                " — folding symbol-dependent constants needs an upper bound on the dynamic_shapes"
                " Dim"
            )
        raw_lower, raw_upper = found
        # MUST: finiteness is checked through float. The upper bound of an unset `Dim(max=…)` is
        # torch's `int_oo`; sympy's `is_finite` returns True for it while `int()` raises
        # internally (the type cannot tell).
        if not math.isfinite(float(raw_upper)):
            raise NotImplementedError(
                f"the upper bound of symbol {name} is not finite ({raw_upper}) — symbol-dependent"
                f" constants must be baked at Tmax (declare dynamic_shapes with"
                f" Dim('…', min=…, max=Tmax))"
            )
        upper = int(raw_upper)
        lower = max(int(raw_lower), 0)
        if upper - 1 < max(lower, MIN_PROBE_VALUE):
            raise ValueError(
                f"symbol {name} with range [{lower}, {upper}] does not allow folding's 2-point"
                f" evaluation (the second point {upper - 1} is below the lower bound {lower} /"
                f" the line that avoids 0-1 specialization {MIN_PROBE_VALUE}, so the"
                " commutativity check becomes tautological)"
            )
        return lower, upper

    def _point_value(self, name: str, point: str) -> int:
        """Symbol value at an evaluation point. PROBE is Tmax − 1 (all symbols move together)."""
        _, upper = self._sym_range(name)
        return upper if point == FOLD_POINT_MAX else upper - 1

    def _sym_value(self, value: torch.SymInt, point: str) -> int:
        """Turn a SymInt into a concrete value at an evaluation point."""
        if not value.node.expr.free_symbols:
            return int(value)
        expr = self._sym_parts(value)
        return expr.coeff * self._point_value(expr.sym, point) + expr.offset

    def _assign_input_symbols(self) -> None:
        """Assign IR symbol names in order of the user input placeholders (the only naming path).

        Leaving naming to consumption order (the node walk) could swap names for the same graph
        depending on the order of intermediate nodes. Input shapes are a deterministic sequence
        fixed by forward's argument order, so the names are fixed here before conversion. After the
        walk the naming is sealed, and late naming of a symbol not in the inputs is rejected.
        """
        for node in self.gm.graph.nodes:
            if node.op != "placeholder" or node.name not in self.user_inputs:
                continue
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue
            for dim in val.shape:
                if isinstance(dim, torch.SymInt) and dim.node.expr.free_symbols:
                    self._sym_parts(dim)
        self.symbols_assigned = True

    # ---- Constant folding ---------------------------------------------------

    def _classify_foldable(self) -> set[str]:
        """Set of node names of subtrees depending only on constants and shape symbols (ADR 0010).

        Eligible leaves are only lifted constants (independent of T) and nodes carrying SymInts
        (sym_size / SymInt arithmetic). Folding through parameters / buffers could bake huge
        constants, so it is ineligible (weights travel as initializers).

        MUST: symbol dependence is allowed because the "bake at Tmax, cut out with a prefix slice"
        representation came in (ADR 0010). A subtree that consumed a symbol **as a value**,
        however, bakes a constant that is right only at Tmax, so a node reached by a symbol at a
        position other than an extent position of {@link SYMBOL_EXTENT_ARGS} is dropped here
        (downstream nodes fall out naturally through the dependency chain). Forms that reference a
        position on a symbolic axis ({@link _references_symbolic_position}) are likewise dropped
        by structure. The remaining commutativity is caught by the 2-point observation of
        _check_prefix_commutes.
        """
        foldable: set[str] = set()
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                if node.name in self.lifted_constants:
                    foldable.add(node.name)
                continue
            if node.op != "call_function":
                continue
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                # A node that carries no tensor (sym_size / SymInt arithmetic) depends only on
                # shape information. Even with symbols it becomes concrete by substitution at each
                # evaluation point, so it is eligible as a leaf.
                if isinstance(val, torch.SymInt) or (
                    isinstance(val, int) and not isinstance(val, bool)
                ):
                    foldable.add(node.name)
                continue
            if node.target is aten.expand.default:
                # expand stops at the frontier as a rule (folding multiplies the materialization).
                # A **no-op expand whose output shape equals its input**, however, adds not a
                # single element, so there is no reason to cut the folding chain (observed: RoPE's
                # `inv_freq[None,:,None].expand(batch, -1, 1)` takes this form at batch=1, and
                # stopping here would require sin / cos in the IR vocabulary — ADR 0016).
                if not same_extents(node.meta["val"], node.args[0].meta.get("val")):
                    continue
            elif node.target not in FOLDABLE_OPS:
                continue
            if node.target is aten.view.default:
                source = node.args[0].meta.get("val")
                if _has_free_symbols(source) or _has_free_symbols(val):
                    # DECIDED: a dynamic view can move row boundaries with the length. Only
                    # adding / removing unit axes is folded; the rest is reordered by the runtime
                    # after the prefix is cut.
                    # [ADR 0010](../../../../docs/decisions/0010-symbolic-constant-folding.md)
                    source_axes = tuple(d for d in extent_keys(source.shape) if d != 1)
                    target_axes = tuple(d for d in extent_keys(val.shape) if d != 1)
                    if source_axes != target_axes:
                        continue
            if _uses_symbol_as_data(node):
                # A subtree where a symbol was promoted to tensor data (`scalar_tensor(T)` etc.).
                # Baked, it becomes a constant right only at Tmax, and the 2-point evaluation can
                # agree by accident (the counterexample in the module docstring).
                continue
            if _references_symbolic_position(node):
                # Forms referencing a position on a symbolic axis and index with a mask index
                # (targets and reasons in _references_symbolic_position).
                continue
            if all(dep.name in foldable for dep in node.all_input_nodes):
                foldable.add(node.name)
        return foldable

    def _fold_eval(self, node: Node, point: str) -> Any:
        """Evaluate an eligible subtree for real at the given point (all symbols substituted)."""
        cache = self._fold_cache.setdefault(point, {})
        cached = cache.get(node.name)
        if cached is not None:
            return cached
        if node.op == "placeholder":
            return self.state[self.placeholder_fqn[node.name]]
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            # A node that carries no tensor is evaluated by substituting the expression (not run).
            return self._sym_value(val, point) if isinstance(val, torch.SymInt) else int(val)

        def resolve(arg: Any) -> Any:
            if isinstance(arg, Node):
                return self._fold_eval(arg, point)
            if isinstance(arg, torch.SymInt):
                return self._sym_value(arg, point)
            if isinstance(arg, (list, tuple)):
                return type(arg)(resolve(item) for item in arg)
            return arg

        out = node.target(
            *resolve(tuple(node.args)), **{k: resolve(v) for k, v in node.kwargs.items()}
        )
        cache[node.name] = out
        return out

    def _add_const(self, tensor: torch.Tensor, where: str) -> str:
        """Deduplicate a constant tensor into an initializer and return its IR value name.

        Semantic dtypes are f32 and **i32** (ADR 0010 — indices of the relative-position bucket
        table). i64 is dropped to i32 by boundary normalization (out of range fails loudly). There
        is no bool initializer in the vocabulary.
        """
        data = tensor.detach().contiguous()
        if data.dtype is torch.int64:
            # MUST: the single conversion point is normalize_boundary_tensor (with the range
            # check — ADR 0009).
            data = normalize_boundary_tensor(data, where)
        storage = _CONST_STORAGE.get(data.dtype)
        if storage is None:
            raise NotImplementedError(
                f"{where}: a constant of dtype {tensor.dtype} cannot be an IR v1 initializer"
                f" ({' / '.join(sorted(set(_CONST_STORAGE.values())))} only)"
            )
        dtype = DTYPE_NAMES[data.dtype]
        # Fed through incremental updates (so that the `tobytes()` copy plus the concatenation copy
        # do not take one constant's worth of RAM twice). The fed byte sequence stays "declaration
        # JSON ‖ raw bytes" **unchanged**, so the digest does not change by a single bit.
        hasher = hashlib.sha256()
        hasher.update(json.dumps([str(data.dtype), list(data.shape)]).encode())
        hasher.update(memoryview(data.numpy()).cast("B"))
        digest = hasher.hexdigest()
        # MUST: the identity key is the **full hexdigest**. Matching on the 16 hex (64 bit)
        # shortened for the name would collapse 2 colliding constants into one without comparing
        # contents = a graph that silently shares a different value. Only the name keeps the
        # short form, as before (IR readability).
        existing = self._const_by_digest.get(digest)
        if existing is not None:
            return existing
        short = digest[:16]
        name = f"const_{short}"
        key = f"const.{short}"
        if name in self.graph.initializers or key in self.tensors:
            # The full digests differ but the short forms match (= a name collision).
            # Probabilistically unreachable, but letting it through would give a silently wrong
            # graph in which the overwrite erases the earlier constant, so it fails.
            raise AssertionError(
                f"{where}: constant name '{name}' collided with another constant (digest {digest})"
            )
        self.tensors[key] = data
        self.graph.initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=storage))
        self.graph.values[name] = IrValue(dtype=dtype, shape=[int(d) for d in data.shape])
        self._const_by_digest[digest] = name
        return name

    def _emit_folded(self, node: Node) -> str:
        """Emit a folding-frontier node as "Tmax constant + sym_prefix_slice" (ADR 0010).

        With no symbolic dimension it is the constant itself (as before). With one, a node is added
        that cuts out the prefix of the constant baked at Tmax at run time.
        """
        where = f"folded constant {node.name} ({node.target})"
        folded = self._fold_eval(node, FOLD_POINT_MAX)
        val = self._meta_val(node)
        slices: list[dict[str, int]] = []
        symbols: set[str] = set()
        for index, dim in enumerate(val.shape):
            if not isinstance(dim, torch.SymInt) or not dim.node.expr.free_symbols:
                continue
            expr = self._sym_parts(dim)
            symbols.add(expr.sym)
            slices.append({"dim": index, "coeff": expr.coeff, "offset": expr.offset})
        if len(symbols) > 1:
            # sym_prefix_slice has one sym per node (IR v1's attrs contract). A constant spanning
            # several symbols has no representation, so it fails instead of being folded silently.
            raise NotImplementedError(
                f"{where}: a folded constant spanning several symbols {sorted(symbols)} is not"
                " supported by IR v1"
            )
        sym = next(iter(symbols)) if symbols else None
        self._check_prefix_commutes(node, folded, sym, slices)
        const_name = self._add_const(folded, where)
        # MUST: check that the dtype / shape of the baked tensor agree with the meta declaration.
        # Letting a disagreement through makes sym_prefix_slice's input and output point at
        # different buffers.
        declared = self.graph.values[const_name]
        expected_dtype = self._dtype(val, where)
        if declared.dtype != expected_dtype:
            raise AssertionError(
                f"{where}: dtype {declared.dtype} of the baked constant differs from the declared"
                f" {expected_dtype}"
            )
        if not slices:
            if declared.shape != self._shape(val):
                raise AssertionError(
                    f"{where}: shape {declared.shape} of the baked constant differs from the"
                    f" declared {self._shape(val)}"
                )
            return const_name
        # MUST: the prefix slice runs as a strided materializing copy (no new kernel — ADR 0010).
        # The rank cap applies as soon as there is even one symbolic dimension, so it fails here
        # before emitting (empty slices = the constant itself, which has no dispatch and so no
        # cap).
        rank = len(val.shape)
        if not 1 <= rank <= STRIDED_RANK:
            raise NotImplementedError(
                f"{where}: a symbolic prefix slice of rank {rank} cannot run on the strided kernel"
                f" (rank ≤ {STRIDED_RANK})"
            )
        return self._emit(
            "sym_prefix_slice", [const_name], node.name, val, {"sym": sym, "slices": slices}
        )

    def _check_prefix_commutes(
        self, node: Node, folded: Any, sym: str | None, slices: Sequence[Mapping[str, int]]
    ) -> None:
        """**Observe by 2-point evaluation** that a fold commutes with a prefix slice (ADR 0010).

        A subtree that uses a symbol as a "value" (`arange(T-1, -1, -1)`, `full([T], T)` etc.) can
        be built from allowlisted ops alone, and the leading elements of the constant baked at Tmax
        then differ from the right run-time values — the numbers change with no exception or
        warning, so it is **stopped by a check, not by a declaration**.

        The prefix length is evaluated as `coeff·sym+offset` (the coefficient is a linear map of the
        length, so commutativity means the same). With several symbols all of them move together at
        the second point (moving one at a time only sees cross-sections with the others fixed at
        Tmax and misses product forms between symbols).

        NOTE: the comparison is **bit-exact** ({@link _bitwise_equal}). Numeric equality lets
        `-0.0 == 0.0` through, so a match down to the sign bit is required. NaN passes only when
        the payload matches too.
        """
        probed = self._fold_eval(node, FOLD_POINT_PROBE)
        expected = folded
        if sym is not None and isinstance(expected, torch.Tensor):
            bound = self._point_value(sym, FOLD_POINT_PROBE)
            for entry in slices:
                length = entry["coeff"] * bound + entry["offset"]
                expected = expected.narrow(entry["dim"], 0, length)
        if (
            isinstance(probed, torch.Tensor)
            and isinstance(expected, torch.Tensor)
            and _bitwise_equal(probed, expected)
        ):
            return
        points = {
            name: (
                self._point_value(name, FOLD_POINT_MAX),
                self._point_value(name, FOLD_POINT_PROBE),
            )
            for name in self.sym_names.values()
        }
        raise NotImplementedError(
            f"folded subtree does not commute with a prefix slice (it uses a symbol as a value): "
            f"{node.name} ({node.target}) — the prefix of the constant baked at Tmax and the"
            f" result at the second evaluation point do not match"
            f" (symbol: Tmax, second point = {points})"
        )

    # ---- Conversion -------------------------------------------------------

    def _dim(self, dim: Any) -> int | str:
        if isinstance(dim, torch.SymInt):
            if dim.node.expr.free_symbols:
                return format_dim(self._sym_parts(dim))
            return int(dim)
        return int(dim)

    def _shape(self, tensor: torch.Tensor) -> list[int | str]:
        return [self._dim(dim) for dim in tensor.shape]

    def _dtype(self, tensor: torch.Tensor, where: str) -> str:
        name = DTYPE_NAMES.get(tensor.dtype)
        if name is None:
            raise NotImplementedError(
                f"{where}: torch dtype {tensor.dtype} is not in IR v1's semantic dtype vocabulary"
                f" ({', '.join(DTYPE_NAMES.values())})"
            )
        return name

    def _meta_val(self, node: Node) -> torch.Tensor:
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            raise NotImplementedError(f"value that is not a tensor: {node.name} ({node.target})")
        return val

    def _register_value(self, name: str, val: torch.Tensor) -> None:
        self.graph.values[name] = IrValue(
            dtype=self._dtype(val, f"value '{name}'"), shape=self._shape(val)
        )

    def _emit(
        self,
        op: str,
        ins: list[str],
        out_name: str,
        val: torch.Tensor,
        attrs: Mapping[str, Any],
    ) -> str:
        """Add a node with CSE and return the output value name."""
        if op not in EMITTABLE_OPS:
            raise AssertionError(f"IR op not registered in EMITTABLE_OPS: {op}")
        # MUST: the CSE key includes attrs. The same (op, ins) with different attrs is a different
        # value (`cast(x, to=bool)` vs `cast(x, to=f32)`); dropping attrs makes one silently vanish.
        # MUST: the output shape is included too. reshape / expand have no attrs — "the declared
        # output shape is the target shape" (ADR 0011) — so with a key without shape, separate
        # `reshape`s of the same input collapse into one and one consumer silently reads a value of
        # a different shape.
        shape = self._shape(val)
        key = json.dumps([op, ins, dict(attrs), shape], sort_keys=True)
        cached = self._cse.get(key)
        if cached is not None:
            return cached
        self.graph.nodes.append(IrNode(op=op, ins=list(ins), outs=[out_name], attrs=dict(attrs)))
        self._register_value(out_name, val)
        self._cse[key] = out_name
        return out_name

    def convert(self) -> tuple[IrGraph, dict[str, torch.Tensor]]:
        foldable = self._classify_foldable()
        self._assign_input_symbols()
        self._assert_all_ops_supported(foldable)

        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self._handle_placeholder(node)
            elif node.op == "call_function":
                if node.name in foldable:
                    continue  # evaluated from the frontier's consumer (DCE'd naturally if unused)
                self._handle_call(node, foldable)
            elif node.op == "output":
                self.graph.outputs = [
                    self._in_name(out, foldable) for out in self._user_outputs(node)
                ]
            else:
                raise NotImplementedError(f"unsupported node kind: {node.op}")
        # Declare only the symbols that actually reached a shape (a static graph gets []). An
        # unused symbol would make the runtime fail with "no input to bind it".
        self.graph.symbols = list(self.sym_names.values())
        # MUST: output names are a **set** (the outputs field of docs/ir-v2.md; the reader
        # `verify.parse_ir_graph` rejects duplicates). Duplicates arise on 2 paths: ① a forward
        # that returns the same value twice ② `_emit`'s CSE collapsing 2 structurally identical
        # outputs into one. Writing (`publish.publish_container`) does not go through
        # `parse_ir_graph`, so a distribution that "could be written but not read" would remain —
        # it fails at the exit of assembly.
        duplicated = sorted(
            {name for name in self.graph.outputs if self.graph.outputs.count(name) > 1}
        )
        if duplicated:
            raise NotImplementedError(
                "assigning the same value to several outputs is not supported by IR v1 (output"
                f" names are a set): duplicates {duplicated}"
            )
        # MUST: at the exit of conversion, recompute every node's output shape from the contracts
        # and match it against the declaration. The declaration comes from torch's meta, so a form
        # that disagrees with a contract rule (gather's leading-dimension match / conv1d's output
        # length / reshape's element count, etc.) passes as "legal for torch" and first fails when
        # the runtime builds the Session — that gap is closed here.
        assert_graph_shapes(self.graph)
        return self.graph, self.tensors

    def _assert_all_ops_supported(self, foldable: set[str]) -> None:
        """Collect every unsupported aten op, then fail once (no partial approximation)."""
        missing: list[Node] = []
        for node in self.gm.graph.nodes:
            if node.op != "call_function" or node.name in foldable:
                continue
            if node.target in SKIP_OPS or node.target in ATEN_HANDLERS:
                continue
            if not isinstance(node.meta.get("val"), torch.Tensor):
                # Shape machinery that only carries SymInt/int (the arity check fails it if it is
                # used as a value)
                continue
            missing.append(node)
        if missing:
            raise UnsupportedAtenOpsError(missing)

    def _user_outputs(self, out_node: Node) -> list[Node]:
        """Return only the user outputs among the output node's arguments.

        torch.export puts BUFFER_MUTATION and the like before USER_OUTPUT, so passing them through
        shifts the output list. IR v1 cannot express state updates, so they fail as unsupported
        instead of being dropped silently. The correspondence between specs and output arguments
        holds only "by position" (normalize does not update graph_signature, so
        `output_specs[].arg.name` can remain a node name that no longer exists).
        """
        specs = self.ep.graph_signature.output_specs
        args = list(out_node.args[0])
        if len(specs) != len(args):
            raise AssertionError(
                f"output_specs and output arguments differ in count: {len(specs)} vs {len(args)}"
            )
        other = sorted({s.kind.name for s in specs if s.kind is not OutputKind.USER_OUTPUT})
        if other:
            raise NotImplementedError(
                f"output kinds other than user output are not supported by IR v1: {other}"
            )
        return [
            arg
            for arg, spec in zip(args, specs, strict=True)
            if spec.kind is OutputKind.USER_OUTPUT
        ]

    def _handle_placeholder(self, node: Node) -> None:
        if node.name in self.user_inputs:
            val = self._meta_val(node)
            # MUST: inputs are declared only in inputs[] (also writing them in values{} breaks IR
            # v1's "exactly one declaration" and is rejected as a double declaration —
            # docs/ir-v2.md).
            self.graph.inputs.append(
                IrInput(
                    name=node.name,
                    dtype=self._dtype(val, f"input '{node.name}'"),
                    shape=self._shape(val),
                )
            )
            self.value_of[node.name] = node.name
            return
        if node.name not in self.placeholder_fqn:
            raise NotImplementedError(f"placeholder of unknown origin: {node.name}")
        # Parameter / buffer / constant — _in_name turns it into an initializer when it is
        # actually referenced

    def _materialize_initializer(self, node: Node) -> None:
        fqn = self.placeholder_fqn[node.name]
        tensor = self.state[fqn].detach().contiguous()
        where = f"initializer '{node.name}' ({fqn})"
        if tensor.dtype is not torch.float32:
            raise NotImplementedError(
                f"{where}: dtype {tensor.dtype} cannot be an IR v1 initializer (semantic f32 only)"
            )
        self.tensors[fqn] = tensor
        self.graph.initializers[node.name] = IrInitializer(
            tensor=fqn, storage=IrStorage(dtype="f32")
        )
        self._register_value(node.name, tensor)
        self.value_of[node.name] = node.name

    def _in_name(self, node: Node, foldable: set[str]) -> str:
        while node.name in self.alias_of:
            node = self.alias_of[node.name]
        existing = self.value_of.get(node.name)
        if existing is not None:
            return existing
        if node.op == "placeholder":
            # User inputs were registered by placeholder handling, so only initializers get here
            self._materialize_initializer(node)
            return self.value_of[node.name]
        if node.name in foldable:
            name = self._emit_folded(node)
            self.value_of[node.name] = name
            return name
        raise AssertionError(f"unresolved input value: {node.name} (topological order broken)")

    def _tensor_args(self, node: Node) -> list[Node]:
        """Return the tensor-valued Nodes among the positional arguments, in order, as inputs.

        all_input_nodes cannot be used because it deduplicates by the keys of an internal dict — a
        form like `x*x` that takes the same value as both operands correctly lists the same value
        name twice. Nodes that only carry a SymInt are not tensors and are excluded; a form that
        uses one as a value fails the input-count check.

        MUST: **flatten list arguments by exactly one level** (the first argument of
        `aten.cat([a, b], dim)` is a **list** of tensors; passing it through gives 0 tensor
        inputs). Only "a Node whose meta is a tensor" is picked up here, so int lists (permute's
        dims / view's sizes) pass through as before.
        """
        out: list[Node] = []
        for arg in node.args:
            items = arg if isinstance(arg, (list, tuple)) else (arg,)
            for item in items:
                if isinstance(item, Node) and isinstance(item.meta.get("val"), torch.Tensor):
                    out.append(item)
        return out

    def _check_arity(self, node: Node, spec: Emitted, ins: list[str]) -> None:
        if len(ins) == spec.ins:
            return
        symints = [
            arg.name
            for arg in node.args
            if isinstance(arg, Node) and not isinstance(arg.meta.get("val"), torch.Tensor)
        ]
        hint = (
            f" — a form that uses SymInt {symints} as a value is not supported by IR v1"
            if symints
            else ""
        )
        raise NotImplementedError(
            f"{node.target}: {spec.op} must have {spec.ins} tensor inputs, got {len(ins)}"
            f"{hint} (args={node.args}, kwargs={node.kwargs})"
        )

    def _handle_call(self, node: Node, foldable: set[str]) -> None:
        if node.target in SKIP_OPS:
            check = SKIP_OPS[node.target]
            if check is not None:
                check(node)
            self.alias_of[node.name] = node.all_input_nodes[0]
            return
        if not isinstance(node.meta.get("val"), torch.Tensor):
            # Shape machinery that only carries SymInt/int (sym_size / operator.mul etc.). The IR
            # value is already expressed as a shape expression — the arity check fails it if it is
            # consumed at a tensor position.
            return
        handler = ATEN_HANDLERS[node.target]  # missing ones failed in _assert_all_ops_supported
        if any(_holds_node(v) for v in node.kwargs.values()):
            raise NotImplementedError(
                f"{node.target}: a form that takes a tensor through kwargs is not supported"
                f" (kwargs={node.kwargs})"
            )
        ins = [self._in_name(arg, foldable) for arg in self._tensor_args(node)]
        spec = handler(node)
        for value, length in spec.synth_consts:
            # MUST: appended at the end in declaration order (optional slots are always at the
            # end — bias / affine / weight). They are synthesized through _add_const, so constants
            # with the same (value, length) collapse into one by digest (698 bias-less linears
            # still give one initializer per length).
            ins = [
                *ins,
                self._add_const(
                    torch.full((length,), value, dtype=torch.float32),
                    f"constant slot synthesis of {node.target} ({node.name}: {value} × {length})",
                ),
            ]
        self._check_arity(node, spec, ins)
        self.value_of[node.name] = self._emit(
            spec.op, ins, node.name, self._meta_val(node), spec.attrs
        )


def convert(
    ep: ExportedProgram, *, symbol_names: Sequence[str] = ("T",)
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Return the IR graph and the dict of stored tensors.

    symbol_names gives the symbol names that go into the IR, in order of appearance of the user
    input placeholders (= forward's argument order).
    """
    return Converter(ep, symbol_names=symbol_names).convert()


def curated_decompositions(preserved: Sequence[str] = PRESERVED_OP_PREFIXES):
    """Return the decomposition table with the "preserved high-level ops" taken out.

    Removes from default_decompositions the ops that, once core-decomposed, no longer map 1:1 to
    a kernel. The default is 13 ops (PRESERVED_OP_PREFIXES — ADR 0007's 9 ops plus leaky_relu in
    ADR 0015, rms_norm in ADR 0017, conv3d in ADR 0118 decision 1 and group_norm in ADR 0138
    decision 7).

    NOTE: preserving `rms_norm` is not enough. Only diffusers' `nn.RMSNorm` emits `aten.rms_norm`;
    hand-written decomposed forms (Qwen3 / DiT) are synthesized by normalize's `_fold_rms_norm`
    (the 2 supply routes of ADR 0016 / 0017).

    NOTE: SDPA (ADR 0023) is **not** in the default — the 13th is given only to targets that pass
    {@link PRESERVED_OP_PREFIXES_WITH_ATTENTION} explicitly.
    """
    from torch.export import default_decompositions

    table = default_decompositions()
    for overload in list(table):
        if str(overload).startswith(tuple(preserved)):
            del table[overload]
    return table
