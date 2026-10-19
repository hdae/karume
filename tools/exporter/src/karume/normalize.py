"""FX graph normalization after the curated decomp — equivalence rewrites that add no IR vocabulary.

Called before convert(). Each pass is a `(graph, stats) -> None` function and runs in the order
`_passes()` lists them (when adding a pass that depends on the order, leave the reason in a comment
there).

- `_drop_metadata_asserts`: removes `aten._assert_tensor_metadata`. eliminate_dead_code treats it
  as impure and keeps it, so already-replaced subtrees would survive through the assert.
- `_pow_neg_half_to_rsqrt`: `pow.Tensor_Scalar(x, -0.5)` → `rsqrt(x)` (equivalent over the whole
  domain — equal for x>0, both +inf at x=0, both NaN for x<0). transformers' Gemma4RMSNorm writes
  pow instead of rsqrt because of torch/JAX compiler differences, so it is brought into the rsqrt
  form `_fold_rms_norm` picks up. An rsqrt left unfolded into rms_norm appears in the full list of
  unsupported ops (rsqrt is not in the IR vocabulary — it does not pass silently).
- `_fold_rms_norm`: `mul(weight, mul(x, rsqrt(mean(x²)+eps)))` → one `rms_norm` node
  (ADR 0016 / 0017. A hand-written RMSNorm cannot be folded by preservation — it needs an FX
  pattern match). The weightless form `mul(x, rsqrt(mean(x²)+eps))` is also picked up, **after
  the weighted fold** (Gemma4's `v_norm` = RMSNorm with_scale=False. The ones synthesis is on the
  `_h_rms_norm` side).
- `_drop_safe_softmax_guard`: folds SDPA's safe-softmax guard (`eq(-inf)` / `any.dim` /
  `full_like` / `where`). If it can be proven inactive it is **removed** (ADR 0016); otherwise it
  is **constructively replaced** with `safe_softmax` (ADR 0044).
- `_additive_attn_mask`: a preserved SDPA's bool `attn_mask` → additive f32 (0 / −inf)
  (ADR 0023's mask contract. The same ops and constants as torch's own bool → additive
  conversion).
- `_lower_unit_expand`: GQA repeat_kv's `unsqueeze → expand → (clone) → view` → rank≤3.
- `_lower_split_unbind`: RoPE's "last-dimension split → width-1 slice → squeeze" → a
  last-dimension slice (forms whose consumers do not line up are skipped, not rewritten).
- `_lower_reshape_permute`: decomposes a rank≥5 `reshape → permute → reshape` (patchify /
  unpatchify) into a sequence of rank-4 adjacent transposes.
- `_compose_permute_chains`: removes identity `permute`s and composes adjacent `permute`s (p∘q).
  The intermediate materializing copies and dispatches go away accordingly.
- `_pow2_to_mul`: `pow.Tensor_Scalar(x, 2)` → `mul(x, x)`. The M0 vocabulary has no pow, but it
  has the equivalent mul (numerically identical too — it just becomes one multiplication).
- `_drop_identity_repeat`: `repeat(x, [1,…,1])` → x (recon §4-1. All 4 observed are this form).
- `_drop_identity_add`: `add(x, 0)` → x (recon §4-2. The 2 initial values of the c2p / p2c adds).
- `_promote_scalar_operands`: `add/sub/mul/div(tensor, scalar)` → binary op + rank-1 constant
  (recon §4-4. The only way that adds neither IR vocabulary nor attrs).
- `_eq_zero_to_not_bool`: `eq(x, 0)` → `bitwise_not(cast(x, bool))` (ADR 0015's ruling. It
  follows from the cast convention "x → bool is x != 0" and needs no new op).
- `_select_to_squeeze`: `select.int(axis of length 1, index 0)` → `squeeze` (general hygiene). A
  static axis of length 2 or more opens into `slice(width 1) + squeeze` (first seen in Gemma4's
  per-layer PLE slice).
- `_split_to_slices`: `split_with_sizes` + `getitem` → a sequence of `slice`s (ADR 0014. The IR
  has no multi-output op, so a split opens into one slice per extraction point).

MUST: `_fold_rms_norm` runs **before both** `_pow2_to_mul` and `_promote_scalar_operands`
(ADR 0016's order MUST). Even when `pow(x,2)` collapses into `mul(x,x)` the pattern match itself
survives because `_is_square_of` looks at both forms, but once the eps of `add(mean, eps)` is
promoted to a rank-1 constant the Python-scalar match misses and it can no longer be folded.

MUST: `_pow_neg_half_to_rsqrt` runs **before** `_fold_rms_norm` (the matcher sees only the rsqrt
form — afterwards, a pow-form RMSNorm would fall wholesale into the unsupported-op list).

MUST: the 3 chain passes (rank lowering) fire **only on forms containing a value of rank >
STRIDED_RANK** (ADR 0016). It is a safety line so they do not misfire on existing graphs (SBV2 /
DeBERTa are all rank ≤ 4), and it creates no new constant — "a rank the strided kernel cannot run"
is the firing condition itself.

MUST: `_drop_identity_add` runs **before** `_promote_scalar_operands`. In the reverse order
`add(x, 0)` becomes `add(x, const 0)`, is no longer an identity-removal target (the Python scalar
0), and a useless constant and one dispatch remain.

MUST: the ExportedProgram's graph_signature is not updated. Rewriting the output node leaves
`output_specs[].arg.name` as names of nodes that no longer exist, but this is harmless because the
only consumer, convert._user_outputs, matches specs and output arguments "by position" (keeping
the names in sync would be duplicated management of derived state).
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.export import ExportedProgram
from torch.fx import Graph, Node

from karume.extents import extent_key, extent_keys
from karume.ops import STRIDED_RANK

aten = torch.ops.aten

#: The 2 evaluation points that look for counterexamples to the safe-softmax guard. Passing them
#: does not prove inactivity over every length, so symbol-dependent masks are detected separately
#: and keep safe_softmax.
GUARD_PROBE_LENGTHS = (5, 9)

NEG_INF = float("-inf")

#: Flag set on a softmax node whose safe-softmax guard was constructively replaced (ADR 0044).
#:
#: The FX graph layer has no aten op called "safe_softmax", so the fact that the guard was folded
#: travels in the meta of the original softmax node, and `aten_handlers._h_softmax` branches the op
#: name on it. **Do not put a pass that clears meta after this pass** — if the flag drops, it is
#: baked as a plain softmax and all-(-inf) rows silently become NaN.
SAFE_SOFTMAX_META = "karume_safe_softmax"

#: Reshape-like ops that keep the logical element order (forms the rank-lowering passes may
#: regroup freely).
_RESHAPE_OPS = (
    aten.view.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.unsqueeze.default,
)


def _val(node: Node) -> torch.Tensor:
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise NotImplementedError(
            f"a value that is not a tensor cannot be rank-lowered: {node.name}"
        )
    return val


def _rank(node: Any) -> int:
    val = node.meta.get("val") if isinstance(node, Node) else None
    return val.dim() if isinstance(val, torch.Tensor) else 0


def _static_size(dim: Any) -> int | None:
    """Concrete value of a dimension length (None if it contains a symbol).

    MUST: do not take a SymInt with `int()` — it silently specializes to the export-time hint, and
    "a symbolic dimension whose hint is 1" turns into a squeeze and a silent wrong value at run
    time.
    """
    if isinstance(dim, torch.SymInt):
        return None if dim.node.expr.free_symbols else int(dim)
    return int(dim)


def _static_index(value: Any) -> int | None:
    """Concrete value of an index or range end (None for a symbol, an FX Node or `None`).

    MUST: for the same reason as {@link _static_size}, do not take a SymInt with `int()`. In
    addition, 2 representations are explicitly mapped to None — ① a symbolic value passed as an
    **FX Node** (`_static_size` hits a TypeError on `int(Node)`, so passing it through gives a bare
    crash instead of a diagnostic) ② an omission that arrives as an actual argument, as in
    `slice(start=None)`. `aten.select.int`'s index and `aten.slice.Tensor`'s start / end are SymInt
    in the schema, so taking the hint gives a silent wrong value at run time.
    """
    if isinstance(value, Node) or value is None:
        return None
    return _static_size(value)


def _numel(shape: Any) -> Any:
    return math.prod([1, *shape])


def _is_reshape(node: Any) -> bool:
    return isinstance(node, Node) and node.op == "call_function" and node.target in _RESHAPE_OPS


def _sole_user(node: Node) -> Node | None:
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _insert(graph: Graph, anchor: Node, target: Any, args: tuple, ref: torch.Tensor, **kwargs: Any):
    """Insert a node right before `anchor` and synthesize its meta["val"] in fake mode.

    convert requires `meta["val"]` (a FakeTensor) on every node, so an inserted node without a
    declaration would not make it to the exit of conversion (assert_graph_shapes).
    """
    with graph.inserting_before(anchor):
        node = graph.call_function(target, args, kwargs)
    resolved = tuple(a.meta["val"] if isinstance(a, Node) else a for a in args)
    with ref.fake_mode:
        node.meta["val"] = target(*resolved, **kwargs)
    return node


def _insert_contiguous(graph: Graph, anchor: Node, src: Node, ref: torch.Tensor) -> Node:
    """Materializing copy placed right after a permute / expand.

    On the IR side permute / expand are strided materializing copies, so this is semantically
    redundant (convert's SKIP_OPS absorbs it as an identity), but without it the meta synthesis of
    the following view (torch's view rules) does not go through.
    """
    return _insert(
        graph, anchor, aten.clone.default, (src,), ref, memory_format=torch.contiguous_format
    )


def _as_contiguous(graph: Graph, anchor: Node, src: Node, ref: torch.Tensor) -> Node:
    """Materialize a non-contiguous value before a view (a contiguous one is returned as is)."""
    return src if _val(src).is_contiguous() else _insert_contiguous(graph, anchor, src, ref)


def _replace(graph: Graph, old: Node, new: Node) -> None:
    old.replace_all_uses_with(new)
    graph.erase_node(old)


def _drop_metadata_asserts(graph: Graph, stats: Counter) -> None:
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is aten._assert_tensor_metadata.default:
            graph.erase_node(node)
            stats["drop_metadata_assert"] += 1


# ---- Folding the RMS normalization pattern (ADR 0016 / 0017)------------------------


def _pow_neg_half_to_rsqrt(graph: Graph, stats: Counter) -> None:
    """`aten.pow(x, -0.5)` → `rsqrt(x)` (normalization ahead of the rms_norm pattern).

    transformers' Gemma4RMSNorm writes `pow(ms, -0.5)` instead of `rsqrt` because of torch/JAX
    compiler differences (the note in modeling_gemma4.py). It is equivalent to rsqrt over the whole
    domain (equal for x>0, both +inf at x=0, both NaN for x<0 — the same treatment as
    `_is_square_of` regarding the pow/mul forms as equivalent). An rsqrt left unfolded into
    rms_norm is not in the IR vocabulary, so it appears in the full list of unsupported ops (no
    silent approximation).

    MUST: the dtype must be unchanged (same wording as `_pow2_to_mul` — a form with promotion is
    not an identity rewrite). The exponent is the float -0.5 only (there is no -0.5 for int, but
    the bool exclusion is kept for consistency).
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.pow.Tensor_Scalar:
            continue
        exponent = node.args[1]
        if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
            continue
        if float(exponent) != -0.5:
            continue
        src = node.args[0]
        if not isinstance(src, Node):
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if not isinstance(src_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if src_val.dtype is not out_val.dtype:
            continue
        with graph.inserting_before(node):
            replacement = graph.call_function(aten.rsqrt.default, (src,))
        # convert requires meta["val"] (a FakeTensor) on every node.
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        stats["pow-0.5->rsqrt"] += 1


def _eps_operand(node: Any) -> float | None:
    """Return the eps of `add(mean, eps)` (the scalar form of `add.Tensor` and `add.Scalar`)."""
    if not isinstance(node, Node) or node.op != "call_function":
        return None
    if node.target not in (aten.add.Tensor, aten.add.Scalar):
        return None
    if node.kwargs.get("alpha", 1) != 1 or len(node.args) != 2:
        return None
    rhs = node.args[1]
    if isinstance(rhs, bool) or not isinstance(rhs, (int, float)):
        return None
    return float(rhs)


def _is_last_dim_mean(node: Any) -> bool:
    """Whether this is a keepdim `mean.dim` over the last dimension only (rms_norm reduces it)."""
    if not isinstance(node, Node) or node.op != "call_function" or node.target is not aten.mean.dim:
        return False
    src = _val(node.args[0])
    dims = [int(d) for d in node.args[1]]
    keepdim = bool(node.args[2]) if len(node.args) > 2 else bool(node.kwargs.get("keepdim", False))
    return keepdim and dims in ([-1], [src.dim() - 1])


def _is_square_of(node: Any, x: Node) -> bool:
    """Whether the node is x². Depending on the decomp, both `pow(x,2)` and `mul(x,x)` appear."""
    if not isinstance(node, Node) or node.op != "call_function":
        return False
    if node.target is aten.pow.Tensor_Scalar:
        return node.args[0] is x and node.args[1] == 2
    return node.target is aten.mul.Tensor and node.args[0] is x and node.args[1] is x


def _rms_norm_operands(node: Node) -> tuple[Node, Node, float] | None:
    """Decompose `mul(weight, mul(x, rsqrt(mean(x²)+eps)))` into (x, weight, eps).

    Both muls are commutative, so both orders are tried (observed: Qwen3 has
    `mul(mul(x, rsqrt), weight)`, DiT's QK norm the reverse).
    """
    if node.op != "call_function" or node.target is not aten.mul.Tensor:
        return None
    for scaled, weight in (node.args[:2], node.args[1::-1]):
        if not isinstance(scaled, Node) or not isinstance(weight, Node):
            continue
        if scaled.op != "call_function" or scaled.target is not aten.mul.Tensor:
            continue
        for x, rsqrt in (scaled.args[:2], scaled.args[1::-1]):
            if not isinstance(x, Node) or not isinstance(rsqrt, Node):
                continue
            if rsqrt.op != "call_function" or rsqrt.target is not aten.rsqrt.default:
                continue
            eps = _eps_operand(rsqrt.args[0])
            if eps is None:
                continue
            mean = rsqrt.args[0].args[0]
            if not _is_last_dim_mean(mean) or not _is_square_of(mean.args[0], x):
                continue
            return x, weight, eps
    return None


def _weightless_rms_norm_operands(node: Node) -> tuple[Node, float] | None:
    """Decompose `mul(x, rsqrt(mean(x²)+eps))` into (x, eps) (the weightless form).

    First seen in Gemma4's `v_norm` (RMSNorm with_scale=False). This form is **the same shape as
    the inner mul** of the weighted form, so it is applied only in the second walk of
    {@link _fold_rms_norm} (after the nodes killed by the weighted fold are excluded) — breaking the
    order would split a weighted norm into "weightless rms_norm + bare mul" and lose the weight
    fusion.
    """
    if node.op != "call_function" or node.target is not aten.mul.Tensor:
        return None
    for x, rsqrt in (node.args[:2], node.args[1::-1]):
        if not isinstance(x, Node) or not isinstance(rsqrt, Node):
            continue
        if rsqrt.op != "call_function" or rsqrt.target is not aten.rsqrt.default:
            continue
        eps = _eps_operand(rsqrt.args[0])
        if eps is None:
            continue
        mean = rsqrt.args[0].args[0]
        if not _is_last_dim_mean(mean) or not _is_square_of(mean.args[0], x):
            continue
        return x, eps
    return None


def _fold_rms_norm(graph: Graph, stats: Counter) -> None:
    """Fold the decomposed form of RMS normalization into one `aten.rms_norm` node.

    This is supply route ② of ADR 0017. Hand-written implementations (Qwen3 / DiT) and diffusers'
    `nn.RMSNorm` lower to the same pow/mean/rsqrt form under the core decomp, so one pattern picks
    them up whatever the source. The weighted form is folded in the first walk, and the weightless
    form (Gemma4 `v_norm`) is picked up in the **second walk** ({@link
    _weightless_rms_norm_operands} — the reason for the order is in its docstring).

    MUST: a form whose weight is not **rank-1 of the last-dimension length** is not folded.
    rms_norm's contract is "the source of truth for the normalized length = the weight's length"
    (ADR 0017), and folding a broadcast-shaped weight would give semantics different from the
    contract. Left unfolded, mean/rsqrt appear in the full list as unsupported ops (no silent
    approximation).
    MUST: eps is a finite positive number. IR v1 rejects a non-finite eps at the value level.
    """
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        matched = _rms_norm_operands(node)
        if matched is None:
            continue
        x, weight, eps = matched
        x_val, weight_val = _val(x), _val(weight)
        length = _static_size(x_val.shape[-1])
        if length is None or list(weight_val.shape) != [length]:
            continue
        if not math.isfinite(eps) or eps <= 0:
            continue
        new = _insert(graph, node, aten.rms_norm.default, (x, [length], weight, eps), x_val)
        # The output has the original mul's shape and dtype (the meta carries over as the source
        # of truth).
        new.meta.update(node.meta)
        _replace(graph, node, new)
        stats["rms_norm"] += 1
    # Second walk for the weightless form. The weighted fold leaves the inner mul as a dead node
    # with 0 users (DCE is at the end of normalize_graph), so this MUST be applied only to live
    # nodes — without that check the inside of an already folded norm is picked up twice and the
    # weight fusion breaks.
    for node in list(graph.nodes):
        if node.op != "call_function" or not node.users:
            continue
        matched_weightless = _weightless_rms_norm_operands(node)
        if matched_weightless is None:
            continue
        x, eps = matched_weightless
        x_val = _val(x)
        length = _static_size(x_val.shape[-1])
        if length is None:
            continue
        if not math.isfinite(eps) or eps <= 0:
            continue
        # The weight is passed as None (synthesizing ones up to arity 2 is `_h_rms_norm`'s job —
        # ADR 0017's "×1 is an exact identity" is not copied here into a second place).
        new = _insert(graph, node, aten.rms_norm.default, (x, [length], None, eps), x_val)
        new.meta.update(node.meta)
        _replace(graph, node, new)
        stats["rms_norm_weightless"] += 1


# ---- Removing the safe-softmax guard (ADR 0016)----------------------------------


def _introduces_neg_inf(node: Node) -> bool:
    """Whether an argument is a **non-finite** float literal (judged by its value rounded to f32).

    Not only `-inf` but also `+inf` and NaN count as sources — `+inf` turns into -inf through neg /
    sub / a negative mul (`scores - mask * float("inf")`), so the sign is not tracked and the test
    falls on the conservative side. The value is judged with the same rounding as torch's scalar →
    f32 conversion (`-1e39` is finite as a double but -inf in f32).

    NOTE: forms where operations between finite literals overflow in f32
    (`finfo.min + finfo.min`) are not caught — that is not decided by one literal's value and would
    need a test that tracks the range of the operation.
    """

    def holds(value: Any) -> bool:
        if isinstance(value, float):
            return not math.isfinite(float(torch.tensor(value, dtype=torch.float32)))
        if isinstance(value, (list, tuple)):
            return any(holds(item) for item in value)
        return False

    return holds(node.args) or holds(tuple(node.kwargs.values()))


#: Ops whose output is finite (or NaN) even when the input is -inf — propagation of -inf stops
#: here. softmax turns "a row of all -inf" into NaN but not into -inf. NaN is not caught by
#: `eq(-inf)`, so it does not affect the guard's firing condition (the proof that every row is
#: finite is established separately at each step).
_NEG_INF_SANITIZERS = (
    aten.softmax.int,
    aten._softmax.default,
    aten.sigmoid.default,
    aten.tanh.default,
    aten.exp.default,
)

#: Ops that **produce** -inf from finite inputs — they hold no non-finite literal in their
#: arguments, so {@link _introduces_neg_inf} does not count them as sources.
#:
#: `log(0)` / `log1p(-1)` give -inf as their defined output, so they are sources unconditionally.
#: MUST: `rsqrt` / `pow(x, -0.5)` are not added — `x == 0` gives **+inf** and `x < 0` gives NaN;
#: neither makes -inf (the same observation of the domain as in `_pow_neg_half_to_rsqrt`).
#: Widening the sources here would turn an rsqrt left unfolded into rms_norm into a -inf source on
#: the score side, and every attention would fall to the replacement with `safe_softmax` (branch
#: ②).
_NEG_INF_PRODUCERS = (aten.log.default, aten.log1p.default)

#: Ops that produce ±inf only when the divisor is 0 (whether they are a source is decided by the
#: **divisor**).
_NEG_INF_DIVISIONS = (aten.div.Tensor, aten.div.Scalar)


def _produces_neg_inf(node: Node) -> bool:
    """Whether an op can produce -inf from finite inputs (a source without a -inf literal).

    A division counts as a source only in "forms where the divisor cannot be shown non-zero".
    Deciding by op name alone would make even attention's normal path (`scores / sqrt(d)`) a
    source, and practically every guard site would fall to branch ② — that is exactly the
    "abolish branch ①" proposal, which is not what is taken here. A non-zero numeric literal proves
    the divisor is not 0, so it is excluded from the sources.
    """
    if node.target in _NEG_INF_PRODUCERS:
        return True
    if node.target not in _NEG_INF_DIVISIONS:
        return False
    divisor = node.args[1] if len(node.args) > 1 else None
    if isinstance(divisor, bool) or not isinstance(divisor, (int, float)):
        return True  # tensor divisors and unreadable forms fall to "might be 0" (conservative)
    return divisor == 0


class _Placeholders:
    """Table of placeholder values (used by the safe-softmax guard's inactivity proof — ADR 0016).

    torch.export **lifts all parameters / buffers / lifted constants as placeholders**, so assuming
    "placeholders are finite" would let a `register_buffer`'d -inf mask through. The actual values
    can be looked up in the ExportedProgram (`state_dict` / `constants`).

    The only exception is the user's run-time inputs (`graph_signature.user_inputs`): their values
    are known only at run time, so they are **assumed finite**. Passing -inf as a graph input is
    outside the runtime contract, and IR v1's rejection of non-finite values applies only to
    initializers.

    A placeholder whose value cannot be looked up falls to "might be a -inf source" (the caller
    fails loudly). The finiteness test is **lazy** — scanning even huge weights the test never
    touches would slow export down by seconds.
    """

    def __init__(self, ep: ExportedProgram) -> None:
        signature = ep.graph_signature
        targets = {
            **signature.inputs_to_parameters,
            **signature.inputs_to_buffers,
            **signature.inputs_to_lifted_tensor_constants,
        }
        user_inputs = set(signature.user_inputs)
        self._values: dict[str, torch.Tensor] = {}
        self._unknown: set[str] = set()
        for node in ep.graph_module.graph.nodes:
            if node.op != "placeholder" or node.name in user_inputs:
                continue
            target = targets.get(node.name)
            value = ep.state_dict.get(target, ep.constants.get(target)) if target else None
            if isinstance(value, torch.Tensor):
                self._values[node.name] = value
            else:
                self._unknown.add(node.name)
        self._finite: dict[str, bool] = {}

    def value(self, name: str) -> torch.Tensor | None:
        """The placeholder's actual value (None for a user input or an unknown value)."""
        return self._values.get(name)

    def can_be_neg_inf(self, name: str) -> bool:
        """Whether this placeholder can hold non-finite values (user inputs: assumed finite)."""
        if name in self._unknown:
            return True
        value = self._values.get(name)
        if value is None:
            return False
        cached = self._finite.get(name)
        if cached is None:
            cached = not value.is_floating_point() or bool(torch.isfinite(value).all())
            self._finite[name] = cached
        return not cached


def _can_be_neg_inf(node: Node, placeholders: _Placeholders) -> bool:
    """Decide whether the node's value can be -inf by walking back through its dependencies.

    In multi-layer attention the mask enters the upper layers' dependency cone through the lower
    layers' softmax, so "is there a -inf constant in the dependency cone" alone would flag every
    layer falsely. Ops with a finite range are treated as walls to propagation, and only the paths
    that can actually become -inf are looked at. Sources are not only nodes holding a non-finite
    literal (value rounded to f32) ({@link _introduces_neg_inf}) but also ops that **produce** -inf
    from finite inputs ({@link _produces_neg_inf}). Lifted constants / buffers / parameters are
    judged by their actual values in {@link _Placeholders}; any other non-call_function node falls
    to "might be a -inf source".
    """
    seen: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        if current.op == "placeholder":
            if placeholders.can_be_neg_inf(current.name):
                return True
            continue
        if current.op != "call_function":
            return True
        val = current.meta.get("val")
        if isinstance(val, torch.Tensor) and val.dtype is torch.bool:
            continue
        if current.target in _NEG_INF_SANITIZERS:
            continue
        if _introduces_neg_inf(current) or _produces_neg_inf(current):
            return True
        stack.extend(current.all_input_nodes)
    return False


def _subs_symint(value: Any, length: int) -> Any:
    if not isinstance(value, torch.SymInt):
        return value
    expr = value.node.expr
    if not expr.free_symbols:
        return int(expr)
    symbols = list(expr.free_symbols)
    if len(symbols) != 1:
        raise NotImplementedError(f"an expression with several symbols cannot be evaluated: {expr}")
    return int(expr.subs(symbols[0], length))


def _eval_static(
    node: Node, length: int, cache: dict[str, Any], placeholders: _Placeholders
) -> Any:
    """Evaluate a subtree of only constants and shape symbols for real at symbolic length `length`.

    Lifted constants / buffers / parameters resolve to their actual values (a `register_buffer`'d
    mask is not a form that "depends on input values" but a constant whose value is fixed at
    export time).

    MUST: fail as soon as a user-input placeholder is touched. A mask depending on run-time inputs
    is a form whose "inactivity" cannot be proven, so it is not let through silently; the caller
    fails loudly.
    """
    cached = cache.get(node.name)
    if cached is not None:
        return cached
    if node.op == "placeholder":
        value = placeholders.value(node.name)
        if value is None:
            raise NotImplementedError(f"placeholder without an actual value: {node.name}")
        return value
    if node.op != "call_function":
        raise NotImplementedError(f"node that is not a constant subtree: {node.name} ({node.op})")
    val = node.meta.get("val")
    if isinstance(val, torch.SymInt):
        return _subs_symint(val, length)

    def resolve(arg: Any) -> Any:
        if isinstance(arg, Node):
            return _eval_static(arg, length, cache, placeholders)
        if isinstance(arg, torch.SymInt):
            return _subs_symint(arg, length)
        if isinstance(arg, (list, tuple)):
            return type(arg)(resolve(item) for item in arg)
        return arg

    out = node.target(*resolve(tuple(node.args)), **{k: resolve(v) for k, v in node.kwargs.items()})
    cache[node.name] = out
    return out


def _guard_parts(node: Node) -> tuple[Node, Node] | None:
    """Match `where(logical_not(any(logical_not(eq(src,-inf)), -1)), full_like(sm,0), sm)`.

    On a match, returns (softmax node, softmax input). This is SDPA's safe-softmax guard itself
    ("a row whose elements are all -inf returns 0"); anything but an exact match is left alone.
    """
    if node.op != "call_function" or node.target is not aten.where.self:
        return None
    if len(node.args) != 3:
        return None
    cond, zeros, softmax = node.args
    if not all(isinstance(arg, Node) for arg in (cond, zeros, softmax)):
        return None
    if softmax.target is not aten.softmax.int or zeros.target is not aten.full_like.default:
        return None
    if zeros.args[0] is not softmax or zeros.args[1] != 0:
        return None
    if cond.target is not aten.logical_not.default:
        return None
    reduced = cond.args[0]
    if not isinstance(reduced, Node) or reduced.target is not aten.any.dim:
        return None
    if reduced.args[1] not in (-1, _rank(reduced) - 1):
        return None
    if not (len(reduced.args) > 2 and reduced.args[2]):
        return None
    negated = reduced.args[0]
    if not isinstance(negated, Node) or negated.target is not aten.logical_not.default:
        return None
    equality = negated.args[0]
    if not isinstance(equality, Node) or equality.target is not aten.eq.Scalar:
        return None
    if equality.args[1] != NEG_INF:
        return None
    src = softmax.args[0]
    return (softmax, src) if equality.args[0] is src else None


class _UnprovenGuardError(NotImplementedError):
    """A form where the inactivity test itself did not stand because evaluation raised.

    It is routed to `safe_softmax`. Deriving from `NotImplementedError` keeps the caller's single
    catch. Only the statistics are split, so as **not to create a path that silently absorbs
    exceptions** (a real implementation bug slipping in here shows up as a count).
    """


def _assert_guard_inactive(src: Node, placeholders: _Placeholders) -> str:
    """Show that the guard cannot fire (no row whose elements are all -inf); return the reason.

    ① no node in the dependency cone of the softmax input brings in -inf → only finite values
      arrive, so it cannot fire
    ② -inf enters only through "one additive mask" → that mask is evaluated for real at 2 symbolic
       lengths and each row is observed to keep a finite element. If it is symbol-dependent,
       however, this is not used as a proof over every length.

    MUST: anything else is `NotImplementedError`. A form where removing the guard lets NaN flow
    downstream is not accepted here (the caller routes it to the constructive replacement with
    `safe_softmax` — ADR 0044. What this function returns is **strictly a proof that removal is
    safe**, and the test must not be loosened).
    """
    if not _can_be_neg_inf(src, placeholders):
        return "no-neg-inf"
    if src.op != "call_function" or src.target is not aten.add.Tensor:
        raise NotImplementedError(
            f"cannot prove the safe-softmax guard inactive: {src.name} ({src.target})"
            " — -inf enters through a path other than an additive mask"
        )
    left, right = src.args[0], src.args[1]
    if not isinstance(left, Node) or not isinstance(right, Node):
        raise NotImplementedError(
            f"the safe-softmax guard's addition is not between tensors: {src.name}"
        )
    scores, mask = (left, right) if _can_be_neg_inf(right, placeholders) else (right, left)
    if _can_be_neg_inf(scores, placeholders):
        raise NotImplementedError(
            f"safe-softmax guard: the score side has a -inf source too ({scores.name})"
        )
    for length in GUARD_PROBE_LENGTHS:
        try:
            evaluated = _eval_static(mask, length, {}, placeholders)
            rows_have_finite = bool(torch.isfinite(evaluated).any(dim=-1).all())
        except NotImplementedError:
            # A structural "cannot prove" from the test functions — returned to the caller as is.
            raise
        except Exception as cause:
            # Evaluation resolves placeholders to **actual values** while it substitutes symbols
            # with the probe length, so when the two meet in one expression torch raises (shape
            # mismatch, non-tensor). That is a failed test, not a broken exporter — it MUST be
            # normalized to the test result (`NotImplementedError`) so the list of exception types
            # does not leak to the caller.
            raise _UnprovenGuardError(
                f"cannot decide whether the safe-softmax guard is inactive: evaluating mask"
                f" {mask.name} at symbolic length {length} failed with {type(cause).__name__}"
            ) from cause
        if not rows_have_finite:
            raise NotImplementedError(
                f"the safe-softmax guard is not inactive: mask {mask.name} has a row of all -inf"
                f" at symbolic length {length}"
            )
    # A 2-point match does not cover the lengths not evaluated. A symbol-dependent mask keeps the
    # guard and is replaced with safe_softmax (docs/decisions/0044-runtime-attention-mask.md).
    seen: set[str] = set()
    pending = [mask]
    while pending:
        current = pending.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        value = current.meta.get("val")
        dimensions = value.shape if isinstance(value, torch.Tensor) else (value,)
        if any(isinstance(dim, torch.SymInt) and dim.node.expr.free_symbols for dim in dimensions):
            raise NotImplementedError(
                f"safe-softmax guard: inactivity of symbol-dependent mask {mask.name} over every"
                " length is unproven"
            )
        pending.extend(current.all_input_nodes)
    return "masked-rows-nonempty"


def _drop_safe_softmax_guard(graph: Graph, stats: Counter, *, placeholders: _Placeholders) -> None:
    """Fold SDPA's safe-softmax guard (ADR 0016 → made 2-step by ADR 0044).

    ① If inactivity is proven with actual values, the guard is **removed** (ADR 0016's original
       path — the output bytes of existing assets stay closed here).
    ② A form that cannot be proven (a run-time mask, a -inf source on the score side, a static
       mask that really has all -inf rows) does not fail loudly; the whole guard subtree is
       **rewritten into one safe_softmax node**. The graph operation is the same "where → softmax"
       replacement as ①; the only difference is setting {@link SAFE_SOFTMAX_META} on the softmax.

    ② is correct because, with finite scores (non-finite inputs are outside the contract), "all
    elements -inf ⇔ row max = -inf", and safe_softmax's contract (a row whose max is -inf is all 0)
    has **exactly** the same semantics as torch's guard (ADR 0044 decision 2). ① is tried first so
    that forms with a proof add no vocabulary (zero difference from existing IR).
    """
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        parts = _guard_parts(node)
        if parts is None:
            continue
        softmax, src = parts
        try:
            reason = _assert_guard_inactive(src, placeholders)
        except _UnprovenGuardError:
            # A form where the test did not stand because evaluation raised (counted apart from
            # forms structurally declared "cannot prove"). The replacement itself is the same.
            softmax.meta[SAFE_SOFTMAX_META] = True
            reason = "rewritten-unproven"
        except NotImplementedError:
            softmax.meta[SAFE_SOFTMAX_META] = True
            reason = "rewritten-safe"
        stats[f"softmax_guard:{reason}"] += 1
        _replace(graph, node, softmax)


def _additive_attn_mask(graph: Graph, stats: Counter) -> None:
    """Replace a preserved SDPA's **bool** `attn_mask` with an additive f32 mask (0 / −inf).

    Fused attention's mask contract is additive f32 only (ADR 0023). transformers' Gemma3 family,
    on the other hand, passes a bool band mask to SDPA, so preserving it runs into 2 guards: ① the
    dtype guard of `_h_attention` ② the guard that refuses bool constants as initializers.

    The rewrite is the single step `where(mask, scalar_tensor(0.0), scalar_tensor(-inf))`, which is
    exactly the conversion torch's own `_scaled_dot_product_attention_math` applies to a bool mask
    with **the same ops and the same constants** (it matches the form the decomposition path puts
    into the real graph — ADR 0023's bit identity holds on the preserved path as it is). where /
    scalar_tensor are both in `convert.FOLDABLE_OPS`, so constant folding turns them into an f32
    Tmax constant + `sym_prefix_slice`.

    MUST: the same bool mask gets **the same single** additive mask. Duplicating it per SDPA gives
    the same values (CSE and constant deduplication collapse them into one in the end), but runs the
    folding evaluation needlessly once per layer.
    """
    additive: dict[str, Node] = {}
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not aten.scaled_dot_product_attention.default
        ):
            continue
        mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask")
        if not isinstance(mask, Node) or _val(mask).dtype is not torch.bool:
            continue
        replacement = additive.get(mask.name)
        if replacement is None:
            ref = _val(mask)
            keep = _insert(
                graph, node, aten.scalar_tensor.default, (0.0,), ref, dtype=torch.float32
            )
            drop = _insert(
                graph, node, aten.scalar_tensor.default, (NEG_INF,), ref, dtype=torch.float32
            )
            replacement = _insert(graph, node, aten.where.self, (mask, keep, drop), ref)
            additive[mask.name] = replacement
        node.replace_input_with(mask, replacement)
        stats["additive_attn_mask"] += 1


# ---- The 3 rank-lowering passes (fire only on rank > STRIDED_RANK — ADR 0016)----------


def _lower_unit_expand(graph: Graph, stats: Counter) -> None:
    """Lower `unsqueeze → expand → (clone) → view` into an expand sequence of rank≤3.

    GQA's kv duplication (the decomposed form of `repeat_interleave(n, dim)`) is the only form that
    creates rank 5. Collapsing what is before and after the new axis into `[P, n, Q]` keeps the
    duplication semantics as they are at rank 3. `n == 1` (a configuration without GQA) makes the
    whole chain an identity.
    """
    for expand in list(graph.nodes):
        if expand.op != "call_function" or expand.target is not aten.expand.default:
            continue
        if _rank(expand) <= STRIDED_RANK:
            continue
        unsqueeze = expand.args[0]
        if not isinstance(unsqueeze, Node) or unsqueeze.target is not aten.unsqueeze.default:
            continue
        sink = _sole_user(expand)
        if sink is not None and sink.target is aten.clone.default:
            sink = _sole_user(sink)
        if sink is None or sink.target is not aten.view.default:
            continue
        src = unsqueeze.args[0]
        axis = int(unsqueeze.args[1]) % _rank(unsqueeze)
        src_shape = list(_val(src).shape)
        expanded_shape = list(_val(expand).shape)
        # MUST: shapes are matched with `extent_keys` (structural equality of sympy expressions).
        # A bare list `!=` calls `SymInt.__eq__` per element and decides by the hint values, and
        # on top of that the evaluation stacks a late guard substitution onto the already exported
        # ShapeEnv (the substitution reaches `convert._dim` and the dimension is silently baked
        # as static — the MUST at the top of extents.py).
        if extent_keys(expanded_shape[:axis]) != extent_keys(src_shape[:axis]):
            continue
        if extent_keys(expanded_shape[axis + 1 :]) != extent_keys(src_shape[axis:]):
            continue
        count = _static_size(expanded_shape[axis])
        if count is None:
            raise NotImplementedError(
                f"duplication along a symbolic dimension is not supported: {expand.name}"
            )
        out_shape = list(_val(sink).shape)
        if count == 1 and extent_keys(out_shape) == extent_keys(src_shape):
            _replace(graph, sink, src)
            stats["unit_expand->identity"] += 1
            continue
        head, tail = _numel(src_shape[:axis]), _numel(src_shape[axis:])
        ref = _val(src)
        flat = _insert(
            graph,
            sink,
            aten.view.default,
            (_as_contiguous(graph, sink, src, ref), [head, 1, tail]),
            ref,
        )
        wide = _insert(graph, sink, aten.expand.default, (flat, [head, count, tail]), ref)
        wide = _insert_contiguous(graph, sink, wide, ref)
        merged = _insert(graph, sink, aten.view.default, (wide, out_shape), ref)
        _replace(graph, sink, merged)
        stats["unit_expand->rank3"] += 1


def _unbind_consumers(view: Node, *, split_dim: int) -> list[tuple[Node, int, int]] | None:
    """`(squeeze, start, end)` list if every consumer of `view` is "width-1 slice → squeeze".

    The slice is along the split axis. If even one consumer does not fit, None (= this view is not
    an unbind).
    """
    rank = _rank(view)
    rewritten: list[tuple[Node, int, int]] = []
    for user in list(view.users):
        if user.target is not aten.slice.Tensor or len(user.args) < 4:
            return None
        if int(user.args[1]) % rank != split_dim:
            return None
        # start / end are `SymInt?` in the schema — a symbol (raw SymInt / FX Node) or an
        # explicit `None` is not specialized to the hint but falls to **skipping without touching
        # this view** (this function's MUST).
        start, end = _static_index(user.args[2]), _static_index(user.args[3])
        if start is None or end is None:
            return None
        squeeze = _sole_user(user)
        if end - start != 1 or squeeze is None:
            return None
        if squeeze.target not in (aten.squeeze.dim, aten.squeeze.dims) or len(squeeze.args) < 2:
            return None
        # What is dropped must be **exactly the split axis** — a squeeze that drops another
        # length-1 axis (batch axis 0, etc.) is not an unbind, so this view is left alone (the
        # caller's MUST).
        dims = squeeze.args[1] if squeeze.target is aten.squeeze.dims else [squeeze.args[1]]
        if {int(dim) % rank for dim in dims} != {split_dim}:
            return None
        rewritten.append((squeeze, start, end))
    return rewritten


def _lower_split_unbind(graph: Graph, stats: Counter) -> None:
    """Fold `view` (last dim split into (A,B)) → width-1 `slice` → `squeeze` into a last-dim slice.

    The form `apply_rotary_emb(use_real_unbind_dim=-2)` creates. The split is contiguous, so the
    slice at index s has exactly the same order as `[s·B, (s+1)·B)` of the original tensor's last
    dimension.

    MUST: rewrite only **after every consumer has matched** (rewriting only part of a split leaves
    the rest as broken references). If even one is unexpected (a slice not of width 1 / a consumer
    other than squeeze), **skip without touching that view** — the caught form is not necessarily
    an unbind, and failing here would kill a rank 5 of a different form in the middle of
    normalization. If a form that really does not drop to rank ≤ STRIDED_RANK remains, the guard on
    the convert side (the strided family's rank cap) fails it in the full list.
    """
    for view in list(graph.nodes):
        if view.op != "call_function" or view.target is not aten.view.default:
            continue
        if _rank(view) <= STRIDED_RANK:
            continue
        src = view.args[0]
        if not isinstance(src, Node):
            continue
        src_shape = list(_val(src).shape)
        view_shape = list(_val(view).shape)
        if len(view_shape) != len(src_shape) + 1 or extent_keys(view_shape[:-2]) != extent_keys(
            src_shape[:-1]
        ):
            continue
        inner = _static_size(view_shape[-1])
        outer = _static_size(view_shape[-2])
        last = _static_size(src_shape[-1])
        if inner is None or outer is None or last is None or outer * inner != last:
            continue
        rewritten = _unbind_consumers(view, split_dim=len(view_shape) - 2)
        if rewritten is None:
            continue
        for squeeze, start, end in rewritten:
            sliced = _insert(
                graph,
                squeeze,
                aten.slice.Tensor,
                (src, len(src_shape) - 1, start * inner, end * inner),
                _val(src),
            )
            _replace(graph, squeeze, sliced)
            stats["split_unbind->slice"] += 1


def _permute_order(node: Any) -> list[int] | None:
    """Normalized axis order of a `permute` (negative indices folded). None for non-permutes."""
    if not isinstance(node, Node) or node.op != "call_function":
        return None
    if node.target is not aten.permute.default or len(node.args) != 2:
        return None
    src = node.args[0]
    if not isinstance(src, Node):
        return None
    rank = _rank(src)
    raw = node.args[1]
    if not isinstance(raw, (list, tuple)) or len(raw) != rank:
        return None
    # bool is a subtype of int, so it is rejected explicitly (True would turn into axis 1)
    if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in raw):
        return None
    order = [axis % rank for axis in raw]
    return order if sorted(order) == list(range(rank)) else None


def _compose_permute_chains(graph: Graph, stats: Counter) -> None:
    """Remove identity `permute`s and compose adjacent `permute`s into one.

    The IR's permute is a strided materializing copy, so each link of a chain adds one dispatch and
    one full read and write of every element. For `x.permute(p).permute(q)`,
    `z.shape[i] = y.shape[q[i]] = x.shape[p[q[i]]]`, so the composed axis order is `p[q[i]]`.

    **Other consumers of the intermediate permute do not break** — only the downstream input is
    rewritten, and the intermediate node itself stays (as long as it has consumers). Only
    intermediates nobody looks at any more disappear in the later DCE. The graph's node list is in
    topological order, so chains of 3 or more links also fold from upstream in turn and converge to
    one node in a single pass.
    """
    for permute in list(graph.nodes):
        outer = _permute_order(permute)
        if outer is None:
            continue
        first = permute.args[0]
        if not isinstance(first, Node):
            continue
        if outer == list(range(len(outer))):
            _replace(graph, permute, first)
            stats["permute:identity"] += 1
            continue
        inner = _permute_order(first)
        if inner is None:
            continue
        base = first.args[0]
        if not isinstance(base, Node) or len(inner) != len(outer):
            continue
        composed = [inner[axis] for axis in outer]
        if composed == list(range(len(composed))):
            _replace(graph, permute, base)
            stats["permute_chain:identity"] += 1
            continue
        permute.args = (base, composed)
        stats["permute_chain:composed"] += 1


def _reshape_chain_back(node: Node) -> Node:
    """Walk back through reshape-likes and return the most upstream tensor with the same order."""
    current = node
    while _is_reshape(current) and _sole_user(current) is not None:
        src = current.args[0]
        if not isinstance(src, Node):
            break
        current = src
    return current


def _reshape_chain_forward(node: Node) -> Node:
    """Walk down clone / reshape-likes and return the most downstream node with the same order."""
    current = node
    while True:
        user = _sole_user(current)
        if user is None or not (_is_reshape(user) or user.target is aten.clone.default):
            return current
        current = user


def _lower_reshape_permute(graph: Graph, stats: Counter) -> None:
    """Rewrite a rank≥5 `reshape → permute → reshape` into a sequence of rank-4 adjacent transposes.

    DiT's patchify / unpatchify (9 nodes of rank 8) has this form. Any axis permutation decomposes
    into a product of adjacent transpositions, and an adjacent transposition can be expressed as "a
    rank-4 transpose with what is before and after grouped" — every intermediate value is
    contiguous, so views can regroup freely (permute is materialized). Length-1 axes do not
    contribute to the linear index, so they are dropped first to reduce the number of transposes.

    MUST: a form whose end points (the most upstream / downstream of the chain) do not fit in
    rank ≤ STRIDED_RANK fails loudly.
    """
    for permute in list(graph.nodes):
        if permute.op != "call_function" or permute.target is not aten.permute.default:
            continue
        if max(_rank(permute), _rank(permute.args[0])) <= STRIDED_RANK:
            continue
        base = _reshape_chain_back(permute.args[0])
        sink = _reshape_chain_forward(permute)
        if _rank(base) > STRIDED_RANK or _rank(sink) > STRIDED_RANK:
            raise NotImplementedError(
                f"reshape/permute chain without end points of rank ≤ {STRIDED_RANK}: "
                f"{base.name}(rank{_rank(base)}) → {permute.name} → "
                f"{sink.name}(rank{_rank(sink)})"
            )
        shape = list(_val(permute.args[0]).shape)
        order = [int(d) % len(shape) for d in permute.args[1]]
        out_shape = list(_val(sink).shape)
        numel = extent_key(_numel(shape))
        if numel != extent_key(_numel(list(_val(base).shape))) or numel != extent_key(
            _numel(out_shape)
        ):
            raise NotImplementedError(
                f"reshape/permute chain whose element counts disagree: {permute.name}"
            )

        kept = [index for index, size in enumerate(shape) if _static_size(size) != 1]
        position = {axis: slot for slot, axis in enumerate(kept)}
        sizes = [shape[index] for index in kept]
        target = [position[axis] for axis in order if axis in position]

        ref = _val(base)
        current = _as_contiguous(graph, sink, base, ref)
        state = list(range(len(sizes)))
        for slot in range(len(state)):
            index = state.index(target[slot])
            while index > slot:
                head = _numel([sizes[axis] for axis in state[: index - 1]])
                left, right = sizes[state[index - 1]], sizes[state[index]]
                tail = _numel([sizes[axis] for axis in state[index + 1 :]])
                current = _insert(
                    graph, sink, aten.view.default, (current, [head, left, right, tail]), ref
                )
                current = _insert(graph, sink, aten.permute.default, (current, [0, 2, 1, 3]), ref)
                current = _insert_contiguous(graph, sink, current, ref)
                state[index - 1], state[index] = state[index], state[index - 1]
                index -= 1
                stats["reshape_permute:transpose"] += 1
        current = _insert(graph, sink, aten.view.default, (current, out_shape), ref)
        _replace(graph, sink, current)
        stats["reshape_permute"] += 1


def _collect_dead_code(graph: Graph, stats: Counter) -> None:
    """DCE once in the middle of the passes (**the position matters**).

    A structural difference introduced in the ADR 0016 port. Unless the folded original patterns
    (pow / mean / rsqrt, the guard's eq/any/logical_not) are removed here, the following passes
    (`_pow2_to_mul` / `_promote_scalar_operands`) **rewrite dead subtrees and only inflate the
    statistics**. The IR does not change, so it is the kind of degradation that goes unnoticed
    while everything stays green.
    """
    before = len(graph.nodes)
    graph.eliminate_dead_code()
    stats["dead_code_collected"] += before - len(graph.nodes)


def _pow2_to_mul(graph: Graph, stats: Counter) -> None:
    """`aten.pow(x, 2)` → `mul(x, x)` (the M0 vocabulary has no pow but has the equivalent mul).

    MUST: the exponent is only **the integer 2, not a bool**. `2.0` would pass as `2.0 == 2`, but
    applied to an integer tensor it promotes the type (i64 → f32) and gives a value different from
    `mul(x, x)`.
    MUST: the dtype must be unchanged. A form with promotion is not an identity rewrite, so it is
    left alone and goes to the full list of unsupported ops (no silent approximation).
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.pow.Tensor_Scalar:
            continue
        exponent = node.args[1]
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent != 2:
            continue
        src = node.args[0]
        if not isinstance(src, Node):
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if not isinstance(src_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if src_val.dtype is not out_val.dtype:
            continue
        with graph.inserting_before(node):
            replacement = graph.call_function(aten.mul.Tensor, (src, src))
        # convert requires meta["val"] (a FakeTensor) on every node.
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        stats["pow2->mul"] += 1


def _drop_identity_repeat(graph: Graph, stats: Counter) -> None:
    """`aten.repeat(x, [1, …, 1])` → x.

    MUST: fires only when **every argument is 1 and their count = the input rank** (recon §6-4).
    repeat can add leading axes (`[1,1,1]` on rank 2 gives rank 3), so without checking the count
    the identity rewrite silently shifts the shape. A real repeat with B > 1 is not in the IR
    vocabulary, so leaving it unrewritten here puts it in the full list of unsupported ops (no
    silent approximation).
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.repeat.default:
            continue
        src = node.args[0]
        repeats = node.args[1]
        if not isinstance(src, Node) or not isinstance(repeats, (list, tuple)):
            continue
        # A SymInt repeat count is not compared with an int here (it would trigger a guard).
        # Only plain ints are looked at.
        if not all(isinstance(count, int) and not isinstance(count, bool) for count in repeats):
            continue
        if len(repeats) != src.meta["val"].dim() or any(count != 1 for count in repeats):
            continue
        node.replace_all_uses_with(src)
        graph.erase_node(node)
        stats["drop_identity_repeat"] += 1


def _drop_identity_add(graph: Graph, stats: Counter) -> None:
    """`aten.add(x, 0)` → x (only for **the Python scalar 0**).

    NOTE: in f32, `x + 0.0` returns +0.0 only when x = −0.0 (the only difference). The IR
    vocabulary **does have ops that distinguish signed zeros** — `div` and `sqrt`: for x = −0.0,
    `1 / (x + 0.0)` is +∞ in torch while the rewritten `1 / x` is −∞ (`sqrt(±0) = ±0` differs in
    the same direction). In the 10 observed families the downstream of a removed add never reaches
    a div denominator / sqrt argument, so the possibility of divergence is accepted and this
    rewrite stays (docs/limitations.md). When adding a form where −0.0 can arise on the consumer
    side, review this pass's applicability first.
    MUST: a form with type promotion (`add(i64, 0.0)` → f32) is not rewritten. It is a cast, not an
    identity, so it goes to folding or to the list of unsupported ops.
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in (aten.add.Tensor, aten.add.Scalar):
            continue
        if len(node.args) != 2 or node.kwargs.get("alpha", 1) != 1:
            continue
        src, addend = node.args
        if not isinstance(src, Node) or isinstance(addend, bool):
            continue
        if not isinstance(addend, (int, float)) or addend != 0:
            continue
        if src.meta["val"].dtype is not node.meta["val"].dtype:
            continue
        node.replace_all_uses_with(src)
        graph.erase_node(node)
        stats["drop_identity_add"] += 1


#: Binary ops that can have a scalar operand → (replacement, whether the scalar is the left
#: operand).
#:
#: Only `rsub.Scalar(x, s)` has its meaning reversed from its argument order (`s − x`), so it is
#: sent to the left explicitly. For the rest the order is decided by "the non-Node side is the
#: scalar" — because observed graphs also contain forms like `sub.Tensor(1, mask)` where **a Python
#: scalar enters a .Tensor overload on the left** (recon's mask path).
_SCALAR_BINARY: dict[Any, tuple[Any, bool]] = {
    aten.add.Tensor: (aten.add.Tensor, False),
    aten.add.Scalar: (aten.add.Tensor, False),
    aten.sub.Tensor: (aten.sub.Tensor, False),
    aten.sub.Scalar: (aten.sub.Tensor, False),
    aten.mul.Tensor: (aten.mul.Tensor, False),
    aten.mul.Scalar: (aten.mul.Tensor, False),
    aten.div.Tensor: (aten.div.Tensor, False),
    aten.div.Scalar: (aten.div.Tensor, False),
    aten.rsub.Scalar: (aten.sub.Tensor, True),
}


def _promote_scalar_operands(graph: Graph, stats: Counter) -> None:
    """`add/sub/mul/div(tensor, scalar)` → binary op + **rank-1 constant** (absorbed by broadcast).

    The constant is inserted as `aten.full.default([1], value, dtype=…)`. That op is on convert's
    folding allowlist, so if the consumer is a folding target the constant is folded along with it,
    and otherwise it becomes an initializer (an i32 initializer for i32 — ADR 0010).

    MUST: the constant's dtype matches the tensor side, and **forms with type promotion are not
    rewritten** (`div(i64, 128)` returns f32 = it contains a cast). Dropping to a rank-1 constant
    without matching would silently give a different value without meeting the IR binary ops'
    requirement "input dtypes are the same".
    MUST: `alpha != 1` is left alone (the IR vocabulary has no scaled add — convert fails it).
    MUST: non-finite scalars are left alone (IR v1 rejects non-finite values at the value level).
    """
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        entry = _SCALAR_BINARY.get(node.target)
        if entry is None or len(node.args) != 2:
            continue
        target, scalar_is_left = entry
        if node.kwargs.get("alpha", 1) != 1 or set(node.kwargs) - {"alpha"}:
            continue
        left, right = node.args
        # MUST: put the constant "on the side where the original scalar was". sub / div do not
        # commute, and observed graphs have forms where the scalar comes on the **left**, like
        # `sub.Tensor(1, mask)` (recon's mask path) — fixing the order flips the sign silently.
        if scalar_is_left:
            src, scalar, const_first = left, right, True
        elif isinstance(left, Node) and not isinstance(right, Node):
            src, scalar, const_first = left, right, False
        elif isinstance(right, Node) and not isinstance(left, Node):
            src, scalar, const_first = right, left, True
        else:
            continue
        if not isinstance(src, Node) or isinstance(scalar, bool):
            continue
        if not isinstance(scalar, (int, float)) or not math.isfinite(scalar):
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if not isinstance(src_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if src_val.dtype is not out_val.dtype:
            continue
        # MUST: rank 0 (a scalar tensor) is not promoted. Broadcasting with a rank-1 constant is
        # `[] × [1] → [1]` and **raises the rank** — the meta stays the original `[]`, so a graph
        # whose declaration and contents disagree is silently produced. For rank ≥ 1, `[1]` is
        # always absorbed.
        if src_val.dim() < 1:
            continue
        with graph.inserting_before(node):
            const = graph.call_function(aten.full.default, ([1], scalar), {"dtype": src_val.dtype})
        const.meta["val"] = src_val.new_full((1,), scalar)
        args = (const, src) if const_first else (src, const)
        with graph.inserting_before(node):
            replacement = graph.call_function(target, args)
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        stats["promote_scalar_operand"] += 1


def _eq_zero_to_not_bool(graph: Graph, stats: Counter) -> None:
    """`aten.eq(x, 0)` → `bitwise_not(_to_copy(x, bool))` (ADR 0015).

    Lowers the form `scores.masked_fill(attn_mask == 0, -1e4)` (the enc_p / flow rows of recon §2)
    into the existing vocabulary without adding a new comparison op. The equivalence follows from
    the cast convention "x → bool is x != 0": `not (x != 0)` is exactly `x == 0` (it matches for
    NaN and for signed zeros too — `NaN != 0` is true, so negated it is false; `-0.0 != 0` is
    false, so negated it is true).

    MUST: fires only when the right-hand side is **0** (a number that is not a bool). The rewrite
    does not hold for other values, so they are left alone and go to the full list of unsupported
    ops (no silent approximation).
    MUST: the output must be bool. torch's eq always returns bool, but letting through a form whose
    meta carries another dtype (a rewrite with type promotion) makes declaration and contents
    disagree.
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in (aten.eq.Scalar, aten.eq.Tensor):
            continue
        if len(node.args) != 2 or node.kwargs:
            continue
        src, other = node.args
        if not isinstance(src, Node) or isinstance(other, bool):
            continue
        if not isinstance(other, (int, float)) or other != 0:
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if not isinstance(src_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if out_val.dtype is not torch.bool:
            continue
        with graph.inserting_before(node):
            truthy = graph.call_function(aten._to_copy.default, (src,), {"dtype": torch.bool})
            replacement = graph.call_function(aten.bitwise_not.default, (truthy,))
        # convert requires meta["val"] (a FakeTensor) on every node. The cast's output is
        # "x != 0", so it is a bool of the same shape as the original tensor.
        truthy.meta.update(node.meta)
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        stats["eq_zero->bitwise_not_cast"] += 1


def _select_to_squeeze(graph: Graph, stats: Counter) -> None:
    """`select.int` (static axis) → `squeeze` / `slice + squeeze` (general hygiene — ADR 0016).

    An axis of length 1 with index 0 is a single `squeeze` as before (no slice in between — keeps
    existing models' graphs byte-identical). Other static axes open into `slice(width 1) + squeeze`
    (first seen in Gemma4's per-layer PLE slice `per_layer_inputs[:, :, i, :]`. The slice is taken
    by `_h_slice`'s static axis and static range, the squeeze by the existing reshape normalization
    as is).

    MUST: **an axis of symbolic dimension fails loudly**. Taking the length from the export-time
    hint turns "a symbolic dimension whose hint is 1" into a squeeze and a silent wrong value at
    run time (a binding other than the hint) — it is a form that goes wrong quietly, and there is no
    detector other than stopping it here.
    MUST: **a symbolic index likewise fails loudly**. The index of
    `aten.select.int(self, int dim, SymInt index)` is SymInt in the schema, and taking the hint
    bakes `slice(hint, hint+1)` — the same silent wrong value as on the axis-length side comes in
    from the index side.
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.select.int:
            continue
        src = node.args[0]
        if not isinstance(src, Node):
            continue
        src_val = _val(src)
        rank = src_val.dim()
        raw_dim = int(node.args[1])
        if not -rank <= raw_dim < rank:
            continue
        dim = raw_dim % rank
        index = _static_index(node.args[2])
        if index is None:
            raise NotImplementedError(
                f"select with a symbolic index is not supported: {node.name} dim={raw_dim}"
                f" index={node.args[2]} (taking the index from the hint gives a silent wrong value"
                " at run time)"
            )
        size = _static_size(src_val.shape[dim])
        if size is None:
            raise NotImplementedError(
                f"select on an axis of symbolic dimension is not supported: {node.name}"
                f" dim={raw_dim} index={index}"
                " (taking the length from the hint gives a silent wrong value at run time)"
            )
        if not -size <= index < size:
            continue
        if index == 0 and size == 1:
            with graph.inserting_before(node):
                replacement = graph.call_function(aten.squeeze.dims, (src, [dim]))
            replacement.meta.update(node.meta)
            _replace(graph, node, replacement)
            stats["select->squeeze"] += 1
            continue
        # A negative index is normalized following torch's select semantics (-size ≤ index < size).
        start = index % size
        sliced = _insert(graph, node, aten.slice.Tensor, (src, dim, start, start + 1), src_val)
        with graph.inserting_before(node):
            replacement = graph.call_function(aten.squeeze.dims, (sliced, [dim]))
        replacement.meta.update(node.meta)
        _replace(graph, node, replacement)
        stats["select->slice+squeeze"] += 1


def _split_to_slices(graph: Graph, stats: Counter) -> None:
    """`split_with_sizes(x, sizes, dim)` + `getitem` → a sequence of `slice`s (ADR 0014).

    getitem wiring of multi-output aten is deferred until sampling needs it (ADR 0068 addendum 2),
    so a split opens into one slice per extraction point. Observed: ConvFlow / ResidualCoupling's
    `torch.split(x, [half]*2, 1)` (recon §2), whose consumers are always `getitem` with a constant
    index.

    MUST: fires only on forms where **the consumers are getitem only**. Passing the tuple itself
    elsewhere makes the rewrite non-equivalent, so it is left alone and goes to the full list of
    unsupported ops (no silent approximation).
    MUST: the split axis is static (for a split of symbolic length, the correspondence between the
    sum of sizes and the axis length cannot be decided from the declaration alone).
    MUST: check that the sizes sum to exactly the axis length. torch does not allow a remainder,
    but opening into slices without this check would silently let through forms where "the last
    interval falls short / runs over".

    NOTE: the 3 branches "a consumer other than getitem", "symbolic split axis" and "sizes do not
    sum up" are unreachable from torch.export output (FX always opens with getitem, a split with
    static sizes specializes the axis, and torch itself rejects a remainder). They stay as a
    defense, but no test can be written (there is no form to write one with).
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.split_with_sizes.default:
            continue
        src = node.args[0]
        sizes = node.args[1]
        if not isinstance(src, Node) or not isinstance(sizes, (list, tuple)):
            continue
        if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
            continue
        src_val = src.meta.get("val")
        if not isinstance(src_val, torch.Tensor):
            continue
        rank = src_val.dim()
        raw_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if isinstance(raw_dim, bool) or not isinstance(raw_dim, int) or not -rank <= raw_dim < rank:
            continue
        dim = raw_dim % rank
        extent = src_val.shape[dim]
        if isinstance(extent, torch.SymInt) and extent.node.expr.free_symbols:
            continue
        if sum(sizes) != int(extent):
            continue
        users = list(node.users)
        if not users or any(user.target is not operator.getitem for user in users):
            continue
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        replaced = 0
        for user in users:
            index = user.args[1]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(sizes):
                continue
            with graph.inserting_before(user):
                sliced = graph.call_function(
                    aten.slice.Tensor, (src, dim, offsets[index], offsets[index + 1])
                )
            # convert requires meta["val"] (a FakeTensor) on every node. getitem's meta is
            # exactly the one extracted piece.
            sliced.meta.update(user.meta)
            user.replace_all_uses_with(sliced)
            graph.erase_node(user)
            replaced += 1
        if replaced != len(users):
            # A getitem with an unreadable index remains = the split node cannot be removed (a
            # partial rewrite would leave the remaining getitems as broken references, so the
            # slices made so far are left to DCE as well).
            continue
        graph.erase_node(node)
        stats["split_with_sizes->slice"] += replaced


def _passes(placeholders: _Placeholders) -> tuple[Callable[[Graph, Counter], None], ...]:
    """List the passes to run, in order.

    MUST: `_fold_rms_norm` → `_pow2_to_mul` / `_promote_scalar_operands`, and
    `_drop_identity_add` → `_promote_scalar_operands` (see the module docstring).
    `_eq_zero_to_not_bool` comes late — running it earlier gives the same result, but this avoids
    creating an order dependence if a form where scalar promotion grabs `eq(x, 0)` as a binary op
    comes in later.
    """
    return (
        _drop_metadata_asserts,
        # Folding / removal of high-level patterns (run before other passes break the original
        # patterns).
        # MUST: pow(-0.5)→rsqrt comes before _fold_rms_norm (the matcher sees only the rsqrt form).
        _pow_neg_half_to_rsqrt,
        _fold_rms_norm,
        partial(_drop_safe_softmax_guard, placeholders=placeholders),
        _additive_attn_mask,
        # Rank lowering (fires only on rank > STRIDED_RANK)
        _lower_unit_expand,
        _lower_split_unbind,
        _lower_reshape_permute,
        # Does not cross the contiguous copies rank lowering inserts, so one sweep in topological
        # order folds the chains completely
        _compose_permute_chains,
        # MUST: DCE once here (the position matters — _collect_dead_code's docstring)
        _collect_dead_code,
        _pow2_to_mul,
        _drop_identity_repeat,
        _drop_identity_add,
        _promote_scalar_operands,
        _eq_zero_to_not_bool,
        _select_to_squeeze,
        # Order-free (split's inputs and outputs are forms no other pass touches). It is last so
        # that it is clear no other pass revisits the slices it creates.
        _split_to_slices,
    )


def normalize_graph(ep: ExportedProgram) -> dict[str, int]:
    """Normalize the FX graph in place and return the firing count per pass.

    MUST: what it takes is an **ExportedProgram**, not a GraphModule. The safe-softmax guard's
    inactivity proof needs the **actual values** of lifted constants / buffers
    ({@link _Placeholders}).
    """
    gm = ep.graph_module
    graph = gm.graph
    stats: Counter[str] = Counter()
    placeholders = _Placeholders(ep)
    for run in _passes(placeholders):
        run(graph, stats)
    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return dict(stats)
