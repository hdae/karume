"""Op contract table — keeps "arity / dtype rules / attrs schema" per op in one place on the
Python side.

The TS-side source of truth is packages/runtime/src/ops.ts (bodies under
packages/runtime/src/ops/), and this table is its equivalent. The purpose is to match the
exporter's emit set ⊆ the runtime's executable set including dtypes (matching op names alone
misses dtype differences — ADR 0005). When the two diverge, "export is green, only the browser
fails" follows, so adding an op moves the TS contract table, this table and the goldens as one set.

NOTE: deriving output shapes (broadcast / reduction) is not in this table — no consumer on the
Python side computes shapes after binding resolution (the runtime's is in the TS
packages/runtime/src/ops.ts).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from karume.dims import MAX_SAFE_INT, is_symbol_name
from karume.ir import IrNode

#: Unary elementwise. The 6 ops with attrs (clamp / clamp_min / leaky_relu / the 3 comparisons)
#: and the 3 comparisons whose output dtype differs from the input also belong here — the kind
#: means only "elementwise with one input"; the attrs schema and the output dtype live in other
#: fields of the contract table (equivalent to packages/runtime/src/ops.ts).
UNARY_OPS = (
    "neg",
    "abs",
    "exp",
    "log",
    "log1p",
    "sqrt",
    # The only trigonometric op is sin (layer 1 of ADR 0043 — now the Core ATen layer, ADR 0059).
    # FOLDABLE_OPS folds constant RoPE tables, so the vocabulary needs only the form that takes
    # run-time values — DACVAE's Snake activation was the first. `cos` is not added until it shows
    # up in measurements ("no additions for symmetry").
    "sin",
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    # torch's approximate="tanh" form. Its values differ from the erf-form `gelu`, so it is a
    # separate op (a contract with empty attrs has no field to carry the approximation kind).
    "gelu_tanh",
    "bitwise_not",
    "clamp",
    # One-sided clamp (ADR 0017). Making the existing clamp's attrs optional to cover it is
    # rejected because it would break "declared attrs are never filled with defaults" (ADR 0012).
    "clamp_min",
    "leaky_relu",
    "ge_scalar",
    "le_scalar",
    "gt_scalar",
)
#: Binary elementwise (torch-style right-aligned broadcast). `ge` is f32 × f32 → **bool**.
BINARY_OPS = ("add", "sub", "mul", "div", "ge", "bitwise_and")
#: Reduce over one axis (attrs `dim`, no keepdim). The last dimension runs on the row kernel,
#: other axes on the axis variant (docs/research/2026-08-04-vae-axis-reduce-recon.md §5).
REDUCE_OPS = ("sum", "amax", "amin")
#: argmax over the last dimension (ADR 0068 decision 2). Input f32 → output **i32** (indices),
#: **rank preserved** (the fixed form that collapses the last dimension to 1).
#:
#: MUST: no attrs. The axis is fixed to the last dimension (the missing `dim` field itself declares
#: "other axes are not in the vocabulary" — the same narrowing as gather), and there is no
#: `keepdim` field either (only the rank-preserving form).
#: MUST: not in the reduce family (REDUCE_OPS). The attrs, output dtype and rank handling all
#: differ, and the kernel is a separate family (identity is -inf, carries (value, index) pairs).
#: MUST: ties break to the **smallest index**, NaN is the **maximum**, an all -inf row gives
#: **index 0** (all observed to match torch, 2026-08-17). Forms outside this (keepdim=False /
#: a middle dimension / all dimensions) all fail loudly in `aten_handlers._h_argmax`.
ARGMAX_OP = "argmax"
#: top-k over the last dimension (ADR 0068 decision 3 — **the first multi-output op**). Input f32 →
#: **2 outputs** (slot 0 = values f32 in **descending order**, slot 1 = indices i32), both
#: `[..., k]`.
#:
#: MUST: `k` is **required in attrs** (a plan-time constant = static-k). torch's schema is
#: `topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True)` with a SymInt k,
#: but a k decided at run time does not fit the static-shape premise (ADR 0004), so it is not
#: accepted. The accepted domain is `1 <= k <= last dimension` (k=0 is accepted by torch but is
#: not in the vocabulary).
#: MUST: the axis is fixed to the last dimension (no `dim` field), and there are no `largest` /
#: `sorted` fields either (only the descending-sorted largest side is in the vocabulary).
#: MUST: ties break to the **smallest index** (same family as argmax), NaN is the **maximum**.
#: **The value sequence matches torch bit for bit**, while **the index sequence is where karume
#: specifies what torch leaves unspecified** (torch does not guarantee the order of equal
#: elements, and `argmax` and `topk(k=1)` disagree within the same repository — the observation is
#: in the NOTE of packages/runtime/src/kernels/topk.ts).
#: NOTE: there is **no aten handler yet** (wiring the getitem of multi-output aten ops is deferred
#: until sampling needs it — ADR 0068 addendum 2). Not being exportable from torch is expected,
#: and NON_EMITTABLE_OPS is where it is listed.
TOPK_OP = "topk"
#: Ternary elementwise `out = cond ? a : b` (torch's where). All three broadcast right-aligned, and
#: although the condition is the first slot, **the output is the value side** (mapping bool → f32
#: — _OUTPUT_DTYPES).
WHERE_OP = "where"
#: Prefix sum over the last dimension (attrs `dim`, last dimension only — the same narrowing as
#: softmax).
CUMSUM_OP = "cumsum"
MATMUL_OP = "matmul"
#: Batched matmul (ADR 0012). `[B,M,K] × [B,K,N] → [B,M,N]`, **rank-3 only**; rank-2 belongs to
#: matmul (not shared — the contract's shape rule would become rank-polymorphic).
BMM_OP = "bmm"
#: gather over the last dimension (ADR 0012). `out[..., j] = src[..., index[..., j]]`. The axis is
#: fixed to the last dimension and there are no attrs (only dim = -1 was observed). The only op
#: whose dtype differs per input slot.
GATHER_OP = "gather"
#: Semantic dtype conversion (attrs `to` is the target — ADR 0009).
CAST_OP = "cast"

#: Layout ops (ADR 0011). reshape does not change element order (the normalization target of
#: view / squeeze / unsqueeze) and executes as a buffer alias; permute / expand are one kernel
#: family of strided materializing copies.
RESHAPE_OP = "reshape"
PERMUTE_OP = "permute"
EXPAND_OP = "expand"

#: Layout ops, second group (ADR 0014). Only forms that satisfy the full-write invariant (every
#: kernel writes every byte of its output) enter the vocabulary.
#:
#: - `slice` — a static range on a static axis (symbolic axes belong to sym_prefix_slice — no
#:   overlap). Executes by reusing the strided **read** copy family; the only variable is one
#:   offset word in params.
#: - `cat` — concatenation on a static axis. **Variable input count** (OpContract.variadic),
#:   executed by the strided **write** copy family. All inputs together cover the whole output, so
#:   full-write holds per node.
#: - `pad` — constant 0 padding of the last dimension. One dedicated kernel writes the whole output.
#: - `flip` — index reversal on a static axis. A dedicated tiny kernel.
SLICE_OP = "slice"
CAT_OP = "cat"
PAD_OP = "pad"
FLIP_OP = "flip"

#: Symbolic prefix slice (ADR 0010). Cuts the leading `coeff·sym+offset` (after binding) out of a
#: constant baked at Tmax. The input is a static shape without symbols (= the Tmax form), executed
#: by reusing the strided materializing copy (0 new kernels).
SYM_PREFIX_SLICE_OP = "sym_prefix_slice"

#: Fused ops (ADR 0012 / ADR 0015 / ADR 0017). Every op on the "no decomposition" list that
#: started with ADR 0007 (**12 ops today** — the ledger's source of truth is
#: docs/op-vocabulary.md) has a kernel (conv2d showed up in measurements in Anima's VAE decoder —
#: ADR 0017).
#: Fused ops with bias / affine have **fixed arity**, and a conv without bias is normalized to
#: arity 3 by the exporter's zero-bias synthesis (no arity branches in the vocabulary — ADR 0015).
LINEAR_OP = "linear"
LAYER_NORM_OP = "layer_norm"
#: RMSNorm (ADR 0017). **Arity 2** (x, weight — no bias) with attrs eps only. The source of truth
#: for the normalized length is the weight's length (no normalized_shape field — no duplicated
#: state).
RMS_NORM_OP = "rms_norm"
STATIC_QUANTIZE_OP = "static_quantize"
SOFTMAX_OP = "softmax"
#: `softmax` + "**a row whose max is -inf writes all 0**" (ADR 0044). Contract, attrs and shape
#: rule are identical to softmax; this one line is the only difference. `_drop_safe_softmax_guard`
#: emits it for forms where the safe-softmax guard of the SDPA decomposition cannot be dropped by
#: proof from actual values (run-time masks and the like).
SAFE_SOFTMAX_OP = "safe_softmax"
#: Fused attention (ADR 0023). `out = softmax_lastdim((q·scale) @ (k·scale)ᵀ + mask) @ v`.
#: **Arity 3 or 4** (q / k / v + an optional mask); inputs are rank-4 head-first
#: (`[B,H,M,D]` / `[B,Hkv,N,D]` ×2) with **the same D for all three**; the output is `[B,H,M,D]`.
#: H and Hkv use a **divisible broadcast** (`H % Hkv == 0` and `H ≥ Hkv ≥ 1` — GQA / MQA,
#: ADR 0067 decision 1); the ratio `r = H / Hkv` is derived, so there is no attrs field for it.
#: The mask is **f32, rank-4, shape exactly `[1,1,M,N]`** (additive) and broadcasts over B·H.
#: `[B,1,M,N]` / `[1,H,M,N]` / bool / rank≠4 and causal / dropout are not in the vocabulary, and
#: `aten_handlers._h_attention` enumerates all such SDPA forms and fails loudly.
#:
#: **States form** (ADR 0067 decision 4 — a contract extension under the same op name; **the
#: presence of the field decides the form**): when the `states` field is `{ k, v }` it is the
#: autoregressive form that reads "the past from the slots, this step's k/v from `ins`". ins is
#: `[B,Hkv,M,D]` (M = the physical chunk dimension shared with q), and **there is no mask input**
#: (causal fixed, expressed by predicate computation). Sliding is the optional attrs `window`
#: (field absent = the whole context). A node **without** the field keeps the old contract as is,
#: and existing assets and existing checks do not move by a single byte.
#:
#: MUST: `scale` **applies to both q and k (the half-scale contract)** — the same as
#: `√scale_factor` in torch's `_scaled_dot_product_attention_math`. The exporter computes
#: `f32(math.sqrt(scale_factor))` from SDPA's `scale` argument (`1/√D` when omitted) and stores it.
ATTENTION_OP = "attention"
#: A **single-purpose effect op** that writes this step's k / v into a state slot (ADR 0067
#: decision 5). `ins` is one `[B,Hkv,M,D]`, the `states` field is exactly `{ slot }`, and there are
#: **0 outputs** (it defines no value — the first op with zero outputs under the node-level
#: multi-output generalization of ADR 0068 decision 1).
#:
#: MUST: not built into attention (the TVM style is not adopted). Keeping it a single-purpose op is
#: what makes ① the check of full-write / padding-row no-op (cut by queryLength — ADR 0066
#: decision 4) fit in one op, ② the bit-identity check of the read-only attention side simple, and
#: ③ a KV-sharing layer expressible as just "no `state_append` node" (the expressiveness of ORT's
#: kv_empty obtained by the op's absence).
#: MUST: `window` is **optional** (field absent = the whole context). The declaration layer
#: (verify._assert_state_order) checks that the value agrees across all nodes touching the same
#: slot — the logical col → physical row mapping MUST use the same formula for reads and writes
#: (ADR 0067 decision 4), and a different formula on the read side alone is a silent misread.
#: MUST: no logical positions (pastLength / queryLength) in attrs. Run-time scalars that change
#: every step are carried by a context-owned mutable uniform (ADR 0066 addendum of 2026-08-17,
#: item 4).
#: NOTE: there is **no** aten handler (not exportable from torch — the decode graph script emits
#: it). NON_EMITTABLE_OPS is where it is listed.
STATE_APPEND_OP = "state_append"
EMBEDDING_OP = "embedding"
MASKED_FILL_OP = "masked_fill"
CONV1D_OP = "conv1d"
#: 2D convolution (ADR 0017). The weight is **[Cout, Cin/groups, Kh, Kw]**, and the attrs
#: stride / padding / dilation have **2 components, H/W** (groups is a scalar). All four are
#: required.
CONV2D_OP = "conv2d"
#: 3D convolution (extended composite layer — ADR 0118 decision 1). **Unbatched**: x is the rank 4
#: [Cin, T, H, W], the weight is **[Cout, Cin/groups, Kt, Kh, Kw]**, the output is
#: [Cout, Tout, Hout, Wout]. The attrs stride / padding / dilation have **3 components, [T, H, W]**
#: (groups is a scalar), and all four are required.
#: Batched rank 5 is not in the contract (the surrounding cat / slice hit the strided family's
#: rank limit of 4).
#: The semantics cover general groups, but the runtime's GPU-implemented subset is groups == 1 only
#: (fails loudly at plan time — packages/runtime/src/runtime/plan.ts).
CONV3D_OP = "conv3d"
#: Transposed convolution (ADR 0015). The weight is **[Cin, Cout, K]** (transposed relative to
#: conv1d's [Cout, Cin/groups, K]). attrs are stride / padding only, and only forms whose output
#: length is L*stride are accepted.
CONV_TRANSPOSE1D_OP = "conv_transpose1d"

#: Modulated deformable convolution v2 (an atomic op of layer 1' — now the extended atomic layer,
#: ADR 0059 — `torchvision::deform_conv2d`, ADR 0055). **Fixed arity 5**: takes x[B,Cin,H,W] /
#: weight[Cout,Cin,Kh,Kw] / offset[B,2*Kh*Kw,Hout,Wout] / mask[B,Kh*Kw,Hout,Wout] / bias[Cout].
#: The offset channels are nested (kh, kw) with pairs innermost (**even = y / odd = x**), and the
#: mask (modulator) applies **after bilinear interpolation**. Out of bounds is zero-filled (not
#: clamped).
#:
#: MUST: the only attr is `padding`. No stride / dilation / groups / offset_groups fields (= fixed
#: to 1) — measurements (20 sites in the BiRefNet family) are all 1, and the missing field
#: expresses "that form is not in the vocabulary" structurally. Forms outside this all fail loudly
#: in `aten_handlers._h_deform_conv2d`.
#: MUST: mask is a required slot = **DCNv2 only** (DCNv1 with use_mask=False is not in the
#: vocabulary).
DEFORM_CONV2D_OP = "deform_conv2d"

#: Bilinear resample over the 2 spatial axes of NCHW (an atomic op of layer 1 — now the Core ATen
#: layer — `aten.upsample_bilinear2d.vec`). `x[B,C,H,W] -> [B,C,Hout,Wout]` with attrs
#: `output_size` (`[Hout, Wout]`) only.
#:
#: MUST: **align_corners=True only**. No `align_corners` / `mode` / `scale_factor` fields in
#: attrs (the missing field expresses "not in the vocabulary" structurally — ADR 0023 decision 4).
#: Forms outside this all fail loudly in `aten_handlers._h_upsample_bilinear2d`.
#: NOTE: downscaling (Hout < H) goes through the same op with the same formula (torch uses the
#: same op too).
UPSAMPLE_BILINEAR2D_OP = "upsample_bilinear2d"

#: GRU **hidden-side scan** (layer 2 — now the extended composite layer — ADR 0056). Folds only
#: the sequential time direction into one node and **has no input-side GEMM** (the caller prepares
#: gi = x·W_ih^T + b_ih with the existing `linear`). **Fixed arity 4**: takes gi[T,N,3H] / h0[N,H]
#: / w_hh[3H,H] / b_hh[3H] and outputs y[T,N,H]. The gate order in 3H is **r / z / n**.
#:
#: MUST: the scan direction is **split by op name** (not a bool variant in attrs — Python's bool
#: derives from int, and every existing attr check function explicitly rejects bool). The same
#: "an attr variant is a separate op" approach as `gelu` / `gelu_tanh` (ADR 0056 decision 2).
#: MUST: no fields for multiple layers / bidirectional / has_biases=False / batch_first / dropout.
#: The only variable-arity op of IR v1 is `cat`, and `aten.gru`'s Tensor[16] does not fit, so
#: layers and directions are expressed **by lining up nodes**. Forms outside this all fail loudly
#: in `aten_handlers._h_gru_scan`.
GRU_SCAN_OP = "gru_scan"
GRU_SCAN_REVERSE_OP = "gru_scan_reverse"
GRU_SCAN_OPS = (GRU_SCAN_OP, GRU_SCAN_REVERSE_OP)

#: Weight slots **eligible** for low-precision storage (op name → input slot number — ADR 0018).
#: The mirror of `WEIGHT_SLOTS` on the TS side (`packages/runtime/src/ops/names.ts`, re-exported
#: by `packages/runtime/src/ops.ts`); a drift is caught by the conformance table
#: (`weight_slot` in packages/runtime/tests/fixtures/op-contracts.json), matched from both sides.
#:
#: MUST: do not include bias. In the prototype the bias's f32 constant dragged the weight down with
#: it and left 0MB eligible for f16 (the root cause ADR 0006 named) — "bias is always f32" can only
#: be guaranteed by "the bias slot is not considered for eligibility".
WEIGHT_SLOTS: Mapping[str, int] = MappingProxyType(
    {
        LINEAR_OP: 1,
        CONV1D_OP: 1,
        CONV2D_OP: 1,
        CONV3D_OP: 1,
        CONV_TRANSPOSE1D_OP: 1,
        EMBEDDING_OP: 0,
    }
)

#: **Channel axis** of the per-channel scale (op name → axis number of the weight tensor — ADR
#: 0019). The mirror of `WEIGHT_CHANNEL_AXES` on the TS side (`packages/runtime/src/ops/names.ts`,
#: re-exported by `packages/runtime/src/ops.ts`); a drift is caught by the conformance table
#: (`channel_axis` in packages/runtime/tests/fixtures/op-contracts.json), matched from both sides.
#:
#: The output-channel axis. linear `[out,in]` / conv1d `[Cout,Cin/g,K]` / conv2d
#: `[Cout,Cin/g,Kh,Kw]` / conv3d `[Cout,Cin/g,Kt,Kh,Kw]` / embedding `[V,H]` use 0, and **only
#: conv_transpose1d uses 1, with its transposed `[Cin,Cout,K]` layout**.
#:
#: MUST: the key set matches WEIGHT_SLOTS (keeps a new weight slot from being silently quantized
#: as axis 0). test_quantize.py pins that the axes also match `quantize.QUANT_CHANNEL_AXES` (the
#: same table looked up by module type).
WEIGHT_CHANNEL_AXES: Mapping[str, int] = MappingProxyType(
    {
        LINEAR_OP: 0,
        CONV1D_OP: 0,
        CONV2D_OP: 0,
        CONV3D_OP: 0,
        CONV_TRANSPOSE1D_OP: 1,
        EMBEDDING_OP: 0,
    }
)

#: The largest rank the strided copy kernels handle (ADR 0011). The exporter checks it too, so that
#: an excess fails at export time (no "export is green, only the browser fails").
STRIDED_RANK = 4

#: IR v1 semantic dtype vocabulary (docs/ir-v2.md).
SEMANTIC_DTYPES = frozenset({"f32", "i32", "bool"})
#: f32 only (no i32 / bool forms appeared in measured graphs — not enabled for symmetry).
F32_DTYPES = frozenset({"f32"})
#: The mask outer product mul(i64,i64) and the sub of `1 - attention_mask`
#: (docs/research/2026-08-02-deberta-front-recon.md §3-8). They depend on input values and cannot
#: be constant-folded, so the execution side needs them.
F32_I32_DTYPES = frozenset({"f32", "i32"})
BOOL_DTYPES = frozenset({"bool"})
#: Row sum of bool (the count of trues — the output is i32). sdp's searchsorted
#: (docs/research/2026-08-02-sbv2-chain-recon.md §2).
F32_BOOL_DTYPES = frozenset({"f32", "bool"})
#: Integer indices only (gather's index slot — observed as i32[16,T,T]).
I32_DTYPES = frozenset({"i32"})
#: Semantic dtypes that can be transferred as graph inputs (ADR 0009 — every element is 4 bytes).
IO_DTYPES = SEMANTIC_DTYPES
#: Storage dtypes the runtime can execute.
#: `i32` is executed as the target that symbol-dependent constants are baked into (raw int32 —
#: ADR 0010).
#: `f16` is ADR 0018 / `bf16` is ADR 0131 / `i8` is ADR 0019 / `i4` is ADR 0069 / `i2` is
#: ADR 0097 — eligible weight
#: slots stay compressed and GPU-resident, and ineligible ones are decoded to f32 on the CPU at
#: load, so either path executes (the mirror of RuntimeSupport.storage on the TS side). i4 and i2
#: have narrow eligibility — i4 only in the weight slots of linear / embedding / conv1d
#: (`groups == 1`) (`emit.I4_WEIGHT_OPS`), i2 only in the weight slots of linear / embedding, as a
#: fixed packed form (ADR 0097).
M0_STORAGE_DTYPES = frozenset({"f32", "f16", "bf16", "i8", "i4", "i2", "i32"})

OpKind = Literal[
    "unary",
    "binary",
    "where",
    "cumsum",
    "matmul",
    "bmm",
    "gather",
    "row_reduce",
    "argmax",
    "topk",
    "cast",
    "reshape",
    "permute",
    "expand",
    "slice",
    "cat",
    "pad",
    "flip",
    "sym_prefix_slice",
    "linear",
    "layer_norm",
    "rms_norm",
    "static_quantize",
    "softmax",
    "safe_softmax",
    "attention",
    "state_append",
    "embedding",
    "masked_fill",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "deform_conv2d",
    "upsample_bilinear2d",
    "gru_scan",
]

#: attr key → value check function (ADR 0012). Every declared key is required, and an undeclared
#: key fails loudly.
AttrSchema = Mapping[str, Callable[[Any, str], None]]


class OpContractError(Exception):
    """IR that violates an op contract (op not in the contract table, arity violation, attrs
    outside the contract, dtype violation)."""


def _assert_cast_target(value: Any, where: str) -> str:
    """The cast target.

    MUST: f32 → i32 is torch-style truncation (toward 0), x → bool is x != 0. The actual
    representation of bool is u32 0 / 1 (ADR 0009). Rounding rules are written into the contract
    (no silent approximation).
    """
    if not isinstance(value, str) or value not in SEMANTIC_DTYPES:
        raise OpContractError(
            f"{where}: the cast target is not a semantic dtype"
            f" ({' / '.join(sorted(SEMANTIC_DTYPES))}): {value!r}"
        )
    return value


CAST_ATTRS: AttrSchema = {"to": lambda value, where: _assert_cast_target(value, where)}


def integral_float_to_int(value: Any) -> Any:
    """Normalize an integral float such as JSON's `1.0` / `1e0` to int (other values unchanged).

    MUST: every field that reads an integer from JSON passes through this before its type is
    checked. On the TS side JSON.parse returns a single number type, so there is no "integral
    float" distinction (`Number.isSafeInteger(1.0)` is true); if Python rejected floats outright,
    **only the exporter's validation would fail to read what the runtime reads** (the accepted sets
    must not drift in either direction). Non-integral and non-finite floats have `is_integer()`
    False (`inf` / `nan` too), so they fail in the caller's type check as they are; bool is not a
    float, so it passes through and the caller rejects it. Exceeding the safe range is also caught
    by the caller's int check.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _assert_permute_dims(value: Any, where: str) -> list[int]:
    """permute's axis permutation (`dims[d]` = the input dimension that output dimension d takes).

    MUST: negative axis numbers are not accepted (torch's -1 notation is normalized at the exporter
    boundary by convention). Two IRs for the same permutation would split both CSE and matching.
    MUST: duplicates are rejected (a permutation is a bijection). Matching against the input rank
    is done by the runtime's shape computation.
    """
    if not isinstance(value, list) or not value:
        raise OpContractError(f"{where}: permute dims is not a non-empty list: {value!r}")
    value = [integral_float_to_int(dim) for dim in value]
    for dim in value:
        # The upper bound is the safe integer (the same accepted set as `Number.isSafeInteger` on
        # the TS side — letting arbitrary-precision ints through creates "dims the exporter can
        # write but the runtime does not accept").
        if isinstance(dim, bool) or not isinstance(dim, int) or not 0 <= dim <= MAX_SAFE_INT:
            raise OpContractError(
                f"{where}: permute dims has an element that is not a non-negative integer:"
                f" {value!r}"
            )
    if len(set(value)) != len(value):
        raise OpContractError(f"{where}: permute dims {value!r} has duplicates")
    return list(value)


PERMUTE_ATTRS: AttrSchema = {"dims": lambda value, where: _assert_permute_dims(value, where)}


def permute_dims(attrs: Mapping[str, Any], where: str) -> list[int]:
    """The permute node's axis permutation (assumes assert_node_contract has done the checks)."""
    return _assert_permute_dims(attrs.get("dims"), f"{where} attrs.dims")


def _assert_integer_attr(value: Any, where: str, minimum: int) -> int:
    """An integer attr of at least `minimum`.

    MUST: excludes `bool` (Python's bool derives from int, so `True` would pass as 1).
    MUST: also checks the safe-integer upper bound (the TS side has this bound implicitly through
    `Number.isSafeInteger`). Python ints have arbitrary precision, so without the bound one could
    create "attr values the exporter can write but the runtime does not accept" — the accepted set
    must be the same on both sides.
    MUST: JSON's `1.0` / `1e0` are normalized to int (the reason is in
    {@link integral_float_to_int}).
    """
    value = integral_float_to_int(value)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= MAX_SAFE_INT
    ):
        raise OpContractError(f"{where}: is not an integer of at least {minimum}: {value!r}")
    return value


#: slice's cut specification (ADR 0014). Shrinks the `dim` axis to `[start, end)`.
#:
#: MUST: all three are non-negative integers (negative axis notation, negative index notation and
#: torch's huge default end are normalized at the boundary by convention — the same as permute's
#: dims). `start <= end` and `end <= axis length` belong to the shapes layer (a rule that spans keys
#: and the input shape cannot be expressed in an attrs schema — the same split as clamp's min/max).
SLICE_ATTRS: AttrSchema = {
    "dim": lambda value, where: _assert_integer_attr(value, where, 0),
    "start": lambda value, where: _assert_integer_attr(value, where, 0),
    "end": lambda value, where: _assert_integer_attr(value, where, 0),
}

#: The axis of cat / flip (a non-negative axis number; matching against the input rank is in the
#: shapes layer).
AXIS_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: pad's left and right padding widths (**last dimension, constant 0 only**).
#:
#: MUST: negative widths are not accepted (torch's constant_pad_nd can truncate with a negative
#: width) — truncation means slice, and letting it through would allow the same form in two ops
#: and make the output length computation negative.
#: MUST: no fill-value field (only 0 was observed — the same narrowing as conv1d's groups).
PAD_ATTRS: AttrSchema = {
    "left": lambda value, where: _assert_integer_attr(value, where, 0),
    "right": lambda value, where: _assert_integer_attr(value, where, 0),
}


def slice_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int, int]:
    """The slice node's (dim, start, end) (assumes assert_node_contract has done the checks)."""
    return (
        _assert_integer_attr(attrs.get("dim"), f"{where} attrs.dim", 0),
        _assert_integer_attr(attrs.get("start"), f"{where} attrs.start", 0),
        _assert_integer_attr(attrs.get("end"), f"{where} attrs.end", 0),
    )


def axis_dim(attrs: Mapping[str, Any], where: str) -> int:
    """The axis of a cat / flip node (a non-negative axis number)."""
    return _assert_integer_attr(attrs.get("dim"), f"{where} attrs.dim", 0)


def pad_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int]:
    """The pad node's (left, right)."""
    return (
        _assert_integer_attr(attrs.get("left"), f"{where} attrs.left", 0),
        _assert_integer_attr(attrs.get("right"), f"{where} attrs.right", 0),
    )


#: Keys of sym_prefix_slice's cut specification for one axis (shrinks `dim` to the length
#: `coeff·sym+offset`).
PREFIX_SLICE_KEYS = ("dim", "coeff", "offset")


def _assert_prefix_sym(value: Any, where: str) -> str:
    """sym_prefix_slice's `sym` (a symbol name of the dimension language).

    Whether it is bound cannot be known without the graph, so only the spelling is checked here
    (matching against graph.symbols is verify.assert_op_contracts).
    """
    if not isinstance(value, str) or not is_symbol_name(value):
        raise OpContractError(f"{where}: sym_prefix_slice sym is not a symbol name: {value!r}")
    return value


def _assert_prefix_slices(value: Any, where: str) -> list[dict[str, int]]:
    """sym_prefix_slice's `slices`.

    MUST: duplicate axes are rejected (two specifications for the same axis would make one of them
    silently disappear and allow a form that only passes the declared-shape match). Coefficient at
    least 1 and offset at least 0 are the same ranges as the dimension language.
    """
    if not isinstance(value, list) or not value:
        raise OpContractError(
            f"{where}: sym_prefix_slice slices is not a non-empty list: {value!r}"
        )
    slices: list[dict[str, int]] = []
    for index, raw in enumerate(value):
        at = f"{where}[{index}]"
        if not isinstance(raw, dict):
            raise OpContractError(f"{at}: is not an object: {raw!r}")
        unknown = sorted(key for key in raw if key not in PREFIX_SLICE_KEYS)
        if unknown:
            raise OpContractError(f"{at}: unknown keys {unknown}")
        for key in PREFIX_SLICE_KEYS:
            if key not in raw:
                raise OpContractError(f"{at}: key '{key}' is missing")
        slices.append(
            {
                "dim": _assert_integer_attr(raw["dim"], f"{at}.dim", 0),
                "coeff": _assert_integer_attr(raw["coeff"], f"{at}.coeff", 1),
                "offset": _assert_integer_attr(raw["offset"], f"{at}.offset", 0),
            }
        )
    dims = [entry["dim"] for entry in slices]
    if len(set(dims)) != len(dims):
        raise OpContractError(f"{where}: sym_prefix_slice slices has the same dim twice")
    return slices


SYM_PREFIX_SLICE_ATTRS: AttrSchema = {
    "sym": lambda value, where: _assert_prefix_sym(value, where),
    "slices": lambda value, where: _assert_prefix_slices(value, where),
}


def sym_prefix_slice_attrs(
    attrs: Mapping[str, Any], where: str
) -> tuple[str, list[dict[str, int]]]:
    """The sym_prefix_slice node's attrs (assumes assert_node_contract has done the checks)."""
    return (
        _assert_prefix_sym(attrs.get("sym"), f"{where} attrs.sym"),
        _assert_prefix_slices(attrs.get("slices"), f"{where} attrs.slices"),
    )


def _assert_normalized_shape(value: Any, where: str) -> list[int]:
    """layer_norm's normalized axes. Accepts **length 1 (= the last dimension) only**.

    MUST: multi-axis normalization is not accepted "for symmetry" (the vocabulary allowlist of
    ADR 0007 is frozen). All 7 observed are [1024] (docs/research/2026-08-02-deberta-front-recon.md
    §5), and the row kernel assumes the last dimension is contiguous.
    """
    if not isinstance(value, list) or len(value) != 1:
        raise OpContractError(
            f"{where}: layer_norm normalized_shape must be a list of length 1"
            f" (only last-dimension normalization can be executed): {value!r}"
        )
    return [_assert_integer_attr(value[0], f"{where}[0]", 1)]


def _assert_eps(value: Any, where: str, what: str) -> float:
    """eps of a normalization op (layer_norm / rms_norm).

    **Finite positive numbers** only (0 gives 1/sqrt(0) on a row of variance 0, all elements 0).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpContractError(f"{where}: {what} eps is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise OpContractError(f"{where}: {what} eps is not a finite positive number: {value!r}")
    return number


def _assert_finite_attr(value: Any, where: str, what: str) -> float:
    """A scalar attr carried as an f32 word in params. **A finite f32 scalar** (IR v1 rejects
    non-finite values)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpContractError(f"{where}: {what} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise OpContractError(f"{where}: {what} is not a finite number: {value!r}")
    return number


def _assert_fill_value(value: Any, where: str) -> float:
    """masked_fill's fill value.

    The observed value -3.4028234663852886e+38 is exactly the smallest finite f32, and its ulp does
    not move through a JSON round trip or f32 rounding (pinned by tests/test_aten_handlers.py).
    """
    return _assert_finite_attr(value, where, "masked_fill value")


LAYER_NORM_ATTRS: AttrSchema = {
    "normalized_shape": lambda value, where: _assert_normalized_shape(value, where),
    "eps": lambda value, where: _assert_eps(value, where, "layer_norm"),
}


def assert_static_quantize_scale(value: Any, where: str) -> float:
    """SRQ takes its fixed f32 scale without re-rounding it (ADR 0097)."""
    number = _assert_finite_attr(value, where, "static_quantize scale")
    try:
        rounded = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as error:
        raise OpContractError(f"{where}: static_quantize scale must be an f32") from error
    if number < 0 or rounded != number:
        raise OpContractError(
            f"{where}: static_quantize scale must be a non-negative, exact f32 value"
        )
    return number


STATIC_QUANTIZE_ATTRS: AttrSchema = {"scale": assert_static_quantize_scale}


def static_quantize_scale(attrs: Mapping[str, Any], where: str) -> float:
    return assert_static_quantize_scale(attrs.get("scale"), f"{where} attrs.scale")


#: rms_norm's attrs (ADR 0017).
#:
#: MUST: no normalized_shape field. The normalized axis is fixed to the single last dimension, and
#: the length is held by **the weight's length** (layer_norm holds the same fact twice, in attrs and
#: in the weight).
RMS_NORM_ATTRS: AttrSchema = {"eps": lambda value, where: _assert_eps(value, where, "rms_norm")}

#: Negative axis notation is normalized at the exporter boundary by convention (the same as
#: permute's dims). Matching against the last dimension is done by the runtime's shape computation,
#: which knows the rank.
SOFTMAX_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: attention's attrs (ADR 0023).
#:
#: MUST: `scale` is required and never filled with a default (the same rule as the runtime side).
#: Allowing omission would give "SDPA with an explicit scale" and "an IR with a filled-in scale" the
#: same form, and a mix-up would show only in the values.
#: MUST: no mask / causal / dropout fields — the missing field itself expresses "not in the
#: vocabulary" structurally.
ATTENTION_ATTRS: AttrSchema = {
    "scale": lambda value, where: _assert_finite_attr(value, where, "attention scale")
}

#: Sliding window of a state-referencing node (ADR 0067 decisions 4 / 5). An **optional attr**:
#: the absence of the field means "the whole context". When declared, an integer of at least 1.
#:
#: MUST: never filled with a default. "No field = the whole context" and "window = capacity" are
#: different declarations, and filling in would let a mix-up of sliding layers (28 layers of
#: Gemma 4 E2B) and full layers (7 layers) show only in the values.
#: MUST: **presence and value agree** across all nodes touching the same slot (reads and writes
#: MUST use the same formula — checked by verify._assert_state_order; a different formula on the
#: read side alone is a silent misread).
STATE_WINDOW_ATTRS: AttrSchema = {
    "window": lambda value, where: _assert_integer_attr(value, where, 1)
}

#: Declaration of readonly attention (the borrower form that reads only the past — ADR 0096
#: stage 2). An **optional attr**: the absence of the field means "the old states form that also
#: reads this step's k/v".
#:
#: MUST: **only** `True` is accepted. `false` means the same as an absent field, so allowing two
#: spellings would make it possible to write an in-between form such as "a single input with
#: readonly: false" for which it is undecided which rule applies (the same style as `window`'s
#: "the absence of the field is itself the declaration").
STATE_READONLY_ATTRS: AttrSchema = {
    "readonly": lambda value, where: _assert_readonly_attr(value, where)
}


def _assert_readonly_attr(value: Any, where: str) -> bool:
    """`attrs.readonly` is exactly the bool true (the MUST of {@link STATE_READONLY_ATTRS})."""
    if value is not True:
        raise OpContractError(
            f"{where}: readonly must be true ({value!r} — false is the same as an absent field, "
            "and one meaning does not get two spellings)"
        )
    return True


#: Input count of the readonly form of attention (exactly one, q — neither this step's k/v nor a
#: mask).
READONLY_ATTENTION_ARITY = 1


def attention_readonly(attrs: Mapping[str, Any]) -> bool:
    """Whether this is the readonly form (the presence of the field decides the form — the range
    check belongs to assert_node_contract)."""
    return attrs.get("readonly") is True


#: torch's padding_idx is **accepted but has no effect on forward** (a field that keeps the padding
#: row from being updated by gradients; the forward pass is the same as a plain row gather).
#: Dropping it from the contract to ignore it would open a hole in the "unknown attrs fail loudly"
#: rule, so only its range is checked (-1 = the sentinel for unspecified) and it is carried along.
EMBEDDING_ATTRS: AttrSchema = {
    "padding_idx": lambda value, where: _assert_integer_attr(value, where, -1)
}

MASKED_FILL_ATTRS: AttrSchema = {"value": lambda value, where: _assert_fill_value(value, where)}

CLAMP_ATTRS: AttrSchema = {
    "min": lambda value, where: _assert_finite_attr(value, where, "clamp min"),
    "max": lambda value, where: _assert_finite_attr(value, where, "clamp max"),
}

#: clamp_min's attrs (ADR 0017 — the 30 clamp(min=eps) of channel L2 normalization).
#:
#: MUST: no max field. "Filling the missing side with the largest finite f32 and passing it to
#: clamp" replaces semantics without an upper bound by an approximation, and ADR 0017 rejects it by
#: name.
CLAMP_MIN_ATTRS: AttrSchema = {
    "min": lambda value, where: _assert_finite_attr(value, where, "clamp_min min")
}

#: torch's negative_slope. **Required and never filled with a default** (ADR 0015 — dec mixes 0.1
#: and 0.01, and relying on a default would silently get one of them wrong). The handler reads
#: torch's default and puts it into attrs.
LEAKY_RELU_ATTRS: AttrSchema = {
    "negative_slope": lambda value, where: _assert_finite_attr(
        value, where, "leaky_relu negative_slope"
    )
}

#: The comparison operand of ge_scalar / le_scalar / gt_scalar.
SCALAR_COMPARE_ATTRS: AttrSchema = {
    "value": lambda value, where: _assert_finite_attr(value, where, "comparison op value")
}

#: Negative axis notation is normalized at the exporter boundary by convention (the same as
#: softmax's dim).
CUMSUM_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: Reduction axis of the reduce family (sum / amax / amin). **Required** and never filled with a
#: default — allowing "no field = the last dimension" would make an IR meant to reduce the channel
#: axis run silently as a different computation that folds the last dimension (the same reason as
#: conv1d's dilation / groups — ADR 0015).
REDUCE_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: topk's count `k` (ADR 0068 decision 3). **A required positive integer** (= static-k).
#:
#: MUST: k=0 is not accepted (torch accepts it, but "the 0 largest" defines no value). A symbolic k
#: (SymInt / a dimension expression) is also rejected by `_assert_integer_attr` — a k decided at
#: run time does not fit the static-shape premise (ADR 0004), and neither the output shape nor the
#: bytes to allocate would be known at plan time.
#: MUST: the upper bounds are not checked here. `k <= last dimension` needs the input shape, so it
#: is on the shapes.py side; the **implementation limit** (from workgroup storage) needs device
#: limits, so it is on the runtime side (`assertTopkK` in packages/runtime/src/kernels/topk.ts).
TOPK_ATTRS: AttrSchema = {"k": lambda value, where: _assert_integer_attr(value, where, 1)}

#: attrs passed to the elementwise kernel as f32 **at the end of params** (the order is the params
#: layout itself). The exporter keeps this to check a cross-key invariant (clamp's min <= max) in
#: the shape layer (equivalent to SCALAR_PARAM_ATTRS in the TS packages/runtime/src/ops/attrs.ts).
SCALAR_PARAM_ATTRS: dict[str, tuple[str, ...]] = {
    "clamp": ("min", "max"),
    "clamp_min": ("min",),
    "leaky_relu": ("negative_slope",),
    "ge_scalar": ("value",),
    "le_scalar": ("value",),
    "gt_scalar": ("value",),
}

#: conv1d's attrs (dilation / groups added in ADR 0015).
#:
#: MUST: all four are **required** (assert_node_contract requires every key to be present). The
#: property "there is no path that silently runs a value other than 1 as 1", once guaranteed by
#: "no field = fixed to 1", is guaranteed after the fields exist only by **never filling in
#: defaults**.
CONV1D_ATTRS: AttrSchema = {
    "stride": lambda value, where: _assert_integer_attr(value, where, 1),
    "padding": lambda value, where: _assert_integer_attr(value, where, 0),
    "dilation": lambda value, where: _assert_integer_attr(value, where, 1),
    "groups": lambda value, where: _assert_integer_attr(value, where, 1),
}


def _assert_int_pair(value: Any, where: str, minimum: int, what: str) -> tuple[int, int]:
    """A spatial attr of conv2d (2 components, `[H, W]`).

    MUST: accepts **lists of exactly length 2** only. Also allowing the scalar notation (torch's
    stride=1 that applies to both axes) would give the same convolution two IRs and split both CSE
    and the conformance-table matching — normalization is the job of the exporter boundary (the
    handler).
    MUST: H and W are not split into separate keys. 2 axes × 3 attrs would make 6 keys, and "an IR
    that forgot one of them" would no longer look like "a missing required key" but like "a value
    that looks like the default".
    """
    if not isinstance(value, list) or len(value) != 2:
        raise OpContractError(f"{where}: {what} is not a list [H, W] of length 2: {value!r}")
    return (
        _assert_integer_attr(value[0], f"{where}[0]", minimum),
        _assert_integer_attr(value[1], f"{where}[1]", minimum),
    )


#: conv2d's attrs (ADR 0017). The 3 spatial ones have 2 components, H/W; groups is a scalar.
#:
#: MUST: all four are **required, never filled with defaults** (the same rule as conv1d — ADR
#: 0015). Depthwise and asymmetric stride/padding appear in measurements, so allowing omission would
#: silently give an ordinary convolution with symmetric padding.
CONV2D_ATTRS: AttrSchema = {
    "stride": lambda value, where: _assert_int_pair(value, where, 1, "conv2d stride"),
    "padding": lambda value, where: _assert_int_pair(value, where, 0, "conv2d padding"),
    "dilation": lambda value, where: _assert_int_pair(value, where, 1, "conv2d dilation"),
    "groups": lambda value, where: _assert_integer_attr(value, where, 1),
}


def _assert_int_triple(value: Any, where: str, minimum: int, what: str) -> tuple[int, int, int]:
    """A spatial attr of conv3d (3 components, `[T, H, W]` — ADR 0118 decision 1).

    MUST: accepts **lists of exactly length 3** only (the same rule as {@link _assert_int_pair}).
    Also allowing the scalar notation or 2 components would give the same convolution several IRs,
    and 2 components with the time axis dropped could not be told apart as "meant as [H, W]" or
    "a slip for [T, H]".
    """
    if not isinstance(value, list) or len(value) != 3:
        raise OpContractError(f"{where}: {what} is not a list [T, H, W] of length 3: {value!r}")
    return (
        _assert_integer_attr(value[0], f"{where}[0]", minimum),
        _assert_integer_attr(value[1], f"{where}[1]", minimum),
        _assert_integer_attr(value[2], f"{where}[2]", minimum),
    )


#: conv3d's attrs (ADR 0118 decision 1). The 3 spatial ones have 3 components, T/H/W; groups is a
#: scalar.
#:
#: MUST: all four are **required, never filled with defaults** (the same rule as conv1d / conv2d —
#: ADR 0015).
CONV3D_ATTRS: AttrSchema = {
    "stride": lambda value, where: _assert_int_triple(value, where, 1, "conv3d stride"),
    "padding": lambda value, where: _assert_int_triple(value, where, 0, "conv3d padding"),
    "dilation": lambda value, where: _assert_int_triple(value, where, 1, "conv3d dilation"),
    "groups": lambda value, where: _assert_integer_attr(value, where, 1),
}

#: conv_transpose1d's attrs (ADR 0015).
#:
#: MUST: stride >= 1 (stride 0 is a division by zero in the kernel and hangs the GPU —
#: docs/research/2026-08-02-sbv2-chain-recon.md §4).
#: MUST: no output_padding / dilation / groups fields. All observed values are 0 / 1 / 1, and not
#: having the fields structurally removes the path that "silently runs a value outside the
#: measurements with the default".
CONV_TRANSPOSE1D_ATTRS: AttrSchema = {
    "stride": lambda value, where: _assert_integer_attr(value, where, 1),
    "padding": lambda value, where: _assert_integer_attr(value, where, 0),
}

#: deform_conv2d's attrs (layer 1' — now the extended atomic layer — ADR 0055). **The single key
#: `padding` (`[H, W]`) only**.
#:
#: MUST: no stride / dilation / groups / offset_groups keys are added (= fixed to 1). The value has
#: the same form as conv2d's, a list [H, W] of length 2, and the scalar notation is normalized at
#: the boundary.
DEFORM_CONV2D_ATTRS: AttrSchema = {
    "padding": lambda value, where: _assert_int_pair(value, where, 0, "deform_conv2d padding"),
}

#: upsample_bilinear2d's attrs (layer 1 — now the Core ATen layer). **The single key `output_size`
#: (`[Hout, Wout]`) only**.
#:
#: MUST: a list [H, W] of length 2 like conv2d's; the scalar notation is not accepted (the same
#: resample would get two IRs — normalization is the boundary's job).
#: MUST: no `align_corners` / `mode` / `scale_factor` fields are added. The missing field itself
#: declares "that form is not in the vocabulary", and it pairs with the handler's fail loudly.
UPSAMPLE_BILINEAR2D_ATTRS: AttrSchema = {
    "output_size": lambda value, where: _assert_int_pair(
        value, where, 1, "upsample_bilinear2d output_size"
    ),
}


def layer_norm_attrs(attrs: Mapping[str, Any], where: str) -> tuple[list[int], float]:
    """The layer_norm node's (normalized_shape, eps).

    Assumes assert_node_contract has done the checks (this only reads them again).
    """
    return (
        _assert_normalized_shape(attrs.get("normalized_shape"), f"{where} attrs.normalized_shape"),
        _assert_eps(attrs.get("eps"), f"{where} attrs.eps", "layer_norm"),
    )


def rms_norm_eps(attrs: Mapping[str, Any], where: str) -> float:
    """The rms_norm node's eps (assumes assert_node_contract has done the checks)."""
    return _assert_eps(attrs.get("eps"), f"{where} attrs.eps", "rms_norm")


def softmax_dim(attrs: Mapping[str, Any], where: str) -> int:
    """The softmax node's reduction axis (a non-negative axis number)."""
    return _assert_integer_attr(attrs.get("dim"), f"{where} attrs.dim", 0)


def attention_scale(attrs: Mapping[str, Any], where: str) -> float:
    """The attention node's half scale (assumes assert_node_contract has done the checks)."""
    return _assert_finite_attr(attrs.get("scale"), f"{where} attrs.scale", "attention scale")


def state_window(attrs: Mapping[str, Any], where: str) -> int | None:
    """The `window` of a state-referencing node (`None` = the whole context when not declared)."""
    if "window" not in attrs:
        return None
    return _assert_integer_attr(attrs["window"], f"{where} attrs.window", 1)


def cumsum_dim(attrs: Mapping[str, Any], where: str) -> int:
    """The cumsum node's accumulation axis (a non-negative axis number)."""
    return _assert_integer_attr(attrs.get("dim"), f"{where} attrs.dim", 0)


def reduce_dim(attrs: Mapping[str, Any], where: str) -> int:
    """The reduction axis of a reduce-family node (a non-negative axis number — no defaults)."""
    return _assert_integer_attr(attrs.get("dim"), f"{where} attrs.dim", 0)


def topk_k(attrs: Mapping[str, Any], where: str) -> int:
    """The topk node's count k (a positive integer — no defaults)."""
    return _assert_integer_attr(attrs.get("k"), f"{where} attrs.k", 1)


def scalar_param_values(contract: OpContract, attrs: Mapping[str, Any], where: str) -> list[float]:
    """Take the scalar attrs in params order
    (equivalent to scalarParamValues in packages/runtime/src/ops/contracts.ts).

    MUST: clamp's `min <= max` can only be checked here (the attrs schema checks key by key, so it
    cannot express an invariant spanning two keys). Allowing an inversion would let the exporter
    write IR in the direction where WGSL's clamp is undefined.
    """
    keys = SCALAR_PARAM_ATTRS.get(contract.name)
    if keys is None:
        return []
    values = [
        _assert_finite_attr(attrs.get(key), f"{where} attrs.{key}", f"op '{contract.name}' {key}")
        for key in keys
    ]
    if contract.name == "clamp" and values[0] > values[1]:
        raise OpContractError(f"{where}: clamp min {values[0]} exceeds max {values[1]}")
    return values


def conv1d_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int, int, int]:
    """The conv1d node's (stride, padding, dilation, groups)."""
    return (
        _assert_integer_attr(attrs.get("stride"), f"{where} attrs.stride", 1),
        _assert_integer_attr(attrs.get("padding"), f"{where} attrs.padding", 0),
        _assert_integer_attr(attrs.get("dilation"), f"{where} attrs.dilation", 1),
        _assert_integer_attr(attrs.get("groups"), f"{where} attrs.groups", 1),
    )


def conv2d_attrs(
    attrs: Mapping[str, Any], where: str
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]:
    """The conv2d node's (stride, padding, dilation, groups); the 3 spatial ones are (H, W)."""
    return (
        _assert_int_pair(attrs.get("stride"), f"{where} attrs.stride", 1, "conv2d stride"),
        _assert_int_pair(attrs.get("padding"), f"{where} attrs.padding", 0, "conv2d padding"),
        _assert_int_pair(attrs.get("dilation"), f"{where} attrs.dilation", 1, "conv2d dilation"),
        _assert_integer_attr(attrs.get("groups"), f"{where} attrs.groups", 1),
    )


def conv3d_attrs(
    attrs: Mapping[str, Any], where: str
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], int]:
    """The conv3d node's (stride, padding, dilation, groups); the 3 spatial ones are (T, H, W)."""
    return (
        _assert_int_triple(attrs.get("stride"), f"{where} attrs.stride", 1, "conv3d stride"),
        _assert_int_triple(attrs.get("padding"), f"{where} attrs.padding", 0, "conv3d padding"),
        _assert_int_triple(attrs.get("dilation"), f"{where} attrs.dilation", 1, "conv3d dilation"),
        _assert_integer_attr(attrs.get("groups"), f"{where} attrs.groups", 1),
    )


def conv_transpose1d_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int]:
    """The conv_transpose1d node's (stride, padding)."""
    return (
        _assert_integer_attr(attrs.get("stride"), f"{where} attrs.stride", 1),
        _assert_integer_attr(attrs.get("padding"), f"{where} attrs.padding", 0),
    )


def deform_conv2d_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int]:
    """The deform_conv2d node's padding (an (H, W) pair)."""
    return _assert_int_pair(
        attrs.get("padding"),
        f"{where} attrs.padding",
        0,
        "deform_conv2d padding",
    )


def upsample_bilinear2d_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int]:
    """The upsample_bilinear2d node's output_size (an (Hout, Wout) pair)."""
    return _assert_int_pair(
        attrs.get("output_size"),
        f"{where} attrs.output_size",
        1,
        "upsample_bilinear2d output_size",
    )


@dataclass(frozen=True)
class UniformDtypes:
    """A contract in which every slot has the same accepted set and the slots must also have **the
    same type as each other**.

    The output has the same type as the inputs. elementwise / matmul / bmm / row reduce / layout
    use this.
    """

    accept: frozenset[str]


@dataclass(frozen=True)
class PerSlotDtypes:
    """A contract whose accepted set differs per slot (an extension of ADR 0012).

    The slots need not share a type, and the output has **the type of slot 0 (the value side)**.
    gather's src=f32 / index=i32 → out=f32 is the first example.

    MUST: an op whose slots share one accepted set is not written here. UniformDtypes also carries
    the rule that rejects mixes such as `mul(f32, i32)`, and collapsing it to per-slot silently
    removes that rejection.
    """

    slots: tuple[frozenset[str], ...]


SlotDtypes = UniformDtypes | PerSlotDtypes


@dataclass(frozen=True)
class StateFieldContract:
    """Key set and requiredness of the `states` field (ADR 0067 decisions 4 / 5 — the mirror of
    StateFieldContract on the TS side).

    `keys` is **exactly** the key set when the field is non-empty, and `required` is "whether the
    field itself is required". `attention` is `(("k", "v"), required=False)` — the presence of the
    field decides between the old form and the states form. `state_append` is
    `(("slot",), required=True)`.
    """

    keys: tuple[str, ...]
    required: bool


@dataclass(frozen=True)
class OpContract:
    name: str
    kind: OpKind
    #: Number of inputs (the number of outputs is the length of output_dtypes — 2 for `topk`, 0 for
    #: `state_append`, 1 for all others).
    arity: int
    #: The source of truth for per-slot accepted sets and the output derivation (only cast's
    #: output dtype is decided by attrs.to).
    slot_dtypes: SlotDtypes
    #: The **sequence**, **per output slot**, of mappings "slot 0 input dtype → that output's
    #: dtype" (ADR 0068 decision 1). The sequence length is the op's output count; only `topk`
    #: (2) and `state_append` (**0** — an effect op that defines no value, ADR 0067 decision 5)
    #: have a length other than 1.
    #: Each mapping defaults to identity, and only the 3 cases seen in measurements differ
    #: (comparison → bool / sum of bool → i32 / where → the value side). The domain matches the
    #: accepted set of slot 0 exactly (_contract / _slot_contract fill it with identity).
    #:
    #: MUST: the output count is derived from the sequence length (output_count) — holding it in a
    #: separate field would allow "2 mappings, 1 declared", and nobody would check the dtype of the
    #: second output.
    output_dtypes: tuple[Mapping[str, str], ...]
    #: attrs schema. When empty, non-empty attrs fail loudly.
    attrs: AttrSchema
    #: An op with a variable input count (currently only `cat`). When True, `arity` is the
    #: **lower bound** (arity_fits).
    #:
    #: MUST: not set "to relax the arity check". Only cat's input count is variable; for other ops
    #: the count itself is the contract (e.g. fixed arity 3 with bias always present).
    variadic: bool = False
    #: Upper bound of the input count for an op with **an optional trailing input** (currently only
    #: `attention`'s mask). When set, `arity` is the lower bound and this the upper bound, and only
    #: counts in that closed interval are accepted.
    #:
    #: MUST: not used in place of `variadic`. `variadic` is "any number" (cat's concatenation), this
    #: is "one fixed slot more" — collapsing it into a form without an upper bound would make the
    #: contract accept forms where extra inputs are silently ignored (slots the kernel never reads).
    max_arity: int | None = None
    #: **Optional attrs** (currently the single `window` — ADR 0067 decisions 4 / 5). When declared
    #: they pass the range check; when absent, "the absence of the field" carries the meaning.
    #:
    #: MUST: not used to make required attrs (`attrs`) optional. "Declared attrs are never filled
    #: with defaults" (ADR 0012) stays, and only **fields whose absence is itself a different
    #: declaration** go here (`window` absent = the whole context).
    #: MUST: optional attrs are **for the states form only** (assert_node_contract). The sliding
    #: window is part of "the logical col → physical row mapping" and means nothing for a node that
    #: does not reference state — being able to write it in the old form would create "an attr that
    #: is accepted but read by nobody".
    optional_attrs: AttrSchema = field(default_factory=dict)
    #: Contract of the `states` field (None = **an op that cannot have a states field**).
    states: StateFieldContract | None = None

    @property
    def dtypes(self) -> frozenset[str]:
        """The **union** of the accepted sets (a projection used for capability matching and
        diagnostics).

        MUST: derived from slot_dtypes — a hand-written set drifts from the table and creates
        "executable per the support table, fails in the contract check".
        """
        if isinstance(self.slot_dtypes, UniformDtypes):
            return self.slot_dtypes.accept
        return frozenset().union(*self.slot_dtypes.slots)

    def slot_accept(self, slot: int) -> frozenset[str]:
        """The accepted set of input slot `slot` (shared by all slots when uniform)."""
        if isinstance(self.slot_dtypes, UniformDtypes):
            return self.slot_dtypes.accept
        if not 0 <= slot < len(self.slot_dtypes.slots):
            raise OpContractError(f"op '{self.name}' has no input slot {slot}")
        return self.slot_dtypes.slots[slot]

    @property
    def attr_keys(self) -> frozenset[str]:
        """Keys the schema declares as **required** (a projection for support-table matching — no
        duplicated state)."""
        return frozenset(self.attrs)

    @property
    def optional_attr_keys(self) -> frozenset[str]:
        """Keys the schema declares as **optional** (`optional_attrs`).

        MUST: capability matching (verify.assert_runtime_support) gets the **union** of required and
        optional — the enumeration check lists the difference from "implemented attr keys" as
        "unimplemented attrs", so dropping the optional ones would reject a correct states-form
        graph for missing capabilities.
        """
        return frozenset(self.optional_attrs)

    @property
    def output_count(self) -> int:
        """The number of outputs the contract declares (the length of the output dtype mapping
        sequence itself — ADR 0068 decision 1)."""
        return len(self.output_dtypes)

    def output_dtype(self, slot: int, input_dtype: str) -> str:
        """Derive the dtype **of output slot `slot`** from the input dtype of slot 0
        (equivalent to outputDtypeOf in packages/runtime/src/ops/contracts.ts).
        """
        if not 0 <= slot < len(self.output_dtypes):
            raise OpContractError(f"op '{self.name}' has no output slot {slot}")
        mapped = self.output_dtypes[slot].get(input_dtype)
        if mapped is None:
            raise OpContractError(
                f"op '{self.name}' output dtype mapping has no input '{input_dtype}'"
            )
        return mapped


#: Per-op table of enabled dtypes. Ops not listed stay f32 (no enabling all at once — ADR 0007).
_DTYPES: dict[str, frozenset[str]] = {
    "mul": F32_I32_DTYPES,
    "sub": F32_I32_DTYPES,
    "bitwise_not": BOOL_DTYPES,
    # The spline's `inside = (x >= -b) & (x <= b)`
    # (docs/research/2026-08-02-sbv2-chain-recon.md §2) — a logical AND of bools.
    "bitwise_and": BOOL_DTYPES,
    "sum": F32_BOOL_DTYPES,
    # reshape reads and writes no element (aliasing only), so it takes the whole vocabulary. The
    # measurements show all of: view of f32, squeeze/unsqueeze of an i32 mask, unsqueeze of a bool
    # mask (docs/research/2026-08-02-deberta-front-recon.md §2).
    RESHAPE_OP: frozenset(SEMANTIC_DTYPES),
    # gather indices (i32) and the bool masks of the conv path, plus making relative position
    # embeddings 4D (f32 — the enc_p / flow rows of docs/research/2026-08-02-sbv2-chain-recon.md
    # §2). The strided copy family is dtype-parametric, so it stays shared.
    EXPAND_OP: frozenset(SEMANTIC_DTYPES),
    # The baked constants are of two kinds: the relative position bucket table (i32) and the
    # position table (f32). bool initializers are not in the vocabulary, so bool is not enabled.
    SYM_PREFIX_SLICE_OP: F32_I32_DTYPES,
}

#: Table that declares the **sequence** of per-output-slot dtype mappings (ADR 0068 decision 1).
#: Ops not listed have "1 output, identity". The non-identity ones are only: the 4 comparisons
#: (f32 → bool) / sum of bool input (→ an i32 count) / where (bool → f32) / argmax (→ i32
#: indices) / topk (sequence length 2 — slot 0's values identity, slot 1 the i32 indices) /
#: state_append (**an empty sequence** = an effect op that defines no value, ADR 0067 decision 5).
_OUTPUT_DTYPES: dict[str, tuple[dict[str, str], ...]] = {
    "ge": ({"f32": "bool"},),
    "ge_scalar": ({"f32": "bool"},),
    "le_scalar": ({"f32": "bool"},),
    "gt_scalar": ({"f32": "bool"},),
    "sum": ({"f32": "f32", "bool": "i32"},),
    WHERE_OP: ({"bool": "f32"},),
    ARGMAX_OP: ({"f32": "i32"},),
    # The sequence length is the output count (ADR 0068 decision 3 — topk emits 2).
    TOPK_OP: ({}, {"f32": "i32"}),
    # An empty sequence = 0 outputs (ADR 0067 decision 5). This one line is the declaration of "an
    # effect op that defines no value" itself, and `outs: []` is allowed only for ops whose contract
    # declares this.
    STATE_APPEND_OP: (),
}

#: Default for ops without a declaration = **1 output, identity** (an empty mapping fills its
#: whole domain with identity).
_SINGLE_IDENTITY_OUTPUT: tuple[dict[str, str], ...] = ({},)


def _output_dtypes(name: str, slot_dtypes: SlotDtypes) -> tuple[Mapping[str, str], ...]:
    """The sequence of mappings per output slot (undeclared dtypes are filled with identity).

    MUST: each mapping's domain matches the accepted set of slot 0 — a partial mapping would open a
    hole where "the slot check passed but the output dtype is undecided".
    """
    declared = _OUTPUT_DTYPES.get(name, _SINGLE_IDENTITY_OUTPUT)
    domain = slot_dtypes.accept if isinstance(slot_dtypes, UniformDtypes) else slot_dtypes.slots[0]
    return tuple({dtype: slot.get(dtype, dtype) for dtype in domain} for slot in declared)


def _contract(
    name: str,
    kind: OpKind,
    arity: int,
    attrs: AttrSchema | None = None,
    *,
    variadic: bool = False,
    max_arity: int | None = None,
    optional_attrs: AttrSchema | None = None,
    states: StateFieldContract | None = None,
) -> OpContract:
    return _slot_contract(
        name,
        kind,
        arity,
        UniformDtypes(accept=_DTYPES.get(name, F32_DTYPES)),
        attrs,
        variadic=variadic,
        max_arity=max_arity,
        optional_attrs=optional_attrs,
        states=states,
    )


def _slot_contract(
    name: str,
    kind: OpKind,
    arity: int,
    slot_dtypes: SlotDtypes,
    attrs: AttrSchema | None = None,
    *,
    variadic: bool = False,
    max_arity: int | None = None,
    optional_attrs: AttrSchema | None = None,
    states: StateFieldContract | None = None,
) -> OpContract:
    return OpContract(
        name=name,
        kind=kind,
        arity=arity,
        slot_dtypes=slot_dtypes,
        output_dtypes=_output_dtypes(name, slot_dtypes),
        attrs=attrs if attrs is not None else {},
        variadic=variadic,
        max_arity=max_arity,
        optional_attrs=optional_attrs if optional_attrs is not None else {},
        states=states,
    )


#: Schemas of the unary ops that have attrs (ops not listed have empty attrs).
_UNARY_ATTRS: dict[str, AttrSchema] = {
    "clamp": CLAMP_ATTRS,
    "clamp_min": CLAMP_MIN_ATTRS,
    "leaky_relu": LEAKY_RELU_ATTRS,
    "ge_scalar": SCALAR_COMPARE_ATTRS,
    "le_scalar": SCALAR_COMPARE_ATTRS,
    "gt_scalar": SCALAR_COMPARE_ATTRS,
}


OP_CONTRACTS: dict[str, OpContract] = {
    **{name: _contract(name, "unary", 1, _UNARY_ATTRS.get(name)) for name in UNARY_OPS},
    **{name: _contract(name, "binary", 2) for name in BINARY_OPS},
    **{name: _contract(name, "row_reduce", 1, REDUCE_ATTRS) for name in REDUCE_OPS},
    # Per-slot contract of a bool condition and f32 values. The output is **the value side**
    # (mapping bool → f32).
    WHERE_OP: _slot_contract(
        WHERE_OP, "where", 3, PerSlotDtypes(slots=(BOOL_DTYPES, F32_DTYPES, F32_DTYPES))
    ),
    CUMSUM_OP: _contract(CUMSUM_OP, "cumsum", 1, CUMSUM_ATTRS),
    MATMUL_OP: _contract(MATMUL_OP, "matmul", 2),
    BMM_OP: _contract(BMM_OP, "bmm", 2),
    # The first per-slot dtype contract: f32 values and i32 indices mixed, the output has the type
    # of the value side.
    GATHER_OP: _slot_contract(
        GATHER_OP, "gather", 2, PerSlotDtypes(slots=(F32_DTYPES, I32_DTYPES))
    ),
    # argmax over the last dimension (ADR 0068 decision 2). Empty attrs, input f32 → output i32
    # (the mapping is in _OUTPUT_DTYPES); the shape rule holds the rank preservation.
    ARGMAX_OP: _contract(ARGMAX_OP, "argmax", 1),
    # top-k over the last dimension (ADR 0068 decision 3). attrs `k` required, 2 outputs (values
    # f32 + indices i32 — the mapping is in _OUTPUT_DTYPES); shapes.py holds the `[..., k]` shape
    # rule and the accepted domain.
    TOPK_OP: _contract(TOPK_OP, "topk", 1, TOPK_ATTRS),
    CAST_OP: _slot_contract(
        CAST_OP, "cast", 1, UniformDtypes(accept=frozenset(SEMANTIC_DTYPES)), CAST_ATTRS
    ),
    RESHAPE_OP: _contract(RESHAPE_OP, "reshape", 1),
    PERMUTE_OP: _contract(PERMUTE_OP, "permute", 1, PERMUTE_ATTRS),
    EXPAND_OP: _contract(EXPAND_OP, "expand", 1),
    # Layout, second group (ADR 0014). All observed are f32 (enc_p's m_p/logs_p split, coupling's
    # 96/96 split and cat, the pad on the relative-position value side, flow/sdp's Flip —
    # docs/research/2026-08-02-sbv2-chain-recon.md §2).
    SLICE_OP: _contract(SLICE_OP, "slice", 1, SLICE_ATTRS),
    # The only variable-arity op. arity is the **lower bound** 2 (a cat of 1 is an identity copy and
    # does not appear in measurements either).
    CAT_OP: _contract(CAT_OP, "cat", 2, AXIS_ATTRS, variadic=True),
    PAD_OP: _contract(PAD_OP, "pad", 1, PAD_ATTRS),
    FLIP_OP: _contract(FLIP_OP, "flip", 1, AXIS_ATTRS),
    SYM_PREFIX_SLICE_OP: _contract(
        SYM_PREFIX_SLICE_OP, "sym_prefix_slice", 1, SYM_PREFIX_SLICE_ATTRS
    ),
    # Fused ops (ADR 0012). Fused ops with bias / affine have fixed arity 3 — measurements always
    # have bias, and not having a field for "no bias" is itself the fail loudly.
    LINEAR_OP: _contract(LINEAR_OP, "linear", 3),
    LAYER_NORM_OP: _contract(LAYER_NORM_OP, "layer_norm", 3, LAYER_NORM_ATTRS),
    # No bias, so arity 2 (ADR 0017). The handler normalizes the form without weight to arity 2 by
    # synthesizing ones — the same approach as the zero-bias synthesis (ADR 0015).
    RMS_NORM_OP: _contract(RMS_NORM_OP, "rms_norm", 2, RMS_NORM_ATTRS),
    STATIC_QUANTIZE_OP: _contract(STATIC_QUANTIZE_OP, "static_quantize", 1, STATIC_QUANTIZE_ATTRS),
    SOFTMAX_OP: _contract(SOFTMAX_OP, "softmax", 1, SOFTMAX_ATTRS),
    # ADR 0044. The attrs schema **shares the same single one** with softmax (a copy would allow a
    # form where only one of them is narrowed less).
    SAFE_SOFTMAX_OP: _contract(SAFE_SOFTMAX_OP, "safe_softmax", 1, SOFTMAX_ATTRS),
    # Fused attention (ADR 0023). q / k / v and the optional mask, all 4 f32 of the same type
    # (uniform contract). The mask is additive, so it has the value side's dtype, and bool is not
    # accepted.
    # The states field (ADR 0067 decision 4) is **optional** — the presence of the field decides
    # between the old form and the states form, and only the form with the field can declare
    # `window` (the old form's accepted set does not move by a single byte).
    ATTENTION_OP: _contract(
        ATTENTION_OP,
        "attention",
        3,
        ATTENTION_ATTRS,
        max_arity=4,
        # `readonly` is a declaration that changes the input count itself (the single q — ADR 0096
        # stage 2). It follows the same rule as `window`: "declarable only on nodes with a states
        # field".
        optional_attrs={**STATE_WINDOW_ATTRS, **STATE_READONLY_ATTRS},
        states=StateFieldContract(keys=("k", "v"), required=False),
    ),
    # Write into a state slot (ADR 0067 decision 5). 1 input, **0 outputs** (the empty sequence in
    # _OUTPUT_DTYPES is the declaration), states field required.
    STATE_APPEND_OP: _contract(
        STATE_APPEND_OP,
        "state_append",
        1,
        optional_attrs=STATE_WINDOW_ATTRS,
        states=StateFieldContract(keys=("slot",), required=True),
    ),
    # Per-slot contract of f32 values and i32 indices (the same shape as gather — the output has
    # the type of the value side).
    EMBEDDING_OP: _slot_contract(
        EMBEDDING_OP,
        "embedding",
        2,
        PerSlotDtypes(slots=(F32_DTYPES, I32_DTYPES)),
        EMBEDDING_ATTRS,
    ),
    # Per-slot contract of f32 values and a bool condition. The output has the type of slot 0 (the
    # value side).
    MASKED_FILL_OP: _slot_contract(
        MASKED_FILL_OP,
        "masked_fill",
        2,
        PerSlotDtypes(slots=(F32_DTYPES, BOOL_DTYPES)),
        MASKED_FILL_ATTRS,
    ),
    CONV1D_OP: _contract(CONV1D_OP, "conv1d", 3, CONV1D_ATTRS),
    CONV2D_OP: _contract(CONV2D_OP, "conv2d", 3, CONV2D_ATTRS),
    # Unbatched (x is rank 4) 3D convolution (ADR 0118 decision 1). All 3 slots f32 of the same
    # type.
    CONV3D_OP: _contract(CONV3D_OP, "conv3d", 3, CONV3D_ATTRS),
    # A conv without bias is normalized to arity 3 by the exporter's zero-bias synthesis (ADR
    # 0015) — no arity branches in either the kernel or the contract.
    CONV_TRANSPOSE1D_OP: _contract(
        CONV_TRANSPOSE1D_OP, "conv_transpose1d", 3, CONV_TRANSPOSE1D_ATTRS
    ),
    # DCNv2 (layer 1' — now the extended atomic layer — ADR 0055). All 5 slots f32 of the same
    # type. It has a weight but is **not eligible for low-precision storage**, so it is not in
    # WEIGHT_SLOTS.
    DEFORM_CONV2D_OP: _contract(DEFORM_CONV2D_OP, "deform_conv2d", 5, DEFORM_CONV2D_ATTRS),
    # Bilinear resample (layer 1 — now the Core ATen layer). It has no weight, so it is not in
    # WEIGHT_SLOTS.
    UPSAMPLE_BILINEAR2D_OP: _contract(
        UPSAMPLE_BILINEAR2D_OP, "upsample_bilinear2d", 1, UPSAMPLE_BILINEAR2D_ATTRS
    ),
    # GRU hidden-side scan, 2 directions (layer 2 — now the extended composite layer — ADR 0056).
    # All 4 slots f32 of the same type, empty attrs.
    # w_hh is a slot inside the op, so it is **not eligible for low-precision storage** (not in
    # WEIGHT_SLOTS) — the input-side weight belongs to the caller's linear, so f16 / i8 still apply
    # there as before.
    **{name: _contract(name, "gru_scan", 4) for name in GRU_SCAN_OPS},
}

#: Ops in the contract table that convert **cannot emit** (no aten handler = not exportable from
#: torch).
#:
#: MUST: listed with reasons. The golden coverage check (goldens.generate_all) looks at
#: EMITTABLE_OPS, so an op put here passes as "an op without goldens" — the empty set is the
#: default, and adding one makes "not exportable from torch" itself the check (tests/test_convert.py
#: pins that converting `torch.topk` raises UnsupportedAtenOpsError).
#:
#: - `topk` — wiring the getitem of multi-output aten ops is deferred until sampling needs it (ADR
#:   0068 addendum 2).
#: - `state_append` — **no corresponding aten op exists** (an effect op of ADR 0067 decision 5,
#:   emitted by the decode graph script). It cannot come out of the torch.export path in principle.
NON_EMITTABLE_OPS = frozenset({TOPK_OP, STATE_APPEND_OP})
#: The source of truth for the IR op names convert may emit. The check that makes growing the
#: vocabulary an explicit act.
EMITTABLE_OPS = frozenset(OP_CONTRACTS) - NON_EMITTABLE_OPS


#: Ops subject to the rank limit (the strided copy family — the kernels' params have a fixed rank).
#: MUST: the same set as the TS side (the ops for which computeOutputShape in
#: packages/runtime/src/ops/shapes.ts calls the private assertStridedRank). Keeping it on one side
#: only brings back "export is green, only the browser fails" on the rank axis.
STRIDED_RANK_OPS = frozenset(
    {PERMUTE_OP, EXPAND_OP, SLICE_OP, CAT_OP, SYM_PREFIX_SLICE_OP, MASKED_FILL_OP}
)


def assert_strided_rank(rank: int, what: str, where: str) -> None:
    """The rank the strided copy family can execute
    (equivalent to the private assertStridedRank in packages/runtime/src/ops/shapes.ts — not
    exported by the ops.ts barrel).

    MUST: checked in the contract layer. Letting an excess fall through to codegen gives "the
    contract check passed but an internal error at execution", and the diagnostics do not show
    which value of which op is wrong.
    """
    if not 1 <= rank <= STRIDED_RANK:
        raise OpContractError(
            f"{where}: {what} rank {rank} is outside 1..{STRIDED_RANK} (the strided kernel limit)"
        )


def arity_fits(contract: OpContract, count: int) -> bool:
    """Whether the input count fits the contract (for variable-arity ops `arity` is read as the
    **lower bound**).

    MUST: the decision is not scattered among callers (equivalent to arityFits in
    packages/runtime/src/ops/contracts.ts). Writing "== when fixed, >= when variable" in both the
    contract check and the shape computation leaves an old rule behind in one of them.
    """
    if contract.variadic:
        return count >= contract.arity
    upper = contract.arity if contract.max_arity is None else contract.max_arity
    return contract.arity <= count <= upper


def describe_arity(contract: OpContract) -> str:
    """Display form of the contract's arity ("N or more" when variable, "N or M" with an optional
    trailing input)."""
    if contract.variadic:
        return f"{contract.arity} or more"
    return (
        str(contract.arity)
        if contract.max_arity is None
        else f"{contract.arity} or {contract.max_arity}"
    )


def resolve_op_contract(op: str) -> OpContract:
    contract = OP_CONTRACTS.get(op)
    if contract is None:
        raise OpContractError(f"op '{op}' is not in the contract table")
    return contract


def cast_target_dtype(attrs: Mapping[str, Any], where: str) -> str:
    """The cast node's target dtype (assumes assert_node_contract has done the attrs checks)."""
    return _assert_cast_target(attrs.get("to"), f"{where} attrs.to")


def _assert_state_field(contract: OpContract, node: IrNode, keys: list[str], where: str) -> None:
    """Whether the `states` field fits the contract (ADR 0067 decisions 4 / 5 — the mirror of
    assertStateField on the TS side).

    MUST: the key set **matches** the contract's declaration **exactly** (allowing a subset would
    let a states-form attention that forgot `v` pass as a form that "reads only k from state").
    MUST: a node that references **the same slot** from two fields is rejected — a mix-up that
    writes the same slot name for `k` and `v` passes every shape check and becomes a silent wrong
    value that "reads V as K" (the k/v slots have the same shape, so shapes do not catch it).
    """
    declared = contract.states
    if declared is None:
        if keys:
            raise OpContractError(
                f"{where}: op '{contract.name}' has no states field ({sorted(keys)})"
            )
        return
    if not keys:
        if declared.required:
            raise OpContractError(
                f"{where}: op '{contract.name}' states field is missing"
                f" (the contract is exactly {list(declared.keys)})"
            )
        return
    if sorted(keys) != sorted(declared.keys):
        raise OpContractError(
            f"{where}: op '{contract.name}' states field keys are {sorted(keys)}"
            f" (the contract is exactly {sorted(declared.keys)})"
        )
    slots = [node.states[key] for key in declared.keys]
    for index, slot in enumerate(slots):
        if slots.index(slot) != index:
            raise OpContractError(
                f"{where}: op '{contract.name}' states field references the same slot '{slot}' "
                "from several fields (the mix-up detector)"
            )


def assert_node_contract(node: IrNode, where: str) -> OpContract:
    """Check that the node fits its contract and return the contract (shapes not included)."""
    contract = resolve_op_contract(node.op)
    state_keys = list(node.states)
    _assert_state_field(contract, node, state_keys, where)
    # MUST: `readonly` decides the arity **before** the attrs range check. The readonly form has
    # no k/v of this step (it reads only the past), so its input is exactly one, q; applying the
    # old 3-to-4 input rule as is would make "a correct readonly graph fail with an arity
    # violation".
    readonly = (
        bool(state_keys)
        and "readonly" in contract.optional_attrs
        and attention_readonly(node.attrs)
    )
    if readonly:
        if len(node.ins) != READONLY_ATTENTION_ARITY:
            raise OpContractError(
                f"{where}: op '{node.op}' readonly form takes exactly"
                f" {READONLY_ATTENTION_ARITY} input ({len(node.ins)} given — neither this step's"
                " k/v nor a mask, ADR 0096 stage 2)"
            )
    else:
        # MUST: the states form **takes no optional trailing input** (ADR 0067 decision 4 — "causal
        # fixed, no mask tensor materialized"). Without narrowing the upper bound, a states-form
        # attention with a mask would be accepted as "a form whose mask nobody reads".
        if state_keys and contract.max_arity is not None and len(node.ins) > contract.arity:
            raise OpContractError(
                f"{where}: op '{node.op}' states form takes exactly {contract.arity} inputs"
                f" ({len(node.ins)} given — no optional trailing input, ADR 0067 decision 4)"
            )
        if not arity_fits(contract, len(node.ins)):
            raise OpContractError(
                f"{where}: op '{node.op}' input count is {len(node.ins)}"
                f" (the contract is {describe_arity(contract)})"
            )
    if len(node.outs) != contract.output_count:
        raise OpContractError(
            f"{where}: op '{node.op}' output count is {len(node.outs)}"
            f" (the contract is {contract.output_count})"
        )
    unknown = sorted(
        key
        for key in node.attrs
        if key not in contract.attrs and key not in contract.optional_attrs
    )
    if unknown:
        raise OpContractError(f"{where}: op '{node.op}' attrs outside the contract {unknown}")
    for key, check in contract.attrs.items():
        if key not in node.attrs:
            raise OpContractError(f"{where}: op '{node.op}' required attr '{key}' is missing")
        check(node.attrs[key], f"{where} attrs.{key}")
    for key, check in contract.optional_attrs.items():
        if key not in node.attrs:
            continue
        # MUST: optional attrs are for the states form only (OpContract.optional_attrs).
        if not state_keys:
            raise OpContractError(
                f"{where}: op '{node.op}' attrs.{key} can be declared only on a node with a states"
                " field"
            )
        check(node.attrs[key], f"{where} attrs.{key}")
    return contract


def resolve_node_dtypes(
    contract: OpContract,
    node: IrNode,
    input_dtypes: list[str],
    declared_outputs: list[str],
    where: str,
) -> list[str]:
    """Check the node's semantic dtypes and return **the sequence of output dtypes** (in output slot
    order — equivalent to packages/runtime/src/ops.ts).

    MUST: output dtypes are derived from the contract, not taken from the declaration on trust.
    There are only two derivations — cast uses attrs.to, everything else **passes the dtype of slot
    0 through each output slot's mapping** (identity by default). Uniform contracts additionally
    require all slots to have the same type. A graph that disagrees with its declaration becomes a
    silent wrong value, read as a different TypedArray on the runtime side.
    """
    for slot, (name, dtype) in enumerate(zip(node.ins, input_dtypes, strict=True)):
        accept = contract.slot_accept(slot)
        if dtype not in accept:
            uniform = isinstance(contract.slot_dtypes, UniformDtypes)
            slot_note = "" if uniform else f" in slot {slot}"
            raise OpContractError(
                f"{where} input '{name}': op '{contract.name}' cannot execute{slot_note}"
                f" semantic dtype '{dtype}' (supported: {', '.join(sorted(accept))})"
            )
    expected: list[str]
    if contract.kind == "cast":
        expected = [cast_target_dtype(node.attrs, where)]
    else:
        # An op whose accepted set differs per slot does not require the slots to share a type
        # (that is what per-slot means). Only uniform contracts reject mixes.
        if isinstance(contract.slot_dtypes, UniformDtypes) and any(
            dtype != input_dtypes[0] for dtype in input_dtypes
        ):
            raise OpContractError(
                f"{where}: op '{contract.name}' input dtypes are mixed ({', '.join(input_dtypes)})"
            )
        expected = [
            contract.output_dtype(slot, input_dtypes[0]) for slot in range(contract.output_count)
        ]
    if len(declared_outputs) != len(expected):
        raise OpContractError(
            f"{where}: op '{contract.name}' declares {len(declared_outputs)} outputs"
            f" (the contract is {len(expected)})"
        )
    for slot, (declared, dtype) in enumerate(zip(declared_outputs, expected, strict=True)):
        if declared != dtype:
            raise OpContractError(
                f"{where}: output '{node.outs[slot]}' declared dtype '{declared}' differs from"
                f" the contract's '{dtype}'"
            )
    return expected
