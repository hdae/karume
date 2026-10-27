/**
 * IR that violates an op contract (unsupported op, arity violation, attrs outside the contract,
 * dtype violation, shape mismatch).
 */
export class OpContractError extends Error {
  override readonly name = "OpContractError";
}

/**
 * Unary elementwise.
 *
 * MUST: the kind means only "elementwise with one input". The 6 ops with attrs
 * (clamp / clamp_min / leaky_relu / ge_scalar / le_scalar / gt_scalar) and the 3 comparisons whose
 * output dtype differs from the input also belong here — the attrs schema and the output dtype live
 * in other fields of the contract table, so the kind is not split for those differences (splitting
 * it would add branches to codegen and the CPU reference for the same reason).
 */
export const UNARY_OPS = [
  "neg",
  "abs",
  "exp",
  "log",
  // WGSL has no log1p built-in (implementation approach: LOG1P_FN in src/codegen/elementwise.ts)
  "log1p",
  "sqrt",
  // The only trigonometric function is sin (Core ATen layer — ADR 0059; layer 1 of the superseded
  // ADR 0043). The exporter folds constant RoPE tables, so the vocabulary needs only the form that
  // takes a **run-time value** — first seen in DACVAE's Snake activation
  // `x + (α+1e-9)⁻¹·sin²(αx)`. `cos` is not added until it shows up in a measurement ("no
  // additions for symmetry").
  "sin",
  "tanh",
  "sigmoid",
  "relu",
  "gelu",
  // torch's `approximate="tanh"` form. Its values differ from the erf-form `gelu`, so it is a
  // separate op — an empty-attrs contract has no field for the approximation kind, and adding one
  // would create, outside the contract, a branch where "the same op name changes the numbers".
  "gelu_tanh",
  "bitwise_not",
  // The 6 ops that carry their attrs scalar at the end of params ({@link SCALAR_PARAM_ATTRS})
  "clamp",
  // One-sided clamp (ADR 0017). Making the existing clamp's attrs optional to cover it would break
  // "declared attrs are not filled with defaults" (ADR 0012), so that is not done — it is a
  // separate op.
  "clamp_min",
  "leaky_relu",
  "ge_scalar",
  "le_scalar",
  "gt_scalar",
] as const;
export type UnaryOpName = (typeof UNARY_OPS)[number];

/**
 * Binary elementwise (right-aligned broadcast as in torch). `ge` is f32 × f32 → **bool**; only its
 * output dtype differs from the input (derived by {@link OUTPUT_DTYPES}).
 */
export const BINARY_OPS = ["add", "sub", "mul", "div", "ge", "bitwise_and"] as const;
export type BinaryOpName = (typeof BINARY_OPS)[number];

/**
 * Reduce over one axis (attrs `dim`, no keepdim). The last dimension runs on the row kernel, other
 * axes on the axis variant (design: docs/research/2026-08-04-vae-axis-reduce-recon.md §5).
 */
export const REDUCE_OPS = ["sum", "amax", "amin"] as const;
export type ReduceOpName = (typeof REDUCE_OPS)[number];

/**
 * argmax over the last dimension (ADR 0068 decision 2). Input f32 → output **i32**,
 * **rank-preserving** (a fixed form that collapses the last dimension to 1).
 *
 * MUST: no `attrs`. The axis is fixed to the last dimension (the absence of a `dim` field is itself
 * the statement "other axes are not in the vocabulary" — the same narrowing as gather), and there is
 * no `keepdim` field either (it expresses the single rank-preserving form). If a need appears, it is
 * added as a separate op, the same discipline as `gelu` / `gelu_tanh`.
 * MUST: not part of the reduce family ({@link REDUCE_OPS}). The attrs, the output dtype and the rank
 * handling all differ, and the kernel is a separate family too (identity is −inf and it carries
 * (value, index) pairs — src/kernels/argmax.ts).
 * MUST: ties break to the **smallest index**, NaN is the **largest**, and an all −inf row gives
 * **index 0** (all as in torch. The source of truth for the implementation contract is
 * src/kernels/argmax.ts and reference/ops.ts).
 */
export const ARGMAX_OP = "argmax";

/**
 * top-k over the last dimension (ADR 0068 decision 3 — **the first op with node-level multiple
 * outputs**). Input f32 → **two outputs** (slot 0 = values f32 in **descending order**, slot 1 =
 * indices i32), both `[…, k]`.
 *
 * MUST: `k` is **required in attrs** (a plan-time constant = static-k). A symbolic k, k=0 and k above
 * the last dimension all fail loudly (accepted domain: `1 ≤ k ≤ last dimension`). torch's schema
 * makes `k` a SymInt, but a k decided at run time does not fit the static-shape premise (ADR 0004),
 * so it is not accepted.
 * MUST: the axis is fixed to the last dimension (the absence of a `dim` field is itself the
 * statement "other axes are not in the vocabulary"), and there are no `largest` / `sorted` fields
 * (**only the largest side, sorted descending,** is in the vocabulary — the smallest side would be a
 * separate op, and unordered output waits until it is asked for).
 * MUST: ties break to the **smallest index** (the same predicate family as argmax), NaN is the
 * **largest**, and an all −inf row also yields k entries from the smallest index. **The value
 * column is bit-identical to torch**, while **the index column is where karume specifies what torch
 * leaves unspecified** (torch does not guarantee the order of equal elements, and `argmax` and
 * `topk(k=1)` disagree within the same repository — the measurement is in the NOTE of
 * src/kernels/topk.ts).
 */
export const TOPK_OP = "topk";

/**
 * Ternary elementwise `out = cond ? a : b` (torch's `where`).
 *
 * MUST: all three broadcast right-aligned (as in torch). The condition slot comes first to match
 * torch's argument order (`where(cond, a, b)`), and **the output has the type of the value side,
 * not of the condition** (derived by {@link OUTPUT_DTYPES} as bool → f32).
 */
export const WHERE_OP = "where";

/**
 * Prefix sum over the last dimension (`out[…, j] = Σ_{i ≤ j} x[…, i]`, attrs `dim`).
 *
 * MUST: only **the last dimension** is accepted as the axis (the same narrowing as softmax — the row
 * kernel assumes the reduced axis is contiguous). Carrying `dim` in attrs also follows softmax:
 * negative axis notation is normalised at the exporter boundary, and the runtime re-checks that it
 * matches the last dimension in the shape computation, where the rank is known (one more check that
 * stops a mismatch between the declaration and the actual rank).
 */
export const CUMSUM_OP = "cumsum";

export const MATMUL_OP = "matmul";
/**
 * Batched matmul (ADR 0012). `[B,M,K] × [B,K,N] → [B,M,N]`, **rank-3 only**; rank-2 is the job of
 * {@link MATMUL_OP}. The measurements (the 8 of recon §2) are all rank-3 with batch axes that match
 * exactly (no broadcast, no stride 0).
 *
 * MUST: not shared with matmul. If one op branched to a different kernel by rank, the contract's
 * "input shapes → output shape" would become rank-dependent polymorphism, and gaps in the shape check
 * would become invisible.
 */
export const BMM_OP = "bmm";
/**
 * gather over the last dimension (ADR 0012). `out[..., j] = src[..., index[..., j]]`.
 *
 * MUST: the axis is **fixed to the last dimension** and there are no attrs (only dim = −1 was
 * measured — the same narrowing as softmax. A general dim is widened once a measured need appears).
 * MUST: the **first** per-slot dtype contract (src=f32 / index=i32 → out=f32). The same kind is used
 * by embedding / masked_fill / where; the source of truth for the list is the contract table (the
 * `SlotDtypes` doc and the actual `kind: "perSlot"` values in ops/contracts.ts) — the count is not
 * enumerated here (it is a derivable fact that moves with every op addition, and kept next to the op
 * name constants it is bound to go stale).
 */
export const GATHER_OP = "gather";
/** Semantic dtype conversion (attrs `to` is the target — ADR 0009). */
export const CAST_OP = "cast";

/**
 * Layout ops (ADR 0011).
 *
 * - `reshape` — does not change element order (the normal form of view / squeeze / unsqueeze).
 *   **The output's declared shape is the target shape**, and the contract is only that element
 *   counts match. Executed as a buffer alias (no copy).
 * - `permute` — reorders axes by attrs `dims`. A materialising copy.
 * - `expand` — **right-aligned broadcast to the output's declared shape**. Only dimensions of
 *   length 1 can be expanded (replicating length n → m is not in the vocabulary). A materialising
 *   copy.
 */
export const RESHAPE_OP = "reshape";
export const PERMUTE_OP = "permute";
export const EXPAND_OP = "expand";

/**
 * Layout ops, group 2 (ADR 0014). Only forms that satisfy the full-write invariant (every kernel
 * writes every byte of its output) enter the vocabulary.
 *
 * - `slice` — cuts out a **static axis and static range** given by attrs `dim` / `start` / `end`.
 *   Executed by reusing the existing strided **read** copy family; the only variable is one offset
 *   word in params (as ADR 0011 announced). Slicing a symbolic dimension is the job of
 *   {@link SYM_PREFIX_SLICE_OP}; this op is static only (no overlap — a slice on a symbolic axis is
 *   rejected by plan.ts from the declared shape).
 * - `cat` — concatenation along the **static axis** attrs `dim`. **The number of inputs is
 *   variable** (`variadic` of {@link OpContract}). Executed by the strided **write** copy family,
 *   writing each input into a subregion of the output. A single dispatch writes partially, but all
 *   inputs together cover the whole output, so full-write holds per node.
 * - `pad` — fills the **last dimension with constant 0** by attrs `left` / `right`. One dedicated
 *   kernel writes the whole output (it relies neither on a zero-initialisation guarantee nor on the
 *   noReuse exception).
 * - `flip` — reverses the indices of the static axis attrs `dim`. A dedicated tiny kernel.
 */
export const SLICE_OP = "slice";
export const CAT_OP = "cat";
export const PAD_OP = "pad";
export const FLIP_OP = "flip";

/**
 * Symbolic prefix slice (ADR 0010). From a constant that the exporter baked by **actually
 * evaluating, at Tmax,** a subtree that depends on the symbol T, cuts out the **leading** part of
 * length `coeff·sym+offset` after binding.
 *
 * - The input is a value with a **static shape without symbols** (in practice an initializer the
 *   exporter baked). This invariant is checked by plan.ts's `validateGraphContracts` — if the input
 *   side had a symbolic shape, the "Tmax form" premise would break and the read stride would change
 *   per run.
 * - The output shape is computed from attrs (+ bindings) and matched against the declaration
 *   (unlike reshape / expand, where "the declaration is the target shape").
 * - Executed as a strided materialising copy (offset 0, contiguous stride of the input side) — no
 *   new kernel.
 */
export const SYM_PREFIX_SLICE_OP = "sym_prefix_slice";

/**
 * Fused ops (ADR 0012 / ADR 0015 / ADR 0017). The "do not decompose" list that started with ADR
 * 0007 (**currently 12 ops** — the ledger's source of truth is docs/op-vocabulary.md, "12 ops whose
 * decomposition is forbidden (fusion kept)" under "Design findings carried over"; `leaky_relu` made
 * it 10, `rms_norm` 11 and `scaled_dot_product_attention` 12) all have kernels (`conv2d` showed up in
 * the measurements of Anima's VAE decoder — ADR 0017). `rms_norm` is the first that **preservation
 * alone cannot fully supply**: the hand-written decomposed form (Qwen3 / DiT) is composed into one
 * node by the exporter's folding pass (two supply routes — ADR 0016 / 0017). Only the 12th,
 * attention, is **not on the exporter's default preservation list** (per-target opt-in — the reason
 * is in op-vocabulary.md).
 *
 * MUST: acceptance constraints are expressed **by this contract table + clear diagnostics**, not by
 * the IR vocabulary (ADR 0007). Carving "only the measured forms pass" into the IR schema would make
 * every widening of a form an incompatible revision of the distribution.
 *
 * - `linear` — `x[…,in] × W[out,in] + b[out]`. **The bias is always present (arity fixed at 3)**,
 *   and the bias-less form in the IR (arity 2) fails loudly. For an upstream linear without bias,
 *   the exporter synthesises a zero bias and normalises to arity 3 (ADR 0016).
 * - `layer_norm` — attrs `normalized_shape` / `eps`. The affine part (weight / bias) is always
 *   present too.
 * - `softmax` — attrs `dim`. **Only the last dimension** is accepted (only −1 was measured — the same
 *   narrowing as gather).
 * - `safe_softmax` — the same contract as `softmax` + "a row whose max is −inf is all 0" (ADR 0044).
 * - `embedding` — row gather of `weight f32[V,H]` by `index i32[…]`. attrs `padding_idx`.
 * - `masked_fill` — attrs `value` (f32 scalar). The mask is bool with **right-aligned broadcast**.
 * - `conv1d` — attrs `stride` / `padding` / `dilation` / `groups` (the latter two added by ADR
 *   0015). The design value of "no field = fixed at 1" carries over as "a field present = required,
 *   no default filling, inconsistency fails loudly". The weight is `[Cout, Cin/groups, K]`.
 * - `conv_transpose1d` — attrs `stride` / `padding`. The weight is **`[Cin, Cout, K]`** (transposed
 *   relative to conv1d). Only forms whose output length is `L·stride` (`2·padding == K − stride`) are
 *   accepted, and the general form fails loudly (ADR 0015 — widened when there is demand).
 * - `conv2d` — attrs `stride` / `padding` / `dilation` / `groups` (the three spatial ones have **two
 *   components, H/W**). `x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout]` (ADR 0017).
 * - `rms_norm` — attrs `eps`, **arity 2** (x, weight). `y = x · rsqrt(mean(x², last dimension)
 *   + eps) · weight`. Unlike layer_norm it has no bias and does not subtract the mean (ADR 0017).
 */
export const LINEAR_OP = "linear";
export const LAYER_NORM_OP = "layer_norm";
/**
 * Fused attention (ADR 0023). `out = softmax_lastdim((q·scale) @ (k·scale)ᵀ + mask) @ v`.
 *
 * - **Arity 3 or 4** (q / k / v + an optional mask). A different mechanism from variable arity
 *   (`cat`), with an upper bound ({@link ContractBase.maxArity}) — not "any number" but "only one
 *   more input, the mask".
 * - Inputs are **rank-4 head-first** (`q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]`) and
 *   contiguous. The output is `[B,H,M,D]`. **D is the same for all three** (every measured attention
 *   is like that, and allowing a different length on the v side alone would let "IR with D mixed up"
 *   pass the shape check).
 * - **GQA / MQA is a divisible broadcast** (ADR 0067 decision 1). Only forms with `H % Hkv == 0`
 *   are accepted, and `r = H / Hkv` is a **derived value** (no attrs field). The exact B match, equal
 *   Hkv between k/v, equal D for all three and the N=0 rejection stay as lines that detect mix-ups.
 * - **The mask is f32, rank-4, with shape exactly `[1,1,M,N]`** (additive — `S' = S + mask`) and is
 *   broadcast to all B·H batches. `[B,1,M,N]` / `[1,H,M,N]` / bool / rank≠4 are **not accepted**
 *   (widened when a demand for run-time masks appears — the discipline where a missing field
 *   expresses "not in the vocabulary" structurally). causal / dropout are still not in the
 *   vocabulary.
 * - **The states form** (ADR 0067 decision 4 — a contract extension under the same op name, where
 *   **the presence of the field decides the form**): when the `states` field is `{ k, v }`, it
 *   becomes the autoregressive form that reads "the past from the slots, this step's k/v from
 *   `ins`". ins are `[B,Hkv,M,D]` (M = the physical chunk dimension shared with q), and **it takes no
 *   mask input** (causal is fixed and expressed by predicate computation). Sliding is the optional
 *   attrs `window` (no field = the whole context). Nodes **without** the field keep the previous
 *   contract exactly, and existing assets and existing checks do not move by a single byte.
 *
 * MUST: `scale` is **applied to both q and k (the half-scale contract)**. It is equivalent to
 * torch's `aten::_scaled_dot_product_attention_math` writing `q *= √scale_factor; k *=
 * √scale_factor`, which is why it is an unfamiliar value containing a `√`. **Do not change it to the
 * form applied once after the dot product (full scale)** — the rounding sequence would change
 * and bit identity with the decomposed path (mul → permute → bmm → softmax → expand → bmm) would be
 * lost (the core of ADR 0023's design — decision 2).
 */
export const ATTENTION_OP = "attention";
/**
 * A **single-purpose effect op** that writes this step's k / v into a state slot (ADR 0067
 * decision 5). `ins` is one `[B,Hkv,M,D]`, the `states` field is exactly `{ slot }`, and **it has 0
 * outputs** (it defines no value — the first op to use the zero-output case of node-level
 * multi-output, ADR 0068 decision 1).
 *
 * MUST: not built into attention (the TVM style is not adopted). Precisely because it stays a
 * single-purpose op, ① the full-write / padding-row no-op check (queryLength cuts it — ADR 0066
 * decision 4) is done in one op, ② the attention side stays read-only, which keeps bit-identity
 * checks simple, and ③ a KV-sharing layer is expressed simply by "having no `state_append` node"
 * (the same expressive power as ORT's kv_empty, obtained from the absence of an op).
 * MUST: `window` is **optional** (no field = the whole context). The declaration layer
 * (`validateGraphContracts`) checks that the value agrees across every node that touches the same
 * slot — the mapping from logical col to physical row MUST use the same formula for reads and writes
 * (ADR 0067 decision 4), and a different formula on the read side alone would cause silent misreads.
 * MUST: logical positions (pastLength / queryLength) are not carried in attrs. Run-time scalars that
 * change every step are carried by a context-owned mutable uniform (ADR 0066 addendum 4).
 */
export const STATE_APPEND_OP = "state_append";

export const RMS_NORM_OP = "rms_norm";
export const STATIC_QUANTIZE_OP = "static_quantize";
export const SOFTMAX_OP = "softmax";
/**
 * `softmax` + "**writes all 0 for a row whose max is −inf**" (ADR 0044). The contract, attrs and
 * shape rules are the same as `softmax`; this one rule is the only difference.
 *
 * It originates in the safe-softmax guard that torch's SDPA decomposition puts over `softmax`
 * (`where(¬any(¬eq(src,−inf)), 0, softmax(src))`); when the mask is a run-time value, a proof that
 * "the guard never fires" cannot be made in principle (the exporter's `_drop_safe_softmax_guard`).
 * With finite scores, "all elements −inf ⇔ row max = −inf", so this op has **exactly** the same
 * semantics as the guard.
 *
 * MUST: keep it a separate op from `softmax`. Adding an empty-row branch to plain `softmax` would
 * remove the existing narrowing that treats an all −inf row as a contract violation (ADR 0044
 * decision 3; for fused attention that rule was withdrawn by the ADR 0044 addendum of 2026-08-31,
 * which makes an all-masked row a valid input with output 0), and "a contract violation that should
 * surface as NaN" would silently become 0.
 */
export const SAFE_SOFTMAX_OP = "safe_softmax";
export const EMBEDDING_OP = "embedding";
export const MASKED_FILL_OP = "masked_fill";
export const CONV1D_OP = "conv1d";
export const CONV2D_OP = "conv2d";
/**
 * 3D convolution (extended composite layer — ADR 0118 decision 1). **unbatched**:
 * `x[Cin,T,H,W] * W[Cout,Cin/groups,Kt,Kh,Kw] + b[Cout] → [Cout,Tout,Hout,Wout]`.
 *
 * - attrs `stride` / `padding` / `dilation` have **3 components, `[T, H, W]`**; `groups` is a
 *   scalar. All four are required and not filled with defaults (the same discipline as conv2d's
 *   `[H, W]` — ADR 0015).
 * - **Arity fixed at 3** (a conv without bias is normalised by the exporter's zero-bias synthesis —
 *   as for the conv family).
 * - The output length per axis is `floor((L + 2·padding − dilation·(K−1) − 1) / stride) + 1`.
 *   Padding is **symmetric** zeros per axis. Causal padding (one side of time only) is not in the
 *   vocabulary — the recipe expresses it by cat-ing the previous chunk's cache onto the front of the
 *   time axis (ADR 0118 decision 2).
 *
 * MUST: x is **rank 4 (unbatched) only**. A batched rank 5 is not in the contract because the
 * surrounding cat / slice hit the strided family's rank limit of 4 (if demand appears, it is added
 * by discriminating on rank — the generalisation condition of ADR 0118 decision 1).
 * NOTE: the semantics cover general groups, but the GPU's implemented subset is only `groups == 1`
 * (implicit GEMM). `groups > 1` fails loudly when the Session is built (`validateGraphContracts` in
 * src/runtime/plan.ts).
 */
export const CONV3D_OP = "conv3d";
export const CONV_TRANSPOSE1D_OP = "conv_transpose1d";

/**
 * Modulated deformable convolution v2 (an extended atomic op — `torchvision::deform_conv2d`.
 * ADR 0055). **Arity fixed at 5**, taking `x[B,Cin,H,W]` / `weight[Cout,Cin,Kh,Kw]` /
 * `offset[B,2·Kh·Kw,Hout,Wout]` / `mask[B,Kh·Kw,Hout,Wout]` / `bias[Cout]`.
 *
 * - The channel order of `offset` nests `(kh, kw)` with pairs innermost, **even = y / odd = x**.
 * - `mask` (the modulator) is applied **after bilinear interpolation** (`(m · v) · w`).
 * - Out of bounds is **zero-filled** (if the centre lies outside `(−1, in)` the whole tap is 0;
 *   even inside, an out-of-range corner alone is 0 — not a border clamp).
 *
 * MUST: **no fields** for `stride` / `dilation` / `groups` / `offset_groups` (= fixed at 1). All the
 * measurements (20 sites across the BiRefNet family) are 1, and the absence of the field is itself
 * the statement "that form is not in the vocabulary" (the same move as `output_padding` of
 * conv_transpose1d — ADR 0023 decision 4).
 * MUST: `mask` is a required slot = **DCNv2 only**. Having no field that expresses `use_mask=False`
 * (DCNv1) is what makes it fail loudly.
 * MUST: the output spatial size is derived from x / weight / `padding`, and the two spatial axes of
 * `offset` / `mask` are **only matched against it** (taking the same fact from two places would let
 * a shape mismatch slip through).
 */
export const DEFORM_CONV2D_OP = "deform_conv2d";

/**
 * Bilinear resample over the two spatial axes of NCHW (a Core ATen layer op —
 * `aten.upsample_bilinear2d.vec`). `x[B,Cin,H,W] → [B,Cin,Hout,Wout]`; the only attr is
 * `output_size` (`[Hout, Wout]`).
 *
 * MUST: **`align_corners = True` only**, and there is no `align_corners` field in attrs (the
 * discipline where a missing field expresses "not in the vocabulary" structurally — ADR 0023
 * decision 4). `False` has a different coordinate formula (`scale·(i+0.5) − 0.5`) and different edge
 * handling, so accepting it would change the numbers under the same op name. If demand appears, it
 * is added as a **separate op**, the same move as `gelu` / `gelu_tanh`.
 * MUST: no `mode` field either (nearest / bicubic / area / antialias are not in the vocabulary). The
 * scale factor is **derived at run time** from `output_size` and the input — there is no field for
 * `scale_factor` either (it would give two IR spellings of the same form).
 * NOTE: downscaling (`Hout < H`) also goes through the same op and formula (torch uses the same op
 * too). Reading only 2 taps because there is no antialias is torch's specified behaviour, and
 * differs from `area`.
 */
export const UPSAMPLE_BILINEAR2D_OP = "upsample_bilinear2d";

/**
 * Group normalisation over channel groups of NCHW-like layouts (a Core ATen layer op —
 * `aten.native_group_norm`; the exporter takes the preserved `aten.group_norm` — ADR NNNN
 * decision 7). `x[N,C,…] → [N,C,…]` (rank ≥ 2, any trailing extent), with `weight[C]` /
 * `bias[C]` and attrs `num_groups` / `eps`:
 *
 * ```
 * group g of sample n = channels g·C/G … (g+1)·C/G − 1, every trailing element (contiguous)
 * out = (x − mean_ng) / sqrt(var_ng + eps) · weight[c] + bias[c]
 * ```
 *
 * MUST: **the population variance** (divide by the group's element count — the normalisation-layer
 * convention, as for layer_norm) and **two passes** (mean, then the sum of squared deviations — a
 * group holds millions of elements in a VAE at full resolution, where `E[x²] − E[x]²` cancels).
 * MUST: the affine is always present (**arity fixed at 3**, as layer_norm); the exporter synthesises
 * ones / zeros for a module without affine. `C` divisible by `num_groups` is a shape-rule check.
 */
export const GROUP_NORM_OP = "group_norm";

/**
 * The **hidden-side scan** of a GRU (extended composite layer — ADR 0056). Folds only the sequential
 * recurrence over time into one node and **has no input-side GEMM** (the caller prepares
 * `gi = x·W_ihᵀ + b_ih` with the existing `linear`).
 *
 * **Arity fixed at 4**, taking `gi[T,N,3H]` / `h0[N,H]` / `w_hh[3H,H]` / `b_hh[3H]`; the output is
 * `y[T,N,H]` (`h` for every step). The gate order of 3H is **r / z / n**.
 *
 * ```
 * r = sigmoid(gh_r + gi_r) / z = sigmoid(gh_z + gi_z) / n = tanh(gi_n + gh_n·r)
 * h' = (h − n)·z + n            (gh = W_hh·h + b_hh)
 * ```
 *
 * MUST: the scan direction is **split by op name** (`gru_scan` / `gru_scan_reverse`), not made a
 * bool attrs variant — there is no precedent for bool in attrs (Python's bool derives from int, so a
 * new validation mechanism would be needed), and it follows the same "an attr variant is a separate
 * op" move as `gelu` / `gelu_tanh` (ADR 0056 decision 2).
 * MUST: the reverse op also writes **its output in forward time order**. `flip` rejects symbolic axes
 * (ADR 0014 / 0046), so folding the scan direction into the op is the only form that lets a symbolic
 * T through.
 * MUST: no fields for multiple layers / bidirectional. Putting `aten.gru`'s `Tensor[16]` (the weight
 * list per layer × direction) on one node would make the op contain the layer loop and the `cat`
 * between layers — layers and directions are expressed by **lining up nodes** (ADR 0056 decision 7).
 * `h_n` is not returned — the GRU is expressed as a **single-output contract** with output `y` only;
 * returning `h_n` MAY be added as a revision of the op contract once a consumer that needs it
 * appears (re-graded 2026-10-18 — ADR 0056 decision 7). Node-level multiple outputs are allowed in
 * the IR (ADR 0068), but this contract has not been widened.
 * MUST: no fields for `has_biases=False` / `batch_first` / `dropout`. The absence of the field is
 * itself the statement "that form is not in the vocabulary" (ADR 0023 decision 4).
 */
export const GRU_SCAN_OPS = ["gru_scan", "gru_scan_reverse"] as const;
export type GruScanOpName = (typeof GRU_SCAN_OPS)[number];

/**
 * The **gated delta rule scan**, the recurrence of a Gated DeltaNet layer (extended composite layer
 * — ADR 0135; the value form, for prefill). Only the recurrence is folded into one node: the
 * projections, the causal conv, the gates, the q / k l2-normalisation and the gated norm stay
 * existing ops around it.
 *
 * **Arity fixed at 5, uniform f32, empty attrs**: `q[B,T,Hk,dk]` (l2-normalised, not scaled) /
 * `k[B,T,Hk,dk]` (l2-normalised) / `v[B,T,Hv,dv]` / `g[B,T,Hv]` (log decay) / `beta[B,T,Hv]` →
 * `o[B,T,Hv,dv]`. Per batch element, value head h (reading key head `⌊h/r⌋`, `r = Hv/Hk`) and
 * token t, from `S_0 = 0`:
 *
 * ```
 * S' = exp(g_t)·S_{t−1} / δ = β_t·(v_t − S'ᵀk_t) / S_t = S' + k_t δᵀ / o_t = S_tᵀ(q_t·dk^-½)
 * ```
 *
 * The layout is upstream's interface layout `[B,T,H,d]`; T is the symbolic length axis. The head
 * broadcast is **derived from the shapes** (`Hv % Hk == 0`, value head h reads key head `⌊h/r⌋` =
 * `repeat_interleave(r, dim=2)` — the same rule as attention's GQA, ADR 0067), and the q scale
 * `dk^-½` is derived from dk (no attr).
 * MUST: no fields for the initial state, the final state, the q / k l2-normalisation, the scale or
 * the head repeat — the absence of a field is the statement that the form is not in the vocabulary
 * (ADR 0135 decision 1). A carried state is the states form (decision 4, not in the vocabulary yet);
 * a variant becomes a separate op name.
 * NOTE: the semantics (shape rule, CPU reference) hold for any B, T ≥ 1, dk, dv and `Hk | Hv`; the
 * GPU's implemented subset is `dk = dv = 128` (`GATED_DELTA_SCAN_HEAD_DIM` in codegen/limits.ts),
 * and other head dims fail loudly when the Session is built (`validateGraphContracts` in
 * runtime/plan.ts — ADR 0064 axis A).
 */
export const GATED_DELTA_SCAN_OP = "gated_delta_scan";

/**
 * Weight slots for which low-precision storage is **eligible** (op name → input slot index — ADR
 * 0018).
 *
 * Only the 6 slots whose size dominates in measurements are listed (conv3d added by ADR 0118
 * decision 1 — the flattened weight is exactly the A tile of the implicit GEMM, so the decode path
 * is shared with conv1d / conv2d). If an initializer has even one consumer not listed here (bias /
 * norm-type weights / other ops), it is ineligible and decoded to f32 on the CPU at load time.
 *
 * MUST: bias is not included. In the prototype, the f32 constant of a bias dragged its weight down
 * with it and made the eligible f16 bytes 0 MB (the target ADR 0006 named for a root-cause fix) —
 * "bias is always f32" can be guaranteed only by "keeping bias slots out of the eligibility
 * verdict".
 */
export const WEIGHT_SLOTS: ReadonlyMap<string, number> = new Map([
  [LINEAR_OP, 1],
  [CONV1D_OP, 1],
  [CONV2D_OP, 1],
  [CONV3D_OP, 1],
  [CONV_TRANSPOSE1D_OP, 1],
  [EMBEDDING_OP, 0],
]);

/**
 * The **channel axis** of the per-channel scale (op name → axis index of the weight tensor — ADR
 * 0019).
 *
 * It is the output-channel axis: 0 for linear `[out,in]` / conv1d `[Cout,Cin/g,K]` / conv2d
 * `[Cout,Cin/g,Kh,Kw]` / conv3d `[Cout,Cin/g,Kt,Kh,Kw]` / embedding `[V,H]`, and **1 for
 * conv_transpose1d only, whose layout `[Cin,Cout,K]` is transposed**.
 *
 * MUST: keep the key set equal to {@link WEIGHT_SLOTS} (pinned by tests/ops_conformance_test.ts).
 * If only one of them grows, the i8 of a new weight slot silently runs as "a scale on axis 0".
 */
export const WEIGHT_CHANNEL_AXES: ReadonlyMap<string, number> = new Map([
  [LINEAR_OP, 0],
  [CONV1D_OP, 0],
  [CONV2D_OP, 0],
  [CONV3D_OP, 0],
  [CONV_TRANSPOSE1D_OP, 1],
  [EMBEDDING_OP, 0],
]);
