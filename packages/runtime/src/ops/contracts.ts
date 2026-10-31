import type { OpSupport, RuntimeSupport } from "./support.ts";
import { type IrDtype, type IrNode, SEMANTIC_DTYPES } from "../format/ir.ts";
import {
  assertFiniteAttr,
  ATTENTION_ATTRS,
  type AttrSchema,
  attrValue,
  AXIS_ATTRS,
  CAST_ATTRS,
  castTargetDtype,
  CLAMP_ATTRS,
  CLAMP_MIN_ATTRS,
  CONV1D_ATTRS,
  CONV2D_ATTRS,
  CONV3D_ATTRS,
  CONV_TRANSPOSE1D_ATTRS,
  CUMSUM_ATTRS,
  DEFORM_CONV2D_ATTRS,
  EMBEDDING_ATTRS,
  LAYER_NORM_ATTRS,
  LEAKY_RELU_ATTRS,
  MASKED_FILL_ATTRS,
  PAD_ATTRS,
  PERMUTE_ATTRS,
  REDUCE_ATTRS,
  RMS_NORM_ATTRS,
  SCALAR_COMPARE_ATTRS,
  SCALAR_PARAM_ATTRS,
  SLICE_ATTRS,
  SOFTMAX_ATTRS,
  STATE_READONLY_ATTRS,
  STATE_WINDOW_ATTRS,
  stateReadonly,
  STATIC_QUANTIZE_ATTRS,
  SYM_PREFIX_SLICE_ATTRS,
  TOPK_ATTRS,
  UPSAMPLE_BILINEAR2D_ATTRS,
} from "./attrs.ts";
import {
  ARGMAX_OP,
  ATTENTION_OP,
  BINARY_OPS,
  type BinaryOpName,
  BMM_OP,
  CAST_OP,
  CAT_OP,
  CONV1D_OP,
  CONV2D_OP,
  CONV3D_OP,
  CONV_TRANSPOSE1D_OP,
  CUMSUM_OP,
  DEFORM_CONV2D_OP,
  EMBEDDING_OP,
  EXPAND_OP,
  FLIP_OP,
  GATED_DELTA_SCAN_OP,
  GATHER_OP,
  GRU_SCAN_OPS,
  type GruScanOpName,
  LAYER_NORM_OP,
  LINEAR_OP,
  MASKED_FILL_OP,
  MATMUL_OP,
  OpContractError,
  PAD_OP,
  PERMUTE_OP,
  REDUCE_OPS,
  type ReduceOpName,
  RESHAPE_OP,
  RMS_NORM_OP,
  SAFE_SOFTMAX_OP,
  SLICE_OP,
  SOFTMAX_OP,
  STATE_APPEND_OP,
  STATIC_QUANTIZE_OP,
  SYM_PREFIX_SLICE_OP,
  TOPK_OP,
  UNARY_OPS,
  type UnaryOpName,
  UPSAMPLE_BILINEAR2D_OP,
  WHERE_OP,
} from "./names.ts";

/**
 * The input count of the readonly (reads only the past) states-form attention — **exactly one, q**
 * (ADR 0096 stage 2 §1.2). The contract layer ({@link assertNodeContract}) and the shape layer
 * (`computeOutputShape`) read **the same single constant**.
 */
export const READONLY_ATTENTION_ARITY = 1;

/**
 * The op kind. **Derived** from `kind` of {@link OpContract} (holding the enumeration twice would let
 * one copy stay stale when an op is added to the vocabulary, with the type check still green).
 */
export type OpKind = OpContract["kind"];

/**
 * The dtype contract of the input slots (an extension of ADR 0012).
 *
 * - `uniform` — every slot has the same accepted set, and the slots are also required to **share
 *   one type** (the output too). Elementwise / matmul / bmm / row reduce / layout ops use this.
 * - `perSlot` — each slot has its own accepted set, and no type agreement between slots is
 *   required. The output has **the type of slot 0 (the value side)**. gather (src=f32 / index=i32 →
 *   out=f32) is the first example.
 *
 * MUST: an op whose slots share one accepted set is not written as `perSlot`. `uniform` also carries
 * the rule that rejects mixes such as `mul(f32, i32)`, and collapsing it into perSlot would silently
 * remove that rejection.
 * MUST: the output dtype is derived **by passing slot 0's dtype through the mapping**
 * ({@link ContractBase.outputDtypes}). There is no field to choose the source slot per op — an op
 * whose value side is not first, like `where(cond, a, b)`, can also be expressed by the mapping
 * (bool → f32). Adding the field would give two degrees of freedom, "which slot to take from" and
 * "what to map to", that hold the same fact twice.
 */
export type SlotDtypes =
  | { readonly kind: "uniform"; readonly accept: readonly IrDtype[] }
  | { readonly kind: "perSlot"; readonly slots: readonly (readonly IrDtype[])[] };

type ContractBase = {
  /**
   * The **union** of the semantic dtypes accepted **for inputs**. A projection used by the
   * capability projection ({@link RUNTIME_SUPPORT}) and by the diagnostics of the enumerating check;
   * the source of truth for the per-slot contract is {@link slotDtypes}.
   *
   * MUST: derived from {@link slotDtypes} ({@link unionDtypes}), not written by hand — double
   * bookkeeping grows "executable per the support table, rejected by the contract check".
   */
  readonly dtypes: readonly IrDtype[];
  /** The source of truth for the per-slot accepted sets and for deriving outputs. */
  readonly slotDtypes: SlotDtypes;
  /**
   * The **list**, **per output slot**, of mappings "slot 0's input dtype → that output's dtype" (ADR
   * 0068 decision 1). The list length is the op's output count; only `topk` (2) and `state_append`
   * (**0** — an effect op that defines no value, ADR 0067 decision 5) have a length other than 1.
   * Each mapping defaults to identity (same type as the input); the exceptions are only the forms
   * that showed up in measurements (listed in {@link OUTPUT_DTYPES}): comparisons (f32 → bool) /
   * `sum` of bool input (→ an i32 count) / `where` (condition bool → value f32) / the index outputs
   * of `argmax` and `topk` (f32 → i32).
   *
   * MUST: the output count is derived from this list's length ({@link outputCountOf}) — a separate
   * output-count field would make it possible to write the disagreement "2 dtype columns but 1
   * declared output".
   * MUST: the domain of each mapping **equals** slot 0's accepted set ({@link slotContract} fills it
   * with identity). A partial mapping would leave a hole where a dtype that passed the slot check
   * has no defined output.
   * NOTE: only `cast` does not use this mapping (its output is attrs.to — {@link resolveNodeDtypes}).
   */
  readonly outputDtypes: readonly ReadonlyMap<IrDtype, IrDtype>[];
  readonly attrs: AttrSchema;
  /**
   * An op with a variable number of inputs (currently only `cat`). When true, `arity` of
   * {@link OpContract} is read as a **lower bound** ({@link assertArity}).
   *
   * MUST: this field is not set to "loosen the arity check". Only cat's input count is variable;
   * for other ops the count itself is the contract (such as the fixed arity 3 with an always-present
   * bias).
   * NOTE: the capability projection ({@link slotDtypesOf}) creates slots only up to the lower bound.
   * Under a uniform contract every slot has the same accepted set, so the conclusion agrees with the
   * enumerating check (`assertRuntimeSupport` in ops/support.ts), which looks at surplus slots via
   * the union.
   */
  readonly variadic?: true;
  /**
   * The upper bound of the input count for an op with **optional trailing inputs** (currently only
   * the mask of `attention`). When set, `arity` is the lower bound and this the upper bound, and
   * only counts in that closed interval are accepted.
   *
   * MUST: not used in place of `variadic`. `variadic` means "any number" (cat's concatenation);
   * this means "one fixed slot more" — collapsing it into an unbounded expression would let the
   * contract accept a form where the extra input is silently ignored (a slot the kernel does not
   * read).
   * NOTE: the capability projection ({@link slotDtypesOf}) creates slots **up to the upper bound**
   * (an optional slot also has a dtype contract — under uniform, the same accepted set as the rest).
   */
  readonly maxArity?: number;
  /**
   * **Optional attrs** (currently two: `window` (ADR 0067 decisions 4 / 5) and `readonly` (ADR 0096
   * stage 2 §1.2)). When declared they go through the range check; when absent, "the absence of the
   * field" itself carries meaning.
   *
   * MUST: not used to make required attrs ({@link ContractBase.attrs}) optional. "Declared attrs are
   * not filled with defaults" (ADR 0012) is invariant, and only **fields whose absence is itself a
   * different declaration** belong here (`window` absent = the whole context).
   * MUST: optional attrs are **for the states form only** ({@link assertNodeContract}). The sliding
   * window is part of the "logical col → physical row" mapping and means nothing for a node that does
   * not reference state — if it could be written on the previous form, there would be "an attr that
   * is accepted but read by nobody".
   */
  readonly optionalAttrs?: AttrSchema;
  /**
   * The **exact input count** of the states form that declares `attrs.readonly` (ADR 0096 stage 2
   * §1.2) (currently 1 = q only, for `attention`). Omitted = an op that cannot declare readonly.
   *
   * MUST: a **separate field** from the closed interval `arity` / `maxArity`. readonly is not "one
   * more optional trailing input" but the form where "this step's k/v **disappear structurally**";
   * expressing it as an interval would give an accepted set full of holes like `1..4`, and broken IR
   * with 2 ins would pass.
   */
  readonly readonlyArity?: number;
  /**
   * The contract of the `states` field (omitted = **an op that cannot have a states field**). `keys`
   * is **exactly** the key set when the field is non-empty, and `required` says whether the field
   * itself is required.
   *
   * `attention` is `{ keys: ["k","v"], required: false }` — the presence of the field distinguishes
   * the previous form from the states form (ADR 0067 decision 4). `state_append` is
   * `{ keys: ["slot"], required: true }`.
   */
  readonly states?: StateFieldContract;
};

/** {@link ContractBase.states} — the key set of the states field and whether it is required. */
export type StateFieldContract = {
  readonly keys: readonly string[];
  readonly required: boolean;
};

/**
 * Ties kind and op name together as a discriminated union, so that consumers (codegen selection, CPU
 * reference) need not write casts like `op as UnaryOpName` that cannot catch mix-ups.
 * `arity` is the number of inputs; the number of outputs is the list length of
 * {@link ContractBase.outputDtypes} (2 for `topk`, 0 for `state_append`, 1 for everything else).
 */
export type OpContract =
  | (ContractBase & { readonly kind: "unary"; readonly name: UnaryOpName; readonly arity: 1 })
  | (ContractBase & { readonly kind: "binary"; readonly name: BinaryOpName; readonly arity: 2 })
  | (ContractBase & { readonly kind: "where"; readonly name: typeof WHERE_OP; readonly arity: 3 })
  | (ContractBase & { readonly kind: "cumsum"; readonly name: typeof CUMSUM_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "matmul"; readonly name: "matmul"; readonly arity: 2 })
  | (ContractBase & { readonly kind: "bmm"; readonly name: typeof BMM_OP; readonly arity: 2 })
  | (ContractBase & {
      readonly kind: "gather";
      readonly name: typeof GATHER_OP;
      readonly arity: 2;
    })
  | (ContractBase & { readonly kind: "rowReduce"; readonly name: ReduceOpName; readonly arity: 1 })
  // argmax over the last dimension (ADR 0068 decision 2). The kind is separate from the reduce family
  // because all four aspects differ — attrs (none), output dtype (i32), rank handling (preserved) and
  // kernel family (identity −inf + index tracking) — and so that the consumers' exhaustive switches
  // must declare "which semantics to execute" (the same reason safe_softmax is a separate kind from
  // softmax).
  | (ContractBase & { readonly kind: "argmax"; readonly name: typeof ARGMAX_OP; readonly arity: 1 })
  // top-k over the last dimension (ADR 0068 decision 3). **The only multi-output op** (values f32 +
  // indices i32); the output count is derived from the list length of the output dtype mapping
  // ({@link outputCountOf}). The kind is separate from argmax because the output count, attrs (`k`
  // required) and shape rule (last dimension → k) all differ.
  | (ContractBase & { readonly kind: "topk"; readonly name: typeof TOPK_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "cast"; readonly name: typeof CAST_OP; readonly arity: 1 })
  | (ContractBase & {
      readonly kind: "reshape";
      readonly name: typeof RESHAPE_OP;
      readonly arity: 1;
    })
  | (ContractBase & {
      readonly kind: "permute";
      readonly name: typeof PERMUTE_OP;
      readonly arity: 1;
    })
  | (ContractBase & {
      readonly kind: "expand";
      readonly name: typeof EXPAND_OP;
      readonly arity: 1;
    })
  | (ContractBase & { readonly kind: "slice"; readonly name: typeof SLICE_OP; readonly arity: 1 })
  // The only variable-arity op. `arity` is the **lower bound** 2 (a cat of one input is an identity
  // copy and does not show up in measurements either — having no representation for it is itself
  // the fail-loudly).
  | (ContractBase & {
      readonly kind: "cat";
      readonly name: typeof CAT_OP;
      readonly arity: 2;
      readonly variadic: true;
    })
  | (ContractBase & { readonly kind: "pad"; readonly name: typeof PAD_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "flip"; readonly name: typeof FLIP_OP; readonly arity: 1 })
  | (ContractBase & {
      readonly kind: "symPrefixSlice";
      readonly name: typeof SYM_PREFIX_SLICE_OP;
      readonly arity: 1;
    })
  | (ContractBase & { readonly kind: "linear"; readonly name: typeof LINEAR_OP; readonly arity: 3 })
  | (ContractBase & {
      readonly kind: "layerNorm";
      readonly name: typeof LAYER_NORM_OP;
      readonly arity: 3;
    })
  // The affine part is the weight only, so arity 2 (no bias field — ADR 0017).
  | (ContractBase & {
      readonly kind: "rmsNorm";
      readonly name: typeof RMS_NORM_OP;
      readonly arity: 2;
    })
  | (ContractBase & {
      readonly kind: "staticQuantize";
      readonly name: typeof STATIC_QUANTIZE_OP;
      readonly arity: 1;
    })
  | (ContractBase & {
      readonly kind: "softmax";
      readonly name: typeof SOFTMAX_OP;
      readonly arity: 1;
    })
  // A separate op with the same contract as softmax (empty row → all 0. ADR 0044). The kind is
  // separate so that the exhaustive switches of the executor / CPU reference must declare "which
  // semantics to execute".
  | (ContractBase & {
      readonly kind: "safeSoftmax";
      readonly name: typeof SAFE_SOFTMAX_OP;
      readonly arity: 1;
    })
  // Fused attention (ADR 0023). Arity 3–4 (the 4th is the optional additive mask), attrs `scale`
  // required, rank-4 head-first.
  | (ContractBase & {
      readonly kind: "attention";
      readonly name: typeof ATTENTION_OP;
      readonly arity: 3;
      readonly maxArity: 4;
      readonly readonlyArity: typeof READONLY_ATTENTION_ARITY;
    })
  // The effect op that writes this step's k/v into a state slot (ADR 0067 decision 5). The first op
  // with **0 outputs**, and the `states` field is required (exactly `{ slot }`). The kind is separate
  // from attention because the output count, the states field keys and the kernel family (ring
  // writes) all differ.
  | (ContractBase & {
      readonly kind: "stateAppend";
      readonly name: typeof STATE_APPEND_OP;
      readonly arity: 1;
    })
  | (ContractBase & {
      readonly kind: "embedding";
      readonly name: typeof EMBEDDING_OP;
      readonly arity: 2;
    })
  | (ContractBase & {
      readonly kind: "maskedFill";
      readonly name: typeof MASKED_FILL_OP;
      readonly arity: 2;
    })
  | (ContractBase & {
      readonly kind: "conv1d";
      readonly name: typeof CONV1D_OP;
      readonly arity: 3;
    })
  | (ContractBase & {
      readonly kind: "conv2d";
      readonly name: typeof CONV2D_OP;
      readonly arity: 3;
    })
  // Unbatched 3D convolution (extended composite layer — ADR 0118 decision 1). Fixed at 3 inputs:
  // x / weight / bias.
  | (ContractBase & {
      readonly kind: "conv3d";
      readonly name: typeof CONV3D_OP;
      readonly arity: 3;
    })
  | (ContractBase & {
      readonly kind: "convTranspose1d";
      readonly name: typeof CONV_TRANSPOSE1D_OP;
      readonly arity: 3;
    })
  // DCNv2 (ADR 0055). Fixed at 5 inputs: x / weight / offset / mask / bias — mask being a required
  // slot expresses structurally that "DCNv1 is not in the vocabulary".
  | (ContractBase & {
      readonly kind: "deformConv2d";
      readonly name: typeof DEFORM_CONV2D_OP;
      readonly arity: 5;
    })
  // Bilinear resample (Core ATen layer). The only input is x (no weight, no bias), and the output
  // spatial size is declared by attrs `output_size`.
  | (ContractBase & {
      readonly kind: "upsampleBilinear2d";
      readonly name: typeof UPSAMPLE_BILINEAR2D_OP;
      readonly arity: 1;
    })
  // The hidden-side GRU scan (ADR 0056). One kind holds the 2 ops that differ only in scan direction
  // (as for the reduce family and gelu / gelu_tanh — the contract is fully identical, and consumers
  // look up the direction by `name`).
  | (ContractBase & {
      readonly kind: "gruScan";
      readonly name: GruScanOpName;
      readonly arity: 4;
    })
  // The gated delta rule scan (ADR 0135, the value form). Fixed at 5 inputs: q / k / v / g / beta.
  | (ContractBase & {
      readonly kind: "gatedDeltaScan";
      readonly name: typeof GATED_DELTA_SCAN_OP;
      readonly arity: 5;
    });

/** f32-only ops (no i32 / bool form appeared in measured graphs — not unlocked for symmetry). */
const F32: readonly IrDtype[] = ["f32"];
/**
 * Ops that can execute both f32 and i32. Grounds: the DeBERTa front measurements (recon §3-8): the
 * mask outer product `mul(i64, i64)` and the `sub` of `1 - attention_mask`. Both depend on input
 * values and cannot be constant-folded, so the executor needs them.
 */
const F32_I32: readonly IrDtype[] = ["f32", "i32"];
/** bool only (negation and logical AND mean something only for booleans). */
const BOOL: readonly IrDtype[] = ["bool"];
/**
 * Ops that can execute both f32 and bool. Grounds: sdp's spline (the searchsorted row of recon §2):
 * `sum(inputs[…,None] >= bl, dim=-1)` is **a row sum of bool** (= the number of trues), and the
 * output is i32 ({@link OUTPUT_DTYPES}). It coexists with the f32 sum.
 */
const F32_BOOL: readonly IrDtype[] = ["f32", "bool"];
/** Integer indices only (gather's index slot — measured as i32[16,T,T]). */
const I32: readonly IrDtype[] = ["i32"];
/** cast's **input** side (the output is attrs.to), and reshape, which touches no element value. */
const ANY_DTYPE: readonly IrDtype[] = SEMANTIC_DTYPES;

const NO_ATTRS: AttrSchema = {};

/** The per-op dtype unlock table. Ops not listed stay f32. */
const DTYPES: ReadonlyMap<string, readonly IrDtype[]> = new Map([
  ["mul", F32_I32],
  ["sub", F32_I32],
  ["bitwise_not", BOOL],
  // spline's `inside = (x >= -b) & (x <= b)` (recon §2) — a logical AND of bools. Integer
  // bitwise_and means something else, so it is not unlocked (the same narrowing as bitwise_not).
  ["bitwise_and", BOOL],
  ["sum", F32_BOOL],
  // reshape reads and writes no element (aliasing only), so the whole vocabulary. The measurements
  // also show f32 views, squeeze/unsqueeze of i32 masks and unsqueeze of bool masks (recon §2).
  [RESHAPE_OP, ANY_DTYPE],
  // Besides gather indices (i32) and the bool masks of the conv path, f32 is also unlocked for the
  // 4D lift of the relative position embeddings `ek.expand(...)` / `ev.expand(...)` (f32 — the
  // enc_p / flow rows of recon §2). The strided copy family is dtype-parametric, so the kernel stays
  // shared.
  [EXPAND_OP, ANY_DTYPE],
  // The baked constants come in two kinds: the relative position bucket table (i32) and the position
  // table (f32). bool initializers are not in the vocabulary (INITIALIZER_DTYPES in ir.ts), so bool
  // is not unlocked.
  [SYM_PREFIX_SLICE_OP, F32_I32],
]);

const dtypesOf = (name: string): readonly IrDtype[] => DTYPES.get(name) ?? F32;

/**
 * The table that declares the **list** of per-output-slot dtype mappings. Ops not listed have "one
 * output, identity (same type as the input)" ({@link SINGLE_IDENTITY_OUTPUT}). Each list element is
 * "slot 0's input dtype → that output's dtype", and identity dtypes need not be written
 * ({@link slotContract} fills them in).
 *
 * MUST: the mapping ranges cover only the forms that showed up in measurements (ADR 0009's "per op,
 * based on measurements").
 * - The 4 comparisons — spline's `inside` test and searchsorted (recon §2). Booleans, so bool.
 * - `sum` with bool input — `sum(x >= bl, dim=-1)` is **the number of trues**, so i32 (counting in
 *   f32 silently rounds once past 2^24). f32 input stays f32 as before.
 * - `where` — the condition is the first slot, so only the mapping can express "the output is the
 *   value side".
 * - `argmax` — the output is **an index**, so i32 (ADR 0068 decision 2). Returned as f32 it would
 *   round to the neighbouring token once the vocabulary exceeds 2^24 (the same reason as `sum` of
 *   bool).
 * - `topk` — **the list length is 2** (ADR 0068 decision 3). slot 0 is the values, so identity
 *   (f32 → f32, filled in by the empty mapping); slot 1 is the indices, so i32. **The list length is
 *   the output count as is**, so this one row is itself the declaration "topk outputs 2".
 * - `state_append` — **the list is empty** (ADR 0067 decision 5). An effect op that defines no
 *   value; this one row is itself the declaration "0 outputs" (`outs: []` is allowed only for ops
 *   whose contract declares this).
 */
const OUTPUT_DTYPES: ReadonlyMap<string, readonly ReadonlyMap<IrDtype, IrDtype>[]> = new Map([
  ["ge", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  ["ge_scalar", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  ["le_scalar", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  ["gt_scalar", [new Map<IrDtype, IrDtype>([["f32", "bool"]])]],
  [
    "sum",
    [
      new Map<IrDtype, IrDtype>([
        ["f32", "f32"],
        ["bool", "i32"],
      ]),
    ],
  ],
  [WHERE_OP, [new Map<IrDtype, IrDtype>([["bool", "f32"]])]],
  [ARGMAX_OP, [new Map<IrDtype, IrDtype>([["f32", "i32"]])]],
  [TOPK_OP, [new Map<IrDtype, IrDtype>(), new Map<IrDtype, IrDtype>([["f32", "i32"]])]],
  [STATE_APPEND_OP, []],
]);

/** The default for ops without a declaration = **one output, identity** (empty map = all identity). */
const SINGLE_IDENTITY_OUTPUT: readonly ReadonlyMap<IrDtype, IrDtype>[] = [
  new Map<IrDtype, IrDtype>(),
];

/**
 * Reads the scalar attrs in params order (assumes {@link assertNodeContract} has already checked
 * them, but some paths call this directly from the CPU reference, so the ranges are checked here
 * too).
 *
 * MUST: clamp's `min <= max` can only be checked here (the attrs schema checks per key, so it cannot
 * express an invariant spanning two keys). Allowing an inversion would give "a clamp where only one
 * side takes effect", depending on the branch order on the WGSL side, and the difference from the
 * CPU reference would appear or not depending on the input's position.
 */
export const scalarParamValues = (
  found: OpContract,
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): readonly number[] => {
  const keys = SCALAR_PARAM_ATTRS.get(found.name);
  if (keys === undefined) return [];
  const values = keys.map((key) =>
    assertFiniteAttr(attrValue(attrs, key), `${where} attrs.${key}`, `op '${found.name}' ${key}`),
  );
  if (found.name === "clamp" && values[0] > values[1]) {
    throw new OpContractError(`${where}: clamp min ${values[0]} exceeds max ${values[1]}`);
  }
  return values;
};

/**
 * Folds the per-slot accepted sets into one union. The order is fixed to the vocabulary's
 * declaration order ({@link SEMANTIC_DTYPES}) — it is a projection used for display and matching, so
 * the order must not change with the slot order.
 */
const unionDtypes = (slotDtypes: SlotDtypes): readonly IrDtype[] =>
  slotDtypes.kind === "uniform"
    ? slotDtypes.accept
    : SEMANTIC_DTYPES.filter((dtype) => slotDtypes.slots.some((accept) => accept.includes(dtype)));

const contract = (name: string, attrs: AttrSchema = NO_ATTRS): ContractBase =>
  slotContract(name, { kind: "uniform", accept: dtypesOf(name) }, attrs);

/** Slot 0's accepted set (every slot's under uniform) — the domain of the output dtype mapping. */
const firstSlotAccept = (slotDtypes: SlotDtypes): readonly IrDtype[] =>
  slotDtypes.kind === "uniform" ? slotDtypes.accept : slotDtypes.slots[0];

const slotContract = (
  name: string,
  slotDtypes: SlotDtypes,
  attrs: AttrSchema = NO_ATTRS,
): ContractBase => {
  const domain = firstSlotAccept(slotDtypes);
  // dtypes without a declaration are filled with the identity mapping (the invariant "domain = slot
  // 0's accepted set" is met structurally per output slot — forgetting a row in the table does not
  // become "a dtype whose output is undefined").
  const outputDtypes = (OUTPUT_DTYPES.get(name) ?? SINGLE_IDENTITY_OUTPUT).map(
    (slot) => new Map<IrDtype, IrDtype>(domain.map((dtype) => [dtype, slot.get(dtype) ?? dtype])),
  );
  return { dtypes: unionDtypes(slotDtypes), slotDtypes, outputDtypes, attrs };
};

/** Schemas of unary ops with attrs (ops not listed have empty attrs). */
const UNARY_ATTRS: ReadonlyMap<string, AttrSchema> = new Map([
  ["clamp", CLAMP_ATTRS],
  ["clamp_min", CLAMP_MIN_ATTRS],
  ["leaky_relu", LEAKY_RELU_ATTRS],
  ["ge_scalar", SCALAR_COMPARE_ATTRS],
  ["le_scalar", SCALAR_COMPARE_ATTRS],
  ["gt_scalar", SCALAR_COMPARE_ATTRS],
]);

export const OP_CONTRACTS: ReadonlyMap<string, OpContract> = new Map<string, OpContract>([
  ...UNARY_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name, UNARY_ATTRS.get(name) ?? NO_ATTRS), kind: "unary", name, arity: 1 },
  ]),
  ...BINARY_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name), kind: "binary", name, arity: 2 },
  ]),
  ...REDUCE_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name, REDUCE_ATTRS), kind: "rowReduce", name, arity: 1 },
  ]),
  // A per-slot contract of condition bool and value f32. The output is **the value side** (mapping
  // bool → f32 — the discipline of having no field that points at the source slot).
  [
    WHERE_OP,
    {
      ...slotContract(WHERE_OP, { kind: "perSlot", slots: [BOOL, F32, F32] }),
      kind: "where",
      name: WHERE_OP,
      arity: 3,
    },
  ],
  [
    CUMSUM_OP,
    {
      ...contract(CUMSUM_OP, CUMSUM_ATTRS),
      kind: "cumsum",
      name: CUMSUM_OP,
      arity: 1,
    },
  ],
  [MATMUL_OP, { ...contract(MATMUL_OP), kind: "matmul", name: MATMUL_OP, arity: 2 }],
  [BMM_OP, { ...contract(BMM_OP), kind: "bmm", name: BMM_OP, arity: 2 }],
  // The first per-slot dtype contract: value f32 and index i32 mixed; the output has the value
  // side's type.
  [
    GATHER_OP,
    {
      ...slotContract(GATHER_OP, { kind: "perSlot", slots: [F32, I32] }),
      kind: "gather",
      name: GATHER_OP,
      arity: 2,
    },
  ],
  // argmax over the last dimension (ADR 0068 decision 2). Empty attrs, input f32 → output i32 (the
  // mapping is in OUTPUT_DTYPES); the shape rule holds the rank preservation.
  [ARGMAX_OP, { ...contract(ARGMAX_OP), kind: "argmax", name: ARGMAX_OP, arity: 1 }],
  // top-k over the last dimension (ADR 0068 decision 3). attrs `k` required, 2 outputs (values f32 +
  // indices i32 — the mapping is in OUTPUT_DTYPES); the shape computation holds the `[…, k]` shape
  // rule and the accepted domain.
  [TOPK_OP, { ...contract(TOPK_OP, TOPK_ATTRS), kind: "topk", name: TOPK_OP, arity: 1 }],
  [
    CAST_OP,
    {
      ...slotContract(CAST_OP, { kind: "uniform", accept: ANY_DTYPE }, CAST_ATTRS),
      kind: "cast",
      name: CAST_OP,
      arity: 1,
    },
  ],
  [RESHAPE_OP, { ...contract(RESHAPE_OP), kind: "reshape", name: RESHAPE_OP, arity: 1 }],
  [
    PERMUTE_OP,
    {
      ...contract(PERMUTE_OP, PERMUTE_ATTRS),
      kind: "permute",
      name: PERMUTE_OP,
      arity: 1,
    },
  ],
  [EXPAND_OP, { ...contract(EXPAND_OP), kind: "expand", name: EXPAND_OP, arity: 1 }],
  // Layout group 2 (ADR 0014). All measurements are f32 (the m_p/logs_p split of enc_p, the 96/96
  // split and cat of coupling, the pad on the relative-position value side, the Flip of flow/sdp —
  // recon §2), so the dtype is not in the unlock table = f32 only.
  [SLICE_OP, { ...contract(SLICE_OP, SLICE_ATTRS), kind: "slice", name: SLICE_OP, arity: 1 }],
  [
    CAT_OP,
    {
      ...contract(CAT_OP, AXIS_ATTRS),
      kind: "cat",
      name: CAT_OP,
      arity: 2,
      variadic: true,
    },
  ],
  [PAD_OP, { ...contract(PAD_OP, PAD_ATTRS), kind: "pad", name: PAD_OP, arity: 1 }],
  [FLIP_OP, { ...contract(FLIP_OP, AXIS_ATTRS), kind: "flip", name: FLIP_OP, arity: 1 }],
  [
    SYM_PREFIX_SLICE_OP,
    {
      ...contract(SYM_PREFIX_SLICE_OP, SYM_PREFIX_SLICE_ATTRS),
      kind: "symPrefixSlice",
      name: SYM_PREFIX_SLICE_OP,
      arity: 1,
    },
  ],
  // Fused ops (ADR 0012). The 3 with bias / affine have arity fixed at 3 — the measurements always
  // have a bias, and having no field that expresses "no bias" is itself the fail-loudly.
  [LINEAR_OP, { ...contract(LINEAR_OP), kind: "linear", name: LINEAR_OP, arity: 3 }],
  [
    LAYER_NORM_OP,
    {
      ...contract(LAYER_NORM_OP, LAYER_NORM_ATTRS),
      kind: "layerNorm",
      name: LAYER_NORM_OP,
      arity: 3,
    },
  ],
  // No bias, so arity 2 (ADR 0017). A form without weight is normalised to arity 2 by the exporter
  // synthesising ones — the same move as the zero-bias synthesis (ADR 0015).
  [
    RMS_NORM_OP,
    {
      ...contract(RMS_NORM_OP, RMS_NORM_ATTRS),
      kind: "rmsNorm",
      name: RMS_NORM_OP,
      arity: 2,
    },
  ],
  [
    STATIC_QUANTIZE_OP,
    {
      ...contract(STATIC_QUANTIZE_OP, STATIC_QUANTIZE_ATTRS),
      kind: "staticQuantize",
      name: STATIC_QUANTIZE_OP,
      arity: 1,
    },
  ],
  [
    SOFTMAX_OP,
    {
      ...contract(SOFTMAX_OP, SOFTMAX_ATTRS),
      kind: "softmax",
      name: SOFTMAX_OP,
      arity: 1,
    },
  ],
  // ADR 0044. The attrs schema **shares the same single object** with softmax (a copy would make it
  // possible for only one of them to loosen its narrowing).
  [
    SAFE_SOFTMAX_OP,
    {
      ...contract(SAFE_SOFTMAX_OP, SOFTMAX_ATTRS),
      kind: "safeSoftmax",
      name: SAFE_SOFTMAX_OP,
      arity: 1,
    },
  ],
  // Fused attention (ADR 0023). q / k / v and the optional mask, all 4 f32 of one type (uniform
  // contract). The mask is additive, so it has the value side's dtype, and bool is not accepted.
  // The states field (ADR 0067 decision 4) is **optional** — its presence distinguishes the previous
  // form from the states form, and only the form with the field can declare `window` (the accepted
  // set of the previous form does not move by a single byte).
  [
    ATTENTION_OP,
    {
      ...contract(ATTENTION_OP, ATTENTION_ATTRS),
      kind: "attention",
      name: ATTENTION_OP,
      arity: 3,
      maxArity: 4,
      // `window` and `readonly` are both optional attrs for the states form only (ADR 0067 decision
      // 4 / ADR 0096 stage 2 §1.2). `state_append` shares only `window` (append has no readonly), so
      // they are combined here and STATE_WINDOW_ATTRS itself stays unchanged.
      optionalAttrs: { ...STATE_WINDOW_ATTRS, ...STATE_READONLY_ATTRS },
      readonlyArity: READONLY_ATTENTION_ARITY,
      states: { keys: ["k", "v"], required: false },
    },
  ],
  // Writes into a state slot (ADR 0067 decision 5). One input, **0 outputs** (the empty list in
  // OUTPUT_DTYPES is that declaration), states field required.
  [
    STATE_APPEND_OP,
    {
      ...contract(STATE_APPEND_OP),
      kind: "stateAppend",
      name: STATE_APPEND_OP,
      arity: 1,
      optionalAttrs: STATE_WINDOW_ATTRS,
      states: { keys: ["slot"], required: true },
    },
  ],
  // A per-slot contract of value f32 and index i32 (the same kind as gather — the output has the
  // value side's type).
  [
    EMBEDDING_OP,
    {
      ...slotContract(EMBEDDING_OP, { kind: "perSlot", slots: [F32, I32] }, EMBEDDING_ATTRS),
      kind: "embedding",
      name: EMBEDDING_OP,
      arity: 2,
    },
  ],
  // A per-slot contract of value f32 and condition bool. The output has the type of slot 0 (the
  // value side).
  [
    MASKED_FILL_OP,
    {
      ...slotContract(MASKED_FILL_OP, { kind: "perSlot", slots: [F32, BOOL] }, MASKED_FILL_ATTRS),
      kind: "maskedFill",
      name: MASKED_FILL_OP,
      arity: 2,
    },
  ],
  [
    CONV1D_OP,
    {
      ...contract(CONV1D_OP, CONV1D_ATTRS),
      kind: "conv1d",
      name: CONV1D_OP,
      arity: 3,
    },
  ],
  [
    CONV2D_OP,
    {
      ...contract(CONV2D_OP, CONV2D_ATTRS),
      kind: "conv2d",
      name: CONV2D_OP,
      arity: 3,
    },
  ],
  // Unbatched 3D convolution (x is rank 4 — ADR 0118 decision 1). All 3 slots are f32 of one type.
  [
    CONV3D_OP,
    {
      ...contract(CONV3D_OP, CONV3D_ATTRS),
      kind: "conv3d",
      name: CONV3D_OP,
      arity: 3,
    },
  ],
  // A conv without bias is normalised to arity 3 by the exporter's zero-bias synthesis (ADR 0015) —
  // no arity branch is brought into the kernel or the contract.
  [
    CONV_TRANSPOSE1D_OP,
    {
      ...contract(CONV_TRANSPOSE1D_OP, CONV_TRANSPOSE1D_ATTRS),
      kind: "convTranspose1d",
      name: CONV_TRANSPOSE1D_OP,
      arity: 3,
    },
  ],
  // DCNv2 (extended atomic layer, ADR 0055). All 5 slots are f32 of one type (offset / mask also
  // have the value side's dtype). It has a weight but is **ineligible for low-precision storage**,
  // so it is not in WEIGHT_SLOTS.
  [
    DEFORM_CONV2D_OP,
    {
      ...contract(DEFORM_CONV2D_OP, DEFORM_CONV2D_ATTRS),
      kind: "deformConv2d",
      name: DEFORM_CONV2D_OP,
      arity: 5,
    },
  ],
  // Bilinear resample (Core ATen layer). It has no weight, so it is not in WEIGHT_SLOTS.
  [
    UPSAMPLE_BILINEAR2D_OP,
    {
      ...contract(UPSAMPLE_BILINEAR2D_OP, UPSAMPLE_BILINEAR2D_ATTRS),
      kind: "upsampleBilinear2d",
      name: UPSAMPLE_BILINEAR2D_OP,
      arity: 1,
    },
  ],
  // The two directions of the hidden-side GRU scan (extended composite layer, ADR 0056). All 4
  // slots are f32 of one type, empty attrs. `w_hh` is an in-op slot, so it is **ineligible for
  // low-precision storage** (not in WEIGHT_SLOTS) — the input-side weight is held by the caller's
  // `linear`, so f16 / i8 still apply there as before.
  ...GRU_SCAN_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name), kind: "gruScan", name, arity: 4 },
  ]),
  // The gated delta rule scan (extended composite layer, ADR 0135 — the value form). All 5 slots
  // are f32 of one type (upstream computes g and the state in f32 and casts q / k / v / β to f32
  // before the recurrence), empty attrs (the q scale and the head broadcast are derived from the
  // shapes). It has no weight slot.
  [
    GATED_DELTA_SCAN_OP,
    {
      ...contract(GATED_DELTA_SCAN_OP),
      kind: "gatedDeltaScan",
      name: GATED_DELTA_SCAN_OP,
      arity: 5,
    },
  ],
]);

/** The attrs keys the contract declares as **required** (fields every node must have). */
export const attrKeysOf = (found: OpContract): readonly string[] => Object.keys(found.attrs);

/**
 * The attrs keys the contract declares as **optional** ({@link ContractBase.optionalAttrs}).
 *
 * MUST: the capability projection ({@link RUNTIME_SUPPORT}) carries the **union** of required and
 * optional keys — the enumerating check (`assertRuntimeSupport` in ops/support.ts) lists the
 * difference from "the implemented attr keys" as "unimplemented attrs", so dropping the optional
 * ones would reject correct states-form graphs as missing capability.
 */
export const optionalAttrKeysOf = (found: OpContract): readonly string[] =>
  found.optionalAttrs === undefined ? [] : Object.keys(found.optionalAttrs);

/**
 * The **number of outputs** the contract declares. It is the list length of the output dtype
 * mapping as is (ADR 0068 decision 1).
 *
 * MUST: the count is not held in a separate field — holding a derivable fact twice would let a
 * contract that grew only its dtype list pass as "1 declared, 2 mapped", and nobody would check the
 * second output's dtype.
 */
export const outputCountOf = (found: OpContract): number => found.outputDtypes.length;

/**
 * The per-input-slot accepted sets (capability projection). A uniform contract is **replicated for
 * as many slots as it can accept** (including optional trailing slots —
 * {@link ContractBase.maxArity}), so that consumers (the enumerating check) can look them up by slot
 * index.
 */
export const slotDtypesOf = (found: OpContract): readonly (readonly IrDtype[])[] => {
  const slots = found.slotDtypes;
  return slots.kind === "uniform"
    ? Array.from({ length: found.maxArity ?? found.arity }, () => slots.accept)
    : slots.slots;
};

/**
 * The semantic dtypes that can be transferred as graph inputs. Every element type is 4 bytes, and
 * bool is carried as u32 0/1 (ADR 0009). A separate axis from the op dtype sets — the executor also
 * transfers inputs that no node consumes.
 */
export const IO_DTYPES: readonly IrDtype[] = SEMANTIC_DTYPES;

/**
 * The support table passed to `assertRuntimeSupport`. MUST be derived from the contract table only
 * — a hand-written set drifts from the table and produces "supported per the declaration, fails at
 * execution". It carries not only op names but semantic dtypes and attrs too (the difference that
 * name-only matching misses is the real bug of recon §3-9).
 */
export const RUNTIME_SUPPORT: RuntimeSupport = {
  ops: new Map<string, OpSupport>(
    [...OP_CONTRACTS].map(([name, found]) => [
      name,
      {
        dtypes: new Set(found.dtypes),
        slotDtypes: slotDtypesOf(found).map((accept) => new Set(accept)),
        // cast's output is decided by attrs.to (the mapping is filled with identity, but its range
        // is too narrow), so the enumerating check gets the whole vocabulary. Otherwise it is
        // exactly the range of each output slot's mapping.
        outDtypes:
          found.kind === "cast"
            ? [new Set(SEMANTIC_DTYPES)]
            : found.outputDtypes.map((slot) => new Set(slot.values())),
        attrKeys: new Set([...attrKeysOf(found), ...optionalAttrKeysOf(found)]),
      },
    ]),
  ),
  // Raw int32 storage (ADR 0010) is executed as the target of baked symbol-dependent constants. f16
  // (ADR 0018), i8 (ADR 0019), i4 (ADR 0069) and i2 (ADR 0097 addendum 1) have execution paths
  // (eligible weight slots stay compressed and GPU-resident; ineligible ones are decoded to f32 on
  // the CPU at load time), so **they can be executed whatever initializer they are attached to**.
  // Only i4 eligibility is narrower (the weight slots of linear / embedding / conv1d with groups ==
  // 1 — ADR 0069 decision 5 and its addenda, `i4EligibleInitializers` in runtime/plan.ts; ineligible
  // ones fall back to CPU decode). i2 eligibility is the weight slots of linear / embedding. bf16
  // (ADR 0131) uses the same path as f16 (eligible: the same 6 slots as f16, `WEIGHT_SLOTS`;
  // ineligible ones are decoded on the CPU).
  storage: new Set(["f32", "f16", "bf16", "i8", "i4", "i2", "i32"]),
  io: new Set(IO_DTYPES),
};

/** What the runtime can execute, for queries (a projection in a stable shape for the public API). */
export type RuntimeCapabilities = {
  readonly ops: readonly string[];
  readonly storage: readonly string[];
};

export const capabilities = (): RuntimeCapabilities => ({
  ops: [...RUNTIME_SUPPORT.ops.keys()].sort(),
  storage: [...RUNTIME_SUPPORT.storage].sort(),
});

/**
 * Whether the input count fits the contract (for variable-arity ops, `arity` is read as a **lower
 * bound**).
 *
 * MUST: the verdict is not scattered across callers. Writing "=== if fixed, >= if variable" in 3
 * places (contract check / shape computation / CPU reference) leaves a stale verdict in one place,
 * as happened when cat was added. The exception **type** differs per layer (OpContractError in the
 * contract layer, ReferenceOpError in the CPU reference), so only the predicate and the display are
 * shared.
 */
export const arityFits = (found: OpContract, count: number): boolean =>
  found.variadic === true
    ? count >= found.arity
    : count >= found.arity && count <= (found.maxArity ?? found.arity);

/** The arity as displayed ("N or more" if variadic, "N or M" with an optional trailing input). */
export const describeArity = (found: OpContract): string => {
  if (found.variadic === true) return `${found.arity} or more`;
  return found.maxArity === undefined ? `${found.arity}` : `${found.arity} or ${found.maxArity}`;
};

export const assertArity = (
  found: OpContract,
  count: number,
  what: string,
  where: string,
): void => {
  if (arityFits(found, count)) return;
  throw new OpContractError(
    `${where}: op '${found.name}' ${what} is ${count} (contract: ${describeArity(found)})`,
  );
};

export const resolveOpContract = (op: string): OpContract => {
  const found = OP_CONTRACTS.get(op);
  if (found === undefined) {
    throw new OpContractError(`op '${op}' is not in the contract table`);
  }
  return found;
};

/**
 * Checks that the node fits its contract and returns the contract (shapes not included — only the
 * rules decided before binding).
 *
 * MUST: attrs are matched with `Object.hasOwn` only. The schemas are plain object literals, so keys
 * from the IR (`toString` and the like) would turn into "known keys" through the prototype.
 */
export const assertNodeContract = (node: IrNode, where: string): OpContract => {
  const found = resolveOpContract(node.op);
  const stateKeys = Object.keys(node.states);
  assertStateField(found, node, stateKeys, where);
  if (assertReadonlyForm(found, node, stateKeys, where)) {
    // readonly is a separate form that narrows ins to 1 (q), so it skips the closed-interval check
    // below.
    if (node.ins.length !== found.readonlyArity) {
      throw new OpContractError(
        `${where}: op '${node.op}' readonly (past-only) form takes exactly ${found.readonlyArity} inputs` +
          ` (${node.ins.length} given — this step's k / v are not taken; ADR 0096 stage 2 §1.2)`,
      );
    }
  } else {
    // MUST: the states form **takes no optional trailing input** (ADR 0067 decision 4 — "causal is
    // fixed and no mask tensor is materialised"). Without narrowing the upper bound, a states-form
    // attention with a mask would be accepted as "a form whose mask nobody reads".
    if (stateKeys.length > 0 && found.maxArity !== undefined && node.ins.length > found.arity) {
      throw new OpContractError(
        `${where}: op '${node.op}' states form takes exactly ${found.arity} inputs` +
          ` (${node.ins.length} given — no optional trailing input; ADR 0067 decision 4)`,
      );
    }
    assertArity(found, node.ins.length, "input count", where);
  }
  if (node.outs.length !== outputCountOf(found)) {
    throw new OpContractError(
      `${where}: op '${node.op}' output count is ${node.outs.length} (contract: ${outputCountOf(found)})`,
    );
  }
  assertAttrKeys(found, node.attrs, stateKeys.length > 0, where);
  assertAttrValues(found, node.attrs, where);
  return found;
};

/**
 * The **range** of attrs and the presence of required keys (the key set is {@link assertAttrKeys}).
 * One function shared by the node contract check and the CPU reference entry
 * (`applyReferenceOpOutputs` in `reference/ops.ts`) — without it on the reference side, the
 * reference alone would return a value for values the contract rejects (such as embedding's
 * `padding_idx: "bad"`).
 *
 * MUST: match with `Object.hasOwn` only (the same reason as {@link assertNodeContract}).
 */
export const assertAttrValues = (
  found: OpContract,
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): void => {
  for (const key of attrKeysOf(found)) {
    if (!Object.hasOwn(attrs, key)) {
      throw new OpContractError(`${where}: op '${found.name}' is missing required attr '${key}'`);
    }
    found.attrs[key](attrs[key], `${where} attrs.${key}`);
  }
  const optional = found.optionalAttrs;
  if (optional !== undefined) {
    for (const key of Object.keys(optional)) {
      if (!Object.hasOwn(attrs, key)) continue;
      optional[key](attrs[key], `${where} attrs.${key}`);
    }
  }
};

/**
 * Whether the **key set** of attrs fits the contract — two points: rejecting keys outside the
 * contract, and optional keys being for the states form only ({@link ContractBase.optionalAttrs}).
 * It does not look at ranges or at the presence of required keys.
 *
 * One function shared by the node contract check ({@link assertNodeContract}) and the CPU reference
 * entry (`applyReferenceOpOutputs` in `reference/ops.ts` — which has no states). Without the same
 * matching on the reference side, the reference alone would silently ignore attrs the contract
 * rejects (for example `window` on an attention without a states field) and return values, and the
 * oracle would turn a contract violation into values.
 *
 * MUST: match with `Object.hasOwn` only (the same reason as {@link assertNodeContract}).
 */
export const assertAttrKeys = (
  found: OpContract,
  attrs: Readonly<Record<string, unknown>>,
  hasStates: boolean,
  where: string,
): void => {
  const optional = found.optionalAttrs;
  const unknown = Object.keys(attrs).filter(
    (key) =>
      !Object.hasOwn(found.attrs, key) && (optional === undefined || !Object.hasOwn(optional, key)),
  );
  if (unknown.length > 0) {
    throw new OpContractError(
      `${where}: op '${found.name}' has attrs outside the contract [${unknown.sort().join(", ")}]`,
    );
  }
  if (optional === undefined || hasStates) return;
  for (const key of Object.keys(optional)) {
    if (!Object.hasOwn(attrs, key)) continue;
    throw new OpContractError(
      `${where}: op '${found.name}' attrs.${key} can be declared only on a node with a states field`,
    );
  }
};

/**
 * Settles whether `attrs.readonly` (ADR 0096 stage 2 §1.2) is declared, **before the arity check**.
 *
 * Only three points are checked (the range is held by {@link STATE_READONLY_ATTRS}): whether the
 * contract's op has readonly, whether there is a `states` field, and whether the value is `true`.
 *
 * MUST: "only nodes with a states field" is enforced here too (the same rule as the optional-attrs
 * loop below, applied first). Deferring it would make a readonly node without a states field fail
 * with a diagnostic **for a different reason**, "input count 1 (contract is 3 or 4)", and whoever
 * fixes it would not find their way to the readonly spelling.
 */
const assertReadonlyForm = (
  found: OpContract,
  node: IrNode,
  stateKeys: readonly string[],
  where: string,
): boolean => {
  if (!Object.hasOwn(node.attrs, "readonly")) return false;
  if (found.readonlyArity === undefined) {
    throw new OpContractError(
      `${where}: op '${found.name}' has no attrs.readonly (only the states-form attention has ` +
        "a form that reads the past only — ADR 0096 stage 2 §1.2)",
    );
  }
  if (stateKeys.length === 0) {
    throw new OpContractError(
      `${where}: op '${found.name}' attrs.readonly can be declared only on a node with a states field`,
    );
  }
  return stateReadonly(node.attrs, where);
};

/**
 * Whether the `states` field fits the contract (ADR 0067 decisions 4 / 5).
 *
 * MUST: the key set **equals** the contract's declaration (allowing a subset would let a
 * states-form attention that forgot to write `v` pass as the form "reads only k from state").
 * MUST: reject a node that references **the same slot** from two fields — a mix-up writing the same
 * slot name for `k` and `v` passes every shape check and becomes the silent wrong value "reads V as
 * K" (the k/v slots have the same shape, so shapes do not catch it).
 */
const assertStateField = (
  found: OpContract,
  node: IrNode,
  keys: readonly string[],
  where: string,
): void => {
  const declared = found.states;
  if (declared === undefined) {
    if (keys.length > 0) {
      throw new OpContractError(
        `${where}: op '${found.name}' has no states field ([${[...keys].sort().join(", ")}])`,
      );
    }
    return;
  }
  if (keys.length === 0) {
    if (declared.required) {
      throw new OpContractError(
        `${where}: op '${found.name}' states field is missing (contract: exactly [${declared.keys.join(
          ", ",
        )}])`,
      );
    }
    return;
  }
  const expected = [...declared.keys].sort();
  const actual = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new OpContractError(
      `${where}: op '${found.name}' states field keys are [${actual.join(", ")}] (contract: exactly [${expected.join(
        ", ",
      )}])`,
    );
  }
  const slots = declared.keys.map((key) => node.states[key]);
  const aliased = slots.findIndex((slot, index) => slots.indexOf(slot) !== index);
  if (aliased >= 0) {
    throw new OpContractError(
      `${where}: op '${found.name}' states field references the same slot '${
        slots[aliased]
      }' from more than one field (mix-up detection)`,
    );
  }
};

/**
 * Derives the dtype **of output slot `slot`** from slot 0's input dtype
 * ({@link ContractBase.outputDtypes}).
 *
 * The mapping's domain equals slot 0's accepted set ({@link slotContract} fills it with identity),
 * so a dtype that passed the slot check can always be looked up — failing to look it up is a broken
 * internal invariant of the runtime.
 */
export const outputDtypeOf = (
  found: OpContract,
  slot: number,
  inputDtype: IrDtype,
  where: string,
): IrDtype => {
  const column = found.outputDtypes[slot];
  if (column === undefined) {
    throw new OpContractError(`${where}: op '${found.name}' has no output slot ${slot}`);
  }
  const mapped = column.get(inputDtype);
  if (mapped === undefined) {
    throw new OpContractError(
      `${where}: op '${found.name}' output dtype mapping has no input '${inputDtype}'`,
    );
  }
  return mapped;
};

/**
 * Matches against the **union** of the accepted sets. A loose check across slots, so ops with a
 * per-slot contract use {@link assertSlotDtype} (this function only checks "is it a dtype the op
 * touches").
 */
export const assertDtype = (found: OpContract, dtype: IrDtype, where: string): void => {
  if (!found.dtypes.includes(dtype)) {
    throw new OpContractError(
      `${where}: op '${found.name}' cannot run semantic dtype '${dtype}' (supported: ${found.dtypes.join(
        ", ",
      )})`,
    );
  }
};

/**
 * The per-input-slot dtype check. Under a uniform contract it is the same as matching against the
 * union (the diagnostic text too); only under a perSlot contract does it look at the slot index.
 */
export const assertSlotDtype = (
  found: OpContract,
  slot: number,
  dtype: IrDtype,
  where: string,
): void => {
  if (found.slotDtypes.kind === "uniform") {
    assertDtype(found, dtype, where);
    return;
  }
  const accept = found.slotDtypes.slots[slot];
  if (accept === undefined) {
    throw new OpContractError(`${where}: op '${found.name}' has no input slot ${slot}`);
  }
  if (!accept.includes(dtype)) {
    throw new OpContractError(
      `${where}: op '${found.name}' input slot ${slot} cannot run semantic dtype '${dtype}' (supported: ${accept.join(
        ", ",
      )})`,
    );
  }
};

/**
 * The check of the input semantic dtypes (per-slot accepted sets + the same type across slots under
 * a uniform contract). One function shared by the node dtype check ({@link resolveNodeDtypes}) and
 * the CPU reference entry. `label` is the diagnostic name of input `index`.
 */
export const assertInputDtypes = (
  found: OpContract,
  inputDtypes: readonly IrDtype[],
  label: (index: number) => string,
  where: string,
): void => {
  inputDtypes.forEach((dtype, index) => {
    assertSlotDtype(found, index, dtype, label(index));
  });
  // Ops whose slots have different accepted sets do not require the same type across slots (that is
  // what perSlot means). Only a uniform contract rejects mixes.
  if (found.slotDtypes.kind === "uniform") {
    const mixed = inputDtypes.findIndex((dtype) => dtype !== inputDtypes[0]);
    if (mixed >= 0) {
      throw new OpContractError(
        `${where}: op '${found.name}' input dtypes are mixed (${inputDtypes.join(", ")})`,
      );
    }
  }
};

/**
 * Checks the node's semantic dtypes and **returns the list of output dtypes** (in output slot
 * order).
 *
 * MUST: the output dtypes are derived from the contract and matched against the declaration, not
 * taken on trust. Trusting the declaration would let a graph whose cast attrs.to and values{}
 * declaration disagree pass "as declared", and readback would read it as a different TypedArray — a
 * silent wrong value.
 * Outputs are derived in only two ways: cast from attrs.to, everything else **by passing slot 0's
 * dtype through each output slot's mapping** ({@link ContractBase.outputDtypes} — identity by
 * default). A uniform contract additionally requires the same type across slots (mixed-type
 * arithmetic is not in the vocabulary).
 */
export const resolveNodeDtypes = (
  found: OpContract,
  node: IrNode,
  inputDtypes: readonly IrDtype[],
  declaredOutputs: readonly IrDtype[],
  where: string,
): readonly IrDtype[] => {
  assertInputDtypes(found, inputDtypes, (index) => `${where} input '${node.ins[index]}'`, where);
  const expected: readonly IrDtype[] =
    found.kind === "cast"
      ? [castTargetDtype(node.attrs, where)]
      : found.outputDtypes.map((_, slot) => outputDtypeOf(found, slot, inputDtypes[0], where));
  if (declaredOutputs.length !== expected.length) {
    throw new OpContractError(
      `${where}: op '${found.name}' declares ${declaredOutputs.length} outputs (contract: ${expected.length})`,
    );
  }
  expected.forEach((dtype, slot) => {
    if (declaredOutputs[slot] !== dtype) {
      throw new OpContractError(
        `${where}: output '${node.outs[slot]}' declared dtype '${
          declaredOutputs[slot]
        }' differs from the contract's '${dtype}'`,
      );
    }
  });
  return expected;
};
