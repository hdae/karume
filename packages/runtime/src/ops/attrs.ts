import { f32EpsDefect } from "../codegen/f32-eps.ts";
import { isSymbolName } from "../format/dims.ts";
import { type IrDtype, isSemanticDtype, SEMANTIC_DTYPES } from "../format/ir.ts";
import { OpContractError } from "./names.ts";

/**
 * attrs schema = attr key → check of the value (ADR 0012).
 *
 * MUST: every declared key is **required**, and keys outside the declaration fail loudly. An attr
 * whose absence is itself a separate declaration is not mixed into the required schema; it is
 * declared separately in the contract's `optionalAttrs` (ops/contracts.ts).
 */
export type AttrSchema = Readonly<Record<string, (value: unknown, where: string) => void>>;
/**
 * Checks the target of a cast. Accepts the IR's semantic dtype vocabulary as is (a same-type cast is
 * an identity copy).
 *
 * MUST: f32 → i32 is **truncation (toward 0)** as in torch, and x → bool is **x != 0**. The actual
 * representation of bool is u32 0 / 1 (ADR 0009). Unless the rounding convention is stated in the
 * contract, the GPU's `i32(x)` (WGSL also truncates) and the CPU reference drift apart silently.
 */
const assertCastTarget = (value: unknown, where: string): IrDtype => {
  if (!isSemanticDtype(value)) {
    throw new OpContractError(
      `${where}: cast target is not a semantic dtype (${SEMANTIC_DTYPES.join(" / ")}): ${JSON.stringify(
        value,
      )}`,
    );
  }
  return value;
};

/**
 * The target dtype of a cast node (assumes {@link assertNodeContract} has already checked the
 * attrs).
 */
export const castTargetDtype = (attrs: Readonly<Record<string, unknown>>, where: string): IrDtype =>
  assertCastTarget(Object.hasOwn(attrs, "to") ? attrs["to"] : undefined, `${where} attrs.to`);

export const CAST_ATTRS: AttrSchema = {
  to: (value, where) => {
    assertCastTarget(value, where);
  },
};

/**
 * Checks the axis permutation of permute. `dims[d]` = the input dimension that output dimension d
 * takes.
 *
 * MUST: negative axis indices are not accepted. torch's `-1` notation is normalised at the exporter
 * boundary by convention; accepting both notations at runtime would give two IR spellings of the
 * same permutation.
 * MUST: duplicates are rejected (a permutation is a bijection). Allowing them would turn it into a
 * "replication" that reads the same input axis twice, and only the stride computation would pass
 * while element counts disagree. Matching against the input rank happens in the shape computation
 * (the rank is known only after binding).
 */
const assertPermuteDims = (value: unknown, where: string): readonly number[] => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new OpContractError(
      `${where}: permute dims is not a non-empty array: ${JSON.stringify(value)}`,
    );
  }
  const dims: number[] = value.map((dim) => {
    if (typeof dim !== "number" || !Number.isSafeInteger(dim) || dim < 0) {
      throw new OpContractError(
        `${where}: permute dims has an element that is not a non-negative integer: ${JSON.stringify(value)}`,
      );
    }
    return dim;
  });
  if (new Set(dims).size !== dims.length) {
    throw new OpContractError(`${where}: permute dims [${dims.join(",")}] has duplicates`);
  }
  return dims;
};

/**
 * The axis permutation of a permute node (assumes {@link assertNodeContract} has already checked
 * the attrs).
 */
export const permuteDims = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): readonly number[] =>
  assertPermuteDims(
    Object.hasOwn(attrs, "dims") ? attrs["dims"] : undefined,
    `${where} attrs.dims`,
  );

export const PERMUTE_ATTRS: AttrSchema = {
  dims: (value, where) => {
    assertPermuteDims(value, where);
  },
};

/**
 * Reads an attrs value without touching prototype pollution (`Object.hasOwn` only — a cross-cutting
 * invariant).
 */
export const attrValue = (attrs: Readonly<Record<string, unknown>>, key: string): unknown =>
  Object.hasOwn(attrs, key) ? attrs[key] : undefined;

/**
 * An integer attr of at least `min`.
 * MUST: check `typeof value === "number"` first (`Number.isSafeInteger` returns false for
 * non-numbers so they would be caught, but a form where a boolean slips through the comparison
 * without being converted to a number can come in from JSON).
 */
const assertIntegerAttr = (value: unknown, where: string, min: number): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min) {
    throw new OpContractError(`${where}: not an integer >= ${min}: ${JSON.stringify(value)}`);
  }
  return value;
};

/**
 * The cut-out spec of slice (ADR 0014). Shrinks axis `dim` to `[start, end)`.
 *
 * MUST: all three are **non-negative integers** (negative axis notation and negative index notation
 * are normalised at the exporter boundary by convention — as for permute's dims). Bringing in
 * torch's `x[..., -1]` or the huge default end as is would give two IR spellings of the same slice.
 * MUST: `start <= end` and `end <= the input's axis length` are checked in the shape computation
 * (invariants that span keys and rules that need the input shape cannot be expressed in an attrs
 * schema — the same split as clamp's min/max).
 */
export type SliceAttrs = {
  readonly dim: number;
  readonly start: number;
  readonly end: number;
};

export const SLICE_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  start: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  end: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/** The attrs of a slice node (assumes {@link assertNodeContract} has already checked them). */
export const sliceAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): SliceAttrs => ({
  dim: assertIntegerAttr(attrValue(attrs, "dim"), `${where} attrs.dim`, 0),
  start: assertIntegerAttr(attrValue(attrs, "start"), `${where} attrs.start`, 0),
  end: assertIntegerAttr(attrValue(attrs, "end"), `${where} attrs.end`, 0),
});

/**
 * The axis of cat / flip (a non-negative axis index; matching against the input rank happens in the
 * shape computation).
 */
export const AXIS_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

export const catDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} attrs.dim`, 0);

export const flipDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} attrs.dim`, 0);

/**
 * The left / right padding widths of pad (**last dimension, constant 0 only** — ADR 0014).
 *
 * MUST: negative widths are not accepted (torch's `constant_pad_nd` can trim with a negative
 * width). Trimming means slice, not pad; letting it through would mean "the same form can be written
 * with two ops", and the dedicated kernel's output length computation would go negative.
 * MUST: no fill value in attrs. Only 0 was measured, and having no field structurally closes the
 * path that "silently runs a non-zero value as 0" (the same narrowing as conv1d's groups).
 */
export type PadAttrs = {
  readonly left: number;
  readonly right: number;
};

export const PAD_ATTRS: AttrSchema = {
  left: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  right: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

export const padAttrs = (attrs: Readonly<Record<string, unknown>>, where: string): PadAttrs => ({
  left: assertIntegerAttr(attrValue(attrs, "left"), `${where} attrs.left`, 0),
  right: assertIntegerAttr(attrValue(attrs, "right"), `${where} attrs.right`, 0),
});

/** The cut-out spec of sym_prefix_slice for one axis (shrinks `dim` to length `coeff·sym+offset`). */
export type PrefixSlice = {
  readonly dim: number;
  readonly coeff: number;
  readonly offset: number;
};

const PREFIX_SLICE_KEYS = ["dim", "coeff", "offset"] as const;

/**
 * Checks the `slices` of sym_prefix_slice.
 *
 * MUST: duplicate axes are rejected. Two specs on the same axis would make one silently disappear by
 * "last wins", producing a form that passes only the match against the declared shape.
 * MUST: the coefficient is at least 1 and the offset at least 0 (the same range as the dimension
 * language `coeff·sym+offset` — allowing negatives makes the prefix length negative and only the
 * element count computation passes, with 0).
 * MUST: keys are looked up with `Object.hasOwn` only (objects from JSON appear to "have" `toString`
 * and the like through the prototype — a cross-cutting invariant).
 */
const assertPrefixSlices = (value: unknown, where: string): readonly PrefixSlice[] => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new OpContractError(
      `${where}: sym_prefix_slice slices is not a non-empty array: ${JSON.stringify(value)}`,
    );
  }
  const slices: PrefixSlice[] = value.map((raw, index) => {
    const at = `${where}[${index}]`;
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      throw new OpContractError(`${at}: not an object: ${JSON.stringify(raw)}`);
    }
    const entry: Record<string, unknown> = raw as Record<string, unknown>;
    const unknown = Object.keys(entry).filter(
      (key) => !PREFIX_SLICE_KEYS.some((known) => known === key),
    );
    if (unknown.length > 0) {
      throw new OpContractError(`${at}: unknown keys [${unknown.sort().join(", ")}]`);
    }
    for (const key of PREFIX_SLICE_KEYS) {
      if (!Object.hasOwn(entry, key)) throw new OpContractError(`${at}: key '${key}' is missing`);
    }
    return {
      dim: assertIntegerAttr(entry["dim"], `${at}.dim`, 0),
      coeff: assertIntegerAttr(entry["coeff"], `${at}.coeff`, 1),
      offset: assertIntegerAttr(entry["offset"], `${at}.offset`, 0),
    };
  });
  const dims = slices.map((slice) => slice.dim);
  if (new Set(dims).size !== dims.length) {
    throw new OpContractError(`${where}: sym_prefix_slice slices has the same dim twice`);
  }
  return slices;
};

/**
 * The `sym` of sym_prefix_slice (a symbol name of the dimension language — whether it is bound is
 * checked on the graph side).
 */
const assertPrefixSym = (value: unknown, where: string): string => {
  if (typeof value !== "string" || !isSymbolName(value)) {
    throw new OpContractError(
      `${where}: sym_prefix_slice sym is not a symbol name: ${JSON.stringify(value)}`,
    );
  }
  return value;
};

export const SYM_PREFIX_SLICE_ATTRS: AttrSchema = {
  sym: (value, where) => {
    assertPrefixSym(value, where);
  },
  slices: (value, where) => {
    assertPrefixSlices(value, where);
  },
};

/**
 * The attrs of a sym_prefix_slice node (assumes {@link assertNodeContract} has already checked
 * them).
 */
export const symPrefixSliceAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly sym: string; readonly slices: readonly PrefixSlice[] } => ({
  sym: assertPrefixSym(attrValue(attrs, "sym"), `${where} attrs.sym`),
  slices: assertPrefixSlices(attrValue(attrs, "slices"), `${where} attrs.slices`),
});

/**
 * The normalised axes of layer_norm. **Only length 1 (= the last dimension)** is accepted.
 *
 * MUST: multi-axis normalisation is not accepted "for symmetry" (ADR 0007's frozen vocabulary
 * allowlist). All 7 measured ones are `[1024]` (recon §5), and the row kernel is built on the
 * contiguous layout of the last dimension. Matching against the axis (whether it equals the tail of
 * `x.shape`) happens in the shape computation after binding.
 */
const assertNormalizedShape = (value: unknown, where: string): readonly number[] => {
  if (!Array.isArray(value) || value.length !== 1) {
    throw new OpContractError(
      `${where}: layer_norm normalized_shape must be an array of length 1 (only last-dimension normalisation can run): ${JSON.stringify(
        value,
      )}`,
    );
  }
  return [assertIntegerAttr(value[0], `${where}[0]`, 1)];
};

/**
 * The eps of the normalisation ops (layer_norm / rms_norm). **Finite positive numbers** only
 * (allowing 0 makes `1/sqrt(0)` inf for a row with variance 0 and all elements 0, and "normalising a
 * constant row" silently produces NaN).
 *
 * MUST: it **rounds to a normal f32 number** (values like `1e-50` that round to 0 and values like
 * `1e-40` that round to a subnormal are both rejected). The GPU carries eps in an f32 word of params
 * (kernels/layer-norm.ts / rms-norm.ts), but WGSL does not require subnormals to be preserved — on an
 * FTZ implementation an f32 subnormal collapses to 0, and the reason above for "not allowing 0"
 * comes back **on the GPU side only** (on an all-zero row the GPU gives NaN while the CPU gives 0, so
 * the meaning splits).
 * MUST: it **does not become infinite when rounded to f32** (an overshoot like `1e39`). If the
 * rounded value is `+Inf`, the GPU computes `1/sqrt(… + Inf) = 0` and the output is the bias alone,
 * while the CPU reference keeps normalising with a finite f64 — the same split as the undershoot, in
 * the opposite direction (both are silent wrong values with no exception).
 */
const assertEps = (value: unknown, where: string, what: string): number => {
  // The verdict is shared with the kernels' params (kernels/layer-norm.ts / rms-norm.ts) — if only
  // one side let a subnormal through, only the path that calls the kernel directly would run with
  // eps = 0 on the GPU.
  const defect = typeof value === "number" ? f32EpsDefect(value) : "not-positive";
  if (typeof value !== "number" || defect === "not-positive") {
    throw new OpContractError(
      `${where}: ${what} eps is not a finite positive number: ${JSON.stringify(value)}`,
    );
  }
  if (defect === "subnormal") {
    throw new OpContractError(
      `${where}: ${what} eps does not round to a normal f32 number (an FTZ implementation would run with eps=0 on the GPU side only): ${JSON.stringify(
        value,
      )}`,
    );
  }
  if (defect === "overflow") {
    throw new OpContractError(
      `${where}: ${what} eps becomes infinite when rounded to f32 (the GPU side alone would run with eps=Inf): ${JSON.stringify(
        value,
      )}`,
    );
  }
  return value;
};

/**
 * A scalar attr carried in an f32 word of params. **A finite f32 scalar**.
 *
 * MUST: this is the only check that rejects non-finite scalar attrs regardless of the supplier. For
 * `krm`, the descriptor reader (`openContainer`) rejects non-finite numbers, but the in-memory
 * container path (`openMemoryContainer`) does not check for them (the container-format pitfalls in
 * .claude/ACTIVE_DESIGN.md).
 *
 * NOTE: values are not narrowed to those exactly representable in f32 — under the convention of
 * rounding to f32 when applied (the GPU's f32 word of params, `Math.fround` in the CPU reference),
 * both sides agree.
 *
 * MUST: the rounded value is **finite** too. `1e39` is finite in f64 but becomes `+Inf` in f32, and
 * a contract declared as "a finite f32 scalar" would silently let a non-finite value through (both
 * sides run with `Inf` alike, so the values do not split, but another op turns it into a wrong value,
 * as in `0 * Inf = NaN`). If the intent is to pin to the upper bound, the largest finite f32 value
 * can be written, so there is no reason to widen the accepted set either.
 */
export const assertFiniteAttr = (value: unknown, where: string, what: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new OpContractError(`${where}: ${what} is not a finite number: ${JSON.stringify(value)}`);
  }
  if (!Number.isFinite(Math.fround(value))) {
    throw new OpContractError(
      `${where}: ${what} becomes infinite when rounded to f32: ${JSON.stringify(value)}`,
    );
  }
  return value;
};

/**
 * The fill value of masked_fill. The measured value −3.4028234663852886e+38 is exactly the smallest
 * finite f32 value, and its ulp moves neither on a JSON round trip nor on f32 rounding (pinned by
 * tests/ops_conformance_test.ts).
 */
const assertFillValue = (value: unknown, where: string): number =>
  assertFiniteAttr(value, where, "masked_fill value");

export const LAYER_NORM_ATTRS: AttrSchema = {
  eps: (value, where) => {
    assertEps(value, where, "layer_norm");
  },
  normalized_shape: (value, where) => {
    assertNormalizedShape(value, where);
  },
};

/**
 * The attrs of rms_norm (ADR 0017).
 *
 * MUST: no `normalized_shape` field. The normalised axis is fixed to the single last dimension, and
 * the length is carried by **the weight's length** (layer_norm holds the same fact twice, in attrs
 * and in the weight, and the shape computation checks that they agree every time). Adding the field
 * would add the same double bookkeeping for one more op.
 */
export const RMS_NORM_ATTRS: AttrSchema = {
  eps: (value, where) => {
    assertEps(value, where, "rms_norm");
  },
};

/** The eps of an rms_norm node (assumes {@link assertNodeContract} has already checked it). */
export const rmsNormEps = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertEps(attrValue(attrs, "eps"), `${where} attrs.eps`, "rms_norm");

export const SOFTMAX_ATTRS: AttrSchema = {
  // Negative axis notation is normalised at the exporter boundary by convention (as for permute's
  // dims). Matching against the last dimension happens in the shape computation, where the rank is
  // known.
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * The attrs of attention (ADR 0023).
 *
 * MUST: `scale` is **required and not filled with a default**. By convention the exporter computes
 * `f32(√scale_factor)` from SDPA's `scale` argument (`1/√D` when omitted) and writes it; if the
 * runtime filled in "1/√D when absent", the IR of an SDPA with an explicit `scale` and the filled-in
 * IR would look the same, and a mix-up would show only in the values (the same discipline as
 * conv1d's dilation / groups — ADR 0015).
 * MUST: no mask / causal / dropout fields. The absence of the field itself expresses structurally
 * that "that form is not in the vocabulary" (the same narrowing as pad's fill value).
 */
export const ATTENTION_ATTRS: AttrSchema = {
  scale: (value, where) => {
    assertFiniteAttr(value, where, "attention scale");
  },
};

/** The half scale of an attention node (assumes {@link assertNodeContract} has already checked it). */
export const attentionScale = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertFiniteAttr(attrValue(attrs, "scale"), `${where} attrs.scale`, "attention scale");

/**
 * The sliding window of a state-referencing node (ADR 0067 decisions 4 / 5). **An optional attr**;
 * the absence of the field means "the whole context". When declared, an integer of at least 1.
 *
 * MUST: not filled with a default. "No field = the whole context" and "window = capacity" are
 * different declarations; filling it in would make a mix-up between the sliding layers (28 layers
 * of Gemma 4 E2B) and the full layers (7 layers) show only in the values.
 * MUST: **presence and value agree** across every node that touches the same slot (the same-formula
 * MUST for reads and writes — checked by `validateGraphContracts` in `runtime/plan.ts`. A different
 * formula on the read side alone means silent misreads).
 */
export const STATE_WINDOW_ATTRS: AttrSchema = {
  window: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/**
 * The declaration attr of readonly attention (ADR 0096 stage 2 §1.2).
 * **Optional and `true` only**; the absence of the field means "the previous states form that also
 * reads this step's k/v".
 *
 * MUST: `false` is not accepted (the same style as `window` — no two spellings that mean the same as
 * the absence of the field).
 * MUST: not filled with a default. Filling it in would make a mix-up between "the past-only form
 * without ins" and "the form that adds this step's k/v" show only in the values, after it has passed
 * the arity check.
 */
export const STATE_READONLY_ATTRS: AttrSchema = {
  readonly: (value, where) => {
    if (value !== true) {
      throw new OpContractError(
        `${where}: ${JSON.stringify(value)} is not allowed (readonly accepts only true — ` +
          "omitting the field means the states form that also reads this step's k/v)",
      );
    }
  },
};

/**
 * Whether the `attrs` of a states-form attention declare **readonly** (the form that reads only the
 * past).
 *
 * MUST: look up with `Object.hasOwn` only (keys from Object.prototype do not slip through).
 */
export const stateReadonly = (attrs: Readonly<Record<string, unknown>>, where: string): boolean => {
  if (!Object.hasOwn(attrs, "readonly")) return false;
  STATE_READONLY_ATTRS["readonly"](attrs["readonly"], `${where} attrs.readonly`);
  return true;
};

/** The `window` of a state-referencing node (`undefined` when not declared = the whole context). */
export const stateWindow = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): number | undefined =>
  Object.hasOwn(attrs, "window")
    ? assertIntegerAttr(attrs["window"], `${where} attrs.window`, 1)
    : undefined;

export const EMBEDDING_ATTRS: AttrSchema = {
  /**
   * torch's `padding_idx`. **Accepted, but it has no effect in forward** (the field exists so that
   * gradients do not update the padding row; the forward pass is exactly a plain row gather — torch's
   * `F.embedding` does not consult it in forward either). So it is passed neither to the kernel nor
   * to the CPU reference.
   *
   * MUST: it is still listed in the contract table. Dropping and ignoring the attrs would open a hole
   * in the "unknown attrs fail loudly" discipline, and the next field that does affect forward would
   * slip through for the same reason.
   * The range follows torch: at least `-1` (the sentinel for "not specified").
   */
  padding_idx: (value, where) => {
    assertIntegerAttr(value, where, -1);
  },
};

export const MASKED_FILL_ATTRS: AttrSchema = {
  value: (value, where) => {
    assertFillValue(value, where);
  },
};

export const CLAMP_ATTRS: AttrSchema = {
  min: (value, where) => {
    assertFiniteAttr(value, where, "clamp min");
  },
  max: (value, where) => {
    assertFiniteAttr(value, where, "clamp max");
  },
};

/**
 * The attrs of clamp_min (ADR 0017 — the 30 `clamp(min=eps)` of channel L2 normalisation).
 *
 * MUST: no `max` field. "Fill the missing side with the largest finite f32 value and pass it to
 * clamp" is an approximation that replaces semantics without an upper bound by "an upper bound large
 * enough to saturate", and ADR 0017 rejects it by name. The absence of the field itself expresses
 * structurally the separation from clamp, which requires both sides.
 */
export const CLAMP_MIN_ATTRS: AttrSchema = {
  min: (value, where) => {
    assertFiniteAttr(value, where, "clamp_min min");
  },
};

export const LEAKY_RELU_ATTRS: AttrSchema = {
  /**
   * torch's `negative_slope`. **Required, not filled with a default** (ADR 0015) — dec mixes 0.1
   * (ups / ResBlock) and 0.01 (the last stage, with the positional argument omitted), and relying on
   * a default would silently get one of them wrong. The exporter reads torch's default and writes it
   * into attrs (made explicit at the boundary).
   */
  negative_slope: (value, where) => {
    assertFiniteAttr(value, where, "leaky_relu negative_slope");
  },
};

/** The comparison operand of ge_scalar / le_scalar / gt_scalar (a finite f32 scalar). */
export const SCALAR_COMPARE_ATTRS: AttrSchema = {
  value: (value, where) => {
    assertFiniteAttr(value, where, "comparison op value");
  },
};

export const CUMSUM_ATTRS: AttrSchema = {
  // Negative axis notation is normalised at the exporter boundary by convention (as for softmax's
  // dim). Matching against the last dimension happens in the shape computation, where the rank is
  // known.
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * The reduced axis of the reduce family (sum / amax / amin).
 *
 * MUST: **required** (not filled with a default — the same reason as conv1d's dilation / groups).
 * Allowing "no field = the last dimension" would make IR meant to reduce the channel axis silently
 * run as a different computation that collapses the last dimension (combinations where the shapes
 * happen to match do exist).
 * Negative axis notation is normalised at the exporter boundary by convention, and matching against
 * the rank happens in the shape computation.
 */
export const REDUCE_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * The count `k` of topk (ADR 0068 decision 3).
 *
 * MUST: **a required positive integer** (= static-k). torch accepts k=0 (an empty output), but a node
 * that returns "0 of the largest" is the same as defining no value, so it is not in the vocabulary.
 * A symbolic k (a dimension-expression string) is also rejected here by `typeof value !== "number"`
 * — a k decided at run time does not fit the static-shape premise (ADR 0004), and neither the output
 * shape nor the allocated bytes could be decided at plan time.
 * MUST: upper bounds are **not checked here**. `k ≤ last dimension` needs the input rank, so it is in
 * the shape computation; the **implementation limit** (from workgroup storage) needs the device
 * limits, so it is in the recipe assembly (`assertTopkK` in src/kernels/topk.ts) — the same split of
 * layers as clamp's min/max.
 */
export const TOPK_ATTRS: AttrSchema = {
  k: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/**
 * The attrs of conv1d (`dilation` / `groups` added by ADR 0015).
 *
 * MUST: all four are **required** ({@link assertNodeContract} requires every key to be present).
 * The property "there is no path that silently runs a value other than 1 as 1", once guaranteed by
 * "no field = fixed at 1", is guaranteed after adding the fields only by "no default filling" — the
 * moment `dilation` / `groups` become optional, depthwise IR silently runs as an ordinary
 * convolution.
 */
export const CONV1D_ATTRS: AttrSchema = {
  dilation: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  padding: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  stride: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/**
 * The attrs of conv_transpose1d (ADR 0015).
 *
 * MUST: `stride >= 1`. stride 0 is not even the form "one input point contributes to only one output
 * point"; depending on the implementation the loop does not advance and **the GPU hangs** (no
 * exception) — it is blocked both in the contract check and in the params check (recon §4).
 * MUST: no fields for `output_padding` / `dilation` / `groups`. The measurements are all 0 / 1 / 1,
 * and having no field structurally closes the path that "silently runs a value other than 0 / 1 with
 * the default" (the same narrowing as pad's fill value). They are widened once measurements show up.
 */
export const CONV_TRANSPOSE1D_ATTRS: AttrSchema = {
  padding: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  stride: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/** The attrs of a layer_norm node (assumes {@link assertNodeContract} has already checked them). */
export const layerNormAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly normalizedShape: readonly number[]; readonly eps: number } => ({
  normalizedShape: assertNormalizedShape(
    attrValue(attrs, "normalized_shape"),
    `${where} attrs.normalized_shape`,
  ),
  eps: assertEps(attrValue(attrs, "eps"), `${where} attrs.eps`, "layer_norm"),
});

/** The reduced axis of a softmax node (a non-negative axis index). */
export const softmaxDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} attrs.dim`, 0);

/** The fill value of a masked_fill node. */
export const maskedFillValue = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertFillValue(attrValue(attrs, "value"), `${where} attrs.value`);

/** The accumulation axis of a cumsum node (a non-negative axis index). */
export const cumsumDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} attrs.dim`, 0);

/** The reduced axis of a reduce-family node (a non-negative axis index — no default filling). */
export const reduceDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} attrs.dim`, 0);

/** The count k of a topk node (a positive integer — no default filling). */
export const topkK = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "k"), `${where} attrs.k`, 1);

/**
 * Attrs passed to the elementwise kernel as f32 **at the end of params** (the order is the params
 * layout as is).
 *
 * MUST: the values are not baked into the WGSL. Baking them would add one pipeline per distinct
 * value (the same reason as masked_fill's fill value — src/kernels/masked-fill.ts).
 * MUST: this is the single source of truth for the order. Relying on the key order of the attrs
 * schema (the object's insertion order) would let a reordering of the declaration silently shift the
 * codegen indices and the params write order.
 */
export const SCALAR_PARAM_ATTRS: ReadonlyMap<string, readonly string[]> = new Map([
  ["clamp", ["min", "max"]],
  ["clamp_min", ["min"]],
  ["leaky_relu", ["negative_slope"]],
  ["ge_scalar", ["value"]],
  ["le_scalar", ["value"]],
  ["gt_scalar", ["value"]],
]);

/** The number of f32 scalars the op puts at the end of params (used by codegen to decide indices). */
export const scalarParamCount = (op: string): number => SCALAR_PARAM_ATTRS.get(op)?.length ?? 0;

/**
 * A spatial attr of conv2d (two components, `[H, W]`).
 *
 * MUST: only **arrays of length exactly 2** are accepted. Also allowing the scalar notation (torch's
 * `stride=1` applying to both axes) would give two IR spellings of the same convolution and split
 * both CSE and the conformance-table matching — normalisation is the exporter boundary's job (the
 * same split as permute's negative axis notation).
 * MUST: H and W are not split into separate keys (`stride_h` / `stride_w`). Two axes times three
 * attrs makes 6 keys, and "IR that forgot to write one of them" would degrade from looking like "a
 * missing required key" to "a value that looks like the default". The type expresses that they are a
 * pair.
 */
const assertIntPair = (
  value: unknown,
  where: string,
  min: number,
  what: string,
): readonly [number, number] => {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new OpContractError(
      `${where}: ${what} is not an array [H, W] of length 2: ${JSON.stringify(value)}`,
    );
  }
  return [
    assertIntegerAttr(value[0], `${where}[0]`, min),
    assertIntegerAttr(value[1], `${where}[1]`, min),
  ];
};

/**
 * The attrs of conv2d (ADR 0017). The three spatial ones have two components, H/W; `groups` is a
 * scalar.
 *
 * MUST: all four are **required, with no default filling** (the same discipline as conv1d — ADR
 * 0015). Depthwise (groups = Cin = Cout) and asymmetric stride/padding appear in measurements, so
 * allowing omission would silently run them as an ordinary convolution with symmetric padding.
 */
export const CONV2D_ATTRS: AttrSchema = {
  dilation: (value, where) => {
    assertIntPair(value, where, 1, "conv2d dilation");
  },
  groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  padding: (value, where) => {
    assertIntPair(value, where, 0, "conv2d padding");
  },
  stride: (value, where) => {
    assertIntPair(value, where, 1, "conv2d stride");
  },
};

/** The stride / padding / dilation (H/W pairs) and groups of a conv2d node. */
export type Conv2dAttrs = {
  readonly stride: readonly [number, number];
  readonly padding: readonly [number, number];
  readonly dilation: readonly [number, number];
  readonly groups: number;
};

export const conv2dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): Conv2dAttrs => ({
  stride: assertIntPair(attrValue(attrs, "stride"), `${where} attrs.stride`, 1, "conv2d stride"),
  padding: assertIntPair(
    attrValue(attrs, "padding"),
    `${where} attrs.padding`,
    0,
    "conv2d padding",
  ),
  dilation: assertIntPair(
    attrValue(attrs, "dilation"),
    `${where} attrs.dilation`,
    1,
    "conv2d dilation",
  ),
  groups: assertIntegerAttr(attrValue(attrs, "groups"), `${where} attrs.groups`, 1),
});

/**
 * A spatial attr of conv3d (three components, `[T, H, W]` — ADR 0118 decision 1).
 *
 * MUST: only **arrays of length exactly 3** are accepted (the same discipline as conv2d's
 * {@link assertIntPair}). Also allowing the scalar notation or two components would give several IR
 * spellings of the same convolution, and two components without the time axis could not be told
 * apart as "meant as [H, W]" or "a typo for [T, H]".
 */
const assertIntTriple = (
  value: unknown,
  where: string,
  min: number,
  what: string,
): readonly [number, number, number] => {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new OpContractError(
      `${where}: ${what} is not an array [T, H, W] of length 3: ${JSON.stringify(value)}`,
    );
  }
  return [
    assertIntegerAttr(value[0], `${where}[0]`, min),
    assertIntegerAttr(value[1], `${where}[1]`, min),
    assertIntegerAttr(value[2], `${where}[2]`, min),
  ];
};

/**
 * The attrs of conv3d (ADR 0118 decision 1). The three spatial ones have three components, T/H/W;
 * `groups` is a scalar.
 *
 * MUST: all four are **required, with no default filling** (the same discipline as conv1d / conv2d
 * — ADR 0015).
 */
export const CONV3D_ATTRS: AttrSchema = {
  dilation: (value, where) => {
    assertIntTriple(value, where, 1, "conv3d dilation");
  },
  groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  padding: (value, where) => {
    assertIntTriple(value, where, 0, "conv3d padding");
  },
  stride: (value, where) => {
    assertIntTriple(value, where, 1, "conv3d stride");
  },
};

/** The stride / padding / dilation (T/H/W triples) and groups of a conv3d node. */
export type Conv3dAttrs = {
  readonly stride: readonly [number, number, number];
  readonly padding: readonly [number, number, number];
  readonly dilation: readonly [number, number, number];
  readonly groups: number;
};

export const conv3dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): Conv3dAttrs => ({
  stride: assertIntTriple(attrValue(attrs, "stride"), `${where} attrs.stride`, 1, "conv3d stride"),
  padding: assertIntTriple(
    attrValue(attrs, "padding"),
    `${where} attrs.padding`,
    0,
    "conv3d padding",
  ),
  dilation: assertIntTriple(
    attrValue(attrs, "dilation"),
    `${where} attrs.dilation`,
    1,
    "conv3d dilation",
  ),
  groups: assertIntegerAttr(attrValue(attrs, "groups"), `${where} attrs.groups`, 1),
});

/**
 * The attrs of deform_conv2d (extended atomic layer, ADR 0055). **Only the one key `padding`
 * (`[H, W]`)**.
 *
 * MUST: **do not add** the keys `stride` / `dilation` / `groups` / `offset_groups`. The measurements
 * are all 1, and the absence of the field expresses structurally that "that form is not in the
 * vocabulary" (paired with the fail-loudly at the exporter boundary). The value has the same form as
 * conv2d: an array `[H, W]` of length 2.
 */
export const DEFORM_CONV2D_ATTRS: AttrSchema = {
  padding: (value, where) => {
    assertIntPair(value, where, 0, "deform_conv2d padding");
  },
};

/** The spatial padding (`[H, W]`) of a deform_conv2d node. */
export type DeformConv2dAttrs = {
  readonly padding: readonly [number, number];
};

export const deformConv2dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): DeformConv2dAttrs => ({
  padding: assertIntPair(
    attrValue(attrs, "padding"),
    `${where} attrs.padding`,
    0,
    "deform_conv2d padding",
  ),
});

/**
 * The attrs of upsample_bilinear2d (Core ATen layer). **Only the one key `output_size`
 * (`[Hout, Wout]`)**.
 *
 * MUST: an array `[H, W]` of length 2, as for conv2d; the scalar notation is not accepted (it would
 * give two IR spellings of the same resample — normalisation is the exporter boundary's job).
 * MUST: **do not add** the fields `align_corners` / `mode` / `scale_factor`. The absence of the field
 * is itself the statement "that form is not in the vocabulary", paired with the fail-loudly on the
 * exporter side.
 */
export const UPSAMPLE_BILINEAR2D_ATTRS: AttrSchema = {
  output_size: (value, where) => {
    assertIntPair(value, where, 1, "upsample_bilinear2d output_size");
  },
};

/**
 * The attrs of group_norm (Core ATen layer — ADR 0138 decision 7): `num_groups` (a positive
 * integer; dividing the channel count is the shape rule's check, where the input is known) and
 * `eps` (the normalisation ops' rule — a finite positive number that rounds to a normal f32).
 */
export const GROUP_NORM_ATTRS: AttrSchema = {
  eps: (value, where) => {
    assertEps(value, where, "group_norm");
  },
  num_groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/** The attrs of a group_norm node (assumes {@link assertNodeContract} has already checked them). */
export const groupNormAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly groups: number; readonly eps: number } => ({
  groups: assertIntegerAttr(attrValue(attrs, "num_groups"), `${where} attrs.num_groups`, 1),
  eps: assertEps(attrValue(attrs, "eps"), `${where} attrs.eps`, "group_norm"),
});

/** The output spatial size (`[Hout, Wout]`) of an upsample_bilinear2d node. */
export type UpsampleBilinear2dAttrs = {
  readonly outputSize: readonly [number, number];
};

export const upsampleBilinear2dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): UpsampleBilinear2dAttrs => ({
  outputSize: assertIntPair(
    attrValue(attrs, "output_size"),
    `${where} attrs.output_size`,
    1,
    "upsample_bilinear2d output_size",
  ),
});

/** The stride / padding / dilation / groups of a conv1d node. */
export type Conv1dAttrs = {
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;
};

export const conv1dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): Conv1dAttrs => ({
  stride: assertIntegerAttr(attrValue(attrs, "stride"), `${where} attrs.stride`, 1),
  padding: assertIntegerAttr(attrValue(attrs, "padding"), `${where} attrs.padding`, 0),
  dilation: assertIntegerAttr(attrValue(attrs, "dilation"), `${where} attrs.dilation`, 1),
  groups: assertIntegerAttr(attrValue(attrs, "groups"), `${where} attrs.groups`, 1),
});

/** The stride / padding of a conv_transpose1d node. */
export const convTranspose1dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly stride: number; readonly padding: number } => ({
  stride: assertIntegerAttr(attrValue(attrs, "stride"), `${where} attrs.stride`, 1),
  padding: assertIntegerAttr(attrValue(attrs, "padding"), `${where} attrs.padding`, 0),
});

/** The scale of fixed SRQ uses the stored f32 value as is (ADR 0097). */
const assertStaticQuantizeScale = (value: unknown, where: string): number => {
  const scale = assertFiniteAttr(value, where, "static_quantize scale");
  if (scale < 0 || Math.fround(scale) !== scale) {
    throw new OpContractError(
      `${where}: static_quantize scale must be non-negative and an exact f32 value`,
    );
  }
  return scale;
};
export const STATIC_QUANTIZE_ATTRS: AttrSchema = {
  scale: (value, where) => {
    assertStaticQuantizeScale(value, where);
  },
};
export const staticQuantizeScale = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): number => assertStaticQuantizeScale(attrValue(attrs, "scale"), `${where} attrs.scale`);
