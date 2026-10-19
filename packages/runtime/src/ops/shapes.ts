import { GRU_SCAN_MAX_HIDDEN } from "../codegen/limits.ts";
import { STRIDED_RANK } from "../codegen/strided.ts";
import {
  attentionScale,
  catDim,
  conv1dAttrs,
  conv2dAttrs,
  conv3dAttrs,
  convTranspose1dAttrs,
  cumsumDim,
  deformConv2dAttrs,
  flipDim,
  layerNormAttrs,
  padAttrs,
  permuteDims,
  reduceDim,
  rmsNormEps,
  sliceAttrs,
  softmaxDim,
  stateReadonly,
  stateWindow,
  staticQuantizeScale,
  symPrefixSliceAttrs,
  topkK,
  upsampleBilinear2dAttrs,
} from "./attrs.ts";
import {
  assertArity,
  type OpContract,
  READONLY_ATTENTION_ARITY,
  scalarParamValues,
} from "./contracts.ts";
import { OpContractError } from "./names.ts";

export const numel = (shape: readonly number[]): number =>
  shape.reduce((count, dim) => count * dim, 1);

/**
 * Right-aligned broadcast as in torch. A dimension may only "match" or "be 1 on one side".
 * MUST: do not decide the result with `max(a, b)` — the pair 0 and 1 (`max` gives 1) is 0 in torch.
 */
export const broadcastShapes = (
  a: readonly number[],
  b: readonly number[],
  where: string,
): number[] => {
  const rank = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < rank; i += 1) {
    const da = a[a.length - rank + i] ?? 1;
    const db = b[b.length - rank + i] ?? 1;
    if (da === db || db === 1) {
      out.push(da);
    } else if (da === 1) {
      out.push(db);
    } else {
      throw new OpContractError(
        `${where}: shapes [${a.join(",")}] and [${b.join(",")}] cannot be broadcast right-aligned`,
      );
    }
  }
  return out;
};

/**
 * Extra inputs for ops whose output shape **cannot be derived from the inputs**.
 *
 * reshape / expand have the contract "the output's declared shape is the target shape" (ADR 0011),
 * so they cannot be computed without the declaration. permute needs the axis permutation in attrs.
 */
export type ShapeContext = {
  /** The declared shape after binding (required for reshape / expand). */
  readonly declared?: readonly number[];
  /** The node's attrs (required for permute / layer_norm / softmax / conv1d / sym_prefix_slice). */
  readonly attrs?: Readonly<Record<string, unknown>>;
  /**
   * Symbol bindings (required for sym_prefix_slice). The prefix length is `coeff·sym+offset` and
   * cannot be derived from the input shape.
   *
   * MUST: look up with `Object.hasOwn` only (`bindings[sym] !== undefined` lets `toString` and the
   * like from Object.prototype through, and the arithmetic after that turns into NaN — a
   * cross-cutting invariant).
   */
  readonly bindings?: Readonly<Record<string, number>>;
  /**
   * The node's `states` field ({@link IrNode.states} — key fixed by the contract → slot name).
   * **Being non-empty is itself what identifies "the states form"** (ADR 0067 decision 4 — "the
   * presence of the field decides the form"); like `attrs`, it is context that "passes the node's
   * field as is".
   */
  readonly states?: Readonly<Record<string, string>>;
  /**
   * The slot shapes after binding (slot name → concrete shape including capacity).
   *
   * MUST: required for state-referencing nodes (fails loudly when absent). The slots are allocated
   * by the GenerationContext (ADR 0066 decision 1), so a path without this = one-shot execution
   * cannot run state-referencing nodes — computing it silently as the previous form would be a
   * different computation that does not read the past KV.
   */
  readonly stateShapes?: ReadonlyMap<string, readonly number[]>;
};

/**
 * The rank limit of the strided copy family (permute / expand / sym_prefix_slice / masked_fill).
 *
 * MUST: **checked in the contract layer**. The kernel params are fixed at rank
 * {@link STRIDED_RANK}, and an excess would only become a CodegenError once it reaches the codegen
 * layer (`stridedParams`) — to the user, "the contract check passed but there was an internal
 * error at execution", without saying which input of which op is wrong. Failing here gives a
 * diagnostic with the op and input names, and aligns the accepted range of the CPU reference (whose
 * entry `applyReferenceOpOutputs` goes through {@link computeOutputShape}) with the GPU. Only
 * sym_prefix_slice needs bindings, so the reference entry does not go through this function for it
 * (attrs and dtypes do go through).
 */
const assertStridedRank = (rank: number, what: string, where: string): void => {
  if (rank < 1 || rank > STRIDED_RANK) {
    throw new OpContractError(
      `${where}: ${what} rank ${rank} is outside 1..${STRIDED_RANK} (the limit of the strided kernels)`,
    );
  }
};

const requireDeclared = (
  context: ShapeContext,
  found: OpContract,
  where: string,
): readonly number[] => {
  if (context.declared === undefined) {
    throw new OpContractError(
      `${where}: op '${found.name}' output shape is the declared target shape (ShapeContext.declared is required)`,
    );
  }
  return context.declared;
};

/**
 * The output list of a single-output op (each arm states explicitly that there is one output —
 * ADR 0068 decision 1). Arms of multi-output ops build the list directly and return it.
 */
const sole = (shape: number[]): number[][] => [shape];

/**
 * Whether the node references state slots (the presence of the field decides the form — ADR 0067
 * decision 4).
 */
const referencesStates = (context: ShapeContext): boolean =>
  context.states !== undefined && Object.keys(context.states).length > 0;

/**
 * Looks up the **resolved shape** of the slot that a key of the `states` field points at (ADR 0067
 * decision 4).
 *
 * MUST: a path without resolved shapes fails loudly ({@link ShapeContext.stateShapes}).
 */
const stateSlotShape = (
  context: ShapeContext,
  key: string,
  where: string,
): { readonly name: string; readonly shape: readonly number[] } => {
  const states = context.states;
  if (states === undefined || !Object.hasOwn(states, key)) {
    throw new OpContractError(`${where}: states field has no '${key}'`);
  }
  const name = states[key];
  const shape = context.stateShapes?.get(name);
  if (shape === undefined) {
    throw new OpContractError(
      `${where}: state slot '${name}' shape is not resolved` +
        ` (running a node that references state requires a GenerationContext — ADR 0066 decision 1)`,
    );
  }
  return { name, shape };
};

/**
 * The physical form of a state slot (①②③ of ADR 0067 decision 4). A slot is **fixed at
 * `[B, Hkv, C, D]`**, its B / Hkv / D match this step's chunk (`ins`), and for sliding (attrs
 * `window`) `window ≤ C` must hold.
 *
 * MUST: fail here, as the **state extension** of the shape check that otherwise looks only at
 * ordinary values. A slot mix-up (referencing another layer's slot) is an OOB access when the
 * capacity differs, and a silent misread when the capacity is the same.
 * NOTE: `pastLength + queryLength ≤ C` for a full slot needs the logical lengths at run time, so it
 * is on the context side (④ of ADR 0067 decision 4) — it cannot be checked here.
 */
const assertStateSlotForm = (
  slot: { readonly name: string; readonly shape: readonly number[] },
  ins: readonly number[],
  window: number | undefined,
  where: string,
): void => {
  const show = `state slot '${slot.name}' [${slot.shape.join(",")}] / input [${ins.join(",")}]`;
  if (slot.shape.length !== 4) {
    throw new OpContractError(`${where}: ${show} — the slot must be rank-4 [B,Hkv,C,D]`);
  }
  if (slot.shape[0] !== ins[0] || slot.shape[1] !== ins[1] || slot.shape[3] !== ins[3]) {
    throw new OpContractError(`${where}: ${show} — slot and input B / Hkv / D mismatch`);
  }
  if (window !== undefined && window > slot.shape[2]) {
    throw new OpContractError(
      `${where}: ${show} — attrs.window ${window} exceeds the slot capacity ${slot.shape[2]}`,
    );
  }
};

/**
 * Whether this node is a **readonly (past-only) states-form attention** (ADR 0096 stage 2 §1.2).
 * Decided by three things — "the op can have readonly" × "the states field is non-empty" ×
 * "`attrs.readonly` is true" — and the arity check and the shape computation read **the same single
 * predicate**.
 */
const isReadonlyStateAttention = (
  found: OpContract,
  context: ShapeContext,
  where: string,
): boolean =>
  found.readonlyArity !== undefined &&
  referencesStates(context) &&
  stateReadonly(context.attrs ?? {}, where);

/**
 * The form of a readonly states-form attention (ADR 0096 stage 2 §1.2).
 *
 * ins is only q, so the `[B,Hkv,C,D]` is matched **between the slots** and against q: q is
 * `[B,H,1,D]` (**M = 1 MUST** — stage 2 reads only "the one row at logical position P−1"), the
 * slots have the same k / v shape with B and D matching q, `H % Hkv == 0`, and for sliding
 * `window ≤ C`. The output has q's shape.
 *
 * MUST: reject anything but `M = 1` here. A readonly form with M > 1 requires semantics other than
 * "one query row looks at the whole past" (a different past per row), but the kernel looks only at
 * row 0's past — letting it through would make the second and later rows read **the same column
 * range**, a silent wrong value.
 */
const readonlyStateAttentionShape = (
  q: readonly number[],
  context: ShapeContext,
  where: string,
): number[] => {
  const kSlot = stateSlotShape(context, "k", where);
  const vSlot = stateSlotShape(context, "v", where);
  const show = `q [${q.join(",")}] / slot '${kSlot.name}' [${kSlot.shape.join(",")}]`;
  if (q.length !== 4) {
    throw new OpContractError(
      `${where}: readonly attention accepts only rank-4 q[B,H,1,D]: [${q.join(",")}]`,
    );
  }
  if (kSlot.shape.length !== 4 || vSlot.shape.length !== 4) {
    throw new OpContractError(
      `${where}: readonly attention slots must be rank-4 [B,Hkv,C,D] ${show}` +
        ` / slot '${vSlot.name}' [${vSlot.shape.join(",")}]`,
    );
  }
  // MUST: the k / v slots have **the same shape** (the same reason as the states form — a pair with
  // different capacities wraps on one side first).
  if (kSlot.shape.some((dim, axis) => dim !== vSlot.shape[axis])) {
    throw new OpContractError(
      `${where}: readonly attention k / v slots differ in shape ('${kSlot.name}' [${kSlot.shape.join(
        ",",
      )}] / '${vSlot.name}' [${vSlot.shape.join(",")}])`,
    );
  }
  if (q[0] !== kSlot.shape[0] || q[3] !== kSlot.shape[3]) {
    throw new OpContractError(`${where}: readonly attention B / D mismatch ${show}`);
  }
  if (q[0] < 1) {
    throw new OpContractError(`${where}: readonly attention B is not positive ${show}`);
  }
  const kvHeads = kSlot.shape[1];
  if (kvHeads < 1 || q[1] < kvHeads || q[1] % kvHeads !== 0) {
    throw new OpContractError(
      `${where}: readonly attention H ${q[1]} is not a positive integer multiple of Hkv ${kvHeads}` +
        ` (GQA needs H % Hkv == 0 and H ≥ Hkv ≥ 1 — ADR 0067 decision 1) ${show}`,
    );
  }
  if (q[2] !== 1) {
    throw new OpContractError(
      `${where}: readonly attention needs q M (axis 2) of exactly 1 (${q[2]}) — ` +
        "it reads only the one row at logical position P−1 (ADR 0096 stage 2 §1.2)",
    );
  }
  const window = stateWindow(context.attrs ?? {}, where);
  if (window !== undefined && window > kSlot.shape[2]) {
    throw new OpContractError(
      `${where}: ${show} — attrs.window ${window} exceeds the slot capacity ${kSlot.shape[2]}`,
    );
  }
  return [...q];
};

/**
 * Computes the **list of shapes in output slot order** from the input shapes after binding (ADR
 * 0068 decision 1). The list length equals the output count the contract declares (the list length
 * of the output dtype mapping) — only `topk` returns 2 (values + indices — ADR 0068 decision 3),
 * only `state_append` returns **an empty list** (an effect op that defines no value — ADR 0067
 * decision 5), and everything else returns 1.
 */
export const computeOutputShape = (
  found: OpContract,
  inputShapes: readonly (readonly number[])[],
  where: string,
  context: ShapeContext = {},
): number[][] => {
  // The readonly (past-only) form has only one ins, q (ADR 0096 stage 2 §1.2). The contract layer
  // rejects it too, but direct calls from the CPU reference and the conformance table come through
  // here (a double net, as for the mask rejection).
  if (isReadonlyStateAttention(found, context, where)) {
    if (inputShapes.length !== READONLY_ATTENTION_ARITY) {
      throw new OpContractError(
        `${where}: readonly attention takes exactly ${READONLY_ATTENTION_ARITY} input shapes` +
          ` (${inputShapes.length} given)`,
      );
    }
  } else {
    assertArity(found, inputShapes.length, "input shape count", where);
  }
  switch (found.kind) {
    case "unary":
      // MUST: check the ranges of the scalar attrs and **invariants that span keys** (clamp's
      // min <= max) here. This computation is the only common path every node goes through, and the
      // attrs schema can express only per-key checks (planGraph in src/runtime/plan.ts calls this
      // for every node).
      scalarParamValues(found, context.attrs ?? {}, where);
      return sole([...inputShapes[0]]);
    case "cast":
      return sole([...inputShapes[0]]);
    case "binary":
      return sole(broadcastShapes(inputShapes[0], inputShapes[1], `${where} (${found.name})`));
    case "where": {
      // As in torch, the three broadcast right-aligned (the condition widens by the same rule as
      // the values).
      const [cond, a, b] = inputShapes;
      const label = `${where} (${found.name})`;
      return sole(broadcastShapes(broadcastShapes(cond, a, label), b, label));
    }
    case "cumsum": {
      const shape = inputShapes[0];
      const dim = cumsumDim(context.attrs ?? {}, where);
      // MUST: nothing but the last dimension is accepted (the same reason as softmax — the row
      // kernel assumes the reduced axis is contiguous, and letting it through would silently fold a
      // different axis).
      if (shape.length < 1 || dim !== shape.length - 1) {
        throw new OpContractError(
          `${where}: cumsum accepts only the last dimension (attrs.dim=${dim} / input [${shape.join(",")}])`,
        );
      }
      // An axis of length 0 passes through (the identity of a prefix sum is 0; an empty row stays
      // empty).
      return sole([...shape]);
    }
    case "matmul": {
      const [a, b] = inputShapes;
      if (a.length !== 2 || b.length !== 2) {
        throw new OpContractError(
          `${where}: matmul accepts only rank-2 × rank-2 ([${a.join(",")}] × [${b.join(",")}])`,
        );
      }
      if (a[1] !== b[0]) {
        throw new OpContractError(
          `${where}: matmul contraction dimension mismatch [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      return sole([a[0], b[1]]);
    }
    case "bmm": {
      const [a, b] = inputShapes;
      // MUST: rank-2 does not pass (it is matmul's job). Sharing would let "the form without the
      // batch axis" pass under the same op name, and a B mix-up would slip through the shape check.
      if (a.length !== 3 || b.length !== 3) {
        throw new OpContractError(
          `${where}: bmm accepts only rank-3 × rank-3 (rank-2 is matmul): [${a.join(",")}] × [${b.join(
            ",",
          )}]`,
        );
      }
      if (a[0] !== b[0]) {
        throw new OpContractError(
          `${where}: bmm batch dimension mismatch [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      if (a[2] !== b[1]) {
        throw new OpContractError(
          `${where}: bmm contraction dimension mismatch [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      return sole([a[0], a[1], b[2]]);
    }
    case "gather": {
      const [src, index] = inputShapes;
      // The contract is "last dimension fixed" — the leading dimensions of src and index match
      // exactly and only the last dimension is free. Narrower than torch's general gather (other
      // axes may be at most src's length), but enough for the measured forms.
      if (src.length === 0 || index.length !== src.length) {
        throw new OpContractError(
          `${where}: gather needs src and index of the same rank (1 or more): [${src.join(",")}] / [${index.join(
            ",",
          )}]`,
        );
      }
      const mismatch = src.findIndex((dim, d) => d < src.length - 1 && dim !== index[d]);
      if (mismatch >= 0) {
        throw new OpContractError(
          `${where}: gather leading dimension ${mismatch} mismatch [${src.join(",")}] / [${index.join(
            ",",
          )}]`,
        );
      }
      // The output has index's shape (values are taken from src). The range of the indices depends
      // on run-time data, so the shape contract does not check it (policy: src/kernels/gather.ts and
      // reference/ops.ts).
      return sole([...index]);
    }
    case "rowReduce": {
      const shape = inputShapes[0];
      if (shape.length === 0) {
        throw new OpContractError(
          `${where}: reduce input must be rank 1 or more (a scalar cannot be reduced)`,
        );
      }
      const dim = reduceDim(context.attrs ?? {}, where);
      // MUST: negatives and out-of-rank values fail loudly (normalising negative axis notation is
      // the exporter boundary's responsibility; filling in `% rank` here would silently absorb "a
      // mismatch between the declaration and the actual rank" into another axis).
      if (dim >= shape.length) {
        throw new OpContractError(
          `${where}: op '${found.name}' attrs.dim=${dim} is out of range for rank ${shape.length}`,
        );
      }
      // amax/amin over an empty axis has no definable identity (torch rejects it too). sum gives 0.
      if (shape[dim] === 0 && found.name !== "sum") {
        throw new OpContractError(`${where}: op '${found.name}' cannot reduce an axis of length 0`);
      }
      return sole([...shape.slice(0, dim), ...shape.slice(dim + 1)]);
    }
    case "argmax": {
      const shape = inputShapes[0];
      // Unlike the reduce family, the axis is not in attrs but **fixed to the last dimension** (the
      // absence of the field states "other axes are not in the vocabulary" — ADR 0068 decision 2).
      if (shape.length === 0) {
        throw new OpContractError(
          `${where}: argmax input must be rank 1 or more (a scalar cannot be reduced)`,
        );
      }
      const last = shape.length - 1;
      // An axis of length 0 has no "index of the maximum" (torch rejects it too). The same narrowing
      // as amax / amin; it stops, on the contract side, the form where the kernel's sentinel index
      // leaks into the output.
      if (shape[last] === 0) {
        throw new OpContractError(`${where}: argmax cannot reduce a last dimension of length 0`);
      }
      // **Rank-preserving** (a fixed form with no `keepdim`-like field — collapses the last
      // dimension to 1).
      return sole([...shape.slice(0, last), 1]);
    }
    case "topk": {
      const shape = inputShapes[0];
      // The axis is fixed to the last dimension (the same narrowing as argmax). Multi-output, so the
      // only arm that **builds the list directly and returns it**.
      if (shape.length === 0) {
        throw new OpContractError(
          `${where}: topk input must be rank 1 or more (a scalar cannot be reduced)`,
        );
      }
      const last = shape.length - 1;
      const k = topkK(context.attrs ?? {}, where);
      // MUST: the accepted domain is `1 ≤ k ≤ last dimension` (ADR 0068 decision 3). The attrs
      // schema checks the lower bound and this checks the upper bound (it cannot be decided without
      // the input rank). A last dimension of length 0 fails right here against k ≥ 1 — one predicate
      // rejects the forms where "the k largest" cannot be defined.
      if (k > shape[last]) {
        throw new OpContractError(
          `${where}: topk attrs.k=${k} exceeds the last dimension ${shape[last]} (input [${shape.join(
            ",",
          )}])`,
        );
      }
      // Values and indices have the same shape (`[…, k]`). The rank is preserved; only the last
      // dimension becomes k.
      const outShape = [...shape.slice(0, last), k];
      return [outShape, [...outShape]];
    }
    case "reshape": {
      const target = requireDeclared(context, found, where);
      const source = inputShapes[0];
      // The contract is only that element counts match (element order unchanged). Loosening it
      // would make the size of the aliased real buffer disagree with the declared shape, and
      // readback would read out of range.
      if (numel(source) !== numel(target)) {
        throw new OpContractError(
          `${where}: reshape element count mismatch [${source.join(",")}] → [${target.join(",")}]`,
        );
      }
      return sole([...target]);
    }
    case "expand": {
      const target = requireDeclared(context, found, where);
      const source = inputShapes[0];
      assertStridedRank(source.length, "expand input", where);
      assertStridedRank(target.length, "expand output", where);
      if (target.length < source.length) {
        throw new OpContractError(
          `${where}: expand cannot lower the rank [${source.join(",")}] → [${target.join(",")}]`,
        );
      }
      // Right-aligned; each input dimension either "matches the target" or "is length 1 (replicated
      // with stride 0)".
      const offset = target.length - source.length;
      source.forEach((extent, index) => {
        if (extent !== 1 && extent !== target[offset + index]) {
          throw new OpContractError(
            `${where}: expand cannot expand dimension ${index} (${extent}), which is not of length 1, to ${
              target[offset + index]
            }`,
          );
        }
      });
      return sole([...target]);
    }
    case "slice": {
      const source = inputShapes[0];
      // Executed by reusing the strided read copy family (ADR 0014), so the rank limit is the same.
      assertStridedRank(source.length, "slice input", where);
      const { dim, start, end } = sliceAttrs(context.attrs ?? {}, where);
      if (dim >= source.length) {
        throw new OpContractError(
          `${where}: slice dim ${dim} is outside input rank ${source.length}`,
        );
      }
      // MUST: out-of-range slices do not pass. On the GPU they would read neighbouring values
      // (another row, another buffer) without an exception, and only the shape check would pass "as
      // declared".
      if (end > source[dim]) {
        throw new OpContractError(
          `${where}: slice end ${end} exceeds axis ${dim} length ${source[dim]}`,
        );
      }
      // MUST: an invariant that spans keys (the same split as clamp's min <= max) — the attrs schema
      // can express only per-key checks. Allowing an inversion makes the length negative and only
      // the element count passes, with 0.
      if (start > end) {
        throw new OpContractError(`${where}: slice start ${start} exceeds end ${end}`);
      }
      const out = [...source];
      out[dim] = end - start;
      return sole(out);
    }
    case "cat": {
      const dim = catDim(context.attrs ?? {}, where);
      const first = inputShapes[0];
      // Executed by the strided write copy family (ADR 0014) — the output-side strides go into
      // params, so the rank limit applies to the output (= same rank as the inputs).
      assertStridedRank(first.length, "cat input", where);
      if (dim >= first.length) {
        throw new OpContractError(`${where}: cat dim ${dim} is outside input rank ${first.length}`);
      }
      let total = 0;
      inputShapes.forEach((shape, index) => {
        if (shape.length !== first.length) {
          throw new OpContractError(
            `${where}: cat input ${index} rank ${shape.length} differs from input 0 rank ${first.length}`,
          );
        }
        // MUST: every axis **other than** the concatenation axis must match (as in torch). Loosening
        // it would leave part of the output unwritten by any input, breaking the full-write
        // invariant.
        shape.forEach((extent, axis) => {
          if (axis !== dim && extent !== first[axis]) {
            throw new OpContractError(
              `${where}: cat input ${index} [${shape.join(",")}] differs from input 0 [${first.join(
                ",",
              )}] on axis ${axis} (the concatenation axis is ${dim})`,
            );
          }
        });
        total += shape[dim];
      });
      const out = [...first];
      // Output axis length = sum of the input axis lengths. This rule itself guarantees "all inputs
      // together cover the whole output" (full-write — ADR 0014), and the executor matches the sum of
      // the write positions against it.
      out[dim] = total;
      return sole(out);
    }
    case "pad": {
      const source = inputShapes[0];
      if (source.length < 1) {
        throw new OpContractError(
          `${where}: pad input must be rank 1 or more (it needs a last dimension)`,
        );
      }
      const { left, right } = padAttrs(context.attrs ?? {}, where);
      const out = [...source];
      out[out.length - 1] = source[source.length - 1] + left + right;
      return sole(out);
    }
    case "flip": {
      const source = inputShapes[0];
      const dim = flipDim(context.attrs ?? {}, where);
      if (source.length < 1 || dim >= source.length) {
        throw new OpContractError(
          `${where}: flip dim ${dim} is outside input rank ${source.length}`,
        );
      }
      // Reversal does not change the shape (identity shape rule).
      return sole([...source]);
    }
    case "symPrefixSlice": {
      const source = inputShapes[0];
      // The output rank equals the input's (it only cuts the head of each axis), so checking the
      // input side is enough.
      assertStridedRank(source.length, "sym_prefix_slice input", where);
      const { sym, slices } = symPrefixSliceAttrs(context.attrs ?? {}, where);
      const bindings = context.bindings;
      if (bindings === undefined || !Object.hasOwn(bindings, sym)) {
        throw new OpContractError(
          `${where}: sym_prefix_slice sym '${sym}' is not bound (ShapeContext.bindings)`,
        );
      }
      const bound = bindings[sym];
      const out = [...source];
      for (const slice of slices) {
        if (slice.dim >= source.length) {
          throw new OpContractError(
            `${where}: sym_prefix_slice dim ${slice.dim} is outside input rank ${source.length}`,
          );
        }
        const length = slice.coeff * bound + slice.offset;
        // MUST: no prefix longer than the constant side (the Tmax form). The excess would read out
        // of the constant buffer's range, and the GPU would return neighbouring values without an
        // exception.
        if (length > source[slice.dim]) {
          throw new OpContractError(
            `${where}: sym_prefix_slice prefix length ${slice.coeff}·${sym}+${slice.offset}=${length} exceeds the constant dimension ${
              source[slice.dim]
            } (Tmax exceeded)`,
          );
        }
        out[slice.dim] = length;
      }
      return sole(out);
    }
    case "linear": {
      const [x, weight, bias] = inputShapes;
      if (x.length < 1 || weight.length !== 2 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: linear expects x[…,in] × W[out,in] + b[out] (rank ≥ 1 / 2 / 1): [${x.join(
            ",",
          )}] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [out, inFeatures] = weight;
      if (x[x.length - 1] !== inFeatures) {
        throw new OpContractError(
          `${where}: linear input feature count mismatch [${x.join(",")}] × [${weight.join(",")}]`,
        );
      }
      if (bias[0] !== out) {
        throw new OpContractError(
          `${where}: linear bias length ${bias[0]} differs from the output feature count ${out}`,
        );
      }
      return sole([...x.slice(0, -1), out]);
    }
    case "layerNorm": {
      const [x, weight, bias] = inputShapes;
      const { normalizedShape } = layerNormAttrs(context.attrs ?? {}, where);
      // The contract is "last dimension only" (attrs already checked). Here the actual shape's tail
      // is matched.
      if (x.length < 1 || x[x.length - 1] !== normalizedShape[0]) {
        throw new OpContractError(
          `${where}: layer_norm normalized_shape [${normalizedShape.join(",")}] differs from the last dimension of input [${x.join(
            ",",
          )}]`,
        );
      }
      for (const [name, shape] of [
        ["weight", weight],
        ["bias", bias],
      ] as const) {
        if (shape.length !== 1 || shape[0] !== normalizedShape[0]) {
          throw new OpContractError(
            `${where}: layer_norm ${name} [${shape.join(",")}] differs from normalized_shape [${normalizedShape.join(
              ",",
            )}]`,
          );
        }
      }
      return sole([...x]);
    }
    case "staticQuantize":
      staticQuantizeScale(context.attrs ?? {}, where);
      return sole([...inputShapes[0]]);
    case "rmsNorm": {
      const [x, weight] = inputShapes;
      // MUST: eps is read here too (so that its range applies on the path that does not go through
      // the attrs schema — direct calls from the CPU reference. The same role as unary's
      // scalarParamValues).
      rmsNormEps(context.attrs ?? {}, where);
      if (x.length < 1) {
        throw new OpContractError(
          `${where}: rms_norm input must be rank 1 or more (it needs a last dimension)`,
        );
      }
      const dim = x[x.length - 1];
      // MUST: an axis of length 0 cannot be reduced (a sum of squares 0 / element count 0 makes the
      // mean 0/0 = NaN — the same narrowing as softmax).
      if (dim === 0) {
        throw new OpContractError(`${where}: rms_norm cannot normalise an axis of length 0`);
      }
      // The source of truth for the normalised length is **the weight's length** (no
      // normalized_shape field in attrs — ADR 0017).
      if (weight.length !== 1 || weight[0] !== dim) {
        throw new OpContractError(
          `${where}: rms_norm weight [${weight.join(",")}] does not match input [${x.join(
            ",",
          )}]: expected rank-1 of the last dimension length ${dim}`,
        );
      }
      return sole([...x]);
    }
    case "conv2d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding, dilation, groups } = conv2dAttrs(context.attrs ?? {}, where);
      if (x.length !== 4 || weight.length !== 4 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv2d expects x[B,Cin,H,W] / W[Cout,Cin/groups,Kh,Kw] / b[Cout] (rank 4 / 4 / 1): [${x.join(
            ",",
          )}] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn] = x;
      const [channelsOut, weightIn] = weight;
      // MUST: the same discipline as conv1d — for a form that does not divide, the kernel's
      // `Cin/groups` truncates and the band of input channels it reads silently shifts.
      if (channelsIn % groups !== 0 || channelsOut % groups !== 0) {
        throw new OpContractError(
          `${where}: conv2d groups ${groups} does not divide Cin ${channelsIn} / Cout ${channelsOut}`,
        );
      }
      // MUST: the weight is **[Cout, Cin/groups, Kh, Kw]**. A mix-up with a matching element count
      // ([Cin/groups, Cout, Kh, Kw], or Kh/Kw swapped) can slip through the shape check, so the tests
      // pin asymmetric forms with Cin ≠ Cout and Kh ≠ Kw (the lesson of conv_transpose1d — ADR 0017).
      if (weightIn !== channelsIn / groups) {
        throw new OpContractError(
          `${where}: conv2d weight should be [Cout, Cin/groups, Kh, Kw] (Cin/groups = ${
            channelsIn / groups
          }) but is [${weight.join(",")}] (x is [${x.join(",")}] / groups ${groups})`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv2d bias length ${bias[0]} differs from the output channels ${channelsOut}`,
        );
      }
      // The spatial axes apply the same general form to H / W independently (axis is the subject of
      // the diagnostic).
      const spatial = (axis: 0 | 1, name: string): number => {
        const length = x[2 + axis];
        const kernel = weight[2 + axis];
        // MUST: for the same reason as conv1d, K = 0 fails before the output length formula
        // (`dilation·(K−1)` turns negative and the output becomes longer than the input — the params
        // side rejects it as a positive integer).
        if (kernel < 1) {
          throw new OpContractError(
            `${where}: conv2d kernel length K${name.toLowerCase()} must be a positive integer (weight [${weight.join(
              ",",
            )}])`,
          );
        }
        const span = length + 2 * padding[axis] - dilation[axis] * (kernel - 1) - 1;
        if (span < 0) {
          throw new OpContractError(
            `${where}: conv2d input ${name} ${length} (padding ${padding[axis]}) is shorter than the kernel span with dilation ${
              dilation[axis]
            }: ${dilation[axis] * (kernel - 1) + 1}`,
          );
        }
        return Math.floor(span / stride[axis]) + 1;
      };
      return sole([x[0], channelsOut, spatial(0, "H"), spatial(1, "W")]);
    }
    case "conv3d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding, dilation, groups } = conv3dAttrs(context.attrs ?? {}, where);
      // MUST: x is unbatched rank 4 only (ADR 0118 decision 1). Accepting batched rank 5 could let
      // a mix-up that reads the leading axis as Cin slip through with a matching element count.
      if (x.length !== 4 || weight.length !== 5 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv3d expects x[Cin,T,H,W] / W[Cout,Cin/groups,Kt,Kh,Kw] / b[Cout] (rank 4 / 5 / 1 — x is unbatched): [${x.join(
            ",",
          )}] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [channelsIn] = x;
      const [channelsOut, weightIn] = weight;
      // MUST: the same discipline as conv1d / conv2d — for a form that does not divide, `Cin/groups`
      // truncates and the band of input channels read silently shifts.
      if (channelsIn % groups !== 0 || channelsOut % groups !== 0) {
        throw new OpContractError(
          `${where}: conv3d groups ${groups} does not divide Cin ${channelsIn} / Cout ${channelsOut}`,
        );
      }
      // MUST: the weight is **[Cout, Cin/groups, Kt, Kh, Kw]**. A mix-up with a matching element
      // count can slip through the shape check, so the tests pin asymmetric forms with Cin ≠ Cout
      // and Kt ≠ Kh ≠ Kw.
      if (weightIn !== channelsIn / groups) {
        throw new OpContractError(
          `${where}: conv3d weight should be [Cout, Cin/groups, Kt, Kh, Kw] (Cin/groups = ${
            channelsIn / groups
          }) but is [${weight.join(",")}] (x is [${x.join(",")}] / groups ${groups})`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv3d bias length ${bias[0]} differs from the output channels ${channelsOut}`,
        );
      }
      // The 3 axes (T / H / W) apply the same general form independently (axis is the attrs
      // component index, name the subject of the diagnostic).
      const axisLength = (axis: 0 | 1 | 2, name: string): number => {
        const length = x[1 + axis];
        const kernel = weight[2 + axis];
        // MUST: for the same reason as conv1d / conv2d, K = 0 fails before the output length formula
        // (`dilation·(K−1)` turns negative and the output becomes longer than the input — the params
        // side rejects it as a positive integer).
        if (kernel < 1) {
          throw new OpContractError(
            `${where}: conv3d kernel length K${name.toLowerCase()} must be a positive integer (weight [${weight.join(
              ",",
            )}])`,
          );
        }
        const span = length + 2 * padding[axis] - dilation[axis] * (kernel - 1) - 1;
        if (span < 0) {
          throw new OpContractError(
            `${where}: conv3d input ${name} ${length} (padding ${padding[axis]}) is shorter than the kernel span with dilation ${
              dilation[axis]
            }: ${dilation[axis] * (kernel - 1) + 1}`,
          );
        }
        return Math.floor(span / stride[axis]) + 1;
      };
      return sole([channelsOut, axisLength(0, "T"), axisLength(1, "H"), axisLength(2, "W")]);
    }
    // safe_softmax has the same shape rule and attrs as softmax (only the empty-row values differ —
    // ADR 0044).
    case "softmax":
    case "safeSoftmax": {
      const shape = inputShapes[0];
      const dim = softmaxDim(context.attrs ?? {}, where);
      // MUST: a general dim is not accepted "to be implemented someday". Anything but the last
      // dimension breaks the row kernel's premise (a contiguous reduced axis), and letting it through
      // would silently fold a different axis.
      if (shape.length < 1 || dim !== shape.length - 1) {
        throw new OpContractError(
          `${where}: ${found.name} accepts only the last dimension (attrs.dim=${dim} / input [${shape.join(",")}])`,
        );
      }
      if (shape[shape.length - 1] === 0) {
        // softmax over an empty axis has no definable amax identity (the same reason as row reduce).
        throw new OpContractError(`${where}: ${found.name} cannot reduce an axis of length 0`);
      }
      return sole([...shape]);
    }
    case "attention": {
      // MUST: scale is read here too (so that its range applies even to direct calls from the CPU
      // reference that skip the attrs schema. The same role as rms_norm's eps / unary's
      // scalarParamValues).
      attentionScale(context.attrs ?? {}, where);
      if (isReadonlyStateAttention(found, context, where)) {
        return sole(readonlyStateAttentionShape(inputShapes[0], context, where));
      }
      const [q, k, v, mask] = inputShapes;
      const show = `[${q.join(",")}] / [${k.join(",")}] / [${v.join(",")}]`;
      if (q.length !== 4 || k.length !== 4 || v.length !== 4) {
        throw new OpContractError(
          `${where}: attention accepts only rank-4 q[B,H,M,D] / k[B,Hkv,N,D] / v[B,Hkv,N,D]: ${show}`,
        );
      }
      // MUST: B **matches exactly** (looking only at the product lets a B/H mix-up through — the
      // kernel folds B·H into one axis, so a form that does not even show in the values can be made).
      if (q[0] !== k[0] || q[0] !== v[0]) {
        throw new OpContractError(`${where}: attention axis 0 (B) mismatch ${show}`);
      }
      // MUST: `B >= 1` is common to both forms (before the states-form branch). B = 0 satisfies the
      // exact match above and slips through, and then the row-block planner of the execution phase
      // fails with a diagnostic that **does not name attention**, "bytes per row must be a positive
      // integer (0)" (planRowBlocks in fusion.ts).
      if (q[0] < 1) {
        throw new OpContractError(`${where}: attention B ${q[0]} is not positive ${show}`);
      }
      // MUST: the Hkv of k / v also **matches exactly** (GQA loosens only the relation to q — ADR
      // 0067 decision 1: "exact B match, equal Hkv of k and v, equal D and the N = 0 rejection stay as
      // detectors of mix-ups").
      if (k[1] !== v[1]) {
        throw new OpContractError(`${where}: attention Hkv (axis 1 of k / v) mismatch ${show}`);
      }
      // MUST: `Hkv ≥ 1` is checked **before** the equality short-circuit below — `(H,Hkv) = (0,0)`
      // does not satisfy `q[1] !== k[1]`, so it never reaches the divisibility branch and "IR that
      // dropped the whole head axis" slips through. `H = 0` alone pairs with `Hkv ≥ 1`, so the
      // `H ≥ Hkv` branch below rejects it (the mirror of `_attention` on the Python side).
      if (k[1] < 1) {
        throw new OpContractError(
          `${where}: attention Hkv ${k[1]} is not positive (H must be a positive integer multiple of Hkv — GQA needs` +
            ` H % Hkv == 0 and H ≥ Hkv ≥ 1; ADR 0067 decision 1) ${show}`,
        );
      }
      // GQA = **divisible broadcast** (ADR 0067 decision 1). Only `H % Hkv == 0` and `H ≥ Hkv` are
      // accepted, and `r = H / Hkv` is derived (no attrs field). MQA with Hkv = 1 uses the same
      // formula.
      // MUST: reject `H < Hkv` by a separate condition — `H = 0` satisfies `0 % Hkv == 0`, so
      // checking only divisibility lets "IR that dropped H entirely" through (`Hkv = 0` is rejected
      // first by the `Hkv ≥ 1` branch above).
      // The broadcast always goes kv → q; a form that widens the q side is not in the vocabulary.
      if (q[1] !== k[1] && (q[1] < k[1] || q[1] % k[1] !== 0)) {
        throw new OpContractError(
          `${where}: attention H ${q[1]} is not a positive integer multiple of Hkv ${k[1]} (GQA needs` +
            ` H % Hkv == 0 and H ≥ Hkv — ADR 0067 decision 1) ${show}`,
        );
      }
      // MUST: D is the same for all three (allowing a different length on the v side alone would let
      // a mix-up escape the element count).
      if (q[3] !== k[3] || q[3] !== v[3]) {
        throw new OpContractError(`${where}: attention D (axis 3) mismatch ${show}`);
      }
      // The states form (ADR 0067 decision 4). The head / D rules so far are common to both forms,
      // and they branch from here: the previous form looks at the N of k/v (the full length, past +
      // present), while the states form's ins are **only this step's chunk**, so M agrees across the
      // three and the past is held by the slot capacity C.
      if (referencesStates(context)) {
        // MUST: no mask is taken (causal is fixed and computed by predicate — no mask tensor is
        // materialised). The contract layer rejects it too, but direct calls from the CPU reference
        // and the conformance table come through here.
        if (mask !== undefined) {
          throw new OpContractError(
            `${where}: states-form attention takes no mask (causal is fixed — ADR 0067 decision 4)` +
              ` ${show} + mask [${mask.join(",")}]`,
          );
        }
        // MUST: M **matches exactly** across q / k / v (they share this step's chunk — unlike the free
        // N of the previous form, a mismatch here is nothing but IR that "mixed in another step's
        // k/v").
        if (k[2] !== q[2] || v[2] !== q[2]) {
          throw new OpContractError(
            `${where}: states-form attention needs the same M (axis 2) in q / k / v ${show}`,
          );
        }
        // softmax over an empty axis has no definable amax identity (the same reason as the previous
        // form's N=0 rejection).
        if (q[2] < 1) {
          throw new OpContractError(
            `${where}: states-form attention cannot handle a chunk with M = 0 ${show}`,
          );
        }
        const window = stateWindow(context.attrs ?? {}, where);
        const kSlot = stateSlotShape(context, "k", where);
        const vSlot = stateSlotShape(context, "v", where);
        // MUST: the k / v slots have **the same shape** (pairing slots with different capacities
        // makes a ring where one side wraps first, and the values silently drift).
        if (
          kSlot.shape.length !== vSlot.shape.length ||
          kSlot.shape.some((dim, axis) => dim !== vSlot.shape[axis])
        ) {
          throw new OpContractError(
            `${where}: attention k / v slots differ in shape ('${kSlot.name}' [${kSlot.shape.join(
              ",",
            )}] / '${vSlot.name}' [${vSlot.shape.join(",")}])`,
          );
        }
        assertStateSlotForm(kSlot, k, window, where);
        assertStateSlotForm(vSlot, v, window, where);
        return sole([...q]);
      }
      if (k[2] !== v[2]) {
        throw new OpContractError(`${where}: attention N (axis 2 of k / v) mismatch ${show}`);
      }
      // softmax over an empty axis has no definable amax identity (the same reason as softmax / row
      // reduce).
      if (k[2] === 0) {
        throw new OpContractError(`${where}: attention cannot reduce an N of length 0 ${show}`);
      }
      // M >= 1 is required for the previous form too (the states form explicitly rejects the same
      // form above). M = 0 would fail in planRowBlocks with "the row count must be a positive integer
      // (0)", a message that does not name attention.
      if (q[2] < 1) {
        throw new OpContractError(`${where}: attention cannot handle a q with M = 0 ${show}`);
      }
      if (mask !== undefined) {
        // MUST: the mask is **exactly [1,1,M,N]**. It is an additive term broadcast to B·H, so letting
        // through forms where "only some axes are materialised", such as `[B,1,M,N]` / `[1,H,M,N]`,
        // would make the kernel (which folds B·H into one axis and hands the first batch's mask to
        // every batch) silently apply another batch's mask. Widening it means revising the contract
        // together with the index arithmetic.
        const shown = `${show} + mask [${mask.join(",")}]`;
        if (mask.length !== 4 || mask[0] !== 1 || mask[1] !== 1) {
          throw new OpContractError(
            `${where}: attention mask must be rank-4 [1,1,M,N] (B / H always broadcast): ${shown}`,
          );
        }
        if (mask[2] !== q[2] || mask[3] !== k[2]) {
          throw new OpContractError(`${where}: attention mask M / N do not match q / k ${shown}`);
        }
      }
      return sole([...q]);
    }
    // An effect op that defines no value (ADR 0067 decision 5). Returning **an empty list** without
    // going through `sole` is itself the declaration "0 outputs" (the same style as the multi-output
    // arm building its list).
    case "stateAppend": {
      const x = inputShapes[0];
      if (x.length !== 4) {
        throw new OpContractError(
          `${where}: state_append input must be rank-4 [B,Hkv,M,D]: [${x.join(",")}]`,
        );
      }
      const window = stateWindow(context.attrs ?? {}, where);
      assertStateSlotForm(stateSlotShape(context, "slot", where), x, window, where);
      return [];
    }
    case "embedding": {
      const [weight, index] = inputShapes;
      if (weight.length !== 2) {
        throw new OpContractError(
          `${where}: embedding weight must be rank-2 [V,H]: [${weight.join(",")}]`,
        );
      }
      if (index.length < 1) {
        throw new OpContractError(
          `${where}: embedding index must be rank 1 or more (there is no scalar index)`,
        );
      }
      // The index range 0 <= index < V depends on run-time data, so the shape contract does not check
      // it (out-of-range handling is the ruling in src/kernels/embedding.ts — NaN contamination on the
      // GPU / throw in the CPU reference).
      return sole([...index, weight[1]]);
    }
    case "maskedFill": {
      const [x, mask] = inputShapes;
      // The output has x's shape. The mask also builds right-aligned broadcast strides, so it needs
      // rank 1 or more (a rank-0 mask used to slip through the contract and fail in the codegen
      // layer).
      assertStridedRank(x.length, "masked_fill x", where);
      assertStridedRank(mask.length, "masked_fill mask", where);
      // MUST: the output **always has x's shape** (the mask side is only read with right-aligned
      // broadcast). Using broadcastShapes as is would also let through forms where the mask widens x
      // (mask [4] × x [1]), and the fill value would leak into elements that do not exist. This is an
      // asymmetric check that only looks at "the mask fits into x".
      if (mask.length > x.length) {
        throw new OpContractError(
          `${where}: masked_fill mask rank ${mask.length} exceeds x rank ${x.length} (mask broadcasts right-aligned only)`,
        );
      }
      const offset = x.length - mask.length;
      mask.forEach((extent, index) => {
        if (extent !== 1 && extent !== x[offset + index]) {
          throw new OpContractError(
            `${where}: masked_fill mask [${mask.join(",")}] cannot be broadcast right-aligned to x [${x.join(
              ",",
            )}] (dimension ${index})`,
          );
        }
      });
      return sole([...x]);
    }
    case "conv1d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding, dilation, groups } = conv1dAttrs(context.attrs ?? {}, where);
      if (x.length !== 3 || weight.length !== 3 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv1d expects x[B,Cin,L] / W[Cout,Cin/groups,K] / b[Cout] (rank 3 / 3 / 1): [${x.join(
            ",",
          )}] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn, length] = x;
      const [channelsOut, weightIn, kernel] = weight;
      // MUST: the group split must divide on both sides; that is the contract (depthwise is groups =
      // Cin = Cout). Letting a non-dividing form through makes the kernel's `Cin/groups` truncate,
      // and the input channels it reads silently shift.
      if (channelsIn % groups !== 0 || channelsOut % groups !== 0) {
        throw new OpContractError(
          `${where}: conv1d groups ${groups} does not divide Cin ${channelsIn} / Cout ${channelsOut}`,
        );
      }
      if (weightIn !== channelsIn / groups) {
        throw new OpContractError(
          `${where}: conv1d weight should be [Cout, Cin/groups, K] (Cin/groups = ${
            channelsIn / groups
          }) but is [${weight.join(",")}] (x is [${x.join(",")}] / groups ${groups})`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv1d bias length ${bias[0]} differs from the output channels ${channelsOut}`,
        );
      }
      // MUST: K = 0 is not fed into the output length formula (`dilation·(K−1)` turns negative and a
      // form whose output is **longer** than the input is derived). The params (kernels/conv1d.ts)
      // reject it as a positive integer, so if this check were missing, the contract layer and the
      // params layer would split the accepted set.
      if (kernel < 1) {
        throw new OpContractError(
          `${where}: conv1d kernel length K must be a positive integer (weight [${weight.join(",")}])`,
        );
      }
      // The general form with dilation. Even for K=1, d·(K−1) = 0, so it agrees with the old formula.
      const span = length + 2 * padding - dilation * (kernel - 1) - 1;
      if (span < 0) {
        throw new OpContractError(
          `${where}: conv1d input length ${length} (padding ${padding}) is shorter than the kernel span with dilation ${dilation}: ${
            dilation * (kernel - 1) + 1
          }`,
        );
      }
      return sole([x[0], channelsOut, Math.floor(span / stride) + 1]);
    }
    case "convTranspose1d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding } = convTranspose1dAttrs(context.attrs ?? {}, where);
      if (x.length !== 3 || weight.length !== 3 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv_transpose1d expects x[B,Cin,L] / W[Cin,Cout,K] / b[Cout] (rank 3 / 3 / 1): [${x.join(
            ",",
          )}] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn, length] = x;
      const [weightIn, channelsOut, kernel] = weight;
      // MUST: the weight is [Cin, Cout, K], **transposed** relative to conv1d. A mix-up can produce a
      // form with a matching element count (Cin == Cout), so the tests pin asymmetric channel counts
      // (ADR 0015).
      if (weightIn !== channelsIn) {
        throw new OpContractError(
          `${where}: conv_transpose1d weight should be [Cin, Cout, K] (Cin = ${channelsIn}) but is [${weight.join(
            ",",
          )}] (x is [${x.join(",")}])`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv_transpose1d bias length ${bias[0]} differs from the output channels ${channelsOut}`,
        );
      }
      // MUST: only forms whose output length is exactly L·stride (`2p == K − s`) are accepted. The
      // general form `(L−1)·s − 2p + K` would still be linear in the symbolic length L, but all the
      // measurements (the 5 ups of dec) have this form, so it is widened once there is demand — the
      // general form is not silently let through (ADR 0015).
      if (2 * padding !== kernel - stride) {
        throw new OpContractError(
          `${where}: conv_transpose1d accepts only the form 2·padding == K − stride (K ${kernel} / stride ${stride} / padding ${padding} — otherwise the output length is not L·stride)`,
        );
      }
      return sole([x[0], channelsOut, length * stride]);
    }
    case "deformConv2d": {
      const [x, weight, offset, mask, bias] = inputShapes;
      const { padding } = deformConv2dAttrs(context.attrs ?? {}, where);
      if (
        x.length !== 4 ||
        weight.length !== 4 ||
        offset.length !== 4 ||
        mask.length !== 4 ||
        bias.length !== 1
      ) {
        throw new OpContractError(
          `${where}: deform_conv2d expects x[B,Cin,H,W] / W[Cout,Cin,Kh,Kw] / offset[B,2·Kh·Kw,Hout,Wout] / mask[B,Kh·Kw,Hout,Wout] / b[Cout] (rank 4/4/4/4/1): [${x.join(
            ",",
          )}] / [${weight.join(",")}] / [${offset.join(",")}] / [${mask.join(",")}] / [${bias.join(
            ",",
          )}]`,
        );
      }
      const [batch, channelsIn] = x;
      const [channelsOut, weightIn, kernelH, kernelW] = weight;
      // MUST: all static geometry is positive integers (**the same check** as the positive set of the
      // params `deformConv2dParams` — it makes the kernel doc's claim "doubled by the contract check"
      // a fact). In particular a spatial length of 0 can derive even a positive output shape
      // depending on padding, every tap falls into the out-of-range branch, and the result is a
      // silent wrong value of **the bias alone** (the GPU fails in params, but the CPU reference
      // returns that value as a valid result).
      // Hout / Wout are checked by `spatial` below with `length < 1`, so only the static input side
      // is checked here.
      for (const [name, value] of [
        ["batch", batch],
        ["channels_in", channelsIn],
        ["channels_out", channelsOut],
        ["height_in", x[2]],
        ["width_in", x[3]],
        ["kernel_h", kernelH],
        ["kernel_w", kernelW],
      ] as const) {
        if (value < 1) {
          throw new OpContractError(
            `${where}: deform_conv2d ${name} must be a positive integer (x [${x.join(",")}] / W [${weight.join(
              ",",
            )}])`,
          );
        }
      }
      // MUST: no groups field = fixed at 1, so the weight's second axis is Cin itself. A mix-up
      // ([Cin, Cout, Kh, Kw]) can produce a form with a matching element count, so the tests pin
      // Cin ≠ Cout.
      if (weightIn !== channelsIn) {
        throw new OpContractError(
          `${where}: deform_conv2d weight should be [Cout, Cin, Kh, Kw] (Cin = ${channelsIn}) but is [${weight.join(
            ",",
          )}] (x is [${x.join(",")}])`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: deform_conv2d bias length ${bias[0]} differs from the output channels ${channelsOut}`,
        );
      }
      // The spatial length is the general form with stride 1 / dilation 1 (no field = fixed at 1).
      const spatial = (axis: 0 | 1, name: string): number => {
        const kernel = weight[2 + axis];
        const length = x[2 + axis] + 2 * padding[axis] - (kernel - 1);
        if (length < 1) {
          throw new OpContractError(
            `${where}: deform_conv2d input ${name} ${x[2 + axis]} (padding ${
              padding[axis]
            }) is shorter than the kernel span ${kernel}`,
          );
        }
        return length;
      };
      const heightOut = spatial(0, "H");
      const widthOut = spatial(1, "W");
      const taps = kernelH * kernelW;
      // MUST: offset / mask are **only matched** (the source of truth for the output shape is x +
      // weight + attrs). Taking Hout from the offset side would let "IR where only offset has a
      // different shape" slip through.
      // No offset_groups field = fixed at 1, so the channel lengths are exactly 2·Kh·Kw / Kh·Kw.
      for (const [name, shape, channels] of [
        ["offset", offset, 2 * taps],
        ["mask", mask, taps],
      ] as const) {
        if (
          shape[0] !== batch ||
          shape[1] !== channels ||
          shape[2] !== heightOut ||
          shape[3] !== widthOut
        ) {
          throw new OpContractError(
            `${where}: deform_conv2d ${name} should be [${batch},${channels},${heightOut},${widthOut}] (offset_groups = 1) but is [${shape.join(
              ",",
            )}]`,
          );
        }
      }
      return sole([batch, channelsOut, heightOut, widthOut]);
    }
    case "upsampleBilinear2d": {
      const x = inputShapes[0];
      const { outputSize } = upsampleBilinear2dAttrs(context.attrs ?? {}, where);
      if (x.length !== 4) {
        throw new OpContractError(
          `${where}: upsample_bilinear2d accepts only x[B,C,H,W] (rank 4): [${x.join(",")}]`,
        );
      }
      // MUST: spatial axes of length 0 do not pass. The scale is `(in − 1) / (out − 1)`, so in = 0
      // gives a negative scale and reads go outside the input (the GPU returns neighbouring values
      // without an exception).
      for (const [name, axis] of [
        ["H", 2],
        ["W", 3],
      ] as const) {
        if (x[axis] === 0) {
          throw new OpContractError(
            `${where}: upsample_bilinear2d cannot interpolate a spatial axis ${name} of length 0 ([${x.join(
              ",",
            )}])`,
          );
        }
      }
      return sole([x[0], x[1], outputSize[0], outputSize[1]]);
    }
    case "gruScan": {
      const [gi, initial, weight, bias] = inputShapes;
      if (gi.length !== 3 || initial.length !== 2 || weight.length !== 2 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: ${found.name} expects gi[T,N,3H] / h0[N,H] / W_hh[3H,H] / b_hh[3H] (rank 3/2/2/1): [${gi.join(
            ",",
          )}] / [${initial.join(",")}] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [time, batch, gates] = gi;
      const [initialBatch, hidden] = initial;
      // MUST: the single source of truth for the hidden width is h0's last dimension (gi / W_hh /
      // b_hh are **only matched against it**). Taking the same fact from two places would allow a
      // form where a 3H / H mix-up slips through.
      // MUST: the range of H is checked here too (the kernel maps 1 lane = 1 hidden unit, and the
      // limit's source of truth is codegen/limits.ts — a by-design constraint published in
      // docs/limitations.md). Without it in the contract layer, H=0 / H>256 would slip through to
      // the params, and only the CPU reference, which goes through `computeOutputShape`, could run
      // the form.
      if (hidden < 1 || hidden > GRU_SCAN_MAX_HIDDEN) {
        throw new OpContractError(
          `${where}: ${found.name} hidden width H ${hidden} is outside 1..${GRU_SCAN_MAX_HIDDEN} (h0 [${initial.join(
            ",",
          )}] — the limit of the one-lane-per-hidden-unit assignment; ADR 0056 decision 5)`,
        );
      }
      if (batch !== initialBatch) {
        throw new OpContractError(
          `${where}: ${found.name} batch differs between gi [${gi.join(",")}] and h0 [${initial.join(
            ",",
          )}]`,
        );
      }
      if (gates !== 3 * hidden) {
        throw new OpContractError(
          `${where}: ${found.name} gi last dimension ${gates} is not 3·H (H = ${hidden}) (the gates are r / z / n, three in all)`,
        );
      }
      if (weight[0] !== 3 * hidden || weight[1] !== hidden) {
        throw new OpContractError(
          `${where}: ${found.name} W_hh should be [3H, H] = [${3 * hidden},${hidden}] but is [${weight.join(
            ",",
          )}]`,
        );
      }
      if (bias[0] !== 3 * hidden) {
        throw new OpContractError(
          `${where}: ${found.name} b_hh length ${bias[0]} differs from 3·H = ${3 * hidden}`,
        );
      }
      return sole([time, batch, hidden]);
    }
    case "permute": {
      const source = inputShapes[0];
      // The output rank equals the input's (it only reorders), so checking the input side is enough.
      assertStridedRank(source.length, "permute input", where);
      const dims = permuteDims(context.attrs ?? {}, where);
      if (dims.length !== source.length) {
        throw new OpContractError(
          `${where}: permute dims [${dims.join(",")}] do not match input rank ${source.length}`,
        );
      }
      return sole(
        dims.map((dim) => {
          if (dim >= source.length) {
            throw new OpContractError(
              `${where}: permute dims has an axis outside input rank ${source.length}: ${dim}`,
            );
          }
          return source[dim];
        }),
      );
    }
  }
};
