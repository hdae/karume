/**
 * The strategies of `linear` and their selection — one ordered table instead of a chain of branches.
 *
 * `linear` is one op (`[m,k] × [n,k]ᵀ + bias`); what differs is how a kernel computes it. Each way is a
 * **strategy**: the default GEMM skeleton, the GEMV family for few rows, the integer dot products on
 * int8 activations, and so on. Which one a linear takes depends on its shape, on the storage of its
 * weight, on the Session's options and on what the fusion pass declared for it (packed activations, a
 * folded producer).
 *
 * {@link resolveLinearStrategy} is the only place that decides it ({@link selectLinearStrategy} is the
 * same decision for a caller that fails on a refusal). It walks {@link LINEAR_SELECTION}
 * from the top: an entry either refuses the linear (the combination is not implemented — a loud
 * failure, never a silent fall to another path) or names a strategy when the linear is its case. The
 * order is part of the meaning: a linear whose activation arrives packed must be taken by the integer
 * GEMV before the i8a8 path is considered, the float-storage GEMV comes before the byte-plane GEMV,
 * and the default GEMM is last.
 *
 * {@link LINEAR_STRATEGIES} states, per strategy, how far its numbers may stand from the reference
 * kernel and what kind of evidence it carries (the vocabulary of src/runtime/strategy-level.ts) — the
 * vocabulary of ADR 0152 decisions 2 and 3. Nothing selects by level yet; a strategy is still admitted
 * by the Session option named in its row.
 *
 * NOTE: device-free — it imports kernel modules for their eligibility tables only, no GPU module, so
 * the recipe builder, the fusion pass and the estimate can all call it. Not part of the public surface
 * of mod.ts (ADR 0008).
 */

import { type GemmCompute, gemmUsesVec4 } from "../kernels/gemm.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  isFloatStorage,
  LINEAR_GEMV_MAX_ROWS,
  type LinearGemvChoices,
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  type LinearGemvParallelLanes,
  linearGemvParallelLanes,
  linearGemvParallelRowBlock,
  type LinearGemvParallelRows,
  type LinearGemvRowsVariant,
  linearGemvStagedEligible,
  linearGemvUnit,
  type LinearGemvVariant,
} from "../kernels/linear-gemv.ts";
import type { IntegerGemvVariant } from "../kernels/linear-gemv-integer.ts";
import type { PlanesGemvVariant } from "../kernels/linear-gemv-planes.ts";
import { type LinearSplitKSlices, linearSplitKSlices } from "../kernels/linear-splitk.ts";
import { quantizeRowsPrologueEligible } from "../kernels/quantize-rows.ts";
import type { WeightStorage } from "../kernels/weight-storage.ts";
import { numel } from "../ops.ts";
import type { LinearPrologue, PackedActivations } from "./fusion-rule.ts";
import {
  linearIntegerGemvVariant,
  linearPlanesGemvVariant,
  linearTakesI8a8,
} from "./linear-route.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";
import type { LinearGemvReduce } from "./session-types.ts";
import type { StrategyInfo } from "./strategy-level.ts";

export type LinearStrategyId =
  | "integer-gemv"
  | "i8a8"
  | "gemv-m1"
  | "gemv-rows"
  | "gemv-parallel"
  | "gemv-parallel-packed"
  | "gemv-parallel-staged"
  | "gemv-parallel-subgroup32"
  | "planes-gemv"
  | "gemm-splitk"
  | "gemm-f32"
  | "gemm-f16";

/**
 * What each strategy is, by the rule: `exact` only for the reference kernel and the kernels whose
 * numeric contract is its bits; `reorder` only where the recorded evidence is a summation-order band;
 * everything else is an approximation named by its option.
 */
export const LINEAR_STRATEGIES: Readonly<Record<LinearStrategyId, StrategyInfo>> = {
  "gemm-f32": { level: "exact", evidence: "reference", admittedBy: undefined },
  "gemv-m1": { level: "exact", evidence: "mirror", admittedBy: undefined },
  "gemv-rows": { level: "exact", evidence: "mirror", admittedBy: undefined },
  "gemv-parallel": {
    level: "reorder",
    evidence: "reorder",
    admittedBy: 'numerics: linear at "reorder"',
  },
  "gemv-parallel-staged": {
    level: "reorder",
    evidence: "reorder",
    admittedBy: 'numerics: linear at "reorder"',
  },
  "gemv-parallel-subgroup32": {
    level: "reorder",
    evidence: "reorder",
    admittedBy: 'numerics: linear at "reorder" with REORDER_STRATEGIES.linearGemv',
  },
  "planes-gemv": {
    level: "reorder",
    evidence: "reorder",
    admittedBy: 'numerics: linear at "reorder" with REORDER_STRATEGIES.linearGemvPlanes',
  },
  "gemm-splitk": {
    level: "reorder",
    evidence: "reorder",
    admittedBy: "the internal option LINEAR_SPLIT_K",
  },
  "gemv-parallel-packed": {
    level: "approximation",
    evidence: "mirror",
    admittedBy: "packedStaticQuantize: true",
  },
  "integer-gemv": { level: "approximation", evidence: "mirror", admittedBy: "integerGemv: true" },
  i8a8: { level: "approximation", evidence: "mirror", admittedBy: 'linearCompute: "a8"' },
  "gemm-f16": {
    level: "approximation",
    evidence: "composition",
    admittedBy: 'linearCompute: "f16"',
  },
};

/** The Session options a linear's selection reads (resolved values). */
export type LinearKnobs = {
  readonly linearCompute: "f32" | "a8" | "f16";
  readonly linearGemvReduce: LinearGemvReduce;
  readonly integerGemv: boolean;
  readonly linearGemvPlanes: boolean;
  readonly linearGemvRowsThreadTarget: number | undefined;
  /**
   * Whether the split-K GEMM may take the linears it is eligible for (the internal option
   * `LINEAR_SPLIT_K` of session-types.ts — no level selects it; SPEC:
   * docs/spec/numerics.md#strategies-admitted-only-by-a-private-key).
   */
  readonly linearSplitK: boolean;
};

/** Everything the selection of one linear depends on. */
export type LinearFacts = {
  /** The linear's plan (after the fusion pass re-pointed its input for a folded producer). */
  readonly plan: NodePlan;
  /** The resident storage of the weight slot. */
  readonly weightStorage: WeightStorage;
  /** The i4 group length of the slot (`undefined` for any other storage). */
  readonly groupSize: number | undefined;
  readonly knobs: LinearKnobs;
  /**
   * The GEMV choices of the Session's geometry profile (`linearGemv` of the profile the Session
   * resolved — src/kernels/geometry-profile.ts): which sequential variant, how many rows per row
   * block (unless `knobs.linearGemvRowsThreadTarget` replaces the target) and which shapes take the
   * parallel GEMV and how. The recipe builder, the fusion pass and the estimate pass the same value
   * a Session runs with.
   */
  readonly linearGemv: LinearGemvChoices;
  /**
   * The side of the packed-activation pairing the fusion pass declared for this linear, if any
   * (ADR 0105). The fusion pass asks with `{ role: "read" }` before it declares the pairing: "would
   * this linear read packed words?".
   */
  readonly packed: Pick<PackedActivations, "role"> | undefined;
  /**
   * The producer the fusion pass folded into this linear's step ①, if any (perf-ledger F-4). The
   * fusion pass asks with the producer's op before it folds: "would this linear absorb it?".
   */
  readonly prologue: LinearPrologue | undefined;
};

/** The knobs as a consumer holds them — optional Session values — with the omitted values filled in. */
export const linearKnobsOf = (options: {
  readonly linearCompute?: LinearKnobs["linearCompute"] | undefined;
  readonly linearGemvReduce?: LinearGemvReduce | undefined;
  readonly integerGemv?: boolean | undefined;
  readonly linearGemvPlanes?: boolean | undefined;
  readonly linearGemvRowsThreadTarget?: number | undefined;
  readonly linearSplitK?: boolean | undefined;
}): LinearKnobs => ({
  linearCompute: options.linearCompute ?? "f32",
  linearGemvReduce: options.linearGemvReduce ?? "sequential",
  integerGemv: options.integerGemv ?? false,
  linearGemvPlanes: options.linearGemvPlanes ?? false,
  linearGemvRowsThreadTarget: options.linearGemvRowsThreadTarget,
  linearSplitK: options.linearSplitK ?? false,
});

/** The strategy chosen for one linear, with what its kernel is built from. */
export type LinearChoice =
  | { readonly id: "integer-gemv"; readonly variant: IntegerGemvVariant }
  | { readonly id: "i8a8" }
  | { readonly id: "planes-gemv"; readonly variant: PlanesGemvVariant }
  | LinearSplitKChoice
  | LinearGemvChoice
  | LinearGemmChoice;

/**
 * The split-K GEMM: the slice count, and the reference GEMM's output form and i4 group length (the
 * partial is the split of that kernel).
 */
export type LinearSplitKChoice = {
  readonly id: "gemm-splitk";
  readonly slices: LinearSplitKSlices;
  readonly v4: boolean;
  readonly groupSize: number | undefined;
};

/** The default GEMM skeleton: its compute, its output form and the i4 group length of the weight. */
export type LinearGemmChoice = {
  readonly id: "gemm-f32" | "gemm-f16";
  readonly compute: GemmCompute;
  readonly v4: boolean;
  readonly groupSize: number | undefined;
};

/** A member of the GEMV family: which thread owns which output, and how the K reduction is split. */
export type LinearGemvChoice = {
  readonly id:
    | "gemv-m1"
    | "gemv-rows"
    | "gemv-parallel"
    | "gemv-parallel-packed"
    | "gemv-parallel-staged"
    | "gemv-parallel-subgroup32";
  readonly storage: WeightStorage;
  readonly groupSize: number | undefined;
  /** The lane count of the parallel forms (`undefined` for the sequential ones). */
  readonly lanes: LinearGemvParallelLanes | undefined;
  /** The column form of the sequential M = 1 kernel (also the column width of its dispatch). */
  readonly variant: LinearGemvVariant;
  /** The row-block form (`undefined` unless the id is `gemv-rows`). */
  readonly rowsVariant: LinearGemvRowsVariant | undefined;
  /**
   * The rows a thread of the row-blocked parallel kernel holds (`gemv-parallel-staged` at M ≥ 2 of a
   * shape whose profile row carries a row block); absent: one workgroup row per row.
   */
  readonly parallelRows?: LinearGemvParallelRows;
};

/** The facts with what every entry derives from them, computed once per selection. */
type Derived = LinearFacts & {
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /** `linear [x] × [weight]` — the head of a refusal's message. */
  readonly where: string;
  readonly v4: boolean;
  readonly compute: GemmCompute;
  readonly integer: IntegerGemvVariant | undefined;
  readonly planes: PlanesGemvVariant | undefined;
  readonly i8a8: boolean;
};

type SelectionEntry =
  /** Refuses the linear: returns the message of the failure, or `undefined` when it does not apply. */
  | { readonly refuse: (linear: Derived) => string | undefined }
  /** Names a strategy when the linear is its case. */
  | { readonly choose: (linear: Derived) => LinearChoice | undefined };

/** The row counts the GEMV family was measured for. */
const gemvRows = (m: number): boolean => m >= 1 && m <= LINEAR_GEMV_MAX_ROWS;

/**
 * A member of the GEMV family for this storage (one thread = one output column × row block, no shared
 * tile). Which member: the parallel forms only when the resolved `linearGemvReduce` asks for them
 * (linear at the level `reorder`) and the profile's table has lanes for the shape; otherwise M = 1
 * takes the column form and M ≥ 2 the row-block form.
 *
 * MUST: a choice that changes numbers happens only when explicitly requested — at the level `exact`
 * (`linearGemvReduce: "sequential"`) the reference path's kernel and WGSL stay as they are.
 * The staged-activation form (K-90) is bit-identical to the plain parallel one, so it is taken wherever
 * the profile's table marks the shape, without an option of its own. So is its row-blocked form at
 * M ≥ 2 (`parallelRows`), which a shape takes where its profile row carries a row block — whether or
 * not the row is staged at M = 1.
 * The subgroup form exists for i2 / i4 / i8 storage: a float-stored weight under
 * `"parallel-subgroup32"` takes the plain or staged parallel form, which sums in the same order.
 */
const gemvChoice = (
  linear: Derived,
  storage: WeightStorage,
  groupSize: number | undefined,
  packed: LinearFacts["packed"],
): LinearGemvChoice => {
  const { m, n, k, knobs, linearGemv } = linear;
  const subgroup = knobs.linearGemvReduce === "parallel-subgroup32" && !isFloatStorage(storage);
  const lanes =
    knobs.linearGemvReduce !== "sequential"
      ? linearGemvParallelLanes(linearGemv, storage, m, n, k, groupSize)
      : undefined;
  const rowsVariant =
    m === 1 || lanes !== undefined
      ? undefined
      : defaultLinearGemvRowsVariant(linearGemv, storage, m, n, knobs.linearGemvRowsThreadTarget);
  const variant = rowsVariant ?? defaultLinearGemvVariant(linearGemv, { storage, n, k });
  // Only the parallel (non-subgroup) form has a packed-activation kernel; past the refusal above a
  // packed linear always has lanes and no subgroup reduction.
  const parallelRows =
    lanes !== undefined && !subgroup && packed === undefined
      ? linearGemvParallelRowBlock(linearGemv, storage, m, n, k, groupSize)
      : undefined;
  const staged =
    lanes !== undefined &&
    !subgroup &&
    packed === undefined &&
    (parallelRows !== undefined ||
      linearGemvStagedEligible(linearGemv, storage, m, n, k, groupSize) !== undefined);
  const id: LinearGemvChoice["id"] =
    lanes !== undefined
      ? subgroup
        ? "gemv-parallel-subgroup32"
        : packed !== undefined
          ? "gemv-parallel-packed"
          : staged
            ? "gemv-parallel-staged"
            : "gemv-parallel"
      : rowsVariant === undefined
        ? "gemv-m1"
        : "gemv-rows";
  return {
    id,
    storage,
    groupSize,
    lanes,
    variant,
    rowsVariant,
    ...(parallelRows === undefined ? {} : { parallelRows }),
  };
};

/**
 * The selection, in order. The comments keep each entry's reason; the conditions are the ones the
 * kernels' numeric contracts and binding layouts require.
 */
export const LINEAR_SELECTION: readonly SelectionEntry[] = [
  {
    refuse: ({ weightStorage, knobs }) =>
      weightStorage === "i2" && knobs.linearCompute !== "f32"
        ? "linear: i2 residency supports only linearCompute 'f32'"
        : undefined,
  },
  // A linear declared to receive packed activations (ADR 0105) that lands on neither the parallel
  // GEMV's packed form nor the integer GEMV would have f32 words flow into a `vec4<u32>` binding — a
  // silent wrong value. The pairing is accepted in one place, fusion.ts; it is checked here too, with
  // the same predicates. The byte-plane GEMV is the opposite case: it binds f32 quads, so packed words
  // there would be read as f32.
  {
    refuse: ({ packed, planes, where }) =>
      packed !== undefined && planes !== undefined
        ? `${where}: declared as packed activations but is a byte-plane GEMV linear, which reads f32 activations`
        : undefined,
  },
  {
    refuse: (linear) => {
      const { packed, knobs, linearGemv, integer, weightStorage, groupSize, m, n, k, where } =
        linear;
      if (packed === undefined) return undefined;
      const eligible =
        packed.role === "read" &&
        knobs.linearGemvReduce === "parallel" &&
        knobs.linearCompute === "f32" &&
        (integer !== undefined ||
          linearGemvPackedEligible(linearGemv, weightStorage, m, n, k, groupSize) !== undefined);
      return eligible
        ? undefined
        : `${where}: declared as packed activations but lands on neither the parallel GEMV nor the integer GEMV`;
    },
  },
  // The integer GEMV (`integerGemv`): only when the activation does arrive packed; a linear the route
  // function accepts whose producer was not packed keeps the entries below.
  {
    choose: ({ packed, integer }) =>
      packed !== undefined && integer !== undefined
        ? { id: "integer-gemv", variant: integer }
        : undefined,
  },
  // A linear declared to absorb its producer (perf-ledger F-4) that does not take the i8a8 path with a
  // row the prologue kernel accepts would run without the folded op — a silent wrong value.
  {
    refuse: ({ prologue, i8a8, k, where }) =>
      prologue !== undefined && !(i8a8 && quantizeRowsPrologueEligible(k))
        ? `${where}: declared with a ${prologue.op} prologue (perf-ledger F-4) but does not take the i8a8 path with a prologue-eligible row`
        : undefined,
  },
  // The integer dot-product path (opt-in × integer residency × k > 0 × k % 4 == 0 — linear-route.ts).
  { choose: ({ i8a8 }) => (i8a8 ? { id: "i8a8" } : undefined) },
  // f16 compute cannot be combined with integer-resident weights (w8a16 / w4a16 are not implemented).
  // It fails instead of dropping to f32: dropping would give "f16 has no effect on the integer layers
  // only", which nobody notices without reading the diagnostic keys.
  {
    refuse: ({ compute, weightStorage, where }) =>
      compute === "f16" && weightStorage === "i8"
        ? `${where}: linearCompute 'f16' cannot be combined with i8-resident weights (w8a16 is not implemented). ` +
          "Set linearCompute to 'a8', or store this weight as f32 / f16 / bf16"
        : undefined,
  },
  {
    refuse: ({ compute, weightStorage, where }) =>
      compute === "f16" && weightStorage === "i4"
        ? `${where}: linearCompute 'f16' cannot be combined with i4-resident weights (w4a16 is not implemented). ` +
          "Set linearCompute to 'a8', or store this weight as f32 / f16 / bf16"
        : undefined,
  },
  // Small M branches to the GEMV family (ADR 0082). In the bucket M ≤ 64 the default GEMM skeleton has
  // only 4 of its 64 threads write output, and the double barrier per K tile exposes the weight-read
  // latency serially. The sequential members keep the reduction order, the spelling of the
  // multiply-add and the order of adding bias of the default path, so they are bit-identical to it
  // (the numeric contract of src/kernels/linear-gemv.ts; the check is tests/gpu_linear_gemv_test.ts).
  //
  // f16 / bf16 / f32 storage: f32 compute, one weight word = 8 elements for f16 / bf16 and 4 for f32.
  // The sequential member is the M = 1 kernel alone (ADR 0082 addenda 6 and 7; bf16 is ADR 0131) —
  // there is no row block for these storages, so more rows keep the GEMM. Where linear is at `reorder`
  // and the profile's table names the shape, 1 to 8 rows take the parallel GEMV, as for an integer
  // storage: decode and the verification of a few rows share one kernel and one summation order.
  // `v4` is not a GEMV requirement (the output is written as scalars), but every swept shape was v4,
  // so the eligibility stays inside the measured range.
  {
    choose: (linear) => {
      const { m, n, weightStorage, compute, v4, k, knobs, linearGemv } = linear;
      if (!isFloatStorage(weightStorage)) return undefined;
      if (compute !== "f32" || !v4 || k <= 0 || k % linearGemvUnit(weightStorage) !== 0) {
        return undefined;
      }
      const parallel =
        knobs.linearGemvReduce !== "sequential" &&
        linearGemvParallelEligible(linearGemv, weightStorage, m, n, k) !== undefined;
      return m === 1 || parallel
        ? gemvChoice(linear, weightStorage, undefined, undefined)
        : undefined;
    },
  },
  // The byte-plane GEMV (`REORDER_STRATEGIES.linearGemvPlanes`, linear at `reorder`): an i2 linear
  // of a measured shape at M = 1..8 takes it in place of the i2 GEMV family below. Its activation
  // is f32 here (a packed one was refused above), and i2 computes in f32 (the first entry).
  {
    choose: ({ planes }) =>
      planes !== undefined ? { id: "planes-gemv", variant: planes } : undefined,
  },
  // i2 / i4 / i8 storage: 1 ≤ M ≤ LINEAR_GEMV_MAX_ROWS (the range measured with one thread owning the
  // whole reduction of one output column × row block), f32 compute, and `k` a multiple of the elements
  // of one weight word (16 B alignment of the `vec4<u32>` weight binding: 32 elements for i4, 16 for
  // i8). For i4 the group length must be a multiple of the word too: a word crossing a group would need
  // more than one scale — a silent wrong value. An i8 scale is one per output channel, bound once
  // outside the reduction.
  {
    choose: (linear) => {
      const { m, weightStorage, compute, v4, k, packed } = linear;
      return gemvRows(m) &&
        weightStorage === "i2" &&
        compute === "f32" &&
        v4 &&
        k % linearGemvUnit("i2") === 0
        ? gemvChoice(linear, "i2", undefined, packed)
        : undefined;
    },
  },
  {
    choose: (linear) => {
      const { m, weightStorage, compute, v4, k, groupSize, packed } = linear;
      const unit = linearGemvUnit("i4");
      return gemvRows(m) &&
        weightStorage === "i4" &&
        compute === "f32" &&
        v4 &&
        groupSize !== undefined &&
        groupSize % unit === 0 &&
        k % unit === 0
        ? gemvChoice(linear, "i4", groupSize, packed)
        : undefined;
    },
  },
  {
    choose: (linear) => {
      const { m, weightStorage, compute, v4, k, packed } = linear;
      return gemvRows(m) &&
        weightStorage === "i8" &&
        compute === "f32" &&
        v4 &&
        k % linearGemvUnit("i8") === 0
        ? gemvChoice(linear, "i8", undefined, packed)
        : undefined;
    },
  },
  // The split-K GEMM (src/kernels/linear-splitk.ts): the same products added slice by slice. Only
  // where the internal option allows it, under f32 compute (any storage the GEMM skeleton reads),
  // for a reduction long enough to cut. It comes after the packed and i8a8 entries and the GEMV
  // family, so the linears they take keep their kernels; f16 compute keeps its own GEMM.
  {
    choose: ({ knobs, compute, k, v4, groupSize }) => {
      if (!knobs.linearSplitK || compute !== "f32") return undefined;
      const slices = linearSplitKSlices(k);
      return slices === undefined ? undefined : { id: "gemm-splitk", slices, v4, groupSize };
    },
  },
  // The default GEMM skeleton (the reference kernel under f32 compute).
  {
    choose: ({ compute, v4, groupSize }) => ({
      id: compute === "f16" ? "gemm-f16" : "gemm-f32",
      compute,
      v4,
      groupSize,
    }),
  },
];

/** The outcome of the selection: a strategy, or the message of the refusal. */
export type LinearResolution = { readonly choice: LinearChoice } | { readonly refusal: string };

/**
 * The strategy of one linear, or why the combination is refused.
 *
 * MUST: the only decision. The recipe builder builds what it returns; the fusion pass asks it whether
 * a linear would take what it is about to declare (packed activations, a folded producer); the
 * estimate asks it which linears have node temporaries. Nothing re-derives which kernel a linear
 * takes.
 */
export const resolveLinearStrategy = (facts: LinearFacts): LinearResolution => {
  const { plan, weightStorage, groupSize, knobs } = facts;
  const [x, weight] = plan.inputShapes;
  const n = weight[0];
  const k = weight[1];
  const linear: Derived = {
    ...facts,
    m: numel(x.slice(0, -1)),
    n,
    k,
    where: `linear [${x.join(",")}] × [${weight.join(",")}]`,
    v4: gemmUsesVec4(k, n),
    compute: knobs.linearCompute === "f16" ? "f16" : "f32",
    integer: linearIntegerGemvVariant(knobs.integerGemv, plan, weightStorage, groupSize),
    planes: linearPlanesGemvVariant(knobs.linearGemvPlanes, plan, weightStorage),
    i8a8: linearTakesI8a8(knobs.linearCompute, weightStorage, k),
  };
  for (const entry of LINEAR_SELECTION) {
    if ("refuse" in entry) {
      const refusal = entry.refuse(linear);
      if (refusal !== undefined) return { refusal };
      continue;
    }
    const choice = entry.choose(linear);
    if (choice !== undefined) return { choice };
  }
  // The last entry always chooses; reaching here means the table lost it.
  return { refusal: `${linear.where}: no linear strategy applies` };
};

/** {@link resolveLinearStrategy} for a caller that fails on a refusal ({@link ExecutionError}). */
export const selectLinearStrategy = (facts: LinearFacts): LinearChoice => {
  const resolution = resolveLinearStrategy(facts);
  if ("refusal" in resolution) throw new ExecutionError(resolution.refusal);
  return resolution.choice;
};
