/**
 * What the GEMV sweep measures for one case and row count: the **baseline** (what a Session at the
 * level `exact` runs for the shape with the default profile), the **candidates** in two classes, and
 * each one's program. The baseline is a kernel of the GEMV family, except for a float-stored weight
 * (bf16 / f16 / f32) at M ≥ 2: that storage has no row-block form, so its Session runs the GEMM
 * there, and the GEMM is what the parallel GEMV is read against. The GEMV choices are the default
 * profile's; the GEMM's tile geometry is not a GEMV choice, and a Session takes it from the profile
 * its device resolves to — so the GEMM reference is built with that profile ({@link gemvProgram}'s
 * `reference`), not with the default one.
 *
 * - *identical* — the baseline's summation order, so the output words must be the baseline's: the
 *   sequential M = 1 kernel's variants (columns per workgroup × words read ahead) and, for M ≥ 2, the
 *   row-block heights.
 * - *reorder* — another summation order, held to the band of `linear`: the parallel GEMV at each
 *   lane count, plain and with staged activations, for M = 1..8, its subgroup form where the
 *   device has the features the kernel needs, and for M ≥ 2 its row-blocked form at each height
 *   the kernel exists at.
 *
 * A candidate is a `LinearGemvChoice` (src/runtime/linear-strategies.ts; a baseline may be the
 * GEMM's choice — {@link GemvKernelChoice}), and its dispatches are
 * `linearProgramFor`'s (src/runtime/linear-program.ts): nothing here restates a kernel's bindings,
 * params or grid. The baseline and what the default profile chooses at each level come from the
 * selection itself (`resolveLinearStrategy`), asked with the knobs a Session at that level resolves
 * (`executionKnobDefaults`). The packed-activation form is not measured: it reads the codes of the
 * quantizing producer, which the sweep does not run. What the derivation reads each group for
 * ({@link gemvGroupRole}) also decides which reorder candidates a run carries from its M = 1 group on
 * to M = 4 and 8 ({@link gemvCarriedFromM1}), with the derivation's threshold and steadiness bound.
 *
 * NOTE: device-free. The device enters only through {@link GemvGates}.
 */

import {
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
  type ResolvedGeometryProfile,
} from "../kernels/geometry-profile.ts";
import {
  defaultLinearGemvVariant,
  isFloatStorage,
  LINEAR_GEMV_PARALLEL_MAX_ROWS,
  LINEAR_GEMV_PARALLEL_ROWS,
  type LinearGemvParallelLanes,
  linearGemvParallelRowsFits,
  linearGemvRowsForShape,
  type LinearGemvVariant,
} from "../kernels/linear-gemv.ts";
import { LINEAR_OP } from "../ops.ts";
import { OP_CONTRACTS } from "../ops/contracts.ts";
import { type LinearProgram, linearProgramFor } from "../runtime/linear-program.ts";
import {
  type LinearGemmChoice,
  type LinearGemvChoice,
  linearKnobsOf,
  resolveLinearStrategy,
} from "../runtime/linear-strategies.ts";
import type { NumericLevel } from "../runtime/numerics.ts";
import type { NodePlan } from "../runtime/plan.ts";
import { executionKnobDefaults } from "../runtime/session-options-check.ts";
import { GEMV_CASES, type GemvCase } from "./gemv-cases.ts";

/**
 * The candidate sets: `quick` (M = 1, a small identical grid), `full` (every row count a derivation
 * rule reads — the parallel GEMV's range, M = 1, 4 and 8 —, the whole grid) and `row-blocks` (the
 * row-block form's row counts beyond that range, M = 16 and 64, which only a future row-block rule
 * would read).
 */
export const GEMV_CANDIDATE_SETS = ["quick", "full", "row-blocks"] as const;
export type GemvCandidateSet = (typeof GEMV_CANDIDATE_SETS)[number];

export const isGemvCandidateSet = (value: string): value is GemvCandidateSet =>
  (GEMV_CANDIDATE_SETS as readonly string[]).includes(value);

/** The summation-order class of a candidate. */
export type GemvCandidateClass = "identical" | "reorder";

/**
 * The choice of a kernel the sweep measures: a member of the GEMV family, or — only as the reference
 * of a float-stored weight at M ≥ 2 — the GEMM under f32 compute.
 */
export type GemvKernelChoice = LinearGemvChoice | LinearGemmChoice;

/** One kernel the sweep measures for a case and row count. */
export type GemvCandidate = {
  /**
   * Unique within a case and row count: `c32u4`, `c32u4r8`, `parallel-l4`, `staged-l32`,
   * `staged-l4r4` (the row-blocked form, 4 rows a thread), `subgroup32-l8`, and `gemm` for the
   * GEMM as a reference.
   */
  readonly name: string;
  readonly class: GemvCandidateClass;
  readonly choice: GemvKernelChoice;
};

/** What the device decides about the candidates: the subgroup form, and the grids it can dispatch. */
export type GemvGates = {
  /** The device has `subgroups`, `subgroup-size-control` and the WGSL `subgroup_id`. */
  readonly subgroups: boolean;
  /** A candidate whose grid exceeds this on an axis is no candidate on the device. */
  readonly maxComputeWorkgroupsPerDimension: number;
};

/** The lane counts of the parallel GEMV (`LinearGemvParallelLanes`). */
const GEMV_LANES: readonly LinearGemvParallelLanes[] = [2, 4, 8, 16, 32];

/** The columns per workgroup of the sequential M = 1 kernel's grid. */
const GEMV_COLS: readonly number[] = [16, 32, 64, 128, 256];

/** The words read ahead of the sequential M = 1 kernel's grid. */
const GEMV_UNROLLS: readonly number[] = [1, 2, 4, 8, 16];

/** The profile the baseline and the profile's choices are read from. */
const PROFILE: ResolvedGeometryProfile = DEFAULT_GEOMETRY_PROFILE;

/** Whether a kernel's choice is the GEMM's (the reference of a float-stored weight at M ≥ 2). */
export const isGemmChoice = (choice: GemvKernelChoice): choice is LinearGemmChoice =>
  choice.id === "gemm-f32" || choice.id === "gemm-f16";

/** The candidate's name (the class and the shape are the row's own fields). */
const gemvChoiceName = (choice: GemvKernelChoice): string => {
  // One name whatever the tile geometry: the kernel's fingerprint tells two GEMMs apart
  if (isGemmChoice(choice)) return "gemm";
  const { cols, unroll } = choice.variant;
  switch (choice.id) {
    case "gemv-m1":
      return `c${cols}u${unroll}`;
    case "gemv-rows":
      return `c${choice.rowsVariant?.cols ?? cols}u${choice.rowsVariant?.unroll ?? unroll}r${choice.rowsVariant?.rows ?? 1}`;
    case "gemv-parallel":
      return `parallel-l${choice.lanes ?? 0}`;
    case "gemv-parallel-staged":
      return `staged-l${choice.lanes ?? 0}${choice.parallelRows === undefined ? "" : `r${choice.parallelRows}`}`;
    case "gemv-parallel-subgroup32":
      return `subgroup32-l${choice.lanes ?? 0}`;
    case "gemv-parallel-packed":
      return `packed-l${choice.lanes ?? 0}`;
  }
};

/** The class a choice's summation order puts it in (the GEMM sums in the reference order). */
const gemvChoiceClass = (choice: GemvKernelChoice): GemvCandidateClass =>
  choice.id === "gemv-m1" || choice.id === "gemv-rows" || isGemmChoice(choice)
    ? "identical"
    : "reorder";

const candidateOf = (choice: GemvKernelChoice): GemvCandidate => ({
  name: gemvChoiceName(choice),
  class: gemvChoiceClass(choice),
  choice,
});

/**
 * The plan of the one `linear` node `[m, k] × [n, k]ᵀ + [n]` the selection reads (it reads the input
 * shapes; the node and its contract name the op).
 */
const linearNodePlan = (gemvCase: GemvCase, m: number): NodePlan => {
  const contract = OP_CONTRACTS.get(LINEAR_OP);
  if (contract === undefined) throw new Error(`The op ${LINEAR_OP} has no contract`);
  const { n, k } = gemvCase;
  return {
    node: { op: LINEAR_OP, ins: ["x", "weight", "bias"], outs: ["out"], attrs: {}, states: {} },
    contract,
    inputShapes: [[m, k], [n, k], [n]],
    inputDtypes: ["f32", "f32", "f32"],
    outputs: [{ name: "out", shape: [m, n], dtype: "f32" }],
  };
};

/**
 * What a Session with linear at `level` runs for the case at `m` rows with the default profile: the
 * selection's choice, with the knobs the level resolves to. Throws when the selection takes neither
 * the GEMV family nor — for a float-stored weight at M ≥ 2, which has no sequential GEMV — the GEMM
 * under f32 compute (a case outside the family's range).
 */
export const gemvProfileChoice = (
  gemvCase: GemvCase,
  m: number,
  level: NumericLevel,
): GemvCandidate => {
  const resolution = resolveLinearStrategy({
    plan: linearNodePlan(gemvCase, m),
    weightStorage: gemvCase.storage,
    groupSize: gemvCase.group,
    knobs: linearKnobsOf(executionKnobDefaults({ numerics: { ops: { linear: level } } })),
    linearGemv: PROFILE.linearGemv,
    packed: undefined,
    prologue: undefined,
  });
  if ("refusal" in resolution) throw new Error(`Case ${gemvCase.id}: ${resolution.refusal}`);
  const { choice } = resolution;
  if (choice.id === "gemm-f32" && isFloatStorage(gemvCase.storage) && m > 1) {
    return candidateOf(choice);
  }
  if (
    choice.id !== "gemv-m1" &&
    choice.id !== "gemv-rows" &&
    choice.id !== "gemv-parallel" &&
    choice.id !== "gemv-parallel-staged" &&
    choice.id !== "gemv-parallel-subgroup32" &&
    choice.id !== "gemv-parallel-packed"
  ) {
    throw new Error(
      `Case ${gemvCase.id} at ${m} rows takes ${choice.id} at the level ${level}, not the GEMV family`,
    );
  }
  return candidateOf(choice);
};

/** The baseline of a case at `m` rows: what a Session at the level `exact` runs. */
export const gemvBaseline = (gemvCase: GemvCase, m: number): GemvCandidate =>
  gemvProfileChoice(gemvCase, m, "exact");

/**
 * The row counts a set measures of a case: M = 1 for `quick`, the case's row counts of the parallel
 * range for `full`, the ones beyond it for `row-blocks`.
 */
export const gemvRowCountsIn = (gemvCase: GemvCase, set: GemvCandidateSet): readonly number[] =>
  gemvCase.rows.filter((m) =>
    set === "quick"
      ? m === 1
      : set === "full"
        ? m <= LINEAR_GEMV_PARALLEL_MAX_ROWS
        : m > LINEAR_GEMV_PARALLEL_MAX_ROWS,
  );

/**
 * What the derivation (`deriveLinearGemvChoices`) reads the kernels of a class at a row count for —
 * one statement that the derivation, the sweep's quick look, the sweep's skipping at M = 4 and 8
 * ({@link gemvCarriedFromM1}) and the plan of what is missing share:
 * - `winner`: M = 1. The derivation picks, among the candidates that clear the threshold over the
 *   class's reference (the baseline; the default's choice at `reorder`), the one with the largest
 *   gain.
 * - `no-regression`: the reorder class at the other row counts of the parallel range (a parallel
 *   row serves M = 1 to 8). The derivation asks only that a candidate is not slower than the
 *   default's choice at `reorder`.
 * - `unread`: the identical class at M ≥ 2 (the row-block heights, which no rule turns into the
 *   thread target) and every row count beyond the parallel range.
 */
export type GemvGroupRole = "winner" | "no-regression" | "unread";

export const gemvGroupRole = (m: number, candidateClass: GemvCandidateClass): GemvGroupRole =>
  m === 1
    ? "winner"
    : candidateClass === "reorder" && m <= LINEAR_GEMV_PARALLEL_MAX_ROWS
      ? "no-regression"
      : "unread";

/**
 * The adoption threshold of the GEMV choices (the maintainer, 2026-11-17): a candidate replaces the
 * default profile's choice for a shape only when it is at least this much faster than that choice in
 * every group that is evidence. It is not the tile geometry's `minSpeedup`.
 */
export const GEMV_MIN_SPEEDUP = 1.1;

/**
 * The steadiness bound: a group whose baseline rounds spread (`baselineSpread`, the baseline's
 * largest round ÷ its smallest) beyond this is unsteady, and no evidence.
 *
 * Source: the two `karume-gemv-sweep/2` records of the `quick` set on the RTX 3080 Ti under Node
 * (2026-11-17T11-51-15 and T12-29-00), grouped by the worse spread of a case over the two records: a
 * candidate's gain over the table's choice agrees between the records within ±10 % for 95.6 % of the
 * rows below 1.05, 96.1 % in [1.05, 1.10), 97.2 % in [1.10, 1.25) and 90.3 % from 1.25 up (12 of 66
 * cases). Below 1.25 the spread does not predict a less repeatable gain: the device slows as a case
 * goes on (the last round is the slowest in 39 % of the rows), every kernel of a round sees it alike,
 * and the minimum reads the early rounds.
 */
export const GEMV_STEADY_SPREAD = 1.25;

/** A row of a case's M = 1 group as {@link gemvCarriedFromM1} reads it (a record's row has these fields). */
export type GemvM1Row = {
  readonly candidate: string;
  readonly perDispatch?: number;
  readonly withinBand?: boolean;
  readonly nonFiniteOutputs?: number;
  readonly screened?: true;
  readonly error?: string;
};

/**
 * A case's M = 1 group as the same run measured it: its summary's baseline spread and default
 * choice at `reorder`, and its rows (a record's summary and rows have these fields).
 */
export type GemvM1Group = {
  readonly baselineSpread?: number;
  readonly profile: { readonly reorder: string };
  readonly rows: readonly GemvM1Row[];
};

/**
 * Whether the sweep measures a candidate of a case at `m` rows, given the case's M = 1 group of the
 * same run (`undefined` when the run has no M = 1 group to read: narrowed by row count, stopped
 * before it, or its baseline was not timed). A row count of the `no-regression` role is read only for a candidate that M = 1 would adopt
 * (`deriveLinearGemvChoices` asks there only that such a candidate is not slower than the default's
 * choice), so a reorder candidate whose own M = 1 row already leaves it unadoptable is skipped
 * there. Its M = 1 row leaves it unadoptable when it
 * 1. has no time (screened by the quick look, or failed), or
 * 2. did not hold its check (outside the band, or a non-finite output), or
 * 3. is less than {@link GEMV_MIN_SPEEDUP} faster than the default's choice at `reorder` in a steady
 *    group (baseline spread at most {@link GEMV_STEADY_SPREAD}). An unsteady group is no evidence,
 *    so its times rule nothing out; without the default's time at M = 1 the gain is unknown.
 *
 * A row-blocked candidate (`choice.parallelRows`) has no row at M = 1: it is the form a parallel
 * row of its lanes takes at M ≥ 2, so it is read by the rows of that lane count — measured when the
 * default's choice at `reorder` at M = 1 has its lanes (the row the derivation keeps), or when the
 * plain or the staged candidate of its lanes is carried by the conditions above.
 *
 * Measured whatever M = 1 says: the references at `m` (`references` — the baseline and the default's
 * choice at `reorder`, which every gain is read against), the default's choice at `reorder` at M = 1
 * (the same kernel unless the default's row carries a row block), the identical class, every role
 * but `no-regression`, and a candidate the M = 1 group has no row of.
 */
export const gemvCarriedFromM1 = (
  m: number,
  candidate: Pick<GemvCandidate, "name" | "class"> & {
    readonly choice?: Pick<LinearGemvChoice, "lanes" | "parallelRows"> | LinearGemmChoice;
  },
  references: { readonly exact: string; readonly reorder: string },
  first: GemvM1Group | undefined,
): boolean => {
  if (first === undefined || gemvGroupRole(m, candidate.class) !== "no-regression") return true;
  if (candidate.name === references.exact || candidate.name === references.reorder) return true;
  // The default's choice at M = 1 is its row's own kernel: where the row carries a row block, the
  // reference at `m` is the row-blocked one, and this is what shows whether the block pays here
  if (candidate.name === first.profile.reorder) return true;
  // The GEMM (a float-stored weight's reference) has no lanes and no row block
  const { choice } = candidate;
  const lanes =
    choice !== undefined && "parallelRows" in choice && choice.parallelRows !== undefined
      ? choice.lanes
      : undefined;
  if (lanes === undefined) return carriedByItsRow(candidate.name, first);
  return [`parallel-l${lanes}`, `staged-l${lanes}`].some(
    (base) => base === first.profile.reorder || carriedByItsRow(base, first),
  );
};

/** Whether the M = 1 row of `name` leaves the candidate adoptable ({@link gemvCarriedFromM1}'s conditions). */
const carriedByItsRow = (name: string, first: GemvM1Group): boolean => {
  const row = first.rows.find((one) => one.candidate === name);
  if (row === undefined) return true;
  if (row.perDispatch === undefined || row.screened === true || row.error !== undefined) {
    return false;
  }
  if (row.withinBand !== true || (row.nonFiniteOutputs ?? 0) !== 0) return false;
  const steady = first.baselineSpread !== undefined && first.baselineSpread <= GEMV_STEADY_SPREAD;
  if (!steady) return true;
  const reference = first.rows.find((one) => one.candidate === first.profile.reorder)?.perDispatch;
  return reference === undefined || reference / row.perDispatch >= GEMV_MIN_SPEEDUP;
};

/** The value of `grid` next below and next above `value`, and `value` itself. */
const around = (grid: readonly number[], value: number): number[] => {
  const below = grid.filter((entry) => entry < value).at(-1);
  const above = grid.find((entry) => entry > value);
  return [below, value, above].filter((entry): entry is number => entry !== undefined);
};

/** The sequential M = 1 variants of a set: around the baseline's for `quick`, the whole grid for `full`. */
const m1Variants = (baseline: LinearGemvVariant, set: GemvCandidateSet): LinearGemvVariant[] => {
  const cols = set === "quick" ? around(GEMV_COLS, baseline.cols) : GEMV_COLS;
  const unrolls = set === "quick" ? around(GEMV_UNROLLS, baseline.unroll) : GEMV_UNROLLS;
  return cols.flatMap((colsValue) => unrolls.map((unroll) => ({ cols: colsValue, unroll })));
};

/**
 * The identical class at `m` rows: at M = 1 the column form's variants, at M ≥ 2 the row-block heights
 * (powers of two up to the tallest the selection can choose for the shape — `linearGemvRowsForShape`
 * with a target it always meets), with the profile's columns and read-ahead. A float-stored weight
 * has no row-block form, so its identical class at M ≥ 2 is its baseline alone (the GEMM).
 */
const identicalChoices = (
  gemvCase: GemvCase,
  m: number,
  set: GemvCandidateSet,
  baseline: GemvKernelChoice,
): LinearGemvChoice[] => {
  const { storage, group, n } = gemvCase;
  if (isGemmChoice(baseline)) return [];
  if (m === 1) {
    return m1Variants(baseline.variant, set).map((variant): LinearGemvChoice => ({
      id: "gemv-m1",
      storage,
      groupSize: group,
      lanes: undefined,
      variant,
      rowsVariant: undefined,
    }));
  }
  const general = defaultLinearGemvVariant(PROFILE.linearGemv);
  const tallest = linearGemvRowsForShape(PROFILE.linearGemv, storage, m, n, 1);
  const heights: number[] = [];
  for (let rows = 1; rows <= tallest; rows *= 2) heights.push(rows);
  // The selection names the row-block form by its row variant in both fields (`gemvChoice`).
  return heights.map((rows): LinearGemvChoice => {
    const rowsVariant = { ...general, rows };
    return {
      id: "gemv-rows",
      storage,
      groupSize: group,
      lanes: undefined,
      variant: rowsVariant,
      rowsVariant,
    };
  });
};

/**
 * The reorder class at `m` rows (M ≤ 8): every lane count, plain, staged and — when gated in, and for
 * an integer storage, the only ones with that form — subgroup; then, for M ≥ 2, the row-blocked form
 * at every lane count and every height of at most `m` rows the kernel exists at.
 */
const reorderChoices = (gemvCase: GemvCase, m: number, gates: GemvGates): LinearGemvChoice[] => {
  if (m > LINEAR_GEMV_PARALLEL_MAX_ROWS) return [];
  const { storage, group, n, k } = gemvCase;
  const variant = defaultLinearGemvVariant(PROFILE.linearGemv, { storage, n, k });
  const ids: LinearGemvChoice["id"][] = ["gemv-parallel", "gemv-parallel-staged"];
  // The subgroup form exists for the integer storages only
  if (gates.subgroups && !isFloatStorage(storage)) ids.push("gemv-parallel-subgroup32");
  const rowBlocks = GEMV_LANES.flatMap((lanes) =>
    LINEAR_GEMV_PARALLEL_ROWS.filter(
      (rows) => rows <= m && linearGemvParallelRowsFits(storage, lanes, rows),
    ).map((rows): LinearGemvChoice => ({
      id: "gemv-parallel-staged",
      storage,
      groupSize: group,
      lanes,
      variant,
      rowsVariant: undefined,
      parallelRows: rows,
    })),
  );
  return [
    ...ids.flatMap((id) =>
      GEMV_LANES.map((lanes): LinearGemvChoice => ({
        id,
        storage,
        groupSize: group,
        lanes,
        variant,
        rowsVariant: undefined,
      })),
    ),
    ...rowBlocks,
  ];
};

/** The facts `linearProgramFor` builds a case's program from at `m` rows. */
const programFacts = (gemvCase: GemvCase, m: number, reference: GeometryProfile) =>
  ({
    x: [m, gemvCase.k],
    weight: [gemvCase.n, gemvCase.k],
    m,
    n: gemvCase.n,
    k: gemvCase.k,
    weightStorage: gemvCase.storage,
    groupSize: gemvCase.group,
    // No packed pairing and no folded producer: the sweep measures the f32-activation forms.
    packedScale: undefined,
    prologue: undefined,
    // Read by the GEMM and i8a8 programs only: the GEMM that is a float-stored weight's reference
    // at M ≥ 2 takes its tile geometry from here. A GEMV program reads nothing of it.
    geometryProfile: reference,
    linearI8a8Dot: "emu",
  }) as const;

/**
 * The program of a choice for a case at `m` rows (`linearProgramFor`).
 *
 * `reference` is the profile whose GEMM geometry the GEMM takes — the reference of a float-stored
 * case at M ≥ 2. MUST: where a row is measured or read as evidence it is the profile the device
 * resolves to (the sweep: `sessionGeometryProfile` of its device; the derivation: the table it
 * derives for), because a parallel row is adopted only if it is not slower than what a Session on
 * that device runs. The default profile's is for callers that build GEMV programs alone.
 */
export const gemvProgram = (
  gemvCase: GemvCase,
  m: number,
  choice: GemvKernelChoice,
  reference: GeometryProfile = PROFILE,
): LinearProgram => linearProgramFor(choice, programFacts(gemvCase, m, reference));

/** The workgroup count of each axis of a program's dispatch (`ceil(items / tile)`, no limit). */
export const programWorkgroups = (
  program: LinearProgram,
): readonly (readonly [number, number, number])[] =>
  program.dispatches.map(({ grid }) =>
    grid.kind === "tiled"
      ? [
          Math.ceil(grid.axes[0].items / grid.axes[0].tile),
          Math.ceil(grid.axes[1].items / grid.axes[1].tile),
          Math.ceil(grid.axes[2].items / grid.axes[2].tile),
        ]
      : [Math.ceil(grid.items / grid.workgroupSize), 1, 1],
  );

/** Whether every dispatch of a choice's program fits the device's grid limit (no limit: no program built). */
const dispatchable = (
  gemvCase: GemvCase,
  m: number,
  choice: GemvKernelChoice,
  limit: number,
): boolean =>
  limit === Number.POSITIVE_INFINITY ||
  programWorkgroups(gemvProgram(gemvCase, m, choice)).every((axes) =>
    axes.every((count) => count <= limit),
  );

/**
 * The candidates of a case at `m` rows in a set, in measuring order (identical, then reorder),
 * without the baseline (measured first, as the baseline row) and without a candidate whose grid
 * exceeds the device's limit (the 262144-column heads at 32 lanes, for example).
 */
export const gemvCandidates = (
  gemvCase: GemvCase,
  m: number,
  set: GemvCandidateSet,
  gates: GemvGates,
): GemvCandidate[] => {
  const baseline = gemvBaseline(gemvCase, m);
  return [
    ...identicalChoices(gemvCase, m, set, baseline.choice),
    ...reorderChoices(gemvCase, m, gates),
  ]
    .map(candidateOf)
    .filter(
      (candidate) =>
        candidate.name !== baseline.name &&
        dispatchable(gemvCase, m, candidate.choice, gates.maxComputeWorkgroupsPerDimension),
    );
};

/** What the sweep measures of a case at one row count. */
export type GemvMeasurementPlan = {
  readonly m: number;
  readonly baseline: GemvCandidate;
  /** In measuring order, without the baseline. */
  readonly candidates: readonly GemvCandidate[];
  /** The candidates the default profile chooses at each level (by name). */
  readonly profile: { readonly exact: string; readonly reorder: string };
};

/**
 * Narrows what a sweep measures of a set: the row counts and the candidates, by name (absent: all).
 * The references — the baseline and the default profile's choice at `reorder` — are measured either
 * way: every gain is read against them.
 */
export type GemvSelection = {
  readonly rows?: readonly number[];
  readonly candidates?: readonly string[];
};

/**
 * The measurement plans of a case in a set, one per row count the set measures (and the selection
 * keeps), with the candidates the selection names and the reference at `reorder`.
 */
export const gemvMeasurementPlans = (
  gemvCase: GemvCase,
  set: GemvCandidateSet,
  gates: GemvGates,
  selection: GemvSelection = {},
): GemvMeasurementPlan[] =>
  gemvRowCountsIn(gemvCase, set)
    .filter((m) => selection.rows === undefined || selection.rows.includes(m))
    .map((m) => {
      const baseline = gemvBaseline(gemvCase, m);
      const reorder = gemvProfileChoice(gemvCase, m, "reorder").name;
      const named = selection.candidates;
      return {
        m,
        baseline,
        candidates: gemvCandidates(gemvCase, m, set, gates).filter(
          ({ name }) => named === undefined || name === reorder || named.includes(name),
        ),
        profile: { exact: baseline.name, reorder },
      };
    });

/** Gates that hold every candidate: the subgroup form, and no grid limit. */
const EVERY_CANDIDATE: GemvGates = {
  subgroups: true,
  maxComputeWorkgroupsPerDimension: Number.POSITIVE_INFINITY,
};

/**
 * Every kernel the current runtime can measure of a case at `m` rows in a set, whatever the device:
 * the baseline first, then the set's candidates with the subgroup form and no grid limit (by default
 * the whole grid, which holds every set's). A record's row stands for the current runtime only when
 * its candidate is one of these and builds the same kernel.
 */
export const gemvKernelsAt = (
  gemvCase: GemvCase,
  m: number,
  set: GemvCandidateSet = "full",
): GemvCandidate[] => [
  gemvBaseline(gemvCase, m),
  ...gemvCandidates(gemvCase, m, set, EVERY_CANDIDATE),
];

/**
 * Every candidate name of the case table, at every row count of every set, with the subgroup form
 * and no grid limit, each once in the order first met: the names `GemvSweepOptions.candidates`
 * takes. A new array on every call.
 */
export const gemvSweepCandidateNames = (cases: readonly GemvCase[] = GEMV_CASES): string[] => [
  ...new Set(
    cases.flatMap((gemvCase) =>
      gemvCase.rows.flatMap((m) =>
        gemvCandidates(gemvCase, m, "full", EVERY_CANDIDATE).map(({ name }) => name),
      ),
    ),
  ),
];
