/**
 * The geometry sweep entry's settings: its declared parameters, what a run is asked for (candidate
 * set, op families, cases, geometries, the quick look's margin, rounds, idle seconds between cases)
 * and its plan (the case list `runGeometrySweep` is called with, one case per call). Pure, so the
 * case selection is tested without a GPU.
 *
 * The case list depends on the candidate set: the evaluation set `k88` sweeps a table of its own
 * (`geometrySweepCases("k88")`). Every function that takes `cases` takes the list of the chosen set.
 * The `explore` set sweeps `full`'s table class by class.
 */

import {
  geometryExploreOrder,
  type GeometrySweepCandidateSet,
  type GeometrySweepCase,
  type GeometrySweepOp,
  geometrySweepCases,
  QUICK_LOOK_MARGIN,
} from "@karume/runtime/tune";

import {
  choicesValue,
  choiceValue,
  integerValue,
  numberValue,
  type Parameter,
  ParameterError,
  type ParameterValues,
  textValue,
} from "../parameters.ts";
import { K88_SET } from "./k88.ts";

/**
 * The candidate sets in growing size (the runtime's default is `quick+`; `explore` measures `full`'s
 * candidates in fewer rows), then the evaluation set `k88`.
 */
export const CANDIDATE_SETS = [
  "quick",
  "quick+",
  "explore",
  "full",
  "k88",
] as const satisfies readonly GeometrySweepCandidateSet[];

/** What a run is asked for: the entry's typed parameters. */
export type SweepSelection = {
  readonly candidateSet: GeometrySweepCandidateSet;
  /** The op families swept. */
  readonly ops: readonly GeometrySweepOp[];
  /**
   * The chosen cases; empty = every case of `ops` (the runtime's own meaning of an empty list). A
   * chosen case of an op that is not chosen is not swept.
   */
  readonly cases: readonly string[];
  /**
   * The candidate geometries to measure, by name, besides the default geometry (always measured);
   * empty = every candidate of the set. Every name must be a candidate of every chosen case (the
   * runtime refuses it otherwise, before the GPU is touched).
   */
  readonly geometries: readonly string[];
  /** The quick look's margin (0: off). */
  readonly quickLookMargin: number;
  readonly rounds: number;
  /** Idle time before every case but the first (cooling). */
  readonly idleSeconds: number;
};

/** A checked selection: what one sweep runs. */
export type SweepPlan = {
  /**
   * The cases in the table's order — one `runGeometrySweep` call each; for the `explore` set class
   * by class (`geometryExploreOrder`), as one call over them measures them.
   */
  readonly caseIds: readonly string[];
  /** The op families of those cases, in the table's order (`ops` of every call). */
  readonly ops: readonly GeometrySweepOp[];
  readonly candidateSet: GeometrySweepCandidateSet;
  readonly geometries: readonly string[];
  readonly quickLookMargin: number;
  readonly rounds: number;
  /** Idle time before every case but the first (cooling). */
  readonly idleSeconds: number;
};

/** The op families of the table, in its order (each once). */
export const sweepOps = (cases: readonly GeometrySweepCase[]): GeometrySweepOp[] => [
  ...new Set(cases.map(({ op }) => op)),
];

/** The sweep's own case table, then the evaluation set's (the two tables a run can choose from). */
const BOTH_TABLES = [...geometrySweepCases(), ...geometrySweepCases(K88_SET)];

/** Every op family of the two case tables, in table order (the options of `ops`). */
export const SWEEP_OPS: readonly GeometrySweepOp[] = sweepOps(BOTH_TABLES);

/** Every case id of the two case tables, in table order (the options of `cases`). */
export const SWEEP_CASE_IDS: readonly string[] = [...new Set(BOTH_TABLES.map(({ id }) => id))];

/** The entry id of the geometry sweep: the workbench's entry and the record's `entry.id`. */
export const GEOMETRY_SWEEP_ENTRY_ID = "geometry-sweep";

/**
 * The geometry sweep entry's parameters, in the order a record's `entry.parameters` holds them: the
 * workbench's entry declares them, and the benchmark app parses its run's values with them.
 */
export const GEOMETRY_SWEEP_PARAMETERS: readonly Parameter[] = [
  {
    kind: "choice",
    name: "candidateSet",
    label: "Candidate set",
    options: CANDIDATE_SETS,
    default: "quick+",
  },
  {
    kind: "choices",
    name: "ops",
    label: "Op families",
    options: SWEEP_OPS,
    default: sweepOps(geometrySweepCases()),
  },
  {
    kind: "choices",
    name: "cases",
    label: "Cases (none: every case of the chosen op families in the chosen set)",
    options: SWEEP_CASE_IDS,
    default: [],
  },
  {
    kind: "text",
    name: "geometries",
    label:
      "Candidate geometries, comma-separated names (none: every candidate of the set; the default geometry is always measured)",
    default: "",
  },
  {
    kind: "number",
    name: "quickLookMargin",
    label: "Quick look margin (0: off — every geometry is timed in full)",
    min: 0,
    max: 1000,
    default: QUICK_LOOK_MARGIN,
  },
  {
    kind: "integer",
    name: "rounds",
    label: "Rounds per geometry (the min is kept)",
    min: 1,
    max: Number.MAX_SAFE_INTEGER,
    default: 5,
  },
  {
    kind: "integer",
    name: "idleSeconds",
    label: "Idle before every case but the first (s)",
    min: 0,
    max: Number.MAX_SAFE_INTEGER,
    default: 10,
  },
];

/**
 * The selection the entry's parameters (`candidateSet`, `ops`, `cases`, `geometries` — names
 * separated by commas —, `quickLookMargin`, `rounds`, `idleSeconds`) give. A margin between 0 and
 * 1 is a {@link ParameterError}: it would screen out every geometry but the default. So are
 * geometries named for the `explore` set, which chooses what it measures of `full` itself.
 */
export const sweepSelection = (values: ParameterValues): SweepSelection => {
  const quickLookMargin = numberValue(values, "quickLookMargin");
  if (quickLookMargin > 0 && quickLookMargin < 1) {
    throw new ParameterError(
      "quickLookMargin",
      `${quickLookMargin} is neither 0 (the quick look off) nor at least 1`,
    );
  }
  const candidateSet = choiceValue(values, "candidateSet", CANDIDATE_SETS);
  const geometries = textValue(values, "geometries")
    .split(",")
    .map((name) => name.trim())
    .filter((name) => name !== "");
  if (candidateSet === "explore" && geometries.length > 0) {
    throw new ParameterError(
      "geometries",
      "the explore set measures the candidates of full and drops them itself: leave geometries empty",
    );
  }
  return {
    candidateSet,
    ops: choicesValue(values, "ops", SWEEP_OPS),
    cases: choicesValue(values, "cases", SWEEP_CASE_IDS),
    geometries,
    quickLookMargin,
    rounds: integerValue(values, "rounds"),
    idleSeconds: integerValue(values, "idleSeconds"),
  };
};

/**
 * The plan of a selection over the chosen set's case table: the chosen cases in the table's order,
 * class by class for the `explore` set (`geometryExploreOrder`). Throws {@link ParameterError} for a
 * chosen case the table does not have (naming it), and an error when no case is left to sweep.
 */
export const planSweep = (
  selection: SweepSelection,
  cases: readonly GeometrySweepCase[],
): SweepPlan => {
  const ids = new Set(cases.map(({ id }) => id));
  const foreign = selection.cases.filter((id) => !ids.has(id));
  if (foreign.length > 0) {
    throw new ParameterError(
      "cases",
      `${foreign.join(", ")} ${foreign.length === 1 ? "is not a case" : "are not cases"} of the ${selection.candidateSet} set`,
    );
  }
  const chosenIds = selection.cases.length === 0 ? ids : new Set(selection.cases);
  const chosen = cases.filter(({ id, op }) => selection.ops.includes(op) && chosenIds.has(id));
  if (chosen.length === 0) throw new Error("Choose at least one case");
  return {
    caseIds: (selection.candidateSet === "explore" ? geometryExploreOrder(chosen) : chosen).map(
      ({ id }) => id,
    ),
    ops: sweepOps(chosen),
    candidateSet: selection.candidateSet,
    geometries: selection.geometries,
    quickLookMargin: selection.quickLookMargin,
    rounds: selection.rounds,
    idleSeconds: selection.idleSeconds,
  };
};
