/**
 * The geometry sweep entry's settings: what a run is asked for (candidate set, op families, cases,
 * rounds, idle seconds between cases) and its plan (the case list `runGeometrySweep` is called with,
 * one case per call). Pure, so the case selection is tested without a GPU.
 *
 * The case list depends on the candidate set: the evaluation set `k88` sweeps a table of its own
 * (`geometrySweepCases("k88")`). Every function that takes `cases` takes the list of the chosen set.
 */

import {
  type GeometrySweepCandidateSet,
  type GeometrySweepCase,
  type GeometrySweepOp,
  geometrySweepCases,
} from "@karume/runtime/tune";

import {
  choicesValue,
  choiceValue,
  integerValue,
  ParameterError,
  type ParameterValues,
} from "../entry.ts";
import { K88_SET } from "./k88.ts";

/**
 * The candidate sets in growing size (the runtime's default is `quick+`), then the evaluation set
 * `k88`.
 */
export const CANDIDATE_SETS = [
  "quick",
  "quick+",
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
  readonly rounds: number;
  /** Idle time before every case but the first (cooling). */
  readonly idleSeconds: number;
};

/** A checked selection: what one sweep runs. */
export type SweepPlan = {
  /** The cases in the table's order — one `runGeometrySweep` call each. */
  readonly caseIds: readonly string[];
  /** The op families of those cases, in the table's order (`ops` of every call). */
  readonly ops: readonly GeometrySweepOp[];
  readonly candidateSet: GeometrySweepCandidateSet;
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

/** The selection the entry's parameters (`candidateSet`, `ops`, `cases`, `rounds`, `idleSeconds`) give. */
export const sweepSelection = (values: ParameterValues): SweepSelection => ({
  candidateSet: choiceValue(values, "candidateSet", CANDIDATE_SETS),
  ops: choicesValue(values, "ops", SWEEP_OPS),
  cases: choicesValue(values, "cases", SWEEP_CASE_IDS),
  rounds: integerValue(values, "rounds"),
  idleSeconds: integerValue(values, "idleSeconds"),
});

/**
 * The plan of a selection over the chosen set's case table. Throws {@link ParameterError} for a
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
    caseIds: chosen.map(({ id }) => id),
    ops: sweepOps(chosen),
    candidateSet: selection.candidateSet,
    rounds: selection.rounds,
    idleSeconds: selection.idleSeconds,
  };
};
