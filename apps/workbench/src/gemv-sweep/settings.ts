/**
 * The GEMV sweep entry's settings: what a run is asked for (candidate set, cases, row counts,
 * candidates, the quick look's margin, rounds, idle seconds between cases) and its plan (the case
 * list `runGemvSweep` is called with, one case per call). Pure, so the case selection is tested
 * without a GPU.
 */

import {
  type GemvSweepCandidateSet,
  type GemvSweepCase,
  gemvSweepCandidateNames,
  gemvSweepCases,
} from "@karume/runtime/tune";

import {
  choicesValue,
  choiceValue,
  integerValue,
  numberValue,
  ParameterError,
  type ParameterValues,
} from "../entry.ts";

/** The candidate sets, the smaller first (the default). */
export const GEMV_CANDIDATE_SETS = [
  "quick",
  "full",
  "row-blocks",
] as const satisfies readonly GemvSweepCandidateSet[];

/** Every case id of the runtime's table, in its order (the options of `cases`). */
export const GEMV_CASE_IDS: readonly string[] = gemvSweepCases().map(({ id }) => id);

/** Every row count of the runtime's table, ascending, as text (the options of `rows`). */
export const GEMV_ROW_COUNTS: readonly string[] = [
  ...new Set(gemvSweepCases().flatMap(({ rows }) => rows)),
]
  .sort((a, b) => a - b)
  .map(String);

/** Every candidate name of the runtime's table (the options of `candidates`). */
export const GEMV_CANDIDATE_NAMES: readonly string[] = gemvSweepCandidateNames();

/** What a run is asked for: the entry's typed parameters. */
export type GemvSweepSelection = {
  readonly candidateSet: GemvSweepCandidateSet;
  /** The chosen cases; empty = every case. */
  readonly cases: readonly string[];
  /** The chosen row counts; empty = every row count of the set. */
  readonly rows: readonly number[];
  /** The candidates measured besides the references; empty = every candidate. */
  readonly candidates: readonly string[];
  /** The quick look's margin (0: off). */
  readonly quickLookMargin: number;
  readonly rounds: number;
  /** Idle time before every case but the first (cooling). */
  readonly idleSeconds: number;
};

/** A checked selection: what one sweep runs. */
export type GemvSweepPlan = {
  /** The cases in the table's order — one `runGemvSweep` call each. */
  readonly caseIds: readonly string[];
  readonly candidateSet: GemvSweepCandidateSet;
  readonly rows: readonly number[];
  readonly candidates: readonly string[];
  readonly quickLookMargin: number;
  readonly rounds: number;
  readonly idleSeconds: number;
};

/**
 * The selection the entry's parameters (`candidateSet`, `cases`, `rows`, `candidates`,
 * `quickLookMargin`, `rounds`, `idleSeconds`) give. A margin between 0 and 1 is a
 * {@link ParameterError}: it would screen out every kernel but the references.
 */
export const gemvSweepSelection = (values: ParameterValues): GemvSweepSelection => {
  const quickLookMargin = numberValue(values, "quickLookMargin");
  if (quickLookMargin > 0 && quickLookMargin < 1) {
    throw new ParameterError(
      "quickLookMargin",
      `${quickLookMargin} is neither 0 (the quick look off) nor at least 1`,
    );
  }
  return {
    candidateSet: choiceValue(values, "candidateSet", GEMV_CANDIDATE_SETS),
    cases: choicesValue(values, "cases", GEMV_CASE_IDS),
    rows: choicesValue(values, "rows", GEMV_ROW_COUNTS).map(Number),
    candidates: choicesValue(values, "candidates", GEMV_CANDIDATE_NAMES),
    quickLookMargin,
    rounds: integerValue(values, "rounds"),
    idleSeconds: integerValue(values, "idleSeconds"),
  };
};

/** The plan of a selection over the case table, in the table's order. */
export const planGemvSweep = (
  selection: GemvSweepSelection,
  cases: readonly GemvSweepCase[] = gemvSweepCases(),
): GemvSweepPlan => ({
  caseIds: cases
    .map(({ id }) => id)
    .filter((id) => selection.cases.length === 0 || selection.cases.includes(id)),
  candidateSet: selection.candidateSet,
  rows: selection.rows,
  candidates: selection.candidates,
  quickLookMargin: selection.quickLookMargin,
  rounds: selection.rounds,
  idleSeconds: selection.idleSeconds,
});
