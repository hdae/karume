/**
 * The GEMV sweep entry's settings: what a run is asked for (candidate set, cases, rounds, idle
 * seconds between cases) and its plan (the case list `runGemvSweep` is called with, one case per
 * call). Pure, so the case selection is tested without a GPU.
 */

import {
  type GemvSweepCandidateSet,
  type GemvSweepCase,
  gemvSweepCases,
} from "@karume/runtime/tune";

import { choicesValue, choiceValue, integerValue, type ParameterValues } from "../entry.ts";

/** The candidate sets, the smaller first (the default). */
export const GEMV_CANDIDATE_SETS = [
  "quick",
  "full",
] as const satisfies readonly GemvSweepCandidateSet[];

/** Every case id of the runtime's table, in its order (the options of `cases`). */
export const GEMV_CASE_IDS: readonly string[] = gemvSweepCases().map(({ id }) => id);

/** What a run is asked for: the entry's typed parameters. */
export type GemvSweepSelection = {
  readonly candidateSet: GemvSweepCandidateSet;
  /** The chosen cases; empty = every case. */
  readonly cases: readonly string[];
  readonly rounds: number;
  /** Idle time before every case but the first (cooling). */
  readonly idleSeconds: number;
};

/** A checked selection: what one sweep runs. */
export type GemvSweepPlan = {
  /** The cases in the table's order — one `runGemvSweep` call each. */
  readonly caseIds: readonly string[];
  readonly candidateSet: GemvSweepCandidateSet;
  readonly rounds: number;
  readonly idleSeconds: number;
};

/** The selection the entry's parameters (`candidateSet`, `cases`, `rounds`, `idleSeconds`) give. */
export const gemvSweepSelection = (values: ParameterValues): GemvSweepSelection => ({
  candidateSet: choiceValue(values, "candidateSet", GEMV_CANDIDATE_SETS),
  cases: choicesValue(values, "cases", GEMV_CASE_IDS),
  rounds: integerValue(values, "rounds"),
  idleSeconds: integerValue(values, "idleSeconds"),
});

/** The plan of a selection over the case table, in the table's order. */
export const planGemvSweep = (
  selection: GemvSweepSelection,
  cases: readonly GemvSweepCase[] = gemvSweepCases(),
): GemvSweepPlan => ({
  caseIds: cases
    .map(({ id }) => id)
    .filter((id) => selection.cases.length === 0 || selection.cases.includes(id)),
  candidateSet: selection.candidateSet,
  rounds: selection.rounds,
  idleSeconds: selection.idleSeconds,
});
