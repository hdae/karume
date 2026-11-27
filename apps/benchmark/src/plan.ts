/**
 * What a run measures: the GEMV sweep's and the geometry sweep's `explore` sets at their defaults
 * (rounds, margins) with no idle time between cases (`IDLE_SECONDS`), each sweep planned by the
 * shared settings exactly as the workbench's entry plans it, and the run's cases in order with the
 * kind the estimate reads them by. The URL may narrow a run for a short trial (`?cases=`,
 * comma-separated ids of either table) and set the idle time (`?idleSeconds=`); nothing else. Pure.
 *
 * SPEC: docs/spec/benchmark.md#what-a-run-measures
 */

import {
  gemvSentinels,
  gemvSweepCases,
  geometryCaseClass,
  geometrySweepCases,
} from "@karume/runtime/tune";
import {
  GEMV_CASE_IDS,
  GEMV_SWEEP_ENTRY_ID,
  GEMV_SWEEP_PARAMETERS,
  type GemvSweepPlan,
  gemvSweepSelection,
  planGemvSweep,
} from "@karume/sweep-app-support/gemv-sweep/settings";
import {
  GEOMETRY_SWEEP_ENTRY_ID,
  GEOMETRY_SWEEP_PARAMETERS,
  planSweep,
  type SweepPlan,
  sweepSelection,
} from "@karume/sweep-app-support/geometry-sweep/settings";
import {
  integerValue,
  ParameterError,
  type ParameterValues,
  parseParameters,
} from "@karume/sweep-app-support/parameters";

/** The two sweeps of a run, in the order they run. */
export type SweepId = "gemv" | "geometry";

/**
 * What a case is for the estimate: a GEMV case is a `sentinel` (measured with every candidate) or
 * `other` (the candidates its sentinels predict); a geometry case is the `first-of-class` (every
 * candidate of `full`) or `later` (what the class's earlier cases left).
 */
export type CaseKind = "sentinel" | "other" | "first-of-class" | "later";

/** One case of the run, in run order. */
export type RunCase = {
  readonly sweep: SweepId;
  readonly caseId: string;
  readonly kind: CaseKind;
};

/** A sweep of the run: its workbench entry, the typed parameters its record holds and its plan. */
export type PlannedSweep =
  | {
      readonly sweep: "gemv";
      readonly entryId: typeof GEMV_SWEEP_ENTRY_ID;
      readonly parameters: ParameterValues;
      readonly plan: GemvSweepPlan;
    }
  | {
      readonly sweep: "geometry";
      readonly entryId: typeof GEOMETRY_SWEEP_ENTRY_ID;
      readonly parameters: ParameterValues;
      readonly plan: SweepPlan;
    };

/** What the URL asked for (absent: not given). */
export type Trial = {
  /** Case ids of either table; a sweep none of them belongs to does not run. */
  readonly cases?: readonly string[];
  readonly idleSeconds?: number;
};

export type BenchmarkPlan = {
  /** The sweeps that run, in order. */
  readonly sweeps: readonly PlannedSweep[];
  /** Every case of the run, sweep after sweep, each sweep's in its plan's order. */
  readonly cases: readonly RunCase[];
  /** Idle before every case of the run but the first (the second sweep's first case too). */
  readonly idleSeconds: number;
  readonly trial: Trial;
};

/** Which sweeps a person chose. */
export type SweepChoice = Readonly<Record<SweepId, boolean>>;

const TRIAL_KEYS = ["cases", "idleSeconds"] as const;

/**
 * The idle time before every case of a run but the first when the trial gives none, for both
 * sweeps: none. The warm start that opens every group brings the device to a steady state whatever
 * came before it; on an RTX 3080 Ti the GEMV explore set's derivation moved no more without the
 * idle time than between two runs with it (the measurement: `gemvSweepIdleSeconds` of the shared
 * GEMV settings), and the run is a third shorter.
 */
const IDLE_SECONDS = 0;

/** The geometry explore set's table (`full`'s). */
const geometryCaseIds = (): readonly string[] => geometrySweepCases("explore").map(({ id }) => id);

/**
 * The trial a URL query asks for: `cases` (comma-separated ids of either sweep's table, at least one)
 * and `idleSeconds` (an integer of at least 0, the sweeps' own bounds). Any other parameter, an
 * unknown case id, an empty list and a value that does not fit throw {@link ParameterError}.
 */
export const trialOfQuery = (query: string): Trial => {
  const params = new URLSearchParams(query);
  for (const key of params.keys()) {
    if (!TRIAL_KEYS.some((known) => known === key)) {
      throw new ParameterError(key, "this page takes only cases and idleSeconds");
    }
  }
  const casesText = params.get("cases");
  const idleText = params.get("idleSeconds");
  let cases: string[] | undefined;
  if (casesText !== null) {
    cases = casesText.split(",").filter((id) => id !== "");
    if (cases.length === 0) throw new ParameterError("cases", "name at least one case");
    const known = new Set([...GEMV_CASE_IDS, ...geometryCaseIds()]);
    const unknown = cases.filter((id) => !known.has(id));
    if (unknown.length > 0) {
      throw new ParameterError(
        "cases",
        `${unknown.join(", ")} ${unknown.length === 1 ? "is no case" : "are no cases"} of either sweep`,
      );
    }
  }
  const idleSeconds =
    idleText === null
      ? undefined
      : integerValue(
          parseParameters(GEOMETRY_SWEEP_PARAMETERS, { idleSeconds: idleText }),
          "idleSeconds",
        );
  return {
    ...(cases === undefined ? {} : { cases }),
    ...(idleSeconds === undefined ? {} : { idleSeconds }),
  };
};

/** The parameter text a sweep's record is parsed from: the explore set, its cases, the idle time. */
const givenOf = (
  caseIds: readonly string[] | undefined,
  idleSeconds: number,
): Record<string, string> => ({
  candidateSet: "explore",
  ...(caseIds === undefined ? {} : { cases: caseIds.join(",") }),
  idleSeconds: String(idleSeconds),
});

/** The cases of a sweep the trial names (undefined: no narrowing; empty: the sweep does not run). */
const trialCasesIn = (trial: Trial, table: readonly string[]): readonly string[] | undefined =>
  trial.cases?.filter((id) => table.includes(id));

/** The kind of each case of a planned sweep, in its plan's order. */
const runCasesOf = (sweep: PlannedSweep): RunCase[] => {
  if (sweep.sweep === "gemv") {
    const sentinels = new Set(gemvSentinels(gemvSweepCases()));
    return sweep.plan.caseIds.map((caseId) => ({
      sweep: "gemv",
      caseId,
      kind: sentinels.has(caseId) ? "sentinel" : "other",
    }));
  }
  const seen = new Set<string>();
  return sweep.plan.caseIds.map((caseId) => {
    const caseClass = geometryCaseClass(caseId);
    const first = !seen.has(caseClass);
    seen.add(caseClass);
    return { sweep: "geometry", caseId, kind: first ? "first-of-class" : "later" };
  });
};

/**
 * The run of the chosen sweeps under a trial: each sweep's parameters parsed by its entry's own
 * declarations (so its record's `entry.parameters` is what the workbench writes for the same run)
 * and planned by the shared settings, both with the trial's idle time, else {@link IDLE_SECONDS}. A
 * sweep that is not chosen, or that the trial names no case of, does not run.
 */
export const planBenchmark = (choice: SweepChoice, trial: Trial): BenchmarkPlan => {
  const idleSeconds = trial.idleSeconds ?? IDLE_SECONDS;
  const sweeps: PlannedSweep[] = [];
  const gemvCases = trialCasesIn(trial, GEMV_CASE_IDS);
  if (choice.gemv && gemvCases?.length !== 0) {
    const parameters = parseParameters(GEMV_SWEEP_PARAMETERS, givenOf(gemvCases, idleSeconds));
    sweeps.push({
      sweep: "gemv",
      entryId: GEMV_SWEEP_ENTRY_ID,
      parameters,
      plan: planGemvSweep(gemvSweepSelection(parameters)),
    });
  }
  const geometryCases = trialCasesIn(trial, geometryCaseIds());
  if (choice.geometry && geometryCases?.length !== 0) {
    const parameters = parseParameters(
      GEOMETRY_SWEEP_PARAMETERS,
      givenOf(geometryCases, idleSeconds),
    );
    const selection = sweepSelection(parameters);
    sweeps.push({
      sweep: "geometry",
      entryId: GEOMETRY_SWEEP_ENTRY_ID,
      parameters,
      plan: planSweep(selection, geometrySweepCases(selection.candidateSet)),
    });
  }
  return { sweeps, cases: sweeps.flatMap(runCasesOf), idleSeconds, trial };
};
