/**
 * The GEMV sweep entry's settings: its declared parameters, what a run is asked for (candidate set,
 * cases, row counts, candidates, the quick look's margin, rounds, idle seconds between cases) and
 * its plan (the case list `runGemvSweep` is called with, one case per call — the sentinels first for
 * the `explore` set). Pure, so the case selection is tested without a GPU.
 */

import {
  type GemvSweepCandidateSet,
  type GemvSweepCase,
  gemvExploreOrder,
  gemvSweepCandidateNames,
  gemvSweepCases,
  gemvSweepQuickLookMargin,
} from "@karume/runtime/tune";

import {
  choicesValue,
  choiceValue,
  integerValue,
  type Parameter,
  ParameterError,
  type ParameterValues,
  textValue,
} from "../parameters.ts";

/** The candidate sets, the smaller first (the default). */
export const GEMV_CANDIDATE_SETS = [
  "quick",
  "full",
  "row-blocks",
  "explore",
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

/** The entry id of the GEMV sweep: the workbench's entry and the record's `entry.id`. */
export const GEMV_SWEEP_ENTRY_ID = "gemv-sweep";

/**
 * The GEMV sweep entry's parameters, in the order a record's `entry.parameters` holds them: the
 * workbench's entry declares them, and the benchmark app parses its run's values with them.
 */
export const GEMV_SWEEP_PARAMETERS: readonly Parameter[] = [
  {
    kind: "choice",
    name: "candidateSet",
    label:
      "Candidate set (quick: M = 1; full: M = 1, 4 and 8, the whole grid; row-blocks: M = 16 and 64; explore: M = 1, 4 and 8, the whole grid on the sentinels, the predicted candidates elsewhere)",
    options: GEMV_CANDIDATE_SETS,
    default: "quick",
  },
  {
    kind: "choices",
    name: "cases",
    label: "Cases (none: every case)",
    options: GEMV_CASE_IDS,
    default: [],
  },
  {
    kind: "choices",
    name: "rows",
    label: "Row counts (none: every row count of the set)",
    options: GEMV_ROW_COUNTS,
    default: [],
  },
  {
    kind: "choices",
    name: "candidates",
    label: "Candidates besides the references (none: every candidate; the explore set takes none)",
    options: GEMV_CANDIDATE_NAMES,
    default: [],
  },
  {
    // Text, so that empty can mean the set's default (×1.25; ×1.10 for explore)
    kind: "text",
    name: "quickLookMargin",
    label:
      "Quick look margin (empty: the set's default, ×1.25 and ×1.10 for explore; 0: off — every kernel is timed in full)",
    default: "",
  },
  {
    kind: "integer",
    name: "rounds",
    label: "Rounds per kernel (the min is kept)",
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

/** What a run is asked for: the entry's typed parameters. */
export type GemvSweepSelection = {
  readonly candidateSet: GemvSweepCandidateSet;
  /** The chosen cases; empty = every case. */
  readonly cases: readonly string[];
  /** The chosen row counts; empty = every row count of the set. */
  readonly rows: readonly number[];
  /** The candidates measured besides the references; empty = every candidate. */
  readonly candidates: readonly string[];
  /** The quick look's margin (0: off; absent: the set's default). */
  readonly quickLookMargin?: number;
  readonly rounds: number;
  /** Idle time before every case but the first (cooling). */
  readonly idleSeconds: number;
};

/** A checked selection: what one sweep runs. */
export type GemvSweepPlan = {
  /**
   * The cases in the table's order — one `runGemvSweep` call each; for the `explore` set the
   * sentinels of the table first (each part in the table's order).
   */
  readonly caseIds: readonly string[];
  readonly candidateSet: GemvSweepCandidateSet;
  readonly rows: readonly number[];
  readonly candidates: readonly string[];
  /** The margin every call runs with: the one given, or the set's default. */
  readonly quickLookMargin: number;
  readonly rounds: number;
  readonly idleSeconds: number;
};

/**
 * The quick look's margin of the parameter's text: none for an empty text (the set's default), else
 * a number that is 0 or at least 1 (between 0 and 1 it would screen out every kernel but the
 * references) — anything else is a {@link ParameterError}.
 */
const marginOf = (text: string): number | undefined => {
  if (text === "") return undefined;
  const margin = Number(text);
  if (text.trim() !== text || !Number.isFinite(margin) || margin < 0) {
    throw new ParameterError(
      "quickLookMargin",
      `expected empty (the set's default), 0 or a number of at least 1, got "${text}"`,
    );
  }
  if (margin > 0 && margin < 1) {
    throw new ParameterError(
      "quickLookMargin",
      `${margin} is neither 0 (the quick look off) nor at least 1`,
    );
  }
  return margin;
};

/**
 * The selection the entry's parameters (`candidateSet`, `cases`, `rows`, `candidates`,
 * `quickLookMargin`, `rounds`, `idleSeconds`) give. A margin that is not a number, one between 0
 * and 1, and candidates named for the `explore` set (which chooses its own) are a
 * {@link ParameterError}.
 */
export const gemvSweepSelection = (values: ParameterValues): GemvSweepSelection => {
  const quickLookMargin = marginOf(textValue(values, "quickLookMargin"));
  const candidateSet = choiceValue(values, "candidateSet", GEMV_CANDIDATE_SETS);
  const candidates = choicesValue(values, "candidates", GEMV_CANDIDATE_NAMES);
  if (candidateSet === "explore" && candidates.length > 0) {
    throw new ParameterError(
      "candidates",
      "the explore set chooses its candidates from its sentinels: leave candidates empty",
    );
  }
  return {
    candidateSet,
    cases: choicesValue(values, "cases", GEMV_CASE_IDS),
    rows: choicesValue(values, "rows", GEMV_ROW_COUNTS).map(Number),
    candidates,
    ...(quickLookMargin === undefined ? {} : { quickLookMargin }),
    rounds: integerValue(values, "rounds"),
    idleSeconds: integerValue(values, "idleSeconds"),
  };
};

/**
 * The plan of a selection over the case table: the cases in the table's order (the sentinels first
 * for the `explore` set, as one `runGemvSweep` call over them measures them) and the margin the set
 * runs with (`gemvSweepQuickLookMargin`: the one given, else the set's default).
 */
export const planGemvSweep = (
  selection: GemvSweepSelection,
  cases: readonly GemvSweepCase[] = gemvSweepCases(),
): GemvSweepPlan => {
  const chosen = cases.filter(
    ({ id }) => selection.cases.length === 0 || selection.cases.includes(id),
  );
  return {
    caseIds: (selection.candidateSet === "explore" ? gemvExploreOrder(chosen) : chosen).map(
      ({ id }) => id,
    ),
    candidateSet: selection.candidateSet,
    rows: selection.rows,
    candidates: selection.candidates,
    quickLookMargin: gemvSweepQuickLookMargin(selection.candidateSet, selection.quickLookMargin),
    rounds: selection.rounds,
    idleSeconds: selection.idleSeconds,
  };
};
