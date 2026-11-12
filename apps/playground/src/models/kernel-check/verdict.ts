/**
 * The verdicts and the record of the kernel-check page, from the structured results of
 * `@karume/kernel-checks` (pure: the page and `vp test` see the same functions).
 *
 * A check passes only when it ran, returned the parts, cases and mutants it exports the counts of,
 * and every criterion of every case has 0 failing words, every requirement is met and every mutant
 * mismatched (on its one case and on every case its code path runs on). A check the device cannot
 * run (a missing feature) is "not available" — never a pass — and keeps the whole page from
 * reading PASS, as does a check that was not run. A check with a part the device cannot run
 * (`CheckPart.notRun`) and nothing failing is "incomplete", which keeps the page from PASS too.
 */

import type {
  CheckCase,
  CheckPart,
  ExpectedCases,
  FoldClass,
  MutantMatrixRow,
  MutantResult,
  Requirement,
  WordFailure,
} from "@karume/kernel-checks";

import type { RecordEnvironment } from "@/lib/environment";

export const KERNEL_CHECK_IDS = [
  "gemv-subgroup32",
  "rms-subgroup32",
  "rms-norm-prefetch",
  "state-attention-rewrites",
  "gemv-integer",
  "gemv-planes",
  "state-kv16",
  "score-f16-store",
  "state-bidirectional",
  "expert-linear",
  "fusion-twins",
  "attention-online-tiled",
] as const;
export type KernelCheckId = (typeof KERNEL_CHECK_IDS)[number];

/** What one check returned: its parts and, for the state-attention check, mutants and fold classes. */
export type CheckReport = {
  readonly parts: readonly CheckPart[];
  /** The case count each part MUST have, by part id (the counts the check exports). */
  readonly expectedCases: ExpectedCases;
  /** The mutant count the one-case list and the matrix MUST each have (absent: none). */
  readonly expectedMutants?: number;
  readonly mutants?: readonly MutantResult[];
  readonly mutantMatrix?: readonly MutantMatrixRow[];
  readonly foldClasses?: readonly FoldClass[];
};

export type CheckOutcome =
  | { readonly kind: "not-available"; readonly reason: string }
  | { readonly kind: "error"; readonly message: string; readonly seconds: number }
  | { readonly kind: "done"; readonly report: CheckReport; readonly seconds: number };

export type CheckOutcomes = Readonly<Partial<Record<KernelCheckId, CheckOutcome>>>;

/** `incomplete`: the check ran and nothing failed, but a part of it could not run on this device. */
export type Verdict = "pass" | "fail" | "incomplete" | "not-available" | "error";

/** The whole page: `incomplete` = nothing failed, but a check could not run here or was not run. */
export type OverallVerdict = "pass" | "fail" | "incomplete" | "none";

/** The first failing word of a part, with where it is. */
export type FirstFailure = {
  readonly case: string;
  readonly criterion: string;
  readonly failure: WordFailure;
};

export type PartSummary = {
  readonly id: string;
  readonly title: string;
  readonly cases: number;
  readonly failingCases: number;
  /** Failing words over every criterion of every case. */
  readonly mismatchingWords: number;
  readonly unmetRequirements: readonly Requirement[];
  readonly firstFailure?: FirstFailure;
  /** Why the device could not run the part (it has no cases). */
  readonly notRun?: string;
};

const caseFails = (testCase: CheckCase): boolean =>
  testCase.criteria.some(({ failures }) => failures > 0);

export const summarizePart = (part: CheckPart): PartSummary => {
  let mismatchingWords = 0;
  let firstFailure: FirstFailure | undefined;
  for (const testCase of part.cases) {
    for (const criterion of testCase.criteria) {
      mismatchingWords += criterion.failures;
      if (firstFailure === undefined && criterion.failure !== undefined) {
        firstFailure = {
          case: testCase.label,
          criterion: criterion.name,
          failure: criterion.failure,
        };
      }
    }
  }
  const summary = {
    id: part.id,
    title: part.title,
    cases: part.cases.length,
    failingCases: part.cases.filter(caseFails).length,
    mismatchingWords,
    unmetRequirements: part.requirements.filter(({ met }) => !met),
    ...(part.notRun === undefined ? {} : { notRun: part.notRun }),
  };
  return firstFailure === undefined ? summary : { ...summary, firstFailure };
};

/** A mutant passes when it changed at least one word (the comparison is not vacuous). */
export const mutantMismatched = (mutant: MutantResult): boolean => mutant.differingWords > 0;

/**
 * A mutant matrix row passes when the mutated code path ran on some case, every such case
 * mismatched and no case it does not run on did (the contraction-only cases carry no requirement).
 */
export const matrixRowPasses = (row: MutantMatrixRow): boolean =>
  row.acting > 0 && row.actingMatched.length === 0 && row.idleMismatched.length === 0;

/** A list's length against the expected one (an absent list or count is none). */
const listCountMismatch = (
  name: string,
  list: readonly unknown[] | undefined,
  expected = 0,
): readonly string[] => {
  const count = list?.length ?? 0;
  return count === expected ? [] : [`${name}: ${count}, expected ${expected}`];
};

/**
 * Where the report's parts, cases and mutants differ in number from what the check exports: a
 * part that ran fewer cases (none at all, say) compared fewer words, so its clean result proves less.
 * A part the device could not run (`notRun`) is returned but has no cases; the verdict reads it as
 * not run instead.
 */
export const countMismatches = (report: CheckReport): readonly string[] => {
  const { parts, expectedCases, expectedMutants } = report;
  const returned = new Set(parts.map(({ id }) => id));
  return [
    ...parts.flatMap(({ id, cases, notRun }) => {
      if (!Object.hasOwn(expectedCases, id)) return [`${id}: not a part of this check`];
      if (notRun !== undefined && cases.length === 0) return [];
      const expected = expectedCases[id];
      return cases.length === expected
        ? []
        : [`${id}: ${cases.length} cases, expected ${expected}`];
    }),
    ...Object.entries(expectedCases)
      .filter(([id]) => !returned.has(id))
      .map(([id, expected]) => `${id}: missing, expected ${expected} cases`),
    ...listCountMismatch("mutants", report.mutants, expectedMutants),
    ...listCountMismatch("mutant matrix rows", report.mutantMatrix, expectedMutants),
  ];
};

const resultsPass = ({ parts, mutants = [], mutantMatrix = [] }: CheckReport): boolean =>
  parts
    .map(summarizePart)
    .every(
      ({ failingCases, unmetRequirements }) => failingCases === 0 && unmetRequirements.length === 0,
    ) &&
  mutants.every(mutantMismatched) &&
  mutantMatrix.every(matrixRowPasses);

const reportPasses = (report: CheckReport): boolean =>
  countMismatches(report).length === 0 && resultsPass(report);

export const verdictOf = (outcome: CheckOutcome): Verdict => {
  if (outcome.kind !== "done") return outcome.kind;
  if (!reportPasses(outcome.report)) return "fail";
  return outcome.report.parts.some(({ notRun }) => notRun !== undefined) ? "incomplete" : "pass";
};

/** The groups of the fold classes kept only for K-95 (checked, but the product does not fold them). */
const k95Groups = (report: CheckReport): ReadonlySet<string> =>
  new Set(
    (report.foldClasses ?? [])
      .filter(({ use, productFoldHeads }) => use === "k95" && productFoldHeads === 1)
      .map(({ group }) => group),
  );

/**
 * The verdict of a check without the cases of the classes kept only for K-95, or undefined for a
 * check without fold classes. On D3D12 the strict verdict fails on E4B's F = 4 until K-95 lands;
 * this one still fails on anything the product runs (a selected fold class, ③', the fused kernel,
 * a requirement, a mutant) and on a count that differs from the check's (counted before the
 * filter).
 */
export const productVerdictOf = (outcome: CheckOutcome): "pass" | "fail" | undefined => {
  if (outcome.kind !== "done" || outcome.report.foldClasses === undefined) return undefined;
  const excluded = k95Groups(outcome.report);
  const parts = outcome.report.parts.map((part) => ({
    ...part,
    cases: part.cases.filter(({ group }) => group === undefined || !excluded.has(group)),
  }));
  return countMismatches(outcome.report).length === 0 && resultsPass({ ...outcome.report, parts })
    ? "pass"
    : "fail";
};

/** Over every check of the page: one that was not run counts like one that could not run. */
export const overallVerdict = (outcomes: CheckOutcomes): OverallVerdict => {
  const verdicts = KERNEL_CHECK_IDS.map((id) => {
    const outcome = outcomes[id];
    return outcome === undefined ? "not-run" : verdictOf(outcome);
  });
  if (verdicts.every((verdict) => verdict === "not-run")) return "none";
  if (verdicts.some((verdict) => verdict === "fail" || verdict === "error")) return "fail";
  if (
    verdicts.some(
      (verdict) => verdict === "incomplete" || verdict === "not-available" || verdict === "not-run",
    )
  ) {
    return "incomplete";
  }
  return "pass";
};

/** The checks without an outcome, in check order. */
export const notRunChecks = (outcomes: CheckOutcomes): readonly KernelCheckId[] =>
  KERNEL_CHECK_IDS.filter((id) => outcomes[id] === undefined);

/** The outcomes that exist, in check order. */
export const outcomeList = (
  outcomes: CheckOutcomes,
): readonly { readonly id: KernelCheckId; readonly outcome: CheckOutcome }[] =>
  KERNEL_CHECK_IDS.flatMap((id) => {
    const outcome = outcomes[id];
    return outcome === undefined ? [] : [{ id, outcome }];
  });

/**
 * The outcomes without those of the checks a run is about to run: until a check settles again it
 * reads as not run, so cancelling its re-run never leaves its earlier verdict standing.
 */
export const withoutOutcomes = (
  outcomes: CheckOutcomes,
  ids: readonly KernelCheckId[],
): CheckOutcomes =>
  Object.fromEntries(
    outcomeList(outcomes)
      .filter(({ id }) => !ids.includes(id))
      .map(({ id, outcome }) => [id, outcome]),
  );

/**
 * What the subgroup checks need and the device or WGSL lacks (`acquireGpu` requests the features only
 * when all three are there).
 */
export const subgroupShortfall = (
  adapterFeatures: ReadonlySet<string>,
  wgslLanguageFeatures: ReadonlySet<string>,
): readonly string[] => [
  ...["subgroups", "subgroup-size-control"].filter((feature) => !adapterFeatures.has(feature)),
  ...(wgslLanguageFeatures.has("subgroup_id") ? [] : ["WGSL subgroup_id"]),
];

/** How the product uses a fold class. */
export type FoldClassUse = "in use" | "kept for K-95" | "not selected";

export type FoldClassRow = FoldClass & {
  readonly status: FoldClassUse;
  readonly cases: number;
  readonly failingCases: number;
};

/**
 * The folded ①' classes with the product's use of each and how their cases fared: a FAIL confined
 * to classes the product does not fold (E4B's F = 4 on D3D12) leaves the product's choice intact.
 */
export const foldClassRows = (report: CheckReport): readonly FoldClassRow[] => {
  const cases = report.parts.flatMap((part) => part.cases);
  return (report.foldClasses ?? []).map((foldClass) => {
    const ofClass = cases.filter(({ group }) => group === foldClass.group);
    return {
      ...foldClass,
      status:
        foldClass.productFoldHeads > 1
          ? "in use"
          : foldClass.use === "k95"
            ? "kept for K-95"
            : "not selected",
      cases: ofClass.length,
      failingCases: ofClass.filter(caseFails).length,
    };
  });
};

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const KERNEL_CHECK_RECORD_FORMAT = "karume-playground-kernel-check/1";

export type CheckRecord =
  | {
      readonly id: KernelCheckId;
      readonly verdict: "not-run";
    }
  | {
      readonly id: KernelCheckId;
      readonly verdict: "not-available";
      readonly reason: string;
    }
  | {
      readonly id: KernelCheckId;
      readonly verdict: "error";
      readonly seconds: number;
      readonly error: string;
    }
  | {
      readonly id: KernelCheckId;
      /** {@link verdictOf}: `incomplete` when a part was not run here (its `notRun` says why). */
      readonly verdict: "pass" | "fail" | "incomplete";
      /** {@link productVerdictOf}: present for the check with fold classes. */
      readonly productVerdict?: "pass" | "fail";
      readonly seconds: number;
      /** {@link countMismatches} (empty when the check returned what it exports the counts of). */
      readonly countMismatches: readonly string[];
      readonly parts: readonly (PartSummary & {
        readonly requirements: readonly Requirement[];
        /** Every case with its criteria (failing words and the first or worst failing word). */
        readonly results: readonly CheckCase[];
      })[];
      readonly mutants?: readonly MutantResult[];
      readonly mutantMatrix?: readonly MutantMatrixRow[];
      readonly foldClasses?: readonly FoldClassRow[];
    };

export type KernelCheckRecord = {
  readonly format: typeof KERNEL_CHECK_RECORD_FORMAT;
  readonly page: "kernel-check";
} & RecordEnvironment & {
    /** The adapter's features (the device's are in `device.features`). */
    readonly adapterFeatures: readonly string[];
    readonly verdict: OverallVerdict;
    readonly checks: readonly CheckRecord[];
  };

const checkRecord = (id: KernelCheckId, outcome: CheckOutcome | undefined): CheckRecord => {
  if (outcome === undefined) return { id, verdict: "not-run" };
  if (outcome.kind === "not-available") {
    return { id, verdict: "not-available", reason: outcome.reason };
  }
  if (outcome.kind === "error") {
    return { id, verdict: "error", seconds: outcome.seconds, error: outcome.message };
  }
  const { report } = outcome;
  const ran = verdictOf(outcome);
  const verdict = ran === "pass" || ran === "incomplete" ? ran : "fail";
  const productVerdict = productVerdictOf(outcome);
  return {
    id,
    verdict,
    ...(productVerdict === undefined ? {} : { productVerdict }),
    seconds: outcome.seconds,
    countMismatches: countMismatches(report),
    parts: report.parts.map((part) => ({
      ...summarizePart(part),
      requirements: part.requirements,
      results: part.cases,
    })),
    ...(report.mutants === undefined ? {} : { mutants: report.mutants }),
    ...(report.mutantMatrix === undefined ? {} : { mutantMatrix: report.mutantMatrix }),
    ...(report.foldClasses === undefined ? {} : { foldClasses: foldClassRows(report) }),
  };
};

export const kernelCheckRecord = (
  environment: RecordEnvironment,
  adapterFeatures: readonly string[],
  outcomes: CheckOutcomes,
): KernelCheckRecord => {
  return {
    format: KERNEL_CHECK_RECORD_FORMAT,
    page: "kernel-check",
    ...environment,
    adapterFeatures: [...adapterFeatures].sort(),
    verdict: overallVerdict(outcomes),
    // Every check, a not-run one included: a record of a partial run never reads as a full one
    checks: KERNEL_CHECK_IDS.map((id) => checkRecord(id, outcomes[id])),
  };
};

/** A u32 word as `0x` + 8 hex digits. */
export const hexWord = (word: number): string => `0x${word.toString(16).padStart(8, "0")}`;
