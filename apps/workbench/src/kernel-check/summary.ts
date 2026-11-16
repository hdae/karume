/**
 * What a person reads first about a kernel-check run: the overall verdict with the count of each
 * check verdict, and one line per check that is not a pass — for a failure, its first failing word.
 * Pure, over the same outcomes the record body is built from.
 */

import type { EntryResult, Verdict as EntryVerdict } from "../entry.ts";
import {
  type CheckOutcomes,
  hexWord,
  KERNEL_CHECK_IDS,
  type KernelCheckId,
  kernelCheckRecord,
  overallVerdict,
  type OverallVerdict,
  productVerdictOf,
  summarizePart,
  type Verdict,
  verdictOf,
} from "./verdict.ts";

type CheckVerdict = Verdict | "not-run";

const VERDICT_TEXT: Readonly<Record<CheckVerdict | OverallVerdict, string>> = {
  pass: "PASS",
  fail: "FAIL",
  error: "ERROR",
  "not-available": "not available",
  incomplete: "INCOMPLETE",
  "not-run": "not run",
  none: "not run",
};

/** The entry contract's verdict: a run that settled nothing (`none`) proves nothing, like an incomplete one. */
export const entryVerdict = (overall: OverallVerdict): EntryVerdict =>
  overall === "pass" || overall === "fail" ? overall : "incomplete";

const COUNTED: readonly (readonly [CheckVerdict, string])[] = [
  ["pass", "passed"],
  ["fail", "failed"],
  ["incomplete", "incomplete"],
  ["not-available", "not available"],
  ["error", "error"],
  ["not-run", "not run"],
];

/** The first failing word over the check's parts, as one phrase (absent when no word failed). */
const firstFailingWord = (outcomes: CheckOutcomes, id: KernelCheckId): string => {
  const outcome = outcomes[id];
  if (outcome?.kind !== "done") return "";
  for (const part of outcome.report.parts.map(summarizePart)) {
    const first = part.firstFailure;
    if (first === undefined) continue;
    const { failure } = first;
    return ` — ${part.id}: ${failure.at} failure in ${first.case} — ${first.criterion} — word ${failure.index}: expected ${hexWord(failure.expected)}, actual ${hexWord(failure.actual)}${failure.note === undefined ? "" : ` (${failure.note})`}`;
  }
  return "";
};

/** The summary lines of a run's outcomes (every check id, a missing outcome read as not run). */
export const summaryLines = (outcomes: CheckOutcomes): readonly string[] => {
  const verdicts = KERNEL_CHECK_IDS.map((id) => {
    const outcome = outcomes[id];
    return { id, verdict: outcome === undefined ? "not-run" : verdictOf(outcome) } as const;
  });
  const counts = COUNTED.map(
    ([verdict, label]) =>
      `${verdicts.filter((check) => check.verdict === verdict).length} ${label}`,
  );
  return [
    `${VERDICT_TEXT[overallVerdict(outcomes)]} — checks: ${counts.join(", ")}`,
    ...verdicts
      .filter(({ verdict }) => verdict !== "pass")
      .map(({ id, verdict }) => {
        const outcome = outcomes[id];
        const product = outcome === undefined ? undefined : productVerdictOf(outcome);
        return `${id}: ${VERDICT_TEXT[verdict]}${product === undefined ? "" : ` (outside the K-95 classes: ${VERDICT_TEXT[product]})`}${firstFailingWord(outcomes, id)}`;
      }),
  ];
};

/** The entry's result: the contract verdict, the summary and the record body. */
export const kernelCheckResult = (
  adapterFeatures: readonly string[],
  outcomes: CheckOutcomes,
): EntryResult => {
  const data = kernelCheckRecord(adapterFeatures, outcomes);
  return { verdict: entryVerdict(data.verdict), summary: summaryLines(outcomes), data };
};
