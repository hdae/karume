/**
 * The session A/B entries' results: the verdict, which is identity (every arm's outputs equal to the
 * base's first run, over every run), and the summary lines — the judgement of each compared arm (a
 * measurement: adopt, reject or inconclusive), the counts, and identity. Pure, so `vp test` covers
 * it.
 */

import type {
  SessionAbArmRecord,
  SessionAbJudgement,
  SessionAbRecord,
  SessionAbVerdict,
} from "@karume/runtime/tune";

import type { EntryResult, Verdict } from "../entry.ts";

import type {
  SessionAbArmCounts,
  SessionAbGemmaData,
  SessionAbMossData,
  SessionAbVaeData,
} from "./data.ts";
import { countedPhaseOf } from "./gemma-arm.ts";

/** `fail` when any arm's outputs differ from the base's first run in any run, else `pass`. */
export const identityVerdict = (record: SessionAbRecord): Verdict =>
  record.arms.every((arm) => arm.identity.worst.failures === 0) ? "pass" : "fail";

/** "identical (N runs)", or how many words of which run differ from the base's first, and where first. */
const identityText = (arm: SessionAbArmRecord): string => {
  const { runs, worst } = arm.identity;
  if (worst.failures === 0) return `identical (${runs} runs)`;
  const where = `round ${worst.round}, visit ${worst.visit}, run ${worst.run}`;
  const first =
    worst.first === undefined
      ? ""
      : `; first at ${worst.first.output}[${worst.first.index}] (${worst.first.cause}): expected ${worst.first.expected}, got ${worst.first.actual}`;
  return `${worst.failures} of ${worst.words} words differ (${where}${first})`;
};

const identityLine = (record: SessionAbRecord): string =>
  `Identity against the base's first run: ${record.arms
    .map((arm) => `${arm.name} ${identityText(arm)}`)
    .join("; ")}`;

/** The word a verdict is shown with. */
const VERDICT_LABELS: Readonly<Record<SessionAbVerdict, string>> = {
  adopt: "Adopt",
  reject: "Reject",
  inconclusive: "Inconclusive",
};

const signedMs = (value: number): string =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)} ms`;

const gainText = (judgement: SessionAbJudgement): string => `${(judgement.gain * 100).toFixed(2)}%`;

const intervalLabel = (judgement: SessionAbJudgement): string =>
  `Interval of the median difference (${(judgement.intervalCoverage * 100).toFixed(1)}% coverage)`;

const intervalText = (judgement: SessionAbJudgement): string =>
  `${signedMs(judgement.deltaLowMs)} to ${signedMs(judgement.deltaHighMs)}`;

/**
 * One compared arm's judgement in one line: the verdict, the gain and the interval of the median
 * difference.
 */
export const judgementLine = (judgement: SessionAbJudgement): string =>
  `${judgement.arm} against ${judgement.base}: ${VERDICT_LABELS[judgement.verdict]} · gain ${gainText(judgement)} · ${intervalLabel(judgement).toLowerCase()} ${intervalText(judgement)}`;

const signed = (value: number): string => `${value >= 0 ? "+" : "−"}${Math.abs(value)}`;

/**
 * The counts in one line: each of `rules`' hits per arm (0 where an arm has no such counter), each
 * arm's dispatches and every other arm's difference from the first (the base).
 */
const countsLine = (
  heading: string,
  counts: readonly SessionAbArmCounts[],
  rules: readonly string[],
): string => {
  const [base] = counts;
  const hits = rules.map(
    (rule) =>
      `${rule} hits ${counts.map(({ arm, fusions }) => `${arm} ${fusions[rule] ?? 0}`).join(", ")}`,
  );
  const dispatches = `dispatches ${counts.map(({ arm, dispatches }) => `${arm} ${dispatches}`).join(", ")}`;
  const differences =
    base === undefined
      ? []
      : counts
          .slice(1)
          .map(
            ({ arm, dispatches }) => `${arm} − ${base.arm} ${signed(dispatches - base.dispatches)}`,
          );
  return `${heading}: ${[...hits, dispatches].join("; ")}${differences.length === 0 ? "" : ` (${differences.join(", ")})`}`;
};

/**
 * An arm's counts as the runner's record keeps them (a target whose arms report their counts to
 * the runner, the `sessionArm`s of the Session targets); `undefined` when the arm reported none.
 */
export const recordedCounts = (arm: SessionAbArmRecord): SessionAbArmCounts | undefined =>
  arm.fusions === undefined || arm.dispatches === undefined
    ? undefined
    : { arm: arm.name, fusions: arm.fusions, dispatches: arm.dispatches };

/** The runner record's counts per arm; the data builders refused a record without them. */
const runnerCounts = (arms: readonly SessionAbArmRecord[]): readonly SessionAbArmCounts[] =>
  arms.map((arm) => {
    const counts = recordedCounts(arm);
    if (counts === undefined) throw new Error(`The arm ${arm.name} reported no counts`);
    return counts;
  });

/** The result of an A/B on a Session target (the MOSS-Image DiT, a VAE). */
export const sessionAbComponentResult = (
  data: SessionAbMossData | SessionAbVaeData,
): EntryResult => {
  const rules =
    data.target === "moss-dit-forward" ? [data.rule.name] : data.rules.map(({ name }) => name);
  return {
    verdict: identityVerdict(data.record),
    summary: [
      ...data.judgements.map(judgementLine),
      countsLine("Counts per forward", runnerCounts(data.record.arms), rules),
      identityLine(data.record),
    ],
    data,
  };
};

/** The result of an A/B on the Gemma 4 decode (the probe's counts when it ran). */
export const sessionAbGemmaResult = (data: SessionAbGemmaData): EntryResult => ({
  verdict: identityVerdict(data.record),
  summary: [
    ...data.judgements.map(judgementLine),
    ...(data.counts === undefined
      ? []
      : [
          countsLine(`Probe counts (first ${countedPhaseOf(data.request)} run)`, data.counts, [
            data.rule.name,
          ]),
        ]),
    identityLine(data.record),
  ],
  data,
});
