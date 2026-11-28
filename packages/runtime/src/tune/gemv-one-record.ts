/**
 * The rule of one steady record, as a pure function of the evidence a derivation reads for one case
 * and class and of the margin ({@link decideOneRecord}): what the derivation (gemv-derive.ts) calls
 * for every decision whose shape a steady record measured, and what a re-decision of its output
 * under another margin calls again on the evidence the output carries — one code path, so the two
 * cannot drift apart. Every reading is margin-free; the margin enters here only.
 *
 * - A reading is evidence when its reference's re-timed pass (`retime`: that pass ÷ the rounds'
 *   minimum) departs from 1 by no more than the bar a gain must clear, margin × gap × referenceGap,
 *   in either direction; a reference whose re-timed pass read no time (`unread`) tested nothing, so
 *   no reading against it is evidence (a record without the re-timing has none to test).
 * - At M = 1 a candidate qualifies when its check held, at least one reading is evidence, and every
 *   reading that is evidence has a gain of at least margin × gap × referenceGap (its break-even
 *   margin at least the margin).
 * - A parallel row's row block (or none) replaces its reference row's when, at the other row counts
 *   where the two run different kernels, each has evidence, every reading that is evidence has a
 *   ratio of at least 1, and at one row count or more every one is at least margin × the two gaps;
 *   among several, the one whose smallest ratio is largest.
 * - At the other row counts the kernel the derived row runs must not be slower than the default's
 *   choice (a gain of at least 1, no margin) in every reading that is evidence, and every record
 *   whose M = 1 reading was evidence must have one there: the time is in the same record.
 * - Among the candidates that qualify, the one whose smallest gain at M = 1 is largest. Where the
 *   table holds another choice than the default's for the shape, that choice stays unless one of
 *   those candidates — or the default's choice itself — beats it by the bar of M = 1 in the same
 *   group; a table's choice no steady record timed stays as it is.
 *
 * SPEC: docs/spec/geometry-profiles.md#deriving-the-gemv-choices
 */

import type { LinearGemvParallelRows } from "../kernels/linear-gemv.ts";
import type { GemvCandidateClass } from "./gemv-candidates.ts";

/**
 * One steady record's reading of a kernel against another kernel of the same group: the other's
 * time ÷ this one's (`gain`), both gaps, the other's re-timing, and the break-even margin
 * `gain ÷ (gap × referenceGap)` — the largest margin at which the gain clears its bar.
 */
export type GemvPairReading = {
  /** The record (its path, as given). */
  readonly record: string;
  readonly caseId: string;
  readonly m: number;
  readonly gain: number;
  /** This kernel's gap (its second-smallest round ÷ its smallest). */
  readonly gap: number;
  /** The other kernel's gap. */
  readonly referenceGap: number;
  /**
   * The other kernel's re-timed pass ÷ its rounds' minimum; `unread` when that pass read no time
   * (clamped or 0); absent when it was not re-timed (a record written before the re-timing).
   */
  readonly retime?: number | "unread";
  readonly breakEven: number;
};

/**
 * One row count besides M = 1 of a parallel row: the kernel the row runs there, its readings against
 * the default's choice there, a failure or failed check of it in a current group, and — where it is
 * not the reference row's kernel — its readings against that kernel (the row block's ratio).
 */
export type GemvRowCountEvidence = {
  readonly m: number;
  readonly kernel: string;
  /** Against the default's choice at `m`. */
  readonly readings: readonly GemvPairReading[];
  readonly ruledOut?: string;
  /** Against the reference row's kernel at `m`, where the two differ. */
  readonly block?: { readonly reference: string; readonly readings: readonly GemvPairReading[] };
};

/** A row block a row can carry (absent: none) and its row's kernels at the other row counts. */
export type GemvRowBlockOption = {
  readonly rowBlock?: LinearGemvParallelRows;
  readonly at: readonly GemvRowCountEvidence[];
};

/** A candidate a steady record timed at M = 1. */
export type GemvCandidateEvidence = {
  readonly candidate: string;
  /** Why it is out whatever its time: it failed, or its check failed, in a current group. */
  readonly ruledOut?: string;
  /** Against the default's choice at M = 1. */
  readonly m1: readonly GemvPairReading[];
  /** Against the table's choice at M = 1, where that is another kernel than the default's. */
  readonly vsTable: readonly GemvPairReading[];
  /** The parallel row it would give, under each row block its lanes can carry (none first); empty for the identical class. */
  readonly rows: readonly GemvRowBlockOption[];
};

/** A kept row's row block: the options against that row (its own row block is the reference). */
export type GemvKeptRow = {
  readonly choice: string;
  readonly rowBlock?: LinearGemvParallelRows;
  /** Every other row block its lanes can carry (or none), against the row. */
  readonly options: readonly GemvRowBlockOption[];
};

/** What the rule of one steady record reads for one case and class. */
export type GemvOneRecordEvidence = {
  readonly caseId: string;
  readonly class: GemvCandidateClass;
  /** The default profile's choice at M = 1 (by candidate name) and its row's row block. */
  readonly defaultChoice: string;
  readonly defaultRowBlock?: LinearGemvParallelRows;
  /** The table's choice at M = 1 and its row's row block (the default's where the table holds it). */
  readonly tableChoice: string;
  readonly tableRowBlock?: LinearGemvParallelRows;
  /** Why the table's choice is not held: it failed or its check failed in a current group. */
  readonly tableRuledOut?: string;
  /** Whether a steady record timed the table's choice at M = 1 (true where it is the default's). */
  readonly tableTimed: boolean;
  /** Every candidate a steady record timed at M = 1, the default's choice not among them. */
  readonly candidates: readonly GemvCandidateEvidence[];
  /** The default's choice against the table's at M = 1, where they differ. */
  readonly defaultVsTable: readonly GemvPairReading[];
  /** The row blocks a kept row can move to: the table's row and, where it differs, the default's. */
  readonly keptRows: readonly GemvKeptRow[];
};

/** The outcome of the rule of one steady record for one case and class. */
export type GemvOneRecordOutcome = {
  /** The derived choice at M = 1 (a candidate, the table's choice or the default's). */
  readonly choice: string;
  /** The derived row's row block (absent: none). */
  readonly rowBlock?: LinearGemvParallelRows;
  /** `kept`: the table's choice and its row block stay; `adopted`: anything else. */
  readonly outcome: "adopted" | "kept";
  /** Why: the first way the nearest candidate fails, or why the table's choice stays. */
  readonly reason: string;
  /** The adopted candidate, or the nearest one when kept (absent when none was timed). */
  readonly candidate?: string;
};

const ratio = (value: number): string => `×${value.toFixed(3)}`;

/** Whether a reading is evidence at `margin`: its reference's re-timing departs within the bar. */
export const gemvReadingIsEvidence = (reading: GemvPairReading, margin: number): boolean => {
  if (reading.retime === undefined) return true;
  if (reading.retime === "unread") return false;
  const departure = Math.max(reading.retime, 1 / reading.retime);
  return departure <= margin * reading.gap * reading.referenceGap;
};

/** The readings of `readings` that are evidence at `margin`. */
const evidenceOf = (readings: readonly GemvPairReading[], margin: number): GemvPairReading[] =>
  readings.filter((reading) => gemvReadingIsEvidence(reading, margin));

/** The first way the readings of one comparison fail the bar of M = 1 at `margin` (undefined: none). */
const barFailure = (
  readings: readonly GemvPairReading[],
  margin: number,
  what: string,
): string | undefined => {
  const evidence = evidenceOf(readings, margin);
  if (evidence.length === 0) {
    return readings.length === 0
      ? `no steady record timed it against ${what} at M=1`
      : `no reading against ${what} at M=1 is evidence: the reference's re-timing departs beyond the bar or read no time`;
  }
  const short = evidence.find((reading) => reading.breakEven < margin);
  return short === undefined
    ? undefined
    : `${ratio(short.gain)} against ${what} in ${short.record}, under its bar ${ratio(margin * short.gap * short.referenceGap)} (break-even ${ratio(short.breakEven)})`;
};

/** The smallest gain of the readings that are evidence (0 without any). */
const smallestGain = (readings: readonly GemvPairReading[], margin: number): number => {
  const evidence = evidenceOf(readings, margin);
  return evidence.length === 0 ? 0 : Math.min(...evidence.map(({ gain }) => gain));
};

/**
 * Why a row block does not replace its reference row's (undefined: it does), from the row counts
 * where their kernels differ.
 */
export const gemvRowBlockRefusal = (
  option: GemvRowBlockOption,
  margin: number,
): string | undefined => {
  const differing = option.at.filter((at) => at.block !== undefined);
  if (differing.length === 0) return "runs the reference's kernel at every other row count";
  for (const at of differing) {
    if (at.ruledOut !== undefined) return at.ruledOut;
    const evidence = evidenceOf(at.block?.readings ?? [], margin);
    if (evidence.length === 0) return `no steady record is evidence at M=${at.m}`;
    const slower = evidence.find(({ gain }) => gain < 1);
    if (slower !== undefined) {
      return `slower than the reference row's kernel in ${slower.record} at M=${at.m} (${ratio(slower.gain)})`;
    }
  }
  const clears = differing.some((at) =>
    evidenceOf(at.block?.readings ?? [], margin).every(({ breakEven }) => breakEven >= margin),
  );
  return clears ? undefined : `at no row count above its bar of ${ratio(margin)} × the two gaps`;
};

/** The smallest ratio of a row block over the row counts where it differs (0 without any). */
const rowBlockScore = (option: GemvRowBlockOption, margin: number): number => {
  const gains = option.at.flatMap((at) =>
    evidenceOf(at.block?.readings ?? [], margin).map(({ gain }) => gain),
  );
  return gains.length === 0 ? 0 : Math.min(...gains);
};

/** Of the options that replace their reference row's, the one whose smallest ratio is largest. */
const pickOption = (
  options: readonly GemvRowBlockOption[],
  margin: number,
): GemvRowBlockOption | undefined =>
  options
    .filter((option) => gemvRowBlockRefusal(option, margin) === undefined)
    .reduce<GemvRowBlockOption | undefined>(
      (best, option) =>
        best === undefined || rowBlockScore(option, margin) > rowBlockScore(best, margin)
          ? option
          : best,
      undefined,
    );

/**
 * Why the row a candidate gives with `option` fails at the other row counts (undefined: it holds):
 * slower than the default's choice in a reading that is evidence, or no time there in a record whose
 * M = 1 reading was evidence.
 */
const rowFailure = (
  option: GemvRowBlockOption,
  m1Records: ReadonlySet<string>,
  margin: number,
): string | undefined => {
  for (const at of option.at) {
    if (at.ruledOut !== undefined) return at.ruledOut;
    const evidence = evidenceOf(at.readings, margin);
    const slower = evidence.find(({ gain }) => gain < 1);
    if (slower !== undefined) {
      return `${at.kernel} is slower than the default's choice in ${slower.record} at M=${at.m} (${ratio(slower.gain)})`;
    }
    const timed = new Set(evidence.map(({ record }) => record));
    const missing = [...m1Records].find((record) => !timed.has(record));
    if (missing !== undefined) {
      return `${at.kernel} has no time at M=${at.m} in ${missing}, which timed it at M=1 (a parallel row serves M = 1 to 8)`;
    }
  }
  return undefined;
};

/** A candidate as the rule reads it at a margin. */
type Judged = {
  readonly evidence: GemvCandidateEvidence;
  readonly failure?: string;
  readonly rowBlock?: LinearGemvParallelRows;
  readonly smallest: number;
};

const judge = (candidate: GemvCandidateEvidence, margin: number): Judged => {
  const smallest = smallestGain(candidate.m1, margin);
  if (candidate.ruledOut !== undefined) {
    return { evidence: candidate, failure: candidate.ruledOut, smallest };
  }
  const m1 = barFailure(candidate.m1, margin, "the default's choice");
  if (m1 !== undefined) return { evidence: candidate, failure: m1, smallest };
  if (candidate.rows.length === 0) return { evidence: candidate, smallest };
  const [bare, ...blocked] = candidate.rows;
  if (bare === undefined) throw new Error(`${candidate.candidate}: no row without a row block`);
  const option = pickOption(blocked, margin) ?? bare;
  const m1Records = new Set(evidenceOf(candidate.m1, margin).map(({ record }) => record));
  const failure = rowFailure(option, m1Records, margin);
  return {
    evidence: candidate,
    ...(failure === undefined ? {} : { failure }),
    ...(option.rowBlock === undefined ? {} : { rowBlock: option.rowBlock }),
    smallest,
  };
};

/** The judged candidate whose smallest gain is largest (the first of a tie). */
const largest = (judged: readonly Judged[]): Judged | undefined =>
  judged.reduce<Judged | undefined>(
    (best, entry) => (best === undefined || entry.smallest > best.smallest ? entry : best),
    undefined,
  );

/** The row block a kept row ends with: its own, or the option that replaces it. */
const keptRowBlock = (
  evidence: GemvOneRecordEvidence,
  choice: string,
  margin: number,
): LinearGemvParallelRows | undefined => {
  const kept = evidence.keptRows.find((row) => row.choice === choice);
  if (kept === undefined) {
    return choice === evidence.tableChoice ? evidence.tableRowBlock : evidence.defaultRowBlock;
  }
  const other = pickOption(kept.options, margin);
  return other === undefined ? kept.rowBlock : other.rowBlock;
};

/**
 * The rule of one steady record for one case and class at `margin` (the module doc). Pure: the
 * derivation and a re-decision of its output under another margin both call it.
 */
export const decideOneRecord = (
  evidence: GemvOneRecordEvidence,
  margin: number,
): GemvOneRecordOutcome => {
  const judged = evidence.candidates.map((candidate) => judge(candidate, margin));
  const qualifying = judged.filter(({ failure }) => failure === undefined);
  const nearest = largest(qualifying) ?? largest(judged);
  const near = nearest === undefined ? {} : { candidate: nearest.evidence.candidate };
  const table = evidence.tableChoice;
  const held = evidence.tableRuledOut === undefined;
  const result = (
    choice: string,
    rowBlock: LinearGemvParallelRows | undefined,
    reason: string,
    candidate?: string,
  ): GemvOneRecordOutcome => ({
    choice,
    ...(rowBlock === undefined ? {} : { rowBlock }),
    outcome: choice === table && rowBlock === evidence.tableRowBlock ? "kept" : "adopted",
    reason,
    ...(candidate === undefined ? near : { candidate }),
  });
  // The table holds the default's choice, or a choice that is no longer held: the default's choice
  // is the one to beat
  if (table === evidence.defaultChoice || !held) {
    const best = largest(qualifying);
    if (best !== undefined) {
      return result(
        best.evidence.candidate,
        best.rowBlock,
        `qualifies by ${ratio(best.smallest)} at M=1`,
        best.evidence.candidate,
      );
    }
    const why =
      nearest === undefined
        ? "no candidate was timed at M=1 by a steady record"
        : `no candidate qualifies (nearest ${nearest.evidence.candidate}: ${nearest.failure ?? "?"})`;
    const prefix = held
      ? ""
      : `the table's choice ${table} is not held (${evidence.tableRuledOut ?? "?"}); `;
    const choice = evidence.defaultChoice;
    return result(choice, keptRowBlock(evidence, choice, margin), `${prefix}${why}`);
  }
  // The table holds another choice: it stays unless beaten in the same group
  if (!evidence.tableTimed) {
    return result(
      table,
      evidence.tableRowBlock,
      `no steady record timed the table's choice ${table}, so none moves it`,
    );
  }
  const challengers = [
    ...qualifying.map((entry) => ({
      choice: entry.evidence.candidate,
      rowBlock: entry.rowBlock,
      smallest: entry.smallest,
      vsTable: entry.evidence.vsTable,
    })),
    {
      choice: evidence.defaultChoice,
      rowBlock: keptRowBlock(evidence, evidence.defaultChoice, margin),
      smallest: 1,
      vsTable: evidence.defaultVsTable,
    },
  ].filter(({ choice }) => choice !== table);
  const beating = challengers.filter(
    ({ vsTable }) => barFailure(vsTable, margin, "the table's choice") === undefined,
  );
  const best = beating.reduce<(typeof beating)[number] | undefined>(
    (top, entry) => (top === undefined || entry.smallest > top.smallest ? entry : top),
    undefined,
  );
  if (best !== undefined) {
    return result(
      best.choice,
      best.rowBlock,
      `beats the table's choice ${table} by its bar at M=1`,
      best.choice === evidence.defaultChoice ? undefined : best.choice,
    );
  }
  const closest = challengers[0];
  const why =
    closest === undefined
      ? "no challenger"
      : `nearest ${closest.choice}: ${barFailure(closest.vsTable, margin, "the table's choice") ?? "?"}`;
  return result(
    table,
    keptRowBlock(evidence, table, margin),
    `no challenger beats the table's choice ${table} (${why})`,
  );
};
