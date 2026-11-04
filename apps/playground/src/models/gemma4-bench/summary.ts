/**
 * The bench's summary: per arm and prompt, the spread (min / median / max) of every figure over the
 * warm chats of all rounds. Both the median and the min are kept — the machine shows transient
 * 5–45× slow kernels for seconds at a time, which move a median less than a mean and a min not at
 * all. Pure, so `vp test` covers it.
 *
 * For a split arm (one timed pass per dispatch) the summary also breaks the decode GPU time down per
 * op family and pipeline key ({@link SplitSummary}).
 */

import type { Spread } from "@/lib/record";

import { familyOfKey, OP_FAMILY_LABELS, type OpFamily } from "./family";
import { type ChatMeasurement, type KeyPerToken, median, round4 } from "./measure";
import { armLabel, type BenchArm, type BenchPromptId, benchPrompt } from "./plan";

/** What the summary reads of one chat. */
export type SummaryInput = {
  readonly arm: number;
  readonly prompt: BenchPromptId;
  /** Chat 0 of its prompt in its visit (left out). */
  readonly cold: boolean;
  /** sha256 of the generated token ids (equal across runs of a greedy arm unless the output moved). */
  readonly tokenIdsSha256: string;
  readonly measurement: ChatMeasurement;
};

/** The figures summarized, in display order. */
export const SUMMARY_METRICS = [
  "decodeMsPerToken",
  "decodeTokensPerSecond",
  "ttftMs",
  "prefillTokensPerSecond",
  "gpuBusyMsPerToken",
  "gpuBusyMsPerTokenMedian",
  "wallMinusGpuMsPerToken",
  "dispatchesPerToken",
  "passesPerToken",
  "submitsPerToken",
  "decodeTokens",
] as const;

export type SummaryMetric = (typeof SUMMARY_METRICS)[number];

/** Where each figure comes from in a chat's measurement. */
const READ_METRIC: Readonly<Record<SummaryMetric, (m: ChatMeasurement) => number | undefined>> = {
  decodeMsPerToken: (m: ChatMeasurement) => m.decodeMsPerToken,
  decodeTokensPerSecond: (m: ChatMeasurement) => m.decodeTokensPerSecond,
  ttftMs: (m: ChatMeasurement) => m.ttftMs,
  prefillTokensPerSecond: (m: ChatMeasurement) => m.prefillTokensPerSecond,
  gpuBusyMsPerToken: (m: ChatMeasurement) => m.tap?.perToken?.gpuBusyMs,
  gpuBusyMsPerTokenMedian: (m: ChatMeasurement) => m.tap?.perToken?.gpuBusyMsMedian,
  wallMinusGpuMsPerToken: (m: ChatMeasurement) => m.tap?.perToken?.wallMinusGpuMs,
  dispatchesPerToken: (m: ChatMeasurement) => m.tap?.perToken?.dispatches,
  passesPerToken: (m: ChatMeasurement) => m.tap?.perToken?.passes,
  submitsPerToken: (m: ChatMeasurement) => m.tap?.perToken?.submits,
  decodeTokens: (m: ChatMeasurement) => m.decodeTokens,
};

export type ArmPromptSummary = {
  readonly arm: number;
  readonly prompt: BenchPromptId;
  readonly warmChats: number;
  /** A figure no warm chat measured is absent. */
  readonly spreads: Partial<Record<SummaryMetric, Spread>>;
  /** Distinct generated-token sequences among the warm chats (1 = every round produced the same). */
  readonly distinctOutputs: number;
  /** Median decode ms per token ÷ arm 1's on the same prompt (absent for arm 1 and when either is missing). */
  readonly decodeVsFirstArm?: number;
  /** The decode window per op family and pipeline key (present when a warm chat ran split). */
  readonly split?: SplitSummary;
};

/** One row of the split breakdown: medians over the warm chats, per decode token. */
export type SplitRow = {
  readonly dispatches: number;
  /** Absent when no chat timed every pass of the row. */
  readonly gpuMs?: number;
};

export type SplitKeyRow = SplitRow & { readonly key: string; readonly family: OpFamily };
export type SplitFamilyRow = SplitRow & { readonly family: OpFamily };

/**
 * The split breakdown of one arm × prompt. Every row is the median over the warm chats (a chat
 * without the key counts 0; a chat that left a pass of the row untimed is left out of its GPU ms).
 * The extra pass boundaries inflate every time in it: read the shares.
 */
export type SplitSummary = {
  /** Warm chats with a per-key breakdown. */
  readonly chats: number;
  /** Σ GPU ms per token over every key (median over the chats). */
  readonly totalGpuMs?: number;
  /** GPU ms descending. */
  readonly families: readonly SplitFamilyRow[];
  /** GPU ms descending. */
  readonly keys: readonly SplitKeyRow[];
};

/** min / median / max of the values (`undefined` for none); the median at four decimals. */
export const spreadOf = (values: readonly number[]): Spread | undefined => {
  const middle = median(values);
  if (middle === undefined) return undefined;
  return {
    count: values.length,
    min: Math.min(...values),
    median: round4(middle),
    max: Math.max(...values),
  };
};

const FAMILY_ORDER = Object.keys(OP_FAMILY_LABELS);

/** GPU ms descending, an untimed row last. */
const byGpuMsDescending = (left: SplitRow, right: SplitRow): number =>
  (right.gpuMs ?? -1) - (left.gpuMs ?? -1);

/** Median of each chat's figure (`undefined` = untimed, left out); `undefined` when none is timed. */
const medianOfTimed = (values: readonly (number | undefined)[]): number | undefined => {
  const timed = values.filter((value) => value !== undefined);
  const middle = median(timed);
  return middle === undefined ? undefined : round4(middle);
};

/** Σ over rows: dispatches, and GPU ms when every row is timed. */
const sumRows = (rows: readonly KeyPerToken[]): SplitRow => {
  let dispatches = 0;
  let gpuMs: number | undefined = 0;
  for (const row of rows) {
    dispatches += row.dispatches;
    gpuMs = gpuMs === undefined || row.gpuMs === undefined ? undefined : gpuMs + row.gpuMs;
  }
  return { dispatches, ...(gpuMs === undefined ? {} : { gpuMs }) };
};

/** The medians over chats of rows grouped by `groupOf` (a chat without the group counts 0). */
const groupMedians = <Group extends string>(
  chats: readonly (readonly KeyPerToken[])[],
  groupOf: (key: string) => Group,
): ReadonlyMap<Group, SplitRow> => {
  const perChat = chats.map((rows) => {
    const groups = new Map<Group, KeyPerToken[]>();
    for (const row of rows) {
      const group = groupOf(row.key);
      groups.set(group, [...(groups.get(group) ?? []), row]);
    }
    return new Map([...groups].map(([group, grouped]) => [group, sumRows(grouped)]));
  });
  const names = new Set(perChat.flatMap((groups) => [...groups.keys()]));
  const medians = new Map<Group, SplitRow>();
  for (const name of names) {
    const rows = perChat.map((groups) => groups.get(name) ?? { dispatches: 0, gpuMs: 0 });
    const gpuMs = medianOfTimed(rows.map((row) => row.gpuMs));
    medians.set(name, {
      dispatches: round4(median(rows.map((row) => row.dispatches)) ?? 0),
      ...(gpuMs === undefined ? {} : { gpuMs }),
    });
  }
  return medians;
};

/** The split breakdown over the per-key figures of the warm chats (`undefined` for none). */
export const summarizeSplit = (
  chats: readonly (readonly KeyPerToken[])[],
): SplitSummary | undefined => {
  if (chats.length === 0) return undefined;
  const keys = [...groupMedians(chats, (key) => key)]
    .map(([key, row]): SplitKeyRow => ({ key, family: familyOfKey(key), ...row }))
    .toSorted((left, right) => byGpuMsDescending(left, right) || left.key.localeCompare(right.key));
  const families = [...groupMedians(chats, familyOfKey)]
    .map(([family, row]): SplitFamilyRow => ({ family, ...row }))
    .toSorted(
      (left, right) =>
        byGpuMsDescending(left, right) ||
        FAMILY_ORDER.indexOf(left.family) - FAMILY_ORDER.indexOf(right.family),
    );
  const totalGpuMs = medianOfTimed(chats.map((rows) => sumRows(rows).gpuMs));
  return {
    chats: chats.length,
    ...(totalGpuMs === undefined ? {} : { totalGpuMs }),
    families,
    keys,
  };
};

/**
 * One entry per arm × prompt, arms in order and prompts in `prompts` order (pairs without a warm
 * chat yet are left out, so a running bench summarizes what it has).
 */
export const summarizeRuns = (
  runs: readonly SummaryInput[],
  armCount: number,
  prompts: readonly BenchPromptId[],
): readonly ArmPromptSummary[] => {
  const entries: Omit<ArmPromptSummary, "decodeVsFirstArm">[] = [];
  for (let arm = 0; arm < armCount; arm += 1) {
    for (const prompt of prompts) {
      const warm = runs.filter((run) => run.arm === arm && run.prompt === prompt && !run.cold);
      if (warm.length === 0) continue;
      const spreads: Partial<Record<SummaryMetric, Spread>> = {};
      for (const metric of SUMMARY_METRICS) {
        const values = warm
          .map(({ measurement }) => READ_METRIC[metric](measurement))
          .filter((value) => value !== undefined);
        const spread = spreadOf(values);
        if (spread !== undefined) spreads[metric] = spread;
      }
      const split = summarizeSplit(
        warm.flatMap(({ measurement }) => {
          const perKey = measurement.tap?.perKey;
          return perKey === undefined ? [] : [perKey];
        }),
      );
      entries.push({
        arm,
        prompt,
        warmChats: warm.length,
        spreads,
        distinctOutputs: new Set(warm.map(({ tokenIdsSha256 }) => tokenIdsSha256)).size,
        ...(split === undefined ? {} : { split }),
      });
    }
  }
  return entries.map((entry) => {
    const first = entries.find(({ arm, prompt }) => arm === 0 && prompt === entry.prompt);
    const own = entry.spreads.decodeMsPerToken?.median;
    const base = first?.spreads.decodeMsPerToken?.median;
    return entry.arm === 0 || own === undefined || base === undefined || !(base > 0)
      ? entry
      : { ...entry, decodeVsFirstArm: round4(own / base) };
  });
};

const msText = (value: number | undefined): string =>
  value === undefined ? "—" : value.toFixed(3);

/** `value` as a percentage of `total` at 0.1 % ("—" when either is missing or the total is not positive). */
export const shareText = (value: number | undefined, total: number | undefined): string =>
  value === undefined || total === undefined || !(total > 0)
    ? "—"
    : `${((value / total) * 100).toFixed(1)}%`;

/**
 * The split breakdowns of the summary as plain text (the page's `#split-summary`): per arm × prompt
 * a header, the families, then the keys — GPU ms per token, share of the total, dispatches per token.
 */
export const splitSummaryText = (
  summary: readonly ArmPromptSummary[],
  arms: readonly Pick<BenchArm, "mount" | "model" | "quant" | "tap" | "split">[],
): string => {
  const lines: string[] = [];
  for (const entry of summary) {
    const { split } = entry;
    if (split === undefined) continue;
    const total = split.totalGpuMs;
    const row = (gpuMs: number | undefined, dispatches: number, name: string): string =>
      `  ${msText(gpuMs).padStart(8)} ${shareText(gpuMs, total).padStart(6)} ${dispatches.toFixed(1).padStart(8)}  ${name}`;
    if (lines.length > 0) lines.push("");
    lines.push(
      `arm ${entry.arm + 1} ${armLabel(arms[entry.arm])} · ${benchPrompt(entry.prompt).label} · ${split.chats} warm chats · GPU ${msText(total)} ms/token (split)`,
      `  ${"ms/tok".padStart(8)} ${"share".padStart(6)} ${"disp/tok".padStart(8)}  family / key`,
      ...split.families.map((family) =>
        row(family.gpuMs, family.dispatches, OP_FAMILY_LABELS[family.family]),
      ),
      ...split.keys.map((key) => row(key.gpuMs, key.dispatches, `${key.family}  ${key.key}`)),
    );
  }
  return lines.join("\n");
};
