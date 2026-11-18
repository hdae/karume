/**
 * The bench's summary: per arm and prompt, the spread (min / median / max) of every figure over the
 * warm chats of all rounds. Both the median and the min are kept — the machine shows transient
 * 5–45× slow kernels for seconds at a time, which move a median less than a mean and a min not at
 * all. Pure, so `vp test` covers it.
 *
 * For a split arm (one timed pass per dispatch) the summary also breaks the decode GPU time down per
 * op family and pipeline key ({@link SplitSummary}); for an arm in the tap's host mode it has the
 * host's segments of a decode token ({@link HostSummary}).
 */

import { familyOfKey, OP_FAMILY_LABELS, type OpFamily } from "./family.ts";
import {
  type ChatMeasurement,
  HOST_SEGMENTS,
  type HostFigures,
  type HostSegment,
  type KeyPerToken,
  median,
  round4,
} from "./measure.ts";
import { armLabel, type BenchPromptId, benchPrompt } from "./plan.ts";

/**
 * min / median / max of a non-empty list (the median of an even count is the mean of the middle
 * two). Copied from the Playground's record helpers (`apps/playground/src/lib/record.ts`).
 */
export type Spread = {
  readonly count: number;
  readonly min: number;
  readonly median: number;
  readonly max: number;
};

/** What the summary reads of one chat. */
export type SummaryInput = {
  readonly arm: number;
  readonly prompt: BenchPromptId;
  /** Chat 0 of its prompt in its visit (left out). */
  readonly cold: boolean;
  /** sha256 of the generated token ids (equal across runs of a greedy arm unless the output moved). */
  readonly tokenIdsSha256: string;
  readonly measurement: ChatMeasurement;
  /** The chat's stop, for the counters of the path its decode steps took (absent = none reported). */
  readonly stop?: { readonly candidateSelection?: CandidateCounts };
};

/**
 * The candidate selection's counters (`StopRecord.candidateSelection`): the leading logits a sampled
 * decode step read back in place of the whole row (0 = the chat did not use the selection), the
 * steps drawn from them, and the steps that read the whole row after all.
 */
export type CandidateCounts = {
  readonly count: number;
  readonly steps: number;
  readonly fallbacks: number;
};

/** The figures summarized, in display order. */
export const SUMMARY_METRICS = [
  "decodeMsPerToken",
  "tokenIntervalMsMedian",
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
  tokenIntervalMsMedian: (m: ChatMeasurement) => m.tokenIntervalMsMedian,
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
  /** The host's segments of a decode token (present when a warm chat ran in the tap's host mode). */
  readonly host?: HostSummary;
  /**
   * The candidate selection over the warm chats (present when a warm chat's stop reports it): the
   * largest count a chat asked for, and the steps and the whole-row fallbacks summed. A sampled arm
   * without it read the whole row at every step.
   */
  readonly candidateSelection?: CandidateCounts;
};

/**
 * The host's segments of one arm × prompt over the warm chats that have them, in ms per decode
 * token. `meanMs` spreads each chat's mean over its tokens (the segments' medians add up to about
 * the median decode ms per token); `medianMs` spreads each chat's median over its tokens (what a
 * token usually pays, without the slow ones).
 */
export type HostSummary = {
  readonly chats: number;
  readonly meanMs: Readonly<Record<HostSegment, Spread>>;
  readonly medianMs: Readonly<Record<HostSegment, Spread>>;
  /** Medians over the chats, per decode token. */
  readonly submits: number;
  readonly queueWrites: number;
  readonly maps: number;
  readonly mappedBytes: number;
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

/** The host's segments over the warm chats' figures (`undefined` for none). */
export const summarizeHost = (chats: readonly HostFigures[]): HostSummary | undefined => {
  if (chats.length === 0) return undefined;
  const spreads = (read: (figures: HostFigures) => Readonly<Record<HostSegment, number>>) => {
    const entries = HOST_SEGMENTS.map((segment): [HostSegment, Spread] => {
      const spread = spreadOf(chats.map((figures) => read(figures)[segment]));
      if (spread === undefined) throw new Error("No chat to spread");
      return [segment, spread];
    });
    // Every segment has an entry (the list is the union's).
    return Object.fromEntries(entries) as Record<HostSegment, Spread>;
  };
  const middle = (read: (figures: HostFigures) => number): number =>
    round4(median(chats.map(read)) ?? 0);
  return {
    chats: chats.length,
    meanMs: spreads((figures) => figures.meanMs),
    medianMs: spreads((figures) => figures.medianMs),
    submits: middle((figures) => figures.submits),
    queueWrites: middle((figures) => figures.queueWrites),
    maps: middle((figures) => figures.maps),
    mappedBytes: middle((figures) => figures.mappedBytes),
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
      const host = summarizeHost(
        warm.flatMap(({ measurement }) =>
          measurement.host === undefined ? [] : [measurement.host],
        ),
      );
      const selections = warm.flatMap(({ stop }) =>
        stop?.candidateSelection === undefined ? [] : [stop.candidateSelection],
      );
      entries.push({
        arm,
        prompt,
        warmChats: warm.length,
        spreads,
        distinctOutputs: new Set(warm.map(({ tokenIdsSha256 }) => tokenIdsSha256)).size,
        ...(split === undefined ? {} : { split }),
        ...(host === undefined ? {} : { host }),
        ...(selections.length === 0
          ? {}
          : {
              candidateSelection: {
                count: Math.max(...selections.map(({ count }) => count)),
                steps: selections.reduce((sum, { steps }) => sum + steps, 0),
                fallbacks: selections.reduce((sum, { fallbacks }) => sum + fallbacks, 0),
              },
            }),
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
 * The split breakdowns of the summary as plain lines (the entry's summary ends with them): per arm ×
 * prompt a header, the families, then the keys — GPU ms per token, share of the total, dispatches
 * per token — with an empty line between two breakdowns.
 */
export const splitSummaryText = (
  summary: readonly ArmPromptSummary[],
  arms: readonly Parameters<typeof armLabel>[0][],
): readonly string[] => {
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
  return lines;
};

type LabelledArm = Parameters<typeof armLabel>[0];

/**
 * The host's segments of the summary as plain lines (the entry's summary ends with them): per arm ×
 * prompt a header, the segments' names, three rows — the median and the minimum over the warm chats
 * of the chat's mean per decode token, and the median over the chats of the chat's median over its
 * tokens — and the WebGPU calls per token, with an empty line between two arms.
 */
export const hostSummaryText = (
  summary: readonly ArmPromptSummary[],
  arms: readonly LabelledArm[],
): readonly string[] => {
  const lines: string[] = [];
  for (const entry of summary) {
    const { host } = entry;
    if (host === undefined) continue;
    const row = (name: string, read: (segment: HostSegment) => number): string =>
      `  ${name.padEnd(18)}${HOST_SEGMENTS.map((segment) => msText(read(segment)).padStart(9)).join("")}${msText(
        HOST_SEGMENTS.reduce((sum, segment) => sum + read(segment), 0),
      ).padStart(9)}`;
    if (lines.length > 0) lines.push("");
    lines.push(
      `arm ${entry.arm + 1} ${armLabel(arms[entry.arm])} · ${benchPrompt(entry.prompt).label} · ${host.chats} warm chats · host segments, ms per decode token`,
      `  ${"".padEnd(18)}${[...HOST_SEGMENTS, "sum"].map((name) => name.padStart(9)).join("")}`,
      row("mean (median)", (segment) => host.meanMs[segment].median),
      row("mean (min)", (segment) => host.meanMs[segment].min),
      row("per-token median", (segment) => host.medianMs[segment].median),
      `  ${host.submits.toFixed(1)} submits · ${host.queueWrites.toFixed(1)} queue writes · ${host.maps.toFixed(1)} maps · ${Math.round(host.mappedBytes).toLocaleString("en-US")} B mapped per token`,
    );
  }
  return lines;
};
