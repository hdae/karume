/**
 * The bench's summary: per arm and prompt, the spread (min / median / max) of every figure over the
 * warm chats of all rounds. Both the median and the min are kept — the machine shows transient
 * 5–45× slow kernels for seconds at a time, which move a median less than a mean and a min not at
 * all. Pure, so `vp test` covers it.
 */

import type { Spread } from "@/lib/record";

import type { BenchPromptId } from "./plan";
import { type ChatMeasurement, median, round4 } from "./measure";

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
      entries.push({
        arm,
        prompt,
        warmChats: warm.length,
        spreads,
        distinctOutputs: new Set(warm.map(({ tokenIdsSha256 }) => tokenIdsSha256)).size,
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
