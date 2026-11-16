/**
 * The `gemma4-bench` entry's result: the data body of what finished, and the summary — how far the
 * run got, one line per arm and prompt over the warm chats, then the split arms' breakdown. A bench
 * that finished has no verdict (it only measures); one that a failure stopped after a chat keeps
 * what finished with the verdict `incomplete`; a cancelled one keeps what finished (the record says
 * cancelled); one that failed before any chat rethrows. Pure, so `vp test` covers it.
 */

import type { DistributionFacts, EntryResult } from "../entry.ts";
import type { GemmaMount } from "../gemma4/settings.ts";
import { buildBenchData, type DecodeBenchData } from "./data.ts";
import { armLabel, type BenchArm, type BenchSettings, benchPrompt } from "./plan.ts";
import type { BenchOutcome, BenchStop, RunArm } from "./run.ts";
import { type ArmPromptSummary, splitSummaryText } from "./summary.ts";

const fixed = (value: number | undefined, digits: number): string =>
  value === undefined ? "—" : value.toFixed(digits);

/**
 * One arm and prompt in a line: the arm's label, the median decode rate (and the max), the median
 * decode ms per token, its ratio to arm 1's on the same prompt, the median GPU ms per token of a
 * tapped arm, and whether its warm chats generated the same tokens. Rates at 0.1, times and ratios
 * at 0.001.
 */
const armPromptLine = (
  entry: ArmPromptSummary,
  arm: Pick<BenchArm, "mount" | "model" | "quant" | "tap" | "split" | "session" | "pipeline">,
): string => {
  const { spreads } = entry;
  const rate = spreads.decodeTokensPerSecond;
  return [
    `arm ${entry.arm + 1} ${armLabel(arm)}`,
    benchPrompt(entry.prompt).label,
    `decode ${rate === undefined ? "— tok/s" : `${rate.median.toFixed(1)} tok/s (max ${rate.max.toFixed(1)})`}`,
    `${fixed(spreads.decodeMsPerToken?.median, 3)} ms/token`,
    ...(entry.decodeVsFirstArm === undefined
      ? []
      : [`×${entry.decodeVsFirstArm.toFixed(3)} vs arm 1`]),
    ...(arm.tap ? [`GPU ${fixed(spreads.gpuBusyMsPerToken?.median, 3)} ms/token`] : []),
    entry.distinctOutputs === 1 ? "outputs identical" : `outputs ${entry.distinctOutputs} distinct`,
  ].join(" · ");
};

const stopText = (stop: BenchStop): string | undefined => {
  switch (stop.kind) {
    case "completed":
      return undefined;
    case "cancelled":
      return "stopped: cancelled";
    case "failed":
      return `stopped: ${stop.error instanceof Error ? stop.error.message : String(stop.error)}`;
  }
};

export type DecodeBenchResult = EntryResult & { readonly data: DecodeBenchData };

/** The result of a bench run's outcome (see the module's comment for the verdict). */
export const decodeBenchResult = (input: {
  /** Whether the device timed passes (the adapter listed timestamp-query). */
  readonly timestampQuery: boolean;
  readonly settings: BenchSettings;
  readonly arms: readonly RunArm[];
  /** The distribution of every mount an arm reads. */
  readonly distributions: Partial<Readonly<Record<GemmaMount, DistributionFacts>>>;
  readonly outcome: BenchOutcome;
}): DecodeBenchResult => {
  const { settings, arms, outcome } = input;
  const { stop } = outcome;
  if (stop.kind === "failed" && outcome.chats.length === 0) throw stop.error;
  const data = buildBenchData({
    timestampQuery: input.timestampQuery,
    complete: stop.kind === "completed",
    settings,
    arms,
    distributions: input.distributions,
    visits: outcome.visits,
    chats: outcome.chats,
  });
  const stopped = stopText(stop);
  const split = splitSummaryText(data.summary, arms);
  return {
    ...(stop.kind === "failed" ? { verdict: "incomplete" as const } : {}),
    summary: [
      `${outcome.visits.length} of ${arms.length * settings.rounds} visits, ${outcome.chats.length} chats (cold ones included)${stopped === undefined ? "" : ` · ${stopped}`}`,
      ...data.summary.map((entry) => armPromptLine(entry, arms[entry.arm])),
      ...(split === "" ? [] : split.split("\n")),
    ],
    data,
  };
};
