import { describe, expect, it } from "vite-plus/test";

import type { ChatMeasurement } from "../src/models/gemma4-bench/measure.ts";
import type { BenchPromptId } from "../src/models/gemma4-bench/plan.ts";
import { spreadOf, type SummaryInput, summarizeRuns } from "../src/models/gemma4-bench/summary.ts";

const measurement = (decodeMsPerToken: number, gpuBusyMs?: number): ChatMeasurement => ({
  timing: { totalSeconds: 1, decodeTokens: 127 },
  decodeTokens: 127,
  decodeMsPerToken,
  decodeTokensPerSecond: 1000 / decodeMsPerToken,
  ttftMs: 50,
  prefillTokensPerSecond: 400,
  ...(gpuBusyMs === undefined
    ? {}
    : {
        tap: {
          prefill: { passes: 3, dispatches: 900, submits: 2, busyMs: 20, untimedPasses: 0 },
          decode: { passes: 254, dispatches: 127_000, submits: 127, untimedPasses: 0 },
          perToken: {
            gpuBusyMs,
            gpuBusyMsMedian: gpuBusyMs,
            wallMinusGpuMs: decodeMsPerToken - gpuBusyMs,
            dispatches: 1000,
            passes: 2,
            submits: 1,
          },
        },
      }),
});

const run = (
  arm: number,
  decodeMsPerToken: number,
  patch: Partial<Omit<SummaryInput, "measurement">> & { gpu?: number } = {},
): SummaryInput => ({
  arm,
  prompt: patch.prompt ?? "few-ids",
  cold: patch.cold ?? false,
  tokenIdsSha256: patch.tokenIdsSha256 ?? "same",
  measurement: measurement(decodeMsPerToken, patch.gpu),
});

const PROMPTS: readonly BenchPromptId[] = ["few-ids", "many-ids"];

describe("the summary", () => {
  it("leaves the cold chats out", () => {
    const [entry] = summarizeRuns(
      [run(0, 500, { cold: true }), run(0, 10), run(0, 12)],
      1,
      PROMPTS,
    );
    expect(entry.warmChats).toBe(2);
    expect(entry.spreads.decodeMsPerToken).toEqual({ count: 2, min: 10, median: 11, max: 12 });
  });

  it("keeps the min next to the median, so a transient slow chat shows in neither", () => {
    const [entry] = summarizeRuns([run(0, 10), run(0, 11), run(0, 450)], 1, PROMPTS);
    expect(entry.spreads.decodeMsPerToken?.median).toBe(11);
    expect(entry.spreads.decodeMsPerToken?.min).toBe(10);
  });

  it("summarizes each arm and prompt apart, arms in order then prompts in order", () => {
    const entries = summarizeRuns(
      [run(1, 8, { prompt: "many-ids" }), run(0, 10), run(1, 7), run(0, 9, { prompt: "many-ids" })],
      2,
      PROMPTS,
    );
    expect(entries.map(({ arm, prompt }) => `${arm}:${prompt}`)).toEqual([
      "0:few-ids",
      "0:many-ids",
      "1:few-ids",
      "1:many-ids",
    ]);
  });

  it("leaves out a pair with no warm chat yet (a running bench summarizes what it has)", () => {
    expect(summarizeRuns([run(0, 10), run(1, 9, { cold: true })], 2, PROMPTS)).toHaveLength(1);
  });

  it("compares each arm's median decode time with arm 1's on the same prompt", () => {
    const entries = summarizeRuns([run(0, 10), run(0, 10), run(1, 8), run(1, 8)], 2, PROMPTS);
    expect(entries[0].decodeVsFirstArm).toBeUndefined();
    expect(entries[1].decodeVsFirstArm).toBe(0.8);
  });

  it("summarizes the tap's figures only from the chats that have them (a tap-off arm has none)", () => {
    const [tapped, control] = summarizeRuns(
      [run(0, 10, { gpu: 6 }), run(0, 12, { gpu: 7 }), run(1, 9), run(1, 9)],
      2,
      PROMPTS,
    );
    expect(tapped.spreads.gpuBusyMsPerToken).toEqual({ count: 2, min: 6, median: 6.5, max: 7 });
    expect(tapped.spreads.wallMinusGpuMsPerToken?.median).toBe(4.5);
    expect(control.spreads.gpuBusyMsPerToken).toBeUndefined();
    expect(control.spreads.decodeMsPerToken?.median).toBe(9);
  });

  it("counts the distinct outputs among the warm chats (1 = every round generated the same)", () => {
    const [same, moved] = summarizeRuns(
      [
        run(0, 10, { tokenIdsSha256: "a" }),
        run(0, 10, { tokenIdsSha256: "a" }),
        run(0, 10, { tokenIdsSha256: "z", cold: true }),
        run(1, 10, { tokenIdsSha256: "a" }),
        run(1, 10, { tokenIdsSha256: "b" }),
      ],
      2,
      PROMPTS,
    );
    expect(same.distinctOutputs).toBe(1);
    expect(moved.distinctOutputs).toBe(2);
  });
});

describe("a spread", () => {
  it("takes the mean of the middle two for an even count", () => {
    expect(spreadOf([4, 1, 3, 2])).toEqual({ count: 4, min: 1, median: 2.5, max: 4 });
  });

  it("is undefined for no values", () => {
    expect(spreadOf([])).toBeUndefined();
  });
});
