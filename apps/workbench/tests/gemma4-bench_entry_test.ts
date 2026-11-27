import { describe, expect, it } from "vite-plus/test";
import { ParameterError, parseParameters } from "@karume/sweep-app-support/parameters";

import gemmaBench from "../src/entries/local/gemma4-bench.ts";
import type { BenchSettings } from "../src/gemma4-bench/plan.ts";
import { armGemvChoices, gemvChoicesLine } from "../src/gemma4-bench/profile.ts";
import { benchProgressLine } from "../src/gemma4-bench/progress.ts";
import { decodeBenchResult } from "../src/gemma4-bench/result.ts";
import { type RunArm, settleDecodeBench } from "../src/gemma4-bench/run.ts";
import {
  type BenchFakeOptions,
  fakeBenchDependencies,
  SPLIT_LABEL,
} from "./helpers/gemma4-bench-fakes.ts";
import { REGULAR } from "./helpers/gemma4-bench-manifests.ts";
import { hex, memoryDistributions } from "./helpers/session-ab-manifests.ts";

const SIGNAL = new AbortController().signal;

/** The typed parameters of `given` (strings, as a URL or a command line gives them). */
const parameters = (given: Readonly<Record<string, string>> = {}) =>
  parseParameters(gemmaBench.parameters, given);

const FACTS = {
  gemma4: { directory: "models/karume-gemma4", manifestSha256: hex("1"), read: "directory" },
  "gemma4-qat": {
    directory: "models/karume-gemma4-qat",
    manifestSha256: hex("2"),
    read: "directory",
  },
} as const;

const REGULAR_ARM: RunArm = {
  mount: "gemma4",
  model: "e2b",
  quant: "bf16",
  quantLabel: "bf16 label",
  isDefaultQuant: true,
  tap: true,
  sampler: { temperature: 0 },
  distribution: { ref: { repo: "test/gemma4" }, facts: FACTS.gemma4 },
};

const QAT_ARM: RunArm = {
  mount: "gemma4-qat",
  model: "e2b",
  quant: "i4",
  quantLabel: "i4 label",
  isDefaultQuant: true,
  tap: false,
  sampler: { temperature: 0 },
  distribution: { ref: { repo: "test/gemma4-qat" }, facts: FACTS["gemma4-qat"] },
};

const SETTINGS: BenchSettings = {
  chats: 2,
  newTokens: 4,
  rounds: 1,
  sampler: "greedy",
  prompts: ["few-ids"],
};

/** One fake bench run of `arms` and its result. */
const benchOf = async (
  arms: readonly RunArm[],
  options: BenchFakeOptions & { readonly signal?: AbortSignal } = {},
) => {
  const settings = SETTINGS;
  const outcome = await settleDecodeBench({
    arms,
    settings,
    signal: options.signal ?? SIGNAL,
    dependencies: fakeBenchDependencies(options).dependencies,
    onProgress: (progress) =>
      options.events?.push(benchProgressLine(progress, arms, settings.chats)),
  });
  const result = () => decodeBenchResult({ timestampQuery: true, settings, arms, outcome });
  return { outcome, result };
};

describe("the gemma4-bench entry", () => {
  describe("its parameters", () => {
    it("default to the default arms, 3 chats of 128 new tokens over 4 rounds, greedy, both prompts, subgroups when the device has them, cached", () => {
      expect(parameters()).toEqual({
        arms: "",
        chats: 3,
        newTokens: 128,
        rounds: 4,
        sampler: "greedy",
        prompts: ["few-ids", "many-ids"],
        subgroups: true,
        source: "cached",
      });
    });

    it("need a warm chat, a decode token and a round, and take at most 1000 chats", () => {
      expect(parameters({ chats: "2", newTokens: "2", rounds: "1" })).toMatchObject({
        chats: 2,
        newTokens: 2,
        rounds: 1,
      });
      for (const [name, value] of [
        ["chats", "1"],
        ["newTokens", "1"],
        ["rounds", "0"],
        ["chats", "1001"],
      ] as const) {
        expect(() => parameters({ [name]: value }), `${name}=${value}`).toThrow(
          new RegExp(`^Parameter ${name}: expected an integer from`),
        );
      }
    });
  });

  describe("its run", () => {
    const runWith = (given: Readonly<Record<string, string>>, opened: string[]) =>
      gemmaBench.run({
        parameters: parameters(given),
        // No GPU here: a run that got past its checks would fail on the missing navigator.gpu.
        gpuOptions: {},
        signal: SIGNAL,
        progress: () => undefined,
        openDistribution: memoryDistributions({ gemma4: REGULAR }, opened),
        openInput: () => Promise.reject(new Error("the entry reads no input file")),
      });

    it("refuses an arm that does not fit the form before it reads a distribution or a GPU", async () => {
      const opened: string[] = [];
      await expect(runWith({ arms: "gemma4/e2b@" }, opened)).rejects.toThrow(
        `Parameter arms: arm 1 "gemma4/e2b@": names an empty preset`,
      );
      expect(opened).toEqual([]);
    });

    it("refuses an arm its distribution does not have before it acquires a GPU", async () => {
      const opened: string[] = [];
      await expect(runWith({ arms: "gemma4/e2b@i4" }, opened)).rejects.toThrow(
        new ParameterError(
          "arms",
          `arm 1 "gemma4/e2b@i4": The gemma4 model 'e2b' has no preset 'i4'; its presets are bf16.`,
        ),
      );
      expect(opened).toEqual(["gemma4:cached"]);
    });

    it("refuses no prompt before it reads anything", async () => {
      const opened: string[] = [];
      await expect(runWith({ prompts: "" }, opened)).rejects.toThrow(ParameterError);
      expect(opened).toEqual([]);
    });
  });

  describe("its progress", () => {
    it("names the visit, its round and arm, then the build or the chat (chat 0 the cold one)", () => {
      const arms = [REGULAR_ARM, QAT_ARM];
      const visit = { visit: 2, round: 1, arm: 1 };
      expect(benchProgressLine({ kind: "loading", visit, visits: 4 }, arms, 3)).toBe(
        "visit 3 / 4 (round 2) · arm 2 QAT q4_0 e2b i4 (tap off) · building the pipeline",
      );
      expect(
        benchProgressLine({ kind: "chat", visit, visits: 4, prompt: "many-ids", chat: 0 }, arms, 3),
      ).toBe(
        "visit 3 / 4 (round 2) · arm 2 QAT q4_0 e2b i4 (tap off) · GPU explainer chat 1 / 3 (cold)",
      );
      expect(
        benchProgressLine({ kind: "chat", visit, visits: 4, prompt: "few-ids", chat: 2 }, arms, 3),
      ).toBe("visit 3 / 4 (round 2) · arm 2 QAT q4_0 e2b i4 (tap off) · Story request chat 3 / 3");
    });

    it("is reported before each build and before each chat's pause, outside the measured chat", async () => {
      const events: string[] = [];
      await benchOf([REGULAR_ARM], { events });
      expect(events).toEqual([
        "visit 1 / 1 (round 1) · arm 1 Gemma 4 e2b bf16 · building the pipeline",
        "build e2b",
        "visit 1 / 1 (round 1) · arm 1 Gemma 4 e2b bf16 · Story request chat 1 / 2 (cold)",
        "pause",
        "chat",
        "visit 1 / 1 (round 1) · arm 1 Gemma 4 e2b bf16 · Story request chat 2 / 2",
        "pause",
        "chat",
      ]);
    });
  });

  describe("its result", () => {
    it("has no verdict for a bench that finished, and one line per arm and prompt: rates, ms/token, the token interval, the ratio to arm 1, GPU time when tapped, identity", async () => {
      const { result } = await benchOf([REGULAR_ARM, QAT_ARM]);
      const finished = result();
      expect(Object.hasOwn(finished, "verdict")).toBe(false);
      // Each chat: 3 decode tokens over 3 ms, one 0.25 ms pass per decode step.
      expect(finished.summary).toEqual([
        "2 of 2 visits, 4 chats (cold ones included)",
        "arm 1 Gemma 4 e2b bf16 · Story request · decode 1000.0 tok/s (max 1000.0) · 1.000 ms/token · token interval 1.000 ms (min 1.000) · GPU 0.250 ms/token · outputs identical",
        "arm 2 QAT q4_0 e2b i4 (tap off) · Story request · decode 1000.0 tok/s (max 1000.0) · 1.000 ms/token · token interval 1.000 ms (min 1.000) · ×1.000 vs arm 1 · outputs identical",
      ]);
    });

    it("adds one line per arm with the GEMV choices it ran with in a bench with a profile switch, after the arms' lines, and states them in the data", async () => {
      const switched: RunArm = { ...REGULAR_ARM, profile: ["default-gemv"] };
      const arms = [REGULAR_ARM, switched];
      const { outcome } = await benchOf(arms);
      const ampere = { vendor: "nvidia", architecture: "ampere", description: "" };
      const gemvChoices = [armGemvChoices(ampere, []), armGemvChoices(ampere, ["default-gemv"])];
      const finished = decodeBenchResult({
        timestampQuery: true,
        settings: SETTINGS,
        arms,
        outcome,
        gemvChoices,
      });
      expect(finished.summary.slice(1, 3).map((line) => line.split(" · ")[0])).toEqual([
        "arm 1 Gemma 4 e2b bf16",
        "arm 2 Gemma 4 e2b bf16 (default GEMV)",
      ]);
      expect(finished.summary.slice(3)).toEqual([
        gemvChoicesLine("arm 1 Gemma 4 e2b bf16", gemvChoices[0]),
        gemvChoicesLine("arm 2 Gemma 4 e2b bf16 (default GEMV)", gemvChoices[1]),
      ]);
      expect(finished.data.arms.map((arm) => arm.gemvChoices)).toEqual(gemvChoices);
      // A bench without a switch says nothing of them
      const plain = decodeBenchResult({ timestampQuery: true, settings: SETTINGS, arms, outcome });
      expect(plain.summary).toHaveLength(3);
    });

    it("keeps the data body: the workload with its prompt texts, the arms with their distributions, every visit and chat, the summary", async () => {
      const { result } = await benchOf([REGULAR_ARM, QAT_ARM]);
      const { data } = result();
      expect(data.timestampQuery).toBe(true);
      expect(data.complete).toBe(true);
      expect(data.settings).toEqual({
        chats: 2,
        newTokens: 4,
        rounds: 1,
        sampler: "greedy",
        prompts: [
          {
            id: "few-ids",
            text: "Write a long story about a lighthouse keeper and the sea.",
            distinctIds: "< 32",
          },
        ],
      });
      expect(data.arms).toEqual([
        {
          index: 1,
          mount: "gemma4",
          distribution: FACTS.gemma4,
          pipeline: "gemma.Pipeline",
          model: "e2b",
          preset: { quant: "bf16", label: "bf16 label", isDefault: true },
          tap: true,
          split: false,
          hostMarks: false,
          signal: true,
          integerGemv: false,
          drafter: false,
          sessionOptions: {},
          pipelineOptions: {},
          sampler: { temperature: 0 },
        },
        expect.objectContaining({ index: 2, mount: "gemma4-qat", tap: false }),
      ]);
      expect(data.visits.map(({ visit, arm }) => [visit, arm])).toEqual([
        [0, 0],
        [1, 1],
      ]);
      expect(data.chats.map(({ arm, chat, cold }) => [arm, chat, cold])).toEqual([
        [0, 0, true],
        [0, 1, false],
        [1, 0, true],
        [1, 1, false],
      ]);
      expect(data.summary.map(({ arm, warmChats }) => [arm, warmChats])).toEqual([
        [0, 1],
        [1, 1],
      ]);
    });

    it("ends with the split breakdown of a split arm", async () => {
      const { result } = await benchOf([{ ...REGULAR_ARM, split: true }]);
      expect(result().summary).toEqual([
        "1 of 1 visits, 2 chats (cold ones included)",
        "arm 1 Gemma 4 e2b bf16 (split) · Story request · decode 1000.0 tok/s (max 1000.0) · 1.000 ms/token · token interval 1.000 ms (min 1.000) · GPU 0.250 ms/token · outputs identical",
        "arm 1 Gemma 4 e2b bf16 (split) · Story request · 1 warm chats · GPU 0.250 ms/token (split)",
        "    ms/tok  share disp/tok  family / key",
        "     0.250 100.0%      1.0  linear",
        `     0.250 100.0%      1.0  linear  ${SPLIT_LABEL}`,
      ]);
    });

    it("ends with the host's segments of an arm in the tap's host mode, and gives that arm no GPU time", async () => {
      const { result } = await benchOf([{ ...REGULAR_ARM, hostMarks: true }]);
      const { summary, data } = result();
      expect(summary.slice(0, 3)).toEqual([
        "1 of 1 visits, 2 chats (cold ones included)",
        "arm 1 Gemma 4 e2b bf16 (host) · Story request · decode 100.0 tok/s (max 100.0) · 10.000 ms/token · token interval 10.000 ms (min 10.000) · outputs identical",
        "arm 1 Gemma 4 e2b bf16 (host) · Story request · 1 warm chats · host segments, ms per decode token",
      ]);
      expect(summary).toHaveLength(8);
      expect(data.arms[0]).toMatchObject({
        tap: true,
        split: false,
        hostMarks: true,
        signal: true,
      });
      expect(data.chats[1].measurement.host?.intervals).toBe(3);
    });

    it("says where the decode steps of the warm chats drew from when the chats' stops report the candidate selection", async () => {
      const drawn = await benchOf([REGULAR_ARM], {
        candidateSelection: { count: 65, steps: 3, fallbacks: 0 },
      });
      expect(drawn.result().summary[1]).toMatch(
        / · outputs identical · 65 candidates: 3 steps, 0 whole-row$/,
      );
      const unused = await benchOf([REGULAR_ARM], {
        candidateSelection: { count: 0, steps: 0, fallbacks: 0 },
      });
      expect(unused.result().summary[1]).toMatch(/ · outputs identical · candidates not used$/);
      const without = await benchOf([REGULAR_ARM]);
      expect(without.result().summary[1]).toMatch(/ · outputs identical$/);
    });

    it("says a drafter arm's tokens per cycle and accepted drafts over its warm chats, and records the arm as such", async () => {
      const speculation = {
        cycles: 2,
        draftRuns: 2,
        drafted: 6,
        accepted: 1,
        delivered: 3,
        acceptedHistogram: [1, 1, 0, 0],
      };
      const { result } = await benchOf([REGULAR_ARM, { ...REGULAR_ARM, drafter: true }], {
        speculation,
      });
      const { summary, data } = result();
      expect(summary[1]).toMatch(/^arm 1 Gemma 4 e2b bf16 · .* · outputs identical$/);
      expect(summary[2]).toMatch(
        /^arm 2 Gemma 4 e2b bf16 \(drafter\) · .* · outputs identical · 1\.500 tokens\/cycle \(accepted 1\/6 drafts over 2 cycles\)$/,
      );
      // The tokens of one cycle arrive together: the interval between two is no decode step's time
      expect(summary[1]).toContain(" · token interval ");
      expect(summary[2]).not.toContain("token interval");
      expect(data.arms.map(({ drafter }) => drafter)).toEqual([false, true]);
      expect(data.chats.filter(({ arm }) => arm === 1).map(({ stop }) => stop.speculation)).toEqual(
        [speculation, speculation],
      );
    });

    it("records an arm sent without a signal as such", async () => {
      const { result } = await benchOf([{ ...REGULAR_ARM, noSignal: true }]);
      expect(result().data.arms[0]).toMatchObject({ signal: false, hostMarks: false });
      expect(result().summary[1]).toContain("arm 1 Gemma 4 e2b bf16 (no signal) · ");
    });

    it("keeps what finished with the verdict incomplete when a failure stopped the bench after a chat", async () => {
      const { outcome, result } = await benchOf([REGULAR_ARM, QAT_ARM], { failChatAt: 2 });
      expect(outcome.stop).toEqual({ kind: "failed", error: new Error("chat failed") });
      const stopped = result();
      expect(stopped.verdict).toBe("incomplete");
      expect(stopped.summary[0]).toBe(
        "2 of 2 visits, 2 chats (cold ones included) · stopped: chat failed",
      );
      expect(stopped.data.complete).toBe(false);
      expect(stopped.data.chats).toHaveLength(2);
    });

    it("rethrows a failure that came before any chat", async () => {
      const { outcome, result } = await benchOf([REGULAR_ARM], { failBuild: true });
      expect(outcome.chats).toEqual([]);
      expect(result).toThrow(new Error("build failed"));
    });

    it("keeps what finished without a verdict when the bench was cancelled", async () => {
      const controller = new AbortController();
      let pauses = 0;
      const { outcome, result } = await benchOf([REGULAR_ARM, QAT_ARM], {
        signal: controller.signal,
        onPause: () => {
          pauses += 1;
          // The second chat's pause: the first chat has finished.
          if (pauses === 2) controller.abort(new Error("Cancelled"));
        },
      });
      expect(outcome.stop).toEqual({ kind: "cancelled" });
      const cancelled = result();
      expect(Object.hasOwn(cancelled, "verdict")).toBe(false);
      expect(cancelled.summary).toEqual([
        "1 of 2 visits, 1 chats (cold ones included) · stopped: cancelled",
      ]);
      expect(cancelled.data.complete).toBe(false);
    });
  });
});
