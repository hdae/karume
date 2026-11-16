import type { ChatStop, ChatStream } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import gemmaBench from "../src/entries/local/gemma4-bench.ts";
import { ParameterError, parseParameters } from "../src/entry.ts";
import type { BenchSettings } from "../src/gemma4-bench/plan.ts";
import { benchProgressLine } from "../src/gemma4-bench/progress.ts";
import { decodeBenchResult } from "../src/gemma4-bench/result.ts";
import {
  type BenchDependencies,
  type BenchPipeline,
  type BenchProbe,
  type RunArm,
  settleDecodeBench,
} from "../src/gemma4-bench/run.ts";

import { REGULAR } from "./helpers/gemma4-bench-manifests.ts";
import { hex, memoryDistributions } from "./helpers/session-ab-manifests.ts";

const SIGNAL = new AbortController().signal;

/** The typed parameters of `given` (strings, as a URL or a command line gives them). */
const parameters = (given: Readonly<Record<string, string>> = {}) =>
  parseParameters(gemmaBench.parameters, given);

const LINEAR = "linear_gemv_parallel:wi4g32:l32";

const REGULAR_ARM: RunArm = {
  mount: "gemma4",
  model: "e2b",
  quant: "bf16",
  quantLabel: "bf16 label",
  isDefaultQuant: true,
  tap: true,
  sampler: { temperature: 0 },
};

const QAT_ARM: RunArm = {
  mount: "gemma4-qat",
  model: "e2b",
  quant: "i4",
  quantLabel: "i4 label",
  isDefaultQuant: true,
  tap: false,
  sampler: { temperature: 0 },
};

const FACTS = {
  gemma4: { directory: "models/karume-gemma4", manifestSha256: hex("1"), read: "directory" },
  "gemma4-qat": {
    directory: "models/karume-gemma4-qat",
    manifestSha256: hex("2"),
    read: "directory",
  },
} as const;

const SETTINGS: BenchSettings = {
  chats: 2,
  newTokens: 4,
  rounds: 1,
  sampler: "greedy",
  prompts: ["few-ids"],
};

/**
 * A fake probe: while enabled it records one pass of one dispatch per token step (labelled with
 * {@link LINEAR} in the split mode), and `readTimes` times every pass at 0.25 ms.
 */
const fakeProbe = (): BenchProbe & { readonly record: () => void } => {
  let passes: { dispatches: number; label?: string; beginNs?: bigint; endNs?: bigint }[] = [];
  const probe: BenchProbe & { readonly record: () => void } = {
    enabled: false,
    mode: "passes",
    passes: () => passes,
    submits: () => passes.length,
    readTimes: () => {
      passes = passes.map((pass, index) => ({
        ...pass,
        beginNs: BigInt(index * 1_000_000),
        endNs: BigInt(index * 1_000_000 + 250_000),
      }));
      return Promise.resolve();
    },
    reset: () => {
      passes = [];
    },
    record: () => {
      if (probe.enabled) {
        passes.push({ dispatches: 1, ...(probe.mode === "dispatches" ? { label: LINEAR } : {}) });
      }
    },
  };
  return probe;
};

/**
 * What a fake run does besides streaming: `failChatAt` (0-based over the run) fails that chat,
 * `failBuild` fails every build, `onPause` runs at every pause, and `events` logs the builds, the
 * pauses, the chats and the progress lines in order.
 */
type FakeOptions = {
  readonly failChatAt?: number;
  readonly failBuild?: boolean;
  readonly onPause?: () => void;
  readonly events?: string[];
};

/**
 * Fake dependencies: a clock that ticks 1 ms per reading, and pipelines whose `chat` streams
 * `newTokens − 1` tokens then a stop token, one probe pass per step.
 */
const fakeDependencies = (options: FakeOptions): BenchDependencies => {
  const probe = fakeProbe();
  let clock = 0;
  let chats = 0;
  return {
    probe,
    now: () => (clock += 1),
    sha256: (bytes) => Promise.resolve(`sha-${bytes.byteLength}`),
    pause: () => {
      options.events?.push("pause");
      options.onPause?.();
      return Promise.resolve();
    },
    build: (arm) => {
      options.events?.push(`build ${arm.model}`);
      if (options.failBuild === true) return Promise.reject(new Error("build failed"));
      const pipeline: BenchPipeline = {
        encodePrompt: () => [1, 2, 3],
        chat: (_messages, chatOptions) => {
          options.events?.push("chat");
          const index = chats;
          chats += 1;
          let settle!: (stop: ChatStop) => void;
          let reject!: (error: unknown) => void;
          const done = new Promise<ChatStop>((resolve, fail) => {
            settle = resolve;
            reject = fail;
          });
          done.catch(() => {});
          const stream: ChatStream = {
            done,
            [Symbol.asyncIterator]: () => {
              throw new Error("the runner drains with text()");
            },
            text: () => {
              if (index === options.failChatAt) {
                const error = new Error("chat failed");
                reject(error);
                return Promise.reject(error);
              }
              chatOptions.onPrefill?.({ chunk: 1, chunks: 1 });
              const maxNewTokens = chatOptions.maxNewTokens ?? 0;
              for (let token = 0; token < maxNewTokens - 1; token += 1) {
                probe.record();
                chatOptions.onToken?.(100 + token);
              }
              probe.record();
              settle({ reason: "eos", tokens: maxNewTokens, token: 1 });
              return Promise.resolve("text");
            },
          };
          return stream;
        },
        dispose: () => Promise.resolve(),
      };
      return Promise.resolve(pipeline);
    },
  };
};

/** One fake bench run of `arms` and its result. */
const benchOf = async (
  arms: readonly RunArm[],
  options: FakeOptions & { readonly signal?: AbortSignal } = {},
) => {
  const settings = SETTINGS;
  const outcome = await settleDecodeBench({
    arms,
    settings,
    signal: options.signal ?? SIGNAL,
    dependencies: fakeDependencies(options),
    onProgress: (progress) =>
      options.events?.push(benchProgressLine(progress, arms, settings.chats)),
  });
  const result = () =>
    decodeBenchResult({ timestampQuery: true, settings, arms, distributions: FACTS, outcome });
  return { outcome, result };
};

describe("the gemma4-bench entry", () => {
  describe("its parameters", () => {
    it("are the arms, the workload, the device and the source, for a local distribution", () => {
      expect(gemmaBench.id).toBe("gemma4-bench");
      expect(gemmaBench.parameters.map(({ name, kind }) => `${name}:${kind}`)).toEqual([
        "arms:text",
        "chats:integer",
        "newTokens:integer",
        "rounds:integer",
        "sampler:choice",
        "prompts:choices",
        "subgroups:boolean",
        "source:choice",
      ]);
    });

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

    it("need a warm chat, a decode token and a round, and refuse what is not an integer", () => {
      expect(parameters({ chats: "2", newTokens: "2", rounds: "1" })).toMatchObject({
        chats: 2,
        newTokens: 2,
        rounds: 1,
      });
      for (const [name, value] of [
        ["chats", "1"],
        ["newTokens", "1"],
        ["rounds", "0"],
        ["chats", "2.5"],
        ["rounds", "-1"],
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
    it("has no verdict for a bench that finished, and one line per arm and prompt: rates, ms/token, the ratio to arm 1, GPU time when tapped, identity", async () => {
      const { result } = await benchOf([REGULAR_ARM, QAT_ARM]);
      const finished = result();
      expect(Object.hasOwn(finished, "verdict")).toBe(false);
      // Each chat: 3 decode tokens over 3 ms, one 0.25 ms pass per decode step.
      expect(finished.summary).toEqual([
        "2 of 2 visits, 4 chats (cold ones included)",
        "arm 1 Gemma 4 e2b bf16 · Story request · decode 1000.0 tok/s (max 1000.0) · 1.000 ms/token · GPU 0.250 ms/token · outputs identical",
        "arm 2 QAT q4_0 e2b i4 (tap off) · Story request · decode 1000.0 tok/s (max 1000.0) · 1.000 ms/token · ×1.000 vs arm 1 · outputs identical",
      ]);
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
          integerGemv: false,
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
        "arm 1 Gemma 4 e2b bf16 (split) · Story request · decode 1000.0 tok/s (max 1000.0) · 1.000 ms/token · GPU 0.250 ms/token · outputs identical",
        "arm 1 Gemma 4 e2b bf16 (split) · Story request · 1 warm chats · GPU 0.250 ms/token (split)",
        "    ms/tok  share disp/tok  family / key",
        "     0.250 100.0%      1.0  linear",
        `     0.250 100.0%      1.0  linear  ${LINEAR}`,
      ]);
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
