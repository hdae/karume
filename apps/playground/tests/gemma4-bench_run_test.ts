import type { ChatOptions, ChatStop, ChatStream } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import type { BenchSettings } from "../src/models/gemma4-bench/plan.ts";
import {
  type BenchDependencies,
  type BenchPipeline,
  type BenchProbe,
  type ChatRecord,
  type RunArm,
  runDecodeBench,
  type VisitRecord,
} from "../src/models/gemma4-bench/run.ts";

const arm = (model: string, tap = true): RunArm => ({
  mount: model.startsWith("qat-") ? "gemma4-qat-mobile" : "gemma4",
  model,
  quant: "i4-fast",
  quantLabel: "int4 fast",
  isDefaultQuant: true,
  tap,
  sampler: { temperature: 0 },
});

const SETTINGS: BenchSettings = {
  chats: 2,
  newTokens: 4,
  rounds: 2,
  sampler: "greedy",
  prompts: ["few-ids", "many-ids"],
};

/** A probe that records one pass of one dispatch per token while enabled. */
type FakeProbe = BenchProbe & {
  /** Whether the probe was enabled when each chat was sent. */
  readonly enabledAtChat: boolean[];
  /** Records one pass while enabled (the fake pipeline calls it per token). */
  readonly record: () => void;
};

const fakeProbe = (): FakeProbe => {
  let passes: { dispatches: number; beginNs?: bigint; endNs?: bigint }[] = [];
  const probe: FakeProbe = {
    enabled: false,
    enabledAtChat: [],
    passes: () => passes,
    submits: () => passes.length,
    readTimes: () => {
      passes = passes.map((pass, index) => ({
        ...pass,
        beginNs: BigInt(index * 1000),
        endNs: BigInt(index * 1000 + 500),
      }));
      return Promise.resolve();
    },
    reset: () => {
      passes = [];
    },
    record: () => {
      if (probe.enabled) passes.push({ dispatches: 1 });
    },
  };
  return probe;
};

type World = {
  readonly log: string[];
  readonly resident: Set<string>;
  readonly maxResident: { value: number };
  readonly sentOptions: ChatOptions[];
};

/**
 * Fake pipelines: `chat` streams `newTokens − 1` tokens then a stop token, one tick of the clock per
 * token, recording a pass on the probe for every token while it is enabled.
 */
const fakeDependencies = (
  world: World,
  probe: FakeProbe,
  failChatAt?: number,
): BenchDependencies => {
  let clock = 0;
  let chats = 0;
  return {
    probe,
    now: () => (clock += 1),
    sha256: (bytes) => Promise.resolve(`sha-${bytes.byteLength}`),
    pause: () => Promise.resolve(),
    build: (built) => {
      world.log.push(`build ${built.model}`);
      world.resident.add(built.model);
      world.maxResident.value = Math.max(world.maxResident.value, world.resident.size);
      const pipeline: BenchPipeline = {
        encodePrompt: (messages) =>
          Array.from({ length: messages[0].content.length }, (_, index) =>
            (messages[0].content as string).charCodeAt(index),
          ),
        chat: (_messages, options) => {
          world.sentOptions.push(options);
          probe.enabledAtChat.push(probe.enabled);
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
            text: async () => {
              if (index === failChatAt) {
                const error = new Error("chat failed");
                reject(error);
                throw error;
              }
              options.onPrefill?.({ chunk: 1, chunks: 1 });
              const maxNewTokens = options.maxNewTokens;
              if (maxNewTokens === undefined) throw new Error("the runner sets maxNewTokens");
              for (let token = 0; token < maxNewTokens - 1; token += 1) {
                options.signal?.throwIfAborted();
                probe.record();
                options.onToken?.(100 + token);
              }
              probe.record();
              settle({ reason: "eos", tokens: maxNewTokens, token: 1 });
              return "text";
            },
          };
          return stream;
        },
        dispose: () => {
          world.log.push(`dispose ${built.model}`);
          world.resident.delete(built.model);
          return Promise.resolve();
        },
      };
      return Promise.resolve(pipeline);
    },
  };
};

const newWorld = (): World => ({
  log: [],
  resident: new Set(),
  maxResident: { value: 0 },
  sentOptions: [],
});

const runBench = async (input: {
  readonly arms: readonly RunArm[];
  readonly settings?: BenchSettings;
  readonly failChatAt?: number;
  readonly signal?: AbortSignal;
}) => {
  const world = newWorld();
  const probe = fakeProbe();
  const visits: VisitRecord[] = [];
  const chats: ChatRecord[] = [];
  const outcome = runDecodeBench({
    arms: input.arms,
    settings: input.settings ?? SETTINGS,
    signal: input.signal ?? new AbortController().signal,
    dependencies: fakeDependencies(world, probe, input.failChatAt),
    onProgress: () => {},
    onVisit: (visit) => visits.push(visit),
    onChat: (chat) => chats.push(chat),
  });
  const settled = await outcome.then(
    () => ({ ok: true as const }),
    (error: unknown) => ({ ok: false as const, error }),
  );
  return { world, probe, visits, chats, settled };
};

describe("the bench loop", () => {
  it("visits the arms in ABBA order, building each pipeline only after the previous one is released", async () => {
    const { world, settled } = await runBench({ arms: [arm("e2b"), arm("qat-e2b")] });
    expect(settled.ok).toBe(true);
    expect(world.log).toEqual([
      "build e2b",
      "dispose e2b",
      "build qat-e2b",
      "dispose qat-e2b",
      "build qat-e2b",
      "dispose qat-e2b",
      "build e2b",
      "dispose e2b",
    ]);
    expect(world.maxResident.value).toBe(1);
  });

  it("runs every prompt `chats` times per visit, in the set's order, chat 0 marked cold", async () => {
    const { chats } = await runBench({ arms: [arm("e2b")], settings: { ...SETTINGS, rounds: 1 } });
    expect(
      chats.map(({ prompt, chat, cold }) => `${prompt}/${chat}${cold ? " cold" : ""}`),
    ).toEqual(["few-ids/0 cold", "few-ids/1", "many-ids/0 cold", "many-ids/1"]);
  });

  it("sends every chat with the arm's sampler and the token limit, never asking for diagnostics", async () => {
    const { world } = await runBench({ arms: [arm("e2b")], settings: { ...SETTINGS, rounds: 1 } });
    for (const options of world.sentOptions) {
      expect(options.sampler).toEqual({ temperature: 0 });
      expect(options.maxNewTokens).toBe(4);
      expect(Object.keys(options).sort()).toEqual([
        "maxNewTokens",
        "onPrefill",
        "onToken",
        "sampler",
        "signal",
      ]);
    }
  });

  it("enables the probe only for a tapped arm's chats, and measures the tap only there", async () => {
    const { probe, chats } = await runBench({
      arms: [arm("e2b", true), arm("e2b-control", false)],
      settings: { ...SETTINGS, rounds: 1, prompts: ["few-ids"] },
    });
    expect(probe.enabledAtChat).toEqual([true, true, false, false]);
    expect(probe.enabled).toBe(false);
    const tapped = chats.filter((chat) => chat.arm === 0);
    const control = chats.filter((chat) => chat.arm === 1);
    expect(tapped.every(({ measurement }) => measurement.tap !== undefined)).toBe(true);
    expect(control.every(({ measurement }) => measurement.tap === undefined)).toBe(true);
    // Three decode tokens after the first: one pass each (the stop token's step included).
    expect(tapped[1].measurement.tap?.perToken).toMatchObject({ passes: 1, dispatches: 1 });
  });

  it("records the stop, the prompt's distinct ids and the hash of the generated ids", async () => {
    const { chats } = await runBench({ arms: [arm("e2b")], settings: { ...SETTINGS, rounds: 1 } });
    const [first] = chats;
    expect(first.stop).toEqual({ reason: "eos", tokens: 4, token: 1 });
    expect(first.distinctPromptIds).toBe(
      new Set("Write a long story about a lighthouse keeper and the sea.").size,
    );
    // Three generated ids (the stop token is not streamed) as u32 = 12 bytes.
    expect(first.tokenIdsSha256).toBe("sha-12");
  });

  it("numbers each visit with its round and arm, with its load time", async () => {
    const { visits } = await runBench({ arms: [arm("e2b"), arm("qat-e2b")] });
    expect(visits.map(({ visit, round, arm: index }) => [visit, round, index])).toEqual([
      [0, 0, 0],
      [1, 0, 1],
      [2, 1, 1],
      [3, 1, 0],
    ]);
    expect(visits.every(({ loadSeconds }) => loadSeconds >= 0)).toBe(true);
  });

  describe("when a chat fails", () => {
    it("releases the pipeline, keeps what finished, and rejects with the chat's error", async () => {
      const { world, chats, settled } = await runBench({ arms: [arm("e2b")], failChatAt: 1 });
      expect(settled.ok ? undefined : settled.error).toEqual(new Error("chat failed"));
      expect(world.log).toEqual(["build e2b", "dispose e2b"]);
      expect(chats).toHaveLength(1);
    });
  });

  describe("when the run is aborted", () => {
    it("stops before building the next visit", async () => {
      const controller = new AbortController();
      controller.abort(new Error("Cancelled by the user"));
      const { world, settled } = await runBench({
        arms: [arm("e2b")],
        signal: controller.signal,
      });
      expect(settled.ok ? undefined : settled.error).toEqual(new Error("Cancelled by the user"));
      expect(world.log).toEqual([]);
    });
  });
});
