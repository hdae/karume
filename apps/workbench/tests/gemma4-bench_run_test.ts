import { describe, expect, it } from "vite-plus/test";

import type { BenchSettings } from "../src/gemma4-bench/plan.ts";
import { type RunArm, settleDecodeBench } from "../src/gemma4-bench/run.ts";
import { fakeBenchDependencies } from "./helpers/gemma4-bench-fakes.ts";

const arm = (model: string, tap = true): RunArm => ({
  mount: model.startsWith("qat-") ? "gemma4-qat-mobile" : "gemma4",
  model,
  quant: "i4-fast",
  quantLabel: "int4 fast",
  isDefaultQuant: true,
  tap,
  sampler: { temperature: 0 },
  distribution: {
    ref: { repo: `test/${model}` },
    facts: { directory: `models/${model}`, manifestSha256: "00", read: "directory" },
  },
});

const SETTINGS: BenchSettings = {
  chats: 2,
  newTokens: 4,
  rounds: 2,
  sampler: "greedy",
  prompts: ["few-ids", "many-ids"],
};

const runBench = async (input: {
  readonly arms: readonly RunArm[];
  readonly settings?: BenchSettings;
  readonly failChatAt?: number;
  readonly signal?: AbortSignal;
}) => {
  const { dependencies, probe, world } = fakeBenchDependencies({ failChatAt: input.failChatAt });
  const { visits, chats, stop } = await settleDecodeBench({
    arms: input.arms,
    settings: input.settings ?? SETTINGS,
    signal: input.signal ?? new AbortController().signal,
    dependencies,
    onProgress: () => {},
  });
  return { world, probe, visits, chats, stop };
};

describe("the bench loop", () => {
  it("visits the arms in ABBA order, building each pipeline only after the previous one is released", async () => {
    const { world, stop } = await runBench({ arms: [arm("e2b"), arm("qat-e2b")] });
    expect(stop).toEqual({ kind: "completed" });
    expect(world.lifecycle).toEqual([
      "build e2b",
      "dispose e2b",
      "build qat-e2b",
      "dispose qat-e2b",
      "build qat-e2b",
      "dispose qat-e2b",
      "build e2b",
      "dispose e2b",
    ]);
    expect(world.maxResident()).toBe(1);
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
    const { world, probe, chats } = await runBench({
      arms: [arm("e2b", true), arm("e2b-control", false)],
      settings: { ...SETTINGS, rounds: 1, prompts: ["few-ids"] },
    });
    expect(world.enabledAtChat).toEqual([true, true, false, false]);
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
    it("releases the pipeline, keeps what finished, and stops failed with the chat's error", async () => {
      const { world, chats, stop } = await runBench({ arms: [arm("e2b")], failChatAt: 1 });
      expect(stop).toEqual({ kind: "failed", error: new Error("chat failed") });
      expect(world.lifecycle).toEqual(["build e2b", "dispose e2b"]);
      expect(chats).toHaveLength(1);
    });
  });

  describe("when the run is aborted", () => {
    it("stops before building the next visit", async () => {
      const controller = new AbortController();
      controller.abort(new Error("Cancelled by the user"));
      const { world, stop } = await runBench({
        arms: [arm("e2b")],
        signal: controller.signal,
      });
      expect(stop).toEqual({ kind: "cancelled" });
      expect(world.lifecycle).toEqual([]);
    });
  });
});
