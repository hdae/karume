import { describe, expect, it } from "vite-plus/test";

import {
  acquirePipeline,
  type Held,
  type PipelineSlot,
} from "../src/models/irodori/held-pipeline.ts";

/** A fake pipeline that logs its disposal. */
const fakePipeline = (name: string, log: string[]) => ({
  name,
  dispose: async (): Promise<void> => {
    log.push(`dispose ${name}`);
  },
});
type Fake = ReturnType<typeof fakePipeline>;

const memorySlot = (initial?: Held<string, Fake>) => {
  let held = initial;
  const slot: PipelineSlot<string, Fake> = {
    get: () => held,
    set: (next) => {
      held = next;
    },
  };
  return { slot, current: () => held };
};

const same = (left: string, right: string): boolean => left === right;
const always = (): boolean => true;

describe("acquirePipeline", () => {
  it("reuses the held pipeline for the same key without building", async () => {
    const log: string[] = [];
    const old = fakePipeline("old", log);
    const { slot, current } = memorySlot({ key: "a", pipeline: old });
    const result = await acquirePipeline({
      slot,
      key: "a",
      same,
      isCurrent: always,
      signal: new AbortController().signal,
      build: () => Promise.reject(new Error("must not build")),
    });
    expect(result).toEqual({ pipeline: old, built: false });
    expect(current()?.pipeline).toBe(old);
    expect(log).toEqual([]);
  });

  it("disposes the old pipeline before building one for a changed key, then holds the new one", async () => {
    const log: string[] = [];
    const { slot, current } = memorySlot({ key: "a", pipeline: fakePipeline("old", log) });
    const result = await acquirePipeline({
      slot,
      key: "b",
      same,
      isCurrent: always,
      signal: new AbortController().signal,
      onReleasing: () => log.push("releasing"),
      build: async () => {
        log.push(`build (held: ${current() === undefined ? "nothing" : "something"})`);
        return fakePipeline("new", log);
      },
    });
    expect(log).toEqual(["releasing", "dispose old", "build (held: nothing)"]);
    expect(result.built).toBe(true);
    expect(current()).toEqual({ key: "b", pipeline: result.pipeline });
  });

  it("disposes a pipeline built after an abort and throws the abort reason, holding nothing", async () => {
    const log: string[] = [];
    const { slot, current } = memorySlot();
    const controller = new AbortController();
    const reason = new Error("Cancelled by the user");
    const acquiring = acquirePipeline({
      slot,
      key: "a",
      same,
      isCurrent: always,
      signal: controller.signal,
      build: async () => {
        controller.abort(reason);
        return fakePipeline("late", log);
      },
    });
    await expect(acquiring).rejects.toBe(reason);
    expect(log).toEqual(["dispose late"]);
    expect(current()).toBeUndefined();
  });

  it("leaves nothing held when the build fails", async () => {
    const log: string[] = [];
    const { slot, current } = memorySlot({ key: "a", pipeline: fakePipeline("old", log) });
    const failure = new Error("404 karume.json");
    await expect(
      acquirePipeline({
        slot,
        key: "b",
        same,
        isCurrent: always,
        signal: new AbortController().signal,
        build: () => Promise.reject(failure),
      }),
    ).rejects.toBe(failure);
    expect(log).toEqual(["dispose old"]);
    expect(current()).toBeUndefined();
  });

  it("does not build when the run was aborted while the old pipeline was released", async () => {
    const log: string[] = [];
    const controller = new AbortController();
    const old = {
      name: "old",
      dispose: async (): Promise<void> => {
        controller.abort(new Error("The page was left"));
        log.push("dispose old");
      },
    };
    const { slot, current } = memorySlot({ key: "a", pipeline: old });
    await expect(
      acquirePipeline({
        slot,
        key: "b",
        same,
        isCurrent: always,
        signal: controller.signal,
        build: () => Promise.reject(new Error("must not build")),
      }),
    ).rejects.toThrow("The page was left");
    expect(log).toEqual(["dispose old"]);
    expect(current()).toBeUndefined();
  });

  it("disposes a pipeline whose key went stale during the build (a new GPU) and holds nothing", async () => {
    const log: string[] = [];
    const { slot, current } = memorySlot();
    await expect(
      acquirePipeline({
        slot,
        key: "a",
        same,
        isCurrent: (key) => key !== "a",
        signal: new AbortController().signal,
        build: async () => fakePipeline("stale", log),
      }),
    ).rejects.toThrow(/GPU changed while the pipeline was being built/);
    expect(log).toEqual(["dispose stale"]);
    expect(current()).toBeUndefined();
  });
});
