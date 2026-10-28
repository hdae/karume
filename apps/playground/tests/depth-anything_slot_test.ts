import { describe, expect, it } from "vite-plus/test";

import { createPipelineSlot } from "../src/models/depth-anything/pipeline-slot.ts";

/** A fake pipeline that records whether it was disposed. */
type FakePipeline = { readonly name: string; disposed: boolean; dispose: () => Promise<void> };

const fake = (name: string): FakePipeline => {
  const pipeline: FakePipeline = {
    name,
    disposed: false,
    dispose: () => {
      pipeline.disposed = true;
      return Promise.resolve();
    },
  };
  return pipeline;
};

/** A build the test finishes by hand, to land it before or after a settings change. */
const deferredBuild = () => {
  let finish: (pipeline: FakePipeline) => void = () => undefined;
  const promise = new Promise<FakePipeline>((resolve) => {
    finish = resolve;
  });
  return { build: () => promise, finish: (pipeline: FakePipeline) => finish(pipeline) };
};

/** The disposals the slot handed to `track` (useGpu's, in the page). */
let tracked: Promise<void>[] = [];

const newSlot = () => {
  let clock = 0;
  tracked = [];
  return createPipelineSlot<string, FakePipeline>({
    sameKey: (left, right) => left === right,
    now: () => (clock += 100),
    track: (disposal) => {
      tracked.push(disposal);
      return disposal;
    },
  });
};

const signal = () => new AbortController().signal;

describe("createPipelineSlot", () => {
  describe("when the key stays the same", () => {
    it("builds once and hands the same pipeline to later runs", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      let builds = 0;
      const build = () => {
        builds += 1;
        return Promise.resolve(fake(`build ${builds}`));
      };
      const first = await slot.obtain("small/f32", build, signal());
      const second = await slot.obtain("small/f32", build, signal());
      expect(builds).toBe(1);
      expect(first.built).toBe(true);
      expect(first.entry.loadMs).toBe(100);
      expect(second).toEqual({ entry: first.entry, built: false });
      expect(slot.current()).toBe(first.entry);
    });
  });

  describe("when the wanted key moves away", () => {
    it("disposes the open pipeline at once and tells the subscribers", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      const { entry } = await slot.obtain("small/f32", () => Promise.resolve(fake("a")), signal());
      let notified = 0;
      slot.subscribe(() => {
        notified += 1;
      });
      slot.want("small/f16");
      expect(entry.value.disposed).toBe(true);
      expect(slot.current()).toBeUndefined();
      expect(notified).toBe(1);
    });

    it("hands that dispose to track, so the GPU outlives it", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      await slot.obtain("small/f32", () => Promise.resolve(fake("a")), signal());
      expect(tracked).toHaveLength(0);
      slot.want(undefined);
      expect(tracked).toHaveLength(1);
    });

    it("disposes a build that finishes afterwards and rejects the run instead of keeping it", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      const deferred = deferredBuild();
      const running = slot.obtain("small/f32", deferred.build, signal());
      slot.want("small/f16");
      const built = fake("late");
      deferred.finish(built);
      await expect(running).rejects.toThrow(/settings changed/);
      expect(built.disposed).toBe(true);
      expect(slot.current()).toBeUndefined();
    });

    it("refuses a run for a key that is not the wanted one", async () => {
      const slot = newSlot();
      slot.want("small/f16");
      await expect(
        slot.obtain("small/f32", () => Promise.resolve(fake("a")), signal()),
      ).rejects.toThrow(/settings changed/);
    });
  });

  describe("when the page lets go (want(undefined))", () => {
    it("disposes the open pipeline", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      const { entry } = await slot.obtain("small/f32", () => Promise.resolve(fake("a")), signal());
      slot.want(undefined);
      expect(entry.value.disposed).toBe(true);
    });

    it("disposes a build that was in flight when it finishes", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      const deferred = deferredBuild();
      const running = slot.obtain("small/f32", deferred.build, signal());
      slot.want(undefined);
      const built = fake("orphan");
      deferred.finish(built);
      await expect(running).rejects.toThrow();
      expect(built.disposed).toBe(true);
    });

    it("can be wanted again afterwards (React may re-run effects)", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      slot.want(undefined);
      slot.want("small/f32");
      const { built } = await slot.obtain("small/f32", () => Promise.resolve(fake("a")), signal());
      expect(built).toBe(true);
    });
  });

  describe("when the run is cancelled during the build", () => {
    it("disposes the new pipeline and rejects with the cancel reason", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      const controller = new AbortController();
      const deferred = deferredBuild();
      const running = slot.obtain("small/f32", deferred.build, controller.signal);
      const reason = new Error("Cancelled by the user");
      controller.abort(reason);
      const built = fake("cancelled");
      deferred.finish(built);
      await expect(running).rejects.toBe(reason);
      expect(built.disposed).toBe(true);
      expect(slot.current()).toBeUndefined();
    });

    it("lets the next run build again", async () => {
      const slot = newSlot();
      slot.want("small/f32");
      const controller = new AbortController();
      controller.abort();
      await expect(
        slot.obtain("small/f32", () => Promise.resolve(fake("a")), controller.signal),
      ).rejects.toThrow();
      const { built } = await slot.obtain("small/f32", () => Promise.resolve(fake("b")), signal());
      expect(built).toBe(true);
    });
  });

  it("passes a failed build through and lets the next run build again", async () => {
    const slot = newSlot();
    slot.want("small/f32");
    await expect(
      slot.obtain("small/f32", () => Promise.reject(new Error("fetch failed")), signal()),
    ).rejects.toThrow("fetch failed");
    const { built } = await slot.obtain("small/f32", () => Promise.resolve(fake("b")), signal());
    expect(built).toBe(true);
  });

  it("refuses a second build while one is in flight", async () => {
    const slot = newSlot();
    slot.want("small/f32");
    const deferred = deferredBuild();
    const first = slot.obtain("small/f32", deferred.build, signal());
    await expect(slot.obtain("small/f32", deferred.build, signal())).rejects.toThrow(
      /already being built/,
    );
    deferred.finish(fake("a"));
    await expect(first).resolves.toMatchObject({ built: true });
  });
});
