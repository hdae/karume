import { describe, expect, it } from "vite-plus/test";

import { createGpuTap } from "../src/gemma4-bench/tap.ts";
import { fakeWebGpu } from "./helpers/fake-webgpu.ts";

/** Acquires a device through the tap as `acquireGpu` does (adapter, then device). */
const acquire = async (gpu: GPU, features: readonly GPUFeatureName[] = ["shader-f16"]) => {
  const adapter = await gpu.requestAdapter();
  if (adapter === null) throw new Error("no adapter");
  return adapter.requestDevice({ requiredFeatures: [...features] });
};

/** Encodes one command buffer of `passes` compute passes with `dispatches` dispatches each, and submits it. */
const encode = (device: GPUDevice, passes: number, dispatches: number) => {
  const encoder = device.createCommandEncoder();
  for (let index = 0; index < passes; index += 1) {
    const pass = encoder.beginComputePass();
    for (let dispatch = 0; dispatch < dispatches; dispatch += 1) pass.dispatchWorkgroups(1);
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
};

describe("the GPU tap", () => {
  describe("on an adapter with timestamp-query", () => {
    it("requests timestamp-query for the device and hides it from the runtime", async () => {
      const fake = fakeWebGpu({ adapterFeatures: ["timestamp-query", "shader-f16"] });
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      expect([...device.features]).toEqual(["shader-f16"]);
      expect(fake.devices[0].requested).toEqual(["shader-f16", "timestamp-query"]);
      expect(tap.probe(device)?.timestamps).toBe(true);
    });

    it("counts passes, dispatches and submits and times every pass once enabled", async () => {
      const fake = fakeWebGpu();
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      const probe = tap.probe(device);
      if (probe === undefined) throw new Error("no probe");
      probe.enabled = true;
      encode(device, 2, 3);
      encode(device, 1, 5);
      expect(probe.submits()).toBe(2);
      expect(probe.passes().map(({ slot, dispatches }) => [slot, dispatches])).toEqual([
        [0, 3],
        [1, 3],
        [2, 5],
      ]);
      await probe.readTimes();
      expect(probe.passes().map(({ beginNs, endNs }) => [beginNs, endNs])).toEqual([
        [1000n, 1400n],
        [2000n, 2400n],
        [3000n, 3400n],
      ]);
    });

    it("records nothing and leaves the passes untouched while disabled", async () => {
      const fake = fakeWebGpu();
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      const probe = tap.probe(device);
      encode(device, 2, 3);
      expect(probe?.passes()).toEqual([]);
      expect(probe?.submits()).toBe(0);
      expect(fake.realPasses.map(({ descriptor }) => descriptor)).toEqual([undefined, undefined]);
      // The work still reaches the queue.
      expect(fake.devices[0].submitted).toEqual([1]);
    });

    it("keeps a pass's own timestamp writes and leaves that pass untimed", async () => {
      const fake = fakeWebGpu();
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      const probe = tap.probe(device);
      if (probe === undefined) throw new Error("no probe");
      probe.enabled = true;
      const own = { querySet: device.createQuerySet({ type: "timestamp", count: 2 }) };
      const encoder = device.createCommandEncoder();
      encoder.beginComputePass({ timestampWrites: own }).end();
      expect(fake.realPasses[0]?.descriptor?.timestampWrites).toBe(own);
      expect(probe.passes()[0].slot).toBe(-1);
    });

    it("starts over after a reset", async () => {
      const fake = fakeWebGpu();
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      const probe = tap.probe(device);
      if (probe === undefined) throw new Error("no probe");
      probe.enabled = true;
      encode(device, 2, 1);
      probe.reset();
      encode(device, 1, 1);
      expect(probe.submits()).toBe(1);
      expect(probe.passes().map(({ slot }) => slot)).toEqual([0]);
      await probe.readTimes();
      expect(probe.passes()[0].endNs).toBe(3400n);
    });
  });

  describe("on an adapter without timestamp-query", () => {
    it("does not request it, and counts passes without timing them", async () => {
      const fake = fakeWebGpu({ adapterFeatures: ["shader-f16"] });
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      const probe = tap.probe(device);
      if (probe === undefined) throw new Error("no probe");
      expect(probe.timestamps).toBe(false);
      expect(fake.devices[0].requested).toEqual(["shader-f16"]);
      probe.enabled = true;
      encode(device, 2, 4);
      expect(probe.passes().map(({ slot, dispatches }) => [slot, dispatches])).toEqual([
        [-1, 4],
        [-1, 4],
      ]);
      expect(fake.realPasses.map(({ descriptor }) => descriptor)).toEqual([undefined, undefined]);
      await probe.readTimes();
      expect(probe.passes()[0].beginNs).toBeUndefined();
    });
  });

  it("has a probe only for the devices acquired through it, and leaves the inner GPU as it was", async () => {
    const fake = fakeWebGpu();
    const tap = createGpuTap(fake.gpu);
    const direct = await acquire(fake.gpu, []);
    expect(tap.probe(direct)).toBeUndefined();
    expect([...direct.features]).toEqual([]);
    expect(tap.gpu.wgslLanguageFeatures).toBe(fake.gpu.wgslLanguageFeatures);
    expect(tap.gpu.getPreferredCanvasFormat()).toBe("bgra8unorm");
  });
});
