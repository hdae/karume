import { describe, expect, it } from "vite-plus/test";

import { createGpuTap } from "../src/models/gemma4-bench/tap.ts";

/**
 * A fake WebGPU that runs every command at encode time: a timestamped pass writes `begin` / `end`
 * (1,000 ns apart per pass, 400 ns long) into its query set when it ends, `resolveQuerySet` and
 * `copyBufferToBuffer` copy bytes, and a mapped buffer shows its bytes.
 */
const fakeWebGpu = (adapterFeatures: readonly string[]) => {
  const passDescriptors: (GPUComputePassDescriptor | undefined)[] = [];
  let clockNs = 0n;
  const bufferBytes = new WeakMap<object, Uint8Array>();
  const createBuffer = (descriptor: GPUBufferDescriptor) => {
    const bytes = new Uint8Array(descriptor.size);
    const buffer = {
      size: descriptor.size,
      usage: descriptor.usage,
      mapAsync: () => Promise.resolve(),
      getMappedRange: () => bytes.buffer,
      unmap: () => {},
      destroy: () => {},
    };
    bufferBytes.set(buffer, bytes);
    return buffer;
  };
  const createDevice = (descriptor: GPUDeviceDescriptor | undefined) => {
    const submitted: number[] = [];
    const device = {
      features: new Set<string>(descriptor?.requiredFeatures ?? []),
      queue: { submit: (buffers: Iterable<unknown>) => submitted.push([...buffers].length) },
      createBuffer,
      createQuerySet: (query: GPUQuerySetDescriptor) => ({
        values: new BigUint64Array(query.count),
      }),
      createCommandEncoder: () => ({
        beginComputePass: (pass?: GPUComputePassDescriptor) => {
          passDescriptors.push(pass);
          return {
            dispatchWorkgroups: () => {},
            end: () => {
              const writes = pass?.timestampWrites;
              if (writes === undefined) return;
              const { values } = writes.querySet as unknown as { values: BigUint64Array };
              clockNs += 1000n;
              values[writes.beginningOfPassWriteIndex ?? 0] = clockNs;
              values[writes.endOfPassWriteIndex ?? 0] = clockNs + 400n;
            },
          };
        },
        resolveQuerySet: (
          querySet: { values: BigUint64Array },
          first: number,
          count: number,
          destination: object,
          offset: number,
        ) => {
          const target = bufferBytes.get(destination);
          const source = querySet.values.subarray(first, first + count);
          target?.set(new Uint8Array(source.buffer, source.byteOffset, source.byteLength), offset);
        },
        copyBufferToBuffer: (
          source: object,
          sourceOffset: number,
          destination: object,
          destinationOffset: number,
          size: number,
        ) => {
          const from = bufferBytes.get(source);
          bufferBytes
            .get(destination)
            ?.set(from?.subarray(sourceOffset, sourceOffset + size) ?? [], destinationOffset);
        },
        finish: () => ({}),
      }),
    };
    return { device, submitted, requested: [...(descriptor?.requiredFeatures ?? [])] };
  };
  const devices: ReturnType<typeof createDevice>[] = [];
  const adapter = () => ({
    features: new Set(adapterFeatures),
    requestDevice: (descriptor?: GPUDeviceDescriptor) => {
      const created = createDevice(descriptor);
      devices.push(created);
      return Promise.resolve(created.device);
    },
  });
  const gpu = {
    requestAdapter: () => Promise.resolve(adapter()),
    getPreferredCanvasFormat: () => "bgra8unorm",
    wgslLanguageFeatures: new Set(["packed_4x8_integer_dot_product"]),
  } as unknown as GPU;
  return { gpu, passDescriptors, devices };
};

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
      const fake = fakeWebGpu(["timestamp-query", "shader-f16"]);
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      expect([...device.features]).toEqual(["shader-f16"]);
      expect(fake.devices[0].requested).toEqual(["shader-f16", "timestamp-query"]);
      expect(tap.probe(device)?.timestamps).toBe(true);
    });

    it("counts passes, dispatches and submits and times every pass once enabled", async () => {
      const fake = fakeWebGpu(["timestamp-query"]);
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
      const fake = fakeWebGpu(["timestamp-query"]);
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      const probe = tap.probe(device);
      encode(device, 2, 3);
      expect(probe?.passes()).toEqual([]);
      expect(probe?.submits()).toBe(0);
      expect(fake.passDescriptors).toEqual([undefined, undefined]);
      // The work still reaches the queue.
      expect(fake.devices[0].submitted).toEqual([1]);
    });

    it("keeps a pass's own timestamp writes and leaves that pass untimed", async () => {
      const fake = fakeWebGpu(["timestamp-query"]);
      const tap = createGpuTap(fake.gpu);
      const device = await acquire(tap.gpu);
      const probe = tap.probe(device);
      if (probe === undefined) throw new Error("no probe");
      probe.enabled = true;
      const own = { querySet: device.createQuerySet({ type: "timestamp", count: 2 }) };
      const encoder = device.createCommandEncoder();
      encoder.beginComputePass({ timestampWrites: own }).end();
      expect(fake.passDescriptors[0]?.timestampWrites).toBe(own);
      expect(probe.passes()[0].slot).toBe(-1);
    });

    it("starts over after a reset", async () => {
      const fake = fakeWebGpu(["timestamp-query"]);
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
      const fake = fakeWebGpu(["shader-f16"]);
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
      expect(fake.passDescriptors).toEqual([undefined, undefined]);
      await probe.readTimes();
      expect(probe.passes()[0].beginNs).toBeUndefined();
    });
  });

  it("has a probe only for the devices acquired through it, and leaves the inner GPU as it was", async () => {
    const fake = fakeWebGpu(["timestamp-query"]);
    const tap = createGpuTap(fake.gpu);
    const direct = await acquire(fake.gpu, []);
    expect(tap.probe(direct)).toBeUndefined();
    expect([...direct.features]).toEqual([]);
    expect(tap.gpu.wgslLanguageFeatures).toBe(fake.gpu.wgslLanguageFeatures);
    expect(tap.gpu.getPreferredCanvasFormat()).toBe("bgra8unorm");
  });
});
