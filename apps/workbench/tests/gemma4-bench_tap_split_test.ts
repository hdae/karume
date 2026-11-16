import { describe, expect, it } from "vite-plus/test";

import { createGpuTap, TIMED_PASS_SLOTS } from "../src/gemma4-bench/tap.ts";
import { fakeWebGpu, type RealPass } from "./helpers/fake-webgpu.ts";

/** A device acquired through a tap on the fake, with its probe enabled in split mode. */
const splitDevice = async () => {
  const fake = fakeWebGpu();
  const tap = createGpuTap(fake.gpu);
  const adapter = await tap.gpu.requestAdapter();
  if (adapter === null) throw new Error("no adapter");
  const device = await adapter.requestDevice();
  const probe = tap.probe(device);
  if (probe === undefined) throw new Error("no probe");
  probe.enabled = true;
  probe.mode = "dispatches";
  return { fake, device, probe };
};

const pipeline = (label: string) => ({ label }) as unknown as GPUComputePipeline;
const bindGroup = (name: string) => ({ name }) as unknown as GPUBindGroup;

/** The calls of every real pass, with the replayed state and dispatch only (no `end`). */
const callsOf = (realPasses: readonly RealPass[]) =>
  realPasses.map(({ calls }) => calls.filter(([name]) => name !== "end"));

describe("the GPU tap in split mode", () => {
  it("times every dispatch in a pass of its own, labelled with its pipeline", async () => {
    const { device, probe } = await splitDevice();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline("rms_norm:v1:f32:lastdim:wg256"));
    pass.setBindGroup(0, bindGroup("a"));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(pipeline("linear_gemv_parallel:wi4g32:l32"));
    pass.dispatchWorkgroups(2);
    pass.dispatchWorkgroups(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    expect(probe.passes().map(({ slot, dispatches, label }) => [slot, dispatches, label])).toEqual([
      [0, 1, "rms_norm:v1:f32:lastdim:wg256"],
      [1, 1, "linear_gemv_parallel:wi4g32:l32"],
      [2, 1, "linear_gemv_parallel:wi4g32:l32"],
    ]);
    await probe.readTimes();
    expect(probe.passes().map(({ beginNs, endNs }) => [beginNs, endNs])).toEqual([
      [1000n, 1400n],
      [2000n, 2400n],
      [3000n, 3400n],
    ]);
  });

  it("replays the pipeline and every bind group as last set, with its offsets, in each new pass", async () => {
    const { fake, device } = await splitDevice();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline("p"));
    const data = new Uint32Array([7, 8, 9]);
    pass.setBindGroup(1, bindGroup("b"), data, 1, 2);
    pass.setBindGroup(0, bindGroup("a"), [256]);
    // Changing the caller's array afterwards does not change what was set.
    data.set([0, 0, 0]);
    pass.dispatchWorkgroups(4, 5, 6);
    pass.setBindGroup(0, bindGroup("c"));
    pass.dispatchWorkgroups(7);
    pass.end();
    encoder.finish();
    expect(callsOf(fake.realPasses)).toEqual([
      [
        ["setPipeline", "p"],
        ["setBindGroup", 1, { name: "b" }, [8, 9]],
        ["setBindGroup", 0, { name: "a" }, [256]],
        ["dispatch", 4, 5, 6],
        // Forwarded as called (the pass already holds its dispatch).
        ["setBindGroup", 0, { name: "c" }, []],
      ],
      [
        ["setPipeline", "p"],
        ["setBindGroup", 0, { name: "c" }, []],
        ["setBindGroup", 1, { name: "b" }, [8, 9]],
        ["dispatch", 7, 1, 1],
      ],
    ]);
  });

  it("closes the open debug groups at a split and opens them again in the next pass", async () => {
    const { fake, device, probe } = await splitDevice();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline("p"));
    pass.pushDebugGroup("layer 0");
    pass.dispatchWorkgroups(1);
    pass.insertDebugMarker("mark");
    pass.dispatchWorkgroups(1);
    pass.popDebugGroup();
    pass.end();
    encoder.finish();
    expect(callsOf(fake.realPasses)).toEqual([
      [
        ["setPipeline", "p"],
        ["push", "layer 0"],
        ["dispatch", 1, 1, 1],
        ["marker", "mark"],
        ["pop"],
      ],
      [["setPipeline", "p"], ["push", "layer 0"], ["dispatch", 1, 1, 1], ["pop"]],
    ]);
    expect(probe.passes()).toHaveLength(2);
  });

  it("splits indirect dispatches too", async () => {
    const { fake, device, probe } = await splitDevice();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline("p"));
    pass.dispatchWorkgroups(1);
    pass.dispatchWorkgroupsIndirect({} as GPUBuffer, 12);
    pass.end();
    expect(fake.realPasses).toHaveLength(2);
    expect(probe.passes().map(({ dispatches }) => dispatches)).toEqual([1, 1]);
  });

  it("leaves a pass that writes its own timestamps whole and untimed", async () => {
    const { fake, device, probe } = await splitDevice();
    const own = { querySet: device.createQuerySet({ type: "timestamp", count: 2 }) };
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: own });
    pass.setPipeline(pipeline("p"));
    pass.dispatchWorkgroups(1);
    pass.dispatchWorkgroups(1);
    pass.end();
    expect(fake.realPasses).toHaveLength(1);
    expect(fake.realPasses[0].descriptor?.timestampWrites).toBe(own);
    expect(probe.passes().map(({ slot, dispatches }) => [slot, dispatches])).toEqual([[-1, 2]]);
  });

  it("refuses immediate data, which a new pass would lose", async () => {
    const { device } = await splitDevice();
    const pass = device.createCommandEncoder().beginComputePass();
    expect(() => pass.setImmediates(0, new Uint32Array(1))).toThrow(/setImmediates/);
  });

  it("keeps each pair until it is resolved when one encoder times more passes than the query set holds", async () => {
    const { device, probe } = await splitDevice();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline("p"));
    // 4,096 queries = 2,048 pairs: the pairs are reused twice over within this one encoder.
    const dispatches = 2 * 2048 + 10;
    for (let index = 0; index < dispatches; index += 1) pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await probe.readTimes();
    const wrong = probe
      .passes()
      .filter(({ beginNs }, index) => beginNs !== BigInt((index + 1) * 1000));
    expect(probe.passes()).toHaveLength(dispatches);
    expect(wrong).toEqual([]);
  });

  it("leaves a pass untimed rather than overwrite a pair another unfinished encoder has not resolved", async () => {
    const { device, probe } = await splitDevice();
    const first = device.createCommandEncoder();
    const held = first.beginComputePass();
    held.setPipeline(pipeline("held"));
    held.dispatchWorkgroups(1);
    held.end();
    // A second encoder runs a whole round of the query set while the first is still open.
    const second = device.createCommandEncoder();
    const pass = second.beginComputePass();
    pass.setPipeline(pipeline("p"));
    for (let index = 0; index < 2048; index += 1) pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([first.finish()]);
    device.queue.submit([second.finish()]);
    await probe.readTimes();
    const passes = probe.passes();
    expect(passes.map(({ slot }) => slot).slice(-2)).toEqual([2047, -1]);
    expect([passes[0].beginNs, passes[0].endNs]).toEqual([1000n, 1400n]);
  });

  it("counts the passes past its capacity without timing them", async () => {
    const { device, probe } = await splitDevice();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline("p"));
    for (let index = 0; index < TIMED_PASS_SLOTS + 3; index += 1) pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await probe.readTimes();
    const passes = probe.passes();
    expect(passes).toHaveLength(TIMED_PASS_SLOTS + 3);
    expect(passes.slice(-4).map(({ slot }) => slot)).toEqual([TIMED_PASS_SLOTS - 1, -1, -1, -1]);
    expect(passes.slice(-4).map(({ beginNs }) => beginNs)).toEqual([
      BigInt(TIMED_PASS_SLOTS * 1000),
      undefined,
      undefined,
      undefined,
    ]);
  });

  it("records no labels and keeps the product's passes in the default mode", async () => {
    const { fake, device, probe } = await splitDevice();
    probe.mode = "passes";
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline("p"));
    pass.dispatchWorkgroups(1);
    pass.dispatchWorkgroups(1);
    pass.end();
    expect(fake.realPasses).toHaveLength(1);
    expect(probe.passes()).toEqual([{ slot: 0, dispatches: 2 }]);
  });
});
