/**
 * kernel-record: which WGSL texts a test dispatched. The classes here are fakes with the three WebGPU
 * calls the record observes, so the cases run without a GPU.
 */

import { assertEquals, assertNotEquals } from "@std/assert";
import { beforeEach, describe, it } from "vite-plus/test";
import {
  beginKernelRecordFile,
  beginKernelRecordTest,
  endKernelRecordTest,
  installKernelRecord,
  kernelSetFingerprint,
  kernelsInUse,
  kernelsOfFile,
} from "../src/kernel-record.ts";

type Module = { readonly label?: string };
type Pipeline = { readonly label?: string };
type PipelineDescriptor = {
  readonly label?: string;
  readonly compute: { readonly module: Module };
};

class FakeDevice {
  createShaderModule(descriptor: { readonly label?: string; readonly code: string }): Module {
    return { label: descriptor.label };
  }
  createComputePipeline(descriptor: PipelineDescriptor): Pipeline {
    return { label: descriptor.label };
  }
  createComputePipelineAsync(descriptor: PipelineDescriptor): Promise<Pipeline> {
    return Promise.resolve({ label: descriptor.label });
  }
}

class FakePass {
  readonly selected: Pipeline[] = [];
  setPipeline(pipeline: Pipeline): void {
    this.selected.push(pipeline);
  }
}

installKernelRecord({ GPUDevice: FakeDevice, GPUComputePassEncoder: FakePass });

const pipelineOf = async (device: FakeDevice, label: string, code: string): Promise<Pipeline> =>
  await device.createComputePipelineAsync({
    label,
    compute: { module: device.createShaderModule({ label, code }) },
  });

const labels = (kernels: readonly { readonly label: string }[]): string[] =>
  kernels.map((kernel) => kernel.label);

describe("the kernel record", () => {
  // The suite's own setup file opened a bucket for this test; start from an empty file instead.
  beforeEach(() => {
    endKernelRecordTest();
    beginKernelRecordFile();
  });

  it("counts a kernel when a pass selects its pipeline, not when it is compiled", async () => {
    const device = new FakeDevice();
    const compiled = await pipelineOf(device, "add", "fn add() {}");
    assertEquals(kernelsOfFile(), []);
    new FakePass().setPipeline(compiled);
    assertEquals(labels(kernelsOfFile()), ["add"]);
  });

  it("still calls the wrapped method and returns its result", async () => {
    const device = new FakeDevice();
    const pass = new FakePass();
    const pipeline = await pipelineOf(device, "mul", "fn mul() {}");
    pass.setPipeline(pipeline);
    assertEquals(pass.selected, [pipeline]);
    const direct = device.createComputePipeline({
      label: "sync",
      compute: { module: device.createShaderModule({ code: "fn sync() {}" }) },
    });
    assertEquals(direct.label, "sync");
  });

  it("gives a test its own kernels plus the ones used outside any test", async () => {
    const device = new FakeDevice();
    const pass = new FakePass();
    const shared = await pipelineOf(device, "shared", "fn shared() {}");
    const first = await pipelineOf(device, "first", "fn first() {}");
    const second = await pipelineOf(device, "second", "fn second() {}");

    pass.setPipeline(shared); // as a beforeAll would
    beginKernelRecordTest();
    pass.setPipeline(first);
    assertEquals(labels(kernelsInUse()), ["first", "shared"]);
    endKernelRecordTest();

    beginKernelRecordTest();
    pass.setPipeline(second);
    assertEquals(labels(kernelsInUse()), ["second", "shared"]);
    endKernelRecordTest();

    assertEquals(labels(kernelsOfFile()), ["first", "second", "shared"]);
  });

  it("sees a pipeline compiled by an earlier test when a later test reuses it", async () => {
    const device = new FakeDevice();
    const pass = new FakePass();
    beginKernelRecordTest();
    const reused = await pipelineOf(device, "reused", "fn reused() {}");
    pass.setPipeline(reused);
    endKernelRecordTest();

    beginKernelRecordTest();
    pass.setPipeline(reused);
    assertEquals(labels(kernelsInUse()), ["reused"]);
    endKernelRecordTest();
  });

  it("tells two WGSL texts under one label apart, and gives a set one fingerprint in any order", async () => {
    const device = new FakeDevice();
    const pass = new FakePass();
    pass.setPipeline(await pipelineOf(device, "gemm", "fn gemm() { /* v1 */ }"));
    const before = kernelsOfFile();
    pass.setPipeline(await pipelineOf(device, "gemm", "fn gemm() { /* v2 */ }"));
    const after = kernelsOfFile();
    assertEquals(labels(after), ["gemm", "gemm"]);
    assertNotEquals(after[0].wgslSha256, after[1].wgslSha256);
    assertNotEquals(kernelSetFingerprint(before), kernelSetFingerprint(after));
    assertEquals(kernelSetFingerprint(after), kernelSetFingerprint([after[1], after[0]]));
  });

  it("forgets the previous file's kernels when a file begins", async () => {
    const device = new FakeDevice();
    new FakePass().setPipeline(await pipelineOf(device, "old", "fn old() {}"));
    beginKernelRecordFile();
    assertEquals(kernelsOfFile(), []);
    assertEquals(kernelsInUse(), []);
  });
});
