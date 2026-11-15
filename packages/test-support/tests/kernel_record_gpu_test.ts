/**
 * kernel-record on the real WebGPU implementation: the setup file's wrapping of Dawn's classes sees a
 * dispatch of this test. Without an adapter the case is skipped (the fakes of
 * `kernel_record_test.ts` cover the logic).
 */

import { assertEquals } from "@std/assert";
import { it } from "vite-plus/test";
import { kernelsInUse } from "../src/kernel-record.ts";

const WGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() { out[0] = 1.0; }
`;

const sha256Hex = async (text: string): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const adapter = await navigator.gpu?.requestAdapter();

it.skipIf(adapter === undefined || adapter === null)(
  "records the WGSL of a pipeline a compute pass of this test selected",
  async () => {
    const device = await adapter!.requestDevice();
    try {
      const pipeline = await device.createComputePipelineAsync({
        label: "kernel_record_probe",
        layout: "auto",
        compute: { module: device.createShaderModule({ code: WGSL }) },
      });
      const buffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer } }],
        }),
      );
      pass.dispatchWorkgroups(1);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      assertEquals(kernelsInUse(), [
        { label: "kernel_record_probe", wgslSha256: await sha256Hex(WGSL) },
      ]);
    } finally {
      device.destroy();
    }
  },
);
