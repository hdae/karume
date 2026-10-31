// Bit identity of the v2 SRQ search (the level guess, perf-ledger K-97) with v1's binary search on
// the GPU: the standalone kernels (f32 and packed int8 output) and the GEMV's fused epilogue, against
// their v1 WGSL kept test-only in fixtures/wgsl-baseline/ (the generators no longer emit it).
//
// The inputs are adversarial for a guess: every boundary of the table ±2 ulps with both signs, ±0,
// ±inf, NaN payloads, subnormals and random words, under scales that push the division to its edges
// (subnormal and tiny scales overflow it to inf, large ones round many magnitudes to level 0).
//
// Fault injection (2026-10-31, RTX 3080 Ti): dropping the guess's lower check fails both cases;
// dropping the upper check fails only the standalone packed kernel (on NaN, which it does not return
// early), because this GPU's division never proposes a level below the true one. The upper check is
// still the guard on a backend whose division is less accurate (WGSL allows 2.5 ULP).
import { readFileSync } from "node:fs";
import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  linearGemvStaticQuantizePackedParams,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeParams,
  linearGemvStaticQuantizeWgsl,
} from "../src/kernels/linear-gemv.ts";
import {
  STATIC_QUANTIZE_PACKED_WGSL,
  STATIC_QUANTIZE_WGSL,
  staticQuantizePackedParams,
  staticQuantizeParams,
} from "../src/kernels/static-quantize.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const baseline = (name: string): string =>
  readFileSync(new URL(`./fixtures/wgsl-baseline/${name}_v1.wgsl`, import.meta.url), "utf8");

const f32Bits = (value: number): number => new Uint32Array(Float32Array.of(value).buffer)[0];

const SCALES = [0.0371, 1, 1 / 128, 2 ** -126, 1e-30, Math.fround(1.4e-45), 3e-3, 1e30].map(
  (scale) => Math.fround(scale),
);

/** Adversarial input words for one scale (count padded to a multiple of 4 for the packed kernel). */
const inputs = (scale: number): Uint32Array<ArrayBuffer> => {
  const table = staticQuantizeParams(4, scale);
  const words: number[] = [];
  for (let j = 0; j < 128; j += 1) {
    const boundary = table[1 + j];
    for (let delta = -2; delta <= 2; delta += 1) {
      const magnitude = Math.min(Math.max(boundary + delta, 0), 0x7fffffff);
      words.push(magnitude, (magnitude | 0x80000000) >>> 0);
    }
  }
  words.push(0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001);
  words.push(1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000);
  let state = 0x9e3779b9;
  for (let i = 0; i < 8192; i += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    words.push(state);
  }
  while (words.length % 4 !== 0) words.push(0);
  return Uint32Array.from(words);
};

describe.skipIf(!GPU_AVAILABLE)("the v2 SRQ level guess on a real GPU (perf-ledger K-97)", () => {
  it("writes the same words as v1's binary search in the standalone f32 and packed kernels", async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    try {
      const run = async (
        code: string,
        params: Uint32Array<ArrayBuffer>,
        input: Uint32Array<ArrayBuffer>,
        outputWords: number,
      ): Promise<Uint32Array<ArrayBuffer>> => {
        device.pushErrorScope("validation");
        const owned: GPUBuffer[] = [];
        const make = (data: Uint32Array<ArrayBuffer> | number, usage: number): GPUBuffer => {
          const size = typeof data === "number" ? data : data.byteLength;
          const buffer = device.createBuffer({ size, usage: usage | U.COPY_DST });
          owned.push(buffer);
          if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
          return buffer;
        };
        try {
          const out = make(outputWords * 4, U.STORAGE | U.COPY_SRC);
          const staging = make(outputWords * 4, U.MAP_READ);
          const pipeline = await device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
          });
          const group = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [make(params, U.UNIFORM), make(input, U.STORAGE), out].map(
              (buffer, binding) => ({ binding, resource: { buffer } }),
            ),
          });
          device.queue.writeBuffer(out, 0, new Uint32Array(outputWords).fill(0xdeadbeef));
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(Math.ceil(input.length / 128));
          pass.end();
          encoder.copyBufferToBuffer(out, 0, staging, 0, outputWords * 4);
          device.queue.submit([encoder.finish()]);
          await staging.mapAsync(MAP_MODE.READ);
          const words = new Uint32Array(staging.getMappedRange().slice(0));
          staging.unmap();
          return words;
        } finally {
          await device.queue.onSubmittedWorkDone();
          for (const buffer of owned) buffer.destroy();
          assertEquals(await device.popErrorScope(), null);
        }
      };
      for (const scale of SCALES) {
        const input = inputs(scale);
        const f32 = staticQuantizeParams(input.length, scale);
        assertEquals(
          await run(STATIC_QUANTIZE_WGSL, f32, input, input.length),
          await run(baseline("static_quantize"), f32, input, input.length),
          `f32 scale ${scale}`,
        );
        const packed = staticQuantizePackedParams(input.length, scale);
        assertEquals(
          await run(STATIC_QUANTIZE_PACKED_WGSL, packed, input, input.length / 4),
          await run(baseline("static_quantize_packed"), packed, input, input.length / 4),
          `packed scale ${scale}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("writes the same words as v1's unrolled search in the GEMV's fused epilogue", async () => {
    // Zero weights and zero activations make every output `quantize(bits(0 + bias))`, so the bias
    // carries the adversarial words straight into the epilogue.
    const gpu = await acquireGpu();
    const device = gpu.device;
    try {
      for (const variant of [
        { name: "linear_gemv_static_quantize_wi8_l32", storage: "i8", lanes: 32, packed: false },
        {
          name: "linear_gemv_static_quantize_packed_wi2_l2",
          storage: "i2",
          lanes: 2,
          packed: true,
        },
      ] as const) {
        const k = 64;
        for (const scale of SCALES) {
          const bias = inputs(scale);
          const n = bias.length;
          const run = async (code: string): Promise<Uint32Array<ArrayBuffer>> => {
            device.pushErrorScope("validation");
            const owned: GPUBuffer[] = [];
            const make = (
              data: ArrayBufferView<ArrayBuffer> | number,
              usage: number,
            ): GPUBuffer => {
              const size = typeof data === "number" ? data : data.byteLength;
              const buffer = device.createBuffer({ size, usage: usage | U.COPY_DST });
              owned.push(buffer);
              if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
              return buffer;
            };
            try {
              const bits = variant.storage === "i2" ? 2 : 8;
              const params = variant.packed
                ? linearGemvStaticQuantizePackedParams(variant.storage, 1, n, k, 0.5, scale)
                : linearGemvStaticQuantizeParams(variant.storage, 1, n, k, scale);
              // i2 stores q + 2, so the zero weight is the code 2 in every 2-bit field (0xaa per byte).
              const weights = new Uint32Array((n * k * bits) / 32).fill(
                variant.storage === "i2" ? 0xaaaaaaaa : 0,
              );
              const out = make(n * 4, U.STORAGE | U.COPY_SRC);
              const staging = make(n * 4, U.MAP_READ);
              const pipeline = await device.createComputePipelineAsync({
                layout: "auto",
                compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
              });
              const group = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                  make(params, U.UNIFORM),
                  make(new Uint32Array(variant.packed ? k / 4 : k), U.STORAGE),
                  make(weights, U.STORAGE),
                  make(bias, U.STORAGE),
                  out,
                  make(new Float32Array(n).fill(0.01), U.STORAGE),
                ].map((buffer, binding) => ({ binding, resource: { buffer } })),
              });
              device.queue.writeBuffer(out, 0, new Uint32Array(n).fill(0xdeadbeef));
              const encoder = device.createCommandEncoder();
              const pass = encoder.beginComputePass();
              pass.setPipeline(pipeline);
              pass.setBindGroup(0, group);
              pass.dispatchWorkgroups(Math.ceil(n / (128 / variant.lanes)), 1);
              pass.end();
              encoder.copyBufferToBuffer(out, 0, staging, 0, n * 4);
              device.queue.submit([encoder.finish()]);
              await staging.mapAsync(MAP_MODE.READ);
              const words = new Uint32Array(staging.getMappedRange().slice(0));
              staging.unmap();
              return words;
            } finally {
              await device.queue.onSubmittedWorkDone();
              for (const buffer of owned) buffer.destroy();
              assertEquals(await device.popErrorScope(), null);
            }
          };
          const current = variant.packed
            ? linearGemvStaticQuantizePackedWgsl(variant.storage, undefined, variant.lanes)
            : linearGemvStaticQuantizeWgsl(variant.storage, undefined, variant.lanes);
          assertEquals(
            await run(current),
            await run(baseline(variant.name)),
            `${variant.name} scale ${scale}`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  });
});

describe("the scale word the v2 guess divides by", () => {
  it("is the scale's own bits at table word 130 for every positive scale", () => {
    for (const scale of SCALES) assertEquals(staticQuantizeParams(4, scale)[130], f32Bits(scale));
  });
});
