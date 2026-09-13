import { assert, assertEquals } from "@std/assert";
import type { GpuContext } from "../../src/gpu/device.ts";
import {
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvSubgroupWgsl,
  linearGemvUnit,
} from "../../src/kernels/linear-gemv.ts";
import { GEMM_TOLERANCE } from "./op-tolerance.ts";
type Storage = "i2" | "i4" | "i8";

/** 圧縮語の境界、group ごとのscale、符号が揃わない決定的な入力。 */
const fixture = (storage: Storage, n: number, k: number, group?: number): {
  packed: Uint32Array<ArrayBuffer>;
  input: Float32Array<ArrayBuffer>;
  bias: Float32Array<ArrayBuffer>;
  scale: Float32Array<ArrayBuffer>;
  reference: Float64Array<ArrayBuffer>;
} => {
  const bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
  const packed = Uint32Array.from(
    { length: n * k * bits / 32 },
    (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
  );
  const input = Float32Array.from({ length: 8 * k }, (_, i) => Math.sin(i * 0.037) * 0.75);
  const bias = Float32Array.from({ length: n }, (_, i) => (i % 7 - 3) * 0.11);
  const scale = Float32Array.from(
    { length: n * (group === undefined ? 1 : k / group) },
    (_, i) => (i % 17 + 1) * 0.00017,
  );
  const reference = new Float64Array(8 * n);
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0;
      for (let inner = 0; inner < k; inner++) {
        const index = col * k + inner;
        const code = (packed[Math.floor(index * bits / 32)] >>> ((index * bits) % 32)) &
          ((1 << bits) - 1);
        const q = bits === 8 ? (code << 24) >> 24 : code - (1 << (bits - 1));
        const s = scale[group === undefined ? col : col * (k / group) + Math.floor(inner / group)];
        sum += input[row * k + inner] * Math.fround(q * s);
      }
      reference[row * n + col] = sum + bias[col];
    }
  }
  return { packed, input, bias, scale, reference };
};

/** Chromeと対応Denoで同じ端数・加算順の門を使う。速度計測ではない。 */
export const checkGemvSubgroup = async (
  gpu: GpuContext,
): Promise<{ cases: number; elements: number }> => {
  const device = gpu.device;
  let cases = 0, elements = 0;
  for (const storage of ["i2", "i4", "i8"] as const) {
    const unit = linearGemvUnit(storage);
    const shapes = [{ k: unit, group: storage === "i4" ? 32 : undefined }, {
      k: unit * 5,
      group: storage === "i4" ? 32 : undefined,
    }, { k: 4096, group: storage === "i4" ? 32 : undefined }];
    if (storage === "i4") {
      shapes.push({ k: 1536, group: 512 }, { k: 6144, group: 2048 }, { k: 4096, group: 4096 });
    }
    for (const { k, group } of shapes) {
      const n = 36, data = fixture(storage, n, k, group), owned: GPUBuffer[] = [];
      device.pushErrorScope("validation");
      const make = (
        value: number | ArrayBufferView<ArrayBuffer>,
        usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      ): GPUBuffer => {
        const buffer = device.createBuffer({
          size: typeof value === "number" ? value : value.byteLength,
          usage,
        });
        owned.push(buffer);
        if (typeof value !== "number") device.queue.writeBuffer(buffer, 0, value);
        return buffer;
      };
      let validationError: GPUError | null = null;
      try {
        const buffers = [
          make(
            linearGemvParams(storage, 8, n, k, group),
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          ),
          make(data.input),
          make(data.packed),
          make(data.bias),
          make(
            8 * n * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          ),
          make(data.scale),
        ];
        const mapped = make(8 * n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        for (const lanes of [2, 4, 8, 16, 32] as const) {
          let first: Uint32Array<ArrayBuffer> | undefined;
          for (const m of [1, 4, 8]) {
            const outputs: Uint32Array<ArrayBuffer>[] = [];
            for (const shader of [linearGemvParallelWgsl, linearGemvSubgroupWgsl]) {
              device.queue.writeBuffer(buffers[0], 0, linearGemvParams(storage, m, n, k, group));
              device.queue.writeBuffer(buffers[4], 0, new Uint32Array(8 * n).fill(0x7fc00000));
              const pipeline = await device.createComputePipelineAsync({
                layout: "auto",
                compute: {
                  module: device.createShaderModule({ code: shader(storage, group, lanes) }),
                  entryPoint: "main",
                },
              });
              const bindings = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
              });
              const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
              pass.setPipeline(pipeline);
              pass.setBindGroup(0, bindings);
              pass.dispatchWorkgroups(Math.ceil(n / (128 / lanes)), m);
              pass.end();
              encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, 8 * n * 4);
              device.queue.submit([encoder.finish()]);
              await mapped.mapAsync(GPUMapMode.READ);
              const out = new Uint32Array(mapped.getMappedRange().slice(0));
              mapped.unmap();
              for (const value of out.subarray(m * n)) {
                assertEquals(value, 0x7fc00000, "未使用行を上書きしない");
              }
              outputs.push(out.slice(0, m * n));
            }
            const label = `${storage} K${k} L${lanes} M${m}`;
            assertEquals(outputs[1], outputs[0], `${label}: parallelとのu32一致`);
            first ??= outputs[1].slice(0, n);
            assertEquals(outputs[1].subarray(0, n), first, `${label}: decode/verify`);
            const values = new Float32Array(outputs[1].buffer);
            for (let i = 0; i < values.length; i++) {
              const expected = data.reference[i];
              assert(Number.isFinite(values[i]));
              assert(
                Math.abs(values[i] - expected) <=
                  GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(expected),
                `${label}: FP64 ${i}`,
              );
            }
            cases++;
            elements += m * n;
          }
        }
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) buffer.destroy();
        validationError = await device.popErrorScope();
      }
      assertEquals(validationError, null);
    }
  }
  return { cases, elements };
};
