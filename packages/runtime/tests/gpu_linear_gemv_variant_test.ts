import { assert, assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  defaultLinearGemvVariant,
  linearGemvParams,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";

// 大語彙形で選ばれるWGSLはNを焼かない。端の列・先読み語の端を小さい形で検査し、
// 実形262144×1536はChrome実重み比較とGemmaの既存logits/golden門で検証する。
Deno.test({
  name: "INT8のc16変種はc32とビット一致し、列と先読みの端を正しく書く（実GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    const selected = defaultLinearGemvVariant({ storage: "i8", n: 262144, k: 1536 });
    assertEquals(selected.cols, 16);
    try {
      for (const [n, k] of [[68, 1536], [4, 16], [36, 80]]) {
        device.pushErrorScope("validation");
        const owned: GPUBuffer[] = [];
        const make = (
          data: number | ArrayBufferView<ArrayBuffer>,
          usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        ): GPUBuffer => {
          const buffer = device.createBuffer({
            size: typeof data === "number" ? data : data.byteLength,
            usage,
          });
          owned.push(buffer);
          if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
          return buffer;
        };
        try {
          const packed = Uint32Array.from(
            { length: n * k / 4 },
            (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
          );
          const input = Float32Array.from({ length: k }, (_, i) => Math.sin(i * 0.037) * 0.75);
          const scale = Float32Array.from({ length: n }, (_, i) => (i % 17 + 1) * 0.00017);
          const bias = Float32Array.from({ length: n }, (_, i) => (i % 7 - 3) * 0.11);
          const buffers = [
            make(linearGemvParams("i8", 1, n, k), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
            make(input),
            make(packed),
            make(bias),
            make(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
            make(scale),
          ];
          const mapped = make(n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          let expected: Uint32Array<ArrayBuffer> | undefined;
          for (const variant of [defaultLinearGemvVariant(), selected]) {
            const pipeline = await device.createComputePipelineAsync({
              layout: "auto",
              compute: {
                module: device.createShaderModule({
                  code: linearGemvWgsl("i8", undefined, variant),
                }),
                entryPoint: "main",
              },
            });
            const bindings = device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindings);
            pass.dispatchWorkgroups(Math.ceil(n / variant.cols));
            pass.end();
            encoder.copyBufferToBuffer(buffers[4], 0, mapped, 0, n * 4);
            device.queue.submit([encoder.finish()]);
            await mapped.mapAsync(GPUMapMode.READ);
            const actual = new Uint32Array(mapped.getMappedRange().slice(0));
            mapped.unmap();
            expected ??= actual;
            assertEquals(actual, expected, `c${variant.cols} N=${n} K=${k}`);
            const values = new Float32Array(actual.buffer);
            for (let row = 0; row < n; row++) {
              let reference = 0;
              for (let col = 0; col < k; col++) {
                const index = row * k + col;
                const q = (packed[Math.floor(index / 4)] << (24 - (index % 4) * 8)) >> 24;
                reference += input[col] * Math.fround(q * scale[row]);
              }
              reference += bias[row];
              assert(Number.isFinite(values[row]));
              assert(
                Math.abs(values[row] - reference) <=
                  GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference),
                `CPU reference N=${n} K=${k} row=${row}`,
              );
            }
          }
        } finally {
          await device.queue.onSubmittedWorkDone();
          for (const buffer of owned) buffer.destroy();
          const error = await device.popErrorScope();
          assertEquals(error, null);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
