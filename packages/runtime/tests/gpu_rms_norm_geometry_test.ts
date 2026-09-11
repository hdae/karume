// RMS の 128 スレッド版は、256 版の下半分の加算順序を維持する。
import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  RMS_NORM_128_KEY,
  RMS_NORM_128_WGSL,
  RMS_NORM_KEY,
  RMS_NORM_WGSL,
  rmsNormParams,
} from "../src/kernels/rms-norm.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, refTensor } from "../src/reference/ops.ts";
import { createSession } from "../src/runtime/executor.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";
import { opTolerance } from "./helpers/op-tolerance.ts";

const direct = async (
  gpu: GpuContext,
  wgsl: string,
  x: Float32Array<ArrayBuffer>,
  weight: Float32Array<ArrayBuffer>,
): Promise<Float32Array<ArrayBuffer>> => {
  const device = gpu.device;
  const buffers: GPUBuffer[] = [];
  let values: Float32Array<ArrayBuffer>;
  let error: GPUError | null = null;
  device.pushErrorScope("validation");
  try {
    const buffer = (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer> | number,
      usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
    ): GPUBuffer => {
      const value = device.createBuffer({
        size: typeof data === "number" ? data : data.byteLength,
        usage,
      });
      buffers.push(value);
      if (typeof data !== "number") device.queue.writeBuffer(value, 0, data);
      return value;
    };
    const rows = x.length / weight.length;
    const params = buffer(rmsNormParams(rows, weight.length, 1e-6), U.UNIFORM | U.COPY_DST);
    const input = buffer(x), w = buffer(weight), output = buffer(x.byteLength);
    const staging = buffer(x.byteLength, U.MAP_READ | U.COPY_DST);
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: "main" },
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [params, input, w, output].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    // 行を再訪させる。行ごとに値が異なるので、scratch の寿命と grid-stride の誤りも値に出る。
    pass.dispatchWorkgroups(Math.min(rows, 2));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, x.byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE.READ);
    values = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
  } finally {
    device.queue.submit([]);
    await device.queue.onSubmittedWorkDone();
    error = await device.popErrorScope();
    for (const buffer of buffers) buffer.destroy();
  }
  if (error) throw new Error(error.message);
  return values;
};

const model = (rows: number, dim: number, weight: Float32Array<ArrayBuffer>): ArrayBuffer => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["rms_norm"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [rows, dim] }],
    outputs: ["y"],
    initializers: { w: { tensor: "w", storage: { dtype: "f32" } } },
    values: { w: { dtype: "f32", shape: [dim] }, y: { dtype: "f32", shape: [rows, dim] } },
    nodes: [{ op: "rms_norm", ins: ["x", "w"], outs: ["y"], attrs: { eps: 1e-6 } }],
  };
  return buildSafetensors(
    [{ name: "w", dtype: "F32", shape: [dim], data: f32Bytes([...weight]) }],
    { karume_ir: JSON.stringify(graph) },
  );
};
const bits = (data: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

Deno.test({
  name:
    "rms_norm の 128 スレッド版は端の幅でも従来版とビット一致し、128 を超える幅は従来版を選ぶ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const dim of [1, 3, 17, 64, 127, 128, 129, 257]) {
        const rows = 19;
        const x = Float32Array.from(
          { length: rows * dim },
          (_, i) => Math.sin(i * .731) * (1 + i % 11),
        );
        const weight = Float32Array.from({ length: dim }, (_, i) => Math.cos(i * .23) * 1.3);
        const expected = await direct(gpu, RMS_NORM_WGSL, x, weight);
        if (dim <= 128) {
          assertEquals(bits(await direct(gpu, RMS_NORM_128_WGSL, x, weight)), bits(expected));
        }
        const session = await createSession(gpu, openModel(model(rows, dim, weight)));
        try {
          const input = refTensor([rows, dim], x);
          const { y } = await session.run({ x: input });
          assert(y.dtype === "f32");
          assertEquals(bits(y.data), bits(expected), `dim=${dim}`);
          if (TIMESTAMP_QUERY_AVAILABLE) {
            assertEquals(session.diagnostics().lastRunTiming?.entries.map((e) => e.key), [
              dim <= 128 ? RMS_NORM_128_KEY : RMS_NORM_KEY,
            ]);
          }
          const reference = applyReferenceOp("rms_norm", [input, refTensor([dim], weight)], {
            eps: 1e-6,
          });
          const report = compareTensors(y, reference, opTolerance("rms_norm"));
          assert(report.pass, formatAllclose(report));
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "rms_norm の空レーン削減は ±0・大きい有限値・非有限値の伝播を維持する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const dim of [1, 17, 128]) {
        const x = new Float32Array(dim * 7);
        for (let i = 0; i < dim; i++) {
          x[i] = i % 2 === 0 ? -0 : 0;
          x[dim + i] = (i % 2 === 0 ? -1 : 1) * 1e-40;
          x[dim * 2 + i] = (i % 2 === 0 ? -1 : 1) * 1e20;
          x[dim * 3 + i] = 1 + i;
          x[dim * 4 + i] = 2 - i;
          x[dim * 5 + i] = 3 + i;
          x[dim * 6 + i] = Math.sin(i + .1);
        }
        x[dim * 3] = Infinity;
        x[dim * 4] = -Infinity;
        x[dim * 5] = NaN;
        const weight = Float32Array.from({ length: dim }, (_, i) => i % 2 === 0 ? -1 : .3);
        const expected = await direct(gpu, RMS_NORM_WGSL, x, weight);
        const actual = await direct(gpu, RMS_NORM_128_WGSL, x, weight);
        for (let i = 0; i < x.length; i++) {
          // NaN の payload は op の契約外。伝播する位置を検査し、それ以外は ±0 も含む u32 一致。
          if (Number.isNaN(expected[i])) assert(Number.isNaN(actual[i]), `dim=${dim} i=${i}`);
          else assertEquals(bits(actual)[i], bits(expected)[i], `dim=${dim} i=${i}`);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
