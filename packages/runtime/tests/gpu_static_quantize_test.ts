// static_quantize の実 GPU 突合（CPU 参照側の門は static_quantize_test.ts）。
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import { STATIC_QUANTIZE_WGSL, staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import { referenceStaticQuantize, refTensor } from "../src/reference/ops.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { type DeclarationJson, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { bits, type Case, cases } from "./helpers/static-quantize-oracle.ts";

const declaration = (shape: readonly number[], scale: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["static_quantize"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [...shape] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [...shape] } },
  nodes: [{ op: "static_quantize", ins: ["x"], outs: ["y"], attrs: { scale } }],
});

const direct = async (gpu: GpuContext, c: Case): Promise<Uint32Array<ArrayBuffer>> => {
  const d = gpu.device, buffers: GPUBuffer[] = [];
  d.pushErrorScope("validation");
  let result: Uint32Array<ArrayBuffer>;
  let error: GPUError | null = null;
  try {
    const upload = (data: Uint32Array<ArrayBuffer>, usage: number): GPUBuffer => {
      const b = d.createBuffer({ size: data.byteLength, usage: usage | U.COPY_DST });
      buffers.push(b);
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    const params = upload(staticQuantizeParams(c.x.length, c.scale), U.UNIFORM);
    const x = upload(bits(c.x), U.STORAGE);
    const y = d.createBuffer({ size: c.x.byteLength, usage: U.STORAGE | U.COPY_SRC });
    buffers.push(y);
    const staging = d.createBuffer({ size: c.x.byteLength, usage: U.COPY_DST | U.MAP_READ });
    buffers.push(staging);
    const p = await d.createComputePipelineAsync({
      layout: "auto",
      compute: { module: d.createShaderModule({ code: STATIC_QUANTIZE_WGSL }), entryPoint: "main" },
    });
    const bg = d.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [params, x, y].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(y, 0, staging, 0, c.x.byteLength);
    d.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE.READ);
    result = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
  } finally {
    await d.queue.onSubmittedWorkDone();
    error = await d.popErrorScope();
    for (const b of buffers) b.destroy();
  }
  if (error) throw Error(error.message);
  return result;
};

Deno.test({
  name: "static_quantize は Session と grid-stride の複数巡回で公式 CPU とビット一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const c of await cases()) {
        assertEquals(await direct(gpu, c), c.y, `direct scale=${c.scale}`);
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(declaration([c.x.length], c.scale), []),
          GRAPH_NAME,
        );
        try {
          for (let repeat = 0; repeat < 2; repeat++) {
            const { y } = await session.run({ x: refTensor([c.x.length], c.x) });
            assert(y.dtype === "f32");
            assertEquals(bits(y.data), c.y, `Session scale=${c.scale}`);
          }
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
  name: "static_quantize はスカラ・空軸・多次元の shape を保つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of [[], [0, 16], [2, 3, 17]]) {
        const x = Float32Array.from(
          { length: shape.reduce((n, d) => n * d, 1) },
          (_, i) => (i - 9) * 0.25,
        );
        const input = refTensor(shape, x),
          expected = referenceStaticQuantize(input, { scale: 0.5 });
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(declaration(shape, 0.5), []),
          GRAPH_NAME,
        );
        try {
          const { y } = await session.run({ x: input });
          assert(y.dtype === "f32" && expected.dtype === "f32");
          assertEquals(y.shape, shape);
          assertEquals(bits(y.data), bits(expected.data));
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
