// 公式 Torch CPU の期待ビット列。表方式と独立した CPU 参照の両方を同じ oracle に突き合わせる。
import { assert, assertEquals, assertThrows } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { parseSafetensors } from "../src/format/safetensors.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import { STATIC_QUANTIZE_WGSL, staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import { assertNodeContract } from "../src/ops.ts";
import { referenceStaticQuantize, refTensor } from "../src/reference/ops.ts";
import { createSession } from "../src/runtime/executor.ts";
import { buildSafetensors, type GraphJson } from "./helpers/format.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const bits = (data: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

type Case = { scale: number; x: Float32Array<ArrayBuffer>; y: Uint32Array<ArrayBuffer> };
const cases = async (): Promise<readonly Case[]> => {
  const bytes = await Deno.readFile(
    new URL("./fixtures/static-quantize-oracle.safetensors", import.meta.url),
  );
  const file = parseSafetensors(bytes.buffer);
  const scales: unknown = JSON.parse(file.metadata.get("scale_bits") ?? "null");
  assert(
    Array.isArray(scales) &&
      scales.every((s: unknown) => typeof s === "number" && Number.isInteger(s)),
  );
  const read = (name: string): Float32Array<ArrayBuffer> => {
    const tensor = file.tensors.get(name);
    assert(tensor?.dtype === "F32");
    return new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4);
  };
  return scales.map((scale: number, i: number) => ({
    scale: new Float32Array(Uint32Array.of(scale).buffer)[0],
    x: read(`x${i}`),
    y: bits(read(`y${i}`)),
  }));
};

const model = (shape: readonly number[], scale: number): ArrayBuffer => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["static_quantize"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [...shape] }],
    outputs: ["y"],
    initializers: {},
    values: { y: { dtype: "f32", shape: [...shape] } },
    nodes: [{ op: "static_quantize", ins: ["x"], outs: ["y"], attrs: { scale } }],
  };
  return buildSafetensors([], { karume_ir: JSON.stringify(graph) });
};

Deno.test("static_quantize の独立 CPU 参照は公式の境界・特殊値 24,416 入力とビット一致する", async () => {
  const fixture = await cases();
  assertEquals(fixture.length, 30);
  assertEquals(fixture.reduce((sum, c) => sum + c.x.length, 0), 24416);
  for (const c of fixture) {
    const before = bits(c.x).slice();
    const result = referenceStaticQuantize(refTensor([c.x.length], c.x), { scale: c.scale });
    assert(result.dtype === "f32");
    assertEquals(bits(result.data), c.y, `scale=${c.scale}`);
    assertEquals(bits(c.x), before, "入力を変更しない");
  }
});

Deno.test("static_quantize は scale を暗黙に丸めず不正属性・dtype を拒否する", () => {
  for (const scale of [-1, NaN, Infinity, -Infinity, 0.1, 1e-50, 1e39, undefined, true, "1"]) {
    assertThrows(() =>
      assertNodeContract({
        op: "static_quantize",
        ins: ["x"],
        outs: ["y"],
        attrs: { scale },
        states: {},
      }, "test")
    );
  }
  assertThrows(() => referenceStaticQuantize(refTensor([1], Int32Array.of(1)), { scale: 1 }));
  for (const scale of [-1, NaN, Infinity, 0.1, 1e-50, 1e39]) {
    assertThrows(() => staticQuantizeParams(1, scale));
  }
  for (const count of [-1, 0.5, 2 ** 32]) assertThrows(() => staticQuantizeParams(count, 1));
  const payload = Uint32Array.of(0x80000000, 0x7fa00001, 0xffa00002);
  const identity = referenceStaticQuantize(refTensor([3], new Float32Array(payload.buffer)), {
    scale: -0,
  });
  assert(identity.dtype === "f32");
  assertEquals(bits(identity.data), payload);
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
        const session = await createSession(gpu, openModel(model([c.x.length], c.scale)));
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
        const session = await createSession(gpu, openModel(model(shape, 0.5)));
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
