// INT2 の packed 実行・CPU展開・借用を、同じ復元値の f32 経路と突き合わせる。
import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSession, createSessionFromShards, type Tensor } from "../src/runtime/executor.ts";
import { estimateSessionMemory } from "../src/runtime/estimate.ts";
import { buildSafetensors, f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";
import { shardStream } from "./helpers/shard-fixture.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

const weight = (
  n: number,
  k: number,
): {
  bytes: Uint8Array<ArrayBuffer>;
  scale: Float32Array<ArrayBuffer>;
  values: Float32Array<ArrayBuffer>;
} => {
  const bytes = new Uint8Array(n * k / 4),
    scale = Float32Array.from({ length: n }, (_, r) => Math.fround(0.13 + (r % 7) * 0.031));
  const values = new Float32Array(n * k);
  for (let i = 0; i < values.length; i++) {
    const u = (i ^ Math.floor(i / 7) ^ Math.floor(i / k)) & 3;
    bytes[i >> 2] |= u << ((i & 3) * 2);
    values[i] = Math.fround((u - 2) * scale[Math.floor(i / k)]);
  }
  return { bytes, scale, values };
};
const linearGraph = (
  m: number,
  n: number,
  k: number,
  storage: "i2" | "f32",
  expanded = false,
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
  outputs: expanded ? ["y", "w"] : ["y"],
  initializers: {
    w: {
      tensor: "w",
      storage: storage === "i2" ? { dtype: storage, scale: "scale" } : { dtype: storage },
    },
    b: { tensor: "b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    y: { dtype: "f32", shape: [m, n] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});
const makeModel = (
  graph: GraphJson,
  n: number,
  k: number,
  w: ReturnType<typeof weight>,
): ArrayBuffer => {
  const tensors: TensorSpec[] = [];
  if (graph.initializers.b) {
    tensors.push({
      name: "b",
      dtype: "F32",
      shape: [n],
      data: f32Bytes(Array.from({ length: n }, (_, i) => (i % 5 - 2) * 0.17)),
    });
  }
  if (graph.initializers.w.shared === undefined) {
    if (graph.initializers.w.storage.dtype === "i2") {
      tensors.push({
        name: "scale",
        dtype: "F32",
        shape: [n, 1],
        data: new Uint8Array(w.scale.buffer),
      });
      tensors.push({ name: "w", dtype: "I2", shape: [n, k], data: w.bytes });
    } else {tensors.push({
        name: "w",
        dtype: "F32",
        shape: [n, k],
        data: new Uint8Array(w.values.buffer),
      });}
  }
  return buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });
};
const bits = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 4);

Deno.test({
  name: "INT2 linear は GEMV・行ブロック・prefill GEMM・端列で f32 展開とビット一致する",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const [m, n, k] of [[1, 68, 192], [3, 68, 192], [8, 8192, 64], [64, 68, 64], [65, 68, 64], [
          65,
          68,
          48,
        ], [17, 19, 48]]
      ) {
        const w = weight(n, k),
          x = {
            dtype: "f32",
            shape: [m, k],
            data: Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.37) * 0.3),
          } satisfies Tensor;
        const model = openModel(makeModel(linearGraph(m, n, k, "i2"), n, k, w));
        const session = await createSession(gpu, model);
        const baseline = await createSession(
          gpu,
          openModel(makeModel(linearGraph(m, n, k, "f32"), n, k, w)),
        );
        try {
          const actual = (await session.run({ x })).y, expected = (await baseline.run({ x })).y;
          assertEquals(bits(actual), bits(expected), `${m},${n},${k}`);
          const diagnostics = session.diagnostics();
          assertEquals(
            diagnostics.storage.residentCompressedBytes,
            w.bytes.byteLength + w.scale.byteLength,
          );
          assertEquals(diagnostics.storage.hostExpandedBytes, 0);
          assertEquals(
            estimateSessionMemory(model).resident.weights.compressedBytes,
            w.bytes.byteLength + w.scale.byteLength,
          );
          if (TIMING_ACQUIRE_OPTIONS.gpuTiming) {
            const keys = diagnostics.lastRunTiming?.entries.map((e) => e.key) ?? [];
            assert(keys.some((key) => key.includes(":wi2")));
            assertEquals(
              keys.some((key) => key.startsWith("linear_gemv:")),
              m <= 64 && n % 4 === 0 && k % 64 === 0,
            );
          }
        } finally {
          await baseline.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "INT2 embedding は全2bit位置を復元し同じ表を linear へ借用できる",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const n = 7, k = 64, w = weight(n, k);
    const graph = linearGraph(1, n, k, "i2");
    delete graph.initializers.b;
    delete graph.values.b;
    graph.requires.ops = ["embedding"];
    graph.inputs = [{ name: "ids", dtype: "i32", shape: [3] }];
    graph.values.y.shape = [3, k];
    graph.nodes = [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }];
    try {
      const owner = await createSession(gpu, openModel(makeModel(graph, n, k, w)));
      try {
        const y =
          (await owner.run({ ids: { dtype: "i32", shape: [3], data: Int32Array.of(6, 0, 3) } })).y;
        const expected = new Float32Array(3 * k);
        [6, 0, 3].forEach((row, i) =>
          expected.set(w.values.subarray(row * k, (row + 1) * k), i * k)
        );
        assertEquals(bits(y), new Uint32Array(expected.buffer));
        const borrowed = linearGraph(1, n, k, "i2");
        borrowed.initializers.w = { shared: { tensor: "w" }, storage: { dtype: "i2" } };
        const borrower = await createSession(gpu, openModel(makeModel(borrowed, n, k, w)), {
          sharedWeights: { w: owner.exportWeight("w") },
        });
        const baseline = await createSession(
          gpu,
          openModel(makeModel(linearGraph(1, n, k, "f32"), n, k, w)),
        );
        try {
          const x = {
            dtype: "f32",
            shape: [1, k],
            data: Float32Array.from({ length: k }, (_, i) => Math.cos(i) * 0.1),
          } satisfies Tensor;
          assertEquals(bits((await borrower.run({ x })).y), bits((await baseline.run({ x })).y));
          assertEquals(borrower.diagnostics().storage.residentCompressedBytes, 0);
          await assertRejects(() => owner.dispose());
        } finally {
          await baseline.dispose();
          await borrower.dispose();
        }
      } finally {
        await owner.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "INT2 の一般消費はCPUで展開し非対応の計算指定を拒否する",
  ignore: !GPU_AVAILABLE || !SHADER_F16_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    const n = 4, k = 64, w = weight(n, k);
    try {
      const expanded = await createSession(
        gpu,
        openModel(makeModel(linearGraph(1, n, k, "i2", true), n, k, w)),
      );
      try {
        const result = await expanded.run({
          x: { dtype: "f32", shape: [1, k], data: new Float32Array(k) },
        });
        assertEquals(bits(result.w), new Uint32Array(w.values.buffer));
        assertEquals(expanded.diagnostics().storage.hostExpandedBytes, w.values.byteLength);
        assertEquals(expanded.diagnostics().storage.residentCompressedBytes, 0);
      } finally {
        await expanded.dispose();
      }
      for (const linearCompute of ["a8", "f16"] as const) {
        const session = await createSession(
          gpu,
          openModel(makeModel(linearGraph(1, n, k, "i2"), n, k, w)),
          { linearCompute },
        );
        try {
          await assertRejects(
            () => session.run({ x: { dtype: "f32", shape: [1, k], data: new Float32Array(k) } }),
            Error,
            "i2",
          );
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
  name: "INT2 の行分割ロードは全量ロードと一致する（packed / CPU展開）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const n = 7, k = 64, w = weight(n, k);
    try {
      for (const expanded of [false, true]) {
        const graph = linearGraph(1, n, k, "i2", expanded);
        const shards = [
          buildSafetensors([], { karume_ir: JSON.stringify(graph) }),
          buildSafetensors([
            {
              name: "b",
              dtype: "F32",
              shape: [n],
              data: f32Bytes(Array.from({ length: n }, (_, i) => (i % 5 - 2) * 0.17)),
            },
            { name: "scale", dtype: "F32", shape: [n, 1], data: new Uint8Array(w.scale.buffer) },
            {
              name: "w#00001-of-00002",
              dtype: "I2",
              shape: [2, k],
              data: w.bytes.subarray(0, 2 * k / 4),
            },
          ]),
          buildSafetensors([{
            name: "w#00002-of-00002",
            dtype: "I2",
            shape: [n - 2, k],
            data: w.bytes.subarray(2 * k / 4),
          }]),
        ];
        const full = await createSession(gpu, openModel(makeModel(graph, n, k, w)));
        const partial = await createSessionFromShards(gpu, shardStream(shards));
        try {
          const x = {
            dtype: "f32",
            shape: [1, k],
            data: Float32Array.from({ length: k }, (_, i) => Math.sin(i) * 0.4),
          } satisfies Tensor;
          const expected = await full.run({ x }), actual = await partial.run({ x });
          for (const name of graph.outputs) assertEquals(bits(actual[name]), bits(expected[name]));
          assertEquals(partial.diagnostics().storage, full.diagnostics().storage);
        } finally {
          await partial.dispose();
          await full.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "INT2 行ブロックは長い縮約と端の行でも M=1 の積和とビット一致する",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // r1 / r2 / r4、語の先読みと残り、長い K を跨ぐ。scale と bias は2冪に限定しない。
      for (const [m, n, k] of [[3, 68, 320], [3, 8192, 320], [7, 8192, 576], [2, 68, 12288]]) {
        const w = weight(n, k);
        const rows = await createSession(
          gpu,
          openModel(makeModel(linearGraph(m, n, k, "i2"), n, k, w)),
        );
        const single = await createSession(
          gpu,
          openModel(makeModel(linearGraph(1, n, k, "i2"), n, k, w)),
        );
        try {
          const data = Float32Array.from(
            { length: m * k },
            (_, i) => Math.sin(i * 0.37) * (i % 7 === 0 ? 19.3 : 0.073),
          );
          const actual = bits((await rows.run({ x: { dtype: "f32", shape: [m, k], data } })).y);
          for (let row = 0; row < m; row++) {
            const expected = bits(
              (await single.run({
                x: { dtype: "f32", shape: [1, k], data: data.slice(row * k, (row + 1) * k) },
              })).y,
            );
            assertEquals(
              actual.subarray(row * n, (row + 1) * n),
              expected,
              `${m},${n},${k} row=${row}`,
            );
          }
        } finally {
          await single.dispose();
          await rows.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
