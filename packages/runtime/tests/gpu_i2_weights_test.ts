// Compares INT2's packed execution, CPU expansion and borrowing against the f32 path with the same
// decoded values.
import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { TensorInput, WriteOptions } from "./helpers/container-write.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const weight = (
  n: number,
  k: number,
): {
  bytes: Uint8Array<ArrayBuffer>;
  scale: Float32Array<ArrayBuffer>;
  values: Float32Array<ArrayBuffer>;
} => {
  const bytes = new Uint8Array((n * k) / 4),
    scale = Float32Array.from({ length: n }, (_, r) => Math.fround(0.13 + (r % 7) * 0.031));
  const values = new Float32Array(n * k);
  for (let i = 0; i < values.length; i++) {
    const u = (i ^ Math.floor(i / 7) ^ Math.floor(i / k)) & 3;
    bytes[i >> 2] |= u << ((i & 3) * 2);
    values[i] = Math.fround((u - 2) * scale[Math.floor(i / k)]);
  }
  return { bytes, scale, values };
};
const linearDeclaration = (m: number, n: number, k: number, expanded = false): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
  outputs: expanded ? ["y", "w"] : ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    y: { dtype: "f32", shape: [m, n] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});
/**
 * Builds the supply from the declaration + decoded values (`int2-off` is per-channel, so groupSize
 * = row length k).
 */
const weightTensors = (
  declaration: DeclarationJson,
  n: number,
  k: number,
  storage: "i2" | "f32",
  w: ReturnType<typeof weight>,
): TensorInput[] => {
  const tensors: TensorInput[] = [];
  if (declaration.initializers.b !== undefined) {
    tensors.push({
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes(Array.from({ length: n }, (_, i) => ((i % 5) - 2) * 0.17)),
      encoding: { codec: "f32" },
    });
  }
  if (declaration.initializers.w.shared === undefined) {
    tensors.push(
      storage === "i2"
        ? {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: w.bytes,
            encoding: {
              codec: "int2-off",
              groupSize: k,
              scale: { bytes: new Uint8Array(w.scale.buffer), dtype: "f32" },
            },
          }
        : {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: new Uint8Array(w.values.buffer),
            encoding: { codec: "f32" },
          },
    );
  }
  return tensors;
};
const openLinear = (
  declaration: DeclarationJson,
  n: number,
  k: number,
  storage: "i2" | "f32",
  w: ReturnType<typeof weight>,
  options?: WriteOptions,
): Promise<OpenedContainer> =>
  openModelBytes(declaration, weightTensors(declaration, n, k, storage, w), options);
const bits = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 4);

test.skipIf(!GPU_AVAILABLE)(
  "INT2 linear matches the f32 expansion bit for bit in GEMV, row blocks, prefill GEMM and edge columns",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [m, n, k] of [
        [1, 68, 192],
        [3, 68, 192],
        [8, 8192, 64],
        [64, 68, 64],
        [65, 68, 64],
        [65, 68, 48],
        [17, 19, 48],
      ]) {
        const w = weight(n, k),
          x = {
            dtype: "f32",
            shape: [m, k],
            data: Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.37) * 0.3),
          } satisfies Tensor;
        const prepared = prepareContainer(
          await openLinear(linearDeclaration(m, n, k), n, k, "i2", w),
          GRAPH_NAME,
        );
        const session = await prepared.createContainerSession(gpu);
        const baseline = await createSessionFromContainer(
          gpu,
          await openLinear(linearDeclaration(m, n, k), n, k, "f32", w),
          GRAPH_NAME,
        );
        try {
          const actual = (await session.run({ x })).y,
            expected = (await baseline.run({ x })).y;
          assertEquals(bits(actual), bits(expected), `${m},${n},${k}`);
          const diagnostics = session.diagnostics();
          assertEquals(
            diagnostics.storage.residentCompressedBytes,
            w.bytes.byteLength + w.scale.byteLength,
          );
          assertEquals(diagnostics.storage.hostExpandedBytes, 0);
          assertEquals(
            prepared.estimate().resident.weights.compressedBytes,
            w.bytes.byteLength + w.scale.byteLength,
          );
          const keys = requireCensus(diagnostics.lastRunPipelines, "i2 linear").map(
            (row) => row.key,
          );
          assert(keys.some((key) => key.includes(":wi2")));
          assertEquals(
            keys.some((key) => key.startsWith("linear_gemv:")),
            m <= 64 && n % 4 === 0 && k % 64 === 0,
          );
        } finally {
          await baseline.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "INT2 embedding decodes every 2-bit position and can lend the same table to linear",
  async () => {
    const gpu = await acquireGpu();
    const n = 7,
      k = 64,
      w = weight(n, k);
    const graph = linearDeclaration(1, n, k);
    delete graph.initializers.b;
    delete graph.values.b;
    graph.requires.ops = ["embedding"];
    graph.inputs = [{ name: "ids", dtype: "i32", shape: [3] }];
    graph.values.y.shape = [3, k];
    graph.nodes = [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }];
    try {
      const owner = await createSessionFromContainer(
        gpu,
        await openLinear(graph, n, k, "i2", w),
        GRAPH_NAME,
      );
      try {
        const y = (
          await owner.run({ ids: { dtype: "i32", shape: [3], data: Int32Array.of(6, 0, 3) } })
        ).y;
        const expected = new Float32Array(3 * k);
        [6, 0, 3].forEach((row, i) =>
          expected.set(w.values.subarray(row * k, (row + 1) * k), i * k),
        );
        assertEquals(bits(y), new Uint32Array(expected.buffer));
        const borrowed = linearDeclaration(1, n, k);
        borrowed.initializers.w = { shared: true };
        const borrower = await createSessionFromContainer(
          gpu,
          await openLinear(borrowed, n, k, "i2", w),
          GRAPH_NAME,
          { sharedWeights: { w: owner.exportWeight("w") } },
        );
        const baseline = await createSessionFromContainer(
          gpu,
          await openLinear(linearDeclaration(1, n, k), n, k, "f32", w),
          GRAPH_NAME,
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
);

test.skipIf(!GPU_AVAILABLE || !SHADER_F16_AVAILABLE)(
  "general INT2 consumers expand on the CPU, and unsupported compute options are rejected",
  async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    const n = 4,
      k = 64,
      w = weight(n, k);
    try {
      const expanded = await createSessionFromContainer(
        gpu,
        await openLinear(linearDeclaration(1, n, k, true), n, k, "i2", w),
        GRAPH_NAME,
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
        const session = await createSessionFromContainer(
          gpu,
          await openLinear(linearDeclaration(1, n, k), n, k, "i2", w),
          GRAPH_NAME,
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
);

test.skipIf(!GPU_AVAILABLE)(
  "a row-split INT2 load matches a full load (packed / CPU expansion)",
  async () => {
    const gpu = await acquireGpu();
    const n = 7,
      k = 64,
      w = weight(n, k);
    try {
      for (const expanded of [false, true]) {
        const graph = linearDeclaration(1, n, k, expanded);
        // Lower the block limit to 4 times one row (16 B) so the writer splits w into pieces.
        const split = await openLinear(graph, n, k, "i2", w, { blockBytes: 64 });
        assertEquals(
          split.graphs[GRAPH_NAME].supplies.get("w")?.blocks.length,
          2,
          "no piece split happened (the detector would spin idle)",
        );
        const full = await createSessionFromContainer(
          gpu,
          await openLinear(graph, n, k, "i2", w),
          GRAPH_NAME,
        );
        const partial = await createSessionFromContainer(gpu, split, GRAPH_NAME);
        try {
          const x = {
            dtype: "f32",
            shape: [1, k],
            data: Float32Array.from({ length: k }, (_, i) => Math.sin(i) * 0.4),
          } satisfies Tensor;
          const expected = await full.run({ x }),
            actual = await partial.run({ x });
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
);

test.skipIf(!GPU_AVAILABLE)(
  "INT2 row blocks match the M=1 multiply-accumulate bit for bit even with long reductions and edge rows",
  async () => {
    const gpu = await acquireGpu();
    try {
      // Crosses r1 / r2 / r4, word prefetch and the remainder, and long K. scale and bias are not
      // limited to powers of 2.
      for (const [m, n, k] of [
        [3, 68, 320],
        [3, 8192, 320],
        [7, 8192, 576],
        [2, 68, 12288],
      ]) {
        const w = weight(n, k);
        const rows = await createSessionFromContainer(
          gpu,
          await openLinear(linearDeclaration(m, n, k), n, k, "i2", w),
          GRAPH_NAME,
        );
        const single = await createSessionFromContainer(
          gpu,
          await openLinear(linearDeclaration(1, n, k), n, k, "i2", w),
          GRAPH_NAME,
        );
        try {
          const data = Float32Array.from(
            { length: m * k },
            (_, i) => Math.sin(i * 0.37) * (i % 7 === 0 ? 19.3 : 0.073),
          );
          const actual = bits((await rows.run({ x: { dtype: "f32", shape: [m, k], data } })).y);
          for (let row = 0; row < m; row++) {
            const expected = bits(
              (
                await single.run({
                  x: { dtype: "f32", shape: [1, k], data: data.slice(row * k, (row + 1) * k) },
                })
              ).y,
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
);
