import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSession } from "../src/runtime/executor.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

describe({
  name: "GEMV並列加算の適用箇所（実GPU）",
  ignore: !TIMESTAMP_QUERY_AVAILABLE,
  fn: () => {
    it("明示指定の対象だけに適用し、M1/4/8で全出力列のu32一致を保つ", async () => {
      const gpu = await acquireGpu({ gpuTiming: true });
      try {
        for (
          const shape of [
            { storage: "i2", dtype: "I2", bits: 2, n: 12288, k: 1536, group: undefined },
            { storage: "i4", dtype: "I4", bits: 4, n: 256, k: 1536, group: 32 },
            { storage: "i4", dtype: "I4", bits: 4, n: 256, k: 1536, group: 512 },
            { storage: "i4", dtype: "I4", bits: 4, n: 1536, k: 2048, group: 2048 },
            { storage: "i4", dtype: "I4", bits: 4, n: 1536, k: 4096, group: 4096 },
            { storage: "i8", dtype: "I8", bits: 8, n: 256, k: 1536, group: undefined },
          ] as const
        ) {
          const { storage, dtype, n, k, bits, group } = shape;
          const w = Uint8Array.from(
            { length: n * k * bits / 8 },
            (_, i) => (Math.imul(i + 1, 0x9e3779b9) >>> 24),
          );
          const scaleShape = group === undefined ? [n, 1] : [n, k / group];
          let expected: Uint32Array<ArrayBuffer> | undefined;
          for (const m of [1, 4, 8, 9]) {
            for (const parallel of m === 1 ? [false, true] : [true]) {
              const graph: GraphJson = {
                format: "karume-ir",
                version: 1,
                requires: { ops: ["linear"] },
                symbols: [],
                inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
                outputs: ["y"],
                initializers: {
                  w: {
                    tensor: "w",
                    storage: {
                      dtype: storage,
                      scale: "s",
                      ...(group === undefined ? {} : { group_size: group }),
                    },
                  },
                  b: { tensor: "b", storage: { dtype: "f32" } },
                },
                values: {
                  w: { dtype: "f32", shape: [n, k] },
                  b: { dtype: "f32", shape: [n] },
                  y: { dtype: "f32", shape: [m, n] },
                },
                nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
              };
              const model = buildSafetensors([
                { name: "w", dtype, shape: [n, k], data: w },
                {
                  name: "s",
                  dtype: "F32",
                  shape: scaleShape,
                  data: f32Bytes(
                    Array.from({
                      length: scaleShape.reduce((a, b) => a * b, 1),
                    }, (_, i) => (i % 17 + 1) * 0.00017),
                  ),
                },
                {
                  name: "b",
                  dtype: "F32",
                  shape: [n],
                  data: f32Bytes(Array.from({ length: n }, (_, i) => (i % 7 - 3) * 0.11)),
                },
              ], { karume_ir: JSON.stringify(graph) });
              const session = await createSession(
                gpu,
                openModel(model),
                parallel ? { linearGemvReduce: "parallel" } : {},
              );
              try {
                const input = Float32Array.from(
                  { length: m * k },
                  (_, i) => Math.sin(i * 0.037) * 0.75,
                );
                const output =
                  (await session.run({ x: { dtype: "f32", shape: [m, k], data: input } })).y;
                assertEquals(output.shape, [m, n]);
                for (const value of output.data) assert(Number.isFinite(value));
                const keys = session.diagnostics().lastRunTiming?.entries.map((entry) => entry.key);
                assert(keys !== undefined && keys.length > 0);
                assertEquals(
                  keys.some((key) => key.startsWith("linear_gemv_parallel")),
                  parallel && m <= 8,
                  `${storage} M${m} ${keys}`,
                );
                if (parallel && m <= 8) {
                  const first = new Uint32Array(output.data.buffer, output.data.byteOffset, n)
                    .slice();
                  expected ??= first;
                  assertEquals(first, expected);
                }
              } finally {
                await session.dispose();
              }
            }
          }
        }
      } finally {
        gpu.destroy();
      }
    });
  },
});
