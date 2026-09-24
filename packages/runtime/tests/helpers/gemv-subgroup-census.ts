import { assert, assertEquals } from "@std/assert";
import type { CodecName } from "../../src/format/container/codecs.ts";
import type { GpuContext } from "../../src/gpu/device.ts";
import { createSessionFromContainer } from "../../src/runtime/executor.ts";
import type { TensorInput } from "./container-write.ts";
import { type DeclarationJson, f32Bytes, GRAPH_NAME, openModelBytes } from "./model-fixture.ts";

/** 格納の呼び名 → codec 台帳の登録名（i2 は per-channel の `int2-off`）。 */
const CODEC: Readonly<Record<"i2" | "i4" | "i8", CodecName>> = {
  i2: "int2-off",
  i4: "int4-sym-g",
  i8: "int8-sym",
};

export const checkGemvSubgroupCensus = async (gpu: GpuContext): Promise<void> => {
  for (
    const shape of [
      { storage: "i2", bits: 2, n: 12288, k: 1536, group: undefined },
      { storage: "i4", bits: 4, n: 256, k: 1536, group: 32 },
      { storage: "i4", bits: 4, n: 256, k: 1536, group: 512 },
      { storage: "i4", bits: 4, n: 1536, k: 2048, group: 2048 },
      { storage: "i4", bits: 4, n: 1536, k: 4096, group: 4096 },
      { storage: "i8", bits: 8, n: 256, k: 1536, group: undefined },
      { storage: "i8", bits: 8, n: 36, k: 1536, group: undefined },
    ] as const
  ) {
    const { storage, n, k, bits, group } = shape;
    const w = Uint8Array.from(
      { length: n * k * bits / 8 },
      (_, i) => (Math.imul(i + 1, 0x9e3779b9) >>> 24),
    );
    // per-channel（i2 / i8）は groups = 1、group 量子化（i4）は 行長 / group。
    const groups = group === undefined ? 1 : k / group;
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: w,
        encoding: {
          codec: CODEC[storage],
          groupSize: group ?? k,
          scale: {
            dtype: "f32",
            bytes: f32Bytes(
              Array.from({ length: n * groups }, (_, i) => (i % 17 + 1) * 0.00017),
            ),
          },
        },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: f32Bytes(Array.from({ length: n }, (_, i) => (i % 7 - 3) * 0.11)),
        encoding: { codec: "f32" },
      },
    ];
    let expected: Uint32Array<ArrayBuffer> | undefined;
    for (const m of [1, 4, 8, 9]) {
      for (const parallel of m === 1 ? [false, true] : [true]) {
        const graph: DeclarationJson = {
          format: "karume-ir",
          version: 2,
          requires: { ops: ["linear"] },
          symbols: [],
          inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
          outputs: ["y"],
          initializers: { w: {}, b: {} },
          values: {
            w: { dtype: "f32", shape: [n, k] },
            b: { dtype: "f32", shape: [n] },
            y: { dtype: "f32", shape: [m, n] },
          },
          nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
        };
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(graph, tensors),
          GRAPH_NAME,
          parallel ? { linearGemvReduce: "parallel-subgroup32" } : {},
        );
        try {
          const input = Float32Array.from(
            { length: m * k },
            (_, i) => Math.sin(i * 0.037) * 0.75,
          );
          const output = (await session.run({ x: { dtype: "f32", shape: [m, k], data: input } })).y;
          assertEquals(output.shape, [m, n]);
          for (const value of output.data) assert(Number.isFinite(value));
          const keys = session.diagnostics().lastRunTiming?.entries.map((entry) => entry.key);
          assert(keys !== undefined && keys.length > 0);
          assertEquals(
            keys.some((key) =>
              key.startsWith("linear_gemv_parallel") && key.endsWith(":subgroup32")
            ),
            parallel && m <= 8 && n !== 36,
            `${storage} M${m} ${keys}`,
          );
          if (parallel && m <= 8 && n !== 36) {
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
};
