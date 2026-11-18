import { assert, assertEquals } from "@std/assert";
import {
  checkGemvParallelRows,
  type CheckPart,
  GEMV_PARALLEL_ROWS_CASES,
} from "@karume/kernel-checks";
import { describe, it } from "vite-plus/test";
import type { CodecName } from "../src/format/container/codecs.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import {
  linearGemvParallelKey,
  type LinearGemvParallelLanes,
  type LinearGemvParallelRows,
  linearGemvParallelRowsKey,
  type LinearGemvParallelShape,
  linearGemvParallelStagedKey,
} from "../src/kernels/linear-gemv.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/** The cases of a part that fail the criterion whose name starts with `name`. */
const failing = (part: CheckPart, name: string): number =>
  part.cases.filter((one) =>
    one.criteria.some((criterion) => criterion.name.startsWith(name) && criterion.failures > 0),
  ).length;

describe.skipIf(!GPU_AVAILABLE)("row-blocked parallel GEMV (real GPU)", () => {
  it("writes the parallel GEMV's u32 words at the same lanes for every storage, lane count, block height and row count", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkGemvParallelRows(gpu);
      assertEquals(part.cases.length, GEMV_PARALLEL_ROWS_CASES["gemv-parallel-rows"]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("is caught reading a row's activations from another row (fault: row 1 staged from row 0)", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkGemvParallelRows(gpu, {
        mutate: (wgsl) => wgsl.replace("staged_quad1 = x[xr1 + v];", "staged_quad1 = x[xr0 + v];"),
      });
      // Every case computes a second row from the first row's activations: no case is identical.
      assertEquals(
        failing(part, "row-blocked vs parallel"),
        GEMV_PARALLEL_ROWS_CASES["gemv-parallel-rows"],
      );
      // Row 0 itself is untouched by the fault.
      assertEquals(failing(part, "row 0 vs the parallel GEMV's row at M = 1"), 0);
    } finally {
      gpu.destroy();
    }
  });

  it("is caught storing a row past M (fault: the guard of row 1 removed)", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkGemvParallelRows(gpu, {
        mutate: (wgsl) => wgsl.replace("if (row0 + 1u < dims.m) {", "if (true) {"),
      });
      // Only a row count that leaves the last block one row has a second row to spill.
      const spilled = failing(part, "row-blocked: rows past M left unwritten");
      assert(spilled > 0, "no case caught the spilled row");
      for (const one of part.cases) {
        const [, rows, m] = /R(\d+) M(\d+)$/.exec(one.label) ?? [];
        const reaches = Number(m) % Number(rows) === 1;
        const failed = one.criteria.some(
          (criterion) =>
            criterion.name.startsWith("row-blocked: rows past M") && criterion.failures > 0,
        );
        assertEquals(failed, reaches, one.label);
      }
    } finally {
      gpu.destroy();
    }
  });
});

type Storage = "i2" | "i4" | "i8";

const CODEC: Readonly<Record<Storage, CodecName>> = {
  i2: "int2-off",
  i4: "int4-sym-g",
  i8: "int8-sym",
};

/** A shape of no model, so the default table's rows play no part. */
const SHAPE = { storage: "i4", bits: 4, n: 1280, k: 1024, group: 32 } as const;

/** A table whose GEMV choices are the default's plus one row of {@link SHAPE}. */
const tableWith = (
  id: string,
  lanes: LinearGemvParallelLanes,
  flags: { readonly staged?: true; readonly rowBlock?: LinearGemvParallelRows },
): GeometryProfile => {
  const row: LinearGemvParallelShape = {
    storage: SHAPE.storage,
    n: SHAPE.n,
    k: SHAPE.k,
    group: SHAPE.group,
    lanes,
    packedActivations: false,
    ...(flags.staged === undefined ? {} : { stagedActivations: true }),
    ...(flags.rowBlock === undefined ? {} : { rowBlock: flags.rowBlock }),
  };
  const defaults = DEFAULT_GEOMETRY_PROFILE.linearGemv;
  return {
    ...DEFAULT_GEOMETRY_PROFILE,
    id,
    linearGemv: { ...defaults, parallelShapes: [row, ...defaults.parallelShapes] },
  };
};

/** The pipeline keys and the output words of one `linear` of {@link SHAPE} at `m` rows under a table. */
const runLinear = async (
  table: GeometryProfile,
  m: number,
  level: "exact" | "reorder",
): Promise<{ keys: string[]; words: Uint32Array<ArrayBuffer> }> => {
  const { storage, n, k, bits, group } = SHAPE;
  const gpu = await acquireGpu({ geometryProfile: table });
  try {
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: Uint8Array.from(
          { length: (n * k * bits) / 8 },
          (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 24,
        ),
        encoding: {
          codec: CODEC[storage],
          groupSize: group,
          scale: {
            dtype: "f32",
            bytes: f32Bytes(
              Array.from({ length: n * (k / group) }, (_, i) => ((i % 17) + 1) * 0.00017),
            ),
          },
        },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: f32Bytes(Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11)),
        encoding: { codec: "f32" },
      },
    ];
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
      { numerics: { ops: { linear: level } } },
    );
    try {
      // Every row differs from every other.
      const input = Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.037) * 0.75);
      const output = (await session.run({ x: { dtype: "f32", shape: [m, k], data: input } })).y;
      const keys = requireCensus(session.diagnostics().lastRunPipelines, `M${m} ${level}`).map(
        (row) => row.key,
      );
      const words = new Uint32Array(output.data.buffer, output.data.byteOffset, m * n).slice();
      return { keys, words };
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

describe.skipIf(!GPU_AVAILABLE)("row-blocked parallel GEMV in a Session (real GPU)", () => {
  it("runs at M = 2..8 on a shape whose profile row carries a row block, with the words of the row without one", async () => {
    const plain = tableWith("row-block-off", 4, {});
    const blocked = tableWith("row-block-on", 4, { rowBlock: 4 });
    const { storage, group, n } = SHAPE;
    const decode = await runLinear(plain, 1, "reorder");
    for (const [m, rows] of [
      [2, 2],
      [3, 2],
      [4, 4],
      [5, 4],
      [8, 4],
    ] as const) {
      const without = await runLinear(plain, m, "reorder");
      const withBlock = await runLinear(blocked, m, "reorder");
      assertEquals(without.keys, [linearGemvParallelKey(storage, group, 4)], `M${m}`);
      assertEquals(withBlock.keys, [linearGemvParallelRowsKey(storage, group, 4, rows)], `M${m}`);
      assertEquals(withBlock.words, without.words, `M${m}`);
      // Decode and verify agree: row 0 is the M = 1 row
      assertEquals(withBlock.words.slice(0, n), decode.words, `M${m} row 0`);
    }
  });

  it("leaves M = 1, the level exact and a row count past the parallel range on the kernels they ran before", async () => {
    const plain = tableWith("row-block-off", 4, {});
    const blocked = tableWith("row-block-on", 4, { rowBlock: 4 });
    for (const [m, level] of [
      [1, "reorder"],
      [4, "exact"],
      [9, "reorder"],
    ] as const) {
      const without = await runLinear(plain, m, level);
      const withBlock = await runLinear(blocked, m, level);
      assertEquals(withBlock.keys, without.keys, `M${m} ${level}`);
      assertEquals(withBlock.words, without.words, `M${m} ${level}`);
    }
  });

  it("takes the row block whatever the row's staged flag: the flag is the form at M = 1", async () => {
    const { storage, group } = SHAPE;
    const staged = tableWith("row-block-staged", 32, { staged: true, rowBlock: 2 });
    assertEquals((await runLinear(staged, 1, "reorder")).keys, [
      linearGemvParallelStagedKey(storage, group, 32),
    ]);
    assertEquals((await runLinear(staged, 4, "reorder")).keys, [
      linearGemvParallelRowsKey(storage, group, 32, 2),
    ]);
  });
});
