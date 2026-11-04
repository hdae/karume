// Real-GPU check of the skinny-M tile geometry (`gemmGeometryForRows` in
// src/kernels/gemm-geometry.ts).
//
// The geometry may decide **only the work split** (ADR 0022). So there are two things to look at,
// and both are built on the premise that a different geometry runs only for small M:
//
// 1. **Agreement with the CPU reference** — the three buckets small M (M <= 64) / medium M
//    (65..512) / default, and shapes that cross their boundaries.
//    Changing the tile edge also moves the partial tiles, so `m % tileM != 0` / `n % tileN != 0` /
//    `k % 16 != 0` are mixed in. If the dispatch count and the generated geometry disagree, whole
//    output tiles go missing, and that shows up here as a difference from the reference.
// 2. **Bit identity across buckets** — with the same weights and the same row values, the first 4
//    rows of the M=4 output (small-M geometry) and the M=128 output (default geometry) **must not
//    differ by a single bit**. The K reduction order per output element (outer t ascending, inner
//    kk ascending, K tile 16) is the same whatever the geometry, and this is the unit-level form of
//    the measured claim that "swapping the geometry does not break the PNG / WAV sha256 checks".
//    MUST: if it breaks, suspect **the claim itself**, not the geometry choice (go back to the
//    design decision).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor } from "../src/reference/ops.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * Deterministic sequences with signs and magnitudes spread enough to cause cancellation (no random
 * numbers).
 */
const XS = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;
const WS = (index: number): number => ((index % 7) - 3) * 0.5 - 0.0625;
const BS = (index: number): number => ((index % 5) - 2) * 0.25;

type ShapeCase = {
  readonly name: string;
  readonly op: "linear" | "matmul" | "bmm";
  readonly inputs: readonly RefTensor[];
  readonly outShape: readonly number[];
};

const runCase = async (gpu: GpuContext, testCase: ShapeCase): Promise<Tensor> => {
  const graph = singleOpDeclaration(
    testCase.op,
    testCase.inputs.map((input) => input.shape),
    [testCase.outShape],
  );
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    const named: Record<string, Tensor> = {};
    testCase.inputs.forEach((input, index) => {
      named[`x${index}`] = input;
    });
    return (await session.run(named))["y"];
  } finally {
    await session.dispose();
  }
};

const linearCase = (m: number, k: number, n: number): ShapeCase => ({
  name: `linear m${m} k${k} n${n}`,
  op: "linear",
  inputs: [fill([m, k], XS), fill([n, k], WS), fill([n], BS)],
  outShape: [m, n],
});

const matmulCase = (m: number, k: number, n: number): ShapeCase => ({
  name: `matmul m${m} k${k} n${n}`,
  op: "matmul",
  inputs: [fill([m, k], XS), fill([k, n], WS)],
  outShape: [m, n],
});

const bmmCase = (batch: number, m: number, k: number, n: number): ShapeCase => ({
  name: `bmm b${batch} m${m} k${k} n${n}`,
  op: "bmm",
  inputs: [fill([batch, m, k], XS), fill([batch, k, n], WS)],
  outShape: [batch, m, n],
});

/**
 * Shape selection. It has **both the inside of each bucket and the side one past the boundary**
 * (touching only one side of a boundary cannot detect mixing up `<` and `<=` in the table's
 * comparisons).
 * `n` goes through both the v4 path (`k % 4 == 0 && n % 4 == 0`) and the scalar path.
 */
const SHAPE_CASES: readonly ShapeCase[] = [
  // Small-M bucket (tileM 16 / tileN 16). Includes the measured shapes M=1 / M=4
  linearCase(1, 20, 68),
  linearCase(4, 20, 68),
  linearCase(4, 37, 23),
  linearCase(1, 1, 7),
  linearCase(16, 132, 64),
  // Upper half of the same bucket (17–64 — the range with 2–4 row tiles). Lower end 17 and upper
  // end 64
  linearCase(17, 64, 33),
  linearCase(33, 16, 68),
  linearCase(64, 20, 68),
  // Medium-M bucket (65..512 — tileM 64 / tileN 32). Lower end 65 and upper end 512
  linearCase(65, 16, 16),
  linearCase(127, 40, 132),
  linearCase(512, 20, 36),
  // The side that falls to the default geometry (outside the boundary)
  linearCase(513, 16, 16),
  matmulCase(4, 20, 68),
  matmulCase(5, 37, 23),
  matmulCase(17, 16, 64),
  matmulCase(64, 132, 36),
  matmulCase(65, 20, 20),
  matmulCase(513, 8, 12),
  // bmm shapes also check that the batch axis (the dispatch z) is independent of the geometry (B /
  // M / K / N all distinct)
  bmmCase(3, 4, 20, 68),
  bmmCase(2, 5, 7, 9),
  bmmCase(3, 17, 12, 36),
  bmmCase(2, 65, 16, 16),
];

test.skipIf(!GPU_AVAILABLE)(
  "the 3 GEMM ops with small M match the CPU reference across bucket boundaries (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of SHAPE_CASES) {
        const actual = await runCase(gpu, testCase);
        const expected = applyReferenceOp(testCase.op, testCase.inputs, {}, testCase.outShape);
        assertEquals(actual.shape, expected.shape, testCase.name);
        const report = compareTensors(actual, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/** Raw f32 bits (looks at a 1-bit difference, -0 / NaN included). */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const assertRowsBitEqual = (
  actual: Tensor,
  expected: Tensor,
  columns: number,
  rows: number,
  where: string,
): void => {
  const a = bits(actual);
  const b = bits(expected);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const index = row * columns + col;
      assert(
        a[index] === b[index],
        `${where}: [${row},${col}] has different bits（0x${a[index].toString(16)} vs ` +
          `0x${b[index].toString(16)} = ${actual.data[index]} vs ${expected.data[index]}）`,
      );
    }
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the output of the same row does not differ by a single bit across buckets (the geometry only splits the work / real GPU)",
  async () => {
    const k = 132;
    const n = 68;
    // Four runs that differ only in the row count. 513 = default geometry (the baseline) / 128 =
    // medium M (M64N32) / 33 and 4 = small M (M16N16 — two points with the same geometry but
    // different partial-tile positions). Checks that every bucket matches the baseline bit for bit
    const rowCounts = [513, 128, 33, 4] as const;
    const gpu = await acquireGpu();
    try {
      for (const op of ["linear", "matmul"] as const) {
        const weight = op === "linear" ? fill([n, k], WS) : fill([k, n], WS);
        const bias = fill([n], BS);
        const outputs = new Map<number, Tensor>();
        for (const rows of rowCounts) {
          // Row-major, so the flat index `r·k + c` does not depend on the row count = the input
          // values of row r are shared by the three runs
          const x = fill([rows, k], XS);
          const inputs = op === "linear" ? [x, weight, bias] : [x, weight];
          outputs.set(
            rows,
            await runCase(gpu, {
              name: `${op} m${rows}`,
              op,
              inputs,
              outShape: [rows, n],
            }),
          );
        }
        const reference = outputs.get(513);
        assert(reference !== undefined);
        for (const rows of [128, 33, 4] as const) {
          const actual = outputs.get(rows);
          assert(actual !== undefined);
          assertRowsBitEqual(actual, reference, n, rows, `${op} m${rows} vs m513`);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
