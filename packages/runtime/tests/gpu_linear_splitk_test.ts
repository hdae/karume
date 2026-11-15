/**
 * The split-K linear (`src/kernels/linear-splitk.ts`) on a real GPU — a `reorder` strategy: the
 * products are the reference kernel's and only the order of the additions differs, so its checks are
 * a band, not bit equality with the reference kernel.
 *
 * 1. Against the f64 reference: every output inside the op's band (`GEMM_TOLERANCE`).
 * 2. Against the reference kernel on the same input: inside twice that band.
 * 3. Structure, checked exactly: on inputs whose products and sums are exact in f32 the order of the
 *    additions cannot matter, so the output equals the reference kernel's bit for bit (a dropped term,
 *    a wrong slice or a missed edge element shows here); a NaN in one input row reaches every output
 *    of that row and no other; two runs of one Session give the same bits.
 * 4. Fault injection (recorded 2026-11-15, RTX 3080 Ti): with the partial pass starting each slice one
 *    element late (`first = slice * per + 1u`, so it drops each slice's first term and reads one past
 *    its end) the band check failed at the first shape (k = 512: worst |error| 6.7e-2, 333 times the
 *    band) and the exact check failed.
 *
 * It also checks where the strategy applies: only under the private option `LINEAR_SPLIT_K`, for an
 * f32-stored weight whose reduction is long enough to cut, and never for a single row (the GEMV
 * family keeps it).
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { GpuContext } from "../src/gpu/device.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  linearSplitKCombineKey,
  linearSplitKPartialKey,
  linearSplitKSlices,
} from "../src/kernels/linear-splitk.ts";
import { GEMM_TOLERANCE } from "../src/reference/op-tolerance.ts";
import { createSessionFromContainer, LINEAR_SPLIT_K } from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";

type Shape = { readonly m: number; readonly n: number; readonly k: number };

type Values = {
  readonly x: Float32Array<ArrayBuffer>;
  readonly w: Float32Array<ArrayBuffer>;
  readonly bias: Float32Array<ArrayBuffer>;
};

/** Values with mixed signs and magnitudes, so the partial sums cancel and round. */
const irregular = ({ m, n, k }: Shape): Values => ({
  x: Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.037) * 0.75),
  w: Float32Array.from({ length: n * k }, (_, i) => Math.cos(i * 0.011 + 0.3) * 0.5),
  bias: Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11),
});

/**
 * Small integers: every product and every partial sum is an integer far below 2^24, so each is exact
 * in f32 and the sum does not depend on the order of the additions.
 */
const exact = ({ m, n, k }: Shape): Values => ({
  x: Float32Array.from({ length: m * k }, (_, i) => ((i * 7) % 5) - 2),
  w: Float32Array.from({ length: n * k }, (_, i) => ((i * 11) % 7) - 3),
  bias: Float32Array.from({ length: n }, (_, i) => (i % 9) - 4),
});

/** The op's meaning: the sum in f64 over the f32 inputs, then the bias. */
const referenceF64 = ({ m, n, k }: Shape, { x, w, bias }: Values): Float64Array<ArrayBuffer> => {
  const y = new Float64Array(m * n);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      let sum = 0;
      for (let inner = 0; inner < k; inner += 1) sum += x[row * k + inner] * w[col * k + inner];
      y[row * n + col] = sum + bias[col];
    }
  }
  return y;
};

type Run = {
  readonly y: Float32Array<ArrayBuffer>;
  /** The pipeline keys of the run's dispatches. */
  readonly keys: readonly string[];
  /** The same request once more on the same Session. */
  readonly again: Float32Array<ArrayBuffer>;
};

const runLinear = async (
  gpu: GpuContext,
  { m, n, k }: Shape,
  { x, w, bias }: Values,
  splitK: boolean,
): Promise<Run> => {
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
  const tensors: readonly TensorInput[] = [
    { graph: GRAPH_NAME, initializer: "w", bytes: f32Bytes(w), encoding: { codec: "f32" } },
    { graph: GRAPH_NAME, initializer: "b", bytes: f32Bytes(bias), encoding: { codec: "f32" } },
  ];
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, tensors),
    GRAPH_NAME,
    splitK ? { [LINEAR_SPLIT_K]: true } : {},
  );
  try {
    const request = { x: { dtype: "f32", shape: [m, k], data: x } } as const;
    const first = (await session.run(request)).y;
    assertEquals(first.shape, [m, n]);
    const keys = (session.diagnostics().lastRunPipelines ?? []).map((row) => row.key);
    const again = (await session.run(request)).y;
    assert(first.data instanceof Float32Array && again.data instanceof Float32Array);
    return { y: first.data.slice(), keys, again: again.data.slice() };
  } finally {
    await session.dispose();
  }
};

const bits = (values: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(values.buffer, values.byteOffset, values.length).slice();

const usesSplitK = (keys: readonly string[]): boolean =>
  keys.includes(linearSplitKPartialKey()) && keys.includes(linearSplitKCombineKey());

/** The worst |value − reference| and the worst ratio of it to the band at that element. */
const worst = (
  values: Float32Array<ArrayBuffer>,
  reference: ArrayLike<number>,
  scale: number,
): { readonly maxAbs: number; readonly bandUse: number } => {
  let maxAbs = 0;
  let bandUse = 0;
  for (let i = 0; i < values.length; i += 1) {
    const error = Math.abs(values[i] - reference[i]);
    const band = scale * (GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference[i]));
    maxAbs = Math.max(maxAbs, error);
    bandUse = Math.max(bandUse, error / band);
  }
  return { maxAbs, bandUse };
};

/** m and n off the 8 × 8 tile of the partial pass; k for 2, 4 and 16 slices. */
const SHAPES: readonly Shape[] = [
  { m: 3, n: 5, k: 512 },
  { m: 16, n: 64, k: 1536 },
  { m: 9, n: 37, k: 8960 },
];

describe.skipIf(!GPU_AVAILABLE)("the split-K linear (real GPU)", () => {
  it("stays inside the op's band against the f64 reference and inside twice it against the reference kernel", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const values = irregular(shape);
        const reference = referenceF64(shape, values);
        const plain = await runLinear(gpu, shape, values, false);
        const split = await runLinear(gpu, shape, values, true);
        const where = `[${shape.m},${shape.k}] × [${shape.n},${shape.k}]`;
        assert(usesSplitK(split.keys), `${where}: ${split.keys.join(" ")}`);
        const ofSplit = worst(split.y, reference, 1);
        const ofPlain = worst(plain.y, reference, 1);
        const between = worst(split.y, plain.y, 2);
        console.log(
          `[linear-splitk] ${where} (${linearSplitKSlices(shape.k)} slices): |error| against f64 — ` +
            `split-K ${ofSplit.maxAbs.toExponential(2)} (${(ofSplit.bandUse * 100).toFixed(1)}% of the band), ` +
            `reference kernel ${ofPlain.maxAbs.toExponential(2)} (${(ofPlain.bandUse * 100).toFixed(1)}%); ` +
            `between the two ${between.maxAbs.toExponential(2)}`,
        );
        assert(ofSplit.bandUse <= 1, `${where}: outside the band against f64 (${ofSplit.maxAbs})`);
        assert(
          between.bandUse <= 1,
          `${where}: outside twice the band against the reference kernel`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("equals the reference kernel bit for bit on inputs whose sums are exact, edges included", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const values = exact(shape);
        const plain = await runLinear(gpu, shape, values, false);
        const split = await runLinear(gpu, shape, values, true);
        assert(usesSplitK(split.keys));
        assertEquals(bits(split.y), bits(plain.y));
        // The exact value itself, so that two kernels wrong in the same way do not pass.
        assertEquals(Array.from(split.y), Array.from(referenceF64(shape, values)));
      }
    } finally {
      gpu.destroy();
    }
  });

  it("carries a NaN of one input row to every output of that row and to no other, and repeats its bits", async () => {
    const gpu = await acquireGpu();
    try {
      const shape = { m: 9, n: 37, k: 1536 };
      const values = irregular(shape);
      const poisoned = 4;
      // In the last slice, so the combine pass is what carries it.
      values.x[poisoned * shape.k + shape.k - 1] = Number.NaN;
      const split = await runLinear(gpu, shape, values, true);
      for (let row = 0; row < shape.m; row += 1) {
        for (let col = 0; col < shape.n; col += 1) {
          assertEquals(
            Number.isNaN(split.y[row * shape.n + col]),
            row === poisoned,
            `row ${row} col ${col}`,
          );
        }
      }
      assertEquals(bits(split.again), bits(split.y));
    } finally {
      gpu.destroy();
    }
  });

  it("is taken only under the private option, for a reduction long enough to cut, and never for one row", async () => {
    const gpu = await acquireGpu();
    try {
      const long = { m: 16, n: 64, k: 1536 };
      assertEquals(usesSplitK((await runLinear(gpu, long, irregular(long), false)).keys), false);
      const short = { m: 16, n: 64, k: 448 };
      assertEquals(linearSplitKSlices(short.k), undefined);
      assertEquals(usesSplitK((await runLinear(gpu, short, irregular(short), true)).keys), false);
      const oneRow = { m: 1, n: 64, k: 1536 };
      assertEquals(usesSplitK((await runLinear(gpu, oneRow, irregular(oneRow), true)).keys), false);
    } finally {
      gpu.destroy();
    }
  });
});

describe("linearSplitKSlices", () => {
  it("takes the most slices that divide k and leave each at least 256 terms", () => {
    assertEquals(linearSplitKSlices(8960), 16);
    assertEquals(linearSplitKSlices(2048), 8);
    assertEquals(linearSplitKSlices(1536), 4);
    assertEquals(linearSplitKSlices(512), 2);
    assertEquals(linearSplitKSlices(511), undefined);
    assertEquals(linearSplitKSlices(513), undefined);
  });
});
