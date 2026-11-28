/**
 * The split-K linear (`src/kernels/linear-splitk.ts` — the split of the GEMM skeleton) on a real GPU
 * — a `reorder` strategy: the products are the reference kernel's and only the order of the
 * additions differs, so its checks are bands, structure and one exact identity, not bit equality
 * with the reference kernel at two or more slices. Every weight storage the GEMM skeleton reads
 * under f32 compute (f32, f16, bf16, i8, i4 at group lengths 16 / 32 / 128, i2) is checked against
 * the reference GEMM of the same storage and tile geometry, and against an f64 evaluation over the
 * weights as the kernels decode them (`helpers/linear-storage-weights.ts`).
 *
 * 1. to 3. The class-R check of `@karume/kernel-checks` (`checkLinearSplitK` — the cases the
 *    workbench's kernel-check entry runs): the f64 band, the A/B band against the reference GEMM
 *    (`LINEAR_SPLITK_AB_BAND`, a declaration whose derivation is the constant's doc), the one-slice
 *    identity on every tile geometry, exact sums, the NaN row, each slice's term count, repeats and
 *    nothing written past the outputs; and the same bands through a Session.
 * 4. Fault injection, through the check's `mutate`: every slice starting one K tile late (each
 *    slice drops its first tile and reads the next slice's first) turns the f64 band red on every
 *    case of the bands part, and the one-slice identity and the exact sums red as well. Recorded
 *    2026-11-28 on an RTX 3080 Ti (Dawn / Vulkan) through a Session with the same edit made in the
 *    generator: at k = 512 the worst |error| was 2.57, about 12,100 times the band.
 *
 * It also checks where the strategy applies: only under the private option `LINEAR_SPLIT_K`, under
 * f32 compute for a reduction long enough to cut, and never for a single row (the GEMV family keeps
 * it); through a Session, every storage takes the partial of its own key.
 */

import { assert, assertEquals } from "@std/assert";
import {
  type CheckPart,
  checkLinearSplitK,
  LINEAR_SPLITK_AB_BAND,
  LINEAR_SPLITK_CASES,
  LINEAR_SPLITK_F64_BAND,
} from "@karume/kernel-checks";
import { describe, it } from "vite-plus/test";
import type { GpuContext } from "../src/gpu/device.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { linearSplitKCombineKey, linearSplitKSlices } from "../src/kernels/linear-splitk.ts";
import { GEMM_TOLERANCE } from "../src/reference/op-tolerance.ts";
import { createSessionFromContainer, LINEAR_SPLIT_K } from "../src/runtime/executor.ts";
import type { EncodingInput, TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import {
  type EncodedLinearWeight,
  encodeLinearWeight,
  type LinearWeightStorage,
  storageLabel,
  type WeightFixture,
} from "./helpers/linear-storage-weights.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";

type Shape = { readonly m: number; readonly n: number; readonly k: number };

/** Every weight storage the GEMM skeleton reads under f32 compute (i4 at three group lengths). */
const STORAGES: readonly LinearWeightStorage[] = [
  { storage: "f32" },
  { storage: "f16" },
  { storage: "bf16" },
  { storage: "i8" },
  { storage: "i4", groupSize: 16 },
  { storage: "i4", groupSize: 32 },
  { storage: "i4", groupSize: 128 },
  { storage: "i2" },
];

/**
 * Activations `[m, k]` and the bias `[n]`: `irregular` values of mixed signs and magnitudes, so the
 * partial sums cancel and round; `exact` small integers, so with the exact weights every product
 * and partial sum is an integer far below 2^24.
 */
const activations = ({ m, k }: Shape, fixture: WeightFixture): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: m * k }, (_, i) =>
    fixture === "exact" ? ((i * 7) % 5) - 2 : Math.sin(i * 0.037) * 0.75,
  );

const biasOf = (n: number, fixture: WeightFixture): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: n }, (_, i) =>
    fixture === "exact" ? (i % 9) - 4 : ((i % 7) - 3) * 0.11,
  );

/** The op's meaning: the sum in f64 over the decoded weights, then the bias. */
const referenceF64 = (
  { m, n, k }: Shape,
  x: Float32Array<ArrayBuffer>,
  w: Float32Array<ArrayBuffer>,
  bias: Float32Array<ArrayBuffer>,
): Float64Array<ArrayBuffer> => {
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

/** The op band at an f64 value. */
const band = (reference: number): number =>
  GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference);

/**
 * The worst |value − against| and the worst ratio of it to the op band at the f64 value of that
 * element (a non-finite value counts as infinitely far).
 */
const worst = (
  values: ArrayLike<number>,
  against: ArrayLike<number>,
  reference: Float64Array<ArrayBuffer>,
): { readonly maxAbs: number; readonly bandUse: number } => {
  let maxAbs = 0;
  let bandUse = 0;
  for (let i = 0; i < values.length; i += 1) {
    const error = Math.abs(values[i] - against[i]);
    const finite = Number.isFinite(error) ? error : Infinity;
    maxAbs = Math.max(maxAbs, finite);
    bandUse = Math.max(bandUse, finite / band(reference[i]));
  }
  return { maxAbs, bandUse };
};

const bits = (values: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(values.buffer, values.byteOffset, values.length).slice();

// ── Through a Session ────────────────────────────────────────────────────────────────────────────

/** The container encoding of an encoded weight (its exact bytes, its scale). */
const weightTensor = (weight: EncodedLinearWeight, n: number, k: number): TensorInput => {
  const bitsPer = { f32: 32, f16: 16, bf16: 16, i8: 8, i4: 4, i2: 2 }[weight.storage];
  const bytes = new Uint8Array(weight.words.buffer, 0, (n * k * bitsPer) / 8).slice();
  const scale =
    weight.scale === undefined
      ? undefined
      : { bytes: f32Bytes([...weight.scale]), dtype: "f32" as const };
  const encoding: EncodingInput =
    weight.storage === "f32" || weight.storage === "f16" || weight.storage === "bf16"
      ? { codec: weight.storage }
      : weight.storage === "i4"
        ? { codec: "int4-sym-g", groupSize: weight.groupSize, scale }
        : { codec: weight.storage === "i8" ? "int8-sym" : "int2-off", groupSize: k, scale };
  return { graph: GRAPH_NAME, initializer: "w", bytes, encoding };
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
  x: Float32Array<ArrayBuffer>,
  weight: EncodedLinearWeight,
  bias: Float32Array<ArrayBuffer>,
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
    weightTensor(weight, n, k),
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

/** Whether the run took the split (its partial at `slices` and the combine) and no plain GEMM. */
const usesSplitK = (keys: readonly string[], slices?: number, storagePart = ""): boolean =>
  keys.some(
    (key) =>
      key.startsWith("linear_splitk_gemm:") &&
      key.includes(`${storagePart}:s`) &&
      (slices === undefined || key.endsWith(`:s${slices}`)),
  ) &&
  keys.includes(linearSplitKCombineKey()) &&
  !keys.some((key) => key.startsWith("linear:"));

const F32: LinearWeightStorage = { storage: "f32" };

/** m and n off the reference geometry's tiles; k for 2, 4 and 16 slices. */
const SHAPES: readonly Shape[] = [
  { m: 3, n: 5, k: 512 },
  { m: 16, n: 64, k: 1536 },
  { m: 9, n: 37, k: 8960 },
];

describe.skipIf(!GPU_AVAILABLE)("the split-K linear through a Session (real GPU)", () => {
  it("stays inside the op's band against the f64 reference and inside the declared A/B band against the reference kernel", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const x = activations(shape, "irregular");
        const weight = encodeLinearWeight(F32, shape.n, shape.k, "irregular");
        const bias = biasOf(shape.n, "irregular");
        const reference = referenceF64(shape, x, weight.values, bias);
        const plain = await runLinear(gpu, shape, x, weight, bias, false);
        const split = await runLinear(gpu, shape, x, weight, bias, true);
        const slices = linearSplitKSlices(shape.k);
        const where = `[${shape.m},${shape.k}] × [${shape.n},${shape.k}]`;
        assert(usesSplitK(split.keys, slices), `${where}: ${split.keys.join(" ")}`);
        const ofSplit = worst(split.y, reference, reference);
        const ofPlain = worst(plain.y, reference, reference);
        const between = worst(split.y, plain.y, reference);
        console.log(
          `[linear-splitk] ${where} (${slices} slices): |error| against f64 — ` +
            `split-K ${ofSplit.maxAbs.toExponential(2)} (${(ofSplit.bandUse * 100).toFixed(1)}% of the band), ` +
            `reference kernel ${ofPlain.maxAbs.toExponential(2)} (${(ofPlain.bandUse * 100).toFixed(1)}%); ` +
            `between the two ${between.maxAbs.toExponential(2)} (${(between.bandUse * 100).toFixed(1)}%)`,
        );
        assert(ofSplit.bandUse <= 1, `${where}: outside the band against f64 (${ofSplit.maxAbs})`);
        assert(
          between.bandUse <= LINEAR_SPLITK_AB_BAND,
          `${where}: outside the A/B band against the reference kernel (${between.maxAbs})`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("takes the partial of the weight's own storage for every storage, inside both bands", async () => {
    const gpu = await acquireGpu();
    try {
      // Past the GEMV family's 64 rows, so the integer storages reach the split too.
      const shape = { m: 128, n: 64, k: 1536 };
      const x = activations(shape, "irregular");
      const bias = biasOf(shape.n, "irregular");
      for (const storage of STORAGES) {
        const weight = encodeLinearWeight(storage, shape.n, shape.k, "irregular");
        const reference = referenceF64(shape, x, weight.values, bias);
        const plain = await runLinear(gpu, shape, x, weight, bias, false);
        const split = await runLinear(gpu, shape, x, weight, bias, true);
        const where = storageLabel(storage);
        // The storage's key part as the plain GEMM's key carries it (`:wi4g32`, none for f32).
        const storagePart = linearKey(
          storage.storage,
          true,
          "f32",
          undefined,
          storage.groupSize,
        ).replace(/^linear:v\d+:f32:reg128x128r8x8w16v4/, "");
        assert(usesSplitK(split.keys, 4, storagePart), `${where}: ${split.keys.join(" ")}`);
        const ofSplit = worst(split.y, reference, reference);
        const between = worst(split.y, plain.y, reference);
        console.log(
          `[linear-splitk] ${where} [128,1536] × [64,1536] (4 slices): against f64 ${(ofSplit.bandUse * 100).toFixed(1)}% of the band, ` +
            `against the reference kernel ${(between.bandUse * 100).toFixed(1)}%`,
        );
        assert(ofSplit.bandUse <= 1, `${where}: outside the band against f64 (${ofSplit.maxAbs})`);
        assert(between.bandUse <= LINEAR_SPLITK_AB_BAND, `${where}: outside the A/B band`);
        assertEquals(bits(split.again), bits(split.y), `${where}: a second run`);
      }
    } finally {
      gpu.destroy();
    }
  });

  it("equals the reference kernel bit for bit on inputs whose sums are exact, edges included", async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const x = activations(shape, "exact");
        const weight = encodeLinearWeight(F32, shape.n, shape.k, "exact");
        const bias = biasOf(shape.n, "exact");
        const plain = await runLinear(gpu, shape, x, weight, bias, false);
        const split = await runLinear(gpu, shape, x, weight, bias, true);
        assert(usesSplitK(split.keys));
        assertEquals(bits(split.y), bits(plain.y));
        // The exact value itself, so that two kernels wrong in the same way do not pass.
        assertEquals(Array.from(split.y), Array.from(referenceF64(shape, x, weight.values, bias)));
      }
    } finally {
      gpu.destroy();
    }
  });

  it("carries a NaN of one input row to every output of that row and to no other, and repeats its bits", async () => {
    const gpu = await acquireGpu();
    try {
      const shape = { m: 9, n: 37, k: 1536 };
      const x = activations(shape, "irregular");
      const poisoned = 4;
      // In the last slice, so the combine pass is what carries it.
      x[poisoned * shape.k + shape.k - 1] = Number.NaN;
      const weight = encodeLinearWeight(F32, shape.n, shape.k, "irregular");
      const split = await runLinear(gpu, shape, x, weight, biasOf(shape.n, "irregular"), true);
      assert(usesSplitK(split.keys));
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
      const run = async (shape: Shape, splitK: boolean): Promise<Run> =>
        runLinear(
          gpu,
          shape,
          activations(shape, "irregular"),
          encodeLinearWeight(F32, shape.n, shape.k, "irregular"),
          biasOf(shape.n, "irregular"),
          splitK,
        );
      const long = { m: 16, n: 64, k: 1536 };
      assertEquals(usesSplitK((await run(long, false)).keys), false);
      const short = { m: 16, n: 64, k: 448 };
      assertEquals(linearSplitKSlices(short.k), undefined);
      assertEquals(usesSplitK((await run(short, true)).keys), false);
      const oneRow = { m: 1, n: 64, k: 1536 };
      assertEquals(usesSplitK((await run(oneRow, true)).keys), false);
    } finally {
      gpu.destroy();
    }
  });
});

// ── The shared check ────────────────────────────────────────────────────────────────────────────

/** Replaces `from` once; a missing anchor throws instead of leaving the kernel unmutated. */
const mustReplace = (wgsl: string, from: string, to: string): string => {
  if (!wgsl.includes(from)) throw new Error(`mutation anchor not found: ${from}`);
  return wgsl.replace(from, to);
};

/** Every slice starts one K tile late: it drops its first tile and reads the next slice's first. */
const sliceOneTileLate = (wgsl: string): string =>
  mustReplace(
    wgsl,
    "let t_first = wid.z * slice_tiles + min(wid.z, slice_extra);",
    "let t_first = wid.z * slice_tiles + min(wid.z, slice_extra) + 1u;",
  );

/** The failures of one criterion in every case of a part. */
const failuresOf = (part: CheckPart | undefined, criterion: string): readonly number[] => {
  assert(part !== undefined, "the part is missing");
  return part.cases.map(({ label, criteria }) => {
    const found = criteria.find(({ name }) => name === criterion);
    assert(found !== undefined, `${label}: no '${criterion}' criterion`);
    return found.failures;
  });
};

describe.skipIf(!GPU_AVAILABLE)("the split-K linear: the class-R check (real GPU)", () => {
  it("passes every criterion of every case of every part of the shared check", async () => {
    const gpu = await acquireGpu();
    try {
      const parts = await checkLinearSplitK(gpu);
      // Every part ran its whole grid (the counts are pinned in the package's own test).
      assertEquals(
        Object.fromEntries(parts.map((part) => [part.id, part.cases.length])),
        LINEAR_SPLITK_CASES,
      );
      for (const part of parts) assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("is not vacuous: a slice starting one K tile late fails the f64 band on every bands case, the one-slice identity and the exact sums", async () => {
    const gpu = await acquireGpu();
    try {
      const parts = await checkLinearSplitK(gpu, { mutate: sliceOneTileLate });
      const byId = new Map(parts.map((part) => [part.id, part]));
      const bands = failuresOf(byId.get("linear-splitk-bands"), LINEAR_SPLITK_F64_BAND);
      assertEquals(bands.length, LINEAR_SPLITK_CASES["linear-splitk-bands"]);
      assert(
        bands.every((failures) => failures > 0),
        `${bands.filter((f) => f === 0).length} bands cases passed`,
      );
      const identity = failuresOf(
        byId.get("linear-splitk-identity"),
        "one slice: the reference GEMM's words (u32)",
      );
      assert(identity.every((failures) => failures > 0));
      const exact = failuresOf(
        byId.get("linear-splitk-exact"),
        "exact sums: the exact value (u32)",
      );
      assert(exact.every((failures) => failures > 0));
    } finally {
      gpu.destroy();
    }
  });
});

describe("linearSplitKSlices", () => {
  it("takes the most slices whose shortest slice holds 16 K tiles, whether or not they divide the tiles or K", () => {
    assertEquals(linearSplitKSlices(8960), 16);
    assertEquals(linearSplitKSlices(4304), 16);
    assertEquals(linearSplitKSlices(2048), 8);
    assertEquals(linearSplitKSlices(1536), 4);
    assertEquals(linearSplitKSlices(1000), 2);
    assertEquals(linearSplitKSlices(512), 2);
    assertEquals(linearSplitKSlices(497), 2);
    assertEquals(linearSplitKSlices(496), undefined);
    assertEquals(linearSplitKSlices(448), undefined);
  });
});
