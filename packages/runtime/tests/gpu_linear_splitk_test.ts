/**
 * The split-K linear (`src/kernels/linear-splitk.ts` — the split of the GEMM skeleton) on a real GPU
 * — a `reorder` strategy: the products are the reference kernel's and only the order of the
 * additions differs, so its checks are bands, structure and one exact identity, not bit equality
 * with the reference kernel at two or more slices. Every weight storage the GEMM skeleton reads
 * under f32 compute (f32, f16, bf16, i8, i4 at group lengths 16 / 32 / 128, i2) is checked against
 * the reference GEMM of the same storage and tile geometry, and against an f64 evaluation over the
 * weights as the kernels decode them (`helpers/linear-storage-weights.ts`).
 *
 * 1. Against the f64 reference: every output inside the op's band (`GEMM_TOLERANCE`).
 * 2. Against the reference kernel on the same input: inside {@link SPLITK_AB_BAND} times that band
 *    (a declaration — its derivation is the constant's doc).
 * 3. Structure, checked exactly: with one slice the two dispatches give the reference GEMM's words
 *    for every storage on every tile geometry the linear GEMM has, scalar and vec4 (the split shares
 *    the GEMM's loop and only regroups it); on inputs whose products and sums are exact in f32 the
 *    order of the additions cannot matter, so the output equals the reference kernel's bit for bit
 *    at every slice count, a tile count no slice count divides and a partial last K tile included (a
 *    dropped term, a wrong slice or a missed edge element shows here); on all-ones inputs each
 *    slice's partial sums count its own terms (the first and last slices' lengths, more slices than
 *    tiles); a NaN in one input row reaches every output of that row and no other; two runs give the
 *    same bits; nothing is written past the outputs or the partial sums.
 * 4. Fault injection (recorded 2026-11-28, RTX 3080 Ti, Dawn / Vulkan): with every slice starting
 *    one K tile late (`t_first = wid.z * slice_tiles + min(wid.z, slice_extra) + 1u` in the
 *    generator, so each slice drops its first tile and reads the next slice's first) the f64 band
 *    failed at the first shape (k = 512: worst |error| 2.57, about 12,100 times the band), the exact
 *    check failed, and the one-slice identity failed at the first geometry it compared.
 *
 * It also checks where the strategy applies: only under the private option `LINEAR_SPLIT_K`, under
 * f32 compute for a reduction long enough to cut, and never for a single row (the GEMV family keeps
 * it); through a Session, every storage takes the partial of its own key.
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { GpuContext } from "../src/gpu/device.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  defaultGemmGeometry,
  type GemmGeometry,
  gemmTileM,
  gemmTileN,
} from "../src/kernels/gemm-geometry.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { linearSplitKCombineKey, linearSplitKSlices } from "../src/kernels/linear-splitk.ts";
import { GEMM_TOLERANCE } from "../src/reference/op-tolerance.ts";
import { createSessionFromContainer, LINEAR_SPLIT_K } from "../src/runtime/executor.ts";
import type { EncodingInput, TensorInput } from "./helpers/container-write.ts";
import { linearGemmGeometries } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type SplitKDispatchCase,
  SPLITK_UNWRITTEN,
  splitKRunner,
} from "./helpers/linear-splitk-dispatch.ts";
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

/**
 * The A/B band against the reference kernel, as a multiple of the op band at the f64 value
 * (`GEMM_TOLERANCE.atol + rtol · |f64|`). A declaration (ADR 0147 decision 2: at least 4 × the worst
 * measured, at most 2 × the f64 band):
 *
 * | Case | Worst \|split − reference\| / op band |
 * | --- | --- |
 * | f32 through a Session, 3 × 512 × 5 / 16 × 1536 × 64 / 9 × 8960 × 37 (m × k × n) at 2 / 4 / 16 slices | 0.028 / 0.144 / 0.393 |
 * | every storage through a Session, 128 × 1536 × 64 at 4 slices | 0.122 (i4 g16) to 0.192 (i2) |
 * | every storage by raw dispatches, 37 × 4096 × 68 and 37 × 4300 (i4: 4224) × 69, 2 to 16 slices, two geometries | 0.390 (bf16, K 4096, 4 slices) |
 *
 * Measured 2026-11-28 on an RTX 3080 Ti (Dawn / Vulkan; the Session cases at the nvidia-ampere
 * profile's geometries) with the `irregular` fixtures: the worst is 0.393, so the band is at least
 * 1.57 and at most 2.
 */
const SPLITK_AB_BAND = 2;

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

const floats = (words: Uint32Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  new Float32Array(words.buffer, words.byteOffset, words.length);

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
          between.bandUse <= SPLITK_AB_BAND,
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
        assert(between.bandUse <= SPLITK_AB_BAND, `${where}: outside the A/B band`);
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

// ── By raw dispatches ────────────────────────────────────────────────────────────────────────────

/** A raw case of a storage and fixture at a geometry and slice count. */
const rawCase = (
  storage: LinearWeightStorage,
  fixture: WeightFixture,
  geometry: GemmGeometry,
  shape: Shape,
  slices: number,
): SplitKDispatchCase => ({
  geometry,
  ...shape,
  slices,
  weight: encodeLinearWeight(storage, shape.n, shape.k, fixture),
  x: activations(shape, fixture),
  bias: biasOf(shape.n, fixture),
});

/** The rows of a raw case's output whose words hold a NaN. */
const nanRows = (words: Uint32Array<ArrayBuffer>, n: number): Set<number> => {
  const values = floats(words);
  const rows = new Set<number>();
  for (let i = 0; i < values.length; i += 1)
    if (Number.isNaN(values[i])) rows.add(Math.floor(i / n));
  return rows;
};

/** A K whose tile count every slice count divides, and one it does not (with a partial last tile where the storage allows). */
const evenK = (): number => 4096;
const unevenK = ({ storage, groupSize }: LinearWeightStorage): number =>
  storage === "i4" ? 33 * Math.max(groupSize, 128) : 4300;

/**
 * The tile geometries of the bands: the default 128 × 128 and the 16 × 16 of up to 64 rows. The
 * bands are a property of the slices' grouping, which no geometry changes (a geometry only
 * reassigns outputs to threads); the other geometries and the skeleton forms are covered by the
 * one-slice identity and the exact sums.
 */
const BAND_GEOMETRIES: readonly GemmGeometry[] = [
  defaultGemmGeometry(),
  { regM: 1, regN: 4, wgX: 4, wgY: 16 },
];

const SLICE_COUNTS = [2, 4, 8, 16] as const;

describe.skipIf(!GPU_AVAILABLE)("the split-K linear by raw dispatches (real GPU)", () => {
  it("gives the reference GEMM's words at one slice for every storage on every tile geometry the linear GEMM has, scalar and vec4", async () => {
    const gpu = await acquireGpu();
    try {
      const run = splitKRunner(gpu);
      const geometries = linearGemmGeometries();
      // The embedded profiles hold this many today; a new one joins the loop by itself.
      assert(geometries.length >= 10, `${geometries.length} geometries`);
      for (const storage of STORAGES) {
        // A partial last K tile where the storage allows (i4's K is a multiple of its group).
        const k = storage.storage === "i4" ? 2 * storage.groupSize : 100;
        for (const geometry of geometries) {
          const rows = gemmTileM(geometry) + 3;
          // Two row tiles and two column tiles with ragged edges: vec4 for every storage, and the
          // scalar kernel for f32 on every geometry and for the other storages on the default.
          const scalar =
            storage.storage === "f32" ||
            JSON.stringify(geometry) === JSON.stringify(defaultGemmGeometry());
          const columns = [gemmTileN(geometry) + 4, ...(scalar ? [gemmTileN(geometry) + 5] : [])];
          for (const n of columns) {
            const shape = { m: rows, n, k };
            const where = `${storageLabel(storage)} ${linearKey(storage.storage, gemmUsesVec4(k, n), "f32", undefined, storage.groupSize, geometry)} ${rows}×${n}×${k}`;
            const result = await run(rawCase(storage, "irregular", geometry, shape, 1));
            assert(!result.reference.includes(SPLITK_UNWRITTEN), `${where}: unwritten outputs`);
            assertEquals(result.split, result.reference, where);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  });

  it("stays inside both bands at every slice count for every storage, a tile count no count divides included, repeats its words and writes nothing past its outputs", async () => {
    const gpu = await acquireGpu();
    try {
      const run = splitKRunner(gpu);
      let worstAb = 0;
      let worstAbWhere = "";
      let worstF64 = 0;
      for (const storage of STORAGES) {
        // A vec4 shape at the even K, a scalar one at the uneven K; m and n off every tile.
        for (const shape of [
          { m: 37, n: 68, k: evenK() },
          { m: 37, n: 69, k: unevenK(storage) },
        ]) {
          const base = rawCase(storage, "irregular", BAND_GEOMETRIES[0], shape, 1);
          const reference = referenceF64(shape, base.x, base.weight.values, base.bias);
          for (const geometry of BAND_GEOMETRIES) {
            for (const slices of SLICE_COUNTS) {
              const where = `${storageLabel(storage)} ${shape.m}×${shape.n}×${shape.k} ${linearKey(storage.storage, gemmUsesVec4(shape.k, shape.n), "f32", undefined, storage.groupSize, geometry)} s${slices}`;
              const result = await run({ ...base, geometry, slices });
              const ofSplit = worst(floats(result.split), reference, reference);
              const between = worst(floats(result.split), floats(result.reference), reference);
              worstF64 = Math.max(worstF64, ofSplit.bandUse);
              if (between.bandUse > worstAb) {
                worstAb = between.bandUse;
                worstAbWhere = where;
              }
              assert(
                ofSplit.bandUse <= 1,
                `${where}: outside the band against f64 (${ofSplit.maxAbs})`,
              );
              assert(
                between.bandUse <= SPLITK_AB_BAND,
                `${where}: outside the A/B band (${between.maxAbs})`,
              );
              assertEquals(result.again, result.split, `${where}: a second dispatch`);
              assert(
                result.tails.every((word) => word === SPLITK_UNWRITTEN),
                `${where}: written past m · n`,
              );
              assert(
                result.partials
                  .subarray(slices * shape.m * shape.n)
                  .every((word) => word === SPLITK_UNWRITTEN),
                `${where}: written past the partial sums`,
              );
            }
          }
        }
      }
      console.log(
        `[linear-splitk] raw bands: worst against f64 ${(worstF64 * 100).toFixed(1)}% of the band; ` +
          `worst against the reference GEMM ${(worstAb * 100).toFixed(1)}% (${worstAbWhere})`,
      );
    } finally {
      gpu.destroy();
    }
  });

  it("equals the reference GEMM's words and the exact value on exact sums at every slice count for every storage, and on every tile geometry", async () => {
    const gpu = await acquireGpu();
    try {
      const run = splitKRunner(gpu);
      // Every geometry and skeleton form (vec4, so the forms apply), where the slices' bases are
      // what a one-slice run cannot reach: f32 at 4 and 16 slices of a tile count neither divides.
      for (const geometry of linearGemmGeometries()) {
        const shape = { m: gemmTileM(geometry) + 3, n: gemmTileN(geometry) + 4, k: 4300 };
        const base = rawCase(F32, "exact", geometry, shape, 1);
        const expected = Float32Array.from(
          referenceF64(shape, base.x, base.weight.values, base.bias),
        );
        for (const slices of [4, 16]) {
          const where = `${linearKey("f32", true, "f32", undefined, undefined, geometry)} s${slices}`;
          const result = await run({ ...base, slices });
          assertEquals(result.split, result.reference, where);
          assertEquals(result.split, bits(expected), `${where}: the exact value`);
        }
      }
      for (const storage of STORAGES) {
        const shape = { m: 37, n: 69, k: unevenK(storage) };
        const base = rawCase(storage, "exact", BAND_GEOMETRIES[0], shape, 1);
        const expected = Float32Array.from(
          referenceF64(shape, base.x, base.weight.values, base.bias),
        );
        for (const slices of SLICE_COUNTS) {
          const where = `${storageLabel(storage)} K${shape.k} s${slices}`;
          const result = await run({ ...base, slices });
          assertEquals(result.split, result.reference, where);
          assertEquals(result.split, bits(expected), `${where}: the exact value`);
        }
      }
    } finally {
      gpu.destroy();
    }
  });

  it("carries a NaN of one input row to every output of that row and no other, for every storage", async () => {
    const gpu = await acquireGpu();
    try {
      const run = splitKRunner(gpu);
      for (const storage of STORAGES) {
        const shape = { m: 37, n: 68, k: evenK() };
        const testCase = rawCase(storage, "irregular", BAND_GEOMETRIES[0], shape, 4);
        // The last element of row 21: read by the last slice only.
        testCase.x[21 * shape.k + shape.k - 1] = Number.NaN;
        const result = await run(testCase);
        const rows = nanRows(result.split, shape.n);
        assertEquals([...rows], [21], storageLabel(storage));
        const values = floats(result.split).subarray(21 * shape.n, 22 * shape.n);
        assert(
          values.every(Number.isNaN),
          `${storageLabel(storage)}: a column of row 21 is not NaN`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("deals the K tiles out in whole tiles: each slice's partial sums count its own terms", async () => {
    const gpu = await acquireGpu();
    try {
      const run = splitKRunner(gpu);
      // 269 tiles, the last of 12 terms, in 16 slices of 17 and 16 tiles; 63 tiles in 2 slices of
      // 32 and 31, the last of 8 terms; 32 tiles in 8 slices of 4; 2 tiles in 4 slices, two of them
      // empty (K smaller than the slices).
      for (const [k, slices] of [
        [4300, 16],
        [1000, 2],
        [512, 8],
        [20, 4],
      ] as const) {
        const shape = { m: 5, n: 8, k };
        const w = new Float32Array(shape.n * k).fill(1);
        const result = await run({
          geometry: BAND_GEOMETRIES[0],
          ...shape,
          slices,
          weight: { storage: "f32", words: new Uint32Array(w.buffer), scale: undefined, values: w },
          x: new Float32Array(shape.m * k).fill(1),
          bias: new Float32Array(shape.n),
        });
        const tiles = Math.ceil(k / 16);
        const per = Math.floor(tiles / slices);
        const extra = tiles % slices;
        let first = 0;
        let total = 0;
        for (let slice = 0; slice < slices; slice += 1) {
          const end = first + per + (slice < extra ? 1 : 0);
          const terms = Math.max(0, Math.min(end * 16, k) - first * 16);
          const counts = floats(result.partials).subarray(
            slice * shape.m * shape.n,
            (slice + 1) * shape.m * shape.n,
          );
          assert(
            counts.every((count) => count === terms),
            `K${k} s${slices}: slice ${slice} counts ${counts[0]}, not ${terms}`,
          );
          total += counts[0];
          first = end;
        }
        // The GPU's counts add up to K: no term dropped, none taken twice.
        assertEquals(total, k);
        assert(
          floats(result.split).every((value) => value === k),
          `K${k} s${slices}: the sums`,
        );
      }
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
