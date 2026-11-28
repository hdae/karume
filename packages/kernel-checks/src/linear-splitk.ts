/**
 * The split-K linear (`src/kernels/linear-splitk.ts` — the split of the GEMM skeleton): the contract
 * class R check of ADR 0147 decision 2, for every weight storage the GEMM skeleton reads under f32
 * compute (f32, f16, bf16, i8, i4 at group lengths 16 / 32 / 128, i2), against the reference GEMM of
 * the same storage and tile geometry and an f64 evaluation over the weights as the kernels decode
 * them. Raw dispatches (`packages/runtime/tests/helpers/linear-splitk-dispatch.ts`); the weights are
 * `packages/runtime/tests/helpers/linear-storage-weights.ts`'s.
 *
 * Parts:
 *
 * - `linear-splitk-identity` (point 3): with one slice the two passes give the reference GEMM's
 *   words, on every tile geometry the linear GEMM takes (the row-count rules of the default profile
 *   and of every embedded one): vec4 for every storage, scalar for f32 on every geometry and for the
 *   other storages on the default one; two row and two column tiles with ragged edges and a partial
 *   last K tile where the storage allows.
 * - `linear-splitk-bands` (points 1 to 3): at 2, 4, 8 and 16 slices, a K whose tile count every count
 *   divides (vec4) and one it does not, with a partial last tile where the storage allows (scalar),
 *   on the default 128 × 128 geometry and the 16 × 16 one: every output within the op band
 *   (`GEMM_TOLERANCE`) of the f64 evaluation ({@link LINEAR_SPLITK_F64_BAND}); within
 *   {@link LINEAR_SPLITK_AB_BAND} times that band of the reference GEMM's output; a second dispatch
 *   writes the same words; nothing is written past the `m · n` outputs or the `slices · m · n` partial
 *   sums.
 * - `linear-splitk-exact` (point 3): on inputs whose products and sums are exact in f32 the order of
 *   the additions cannot matter, so the output equals the reference GEMM's words and the exact value —
 *   for every storage at every slice count, and for f32 on every tile geometry and skeleton form at 4
 *   and 16 slices (the slices' bases, which a one-slice run never reaches).
 * - `linear-splitk-nan` (point 3): a NaN at the last element of one row reaches every output of that
 *   row and no other, for every storage.
 * - `linear-splitk-slices` (point 3): on all-ones inputs each slice's partial sums count its own
 *   terms — the first and last slices' lengths, a K off the K tile, more slices than tiles — and the
 *   counts add up to K (no tile dropped, none taken twice).
 *
 * Point 4 (a fault injection that turns point 1 red) is not part of the check: the node GPU test
 * (packages/runtime/tests/gpu_linear_splitk_test.ts) runs it on a mutated split partial through
 * {@link LinearSplitKCheckOptions.mutate}.
 *
 * Not checked, on purpose: equality with the one-thread-per-output kernel the split replaced. It
 * held on every shape measured on one device, a fact of that driver's FMA contraction, not a
 * contract.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { gemmUsesVec4 } from "../../runtime/src/kernels/gemm.ts";
import {
  defaultGemmGeometry,
  type GemmGeometry,
  gemmTileM,
  gemmTileN,
} from "../../runtime/src/kernels/gemm-geometry.ts";
import { linearKey } from "../../runtime/src/kernels/linear.ts";
import { GEMM_TOLERANCE } from "../../runtime/src/reference/op-tolerance.ts";
import { linearGemmGeometries } from "../../runtime/tests/helpers/geometry.ts";
import {
  type SplitKDispatchCase,
  type SplitKDispatchRun,
  SPLITK_UNWRITTEN,
  splitKRunner,
} from "../../runtime/tests/helpers/linear-splitk-dispatch.ts";
import {
  encodeLinearWeight,
  type LinearWeightStorage,
  storageLabel,
  type WeightFixture,
} from "../../runtime/tests/helpers/linear-storage-weights.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareCount,
  compareFill,
  compareWords,
  type Criterion,
  type ExpectedCases,
  f32Word,
  scopedGpuWork,
  throwIfAborted,
  type WordFailure,
} from "./result.ts";

/**
 * The A/B band against the reference GEMM, as a multiple of the op band at the f64 value
 * (`GEMM_TOLERANCE.atol + rtol · |f64|`). A declaration (ADR 0147 decision 2: at least 4 × the worst
 * measured, at most 2 × the f64 band); the runtime's GPU test applies it through a Session too:
 *
 * | Case | Worst \|split − reference\| / op band |
 * | --- | --- |
 * | f32 through a Session, 3 × 512 × 5 / 16 × 1536 × 64 / 9 × 8960 × 37 (m × k × n) at 2 / 4 / 16 slices | 0.028 / 0.144 / 0.393 |
 * | every storage through a Session, 128 × 1536 × 64 at 4 slices | 0.122 (i4 g16) to 0.192 (i2) |
 * | this check's bands part (every storage, 2 to 16 slices, two geometries) | 0.390 (bf16, K 4096, 4 slices) |
 *
 * Measured 2026-11-28 on an RTX 3080 Ti (Dawn / Vulkan; the Session cases at the nvidia-ampere
 * profile's geometries) with the `irregular` fixtures: the worst is 0.393, so the band is at least
 * 1.57 and at most 2.
 */
export const LINEAR_SPLITK_AB_BAND = 2;

/** The name of point 1's criterion (the node test's fault injection reads it). */
export const LINEAR_SPLITK_F64_BAND = "split within the GEMM band of f64";

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

const F32: LinearWeightStorage = { storage: "f32" };
const SLICE_COUNTS = [2, 4, 8, 16] as const;
const DEFAULT = defaultGemmGeometry();
/** The 16 × 16 tile the GEMM takes up to 64 rows. */
const SKINNY: GemmGeometry = { regM: 1, regN: 4, wgX: 4, wgY: 16 };
const BAND_GEOMETRIES: readonly GemmGeometry[] = [DEFAULT, SKINNY];
/** 256 tiles: every slice count divides it. */
const EVEN_K = 4096;
/**
 * 269 tiles, the last of 12 terms; for i4 (K a multiple of the group) 264 tiles, which 16 slices do
 * not divide.
 */
const unevenK = ({ storage }: LinearWeightStorage): number => (storage === "i4" ? 4224 : 4300);

/** `(K, slices)` of the slice-length part, with the reason each is there. */
const SLICE_RUNS: readonly (readonly [number, number])[] = [
  // 269 tiles, the last of 12 terms: 13 slices of 17 tiles and 3 of 16.
  [4300, 16],
  // 63 tiles, the last of 8 terms: 32 and 31.
  [1000, 2],
  // 32 tiles: 8 slices of 4.
  [512, 8],
  // 2 tiles in 4 slices: two slices empty.
  [20, 4],
];

const sameGeometry = (a: GemmGeometry, b: GemmGeometry): boolean =>
  JSON.stringify(a) === JSON.stringify(b);

/** The shapes of the one-slice identity on one geometry for one storage. */
const identityShapes = (
  storage: LinearWeightStorage,
  geometry: GemmGeometry,
): readonly { readonly m: number; readonly n: number; readonly k: number }[] => {
  // A partial last K tile where the storage allows (i4's K is a multiple of its group).
  const k = storage.storage === "i4" ? 2 * storage.groupSize : 100;
  const m = gemmTileM(geometry) + 3;
  const scalar = storage.storage === "f32" || sameGeometry(geometry, DEFAULT);
  return [
    { m, n: gemmTileN(geometry) + 4, k },
    ...(scalar ? [{ m, n: gemmTileN(geometry) + 5, k }] : []),
  ];
};

const GEOMETRY_COUNT = linearGemmGeometries().length;

export const LINEAR_SPLITK_CASES = {
  // vec4 for every storage on every geometry, scalar for f32 on every geometry and for the other
  // seven on the default one.
  "linear-splitk-identity": GEOMETRY_COUNT * STORAGES.length + GEOMETRY_COUNT + STORAGES.length - 1,
  "linear-splitk-bands": STORAGES.length * 2 * BAND_GEOMETRIES.length * SLICE_COUNTS.length,
  "linear-splitk-exact": STORAGES.length * SLICE_COUNTS.length + GEOMETRY_COUNT * 2,
  "linear-splitk-nan": STORAGES.length,
  "linear-splitk-slices": SLICE_RUNS.length,
} as const satisfies ExpectedCases;

type Shape = { readonly m: number; readonly n: number; readonly k: number };

const activations = ({ m, k }: Shape, fixture: WeightFixture): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: m * k }, (_, i) =>
    fixture === "exact" ? ((i * 7) % 5) - 2 : Math.sin(i * 0.037) * 0.75,
  );

const biasOf = (n: number, fixture: WeightFixture): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: n }, (_, i) =>
    fixture === "exact" ? (i % 9) - 4 : ((i % 7) - 3) * 0.11,
  );

const caseOf = (
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

/** The op's meaning: the sum in f64 over the decoded weights, then the bias. */
const referenceF64 = ({
  geometry: _geometry,
  m,
  n,
  k,
  x,
  weight,
  bias,
}: SplitKDispatchCase): Float64Array<ArrayBuffer> => {
  const w = weight.values;
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

const floats = (words: Uint32Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  new Float32Array(words.buffer, words.byteOffset, words.length);

/** The reference op band around an f64 value (class R: the GEMM row of op-tolerance.ts). */
const band = (reference: number): number =>
  GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference);

/**
 * Every output finite and within `factor` × the band of `reference` around `against` (the f64
 * values themselves for point 1, the reference GEMM's outputs for point 2).
 */
const withinBand = (
  name: string,
  actual: Uint32Array<ArrayBuffer>,
  against: ArrayLike<number>,
  reference: Float64Array<ArrayBuffer>,
  factor: number,
): Criterion => {
  const values = floats(actual);
  let failures = 0;
  let worst: WordFailure | undefined;
  let worstError = -1;
  for (let i = 0; i < values.length; i++) {
    const error = Math.abs(values[i] - against[i]);
    const bound = factor * band(reference[i]);
    // A non-finite output fails whatever the error (NaN compares false).
    const finite = Number.isFinite(values[i]);
    if (finite && error <= bound) continue;
    failures += 1;
    const rank = finite ? error : Number.POSITIVE_INFINITY;
    if (rank > worstError) {
      worstError = rank;
      worst = {
        index: i,
        expected: f32Word(against[i]),
        actual: actual[i],
        at: "worst",
        note: `against ${against[i]}, got ${values[i]} (bound ${bound}, f64 ${reference[i]})`,
      };
    }
  }
  return worst === undefined
    ? { name, words: values.length, failures }
    : { name, words: values.length, failures, failure: worst };
};

/** Whether every word of a row range is a NaN (`nan`) or none is (`!nan`), one failure per word. */
const nanCriterion = (name: string, words: Uint32Array<ArrayBuffer>, nan: boolean): Criterion => {
  const values = floats(words);
  let failures = 0;
  let failure: WordFailure | undefined;
  for (let i = 0; i < values.length; i++) {
    if (Number.isNaN(values[i]) === nan) continue;
    failures += 1;
    failure ??= {
      index: i,
      expected: nan ? SPLITK_UNWRITTEN : words[i],
      actual: words[i],
      at: "first",
      note: nan ? "not NaN" : "NaN",
    };
  }
  return failure === undefined
    ? { name, words: values.length, failures }
    : { name, words: values.length, failures, failure };
};

/** The criteria every run shares: the reference wrote every output, nothing past them. */
const writtenCriteria = (run: SplitKDispatchRun, slices: number, outputs: number): Criterion[] => [
  {
    name: "the reference GEMM wrote every output",
    words: run.reference.length,
    failures: run.reference.filter((word) => word === SPLITK_UNWRITTEN).length,
  },
  compareFill("nothing written past the m × n outputs", run.tails, SPLITK_UNWRITTEN),
  compareFill(
    "nothing written past the slices × m × n partial sums",
    run.partials.subarray(slices * outputs),
    SPLITK_UNWRITTEN,
  ),
];

export type LinearSplitKCheckOptions = CheckOptions & {
  /**
   * Rewrites the split partial's WGSL before it is compiled (the node test's fault injection;
   * absent = the generator's own WGSL). A rewrite that changes nothing throws.
   */
  readonly mutate?: (wgsl: string) => string;
};

const label = (
  storage: LinearWeightStorage,
  geometry: GemmGeometry,
  { m, n, k }: Shape,
  slices: number,
): string =>
  `${storageLabel(storage)} ${m}×${n}×${k} ${linearKey(
    storage.storage,
    gemmUsesVec4(k, n),
    "f32",
    undefined,
    storage.groupSize,
    geometry,
  ).replace(/^linear:v\d+:f32:/, "")} s${slices}`;

/** Runs the check. Needs no optional feature. */
export const checkLinearSplitK = async (
  gpu: GpuContext,
  options: LinearSplitKCheckOptions = {},
): Promise<readonly CheckPart[]> => {
  const run = splitKRunner(gpu, { mutate: options.mutate });
  const dispatch = (where: string, testCase: SplitKDispatchCase): Promise<SplitKDispatchRun> =>
    scopedGpuWork(gpu, `linear split-K ${where}`, () => run(testCase));
  const parts: CheckPart[] = [];
  const part = async (
    id: keyof typeof LINEAR_SPLITK_CASES,
    title: string,
    body: (push: (checkCase: CheckCase) => void) => Promise<void>,
  ): Promise<void> => {
    const cases: CheckCase[] = [];
    const total = LINEAR_SPLITK_CASES[id];
    await body((checkCase) => {
      cases.push(checkCase);
      options.onProgress?.({ part: id, done: cases.length, total });
    });
    parts.push({ id, title, cases, requirements: [] });
  };

  await part(
    "linear-splitk-identity",
    "Split-K at one slice vs the reference GEMM, every geometry",
    async (push) => {
      for (const storage of STORAGES) {
        for (const geometry of linearGemmGeometries()) {
          for (const shape of identityShapes(storage, geometry)) {
            throwIfAborted(options);
            const where = label(storage, geometry, shape, 1);
            const result = await dispatch(where, caseOf(storage, "irregular", geometry, shape, 1));
            push({
              label: where,
              criteria: [
                compareWords(
                  "one slice: the reference GEMM's words (u32)",
                  result.split,
                  result.reference,
                ),
                ...writtenCriteria(result, 1, shape.m * shape.n),
              ],
            });
          }
        }
      }
    },
  );

  await part(
    "linear-splitk-bands",
    "Split-K at 2 to 16 slices: f64 band, A/B against the reference GEMM, structure",
    async (push) => {
      for (const storage of STORAGES) {
        for (const shape of [
          { m: 37, n: 68, k: EVEN_K },
          { m: 37, n: 69, k: unevenK(storage) },
        ]) {
          const base = caseOf(storage, "irregular", DEFAULT, shape, 1);
          const reference = referenceF64(base);
          for (const geometry of BAND_GEOMETRIES) {
            for (const slices of SLICE_COUNTS) {
              throwIfAborted(options);
              const where = label(storage, geometry, shape, slices);
              const result = await dispatch(where, { ...base, geometry, slices });
              push({
                label: where,
                criteria: [
                  withinBand(LINEAR_SPLITK_F64_BAND, result.split, reference, reference, 1),
                  withinBand(
                    `split within ${LINEAR_SPLITK_AB_BAND} × the f64 band of the reference GEMM`,
                    result.split,
                    floats(result.reference),
                    reference,
                    LINEAR_SPLITK_AB_BAND,
                  ),
                  compareWords(
                    "second dispatch of the same input (u32)",
                    result.again,
                    result.split,
                  ),
                  ...writtenCriteria(result, slices, shape.m * shape.n),
                ],
              });
            }
          }
        }
      }
    },
  );

  await part(
    "linear-splitk-exact",
    "Split-K on exact sums: the reference GEMM's words and the exact value",
    async (push) => {
      const exactCase = async (
        storage: LinearWeightStorage,
        geometry: GemmGeometry,
        shape: Shape,
        slices: number,
      ): Promise<void> => {
        throwIfAborted(options);
        const testCase = caseOf(storage, "exact", geometry, shape, slices);
        const expected = new Uint32Array(Float32Array.from(referenceF64(testCase)).buffer);
        const where = label(storage, geometry, shape, slices);
        const result = await dispatch(where, testCase);
        push({
          label: where,
          criteria: [
            compareWords(
              "exact sums: the reference GEMM's words (u32)",
              result.split,
              result.reference,
            ),
            compareWords("exact sums: the exact value (u32)", result.split, expected),
          ],
        });
      };
      for (const storage of STORAGES) {
        for (const slices of SLICE_COUNTS) {
          await exactCase(storage, DEFAULT, { m: 37, n: 69, k: unevenK(storage) }, slices);
        }
      }
      // vec4, so the skeleton forms apply; the slices' bases are what one slice never reaches.
      for (const geometry of linearGemmGeometries()) {
        const shape = { m: gemmTileM(geometry) + 3, n: gemmTileN(geometry) + 4, k: 4300 };
        for (const slices of [4, 16]) await exactCase(F32, geometry, shape, slices);
      }
    },
  );

  await part("linear-splitk-nan", "Split-K: a NaN stays in its row", async (push) => {
    const shape = { m: 37, n: 68, k: EVEN_K };
    const poisoned = 21;
    for (const storage of STORAGES) {
      throwIfAborted(options);
      const testCase = caseOf(storage, "irregular", DEFAULT, shape, 4);
      // The last element of the row: read by the last slice only.
      testCase.x[poisoned * shape.k + shape.k - 1] = Number.NaN;
      const where = label(storage, DEFAULT, shape, 4);
      const result = await dispatch(where, testCase);
      const row = poisoned * shape.n;
      push({
        label: where,
        criteria: [
          nanCriterion(
            "a NaN in x reaches every output of its row",
            result.split.subarray(row, row + shape.n),
            true,
          ),
          nanCriterion(
            "and no output of another row",
            Uint32Array.from([
              ...result.split.subarray(0, row),
              ...result.split.subarray(row + shape.n),
            ]),
            false,
          ),
        ],
      });
    }
  });

  await part(
    "linear-splitk-slices",
    "Split-K: each slice's partial sums count its own terms",
    async (push) => {
      for (const [k, slices] of SLICE_RUNS) {
        throwIfAborted(options);
        const shape = { m: 5, n: 8, k };
        const w = new Float32Array(shape.n * k).fill(1);
        const where = `f32 ${shape.m}×${shape.n}×${k} s${slices} (all ones)`;
        const result = await dispatch(where, {
          geometry: DEFAULT,
          ...shape,
          slices,
          weight: { storage: "f32", words: new Uint32Array(w.buffer), scale: undefined, values: w },
          x: new Float32Array(shape.m * k).fill(1),
          bias: new Float32Array(shape.n),
        });
        // Slice z takes tiles / slices tiles, and one more for the first tiles % slices slices.
        const tiles = Math.ceil(k / 16);
        const outputs = shape.m * shape.n;
        const expected = new Float32Array(slices * outputs);
        let first = 0;
        for (let slice = 0; slice < slices; slice += 1) {
          const end = first + Math.floor(tiles / slices) + (slice < tiles % slices ? 1 : 0);
          expected.fill(
            Math.max(0, Math.min(end * 16, k) - first * 16),
            slice * outputs,
            (slice + 1) * outputs,
          );
          first = end;
        }
        // The GPU's own counts of output 0, summed over the slices.
        const counted = floats(result.partials);
        let total = 0;
        for (let slice = 0; slice < slices; slice += 1) total += counted[slice * outputs];
        push({
          label: where,
          criteria: [
            compareWords(
              "each slice's partial sums count its terms (u32)",
              result.partials.subarray(0, slices * outputs),
              new Uint32Array(expected.buffer),
            ),
            compareCount("the slices' counts add up to K", total, k),
            compareWords(
              "the sums are K (u32)",
              result.split,
              new Uint32Array(new Float32Array(outputs).fill(k).buffer),
            ),
          ],
        });
      }
    },
  );
  return parts;
};
