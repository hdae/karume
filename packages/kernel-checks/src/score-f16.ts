/**
 * Check 7 — the rounding of the f16 attention scores (`attentionScoreStorage: "f16"`, ADR 0031). Its
 * contract is one rounding point at the store, equal to rounding on the host, and that store is
 * `pack2x16float`, whose rounding WGSL leaves open. So this is the check to run on each backend.
 *
 * ①QK runs on one-hot keys, so every S element is exactly one q value and the expected words need no
 * GPU reference. The f32 form of ①QK is enough for every S f16 kernel, a8 included, because they
 * all store through the same `score_store`.
 *
 * The values stay inside the finite normal range of f16: subnormals may flush and an out-of-range
 * value makes its whole word indeterminate.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { attentionScoreUsesF16 } from "../../runtime/src/kernels/score-storage.ts";
import { runAttentionQkS16 } from "../../runtime/tests/helpers/state-dispatch.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareWords,
  scopedGpuWork,
  throwIfAborted,
} from "./result.ts";

type ScoreCase = {
  readonly label: string;
  readonly rows: number;
  readonly columns: number;
  readonly values: readonly number[];
};

const DEPTH = 4;

/** Values around f16's rounding boundaries. No −0: an S element is a sum, and `−0 + 0` is `+0`. */
const BOUNDARY_VALUES: readonly number[] = [
  1,
  -1,
  1 + 2 ** -11, // tie → even side 1.0
  1 + 3 * 2 ** -11, // tie → even side 1 + 2^-9
  1 + 2 ** -11 + 2 ** -20, // above the tie
  -(1 + 2 ** -11),
  2 ** -14, // the smallest normal
  65504, // the largest finite
  65519, // below the overflow tie
  0.1,
  22.143,
  3.3,
];

/** Fixed pseudo-random values in (−8, 8): the same on every backend. */
const randomValues = (count: number): readonly number[] => {
  let state = 700_001;
  return Array.from({ length: count }, () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return Math.fround(state / 2 ** 29 - 8);
  });
};

const CASES: readonly ScoreCase[] = [
  { label: "boundary values", rows: 8, columns: 16, values: BOUNDARY_VALUES },
  { label: "random values", rows: 256, columns: 16, values: randomValues(256 * DEPTH) },
];

const inputsOf = (
  spec: ScoreCase,
): { readonly q: Float32Array<ArrayBuffer>; readonly k: Float32Array<ArrayBuffer> } => {
  const q = new Float32Array(spec.rows * DEPTH);
  for (let i = 0; i < q.length; i += 1) q[i] = spec.values[i % spec.values.length];
  const k = new Float32Array(spec.columns * DEPTH);
  for (let column = 0; column < spec.columns; column += 1) {
    k[column * DEPTH + (column % DEPTH)] = 1;
  }
  return { q, k };
};

/** What the host's rounding stores, word for word. Column `c` of a row is its q element `c % D`. */
const expectedWords = (spec: ScoreCase, q: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const pair = new Float16Array(1);
  const half = new Uint16Array(pair.buffer);
  const words = new Uint32Array((spec.rows * spec.columns) / 2);
  for (let index = 0; index < spec.rows * spec.columns; index += 1) {
    const column = index % spec.columns;
    pair[0] = q[Math.floor(index / spec.columns) * DEPTH + (column % DEPTH)];
    words[index >> 1] += index % 2 === 1 ? half[0] * 2 ** 16 : half[0];
  }
  return words;
};

export const SCORE_F16_CASES = { "score-f16-store": CASES.length };

export const checkScoreF16Store = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const cases: CheckCase[] = [];
  let roundedValues = 0;
  for (const spec of CASES) {
    throwIfAborted(options);
    if (!attentionScoreUsesF16(DEPTH, spec.columns)) throw new Error(`${spec.label}: not eligible`);
    const { q, k } = inputsOf(spec);
    roundedValues += q.filter((value) => Math.f16round(value) !== value).length;
    const shape = { rows: spec.rows, columns: spec.columns, depth: DEPTH };
    const actual = await scopedGpuWork(gpu, spec.label, () =>
      runAttentionQkS16(gpu.device, shape, q, k),
    );
    cases.push({
      label: spec.label,
      criteria: [compareWords("S words vs the host's rounding", actual, expectedWords(spec, q))],
    });
    options.onProgress?.({ part: "score-f16-store", done: cases.length, total: CASES.length });
  }
  return {
    id: "score-f16-store",
    title: "f16 attention scores vs the host's rounding (S words)",
    cases,
    requirements: [
      {
        id: "rounded",
        name: "some inputs actually round",
        met: roundedValues > 0,
        detail: `${roundedValues} rounded values`,
      },
    ],
  };
};

/** The words that differ when the store swaps the halves of a word (the comparison can fail). */
export const scoreF16StoreSwapMutant = async (gpu: GpuContext): Promise<number> => {
  const spec = CASES[0];
  const { q, k } = inputsOf(spec);
  const actual = await runAttentionQkS16(
    gpu.device,
    { rows: spec.rows, columns: spec.columns, depth: DEPTH },
    q,
    k,
    { mutate: (wgsl) => wgsl.replace("pack2x16float(value.xy)", "pack2x16float(value.yx)") },
  );
  return compareWords("mutant", actual, expectedWords(spec, q)).failures;
};
