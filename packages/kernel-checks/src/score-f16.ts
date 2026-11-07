/**
 * Check 7 — the rounding of the f16 attention scores (`attentionScoreStorage: "f16"`, ADR 0031
 * addendum 2026-11-07). The store is `pack2x16float`, whose rounding WGSL leaves open: Vulkan on the
 * RTX 3080 Ti rounds to nearest with ties to even, and D3D12 on the RTX 5070 Ti rounds toward zero. The
 * contract is that a backend does one of the two on every value, so the check passes on either and
 * records which one it saw. Any other word fails, and so does a mix of the two.
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

type Rounding = "nearest" | "toward zero";

/** What the host stores in `rounding`, word for word. Column `c` of a row is its q element `c % D`. */
const expectedWords = (
  spec: ScoreCase,
  q: Float32Array<ArrayBuffer>,
  rounding: Rounding,
): Uint32Array<ArrayBuffer> => {
  const pair = new Float16Array(1);
  const half = new Uint16Array(pair.buffer);
  const words = new Uint32Array((spec.rows * spec.columns) / 2);
  for (let index = 0; index < spec.rows * spec.columns; index += 1) {
    const column = index % spec.columns;
    const value = q[Math.floor(index / spec.columns) * DEPTH + (column % DEPTH)];
    pair[0] = value;
    // One step down in magnitude: the values here are normal, so the bits step with the value.
    const bits = rounding === "toward zero" && Math.abs(pair[0]) > Math.abs(value) ? -1 : 0;
    words[index >> 1] += (index % 2 === 1 ? 2 ** 16 : 1) * (half[0] + bits);
  }
  return words;
};

const failuresAgainst = (
  rounding: Rounding,
  stored: readonly {
    readonly spec: ScoreCase;
    readonly q: Float32Array<ArrayBuffer>;
    readonly actual: Uint32Array<ArrayBuffer>;
  }[],
): number =>
  stored.reduce(
    (sum, { spec, q, actual }) =>
      sum + compareWords(rounding, actual, expectedWords(spec, q, rounding)).failures,
    0,
  );

export const SCORE_F16_CASES = { "score-f16-store": CASES.length };

const checkStore = async (
  gpu: GpuContext,
  options: CheckOptions,
  mutate?: (wgsl: string) => string,
): Promise<CheckPart> => {
  const stored = [];
  let roundedValues = 0;
  for (const spec of CASES) {
    throwIfAborted(options);
    if (!attentionScoreUsesF16(DEPTH, spec.columns)) throw new Error(`${spec.label}: not eligible`);
    const { q, k } = inputsOf(spec);
    roundedValues += q.filter((value) => Math.f16round(value) !== value).length;
    const shape = { rows: spec.rows, columns: spec.columns, depth: DEPTH };
    const actual = await scopedGpuWork(gpu, spec.label, () =>
      runAttentionQkS16(gpu.device, shape, q, k, mutate === undefined ? {} : { mutate }),
    );
    stored.push({ spec, q, actual });
    options.onProgress?.({ part: "score-f16-store", done: stored.length, total: CASES.length });
  }
  // One rounding for every case: a backend that mixes the two fails against the closer one.
  const rounding: Rounding =
    failuresAgainst("nearest", stored) <= failuresAgainst("toward zero", stored)
      ? "nearest"
      : "toward zero";
  const cases: CheckCase[] = stored.map(({ spec, q, actual }) => ({
    label: spec.label,
    criteria: [
      compareWords(
        `S words vs the host rounding ${rounding}`,
        actual,
        expectedWords(spec, q, rounding),
      ),
    ],
  }));
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
      {
        id: "rounding",
        name: "which rounding this backend's pack2x16float does",
        met: true,
        detail: rounding,
      },
    ],
  };
};

export const checkScoreF16Store = (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => checkStore(gpu, options);

const TOWARD_ZERO_WGSL = `fn toward_zero(v: f32) -> u32 {
  let f = bitcast<u32>(v);
  return ((f >> 16u) & 0x8000u) | (((f & 0x7fffffffu) - 0x38000000u) >> 13u);
}

fn score_store(`;

/**
 * The check over a store that rounds toward zero in WGSL, for a GPU whose `pack2x16float` rounds to
 * nearest: the other accepted result has to pass and be named without a D3D12 at hand.
 */
export const checkScoreF16TowardZeroStore = (gpu: GpuContext): Promise<CheckPart> =>
  checkStore(gpu, {}, (wgsl) =>
    wgsl
      .replace("fn score_store(", TOWARD_ZERO_WGSL)
      .replace("pack2x16float(value.xy)", "toward_zero(value.x) | (toward_zero(value.y) << 16u)")
      .replace("pack2x16float(value.zw)", "toward_zero(value.z) | (toward_zero(value.w) << 16u)"),
  );

const MUTANTS: readonly { readonly label: string; readonly from: string; readonly to: string }[] = [
  {
    label: "the store swaps the two halves of a word",
    from: "pack2x16float(value.xy)",
    to: "pack2x16float(value.yx)",
  },
  {
    label: "the store clears the low bit of both halves, which is neither rounding",
    from: "pack2x16float(value.zw)",
    to: "(pack2x16float(value.zw) & 0xfffefffeu)",
  },
];

/** The words each mutant differs in from the closer rounding (accepting two is not accepting all). */
export const scoreF16StoreMutants = async (
  gpu: GpuContext,
): Promise<readonly { readonly label: string; readonly differingWords: number }[]> => {
  const spec = CASES[0];
  const { q, k } = inputsOf(spec);
  const results: { readonly label: string; readonly differingWords: number }[] = [];
  for (const { label, from, to } of MUTANTS) {
    const actual = await runAttentionQkS16(
      gpu.device,
      { rows: spec.rows, columns: spec.columns, depth: DEPTH },
      q,
      k,
      {
        mutate: (wgsl) => {
          if (!wgsl.includes(from)) throw new Error(`mutant anchor is gone: ${label}`);
          return wgsl.replace(from, to);
        },
      },
    );
    const stored = [{ spec, q, actual }];
    results.push({
      label,
      differingWords: Math.min(
        failuresAgainst("nearest", stored),
        failuresAgainst("toward zero", stored),
      ),
    });
  }
  return results;
};
