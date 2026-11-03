/**
 * **CPU reference** of the integer GEMV (src/kernels/linear-gemv-integer.ts): the mirror its
 * mirror-class (E) contract is checked against, word for word.
 *
 * The group sums are exact on both sides (the kernel's lane sums stay at or below 2^24 in f32 and
 * meet in i32, and its per-step `round(255 · s)` recovers each step's exact integer), so the reference
 * adds plain JS integers in ascending k. The roundings of the epilogue are each `Math.fround` of a
 * correctly rounded IEEE operation — the kernel separates them with its run-time rounding barrier, so
 * no fused multiply-add can merge two of them.
 *
 * MUST: do not copy the kernel's lane / row / plane / group-loop structure here. A plain loop over
 * (row, channel, group, k) is what keeps the work assignment and the byte-plane decode under
 * verification.
 */

import type { IntegerGemvWeight } from "../kernels/linear-gemv-integer.ts";
import { referenceStaticQuantize } from "./ops.ts";

export type IntegerGemvReferenceInput = {
  readonly weight: IntegerGemvWeight;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /** The packed int8 activation codes: `m · k / 4` words, 4 codes per word, low byte first. */
  readonly codes: Uint32Array<ArrayBuffer>;
  readonly xScale: number;
  /** The weight words exactly as bound (row-major, low bits first). */
  readonly weights: Uint32Array<ArrayBuffer>;
  /** One scale per output channel (i4: per channel and group, row-major). */
  readonly wscale: Float32Array<ArrayBuffer>;
  readonly bias: Float32Array<ArrayBuffer>;
  /** The output-side SRQ scale; absent = plain f32 output. */
  readonly outputScale?: number;
};

const activationCode = (codes: Uint32Array<ArrayBuffer>, index: number): number =>
  ((codes[index >> 2] >>> ((index & 3) * 8)) << 24) >> 24;

/** The signed weight value of flat element `index`: i2 stored − 2, i4 stored − 8, i8 as stored. */
const weightValue = (
  storage: IntegerGemvWeight["storage"],
  weights: Uint32Array<ArrayBuffer>,
  index: number,
): number => {
  if (storage === "i2") return ((weights[index >> 4] >>> ((index & 15) * 2)) & 3) - 2;
  if (storage === "i4") return ((weights[index >> 3] >>> ((index & 7) * 4)) & 15) - 8;
  return ((weights[index >> 2] >>> ((index & 3) * 8)) << 24) >> 24;
};

/** The output words (f32 bits, or the SRQ result's bits with `outputScale`), `m · n`. */
export const referenceIntegerGemv = (
  input: IntegerGemvReferenceInput,
): Uint32Array<ArrayBuffer> => {
  const { weight, m, n, k, codes, xScale, weights, wscale, bias, outputScale } = input;
  const [groups, length] = weight.storage === "i4" ? [weight.groups, weight.groupSize] : [1, k];
  const values = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let total = 0;
      for (let g = 0; g < groups; g++) {
        let sum = 0;
        for (let e = g * length; e < (g + 1) * length; e++) {
          sum +=
            weightValue(weight.storage, weights, col * k + e) * activationCode(codes, row * k + e);
        }
        total = Math.fround(total + Math.fround(Math.fround(sum) * wscale[col * groups + g]));
      }
      const scaled = Math.fround(total * xScale);
      values[row * n + col] = Math.fround(scaled + bias[col]);
    }
  }
  const result =
    outputScale === undefined
      ? values
      : referenceStaticQuantize(
          { dtype: "f32", shape: [m, n], data: values },
          { scale: outputScale },
        ).data;
  const bits = new Uint32Array(m * n);
  bits.set(new Uint32Array(result.buffer, result.byteOffset, result.length));
  return bits;
};
