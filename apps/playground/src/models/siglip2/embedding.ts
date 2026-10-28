/**
 * What the page shows and records of an image embedding: its statistics, the cosine similarity of
 * two embeddings and the raw bytes the sha256 is taken of. Pure, so `vp test` covers it.
 *
 * `Siglip2Pipeline.embed` returns `pooler_output` itself (not L2-normalized — the pipeline's module
 * doc), so the norm is a property of the model's output worth recording, and the cosine normalizes
 * on its own.
 */

/** How many leading values the page shows and the record keeps. */
export const FIRST_VALUES = 8;

/**
 * A value as the record keeps it. JSON has no NaN / ±Infinity (`JSON.stringify` writes `null` for
 * all three), so a non-finite value is spelled as a string — the record must tell them apart, since
 * flagging them is what `nonFinite` is for.
 */
export type RecordedValue = number | "NaN" | "Infinity" | "-Infinity";

/** Statistics of one embedding. `summary` is absent when non-finite values make it meaningless. */
export type EmbeddingStats = {
  readonly dimension: number;
  readonly nonFinite: number;
  /** The first {@link FIRST_VALUES} values, at 6 decimals (non-finite ones as {@link RecordedValue} strings). */
  readonly first: readonly RecordedValue[];
  readonly summary?: {
    readonly l2Norm: number;
    readonly min: number;
    readonly max: number;
    readonly mean: number;
  };
};

/** Six decimals: readable, diff-friendly, and far finer than the run-to-run noise of a GPU. */
const sixDecimals = (value: number): number => Number(value.toFixed(6));

const recordedValue = (value: number): RecordedValue => {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  return sixDecimals(value);
};

export const embeddingStats = (values: Float32Array): EmbeddingStats => {
  let nonFinite = 0;
  let squares = 0;
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      nonFinite += 1;
      continue;
    }
    squares += value * value;
    sum += value;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const first = Array.from(values.subarray(0, FIRST_VALUES), recordedValue);
  if (nonFinite > 0 || values.length === 0) {
    return { dimension: values.length, nonFinite, first };
  }
  return {
    dimension: values.length,
    nonFinite,
    first,
    summary: {
      l2Norm: sixDecimals(Math.sqrt(squares)),
      min: sixDecimals(min),
      max: sixDecimals(max),
      mean: sixDecimals(sum / values.length),
    },
  };
};

/**
 * The cosine similarity of two embeddings, accumulated in f64. Fails loudly on a length mismatch
 * (embeddings of two different models), on non-finite values and on a zero vector — each would
 * otherwise come out as a plausible-looking number or NaN.
 */
export const cosineSimilarity = (left: Float32Array, right: Float32Array): number => {
  if (left.length !== right.length) {
    throw new RangeError(
      `Cannot compare embeddings of different lengths (${left.length} and ${right.length})`,
    );
  }
  let dot = 0;
  let leftSquares = 0;
  let rightSquares = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      throw new RangeError(`Element ${index} is not finite (${a}, ${b})`);
    }
    dot += a * b;
    leftSquares += a * a;
    rightSquares += b * b;
  }
  if (leftSquares === 0 || rightSquares === 0) {
    throw new RangeError("Cannot take the cosine of a zero vector");
  }
  return dot / Math.sqrt(leftSquares * rightSquares);
};
