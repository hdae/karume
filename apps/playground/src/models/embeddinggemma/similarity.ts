/**
 * What the page shows and records of the embeddings of one run: per vector its statistics, and the
 * similarity of every pair of rows. Pure, so `vp test` covers it.
 *
 * `embeddinggemma.Pipeline.embed` returns a unit vector (also at a shorter width — the pipeline
 * re-normalises the leading slice), so the cosine similarity of two results is their dot product;
 * the page takes the dot product and records each vector's L2 norm, which shows whether the unit
 * length held.
 */

/** How many leading values the page shows and the record keeps. */
export const FIRST_VALUES = 8;

/**
 * A value as the record keeps it. JSON has no NaN / ±Infinity (`JSON.stringify` writes `null` for
 * all three), so a non-finite value is spelled as a string.
 */
export type RecordedValue = number | "NaN" | "Infinity" | "-Infinity";

/** Statistics of one embedding. `l2Norm` is null when non-finite values make it meaningless. */
export type EmbeddingStats = {
  readonly dimension: number;
  readonly nonFinite: number;
  readonly l2Norm: number | null;
  /** The first {@link FIRST_VALUES} values, at 6 decimals. */
  readonly first: readonly RecordedValue[];
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
  for (const value of values) {
    if (Number.isFinite(value)) squares += value * value;
    else nonFinite += 1;
  }
  return {
    dimension: values.length,
    nonFinite,
    l2Norm: nonFinite > 0 ? null : sixDecimals(Math.sqrt(squares)),
    first: Array.from(values.subarray(0, FIRST_VALUES), recordedValue),
  };
};

/**
 * The dot product of two embeddings, accumulated in f64 — their cosine similarity, since the
 * pipeline returns unit vectors. Fails loudly on a length mismatch (two different widths) and on
 * non-finite values, which would otherwise come out as a plausible-looking number or NaN.
 */
export const dotProduct = (left: Float32Array, right: Float32Array): number => {
  if (left.length !== right.length) {
    throw new RangeError(
      `Cannot compare embeddings of different lengths (${left.length} and ${right.length})`,
    );
  }
  let dot = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      throw new RangeError(`Element ${index} is not finite (${a}, ${b})`);
    }
    dot += a * b;
  }
  return dot;
};

/**
 * The similarity of every pair of rows at 6 decimals (`matrix[i][j]`; the diagonal is each vector
 * with itself, about 1). A cell is null when either row has no embedding (the pipeline refused it)
 * or holds non-finite values — the row's own statistics say which. A length mismatch still throws:
 * one run embeds every row at one width, so a mismatch is the page's bug, not a value to show.
 */
export const similarityMatrix = (
  vectors: readonly (Float32Array | undefined)[],
): readonly (readonly (number | null)[])[] => {
  const comparable = vectors.map((vector) =>
    vector === undefined || !vector.every(Number.isFinite) ? undefined : vector,
  );
  return comparable.map((left) =>
    comparable.map((right) =>
      left === undefined || right === undefined ? null : sixDecimals(dotProduct(left, right)),
    ),
  );
};
