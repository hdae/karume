/**
 * Reduction from a sequence to a vector (left on the host — ADR 0047 decision 4).
 *
 * Needed in 2 places: the mean token prepended to the output of the `speaker` graph (upstream
 * `_prepend_masked_mean_token`), and `caption_vec` of `duration` (the masked mean of the
 * `caption_norm`-ed sequence — upstream `DurationPredictor._caption_vec`).
 *
 * Both are **called only on the B=1, all-True-mask path** (no reference / an empty caption is
 * short-circuited by the host placing zeros), so upstream's `sum(x·mask)/clamp_min(sum(mask),1)`
 * reduces to a plain mean.
 *
 * MUST: f32 rounding is applied one operation at a time (a sequential sum per column).
 */

const f32 = Math.fround;

/** Mean over rows `[width]`. */
export const rowMean = (
  state: Float32Array<ArrayBuffer>,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  if (state.length !== rows * width) {
    throw new Error(`系列の長さ ${state.length} が ${rows}×${width} と違う`);
  }
  if (rows < 1) throw new Error("系列が空（行平均を作れない）");
  const out = new Float32Array(width);
  for (let column = 0; column < width; column += 1) {
    let sum = 0;
    for (let row = 0; row < rows; row += 1) sum = f32(sum + state[row * width + column]);
    out[column] = f32(sum / rows);
  }
  return out;
};

/**
 * `[(rows+1)·width]` with the mean token prepended at the start.
 *
 * MUST: prepended, not appended — `speaker_vec` of `duration` takes `speaker_state[:,0]` (= this
 * mean token) by contract, and if the position shifts, another row becomes the speaker vector.
 */
export const prependMeanToken = (
  state: Float32Array<ArrayBuffer>,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  const mean = rowMean(state, rows, width);
  const out = new Float32Array((rows + 1) * width);
  out.set(mean, 0);
  out.set(state, width);
  return out;
};
