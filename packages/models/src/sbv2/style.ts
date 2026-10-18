/**
 * **Row lookup** of style vectors and speaker embeddings (the distribution's 2 tables → inputs of
 * front / voice).
 *
 * The distribution takes the form "ship the tables and pick a row at run time", and the name → row
 * number mapping is held by `pipelineConfig`'s `styles` / `speakers` (ADR 0038 §2 /
 * `tools/export-recipes/sbv2/distribution.py` — the 2 tables and pipelineConfig form one set of 3).
 *
 * ## MUST: when rows shift, the shapes still match
 *
 * Style IDs and speaker IDs are both **the physical rows of the table**, so a shift still loads and
 * runs, and stays silent with only **the voice of a different style / different speaker** coming
 * out. The validity of the row numbers (being a permutation) is owned by `config.ts`, and the
 * comparison with the tables' row and column counts by the caller (`pipeline.ts` compares them
 * with the static dimensions of the graph inputs).
 *
 * ## Correspondence with the reference implementation
 *
 * {@link styleVector} is the same formula as `TTSModel.__get_style_vector` (canonical:
 * `style_vector` in `tools/export-recipes/sbv2/demo.py`) and relies on SBV2's convention that **the
 * mean style is row 0**. `weight = 1` gives the selected row itself, `0` degenerates to the mean
 * row.
 */

import { parseSafetensors, type Tensor } from "@karume/runtime";
import { Sbv2InputError } from "./errors.ts";

/**
 * Tensor names of the tables inside the distribution's safetensors (paired with `SBV2_STYLE_KEY` /
 * `SBV2_SPEAKER_KEY` in `tools/export-recipes/sbv2/distribution.py`).
 */
export const STYLE_TENSOR = "style_vectors";
export const SPEAKER_TENSOR = "speaker_embeddings";

/** A 2-D f32 table (rows = styles / speakers, columns = features). */
export type Sbv2Table = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly rows: number;
  readonly cols: number;
};

/**
 * Reads a safetensors file holding exactly one tensor as a `[rows, cols]` f32 table.
 *
 * Assets are an external boundary, so the tensor name, dtype and rank are all checked before use (a
 * broken table can stay silent not as "cannot look up a row" but as "looks up a different row").
 */
export const parseSbv2Table = (buffer: ArrayBuffer, name: string): Sbv2Table => {
  const file = parseSafetensors(buffer);
  const view = file.tensors.get(name);
  if (view === undefined) {
    throw new Error(
      `資産にテンソル '${name}' が無い（入っているもの: ${[...file.tensors.keys()].join(" / ")}）`,
    );
  }
  if (view.dtype !== "F32") throw new Error(`テンソル '${name}' が F32 でない（${view.dtype}）`);
  if (view.shape.length !== 2) {
    throw new Error(`テンソル '${name}' の形 [${view.shape}] が [行, 列] の 2 次元でない`);
  }
  const [rows, cols] = view.shape;
  return {
    data: new Float32Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)),
    rows,
    cols,
  };
};

const assertRow = (
  rows: number,
  cols: number,
  index: number,
  data: Float32Array<ArrayBuffer>,
): void => {
  if (data.length !== rows * cols) {
    throw new Error(`表の要素数 ${data.length} が ${rows}×${cols} と違う`);
  }
  if (!Number.isInteger(index) || index < 0 || index >= rows) {
    throw new Error(`行番号 ${index} が表の範囲外（0..${rows - 1}）`);
  }
};

/**
 * Check of the acceptance set of `styleWeight` (**a finite number**). It is input-caused, so the
 * type is `Sbv2InputError` (the classification axis of `errors.ts` — NaN / Inf pass once the
 * caller fixes the value).
 *
 * NOTE: it is `export`ed because **the generation entry** (`synthesizeSbv2` in `pipeline.ts`) fails
 * early on the same set. The table lookup comes only after the text layer, so with the check only
 * here a typo would fail after paying one text_encoder run. This is the one owner of the acceptance
 * set, and the entry calls it without copying the condition (writing the condition on both sides is
 * bound to split).
 */
export const assertFiniteStyleWeight = (weight: number): void => {
  if (!Number.isFinite(weight)) throw new Sbv2InputError(`styleWeight ${weight} が有限の数でない`);
};

/**
 * Builds the style vector `[1, cols]` (`mean + (picked − mean) · weight`).
 *
 * MUST: `mean` is **row 0** (SBV2's convention — the first row of `style_vectors.npy` is the mean
 * style). An implementation that recomputes the mean per column silently produces a different
 * vector, only when weight ≠ 1.
 */
export const styleVector = (
  table: Float32Array<ArrayBuffer>,
  rows: number,
  cols: number,
  index: number,
  weight: number,
): Tensor => {
  assertRow(rows, cols, index, table);
  assertFiniteStyleWeight(weight);
  const picked = index * cols;
  const data = new Float32Array(cols);
  for (let i = 0; i < cols; i += 1) {
    const mean = table[i];
    data[i] = mean + (table[picked + i] - mean) * weight;
  }
  return { dtype: "f32", shape: [1, cols], data };
};

/**
 * Builds the speaker embedding `g` as `[1, cols, 1]`. front and voice both take this shape as input
 * (the trailing 1 is the time axis — the broadcast partner of conv1d).
 */
export const speakerEmbedding = (
  table: Float32Array<ArrayBuffer>,
  rows: number,
  cols: number,
  index: number,
): Tensor => {
  assertRow(rows, cols, index, table);
  const begin = index * cols;
  const data = new Float32Array(cols);
  data.set(table.subarray(begin, begin + cols));
  return { dtype: "f32", shape: [1, cols, 1], data };
};
