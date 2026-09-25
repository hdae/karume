/**
 * スタイルベクトルと話者埋め込みの**行引き**（配布形の 2 つの表 → front / voice の入力）。
 *
 * 配布形は「表を配って実行時に行を引く」形で、名前 → 行番号の対応は `pipelineConfig` の
 * `styles` / `speakers` が持つ（ADR 0038 §2 / `tools/export-recipes/sbv2/distribution.py` —
 * 表 2 本と pipelineConfig の 3 つで 1 組）。
 *
 * ## MUST: 行がずれても shape は合う
 *
 * スタイル ID も話者 ID も**表の物理行そのもの**なので、ずれてもロードも実行も通り、
 * **別のスタイル・別の話者の声が出る**だけで沈黙する。行番号の妥当性（順列であること）は
 * `config.ts` が、表の行数・列数との突合は呼び出し側（`pipeline.ts` がグラフ入力の静的次元と
 * 突き合わせる）が受け持つ。
 *
 * ## 参照実装との対応
 *
 * {@link styleVector} は `TTSModel.__get_style_vector` と同式（正本は
 * `tools/export-recipes/sbv2/demo.py` の `style_vector`）で、**平均スタイルは行 0** という SBV2 の
 * 規約に乗っている。`weight = 1` なら選んだ行そのもの、`0` なら平均行に縮退する。
 */

import { parseSafetensors, type Tensor } from "@karume/runtime";
import { Sbv2InputError } from "./errors.ts";

/**
 * 配布形の safetensors に入っている表のテンソル名（`tools/export-recipes/sbv2/distribution.py` の
 * `SBV2_STYLE_KEY` / `SBV2_SPEAKER_KEY` と対）。
 */
export const STYLE_TENSOR = "style_vectors";
export const SPEAKER_TENSOR = "speaker_embeddings";

/** 2 次元の f32 表（行 = スタイル / 話者、列 = 特徴）。 */
export type Sbv2Table = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly rows: number;
  readonly cols: number;
};

/**
 * 1 テンソルだけを持つ safetensors を `[rows, cols]` の f32 表として読む。
 *
 * 資産は外部境界なので、テンソル名・dtype・次元数を全て検査してから使う（壊れた表は
 * 「行を引けない」ではなく「別の行を引く」形で沈黙しうる）。
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

const assertRow = (rows: number, cols: number, index: number, data: Float32Array): void => {
  if (data.length !== rows * cols) {
    throw new Error(`表の要素数 ${data.length} が ${rows}×${cols} と違う`);
  }
  if (!Number.isInteger(index) || index < 0 || index >= rows) {
    throw new Error(`行番号 ${index} が表の範囲外（0..${rows - 1}）`);
  }
};

/**
 * `styleWeight` の受理集合（**有限の数**）を見る門。入力起因なので型は `Sbv2InputError`
 * （`errors.ts` の分類軸 — NaN / Inf は呼び手が値を直せば通る）。
 *
 * NOTE: `export` は**生成の入口**（`pipeline.ts` の `synthesizeSbv2`）が同じ集合で先に落とす
 * ため。表引きに入るのはテキスト層を抜けた後なので、ここだけに検査があると打ち間違いが
 * text_encoder を 1 run 払ってから落ちる。受理集合の所有者はこの 1 本で、入口は条件を写さずに
 * これを呼ぶ（両側に条件を書くと必ず割れる）。
 */
export const assertFiniteStyleWeight = (weight: number): void => {
  if (!Number.isFinite(weight)) throw new Sbv2InputError(`styleWeight ${weight} が有限の数でない`);
};

/**
 * スタイルベクトル `[1, cols]` を作る（`mean + (picked − mean) · weight`）。
 *
 * MUST: `mean` は**行 0**（SBV2 の規約 — `style_vectors.npy` の先頭行が平均スタイル）。
 * 平均を列ごとに計算し直す実装にすると、weight ≠ 1 のときだけ静かに別のベクトルになる。
 */
export const styleVector = (
  table: Float32Array,
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
 * 話者埋め込み `g` を `[1, cols, 1]` で作る。front / voice はどちらもこの形を入力に取る
 * （末尾の 1 は時間軸 — conv1d の broadcast 相手）。
 */
export const speakerEmbedding = (
  table: Float32Array,
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
