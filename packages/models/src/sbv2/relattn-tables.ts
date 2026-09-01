/**
 * 窓付き相対位置注意の `(T, T)` 表を実長で生成するホスト実装（SBV2 の flow / voice 用）。
 *
 * ## なぜ runtime ではなくここにあるか
 *
 * この表は **SBV2 固有**（というより「窓付き相対位置注意を gather 化したモデル」固有）で、
 * `@karume/runtime` の語彙ではない。runtime は「IR を実行する汎用ランタイム」で、特定
 * モデルの入力の作り方を知ってはいけない（ADR 0008 の公開面の薄さ）。**利用者はパイプライン
 * （`pipeline.ts` が T ごとに呼ぶ）と検証（`packages/models/tests/`）の 2 つ**で、正本はモデル側
 * の知識を持つ models パッケージに置く（両者ともここを直接 import する — 実装を写した薄い面は
 * 作らない）。
 *
 * ## なぜホストが表を作るのか
 *
 * front（P ≤ 512）は同じ式をグラフ内に持ち、エクスポータの定数畳み込みが Pmax で焼き込む。
 * flow / voice は T ≤ 4096 で表が O(T²) になり、焼き込むと定数だけで 134MB になるため
 * **グラフ入力へ昇格**した（ADR 0013）。その結果、表の値を作る責務がホスト側に移る。
 *
 * ## MUST: Python 側とバイト一致であること
 *
 * 正本は `tools/export-recipes/sbv2/patch.py` の `build_relattn_tables`。式が
 * 割れても **shape は合う**ので、モデルは落ちずに黙って別の埋め込みを読む（沈黙誤値）。
 * `packages/models/tests/sbv2_relattn_parity_test.ts` が golden io に格納された Python 生成の
 * 表とバイト一致を実データで固定する。
 */

import type { Tensor } from "@karume/runtime";

/**
 * 相対位置注意の窓幅。SBV2 の enc_p / flow は 4（埋め込みは `2w+1 = 9` 行）。
 *
 * MUST（沈黙誤値クラス）: この値がモデルの `window_size` とずれると、`idx_k` は
 * `clamp(rel+4, 0, 8)` のまま**幅の違う埋め込みを gather する**。要素数の辻褄は合うので
 * shape エラーにならず、出力だけが静かに誤る。Python 側は `export_sbv2._assert_window_size`
 * が ckpt ロード時に落とし、ホスト側はパリティテストが**モデルコンテナに焼き込まれた
 * value 側の表の幅**（`2w+1`）と突き合わせて落とす — 両側に門を置くのは、片側だけだと
 * 「ホストとゴールデンが同じ誤りを共有して検証をすり抜ける」経路が残るため。
 */
export const RELATTN_WINDOW_SIZE = 4;

/** `buildRelattnTables` の返り値（グラフ入力 `idx_k` / `valid` にそのまま渡せる形）。 */
type RelattnTables = {
  /** `idx_k[i][j] = clamp(w + j − i, 0, 2w)` — key 側の埋め込み添字。 */
  readonly idxK: Tensor;
  /** `valid[i][j] = |j − i| ≤ w ? 1 : 0` — 窓外を落とす 0/1 マスク。 */
  readonly valid: Tensor;
};

/**
 * 実長 `length` の相対位置表を作る。
 *
 * `idx_k` は clamp 済みなので窓外でも**範囲内の別の埋め込み**を指す（gather の範囲外規約に
 * 依存しない形 — ADR 0013）。窓外の寄与を消すのは `valid` の乗算で、この 2 本は必ず対で
 * 使う。
 */
export const buildRelattnTables = (
  length: number,
  windowSize: number = RELATTN_WINDOW_SIZE,
): RelattnTables => {
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError(`相対位置表の長さ ${length} が 1 以上の整数でない`);
  }
  if (!Number.isInteger(windowSize) || windowSize < 0) {
    throw new RangeError(`窓幅 ${windowSize} が 0 以上の整数でない`);
  }
  const idxK = new Int32Array(length * length);
  const valid = new Float32Array(length * length);
  const upper = 2 * windowSize;
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const rel = j - i;
      const shifted = rel + windowSize;
      idxK[i * length + j] = shifted < 0 ? 0 : shifted > upper ? upper : shifted;
      valid[i * length + j] = rel >= -windowSize && rel <= windowSize ? 1 : 0;
    }
  }
  const shape = [length, length] as const;
  return {
    idxK: { dtype: "i32", shape: [...shape], data: idxK },
    valid: { dtype: "f32", shape: [...shape], data: valid },
  };
};
