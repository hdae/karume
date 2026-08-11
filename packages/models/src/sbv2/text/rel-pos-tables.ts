/**
 * DeBERTa-v2 の disentangled attention が使う相対位置の添字表を実長で生成するホスト実装。
 *
 * ## なぜホストが表を作るのか
 *
 * この 2 表は T だけで決まる定数なので、エクスポータの定数畳み込みは Tmax = 512 で
 * `[1,512,512]` の i32 を **2 本焼き込んでいた**（実測 2,097,152 B = 配布形の 0.68%）。中身は
 * Toeplitz でしかも互いに転置なので、実体は 1,023 要素の i32 ベクトルに等しい。flow / voice が
 * 同じ理由で表をグラフ入力へ昇格した前例（ADR 0013）に倣って外部供給へ移した（ADR 0044 波 3）。
 *
 * ## MUST: Python 側とバイト一致であること
 *
 * 正本は `tools/exporter/karume/patch_deberta.py` の `build_rel_pos_tables`。式が割れても
 * **shape は合う**ので、モデルは落ちずに黙って別の位置埋め込みを gather する（沈黙誤値）。
 * `packages/models/tests/sbv2_rel_pos_parity_test.ts` が golden io に格納された Python 生成の表と
 * バイト一致を実データで固定する。
 *
 * 元の式は transformers の `make_log_bucket_position` で、torch は **float32** で計算する。
 * ホスト側の float64 と一致することは実測で確認済み（T = 11 / 35 / 128 / 300 / 512 の全点で
 * 不一致 0 — ADR 0044 波 3）。log と ceil の境界に float32/float64 の差が出ないという実測命題
 * なので、`bucketSize` / `maxPosition` を変える時は測り直す。
 */

import type { Tensor } from "@karume/runtime";

/** {@link buildRelPosTables} の返り値（グラフ入力 `c2p_pos` / `p2c_pos` にそのまま渡せる形）。 */
export type RelPosTables = {
  /** content→position の gather 添字。`c2p[i][j] = clamp(bucket(i − j) + span, 0, 2·span − 1)`。 */
  readonly c2pPos: Tensor;
  /** position→content の gather 添字。**`c2p` の転置**（バケット化が奇関数なので厳密に成立）。 */
  readonly p2cPos: Tensor;
};

/**
 * `make_log_bucket_position`（transformers）と同じ式で `rel` をバケット化する。
 *
 * 近傍（`|rel| < mid`）は線形のまま、遠方は対数で圧縮する。`abs_pos <= mid` の分岐が
 * `mid - 1` の代入と重なるので、近傍は必ず `rel` そのものを返す。
 */
const logBucketPosition = (rel: number, bucketSize: number, maxPosition: number): number => {
  const mid = Math.floor(bucketSize / 2);
  const absPos = rel < mid && rel > -mid ? mid - 1 : Math.abs(rel);
  if (absPos <= mid) return rel;
  const sign = Math.sign(rel);
  const logPos = Math.ceil(
    (Math.log(absPos / mid) / Math.log((maxPosition - 1) / mid)) * (mid - 1),
  ) + mid;
  return logPos * sign;
};

/**
 * 実長 `length` の添字表 2 本を作る。
 *
 * `bucketSize` / `maxPosition` は配布形の `symbols.json`（`bertRelPos`）が持つ — DeBERTa の
 * config 由来の値をホストへ写経しないため（ADR 0039 決定 3 と同じ規律）。
 *
 * 表は `i − j` にしか依存しない（Toeplitz）ので、対角ごとの値 `2·length − 1` 本を先に作って
 * から展開する。T = 512 でも実際の bucket 計算は 1,023 回で済む。
 */
export const buildRelPosTables = (
  length: number,
  bucketSize: number,
  maxPosition: number,
): RelPosTables => {
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError(`相対位置表の長さ ${length} が 1 以上の整数でない`);
  }
  if (!Number.isInteger(bucketSize) || bucketSize < 2) {
    throw new RangeError(`bucketSize ${bucketSize} が 2 以上の整数でない`);
  }
  if (!Number.isInteger(maxPosition) || maxPosition < 2) {
    throw new RangeError(`maxPosition ${maxPosition} が 2 以上の整数でない`);
  }
  const span = bucketSize;
  const upper = 2 * span - 1;
  // 対角 d = i − j ∈ [−(length−1), length−1] の値（表は d にしか依存しない）。
  const diagonal = new Int32Array(2 * length - 1);
  for (let d = -(length - 1); d <= length - 1; d += 1) {
    const shifted = logBucketPosition(d, bucketSize, maxPosition) + span;
    diagonal[d + length - 1] = shifted < 0 ? 0 : shifted > upper ? upper : shifted;
  }
  const c2p = new Int32Array(length * length);
  const p2c = new Int32Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const value = diagonal[i - j + length - 1];
      c2p[i * length + j] = value;
      p2c[j * length + i] = value;
    }
  }
  return {
    c2pPos: { dtype: "i32", shape: [length, length], data: c2p },
    p2cPos: { dtype: "i32", shape: [length, length], data: p2c },
  };
};
