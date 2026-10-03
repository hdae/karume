/**
 * Wan2.1 の umT5 の相対位置のバケット表をホストで作る（ADR 0119 決定 3）。
 *
 * T5 系の attention は位置の差（キーの位置 − クエリの位置）を 32 個のバケットに落とし、層ごとの表
 * `[32, heads]` から引いた値をスコアに足す。表 `[L, L]` は有効長 L だけで決まるが、L はグラフの
 * 動的形の 1 シンボル（決定 4）なので、定数に焼くと最大長の表から `[:L, :L]` を切り出す形が要る。
 * DeBERTa（ADR 0045）と Anima の RoPE（ADR 0034）の入力昇格に倣い、ホストが作ってグラフ入力で渡す。
 *
 * ## MUST: Python 側とバイト一致であること
 *
 * 正本は recipe `tools/export-recipes/wan/umt5_patch.py` の `relative_position_buckets`（上流
 * transformers 5.14.1 の `UMT5Attention._relative_position_bucket` をそのまま呼ぶ）。式が割れても
 * shape は合うので、モデルは落ちずに別の距離のバイアスを黙って足す（沈黙誤値）。golden の表も
 * Python が作るので、ホストとゴールデンが同じ誤りを共有すると E2E もすり抜ける。
 * `packages/models/tests/wan_umt5_relative_position_test.ts` が fixture（`fixtures/wan-umt5/`）と
 * 有効長 2〜512 の全表をバイト一致で縛る（sbv2 の `sbv2_rel_pos_parity_test.ts` と同じ責務）。
 *
 * 上流は torch の f32 の `log` で、ここは f64 の `Math.log`。境界（距離 8・12・16・23・32・46・64・91）
 * で切り捨てが割れないことは実測命題（調査 2026-10-03-umt5-encoder-recon §4.3 — 距離 ±511 の全件で
 * 一致）なので、構成（{@link WAN_UMT5_RELATIVE_POSITION}）や上限を変えるときは測り直す。
 *
 * MUST: モジュール副作用ゼロ（CLAUDE.md）。
 */

import type { Tensor } from "@karume/runtime";
import { WAN_PROMPT_MIN_TOKENS } from "../text/tokenizer.ts";

/** i32 のテンソル（バケット表の型 — Session の入力にそのまま渡せる）。 */
export type I32Tensor = Extract<Tensor, { readonly dtype: "i32" }>;

/**
 * バケットの構成（umT5-XXL の `text_encoder/config.json` — 双方向〈エンコーダ〉の枝だけ）。
 *
 * 構成の宣言の席は持たない: Python と突き合わせたのはこの 1 組だけで、宣言できるようにすると
 * 検証していない組み合わせを配布側が選べてしまう（`scheduler.ts` の `WAN_UNIPC_CONFIG` と同じ判断）。
 * 値が pin した config と一致することは recipe の pytest（`test_umt5_host_fixture.py`）と
 * fixture のテストが縛る。
 */
export const WAN_UMT5_RELATIVE_POSITION = { numBuckets: 32, maxDistance: 128 } as const;

/**
 * 有効長の上限（グラフの記号次元の上限・上流の `max_sequence_length` — 決定 4）。Python の表との
 * 一致を確かめたのもここまで（距離 ±511）。
 */
export const WAN_UMT5_MAX_LENGTH = 512;

/** 距離 `distance`（キーの位置 − クエリの位置）のバケット（上流の式の双方向の枝）。 */
const bucketOf = (distance: number): number => {
  const { numBuckets, maxDistance } = WAN_UMT5_RELATIVE_POSITION;
  // 双方向: 半分ずつを向きに割り当てる（正の距離 = キーがクエリより後ろ — 後半の 16〜31）。
  const half = numBuckets / 2;
  const base = distance > 0 ? half : 0;
  const magnitude = Math.abs(distance);
  // 半分の半分（0〜7）は距離そのまま、残り（8〜15）は max_distance までの対数の刻み。
  const maxExact = half / 2;
  if (magnitude < maxExact) return base + magnitude;
  // 上流の `log_ratio.to(torch.long)`（0 への切り捨て — 対数の比は 0 以上なので floor と同じ）。
  const large = maxExact +
    Math.trunc(
      (Math.log(magnitude / maxExact) / Math.log(maxDistance / maxExact)) * (half - maxExact),
    );
  return base + Math.min(large, half - 1);
};

/**
 * 有効長 `length` のバケット表 `[L, L]`（i32・行優先）。`table[i][j] = bucket(j − i)`（上流の
 * `compute_bias` と同じ向き — 行がクエリ・列がキー）。
 *
 * 表は `j − i` にしか依存しない（Toeplitz）ので、対角ごとの値 `2L − 1` 本を先に作ってから展開する
 * （L = 512 でもバケットの計算は 1,023 回）。
 */
export const buildUmt5RelativePositionBuckets = (length: number): I32Tensor => {
  if (!Number.isInteger(length) || length < WAN_PROMPT_MIN_TOKENS || length > WAN_UMT5_MAX_LENGTH) {
    throw new RangeError(
      `umT5 の有効長 ${length} が ${WAN_PROMPT_MIN_TOKENS}〜${WAN_UMT5_MAX_LENGTH} の整数でない`,
    );
  }
  // 対角 d = j − i ∈ [−(L−1), L−1] の値（添字は d + L − 1）。
  const diagonal = new Int32Array(2 * length - 1);
  for (let distance = 1 - length; distance < length; distance += 1) {
    diagonal[distance + length - 1] = bucketOf(distance);
  }
  const data = new Int32Array(length * length);
  for (let query = 0; query < length; query += 1) {
    // 行 i は対角の値の連続した区間（d = −i 〜 L−1−i）をそのまま写したもの。
    const start = length - 1 - query;
    data.set(diagonal.subarray(start, start + length), query * length);
  }
  return { dtype: "i32", shape: [length, length], data };
};
