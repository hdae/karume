/**
 * `dit` の連結マスク `[1,1,1,S+Σcaps]` の構築（ADR 0047 決定 1 / 4）。
 *
 * 並びは **self（latent 全 1）→ text → speaker → caption** で、これは `dit` グラフ内の K/V の
 * 連結順そのもの。各条件区間は「使った長さの prefix だけ True」で、残りは右 pad ぶんの False。
 *
 * MUST: 区間の順とオフセットを崩さない。長さの合計が同じなら shape も実行も通り、**別の
 * 条件を落とした**結果が沈黙で出る（exporter の `TestSegmentMasks` が同じ値を固定している）。
 *
 * uncond 変種は「該当区間だけ全 False」— cond の state をそのまま渡してマスクだけを落とすと
 * 上流の uncond（state 0 + マスク 0）とビット一致する、というのが ADR 0047 決定 1 の実測。
 */

/** 条件区間の名前（= CFG の変種名 / K/V の連結順）。 */
export type IrodoriSegment = "text" | "speaker" | "caption";

/** 連結順。**この順が正本**（exporter の `DIT_UNCOND_VARIANTS` と同綴り・同順）。 */
export const SEGMENT_ORDER: readonly IrodoriSegment[] = ["text", "speaker", "caption"];

/** 区間ごとの長さ（`used` = 実際に埋めた行数 / `caps` = グラフの宣言長）。 */
export type SegmentLengths = Readonly<Record<IrodoriSegment, number>>;

/**
 * 連結マスクを組む。`uncond` を渡すとその区間だけ全 False になる。
 *
 * dtype は bool（IR の境界規約 = u32 の 0/1 — ADR 0009）。
 */
export const buildDitMask = (
  latentLength: number,
  used: SegmentLengths,
  caps: SegmentLengths,
  uncond?: IrodoriSegment,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isInteger(latentLength) || latentLength < 1) {
    throw new RangeError(`latent 長 ${latentLength} が 1 以上の整数でない`);
  }
  let total = latentLength;
  for (const segment of SEGMENT_ORDER) {
    if (used[segment] > caps[segment]) {
      throw new Error(
        `${segment} 区間の使用長 ${used[segment]} が宣言長 ${caps[segment]} を超えている`,
      );
    }
    total += caps[segment];
  }
  const mask = new Uint32Array(total);
  mask.fill(1, 0, latentLength);
  let offset = latentLength;
  for (const segment of SEGMENT_ORDER) {
    const length = uncond === segment ? 0 : used[segment];
    mask.fill(1, offset, offset + length);
    offset += caps[segment];
  }
  return mask;
};
