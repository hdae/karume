/**
 * 継続長（front の logw → 音素ごとのフレーム数 → フレーム展開インデックス）。
 *
 * 参照実装 `SynthesizerTrnJPExtra.infer` の
 * `logw = sdp(...)·r + dp(...)·(1−r)` → `w = exp(logw)·x_mask·length_scale` →
 * `w_ceil = ceil(w)` → `attn = generate_path(w_ceil, ...)` と同義。**attn 行列（Ty×P）は
 * 作らない** — 単調パスなので `expandIdx`（長さ Ty の音素添字列）で同じ情報を持てる。
 * 続く m_p / logs_p の展開は行列積ではなく gather になる（ADR 0013 のホスト責務）。
 */

type DurationPlan = {
  /** 音素ごとのフレーム数（ceil 済み・マスク適用済み）。 */
  readonly wCeil: Int32Array<ArrayBuffer>;
  /** 総フレーム数 Ty。 */
  readonly totalFrames: number;
  /** フレーム → 音素の展開インデックス（長さ Ty）。 */
  readonly expandIdx: Int32Array<ArrayBuffer>;
};

/**
 * logw（sdp / dp）から音素ごとの継続長とフレーム展開インデックスを作る。
 *
 * @param logwSdp front の出力 0（`[1,1,P]` の生データ）。
 * @param logwDp front の出力 1（同上）。
 * @param xMask front の入力 x_mask（`[1,1,P]`）。パディング列を 0 フレームに落とす。
 * @param maxFrames 総フレーム数の上限（配布形の `pipelineConfig.maxFrames`）。
 *
 * MUST: 上限の検査は `expandIdx` の確保**より前**に置く。総フレーム数は logw と lengthScale
 * の積で決まる要求依存の数で、下流の `(T, T)` 表（8·T² bytes 級）どころか長さ T の展開列
 * すら確保できない大きさになりうる — 確保してから見ると原因の遠い OOM になる。
 */
export const durationsToFrames = (
  logwSdp: Float32Array,
  logwDp: Float32Array,
  xMask: Float32Array,
  sdpRatio: number,
  lengthScale: number,
  maxFrames: number,
): DurationPlan => {
  const phonemes = logwSdp.length;
  if (logwDp.length !== phonemes || xMask.length !== phonemes) {
    throw new Error(
      `長さ不一致 logw_sdp=${phonemes} logw_dp=${logwDp.length} x_mask=${xMask.length}`,
    );
  }
  const wCeil = new Int32Array(phonemes);
  let total = 0;
  for (let i = 0; i < phonemes; i += 1) {
    const logw = logwSdp[i] * sdpRatio + logwDp[i] * (1 - sdpRatio);
    const frames = Math.ceil(Math.exp(logw) * xMask[i] * lengthScale);
    if (!Number.isFinite(frames) || frames < 0) {
      // 発散した logw（front の値が壊れている）を「巨大な Ty」として下流へ流すと、
      // 確保サイズだけが異常になって原因の遠い OOM になる。ここで落とす。
      throw new Error(`音素 ${i} の継続長 ${frames} が有限の非負値でない`);
    }
    wCeil[i] = frames;
    total += frames;
  }
  if (total < 1) {
    throw new Error("総フレーム数が 0（発話にならない — x_mask か front 出力を疑う）");
  }
  if (total > maxFrames) {
    throw new Error(
      `総フレーム数 ${total} が配布形の上限 maxFrames=${maxFrames} を超えている` +
        "（text を短く分けるか lengthScale を下げる）",
    );
  }
  const expandIdx = new Int32Array(total);
  let position = 0;
  for (let i = 0; i < phonemes; i += 1) {
    expandIdx.fill(i, position, position + wCeil[i]);
    position += wCeil[i];
  }
  return { wCeil, totalFrames: total, expandIdx };
};
