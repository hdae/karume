/**
 * latent 長 S の決定（上流 `inference_runtime._synthesize` の 2 経路）。
 *
 * MUST: 丸めは Python の `round`（**銀行家丸め** = 0.5 ちょうどは偶数側）。`Math.round` は
 * 常に上へ倒すので、予測フレーム数が .5 ちょうどのときだけ 1 フレームずれる — 出力の長さが
 * 変わるだけで例外は出ず、golden 突合まで沈黙する。
 *
 * MUST: `expm1` は f32 で踏む。グラフ出力（log frames）は f32 なので、f64 のまま指数へ通すと
 * 境界ケースで丸め先が変わりうる。
 */

const f32 = Math.fround;

/** S の clamp 範囲と秒 ↔ フレーム換算（正本は manifest の `pipelineConfig`）。 */
export type FrameBounds = {
  readonly frameRate: number;
  readonly minSeconds: number;
  readonly maxSeconds: number;
};

/**
 * Python の `round(x)`（half-to-even）。
 *
 * NOTE: 0.5 ちょうどが観測できるのは値が 2 進で厳密に半整数のときだけで、それ以外は
 * 単純な最近接丸めと同じ。判定を `x - floor(x)` で書いているのは、その厳密な半整数を
 * 取りこぼさないため。
 */
export const bankerRound = (value: number): number => {
  if (!Number.isFinite(value)) throw new RangeError(`丸め対象 ${value} が有限の数でない`);
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction > 0.5) return lower + 1;
  if (fraction < 0.5) return lower;
  return lower % 2 === 0 ? lower : lower + 1;
};

/** clamp の下限 / 上限フレーム（上流と同じ `ceil` / `floor` の非対称）。 */
const clampRange = (bounds: FrameBounds): { readonly min: number; readonly max: number } => ({
  min: Math.max(1, Math.ceil(bounds.minSeconds * bounds.frameRate)),
  max: Math.max(1, Math.floor(bounds.maxSeconds * bounds.frameRate)),
});

/**
 * duration グラフの出力（log frames）から S を決める。
 *
 * `expm1` → 銀行家丸め → `[minFrames, maxFrames]` clamp（上流 `_synthesize`）。
 * `duration_scale` は既定 1.0 で、実行時ノブとしては持たない（配布形の既定に無い）。
 */
export const sequenceLengthFromLogFrames = (logFrames: number, bounds: FrameBounds): number => {
  const frames = f32(Math.expm1(f32(logFrames)));
  if (!Number.isFinite(frames)) {
    throw new Error(`duration の出力 ${logFrames} から得たフレーム数が有限でない`);
  }
  const { min, max } = clampRange(bounds);
  return Math.max(min, Math.min(max, bankerRound(frames)));
};

/**
 * 秒の直接指定から S を決める（上流の `manual_seconds` 経路 — duration グラフを回さない）。
 *
 * 上流は `[minSeconds, maxSeconds]` へ clamp した秒を**サンプル数へ落としてから**
 * `ceil(samples / hopLength)` でフレーム化する。
 *
 * MUST（既知の差分）: ここはフレームレート経由で `ceil(clamped × frameRate)` を採る。
 * `sampleRate` / `hopLength` は `pipelineConfig` が運んでいない（codec 波が入るまで配布形に
 * 無い）ためで、両者が食い違うのは「秒 × frameRate が整数のすぐ上（1 サンプル未満）」に
 * 落ちる入力だけ。差は最大 1 フレーム（= 1/frameRate 秒）で、値ではなく長さにしか出ない。
 * codec 波で `sampleRate` / `hopLength` が配布形に載ったら上流の綴りへ寄せること。
 */
export const sequenceLengthFromSeconds = (seconds: number, bounds: FrameBounds): number => {
  if (!Number.isFinite(seconds)) throw new Error(`durationSeconds ${seconds} が有限の数でない`);
  // 上流はフレーム側の clamp を掛け直さない（秒を先に clamp してあるので範囲は自動で収まる）。
  const clamped = Math.min(bounds.maxSeconds, Math.max(bounds.minSeconds, seconds));
  return Math.max(1, Math.ceil(clamped * bounds.frameRate));
};
