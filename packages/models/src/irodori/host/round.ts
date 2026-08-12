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

/** S の clamp 範囲と秒 ↔ フレーム換算（`clampRange` だけが使う内側の面）。 */
type FrameBounds = {
  readonly frameRate: number;
  readonly minSeconds: number;
  readonly maxSeconds: number;
};

/**
 * S の決定に要る数（正本は manifest の `pipelineConfig`）。秒 → フレームと
 * 秒 → サンプル → フレームの両方を持つのは、経路ごとに上流の式が違うため。整合
 * （`sampleRate == frameRate × hopLength`）は `config.ts` が parse 時に見ている。
 */
export type SampleBounds = FrameBounds & {
  readonly sampleRate: number;
  readonly hopLength: number;
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
 * 決まった latent 長 S と、波形をそこから切り出す長さ（サンプル）。
 *
 * `targetSamples` を S と一緒に返すのは、2 つの経路で式が違うため（duration 経路は
 * `S × hopLength` ちょうど・手動秒経路は最大 `hopLength − 1` サンプル短い）。呼び出し側で
 * 組み直すと、片方の経路だけが黙って全長を出す。
 */
export type SequencePlan = {
  readonly frames: number;
  readonly targetSamples: number;
};

/**
 * duration グラフの出力（log frames）から S を決める。
 *
 * `expm1` → 銀行家丸め → `[minFrames, maxFrames]` clamp（上流 `_synthesize`）。
 * `duration_scale` は既定 1.0 で、実行時ノブとしては持たない（配布形の既定に無い）。
 * この経路の `targetSamples` は `S × hopLength` ちょうど（切り出しは実質 no-op）。
 */
export const sequenceLengthFromLogFrames = (
  logFrames: number,
  bounds: SampleBounds,
): SequencePlan => {
  const frames = f32(Math.expm1(f32(logFrames)));
  if (!Number.isFinite(frames)) {
    throw new Error(`duration の出力 ${logFrames} から得たフレーム数が有限でない`);
  }
  const { min, max } = clampRange(bounds);
  const bounded = Math.max(min, Math.min(max, bankerRound(frames)));
  return { frames: bounded, targetSamples: bounded * bounds.hopLength };
};

/**
 * 秒の直接指定から S と切り出し長を決める（上流の `manual_seconds` 経路 — duration グラフを
 * 回さない）。
 *
 * 上流（`_synthesize`）の綴りそのまま: `[minSeconds, maxSeconds]` へ clamp した秒を
 * **サンプル数へ落としてから**（`max(1, trunc(clamped × sampleRate))`）フレーム化する
 * （`ceil(targetSamples / hopLength)`）。
 *
 * MUST: `frameRate` 経由の `ceil(clamped × frameRate)` で代用しない。両者は「秒 × frameRate が
 * 整数のすぐ上（1 サンプル未満）」で 1 フレームずれる（例: 1.000005 秒 → 上流 25 / frameRate
 * 経由 26）。値ではなく**長さ**にしか出ないので、突合門が無ければ気づけない差になる。
 *
 * `targetSamples` は S × hopLength より最大 `hopLength − 1` サンプル短い — 呼び出し側は
 * 波形をこの長さで切る（切らないと指定秒より長い音声が出る）。
 */
export const sequenceLengthFromSeconds = (
  seconds: number,
  bounds: SampleBounds,
): SequencePlan => {
  if (!Number.isFinite(seconds)) throw new Error(`durationSeconds ${seconds} が有限の数でない`);
  // 上流はフレーム側の clamp を掛け直さない（秒を先に clamp してあるので範囲は自動で収まる）。
  const clamped = Math.min(bounds.maxSeconds, Math.max(bounds.minSeconds, seconds));
  const targetSamples = Math.max(1, Math.trunc(clamped * bounds.sampleRate));
  return { frames: Math.ceil(targetSamples / bounds.hopLength), targetSamples };
};
