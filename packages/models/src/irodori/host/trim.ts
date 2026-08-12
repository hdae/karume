/**
 * 末尾トリム（上流 `inference_runtime.find_flattening_point` + `_synthesize` の切り出し）。
 *
 * ## MUST: 判定は **z（latent）上**で行い、切るのは**波形**
 *
 * 上流は「latent の末尾が平坦かつ 0 近傍になる位置」を latent 上で探し、その位置に
 * `hopLength` を掛けたサンプル位置で**波形を**切る。latent を切ってから decode すると、
 * 境界の padding が変わるので全長 decode の先頭部分とビット一致しない（recon 実測）。
 * したがって decode は常に全長で行い、切るのは最後の 1 回だけ。
 *
 * ## しきい値は上流 `SamplingRequest` の既定（実行時ノブとしては持たない）
 *
 * `tail_window_size` / `tail_std_threshold` / `tail_mean_threshold` は上流ではリクエスト側の
 * 欄だが、この実装は既定しか出さないので定数で持つ（`pipelineConfig` に載っていない = 配布形
 * が別の値を宣言する手段が無い）。golden との突合は
 * `outputs/series/dacvae-32dim/host/trim.safetensors` の 3 ケース。
 */

/** 末尾を見る窓（latent フレーム数）。 */
const WINDOW = 20;
/** 窓全体（`WINDOW × latentDim` スカラー）の母標準偏差のしきい値。 */
const STD_THRESHOLD = 0.05;
/** 窓全体の平均の、目標値 0 からの許容差。 */
const MEAN_THRESHOLD = 0.1;

/**
 * 末尾が平坦になり始める latent フレーム位置を返す（見つからなければ `frames`）。
 *
 * 上流と同じく **`WINDOW` 行のゼロ pad を末尾に足してから**先頭から走査する — pad があるので
 * 窓は常に `WINDOW` 行ぶんのスカラーを持ち、末尾に近い窓ほどゼロで薄まる。走査は `i = frames −
 * 1` までで、`i = frames`（全部 pad）は見ない（無条件に成立してしまうため）。
 *
 * MUST: 母標準偏差（torch の `std(unbiased=False)`）— 不偏だと窓が小さいほど値が持ち上がり、
 * しきい値 0.05 の判定がフレーム単位でずれる。
 */
export const findFlatteningPoint = (
  latent: Float32Array,
  frames: number,
  latentDim: number,
): number => {
  if (!Number.isInteger(frames) || frames < 0) {
    throw new Error(`findFlatteningPoint: frames ${frames} が非負整数でない`);
  }
  if (!Number.isInteger(latentDim) || latentDim <= 0) {
    throw new Error(`findFlatteningPoint: latentDim ${latentDim} が正の整数でない`);
  }
  if (latent.length !== frames * latentDim) {
    throw new Error(
      `findFlatteningPoint: latent の要素数 ${latent.length} が ${frames}×${latentDim} と違う`,
    );
  }
  const count = WINDOW * latentDim;
  for (let start = 0; start < frames; start += 1) {
    // ゼロ pad の行は和にも二乗和にも寄与しないので、実在する行だけ走ればよい。
    const end = Math.min(frames, start + WINDOW) * latentDim;
    const from = start * latentDim;
    let sum = 0;
    for (let index = from; index < end; index += 1) sum += latent[index];
    const mean = sum / count;
    if (Math.abs(mean) >= MEAN_THRESHOLD) continue;
    // 2 パス（平均を出してから偏差二乗和）。`E[x²] − E[x]²` は std が 0 に近いこの用途で
    // 桁落ちして負の分散を作りうる — 判定したいのがまさにその領域なので使わない。
    let squared = 0;
    for (let index = from; index < end; index += 1) {
      const deviation = latent[index] - mean;
      squared += deviation * deviation;
    }
    // pad の行は `0 − mean` の偏差を持つ（上流も pad ごと std を採る）。
    const padded = (count - (end - from)) * mean * mean;
    if (Math.sqrt((squared + padded) / count) < STD_THRESHOLD) return start;
  }
  return frames;
};

/**
 * 波形を切る長さ（サンプル）を決める（上流 `_synthesize` の `max_samples`）。
 *
 * `flatteningPoint` が 0 のときだけ切らない — 上流が `flattening_samples > 0` を条件にして
 * いるためで、「先頭から平坦」= 全部無音と判定されたときに 0 サンプルの音声を返さない。
 */
export const trimmedSampleCount = (
  targetSamples: number,
  flatteningPoint: number,
  hopLength: number,
): number => {
  const flatteningSamples = flatteningPoint * hopLength;
  return flatteningSamples > 0 ? Math.min(targetSamples, flatteningSamples) : targetSamples;
};
