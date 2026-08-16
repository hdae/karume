/**
 * 参照音声（wav）→ codec encoder の入力までのホスト前処理（上流 `codec.py` の
 * `_normalize_loudness` / `encode_waveform` と `DACVAE._pad`）。
 *
 * 段は 3 つで、順序に意味がある: **切り詰め → 正規化 → reflect pad**。
 * 切り詰めを正規化の後に回すと、捨てる区間の音量が LUFS に混ざる（上流は
 * `wav[:, :int(max_ref_seconds·sr)]` を先に取る）。pad を正規化より前に回すと、鏡像の区間が
 * peak 判定に混ざる。
 *
 * 数の正本は golden `outputs/series/dacvae-32dim/host/`（`dacvae_host.py` が上流を**呼んで**
 * 書く 5 ケース）で、突合は `packages/models/tests/irodori_reference_test.ts`。
 */

import { integratedLoudness } from "./loudness.ts";

/**
 * 参照音声の目標ラウドネス（LUFS）。
 *
 * 上流 `inference_runtime.SamplingRequest.ref_normalize_db` の既定で、`pipelineConfig` には
 * 載っていない（配布形が別の値を宣言する手段が無い）。末尾トリムのしきい値（`trim.ts`）と
 * 同じ扱いで、**実行時ノブとしては持たない**。
 */
const TARGET_DB = -16;

/**
 * dB → 線形の換算係数（`ln(10)/20`）。audiotools の `AudioSignal.GAIN_FACTOR` と同じ数で、
 * golden `meta.json` の `gainFactor` が実測値を持っている。
 */
const GAIN_FACTOR = Math.LN10 / 20;

/** {@link normalizeReference} の結果。スカラーは golden 突合の観測点として返す。 */
type NormalizedReference = {
  readonly data: Float32Array<ArrayBuffer>;
  /** 測った integrated loudness（LUFS）。 */
  readonly refDb: number;
  /** LUFS 利得（`10^((−16 − refDb)/20)`）。 */
  readonly loudnessGain: number;
  /** LUFS 利得を当てた後の peak（1 を超えたときだけ次の peak 利得が 1 未満になる）。 */
  readonly peakBeforeScale: number;
  /** peak 制限の利得（`peak > 1` のときだけ `1/peak`・それ以外は厳密に 1）。 */
  readonly peakGain: number;
};

/**
 * LUFS 正規化 + peak 制限（上流 `_normalize_loudness` → `ensure_max_of_audio`）。
 *
 * 掛けるのは **2 本の利得だけ**で、その積が正規化の全て（`dacvae_host.py` の
 * `_decomposition_evidence` が上流の出力とのビット一致を毎回実測している）。
 */
export const normalizeReference = (
  samples: Float32Array,
  sampleRate: number,
): NormalizedReference => {
  const refDb = integratedLoudness(samples, sampleRate);
  const loudnessGain = Math.exp((TARGET_DB - refDb) * GAIN_FACTOR);
  const scaled = new Float32Array(samples.length) as Float32Array<ArrayBuffer>;
  let peakBeforeScale = 0;
  for (let i = 0; i < samples.length; i += 1) {
    // MUST: 非有限サンプルはここで落とす。NaN は `magnitude > peakBeforeScale` が常に false
    // なので peak 判定を素通りし、+Inf は `peakGain = 1/∞ = 0` で**全サンプルを 0 に潰す**
    // （残るのは NaN 1 点だけ）— どちらも例外にならず「ほぼ無音の参照音声」として通る。
    if (!Number.isFinite(samples[i])) {
      throw new Error(`irodori: 参照音声の ${i} 番目のサンプルが非有限（${samples[i]}）`);
    }
    // 代入した時点で f32 へ丸まる。peak は**丸めた後の値**で採る（上流も f32 テンソルの max）。
    scaled[i] = samples[i] * loudnessGain;
    const magnitude = Math.abs(scaled[i]);
    if (magnitude > peakBeforeScale) peakBeforeScale = magnitude;
  }
  const peakGain = peakBeforeScale > 1 ? 1 / peakBeforeScale : 1;
  if (peakGain !== 1) {
    for (let i = 0; i < scaled.length; i += 1) scaled[i] *= peakGain;
  }
  return { data: scaled, refDb, loudnessGain, peakBeforeScale, peakGain };
};

/**
 * 末尾を鏡像で埋めて長さを `hopLength` の倍数へ揃える（上流 `DACVAE._pad` の reflect pad）。
 *
 * 鏡像は**端のサンプルを含まない**（`[a,b,c,d]` に 2 足すと `[a,b,c,d,c,b]`）— torch の
 * `F.pad(mode="reflect")` の綴り。長さが既に倍数なら入力をそのまま返す（写しも作らない）。
 *
 * MUST: `hopLength` 未満の入力は落とす。pad 量は最大 `hopLength − 1` なので、それより短い
 * 入力では鏡像が自分の先頭を飛び越える（torch も同じ条件で例外を投げる）。
 */
export const reflectPadToHop = (
  samples: Float32Array<ArrayBuffer>,
  hopLength: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(hopLength) || hopLength <= 0) {
    throw new RangeError(`reflectPadToHop: hopLength ${hopLength} が正の整数でない`);
  }
  if (samples.length < hopLength) {
    throw new RangeError(
      `reflectPadToHop: 参照音声が ${samples.length} サンプルしかない（hopLength ${hopLength} が要る）`,
    );
  }
  const pad = (hopLength - (samples.length % hopLength)) % hopLength;
  if (pad === 0) return samples;
  const padded = new Float32Array(samples.length + pad) as Float32Array<ArrayBuffer>;
  padded.set(samples);
  for (let i = 0; i < pad; i += 1) padded[samples.length + i] = samples[samples.length - 2 - i];
  return padded;
};
