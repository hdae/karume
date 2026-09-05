/**
 * 母音検出 CRNN の入力特徴（80 次元 log-mel + DSP 3 次元 = 83）を波形から作る。
 * **グラフの外**にある前段で、`tools/export-recipes/vowel_detector/export.py` が emit する
 * グラフの入力 `features f32 [1, T10, 83]` をここが埋める。
 *
 * ## 置き場（`src/audio/` へ上げない）
 *
 * これはこのモデルの**学習時の契約**そのもの（n_fft 512 / win 400 / hop 160 / 80 mel /
 * Slaney 基底を資産で配る / 発話内 z 化）で、他のファミリと共有できる部分が無い。音声を
 * **生成**する irodori・sbv2 は mel を一切通らないので、現時点の利用者はこのモデル 1 つ
 * （`src/image/preprocess.ts` が共有層に居るのは siglip2 と birefnet の 2 家族が同じ
 * resize → rescale → normalize を通るため）。単一利用の抽象化を作らない、という規約どおり
 * ファミリの下に置く。
 *
 * ## 仕様の正本は Python 側
 *
 * `training/src/vowel_detector/features.py` + `dsp.py`（上流 `@hdae/vowel-detector`・MIT）。
 * 上流には同じ移植の TS 実装もあるが、**食い違ったら Python が正**。移植で守るべき順序と
 * dtype は以下（`np.fft` は f32 入力でも f64 で回るので、そこだけ f64 が正しい）:
 *
 * | 段                          | Python                                   | ここ            |
 * | --------------------------- | ---------------------------------------- | --------------- |
 * | hann 窓                     | `np.hanning(401)[:-1]` を f32 へ         | f32（同値）     |
 * | 窓掛け                      | f32 × f32 → f32                          | f32             |
 * | STFT                        | `np.fft.rfft` → **f64**                  | f64（`fft.ts`） |
 * | mel 積和                    | f64 スペクトル @ f32 基底 → f64          | f64             |
 * | `log(mel + 1e-5)`           | f64 → **f32 へ丸め**                     | 同じ            |
 * | 発話内 z 化                 | f32 配列の mean/std                      | f64 で積算      |
 * | 自己相関                    | `np.fft.rfft`/`irfft` → **f64**          | f64             |
 * | RMS・ZCR                    | f32 配列の平均                           | f64 で積算      |
 *
 * 残る差は「numpy が f32 のまま対和で縮約する箇所を f64 で積算している」ぶんだけで、実測の
 * max abs diff は **4.77e-7 = 参照を f32 で保存したことによる 2 ULP**（導出は
 * `tests/vowel_detector_host_test.ts` の冒頭）。f32 の対和を写経すると偽の一致になるので
 * 採らない（`src/irodori/host/loudness.ts` と同じ判断）。
 *
 * ## mel 基底は焼かずに資産から受け取る
 *
 * `librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000)`（Slaney 正規化）の
 * `[80, 257]` を、manifest v2 の **`assets` 席**（quant 選択に依存しない無条件ファイル —
 * ADR 0041 §3）へ f32 safetensors 1 テンソルで載せ、パイプラインが読んで
 * {@link extractFeatures} へ渡す。sbv2 の `style_vectors` / `speaker_embeddings` と同じ席・
 * 同じ流儀（`src/sbv2/style.ts`）。
 *
 * 式から作り直す選択は**採らない**: 上流も「JS には行列をエクスポートして共有する」形で、
 * mel 尺度の定義（Slaney / HTK）と正規化の綴り方が 1 つでもずれると、値は**それらしいまま**
 * 学習時と違う特徴になる（80 本の三角窓は目視で異常に見えない）。82KB の資産 1 本で
 * その一致を持ち回るほうが素性が良い。
 */

import { Fft } from "./fft.ts";

/** 入力波形のサンプリング周波数。リサンプルはしない（違う周波数は呼び出し側が弾く）。 */
export const SAMPLE_RATE = 16000;
/** STFT の DFT 長。radix-2 で済むよう 2 冪（Python 側 `features.py` の MUST）。 */
export const N_FFT = 512;
/** hann 窓の長さ（25ms）。DFT 長との差はフレーム**中央**に置く（librosa の規約）。 */
export const WIN_LENGTH = 400;
/** フレームの進み（10ms）。 */
export const HOP = 160;
/** mel フィルタ本数。 */
export const N_MELS = 80;
/** mel 基底の列数（`N_FFT / 2 + 1`）。 */
export const MEL_BINS = N_FFT / 2 + 1;
/** DSP 補助特徴の窓（32ms）。log-mel と違い窓関数は掛けない。 */
export const DSP_WINDOW = 512;
/** グラフ入力の特徴次元（`N_MELS` + 有声性 + log エネルギー + ZCR）。 */
export const FEATURE_DIM = N_MELS + 3;

/** 窓をフレーム中央へ置くオフセット（librosa は n_fft 単位でフレームを切る）。 */
const WIN_OFFSET = (N_FFT - WIN_LENGTH) / 2;
/** 自己相関のピーク探索範囲（lag = 16000 / Hz）— 400Hz から 60Hz まで。 */
const LAG_MIN = 40;
const LAG_MAX = 267;
/** log の下駄（log-mel・log エネルギーで共通）。 */
const LOG_EPSILON = 1e-5;
/** 発話内 z 化の分母の下駄。 */
const STD_EPSILON = 1e-5;
/** log エネルギーの圧縮（発話ピーク比を 10 で割って値域を揃える）。 */
const LOG_ENERGY_SCALE = 10;

/** {@link extractFeatures} の結果。 */
type VowelFeatures = {
  /** `[frames, FEATURE_DIM]` の行優先。グラフ入力 `features` にそのまま渡せる。 */
  readonly data: Float32Array<ArrayBuffer>;
  /** 10ms グリッドのフレーム数（グラフの出力は stride 2 でこの半分になる）。 */
  readonly frames: number;
};

/** periodic hann 窓（`np.hanning(length + 1)[:-1]` と同値）。値は f32 へ丸める。 */
const hannWindow = (length: number): Float32Array => {
  const window = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    window[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / length);
  }
  return window;
};

/**
 * `np.signbit` と同じ判定（**−0 は負**）。
 *
 * MUST: `value < 0` だけで済ませない。ZCR は Python 側が `np.signbit` の隣接比較で数えて
 * いるので、−0 を含む波形が来たときだけ交差数がずれ、82 番目の特徴が静かに変わる。
 * `decodeWav` の int16 経路には −0 が出ないが、IEEE float の WAV とホストで加工された波形は
 * −0 を運びうる（正本と同じ判定にしておけば、そこで挙動を考え直さずに済む）。
 */
const signbit = (value: number): boolean => value < 0 || (value === 0 && 1 / value < 0);

/**
 * 16kHz モノラル波形 → `[frames, FEATURE_DIM]` の特徴。
 *
 * `melBasis` は `[N_MELS, MEL_BINS]` の行優先（資産の f32 safetensors そのまま）。
 *
 * フレーム数は `floor((samples − N_FFT) / HOP) + 1`（librosa の `center=False` と同じ
 * 数え方）。この式のもとでは DSP 側の 512 サンプル窓も**必ず波形の内側に収まる**ので、
 * Python 側 `frame_signal` のゼロ pad は発火しない（境界の読みは添字の外へ出ない）。
 *
 * 1 フレームも取れない長さは fail loudly（Python は空配列を返し、後段の `mean()` が NaN を
 * 撒く — 黙って NaN の特徴を作るより落とす）。
 */
export const extractFeatures = (audio: Float32Array, melBasis: Float32Array): VowelFeatures => {
  if (melBasis.length !== N_MELS * MEL_BINS) {
    throw new Error(
      `extractFeatures: mel 基底が ${melBasis.length} 要素（${N_MELS}×${MEL_BINS} = ${
        N_MELS * MEL_BINS
      } が要る）`,
    );
  }
  const frames = Math.floor((audio.length - N_FFT) / HOP) + 1;
  if (frames < 1) {
    throw new Error(
      `extractFeatures: 波形が ${audio.length} サンプルしかない` +
        `（${N_FFT} サンプル = ${(N_FFT / SAMPLE_RATE) * 1000}ms 以上が要る）`,
    );
  }

  const window = hannWindow(WIN_LENGTH);
  const melFft = new Fft(N_FFT);
  const dspFft = new Fft(2 * DSP_WINDOW);
  const windowed = new Float32Array(WIN_LENGTH);
  const centered = new Float32Array(DSP_WINDOW);
  const power = new Float64Array(MEL_BINS);
  const autocorrelation = new Float64Array(LAG_MAX + 1);

  const logMel = new Float32Array(frames * N_MELS);
  const voiced = new Float32Array(frames);
  const logEnergy = new Float32Array(frames);
  const zeroCrossingRate = new Float32Array(frames);

  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * HOP;
    // --- log-mel（窓はフレーム中央の WIN_LENGTH サンプルに掛かる） ---
    for (let index = 0; index < WIN_LENGTH; index += 1) {
      windowed[index] = audio[offset + WIN_OFFSET + index] * window[index];
    }
    melFft.powerSpectrum(windowed, power);
    for (let mel = 0; mel < N_MELS; mel += 1) {
      const row = mel * MEL_BINS;
      let sum = 0;
      for (let bin = 0; bin < MEL_BINS; bin += 1) sum += power[bin] * melBasis[row + bin];
      logMel[frame * N_MELS + mel] = Math.log(sum + LOG_EPSILON);
    }

    // --- DSP 補助（窓関数は無し・平均だけ抜く） ---
    let mean = 0;
    for (let index = 0; index < DSP_WINDOW; index += 1) mean += audio[offset + index];
    mean /= DSP_WINDOW;
    let squared = 0;
    let crossings = 0;
    for (let index = 0; index < DSP_WINDOW; index += 1) {
      const sample = audio[offset + index];
      const value = sample - mean;
      centered[index] = value;
      squared += value * value;
      // MUST: 交差は**生の波形**で数える（Python も平均を抜く前の frames を見ている）。
      if (index > 0 && signbit(sample) !== signbit(audio[offset + index - 1])) crossings += 1;
    }
    dspFft.autocorrelation(centered, autocorrelation);
    let peak = Number.NEGATIVE_INFINITY;
    for (let lag = LAG_MIN; lag <= LAG_MAX; lag += 1) {
      peak = Math.max(peak, autocorrelation[lag]);
    }
    voiced[frame] = autocorrelation[0] > 1e-10 ? peak / autocorrelation[0] : 0;
    logEnergy[frame] = Math.log(Math.sqrt(squared / DSP_WINDOW) + LOG_EPSILON);
    zeroCrossingRate[frame] = crossings / (DSP_WINDOW - 1);
  }

  // log-mel は**発話全体の 1 スカラー**で z 化する（列ごとでも行ごとでもない）。
  let sum = 0;
  for (const value of logMel) sum += value;
  const mean = sum / logMel.length;
  let variance = 0;
  for (const value of logMel) variance += (value - mean) * (value - mean);
  const deviation = Math.sqrt(variance / logMel.length) + STD_EPSILON;

  // log エネルギーは**発話ピーク比**（0 以下）。
  let peakLogEnergy = Number.NEGATIVE_INFINITY;
  for (const value of logEnergy) peakLogEnergy = Math.max(peakLogEnergy, value);

  const data = new Float32Array(frames * FEATURE_DIM);
  for (let frame = 0; frame < frames; frame += 1) {
    const row = frame * FEATURE_DIM;
    for (let mel = 0; mel < N_MELS; mel += 1) {
      data[row + mel] = (logMel[frame * N_MELS + mel] - mean) / deviation;
    }
    data[row + N_MELS] = voiced[frame];
    data[row + N_MELS + 1] = (logEnergy[frame] - peakLogEnergy) / LOG_ENERGY_SCALE;
    data[row + N_MELS + 2] = zeroCrossingRate[frame];
  }
  return { data, frames };
};
