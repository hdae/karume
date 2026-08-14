/**
 * radix-2 の反復 FFT。母音検出の特徴抽出が要る 2 つの量 — 実信号の**パワースペクトル**と
 * **自己相関** — だけを出す。
 *
 * ## なぜ `src/audio/` ではなくファミリの下なのか
 *
 * 汎用の FFT ではない（実信号・2 冪・スペクトルと自己相関の 2 本だけ・スクラッチ常駐）。
 * 利用者は {@link ./features.ts} 1 つで、共有層へ上げる根拠が無い（`src/audio/wav.ts` は
 * 「音声モデルは総じて WAV を読み書きする」から共有層に居る）。第 2 の利用者が現れたときに
 * 上げれば良い。
 *
 * ## MUST: 変換は f64 で回す（上流 TS 実装は f32）
 *
 * 仕様の正本は Python 側（`features.py` / `dsp.py`）で、そこは `np.fft` を通る = **必ず f64**
 * （numpy の FFT は f32 入力でも倍精度へ上げる）。上流の TS 実装は Float32Array でバタフライを
 * 回しているぶんだけ正本から離れており、Python 参照との max abs diff は **5.42e-6**（実測）。
 * ここを f64 にすると dtype の流れが Python と 1 段ずつ揃い、**4.77e-7 = 参照を f32 で
 * 保存したことによる 2 ULP** まで縮む（= 数値としては底に当たっている。導出は
 * `tests/vowel_detector_host_test.ts` の冒頭）。
 *
 * 丸めを写経して偽の一致を作るのではなく、**正本と同じ精度で計算して差を実測する**という
 * 選択は `src/irodori/host/loudness.ts` と同じ（あちらは逆に上流が f32 の IIR で、写しても
 * ビット一致が得られないので f64 を選んだ）。
 */

/** サイズ `size` のビット反転表（DIT の入力並べ替え用）。 */
const bitReverseTable = (size: number): Uint32Array => {
  const bits = Math.log2(size);
  const table = new Uint32Array(size);
  for (let index = 0; index < size; index += 1) {
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed |= ((index >> bit) & 1) << (bits - 1 - bit);
    }
    table[index] = reversed;
  }
  return table;
};

/**
 * 変換長 `size`（2 冪）の FFT。回転因子とスクラッチを持ち回るのでフレームごとに作り直さない。
 *
 * 状態はスクラッチだけ（呼び出しの間に持ち越す意味は無い）。`powerSpectrum` /
 * `autocorrelation` は互いのスクラッチを壊すので、結果は呼び出しごとに `out` へ取り切る。
 */
export class Fft {
  readonly size: number;
  readonly #reversed: Uint32Array;
  readonly #cos: Float64Array;
  readonly #sin: Float64Array;
  readonly #re: Float64Array;
  readonly #im: Float64Array;
  readonly #scratch: Float64Array;

  constructor(size: number) {
    if (!Number.isInteger(size) || size < 2 || (size & (size - 1)) !== 0) {
      throw new RangeError(`Fft: 変換長 ${size} が 2 以上の 2 冪でない`);
    }
    this.size = size;
    this.#reversed = bitReverseTable(size);
    this.#cos = new Float64Array(size / 2);
    this.#sin = new Float64Array(size / 2);
    for (let index = 0; index < size / 2; index += 1) {
      this.#cos[index] = Math.cos((-2 * Math.PI * index) / size);
      this.#sin[index] = Math.sin((-2 * Math.PI * index) / size);
    }
    this.#re = new Float64Array(size);
    this.#im = new Float64Array(size);
    this.#scratch = new Float64Array(size);
  }

  /**
   * 実信号（長さ ≤ `size`・不足は**末尾**ゼロ詰め）のパワースペクトル `|X[k]|²` を
   * `out[0..size/2]` へ書く（`np.abs(np.fft.rfft(x, n=size))**2` と同じ並び）。
   */
  powerSpectrum(input: Float32Array, out: Float64Array): void {
    const bins = this.size / 2 + 1;
    if (input.length > this.size) {
      throw new RangeError(`Fft: 入力 ${input.length} 点が変換長 ${this.size} を超えている`);
    }
    if (out.length < bins) {
      throw new RangeError(`Fft: 出力 ${out.length} 点が rfft の ${bins} ビンに足りない`);
    }
    this.#load(input);
    this.#butterflies();
    for (let bin = 0; bin < bins; bin += 1) {
      out[bin] = this.#re[bin] * this.#re[bin] + this.#im[bin] * this.#im[bin];
    }
  }

  /**
   * 実信号の自己相関（lag 0..`out.length − 1`）を `out` へ書く。
   *
   * MUST: 変換長は信号長の **2 倍以上**（巡回相関が線形相関に化けるゼロ詰めの条件）。
   * パワースペクトルは実かつ偶対称なので、逆変換は同じバタフライを掛けて `size` で割れば
   * 得られる（`np.fft.irfft(S·conj(S))` と同値）。
   */
  autocorrelation(input: Float32Array, out: Float64Array): void {
    if (input.length * 2 > this.size) {
      throw new RangeError(
        `Fft: 自己相関には信号長 ${input.length} の 2 倍以上の変換長が要る（${this.size} しかない）`,
      );
    }
    if (out.length > this.size) {
      throw new RangeError(`Fft: lag ${out.length - 1} が変換長 ${this.size} の外`);
    }
    this.#load(input);
    this.#butterflies();
    for (let bin = 0; bin < this.size; bin += 1) {
      this.#scratch[bin] = this.#re[bin] * this.#re[bin] + this.#im[bin] * this.#im[bin];
    }
    this.#load(this.#scratch);
    this.#butterflies();
    for (let lag = 0; lag < out.length; lag += 1) out[lag] = this.#re[lag] / this.size;
  }

  /** 入力をビット反転順に読み込む（足りない添字はゼロ = 末尾ゼロ詰め）。 */
  #load(values: ArrayLike<number>): void {
    for (let index = 0; index < this.size; index += 1) {
      const source = this.#reversed[index];
      this.#re[index] = source < values.length ? values[source] : 0;
      this.#im[index] = 0;
    }
  }

  /** ビット反転済みの `#re` / `#im` をその場で変換する（DIT）。 */
  #butterflies(): void {
    for (let span = 2; span <= this.size; span <<= 1) {
      const half = span >> 1;
      const step = this.size / span;
      for (let start = 0; start < this.size; start += span) {
        for (let offset = 0, twiddle = 0; offset < half; offset += 1, twiddle += step) {
          const lower = start + offset;
          const upper = lower + half;
          const re = this.#re[upper] * this.#cos[twiddle] - this.#im[upper] * this.#sin[twiddle];
          const im = this.#re[upper] * this.#sin[twiddle] + this.#im[upper] * this.#cos[twiddle];
          this.#re[upper] = this.#re[lower] - re;
          this.#im[upper] = this.#im[lower] - im;
          this.#re[lower] += re;
          this.#im[lower] += im;
        }
      }
    }
  }
}
