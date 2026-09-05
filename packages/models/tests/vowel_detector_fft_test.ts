// 母音検出の FFT（`src/vowel-detector/fft.ts`）の門。GPU も実資産も要らない。
//
// 現状この class を踏むのは `extractFeatures` 経由の Python パリティ 1 本だけで、4 つの
// `RangeError` はどれも通っていない。加えて「値が合っている」ことも特徴抽出の合成結果としてしか
// 見ておらず、`powerSpectrum` / `autocorrelation` 単体の定義（`np.fft` と同じ並び・同じ
// 正規化）は縛られていない。ここで押さえるのは 3 点:
//
// ① **受理集合の外側**（2 冪でない変換長 / 入力が変換長超過 / 出力が rfft ビン数に不足 /
//    自己相関の変換長不足）が全部 `RangeError` で落ちる。
// ② **素朴 DFT との一致**（rtol 1e-12）。バタフライ・ビット反転・回転因子の符号が壊れると、
//    特徴は「それらしい別の値」になるだけで shape も値域も合ったまま最後まで通る。
// ③ **ゼロ詰めは末尾**（先頭詰めだと位相が回り、パワースペクトルの平坦性が崩れる）。

import { assertAlmostEquals, assertThrows } from "@std/assert";
import { Fft } from "../src/vowel-detector/fft.ts";

/** 素朴 DFT の `|X[k]|²`（末尾ゼロ詰め・k = 0..size/2）。 */
const naivePowerSpectrum = (input: Float32Array, size: number): number[] => {
  const bins: number[] = [];
  for (let k = 0; k <= size / 2; k += 1) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < size; n += 1) {
      const value = n < input.length ? input[n] : 0;
      const angle = (-2 * Math.PI * k * n) / size;
      re += value * Math.cos(angle);
      im += value * Math.sin(angle);
    }
    bins.push(re * re + im * im);
  }
  return bins;
};

/** 素朴な線形自己相関 `sum_n x[n]·x[n+lag]`（ゼロ詰めの外は 0）。 */
const naiveAutocorrelation = (input: Float32Array, lags: number): number[] => {
  const out: number[] = [];
  for (let lag = 0; lag < lags; lag += 1) {
    let sum = 0;
    for (let n = 0; n + lag < input.length; n += 1) sum += input[n] * input[n + lag];
    out.push(sum);
  }
  return out;
};

/** 決定的な擬似乱数（seed 固定 — グローバル乱数に依存しない）。 */
const pseudoRandom = (length: number, seed: number): Float32Array<ArrayBuffer> => {
  const values = new Float32Array(length);
  let state = seed;
  for (let index = 0; index < length; index += 1) {
    state = (state * 1103515245 + 12345) % 2147483648;
    values[index] = state / 2147483648 - 0.5;
  }
  return values;
};

Deno.test("Fft: 2 冪でない変換長は作れない", () => {
  for (const size of [3, 1, 0, -4, 2.5]) {
    assertThrows(() => new Fft(size), RangeError, "2 以上の 2 冪でない", `size=${size}`);
  }
});

Deno.test("Fft.powerSpectrum: 入力が変換長を超えたら落ちる（黙って切り詰めない）", () => {
  assertThrows(
    () => new Fft(8).powerSpectrum(new Float32Array(9), new Float64Array(5)),
    RangeError,
    "変換長 8 を超えている",
  );
});

Deno.test("Fft.powerSpectrum: 出力が rfft のビン数に足りなければ落ちる", () => {
  // `bins = size/2 + 1` = 5（`np.fft.rfft` と同じ並び）。
  assertThrows(
    () => new Fft(8).powerSpectrum(new Float32Array(4), new Float64Array(4)),
    RangeError,
    "rfft の 5 ビンに足りない",
  );
});

Deno.test("Fft.autocorrelation: 変換長が信号長の 2 倍未満なら落ちる（巡回相関に化ける条件）", () => {
  assertThrows(
    () => new Fft(8).autocorrelation(new Float32Array(5), new Float64Array(4)),
    RangeError,
    "2 倍以上の変換長が要る",
  );
});

Deno.test("Fft.powerSpectrum: 素朴 DFT と一致する（長さ 8 の実信号）", () => {
  const size = 8;
  const input = pseudoRandom(size, 20260904);
  const out = new Float64Array(size / 2 + 1);
  new Fft(size).powerSpectrum(input, out);
  const expected = naivePowerSpectrum(input, size);
  for (let bin = 0; bin < expected.length; bin += 1) {
    // 相対 1e-12 — f64 のバタフライと素朴 DFT の丸め差だけが残る幅。
    assertAlmostEquals(
      out[bin],
      expected[bin],
      Math.abs(expected[bin]) * 1e-12 + 1e-15,
      `bin ${bin}`,
    );
  }
});

Deno.test("Fft.autocorrelation: 素朴な線形自己相関と一致する（長さ 4・変換長 8）", () => {
  const input = pseudoRandom(4, 20260905);
  const out = new Float64Array(4);
  new Fft(8).autocorrelation(input, out);
  const expected = naiveAutocorrelation(input, out.length);
  for (let lag = 0; lag < expected.length; lag += 1) {
    assertAlmostEquals(
      out[lag],
      expected[lag],
      Math.abs(expected[lag]) * 1e-12 + 1e-15,
      `lag ${lag}`,
    );
  }
});

Deno.test("Fft.powerSpectrum: 足りない長さは末尾へゼロ詰めする（先頭詰めなら位相が回る）", () => {
  // `[1, 0]` を変換長 4 で通すと、末尾詰め（= `[1,0,0,0]`）なら全ビンが 1。先頭詰め
  // （`[0,0,1,0]`）でも `|X[k]|²` は 1 のままなので、向きは自己相関側で見る。
  const out = new Float64Array(3);
  new Fft(4).powerSpectrum(Float32Array.from([1, 0]), out);
  for (let bin = 0; bin < out.length; bin += 1) {
    assertAlmostEquals(out[bin], 1, 1e-12, `bin ${bin}`);
  }

  // 向きの門: `[1, 2]` の自己相関は lag 0 = 5・lag 1 = 2。先頭詰めだと lag 1 が 0 になる。
  const lags = new Float64Array(2);
  new Fft(4).autocorrelation(Float32Array.from([1, 2]), lags);
  assertAlmostEquals(lags[0], 5, 1e-12);
  assertAlmostEquals(lags[1], 2, 1e-12);
});
