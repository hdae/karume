// 評価素材を 48kHz → 16kHz へ落とす段の門（GPU も実資産も要らない純関数）。
//
// この段の数値そのものは仕様ではない（素材の同一性は sha256 が持つ）。ここで縛るのは
// **折り返し防止が落ちうる形で効いていること**と、直流利得・対称性・出力長という壊れ方の
// 分かりやすい 3 点。フィルタの綴りを変えたらこの門も e2e の期待 `.lab` も採り直しになる。

import { assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { decimate, lowpassTaps } from "./eval-audio.ts";

/** 台本が実際に使う係数（`lowpassTaps(TARGET_RATE * 0.475, SOURCE_RATE, 193)`）。 */
const FILTER = lowpassTaps(16000 * 0.475, 48000, 193);
const FACTOR = 3;

const rms = (samples: Float32Array): number =>
  Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);

Deno.test("lowpassTaps: 直流利得が 1（掛けたぶんだけ音量が動かない）", () => {
  const sum = FILTER.reduce((total, value) => total + value, 0);
  assertAlmostEquals(sum, 1, 1e-12);
});

Deno.test("lowpassTaps: 係数は中心対称（群遅延が整数サンプル）", () => {
  for (let at = 0; at < FILTER.length; at += 1) {
    const mirrored = FILTER[FILTER.length - 1 - at];
    // 窓の計算が両端で別の位相から始まるので、丸めぶん（実測 1.4e-17）だけ許す。
    assertAlmostEquals(FILTER[at], mirrored, 1e-16);
  }
});

Deno.test("lowpassTaps: 偶数タップは落ちる（中心が取れない）", () => {
  assertThrows(() => lowpassTaps(7600, 48000, 192), Error, "タップ数 192 が奇数でない");
});

Deno.test("decimate: 出力長は floor(入力長 / 間引き率)", () => {
  for (const length of [0, 1, 2, 3, 100, 4801]) {
    const output = decimate(new Float32Array(length), FACTOR, FILTER);
    assertEquals(output.length, Math.floor(length / FACTOR));
  }
});

Deno.test("decimate: 直流入力は中央部で 1.0 のまま通る（窓外 0 詰めの減衰が中央に及ばない）", () => {
  const samples = new Float32Array(4800).fill(1);
  const output = decimate(samples, FACTOR, FILTER);
  // 端 96 サンプル（= half）ぶんは 0 詰めで減衰するので、中央だけを見る。
  const guard = Math.ceil((FILTER.length - 1) / 2 / FACTOR);
  for (let at = guard; at < output.length - guard; at += 1) {
    assertAlmostEquals(output[at], 1, 1e-6);
  }
});

Deno.test("decimate: 目標ナイキスト超の正弦は 1/100 未満へ落ちる（折り返し防止が効いている）", () => {
  const samples = new Float32Array(4800);
  for (let at = 0; at < samples.length; at += 1) {
    samples[at] = Math.sin((2 * Math.PI * 10000 * at) / 48000);
  }
  const output = decimate(samples, FACTOR, FILTER);
  // 端は窓外 0 詰めの立ち上がりが残るので中央だけを見る（阻止域の効きを測る門なので、
  // 端の過渡を混ぜると −58dB の実力が端の値で覆い隠される）。
  const guard = Math.ceil((FILTER.length - 1) / 2 / FACTOR);
  const central = output.slice(guard, output.length - guard);
  assertEquals(
    rms(central) < rms(samples) / 100,
    true,
    `出力 RMS が落ちていない: ${rms(central)}（入力 ${rms(samples)}）`,
  );
});
