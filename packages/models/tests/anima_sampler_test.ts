// Anima のサンプラ（IR に載らない段）の挙動テスト。GPU も資産も要らない純関数。
//
// 参照フィクスチャとの数値パリティ（timesteps_proj 表の突合・latents_mean/std のビット一致）は
// **実 GPU / 実資産の E2E と同じ波**で復帰させる（P3 波 2）。ここは形と不変条件を押さえる。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { cfgEulerStep, needsUncond, sigmaSchedule, timestepsProj } from "../src/anima/sampler.ts";

/** `models/anima-turbo/karume.json` の `pipelineConfig.scheduler` と同じ値（形の検査用）。 */
const SHIFT = 3;
const NUM_TRAIN_TIMESTEPS = 1000;

Deno.test("sigmaSchedule: 長さ steps+1・終端 0・単調減少", () => {
  const sigmas = sigmaSchedule(8, SHIFT);
  assertEquals(sigmas.length, 9);
  assertEquals(sigmas[0], 1, "始点は linspace の start=1（shift を掛けても 3/3 = 1）");
  assertEquals(sigmas[8], 0, "終端 0 が 1 つ足される");
  for (let index = 1; index < 8; index += 1) {
    assert(sigmas[index] < sigmas[index - 1], `σ が単調減少でない（index=${index}）`);
  }
});

Deno.test("sigmaSchedule: steps < 2 は linspace の分母が 0 になるので落とす", () => {
  assertThrows(() => sigmaSchedule(1, SHIFT), RangeError);
  assertThrows(() => sigmaSchedule(2.5, SHIFT), RangeError);
});

Deno.test("sigmaSchedule: shift は manifest の値で効く（1 なら素の linspace）", () => {
  // shift=1 では `1·σ / (0·σ + 1) = σ` に潰れる。scheduler の値を manifest から取る配線が
  // 死んで定数に固定されていたら、この 1 本が落ちる。
  const plain = sigmaSchedule(4, 1);
  assertEquals([...plain].slice(0, 4).map((value) => Math.fround(value)), [
    Math.fround(1),
    Math.fround(1 + (0.25 - 1) / 3),
    Math.fround(1 + 2 * (0.25 - 1) / 3),
    Math.fround(0.25),
  ]);
  const shifted = sigmaSchedule(4, SHIFT);
  assert(shifted[1] !== plain[1], "shift が効いていない");
});

Deno.test("timestepsProj: 前半 cos・後半 sin（flip_sin_to_cos=true）", () => {
  // sigma=0 → timestep=0 → 全周波数で angle=0 なので cos 側が 1・sin 側が 0 になる。
  // 反転していると前半が 0・後半が 1 になり、この 1 本で落ちる。
  const proj = timestepsProj(0, 8, NUM_TRAIN_TIMESTEPS);
  assertEquals([...proj], [1, 1, 1, 1, 0, 0, 0, 0]);
});

Deno.test("timestepsProj: 幅と num_train_timesteps の受理集合", () => {
  assertThrows(() => timestepsProj(0.5, 7, NUM_TRAIN_TIMESTEPS), RangeError);
  assertThrows(() => timestepsProj(0.5, 0, NUM_TRAIN_TIMESTEPS), RangeError);
  assertThrows(() => timestepsProj(0.5, 8, 0), RangeError);
  assertThrows(() => timestepsProj(0.5, 8, 1000.5), RangeError);
});

Deno.test("needsUncond: CFG=1 だけが uncond 不要", () => {
  assertEquals(needsUncond(1), false);
  assertEquals(needsUncond(4), true);
  assertEquals(needsUncond(0), true);
});

Deno.test("cfgEulerStep: CFG=1 は cond をそのまま使う（uncond を経由しない）", () => {
  const previous = Float32Array.from([1, 2, 3]);
  const cond = Float32Array.from([0.25, -0.5, 0.125]);
  const got = cfgEulerStep(previous, cond, undefined, -0.5, 1);
  assertEquals([...got], [1 - 0.125, 2 + 0.25, 3 - 0.0625]);
});

Deno.test("cfgEulerStep: uncond の有無が needsUncond と食い違ったら落とす", () => {
  const previous = Float32Array.from([0, 0]);
  const cond = Float32Array.from([1, 1]);
  const uncond = Float32Array.from([2, 2]);
  assertThrows(
    () => cfgEulerStep(previous, cond, undefined, -0.1, 4),
    Error,
    "uncond 側の DiT 出力を要求する",
  );
  assertThrows(
    () => cfgEulerStep(previous, cond, uncond, -0.1, 1),
    Error,
    "uncond 分岐を計算しない経路",
  );
});

Deno.test("cfgEulerStep: CFG!=1 は uncond + scale·(cond − uncond) を Euler へ渡す", () => {
  const previous = Float32Array.from([0]);
  const got = cfgEulerStep(
    previous,
    Float32Array.from([1]),
    Float32Array.from([0.5]),
    -2,
    4,
  );
  // noise = 0.5 + 4·(1 − 0.5) = 2.5 → 0 + (−2)·2.5 = −5
  assertEquals([...got], [-5]);
});
