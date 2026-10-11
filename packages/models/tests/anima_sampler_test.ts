// Anima のサンプラ（IR に載らない段）の挙動テスト。GPU も資産も要らない純関数。
//
// 参照フィクスチャとの数値パリティ（timesteps_proj 表の突合）は **実 GPU / 実資産の E2E と
// 同じ波**で復帰させる（P3 波 2）。ここは形と不変条件を押さえる。
// NOTE: `latents_mean` / `latents_std` のビット一致は資産不要の門になった（参照フィクスチャを
// 切り出して commit してある — `anima_latents_test.ts`）。

import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { cfgEulerStep, sigmaSchedule, timestepsProj } from "../src/anima/sampler.ts";
import { ModelInputError } from "../src/errors.ts";
// `needsUncond` の定義は更新則に依らない共通処理側にある（`sampler.ts` は import して使う）。
import { needsUncond } from "../src/generation/dpm-solver-multistep.ts";

/** `models/karume-anima/karume.json` の `pipelineConfig.scheduler` と同じ値（形の検査用）。 */
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
  assertThrows(() => sigmaSchedule(1, SHIFT), ModelInputError);
  assertThrows(() => sigmaSchedule(2.5, SHIFT), ModelInputError);
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

Deno.test("sigmaSchedule: 代表 shift で梯子の先頭が厳密に 1（DPM++ 2M の前提）", () => {
  // `dpm-solver-multistep.ts` は「先頭が厳密に 1」から step 0 の `alphaS0 = 0` を導き、自身は
  // 例外を出さない。上側のずれは `sigmaSchedule` の門が拒否し、下側の ulp 級のずれは受理する
  // （下の describe）。代表 shift のこの 4 値は先頭が厳密に 1 になることを固定する。
  for (const shift of [1, 1.5, 3, 12]) {
    assertStrictEquals(sigmaSchedule(8, shift)[0], 1, `shift=${shift} の先頭`);
  }
});

Deno.test("sigmaSchedule: 梯子の成立域の外は fail loudly（黙って別軌道を走らせない）", () => {
  // 分母 `f32(shift−1)+1` が f32 で厳密に 0 へ潰れる域 → 先頭が +Inf。
  assertThrows(() => sigmaSchedule(8, 1e-8), RangeError, "梯子の成立域の外");
  // 潰れはしないが先頭が 1 でなくなる域（実測 0.8388…）。
  assertThrows(() => sigmaSchedule(8, 1e-7), RangeError, "梯子の成立域の外");
  // 先頭は 1 のまま、直後が f32 の刻みに潰れて同値になる域（狭義単調減少が破れる側）。
  assertThrows(() => sigmaSchedule(8, 1e7), RangeError, "狭義単調減少でない");
});

describe("sigmaSchedule: 先頭が 1 から ulp 級にずれる小数 shift", () => {
  // DPM++ 2M は step 0 で `log(1 − σ₀)` を取るので、σ₀ が 1 を 1 ulp でも超えると負数の log で
  // NaN になり、全 step を NaN で走った末に遅れて落ちる。σ₀ < 1 側のずれは `alphaS0 ≈ 6e-8` で無害。
  it("σ₀ が 1 を超える shift（1.32 → 1 + 2⁻²³）は GPU に触る前に RangeError で拒否する", () => {
    // 「梯子の成立域の外」は単調性の門も出すので、先頭の門の文言と σ₀ の値まで照合して
    // 上側の枝で落ちたことを固定する。
    assertThrows(
      () => sigmaSchedule(8, 1.32),
      RangeError,
      `sigma 梯子の先頭が ${1 + 2 ** -23}（1 を超えている`,
    );
  });

  it("σ₀ が 1 を下回る側にずれる shift（1.18 → 1 − 2⁻²³）は受理する", () => {
    const sigmas = sigmaSchedule(8, 1.18);
    assertStrictEquals(
      sigmas[0],
      1 - 2 ** -23,
      "1.18 は下側に 2 ulp（f32 の 1 未満の刻み 2⁻²⁴ × 2）ずれる",
    );
  });
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

Deno.test("cfgEulerStep: f32 で表せない倍率は先に f32 へ丸めてから掛ける（torch と最終桁まで一致）", () => {
  // torch は Python float の倍率を f32 テンソルとの積の前に f32 へ落とす。f64 のまま掛けると、
  // この組では合成が 1 ulp ずれる（0.03789997100830078）。期待値は torch 2.13.0 の CPU で
  // `uncond + 1.1 * (cond - uncond)` を f32 テンソルで評価した値。
  const got = cfgEulerStep(
    Float32Array.from([0]),
    Float32Array.from([0.011]),
    Float32Array.from([0.5]),
    -1,
    1.1,
  );
  assertStrictEquals(got[0], 0.03790003061294556);
});
