// Anima のサンプラ（IR に載らない段）の挙動テスト。GPU も資産も要らない純関数。
//
// 参照フィクスチャとの数値パリティ（timesteps_proj 表の突合）は **実 GPU / 実資産の E2E と
// 同じ波**で復帰させる（P3 波 2）。ここは形と不変条件を押さえる。
// NOTE: `latents_mean` / `latents_std` のビット一致は資産不要の門になった（参照フィクスチャを
// 切り出して commit してある — `anima_latents_test.ts`）。

import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { imageToRgba } from "../src/anima/image.ts";
import {
  assertAcceptableSteps,
  cfgEulerStep,
  sigmaSchedule,
  timestepsProj,
} from "../src/anima/sampler.ts";
import { assertPromptTokenLengths } from "../src/anima/text/tokenizer.ts";
import { ModelInputError } from "../src/errors.ts";
import { assertEncodableText } from "../src/text/code-points.ts";
// `needsUncond` の定義は更新則に依らない共通処理側にある（`sampler.ts` は import して使う）。
import { needsUncond } from "../src/generation/dpm-solver-multistep.ts";

/** `models/karume-anima/karume.json` の `pipelineConfig.scheduler` と同じ値（形の検査用）。 */
const SHIFT = 3;
const NUM_TRAIN_TIMESTEPS = 1000;

test("sigmaSchedule: 長さ steps+1・終端 0・単調減少", () => {
  const sigmas = sigmaSchedule(8, SHIFT);
  assertEquals(sigmas.length, 9);
  assertEquals(sigmas[0], 1, "始点は linspace の start=1（shift を掛けても 3/3 = 1）");
  assertEquals(sigmas[8], 0, "終端 0 が 1 つ足される");
  for (let index = 1; index < 8; index += 1) {
    assert(sigmas[index] < sigmas[index - 1], `σ が単調減少でない（index=${index}）`);
  }
});

test("sigmaSchedule: steps < 2 は linspace の分母が 0 になるので落とす", () => {
  assertThrows(() => sigmaSchedule(1, SHIFT), ModelInputError);
  assertThrows(() => sigmaSchedule(2.5, SHIFT), ModelInputError);
});

test("sigmaSchedule: shift は manifest の値で効く（1 なら素の linspace）", () => {
  // shift=1 では `1·σ / (0·σ + 1) = σ` に潰れる。scheduler の値を manifest から取る配線が
  // 死んで定数に固定されていたら、この 1 本が落ちる。
  const plain = sigmaSchedule(4, 1);
  assertEquals(
    [...plain].slice(0, 4).map((value) => Math.fround(value)),
    [
      Math.fround(1),
      Math.fround(1 + (0.25 - 1) / 3),
      Math.fround(1 + (2 * (0.25 - 1)) / 3),
      Math.fround(0.25),
    ],
  );
  const shifted = sigmaSchedule(4, SHIFT);
  assert(shifted[1] !== plain[1], "shift が効いていない");
});

test("sigmaSchedule: 代表 shift で梯子の先頭が厳密に 1（DPM++ 2M の前提）", () => {
  // `dpm-solver-multistep.ts` は「先頭が厳密に 1」から step 0 の `alphaS0 = 0` を導き、自身は
  // 例外を出さない。上側のずれは `sigmaSchedule` の門が拒否し、下側の ulp 級のずれは受理する
  // （下の describe）。代表 shift のこの 4 値は先頭が厳密に 1 になることを固定する。
  for (const shift of [1, 1.5, 3, 12]) {
    assertStrictEquals(sigmaSchedule(8, shift)[0], 1, `shift=${shift} の先頭`);
  }
});

test("sigmaSchedule: 梯子の成立域の外は fail loudly（黙って別軌道を走らせない）", () => {
  // 分母 `f32(shift−1)+1` が f32 で厳密に 0 へ潰れる域 → 先頭が +Inf。
  assertThrows(
    () => sigmaSchedule(8, 1e-8),
    RangeError,
    "is outside the range where the ladder holds",
  );
  // 潰れはしないが先頭が 1 でなくなる域（実測 0.8388…）。
  assertThrows(
    () => sigmaSchedule(8, 1e-7),
    RangeError,
    "is outside the range where the ladder holds",
  );
  // 先頭は 1 のまま、直後が f32 の刻みに潰れて同値になる域（狭義単調減少が破れる側）。
  assertThrows(() => sigmaSchedule(8, 1e7), RangeError, "is not strictly decreasing");
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
      `The head of the sigma ladder is ${1 + 2 ** -23} (above 1`,
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

test("timestepsProj: 前半 cos・後半 sin（flip_sin_to_cos=true）", () => {
  // sigma=0 → timestep=0 → 全周波数で angle=0 なので cos 側が 1・sin 側が 0 になる。
  // 反転していると前半が 0・後半が 1 になり、この 1 本で落ちる。
  const proj = timestepsProj(0, 8, NUM_TRAIN_TIMESTEPS);
  assertEquals([...proj], [1, 1, 1, 1, 0, 0, 0, 0]);
});

test("timestepsProj: 幅と num_train_timesteps の受理集合", () => {
  assertThrows(() => timestepsProj(0.5, 7, NUM_TRAIN_TIMESTEPS), RangeError);
  assertThrows(() => timestepsProj(0.5, 0, NUM_TRAIN_TIMESTEPS), RangeError);
  assertThrows(() => timestepsProj(0.5, 8, 0), RangeError);
  assertThrows(() => timestepsProj(0.5, 8, 1000.5), RangeError);
});

test("needsUncond: CFG=1 だけが uncond 不要", () => {
  assertEquals(needsUncond(1), false);
  assertEquals(needsUncond(4), true);
  assertEquals(needsUncond(0), true);
});

test("cfgEulerStep: CFG=1 は cond をそのまま使う（uncond を経由しない）", () => {
  const previous = Float32Array.from([1, 2, 3]);
  const cond = Float32Array.from([0.25, -0.5, 0.125]);
  const got = cfgEulerStep(previous, cond, undefined, -0.5, 1);
  assertEquals([...got], [1 - 0.125, 2 + 0.25, 3 - 0.0625]);
});

test("cfgEulerStep: uncond の有無が needsUncond と食い違ったら落とす", () => {
  const previous = Float32Array.from([0, 0]);
  const cond = Float32Array.from([1, 1]);
  const uncond = Float32Array.from([2, 2]);
  assertThrows(
    () => cfgEulerStep(previous, cond, undefined, -0.1, 4),
    Error,
    "requires the DiT output of the uncond side",
  );
  assertThrows(
    () => cfgEulerStep(previous, cond, uncond, -0.1, 1),
    Error,
    "path should not compute the uncond branch",
  );
});

test("cfgEulerStep: CFG!=1 は uncond + scale·(cond − uncond) を Euler へ渡す", () => {
  const previous = Float32Array.from([0]);
  const got = cfgEulerStep(previous, Float32Array.from([1]), Float32Array.from([0.5]), -2, 4);
  // noise = 0.5 + 4·(1 − 0.5) = 2.5 → 0 + (−2)·2.5 = −5
  assertEquals([...got], [-5]);
});

test("cfgEulerStep: f32 で表せない倍率は先に f32 へ丸めてから掛ける（torch と最終桁まで一致）", () => {
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

// ---- Host-facing error type (ADR 0107) --------------------------------------
// Merged from the former `anima_sampler_test.ts`. Its resolution, seed and prompt-lower-bound
// cases were dropped as literal duplicates of `anima_resolution_test.ts`, `anima_random_test.ts`
// and `anima_tokenizer_test.ts`; what stays here is what no family test checks.
//
// 併せて**逆側**も縛る: 内部不変条件の破れがこの型で飛ぶと、ホストは 500 相当を「入力を
// 直せ」と呼び手へ返す。型を広げすぎた退行は入力側のテストでは捕まらない。

describe("Anima の生成要求が受理できないとき", () => {
  it("steps の門と sigma の梯子が同じ ModelInputError・同じ診断を出す", () => {
    // 観測できるのは「2 経路の型と診断が一致する」ことまで。`generate` の入口がこの門を
    // 先に呼ぶこと（ADR 0107 決定 6 の先行拒否）はここからは見えない — 入口へ届くには実資産と
    // GPU が要るので、その順序は GPU レーンの e2e が受け持つ。
    const direct = assertThrows(() => assertAcceptableSteps(1), ModelInputError);
    const viaSchedule = assertThrows(() => sigmaSchedule(1, 3), ModelInputError);
    assertEquals(direct.message, viaSchedule.message, "steps の診断が 2 か所で割れている");
  });

  it("steps の上限は 1000（ちょうどは梯子まで組め、超えると梯子を確保する前に ModelInputError）", () => {
    assertAcceptableSteps(1000);
    assertEquals(sigmaSchedule(1000, 3).length, 1001, "上限ちょうどは境界の内");
    const direct = assertThrows(
      () => assertAcceptableSteps(1001),
      ModelInputError,
      "is not at most 1000",
    );
    const viaSchedule = assertThrows(() => sigmaSchedule(1001, 3), ModelInputError);
    assertEquals(direct.message, viaSchedule.message, "steps の診断が 2 か所で割れている");
    // 門が配列の確保より先に効くことの証拠: 上限の無いコードでは `new Float32Array(2⁵³)` が即座に
    // `RangeError`（Invalid typed array length）を投げるので、型が違ってこの 1 行が赤になる。
    assertThrows(
      () => sigmaSchedule(Number.MAX_SAFE_INTEGER, 3),
      ModelInputError,
      "is not at most 1000",
    );
  });

  it("対にならないサロゲートを含む本文は ModelInputError で飛ぶ", () => {
    // Kept although `anima_tokenizer_test.ts` covers the rejecting side: the accepting side
    // (a paired surrogate) is checked only here, against the owner directly.
    // 条件の正本は `toCodePoints` 1 本のまま。生成要求の本文として通ったときだけ型が上がる。
    const error = assertThrows(
      () => assertEncodableText("a\ud800b", "プロンプト"),
      ModelInputError,
    );
    assert(error.cause instanceof Error, "所有者の診断を cause に残す");
    assert(!(error.cause instanceof ModelInputError), "cause は共有ヘルパの素の Error");
    assertEncodableText("a𐀀b", "プロンプト");
  });
});

describe("Anima の内部不変条件が破れたとき", () => {
  it("プロンプトの id 列が上限を超えるのは ModelInputError ではない（切り詰めの破れ）", () => {
    // 両トークナイザは `maxLength` で切り詰めてから返すので、上限超過はそこが壊れたときにしか
    // 届かない = 呼び手がプロンプトを短くしても直らない 500 相当。
    const error = assertThrows(
      () => assertPromptTokenLengths("プロンプト", 513, 4, 512),
      Error,
      "upper bound 512",
    );
    assert(!(error instanceof ModelInputError), "500 相当が入力起因の型で飛んでいる");
  });

  it("VAE 出力の要素数と寸法の食い違いは ModelInputError ではない（呼び手は直せない）", () => {
    // `imageToRgba` の寸法は幾何（latent の全長 × 縮尺）から来るので、ここが割れているのは
    // 開いた export か配線の破れであって、要求を直しても直らない = 500 相当。
    const error = assertThrows(
      () => imageToRgba(new Float32Array(5), 2, 1),
      Error,
      "element count",
    );
    assert(!(error instanceof ModelInputError), "500 相当が入力起因の型で飛んでいる");
  });
});
