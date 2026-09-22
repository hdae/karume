// Anima の入力起因エラーがホストから 1 本の型で捕まること（ADR
// [0107](../../../docs/decisions/0107-model-input-error.md)）。GPU も実資産も要らない。
//
// 家族ごとのテスト（解像度 / サンプラ / 乱数 / トークナイザ）は**それぞれの受理集合**を見て
// いて、「型が 1 本に揃っているか」は誰も見ていない。複数の家族を同じホストに載せた側は
// `instanceof ModelInputError` だけで 400 / 500 を分けるので、その分岐が成立することを
// 家族の入口ごとに固定する。
//
// 併せて**逆側**も縛る: 内部不変条件の破れがこの型で飛ぶと、ホストは 500 相当を「入力を
// 直せ」と呼び手へ返す。型を広げすぎた退行は入力側のテストでは捕まらない。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { imageToRgba } from "../src/anima/image.ts";
import { Randn } from "../src/anima/random.ts";
import { assertAcceptableResolution, parseResolution } from "../src/anima/resolution.ts";
import { assertAcceptableSteps, sigmaSchedule } from "../src/anima/sampler.ts";
import { assertPromptTokenLengths } from "../src/anima/text/tokenizer.ts";
import { ModelInputError } from "../src/errors.ts";

describe("Anima の生成要求が受理できないとき", () => {
  it("解像度の綴りと値域は ModelInputError で飛ぶ", () => {
    assertThrows(() => parseResolution("1344*768"), ModelInputError);
    assertThrows(() => assertAcceptableResolution({ width: 1000, height: 1024 }), ModelInputError);
  });

  it("steps は入口でも梯子でも同じ ModelInputError で飛ぶ（受理集合の所有者は 1 本）", () => {
    const direct = assertThrows(() => assertAcceptableSteps(1), ModelInputError);
    const viaSchedule = assertThrows(() => sigmaSchedule(1, 3), ModelInputError);
    assertEquals(direct.message, viaSchedule.message, "steps の診断が 2 か所で割れている");
  });

  it("seed は ModelInputError で飛ぶ（生成器を作る前に落ちる）", () => {
    assertThrows(() => new Randn(-1), ModelInputError);
  });

  it("プロンプトの id 列長は ModelInputError で飛ぶ", () => {
    assertThrows(() => assertPromptTokenLengths("プロンプト", 1, 4, 512), ModelInputError);
    assertThrows(() => assertPromptTokenLengths("プロンプト", 513, 4, 512), ModelInputError);
  });
});

describe("Anima の内部不変条件が破れたとき", () => {
  it("VAE 出力の要素数と寸法の食い違いは ModelInputError ではない（呼び手は直せない）", () => {
    // `imageToRgba` の寸法は幾何（latent の全長 × 縮尺）から来るので、ここが割れているのは
    // 開いた export か配線の破れであって、要求を直しても直らない = 500 相当。
    const error = assertThrows(() => imageToRgba(new Float32Array(5), 2, 1), Error, "要素数");
    assert(!(error instanceof ModelInputError), "500 相当が入力起因の型で飛んでいる");
  });
});
