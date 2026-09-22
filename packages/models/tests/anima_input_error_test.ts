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
import { assertEncodableText } from "../src/text/code-points.ts";

describe("Anima の生成要求が受理できないとき", () => {
  it("解像度の綴りと値域は ModelInputError で飛ぶ", () => {
    assertThrows(() => parseResolution("1344*768"), ModelInputError);
    assertThrows(() => assertAcceptableResolution({ width: 1000, height: 1024 }), ModelInputError);
  });

  it("steps の門と sigma の梯子が同じ ModelInputError・同じ診断を出す", () => {
    // 観測できるのは「2 経路の型と診断が一致する」ことまで。`generate` の入口がこの門を
    // 先に呼ぶこと（ADR 0107 決定 6 の先行拒否）はここからは見えない — 入口へ届くには実資産と
    // GPU が要るので、その順序は GPU レーンの e2e が受け持つ。
    const direct = assertThrows(() => assertAcceptableSteps(1), ModelInputError);
    const viaSchedule = assertThrows(() => sigmaSchedule(1, 3), ModelInputError);
    assertEquals(direct.message, viaSchedule.message, "steps の診断が 2 か所で割れている");
  });

  it("seed の値域違反は乱数生成器のコンストラクタで ModelInputError になる", () => {
    assertThrows(() => new Randn(-1), ModelInputError);
  });

  it("プロンプトの id 列が下限に満たないと ModelInputError で飛ぶ", () => {
    assertThrows(() => assertPromptTokenLengths("プロンプト", 1, 4, 512), ModelInputError);
  });

  it("対にならないサロゲートを含む本文は ModelInputError で飛ぶ", () => {
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
      "上限 512",
    );
    assert(!(error instanceof ModelInputError), "500 相当が入力起因の型で飛んでいる");
  });

  it("VAE 出力の要素数と寸法の食い違いは ModelInputError ではない（呼び手は直せない）", () => {
    // `imageToRgba` の寸法は幾何（latent の全長 × 縮尺）から来るので、ここが割れているのは
    // 開いた export か配線の破れであって、要求を直しても直らない = 500 相当。
    const error = assertThrows(() => imageToRgba(new Float32Array(5), 2, 1), Error, "要素数");
    assert(!(error instanceof ModelInputError), "500 相当が入力起因の型で飛んでいる");
  });
});
