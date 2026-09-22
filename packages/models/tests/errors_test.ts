// 家族横断の入力起因エラー（ADR
// [0107](../../../docs/decisions/0107-model-input-error.md)）の挙動テスト。GPU も実資産も要らない。
//
// 縛るのは 2 点:
//
// - **`instanceof` で分岐できること** — ホストが 400 / 500 を分ける唯一の手段なので、派生
//   2 本が親を通ること自体が仕様である（`extends` の付け替えは型検査では咎められない）。
// - **受理集合の所有者が 1 本であること** — `assertAcceptableSeed` の境界（0 と
//   `Number.MAX_SAFE_INTEGER` を通し、その外を落とす）を値で固定する。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { ModelInputError } from "../src/errors.ts";
import { assertAcceptableSeed } from "../src/request-gates.ts";
import { Sbv2InputError } from "../src/sbv2/errors.ts";
import { GenerationCapacityError } from "../src/generation/sequence.ts";

describe("ModelInputError", () => {
  it("name は綴りどおりで、ログと err.name の分岐から読める", () => {
    assertEquals(new ModelInputError("値域の外").name, "ModelInputError");
  });

  it("message をそのまま保つ", () => {
    const error = new ModelInputError("seed -1 が非負の安全整数でない");
    assertEquals(error.message, "seed -1 が非負の安全整数でない");
  });

  it("cause を透過する（包み直しても原因を辿れる）", () => {
    const cause = new Error("元の失敗");
    assertEquals(new ModelInputError("受理できない", { cause }).cause, cause);
  });

  it("Error の派生である（既存の catch の枝を壊さない）", () => {
    assert(new ModelInputError("受理できない") instanceof Error);
  });
});

describe("家族の入力起因エラーをホストから見たとき", () => {
  it("Sbv2InputError は ModelInputError で捕まる（name は SBV2 のまま）", () => {
    const error = new Sbv2InputError("スタイル名が配布形に無い");
    assert(error instanceof ModelInputError);
    assertEquals(error.name, "Sbv2InputError");
  });

  it("GenerationCapacityError は ModelInputError で捕まる（切り詰めの欄は保たれる）", () => {
    const error = new GenerationCapacityError("会話が容量に入らない", {
      constraint: "capacity",
      pastLength: 100,
      promptLength: 40,
      requestedNewTokens: 64,
      limit: 128,
      maxNewTokens: -11,
    });
    assert(error instanceof ModelInputError);
    assertEquals(error.name, "GenerationCapacityError");
    assertEquals([error.constraint, error.maxNewTokens], ["capacity", -11]);
  });
});

describe("assertAcceptableSeed（3 家族が共有する seed の受理集合）", () => {
  it("非負の安全整数を通す（両端を含む）", () => {
    assertAcceptableSeed(0);
    assertAcceptableSeed(Number.MAX_SAFE_INTEGER);
  });

  it("負・端数・NaN・安全整数の外を ModelInputError で落とす", () => {
    for (const seed of [-1, 1.5, Number.NaN, 2 ** 53]) {
      assertThrows(
        () => assertAcceptableSeed(seed),
        ModelInputError,
        "非負の安全整数でない",
      );
    }
  });
});
