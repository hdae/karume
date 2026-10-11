// 停止文字列フィルタ（`src/text/detokenizer.ts`）の**失敗の分類**だけを見る門。GPU も実資産も
// 要らない。逐次復号そのものの挙動は `gemma4_chat_test.ts` が縛っており、ここで重ねない。
//
// 縛るのは ADR 0107 の分類: 停止文字列の指定は呼び手が書いた値なので、空文字列も重複も
// `ModelInputError`（HTTP サーバーなら 400）で落ちる。型が家族ごとに割れていると、複数家族を
// 載せたホストは 400 を切り出すのに**メッセージの綴りを読む**しかなくなる。

import { assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { createStopStringFilter } from "../src/text/detokenizer.ts";

Deno.test("createStopStringFilter: 受理できない停止文字列は入力起因で落ちる", async (t) => {
  await t.step("空文字列（常に一致する = 1 文字も出せない指定）", () => {
    assertThrows(
      () => createStopStringFilter(["END", ""]),
      ModelInputError,
      "stopStrings[1] が空文字列",
    );
  });

  await t.step("重複（同じ条件を 2 度書いた以上の意味を持てない）", () => {
    assertThrows(
      () => createStopStringFilter(["END", "END"]),
      ModelInputError,
      'stopStrings に "END" が 2 度出る',
    );
  });

  await t.step("受理される指定は通る（門が恒真に落ちていないことの対）", () => {
    createStopStringFilter([]);
    createStopStringFilter(["END", "\n\n"]);
  });
});

Deno.test("createStopStringFilter: 型を外した指定は暗黙変換せずに入力起因で落ちる", async (t) => {
  /** 型を外した JS / JSON 由来の指定（TS の型検査を通らない値を門へ届ける）。 */
  const untyped = (value: unknown): readonly string[] => value as readonly string[];

  await t.step("文字列そのもの（文字の配列に展開して最初の 1 文字で止めない）", () => {
    // "END" を展開すると "E" / "N" / "D" の 3 本になり、"SENTENCE" を最初の E で止める。
    assertThrows(() => createStopStringFilter(untyped("END")), ModelInputError, "stopStrings");
  });

  await t.step("配列でない iterable（Set）", () => {
    assertThrows(
      () => createStopStringFilter(untyped(new Set(["END"]))),
      ModelInputError,
      "stopStrings",
    );
  });

  await t.step("文字列でない要素（indexOf の暗黙変換で別の位置に止めない）", () => {
    // 123 は "123" として扱われ、"x123y" を "x" で止める。
    for (const element of [123, null, undefined]) {
      assertThrows(
        () => createStopStringFilter(untyped(["END", element])),
        ModelInputError,
        "stopStrings[1]",
        String(element),
      );
    }
  });

  await t.step("疎な配列の穴（forEach のように飛ばして黙って捨てない）", () => {
    // [, "END"] の 0 番は穴。穴を飛ばすと "END" だけの指定として受理されてしまう。
    const sparse = [, "END"];
    assertThrows(
      () => createStopStringFilter(untyped(sparse)),
      ModelInputError,
      "stopStrings[0] undefined",
    );
  });
});
