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
