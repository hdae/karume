// 後始末の段並べ（`src/session/dispose-steps.ts` の `disposeSteps`）の門 — 常駐 Session を
// 持つ家族の `dispose()` が共有する 1 本。GPU も資産も要らない。
//
// 縛るのは 3 点:
//
// ① **前段が失敗しても後段は必ず走る**。各家族の `openXState` は catch で「内部で取った GPU は
//    必ず返す」を掲げているが、`dispose()` が `await session.dispose(); gpu.destroy();` の直列
//    2 段だと `Session.dispose()` の reject（`RunArena` が flush の失敗を伝播させる）で後段へ
//    到達しない。しかも `#disposal` に rejected promise が居座るので 2 度目も再試行にならない。
// ② **失敗 1 件はその例外そのもの**が投げられる（呼び手が原因で分岐できる形を壊さない）。
// ③ **失敗 2 件以上は `AggregateError`** で全部運ぶ（`finally` で後段の例外が前段を上書きする
//    形＝どの段が最初に壊れたかが消える形へ戻さない）。
//
// NOTE: 家族側の `dispose()` が「内部取得の GPU だけを破棄する」「2 度目も同じ完了を返す」ことは
// `#disposal ??= #chain(...)` の側の性質で、実 GPU の e2e が踏む（内部取得の GPU をスタブに
// 差し替える口が無いため、ここでは helper の振る舞いだけを縛る）。

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { disposeSteps } from "../src/session/dispose-steps.ts";

Deno.test("disposeSteps: 全段が成功すれば宣言順に 1 回ずつ走って解決する", async () => {
  const order: string[] = [];
  await disposeSteps([
    async () => {
      await Promise.resolve();
      order.push("session");
    },
    () => {
      order.push("gpu");
    },
  ]);
  assertEquals(order, ["session", "gpu"]);
});

Deno.test("disposeSteps: 前段が reject しても後段は必ず 1 回走り、元の失敗がそのまま投げられる", async () => {
  const lost = new Error("device lost");
  let destroyCount = 0;
  const caught = await assertRejects(
    () =>
      disposeSteps([
        () => Promise.reject(lost),
        () => {
          destroyCount += 1;
        },
      ]),
    Error,
    "device lost",
  );
  // 「元の失敗そのもの」= 包み直していない（呼び手が原因で分岐できる）。
  assertStrictEquals(caught, lost);
  assertEquals(destroyCount, 1, "後段（gpu.destroy 相当）へ到達していない");
});

Deno.test("disposeSteps: 後段が投げても前段の失敗を上書きしない（2 件は AggregateError）", async () => {
  const lost = new Error("device lost");
  const destroyFailed = new Error("destroy failed");
  const caught = await assertRejects(
    () => disposeSteps([() => Promise.reject(lost), () => Promise.reject(destroyFailed)]),
    AggregateError,
  );
  assert(caught instanceof AggregateError, `AggregateError でない: ${caught}`);
  // 2 件とも運ぶ（`try/finally` 版はここで前段が消える）。順序は宣言順。
  assertEquals(caught.errors, [lost, destroyFailed]);
});

Deno.test("disposeSteps: 同期 throw の段も後段を止めない", async () => {
  let destroyCount = 0;
  await assertRejects(
    () =>
      disposeSteps([
        () => {
          throw new Error("sync failure");
        },
        () => {
          destroyCount += 1;
        },
      ]),
    Error,
    "sync failure",
  );
  assertEquals(destroyCount, 1);
});
