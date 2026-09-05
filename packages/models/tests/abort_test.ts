// 段の境目の中断検査（`src/concurrency/abort.ts` の `settleAbort`）の門。GPU も資産も要らない。
//
// 縛るのは 4 点:
//
// ① `signal` 未指定なら**1 度も譲らない**（購読していない呼び出しにタスク 1 往復を乗せない）。
// ② 既に中断済みなら `signal.reason` **そのもの**を投げる（包み直さない）。
// ③ **譲っている間に届いた中断を観測する**（本命）。`abort()` の配送はタスクなので、譲り先を
//    マイクロタスク（`await Promise.resolve()`）へ書き換えると「現在のタスクの中に留まったまま」
//    になり、中断が**効かない**という沈黙で退行する。恒真でないことは、同じ形をマイクロタスク
//    譲りで書いた擬似実装がその中断を拾わないことで示す（陽性対照）。
// ④ 中断しなければ resolve する（門が常に投げる形へ倒れていない）。
//
// NOTE: 譲り先が `setTimeout` か `MessageChannel` かは③④の観測では区別できない（どちらも
// タスク）。そこは実装側の MUST（非表示タブの timer throttling を生成速度に持ち込まない）で、
// ブラウザ実測の話なのでここでは縛らない。ここが縛るのは「マクロタスクへ譲っている」こと。

import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { settleAbort } from "../src/concurrency/abort.ts";

/** 譲り先をマイクロタスクにした擬似実装（③の陽性対照 — 退行したときの姿）。 */
const settleAbortViaMicrotask = async (signal: AbortSignal | undefined): Promise<void> => {
  if (signal === undefined) return;
  await Promise.resolve();
  signal.throwIfAborted();
};

Deno.test("settleAbort: signal 未指定なら 1 度も譲らない（マクロタスクを挟まない）", async () => {
  const order: string[] = [];
  // 対照 = 先に仕掛けたタスク 1 本。`settleAbort` が譲っていればこちらが先に決着する
  // （マイクロタスクだけで済む決着は、既に待ち行列にあるタスクより必ず先に来る）。
  const task = new Promise<void>((resolve) => setTimeout(resolve, 0))
    .then(() => order.push("task"));
  const settled = settleAbort(undefined).then(() => order.push("settleAbort"));
  await Promise.all([task, settled]);
  assertEquals(order, ["settleAbort", "task"]);
});

Deno.test("settleAbort: signal 付きならマクロタスクへ譲る（マイクロタスクより後に決着する）", async () => {
  const order: string[] = [];
  const controller = new AbortController();
  const settled = settleAbort(controller.signal).then(() => order.push("settleAbort"));
  // マイクロタスクを数段挟んでも、タスクへ譲っている限りこちらが先に終わる。
  const microtasks = Promise.resolve().then(() => Promise.resolve()).then(() =>
    order.push("microtask")
  );
  await Promise.all([microtasks, settled]);
  assertEquals(order, ["microtask", "settleAbort"]);
});

Deno.test("settleAbort: 既に中断済みなら reason そのものを投げる", async () => {
  const reason = new Error("ロードを打ち切った");
  const caught = await assertRejects(
    () => settleAbort(AbortSignal.abort(reason)),
    Error,
    "ロードを打ち切った",
  );
  assertStrictEquals(caught, reason);
});

Deno.test("settleAbort: 譲っている間に届いた abort() を観測する", async () => {
  const reason = new Error("マイクロタスク経由の中断");
  const controller = new AbortController();
  const settled = settleAbort(controller.signal);
  // 中断はマイクロタスクチェーンから届く = `settleAbort` を呼んだ時点ではまだ中断していない。
  await Promise.resolve().then(() => Promise.resolve()).then(() => controller.abort(reason));
  const caught = await assertRejects(() => settled, Error, "マイクロタスク経由の中断");
  assertStrictEquals(caught, reason);

  // 陽性対照 — マイクロタスク譲りの擬似実装は同じ中断を**拾わない**（= このケースは恒真でない）。
  const other = new AbortController();
  const missed = settleAbortViaMicrotask(other.signal);
  await Promise.resolve().then(() => Promise.resolve()).then(() => other.abort(reason));
  await missed;
});

Deno.test("settleAbort: 中断されなければ解決する", async () => {
  const controller = new AbortController();
  await settleAbort(controller.signal);
  assertEquals(controller.signal.aborted, false);
});
