// 直列化鎖（`src/concurrency/serial.ts`）の挙動テスト。GPU も実資産も要らない。
//
// パイプライン 3 家族が「グラフは 1 本ずつ」を公開 API 側で守る土台なので、実 GPU の並行
// generate を待たずにここで縛る。決着の順を握るために、操作の中身は**手で決着させる
// promise**（`Promise.withResolvers`）にしてある — タイマー待ちに頼ると順序が緩む。

import { assertEquals, assertRejects } from "@std/assert";
import { createOperationChain } from "../src/concurrency/serial.ts";

/** 積まれた操作が進めるところまで進むのを待つ（microtask を全て流すマクロタスク 1 拍）。 */
const settleMicrotasks = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

Deno.test("createOperationChain: 並行に積んでも 1 本ずつ走る（後発は先発の決着まで開始しない）", async () => {
  const chain = createOperationChain();
  const events: string[] = [];
  const first = Promise.withResolvers<void>();
  const second = Promise.withResolvers<void>();

  const a = chain(async () => {
    events.push("a:start");
    await first.promise;
    events.push("a:end");
    return "a";
  });
  const b = chain(async () => {
    events.push("b:start");
    await second.promise;
    events.push("b:end");
    return "b";
  });

  await settleMicrotasks();
  assertEquals(events, ["a:start"]);

  first.resolve();
  assertEquals(await a, "a");
  await settleMicrotasks();
  assertEquals(events, ["a:start", "a:end", "b:start"]);

  second.resolve();
  assertEquals(await b, "b");
  assertEquals(events, ["a:start", "a:end", "b:start", "b:end"]);
});

Deno.test("createOperationChain: 失敗は呼び出し側へそのまま届き、鎖は次へ解放される", async () => {
  const chain = createOperationChain();
  const events: string[] = [];

  const failing = chain(() => {
    events.push("failing");
    throw new Error("boom");
  });
  const rejected = assertRejects(() => failing, Error, "boom");
  const next = chain(() => {
    events.push("next");
    return 7;
  });

  assertEquals(await next, 7);
  assertEquals(events, ["failing", "next"]);
  await rejected;
});

Deno.test("createOperationChain: 後から積んだ操作は先行が全て決着してから走る（dispose が待つ形）", async () => {
  const chain = createOperationChain();
  const done: string[] = [];
  const gate = Promise.withResolvers<void>();

  const slow = chain(async () => {
    await gate.promise;
    done.push("slow");
  });
  const failing = chain(() => {
    done.push("failing");
    throw new Error("boom");
  });
  const rejected = assertRejects(() => failing, Error, "boom");
  // 破棄に相当する操作（同期）。先行が 1 本でも生きている間は走ってはならない。
  const close = chain(() => {
    done.push("close");
  });

  await settleMicrotasks();
  assertEquals(done, []);

  gate.resolve();
  await close;
  assertEquals(done, ["slow", "failing", "close"]);
  await slow;
  await rejected;
});
