import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { closeableGenerator } from "../src/concurrency/closeable-generator.ts";

Deno.test("未開始の終了は本体を実行せず、一度だけ通知する", async () => {
  for (const method of ["return", "throw"] as const) {
    let entered = 0;
    let finalized = 0;
    const failures: unknown[] = [];
    const error = new Error("closed before next");
    const iterator = closeableGenerator(
      (async function* () {
        entered += 1;
        yield 1;
      })(),
      (failure) => {
        finalized += 1;
        failures.push(failure?.error);
      },
    );
    if (method === "return") await iterator.return();
    else assertStrictEquals(await assertRejects(() => iterator.throw(error)), error);
    await iterator.return();
    assertEquals(await iterator.next(), { done: true, value: undefined });
    assertEquals({ entered, finalized }, { entered: 0, finalized: 1 });
    assertEquals(failures, [method === "throw" ? error : undefined]);
  }
});

Deno.test("開始直後の return は実行中の next を追い越さず本体の finally を通す", async () => {
  const ready = Promise.withResolvers<void>();
  let finalized = 0;
  let unstarted = 0;
  const iterator = closeableGenerator(
    (async function* () {
      try {
        await ready.promise;
        yield 1;
      } finally {
        finalized += 1;
      }
    })(),
    () => {
      unstarted += 1;
    },
  );
  const next = iterator.next();
  const returned = iterator.return();
  ready.resolve();
  assertEquals(await next, { done: false, value: 1 });
  assertEquals(await returned, { done: true, value: undefined });
  assertEquals({ finalized, unstarted }, { finalized: 1, unstarted: 0 });
});

Deno.test("未開始 return の終了値が拒否されても所有者を解放し、次の終了は成功する", async () => {
  const value = Promise.withResolvers<void>();
  const cleanup = Promise.withResolvers<void>();
  const error = new Error("return value failed");
  const failures: unknown[] = [];
  const iterator = closeableGenerator(
    (async function* () {
      yield 1;
    })(),
    async (failure) => {
      failures.push(failure?.error);
      await cleanup.promise;
    },
  );
  const returned = iterator.return(value.promise);
  const rejected = assertRejects(() => returned);
  const next = iterator.next();
  value.reject(error);
  cleanup.resolve();
  assertStrictEquals(await rejected, error);
  assertEquals(await next, { done: true, value: undefined });
  assertEquals(await iterator.return(), { done: true, value: undefined });
  assertEquals(failures, [error]);
});
