import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { closeableGenerator } from "../src/concurrency/closeable-generator.ts";

test("closing before start does not run the body and notifies once", async () => {
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

test("return right after start does not overtake the running next and goes through the body's finally", async () => {
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

test("even when the completion value of a return before start is rejected, the owner is released and the next close succeeds", async () => {
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
