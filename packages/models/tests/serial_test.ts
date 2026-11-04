// Behaviour tests of the serialisation chain (`src/concurrency/serial.ts`). Needs neither a GPU nor
// real assets.
//
// It is the foundation on which three pipeline families keep "one graph at a time" on the public
// API side, so it is pinned here rather than waiting for concurrent generate calls on a real GPU.
// To control the settlement order, the operations' bodies are **promises settled by hand**
// (`Promise.withResolvers`) — relying on timer waits would loosen the order.

import { test } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import { createOperationChain } from "../src/concurrency/serial.ts";

/**
 * Waits until the queued operations have progressed as far as they can (one macrotask tick that
 * drains all microtasks).
 */
const settleMicrotasks = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

test("createOperationChain: operations queued concurrently run one at a time (a later one does not start until the earlier one settles)", async () => {
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

test("createOperationChain: a failure reaches the caller as is, and the chain is released to the next operation", async () => {
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

test("createOperationChain: an operation queued later runs after all earlier ones have settled (the way dispose waits)", async () => {
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
  // An operation standing in for disposal (synchronous). It must not run while any earlier one is
  // still alive.
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
