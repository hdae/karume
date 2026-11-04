// Tests of the abort check at stage boundaries (`settleAbort` in `src/concurrency/abort.ts`). Needs
// neither a GPU nor assets.
//
// It pins four points:
//
// ① Without a `signal` it **never yields** (a caller that does not subscribe pays no task round
//   trip).
// ② When already aborted it throws `signal.reason` **itself** (no re-wrapping).
// ③ It **observes an abort that arrives while it is yielding** (the main point). `abort()` is
//   delivered as a task, so rewriting the yield as a microtask (`await Promise.resolve()`) would
//   "stay inside the current task", and the regression would be silent: the abort would **not take
//   effect**. That the check is not vacuous is shown by a fake implementation of the same shape
//   that yields to a microtask and misses that abort (positive control).
// ④ Without an abort it resolves (the check has not collapsed into always throwing).
//
// NOTE: Whether the yield goes through `setTimeout` or `MessageChannel` cannot be told apart by
// observing ③ and ④ (both are tasks). The implementation is fixed on `setTimeout(0)` to keep the
// order relative to aborts delivered by timers, and accepts the timer throttling of hidden tabs —
// the source of truth for the reason is the NOTE in `src/concurrency/abort.ts`. That is a matter of
// browser measurement, so it is not pinned here. What this file pins is "it yields to a macrotask".

import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { settleAbort } from "../src/concurrency/abort.ts";

/**
 * A fake implementation that yields to a microtask (③'s positive control — what a regression looks
 * like).
 */
const settleAbortViaMicrotask = async (signal: AbortSignal | undefined): Promise<void> => {
  if (signal === undefined) return;
  await Promise.resolve();
  signal.throwIfAborted();
};

test("settleAbort: without a signal it never yields (no macrotask in between)", async () => {
  const order: string[] = [];
  // The control = one task scheduled first. If `settleAbort` yields, that task settles first (a
  // settlement that needs only microtasks always comes before a task already in the queue).
  const task = new Promise<void>((resolve) => setTimeout(resolve, 0)).then(() =>
    order.push("task"),
  );
  const settled = settleAbort(undefined).then(() => order.push("settleAbort"));
  await Promise.all([task, settled]);
  assertEquals(order, ["settleAbort", "task"]);
});

test("settleAbort: with a signal it yields to a macrotask (settles after microtasks)", async () => {
  const order: string[] = [];
  const controller = new AbortController();
  const settled = settleAbort(controller.signal).then(() => order.push("settleAbort"));
  // Even with several microtask hops, this one finishes first as long as the yield goes to a task.
  const microtasks = Promise.resolve()
    .then(() => Promise.resolve())
    .then(() => order.push("microtask"));
  await Promise.all([microtasks, settled]);
  assertEquals(order, ["microtask", "settleAbort"]);
});

test("settleAbort: when already aborted it throws the reason itself", async () => {
  const reason = new Error("the load was aborted");
  const caught = await assertRejects(
    () => settleAbort(AbortSignal.abort(reason)),
    Error,
    "the load was aborted",
  );
  assertStrictEquals(caught, reason);
});

test("settleAbort: observes an abort() that arrives while it is yielding", async () => {
  const reason = new Error("abort through a microtask");
  const controller = new AbortController();
  const settled = settleAbort(controller.signal);
  // The abort arrives from a microtask chain = it has not happened yet when `settleAbort` is
  // called.
  await Promise.resolve()
    .then(() => Promise.resolve())
    .then(() => controller.abort(reason));
  const caught = await assertRejects(() => settled, Error, "abort through a microtask");
  assertStrictEquals(caught, reason);

  // Positive control — the fake that yields to a microtask **misses** the same abort (= this case
  // is not vacuous).
  const other = new AbortController();
  const missed = settleAbortViaMicrotask(other.signal);
  await Promise.resolve()
    .then(() => Promise.resolve())
    .then(() => other.abort(reason));
  await missed;
});

test("settleAbort: resolves when not aborted", async () => {
  const controller = new AbortController();
  await settleAbort(controller.signal);
  assertEquals(controller.signal.aborted, false);
});
