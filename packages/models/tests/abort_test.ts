// Tests of the abort check at stage boundaries (`settleAbort` in `src/concurrency/abort.ts`) and of
// its form for a loop of runs (`createAbortLap`, at the end of the file). Needs neither a GPU nor
// assets.
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

import { afterEach, beforeEach, describe, it, test, vi } from "vite-plus/test";
import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { createAbortLap, settleAbort } from "../src/concurrency/abort.ts";

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

// ---- The abort checks of a loop of runs (`createAbortLap`) ----
//
// Fake timers decide what the event loop has delivered: `vi.runAllTimers()` is "the event loop ran
// its timers" (what awaiting a run on the GPU lets happen), and not calling it is a lap that awaited
// no task. A `settle()` that waits is told from one that does not by whether it has resolved once
// the microtasks have run.

/** Lets every microtask queued so far run (the hops of an async function's return included). */
const microtasks = async (): Promise<void> => {
  for (let hop = 0; hop < 8; hop += 1) await Promise.resolve();
};

/** Whether `promise` has resolved once the microtasks have run. */
const resolvedNow = async (promise: Promise<void>): Promise<boolean> => {
  let resolved = false;
  void promise.then(
    () => {
      resolved = true;
    },
    () => {},
  );
  await microtasks();
  return resolved;
};

describe("the abort check of a loop of runs", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  describe("at the loop's first boundary", () => {
    it("yields to a timer, as a single stage boundary does", async () => {
      const lap = createAbortLap(new AbortController().signal);
      const first = lap.settle();
      assertEquals(await resolvedNow(first), false);
      vi.runAllTimers();
      await first;
    });
  });

  describe("at a boundary whose lap let the event loop deliver the timer armed at the previous one", () => {
    it("checks at once, without a timer of its own", async () => {
      const lap = createAbortLap(new AbortController().signal);
      const first = lap.settle();
      vi.runAllTimers();
      await first;
      // The lap's run is awaited: the event loop delivers its timers meanwhile.
      vi.runAllTimers();
      assertEquals(await resolvedNow(lap.settle()), true);
    });

    it("throws the reason of an abort that a task delivered during the lap", async () => {
      const controller = new AbortController();
      const lap = createAbortLap(controller.signal);
      const first = lap.settle();
      vi.runAllTimers();
      await first;
      const reason = new Error("stopped while the run was awaited");
      setTimeout(() => controller.abort(reason), 0);
      vi.runAllTimers();
      const caught = await assertRejects(() => lap.settle(), Error);
      assertStrictEquals(caught, reason);
      // A loop that stops has no next boundary to arm a timer for.
      assertEquals(vi.getTimerCount(), 0);
    });

    it("arms one timer for the next boundary", async () => {
      const lap = createAbortLap(new AbortController().signal);
      const first = lap.settle();
      vi.runAllTimers();
      await first;
      vi.runAllTimers();
      await lap.settle();
      assertEquals(vi.getTimerCount(), 1);
    });
  });

  describe("at a boundary whose lap awaited no task", () => {
    it("yields to a timer armed now, so a loop of runs that settle in microtasks still lets the event loop turn", async () => {
      const lap = createAbortLap(new AbortController().signal);
      const first = lap.settle();
      vi.runAllTimers();
      await first;
      // No timer is delivered between the two boundaries.
      const second = lap.settle();
      assertEquals(await resolvedNow(second), false);
      // The timer of the previous boundary is dropped, not waited for: one timer is pending.
      assertEquals(vi.getTimerCount(), 1);
      vi.runAllTimers();
      await second;
    });

    it("observes an abort a timer armed during the lap delivers, before the loop goes on", async () => {
      // Real timers: what is pinned is the order of two timers with the microtasks that run between
      // them, which the fake clock does not model (it fires every timer of one instant in one call).
      vi.useRealTimers();
      const controller = new AbortController();
      const lap = createAbortLap(controller.signal);
      await lap.settle();
      const reason = new Error("stopped by a timer armed during the lap");
      // Armed after the previous boundary's timer and before this boundary's own. A boundary that
      // waited for the earlier timer instead of arming its own would go on before this one fires.
      setTimeout(() => controller.abort(reason), 0);
      const caught = await assertRejects(() => lap.settle(), Error);
      assertStrictEquals(caught, reason);
    });
  });

  describe("without a signal", () => {
    it("never yields and arms no timer", async () => {
      const lap = createAbortLap(undefined);
      assertEquals(await resolvedNow(lap.settle()), true);
      assertEquals(await resolvedNow(lap.settle()), true);
      assertEquals(vi.getTimerCount(), 0);
    });
  });

  describe("when the signal is already aborted", () => {
    it("throws the reason itself after the first boundary's yield", async () => {
      const reason = new Error("aborted before the loop");
      const lap = createAbortLap(AbortSignal.abort(reason));
      const first = lap.settle();
      void first.catch(() => {});
      vi.runAllTimers();
      const caught = await assertRejects(() => first, Error);
      assertStrictEquals(caught, reason);
    });
  });

  describe("when the loop is left", () => {
    it("drops the timer armed for the boundary that will not come", async () => {
      const lap = createAbortLap(new AbortController().signal);
      const first = lap.settle();
      vi.runAllTimers();
      await first;
      assertEquals(vi.getTimerCount(), 1);
      lap.clear();
      assertEquals(vi.getTimerCount(), 0);
    });
  });
});
