import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { createByteAdmission } from "../src/concurrency.ts";

/**
 * A test rig for the byte budget. It runs in the same shape as the dispatch loop of `fetchAssets`
 * (one loop admits, then dispatches; each job releases when it finishes) and records the peak of
 * the in-flight total.
 */
const dispatch = async (
  maxBytes: number,
  sizes: readonly number[],
): Promise<{ peak: number; order: number[] }> => {
  const admission = createByteAdmission(maxBytes);
  const order: number[] = [];
  const running: Promise<void>[] = [];
  let inFlight = 0;
  let peak = 0;
  for (const [index, size] of sizes.entries()) {
    await admission.admit(size);
    order.push(index);
    inFlight += size;
    peak = Math.max(peak, inFlight);
    running.push(
      (async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        inFlight -= size;
        admission.release(size);
      })(),
    );
  }
  await Promise.all(running);
  return { peak, order };
};

test("createByteAdmission: the in-flight total does not exceed the budget", async () => {
  const { peak, order } = await dispatch(100, [40, 40, 40, 40, 40]);
  assert(peak <= 100, `in-flight grew to ${peak} bytes`);
  assert(
    peak > 40,
    "only one at a time passes (a second request that fits the budget is made to wait)",
  );
  assertEquals(order.length, 5, "not every request was dispatched");
});

test("createByteAdmission: with 0 in flight, a single request over the budget still passes (no deadlock)", async () => {
  // Waiting when nobody is left to release a slot deadlocks forever. A single 250 against a budget
  // of 100 must pass.
  const { peak, order } = await dispatch(100, [250]);
  assertEquals(order, [0]);
  assertEquals(peak, 250);
});

test("createByteAdmission: an oversized request does not overlap even one other request", async () => {
  // The "pass when 0 in flight" exception must not drag others along and break the budget twice.
  const { peak } = await dispatch(100, [60, 250, 60]);
  assertEquals(peak, 250, `the oversized request overlapped another and reached ${peak} bytes`);
});

test("createByteAdmission: two releases in a row do not lose a waiter", async () => {
  const admission = createByteAdmission(100);
  await admission.admit(50);
  await admission.admit(50);
  let admitted = false;
  const pending = admission.admit(50).then(() => {
    admitted = true;
  });
  await Promise.resolve();
  assertEquals(admitted, false, "admitted although the budget is full");
  // There is only one wake-up slot. If two releases in a row drop one, the waiter never wakes.
  admission.release(50);
  admission.release(50);
  await pending;
  assertEquals(admitted, true, "a release was lost and the waiter stalled (lost wakeup)");
});
