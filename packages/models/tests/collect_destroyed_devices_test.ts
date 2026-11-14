// Behaviour of the mitigation in the tests' GPU acquisition (prompting collection of destroyed
// devices — a symptomatic workaround). Uses no GPU.

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { collectDestroyedDevices, GC_SETTLE_MS } from "./helpers/collect-destroyed-devices.ts";

// Slack for the rounding of timer firing times (setTimeout never fires earlier than requested, but
// the measuring clock's granularity is allowed).
const CLOCK_SLACK_MS = 2;

/**
 * Whether the Promise of `start()` resolves without waiting for a single timer (macrotask). Judging
 * the wait by an elapsed-time threshold wobbles on a loaded machine, so it races against a 0 ms
 * timer instead (a resolution that waits for no timer arrives first through a microtask). The watch
 * timer is queued before `start()`, so if the other side waits for even one 0 ms timer the watch
 * arrives first.
 */
const resolvesWithoutTimer = async (start: () => Promise<void>): Promise<boolean> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timerFired = new Promise<"timer">((resolve) => {
    timer = setTimeout(() => resolve("timer"), 0);
  });
  try {
    return (await Promise.race([start().then(() => "helper" as const), timerFired])) === "helper";
  } finally {
    clearTimeout(timer);
  }
};

describe("collectDestroyedDevices", () => {
  describe("when gc exists", () => {
    it("calls gc exactly twice and resolves after the wait following the second call", async () => {
      const calls: number[] = [];
      const start = performance.now();
      await collectDestroyedDevices({ gc: () => calls.push(performance.now()) });
      const resolved = performance.now();
      assertEquals(calls.length, 2);
      const [first, second] = calls;
      assert(first - start < GC_SETTLE_MS, "the first gc was not called before the wait");
      assert(
        second - first >= GC_SETTLE_MS - CLOCK_SLACK_MS,
        `the wait after the first call is short (${second - first} ms)`,
      );
      assert(
        resolved - second >= GC_SETTLE_MS - CLOCK_SLACK_MS,
        `resolved before the wait after the second call ended (${resolved - second} ms)`,
      );
    });
  });

  describe("when gc is absent", () => {
    it("calls nothing and resolves without waiting", async () => {
      assert(
        await resolvesWithoutTimer(() => collectDestroyedDevices({})),
        "it waited although gc is absent",
      );
    });

    it("when gc is not a function it is not called and resolves without waiting", async () => {
      // Even if something else carries the name `gc` in a run without `--expose-gc`, it neither
      // fails trying to call it nor waits.
      assert(
        await resolvesWithoutTimer(() => collectDestroyedDevices({ gc: "not a function" })),
        "it waited although gc is not a function",
      );
    });
  });

  describe("default (globalThis)", () => {
    it("in a run with --expose-gc it resolves after waiting twice, and in a run without it resolves without waiting", async () => {
      // Branches on the run's flag: a `vp test` run (the Vitest config passes the flag) sees the
      // former, a plain run without the flag the latter.
      const host: object = globalThis;
      const exposedGc = "gc" in host ? host.gc : undefined;
      if (typeof exposedGc !== "function") {
        assert(
          await resolvesWithoutTimer(() => collectDestroyedDevices()),
          "it waited although gc is absent",
        );
        return;
      }
      const start = performance.now();
      await collectDestroyedDevices();
      const elapsed = performance.now() - start;
      assert(
        elapsed >= 2 * (GC_SETTLE_MS - CLOCK_SLACK_MS),
        `the default globalThis.gc is not used (resolved in ${elapsed} ms)`,
      );
    });
  });
});
