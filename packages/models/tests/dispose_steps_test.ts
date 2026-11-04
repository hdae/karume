// A check of the cleanup step sequencing (`disposeSteps` in `src/session/dispose-steps.ts`) — the single helper
// shared by the `dispose()` of families that hold resident Sessions. Needs neither a GPU nor assets.
//
// Three things are pinned:
//
// ① **Later steps always run even if an earlier step fails.** Each family's `openXState` promises in its catch
//    that "a GPU acquired internally is always released", but if `dispose()` were the two serial steps
//    `await session.dispose(); gpu.destroy();`, a reject from `Session.dispose()` (`RunArena` propagates a flush
//    failure) would never reach the later step. Worse, a rejected promise would sit in `#disposal`, so a second
//    call would not retry either.
// ② **A single failure is thrown as that exception itself** (does not break the shape the caller branches on).
// ③ **Two or more failures are carried together as an `AggregateError`** (no return to the shape where `finally`
//    lets a later step's exception overwrite an earlier one = the shape that loses which step broke first).
//
// NOTE: that the family's `dispose()` "destroys only an internally acquired GPU" and "returns the same completion
// on a second call" is a property of `#disposal ??= #chain(...)`, exercised by the real-GPU e2e (there is no way
// to stub the internally acquired GPU, so only the helper's behaviour is pinned here).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { disposeSteps } from "../src/session/dispose-steps.ts";

test("disposeSteps: when every step succeeds, each runs once in declaration order and it resolves", async () => {
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

test("disposeSteps: a later step still runs exactly once when an earlier one rejects, and the original failure is thrown as is", async () => {
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
  // "The original failure itself" = not rewrapped (the caller can branch on the cause).
  assertStrictEquals(caught, lost);
  assertEquals(destroyCount, 1, "the later step (the gpu.destroy equivalent) was not reached");
});

test("disposeSteps: a throwing later step does not overwrite the earlier failure (two failures are an AggregateError)", async () => {
  const lost = new Error("device lost");
  const destroyFailed = new Error("destroy failed");
  const caught = await assertRejects(
    () => disposeSteps([() => Promise.reject(lost), () => Promise.reject(destroyFailed)]),
    AggregateError,
  );
  assert(caught instanceof AggregateError, `not an AggregateError: ${caught}`);
  // Both are carried (the `try/finally` version loses the earlier one here). The order is the declaration order.
  assertEquals(caught.errors, [lost, destroyFailed]);
});

test("disposeSteps: a step that throws synchronously does not stop later steps", async () => {
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
