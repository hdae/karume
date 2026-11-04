// Checks of the release wait (`settleReleasedMemory` in `src/session/settle-released-memory.ts`) —
// the one function shared by families whose next stage makes a large allocation right after the
// previous stage is torn down (its users are anima's offload and the encoder stage of Wan2.2 I2V).
// Pinned with a fake GpuContext (needs neither a GPU nor assets).
//
// WHY here: checks of a shared layer should run whichever of its users' lanes is run. The file name
// has no family name, so it goes into `test:core` (in a file named after a family it would not run
// in lanes other than core and that family — the lane convention of the ADR 0005 addendum).

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { settleReleasedMemory } from "../src/session/settle-released-memory.ts";
import { losableGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

describe("settleReleasedMemory (waiting for the release after an offload)", () => {
  it("issues an empty submit, then waits for onSubmittedWorkDone, and leaves no loss subscription behind", async () => {
    // The empty submit is "the call that surfaces the loss of an invalidated device" (the doc of
    // settle-released-memory.ts).
    const calls: string[] = [];
    const { gpu } = losableGpuContext(undefined, {
      submit: (commandBuffers) => void calls.push(`submit:${commandBuffers.length}`),
      onSubmittedWorkDone: () => {
        calls.push("onSubmittedWorkDone");
        return Promise.resolve();
      },
    });
    const baseline = gpu.pendingLostListeners;
    await settleReleasedMemory(gpu);
    assertEquals(calls, ["submit:0", "onSubmittedWorkDone"]);
    assertEquals(gpu.pendingLostListeners, baseline, "loss subscriptions were left behind");
  });

  it("returns without waiting when the empty submit surfaces the loss, even if onSubmittedWorkDone never resolves", async () => {
    // Deno resolves the loss of an invalidated device only at the next validity check (here,
    // submit).
    const { gpu, lose } = losableGpuContext(undefined, {
      submit: () => lose(),
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    const baseline = gpu.pendingLostListeners;
    await settleReleasedMemory(gpu);
    assertEquals(gpu.pendingLostListeners, baseline);
  });

  it("returns without waiting once the device is lost, even if onSubmittedWorkDone never resolves", async () => {
    // Guards against implementations whose onSubmittedWorkDone never resolves after a loss (MUST).
    // Without racing the two, this would hang here.
    const { gpu, lose } = losableGpuContext(undefined, {
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    const baseline = gpu.pendingLostListeners;
    const settled = settleReleasedMemory(gpu);
    lose();
    await settled;
    assertEquals(gpu.pendingLostListeners, baseline);
  });

  it("returns without waiting for a device that is already lost", async () => {
    const { gpu, lose } = losableGpuContext(undefined, {
      onSubmittedWorkDone: () => new Promise<void>(() => {}),
    });
    lose();
    // The loss notification arrives through the resolution of device.lost (a microtask).
    await Promise.resolve();
    assert(gpu.lost !== undefined, "premise: the loss has been observed");
    await settleReleasedMemory(gpu);
  });
});
