/**
 * A tool that waits for the release of destroyed buffers (**pipeline-independent shared processing**
 * — used by families where the next stage makes a large allocation right after the previous stage
 * is torn down).
 *
 * MUST: not exported from the barrel (for the same reason as the co-located `options.ts` /
 * `dispose-steps.ts` — an internal mechanism, not a surface the user touches).
 */

import type { GpuContext } from "@karume/runtime";

/**
 * Issues one empty `queue.submit([])` and then waits for `onSubmittedWorkDone` (raced against device
 * loss).
 *
 * WHY: on Intel / wgpu (B570) the release by `destroy()` is delayed until the next device poll, and
 * re-allocating right after a teardown can hit OOM (docs/known-issues.md, the "Intel Arc B570"
 * section). In a plain WebGPU probe this wait alone let the re-allocation succeed
 * (docs/research/2026-09-26-anima-residency-bench.md). No fixed sleep is added.
 * The empty submit is there to flush wgpu's pending destroys, and to surface here the loss of a
 * device invalidated by a fatal OOM (Deno resolves `device.lost` only on a call that checks
 * validity).
 *
 * MUST: race against device loss — some implementations may never resolve `onSubmittedWorkDone`
 * after the loss (the doc of the runtime's `raceDeviceLost`). On loss it returns without waiting, and
 * the next stage fails loudly with the loss exception.
 */
export const settleReleasedMemory = async (gpu: GpuContext): Promise<void> => {
  let unsubscribe: () => void = () => {};
  const lost = new Promise<void>((resolve) => {
    unsubscribe = gpu.onLost(() => resolve());
  });
  try {
    gpu.device.queue.submit([]);
    await Promise.race([gpu.device.queue.onSubmittedWorkDone(), lost]);
  } finally {
    unsubscribe();
  }
};
