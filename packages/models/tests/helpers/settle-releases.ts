/**
 * Waits for the release of destroyed buffers (the clean-up of real-GPU e2e — an empty `submit` →
 * `onSubmittedWorkDone`, raced against device loss).
 *
 * WHY: on the B570 (Linux xe / wgpu) the release by `destroy()` is delayed until the next device
 * poll (docs/known-issues.md, section "Intel Arc B570"). Measured on 2026-10-02 (plain WebGPU):
 * destroying 7 GiB and allocating it again on the same device is an OOM, and passes after this
 * wait. Dropping the device without waiting makes 7 GiB on another device an OOM (the memory stays
 * unreleased and keeps eating the budget); dropping it after the wait passes. So e2e that hold
 * GB-scale allocations call this between runs, between Sessions, and before dropping the device
 * (to leave the budget for the tests that follow).
 */

import type { GpuContext } from "@karume/runtime";

export const settleReleases = async (gpu: GpuContext): Promise<void> => {
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
