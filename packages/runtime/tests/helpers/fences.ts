// Mechanical check of the fence count (the number of `queue.onSubmittedWorkDone` calls).
//
// An optimization that "waits less" can silently regress while the values stay correct (falling
// back to a flush internally raises no exception and no warning), so a check has to count that
// the fences really went down. The test side wraps the queue's surface so the implementation
// needs no extra observation points.

import type { GpuContext } from "../../src/gpu/device.ts";

export type FenceCounter = {
  /** The number of `onSubmittedWorkDone` calls since wrapping. */
  readonly count: () => number;
  /**
   * Restores the original surface (MUST: always call it in finally — otherwise the count leaks
   * into other tests).
   */
  restore: () => void;
};

export const countFences = (gpu: GpuContext): FenceCounter => {
  const queue = gpu.device.queue;
  const original = queue.onSubmittedWorkDone.bind(queue);
  let fences = 0;
  Object.defineProperty(queue, "onSubmittedWorkDone", {
    configurable: true,
    writable: true,
    value: () => {
      fences += 1;
      return original();
    },
  });
  return {
    count: () => fences,
    restore: () => {
      Object.defineProperty(queue, "onSubmittedWorkDone", {
        configurable: true,
        writable: true,
        value: original,
      });
    },
  };
};
