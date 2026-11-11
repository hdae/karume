/**
 * A trace of what a disposal does on a real device: every buffer destroy, every queue submit and
 * every completion wait that resolved, in order. The WebGPU objects of the device are wrappers that
 * may be created anew on each access, so the methods are replaced on their prototypes and restored
 * by `restore` (call it in a `finally`).
 */

import type { GpuContext } from "../../src/gpu/device.ts";
import { BUFFER_USAGE } from "../../src/gpu/webgpu-constants.ts";

export type QueueTrace = {
  /** `"destroy"`, `"submit"` and `"done"` (a completion wait resolved), in order. */
  readonly log: string[];
  restore(): void;
};

type Method = (this: unknown, ...args: unknown[]) => unknown;

const isMethod = (value: unknown): value is Method => typeof value === "function";

/** Replaces the method `name` of `owner` by one that calls `before`, the original, then `after`. */
const intercept = (
  owner: object,
  name: string,
  before: () => void,
  after?: (result: unknown) => void,
): (() => void) => {
  const descriptor = Object.getOwnPropertyDescriptor(owner, name);
  const original: unknown = descriptor?.value;
  if (descriptor === undefined || !isMethod(original)) {
    throw new Error(`queue trace: '${name}' is not a method of the prototype`);
  }
  Object.defineProperty(owner, name, {
    ...descriptor,
    value: function (this: unknown, ...args: unknown[]): unknown {
      before();
      const result = original.apply(this, args);
      after?.(result);
      return result;
    },
  });
  return (): void => {
    Object.defineProperty(owner, name, descriptor);
  };
};

const prototypeOf = (value: object): object => {
  const prototype: unknown = Object.getPrototypeOf(value);
  if (typeof prototype !== "object" || prototype === null) {
    throw new Error("queue trace: the WebGPU object has no prototype");
  }
  return prototype;
};

export const traceQueue = (gpu: GpuContext): QueueTrace => {
  const log: string[] = [];
  const probe = gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.COPY_DST });
  const buffers = prototypeOf(probe);
  probe.destroy();
  const queues = prototypeOf(gpu.device.queue);
  const restores = [
    intercept(buffers, "destroy", () => log.push("destroy")),
    intercept(queues, "submit", () => log.push("submit")),
    intercept(
      queues,
      "onSubmittedWorkDone",
      () => {},
      (work) => {
        if (!(work instanceof Promise)) throw new Error("queue trace: no completion promise");
        // Registered before the caller's own continuation: "done" is logged before the caller goes on
        void work.then(() => log.push("done"));
      },
    ),
  ];
  return {
    log,
    restore: (): void => {
      for (const restore of restores) restore();
    },
  };
};
