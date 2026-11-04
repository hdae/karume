/**
 * Decides whether real-GPU tests can run (the models version).
 *
 * MUST: an environment without an adapter is an **explicit SKIP** through `test({ ignore })` (do
 * not delete tests and go silently green — ADR 0005). The decision is needed when tests are
 * registered, so it runs once, at module evaluation. An environment where `requestAdapter()` throws
 * is a "broken environment", so the error propagates rather than being swallowed.
 *
 * NOTE: the runtime's helper of the same name (`packages/runtime/tests/helpers/gpu.ts`) is not
 * imported — when a package's tests depend on another package's test internals, that side's checks
 * (`gpu_gate_test`) and environment variables leak into this one. Only the conventions (one
 * decision, the warning text, explicit SKIP) are matched.
 */

import { acquireGpu, type AcquireGpuOptions, type GpuContext } from "@karume/runtime";
import { collectDestroyedDevices } from "./collect-destroyed-devices.ts";

const detectAdapter = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  return (await gpu.requestAdapter()) !== null;
};

export const GPU_AVAILABLE: boolean = await detectAdapter();

/**
 * The tests' way to get a GPU: prompts the collection of destroyed devices, then calls
 * `acquireGpu`.
 *
 * NOTE: a symptomatic workaround (see the note on {@link collectDestroyedDevices} — Deno's
 * `destroy()` does not return VRAM). Only the e2e of the Wan lanes (`test:models:wan` /
 * `test:models:wan-ti2v`) go through it today; other tests, and the devices pipelines acquire
 * internally (loads that do not pass `gpu`), do not.
 */
export const acquireTestGpu = async (options?: AcquireGpuOptions): Promise<GpuContext> => {
  await collectDestroyedDevices();
  return await acquireGpu(options);
};

if (!GPU_AVAILABLE) {
  console.warn(
    "[karume] no GPUAdapter, so the models real-GPU tests are SKIPped" +
      " (a release requires the real GPU to be green — ADR 0005)",
  );
}
