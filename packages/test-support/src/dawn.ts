/**
 * Dawn (npm `webgpu`) as `navigator.gpu`, for a maintainer script run by plain Node outside the
 * test runner (the regression corpus's authoring tool). Tests get the same from the setup file
 * (`./setup.ts`), which also switches on the kernel and input records.
 *
 * `KARUME_DAWN_FLAGS` replaces the Dawn toggles, as in the setup file (default: timestamps not
 * quantised).
 *
 * The verification cache reads the adapter's facts here too: in a test worker (the setup file) for the
 * record, and in the main process before a `reuse` run for the decision.
 */

import { create, globals } from "webgpu";
import type { AdapterFacts } from "./verify-cache.ts";

/** The Dawn toggles: `KARUME_DAWN_FLAGS` split on whitespace, or timestamps not quantised. */
export const dawnFlags = (): string[] =>
  (process.env.KARUME_DAWN_FLAGS ?? "disable-dawn-features=timestamp_quantization")
    .split(/\s+/)
    .filter((flag) => flag !== "");

/** Installs Dawn's WebGPU globals and `navigator.gpu` (a no-op when `navigator.gpu` exists). */
export const installDawn = (): void => {
  const current: GPU | undefined = navigator.gpu;
  if (current !== undefined) return;
  Object.assign(globalThis, globals);
  Object.defineProperty(navigator, "gpu", {
    value: create(dawnFlags()),
    configurable: true,
    enumerable: true,
  });
};

/** The facts of the adapter `gpu` gives, as the verification cache keys on them (`undefined` without one). */
export const adapterFactsOf = async (gpu: GPU | undefined): Promise<AdapterFacts | undefined> => {
  const adapter = await gpu?.requestAdapter();
  if (adapter === undefined || adapter === null) return undefined;
  const limits: Record<string, number> = {};
  // GPUSupportedLimits keeps its values on the prototype, so they are read by name.
  for (const name in adapter.limits) {
    const value: unknown = Reflect.get(adapter.limits, name);
    if (typeof value === "number") limits[name] = value;
  }
  const { vendor, architecture, device, description } = adapter.info;
  return {
    vendor,
    architecture,
    device,
    description,
    features: [...adapter.features].sort(),
    limits,
  };
};

/**
 * The adapter's facts as a test worker sees them, read outside one: Dawn with the same toggles, and
 * none under `KARUME_NO_DAWN=1` or when Dawn cannot be created (the setup file's two cases).
 */
export const workerAdapterFacts = async (): Promise<AdapterFacts | undefined> => {
  if (process.env.KARUME_NO_DAWN === "1") return undefined;
  let gpu: GPU;
  try {
    gpu = create(dawnFlags());
  } catch {
    return undefined;
  }
  return adapterFactsOf(gpu);
};
