/**
 * Dawn (npm `webgpu`) as `navigator.gpu`, for a maintainer script run by plain Node outside the
 * test runner (the regression corpus's authoring tool). Tests get the same from the setup file
 * (`./setup.ts`), which also switches on the kernel and input records.
 *
 * `KARUME_DAWN_FLAGS` replaces the Dawn toggles, as in the setup file (default: timestamps not
 * quantised).
 */

import { create, globals } from "webgpu";

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
