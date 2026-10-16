/**
 * Vitest setup file: give Node the WebGPU surface the tests were written against.
 *
 * - `navigator.gpu` comes from Dawn (npm `webgpu`). Created once per worker process and reused
 *   across test files (Vitest re-evaluates setup files per file, but the native instance must not
 *   be re-created each time).
 * - The WebGPU namespaces and error classes (`GPUBufferUsage`, `GPUMapMode`, `GPUOutOfMemoryError`, ...)
 *   are globals in Deno and browsers; tests reference them directly, so install Dawn's copies.
 *   Package sources MUST NOT rely on these globals (see runtime/src/gpu/webgpu-constants.ts).
 *
 * `KARUME_DAWN_FLAGS` passes Dawn toggles, e.g. `disable-dawn-features=timestamp_quantization`.
 */

import { create, globals } from "webgpu";

type DawnHost = typeof globalThis & { __karumeDawnGpu?: GPU };

const host = globalThis as DawnHost;
if (host.__karumeDawnGpu === undefined) {
  const flags = (process.env.KARUME_DAWN_FLAGS ?? "").split(/\s+/).filter((flag) => flag !== "");
  host.__karumeDawnGpu = create(flags);
  Object.assign(globalThis, globals);
}

const current: GPU | undefined = navigator.gpu;
if (current === undefined) {
  Object.defineProperty(navigator, "gpu", {
    value: host.__karumeDawnGpu,
    configurable: true,
    enumerable: true,
  });
}
