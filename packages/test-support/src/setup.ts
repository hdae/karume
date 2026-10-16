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
 * `KARUME_DAWN_FLAGS` passes Dawn toggles, e.g. `disable-dawn-features=timestamp_quantization`;
 * `KARUME_NO_DAWN=1` skips Dawn entirely (no `navigator.gpu`).
 */

import { create, globals } from "webgpu";

type DawnHost = typeof globalThis & { __karumeDawnGpu?: GPU };

const host = globalThis as DawnHost;
// KARUME_NO_DAWN=1 runs the suite as if the machine had no WebGPU (CPU-only cases; the all-skip
// gates still apply) — useful next to another process that owns the GPU.
if (host.__karumeDawnGpu === undefined && process.env.KARUME_NO_DAWN !== "1") {
  const flags = (process.env.KARUME_DAWN_FLAGS ?? "").split(/\s+/).filter((flag) => flag !== "");
  try {
    host.__karumeDawnGpu = create(flags);
  } catch (error) {
    // A machine without a usable WebGPU backend (e.g. a CI runner without Vulkan) behaves like a
    // browser without WebGPU: `navigator.gpu` stays undefined, GPU cases are skipped, and the
    // all-skip gate (KARUME_ALLOW_NO_GPU) decides whether that is acceptable.
    console.warn(`[karume] Dawn could not be created; running without navigator.gpu: ${String(error)}`);
  }
  Object.assign(globalThis, globals);
}

const current: GPU | undefined = navigator.gpu;
if (current === undefined && host.__karumeDawnGpu !== undefined) {
  Object.defineProperty(navigator, "gpu", {
    value: host.__karumeDawnGpu,
    configurable: true,
    enumerable: true,
  });
}
