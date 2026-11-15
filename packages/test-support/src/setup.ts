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
 * - The kernel record (`./kernel-record.ts`) is switched on: which WGSL texts each test dispatches,
 *   for the sha256 rows and the verification cache. It only observes three WebGPU calls.
 * - The input record (`./input-record.ts`) is switched on: which files a test file reads, which
 *   directories it lists and which commands it starts. When the file ends, both records are left
 *   for the verification cache's reporter (`./verify-cache-reporter.ts`).
 *
 * `KARUME_DAWN_FLAGS` replaces the Dawn toggles (default: `disable-dawn-features=timestamp_quantization`);
 * `KARUME_NO_DAWN=1` skips Dawn entirely (no `navigator.gpu`).
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { afterAll, afterEach, beforeEach, expect } from "vite-plus/test";
import { create, globals } from "webgpu";
import { beginInputRecordFile, inputsOfFile, installInputRecord } from "./input-record.ts";
import {
  beginKernelRecordFile,
  beginKernelRecordTest,
  endKernelRecordTest,
  installKernelRecord,
  kernelsOfFile,
} from "./kernel-record.ts";
import { pendingPathOf } from "./verify-cache-reporter.ts";
import type { AdapterFacts, WorkerRecord } from "./verify-cache.ts";

type DawnHost = typeof globalThis & { __karumeDawnGpu?: GPU };

const host = globalThis as DawnHost;
// The namespaces and error classes exist in Deno and browsers even when no adapter does, and tests
// read them at module scope — install them unconditionally.
Object.assign(globalThis, globals);
// KARUME_NO_DAWN=1 runs the suite as if the machine had no WebGPU (CPU-only cases; the all-skip
// gates still apply) — useful next to another process that owns the GPU.
if (host.__karumeDawnGpu === undefined && process.env.KARUME_NO_DAWN !== "1") {
  // Dawn quantises timestamps to 65,536 ns by default, which turns short dispatches into 0 ns;
  // the suite measures per-dispatch GPU time, so the quantisation is disabled unless the caller
  // provides its own toggles.
  const flags = (process.env.KARUME_DAWN_FLAGS ?? "disable-dawn-features=timestamp_quantization")
    .split(/\s+/)
    .filter((flag) => flag !== "");
  try {
    host.__karumeDawnGpu = create(flags);
  } catch (error) {
    // A machine without a usable WebGPU backend (e.g. a CI runner without Vulkan) behaves like a
    // browser without WebGPU: `navigator.gpu` stays undefined, GPU cases are skipped, and the
    // all-skip gate (KARUME_ALLOW_NO_GPU) decides whether that is acceptable.
    console.warn(
      `[karume] Dawn could not be created; running without navigator.gpu: ${String(error)}`,
    );
  }
}

const current: GPU | undefined = navigator.gpu;
if (current === undefined && host.__karumeDawnGpu !== undefined) {
  Object.defineProperty(navigator, "gpu", {
    value: host.__karumeDawnGpu,
    configurable: true,
    enumerable: true,
  });
}

// A setup file is evaluated once per test file: the classes are wrapped once per worker, the buckets
// start empty for every file, and each test gets its own bucket. The classes are the globals installed
// above (the package types `globals` as a plain object).
installKernelRecord({ GPUDevice, GPUComputePassEncoder });
beginKernelRecordFile();
beforeEach(beginKernelRecordTest);
afterEach(endKernelRecordTest);

installInputRecord();
beginInputRecordFile();

const adapterFacts = async (): Promise<AdapterFacts | undefined> => {
  const adapter = await navigator.gpu?.requestAdapter();
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

// Registered first, so it runs after the test file's own `afterAll` hooks: what they read or
// dispatch is still part of the file's record.
afterAll(async () => {
  const testPath = expect.getState().testPath;
  if (testPath === undefined) return;
  // The inputs are taken before the write below, which is not one of the file's reads.
  const record: WorkerRecord = {
    testPath,
    kernels: kernelsOfFile(),
    inputs: inputsOfFile(),
    adapter: await adapterFacts(),
  };
  const path = pendingPathOf(testPath);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(record));
});
