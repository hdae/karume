/**
 * Helpers for the run record a model page shows and saves: counting non-finite values, turning
 * frames into one RGBA buffer and hashing it, and summarizing Session diagnostics. Pure functions
 * (WebCrypto is the only platform API), so they run the same in the browser and in `vp test`.
 */

import type { SessionDiagnostics } from "@karume/runtime";

/** The number of NaN / ±Infinity elements. */
export const countNonFinite = (data: Float32Array): number => {
  let count = 0;
  for (const value of data) if (!Number.isFinite(value)) count += 1;
  return count;
};

/** Frames as one contiguous RGBA buffer (frame-major), with a view per frame. */
export type RgbaStack = {
  readonly width: number;
  readonly height: number;
  readonly bytes: Uint8ClampedArray<ArrayBuffer>;
  readonly frame: (index: number) => Uint8ClampedArray<ArrayBuffer>;
  readonly frames: number;
};

/**
 * Converts every frame with `toRgba` (the model's own pixel rule, e.g. `wanFrameToRgba`) into one
 * buffer. The frames are views into that buffer, so the sha256 over all frames needs no second copy.
 */
export const stackRgbaFrames = (
  frames: number,
  width: number,
  height: number,
  toRgba: (frame: number) => Uint8ClampedArray,
): RgbaStack => {
  const frameBytes = width * height * 4;
  const bytes = new Uint8ClampedArray(frames * frameBytes);
  for (let index = 0; index < frames; index += 1) {
    const rgba = toRgba(index);
    if (rgba.length !== frameBytes) {
      throw new Error(`Frame ${index} has ${rgba.length} RGBA bytes, expected ${frameBytes}`);
    }
    bytes.set(rgba, index * frameBytes);
  }
  return {
    width,
    height,
    bytes,
    frames,
    frame: (index) => bytes.subarray(index * frameBytes, (index + 1) * frameBytes),
  };
};

/** Lower-case hex sha256 of `bytes` (WebCrypto). */
export const sha256Hex = async (
  bytes: Uint8Array<ArrayBuffer> | Uint8ClampedArray<ArrayBuffer>,
) => {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
};

/** What the record keeps of one component's Session diagnostics. */
export type DiagnosticsSummary = {
  /** How many runs reported diagnostics (one Session run = one report). */
  readonly runs: number;
  /** Weight arena bytes actually allocated, in GiB (from the latest report). */
  readonly weightsAllocatedGiB: number;
  /** Submit and dispatch counts as the latest report gives them. */
  readonly submitCount: number;
  readonly dispatchCount: number;
  readonly geometryProfile: string;
};

const GIB = 2 ** 30;

/** Folds one more report of a component into its summary (`previous` = earlier reports). */
export const summarizeDiagnostics = (
  previous: DiagnosticsSummary | undefined,
  diagnostics: {
    readonly weights: Pick<SessionDiagnostics["weights"], "allocatedBytes">;
    readonly submit: Pick<SessionDiagnostics["submit"], "submitCount" | "dispatchCount">;
    readonly geometryProfile: string;
  },
): DiagnosticsSummary => ({
  runs: (previous?.runs ?? 0) + 1,
  weightsAllocatedGiB: Number((diagnostics.weights.allocatedBytes / GIB).toFixed(3)),
  submitCount: diagnostics.submit.submitCount,
  dispatchCount: diagnostics.submit.dispatchCount,
  geometryProfile: diagnostics.geometryProfile,
});

/** The adapter fields a record names (`GPUAdapterInfo` — `description` is the driver on Dawn). */
export type AdapterSummary = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

export const summarizeAdapter = (info: GPUAdapterInfo): AdapterSummary => ({
  vendor: info.vendor,
  architecture: info.architecture,
  device: info.device,
  description: info.description,
});

/** Seconds with millisecond precision (records stay readable and diff-friendly). */
export const seconds = (milliseconds: number): number => Number((milliseconds / 1000).toFixed(3));
