/**
 * Helpers for the run record a model page shows and saves: counting non-finite values, turning
 * frames into one RGBA buffer and hashing it, summarizing Session diagnostics (build cost, the
 * pipelines of the last run), the device facts and the per-step timeline. Pure functions (WebCrypto
 * is the only platform API), so they run the same in the browser and in `vp test`.
 */

import type { SessionBuildStats, SessionDiagnostics } from "@karume/runtime";

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

/** Milliseconds with 0.1 ms precision. */
const milliseconds = (value: number): number => Number(value.toFixed(1));

/**
 * What the record keeps of a Session's construction cost (`buildStats` — fixed when the Session
 * is built). The host-side fields (supply wait / decode / buffer creation / `writeBuffer` issue)
 * and the GPU wait (`uploadFenceMs`, which absorbs the real transfer) are kept apart, as the
 * runtime reports them. `supplyWaitMs` is where reading the weights (network, cache) lands.
 */
export type BuildCost = {
  readonly partCount: number;
  readonly supplyWaitMs: number;
  readonly decodeMs: number;
  readonly bufferCreateMs: number;
  readonly writeBufferIssueMs: number;
  readonly uploadFenceMs: number;
  readonly uploadedBytes: number;
};

export const buildCostOf = (stats: SessionBuildStats): BuildCost => ({
  partCount: stats.partCount,
  supplyWaitMs: milliseconds(stats.supplyWaitMs),
  decodeMs: milliseconds(stats.decodeMs),
  bufferCreateMs: milliseconds(stats.bufferCreateMs),
  writeBufferIssueMs: milliseconds(stats.writeBufferIssueMs),
  uploadFenceMs: milliseconds(stats.uploadFenceMs),
  uploadedBytes: stats.uploadedBytes,
});

/** The field-wise sum of several build costs (the run's total over its components). */
export const totalBuildCost = (costs: Iterable<BuildCost>): BuildCost => {
  const total = {
    partCount: 0,
    supplyWaitMs: 0,
    decodeMs: 0,
    bufferCreateMs: 0,
    writeBufferIssueMs: 0,
    uploadFenceMs: 0,
    uploadedBytes: 0,
  };
  for (const cost of costs) {
    total.partCount += cost.partCount;
    total.supplyWaitMs += cost.supplyWaitMs;
    total.decodeMs += cost.decodeMs;
    total.bufferCreateMs += cost.bufferCreateMs;
    total.writeBufferIssueMs += cost.writeBufferIssueMs;
    total.uploadFenceMs += cost.uploadFenceMs;
    total.uploadedBytes += cost.uploadedBytes;
  }
  return {
    ...total,
    supplyWaitMs: milliseconds(total.supplyWaitMs),
    decodeMs: milliseconds(total.decodeMs),
    bufferCreateMs: milliseconds(total.bufferCreateMs),
    writeBufferIssueMs: milliseconds(total.writeBufferIssueMs),
    uploadFenceMs: milliseconds(total.uploadFenceMs),
  };
};

/** Planned dispatches of one pipeline key in a run (`lastRunPipelines`). */
export type PipelineCount = { readonly key: string; readonly dispatchCount: number };

/** One component's record: the diagnostics summary, its build cost and its last run's pipelines. */
export type ComponentRecord = DiagnosticsSummary & {
  /** From the latest report (the Session that ran last). */
  readonly build: BuildCost;
  /** Absent only when the last run failed while planning (the runtime reports none then). */
  readonly lastRunPipelines?: readonly PipelineCount[];
};

/** Folds one more report of a component into its record (`previous` = earlier reports). */
export const foldComponentDiagnostics = (
  previous: ComponentRecord | undefined,
  report: Parameters<typeof summarizeDiagnostics>[1] & {
    readonly buildStats: SessionBuildStats;
    readonly lastRunPipelines: SessionDiagnostics["lastRunPipelines"];
  },
): ComponentRecord => ({
  ...summarizeDiagnostics(previous, report),
  build: buildCostOf(report.buildStats),
  ...(report.lastRunPipelines === undefined
    ? {}
    : {
        lastRunPipelines: report.lastRunPipelines.map(({ key, dispatchCount }) => ({
          key,
          dispatchCount,
        })),
      }),
});

/** The op of a pipeline key (`attention_online:v1:i8a8:…` → `attention_online`). */
export const pipelineOp = (key: string): string => {
  const colon = key.indexOf(":");
  return colon < 0 ? key : key.slice(0, colon);
};

/**
 * The attention kernels of a run: `attention_online` for the online route, `attention_qk` /
 * `attention_stats` / `attention_pv` for the materialized one (the spelling of keys is a diagnostic
 * label of the runtime, outside semver).
 */
export const attentionPipelines = (pipelines: readonly PipelineCount[]): readonly PipelineCount[] =>
  pipelines.filter(({ key }) => pipelineOp(key).startsWith("attention"));

/** The device limits a record names (the ones the attention and GEMM routes depend on). */
export const DEVICE_LIMITS = [
  "maxComputeWorkgroupStorageSize",
  "maxStorageBufferBindingSize",
  "maxBufferSize",
] as const;

/** What the record keeps of the page's `GpuContext` (sets as sorted arrays). */
export type DeviceFacts = {
  readonly features: readonly string[];
  readonly wgslLanguageFeatures: readonly string[];
  readonly subgroupLanesLinear: boolean;
  readonly gpuTimingEnabled: boolean;
  readonly limits: Readonly<Record<(typeof DEVICE_LIMITS)[number], number>>;
};

export const summarizeDevice = (gpu: {
  readonly features: ReadonlySet<string>;
  readonly wgslLanguageFeatures: ReadonlySet<string>;
  readonly subgroupLanesLinear: boolean;
  readonly gpuTimingEnabled: boolean;
  readonly limits: Readonly<Record<(typeof DEVICE_LIMITS)[number], number>>;
}): DeviceFacts => ({
  features: [...gpu.features].sort(),
  wgslLanguageFeatures: [...gpu.wgslLanguageFeatures].sort(),
  subgroupLanesLinear: gpu.subgroupLanesLinear,
  gpuTimingEnabled: gpu.gpuTimingEnabled,
  limits: {
    maxComputeWorkgroupStorageSize: gpu.limits.maxComputeWorkgroupStorageSize,
    maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
    maxBufferSize: gpu.limits.maxBufferSize,
  },
});

/**
 * Seconds between consecutive marks (`performance.now()` milliseconds), the first measured from
 * `start`. With the stage start as `start` and the per-step events as marks, the first interval
 * also holds the Session creation and the weight upload.
 */
export const intervalSeconds = (start: number, marks: readonly number[]): number[] => {
  const intervals: number[] = [];
  let previous = start;
  for (const mark of marks) {
    intervals.push(seconds(mark - previous));
    previous = mark;
  }
  return intervals;
};

/** min / median / max of a non-empty list (the median of an even count is the mean of the middle two). */
export type Spread = {
  readonly count: number;
  readonly min: number;
  readonly median: number;
  readonly max: number;
};

/**
 * The first interval on its own (it carries the Session build) and the spread of the rest
 * (`undefined` for no intervals; `rest` is absent for a single one).
 */
export type IntervalSummary = { readonly first: number; readonly rest?: Spread };

export const summarizeIntervals = (values: readonly number[]): IntervalSummary | undefined => {
  if (values.length === 0) return undefined;
  const [first, ...rest] = values;
  if (rest.length === 0) return { first };
  const sorted = rest.toSorted((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return {
    first,
    rest: {
      count: sorted.length,
      min: sorted[0],
      median: Number(median.toFixed(4)),
      max: sorted[sorted.length - 1],
    },
  };
};
