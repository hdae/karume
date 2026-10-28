/**
 * Helpers for the run record a model page shows and saves: counting non-finite values, turning
 * frames into one RGBA buffer and hashing it (and the sha256 of f32 values and RGB8 input), the
 * source a run read, summarizing Session diagnostics (build cost, the pipelines of the last run, a
 * run's share of the submit counts, GPU time per kernel), the device facts, the stage clock and the
 * per-step timeline. Pure functions (WebCrypto is the only platform API), so they run the same in
 * the browser and in `vp test`.
 */

import type { HubRepoRef } from "@karume/hub";
import type { SessionBuildStats, SessionDiagnostics } from "@karume/runtime";

import type { PipelineSource } from "@/lib/distributions";

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
 * Converts every frame with `toRgba` (the model's own pixel rule, e.g. `wan.frameToRgba`) into one
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

/**
 * The bytes of an f32 array as they sit in memory (a view, no copy). Every platform WebGPU runs on
 * is little-endian, so these are the little-endian bytes and their sha256 compares across machines.
 */
export const f32Bytes = (values: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(values.buffer, values.byteOffset, values.byteLength);

/** sha256 of the f32 values ({@link f32Bytes}). */
export const f32Sha256 = (values: Float32Array<ArrayBuffer>): Promise<string> =>
  sha256Hex(f32Bytes(values));

/**
 * sha256 of an RGB8 image's bytes as handed to a pipeline. Copied once: the families type the pixels
 * over any `ArrayBufferLike`, and WebCrypto takes no shared buffer.
 */
export const rgb8Sha256 = (image: { readonly data: Uint8Array }): Promise<string> =>
  sha256Hex(new Uint8Array(image.data));

/** The `source` field of a record: the cached route's repo reference, or `direct`. */
export type SourceRecord = { readonly kind: "direct" } | ({ readonly kind: "cached" } & HubRepoRef);

/** The cache generation is the page's rebuild counter, not a fact of the run, so it is left out. */
export const sourceRecord = (source: PipelineSource): SourceRecord =>
  source.kind === "cached" ? { kind: "cached", ...source.ref } : { kind: "direct" };

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

/** Submit and dispatch counts as a Session diagnostics report carries them (totals over its life). */
export type SubmitCounts = { readonly submitCount: number; readonly dispatchCount: number };

/** The counts of a Session that has not run yet (the runtime's counters start at 0). */
export const NO_SUBMITS: SubmitCounts = { submitCount: 0, dispatchCount: 0 };

/**
 * The submits and dispatches between two reports of one Session. The Session's scheduler counts them
 * over its whole life (it never resets them; only the GPU timing is per run), so a page that keeps
 * one Session for many runs records a run's share as the difference. Throws when a count went down:
 * the two reports came from different Sessions.
 */
export const submitCountsSince = (before: SubmitCounts, after: SubmitCounts): SubmitCounts => {
  const submitCount = after.submitCount - before.submitCount;
  const dispatchCount = after.dispatchCount - before.dispatchCount;
  if (submitCount < 0 || dispatchCount < 0) {
    throw new Error(
      `A Session's submit counts went down (submits ${before.submitCount} → ${after.submitCount}, dispatches ${before.dispatchCount} → ${after.dispatchCount}): the two reports are not from one Session`,
    );
  }
  return { submitCount, dispatchCount };
};

/**
 * Turns the reports of one Session (one report per Session run) into per-run reports: the returned
 * function replaces the counts with {@link submitCountsSince} the previous report it saw; the first
 * report counts from the Session's creation. Make one per Session.
 */
export const perRunSubmitCounts = (): (<Report extends { readonly submit: SubmitCounts }>(
  report: Report,
) => Report) => {
  let previous = NO_SUBMITS;
  return (report) => {
    const run = {
      ...report,
      submit: { ...report.submit, ...submitCountsSince(previous, report.submit) },
    };
    previous = {
      submitCount: report.submit.submitCount,
      dispatchCount: report.submit.dispatchCount,
    };
    return run;
  };
};

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

/**
 * One component's record: the diagnostics summary, its build cost, its last run's pipelines and,
 * on a timing device, its GPU time over every timed run.
 */
export type ComponentRecord = DiagnosticsSummary & {
  /** From the latest report (the Session that ran last). */
  readonly build: BuildCost;
  /** Absent only when the last run failed while planning (the runtime reports none then). */
  readonly lastRunPipelines?: readonly PipelineCount[];
  /** Absent when no run of the component was timed ({@link tallyTiming}). */
  readonly timing?: ComponentTiming;
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

/** GPU time of one pipeline key, summed over a component's timed runs. */
export type KeyTiming = {
  readonly key: string;
  readonly ms: number;
  readonly dispatchCount: number;
};

/** The same, folded per op ({@link pipelineOp}). */
export type OpTiming = { readonly op: string; readonly ms: number; readonly dispatchCount: number };

/** A component's GPU time (`lastRunTiming`) summed over its timed runs, in ms at 0.1 ms. */
export type ComponentTiming = {
  /** Runs that reported a breakdown. */
  readonly runs: number;
  readonly totalMs: number;
  readonly dispatchCount: number;
  /** Samples the runtime clamped to 0 (begin > end); non-zero = the breakdown itself is questionable. */
  readonly clampedNegativeSamples: number;
  /** Descending by time (ties by name), like the runtime's table. */
  readonly ops: readonly OpTiming[];
  readonly keys: readonly KeyTiming[];
};

type TimeSum = { readonly ns: number; readonly dispatchCount: number };

/**
 * The running sums behind {@link ComponentTiming}, in the runtime's nanoseconds: they are rounded to
 * 0.1 ms once, by {@link timingOf}, so the rounding does not pile up over hundreds of runs.
 */
export type TimingTally = {
  readonly runs: number;
  readonly clampedNegativeSamples: number;
  readonly keys: ReadonlyMap<string, TimeSum>;
};

/**
 * Folds one run's `lastRunTiming` into a component's tally (`previous` = earlier runs). A report
 * with no breakdown leaves the tally as it is: `undefined` comes from a device without timing, an
 * empty table from a run with nothing timed (a batch section on a timing device — Wan's VAE decoder).
 */
export const tallyTiming = (
  previous: TimingTally | undefined,
  timing: SessionDiagnostics["lastRunTiming"],
): TimingTally | undefined => {
  if (timing === undefined || timing.entries.length === 0) return previous;
  const keys = new Map(previous?.keys);
  for (const { key, ns, dispatchCount } of timing.entries) {
    const sum = keys.get(key);
    keys.set(key, {
      ns: (sum?.ns ?? 0) + ns,
      dispatchCount: (sum?.dispatchCount ?? 0) + dispatchCount,
    });
  }
  return {
    runs: (previous?.runs ?? 0) + 1,
    clampedNegativeSamples: (previous?.clampedNegativeSamples ?? 0) + timing.clampedNegativeSamples,
    keys,
  };
};

const NS_PER_MS = 1e6;

const descendingTime = (
  [leftName, left]: readonly [string, TimeSum],
  [rightName, right]: readonly [string, TimeSum],
): number => right.ns - left.ns || (leftName < rightName ? -1 : leftName > rightName ? 1 : 0);

/** The record of a tally: per key and per op, the total from the exact sum (not from rounded rows). */
export const timingOf = (tally: TimingTally): ComponentTiming => {
  const ops = new Map<string, TimeSum>();
  let totalNs = 0;
  let dispatchCount = 0;
  for (const [key, sum] of tally.keys) {
    const op = pipelineOp(key);
    const opSum = ops.get(op);
    ops.set(op, {
      ns: (opSum?.ns ?? 0) + sum.ns,
      dispatchCount: (opSum?.dispatchCount ?? 0) + sum.dispatchCount,
    });
    totalNs += sum.ns;
    dispatchCount += sum.dispatchCount;
  }
  return {
    runs: tally.runs,
    totalMs: milliseconds(totalNs / NS_PER_MS),
    dispatchCount,
    clampedNegativeSamples: tally.clampedNegativeSamples,
    ops: [...ops].toSorted(descendingTime).map(([op, sum]) => ({
      op,
      ms: milliseconds(sum.ns / NS_PER_MS),
      dispatchCount: sum.dispatchCount,
    })),
    keys: [...tally.keys].toSorted(descendingTime).map(([key, sum]) => ({
      key,
      ms: milliseconds(sum.ns / NS_PER_MS),
      dispatchCount: sum.dispatchCount,
    })),
  };
};

/** The `limit` most expensive ops with their share (0–1) of the component's timed total. */
export const topOpShares = (
  timing: ComponentTiming,
  limit: number,
): readonly { readonly op: string; readonly ms: number; readonly share: number }[] =>
  timing.ops.slice(0, limit).map(({ op, ms }) => ({
    op,
    ms,
    share: timing.totalMs === 0 ? 0 : ms / timing.totalMs,
  }));

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

/** `performance.now()` readings of each stage's start and end (from a pipeline's `stage` events). */
export type StageTimes<Stage extends string> = Partial<
  Record<Stage, { readonly start: number; readonly end?: number }>
>;

/**
 * Folds one `stage` event into the clock: a start sets the stage's start, an end its end (an end
 * without a start counts from the end itself).
 */
export const foldStageEvent = <Stage extends string>(
  times: StageTimes<Stage>,
  stage: Stage,
  at: "start" | "end",
  now: number,
): StageTimes<Stage> =>
  at === "start"
    ? { ...times, [stage]: { start: now } }
    : { ...times, [stage]: { start: times[stage]?.start ?? now, end: now } };

/** Seconds of every stage that ended, in `order`. */
export const stageSecondsOf = <Stage extends string>(
  times: StageTimes<Stage>,
  order: readonly Stage[],
): Partial<Record<Stage, number>> => {
  const result: Partial<Record<Stage, number>> = {};
  for (const stage of order) {
    const clock = times[stage];
    if (clock?.end !== undefined) result[stage] = seconds(clock.end - clock.start);
  }
  return result;
};

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

/** The adapter in one line (`vendor · architecture · device · description`, empty fields left out). */
export const adapterText = (info: GPUAdapterInfo): string =>
  Object.values(summarizeAdapter(info))
    .filter((value) => value !== "")
    .join(" · ");
