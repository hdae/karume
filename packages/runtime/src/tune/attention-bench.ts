/**
 * The attention route bench (`runAttentionRouteBench` of `@karume/runtime/tune`): times the two a8
 * routes of fused attention at given shapes with the production kernels, on the caller's device.
 *
 * - `s16` — the three dispatches with S stored in f16 (`attentionCompute: "a8"` ×
 *   `attentionScoreStorage: "f16"`): ①QK i8a8, ② row statistics, ③PV i8a8, each timed on its own
 *   and summed.
 * - `online` — the online int8 kernel (`attentionScoreStorage: "none"`, ADR 0132): one
 *   `attention_online` dispatch, in the Vᵀ key-axis form the route takes on this device (`keyWords`
 *   — the profile's choice, else N's own; ADR 0132 addendum 2026-11-12).
 * - `quantization` (optional) — the four quantisation dispatches (q and k per token, the permute to
 *   Vᵀ and its quantization in the form N gives — the s16 route's, and the online route's where the
 *   profile does not choose `tiled`). They run once per node, outside the route totals.
 * - `onlineTiled` (opt-in) — the tiled key axis (`AttentionOnlineKeyWords` `tiled`, perf-ledger
 *   K-106) against the form N gives, whatever the profile chooses: the online kernel reading Vᵀ in
 *   its own key order (B) timed against the kernel reading N's form (A) in the same rounds
 *   ({@link timeLaunchPair}: A B B A per round, a per-round ratio, so a thermal drift within a round
 *   cancels), with its producer (the tiled `quantize_rows`) timed beside N's in the quantization
 *   line, and an identity block: both producers on the same f32 Vᵀ, both kernels on their outputs, O
 *   compared word for word. Both arms are the production route's generator calls for their form,
 *   and A stays N's form on a table that chooses `tiled`, so a record is the input the profile
 *   derivation reads (`deriveGeometryProfile`) on any table.
 *
 * It exists because the Wan2.2 pipeline cannot run on a timing-enabled device, while the per-kernel
 * GPU time of the two routes is what tells why a route gains on one machine and not on another
 * (perf-ledger K-82). It is not a sweep case: the sweep's case set feeds the geometry profiles'
 * fingerprint, and this bench chooses nothing — it only measures what production would run.
 *
 * ## What is taken from production (MUST: the same calls, not a second spelling)
 *
 * Keys, WGSL, params, binding order and workgroup counts come from the generator calls
 * `buildAttention` makes (src/runtime/recipe-builders/attention.ts — the `file:line` sits on each
 * launch below), and the binding tables are checked against the WGSL's storage roles
 * (`assertBindingRoles` of harness.ts). The route decision (`attentionRoute`), the device facts
 * (`attentionDeviceFacts`), the combination guard (`fusedAttentionViolation`), the row blocks of
 * the s16 route (`planAttentionBlocks` under the device's binding limit, ADR 0124) and the windows
 * of the online route (`planAttentionOnlineWindows`) are the production functions themselves. The
 * i8a8 geometry is the one Session construction selects (`sessionGeometryProfile`, called here) and
 * the integer dot-product variant is decided the way Session construction does
 * (`resolveAttentionI8a8Dot`, src/runtime/session-build.ts:301-321): emulated when the WGSL
 * language feature is not listed, otherwise the verdict of the attention canary.
 *
 * NOTE: the canary is run directly (`decideAttentionI8a8Dot`), not through the per-device memo of
 * `GpuContext` — the memo has exactly one call site, Session construction (its doc). It costs one
 * extra submit per bench run.
 *
 * ## Measurement
 *
 * The protocol is the sweep's (harness.ts): `reps` calibrated by wall clock so that one pass is
 * about {@link TARGET_PASS_MS}, idle runs before measuring, the min of `rounds`, GPU timestamps when
 * the device has timestamp-query and the wall clock otherwise. Inputs are deterministic pseudo-random
 * values in the quantized formats (i8 in [-127, 127] and positive scales — the harness fills); the
 * routes do not read the quantization's outputs, so their inputs are the same with or without the
 * quantization line.
 *
 * One measured dispatch is one row block: by default the first block production would run for the
 * shape (`s16`: the first of `planAttentionBlocks`; `online`: the first window of
 * `planAttentionOnlineWindows`), so the two routes may measure different row counts. Times are also
 * given per 1,000 query rows, and the ratio compares those.
 *
 * MUST: never run alongside inference on the same GPU (the measurements would be meaningless and the
 * two would compete) — the runtime cannot see other work on the device, so this is the caller's
 * responsibility, as for `runGeometrySweep`.
 */

import type { GpuContext } from "../gpu/device.ts";
import {
  type AttentionI8a8Decision,
  decideAttentionI8a8Dot,
} from "../gpu/attention-dp4a-canary.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../codegen/strided.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../kernels/attention-i8a8.ts";
import {
  ATTENTION_ONLINE_ROWS,
  type AttentionOnlineReduce,
  attentionOnlineI8a8Eligible,
  attentionOnlineI8a8Key,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
  attentionOnlineI8a8WorkgroupStorageBytes,
  type AttentionOnlineKeyWords,
  attentionOnlineKeyWords,
} from "../kernels/attention-online-i8a8.ts";
import {
  ATTENTION_STATS_STRIDE,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsRegCache,
  attentionStatsWgsl,
} from "../kernels/attention.ts";
import type { GeometryProfile } from "../kernels/geometry-profile.ts";
import { type I8a8Geometry, i8a8TileM, i8a8TileN } from "../kernels/i8a8-geometry.ts";
import { dp4aAvailable } from "../kernels/linear-i8a8.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../kernels/quantize-rows.ts";
import { attentionDeviceFacts } from "../runtime/attention-device.ts";
import { type AttentionDeviceFacts, attentionRoute } from "../runtime/attention-route.ts";
import { rowBlockLimit } from "../runtime/fusion.ts";
import {
  attentionI8a8TempBytes,
  attentionScoreElementBytes,
  attentionValueQuantization,
  fusedAttentionViolation,
  planAttentionBlocks,
  planAttentionOnlineWindows,
} from "../runtime/recipe-builders/attention.ts";
import { sessionGeometryProfile } from "../runtime/geometry-selection.ts";
import type { I8a8Dot } from "../runtime/session-types.ts";
import {
  compareBufferWords,
  createResources,
  createSweepContext,
  destroySweepContext,
  type Launch,
  type PairedRound,
  type PairedTiming,
  prepareLaunch,
  type ResourceSpec,
  runLaunchOnce,
  SWEEP_MAX_REPS,
  type SweepContext,
  timeLaunch,
  timeLaunchPair,
  type Timing,
} from "./harness.ts";
import { ROUNDS, TARGET_PASS_MS, WARMUP_MIN_RUNS, WARMUP_NS } from "./measurement.ts";
import type { ReportAdapter, TimingUnit } from "./report.ts";
import { timestampUnitFor } from "./sweep.ts";

/**
 * The format of the bench record. `/2`: `settings.onlineTiled`, and per shape the optional
 * `onlineTiled`, `tiledVsOnline`, `tiledVsOnlineWithProducer` and `quantization.tiledVt`.
 */
export const ATTENTION_BENCH_FORMAT = "karume-attention-bench/2";

/**
 * One attention shape (batch 1 — `heads` is the folded `B·H` axis). `queries` is the node's full
 * query count M (it decides the production split); `rowsPerDispatch` overrides the rows one measured
 * dispatch covers (default: the first row block production would run for the route).
 */
export type AttentionBenchShape = {
  /** A name for the record (free text). */
  readonly id: string;
  readonly heads: number;
  readonly queries: number;
  readonly keys: number;
  readonly depth: number;
  /** The op's half scale (`√scale_factor`, applied to both q and k — ADR 0023 decision 3). */
  readonly scale: number;
  readonly rowsPerDispatch?: number;
};

/** The half scale the exporter writes for SDPA's default scale `1/√D` (`f32(D^-1/4)`). */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

const WAN22_HEADS = 24;
const WAN22_DEPTH = 128;
/** Latent tokens of Wan2.2 TI2V-5B at 1280×704: 44 × 80 / (2 × 2) patches per latent frame. */
const WAN22_TOKENS_PER_LATENT_FRAME = 880;

/**
 * Wan2.2 TI2V-5B's attention shapes (24 heads, D = 128): self-attention at 33 frames (9 latent
 * frames, N = 7,920) and 121 frames (31 latent frames, N = 27,280), and cross-attention over the 512
 * text tokens at 121 frames.
 */
export const WAN22_ATTENTION_SHAPES: readonly AttentionBenchShape[] = [
  {
    id: "wan22-self-33f",
    heads: WAN22_HEADS,
    queries: 9 * WAN22_TOKENS_PER_LATENT_FRAME,
    keys: 9 * WAN22_TOKENS_PER_LATENT_FRAME,
    depth: WAN22_DEPTH,
    scale: halfScale(WAN22_DEPTH),
  },
  {
    id: "wan22-self-121f",
    heads: WAN22_HEADS,
    queries: 31 * WAN22_TOKENS_PER_LATENT_FRAME,
    keys: 31 * WAN22_TOKENS_PER_LATENT_FRAME,
    depth: WAN22_DEPTH,
    scale: halfScale(WAN22_DEPTH),
  },
  {
    id: "wan22-cross-121f",
    heads: WAN22_HEADS,
    queries: 31 * WAN22_TOKENS_PER_LATENT_FRAME,
    keys: 512,
    depth: WAN22_DEPTH,
    scale: halfScale(WAN22_DEPTH),
  },
];

/** Options of {@link runAttentionRouteBench} (all optional). */
export type AttentionBenchOptions = {
  /** The shapes (default {@link WAN22_ATTENTION_SHAPES}). */
  readonly shapes?: readonly AttentionBenchShape[];
  /** Measurement rounds per dispatch (default 5 — the representative value is their min). */
  readonly rounds?: number;
  /** Also time the four shared quantization dispatches (default true). */
  readonly quantization?: boolean;
  /**
   * Also time the candidate `onlineTiled` (perf-ledger K-106) against `online` at each shape, its
   * producer against today's in the quantization line, and check the pair's identity (default false).
   */
  readonly onlineTiled?: boolean;
  /** Abort (stops before the next dispatch is measured — the record keeps the finished shapes). */
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AttentionBenchProgress) => void;
};

export type AttentionBenchProgress =
  | { readonly kind: "status"; readonly message: string }
  | { readonly kind: "shape"; readonly result: AttentionBenchShapeResult };

/** The time of one measured dispatch. */
export type AttentionBenchStage = {
  readonly stage: string;
  readonly key: string;
  readonly workgroups: readonly [number, number, number];
  readonly reps?: number;
  /** Per-round pass times in the record's unit (timestamp ns, or wall ns). */
  readonly rounds?: readonly number[];
  readonly wallRounds?: readonly number[];
  readonly clampedNegativeSamples?: number;
  /** ms per dispatch (GPU time for unit `ns`, otherwise the wall clock). */
  readonly msPerDispatch?: number;
  readonly wallMsPerDispatch?: number;
  readonly error?: string;
};

/** The facts of one route at one shape (`eligible: false` = production would not run it). */
export type AttentionBenchRoute =
  | { readonly eligible: false; readonly reason: string }
  | {
      readonly eligible: true;
      /** Query rows one measured dispatch covers. */
      readonly rows: number;
      /** The production split for the shape (row blocks of S / online windows). */
      readonly productionDispatches: number;
      readonly productionRows: number;
      readonly stages: readonly AttentionBenchStage[];
      /** The route's total per dispatch (absent when a stage failed). */
      readonly msPerDispatch?: number;
      readonly msPer1000Rows?: number;
      /** Integer TOPS: `4 · heads · rows · keys · depth` operations (QK and PV MACs) per dispatch. */
      readonly tops?: number;
      /** `msPer1000Rows × queries / 1000` — the node's route time, quantization excluded. */
      readonly estimatedNodeMs?: number;
      /** The online route's reduction form and the workgroup storage it declares. */
      readonly reduce?: AttentionOnlineReduce;
      /** The online route's Vᵀ key-axis form (the route's `keyWords` on this device). */
      readonly keyWords?: AttentionOnlineKeyWords;
      readonly workgroupStorageBytes?: number;
      /** The s16 route: whether the row-window variant ran (several production blocks). */
      readonly rowWindow?: boolean;
      readonly error?: string;
    };

/** One paired round as recorded: the harness's round and its ratio B ÷ A (absent when a pass was clamped). */
export type AttentionBenchPairedRound = PairedRound & { readonly ratio?: number };

/** Two dispatches timed in the same rounds ({@link timeLaunchPair}): A = today's, B = the candidate. */
export type AttentionBenchPair = {
  readonly stages: readonly [a: string, b: string];
  readonly keys: readonly [a: string, b: string];
  readonly workgroups: readonly [
    a: readonly [number, number, number],
    b: readonly [number, number, number],
  ];
  /** The repetitions of every pass of both arms. */
  readonly reps?: number;
  /** Per round, in the order A B B A (record unit; the wall clock alongside). */
  readonly rounds?: readonly AttentionBenchPairedRound[];
  /** ms per dispatch of each arm: the min over its passes (GPU time for unit `ns`, else the wall clock). */
  readonly aMsPerDispatch?: number;
  readonly bMsPerDispatch?: number;
  /** The median of the per-round ratios B ÷ A (below 1 = the candidate faster). */
  readonly ratio?: number;
  readonly error?: string;
};

/** Both producers on the same f32 Vᵀ, both kernels on their outputs: O and the Vᵀ scales word for word. */
export type AttentionBenchIdentity = {
  readonly identical: boolean;
  /** O words compared (the whole O of the node; rows past the measured window are 0 on both sides). */
  readonly words?: number;
  readonly differingWords?: number;
  readonly differingScaleWords?: number;
  /** The first differing O word (index, today's word, the candidate's word). */
  readonly first?: { readonly index: number; readonly a: number; readonly b: number };
  readonly error?: string;
};

/** The candidate `onlineTiled` at one shape (`eligible: false` = the online route does not run). */
export type AttentionBenchTiled =
  | { readonly eligible: false; readonly reason: string }
  | {
      readonly eligible: true;
      /** Query rows of the measured dispatch (the online route's). */
      readonly rows: number;
      /** `online` (A) and `onlineTiled` (B) in the same rounds. */
      readonly kernel: AttentionBenchPair;
      /** The node's route time of each arm from the paired minimums (ms per row × the node's rows). */
      readonly onlineNodeMs?: number;
      readonly tiledNodeMs?: number;
      readonly identity: AttentionBenchIdentity;
    };

export type AttentionBenchShapeResult = {
  readonly shape: AttentionBenchShape;
  readonly s16: AttentionBenchRoute;
  readonly online: AttentionBenchRoute;
  /** `online.msPer1000Rows / s16.msPer1000Rows` (below 1 = online faster). */
  readonly onlineVsS16?: number;
  /** The candidate (present when `onlineTiled` was requested). */
  readonly onlineTiled?: AttentionBenchTiled;
  /** The kernel alone: `onlineTiled.kernel.ratio` (time tiled ÷ online; below 1 = tiled faster). */
  readonly tiledVsOnline?: number;
  /**
   * Kernel and producer per node: (tiled node ms + tiled producer ms) ÷ (online node ms + today's
   * producer ms), from the paired minimums (needs the quantization line).
   */
  readonly tiledVsOnlineWithProducer?: number;
  /** The four shared quantization dispatches of one node (absent when not requested or no route ran). */
  readonly quantization?: {
    readonly stages: readonly AttentionBenchStage[];
    /** The four stages above (today's dispatches — the candidate's producer is not in it). */
    readonly msPerNode?: number;
    /** `quantize_vt` (A) and the tiled producer (B) in the same rounds (with `onlineTiled`). */
    readonly tiledVt?: AttentionBenchPair;
  };
};

export type AttentionBenchReport = {
  readonly format: typeof ATTENTION_BENCH_FORMAT;
  readonly startedAt: string;
  readonly date: string;
  readonly aborted: boolean;
  readonly userAgent: string;
  readonly adapter: ReportAdapter;
  readonly timing: { readonly unit: TimingUnit; readonly gpuTimingEnabled: boolean };
  readonly variants: AttentionBenchVariants;
  readonly settings: {
    readonly rounds: number;
    readonly quantization: boolean;
    readonly onlineTiled: boolean;
    readonly targetPassMs: number;
    readonly maxReps: number;
    readonly warmupNs: number;
    readonly warmupMinRuns: number;
  };
  readonly shapes: readonly AttentionBenchShapeResult[];
};

/** The device and variant facts the routes were chosen from. */
export type AttentionBenchVariants = {
  readonly geometryProfile: string;
  readonly attentionQkGeometry: I8a8Geometry;
  readonly attentionPvGeometry: I8a8Geometry;
  readonly dp4a: {
    /** The WGSL language feature `packed_4x8_integer_dot_product` is listed. */
    readonly listed: boolean;
    readonly dot: I8a8Dot;
    readonly canary?: Pick<
      AttentionI8a8Decision,
      "branch" | "exact" | "maxAbsError" | "maxBandRatio"
    >;
  };
  readonly device: AttentionDeviceFacts & {
    readonly subgroupLanesLinear: boolean;
    readonly features: {
      readonly subgroups: boolean;
      readonly subgroupSizeControl: boolean;
      readonly subgroupId: boolean;
    };
    readonly maxStorageBufferBindingSize: number;
    readonly maxComputeWorkgroupsPerDimension: number;
  };
};

const F32 = 4;

/**
 * Turns `(binding, resource)` pairs into the positional table of {@link Launch} (`bindings[i]` is
 * binding `i + 1`). Fails when the numbers are not exactly 1..n — the production dispatches name
 * their scale bindings by constant, so a constant that moved shows here instead of binding by
 * guesswork.
 */
export const positionalBindings = (
  where: string,
  entries: readonly (readonly [binding: number, resource: string])[],
): string[] => {
  const sorted = entries.toSorted(([left], [right]) => left - right);
  sorted.forEach(([binding], index) => {
    if (binding !== index + 1) {
      throw new Error(
        `${where}: bindings [${sorted.map(([at]) => at).join(", ")}] are not exactly 1..${sorted.length}`,
      );
    }
  });
  return sorted.map(([, resource]) => resource);
};

/** Integer operations of one attention dispatch: the QK and PV products, 2 per multiply-accumulate. */
export const attentionIntegerOps = (
  heads: number,
  rows: number,
  keys: number,
  depth: number,
): number => 4 * heads * rows * keys * depth;

/** The online route's measured rows: whole workgroups of 64 rows, clamped at the node's M. */
export const onlineDispatchRows = (rows: number, queries: number): number =>
  Math.min(queries, Math.ceil(rows / ATTENTION_ONLINE_ROWS) * ATTENTION_ONLINE_ROWS);

/** `numerator / denominator` when both are positive and finite (JSON has no Infinity / NaN). */
export const finiteRatio = (numerator?: number, denominator?: number): number | undefined =>
  numerator !== undefined &&
  denominator !== undefined &&
  numerator > 0 &&
  denominator > 0 &&
  Number.isFinite(numerator / denominator)
    ? numerator / denominator
    : undefined;

/** The median (the mean of the middle two for an even count; absent for none). */
export const medianOf = (values: readonly number[]): number | undefined => {
  if (values.length === 0) return undefined;
  const sorted = values.toSorted((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

/** A round's ratio B ÷ A over its two passes per arm (absent when a pass was clamped). */
export const pairedRoundRatio = (round: PairedRound): number | undefined =>
  round.clamped > 0 ? undefined : finiteRatio(round.b[0] + round.b[1], round.a[0] + round.a[1]);

/**
 * The record of a paired timing: each round with its ratio, the median ratio (the representative —
 * each round's two arms ran under the same clocks), and each arm's min over its passes per dispatch
 * (GPU time for unit `ns`, otherwise the wall clock; a clamped 0 is never the min).
 */
export const summarizePair = (
  unit: TimingUnit,
  timing: PairedTiming,
): Pick<AttentionBenchPair, "reps" | "rounds" | "aMsPerDispatch" | "bMsPerDispatch" | "ratio"> => {
  const rounds = timing.rounds.map((round): AttentionBenchPairedRound => {
    const ratio = pairedRoundRatio(round);
    return ratio === undefined ? round : { ...round, ratio };
  });
  const msOf = (passes: readonly number[]): number | undefined => {
    const kept = passes.filter((value) => value > 0);
    return kept.length === 0 ? undefined : Math.min(...kept) / timing.reps / 1e6;
  };
  const arm = (gpu: "a" | "b", wall: "wallA" | "wallB"): number | undefined =>
    msOf(timing.rounds.flatMap((round) => (unit === "ns" ? round[gpu] : round[wall])));
  const aMs = arm("a", "wallA");
  const bMs = arm("b", "wallB");
  const ratio = medianOf(
    rounds.flatMap((round) => (round.ratio === undefined ? [] : [round.ratio])),
  );
  return {
    reps: timing.reps,
    rounds,
    ...(aMs === undefined ? {} : { aMsPerDispatch: aMs }),
    ...(bMs === undefined ? {} : { bMsPerDispatch: bMs }),
    ...(ratio === undefined ? {} : { ratio }),
  };
};

/**
 * Kernel and producer per node, candidate ÷ today: (B node ms + B producer ms) ÷ (A node ms + A
 * producer ms), absent unless all four are known.
 */
export const withProducerRatio = (
  kernel: { readonly onlineNodeMs?: number; readonly tiledNodeMs?: number },
  producer: { readonly aMsPerDispatch?: number; readonly bMsPerDispatch?: number } | undefined,
): number | undefined =>
  kernel.onlineNodeMs === undefined ||
  kernel.tiledNodeMs === undefined ||
  producer?.aMsPerDispatch === undefined ||
  producer.bMsPerDispatch === undefined
    ? undefined
    : finiteRatio(
        kernel.tiledNodeMs + producer.bMsPerDispatch,
        kernel.onlineNodeMs + producer.aMsPerDispatch,
      );

const isPositiveInteger = (value: number): boolean => Number.isSafeInteger(value) && value >= 1;

/** Checks the options before the GPU is touched (a misspelled shape is never run as something else). */
export const checkAttentionBenchShape = (shape: AttentionBenchShape): void => {
  for (const field of ["heads", "queries", "keys", "depth"] as const) {
    if (!isPositiveInteger(shape[field])) {
      throw new Error(`${shape.id}: ${field} must be a positive integer (${shape[field]})`);
    }
  }
  if (!Number.isFinite(Math.fround(shape.scale)) || shape.scale <= 0) {
    throw new Error(`${shape.id}: scale must be a positive finite f32 (${shape.scale})`);
  }
  const rows = shape.rowsPerDispatch;
  if (rows !== undefined && (!isPositiveInteger(rows) || rows > shape.queries)) {
    throw new Error(
      `${shape.id}: rowsPerDispatch must be an integer in 1..${shape.queries} (${rows})`,
    );
  }
};

const describeCause = (cause: unknown): string =>
  cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);

/** GPU ns per dispatch for unit `ns`, otherwise the wall clock (raw ticks are not ns). */
const realNs = (unit: TimingUnit, timing: Timing): number | undefined =>
  unit === "ns" ? timing.perDispatch : timing.wallNsPerDispatch;

/** The bench's per-run state (the context, the variant choice and the progress hooks). */
type Bench = {
  readonly context: SweepContext;
  readonly dot: I8a8Dot;
  readonly profile: GeometryProfile;
  readonly device: AttentionDeviceFacts;
  readonly rounds: number;
  readonly limit: number;
  readonly signal?: AbortSignal;
  readonly status: (message: string) => void;
};

class BenchAborted extends Error {}

const throwIfAborted = (bench: Bench): void => {
  if (bench.signal?.aborted === true) throw new BenchAborted("aborted");
};

/** Measures one dispatch (the stage's write target is reset to 0 before its first run). */
const timeStage = async (
  bench: Bench,
  stage: string,
  launch: Launch,
  resources: ReadonlyMap<string, GPUBuffer>,
): Promise<AttentionBenchStage> => {
  throwIfAborted(bench);
  const base = { stage, key: launch.key, workgroups: launch.workgroups };
  try {
    const output = resources.get(launch.writes[0]);
    if (output === undefined)
      throw new Error(`${launch.key}: output ${launch.writes[0]} is missing`);
    const ready = await prepareLaunch(bench.context, launch, resources);
    let timing: Timing;
    try {
      timing = await timeLaunch(bench.context, ready, output, bench.rounds);
    } finally {
      ready.params.destroy();
    }
    const ns = realNs(bench.context.unit, timing);
    return {
      ...base,
      reps: timing.reps,
      rounds: timing.rounds,
      wallRounds: timing.wallRounds,
      clampedNegativeSamples: timing.clampedNegativeSamples,
      ...(ns === undefined ? {} : { msPerDispatch: ns / 1e6 }),
      wallMsPerDispatch: timing.wallNsPerDispatch / 1e6,
      ...(ns === undefined ? { error: "every round's timestamp was negative" } : {}),
    };
  } catch (cause) {
    if (cause instanceof BenchAborted) throw cause;
    return { ...base, error: describeCause(cause) };
  }
};

/** Measures two dispatches in the same rounds (each stage's write target is reset to 0 first). */
const timePair = async (
  bench: Bench,
  stages: readonly [string, string],
  a: Launch,
  b: Launch,
  resources: ReadonlyMap<string, GPUBuffer>,
): Promise<AttentionBenchPair> => {
  throwIfAborted(bench);
  const base = {
    stages,
    keys: [a.key, b.key] as const,
    workgroups: [a.workgroups, b.workgroups] as const,
  };
  try {
    const outputA = resources.get(a.writes[0]);
    const outputB = resources.get(b.writes[0]);
    if (outputA === undefined || outputB === undefined) {
      throw new Error(`${a.key} / ${b.key}: output ${a.writes[0]} / ${b.writes[0]} is missing`);
    }
    const readyA = await prepareLaunch(bench.context, a, resources);
    try {
      const readyB = await prepareLaunch(bench.context, b, resources);
      try {
        const timing = await timeLaunchPair(
          bench.context,
          readyA,
          readyB,
          { a: outputA, b: outputB },
          bench.rounds,
        );
        return { ...base, ...summarizePair(bench.context.unit, timing) };
      } finally {
        readyB.params.destroy();
      }
    } finally {
      readyA.params.destroy();
    }
  } catch (cause) {
    if (cause instanceof BenchAborted) throw cause;
    return { ...base, error: describeCause(cause) };
  }
};

/** Allocates and fills `specs`, runs `body`, and releases them (also on failure). */
const withResources = async <T>(
  bench: Bench,
  specs: readonly ResourceSpec[],
  body: (resources: ReadonlyMap<string, GPUBuffer>) => Promise<T>,
): Promise<T> => {
  const resources = await createResources(bench.context, specs);
  try {
    return await body(resources);
  } finally {
    for (const buffer of resources.values()) buffer.destroy();
  }
};

/** Sums the stages into the route's totals (absent when any stage has no time). */
const routeTotals = (
  shape: AttentionBenchShape,
  rows: number,
  stages: readonly AttentionBenchStage[],
): Pick<
  Extract<AttentionBenchRoute, { eligible: true }>,
  "msPerDispatch" | "msPer1000Rows" | "tops" | "estimatedNodeMs" | "error"
> => {
  const failed = stages.find((stage) => stage.msPerDispatch === undefined);
  if (failed !== undefined) return { error: `${failed.stage}: ${failed.error ?? "no time"}` };
  const ms = stages.reduce((sum, stage) => sum + (stage.msPerDispatch ?? 0), 0);
  if (!(ms > 0)) return { msPerDispatch: ms, error: "the summed time is 0" };
  const msPer1000Rows = (ms / rows) * 1000;
  return {
    msPerDispatch: ms,
    msPer1000Rows,
    tops: attentionIntegerOps(shape.heads, rows, shape.keys, shape.depth) / (ms * 1e6) / 1e3,
    estimatedNodeMs: (msPer1000Rows * shape.queries) / 1000,
  };
};

/** Bytes of the quantised Vᵀ of a shape's node in a key-axis form (the production function). */
const valueBytes = (shape: AttentionBenchShape, keyWords: AttentionOnlineKeyWords): number =>
  attentionI8a8TempBytes(
    { batch: shape.heads, rows: shape.queries, cols: shape.keys, depth: shape.depth },
    keyWords,
  ).pv.vq;

/**
 * The shared i8a8 inputs and the output O (both routes read the same buffers). The quantised Vᵀ is
 * sized in `keyWords`, the online route's form on this device (the largest a shared reader takes:
 * the tiled form's `16 · ceil(N / 64)` words per column are never fewer than N's `ceil(N / 4)`).
 */
const sharedResources = (
  shape: AttentionBenchShape,
  keyWords: AttentionOnlineKeyWords,
): ResourceSpec[] => {
  const { heads, queries, keys, depth } = shape;
  return [
    // Quantized q / k (four i8 per word) and their per-token scales (attention.ts:1386-1390 via
    // attentionI8a8TempBytes), the quantized Vᵀ[heads, D, words of the form] and its per-column
    // scales (attention.ts:1505-1508, the same function), and O[heads, M, D] in f32
    { name: "qq", bytes: heads * queries * depth, fill: "i8" },
    { name: "qscale", bytes: heads * queries * F32, fill: "scale" },
    { name: "kq", bytes: heads * keys * depth, fill: "i8" },
    { name: "kscale", bytes: heads * keys * F32, fill: "scale" },
    { name: "vq", bytes: valueBytes(shape, keyWords), fill: "i8" },
    { name: "vscale", bytes: heads * depth * F32, fill: "scale" },
    { name: "o", bytes: heads * queries * depth * F32, fill: "none" },
  ];
};

/** The s16 route at one shape (three dispatches of one row block). */
const benchS16 = async (
  bench: Bench,
  shape: AttentionBenchShape,
  shared: ReadonlyMap<string, GPUBuffer>,
): Promise<AttentionBenchRoute> => {
  const { id, heads, queries, keys, depth, scale } = shape;
  const where = `${id} s16`;
  const route = attentionRoute(
    { attentionCompute: "a8", attentionScoreStorage: "f16" },
    depth,
    keys,
    bench.device,
  );
  if (route.online !== undefined || route.qk !== "i8a8" || route.pv !== "i8a8") {
    return {
      eligible: false,
      reason: `production runs ①QK ${route.qk} / ③PV ${route.pv} with S in ${route.score} for D=${depth}, N=${keys} (not the a8 s16 route)`,
    };
  }
  if (route.score !== "f16") {
    return {
      eligible: false,
      reason: `S cannot be stored in f16 for D=${depth}, N=${keys} (production stores it in ${route.score})`,
    };
  }
  const violation = fusedAttentionViolation(
    where,
    { heads, kvHeads: heads, depth, cols: keys },
    { i8a8: true, mask: false, route },
  );
  if (violation !== undefined) return { eligible: false, reason: violation };

  // attention.ts:570-579 — the even split under the device's binding limit (no budget: the Wan
  // pipeline passes none)
  const scoreBytes = attentionScoreElementBytes("a8", route);
  const blocks = planAttentionBlocks(
    { batch: heads, rows: queries, cols: keys, scoreBytes },
    rowBlockLimit(bench.context.gpu.limits.maxStorageBufferBindingSize, undefined),
  );
  const rows = shape.rowsPerDispatch ?? blocks[0].rows;
  // attention.ts:579 — the row-window variant runs whenever the node is split; a measured block
  // smaller than M is one block of such a split
  const rowWindow = rows < queries;
  const window = rowWindow ? { offset: 0, rowsFull: queries } : undefined;
  const qkGeometry = bench.profile.i8a8.attentionQk;
  const pvGeometry = bench.profile.i8a8.attentionPv;
  const dp4a = bench.dot === "dp4a";
  const { limit } = bench;
  // attention.ts:1349-1356
  const qkV4 = attentionQkI8a8UsesVec4(keys);
  const qk: Launch = {
    key: attentionQkI8a8Key(qkV4, dp4a, "f16", qkGeometry, rowWindow),
    wgsl: attentionQkI8a8Wgsl(qkV4, dp4a, "f16", qkGeometry, rowWindow),
    // attention.ts:679-681
    params: attentionQkI8a8Params(rows, keys, depth, scale, window),
    // attention.ts:683-689
    bindings: positionalBindings(`${where} ①QK`, [
      [1, "qq"],
      [2, "kq"],
      [3, "s"],
      [ATTENTION_QK_Q_SCALE_BINDING, "qscale"],
      [ATTENTION_QK_K_SCALE_BINDING, "kscale"],
    ]),
    writes: ["s"],
    // attention.ts:690-694
    workgroups: [
      tiledWorkgroups(keys, i8a8TileN(qkGeometry), limit, `${where} ①QK`),
      tiledWorkgroups(rows, i8a8TileM(qkGeometry), limit, `${where} ①QK`),
      tiledWorkgroups(heads, 1, limit, `${where} ①QK`),
    ],
  };
  // attention.ts:650-655 (a8 computes ② in f32 — `compute` is "f32" unless attentionCompute is f16)
  const regCache = attentionStatsRegCache(keys);
  const stats: Launch = {
    key: attentionStatsKey("f32", "f16", regCache),
    wgsl: attentionStatsWgsl("f32", "f16", regCache),
    // attention.ts:727-735
    params: attentionStatsParams(heads * rows, keys, regCache),
    bindings: ["s", "stats"],
    writes: ["stats"],
    workgroups: [gridStrideWorkgroups(heads * rows, 1, limit), 1, 1],
  };
  // attention.ts:1470-1477
  const pvV4 = attentionPvI8a8UsesVec4(depth);
  const pv: Launch = {
    key: attentionPvI8a8Key(pvV4, dp4a, "f16", pvGeometry, rowWindow),
    wgsl: attentionPvI8a8Wgsl(pvV4, dp4a, "f16", pvGeometry, rowWindow),
    // attention.ts:747-749
    params: attentionPvI8a8Params(rows, depth, keys, window),
    // attention.ts:751-757
    bindings: positionalBindings(`${where} ③PV`, [
      [1, "s"],
      [2, "vq"],
      [3, "stats"],
      [4, "o"],
      [ATTENTION_PV_V_SCALE_BINDING, "vscale"],
    ]),
    writes: ["o"],
    // attention.ts:758-762
    workgroups: [
      tiledWorkgroups(depth, i8a8TileN(pvGeometry), limit, `${where} ③PV`),
      tiledWorkgroups(rows, i8a8TileM(pvGeometry), limit, `${where} ③PV`),
      tiledWorkgroups(heads, 1, limit, `${where} ③PV`),
    ],
  };
  const base = {
    eligible: true,
    rows,
    productionDispatches: blocks.length,
    productionRows: blocks[0].rows,
    rowWindow,
  } as const;
  // S and the row statistics of one block (attention.ts:666-667)
  const blockSpecs: ResourceSpec[] = [
    { name: "s", bytes: heads * rows * keys * scoreBytes, fill: "none" },
    { name: "stats", bytes: heads * rows * ATTENTION_STATS_STRIDE * F32, fill: "none" },
  ];
  let stages: AttentionBenchStage[];
  try {
    stages = await withResources(bench, blockSpecs, async (block) => {
      const resources = new Map([...shared, ...block]);
      const measured: AttentionBenchStage[] = [];
      // ① first, then ② and ③: each stage's repeated runs rewrite the same values, so ③ reads the
      // S and statistics ① and ② left
      for (const [stage, launch] of [
        ["qk", qk],
        ["stats", stats],
        ["pv", pv],
      ] as const) {
        bench.status(`${id}: s16 ${stage} (${rows} rows)`);
        measured.push(await timeStage(bench, stage, launch, resources));
      }
      return measured;
    });
  } catch (cause) {
    if (cause instanceof BenchAborted) throw cause;
    return { ...base, stages: [], error: describeCause(cause) };
  }
  return { ...base, stages, ...routeTotals(shape, rows, stages) };
};

/** Why the online kernel does not run (the two conditions of `attentionRoute`). */
const onlineIneligibleReason = (
  shape: AttentionBenchShape,
  device: AttentionDeviceFacts,
): string =>
  attentionOnlineI8a8Eligible(shape.depth, shape.keys)
    ? `the device holds neither reduction form (subgroups ${device.subgroups}; workgroup storage sg32 ${attentionOnlineI8a8WorkgroupStorageBytes(
        shape.depth,
        "sg32",
      )} B / wg ${attentionOnlineI8a8WorkgroupStorageBytes(shape.depth, "wg")} B against ${
        device.maxComputeWorkgroupStorageSize
      } B)`
    : `the shape is not eligible (D=${shape.depth}, N=${shape.keys}: needs D % 64 == 0, D ≤ 128, N ≥ 1)`;

/** The resources an online dispatch binds (today's route reads the shared `vq` / `vscale` / `o`). */
type OnlineBuffers = { readonly vq: string; readonly vs: string; readonly o: string };
const SHARED_ONLINE: OnlineBuffers = { vq: "vq", vs: "vscale", o: "o" };

/**
 * One online dispatch of the first window (`rows` rows) in the key-axis form `keyWords`: the
 * route's form for the online row, N's form and `tiled` for the two arms of the candidate pair —
 * the same generator calls either way.
 */
const onlineLaunch = (
  bench: Bench,
  shape: AttentionBenchShape,
  reduce: AttentionOnlineReduce,
  rows: number,
  keyWords: AttentionOnlineKeyWords,
  buffers: OnlineBuffers = SHARED_ONLINE,
): Launch => {
  const { id, heads, queries, keys, depth, scale } = shape;
  const where = `${id} online${keyWords === "tiled" ? " tiled" : ""}`;
  const dp4a = bench.dot === "dp4a";
  return {
    // attention.ts:1591-1596
    key: attentionOnlineI8a8Key(dp4a, depth, reduce, "bhsd", keyWords),
    wgsl: attentionOnlineI8a8Wgsl(dp4a, depth, reduce, "bhsd", keyWords),
    // attention.ts:1603-1606 (the first window: row_offset 0, m = the node's M)
    params: attentionOnlineI8a8Params(queries, keys, 0, scale, keyWords),
    // attention.ts:1607-1615
    bindings: positionalBindings(where, [
      [1, "qq"],
      [2, "kq"],
      [3, buffers.vq],
      [4, buffers.o],
      [5, "qscale"],
      [6, "kscale"],
      [7, buffers.vs],
    ]),
    writes: [buffers.o],
    // attention.ts:1618-1622
    workgroups: [
      tiledWorkgroups(rows, ATTENTION_ONLINE_ROWS, bench.limit, where),
      tiledWorkgroups(heads, 1, bench.limit, where),
      1,
    ],
  };
};

/**
 * The Vᵀ producer in the key-axis form `keyWords` over the `rows` rows `(head, d)` of Vᵀ, row
 * length N (attention.ts `attentionValueQuantization` — the production launch, on the grid-stride
 * grid production gives it): N's form for the quantization line and the candidate pair's A, `tiled`
 * for its B. At N % 4 != 0 only the online route quantises (s16 is not eligible there).
 */
const valueQuantizeLaunch = (
  bench: Bench,
  rows: number,
  keys: number,
  keyWords: AttentionOnlineKeyWords,
  names: readonly [vt: string, vq: string, vs: string],
): Launch => {
  const { geometry, key, wgsl, params } = attentionValueQuantization(rows, keys, keyWords);
  return {
    key,
    wgsl,
    params,
    bindings: [...names],
    writes: [names[1], names[2]],
    workgroups: [gridStrideWorkgroups(rows, geometry.rowsPerGroup, bench.limit), 1, 1],
  };
};

/** The online route at one shape (one dispatch of one window). */
const benchOnline = async (
  bench: Bench,
  shape: AttentionBenchShape,
  shared: ReadonlyMap<string, GPUBuffer>,
): Promise<AttentionBenchRoute> => {
  const { id, heads, queries, keys, depth } = shape;
  const where = `${id} online`;
  const route = attentionRoute(
    { attentionCompute: "a8", attentionScoreStorage: "none" },
    depth,
    keys,
    bench.device,
  );
  if (route.online === undefined) {
    return { eligible: false, reason: onlineIneligibleReason(shape, bench.device) };
  }
  const violation = fusedAttentionViolation(
    where,
    { heads, kvHeads: heads, depth, cols: keys },
    { i8a8: true, mask: false, route },
  );
  if (violation !== undefined) return { eligible: false, reason: violation };
  const reduce = route.online;
  // attention.ts:1597 — the first window production dispatches
  const windows = planAttentionOnlineWindows(queries, { batch: heads, cols: keys, depth });
  const rows = onlineDispatchRows(shape.rowsPerDispatch ?? windows[0].rows, queries);
  const launch = onlineLaunch(bench, shape, reduce, rows, route.keyWords);
  bench.status(`${id}: online ${reduce} ${route.keyWords} (${rows} rows)`);
  const stages = [await timeStage(bench, "online", launch, shared)];
  return {
    eligible: true,
    rows,
    productionDispatches: windows.length,
    productionRows: windows[0].rows,
    reduce,
    keyWords: route.keyWords,
    workgroupStorageBytes: attentionOnlineI8a8WorkgroupStorageBytes(depth, reduce),
    stages,
    ...routeTotals(shape, rows, stages),
  };
};

/**
 * The identity block of the candidate: N's producer and the tiled producer on the same f32 Vᵀ, then
 * N's kernel and the tiled kernel (the first window) on their outputs. Every O word and every Vᵀ
 * scale word must be equal; O starts at 0 on both sides, so rows past the window compare 0 with 0. A
 * failure to run is recorded as not identical, with the error.
 */
const benchIdentity = async (
  bench: Bench,
  shape: AttentionBenchShape,
  shared: ReadonlyMap<string, GPUBuffer>,
  reduce: AttentionOnlineReduce,
  rows: number,
): Promise<AttentionBenchIdentity> => {
  const { id, heads, queries, keys, depth } = shape;
  const own = attentionOnlineKeyWords(keys);
  try {
    const valueRows = heads * depth;
    const launches: readonly Launch[] = [
      valueQuantizeLaunch(bench, valueRows, keys, own, ["vtIdentity", "vqA", "vsA"]),
      valueQuantizeLaunch(bench, valueRows, keys, "tiled", ["vtIdentity", "vqB", "vsB"]),
      onlineLaunch(bench, shape, reduce, rows, own, {
        vq: "vqA",
        vs: "vsA",
        o: "oA",
      }),
      onlineLaunch(bench, shape, reduce, rows, "tiled", { vq: "vqB", vs: "vsB", o: "oB" }),
    ];
    const oBytes = heads * queries * depth * F32;
    const scaleBytes = valueRows * F32;
    const specs: ResourceSpec[] = [
      { name: "vtIdentity", bytes: valueRows * keys * F32, fill: "f32" },
      { name: "vqA", bytes: valueBytes(shape, own), fill: "none" },
      { name: "vsA", bytes: scaleBytes, fill: "none" },
      { name: "vqB", bytes: valueBytes(shape, "tiled"), fill: "none" },
      { name: "vsB", bytes: scaleBytes, fill: "none" },
      { name: "oA", bytes: oBytes, fill: "none" },
      { name: "oB", bytes: oBytes, fill: "none" },
    ];
    bench.status(`${id}: tiled identity (both producers, both kernels)`);
    return await withResources(bench, specs, async (extra) => {
      const resources = new Map([...shared, ...extra]);
      for (const launch of launches) {
        throwIfAborted(bench);
        const ready = await prepareLaunch(bench.context, launch, resources);
        try {
          await runLaunchOnce(bench.context, ready, resources.get(launch.writes[0]));
        } finally {
          ready.params.destroy();
        }
      }
      const buffer = (name: string): GPUBuffer => {
        const found = resources.get(name);
        if (found === undefined) throw new Error(`${id}: resource ${name} is missing`);
        return found;
      };
      const scales = await compareBufferWords(
        bench.context,
        buffer("vsA"),
        buffer("vsB"),
        scaleBytes,
      );
      const o = await compareBufferWords(bench.context, buffer("oA"), buffer("oB"), oBytes);
      return {
        identical: o.differing === 0 && scales.differing === 0,
        words: o.words,
        differingWords: o.differing,
        differingScaleWords: scales.differing,
        ...(o.first === undefined ? {} : { first: o.first }),
      };
    });
  } catch (cause) {
    if (cause instanceof BenchAborted) throw cause;
    return { identical: false, error: describeCause(cause) };
  }
};

/**
 * The candidate at one shape: `online` in N's form (A) and `onlineTiled` (B) on the online route's
 * first window, in the same rounds, then the identity block. The tiled kernel reads its own Vᵀ
 * buffer (the tiled size, filled like the shared one — the timing does not depend on the values).
 */
const benchOnlineTiled = async (
  bench: Bench,
  shape: AttentionBenchShape,
  shared: ReadonlyMap<string, GPUBuffer>,
  online: AttentionBenchRoute,
): Promise<AttentionBenchTiled> => {
  if (!online.eligible || online.reduce === undefined) {
    return {
      eligible: false,
      reason: `the online route does not run here${online.eligible ? "" : ` (${online.reason})`}`,
    };
  }
  const { id, keys, queries } = shape;
  const { reduce, rows } = online;
  let a: Launch;
  let b: Launch;
  try {
    a = onlineLaunch(bench, shape, reduce, rows, attentionOnlineKeyWords(keys));
    b = onlineLaunch(bench, shape, reduce, rows, "tiled", { ...SHARED_ONLINE, vq: "vqTiled" });
  } catch (cause) {
    return { eligible: false, reason: `the candidate cannot be built: ${describeCause(cause)}` };
  }
  bench.status(`${id}: online vs online tiled, A B B A (${rows} rows)`);
  const kernel = await withResources(
    bench,
    [{ name: "vqTiled", bytes: valueBytes(shape, "tiled"), fill: "i8" }],
    (extra) => timePair(bench, ["online", "onlineTiled"], a, b, new Map([...shared, ...extra])),
  );
  const nodeMs = (ms: number | undefined): number | undefined =>
    ms === undefined ? undefined : (ms * queries) / rows;
  const onlineNodeMs = nodeMs(kernel.aMsPerDispatch);
  const tiledNodeMs = nodeMs(kernel.bMsPerDispatch);
  const identity = await benchIdentity(bench, shape, shared, reduce, rows);
  return {
    eligible: true,
    rows,
    kernel,
    ...(onlineNodeMs === undefined ? {} : { onlineNodeMs }),
    ...(tiledNodeMs === undefined ? {} : { tiledNodeMs }),
    identity,
  };
};

/** The four quantization dispatches of one node (attention.ts:1373-1424 and :1493-1559). */
const benchQuantization = async (
  bench: Bench,
  shape: AttentionBenchShape,
  shared: ReadonlyMap<string, GPUBuffer>,
  tiledVt: boolean,
): Promise<NonNullable<AttentionBenchShapeResult["quantization"]>> => {
  const { id, heads, queries, keys, depth } = shape;
  const { limit } = bench;
  const tokens = quantizeRowsGeometry(depth);
  const quantize = (
    geometry: ReturnType<typeof quantizeRowsGeometry>,
    rows: number,
    dim: number,
    names: readonly [string, string, string],
  ): Launch => ({
    key: quantizeRowsKey(geometry),
    wgsl: quantizeRowsWgsl(geometry),
    params: quantizeRowsParams(rows, dim),
    bindings: [...names],
    writes: [names[1], names[2]],
    workgroups: [gridStrideWorkgroups(rows, geometry.rowsPerGroup, limit), 1, 1],
  });
  const valueRows = heads * depth;
  // N's form (the pair's A — a fixed baseline whatever form the profile chooses)
  const quantizeVt = valueQuantizeLaunch(bench, valueRows, keys, attentionOnlineKeyWords(keys), [
    "vt",
    "vq",
    "vscale",
  ]);
  const tiledProducer = tiledVt
    ? valueQuantizeLaunch(bench, valueRows, keys, "tiled", ["vt", "vqTiled", "vscaleTiled"])
    : undefined;
  const stridedSpec = { dtype: "f32" } as const;
  const launches: readonly (readonly [string, Launch])[] = [
    // attention.ts:1402-1423 (q per token, then k per token)
    ["quantize_q", quantize(tokens, heads * queries, depth, ["q", "qq", "qscale"])],
    ["quantize_k", quantize(tokens, heads * keys, depth, ["k", "kq", "kscale"])],
    // attention.ts:1514-1535 (v[heads, N, D] → Vᵀ[heads, D, N]; storage params)
    [
      "permute_v",
      {
        key: stridedKey(stridedSpec),
        wgsl: stridedWgsl(stridedSpec),
        params: stridedParams(
          [heads, depth, keys],
          permuteSrcStrides([heads, keys, depth], [0, 2, 1]),
          0,
        ),
        paramsUsage: "storage",
        bindings: ["v", "vt"],
        writes: ["vt"],
        workgroups: [
          gridStrideWorkgroups(heads * depth * keys, STRIDED_WORKGROUP_SIZE, limit),
          1,
          1,
        ],
      },
    ],
    // attention.ts:1539-1558 (rows = (head, d), row length N — the padded form at N % 4 != 0)
    ["quantize_vt", quantizeVt],
  ];
  const specs: ResourceSpec[] = [
    { name: "q", bytes: heads * queries * depth * F32, fill: "f32" },
    { name: "k", bytes: heads * keys * depth * F32, fill: "f32" },
    { name: "v", bytes: heads * keys * depth * F32, fill: "f32" },
    { name: "vt", bytes: heads * depth * keys * F32, fill: "none" },
    // The candidate producer's outputs (only with the candidate)
    ...(tiledProducer !== undefined
      ? [
          { name: "vqTiled", bytes: valueBytes(shape, "tiled"), fill: "none" } as const,
          { name: "vscaleTiled", bytes: valueRows * F32, fill: "none" } as const,
        ]
      : []),
  ];
  const { stages, pair } = await withResources(bench, specs, async (inputs) => {
    const resources = new Map([...shared, ...inputs]);
    const measured: AttentionBenchStage[] = [];
    for (const [stage, launch] of launches) {
      bench.status(`${id}: ${stage}`);
      measured.push(await timeStage(bench, stage, launch, resources));
    }
    if (tiledProducer === undefined) return { stages: measured };
    // permute_v above left Vᵀ in `vt`, so both producers read the same values
    bench.status(`${id}: quantize_vt vs the tiled producer, A B B A`);
    return {
      stages: measured,
      pair: await timePair(
        bench,
        ["quantize_vt", "quantize_vt_tiled"],
        quantizeVt,
        tiledProducer,
        resources,
      ),
    };
  });
  const complete = stages.every((stage) => stage.msPerDispatch !== undefined);
  return {
    stages,
    ...(complete
      ? { msPerNode: stages.reduce((sum, stage) => sum + (stage.msPerDispatch ?? 0), 0) }
      : {}),
    ...(pair === undefined ? {} : { tiledVt: pair }),
  };
};

/**
 * A route whose planning throws (a row that does not fit the binding limit, a workgroup count over
 * the device limit — production fails loudly there too) is reported as not eligible with the
 * reason, so that one shape does not stop the bench.
 */
const planned = async (route: () => Promise<AttentionBenchRoute>): Promise<AttentionBenchRoute> => {
  try {
    return await route();
  } catch (cause) {
    if (cause instanceof BenchAborted) throw cause;
    return { eligible: false, reason: `production fails: ${describeCause(cause)}` };
  }
};

/** The candidate, with a failure to allocate or build reported as not eligible (as {@link planned}). */
const plannedTiled = async (
  candidate: () => Promise<AttentionBenchTiled>,
): Promise<AttentionBenchTiled> => {
  try {
    return await candidate();
  } catch (cause) {
    if (cause instanceof BenchAborted) throw cause;
    return { eligible: false, reason: `the candidate fails: ${describeCause(cause)}` };
  }
};

const benchShape = async (
  bench: Bench,
  shape: AttentionBenchShape,
  quantization: boolean,
  tiled: boolean,
): Promise<AttentionBenchShapeResult> => {
  bench.status(`${shape.id}: preparing inputs`);
  // The online route's Vᵀ form on this device (N's own where the route does not run online)
  const onlineRoute = attentionRoute(
    { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape.depth,
    shape.keys,
    bench.device,
  );
  const sharedForm =
    onlineRoute.online === undefined ? attentionOnlineKeyWords(shape.keys) : onlineRoute.keyWords;
  return await withResources(bench, sharedResources(shape, sharedForm), async (shared) => {
    const s16 = await planned(() => benchS16(bench, shape, shared));
    const online = await planned(() => benchOnline(bench, shape, shared));
    const ratio = finiteRatio(
      online.eligible ? online.msPer1000Rows : undefined,
      s16.eligible ? s16.msPer1000Rows : undefined,
    );
    const onlineTiled = tiled
      ? await plannedTiled(() => benchOnlineTiled(bench, shape, shared, online))
      : undefined;
    const measureQuantization = quantization && (s16.eligible || online.eligible);
    const quantized = measureQuantization
      ? await benchQuantization(bench, shape, shared, onlineTiled?.eligible === true)
      : undefined;
    const tiledVsOnline = onlineTiled?.eligible === true ? onlineTiled.kernel.ratio : undefined;
    const withProducer =
      onlineTiled?.eligible === true
        ? withProducerRatio(onlineTiled, quantized?.tiledVt)
        : undefined;
    return {
      shape,
      s16,
      online,
      ...(ratio === undefined ? {} : { onlineVsS16: ratio }),
      ...(onlineTiled === undefined ? {} : { onlineTiled }),
      ...(tiledVsOnline === undefined ? {} : { tiledVsOnline }),
      ...(withProducer === undefined ? {} : { tiledVsOnlineWithProducer: withProducer }),
      ...(quantized === undefined ? {} : { quantization: quantized }),
    };
  });
};

/**
 * The integer dot-product variant Session construction would choose for attention
 * (src/runtime/session-build.ts:301-321, without the test-only `I8A8_DOT`): emulated when the WGSL
 * language feature is not listed, otherwise the canary's verdict on the profile's geometry.
 */
const chooseDot = async (
  gpu: GpuContext,
  profile: GeometryProfile,
): Promise<AttentionBenchVariants["dp4a"]> => {
  if (!dp4aAvailable(gpu.wgslLanguageFeatures)) return { listed: false, dot: "emu" };
  const decision = await decideAttentionI8a8Dot(gpu, {
    qk: profile.i8a8.attentionQk,
    pv: profile.i8a8.attentionPv,
  });
  return {
    listed: true,
    dot: decision.dot,
    canary: {
      branch: decision.branch,
      exact: decision.exact,
      maxAbsError: decision.maxAbsError,
      maxBandRatio: decision.maxBandRatio,
    },
  };
};

/**
 * Times the s16 and online routes of a8 fused attention at each shape on `gpu` and returns the
 * record (`karume-attention-bench/2`, JSON-serializable); with `onlineTiled`, also the candidate. Uses GPU timestamps when the device was
 * acquired with `gpuTiming: true`, the wall clock otherwise. A route production would not run on
 * this device or shape is reported as not eligible; a failed dispatch is kept as an `error` of its
 * stage and the bench goes on. Option errors throw before the GPU is touched.
 *
 * MUST: do not run inference on the same GPU meanwhile (the caller's contract — module doc).
 */
export const runAttentionRouteBench = async (
  gpu: GpuContext,
  options: AttentionBenchOptions = {},
): Promise<AttentionBenchReport> => {
  const shapes = options.shapes ?? WAN22_ATTENTION_SHAPES;
  if (shapes.length === 0) throw new Error("No shape to measure");
  shapes.forEach(checkAttentionBenchShape);
  const rounds = options.rounds ?? ROUNDS;
  if (!isPositiveInteger(rounds)) {
    throw new Error(`rounds must be an integer of at least 1 (${rounds})`);
  }
  const quantization = options.quantization ?? true;
  const onlineTiled = options.onlineTiled ?? false;
  const startedAt = new Date().toISOString();
  const notify = options.onProgress;
  const profile = sessionGeometryProfile(gpu);
  notify?.({ kind: "status", message: "deciding the integer dot-product variant (canary)" });
  const dp4a = await chooseDot(gpu, profile);
  const device = attentionDeviceFacts(gpu);
  const context = await createSweepContext(gpu, timestampUnitFor(navigator.userAgent));
  const results: AttentionBenchShapeResult[] = [];
  let aborted = false;
  try {
    const bench: Bench = {
      context,
      dot: dp4a.dot,
      profile,
      device,
      rounds,
      limit: gpu.limits.maxComputeWorkgroupsPerDimension,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      status: (message) => notify?.({ kind: "status", message }),
    };
    for (const shape of shapes) {
      let result: AttentionBenchShapeResult;
      try {
        result = await benchShape(bench, shape, quantization, onlineTiled);
      } catch (cause) {
        if (!(cause instanceof BenchAborted)) throw cause;
        aborted = true;
        break;
      }
      results.push(result);
      // MUST: the user's callback runs outside the measurement's try (an exception it throws is the
      // caller's, not a measurement failure — the same rule as the sweep)
      notify?.({ kind: "shape", result });
    }
  } finally {
    destroySweepContext(context);
  }
  const info = gpu.adapterInfo;
  return {
    format: ATTENTION_BENCH_FORMAT,
    startedAt,
    date: new Date().toISOString(),
    aborted,
    userAgent: navigator.userAgent,
    adapter: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
    timing: { unit: context.unit, gpuTimingEnabled: gpu.gpuTimingEnabled },
    variants: {
      geometryProfile: profile.id,
      attentionQkGeometry: profile.i8a8.attentionQk,
      attentionPvGeometry: profile.i8a8.attentionPv,
      dp4a,
      device: {
        ...device,
        subgroupLanesLinear: gpu.subgroupLanesLinear,
        features: {
          subgroups: gpu.features.has("subgroups"),
          subgroupSizeControl: gpu.features.has("subgroup-size-control"),
          subgroupId: gpu.wgslLanguageFeatures.has("subgroup_id"),
        },
        maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension: gpu.limits.maxComputeWorkgroupsPerDimension,
      },
    },
    settings: {
      rounds,
      quantization,
      onlineTiled,
      targetPassMs: TARGET_PASS_MS,
      maxReps: SWEEP_MAX_REPS,
      warmupNs: WARMUP_NS,
      warmupMinRuns: WARMUP_MIN_RUNS,
    },
    shapes: results,
  };
};
