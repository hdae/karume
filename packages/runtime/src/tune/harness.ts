/**
 * The measurement core of the tile-geometry sweep (perf-ledger K-70 — used by the sweep entry point
 * `runGeometrySweep` (sweep.ts), which the Deno CLI, the sweep tab of the browser page
 * `tools/gpu-lab` (both removed in 0.15) and user applications share. It uses only the WebGPU
 * standard API and the runtime's src).
 *
 * It measures the time of one dispatch, changing **only the geometry** while keeping the same shape
 * and the same inputs. WGSL and keys are built by passing an explicit geometry to the runtime's
 * generator entry points (`linearWgsl(…, geometry)` etc. — the Session path is not used), so the
 * measured kernels are exactly the output of the same generators as the production kernels.
 *
 * ## Measurement protocol (measurement.ts — shared with tools/opbench, removed in 0.15)
 *
 * 1. Stack `reps` copies of the same dispatch in one compute pass, and take the difference of the
 *    pass's `timestampWrites` (beginning / end) ÷ `reps` as the time of one dispatch. `reps` is the
 *    count that makes one pass ≈ {@link TARGET_PASS_MS} ({@link calibrateReps}; the upper bound is
 *    the sweep-specific {@link SWEEP_MAX_REPS}). A calibration whose pass comes out below
 *    {@link RECALIBRATE_BELOW} of the target while its reps are under the bound is done again after
 *    a warm start, up to {@link CALIBRATION_ATTEMPTS} calibrations ({@link calibrate}): an estimate
 *    taken while the dispatch was slow (a cold device, a stall) would leave every round a short pass
 *    on a device that is no longer under load.
 * 2. The representative value is the **min** of `rounds` (absorbs thermal drift). A round whose
 *    timestamp difference is negative is clamped to 0 and kept in the record, but excluded from the
 *    min candidates (if every round is negative, the row fails).
 * 3. Before the first measurement of a case (its buffers and prelude made), a **warm start** runs
 *    that measurement's dispatch under load, each pass kept near the target length, until
 *    {@link WARM_START_PASSES} consecutive passes agree within {@link WARM_START_TOLERANCE},
 *    spending at least {@link WARM_START_MIN_NS} and at most {@link WARM_START_MAX_NS} of load (and
 *    at most {@link WARM_START_MAX_PASSES} passes) ({@link warmStart}): the case's first
 *    calibration then reads a device that left its idle
 *    clocks. For each geometry, before measuring, run passes idle until "cumulative ≥
 *    {@link WARMUP_NS} and ≥ {@link WARMUP_MIN_RUNS} runs" (the same lower bounds as opbench's
 *    `pinClocks`, with the same cut-off of {@link WARMUP_MAX_PASSES} passes). The cumulative time
 *    counts GPU time for unit `ns` and wall clock for `deno-raw-tick` / `wall` (raw ticks are not
 *    ns).
 * 4. The **quick look** (both sweeps — measurement.ts's `QUICK_LOOK_MARGIN` states the rule and its
 *    evidence): after the warm start of a group of kernels measured together (a geometry case; a
 *    GEMV case at one row count), every kernel gets one short timed pass — the calibration's probe
 *    with timestamps, read at its last pass ({@link quickLook}) — in a shuffled order
 *    ({@link quickLookOrder}). Only the references (the default geometry; the GEMV baseline and the
 *    default profile's choice at `reorder`) and the kernels whose quick time is within the margin of
 *    their class's bar ({@link quickLookGoesOn}) go on to the calibration's check (from the probe's
 *    reps), the correctness check and the rounds. A kernel that does not go on is a row with its
 *    quick time and `screened: true`: no rounds, no correctness verdict, never evidence for a
 *    derivation. A margin of 0 turns the quick look off.
 * 5. At the end of a case, measure the default geometry once more and keep the ratio to the first
 *    (`driftRatio`) per case — a sign of whether heat or clocks moved during the sweep (whether the
 *    default value the ratios rest on is unstable).
 * 6. Never compare absolute values with another machine — only ratios within one machine (speed
 *    relative to the default geometry) are valid.
 *
 * NOTE: unlike opbench, the input of the `reps` estimate is the **wall clock** (stack `reps` by
 * doubling until the pass's wall clock reaches half the target, then take the slope of the last two
 * points = the increment of one dispatch). Deno does not convert timestamps to ns (1 tick = 52.08
 * ns on B570), so an estimate from timestamps inflates the count 52-fold, and Chrome without
 * developer flags rounds to 100 µs, so a short dispatch reads as 0 and gets stacked up to the
 * bound. The slope is taken to cancel the submit-to-completion floor (≈ 11 ms on Deno — measured on
 * B570, the average gave half the count). A device without timestamp-query runs the same protocol
 * by wall clock (unit `wall`).
 *
 * ## Binding sizes (where they are copied from)
 *
 * Bindings are passed **by position**, as in production: each {@link Launch}'s `bindings` is a
 * table copying the order of `binds` (from binding 1 on) that recipe-builders pass to the same
 * kernel. The table is checked against the storage roles that PipelineCache takes from the WGSL
 * (`roles` — `parseStorageRoles` in src/gpu/pipeline-cache.ts); if the binding count or the
 * positions of the write targets disagree, the row fails ({@link assertBindingRoles}). A swap
 * between bindings of the same role is invisible to the roles — for linear / matmul / bmm the
 * CPU-reference comparison test (tests/tune_harness_test.ts) is the detector. The size formulas are
 * copied from each recipe-builder's allocations and params functions (`file:line` on each branch of
 * {@link casePlan}).
 *
 * ## Checking bit identity
 *
 * For each geometry the output is reset to 0 before running, and the output digest
 * ({@link outputDigest}) is compared with the default geometry's row. A match is expected: for f32
 * by ADR 0022 decision 3 (the K reduction order does not depend on the geometry), for i8a8 by the
 * exactness of the integer reduction. A geometry that does not match fails the adoption criteria
 * however fast it is.
 */
// WHY: the public surface (mod.ts) has no errorScope lock, no device-lost race and no pipeline
// cache. This couples to the internal surface on purpose, to measure with the same cache
// (key-collision detection included) and scope discipline as production.
import { type GpuContext, RUNTIME_INTERNAL } from "../gpu/device.ts";
import type { StorageRoles } from "../gpu/pipeline-cache.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "../gpu/error-scope.ts";
import { BUFFER_USAGE, MAP_MODE } from "../gpu/webgpu-constants.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import { gemmMTileGeometry, gemmUsesVec4 } from "../kernels/gemm.ts";
import {
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryForRows,
  gemmTileM,
  gemmTileN,
} from "../kernels/gemm-geometry.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8TileM,
  i8a8TileN,
} from "../kernels/i8a8-geometry.ts";
import { DEFAULT_GEOMETRY_PROFILE, i8a8LinearGeometry } from "../kernels/geometry-profile.ts";
import { linearKey, linearParams, linearWgsl } from "../kernels/linear.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../kernels/matmul.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../kernels/bmm.ts";
import {
  dp4aAvailable,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8UsesVecFill,
  linearI8a8Wgsl,
} from "../kernels/linear-i8a8.ts";
import {
  ATTENTION_STATS_STRIDE,
  attentionPvKey,
  attentionPvParams,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsWgsl,
} from "../kernels/attention.ts";
import {
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
  type Conv2dDims,
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dUsesVec4,
} from "../kernels/conv2d.ts";
import {
  conv3dI8a8Kernel,
  conv3dI8a8Params,
  conv3dI8a8Workgroups,
} from "../kernels/conv3d-i8a8.ts";
import { scoreStorageBytes } from "../kernels/score-storage.ts";
import {
  CALIBRATION_ATTEMPTS,
  calibrateReps,
  RECALIBRATE_BELOW,
  TARGET_PASS_MS,
  WARM_START_MAX_NS,
  WARM_START_MIN_NS,
  WARM_START_PASSES,
  WARM_START_TOLERANCE,
  WARMUP_MIN_RUNS,
  WARMUP_NS,
} from "./measurement.ts";
import {
  type AttentionCase,
  type BmmCase,
  caseFlops,
  caseShape,
  type CaseWeight,
  type Conv2dCase,
  type Conv3dI8a8Case,
  type LinearCase,
  type MatmulCase,
  type SweepCase,
} from "./cases.ts";
import {
  type CandidateSet,
  conv2dCandidate,
  conv2dCandidatesIn,
  gemmCandidate,
  gemmCandidatesIn,
  type GeometryCandidate,
  i8a8Candidate,
  i8a8CandidatesIn,
  i8a8LinearCandidatesIn,
  k88Conv2dCandidates,
  k88GemmCandidates,
} from "./geometries.ts";
import {
  type CaseSummary,
  compareToDefault,
  type DefaultReference,
  type SweepRow,
  type TimingUnit,
} from "./report.ts";

/**
 * The sweep's upper bound of `reps` (protocol rule 1). opbench's `MAX_REPS` (1024; opbench was
 * removed in 0.15) is a cut-off because its output readback and memory grow with the repetitions,
 * but the sweep overdraws the same buffer, so more repetitions add neither readback nor memory. At
 * 15 µs per dispatch, one pass ≈ {@link TARGET_PASS_MS} needs 5,334 repetitions (1024 gives only
 * 15 ms, and the default geometry's re-measurement ratio fluctuated between 0.70 and 1.55); even at
 * 5 µs, 16,000 stays within the bound. opbench's `MAX_REPS` is not changed (its reason for the
 * cut-off still holds).
 */
export const SWEEP_MAX_REPS = 16384;

/**
 * The cut-off of idle runs (protocol rule 3 — the same 64 as opbench's `pinClocks`). Keeps idle
 * runs from going on forever when a short dispatch, with reps capped at the bound, never reaches
 * {@link WARMUP_NS} cumulatively.
 */
export const WARMUP_MAX_PASSES = 64;

/**
 * The pass wall clock that closes the `reps` estimate (half the target length — the module doc's
 * NOTE).
 */
const PROBE_MIN_NS = (TARGET_PASS_MS / 2) * 1e6;

/**
 * The cut-off of a warm start's passes (protocol rule 3): its ceiling of load over the shortest pass
 * it keeps while its reps are under the bound (half the target — a shorter pass is resized), 125. So
 * it binds only on a dispatch whose passes stay short at {@link SWEEP_MAX_REPS}: such a pass costs
 * more time than it loads the device (encoding and submitting 16,384 dispatches), and the bounds of
 * load alone would take far longer than they say.
 */
export const WARM_START_MAX_PASSES = Math.ceil(WARM_START_MAX_NS / PROBE_MIN_NS);

/**
 * One chunk of output readback and input writes (64 MiB — a 1 GiB S is never loaded into JS whole).
 */
export const DIGEST_CHUNK_BYTES = 64 * 1024 * 1024;

/** The chunk for writing inputs (bounds the JS-side peak holding the generated random sequence). */
const FILL_CHUNK_BYTES = 16 * 1024 * 1024;

export type SweepSettings = {
  /** The number of measurement rounds (default: measurement.ts's `ROUNDS`). */
  readonly rounds: number;
  /** The candidate set ({@link CandidateSet} in geometries.ts). */
  readonly candidateSet: CandidateSet;
  /**
   * The candidate geometries to measure, by name, in this order ({@link selectedCandidates} —
   * absent: every candidate of the set). The default geometry's row is measured either way.
   */
  readonly geometries?: readonly string[];
  /** The unit when timestamp-query is enabled (Chrome = `ns`, Deno = `deno-raw-tick`). */
  readonly timestampUnit: "ns" | "deno-raw-tick";
  /**
   * The quick look's margin (protocol rule 4 — measurement.ts's `QUICK_LOOK_MARGIN` by default in
   * the entry point; 0 turns the quick look off and every geometry is timed).
   */
  readonly quickLookMargin: number;
};

export type SweepHooks = {
  /** One progress line (the page's status line, the CLI's standard error). */
  readonly onProgress?: (message: string) => void;
  /** Called each time a row is final (the default geometry's row comes first). */
  readonly onRow?: (row: SweepRow) => void;
  /**
   * Called each time a case is fully measured (after the case's last row, including the default
   * geometry's re-measurement).
   */
  readonly onCase?: (summary: CaseSummary) => void;
  /** Abort (stops once the current geometry is measured — no rows are made for the rest). */
  readonly signal?: AbortSignal;
};

/**
 * The contents of one input (deterministic pseudo-random — the seed is fixed by the resource
 * order).
 */
export type Fill = "f32" | "f16" | "i8" | "scale" | "none";

/**
 * One resource. The name is its identifier within the resource table (the same name across kernels
 * is the same buffer — aligned with the WGSL binding names).
 */
export type ResourceSpec = {
  readonly name: string;
  readonly bytes: number;
  readonly fill: Fill;
};

/** The generated artifacts and sizes of one dispatch. */
export type Launch = {
  readonly key: string;
  readonly wgsl: string;
  readonly params: Uint32Array<ArrayBuffer>;
  /**
   * The params buffer's usage at binding 0 (default `uniform`). `storage` is for the kernels that
   * read their params as `var<storage, read>` (the strided copy family — recipe-builders pass
   * `PARAMS_STORAGE_USAGE` for them); the binding-role check then expects binding 0 among the
   * reads.
   */
  readonly paramsUsage?: "uniform" | "storage";
  /**
   * The names of the resources passed from binding 1 on (a table copying the order of
   * recipe-builders' `binds` — positional binding as in production). `bindings[i]` is binding
   * `i + 1`.
   */
  readonly bindings: readonly string[];
  /**
   * The resources the kernel writes (checked against the WGSL's `read_write` —
   * {@link assertBindingRoles}).
   */
  readonly writes: readonly string[];
  readonly workgroups: readonly [number, number, number];
};

export type CasePlan = {
  readonly resources: readonly ResourceSpec[];
  /**
   * The prelude run once with the default geometry (①QK → ②, which make ③PV's S and row
   * statistics).
   */
  readonly prelude: readonly Launch[];
  readonly output: string;
  readonly defaultCandidate: GeometryCandidate;
  readonly launch: (candidate: GeometryCandidate) => Launch;
};

const F32 = 4;

const gemmOf = (candidate: GeometryCandidate, where: string): GemmGeometry => {
  if (candidate.family !== "gemm") {
    throw new Error(`${where}: not a geometry of the f32 skeleton (${candidate.name})`);
  }
  return candidate.geometry;
};

const i8a8Of = (candidate: GeometryCandidate, where: string): I8a8Geometry => {
  if (candidate.family !== "i8a8") {
    throw new Error(`${where}: not an i8a8 geometry (${candidate.name})`);
  }
  return candidate.geometry;
};

/**
 * The size of a raw weight buffer = the stored byte length rounded up to 4 bytes (the same as the
 * raw-byte classes at src/runtime/session-build.ts:905 — f16 is 2 B / element, i8 is 1 B /
 * element).
 */
const rawWeightBytes = (payloadBytes: number): number =>
  payloadBytes + ((4 - (payloadBytes % 4)) % 4);

/**
 * The weight buffer of an f32-compute case (`n` rows of `k` elements in `weight` storage) and its
 * scale (per channel for i8, per group for i4 — none for f32 / f16): the raw-byte classes of
 * src/runtime/session-build.ts with the scale lengths of format/container/codecs.ts. i4 nibbles take
 * the i8 fill (any bit pattern is a valid nibble).
 */
const weightResources = (
  weight: CaseWeight,
  n: number,
  k: number,
  groupSize: number | undefined,
): ResourceSpec[] => {
  switch (weight) {
    case "f32":
      return [{ name: "w", bytes: n * k * F32, fill: "f32" }];
    case "f16":
      return [{ name: "w", bytes: rawWeightBytes(n * k * 2), fill: "f16" }];
    case "i8":
      return [{ name: "w", bytes: rawWeightBytes(n * k), fill: "i8" }];
    case "i4": {
      if (groupSize === undefined) throw new Error("an i4 case needs its group length");
      return [{ name: "w", bytes: rawWeightBytes(Math.ceil((n * k) / 2)), fill: "i8" }];
    }
  }
};

/** The scale resource of an i8 / i4 weight (bound after the output — LINEAR_SCALE_BINDING). */
const scaleResources = (
  weight: CaseWeight,
  n: number,
  k: number,
  groupSize: number | undefined,
): ResourceSpec[] =>
  weight === "i8"
    ? [{ name: "wscale", bytes: n * F32, fill: "scale" }]
    : weight === "i4"
      ? [{ name: "wscale", bytes: n * Math.ceil(k / (groupSize ?? 1)) * F32, fill: "scale" }]
      : [];

/**
 * f32 linear (f16-stored weights unless the case names another storage — the `k88` cases) — the
 * GEMM path of src/runtime/linear-program.ts (`linearGemmProgram`).
 */
const linearPlan = (sweepCase: LinearCase, limit: number): CasePlan => {
  const { m, n, k, groupSize } = sweepCase;
  const weight = sweepCase.weight ?? "f16";
  const v4 = gemmUsesVec4(k, n);
  const scale = scaleResources(weight, n, k, groupSize);
  return {
    resources: [
      // x[m,k] / W[n,k] / b[n] / out[m,n] (bindings 1..4 = the order of binds, linear.ts:360-364;
      // f32 / f16 have no scale, i8 / i4 bind it at 5)
      { name: "x", bytes: m * k * F32, fill: "f32" },
      ...weightResources(weight, n, k, groupSize),
      { name: "bias", bytes: n * F32, fill: "f32" },
      { name: "out", bytes: m * n * F32, fill: "none" },
      ...scale,
    ],
    prelude: [],
    output: "out",
    // The sweep's default = the default profile's geometry (row-count bucket
    // `gemmGeometryForRows(m)` — the ratio baseline). The geometry a Session actually uses is the
    // adapter's profile (src/kernels/geometry-profile.ts)
    defaultCandidate: gemmCandidate(gemmGeometryForRows(m)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        key: linearKey(weight, v4, "f32", m, groupSize, geometry),
        wgsl: linearWgsl(weight, v4, "f32", m, groupSize, geometry),
        params: linearParams(m, n, k),
        bindings: ["x", "w", "bias", "out", ...scale.map(({ name }) => name)],
        writes: ["out"],
        // linear.ts:365-369
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
          1,
        ],
      };
    },
  };
};

/** f32 matmul — src/runtime/recipe-builders/linear.ts:67-107 (buildMatmul). */
const matmulPlan = (sweepCase: MatmulCase, limit: number): CasePlan => {
  const { m, n, k } = sweepCase;
  // linear.ts:79
  const v4 = gemmUsesVec4(k, n);
  return {
    resources: [
      // a[m,k] / b[k,n] / c[m,n] (bindings 1..3 = in the order binds[0] / binds[1] / outs[0],
      // linear.ts:96-100. The names are the WGSL binding names — denseWgsl in kernels/gemm.ts)
      { name: "a", bytes: m * k * F32, fill: "f32" },
      { name: "b", bytes: k * n * F32, fill: "f32" },
      { name: "c", bytes: m * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "c",
    // The sweep's default = the default profile's geometry (row-count bucket
    // `gemmGeometryForRows(m)` — the ratio baseline).
    // The geometry a Session actually uses is the adapter's profile (gemmRowsGeometry at
    // linear.ts:84)
    defaultCandidate: gemmCandidate(gemmGeometryForRows(m)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        // linear.ts:84-86
        key: matmulKey(v4, m, geometry),
        wgsl: matmulWgsl(v4, m, geometry),
        params: matmulParams(m, n, k),
        bindings: ["a", "b", "c"],
        writes: ["c"],
        // linear.ts:101-105
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
          1,
        ],
      };
    },
  };
};

/** f32 bmm (the batch on the z axis) — src/runtime/recipe-builders/linear.ts:114-155 (buildBmm). */
const bmmPlan = (sweepCase: BmmCase, limit: number): CasePlan => {
  const { batch, m, n, k } = sweepCase;
  // linear.ts:124
  const v4 = gemmUsesVec4(k, n);
  return {
    resources: [
      // a[batch,m,k] / b[batch,k,n] / c[batch,m,n] (bindings 1..3 = linear.ts:143-147)
      { name: "a", bytes: batch * m * k * F32, fill: "f32" },
      { name: "b", bytes: batch * k * n * F32, fill: "f32" },
      { name: "c", bytes: batch * m * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "c",
    // The sweep's default = the default profile's geometry (the bucket of one matrix's m —
    // linear.ts:125-128)
    defaultCandidate: gemmCandidate(gemmGeometryForRows(m)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        // linear.ts:129-133 (no row window)
        key: bmmKey(v4, m, undefined, geometry),
        wgsl: bmmWgsl(v4, m, undefined, geometry),
        params: bmmParams(m, n, k, batch),
        bindings: ["a", "b", "c"],
        writes: ["c"],
        // linear.ts:148-153 (the batch on the z axis, 1 workgroup = 1 batch)
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
          tiledWorkgroups(batch, 1, limit, sweepCase.id),
        ],
      };
    },
  };
};

/**
 * i8a8 linear (i8 per-channel weights) — the ② GEMM of
 * src/runtime/linear-program.ts (`linearI8a8Program`).
 */
const i8a8LinearPlan = (sweepCase: LinearCase, limit: number, dp4a: boolean): CasePlan => {
  const { m, n, k } = sweepCase;
  const v4 = linearI8a8UsesVec4(n);
  return {
    resources: [
      // The quantized activations xq (four i8 packed per word, linear.ts:519) and the per-token
      // scale xs (linear.ts:520). In production quantize_rows makes them — here values of the same
      // format (kernels/quantize-rows.ts: row = token, `row·dim/4` words, ±127) are written
      // directly (① has no geometry, so it is outside the sweep)
      { name: "xq", bytes: m * (k / 4) * 4, fill: "i8" },
      { name: "xscale", bytes: m * F32, fill: "scale" },
      // The i8 weights [n,k] (a raw-byte class) and the per-channel scale [n,1] (groupScaleShape +
      // scaleBytes in format/container/codecs.ts = n·4 B). Bindings: linear.ts:564-575
      { name: "w", bytes: rawWeightBytes(n * k), fill: "i8" },
      { name: "wscale", bytes: n * F32, fill: "scale" },
      { name: "bias", bytes: n * F32, fill: "f32" },
      { name: "out", bytes: m * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "out",
    // The sweep's default = the default profile's i8a8 linear rule for the case's M (the ratio
    // baseline — selected by the same function as the builder, `i8a8LinearGeometry`)
    defaultCandidate: i8a8Candidate(i8a8LinearGeometry(DEFAULT_GEOMETRY_PROFILE, m)),
    launch: (candidate) => {
      // One geometry for the fill form, the key, the WGSL and the grid (i8 weights: the builder's
      // w4a8 group fallback never applies)
      const geometry = i8a8Of(candidate, sweepCase.id);
      // The fill form of the vec4 skeleton, decided as linear.ts decides it (a geometry without the
      // skeleton: the word fill, so its key and WGSL do not move)
      const vecFill = linearI8a8UsesVecFill(geometry, "i8", k);
      return {
        key: linearI8a8Key(v4, dp4a, geometry, "i8", undefined, vecFill),
        wgsl: linearI8a8Wgsl(v4, dp4a, geometry, "i8", undefined, vecFill),
        params: linearI8a8Params(m, n, k),
        bindings: ["xq", "w", "bias", "out", "wscale", "xscale"],
        writes: ["out"],
        // linear.ts:576-580
        workgroups: [
          tiledWorkgroups(n, i8a8TileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, i8a8TileM(geometry), limit, sweepCase.id),
          1,
        ],
      };
    },
  };
};

/**
 * ② Row statistics (the variant matching S's storage format, two reads without regcache — it only
 * makes ③PV's input, so its speed does not matter). params and dispatch are
 * src/runtime/recipe-builders/attention.ts:653-661.
 */
const statsLaunch = (sweepCase: AttentionCase, limit: number): Launch => {
  const rows = sweepCase.batchHeads * sweepCase.m;
  return {
    key: attentionStatsKey("f32", sweepCase.score),
    wgsl: attentionStatsWgsl("f32", sweepCase.score),
    params: attentionStatsParams(rows, sweepCase.n),
    bindings: ["s", "stats"],
    writes: ["stats"],
    workgroups: [gridStrideWorkgroups(rows, 1, limit), 1, 1],
  };
};

/**
 * Resources shared by fused attention (src/runtime/recipe-builders/attention.ts). S =
 * `B·H·M·N·storage width` (scoreBytes at :496, :592), row statistics = `B·H·M·2` words (:593),
 * O = `B·H·M·D` f32.
 */
const attentionShared = (sweepCase: AttentionCase): ResourceSpec[] => {
  const { batchHeads, m, n, d } = sweepCase;
  const scores: ResourceSpec = {
    name: "s",
    bytes: batchHeads * m * n * scoreStorageBytes(sweepCase.score),
    fill: "none",
  };
  if (sweepCase.stage === "qk") return [scores];
  return [
    scores,
    { name: "stats", bytes: batchHeads * m * ATTENTION_STATS_STRIDE * F32, fill: "none" },
    { name: "o", bytes: batchHeads * m * d * F32, fill: "none" },
  ];
};

/** f32 fused attention (S stored in f32) — src/runtime/recipe-builders/attention.ts:425-734. */
const attentionPlan = (sweepCase: AttentionCase, limit: number): CasePlan => {
  const { batchHeads, m, n, d, scale } = sweepCase;
  const qkV4 = gemmUsesVec4(d, n);
  const pvV4 = gemmUsesVec4(n, d);
  const qk = (geometry: GemmGeometry): Launch => ({
    key: attentionQkKey(qkV4, "f32", "f32", false, false, false, geometry),
    wgsl: attentionQkWgsl(qkV4, "f32", "f32", false, false, false, geometry),
    params: attentionQkParams(m, n, d, scale),
    // attention.ts:632-637 (no mask)
    bindings: ["q", "k", "s"],
    writes: ["s"],
    // attention.ts:638-642
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const pv = (geometry: GemmGeometry): Launch => ({
    key: attentionPvKey(pvV4, "f32", "f32", false, false, geometry),
    wgsl: attentionPvWgsl(pvV4, "f32", "f32", false, false, geometry),
    // attention.ts:696-698 (m = M, n = D, k = N)
    params: attentionPvParams(m, d, n),
    // attention.ts:700-705
    bindings: ["s", "v", "stats", "o"],
    writes: ["o"],
    // attention.ts:706-710
    workgroups: [
      tiledWorkgroups(d, gemmTileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const stagePv = sweepCase.stage === "pv";
  // The sweep's default = the default profile's geometry (`defaultGemmGeometry()` for both ①QK and
  // ③PV — the ratio baseline). The geometry a Session actually uses is the adapter's profile
  // (src/kernels/geometry-profile.ts)
  const fallback = defaultGemmGeometry();
  return {
    resources: [
      // q[B·H,M,D] / k[B·H,N,D] / v[B·H,N,D] (the rank-4 head-first inputs folded into the batch
      // axis)
      { name: "q", bytes: batchHeads * m * d * F32, fill: "f32" },
      { name: "k", bytes: batchHeads * n * d * F32, fill: "f32" },
      ...(stagePv ? [{ name: "v", bytes: batchHeads * n * d * F32, fill: "f32" as const }] : []),
      ...attentionShared(sweepCase),
    ],
    prelude: stagePv ? [qk(fallback), statsLaunch(sweepCase, limit)] : [],
    output: stagePv ? "o" : "s",
    defaultCandidate: gemmCandidate(fallback),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return stagePv ? pv(geometry) : qk(geometry);
    },
  };
};

/**
 * i8a8 fused attention (S stored in f16 = s16) — src/runtime/recipe-builders/attention.ts:425-734
 * and the preceding allocations :1276-1279 (q / k quantization) and :1376-1377 (Vᵀ quantization).
 */
const i8a8AttentionPlan = (sweepCase: AttentionCase, limit: number, dp4a: boolean): CasePlan => {
  const { batchHeads, m, n, d, scale, score } = sweepCase;
  const qkV4 = attentionQkI8a8UsesVec4(n);
  const pvV4 = attentionPvI8a8UsesVec4(d);
  const qk = (geometry: I8a8Geometry): Launch => ({
    key: attentionQkI8a8Key(qkV4, dp4a, score, geometry),
    wgsl: attentionQkI8a8Wgsl(qkV4, dp4a, score, geometry),
    params: attentionQkI8a8Params(m, n, d, scale),
    // attention.ts:609-615
    bindings: ["qq", "kq", "s", "qscale", "kscale"],
    writes: ["s"],
    // attention.ts:616-620
    workgroups: [
      tiledWorkgroups(n, i8a8TileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, i8a8TileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const pv = (geometry: I8a8Geometry): Launch => ({
    key: attentionPvI8a8Key(pvV4, dp4a, score, geometry),
    wgsl: attentionPvI8a8Wgsl(pvV4, dp4a, score, geometry),
    // attention.ts:673-675 (m = M, n = D, k = N)
    params: attentionPvI8a8Params(m, d, n),
    // attention.ts:677-683
    bindings: ["s", "vq", "stats", "o", "vscale"],
    writes: ["o"],
    // attention.ts:684-688
    workgroups: [
      tiledWorkgroups(d, i8a8TileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, i8a8TileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const stagePv = sweepCase.stage === "pv";
  // The sweep's default = the default profile's i8a8 attention geometry (separate for ①QK / ③PV —
  // the ratio baseline)
  const qkDefault = defaultI8a8Geometry("attention_qk");
  const pvDefault = defaultI8a8Geometry("attention_pv");
  return {
    resources: [
      // qq / kq pack four i8 per word (B·H·M·D / B·H·N·D bytes), qs / ks are per-token scales
      // (:1276-1279)
      { name: "qq", bytes: batchHeads * m * d, fill: "i8" },
      { name: "qscale", bytes: batchHeads * m * F32, fill: "scale" },
      { name: "kq", bytes: batchHeads * n * d, fill: "i8" },
      { name: "kscale", bytes: batchHeads * n * F32, fill: "scale" },
      ...(stagePv
        ? [
            // Per-column quantization of Vᵀ[B·H,D,N] (row = (b,h,d), N contiguous — :1376-1377)
            { name: "vq", bytes: batchHeads * d * n, fill: "i8" as const },
            { name: "vscale", bytes: batchHeads * d * F32, fill: "scale" as const },
          ]
        : []),
      ...attentionShared(sweepCase),
    ],
    prelude: stagePv ? [qk(qkDefault), statsLaunch(sweepCase, limit)] : [],
    output: stagePv ? "o" : "s",
    defaultCandidate: i8a8Candidate(stagePv ? pvDefault : qkDefault),
    launch: (candidate) => {
      const geometry = i8a8Of(candidate, sweepCase.id);
      return stagePv ? pv(geometry) : qk(geometry);
    },
  };
};

/**
 * conv2d's implicit GEMM (f16-stored weights and B = 1 unless the case names others — the `k88`
 * cases) — src/runtime/recipe-builders/conv.ts:381-424.
 */
const conv2dPlan = (sweepCase: Conv2dCase, limit: number): CasePlan => {
  const { channelsIn, channelsOut, height, width } = sweepCase;
  const batch = sweepCase.batch ?? 1;
  const weight = sweepCase.weight ?? "f16";
  const dims: Conv2dDims = {
    batch,
    channelsIn,
    channelsOut,
    heightIn: height,
    widthIn: width,
    heightOut: height,
    widthOut: width,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    groups: 1,
  };
  // conv.ts:390-394 (M = Cout, N = Hout·Wout, K = Cin·Kh·Kw)
  const n = height * width;
  const kFlat = channelsIn * 9;
  const v4 = conv2dUsesVec4(kFlat, width, 1);
  const mTile = conv2dIgemmMTile(channelsOut);
  const scale = scaleResources(weight, channelsOut, kFlat, undefined);
  return {
    resources: [
      // x[B,Cin,H,W] / W[Cout,Cin,3,3] / b[Cout] / out[B,Cout,H,W] (bindings 1..4 =
      // conv.ts:413-417; the i8 scale at CONV2D_SCALE_BINDING = 5)
      { name: "x", bytes: batch * channelsIn * n * F32, fill: "f32" },
      ...weightResources(weight, channelsOut, kFlat, undefined),
      { name: "bias", bytes: channelsOut * F32, fill: "f32" },
      { name: "out", bytes: batch * channelsOut * n * F32, fill: "none" },
      ...scale,
    ],
    prelude: [],
    output: "out",
    // The sweep's default = the default profile's conv2d geometry (per m-tile class of 64 / 32 rows
    // — the ratio baseline)
    defaultCandidate: conv2dCandidate(gemmMTileGeometry(mTile)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        key: conv2dIgemmKey(weight, v4, mTile, geometry),
        wgsl: conv2dIgemmWgsl(weight, v4, mTile, geometry),
        params: conv2dIgemmParams(dims),
        bindings: ["x", "w", "bias", "out", ...scale.map(({ name }) => name)],
        writes: ["out"],
        // conv.ts:418-422
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(channelsOut, gemmTileM(geometry), limit, sweepCase.id),
          tiledWorkgroups(dims.batch, 1, limit, sweepCase.id),
        ],
      };
    },
  };
};

/**
 * conv3d's w8a8 implicit GEMM (i8 tap-major weights, per-channel scale) — the ② GEMM of
 * src/runtime/recipe-builders/conv.ts buildConv3dI8a8.
 */
const i8a8Conv3dPlan = (sweepCase: Conv3dI8a8Case, limit: number, dp4a: boolean): CasePlan => {
  const { dims } = sweepCase;
  const pixelsIn = dims.timeIn * dims.heightIn * dims.widthIn;
  const pixelsOut = dims.timeOut * dims.heightOut * dims.widthOut;
  const taps = dims.kernelT * dims.kernelH * dims.kernelW;
  return {
    resources: [
      // The per-pixel quantised activations xq ([P_in][Cin/4] words, four i8 packed per word) and
      // the per-pixel scale xs (the temporaries of conv.ts buildConv3dI8a8). In production the
      // channel-major quantize_rows makes them — here values of the same format (±127) are written
      // directly (① has no geometry, so it is outside the sweep)
      { name: "xq", bytes: pixelsIn * (dims.channelsIn / 4) * 4, fill: "i8" },
      { name: "xscale", bytes: pixelsIn * F32, fill: "scale" },
      // The i8 tap-major weights [Cout][taps][Cin] (a raw-byte class — the order inside a row does
      // not change the timing or the bits) and the per-channel scale [Cout]
      { name: "w", bytes: rawWeightBytes(dims.channelsOut * taps * dims.channelsIn), fill: "i8" },
      { name: "wscale", bytes: dims.channelsOut * F32, fill: "scale" },
      { name: "bias", bytes: dims.channelsOut * F32, fill: "f32" },
      { name: "out", bytes: dims.channelsOut * pixelsOut * F32, fill: "none" },
    ],
    prelude: [],
    output: "out",
    // The sweep's default = the default profile's conv3d geometry (the ratio baseline)
    defaultCandidate: i8a8Candidate(defaultI8a8Geometry("conv3d")),
    launch: (candidate) => {
      const geometry = i8a8Of(candidate, sweepCase.id);
      // The kernel conv.ts buildConv3dI8a8 dispatches (store variant, key and WGSL from one place)
      const { key, wgsl } = conv3dI8a8Kernel(dims, dp4a, geometry);
      return {
        key,
        wgsl,
        params: conv3dI8a8Params(dims),
        // conv.ts buildConv3dI8a8: xq, w, bias, out, wscale (CONV3D_SCALE_BINDING), xs
        // (CONV3D_ACT_SCALE_BINDING)
        bindings: ["xq", "w", "bias", "out", "wscale", "xscale"],
        writes: ["out"],
        workgroups: conv3dI8a8Workgroups(dims, geometry, limit, sweepCase.id),
      };
    },
  };
};

/**
 * A case's resource table, prelude and per-candidate dispatch (the tests take the resource order
 * and contents from here).
 */
export const casePlan = (sweepCase: SweepCase, limit: number, dp4a: boolean): CasePlan => {
  switch (sweepCase.op) {
    case "linear":
      return linearPlan(sweepCase, limit);
    case "matmul":
      return matmulPlan(sweepCase, limit);
    case "bmm":
      return bmmPlan(sweepCase, limit);
    case "i8a8-linear":
      return i8a8LinearPlan(sweepCase, limit, dp4a);
    case "attention":
      return attentionPlan(sweepCase, limit);
    case "i8a8-attention":
      return i8a8AttentionPlan(sweepCase, limit, dp4a);
    case "conv2d":
      return conv2dPlan(sweepCase, limit);
    case "i8a8-conv3d":
      return i8a8Conv3dPlan(sweepCase, limit, dp4a);
  }
};

/**
 * The candidates per op family (the set `set`). Even when the default geometry is not included,
 * {@link sweepCase} adds it at the front.
 */
export const candidatesFor = (sweepCase: SweepCase, set: CandidateSet): GeometryCandidate[] => {
  switch (sweepCase.op) {
    case "linear":
    case "matmul":
    case "bmm":
      // The evaluation set k88 measures the geometries the tables give this row count
      return set === "k88" ? k88GemmCandidates(sweepCase.m) : gemmCandidatesIn(set);
    case "attention":
      return gemmCandidatesIn(set);
    case "conv2d":
      return set === "k88" ? k88Conv2dCandidates(sweepCase.channelsOut) : conv2dCandidatesIn(set);
    case "i8a8-linear":
      // `full` gives the i8a8 linear its own grid (the skeleton and the K64 tiles included)
      return i8a8LinearCandidatesIn(set);
    case "i8a8-attention":
    case "i8a8-conv3d":
      return i8a8CandidatesIn(set);
  }
};

/**
 * The candidates one case measures: every candidate of `set` ({@link candidatesFor}) when
 * `geometries` is absent, else those geometries by name, in the order given (ADR 0117 addendum
 * 2026-11-12, "the `geometries` option"). A name that is not a candidate of the case throws, and
 * so does the case's default geometry — the sweep entry point checks every planned case before it
 * touches the GPU. The default geometry is not chosen here: {@link sweepCase} always measures it
 * first, as the default row.
 */
export const selectedCandidates = (
  sweepCase: SweepCase,
  set: CandidateSet,
  geometries: readonly string[] | undefined,
): GeometryCandidate[] => {
  const candidates = candidatesFor(sweepCase, set);
  if (geometries === undefined) return candidates;
  // MUST: refuse the default's name. It is measured only as the default row (sweepCase drops it
  // from the candidates), so `settings.geometries` would list as a measured candidate a geometry
  // that has no candidate row. The name is read from the case plan the sweep runs, which builds no
  // GPU object; the workgroup limit and dp4a do not enter any plan's default candidate, so the
  // check needs no device
  const defaultName = casePlan(sweepCase, Number.POSITIVE_INFINITY, false).defaultCandidate.name;
  if (geometries.includes(defaultName)) {
    throw new Error(
      `Geometry ${defaultName} is the default geometry of case ${sweepCase.id}: its row is always measured, so do not list it in geometries`,
    );
  }
  return geometries.map((name) => {
    const found = candidates.find((candidate) => candidate.name === name);
    if (found === undefined) {
      throw new Error(
        `Geometry ${name} is not a candidate of case ${sweepCase.id} in candidate set ${set}`,
      );
    }
    return found;
  });
};

const sortedList = (values: Iterable<number>): string =>
  [...values].sort((a, b) => a - b).join(", ");

/**
 * Checks that the copied binding table ({@link Launch}'s `bindings` / `writes`) fits the WGSL's
 * storage roles (PipelineCache's `roles`): the storage bindings line up exactly at binding 1..n,
 * and the positions of the write targets (`read_write`) match the positions of the table's
 * `writes`. A disagreement signals that the source kernel or the order in recipe-builders changed,
 * so the row fails instead of binding by guesswork.
 */
export const assertBindingRoles = (
  launch: Pick<Launch, "key" | "bindings" | "writes" | "paramsUsage">,
  roles: StorageRoles,
): void => {
  const declared = sortedList([...roles.reads, ...roles.writes]);
  const expected = sortedList([
    ...(launch.paramsUsage === "storage" ? [0] : []),
    ...launch.bindings.map((_, index) => index + 1),
  ]);
  if (declared !== expected) {
    throw new Error(
      `${launch.key}: the WGSL storage bindings [${declared}] differ from the positions in the mirrored table [${expected}] (${launch.bindings.join(
        ", ",
      )})`,
    );
  }
  const expectedWrites = sortedList(
    launch.bindings.flatMap((name, index) => (launch.writes.includes(name) ? [index + 1] : [])),
  );
  const declaredWrites = sortedList(roles.writes);
  if (declaredWrites !== expectedWrites) {
    throw new Error(
      `${launch.key}: the WGSL write bindings [${declaredWrites}] differ from the write targets in the mirrored table [${expectedWrites}] (${launch.writes.join(
        ", ",
      )})`,
    );
  }
};

/** xorshift32 (deterministic, seed ≠ 0). */
const xorshift = (seed: number): (() => number) => {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state ^ (state << 13)) >>> 0;
    state = (state ^ (state >>> 17)) >>> 0;
    state = (state ^ (state << 5)) >>> 0;
    return state;
  };
};

const TWO_POW_32 = 2 ** 32;

/** Makes one word's value in the `fill` format (returned as u32 bits). */
const wordFiller = (fill: Exclude<Fill, "none">, next: () => number): (() => number) => {
  const bits = new Float32Array(1);
  const word = new Uint32Array(bits.buffer);
  switch (fill) {
    case "f32":
      // f32 in [-1, 1)
      return () => {
        bits[0] = (next() / TWO_POW_32) * 2 - 1;
        return word[0];
      };
    case "scale":
      // Positive scales (near the value that maps |x| ≤ 1 onto ±127 — [0.5, 1.5) / 127)
      return () => {
        bits[0] = (0.5 + next() / TWO_POW_32) / 127;
        return word[0];
      };
    case "f16": {
      // Packs two f16 normal numbers (|v| ∈ [2^-4, 1), random sign) (low half = even index —
      // `unpack2x16float(w[i >> 1])[i & 1]` in kernels/weight-storage.ts)
      const half = (): number => {
        const r = next();
        return (((r >>> 12) & 1) << 15) | ((11 + (r & 3)) << 10) | ((r >>> 2) & 0x3ff);
      };
      return () => (half() | (half() << 16)) >>> 0;
    }
    case "i8":
      // Packs four i8 (lane j = byte j — the same order as `pack4xI8` / `unpack4xI8`). Values in
      // [-127, 127] (the quantization grid — the ±127 of kernels/quantize-rows.ts)
      return () => {
        const r = next();
        let packed = 0;
        for (let lane = 0; lane < 4; lane += 1) {
          const byte = (r >>> (lane * 8)) & 0xff;
          packed |= ((byte === 255 ? 0 : byte - 127) & 0xff) << (lane * 8);
        }
        return packed >>> 0;
      };
  }
};

/**
 * The sequence of u32 words that fills entry `index` of the resource table (deterministic — the
 * single path shared by {@link createResources} and the tests' CPU reference).
 */
export const resourceWordStream = (fill: Exclude<Fill, "none">, index: number): (() => number) =>
  wordFiller(fill, xorshift(0x9e3779b9 ^ (index + 1)));

/**
 * The device-related bundle used for measuring (one sweep's worth — cleaned up by
 * {@link destroySweepContext}).
 */
export type SweepContext = {
  readonly gpu: GpuContext;
  readonly unit: TimingUnit;
  /**
   * The wall clock in milliseconds: `performance.now` unless the caller gave another (a test's, so
   * that a pass takes the time the test states and no timer's lateness enters a calibration).
   */
  readonly now: () => number;
  readonly dp4a: boolean;
  /** Output readback (in {@link DIGEST_CHUNK_BYTES} chunks). */
  readonly staging: GPUBuffer;
  /** The pass's begin / end timestamps (absent on a device without timestamp-query). */
  readonly timestamps?: {
    readonly querySet: GPUQuerySet;
    readonly resolve: GPUBuffer;
    readonly read: GPUBuffer;
  };
};

/**
 * Runs `body` synchronously wrapped in errorScopes (validation + out-of-memory). The section is
 * inside the per-device scope lock (src/gpu/context.ts "Invariants of errorScope sections").
 */
const scoped = <T>(gpu: GpuContext, label: string, body: () => T): Promise<T> =>
  gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
    pushFailureScopes(gpu.device);
    let value: T;
    try {
      value = body();
    } catch (cause) {
      await discardFailureScopes(gpu.device);
      throw cause;
    }
    const failure = await popFailureScopes(gpu.device, label);
    if (failure !== undefined) throw failure;
    return value;
  });

/**
 * Submits one command buffer, waits for completion and returns the submit-to-completion wall clock
 * (ns, on the context's clock). The errorScope pop is issued in the same synchronous section as the submit and awaited
 * alongside the completion wait (the pop's round trip is not added serially to the wall clock).
 */
const submitAndWait = (
  { gpu, now }: Pick<SweepContext, "gpu" | "now">,
  label: string,
  encode: (encoder: GPUCommandEncoder) => void,
): Promise<number> =>
  gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
    const { device } = gpu;
    pushFailureScopes(device);
    let started: number;
    try {
      const encoder = device.createCommandEncoder({ label });
      encode(encoder);
      const commands = encoder.finish();
      started = now();
      device.queue.submit([commands]);
    } catch (cause) {
      await discardFailureScopes(device);
      throw cause;
    }
    const failure = popFailureScopes(device, label);
    const finished = gpu[RUNTIME_INTERNAL].raceDeviceLost(
      device.queue.onSubmittedWorkDone().then(now),
      label,
    );
    const [error, doneAt] = await Promise.all([failure, finished]);
    if (error !== undefined) throw error;
    return (doneAt - started) * 1e6;
  });

/**
 * Creates the sweep context (one timestamp query set and one readback staging buffer). The unit is
 * `timestampUnit` when timestamp-query is enabled, `wall` otherwise. `now` is the wall clock in
 * milliseconds (`performance.now` by default).
 */
export const createSweepContext = async (
  gpu: GpuContext,
  timestampUnit: SweepSettings["timestampUnit"],
  now: () => number = () => performance.now(),
): Promise<SweepContext> => {
  const { device } = gpu;
  const unit: TimingUnit = gpu.gpuTimingEnabled ? timestampUnit : "wall";
  const stagingBytes = Math.min(DIGEST_CHUNK_BYTES, Math.floor(gpu.limits.maxBufferSize / 8) * 8);
  const created = await scoped(gpu, "geometry-sweep context", () => ({
    staging: device.createBuffer({
      label: "geometry-sweep-staging",
      size: stagingBytes,
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    }),
    timestamps: gpu.gpuTimingEnabled
      ? {
          querySet: device.createQuerySet({ type: "timestamp", count: 2 }),
          resolve: device.createBuffer({
            label: "geometry-sweep-resolve",
            size: 16,
            usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
          }),
          read: device.createBuffer({
            label: "geometry-sweep-timestamps",
            size: 16,
            usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
          }),
        }
      : undefined,
  }));
  return {
    gpu,
    unit,
    now,
    // The variants of i8a8's integer dot product have identical numerics (kernels/linear-i8a8.ts) —
    // the choice is the same as production linear's, "the list of WGSL language features"
    // (src/runtime/session-build.ts:610 / :1047). Production attention chooses by a canary
    // (gpu/attention-dp4a-canary.ts), but the sweep only looks at speed, so it does not run it.
    dp4a: dp4aAvailable(gpu.wgslLanguageFeatures),
    staging: created.staging,
    ...(created.timestamps === undefined ? {} : { timestamps: created.timestamps }),
  };
};

/** Releases the context's resources (the device belongs to the caller — not destroyed here). */
export const destroySweepContext = (context: SweepContext): void => {
  context.staging.destroy();
  context.timestamps?.querySet.destroy();
  context.timestamps?.resolve.destroy();
  context.timestamps?.read.destroy();
};

/**
 * Allocates the resources and fills them with deterministic values (on failure, releases what was
 * allocated, then throws).
 */
export const createResources = async (
  context: SweepContext,
  specs: readonly ResourceSpec[],
): Promise<Map<string, GPUBuffer>> => {
  const { gpu } = context;
  const buffers = new Map<string, GPUBuffer>();
  try {
    for (const [index, spec] of specs.entries()) {
      const size = Math.max(4, Math.ceil(spec.bytes / 4) * 4);
      if (size > gpu.limits.maxStorageBufferBindingSize) {
        throw new Error(
          `${spec.name}: ${size} B exceeds maxStorageBufferBindingSize ${gpu.limits.maxStorageBufferBindingSize}`,
        );
      }
      const buffer = await scoped(gpu, `createBuffer(${spec.name})`, () =>
        gpu.device.createBuffer({
          label: `geometry-sweep-${spec.name}`,
          size,
          usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC,
        }),
      );
      buffers.set(spec.name, buffer);
      if (spec.fill === "none") continue;
      const word = resourceWordStream(spec.fill, index);
      for (let offset = 0; offset < size; offset += FILL_CHUNK_BYTES) {
        const chunk = new Uint32Array(Math.min(FILL_CHUNK_BYTES, size - offset) / 4);
        for (let at = 0; at < chunk.length; at += 1) chunk[at] = word();
        await scoped(gpu, `writeBuffer(${spec.name})`, () =>
          gpu.device.queue.writeBuffer(buffer, offset, chunk),
        );
      }
    }
  } catch (cause) {
    for (const buffer of buffers.values()) buffer.destroy();
    throw cause;
  }
  return buffers;
};

/** A dispatch resolved down to pipeline, params and bind group. */
export type ReadyLaunch = {
  readonly launch: Launch;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly params: GPUBuffer;
};

export const prepareLaunch = async (
  context: SweepContext,
  launch: Launch,
  resources: ReadonlyMap<string, GPUBuffer>,
): Promise<ReadyLaunch> => {
  const { gpu } = context;
  // The pipeline comes from the device-lifetime cache (a different WGSL for the same key raises
  // PipelineKeyConflictError — which also detects an explicit-geometry key losing its
  // discriminating power)
  const { pipeline, layout, roles } = await gpu[RUNTIME_INTERNAL].withScopeLock(() =>
    gpu[RUNTIME_INTERNAL].pipelines().get(launch.key, launch.wgsl),
  );
  assertBindingRoles(launch, roles);
  const params = await scoped(gpu, `params(${launch.key})`, () => {
    const buffer = gpu.device.createBuffer({
      label: `geometry-sweep-params`,
      size: Math.max(16, launch.params.byteLength),
      usage:
        (launch.paramsUsage === "storage" ? BUFFER_USAGE.STORAGE : BUFFER_USAGE.UNIFORM) |
        BUFFER_USAGE.COPY_DST,
    });
    gpu.device.queue.writeBuffer(buffer, 0, launch.params);
    return buffer;
  });
  try {
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: params } }];
    for (const [index, name] of launch.bindings.entries()) {
      const buffer = resources.get(name);
      if (buffer === undefined) throw new Error(`${launch.key}: resource ${name} is missing`);
      entries.push({ binding: index + 1, resource: { buffer } });
    }
    const bindGroup = await scoped(gpu, `createBindGroup(${launch.key})`, () =>
      gpu.device.createBindGroup({ layout, entries }),
    );
    return { launch, pipeline, bindGroup, params };
  } catch (cause) {
    params.destroy();
    throw cause;
  }
};

/** One pass's times. */
export type PassSample = {
  readonly wallNs: number;
  /** The timestamp difference (unit = the context's unit). Absent on passes that are not timed. */
  readonly gpu?: number;
  readonly clamped: boolean;
};

/**
 * Runs `reps` copies of the same dispatch in one compute pass. With `timed`, writes timestamps at
 * the pass's begin / end and returns the difference. With `clear`, resets that buffer to 0 before
 * the pass.
 */
const runPass = async (
  context: SweepContext,
  ready: ReadyLaunch,
  reps: number,
  timed: boolean,
  clear?: GPUBuffer,
): Promise<PassSample> => {
  const timestamps = timed ? context.timestamps : undefined;
  const [x, y, z] = ready.launch.workgroups;
  const wallNs = await submitAndWait(context, ready.launch.key, (encoder) => {
    if (clear !== undefined) encoder.clearBuffer(clear);
    const pass = encoder.beginComputePass(
      timestamps === undefined
        ? {}
        : {
            timestampWrites: {
              querySet: timestamps.querySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          },
    );
    pass.setPipeline(ready.pipeline);
    pass.setBindGroup(0, ready.bindGroup);
    for (let rep = 0; rep < reps; rep += 1) pass.dispatchWorkgroups(x, y, z);
    pass.end();
    if (timestamps !== undefined) {
      encoder.resolveQuerySet(timestamps.querySet, 0, 2, timestamps.resolve, 0);
      encoder.copyBufferToBuffer(timestamps.resolve, 0, timestamps.read, 0, 16);
    }
  });
  if (timestamps === undefined) return { wallNs, clamped: false };
  await context.gpu[RUNTIME_INTERNAL].raceDeviceLost(
    timestamps.read.mapAsync(MAP_MODE.READ),
    "timestamp readback",
  );
  const stamps = new BigUint64Array(timestamps.read.getMappedRange());
  const delta = stamps[1] - stamps[0];
  timestamps.read.unmap();
  // MUST: clamp negatives to 0 and keep the count (no silent dropping — same as ADR 0021). The
  // caller (timeLaunch) excludes them from the min candidates
  return { wallNs, gpu: delta < 0n ? 0 : Number(delta), clamped: delta < 0n };
};

const hex = (bytes: ArrayBuffer): string =>
  Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("");

/**
 * The output digest = **the SHA-256 of the concatenated SHA-256s of each 64 MiB chunk** (two
 * levels). S reaches 1 GiB, so each chunk is copied to staging and read, keeping the JS-side peak
 * at one chunk. Same output ⇔ same digest, so for judging a match between geometries it is as
 * strong as a plain SHA-256.
 */
export const outputDigest = async (context: SweepContext, buffer: GPUBuffer): Promise<string> => {
  const { gpu, staging } = context;
  const chunks: ArrayBuffer[] = [];
  for (let offset = 0; offset < buffer.size; offset += staging.size) {
    const length = Math.min(staging.size, buffer.size - offset);
    await submitAndWait(context, "output readback", (encoder) => {
      encoder.copyBufferToBuffer(buffer, offset, staging, 0, length);
    });
    await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      staging.mapAsync(MAP_MODE.READ, 0, length),
      "output readback",
    );
    // digest copies its input at call time (as WebCrypto specifies), but to stay independent of
    // implementation differences, the data is copied to the JS side before unmap and then passed
    const copy = new Uint8Array(staging.getMappedRange(0, length)).slice();
    staging.unmap();
    chunks.push(await crypto.subtle.digest("SHA-256", copy));
  }
  return await joinChunkDigests(chunks);
};

/**
 * Concatenates the per-chunk SHA-256s and takes their SHA-256 (the second level of
 * {@link outputDigest} — the tests use it too).
 */
export const joinChunkDigests = async (chunks: readonly ArrayBuffer[]): Promise<string> => {
  const joined = new Uint8Array(chunks.length * 32);
  chunks.forEach((chunk, index) => joined.set(new Uint8Array(chunk), index * 32));
  return hex(await crypto.subtle.digest("SHA-256", joined));
};

const describeCause = (cause: unknown): string =>
  cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);

/** The row of a failed pair (the sweep does not stop). */
const errorRow = (
  sweepCase: SweepCase,
  candidate: GeometryCandidate,
  isDefault: boolean,
  cause: unknown,
): SweepRow => ({
  caseId: sweepCase.id,
  op: sweepCase.op,
  shape: caseShape(sweepCase),
  censusCount: sweepCase.censusCount,
  geometry: candidate.name,
  geometryParams: candidate.geometry,
  isDefault,
  error: describeCause(cause),
});

/**
 * Idle runs (protocol rule 3). The cumulative time counts the pass's GPU time for unit `ns` and
 * wall clock otherwise (Deno's raw ticks are not ns, so they cannot be compared with the lower
 * bound in ns).
 */
const warmUp = async (context: SweepContext, ready: ReadyLaunch, reps: number): Promise<void> => {
  const timed = context.unit === "ns";
  let total = 0;
  for (let pass = 0; pass < WARMUP_MAX_PASSES; pass += 1) {
    if (pass >= WARMUP_MIN_RUNS && total >= WARMUP_NS) return;
    const sample = await runPass(context, ready, reps, timed);
    total += timed ? (sample.gpu ?? 0) : sample.wallNs;
  }
};

/** The time record of one geometry (the measurement fields of a row). */
export type Timing = {
  readonly reps: number;
  readonly rounds: readonly number[];
  readonly wallRounds: readonly number[];
  readonly clampedNegativeSamples: number;
  /** The min of the non-negative rounds ÷ reps (absent if every round is negative). */
  readonly perDispatch?: number;
  readonly wallNsPerDispatch: number;
};

/** A probe's outcome: the reps that fill the target, and its last pass with that pass's dispatches. */
type Probe = {
  readonly reps: number;
  readonly last: { readonly dispatches: number; readonly sample: PassSample };
};

/**
 * Estimates the `reps` that make one pass of `ready` about {@link TARGET_PASS_MS} (the module doc's
 * NOTE — double by wall clock and take the slope of the last two points). With `timed`, its passes
 * also take timestamps (the quick look reads its last one); the estimate reads the wall clock either
 * way.
 */
const probe = async (context: SweepContext, ready: ReadyLaunch, timed: boolean): Promise<Probe> => {
  let reps = 1;
  let previous: { readonly reps: number; readonly wallNs: number } | undefined;
  for (;;) {
    const sample = await runPass(context, ready, reps, timed);
    const { wallNs } = sample;
    if (wallNs >= PROBE_MIN_NS || reps >= SWEEP_MAX_REPS) {
      const average = wallNs / reps;
      const slope =
        previous === undefined ? 0 : (wallNs - previous.wallNs) / (reps - previous.reps);
      // Without a usable slope (reached on the first point / negative from noise), fall back to the
      // average. A quarter of the average is the floor, so that a slope flattened by noise does
      // not push the pass beyond 4 times the target
      const probeNs = slope > 0 ? Math.max(slope, average / 4) : average;
      return {
        reps: calibrateReps(probeNs, TARGET_PASS_MS, SWEEP_MAX_REPS),
        last: { dispatches: reps, sample },
      };
    }
    previous = { reps, wallNs };
    reps = Math.min(SWEEP_MAX_REPS, reps * 2);
  }
};

/** The reps of a probe without timestamps (the calibration's and the warm start's). */
const probeReps = async (context: SweepContext, ready: ReadyLaunch): Promise<number> =>
  (await probe(context, ready, false)).reps;

/** What a kernel's quick look read (protocol rule 4). */
export type QuickLook = {
  /** The reps the probe gives for the target pass: the calibration goes on from them. */
  readonly reps: number;
  /**
   * The probe's last pass ÷ its dispatches (the context's unit: its timestamp difference, or its
   * wall clock under `wall`); absent when that pass's difference was negative.
   */
  readonly perDispatch?: number;
};

/**
 * The quick look at one kernel (protocol rule 4): the calibration's probe with timestamps on its
 * passes, its last pass — at least half the target long by the wall clock, or at the reps bound —
 * read as the kernel's time. A kernel that goes on calibrates from the probe's reps
 * ({@link calibrate}), so it pays for no second probe.
 */
export const quickLook = async (context: SweepContext, ready: ReadyLaunch): Promise<QuickLook> => {
  const { reps, last } = await probe(context, ready, context.unit !== "wall");
  const { sample, dispatches } = last;
  return sample.clamped
    ? { reps }
    : { reps, perDispatch: (sample.gpu ?? sample.wallNs) / dispatches };
};

/** One kernel the quick look has read, as {@link quickLookGoesOn} weighs it. */
export type QuickLookKernel = {
  readonly name: string;
  /** The kernels of one class are weighed against one bar. */
  readonly class: string;
  /** Its quick look's time per dispatch (absent: nothing to screen it on — it goes on). */
  readonly perDispatch?: number;
  /** A reference of the group (the baseline, a default's choice, the default geometry): it always goes on. */
  readonly reference: boolean;
};

/**
 * What the margin multiplies for a class: the fastest quick time of the class (the derivation picks
 * a winner among them), or the quick time of the kernel named (the derivation asks only whether a
 * kernel clears a threshold over that reference; the fastest when the reference has no quick time).
 */
export type QuickLookBar =
  | { readonly against: "fastest" }
  | { readonly against: "reference"; readonly reference: string };

/**
 * Which kernels go on after the quick look (protocol rule 4), in the order given: every reference,
 * every kernel without a quick time, and every kernel whose quick time is at most `margin` times its
 * class's bar (`barOf`). A margin of 0 lets every kernel go on (the quick look is off). Pure.
 */
export const quickLookGoesOn = (
  kernels: readonly QuickLookKernel[],
  margin: number,
  barOf: (kernelClass: string) => QuickLookBar,
): boolean[] => {
  if (margin === 0) return kernels.map(() => true);
  const fastest = new Map<string, number>();
  for (const kernel of kernels) {
    if (kernel.perDispatch === undefined) continue;
    const best = fastest.get(kernel.class);
    if (best === undefined || kernel.perDispatch < best) {
      fastest.set(kernel.class, kernel.perDispatch);
    }
  }
  return kernels.map((kernel) => {
    if (kernel.reference || kernel.perDispatch === undefined) return true;
    const bar = barOf(kernel.class);
    const reference =
      bar.against === "reference"
        ? kernels.find((other) => other.name === bar.reference)?.perDispatch
        : undefined;
    const against = reference ?? fastest.get(kernel.class);
    return against === undefined || kernel.perDispatch <= margin * against;
  });
};

/** FNV-1a 32 bit of a string's UTF-16 code units (a seed; no fingerprint). */
const seedOf = (label: string): number => {
  let hash = 0x811c9dc5;
  for (let index = 0; index < label.length; index += 1) {
    hash = Math.imul(hash ^ label.charCodeAt(index), 0x01000193) >>> 0;
  }
  return hash;
};

/**
 * The order of a group's quick looks (protocol rule 4): the items shuffled by a generator seeded
 * from the group's `label`, so a kernel's place does not follow the plan's order (the baseline first,
 * one class after the other) — a device whose state moves during the quick looks does not favour
 * the kernels of one class — and the same group takes the same order in every run.
 */
export const quickLookOrder = <T>(items: readonly T[], label: string): T[] => {
  const next = xorshift(seedOf(label));
  const order = [...items];
  for (let index = order.length - 1; index > 0; index -= 1) {
    const other = next() % (index + 1);
    [order[index], order[other]] = [order[other], order[index]];
  }
  return order;
};

/**
 * The time a pass kept the device loaded (ns): its timestamp difference for unit `ns`, else its
 * wall clock — also for a pass whose difference is not positive (raw ticks are not ns, and a broken
 * timestamp would never add up to a bound).
 */
const loadNsOf = (context: SweepContext, sample: PassSample): number =>
  context.unit === "ns" && !sample.clamped && sample.gpu !== undefined && sample.gpu > 0
    ? sample.gpu
    : sample.wallNs;

/**
 * Whether the last passes agree: {@link WARM_START_PASSES} positive times whose largest ÷ smallest
 * is at most 1 + {@link WARM_START_TOLERANCE}.
 */
const passesAgree = (times: readonly number[]): boolean => {
  if (times.length < WARM_START_PASSES) return false;
  const smallest = Math.min(...times);
  return smallest > 0 && Math.max(...times) / smallest <= 1 + WARM_START_TOLERANCE;
};

/** What a warm start did. */
export type WarmStart = {
  /**
   * The last passes agreed (else it stopped at {@link WARM_START_MAX_NS} of load or after
   * {@link WARM_START_MAX_PASSES} passes).
   */
  readonly stable: boolean;
  /** The time under load it spent (ns, summed over its passes; the probe that sized them not counted). */
  readonly loadNs: number;
  /** Its passes (likewise without the probe's). */
  readonly passes: number;
};

/**
 * Brings the device to a steady state on `ready` (protocol rule 3): passes of the dispatch, back to
 * back, until the last {@link WARM_START_PASSES} agree within {@link WARM_START_TOLERANCE} with at
 * least {@link WARM_START_MIN_NS} of load spent, or until {@link WARM_START_MAX_NS} of load is
 * spent or {@link WARM_START_MAX_PASSES} passes have run. A pass compares by its timestamp
 * difference (its wall clock without timestamps); a clamped difference agrees with nothing. The
 * first passes take the reps a probe gives, which a cold device underestimates: a pass that falls
 * below half the target as the dispatch speeds up gets the reps that fill the target again (the
 * comparison then starts over — passes of other lengths do not compare), so the load stays
 * continuous.
 */
export const warmStart = async (context: SweepContext, ready: ReadyLaunch): Promise<WarmStart> => {
  const timed = context.unit !== "wall";
  let reps = await probeReps(context, ready);
  let times: number[] = [];
  let loadNs = 0;
  let passes = 0;
  for (;;) {
    const sample = await runPass(context, ready, reps, timed);
    passes += 1;
    const load = loadNsOf(context, sample);
    loadNs += load;
    times = [...times, sample.clamped ? 0 : (sample.gpu ?? sample.wallNs)].slice(
      -WARM_START_PASSES,
    );
    if (loadNs >= WARM_START_MIN_NS && passesAgree(times)) return { stable: true, loadNs, passes };
    if (loadNs >= WARM_START_MAX_NS || passes >= WARM_START_MAX_PASSES) {
      return { stable: false, loadNs, passes };
    }
    if (load < PROBE_MIN_NS) {
      const filling = calibrateReps(load / reps, TARGET_PASS_MS, SWEEP_MAX_REPS);
      if (filling !== reps) {
        reps = filling;
        times = [];
      }
    }
  }
};

/** The pass length below which a calibration is done again (ns — protocol rule 1). */
const RECALIBRATE_BELOW_NS = RECALIBRATE_BELOW * TARGET_PASS_MS * 1e6;

/**
 * The `reps` of `ready` (protocol rule 1): a probe, then one pass at its reps that checks it. A pass
 * below {@link RECALIBRATE_BELOW} of the target means the probe read the dispatch slower than it
 * runs: a warm start loads the device and the probe is taken again, up to
 * {@link CALIBRATION_ATTEMPTS} probes, the last accepted unchecked. Reps at {@link SWEEP_MAX_REPS}
 * are accepted as they are (a dispatch too short to fill the target at the bound is measured at the
 * bound). `probed` is the reps of a probe already taken (the kernel's quick look): it stands for the
 * first probe.
 */
export const calibrate = async (
  context: SweepContext,
  ready: ReadyLaunch,
  probed?: number,
): Promise<number> => {
  let reps = probed ?? (await probeReps(context, ready));
  for (let attempt = 1; attempt < CALIBRATION_ATTEMPTS && reps < SWEEP_MAX_REPS; attempt += 1) {
    const sample = await runPass(context, ready, reps, context.unit === "ns");
    if (loadNsOf(context, sample) >= RECALIBRATE_BELOW_NS) return reps;
    await warmStart(context, ready);
    reps = await probeReps(context, ready);
  }
  return reps;
};

/** One measured pass of `reps` dispatches — a round (timestamps unless the unit is `wall`). */
export const measurePass = (
  context: SweepContext,
  ready: ReadyLaunch,
  reps: number,
): Promise<PassSample> => runPass(context, ready, reps, context.unit !== "wall");

/**
 * The timing of rounds of `reps` dispatches per pass (protocol rule 2): a round's value is its
 * timestamp difference, or its wall clock when the pass was not timed. Takes at least one round.
 */
export const roundsTiming = (reps: number, samples: readonly PassSample[]): Timing => {
  const valueOf = (sample: PassSample): number => sample.gpu ?? sample.wallNs;
  const rounds = samples.map(valueOf);
  const wallRounds = samples.map((sample) => sample.wallNs);
  // MUST: a sample clamped to 0 stays in the record but is excluded from the min candidates (a 0
  // taking the min would turn the speed ratio into infinity)
  const kept = samples.filter((sample) => !sample.clamped).map(valueOf);
  return {
    reps,
    rounds,
    wallRounds,
    clampedNegativeSamples: rounds.length - kept.length,
    ...(kept.length === 0 ? {} : { perDispatch: Math.min(...kept) / reps }),
    wallNsPerDispatch: Math.min(...wallRounds) / reps,
  };
};

/**
 * Measures one geometry: reset the output to 0 and run once → with `warmStart` (the first
 * measurement of a case without a quick look), a warm start on it → calibrate `reps` (from
 * `probed`, its quick look's reps, when given) → idle runs → `rounds` measurements (protocol rules
 * 1–3).
 */
export const timeLaunch = async (
  context: SweepContext,
  ready: ReadyLaunch,
  output: GPUBuffer,
  rounds: number,
  options: { readonly warmStart?: boolean; readonly probed?: number } = {},
): Promise<Timing> => {
  // The first run starts from an output reset to 0 (unwritten tiles show as a digest mismatch)
  await runPass(context, ready, 1, false, output);
  if (options.warmStart === true) await warmStart(context, ready);
  const reps = await calibrate(context, ready, options.probed);
  await warmUp(context, ready, reps);
  const samples: PassSample[] = [];
  for (let round = 0; round < rounds; round += 1) {
    samples.push(await measurePass(context, ready, reps));
  }
  return roundsTiming(reps, samples);
};

/** The order of the four passes of one paired round: A, B, B, A (a drift within the round cancels). */
export const PAIR_ORDER = ["a", "b", "b", "a"] as const;

/** One paired round: the two passes of each arm, in the record's unit, and their wall clocks. */
export type PairedRound = {
  readonly a: readonly [number, number];
  readonly b: readonly [number, number];
  readonly wallA: readonly [number, number];
  readonly wallB: readonly [number, number];
  /** Passes of the round whose timestamp difference was negative (clamped to 0). */
  readonly clamped: number;
};

export type PairedTiming = {
  /** The same repetitions for both arms (calibrated on A), so the passes keep one duty cycle. */
  readonly reps: number;
  readonly rounds: readonly PairedRound[];
};

/**
 * Measures two dispatches in the same rounds (an A/B whose thermal drift cancels): reset both
 * outputs and run each once → estimate `reps` on A → idle runs of A, then B → `rounds` rounds of
 * {@link PAIR_ORDER}. The two arms share `reps`, so a per-round ratio compares passes of one length
 * taken under the same clocks.
 */
export const timeLaunchPair = async (
  context: SweepContext,
  a: ReadyLaunch,
  b: ReadyLaunch,
  outputs: { readonly a: GPUBuffer; readonly b: GPUBuffer },
  rounds: number,
): Promise<PairedTiming> => {
  await runPass(context, a, 1, false, outputs.a);
  await runPass(context, b, 1, false, outputs.b);
  const reps = await probeReps(context, a);
  await warmUp(context, a, reps);
  await warmUp(context, b, reps);
  const timed = context.unit !== "wall";
  const measured: PairedRound[] = [];
  for (let round = 0; round < rounds; round += 1) {
    const samples: { readonly arm: "a" | "b"; readonly sample: PassSample }[] = [];
    for (const arm of PAIR_ORDER) {
      samples.push({ arm, sample: await runPass(context, arm === "a" ? a : b, reps, timed) });
    }
    const of = (arm: "a" | "b", pick: (sample: PassSample) => number): [number, number] => {
      const [first, second] = samples.filter((entry) => entry.arm === arm);
      return [pick(first.sample), pick(second.sample)];
    };
    measured.push({
      a: of("a", (sample) => sample.gpu ?? sample.wallNs),
      b: of("b", (sample) => sample.gpu ?? sample.wallNs),
      wallA: of("a", (sample) => sample.wallNs),
      wallB: of("b", (sample) => sample.wallNs),
      clamped: samples.filter(({ sample }) => sample.clamped).length,
    });
  }
  return { reps, rounds: measured };
};

/** Runs one dispatch once (no timestamps); with `clear`, resets that buffer to 0 first. */
export const runLaunchOnce = async (
  context: SweepContext,
  ready: ReadyLaunch,
  clear?: GPUBuffer,
): Promise<void> => {
  await runPass(context, ready, 1, false, clear);
};

/** Reads `bytes` of `buffer` from `offset` through the context's staging buffer (a copy). */
const readChunk = async (
  context: SweepContext,
  buffer: GPUBuffer,
  offset: number,
  bytes: number,
): Promise<Uint32Array<ArrayBuffer>> => {
  const { gpu, staging } = context;
  await submitAndWait(context, "buffer readback", (encoder) => {
    encoder.copyBufferToBuffer(buffer, offset, staging, 0, bytes);
  });
  await gpu[RUNTIME_INTERNAL].raceDeviceLost(
    staging.mapAsync(MAP_MODE.READ, 0, bytes),
    "buffer readback",
  );
  const copy = new Uint32Array(staging.getMappedRange(0, bytes).slice(0));
  staging.unmap();
  return copy;
};

/**
 * Reads the first `bytes` of `buffer` into one array, one staging chunk at a time (the GEMV sweep's
 * comparison with its host oracle).
 */
export const readBufferWords = async (
  context: SweepContext,
  buffer: GPUBuffer,
  bytes: number,
): Promise<Uint32Array<ArrayBuffer>> => {
  const words = new Uint32Array(bytes / 4);
  for (let offset = 0; offset < bytes; offset += context.staging.size) {
    const length = Math.min(context.staging.size, bytes - offset);
    words.set(await readChunk(context, buffer, offset, length), offset / 4);
  }
  return words;
};

/**
 * Compares the first `bytes` of two buffers word for word, one staging chunk at a time (the JS
 * side holds two chunks at most): the number of words, of differing words, and the first difference.
 */
export const compareBufferWords = async (
  context: SweepContext,
  a: GPUBuffer,
  b: GPUBuffer,
  bytes: number,
): Promise<{
  readonly words: number;
  readonly differing: number;
  readonly first?: { readonly index: number; readonly a: number; readonly b: number };
}> => {
  let differing = 0;
  let first: { readonly index: number; readonly a: number; readonly b: number } | undefined;
  for (let offset = 0; offset < bytes; offset += context.staging.size) {
    const length = Math.min(context.staging.size, bytes - offset);
    const left = await readChunk(context, a, offset, length);
    const right = await readChunk(context, b, offset, length);
    for (let at = 0; at < left.length; at += 1) {
      if (left[at] === right[at]) continue;
      differing += 1;
      first ??= { index: offset / 4 + at, a: left[at], b: right[at] };
    }
  }
  return { words: bytes / 4, differing, ...(first === undefined ? {} : { first }) };
};

/**
 * Prepares and measures one candidate's dispatch from a calibration of its own (params are released
 * after measuring) — the default geometry's re-measurement at the end of a case (protocol rule 5).
 */
const timeCandidate = async (
  context: SweepContext,
  plan: CasePlan,
  resources: ReadonlyMap<string, GPUBuffer>,
  candidate: GeometryCandidate,
  rounds: number,
): Promise<{ readonly launch: Launch; readonly timing: Timing }> => {
  const output = resources.get(plan.output);
  if (output === undefined) throw new Error(`Output ${plan.output} is missing`);
  const ready = await prepareLaunch(context, plan.launch(candidate), resources);
  try {
    return { launch: ready.launch, timing: await timeLaunch(context, ready, output, rounds) };
  } finally {
    ready.params.destroy();
  }
};

const ALL_ROUNDS_NEGATIVE =
  "timestamp negative in every round (only samples clamped to 0, so no min can be taken)";

/** The identity fields of a geometry's row. */
const rowHead = (
  sweepCase: SweepCase,
  candidate: GeometryCandidate,
  isDefault: boolean,
): Pick<
  SweepRow,
  "caseId" | "op" | "shape" | "censusCount" | "geometry" | "geometryParams" | "isDefault"
> => ({
  caseId: sweepCase.id,
  op: sweepCase.op,
  shape: caseShape(sweepCase),
  censusCount: sweepCase.censusCount,
  geometry: candidate.name,
  geometryParams: candidate.geometry,
  isDefault,
});

/** The row of a geometry the quick look did not let go on (protocol rule 4): its quick time only. */
const screenedRow = (
  sweepCase: SweepCase,
  candidate: GeometryCandidate,
  ready: ReadyLaunch,
  quick: QuickLook | undefined,
): SweepRow => ({
  ...rowHead(sweepCase, candidate, false),
  key: ready.launch.key,
  workgroups: ready.launch.workgroups,
  ...(quick?.perDispatch === undefined ? {} : { quickPerDispatch: quick.perDispatch }),
  screened: true,
});

/**
 * Measures one prepared geometry (protocol rules 1–3, its calibration going on from its quick
 * look's reps when it had one) and reads its output's digest.
 */
const measureGeometry = async (
  context: SweepContext,
  sweepCase: SweepCase,
  output: GPUBuffer,
  candidate: GeometryCandidate,
  isDefault: boolean,
  rounds: number,
  ready: ReadyLaunch,
  quick: QuickLook | undefined,
): Promise<SweepRow> => {
  const timing = await timeLaunch(
    context,
    ready,
    output,
    rounds,
    quick === undefined ? {} : { probed: quick.reps },
  );
  const measured = {
    ...rowHead(sweepCase, candidate, isDefault),
    key: ready.launch.key,
    workgroups: ready.launch.workgroups,
    ...(quick?.perDispatch === undefined ? {} : { quickPerDispatch: quick.perDispatch }),
    reps: timing.reps,
    rounds: timing.rounds,
    wallRounds: timing.wallRounds,
    clampedNegativeSamples: timing.clampedNegativeSamples,
  } satisfies SweepRow;
  if (timing.perDispatch === undefined) {
    return { ...measured, wallNsPerDispatch: timing.wallNsPerDispatch, error: ALL_ROUNDS_NEGATIVE };
  }
  // Raw ticks are not ns, so TFLOPS comes from the wall clock (the definition of `tflops` in
  // report.ts)
  const realNs = context.unit === "ns" ? timing.perDispatch : timing.wallNsPerDispatch;
  return {
    ...measured,
    perDispatch: timing.perDispatch,
    wallNsPerDispatch: timing.wallNsPerDispatch,
    ...(realNs > 0 ? { tflops: caseFlops(sweepCase) / realNs / 1e3 } : {}),
    outputSha256: await outputDigest(context, output),
  };
};

/** Where one geometry of a case stands before its row is made. */
type GeometryState =
  | { readonly kind: "planned" }
  | { readonly kind: "prepared"; readonly ready: ReadyLaunch; readonly quick?: QuickLook }
  | { readonly kind: "failed"; readonly cause: unknown };

/** The result of one case (the rows of every geometry and the per-case record). */
export type CaseResult = {
  readonly rows: readonly SweepRow[];
  readonly summary: CaseSummary;
  /**
   * Whether an abort (`signal`) stopped it before the candidate geometries or the default
   * re-measurement were done.
   */
  readonly aborted: boolean;
};

/**
 * Measures one case: every geometry's pipeline first, a warm start on the default geometry (protocol
 * rule 3), the quick look at every geometry in a shuffled order and the screening against the
 * default's quick time (protocol rule 4 — skipped at a margin of 0), then the default geometry and
 * every candidate that goes on, in candidate order, each timed from its own calibration (rules 1–3);
 * a screened candidate's row holds its quick time only. When done, measures the default geometry once
 * more and keeps the ratio to the first in the per-case record (rule 5). If planning, resource
 * allocation or the prelude fails, every geometry becomes a failure row; a geometry whose pipeline,
 * warm start, quick look or timing fails becomes its own failure row. If the device is lost, the
 * rest become failure rows; if aborted, no rows are made for the rest (and no re-measurement — an
 * abort before the timing makes no row of the case). The case's wall clock (`elapsedMs`) covers every
 * step from planning to the re-measurement.
 */
export const sweepCase = async (
  context: SweepContext,
  target: SweepCase,
  candidates: readonly GeometryCandidate[],
  settings: Pick<SweepSettings, "rounds" | "quickLookMargin">,
  hooks: SweepHooks = {},
): Promise<CaseResult> => {
  const { gpu } = context;
  const started = performance.now();
  const rows: SweepRow[] = [];
  let aborted = false;
  const emit = (row: SweepRow): void => {
    rows.push(row);
    hooks.onRow?.(row);
  };
  const finish = (summary: Omit<CaseSummary, "elapsedMs" | "screened">): CaseResult => {
    const timed: CaseSummary = {
      ...summary,
      elapsedMs: performance.now() - started,
      screened: rows.filter((row) => row.screened === true).length,
    };
    hooks.onCase?.(timed);
    return { rows, summary: timed, aborted };
  };
  let plan: CasePlan;
  try {
    plan = casePlan(target, gpu.limits.maxComputeWorkgroupsPerDimension, context.dp4a);
  } catch (cause) {
    // The default geometry was not resolved either, so only the candidates' rows are kept as
    // failures
    candidates.forEach((candidate) => emit(errorRow(target, candidate, false, cause)));
    return finish({ caseId: target.id });
  }
  const ordered = [
    plan.defaultCandidate,
    ...candidates.filter((candidate) => candidate.name !== plan.defaultCandidate.name),
  ];
  let resources: Map<string, GPUBuffer>;
  // MUST: the user's callbacks (onProgress / onRow / onCase) are called outside the measurement's
  // try — called inside, an exception thrown by the user's code would turn into a measurement
  // failure (a failure row, defaultRepeatError), get written into the record, and the notifications
  // would be duplicated. The user's exception propagates to the caller as is (the outer finally
  // releases the resources). emit / finish inside a catch clause are not re-caught, so they can
  // stay there
  hooks.onProgress?.(`${target.id}: preparing inputs`);
  try {
    resources = await createResources(context, plan.resources);
  } catch (cause) {
    ordered.forEach((candidate, index) => emit(errorRow(target, candidate, index === 0, cause)));
    return finish({ caseId: target.id });
  }
  try {
    try {
      for (const launch of plan.prelude) {
        const ready = await prepareLaunch(context, launch, resources);
        try {
          await runPass(context, ready, 1, false);
        } finally {
          ready.params.destroy();
        }
      }
    } catch (cause) {
      ordered.forEach((candidate, index) => emit(errorRow(target, candidate, index === 0, cause)));
      return finish({ caseId: target.id });
    }
    const output = resources.get(plan.output);
    if (output === undefined) throw new Error(`${target.id}: output ${plan.output} is missing`);
    const states: GeometryState[] = ordered.map(() => ({ kind: "planned" }));
    const prepared: ReadyLaunch[] = [];
    const abortRequested = (): boolean => hooks.signal?.aborted === true;
    let reference: DefaultReference = {};
    try {
      // 1. Every pipeline first (a compile idles the device)
      for (const [index, candidate] of ordered.entries()) {
        if (abortRequested()) {
          aborted = true;
          return finish({ caseId: target.id });
        }
        if (gpu.lost !== undefined) break;
        try {
          const ready = await prepareLaunch(context, plan.launch(candidate), resources);
          prepared.push(ready);
          states[index] = { kind: "prepared", ready };
        } catch (cause) {
          states[index] = { kind: "failed", cause };
        }
      }
      // 2. The warm start, on the default geometry unless it failed (then on the next that did not)
      for (const [index, state] of states.entries()) {
        if (abortRequested()) {
          aborted = true;
          return finish({ caseId: target.id });
        }
        if (gpu.lost !== undefined) break;
        if (state.kind !== "prepared") continue;
        hooks.onProgress?.(`${target.id}: warm start on ${ordered[index].name}`);
        try {
          await warmStart(context, state.ready);
          break;
        } catch (cause) {
          states[index] = { kind: "failed", cause };
        }
      }
      // 3. The quick looks, in an order that does not follow the candidates'
      if (settings.quickLookMargin > 0) {
        const order = quickLookOrder([...states.keys()], target.id);
        for (const [position, index] of order.entries()) {
          if (abortRequested()) {
            aborted = true;
            return finish({ caseId: target.id });
          }
          if (gpu.lost !== undefined) break;
          const state = states[index];
          if (state.kind !== "prepared") continue;
          hooks.onProgress?.(`${target.id}: quick look ${position + 1}/${order.length}`);
          try {
            states[index] = { ...state, quick: await quickLook(context, state.ready) };
          } catch (cause) {
            states[index] = { kind: "failed", cause };
          }
        }
      }
      // A geometry is judged against the default in every case of its class: the default's quick
      // time is the bar
      const goesOn = quickLookGoesOn(
        ordered.map((candidate, index) => {
          const state = states[index];
          const quick = state.kind === "prepared" ? state.quick?.perDispatch : undefined;
          return {
            name: candidate.name,
            class: "geometry",
            ...(quick === undefined ? {} : { perDispatch: quick }),
            reference: index === 0,
          };
        }),
        settings.quickLookMargin,
        () => ({ against: "reference", reference: plan.defaultCandidate.name }),
      );
      // 4. The default geometry and the candidates that go on, in candidate order
      for (const [index, candidate] of ordered.entries()) {
        if (abortRequested()) {
          aborted = true;
          break;
        }
        const isDefault = index === 0;
        const state = states[index];
        if (gpu.lost !== undefined || state.kind === "planned") {
          const lost = new Error(`device lost: ${gpu.lost?.message ?? "no loss reported"}`);
          emit(errorRow(target, candidate, isDefault, lost));
          continue;
        }
        let row: SweepRow;
        if (state.kind === "failed") {
          row = errorRow(target, candidate, isDefault, state.cause);
        } else if (goesOn[index] !== true) {
          row = screenedRow(target, candidate, state.ready, state.quick);
        } else {
          hooks.onProgress?.(`${target.id}: ${candidate.name} (${index + 1}/${ordered.length})`);
          try {
            row = await measureGeometry(
              context,
              target,
              output,
              candidate,
              isDefault,
              settings.rounds,
              state.ready,
              state.quick,
            );
          } catch (cause) {
            row = errorRow(target, candidate, isDefault, cause);
          }
        }
        if (isDefault) {
          reference = { perDispatch: row.perDispatch, outputSha256: row.outputSha256 };
        }
        emit(compareToDefault(row, reference));
      }
    } finally {
      for (const ready of prepared) ready.params.destroy();
    }
    const initial = reference.perDispatch;
    // A case whose first default measurement failed, is 0 (a broken timestamp with a difference of
    // 0 — the ratio would turn into infinity or NaN), or hit device lost is not re-measured even
    // without an abort (no ratio baseline — the generator drops the case from the ratio material)
    if (initial === undefined || !(initial > 0) || gpu.lost !== undefined) {
      return finish({ caseId: target.id });
    }
    if (hooks.signal?.aborted === true) {
      aborted = true;
      return finish({ caseId: target.id });
    }
    hooks.onProgress?.(`${target.id}: re-measuring the default geometry`);
    let repeat: Pick<CaseSummary, "defaultRepeat" | "defaultRepeatError">;
    try {
      const { timing } = await timeCandidate(
        context,
        plan,
        resources,
        plan.defaultCandidate,
        settings.rounds,
      );
      if (timing.perDispatch === undefined) throw new Error(ALL_ROUNDS_NEGATIVE);
      // A re-measurement of 0 gives a ratio of 0, and the generator, which accepts only positive
      // ratios, would reject the whole record — keep it as a failure so that only the case is
      // dropped
      if (!(timing.perDispatch > 0)) {
        throw new Error(
          "perDispatch of the default re-measurement is 0 (timestamp difference 0 — no ratio can be computed)",
        );
      }
      repeat = {
        defaultRepeat: {
          perDispatch: timing.perDispatch,
          driftRatio: timing.perDispatch / initial,
        },
      };
    } catch (cause) {
      repeat = { defaultRepeatError: describeCause(cause) };
    }
    // onCase is outside the re-measurement's try (the MUST at the top of the function — a user
    // exception does not turn into a re-measurement failure, and nothing is notified twice)
    return finish({ caseId: target.id, ...repeat });
  } finally {
    for (const buffer of resources.values()) buffer.destroy();
  }
};

/** The result of the whole sweep (rows and per-case records). */
export type SweepResult = {
  readonly rows: readonly SweepRow[];
  readonly cases: readonly CaseSummary[];
  /**
   * Whether an abort (`signal`) stopped it before the planned cases, geometries and default
   * re-measurements were done. An abort that arrives after everything was measured does not count
   * (the record is complete).
   */
  readonly aborted: boolean;
};

/**
 * Sweeps the cases in order (if the device is lost or the sweep aborted, the remaining cases are
 * not run).
 */
export const runSweep = async (
  context: SweepContext,
  cases: readonly SweepCase[],
  settings: SweepSettings,
  hooks: SweepHooks = {},
): Promise<SweepResult> => {
  const rows: SweepRow[] = [];
  const summaries: CaseSummary[] = [];
  let aborted = false;
  for (const target of cases) {
    if (context.gpu.lost !== undefined) break;
    if (hooks.signal?.aborted === true) {
      aborted = true;
      break;
    }
    const result = await sweepCase(
      context,
      target,
      selectedCandidates(target, settings.candidateSet, settings.geometries),
      settings,
      hooks,
    );
    rows.push(...result.rows);
    summaries.push(result.summary);
    if (result.aborted) {
      aborted = true;
      break;
    }
  }
  return { rows, cases: summaries, aborted };
};
