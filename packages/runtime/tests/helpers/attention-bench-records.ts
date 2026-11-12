// Synthetic attention-bench records (`karume-attention-bench/2`) for the tests of the profile
// derivation's second input (src/tune/derive.ts `deriveAttentionOnline` — ADR 0117 addendum
// 2026-11-12). Keys, producers and the measured rows come from the current runtime's own functions,
// so a record built here is one the derivation accepts unless a test breaks a field on purpose. No GPU.

import {
  type AttentionOnlineReduce,
  attentionOnlineI8a8Key,
  attentionOnlineKeyWords,
} from "../../src/kernels/attention-online-i8a8.ts";
import {
  attentionValueQuantization,
  planAttentionOnlineWindows,
} from "../../src/runtime/recipe-builders/attention.ts";
import { ATTENTION_BENCH_FORMAT, onlineDispatchRows } from "../../src/tune/attention-bench.ts";
import {
  ATTENTION_BENCH_SHAPES,
  type AttentionBenchShape,
} from "../../src/tune/attention-shapes.ts";
import { gemmCandidate } from "../../src/tune/geometries.ts";
import { DEFAULT, FAST } from "./sweep-records.ts";
import { REPORT_FORMAT } from "../../src/tune/report.ts";

/** The adapter of the sweep fixtures (sweep-records.ts `sweep`), which the bench records must share. */
export const BENCH_ADAPTER = {
  vendor: "apple",
  architecture: "metal-3",
  device: "",
  description: "Test GPU",
} as const;

/** One shape's timing and identity in a synthetic record (every field optional). */
export type ShapeTiming = {
  /** The minimum pass of each arm of the kernel pair (A = N's form, B = tiled), record units. */
  readonly kernel?: { readonly a: number; readonly b: number };
  /** The minimum pass of each arm of the producer pair. */
  readonly producer?: { readonly a: number; readonly b: number };
  /** A's passes in rounds 1–3 at this multiple of its minimum (a stall burst; 1 = none). */
  readonly stall?: number;
  readonly differingWords?: number;
  readonly differingScaleWords?: number;
  readonly kernelError?: string;
  /** Replaces the measured keys (a record of another runtime). */
  readonly kernelKeys?: readonly [string, string];
  readonly producerKeys?: readonly [string, string];
  /** Replaces the measured rows. */
  readonly rows?: number;
  /** Replaces the shape (a record of another shape set). */
  readonly shape?: AttentionBenchShape;
  /** Rounds every other pass up to a multiple of this step (give minimums that are multiples too). */
  readonly quantum?: number;
};

const KERNEL_REPS = 4;
const PRODUCER_REPS = 100;
const REDUCE: AttentionOnlineReduce = "sg32";

/**
 * Five paired rounds whose arms' minimum passes are `a` and `b` (A's passes in rounds 1–3 multiplied
 * by `stall`).
 */
const pairRounds = (a: number, b: number, stall = 1, quantum = 0): Record<string, unknown>[] =>
  [0, 1, 2, 3, 4].map((round) => {
    const slow = round >= 1 && round <= 3 ? stall : 1;
    // Rounded up to the quantum (Chrome's 100 µs timestamps), so the minimums stay a and b
    const q = (pass: number): number =>
      quantum === 0 ? pass : Math.ceil(pass / quantum) * quantum;
    return {
      a: [round === 0 ? a : q(a * 1.02 * slow), q(a * 1.01 * slow)],
      b: [q(b * 1.01), round === 4 ? b : q(b * 1.02)],
      wallA: [a, a],
      wallB: [b, b],
      clamped: 0,
    };
  });

/** The gain the derivation reads from a shape's timing (kernel + producer per node, minimums). */
export const expectedGain = (shape: AttentionBenchShape, timing: ShapeTiming = {}): number => {
  const kernel = timing.kernel ?? DEFAULT_KERNEL;
  const producer = timing.producer ?? DEFAULT_PRODUCER;
  const scale = shape.queries / measuredRows(shape);
  return (
    ((kernel.a / KERNEL_REPS) * scale + producer.a / PRODUCER_REPS) /
    ((kernel.b / KERNEL_REPS) * scale + producer.b / PRODUCER_REPS)
  );
};

/** The default timing: the tiled kernel ×1.10 faster, the producers equal and small. */
const DEFAULT_KERNEL = { a: 110_000_000, b: 100_000_000 };
const DEFAULT_PRODUCER = { a: 1_000_000, b: 1_000_000 };

/** The rows of the first window production runs (the measured dispatch). */
const measuredRows = (shape: AttentionBenchShape): number => {
  const windows = planAttentionOnlineWindows(shape.queries, {
    batch: shape.heads,
    cols: shape.keys,
    depth: shape.depth,
  });
  return onlineDispatchRows(windows[0].rows, shape.queries);
};

const shapeResult = (
  base: AttentionBenchShape,
  timing: ShapeTiming,
  dp4a: boolean,
): Record<string, unknown> => {
  const shape = timing.shape ?? base;
  const { heads, keys, depth } = shape;
  const own = attentionOnlineKeyWords(keys);
  const rows = timing.rows ?? measuredRows(shape);
  const kernel = timing.kernel ?? DEFAULT_KERNEL;
  const producer = timing.producer ?? DEFAULT_PRODUCER;
  const differingWords = timing.differingWords ?? 0;
  const differingScaleWords = timing.differingScaleWords ?? 0;
  return {
    shape,
    s16: { eligible: false, reason: "not part of these tests" },
    online: {
      eligible: true,
      rows,
      productionDispatches: 1,
      productionRows: rows,
      reduce: REDUCE,
      keyWords: own,
      stages: [],
    },
    onlineTiled: {
      eligible: true,
      rows,
      kernel: {
        stages: ["online", "onlineTiled"],
        keys: timing.kernelKeys ?? [
          attentionOnlineI8a8Key(dp4a, depth, REDUCE, "bhsd", own),
          attentionOnlineI8a8Key(dp4a, depth, REDUCE, "bhsd", "tiled"),
        ],
        workgroups: [
          [1, heads, 1],
          [1, heads, 1],
        ],
        ...(timing.kernelError === undefined
          ? {
              reps: KERNEL_REPS,
              rounds: pairRounds(kernel.a, kernel.b, timing.stall, timing.quantum),
            }
          : { error: timing.kernelError }),
      },
      identity: {
        identical: differingWords === 0 && differingScaleWords === 0,
        words: heads * shape.queries * depth,
        differingWords,
        differingScaleWords,
      },
    },
    quantization: {
      stages: [],
      tiledVt: {
        stages: ["quantize_vt", "quantize_vt_tiled"],
        keys: timing.producerKeys ?? [
          attentionValueQuantization(heads * depth, keys, own).key,
          attentionValueQuantization(heads * depth, keys, "tiled").key,
        ],
        workgroups: [
          [1, 1, 1],
          [1, 1, 1],
        ],
        reps: PRODUCER_REPS,
        rounds: pairRounds(producer.a, producer.b, 1, timing.quantum),
      },
    },
  };
};

/**
 * A record of the given shapes (default: the whole bench set with the default timing). `timings`
 * overrides per shape id; `shapes` narrows the record to some shapes (one record per shape, as the
 * Node bench writes them).
 */
export const benchRecord = (
  options: {
    readonly timings?: Readonly<Record<string, ShapeTiming>>;
    readonly shapes?: readonly AttentionBenchShape[];
    readonly adapter?: Readonly<Record<string, string>>;
    readonly unit?: string;
    readonly dot?: "dp4a" | "emu";
    readonly date?: string;
    readonly onlineTiled?: boolean;
  } = {},
): Record<string, unknown> => {
  const dot = options.dot ?? "dp4a";
  const shapes = options.shapes ?? ATTENTION_BENCH_SHAPES;
  return {
    format: ATTENTION_BENCH_FORMAT,
    startedAt: "2026-11-12T00:00:00.000Z",
    date: options.date ?? "2026-11-12T00:00:00.000Z",
    aborted: false,
    userAgent: "Node.js/24",
    adapter: options.adapter ?? BENCH_ADAPTER,
    timing: { unit: options.unit ?? "ns", gpuTimingEnabled: true },
    variants: {
      geometryProfile: "test",
      dp4a: { listed: true, dot },
      device: { subgroups: true, maxComputeWorkgroupStorageSize: 49152 },
    },
    settings: {
      rounds: 5,
      quantization: true,
      onlineTiled: options.onlineTiled ?? true,
      targetPassMs: 80,
      maxReps: 16384,
      warmupNs: 500_000_000,
      warmupMinRuns: 3,
    },
    shapes: shapes.map((shape) =>
      shapeResult(shape, options.timings?.[shape.id] ?? {}, dot === "dp4a"),
    ),
  };
};

/** A raw sweep record of one linear case (the default and a ×1.2 geometry — sweep-records.ts `sweep`). */
export const sweepRecord = (): Record<string, unknown> => ({
  format: REPORT_FORMAT,
  date: "2026-09-29T00:00:00.000Z",
  adapter: BENCH_ADAPTER,
  settings: { candidateSet: "full" },
  gpuTiming: { feature: true, unit: "ns", quantized: false },
  cases: [{ caseId: "linear-m1024", defaultRepeat: { perDispatch: 1, driftRatio: 1 } }],
  rows: [
    {
      caseId: "linear-m1024",
      op: "linear",
      shape: "M1024 N64 K64",
      geometry: gemmCandidate(DEFAULT).name,
      geometryParams: DEFAULT,
      isDefault: true,
      speedupVsDefault: 1,
      identicalToDefault: true,
    },
    {
      caseId: "linear-m1024",
      op: "linear",
      shape: "M1024 N64 K64",
      geometry: gemmCandidate(FAST).name,
      geometryParams: FAST,
      isDefault: false,
      speedupVsDefault: 1.2,
      identicalToDefault: true,
    },
  ],
});
