/**
 * The GEMV choices of linear as data of the geometry profile (`GeometryProfile.linearGemv`): the
 * values moved from module constants of src/kernels/linear-gemv.ts into the default profile without
 * changing a selection; a table without them runs with the default profile's, resolved where a
 * Session selects its profile; a broken one is refused; and the profile is really the source — a
 * table with other choices changes the selection, the pipeline key and the fusion pass. No GPU.
 */

import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
  resolveGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  type LinearGemvChoices,
  type LinearGemvParallelLanes,
  type LinearGemvParallelShape,
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvParallelLanes,
  linearGemvRowsForShape,
  linearGemvStagedEligible,
  linearGemvUnit,
  type LinearGemvVariant,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { FUSION_RULES, type FusionWeightLayout, planFusions } from "../src/runtime/fusion.ts";
import { sessionGeometryProfile } from "../src/runtime/geometry-selection.ts";
import { linearProgramFor } from "../src/runtime/linear-program.ts";
import {
  type LinearKnobs,
  resolveLinearStrategy,
  selectLinearStrategy,
} from "../src/runtime/linear-strategies.ts";
import { countUses, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

/** The default profile's GEMV choices. */
const GEMV = DEFAULT_GEOMETRY_PROFILE.linearGemv;

// ---------------------------------------------------------------------------
// The old module constants and lookups, copied from src/kernels/linear-gemv.ts as it stood before
// the move.
//
// Why this copy exists: the move must change nothing, and the old code is gone. These literals are
// the old `PARALLEL_SHAPES` (rows, order and flags), the old sequential variant and the old thread
// target, and the functions below are the old lookups that read them. A later change of a value
// edits the default profile (re-measuring as its comments require) and replaces this copy with it.
// ---------------------------------------------------------------------------

const OLD_PARALLEL_SHAPES: readonly LinearGemvParallelShape[] = [
  { storage: "i4", n: 8960, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 256, k: 1536, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 6144, k: 1536, group: 32, lanes: 4, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 6144,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 1536, k: 256, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 1536, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 12288, k: 1536, group: 32, lanes: 4, packedActivations: false },
  {
    storage: "i4",
    n: 1536,
    k: 12288,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false },
  { storage: "i4", n: 2048, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 256, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i4", n: 6144, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
  { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 1536, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 1536, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
  { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false },
  { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
  { storage: "i4", n: 10752, k: 2560, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 2560, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 2560, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 256, k: 2560, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 2560,
    k: 2048,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 10240, k: 2560, group: 32, lanes: 4, packedActivations: false },
  {
    storage: "i4",
    n: 2560,
    k: 10240,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 2560, k: 256, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 2560, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1024, k: 2560, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 2560,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false },
  { storage: "i4", n: 2048, k: 2560, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 2560, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i4", n: 10240, k: 2560, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
  { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false },
  { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 2560, group: 512, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1024, k: 2560, group: 512, lanes: 32, packedActivations: false },
  { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
  { storage: "i4", n: 15360, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 8192, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 3840, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 512, k: 3840, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 3840,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 3840,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 3840,
    k: 15360,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 262144, k: 3840, group: 32, lanes: 16, packedActivations: false },
  { storage: "i4", n: 8192, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 4096, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2048, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 2112, k: 2816, group: 32, lanes: 4, packedActivations: false },
  { storage: "i4", n: 1024, k: 2816, group: 32, lanes: 32, packedActivations: false },
  { storage: "i4", n: 128, k: 2816, group: 32, lanes: 32, packedActivations: false },
  {
    storage: "i4",
    n: 2816,
    k: 4096,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 2816,
    k: 8192,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  {
    storage: "i4",
    n: 2816,
    k: 2112,
    group: 32,
    lanes: 32,
    packedActivations: false,
    stagedActivations: true,
  },
  { storage: "i4", n: 262144, k: 2816, group: 32, lanes: 4, packedActivations: false },
];

const OLD_ROWS_THREAD_TARGET = 16384;

type Shape = { readonly storage: WeightStorage; readonly n: number; readonly k: number };

const oldDefaultVariant = (shape?: Shape): LinearGemvVariant => ({
  cols: shape?.storage === "i8" && shape.n === 262144 && shape.k === 1536 ? 16 : 32,
  unroll: 4,
});

/** The old row-block height (the ceiling of 256 elements per word stayed in the kernel). */
const oldRowsForShape = (
  storage: WeightStorage,
  m: number,
  n: number,
  threadTarget: number = OLD_ROWS_THREAD_TARGET,
): number => {
  const cap = 256 / linearGemvUnit(storage);
  let rows = 1;
  while (rows * 2 <= Math.min(m, cap)) rows *= 2;
  while (rows > 1 && n * Math.ceil(m / rows) < threadTarget) rows /= 2;
  return rows;
};

const oldParallelShapeFor = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelShape | undefined => {
  if (m < 1 || m > 8) return undefined;
  return OLD_PARALLEL_SHAPES.find(
    (shape) => shape.storage === storage && shape.n === n && shape.k === k && shape.group === group,
  );
};

const oldParallelEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  if (storage !== "i2" && storage !== "i4" && storage !== "i8") return undefined;
  if (!gemmUsesVec4(k, n)) return undefined;
  if (k <= 0 || k % linearGemvUnit(storage) !== 0) return undefined;
  if (storage === "i4") {
    if (group === undefined || group % linearGemvUnit("i4") !== 0) return undefined;
  } else if (group !== undefined) return undefined;
  return oldParallelShapeFor(storage, m, n, k, group)?.lanes;
};

const oldPackedEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  const lanes = oldParallelEligible(storage, m, n, k, group);
  if (lanes === undefined) return undefined;
  return oldParallelShapeFor(storage, m, n, k, group)?.packedActivations === true
    ? lanes
    : undefined;
};

const oldStagedEligible = (
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): LinearGemvParallelLanes | undefined => {
  const lanes = oldParallelEligible(storage, m, n, k, group);
  if (lanes === undefined) return undefined;
  const shape = oldParallelShapeFor(storage, m, n, k, group);
  return shape?.stagedActivations === true && !shape.packedActivations ? lanes : undefined;
};

/** The four parallel-family answers for one shape, new (with `choices`) or old. */
const parallelAnswers = (
  choices: LinearGemvChoices | undefined,
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  group?: number,
): readonly (LinearGemvParallelLanes | undefined)[] =>
  choices === undefined
    ? [
        oldParallelShapeFor(storage, m, n, k, group)?.lanes,
        oldParallelEligible(storage, m, n, k, group),
        oldPackedEligible(storage, m, n, k, group),
        oldStagedEligible(storage, m, n, k, group),
      ]
    : [
        linearGemvParallelLanes(choices, storage, m, n, k, group),
        linearGemvParallelEligible(choices, storage, m, n, k, group),
        linearGemvPackedEligible(choices, storage, m, n, k, group),
        linearGemvStagedEligible(choices, storage, m, n, k, group),
      ];

const STORAGES: readonly WeightStorage[] = ["i2", "i4", "i8", "f16", "bf16", "f32"];

describe("the default profile holds the GEMV choices the kernel module held", () => {
  it("holds the old table of parallel shapes (rows, order and flags), the old sequential variant and the old thread target", () => {
    assertEquals(GEMV.parallelShapes, OLD_PARALLEL_SHAPES);
    assertEquals(GEMV.rowsThreadTarget, OLD_ROWS_THREAD_TARGET);
    assertEquals(GEMV.sequential, {
      variant: { cols: 32, unroll: 4 },
      shapes: [{ storage: "i8", n: 262144, k: 1536, variant: { cols: 16, unroll: 4 } }],
    });
  });

  it("answers every row of the old table as the old lookups did, inside and past the parallel range of M", () => {
    for (const { storage, n, k, group } of OLD_PARALLEL_SHAPES) {
      for (const m of [0, 1, 2, 4, 8, 9, 64]) {
        assertEquals(
          parallelAnswers(GEMV, storage, m, n, k, group),
          parallelAnswers(undefined, storage, m, n, k, group),
          `${storage} ${n}×${k} g${group} M${m}`,
        );
      }
    }
  });

  it("answers a grid of shapes off the table as the old lookups did", () => {
    let compared = 0;
    for (const storage of STORAGES) {
      for (const n of [64, 256, 260, 1536, 2048, 8960, 262144]) {
        for (const k of [256, 1536, 1552, 2048, 2560]) {
          for (const group of [undefined, 32, 64, 512]) {
            for (const m of [1, 8, 9]) {
              assertEquals(
                parallelAnswers(GEMV, storage, m, n, k, group),
                parallelAnswers(undefined, storage, m, n, k, group),
                `${storage} ${n}×${k} g${group} M${m}`,
              );
              compared += 1;
            }
          }
        }
      }
    }
    assertEquals(compared, 2520);
  });

  it("chooses the old sequential variant and the old row-block height", () => {
    assertEquals(defaultLinearGemvVariant(GEMV), oldDefaultVariant());
    for (const storage of STORAGES) {
      for (const [n, k] of [
        [262144, 1536],
        [262144, 2560],
        [262140, 1536],
        [6144, 1536],
        [64, 64],
      ]) {
        assertEquals(
          defaultLinearGemvVariant(GEMV, { storage, n, k }),
          oldDefaultVariant({ storage, n, k }),
          `${storage} ${n}×${k}`,
        );
      }
    }
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const n of [1, 256, 2048, 4096, 12288, 262144]) {
        for (let m = 1; m <= 64; m += 1) {
          assertEquals(
            linearGemvRowsForShape(GEMV, storage, m, n),
            oldRowsForShape(storage, m, n),
            `${storage} n=${n} m=${m}`,
          );
          assertEquals(
            defaultLinearGemvRowsVariant(GEMV, storage, m, n),
            { ...oldDefaultVariant(), rows: oldRowsForShape(storage, m, n) },
            `${storage} n=${n} m=${m}: the row-block variant`,
          );
        }
      }
    }
  });
});

// ---------------------------------------------------------------------------
// The profile is the source: a table with other choices changes what a linear selects and the key
// of the kernel it builds (through the selection and the program the recipe builder uses).
// ---------------------------------------------------------------------------

/** The Session's omitted values. */
const KNOBS: LinearKnobs = {
  linearCompute: "f32",
  linearGemvReduce: "sequential",
  integerGemv: false,
  linearGemvPlanes: false,
  linearGemvRowsThreadTarget: undefined,
  linearSplitK: false,
};

const REORDER: Partial<LinearKnobs> = { linearGemvReduce: "parallel" };

/** The plan of one `linear` node `[m, k] × [n, k]ᵀ + [n]`. */
const linearPlan = (m: number, n: number, k: number): NodePlan => {
  const ir = mergeGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  });
  const [plan] = planGraph(ir, {}).nodes;
  return plan;
};

type Linear = {
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly storage: WeightStorage;
  readonly group?: number;
  readonly knobs?: Partial<LinearKnobs>;
  /** Whether the linear is declared to read packed activations (the fusion pass's pairing). */
  readonly packed?: boolean;
};

/** The pipeline key of the one dispatch `linear` builds under `choices`. */
const keyOf = (choices: LinearGemvChoices, linear: Linear): string => {
  const { m, n, k, storage, group, packed = false } = linear;
  const plan = linearPlan(m, n, k);
  const choice = selectLinearStrategy({
    plan,
    weightStorage: storage,
    groupSize: group,
    knobs: { ...KNOBS, ...linear.knobs },
    linearGemv: choices,
    packed: packed ? { role: "read" } : undefined,
    prologue: undefined,
  });
  const [x, weight] = plan.inputShapes;
  const program = linearProgramFor(choice, {
    x,
    weight,
    m,
    n,
    k,
    weightStorage: storage,
    groupSize: group,
    packedScale: packed ? 0.5 : undefined,
    prologue: undefined,
    geometryProfile: DEFAULT_GEOMETRY_PROFILE,
    linearI8a8Dot: "dp4a",
  });
  assertEquals(program.dispatches.length, 1);
  return program.dispatches[0].key;
};

/** A measured row of the default table: i4 g32, lanes 4, neither packed nor staged. */
const WIDE: Linear = { m: 1, n: 8960, k: 1536, storage: "i4", group: 32, knobs: REORDER };
/** A measured row of the default table: i4 g32, lanes 32, staged. */
const STAGED: Linear = { m: 1, n: 3840, k: 4096, storage: "i4", group: 32, knobs: REORDER };

/** Whether a row of a table names the shape of `linear`. */
const rowOf =
  (linear: Linear) =>
  (row: LinearGemvParallelShape): boolean =>
    row.storage === linear.storage &&
    row.n === linear.n &&
    row.k === linear.k &&
    row.group === linear.group;

/** The default table's row for the shape of `linear` (a missing row is a bug of the test). */
const measured = (linear: Linear): LinearGemvParallelShape => {
  const row = GEMV.parallelShapes.find(rowOf(linear));
  if (row === undefined) throw new Error(`the default table names no row ${linear.n}×${linear.k}`);
  return row;
};

/** The default choices with the row of `linear` replaced by `row` (dropped when `undefined`). */
const withRow = (linear: Linear, row: LinearGemvParallelShape | undefined): LinearGemvChoices => {
  const kept = GEMV.parallelShapes.filter((candidate) => !rowOf(linear)(candidate));
  return { ...GEMV, parallelShapes: row === undefined ? kept : [row, ...kept] };
};

describe("a profile with other GEMV choices changes the selection and the pipeline key", () => {
  it("the sequential variant: the M=1 kernel and every row block take the table's", () => {
    const other: LinearGemvChoices = {
      ...GEMV,
      sequential: { ...GEMV.sequential, variant: { cols: 64, unroll: 2 } },
    };
    const decode: Linear = { m: 1, n: 64, k: 64, storage: "i4", group: 32 };
    const verify: Linear = { ...decode, m: 8 };
    assertEquals(keyOf(GEMV, decode), "linear_gemv:v1:f32:c32u4:wi4g32");
    assertEquals(keyOf(other, decode), "linear_gemv:v1:f32:c64u2:wi4g32");
    assertEquals(keyOf(GEMV, verify), "linear_gemv:v2:f32:c32u4r1:wi4g32");
    assertEquals(keyOf(other, verify), "linear_gemv:v2:f32:c64u2r1:wi4g32");
  });

  it("a per-shape exception of the sequential variant: only the named shape takes it, and a table without the default's exception drops it", () => {
    const other: LinearGemvChoices = {
      ...GEMV,
      sequential: {
        variant: GEMV.sequential.variant,
        shapes: [{ storage: "i4", n: 64, k: 64, variant: { cols: 16, unroll: 4 } }],
      },
    };
    const named: Linear = { m: 1, n: 64, k: 64, storage: "i4", group: 32 };
    assertEquals(keyOf(other, named), "linear_gemv:v1:f32:c16u4:wi4g32");
    assertEquals(keyOf(other, { ...named, n: 128 }), "linear_gemv:v1:f32:c32u4:wi4g32");
    const head: Linear = { m: 1, n: 262144, k: 1536, storage: "i8" };
    assertEquals(keyOf(GEMV, head), "linear_gemv:v1:f32:c16u4:wi8");
    assertEquals(keyOf(other, head), "linear_gemv:v1:f32:c32u4:wi8");
  });

  it("the row-block thread target: the table's moves the row count, and SessionOptions.linearGemvRowsThreadTarget replaces it", () => {
    const other: LinearGemvChoices = { ...GEMV, rowsThreadTarget: 2048 };
    const block: Linear = { m: 4, n: 2048, k: 64, storage: "i4", group: 32 };
    assertEquals(keyOf(GEMV, block), "linear_gemv:v2:f32:c32u4r1:wi4g32");
    assertEquals(keyOf(other, block), "linear_gemv:v2:f32:c32u4r4:wi4g32");
    // The option replaces the profile's target in either direction
    assertEquals(
      keyOf(other, { ...block, knobs: { linearGemvRowsThreadTarget: 16384 } }),
      "linear_gemv:v2:f32:c32u4r1:wi4g32",
    );
    assertEquals(
      keyOf(GEMV, { ...block, knobs: { linearGemvRowsThreadTarget: 2048 } }),
      "linear_gemv:v2:f32:c32u4r4:wi4g32",
    );
  });

  it("a parallel row added: a shape the default table does not name takes the parallel GEMV with the row's lanes", () => {
    const shape: Linear = { m: 1, n: 64, k: 1536, storage: "i4", group: 32, knobs: REORDER };
    const other = withRow(shape, {
      storage: "i4",
      n: 64,
      k: 1536,
      group: 32,
      lanes: 8,
      packedActivations: false,
    });
    assertEquals(keyOf(GEMV, shape), "linear_gemv:v1:f32:c32u4:wi4g32");
    assertEquals(keyOf(other, shape), "linear_gemv_parallel:wi4g32:l8");
  });

  it("a parallel row removed: a measured shape falls back to the sequential GEMV even at reorder", () => {
    const other = withRow(WIDE, undefined);
    assertEquals(keyOf(GEMV, WIDE), "linear_gemv_parallel:wi4g32:l4");
    assertEquals(keyOf(other, WIDE), "linear_gemv:v1:f32:c32u4:wi4g32");
  });

  it("the packed flag: a row marked packed reads packed activations, a row without it refuses the pairing", () => {
    const other = withRow(WIDE, { ...measured(WIDE), packedActivations: true });
    const { m, n, k, storage, group } = WIDE;
    assertEquals(linearGemvPackedEligible(GEMV, storage, m, n, k, group), undefined);
    assertEquals(linearGemvPackedEligible(other, storage, m, n, k, group), 4);
    assertEquals(
      keyOf(other, { ...WIDE, packed: true }),
      "linear_gemv_parallel:wi4g32:l4:packed-x-i8",
    );
    const refused = resolveLinearStrategy({
      plan: linearPlan(m, n, k),
      weightStorage: storage,
      groupSize: group,
      knobs: { ...KNOBS, ...REORDER },
      linearGemv: GEMV,
      packed: { role: "read" },
      prologue: undefined,
    });
    assert("refusal" in refused, "the default table pairs the row");
  });

  it("the staged flag: a row marked staged takes the staged kernel, a row without it the plain one", () => {
    assertEquals(keyOf(GEMV, WIDE), "linear_gemv_parallel:wi4g32:l4");
    assertEquals(
      keyOf(withRow(WIDE, { ...measured(WIDE), stagedActivations: true }), WIDE),
      "linear_gemv_parallel:wi4g32:l4:staged-x",
    );
    assertEquals(keyOf(GEMV, STAGED), "linear_gemv_parallel:wi4g32:l32:staged-x");
    const { stagedActivations: _staged, ...unstaged } = measured(STAGED);
    assertEquals(keyOf(withRow(STAGED, unstaged), STAGED), "linear_gemv_parallel:wi4g32:l32");
  });

  it("the fusion pass reads the same table: linear → static_quantize fuses only on a shape the table names", () => {
    const { n, k } = WIDE;
    const ir = mergeGraph({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear", "static_quantize"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [1, k] }],
      outputs: ["q"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: [n, k] },
        b: { dtype: "f32", shape: [n] },
        y: { dtype: "f32", shape: [1, n] },
        q: { dtype: "f32", shape: [1, n] },
      },
      nodes: [
        { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
        { op: "static_quantize", ins: ["y"], outs: ["q"], attrs: { scale: Math.fround(0.021) } },
      ],
    });
    const fusedWith = (choices: LinearGemvChoices): number =>
      planFusions(planGraph(ir, {}).nodes, {
        rules: FUSION_RULES,
        linearGemv: choices,
        useCounts: countUses(ir),
        outputNames: new Set(ir.outputs),
        limits: {
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
          maxComputeWorkgroupsPerDimension: 65535,
        },
        linearGemvReduce: "parallel",
        linearCompute: "f32",
        weightLayouts: new Map<string, FusionWeightLayout>([
          ["w", { storage: "i4", groupSize: 32 }],
        ]),
      }).counts.linearStaticQuantize;
    assertEquals(fusedWith(GEMV), 1);
    assertEquals(fusedWith(withRow(WIDE, undefined)), 0);
  });
});

// ---------------------------------------------------------------------------
// Resolution: a table without the field runs with the default profile's choices.
// ---------------------------------------------------------------------------

const ZERO_LIMITS: RequiredLimits = {
  maxBufferSize: 0,
  maxStorageBufferBindingSize: 0,
  maxUniformBufferBindingSize: 0,
  maxStorageBuffersPerShaderStage: 0,
  maxUniformBuffersPerShaderStage: 0,
  maxComputeWorkgroupStorageSize: 0,
  maxComputeInvocationsPerWorkgroup: 0,
  maxComputeWorkgroupSizeX: 0,
  maxComputeWorkgroupSizeY: 0,
  maxComputeWorkgroupSizeZ: 0,
  maxComputeWorkgroupsPerDimension: 0,
};

/** A real GpuContext over a fake device, with `injected` as the table `acquireGpu` was given. */
const gpuWith = (injected: GeometryProfile | undefined): GpuContext =>
  new GpuContext(fakeDevice(), readAdapterInfo({}), ZERO_LIMITS, new Set(), undefined, injected);

/** A table without the GEMV choices (the shape of every embedded table). */
const { linearGemv: _default, ...WITHOUT } = { ...DEFAULT_GEOMETRY_PROFILE, id: "without-gemv" };

/** A table with GEMV choices of its own. */
const OWN: GeometryProfile = {
  ...WITHOUT,
  id: "own-gemv",
  linearGemv: { ...GEMV, rowsThreadTarget: 2048 },
};

describe("a table without the GEMV choices runs with the default profile's, resolved where a Session selects its profile", () => {
  it("no embedded table carries the field, so every adapter runs with the default profile's choices", () => {
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      assertEquals("linearGemv" in profile, false, profile.id);
      assertStrictEquals(resolveGeometryProfile(profile).linearGemv, GEMV, profile.id);
    }
  });

  it("resolveGeometryProfile fills in the default's choices and keeps every other field of the table, and keeps a table's own", () => {
    const resolved = resolveGeometryProfile(WITHOUT);
    assertStrictEquals(resolved.linearGemv, GEMV);
    const { linearGemv: _filled, ...rest } = resolved;
    assertEquals(rest, WITHOUT);
    assertEquals(resolveGeometryProfile(OWN), OWN);
  });

  it("sessionGeometryProfile resolves the injected table and the selected one", () => {
    assertStrictEquals(sessionGeometryProfile(gpuWith(WITHOUT)).linearGemv, GEMV);
    assertEquals(sessionGeometryProfile(gpuWith(WITHOUT)).id, "without-gemv");
    assertEquals(sessionGeometryProfile(gpuWith(OWN)).linearGemv, OWN.linearGemv);
    // An adapter no embedded table matches selects the default profile
    const selected = sessionGeometryProfile(gpuWith(undefined));
    assertEquals(selected.id, "default");
    assertStrictEquals(selected.linearGemv, GEMV);
  });
});

// ---------------------------------------------------------------------------
// The guard: a table carrying the field passes assertGeometryProfile only with well-formed choices.
// ---------------------------------------------------------------------------

/** `OWN` with the value at `path` (inside `linearGemv`) replaced, as a JS caller could pass it. */
const brokenAt = (path: readonly (string | number)[], value: unknown): unknown => {
  const copy: unknown = structuredClone({ ...OWN, id: "broken-gemv" });
  let target: unknown = copy;
  for (const key of path.slice(0, -1)) target = Reflect.get(Object(target), key);
  Reflect.set(Object(target), path[path.length - 1], value);
  return copy;
};

describe("assertGeometryProfile checks the GEMV choices", () => {
  it("accepts the default profile and a table with choices of its own", () => {
    assertGeometryProfile(DEFAULT_GEOMETRY_PROFILE);
    assertGeometryProfile(OWN);
    assertGeometryProfile(WITHOUT);
  });

  it("rejects malformed choices naming the field", () => {
    const where = "Geometry profile 'broken-gemv'";
    for (const [path, value, message] of [
      [["linearGemv"], 5, "linearGemv is not an object"],
      [
        ["linearGemv", "rowsThreadTarget"],
        0,
        "linearGemv.rowsThreadTarget must be a positive integer (0)",
      ],
      [
        ["linearGemv", "rowsThreadTarget"],
        Number.NaN,
        "linearGemv.rowsThreadTarget must be a positive integer (NaN)",
      ],
      [
        ["linearGemv", "sequential", "variant", "cols"],
        300,
        "linearGemv.sequential.variant: cols must be an integer in 1..256 (300)",
      ],
      [
        ["linearGemv", "sequential", "variant", "unroll"],
        "4",
        "linearGemv.sequential.variant: cols and unroll must be numbers",
      ],
      [
        ["linearGemv", "sequential", "shapes", 0, "storage"],
        "i3",
        "linearGemv.sequential.shapes[0].storage must be one of",
      ],
      [["linearGemv", "parallelShapes"], {}, "linearGemv.parallelShapes is not an array"],
      [["linearGemv", "parallelShapes"], new Array(1), "linearGemv.parallelShapes[0] is missing"],
      [
        ["linearGemv", "parallelShapes", 0, "lanes"],
        3,
        "linearGemv.parallelShapes[0].lanes must be one of 2 / 4 / 8 / 16 / 32 (3)",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "storage"],
        "i3",
        "linearGemv.parallelShapes[0].storage must be one of i2 / i4 / i8 / bf16 / f16 / f32",
      ],
      // Row 0 is an i4 row with its group: a float storage takes none
      [
        ["linearGemv", "parallelShapes", 0, "storage"],
        "f32",
        "linearGemv.parallelShapes[0]: a row of f32 storage takes no group and no packedActivations",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "n"],
        0,
        "linearGemv.parallelShapes[0].n must be a positive integer",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "group"],
        1.5,
        "linearGemv.parallelShapes[0].group must be a positive integer",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "packedActivations"],
        undefined,
        "linearGemv.parallelShapes[0].packedActivations is not a boolean",
      ],
      [
        ["linearGemv", "parallelShapes", 0, "stagedActivations"],
        false,
        "linearGemv.parallelShapes[0].stagedActivations must be true or absent",
      ],
    ] as const) {
      assertThrows(
        () => assertGeometryProfile(brokenAt(path, value)),
        CodegenError,
        `${where}: ${message}`,
      );
    }
  });
});
