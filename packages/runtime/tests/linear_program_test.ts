/**
 * The dispatch programs of `linear` (`src/runtime/linear-program.ts`), without a device: for one
 * representative linear of every strategy (the strategy taken by the selection,
 * `resolveLinearStrategy`), each dispatch binds exactly what its WGSL declares, binds only
 * temporaries the program declares, is named by the key its kernel module gives, and generates the
 * same WGSL each time. The loud failures that need no device fail in the program constructor.
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { DEFAULT_GEOMETRY_PROFILE, gemmRowsGeometry } from "../src/kernels/geometry-profile.ts";
import { linearKey } from "../src/kernels/linear.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelLanes,
  linearGemvParallelPackedKey,
  linearGemvParallelStagedKey,
  linearGemvRowsKey,
  linearGemvSubgroupKey,
} from "../src/kernels/linear-gemv.ts";
import { integerGemvKey, integerGemvVariantFor } from "../src/kernels/linear-gemv-integer.ts";
import { planesGemvKey, planesGemvVariantFor } from "../src/kernels/linear-gemv-planes.ts";
import {
  linearI8a8Key,
  linearI8a8UsesVec4,
  linearI8a8UsesVecFill,
} from "../src/kernels/linear-i8a8.ts";
import { linearSplitKCombineKey, linearSplitKPartialKey } from "../src/kernels/linear-splitk.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsPrologueKey,
} from "../src/kernels/quantize-rows.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import type { LinearPrologue } from "../src/runtime/fusion-rule.ts";
import {
  type LinearProgram,
  linearI8a8Geometry,
  linearProgramFor,
} from "../src/runtime/linear-program.ts";
import {
  LINEAR_STRATEGIES,
  type LinearChoice,
  type LinearKnobs,
  type LinearStrategyId,
  resolveLinearStrategy,
} from "../src/runtime/linear-strategies.ts";
import { ExecutionError, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import type { I8a8Dot } from "../src/runtime/session-types.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

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

/** The Session's omitted values. */
const DEFAULTS: LinearKnobs = {
  linearCompute: "f32",
  linearGemvReduce: "sequential",
  integerGemv: false,
  linearGemvPlanes: false,
  linearGemvRowsThreadTarget: undefined,
  linearSplitK: false,
};

const PROFILE = DEFAULT_GEOMETRY_PROFILE;

/** A value the kernel module's measured table must name for the case (a missing row is a test bug). */
const named = <T>(value: T | undefined, what: string): T => {
  if (value === undefined) throw new Error(`the measured table names no ${what}`);
  return value;
};

type Shape = { readonly m: number; readonly n: number; readonly k: number };

type Case = {
  readonly name: string;
  readonly id: LinearStrategyId;
  readonly shape: Shape;
  readonly storage: WeightStorage;
  readonly groupSize?: number;
  readonly knobs?: Partial<LinearKnobs>;
  /** The producer SRQ's scale of a declared packed-activation pairing (the reading side). */
  readonly packedScale?: number;
  readonly prologue?: LinearPrologue;
  readonly dot?: I8a8Dot;
  /** The key of each dispatch, from the kernel module's key function for the case's facts. */
  readonly keys: () => readonly string[];
};

/** The measured rows the strategies that take measured shapes only are chosen for. */
const PARALLEL_I4: Shape = { m: 1, n: 8960, k: 1536 };
const STAGED_I4: Shape = { m: 1, n: 3840, k: 4096 };
const PACKED_I2: Shape = { m: 1, n: 1536, k: 12288 };
const INTEGER_I2: Shape = { m: 1, n: 12288, k: 1536 };
const LM_HEAD_I2: Shape = { m: 1, n: 262144, k: 1536 };

const gemmKey = (
  storage: WeightStorage,
  compute: "f32" | "f16",
  { m, n, k }: Shape,
  groupSize?: number,
): string =>
  linearKey(storage, gemmUsesVec4(k, n), compute, m, groupSize, gemmRowsGeometry(PROFILE, m));

const i8a8Key = (
  storage: "i8" | "i4",
  { m, n, k }: Shape,
  dp4a: boolean,
  groupSize?: number,
): string => {
  const geometry = linearI8a8Geometry(PROFILE, m, storage, groupSize);
  return linearI8a8Key(
    linearI8a8UsesVec4(n),
    dp4a,
    geometry,
    storage,
    groupSize,
    linearI8a8UsesVecFill(geometry, storage, k),
  );
};

const CASES: readonly Case[] = [
  {
    name: "the default GEMM, f32 storage",
    id: "gemm-f32",
    shape: { m: 128, n: 64, k: 64 },
    storage: "f32",
    keys: () => [gemmKey("f32", "f32", { m: 128, n: 64, k: 64 })],
  },
  {
    name: "the default GEMM, i4 storage",
    id: "gemm-f32",
    shape: { m: 128, n: 64, k: 64 },
    storage: "i4",
    groupSize: 32,
    keys: () => [gemmKey("i4", "f32", { m: 128, n: 64, k: 64 }, 32)],
  },
  {
    name: "the f16 GEMM",
    id: "gemm-f16",
    shape: { m: 128, n: 64, k: 64 },
    storage: "f16",
    knobs: { linearCompute: "f16" },
    keys: () => [gemmKey("f16", "f16", { m: 128, n: 64, k: 64 })],
  },
  {
    name: "the split-K GEMM",
    id: "gemm-splitk",
    shape: { m: 16, n: 64, k: 1536 },
    storage: "f32",
    knobs: { linearSplitK: true },
    keys: () => [linearSplitKPartialKey(), linearSplitKCombineKey()],
  },
  {
    name: "the M = 1 GEMV, f16 storage",
    id: "gemv-m1",
    shape: { m: 1, n: 64, k: 64 },
    storage: "f16",
    keys: () => [
      linearGemvKey("f16", undefined, defaultLinearGemvVariant({ storage: "f16", n: 64, k: 64 })),
    ],
  },
  {
    name: "the M = 1 GEMV, i8 storage",
    id: "gemv-m1",
    shape: { m: 1, n: 64, k: 64 },
    storage: "i8",
    keys: () => [
      linearGemvKey("i8", undefined, defaultLinearGemvVariant({ storage: "i8", n: 64, k: 64 })),
    ],
  },
  {
    name: "the row-block GEMV, i4 storage",
    id: "gemv-rows",
    shape: { m: 8, n: 64, k: 64 },
    storage: "i4",
    groupSize: 32,
    keys: () => [linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 8, 64))],
  },
  {
    name: "the parallel GEMV",
    id: "gemv-parallel",
    shape: PARALLEL_I4,
    storage: "i4",
    groupSize: 32,
    knobs: { linearGemvReduce: "parallel" },
    keys: () => [
      linearGemvParallelKey(
        "i4",
        32,
        named(linearGemvParallelLanes("i4", 1, PARALLEL_I4.n, PARALLEL_I4.k, 32), "lanes"),
      ),
    ],
  },
  {
    name: "the parallel GEMV with staged activations",
    id: "gemv-parallel-staged",
    shape: STAGED_I4,
    storage: "i4",
    groupSize: 32,
    knobs: { linearGemvReduce: "parallel" },
    keys: () => [
      linearGemvParallelStagedKey(
        "i4",
        32,
        named(linearGemvParallelLanes("i4", 1, STAGED_I4.n, STAGED_I4.k, 32), "lanes"),
      ),
    ],
  },
  {
    name: "the parallel GEMV reading packed activations",
    id: "gemv-parallel-packed",
    shape: PACKED_I2,
    storage: "i2",
    knobs: { linearGemvReduce: "parallel" },
    packedScale: 0.5,
    keys: () => [
      linearGemvParallelPackedKey(
        "i2",
        undefined,
        named(linearGemvParallelLanes("i2", 1, PACKED_I2.n, PACKED_I2.k), "lanes"),
      ),
    ],
  },
  {
    name: "the subgroup GEMV",
    id: "gemv-parallel-subgroup32",
    shape: PARALLEL_I4,
    storage: "i4",
    groupSize: 32,
    knobs: { linearGemvReduce: "parallel-subgroup32" },
    keys: () => [
      linearGemvSubgroupKey(
        "i4",
        32,
        named(linearGemvParallelLanes("i4", 1, PARALLEL_I4.n, PARALLEL_I4.k, 32), "lanes"),
      ),
    ],
  },
  {
    name: "the byte-plane GEMV",
    id: "planes-gemv",
    shape: LM_HEAD_I2,
    storage: "i2",
    knobs: { linearGemvPlanes: true },
    keys: () => [
      planesGemvKey(named(planesGemvVariantFor("i2", 1, LM_HEAD_I2.n, LM_HEAD_I2.k), "variant")),
    ],
  },
  {
    name: "the integer GEMV",
    id: "integer-gemv",
    shape: INTEGER_I2,
    storage: "i2",
    knobs: { linearGemvReduce: "parallel", integerGemv: true },
    packedScale: 0.5,
    keys: () => [
      integerGemvKey(
        named(integerGemvVariantFor("i2", undefined, 1, INTEGER_I2.n, INTEGER_I2.k), "variant"),
        false,
      ),
    ],
  },
  {
    name: "the i8a8 path, i8 storage",
    id: "i8a8",
    shape: { m: 128, n: 64, k: 64 },
    storage: "i8",
    knobs: { linearCompute: "a8" },
    dot: "dp4a",
    keys: () => [
      quantizeRowsKey(quantizeRowsGeometry(64)),
      i8a8Key("i8", { m: 128, n: 64, k: 64 }, true),
    ],
  },
  {
    name: "the i8a8 path, i4 storage (w4a8)",
    id: "i8a8",
    shape: { m: 1, n: 64, k: 64 },
    storage: "i4",
    groupSize: 32,
    knobs: { linearCompute: "a8" },
    dot: "emu",
    keys: () => [
      quantizeRowsKey(quantizeRowsGeometry(64)),
      i8a8Key("i4", { m: 1, n: 64, k: 64 }, false, 32),
    ],
  },
  {
    name: "the i8a8 path with a folded producer",
    id: "i8a8",
    shape: { m: 128, n: 64, k: 64 },
    storage: "i8",
    knobs: { linearCompute: "a8" },
    prologue: { op: "gelu_tanh" },
    keys: () => [
      quantizeRowsPrologueKey("gelu_tanh", 64),
      i8a8Key("i8", { m: 128, n: 64, k: 64 }, true),
    ],
  },
];

/** The strategy the selection takes for the case, failing when it is not the one the case names. */
const choiceOf = (testCase: Case): { readonly plan: NodePlan; readonly choice: LinearChoice } => {
  const { m, n, k } = testCase.shape;
  const plan = linearPlan(m, n, k);
  const resolution = resolveLinearStrategy({
    plan,
    weightStorage: testCase.storage,
    groupSize: testCase.groupSize,
    knobs: { ...DEFAULTS, ...testCase.knobs },
    packed: testCase.packedScale === undefined ? undefined : { role: "read" },
    prologue: testCase.prologue,
  });
  if ("refusal" in resolution) throw new Error(`${testCase.name}: ${resolution.refusal}`);
  assertEquals(resolution.choice.id, testCase.id, testCase.name);
  return { plan, choice: resolution.choice };
};

const programOf = (
  testCase: Case,
  overrides: { readonly packedScale?: number | undefined } = {},
): LinearProgram => {
  const { plan, choice } = choiceOf(testCase);
  const [x, weight] = plan.inputShapes;
  const { m, n, k } = testCase.shape;
  return linearProgramFor(choice, {
    x,
    weight,
    m,
    n,
    k,
    weightStorage: testCase.storage,
    groupSize: testCase.groupSize,
    packedScale: "packedScale" in overrides ? overrides.packedScale : testCase.packedScale,
    prologue: testCase.prologue,
    geometryProfile: PROFILE,
    linearI8a8Dot: testCase.dot ?? "dp4a",
  });
};

/**
 * Whether a `weightScale` binding is bound for a weight of this storage. The rule found in the recipe
 * builder (`#weightScaleBindings` in src/runtime/recipe-builder.ts): no entry for a weight without a
 * resident record (raw f32) or resident as f16 / bf16, the one scale buffer for every other storage
 * a linear reads (i2 / i4 / i8 — the residency record pairs them with a scale). The WGSL declares
 * `wscale` on the same condition (`weightLoaderWgsl` in src/kernels/weight-storage.ts for the GEMM
 * skeleton, the GEMV bindings, and the kernels that read only integer storage).
 */
const hasScaleBuffer = (storage: WeightStorage): boolean =>
  storage === "i2" || storage === "i4" || storage === "i8";

/** The binding numbers a WGSL text declares in group 0, ascending. */
const declaredBindings = (wgsl: string): number[] =>
  [...wgsl.matchAll(/@group\(0\)\s*@binding\((\d+)\)/g)]
    .map((match) => Number(match[1]))
    .toSorted((a, b) => a - b);

describe("linearProgramFor: one representative linear per strategy", () => {
  it("covers every strategy of linear", () => {
    assertEquals(
      new Set<string>(CASES.map(({ id }) => id)),
      new Set<string>(Object.keys(LINEAR_STRATEGIES)),
    );
  });

  it("binds exactly the bindings each dispatch's WGSL declares besides the uniform, the weight's scale only where its storage has one", () => {
    for (const testCase of CASES) {
      for (const [index, dispatch] of programOf(testCase).dispatches.entries()) {
        const bound = dispatch.bindings
          .filter(({ operand }) => operand !== "weightScale" || hasScaleBuffer(testCase.storage))
          .map(({ binding }) => binding)
          .toSorted((a, b) => a - b);
        const declared = declaredBindings(dispatch.wgsl()).filter((binding) => binding !== 0);
        assertEquals(bound, declared, `${testCase.name}, dispatch ${index} (${dispatch.key})`);
      }
    }
  });

  it("binds only temporaries the program declares", () => {
    for (const testCase of CASES) {
      const program = programOf(testCase);
      for (const dispatch of program.dispatches) {
        for (const { operand } of dispatch.bindings) {
          if (typeof operand !== "object") continue;
          assert(
            Number.isInteger(operand.temp) &&
              operand.temp >= 0 &&
              operand.temp < program.temps.length,
            `${testCase.name}: ${dispatch.key} binds temporary ${operand.temp} of ${program.temps.length}`,
          );
        }
      }
    }
  });

  it("names each dispatch by the key its kernel module gives for the linear's facts, and generates the same WGSL each time", () => {
    for (const testCase of CASES) {
      const { dispatches } = programOf(testCase);
      assertEquals(
        dispatches.map(({ key }) => key),
        testCase.keys(),
        testCase.name,
      );
      for (const dispatch of dispatches) {
        assertEquals(dispatch.wgsl(), dispatch.wgsl(), `${testCase.name}: ${dispatch.key}`);
      }
    }
  });
});

describe("linearProgramFor: the loud failures that need no device", () => {
  it("refuses an i8 linear on the i8a8 path whose k passes the i32 reduction guard", () => {
    const k = 131072 + 16;
    const testCase: Case = {
      name: "i8a8 past the k guard",
      id: "i8a8",
      shape: { m: 4, n: 64, k },
      storage: "i8",
      knobs: { linearCompute: "a8" },
      keys: () => [],
    };
    assertThrows(
      () => programOf(testCase),
      ExecutionError,
      `linear i8a8 [4,${k}] × [64,${k}]: k=${k} exceeds the i32 reduction guard 131072 of the i8a8 path`,
    );
  });

  it("refuses a packed strategy built without the pairing's scale", () => {
    for (const id of ["integer-gemv", "gemv-parallel-packed"] as const) {
      const testCase = CASES.find((entry) => entry.id === id);
      assert(testCase !== undefined);
      assertThrows(
        () => programOf(testCase, { packedScale: undefined }),
        ExecutionError,
        `linear: the strategy ${id} was chosen without a packed pairing`,
      );
    }
  });
});
