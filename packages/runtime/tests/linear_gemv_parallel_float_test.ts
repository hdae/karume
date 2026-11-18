/**
 * The parallel GEMV for float-stored weights (f32 / f16 / bf16), without a GPU: what its kernels'
 * text is made of, which shapes it is eligible for, when the selection takes it and what the program
 * of the choice dispatches. The numbers are checked on a GPU in
 * gpu_linear_gemv_parallel_float_test.ts.
 */

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import {
  defaultLinearGemvVariant,
  type LinearGemvChoices,
  linearGemvKey,
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvParallelKey,
  type LinearGemvParallelLanes,
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvParallelParams,
  type LinearGemvParallelShape,
  linearGemvParallelStagedKey,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvStagedEligible,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeParams,
  linearGemvStaticQuantizeWgsl,
  linearGemvSubgroupWgsl,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import { FUSION_RULES, type FusionWeightLayout, planFusions } from "../src/runtime/fusion.ts";
import { linearProgramFor } from "../src/runtime/linear-program.ts";
import {
  LINEAR_STRATEGIES,
  type LinearChoice,
  type LinearKnobs,
  selectLinearStrategy,
} from "../src/runtime/linear-strategies.ts";
import { countUses, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

const FLOATS = ["bf16", "f16", "f32"] as const;
type Float = (typeof FLOATS)[number];
const LANES = [2, 4, 8, 16, 32] as const satisfies readonly LinearGemvParallelLanes[];
const DEFAULTS = DEFAULT_GEOMETRY_PROFILE.linearGemv;

/** The lines of a kernel that add to the accumulator. */
const macLines = (wgsl: string): string[] =>
  wgsl
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("acc = "));

describe("the text of the parallel GEMV for a float-stored weight", () => {
  it("is generated for every lane count, plain and staged, each under a key of its own, the same text for the same key", () => {
    const keys = new Map<string, string>();
    for (const storage of FLOATS) {
      const sequentialKey = linearGemvKey(storage, undefined, defaultLinearGemvVariant(DEFAULTS));
      for (const lanes of LANES) {
        for (const [key, wgsl] of [
          [
            linearGemvParallelKey(storage, undefined, lanes),
            linearGemvParallelWgsl(storage, undefined, lanes),
          ],
          [
            linearGemvParallelStagedKey(storage, undefined, lanes),
            linearGemvParallelStagedWgsl(storage, undefined, lanes),
          ],
        ] as const) {
          assertNotEquals(key, sequentialKey);
          assert(!keys.has(key), key);
          assert(![...keys.values()].includes(wgsl), key);
          keys.set(key, wgsl);
        }
        assertEquals(
          linearGemvParallelWgsl(storage, undefined, lanes),
          keys.get(linearGemvParallelKey(storage, undefined, lanes)),
        );
      }
    }
    assertEquals(keys.size, 30);
    assertEquals(linearGemvParallelKey("bf16", undefined, 16), "linear_gemv_parallel:wbf16:l16");
    assertEquals(linearGemvParallelKey("f32", undefined, 4), "linear_gemv_parallel:l4");
    assertEquals(
      linearGemvParallelStagedKey("f16", undefined, 32),
      "linear_gemv_parallel:wf16:l32:staged-x",
    );
  });

  it("multiplies the sequential kernel's operands in its order, spelled with fma (no acc + x * d is left to the compiler)", () => {
    for (const storage of FLOATS) {
      // One word of the sequential kernel: its remainder loop, whose slot is the parallel form's
      const sequential = macLines(linearGemvWgsl(storage, undefined, { cols: 32, unroll: 1 }));
      const word = sequential.filter((line) => line.includes("xt_") || line.includes("xft."));
      assert(word.length === (storage === "f32" ? 4 : 8), `${storage}: ${word.length}`);
      const asFma = word.map((line) => {
        const match = /^acc = acc \+ (\S+) \* (.+);$/.exec(line);
        if (match === null) throw new Error(`not a multiply-add: ${line}`);
        return `acc = fma(${match[1]}, ${match[2]}, acc);`;
      });
      for (const lanes of LANES) {
        for (const wgsl of [
          linearGemvParallelWgsl(storage, undefined, lanes),
          linearGemvParallelStagedWgsl(storage, undefined, lanes),
        ]) {
          assertEquals(macLines(wgsl), asFma, `${storage} l${lanes}`);
        }
      }
    }
  });

  it("declares no scale binding, reads its activation from workgroup memory when staged, and binds what the plain form binds", () => {
    for (const storage of FLOATS) {
      const plain = linearGemvParallelWgsl(storage, undefined, 8);
      const staged = linearGemvParallelStagedWgsl(storage, undefined, 8);
      for (const wgsl of [plain, staged]) {
        assert(!wgsl.includes("wscale"), storage);
        assertEquals(wgsl.match(/@binding\(\d\)/g), [
          "@binding(0)",
          "@binding(1)",
          "@binding(2)",
          "@binding(3)",
          "@binding(4)",
        ]);
      }
      // 8 lanes × the activation quads of one word (2 for the 8-element words, 1 for f32's 4)
      assert(
        staged.includes(`var<workgroup> sx: array<vec4<f32>, ${storage === "f32" ? 8 : 16}u>;`),
        storage,
      );
      // The only read of x in the staged form is the copy into workgroup memory
      assertEquals(staged.match(/\bx\[/g)?.length, 1, storage);
      assert(!plain.includes("sx["), storage);
    }
  });

  it("has no packed-activation, static-quantize or subgroup form: the text and the uniform of each fail naming the storage and the form (fault: asked for anyway)", () => {
    const refusal = (storage: Float, form: string): string =>
      `linear_gemv_parallel: ${storage} storage has no ${form} form (the plain and the staged-activation forms only)`;
    for (const storage of FLOATS) {
      for (const [form, ask] of [
        ["packed-activation", () => linearGemvParallelPackedWgsl(storage, undefined, 4)],
        ["packed-activation", () => linearGemvStaticQuantizePackedWgsl(storage, undefined, 4)],
        ["packed-activation", () => linearGemvParallelPackedParams(storage, 1, 256, 1536, 0.5)],
        ["static-quantize", () => linearGemvStaticQuantizeWgsl(storage, undefined, 4)],
        ["static-quantize", () => linearGemvStaticQuantizeParams(storage, 1, 256, 1536, 0.5)],
        ["subgroup", () => linearGemvSubgroupWgsl(storage, undefined, 4)],
      ] as const) {
        assertThrows(ask, CodegenError, refusal(storage, form));
      }
    }
    // The integer storages keep every form
    linearGemvSubgroupWgsl("i8", undefined, 4);
    linearGemvParallelPackedParams("i8", 1, 256, 1536, 0.5);
  });
});

describe("the uniform of the parallel GEMV", () => {
  it("takes 1 to 8 rows of a float-stored weight, where the sequential kernel's takes one", () => {
    for (const storage of FLOATS) {
      for (const m of [1, 4, 8]) {
        assertEquals([...linearGemvParallelParams(storage, m, 1536, 2048)].slice(0, 3), [
          m,
          1536,
          2048,
        ]);
      }
      assertEquals(
        linearGemvParallelParams(storage, 1, 1536, 2048),
        linearGemvParams(storage, 1, 1536, 2048),
      );
      assertThrows(() => linearGemvParams(storage, 4, 1536, 2048), CodegenError);
    }
  });

  it("is the sequential family's for an integer storage", () => {
    assertEquals(
      linearGemvParallelParams("i4", 8, 256, 1536, 32),
      linearGemvParams("i4", 8, 256, 1536, 32),
    );
  });

  it("refuses a row count outside 1 to 8 and a reduction length that is not whole words (fault: m = 9, k = 1540)", () => {
    assertThrows(() => linearGemvParallelParams("bf16", 9, 1536, 2048), CodegenError, "1..8");
    assertThrows(() => linearGemvParallelParams("i4", 9, 256, 1536, 32), CodegenError, "1..8");
    assertThrows(() => linearGemvParallelParams("bf16", 0, 1536, 2048), CodegenError);
    assertThrows(() => linearGemvParallelParams("bf16", 1, 1536, 1540), CodegenError, "multiple");
    assertThrows(() => linearGemvParallelParams("f32", 1, 1536, 1538), CodegenError, "multiple");
  });
});

/** The default choices with rows of float storages added (a shape per storage). */
const rowsOf = (...rows: readonly LinearGemvParallelShape[]): LinearGemvChoices => ({
  ...DEFAULTS,
  parallelShapes: [...DEFAULTS.parallelShapes, ...rows],
});

const BF16_ROW: LinearGemvParallelShape = {
  storage: "bf16",
  n: 1536,
  k: 2048,
  lanes: 32,
  packedActivations: false,
};
const STAGED_F16_ROW: LinearGemvParallelShape = {
  storage: "f16",
  n: 256,
  k: 4096,
  lanes: 16,
  packedActivations: false,
  stagedActivations: true,
};
const F32_ROW: LinearGemvParallelShape = {
  storage: "f32",
  n: 8960,
  k: 1536,
  lanes: 4,
  packedActivations: false,
};
const WITH_ROWS = rowsOf(BF16_ROW, STAGED_F16_ROW, F32_ROW);

describe("the eligibility of a float-stored shape", () => {
  it("is the lanes of the shape's row at 1 to 8 rows, and nothing past them, for another storage or for another shape", () => {
    for (const m of [1, 2, 4, 8]) {
      assertEquals(linearGemvParallelEligible(WITH_ROWS, "bf16", m, 1536, 2048), 32);
      assertEquals(linearGemvParallelEligible(WITH_ROWS, "f16", m, 256, 4096), 16);
      assertEquals(linearGemvParallelEligible(WITH_ROWS, "f32", m, 8960, 1536), 4);
    }
    assertEquals(linearGemvParallelEligible(WITH_ROWS, "bf16", 9, 1536, 2048), undefined);
    assertEquals(linearGemvParallelEligible(WITH_ROWS, "f16", 1, 1536, 2048), undefined);
    assertEquals(linearGemvParallelEligible(WITH_ROWS, "bf16", 1, 1536, 4096), undefined);
    // An i4 linear of the row's shape names a group, which no float row has
    assertEquals(linearGemvParallelEligible(WITH_ROWS, "i4", 1, 1536, 2048, 64), undefined);
  });

  it("is nothing without a row: the default profile names no float shape it was not measured for", () => {
    for (const storage of FLOATS) {
      assertEquals(linearGemvParallelEligible(DEFAULTS, storage, 1, 300, 1536), undefined);
    }
  });

  it("is staged only where the row says so, and never packed", () => {
    assertEquals(linearGemvStagedEligible(WITH_ROWS, "f16", 1, 256, 4096), 16);
    assertEquals(linearGemvStagedEligible(WITH_ROWS, "bf16", 1, 1536, 2048), undefined);
    for (const row of [BF16_ROW, STAGED_F16_ROW, F32_ROW]) {
      assertEquals(linearGemvPackedEligible(WITH_ROWS, row.storage, 1, row.n, row.k), undefined);
      // Not even from a table that marks the row packed (one the profile's guard refuses)
      const marked = rowsOf({ ...row, packedActivations: true });
      assertEquals(linearGemvPackedEligible(marked, row.storage, 1, row.n, row.k), undefined);
    }
  });

  it("stays nothing for a reduction length that is not whole words or an output that is not whole quads (fault: rows that name such shapes)", () => {
    const odd = rowsOf(
      { storage: "bf16", n: 1536, k: 2044, lanes: 4, packedActivations: false },
      { storage: "f32", n: 1538, k: 2048, lanes: 4, packedActivations: false },
    );
    assertEquals(linearGemvParallelEligible(odd, "bf16", 1, 1536, 2044), undefined);
    assertEquals(linearGemvParallelEligible(odd, "f32", 1, 1538, 2048), undefined);
  });
});

describe("a geometry profile with rows of float storages", () => {
  const profileOf = (row: Record<string, unknown>): GeometryProfile => ({
    ...DEFAULT_GEOMETRY_PROFILE,
    id: "test",
    linearGemv: {
      ...DEFAULTS,
      parallelShapes: [row as unknown as LinearGemvParallelShape],
    },
  });

  it("is accepted with plain and staged rows of each float storage", () => {
    assertGeometryProfile({ ...DEFAULT_GEOMETRY_PROFILE, id: "test", linearGemv: WITH_ROWS });
  });

  it("is refused when a float row names a group or packed activations, naming the row (fault: each added to a valid row)", () => {
    assertThrows(
      () => assertGeometryProfile(profileOf({ ...BF16_ROW, group: 32 })),
      CodegenError,
      "linearGemv.parallelShapes[0]: a row of bf16 storage takes no group and no packedActivations",
    );
    assertThrows(
      () => assertGeometryProfile(profileOf({ ...F32_ROW, packedActivations: true })),
      CodegenError,
      "linearGemv.parallelShapes[0]: a row of f32 storage takes no group and no packedActivations",
    );
  });
});

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

const KNOBS: LinearKnobs = {
  linearCompute: "f32",
  linearGemvReduce: "sequential",
  integerGemv: false,
  linearGemvPlanes: false,
  linearGemvRowsThreadTarget: undefined,
  linearSplitK: false,
};

const choose = (
  storage: Float,
  m: number,
  n: number,
  k: number,
  knobs: Partial<LinearKnobs> = {},
  linearGemv: LinearGemvChoices = WITH_ROWS,
): LinearChoice =>
  selectLinearStrategy({
    plan: linearPlan(m, n, k),
    weightStorage: storage,
    groupSize: undefined,
    knobs: { ...KNOBS, ...knobs },
    linearGemv,
    packed: undefined,
    prologue: undefined,
  });

const REORDER = { linearGemvReduce: "parallel" } as const;

describe("the selection of a float-stored linear", () => {
  it("keeps the exact kernels at exact whatever the profile's rows: the sequential GEMV at one row, the GEMM above", () => {
    for (const { storage, n, k } of [BF16_ROW, STAGED_F16_ROW, F32_ROW]) {
      assertEquals(choose(storage as Float, 1, n, k).id, "gemv-m1");
      for (const m of [2, 4, 8, 9]) {
        const { id } = choose(storage as Float, m, n, k);
        assertEquals(id, "gemm-f32", `${storage} M${m}`);
        assertEquals(LINEAR_STRATEGIES[id].level, "exact");
      }
    }
  });

  it("takes the parallel GEMV at reorder for 1 to 8 rows of a shape with a row, with the row's lanes and staged flag", () => {
    for (const m of [1, 2, 4, 8]) {
      const plain = choose("bf16", m, 1536, 2048, REORDER);
      assertEquals(plain.id, "gemv-parallel", `M${m}`);
      assert("lanes" in plain && plain.lanes === 32);
      const staged = choose("f16", m, 256, 4096, REORDER);
      assertEquals(staged.id, "gemv-parallel-staged", `M${m}`);
      assert("lanes" in staged && staged.lanes === 16);
      assertEquals(LINEAR_STRATEGIES[plain.id].level, "reorder");
    }
  });

  it("leaves 9 rows and more on the GEMM, and a shape without a row on the exact kernels, at reorder too", () => {
    assertEquals(choose("bf16", 9, 1536, 2048, REORDER).id, "gemm-f32");
    assertEquals(choose("bf16", 64, 1536, 2048, REORDER).id, "gemm-f32");
    assertEquals(choose("bf16", 1, 1536, 4096, REORDER).id, "gemv-m1");
    assertEquals(choose("bf16", 4, 1536, 4096, REORDER).id, "gemm-f32");
    // The default profile's rows are the measured ones only
    assertEquals(choose("f32", 1, 300, 1536, REORDER, DEFAULTS).id, "gemv-m1");
  });

  it("takes the plain or staged form under the subgroup option, which has no float kernel", () => {
    const subgroup = { linearGemvReduce: "parallel-subgroup32" } as const;
    assertEquals(choose("bf16", 1, 1536, 2048, subgroup).id, "gemv-parallel");
    assertEquals(choose("f16", 4, 256, 4096, subgroup).id, "gemv-parallel-staged");
  });

  it("keeps f16 compute on the f16 GEMM: the parallel GEMV computes in f32", () => {
    assertEquals(
      choose("bf16", 1, 1536, 2048, { ...REORDER, linearCompute: "f16" }).id,
      "gemm-f16",
    );
  });
});

describe("the linear → static_quantize fusion on a float-stored weight", () => {
  /** `linear [1, k] × [n, k]` → `static_quantize`, as the fusion pass plans it with `layouts`. */
  const fused = (
    n: number,
    k: number,
    layouts: ReadonlyMap<string, FusionWeightLayout>,
    linearGemv: LinearGemvChoices,
  ): number => {
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
    return planFusions(planGraph(ir, {}).nodes, {
      rules: FUSION_RULES,
      linearGemv,
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      limits: {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      },
      linearGemvReduce: "parallel",
      linearCompute: "f32",
      weightLayouts: layouts,
    }).counts.linearStaticQuantize;
  };

  it("does not fuse where the table names the float shape: the fused epilogue is the integer storages' (fault: a bf16, an f16 and an f32 weight on a row of its own)", () => {
    assertEquals(fused(1536, 2048, new Map([["w", { storage: "bf16" }]]), WITH_ROWS), 0);
    assertEquals(fused(256, 4096, new Map([["w", { storage: "f16" }]]), WITH_ROWS), 0);
    // An f32-resident weight has no entry in the layouts
    assertEquals(fused(8960, 1536, new Map(), WITH_ROWS), 0);
    // The control: the same pass fuses the i4 linear of the default table's first row
    assertEquals(
      fused(8960, 1536, new Map([["w", { storage: "i4", groupSize: 32 }]]), WITH_ROWS),
      1,
    );
  });
});

describe("the program of the parallel GEMV for a float-stored weight", () => {
  const facts = (storage: Float, m: number, n: number, k: number) =>
    ({
      x: [m, k],
      weight: [n, k],
      m,
      n,
      k,
      weightStorage: storage,
      groupSize: undefined,
      packedScale: undefined,
      prologue: undefined,
      geometryProfile: DEFAULT_GEOMETRY_PROFILE,
      linearI8a8Dot: "emu",
    }) as const;

  it("is one dispatch of the row's kernel over ceil(n ÷ columns per workgroup) × m workgroups, with the three-word uniform", () => {
    for (const m of [1, 8]) {
      const plain = linearProgramFor(
        choose("bf16", m, 1536, 2048, REORDER),
        facts("bf16", m, 1536, 2048),
      );
      assertEquals(plain.temps, []);
      assertEquals(plain.dispatches.length, 1);
      const [dispatch] = plain.dispatches;
      assertEquals(dispatch.key, "linear_gemv_parallel:wbf16:l32");
      assertEquals(dispatch.wgsl(), linearGemvParallelWgsl("bf16", undefined, 32));
      assertEquals([...dispatch.params].slice(0, 3), [m, 1536, 2048]);
      assertEquals(dispatch.grid, {
        kind: "tiled",
        axes: [
          { items: 1536, tile: 4 },
          { items: m, tile: 1 },
          { items: 1, tile: 1 },
        ],
      });
      const staged = linearProgramFor(
        choose("f16", m, 256, 4096, REORDER),
        facts("f16", m, 256, 4096),
      );
      assertEquals(staged.dispatches[0].key, "linear_gemv_parallel:wf16:l16:staged-x");
      assertEquals(staged.dispatches[0].wgsl(), linearGemvParallelStagedWgsl("f16", undefined, 16));
    }
  });
});
