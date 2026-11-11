/**
 * The byte-plane GEMV (`SessionOptions.linearGemvPlanes`) without a GPU: the kernel's eligibility,
 * key, params and WGSL contract, the option check, the one eligibility function
 * (`linearPlanesGemvVariant`), the fusion plan around a byte-plane linear, and the linear recipe's
 * route — read off a `StepRecipeBuilder` with a stand-in for the GPU surface. The kernel's numerics
 * on a GPU are gpu_linear_gemv_planes_test.ts.
 */

import { assert, assertEquals, assertNotEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError, DispatchLimitError } from "../src/codegen/errors.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import {
  defaultLinearGemvRowsVariant,
  linearGemvKey,
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvRowsKey,
} from "../src/kernels/linear-gemv.ts";
import { integerGemvVariantFor } from "../src/kernels/linear-gemv-integer.ts";
import {
  planesGemvKey,
  type PlanesGemvLanes,
  planesGemvParams,
  type PlanesGemvRows,
  type PlanesGemvVariant,
  planesGemvVariantFor,
  planesGemvWgsl,
  planesGemvWorkgroups,
} from "../src/kernels/linear-gemv-planes.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import {
  FUSION_RULES,
  type FusionPlan,
  type FusionWeightLayout,
  planFusions,
} from "../src/runtime/fusion.ts";
import { linearPlanesGemvVariant } from "../src/runtime/linear-route.ts";
import { countUses, ExecutionError, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import { type BindingSource, StepRecipeBuilder } from "../src/runtime/recipe.ts";
import type { RecipeBuildFace } from "../src/runtime/recipe-builder.ts";
import { buildLinear } from "../src/runtime/recipe-builders/linear.ts";
import { sessionOptionsViolation } from "../src/runtime/session-options-check.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";

/** The QAT E2B lm_head, the measured shape, and its variant. */
const LM_HEAD = { n: 262144, k: 1536, variant: { lanes: 8, rows: 4 } } as const;

/** Every (lanes, rows) the generator accepts: rows ≤ lanes (one writer per row). */
const VARIANTS: readonly PlanesGemvVariant[] = ([4, 8, 16, 32] as const).flatMap((lanes) =>
  ([1, 2, 4, 8] as const).filter((rows) => rows <= lanes).map((rows) => ({ lanes, rows })),
);

const MS = [1, 2, 3, 4, 5, 6, 7, 8] as const;

/** A variant with values outside the type (what a JS caller can pass). */
const looseVariant = (lanes: number, rows: number): PlanesGemvVariant => {
  const variant: { lanes: PlanesGemvLanes; rows: PlanesGemvRows } = { lanes: 4, rows: 1 };
  Reflect.set(variant, "lanes", lanes);
  Reflect.set(variant, "rows", rows);
  return variant;
};

describe("byte-plane GEMV kernel: eligibility (planesGemvVariantFor)", () => {
  it("names the measured variant of the QAT E2B lm_head (i2 262144 × 1536) at M = 1..8", () => {
    for (const m of MS) {
      assertEquals(planesGemvVariantFor("i2", m, LM_HEAD.n, LM_HEAD.k), LM_HEAD.variant, `M=${m}`);
    }
  });

  it("names nothing for another storage, M outside 1..8, or an unmeasured shape", () => {
    for (const storage of ["i4", "i8", "f16", "bf16", "f32"] as const) {
      assertEquals(planesGemvVariantFor(storage, 1, LM_HEAD.n, LM_HEAD.k), undefined, storage);
    }
    for (const m of [0, 9]) {
      assertEquals(planesGemvVariantFor("i2", m, LM_HEAD.n, LM_HEAD.k), undefined, `M=${m}`);
    }
    for (const [n, k] of [
      [LM_HEAD.n, 1552],
      [LM_HEAD.n, 1544],
      [LM_HEAD.n, 0],
      [131072, LM_HEAD.k],
      [12288, LM_HEAD.k],
      [LM_HEAD.k, LM_HEAD.n],
    ]) {
      assertEquals(planesGemvVariantFor("i2", 1, n, k), undefined, `${n}×${k}`);
    }
  });
});

describe("byte-plane GEMV kernel: keys and WGSL", () => {
  it("gives byte-identical WGSL and the same key for the same variant", () => {
    for (const variant of VARIANTS) {
      const copy = { ...variant };
      assertEquals(planesGemvWgsl(variant), planesGemvWgsl(copy));
      assertEquals(planesGemvKey(variant), planesGemvKey(copy));
    }
  });

  it("gives every variant its own key and its own WGSL", () => {
    assertEquals(VARIANTS.length, 15);
    assertEquals(new Set(VARIANTS.map(planesGemvKey)).size, 15);
    assertEquals(new Set(VARIANTS.map(planesGemvWgsl)).size, 15);
  });

  it("names the storage and the geometry in the key", () => {
    assertEquals(planesGemvKey(LM_HEAD.variant), "linear_gemv_planes:v1:wi2:l8r4");
    assertEquals(planesGemvKey({ lanes: 4, rows: 1 }), "linear_gemv_planes:v1:wi2:l4r1");
    assertEquals(planesGemvKey({ lanes: 32, rows: 8 }), "linear_gemv_planes:v1:wi2:l32r8");
  });

  it("refuses a variant outside the generated table", () => {
    for (const variant of [looseVariant(2, 1), looseVariant(64, 1), looseVariant(8, 16)]) {
      assertThrows(() => planesGemvKey(variant), CodegenError, "invalid variant");
      assertThrows(() => planesGemvWgsl(variant), CodegenError, "invalid variant");
    }
    assertThrows(
      () => planesGemvKey({ lanes: 4, rows: 8 }),
      CodegenError,
      "rows must not exceed lanes",
    );
  });
});

describe("byte-plane GEMV kernel: params and grid", () => {
  it("lays out m, n, k as the linear's uniform", () => {
    assertEquals([...planesGemvParams(1, LM_HEAD.n, LM_HEAD.k)], [1, LM_HEAD.n, LM_HEAD.k, 0]);
    assertEquals([...planesGemvParams(8, 37, 16)], [8, 37, 16, 0]);
  });

  it("refuses M outside 1..8 and a K that is not a positive multiple of 16", () => {
    for (const m of [0, 9, 1.5]) {
      assertThrows(() => planesGemvParams(m, 16, 16), CodegenError, "m must be in 1..8");
    }
    for (const k of [0, 24, -16, 1544]) {
      assertThrows(
        () => planesGemvParams(1, 16, k),
        CodegenError,
        "k must be a positive multiple of 16",
      );
    }
  });

  it("covers n channels with (128 / lanes) · rows channels per workgroup", () => {
    // The lm_head: 262144 / (16 groups × 4 channels).
    assertEquals(planesGemvWorkgroups(LM_HEAD.n, LM_HEAD.variant), 4096);
    assertEquals(planesGemvWorkgroups(37, { lanes: 4, rows: 1 }), 2);
    assertEquals(planesGemvWorkgroups(1029, { lanes: 32, rows: 8 }), 33);
  });
});

describe("linearGemvPlanes option check", () => {
  it("accepts linearGemvPlanes alone or with any other option, and false anywhere", () => {
    assertEquals(sessionOptionsViolation({ linearGemvPlanes: true }), undefined);
    assertEquals(sessionOptionsViolation({ linearGemvPlanes: false }), undefined);
    assertEquals(
      sessionOptionsViolation({
        linearGemvReduce: "parallel",
        fuseLinearStaticQuantize: true,
        packedStaticQuantize: true,
        integerGemv: true,
        linearGemvPlanes: true,
      }),
      undefined,
    );
  });

  it("rejects a value that is not a boolean, without converting it", () => {
    let conversions = 0;
    const object = {
      [Symbol.toPrimitive](): boolean {
        conversions++;
        return true;
      },
    };
    for (const value of [null, 1, "true", [], object]) {
      const options: SessionOptions = {};
      Object.defineProperty(options, "linearGemvPlanes", { value, enumerable: true });
      assertEquals(sessionOptionsViolation(options), "options.linearGemvPlanes must be a boolean");
    }
    assertEquals(conversions, 0);
  });
});

// ---------------------------------------------------------------------------
// The plan and the recipe
// ---------------------------------------------------------------------------

const X_SCALE = Math.fround(0.0137);
const OUTPUT_SCALE = Math.fround(0.021);

type GraphOptions = {
  readonly n: number;
  readonly k: number;
  readonly m?: number;
  /** A `static_quantize` producing the linear's input (the packed pairing's producer). */
  readonly inputQuantize?: boolean;
  /** A `static_quantize` after the linear (the linear→SRQ fusion's window). */
  readonly outputQuantize?: boolean;
};

/**
 * `[src → static_quantize →] x → linear → y [→ static_quantize → z]`. The weight is declared
 * borrowed (`shared`), so no payload is built for the 100 MB lm_head: the plan reads only shapes,
 * the fusion pass reads the storage from `weightLayouts` and the recipe from the face.
 */
const planesGraph = ({
  n,
  k,
  m = 1,
  inputQuantize = false,
  outputQuantize = false,
}: GraphOptions): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: inputQuantize || outputQuantize ? ["linear", "static_quantize"] : ["linear"] },
  symbols: [],
  inputs: [{ name: inputQuantize ? "src" : "x", dtype: "f32", shape: [m, k] }],
  outputs: [outputQuantize ? "z" : "y"],
  initializers: { w: { shared: true }, b: {} },
  values: {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    y: { dtype: "f32", shape: [m, n] },
    ...(inputQuantize ? { x: { dtype: "f32", shape: [m, k] } } : {}),
    ...(outputQuantize ? { z: { dtype: "f32", shape: [m, n] } } : {}),
  },
  nodes: [
    ...(inputQuantize
      ? [{ op: "static_quantize", ins: ["src"], outs: ["x"], attrs: { scale: X_SCALE } }]
      : []),
    { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
    ...(outputQuantize
      ? [{ op: "static_quantize", ins: ["y"], outs: ["z"], attrs: { scale: OUTPUT_SCALE } }]
      : []),
  ],
});

const nodesOf = (options: GraphOptions): readonly NodePlan[] =>
  planGraph(mergeGraph(planesGraph(options)), {}).nodes;

/** The graph's linear node plan (the eligibility function's and the recipe's input). */
const linearPlan = (options: GraphOptions): NodePlan => {
  const found = nodesOf(options).find((plan) => plan.node.op === "linear");
  if (found === undefined) throw new Error("the graph has no linear");
  return found;
};

const lmHead = (overrides: Partial<GraphOptions> = {}): GraphOptions => ({
  n: LM_HEAD.n,
  k: LM_HEAD.k,
  ...overrides,
});

describe("linearPlanesGemvVariant (the one eligibility function)", () => {
  it("names the variant of an i2 lm_head at M = 1..8 when the option is on", () => {
    for (const m of MS) {
      assertEquals(
        linearPlanesGemvVariant(true, linearPlan(lmHead({ m })), "i2"),
        LM_HEAD.variant,
        `M=${m}`,
      );
    }
  });

  it("names nothing with the option off, another storage, M > 8, another shape or another op", () => {
    const plan = linearPlan(lmHead());
    assertEquals(linearPlanesGemvVariant(false, plan, "i2"), undefined);
    assertEquals(linearPlanesGemvVariant(undefined, plan, "i2"), undefined);
    for (const storage of ["i4", "i8", "f16", "bf16", "f32", undefined] as const) {
      assertEquals(linearPlanesGemvVariant(true, plan, storage), undefined, String(storage));
    }
    assertEquals(linearPlanesGemvVariant(true, linearPlan(lmHead({ m: 9 })), "i2"), undefined);
    const gateUp = linearPlan({ n: 12288, k: 1536 });
    assertEquals(linearPlanesGemvVariant(true, gateUp, "i2"), undefined, "i2 12288×1536");
    const quantize = nodesOf(lmHead({ inputQuantize: true })).find(
      (node) => node.node.op === "static_quantize",
    );
    assert(quantize !== undefined);
    assertEquals(linearPlanesGemvVariant(true, quantize, "i2"), undefined);
  });
});

type Context = Parameters<typeof planFusions>[1];

/** The fusion plan of a graph, its weight i2-resident, under `context`. */
const planOf = (options: GraphOptions, context: Partial<Context> = {}): FusionPlan => {
  const ir = mergeGraph(planesGraph(options));
  const layout: FusionWeightLayout = { storage: "i2" };
  return planFusions(planGraph(ir, {}).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    weightLayouts: new Map([["w", layout]]),
    ...context,
  });
};

/** The whole plan as comparable text (each dispatch's WGSL and params written out). */
const planText = (plan: FusionPlan): string =>
  JSON.stringify(
    {
      counts: plan.counts,
      steps: plan.steps.map((step) =>
        step.kind === "fused"
          ? {
              ...step,
              dispatches: step.dispatches.map((dispatch) => ({
                ...dispatch,
                wgsl: dispatch.wgsl(),
                params: [...dispatch.params],
              })),
            }
          : step,
      ),
    },
    (_key, value: unknown) => (typeof value === "function" ? "[function]" : value),
  );

/** The options under which the packed pairing and the linear→SRQ rule look at a linear. */
const FUSING: Partial<Context> = {
  linearGemvReduce: "parallel",
  linearCompute: "f32",
  fuseLinearStaticQuantize: true,
  packedStaticQuantize: true,
  integerGemv: true,
};

describe("the fusion pass and a byte-plane linear", () => {
  it("finds the lm_head in no parallel GEMV, packed or integer GEMV row at M = 1..8", () => {
    // So no fusion rule or pairing claims it today, with the option or without: the guards that
    // keep them off a byte-plane linear reject no window and pair nothing they would otherwise.
    for (const m of MS) {
      assertEquals(linearGemvParallelEligible("i2", m, LM_HEAD.n, LM_HEAD.k), undefined);
      assertEquals(linearGemvPackedEligible("i2", m, LM_HEAD.n, LM_HEAD.k), undefined);
      assertEquals(integerGemvVariantFor("i2", undefined, m, LM_HEAD.n, LM_HEAD.k), undefined);
    }
  });

  it("leaves the producer SRQ f32 and the SRQ after it a node, the option on or off", () => {
    const options = lmHead({ inputQuantize: true, outputQuantize: true });
    const off = planOf(options, FUSING);
    const on = planOf(options, { ...FUSING, linearGemvPlanes: true });
    assertEquals(on.counts.packedStaticQuantize, 0);
    assertEquals(on.counts.linearStaticQuantize, 0);
    assertEquals(
      on.steps.map((step) => (step.kind === "node" ? step.plan.node.op : step.rule)),
      ["static_quantize", "linear", "static_quantize"],
    );
    for (const step of on.steps) {
      assert(step.kind !== "node" || step.packedActivations === undefined, "nothing packed");
    }
    assertEquals(planText(on), planText(off));
  });
});

describe("linearGemvPlanes off: the plan does not change", () => {
  const graphs: readonly (readonly [string, GraphOptions])[] = [
    ["the lm_head", lmHead()],
    ["the lm_head between two SRQs", lmHead({ inputQuantize: true, outputQuantize: true })],
    ["the lm_head at M = 8", lmHead({ m: 8, outputQuantize: true })],
    // A parallel GEMV row with a packed producer and the linear→SRQ fusion (down 1536×12288).
    [
      "an i2 down between two SRQs",
      { n: 1536, k: 12288, inputQuantize: true, outputQuantize: true },
    ],
  ];
  const contexts: readonly Partial<Context>[] = [{}, FUSING];

  it("plans exactly as without the field when the option is false", () => {
    for (const [label, options] of graphs) {
      for (const context of contexts) {
        assertEquals(
          planText(planOf(options, { ...context, linearGemvPlanes: false })),
          planText(planOf(options, context)),
          `${label} ${JSON.stringify(context)}`,
        );
      }
    }
  });

  it("plans exactly as with the option off when it is on: the route is the recipe's", () => {
    for (const [label, options] of graphs) {
      for (const context of contexts) {
        assertEquals(
          planText(planOf(options, { ...context, linearGemvPlanes: true })),
          planText(planOf(options, context)),
          `${label} ${JSON.stringify(context)}`,
        );
      }
    }
  });
});

type Route = {
  readonly key: string;
  /** The WGSL the pipeline cache was asked to compile for `key`. */
  readonly wgsl: string | undefined;
  readonly params: readonly number[];
  readonly bindings: readonly { readonly binding: number; readonly source: BindingSource }[];
  readonly workgroups: unknown;
};

const SCALE_SOURCE: BindingSource = { kind: "value", name: "w.scale" };

/**
 * The surface the linear builder reads, for an i2-resident weight. GPU objects are opaque
 * placeholders the builder only passes through; the pipeline cache records the WGSL of each key it
 * is asked for, and `writeParams` the uniforms.
 */
const linearFace = (
  linearGemvPlanes: boolean,
  record: { readonly compiled: Map<string, string>; readonly params: Uint32Array[] },
  { storage = "i2", limit = 65535 }: { storage?: WeightStorage; limit?: number } = {},
): RecipeBuildFace => {
  const opaque = {};
  const face = {
    state: {
      gpu: { limits: { maxComputeWorkgroupsPerDimension: limit } },
      cache: {
        get: async (key: string, wgsl: string) => {
          record.compiled.set(key, wgsl);
          return { pipeline: opaque, layout: opaque, roles: opaque };
        },
      },
      linearCompute: "f32",
      linearGemvReduce: "sequential",
      integerGemv: false,
      linearGemvPlanes,
      linearGemvRowsThreadTarget: undefined,
      geometryProfile: DEFAULT_GEOMETRY_PROFILE,
    },
    writeParams: (params: Uint32Array<ArrayBuffer>) => {
      record.params.push(params);
      return opaque;
    },
    weightStorage: () => storage,
    weightLayout: () => undefined,
    weightScaleBindings: (_step: NodePlan, binding: number) => [{ binding, source: SCALE_SOURCE }],
    weightGroupSize: () => {
      throw new Error("an i2 weight has no group length");
    },
  };
  // MUST: only the fields above are read; a new read reaches `undefined` and fails the test.
  return face as unknown as RecipeBuildFace;
};

const BINDS: readonly BindingSource[] = [
  { kind: "value", name: "x" },
  { kind: "value", name: "w" },
  { kind: "value", name: "b" },
];
const OUTS: readonly BindingSource[] = [{ kind: "value", name: "y" }];

/** The one dispatch `buildLinear` queues for the graph's linear. */
const routeOf = async (
  options: GraphOptions,
  linearGemvPlanes: boolean,
  face: { storage?: WeightStorage; limit?: number } = {},
): Promise<Route> => {
  const record = { compiled: new Map<string, string>(), params: [] as Uint32Array[] };
  const builder = new StepRecipeBuilder();
  await buildLinear(
    linearFace(linearGemvPlanes, record, face),
    linearPlan(options),
    BINDS,
    OUTS,
    builder,
    undefined,
    undefined,
  );
  assertEquals(builder.dispatches.length, 1, "one dispatch per linear");
  assertEquals(record.params.length, 1, "one uniform per dispatch");
  const [dispatch] = builder.dispatches;
  return {
    key: dispatch.key,
    wgsl: record.compiled.get(dispatch.key),
    params: [...record.params[0]],
    bindings: dispatch.bindings,
    workgroups: dispatch.workgroups,
  };
};

describe("the linear recipe's route", () => {
  it("runs an i2 lm_head at M = 1..8 on the byte-plane GEMV with the option on", async () => {
    const { n, k, variant } = LM_HEAD;
    for (const m of [1, 8]) {
      assertEquals(
        await routeOf(lmHead({ m }), true),
        {
          key: planesGemvKey(variant),
          wgsl: planesGemvWgsl(variant),
          params: [...planesGemvParams(m, n, k)],
          // The linear's numbering: 1 x / 2 w / 3 bias / 4 out / 5 the per-channel scale.
          bindings: [
            { binding: 1, source: BINDS[0] },
            { binding: 2, source: BINDS[1] },
            { binding: 3, source: BINDS[2] },
            { binding: 4, source: OUTS[0] },
            { binding: 5, source: SCALE_SOURCE },
          ],
          workgroups: [4096, m, 1],
        },
        `M=${m}`,
      );
    }
  });

  it("keeps the sequential GEMV (M = 1) and the row-block variant (M = 8) with the option off", async () => {
    const off1 = await routeOf(lmHead(), false);
    assertEquals(off1.key, linearGemvKey("i2"));
    assertEquals(off1.key, "linear_gemv:v1:f32:c32u4:wi2");
    const off8 = await routeOf(lmHead({ m: 8 }), false);
    assertEquals(
      off8.key,
      linearGemvRowsKey("i2", undefined, defaultLinearGemvRowsVariant("i2", 8, LM_HEAD.n)),
    );
    assertNotEquals(off1, await routeOf(lmHead(), true));
  });

  it("routes every linear it does not name exactly as with the option off", async () => {
    const ineligible: readonly (readonly [string, GraphOptions, WeightStorage])[] = [
      ["M = 9", lmHead({ m: 9 }), "i2"],
      ["an i2 gate / up 12288×1536", { n: 12288, k: 1536 }, "i2"],
      ["an i8 lm_head", lmHead(), "i8"],
    ];
    for (const [label, options, storage] of ineligible) {
      assertEquals(
        await routeOf(options, true, { storage }),
        await routeOf(options, false, { storage }),
        label,
      );
    }
  });

  it("fails loudly on a byte-plane linear declared to read packed activations", async () => {
    const record = { compiled: new Map<string, string>(), params: [] as Uint32Array[] };
    await assertRejects(
      () =>
        buildLinear(
          linearFace(true, record),
          linearPlan(lmHead()),
          BINDS,
          OUTS,
          new StepRecipeBuilder(),
          { role: "read", scale: X_SCALE },
          undefined,
        ),
      ExecutionError,
      "is a byte-plane GEMV linear, which reads f32 activations",
    );
  });

  it("fails loudly when the grid passes the device's workgroup limit", async () => {
    await assertRejects(
      () => routeOf(lmHead(), true, { limit: 4095 }),
      DispatchLimitError,
      "linear gemv planes [1,1536] × [262144,1536]",
    );
  });
});
