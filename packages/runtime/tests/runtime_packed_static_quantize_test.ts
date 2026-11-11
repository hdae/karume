/**
 * The check of **pairing and generated code** for packed int8 activations (ADR 0105) (no GPU
 * needed).
 *
 * Pairing holds only as a pair of "the producer SRQ writes packed" and "the consumer linear reads
 * packed". If only one side is set, f32 words flow into a `vec4<u32>` binding = a silent wrong
 * value with no exception, so both the accepted set (option, consumer, storage, shape) and the
 * number of activation loads in the generated WGSL are pinned.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { IrGraph } from "../src/format/ir.ts";
import { mergedGraph } from "../src/format/container/bind.ts";
import { FUSION_RULES, type FusionWeightLayout, planFusions } from "../src/runtime/fusion.ts";
import { countUses, planGraph } from "../src/runtime/plan.ts";
import {
  linearGemvPackedEligible,
  linearGemvParallelEligible,
  linearGemvParallelPackedKey,
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvParallelWgsl,
  linearGemvStaticQuantizeKey,
  linearGemvStaticQuantizePackedKey,
  linearGemvStaticQuantizePackedParams,
} from "../src/kernels/linear-gemv.ts";
import {
  staticQuantizePackedParams,
  staticQuantizeParams,
} from "../src/kernels/static-quantize.ts";
import { mergeGraph, type StorageMap, type StorageSpec } from "./helpers/merged-graph.ts";
import {
  linearStaticQuantizeModel,
  type LinearStaticQuantizeOptions,
} from "./helpers/linear-static-quantize-graph.ts";
import { type DeclarationJson, GRAPH_NAME } from "./helpers/model-fixture.ts";

type Storage = "i2" | "i4" | "i8";

/**
 * One row of the measured table (`PARALLEL_SHAPES`) — points to the consumer linear's shape as is.
 */
type Shape = {
  readonly storage: Storage;
  readonly n: number;
  readonly k: number;
  readonly group?: number;
  readonly lanes: 2 | 4 | 8 | 16 | 32;
};

/**
 * Rows with `packedActivations: true` (the shapes that helped in the per-key measurements of ADR
 * 0105 addendum 1). All are "long K, 32 lanes" = shapes where one thread reads many activations per
 * weight word.
 */
const PACKED_SHAPES: readonly Shape[] = [
  { storage: "i2", n: 1536, k: 12288, lanes: 32 },
  { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32 },
  { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32 },
  { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32 },
];

/**
 * A row with `packedActivations: false` — it falls back to parallel GEMV but is not made packed.
 */
const PLAIN_SHAPES: readonly Shape[] = [
  { storage: "i2", n: 12288, k: 1536, lanes: 2 },
  { storage: "i4", n: 6144, k: 1536, group: 512, lanes: 4 },
  { storage: "i8", n: 256, k: 1536, lanes: 32 },
  // Even with the same (storage, n, k), a different group length is a different row = a different
  // decision (used by `plainConsumer` below).
  { storage: "i4", n: 1536, k: 2048, group: 32, lanes: 32 },
];

/**
 * The default consumer — representative of the measured shapes where packed helps (the i2 down
 * shape).
 */
const DEFAULT_SHAPE = PACKED_SHAPES[0];

type Options = {
  /** The consumer linear's shape (a row of the measured table). */
  shape?: Shape;
  m?: number;
  scale?: number;
  /** Adds a consumer other than linear (neg) to the SRQ output. */
  extraConsumer?: boolean;
  /**
   * Also feeds the SRQ output to a linear of **a shape that does not fall back to parallel GEMV**.
   */
  unparallelConsumer?: boolean;
  /**
   * Also feeds the SRQ output to a linear of **a row with the flag false** (which does fall back to
   * parallel GEMV).
   */
  plainConsumer?: boolean;
  /** Consumes the SRQ output in a linear's **weight** slot (a slot other than the activation). */
  weightConsumer?: boolean;
  /** Makes the SRQ output a graph output. */
  publicQuantized?: boolean;
  /** Puts a linear right before the SRQ so the linear→SRQ fusion (ADR 0103) swallows it. */
  afterLinear?: boolean;
};

const GROUP = 512;
/**
 * The contraction length of the preceding linear (`afterLinear`) — `i2 12288×1536` is a measured
 * shape in PARALLEL_SHAPES.
 */
const PRE_K = 1536;

/**
 * Storage of one weight (the IR v2 declaration does not carry it — a binding-table field; the group
 * length comes from the measured table's row).
 */
const weightStorage = (shape: Shape): StorageSpec =>
  shape.storage === "i4"
    ? { codec: "int4-sym-g", groupSize: shape.group ?? GROUP }
    : { codec: shape.storage === "i2" ? "int2-off" : "int8-sym" };

/** The minimal `x → static_quantize → linear` form (the consumer can be broken through options). */
const packedGraph = (o: Options = {}): DeclarationJson => {
  const shape = o.shape ?? DEFAULT_SHAPE,
    m = o.m ?? 1,
    k = shape.k,
    n = shape.n;
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "static_quantize", "neg"] },
    symbols: [],
    inputs: [{ name: "src", dtype: "f32", shape: [m, o.afterLinear ? PRE_K : k] }],
    outputs: ["y", ...(o.publicQuantized ? ["xq"] : [])],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      xq: { dtype: "f32", shape: [m, k] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [],
  };
  if (o.afterLinear) {
    // The adjacent linear → SRQ form (the spelling ADR 0103's fusion matches). The SRQ's input is
    // the linear's output.
    graph.values.pre = { dtype: "f32", shape: [m, k] };
    graph.initializers.wp = {};
    graph.initializers.bp = {};
    graph.values.wp = { dtype: "f32", shape: [k, PRE_K] };
    graph.values.bp = { dtype: "f32", shape: [k] };
    graph.nodes.push({ op: "linear", ins: ["src", "wp", "bp"], outs: ["pre"], attrs: {} });
  }
  graph.nodes.push({
    op: "static_quantize",
    ins: [o.afterLinear ? "pre" : "src"],
    outs: ["xq"],
    attrs: { scale: o.scale ?? Math.fround(0.00071) },
  });
  graph.nodes.push({ op: "linear", ins: ["xq", "w", "b"], outs: ["y"], attrs: {} });
  if (o.extraConsumer) {
    graph.values.copy = { dtype: "f32", shape: [m, k] };
    graph.nodes.push({ op: "neg", ins: ["xq"], outs: ["copy"], attrs: {} });
    graph.outputs.push("copy");
  }
  if (o.unparallelConsumer) {
    // n=260 is a shape not in PARALLEL_SHAPES (= a linear that does not fall back to parallel
    // GEMV).
    graph.initializers.w2 = {};
    graph.initializers.b2 = {};
    graph.values.w2 = { dtype: "f32", shape: [260, k] };
    graph.values.b2 = { dtype: "f32", shape: [260] };
    graph.values.y2 = { dtype: "f32", shape: [m, 260] };
    graph.nodes.push({ op: "linear", ins: ["xq", "w2", "b2"], outs: ["y2"], attrs: {} });
    graph.outputs.push("y2");
  }
  if (o.plainConsumer) {
    // It does fall back to parallel GEMV, but is **a row with the flag false** (same n/k, only the
    // group length differs).
    const plain = PLAIN_SHAPES.find((s) => s.storage === shape.storage && s.k === k);
    if (plain === undefined) throw Error(`no measured shape with the flag false for k=${k}`);
    graph.initializers.w4 = {};
    graph.initializers.b4 = {};
    graph.values.w4 = { dtype: "f32", shape: [plain.n, k] };
    graph.values.b4 = { dtype: "f32", shape: [plain.n] };
    graph.values.y4 = { dtype: "f32", shape: [m, plain.n] };
    graph.nodes.push({ op: "linear", ins: ["xq", "w4", "b4"], outs: ["y4"], attrs: {} });
    graph.outputs.push("y4");
  }
  if (o.weightConsumer) {
    // A linear that takes the same value in a slot other than the activation (the weight). The
    // binding expects f32 words, so packed is impossible.
    graph.initializers.b3 = {};
    graph.values.b3 = { dtype: "f32", shape: [m] };
    graph.values.y3 = { dtype: "f32", shape: [m, m] };
    graph.nodes.push({ op: "linear", ins: ["xq", "xq", "b3"], outs: ["y3"], attrs: {} });
    graph.outputs.push("y3");
  }
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  return graph;
};

/**
 * Storage per weight initializer of the graph (merge input — taken from the same measured table
 * rows as `weightLayouts`).
 */
const storageMap = (o: Options = {}): StorageMap => {
  const shape = o.shape ?? DEFAULT_SHAPE;
  const plain = PLAIN_SHAPES.find((s) => s.storage === shape.storage && s.k === shape.k);
  return {
    w: weightStorage(shape),
    ...(o.afterLinear ? { wp: { codec: "int2-off" } satisfies StorageSpec } : {}),
    ...(o.plainConsumer && plain !== undefined ? { w4: weightStorage(plain) } : {}),
  };
};

const plan = (o: Options = {}, context: Partial<Parameters<typeof planFusions>[1]> = {}) => {
  const ir = mergeGraph(packedGraph(o), storageMap(o));
  const shape = o.shape ?? DEFAULT_SHAPE;
  const layout = (s: Shape): FusionWeightLayout =>
    s.storage === "i4" ? { storage: "i4", groupSize: s.group as number } : { storage: s.storage };
  const plain = PLAIN_SHAPES.find((s) => s.storage === shape.storage && s.k === shape.k);
  return planFusions(planGraph(ir, {}).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    packedStaticQuantize: true,
    linearGemvReduce: "parallel",
    linearCompute: "f32",
    weightLayouts: new Map<string, FusionWeightLayout>([
      ["w", layout(shape)],
      ["wp", { storage: "i2" }],
      ...(plain === undefined
        ? []
        : [["w4", layout(plain)] satisfies [string, FusionWeightLayout]]),
    ]),
    ...context,
  });
};

/** The packed role attached to plain nodes (looked up by op name). */
const roles = (steps: ReturnType<typeof plan>["steps"]): readonly string[] =>
  steps.flatMap((step) =>
    step.kind === "node" && step.packedActivations !== undefined
      ? [`${step.plan.node.op}:${step.packedActivations.role}`]
      : [],
  );

describe("pairing of packed int8 activations (ADR 0105)", () => {
  it("for a measured shape with the flag true, the SRQ and the consumer linear get paired roles and the counter goes up", () => {
    for (const shape of PACKED_SHAPES) {
      for (const m of [1, 4, 8]) {
        const scale = Math.fround(0.00071);
        const label = `${JSON.stringify(shape)} M=${m}`;
        const fused = plan({ shape, m });
        assertEquals(fused.counts.packedStaticQuantize, 1, label);
        assertEquals(roles(fused.steps), ["static_quantize:write", "linear:read"], label);
        for (const step of fused.steps) {
          if (step.kind !== "node" || step.packedActivations === undefined) continue;
          assertEquals(
            step.packedActivations.scale,
            scale,
            "both sides carry the producer SRQ's scale",
          );
        }
      }
    }
  });

  /**
   * Per-shape decisions (ADR 0105 addendum 1). Falling back to parallel GEMV is not a sufficient
   * condition for packed — rows that did not help in measurements are passed as f32.
   */
  it("a consumer that falls back to a measured shape with the flag false is not made packed, even under parallel GEMV", () => {
    for (const shape of PLAIN_SHAPES) {
      const fused = plan({ shape });
      assertEquals(fused.counts.packedStaticQuantize, 0, JSON.stringify(shape));
      assertEquals(roles(fused.steps), [], JSON.stringify(shape));
    }
  });

  it("an SRQ mixed with even one row with the flag false stays f32 (split by group length even for the same n/k)", () => {
    // i4 1536×2048: group 2048 is true / group 32 is false (different rows of the same (storage, n,
    // k)).
    const shape = PACKED_SHAPES[1];
    assertEquals(plan({ shape }).counts.packedStaticQuantize, 1, "packed with true rows only");
    const mixed = plan({ shape, plainConsumer: true });
    assertEquals(mixed.counts.packedStaticQuantize, 0, "f32 once one false row is mixed in");
    assertEquals(roles(mixed.steps), []);
  });

  it("stays f32 unless the option, reduction method and compute method all line up", () => {
    for (const context of [
      { packedStaticQuantize: undefined },
      { packedStaticQuantize: false },
      { linearGemvReduce: "sequential" },
      { linearGemvReduce: "parallel-subgroup32" },
      { linearCompute: "a8" },
      { linearCompute: "f16" },
      { weightLayouts: new Map() },
      { weightLayouts: new Map([["w", { storage: "f16" }]]) },
    ] satisfies Partial<Parameters<typeof planFusions>[1]>[]) {
      const fused = plan({}, context);
      assertEquals(fused.counts.packedStaticQuantize, 0, JSON.stringify(context));
      assertEquals(roles(fused.steps), [], JSON.stringify(context));
    }
  });

  it("stays f32 if even one consumer is not a parallel GEMV activation", () => {
    for (const o of [
      { extraConsumer: true },
      { unparallelConsumer: true },
      { weightConsumer: true },
      { publicQuantized: true },
      { m: 9 },
      { scale: 0 },
    ] satisfies Options[]) {
      const fused = plan(o);
      assertEquals(fused.counts.packedStaticQuantize, 0, JSON.stringify(o));
      assertEquals(roles(fused.steps), [], JSON.stringify(o));
    }
  });

  it("an SRQ swallowed by the linear→SRQ fusion is not a packed target (the output side is out of scope)", () => {
    const fused = plan({ afterLinear: true }, { fuseLinearStaticQuantize: true });
    assertEquals(fused.counts.linearStaticQuantize, 1);
    assertEquals(fused.counts.packedStaticQuantize, 0);
    assertEquals(roles(fused.steps), []);
    // With fusion off it goes back to a plain node, and the same SRQ becomes a packed target.
    const unfused = plan({ afterLinear: true }, { fuseLinearStaticQuantize: false });
    assertEquals(unfused.counts.packedStaticQuantize, 1);
    assertEquals(roles(unfused.steps), ["static_quantize:write", "linear:read"]);
  });

  it("even when a linear taking packed activations heads a linear→SRQ fusion, the fused side falls back to the packed key", () => {
    // The spelling `src → SRQ → linear → SRQ`: the first SRQ produces packed, the second is the
    // output-side epilogue.
    const graph = packedGraph({ shape: DEFAULT_SHAPE });
    graph.values.z = { dtype: "f32", shape: graph.values.y.shape };
    graph.nodes.push({
      op: "static_quantize",
      ins: ["y"],
      outs: ["z"],
      attrs: { scale: Math.fround(0.0013) },
    });
    graph.outputs = ["z"];
    const ir: IrGraph = mergeGraph(graph, storageMap({ shape: DEFAULT_SHAPE }));
    const fused = planFusions(planGraph(ir, {}).nodes, {
      rules: FUSION_RULES,
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      limits: {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      },
      packedStaticQuantize: true,
      fuseLinearStaticQuantize: true,
      linearGemvReduce: "parallel",
      linearCompute: "f32",
      weightLayouts: new Map([["w", { storage: "i2" }]]),
    });
    assertEquals(fused.counts.packedStaticQuantize, 1);
    assertEquals(fused.counts.linearStaticQuantize, 1);
    const step = fused.steps.find((s) => s.kind === "fused");
    if (step === undefined || step.kind !== "fused") throw Error("not fused");
    assertEquals(step.dispatches[0].key, linearGemvStaticQuantizePackedKey("i2", undefined, 32));
    // The output-side SRQ's table is words 4–263, and the activation scale is the last word, 264.
    assertEquals(
      step.dispatches[0].params,
      linearGemvStaticQuantizePackedParams(
        "i2",
        1,
        DEFAULT_SHAPE.n,
        DEFAULT_SHAPE.k,
        Math.fround(0.00071),
        Math.fround(0.0013),
      ),
    );
  });
});

describe("per-shape decisions (ADR 0105 addendum 1)", () => {
  it("the packed predicate is a proper subset of the parallel GEMV predicate (it returns lanes only for rows that helped)", () => {
    for (const shape of PACKED_SHAPES) {
      const label = JSON.stringify(shape);
      const { storage, n, k, group, lanes } = shape;
      assertEquals(linearGemvParallelEligible(storage, 1, n, k, group), lanes, label);
      assertEquals(linearGemvPackedEligible(storage, 1, n, k, group), lanes, label);
    }
    for (const shape of PLAIN_SHAPES) {
      const label = JSON.stringify(shape);
      const { storage, n, k, group, lanes } = shape;
      assertEquals(linearGemvParallelEligible(storage, 1, n, k, group), lanes, label);
      assertEquals(linearGemvPackedEligible(storage, 1, n, k, group), undefined, label);
    }
  });

  it("shapes not in the table and M > 8 are undefined for both predicates", () => {
    assertEquals(linearGemvPackedEligible("i2", 1, 260, 1536), undefined, "n not in the table");
    const { storage, n, k, group } = PACKED_SHAPES[0];
    assertEquals(linearGemvParallelEligible(storage, 9, n, k, group), undefined, "M=9 parallel");
    assertEquals(linearGemvPackedEligible(storage, 9, n, k, group), undefined, "M=9 packed");
  });
});

describe("generated code for packed activations (ADR 0105)", () => {
  it("activation loads per weight word drop from 16→4 for i2 / 8→2 for i4 / 4→1 for i8", () => {
    const loads = (wgsl: string): number => (wgsl.match(/= x\[/g) ?? []).length;
    const cases = [
      ["i2", undefined, 2, 16, 4],
      ["i4", GROUP, 4, 8, 2],
      ["i8", undefined, 16, 4, 1],
    ] as const;
    for (const [storage, group, lanes, plain, packed] of cases) {
      assertEquals(
        loads(linearGemvParallelWgsl(storage, group, lanes)),
        plain,
        `${storage} current`,
      );
      assertEquals(
        loads(linearGemvParallelPackedWgsl(storage, group, lanes)),
        packed,
        `${storage} packed`,
      );
    }
  });

  it("the packed variant differs only in the binding's element type and the end of Dims (the key splits on its last fragment)", () => {
    const packed = linearGemvParallelPackedWgsl("i8", undefined, 16);
    assertEquals(packed.includes("x: array<vec4<u32>>"), true);
    assertEquals(packed.includes("x_scale: f32,"), true);
    assertEquals(linearGemvParallelWgsl("i8", undefined, 16).includes("x: array<vec4<f32>>"), true);
    assertEquals(
      linearGemvParallelPackedKey("i8", undefined, 16),
      "linear_gemv_parallel:wi8:l16:packed-x-i8",
    );
    assertEquals(
      linearGemvStaticQuantizePackedKey("i4", GROUP, 4),
      "linear_gemv_parallel:wi4g512:l4:static-quantize:v2:packed-x-i8",
    );
  });

  it("params put the activation scale at the end of Dims and refuse an unrepresentable scale by failing loudly", () => {
    const scale = Math.fround(0.00071);
    const bits = new Uint32Array(Float32Array.of(scale).buffer)[0];
    const plain = linearGemvParallelPackedParams("i2", 1, 12288, 1536, scale);
    assertEquals(plain.length, 4);
    assertEquals([...plain.subarray(0, 3)], [1, 12288, 1536]);
    assertEquals(plain[3], bits);
    const fused = linearGemvStaticQuantizePackedParams("i2", 1, 12288, 1536, scale, scale);
    assertEquals(fused.length, 268);
    assertEquals(fused[264], bits);
    // The output-side SRQ's table stays at its existing position (from word 4).
    assertEquals([...fused.subarray(4, 264)], [...staticQuantizeParams(12288, scale)]);
    for (const bad of [0, -1, Number.NaN, 0.00071]) {
      assertThrows(() => linearGemvParallelPackedParams("i2", 1, 12288, 1536, bad));
    }
  });

  it("the SRQ params of a packed output refuse the identity scale and element counts that are not a multiple of 4", () => {
    const scale = Math.fround(0.00071);
    assertEquals(
      [...staticQuantizePackedParams(1536, scale)],
      [...staticQuantizeParams(1536, scale)],
    );
    assertThrows(() => staticQuantizePackedParams(1536, 0), Error, "identity");
    assertThrows(() => staticQuantizePackedParams(1534, scale), Error, "multiple of 4");
  });
});

/** Plans the `x → linear → static_quantize` graph with the linear→SRQ fusion (ADR 0103) enabled. */
const fuseLinearStaticQuantize = (
  o: LinearStaticQuantizeOptions = {},
  context: Partial<Parameters<typeof planFusions>[1]> = {},
) => {
  // The merge result (an IrGraph with storage) does not depend on the source — one in-memory
  // container is enough.
  const graph = mergedGraph(linearStaticQuantizeModel(o).graphs[GRAPH_NAME], GRAPH_NAME);
  const weight: FusionWeightLayout =
    o.storage === "i4"
      ? { storage: "i4", groupSize: o.group ?? 512 }
      : { storage: o.storage ?? "i8" };
  return planFusions(planGraph(graph, {}).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    fuseLinearStaticQuantize: true,
    linearGemvReduce: "parallel",
    linearCompute: "f32",
    weightLayouts: new Map([["w", weight]]),
    ...context,
  });
};

describe("optional linear→static_quantize fusion", () => {
  it("accepts only explicit opt-in and measured shapes, and keeps the outputs, external inputs and the scale's borrowing source", () => {
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const m of [1, 4, 8]) {
        const n = storage === "i2" ? 12288 : 256,
          planned = fuseLinearStaticQuantize({ storage, m, n }),
          step = planned.steps[0];
        assertEquals(planned.counts.linearStaticQuantize, 1);
        if (step.kind !== "fused") throw Error("not fused");
        assertEquals(step.ins, ["x", "w", "b"]);
        assertEquals(step.binds, ["x", "w", "b"]);
        assertEquals(step.outputName, "y");
        assertEquals(step.temps, []);
        assertEquals(
          step.dispatches[0].key,
          linearGemvStaticQuantizeKey(
            storage,
            storage === "i4" ? 512 : undefined,
            storage === "i2" ? 2 : 32,
          ),
        );
        assertEquals(step.dispatches[0].operands?.at(-1), { kind: "weightScale", index: 1 });
        assertEquals(step.dispatches[0].params[3], 0);
      }
    }
  });
  it("leaves no opt-in, another compute method, expanded weights and unvalidated shapes on the existing path", () => {
    for (const context of [
      { fuseLinearStaticQuantize: undefined },
      { fuseLinearStaticQuantize: false },
      { linearGemvReduce: "sequential" },
      { linearGemvReduce: "parallel-subgroup32" },
      { linearCompute: "a8" },
      { linearCompute: "f16" },
      { weightLayouts: new Map() },
      { weightLayouts: new Map([["w", { storage: "f16" }]]) },
    ] satisfies Partial<Parameters<typeof planFusions>[1]>[]) {
      assertEquals(fuseLinearStaticQuantize({}, context).counts.linearStaticQuantize, 0);
    }
    for (const o of [
      { m: 9 },
      { n: 260 },
      { k: 1504 },
      { publicLinear: true },
      { sharedLinear: true },
      { interpose: true },
    ])
      assertEquals(fuseLinearStaticQuantize(o).counts.linearStaticQuantize, 0);
  });
  it("scale 0 keeps the existing identity operation and does not silently shrink the tile count limit", () => {
    const step = fuseLinearStaticQuantize({ scale: 0 }).steps[0];
    if (step.kind !== "fused") throw Error("not fused");
    assertEquals(step.dispatches[0].params[262], 1);
    assertThrows(() =>
      fuseLinearStaticQuantize(
        {},
        {
          limits: {
            maxStorageBufferBindingSize: 128 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 2,
          },
        },
      ),
    );
  });
});
