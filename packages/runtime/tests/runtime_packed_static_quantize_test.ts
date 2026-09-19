/**
 * packed int8 活性（ADR 0105）の**対付けと生成物**の門（GPU 不要）。
 *
 * 対付けは「生産側 SRQ を packed で書く」と「消費側 linear を packed で読む」の対でしか
 * 成立しない。片側だけ立つと `vec4<u32>` 束縛に f32 の語が流れる = 例外なしの沈黙誤値に
 * なるので、受理集合（席・消費先・格納・形）と、生成 WGSL の活性ロード本数の両方を固定する。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseIrGraph } from "../src/format/ir.ts";
import { type FusionWeightLayout, planFusions } from "../src/runtime/fusion.ts";
import { countUses, planGraph } from "../src/runtime/plan.ts";
import {
  linearGemvParallelPackedKey,
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvParallelWgsl,
  linearGemvStaticQuantizePackedKey,
  linearGemvStaticQuantizePackedParams,
} from "../src/kernels/linear-gemv.ts";
import {
  staticQuantizePackedParams,
  staticQuantizeParams,
} from "../src/kernels/static-quantize.ts";
import type { GraphJson } from "./helpers/format.ts";

type Storage = "i2" | "i4" | "i8";

type Options = {
  /** 重みの格納形（並列 GEMV の実測形だけが packed の対象になる）。 */
  storage?: Storage;
  m?: number;
  k?: number;
  scale?: number;
  /** SRQ 出力に linear 以外の消費先（neg）を足す。 */
  extraConsumer?: boolean;
  /** SRQ 出力を**並列 GEMV に落ちない形**の linear にも食わせる。 */
  unparallelConsumer?: boolean;
  /** SRQ 出力を linear の**重み**スロットで消費する（活性スロット以外）。 */
  weightConsumer?: boolean;
  /** SRQ 出力を graph output にする。 */
  publicQuantized?: boolean;
  /** SRQ の直前に linear を置き、linear→SRQ 融合（ADR 0103）に飲ませる。 */
  afterLinear?: boolean;
};

const N_BY_STORAGE: Readonly<Record<Storage, number>> = { i2: 12288, i4: 6144, i8: 256 };
const GROUP = 512;
/** 前段 linear（`afterLinear`）の縮約長 — `i8 1536×256` は PARALLEL_SHAPES の実測形。 */
const PRE_K = 256;

/** `x → static_quantize → linear` の最小形（消費先を options で崩せる）。 */
const packedGraph = (o: Options = {}): GraphJson => {
  const storage = o.storage ?? "i2",
    m = o.m ?? 1,
    k = o.k ?? 1536,
    n = N_BY_STORAGE[storage];
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear", "static_quantize", "neg"] },
    symbols: [],
    inputs: [{ name: "src", dtype: "f32", shape: [m, o.afterLinear ? PRE_K : k] }],
    outputs: ["y", ...(o.publicQuantized ? ["xq"] : [])],
    initializers: {
      w: {
        tensor: "w",
        storage: {
          dtype: storage,
          scale: "s",
          ...(storage === "i4" ? { group_size: GROUP } : {}),
        },
      },
      b: { tensor: "b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      xq: { dtype: "f32", shape: [m, k] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [],
  };
  if (o.afterLinear) {
    // linear → SRQ の隣接形（ADR 0103 の融合が掴む綴り）。SRQ の入力を linear 出力にする。
    graph.values.pre = { dtype: "f32", shape: [m, k] };
    graph.initializers.wp = { tensor: "wp", storage: { dtype: "i8", scale: "sp" } };
    graph.initializers.bp = { tensor: "bp", storage: { dtype: "f32" } };
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
    // n=260 は PARALLEL_SHAPES に無い形（= 並列 GEMV へ落ちない linear）。
    graph.initializers.w2 = { tensor: "w2", storage: { dtype: "f32" } };
    graph.initializers.b2 = { tensor: "b2", storage: { dtype: "f32" } };
    graph.values.w2 = { dtype: "f32", shape: [260, k] };
    graph.values.b2 = { dtype: "f32", shape: [260] };
    graph.values.y2 = { dtype: "f32", shape: [m, 260] };
    graph.nodes.push({ op: "linear", ins: ["xq", "w2", "b2"], outs: ["y2"], attrs: {} });
    graph.outputs.push("y2");
  }
  if (o.weightConsumer) {
    // 活性スロット以外（重み）で同じ値を取る linear。f32 の語を期待する束縛なので packed 不可。
    graph.initializers.b3 = { tensor: "b3", storage: { dtype: "f32" } };
    graph.values.b3 = { dtype: "f32", shape: [m] };
    graph.values.y3 = { dtype: "f32", shape: [m, m] };
    graph.nodes.push({ op: "linear", ins: ["xq", "xq", "b3"], outs: ["y3"], attrs: {} });
    graph.outputs.push("y3");
  }
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  return graph;
};

const plan = (
  o: Options = {},
  context: Partial<Parameters<typeof planFusions>[1]> = {},
) => {
  const ir = parseIrGraph(JSON.stringify(packedGraph(o)));
  const storage = o.storage ?? "i2";
  const weight: FusionWeightLayout = storage === "i4"
    ? { storage: "i4", groupSize: GROUP }
    : { storage };
  return planFusions(planGraph(ir, {}).nodes, {
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
      ["w", weight],
      ["wp", { storage: "i8" }],
    ]),
    ...context,
  });
};

/** 素のノードに付いた packed の役割（op 名で引く）。 */
const roles = (steps: ReturnType<typeof plan>["steps"]): readonly string[] =>
  steps.flatMap((step) =>
    step.kind === "node" && step.packedActivations !== undefined
      ? [`${step.plan.node.op}:${step.packedActivations.role}`]
      : []
  );

describe("packed int8 活性の対付け（ADR 0105）", () => {
  it("SRQ とその消費先 linear に対で役割が付き、カウンタが立つ", () => {
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const m of [1, 4, 8]) {
        const scale = Math.fround(0.00071);
        const fused = plan({ storage, m });
        assertEquals(fused.counts.packedStaticQuantize, 1, `${storage} M=${m}`);
        assertEquals(roles(fused.steps), [
          "static_quantize:write",
          "linear:read",
        ], `${storage} M=${m}`);
        for (const step of fused.steps) {
          if (step.kind !== "node" || step.packedActivations === undefined) continue;
          assertEquals(step.packedActivations.scale, scale, "生産側 SRQ の scale を両側が持つ");
        }
      }
    }
  });

  it("席・縮約方式・計算方式が揃わなければ f32 のまま", () => {
    for (
      const context of [
        { packedStaticQuantize: undefined },
        { packedStaticQuantize: false },
        { linearGemvReduce: "sequential" },
        { linearGemvReduce: "parallel-subgroup32" },
        { linearCompute: "a8" },
        { linearCompute: "f16" },
        { weightLayouts: new Map() },
        { weightLayouts: new Map([["w", { storage: "f16" }]]) },
      ] satisfies Partial<Parameters<typeof planFusions>[1]>[]
    ) {
      const fused = plan({}, context);
      assertEquals(fused.counts.packedStaticQuantize, 0, JSON.stringify(context));
      assertEquals(roles(fused.steps), [], JSON.stringify(context));
    }
  });

  it("消費先が 1 本でも並列 GEMV の活性でなければ f32 のまま", () => {
    for (
      const o of [
        { extraConsumer: true },
        { unparallelConsumer: true },
        { weightConsumer: true },
        { publicQuantized: true },
        { m: 9 },
        { scale: 0 },
      ] satisfies Options[]
    ) {
      const fused = plan(o);
      assertEquals(fused.counts.packedStaticQuantize, 0, JSON.stringify(o));
      assertEquals(roles(fused.steps), [], JSON.stringify(o));
    }
  });

  it("linear→SRQ 融合に飲まれた SRQ は packed の対象にしない（出力側は範囲外）", () => {
    const fused = plan({ afterLinear: true }, { fuseLinearStaticQuantize: true });
    assertEquals(fused.counts.linearStaticQuantize, 1);
    assertEquals(fused.counts.packedStaticQuantize, 0);
    assertEquals(roles(fused.steps), []);
    // 融合を切れば素のノードに戻り、同じ SRQ が packed の対象になる。
    const unfused = plan({ afterLinear: true }, { fuseLinearStaticQuantize: false });
    assertEquals(unfused.counts.packedStaticQuantize, 1);
    assertEquals(roles(unfused.steps), ["static_quantize:write", "linear:read"]);
  });

  it("packed 活性を取る linear が linear→SRQ 融合の頭でも、融合側が packed のキーへ落ちる", () => {
    // `src → SRQ → linear → SRQ` の綴り: 前段 SRQ が packed 生産、後段は出力側エピローグ。
    const graph = packedGraph({ storage: "i2" });
    graph.values.z = { dtype: "f32", shape: graph.values.y.shape };
    graph.nodes.push({
      op: "static_quantize",
      ins: ["y"],
      outs: ["z"],
      attrs: { scale: Math.fround(0.0013) },
    });
    graph.outputs = ["z"];
    const ir = parseIrGraph(JSON.stringify(graph));
    const fused = planFusions(planGraph(ir, {}).nodes, {
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
    if (step === undefined || step.kind !== "fused") throw Error("融合されていない");
    assertEquals(
      step.dispatches[0].key,
      linearGemvStaticQuantizePackedKey("i2", undefined, 2),
    );
    // 出力側 SRQ の表は語 4〜263、活性 scale は末尾の語 264。
    assertEquals(
      step.dispatches[0].params,
      linearGemvStaticQuantizePackedParams(
        "i2",
        1,
        12288,
        1536,
        Math.fround(0.00071),
        Math.fround(0.0013),
      ),
    );
  });
});

describe("packed 活性の生成物（ADR 0105）", () => {
  it("重み 1 語あたりの活性ロードが i2 16→4 / i4 8→2 / i8 4→1 本に減る", () => {
    const loads = (wgsl: string): number => (wgsl.match(/= x\[/g) ?? []).length;
    const cases = [
      ["i2", undefined, 2, 16, 4],
      ["i4", GROUP, 4, 8, 2],
      ["i8", undefined, 16, 4, 1],
    ] as const;
    for (const [storage, group, lanes, plain, packed] of cases) {
      assertEquals(loads(linearGemvParallelWgsl(storage, group, lanes)), plain, `${storage} 現行`);
      assertEquals(
        loads(linearGemvParallelPackedWgsl(storage, group, lanes)),
        packed,
        `${storage} packed`,
      );
    }
  });

  it("packed 変種は束縛の要素型と Dims の末尾だけが違う（キーは末尾の断片で分かれる）", () => {
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
      "linear_gemv_parallel:wi4g512:l4:static-quantize:v1:packed-x-i8",
    );
  });

  it("params は活性 scale を Dims の末尾へ置き、表せない scale を fail loudly で断る", () => {
    const scale = Math.fround(0.00071);
    const bits = new Uint32Array(Float32Array.of(scale).buffer)[0];
    const plain = linearGemvParallelPackedParams("i2", 1, 12288, 1536, scale);
    assertEquals(plain.length, 4);
    assertEquals([...plain.subarray(0, 3)], [1, 12288, 1536]);
    assertEquals(plain[3], bits);
    const fused = linearGemvStaticQuantizePackedParams("i2", 1, 12288, 1536, scale, scale);
    assertEquals(fused.length, 268);
    assertEquals(fused[264], bits);
    // 出力側 SRQ の表は既存の位置（語 4〜）のまま。
    assertEquals([...fused.subarray(4, 264)], [...staticQuantizeParams(12288, scale)]);
    for (const bad of [0, -1, Number.NaN, 0.00071]) {
      assertThrows(() => linearGemvParallelPackedParams("i2", 1, 12288, 1536, bad));
    }
  });

  it("packed 出力の SRQ params は恒等 scale と 4 の倍数でない要素数を断る", () => {
    const scale = Math.fround(0.00071);
    assertEquals([...staticQuantizePackedParams(1536, scale)], [
      ...staticQuantizeParams(1536, scale),
    ]);
    assertThrows(() => staticQuantizePackedParams(1536, 0), Error, "恒等");
    assertThrows(() => staticQuantizePackedParams(1534, scale), Error, "4 の倍数");
  });
});
