// GeGLU の活性側 `gelu_tanh(g) · u` の strict peephole（融合ルール geluTanhMul —
// src/runtime/fusion.ts）。エクスポータが出す隣接 2 ノードだけを 1 dispatch へ畳み、掴めない形は
// 素の列へ落ちる。ここで見るのは**同一バックエンド上での値の一致**（丸め障壁が効いているか）。

import { assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { geluTanhMulKey } from "../src/kernels/gelu-tanh-mul.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

// 256 スレッド workgroup の端数を通す長さ。
const SHAPE = [257] as const;

/** mul の入力順（`gu` = `mul(gelu, u)` / `ug` = その逆）。 */
type MulOrder = "gu" | "ug";

type GeluGraphOptions = {
  readonly interpose?: boolean;
  readonly geluOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly gExtraConsumer?: boolean;
  /** ゲートを長さ 1 の broadcast にする（mul としては合法だが受理集合の外）。 */
  readonly broadcastGate?: boolean;
  /** 活性を erf 型の `gelu`（別 op）にする。 */
  readonly erf?: boolean;
};

/**
 * `gelu_tanh(g) → mul(gelu, u)` の実 export 形（gemma4 decode の per-layer 入力ゲート）。
 * `interpose` は 0 dispatch の別名を 1 本挟むだけで、値も物理 dispatch 数も変えずに
 * strict matcher の隣接条件だけを外す（= 同一バックエンド上の正本を作る）。
 */
const geluGraph = (order: MulOrder, options: GeluGraphOptions = {}): GraphJson => {
  const geluName = "gelu";
  const mulGeluName = options.interpose ? "gelu_alias" : geluName;
  const gateShape = options.broadcastGate ? [1] : [...SHAPE];
  const values: GraphJson["values"] = {
    gelu: { dtype: "f32", shape: [...SHAPE] },
    y: { dtype: "f32", shape: [...SHAPE] },
  };
  const inputs: GraphJson["inputs"] = [
    { name: "g", dtype: "f32", shape: [...SHAPE] },
    { name: "u", dtype: "f32", shape: gateShape },
  ];
  if (options.interpose) values.gelu_alias = { dtype: "f32", shape: [...SHAPE] };
  if (options.extraConsumer) values.gelu_copy = { dtype: "f32", shape: [...SHAPE] };
  if (options.gExtraConsumer) values.g_copy = { dtype: "f32", shape: [...SHAPE] };

  const nodes: GraphJson["nodes"] = [
    { op: options.erf ? "gelu" : "gelu_tanh", ins: ["g"], outs: [geluName], attrs: {} },
  ];
  if (options.interpose) {
    nodes.push({ op: "reshape", ins: [geluName], outs: [mulGeluName], attrs: {} });
  }
  nodes.push({
    op: "mul",
    ins: order === "gu" ? [mulGeluName, "u"] : ["u", mulGeluName],
    outs: ["y"],
    attrs: {},
  });
  if (options.extraConsumer) {
    nodes.push({ op: "neg", ins: [geluName], outs: ["gelu_copy"], attrs: {} });
  }
  if (options.gExtraConsumer) {
    nodes.push({ op: "neg", ins: ["g"], outs: ["g_copy"], attrs: {} });
  }

  const outputs = [
    ...(options.geluOutput ? [geluName] : []),
    "y",
    ...(options.extraConsumer ? ["gelu_copy"] : []),
    ...(options.gExtraConsumer ? ["g_copy"] : []),
  ];
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs,
    initializers: {},
    values,
    nodes,
  };
};

const words = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const isNanWord = (word: number): boolean => (word & 0x7fffffff) > 0x7f800000;

/** 非 NaN はビット一致、NaN はバックエンド差を許して分類だけ一致させる。 */
const assertFloatParity = (actual: Uint32Array, expected: Uint32Array, label: string): void => {
  assertEquals(actual.length, expected.length, label + ": length");
  for (let i = 0; i < actual.length; i++) {
    if (isNanWord(expected[i])) {
      assertEquals(isNanWord(actual[i]), true, `${label}: NaN classification index=${i}`);
    } else {
      assertEquals(actual[i], expected[i], `${label}: word index=${i}`);
    }
  }
};

/**
 * 活性側の入力。特殊ビット列に加えて **tanh の飽和域**（内側引数 √(2/π)·(x + 0.044715x³) が
 * ±44.36 を超える帯 = 前活性 |x| ≳ 10.05）を正負とも通す — 打ち切り `tanh_stable` が primitive と
 * 融合版で同じ帯に効いていなければ、ここだけが割れる。
 */
const geluInput = (): Tensor => {
  const g = fill(SHAPE, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  // 有限値 / 符号付きゼロ / subnormal / 最小正規化数 / overflow 境界 / ±Inf / NaN を同居させる。
  words(g).set([
    0x00000000,
    0x80000000,
    0x00000001,
    0x80000001,
    0x007fffff,
    0x807fffff,
    0x00800000,
    0x80800000,
    0x7f800000,
    0xff800000,
    0x7fc01234,
    0xff801234,
  ]);
  // 打ち切り閾値の直下 / 直上 / 深い飽和域（負側は gelu_tanh が ±0 へ落ちる帯）。
  g.data.set([9.4, -9.4, 10.05, -10.05, 12, -12, 40, -40, 100, -100], 12);
  return g;
};

/** ゲート側の入力（活性と独立な値 — 実測形も別テンソル）。 */
const gateInput = (): Tensor => {
  const u = fill(SHAPE, (i) => Math.cos(i * 0.19) * 2.7 - i / 911);
  words(u).set([0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234], 22);
  return u;
};

Deno.test({
  name:
    "gelu_tanh·mul 融合は両順で primitive と有限ビット / NaN 分類が一致し、2→1 dispatch へ畳む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    const g = geluInput();
    const u = gateInput();
    try {
      for (const order of ["gu", "ug"] as const) {
        const fused = await createSession(gpu, openModel(graphModelBuffer(geluGraph(order))));
        const primitive = await createSession(
          gpu,
          openModel(graphModelBuffer(geluGraph(order, { interpose: true }))),
        );
        try {
          const fusedOut = (await fused.run({ g, u })).y;
          const primitiveOut = (await primitive.run({ g, u })).y;
          assertFloatParity(words(fusedOut), words(primitiveOut), `${order}: fused vs primitive`);
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${order}: fused 1 dispatch`);
          assertEquals(
            primitive.diagnostics().submit.dispatchCount,
            2,
            `${order}: gelu_tanh + mul primitive`,
          );
          // 融合が黙って外れれば値は正しいまま dispatch だけ増える。カウンタで直接押さえる。
          assertEquals(
            fused.diagnostics().lastRunFusions?.geluTanhMul,
            1,
            `${order}: 融合カウンタ`,
          );
          assertEquals(
            primitive.diagnostics().lastRunFusions?.geluTanhMul,
            0,
            `${order}: 反例のカウンタは 0`,
          );
          const timing = fused.diagnostics().lastRunTiming;
          if (timing !== undefined) {
            assertEquals(
              timing.entries.map((entry) => entry.key),
              [geluTanhMulKey(order === "gu" ? "gelu-u" : "u-gelu")],
              `${order}: timing key`,
            );
          }
        } finally {
          await fused.dispose();
          await primitive.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "gelu_tanh·mul で g を消費した後も後続 consumer まで入力バッファを保持する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const g = geluInput();
    const u = gateInput();
    const fused = await createSession(
      gpu,
      openModel(graphModelBuffer(geluGraph("gu", { gExtraConsumer: true }))),
    );
    const primitive = await createSession(
      gpu,
      openModel(graphModelBuffer(geluGraph("gu", { interpose: true, gExtraConsumer: true }))),
    );
    try {
      const fusedOut = await fused.run({ g, u });
      const primitiveOut = await primitive.run({ g, u });
      assertFloatParity(words(fusedOut.y), words(primitiveOut.y), "gelu_tanh·mul output");
      assertFloatParity(words(fusedOut.g_copy), words(primitiveOut.g_copy), "late g consumer");
      assertEquals(fused.diagnostics().submit.dispatchCount, 2, "融合 + late consumer");
      assertEquals(primitive.diagnostics().submit.dispatchCount, 3, "primitive + late consumer");
    } finally {
      await fused.dispose();
      await primitive.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "gelu_tanh·mul の内部 output / 別 consumer / 別名 / broadcast ゲート / erf 型 gelu は strict matcher から fallback する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const g = geluInput();
    const u = gateInput();
    const scalarGate = fill([1], () => 1.75);
    const cases: readonly {
      readonly name: string;
      readonly graph: GraphJson;
      readonly inputs: Readonly<Record<string, Tensor>>;
      readonly dispatches: number;
    }[] = [
      {
        name: "internal output",
        graph: geluGraph("gu", { geluOutput: true }),
        inputs: { g, u },
        dispatches: 2,
      },
      {
        name: "extra consumer",
        graph: geluGraph("ug", { extraConsumer: true }),
        inputs: { g, u },
        dispatches: 3,
      },
      {
        name: "interposed alias",
        graph: geluGraph("gu", { interpose: true }),
        inputs: { g, u },
        dispatches: 2,
      },
      {
        name: "broadcast gate",
        graph: geluGraph("gu", { broadcastGate: true }),
        inputs: { g, u: scalarGate },
        dispatches: 2,
      },
      {
        name: "erf 型 gelu",
        graph: geluGraph("gu", { erf: true }),
        inputs: { g, u },
        dispatches: 2,
      },
    ];
    try {
      for (const testCase of cases) {
        const session = await createSession(gpu, openModel(graphModelBuffer(testCase.graph)));
        try {
          await session.run(testCase.inputs);
          assertEquals(
            session.diagnostics().submit.dispatchCount,
            testCase.dispatches,
            `${testCase.name}: fallback dispatch count`,
          );
          assertEquals(
            session.diagnostics().lastRunFusions?.geluTanhMul,
            0,
            `${testCase.name}: 融合カウンタ 0`,
          );
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
