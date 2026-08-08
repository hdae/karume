// SiLU = x * sigmoid(x) の strict peephole（融合ルール silu — src/runtime/fusion.ts）。
// エクスポータが出す隣接 2 ノードだけを 1 dispatch へ畳み、掴めない形は素の列へ落ちる。

import { assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { siluKey } from "../src/kernels/silu.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

// 256 スレッド workgroup の端数を通す長さ。
const SHAPE = [257] as const;

type MulOrder = "xs" | "sx";

type SiluGraphOptions = {
  readonly interpose?: boolean;
  readonly sigmoidOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly xExtraConsumer?: boolean;
  readonly otherMul?: boolean;
};

/**
 * `sigmoid(x) → mul(x, sigmoid(x))` の実 export 形。
 * `interpose` は 0 dispatch の別名を 1 本挟むだけで、値も物理 dispatch 数も変えずに
 * strict matcher の隣接条件だけを外す（= 同一バックエンド上の正本を作る）。
 */
const siluGraph = (order: MulOrder, options: SiluGraphOptions = {}): GraphJson => {
  const sigmoidName = "sigmoid";
  const mulSigmoidName = options.interpose ? "sigmoid_alias" : sigmoidName;
  const values: GraphJson["values"] = {
    sigmoid: { dtype: "f32", shape: [...SHAPE] },
    y: { dtype: "f32", shape: [...SHAPE] },
  };
  const inputs: GraphJson["inputs"] = [{ name: "x", dtype: "f32", shape: [...SHAPE] }];
  if (options.interpose) values.sigmoid_alias = { dtype: "f32", shape: [...SHAPE] };
  if (options.extraConsumer) values.sigmoid_copy = { dtype: "f32", shape: [...SHAPE] };
  if (options.xExtraConsumer) values.x_copy = { dtype: "f32", shape: [...SHAPE] };
  if (options.otherMul) {
    inputs.push({ name: "z", dtype: "f32", shape: [...SHAPE] });
  }

  const nodes: GraphJson["nodes"] = [{ op: "sigmoid", ins: ["x"], outs: [sigmoidName], attrs: {} }];
  if (options.interpose) {
    nodes.push({ op: "reshape", ins: [sigmoidName], outs: [mulSigmoidName], attrs: {} });
  }
  const mulInputs = options.otherMul
    ? [mulSigmoidName, "z"]
    : order === "xs"
    ? ["x", mulSigmoidName]
    : [mulSigmoidName, "x"];
  nodes.push({ op: "mul", ins: mulInputs, outs: ["y"], attrs: {} });
  if (options.extraConsumer) {
    nodes.push({ op: "neg", ins: [sigmoidName], outs: ["sigmoid_copy"], attrs: {} });
  }
  if (options.xExtraConsumer) {
    nodes.push({ op: "neg", ins: ["x"], outs: ["x_copy"], attrs: {} });
  }

  const outputs = [
    ...(options.sigmoidOutput ? [sigmoidName] : []),
    "y",
    ...(options.extraConsumer ? ["sigmoid_copy"] : []),
    ...(options.xExtraConsumer ? ["x_copy"] : []),
  ];
  return {
    format: "karume-ir",
    version: 1,
    requires: {
      ops: [
        "sigmoid",
        "mul",
        ...(options.interpose ? ["reshape"] : []),
        ...(options.extraConsumer || options.xExtraConsumer ? ["neg"] : []),
      ],
    },
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

const testInput = (): Tensor => {
  const x = fill(SHAPE, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  // 有限値 / 符号付きゼロ / subnormal / 最小正規化数 / overflow 境界 / ±Inf / NaN を同居させる。
  words(x).set([
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
  x.data.set([80, -80, 88, -88, 100, -100], 12);
  return x;
};

Deno.test({
  name:
    "SiLU 融合は xs / sx 両順で primitive と有限ビット / NaN 分類が一致し、2→1 dispatch へ畳む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    const x = testInput();
    try {
      for (const order of ["xs", "sx"] as const) {
        const fused = await createSession(gpu, openModel(graphModelBuffer(siluGraph(order))));
        const primitive = await createSession(
          gpu,
          openModel(graphModelBuffer(siluGraph(order, { interpose: true }))),
        );
        try {
          const fusedOut = (await fused.run({ x })).y;
          const primitiveOut = (await primitive.run({ x })).y;
          assertFloatParity(words(fusedOut), words(primitiveOut), `${order}: fused vs primitive`);
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${order}: fused 1 dispatch`);
          assertEquals(
            primitive.diagnostics().submit.dispatchCount,
            2,
            `${order}: sigmoid + mul primitive`,
          );
          // 融合が黙って外れれば値は正しいまま dispatch だけ増える。カウンタで直接押さえる。
          assertEquals(fused.diagnostics().lastRunFusions?.silu, 1, `${order}: 融合カウンタ`);
          assertEquals(
            primitive.diagnostics().lastRunFusions?.silu,
            0,
            `${order}: 反例のカウンタは 0`,
          );
          const timing = fused.diagnostics().lastRunTiming;
          if (timing !== undefined) {
            assertEquals(
              timing.entries.map((entry) => entry.key),
              [siluKey(order === "xs" ? "x-sigmoid" : "sigmoid-x")],
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
  name: "SiLU で x を 2 回 release しても後続 consumer まで入力バッファを保持する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const x = testInput();
    const fused = await createSession(
      gpu,
      openModel(graphModelBuffer(siluGraph("xs", { xExtraConsumer: true }))),
    );
    const primitive = await createSession(
      gpu,
      openModel(graphModelBuffer(siluGraph("xs", { interpose: true, xExtraConsumer: true }))),
    );
    try {
      const fusedOut = await fused.run({ x });
      const primitiveOut = await primitive.run({ x });
      assertFloatParity(words(fusedOut.y), words(primitiveOut.y), "SiLU output");
      assertFloatParity(words(fusedOut.x_copy), words(primitiveOut.x_copy), "late x consumer");
      assertEquals(fused.diagnostics().submit.dispatchCount, 2, "SiLU + late consumer");
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
    "SiLU の内部 sigmoid output / 別 consumer / 別名 / 別 mul 入力は strict matcher から fallback する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const x = testInput();
    const z = fill(SHAPE, (i) => Math.cos(i * 0.19) * 2.7 - i / 911);
    const cases: readonly {
      readonly name: string;
      readonly graph: GraphJson;
      readonly inputs: Readonly<Record<string, Tensor>>;
      readonly dispatches: number;
    }[] = [
      {
        name: "internal output",
        graph: siluGraph("xs", { sigmoidOutput: true }),
        inputs: { x },
        dispatches: 2,
      },
      {
        name: "extra consumer",
        graph: siluGraph("sx", { extraConsumer: true }),
        inputs: { x },
        dispatches: 3,
      },
      {
        name: "interposed alias",
        graph: siluGraph("xs", { interpose: true }),
        inputs: { x },
        dispatches: 2,
      },
      {
        name: "mul(s,z)",
        graph: siluGraph("xs", { otherMul: true }),
        inputs: { x, z },
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
            session.diagnostics().lastRunFusions?.silu,
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
