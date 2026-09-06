// DiT のゲート付き残差 `mul(gate, x) → add(residual, ·)` の strict peephole（融合ルール
// gatedResidual — src/runtime/fusion.ts）。エクスポータが出す隣接 2 ノードだけを 1 dispatch へ
// 畳み、掴めない形は素の列へ落ちる。ここで見るのは**同一バックエンド上での値の一致**
// （積の格納境界を消す fma 縮約が丸め障壁で止まっているか）。

import { assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { gatedResidualKey } from "../src/kernels/gated-residual.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/**
 * 行幅 257 は **256 の倍数でない**（workgroup 256 の端数と、行境界が block 境界に揃わない形を
 * 同時に通す）。rank 3 の `[1,5,257]` は要素数 1285 でこれも端数。
 */
const DIM = 257;
const ROW_SHAPE = [1, 5, DIM] as const;
const ROW_SHAPE_4 = [1, 2, 3, DIM] as const;

/** mul の入力順（`gx` = `mul(gate, x)` = anima / irodori 形 / `xg` = gemma4 の RoPE 末尾形）。 */
type MulOrder = "gx" | "xg";

type GatedGraphOptions = {
  /** rank 4 の行（実測は rank 3 の DiT 形と rank 4 の RoPE 末尾形の両方）。 */
  readonly rank4?: boolean;
  /** 中間 mul の直後に 0 dispatch の別名を 1 本挟む（隣接条件だけを外す）。 */
  readonly interpose?: boolean;
  /** 中間 mul を graph output にする。 */
  readonly productOutput?: boolean;
  /** 中間 mul に別 consumer を足す。 */
  readonly extraConsumer?: boolean;
  /** 残差の後で x を読み直す consumer を足す（bind の寿命が延びる形）。 */
  readonly lateXConsumer?: boolean;
  /** ゲートを x と同 shape にする（broadcast でない素の mul）。 */
  readonly denseGate?: boolean;
  /** 行そのものを `[1,…,1,dim]` にする（mul の両入力が broadcast）。 */
  readonly flatRow?: boolean;
  /** 残差も broadcast にする（rms_norm ベース adaLN の変調形）。 */
  readonly broadcastResidual?: boolean;
  /** add の入力順を入れ替える（観測外の slot 順）。 */
  readonly swappedAdd?: boolean;
};

const rowOf = (options: GatedGraphOptions): readonly number[] => {
  const row = options.rank4 ? [...ROW_SHAPE_4] : [...ROW_SHAPE];
  return options.flatRow ? [...row.slice(0, -1).map(() => 1), DIM] : row;
};

const broadcastOf = (row: readonly number[]): readonly number[] => [
  ...row.slice(0, -1).map(() => 1),
  DIM,
];

/**
 * 実 export 形 `mul(gate[1,…,1,dim], x) → add(residual, ·)`（anima transformer の 84 対 /
 * irodori DiT の 24 対）。`interpose` は 0 dispatch の別名を 1 本挟むだけで、値も物理 dispatch 数も
 * 変えずに strict matcher の隣接条件だけを外す（= 同一バックエンド上の正本を作る）。
 */
const gatedGraph = (order: MulOrder, options: GatedGraphOptions = {}): GraphJson => {
  const dtype = "f32";
  const row = rowOf(options);
  const gate = options.denseGate ? row : broadcastOf(row);
  const residual = options.broadcastResidual ? broadcastOf(row) : row;
  const values: GraphJson["values"] = {
    p: { dtype, shape: [...row] },
    y: { dtype, shape: [...row] },
  };
  const inputs: GraphJson["inputs"] = [
    { name: "gate", dtype, shape: [...gate] },
    { name: "x", dtype, shape: [...row] },
    { name: "residual", dtype, shape: [...residual] },
  ];
  const nodes: GraphJson["nodes"] = [
    { op: "mul", ins: order === "gx" ? ["gate", "x"] : ["x", "gate"], outs: ["p"], attrs: {} },
  ];
  let product = "p";
  if (options.interpose) {
    values.p_alias = { dtype, shape: [...row] };
    nodes.push({ op: "reshape", ins: ["p"], outs: ["p_alias"], attrs: {} });
    product = "p_alias";
  }
  nodes.push({
    op: "add",
    ins: options.swappedAdd ? [product, "residual"] : ["residual", product],
    outs: ["y"],
    attrs: {},
  });
  const extraOutputs: string[] = [];
  if (options.extraConsumer) {
    values.p_copy = { dtype, shape: [...row] };
    nodes.push({ op: "neg", ins: ["p"], outs: ["p_copy"], attrs: {} });
    extraOutputs.push("p_copy");
  }
  if (options.lateXConsumer) {
    values.x_copy = { dtype, shape: [...row] };
    nodes.push({ op: "neg", ins: ["x"], outs: ["x_copy"], attrs: {} });
    extraOutputs.push("x_copy");
  }
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...(options.productOutput ? ["p"] : []), "y", ...extraOutputs],
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

/** 特殊ビット列（±0 / subnormal / 最小正規化数 / ±Inf / NaN）を先頭に敷いた決定的な入力。 */
const specials = (tensor: Tensor, at: number): Tensor => {
  words(tensor).set([
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
  ], at);
  return tensor;
};

/**
 * ゲート。**行内で 1 周期を跨ぐ**ように作る（`i % dim` の写像が素の broadcast mul と同じ
 * ことを、行ごとに違う積で見るため）。大きな値と非正規数を混ぜて、`residual + gate·x` が
 * fma へ縮約されたときに 1 丸め / 2 丸めの差が出る帯を通す。
 */
const gateInput = (shape: readonly number[]): Tensor => {
  const gate = fill(shape, (i) => Math.cos(i * 0.31) * 1.75 + (i % 7) * 1e-3);
  gate.data.set([3.0e38, -3.0e38, 1.0e-40, -1.0e-40, 1 + 2 ** -23, 1 - 2 ** -24], 12);
  return specials(gate, 0);
};

/** 被ゲート側。積が正規化数の端へ寄る帯（大きな値 × 大きな値）も通す。 */
const xInput = (shape: readonly number[]): Tensor => {
  const x = fill(shape, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  x.data.set([2.0e38, 2.0e38, 1.0e-38, -1.0e-38, 2 ** 24, -(2 ** 24)], 12);
  return specials(x, 24);
};

/**
 * 残差。**積と桁が大きく離れた値**を混ぜる — `residual + gate·x` を 1 丸めの fma で計算すると、
 * 積の下位ビットが残差の指数へ生き残ってビット差になる（丸め障壁が効いていなければここが割れる）。
 */
const residualInput = (shape: readonly number[]): Tensor => {
  const residual = fill(shape, (i) => Math.cos(i * 0.11) * 4.25 - i / 613);
  residual.data.set([2 ** 30, -(2 ** 30), 2 ** -30, -(2 ** -30), 1, -1], 12);
  return specials(residual, 36);
};

const inputsFor = (options: GatedGraphOptions = {}): Readonly<Record<string, Tensor>> => {
  const row = rowOf(options);
  return {
    gate: gateInput(options.denseGate ? row : broadcastOf(row)),
    x: xInput(row),
    residual: residualInput(options.broadcastResidual ? broadcastOf(row) : row),
  };
};

Deno.test({
  name:
    "gated residual 融合は両順・rank 3 / 4 で primitive と有限ビット / NaN 分類が一致し、2→1 dispatch へ畳む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const order of ["gx", "xg"] as const) {
        for (const rank4 of [false, true]) {
          const label = `${order} rank${rank4 ? 4 : 3}`;
          const inputs = inputsFor({ rank4 });
          const fused = await createSession(
            gpu,
            openModel(graphModelBuffer(gatedGraph(order, { rank4 }))),
          );
          const primitive = await createSession(
            gpu,
            openModel(graphModelBuffer(gatedGraph(order, { rank4, interpose: true }))),
          );
          try {
            const fusedOut = (await fused.run(inputs)).y;
            const primitiveOut = (await primitive.run(inputs)).y;
            assertFloatParity(words(fusedOut), words(primitiveOut), `${label}: fused vs primitive`);
            assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${label}: fused 1 dispatch`);
            assertEquals(
              primitive.diagnostics().submit.dispatchCount,
              2,
              `${label}: mul + add primitive`,
            );
            // 融合が黙って外れれば値は正しいまま dispatch だけ増える。カウンタで直接押さえる。
            assertEquals(
              fused.diagnostics().lastRunFusions?.gatedResidual,
              1,
              `${label}: 融合カウンタ`,
            );
            assertEquals(
              primitive.diagnostics().lastRunFusions?.gatedResidual,
              0,
              `${label}: 反例のカウンタは 0`,
            );
            const timing = fused.diagnostics().lastRunTiming;
            if (timing !== undefined) {
              assertEquals(
                timing.entries.map((entry) => entry.key),
                [gatedResidualKey(order === "gx" ? "gate-x" : "x-gate")],
                `${label}: timing key`,
              );
            }
          } finally {
            await fused.dispose();
            await primitive.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "gated residual で x を消費した後も後続 consumer まで入力バッファを保持する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const inputs = inputsFor();
    const fused = await createSession(
      gpu,
      openModel(graphModelBuffer(gatedGraph("gx", { lateXConsumer: true }))),
    );
    const primitive = await createSession(
      gpu,
      openModel(graphModelBuffer(gatedGraph("gx", { interpose: true, lateXConsumer: true }))),
    );
    try {
      const fusedOut = await fused.run(inputs);
      const primitiveOut = await primitive.run(inputs);
      assertFloatParity(words(fusedOut.y), words(primitiveOut.y), "gated residual output");
      assertFloatParity(words(fusedOut.x_copy), words(primitiveOut.x_copy), "late x consumer");
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
    "gated residual の内部 output / 別 consumer / 別名 / 同 shape ゲート / 両方 broadcast / 残差 broadcast / add の順は strict matcher から fallback する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cases: readonly {
      readonly name: string;
      readonly order: MulOrder;
      readonly options: GatedGraphOptions;
      readonly dispatches: number;
    }[] = [
      { name: "internal output", order: "gx", options: { productOutput: true }, dispatches: 2 },
      { name: "extra consumer", order: "xg", options: { extraConsumer: true }, dispatches: 3 },
      { name: "interposed alias", order: "gx", options: { interpose: true }, dispatches: 2 },
      { name: "ゲートが x と同 shape", order: "gx", options: { denseGate: true }, dispatches: 2 },
      { name: "両方 broadcast", order: "gx", options: { flatRow: true }, dispatches: 2 },
      {
        name: "残差が broadcast（adaLN 変調形）",
        order: "gx",
        options: { broadcastResidual: true },
        dispatches: 2,
      },
      { name: "add(mul, residual)", order: "gx", options: { swappedAdd: true }, dispatches: 2 },
    ];
    try {
      for (const testCase of cases) {
        const graph = gatedGraph(testCase.order, testCase.options);
        const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
        try {
          await session.run(inputsFor(testCase.options));
          assertEquals(
            session.diagnostics().submit.dispatchCount,
            testCase.dispatches,
            `${testCase.name}: fallback dispatch count`,
          );
          assertEquals(
            session.diagnostics().lastRunFusions?.gatedResidual,
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
