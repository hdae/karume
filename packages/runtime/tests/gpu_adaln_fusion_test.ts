// adaLN = `layer_norm → (1 + scale) → mul → add` の strict peephole（融合ルール adaln —
// src/runtime/fusion.ts）。エクスポータが出す**非隣接の窓 6 / 7 ノード**だけを 1 dispatch へ
// 畳み、掴めない形は素の列へ落ちる。
//
// この門が見るのは値のビット parity 一本。融合カーネルは中間 3 本（正規化結果 /`1 + scale` /
// 積）の storage 往復を消すので、丸め位置が素の列とずれれば有限値のビット列が動く。
// workgroup u32 staging による丸め障壁が実バックエンドで効いているかは、ここでしか分からない。

import { assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { ADALN_NORM_KEY } from "../src/kernels/adaln-norm.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/** 行内は 256 スレッド workgroup の端数を通す長さ、行数は grid-stride を 1 周以上させる。 */
const ROWS = 5;
const DIM = 257;
const ROW_SHAPE = [1, ROWS, DIM] as const;
const MOD_SHAPE = [1, 1, DIM] as const;

type AdalnGraphOptions = {
  /** 実測どおり gate の reshape も置く（窓 7）。false なら窓 6。 */
  readonly gate?: boolean;
  /**
   * 正規化結果に 0 dispatch の別名を 1 本挟む。値も物理 dispatch 数も変えずに strict matcher
   * の結線条件だけを外す（= 同一バックエンド上の正本を作る）。
   */
  readonly interpose?: boolean;
  /** 正規化結果を graph output にする（内部値の private 条件を外す反例）。 */
  readonly normOutput?: boolean;
  /** `1 + scale` に別 consumer を足す反例。 */
  readonly modulationExtraConsumer?: boolean;
};

const adalnGraph = (options: AdalnGraphOptions = {}): GraphJson => {
  const values: GraphJson["values"] = {
    t: { dtype: "f32", shape: [...ROW_SHAPE] },
    shift: { dtype: "f32", shape: [...MOD_SHAPE] },
    scale: { dtype: "f32", shape: [...MOD_SHAPE] },
    s: { dtype: "f32", shape: [...MOD_SHAPE] },
    p: { dtype: "f32", shape: [...ROW_SHAPE] },
    y: { dtype: "f32", shape: [...ROW_SHAPE] },
  };
  const inputs: GraphJson["inputs"] = [
    { name: "x", dtype: "f32", shape: [...ROW_SHAPE] },
    { name: "ln_weight", dtype: "f32", shape: [DIM] },
    { name: "ln_bias", dtype: "f32", shape: [DIM] },
    { name: "one", dtype: "f32", shape: [1] },
    { name: "shift_src", dtype: "f32", shape: [1, DIM] },
    { name: "scale_src", dtype: "f32", shape: [1, DIM] },
  ];
  const nodes: GraphJson["nodes"] = [{
    op: "layer_norm",
    ins: ["x", "ln_weight", "ln_bias"],
    outs: ["t"],
    attrs: { normalized_shape: [DIM], eps: 0.000001 },
  }];
  let normForMul = "t";
  if (options.interpose) {
    values.t_alias = { dtype: "f32", shape: [...ROW_SHAPE] };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["t_alias"], attrs: {} });
    normForMul = "t_alias";
  }
  nodes.push(
    { op: "reshape", ins: ["shift_src"], outs: ["shift"], attrs: {} },
    { op: "reshape", ins: ["scale_src"], outs: ["scale"], attrs: {} },
  );
  if (options.gate !== false) {
    inputs.push({ name: "gate_src", dtype: "f32", shape: [1, DIM] });
    values.gate = { dtype: "f32", shape: [...MOD_SHAPE] };
    nodes.push({ op: "reshape", ins: ["gate_src"], outs: ["gate"], attrs: {} });
  }
  nodes.push(
    { op: "add", ins: ["scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: [normForMul, "s"], outs: ["p"], attrs: {} },
    { op: "add", ins: ["p", "shift"], outs: ["y"], attrs: {} },
  );
  if (options.modulationExtraConsumer) {
    values.s_copy = { dtype: "f32", shape: [...MOD_SHAPE] };
    nodes.push({ op: "neg", ins: ["s"], outs: ["s_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [
      ...(options.normOutput ? ["t"] : []),
      "y",
      // gate は鎖の外で消費される値（実 IR では次の block が食う）。
      ...(options.gate !== false ? ["gate"] : []),
      ...(options.modulationExtraConsumer ? ["s_copy"] : []),
    ],
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
 * 行ごとに性格を変える。0〜2 行目は有限（分散の大小と符号付きゼロ）、3 行目は大きさを
 * 振り切って正規化の桁落ちを踏み、4 行目に ±Inf / NaN を混ぜて行全体を NaN にする。
 */
const testInput = (): Tensor => {
  const x = fill(ROW_SHAPE, (i) => {
    const row = Math.floor(i / DIM);
    const at = i % DIM;
    if (row === 0) return Math.sin(at * 0.37) * 3.1 + at / 997;
    if (row === 1) return Math.cos(at * 0.11) * 1e-4;
    if (row === 2) return (at % 2 === 0 ? 1 : -1) * (at / 3);
    if (row === 3) return Math.sin(at * 0.017) * 1e5 + 1e6;
    return Math.tan(at * 0.05);
  });
  // 4 行目に非有限を差し込む（mean が NaN になり、行全体が NaN 分類へ落ちる）。
  words(x).set([0x7f800000, 0xff800000, 0x7fc01234], 4 * DIM);
  // 2 行目の先頭は ±0 / subnormal / 最小正規化数。
  words(x).set([0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000], 2 * DIM);
  return x;
};

const testInputs = (options: AdalnGraphOptions = {}): Readonly<Record<string, Tensor>> => ({
  x: testInput(),
  ln_weight: fill([DIM], (i) => 1 + Math.sin(i * 0.21) * 0.4),
  ln_bias: fill([DIM], (i) => Math.cos(i * 0.13) * 0.2),
  one: fill([1], () => 1),
  // scale は ±0 近傍から大きめまで振る（`t · (scale + 1)` の分配則が起きれば有限値で割れる）。
  shift_src: fill([1, DIM], (i) => Math.sin(i * 0.07) * 2.5),
  scale_src: fill([1, DIM], (i) => Math.cos(i * 0.29) * 3 - 1),
  ...(options.gate !== false ? { gate_src: fill([1, DIM], (i) => i / 31) } : {}),
});

Deno.test({
  name:
    "adaLN 融合は窓 6 / 7 とも primitive と有限ビット / NaN 分類が一致し、4→1 dispatch へ畳む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const gate of [true, false] as const) {
        const label = `窓 ${gate ? 7 : 6}`;
        const inputs = testInputs({ gate });
        const fused = await createSession(gpu, openModel(graphModelBuffer(adalnGraph({ gate }))));
        const primitive = await createSession(
          gpu,
          openModel(graphModelBuffer(adalnGraph({ gate, interpose: true }))),
        );
        try {
          const fusedOut = (await fused.run(inputs)).y;
          const primitiveOut = (await primitive.run(inputs)).y;
          assertFloatParity(words(fusedOut), words(primitiveOut), `${label}: fused vs primitive`);
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${label}: fused 1 dispatch`);
          assertEquals(
            primitive.diagnostics().submit.dispatchCount,
            4,
            `${label}: layer_norm + add + mul + add`,
          );
          // 融合が黙って外れれば値は正しいまま dispatch だけ増える。カウンタで直接押さえる。
          assertEquals(fused.diagnostics().lastRunFusions?.adaln, 1, `${label}: 融合カウンタ`);
          assertEquals(
            primitive.diagnostics().lastRunFusions?.adaln,
            0,
            `${label}: 反例のカウンタは 0`,
          );
          const timing = fused.diagnostics().lastRunTiming;
          if (timing !== undefined) {
            assertEquals(
              timing.entries.map((entry) => entry.key),
              [ADALN_NORM_KEY],
              `${label}: timing key`,
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
  name: "adaLN の内部 output / 別 consumer / 別名は strict matcher から fallback する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const inputs = testInputs();
    const cases: readonly { readonly name: string; readonly options: AdalnGraphOptions }[] = [
      { name: "internal output", options: { normOutput: true } },
      { name: "extra consumer", options: { modulationExtraConsumer: true } },
      { name: "interposed alias", options: { interpose: true } },
    ];
    try {
      // 反例はどれも「値は正しいまま dispatch が 4 本に戻る」形。正本（融合）の y と
      // ビット一致することまで見る。
      const reference = await createSession(gpu, openModel(graphModelBuffer(adalnGraph())));
      const expected = words((await reference.run(inputs)).y);
      try {
        for (const testCase of cases) {
          const session = await createSession(
            gpu,
            openModel(graphModelBuffer(adalnGraph(testCase.options))),
          );
          try {
            const output = await session.run(inputs);
            assertFloatParity(words(output.y), expected, `${testCase.name}: fallback の値`);
            assertEquals(
              session.diagnostics().submit.dispatchCount,
              testCase.name === "extra consumer" ? 5 : 4,
              `${testCase.name}: fallback dispatch count`,
            );
            assertEquals(
              session.diagnostics().lastRunFusions?.adaln,
              0,
              `${testCase.name}: 融合カウンタ 0`,
            );
          } finally {
            await session.dispose();
          }
        }
      } finally {
        await reference.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});
