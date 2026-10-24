// adaLN = `layer_norm → (1 + scale) → mul → add` の strict peephole（融合ルール adaln —
// src/runtime/fusion.ts）。エクスポータが出す**非隣接の窓 6 / 7 ノード**だけを 1 dispatch へ
// 畳み、掴めない形は素の列へ落ちる。
//
// この門が見るのは値のビット parity 一本。融合カーネルは中間 3 本（正規化結果 /`1 + scale` /
// 積）の storage 往復を消すので、丸め位置が素の列とずれれば有限値のビット列が動く。
// workgroup u32 staging による丸め障壁が実バックエンドで効いているかは、ここでしか分からない。

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import {
  ADALN_MODULATION_PACK_KEY,
  ADALN_NORM_KEY,
  ADALN_NORM_TOKEN_KEY,
} from "../src/kernels/adaln-norm.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

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

const adalnGraph = (options: AdalnGraphOptions = {}): DeclarationJson => {
  const values: DeclarationJson["values"] = {
    t: { dtype: "f32", shape: [...ROW_SHAPE] },
    shift: { dtype: "f32", shape: [...MOD_SHAPE] },
    scale: { dtype: "f32", shape: [...MOD_SHAPE] },
    s: { dtype: "f32", shape: [...MOD_SHAPE] },
    p: { dtype: "f32", shape: [...ROW_SHAPE] },
    y: { dtype: "f32", shape: [...ROW_SHAPE] },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype: "f32", shape: [...ROW_SHAPE] },
    { name: "ln_weight", dtype: "f32", shape: [DIM] },
    { name: "ln_bias", dtype: "f32", shape: [DIM] },
    { name: "one", dtype: "f32", shape: [1] },
    { name: "shift_src", dtype: "f32", shape: [1, DIM] },
    { name: "scale_src", dtype: "f32", shape: [1, DIM] },
  ];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [DIM], eps: 0.000001 },
    },
  ];
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
    version: 2,
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

test.skipIf(!GPU_AVAILABLE)(
  "adaLN 融合は窓 6 / 7 とも primitive と有限ビット / NaN 分類が一致し、4→1 dispatch へ畳む（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const gate of [true, false] as const) {
        const label = `窓 ${gate ? 7 : 6}`;
        const inputs = testInputs({ gate });
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnGraph({ gate }), []),
          GRAPH_NAME,
        );
        const primitive = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnGraph({ gate, interpose: true }), []),
          GRAPH_NAME,
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
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, label).map((row) => row.key),
            [ADALN_NORM_KEY],
            `${label}: census key`,
          );
        } finally {
          await fused.dispose();
          await primitive.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "adaLN の内部 output / 別 consumer / 別名は strict matcher から fallback する（実 GPU）",
  async () => {
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
      const reference = await createSessionFromContainer(
        gpu,
        await openModelBytes(adalnGraph(), []),
        GRAPH_NAME,
      );
      const expected = words((await reference.run(inputs)).y);
      try {
        for (const testCase of cases) {
          const session = await createSessionFromContainer(
            gpu,
            await openModelBytes(adalnGraph(testCase.options), []),
            GRAPH_NAME,
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
);

// ------------------------------------------------------------ per-token form (Wan2.2's DiT)

/**
 * Wan2.2's per-token adaLN: `layer_norm, where(mask, scale_c, scale_g), add(+one), mul,
 * where(mask, shift_c, shift_g), add` — 6 contiguous nodes.
 *
 * `interpose=true` puts a zero-dispatch reshape on the normalised value (the same trick as the
 * broadcast form): the values and the physical dispatches stay the same and only the exact matcher
 * stops matching, so it is the unfused reference on the same backend.
 */
const adalnTokenGraph = (rows: number, dim: number, interpose: boolean): DeclarationJson => {
  const row = [1, rows, dim];
  const vector = [1, 1, dim];
  const values: DeclarationJson["values"] = {
    t: { dtype: "f32", shape: row },
    sel_scale: { dtype: "f32", shape: row },
    s: { dtype: "f32", shape: row },
    p: { dtype: "f32", shape: row },
    sel_shift: { dtype: "f32", shape: row },
    y: { dtype: "f32", shape: row },
  };
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [dim], eps: 0.000001 },
    },
  ];
  if (interpose) {
    values.t_alias = { dtype: "f32", shape: row };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["t_alias"], attrs: {} });
  }
  nodes.push(
    { op: "where", ins: ["mask", "scale_c", "scale_g"], outs: ["sel_scale"], attrs: {} },
    { op: "add", ins: ["sel_scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: [interpose ? "t_alias" : "t", "s"], outs: ["p"], attrs: {} },
    { op: "where", ins: ["mask", "shift_c", "shift_g"], outs: ["sel_shift"], attrs: {} },
    { op: "add", ins: ["p", "sel_shift"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: row },
      { name: "ln_weight", dtype: "f32", shape: [dim] },
      { name: "ln_bias", dtype: "f32", shape: [dim] },
      { name: "mask", dtype: "bool", shape: [1, rows, 1] },
      { name: "one", dtype: "f32", shape: [1] },
      { name: "scale_c", dtype: "f32", shape: vector },
      { name: "scale_g", dtype: "f32", shape: vector },
      { name: "shift_c", dtype: "f32", shape: vector },
      { name: "shift_g", dtype: "f32", shape: vector },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * Rows of different character (as {@link testInput}): wide / tiny variance, signed alternation,
 * a large offset that cancels in the mean, a row with ±Inf / NaN, and ±0 / subnormals at the head
 * of row 2.
 */
const tokenX = (rows: number, dim: number): Tensor => {
  const x = fill([1, rows, dim], (i) => {
    const row = Math.floor(i / dim);
    const at = i % dim;
    switch (row % 5) {
      case 0:
        return Math.sin(at * 0.37 + row) * 3.1 + at / 997;
      case 1:
        return Math.cos(at * 0.11) * 1e-4;
      case 2:
        return (at % 2 === 0 ? 1 : -1) * (at / 3);
      case 3:
        return Math.sin(at * 0.017) * 1e5 + 1e6;
      default:
        return Math.tan(at * 0.05);
    }
  });
  if (rows > 4) words(x).set([0x7f800000, 0xff800000, 0x7fc01234], 4 * dim);
  if (rows > 2) words(x).set([0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000], 2 * dim);
  return x;
};

/**
 * The four vectors have different values and signs (so picking the wrong side changes the bits),
 * with -0, a subnormal and a NaN payload in fixed slots.
 */
const tokenInputs = (
  rows: number,
  dim: number,
  mask: (row: number) => number,
): Readonly<Record<string, Tensor>> => {
  const scaleCondition = fill([1, 1, dim], (i) => Math.cos(i * 0.29) * 3 - 1);
  const scaleGeneration = fill([1, 1, dim], (i) => Math.sin(i * 0.31) * 2.5 + 0.5);
  const shiftCondition = fill([1, 1, dim], (i) => Math.sin(i * 0.07) * 2.5);
  const shiftGeneration = fill([1, 1, dim], (i) => -Math.cos(i * 0.05) * 1.5);
  words(scaleCondition).set([0x80000000, 0x00000001], 0);
  words(scaleGeneration).set([0x807fffff], 1);
  words(shiftCondition).set([0x7fc01234], 2);
  words(shiftGeneration).set([0x80000000, 0x00000003], 3);
  return {
    x: tokenX(rows, dim),
    ln_weight: fill([dim], (i) => 1 + Math.sin(i * 0.21) * 0.4),
    ln_bias: fill([dim], (i) => Math.cos(i * 0.13) * 0.2),
    mask: fill([1, rows, 1], mask, "bool"),
    one: fill([1], () => 1),
    scale_c: scaleCondition,
    scale_g: scaleGeneration,
    shift_c: shiftCondition,
    shift_g: shiftGeneration,
  };
};

/** Mixed (the I2V shape: the first latent frame is the condition side), all condition, all generation. */
const MASKS: readonly (readonly [string, (row: number) => number])[] = [
  ["mixed mask", (row) => (row % 3 === 1 ? 0 : 1)],
  ["all condition", () => 1],
  ["all generation", () => 0],
];

test.skipIf(!GPU_AVAILABLE)(
  "per-token adaLN fusion matches the unfused sequence bit for bit (finite) / by NaN class for mixed and uniform masks, and folds 6 dispatches into 2 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // 257 = a partial last block; 3072 = Wan2.2's width; 8 = narrower than one block.
      for (const [rows, dim] of [
        [5, 257],
        [7, 3072],
        [4, 8],
      ] as const) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnTokenGraph(rows, dim, false), []),
          GRAPH_NAME,
        );
        const primitive = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnTokenGraph(rows, dim, true), []),
          GRAPH_NAME,
        );
        try {
          for (const [maskLabel, mask] of MASKS) {
            const label = `rows=${rows} dim=${dim} ${maskLabel}`;
            const inputs = tokenInputs(rows, dim, mask);
            // The submit statistics accumulate over a session's runs, so compare the increments.
            const fusedBefore = fused.diagnostics().submit.dispatchCount;
            const primitiveBefore = primitive.diagnostics().submit.dispatchCount;
            const fusedOut = (await fused.run(inputs)).y;
            const primitiveOut = (await primitive.run(inputs)).y;
            assertFloatParity(words(fusedOut), words(primitiveOut), `${label}: fused vs primitive`);
            assertEquals(
              fused.diagnostics().submit.dispatchCount - fusedBefore,
              2,
              `${label}: pack + body`,
            );
            assertEquals(
              primitive.diagnostics().submit.dispatchCount - primitiveBefore,
              6,
              `${label}: layer_norm + where + add + mul + where + add`,
            );
            assertEquals(fused.diagnostics().lastRunFusions?.adaln, 1, `${label}: fused counter`);
            assertEquals(
              primitive.diagnostics().lastRunFusions?.adaln,
              0,
              `${label}: reference counter`,
            );
            assertEquals(
              requireCensus(fused.diagnostics().lastRunPipelines, label)
                .map((row) => row.key)
                .sort(),
              [ADALN_MODULATION_PACK_KEY, ADALN_NORM_TOKEN_KEY].sort(),
              `${label}: census keys`,
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
);

test.skipIf(!GPU_AVAILABLE)(
  "per-token adaLN matches the unfused sequence when both the pack and the rows run several grid-stride passes (real GPU)",
  async () => {
    // Two workgroups per dispatch: the pack covers 1,100 columns in three passes and the body
    // covers 5 rows in three passes.
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 2 } });
    const rows = 5;
    const dim = 1100;
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupsPerDimension, 2);
      const fused = await createSessionFromContainer(
        gpu,
        await openModelBytes(adalnTokenGraph(rows, dim, false), []),
        GRAPH_NAME,
      );
      const primitive = await createSessionFromContainer(
        gpu,
        await openModelBytes(adalnTokenGraph(rows, dim, true), []),
        GRAPH_NAME,
      );
      try {
        const inputs = tokenInputs(rows, dim, (row) => row % 2);
        assertFloatParity(
          words((await fused.run(inputs)).y),
          words((await primitive.run(inputs)).y),
          "grid-stride",
        );
        assertEquals(fused.diagnostics().lastRunFusions?.adaln, 1);
      } finally {
        await fused.dispose();
        await primitive.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
