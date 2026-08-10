// transient slot の GPU backing（導出済み計画にヒットした run が使う Session 常駐バッファ群）の門。
//
// backed run はアリーナの確保・参照計数を通らず、**前 run の残骸が載ったバッファ**へそのまま
// 書き込む。正しさの根拠は full-write（ADR 0014）だけなので、検出器は「値を変えながら同じ
// bindings で回した出力が、非 backed 実行とビット単位で一致するか」— stale slot が 1 本でも
// 混じれば前 run の値が残り、例外は 1 つも出ない。
//
// 併せて ①単発 run は slot メモリを一切払わない ②signature の切替で作り直される
// ③LRU 追い出しで返る ④slot の総バイト数が現行 run のプール確保と一致する（footprint 不変）
// を固定する。④が崩れると VRAM の前提（常駐化しても新しいピークは生まれない）が崩れる。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSession, type Session, type Tensor } from "../src/runtime/executor.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** y = silu(x·w + b)（x: [T,3] → y: [T,2]）。末尾 2 ノードが silu 融合に掴まれる。 */
const GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add", "sigmoid", "mul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "proj.weight", storage: { dtype: "f32" } },
    b: { tensor: "proj.bias", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [3, 2] },
    b: { dtype: "f32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 2] },
    g: { dtype: "f32", shape: ["T", 2] },
    s: { dtype: "f32", shape: ["T", 2] },
    y: { dtype: "f32", shape: ["T", 2] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "sigmoid", ins: ["g"], outs: ["s"], attrs: {} },
    { op: "mul", ins: ["g", "s"], outs: ["y"], attrs: {} },
  ],
};

const modelBytes = (): ArrayBuffer =>
  graphModelBuffer(GRAPH, [
    {
      name: "proj.weight",
      dtype: "F32",
      shape: [3, 2],
      data: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1]),
    },
    { name: "proj.bias", dtype: "F32", shape: [2], data: f32Bytes([0.125, -0.5]) },
  ]);

/** `phase` ごとに値が変わる入力（同じ値を配ると stale slot が検出できない）。 */
const input = (rows: number, phase = 0): Tensor => ({
  dtype: "f32",
  shape: [rows, 3],
  data: Float32Array.from({ length: rows * 3 }, (_, i) => ((i + phase * 3) % 9 - 4) * 0.5),
});

/** 出力のビット列。値の一致は「バイト同値」で見る（丸めの取り違えを許容しない）。 */
const bits = (tensor: Tensor): readonly number[] =>
  Array.from(new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length));

/** Session を作り直して 1 run だけ回す = 必ず非 backed（backing はヒット run でしか作らない）。 */
const runFresh = async (
  gpu: Awaited<ReturnType<typeof acquireGpu>>,
  inputs: Parameters<Session["run"]>[0],
  model: ArrayBuffer = modelBytes(),
): Promise<Tensor> => {
  const session = await createSession(gpu, openModel(model));
  try {
    const outputs = await session.run(inputs);
    assertEquals(
      session.diagnostics().planBacking,
      { residentBytes: 0, buildCount: 0 },
      "単発 run は slot メモリを一切払わない",
    );
    return outputs["y"];
  } finally {
    await session.dispose();
  }
};

Deno.test({
  name: "backed run の出力は非 backed 実行とビット単位で一致する（stale slot の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      // 1 run 目 = ミス（アリーナ経路）/ 2 run 目 = backing 構築 / 3 run 目 = backed 高速路。
      const phases = [0, 1, 2];
      const actual: Tensor[] = [];
      for (const phase of phases) actual.push((await session.run({ x: input(4, phase) }))["y"]);
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 1 });
      assertEquals(session.diagnostics().planBacking.buildCount, 1, "構築は 1 度だけ");

      // 恒真化の門: 入力を変えても出力が同じなら、stale slot が残っていても緑になる。
      assert(
        JSON.stringify(bits(actual[0])) !== JSON.stringify(bits(actual[1])),
        "phase ごとに出力が変わっていない（検出器として空振る）",
      );

      for (const phase of phases) {
        const expected = await runFresh(gpu, { x: input(4, phase) });
        assertEquals(actual[phase].shape, expected.shape);
        assertEquals(bits(actual[phase]), bits(expected), `phase ${phase} のビット一致`);
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "backing は活性 signature のぶんだけ常駐し、切替で作り直される（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().planBacking,
        { residentBytes: 0, buildCount: 0 },
        "ミス run は backing を作らない",
      );

      await session.run({ x: input(4) });
      const narrow = session.diagnostics().planBacking;
      assert(narrow.residentBytes > 0, "初ヒットで slot が常駐する");
      assertEquals(narrow.buildCount, 1);

      await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().planBacking,
        narrow,
        "同じ signature では作り直さない（毎 run 再構築の沈黙劣化の観測点）",
      );

      // 別 signature はミス run では作らず、ヒットして初めて置き換わる（容量 1）。
      await session.run({ x: input(9) });
      assertEquals(session.diagnostics().planBacking, narrow, "別 signature のミス run では不変");
      await session.run({ x: input(9) });
      const wide = session.diagnostics().planBacking;
      assertEquals(wide.buildCount, 2, "切替で作り直す");
      assert(
        wide.residentBytes > narrow.residentBytes,
        `旧 backing の常駐分が置き換わる（${narrow.residentBytes} → ${wide.residentBytes}）`,
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "backed 計画が LRU から追い出されると backing も返り、後続 run は完走する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      await session.run({ x: input(4) });
      await session.run({ x: input(4) });
      assert(session.diagnostics().planBacking.residentBytes > 0);

      // 上限 4 本。T=4 は最古なので、5 種類目（T=5）の登録で追い出される。
      for (const rows of [1, 2, 3, 5]) await session.run({ x: input(rows) });
      assertEquals(
        session.diagnostics().planBacking,
        { residentBytes: 0, buildCount: 1 },
        "追い出された計画の backing は返る（作り直しは起きていない）",
      );

      // 追い出し + 破棄の後も、残った計画のヒット run は正しく完走する。
      const hit = (await session.run({ x: input(5) }))["y"];
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().planBacking.buildCount, 2);
      assertEquals(bits(hit), bits(await runFresh(gpu, { x: input(5) })));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * attention（ノード内一時 2 本）+ reshape（別名・グラフ出力）。slot 導出の 3 経路
 * （出力確保 / 一時の入れ子寿命 / 別名越しのピン留め）を 1 本のグラフで踏む。
 */
const HALF_SCALE = Math.fround(Math.sqrt(1 / Math.sqrt(4)));

const TEMP_GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["attention", "reshape"] },
  symbols: ["M"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 1, "M", 4] },
    { name: "k", dtype: "f32", shape: [1, 1, 4, 4] },
    { name: "v", dtype: "f32", shape: [1, 1, 4, 4] },
  ],
  outputs: ["y"],
  initializers: {},
  values: {
    a: { dtype: "f32", shape: [1, 1, "M", 4] },
    y: { dtype: "f32", shape: ["M", 4] },
  },
  nodes: [
    { op: "attention", ins: ["q", "k", "v"], outs: ["a"], attrs: { scale: HALF_SCALE } },
    { op: "reshape", ins: ["a"], outs: ["y"], attrs: {} },
  ],
};

const ROWS = 6;
const tempInputs = (phase: number) => ({
  q: fill([1, 1, ROWS, 4], (i) => ((i + phase) % 7 - 3) * 0.25),
  k: fill([1, 1, 4, 4], (i) => ((i % 5) - 2) * 0.5),
  v: fill([1, 1, 4, 4], (i) => ((i % 3) - 1) * 0.75),
});

/** アリーナが 1 本のバッファに配る大きさ（最小 4 バイト + 4 バイト整列）。 */
const bufferBytes = (count: number): number => Math.max(4, count * 4);

Deno.test({
  name: "slot の総バイト数は非 backed run のプール確保と一致する（footprint 不変の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // プール外（入力アップロードと readback staging）のぶん。アリーナの allocatedBytes から
    // これを引いた残りが「dispatch が書く出力ストレージの実確保」= slot 表の総バイト数。
    const hostBytes = bufferBytes(ROWS * 4) + bufferBytes(16) * 2 + bufferBytes(ROWS * 4);
    const reference = await createSession(gpu, openModel(graphModelBuffer(TEMP_GRAPH)));
    let pooledBytes = 0;
    try {
      await reference.run(tempInputs(0));
      const stats = reference.diagnostics().lastRun;
      assert(stats !== undefined, "1 run 目のアリーナ実績が無い");
      pooledBytes = stats.allocatedBytes - hostBytes;
      assert(pooledBytes > 0, `プール確保が 0（門が空振る）: ${stats.allocatedBytes}`);
    } finally {
      await reference.dispose();
    }

    const session = await createSession(gpu, openModel(graphModelBuffer(TEMP_GRAPH)));
    try {
      const first = (await session.run(tempInputs(1)))["y"];
      const second = (await session.run(tempInputs(1)))["y"];
      assertEquals(
        session.diagnostics().planBacking.residentBytes,
        pooledBytes,
        "slot 表が現行のプール確保と食い違う（VRAM の前提が崩れる）",
      );
      // 一時と別名を踏む形でも backed 出力は非 backed と一致する（別名越しのピン留めも門）。
      assertEquals(bits(second), bits(first));
      assertEquals(second.shape, [ROWS, 4]);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
