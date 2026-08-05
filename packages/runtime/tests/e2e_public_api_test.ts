// 公開面（mod.ts）だけで書く E2E。ADR 0008 の「面と実装の乖離を機械検出する」常設テスト。
// MUST: src/ を直接 import しない — 内部直参照で書くと、公開面から落ちた機能を検出できない。

import { assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import {
  acquireGpu,
  capabilities,
  createSession,
  openModel,
  SafetensorsError,
  type Tensor,
} from "../mod.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** y[T] = Σ_c (x[T,3] · w[3,2] + b[2])。記号次元 T を含む 3 ノードのグラフ。 */
const GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add", "sum"] },
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
    y: { dtype: "f32", shape: ["T"] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "sum", ins: ["g"], outs: ["y"], attrs: { dim: 1 } },
  ],
};

const W = [0.5, -1.5, 2, 0.25, -0.75, 1];
const B = [0.125, -0.5];

const modelBytes = (): ArrayBuffer =>
  graphModelBuffer(GRAPH, [
    { name: "proj.weight", dtype: "F32", shape: [3, 2], data: f32Bytes(W) },
    { name: "proj.bias", dtype: "F32", shape: [2], data: f32Bytes(B) },
  ]);

/** ノードごとに f32 へ丸めながら手計算する（実装とは独立した参照）。 */
const expectedRows = (x: readonly number[], rows: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    let total = 0;
    for (let col = 0; col < 2; col += 1) {
      let dot = 0;
      for (let i = 0; i < 3; i += 1) dot += x[row * 3 + i] * W[i * 2 + col];
      total += Math.fround(Math.fround(dot) + B[col]);
    }
    out[row] = Math.fround(total);
  }
  return out;
};

Deno.test({
  name: "公開面だけでモデルを開き、記号次元込みのグラフを実行できる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const model = openModel(modelBytes());
    const session = await createSession(gpu, model);
    try {
      const rows = 4;
      const values = Array.from({ length: rows * 3 }, (_, i) => ((i % 9) - 4) * 0.5);
      const x: Tensor = { dtype: "f32", shape: [rows, 3], data: Float32Array.from(values) };

      const outputs = await session.run({ x });
      assertEquals(Object.keys(outputs), ["y"]);
      assertEquals(outputs["y"].shape, [rows]);
      const expected = expectedRows(values, rows);
      for (let row = 0; row < rows; row += 1) {
        assertAlmostEquals(outputs["y"].data[row], expected[row], 1e-5, `row ${row}`);
      }

      // 記号次元は run ごとに束縛し直される
      const wide = Array.from({ length: 9 * 3 }, (_, i) => (i % 5) - 2);
      const outputs2 = await session.run({
        x: { dtype: "f32", shape: [9, 3], data: Float32Array.from(wide) },
      }, { T: 9 });
      assertEquals(outputs2["y"].shape, [9]);
      const expected2 = expectedRows(wide, 9);
      for (let row = 0; row < 9; row += 1) {
        assertAlmostEquals(outputs2["y"].data[row], expected2[row], 1e-5, `wide row ${row}`);
      }

      const diagnostics = session.diagnostics();
      assertEquals(diagnostics.pipelineCount, 3);
      assertEquals(diagnostics.weights.allocCount, 2);
      assertEquals((diagnostics.lastRun?.allocCount ?? 0) > 0, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * 0 要素テンソル（shape に 0 を含む）の経路。要素数 0 は dispatch 数 0・確保サイズ 0 を
 * 生み、各所の下限クランプ（最小 4 バイト）に乗って通っている。shape を保ったまま長さ 0 の
 * data が返ること、そして同じ Session の後続 run が壊れないことをここで固定する。
 */
Deno.test({
  name: "0 要素テンソル（記号次元 T=0）が shape を保ったまま長さ 0 で返る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      const empty: Tensor = { dtype: "f32", shape: [0, 3], data: new Float32Array(0) };
      const outputs = await session.run({ x: empty });
      assertEquals(Object.keys(outputs), ["y"]);
      assertEquals(outputs["y"].shape, [0], "0 要素でも shape はスカラに縮退しない");
      assertEquals(outputs["y"].data.length, 0);
      // 3 ノードとも「要素ゼロの dispatch」としてエンコードされる（黙って飛ばさない）
      assertEquals(session.diagnostics().submit.dispatchCount, 3);

      // 0 要素の run が Session の状態（プール・パイプライン）を壊していない
      const values = [1, -2, 0.5, 3, 0.25, -1];
      const next = await session.run({
        x: { dtype: "f32", shape: [2, 3], data: Float32Array.from(values) },
      });
      assertEquals(next["y"].shape, [2]);
      const expected = expectedRows(values, 2);
      for (let row = 0; row < 2; row += 1) {
        assertAlmostEquals(next["y"].data[row], expected[row], 1e-5, `row ${row}`);
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test("公開面が capability 照会とモデル解析の失敗型を提供する", () => {
  const ops = capabilities().ops;
  for (const op of ["matmul", "add", "sum", "gelu", "amax"]) {
    assertEquals(ops.includes(op), true, op);
  }
  assertEquals(capabilities().storage, ["f16", "f32", "i32", "i8"]);
  assertThrows(() => openModel(new ArrayBuffer(4)), SafetensorsError);
});

Deno.test("この E2E は公開面（mod.ts）以外の実装を import しない", async () => {
  const source = await Deno.readTextFile(new URL(import.meta.url));
  const specifiers = [...source.matchAll(/from "([^"]+)"/g)].map((match) => match[1]);
  assertEquals(specifiers.filter((path) => path.includes("/src/")), []);
});
