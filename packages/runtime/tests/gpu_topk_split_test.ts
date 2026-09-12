import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, estimateSessionMemory, openModel } from "../mod.ts";
import { graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import type { GraphJson } from "./helpers/format.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

Deno.test({
  name: "topk k=1: 分割境界と大語彙でNaN・同点・符号付きゼロの値ビットを保つ",
  ignore: !GPU_AVAILABLE,
  async fn() {
    const gpu = await acquireGpu();
    try {
      for (const dim of [16383, 16384, 16385, 262144]) {
        const rows = 10,
          data = new Float32Array(rows * dim).fill(-Infinity),
          bits = new Uint32Array(data.buffer);
        const indices = [4095, 4096, 0, 7, 8, dim - 1, 4097, 10, 9, 17];
        const values = [
          0x40e00000,
          0xffc01234,
          0xff800000,
          0x80000000,
          0x7f800000,
          0x41400000,
          0x7fc05678,
          0x00000000,
          0x7fa01234,
          0xffa03456,
        ];
        for (let r = 0; r < rows; r++) bits[r * dim + indices[r]] = values[r];
        bits[4096] = values[0];
        bits[dim + 8192] = 0x7fc08765;
        bits[3 * dim + 8192] = 0;
        bits[4 * dim + 8192] = 0x7f800000;
        bits[6 * dim + 8192] = 0xffc04321;
        bits[7 * dim + 8192] = 0x80000000;
        const graph = singleOpGraph("topk", [[rows, dim]], [[rows, 1], [rows, 1]], {
          outDtypes: ["f32", "i32"],
          attrs: { k: 1 },
        });
        const model = openModel(graphModelBuffer(graph));
        const expected = estimateSessionMemory(model);
        const session = await createSession(gpu, model);
        try {
          const out = await session.run({
            x0: { dtype: "f32", shape: [rows, dim], data },
          });
          assert(out.y.dtype === "f32" && out.y1.dtype === "i32");
          assertEquals(Array.from(out.y1.data), indices);
          assertEquals(
            Array.from(
              new Uint32Array(
                out.y.data.buffer,
                out.y.data.byteOffset,
                out.y.data.length,
              ),
            ),
            values,
          );
          const actual = session.diagnostics().lastRun?.peakTransientBytes;
          assert(actual !== undefined);
          assert(expected.scenarios[0].workspaceBytes >= actual);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

const intermediateGraph = (rows: number, dim: number): GraphJson => ({
  format: "karume-ir",
  version: 1,
  symbols: [],
  requires: { ops: ["neg", "topk"] },
  inputs: [{ name: "x", dtype: "f32", shape: [rows, dim] }],
  outputs: ["value", "index"],
  initializers: {},
  values: {
    z: { dtype: "f32", shape: [rows, dim] },
    value: { dtype: "f32", shape: [rows, 1] },
    index: { dtype: "i32", shape: [rows, 1] },
  },
  nodes: [{ op: "neg", ins: ["x"], outs: ["z"], attrs: {} }, {
    op: "topk",
    ins: ["z"],
    outs: ["value", "index"],
    attrs: { k: 1 },
  }],
});

Deno.test({
  name: "topk k=1: 中間入力をmergeまで保ち初回とbacking再利用で同じ値を返す",
  ignore: !GPU_AVAILABLE,
  async fn() {
    const rows = 8, dim = 32769;
    const gpu = await acquireGpu();
    try {
      const session = await createSession(
        gpu,
        openModel(graphModelBuffer(intermediateGraph(rows, dim))),
      );
      try {
        for (let frame = 0; frame < 2; frame++) {
          const data = new Float32Array(rows * dim).fill(1000);
          const indices = [3, 4096, 0, 32768, 7, 16384, 9, 511].map((i) => (i + frame * 13) % dim);
          const values = Float32Array.from({ length: rows }, (_, i) => 10 + frame * 0.31 + i * 0.7);
          for (let row = 0; row < rows; row++) {
            const index = indices[row];
            data[row * dim + index] = -values[row];
            if (index + 1 < dim) data[row * dim + index + 1] = -values[row];
          }
          const outputs = await session.run({ x: { dtype: "f32", shape: [rows, dim], data } });
          assert(outputs.value.dtype === "f32" && outputs.index.dtype === "i32");
          assertEquals(Array.from(outputs.index.data), indices);
          assertEquals(Array.from(outputs.value.data), Array.from(values));
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test("topk1 mergeで同時に必要な元入力・部分結果・2出力を見積りに含める", () => {
  const rows = 512, dim = 16384;
  const model = openModel(graphModelBuffer(intermediateGraph(rows, dim)));
  const estimate = estimateSessionMemory(model);
  // mergeは元入力と部分結果を読みつつ、値と添字を別の領域に書く必要がある。
  const minimum = rows * dim * 4 + rows * 4 * 8 + rows * 8;
  const workspace = estimate.scenarios[0].workspaceBytes;
  assert(workspace >= minimum, `workspace ${workspace} < simultaneously live ${minimum}`);
});

Deno.test({
  name: "topk k=1: 外側が空なら分割用一時を作らず空の2出力を返す",
  ignore: !GPU_AVAILABLE,
  async fn() {
    const gpu = await acquireGpu();
    try {
      for (const dim of [16383, 16384, 262144]) {
        const model = openModel(graphModelBuffer(singleOpGraph(
          "topk",
          [[0, dim]],
          [[0, 1], [0, 1]],
          { outDtypes: ["f32", "i32"], attrs: { k: 1 } },
        )));
        const expected = estimateSessionMemory(model);
        const session = await createSession(gpu, model);
        try {
          for (let frame = 0; frame < 2; frame++) {
            const out = await session.run({
              x0: { dtype: "f32", shape: [0, dim], data: new Float32Array(0) },
            });
            assertEquals(out.y.data.length, 0);
            assertEquals(out.y1.data.length, 0);
            const actual = session.diagnostics().lastRun?.peakTransientBytes;
            assert(actual !== undefined);
            assert(expected.scenarios[0].workspaceBytes >= actual);
          }
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
