import { assert, assertEquals } from "@std/assert";
import { acquireGpu, createSession, openModel } from "../mod.ts";
import { graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** CPU の通常の数値比較から順位を作る。GPU の整数順序変換を複製しない。 */
const rank = (data: Float32Array<ArrayBuffer>, base: number, dim: number): number[] =>
  Array.from({ length: dim }, (_, i) => i).sort((a, b) => {
    const va = data[base + a], vb = data[base + b];
    if (Number.isNaN(va)) return Number.isNaN(vb) ? a - b : -1;
    if (Number.isNaN(vb)) return 1;
    return va > vb ? -1 : va < vb ? 1 : a - b;
  });

Deno.test({
  name: "argmax/topk: 非正規数をゼロと区別し、短い行と分割境界でCPUと同じ値・添字を返す",
  ignore: !GPU_AVAILABLE,
  async fn() {
    const cases = [
      [0, 1, 2, 0x80000001],
      [0x80000002, 0x80000001, 0x80000003, 0xff800000],
      [0x80000000, 0, 1, 0x007fffff],
      [0x80000001, 0, 0x80000000, 0xff800000],
      [0, 0x80000000, 0x00800000, 0x007fffff],
      [0xff800000, 0x807fffff, 0x80800000, 0xff7fffff],
      [0, 1, 1, 0x80000000],
      [0xffa00001, 0x7fc00002, 0x7f800000, 0xffc00003],
      [0x7f800000, 0x7f7fffff, 0x7f800000, 0],
      [0x80000000, 0, 0x80000000, 0],
    ];
    const gpu = await acquireGpu();
    try {
      for (const dim of [4, 257, 16383, 16384, 16385, 262144]) {
        const rows = cases.length, data = new Float32Array(rows * dim).fill(-Infinity);
        const bits = new Uint32Array(data.buffer);
        // 最小 index 同点を、レーン・部分区間の境界を越えて検査する。
        const positions = [0, Math.floor(dim / 3), Math.floor(dim * 2 / 3), dim - 1];
        for (let r = 0; r < rows; r++) {
          for (let i = 0; i < positions.length; i++) bits[r * dim + positions[i]] = cases[r][i];
        }
        const expected = cases.map((_, r) => rank(data, r * dim, dim)[0]);
        for (const op of ["argmax", "topk"] as const) {
          const graph = singleOpGraph(
            op,
            [[rows, dim]],
            op === "topk" ? [[rows, 1], [rows, 1]] : [[rows, 1]],
            {
              outDtypes: op === "topk" ? ["f32", "i32"] : ["i32"],
              attrs: op === "topk" ? { k: 1 } : {},
            },
          );
          const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
          try {
            for (let repeat = 0; repeat < 2; repeat++) {
              const out = await session.run({ x0: { dtype: "f32", shape: [rows, dim], data } });
              const indices = op === "topk" ? out.y1 : out.y;
              assert(indices.dtype === "i32");
              assertEquals([...indices.data], expected, `${op}, dim=${dim}, repeat=${repeat}`);
              if (op === "topk") {
                assert(out.y.dtype === "f32");
                assertEquals(
                  [...new Uint32Array(out.y.data.buffer, out.y.data.byteOffset, out.y.data.length)],
                  expected.map((i, r) => bits[r * dim + i]),
                  "選択した値のビット列を保つ",
                );
              }
            }
          } finally {
            await session.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "topk: 非正規数・NaN・符号付きゼロの複数候補をCPUと同じ順序・値ビットで返す",
  ignore: !GPU_AVAILABLE,
  async fn() {
    const edgeBits = [
      0x80000000,
      1,
      0x80000001,
      0x007fffff,
      0x807fffff,
      0,
      0x00800000,
      0x80800000,
      0x7f800000,
      0xff800000,
      0x7fa12345,
      0xffc05678,
      0x7f7fffff,
      0xff7fffff,
      2,
      0x80000002,
    ];
    const gpu = await acquireGpu();
    try {
      for (const [dim, k] of [[16, 16], [257, 4], [257, 63], [16385, 4]]) {
        const rows = 4, data = new Float32Array(rows * dim), bits = new Uint32Array(data.buffer);
        let state = 123456789;
        for (let i = 0; i < bits.length; i++) {
          state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
          const row = Math.floor(i / dim);
          // row 0/1 は全要素が正/負の非正規数、row 2 は境界と重複、row 3 は全ビット分布。
          bits[i] = row === 0
            ? state & 0x7fffff
            : row === 1
            ? (state & 0x7fffff) | 0x80000000
            : row === 2
            ? edgeBits[i % edgeBits.length]
            : state;
        }
        const expected = Array.from(
          { length: rows },
          (_, r) => rank(data, r * dim, dim).slice(0, k),
        );
        const session = await createSession(
          gpu,
          openModel(graphModelBuffer(singleOpGraph(
            "topk",
            [[rows, dim]],
            [[rows, k], [rows, k]],
            { outDtypes: ["f32", "i32"], attrs: { k } },
          ))),
        );
        try {
          const out = await session.run({ x0: { dtype: "f32", shape: [rows, dim], data } });
          assert(out.y.dtype === "f32" && out.y1.dtype === "i32");
          assertEquals([...out.y1.data], expected.flat(), `dim=${dim}, k=${k}`);
          assertEquals(
            [...new Uint32Array(out.y.data.buffer, out.y.data.byteOffset, out.y.data.length)],
            expected.flatMap((row, r) => row.map((i) => bits[r * dim + i])),
            "全候補の値ビットを保つ",
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
