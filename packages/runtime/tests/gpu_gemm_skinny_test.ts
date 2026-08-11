// skinny-M タイル幾何（src/kernels/gemm-geometry.ts の `gemmGeometryForRows`）の実 GPU 検証。
//
// 幾何が決めてよいのは**担当割りだけ**（ADR 0022）。したがって見るべきは 2 つで、どちらも
// 「小さい M でだけ別の幾何が走る」ことを前提に組んである:
//
// 1. **CPU 参照との一致** — 小 M（M <= 64）/ 中 M（65..512）/ 既定の 3 バケットと、
//    その境界を跨ぐ形。
//    タイル辺が変わると端数タイルの位置も変わるので、`m % tileM != 0` / `n % tileN != 0` /
//    `k % 16 != 0` を混ぜる。dispatch 数と生成物の幾何が食い違えば出力タイルが丸ごと欠けるが、
//    その欠落はここで参照との差として出る。
// 2. **バケットを跨いだビット同一** — 同じ重み・同じ行の値なら、M=4（小 M 幾何）の出力と
//    M=128（既定幾何）の出力の先頭 4 行は**1 ビットも違ってはならない**。1 出力要素あたりの
//    K 縮約順（外側 t 昇順・内側 kk 昇順・K タイル 16）が幾何によらず同一だからで、これが
//    「幾何を差し替えても PNG / WAV の sha256 門が割れない」という実測命題の単体版にあたる。
//    MUST: 割れたら幾何の選択ではなく**命題そのもの**を疑う（設計判断へ戻す）。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, type RefTensor } from "../src/reference/ops.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 桁落ちが起きる程度に符号と大きさを散らした決定的な列（乱数は使わない）。 */
const XS = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;
const WS = (index: number): number => ((index % 7) - 3) * 0.5 - 0.0625;
const BS = (index: number): number => ((index % 5) - 2) * 0.25;

type ShapeCase = {
  readonly name: string;
  readonly op: "linear" | "matmul" | "bmm";
  readonly inputs: readonly RefTensor[];
  readonly outShape: readonly number[];
};

const runCase = async (gpu: GpuContext, testCase: ShapeCase): Promise<Tensor> => {
  const graph = singleOpGraph(
    testCase.op,
    testCase.inputs.map((input) => input.shape),
    testCase.outShape,
  );
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    const named: Record<string, Tensor> = {};
    testCase.inputs.forEach((input, index) => {
      named[`x${index}`] = input;
    });
    return (await session.run(named))["y"];
  } finally {
    await session.dispose();
  }
};

const linearCase = (m: number, k: number, n: number): ShapeCase => ({
  name: `linear m${m} k${k} n${n}`,
  op: "linear",
  inputs: [fill([m, k], XS), fill([n, k], WS), fill([n], BS)],
  outShape: [m, n],
});

const matmulCase = (m: number, k: number, n: number): ShapeCase => ({
  name: `matmul m${m} k${k} n${n}`,
  op: "matmul",
  inputs: [fill([m, k], XS), fill([k, n], WS)],
  outShape: [m, n],
});

const bmmCase = (batch: number, m: number, k: number, n: number): ShapeCase => ({
  name: `bmm b${batch} m${m} k${k} n${n}`,
  op: "bmm",
  inputs: [fill([batch, m, k], XS), fill([batch, k, n], WS)],
  outShape: [batch, m, n],
});

/**
 * 形状の選定。**バケットの内側と、境界を 1 越えた側の両方**を持つ（境界を片側しか踏まないと、
 * 表の比較演算子が `<` か `<=` かの取り違えを検出できない）。
 * `n` は v4 経路（`k % 4 == 0 && n % 4 == 0`）とスカラ経路の両方を通す。
 */
const SHAPE_CASES: readonly ShapeCase[] = [
  // 小 M バケット（tileM 16 / tileN 16）。実測形の M=1 / M=4 を含む
  linearCase(1, 20, 68),
  linearCase(4, 20, 68),
  linearCase(4, 37, 23),
  linearCase(1, 1, 7),
  linearCase(16, 132, 64),
  // 同バケット上半分（17〜64 — 行タイルが 2〜4 枚になる領域）。下端 17 と上端 64
  linearCase(17, 64, 33),
  linearCase(33, 16, 68),
  linearCase(64, 20, 68),
  // 中 M バケット（65..512 — tileM 64 / tileN 32）。下端 65 と上端 512
  linearCase(65, 16, 16),
  linearCase(127, 40, 132),
  linearCase(512, 20, 36),
  // 既定幾何へ落ちる側（境界の外）
  linearCase(513, 16, 16),
  matmulCase(4, 20, 68),
  matmulCase(5, 37, 23),
  matmulCase(17, 16, 64),
  matmulCase(64, 132, 36),
  matmulCase(65, 20, 20),
  matmulCase(513, 8, 12),
  // bmm はバッチ軸（dispatch の z）と幾何が独立であることも同時に見る形（B / M / K / N 全て別）
  bmmCase(3, 4, 20, 68),
  bmmCase(2, 5, 7, 9),
  bmmCase(3, 17, 12, 36),
  bmmCase(2, 65, 16, 16),
];

Deno.test({
  name: "小 M の GEMM 3 op はバケット境界を跨いで CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of SHAPE_CASES) {
        const actual = await runCase(gpu, testCase);
        const expected = applyReferenceOp(testCase.op, testCase.inputs, {}, testCase.outShape);
        assertEquals(actual.shape, expected.shape, testCase.name);
        const report = compareTensors(actual, expected);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

/** f32 の生ビット列（-0 / NaN も含めて 1 ビットの差を見る）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const assertRowsBitEqual = (
  actual: Tensor,
  expected: Tensor,
  columns: number,
  rows: number,
  where: string,
): void => {
  const a = bits(actual);
  const b = bits(expected);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const index = row * columns + col;
      assert(
        a[index] === b[index],
        `${where}: [${row},${col}] が別ビット（0x${a[index].toString(16)} vs ` +
          `0x${b[index].toString(16)} = ${actual.data[index]} vs ${expected.data[index]}）`,
      );
    }
  }
};

Deno.test({
  name: "バケットが違っても同じ行の出力は 1 ビットも違わない（幾何は担当割りのみ / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const k = 132;
    const n = 68;
    // 行数だけを変えた 4 本。513 = 既定幾何（基準）/ 128 = 中 M（M64N32）/ 33・4 = 小 M
    //（M16N16 — 同じ幾何でも端数タイルの位置が違う 2 点）。全バケットが基準と
    // 1 ビットも違わないことを見る
    const rowCounts = [513, 128, 33, 4] as const;
    const gpu = await acquireGpu();
    try {
      for (const op of ["linear", "matmul"] as const) {
        const weight = op === "linear" ? fill([n, k], WS) : fill([k, n], WS);
        const bias = fill([n], BS);
        const outputs = new Map<number, Tensor>();
        for (const rows of rowCounts) {
          // 行優先なので平坦添字 `r·k + c` は行数によらず同じ = 行 r の入力値は 3 本で共通
          const x = fill([rows, k], XS);
          const inputs = op === "linear" ? [x, weight, bias] : [x, weight];
          outputs.set(
            rows,
            await runCase(gpu, {
              name: `${op} m${rows}`,
              op,
              inputs,
              outShape: [rows, n],
            }),
          );
        }
        const reference = outputs.get(513);
        assert(reference !== undefined);
        for (const rows of [128, 33, 4] as const) {
          const actual = outputs.get(rows);
          assert(actual !== undefined);
          assertRowsBitEqual(actual, reference, n, rows, `${op} m${rows} vs m513`);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
