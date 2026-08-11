// safe_softmax（ADR 0044）の**ビット同一の門**。
//
// 同じ入力に対し、
//
//   ① 既存 GPU op 列で組んだ「分解ガード相当」の参照
//      （`ge_scalar → sum → cast → bitwise_not → masked_fill(softmax(x), 空行, 0)`）
//   ② `safe_softmax` op 1 ノード
//
// を**同じ実 GPU**で流し、出力を **f32 のビット列**で突き合わせる。allclose ではなく
// `Uint32Array` の完全一致で見るのが要点で、tolerance に隠れる丸め列の変化（例: ②③ の縮約
// 順序が素の softmax からずれる / `1.0 / Σ` を割り算へ戻す）はここでしか検出できない。
//
// ① が torch のガード `where(¬any(¬eq(src,−inf)), 0, softmax(src))` の op ごとの写しである
// ことは {@link guardedGraph} の対応表が示す。
//
// MUST: 恒真化しないこと。①② は**別々のグラフ**を別々の Session で走らせる（safe_softmax の
// CPU 参照と突き合わせるだけでは、GPU 側の空行分岐が「素の softmax と同じ経路を通っている」
// ことを確かめられない）。
// MUST: **全 −inf 行を含むケース**を必ず回す（故障注入 — この行が無いと ① の masked_fill が
// 一度も発火せず、両者が素の softmax として一致するだけの恒真テストになる）。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** f32 の最小有限値。「行 max が −inf でない」は `amax_row >= この値` と同値。 */
const NEG_F32_MAX = -3.4028234663852886e+38;

/**
 * 分解ガード相当の参照グラフ（既存語彙だけで組む）。
 *
 * torch のガード `where(¬any(¬eq(src,−inf)), 0, softmax(src))` を op ごとに写したもの:
 *
 * | torch                 | ここ                                    |
 * | --------------------- | --------------------------------------- |
 * | `¬eq(src, −inf)`      | `ge_scalar(src, −F32_MAX)`（bool）      |
 * | `any(…, −1)`          | `sum`（bool → i32）→ `cast` to bool     |
 * | `logical_not`         | `bitwise_not`                           |
 * | `where(…, 0, softmax)`| `masked_fill(softmax(src), 空行, 0)`    |
 *
 * `ge_scalar(v, −F32_MAX)` が `¬eq(v, −inf)` と同値なのは、f32 の有限値が必ず −F32_MAX 以上
 * だから（入力は「有限か −inf」— ADR 0044 決定 3 の契約）。
 *
 * MUST: 行 max（`amax`）で空行を判定しない。ランタイムの行 reduce は identity が ±F32_MAX で
 * （limitations.md）、全 −inf 行の `amax` は −inf ではなく −F32_MAX を返す — masked_fill の
 * 埋め値と区別が付かない。要素ごとに判定してから畳むこの形が正しい写像。
 */
const guardedGraph = (rows: number, dim: number): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: {
    ops: ["softmax", "ge_scalar", "sum", "cast", "bitwise_not", "reshape", "masked_fill"],
  },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [rows, dim] }],
  outputs: ["y"],
  initializers: {},
  values: {
    probs: { dtype: "f32", shape: [rows, dim] },
    finiteEl: { dtype: "bool", shape: [rows, dim] },
    count: { dtype: "i32", shape: [rows] },
    anyFinite: { dtype: "bool", shape: [rows] },
    emptyRow: { dtype: "bool", shape: [rows] },
    empty: { dtype: "bool", shape: [rows, 1] },
    y: { dtype: "f32", shape: [rows, dim] },
  },
  nodes: [
    { op: "softmax", ins: ["x"], outs: ["probs"], attrs: { dim: 1 } },
    { op: "ge_scalar", ins: ["x"], outs: ["finiteEl"], attrs: { value: NEG_F32_MAX } },
    { op: "sum", ins: ["finiteEl"], outs: ["count"], attrs: { dim: 1 } },
    { op: "cast", ins: ["count"], outs: ["anyFinite"], attrs: { to: "bool" } },
    { op: "bitwise_not", ins: ["anyFinite"], outs: ["emptyRow"], attrs: {} },
    { op: "reshape", ins: ["emptyRow"], outs: ["empty"], attrs: {} },
    // mask は [rows,1] を右詰め broadcast で [rows,dim] へ広げて読む
    { op: "masked_fill", ins: ["probs", "empty"], outs: ["y"], attrs: { value: 0 } },
  ],
});

const run = async (
  gpu: GpuContext,
  graph: GraphJson,
  inputs: Readonly<Record<string, FilledTensor>>,
): Promise<Tensor> => {
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    return (await session.run(inputs))["y"];
  } finally {
    await session.dispose();
  }
};

/** f32 のビット列（`0.0` と `-0.0` の差も、末尾 1 ulp の差も見える形）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

/** 決定的なスコア列（乱数は使わない — 失敗が再現しないため）。 */
const SCORE = (i: number): number => (((i * 13) % 29) - 14) * 0.37;

type Case = {
  readonly name: string;
  readonly rows: number;
  readonly dim: number;
  /** 全要素 −inf にする行の集合。 */
  readonly emptyRows: readonly number[];
  /** 行内で個別に −inf にする列の判定（加算マスクの穴あき部分）。 */
  readonly blocked: (row: number, col: number) => boolean;
};

const CASES: readonly Case[] = [
  // 空行なし — ここは素の softmax とビット同一になる（変種が別実装へ退化していない門）
  { name: "空行なし [5,9]", rows: 5, dim: 9, emptyRows: [], blocked: () => false },
  // 実行時 bool マスクの実測形: 一部の列が −inf で、1 行だけ全 −inf
  {
    name: "全 −inf 行 1 本 [4,7]",
    rows: 4,
    dim: 7,
    emptyRows: [2],
    blocked: (row, col) => (row + col) % 3 === 2,
  },
  // 空行が連続する形（CFG の uncond バッチのように行がまとまって落ちる）
  {
    name: "空行 2 本連続 [6,5]",
    rows: 6,
    dim: 5,
    emptyRows: [1, 2],
    blocked: (_, col) => col === 0,
  },
  // 行長 > workgroup 幅（256）— ① の grid-stride 走査とツリー縮約の両方を通る
  {
    name: "行長 300（workgroup 超え）[3,300]",
    rows: 3,
    dim: 300,
    emptyRows: [1],
    blocked: (_, col) => col % 5 === 4,
  },
  // 行数 > workgroup 数の grid-stride も踏む（1 workgroup が複数行を畳む）
  {
    name: "行数 130（grid-stride）[130,4]",
    rows: 130,
    dim: 4,
    emptyRows: [0, 65, 129],
    blocked: (row, col) => (row + col) % 4 === 3,
  },
];

Deno.test({
  name:
    "safe_softmax 1 ノードの出力が分解ガード相当（softmax + masked_fill）と**ビット単位で一致**する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const { name, rows, dim, emptyRows } = testCase;
        const empty = new Set(emptyRows);
        const x = fill([rows, dim], (i) => {
          const row = Math.floor(i / dim);
          const col = i % dim;
          return empty.has(row) || testCase.blocked(row, col) ? Number.NEGATIVE_INFINITY : SCORE(i);
        });

        const safe = await run(
          gpu,
          singleOpGraph("safe_softmax", [[rows, dim]], [rows, dim], { attrs: { dim: 1 } }),
          { x0: x },
        );
        const guarded = await run(gpu, guardedGraph(rows, dim), { x });

        assertEquals(safe.shape, guarded.shape, name);
        const a = bits(safe);
        const b = bits(guarded);
        const mismatches: number[] = [];
        for (let i = 0; i < a.length; i += 1) {
          if (a[i] !== b[i]) mismatches.push(i);
        }
        assertEquals(
          mismatches.length,
          0,
          `${name}: ${mismatches.length} 要素がビット不一致（先頭 ${
            mismatches.slice(0, 4).map((i) =>
              `#${i} safe=0x${a[i].toString(16)} guarded=0x${b[i].toString(16)}`
            ).join(" / ")
          }）`,
        );

        // 故障注入の確認: 空行が本当に 0 で、非空行は 0 でない（両者が「素の softmax の
        // NaN 同士」で一致しているだけの状態を排除する）。
        const values = safe.data as Float32Array;
        for (let row = 0; row < rows; row += 1) {
          const slice = [...values.slice(row * dim, (row + 1) * dim)];
          assertEquals(slice.every(Number.isFinite), true, `${name}: 行 ${row} に非有限値`);
          if (empty.has(row)) {
            assertEquals(slice, Array(dim).fill(0), `${name}: 空行 ${row} が 0 でない`);
          } else {
            assert(slice.some((v) => v > 0), `${name}: 非空行 ${row} が全 0`);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

// 空行を持たない入力では safe_softmax は**素の softmax とビット同一**でなければならない
// （① の identity を −inf に変えた影響が有限行へ漏れていないことの直接の門 — 上の parity は
// masked_fill を挟むので、素の softmax そのものとの一致は別に見る）。
Deno.test({
  name: "safe_softmax は −inf を含まない入力で素の softmax とビット単位で一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const [rows, dim] of [[4, 9], [3, 300], [130, 4]] as const) {
        const x = fill([rows, dim], SCORE);
        const safe = await run(
          gpu,
          singleOpGraph("safe_softmax", [[rows, dim]], [rows, dim], { attrs: { dim: 1 } }),
          { x0: x },
        );
        const plain = await run(
          gpu,
          singleOpGraph("softmax", [[rows, dim]], [rows, dim], { attrs: { dim: 1 } }),
          { x0: x },
        );
        assertEquals([...bits(safe)], [...bits(plain)], `[${rows},${dim}]`);
      }
    } finally {
      gpu.destroy();
    }
  },
});
