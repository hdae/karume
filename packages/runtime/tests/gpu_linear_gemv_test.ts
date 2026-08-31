// linear の **GEMV 族**（M=1 × 重み i4 — ADR 0082）の実 GPU 門。
//
// この族の存在理由は速度だけで、**値は既定経路（src/kernels/gemm.ts の linear）と 1 ビットも
// 違ってはならない**（ADR 0022 決定 3 が自由と認めるのは担当割りだけ）。よって見るのは 3 つ:
//
// 1. **既定経路との u32 完全一致** — 同じ重み・同じ入力行なら、M=1（GEMV 経路）の出力と
//    M=2（既定の M16N16 幾何）の**先頭行**が 1 ビットも違わない。1 出力要素あたりの K 縮約順が
//    経路によらず「k 昇順の逐次」で同じことの直接証明で、tests/gpu_gemm_skinny_test.ts の
//    「バケット跨ぎビット同一」を族跨ぎへ延長した形にあたる。
//    MUST: 割れたら変種の選択ではなく**ビット同一という命題そのもの**を疑う（設計判断へ戻す）。
// 2. **CPU 参照との一致** — 1 だけだと「両経路が同じだけ壊れている」を排除できない（構造上
//    起こりにくいが、比較対象の M=2 が何かの拍子に GEMV へ流れていれば恒真化する）。
// 3. **門（分岐の適用条件）そのもの** — パイプラインキーの側から、M=1 でだけ GEMV 族が立ち、
//    M>1・group 16・n%4≠0 では従来の `linear:` キーのままであることを見る。
//
// MUST: 形は **本番 12 形が一度も踏まない端**を専用に持つ（実モデルの decode は
// `n % 32 == 0` かつ `units % 4 == 0` しか出さないので、端の workgroup の早期 return も
// 先読みループの端数も本番形では 1 度も走らない）:
//   - `n % 32 != 0` — 最終 workgroup が部分的（`col >= dims.n` の早期 return が効く）
//   - `units % 4 != 0`（`units = k / 32`）— 先読み 4 本のループが端数を残す
//   - `units < 4` — 先読みループが**一度も回らず**端数ループだけで縮約が終わる
// MUST: 重みは **group ごとに大きさを変える**（scale が全 group で同じだと、group scale の
// 添字〈`(unit · 32) >> shift`〉の取り違えが一切値に出ない — gpu_i4_weights_test.ts と同じ罠）。
// MUST: 隣接要素の符号を交互にする（pack の上下 nibble の取り違えは対称パターンでは値が合う）。
//
// 検出できる変異（設計時に確認した故障注入 — 2026-08-31）:
// - group scale の shift を 1 段ずらす（`>> shift` → `>> shift+1`）→ 1 が落ちる
//   （差は 3 倍規模で、2 の allclose でも落ちる）
// - 語内の積和を「上位 nibble 先」へ並べ替える（積の集合は同じで**加算順だけ**が変わる）→
//   1 だけが落ちる（実測差は 1 ULP = `0xc083c922` vs `0xc083c921` で、2 の allclose は素通り）。
//   **u32 完全一致でなければ意味を持たない門**であることの実証。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { linearGemvKey } from "../src/kernels/linear-gemv.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, type FilledTensor } from "./helpers/graph.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/**
 * 活性。平坦添字の関数なので、**行 0 の値は m によらず同じ**（行優先で `r·k + c`）—
 * M=1 と M=2 の先頭行を突き合わせられる条件そのもの。桁落ちが起きる程度に符号と大きさを散らす。
 */
const XS = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;

/** bias（GEMV では縮約の**外**で最後に 1 度だけ足される — 順序が動けば値に出る）。 */
const BS = (index: number): number => ((index % 5) - 2) * 0.25;

/** group ごとに振幅が違い、隣接要素の符号が交互になる重み（上の 2 つの MUST）。 */
const weightAt = (cols: number, groupSize: number) => (index: number): number => {
  const group = Math.floor((index % cols) / groupSize);
  const row = Math.floor(index / cols);
  const base = (0.125 + (index % 11) * 0.5) * (index % 2 === 0 ? 1 : -1);
  return base * (1 + group * 0.75 + (row % 5) * 0.25);
};

type GemvCase = {
  readonly name: string;
  readonly k: number;
  readonly n: number;
  readonly groupSize: number;
};

/**
 * 形の選定。`units = k / 32`（重み語の本数）と `n % 32`（最終 workgroup の埋まり方）が
 * 独立の軸で、group 長は scale 添字の shift を動かす軸。
 * MUST: `n % 4 == 0`（門が v4 を要求する — recipe-builder の `#buildLinear`）。
 */
const CASES: readonly GemvCase[] = [
  // 端がどこにも無い基準形（units = 4 ちょうど・n は workgroup 2 枚ちょうど）
  { name: "整除形 k128 n64 g32", k: 128, n: 64, groupSize: 32 },
  // units = 5 → 先読み 4 本の後に端数 1 本 / n = 100 は最終 workgroup が 4 列だけ
  { name: "端数 units5 n100 g32", k: 160, n: 100, groupSize: 32 },
  // units = 6（端数 2 本）・group 64 = 1 語おきに scale が変わる形
  { name: "端数 units6 n36 g64", k: 192, n: 36, groupSize: 64 },
  // units = 8・group 128 = 4 語で 1 scale（shift の焼き込みが最も効く形）
  { name: "整除形 units8 n68 g128", k: 256, n: 68, groupSize: 128 },
  // units = 3 < 先読み 4 → 先読みループが一度も回らない（端数ループだけで縮約が終わる）
  { name: "先読み不成立 units3 n4 g32", k: 96, n: 4, groupSize: 32 },
  // units = 7（端数 3 本）・n = 32 は workgroup ちょうど 1 枚
  { name: "端数 units7 n32 g32", k: 224, n: 32, groupSize: 32 },
];

/** `linear(x, w, b)` 1 本のグラフ（w は i4 + group scale）。`m` だけが経路を分ける。 */
const linearI4Model = (
  testCase: GemvCase,
  m: number,
  quantized: ReturnType<typeof quantizeI4>,
  bias: FilledTensor,
): ArrayBuffer => {
  const { k, n, groupSize } = testCase;
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: groupSize } },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  return buildSafetensors(
    [
      { name: "m.w", dtype: "I4", shape: [n, k], data: quantized.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [...quantized.scaleShape],
        data: f32Bytes([...quantized.scale]),
      },
      { name: "m.b", dtype: "F32", shape: [n], data: f32Bytes([...bias.data]) },
    ],
    { karume_ir: JSON.stringify(graph) },
  );
};

type RunResult = {
  readonly output: Tensor;
  /** その run で実際に走ったパイプラインキー（`timestamp-query` 不在なら空）。 */
  readonly keys: readonly string[];
};

const runLinear = async (
  gpu: GpuContext,
  testCase: GemvCase,
  m: number,
  quantized: ReturnType<typeof quantizeI4>,
  bias: FilledTensor,
): Promise<RunResult> => {
  const session = await createSession(gpu, openModel(linearI4Model(testCase, m, quantized, bias)));
  try {
    const output = (await session.run({ x: fill([m, testCase.k], XS) }))["y"];
    const entries = session.diagnostics().lastRunTiming?.entries ?? [];
    return { output, keys: entries.map((entry) => entry.key) };
  } finally {
    await session.dispose();
  }
};

/** f32 の生ビット列（-0 / NaN も含めて 1 ビットの差を見る）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

Deno.test({
  name: "M=1 の i4 linear は GEMV 族で走っても既定経路と 1 ビットも違わない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const testCase of CASES) {
        const { name, k, n, groupSize } = testCase;
        const weight = fill([n, k], weightAt(k, groupSize));
        const bias = fill([n], BS);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);

        const gemv = await runLinear(gpu, testCase, 1, quantized, bias);
        // 比較相手は M=2（既定の M16N16 幾何）。1 出力要素あたりの K 縮約順は M に依らないので
        // （ADR 0022 決定 3 — gpu_gemm_skinny_test.ts のバケット跨ぎ門が同じ命題を見ている）、
        // 先頭行が GEMV の全出力に対する参照になる。
        const gemm = await runLinear(gpu, testCase, 2, quantized, bias);

        assertEquals(gemv.output.shape, [1, n], `${name}: 出力の形`);
        const actual = bits(gemv.output);
        const expected = bits(gemm.output);
        for (let col = 0; col < n; col += 1) {
          assert(
            actual[col] === expected[col],
            `${name}: 列 ${col} が既定経路と別ビット（0x${actual[col].toString(16)} vs ` +
              `0x${expected[col].toString(16)} = ${gemv.output.data[col]} vs ` +
              `${gemm.output.data[col]}）`,
          );
        }

        // 「両経路が同じだけ壊れている」を排除する（比較相手が GEMV へ流れていれば 1 は恒真）。
        const reference = applyReferenceOp(
          "linear",
          [
            fill([1, k], XS) as RefTensor,
            refTensor(weight.shape, quantized.values),
            bias as RefTensor,
          ],
          {},
          [1, n],
        );
        const report = compareTensors(gemv.output, reference, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

/** 門（`#buildLinear` の分岐）1 件ぶんの期待キー。 */
type DoorCase = {
  readonly name: string;
  readonly shape: GemvCase;
  readonly m: number;
  /** 真ならこの形は GEMV 族へ、偽なら既定の `linear:` へ落ちる。 */
  readonly gemv: boolean;
  /** 既定へ落ちる形で期待する `linear:` キー（GEMV へ入る形では未使用）。 */
  readonly fallback?: string;
};

/**
 * MUST: 門の**各条件を 1 つずつだけ外した形**を並べる。まとめて外すと、どの条件が門を
 * 閉じているのか（あるいは条件が 1 つも効いていないのか）が区別できない。
 */
const DOOR_CASES: readonly DoorCase[] = [
  {
    name: "M=1 × i4 × g32 × v4",
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: 1,
    gemv: true,
  },
  {
    // 行数だけを外す（decode 以外は従来どおり）
    name: "M=2（行数の条件だけ外す）",
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: 2,
    gemv: false,
    fallback: linearKey("i4", true, "f32", 2, 32),
  },
  {
    // group 長だけを外す（16 = 1 語 32 要素が group を跨ぐ形）
    name: "group 16（group 長の条件だけ外す）",
    shape: { name: "", k: 128, n: 64, groupSize: 16 },
    m: 1,
    gemv: false,
    fallback: linearKey("i4", true, "f32", 1, 16),
  },
  {
    // v4 だけを外す（n % 4 != 0 — 既定のスカラ変種へ）
    name: "n=33（v4 の条件だけ外す）",
    shape: { name: "", k: 128, n: 33, groupSize: 32 },
    m: 1,
    gemv: false,
    fallback: linearKey("i4", false, "f32", 1, 32),
  },
];

Deno.test({
  name: "GEMV 族の門は M=1 × i4 × group32 以上 × v4 でだけ開く（実 GPU / 診断キー）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const door of DOOR_CASES) {
        const { k, n, groupSize } = door.shape;
        const weight = fill([n, k], weightAt(k, groupSize));
        const bias = fill([n], BS);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);
        const { keys } = await runLinear(gpu, door.shape, door.m, quantized, bias);
        // MUST: 列挙が無い device では診断が空になる（キー検査は数値側の門に任せて素通り）。
        if (keys.length === 0) continue;
        const shown = keys.join(" / ");
        const gemvKey = linearGemvKey(groupSize >= 32 ? groupSize : 32);
        assertEquals(
          keys.includes(gemvKey),
          door.gemv,
          `${door.name}: GEMV 族のキー（${gemvKey}）の有無が期待と違う（走った内訳: ${shown}）`,
        );
        if (!door.gemv) {
          assert(
            door.fallback !== undefined && keys.includes(door.fallback),
            `${door.name}: 既定経路のキー ${door.fallback} で走っていない（内訳: ${shown}）`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
