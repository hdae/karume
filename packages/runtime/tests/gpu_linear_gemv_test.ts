// linear の **GEMV 族**（1 ≤ M ≤ {@link LINEAR_GEMV_MAX_ROWS} × 重み i4 / i8 — ADR 0082 /
// perf-ledger K-16（M=1）/ K-21（M ≥ 2 の行ブロック））の実 GPU 門。
//
// この族の存在理由は速度だけで、**値は既定経路（src/kernels/gemm.ts の linear）と 1 ビットも
// 違ってはならない**（ADR 0022 決定 3 が自由と認めるのは担当割りだけ）。よって見るのは 3 つ:
//
// 1. **既定経路との u32 完全一致** — 同じ重み・同じ入力行なら、GEMV 経路の出力と既定 GEMM の
//    **先頭 m 行**が 1 ビットも違わない。1 出力要素あたりの K 縮約順が経路によらず
//    「k 昇順の逐次」で同じことの直接証明で、tests/gpu_gemm_skinny_test.ts の
//    「バケット跨ぎビット同一」を族跨ぎへ延長した形にあたる。
//    MUST: 割れたら変種の選択ではなく**ビット同一という命題そのもの**を疑う（設計判断へ戻す）。
//    MUST: 比較相手は **M = {@link GEMM_ROWS}（= 上限 + 1）**で走らせる。上限までは行ブロック
//    GEMV が受けるようになった（K-21）ので、以前の M=2 は既定経路ではなく**同じ族の別変種**に
//    なり、突き合わせが族内の自己比較へ退化する。活性 `XS` は平坦添字の関数（行優先 r·k + c）
//    なので、行 0..m−1 の値は m によらず同じ = 先頭 m 行がそのまま参照になる。
// 2. **CPU 参照との一致** — 1 だけだと「両経路が同じだけ壊れている」を排除できない（構造上
//    起こりにくいが、比較対象が何かの拍子に GEMV へ流れていれば恒真化する）。
// 3. **門（分岐の適用条件）そのもの** — パイプラインキーの側から、1 ≤ M ≤ 上限でだけ GEMV 族が
//    立ち（M=1 は M=1 変種・M ≥ 2 は行ブロック変種）、上限 + 1・group 16・n%4≠0 では従来の
//    `linear:` キーのままであることを見る。
//
// **行ブロック（M ≥ 2）が足す形の軸**（M=1 では 1 度も走らない断片）:
//   - `rows > 1` — 1 スレッドが重み語を 1 回読んで `rows` 行と積和する（行間で共有するのは
//     整数の重み語と scale だけ）
//   - **y タイル複数**（`ceil(m / rows) > 1`）— 重みをタイルごとに読み直す
//   - **部分的な最終 y タイル**（`m % rows != 0`）— 範囲外の行は x の最終行を読み直し、
//     書き戻しだけを `row < dims.m` で落とす（読みの範囲外と書きの範囲外が別処理）
//   - `rows == 1` の y タイル形 — M=1 変種とは別テキスト（行頭 `xr0` 経由の x 読み + 書き戻し
//     ガード）で、n が小さい形では上限の M でもこれが選ばれる
//
// MUST: 形は **本番 12 形が一度も踏まない端**を専用に持つ（実モデルの decode は
// `n % 32 == 0` かつ `units % 4 == 0` しか出さないので、端の workgroup の早期 return も
// 先読みループの端数も本番形では 1 度も走らない）:
//   - `n % 32 != 0` — 最終 workgroup が部分的（`col >= dims.n` の早期 return が効く）
//   - `units % 4 != 0`（`units = k / 32`）— 先読み 4 本のループが端数を残す
//   - `units < 4` — 先読みループが**一度も回らず**端数ループだけで縮約が終わる
// MUST: 重みは **scale の単位ごとに大きさを変える**（i4 = group ごと・i8 = 出力チャネルごと）。
// 全単位で同じ scale だと、添字〈i4 は `(unit · 32) >> shift`・i8 は `wscale[col]`〉の
// 取り違えが一切値に出ない（gpu_i4_weights_test.ts / gpu_i8_weights_test.ts と同じ罠）。
// MUST: 隣接要素の符号を交互にする（i4 は pack の上下 nibble・i8 は語内レーンの取り違えが、
// 対称パターンでは値の上で打ち消し合う）。
//
// 格納 2 種は**同じ 3 観点を別々に**踏む（i4 = 1 語 32 要素 × 語ごとの group scale /
// i8 = 1 語 16 要素 × 縮約の外で 1 度だけ引くチャネル scale — 生成の別枝なので片方の緑は
// もう片方の根拠にならない）。i8 の `units` は `k / 16` で、端の 3 条件は同じ意味を持つ。
//
// 検出できる変異（設計時に確認した故障注入 — 2026-08-31 / 行ブロックは 2026-09-07）:
// - group scale の shift を 1 段ずらす（`>> shift` → `>> shift+1`）→ 1 が落ちる
//   （差は 3 倍規模で、2 の allclose でも落ちる）
// - 語内の積和を「上位 nibble 先」へ並べ替える（積の集合は同じで**加算順だけ**が変わる）→
//   1 だけが落ちる（実測差は 1 ULP = `0xc083c922` vs `0xc083c921` で、2 の allclose は素通り）。
//   **u32 完全一致でなければ意味を持たない門**であることの実証。
// - **行ブロック側だけ**同じ並べ替えを入れる（`unitMacsI4` の行ごとの MAC で下位 / 上位 nibble の
//   2 行を入れ替え、M=1 変種〈`row === undefined`〉の生成物は 1 バイトも動かさない）→
//   **行ブロックの 1 だけ**が落ちる（i4 の rows4 ケース・行 0 列 2 で `0xc083c922` vs
//   `0xc083c921` = 1 ULP。差 4.8e-7 / rel 1.2e-7 なので 2 の allclose は素通りし、3 も緑）。
//   同時に M=1 の 3 観点・i8 側・M=1 の WGSL スナップショット 3 本は全て緑のままで、動くのは
//   行ブロックのスナップショット（linear_gemv_r*.wgsl）だけ — **M=1 の緑は行ブロックの
//   根拠にならない**ことの実証（2026-09-07 実測）。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  defaultLinearGemvRowsVariant,
  LINEAR_GEMV_MAX_ROWS,
  linearGemvKey,
  linearGemvRowsForShape,
  linearGemvRowsKey,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, type FilledTensor } from "./helpers/graph.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/**
 * 活性。平坦添字の関数なので、**行 r の値は m によらず同じ**（行優先で `r·k + c`）—
 * GEMV の m 行と既定 GEMM の先頭 m 行を突き合わせられる条件そのもの。桁落ちが起きる程度に
 * 符号と大きさを散らす。
 */
const XS = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;

/**
 * ビット同一の比較相手を走らせる行数 = **門の外の最小値**（上限 + 1）。
 * MUST: 上限以下を比較相手にしない — GEMV 族が受ける区間なので、突き合わせが族内の
 * 自己比較へ退化して恒真になる。
 * NOTE: 上限 + 1 = 65 は幾何 M64N32（gemm-geometry.ts の `gemmGeometryForRows`）で、GEMV 族が
 * 実際に置き換える M ≤ 64 の幾何は M16N16。この門が直接見るのは「GEMV == M64N32 の GEMM」で、
 * 「== M16N16」は幾何跨ぎのビット同一（gpu_gemm_skinny_test.ts のバケット跨ぎ門）を経由した推移。
 */
const GEMM_ROWS = LINEAR_GEMV_MAX_ROWS + 1;

/**
 * 比較相手が**本当に既定経路で走った**ことの検査（診断が取れる device のみ）。
 * MUST: 門の上限（recipe-builder）と `LINEAR_GEMV_MAX_ROWS` がずれて比較相手が GEMV 族へ流れると、
 * u32 一致は族内の自己比較になって恒真化する — その退化をここで機械的に止める。
 */
const assertRanGemm = (name: string, keys: readonly string[]): void => {
  if (keys.length === 0) return;
  assert(
    !ranGemv(keys),
    `${name}: 比較相手（M=${GEMM_ROWS}）が GEMV 族で走った（内訳: ${keys.join(" / ")}）`,
  );
};

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
 * MUST: `n % 4 == 0`（門が v4 を要求する — recipe-builders/linear.ts の `buildLinear`）。
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

/** 組み上げたモデル 1 本を走らせて出力と走ったキーを返す（格納 2 種で共有）。 */
const runLinear = async (
  gpu: GpuContext,
  model: ArrayBuffer,
  m: number,
  k: number,
): Promise<RunResult> => {
  const session = await createSession(gpu, openModel(model));
  try {
    const output = (await session.run({ x: fill([m, k], XS) }))["y"];
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

        const gemv = await runLinear(gpu, linearI4Model(testCase, 1, quantized, bias), 1, k);
        // 比較相手は M = 上限 + 1（門の外 = 既定の GEMM 骨格）。1 出力要素あたりの K 縮約順は
        // M に依らないので（ADR 0022 決定 3 — gpu_gemm_skinny_test.ts のバケット跨ぎ門が同じ
        // 命題を見ている）、先頭行が GEMV の全出力に対する参照になる。
        const gemm = await runLinear(
          gpu,
          linearI4Model(testCase, GEMM_ROWS, quantized, bias),
          GEMM_ROWS,
          k,
        );
        assertRanGemm(name, gemm.keys);

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

/** 門（`buildLinear` の分岐）1 件ぶんの期待キー。 */
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
 * 診断キーの列から「GEMV 族が走ったか」を族名だけで判定する。
 *
 * MUST: **特定の変種キーの有無ではなく族名の接頭辞**で見る。変種キーで見ると、門が閉じている
 * はずの形が**別の変種で**走ったときに「期待したキーが無い」= 門が閉じたと読めてしまう。
 */
const ranGemv = (keys: readonly string[]): boolean =>
  keys.some((key) => key.startsWith("linear_gemv:"));

/**
 * 門が開いた形で期待する族のキー。M=1 は M=1 変種（decode の生成物を動かさない）、
 * M ≥ 2 は行ブロック変種で、`rows` は (格納, m, n) の純関数から来る。
 */
const expectedGemvKey = (
  storage: WeightStorage,
  m: number,
  n: number,
  groupSize?: number,
): string =>
  m === 1
    ? linearGemvKey(storage, groupSize)
    : linearGemvRowsKey(storage, groupSize, defaultLinearGemvRowsVariant(storage, m, n));

/**
 * MUST: 門の**各条件を 1 つずつだけ外した形**を並べる。まとめて外すと、どの条件が門を
 * 閉じているのか（あるいは条件が 1 つも効いていないのか）が区別できない。
 * MUST: 行数以外の条件（group 長 / v4）は **M=1 と M ≥ 2 の両方で 1 つずつ外す**。行ブロックは
 * 別の生成枝なので、M=1 で門が閉じることは M ≥ 2 で閉じることの根拠にならない。
 */
const DOOR_CASES: readonly DoorCase[] = [
  {
    name: "M=1 × i4 × g32 × v4",
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: 1,
    gemv: true,
  },
  {
    // 行数の条件を満たす下端（M ≥ 2 = 行ブロック変種 — perf-ledger K-21）
    name: "M=2 × i4 × g32 × v4（行ブロックの下端）",
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: 2,
    gemv: true,
  },
  {
    // 上限ちょうど（門の内側の上端）
    name: `M=${LINEAR_GEMV_MAX_ROWS} × i4 × g32 × v4（行ブロックの上端）`,
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: LINEAR_GEMV_MAX_ROWS,
    gemv: true,
  },
  {
    // 行数だけを外す（上限 + 1 — その外は既定の GEMM 骨格のまま）
    name: `M=${GEMM_ROWS}（行数の条件だけ外す）`,
    shape: { name: "", k: 128, n: 64, groupSize: 32 },
    m: GEMM_ROWS,
    gemv: false,
    fallback: linearKey("i4", true, "f32", GEMM_ROWS, 32),
  },
  {
    // group 長だけを外す（16 = 1 語 32 要素が group を跨ぐ形）
    name: "group 16 × M=1（group 長の条件だけ外す）",
    shape: { name: "", k: 128, n: 64, groupSize: 16 },
    m: 1,
    gemv: false,
    fallback: linearKey("i4", true, "f32", 1, 16),
  },
  {
    // 同じ条件を行ブロック側でも外す
    name: "group 16 × M=2（group 長の条件だけ外す）",
    shape: { name: "", k: 128, n: 64, groupSize: 16 },
    m: 2,
    gemv: false,
    fallback: linearKey("i4", true, "f32", 2, 16),
  },
  {
    // v4 だけを外す（n % 4 != 0 — 既定のスカラ変種へ）
    name: "n=33 × M=1（v4 の条件だけ外す）",
    shape: { name: "", k: 128, n: 33, groupSize: 32 },
    m: 1,
    gemv: false,
    fallback: linearKey("i4", false, "f32", 1, 32),
  },
  {
    // 同じ条件を行ブロック側でも外す
    name: "n=33 × M=2（v4 の条件だけ外す）",
    shape: { name: "", k: 128, n: 33, groupSize: 32 },
    m: 2,
    gemv: false,
    fallback: linearKey("i4", false, "f32", 2, 32),
  },
];

Deno.test({
  name: `GEMV 族の門は 1 ≤ M ≤ ${LINEAR_GEMV_MAX_ROWS} × i4 × group32 以上 × v4 でだけ開く` +
    "（実 GPU / 診断キー）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const door of DOOR_CASES) {
        const { k, n, groupSize } = door.shape;
        const weight = fill([n, k], weightAt(k, groupSize));
        const bias = fill([n], BS);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);
        const { keys } = await runLinear(
          gpu,
          linearI4Model(door.shape, door.m, quantized, bias),
          door.m,
          k,
        );
        // MUST: 列挙が無い device では診断が空になる（キー検査は数値側の門に任せて素通り）。
        if (keys.length === 0) continue;
        const shown = keys.join(" / ");
        assertEquals(
          ranGemv(keys),
          door.gemv,
          `${door.name}: GEMV 族が走ったかどうかが期待と違う（走った内訳: ${shown}）`,
        );
        if (door.gemv) {
          // 開いた形は**どの変種で**開いたかまで見る（M=1 変種と行ブロック変種の取り違えは
          // 族名だけでは見えない）。
          const key = expectedGemvKey("i4", door.m, n, groupSize);
          assert(
            keys.includes(key),
            `${door.name}: 期待した変種のキー ${key} で走っていない（内訳: ${shown}）`,
          );
        } else {
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

// ---------------------------------------------------------------------------
// i8 格納（lm_head — perf-ledger K-16）。i4 と同じ 3 観点を、別枝の生成（1 語 16 要素・
// scale は出力チャネルごと 1 本）に対して独立に踏む。
// ---------------------------------------------------------------------------

type GemvI8Case = {
  readonly name: string;
  readonly k: number;
  readonly n: number;
};

/**
 * 出力チャネルごとに振幅（= scale）が違い、隣接要素の符号が交互になる重み。
 * MUST: チャネルで振幅を変える — 全チャネル同じ scale だと `wscale[col]` の添字取り違えが
 * 一切値に出ない（i4 の group ごと MUST と同型の罠）。
 */
const weightAtI8 = (k: number) => (index: number): number => {
  const row = Math.floor(index / k);
  const base = (0.125 + (index % 11) * 0.5) * (index % 2 === 0 ? 1 : -1);
  return base * (1 + (row % 7) * 0.5);
};

/**
 * 形の選定。`units = k / 16`（重み語の本数）と `n % 32`（最終 workgroup の埋まり方）が
 * 独立の軸で、本番形（n 262,144 × k 1,536 = units 96・n % 32 == 0）はどの端も踏まない。
 * MUST: `n % 4 == 0`（門が v4 を要求する — recipe-builders/linear.ts の `buildLinear`）。
 */
const I8_CASES: readonly GemvI8Case[] = [
  // 端がどこにも無い基準形（units = 4 ちょうど・n は workgroup 2 枚ちょうど）
  { name: "整除形 k64 n64", k: 64, n: 64 },
  // units = 5 → 先読み 4 本の後に端数 1 本 / n = 100 は最終 workgroup が 4 列だけ
  { name: "端数 units5 n100", k: 80, n: 100 },
  // units = 6（端数 2 本）・n = 36 も最終 workgroup が部分的
  { name: "端数 units6 n36", k: 96, n: 36 },
  // units = 8（整除）・n = 68 は workgroup 2 枚 + 4 列
  { name: "整除形 units8 n68", k: 128, n: 68 },
  // units = 3 < 先読み 4 → 先読みループが一度も回らない（端数ループだけで縮約が終わる）
  { name: "先読み不成立 units3 n4", k: 48, n: 4 },
  // units = 7（端数 3 本）・n = 32 は workgroup ちょうど 1 枚
  { name: "端数 units7 n32", k: 112, n: 32 },
];

/** `linear(x, w, b)` 1 本のグラフ（w は i8 + 出力チャネルごとの scale）。`m` だけが経路を分ける。 */
const linearI8Model = (
  testCase: GemvI8Case,
  m: number,
  quantized: ReturnType<typeof quantizeI8>,
  bias: FilledTensor,
): ArrayBuffer => {
  const { k, n } = testCase;
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
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
      { name: "m.w", dtype: "I8", shape: [n, k], data: quantized.bytes },
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

Deno.test({
  name: "M=1 の i8 linear は GEMV 族で走っても既定経路と 1 ビットも違わない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const testCase of I8_CASES) {
        const { name, k, n } = testCase;
        const weight = fill([n, k], weightAtI8(k));
        const bias = fill([n], BS);
        // チャネル軸は 0（linear の重みは [n,k] — ADR 0019 / 0024 の MUST ④）。
        const quantized = quantizeI8(weight.data, weight.shape, 0);

        const gemv = await runLinear(gpu, linearI8Model(testCase, 1, quantized, bias), 1, k);
        // 比較相手は M = 上限 + 1（門の外 = 既定の GEMM 骨格 — i4 側と同文）。
        const gemm = await runLinear(
          gpu,
          linearI8Model(testCase, GEMM_ROWS, quantized, bias),
          GEMM_ROWS,
          k,
        );
        assertRanGemm(name, gemm.keys);

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

/** 門（i8 側）1 件ぶんの期待キー。i8 は group を持たないので軸は M / k / n の 3 本。 */
type DoorI8Case = {
  readonly name: string;
  readonly shape: GemvI8Case;
  readonly m: number;
  readonly gemv: boolean;
  readonly fallback?: string;
};

/**
 * MUST: 門の**各条件を 1 つずつだけ外した形**を並べる（i4 側と同じ規律）。
 * `k % 16 != 0` は i8 固有の軸で、外すと 1 語 16 要素の縮約が行の末尾を黙って落とす。
 * MUST: 行数以外の条件は M=1 と M ≥ 2 の両方で外す（i4 側と同文 — 生成枝が別）。
 */
const DOOR_I8_CASES: readonly DoorI8Case[] = [
  {
    name: "M=1 × i8 × k%16 × v4",
    shape: { name: "", k: 64, n: 64 },
    m: 1,
    gemv: true,
  },
  {
    // 行数の条件を満たす下端（M ≥ 2 = 行ブロック変種）
    name: "M=2 × i8 × k%16 × v4（行ブロックの下端）",
    shape: { name: "", k: 64, n: 64 },
    m: 2,
    gemv: true,
  },
  {
    // 上限ちょうど（門の内側の上端）
    name: `M=${LINEAR_GEMV_MAX_ROWS} × i8 × k%16 × v4（行ブロックの上端）`,
    shape: { name: "", k: 64, n: 64 },
    m: LINEAR_GEMV_MAX_ROWS,
    gemv: true,
  },
  {
    // 行数だけを外す（上限 + 1）
    name: `M=${GEMM_ROWS}（行数の条件だけ外す）`,
    shape: { name: "", k: 64, n: 64 },
    m: GEMM_ROWS,
    gemv: false,
    fallback: linearKey("i8", true, "f32", GEMM_ROWS),
  },
  {
    // 縮約の刻みだけを外す（k=68 は 4 の倍数なので v4 は立ったまま）
    name: "k=68 × M=1（k % 16 の条件だけ外す）",
    shape: { name: "", k: 68, n: 64 },
    m: 1,
    gemv: false,
    fallback: linearKey("i8", true, "f32", 1),
  },
  {
    // 同じ条件を行ブロック側でも外す
    name: "k=68 × M=2（k % 16 の条件だけ外す）",
    shape: { name: "", k: 68, n: 64 },
    m: 2,
    gemv: false,
    fallback: linearKey("i8", true, "f32", 2),
  },
  {
    // v4 だけを外す（n % 4 != 0 — 既定のスカラ変種へ）
    name: "n=33 × M=1（v4 の条件だけ外す）",
    shape: { name: "", k: 64, n: 33 },
    m: 1,
    gemv: false,
    fallback: linearKey("i8", false, "f32", 1),
  },
  {
    // 同じ条件を行ブロック側でも外す
    name: "n=33 × M=2（v4 の条件だけ外す）",
    shape: { name: "", k: 64, n: 33 },
    m: 2,
    gemv: false,
    fallback: linearKey("i8", false, "f32", 2),
  },
];

Deno.test({
  name: `GEMV 族の門は 1 ≤ M ≤ ${LINEAR_GEMV_MAX_ROWS} × i8 × k%16 × v4 でだけ開く` +
    "（実 GPU / 診断キー）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const door of DOOR_I8_CASES) {
        const { k, n } = door.shape;
        const weight = fill([n, k], weightAtI8(k));
        const bias = fill([n], BS);
        const quantized = quantizeI8(weight.data, weight.shape, 0);
        const { keys } = await runLinear(
          gpu,
          linearI8Model(door.shape, door.m, quantized, bias),
          door.m,
          k,
        );
        // MUST: 列挙が無い device では診断が空になる（キー検査は数値側の門に任せて素通り）。
        if (keys.length === 0) continue;
        const shown = keys.join(" / ");
        assertEquals(
          ranGemv(keys),
          door.gemv,
          `${door.name}: GEMV 族が走ったかどうかが期待と違う（走った内訳: ${shown}）`,
        );
        if (door.gemv) {
          const key = expectedGemvKey("i8", door.m, n);
          assert(
            keys.includes(key),
            `${door.name}: 期待した変種のキー ${key} で走っていない（内訳: ${shown}）`,
          );
        } else {
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

// ---------------------------------------------------------------------------
// 行ブロック変種（M ≥ 2 — ADR 0082 追記 5 / perf-ledger K-21）。M=1 変種と同じ 3 観点を、
// **行ブロックが足す 4 つの断片**（rows > 1 の行共有 / 複数 y タイル / 部分的な最終 y タイル /
// rows == 1 の y タイル形）へ差し向ける。格納 2 種は別枝なので独立に踏む。
// ---------------------------------------------------------------------------

/**
 * 行ブロック 1 件の形。`rows` / `tiles` / `lastTileRows` は **`linearGemvRowsForShape` の実装表から
 * 導いた値**で、ケースがどの断片を踏むつもりなのかを書き留めたもの。
 *
 * MUST: 3 つとも表明する。形だけ書いて意図を書かないと、選択が変わったときにケースが黙って
 * 「rows 1 の y タイルばかり」へ滑り、行共有も部分タイルも 1 度も走らないまま緑になる。
 */
type RowsCase = {
  readonly name: string;
  readonly k: number;
  readonly n: number;
  /** i4 のみ（i8 は group を持たない）。 */
  readonly groupSize?: number;
  readonly m: number;
  /** 期待する行ブロックの高さ（1 スレッドが持つ行数）。 */
  readonly rows: number;
  /** 期待する y タイル枚数 = `ceil(m / rows)`（重みを読み直す回数）。 */
  readonly tiles: number;
  /** 最終 y タイルで実際に書き戻す行数（`< rows` なら部分タイル）。 */
  readonly lastTileRows: number;
  /** 長いKは短縮約用の許容差でなく、独立CPU参照からの前進誤差上界を検査する。 */
  readonly longReference?: boolean;
};

/**
 * i4 の形。`units = k / 32`（先読み 4 本の端数）と `n % 32`（最終 workgroup の埋まり方）は
 * M=1 側と同じ軸で、ここに rows / y タイルの軸が乗る。
 * MUST: `n % 4 == 0`（門が v4 を要求する）。
 */
const ROWS_CASES: readonly RowsCase[] = [
  // 関数の呼び出しをまたぐ長いKと部分タイルでも、通常GEMMとの丸めを維持する。
  {
    name: "長いK・rows2 × 部分タイル（k4096 n4096 m9）",
    longReference: true,
    k: 4096,
    n: 4096,
    groupSize: 64,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // rows > 1 × y タイル複数（行共有と読み直しが両方走る基準形・units 4 ちょうど）
  {
    name: "rows4 × タイル8（k128 n2048 g32 m32）",
    k: 128,
    n: 2048,
    groupSize: 32,
    m: 32,
    rows: 4,
    tiles: 8,
    lastTileRows: 4,
  },
  // 天井の rows（i4 は 1 語 32 要素 × 8 行 = 256 で頭打ち — コンパイル費の天井）・units 3 < 先読み 4
  {
    name: "天井rows8 × タイル4 × 先読み不成立（k96 n16384 g32 m32）",
    k: 96,
    n: 16384,
    groupSize: 32,
    m: 32,
    rows: 8,
    tiles: 4,
    lastTileRows: 8,
  },
  // 最終 y タイルが部分的（9 = 2×4 + 1 → 最後の 1 行だけ書き戻す）・n % 32 = 4 で端の
  // workgroup も部分的・units 5（端数 1 本）
  {
    name: "部分タイル rows2 × 最終1行 × n%32≠0（k160 n4100 g32 m9）",
    k: 160,
    n: 4100,
    groupSize: 32,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // 部分タイル（7 = 4 + 3）× group 64（scale の shift が恒等式に縮まない）・units 6（端数 2 本）
  {
    name: "部分タイル rows4 × 最終3行 × g64（k192 n16384 g64 m7）",
    k: 192,
    n: 16384,
    groupSize: 64,
    m: 7,
    rows: 4,
    tiles: 2,
    lastTileRows: 3,
  },
  // rows 1 の y タイル形（n が小さいと上限近い M でもこれが選ばれる）— M=1 変種と別テキスト
  {
    name: "rows1 × タイル33（k128 n64 g32 m33）",
    k: 128,
    n: 64,
    groupSize: 32,
    m: 33,
    rows: 1,
    tiles: 33,
    lastTileRows: 1,
  },
  // 門の上端ちょうど（M = LINEAR_GEMV_MAX_ROWS）・units 7（端数 3 本）
  {
    name: "上限 m64 × 天井rows8 × タイル8（k224 n8192 g32）",
    k: 224,
    n: 8192,
    groupSize: 32,
    m: LINEAR_GEMV_MAX_ROWS,
    rows: 8,
    tiles: 8,
    lastTileRows: 8,
  },
];

/**
 * i8 の形。`units = k / 16` で、天井は 1 語 16 要素 × 16 行 = 256（i4 の倍の行数まで伸びる）。
 * MUST: `n % 4 == 0`（門が v4 を要求する）。
 */
const ROWS_I8_CASES: readonly RowsCase[] = [
  // 関数の呼び出しをまたぐ長いKと部分タイルでも、通常GEMMとの丸めを維持する。
  {
    name: "長いK・rows2 × 部分タイル（k4096 n4096 m9）",
    longReference: true,
    k: 4096,
    n: 4096,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // rows > 1 × y タイル複数・units 5（端数 1 本）
  {
    name: "rows8 × タイル4（k80 n4096 m32）",
    k: 80,
    n: 4096,
    m: 32,
    rows: 8,
    tiles: 4,
    lastTileRows: 8,
  },
  // 天井の rows 16（i4 の 8 と別値であることがこのケースの主眼）・y タイル 2 枚・units 3
  {
    name: "天井rows16 × タイル2 × 先読み不成立（k48 n16384 m32）",
    k: 48,
    n: 16384,
    m: 32,
    rows: 16,
    tiles: 2,
    lastTileRows: 16,
  },
  // 最終 y タイルが部分的（9 = 2×4 + 1）・n % 32 = 4・units 7（端数 3 本）
  {
    name: "部分タイル rows2 × 最終1行 × n%32≠0（k112 n4100 m9）",
    k: 112,
    n: 4100,
    m: 9,
    rows: 2,
    tiles: 5,
    lastTileRows: 1,
  },
  // rows 1 の y タイル形
  {
    name: "rows1 × タイル33（k64 n64 m33）",
    k: 64,
    n: 64,
    m: 33,
    rows: 1,
    tiles: 33,
    lastTileRows: 1,
  },
  // 門の上端ちょうど × 天井 rows 16（y タイル 4 枚）
  {
    name: "上限 m64 × 天井rows16 × タイル4（k112 n8192）",
    k: 112,
    n: 8192,
    m: LINEAR_GEMV_MAX_ROWS,
    rows: 16,
    tiles: 4,
    lastTileRows: 16,
  },
];

/**
 * 行ブロック 1 ケースの検査（格納 2 種で共有）。
 *
 * ① 形の意図（rows / y タイル枚数 / 部分タイルの行数）が実装表と一致する
 * ② 既定 GEMM（M = {@link GEMM_ROWS}）の**先頭 m 行**と u32 完全一致（全要素）
 * ③ CPU 参照と allclose（両経路が同じだけ壊れている場合の排除）
 * ④ 期待した行ブロック変種のキーで実際に走った（診断が取れる device のみ — ②③が
 *    既定経路のままでも緑になる形＝恒真化の排除）
 */
const checkRowsCase = async (
  gpu: GpuContext,
  storage: WeightStorage,
  rowsCase: RowsCase,
  build: (m: number) => ArrayBuffer,
  weightShape: readonly number[],
  values: Float32Array<ArrayBuffer>,
  bias: FilledTensor,
): Promise<void> => {
  const { name, k, n, m, groupSize } = rowsCase;
  // ① 形の意図（実装表が動いたらケースの狙いごと赤くする）
  assertEquals(linearGemvRowsForShape(storage, m, n), rowsCase.rows, `${name}: 行ブロックの高さ`);
  const tiles = Math.ceil(m / rowsCase.rows);
  assertEquals(tiles, rowsCase.tiles, `${name}: y タイル枚数`);
  assertEquals(
    m - (tiles - 1) * rowsCase.rows,
    rowsCase.lastTileRows,
    `${name}: 最終 y タイルの有効行数`,
  );

  const gemv = await runLinear(gpu, build(m), m, k);
  const gemm = await runLinear(gpu, build(GEMM_ROWS), GEMM_ROWS, k);
  assertRanGemm(name, gemm.keys);

  // ② 全要素の u32 完全一致（GEMM 側は先頭 m 行 = 同じ平坦添字の区間）
  assertEquals(gemv.output.shape, [m, n], `${name}: 出力の形`);
  const actual = bits(gemv.output);
  const expected = bits(gemm.output);
  for (let index = 0; index < m * n; index += 1) {
    if (actual[index] === expected[index]) continue;
    assert(
      false,
      `${name}: 行 ${Math.floor(index / n)} 列 ${index % n} が既定経路と別ビット` +
        `（0x${actual[index].toString(16)} vs 0x${expected[index].toString(16)} = ` +
        `${gemv.output.data[index]} vs ${gemm.output.data[index]}）`,
    );
  }

  // ③ CPU参照。短いKの既存許容差はそのまま使う。
  const reference = applyReferenceOp(
    "linear",
    [fill([m, k], XS) as RefTensor, refTensor(weightShape, values), bias as RefTensor],
    {},
    [m, n],
  );
  if (rowsCase.longReference) {
    // K=4096へK=72の経験的な帯を外挿しない。f32の隣接値への丸めを含むu=2^-23、
    // 乗算K回・加算K回・bias・参照の最終丸めからγ(2K+2)·Σ|項|を上界とする。
    // この入力は有限かつnormal。FMAの有無を固定しない。最適化の退行は②の全ビット比較で検出する。
    const input = fill([m, k], XS).data;
    const errorFactor = (2 * k + 2) * 2 ** -23;
    const gamma = errorFactor / (1 - errorFactor);
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        let magnitude = Math.abs(bias.data[col]);
        for (let inner = 0; inner < k; inner++) {
          magnitude += Math.abs(input[row * k + inner] * values[col * k + inner]);
        }
        const index = row * n + col;
        const error = Math.abs(gemv.output.data[index] - reference.data[index]);
        assert(
          error <= gamma * magnitude,
          `${name}: CPU参照 ${index}: ${error} > ${gamma * magnitude}`,
        );
      }
    }
  } else {
    const report = compareTensors(gemv.output, reference, GEMM_TOLERANCE);
    assertEquals(report.pass, true, `${name}: ${formatAllclose(report)}`);
  }

  // ④ 行ブロック変種のキーで走ったこと（②③だけだと既定経路のままでも緑になる）
  if (gemv.keys.length === 0) return;
  const key = linearGemvRowsKey(storage, groupSize, defaultLinearGemvRowsVariant(storage, m, n));
  assert(
    gemv.keys.includes(key),
    `${name}: 行ブロックのキー ${key} で走っていない（内訳: ${gemv.keys.join(" / ")}）`,
  );
};

Deno.test({
  name: "行ブロック（M ≥ 2）の i4 linear は既定経路と 1 ビットも違わない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const rowsCase of ROWS_CASES) {
        const { k, n } = rowsCase;
        const groupSize = rowsCase.groupSize ?? 32;
        const weight = fill([n, k], weightAt(k, groupSize));
        const bias = fill([n], BS);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);
        const shape: GemvCase = { name: rowsCase.name, k, n, groupSize };
        await checkRowsCase(
          gpu,
          "i4",
          rowsCase,
          (m) => linearI4Model(shape, m, quantized, bias),
          weight.shape,
          quantized.values,
          bias,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "行ブロック（M ≥ 2）の i8 linear は既定経路と 1 ビットも違わない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const rowsCase of ROWS_I8_CASES) {
        const { k, n } = rowsCase;
        const weight = fill([n, k], weightAtI8(k));
        const bias = fill([n], BS);
        // チャネル軸は 0（linear の重みは [n,k] — ADR 0019 / 0024 の MUST ④）。
        const quantized = quantizeI8(weight.data, weight.shape, 0);
        const shape: GemvI8Case = { name: rowsCase.name, k, n };
        await checkRowsCase(
          gpu,
          "i8",
          rowsCase,
          (m) => linearI8Model(shape, m, quantized, bias),
          weight.shape,
          quantized.values,
          bias,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});
