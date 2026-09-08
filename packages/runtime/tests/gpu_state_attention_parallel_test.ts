// states 形 attention の**並列縮約変種 2 段**（席は 1 つ — `stateAttentionReduce: "parallel"`）の
// A/B 帯門（ADR 0058 決定 4 ②）:
//   ③' = ③PV の KV 並列縮約（perf-ledger K-12）
//   ①' = ①QK の D 並列縮約（perf-ledger K-14）
// 参照経路（① ③ の逐次）の門は gpu_state_attention_test.ts のまま 1 文字も動かさない
// （同決定 4 ①の凍結）。
//
// どちらも縮約順が違うので参照経路とビット同一ではない。ここで見るのは
//   ① f64 参照との allclose（参照経路と同じ許容差 — 順序が変わっても正しさの帯は同じ）
//   ② 参照経路との A/B 帯（実測ドリフト × マージン — 下の `AB_TOLERANCE` /
//      `QK_AB_TOLERANCE` の導出表）
//   ③ 構造的保証は参照経路と同じ: pad 行（空行を包含）は**厳密 0**・①' は述語外の S が
//      **厳密 −inf のビット列**で `[live, col_cap)` と pad 行を 1 語も書かない・容量 C を
//      変えてもビット同一・行ブロックを割ってもビット同一・同一入力の 2 回 dispatch は
//      ビット同一（決定性）
//   ④ 故障注入: 木縮約を潰す / レーンの走査幅を変える変異が ① で落ちる（門が空振りでない証明）
//
// census 門（同決定 4 ③ — 席を指定したとき ①' と ③' が実際に走る）は
// gpu_state_execution_test.ts。

import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import {
  referenceStateAttention,
  type StateAttentionRefInput,
} from "../src/reference/state-attention.ts";
import { stateColumnBase, stateLiveColumns, stateSliding } from "../src/kernels/state-attention.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  assertMutated,
  caseColCap,
  halfScale,
  runStateAttention,
  seeded,
  STATE_S_POISON,
  type StateCase,
  type StateInputs,
  type StatePipelineCache,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** f64 参照との許容差（gpu_state_attention_test.ts の `STATE_TOLERANCE` と同値・同根拠）。 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/**
 * ③ と ③' の A/B 帯。
 *
 * 導出: 両者とも f32 で `live` 本の積を足すが順序が違う（③ = 昇順逐次・③' = 16 レーン部分和 →
 * 固定順の木）。差の上界は f32 の縮約誤差 2 本ぶんで、出力の大きさ O(1)。実測最悪は下の
 * `PARALLEL_CASES` 全件（縮約長 `live` は 1 〜 16,384）で
 * **vs ③ maxAbs 2.38e-7 / vs f64 参照 3.99e-7**（2026-09-03 / RTX 3080 Ti・Vulkan）。
 * `atol = 5e-6` はそこへ約 20 倍の余裕（f64 参照との帯と同じ値 — 両者が f64 参照と同じ帯に
 * 居るなら互いの差もその帯の 2 倍以内）。`rtol = 0` の理由は STATE_TOLERANCE と同じ。
 * NOTE: この最悪値を作るのは live が**小さい**側のケースで、live 4,096 / 16,384 の
 * 射程宣言ケースを足しても最悪値は動かない（重みが多数の列へ散るぶん出力の変動が小さくなる）。
 * メイン実測（レビュー 2026-09-03）では live 65,536 まで同傾向 — ③' vs f64 1.8e-8 /
 * ③ vs f64 3.6e-8 / ③' vs ③ 5.0e-8 で、いずれも帯 5e-6 の 1/100 に収まる。
 * NOTE: 実測値はドライバの fma 使用に依るので、別アダプタでは「実測 → 余裕を積む」の手順ごと
 * 繰り返す（テストは最悪値を毎回 stdout に出す）。
 */
const AB_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/**
 * ① と ①' の A/B 帯（出力 O で見る — S の差は softmax を通ってから O に出る）。
 *
 * 導出は 2 段:
 * 1. **S の差**: 両者とも同じ `D` 本の積（半スケールを q 側と k 側の両方に載せた積 —
 *    生成は src/kernels/state-attention.ts の `stateScoreFn` 1 箇所）を f32 で足すが順序が
 *    違う（① = `d` 昇順の逐次・
 *    ①' = 16 レーン部分和 → 固定順の木）。f32 の縮約誤差の上界は 2 通りとも
 *    `O(D · eps · Σ|項|)`（eps = 2⁻²³ ≈ 1.19e-7）で、差はその 2 本ぶん。本ファイルの格子は
 *    `D ≤ 32`・項の大きさ ≤ 約 2.6（`|q| ≤ 1.87 · |k| ≤ 2.07 · scale²` で scale² = 1/√D）
 *    なので、`|S|` は O(10) 以下・S の差の上界は **約 1e-5**。
 * 2. **O への伝播**: `p = exp(S − m)·inv` なので S の差 δ は重み比を `exp(δ) ≈ 1 + δ` 倍に
 *    しか動かさず、正規化（Σp = 1）で共通因子は打ち消える。したがって O の差は
 *    `δ · max|V|` の程度に収まる（`|V| ≤ 2.48`）。
 *
 * 実測最悪は下の `PARALLEL_CASES` 全件で **vs ① maxAbs 3.58e-7 / vs f64 参照 4.17e-7**
 * （S そのものの差は 4.77e-7・席そのままの組〈①' + ③'〉vs f64 参照は 3.58e-7 —
 * 2026-09-06 / RTX 3080 Ti・Vulkan。テストが毎回 stdout に出す）。`atol = 5e-6` は ③' の
 * A/B 帯と同値で、実測へ約 14 倍・上の上界にも余裕を持つ（両者が f64 参照の帯 5e-6 に居るなら
 * 互いの差もその 2 倍以内、という同じ論法）。
 * `rtol = 0` の理由は STATE_TOLERANCE と同じ。
 * NOTE: 実測値はドライバの fma 使用に依るので、別アダプタでは「実測 → 余裕を積む」の手順ごと
 * 繰り返す。
 */
const QK_AB_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;
const SLOT_POISON_K = 9;
const SLOT_POISON_V = 400;

/** gpu_state_attention_test.ts の `makeInputs` と同じ毒値規約（非 resident 行・pad 行に毒）。 */
const makeInputs = (spec: StateCase): StateInputs => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = spec;
  const kvPlanes = batch * kvHeads;
  const q = seeded(batch * heads * chunkRows * depth, QUERY);
  const insK = seeded(kvPlanes * chunkRows * depth, KEY);
  const insV = seeded(kvPlanes * chunkRows * depth, VALUE);
  const slotK = seeded(kvPlanes * capacity * depth, (i) => KEY(i + 3));
  const slotV = seeded(kvPlanes * capacity * depth, (i) => VALUE(i + 5));
  const resident = new Set<number>();
  const base = stateColumnBase(window, past);
  for (let col = base; col < past; col += 1) {
    resident.add(stateSliding(window) ? col % capacity : col);
  }
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < capacity; row += 1) {
      if (resident.has(row)) continue;
      for (let d = 0; d < depth; d += 1) {
        slotK[(plane * capacity + row) * depth + d] = SLOT_POISON_K;
        slotV[(plane * capacity + row) * depth + d] = SLOT_POISON_V;
      }
    }
    for (let row = query; row < chunkRows; row += 1) {
      for (let d = 0; d < depth; d += 1) {
        insK[(plane * chunkRows + row) * depth + d] = SLOT_POISON_K;
        insV[(plane * chunkRows + row) * depth + d] = SLOT_POISON_V;
      }
    }
  }
  return { q, insK, insV, slotK, slotV };
};

const refInput = (spec: StateCase, inputs: StateInputs): StateAttentionRefInput => ({
  batch: spec.batch,
  heads: spec.heads,
  kvHeads: spec.kvHeads,
  chunkRows: spec.chunkRows,
  depth: spec.depth,
  capacity: spec.capacity,
  window: spec.window,
  past: spec.past,
  query: spec.query,
  scale: halfScale(spec.depth),
  ...inputs,
});

/**
 * ③' が ③ と違う経路を踏む形を優先した格子: live がレーン数 16 を跨ぐ（部分和が複数レーンに
 * 散る）・live が 16 の倍数ちょうど / 端数・D が TILE_X の端数（`d ≥ D` のレーンが barrier に
 * 参加する経路）・行ブロック複数・MQA 8:1 decode（Gemma 4 E2B 型）・sliding の ring 跨ぎ。
 */
const PARALLEL_CASES: readonly StateCase[] = [
  {
    name: "full r1 decode P0（live 1 = レーン 0 だけ）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 16,
    window: 0,
    past: 0,
    query: 1,
  },
  {
    name: "full r8 MQA decode live 18（16 を跨ぐ）",
    batch: 2,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 17,
    query: 1,
  },
  {
    name: "full r8 MQA decode live 48（16 の倍数ちょうど）D 20（端数）",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 20,
    capacity: 64,
    window: 0,
    past: 47,
    query: 1,
  },
  {
    name: "full r8 MQA decode live 301",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 32,
    capacity: 512,
    window: 0,
    past: 300,
    query: 1,
  },
  // 射程宣言（帯の根拠を実運用の live まで伸ばす — レビュー W-G1-1）。縮約長 `live` が
  // 4 桁 / 5 桁でも帯が桁で動かないことを門にする。形は Gemma 4 E2B の decode（MQA 8:1・D 32）。
  {
    name: "full r8 MQA decode live 4096",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 32,
    capacity: 4096,
    window: 0,
    past: 4095,
    query: 1,
  },
  {
    name: "full r8 MQA decode live 16384",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 32,
    capacity: 16384,
    window: 0,
    past: 16383,
    query: 1,
  },
  {
    name: "full r2 pad B2 D6（prefill・pad 行あり）",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    chunkRows: 6,
    depth: 6,
    capacity: 32,
    window: 0,
    past: 5,
    query: 3,
  },
  {
    name: "full r1 rowblock 4x3",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 9,
    depth: 8,
    capacity: 32,
    window: 0,
    past: 3,
    query: 9,
    rowsBlock: 4,
  },
  {
    name: "sliding W8 r1 ring wrap decode",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 8,
    window: 8,
    past: 21,
    query: 1,
  },
  {
    name: "sliding W32 r2 prefill pad（live 36 が 16 を跨ぐ）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 6,
    depth: 8,
    capacity: 64,
    window: 32,
    past: 40,
    query: 5,
  },
  {
    name: "sliding W512 r8 MQA decode（Gemma 4 E2B 型）",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 16,
    capacity: 512,
    window: 512,
    past: 700,
    query: 1,
  },
];

/** スロット行を容量 `capacity` の器へ写す（余った行は毒値 — 読まれたら値が跳ねる）。 */
const widenSlots = (spec: StateCase, inputs: StateInputs, capacity: number): StateInputs => {
  const kvPlanes = spec.batch * spec.kvHeads;
  const slotK = seeded(kvPlanes * capacity * spec.depth, () => SLOT_POISON_K);
  const slotV = seeded(kvPlanes * capacity * spec.depth, () => SLOT_POISON_V);
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < spec.capacity; row += 1) {
      for (let d = 0; d < spec.depth; d += 1) {
        const from = (plane * spec.capacity + row) * spec.depth + d;
        const to = (plane * capacity + row) * spec.depth + d;
        slotK[to] = inputs.slotK[from];
        slotV[to] = inputs.slotV[from];
      }
    }
  }
  return { ...inputs, slotK, slotV };
};

const padRows = (spec: StateCase): readonly number[] => {
  const rows: number[] = [];
  for (let row = spec.query; row < spec.chunkRows; row += 1) rows.push(row);
  return rows;
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const assertBitIdentical = (
  a: Float32Array<ArrayBuffer>,
  b: Float32Array<ArrayBuffer>,
  label: string,
): void => {
  assertEquals(a.length, b.length, `${label}: 長さ`);
  const ab = bitsOf(a);
  const bb = bitsOf(b);
  for (let i = 0; i < ab.length; i += 1) {
    if (ab[i] !== bb[i]) {
      throw new Error(`${label}: 要素 ${i} がビット不一致（${a[i]} vs ${b[i]}）`);
    }
  }
};

Deno.test({
  name: "states 形 ③' KV 並列縮約は f64 参照と ③ の両方の帯に収まり、構造的保証を保つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      let worstRef = 0;
      let worstAb = 0;
      for (const spec of PARALLEL_CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const parallel = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          pvReduce: "parallel",
        });
        const sequential = await runStateAttention(gpu.device, spec, inputs, { cache });

        // ① f64 参照との帯
        const ref = compareTensors({ dtype: "f32", data: parallel.out }, expected, STATE_TOLERANCE);
        assert(ref.pass, `${spec.name} vs 参照: ${formatAllclose(ref)}`);
        worstRef = Math.max(worstRef, ref.maxAbsError);
        // ② ③ との A/B 帯
        const ab = compareTensors({ dtype: "f32", data: parallel.out }, {
          dtype: "f32",
          data: sequential.out,
        }, AB_TOLERANCE);
        assert(ab.pass, `${spec.name} ③ vs ③': ${formatAllclose(ab)}`);
        worstAb = Math.max(worstAb, ab.maxAbsError);

        // ③ pad 行（空行を包含）は厳密 0（Object.is — −0 も NaN も通さない）
        const rowsOf = spec.batch * spec.heads;
        for (let z = 0; z < rowsOf; z += 1) {
          for (const row of padRows(spec)) {
            for (let d = 0; d < spec.depth; d += 1) {
              const at = (z * spec.chunkRows + row) * spec.depth + d;
              assert(
                Object.is(parallel.out[at], 0),
                `${spec.name}: pad 行 (z ${z}, row ${row}, d ${d}) が厳密 0 でない（${
                  parallel.out[at]
                }）`,
              );
            }
          }
        }

        // ③ 決定性（同一入力の 2 回 dispatch がビット同一）
        const again = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          pvReduce: "parallel",
        });
        assertBitIdentical(parallel.out, again.out, `${spec.name}: 決定性`);

        // ③ 容量非依存（C を大きくしても出力ビット同一 — 仕事量条件の裏）。スロットは**同じ行を
        // 広い容量の器へ写す**（seeded を引き直すと平面オフセットが動いて別の値になる）
        if (!stateSliding(spec.window)) {
          const wide: StateCase = { ...spec, capacity: spec.capacity * 8 };
          const wideInputs = widenSlots(spec, inputs, wide.capacity);
          const wideOut = await runStateAttention(gpu.device, wide, wideInputs, {
            cache,
            pvReduce: "parallel",
          });
          assertBitIdentical(parallel.out, wideOut.out, `${spec.name}: 容量 ×8`);
        }

        // ③ 行ブロック非依存（M > 1 の形は 1 枚 vs 複数枚がビット同一）
        if (spec.chunkRows > 1) {
          const split: StateCase = {
            ...spec,
            rowsBlock: Math.max(1, Math.ceil(spec.chunkRows / 3)),
          };
          const one: StateCase = { ...spec, rowsBlock: spec.chunkRows };
          const a = await runStateAttention(gpu.device, one, inputs, {
            cache,
            pvReduce: "parallel",
          });
          const b = await runStateAttention(gpu.device, split, inputs, {
            cache,
            pvReduce: "parallel",
          });
          assertBitIdentical(a.out, b.out, `${spec.name}: 行ブロック 1 枚 vs 3 枚`);
        }
      }
      console.log(
        `③' 実測最悪: vs 参照 maxAbs ${worstRef.toExponential(2)} / vs ③ maxAbs ${
          worstAb.toExponential(2)
        }（帯 ${STATE_TOLERANCE.atol} / ${AB_TOLERANCE.atol}）`,
      );
    } finally {
      gpu.destroy();
    }
  },
});

/** 故障注入 — 各変異が①の帯で落ちること（門が空振りでない証明）。 */
Deno.test({
  name: "states 形 ③' の故障注入（木縮約を潰す / レーン幅を変える）は参照との帯で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // live がレーン数を跨ぐ形でないと「レーン 0 の部分和だけ」でも一致してしまう
      const spec = PARALLEL_CASES.find((c) => c.name.includes("live 301"))!;
      const inputs = makeInputs(spec);
      const expected = referenceStateAttention(refInput(spec, inputs));
      const mutations: readonly [string, (wgsl: string) => string][] = [
        [
          "木縮約を潰す（stride ループを回さない）",
          (wgsl) => wgsl.replace("var stride = 8u;", "var stride = 0u;"),
        ],
        [
          "レーンの走査幅を 32 にする（列の半分を落とす）",
          (wgsl) => wgsl.replace("cl = cl + 16u", "cl = cl + 32u"),
        ],
      ];
      for (const [label, mutate] of mutations) {
        const result = await runStateAttention(gpu.device, spec, inputs, {
          pvReduce: "parallel",
          mutate: (kernel, wgsl) => {
            if (kernel !== "pv") return wgsl;
            const mutated = mutate(wgsl);
            assertMutated(wgsl, mutated, label);
            return mutated;
          },
        });
        const report = compareTensors(
          { dtype: "f32", data: result.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(
          report.pass,
          false,
          `${label}: 変異が帯を通ってしまった（${formatAllclose(report)}）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * ①' の S を ① と突き合わせる（**構造は厳密・値だけ帯**）。単一ブロックのケース専用
 * （局所行 = グローバル行・`runStateAttention` が返す S が 1 枚に閉じる）。
 *
 * 見るのは 4 つで、どれも tolerance に隠れない構造の側:
 *   ① 述語内は両者とも有限で毒値でない（書かれている） — 値の差は戻り値の最悪差で返す
 *   ② 述語外は両者とも **厳密 −inf のビット列**（同じ位置が同じビット）
 *   ③ `[live, col_cap)` の残骸は両者とも毒値のまま（①' の列添字は `wid.x·16 + lid.x` なので、
 *      端数タイルのレーンを止め損ねるとここが書かれる）
 *   ④ pad 行は 1 語も書かれない（仕事量が M でなく Q に比例することの直接の裏）
 */
type ScoreCensus = { written: number; masked: number; beyond: number; skipped: number };

const assertQkScores = (
  spec: StateCase,
  parallel: Float32Array<ArrayBuffer>,
  sequential: Float32Array<ArrayBuffer>,
  census: ScoreCensus,
): number => {
  const colCap = caseColCap(spec);
  const base = stateColumnBase(spec.window, spec.past);
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  const rows = spec.chunkRows;
  const effRows = Math.min(spec.query, rows);
  const bitsP = bitsOf(parallel);
  const bitsS = bitsOf(sequential);
  let worst = 0;
  for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
    for (let local = 0; local < effRows; local += 1) {
      const limit = spec.past + local;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        const inWindow = col <= limit &&
          (!stateSliding(spec.window) || limit - col < spec.window);
        const at = (plane * rows + local) * colCap + cl;
        if (inWindow) {
          assert(
            Number.isFinite(parallel[at]) && parallel[at] !== STATE_S_POISON,
            `${spec.name}: ①' の述語内 (${local},${cl}) が書かれていない（${parallel[at]}）`,
          );
          worst = Math.max(worst, Math.abs(parallel[at] - sequential[at]));
          census.written += 1;
        } else {
          assertEquals(
            [bitsP[at], bitsS[at]],
            [0xff800000, 0xff800000],
            `${spec.name}: 述語外 (${local},${cl}) が −inf のビット列でない（① と ①' の突合）`,
          );
          census.masked += 1;
        }
      }
      for (let cl = live; cl < colCap; cl += 1) {
        const at = (plane * rows + local) * colCap + cl;
        assertEquals(
          parallel[at],
          STATE_S_POISON,
          `${spec.name}: ①' が live の外 (${local},${cl}) を書いた（端数タイルのレーンが止まっていない）`,
        );
        census.beyond += 1;
      }
    }
    for (let local = effRows; local < rows; local += 1) {
      for (let cl = 0; cl < colCap; cl += 1) {
        const at = (plane * rows + local) * colCap + cl;
        assertEquals(
          parallel[at],
          STATE_S_POISON,
          `${spec.name}: ①' が pad 行 (${local},${cl}) の S を書いた（M 行を回している）`,
        );
        census.skipped += 1;
      }
    }
  }
  return worst;
};

/**
 * ①' の帯門。格子は {@link PARALLEL_CASES} をそのまま使う — `D` が 4 / 6 / 8（レーンが遊ぶ）・
 * 16（1 レーン 1 本ちょうど）・20（端数 = `d ≥ D` のレーンが barrier に参加する経路）・
 * 32（1 レーン 2 本）と、live が 16 の倍数 / 端数・pad 行あり・行ブロック複数・sliding の
 * ring 跨ぎを既に覆っているため（③' の格子と要求が一致する）。
 *
 * MUST: ③ は**参照経路に固定**して撃つ（席は 2 段を一緒に切り替えるが、ここで両方替えると
 * 差が ①' 由来か ③' 由来か分けられない）。席そのままの組（①' + ③'）は最後に f64 参照との
 * 帯だけ見る。
 */
Deno.test({
  name: "states 形 ①' D 並列縮約は f64 参照と ① の両方の帯に収まり、構造的保証を保つ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      let worstRef = 0;
      let worstAb = 0;
      let worstScore = 0;
      let worstSeat = 0;
      const census: ScoreCensus = { written: 0, masked: 0, beyond: 0, skipped: 0 };
      for (const spec of PARALLEL_CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const parallel = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          qkReduce: "parallel",
        });
        const sequential = await runStateAttention(gpu.device, spec, inputs, { cache });

        // ① f64 参照との帯
        const ref = compareTensors({ dtype: "f32", data: parallel.out }, expected, STATE_TOLERANCE);
        assert(ref.pass, `${spec.name} vs 参照: ${formatAllclose(ref)}`);
        worstRef = Math.max(worstRef, ref.maxAbsError);
        // ② ① との A/B 帯
        const ab = compareTensors({ dtype: "f32", data: parallel.out }, {
          dtype: "f32",
          data: sequential.out,
        }, QK_AB_TOLERANCE);
        assert(ab.pass, `${spec.name} ① vs ①': ${formatAllclose(ab)}`);
        worstAb = Math.max(worstAb, ab.maxAbsError);

        // ③ S の構造（述語外 −inf のビット一致・live の外と pad 行の非書き込み）。
        // 行ブロックを割るケースは S が最後の 1 枚しか戻らないので単一ブロックだけ見る
        if (spec.rowsBlock === undefined) {
          worstScore = Math.max(
            worstScore,
            assertQkScores(spec, parallel.scores, sequential.scores, census),
          );
        }

        // ④ 決定性（同一入力の 2 回 dispatch がビット同一）
        const again = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          qkReduce: "parallel",
        });
        assertBitIdentical(parallel.out, again.out, `${spec.name}: 決定性`);

        // ⑤ 容量非依存（C を大きくしても出力ビット同一 — 仕事量条件の裏）
        if (!stateSliding(spec.window)) {
          const wide: StateCase = { ...spec, capacity: spec.capacity * 8 };
          const wideInputs = widenSlots(spec, inputs, wide.capacity);
          const wideOut = await runStateAttention(gpu.device, wide, wideInputs, {
            cache,
            qkReduce: "parallel",
          });
          assertBitIdentical(parallel.out, wideOut.out, `${spec.name}: 容量 ×8`);
        }

        // ⑥ 行ブロック非依存（M > 1 の形は 1 枚 vs 複数枚がビット同一）
        if (spec.chunkRows > 1) {
          const split: StateCase = {
            ...spec,
            rowsBlock: Math.max(1, Math.ceil(spec.chunkRows / 3)),
          };
          const one: StateCase = { ...spec, rowsBlock: spec.chunkRows };
          const a = await runStateAttention(gpu.device, one, inputs, {
            cache,
            qkReduce: "parallel",
          });
          const b = await runStateAttention(gpu.device, split, inputs, {
            cache,
            qkReduce: "parallel",
          });
          assertBitIdentical(a.out, b.out, `${spec.name}: 行ブロック 1 枚 vs 3 枚`);
        }

        // ⑦ 席そのままの組（①' + ③'）も f64 参照の帯に居る
        const seat = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          qkReduce: "parallel",
          pvReduce: "parallel",
        });
        const seatRef = compareTensors(
          { dtype: "f32", data: seat.out },
          expected,
          STATE_TOLERANCE,
        );
        assert(seatRef.pass, `${spec.name} 席 parallel vs 参照: ${formatAllclose(seatRef)}`);
        worstSeat = Math.max(worstSeat, seatRef.maxAbsError);
      }
      // 門が空振りしていない（述語内・述語外・live の外・pad 行が全て実在する格子である）
      assert(census.written > 0 && census.masked > 0, `S の門が空振り: ${JSON.stringify(census)}`);
      assert(census.beyond > 0 && census.skipped > 0, `S の門が空振り: ${JSON.stringify(census)}`);
      console.log(
        `①' 実測最悪: vs 参照 maxAbs ${worstRef.toExponential(2)} / vs ① maxAbs ${
          worstAb.toExponential(2)
        } / S の差 ${worstScore.toExponential(2)} / 席 ①'+③' vs 参照 ${
          worstSeat.toExponential(2)
        }（帯 ${STATE_TOLERANCE.atol} / ${QK_AB_TOLERANCE.atol}）`,
      );
    } finally {
      gpu.destroy();
    }
  },
});

/** 故障注入 — 各変異が①の帯で落ちること（門が空振りでない証明）。 */
Deno.test({
  name: "states 形 ①' の故障注入（木縮約を潰す / レーン幅を変える）は参照との帯で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // D がレーン数を跨ぐ形でないと「レーン 0 の部分和だけ」でも一致してしまう（D 32 = 2 本／レーン）
      const spec = PARALLEL_CASES.find((c) => c.name.includes("live 301"))!;
      const inputs = makeInputs(spec);
      const expected = referenceStateAttention(refInput(spec, inputs));
      const mutations: readonly [string, (wgsl: string) => string][] = [
        [
          "木縮約を潰す（stride ループを回さない）",
          (wgsl) => wgsl.replace("var stride = 8u;", "var stride = 0u;"),
        ],
        [
          "レーンの走査幅を 32 にする（D の半分を落とす）",
          (wgsl) => wgsl.replace("d = d + 16u", "d = d + 32u"),
        ],
      ];
      for (const [label, mutate] of mutations) {
        const result = await runStateAttention(gpu.device, spec, inputs, {
          qkReduce: "parallel",
          mutate: (kernel, wgsl) => {
            if (kernel !== "qk") return wgsl;
            const mutated = mutate(wgsl);
            assertMutated(wgsl, mutated, label);
            return mutated;
          },
        });
        const report = compareTensors(
          { dtype: "f32", data: result.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(
          report.pass,
          false,
          `${label}: 変異が帯を通ってしまった（${formatAllclose(report)}）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});
