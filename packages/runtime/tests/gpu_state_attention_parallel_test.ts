// states 形 attention ③PV の **KV 並列縮約変種**（③' — `stateAttentionReduce: "parallel"`・
// perf-ledger K-12）の A/B 帯門（ADR 0058 決定 4 ②）。参照経路（③ 逐次）の門は
// gpu_state_attention_test.ts のまま 1 文字も動かさない（同決定 4 ①の凍結）。
//
// ③' は縮約順が違うので ③ とビット同一ではない。ここで見るのは
//   ① f64 参照との allclose（③ と同じ許容差 — 順序が変わっても正しさの帯は同じ）
//   ② ③ との A/B 帯（実測ドリフト × マージン — 下の `AB_TOLERANCE` の導出表）
//   ③ 構造的保証は ③ と同じ: pad 行（空行を包含）は**厳密 0**・容量 C を変えてもビット同一・
//      行ブロックを割ってもビット同一・同一入力の 2 回 dispatch はビット同一（決定性）
//   ④ 故障注入: 木縮約を潰す / レーンの走査幅を変える変異が ① で落ちる（門が空振りでない証明）
//
// census 門（同決定 4 ③ — 席を指定したとき ③' が実際に走る）は gpu_state_execution_test.ts。

import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import {
  referenceStateAttention,
  type StateAttentionRefInput,
} from "../src/reference/state-attention.ts";
import { stateColumnBase, stateSliding } from "../src/kernels/state-attention.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  assertMutated,
  halfScale,
  runStateAttention,
  seeded,
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
    resident.add(stateSliding(window) ? col % window : col);
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
