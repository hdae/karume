// states 形 ①QK の **K 行タイル共有変種 ①ₜ**（GEMM 骨格 — perf-ledger K-13）の門。
//
// ①ₜ の入場条件は「① と**ビット同一**」（縮約順が変わる ①' と違い、席に依らない既定経路に
// なるのはこの一点だけを根拠にする）。したがってここで見るのは帯ではなく **S バッファの
// 語（u32）単位の完全一致**で、値そのものの正しさ（f64 参照との突合）は ① 側の門
// （gpu_state_attention_test.ts）が持ち続ける — ①ₜ はそこへ「① と 1 ビットも違わない」を継ぐ。
//
// 語で比べる対象は 3 種類あって、どれも f32 の `===` では比べられない:
//   ① 述語外の **−inf**（`0xff800000`）
//   ② `[live, col_cap)` の**残骸**と pad 行の**非書き込み**（毒値 2¹⁰⁰ が残っていること）
//   ③ 述語内の実値
// 「書かれた語数」を spec から独立に数えて突き合わせるので、両経路が揃って何も書かない形は
// この門を通れない（門が空振りでないことの構造的な裏）。
//
// 格子: {full, sliding} × {r=1, r=2} × M ∈ {16, 40, 100, 768}（幾何バケット 3 種を跨ぐ）×
// {P=0（ins だけ）, P>0（2 源）, P≥W（ring wrap）} × {pad 行あり / なし} ×
// {行ブロック 1 枚 / 複数枚} × {live がタイル辺の倍数でない}。
//
// MUST: 故障注入で門が落ちることを同じファイルで示す（B ローダの半スケール除去・列基点の
// 2 源取り違え）。落ちない門は何も見ていない。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  STATE_NEG_INF_BITS,
  stateColumnBase,
  stateEffectiveRows,
  stateLiveColumns,
  stateSliding,
} from "../src/kernels/state-attention.ts";
import {
  assertMutated,
  caseColCap,
  runStateQk,
  seeded,
  STATE_S_POISON,
  type StateCase,
  type StateInputs,
  type StatePipelineCache,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 決定的な入力列（gpu_state_attention_test.ts と同じ生成規約 — 乱数は使わない）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** 読まれてはいけない場所に置く毒値（読まれたら値が跳ねる）。 */
const SLOT_POISON_K = 9;
const SLOT_POISON_V = 400;

/** S の毒値のビット列（2¹⁰⁰ は f32 で厳密なので語比較が成立する）。 */
const POISON_BITS = new Uint32Array(new Float32Array([STATE_S_POISON]).buffer)[0];

/**
 * ①③ が読む 5 本を組む（gpu_state_attention_test.ts の `makeInputs` と同じ毒値規約）。
 *
 * MUST: **非 resident なスロット物理行**と **ins の pad 行**に毒を置く。①ₜ は述語外の列でも
 * K を読む（読む行は live 範囲内なので契約上安全）ので、「範囲外まで読んだ」誤りはこの毒でしか
 * 値に出ない。
 */
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

/**
 * 格子。**幾何バケットの 3 段**（M ≤ 64 = M16N16 / 65..512 = M64N32 / それ以上 = 既定
 * M128N128）を跨ぎ、どの段でも `live` と有効行がタイル辺の倍数でない形を混ぜる。
 *
 * MUST: full は `P + Q ≤ C`（run 前検査と同じ条件）を満たす（破ると `col_cap` が live に
 * 足りず、① 側の値域門で落ちる）。
 */
const CASES: readonly StateCase[] = [
  // --- 最小バケット M16N16（tileM = tileN = 16）---
  {
    name: "full r1 M16 P0（ins だけ・live = タイル辺ちょうど）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 0,
    query: 16,
  },
  {
    name: "full r1 M16 P13（2 源・live 29 = 辺の倍数でない）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  {
    name: "full r2 M16 P13（GQA・2 源）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  {
    name: "full r2 M16 P13 B2（batch > 1）",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  {
    name: "full r1 M40 P0（有効行 40 = 辺 16 の倍数でない）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 128,
    window: 0,
    past: 0,
    query: 40,
  },
  {
    name: "full r2 M40 P37 Q17（pad 行あり・2 源）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 128,
    window: 0,
    past: 37,
    query: 17,
  },
  {
    name: "full r2 M40 P37 行ブロック 12（分割あり）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 128,
    window: 0,
    past: 37,
    query: 40,
    rowsBlock: 12,
  },
  {
    name: "full r1 M16 D24（D が K タイル 16 の倍数でない）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 24,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  // --- sliding（下限述語 + ring 写像）---
  {
    name: "sliding r1 M16 W8 P0（窓 < Q・past 無し）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 0,
    query: 16,
  },
  {
    name: "sliding r1 M16 W8 P20（ring wrap 跨ぎ・live = col_cap）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 20,
    query: 16,
  },
  {
    name: "sliding r2 M16 W8 P5（GQA・past < W−1）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 5,
    query: 16,
  },
  {
    name: "sliding r2 M40 W12 P30 Q23（pad 行 + ring wrap）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 12,
    window: 12,
    past: 30,
    query: 23,
  },
  // --- 中 M バケット M64N32（tileM 64 / tileN 32）---
  {
    name: "full r2 M100 P0（live 100 = 辺 32 の倍数でない）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 100,
    depth: 16,
    capacity: 256,
    window: 0,
    past: 0,
    query: 100,
  },
  {
    name: "full r2 M100 P37 Q60（pad 行 + 2 源）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 100,
    depth: 16,
    capacity: 256,
    window: 0,
    past: 37,
    query: 60,
  },
  {
    name: "sliding r1 M100 W20 P64 行ブロック 33（分割あり）",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 100,
    depth: 16,
    capacity: 20,
    window: 20,
    past: 64,
    query: 100,
    rowsBlock: 33,
  },
  // --- 既定バケット M128N128（prefill の実測形 M = 768）---
  {
    name: "full r1 M768 P0（既定幾何・行タイル 6 枚）",
    batch: 1,
    heads: 1,
    kvHeads: 1,
    chunkRows: 768,
    depth: 16,
    capacity: 800,
    window: 0,
    past: 0,
    query: 768,
  },
  {
    name: "sliding r2 M768 W64 P500 行ブロック 256（分割あり・2 源）",
    batch: 1,
    heads: 2,
    kvHeads: 1,
    chunkRows: 768,
    depth: 16,
    capacity: 64,
    window: 64,
    past: 500,
    query: 768,
    rowsBlock: 256,
  },
];

/** 1 ケースの行ブロックの割り（`runStateQk` の走査と同じ式）。 */
const blocksOf = (spec: StateCase): readonly { rows: number; offset: number }[] => {
  const rowsBlock = spec.rowsBlock ?? spec.chunkRows;
  const blocks: { rows: number; offset: number }[] = [];
  for (let offset = 0; offset < spec.chunkRows; offset += rowsBlock) {
    blocks.push({ rows: Math.min(rowsBlock, spec.chunkRows - offset), offset });
  }
  return blocks;
};

/**
 * S の語を**書かれた / 残骸**の 2 群に分けて数える（門が空振りでないことの構造的な裏）。
 *
 * 書かれる語数は spec だけから決まる（`B·H × 有効行 × live`）ので、実装を一切参照せずに
 * 期待値を出せる。残りは毒値のままでなければならない。
 */
const assertWriteCoverage = (
  spec: StateCase,
  words: Uint32Array<ArrayBuffer>,
  block: { rows: number; offset: number },
  label: string,
): void => {
  const batchHeads = spec.batch * spec.heads;
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  const rows = stateEffectiveRows(block.rows, block.offset, spec.query);
  let poison = 0;
  for (let index = 0; index < words.length; index += 1) {
    if (words[index] === POISON_BITS) poison += 1;
  }
  assertEquals(
    words.length - poison,
    batchHeads * rows * live,
    `${label}: 書かれた語数が B·H × 有効行 × live と違う`,
  );
};

/**
 * 行ブロックの割り方に依らない形へ畳む — `(z, chunk 内のグローバル行, live 列)` の順に、
 * **書かれる行だけ**を並べた語列。行ブロックを割ると S の行ストライド（`rows_block`）が
 * 変わるので、割り方の違う 2 回を突き合わせるにはこの正規化が要る。
 */
const liveRows = (
  spec: StateCase,
  blocks: readonly Uint32Array<ArrayBuffer>[],
): number[] => {
  const colCap = caseColCap(spec);
  const batchHeads = spec.batch * spec.heads;
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  const out: number[] = [];
  for (let z = 0; z < batchHeads; z += 1) {
    blocksOf(spec).forEach((block, index) => {
      const rows = stateEffectiveRows(block.rows, block.offset, spec.query);
      for (let local = 0; local < rows; local += 1) {
        const base = (z * block.rows + local) * colCap;
        for (let col = 0; col < live; col += 1) out.push(blocks[index][base + col]);
      }
    });
  }
  return out;
};

/** 語単位の完全一致（① との差はどれ 1 つも許さない）。 */
const assertSameWords = (
  actual: readonly Uint32Array<ArrayBuffer>[],
  expected: readonly Uint32Array<ArrayBuffer>[],
  label: string,
): void => {
  assertEquals(actual.length, expected.length, `${label}: 行ブロックの本数`);
  for (let block = 0; block < actual.length; block += 1) {
    const got = actual[block];
    const want = expected[block];
    assertEquals(got.length, want.length, `${label}: ブロック ${block} の語数`);
    for (let index = 0; index < got.length; index += 1) {
      if (got[index] === want[index]) continue;
      throw new Error(
        `${label}: ブロック ${block} の語 ${index} が違う（①ₜ 0x${
          got[index].toString(16).padStart(8, "0")
        } / ① 0x${want[index].toString(16).padStart(8, "0")}）`,
      );
    }
  }
};

Deno.test({
  name: "①ₜ（K 行タイル共有）は ① と S バッファが語単位で完全一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    // 格子全体で「見えたか」を数える。−inf も残骸も**ケースによっては 0 語が正しい**
    // （live = col_cap かつ pad 行が無い形など）ので、per-case ではなく格子で見る。
    let negInf = 0;
    let poison = 0;
    let value = 0;
    try {
      for (const spec of CASES) {
        const inputs = makeInputs(spec);
        const reference = await runStateQk(gpu.device, spec, inputs, "sequential", { cache });
        const tiled = await runStateQk(gpu.device, spec, inputs, "tiled", { cache });
        assertSameWords(tiled, reference, spec.name);
        // 門が空振りでないこと: 書かれた語数が spec（`B·H × 有効行 × live`）どおり。
        // 両経路が揃って何も書かない形はここで落ちる。
        blocksOf(spec).forEach((block, index) => {
          assertWriteCoverage(spec, tiled[index], block, `${spec.name} ①ₜ block ${index}`);
          assertWriteCoverage(spec, reference[index], block, `${spec.name} ① block ${index}`);
        });
        for (const words of tiled) {
          for (const word of words) {
            if (word === STATE_NEG_INF_BITS) negInf += 1;
            else if (word === POISON_BITS) poison += 1;
            else value += 1;
          }
        }
      }
    } finally {
      gpu.destroy();
    }
    assert(negInf > 0, "格子のどのケースでも述語外の −inf が出ていない");
    assert(poison > 0, "格子のどのケースでも残骸（[live, col_cap) / pad 行）が出ていない");
    assert(value > 0, "格子のどのケースでも実値が 1 語も書かれていない");
  },
});

Deno.test({
  name:
    "①ₜ は 2 回 dispatch・容量 C・行ブロックの割り方のどれを変えても書いた語が動かない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      const spec = CASES[2];
      const inputs = makeInputs(spec);
      const first = await runStateQk(gpu.device, spec, inputs, "tiled", { cache });
      const second = await runStateQk(gpu.device, spec, inputs, "tiled", { cache });
      assertSameWords(first, second, `${spec.name}（2 回目）`);

      // 容量 C を変えても値は動かない（ADR 0066 決定 3 の仕事量条件の裏 — S の中身が容量に
      // 依存しないこと）。full は `col_cap = C` なので行ストライドが変わる → live 範囲だけを
      // 取り出して突き合わせる。
      const small: StateCase = { ...spec, name: `${spec.name} C64`, capacity: 64 };
      const large: StateCase = { ...spec, name: `${spec.name} C160`, capacity: 160 };
      // MUST: 大容量側のスロットは**同じ物理行に同じ値**を置き直す（生成器が添字依存なので、
      // 容量を変えたまま作り直すと平面ごとの起点がずれて「入力が違う 2 回」を比べてしまう）。
      const smallInputs = makeInputs(small);
      const relayout = (source: Float32Array<ArrayBuffer>, poison: number) => {
        const planes = small.batch * small.kvHeads;
        const grown = seeded(planes * large.capacity * small.depth, () => poison);
        for (let plane = 0; plane < planes; plane += 1) {
          for (let row = 0; row < small.capacity; row += 1) {
            for (let d = 0; d < small.depth; d += 1) {
              grown[(plane * large.capacity + row) * small.depth + d] =
                source[(plane * small.capacity + row) * small.depth + d];
            }
          }
        }
        return grown;
      };
      const largeInputs: StateInputs = {
        ...smallInputs,
        slotK: relayout(smallInputs.slotK, SLOT_POISON_K),
        slotV: relayout(smallInputs.slotV, SLOT_POISON_V),
      };
      const live = stateLiveColumns(spec.window, spec.past, spec.query);
      const rows = spec.batch * spec.heads * spec.chunkRows;
      const liveRange = (words: Uint32Array<ArrayBuffer>, colCap: number): number[] => {
        const out: number[] = [];
        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < live; col += 1) out.push(words[row * colCap + col]);
        }
        return out;
      };
      const smallWords = await runStateQk(gpu.device, small, smallInputs, "tiled", { cache });
      const largeWords = await runStateQk(gpu.device, large, largeInputs, "tiled", { cache });
      assertEquals(
        liveRange(smallWords[0], caseColCap(small)),
        liveRange(largeWords[0], caseColCap(large)),
        "容量 C を変えると live 範囲の S が動いた",
      );

      // 行ブロックの割り方も値を動かさない（ADR 0067 決定 7）。S の行ストライドが変わるので
      // `(z, グローバル行, live 列)` へ畳んでから突き合わせる。1 枚 / 3 枚 / 端数ありの 3 通り。
      const whole: StateCase = { ...spec, name: `${spec.name} 1 枚` };
      const wholeWords = await runStateQk(gpu.device, whole, inputs, "tiled", { cache });
      assert(liveRows(whole, wholeWords).length > 0, "畳んだ語列が空（突合が空振りしている）");
      for (const rowsBlock of [8, 5, 1]) {
        const split: StateCase = { ...spec, name: `${spec.name} split ${rowsBlock}`, rowsBlock };
        const splitWords = await runStateQk(gpu.device, split, inputs, "tiled", { cache });
        assertEquals(
          liveRows(split, splitWords),
          liveRows(whole, wholeWords),
          `行ブロック ${rowsBlock} 枚割りで S の語が動いた`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "①ₜ の故障注入（半スケール除去 / 2 源の取り違え）が語一致の門で落ちる",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    // P > 0（2 源）・sliding でない・GQA の 1 ケースで両変異を撃つ。
    const spec = CASES[2];
    const inputs = makeInputs(spec);
    try {
      const reference = await runStateQk(gpu.device, spec, inputs, "sequential", { cache });
      for (
        const [label, mutate] of [
          // ① B ローダの半スケールを外す（k 側にだけ scale が掛からない = 1 項の式が ① と違う）
          [
            "B ローダの半スケール除去",
            (wgsl: string) => wgsl.replace(/^ *(wv\d+) = \1 \* dims\.scale;\n/gm, ""),
          ],
          // ② 列基点の 2 源を取り違える（全列を ins から読む）
          [
            "列基点の 2 源取り違え",
            (wgsl: string) => wgsl.replace(/(let kpast\d+ = kcol\d+) < past;/g, "$1 < 0u;"),
          ],
        ] as const
      ) {
        let mutated = false;
        const broken = await runStateQk(gpu.device, spec, inputs, "tiled", {
          cache,
          mutate: (_kernel, wgsl) => {
            const after = mutate(wgsl);
            assertMutated(wgsl, after, label);
            mutated = true;
            return after;
          },
        });
        assert(mutated, `${label}: 変異が呼ばれていない`);
        let differs = false;
        for (let block = 0; block < broken.length && !differs; block += 1) {
          for (let index = 0; index < broken[block].length; index += 1) {
            if (broken[block][index] !== reference[block][index]) {
              differs = true;
              break;
            }
          }
        }
        assert(differs, `${label}: 変異を入れても S が ① と一致した（門が空振り）`);
      }
    } finally {
      gpu.destroy();
    }
  },
});
