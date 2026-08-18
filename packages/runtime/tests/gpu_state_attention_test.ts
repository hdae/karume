// states 形 attention 3 段 + `state_append`（ADR 0067 決定 4〜7）の**実 GPU 直接 dispatch 門**。
// executor / recipe-builder への結線は波 D-3 なので、ここではカーネルを手組みバッファで直接撃ち、
// CPU 参照（src/reference/state-attention.ts）と突き合わせる。
//
// 格子は {full, sliding} × {r=1,2,8} × {P=0 / 1..W−1 / ≥W（ring wrap 跨ぎ）} × {Q=1, chunk>1} ×
// {pad 行あり / なし} × {B=1, B>1} × {行ブロック 1 枚 / 複数枚}。
//
// MUST: 突合だけで終わらせない。states 形の**構造的保証**は tolerance に隠れるので、
//   ① 空行の出力は**厳密 0**（Object.is）
//   ② 述語外の S は**厳密 −inf のビット列**（0xff800000）
//   ③ 非 resident なスロット行・pad 行の ins には毒値を置き、読まれたら値が跳ねる
//   ④ 容量 C を変えても出力が**ビット同一**（仕事量条件の裏 = 値が容量に依存しない）
//   ⑤ 行ブロックを割っても出力が**ビット同一**（ADR 0067 決定 7）
// を別々に見る。
// MUST: 故障注入で**各門がそれぞれの変異を検出する**ことを同じテストで示す（検出しない門は
// 何も見ていない）。

import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import {
  referenceStateAppend,
  referenceStateAttention,
  type StateAttentionRefInput,
} from "../src/reference/state-attention.ts";
import { stateColumnBase, stateLiveColumns, stateSliding } from "../src/kernels/state-attention.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  assertMutated,
  caseColCap,
  halfScale,
  runStateAppend,
  runStateAttention,
  seeded,
  STATE_S_POISON,
  type StateCase,
  type StateInputs,
  type StatePipelineCache,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * 突合の許容誤差。
 *
 * 根拠: 参照は f64 で積んで格納時に 1 度だけ丸める・GPU は f32 の逐次累積で、差は内積長 `D` と
 * 縮約長 `live` に比例する。下の `PARITY_CASES` 全 16 ケースの**実測最悪は maxAbs 6.6e-7**
 * （2026-08-18 / 出力の大きさは O(1)）で、`atol = 5e-6` はそこへ約 7.6 倍の余裕を積んだ値。
 * MUST: `DEFAULT_TOLERANCE`（atol 1e-5 / rtol 1e-3）を使わない — rtol 1e-3 は「窓の外の key が
 * 混ざった」級の誤りを平気で通す（このファイルの門はそれを見るためにある）。
 * MUST: `rtol = 0`。states 形は**厳密 0 が正解の要素**（空行・述語外の寄与）を正規に含むので、
 * 相対項は 0 近傍で効かない一方、大きい要素の誤りを隠す側にだけ働く。
 * NOTE: 実測値はドライバの fma 使用に依るので、別アダプタでの再測定は「実測 → 余裕を積む」の
 * 手順ごと繰り返すこと（テストは最悪値を毎回 stdout に出す）。
 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/** 決定的な入力列（乱数は使わない — 失敗が再現しないため）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** 読まれてはいけない場所に置く毒値（読まれたら tolerance を軽く超える大きさ）。 */
const SLOT_POISON_K = 9;
const SLOT_POISON_V = 400;

/**
 * ①③ が読む 5 本を組む。
 *
 * MUST: **非 resident なスロット物理行**（sliding のエビクト済み行 / full の未書込み行）と
 * **ins の pad 行**（`row ≥ Q`）には毒値を置く。契約上どちらも読まれないので、読む実装が
 * 値で落ちる（構造の誤りを tolerance の内側に隠さない唯一の方法）。
 */
const makeInputs = (spec: StateCase): StateInputs => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = spec;
  const kvPlanes = batch * kvHeads;
  const q = seeded(batch * heads * chunkRows * depth, QUERY);
  const insK = seeded(kvPlanes * chunkRows * depth, KEY);
  const insV = seeded(kvPlanes * chunkRows * depth, VALUE);
  const slotK = seeded(kvPlanes * capacity * depth, (i) => KEY(i + 3));
  const slotV = seeded(kvPlanes * capacity * depth, (i) => VALUE(i + 5));
  // resident な物理行の集合（論理 col [colBase, past) の像）
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
    // ins の pad 行（row ≥ Q）は論理 col 空間 [0, P+Q) に入らない = 読まれない
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

const PARITY_CASES: readonly StateCase[] = [
  // ── full ──────────────────────────────────────────────────────────────────
  // decode の最小形（Q=1・P=0 = 自己参照だけの causal 三角）
  {
    name: "full r1 decode P0",
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
  // prefill chunk（Q=M・pad 行なし）× B>1
  {
    name: "full r1 prefill B2",
    batch: 2,
    heads: 4,
    kvHeads: 4,
    chunkRows: 6,
    depth: 8,
    capacity: 32,
    window: 0,
    past: 0,
    query: 6,
  },
  // pad 行あり（Q<M）× GQA r=2 × D 端数
  {
    name: "full r2 pad B2 D6",
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
  // MQA 8:1 の decode（Gemma 4 E2B 型）× 大きめ P
  {
    name: "full r8 MQA decode",
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
  // 行ブロック 3 枚（端数ブロックを含む）
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
  // タイル境界（live = 16 の倍数ちょうど・D = 16 の倍数ちょうど）
  {
    name: "full r2 tile exact",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 32,
    capacity: 64,
    window: 0,
    past: 16,
    query: 16,
  },
  // ── sliding ───────────────────────────────────────────────────────────────
  // P=0（窓が効かない decode）
  {
    name: "sliding W8 P0 decode",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 8,
    window: 8,
    past: 0,
    query: 1,
  },
  // 0 < P < W−1（全 past が resident・wrap しない）+ pad 行
  {
    name: "sliding W8 P3 pad",
    batch: 2,
    heads: 4,
    kvHeads: 4,
    chunkRows: 4,
    depth: 8,
    capacity: 8,
    window: 8,
    past: 3,
    query: 2,
  },
  // P = W−1（下限が効き始める境界）
  {
    name: "sliding W4 P3 edge",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 3,
    depth: 4,
    capacity: 4,
    window: 4,
    past: 3,
    query: 2,
  },
  // P ≫ W（ring が何周もした後 — 物理行の写像が効く）
  {
    name: "sliding W4 P10 wrap",
    batch: 2,
    heads: 4,
    kvHeads: 4,
    chunkRows: 4,
    depth: 8,
    capacity: 4,
    window: 4,
    past: 10,
    query: 2,
  },
  // C > W（読み側が `% C` に化けたら落ちる形）
  {
    name: "sliding W4 C8 wrap",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 3,
    depth: 4,
    capacity: 8,
    window: 4,
    past: 9,
    query: 2,
  },
  // GQA r=2 × wrap × pad 行
  {
    name: "sliding W6 r2 wrap pad",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    chunkRows: 5,
    depth: 6,
    capacity: 6,
    window: 6,
    past: 13,
    query: 3,
  },
  // MQA 8:1 × wrap decode（Gemma 4 E2B の sliding 層型）
  {
    name: "sliding W8 r8 MQA",
    batch: 2,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 21,
    query: 1,
  },
  // **空行が正規に出る形**（pad 行が窓から落ちる）
  {
    name: "sliding W2 empty rows",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 4,
    depth: 4,
    capacity: 2,
    window: 2,
    past: 0,
    query: 1,
  },
  // Q > W（今 step の key が窓より多い — current は ins から読むので正規形）
  {
    name: "sliding W2 Q4 gt W",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 4,
    depth: 4,
    capacity: 2,
    window: 2,
    past: 5,
    query: 4,
  },
  // 行ブロック 2 枚 × sliding
  {
    name: "sliding W6 rowblock",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 8,
    depth: 8,
    capacity: 6,
    window: 6,
    past: 20,
    query: 8,
    rowsBlock: 5,
  },
  // **P が u32 の上限近く**（下限述語の引き算がどちら向きかで値が変わる唯一の形）。
  // `col + W > limit` の加算形だと `P + 0` に対する causal 対角 `col = P` で
  // `P + 2` が 2^32 を跨いで 0 に巻き戻り、自己参照の key が窓外と判定されて値が飛ぶ。
  {
    name: "sliding W2 P near u32 max",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 2,
    window: 2,
    past: 0xfffffffe,
    query: 1,
  },
];

/** 参照が「空行」と判定する行（`[B·H, M]` の平坦添字）。 */
const emptyRows = (spec: StateCase): readonly number[] => {
  const rows: number[] = [];
  const base = stateColumnBase(spec.window, spec.past);
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
    for (let row = 0; row < spec.chunkRows; row += 1) {
      const limit = spec.past + row;
      let any = false;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        if (col <= limit && (!stateSliding(spec.window) || col + spec.window > limit)) {
          any = true;
          break;
        }
      }
      if (!any) rows.push(plane * spec.chunkRows + row);
    }
  }
  return rows;
};

/** pad 行（`row ≥ Q`）の `[B·H, M]` 平坦添字。 */
const padRows = (spec: StateCase): readonly number[] => {
  const rows: number[] = [];
  for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
    for (let row = spec.query; row < spec.chunkRows; row += 1) {
      rows.push(plane * spec.chunkRows + row);
    }
  }
  return rows;
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

Deno.test({
  name: "states 形 attention が CPU 参照と一致する（実 GPU・full / sliding × GQA × 行ブロック）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    let worstAbs = 0;
    let worstRel = 0;
    try {
      for (const spec of PARITY_CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const actual = await runStateAttention(gpu.device, spec, inputs, { cache });
        const report = compareTensors(
          { dtype: "f32", data: actual.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(report.pass, true, `${spec.name}: ${formatAllclose(report)}`);
        worstAbs = Math.max(worstAbs, report.maxAbsError);
        worstRel = Math.max(worstRel, report.maxRelError);
        // ① pad 行は**厳密 0**（tolerance の内側に隠れる誤りを別途締める）。空行はその
        //   部分集合（valid 行は causal 自己参照で必ず非空）なので、包含も併せて見る —
        //   崩れたら「空行 → 0」を pad 行の 0 書きが包含している根拠が消える
        const empties = emptyRows(spec);
        const pads = padRows(spec);
        assertEquals(
          empties.filter((row) => !pads.includes(row)),
          [],
          `${spec.name}: 空行が pad 行の外に出た（空行 ⊂ pad 行が崩れている）`,
        );
        for (const row of pads) {
          for (let d = 0; d < spec.depth; d += 1) {
            assertEquals(
              Object.is(actual.out[row * spec.depth + d], 0),
              true,
              `${spec.name}: pad 行 ${row} の出力が厳密 0 でない（${
                actual.out[row * spec.depth + d]
              }）`,
            );
          }
        }
        // ③ 出力に毒値が残っていない（pad 行を含む全 M 行が書かれた = full-write）
        assertEquals(
          actual.out.some((value) => value === STATE_S_POISON),
          false,
          `${spec.name}: 出力に毒値が残っている（③PV が全行を書いていない）`,
        );
      }
    } finally {
      gpu.destroy();
    }
    // 実測最悪値を tolerance の根拠として残す（緩めた瞬間・誤差が伸びた瞬間に桁で気づける）
    console.log(`[states parity] maxAbs=${worstAbs} maxRel=${worstRel}`);
  },
});

/**
 * 述語の直接検査に使う 1 枚ブロックのケース（S を読み戻してビット列で見る）。
 *
 * 条件は 3 つ: ①有効行に述語内・述語外の列が**両方**実在する（causal 上限は全ケース・sliding の
 * 下限は `P10 wrap` / `W6 r2 wrap pad` が踏む）②pad 行が実在する（S が書かれないことの門）
 * ③行ブロック 1 枚（局所行 = グローバル行）。
 * NOTE: 波 D-7 で ① が有効行だけを覆うようになったので、`sliding W2 empty rows`（Q=1 で
 * 有効行が row 0 の 1 本・その 1 列は必ず述語内）はここでは述語外の列を作れない。同ケースは
 * parity 側と故障注入（③PV の pad 行 0 書き）で引き続き踏む。
 */
const PREDICATE_CASES: readonly StateCase[] = [
  PARITY_CASES[2],
  PARITY_CASES[7],
  PARITY_CASES[9],
  PARITY_CASES[11],
];

Deno.test({
  name: "①QK が書く S は述語外が厳密 −inf・live 範囲は全列が書かれる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      for (const spec of PREDICATE_CASES) {
        const inputs = makeInputs(spec);
        const { scores } = await runStateAttention(gpu.device, spec, inputs, { cache });
        const colCap = caseColCap(spec);
        const base = stateColumnBase(spec.window, spec.past);
        const live = stateLiveColumns(spec.window, spec.past, spec.query);
        const rowsBlock = spec.rowsBlock ?? spec.chunkRows;
        // 単一ブロックのケースだけを並べているので rowOffset = 0 = 有効行は先頭 Q 行
        const effRows = Math.min(spec.query, rowsBlock);
        const bits = bitsOf(scores);
        let masked = 0;
        let written = 0;
        let skipped = 0;
        for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
          for (let local = 0; local < effRows; local += 1) {
            // 単一ブロックのケースだけを並べているので、局所行 = グローバル行
            const limit = spec.past + local;
            for (let cl = 0; cl < live; cl += 1) {
              const col = base + cl;
              const inWindow = col <= limit &&
                (!stateSliding(spec.window) || limit - col < spec.window);
              const at = (plane * rowsBlock + local) * colCap + cl;
              if (inWindow) {
                assertEquals(
                  Number.isFinite(scores[at]),
                  true,
                  `${spec.name}: 述語内 (${local},${cl}) が有限でない`,
                );
                assertEquals(
                  scores[at] === STATE_S_POISON,
                  false,
                  `${spec.name}: 述語内 (${local},${cl}) に毒値が残っている（書かれていない）`,
                );
                written += 1;
              } else {
                assertEquals(
                  bits[at],
                  0xff800000,
                  `${spec.name}: 述語外 (${local},${cl}) が −inf のビット列でない`,
                );
                masked += 1;
              }
            }
          }
          // pad 行の S は**1 語も書かれない**（仕事量が M ではなく Q に比例することの直接の裏 —
          // 毒値がそのまま残る。ここが書かれていたら ① が全 M 行を回している）
          for (let local = effRows; local < rowsBlock; local += 1) {
            for (let cl = 0; cl < live; cl += 1) {
              const at = (plane * rowsBlock + local) * colCap + cl;
              assertEquals(
                scores[at],
                STATE_S_POISON,
                `${spec.name}: pad 行 (${local},${cl}) の S が書かれている（① が M 行を回した）`,
              );
              skipped += 1;
            }
          }
        }
        // 門が空振りしていない（両側の列と pad 行が実在する形を選んである）
        assert(written > 0, `${spec.name}: 述語内の列が 1 つも無い`);
        assert(masked > 0, `${spec.name}: 述語外の列が 1 つも無い（下限 / 上限が効いていない形）`);
        assert(skipped > 0, `${spec.name}: pad 行が 1 つも無い（仕事量の門が空振り）`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "state_append がスロットへ書いた値を ①③ が読み戻す（ring 往復・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // W=4 / C=8（読み側が `% C` に化けたら落ちる）・P=6 で ring が一周した後の形
      const spec: StateCase = {
        name: "append→attention round trip",
        batch: 2,
        heads: 4,
        kvHeads: 2,
        chunkRows: 3,
        depth: 4,
        capacity: 8,
        window: 4,
        past: 6,
        query: 2,
      };
      const kvPlanes = spec.batch * spec.kvHeads;
      const appendSpec = {
        kvPlanes,
        chunkRows: spec.chunkRows,
        depth: spec.depth,
        capacity: spec.capacity,
        window: spec.window,
      };
      // 「前 step までの append」を 3 回（P: 0→2→4→6）積んで実際のスロットを作る
      let slotK = seeded(kvPlanes * spec.capacity * spec.depth, () => STATE_S_POISON);
      let slotV = slotK.slice();
      let expectedK = slotK.slice();
      let expectedV = slotV.slice();
      for (let step = 0; step < 3; step += 1) {
        const past = step * 2;
        const xK = seeded(kvPlanes * spec.chunkRows * spec.depth, (i) => KEY(i + step * 13));
        const xV = seeded(kvPlanes * spec.chunkRows * spec.depth, (i) => VALUE(i + step * 7));
        const lengths = { ...appendSpec, past, query: 2 };
        slotK = await runStateAppend(gpu.device, lengths, xK, slotK, { cache });
        slotV = await runStateAppend(gpu.device, lengths, xV, slotV, { cache });
        // MUST: copy なので**ビット一致**（参照は同じ写像で書いた slot の複製）
        expectedK = referenceStateAppend({ ...lengths, x: xK, slot: expectedK })
          .data as Float32Array<ArrayBuffer>;
        expectedV = referenceStateAppend({ ...lengths, x: xV, slot: expectedV })
          .data as Float32Array<ArrayBuffer>;
        assertEquals([...bitsOf(slotK)], [...bitsOf(expectedK)], `step ${step}: k スロット`);
        assertEquals([...bitsOf(slotV)], [...bitsOf(expectedV)], `step ${step}: v スロット`);
      }
      // wrap で潰れた行（論理 col 0,1 = 物理行 0,1 が論理 col 4,5 で上書き済み）が
      // 実際に「古い値ではない」ことを確認する
      assert(
        slotK.some((value) => value !== STATE_S_POISON),
        "append が 1 行も書いていない",
      );
      // 読み戻し: この slot をそのまま ①③ に食わせる（pad 行の ins は毒値のまま）
      const inputs = { ...makeInputs(spec), slotK, slotV };
      const expected = referenceStateAttention(refInput(spec, inputs));
      const actual = await runStateAttention(gpu.device, spec, inputs, { cache });
      const report = compareTensors({ dtype: "f32", data: actual.out }, expected, STATE_TOLERANCE);
      assertEquals(report.pass, true, `ring 往復: ${formatAllclose(report)}`);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "同一入力の 2 回 dispatch はビット単位で同一（states 形の決定性・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      for (const spec of [PARITY_CASES[5], PARITY_CASES[11]]) {
        const inputs = makeInputs(spec);
        const first = await runStateAttention(gpu.device, spec, inputs, { cache });
        const second = await runStateAttention(gpu.device, spec, inputs, { cache });
        assertEquals([...bitsOf(first.out)], [...bitsOf(second.out)], `${spec.name}: 2 回目が違う`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "容量 C と行ブロック分割は出力のビット列を動かさない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // ④ 容量非依存: 同じ (P, Q) で C を 32 → 4096 に増やしても出力はビット同一
      // （増えるのは S の列ストライドと未読の毒値領域だけ = 仕事量条件の裏返し）
      const small: StateCase = {
        name: "capacity 32",
        batch: 2,
        heads: 4,
        kvHeads: 2,
        chunkRows: 4,
        depth: 8,
        capacity: 32,
        window: 0,
        past: 7,
        query: 3,
      };
      const large: StateCase = { ...small, name: "capacity 4096", capacity: 4096 };
      const smallInputs = makeInputs(small);
      // MUST: 大容量側のスロットは**同じ物理行に同じ値**を置き直す（生成器が添字依存なので、
      // 容量を変えたまま作り直すと平面ごとの起点がずれて「入力が違う 2 回」を比べてしまう）
      const relayout = (source: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
        const planes = small.batch * small.kvHeads;
        const grown = seeded(planes * large.capacity * small.depth, () => SLOT_POISON_V);
        for (let plane = 0; plane < planes; plane += 1) {
          const from = plane * small.capacity * small.depth;
          grown.set(
            source.subarray(from, from + small.capacity * small.depth),
            plane * large.capacity * small.depth,
          );
        }
        return grown;
      };
      const largeInputs: StateInputs = {
        ...smallInputs,
        slotK: relayout(smallInputs.slotK),
        slotV: relayout(smallInputs.slotV),
      };
      const smallOut = await runStateAttention(gpu.device, small, smallInputs, { cache });
      const largeOut = await runStateAttention(gpu.device, large, largeInputs, { cache });
      assertEquals([...bitsOf(smallOut.out)], [...bitsOf(largeOut.out)], "容量で出力が動いた");

      // ⑤ 行ブロック非依存: 1 枚と 3 枚（端数込み）で出力はビット同一（ADR 0067 決定 7）
      const whole: StateCase = {
        name: "rowblock whole",
        batch: 1,
        heads: 4,
        kvHeads: 1,
        chunkRows: 7,
        depth: 8,
        capacity: 6,
        window: 6,
        past: 11,
        query: 5,
      };
      const split: StateCase = { ...whole, name: "rowblock 3", rowsBlock: 3 };
      const inputs = makeInputs(whole);
      const wholeOut = await runStateAttention(gpu.device, whole, inputs, { cache });
      const splitOut = await runStateAttention(gpu.device, split, inputs, { cache });
      assertEquals(
        [...bitsOf(wholeOut.out)],
        [...bitsOf(splitOut.out)],
        "行ブロックで出力が動いた",
      );
    } finally {
      gpu.destroy();
    }
  },
});

/** 故障注入 1 件（`apply` は対象カーネルの WGSL だけを書き換える）。 */
type Injection = {
  readonly label: string;
  /** どのケースで撃つか（変異が値に出る形を選ぶ）。 */
  readonly spec: StateCase;
  readonly apply: (kernel: string, wgsl: string) => string;
};

const replaceIn = (
  target: readonly string[],
  from: string,
  to: string,
): (kernel: string, wgsl: string) => string =>
(kernel, wgsl) => {
  if (!target.includes(kernel)) return wgsl;
  const mutated = wgsl.replaceAll(from, to);
  assertMutated(wgsl, mutated, `${kernel}: ${from}`);
  return mutated;
};

const INJECTIONS: readonly Injection[] = [
  // ① sliding の下限述語を落とす（= causal 上限だけの実装。ADR 0067 決定 4 の第 5 巡 high）
  {
    label: "sliding の下限述語を削除",
    spec: PARITY_CASES[9],
    apply: replaceIn(["qk"], " && (limit - col) < params.window", ""),
  },
  // ①' 下限の引き算を u32 で巻き戻る加算形へ戻す（P が u32 上限近くでのみ値が変わる）
  {
    label: "下限述語を col + W > limit の加算形へ差し替え",
    spec: PARITY_CASES[16],
    apply: replaceIn(["qk"], "(limit - col) < params.window", "col + params.window > limit"),
  },
  // ② 読み側の ring 写像だけを `% C` に差し替える（読み書き同式 MUST を破る）
  {
    label: "読み側の ring 写像を col % capacity に差し替え",
    spec: PARITY_CASES[10],
    apply: replaceIn(["qk", "pv"], "col % params.window", "col % params.capacity"),
  },
  // ③ ③PV の pad 行 0 書きを外す（pad 行が live 走査へ落ち、①② が覆っていない S / stats を食う）。
  //   NOTE: 波 D-7 で ①② が有効行だけを覆うようになった結果、**空行は構造的に生じなくなった**
  //   （valid 行は causal 自己参照で必ず非空・空行 ⊂ pad 行）。② の空行ガードは防御として残るが、
  //   もう検出器を持てない — 「空行 / pad 行 → 厳密 0」（ADR 0067 決定 6 / ADR 0066 追記 6）を
  //   守っているのはこの分岐なので、故障注入の相手をそちらへ移した。
  {
    label: "③PV の pad 行 0 書きを削除",
    spec: PARITY_CASES[13],
    apply: replaceIn(["pv"], "    out[at] = 0.0;\n    return;\n", ""),
  },
  // ④ GQA の kv 写像で r を無視する（head 対応が崩れる）
  {
    label: "kv 平面の写像で r を無視",
    spec: PARITY_CASES[11],
    apply: replaceIn(["qk", "pv"], "z / params.kv_repeat", "z"),
  },
  // ⑤ ①QK が述語外を書かない（S の残骸を ② が食う — live 範囲の全書きが要る根拠）
  {
    label: "述語外の S を書かない",
    spec: PARITY_CASES[9],
    apply: replaceIn(
      ["qk"],
      "  s[(z * params.rows_block",
      "  if (!in_window(col, past + row)) { return; }\n  s[(z * params.rows_block",
    ),
  },
  // ⑥ 半スケールを片側だけにする（値が √ 倍ずれる — 契約の写し間違い）
  {
    label: "半スケールを k 側だけにする",
    spec: PARITY_CASES[2],
    apply: replaceIn(["qk"], "(q[q_base + d] * params.scale)", "q[q_base + d]"),
  },
];

Deno.test({
  name: "故障注入: 変異版はいずれも突合で赤くなる（門が実際に検出器であることの実証・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      for (const injection of INJECTIONS) {
        const spec = injection.spec;
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const actual = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          mutate: injection.apply,
        });
        const report = compareTensors(
          { dtype: "f32", data: actual.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(
          report.pass,
          false,
          `故障注入 '${injection.label}' が ${spec.name} で検出されなかった`,
        );
      }
      // ⑦ `state_append` の ring 写像を落とす（書き側 — 往復のビット一致門が検出器）
      const appendSpec = {
        kvPlanes: 2,
        chunkRows: 3,
        depth: 4,
        capacity: 8,
        window: 4,
        past: 6,
        query: 2,
      };
      const x = seeded(appendSpec.kvPlanes * appendSpec.chunkRows * appendSpec.depth, KEY);
      const slot = seeded(appendSpec.kvPlanes * appendSpec.capacity * appendSpec.depth, VALUE);
      const mutated = await runStateAppend(gpu.device, appendSpec, x, slot, {
        cache,
        mutate: replaceIn(["append"], "col % params.window", "col"),
      });
      const expected = referenceStateAppend({ ...appendSpec, x, slot })
        .data as Float32Array<ArrayBuffer>;
      assertEquals(
        [...bitsOf(mutated)].every((value, at) => value === bitsOf(expected)[at]),
        false,
        "故障注入 'append の ring 写像を落とす' が検出されなかった",
      );
    } finally {
      gpu.destroy();
    }
  },
});
