// states 形 attention 3 カーネル + `state_append`（ADR 0067 決定 4〜7）の**GPU を要らない面**:
// キーの変種ビット・生成の決定性・読み書き同式の共有・params の値域門・dispatch 幾何の純関数・
// CPU 参照の意味論。実 GPU との突合は gpu_state_attention_test.ts。
//
// MUST: CPU 参照の意味論は**手計算のオラクル**で押さえる（GPU 突合だけだと、参照と GPU が
// 同じ誤りを共有していても両方緑になる）。窓の集合が値に出る形（in-window の V を col 番号に
// して平均を作る）で、述語の上限・下限・リング写像の 3 つが 1 つの数値に畳まれる。

import { assertAlmostEquals, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import {
  STATE_ATTENTION_TILE_M,
  STATE_ATTENTION_TILE_X,
  STATE_NEG_INF_BITS,
  STATE_STATS_WORKGROUP_SIZE,
  stateAttentionParams,
  stateColumnBase,
  stateLiveColumns,
  statePvKey,
  statePvWgsl,
  statePvWorkgroups,
  stateQkKey,
  stateQkWgsl,
  stateQkWorkgroups,
  stateSliding,
  stateSlotRowWgsl,
  stateStatsKey,
  stateStatsParams,
  stateStatsWgsl,
  stateStatsWorkgroups,
} from "../src/kernels/state-attention.ts";
import {
  STATE_APPEND_WORKGROUP_SIZE,
  stateAppendKey,
  stateAppendParams,
  stateAppendWgsl,
  stateAppendWorkgroups,
} from "../src/kernels/state-append.ts";
import { ReferenceOpError } from "../src/reference/ops.ts";
import {
  referenceStateAppend,
  referenceStateAttention,
  type StateAttentionRefInput,
} from "../src/reference/state-attention.ts";

const VARIANTS: readonly (readonly [boolean, boolean])[] = [
  [false, false],
  [false, true],
  [true, false],
  [true, true],
];

Deno.test("states 形のキーは :sliding / :gqa の 2 ビットだけで分かれる（並び順は 1 箇所）", () => {
  assertEquals(stateQkKey(false, false), "attention_state_qk:v1:f32:wg16x4");
  assertEquals(stateQkKey(true, false), "attention_state_qk:v1:f32:wg16x4:sliding");
  assertEquals(stateQkKey(false, true), "attention_state_qk:v1:f32:wg16x4:gqa");
  assertEquals(stateQkKey(true, true), "attention_state_qk:v1:f32:wg16x4:sliding:gqa");
  assertEquals(statePvKey(true, true), "attention_state_pv:v1:f32:wg16x4:sliding:gqa");
  assertEquals(stateStatsKey(false), "attention_state_stats:v1:f32:wg256");
  assertEquals(stateStatsKey(true), "attention_state_stats:v1:f32:wg256:sliding");
  assertEquals(stateAppendKey(false), "state_append:v1:f32:wg256");
  assertEquals(stateAppendKey(true), "state_append:v1:f32:wg256:sliding");
  // 幾何の定数がキーに出ている（workgroup を動かせばキーが動く = 別パイプライン）
  assertEquals(STATE_ATTENTION_TILE_X, 16);
  assertEquals(STATE_ATTENTION_TILE_M, 4);
  assertEquals(STATE_STATS_WORKGROUP_SIZE, 256);
  assertEquals(STATE_APPEND_WORKGROUP_SIZE, 256);
  // 4 族 × 変種が全て別キー（同一構成が 2 通りのキーを持たない / 別構成が同じキーを持たない）
  const keys = [
    ...VARIANTS.map(([sliding, gqa]) => stateQkKey(sliding, gqa)),
    ...VARIANTS.map(([sliding, gqa]) => statePvKey(sliding, gqa)),
    stateStatsKey(false),
    stateStatsKey(true),
    stateAppendKey(false),
    stateAppendKey(true),
  ];
  assertEquals(new Set(keys).size, keys.length);
});

Deno.test("論理 col → スロット物理行の写像は読み（①③）と書き（state_append）で同一文字列", () => {
  for (const sliding of [false, true]) {
    const fragment = stateSlotRowWgsl(sliding);
    // MUST: 逐語で含まれること。片方だけ別式にすると ring が一周した後の全読みが黙って
    // 別の行を指す（ADR 0067 決定 4 の「読み書き同式 MUST」の機械証明）
    for (
      const [name, wgsl] of [
        ["qk", stateQkWgsl(sliding, false)],
        ["qk gqa", stateQkWgsl(sliding, true)],
        ["pv", statePvWgsl(sliding, false)],
        ["append", stateAppendWgsl(sliding)],
      ] as const
    ) {
      assertEquals(wgsl.includes(fragment), true, `${name}: slot_row の断片が共有されていない`);
    }
  }
  // sliding だけが剰余を持つ（full の写像は恒等）
  assertEquals(stateSlotRowWgsl(true).includes("col % params.window"), true);
  assertEquals(stateSlotRowWgsl(false).includes("%"), false);
});

Deno.test("同じ生成入力からは常に同一の WGSL が出る（states 形の決定性）", () => {
  for (const [sliding, gqa] of VARIANTS) {
    assertEquals(stateQkWgsl(sliding, gqa), stateQkWgsl(sliding, gqa));
    assertEquals(statePvWgsl(sliding, gqa), statePvWgsl(sliding, gqa));
  }
  for (const sliding of [false, true]) {
    assertEquals(stateStatsWgsl(sliding), stateStatsWgsl(sliding));
    assertEquals(stateAppendWgsl(sliding), stateAppendWgsl(sliding));
    // 変種が実際に別物であること（キーだけ分けて中身が同じ = 変種が効いていない）
    assertNotEquals(stateQkWgsl(sliding, false), stateQkWgsl(sliding, true));
  }
  assertNotEquals(stateStatsWgsl(false), stateStatsWgsl(true));
  assertNotEquals(stateAppendWgsl(false), stateAppendWgsl(true));
});

Deno.test("sliding 変種は causal 上限に**下限**を AND する（上限だけの実装を文字列で拒む）", () => {
  // ADR 0067 決定 4 の第 5 巡 high: 上限だけだと row > 0 が窓外 key を row ぶん沈黙混入する
  assertEquals(stateQkWgsl(true, false).includes("col + params.window > limit"), true);
  assertEquals(stateQkWgsl(false, false).includes("col + params.window > limit"), false);
  // 空行ガード（②）— 有限 sentinel でなく -inf identity + 分母ガードの構成
  assertEquals(stateStatsWgsl(false).includes("let empty = amax == neg_inf;"), true);
  assertEquals(STATE_NEG_INF_BITS, 0xff800000);
});

const GEOMETRY = {
  rowsBlock: 4,
  rowOffset: 0,
  chunkRows: 8,
  depth: 6,
  kvRepeat: 2,
  window: 0,
  capacity: 32,
  colCap: 32,
  scale: 0.5,
};

Deno.test("①③ の params は語順どおりに詰まる（同じ幾何を 2 つの表で持たない）", () => {
  const params = stateAttentionParams({ ...GEOMETRY, rowOffset: 4 });
  assertEquals(params.length, 12, "uniform struct の整列で 48 バイト");
  assertEquals([...params.subarray(0, 9)], [4, 4, 8, 6, 2, 0, 32, 32, STATE_NEG_INF_BITS]);
  assertEquals(new Float32Array(params.buffer)[9], 0.5);
  // ② は別 struct（rows / col_cap / window / neg_inf の 4 語ちょうど）
  const stats = stateStatsParams(24, 32, 0);
  assertEquals(stats.length, 4);
  assertEquals([...stats], [24, 32, 0, STATE_NEG_INF_BITS]);
});

Deno.test("①③ の params 門が沈黙誤値になる幾何を全て拒否する", () => {
  // r = 0（WGSL の u32 ゼロ除算は trap せず実装依存値 = kv 平面が化ける）
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, kvRepeat: 0 }),
    CodegenError,
    "kv_repeat",
  );
  // 行ブロックが chunk からはみ出す（q / O の行が別のバッチへ回り込む）
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, rowOffset: 6, rowsBlock: 4 }),
    CodegenError,
    "行ブロック",
  );
  // full: col_cap < C（P + Q ≤ C の context 側検査を通った形が S からはみ出す）
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, colCap: 31 }),
    CodegenError,
    "col_cap",
  );
  // sliding: W > C
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, window: 33, colCap: 40 }),
    CodegenError,
    "window",
  );
  // sliding: col_cap < (W−1) + M
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, window: 8, colCap: 14 }),
    CodegenError,
    "col_cap",
  );
  assertEquals(stateAttentionParams({ ...GEOMETRY, window: 8, colCap: 15 }).length, 12);
  // 非有限 scale / u32 域外
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, scale: Number.NaN }),
    CodegenError,
    "scale",
  );
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, depth: 2 ** 32 }),
    CodegenError,
    "depth",
  );
  assertThrows(() => stateStatsParams(0, 32, 0), CodegenError, "rows");
});

Deno.test("state_append の params 門（容量・正整数・u32 域）", () => {
  const base = { kvPlanes: 2, chunkRows: 4, depth: 3, capacity: 16, window: 0 };
  assertEquals([...stateAppendParams(base)], [2, 4, 3, 16, 0, 0, 0, 0]);
  assertEquals([...stateAppendParams({ ...base, window: 8 }).subarray(0, 5)], [2, 4, 3, 16, 8]);
  assertThrows(
    () => stateAppendParams({ ...base, window: 17 }),
    CodegenError,
    "window",
  );
  assertThrows(() => stateAppendParams({ ...base, depth: 0 }), CodegenError, "正整数");
  assertThrows(() => stateAppendParams({ ...base, capacity: -1 }), CodegenError, "capacity");
});

const DISPATCH = { batchHeads: 6, rowsBlock: 4, depth: 32, window: 0 };
const LIMIT = 65535;

Deno.test("dispatch 幾何は live 列に比例し、容量には依らない（ADR 0066 決定 3 の機構）", () => {
  // ①QK の x 軸 = ⌈live / 16⌉。full は P に比例して伸びる
  assertEquals(stateQkWorkgroups(DISPATCH, 0, 16, LIMIT, "t")[0], 1);
  assertEquals(stateQkWorkgroups(DISPATCH, 16, 16, LIMIT, "t")[0], 2);
  assertEquals(stateQkWorkgroups(DISPATCH, 1008, 16, LIMIT, "t")[0], 64);
  // 単調（P にも Q にも減らない）
  let previous = 0;
  for (const past of [0, 1, 7, 64, 512, 4096]) {
    const groups = stateQkWorkgroups(DISPATCH, past, 1, LIMIT, "t")[0];
    assertEquals(groups >= previous, true, `past=${past} で単調でない`);
    previous = groups;
  }
  // sliding は W で**頭打ち**（P を 8 倍しても live は (W−1)+Q のまま）
  const sliding = { ...DISPATCH, window: 64 };
  const cap = stateQkWorkgroups(sliding, 63, 16, LIMIT, "t")[0];
  assertEquals(stateQkWorkgroups(sliding, 512, 16, LIMIT, "t")[0], cap);
  assertEquals(stateQkWorkgroups(sliding, 131072, 16, LIMIT, "t")[0], cap);
  // 行 / z 軸は静的幾何そのもの
  assertEquals(stateQkWorkgroups(DISPATCH, 100, 4, LIMIT, "t").slice(1), [1, 6]);
  // ②③ は論理長に依らない（走査は invocation の内側 — 総反復回数の側で仕事量条件を満たす）
  assertEquals(stateStatsWorkgroups(DISPATCH, LIMIT, "t"), [24, 1, 1]);
  assertEquals(statePvWorkgroups(DISPATCH, LIMIT, "t"), [2, 1, 6]);
  // ② は grid-stride なので上限超過は**縮退**（②③ の別扱いが崩れていないこと）
  assertEquals(stateStatsWorkgroups({ ...DISPATCH, batchHeads: 4096 }, 100, "t"), [100, 1, 1]);
});

Deno.test("dispatch 幾何は容量 C / col_cap を引数に取らない（64 → 65536 でも同一）", () => {
  // 型として容量欄が無いことの実測版: 同じ (P, Q) で容量だけ違う params を作っても、
  // dispatch へ渡す幾何は 1 語も変わらない（= 容量比例の workgroup 数を書きようがない）
  const small = stateAttentionParams({ ...GEOMETRY, capacity: 64, colCap: 64 });
  const large = stateAttentionParams({ ...GEOMETRY, capacity: 65536, colCap: 65536 });
  assertNotEquals([...small], [...large], "params 側では容量が動いている");
  for (const [past, query] of [[0, 1], [7, 4], [513, 16]] as const) {
    assertEquals(
      stateQkWorkgroups(DISPATCH, past, query, LIMIT, "t"),
      stateQkWorkgroups(DISPATCH, past, query, LIMIT, "t"),
    );
  }
  assertEquals(Object.keys(DISPATCH).includes("capacity"), false);
  assertEquals(Object.keys(DISPATCH).includes("colCap"), false);
});

Deno.test("dispatch の上限超過はタイル系が fail loudly・grid-stride 系が縮退", () => {
  assertThrows(
    () => stateQkWorkgroups(DISPATCH, 100000, 1, 16, "t"),
    DispatchLimitError,
    "上限",
  );
  assertThrows(
    () => statePvWorkgroups({ ...DISPATCH, batchHeads: 70000 }, 65535, "t"),
    DispatchLimitError,
    "上限",
  );
  // 論理長の値域門（Q ≥ 1・u32）
  assertThrows(() => stateQkWorkgroups(DISPATCH, 0, 0, LIMIT, "t"), CodegenError, "queryLength");
  assertThrows(() => stateQkWorkgroups(DISPATCH, -1, 1, LIMIT, "t"), CodegenError, "past");
});

Deno.test("state_append の dispatch は Q に比例し P には依らない", () => {
  const geometry = { kvPlanes: 4, depth: 64 };
  assertEquals(stateAppendWorkgroups(geometry, 1, LIMIT, "t"), [1, 1, 1]);
  assertEquals(stateAppendWorkgroups(geometry, 16, LIMIT, "t"), [16, 1, 1]);
  assertEquals(stateAppendWorkgroups(geometry, 64, LIMIT, "t"), [64, 1, 1]);
  // grid-stride なので上限は縮退
  assertEquals(stateAppendWorkgroups(geometry, 64, 8, "t"), [8, 1, 1]);
  assertThrows(() => stateAppendWorkgroups(geometry, 0, LIMIT, "t"), CodegenError, "正整数");
});

Deno.test("live 範囲の純関数（full は P+Q・sliding は W で頭打ち）", () => {
  assertEquals(stateSliding(0), false);
  assertEquals(stateSliding(1), true);
  // full
  assertEquals(stateColumnBase(0, 100), 0);
  assertEquals(stateLiveColumns(0, 100, 8), 108);
  // sliding: P < W−1 は全 past が resident
  assertEquals(stateColumnBase(4, 2), 0);
  assertEquals(stateLiveColumns(4, 2, 1), 3);
  // sliding: P ≥ W−1 は直前 W−1 行だけ resident（row 0 の窓が丸ごと入る境界）
  assertEquals(stateColumnBase(4, 9), 6);
  assertEquals(stateLiveColumns(4, 9, 1), 4);
  assertEquals(stateLiveColumns(4, 1000, 1), 4);
});

/** D=1 の手計算オラクル用の入力（in-window の V を col 番号にすると出力が「窓の平均」になる）。 */
const windowProbe = (
  options: {
    readonly window: number;
    readonly past: number;
    readonly query: number;
    readonly chunkRows: number;
    readonly capacity: number;
    /** resident でない物理行に置く毒値（読まれたら出力が跳ねる）。 */
    readonly poison: number;
  },
): StateAttentionRefInput => {
  const { window, past, query, chunkRows, capacity, poison } = options;
  const slotK = new Float32Array(capacity);
  const slotV = new Float32Array(capacity).fill(poison);
  const insK = new Float32Array(chunkRows);
  const insV = new Float32Array(chunkRows).fill(poison);
  // K は全て 1（= 全 in-window 列のスコアが同値 → 重みが一様）
  slotK.fill(1);
  insK.fill(1);
  // V(col) = col を**論理 col → 物理行**の写像で置く。resident でない物理行は poison のまま
  const base = stateColumnBase(window, past);
  for (let col = base; col < past; col += 1) {
    slotV[stateSliding(window) ? col % window : col] = col;
  }
  for (let row = 0; row < query; row += 1) insV[row] = past + row;
  return {
    batch: 1,
    heads: 1,
    kvHeads: 1,
    chunkRows,
    depth: 1,
    capacity,
    window,
    past,
    query,
    q: new Float32Array(chunkRows).fill(1),
    insK,
    insV,
    slotK,
    slotV,
    scale: 1,
  };
};

Deno.test("CPU 参照: sliding の窓は両側で切られる（手計算 — 上限だけなら値が跳ねる）", () => {
  // W=4 / P=5 / Q=1: row 0 が見るのは col ∈ {2,3,4,5}（下限 = 5−4+1 = 2）。平均 = 3.5。
  // 上限だけの実装なら col 0,1 相当（= 物理行 0,1 の残骸）が混ざって値が跳ねる
  const probe = windowProbe({
    window: 4,
    past: 5,
    query: 1,
    chunkRows: 1,
    capacity: 4,
    poison: 1e6,
  });
  const out = referenceStateAttention(probe);
  assertEquals(out.shape, [1, 1, 1, 1]);
  assertAlmostEquals(out.data[0], 3.5, 1e-6);
});

Deno.test("CPU 参照: full は先頭から causal 上限まで（pad 行も計算される）", () => {
  // P=3 / Q=2 / M=3: 論理 col 空間は [0, P+Q) = {0..4}（pad 行の key は**論理 col 空間に
  // 入らない** — 足されるのは Q 本だけ）。row 0 → 上限 3 で {0,1,2,3}（平均 1.5）・
  // row 1 → 上限 4 で {0..4}（平均 2）・row 2 は pad 行で上限 5 だが live が {0..4} で
  // 尽きるので同じ {0..4}（平均 2 — 値は契約上無意味だが **書かれる**）
  const probe = windowProbe({
    window: 0,
    past: 3,
    query: 2,
    chunkRows: 3,
    capacity: 8,
    poison: 0,
  });
  const out = referenceStateAttention(probe);
  assertAlmostEquals(out.data[0], 1.5, 1e-6);
  assertAlmostEquals(out.data[1], 2, 1e-6);
  assertAlmostEquals(out.data[2], 2, 1e-6);
});

Deno.test("CPU 参照: 空行（述語を満たす col が無い pad 行）の出力は厳密 0", () => {
  // W=2 / P=0 / Q=1 / M=3: row 2 は下限 col ≥ 1 に対し live が {0} だけ = 空行
  const probe = windowProbe({
    window: 2,
    past: 0,
    query: 1,
    chunkRows: 3,
    capacity: 2,
    poison: 1e6,
  });
  const out = referenceStateAttention(probe);
  assertEquals(Object.is(out.data[2], 0), true, "空行は厳密 0（−0 でも NaN でもない）");
  // row 0 は col 0 だけを見る（V(0) = 0）・row 1 は col 0,1 のうち下限で col 0 が落ち…
  // live = {0} なので row 1 も col 0 だけ（P=0 なので col 0 は ins 行 0）
  assertAlmostEquals(out.data[0], 0, 1e-6);
});

Deno.test("CPU 参照: GQA は kv 平面を ⌊z/r⌋ で写す（剰余写像だと値が変わる）", () => {
  const [batch, heads, kvHeads, chunkRows, depth, capacity] = [2, 4, 2, 1, 1, 1];
  const repeat = heads / kvHeads;
  // kv 平面ごとに違う V を置く（z=0,1 → 平面 0 / z=2,3 → 平面 1 / …）
  const insV = Float32Array.from([10, 20, 30, 40]);
  const out = referenceStateAttention({
    batch,
    heads,
    kvHeads,
    chunkRows,
    depth,
    capacity,
    window: 0,
    past: 0,
    query: 1,
    q: new Float32Array(batch * heads * chunkRows * depth).fill(1),
    insK: new Float32Array(batch * kvHeads * chunkRows * depth).fill(1),
    insV,
    slotK: new Float32Array(batch * kvHeads * capacity * depth),
    slotV: new Float32Array(batch * kvHeads * capacity * depth),
    scale: 1,
  });
  // z = b·H + h に対し ⌊z/r⌋ = b·Hkv + ⌊h/r⌋（B の項が効く形 — 剰余 z%Hkv だと b が消える）
  for (let z = 0; z < batch * heads; z += 1) {
    assertAlmostEquals(out.data[z], insV[Math.floor(z / repeat)], 1e-6, `z=${z}`);
  }
});

Deno.test("CPU 参照 state_append: 先頭 Q 行だけ書き・wrap で旧行を潰し・入力を破壊しない", () => {
  const spec = { kvPlanes: 1, chunkRows: 4, depth: 1, capacity: 4, window: 4 };
  const slot = Float32Array.from([-1, -2, -3, -4]);
  const x = Float32Array.from([100, 200, 300, 400]);
  // P=2 / Q=2 → 論理行 2,3 = 物理行 2,3。pad 行（2,3）は書かない
  const after = referenceStateAppend({ ...spec, past: 2, query: 2, x, slot });
  assertEquals([...after.data], [-1, -2, 100, 200]);
  assertEquals([...slot], [-1, -2, -3, -4], "入力スロットを破壊していない");
  // wrap: P=3 / Q=3 → 論理行 3,4,5 = 物理行 3,0,1
  const wrapped = referenceStateAppend({ ...spec, past: 3, query: 3, x, slot });
  assertEquals([...wrapped.data], [200, 300, -3, 100]);
  // Q > W: 同じ物理行へ複数の論理行が写る → **最後の論理行が残る**
  const collide = referenceStateAppend({
    ...spec,
    window: 2,
    capacity: 2,
    past: 0,
    query: 4,
    x,
    slot: Float32Array.from([-1, -2]),
  });
  assertEquals([...collide.data], [300, 400]);
});

Deno.test("CPU 参照: 形の取り違えは突合の前で落ちる", () => {
  const probe = windowProbe({
    window: 0,
    past: 1,
    query: 1,
    chunkRows: 1,
    capacity: 4,
    poison: 0,
  });
  assertThrows(
    () => referenceStateAttention({ ...probe, q: new Float32Array(2) }),
    ReferenceOpError,
    "q",
  );
  assertThrows(
    () => referenceStateAttention({ ...probe, heads: 3, kvHeads: 2 }),
    ReferenceOpError,
    "割り切れない",
  );
  assertThrows(
    () => referenceStateAttention({ ...probe, past: 4, capacity: 4 }),
    ReferenceOpError,
    "容量",
  );
  assertThrows(
    () => referenceStateAttention({ ...probe, query: 2 }),
    ReferenceOpError,
    "queryLength",
  );
});
