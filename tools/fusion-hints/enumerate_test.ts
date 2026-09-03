/**
 * fusion-hints の列挙面の検証（GPU 不要）。
 *
 * 見る 2 点:
 *
 * 1. **集計** — 合成 IR 1 本で、op 名列ごとの本数と「極大」の印が付くこと（n-gram 列挙は
 *    長い鎖の接頭辞も同じ本数で出すので、極大の印が無いと読む順が壊れる）
 * 2. **実資産の答え合わせ** — 列挙器（`enumerateUnfusedWindows`）が背負う 2 つの性質を、
 *    実配布資産で押さえる。どちらも「数えそこない」と「数えすぎ」の対:
 *    - **融合を切った計画**に掛けると、現行ルールが実資産で掴んでいる鎖が候補として
 *      そのまま出る（op 名列 × 本数が既知のヒット数と一致する）。出なければ取りこぼし
 *    - **現行計画**に掛けると、既に掴めている鎖は 1 本も出ない（二重計上しない）
 *
 * 期待値の出典は packages/runtime/tests/assets_fusion_counts_test.ts（同じ資産・同じ読み方）。
 * 列挙器そのものの単体（合成 IR の窓の作り方）は `fusion.ts` の隣の
 * packages/runtime/tests/runtime_fusion_hints_test.ts にある — ここは**道具側の面**
 * （資産の発見・束縛・集計）を見る（置き場の規則は ADR 0008 追記 2026-09-03）。
 *
 * 資産は git 追跡外なので、無い取得元は**明示 SKIP** する（ADR 0005 — テストを消して無音で
 * 緑にしない）。
 */

import { assertEquals } from "@std/assert";
import { parseIrGraph } from "../../packages/runtime/src/format/ir.ts";
import { bindGraphSymbols, discoverGraphs, readIrGraph } from "./assets.ts";
import { aggregate, enumerateGraph } from "./enumerate.ts";

/**
 * 素の 3 ノード鎖（`neg → add → mul`）。どのルールの綴りにも当たらないので、計画は必ず素の
 * 列になる（= 集計の入力として素直な形）。
 */
const CHAIN_GRAPH = parseIrGraph(JSON.stringify({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "add", "mul"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: [4, 8] },
    { name: "w", dtype: "f32", shape: [4, 8] },
  ],
  outputs: ["c"],
  initializers: {},
  values: {
    a: { dtype: "f32", shape: [4, 8] },
    b: { dtype: "f32", shape: [4, 8] },
    c: { dtype: "f32", shape: [4, 8] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
    { op: "add", ins: ["a", "w"], outs: ["b"], attrs: {} },
    { op: "mul", ins: ["b", "w"], outs: ["c"], attrs: {} },
  ],
}));

Deno.test("合成 IR: 候補の集計は op 名列ごとに数え、同じ先頭からの最長窓を極大として印す", () => {
  const bound = bindGraphSymbols(CHAIN_GRAPH, {}, undefined);
  const windows = enumerateGraph(CHAIN_GRAPH, bound, { maxWindow: 3, fused: false }).windows;
  const rows = aggregate(windows);
  assertEquals(
    rows.map((row) => [row.ops.join(","), row.count, row.maximal, row.windowSizes]),
    [
      // 同数なら鎖の長い順。`neg,add` は同じ先頭（node 0）に長さ 3 の窓があるので極大でない。
      ["neg,add,mul", 1, 1, [3]],
      ["add,mul", 1, 1, [2]],
      ["neg,add", 1, 0, [2]],
    ],
  );
  // `example` は鎖の最終出力名を持つ（IR で鎖を引く手掛かりはこちら — node 添字はステップ順で、
  // 融合ありの計画では IR のノード添字と一致しない）。
  assertEquals(rows[0].example.outputName, "c");
});

// ------------------------------------------------------- 実資産の答え合わせ

/**
 * 資産 1 件ぶんの門。`expected` は**融合を切った計画**で出るべき候補（op 名列 → 本数）、
 * `fusedExpected` は**現行計画**で残る本数（掴めている鎖はここで 0 になる）。
 */
type AssetCase = {
  readonly source: string;
  readonly graph: string;
  readonly binds: Readonly<Record<string, number>>;
  readonly expected: Readonly<Record<string, number>>;
  readonly fusedExpected: Readonly<Record<string, number>>;
};

const SILU = "sigmoid,mul";
/** RoPE の 7 ノード窓（`mul` 先行形と、後置形 = 実際のノード順が違う 2 綴り）。 */
const ROPE_DIRECT_FIRST = "mul,slice,slice,neg,cat,mul,add";
const ROPE_DIRECT_LAST = "slice,slice,neg,cat,mul,mul,add";
/** adaLN の鎖（窓内 passthrough の reshape を除いた並び）。 */
const ADALN = "layer_norm,reshape,reshape,add,mul,add";
const ROW_BLOCK_ATTENTION = "bmm,reshape,add,safe_softmax,expand,reshape,expand,reshape,bmm";
/** upsample2x の 6 ノード鎖（`fusion.ts` の UPSAMPLE2X_RULE と同じ綴り）。 */
const UPSAMPLE2X = "reshape,expand,reshape,reshape,expand,reshape";

const ASSET_CASES: readonly AssetCase[] = [
  {
    source: "models/karume-anima",
    graph: "anima-turbo-v1.1/transformer",
    binds: { S: 4096 },
    expected: { [ROPE_DIRECT_LAST]: 56, [ADALN]: 85, [SILU]: 2 },
    fusedExpected: { [ROPE_DIRECT_LAST]: 0, [ADALN]: 0, [SILU]: 0 },
  },
  {
    source: "models/karume-anima",
    graph: "anima-turbo-v1.1/text_encoder",
    binds: { T: 64 },
    // rope 56 のうち 1 本は窓幅 8（cos / sin 表の `sym_prefix_slice` を窓内 passthrough として
    // 跨ぐ形）。窓内 passthrough を切り出せないと 55 になる。
    expected: { [ROPE_DIRECT_FIRST]: 56, [SILU]: 28 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0, [SILU]: 0 },
  },
  {
    source: "models/karume-anima",
    graph: "anima-turbo-v1.1/vae_decoder",
    binds: {},
    // upsample2x は `reshape` / `expand` だけで綴られる唯一のルール。別名化ノードを列挙から
    // 落とす変更が入ると、ここだけが先に割れる。
    expected: { [SILU]: 29, [UPSAMPLE2X]: 3 },
    fusedExpected: { [SILU]: 0, [UPSAMPLE2X]: 0 },
  },
  {
    source: "models/karume-irodori-v4-small",
    graph: "v4-small/dit",
    binds: { S: 750 },
    // silu 29 = 掴めている 17 + ゲート 12（`mul(v, sigmoid(u))` — 自分自身に掛からないので
    // SILU_RULE の受理集合の外）。op 名列の n-gram はルールの受理集合より広い。
    expected: { [SILU]: 29, [ROW_BLOCK_ATTENTION]: 12 },
    fusedExpected: { [SILU]: 12, [ROW_BLOCK_ATTENTION]: 0 },
  },
  {
    source: "models/karume-irodori-v4-small",
    graph: "v4-small/backbone",
    binds: { T: 256 },
    expected: { [ROPE_DIRECT_FIRST]: 50 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0 },
  },
  {
    source: "outputs/series/embeddinggemma-300m",
    graph: "embeddinggemma-300m",
    binds: { T: 318 },
    expected: { [ROPE_DIRECT_FIRST]: 48 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0 },
  },
  {
    source: "outputs/series/minicpm5-1b-decode",
    graph: "minicpm5-1b-decode",
    binds: { M: 1, C: 640 },
    expected: { [ROPE_DIRECT_FIRST]: 48, [SILU]: 24 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0, [SILU]: 0 },
  },
  {
    source: "outputs/series/gemma4-e2b-decode",
    graph: "gemma4-e2b-decode",
    binds: { M: 1, C: 640 },
    // **既知の穴**: 綴りは 50 箇所とも並ぶのに計画は 15 本しか掴まない（機序は未特定 —
    // docs/research/2026-08-30-gemma4-decode-wallclock.md §4）。残る 35 本が候補に出る。
    expected: { [ROPE_DIRECT_FIRST]: 50 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 35 },
  },
];

const REPO = new URL("../../", import.meta.url);

/**
 * 資産の有無。
 * MUST: NotFound 以外は伝播させる — 全 I/O エラーを「未生成」に丸めると、資産ルートの
 * マウント異常が SKIP に化けて、実行されていない検証が静かに緑になる。
 */
const exists = async (url: URL): Promise<boolean> => {
  try {
    await Deno.stat(url);
    return true;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/**
 * 取得元ごとの有無。SKIP は**取得元 1 件ずつ**で、1 件も無い環境だけテストごと `ignore` する。
 * 全 AND（1 件でも欠けたら全部 SKIP）にすると、系列出力を 1 本消しただけで配布ミラーの門まで
 * 静かに落ちる。
 */
const AVAILABLE = new Map<string, boolean>();
for (const source of new Set(ASSET_CASES.map((entry) => entry.source))) {
  const present = await exists(new URL(source, REPO));
  AVAILABLE.set(source, present);
  if (!present) {
    console.warn(
      `[karume] ${source} が無いため、この資産の融合候補の答え合わせを SKIP する` +
        "（列挙器の取りこぼしは実資産でしか検出できない）",
    );
  }
}
const ANY_ASSET_AVAILABLE = [...AVAILABLE.values()].some((present) => present);

/** 資産 1 件を計画して、`keys` の op 名列ごとの本数を引く。 */
const assetCounts = async (
  entry: AssetCase,
  fused: boolean,
  keys: readonly string[],
): Promise<Record<string, number>> => {
  const root = new URL(`${entry.source}/`, REPO);
  const sources = await discoverGraphs(root);
  const found = sources.find((source) => source.name === entry.graph);
  if (found === undefined) {
    throw new Error(
      `${entry.source} に '${entry.graph}' が無い（見えたのは ${
        sources.map((source) => source.name).join(" / ")
      }）`,
    );
  }
  const graph = await readIrGraph(found.url);
  const bound = bindGraphSymbols(graph, entry.binds, undefined);
  const rows = aggregate(enumerateGraph(graph, bound, { maxWindow: 9, fused }).windows);
  const counts: Record<string, number> = Object.fromEntries(keys.map((key) => [key, 0]));
  for (const row of rows) {
    const key = row.ops.join(",");
    if (Object.hasOwn(counts, key)) counts[key] = row.count;
  }
  return counts;
};

Deno.test({
  name: "実資産: 融合を切った計画に掛けると既知のヒット数が候補として再現する",
  ignore: !ANY_ASSET_AVAILABLE,
  fn: async () => {
    for (const entry of ASSET_CASES) {
      if (AVAILABLE.get(entry.source) !== true) continue;
      const keys = Object.keys(entry.expected);
      assertEquals(await assetCounts(entry, false, keys), entry.expected, entry.graph);
    }
  },
});

Deno.test({
  name: "実資産: 現行計画では掴めている鎖が候補に出ない（二重計上の検出）",
  ignore: !ANY_ASSET_AVAILABLE,
  fn: async () => {
    for (const entry of ASSET_CASES) {
      if (AVAILABLE.get(entry.source) !== true) continue;
      const keys = Object.keys(entry.fusedExpected);
      assertEquals(await assetCounts(entry, true, keys), entry.fusedExpected, entry.graph);
    }
  },
});
