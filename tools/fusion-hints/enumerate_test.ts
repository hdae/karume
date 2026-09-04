/**
 * fusion-hints の列挙面の検証（GPU 不要）。
 *
 * 見る 3 点:
 *
 * 1. **集計** — 合成 IR 1 本で、op 名列ごとの本数と「極大」の印が付くこと（n-gram 列挙は
 *    長い鎖の接頭辞も同じ本数で出すので、極大の印が無いと読む順が壊れる）
 * 2. **シナリオ束縛と CLI 語彙** — opbench と同じ語彙（家族の既定シナリオ・`<component>.SYM`
 *    修飾）で束縛が効くこと、未束縛と記号名の誤綴りは打ち切ること、廃止した `--bind` /
 *    `--default-symbol` と未知のオプション・重複したシナリオ名が理由つきで落ちること
 * 3. **実資産の答え合わせ** — 列挙器（`enumerateUnfusedWindows`）が背負う 2 つの性質を、
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

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { parseIrGraph } from "../../packages/runtime/src/format/ir.ts";
import { readIrGraph, resolveAsset } from "../_shared/assets.ts";
import { defaultScenarios, parseScenario } from "../_shared/scenario.ts";
import { bindGraphSymbols } from "./binding.ts";
import { aggregate, enumerateGraph, type GraphInput, reportScenario } from "./enumerate.ts";
import { parseArgs } from "./main.ts";

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
  const bound = bindGraphSymbols(CHAIN_GRAPH, {});
  const windows = enumerateGraph(CHAIN_GRAPH, bound, { maxWindow: 3, fused: false }).windows;
  const rows = aggregate(windows);
  assertEquals(
    rows.map((row) => [row.ops.join(","), row.count, row.maximal, row.window_sizes]),
    [
      // 同数なら鎖の長い順。`neg,add` は同じ先頭（node 0）に長さ 3 の窓があるので極大でない。
      ["neg,add,mul", 1, 1, [3]],
      ["add,mul", 1, 1, [2]],
      ["neg,add", 1, 0, [2]],
    ],
  );
  // `example` は鎖の最終出力名を持つ（IR で鎖を引く手掛かりはこちら — node 添字はステップ順で、
  // 融合ありの計画では IR のノード添字と一致しない）。
  assertEquals(rows[0].example.output_name, "c");
});

// ------------------------------------------------------- シナリオ束縛と CLI 語彙

/** 記号 `T` を 1 本持つ合成 IR（`neg → add`）。同じ綴りを 2 コンポーネントに置いて使う。 */
const SYMBOLIC_GRAPH = parseIrGraph(JSON.stringify({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "add"] },
  symbols: ["T"],
  inputs: [
    { name: "x", dtype: "f32", shape: [1, "T", 8] },
    { name: "w", dtype: "f32", shape: [1, "T", 8] },
  ],
  outputs: ["b"],
  initializers: {},
  values: {
    a: { dtype: "f32", shape: [1, "T", 8] },
    b: { dtype: "f32", shape: [1, "T", 8] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
    { op: "add", ins: ["a", "w"], outs: ["b"], attrs: {} },
  ],
}));

/** 同じ記号名が別の意味を持つ 2 コンポーネント（irodori の `T` と同じ形）。 */
const TWO_COMPONENTS: readonly GraphInput[] = [
  {
    component: "backbone",
    graph: "unit/backbone",
    path: "backbone/model.safetensors",
    ir: SYMBOLIC_GRAPH,
  },
  {
    component: "codec_encoder",
    graph: "unit/codec_encoder",
    path: "codec_encoder/model.safetensors",
    ir: SYMBOLIC_GRAPH,
  },
];

Deno.test("シナリオ: <component>.SYM 修飾は component ごとに別の値で効く", () => {
  const scenario = parseScenario("mixed=T:7,codec_encoder.T:99");
  const report = reportScenario(scenario, TWO_COMPONENTS, { maxWindow: 3, fused: true });
  assertEquals(report.scenario, "mixed");
  assertEquals(report.binding_source, "cli");
  // 家族共通の `T:7` は backbone に効き、修飾された `codec_encoder.T:99` が同名を上書きする。
  assertEquals(report.graphs.map((graph) => [graph.component, graph.symbols]), [
    ["backbone", { T: 7 }],
    ["codec_encoder", { T: 99 }],
  ]);
  // 束縛はグラフの shape に届いている（候補の窓は同じ 1 本 = 束縛では鎖の綴りは変わらない）。
  assertEquals(report.graphs.map((graph) => graph.rows.map((row) => row.ops.join(","))), [
    ["neg,add"],
    ["neg,add"],
  ]);
});

Deno.test("シナリオ: 未束縛の記号は skip ではなく打ち切り（既定値で計画を進めない）", () => {
  const scenario = parseScenario("partial=backbone.T:7");
  const error = assertThrows(
    () => reportScenario(scenario, TWO_COMPONENTS, { maxWindow: 3, fused: true }),
    Error,
  );
  assertStringIncludes(error.message, "codec_encoder: 記号 T が未束縛");
  assertStringIncludes(error.message, "シナリオ 'partial'");
  assertStringIncludes(error.message, "--scenario");
});

/**
 * 修飾キーの**記号側**の誤綴り。component 名（`assertBindingKeys`）だけを見ていた頃は、
 * `backbone.S` のような綴りが「その束縛は無かった」として静かに通り、既定側の値で解かれた
 * 候補表が出ていた。判定は tools/_shared/scenario.ts の 1 本なので、opbench の census も
 * 同じ文言で落ちる（対の門は tools/opbench/census_test.ts）。
 */
Deno.test("シナリオ: component が宣言しない記号を名指した修飾キーは fail loudly", () => {
  const scenario = parseScenario("typo=T:7,backbone.S:4");
  const error = assertThrows(
    () => reportScenario(scenario, TWO_COMPONENTS, { maxWindow: 3, fused: true }),
    Error,
  );
  assertStringIncludes(error.message, "シナリオ 'typo' の束縛 'backbone.S'");
  assertStringIncludes(error.message, "backbone に記号 'S' が無い");
  assertStringIncludes(error.message, "既知: T");
});

Deno.test("CLI: 廃止した --bind / --default-symbol は理由つきで落ちる（互換シムを置かない）", () => {
  for (
    const argv of [
      ["enumerate", "--source", "models/karume-gemma4-e2b", "--bind", "M=1"],
      ["enumerate", "--source", "models/karume-gemma4-e2b", "--default-symbol", "64"],
    ]
  ) {
    const error = assertThrows(() => parseArgs(argv), Error);
    assertStringIncludes(error.message, "廃止した");
    assertStringIncludes(error.message, "--scenario");
  }
  // 新しい綴りは繰り返して受け、省略時は空（= 家族の既定シナリオ）。
  assertEquals(
    parseArgs([
      "enumerate",
      "--source",
      "models/karume-gemma4-e2b",
      "--scenario",
      "decode=M:1,C:640",
      "--scenario",
      "prefill=M:768,C:640",
    ]).scenarios.map((scenario) => [scenario.name, scenario.bindings]),
    [["decode", { M: 1, C: 640 }], ["prefill", { M: 768, C: 640 }]],
  );
  assertEquals(parseArgs(["enumerate", "--source", "models/karume-gemma4-e2b"]).scenarios, []);
});

Deno.test("CLI: 未知のオプションは USAGE つきで落ちる（黙って既定シナリオへ落ちない）", () => {
  // `--scenarios`（複数形）は 1 文字違いで、捨てられると家族の既定シナリオで走ってしまう。
  const error = assertThrows(
    () => parseArgs(["enumerate", "--source", "models/karume-gemma4-e2b", "--scenarios", "x=M:1"]),
    Error,
  );
  assertStringIncludes(error.message, "未知のオプション '--scenarios'");
  assertStringIncludes(error.message, "使い方:");
});

Deno.test("CLI: --scenario の名前が重複したら落ちる（census と名前で突合するため）", () => {
  const error = assertThrows(
    () =>
      parseArgs([
        "enumerate",
        "--source",
        "models/karume-gemma4-e2b",
        "--scenario",
        "a=M:1",
        "--scenario",
        "a=M:2",
      ]),
    Error,
  );
  assertStringIncludes(error.message, "'a' が重複");
});

// ------------------------------------------------------- 実資産の答え合わせ

/**
 * 資産 1 件ぶんの門。`expected` は**融合を切った計画**で出るべき候補（op 名列 → 本数）、
 * `fusedExpected` は**現行計画**で残る本数（掴めている鎖はここで 0 になる）。
 */
type AssetCase = {
  readonly source: string;
  /** 資産解決が返す component 名（配布形は manifest の綴り・系列出力は根直下なら `model`）。 */
  readonly component: string;
  /** 診断に出す見出し。 */
  readonly graph: string;
  /**
   * 系列ディレクトリ名から家族名を推せない資産で必須（家族は既定シナリオの引き当てにしか
   * 使わず、この門は束縛を明示するので値は表示上の意味しか持たない）。
   */
  readonly family?: string;
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
    component: "transformer",
    graph: "anima-turbo-v1.1/transformer",
    binds: { S: 4096 },
    expected: { [ROPE_DIRECT_LAST]: 56, [ADALN]: 85, [SILU]: 2 },
    fusedExpected: { [ROPE_DIRECT_LAST]: 0, [ADALN]: 0, [SILU]: 0 },
  },
  {
    source: "models/karume-anima",
    component: "text_encoder",
    graph: "anima-turbo-v1.1/text_encoder",
    binds: { T: 64 },
    // rope 56 のうち 1 本は窓幅 8（cos / sin 表の `sym_prefix_slice` を窓内 passthrough として
    // 跨ぐ形）。窓内 passthrough を切り出せないと 55 になる。
    expected: { [ROPE_DIRECT_FIRST]: 56, [SILU]: 28 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0, [SILU]: 0 },
  },
  {
    source: "models/karume-anima",
    component: "vae_decoder",
    graph: "anima-turbo-v1.1/vae_decoder",
    binds: {},
    // upsample2x は `reshape` / `expand` だけで綴られる唯一のルール。別名化ノードを列挙から
    // 落とす変更が入ると、ここだけが先に割れる。
    expected: { [SILU]: 29, [UPSAMPLE2X]: 3 },
    fusedExpected: { [SILU]: 0, [UPSAMPLE2X]: 0 },
  },
  {
    source: "models/karume-irodori-v4-small",
    component: "dit",
    graph: "v4-small/dit",
    binds: { S: 750 },
    // silu 29 = 掴めている 17 + ゲート 12（`mul(v, sigmoid(u))` — 自分自身に掛からないので
    // SILU_RULE の受理集合の外）。op 名列の n-gram はルールの受理集合より広い。
    expected: { [SILU]: 29, [ROW_BLOCK_ATTENTION]: 12 },
    fusedExpected: { [SILU]: 12, [ROW_BLOCK_ATTENTION]: 0 },
  },
  {
    source: "models/karume-irodori-v4-small",
    component: "backbone",
    graph: "v4-small/backbone",
    binds: { T: 256 },
    expected: { [ROPE_DIRECT_FIRST]: 50 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0 },
  },
  {
    source: "outputs/series/embeddinggemma-300m",
    component: "model",
    graph: "embeddinggemma-300m/model",
    // ディレクトリ名から家族名を推せない（`--family` が要る資産の実例）。
    family: "gemma4",
    binds: { T: 318 },
    expected: { [ROPE_DIRECT_FIRST]: 48 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0 },
  },
  {
    source: "outputs/series/minicpm5-1b-decode",
    component: "model",
    graph: "minicpm5-1b-decode/model",
    family: "minicpm5",
    binds: { M: 1, C: 640 },
    expected: { [ROPE_DIRECT_FIRST]: 48, [SILU]: 24 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0, [SILU]: 0 },
  },
  {
    source: "outputs/series/gemma4-e2b-decode",
    component: "model",
    graph: "gemma4-e2b-decode/model",
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
  // 資産の引き当ては census と同じ 1 本（`resolveAsset`）— 格納 dtype は quant 表に従う。
  const asset = await resolveAsset(root, undefined, undefined, entry.family);
  const found = asset.components.find((target) => target.component === entry.component);
  if (found === undefined) {
    throw new Error(
      `${entry.source} に component '${entry.component}' が無い（見えたのは ${
        asset.components.map((target) => target.component).join(" / ")
      }）`,
    );
  }
  // 期待値は `<model>/<component>` のグラフに対するもの — defaultModel が別変種へ動けばここで気付く。
  assertEquals(`${asset.model}/${found.component}`, entry.graph, `${entry.source} の引き当て`);
  const graph = await readIrGraph(found.graphShard);
  const bound = bindGraphSymbols(graph, entry.binds);
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

/**
 * シナリオを省いた既定経路（CLI が `--scenario` 無しで通る道）を配布ミラー 1 本で通す。
 * 期待値は opbench の census 側の門（tools/opbench/census_test.ts）と同じ資産・同じシナリオ
 * なので、両道具が同じ束縛で同じグラフを見ていることがここで揃う。
 */
const GEMMA4_DIR = new URL("../../models/karume-gemma4-e2b/", import.meta.url);
const GEMMA4_AVAILABLE = await exists(new URL("karume.json", GEMMA4_DIR));
if (!GEMMA4_AVAILABLE) {
  console.warn(
    `[karume] ${GEMMA4_DIR.pathname} に karume.json が無いため既定シナリオの実走を SKIP する`,
  );
}

Deno.test({
  name: "実資産: --scenario 省略時は家族の既定シナリオで走る（gemma4 = decode / prefill）",
  ignore: !GEMMA4_AVAILABLE,
  fn: async () => {
    const asset = await resolveAsset(GEMMA4_DIR, undefined, undefined, undefined);
    assertEquals(asset.family, "gemma4");
    const graphs: readonly GraphInput[] = await Promise.all(
      asset.components.map(async (target): Promise<GraphInput> => ({
        component: target.component,
        graph: `${asset.model}/${target.component}`,
        path: target.graphShard.pathname,
        ir: await readIrGraph(target.graphShard),
      })),
    );
    const scenarios = defaultScenarios(asset.family);
    assertEquals(scenarios.map((scenario) => scenario.name), ["decode", "prefill"]);
    const [decode] = scenarios.map((scenario) =>
      reportScenario(scenario, graphs, { maxWindow: 9, fused: true })
    );
    assertEquals(decode.scenario, "decode");
    assertEquals(decode.binding_source, "default");
    assertEquals(decode.graphs.map((graph) => graph.graph), ["e2b/model"]);
    // 束縛は既定表の decode（M=1・C=4096）がそのまま入る。
    assertEquals(decode.graphs[0].symbols, { M: 1, C: 4096 });
    // ヒット数は census 側の門と同じ 15（M=1 の rope）。
    assertEquals(decode.graphs[0].counts.rope, 15);
  },
});
