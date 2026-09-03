/**
 * 静的 census の単体検証（GPU 不要）。
 *
 * 見る 3 点:
 *
 * 1. **行の形** — 合成 IR 1 本で、数値化された shape / dtype / 初期化子の格納 / 融合の帰属 /
 *    隣接（producers / consumers）が揃うこと
 * 2. **束縛の fail loudly** — 記号を渡さないまま census を採ろうとしたら落ちること
 *    （記号のまま出た表は census 加重として使えないので、黙って通すのが一番害が大きい）
 * 3. **実資産の合格線** — gemma4 配布ミラーの静的 census が
 *    `linear 277 / rms_norm 242 / attention 35 / state_append 30` を出すこと。linear の内訳
 *    i4g32 276 + i8 1 は 2026-08-30 の decode 実測 dispatch（wi4g32 276 + wi8 1）と同じ数字で、
 *    「静的 census が実行 1 回の dispatch 数を意味する」ことの実測との突合点
 */

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { parseIrGraph } from "../../packages/runtime/src/format/ir.ts";
import { readIrGraph, resolveAsset } from "./assets.ts";
import { censusComponent, summarizeScenario } from "./census.ts";
import { assertBindingKeys, defaultScenarios, parseScenario } from "./scenario.ts";

const TARGET = { component: "unit", componentDtype: "i4", graphShard: new URL("file:///none") };
const IDENTITY = { family: "unit", model: "unit", quant: "i4" };

/**
 * 合成 IR: `linear(x, w, b) → sigmoid → mul` の 3 ノード。後ろ 2 本は silu ルールが畳むので、
 * 「畳まれたノードにルール名が付く」ことと「畳まれても census に残る」ことを同時に見られる。
 */
const UNIT_GRAPH = parseIrGraph(JSON.stringify({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear", "sigmoid", "mul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: [1, "T", 32] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "w.q", storage: { dtype: "i4", scale: "w.scale", group_size: 16 } },
    b: { tensor: "b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [4, 32] },
    b: { dtype: "f32", shape: [4] },
    h: { dtype: "f32", shape: [1, "T", 4] },
    s: { dtype: "f32", shape: [1, "T", 4] },
    y: { dtype: "f32", shape: [1, "T", 4] },
  },
  nodes: [
    { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
    { op: "sigmoid", ins: ["h"], outs: ["s"], attrs: {} },
    { op: "mul", ins: ["h", "s"], outs: ["y"], attrs: {} },
  ],
}));

const unitScenario = (bindings: Readonly<Record<string, number>>) => ({
  name: "unit",
  bindings,
  source: "cli" as const,
  provenance: "test",
});

Deno.test("合成 IR: 記号は束縛で数値化され、格納と隣接と融合の帰属が行に載る", () => {
  const census = censusComponent(UNIT_GRAPH, TARGET, IDENTITY, unitScenario({ T: 3 }));
  assertEquals(census.rows.length, 3);
  assertEquals(census.rows.map((row) => row.op), ["linear", "sigmoid", "mul"]);

  const [linear, sigmoid, mul] = census.rows;
  // 記号 T=3 が入力・重み・出力の全 shape に効いている（記号のままの欄が 1 つも残らない）。
  assertEquals(linear.in_shapes, [[1, 3, 32], [4, 32], [4]]);
  assertEquals(linear.out_shapes, [[1, 3, 4]]);
  assertEquals(linear.in_dtypes, ["f32", "f32", "f32"]);
  assertEquals(linear.out_dtypes, ["f32"]);
  assertEquals(linear.bindings, { T: 3 });
  assertEquals(linear.binding_source, "cli");

  // 格納は `ins` と同順（初期化子でない入力は null）。i4 は group 長と scale キーまで載る。
  assertEquals(linear.storage, [
    null,
    { tensor: "w.q", dtype: "i4", group_size: 16, scale: "w.scale" },
    { tensor: "b", dtype: "f32" },
  ]);

  // 隣接（③ Fusion 半自動発見の入力になる欄）。
  assertEquals(linear.producers, []);
  assertEquals(linear.consumers, [1, 2]);
  assertEquals(sigmoid.producers, [0]);
  assertEquals(mul.producers, [0, 1]);
  assertEquals(mul.consumers, []);

  // silu が sigmoid + mul を畳む。畳まれたノードも census には残り、ルール名で名指される。
  assertEquals(linear.fused_by, null);
  assertEquals(sigmoid.fused_by, "silu");
  assertEquals(mul.fused_by, "silu");
  // ヒット数 1（= 融合ステップ 1 本）と、畳まれたノード 2 本は別物。
  assertEquals(census.fusionHits.silu, 1);
  assertEquals(census.unusedBindings, []);
  assert(census.rows.every((row) => !row.aliases_input));
});

Deno.test("合成 IR: 束縛していない記号は fail loudly（記号のまま census を出さない）", () => {
  const error = assertThrows(
    () => censusComponent(UNIT_GRAPH, TARGET, IDENTITY, unitScenario({})),
    Error,
  );
  assertStringIncludes(error.message, "記号 T");
  assertStringIncludes(error.message, "未束縛");
});

Deno.test("合成 IR: そのコンポーネントに無い記号を名指した束縛は fail loudly", () => {
  const error = assertThrows(
    () => censusComponent(UNIT_GRAPH, TARGET, IDENTITY, unitScenario({ T: 3, "unit.S": 4 })),
    Error,
  );
  assertStringIncludes(error.message, "記号 'S' が無い");
});

Deno.test("census 加重は同一の (op, shape, dtype, 格納, 融合) を 1 行に畳む", () => {
  const census = censusComponent(UNIT_GRAPH, TARGET, IDENTITY, unitScenario({ T: 3 }));
  const summary = summarizeScenario(
    unitScenario({ T: 3 }),
    census.rows,
    census.unusedBindings,
    census.fusionHits,
  );
  assertEquals(summary.node_count, 3);
  assertEquals(summary.by_op.linear, { nodes: 1, out_elements: 12 });
  // 初期化子の格納シグネチャ（i4 は group 長込み）。素の elementwise は初期化子を持たない。
  assertEquals(summary.by_storage, { "f32+i4g16": 1, none: 2 });
  assertEquals(summary.by_fusion, {
    absorbed: { silu: 2 },
    hits: { silu: 1 },
    plain: 1,
    aliased: 0,
  });
  assertEquals(summary.weights.length, 3);
  assertEquals(summary.weights.every((weight) => weight.count === 1), true);
});

Deno.test("シナリオ: --scenario の綴りと、実在しない component 名の拒否", () => {
  const scenario = parseScenario("prefill=M:768,model.C:4096");
  assertEquals(scenario.name, "prefill");
  assertEquals(scenario.bindings, { M: 768, "model.C": 4096 });
  assertEquals(scenario.source, "cli");
  assertBindingKeys(scenario, ["model"]);
  assertThrows(() => assertBindingKeys(scenario, ["other"]), Error, "component 'model'");
  assertThrows(() => parseScenario("bad"), Error);
  assertThrows(() => parseScenario("bad=M:0"), Error);
});

/**
 * 実資産の合格線。資産（配布ミラー）は git 追跡外なので、無い環境は**明示 SKIP** する
 * （テストを消して無音で緑にしない — ADR 0005）。
 */
const GEMMA4_DIR = new URL("../../models/karume-gemma4-e2b/", import.meta.url);
const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};
const GEMMA4_AVAILABLE = exists(new URL("karume.json", GEMMA4_DIR));
if (!GEMMA4_AVAILABLE) {
  console.warn(
    `[karume] ${GEMMA4_DIR.pathname} に karume.json が無いため opbench の実資産 census を SKIP する`,
  );
}

Deno.test({
  name: "実資産 census: gemma4 は linear 277 / rms_norm 242 / attention 35 / state_append 30",
  ignore: !GEMMA4_AVAILABLE,
  fn: async () => {
    const asset = await resolveAsset(GEMMA4_DIR, undefined, undefined, undefined);
    assertEquals(asset.family, "gemma4");
    assertEquals(asset.components.map((target) => target.component), ["model"]);
    const [target] = asset.components;
    const graph = await readIrGraph(target.graphShard);
    const [decode] = defaultScenarios("gemma4");
    assertEquals(decode.name, "decode");
    const census = censusComponent(graph, target, asset, decode);
    const summary = summarizeScenario(
      decode,
      census.rows,
      census.unusedBindings,
      census.fusionHits,
    );
    assertEquals(summary.by_op.linear.nodes, 277);
    assertEquals(summary.by_op.rms_norm.nodes, 242);
    assertEquals(summary.by_op.attention.nodes, 35);
    assertEquals(summary.by_op.state_append.nodes, 30);
    // linear の格納の内訳が 2026-08-30 の decode 実測 dispatch（wi4g32 276 + lm_head の wi8 1）と
    // 一致する。ここが割れたら「静的 census = 実行 1 回」という前提の方が壊れている。
    const linears = summary.weights.filter((weight) => weight.op === "linear");
    const byStorage = new Map<string, number>();
    for (const weight of linears) {
      byStorage.set(weight.storage, (byStorage.get(weight.storage) ?? 0) + weight.count);
    }
    assertEquals(byStorage.get("f32+i4g32"), 276);
    // lm_head だけが i8 重み（+ f32 bias）。
    assertEquals(byStorage.get("f32+i8"), 1);
    assertEquals(linears.reduce((total, weight) => total + weight.count, 0), 277);
    // 融合ヒットは既設の実資産の門（assets_fusion_counts_test.ts）と同じ 15（M=1）。
    assertEquals(summary.by_fusion.hits, { rope: 15 });
  },
});
