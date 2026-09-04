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
import { readIrGraph, resolveAsset } from "../_shared/assets.ts";
import { buildCensusSummary, censusComponent, summarizeScenario } from "./census.ts";
import { assertBindingKeys, defaultScenarios, parseScenario } from "../_shared/scenario.ts";

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

/**
 * 合成 IR: 同じ入力から出る `permute` 2 本（`dims` だけ違う）と `slice` 2 本（属性の値は同じで
 * **キーの書き順だけ**違う）。どれも出力なので融合の対象にならず、加重が attrs だけで割れる /
 * 畳まれることを他の要因抜きで見られる。
 */
const ATTRS_GRAPH = parseIrGraph(JSON.stringify({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["permute", "slice"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 4, 4, 4] }],
  outputs: ["p1", "p2", "s1", "s2"],
  initializers: {},
  values: {
    p1: { dtype: "f32", shape: [1, 4, 4, 4] },
    p2: { dtype: "f32", shape: [1, 4, 4, 4] },
    s1: { dtype: "f32", shape: [1, 2, 4, 4] },
    s2: { dtype: "f32", shape: [1, 2, 4, 4] },
  },
  nodes: [
    { op: "permute", ins: ["x"], outs: ["p1"], attrs: { dims: [0, 2, 1, 3] } },
    { op: "permute", ins: ["x"], outs: ["p2"], attrs: { dims: [0, 1, 3, 2] } },
    { op: "slice", ins: ["x"], outs: ["s1"], attrs: { dim: 1, start: 0, end: 2 } },
    { op: "slice", ins: ["x"], outs: ["s2"], attrs: { start: 0, end: 2, dim: 1 } },
  ],
}));

/**
 * 合成 IR: 格納の**集合**は同じで**スロット割り当てだけ**違う 2 ノード。どちらも
 * `add(i4g16, f32)` / `add(f32, i4g16)` で、shape も dtype も attrs も融合の帰属も同一なので、
 * 加重キーが格納をスロット同順の列で持っているかだけを分離して見られる（集合シグネチャへ
 * 戻すと 2 行が 1 行 count 2 に畳まれる）。
 *
 * NOTE: 加重キーの doc が例に挙げる `linear` の `[x, W, bias]` はここでは組めない — rank 1 の
 * bias は行長が常に 1 で、i4 の group 長（16 以上）で割り切れない（ADR 0069 決定 2）。
 */
const SLOT_GRAPH = parseIrGraph(JSON.stringify({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [],
  outputs: ["o1", "o2"],
  initializers: {
    a4: { tensor: "a4.q", storage: { dtype: "i4", scale: "a4.scale", group_size: 16 } },
    af: { tensor: "af", storage: { dtype: "f32" } },
    b4: { tensor: "b4.q", storage: { dtype: "i4", scale: "b4.scale", group_size: 16 } },
    bf: { tensor: "bf", storage: { dtype: "f32" } },
  },
  values: {
    a4: { dtype: "f32", shape: [4, 32] },
    af: { dtype: "f32", shape: [4, 32] },
    b4: { dtype: "f32", shape: [4, 32] },
    bf: { dtype: "f32", shape: [4, 32] },
    o1: { dtype: "f32", shape: [4, 32] },
    o2: { dtype: "f32", shape: [4, 32] },
  },
  nodes: [
    { op: "add", ins: ["a4", "af"], outs: ["o1"], attrs: {} },
    { op: "add", ins: ["bf", "b4"], outs: ["o2"], attrs: {} },
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

Deno.test("census 加重は同一の (op, shape, dtype, attrs, 格納, 融合) を 1 行に畳む", () => {
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

  // 加重行の格納は census 行と同じ**スロット同順の列**（集合の署名は別欄）。linear の
  // `[x, W, bias]` で W だけが i4g32 という対応が、加重表だけを見て読める。
  const [linear] = summary.weights.filter((weight) => weight.op === "linear");
  assertEquals(linear.storage.length, linear.in_shapes.length);
  assertEquals(linear.storage, [null, { dtype: "i4", group_size: 16 }, { dtype: "f32" }]);
  assertEquals(linear.storage_signature, "f32+i4g16");
});

Deno.test("census 加重: attrs が違えば別行・attrs のキー順だけの違いは 1 行", () => {
  const census = censusComponent(ATTRS_GRAPH, TARGET, IDENTITY, unitScenario({}));
  const summary = summarizeScenario(
    unitScenario({}),
    census.rows,
    census.unusedBindings,
    census.fusionHits,
  );
  assertEquals(summary.node_count, 4);

  // permute 2 本は shape も dtype も格納も融合も同じ（= attrs 以外の欄では区別が付かない）。
  const permutes = summary.weights.filter((weight) => weight.op === "permute");
  assertEquals(permutes.length, 2);
  assertEquals(permutes.map((weight) => weight.count), [1, 1]);
  assertEquals(new Set(permutes.map((weight) => JSON.stringify(weight.out_shapes))).size, 1);
  assertEquals(
    new Set(permutes.map((weight) => weight.storage_signature)),
    new Set(["none"]),
  );
  // 割れているのは `dims` だけ — [0,2,1,3] と [0,1,3,2] はメモリアクセス形が別物。
  assertEquals(
    permutes.map((weight) => weight.attrs.dims),
    [[0, 2, 1, 3], [0, 1, 3, 2]],
  );

  // slice 2 本は attrs の値が同じでキーの書き順だけ違う → 1 行に畳まれる。
  const slices = summary.weights.filter((weight) => weight.op === "slice");
  assertEquals(slices.length, 1);
  assertEquals(slices[0].count, 2);
});

Deno.test("census 加重: 格納の集合が同じでもスロット割り当てが違えば別行", () => {
  const census = censusComponent(SLOT_GRAPH, TARGET, IDENTITY, unitScenario({}));
  const summary = summarizeScenario(
    unitScenario({}),
    census.rows,
    census.unusedBindings,
    census.fusionHits,
  );
  assertEquals(summary.node_count, 2);
  // 集合シグネチャは 2 本とも同じ = `by_storage` では区別が付かない。
  assertEquals(summary.by_storage, { "f32+i4g16": 2 });
  // それでも加重は 2 行（各 1 本）— スロット列が違うので別のカーネルケースになる。
  assertEquals(summary.weights.length, 2);
  assertEquals(summary.weights.map((weight) => weight.count), [1, 1]);
  assertEquals(summary.weights.map((weight) => weight.storage), [
    [{ dtype: "i4", group_size: 16 }, { dtype: "f32" }],
    [{ dtype: "f32" }, { dtype: "i4", group_size: 16 }],
  ]);
});

/** 一時ディレクトリに置く配布形の manifest（資産解決は shard の実体を読まない）。 */
const DIST_MANIFEST = JSON.stringify({
  defaultModel: "m",
  models: {
    m: {
      pipeline: "gemma4/1",
      defaultQuant: "i8-a8",
      quants: {
        "i8-a8": { weights: { model: "i8" }, session: { linearCompute: "a8" } },
        // `session` の欄ごと無い quant（配布形の多数派）。
        i8: { weights: { model: "i8" } },
      },
      weights: { model: { i8: { shards: [{ path: "model/model.i8.safetensors" }] } } },
    },
  },
});

Deno.test("summary ヘッダの session: 配布形は宣言の写し・系列出力は null", async () => {
  const temp = await Deno.makeTempDir();
  try {
    const distRoot = new URL(`file://${temp}/dist/`);
    await Deno.mkdir(distRoot, { recursive: true });
    await Deno.writeTextFile(new URL("karume.json", distRoot), DIST_MANIFEST);

    const declared = await resolveAsset(distRoot, undefined, undefined, undefined);
    assertEquals(declared.session, { linearCompute: "a8" });
    // manifest 所有の綴りのまま（runtime の SessionOptions へ翻訳しない）。
    assertEquals(buildCensusSummary("dist", declared, []).session, { linearCompute: "a8" });

    // 欄ごと無い quant は「ノブを 1 つも指定しない」= 空の宣言（`null` ではない）。
    const silent = await resolveAsset(distRoot, undefined, "i8", undefined);
    assertEquals(buildCensusSummary("dist", silent, []).session, {});

    // 系列出力は manifest を持たない = 実行変種が宣言されていない（呼び手が与える）。
    const seriesRoot = new URL(`file://${temp}/gemma4-series/`);
    await Deno.mkdir(seriesRoot, { recursive: true });
    await Deno.writeTextFile(new URL("model.f32.safetensors", seriesRoot), "");
    const series = await resolveAsset(seriesRoot, undefined, undefined, undefined);
    assertEquals(series.session, undefined);
    assertEquals(buildCensusSummary("series", series, []).session, null);
  } finally {
    await Deno.remove(temp, { recursive: true });
  }
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
    // 既定 quant `i4` はノブを 1 つも宣言していない（実行変種は呼び手の既定のまま）。
    assertEquals(asset.session, {});
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
      const signature = weight.storage_signature;
      byStorage.set(signature, (byStorage.get(signature) ?? 0) + weight.count);
    }
    assertEquals(byStorage.get("f32+i4g32"), 276);
    // 276 本が畳まれて残る相異なる形は 12（attrs を加重キーに入れても linear は割れない）。
    assertEquals(linears.filter((weight) => weight.storage_signature === "f32+i4g32").length, 12);
    // lm_head だけが i8 重み（+ f32 bias）。
    assertEquals(byStorage.get("f32+i8"), 1);
    assertEquals(linears.reduce((total, weight) => total + weight.count, 0), 277);
    // 融合ヒットは既設の実資産の門（assets_fusion_counts_test.ts）と同じ 15（M=1）。
    assertEquals(summary.by_fusion.hits, { rope: 15 });
  },
});
