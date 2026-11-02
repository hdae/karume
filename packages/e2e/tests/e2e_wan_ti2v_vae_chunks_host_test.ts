// GPU-free checks of Wan2.2 TI2V-5B's two VAE decoder chunk graphs (the f16 containers of the mirror
// `models/karume-wan2.2/`, and the fixtures of the series `wan2.2-ti2v-5b-f16-dyn` in the test-asset root — ADR 0139
// decisions 1 / 2; ADR 0121 stage 4).
//
// 見るのは 3 つ:
//
// - **容器の宣言 → chunk 列の取り決め**: 2.1 と同じホストの関数（`src/wan/vae-chunks.ts` の `wanVaeChunkLayout`）が
//   2.2 の宣言から潜在 48 ch・タイル 16・出口 12 ch・cache の表（32 本・first は `time_conv` の 2 本を除く 30 本）・
//   フレームの形を導くこと。表は `helpers/wan-ti2v-vae.ts`（recipe の表と二重凍結）。
// - **runtime の読み口での IR の主張**: rank（値は rank 4 以下で、rank 5 は conv3d の第 2 入力の initializer だけ）と
//   op の集合（2.1 と同じ語彙 15 種の中）は recipe の `export_vae.assert_chunk_graph` と同じ主張で、それを別の読み手
//   （容器を開いた runtime の宣言）で数える。attention のノードが各グラフ 1 本であることは recipe の検査に無い追加の
//   主張（ADR 0121 決定 6 の mid の単一 head attention）。主張ごとに書き換えた宣言を 1 件ずつ拒ませて、検査が恒真で
//   ないことを示す。
//   NOTE: ノード数・conv3d の本数は固定しない（exporter の分解が無害に変わるたびに落ちる数で、構造の主張は上の 3 つが
//   持つ）。
// - **フィクスチャ**: メタ（タイル・chunk 数・役割・重みの丸め・参照の素性）と形が表どおり。
//
// The container checks need the mirror; the fixture check needs the three fixtures. Without them the cases SKIP
// explicitly, naming what is missing (ADR 0005). Fixtures present only in part are a FAIL, not a SKIP (the three are
// written together). No GPU is used (the real-GPU chunk-sequence check is `e2e_wan_ti2v_vae_tiles_test.ts`).
//
// NOTE: the containers' graph names are the manifest's weights keys (container-v1 §2.1).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { prepareContainer, type PreparedModel } from "@karume/runtime";
import { wanVaeChunkLayout, wanVaeFrameCount } from "../../models/src/wan/vae-chunks.ts";
import { readFixture } from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import { TEST_ASSETS_FETCH as FETCH } from "../../models/tests/helpers/wan-ti2v-dit.ts";
import {
  openWanTi2vComponent,
  WAN_TI2V_ASSEMBLE_COMMAND as ASSEMBLE,
  WAN_TI2V_DIST_ROOT as DIST_ROOT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import { mirrorPresent } from "../../models/tests/helpers/mirror.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import {
  ti2vVaeChunkFixtures,
  ti2vVaeFixtureUrl,
  WAN_TI2V_VAE_CACHE_BYTES as CACHE_BYTES,
  WAN_TI2V_VAE_CACHE_TABLE as CACHE_TABLE,
  WAN_TI2V_VAE_CASES as CASES,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_FIRST_CACHE_BYTES as FIRST_CACHE_BYTES,
  WAN_TI2V_VAE_GOLDEN_DIR as GOLDEN_DIR,
  WAN_TI2V_VAE_LATENT_CHANNELS as LATENT_CHANNELS,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_SAMPLE_CHANNELS as SAMPLE_CHANNELS,
  WAN_TI2V_VAE_SAMPLE_TILE as SAMPLE_TILE,
  WAN_TI2V_VAE_TILE as TILE,
  WAN_TI2V_VAE_TIME_CONV_CACHES as TIME_CONV_CACHES,
} from "../../models/tests/helpers/wan-ti2v-vae.ts";

type Graph = PreparedModel["graph"];

/**
 * chunk グラフの IR に現れてよい op（recipe の `export_vae.EXPECTED_OPS` — 2.1 と同じ 15 種）。SiLU は
 * `x · sigmoid(x)` に分解され、mid の attention は SDPA の保存（ADR 0023）。
 */
const CHUNK_GRAPH_OPS: ReadonlySet<string> = new Set([
  "add",
  "attention",
  "cat",
  "clamp_min",
  "conv2d",
  "conv3d",
  "div",
  "expand",
  "mul",
  "permute",
  "reshape",
  "sigmoid",
  "slice",
  "sqrt",
  "sum",
]);

/** 値の rank の上限（strided コピー族の上限 — ADR 0118 決定 2）。rank 5 は conv3d の重みだけに許す。 */
const MAX_VALUE_RANK = 4;

/** mid の単一 head attention のノードの本数（各グラフ — ADR 0121 決定 6）。 */
const ATTENTION_NODES = 1;

/** フィクスチャのメタの期待値（書き手 `export_vae.emit_targets` の逐語 — 2.2 は unpatchify の前の参照）。 */
const FIXTURE_WEIGHTS = "f16-rounded";
const FIXTURE_REFERENCE =
  "diffusers AutoencoderKLWan._decode chunk loop before unpatchify and clamp (CPU f32)";

const MIRROR_PRESENT = mirrorPresent(DIST_ROOT);
const ASSETS = ti2vVaeChunkFixtures();
const ANY_PRESENT = ASSETS.some(({ present }) => present);
const ALL_PRESENT = ASSETS.every(({ present }) => present);
/**
 * A complete fetch of the pinned test-asset dataset makes a missing fixture a FAIL (the completeness test below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_FIXTURES = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so the GPU-free checks of Wan2.2 TI2V's VAE chunk graphs are SKIPPED. Assemble it: ${ASSEMBLE}`,
  );
}
if (!ANY_PRESENT) {
  console.warn(
    `[karume] no VAE chunk fixture under ${GOLDEN_DIR.pathname}, so the fixture check is SKIPPED ` +
      `(${MISSING_FIXTURES})`,
  );
}

/** Opens and prepares the mirror's f16 container (the graph name is the component key). */
const prepared = async (component: string): Promise<PreparedModel> =>
  prepareContainer(await openWanTi2vComponent(component, "f16"), component);

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** conv3d の第 2 入力（重み）の名前の集合。 */
const conv3dWeights = (graph: Graph): ReadonlySet<string> =>
  new Set(graph.nodes.filter(({ op }) => op === "conv3d").map(({ ins }) => ins[1]));

/**
 * IR の主張（ファイル冒頭の 2 つ目）を破った箇所の一覧（守っていれば空）。値はグラフ入力（`values` に載らない）と
 * `values`（initializer を含む）の両方を見る。
 */
const irViolations = (graph: Graph): readonly string[] => {
  const violations: string[] = [];
  const unexpected = [...new Set(graph.nodes.map(({ op }) => op))]
    .filter((op) => !CHUNK_GRAPH_OPS.has(op))
    .sort();
  if (unexpected.length > 0) violations.push(`語彙外の op ${unexpected.join(" / ")}`);
  const weights = conv3dWeights(graph);
  const shapes = [
    ...graph.inputs.map(({ name, shape }) => [name, shape] as const),
    ...Object.entries(graph.values).map(([name, { shape }]) => [name, shape] as const),
  ];
  for (const [name, shape] of shapes) {
    if (shape.length <= MAX_VALUE_RANK) continue;
    const conv3dWeight =
      shape.length === 5 && weights.has(name) && Object.hasOwn(graph.initializers, name);
    if (!conv3dWeight) {
      violations.push(`rank ${shape.length} の値 '${name}'（rank 5 は conv3d の重みだけ）`);
    }
  }
  const attention = graph.nodes.filter(({ op }) => op === "attention").length;
  if (attention !== ATTENTION_NODES) {
    violations.push(`attention のノードが ${attention} 本（${ATTENTION_NODES} 本のはず）`);
  }
  return violations;
};

test.skipIf(!ANY_PRESENT && !FETCHED)(
  "Wan2.2 TI2V VAE chunk fixtures: the three fixtures are present",
  () => {
    assertEquals(
      ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `missing fixtures (${MISSING_FIXTURES})`,
    );
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V VAE chunk 宣言（GPU 不要）: 潜在 48 ch・タイル 16・出口 12 ch・cache は表の 32 本" +
    "（first は time_conv の 2 本を除く 30 本）・フレーム [12,1,128,128] / [12,4,128,128]",
  async () => {
    const layout = wanVaeChunkLayout(await prepared(FIRST), await prepared(NEXT));
    assertEquals(
      [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
      [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
      "潜在のチャネル数・タイル・フレームの辺・出口のチャネル数",
    );
    assertEquals(
      [...layout.cacheShapes.entries()],
      CACHE_TABLE.map(({ name, shape }) => [name, shape]),
      "cache の表（next の入力の順）",
    );
    assertEquals(
      layout.next.caches,
      CACHE_TABLE.map(({ name }) => name),
    );
    assertEquals(
      layout.first.caches,
      CACHE_TABLE.filter(({ timeConv }) => !timeConv).map(({ name }) => name),
      "first の cache = 表 − time_conv",
    );
    assertEquals(
      layout.next.caches.filter((name) => !layout.first.caches.includes(name)),
      [...TIME_CONV_CACHES],
    );
    assertEquals(TIME_CONV_CACHES, ["cache_11", "cache_18"]);
    assertEquals(layout.first.frameShape, [SAMPLE_CHANNELS, 1, SAMPLE_TILE, SAMPLE_TILE]);
    assertEquals(layout.next.frameShape, [SAMPLE_CHANNELS, 4, SAMPLE_TILE, SAMPLE_TILE]);
    assertEquals(
      [...layout.cacheShapes.values()].reduce((sum, shape) => sum + 4 * elements(shape), 0),
      CACHE_BYTES,
      "cache の合計バイト数（f32）",
    );
    assertEquals(
      layout.first.caches.reduce((sum, name) => {
        const shape = layout.cacheShapes.get(name);
        assert(shape !== undefined, `first の cache '${name}' の形が layout に無い`);
        return sum + 4 * elements(shape);
      }, 0),
      FIRST_CACHE_BYTES,
      "first の cache の合計バイト数（f32）",
    );
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V VAE chunk IR（GPU 不要）: rank 5 は conv3d の重みだけ・op は語彙の 15 種の中・attention は" +
    "各グラフ 1 本",
  async () => {
    for (const component of [FIRST, NEXT]) {
      const { graph } = await prepared(component);
      assertEquals(irViolations(graph), [], component);
      // rank 5 の例外が実際に働いていること（conv3d の重みが 1 本も無ければ上の主張は空振り）。
      const rank5 = Object.entries(graph.values)
        .filter(([, { shape }]) => shape.length === 5)
        .map(([name]) => name)
        .sort();
      assert(rank5.length > 0, `${component}: rank 5 の値が 1 本も無い`);
      assertEquals(rank5, [...conv3dWeights(graph)].sort(), `${component}: rank 5 の値の顔ぶれ`);
    }
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V VAE chunk IR の検査自身（GPU 不要）: 書き換えた宣言を 1 件ずつ拒む" +
    "（initializer でない conv3d の重み・rank 5 のグラフ入力・rank 5 の conv2d の重み・語彙外の op・attention 0 本）",
  async () => {
    // 守っている宣言（上のテスト）から 1 点だけ壊す — 違反がちょうど 1 件で、その主張の文言であること。
    const { graph } = await prepared(NEXT);
    const conv2d = graph.nodes.find(({ op }) => op === "conv2d");
    assert(conv2d !== undefined, "conv2d のノードが無い");
    const weight = conv2d.ins[1];
    assert(
      Object.hasOwn(graph.initializers, weight),
      `conv2d の重み '${weight}' が initializer でない`,
    );
    const declared = graph.values[weight];
    // 語彙外の op へ書き換える先は conv3d 以外（conv3d を書き換えると、その重みの rank 5 の例外も外れる）。
    const multiply = graph.nodes.findIndex(({ op }) => op === "mul");
    const attention = graph.nodes.findIndex(({ op }) => op === "attention");
    assert(multiply >= 0 && attention >= 0, "mul / attention のノードが無い");
    // rank 5 の例外の残りの枝: conv3d の重みでも initializer でなければ拒む・グラフ入力も見る。
    const [conv3dWeight] = [...conv3dWeights(graph)];
    assert(conv3dWeight !== undefined, "conv3d のノードが無い");
    const [input] = graph.inputs;
    assert(input !== undefined && input.shape.length === 4, "rank 4 のグラフ入力が無い");
    const rejected: readonly [string, Graph, string][] = [
      [
        "initializer でない conv3d の重み",
        {
          ...graph,
          initializers: Object.fromEntries(
            Object.entries(graph.initializers).filter(([name]) => name !== conv3dWeight),
          ),
        },
        `rank 5 の値 '${conv3dWeight}'`,
      ],
      [
        "rank 5 のグラフ入力",
        {
          ...graph,
          inputs: graph.inputs.map((declared, index) =>
            index === 0 ? { ...declared, shape: [1, ...declared.shape] } : declared,
          ),
        },
        `rank 5 の値 '${input.name}'`,
      ],
      [
        "rank 5 の非 conv3d の値（conv2d の重みの initializer）",
        {
          ...graph,
          values: { ...graph.values, [weight]: { ...declared, shape: [1, ...declared.shape] } },
        },
        `rank 5 の値 '${weight}'`,
      ],
      [
        "語彙外の op",
        {
          ...graph,
          nodes: graph.nodes.map((node, index) =>
            index === multiply ? { ...node, op: "pad" } : node,
          ),
        },
        "語彙外の op pad",
      ],
      [
        "attention 0 本",
        {
          ...graph,
          nodes: graph.nodes.map((node, index) =>
            index === attention ? { ...node, op: "mul" } : node,
          ),
        },
        "attention のノードが 0 本",
      ],
    ];
    for (const [label, broken, message] of rejected) {
      const violations = irViolations(broken);
      assertEquals(violations.length, 1, `${label}: ${violations.join(" / ")}`);
      assertStringIncludes(violations[0], message, label);
    }
  },
);

test.skipIf(!ALL_PRESENT)(
  "Wan2.2 TI2V VAE chunk フィクスチャ（GPU 不要）: メタ（タイル 16・chunk 数・役割・f16 の丸め・参照の素性）と" +
    "形（latents [48,F,16,16]・frames [12,1+4(F−1),128,128]）が表どおり・seed は 3 本で互いに違う",
  async () => {
    const seeds: string[] = [];
    for (const { name, role, chunks } of CASES) {
      const fixture = await readFixture(ti2vVaeFixtureUrl(name));
      const meta = (key: string): string | undefined => fixture.metadata.get(key);
      assertEquals(
        {
          tile: meta("tile"),
          chunks: meta("chunks"),
          role: meta("role"),
          weights: meta("weights"),
          reference: meta("reference"),
        },
        {
          tile: String(TILE),
          chunks: String(chunks),
          role,
          weights: FIXTURE_WEIGHTS,
          reference: FIXTURE_REFERENCE,
        },
        `${name}: メタ`,
      );
      assertEquals(fixture.latentShape, [LATENT_CHANNELS, chunks, TILE, TILE], `${name}: latents`);
      assertEquals(
        fixture.frameShape,
        [SAMPLE_CHANNELS, wanVaeFrameCount(chunks), SAMPLE_TILE, SAMPLE_TILE],
        `${name}: frames`,
      );
      const seed = meta("seed");
      assert(seed !== undefined, `${name}: メタ 'seed' が無い`);
      seeds.push(seed);
    }
    // 帯の決定用と受入れ用は別の潜在（決定と受入れの独立 — ADR 0118 追記 2026-10-02）。
    assertEquals(new Set(seeds).size, CASES.length, `seed の重複（${seeds.join(" / ")}）`);
  },
);
