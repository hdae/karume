// Wan2.2 TI2V-5B の VAE のタイル decode のホスト側（ADR 0121 段 5 — GPU 不要）。
//
// 見るのは 4 つ:
//
// - **タイル計画の凍結**: タイル 16・重なり = 64 px ÷ 空間の圧縮 16（グラフの比 8 × unpatchify 2）= 潜在 4 の開始位置を
//   値で凍結する（{@link AXIS_STARTS_TI2V} — Python 側 `tools/export-recipes/wan/tests/test_vae_tiling_ti2v.py` の
//   `MIRRORED_STARTS_TI2V` と同じ値の二重凍結）。計画は本番の入口 `planWanGenerationTiles` を通す。832×480 系は
//   記述子の受理集合に無い（公式の対応寸法の外）が、この関数は受理集合を見ないので同じ入口で凍結する（段 5 の
//   タイル参照が受理寸法に依らない数値の照合として使う）。
// - **要求の門**: 2.2 の記述子を `planWanRequest` に通し、受理寸法（1280×704 / 704×1280）・フレーム数の範囲
//   （4n+1 の 5〜49）・既定（1280×704・33 フレーム）を値で縛る。
// - **逆正規化の統計の写し**: `WAN22_LATENTS_MEAN` / `WAN22_LATENTS_STD` と fixture
//   `fixtures/wan-latents/wan22-ti2v.json` のビット一致（反対側は recipe の `wan/tests/test_ti2v_generation_stats.py` が
//   pin の config と照合する）。
// - **実資産の layout**（chunk グラフ 2 本がある機だけ）: 宣言から導いた layout が計画のリテラルと一致し、2.2 の
//   記述子の admission を通り、2.1 の記述子では落ちる。
//
// 資産の在否は chunk グラフ 2 本の組で分ける（1 本も無ければ明示 SKIP・片方だけなら FAIL — ADR 0005）。タイル参照
// `vae_tiles.*` の在否はここでは見ない（GPU の照合の側の組）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { prepareContainer, type PreparedModel } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import type { WanPipelineConfig } from "../src/wan/config.ts";
import { WAN21_GENERATION, WAN22_TI2V_GENERATION } from "../src/wan/descriptor.ts";
import { WAN22_LATENTS_MEAN, WAN22_LATENTS_STD } from "../src/wan/latents.ts";
import { planWanRequest, type PromptGate } from "../src/wan/plan.ts";
import {
  assertWanVaeMatchesGeneration,
  assertWanVaeTilesCover,
  planWanGenerationTiles,
  wanSpatialCompression,
} from "../src/wan/tile-decode.ts";
import { wanVaeChunkLayout } from "../src/wan/vae-chunks.ts";
import { wanVaeBlendExtentAt, wanVaeTileCount } from "../src/wan/vae-tiles.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import {
  ti2vVaeModelUrl,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_GENERATE as GENERATE,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_ROOT as ROOT,
  WAN_TI2V_VAE_SERIES as SERIES,
} from "./helpers/wan-ti2v-vae.ts";

const OWNER = "WanTi2vPipeline";

/** 2.2 の chunk グラフの宣言（タイル 16・patchify 空間のフレームの辺 128・潜在 48 ch・出口 12 ch）のリテラル。 */
const LAYOUT = { latentChannels: 48, tile: 16, sampleTile: 128, sampleChannels: 12 } as const;
/** グラフの入出力の空間比（`sampleTile / tile`）— ブレンド幅は patchify 空間の画素で数える。 */
const GRAPH_SCALE = 8;

/**
 * タイル辺 16・重なりの下限 4 の開始位置（潜在の全長 → 開始位置）。Python 側の `MIRRORED_STARTS_TI2V` と**同じ値**。
 *
 * - 16: 縮退（1 枚）
 * - 30 / 52: 832×480 / 480×832 の 2 辺
 * - 44 / 80: 1280×704 / 704×1280 の 2 辺（対ごとにブレンド幅が違う配置）
 */
const AXIS_STARTS_TI2V: readonly (readonly [number, readonly number[]])[] = [
  [16, [0]],
  [30, [0, 7, 14]],
  [44, [0, 9, 19, 28]],
  [52, [0, 12, 24, 36]],
  [80, [0, 11, 21, 32, 43, 53, 64]],
];

/** 本番の入口で 1 軸ぶん（全長 × 縮退 16）を計画して、行の開始位置を取る。 */
const rowStarts = (extent: number, generation = WAN22_TI2V_GENERATION): readonly number[] =>
  planWanGenerationTiles(LAYOUT, extent, LAYOUT.tile, generation).rows.starts;

Deno.test("Wan2.2 のタイル計画: 開始位置を値で凍結する（Python 側と同じ表・本番の入口を通す）", () => {
  for (const [extent, starts] of AXIS_STARTS_TI2V) {
    assertEquals([...rowStarts(extent)], [...starts], `潜在 ${extent}`);
  }
});

Deno.test("Wan2.2 のタイル計画: 空間の圧縮は 16（グラフの比 8 × unpatchify 2）・受理する 2 寸法はタイルで覆える", () => {
  assertEquals(wanSpatialCompression(LAYOUT, WAN22_TI2V_GENERATION), 16);
  assertWanVaeTilesCover(LAYOUT, WAN22_TI2V_GENERATION, OWNER);
});

Deno.test("Wan2.2 の要求の門: 1280×704 / 704×1280 × 49 フレームまでを通し、832×480 と 53 フレームは受理集合の文言で落とす", () => {
  const gate: PromptGate<string> = { resolve: (text) => text, defaultNegative: () => "negative" };
  const config: WanPipelineConfig = {
    scheduler: { shift: 5 },
    defaults: { steps: 50, guidance: 5 },
  };
  const planOf = (size: { width?: number; height?: number; frames?: number }) =>
    planWanRequest({ prompt: "p", ...size }, gate, LAYOUT, config, WAN22_TI2V_GENERATION);

  // 既定（1280×704・33 フレーム）と縦向きは通り、潜在の形とタイルの枚数は寸法から決まる。
  const byDefault = planOf({});
  assertEquals([byDefault.width, byDefault.height, byDefault.frames], [1280, 704, 33]);
  assertEquals(byDefault.latentShape, [48, 9, 44, 80]);
  assertEquals(wanVaeTileCount(byDefault.tiles), 28);
  assertEquals(planOf({ width: 704, height: 1280, frames: 33 }).latentShape, [48, 9, 80, 44]);

  // 上限は 49（開発機で 2 席とも 50 ステップの完走を確かめた値）。49 は両向きで通る。
  assertEquals(planOf({ frames: 49 }).latentShape, [48, 13, 44, 80]);
  assertEquals(planOf({ width: 704, height: 1280, frames: 49 }).latentShape, [48, 13, 80, 44]);

  // 公式の対応寸法の外（832×480 / 480×832）と上限を超えるフレーム数は入力起因で落ちる。
  for (const [width, height] of [[832, 480], [480, 832]]) {
    assertThrows(
      () => planOf({ width, height }),
      ModelInputError,
      `${width}×${height} が受理集合（1280×704 / 704×1280）に無い`,
    );
  }
  assertThrows(
    () => planOf({ frames: 53 }),
    ModelInputError,
    "frames 53 が受理集合（4n+1 の 5〜49）に無い",
  );

  // 下限は 5（5 は通り、4n+1 でも 1 は落ちる）。4n+1 でない値も落ちる。
  assertEquals(planOf({ frames: 5 }).latentShape, [48, 2, 44, 80]);
  for (const frames of [1, 31]) {
    assertThrows(
      () => planOf({ frames }),
      ModelInputError,
      `frames ${frames} が受理集合（4n+1 の 5〜49）に無い`,
    );
  }
});

/** 4 寸法の計画（出力の画素 → 枚数・開始位置・ブレンド幅〈patchify 空間の画素〉）。 */
const SIZES: readonly {
  readonly width: number;
  readonly height: number;
  readonly count: number;
  readonly rows: readonly number[];
  readonly cols: readonly number[];
  readonly rowBlend: readonly number[];
  readonly colBlend: readonly number[];
}[] = [
  {
    width: 832,
    height: 480,
    count: 12,
    rows: [0, 7, 14],
    cols: [0, 12, 24, 36],
    rowBlend: [72, 72],
    colBlend: [32, 32, 32],
  },
  {
    width: 480,
    height: 832,
    count: 12,
    rows: [0, 12, 24, 36],
    cols: [0, 7, 14],
    rowBlend: [32, 32, 32],
    colBlend: [72, 72],
  },
  {
    width: 1280,
    height: 704,
    count: 28,
    rows: [0, 9, 19, 28],
    cols: [0, 11, 21, 32, 43, 53, 64],
    rowBlend: [56, 48, 56],
    colBlend: [40, 48, 40, 40, 48, 40],
  },
  {
    width: 704,
    height: 1280,
    count: 28,
    rows: [0, 11, 21, 32, 43, 53, 64],
    cols: [0, 9, 19, 28],
    rowBlend: [40, 48, 40, 40, 48, 40],
    colBlend: [56, 48, 56],
  },
];

Deno.test("Wan2.2 のタイル計画: 832×480 / 480×832 は 12 枚・1280×704 / 704×1280 は 28 枚（枚数・開始位置・対ごとのブレンド幅）", () => {
  const compression = wanSpatialCompression(LAYOUT, WAN22_TI2V_GENERATION);
  for (const { width, height, count, rows, cols, rowBlend, colBlend } of SIZES) {
    const where = `${width}×${height}`;
    const plan = planWanGenerationTiles(
      LAYOUT,
      height / compression,
      width / compression,
      WAN22_TI2V_GENERATION,
    );
    assertEquals(
      [plan.latentChannels, plan.sampleChannels, plan.scale],
      [48, 12, GRAPH_SCALE],
      where,
    );
    assertEquals(wanVaeTileCount(plan), count, `${where}: 枚数`);
    assertEquals([...plan.rows.starts], [...rows], `${where}: 行`);
    assertEquals([...plan.cols.starts], [...cols], `${where}: 列`);
    assertEquals(
      rowBlend.map((_, index) => wanVaeBlendExtentAt(plan.rows, GRAPH_SCALE, index + 1)),
      [...rowBlend],
      `${where}: 行のブレンド幅`,
    );
    assertEquals(
      colBlend.map((_, index) => wanVaeBlendExtentAt(plan.cols, GRAPH_SCALE, index + 1)),
      [...colBlend],
      `${where}: 列のブレンド幅`,
    );
  }
});

Deno.test("Wan2.2 のタイル計画の故障注入: 重なりをグラフの比 8 だけで導く（unpatchify を無視して 8）と潜在 44 / 52 / 80 の行が割れる", () => {
  // unpatchify の倍率を落とした記述子 = 圧縮 8・重なり 8。潜在 16 / 30 は重なり 4 でも 8 でも同じ開始位置になる
  // ので、この注入を区別するのは 44 / 52 / 80 の行だけ（そこを名指す）。
  const graphScaleOnly = { ...WAN22_TI2V_GENERATION, vaePatchSize: 1 };
  const broken = AXIS_STARTS_TI2V
    .filter(([extent, starts]) => {
      const injected = rowStarts(extent, graphScaleOnly);
      return injected.length !== starts.length ||
        injected.some((start, index) => start !== starts[index]);
    })
    .map(([extent]) => extent);
  assertEquals(broken, [44, 52, 80]);
});

// ---- 逆正規化の統計の写し（fixture を挟んだ両側の照合）-----------------------------------------------

type LatentStatsFixture = {
  readonly source: { readonly repo: string; readonly revision: string; readonly reference: string };
  readonly mean: readonly number[];
  readonly std: readonly number[];
};

Deno.test("WAN22_LATENTS_MEAN / STD: fixture wan22-ti2v.json とビット一致（48 本ずつ・fixture の値は f32 そのもの）", async () => {
  const fixture = JSON.parse(
    await Deno.readTextFile(new URL("./fixtures/wan-latents/wan22-ti2v.json", import.meta.url)),
  ) as LatentStatsFixture;
  assertEquals(fixture.source.repo, "Wan-AI/Wan2.2-TI2V-5B-Diffusers");
  for (
    const [name, constant, values] of [
      ["mean", WAN22_LATENTS_MEAN, fixture.mean],
      ["std", WAN22_LATENTS_STD, fixture.std],
    ] as const
  ) {
    assertEquals(constant.length, LAYOUT.latentChannels, `${name}: 本数`);
    assertEquals(
      [...constant],
      [...Float32Array.from(values)],
      `${name}: 定数と fixture（f32）の食い違い — 片側だけ書き換えた`,
    );
    // fixture 自身が f32 の最短表記であること（f32 へ丸めて変わる値は config の写し方の誤り）。
    assertEquals([...values], [...Float32Array.from(values)], `${name}: fixture が f32 でない`);
  }
  assert(WAN22_TI2V_GENERATION.latents.mean === WAN22_LATENTS_MEAN, "記述子が別の mean を持つ");
  assert(WAN22_TI2V_GENERATION.latents.std === WAN22_LATENTS_STD, "記述子が別の std を持つ");
});

// ---- 実資産の layout（chunk グラフ 2 本の組で SKIP / 実行を分ける）---------------------------------------

const GRAPHS = [FIRST, NEXT].map((component) => {
  const url = ti2vVaeModelUrl(component);
  return { path: url.pathname, present: modelPresent(url) };
});
const ANY_GRAPH = GRAPHS.some(({ present }) => present);
const ALL_GRAPHS = GRAPHS.every(({ present }) => present);

if (!ANY_GRAPH) {
  console.warn(
    `[karume] ${ROOT.pathname} に Wan2.2 TI2V の VAE の chunk グラフが無いため、実資産の layout の照合を SKIP する` +
      `（重み 2.2GB につきリポジトリ管理外）。生成: ${GENERATE}`,
  );
}

const prepared = async (component: string): Promise<PreparedModel> =>
  prepareContainer(
    await openSeriesContainer(ti2vVaeModelUrl(component)),
    seriesGraph(SERIES, component),
  );

Deno.test({
  name: "Wan2.2 TI2V VAE のタイルの実資産: chunk グラフ 2 本が揃っている",
  // 1 本も無い環境は「生成していない」なので SKIP。1 本でもあるなら欠けは FAIL。
  ignore: !ANY_GRAPH,
  fn: () => {
    assertEquals(
      GRAPHS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `chunk グラフの欠け（生成: ${GENERATE}）`,
    );
  },
});

Deno.test({
  name:
    "Wan2.2 TI2V VAE のタイルの実資産（GPU 不要）: layout は計画のリテラルと一致・2.2 の記述子の admission を通り" +
    "・2.1 の記述子では落ちる",
  ignore: !ALL_GRAPHS,
  fn: async () => {
    const layout = wanVaeChunkLayout(await prepared(FIRST), await prepared(NEXT));
    assertEquals(
      {
        latentChannels: layout.latentChannels,
        tile: layout.tile,
        sampleTile: layout.sampleTile,
        sampleChannels: layout.sampleChannels,
      },
      { ...LAYOUT },
    );
    assertWanVaeMatchesGeneration(layout, WAN22_TI2V_GENERATION, OWNER);
    assertWanVaeTilesCover(layout, WAN22_TI2V_GENERATION, OWNER);
    assertThrows(
      () => assertWanVaeMatchesGeneration(layout, WAN21_GENERATION, OWNER),
      Error,
      `${OWNER}: 逆正規化の統計（mean 16 本・std 16 本）が VAE の潜在 48 チャネルと違う`,
    );
    // 統計の本数だけでなく出口のチャネル数でも落ちること（統計を 48 本にした 2.1 の倍率の記述子）。
    assertThrows(
      () =>
        assertWanVaeMatchesGeneration(
          layout,
          { ...WAN21_GENERATION, latents: WAN22_TI2V_GENERATION.latents },
          OWNER,
        ),
      Error,
      `${OWNER}: VAE の出口 12 チャネルが RGB 3 × unpatchify 1² = 3 と違う`,
    );
  },
});
