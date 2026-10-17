/**
 * Wan2.2 TI2V-5B の動画 VAE の**タイル decode**の参照照合（実 GPU — ADR 0121 段 5 の検収）。
 *
 * 系列 `outputs/series/wan2.2-ti2v-5b-f16-dyn/` の chunk グラフ 2 本（段 4 — 潜在タイル 16・出口 12 ch の patchify
 * 空間）を、本番の計画（`planWanGenerationTiles(…, WAN22_TI2V_GENERATION)` — 重なりは潜在 4）で全画面に敷き詰めて
 * 回す（タイルが外・chunk が内・1 タイル = 1 batch — 2.1 と同じ `decodeWanVaeTiled`）。比べる相手は recipe のタイル
 * 参照（`tools/export-recipes/wan/vae_tiling.py --model ti2v-5b` — 同じ幾何・上流の blend の逐語を patchify 空間で・
 * CPU f32・重みは f16 へ丸めた値）の **unpatchify とクランプの前**の出力。骨格は 2.1 の `e2e_wan_vae_tiles_test.ts`、
 * 帯が未導出の間の扱いは段 4 の chunk 列の照合（section "Chunk sequence" below）と同じ。
 *
 * ## 門
 *
 * 1. **計画**: TS の計画（資産の入力形から）がフィクスチャのメタ（Python の計画 — tile / scale / 開始位置 /
 *    ブレンド幅 / patch_size）と一致する。band は 12 枚（行 0,7,14・列 0,12,24,36）、wide は 28 枚。
 * 2. **縮退門**: 段 4 の chunk 列のフィクスチャの潜在 `[48,9,16,16]`（タイル 1 枚ちょうど）では、タイル経路
 *    （`decodeWanVaeTiled`）が非タイルの chunk 列（`decodeWanVaeTile`）と **Uint32 一致**。
 * 3. **帯**: `band`（832×480・81 フレーム・12 枚 × 21 chunk）の `allclose(atol = BAND, rtol = 0)`。受入れは `accept`
 *    （480×832・9 フレーム — 縦長・別の chunk 境界）と `wide`（1280×704・5 フレーム — 対ごとにブレンド幅が違う）。
 *    さらに `accept` の貼り合わせを VAE 段の末尾（`finishWanVaeFrames` — 形の検査 → unpatchify → 有限性 → クランプ）に
 *    通した RGB を、上流の `unpatchify` → `clamp` を当てた `frames_rgb` と**同じ帯**で比べる（unpatchify は置換・
 *    クランプは 1-Lipschitz なので帯はそのまま — ホストの unpatchify を上流の関数そのものと照合する）。
 *    帯に依らない形でも見る（GPU 不要のテスト）: フィクスチャの `frames`（上流のタイル参照そのもの）に
 *    `finishWanVaeFrames` を当てた RGB が、同じ `frames` から上流が作った `frames_rgb` と **Uint32 一致**（unpatchify
 *    が上流の並び `(c, r, q)` とビット一致 — ADR 0121 の検収表の段 5。帯が未導出の回でも緑になる）。
 * 4. **毒値と非有限**: フレームの常駐を各タイルの decode の前に NaN で埋め直してから回し（ケースの先頭と、
 *    `onTile` で次のタイルの前 — 前のタイルが残した有限の値を写し損ねの隠れ蓑にしない）、全ケースで非有限 0・
 *    device lost なし。
 * 5. **故障注入**（ホスト側 — `accept` の decode 済みタイルを故障ありで貼り合わせる）: 行と列の取り違え・開始位置の
 *    1 潜在ずれ が帯の外に出る。2.1 の「帯の 1e3 倍」の床は持ち込まない（2.2 の注入の効きは測っていない — 段 4 と
 *    同じ判断）。
 *
 * ## 帯の決め方
 *
 * - 帯は決定用 1 本（`band`）の素の突合の実測 maxAbs × 5 を有効数字 2 桁へ切り上げて決める。受入れ（accept /
 *   accept の RGB / wide）は帯の決定に使わない。
 * - `undefined` = 未導出: 各ケースは maxAbs を記録して赤で止まり、故障注入も maxAbs を記録して赤で止まり、最後の
 *   step が帯の候補を出す（{@link bandCandidate}）。計画・縮退門・非有限 0・device lost なしは未導出の回でも緑に
 *   なる（それぞれ独立の step）。最初の実走が導いた値を、実測の表と一緒に定数へ書く。
 *
 * ## 資産
 *
 * タイル参照 `vae_tiles.*` は chunk グラフの組とは別の在否の組: 3 本とも無ければ明示 SKIP・一部だけなら FAIL
 * （このファイルの GPU 不要のテスト）。タイル参照が 1 本でもあるのに chunk グラフの組（容器 2 本 + `vae_chunks.*`
 * — 照合の前提）が揃っていなければ、それも FAIL（照合が一度も走らずにレーンが緑で終わる形を作らない）。chunk の組
 * だけがある機（タイル参照をまだ書いていない）は SKIP のまま。chunk の組の中の欠けは段 4 のホストテスト
 * （`wan_ti2v_vae_chunks_host_test.ts`）も落とす。
 *
 * ## Chunk sequence (ADR 0121 stage 4 acceptance)
 *
 * The GPU tests placed before the tile decode tests check the chunk graphs without a tile plan. They keep
 * their own band ({@link CHUNK_BAND}), cases (`WAN_TI2V_VAE_CASES`) and results family (`wan-ti2v-vae-chunks`),
 * separate from the tile decode's.
 *
 * Wan2.2 TI2V-5B の動画 VAE の **chunk 列の参照照合**（実 GPU — ADR 0121 段 4 の検収）。
 *
 * 系列 `outputs/series/wan2.2-ti2v-5b-f16-dyn/` の chunk グラフ 2 本（`vae_decoder_first` / `vae_decoder_next` —
 * f16 席・潜在タイル 16・出口 12 ch の patchify 空間）を、製品の経路（`decodeWanVaeTile` — 2.1 と同じ関数・
 * 1 タイル = 1 batch・フレームだけを読み戻す）で first → next × (F−1) と回し、上流の非タイル `_decode` の chunk
 * ループ（diffusers・CPU f32・重みは f16 へ丸めた値）の **unpatchify とクランプの前**の出力と比べる。参照と
 * フィクスチャは `tools/export-recipes/wan/export_vae.py --model ti2v-5b` が書く（生成コマンドと表は
 * `helpers/wan-ti2v-vae.ts`）。骨格は 2.1 の `e2e_wan_vae_tiles_test.ts` と同じで、故障注入のループは共有の
 * `helpers/wan-vae-chunk-loop.ts`。
 *
 * ### 帯の決め方
 *
 * - 帯は `band` ケース（9 chunk = 33 フレーム）の素の突合（`rtol = 0`）の実測 maxAbs × 5 から決める（2.1 の
 *   `BAND` と同じ導き方 — {@link CHUNK_BAND}）。受入れは別の潜在・別の chunk 境界の `accept`（5 chunk = 17 フレーム）と
 *   `long`（21 chunk = 81 フレーム — cache を 20 回持ち越す長さ）で、帯の内であることを門にする。帯は long の結果を
 *   見て変えない。
 * - `undefined` = 未導出: 各ケースは maxAbs を記録して赤で止まり、最後の step が帯の候補（band の maxAbs × 5 を
 *   有効数字 2 桁へ切り上げた値）を出す（{@link chunkBandCandidate} — 段 2 の DiT の e2e と同じ形。通る値を仮置きして
 *   検出力の無い門を作らない）。最初の実走が導いた値を、実測の表と一緒に定数へ書く。
 *
 * ### 故障注入
 *
 * - cache 更新忘れ・cache の 2 フレームの逆順・タイルの頭のゼロ化の省略は、`accept` で**帯の外**に出ることを門に
 *   する（2.1 の「帯の 1e3 倍」の床は 2.1 の実測から持ってきた値で、2.2 の注入の効きは測っていないので持ち込まない）。
 *   帯が未導出の回は maxAbs を記録して赤で止まる（注入先は帯を書くのと同じコミットで確定する — 外れてから注入先を
 *   変えると、帯を後から合わせるのと同じ形になる）。
 * - chunk 2 以降に first を使う形は、値を比べる前に `copyOutputs` の大きさで拒まれる。
 * - 注入のループが故障なしなら製品の経路と Uint32 で一致することも門にする（注入の結果が製品の経路を代表している
 *   担保）。
 * - DupUp3D の先頭 slice の欠落は GPU では注入しない（グラフの中のノードで、欠けたグラフは export も eager も形で
 *   落ちる — 門は recipe の pytest と、first のフレームが 4 枚の宣言を拒む TS のホストテスト `wan_vae_chunks_test.ts`）。
 *
 * ### mid の attention（D = 1024）
 *
 * - 両グラフの直近 run の計画が融合 attention の 3 段（`attention_qk` / `attention_stats` / `attention_pv`）を 1 本以上
 *   積むことを門にする（ADR 0121 決定 6 は D の上限の検査が無いまま D = 1024 の実測を段 4 に残した — 値の正しさは
 *   帯が見るが、融合の経路を通ったかは帯からは分からない）。計画の census が無い run は 0 本と数えずに赤にする。
 *
 * 資産が揃っていない環境と GPU 無し環境は明示 SKIP。資産の完全性（一部だけある環境は FAIL）は GPU 不要のホスト
 * テスト（`wan_ti2v_vae_chunks_host_test.ts`）が見る。
 *
 * ## Default run and opt-in upstream parity (maintainer ruling 2026-10-17)
 *
 * Every comparison with an upstream reference under a band (the chunk sequence against the diffusers
 * `_decode`, the tile decode and the accept RGB against the recipe's upstream tiled reference), the
 * fault injections that only prove those bands are live (the cache faults and the seam faults), and
 * the band-candidate steps are opt-in behind `KARUME_UPSTREAM_PARITY=1` (`helpers/upstream-parity.ts`).
 * The default run keeps the checks that do not depend on a band: the asset gates, the GPU-free
 * unpatchify Uint32 identity with `frames_rgb`, the chunk layout and cache size, the fused-attention
 * census, the fault loop's Uint32 identity with the product path, the `copyOutputs` rejection, the
 * TS/Python plan equality, the degenerate Uint32 gate, and every case decoded from NaN-poisoned
 * frames with 0 non-finite and no device lost. The record-only cases (the measured-mode submit GPU
 * time, the VRAM and submit statistics, the untiled `frames_full` observation) are removed.
 */

import { readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import {
  ExecutionError,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import {
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeChunkLayout,
  type WanVaeChunkLayout,
  wanVaeFrameCount,
} from "../../models/src/wan/vae-chunks.ts";
import type { PlanLayout } from "../../models/src/wan/plan.ts";
import {
  assembleWanVaeTiles,
  decodeWanVaeTiled,
  decodeWanVaeTiles,
  wanVaeBlendExtentAt,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../../models/src/wan/vae-tiles.ts";
import { WAN22_TI2V_GENERATION } from "../../models/src/wan/descriptor.ts";
import {
  assertWanVaeMatchesGeneration,
  finishWanVaeFrames,
  planWanGenerationTiles,
  wanSpatialCompression,
} from "../../models/src/wan/tile-decode.ts";
import { disposeSteps } from "../../models/src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  bitsEqual,
  decodeWithFault,
  type Fault,
  type Fixture,
  readFixture,
} from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import {
  ti2vVaeAssets,
  type Ti2vVaeCase,
  ti2vVaeFixtureUrl,
  ti2vVaeModelUrl,
  ti2vVaeTileAssets,
  type Ti2vVaeTileCase,
  ti2vVaeTileFixtureUrl,
  WAN_TI2V_VAE_CACHE_BYTES as CACHE_BYTES,
  WAN_TI2V_VAE_CACHE_TABLE as CACHE_TABLE,
  WAN_TI2V_VAE_CASES as CHUNK_CASES,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_GENERATE as GENERATE_CHUNKS,
  WAN_TI2V_VAE_LATENT_CHANNELS as LATENT_CHANNELS,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_ROOT as ROOT,
  WAN_TI2V_VAE_SAMPLE_CHANNELS as SAMPLE_CHANNELS,
  WAN_TI2V_VAE_SAMPLE_TILE as SAMPLE_TILE,
  WAN_TI2V_VAE_SERIES as SERIES,
  WAN_TI2V_VAE_TILE as TILE,
  WAN_TI2V_VAE_TILE_CASES as CASES,
  WAN_TI2V_VAE_TILES_GENERATE as GENERATE_TILES,
  WAN_TI2V_VAE_TIME_CONV_CACHES as TIME_CONV_CACHES,
} from "../../models/tests/helpers/wan-ti2v-vae.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  openResults,
  type RecordedCase,
  runRecordedCase,
} from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * タイル decode の帯（絶対値 — 出力はクランプ前の patchify 空間のフレーム）。`undefined` = 未導出（モジュール doc
 * 「帯の決め方」）。
 *
 * 導出（2026-10-05・B570）: 決定用 band（12 枚 × 21 chunk）の maxAbs 9.805e-6 × 5 = 4.902e-5 → 有効数字 2 桁へ
 * 切り上げ 5.0e-5。受入れは accept 7.339e-6・accept の RGB 7.339e-6・wide 6.348e-6。故障注入 2 件の maxAbs は
 * 2.18 / 2.26。
 *
 * MUST: 受入れ（accept / accept の RGB / wide）の結果を見てこの値も決定用のケースも変えない。受入れが帯を外れたら、
 * 帯を広げずに原因を調べる。
 */
const BAND: number | undefined = 5.0e-5;
/** 判定と記録の帯（未導出の回は無限の帯として記録し、判定は赤にする）。 */
const TOLERANCE: Tolerance = { atol: BAND ?? Number.POSITIVE_INFINITY, rtol: 0 };

/** VAE 段の末尾の文言の持ち主（2.2 の pipeline の名前 — ホストテストと同じ）。 */
const OWNER = "WanTi2vPipeline";
/** RGB のチャネル数（unpatchify の後の `frames_rgb` の先頭の次元）。 */
const RGB_CHANNELS = 3;

type CaseName = Ti2vVaeTileCase["name"];

/** 縮退門の潜在（段 4 の chunk 列のフィクスチャ `vae_chunks.band` — 潜在 `[48,9,16,16]`・タイル 1 枚ちょうど）。 */
const DEGENERATE_CASE = "band";

const TILE_ASSETS = ti2vVaeTileAssets();
const ANY_TILES = TILE_ASSETS.some(({ present }) => present);
const ALL_TILES = TILE_ASSETS.every(({ present }) => present);
const CHUNK_ASSETS = ti2vVaeAssets();
const CHUNK_ASSETS_PRESENT = CHUNK_ASSETS.every(({ present }) => present);
if (!ANY_TILES) {
  console.warn(
    `[karume] ${ROOT.pathname} に Wan2.2 TI2V の VAE のタイル参照（vae_tiles.*）が無いため、タイル decode の照合を ` +
      `SKIP する。生成: ${GENERATE_TILES}`,
  );
}

/** タイル参照のメタ `weights`（重みの素性 — 書き手 `vae_tiling.write_fixture` と同じ綴り）。 */
const FIXTURE_WEIGHTS = "f16-rounded";
/** タイル参照のメタ `reference`（参照の素性 — `vae_tiling.FIXTURE_REFERENCE_IN_PATCHIFY_SPACE` と同じ綴り）。 */
const FIXTURE_REFERENCE =
  "snapped-tile decode with upstream blend_v / blend_h in patchify space before unpatchify and clamp (CPU f32)";

// 1 本も無い環境は「生成していない」なので SKIP。1 本でもあるなら欠けは FAIL（chunk グラフの組とは別の組）。
test.skipIf(!ANY_TILES)(
  "Wan2.2 TI2V VAE タイル参照: vae_tiles の 3 本（band / accept / wide）が揃っている",
  () => {
    assertEquals(
      TILE_ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `タイル参照の欠け（生成: ${GENERATE_TILES}）`,
    );
  },
);

// タイル参照の書き手は chunk の組に依らない（HF の snapshot から書く）ので、この組み合わせは起こりうる。SKIP に
// すると GPU の照合が一度も走らないままレーンが緑で終わる。
test.skipIf(!ANY_TILES)(
  "Wan2.2 TI2V VAE タイル参照: タイル参照があるなら照合の前提の chunk グラフの組（容器 2 本 + vae_chunks.*）も揃っている",
  () => {
    assertEquals(
      CHUNK_ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `タイル参照はあるが chunk グラフの組が欠けている（生成: ${GENERATE_CHUNKS}）`,
    );
  },
);

/** タイル参照 1 本（F32 のテンソルとメタ — 書き手は `vae_tiling.write_fixture`）。 */
type TileFixture = {
  readonly tensor: (key: string) => {
    readonly data: Float32Array<ArrayBuffer>;
    readonly shape: readonly number[];
  };
  readonly has: (key: string) => boolean;
  readonly meta: (key: string) => string;
};

const readTileFixture = async (url: URL): Promise<TileFixture> => {
  const bytes = await readFile(url);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    tensor: (key) => {
      const tensor = file.tensors.get(key);
      assert(
        tensor !== undefined && tensor.dtype === "F32",
        `${url.pathname}: '${key}' が F32 で無い`,
      );
      return {
        data: new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4),
        shape: tensor.shape,
      };
    },
    has: (key) => file.tensors.has(key),
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: メタ '${key}' が無い`);
      return value;
    },
  };
};

/**
 * 計画の幾何をフィクスチャのメタ（`vae_tiling.TilePlan.meta` + `patch_size`）と同じ綴りへ落とす（TS ↔ Python の
 * 突き合わせ — 数列は `"0,7,14"` の形）。
 */
const planMeta = (plan: WanVaeTilePlan): Record<string, string> => {
  assertEquals(plan.rows.tile, plan.cols.tile, "タイルは正方");
  const blends = (axis: WanVaeTilePlan["rows"]) =>
    axis.starts.slice(1).map((_, index) => wanVaeBlendExtentAt(axis, plan.scale, index + 1));
  return {
    tile: String(plan.rows.tile),
    scale: String(plan.scale),
    rows_starts: plan.rows.starts.join(","),
    cols_starts: plan.cols.starts.join(","),
    rows_blend: blends(plan.rows).join(","),
    cols_blend: blends(plan.cols).join(","),
    patch_size: String(WAN22_TI2V_GENERATION.vaePatchSize),
  };
};

/**
 * 潜在 `[C,F,H,W]` から本番と同じ入口で計画を立てる（`planWanGenerationTiles` を通すので、フィクスチャのメタとの
 * 突き合わせが本番の計画を縛る）。
 */
const planFor = (layout: PlanLayout, latentShape: readonly number[]): WanVaeTilePlan => {
  assertEquals(latentShape.length, 4, `潜在の形 [${latentShape}]`);
  return planWanGenerationTiles(layout, latentShape[2], latentShape[3], WAN22_TI2V_GENERATION);
};

/**
 * 計画の layout のリテラル（GPU 不要のテスト用 — 実資産の layout がこれと一致することは GPU のテストの計画の step と
 * 段 5 のホストテストが見る）。
 */
const LITERAL_LAYOUT: PlanLayout = {
  latentChannels: LATENT_CHANNELS,
  tile: TILE,
  sampleTile: SAMPLE_TILE,
  sampleChannels: SAMPLE_CHANNELS,
};

// 帯に依らない照合（unpatchify は置換・clamp は f32 の値を f32 の値へ写すだけ — 同じ入力なら丸めの差は出ない）。
// 一部だけの組は在否のテストが落とすので、ここは 3 本揃った機だけで回す。
test.skipIf(!ALL_TILES)(
  "Wan2.2 TI2V VAE タイル参照（GPU 不要）: accept の frames に VAE 段の末尾（finishWanVaeFrames）を当てた RGB が" +
    "上流の unpatchify → clamp の frames_rgb と Uint32 一致",
  async () => {
    const spec = CASES.find(({ name }) => name === "accept");
    assert(spec !== undefined && spec.rgb, "accept のケース（frames_rgb を持つ）が表に無い");
    const fixture = await readTileFixture(ti2vVaeTileFixtureUrl("accept"));
    const latents = fixture.tensor("latents");
    const plan = planFor(LITERAL_LAYOUT, latents.shape);
    const compression = wanSpatialCompression(LITERAL_LAYOUT, WAN22_TI2V_GENERATION);
    const want = fixture.tensor("frames_rgb");
    const frames = wanVaeFrameCount(latents.shape[1]);
    const height = spec.height * compression;
    const width = spec.width * compression;
    assertEquals(want.shape, [RGB_CHANNELS, frames, height, width], "frames_rgb の形");
    const got = finishWanVaeFrames(
      // 写しを渡す（クランプがその場で書くかどうかに依らず、フィクスチャの値を残す）。
      Float32Array.from(fixture.tensor("frames").data),
      { frames, width, height, tiles: plan },
      WAN22_TI2V_GENERATION,
      OWNER,
    );
    assertEquals(got.length, want.data.length, "要素数");
    assert(
      bitsEqual(got, want.data),
      "TS の unpatchify → clamp が上流の frames_rgb とビット一致しない",
    );
  },
);

const absMax = (values: Float32Array): number =>
  values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

/** 差の要約（比 = 最大絶対差 ÷ 参照の最大絶対値 — 決定 8 の指標）。 */
const compare = (got: Float32Array, want: Float32Array, tolerance: Tolerance) => {
  assertEquals(got.length, want.length, "要素数");
  const report = allclose(got, want, tolerance);
  const referenceMax = absMax(want);
  return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
};

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);

/** フレームの常駐を NaN で埋める（写し忘れのフレームは NaN のまま残る — 毒値）。 */
const poisonFrames = async (caches: WanVaeChunkCaches, chunks: number): Promise<void> => {
  for (const resident of await caches.frames(chunks)) {
    resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
  }
};

/**
 * `onTile` の口: 最後のタイルを除き、次のタイルの decode の前にフレームの常駐を毒値で埋め直す（タイルの出力は
 * `decodeWanVaeTile` が読み戻した後なので、決着したタイルの値は壊さない）。
 */
const poisonBetweenTiles =
  (caches: WanVaeChunkCaches, plan: WanVaeTilePlan, chunks: number) => async (tile: number) => {
    if (tile < wanVaeTileCount(plan)) await poisonFrames(caches, chunks);
  };

/** 有効数字 2 桁への切り上げ（段 2 / 4 の e2e の帯の候補と同じ規則）。 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(value / unit) * unit).toPrecision(2));
};

const results = openResults("wan-ti2v-vae-tiles");

/**
 * 未導出の帯（`undefined`）の候補を出して赤で止める step の本体（band だけから作り、受入れと故障注入の maxAbs は
 * 並べるが候補に使わない。band が maxAbs を出す前に落ちた回は候補を出さない）。帯が導出済みなら step は SKIP
 * （登録側の `it.skipIf`）。
 */
const bandCandidate = async (
  observed: ReadonlyMap<string, number>,
  faults: ReadonlyMap<string, number>,
): Promise<void> => {
  const band = observed.get("band");
  const candidate =
    band === undefined
      ? "band が maxAbs を出す前に落ちた — 候補にしない（落ちたケースを先に調べる）"
      : band > 0
        ? `band の maxAbs ${band.toExponential(3)} × 5 = ${(band * 5).toExponential(3)}` +
          `（有効数字 2 桁へ切り上げ ${roundUpTwoDigits(band * 5)}）`
        : "band の maxAbs が 0 — × 5 で帯を導けない（照合の経路を先に調べる）";
  const format = (value: number | undefined): string => value?.toExponential(3) ?? "—";
  const accepted = ["accept", "accept-rgb", "wide"].map(
    (label) => `${label} ${format(observed.get(label))}`,
  );
  const injected = [...faults].map(([label, value]) => `${label} ${format(value)}`);
  const message = `帯が未導出 — ${candidate}。受入れの maxAbs（帯の決定に使わない）: ${accepted.join(
    " / ",
  )}。故障注入の maxAbs（C4 で帯の外かを判定）: ${
    injected.length === 0 ? "—" : injected.join(" / ")
  }`;
  await runRecordedCase(results, { id: "band-candidate", failureNote: () => message }, () => {
    console.log(`[wan-ti2v-vae-tiles] ${message}`);
    throw new Error(message);
  });
};

// Chunk sequence (module doc "Chunk sequence"): one tile = one batch, no tile plan.

/**
 * chunk 列の帯（絶対値 — 出力はクランプ前の patchify 空間のフレーム）。`undefined` = 未導出（モジュール doc
 * 「帯の決め方」）。
 *
 * 導出（2026-10-05・B570）: 決定用 band（9 chunk）の maxAbs 9.477e-6 × 5 = 4.739e-5 → 有効数字 2 桁へ切り上げ 4.8e-5。
 * 受入れは accept（5 chunk）7.421e-6・long（21 chunk）1.153e-5。故障注入 3 件の maxAbs は 0.66 / 1.37 / 1.30。
 *
 * MUST: 受入れ（accept / long）の結果を見てこの値も決定用のケースも変えない。受入れが帯を外れたら、帯を広げずに
 * 原因を調べる。
 */
const CHUNK_BAND: number | undefined = 4.8e-5;
/** 判定と記録の帯（未導出の回は無限の帯として記録し、判定は赤にする）。 */
const CHUNK_TOLERANCE: Tolerance = { atol: CHUNK_BAND ?? Number.POSITIVE_INFINITY, rtol: 0 };

type ChunkCaseName = Ti2vVaeCase["name"];

/** 故障注入で外す cache（head の conv — 1 層だけを通る最も効きの小さい席・2.1 と同じ位置）。 */
const DROPPED_CACHE = "cache_31";

/** 記録する融合 attention の 3 段（ADR 0023 — ①QK・②行統計・③PV のパイプラインキーの接頭辞）。 */
const ATTENTION_KERNELS = ["attention_qk", "attention_stats", "attention_pv"] as const;

if (!CHUNK_ASSETS_PRESENT) {
  console.warn(
    `[karume] ${ROOT.pathname} に Wan2.2 TI2V の VAE の chunk グラフとフィクスチャが揃っていないため、` +
      `chunk 列の照合を SKIP する（欠けの判定はホストテスト）。生成: ${GENERATE_CHUNKS}`,
  );
}

const chunkResults = openResults("wan-ti2v-vae-chunks");

/** 容器 2 本を開いて prepare する（グラフ名は系列のグラフ名の表から）。 */
const prepareModels = async () => {
  const [firstOpened, nextOpened] = await Promise.all([
    openSeriesContainer(ti2vVaeModelUrl(FIRST)),
    openSeriesContainer(ti2vVaeModelUrl(NEXT)),
  ]);
  return {
    first: prepareContainer(firstOpened, seriesGraph(SERIES, FIRST)),
    next: prepareContainer(nextOpened, seriesGraph(SERIES, NEXT)),
  };
};

/**
 * 未導出の帯（`undefined`）の候補を出して赤で止める step の本体（band だけから作り、受入れの maxAbs は並べるが
 * 候補に使わない。band が maxAbs を出す前に落ちた回は候補を出さない）。帯が導出済みなら step は SKIP（登録側の
 * `it.skipIf`）。
 */
const chunkBandCandidate = async (observed: ReadonlyMap<ChunkCaseName, number>): Promise<void> => {
  const band = observed.get("band");
  const candidate =
    band === undefined
      ? "band が maxAbs を出す前に落ちた — 候補にしない（落ちたケースを先に調べる）"
      : band > 0
        ? `band の maxAbs ${band.toExponential(3)} × 5 = ${(band * 5).toExponential(3)}` +
          `（有効数字 2 桁へ切り上げ ${roundUpTwoDigits(band * 5)}）`
        : "band の maxAbs が 0 — × 5 で帯を導けない（照合の経路を先に調べる）";
  const accepted = CHUNK_CASES.filter(({ role }) => role === "accept").map(
    ({ name }) => `${name} ${observed.get(name)?.toExponential(3) ?? "—"}`,
  );
  const message = `帯が未導出 — ${candidate}。受入れの maxAbs（帯の決定に使わない）: ${accepted.join(
    " / ",
  )}`;
  await runRecordedCase(chunkResults, { id: "band-candidate", failureNote: () => message }, () => {
    console.log(`[wan-ti2v-vae] ${message}`);
    throw new Error(message);
  });
};

/**
 * 融合 attention の 3 段の dispatch 本数と、`attention` で始まるパイプラインキーの一覧（直近 run の計画から）。
 *
 * MUST: census の無い run を 0 本として数えない（融合の門が黙って空振りする）— 無ければ投げる。
 */
const attentionPipelines = (graph: string, diagnostics: SessionDiagnostics) => {
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) throw new Error(`${graph}: 直近 run の pipeline の census が無い`);
  const dispatches = (kernel: (typeof ATTENTION_KERNELS)[number]): number =>
    pipelines
      .filter(({ key }) => key.split(":")[0] === kernel)
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  return {
    dispatches: {
      attention_qk: dispatches("attention_qk"),
      attention_stats: dispatches("attention_stats"),
      attention_pv: dispatches("attention_pv"),
    },
    keys: pipelines.filter(({ key }) => key.startsWith("attention")).map(({ key }) => key),
  };
};

/** The two product Sessions, the cache residents and the fixtures one chunk-sequence case runs on. */
type ChunkRun = {
  readonly gpu: GpuContext;
  readonly layout: WanVaeChunkLayout;
  readonly sessions: { readonly first: Session; readonly next: Session };
  readonly caches: WanVaeChunkCaches;
  readonly fixtures: Readonly<Record<ChunkCaseName, Fixture>>;
};

/**
 * Registers, in the enclosing describe, a beforeAll that opens the two chunk graphs and the chunk
 * fixtures and builds both product Sessions and the cache residents on one device, and an afterAll
 * that tears them down, so the default suite and the opt-in parity suite share one lifecycle and one
 * teardown order. Returns the accessor the its read the run through.
 */
const registerChunkSessions = (): (() => ChunkRun) => {
  let gpu: GpuContext | undefined;
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let run: ChunkRun | undefined;
  beforeAll(async () => {
    const models = await prepareModels();
    const layout = wanVaeChunkLayout(models.first, models.next);
    const fixtures = {
      band: await readFixture(ti2vVaeFixtureUrl("band")),
      accept: await readFixture(ti2vVaeFixtureUrl("accept")),
      long: await readFixture(ti2vVaeFixtureUrl("long")),
    };

    const device = verifyRunningGpu(await acquireTestGpu());
    gpu = device;
    first = await models.first.createContainerSession(device);
    next = await models.next.createContainerSession(device);
    caches = await WanVaeChunkCaches.create(device, layout);
    run = { gpu: device, layout, sessions: { first, next }, caches, fixtures };
  });
  afterAll(async () => {
    const device = gpu;
    // MUST: Session → 常駐 → device の順で畳む（2.1 の e2e と同じ順・同じ理由 — 常駐は Session の焼き込みから
    // 参照されうるので Session を先に畳む。1 段が落ちても残りの段を必ず通す）。Vitest reports an afterAll failure
    // apart from the its, so a teardown failure cannot overwrite a test failure.
    await disposeSteps([
      () => first?.dispose(),
      () => next?.dispose(),
      () => caches?.dispose(),
      // device を捨てる前に解放を待つ（後続のテストの予算を残す — {@link settleReleases}）。
      () => (device === undefined ? undefined : settleReleases(device)),
      () => device?.destroy(),
    ]);
  });
  return () => {
    if (run === undefined) throw new Error("chunk-sequence setup (beforeAll) did not complete");
    return run;
  };
};

// Default run: the chunk layout and cache size, every case decoded from NaN-poisoned frames with 0
// non-finite, the fused-attention census, the fault loop's Uint32 identity with the product path and
// the copyOutputs rejection. It owns the per-case results.json ids; the band comparison with the
// diffusers `_decode` is the opt-in suite below.
describe.skipIf(!CHUNK_ASSETS_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 TI2V VAE chunk 列: 常駐 cache の first → next の取り決め・毒値のフレームから非有限 0・融合 attention の" +
    " 3 段・故障注入のループと製品の経路の Uint32 一致（実 GPU）",
  () => {
    const chunkRun = registerChunkSessions();

    it("取り決め: 潜在 48 ch・タイル 16・出口 12 ch・cache は表の 32 本（first は time_conv の 2 本を除く 30 本）", () => {
      const { layout } = chunkRun();
      assertEquals(
        [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
        [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
      );
      assertEquals(
        layout.next.caches,
        CACHE_TABLE.map(({ name }) => name),
      );
      assertEquals(
        layout.next.caches.filter((name) => !layout.first.caches.includes(name)),
        [...TIME_CONV_CACHES],
      );
      assertEquals(layout.first.frameShape, [SAMPLE_CHANNELS, 1, SAMPLE_TILE, SAMPLE_TILE]);
      assertEquals(layout.next.frameShape, [SAMPLE_CHANNELS, 4, SAMPLE_TILE, SAMPLE_TILE]);
    });

    it("観測: タイルの頭の cache のゼロ化（ホストからの書き込み）の量と時間", () => {
      const { caches: liveCaches } = chunkRun();
      // 2.1 と同じ記録（GPU 側でゼロを書く口が runtime に無い — `vae-chunks.ts` の doc）。段 5 で 1 生成あたり
      // タイル数ぶん払う。
      const started = performance.now();
      liveCaches.zero();
      const hostMs = performance.now() - started;
      console.log(
        `[wan-ti2v-vae] zero: bytes=${liveCaches.cacheBytes} hostMs=${hostMs.toFixed(1)}`,
      );
      assertEquals(liveCaches.cacheBytes, CACHE_BYTES);
    });

    for (const { name, chunks: expectedChunks } of CHUNK_CASES) {
      it(`${name}: chunk 列を毒値のフレームから回して非有限 0（chunk 数・フレームの形・要素数）`, async () => {
        const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
        await runRecordedCase(chunkResults, { id: name }, async () => {
          const fixture = fixtures[name];
          const chunks = wanVaeChunkCount(layout, fixture.latents);
          assertEquals(chunks, expectedChunks, `${name} の chunk 数`);
          assertEquals(fixture.frameShape, [
            SAMPLE_CHANNELS,
            wanVaeFrameCount(chunks),
            SAMPLE_TILE,
            SAMPLE_TILE,
          ]);
          await poisonFrames(liveCaches, chunks);
          const started = performance.now();
          const got = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
          const elapsedMs = performance.now() - started;
          assertEquals(got.length, fixture.frames.length, `${name}: 要素数`);
          const nonFinite = countNonFinite(got);
          console.log(
            `[wan-ti2v-vae] ${name}: chunks=${chunks} nonFinite=${nonFinite} ` +
              `decodeMs=${elapsedMs.toFixed(0)}`,
          );
          assertEquals(nonFinite, 0, `${name}: 非有限（写されなかったフレーム）`);
          return undefined;
        });
      });
    }

    it("mid の attention（D = 1024）は両グラフで融合 attention の 3 段を通る", () => {
      const { sessions } = chunkRun();
      // 直近のタイルの計画から（帯が未導出でも各ケースは照合の前に回し切るので census は埋まる）。
      for (const [graph, session] of Object.entries(sessions)) {
        const { dispatches } = attentionPipelines(graph, session.diagnostics());
        for (const kernel of ATTENTION_KERNELS) {
          assert(
            dispatches[kernel] > 0,
            `${graph}: ${kernel} の dispatch が 0 本（${JSON.stringify(dispatches)}）`,
          );
        }
      }
    });

    it("故障注入のループは故障なしなら製品の経路と Uint32 で一致", async () => {
      const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
      const fixture = fixtures.accept;
      const chunks = wanVaeChunkCount(layout, fixture.latents);
      // 毒値は各デコードの前に入れ直す（2 本目が 1 本目の書いたフレームを読み戻して一致を装う形を消す）。
      await poisonFrames(liveCaches, chunks);
      const product = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
      await poisonFrames(liveCaches, chunks);
      const looped = await decodeWithFault(gpu, sessions, liveCaches, fixture.latents, {
        kind: "none",
      });
      assertEquals(
        looped.filter((value) => !Number.isFinite(value)).length,
        0,
        "非有限（ループが写さなかったフレーム）",
      );
      assert(bitsEqual(product, looped), "テストのループが製品の経路と違う");
    });

    it("故障注入: chunk 2 以降も first を使う → copyOutputs の大きさで enqueue が拒まれる", async () => {
      const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
      // first の出力フレームは [12,1,…] で、next 用のフレームの常駐（[12,4,…]）へは写せない —
      // runtime の `copyOutputs` の大きさの検査が dispatch を積む前に落とす（値を比べる前の赤）。
      const error = await assertRejects(
        () =>
          decodeWithFault(gpu, sessions, liveCaches, fixtures.accept.latents, {
            kind: "first-for-all",
          }),
        ExecutionError,
        "copyOutputs",
      );
      // 拒んだのが chunk 1 のフレームの常駐と first のフレームの形の食い違いであること（他の写し先の拒否と
      // 取り違えない）。
      assertStringIncludes(error.message, "'frame 1'");
      assertStringIncludes(error.message, `[${layout.first.frameShape.join(",")}]`);
    });
  },
);

// Upstream parity (the chunk sequence against the diffusers `_decode` under the absolute band), so it
// is opt-in, together with the cache fault injections that only prove the band is live and the
// band-candidate step. It writes the fault/* and band-candidate results.json ids; the per-case ids
// belong to the default suite above, and writing them here would duplicate its settlements.
describe.skipIf(!CHUNK_ASSETS_PRESENT || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan2.2 TI2V VAE chunk 列: 常駐 cache の first → next が上流の _decode（unpatchify の前）と帯の中で一致（実 GPU）" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const chunkRun = registerChunkSessions();
    /** ケースごとの maxAbs（帯の候補の材料 — 判定の前に積むので帯の外の回も残る）。 */
    const observed = new Map<ChunkCaseName, number>();

    const compareChunks = (name: ChunkCaseName, got: Float32Array) => {
      const fixture = chunkRun().fixtures[name];
      assertEquals(got.length, fixture.frames.length, `${name}: 要素数`);
      const report = allclose(got, fixture.frames, CHUNK_TOLERANCE);
      const referenceMax = fixture.frames.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
      return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
    };

    for (const { name, chunks: expectedChunks } of CHUNK_CASES) {
      it(`${name}: chunk 列が上流と帯の中で一致（フレームは毒値で始める）`, async () => {
        const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
        const fixture = fixtures[name];
        const chunks = wanVaeChunkCount(layout, fixture.latents);
        assertEquals(chunks, expectedChunks, `${name} の chunk 数`);
        await poisonFrames(liveCaches, chunks);
        const got = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
        const { report, referenceMax, ratio } = compareChunks(name, got);
        console.log(
          `[wan-ti2v-vae] ${name}: chunks=${chunks} maxAbs=${report.maxAbsError} ` +
            `refMax=${referenceMax} ratio=${ratio} nonFinite=${report.nonFiniteCount}`,
        );
        assertEquals(report.nonFiniteCount, 0, `${name}: 非有限（写されなかったフレーム）`);
        observed.set(name, report.maxAbsError);
        if (CHUNK_BAND === undefined) {
          throw new Error(
            `${name}: 帯が未導出（maxAbs ${report.maxAbsError}）— band の maxAbs × 5 を帯の定数へ` +
              "書く（帯の候補は最後の step が出す）",
          );
        }
        assert(report.pass, `${name}: maxAbs ${report.maxAbsError} が帯 ${CHUNK_BAND} の外`);
      });
    }

    const faults: readonly { readonly label: string; readonly fault: Fault }[] = [
      {
        label: "cache 更新忘れ（copyOutputs から 1 本外す）",
        fault: { kind: "drop-cache", cache: DROPPED_CACHE },
      },
      { label: "cache の 2 フレームの順を逆にする", fault: { kind: "swap-cache-frames" } },
      { label: "タイルの頭で cache をゼロに戻さない", fault: { kind: "skip-zero" } },
    ];
    for (const { label, fault } of faults) {
      it(`故障注入: ${label} → accept が帯の外`, async () => {
        const { gpu, sessions, caches: liveCaches, fixtures } = chunkRun();
        await runRecordedCase(
          chunkResults,
          { id: `fault/${fault.kind}` },
          async ({ measurements }) => {
            if (fault.kind === "skip-zero") {
              // 前のタイル（band）の cache を残したまま accept を回す。
              await decodeWanVaeTile(gpu, sessions, liveCaches, fixtures.band.latents);
            }
            const got = await decodeWithFault(
              gpu,
              sessions,
              liveCaches,
              fixtures.accept.latents,
              fault,
            );
            const { report, ratio } = compareChunks("accept", got);
            measurements.push({
              output: `frames(fault ${fault.kind})`,
              maxAbs: report.maxAbsError,
              maxRel: report.maxRelError,
              tolerance: CHUNK_TOLERANCE,
              stage: "karume",
            });
            console.log(
              `[wan-ti2v-vae] fault ${fault.kind}: maxAbs=${report.maxAbsError} ratio=${ratio}` +
                (CHUNK_BAND === undefined
                  ? ""
                  : ` band×${(report.maxAbsError / CHUNK_BAND).toFixed(1)}`),
            );
            if (CHUNK_BAND === undefined) {
              throw new Error(`${label}: 帯が未導出（故障注入の maxAbs ${report.maxAbsError}）`);
            }
            assert(!report.pass, `${label}: 帯 ${CHUNK_BAND} の中に収まった（帯が広すぎる兆候）`);
            return undefined;
          },
        );
      });
    }

    it.skipIf(CHUNK_BAND !== undefined)("帯の候補（未導出）", () => chunkBandCandidate(observed));
  },
);

/** The two product Sessions, the cache residents and the tile fixtures one tile-decode case runs on. */
type TileRun = {
  readonly gpu: GpuContext;
  readonly layout: WanVaeChunkLayout;
  /** Sample pixels per latent pixel (VAE scale × patch size). */
  readonly compression: number;
  readonly sessions: { readonly first: Session; readonly next: Session };
  readonly caches: WanVaeChunkCaches;
  readonly fixtures: Readonly<Record<CaseName, TileFixture>>;
  /** The device-lost reason so far (`undefined` while the device is alive). */
  readonly deviceLost: () => string | undefined;
};

/**
 * Registers, in the enclosing describe, a beforeAll that opens the two chunk graphs and the tile
 * fixtures and builds both product Sessions and the cache residents on one device that records a
 * device loss, and an afterAll that tears them down, so the default suite and the opt-in parity suite
 * share one lifecycle and one teardown order. Returns the accessor the its read the run through.
 */
const registerTileSessions = (): (() => TileRun) => {
  let gpu: GpuContext | undefined;
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let run: TileRun | undefined;
  beforeAll(async () => {
    const models = await prepareModels();
    const layout = wanVaeChunkLayout(models.first, models.next);
    const compression = wanSpatialCompression(layout, WAN22_TI2V_GENERATION);
    const fixtures: Readonly<Record<CaseName, TileFixture>> = {
      band: await readTileFixture(ti2vVaeTileFixtureUrl("band")),
      accept: await readTileFixture(ti2vVaeTileFixtureUrl("accept")),
      wide: await readTileFixture(ti2vVaeTileFixtureUrl("wide")),
    };

    let deviceLost: string | undefined;
    const device = verifyRunningGpu(
      await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      }),
    );
    gpu = device;
    first = await models.first.createContainerSession(device);
    next = await models.next.createContainerSession(device);
    caches = await WanVaeChunkCaches.create(device, layout);
    run = {
      gpu: device,
      layout,
      compression,
      sessions: { first, next },
      caches,
      fixtures,
      deviceLost: () => deviceLost,
    };
  });
  afterAll(async () => {
    const device = gpu;
    // MUST: Session → 常駐 → device の順で畳む（2.1 / 段 4 の e2e と同じ — 1 段が落ちても残りの段を必ず通す）。
    // Vitest reports an afterAll failure apart from the its, so a teardown failure cannot overwrite a test
    // failure.
    await disposeSteps([
      () => first?.dispose(),
      () => next?.dispose(),
      () => caches?.dispose(),
      // device を捨てる前に解放を待つ（後続のテストの予算を残す — {@link settleReleases}）。
      () => (device === undefined ? undefined : settleReleases(device)),
      () => device?.destroy(),
    ]);
  });
  return () => {
    if (run === undefined) throw new Error("tile-decode setup (beforeAll) did not complete");
    return run;
  };
};

/**
 * One tile case through the product tile decode (`decodeWanVaeTiled`), starting from NaN-poisoned
 * frames and poisoning them again between tiles.
 */
const decodeTiledCase = async (run: TileRun, name: CaseName) => {
  const latents = run.fixtures[name].tensor("latents");
  const plan = planFor(run.layout, latents.shape);
  const chunks = latents.shape[1];
  await poisonFrames(run.caches, chunks);
  const started = performance.now();
  const got = await decodeWanVaeTiled(
    run.gpu,
    run.sessions,
    run.caches,
    plan,
    latents.data,
    poisonBetweenTiles(run.caches, plan, chunks),
  );
  return { got, plan, chunks, elapsedMs: performance.now() - started };
};

/**
 * The accept case decoded tile by tile and assembled on the host, starting from NaN-poisoned frames
 * (the decoded tiles are returned for the host-side seam fault injections).
 */
const decodeAcceptCase = async (run: TileRun) => {
  const latents = run.fixtures.accept.tensor("latents");
  const plan = planFor(run.layout, latents.shape);
  const chunks = latents.shape[1];
  await poisonFrames(run.caches, chunks);
  const decodeStarted = performance.now();
  const tiles = await decodeWanVaeTiles(
    run.gpu,
    run.sessions,
    run.caches,
    plan,
    latents.data,
    poisonBetweenTiles(run.caches, plan, chunks),
  );
  const decodeMs = performance.now() - decodeStarted;
  const assembleStarted = performance.now();
  const got = assembleWanVaeTiles(tiles, plan);
  const assembleMs = performance.now() - assembleStarted;
  return { tiles, got, plan, chunks, decodeMs, assembleMs };
};

// Default run: the TS/Python plan equality, the degenerate Uint32 gate, and band / accept / wide
// decoded from NaN-poisoned frames with 0 non-finite and no device lost. It owns the per-case
// results.json ids; the band comparisons with the upstream tiled reference are the opt-in suite below.
describe.skipIf(!ALL_TILES || !CHUNK_ASSETS_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 TI2V VAE タイル decode: 計画が Python と一致・縮退門（Uint32）・band / accept / wide を毒値のフレームから" +
    "回して非有限 0・device lost なし（実 GPU）",
  () => {
    let degenerate: Fixture | undefined;
    beforeAll(async () => {
      degenerate = await readFixture(ti2vVaeFixtureUrl(DEGENERATE_CASE));
    });
    const tileRun = registerTileSessions();
    /** ケース名 → 非有限の数（門 4 — 判定の前に積む）。 */
    const nonFinite = new Map<string, number>();

    /** Records the non-finite count of one case, then fails when it is not 0. */
    const assertFinite = (label: string, got: Float32Array, want: Float32Array, extra: string) => {
      assertEquals(got.length, want.length, "要素数");
      const count = countNonFinite(got);
      nonFinite.set(label, count);
      console.log(`[wan-ti2v-vae-tiles] ${label}: nonFinite=${count}${extra}`);
      if (count > 0) {
        throw new Error(`${label}: 非有限 ${count} 個（写されなかったフレーム）`);
      }
    };

    it("計画: TS の計画（資産の入力形から・2.2 の記述子）がフィクスチャのメタ（Python）と一致", () => {
      const { layout, compression, fixtures } = tileRun();
      assertEquals(
        [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
        [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
      );
      assertWanVaeMatchesGeneration(layout, WAN22_TI2V_GENERATION, OWNER);
      for (const { name, role, chunks, height, width, full, rgb } of CASES) {
        const fixture = fixtures[name];
        const latents = fixture.tensor("latents");
        assertEquals(latents.shape, [LATENT_CHANNELS, chunks, height, width], `${name} の潜在の形`);
        assertEquals(
          {
            role: fixture.meta("role"),
            weights: fixture.meta("weights"),
            reference: fixture.meta("reference"),
          },
          { role, weights: FIXTURE_WEIGHTS, reference: FIXTURE_REFERENCE },
          `${name} の役割と素性（f16 の丸め・patchify 空間の参照）`,
        );
        const plan = planFor(layout, latents.shape);
        const ours = planMeta(plan);
        const theirs = Object.fromEntries(Object.keys(ours).map((key) => [key, fixture.meta(key)]));
        assertEquals(ours, theirs, `${name} の幾何`);
        const frames = wanVaeFrameCount(chunks);
        assertEquals(
          fixture.tensor("frames").shape,
          [SAMPLE_CHANNELS, frames, height * plan.scale, width * plan.scale],
          `${name} の frames の形（patchify 空間）`,
        );
        assertEquals(fixture.has("frames_full"), full, `${name} の frames_full の有無`);
        assertEquals(fixture.has("frames_rgb"), rgb, `${name} の frames_rgb の有無`);
        if (rgb) {
          assertEquals(
            fixture.tensor("frames_rgb").shape,
            [RGB_CHANNELS, frames, height * compression, width * compression],
            `${name} の frames_rgb の形`,
          );
        }
      }
      const band = planFor(layout, fixtures.band.tensor("latents").shape);
      assertEquals(wanVaeTileCount(band), 12);
      assertEquals([...band.rows.starts], [0, 7, 14]);
      assertEquals([...band.cols.starts], [0, 12, 24, 36]);
      assertEquals(wanVaeTileCount(planFor(layout, fixtures.wide.tensor("latents").shape)), 28);
    });

    it("縮退門: 潜在 16×16（1 枚）ではタイル経路 ≡ 非タイルの chunk 列（Uint32）", async () => {
      const { gpu, layout, sessions, caches: liveCaches } = tileRun();
      assert(degenerate !== undefined, "縮退門のフィクスチャ（beforeAll）が読めていない");
      const expectedChunks = CHUNK_CASES.find(({ name }) => name === DEGENERATE_CASE)?.chunks;
      assert(expectedChunks !== undefined, `chunk 列のケース '${DEGENERATE_CASE}' が表に無い`);
      assertEquals(
        degenerate.latentShape,
        [LATENT_CHANNELS, expectedChunks, TILE, TILE],
        "縮退門の潜在の形",
      );
      const plan = planFor(layout, degenerate.latentShape);
      assertEquals(wanVaeTileCount(plan), 1);
      const chunks = wanVaeChunkCount(layout, degenerate.latents);
      await poisonFrames(liveCaches, chunks);
      const tiled = await decodeWanVaeTiled(gpu, sessions, liveCaches, plan, degenerate.latents);
      // 毒値は各デコードの前に入れ直す（2 本目が 1 本目の書いたフレームを読み戻して一致を装う形を消す）。
      await poisonFrames(liveCaches, chunks);
      const direct = await decodeWanVaeTile(gpu, sessions, liveCaches, degenerate.latents);
      assertEquals(
        tiled.length,
        SAMPLE_CHANNELS * wanVaeFrameCount(chunks) * SAMPLE_TILE * SAMPLE_TILE,
      );
      assertEquals(countNonFinite(tiled), 0, "非有限（写されなかったフレーム）");
      assert(bitsEqual(tiled, direct), "タイル経路が chunk 列とビット一致しない");
    });

    for (const name of ["band", "wide"] as const) {
      it(`${name}: 毒値のフレームから回して非有限 0（タイル経路 decodeWanVaeTiled）`, async () => {
        const run = tileRun();
        await runRecordedCase(results, { id: name }, async () => {
          const { got, plan, chunks, elapsedMs } = await decodeTiledCase(run, name);
          assertFinite(
            name,
            got,
            run.fixtures[name].tensor("frames").data,
            ` tiles=${wanVaeTileCount(plan)} chunks=${chunks} decodeMs=${elapsedMs.toFixed(0)}`,
          );
          return undefined;
        });
      });
    }

    it("accept: 毒値のフレームから回して非有限 0（タイルごとの decode + ホストの貼り合わせ）", async () => {
      const run = tileRun();
      await runRecordedCase(results, { id: "accept" }, async () => {
        const { got, plan, chunks, decodeMs, assembleMs } = await decodeAcceptCase(run);
        assertFinite(
          "accept",
          got,
          run.fixtures.accept.tensor("frames").data,
          ` tiles=${wanVaeTileCount(plan)} chunks=${chunks} decodeMs=${decodeMs.toFixed(
            0,
          )} assembleMs=${assembleMs.toFixed(0)}`,
        );
        return undefined;
      });
    });

    it("毒値と device lost: 全ケース（band / accept / wide）で非有限 0・device lost なし", () => {
      const run = tileRun();
      const names = CASES.map(({ name }) => name);
      assertEquals(
        names.filter((name) => !nonFinite.has(name)),
        [],
        "非有限を数える前に落ちたケース",
      );
      assertEquals(
        names.map((name) => nonFinite.get(name)),
        names.map(() => 0),
        `非有限の数（${names.join(" / ")}）`,
      );
      assertEquals(run.deviceLost(), undefined, "device lost");
    });
  },
);

// Upstream parity (band / accept / wide and the accept RGB against the upstream tiled reference under
// the absolute band), so it is opt-in, together with the seam fault injections that only prove the
// band is live and the band-candidate step. It writes the accept-rgb, fault/* and band-candidate
// results.json ids; the per-case ids belong to the default suite above, and writing them here would
// duplicate its settlements.
describe.skipIf(!ALL_TILES || !CHUNK_ASSETS_PRESENT || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan2.2 TI2V VAE タイル decode: 縮退門 + 832×480・81 フレームが上流のタイル参照（unpatchify の前）と帯の中で" +
    "一致・accept の RGB が上流の unpatchify → clamp と帯の中で一致（実 GPU）" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const tileRun = registerTileSessions();
    /** ラベル（ケース名 / `accept-rgb`）→ maxAbs（帯の候補の材料 — 判定の前に積むので帯の外の回も残る）。 */
    const observed = new Map<string, number>();
    /** 故障注入のラベル → maxAbs（記録 — 判定は帯が導出された回だけ）。 */
    const faultObserved = new Map<string, number>();

    /**
     * 1 ケースの判定（maxAbs を記録してから — 未導出の回も候補の材料が残る）。非有限 → 帯が未導出 → 帯の外 の順で
     * 落とす。
     */
    const judge = (
      label: string,
      got: Float32Array,
      want: Float32Array,
      extra = "",
      recorded?: {
        readonly measurements: RecordedCase["measurements"];
        readonly output: string;
      },
    ): void => {
      const { report, referenceMax, ratio } = compare(got, want, TOLERANCE);
      recorded?.measurements.push({
        output: recorded.output,
        maxAbs: report.maxAbsError,
        maxRel: report.maxRelError,
        tolerance: TOLERANCE,
        stage: "karume",
      });
      observed.set(label, report.maxAbsError);
      console.log(
        `[wan-ti2v-vae-tiles] ${label}: maxAbs=${report.maxAbsError} refMax=${referenceMax} ` +
          `ratio=${ratio} nonFinite=${report.nonFiniteCount}${extra}`,
      );
      if (report.nonFiniteCount > 0) {
        throw new Error(`${label}: 非有限 ${report.nonFiniteCount} 個（写されなかったフレーム）`);
      }
      if (BAND === undefined) {
        throw new Error(
          `${label}: 帯が未導出（maxAbs ${report.maxAbsError}）— band の maxAbs × 5 を帯の定数へ書く` +
            "（帯の候補は最後の step が出す）",
        );
      }
      assert(report.pass, `${label}: maxAbs ${report.maxAbsError} が帯 ${BAND} の外`);
    };

    it("band: 832×480・81 フレーム（12 枚 × 21 chunk）が参照と帯の中で一致（フレームは毒値で始める）", async () => {
      const run = tileRun();
      const { got, elapsedMs } = await decodeTiledCase(run, "band");
      judge(
        "band",
        got,
        run.fixtures.band.tensor("frames").data,
        ` decodeMs=${elapsedMs.toFixed(0)}`,
      );
    });

    // 受入れ: decode 済みのタイルを残し、VAE 段の末尾の照合と故障注入（ホスト側の貼り合わせ）に使い回す。
    /** The accept fixture, its reference frames, plan and chunk count (derived from the run's fixtures). */
    const acceptCase = () => {
      const { layout, fixtures } = tileRun();
      const accept = fixtures.accept;
      const latents = accept.tensor("latents");
      return {
        accept,
        want: accept.tensor("frames").data,
        plan: planFor(layout, latents.shape),
        chunks: latents.shape[1],
      };
    };
    let acceptTiles: Float32Array<ArrayBuffer>[] = [];
    let acceptAssembled: Float32Array<ArrayBuffer> | undefined;

    it("accept: 480×832・9 フレーム（別の潜在・chunk 境界・縦長）が帯の中", async () => {
      const { want } = acceptCase();
      const { tiles, got, decodeMs, assembleMs } = await decodeAcceptCase(tileRun());
      acceptTiles = tiles;
      acceptAssembled = got;
      judge(
        "accept",
        got,
        want,
        ` decodeMs=${decodeMs.toFixed(0)} assembleMs=${assembleMs.toFixed(0)}`,
      );
    });

    it("accept の RGB: VAE 段の末尾（finishWanVaeFrames — unpatchify → clamp）が上流の unpatchify → clamp と帯の中", async () => {
      const { compression } = tileRun();
      const { accept, plan: acceptPlan, chunks: acceptChunks } = acceptCase();
      await runRecordedCase(results, { id: "accept-rgb" }, ({ measurements }) => {
        assert(acceptAssembled !== undefined, "accept が貼り合わされていない");
        const spec = CASES.find(({ name }) => name === "accept");
        assert(spec !== undefined, "accept のケースが表に無い");
        const video = finishWanVaeFrames(
          acceptAssembled,
          {
            frames: wanVaeFrameCount(acceptChunks),
            width: spec.width * compression,
            height: spec.height * compression,
            tiles: acceptPlan,
          },
          WAN22_TI2V_GENERATION,
          OWNER,
        );
        judge("accept-rgb", video, accept.tensor("frames_rgb").data, "", {
          measurements,
          output: `frames_rgb(${acceptChunks} chunks × ${wanVaeTileCount(acceptPlan)} tiles)`,
        });
        return Promise.resolve(undefined);
      });
    });

    it("wide: 1280×704・5 フレーム（28 枚・対ごとに違うブレンド幅）が帯の中", async () => {
      const run = tileRun();
      const { got, elapsedMs } = await decodeTiledCase(run, "wide");
      judge(
        "wide",
        got,
        run.fixtures.wide.tensor("frames").data,
        ` decodeMs=${elapsedMs.toFixed(0)}`,
      );
    });

    const faults: readonly {
      readonly id: string;
      readonly label: string;
      readonly plan: (acceptPlan: WanVaeTilePlan) => WanVaeTilePlan;
    }[] = [
      {
        // 縦長でだけ値に出る（正方では対合）。枚数は 4×3 = 3×4 で揃うので形の検査を抜ける。
        id: "swap-axes",
        label: "行と列の取り違え",
        plan: (acceptPlan) => ({ ...acceptPlan, rows: acceptPlan.cols, cols: acceptPlan.rows }),
      },
      {
        // TS と Python の丸めが割れたときの形（decode した位置と貼る位置が 1 潜在ずれる）。
        id: "shift-start",
        label: "開始位置の 1 潜在ずれ",
        plan: (acceptPlan) => ({
          ...acceptPlan,
          rows: {
            ...acceptPlan.rows,
            starts: acceptPlan.rows.starts.map((start, index) => (index === 1 ? start + 1 : start)),
          },
        }),
      },
    ];
    for (const { id, label, plan } of faults) {
      it(`故障注入: ${label} → accept が帯の外`, async () => {
        const { want: acceptWant, plan: acceptPlan } = acceptCase();
        await runRecordedCase(results, { id: `fault/${id}` }, ({ measurements }) => {
          assert(acceptTiles.length > 0, "accept のタイルが decode されていない");
          const got = assembleWanVaeTiles(acceptTiles, plan(acceptPlan));
          const { report, ratio } = compare(got, acceptWant, TOLERANCE);
          measurements.push({
            output: `frames(fault ${id})`,
            maxAbs: report.maxAbsError,
            maxRel: report.maxRelError,
            tolerance: TOLERANCE,
            stage: "karume",
          });
          faultObserved.set(label, report.maxAbsError);
          console.log(
            `[wan-ti2v-vae-tiles] fault ${id}: maxAbs=${report.maxAbsError} ratio=${ratio}` +
              (BAND === undefined ? "" : ` band×${(report.maxAbsError / BAND).toFixed(1)}`),
          );
          if (BAND === undefined) {
            throw new Error(`${label}: 帯が未導出（故障注入の maxAbs ${report.maxAbsError}）`);
          }
          assert(!report.pass, `${label}: 帯 ${BAND} の中に収まった（帯が広すぎる兆候）`);
          return Promise.resolve(undefined);
        });
      });
    }

    it.skipIf(BAND !== undefined)("帯の候補（未導出）", () =>
      bandCandidate(observed, faultObserved),
    );
  },
);
