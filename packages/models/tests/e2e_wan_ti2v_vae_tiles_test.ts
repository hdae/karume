/**
 * Wan2.2 TI2V-5B の動画 VAE の**タイル decode**の参照照合（実 GPU — ADR 0121 段 5 の検収）。
 *
 * 系列 `outputs/series/wan2.2-ti2v-5b-f16-dyn/` の chunk グラフ 2 本（段 4 — 潜在タイル 16・出口 12 ch の patchify
 * 空間）を、本番の計画（`planWanGenerationTiles(…, WAN22_TI2V_GENERATION)` — 重なりは潜在 4）で全画面に敷き詰めて
 * 回す（タイルが外・chunk が内・1 タイル = 1 batch — 2.1 と同じ `decodeWanVaeTiled`）。比べる相手は recipe のタイル
 * 参照（`tools/export-recipes/wan/vae_tiling.py --model ti2v-5b` — 同じ幾何・上流の blend の逐語を patchify 空間で・
 * CPU f32・重みは f16 へ丸めた値）の **unpatchify とクランプの前**の出力。骨格は 2.1 の `e2e_wan_vae_tiles_test.ts`、
 * 帯が未導出の間の扱いは段 4 の `e2e_wan_ti2v_vae_chunks_test.ts` と同じ。
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
 * ## 観測（門ではない）
 *
 * - band の壁時計（cache のゼロ化のホスト書き込み・タイル間の毒値の書き込み・ホストの貼り合わせを含む）。
 * - VRAM の山: `monitorDrmUsage`（`helpers/drm-usage.ts` — 区間ごとの fdinfo の山）と、診断から足し上げた内訳
 *   （`vramBreakdown` + タイルの頭の cache のゼロ化の staging）を、見積り（{@link VRAM_ESTIMATE_GIB}）と並べる。
 * - `accept` の非タイルの上流 decode（`frames_full` — patchify 空間・クランプ前）との差（クランプ前とクランプ後）。
 *   タイル化は近似（受容野がタイル内に閉じる）なので 0 にならない。accept の判定とは別の step で出す（観測の側で
 *   落ちても accept の maxAbs と非有限の数は記録に残る）。
 *
 * ## 資産
 *
 * タイル参照 `vae_tiles.*` は chunk グラフの組とは別の在否の組: 3 本とも無ければ明示 SKIP・一部だけなら FAIL
 * （このファイルの GPU 不要のテスト）。タイル参照が 1 本でもあるのに chunk グラフの組（容器 2 本 + `vae_chunks.*`
 * — 照合の前提）が揃っていなければ、それも FAIL（照合が一度も走らずにレーンが緑で終わる形を作らない）。chunk の組
 * だけがある機（タイル参照をまだ書いていない）は SKIP のまま。chunk の組の中の欠けは段 4 のホストテスト
 * （`wan_ti2v_vae_chunks_host_test.ts`）も落とす。
 */

import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, prepareContainer, type Session } from "@karume/runtime";
import {
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeChunkLayout,
  wanVaeFrameCount,
} from "../src/wan/vae-chunks.ts";
import type { PlanLayout } from "../src/wan/plan.ts";
import {
  assembleWanVaeTiles,
  clampWanVaeFrames,
  decodeWanVaeTiled,
  decodeWanVaeTiles,
  wanVaeBlendExtentAt,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../src/wan/vae-tiles.ts";
import { WAN22_TI2V_GENERATION } from "../src/wan/descriptor.ts";
import {
  assertWanVaeMatchesGeneration,
  finishWanVaeFrames,
  planWanGenerationTiles,
  wanSpatialCompression,
} from "../src/wan/tile-decode.ts";
import { disposeSteps } from "../src/session/dispose-steps.ts";
import { type DrmTimeline, formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { bitsEqual, readFixture, vramBreakdown } from "./helpers/wan-vae-chunk-loop.ts";
import {
  ti2vVaeAssets,
  ti2vVaeFixtureUrl,
  ti2vVaeModelUrl,
  ti2vVaeTileAssets,
  type Ti2vVaeTileCase,
  ti2vVaeTileFixtureUrl,
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
} from "./helpers/wan-ti2v-vae.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import {
  openResults,
  type RecordedCase,
  runRecordedCase,
} from "../../runtime/tests/helpers/results.ts";

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
/** VAE 段の VRAM の見積り（段 4 の診断の合計 3.99 GiB + cache のゼロ化の staging 0.44 GiB — ADR 0121 段 5 の設計）。 */
const VRAM_ESTIMATE_GIB = "4.2〜4.5";

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

Deno.test({
  name: "Wan2.2 TI2V VAE タイル参照: vae_tiles の 3 本（band / accept / wide）が揃っている",
  // 1 本も無い環境は「生成していない」なので SKIP。1 本でもあるなら欠けは FAIL（chunk グラフの組とは別の組）。
  ignore: !ANY_TILES,
  fn: () => {
    assertEquals(
      TILE_ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `タイル参照の欠け（生成: ${GENERATE_TILES}）`,
    );
  },
});

Deno.test({
  name:
    "Wan2.2 TI2V VAE タイル参照: タイル参照があるなら照合の前提の chunk グラフの組（容器 2 本 + vae_chunks.*）も揃っている",
  // タイル参照の書き手は chunk の組に依らない（HF の snapshot から書く）ので、この組み合わせは起こりうる。SKIP に
  // すると GPU の照合が一度も走らないままレーンが緑で終わる。
  ignore: !ANY_TILES,
  fn: () => {
    assertEquals(
      CHUNK_ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `タイル参照はあるが chunk グラフの組が欠けている（生成: ${GENERATE_CHUNKS}）`,
    );
  },
});

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
  const bytes = await Deno.readFile(url);
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

Deno.test({
  name:
    "Wan2.2 TI2V VAE タイル参照（GPU 不要）: accept の frames に VAE 段の末尾（finishWanVaeFrames）を当てた RGB が" +
    "上流の unpatchify → clamp の frames_rgb と Uint32 一致",
  // 帯に依らない照合（unpatchify は置換・clamp は f32 の値を f32 の値へ写すだけ — 同じ入力なら丸めの差は出ない）。
  // 一部だけの組は在否のテストが落とすので、ここは 3 本揃った機だけで回す。
  ignore: !ALL_TILES,
  fn: async () => {
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
});

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

/** 有効数字 2 桁への切り上げ（段 2 / 4 の e2e の帯の候補と同じ規則）。 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(value / unit) * unit).toPrecision(2));
};

const GIB = 1024 ** 3;
const gib = (bytes: number): string => (bytes / GIB).toFixed(3);

const results = openResults("wan-ti2v-vae-tiles");

/**
 * 未導出の帯（`undefined`）の候補を出して赤で止める step（band だけから作り、受入れと故障注入の maxAbs は並べるが
 * 候補に使わない。band が maxAbs を出す前に落ちた回は候補を出さない）。帯が導出済みなら何もしない。
 */
const bandCandidate = async (
  t: Deno.TestContext,
  observed: ReadonlyMap<string, number>,
  faults: ReadonlyMap<string, number>,
): Promise<void> => {
  if (BAND !== undefined) return;
  await t.step("帯の候補（未導出）", async () => {
    const band = observed.get("band");
    const candidate = band === undefined
      ? "band が maxAbs を出す前に落ちた — 候補にしない（落ちたケースを先に調べる）"
      : band > 0
      ? `band の maxAbs ${band.toExponential(3)} × 5 = ${(band * 5).toExponential(3)}` +
        `（有効数字 2 桁へ切り上げ ${roundUpTwoDigits(band * 5)}）`
      : "band の maxAbs が 0 — × 5 で帯を導けない（照合の経路を先に調べる）";
    const format = (value: number | undefined): string => value?.toExponential(3) ?? "—";
    const accepted = ["accept", "accept-rgb", "wide"].map((label) =>
      `${label} ${format(observed.get(label))}`
    );
    const injected = [...faults].map(([label, value]) => `${label} ${format(value)}`);
    const message = `帯が未導出 — ${candidate}。受入れの maxAbs（帯の決定に使わない）: ${
      accepted.join(" / ")
    }。故障注入の maxAbs（C4 で帯の外かを判定）: ${
      injected.length === 0 ? "—" : injected.join(" / ")
    }`;
    await runRecordedCase(results, { id: "band-candidate", failureNote: () => message }, () => {
      console.log(`[wan-ti2v-vae-tiles] ${message}`);
      throw new Error(message);
    });
  });
};

Deno.test({
  name:
    "Wan2.2 TI2V VAE タイル decode: 縮退門 + 832×480・81 フレームが上流のタイル参照（unpatchify の前）と帯の中で" +
    "一致・accept の RGB が上流の unpatchify → clamp と帯の中で一致（実 GPU）",
  ignore: !ALL_TILES || !CHUNK_ASSETS_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    await assertRunningAdapter();
    const [firstOpened, nextOpened] = await Promise.all([
      openSeriesContainer(ti2vVaeModelUrl(FIRST)),
      openSeriesContainer(ti2vVaeModelUrl(NEXT)),
    ]);
    const firstModel = prepareContainer(firstOpened, seriesGraph(SERIES, FIRST));
    const nextModel = prepareContainer(nextOpened, seriesGraph(SERIES, NEXT));
    const layout = wanVaeChunkLayout(firstModel, nextModel);
    const compression = wanSpatialCompression(layout, WAN22_TI2V_GENERATION);
    const fixtures: Readonly<Record<CaseName, TileFixture>> = {
      band: await readTileFixture(ti2vVaeTileFixtureUrl("band")),
      accept: await readTileFixture(ti2vVaeTileFixtureUrl("accept")),
      wide: await readTileFixture(ti2vVaeTileFixtureUrl("wide")),
    };
    const degenerate = await readFixture(ti2vVaeFixtureUrl(DEGENERATE_CASE));
    /** ラベル（ケース名 / `accept-rgb`）→ maxAbs（帯の候補の材料 — 判定の前に積むので帯の外の回も残る）。 */
    const observed = new Map<string, number>();
    /** ケース名 → 非有限の数（門 4 — 判定の前に積む）。 */
    const nonFinite = new Map<string, number>();
    /** 故障注入のラベル → maxAbs（記録 — 判定は帯が導出された回だけ）。 */
    const faultObserved = new Map<string, number>();
    let bandNote: string | undefined;

    let deviceLost: string | undefined;
    const gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    let monitor: ReturnType<typeof monitorDrmUsage> | undefined;
    let timeline: DrmTimeline | undefined;
    let first: Session | undefined;
    let next: Session | undefined;
    let caches: WanVaeChunkCaches | undefined;
    let failure: { readonly error: unknown } | undefined;
    try {
      // 標本化は生成の時点で 1 回読む（読めない権限なら投げる）ので、try の中で作って finally で device を畳む。
      monitor = monitorDrmUsage();
      const drm = monitor;
      drm.enter("構築");
      first = await firstModel.createContainerSession(gpu);
      next = await nextModel.createContainerSession(gpu);
      caches = await WanVaeChunkCaches.create(gpu, layout);
      const sessions = { first, next };
      const liveCaches = caches;
      /**
       * `onTile` の口: 最後のタイルを除き、次のタイルの decode の前にフレームの常駐を毒値で埋め直す（タイルの出力は
       * `decodeWanVaeTile` が読み戻した後なので、決着したタイルの値は壊さない）。
       */
      const poisonBetweenTiles = (plan: WanVaeTilePlan, chunks: number) => async (tile: number) => {
        if (tile < wanVaeTileCount(plan)) await poisonFrames(liveCaches, chunks);
      };

      /**
       * 1 ケースの判定（maxAbs と非有限を記録してから — 未導出の回も候補の材料が残る）。非有限 → 帯が未導出 → 帯の
       * 外 の順で落とす。
       */
      const judge = (
        label: string,
        got: Float32Array,
        want: Float32Array,
        measurements: RecordedCase["measurements"],
        output: string,
        extra = "",
      ): void => {
        const { report, referenceMax, ratio } = compare(got, want, TOLERANCE);
        measurements.push({
          output,
          maxAbs: report.maxAbsError,
          maxRel: report.maxRelError,
          tolerance: TOLERANCE,
          stage: "karume",
        });
        observed.set(label, report.maxAbsError);
        nonFinite.set(label, report.nonFiniteCount);
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

      await t.step(
        "計画: TS の計画（資産の入力形から・2.2 の記述子）がフィクスチャのメタ（Python）と一致",
        () => {
          assertEquals(
            [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
            [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
          );
          assertWanVaeMatchesGeneration(layout, WAN22_TI2V_GENERATION, OWNER);
          for (const { name, role, chunks, height, width, full, rgb } of CASES) {
            const fixture = fixtures[name];
            const latents = fixture.tensor("latents");
            assertEquals(
              latents.shape,
              [LATENT_CHANNELS, chunks, height, width],
              `${name} の潜在の形`,
            );
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
            const theirs = Object.fromEntries(
              Object.keys(ours).map((key) => [key, fixture.meta(key)]),
            );
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
        },
      );

      await t.step(
        "縮退門: 潜在 16×16（1 枚）ではタイル経路 ≡ 非タイルの chunk 列（Uint32）",
        async () => {
          drm.enter("縮退門");
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
          const tiled = await decodeWanVaeTiled(
            gpu,
            sessions,
            liveCaches,
            plan,
            degenerate.latents,
          );
          // 毒値は各デコードの前に入れ直す（2 本目が 1 本目の書いたフレームを読み戻して一致を装う形を消す）。
          await poisonFrames(liveCaches, chunks);
          const direct = await decodeWanVaeTile(gpu, sessions, liveCaches, degenerate.latents);
          assertEquals(
            tiled.length,
            SAMPLE_CHANNELS * wanVaeFrameCount(chunks) * SAMPLE_TILE * SAMPLE_TILE,
          );
          assertEquals(countNonFinite(tiled), 0, "非有限（写されなかったフレーム）");
          assert(bitsEqual(tiled, direct), "タイル経路が chunk 列とビット一致しない");
        },
      );

      await t.step(
        "band: 832×480・81 フレーム（12 枚 × 21 chunk）が参照と帯の中で一致（フレームは毒値で始める）",
        async () => {
          await runRecordedCase(results, { id: "band" }, async ({ measurements }) => {
            const fixture = fixtures.band;
            const latents = fixture.tensor("latents");
            const plan = planFor(layout, latents.shape);
            const chunks = latents.shape[1];
            await poisonFrames(liveCaches, chunks);
            drm.enter("band");
            const started = performance.now();
            const got = await decodeWanVaeTiled(
              gpu,
              sessions,
              liveCaches,
              plan,
              latents.data,
              poisonBetweenTiles(plan, chunks),
            );
            const elapsedMs = performance.now() - started;
            const breakdown = vramBreakdown(
              sessions.first.diagnostics(),
              sessions.next.diagnostics(),
              liveCaches,
            );
            // タイルの頭の cache のゼロ化（`queue.writeBuffer` の staging — 次の submit の完了まで生きる）。
            const zeroStaging = liveCaches.cacheBytes;
            const vram = Object.fromEntries(
              Object.entries({
                ...breakdown,
                zeroStaging,
                totalWithZeroStaging: breakdown.total + zeroStaging,
              }).map(([name, bytes]) => [name, gib(bytes)]),
            );
            bandNote = `band: tiles=${wanVaeTileCount(plan)} chunks=${chunks} decodeMs=${
              elapsedMs.toFixed(0)
            } vram GiB ${JSON.stringify(vram)}（見積り ${VRAM_ESTIMATE_GIB} GiB）`;
            console.log(`[wan-ti2v-vae-tiles] ${bandNote}`);
            judge(
              "band",
              got,
              fixture.tensor("frames").data,
              measurements,
              `frames(${chunks} chunks × ${wanVaeTileCount(plan)} tiles)`,
              ` decodeMs=${elapsedMs.toFixed(0)}`,
            );
            return undefined;
          });
        },
      );

      // 受入れ: decode 済みのタイルを残し、VAE 段の末尾の照合と故障注入（ホスト側の貼り合わせ）に使い回す。
      const accept = fixtures.accept;
      const acceptLatents = accept.tensor("latents");
      const acceptWant = accept.tensor("frames").data;
      const acceptPlan = planFor(layout, acceptLatents.shape);
      const acceptChunks = acceptLatents.shape[1];
      let acceptTiles: Float32Array[] = [];
      let acceptAssembled: Float32Array<ArrayBuffer> | undefined;

      await t.step(
        "accept: 480×832・9 フレーム（別の潜在・chunk 境界・縦長）が帯の中",
        async () => {
          await runRecordedCase(results, { id: "accept" }, async ({ measurements }) => {
            await poisonFrames(liveCaches, acceptChunks);
            drm.enter("accept");
            const decodeStarted = performance.now();
            acceptTiles = await decodeWanVaeTiles(
              gpu,
              sessions,
              liveCaches,
              acceptPlan,
              acceptLatents.data,
              poisonBetweenTiles(acceptPlan, acceptChunks),
            );
            const decodeMs = performance.now() - decodeStarted;
            const assembleStarted = performance.now();
            const got = assembleWanVaeTiles(acceptTiles, acceptPlan);
            const assembleMs = performance.now() - assembleStarted;
            acceptAssembled = got;
            judge(
              "accept",
              got,
              acceptWant,
              measurements,
              `frames(${acceptChunks} chunks × ${wanVaeTileCount(acceptPlan)} tiles)`,
              ` decodeMs=${decodeMs.toFixed(0)} assembleMs=${assembleMs.toFixed(0)}`,
            );
            return undefined;
          });
        },
      );

      await t.step(
        "accept の観測: 非タイルの上流 decode（frames_full）との差（門ではない — タイル化の近似）",
        () => {
          // accept の判定（未導出の回は必ず赤）とは別の step — 判定の成否に依らず出し、ここで落ちても accept の
          // maxAbs と非有限の数は記録に残る。クランプは要素ごとなので patchify 空間のまま掛けても RGB で掛けたのと
          // 同じ値。
          assert(acceptAssembled !== undefined, "accept が貼り合わされていない");
          const full = accept.tensor("frames_full").data;
          const unclamped = compare(acceptAssembled, full, { atol: 0, rtol: 0 });
          const clampedGot = Float32Array.from(acceptAssembled);
          const clampedFull = Float32Array.from(full);
          clampWanVaeFrames(clampedGot);
          clampWanVaeFrames(clampedFull);
          const clamped = compare(clampedGot, clampedFull, { atol: 0, rtol: 0 });
          console.log(
            `[wan-ti2v-vae-tiles] accept vs untiled (observation): unclamped maxAbs=` +
              `${unclamped.report.maxAbsError} ratio=${unclamped.ratio} / clamped maxAbs=` +
              `${clamped.report.maxAbsError} ratio=${clamped.ratio}`,
          );
        },
      );

      await t.step(
        "accept の RGB: VAE 段の末尾（finishWanVaeFrames — unpatchify → clamp）が上流の unpatchify → clamp と帯の中",
        async () => {
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
            judge(
              "accept-rgb",
              video,
              accept.tensor("frames_rgb").data,
              measurements,
              `frames_rgb(${acceptChunks} chunks × ${wanVaeTileCount(acceptPlan)} tiles)`,
            );
            return Promise.resolve(undefined);
          });
        },
      );

      await t.step(
        "wide: 1280×704・5 フレーム（28 枚・対ごとに違うブレンド幅）が帯の中",
        async () => {
          await runRecordedCase(results, { id: "wide" }, async ({ measurements }) => {
            const fixture = fixtures.wide;
            const latents = fixture.tensor("latents");
            const plan = planFor(layout, latents.shape);
            const chunks = latents.shape[1];
            await poisonFrames(liveCaches, chunks);
            drm.enter("wide");
            const started = performance.now();
            const got = await decodeWanVaeTiled(
              gpu,
              sessions,
              liveCaches,
              plan,
              latents.data,
              poisonBetweenTiles(plan, chunks),
            );
            const elapsedMs = performance.now() - started;
            judge(
              "wide",
              got,
              fixture.tensor("frames").data,
              measurements,
              `frames(${chunks} chunks × ${wanVaeTileCount(plan)} tiles)`,
              ` decodeMs=${elapsedMs.toFixed(0)}`,
            );
            return undefined;
          });
        },
      );
      drm.enter("照合の後");

      const faults: readonly {
        readonly id: string;
        readonly label: string;
        readonly plan: () => WanVaeTilePlan;
      }[] = [
        {
          // 縦長でだけ値に出る（正方では対合）。枚数は 4×3 = 3×4 で揃うので形の検査を抜ける。
          id: "swap-axes",
          label: "行と列の取り違え",
          plan: () => ({ ...acceptPlan, rows: acceptPlan.cols, cols: acceptPlan.rows }),
        },
        {
          // TS と Python の丸めが割れたときの形（decode した位置と貼る位置が 1 潜在ずれる）。
          id: "shift-start",
          label: "開始位置の 1 潜在ずれ",
          plan: () => ({
            ...acceptPlan,
            rows: {
              ...acceptPlan.rows,
              starts: acceptPlan.rows.starts.map((start, index) => index === 1 ? start + 1 : start),
            },
          }),
        },
      ];
      for (const { id, label, plan } of faults) {
        await t.step(`故障注入: ${label} → accept が帯の外`, async () => {
          await runRecordedCase(results, { id: `fault/${id}` }, ({ measurements }) => {
            assert(acceptTiles.length > 0, "accept のタイルが decode されていない");
            const got = assembleWanVaeTiles(acceptTiles, plan());
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

      await t.step(
        "毒値と device lost: 全ケース（band / accept / wide）で非有限 0・device lost なし",
        () => {
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
          assertEquals(deviceLost, undefined, "device lost");
        },
      );

      await t.step("記録: VRAM の山（fdinfo）と内訳・band の壁時計（門ではない）", async () => {
        await runRecordedCase(results, { id: "record" }, () => {
          timeline = drm.stop();
          const lines = [
            ...[...timeline.peaks].map(([phase, peak]) =>
              `VRAM 山 [${phase}] ${formatDrmUsage(peak)}`
            ),
            bandNote ?? "band の観測なし（band が decode を終える前に落ちた）",
          ];
          for (const line of lines) console.log(`[wan-ti2v-vae-tiles] ${line}`);
          return Promise.resolve({ status: "pass", note: lines.join(" / ") });
        });
      });

      await bandCandidate(t, observed, faultObserved);
    } catch (error) {
      failure = { error };
      throw error;
    } finally {
      // MUST: Session → 常駐 → device の順で畳む（2.1 / 段 4 の e2e と同じ — 1 段が落ちても残りの段を必ず通し、
      // 本体の失敗を先頭の段に置いて上書きさせない）。
      await disposeSteps([
        () => {
          if (failure !== undefined) throw failure.error;
        },
        // 標本化の interval を残さない（記録の step が止める前に落ちた回）。
        () => {
          if (timeline === undefined) monitor?.stop();
        },
        () => first?.dispose(),
        () => next?.dispose(),
        () => caches?.dispose(),
        // device を捨てる前に解放を待つ（後続のテストの予算を残す — {@link settleReleases}）。
        () => settleReleases(gpu),
        () => gpu.destroy(),
      ]);
    }
  },
});
