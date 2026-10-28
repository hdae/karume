/**
 * Wan2.1 の動画 VAE の**タイル decode**の参照照合（実 GPU — ADR 0118 段 5 の検収）。
 *
 * 系列 `outputs/series/wan2.1-t2v-1.3b-f16-dyn/` の chunk グラフ 2 本（段 4）を、ホストのタイル
 * 計画（`vae-tiles.ts`）で全画面に敷き詰めて回す — タイルが外・chunk が内・1 タイル = 1 batch・
 * フレームだけを読み戻す（`decodeWanVaeTile`）。比べる相手は recipe のタイル参照
 * （`tools/export-recipes/wan/vae_tiling.py` — 同じ幾何・上流の blend の逐語・CPU f32・重みは f16 へ
 * 丸めた値 — ADR 0006）の**クランプ前**の出力。
 *
 * ## 門
 *
 * - **計画**: TS の計画（資産の入力形から導く）がフィクスチャのメタ（Python の計画）と一致する。
 * - **縮退門**: 潜在 32×32（タイル 1 枚 — 段 4 のフィクスチャの潜在）では、タイル経路
 *   （`decodeWanVaeTiled`）が非タイルの chunk 列（`decodeWanVaeTile`）と **Uint32 一致**
 *   （ADR 0033 決定 4 — 切り出しと貼り付けの恒等性を構造で固定する）。
 * - **帯**: `band`（832×480・33 フレーム・12 枚）の `atol = rtol = 0` の素の突合の実測最悪から
 *   帯を決め（{@link BAND}）、受け入れは**別の潜在・別の chunk 境界・縦長**の `accept`
 *   （480×832・9 フレーム）で判定する（ADR 0118 追記 2026-10-02）。
 * - **毒値**: フレームの常駐を NaN で埋めてから回し、非有限が 0 であること（写し忘れのフレームは
 *   NaN のまま残る）。
 * - **故障注入**（ホスト側 — `accept` の decode 済みタイルを故障ありで貼り合わせる）: 行と列の
 *   取り違え・開始位置の 1 潜在ずれ が帯の外に出る（出なければ帯が広すぎる兆候）。
 *
 * Upstream parity is opt-in (`KARUME_UPSTREAM_PARITY=1` — helpers/upstream-parity.ts): the `band` /
 * `accept` frames against the recipe's tiled reference under {@link BAND}, and in the chunk section
 * the chunk sequences against the untiled `_decode` with the cache fault injections that prove that
 * band is live. The plan, the degenerate gate, the NaN poison and the seam fault injections (the
 * host-side assembly against the tiled reference) run by default.
 *
 * 資産が 1 つも無い環境は明示 SKIP、一部だけある環境は FAIL（段 4 の e2e と同じ規律）。
 *
 * This file also holds the step-4 check (single-tile chunk sequence vs the untiled upstream
 * `_decode`, with its own band, cache fault injections and the 81-frame `long` case) — see the
 * "Chunk sequence" section below. Both checks share the series and the two chunk graphs.
 */

import { isNotFound, readFile, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  ExecutionError,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type Session,
} from "@karume/runtime";
import {
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  type WanVaeChunkLayout,
  wanVaeChunkLayout,
  wanVaeFrameCount,
} from "../../models/src/wan/vae-chunks.ts";
import {
  assembleWanVaeTiles,
  decodeWanVaeTiled,
  decodeWanVaeTiles,
  wanVaeBlendExtentAt,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../../models/src/wan/vae-tiles.ts";
import { WAN21_GENERATION } from "../../models/src/wan/descriptor.ts";
import { planWanGenerationTiles } from "../../models/src/wan/tile-decode.ts";
import { disposeSteps } from "../../models/src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  bitsEqual,
  decodeWithFault,
  type Fault,
  readFixture as readChunkFixture,
} from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * タイル decode の帯（絶対値 — 出力は値域 ±1.5 程度のクランプ前のフレーム）。
 *
 * 実測（B570・Deno 2.9.6・`atol = rtol = 0` の素の突合・2026-10-02）:
 *
 * | ケース | 役割   | 潜在              | タイル × chunk | maxAbs  | 参照の max | 比（maxAbs ÷ 参照の max） |
 * | ------ | ------ | ----------------- | -------------- | ------- | ---------- | ------------------------- |
 * | band   | 帯     | `[16,9,60,104]`   | 12 × 9         | 3.87e-6 | 1.462      | 2.65e-6                   |
 * | accept | 受入れ | `[16,3,104,60]`   | 12 × 3         | 4.16e-6 | 1.319      | 3.15e-6                   |
 *
 * 帯 2e-5 は `band` の実測最悪の約 5.2 倍（段 4 の chunk 列の帯 2.5e-5 と同じ桁 — ブレンドは凸結合
 * なので GPU の誤差を広げない）。比は事前の目安（1e-4 — 決定 8）の 1/32 以下。受入れが帯の実測
 * 最悪を少し上回る（4.16e-6 > 3.87e-6）のは潜在の違い — 帯の内側で、帯は動かさない（MUST: 受入れの
 * 結果を見て帯を決め直さない — 追記 2026-10-02）。故障注入（行と列の取り違え 2.08・開始位置の
 * 1 潜在ずれ 1.96）は帯の 5 桁外。
 */
const BAND = 2e-5;
const TOLERANCE: Tolerance = { atol: BAND, rtol: 0 };

const SERIES = "wan2.1-t2v-1.3b-f16-dyn";
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES}/`, import.meta.url);
const FIRST_COMPONENT = "vae_decoder_first";
const NEXT_COMPONENT = "vae_decoder_next";
const MODEL_FILE = "model.krm";
const CASES = ["band", "accept"] as const;
type CaseName = (typeof CASES)[number];

/** 縮退門の潜在（段 4 の chunk 列のフィクスチャ — 潜在 `[16,9,32,32]`・タイル 1 枚ちょうど）。 */
const DEGENERATE_FIXTURE = "vae_chunks.band.safetensors";

/** SKIP 時にそのまま貼れる生成コマンド（chunk グラフ 2 本 + タイル参照）。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --group wan --inexact " +
  "python -m wan.export_vae && uv run --group wan --inexact python -m wan.vae_tiling";

const modelUrl = (component: string): URL => new URL(`${component}/${MODEL_FILE}`, SERIES_ROOT);
const fixtureUrl = (name: CaseName): URL => new URL(`vae_tiles.${name}.safetensors`, SERIES_ROOT);
const degenerateUrl = new URL(DEGENERATE_FIXTURE, SERIES_ROOT);

const fileExists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const PRESENT = [
  modelPresent(modelUrl(FIRST_COMPONENT)),
  modelPresent(modelUrl(NEXT_COMPONENT)),
  fileExists(degenerateUrl),
  ...CASES.map((name) => fileExists(fixtureUrl(name))),
];
const ANY_PRESENT = PRESENT.some(Boolean);
if (!ANY_PRESENT) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に VAE の chunk グラフとタイル参照が無いため Wan の VAE タイル ` +
      `decode の照合を SKIP する。生成: ${GENERATE_COMMAND}`,
  );
}

// 1 つも無い環境は「生成していない」なので SKIP。1 つでもあるなら欠けは FAIL。
test.skipIf(!ANY_PRESENT)("Wan VAE タイル資産: 2 グラフとフィクスチャ 3 本が揃っている", () => {
  assertEquals(
    PRESENT,
    PRESENT.map(() => true),
    `${SERIES_ROOT.pathname} の欠け`,
  );
});

/** 1 本の safetensors（F32 のテンソルとメタ）。 */
type Fixture = {
  readonly tensor: (key: string) => {
    readonly data: Float32Array<ArrayBuffer>;
    readonly shape: readonly number[];
  };
  readonly has: (key: string) => boolean;
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
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
 * 計画の幾何をフィクスチャのメタ（`vae_tiling.TilePlan.meta`）と同じ綴りへ落とす（TS ↔ Python の
 * 突き合わせ — 数列は `"0,14,28"` の形）。
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
  };
};

const absMax = (values: Float32Array): number =>
  values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

/** 差の要約（比 = 最大絶対差 ÷ 参照の最大絶対値 — 決定 8 の指標）。 */
const compare = (got: Float32Array, want: Float32Array, tolerance: Tolerance) => {
  assertEquals(got.length, want.length, "要素数");
  const report = allclose(got, want, tolerance);
  const referenceMax = absMax(want);
  return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
};

/** フレームの常駐を NaN で埋める（写し忘れのフレームは NaN のまま残る — 毒値）。 */
const poisonFrames = async (caches: WanVaeChunkCaches, chunks: number): Promise<void> => {
  for (const resident of await caches.frames(chunks)) {
    resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
  }
};

const nonFiniteCount = (values: Float32Array): number =>
  values.reduce((count, value) => (Number.isFinite(value) ? count : count + 1), 0);

// ---- GPU setup shared by the default cases and the opt-in cases ------------------------------

/** The chunk layout, the two product Sessions and the resident caches one GPU case runs on. */
type VaeRun = {
  readonly gpu: GpuContext;
  readonly layout: WanVaeChunkLayout;
  readonly sessions: { readonly first: Session; readonly next: Session };
  readonly caches: WanVaeChunkCaches;
};

/**
 * Registers, in the enclosing describe, a beforeAll that opens the two chunk graphs and builds their
 * Sessions and the resident caches on one device, and an afterAll that tears them down, so every GPU
 * suite of this file shares one lifecycle and one teardown order. Returns the accessor the its read
 * the run through.
 */
const registerVaeSessions = (): (() => VaeRun) => {
  let gpu: GpuContext | undefined;
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let run: VaeRun | undefined;
  beforeAll(async () => {
    const [firstOpened, nextOpened] = await Promise.all([
      openSeriesContainer(modelUrl(FIRST_COMPONENT)),
      openSeriesContainer(modelUrl(NEXT_COMPONENT)),
    ]);
    const firstModel = prepareContainer(firstOpened, seriesGraph(SERIES, FIRST_COMPONENT));
    const nextModel = prepareContainer(nextOpened, seriesGraph(SERIES, NEXT_COMPONENT));
    const layout = wanVaeChunkLayout(firstModel, nextModel);

    const device: GpuContext = verifyRunningGpu(await acquireTestGpu());
    gpu = device;
    first = await firstModel.createContainerSession(device);
    next = await nextModel.createContainerSession(device);
    caches = await WanVaeChunkCaches.create(device, layout);
    run = { gpu: device, layout, sessions: { first, next }, caches };
  });
  afterAll(async () => {
    const device = gpu;
    // MUST: Session → 常駐 → device の順で畳む（常駐は Session の焼き込みから参照されうるので、
    // Session を先に畳んでから返す）。1 段が落ちても残りの段（device まで）を必ず通す（製品の
    // `Pipeline` の VAE の段と同じ形）。Vitest reports an afterAll failure apart from the its, so a
    // teardown failure cannot overwrite a test failure.
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
    if (run === undefined) throw new Error("VAE session setup (beforeAll) did not complete");
    return run;
  };
};

const results = openResults("wan-vae-tiles");

/**
 * 潜在 `[C,F,H,W]` のフィクスチャから計画を立てる（タイル辺と縮尺は資産の宣言から・重なりは本番と
 * 同じ導出 — `planWanGenerationTiles` を通すので、フィクスチャのメタとの突き合わせが本番の計画を縛る）。
 */
const planFor = (layout: WanVaeChunkLayout, latentShape: readonly number[]): WanVaeTilePlan => {
  assertEquals(latentShape.length, 4, `潜在の形 [${latentShape}]`);
  return planWanGenerationTiles(layout, latentShape[2], latentShape[3], WAN21_GENERATION);
};

/** The band / accept tile fixtures one tile suite reads (`vae_tiles.*`). */
type TileFixtures = { readonly band: Fixture; readonly accept: Fixture };

/** Registers a beforeAll that reads the band / accept tile fixtures; returns their accessor. */
const registerTileFixtures = (): (() => TileFixtures) => {
  let fixtures: TileFixtures | undefined;
  beforeAll(async () => {
    fixtures = {
      band: await readFixture(fixtureUrl("band")),
      accept: await readFixture(fixtureUrl("accept")),
    };
  });
  return () => {
    if (fixtures === undefined) throw new Error("tile fixture setup (beforeAll) did not complete");
    return fixtures;
  };
};

// Default run: the plan against the Python plan, the degenerate Uint32 gate, the NaN-poisoned
// multi-tile decodes and the seam fault injections. The band comparisons of the band / accept
// frames with the recipe's tiled reference (and their results.json ids) are the opt-in suite below.
describe.skipIf(!ANY_PRESENT || !GPU_AVAILABLE)(
  "Wan VAE タイル decode: 計画・縮退門・毒値・貼り合わせの故障注入（実 GPU）",
  () => {
    const tileFixtures = registerTileFixtures();
    let degenerate: ReturnType<Fixture["tensor"]> | undefined;
    beforeAll(async () => {
      degenerate = (await readFixture(degenerateUrl)).tensor("latents");
    });
    const vaeRun = registerVaeSessions();

    it("計画: TS の計画（資産の入力形から）がフィクスチャのメタ（Python）と一致", () => {
      const fixtures = tileFixtures();
      const { layout } = vaeRun();
      for (const name of CASES) {
        const fixture = fixtures[name];
        const ours = planMeta(planFor(layout, fixture.tensor("latents").shape));
        const theirs = Object.fromEntries(Object.keys(ours).map((key) => [key, fixture.meta(key)]));
        assertEquals(ours, theirs, `${name} の幾何`);
      }
      const band = planFor(layout, fixtures.band.tensor("latents").shape);
      assertEquals(wanVaeTileCount(band), 12);
      assertEquals([...band.rows.starts], [0, 14, 28]);
      assertEquals([...band.cols.starts], [0, 24, 48, 72]);
    });

    it("縮退門: 潜在 32×32（1 枚）ではタイル経路 ≡ 非タイルの chunk 列（Uint32）", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      assert(degenerate !== undefined, "縮退門の潜在が読まれていない");
      const plan = planFor(layout, degenerate.shape);
      assertEquals(wanVaeTileCount(plan), 1);
      const chunks = wanVaeChunkCount(layout, degenerate.data);
      await poisonFrames(caches, chunks);
      const tiled = await decodeWanVaeTiled(gpu, sessions, caches, plan, degenerate.data);
      const direct = await decodeWanVaeTile(gpu, sessions, caches, degenerate.data);
      assertEquals(tiled.length, 3 * wanVaeFrameCount(chunks) * 256 * 256);
      assert(tiled.every(Number.isFinite), "非有限（写されなかったフレーム）");
      assert(bitsEqual(tiled, direct), "タイル経路が chunk 列とビット一致しない");
    });

    it("band: 832×480・33 フレームのタイル decode がフレームを全て書く（フレームは毒値で始める）", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      const fixture = tileFixtures().band;
      const latents = fixture.tensor("latents");
      const want = fixture.tensor("frames");
      const plan = planFor(layout, latents.shape);
      const chunks = latents.shape[1];
      assertEquals(latents.shape, [layout.latentChannels, chunks, 60, 104]);
      assertEquals(want.shape, [3, wanVaeFrameCount(chunks), 480, 832]);
      await poisonFrames(caches, chunks);
      const got = await decodeWanVaeTiled(gpu, sessions, caches, plan, latents.data);
      assertEquals(got.length, want.data.length, "要素数");
      assertEquals(nonFiniteCount(got), 0, "band: 非有限（写されなかったフレーム）");
    });

    // 受入れ: decode 済みのタイルを残し、故障注入（ホスト側の貼り合わせ）にも使い回す。
    let acceptTiles: Float32Array<ArrayBuffer>[] = [];

    it("accept: 480×832・9 フレーム（別の潜在・chunk 境界・縦長）のタイルを貼り合わせたフレームが有限", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      const accept = tileFixtures().accept;
      const acceptLatents = accept.tensor("latents");
      const acceptWant = accept.tensor("frames").data;
      const acceptPlan = planFor(layout, acceptLatents.shape);
      acceptTiles = await decodeWanVaeTiles(gpu, sessions, caches, acceptPlan, acceptLatents.data);
      const got = assembleWanVaeTiles(acceptTiles, acceptPlan);
      assertEquals(got.length, acceptWant.length, "要素数");
      assertEquals(nonFiniteCount(got), 0, "accept: 非有限");
    });

    const faults: readonly {
      readonly label: string;
      readonly plan: (acceptPlan: WanVaeTilePlan) => WanVaeTilePlan;
    }[] = [
      {
        // 縦長でだけ値に出る（正方では対合）。枚数は 4×3 = 3×4 で揃うので形の検査を抜ける。
        label: "行と列の取り違え",
        plan: (acceptPlan) => ({ ...acceptPlan, rows: acceptPlan.cols, cols: acceptPlan.rows }),
      },
      {
        // TS と Python の丸めが割れたときの形（decode した位置と貼る位置が 1 潜在ずれる）。
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
    for (const { label, plan } of faults) {
      it(`故障注入: ${label} → accept が帯の外`, () => {
        assert(acceptTiles.length > 0, "accept のタイルが decode されていない");
        const { layout } = vaeRun();
        const accept = tileFixtures().accept;
        const acceptWant = accept.tensor("frames").data;
        const acceptPlan = planFor(layout, accept.tensor("latents").shape);
        const got = assembleWanVaeTiles(acceptTiles, plan(acceptPlan));
        const { report, ratio } = compare(got, acceptWant, TOLERANCE);
        console.log(`[wan-vae-tiles] fault ${label}: maxAbs=${report.maxAbsError} ratio=${ratio}`);
        assert(!report.pass, `${label}: 帯の中に収まった（帯が広すぎる兆候）`);
        assert(report.maxAbsError > 1e3 * BAND, `${label}: maxAbs ${report.maxAbsError}`);
      });
    }
  },
);

// Upstream parity: the band / accept frames against the recipe's tiled reference (the upstream
// blend, CPU f32) under the absolute band, so it is opt-in. It owns the band / accept results.json
// ids and their measurements.
describe.skipIf(!ANY_PRESENT || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  `Wan VAE タイル decode: 832×480・33 フレームが上流の参照と帯の中で一致（実 GPU）${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const tileFixtures = registerTileFixtures();
    const vaeRun = registerVaeSessions();

    it("band: 832×480・33 フレームが参照と帯の中で一致（フレームは毒値で始める）", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      await runRecordedCase(results, { id: "band" }, async ({ measurements }) => {
        const fixture = tileFixtures().band;
        const latents = fixture.tensor("latents");
        const want = fixture.tensor("frames");
        const plan = planFor(layout, latents.shape);
        const chunks = latents.shape[1];
        assertEquals(latents.shape, [layout.latentChannels, chunks, 60, 104]);
        assertEquals(want.shape, [3, wanVaeFrameCount(chunks), 480, 832]);
        await poisonFrames(caches, chunks);
        const got = await decodeWanVaeTiled(gpu, sessions, caches, plan, latents.data);
        const { report, referenceMax, ratio } = compare(got, want.data, TOLERANCE);
        measurements.push({
          output: `frames(${chunks} chunks × ${wanVaeTileCount(plan)} tiles)`,
          maxAbs: report.maxAbsError,
          maxRel: report.maxRelError,
          tolerance: TOLERANCE,
          stage: "karume",
        });
        console.log(
          `[wan-vae-tiles] band: tiles=${wanVaeTileCount(plan)} chunks=${chunks} ` +
            `maxAbs=${report.maxAbsError} refMax=${referenceMax} ratio=${ratio} ` +
            `nonFinite=${report.nonFiniteCount}`,
        );
        assertEquals(report.nonFiniteCount, 0, "band: 非有限（写されなかったフレーム）");
        assert(report.pass, `band: maxAbs ${report.maxAbsError} が帯 ${BAND} の外`);
        return undefined;
      });
    });

    it("accept: 480×832・9 フレーム（別の潜在・chunk 境界・縦長）が帯の中", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      await runRecordedCase(results, { id: "accept" }, async ({ measurements }) => {
        const accept = tileFixtures().accept;
        const acceptLatents = accept.tensor("latents");
        const acceptWant = accept.tensor("frames").data;
        const acceptPlan = planFor(layout, acceptLatents.shape);
        const acceptTiles = await decodeWanVaeTiles(
          gpu,
          sessions,
          caches,
          acceptPlan,
          acceptLatents.data,
        );
        const got = assembleWanVaeTiles(acceptTiles, acceptPlan);
        const { report, referenceMax, ratio } = compare(got, acceptWant, TOLERANCE);
        measurements.push({
          output: `frames(${acceptLatents.shape[1]} chunks × ${wanVaeTileCount(acceptPlan)} tiles)`,
          maxAbs: report.maxAbsError,
          maxRel: report.maxRelError,
          tolerance: TOLERANCE,
          stage: "karume",
        });
        console.log(
          `[wan-vae-tiles] accept: maxAbs=${report.maxAbsError} refMax=${referenceMax} ` +
            `ratio=${ratio} nonFinite=${report.nonFiniteCount}`,
        );
        assertEquals(report.nonFiniteCount, 0, "accept: 非有限");
        assert(report.pass, `accept: maxAbs ${report.maxAbsError} が帯 ${BAND} の外`);
        return undefined;
      });
    });
  },
);

// ---- Chunk sequence (ADR 0118 step 4) --------------------------------------------------------

/*
 * Wan2.1 の動画 VAE の **chunk 列の参照照合**（実 GPU — ADR 0118 段 4 の検収）。
 *
 * 系列 `outputs/series/wan2.1-t2v-1.3b-f16-dyn/` の chunk グラフ 2 本（`vae_decoder_first` /
 * `vae_decoder_next` — f16 席）を、常駐の因果キャッシュで first → next × (F−1) と回し
 * （`decodeWanVaeTile` — 1 タイル = 1 batch・フレームだけを読み戻す）、上流の非タイル `_decode` の
 * chunk ループ（diffusers・CPU f32・重みは f16 へ丸めた値 — ADR 0006）の**クランプ前**の出力と比べる。
 * 参照とフィクスチャは `tools/export-recipes/wan/export_vae.py` が書く（下の CHUNK_GENERATE_COMMAND）。
 *
 * ## 帯の決め方（決定 8 と追記 2026-10-02）
 *
 * - 帯は `band` ケース（潜在 9 chunk = 33 フレーム・タイル 32）の `atol = rtol = 0` の素の突合の
 *   実測最悪から決める（{@link CHUNK_BAND}）。
 * - 受け入れは**別の潜在・別の chunk 境界**の `accept` ケース（5 chunk = 17 フレーム）で判定する。
 * - `long` ケース（21 chunk = 81 フレーム — ADR 0118 段 8）も受入れで、同じ帯の内であることを門にする。
 *   81 フレームの chunk 数そのもので、cache の持ち越しで誤差が積もらないことを見る（band / accept の
 *   9 / 5 chunk の外挿に頼らない）。帯は `band` から導いた値のまま（long の結果を見て変えない）。
 * - 故障注入（cache 更新忘れ・cache の 2 フレームの逆順・タイルの頭のゼロ化の省略・chunk 2 以降に
 *   first）が `accept` で帯の外に出ることを門にする（出なければ帯が広すぎる兆候）。故障注入は
 *   製品の経路と同じ部品で組んだこのテストのループで行い、そのループが故障なしなら製品の経路と
 *   Uint32 で一致することも門にする（注入の結果が製品の経路を代表している担保）。
 *
 * 資産が 1 つも無い環境は明示 SKIP、一部だけある環境は FAIL（dacvae の e2e と同じ規律）。
 */

/**
 * chunk 列の帯（絶対値 — 出力は値域 ±1.3 程度のクランプ前のフレーム）。
 *
 * 実測（B570・Deno 2.9.6・`atol = rtol = 0` の素の突合・band / accept は 2026-10-02・long は 2026-10-03 —
 * 帯を決めた後に初めて回した）:
 *
 * | ケース | 役割   | chunk | フレーム | maxAbs  | 参照の max | 比（maxAbs ÷ 参照の max） |
 * | ------ | ------ | ----- | -------- | ------- | ---------- | ------------------------- |
 * | band   | 帯     | 9     | 33       | 4.65e-6 | 1.297      | 3.58e-6                   |
 * | accept | 受入れ | 5     | 17       | 3.55e-6 | 1.282      | 2.77e-6                   |
 * | long   | 受入れ | 21    | 81       | 4.12e-6 | 1.380      | 2.98e-6                   |
 *
 * 帯 2.5e-5 は `band` の実測最悪の約 5.4 倍。比は事前の目安（1e-4 — 決定 8）の 1/28 で、eager の
 * 書き直しの差（CPU で 2.79e-6 — recipe の `--verify`）と同じ桁 — GPU の縮約順の差はそれに
 * 埋もれる程度。chunk を 9 本・21 本重ねても 5 本と同じ桁（cache の持ち越しで誤差が積もっていない）。
 * 故障注入（cache 更新忘れ 0.545・2 フレームの逆順 1.12・ゼロ化の省略 0.918）は帯の 4 桁以上外。
 */
const CHUNK_BAND = 2.5e-5;
const CHUNK_TOLERANCE: Tolerance = { atol: CHUNK_BAND, rtol: 0 };

const CHUNK_CASES = ["band", "accept", "long"] as const;
type ChunkCaseName = (typeof CHUNK_CASES)[number];

/** `long` の chunk 数（81 フレーム = 1 + 4·20 — 受理集合の上限のフレーム数の chunk 列）。 */
const LONG_CHUNKS = 21;

/** SKIP 時にそのまま貼れる生成コマンド（`tools/export-recipes/wan/export_vae.py`）。 */
const CHUNK_GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --group wan --inexact " + "python -m wan.export_vae";

/** 故障注入で外す cache（head の conv — 1 層だけを通る最も効きの小さい席）。 */
const DROPPED_CACHE = "cache_31";

const chunkFixtureUrl = (name: ChunkCaseName): URL =>
  new URL(`vae_chunks.${name}.safetensors`, SERIES_ROOT);

const CHUNK_PRESENT = [
  modelPresent(modelUrl(FIRST_COMPONENT)),
  modelPresent(modelUrl(NEXT_COMPONENT)),
  ...CHUNK_CASES.map((name) => fileExists(chunkFixtureUrl(name))),
];
const CHUNK_ANY_PRESENT = CHUNK_PRESENT.some(Boolean);
if (!CHUNK_ANY_PRESENT) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に VAE の chunk グラフが無いため Wan の VAE chunk 列の照合を ` +
      `SKIP する。生成: ${CHUNK_GENERATE_COMMAND}`,
  );
}

// 1 つも無い環境は「生成していない」なので SKIP。1 つでもあるなら欠けは FAIL。
test.skipIf(!CHUNK_ANY_PRESENT)(
  "Wan VAE chunk 資産: 2 グラフとフィクスチャ 3 本が揃っている",
  () => {
    assertEquals(
      CHUNK_PRESENT,
      CHUNK_PRESENT.map(() => true),
      `${SERIES_ROOT.pathname} の欠け`,
    );
  },
);

const chunkResults = openResults("wan-vae-chunks");

const readChunkFixtures = async () => ({
  band: await readChunkFixture(chunkFixtureUrl("band")),
  accept: await readChunkFixture(chunkFixtureUrl("accept")),
  long: await readChunkFixture(chunkFixtureUrl("long")),
});

type ChunkFixtures = Awaited<ReturnType<typeof readChunkFixtures>>;

/** Registers a beforeAll that reads the band / accept / long chunk fixtures; returns their accessor. */
const registerChunkFixtures = (): (() => ChunkFixtures) => {
  let fixtures: ChunkFixtures | undefined;
  beforeAll(async () => {
    fixtures = await readChunkFixtures();
  });
  return () => {
    if (fixtures === undefined) throw new Error("chunk fixture setup (beforeAll) did not complete");
    return fixtures;
  };
};

// Default run: the layout declared by the assets, the cache-zeroing size, the NaN-poisoned chunk
// sequences, the Uint32 identity of the fault-injection loop with the product path and the rejected
// first-for-all enqueue. The band comparisons with the untiled upstream `_decode`, the fault
// injections that prove that band is live and their results.json ids are the opt-in suite below.
describe.skipIf(!CHUNK_ANY_PRESENT || !GPU_AVAILABLE)(
  "Wan VAE chunk 列: 取り決め・毒値・故障注入のループの Uint32 一致・first の使い回しの拒否（実 GPU）",
  () => {
    const chunkFixtures = registerChunkFixtures();
    const vaeRun = registerVaeSessions();

    it("取り決め: タイル辺と cache は資産の宣言から（first 30 本・next 32 本）", () => {
      const { layout } = vaeRun();
      assertEquals(layout.tile, 32);
      assertEquals(layout.sampleTile, 256);
      assertEquals(layout.first.caches.length, 30);
      assertEquals(layout.next.caches.length, 32);
      assertEquals(layout.first.frameShape, [3, 1, 256, 256]);
      assertEquals(layout.next.frameShape, [3, 4, 256, 256]);
      assertEquals(
        layout.next.caches.filter((name) => !layout.first.caches.includes(name)),
        ["cache_11", "cache_18"],
      );
    });

    it("observation: bytes and host time of zeroing the caches at the tile head (GPU-side clear)", () => {
      const { caches } = vaeRun();
      // Zeroing is `ResidentTensor.clear` (one clearBuffer submit per cache — no host memcpy or
      // staging). A record, not a gate — the tiled decode pays it once per tile.
      const started = performance.now();
      caches.zero();
      const hostMs = performance.now() - started;
      console.log(`[wan-vae] zero: bytes=${caches.cacheBytes} hostMs=${hostMs.toFixed(1)}`);
      assertEquals(caches.cacheBytes, 154_959_872 * 4);
    });

    for (const name of CHUNK_CASES) {
      it(`${name}: chunk 列がフレームを全て書く（フレームは毒値で始める）`, async () => {
        const { gpu, layout, sessions, caches } = vaeRun();
        const fixture = chunkFixtures()[name];
        const chunks = wanVaeChunkCount(layout, fixture.latents);
        assertEquals(fixture.frameShape, [3, wanVaeFrameCount(chunks), 256, 256]);
        if (name === "long") assertEquals(chunks, LONG_CHUNKS, "long の chunk 数");
        // 毒値: フレームの常駐を NaN で埋めてから回す（写し忘れのフレームは NaN のまま残る）。
        await poisonFrames(caches, chunks);
        const got = await decodeWanVaeTile(gpu, sessions, caches, fixture.latents);
        assertEquals(got.length, fixture.frames.length, `${name}: 要素数`);
        assertEquals(nonFiniteCount(got), 0, `${name}: 非有限（写されなかったフレーム）`);
      });
    }

    it("故障注入のループは故障なしなら製品の経路と Uint32 で一致", async () => {
      const { gpu, sessions, caches } = vaeRun();
      const fixture = chunkFixtures().accept;
      const product = await decodeWanVaeTile(gpu, sessions, caches, fixture.latents);
      const looped = await decodeWithFault(gpu, sessions, caches, fixture.latents, {
        kind: "none",
      });
      assert(bitsEqual(product, looped), "テストのループが製品の経路と違う");
    });

    it("故障注入: chunk 2 以降も first を使う → copyOutputs の大きさで enqueue が拒まれる", async () => {
      const { gpu, sessions, caches } = vaeRun();
      const fixtures = chunkFixtures();
      // first の出力フレームは [3,1,…] で、next 用のフレームの常駐（[3,4,…]）へは写せない —
      // runtime の `copyOutputs` の大きさの検査が dispatch を積む前に落とす（値を比べる前の赤）。
      await assertRejects(
        () =>
          decodeWithFault(gpu, sessions, caches, fixtures.accept.latents, {
            kind: "first-for-all",
          }),
        ExecutionError,
        "copyOutputs",
      );
    });
  },
);

// Upstream parity: the chunk sequences against the untiled upstream `_decode` under the absolute
// band, with the cache fault injections that only prove that band is live, so it is opt-in. It owns
// the band / accept / long results.json ids and their measurements.
describe.skipIf(!CHUNK_ANY_PRESENT || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  `Wan VAE chunk 列: 常駐 cache の first → next が上流の _decode と帯の中で一致（実 GPU）${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const chunkFixtures = registerChunkFixtures();
    const vaeRun = registerVaeSessions();

    const compareChunks = (name: ChunkCaseName, got: Float32Array) => {
      const fixture = chunkFixtures()[name];
      assertEquals(got.length, fixture.frames.length, `${name}: 要素数`);
      const report = allclose(got, fixture.frames, CHUNK_TOLERANCE);
      const referenceMax = fixture.frames.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
      return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
    };

    for (const name of CHUNK_CASES) {
      it(`${name}: chunk 列が上流と帯の中で一致（フレームは毒値で始める）`, async () => {
        const { gpu, layout, sessions, caches } = vaeRun();
        await runRecordedCase(chunkResults, { id: name }, async ({ measurements }) => {
          const fixture = chunkFixtures()[name];
          const chunks = wanVaeChunkCount(layout, fixture.latents);
          assertEquals(fixture.frameShape, [3, wanVaeFrameCount(chunks), 256, 256]);
          if (name === "long") assertEquals(chunks, LONG_CHUNKS, "long の chunk 数");
          // 毒値: フレームの常駐を NaN で埋めてから回す（写し忘れのフレームは NaN のまま残る）。
          await poisonFrames(caches, chunks);
          const got = await decodeWanVaeTile(gpu, sessions, caches, fixture.latents);
          const { report, referenceMax, ratio } = compareChunks(name, got);
          measurements.push({
            output: `frames(${chunks} chunks)`,
            maxAbs: report.maxAbsError,
            maxRel: report.maxRelError,
            tolerance: CHUNK_TOLERANCE,
            stage: "karume",
          });
          console.log(
            `[wan-vae] ${name}: chunks=${chunks} maxAbs=${report.maxAbsError} ` +
              `refMax=${referenceMax} ratio=${ratio} nonFinite=${report.nonFiniteCount}`,
          );
          assertEquals(report.nonFiniteCount, 0, `${name}: 非有限（写されなかったフレーム）`);
          assert(report.pass, `${name}: maxAbs ${report.maxAbsError} が帯 ${CHUNK_BAND} の外`);
          return undefined;
        });
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
        const { gpu, sessions, caches } = vaeRun();
        const fixtures = chunkFixtures();
        if (fault.kind === "skip-zero") {
          // 前のタイル（band）の cache を残したまま accept を回す。
          await decodeWanVaeTile(gpu, sessions, caches, fixtures.band.latents);
        }
        const got = await decodeWithFault(gpu, sessions, caches, fixtures.accept.latents, fault);
        const { report, ratio } = compareChunks("accept", got);
        console.log(`[wan-vae] fault ${fault.kind}: maxAbs=${report.maxAbsError} ratio=${ratio}`);
        assert(!report.pass, `${label}: 帯の中に収まった（帯が広すぎる兆候）`);
        assert(report.maxAbsError > 1e3 * CHUNK_BAND, `${label}: maxAbs ${report.maxAbsError}`);
      });
    }
  },
);
