/**
 * Wan2.2 TI2V-5B の動画 VAE の **chunk 列の参照照合**（実 GPU — ADR 0121 段 4 の検収）。
 *
 * 系列 `outputs/series/wan2.2-ti2v-5b-f16-dyn/` の chunk グラフ 2 本（`vae_decoder_first` / `vae_decoder_next` —
 * f16 席・潜在タイル 16・出口 12 ch の patchify 空間）を、製品の経路（`decodeWanVaeTile` — 2.1 と同じ関数・
 * 1 タイル = 1 batch・フレームだけを読み戻す）で first → next × (F−1) と回し、上流の非タイル `_decode` の chunk
 * ループ（diffusers・CPU f32・重みは f16 へ丸めた値）の **unpatchify とクランプの前**の出力と比べる。参照と
 * フィクスチャは `tools/export-recipes/wan/export_vae.py --model ti2v-5b` が書く（生成コマンドと表は
 * `helpers/wan-ti2v-vae.ts`）。骨格は 2.1 の `e2e_wan_vae_chunks_test.ts` と同じで、故障注入のループは共有の
 * `helpers/wan-vae-chunk-loop.ts`。
 *
 * ## 帯の決め方
 *
 * - 帯は `band` ケース（9 chunk = 33 フレーム）の素の突合（`rtol = 0`）の実測 maxAbs × 5 から決める（2.1 の
 *   `BAND` と同じ導き方 — {@link BAND}）。受入れは別の潜在・別の chunk 境界の `accept`（5 chunk = 17 フレーム）と
 *   `long`（21 chunk = 81 フレーム — cache を 20 回持ち越す長さ）で、帯の内であることを門にする。帯は long の結果を
 *   見て変えない。
 * - `undefined` = 未導出: 各ケースは maxAbs を記録して赤で止まり、最後の step が帯の候補（band の maxAbs × 5 を
 *   有効数字 2 桁へ切り上げた値）を出す（{@link bandCandidate} — 段 2 の DiT の e2e と同じ形。通る値を仮置きして
 *   検出力の無い門を作らない）。最初の実走が導いた値を、実測の表と一緒に定数へ書く。
 *
 * ## 故障注入
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
 * ## mid の attention（D = 1024）
 *
 * - 両グラフの直近 run の計画が融合 attention の 3 段（`attention_qk` / `attention_stats` / `attention_pv`）を 1 本以上
 *   積むことを門にする（ADR 0121 決定 6 は D の上限の検査が無いまま D = 1024 の実測を段 4 に残した — 値の正しさは
 *   帯が見るが、融合の経路を通ったかは帯からは分からない）。計画の census が無い run は 0 本と数えずに赤にする。
 *
 * ## 記録（門ではない — ADR 0121 段 4 の追記へ写す）
 *
 * - VRAM の内訳（重み・backing・cache・フレーム・読み戻しの staging）・融合 attention の 3 段の dispatch 本数と
 *   パイプラインキー・融合の件数・submit の統計。
 * - 1 submit の GPU 時間の最大（B570 は 1 submit 5 s で device lost）: timestamp は計測モード（`gpuTiming`）の
 *   device でしか取れず、その device では batch を開けない（runtime の `GpuContext.beginBatch`）。なので別のテストで、
 *   同じ 2 グラフを `Session.run`（chunk 1 本 = run 1 回・cache はゼロの常駐）で回して測る（値の照合はしない）。
 *   run 1 回目は実測の裏付けが無い初期のチャンク（`initialChunkSize` 本ずつ）、run 2 回目は時間予算で切ったチャンク。
 *   単発 dispatch の最大は経路に依らない。
 * - NOTE: 裏付け前の submit の最大は batch の経路の**近似**。batch の経路も最初のタイルは裏付け前のチャンクで切る
 *   （裏付けの窓は `BatchScope.finish` で閉じる — runtime の `gpu/submit.ts` の H-5 の注記）が、スケジューラは Session
 *   ごとで、詰めかけのチャンクを enqueue をまたいで持ち越すので、チャンクの切れ目が chunk 境界をまたぎ、最も重い
 *   submit の顔ぶれが run の経路と一致しない。
 * - 計測モードのテストはファイルの先頭に置く（batch の経路の照合より先に走る — 1 submit が長すぎて照合の device が
 *   落ちる回でも、その前に dispatch と submit の時間が残る）。
 *
 * 資産が揃っていない環境と GPU 無し環境は明示 SKIP。資産の完全性（一部だけある環境は FAIL）は GPU 不要のホスト
 * テスト（`wan_ti2v_vae_chunks_host_test.ts`）が見る。
 */

import { assert, assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import {
  ExecutionError,
  prepareContainer,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import {
  decodeWanVaeTile,
  WAN_VAE_LATENT_INPUT,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeChunkLayout,
  wanVaeFrameCount,
  wanVaeLatentChunk,
} from "../src/wan/vae-chunks.ts";
import { disposeSteps } from "../src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { lookupTimestampUnitNs } from "./helpers/timestamp-unit.ts";
import {
  bitsEqual,
  decodeWithFault,
  type Fault,
  readFixture,
  vramBreakdown,
} from "./helpers/wan-vae-chunk-loop.ts";
import {
  ti2vVaeAssets,
  type Ti2vVaeCase,
  ti2vVaeFixtureUrl,
  ti2vVaeModelUrl,
  WAN_TI2V_VAE_CACHE_BYTES as CACHE_BYTES,
  WAN_TI2V_VAE_CACHE_TABLE as CACHE_TABLE,
  WAN_TI2V_VAE_CASES as CASES,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_GENERATE as GENERATE,
  WAN_TI2V_VAE_LATENT_CHANNELS as LATENT_CHANNELS,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_ROOT as ROOT,
  WAN_TI2V_VAE_SAMPLE_CHANNELS as SAMPLE_CHANNELS,
  WAN_TI2V_VAE_SAMPLE_TILE as SAMPLE_TILE,
  WAN_TI2V_VAE_SERIES as SERIES,
  WAN_TI2V_VAE_TILE as TILE,
  WAN_TI2V_VAE_TIME_CONV_CACHES as TIME_CONV_CACHES,
} from "./helpers/wan-ti2v-vae.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import {
  assertAdapterMatchesEnvironment,
  assertRunningAdapter,
  ENVIRONMENT,
} from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/**
 * chunk 列の帯（絶対値 — 出力はクランプ前の patchify 空間のフレーム）。`undefined` = 未導出（モジュール doc
 * 「帯の決め方」）。
 *
 * MUST: 受入れ（accept / long）の結果を見てこの値も決定用のケースも変えない。受入れが帯を外れたら、帯を広げずに
 * 原因を調べる。
 */
const BAND: number | undefined = undefined;
/** 判定と記録の帯（未導出の回は無限の帯として記録し、判定は赤にする）。 */
const TOLERANCE: Tolerance = { atol: BAND ?? Number.POSITIVE_INFINITY, rtol: 0 };

type CaseName = Ti2vVaeCase["name"];

/** 故障注入で外す cache（head の conv — 1 層だけを通る最も効きの小さい席・2.1 と同じ位置）。 */
const DROPPED_CACHE = "cache_31";

/** 記録する融合 attention の 3 段（ADR 0023 — ①QK・②行統計・③PV のパイプラインキーの接頭辞）。 */
const ATTENTION_KERNELS = ["attention_qk", "attention_stats", "attention_pv"] as const;

const ASSETS_PRESENT = ti2vVaeAssets().every(({ present }) => present);
if (!ASSETS_PRESENT) {
  console.warn(
    `[karume] ${ROOT.pathname} に Wan2.2 TI2V の VAE の chunk グラフとフィクスチャが揃っていないため、` +
      `chunk 列の照合を SKIP する（欠けの判定はホストテスト）。生成: ${GENERATE}`,
  );
}

const results = openResults("wan-ti2v-vae-chunks");

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

/** 有効数字 2 桁への切り上げ（段 2 の DiT の e2e の帯の候補と同じ規則）。 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(value / unit) * unit).toPrecision(2));
};

/**
 * 未導出の帯（`undefined`）の候補を出して赤で止める step（band だけから作り、受入れの maxAbs は並べるが候補に
 * 使わない。band が maxAbs を出す前に落ちた回は候補を出さない）。帯が導出済みなら何もしない。
 */
const bandCandidate = async (
  t: Deno.TestContext,
  observed: ReadonlyMap<CaseName, number>,
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
    const accepted = CASES.filter(({ role }) => role === "accept")
      .map(({ name }) => `${name} ${observed.get(name)?.toExponential(3) ?? "—"}`);
    const message = `帯が未導出 — ${candidate}。受入れの maxAbs（帯の決定に使わない）: ${
      accepted.join(" / ")
    }`;
    await runRecordedCase(results, { id: "band-candidate", failureNote: () => message }, () => {
      console.log(`[wan-ti2v-vae] ${message}`);
      throw new Error(message);
    });
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
    pipelines.filter(({ key }) => key.split(":")[0] === kernel)
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

/** submit の統計（計測なしの device で読める席 — 1 submit の GPU 時間は計測モードのテスト）。 */
const submitStats = (diagnostics: SessionDiagnostics): string => {
  const { submit } = diagnostics;
  const budget = submit.chunkBudget;
  return [
    `submit ${submit.submitCount} 本・dispatch ${submit.dispatchCount} 本・窓 ${submit.measuredCount} 本`,
    `窓平均の最大 ${budget.maxWindowMeanMs?.toFixed(1) ?? "—"} ms`,
    `推定の最大 ${budget.maxEstimatedMs?.toFixed(1) ?? "—"} ms`,
    `予算超過 ${budget.overBudgetChunks} 本（予算 ${budget.budgetMs} ms）`,
  ].join("・");
};

/**
 * 計測モードの 1 submit の GPU 時間の記録（timestamp の単位は環境キーの表から — 表に無い環境は換算を推測せず
 * 生の値で出す）。
 */
const submitGpuNote = (diagnostics: SessionDiagnostics, unitNs: number | undefined): string => {
  const observed = diagnostics.submit.chunkBudget.submitGpuTime;
  if (observed === undefined) throw new Error("計測モードなのに submit の GPU 時間が無い");
  const time = (value: number): string =>
    unitNs === undefined
      ? `${value} tick（換算の表に無い環境）`
      : `${(value * unitNs / 1e6).toFixed(1)} ms`;
  return [
    `1 submit の GPU 時間の最大 ${time(observed.maxNs)}（${observed.submits} 本）`,
    `裏付け前 ${observed.unbackedSubmits} 本の最大 ${
      observed.maxUnbackedNs === undefined ? "—" : time(observed.maxUnbackedNs)
    }`,
    `単発 dispatch の最大 ${time(observed.maxDispatchNs)}（${observed.maxDispatchKey ?? "—"}）`,
    `submit ${diagnostics.submit.submitCount} 本・dispatch ${diagnostics.submit.dispatchCount} 本`,
  ].join("・");
};

// 照合のテストより先に置く（モジュール doc「記録」の最後の項）。
Deno.test({
  name:
    "Wan2.2 TI2V VAE chunk グラフ（実 GPU・計測モード）: first / next の 1 submit の GPU 時間の最大と単発 dispatch の" +
    "最大を記録する（門ではない・値の照合はしない）",
  ignore: !ASSETS_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    const models = await prepareModels();
    const layout = wanVaeChunkLayout(models.first, models.next);
    // 入力は band の chunk 0（時間は値にほぼ依らない — 実運用の値域の潜在にしておく）。
    const latent = wanVaeLatentChunk(
      layout,
      (await readFixture(ti2vVaeFixtureUrl("band"))).latents,
      0,
    );
    let deviceLost: string | undefined;
    // 計測モード（timestamp-query）: batch は開けないので run で回す（モジュール doc「記録」）。
    const gpu = await acquireTestGpu({
      gpuTiming: true,
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    let caches: WanVaeChunkCaches | undefined;
    let failure: { readonly error: unknown } | undefined;
    try {
      assertAdapterMatchesEnvironment(gpu);
      // cache の常駐（WebGPU の新しいバッファはゼロで始まる）を両グラフの run の入力に束ねる。
      caches = await WanVaeChunkCaches.create(gpu, layout);
      const liveCaches = caches;
      const unitNs = lookupTimestampUnitNs(ENVIRONMENT.key);
      for (const graph of ["first", "next"] as const) {
        await t.step(
          `${graph}: run 2 回（1 回目 = 裏付け前のチャンク・2 回目 = 時間予算で切ったチャンク）`,
          async () => {
            /** 測れた値（測る前に落ちた回は失敗の記録に原因を書く）。 */
            let note: string | undefined;
            await runRecordedCase(
              results,
              { id: `submit/${graph}`, failureNote: (cause) => note ?? String(cause) },
              async () => {
                // Session は 1 本ずつ張る（2 本の重みを計測モードの device に同時に載せない）。
                const session = await models[graph].createContainerSession(gpu);
                let runFailure: { readonly error: unknown } | undefined;
                try {
                  const walls: string[] = [];
                  for (let index = 0; index < 2; index += 1) {
                    const started = performance.now();
                    await session.run({
                      [WAN_VAE_LATENT_INPUT]: latent,
                      ...liveCaches.inputs(layout[graph]),
                    });
                    walls.push(`${((performance.now() - started) / 1000).toFixed(2)} s`);
                    // 次の run の確保の前に、この run の中間と読み戻しの解放を待つ。
                    await settleReleases(gpu);
                  }
                  const diagnostics = session.diagnostics();
                  const measured = `${submitGpuNote(diagnostics, unitNs)}・壁 ${walls.join(" / ")}`;
                  note = measured;
                  console.log(`[wan-ti2v-vae] 計測モード ${graph}: ${measured}`);
                  const observed = diagnostics.submit.chunkBudget.submitGpuTime;
                  assert(
                    observed !== undefined && observed.unbackedSubmits > 0,
                    "1 回目の run の裏付け前のチャンク（initialChunkSize で据え置いた submit）を測れていない",
                  );
                  assertEquals(deviceLost, undefined, "device lost");
                  return { status: "pass", note: measured };
                } catch (error) {
                  runFailure = { error };
                  throw error;
                } finally {
                  // 畳む失敗で本体の失敗（例: device lost）を上書きしない — 本体の失敗を先頭の段で投げ直す。
                  await disposeSteps([
                    () => {
                      if (runFailure !== undefined) throw runFailure.error;
                    },
                    () => session.dispose(),
                    () => settleReleases(gpu),
                  ]);
                }
              },
            );
          },
        );
      }
    } catch (error) {
      failure = { error };
      throw error;
    } finally {
      await disposeSteps([
        () => {
          if (failure !== undefined) throw failure.error;
        },
        () => caches?.dispose(),
        () => settleReleases(gpu),
        () => gpu.destroy(),
      ]);
    }
  },
});

Deno.test({
  name:
    "Wan2.2 TI2V VAE chunk 列: 常駐 cache の first → next が上流の _decode（unpatchify の前）と帯の中で一致（実 GPU）",
  ignore: !ASSETS_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    await assertRunningAdapter();
    const models = await prepareModels();
    const layout = wanVaeChunkLayout(models.first, models.next);
    const fixtures = {
      band: await readFixture(ti2vVaeFixtureUrl("band")),
      accept: await readFixture(ti2vVaeFixtureUrl("accept")),
      long: await readFixture(ti2vVaeFixtureUrl("long")),
    };
    /** ケースごとの maxAbs（帯の候補の材料 — 判定の前に積むので帯の外の回も残る）。 */
    const observed = new Map<CaseName, number>();

    const gpu = await acquireTestGpu();
    let first: Session | undefined;
    let next: Session | undefined;
    let caches: WanVaeChunkCaches | undefined;
    let failure: { readonly error: unknown } | undefined;
    try {
      first = await models.first.createContainerSession(gpu);
      next = await models.next.createContainerSession(gpu);
      caches = await WanVaeChunkCaches.create(gpu, layout);
      const sessions = { first, next };
      const liveCaches = caches;

      const compare = (name: CaseName, got: Float32Array) => {
        const fixture = fixtures[name];
        assertEquals(got.length, fixture.frames.length, `${name}: 要素数`);
        const report = allclose(got, fixture.frames, TOLERANCE);
        const referenceMax = fixture.frames.reduce(
          (max, value) => Math.max(max, Math.abs(value)),
          0,
        );
        return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
      };
      /** 毒値: chunk 0..chunks−1 のフレームの常駐を NaN で埋める（写し忘れのフレームは NaN のまま残る）。 */
      const poisonFrames = async (chunks: number): Promise<void> => {
        for (const resident of await liveCaches.frames(chunks)) {
          resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
        }
      };

      await t.step(
        "取り決め: 潜在 48 ch・タイル 16・出口 12 ch・cache は表の 32 本（first は time_conv の 2 本を除く 30 本）",
        () => {
          assertEquals(
            [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
            [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
          );
          assertEquals(layout.next.caches, CACHE_TABLE.map(({ name }) => name));
          assertEquals(
            layout.next.caches.filter((name) => !layout.first.caches.includes(name)),
            [...TIME_CONV_CACHES],
          );
          assertEquals(layout.first.frameShape, [SAMPLE_CHANNELS, 1, SAMPLE_TILE, SAMPLE_TILE]);
          assertEquals(layout.next.frameShape, [SAMPLE_CHANNELS, 4, SAMPLE_TILE, SAMPLE_TILE]);
        },
      );

      await t.step("観測: タイルの頭の cache のゼロ化（ホストからの書き込み）の量と時間", () => {
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

      for (const { name, chunks: expectedChunks } of CASES) {
        await t.step(
          `${name}: chunk 列が上流と帯の中で一致（フレームは毒値で始める）`,
          async () => {
            await runRecordedCase(results, { id: name }, async ({ measurements }) => {
              const fixture = fixtures[name];
              const chunks = wanVaeChunkCount(layout, fixture.latents);
              assertEquals(chunks, expectedChunks, `${name} の chunk 数`);
              assertEquals(
                fixture.frameShape,
                [SAMPLE_CHANNELS, wanVaeFrameCount(chunks), SAMPLE_TILE, SAMPLE_TILE],
              );
              await poisonFrames(chunks);
              const started = performance.now();
              const got = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
              const elapsedMs = performance.now() - started;
              const { report, referenceMax, ratio } = compare(name, got);
              measurements.push({
                output: `frames(${chunks} chunks)`,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: TOLERANCE,
                stage: "karume",
              });
              console.log(
                `[wan-ti2v-vae] ${name}: chunks=${chunks} maxAbs=${report.maxAbsError} ` +
                  `refMax=${referenceMax} ratio=${ratio} nonFinite=${report.nonFiniteCount} ` +
                  `decodeMs=${elapsedMs.toFixed(0)}`,
              );
              assertEquals(report.nonFiniteCount, 0, `${name}: 非有限（写されなかったフレーム）`);
              observed.set(name, report.maxAbsError);
              if (BAND === undefined) {
                throw new Error(
                  `${name}: 帯が未導出（maxAbs ${report.maxAbsError}）— band の maxAbs × 5 を帯の定数へ` +
                    "書く（帯の候補は最後の step が出す）",
                );
              }
              assert(report.pass, `${name}: maxAbs ${report.maxAbsError} が帯 ${BAND} の外`);
              return undefined;
            });
          },
        );
      }

      await t.step("mid の attention（D = 1024）は両グラフで融合 attention の 3 段を通る", () => {
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

      await t.step(
        "記録: VRAM の内訳・mid の attention（D = 1024）の pipeline・融合・submit の統計（門ではない）",
        async () => {
          await runRecordedCase(results, { id: "record" }, () => {
            // 直近のタイル（long — 21 chunk・フレームの常駐が最大）の後の値。
            const diagnostics = {
              first: sessions.first.diagnostics(),
              next: sessions.next.diagnostics(),
            };
            const lines = [
              `vram ${
                JSON.stringify(vramBreakdown(diagnostics.first, diagnostics.next, liveCaches))
              }`,
              ...Object.entries(diagnostics).flatMap(([graph, stats]) => [
                `attention ${graph} ${JSON.stringify(attentionPipelines(graph, stats))}`,
                `fusions ${graph} ${JSON.stringify(stats.lastRunFusions)}`,
                `submit ${graph} ${submitStats(stats)}`,
              ]),
            ];
            for (const line of lines) console.log(`[wan-ti2v-vae] ${line}`);
            return Promise.resolve({ status: "pass", note: lines.join(" / ") });
          });
        },
      );

      await t.step("故障注入のループは故障なしなら製品の経路と Uint32 で一致", async () => {
        const fixture = fixtures.accept;
        const chunks = wanVaeChunkCount(layout, fixture.latents);
        // 毒値は各デコードの前に入れ直す（2 本目が 1 本目の書いたフレームを読み戻して一致を装う形を消す）。
        await poisonFrames(chunks);
        const product = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
        await poisonFrames(chunks);
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

      const faults: readonly { readonly label: string; readonly fault: Fault }[] = [
        {
          label: "cache 更新忘れ（copyOutputs から 1 本外す）",
          fault: { kind: "drop-cache", cache: DROPPED_CACHE },
        },
        { label: "cache の 2 フレームの順を逆にする", fault: { kind: "swap-cache-frames" } },
        { label: "タイルの頭で cache をゼロに戻さない", fault: { kind: "skip-zero" } },
      ];
      for (const { label, fault } of faults) {
        await t.step(`故障注入: ${label} → accept が帯の外`, async () => {
          await runRecordedCase(
            results,
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
              const { report, ratio } = compare("accept", got);
              measurements.push({
                output: `frames(fault ${fault.kind})`,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: TOLERANCE,
                stage: "karume",
              });
              console.log(
                `[wan-ti2v-vae] fault ${fault.kind}: maxAbs=${report.maxAbsError} ratio=${ratio}` +
                  (BAND === undefined ? "" : ` band×${(report.maxAbsError / BAND).toFixed(1)}`),
              );
              if (BAND === undefined) {
                throw new Error(`${label}: 帯が未導出（故障注入の maxAbs ${report.maxAbsError}）`);
              }
              assert(!report.pass, `${label}: 帯 ${BAND} の中に収まった（帯が広すぎる兆候）`);
              return undefined;
            },
          );
        });
      }

      await t.step(
        "故障注入: chunk 2 以降も first を使う → copyOutputs の大きさで enqueue が拒まれる",
        async () => {
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
        },
      );

      await bandCandidate(t, observed);
    } catch (error) {
      failure = { error };
      throw error;
    } finally {
      // MUST: Session → 常駐 → device の順で畳む（2.1 の e2e と同じ順・同じ理由 — 常駐は Session の焼き込みから
      // 参照されうるので Session を先に畳む。1 段が落ちても残りの段を必ず通し、畳む失敗で本体の失敗を上書きしない）。
      await disposeSteps([
        () => {
          if (failure !== undefined) throw failure.error;
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
