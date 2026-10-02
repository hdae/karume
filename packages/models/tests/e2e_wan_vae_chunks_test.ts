/**
 * Wan2.1 の動画 VAE の **chunk 列の参照照合**（実 GPU — ADR 0118 段 4 の検収）。
 *
 * 系列 `outputs/series/wan2.1-t2v-1.3b-f16-dyn/` の chunk グラフ 2 本（`vae_decoder_first` /
 * `vae_decoder_next` — f16 席）を、常駐の因果キャッシュで first → next × (F−1) と回し
 * （`decodeWanVaeTile` — 1 タイル = 1 batch・フレームだけを読み戻す）、上流の非タイル `_decode` の
 * chunk ループ（diffusers・CPU f32・重みは f16 へ丸めた値 — ADR 0006）の**クランプ前**の出力と比べる。
 * 参照とフィクスチャは `tools/export-recipes/wan/export_vae.py` が書く（下の GENERATE_COMMAND）。
 *
 * ## 帯の決め方（決定 8 と追記 2026-10-02）
 *
 * - 帯は `band` ケース（潜在 9 chunk = 33 フレーム・タイル 32）の `atol = rtol = 0` の素の突合の
 *   実測最悪から決める（{@link BAND}）。
 * - 受け入れは**別の潜在・別の chunk 境界**の `accept` ケース（5 chunk = 17 フレーム）で判定する。
 * - 故障注入（cache 更新忘れ・cache の 2 フレームの逆順・タイルの頭のゼロ化の省略・chunk 2 以降に
 *   first）が `accept` で帯の外に出ることを門にする（出なければ帯が広すぎる兆候）。故障注入は
 *   製品の経路と同じ部品で組んだこのテストのループで行い、そのループが故障なしなら製品の経路と
 *   Uint32 で一致することも門にする（注入の結果が製品の経路を代表している担保）。
 *
 * 資産が 1 つも無い環境は明示 SKIP、一部だけある環境は FAIL（dacvae の e2e と同じ規律）。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  acquireGpu,
  ExecutionError,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type ResidentTensor,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import {
  concatWanVaeFrames,
  decodeWanVaeTile,
  enqueueWanVaeChunk,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeChunkLayout,
  wanVaeFrameCount,
  wanVaeLatentChunk,
} from "../src/wan/vae-chunks.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/**
 * chunk 列の帯（絶対値 — 出力は値域 ±1.3 程度のクランプ前のフレーム）。
 *
 * 実測（B570・Deno 2.9.6・`atol = rtol = 0` の素の突合・2026-10-02）:
 *
 * | ケース | 役割   | chunk | フレーム | maxAbs  | 参照の max | 比（maxAbs ÷ 参照の max） |
 * | ------ | ------ | ----- | -------- | ------- | ---------- | ------------------------- |
 * | band   | 帯     | 9     | 33       | 4.65e-6 | 1.297      | 3.58e-6                   |
 * | accept | 受入れ | 5     | 17       | 3.55e-6 | 1.282      | 2.77e-6                   |
 *
 * 帯 2.5e-5 は `band` の実測最悪の約 5.4 倍。比は事前の目安（1e-4 — 決定 8）の 1/28 で、eager の
 * 書き直しの差（CPU で 2.79e-6 — recipe の `--verify`）と同じ桁 — GPU の縮約順の差はそれに
 * 埋もれる程度。chunk を 9 本重ねても 5 本と同じ桁（cache の持ち越しで誤差が積もっていない）。
 * 故障注入（cache 更新忘れ 0.545・2 フレームの逆順 1.12・ゼロ化の省略 0.918）は帯の 4 桁以上外。
 */
const BAND = 2.5e-5;
const TOLERANCE: Tolerance = { atol: BAND, rtol: 0 };

const SERIES = "wan2.1-t2v-1.3b-f16-dyn";
const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES}/`, import.meta.url);
const FIRST_COMPONENT = "vae_decoder_first";
const NEXT_COMPONENT = "vae_decoder_next";
const MODEL_FILE = "model.krm";
const CASES = ["band", "accept"] as const;
type CaseName = (typeof CASES)[number];

/** SKIP 時にそのまま貼れる生成コマンド（`tools/export-recipes/wan/export_vae.py`）。 */
const GENERATE_COMMAND = "cd tools/export-recipes && uv run --group wan --inexact " +
  "python -m wan.export_vae";

/** 故障注入で外す cache（head の conv — 1 層だけを通る最も効きの小さい席）。 */
const DROPPED_CACHE = "cache_31";

const modelUrl = (component: string): URL => new URL(`${component}/${MODEL_FILE}`, SERIES_ROOT);
const fixtureUrl = (name: CaseName): URL => new URL(`vae_chunks.${name}.safetensors`, SERIES_ROOT);

const fileExists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const PRESENT = [
  modelPresent(modelUrl(FIRST_COMPONENT)),
  modelPresent(modelUrl(NEXT_COMPONENT)),
  ...CASES.map((name) => fileExists(fixtureUrl(name))),
];
const ANY_PRESENT = PRESENT.some(Boolean);
if (!ANY_PRESENT) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に VAE の chunk グラフが無いため Wan の VAE chunk 列の照合を ` +
      `SKIP する。生成: ${GENERATE_COMMAND}`,
  );
}

Deno.test({
  name: "Wan VAE chunk 資産: 2 グラフとフィクスチャ 2 本が揃っている",
  // 1 つも無い環境は「生成していない」なので SKIP。1 つでもあるなら欠けは FAIL。
  ignore: !ANY_PRESENT,
  fn: () => {
    assertEquals(PRESENT, PRESENT.map(() => true), `${SERIES_ROOT.pathname} の欠け`);
  },
});

/** chunk 列のフィクスチャ（逆正規化済みの潜在 `[16,F,t,t]` と上流のクランプ前の出力）。 */
type Fixture = {
  readonly latents: Float32Array<ArrayBuffer>;
  readonly frames: Float32Array<ArrayBuffer>;
  readonly frameShape: readonly number[];
};

const readFixture = async (name: CaseName): Promise<Fixture> => {
  const bytes = await Deno.readFile(fixtureUrl(name));
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (key: string): Float32Array<ArrayBuffer> => {
    const tensor = file.tensors.get(key);
    assert(tensor !== undefined && tensor.dtype === "F32", `${name}: '${key}' が F32 で無い`);
    return new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4);
  };
  const frames = file.tensors.get("frames");
  assert(frames !== undefined, `${name}: 'frames' が無い`);
  return { latents: view("latents"), frames: view("frames"), frameShape: frames.shape };
};

/** 故障注入の種類（無し = 製品の経路と同じ）。 */
type Fault =
  | { readonly kind: "none" }
  | { readonly kind: "skip-zero" }
  | { readonly kind: "drop-cache"; readonly cache: string }
  | { readonly kind: "first-for-all" }
  | { readonly kind: "swap-cache-frames" };

/**
 * 製品の経路（`decodeWanVaeTile`）と同じ部品で組んだ chunk ループ（故障注入の口）。
 *
 * `swap-cache-frames` だけは first の後で区間を閉じ、cache の 2 フレームをホストで入れ替えてから
 * 残りを別の区間で回す（区間の切れ目は値を変えない — 故障なしの一致の門は 1 区間の形で見る）。
 */
const decodeWithFault = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  latents: Float32Array,
  fault: Fault,
): Promise<Float32Array<ArrayBuffer>> => {
  const { layout } = caches;
  const chunks = wanVaeChunkCount(layout, latents);
  const frames = await caches.frames(chunks);
  if (fault.kind !== "skip-zero") caches.zero();
  const read: Record<string, ArrayBuffer> = {};
  const runRange = async (from: number, to: number): Promise<void> => {
    const batch = await gpu.beginBatch();
    try {
      for (let index = from; index < to; index += 1) {
        const head = index === 0 || fault.kind === "first-for-all";
        const graph = head ? layout.first : layout.next;
        const session = head ? sessions.first : sessions.next;
        const latent = wanVaeLatentChunk(layout, latents, index);
        if (fault.kind === "drop-cache" && !head) {
          const copyOutputs = caches.copyOutputs(graph, frames[index]);
          const output = graph.cacheOutputs[graph.caches.indexOf(fault.cache)];
          delete copyOutputs[output];
          await session.enqueue(
            { latent, ...caches.inputs(graph) },
            { batch, copyOutputs },
          );
        } else {
          await enqueueWanVaeChunk(batch, session, graph, caches, latent, frames[index]);
        }
      }
    } catch (cause) {
      await batch.finish().catch(() => undefined);
      throw cause;
    }
    const range = frames.slice(from, to);
    Object.assign(
      read,
      await batch.finishAndRead(
        Object.fromEntries(range.map((resident, offset) => [`frame${from + offset}`, resident])),
      ),
    );
  };
  if (fault.kind === "swap-cache-frames") {
    await runRange(0, 1);
    for (const [name, resident] of Object.entries(caches.inputs(layout.next))) {
      const shape = layout.cacheShapes.get(name);
      assert(shape !== undefined, `cache '${name}' の形が layout に無い`);
      await swapCacheFrames(resident, shape[2] * shape[3]);
    }
    await runRange(1, chunks);
  } else {
    await runRange(0, chunks);
  }
  return concatWanVaeFrames(layout, frames.map((_, index) => read[`frame${index}`]));
};

/**
 * cache `[C,2,h,w]` の 2 フレームの順をホストで入れ替える（故障注入）。行優先なのでチャネル c の
 * 2 フレームは連続した `2·plane`（plane = h·w）。
 */
const swapCacheFrames = async (resident: ResidentTensor, plane: number): Promise<void> => {
  const values = new Float32Array(await resident.read());
  const swapped = new Float32Array(values.length);
  for (let offset = 0; offset < values.length; offset += 2 * plane) {
    swapped.set(values.subarray(offset + plane, offset + 2 * plane), offset);
    swapped.set(values.subarray(offset, offset + plane), offset + plane);
  }
  resident.write(swapped);
};

/** VRAM の内訳（診断から — 生きている確保の和）。 */
const vramBreakdown = (
  first: SessionDiagnostics,
  next: SessionDiagnostics,
  caches: WanVaeChunkCaches,
): Record<string, number> => {
  const session = (diagnostics: SessionDiagnostics): number =>
    diagnostics.weights.allocatedBytes + diagnostics.planBacking.residentBytes +
    diagnostics.planBacking.inputBytes;
  const parts = {
    firstWeights: first.weights.allocatedBytes,
    firstBacking: first.planBacking.residentBytes + first.planBacking.inputBytes,
    nextWeights: next.weights.allocatedBytes,
    nextBacking: next.planBacking.residentBytes + next.planBacking.inputBytes,
    caches: caches.cacheBytes,
    frames: caches.frameBytes,
    // finishAndRead の staging（フレームの合計 — 決着の間だけ生きる）。
    readbackStaging: caches.frameBytes,
  };
  return {
    ...parts,
    total: session(first) + session(next) + 2 * caches.frameBytes + caches.cacheBytes,
  };
};

const results = openResults("wan-vae-chunks");

const bitsEqual = (a: Float32Array, b: Float32Array): boolean => {
  const left = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const right = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return left.length === right.length && left.every((value, index) => value === right[index]);
};

Deno.test({
  name: "Wan VAE chunk 列: 常駐 cache の first → next が上流の _decode と帯の中で一致（実 GPU）",
  ignore: !ANY_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    await assertRunningAdapter();
    const [firstOpened, nextOpened] = await Promise.all([
      openSeriesContainer(modelUrl(FIRST_COMPONENT)),
      openSeriesContainer(modelUrl(NEXT_COMPONENT)),
    ]);
    const firstModel = prepareContainer(firstOpened, seriesGraph(SERIES, FIRST_COMPONENT));
    const nextModel = prepareContainer(nextOpened, seriesGraph(SERIES, NEXT_COMPONENT));
    const layout = wanVaeChunkLayout(firstModel, nextModel);
    const fixtures = {
      band: await readFixture("band"),
      accept: await readFixture("accept"),
    };

    const gpu = await acquireGpu();
    let first: Session | undefined;
    let next: Session | undefined;
    let caches: WanVaeChunkCaches | undefined;
    try {
      first = await firstModel.createContainerSession(gpu);
      next = await nextModel.createContainerSession(gpu);
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

      await t.step("取り決め: タイル辺と cache は資産の宣言から（first 30 本・next 32 本）", () => {
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

      await t.step("観測: タイルの頭の cache のゼロ化（ホストからの書き込み）の量と時間", () => {
        // GPU 側でゼロを書く口（clearBuffer）が runtime に無いので、ゼロ化は `queue.writeBuffer`
        // （ホストの memcpy + staging）になる。門ではなく記録 — 段 5 で 1 生成あたりタイル数ぶん払う。
        const started = performance.now();
        liveCaches.zero();
        const hostMs = performance.now() - started;
        console.log(`[wan-vae] zero: bytes=${liveCaches.cacheBytes} hostMs=${hostMs.toFixed(1)}`);
        assertEquals(liveCaches.cacheBytes, 154_959_872 * 4);
      });

      for (const name of CASES) {
        await t.step(
          `${name}: chunk 列が上流と帯の中で一致（フレームは毒値で始める）`,
          async () => {
            await runRecordedCase(results, { id: name }, async ({ measurements }) => {
              const fixture = fixtures[name];
              const chunks = wanVaeChunkCount(layout, fixture.latents);
              assertEquals(fixture.frameShape, [3, wanVaeFrameCount(chunks), 256, 256]);
              // 毒値: フレームの常駐を NaN で埋めてから回す（写し忘れのフレームは NaN のまま残る）。
              for (const resident of await liveCaches.frames(chunks)) {
                resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
              }
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
                `[wan-vae] ${name}: chunks=${chunks} maxAbs=${report.maxAbsError} ` +
                  `refMax=${referenceMax} ratio=${ratio} nonFinite=${report.nonFiniteCount} ` +
                  `decodeMs=${elapsedMs.toFixed(0)}`,
              );
              if (name === "band") {
                const vram = vramBreakdown(
                  sessions.first.diagnostics(),
                  sessions.next.diagnostics(),
                  liveCaches,
                );
                console.log(`[wan-vae] vram ${JSON.stringify(vram)}`);
                // 観測（門ではない）: nearest ×2 の reshape / expand が融合ルール upsample2x に
                // 乗るか（決定 2）。
                for (const [graph, session] of Object.entries(sessions)) {
                  const fusions = session.diagnostics().lastRunFusions;
                  console.log(`[wan-vae] fusions ${graph} ${JSON.stringify(fusions)}`);
                }
              }
              assertEquals(report.nonFiniteCount, 0, `${name}: 非有限（写されなかったフレーム）`);
              assert(report.pass, `${name}: maxAbs ${report.maxAbsError} が帯 ${BAND} の外`);
              return undefined;
            });
          },
        );
      }

      await t.step("故障注入のループは故障なしなら製品の経路と Uint32 で一致", async () => {
        const fixture = fixtures.accept;
        const product = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
        const looped = await decodeWithFault(gpu, sessions, liveCaches, fixture.latents, {
          kind: "none",
        });
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
          console.log(`[wan-vae] fault ${fault.kind}: maxAbs=${report.maxAbsError} ratio=${ratio}`);
          assert(!report.pass, `${label}: 帯の中に収まった（帯が広すぎる兆候）`);
          assert(report.maxAbsError > 1e3 * BAND, `${label}: maxAbs ${report.maxAbsError}`);
        });
      }

      await t.step(
        "故障注入: chunk 2 以降も first を使う → copyOutputs の大きさで enqueue が拒まれる",
        async () => {
          // first の出力フレームは [3,1,…] で、next 用のフレームの常駐（[3,4,…]）へは写せない —
          // runtime の `copyOutputs` の大きさの検査が dispatch を積む前に落とす（値を比べる前の赤）。
          await assertRejects(
            () =>
              decodeWithFault(gpu, sessions, liveCaches, fixtures.accept.latents, {
                kind: "first-for-all",
              }),
            ExecutionError,
            "copyOutputs",
          );
        },
      );
    } finally {
      // MUST: 常駐 → Session → device の順で畳む（常駐は Session の焼き込みから参照されうるので、
      // Session を先に畳んでから返す）。
      await first?.dispose();
      await next?.dispose();
      caches?.dispose();
      gpu.destroy();
    }
  },
});
