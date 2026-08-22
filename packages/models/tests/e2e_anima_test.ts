/**
 * Anima の**移植の門**（実 GPU）。manifest → 資産 → `AnimaPipeline` → `generate` →
 * `encodePng` まで通し、出力 PNG の sha256 が参照値と**ビット一致**するかだけを見る。
 *
 * 参照値は移行元デモ（S 形 DiT + 常時タイル VAE・turbo 8 step・seed 42）の実測。数値が 1 bit
 * でも動いたら移植のどこかが変わっている — **tolerance 化も参照値の差し替えも禁止**で、
 * 赤のまま止めて差分の内容（PNG バイト長 / 画素の統計 / 実物の PNG）を出す。ここを緩めると
 * 「移植できた」の意味が消える。
 *
 * MUST: 資産は `models/karume-anima-turbo/` と `models/karume-anima/`（untracked・実 GPU 機の
 * ローカル資産）。前者が turbo（8 step / CFG 無し）の門で、後者が **CFG≠1 の門**（素の base
 * 配布形 — 既定が CFG なので、そこだけが 2 本目の text 経路と `cfgEulerStep` を通る）。
 * 無い環境と GPU 無し環境は理由を出して**明示 SKIP**する（テストを消して無音で緑にしない —
 * ADR 0005）。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（fs を持ち込まない — 横断不変条件）で、`fromAssets` はバイト列を受け取るだけの面。
 */

import { assertEquals, assertFalse, assertRejects } from "@std/assert";
import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest } from "@karume/hub";
import {
  type AnimaGenerateEvent,
  AnimaPipeline,
  encodePng,
  type GeneratedImage,
  type ImageSize,
} from "../mod.ts";
import { formatResolution } from "../anima.ts";
import { parseAnimaPipelineConfig } from "../src/anima/config.ts";
import { sigmaSchedule } from "../src/anima/sampler.ts";
import { ANIMA_SPATIAL_COMPRESSION } from "../src/anima/dit-tokens.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** 資産の置き場（リポ直下 `models/karume-anima-turbo/`）。 */
const ASSETS_DIR = new URL("../../../models/karume-anima-turbo/", import.meta.url);
const OUTPUTS_DIR = new URL("../../../outputs/demo/", import.meta.url);

/**
 * 移行元デモの既定プロンプト（参照 PNG を焼いた時の文字列そのもの）。
 * ネガティブは `guidanceScale 1` で 1 文字も使われないので渡さない（渡すと fail loudly）。
 */
const PROMPT = "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, " +
  "smile, upper body, masterpiece, best quality";
const STEPS = 8;
const SEED = 42;

/** 参照値（移行元デモの実測 — 変更禁止）。 */
const REFERENCE = [
  {
    quant: "w8a8-s16",
    resolution: { width: 1024, height: 1024 },
    sha256: "aa013054d0ef6eefd6165462a089545574db227b0845057af52982d55753b608",
  },
  {
    quant: "w8a8-s16",
    resolution: { width: 512, height: 512 },
    sha256: "dd4506de50f346676a35919d471ff7030514992cd337077c04c0dd2ffa332756",
  },
  {
    quant: "f16",
    resolution: { width: 1024, height: 1024 },
    sha256: "6943b541a21e3e22c40661d007bbc638f23365c17a95dd3e8363460abfc610db",
  },
] as const satisfies readonly { quant: string; resolution: ImageSize; sha256: string }[];

const manifestText = await Deno.readTextFile(new URL("karume.json", ASSETS_DIR)).catch(
  () => undefined,
);
const ASSETS_AVAILABLE = manifestText !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} に karume.json が無いため Anima の e2e を SKIP する` +
      "（移植の門は実資産でしか閉じない — exporter の dist.py で焼く）",
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const readManifest = (): Manifest => parseManifest(manifestText as string);

/**
 * quant が要求する資産をローカルから読む（`fetchAssets` のローカル版 — 取得層を通さない
 * 経路で、パイプライン単体を門に掛ける）。同じ path を指すキーは 1 度だけ読む。
 */
const loadLocalAssets = async (
  manifest: Manifest,
  quant: string,
  dir: URL = ASSETS_DIR,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const files = resolveFiles(manifest, { quant });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    const { path } = files[key];
    const cached = byPath.get(path);
    const bytes = cached ?? await Deno.readFile(new URL(path, dir));
    if (cached === undefined) byPath.set(path, bytes);
    assets = { ...assets, [key]: bytes };
  }
  return assets;
};

/** 画素の要約（参照 sha と食い違ったときに「どれくらい違う絵なのか」を言うため）。 */
const describePixels = (image: GeneratedImage): string => {
  const channels = ["R", "G", "B"];
  return channels.map((name, offset) => {
    let sum = 0;
    let min = 255;
    let max = 0;
    for (let at = offset; at < image.data.length; at += 4) {
      const value = image.data[at];
      sum += value;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    return `${name} 平均 ${(sum / (image.width * image.height)).toFixed(2)} / 範囲 ${min}〜${max}`;
  }).join(" / ");
};

/**
 * 参照 sha と食い違ったときの報告。
 *
 * MUST: ここで tolerance に逃げない。参照は sha256 しか無い（参照 PNG のバイト列は持って
 * いない）ので**先頭差分位置は原理的に出せない** — 代わりに実物を `outputs/demo/` へ落として、
 * バイト長・画素統計と併せて人が突き合わせられる形にする。
 */
const mismatchReport = async (
  label: string,
  image: GeneratedImage,
  png: Uint8Array<ArrayBuffer>,
  expected: string,
  actual: string,
): Promise<string> => {
  await Deno.mkdir(OUTPUTS_DIR, { recursive: true });
  const dumped = new URL(`e2e-mismatch-${label}.png`, OUTPUTS_DIR);
  await Deno.writeFile(dumped, png);
  return `${label}: 出力 PNG の sha256 が参照と一致しない\n` +
    `  期待 ${expected}\n  実際 ${actual}\n` +
    `  PNG ${png.length} バイト / 画像 ${image.width}×${image.height}\n` +
    `  画素 ${describePixels(image)}\n` +
    `  実物 ${dumped.pathname}（参照はバイト列ではなく sha256 のみなので先頭差分位置は出せない）`;
};

/**
 * 生成 → PNG → sha 突合。一致しない場合は緩めず、診断を付けて落とす。
 *
 * `onEvent` を渡した呼びも**同じ参照値**で突き合わせる — 観測席が数値に 1 ビットも触って
 * いないことの直接証拠になる（下の onEvent 門）。
 */
const assertReferencePng = async (
  label: string,
  pipeline: AnimaPipeline,
  resolution: ImageSize,
  expected: string,
  onEvent?: (event: AnimaGenerateEvent) => void,
): Promise<void> => {
  const started = performance.now();
  const image = await pipeline.generate({
    prompt: PROMPT,
    resolution,
    steps: STEPS,
    seed: SEED,
    ...(onEvent === undefined ? {} : { onEvent }),
  });
  const png = await encodePng(image.data, image.width, image.height);
  const actual = await sha256Hex(png);
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);
  console.log(`[e2e] ${label}: ${elapsed}s / PNG ${png.length}B / sha256 ${actual}`);
  if (actual !== expected) {
    throw new Error(await mismatchReport(label, image, png, expected, actual));
  }
};

for (const { quant, resolution, sha256 } of REFERENCE) {
  const label = `${quant}-${formatResolution(resolution)}`;
  Deno.test({
    name: `e2e(実GPU): quant ${quant} / ${formatResolution(resolution)} / ${STEPS}step / ` +
      `seed ${SEED} の PNG が参照 sha256 と一致する`,
    ignore: !RUNNABLE,
    fn: async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, quant);
      // `await using` は [Symbol.asyncDispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
      await using pipeline = await AnimaPipeline.fromAssets({ manifest, assets }, { quant });
      await assertReferencePng(label, pipeline, resolution, sha256);
    },
  });
}

// --- CFG（素の base 配布形）--------------------------------------------------
//
// MUST: **CFG≠1 の経路にも実 GPU の門を置く**。`guidanceScale === 1` は uncond 側を 1 度も
// 計算しない（`needsUncond`）ので、上の 3 ケースは cond 1 本しか通っていない —
// CFG の分岐・2 本目の text 経路・`cfgEulerStep` の合成は**どれも実 GPU で 1 度も走って
// いなかった**（波 L で気付いた穴）。素の base 配布形は既定が CFG なので、そこを 1 ケース
// だけ固定する（解像度 512 は所要時間の都合 — CFG は 1 step が forward 2 本）。

/** 素の base 配布形の置き場（untracked・turbo とは別リポ）。 */
const BASE_ASSETS_DIR = new URL("../../../models/karume-anima/", import.meta.url);

/** 参照値（2026-08-22 実測 — 変更禁止）。 */
const BASE_REFERENCE = {
  quant: "w8a8-s16",
  resolution: { width: 512, height: 512 },
  steps: 20,
  guidanceScale: 4,
  negativePrompt: "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
  sha256: "071929c40e90628006eab593842080246140e771a82f8a35762507f4a12e9560",
} as const;

const baseManifestText = await Deno.readTextFile(new URL("karume.json", BASE_ASSETS_DIR)).catch(
  () => undefined,
);
if (baseManifestText === undefined) {
  console.warn(
    `[karume] ${BASE_ASSETS_DIR.pathname} に karume.json が無いため CFG の e2e を SKIP する` +
      "（exporter の dist.py --pipeline anima で焼く）",
  );
}

Deno.test({
  name: `e2e(実GPU): 素の base / CFG ${BASE_REFERENCE.guidanceScale} / ` +
    `${BASE_REFERENCE.steps}step の PNG が参照 sha256 と一致する`,
  ignore: !GPU_AVAILABLE || baseManifestText === undefined,
  fn: async () => {
    const manifest = parseManifest(baseManifestText as string);
    const { quant, resolution, steps, guidanceScale, negativePrompt, sha256 } = BASE_REFERENCE;
    const assets = await loadLocalAssets(manifest, quant, BASE_ASSETS_DIR);
    await using pipeline = await AnimaPipeline.fromAssets({ manifest, assets }, { quant });
    const started = performance.now();
    const image = await pipeline.generate({
      prompt: PROMPT,
      resolution,
      steps,
      guidanceScale,
      negativePrompt,
      seed: SEED,
    });
    const png = await encodePng(image.data, image.width, image.height);
    const actual = await sha256Hex(png);
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    console.log(`[e2e] base-cfg: ${elapsed}s / PNG ${png.length}B / sha256 ${actual}`);
    if (actual !== sha256) {
      throw new Error(await mismatchReport("base-cfg", image, png, sha256, actual));
    }
  },
});

// --- fromPretrained（取得層込み）--------------------------------------------
//
// ローカル HTTP を HF の URL 形で立て、`loadManifest` → `resolveFiles` → `fetchAssets` →
// `fromAssets` の**実経路**を通す。ここだけが manifest の実 sha256 による integrity 検証を
// 実際に走らせる経路で、資産の 3 点セット（path/size/sha256）が現物と合っていることの門でもある。
//
// 解像度が 512 なのは RAM の都合（取得層のメモリキャッシュが資産の写しを持つ）。ビット一致の
// 門としては 1024 と同格 — 参照 sha は上の 512 ケースと同じ値を使う。

const REPO = "karume-test/anima";
const REVISION_SHA = "1234567890abcdef1234567890abcdef12345678";
const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/** HF の 2 経路（revision 解決 API・resolve URL）だけを喋る一時サーバ。 */
const serveAssets = (paths: ReadonlySet<string>): Deno.HttpServer<Deno.NetAddr> =>
  Deno.serve({ port: 0, hostname: "127.0.0.1", onListen: () => {} }, async (request) => {
    const { pathname } = new URL(request.url);
    if (REVISION_RE.test(pathname)) return Response.json({ sha: REVISION_SHA });
    const resolved = RESOLVE_RE.exec(pathname);
    if (resolved === null) return new Response("not found", { status: 404 });
    const [, repo, revision, rawPath] = resolved;
    const path = decodeURIComponent(rawPath);
    if (repo !== REPO || revision !== REVISION_SHA || !paths.has(path)) {
      return new Response("not found", { status: 404 });
    }
    const file = await Deno.open(new URL(path, ASSETS_DIR));
    const { size } = await file.stat();
    return new Response(file.readable, { headers: { "content-length": String(size) } });
  });

Deno.test({
  name: "e2e(実GPU): fromPretrained（取得層 + integrity 検証）の PNG が参照 sha256 と一致する",
  ignore: !RUNNABLE,
  fn: async () => {
    const quant = "w8a8-s16";
    const resolution: ImageSize = { width: 512, height: 512 };
    const files = resolveFiles(readManifest(), { quant });
    const paths = new Set(["karume.json", ...Object.keys(files).map((key) => files[key].path)]);
    const server = serveAssets(paths);
    try {
      const hubUrl = `http://127.0.0.1:${server.addr.port}`;
      // MUST: `caches` は公開面の注入席から渡す（実 Cache Storage に数 GB を書かない）。
      const caches = new MemoryCacheStorage();
      // 観測席（onRunDiagnostics）の run 回数もこの実生成で併せて固定する（DiT は 1 step =
      // 1 run・text 系は各 1 run）。門の sha256 判定には一切影響しない追加観測。
      const observed: string[] = [];
      const pipeline = await AnimaPipeline.fromPretrained({ repo: REPO, hubUrl }, {
        quant,
        caches,
        onRunDiagnostics: (component) => observed.push(component),
      });
      try {
        await assertReferencePng("fromPretrained-512", pipeline, resolution, REFERENCE[1].sha256);
      } finally {
        await pipeline.dispose();
      }
      const counts = new Map<string, number>();
      for (const component of observed) counts.set(component, (counts.get(component) ?? 0) + 1);
      const runsOf = (component: string): number => counts.get(component) ?? 0;
      if (
        runsOf("text_encoder") !== 1 || runsOf("text_conditioner") !== 1 ||
        runsOf("transformer") !== STEPS || runsOf("vae_decoder") < 1
      ) {
        throw new Error(
          "観測席の run 数が実行構造と合わない: " +
            `text_encoder ${runsOf("text_encoder")}（期待 1）/ ` +
            `text_conditioner ${runsOf("text_conditioner")}（期待 1）/ ` +
            `transformer ${runsOf("transformer")}（期待 ${STEPS}）/ ` +
            `vae_decoder ${runsOf("vae_decoder")}（期待 1 以上）`,
        );
      }
      // 取得したものは全て `karume/1` 名前空間へ入る（無認証経路）。注入した側に現物があることで、
      // 注入席が末端の取得層まで届いていること（= 実キャッシュを汚していないこと）を示す。
      const cached = await caches.open("karume/1");
      const entries = (await cached.keys()).length;
      if (entries !== paths.size) {
        throw new Error(
          `キャッシュ名前空間 karume/1 のエントリ数が ${entries}（期待 ${paths.size}）`,
        );
      }
    } finally {
      await server.shutdown();
    }
  },
});

// --- onEvent（生成イベントの観測席）------------------------------------------
//
// 観測席そのものの門。押さえるのは 4 点:
//  ① イベント列が実行構造と**完全一致**する（段 4 本 × start/end / denoise が steps 回 /
//     VAE タイルが枚数ぶん）。枚数は `onRunDiagnostics` の `vae_decoder` run 数から採る
//     （タイル幾何を再実装せず、独立な 2 つの観測を突き合わせる）。
//  ② `copyLatents()` の形が latent の形（解像度 ÷ 空間圧縮率）と一致し、要素数も合う。
//  ③ **PNG の sha256 が観測なしの参照値と同一**。購読側は毎 step 写しを受け取り、その写しを
//     NaN で壊してすらいる — それでも絵が 1 ビットも動かないことが「観測に副作用が無い」の
//     直接証拠になる（写しでなく内部配列を渡していたら、ここが必ず割れる）。
//  ④ コールバックの throw が生成を落とし、その後の `dispose` が正常に効く（例外を握らない
//     流儀の副産物 = step 粒度の中断手段）。

Deno.test({
  name: "e2e(実GPU): onEvent の観測は数値に触らない（イベント列 / 途中 latent / 中断）",
  ignore: !RUNNABLE,
  fn: async (t) => {
    const quant = "w8a8-s16";
    const resolution: ImageSize = { width: 1024, height: 1024 };
    const manifest = readManifest();
    const { scheduler } = parseAnimaPipelineConfig(
      manifest.models[manifest.defaultModel].pipelineConfig,
    );
    const assets = await loadLocalAssets(manifest, quant);
    const runs: string[] = [];
    const pipeline = await AnimaPipeline.fromAssets({ manifest, assets }, {
      quant,
      onRunDiagnostics: (component) => runs.push(component),
    });
    try {
      await t.step("イベント列と途中 latent を固定し、PNG は参照 sha256 のまま", async () => {
        const log: string[] = [];
        const shapes: string[] = [];
        const lengths: number[] = [];
        let firstStep: (() => { data: Float32Array<ArrayBuffer> }) | undefined;
        let firstStepData: Float32Array | undefined;
        await assertReferencePng("onEvent-1024", pipeline, resolution, REFERENCE[0].sha256, (
          event,
        ) => {
          if (event.kind === "stage") {
            log.push(`stage:${event.component}:${event.at}`);
            return;
          }
          if (event.kind === "vae-tile") {
            log.push(`tile:${event.tile}/${event.tiles}`);
            return;
          }
          log.push(`step:${event.step}/${event.steps}@${event.sigma}`);
          const snapshot = event.copyLatents();
          shapes.push(snapshot.shape.join("x"));
          lengths.push(snapshot.data.length);
          // 写しを壊す。内部配列を渡していたら以後の step が NaN 汚染され、PNG の sha が割れる。
          snapshot.data.fill(Number.NaN);
          if (event.step === 1) {
            firstStep = event.copyLatents;
            firstStepData = event.copyLatents().data;
          }
        });

        // ① 実行構造との完全一致。VAE タイル数は診断側の run 数から採る（独立な観測）。
        const tiles = runs.filter((component) => component === "vae_decoder").length;
        const sigmas = sigmaSchedule(STEPS, scheduler.shift);
        assertEquals(
          log,
          [
            "stage:text_encoder:start",
            "stage:text_encoder:end",
            "stage:text_conditioner:start",
            "stage:text_conditioner:end",
            "stage:transformer:start",
            ...Array.from({ length: STEPS }, (_, at) => `step:${at + 1}/${STEPS}@${sigmas[at]}`),
            "stage:transformer:end",
            "stage:vae_decoder:start",
            ...Array.from({ length: tiles }, (_, at) => `tile:${at + 1}/${tiles}`),
            "stage:vae_decoder:end",
          ],
          "イベント列が実行構造と合わない（段の前後 / step 数 / タイル数）",
        );

        // ② 途中 latent の形。解像度から導く（テストに 128 を書かない）。
        const latentHeight = resolution.height / ANIMA_SPATIAL_COMPRESSION;
        const latentWidth = resolution.width / ANIMA_SPATIAL_COMPRESSION;
        const channels = lengths[0] / (latentHeight * latentWidth);
        assertEquals(
          new Set(shapes),
          new Set([`1x${channels}x${latentHeight}x${latentWidth}`]),
          `途中 latent の形が [1,C,${latentHeight},${latentWidth}] でない`,
        );
        assertEquals(
          new Set(lengths),
          new Set([channels * latentHeight * latentWidth]),
          "途中 latent の要素数が形と合わない",
        );

        // ③ 生成が終わった後に呼んでも step 1 の値が返る（lazy だが束縛は作った時点）。
        assertEquals(
          Array.from((firstStep as () => { data: Float32Array<ArrayBuffer> })().data.slice(0, 8)),
          Array.from((firstStepData as Float32Array).slice(0, 8)),
          "生成後に呼んだ copyLatents が step 1 の値を返さない",
        );
      });

      await t.step("コールバックの throw は生成ごと落とす（step 粒度の中断）", async () => {
        const seen: string[] = [];
        const abortAt = 2;
        await assertRejects(
          () =>
            pipeline.generate({
              prompt: PROMPT,
              resolution: { width: 512, height: 512 },
              steps: abortAt,
              seed: SEED,
              onEvent: (event) => {
                seen.push(
                  event.kind === "stage"
                    ? `stage:${event.component}:${event.at}`
                    : event.kind === "denoise-step"
                    ? `step:${event.step}`
                    : `tile:${event.tile}`,
                );
                if (event.kind === "denoise-step" && event.step === abortAt) {
                  throw new Error("購読側で中断");
                }
              },
            }),
          Error,
          "購読側で中断",
        );
        // 例外を握らないので DiT の段で止まる — VAE 段は 1 度も開かない。
        assertEquals(seen.at(-1), `step:${abortAt}`, "中断した step が最後のイベントでない");
        assertFalse(seen.includes("stage:transformer:end"), "中断したのに段が正常終了している");
        assertFalse(
          seen.some((entry) => entry.startsWith("stage:vae_decoder")),
          "中断したのに VAE 段へ入っている",
        );
      });
    } finally {
      // 中断した生成の後でも解放は正常に効く（直列化鎖は失敗を次へ持ち越さない）。
      await pipeline.dispose();
    }
    await assertRejects(
      () => pipeline.generate({ prompt: PROMPT, steps: 2, seed: SEED }),
      Error,
      "dispose 済み",
    );
  },
});
