/**
 * Anima の**移植の門**（実 GPU）。manifest → 資産 → `AnimaPipeline` → `generate` →
 * `encodePng` まで通し、出力 PNG の sha256 が参照値と**ビット一致**するかだけを見る。
 *
 * 参照値は移行元デモ（S 形 DiT + 常時タイル VAE・turbo 8 step・seed 42）の実測。数値が 1 bit
 * でも動いたら移植のどこかが変わっている — **tolerance 化も参照値の差し替えも禁止**で、
 * 赤のまま止めて差分の内容（PNG バイト長 / 画素の統計 / 実物の PNG）を出す。ここを緩めると
 * 「移植できた」の意味が消える。
 *
 * MUST: 資産は `models/anima-turbo/`（untracked・実 GPU 機のローカル資産）。無い環境と GPU 無し
 * 環境は理由を出して**明示 SKIP**する（テストを消して無音で緑にしない — ADR 0005）。
 *
 * NOTE: ローカル読みに `Deno.readFile` を使うのはテストだけ。パッケージ本体は Web 標準 API
 * のみ（fs を持ち込まない — 横断不変条件）で、`fromAssets` はバイト列を受け取るだけの面。
 */

import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest } from "@karume/hub";
import { AnimaPipeline, encodePng, type GeneratedImage, type ImageSize } from "../mod.ts";
import { formatResolution } from "../anima.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** 資産の置き場（リポ直下 `models/anima-turbo/`）。 */
const ASSETS_DIR = new URL("../../../models/anima-turbo/", import.meta.url);
const OUTPUTS_DIR = new URL("../../../outputs/", import.meta.url);

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
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const files = resolveFiles(manifest, { quant });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    const { path } = files[key];
    const cached = byPath.get(path);
    const bytes = cached ?? await Deno.readFile(new URL(path, ASSETS_DIR));
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
 * いない）ので**先頭差分位置は原理的に出せない** — 代わりに実物を `outputs/` へ落として、
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

/** 生成 → PNG → sha 突合。一致しない場合は緩めず、診断を付けて落とす。 */
const assertReferencePng = async (
  label: string,
  pipeline: AnimaPipeline,
  resolution: ImageSize,
  expected: string,
): Promise<void> => {
  const started = performance.now();
  const image = await pipeline.generate({ prompt: PROMPT, resolution, steps: STEPS, seed: SEED });
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
      // `using` は [Symbol.dispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
      using pipeline = await AnimaPipeline.fromAssets({ manifest, assets }, { quant });
      await assertReferencePng(label, pipeline, resolution, sha256);
    },
  });
}

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
      const pipeline = await AnimaPipeline.fromPretrained({ repo: REPO, hubUrl }, {
        quant,
        caches,
      });
      try {
        await assertReferencePng("fromPretrained-512", pipeline, resolution, REFERENCE[1].sha256);
      } finally {
        pipeline.dispose();
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
