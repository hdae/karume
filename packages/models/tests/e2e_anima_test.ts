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
 * **turbo の門も 2 リポ揃って初めて走る** — turbo の共有コンポーネント（text_encoder /
 * text_conditioner / vae_decoder / tokenizer 2 本）は base リポへの越境参照（ADR 0038 §7）で
 * 焼かれていて、現物が turbo ミラー側に無いため。無い環境と GPU 無し環境は理由を出して
 * **明示 SKIP**する（テストを消して無音で緑にしない — ADR 0005）。
 *
 * NOTE: 実配布形の門は基本ぜんぶ**取得層経由**（ローカル HTTP + `fromPretrained`）で通す。
 * turbo は越境参照（別リポの (repo, commit SHA)）を含み、それを解けるのは取得層だけだから。
 * shard 分割そのものは全量面（`fromAssets`）でも読めるので、その 1 本だけ base を
 * `Deno.readFile` + `fromAssets` で通し、取得層経由と**同じ参照 sha256** を要求する
 * （下の「全量面」節 — X2-101）。ファイルを `Deno.open` / `Deno.readFile` で読むのはテスト側
 * だけで、パッケージ本体は Web 標準 API のみ（fs を持ち込まない — 横断不変条件）。
 */

import { assertEquals, assertFalse, assertRejects } from "@std/assert";
import { parseManifest, resolveFiles } from "@karume/hub";
import type { Manifest } from "@karume/hub";
import {
  type AnimaGenerateEvent,
  AnimaPipeline,
  type AnimaSamplerType,
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
/**
 * 素の base 配布形の置き場（untracked・turbo とは別リポ）。CFG の門（下の「CFG」節）に使う
 * ほか、**turbo の共有コンポーネントの現物もここにしか無い** — turbo の manifest は
 * text_encoder / text_conditioner / vae_decoder / tokenizer 2 本を base リポへの越境参照
 * （ADR 0038 §7）で焼くので、turbo ミラー側には transformer しか置かれない。
 */
const BASE_ASSETS_DIR = new URL("../../../models/karume-anima/", import.meta.url);
/** 実行日（モジュールロード時に 1 回だけ確定 — ダンプ先の日付ディレクトリに使う）。 */
const TODAY = new Date().toISOString().slice(0, 10);
/** ミスマッチ時の実物ダンプ先（`outputs/bench/` は消して安全な席 — docs/assets-layout.md）。 */
const OUTPUTS_DIR = new URL(
  `../../../outputs/bench/karume-anima/${TODAY}_e2e-mismatch/`,
  import.meta.url,
);

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
    quant: "f16+dit8-a8-attn8-s16",
    resolution: { width: 1024, height: 1024 },
    sha256: "aa013054d0ef6eefd6165462a089545574db227b0845057af52982d55753b608",
  },
  {
    quant: "f16+dit8-a8-attn8-s16",
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

const baseManifestText = await Deno.readTextFile(new URL("karume.json", BASE_ASSETS_DIR)).catch(
  () => undefined,
);
if (baseManifestText === undefined) {
  console.warn(
    `[karume] ${BASE_ASSETS_DIR.pathname} に karume.json が無いため CFG の e2e を SKIP する` +
      "（exporter の dist.py --pipeline anima で焼く）",
  );
}

/**
 * turbo の門も **base ミラーを要求する** — 共有コンポーネントは越境参照なので、turbo ミラー
 * だけでは text_encoder / vae_decoder の現物が 1 バイトも揃わない（{@link BASE_ASSETS_DIR}）。
 */
const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE && baseManifestText !== undefined;

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const readManifest = (): Manifest => parseManifest(manifestText as string);

// --- ローカル HTTP（HF 形の使い捨てサーバ）------------------------------------
//
// 実資産の門はほぼ全てこの土台に載る。turbo の共有コンポーネントは base リポへの**越境参照**
// （ADR 0038 §7）で turbo ミラーには現物が無く、宣言された (repo, commit SHA) から取れるのは
// 取得層だけだからで、shard 分割された base も本番と同じ経路（graph-first + 逐次流し）で
// 通したいため。全量面（`Deno.readFile` + `fromAssets`）は下の「全量面」節が 1 本だけ持つ。
// 喋るのは hub が実際に叩く 2 経路（revision 解決 API・resolve URL）だけ。

const REPO = "karume-test/anima";
const REVISION_SHA = "1234567890abcdef1234567890abcdef12345678";
const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/**
 * 越境参照の repo → ローカルミラー。turbo の共有コンポーネントは base リポの
 * (repo, commit SHA) を名乗るので、その 1 本だけ別の dir から配る必要がある。
 */
const CROSS_REPO_MIRRORS: ReadonlyMap<string, URL> = new Map([
  ["hdae/karume-anima", BASE_ASSETS_DIR],
]);

/** 取得元 1 つ（repo + revision）が配る内容。 */
type ServedOrigin = { readonly dir: URL; readonly paths: Set<string> };

/**
 * 取得元のキー。**repo だけでは足りない** — 同じ repo が別 revision で現れても取り違えない
 * （越境参照は宣言された commit SHA から取られ、セッションの SHA とは無関係）。
 */
const originKey = (repo: string, revision: string): string => `${repo}@${revision}`;

/** 取得元の表どおりに HF の URL 形で配る一時サーバ（ポート自動割当・127.0.0.1 束縛）。 */
const serveAssets = (
  origins: ReadonlyMap<string, ServedOrigin>,
): Deno.HttpServer<Deno.NetAddr> =>
  Deno.serve({ port: 0, hostname: "127.0.0.1", onListen: () => {} }, async (request) => {
    const { pathname } = new URL(request.url);
    // revision 解決を要求するのはセッション repo だけ（越境参照は commit SHA 固定なので、
    // 取得層が解決要求そのものを出さない）。
    if (REVISION_RE.test(pathname)) return Response.json({ sha: REVISION_SHA });
    const resolved = RESOLVE_RE.exec(pathname);
    if (resolved === null) return new Response("not found", { status: 404 });
    const [, repo, revision, rawPath] = resolved;
    const served = origins.get(originKey(repo, revision));
    const path = decodeURIComponent(rawPath);
    if (served === undefined || !served.paths.has(path)) {
      return new Response("not found", { status: 404 });
    }
    const file = await Deno.open(new URL(path, served.dir));
    const { size } = await file.stat();
    return new Response(file.readable, { headers: { "content-length": String(size) } });
  });

/**
 * `fromPretrained` がこの配布形 / quant で取りに来る path 全部（manifest 自身を含む）を、
 * **取りに行く取得元ごと**に束ねる。セッション repo は `dir` から、越境参照は宣言された
 * (repo, revision) のまま {@link CROSS_REPO_MIRRORS} のミラーから配る。
 */
const servedOrigins = (
  manifest: Manifest,
  quant: string,
  dir: URL,
): Map<string, ServedOrigin> => {
  const origins = new Map<string, ServedOrigin>();
  const add = (repo: string, revision: string, mirror: URL, path: string): void => {
    const key = originKey(repo, revision);
    const served = origins.get(key) ?? { dir: mirror, paths: new Set<string>() };
    served.paths.add(path);
    origins.set(key, served);
  };
  add(REPO, REVISION_SHA, dir, "karume.json");
  const files = resolveFiles(manifest, { quant });
  for (const key of Object.keys(files)) {
    const ref = files[key];
    if (ref.repo === undefined || ref.revision === undefined) {
      add(REPO, REVISION_SHA, dir, ref.path);
      continue;
    }
    const mirror = CROSS_REPO_MIRRORS.get(ref.repo);
    if (mirror === undefined) {
      throw new Error(
        `越境参照 '${ref.repo}' に対応するローカルミラーが無い` +
          `（既知: ${[...CROSS_REPO_MIRRORS.keys()].join(" / ")}）`,
      );
    }
    add(ref.repo, ref.revision, mirror, ref.path);
  }
  return origins;
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
 * いない）ので**先頭差分位置は原理的に出せない** — 代わりに実物を {@link OUTPUTS_DIR} へ
 * 落として、バイト長・画素統計と併せて人が突き合わせられる形にする。
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
 * いないことの直接証拠になる（下の onEvent 門）。`sampler` は request 側の上書き席
 * （省略時は manifest の `scheduler.type`）。
 */
const assertReferencePng = async (
  label: string,
  pipeline: AnimaPipeline,
  resolution: ImageSize,
  expected: string,
  options: {
    readonly sampler?: AnimaSamplerType;
    readonly onEvent?: (event: AnimaGenerateEvent) => void;
  } = {},
): Promise<void> => {
  const { sampler, onEvent } = options;
  const started = performance.now();
  const image = await pipeline.generate({
    prompt: PROMPT,
    resolution,
    steps: STEPS,
    seed: SEED,
    ...(sampler === undefined ? {} : { sampler }),
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

/**
 * turbo の配布形を**取得層経由**（ローカル HTTP + `fromPretrained`）で組み、`body` に渡す。
 * 戻り値は body を抜けた後の pipeline — **既に dispose 済み**で、解放後の面を検査する門
 * （下の onEvent 節）だけがこれを使う。
 *
 * MUST: `caches` は公開面の注入席から渡す（実 Cache Storage に数 GB を書かない）。呼び手が
 * 自分のインスタンスを渡した場合は、body を抜けた後にその中身を検査できる。
 */
const withTurbo = async (
  quant: string,
  options: {
    readonly caches?: CacheStorage;
    readonly onRunDiagnostics?: (component: string) => void;
  },
  body: (pipeline: AnimaPipeline) => Promise<void>,
): Promise<AnimaPipeline> => {
  const manifest = readManifest();
  const server = serveAssets(servedOrigins(manifest, quant, ASSETS_DIR));
  try {
    // `await using` は [Symbol.asyncDispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
    await using pipeline = await AnimaPipeline.fromPretrained(
      { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` },
      {
        quant,
        caches: options.caches ?? new MemoryCacheStorage(),
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
      },
    );
    await body(pipeline);
    return pipeline;
  } finally {
    await server.shutdown();
  }
};

for (const { quant, resolution, sha256 } of REFERENCE) {
  const label = `${quant}-${formatResolution(resolution)}`;
  // request 側 `sampler:"euler"` の明示が manifest 既定（= 公式配布の宣言）とビット同一である
  // ことは、**生成を 1 本も増やさず**最短の 512 ケースだけを明示に振り替えて見る。既定経路の
  // 門は残り 2 ケースと fromPretrained（同じ quant/512 を既定で回して同じ sha を要求する）が
  // 保つので、どちらの経路も裸のまま残らない。
  const sampler: AnimaSamplerType | undefined = resolution.width === 512 ? "euler" : undefined;
  Deno.test({
    name: `e2e(実GPU): quant ${quant} / ${formatResolution(resolution)} / ${STEPS}step / ` +
      `seed ${SEED} の PNG が参照 sha256 と一致する` +
      (sampler === undefined ? "" : `（sampler:"${sampler}" 明示）`),
    ignore: !RUNNABLE,
    fn: async () => {
      await withTurbo(
        quant,
        {},
        (pipeline) =>
          assertReferencePng(label, pipeline, resolution, sha256, {
            ...(sampler === undefined ? {} : { sampler }),
          }),
      );
    },
  });
}

// --- request 側 sampler の上書き（DPM++ 2M）-----------------------------------
//
// 公式配布の manifest は `scheduler.type: "euler"` を宣言している（再裁定 2026-08-25）。この
// 席が**数理まで届いている**ことは、request で `"dpmpp-2m"` を指定した出力が、同じ更新則を
// manifest 側で宣言して実測した golden（d2e0484 — daf86af で退役）と**ビット一致**することで
// 示す。上書き経路と宣言経路が同じバイトを産む ⇒ 実効サンプラーの決まり方だけが違い、数値の
// 綴りは 1 ビットも分岐していない。
//
// MUST: 1024² の上書きは足さない（GPU 時間の上限 — 更新則はホスト側の式で解像度に依らない）。

/** turbo 512²（`REFERENCE[1]` と同じ quant / 解像度 / steps / seed）を DPM++ 2M で回した実測。変更禁止。 */
const TURBO_DPMPP_SHA256 = "bb9b5c81fdf42d033035a91582c3f6e5200152cbbdc1cc2398096bd7309a3979";

Deno.test({
  name: `e2e(実GPU): turbo ${formatResolution(REFERENCE[1].resolution)} を request の ` +
    `sampler:"dpmpp-2m" で回すと DPM++ 2M の参照 sha256 と一致する`,
  ignore: !RUNNABLE,
  fn: async () => {
    const { quant, resolution } = REFERENCE[1];
    await withTurbo(quant, {}, (pipeline) =>
      assertReferencePng(
        `${quant}-${formatResolution(resolution)}-dpmpp`,
        pipeline,
        resolution,
        TURBO_DPMPP_SHA256,
        { sampler: "dpmpp-2m" },
      ));
  },
});

// --- CFG（素の base 配布形）--------------------------------------------------
//
// MUST: **CFG≠1 の経路にも実 GPU の門を置く**。`guidanceScale === 1` は uncond 側を 1 度も
// 計算しない（`needsUncond`）ので、上の 3 ケースは cond 1 本しか通っていない —
// CFG の分岐・2 本目の text 経路・`cfgEulerStep` の合成は**どれも実 GPU で 1 度も走って
// いなかった**（波 L で気付いた穴）。素の base 配布形は既定が CFG なので、そこを 1 ケース
// だけ固定する（解像度 512 は所要時間の都合 — CFG は 1 step が forward 2 本）。

/** 参照値（2026-08-22 実測 — 変更禁止）。 */
const BASE_REFERENCE = {
  quant: "f16+dit8-a8-attn8-s16",
  resolution: { width: 512, height: 512 },
  steps: 20,
  guidanceScale: 4,
  negativePrompt: "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
  sha256: "071929c40e90628006eab593842080246140e771a82f8a35762507f4a12e9560",
} as const;

/** 組み上がった base の pipeline を {@link BASE_REFERENCE} のノブで 1 枚焼き、sha を突き合わせる。 */
const assertBasePng = async (
  label: string,
  pipeline: AnimaPipeline,
  expected: string,
  options: { readonly sampler?: AnimaSamplerType } = {},
): Promise<void> => {
  const { resolution, steps, guidanceScale, negativePrompt } = BASE_REFERENCE;
  const started = performance.now();
  const image = await pipeline.generate({
    prompt: PROMPT,
    resolution,
    steps,
    guidanceScale,
    negativePrompt,
    seed: SEED,
    ...(options.sampler === undefined ? {} : { sampler: options.sampler }),
  });
  const png = await encodePng(image.data, image.width, image.height);
  const actual = await sha256Hex(png);
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);
  console.log(`[e2e] ${label}: ${elapsed}s / PNG ${png.length}B / sha256 ${actual}`);
  if (actual !== expected) {
    throw new Error(await mismatchReport(label, image, png, expected, actual));
  }
};

/**
 * 素の base を {@link BASE_REFERENCE} のノブで 1 枚焼き、参照 sha256 と突き合わせる。
 *
 * 経路は**取得層経由**（ローカル HTTP + `fromPretrained`）— この配布形の text_encoder と
 * transformer は 1GiB 超で shard 分割されていて、越境参照と併せて解けるのは取得層だけ
 * （デモの `--source` と同型 — `examples/shared/local-dist-server.ts`）。分割形は全量面
 * （`fromAssets`）でも読めるが、そちらは全 shard がホスト RAM に同時に載る面で、門としては
 * 下の 1 本（同じ参照 sha を要求する）で別に閉じる。
 */
const assertBaseReferencePng = async (
  label: string,
  expected: string,
  options: { readonly sampler?: AnimaSamplerType } = {},
): Promise<void> => {
  const manifest = parseManifest(baseManifestText as string);
  const { quant } = BASE_REFERENCE;
  const server = serveAssets(servedOrigins(manifest, quant, BASE_ASSETS_DIR));
  try {
    // MUST: `caches` は公開面の注入席から渡す（実 Cache Storage に数 GB を書かない）。
    await using pipeline = await AnimaPipeline.fromPretrained(
      { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` },
      { quant, caches: new MemoryCacheStorage() },
    );
    await assertBasePng(label, pipeline, expected, options);
  } finally {
    await server.shutdown();
  }
};

Deno.test({
  name: `e2e(実GPU): 素の base / CFG ${BASE_REFERENCE.guidanceScale} / ` +
    `${BASE_REFERENCE.steps}step の PNG が参照 sha256 と一致する`,
  ignore: !GPU_AVAILABLE || baseManifestText === undefined,
  fn: () => assertBaseReferencePng("base-cfg", BASE_REFERENCE.sha256),
});

// --- 全量面（fromAssets）で分割配布形を読む -----------------------------------
//
// 「`fromPretrained` で読める配布形は `fromAssets` でも読める」（X2-101）の門。実配布形の
// transformer は quant `i8` でも 2 shard に割れていて、`resolveFiles` は `transformer[0]` /
// `transformer[1]` を返す — その Record をそのまま全量面へ渡し、**上の CFG の門と同じ参照
// sha256** を要求する。同じバイトを別の面から流し込んでいるだけなので、1 ビットも動かないのが
// 正しい（動いたら shard 列の順序かグラフ shard の扱いが壊れている）。

/** ローカル配布形を全量読みする（`examples/shared/local-assets.ts` と同型の面）。 */
const readLocalAssets = async (
  dir: URL,
  manifest: Manifest,
  quant: string,
): Promise<{ manifest: Manifest; assets: Record<string, Uint8Array<ArrayBuffer>> }> => {
  const files = resolveFiles(manifest, { quant });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    const ref = files[key];
    // 越境参照は (repo, commit SHA) からしか取れない — 全量面では解けないので fail loudly。
    if (ref.repo !== undefined) {
      throw new Error(`全量面では越境参照 '${ref.repo}' を解けない（取得キー ${key}）`);
    }
    const bytes = byPath.get(ref.path) ?? await Deno.readFile(new URL(ref.path, dir));
    byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  }
  return { manifest, assets };
};

Deno.test({
  name:
    "e2e(実GPU): 分割配布形を fromAssets（全量面）で組んでも 素の base の参照 sha256 と一致する",
  ignore: !GPU_AVAILABLE || baseManifestText === undefined,
  fn: async () => {
    const { quant } = BASE_REFERENCE;
    const input = await readLocalAssets(
      BASE_ASSETS_DIR,
      parseManifest(baseManifestText as string),
      quant,
    );
    // 分割形であること自体をこの門で確かめる — 1 shard の配布形へ戻った日には、黙って
    // 「全量面の門」に化けるのではなく理由を出して落とす（門の意味が消える方が危ない）。
    const shardKeys = Object.keys(input.assets).filter((key) => key.endsWith("]"));
    if (shardKeys.length < 2) {
      throw new Error(
        `この配布形は shard 分割されていない（取得キー: ${Object.keys(input.assets).join(" / ")}）`,
      );
    }
    await using pipeline = await AnimaPipeline.fromAssets(input, { quant });
    await assertBasePng("base-cfg-fromAssets-shards", pipeline, BASE_REFERENCE.sha256);
  },
});

/**
 * 素の base を {@link BASE_REFERENCE} と同じノブ（CFG 4 / 20step / seed 42 / 同じネガティブ）で
 * DPM++ 2M へ振ったときの実測（d2e0484 の golden — daf86af で退役）。変更禁止。
 *
 * CFG≠1 の経路にも上書き席の門を置く — DPM++ 2M の 2 次項は cond/uncond の**合成後**の値を
 * 履歴に持つので、turbo（CFG 無し）の一致だけでは合成と履歴の噛み合わせが裸のまま残る。
 */
const BASE_DPMPP_SHA256 = "97dad23f6d3bede37b259cbf323b28de6bea60433a431ff859ab3815a08d571c";

Deno.test({
  name: `e2e(実GPU): 素の base を request の sampler:"dpmpp-2m" で回すと ` +
    `DPM++ 2M の参照 sha256 と一致する`,
  ignore: !GPU_AVAILABLE || baseManifestText === undefined,
  fn: () => assertBaseReferencePng("base-cfg-dpmpp", BASE_DPMPP_SHA256, { sampler: "dpmpp-2m" }),
});

// --- fromPretrained（取得層込み）--------------------------------------------
//
// `loadManifest` → `resolveFiles` → 取得 → 構築の**実経路**は turbo の門が全て通るように
// なったが、そこに**注入席とキャッシュの実体**を突き合わせるのはこの 1 本だけ。manifest の実
// sha256 による integrity 検証が走る経路でもあり、資産の 3 点セット（path/size/sha256）が
// 現物と合っていることの門を兼ねる。
//
// 解像度が 512 なのは RAM の都合（取得層のメモリキャッシュが資産の写しを持つ）。ビット一致の
// 門としては 1024 と同格 — 参照 sha は上の 512 ケースと同じ値を使う。

Deno.test({
  name: "e2e(実GPU): fromPretrained（取得層 + integrity 検証）の PNG が参照 sha256 と一致する",
  ignore: !RUNNABLE,
  fn: async () => {
    const quant = "f16+dit8-a8-attn8-s16";
    const resolution: ImageSize = { width: 512, height: 512 };
    const files = resolveFiles(readManifest(), { quant });
    const caches = new MemoryCacheStorage();
    // 観測席（onRunDiagnostics）の run 回数もこの実生成で併せて固定する（DiT は 1 step =
    // 1 run・text 系は各 1 run）。門の sha256 判定には一切影響しない追加観測。
    const observed: string[] = [];
    await withTurbo(
      quant,
      { caches, onRunDiagnostics: (component) => observed.push(component) },
      (pipeline) =>
        assertReferencePng("fromPretrained-512", pipeline, resolution, REFERENCE[1].sha256),
    );
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
    // 注入した側に現物があることで、注入席が末端の取得層まで届いていること（= 実キャッシュを
    // 汚していないこと）を示す。名前空間の名前は取得層が所有するので数えるのは全名前空間ぶん。
    // 期待は **manifest 1 本 + 一意 sha256 の本数** — 資産のキーは内容キーなので、同一バイトの
    // 複数 path は 1 エントリに畳まれる（越境参照ぶんも同じキーの作り方で、リポが違うだけ）。
    let entries = 0;
    for (const namespace of caches.namespaces.values()) entries += namespace.entries.size;
    const expected = new Set(Object.keys(files).map((key) => files[key].sha256)).size + 1;
    if (entries !== expected) {
      throw new Error(`注入したキャッシュのエントリ数が ${entries}（期待 ${expected}）`);
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
    const quant = "f16+dit8-a8-attn8-s16";
    const resolution: ImageSize = { width: 1024, height: 1024 };
    const manifest = readManifest();
    const { scheduler } = parseAnimaPipelineConfig(
      manifest.models[manifest.defaultModel].pipelineConfig,
    );
    const runs: string[] = [];
    // 戻りは**解放済み**の本体（`withTurbo` を抜けた時点で dispose が走っている）。中断した
    // 生成の後でも解放が正常に効くこと（直列化鎖は失敗を次へ持ち越さない）はここで通る。
    const disposed = await withTurbo(quant, {
      onRunDiagnostics: (component) => runs.push(component),
    }, async (pipeline) => {
      await t.step("イベント列と途中 latent を固定し、PNG は参照 sha256 のまま", async () => {
        const log: string[] = [];
        const shapes: string[] = [];
        const lengths: number[] = [];
        let firstStep: (() => { data: Float32Array<ArrayBuffer> }) | undefined;
        let firstStepData: Float32Array | undefined;
        await assertReferencePng("onEvent-1024", pipeline, resolution, REFERENCE[0].sha256, {
          onEvent: (event) => {
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
          },
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
    });
    await assertRejects(
      () => disposed.generate({ prompt: PROMPT, steps: 2, seed: SEED }),
      Error,
      "dispose 済み",
    );
  },
});
