/**
 * Anima の**移植の門**（実 GPU）。manifest → 資産 → `Pipeline` → `generate` →
 * `encodePng` まで通し、出力画素（`encodePng` が IDAT に書く RGB8 の画素列 — {@link rgbPixels}）の sha256 が
 * 参照値と**ビット一致**するかだけを見る。PNG は実物として残すが、sha は圧縮前の画素列に対して採る —
 * IDAT の deflate はランタイムの zlib 実装で変わり（Deno と Node で画素が同一でもバイトが違う）、
 * 行が数値ではなく圧縮器を掴むため（ADR 0106 追記決定 9）。
 *
 * 参照値は `fixtures/references/anima.json` に**環境キーごとの行**で入っている
 * （`helpers/reference.ts`）。この門が主張できるのは「その機で数値が動いていない」ことだけで、
 * クロスデバイスのビット同一は仕様として保証しない（docs/limitations.md）。RTX 3080 Ti の行は
 * turbo が **anima-turbo-v1.1（公式 checkpoint — ADR 0087）の初回実測で凍結**したもの
 * （2026-09-01 の再構造で旧 fused turbo の参照値は退役 — モデル自体が置き換わったため。素の
 * base〈anima-v1.0〉の行は 2026-08-22 実測のまま = 再構造で v1.0 のバイトが 1 bit も動いて
 * いないことの証明を兼ねる）。
 *
 * 数値が 1 bit でも動いたら移植のどこかが変わっている — **tolerance 化は禁止**で、赤のまま
 * 止めて差分の内容（PNG バイト長 / 画素の統計 / 実物の PNG）を出す。ここを緩めると「移植
 * できた」の意味が消える。
 *
 * NOTE: ケースの大半（`f16+dit8-a8-attn8-s16` = 既定席）は opt-in 席で焼いた**実用行**で、参照行は
 * `f16-1024` の 1 ケースだけ（ADR 0110 決定 7 / ADR 0106 追記 2026-09-26 — 行のクラスは quant 席の
 * `session` が空か否かから導く）。実用層のカーネルを意図して変えたときは実用行を同じコミットで
 * `rewrite` する。参照行は凍結で、動いたら赤のまま止める。
 *
 * - **他の機の行を足す**のは `KARUME_REFERENCE=write`（既存の行には触らない）。
 * - **自分の機の行を焼き直す**のは `KARUME_REFERENCE=rewrite` だけで、「何が変わったのか」を
 *   先に言えたときに限る（旧→新はログに出る）。他環境の行は巻き込まない。
 * - 参照値を持たない機ではそのケースを**明示 SKIP** し、参照門（このファイルの末尾）が
 *   「参照が無いので全 SKIP」を赤で知らせる。
 * - 同じバイトを別経路から産む門（`fromPretrained-512` / `onEvent-1024` /
 *   `base-cfg-fromAssets-shards`）は**自分のケース ID の行**を持ちつつ、双子のケースの行とも
 *   一致することを検査する（経路間のビット同一が行の分割で緩まないように）。
 *
 * MUST: 資産は `models/karume-anima/`（untracked・実 GPU 機のローカル資産 — 公式 5 変種同居・
 * 既定 = anima-turbo-v1.1）。turbo（8 step / CFG 無し）の門は既定モデルで、**CFG≠1 の門**
 * （素の base = `model: "anima-v1.0"` — そこだけが 2 本目の text 経路と `cfgEulerStep` を
 * 通る）は同じリポのモデル選択で走る。無い環境と GPU 無し環境は理由を出して**明示 SKIP**する
 * （テストを消して無音で緑にしない — ADR 0005）。
 *
 * NOTE: 実配布形の門は基本ぜんぶ**取得層経由**（ローカル HTTP + `fromPretrained`）で通す
 * （本番と同じ graph-first + 逐次流しの経路）。part 分割そのものは全量面（`fromAssets`）でも
 * 読めるので、その 1 本だけ base を `Deno.readFile` + `fromAssets` で通し、取得層経由と
 * **同じ参照 sha256** を要求する（下の「全量面」節 — X2-101）。ファイルを `Deno.open` /
 * `Deno.readFile` で読むのはテスト側だけで、パッケージ本体は Web 標準 API のみ
 * （fs を持ち込まない — 横断不変条件）。
 *
 * NOTE: 越境参照（別リポの (repo, commit SHA)）の実資産門は、公式リポが自己完結（ADR 0087）
 * なので**追加変種の配布形** `models/karume-anima-extra/`（リリース時に公式リポの commit SHA へ
 * 越境で焼く — release-runbook）が持つ（下の「越境参照」節）。extra のミラーが無い環境では
 * その節だけを明示 SKIP する。
 */

import { createReadStream } from "node:fs";
import { createServer, type ServerResponse } from "node:http";
import { readFile, stat, writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assertEquals, assertFalse, assertRejects, assertStrictEquals } from "@std/assert";
import { type FileRef, parseManifest, resolveSelection, selectionRefs } from "@karume/hub";
import { acquireGpu, type GpuContext, type SessionDiagnostics } from "@karume/runtime";
import type { Manifest } from "@karume/hub";
import {
  type GenerateEvent,
  Pipeline,
  type Residency,
  type SamplerType,
  formatResolution,
  type GeneratedImage,
  type ImageSize,
} from "../../models/anima.ts";
import { encodePng } from "../../models/mod.ts";
import { parseAnimaPipelineConfig } from "../../models/src/anima/config.ts";
import { sigmaSchedule } from "../../models/src/anima/sampler.ts";
import { ANIMA_SPATIAL_COMPRESSION } from "../../models/src/anima/dit-tokens.ts";
import { HEADROOM_MARGIN_BYTES } from "../../models/src/session/headroom-margin.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { MemoryCacheStorage } from "../../models/tests/helpers/memory-cache.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import {
  announceCheck,
  expectedOf,
  openReferences,
  registerReferenceGate,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults } from "../../runtime/tests/helpers/results.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";

/** 資産の置き場（リポ直下 `models/karume-anima/` — 公式 5 変種同居・既定 = turbo-v1.1）。 */
const ASSETS_DIR = new URL("../../../models/karume-anima/", import.meta.url);
/** CFG の門が選ぶモデル（素の base — 既定の turbo は CFG=1 で uncond 側を通らない）。 */
const BASE_MODEL = "anima-v1.0";
/** 参照 digest の席（環境キーごとの行 — 追跡対象）。 */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/anima.json", import.meta.url),
);
/** 実物と決着の置き場（`outputs/verify/<環境キー>/<日付>_anima/` — 消して安全）。 */
const results = openResults("anima");

/**
 * 移行元デモの既定プロンプト（参照 PNG を焼いた時の文字列そのもの）。
 * ネガティブは `guidanceScale 1` で 1 文字も使われないので渡さない（渡すと fail loudly）。
 */
const PROMPT =
  "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, " +
  "smile, upper body, masterpiece, best quality";
const STEPS = 8;
const SEED = 42;

/**
 * 既定モデル（turbo）の門 3 本の条件。ケース ID は `<quant>-<解像度>`（ログのラベル・
 * `fixtures/references/anima.json` の鍵・実物のファイル名を兼ねる）。
 *
 * 内部整合: fromPretrained-512 と 512 / onEvent-1024 と 1024 がそれぞれビット同一
 * （別経路 + NaN 汚染した観測込みで同じバイト列）— 双子の検査として門に入っている。
 */
const REFERENCE = [
  { quant: "f16+dit8-a8-attn8-s16", resolution: { width: 1024, height: 1024 } },
  { quant: "f16+dit8-a8-attn8-s16", resolution: { width: 512, height: 512 } },
  { quant: "f16", resolution: { width: 1024, height: 1024 } },
  // 忠実プリセット（4 役とも上流の bf16 ビット列 — ADR 0131 決定 3・2026-10-17 に全モデルへ）。turbo の
  // 1 ケースだけ行を持つ（他のモデルの bf16 は export 時に上流 checkpoint とのバイト突合で担保）。
  { quant: "bf16", resolution: { width: 1024, height: 1024 } },
] as const satisfies readonly { quant: string; resolution: ImageSize }[];

/** ケース ID（ログのラベル / 参照値の鍵 / 実物のファイル名）。 */
const caseIdOf = (quant: string, resolution: ImageSize): string =>
  `${quant}-${formatResolution(resolution)}`;

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
const ASSETS_AVAILABLE = manifestText !== undefined;
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} に karume.json が無いため Anima の e2e を SKIP する` +
      "（移植の門は実資産でしか閉じない — exporter の dist.py で焼く）",
  );
}

const RUNNABLE = GPU_AVAILABLE && ASSETS_AVAILABLE;

/**
 * 追加変種の配布形（リポ直下 `models/karume-anima-extra/` — anima-wai-v1.0 /
 * anima-copycat-20260610）。共有部品は公式リポへの越境参照なので、公式のミラー
 * （{@link ASSETS_DIR}）と揃って初めて走る。
 */
const EXTRA_ASSETS_DIR = new URL("../../../models/karume-anima-extra/", import.meta.url);
const extraManifestText = await readTextIfPresent(new URL("karume.json", EXTRA_ASSETS_DIR));
if (extraManifestText === undefined) {
  console.warn(
    `[karume] ${EXTRA_ASSETS_DIR.pathname} に karume.json が無いため越境参照の e2e を SKIP する` +
      "（release-runbook の越境参照の手順で公式リポの commit SHA へ焼く）",
  );
}
const EXTRA_RUNNABLE = RUNNABLE && extraManifestText !== undefined;

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const readManifest = (): Manifest => parseManifest(manifestText as string);

// --- ローカル HTTP（HF 形の使い捨てサーバ）------------------------------------
//
// 実資産の門はほぼ全てこの土台に載る — part 分割された配布形を本番と同じ経路
// （graph-first + 逐次流し）で通すため。全量面（`Deno.readFile` + `fromAssets`）は下の
// 「全量面」節が 1 本だけ持つ。喋るのは hub が実際に叩く 2 経路（revision 解決 API・
// resolve URL）だけ。

const REPO = "karume-test/anima";
const REVISION_SHA = "1234567890abcdef1234567890abcdef12345678";
const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/**
 * 越境参照の repo → ローカルミラー。公式リポは自己完結（ADR 0087）で、越境で取るのは
 * extra（{@link EXTRA_ASSETS_DIR}）の共有部品（text_encoder / vae_decoder / tokenizer 2 本）
 * だけ。ミラーは repo 単位で、revision は取得元のキー（{@link originKey}）が manifest の宣言の
 * まま持つ — ミラーの中身がその commit と食い違えば取得層の integrity 検証が落とす。
 */
const CROSS_REPO_MIRRORS: ReadonlyMap<string, URL> = new Map([["hdae/karume-anima", ASSETS_DIR]]);

/** 取得元 1 つ（repo + revision）が配る内容。 */
type ServedOrigin = { readonly dir: URL; readonly paths: Set<string> };

/**
 * 取得元のキー。**repo だけでは足りない** — 同じ repo が別 revision で現れても取り違えない
 * （越境参照は宣言された commit SHA から取られ、セッションの SHA とは無関係）。
 */
const originKey = (repo: string, revision: string): string => `${repo}@${revision}`;

type AssetServer = { readonly addr: { readonly port: number }; shutdown(): Promise<void> };

/** 取得元の表どおりに HF の URL 形で配る一時サーバ（ポート自動割当・127.0.0.1 束縛）。 */
const serveAssets = (origins: ReadonlyMap<string, ServedOrigin>): Promise<AssetServer> =>
  new Promise((listening) => {
    const notFound = (response: ServerResponse): void => {
      response.writeHead(404, { "content-type": "text/plain" }).end("not found");
    };
    const server = createServer((request, response) => {
      void (async () => {
        const { pathname } = new URL(request.url ?? "/", "http://127.0.0.1");
        // revision 解決を要求するのはセッション repo だけ（越境参照は commit SHA 固定なので、
        // 取得層が解決要求そのものを出さない）。
        if (REVISION_RE.test(pathname)) {
          response
            .writeHead(200, { "content-type": "application/json" })
            .end(JSON.stringify({ sha: REVISION_SHA }));
          return;
        }
        const resolved = RESOLVE_RE.exec(pathname);
        if (resolved === null) return notFound(response);
        const [, repo, revision, rawPath] = resolved;
        const served = origins.get(originKey(repo, revision));
        const path = decodeURIComponent(rawPath);
        if (served === undefined || !served.paths.has(path)) return notFound(response);
        const file = new URL(path, served.dir);
        const { size } = await stat(file);
        response.writeHead(200, { "content-length": String(size) });
        createReadStream(file).pipe(response);
      })();
    });
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address !== null ? address.port : 0;
      listening({
        addr: { port },
        shutdown: () =>
          new Promise((done, fail) => {
            server.close((error) => (error === undefined ? done() : fail(error)));
            server.closeAllConnections();
          }),
      });
    });
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
  model?: string,
): Map<string, ServedOrigin> => {
  const origins = new Map<string, ServedOrigin>();
  const add = (repo: string, revision: string, mirror: URL, path: string): void => {
    const key = originKey(repo, revision);
    const served = origins.get(key) ?? { dir: mirror, paths: new Set<string>() };
    served.paths.add(path);
    origins.set(key, served);
  };
  add(REPO, REVISION_SHA, dir, "karume.json");
  const selection = resolveSelection(manifest, {
    quant,
    ...(model === undefined ? {} : { model }),
  });
  const refs: FileRef[] = [
    ...Object.keys(selection.containers).flatMap((name) => [...selection.containers[name].parts]),
    ...Object.keys(selection.assets).map((name) => selection.assets[name]),
  ];
  for (const ref of refs) {
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
  return channels
    .map((name, offset) => {
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
    })
    .join(" / ");
};

/**
 * 参照 sha と食い違ったときの報告。
 *
 * MUST: ここで tolerance に逃げない。参照は sha256 しか無い（参照 PNG のバイト列は持って
 * いない）ので**先頭差分位置は原理的に出せない** — 代わりに実物（結果の席に毎回残る）と
 * バイト長・画素統計を併せて人が突き合わせられる形にする。
 */
const mismatchReport = (
  label: string,
  image: GeneratedImage,
  png: Uint8Array<ArrayBuffer>,
  expected: string,
  actual: string,
  dumped: URL,
): string =>
  `${label}: 出力画素の sha256 が参照と一致しない\n` +
  `  期待 ${expected}\n  実際 ${actual}\n` +
  `  PNG ${png.length} バイト / 画像 ${image.width}×${image.height}\n` +
  `  画素 ${describePixels(image)}\n` +
  `  実物 ${dumped.pathname}（参照はバイト列ではなく sha256 のみなので先頭差分位置は出せない）`;

/**
 * sha を採る画素列: `encodePng` が IDAT に書く RGB8（行優先・alpha を落とした形）。`GeneratedImage.data` は
 * RGBA で alpha は常に 255（`encodePng` が検査する）なので情報は落ちず、実物の PNG を decode した画素列と
 * 同じバイト列になる — 行は実物から再計算できる（ADR 0106 追記決定 9）。
 */
const rgbPixels = (image: GeneratedImage): Uint8Array<ArrayBuffer> => {
  const pixels = new Uint8Array(new ArrayBuffer(image.width * image.height * 3));
  for (let source = 0, target = 0; source < image.data.length; source += 4, target += 3) {
    pixels[target] = image.data[source];
    pixels[target + 1] = image.data[source + 1];
    pixels[target + 2] = image.data[source + 2];
  }
  return pixels;
};

/**
 * 実物を残し、参照値と突き合わせ、決着を結果 JSON へ積む（全門共通の末端）。
 *
 * `twin` は「別経路で同じバイトが出るはず」の相手のケース ID。参照値が環境ごとの行になって
 * からは、経路間のビット同一を「同じ定数を渡す」形では持てないので、**双子の行と実測**を
 * 直接突き合わせる（相手の行がこの環境にまだ無いときだけ検査は成立しない）。
 */
const settlePng = async (
  label: string,
  image: GeneratedImage,
  png: Uint8Array<ArrayBuffer>,
  actual: string,
  elapsedMs: number,
  twin?: string,
): Promise<void> => {
  // 参照値を書きうる経路なので、キーを採ったアダプタと実行アダプタの同一性を先に見る。
  await assertRunningAdapter();
  const artifact = `${label}.png`;
  // 実物は成功・失敗を問わず毎回残す（次に割れたときの A/B の材料は、割れる前に要る）。
  const dumped = results.artifact(artifact);
  await writeFile(dumped, png);
  // 双子の行との突合は**判定だけ**先に済ませる。ここで投げると自分の決着が記録されないまま
  // 実物だけが席に残り、同じ日の前回の決着と組で読まれる（双子が同値なら自分の不一致でもある）。
  const twinSha = twin === undefined ? undefined : references.lookup(twin);
  const twinMismatch =
    twinSha === undefined || twinSha === actual
      ? undefined
      : `${twin} とビット同一でない（${twin} ${twinSha} / 実測 ${actual}）— ` +
        "別経路が同じバイトを産んでいない";
  const check = references.check(label, actual);
  announceCheck(label, check, actual);
  const expected = expectedOf(check);
  await results.record({
    id: label,
    status: twinMismatch === undefined ? check.status : "fail",
    ...(expected === undefined ? {} : { expected }),
    actual,
    artifact,
    elapsedMs,
    ...(twinMismatch === undefined ? {} : { note: twinMismatch }),
  });
  if (twinMismatch !== undefined) throw new Error(`${label} の PNG が ${twinMismatch}`);
  if (check.status === "fail") {
    throw new Error(mismatchReport(label, image, png, check.expected, actual, dumped));
  }
};

/**
 * 生成 → PNG → sha 突合。一致しない場合は緩めず、診断を付けて落とす。
 *
 * `onEvent` を渡した呼びも**同じバイト列**を要求する — 観測席が数値に 1 ビットも触って
 * いないことの直接証拠になる（下の onEvent 門。`twin` で観測なしのケースと突き合わせる）。
 * `sampler` は request 側の上書き席（省略時は manifest の `scheduler.type`）。
 */
const assertReferencePng = async (
  label: string,
  pipeline: Pipeline,
  resolution: ImageSize,
  options: {
    readonly sampler?: SamplerType;
    readonly onEvent?: (event: GenerateEvent) => void;
    readonly twin?: string;
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
  const actual = await sha256Hex(rgbPixels(image));
  const elapsedMs = performance.now() - started;
  console.log(
    `[e2e] ${label}: ${(elapsedMs / 1000).toFixed(1)}s / PNG ${png.length}B / pixels sha256 ${actual}`,
  );
  await settlePng(label, image, png, actual, elapsedMs, options.twin);
};

/** Injection seats for building the turbo distribution. */
type TurboOptions = {
  readonly caches?: CacheStorage;
  readonly onRunDiagnostics?: (component: string, diagnostics: SessionDiagnostics) => void;
  readonly residency?: Residency;
  /** 共有 GPU（呼び手の所有物 — 破棄は呼び手）。 */
  readonly gpu?: GpuContext;
};

/**
 * An opened turbo distribution. Disposing it releases the pipeline through its
 * [Symbol.asyncDispose] and then stops the server; later disposals return the first completion,
 * so a describe can close it inside an it and again in afterAll.
 */
type OpenedTurbo = AsyncDisposable & { readonly pipeline: Pipeline };

/**
 * turbo の配布形を**取得層経由**（ローカル HTTP + `fromPretrained`）で組む。
 * Returns it open: {@link withTurbo} and the describes that share one pipeline across their its
 * (opened in beforeAll, closed in afterAll) build it through this one path.
 *
 * MUST: `caches` は公開面の注入席から渡す（実 Cache Storage に数 GB を書かない）。呼び手が
 * 自分のインスタンスを渡した場合は、閉じた後にその中身を検査できる。
 */
const openTurbo = async (quant: string, options: TurboOptions): Promise<OpenedTurbo> => {
  const manifest = readManifest();
  const server = await serveAssets(servedOrigins(manifest, quant, ASSETS_DIR));
  let pipeline: Pipeline;
  try {
    pipeline = await Pipeline.fromPretrained(
      { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` },
      {
        quant,
        caches: options.caches ?? new MemoryCacheStorage(),
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
        ...(options.residency === undefined ? {} : { residency: options.residency }),
        ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      },
    );
  } catch (error) {
    await server.shutdown();
    throw error;
  }
  let closing: Promise<void> | undefined;
  const close = async (): Promise<void> => {
    try {
      // [Symbol.asyncDispose] 経由の解放をこの実 GPU 経路で検査する意図込み。
      await pipeline[Symbol.asyncDispose]();
    } finally {
      await server.shutdown();
    }
  };
  return {
    pipeline,
    [Symbol.asyncDispose]: () => (closing ??= close()),
  };
};

/** Builds the turbo distribution with {@link openTurbo}, passes it to `body`, then disposes it. */
const withTurbo = async (
  quant: string,
  options: TurboOptions,
  body: (pipeline: Pipeline) => Promise<void>,
): Promise<void> => {
  await using opened = await openTurbo(quant, options);
  await body(opened.pipeline);
};

for (const { quant, resolution } of REFERENCE) {
  const label = caseIdOf(quant, resolution);
  // request 側 `sampler:"euler"` の明示が manifest 既定（= 公式配布の宣言）とビット同一である
  // ことは、**生成を 1 本も増やさず**最短の 512 ケースだけを明示に振り替えて見る。既定経路の
  // 門は残りのケースと fromPretrained（同じ quant/512 を既定で回して同じ sha を要求する）が
  // 保つので、どちらの経路も裸のまま残らない。
  const sampler: SamplerType | undefined = resolution.width === 512 ? "euler" : undefined;
  test.skipIf(!RUNNABLE || references.lacksReference(label))(
    `e2e(実GPU): quant ${quant} / ${formatResolution(resolution)} / ${STEPS}step / ` +
      `seed ${SEED} の PNG が参照 sha256 と一致する` +
      (sampler === undefined ? "" : `（sampler:"${sampler}" 明示）`),
    async () => {
      await withTurbo(quant, {}, (pipeline) =>
        assertReferencePng(label, pipeline, resolution, {
          ...(sampler === undefined ? {} : { sampler }),
        }),
      );
    },
  );
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

/**
 * turbo 512²（`REFERENCE[1]` と同じ quant / 解像度 / steps / seed）を DPM++ 2M で回したケース。
 */
const TURBO_DPMPP_CASE = `${caseIdOf(REFERENCE[1].quant, REFERENCE[1].resolution)}-dpmpp`;

test.skipIf(!RUNNABLE || references.lacksReference(TURBO_DPMPP_CASE))(
  `e2e(実GPU): turbo ${formatResolution(REFERENCE[1].resolution)} を request の ` +
    `sampler:"dpmpp-2m" で回すと DPM++ 2M の参照 sha256 と一致する`,
  async () => {
    const { quant, resolution } = REFERENCE[1];
    await withTurbo(quant, {}, (pipeline) =>
      assertReferencePng(TURBO_DPMPP_CASE, pipeline, resolution, { sampler: "dpmpp-2m" }),
    );
  },
);

// --- CFG（素の base 配布形）--------------------------------------------------
//
// MUST: **CFG≠1 の経路にも実 GPU の門を置く**。`guidanceScale === 1` は uncond 側を 1 度も
// 計算しない（`needsUncond`）ので、上の 3 ケースは cond 1 本しか通っていない —
// CFG の分岐・2 本目の text 経路・`cfgEulerStep` の合成は**どれも実 GPU で 1 度も走って
// いなかった**（波 L で気付いた穴）。素の base 配布形は既定が CFG なので、そこを 1 ケース
// だけ固定する（解像度 512 は所要時間の都合 — CFG は 1 step が forward 2 本）。

/** CFG の門の条件（RTX 3080 Ti の行は 2026-08-22 実測）。 */
const BASE_REFERENCE = {
  quant: "f16+dit8-a8-attn8-s16",
  resolution: { width: 512, height: 512 },
  steps: 20,
  guidanceScale: 4,
  negativePrompt: "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
} as const;

/** セッション repo として配る配布形 1 つ（置き場・manifest・選ぶモデル）。 */
type DistributionSource = {
  readonly dir: URL;
  readonly manifest: Manifest;
  readonly model: string;
};

/** 組み上がった base の pipeline を {@link BASE_REFERENCE} のノブで 1 枚焼き、sha を突き合わせる。 */
const assertBasePng = async (
  label: string,
  pipeline: Pipeline,
  options: { readonly sampler?: SamplerType; readonly twin?: string } = {},
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
  const actual = await sha256Hex(rgbPixels(image));
  const elapsedMs = performance.now() - started;
  console.log(
    `[e2e] ${label}: ${(elapsedMs / 1000).toFixed(1)}s / PNG ${png.length}B / pixels sha256 ${actual}`,
  );
  await settlePng(label, image, png, actual, elapsedMs, options.twin);
};

/**
 * 素の base を {@link BASE_REFERENCE} のノブで 1 枚焼き、参照 sha256 と突き合わせる。
 *
 * 経路は**取得層経由**（ローカル HTTP + `fromPretrained`）— この配布形の text_encoder と
 * transformer は part 分割されていて、越境参照と併せて解けるのは取得層だけ
 * （デモの `--source` と同型 — `examples/shared/local-dist-server.ts`）。分割形は全量面
 * （`fromAssets`）でも読めるが、そちらは全 part がホスト RAM に同時に載る面で、門としては
 * 下の 1 本（同じ参照 sha を要求する）で別に閉じる。
 *
 * `source` はセッション repo として配る配布形（省略時は公式リポの素の base）。越境参照の門
 * （下の「越境参照」節）が extra の変種を同じノブで通すのに使う。`quant` は席の上書き
 * （省略時は {@link BASE_REFERENCE} の実用席 — 忠実席 `bf16` の門が使う）。
 */
const assertBaseReferencePng = async (
  label: string,
  options: {
    readonly sampler?: SamplerType;
    readonly source?: DistributionSource;
    readonly quant?: string;
  } = {},
): Promise<void> => {
  const { dir, manifest, model } = options.source ?? {
    dir: ASSETS_DIR,
    manifest: readManifest(),
    model: BASE_MODEL,
  };
  const quant = options.quant ?? BASE_REFERENCE.quant;
  const server = await serveAssets(servedOrigins(manifest, quant, dir, model));
  try {
    // MUST: `caches` は公開面の注入席から渡す（実 Cache Storage に数 GB を書かない）。
    await using pipeline = await Pipeline.fromPretrained(
      { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` },
      { model, quant, caches: new MemoryCacheStorage() },
    );
    await assertBasePng(label, pipeline, options);
  } finally {
    await server.shutdown();
  }
};

test.skipIf(!RUNNABLE || references.lacksReference("base-cfg"))(
  `e2e(実GPU): 素の base / CFG ${BASE_REFERENCE.guidanceScale} / ` +
    `${BASE_REFERENCE.steps}step の PNG が参照 sha256 と一致する`,
  () => assertBaseReferencePng("base-cfg"),
);

// 忠実プリセット `bf16`（ADR 0131 決定 3 — 4 役とも上流 checkpoint の BF16 のビット列・演算は f32）。
// CFG の門と同じノブで 1 枚焼き、この機の行と突き合わせる。f16 席とは丸めが違う（f16 は指数域が
// 狭い）ので別の行を持ち、双子ではない。bf16 の重みはカーネル内の bitcast で展開される
// （ADR 0131 決定 1）— この門はその経路が配布形の実資産で通ることまで含めて固定する。
test.skipIf(!RUNNABLE || references.lacksReference("base-cfg-bf16"))(
  `e2e(実GPU): 素の base の忠実席 bf16 / CFG ${BASE_REFERENCE.guidanceScale} / ` +
    `${BASE_REFERENCE.steps}step の PNG が参照 sha256 と一致する`,
  () => assertBaseReferencePng("base-cfg-bf16", { quant: "bf16" }),
);

// --- 全量面（fromAssets）で分割配布形を読む -----------------------------------
//
// 「`fromPretrained` で読める配布形は `fromAssets` でも読める」（X2-101）の門。実配布形の
// 部品は容器 1 本が複数 part に割れていて、全量面へは `transformer[0]` / `transformer[1]` /
// … の綴りで渡す — その Record をそのまま全量面へ渡し、**上の CFG の門と同じ参照 sha256** を
// 要求する。同じバイトを別の面から流し込んでいるだけなので、1 ビットも動かないのが正しい
// （動いたら part の並びか descriptor の扱いが壊れている）。

/** ローカル配布形を全量読みする（`examples/shared/local-assets.ts` と同型の面）。 */
const readLocalAssets = async (
  dir: URL,
  manifest: Manifest,
  quant: string,
): Promise<{ manifest: Manifest; assets: Record<string, Uint8Array<ArrayBuffer>> }> => {
  const selection = resolveSelection(manifest, { quant, model: BASE_MODEL });
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const read = async (key: string, ref: FileRef): Promise<void> => {
    // 越境参照は (repo, commit SHA) からしか取れない — 全量面では解けないので fail loudly。
    if (ref.repo !== undefined) {
      throw new Error(`全量面では越境参照 '${ref.repo}' を解けない（キー ${key}）`);
    }
    const bytes = byPath.get(ref.path) ?? (await readFile(new URL(ref.path, dir)));
    byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  // MUST: 長さ 0 の part も並べる（添字が容器の中の id — 飛ばすと以降が 1 つずつ繰り上がる）。
  for (const name of Object.keys(selection.containers)) {
    const { parts } = selection.containers[name];
    for (const [index, ref] of parts.entries()) await read(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selection.assets)) await read(name, selection.assets[name]);
  return { manifest, assets };
};

test.skipIf(!RUNNABLE || references.lacksReference("base-cfg-fromAssets-shards"))(
  "e2e(実GPU): 分割配布形を fromAssets（全量面）で組んでも 素の base の参照 sha256 と一致する",
  async () => {
    const { quant } = BASE_REFERENCE;
    const input = await readLocalAssets(ASSETS_DIR, readManifest(), quant);
    // 分割形であること自体をこの門で確かめる — 単一形の配布形へ戻った日には、黙って
    // 「全量面の門」に化けるのではなく理由を出して落とす（門の意味が消える方が危ない）。
    const partKeys = Object.keys(input.assets).filter((key) => key.endsWith("]"));
    if (partKeys.length < 2) {
      throw new Error(
        `この配布形は part 分割されていない（キー: ${Object.keys(input.assets).join(" / ")}）`,
      );
    }
    await using pipeline = await Pipeline.fromAssets(input, { model: BASE_MODEL, quant });
    await assertBasePng("base-cfg-fromAssets-shards", pipeline, { twin: "base-cfg" });
  },
);

/**
 * 素の base を {@link BASE_REFERENCE} と同じノブ（CFG 4 / 20step / seed 42 / 同じネガティブ）で
 * DPM++ 2M へ振ったケース（RTX 3080 Ti の行は d2e0484 の golden — daf86af で退役）。
 *
 * CFG≠1 の経路にも上書き席の門を置く — DPM++ 2M の 2 次項は cond/uncond の**合成後**の値を
 * 履歴に持つので、turbo（CFG 無し）の一致だけでは合成と履歴の噛み合わせが裸のまま残る。
 */
test.skipIf(!RUNNABLE || references.lacksReference("base-cfg-dpmpp"))(
  `e2e(実GPU): 素の base を request の sampler:"dpmpp-2m" で回すと ` +
    `DPM++ 2M の参照 sha256 と一致する`,
  () => assertBaseReferencePng("base-cfg-dpmpp", { sampler: "dpmpp-2m" }),
);

// --- 越境参照（extra の変種）--------------------------------------------------
//
// extra の変種は transformer / text_conditioner だけを自前で持ち、text_encoder /
// vae_decoder / tokenizer 2 本を公式リポの (repo, commit SHA) への越境参照で焼く
// （ADR 0038 §7）。取得層がそれを**宣言された (repo, revision) のまま**取りに行くことを、
// ローカル HTTP が取得元ごとに配り分ける形で実資産に通す — 越境の path はセッション repo
// 側には 1 本も載せないので、取得層がセッション repo へ取り違えれば 404 で落ちる。
//
// ノブは素の base の CFG の門と同じ（extra の変種はどれも CFG 既定の fine-tune）。

/** extra の変種（ケース ID = `extra-<モデル>-cfg`）。 */
const EXTRA_MODELS = ["anima-wai-v1.0", "anima-copycat-20260610"] as const;
const extraCaseIdOf = (model: string): string => `extra-${model}-cfg`;

for (const model of EXTRA_MODELS) {
  const label = extraCaseIdOf(model);
  test.skipIf(!EXTRA_RUNNABLE || references.lacksReference(label))(
    `e2e(実GPU): extra ${model} を越境参照込みで組み、CFG ${BASE_REFERENCE.guidanceScale} / ` +
      `${BASE_REFERENCE.steps}step の PNG が参照 sha256 と一致する`,
    async () => {
      const manifest = parseManifest(extraManifestText as string);
      const sessionOrigin = originKey(REPO, REVISION_SHA);
      const crossOrigins = [
        ...servedOrigins(manifest, BASE_REFERENCE.quant, EXTRA_ASSETS_DIR, model).keys(),
      ].filter((key) => key !== sessionOrigin);
      // 越境を 1 本も含まない配布形へ戻った日に、黙って「越境の門」に化けないように落とす。
      if (crossOrigins.length === 0) {
        throw new Error(`${model} の選択が越境参照を 1 本も含まない（門の意味が消える）`);
      }
      await assertBaseReferencePng(label, {
        source: { dir: EXTRA_ASSETS_DIR, manifest, model },
      });
    },
  );
}

// --- fromPretrained（取得層込み）--------------------------------------------
//
// `loadManifest` → `resolveSelection` → 取得 → 構築の**実経路**は turbo の門が全て通るように
// なったが、そこに**注入席とキャッシュの実体**を突き合わせるのはこの 1 本だけ。manifest の実
// sha256 による integrity 検証が走る経路でもあり、資産の 3 点セット（path/size/sha256）が
// 現物と合っていることの門を兼ねる。
//
// 解像度が 512 なのは RAM の都合（取得層のメモリキャッシュが資産の写しを持つ）。ビット一致の
// 門としては 1024 と同格 — 上の 512 ケースと**同じバイト**が出ることを双子の検査で要求する。

test.skipIf(!RUNNABLE || references.lacksReference("fromPretrained-512"))(
  "e2e(実GPU): fromPretrained（取得層 + integrity 検証）の PNG が参照 sha256 と一致する",
  async () => {
    const quant = "f16+dit8-a8-attn8-s16";
    const resolution: ImageSize = { width: 512, height: 512 };
    const fetched = selectionRefs(resolveSelection(readManifest(), { quant }));
    const caches = new MemoryCacheStorage();
    // 観測席（onRunDiagnostics）の run 回数もこの実生成で併せて固定する（DiT は 1 step =
    // 1 run・text 系は各 1 run）。門の sha256 判定には一切影響しない追加観測。
    const observed: string[] = [];
    await withTurbo(
      quant,
      { caches, onRunDiagnostics: (component) => observed.push(component) },
      (pipeline) =>
        assertReferencePng("fromPretrained-512", pipeline, resolution, {
          twin: caseIdOf(REFERENCE[1].quant, REFERENCE[1].resolution),
        }),
    );
    const counts = new Map<string, number>();
    for (const component of observed) counts.set(component, (counts.get(component) ?? 0) + 1);
    const runsOf = (component: string): number => counts.get(component) ?? 0;
    if (
      runsOf("text_encoder") !== 1 ||
      runsOf("text_conditioner") !== 1 ||
      runsOf("transformer") !== STEPS ||
      runsOf("vae_decoder") < 1
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
    // 期待は **manifest 1 本 + 一意な (repo, path, sha256) の本数** — 資産のキーは取得層 HF 層の
    // 内容キー `["hf", "model", repo, path, sha256]`（hub の `sources/hf.ts`）で path を含むので、
    // 同一バイトでも path が違えば別エントリになる（越境参照ぶんは repo が越境先に変わるだけ）。
    // NOTE: hub はこのキーの組み立てを公開面に出していないので、同じ 3 要素をここで組む。
    let entries = 0;
    for (const namespace of caches.namespaces.values()) entries += namespace.entries.size;
    const contentKeys = new Set(
      fetched.map((ref) => JSON.stringify([ref.repo ?? REPO, ref.path, ref.sha256])),
    );
    const expected = contentKeys.size + 1;
    if (entries !== expected) {
      throw new Error(`注入したキャッシュのエントリ数が ${entries}（期待 ${expected}）`);
    }
  },
);

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

describe.skipIf(!RUNNABLE || references.lacksReference("onEvent-1024"))(
  "e2e(実GPU): onEvent の観測は数値に触らない（イベント列 / 途中 latent / 中断）",
  () => {
    const quant = "f16+dit8-a8-attn8-s16";
    const resolution: ImageSize = { width: 1024, height: 1024 };
    const runs: string[] = [];
    let scheduler: ReturnType<typeof parseAnimaPipelineConfig>["scheduler"];
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      const manifest = readManifest();
      ({ scheduler } = parseAnimaPipelineConfig(
        manifest.models[manifest.defaultModel].pipelineConfig,
      ));
      turbo = await openTurbo(quant, {
        onRunDiagnostics: (component) => runs.push(component),
      });
      pipeline = turbo.pipeline;
    });
    // After both its: dispose, then check that the disposed pipeline refuses to generate.
    // 中断した生成の後でも解放が正常に効くこと（直列化鎖は失敗を次へ持ち越さない）はここで通る。
    afterAll(async () => {
      if (turbo === undefined) return;
      await turbo[Symbol.asyncDispose]();
      await assertRejects(
        () => pipeline.generate({ prompt: PROMPT, steps: 2, seed: SEED }),
        Error,
        "after dispose",
      );
    });

    it("イベント列と途中 latent を固定し、PNG は参照 sha256 のまま", async () => {
      const log: string[] = [];
      const shapes: string[] = [];
      const lengths: number[] = [];
      let firstStep: (() => { data: Float32Array<ArrayBuffer> }) | undefined;
      let firstStepData: Float32Array | undefined;
      await assertReferencePng("onEvent-1024", pipeline, resolution, {
        twin: caseIdOf(REFERENCE[0].quant, REFERENCE[0].resolution),
        onEvent: (event) => {
          if (event.kind === "stage") {
            log.push(`stage:${event.component}:${event.at}`);
            return;
          }
          if (event.kind === "vae-tile") {
            log.push(`tile:${event.tile}/${event.tiles}`);
            return;
          }
          // 既定の段ごと運転では出ない — 出たら下の期待列と食い違って落ちる。
          if (event.kind === "residency") {
            log.push(`residency:${event.action}/${event.reason}`);
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

    it("コールバックの throw は生成ごと落とす（step 粒度の中断）", async () => {
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
                    : event.kind === "residency"
                      ? `residency:${event.action}/${event.reason}`
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
  },
);

// --- 公開契約（直列化 / dispose / 共有 GPU の所有権）--------------------------
//
// `Pipeline` が掲げる公開契約は 4 つ: ①並行 `generate` は待たされて順に走る
// ②`dispose` は in-flight の完了を待つ（flush-before-destroy）③2 度目以降の `dispose` は
// 同じ完了を返す ④渡された `GpuContext` は破棄しない。GPU 無しの
// `anima_pipeline_test.ts` はコンストラクタが private なので**結線**を組めず、
// `serial_test.ts` が縛るのは `createOperationChain` 単体でしかない。結線そのものは
// パイプライン実体でしか見えないのでここに置く。
//
// MUST: 生成は最小構成（512² / 2 step）で回す — ここで見るのは順序であって画素ではない。

describe.skipIf(!RUNNABLE)(
  "e2e(実GPU): 並行 generate は直列に走り、dispose は in-flight を待ってから破棄する",
  () => {
    const { quant } = REFERENCE[1];
    const resolution: ImageSize = { width: 512, height: 512 };
    const request = { prompt: PROMPT, resolution, steps: 2, seed: SEED };
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      turbo = await openTurbo(quant, {});
      pipeline = turbo.pipeline;
    });
    afterAll(() => turbo?.[Symbol.asyncDispose]());

    it("並行に投げた 2 本の段が交互にならない", async () => {
      const log: string[] = [];
      const trace = (tag: string) => (event: GenerateEvent) => {
        if (event.kind === "stage") log.push(`${tag}:${event.component}:${event.at}`);
      };
      // await せずに 2 本投げる（鎖が外れていれば段が交互に現れる）。
      await Promise.all([
        pipeline.generate({ ...request, onEvent: trace("a") }),
        pipeline.generate({ ...request, onEvent: trace("b") }),
      ]);
      const firstEnd = log.indexOf("a:vae_decoder:end");
      const secondStart = log.indexOf("b:text_encoder:start");
      assertEquals(firstEnd >= 0 && secondStart >= 0, true, `段の記録が足りない: ${log}`);
      assertEquals(
        firstEnd < secondStart,
        true,
        `1 本目の最終段より前に 2 本目が始まっている（鎖が外れている）: ${log}`,
      );
    });

    it("dispose は生成の完了より後に解決し、2 度目も同じ完了を返す", async () => {
      const order: string[] = [];
      const running = pipeline.generate(request).then(() => {
        order.push("generate");
      });
      const disposal = pipeline.dispose();
      assertStrictEquals(pipeline.dispose(), disposal, "2 度目の dispose が別の完了を返す");
      await Promise.all([running, disposal.then(() => order.push("dispose"))]);
      // 破棄が先に解決すると、消費側は「解放済み」と見なして次へ進んでしまう。
      assertEquals(order, ["generate", "dispose"]);
    });
  },
);

test.skipIf(!RUNNABLE)(
  "e2e(実GPU): 渡した GpuContext は dispose で破棄しない（所有権は渡した側）",
  async () => {
    const { quant } = REFERENCE[1];
    const server = await serveAssets(servedOrigins(readManifest(), quant, ASSETS_DIR));
    // 共有 GPU は呼び出し側の所有物。破棄してしまうと、同じ device を使う他の
    // パイプライン（examples は 1 つの GpuContext を家族間で回す）が道連れになる。
    const gpu = await acquireGpu();
    try {
      const hub = { repo: REPO, hubUrl: `http://127.0.0.1:${server.addr.port}` };
      const first = await Pipeline.fromPretrained(hub, {
        quant,
        gpu,
        caches: new MemoryCacheStorage(),
      });
      await first.dispose();
      // device が破棄されていれば、2 本目の構築（Session の重みアップロード）が落ちる。
      const second = await Pipeline.fromPretrained(hub, {
        quant,
        gpu,
        caches: new MemoryCacheStorage(),
      });
      await second.dispose();
    } finally {
      gpu.destroy();
      await server.shutdown();
    }
  },
);

// --- DiT 常駐（residency — ADR 0112）------------------------------------------
//
// 常駐は数値を変えない設計なので、**新しい参照行を 1 本も足さない**（`fixtures/references/anima.json`
// が無変更であることが門）。全ケースを既存の行（turbo 1024 / 512）の双子として回し、常駐の有無・
// 解像度の往復・持ち越しと解放・中断後の作り直し・空き不足での先回りの退避のどれを通っても同じバイトが
// 出ることを要求する。
//
// 寿命は `onRunDiagnostics` の transformer の診断で観測する（同じ Session を使い回していれば
// generate を跨いでも導出済み計画に当たり、backing の構築回数も重みのバイト数も増えない —
// 段ごと運転なら generate ごとに新しい Session なので最初の run は必ず外れる）。
//
// MUST: 双子の突合は `references.check` / `record` を通さない（通すと作るモードで新しい行が書かれる）。
// 双子の行がこの環境に無い機ではケースごと SKIP する（行は既存ケースの走行が作る）。

/** 常駐の双子に使う既存の行（turbo 1024 / 512 — {@link REFERENCE} の 0 と 1）。 */
const RESIDENCY_TWINS = {
  1024: caseIdOf(REFERENCE[0].quant, REFERENCE[0].resolution),
  512: caseIdOf(REFERENCE[1].quant, REFERENCE[1].resolution),
} as const;

const RESIDENCY_RUNNABLE =
  RUNNABLE &&
  references.lookup(RESIDENCY_TWINS[1024]) !== undefined &&
  references.lookup(RESIDENCY_TWINS[512]) !== undefined;

/** イベント 1 つをログの綴りにする（段 / 常駐だけ — step とタイルは数だけ見れば足りる）。 */
const residencyLogOf = (event: GenerateEvent): string | undefined => {
  if (event.kind === "stage") return `stage:${event.component}:${event.at}`;
  if (event.kind === "residency") return `residency:${event.action}/${event.reason}`;
  return undefined;
};

/**
 * 既存の行を双子に 1 枚焼いて突き合わせる（行は書かない）。戻り値はその generate の段 / 常駐の
 * イベント列。実物と決着は結果の席に残す（`label` は結果の id — 参照行の鍵ではない）。
 * `onLogged` はイベント列に 1 つ積むたびに待って呼ぶ（generate の途中で GPU の状態を変える口 —
 * `onEvent` は待たれるので、呼び終わるまで次の段へ進まない）。
 */
const assertResidencyTwin = async (
  label: string,
  pipeline: Pipeline,
  size: 512 | 1024,
  residency: Residency | undefined,
  onLogged?: (entry: string) => Promise<void>,
): Promise<string[]> => {
  const twin = RESIDENCY_TWINS[size];
  const log: string[] = [];
  const started = performance.now();
  const image = await pipeline.generate({
    prompt: PROMPT,
    resolution: { width: size, height: size },
    steps: STEPS,
    seed: SEED,
    ...(residency === undefined ? {} : { residency }),
    onEvent: async (event) => {
      const entry = residencyLogOf(event);
      if (entry === undefined) return;
      log.push(entry);
      await onLogged?.(entry);
    },
  });
  const png = await encodePng(image.data, image.width, image.height);
  const actual = await sha256Hex(rgbPixels(image));
  const elapsedMs = performance.now() - started;
  console.log(
    `[e2e] ${label}: ${(elapsedMs / 1000).toFixed(1)}s / PNG ${png.length}B / pixels sha256 ${actual}`,
  );
  await assertRunningAdapter();
  const expected = references.lookup(twin);
  if (expected === undefined) {
    throw new Error(`${twin} の行がこの環境に無い（SKIP の条件と食い違う）`);
  }
  const artifact = `${label}.png`;
  const dumped = results.artifact(artifact);
  await writeFile(dumped, png);
  await results.record({
    id: label,
    status: actual === expected ? "pass" : "fail",
    expected,
    actual,
    artifact,
    elapsedMs,
    note: `${twin} の双子（DiT 常駐 — 参照行は持たない）`,
  });
  if (actual !== expected) {
    throw new Error(
      mismatchReport(`${label}（双子 ${twin}）`, image, png, expected, actual, dumped),
    );
  }
  return log;
};

/** transformer の run ごとの診断を generate 単位で束ねる観測席。 */
const transformerRuns = () => {
  const perGenerate: SessionDiagnostics[][] = [];
  return {
    perGenerate,
    /** 次の generate の束を始める（generate の直前に呼ぶ）。 */
    begin: () => perGenerate.push([]),
    observe: (component: string, diagnostics: SessionDiagnostics): void => {
      if (component === "transformer") perGenerate.at(-1)?.push(diagnostics);
    },
  };
};

/**
 * 段ごとの `stage` の前後と、常駐イベントを並べた期待列。常駐イベントが出る場所は 3 つだけ:
 * DiT 段の end の直前（`residency`）と、先回りの退避 `evicted/headroom` が出る 2 点 — text_encoder の
 * start より前（`beforeText`）と、transformer の end と vae_decoder の start の間（`beforeVae`）。
 * 後の 2 つは既定で空なので、先回りの退避を狙って起こすケース以外では「どちらの点にも何も出ない」まで
 * を縛る。
 */
const expectedStages = (
  residency: readonly string[],
  headroom: {
    readonly beforeText?: readonly string[];
    readonly beforeVae?: readonly string[];
  } = {},
): string[] => [
  ...(headroom.beforeText ?? []).map((entry) => `residency:${entry}`),
  "stage:text_encoder:start",
  "stage:text_encoder:end",
  "stage:text_conditioner:start",
  "stage:text_conditioner:end",
  "stage:transformer:start",
  ...residency.map((entry) => `residency:${entry}`),
  "stage:transformer:end",
  ...(headroom.beforeVae ?? []).map((entry) => `residency:${entry}`),
  "stage:vae_decoder:start",
  "stage:vae_decoder:end",
];

describe.skipIf(!RESIDENCY_RUNNABLE)(
  'e2e(実GPU): residency "transformer" は同じケースの 2 回目も解像度の往復も参照 sha256 の双子と' +
    "一致し、DiT を読み直さない",
  () => {
    const { quant } = REFERENCE[0];
    const runs = transformerRuns();
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      turbo = await openTurbo(quant, { residency: "transformer", onRunDiagnostics: runs.observe });
      pipeline = turbo.pipeline;
    });
    // The checks over every generate of (a) and (b) run after both its, whatever their outcome,
    // and before the pipeline is disposed.
    afterAll(async () => {
      if (turbo === undefined) return;
      try {
        const all = runs.perGenerate.flat();
        for (const diagnostics of all) {
          assertEquals(
            diagnostics.planBacking.retainedCount <= 1,
            true,
            `backing を ${diagnostics.planBacking.retainedCount} 本保持している`,
          );
        }
        // 重みアリーナは params キャッシュの実体も抱える（`SessionDiagnostics.weights` の doc）ので、
        // 解像度を変えて新しい計画を導いた run では params のぶんだけ本数とバイト数が増える。
        // 「重みを読み直していない」の厳密な形は「最初の run 以降に増えた本数 = 各 run が確保した
        // params の本数」— initializer の再アップロードが 1 本でも混ざれば右辺より大きくなる。
        const [first, ...later] = all;
        const paramsAllocated = later.reduce(
          (sum, diagnostics) => sum + (diagnostics.lastRunParams?.allocCount ?? 0),
          0,
        );
        assertEquals(
          (later.at(-1) ?? first).weights.allocCount - first.weights.allocCount,
          paramsAllocated,
          "generate を跨いで重みアリーナに params 以外の確保が増えた（重みの読み直し）",
        );
      } finally {
        await turbo[Symbol.asyncDispose]();
      }
    });

    const generate = async (label: string, size: 512 | 1024): Promise<void> => {
      runs.begin();
      const log = await assertResidencyTwin(label, pipeline, size, undefined);
      assertEquals(log, expectedStages(["retained/request"]), `${label} のイベント列`);
    };
    const weightsOf = (diagnostics: SessionDiagnostics): number =>
      diagnostics.weights.allocatedBytes;

    it("(a) 1024² を 2 回: 2 回とも双子の行と一致し、2 回目は計画にも backing にも当たる", async () => {
      await generate("residency-1024-first", 1024);
      await generate("residency-1024-second", 1024);
      const [first, second] = runs.perGenerate;
      assertEquals(first.length, STEPS);
      assertEquals(second.length, STEPS);
      // 段ごと運転なら 2 回目の最初の run は新しい Session なので必ず外れる（常駐の直接証拠）。
      assertEquals(
        second[0].lastRunPrepared?.hit,
        true,
        "2 回目の最初の run が導出済み計画に当たらない",
      );
      assertEquals(
        second.at(-1)?.planBacking.buildCount,
        first.at(-1)?.planBacking.buildCount,
        "2 回目の generate で backing を作り直している",
      );
      assertEquals(weightsOf(second[0]), weightsOf(first[0]), "2 回目で重みのバイト数が増えた");
    });

    it("(b) 512² → 1024² → 512² の往復: 各解像度の双子と一致し、保持する backing は 1 本まで", async () => {
      await generate("residency-roundtrip-512-first", 512);
      await generate("residency-roundtrip-1024", 1024);
      await generate("residency-roundtrip-512-second", 512);
    });
  },
);

describe.skipIf(!RESIDENCY_RUNNABLE)(
  "e2e(実GPU): residency の request 上書き（持つ → 手放す）・中断後の作り直し・常駐ありの dispose",
  () => {
    const { quant } = REFERENCE[1];
    const runs = transformerRuns();
    let pipeline: Pipeline;
    let turbo: OpenedTurbo | undefined;
    beforeAll(async () => {
      // 構築オプションは既定（per-stage）のまま、request だけで常駐を切り替える。
      turbo = await openTurbo(quant, { onRunDiagnostics: runs.observe });
      pipeline = turbo.pipeline;
    });
    afterAll(() => turbo?.[Symbol.asyncDispose]());

    it("(c) request 1 = transformer で持ち、request 2 = per-stage が使ってから手放す", async () => {
      runs.begin();
      assertEquals(
        await assertResidencyTwin("residency-request-retain", pipeline, 512, "transformer"),
        expectedStages(["retained/request"]),
      );
      runs.begin();
      assertEquals(
        await assertResidencyTwin("residency-request-release", pipeline, 512, "per-stage"),
        expectedStages(["released/request"]),
      );
      // request 2 は持ち越した DiT を使った（読み直していない）。
      assertEquals(
        runs.perGenerate[1][0].lastRunPrepared?.hit,
        true,
        "持ち越した DiT を使っていない",
      );
    });

    it("(d) onEvent の throw で DiT 段を中断すると常駐 DiT を捨て、次の generate は作り直して一致する", async () => {
      runs.begin();
      await assertResidencyTwin("residency-before-abort", pipeline, 512, "transformer");
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
                event.kind === "denoise-step"
                  ? `step:${event.step}`
                  : (residencyLogOf(event) ?? `tile:${event.kind}`),
              );
              if (event.kind === "denoise-step" && event.step === abortAt) {
                throw new Error("購読側で中断");
              }
            },
          }),
        Error,
        "購読側で中断",
      );
      // 中断の直後に常駐 DiT を捨てたことを名乗り、段は正常終了しない（end は出ない）。
      assertEquals(seen.slice(-2), [`step:${abortAt}`, "residency:evicted/failure"]);
      assertFalse(seen.includes("stage:transformer:end"), "中断したのに段が正常終了している");

      runs.begin();
      assertEquals(
        await assertResidencyTwin("residency-after-abort", pipeline, 512, "transformer"),
        expectedStages(["retained/request"]),
      );
      // 捨てた後なので、最初の run は新しい Session（導出済み計画に当たらない）。
      assertEquals(
        runs.perGenerate.at(-1)?.[0].lastRunPrepared?.hit,
        false,
        "捨てたはずの DiT を使っている",
      );
    });
    // NOTE: 縛るのは「常駐を持ったままの dispose が reject せずに済み、以後の生成を受けない」まで。
    // 常駐 DiT が実際に畳まれたことは観測しない — GpuContext / device に live バイトの診断が無く、
    // 共有 GPU で「dispose 後に DiT 相当が確保できる」を見ても VRAM に余裕のある機では常駐が
    // 残っていても通る（判別にならない）。「常駐 DiT を畳んでから GPU を破棄する」順序と失敗の扱いは
    // `anima_residency_test.ts` の `disposeResidencyThenGpu` が GPU 無しで縛る。
    it("(e) 常駐を持ったままの dispose は投げずに済み、以後は生成を受けない", async () => {
      // The dispose happens here, with the DiT still retained by the last generate of (d)
      // (it ends retained); afterAll's dispose then returns this same completion.
      await turbo?.[Symbol.asyncDispose]();
      await assertRejects(
        () => pipeline.generate({ prompt: PROMPT, steps: 2, seed: SEED }),
        Error,
        "after dispose",
      );
    });
  },
);

// 先回りの退避（H-35）: 常駐 DiT の上の空きを実際に削り、generate が段を張る前に常駐 DiT を退避して
// （OOM を踏まずに）通ることを見る。量る点は 2 つ（`residency.ts` の「退避の 2 本の線」）なので行も 2 本。
// ダミーは書き込まない（`createBuffer` だけ — 書き込みの staging が device を失わせうる経路を門の側で
// 踏まない）。
//
// - text 段の前: 2 回目の generate の前に「1 GiB のダミーを OOM まで積んでから 1 本だけ返す」— 残りは
//   1〜2 GiB（確保の線の手前）で、text_encoder の必要量（見積りのピーク + 最大 part + 余裕 —
//   `residency.ts` の `stageMemoryNeed`）より小さく、常駐 DiT を退避すれば全段が入る。
// - VAE 段の前: 最初の generate の transformer の end（DiT を席に載せた直後）で、1 GiB に続けて
//   256 MiB のダミーも OOM まで積む — 残りは 256 MiB 未満で、どの段の必要量（余裕 512 MiB を含む）より
//   も小さいので、VAE の見積りの大きさに依らず必ず退避する。退避すれば DiT の重みぶんが空き、VAE が入る。

/** ダミー 1 本の大きさ（`maxBufferSize` がこれより小さい device ではそちらに合わせる）。 */
const HEADROOM_DUMMY_BYTES = 1024 * 1024 * 1024;
/** 空きを詰め切るダミーの大きさ（{@link HEADROOM_MARGIN_BYTES} より小さいことが決定性の根拠）。 */
const HEADROOM_TIGHT_DUMMY_BYTES = 256 * 1024 * 1024;
/** 1 回の積み上げの本数の上限（OOM が返らない実装で無限に回らないための門）。 */
const HEADROOM_DUMMY_LIMIT = 32;

/** `size` のダミーを out-of-memory まで `dummies` に積む（積めた本数を返す）。 */
const fillUntilOom = async (
  gpu: GpuContext,
  size: number,
  dummies: GPUBuffer[],
): Promise<number> => {
  for (let count = 0; ; count += 1) {
    if (count >= HEADROOM_DUMMY_LIMIT) {
      throw new Error(`ダミー ${count} 本を積んでも out-of-memory が返らない`);
    }
    gpu.device.pushErrorScope("out-of-memory");
    const buffer = gpu.device.createBuffer({
      label: `e2e-headroom-dummy-${dummies.length}`,
      size,
      usage: GPUBufferUsage.STORAGE,
    });
    const failure = await gpu.device.popErrorScope();
    if (failure !== null) {
      buffer.destroy();
      return count;
    }
    dummies.push(buffer);
  }
};

/**
 * 空きを削る（`tight` = false: 1 GiB を OOM まで積んで 1 本返す / true: 続けて 256 MiB も OOM まで積む）。
 * 解放まで待ってから返る。戻り値は積んだまま残るダミー（破棄は呼び手）。
 */
const squeezeHeadroom = async (gpu: GpuContext, tight: boolean): Promise<GPUBuffer[]> => {
  const size = Math.min(HEADROOM_DUMMY_BYTES, gpu.limits.maxBufferSize);
  const dummies: GPUBuffer[] = [];
  try {
    const large = await fillUntilOom(gpu, size, dummies);
    if (large === 0) throw new Error("前提: ダミーを 1 本も積めなかった");
    const small = tight ? await fillUntilOom(gpu, HEADROOM_TIGHT_DUMMY_BYTES, dummies) : 0;
    if (!tight) dummies.pop()?.destroy();
    await gpu.device.queue.onSubmittedWorkDone();
    const each = `${(size / 2 ** 30).toFixed(2)} GiB`;
    console.log(
      tight
        ? `[e2e] headroom: ${each} × ${large} 本 + 256 MiB × ${small} 本で空きを 256 MiB 未満まで詰めた`
        : `[e2e] headroom: ${each} × ${large - 1} 本を残して 1 本ぶんの空きを作った`,
    );
    return dummies;
  } catch (error) {
    for (const dummy of dummies) dummy.destroy();
    throw error;
  }
};

test.skipIf(!RESIDENCY_RUNNABLE)(
  "e2e(実GPU): residency headroom — 常駐 DiT の上に空きが足りなければ text 段の前に evicted/headroom" +
    " で退避し、2 回目も双子と一致する",
  async () => {
    const { quant } = REFERENCE[1];
    const gpu = await acquireGpu();
    let dummies: GPUBuffer[] = [];
    try {
      await withTurbo(quant, { gpu, residency: "transformer" }, async (pipeline) => {
        assertEquals(
          await assertResidencyTwin("residency-headroom-first", pipeline, 512, undefined),
          expectedStages(["retained/request"]),
          "1 回目のイベント列",
        );
        dummies = await squeezeHeadroom(gpu, false);
        // 2 回目は text 段の前に退避し（格下げ）、DiT 段は段ごと運転で作り直して手放す。PNG は 1 回目と
        // 同じ双子の行と一致する（= 1 回目とビット同一）。
        assertEquals(
          await assertResidencyTwin("residency-headroom-second", pipeline, 512, undefined),
          expectedStages(["released/downgraded"], { beforeText: ["evicted/headroom"] }),
          "2 回目のイベント列",
        );
      });
    } finally {
      for (const dummy of dummies) dummy.destroy();
      gpu.destroy();
    }
  },
);

test.skipIf(!RESIDENCY_RUNNABLE)(
  "e2e(実GPU): residency headroom — 最初の generate でも DiT を席に載せた後に空きが足りなければ" +
    " VAE 段の前に evicted/headroom で退避し、双子と一致する",
  async () => {
    // 決定性の根拠（詰めた後の空き < 256 MiB < 余裕 ≤ VAE の必要量）を門の側でも縛る。
    assertEquals(HEADROOM_TIGHT_DUMMY_BYTES < HEADROOM_MARGIN_BYTES, true);
    const { quant } = REFERENCE[1];
    const gpu = await acquireGpu();
    let dummies: GPUBuffer[] = [];
    try {
      await withTurbo(quant, { gpu, residency: "transformer" }, async (pipeline) => {
        assertEquals(
          await assertResidencyTwin(
            "residency-headroom-vae",
            pipeline,
            512,
            undefined,
            async (entry) => {
              // transformer の end は DiT を席に載せた後・VAE 段の前の量りの直前（pipeline.ts ③'）。
              if (entry === "stage:transformer:end") dummies = await squeezeHeadroom(gpu, true);
            },
          ),
          expectedStages(["retained/request"], { beforeVae: ["evicted/headroom"] }),
          "イベント列",
        );
      });
    } finally {
      for (const dummy of dummies) dummy.destroy();
      gpu.destroy();
    }
  },
);

/** この門が持つケース ID 全部（参照値が無いものを登録時に 1 度だけ知らせる）。 */
const CASE_IDS: readonly string[] = [
  ...REFERENCE.map(({ quant, resolution }) => caseIdOf(quant, resolution)),
  TURBO_DPMPP_CASE,
  "base-cfg",
  "base-cfg-bf16",
  "base-cfg-fromAssets-shards",
  "base-cfg-dpmpp",
  "fromPretrained-512",
  "onEvent-1024",
  // extra のミラーが無い機では節ごと SKIP する — 参照値の欠けとして数えない。
  ...(EXTRA_RUNNABLE ? EXTRA_MODELS.map(extraCaseIdOf) : []),
];
if (RUNNABLE) references.warnMissing(CASE_IDS);

// 「この環境の参照値がまだ無い」を無音の緑にしないための門番（ADR 0005 と同じ流儀）。
// 数えるのは上の現役ケースだけ（廃止済みケースの行が残っていても緑にはしない）。
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
