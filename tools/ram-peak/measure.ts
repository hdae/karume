/**
 * ロード時のホスト RAM ピーク計測（ADR 0108 段階分解表 段 2 の検収②③ — 時点実測の道具）。
 *
 *     # 手元の配布形を直に読む（取得層を通らない）
 *     deno run -A tools/ram-peak/measure.ts --family anima --source models/karume-anima \
 *         --model anima-turbo-v1.1 --quant f16 --steps 2 --size 512
 *     # 疑似 HF + ディレクトリ固定キャッシュで cold / warm を測り分ける
 *     deno run -A tools/ram-peak/measure.ts --family gemma4 --source models/karume-gemma4 \
 *         --state cold --cache-dir outputs/ram-peak/cache
 *     deno run -A tools/ram-peak/measure.ts --family gemma4 --source models/karume-gemma4 \
 *         --state warm --cache-dir outputs/ram-peak/cache
 *     # 部品面（容器 1 本を直に開いて Session にする・1 部品だけ）
 *     deno run -A tools/ram-peak/measure.ts --mode component --source models/karume-anima \
 *         --model anima-turbo-v1.1 --quant f16 --component transformer
 *
 * ## 何を分けて測るか
 *
 * - **取得元 3 通り**（`--state`）: `local` = 手元のディレクトリ（`denoDirectory` — network も
 *   CacheStorage も通らない）/ `cold` = 疑似 HF（`serveLocalDist`）+ **空の**キャッシュ /
 *   `warm` = 同じ疑似 HF + **直前の cold が温めた**キャッシュ。キャッシュの置き場は
 *   `--cache-dir`（`dir-cache.ts` — Deno の組み込み `caches` は置き場を選べないので自前）。
 * - **区間 2 つ**: `load` = `fromPretrained` の決着まで / `run` = 最小の生成 1 回。
 *   `rss` と `external`（ArrayBuffer の実勢）の最大値を区間ごとに出す。
 * - **digest の計数**: `crypto.subtle.digest` を包んで回数と総バイト数を数える（検収③）。
 *   descriptor 2 文書の突合は cold / warm どちらでも掛かるので、宣言された文書長で仕分けて
 *   **payload 側**（資産の全量検証・block ごとの digest）を別に出す。計数の射程は
 *   `probes.ts` の {@link countDigests} の NOTE を読むこと。
 * - **取得の内訳**: part 0 / part 1（const）/ 重み part / 資産 part を**宣言だけ**で出す
 *   （`breakdown.ts` — container-v1 §11）。
 *
 * MUST: 1 構成 = 1 プロセス。ピークはプロセス終端で読む（Linux は `/proc/self/status` の
 * VmHWM = 高水位標・Mac は無いので標本化の最大値のみ）。複数構成を回すのは `matrix.ts`。
 * 出力は JSON 1 行（研究記録の表はこれを集計する）。
 *
 * NOTE: `持越し scale`（piece 分割で part を跨いで持つ companion scale のバイト数）の欄は
 * **runtime に無い**。`packages/runtime/src/runtime/session-build.ts` の `carriedScales` は
 * 構築中だけ生きる内部表で、`SessionDiagnostics` / `StorageDiagnostics` / `SessionBuildStats` の
 * どこにも出ていない。出力では {@link MeasureReport.missingFromRuntime} に「無い」と書く
 * （runtime は触らない — この道具の契約）。
 */

import { denoDirectory } from "../../packages/hub/deno.ts";
import {
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  type LoadManifestOptions,
  openContainerSource,
  resolveSelection,
} from "../../packages/hub/mod.ts";
import {
  acquireGpu,
  createSessionFromContainer,
  openContainer,
  type SessionDiagnostics,
} from "../../packages/runtime/mod.ts";
import { type LocalDistServer, serveLocalDist } from "../../examples/shared/local-dist-server.ts";
import { type FetchBreakdown, fetchBreakdown } from "./breakdown.ts";
import { directoryCaches, type DirectoryCacheStats, resetCacheDirectory } from "./dir-cache.ts";
import { FAMILIES, type FamilyName, isFamilyName, loadFamily } from "./families.ts";
import {
  countDigests,
  type DigestSplit,
  mergeTallies,
  type PhasePeak,
  sampleMemory,
  splitDigest,
} from "./probes.ts";

/** 取得元の 3 通り（検収②の「cold / warm / ローカル」）。 */
export type MeasureState = "cold" | "warm" | "local";

/** 計測の面。`pipeline` = 家族の `fromPretrained` + 最小の生成 / `component` = 容器 1 本。 */
export type MeasureMode = "pipeline" | "component";

export type MeasureOptions = {
  readonly mode: MeasureMode;
  readonly state: MeasureState;
  readonly family: FamilyName;
  /** 配布形ディレクトリ（`karume.json` を持つ）。cold / warm ではここを疑似 HF が配る。 */
  readonly source: string;
  /** `cold` / `warm` で使うキャッシュ置き場（消して安全なディレクトリ）。 */
  readonly cacheDir?: string;
  readonly model?: string;
  readonly quant?: string;
  /** `--mode component` で開く部品名。 */
  readonly component: string;
  /** 取得の内訳を出す部品の部分集合（未指定なら manifest の全部品）。 */
  readonly weights?: readonly string[];
  readonly steps: number;
  readonly size: number;
  readonly maxNewTokens: number;
  /** Session を組む直前に明示 GC（`--v8-flags=--expose-gc` が前提・無ければ no-op）。 */
  readonly explicitGc: boolean;
};

/** ピーク 1 区間ぶんの出力形（MiB）。 */
export type PeakReport = {
  readonly rssMaxMiB: number;
  readonly externalMaxMiB: number;
  readonly samples: number;
};

/** 構築相の費用内訳（`SessionBuildStats` の写し）と低精度格納の内訳（`StorageDiagnostics`）。 */
export type ComponentReport = {
  readonly shardCount: number;
  readonly uploadedMiB: number;
  readonly shardWaitMs: number;
  readonly decodeMs: number;
  readonly bufferCreateMs: number;
  readonly writeBufferIssueMs: number;
  readonly uploadFenceMs: number;
  readonly residentCompressedMiB: number;
  /** 展開 scratch の実勢（`StorageDiagnostics.hostExpandedBytes`）。 */
  readonly hostExpandedMiB: number;
};

export type MeasureReport = {
  readonly schema: "karume/ram-peak/2";
  readonly at: string;
  readonly os: string;
  readonly mode: MeasureMode;
  readonly state: MeasureState;
  readonly family: FamilyName | null;
  readonly component: string | null;
  readonly source: string;
  readonly cacheDir: string | null;
  readonly model: string | null;
  readonly quant: string | null;
  readonly steps: number | null;
  readonly size: number | null;
  readonly maxNewTokens: number | null;
  readonly explicitGc: boolean;
  readonly loadMs: number;
  readonly runMs: number;
  /** Linux の高水位標（`/proc/self/status` の VmHWM）。他 OS では null。 */
  readonly vmHwmMiB: number | null;
  readonly rssBaselineMiB: number;
  readonly externalBaselineMiB: number;
  readonly peaks: {
    readonly load: PeakReport;
    readonly run: PeakReport;
    readonly total: PeakReport;
  };
  readonly digest: {
    readonly total: DigestSplit;
    readonly load: DigestSplit;
    readonly run: DigestSplit;
  };
  /** cold / warm のキャッシュ I/O 実績（local では null）。 */
  readonly cache: DirectoryCacheStats | null;
  /** 宣言だけで閉じた取得の内訳（container-v1 §11）。 */
  readonly fetch: FetchBreakdown;
  /** 部品名 → 構築相の内訳（診断が届いた部品だけ）。 */
  readonly components: Readonly<Record<string, ComponentReport>>;
  /** runtime の診断に**席が無い**ため出せなかった欄（名前と理由）。 */
  readonly missingFromRuntime: readonly string[];
};

const MIB = 1024 * 1024;

const mib = (bytes: number): number => Math.round(bytes / MIB);

const peakReport = (peak: PhasePeak): PeakReport => ({
  rssMaxMiB: mib(peak.rssMaxBytes),
  externalMaxMiB: mib(peak.externalMaxBytes),
  samples: peak.samples,
});

/** Linux の高水位標。他 OS は席そのものが無い（標本化の最大値で読む）。 */
const vmHwm = async (): Promise<number | undefined> => {
  if (Deno.build.os !== "linux") return undefined;
  const row = (await Deno.readTextFile("/proc/self/status")).split("\n").find((line) =>
    line.startsWith("VmHWM:")
  );
  return row === undefined ? undefined : Number(row.replace(/[^0-9]/g, "")) * 1024;
};

/**
 * `持越し scale` の席が runtime に無いことの名乗り。**黙って 0 を書かない** — 0 と「席が無い」は
 * 別の事実で、混ぜると「持越しは起きていない」と読める表になる。
 */
const MISSING_FROM_RUNTIME: readonly string[] = [
  "carriedScaleBytes（piece 分割で part を跨いで持つ companion scale のバイト数）: " +
  "runtime の診断（SessionDiagnostics / StorageDiagnostics / SessionBuildStats）に席が無い。" +
  "実体は session-build.ts の carriedScales（構築中だけ生きる内部表）で、外から読む口は無い。",
];

/** 取得元 1 つぶん（疑似 HF を立てたなら畳む相手も一緒に持つ）。 */
type ResolvedOrigin = {
  /** 取得元ハンドル（`local`）か疑似 HF のリポ参照（`cold` / `warm`）— 文字列は取らない。 */
  readonly from: HubRepoRef | DistributionSource;
  readonly caches?: CacheStorage;
  /** 疑似 HF の実ポートを隠す `fetch`（{@link STABLE_HUB_URL} の NOTE）。 */
  readonly fetch?: typeof globalThis.fetch;
  readonly cacheStats?: () => DirectoryCacheStats;
  readonly server?: LocalDistServer;
};

/**
 * 疑似 HF が**名乗る**ホスト（実体は 127.0.0.1 の自動割当ポート）。
 *
 * NOTE: なぜ実 URL をそのまま使わないか — 取得層のキャッシュキーは、資産は内容キー
 * （`["hf", kind, repo, path, sha256]`）だが **`karume.json` だけは URL そのもの**である
 * （manifest は正本の根なので事前の sha256 を持てない — `packages/hub/src/sources/hf.ts` の
 * `readManifest`）。`serveLocalDist` はポートを自動割当するので、実 URL を使うと**run ごとに
 * manifest のキーが動き**、warm が manifest だけ取り直す（= 真の warm にならない）。固定の
 * 名乗りを与え、実ポートへの差し替えは `fetch` の注入口（`LoadManifestOptions.fetch`）で行う。
 */
const STABLE_HUB_URL = "http://karume-ram-peak.invalid";

/** {@link STABLE_HUB_URL} 宛ての取得だけを実ポートへ差し替える `fetch`。 */
const rewriteToServer = (serverUrl: string): typeof globalThis.fetch => {
  const origin = new URL(serverUrl).origin;
  return (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (!url.startsWith(`${STABLE_HUB_URL}/`)) return globalThis.fetch(input, init);
    const rewritten = `${origin}${url.slice(STABLE_HUB_URL.length)}`;
    // `Request` で来た場合も本体ごと組み直す（method / headers / body を落とさない）。
    return globalThis.fetch(
      typeof input === "string" || input instanceof URL ? rewritten : new Request(rewritten, input),
      init,
    );
  };
};

/**
 * `--state` を取得元に落とす。**計測区間の外**で呼ぶ（cold のキャッシュ消去と疑似 HF の起動は
 * 測る対象ではない）。
 */
const resolveOrigin = async (options: MeasureOptions): Promise<ResolvedOrigin> => {
  if (options.state === "local") return { from: denoDirectory(options.source) };
  const { cacheDir } = options;
  if (cacheDir === undefined) {
    throw new Error(`--state ${options.state} には --cache-dir <ディレクトリ> が要る`);
  }
  // cold は毎回**空のキャッシュ**から始める。warm は直前の cold が温めた同じ置き場を使う。
  if (options.state === "cold") await resetCacheDirectory(cacheDir);
  else await Deno.mkdir(cacheDir, { recursive: true });
  const caches = directoryCaches(cacheDir);
  const server = serveLocalDist(options.source);
  const hubUrl = server.source.hubUrl;
  if (hubUrl === undefined) throw new Error("ram-peak: 疑似 HF が hubUrl を名乗っていない");
  return {
    from: { ...server.source, hubUrl: STABLE_HUB_URL },
    caches,
    fetch: rewriteToServer(hubUrl),
    cacheStats: caches.stats,
    server,
  };
};

/**
 * `--mode component`: manifest から (model, component, quant) の容器 1 本を開いて Session にする。
 *
 * MUST: 選択の外れは既知一覧つきで落とす。model / quant は `resolveSelection` が
 * `ManifestReferenceError` に一覧を添えるので、ここで綴り直さない（同じ判定を 2 実装持たない）。
 */
const runComponent = async (
  options: MeasureOptions,
  origin: ResolvedOrigin,
  hubOptions: LoadManifestOptions,
  onDiagnostics: (component: string, diagnostics: SessionDiagnostics) => void,
): Promise<void> => {
  const loaded = await loadManifest(origin.from, hubOptions);
  const selected = resolveSelection(loaded.manifest, {
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.quant === undefined ? {} : { quant: options.quant }),
  });
  const container = selected.containers[options.component];
  if (container === undefined) {
    throw new Error(
      `${selected.model} / ${selected.quant} に component '${options.component}' が無い` +
        `（既知: ${Object.keys(selected.containers).join(" / ")}）`,
    );
  }
  const opened = await openContainer(
    { kind: "source", source: openContainerSource(loaded, container, hubOptions) },
    container.descriptor,
  );
  const gpu = await acquireGpu();
  try {
    if (options.explicitGc) (globalThis as { gc?: () => void }).gc?.();
    // グラフ名 = 部品名（= manifest の weights キー — container-v1 §12）。
    const session = await createSessionFromContainer(gpu, opened, options.component);
    onDiagnostics(options.component, session.diagnostics());
    await session.dispose();
  } finally {
    gpu.destroy();
  }
};

/**
 * 1 構成を測る。
 *
 * 順序が契約である: ①取得元を用意する（計測の外）②観測器を仕掛ける ③`load` ④`run` ⑤観測器を
 * 外す ⑥取得の内訳を読む（`openContainer` が digest を掛けるので、必ず観測器を外した**後**）。
 */
export const measure = async (options: MeasureOptions): Promise<MeasureReport> => {
  const origin = await resolveOrigin(options);
  try {
    return await measureFrom(options, origin);
  } finally {
    // MUST: 途中でどこが落ちても疑似 HF は畳む（畳まないとリスナーが残り、呼び手が
    // プロセスでない場合 — テスト — に op が漏れる）。
    await origin.server?.close();
  }
};

/** {@link measure} の本体（取得元は用意済み・畳むのは呼び手）。 */
const measureFrom = async (
  options: MeasureOptions,
  origin: ResolvedOrigin,
): Promise<MeasureReport> => {
  const hubOptions: LoadManifestOptions = {
    ...(origin.caches === undefined ? {} : { caches: origin.caches }),
    ...(origin.fetch === undefined ? {} : { fetch: origin.fetch }),
  };
  const diagnostics = new Map<string, SessionDiagnostics>();
  const onDiagnostics = (component: string, seen: SessionDiagnostics): void => {
    // 同じ部品が複数 run で届く家族（gemma4 の prefill / decode）は**最初の 1 本**を採る —
    // 構築相の内訳は Session の寿命を通じて不変なので、後の run で上書きする意味が無い。
    if (!diagnostics.has(component)) diagnostics.set(component, seen);
  };

  const sampler = sampleMemory();
  const digests = countDigests();
  let loadMs = 0;
  let runMs = 0;
  try {
    const started = performance.now();
    if (options.mode === "component") {
      await runComponent(options, origin, hubOptions, onDiagnostics);
      loadMs = performance.now() - started;
    } else {
      const loaded = await loadFamily(options.family, origin.from, {
        ...(options.model === undefined ? {} : { model: options.model }),
        ...(options.quant === undefined ? {} : { quant: options.quant }),
        ...(origin.caches === undefined ? {} : { caches: origin.caches }),
        ...(origin.fetch === undefined ? {} : { fetch: origin.fetch }),
        onDiagnostics,
        knobs: {
          steps: options.steps,
          size: options.size,
          maxNewTokens: options.maxNewTokens,
        },
      });
      loadMs = performance.now() - started;
      sampler.phase("run");
      digests.phase("run");
      const runStarted = performance.now();
      try {
        await loaded.run();
        runMs = performance.now() - runStarted;
      } finally {
        await loaded.dispose();
      }
    }
  } finally {
    sampler.stop();
    digests.stop();
  }

  const peak = await vmHwm();
  // 取得の内訳は**観測器を外した後**に読む（`openContainer` の 2 文書突合が digest 2 回ぶん
  // 掛かるため — 計測中に呼ぶと検収③の計数へ harness 自身の digest が混ざる）。
  const loadedForBreakdown = await loadManifest(origin.from, hubOptions);
  const breakdown: FetchBreakdown = await fetchBreakdown(
    loadedForBreakdown,
    resolveSelection(loadedForBreakdown.manifest, {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
      ...(options.weights === undefined ? {} : { weights: options.weights }),
    }),
    hubOptions,
  );

  const descriptorLengths = new Set(breakdown.descriptorLengths);
  const loadTally = digests.tally("load");
  const runTally = digests.tally("run");
  const components: Record<string, ComponentReport> = {};
  for (const [name, seen] of diagnostics) {
    const stats = seen.buildStats;
    components[name] = {
      shardCount: stats.shardCount,
      uploadedMiB: mib(stats.uploadedBytes),
      shardWaitMs: Math.round(stats.shardWaitMs),
      decodeMs: Math.round(stats.decodeMs),
      bufferCreateMs: Math.round(stats.bufferCreateMs),
      writeBufferIssueMs: Math.round(stats.writeBufferIssueMs),
      uploadFenceMs: Math.round(stats.uploadFenceMs),
      residentCompressedMiB: mib(seen.storage.residentCompressedBytes),
      hostExpandedMiB: mib(seen.storage.hostExpandedBytes),
    };
  }

  const isPipeline = options.mode === "pipeline";
  return {
    schema: "karume/ram-peak/2",
    at: new Date().toISOString(),
    os: Deno.build.os,
    mode: options.mode,
    state: options.state,
    family: isPipeline ? options.family : null,
    component: isPipeline ? null : options.component,
    source: options.source,
    cacheDir: options.cacheDir ?? null,
    model: breakdown.model,
    quant: breakdown.quant,
    // 与えていないノブは null を書く（「既定で走った」と「渡した」を出力で区別する）。
    steps: isPipeline && options.family === "anima" ? options.steps : null,
    size: isPipeline && options.family === "anima" ? options.size : null,
    maxNewTokens: isPipeline && options.family === "gemma4" ? options.maxNewTokens : null,
    explicitGc: options.explicitGc,
    loadMs: Math.round(loadMs),
    runMs: Math.round(runMs),
    vmHwmMiB: peak === undefined ? null : mib(peak),
    rssBaselineMiB: mib(sampler.baseline.rssMaxBytes),
    externalBaselineMiB: mib(sampler.baseline.externalMaxBytes),
    peaks: {
      load: peakReport(sampler.peak("load")),
      run: peakReport(sampler.peak("run")),
      total: peakReport(sampler.total()),
    },
    digest: {
      total: splitDigest(mergeTallies([loadTally, runTally]), descriptorLengths),
      load: splitDigest(loadTally, descriptorLengths),
      run: splitDigest(runTally, descriptorLengths),
    },
    cache: origin.cacheStats?.() ?? null,
    fetch: breakdown,
    components,
    missingFromRuntime: MISSING_FROM_RUNTIME,
  };
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/**
 * 受けるオプション。MUST: 未知のキーは落とす — `--stpes 4` が黙って既定 2 で走ると、
 * 研究記録の 1 行が「何を測ったか」を偽る（出力に残る条件と実際の条件が食い違う）。
 */
const KNOWN: ReadonlySet<string> = new Set([
  "mode",
  "state",
  "family",
  "source",
  "cache-dir",
  "model",
  "quant",
  "component",
  "weights",
  "steps",
  "size",
  "max-new-tokens",
  "gc",
]);

const MODES: readonly MeasureMode[] = ["pipeline", "component"];
const STATES: readonly MeasureState[] = ["cold", "warm", "local"];

/** `--key value` の対だけを受ける（次のフラグを値として食わない）。 */
export const parseArgs = (argv: readonly string[]): Map<string, string> => {
  const args = new Map<string, string>();
  for (let at = 0; at < argv.length; at += 2) {
    const [key, value] = [argv[at], argv[at + 1]];
    if (!key.startsWith("--") || value === undefined) throw new Error(`引数 ${key} が対でない`);
    const name = key.slice(2);
    if (!KNOWN.has(name)) {
      throw new Error(
        `未知のオプション ${key}（既知: ${[...KNOWN].map((known) => `--${known}`).join(" ")}）`,
      );
    }
    args.set(name, value);
  }
  return args;
};

const positiveInteger = (raw: string | undefined, fallback: number, name: string): number => {
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} は 1 以上の整数でなければならない（受け取った値: ${raw}）`);
  }
  return value;
};

/** 引数 → 計測の条件。受理集合の外は既知一覧つきで落とす。 */
export const toOptions = (args: ReadonlyMap<string, string>): MeasureOptions => {
  const mode = args.get("mode") ?? "pipeline";
  if (!(MODES as readonly string[]).includes(mode)) {
    throw new Error(`--mode ${mode} は未対応（${MODES.join(" | ")}）`);
  }
  const state = args.get("state") ?? "local";
  if (!(STATES as readonly string[]).includes(state)) {
    throw new Error(`--state ${state} は未対応（${STATES.join(" | ")}）`);
  }
  const family = args.get("family") ?? "anima";
  if (!isFamilyName(family)) {
    throw new Error(`--family ${family} は未対応（${FAMILIES.join(" | ")}）`);
  }
  const source = args.get("source");
  if (source === undefined) throw new Error("--source <配布形ディレクトリ> は必須");
  const weights = args.get("weights");
  return {
    mode: mode as MeasureMode,
    state: state as MeasureState,
    family,
    source,
    ...(args.get("cache-dir") === undefined ? {} : { cacheDir: args.get("cache-dir") as string }),
    ...(args.get("model") === undefined ? {} : { model: args.get("model") as string }),
    ...(args.get("quant") === undefined ? {} : { quant: args.get("quant") as string }),
    component: args.get("component") ?? "transformer",
    ...(weights === undefined ? {} : { weights: weights.split(",") }),
    steps: positiveInteger(args.get("steps"), 2, "--steps"),
    size: positiveInteger(args.get("size"), 512, "--size"),
    maxNewTokens: positiveInteger(args.get("max-new-tokens"), 8, "--max-new-tokens"),
    explicitGc: args.get("gc") === "true",
  };
};

if (import.meta.main) {
  console.log(JSON.stringify(await measure(toOptions(parseArgs(Deno.args)))));
}
