/**
 * HF リポジトリからの取得（ADR 0038 §5「hub の取得層」）。土台は `@hdae/fetch-cache`
 * （実行時依存ゼロ・Web 標準 API のみ）。
 *
 * 契約:
 * - revision は**セッションあたり 1 回だけ**解決し、manifest も全ファイルも同一 commit SHA に
 *   固定して取得する（可変 ref のまま複数回解決すると manifest と重みが別コミットから来る）。
 * - キャッシュ名前空間は必ず明示（`karume/1`）。`Authorization` を伴う取得は
 *   `karume/1:auth:<credential の sha256 先頭 16 hex>` へ隔離する（gated 資産を無認証経路の
 *   ヒットに供さず、かつ別 credential の写しにも供さない）。
 * - 同時取得は 4 本まで。`AbortSignal` は全取得へ透過。
 * - 進捗総量は content-length ではなく manifest の `size` 合計（path 一意化後）。
 */

import {
  fetchHfFile,
  hfResolveUrl,
  isCommitSha,
  prefetchHfFile,
  resolveHfRevision,
} from "@hdae/fetch-cache/hf";
import {
  HubError,
  HubFetchError,
  IntegrityError,
  ManifestFormatError,
  ManifestReferenceError,
} from "./errors.ts";
import {
  type FileRef,
  type Manifest,
  MANIFEST_FILENAME,
  MAX_MANIFEST_BYTES,
  parseManifest,
} from "./manifest.ts";
import type { ResolvedFiles } from "./resolve.ts";
import { type ByteBudget, createGuardedFetch } from "./transport.ts";

/** 専用キャッシュ名前空間（ライブラリ既定名は他コードと共有されるため使わない）。 */
const CACHE_NAMESPACE = "karume/1";
/** `Authorization` 付き取得の隔離先の前置き（この後ろに credential 由来の suffix が付く）。 */
const AUTH_CACHE_PREFIX = `${CACHE_NAMESPACE}:auth:`;
/** 同時取得数（数十コンポーネントの manifest で接続と RAM を破綻させない）。 */
const CONCURRENCY = 4;

/** 取得対象リポジトリ。`hubUrl` は**アプリが明示指定した場合のみ**有効（manifest からは来ない）。 */
export type HubRepoRef = {
  /** `"owner/name"`。 */
  readonly repo: string;
  /** ブランチ / タグ / commit SHA。既定 `"main"`。SHA を渡すと解決要求が発生しない。 */
  readonly revision?: string;
  /** ミラー用。既定 `https://huggingface.co`。 */
  readonly hubUrl?: string;
};

/** cache I/O 失敗の診断（quota 超過等）。console.warn に任せず**アプリへ届ける**。 */
export type CacheDiagnostic = {
  readonly op: "open" | "match" | "put" | "delete";
  readonly url: string;
  readonly error: unknown;
};

export type LoadManifestOptions = {
  readonly signal?: AbortSignal;
  /** `Authorization` 等。付けた取得は credential ごとの認証専用キャッシュ名前空間へ隔離される。 */
  readonly headers?: HeadersInit;
  /**
   * cache I/O 失敗の通知先。無指定だと「毎起動フル再 DL が黙って常態化」するため、
   * アプリは受け取って `navigator.storage.persist()` の案内などに使う。
   */
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /** `fetch` の差し替え（テスト・カスタム輸送用）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** `CacheStorage` の差し替え（テスト用）。 */
  readonly caches?: CacheStorage;
};

/** 解決済み revision に固定された manifest。以降の取得は全てこの SHA で行う。 */
export type LoadedManifest = {
  readonly repo: string;
  /** 解決済み commit SHA（40 桁）。返り値・エラー・診断に必ず載る。 */
  readonly revisionSha: string;
  readonly hubUrl?: string;
  readonly manifest: Manifest;
};

/**
 * 進捗のフェーズ。`verifying` は sha256 照合中（3.7GB で数秒 — 無言のハングにしない）、
 * `complete` は 1 ファイルの終端（検証を通って bytes が確定した点）。
 *
 * MUST: 1 ファイルの phase は `downloading`* → `verifying` → `complete` の順にだけ進み、
 * 逆行しない（`complete` はファイルごとに 1 回だけ・以降そのファイルの通知は出ない）。
 * 例外は破損キャッシュの self-heal で、`verifying` が拒否した後に network から取り直すため
 * この 1 巡が最初からやり直しになる（`complete` が終端であることは変わらない）。
 */
export type AssetPhase = "downloading" | "verifying" | "complete";

export type AssetProgress = {
  readonly phase: AssetPhase;
  /** イベントを起こしたファイルの path。 */
  readonly path: string;
  /** 取得済みバイトの合計（全ファイル・path 一意化後）。 */
  readonly loaded: number;
  /** manifest の `size` 合計（path 一意化後）。 */
  readonly total: number;
  /**
   * `path` の**そのファイル自身**の受信済みバイト。`loaded` が全ファイルの合計なのに対し
   * こちらは 1 ファイルぶんなので、ファイル別の進捗バーはこの値と {@link fileTotal} で描く。
   *
   * `verifying` / `complete` は全量が揃った点なので常に `fileLoaded === fileTotal`
   * （`downloading` が 1 度も出ないキャッシュヒットでも同じ）。
   */
  readonly fileLoaded: number;
  /** `path` のファイル自身の manifest 由来サイズ（`FileRef.size`）。`total` はこれの合計。 */
  readonly fileTotal: number;
};

export type FetchAssetsOptions = LoadManifestOptions & {
  readonly onProgress?: (progress: AssetProgress) => void;
};

/**
 * 取得に使うキャッシュ名前空間。
 *
 * MUST: 認証付きは credential ごとに分ける — 下層のキャッシュキーは URL のみなので、名前が
 * `Authorization` の**有無**だけだと token A で埋めた写しに token B の同一 URL 要求がヒットする
 * （権限の違う 2 人が同じ端末を使う場面で gated 資産が漏れる）。名前には生の credential を出さず
 * sha256 の先頭 16 hex だけを載せる（CacheStorage の名前は列挙可能なため）。
 */
const cacheNameFor = async (headers?: HeadersInit): Promise<string> => {
  const authorization = headers === undefined
    ? undefined
    : new Headers(headers).get("authorization") ?? undefined;
  if (authorization === undefined) return CACHE_NAMESPACE;
  const digest = await sha256Hex(new TextEncoder().encode(authorization));
  return `${AUTH_CACHE_PREFIX}${digest.slice(0, 16)}`;
};

/** karume 自身の名前空間か（無認証本体と `karume/1:` 配下の認証隔離すべて）。 */
const isHubCacheName = (name: string): boolean =>
  name === CACHE_NAMESPACE || name.startsWith(`${CACHE_NAMESPACE}:`);

const requestInit = (headers?: HeadersInit, signal?: AbortSignal): RequestInit => ({
  ...(headers === undefined ? {} : { headers }),
  ...(signal === undefined ? {} : { signal }),
});

const hfRef = (
  ref: HubRepoRef,
  revision: string,
): { repo: string; revision: string; hubUrl?: string } => ({
  repo: ref.repo,
  revision,
  ...(ref.hubUrl === undefined ? {} : { hubUrl: ref.hubUrl }),
});

/**
 * 中断（呼び出し側の `AbortSignal`）かどうか。中断は取得失敗ではないので `HubFetchError` に
 * 包まず素通しする — 包むと呼び出し側の「自分が止めたのか落ちたのか」の判別が壊れる。
 *
 * MUST: 既定の `abort()` が作る DOMException だけを見ない — `abort(reason)` の custom reason
 * では `fetch` が **reason 自体**で reject するため、任意の値が中断の正体になり得る。判定は
 * 「実際に取得へ渡した signal が aborted で、捕まえた値がその reason と同一」に依る（合成した
 * 場合は合成後の signal — 上流の reason はそのまま伝播する）。
 */
const isAborted = (error: unknown, signal?: AbortSignal): boolean =>
  (error instanceof DOMException && error.name === "AbortError") ||
  (signal?.aborted === true && error === signal.reason);

/**
 * バイト列を小文字 hex の sha256 にする。
 *
 * MUST: 全量コピーしない — 数 GB 級ではコピー 1 回が一時 RAM を倍増させる（8GB 機ターゲットに
 * 実害）。取得層の返す bytes は ArrayBuffer 背面なのでそのまま digest へ渡せる。万一
 * SharedArrayBuffer 背面や部分ビューが来た場合だけコピーで背面を保証する（WebCrypto は
 * SAB を拒否する）。
 */
const sha256Hex = async (bytes: Uint8Array): Promise<string> => {
  const source = isTightView(bytes) ? bytes : new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", source);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
};

/**
 * `openModel` は全量 ArrayBuffer を要求するため、返す bytes は buffer 全体を占めていなければ
 * ならない（slice で辻褄を合わせると RAM ピークが倍増する）。SharedArrayBuffer 背面は
 * ここで弾く（述語が主張する `Uint8Array<ArrayBuffer>` を型の上でも嘘にしない）。
 */
const isTightView = (bytes: Uint8Array): bytes is Uint8Array<ArrayBuffer> =>
  bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 &&
  bytes.byteLength === bytes.buffer.byteLength;

const assertTightView = (bytes: Uint8Array, path: string): Uint8Array<ArrayBuffer> => {
  if (!isTightView(bytes)) {
    throw new Error(
      `hub: ${path} の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes;
};

/**
 * 可変 ref を commit SHA へ解決する。ここが**セッション唯一の解決点**で、以降の取得は
 * 全てこの SHA に固定される。
 */
const resolveRevision = async (
  ref: HubRepoRef,
  revision: string,
  options: LoadManifestOptions,
): Promise<string> => {
  if (isCommitSha(revision)) return revision;
  try {
    return await resolveHfRevision(hfRef(ref, revision), {
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      init: requestInit(options.headers, options.signal),
    });
  } catch (error) {
    if (isAborted(error, options.signal)) throw error;
    throw new HubFetchError(
      `revision '${revision}' の解決に失敗した（repo ${ref.repo}）。可変 ref はオフラインで` +
        `起動できない — revision に commit SHA を渡すと解決要求そのものが発生しない`,
      { repo: ref.repo, cause: error },
    );
  }
};

/**
 * `karume.json` を取得して parse する。revision の解決はここで 1 回だけ行い、結果の SHA を
 * 返り値に載せる（{@link fetchAssets} はその SHA で取得する）。
 */
export const loadManifest = async (
  ref: HubRepoRef,
  options: LoadManifestOptions = {},
): Promise<LoadedManifest> => {
  const revisionSha = await resolveRevision(ref, ref.revision ?? "main", options);
  const target = hfRef(ref, revisionSha);
  const url = hfResolveUrl({ ...target, path: MANIFEST_FILENAME });
  const budget: ByteBudget = {
    maxBytes: MAX_MANIFEST_BYTES,
    exact: false,
    violation: (actual, where) =>
      new ManifestFormatError(
        `manifest: ${MANIFEST_FILENAME} が上限 ${MAX_MANIFEST_BYTES} バイトを超えた` +
          `（${where} = ${actual} — repo ${ref.repo} @ ${revisionSha}）`,
      ),
  };
  // 名前空間の解決は headers が固定である入口で 1 回だけ（digest は非同期）。
  const cacheName = await cacheNameFor(options.headers);
  let bytes: Uint8Array;
  try {
    bytes = await fetchHfFile(target, { path: MANIFEST_FILENAME }, {
      cacheName,
      init: requestInit(options.headers, options.signal),
      fetch: createGuardedFetch(options.fetch ?? globalThis.fetch, new Map([[url, budget]])),
      ...(options.caches === undefined ? {} : { caches: options.caches }),
      ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
    });
  } catch (error) {
    if (error instanceof HubError || isAborted(error, options.signal)) throw error;
    throw new HubFetchError(
      `${MANIFEST_FILENAME} の取得に失敗した（repo ${ref.repo} @ ${revisionSha}）`,
      { repo: ref.repo, revisionSha, path: MANIFEST_FILENAME, cause: error },
    );
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new ManifestFormatError(`manifest: ${MANIFEST_FILENAME} が UTF-8 として読めない`, {
      cause: error,
    });
  }
  return {
    repo: ref.repo,
    revisionSha,
    ...(ref.hubUrl === undefined ? {} : { hubUrl: ref.hubUrl }),
    manifest: parseManifest(text),
  };
};

/**
 * 進捗の `fileTotal` に載せる manifest 由来の size を引く。表は取得対象そのものから作るので
 * 進捗に出る path は必ず引ける — 引けないのは内部の不変条件が破れているときだけなので落とす
 * （0 で埋めるとファイル別の進捗バーが黙って壊れた値を描く）。
 */
const fileSizeOf = (refs: ReadonlyMap<string, FileRef>, path: string): number => {
  const ref = refs.get(path);
  if (ref === undefined) {
    throw new Error(`hub: ${path} の size が取得対象の表に無い（進捗集計の不変条件破れ）`);
  }
  return ref.size;
};

/**
 * 解決済みファイル表を取得する。取得と進捗総量は **path で一意化**され、同じ path を指す
 * 複数のキーには同一のバイト列が入る。
 *
 * MUST: 検証（size / sha256）はキャッシュヒット側にも走る（取得層の `validate` フック経由 —
 * 破損キャッシュは self-heal で 1 往復だけ取り直す）。
 */
export const fetchAssets = async (
  loaded: LoadedManifest,
  files: ResolvedFiles,
  options: FetchAssetsOptions = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const { repo, revisionSha, manifest } = loaded;
  const available = manifest.available;
  const target = hfRef(loaded, revisionSha);

  const keys = Object.keys(files);
  const unique = new Map<string, FileRef>();
  for (const key of keys) {
    const ref = files[key];
    if (!unique.has(ref.path)) unique.set(ref.path, ref);
  }
  const targets = [...unique.values()];
  let total = 0;
  for (const ref of targets) total += ref.size;

  const received = new Map<string, number>();
  const fromNetwork = new Set<string>();
  const emit = (phase: AssetPhase, path: string): void => {
    if (options.onProgress === undefined) return;
    const fileTotal = fileSizeOf(unique, path);
    let sum = 0;
    for (const bytes of received.values()) sum += bytes;
    // verifying / complete は全量が揃った点なので size をそのまま渡す（キャッシュヒットは
    // downloading が 1 度も出ず `received` に載らないため、受信実績から引くと 0 に見える）。
    const fileLoaded = phase === "downloading" ? received.get(path) ?? 0 : fileTotal;
    options.onProgress({ phase, path, loaded: sum, total, fileLoaded, fileTotal });
  };

  const budgets = new Map<string, ByteBudget>();
  for (const ref of targets) {
    budgets.set(hfResolveUrl({ ...target, path: ref.path }), {
      maxBytes: ref.size,
      exact: true,
      onRequest: () => fromNetwork.add(ref.path),
      violation: (actual, where) =>
        new IntegrityError(
          `${ref.path}: ${where} が manifest の size と食い違う（期待 ${ref.size} / 実際 ${actual}）`,
          {
            repo,
            revisionSha,
            path: ref.path,
            expected: String(ref.size),
            actual: String(actual),
            source: "network",
            available,
          },
        ),
    });
  }

  const failure = new AbortController();
  const signal = options.signal === undefined
    ? failure.signal
    : AbortSignal.any([failure.signal, options.signal]);
  const guarded = createGuardedFetch(options.fetch ?? globalThis.fetch, budgets);
  // 名前空間の解決は headers が固定である入口で 1 回だけ（digest は非同期）。
  const cacheName = await cacheNameFor(options.headers);
  const bytesByPath = new Map<string, Uint8Array<ArrayBuffer>>();

  const fetchOne = async (ref: FileRef): Promise<void> => {
    const validate = async (bytes: Uint8Array): Promise<void> => {
      const source = fromNetwork.has(ref.path) ? "network" : "cache";
      if (bytes.byteLength !== ref.size) {
        throw new IntegrityError(
          `${ref.path}: バイト数が manifest と食い違う（期待 ${ref.size} / 実際 ${bytes.byteLength}）`,
          {
            repo,
            revisionSha,
            path: ref.path,
            expected: String(ref.size),
            actual: String(bytes.byteLength),
            source,
            available,
          },
        );
      }
      emit("verifying", ref.path);
      const actual = await sha256Hex(bytes);
      if (actual !== ref.sha256) {
        throw new IntegrityError(
          `${ref.path}: sha256 が manifest と食い違う（期待 ${ref.sha256} / 実際 ${actual}）`,
          {
            repo,
            revisionSha,
            path: ref.path,
            expected: ref.sha256,
            actual,
            source,
            available,
          },
        );
      }
    };
    let bytes: Uint8Array;
    try {
      bytes = await fetchHfFile(target, { path: ref.path, validate }, {
        cacheName,
        init: requestInit(options.headers, signal),
        fetch: guarded,
        // network 側だけ発火する（キャッシュヒットは verifying → complete の 2 点で進む）。
        // `loaded` が `size` を超えないことは受信バイトの門（transport.ts）が保証する。
        onProgress: (progress) => {
          received.set(ref.path, progress.loaded);
          emit("downloading", ref.path);
        },
        ...(options.caches === undefined ? {} : { caches: options.caches }),
        ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, signal)) throw error;
      throw new HubFetchError(`${ref.path} の取得に失敗した（repo ${repo} @ ${revisionSha}）`, {
        repo,
        revisionSha,
        path: ref.path,
        available,
        cause: error,
      });
    }
    bytesByPath.set(ref.path, assertTightView(bytes, ref.path));
    received.set(ref.path, ref.size);
    emit("complete", ref.path);
  };

  let next = 0;
  const worker = async (): Promise<void> => {
    try {
      while (next < targets.length) await fetchOne(targets[next++]);
    } catch (error) {
      // 1 本でも落ちたら残りを止める（fail loud — 全体が reject するのに DL を続ける意味はない）。
      failure.abort(error);
      throw error;
    }
  };
  // MUST: 失敗しても全ワーカーの決着を待ってから抜ける（`Promise.all` の早期 reject だと
  // 呼び出し側が catch した後も取得が背後で走り続け、abort の意味が無くなる）。失敗時は
  // 全員が同じ理由（`failure.abort(error)` の reason = その error）で落ちる。
  const settled = await Promise.allSettled(
    Array.from({ length: Math.min(CONCURRENCY, targets.length) }, () => worker()),
  );
  for (const result of settled) {
    if (result.status === "rejected") throw result.reason;
  }

  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of keys) {
    const bytes = bytesByPath.get(files[key].path);
    if (bytes === undefined) {
      throw new Error(`hub: ${files[key].path} の bytes が揃っていない（取得層の不変条件破れ）`);
    }
    assets = { ...assets, [key]: bytes };
  }
  return assets;
};

/** {@link streamAssets} が 1 本ずつ引き渡す shard。 */
export type StreamedAsset = {
  /** manifest が宣言していた path（`refs` に渡した順で届く）。 */
  readonly path: string;
  /** 検証（size / sha256）を通ったバイト列。buffer 全体を占める。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
};

/** {@link streamAssets} のオプション（全量面 {@link fetchAssets} と同じ透過）。 */
export type StreamAssetsOptions = FetchAssetsOptions;

/**
 * shard を **2 相**で読み、1 本ずつ引き渡す逐次面（ADR 0070 決定 2）。
 *
 * - **相 1（prefetch）**: 最初の yield の前に、全 shard を永続キャッシュへ落とす
 *   （streaming — RAM に全量を載せない・同時 4 本）。`sha256` は通過中に照合され、
 *   不一致はエントリ不成立で fail loud（帯域を捨てた後に全量を握って落ちない）。
 * - **相 2（逐次引き渡し）**: `refs` の順に 1 本ずつ「キャッシュから取得 → size / sha256 照合
 *   （**キャッシュヒット側でも走る** — ADR 0070 の非交渉条件。破損は self-heal で 1 往復だけ
 *   取り直す）→ 呼び手へ渡す → 参照を手放す」。
 *
 * 全量面 {@link fetchAssets} との違いは**渡したバイト列への参照を残さない**ことで、RAM ピークが
 * O(最大 shard) に収まる（全量ホスト保持が成立しない検収モデル級のための面）。全量面は温存して
 * あるので、小モデルは従来どおりそちらを使う。
 *
 * MUST: 相 1 は**最初の `next()` まで開始されない**（async generator の遅延）。呼んだだけでは
 * 何も起きず、`for await` に入るか `next()` を呼んだ時点で DL が始まる。空の `refs`・重複 path も
 * その時点（network に出る前）に {@link ManifestReferenceError} で弾く。
 *
 * NOTE: 相 1 は `caches` が無い環境・キャッシュ書込み失敗（quota 超過等）で **fail loud** になる
 * （下層 prefetch の契約 — バイト列を手元に持たない面なので素 fetch へ縮退する余地が無い。黙って
 * 縮退させると RAM ピークの目標が壊れる）。`onCacheError` の診断が届くのは相 2 だけで、相 1 の
 * cache I/O 失敗は {@link HubFetchError} として上がる。
 */
export const streamAssets = async function* (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: StreamAssetsOptions = {},
): AsyncGenerator<StreamedAsset, void, unknown> {
  const { repo, revisionSha, manifest } = loaded;
  const available = manifest.available;
  const target = hfRef(loaded, revisionSha);

  // 入力検査は相 1 の前に済ませる（相 1 は全 shard を落としてしまうので、network に出た後で
  // 呼び出し側の誤りに気づいても帯域が戻らない）。
  if (refs.length === 0) {
    throw new ManifestReferenceError(
      `streamAssets: 取得対象が 1 つも無い（repo ${repo} @ ${revisionSha}）`,
      { available },
    );
  }
  // 重複検査の表はそのまま進捗の per-file 引き当て（`fileTotal`）にも使う。
  const declared = new Map<string, FileRef>();
  for (const ref of refs) {
    if (declared.has(ref.path)) {
      throw new ManifestReferenceError(
        `streamAssets: path '${ref.path}' が重複している（逐次面は渡された列をそのまま引き渡す —` +
          ` 同じ shard を 2 回受け取るのは呼び出し側の誤り。全量面 fetchAssets は path で一意化する）`,
        { available },
      );
    }
    declared.set(ref.path, ref);
  }

  let total = 0;
  for (const ref of refs) total += ref.size;

  const received = new Map<string, number>();
  const fromNetwork = new Set<string>();
  const emit = (phase: AssetPhase, path: string): void => {
    if (options.onProgress === undefined) return;
    const fileTotal = fileSizeOf(declared, path);
    let sum = 0;
    for (const bytes of received.values()) sum += bytes;
    // verifying / complete は全量が揃った点なので size をそのまま渡す（相 1 が温めた分は
    // 相 2 でキャッシュヒットになり downloading が出ないため、受信実績から引くと 0 に見える）。
    const fileLoaded = phase === "downloading" ? received.get(path) ?? 0 : fileTotal;
    options.onProgress({ phase, path, loaded: sum, total, fileLoaded, fileTotal });
  };

  const budgets = new Map<string, ByteBudget>();
  for (const ref of refs) {
    budgets.set(hfResolveUrl({ ...target, path: ref.path }), {
      maxBytes: ref.size,
      exact: true,
      onRequest: () => fromNetwork.add(ref.path),
      violation: (actual, where) =>
        new IntegrityError(
          `${ref.path}: ${where} が manifest の size と食い違う（期待 ${ref.size} / 実際 ${actual}）`,
          {
            repo,
            revisionSha,
            path: ref.path,
            expected: String(ref.size),
            actual: String(actual),
            source: "network",
            available,
          },
        ),
    });
  }

  const failure = new AbortController();
  const prefetchSignal = options.signal === undefined
    ? failure.signal
    : AbortSignal.any([failure.signal, options.signal]);
  const guarded = createGuardedFetch(options.fetch ?? globalThis.fetch, budgets);
  // 名前空間の解決は headers が固定である入口で 1 回だけ（digest は非同期）。
  const cacheName = await cacheNameFor(options.headers);

  // ---- 相 1: 全 shard を永続キャッシュへ落とす（ここを抜けるまで 1 本も yield しない）。
  const prefetchOne = async (ref: FileRef): Promise<void> => {
    try {
      await prefetchHfFile(target, { path: ref.path, sha256: ref.sha256 }, {
        cacheName,
        init: requestInit(options.headers, prefetchSignal),
        fetch: guarded,
        onProgress: (progress) => {
          received.set(ref.path, progress.loaded);
          emit("downloading", ref.path);
        },
        ...(options.caches === undefined ? {} : { caches: options.caches }),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, prefetchSignal)) throw error;
      // MUST: 捕まえた値の同一性だけで中断を判定しない — prefetch はバイト列を手元に持たない
      // 面なので、転送中断も put の reject として現れ、`cause` に沈めて包まれる。signal が
      // 落ちていればその reason を素通しする（巻き添えなら reason は最初の失敗そのもので、
      // 全量面と同じく全ワーカーが同一の error に収束する）。
      if (prefetchSignal.aborted) throw prefetchSignal.reason;
      throw new HubFetchError(
        `${ref.path} の事前取得に失敗した（repo ${repo} @ ${revisionSha}）`,
        { repo, revisionSha, path: ref.path, available, cause: error },
      );
    }
  };

  let next = 0;
  const worker = async (): Promise<void> => {
    try {
      while (next < refs.length) await prefetchOne(refs[next++]);
    } catch (error) {
      // 1 本でも落ちたら残りを止める（fail loud — 全体が reject するのに DL を続ける意味はない）。
      failure.abort(error);
      throw error;
    }
  };
  // MUST: 失敗しても全ワーカーの決着を待ってから抜ける（全量面と同じ理由 — 早期 reject だと
  // 呼び出し側が catch した後も取得が背後で走り続ける）。
  const settled = await Promise.allSettled(
    Array.from({ length: Math.min(CONCURRENCY, refs.length) }, () => worker()),
  );
  for (const result of settled) {
    if (result.status === "rejected") throw result.reason;
  }

  // 相 1 の network 実績は相 2 の帰属へ持ち越さない。相 2 で network に出るのは破損キャッシュの
  // self-heal だけなので、持ち越すとキャッシュ読出し由来の不一致まで source "network" に化ける。
  fromNetwork.clear();

  // ---- 相 2: 1 本ずつ照合して引き渡す。
  for (const ref of refs) {
    // 相 2 は大半が「キャッシュ読出し + ハッシュ」で network に出ない＝下層の signal 監視が
    // 効かない区間なので、shard の切れ目で明示的に中断を見る（数 GB のハッシュを何本も
    // 回している最中に取り消しが効かないのは中断の透過が壊れているのと同じ）。
    options.signal?.throwIfAborted();

    const validate = async (bytes: Uint8Array): Promise<void> => {
      const source = fromNetwork.has(ref.path) ? "network" : "cache";
      if (bytes.byteLength !== ref.size) {
        throw new IntegrityError(
          `${ref.path}: バイト数が manifest と食い違う（期待 ${ref.size} / 実際 ${bytes.byteLength}）`,
          {
            repo,
            revisionSha,
            path: ref.path,
            expected: String(ref.size),
            actual: String(bytes.byteLength),
            source,
            available,
          },
        );
      }
      emit("verifying", ref.path);
      const actual = await sha256Hex(bytes);
      if (actual !== ref.sha256) {
        throw new IntegrityError(
          `${ref.path}: sha256 が manifest と食い違う（期待 ${ref.sha256} / 実際 ${actual}）`,
          {
            repo,
            revisionSha,
            path: ref.path,
            expected: ref.sha256,
            actual,
            source,
            available,
          },
        );
      }
    };
    let bytes: Uint8Array;
    try {
      bytes = await fetchHfFile(target, { path: ref.path, validate }, {
        cacheName,
        init: requestInit(options.headers, options.signal),
        fetch: guarded,
        // 相 1 が温めた分はキャッシュヒットなので、ここが発火するのは self-heal の取り直しだけ。
        // NOTE: self-heal は evict してから取り直すため、その 1 巡だけ `loaded` はそのファイル
        //       ぶん巻き戻る（phase 契約が認めている「最初からやり直し」と同じ 1 巡）。
        //       `fileLoaded` も同じ 1 巡だけそのファイルの先頭から数え直しになる。
        onProgress: (progress) => {
          received.set(ref.path, progress.loaded);
          emit("downloading", ref.path);
        },
        ...(options.caches === undefined ? {} : { caches: options.caches }),
        ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, options.signal)) throw error;
      throw new HubFetchError(`${ref.path} の取得に失敗した（repo ${repo} @ ${revisionSha}）`, {
        repo,
        revisionSha,
        path: ref.path,
        available,
        cause: error,
      });
    }
    // MUST: yield の直前にも中断を見る — 冒頭の確認だけだと「最終 shard のキャッシュ読出し +
    // sha256 検証の最中に中断された」形が観測されず、取り消したはずのロードが正常完了して
    // 下流の Session 構築まで走る（検証済みバイトを配ってから止まるのでは中断の意味が無い）。
    options.signal?.throwIfAborted();
    const asset = assertTightView(bytes, ref.path);
    received.set(ref.path, ref.size);
    emit("complete", ref.path);
    // MUST: ここで手放す — 引き渡したバイト列を generator 側の表に溜めない（溜めた瞬間に
    // 全量面と同じ RAM 特性に戻り、この面の存在理由が消える）。次の反復に入れば `bytes` の
    // 束縛ごと到達不能になるので、常駐するのは「今の 1 本」だけ。
    yield { path: ref.path, bytes: asset };
  }
};

/**
 * karume が使うキャッシュ名前空間を**全て**消す（無認証 `karume/1` と、credential ごとに
 * 分かれた認証隔離 `karume/1:auth:*`）。認証側の名前は credential 由来で事前に列挙できないため
 * `CacheStorage.keys()` から拾う。「モデルを消して容量を空ける」に対応する面で、他コードの
 * 名前空間には触らない。
 *
 * MUST: 認証側を 1 つも残さない — gated 資産の写しが端末に残り続ける。
 *
 * @returns 少なくとも 1 つが実在して消えたら `true`（元から 1 つも無ければ `false`）。
 */
export const clearHubCache = async (
  options: { readonly caches?: CacheStorage } = {},
): Promise<boolean> => {
  const storage = options.caches ?? globalThis.caches;
  if (storage === undefined) {
    // 黙って no-op にしない。Cache Storage が無い環境（非セキュアオリジン等）で「消したつもり」に
    // なるのが最悪 — 消えていない写しをアプリが消えたものとして扱う。
    throw new Error(
      "hub: この環境に CacheStorage が無いためキャッシュを消せない" +
        "（options.caches で明示的に渡す）",
    );
  }
  const names = (await storage.keys()).filter(isHubCacheName);
  const deleted = await Promise.all(names.map((name) => storage.delete(name)));
  return deleted.includes(true);
};
