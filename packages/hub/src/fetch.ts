/**
 * HF リポジトリからの取得（ADR 0038 §5「hub の取得層」）。土台は `@hdae/fetch-cache`
 * （実行時依存ゼロ・Web 標準 API のみ）。
 *
 * 契約:
 * - revision は**セッションあたり 1 回だけ**解決し、manifest も全ファイルも同一 commit SHA に
 *   固定して取得する（可変 ref のまま複数回解決すると manifest と重みが別コミットから来る）。
 * - キャッシュ名前空間は必ず明示（`karume/1`）。`Authorization` を伴う取得は
 *   `karume/1:auth` へ隔離する（gated 資産を無認証経路のヒットに供さない）。
 * - 同時取得は 4 本まで。`AbortSignal` は全取得へ透過。
 * - 進捗総量は content-length ではなく manifest の `size` 合計（path 一意化後）。
 */

import { fetchHfFile, hfResolveUrl, isCommitSha, resolveHfRevision } from "@hdae/fetch-cache/hf";
import { HubError, HubFetchError, IntegrityError, ManifestFormatError } from "./errors.ts";
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
/** `Authorization` 付き取得の隔離先。 */
const AUTH_CACHE_NAMESPACE = "karume/1:auth";
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
  /** `Authorization` 等。付けた取得は認証専用のキャッシュ名前空間へ隔離される。 */
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

/** 進捗のフェーズ。`verifying` は sha256 照合中（3.7GB で数秒 — 無言のハングにしない）。 */
export type AssetPhase = "downloading" | "verifying";

export type AssetProgress = {
  readonly phase: AssetPhase;
  /** イベントを起こしたファイルの path。 */
  readonly path: string;
  /** 取得済みバイトの合計（全ファイル・path 一意化後）。 */
  readonly loaded: number;
  /** manifest の `size` 合計（path 一意化後）。 */
  readonly total: number;
};

export type FetchAssetsOptions = LoadManifestOptions & {
  readonly onProgress?: (progress: AssetProgress) => void;
};

const hasAuthorization = (headers?: HeadersInit): boolean =>
  headers !== undefined && new Headers(headers).has("authorization");

const cacheNameFor = (headers?: HeadersInit): string =>
  hasAuthorization(headers) ? AUTH_CACHE_NAMESPACE : CACHE_NAMESPACE;

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

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === "AbortError";

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
    if (isAbortError(error)) throw error;
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
  let bytes: Uint8Array;
  try {
    bytes = await fetchHfFile(target, { path: MANIFEST_FILENAME }, {
      cacheName: cacheNameFor(options.headers),
      init: requestInit(options.headers, options.signal),
      fetch: createGuardedFetch(options.fetch ?? globalThis.fetch, new Map([[url, budget]])),
      ...(options.caches === undefined ? {} : { caches: options.caches }),
      ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
    });
  } catch (error) {
    if (error instanceof HubError || isAbortError(error)) throw error;
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
    let sum = 0;
    for (const bytes of received.values()) sum += bytes;
    options.onProgress({ phase, path, loaded: sum, total });
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
  const cacheName = cacheNameFor(options.headers);
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
        // network 側だけ発火する（キャッシュヒットは verifying → 完了の 2 点で進む）。
        // `loaded` が `size` を超えないことは受信バイトの門（transport.ts）が保証する。
        onProgress: (progress) => {
          received.set(ref.path, progress.loaded);
          emit("downloading", ref.path);
        },
        ...(options.caches === undefined ? {} : { caches: options.caches }),
        ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      });
    } catch (error) {
      if (error instanceof HubError || isAbortError(error)) throw error;
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
    emit("downloading", ref.path);
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
