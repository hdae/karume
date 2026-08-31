/**
 * セッションの語彙 — 「どの取得元の、どの世代を、どんな作法で読むか」。取得元の実装
 * （`sources/`）にも共通層（`fetch.ts`）にも属さない、面と面の間で受け渡す型だけを置く。
 *
 * NOTE: `fetch` / `caches` / `headers` は HTTP 取得元の語彙がそのまま公開面に出ているもの。
 * 直接読める取得元（ローカルディレクトリ等）はこれらを**無視する**（`caches` を通らない・
 * `fetch` を呼ばない）。全取得元に共通の作法は `signal` / `onProgress` / `onCacheError` の側で、
 * 取得元ごとの設定は取得元を作る factory（`localDirectory(...)` 等）が受け取る。
 */

import type { Manifest } from "./manifest.ts";
import type { AssetProgress } from "./progress.ts";
import type { PinnedSource, SourceDriver, SourceOrigin } from "./source.ts";

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
  /**
   * `Authorization` 等。取得（revision 解決・ファイル）へそのまま透過する。
   *
   * NOTE: キャッシュは credential で分けない（by-design）— キーにヘッダは入らないので、認証付きで
   * 取得したバイト列は以後の無認証呼び出しにもヒットする。gated 資産の運用予定が無い以上、
   * 隔離は過剰防御だった（詳細は `docs/limitations.md`）。
   */
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

/** 解決済みのセッション（取得元の実装 + そこで固定した世代識別子）。 */
type SessionSource = {
  readonly driver: SourceDriver;
  /** {@link SourceDriver.resolveGeneration} の返り値（世代を持たない取得元では空文字）。 */
  readonly generation: string;
};

// MUST: クラス定義より前に置く（`static` ブロックの TDZ — `source.ts` の同じ綴りを参照）。
let readSession: (loaded: LoadedManifest) => SessionSource;

/**
 * 解決済み世代に固定された manifest。以降の取得は全てこの世代で行う。
 *
 * **取得元そのものを内部欄として運ぶ** — 資産 3 面（`fetchAssets` / `prefetchAssets` /
 * `streamAssets`）は識別欄から取得元を組み立て直すのではなく、この値が持っている取得元を開く。
 * 組み立て直す形だと「HF なら repo と SHA から復元できる」という HF 固有の性質に全面が
 * 寄りかかり、復元手段の無い取得元（ローカルディレクトリ）が入れられない。
 *
 * 公開の識別欄（{@link repo} / {@link revisionSha}）は**診断と表示のための情報**であって、
 * 取得の再構成には使われない。持たない取得元（世代も repo も無いローカル配布形）では欄ごと
 * 現れない — 合成した repo や偽の SHA を名乗らせない（実在しないリポを指す診断になる）。
 *
 * MUST: この型は hub の外で組み立てられない（`mod.ts` は型だけを輸出する）。取得元は
 * `#session` に閉じており、値を作り替えて渡すと取得元ごと失われる。
 */
export class LoadedManifest {
  /** 取得元が repo という概念を持つ場合のみ（HF: `"owner/name"`）。 */
  readonly repo?: string;
  /** 解決済み commit SHA（40 桁）。世代の概念を持つ取得元のみ。 */
  readonly revisionSha?: string;
  readonly manifest: Manifest;
  readonly #session: SessionSource;

  constructor(manifest: Manifest, session: SessionSource, origin: SourceOrigin) {
    this.manifest = manifest;
    this.#session = session;
    if (origin.repo !== undefined) this.repo = origin.repo;
    if (origin.revisionSha !== undefined) this.revisionSha = origin.revisionSha;
  }

  static {
    readSession = (loaded) => loaded.#session;
  }
}

/**
 * セッションの取得元を**その呼び出しの作法で**開く。`fetch` / `caches` / `headers` を面ごとに
 * 差し替えられるのは、取得元が生成時ではなくここで作法を受け取るため（`source.ts` の
 * {@link SourceDriver.pin}）。
 */
export const pinnedSourceOf = (
  loaded: LoadedManifest,
  options: LoadManifestOptions,
): PinnedSource => {
  const session = readSession(loaded);
  return session.driver.pin(session.generation, options);
};

export type FetchAssetsOptions = LoadManifestOptions & {
  readonly onProgress?: (progress: AssetProgress) => void;
};

/** 逐次面 `streamAssets` のオプション（全量面 {@link FetchAssetsOptions} と同じ透過）。 */
export type StreamAssetsOptions = FetchAssetsOptions;
