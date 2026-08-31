/**
 * セッションの語彙 — 「どのリポの、どの世代を、どんな作法で読むか」。取得元の実装
 * （`sources/`）にも共通層（`fetch.ts`）にも属さない、面と面の間で受け渡す型だけを置く。
 *
 * NOTE: `fetch` / `caches` / `headers` は HTTP 取得元の語彙がそのまま公開面に出ているもの
 * （現行の取得元は HF 1 つだけなので、ここに置いても嘘にならない）。取得元が増えたら
 * 「全取得元に共通の作法（`signal` / `onProgress` / `onCacheError`）」と「取得元ごとの設定」
 * へ割り直す席になる — 割り方の裁定はここではなく入口（`fromPretrained` の union）側の話。
 */

import type { Manifest } from "./manifest.ts";
import type { AssetProgress } from "./progress.ts";

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

/**
 * 解決済み revision に固定された manifest。以降の取得は全てこの SHA で行う。
 *
 * MUST: 識別欄（{@link repo} / {@link revisionSha} / {@link hubUrl}）を読むのは hub の中だけに
 * 留める — 組み立ては `loadManifest` の 1 箇所、読み手は `sources/` のアダプター構築と
 * `context.ts` の診断組み立ての 2 箇所しかない。消費側（`@karume/models`）はこの値を
 * **不透明に持ち回る**だけなので、世代識別子を持たない取得元が入っても消費側は動かない。
 */
export type LoadedManifest = {
  readonly repo: string;
  /** 解決済み commit SHA（40 桁）。返り値・エラー・診断に必ず載る。 */
  readonly revisionSha: string;
  readonly hubUrl?: string;
  readonly manifest: Manifest;
};

export type FetchAssetsOptions = LoadManifestOptions & {
  readonly onProgress?: (progress: AssetProgress) => void;
};

/** 逐次面 `streamAssets` のオプション（全量面 {@link FetchAssetsOptions} と同じ透過）。 */
export type StreamAssetsOptions = FetchAssetsOptions;
