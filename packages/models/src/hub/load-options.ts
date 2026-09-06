/**
 * 8 家族の `fromPretrained` が hub へ透過するオプション — **型と写しの 1 本**。
 *
 * MUST: 家族ごとに複製しない。以前は同じ 5 欄と同じスプレッドが 8 か所へ手書きで並んでいて、
 * hub が欄を 1 つ増やしたとき（0.11.0 の {@link FromPretrainedHubOptions.onRetry}）に 8 か所とも
 * 落ちた — 「新しいノブが黙って届かない」形は複製がある限り再発する。
 *
 * NOTE: `signal` はここに含めない — 置き場が家族で割れている（取得と構築の**両方**へ渡す家族は
 * `XPipelineOptions` 側に持ち、取得だけに効く家族は `XFromPretrainedOptions` 側に持つ）。
 * {@link hubLoadOptions} は引数として受けて写すだけで、置き場には触らない。
 */

import type {
  AssetProgress,
  CacheDiagnostic,
  LoadManifestOptions,
  RetryDiagnostic,
} from "@karume/hub";

/**
 * `XFromPretrainedOptions` が共通で持つ取得層のノブ（hub へそのまま透過する）。
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` は **HTTP 取得元専用**のノブで、取得元ハンドル
 * （`localDirectory` / `denoDirectory`）を渡した呼び出しでは 1 つも効かない — 手元の配布形は
 * network も CacheStorage も通らない。
 */
export type FromPretrainedHubOptions = {
  /**
   * `Authorization` 等。取得（revision 解決・ファイル）へそのまま透過する。
   *
   * NOTE: **キャッシュは credential で分けない**（by-design — キーにヘッダは入らないので、
   * 認証付きで取得したバイト列は以後の無認証呼び出しにもヒットする）。ADR 0080 決定 3 /
   * `docs/limitations.md` の「hub: キャッシュは credential で隔離しない」が正本。
   */
  readonly headers?: HeadersInit;
  readonly onProgress?: (progress: AssetProgress) => void;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /**
   * 取得層が 429 / 503 を取り直す 1 回ごとの通知先（待機に入る**前**に届く）。アプリが
   * 「rate limit 中・あと N 秒待つ」を出せるようにするための席で、無指定だと数十秒の沈黙に見える。
   *
   * 届く範囲は manifest 取得（`loadManifest`）と資産取得（shard の逐次面・prefetch・全量面）の
   * **両方** — この 1 本が `fromPretrained` の取得全部へ渡る。リスナーが throw しても取得は
   * 落ちない（取得層が隔離して warn する）。
   */
  readonly onRetry?: (diagnostic: RetryDiagnostic) => void;
  /** `fetch` の差し替え（テスト・カスタム輸送用）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** `CacheStorage` の差し替え（テスト用）。 */
  readonly caches?: CacheStorage;
};

/**
 * `fromPretrained` のオプション → hub のオプション。**定義済みの欄だけ**を写す
 * （`key: undefined` を作らない — 明示的な `undefined` は「無指定」と別物として hub の
 * 分岐に効きうる）。
 *
 * `onProgress` は写さない — 進捗は家族ごとに集約の要否が違い、`loadShardComponents` へ渡す側で
 * 別途載せる（manifest 取得に進捗は無い）。
 */
export const hubLoadOptions = (
  options: FromPretrainedHubOptions & { readonly signal?: AbortSignal },
): LoadManifestOptions => ({
  ...(options.signal === undefined ? {} : { signal: options.signal }),
  ...(options.headers === undefined ? {} : { headers: options.headers }),
  ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
  ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
  ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
  ...(options.caches === undefined ? {} : { caches: options.caches }),
});
