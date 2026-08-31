/**
 * キャッシュ名前空間の後始末。取得元そのものではなく **hub のセッション housekeeping**
 * （旧版の hub が残した名前空間の回収と、利用者向けの一括削除）。
 *
 * NOTE: 対象は取得層（`@hdae/fetch-cache`）が持つ Cache Storage の名前空間。CacheStorage を
 * 通らない取得元が入ると「消すべきキャッシュが無い取得元」が生まれるので、そのときは
 * {@link clearHubCache} の fail loud（下記）を取得元の能力へ寄せ直す席になる。
 */

import { clearCache } from "@hdae/fetch-cache";
import type { CacheDiagnostic } from "./session.ts";

/**
 * 取得層が名前空間を自前で持つ前（`@hdae/fetch-cache` 0.4 以前）に hub が使っていた名前空間。
 * 本体 `karume/1` と、認証隔離だった `karume/1:*` の両系列を指す。
 */
const LEGACY_CACHE_NAMESPACE = "karume/1";

/** 旧名前空間（本体 `karume/1` と、認証隔離だった `karume/1:*`）の名前を列挙する。 */
const legacyCacheNames = async (storage: CacheStorage): Promise<string[]> =>
  (await storage.keys()).filter((name) =>
    name === LEGACY_CACHE_NAMESPACE || name.startsWith(`${LEGACY_CACHE_NAMESPACE}:`)
  );

/**
 * 旧名前空間を回収する。新コードは二度と読まないので、置いておいても容量を占めるだけ
 * （中身は取得層 0.4 以前の形式で、キーも記録ハッシュも現行と互換が無い）。
 *
 * キャッシュは正しさの要件ではなく最適化なので、ここは決してロードを落とさない —
 * `caches` が無い環境（Deno 等）は黙って素通りし、列挙・削除の失敗は診断へ流して続行する。
 */
export const purgeLegacyCaches = async (options: {
  readonly caches?: CacheStorage;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
}): Promise<void> => {
  const storage = options.caches ?? globalThis.caches;
  if (storage === undefined) return;
  let names: readonly string[];
  try {
    names = await legacyCacheNames(storage);
  } catch (error) {
    options.onCacheError?.({ op: "delete", url: LEGACY_CACHE_NAMESPACE, error });
    return;
  }
  for (const name of names) {
    try {
      await storage.delete(name);
    } catch (error) {
      // 失敗した名前空間そのものを名乗る（この診断の `url` はキャッシュ名 — 取得元ではない）。
      options.onCacheError?.({ op: "delete", url: name, error });
    }
  }
};

/**
 * ダウンロード済みの資産を消して容量を空ける。対象は**取得層の名前空間まるごと**と、まだ
 * 残っているなら旧名前空間（`karume/1` 系）。記録ハッシュを信じる既定を疑ったときの回復手段も
 * これ（`recheck` 相当のノブは持たない — DECIDED: `docs/decisions/0080-hub-fetch-cache-050.md`）。
 *
 * NOTE: 取得層 0.5.0 の名前空間は内部固定 1 個で、キーに「どのアプリが温めたか」は入らない。
 * したがって**同じ origin のアプリが `@hdae/fetch-cache` を直接使って温めたものも一緒に消える**
 * （分離する手段がライブラリ側に無い）。repo 単位の細粒度掃除はキャッシュ保守波へ送り、ここは
 * 「全部消す」の 1 択に留める。
 *
 * @returns 何かが実在して消えたら `true`（元から 1 つも無ければ `false`）。
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
  // 旧名前空間はここでも回収する（{@link ../fetch.ts loadManifest} を 1 度も通さずに掃除だけ
  // する呼び方があるため）。こちらは掃除そのものが仕事なので、失敗を握り潰さず素通しする。
  const legacy = await legacyCacheNames(storage);
  const deleted = await Promise.all(legacy.map((name) => storage.delete(name)));
  // 名前空間の名前は取得層が所有するので、綴りを hub に焼かず掃除 API へ委ねる。
  const cleared = await clearCache({ caches: storage });
  return cleared || deleted.includes(true);
};
