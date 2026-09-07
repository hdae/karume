/**
 * `@karume/hub` — 配布 manifest v4（`karume.json` / `karume/4`）の解決と、HF またはローカル
 * ディレクトリからの取得。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — manifest を読む（{@link parseManifest} /
 * {@link loadManifest}）/ 手元の配布形を取得元にする（{@link localDirectory} — ランタイム別の
 * 読み口は `@karume/hub/deno` 等のサブパス）/ モデルと実行構成を選ぶ（{@link resolveFiles}）/
 * 資産を取る
 * （{@link fetchAssets}）/ shard を 2 相で逐次受け取る（{@link streamAssets} — RAM ピーク
 * O(最大 shard)。`docs/decisions/0070-shard-loading-admission.md` 決定 2）/ 資産を先に永続
 * キャッシュへ落とす（{@link prefetchAssets} — 逐次面の相 1 単体）/ 資産 1 本の区間だけを読む
 * （{@link openAsset} — 取得元が持たなければ `undefined`）/ 失敗を型で捌く
 * （{@link HubError} 以下）/ キャッシュの診断を受け取る（{@link CacheDiagnostic}）/
 * 取得層の再試行（429 / 503 の `Retry-After` 追従）の通知を受け取る（{@link RetryDiagnostic}）/
 * キャッシュを消して容量を空ける（{@link clearHubCache}）/ 選択が落とし済みかを照会する
 * （{@link listCachedAssets} — 取りには行かない）/ 選択 1 つぶんの在庫を消す
 * （{@link evictCachedAssets} — 他の選択が使うファイルと越境参照は残す）。
 *
 * 仕様の正本は `docs/decisions/0041-manifest-v2.md`（取得層は `0038-manifest-v1.md` §5）。
 *
 * ## 版と manifest の対応
 *
 * **旧版の manifest は読まない**（major が違えば unsupported format で落ちる — ADR 0041 §1）。
 * JSR 3 本と PyPI `karume`（manifest を書く側）は lockstep で上がるので、下の対応は
 * `@karume/runtime` / `@karume/models` / exporter にもそのまま当てはまる。配布形（HF リポの
 * `karume.json`）を作り直す段取りを事前に読むための表:
 *
 * | パッケージ版 | `format` | 主な変更 |
 * | --- | --- | --- |
 * | 0.1.x | `karume/1` | 初版 |
 * | 0.2.x 〜 0.3.x | `karume/2` | model / quant の 2 軸（ADR 0041） |
 * | 0.4.x | `karume/3` | dtype エントリの shard 欄（ADR 0071） |
 * | 0.5.x | `karume/4` | quant の表示欄 + `requiredLimits`（ADR 0075 / 0038 §7）・ファイル参照の
 * 越境席（`repo` / `revision`）・`session` の計算ノブ値 `i8a8` → `a8`（ADR 0074） |
 *
 * 配布形を上げ直す手順は `docs/release-runbook.md`。
 */

export { MANIFEST_FILENAME, parseManifest } from "./src/manifest.ts";
export type {
  AttentionCompute,
  FileRef,
  GpuFeaturesSpec,
  LinearCompute,
  Manifest,
  ModelEntry,
  PipelineId,
  Quant,
  RequiredLimitName,
  RequiredLimitsSpec,
  ScoreStorage,
  SessionSpec,
  WeightEntry,
  WeightFiles,
} from "./src/manifest.ts";

export { resolveFiles } from "./src/resolve.ts";
export type { ResolvedFiles, ResolveOptions } from "./src/resolve.ts";

/**
 * 取得元。`loadManifest` / `fromPretrained` は HF のリポ参照（{@link HubRepoRef}）か、ここで
 * 作った取得元ハンドル（{@link DistributionSource} — **中身は不透明**）のどちらでも受ける。
 * union を自前で捌く面（`@karume/models` の `fromPretrained`）は {@link isDistributionSource} で
 * 判別する — 不透明ハンドルを構造で見分けようとすると綴り間違いが取得元として通る。
 */
export { localDirectory } from "./src/sources/local.ts";
export type { DirectoryAdapter, LocalDirectoryOptions } from "./src/sources/local.ts";
export { isDistributionSource } from "./src/source.ts";
export type { DistributionSource } from "./src/source.ts";

export { clearHubCache } from "./src/cache.ts";
export { evictCachedAssets, listCachedAssets } from "./src/inventory.ts";
export type {
  CachedAssets,
  CacheInventoryOptions,
  EvictedAssets,
  KeptAsset,
} from "./src/inventory.ts";
export { fetchAssets, loadManifest, prefetchAssets, streamAssets } from "./src/fetch.ts";
export type { StreamedAsset } from "./src/fetch.ts";
/**
 * 資産 1 本の区間読み（{@link openAsset} — 全量ではなく `[offset, offset + length)` だけを引く。
 * **任意能力**なので、持たない取得元では `undefined` が返り、呼び手は全量読みへ倒す）。
 * 作法（{@link LoadManifestOptions}）は他の面と同じくこの呼び出しに渡した分だけが効く。
 */
export { openAsset } from "./src/fetch.ts";
export type { AssetRangeReader } from "./src/source.ts";
export type { AssetPhase, AssetProgress } from "./src/progress.ts";
export type {
  CacheDiagnostic,
  FetchAssetsOptions,
  HubRepoRef,
  LoadedManifest,
  LoadManifestOptions,
  RetryDiagnostic,
  StreamAssetsOptions,
} from "./src/session.ts";

/**
 * エラー型は「利用者の入力・環境に起因する失敗」だけを公開する。取得層の不変条件破れ
 * （bytes が buffer 全体を占めていない等）は hub 側のバグであって利用者が分岐すべきもの
 * ではないため、`Error` のまま届く。
 */
export {
  HubError,
  HubFetchError,
  IntegrityError,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
} from "./src/errors.ts";
export type { AvailableLabels, IntegritySource } from "./src/errors.ts";
