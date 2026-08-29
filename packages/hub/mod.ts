/**
 * `@karume/hub` — 配布 manifest v4（`karume.json` / `karume/4`）の解決と HF からの取得。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — manifest を読む（{@link parseManifest} /
 * {@link loadManifest}）/ モデルと実行構成を選ぶ（{@link resolveFiles}）/ 資産を取る
 * （{@link fetchAssets}）/ shard を 2 相で逐次受け取る（{@link streamAssets} — RAM ピーク
 * O(最大 shard)。`docs/decisions/0070-shard-loading-admission.md` 決定 2）/ 資産を先に永続
 * キャッシュへ落とす（{@link prefetchAssets} — 逐次面の相 1 単体）/ 失敗を型で捌く
 * （{@link HubError} 以下）/ キャッシュの診断を受け取る（{@link CacheDiagnostic}）/
 * キャッシュを消して容量を空ける（{@link clearHubCache}）。
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

export {
  clearHubCache,
  fetchAssets,
  loadManifest,
  prefetchAssets,
  streamAssets,
} from "./src/fetch.ts";
export type {
  AssetPhase,
  AssetProgress,
  CacheDiagnostic,
  FetchAssetsOptions,
  HubRepoRef,
  LoadedManifest,
  LoadManifestOptions,
  StreamAssetsOptions,
  StreamedAsset,
} from "./src/fetch.ts";

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
