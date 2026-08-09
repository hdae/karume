/**
 * `@karume/hub` — 配布 manifest v2（`karume.json` / `karume/2`）の解決と HF からの取得。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — manifest を読む（{@link parseManifest} /
 * {@link loadManifest}）/ モデルと実行構成を選ぶ（{@link resolveFiles}）/ 資産を取る
 * （{@link fetchAssets}）/ 失敗を型で捌く（{@link HubError} 以下）/ キャッシュの診断を受け取る
 * （{@link CacheDiagnostic}）/ キャッシュを消して容量を空ける（{@link clearHubCache}）。
 *
 * 仕様の正本は `docs/decisions/0041-manifest-v2.md`（取得層は `0038-manifest-v1.md` §5）。
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
  ScoreStorage,
  SessionSpec,
  WeightEntry,
  WeightFiles,
} from "./src/manifest.ts";

export { resolveFiles } from "./src/resolve.ts";
export type { ResolvedFiles, ResolveOptions } from "./src/resolve.ts";

export { clearHubCache, fetchAssets, loadManifest } from "./src/fetch.ts";
export type {
  AssetPhase,
  AssetProgress,
  CacheDiagnostic,
  FetchAssetsOptions,
  HubRepoRef,
  LoadedManifest,
  LoadManifestOptions,
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
