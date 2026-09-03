/**
 * 実 GPU テストの実行可否判定（models 版）。
 *
 * MUST: アダプタが無い環境は `Deno.test({ ignore })` で**明示 SKIP** する（テストを消して
 * 無音で緑にしない — ADR 0005）。判定はテスト登録時点で要るため、モジュール評価時に 1 回だけ
 * 行う。`requestAdapter()` が例外を投げる環境は「壊れた環境」なので握り潰さず伝播させる。
 *
 * NOTE: runtime 側の同名ヘルパ（`packages/runtime/tests/helpers/gpu.ts`）は import しない —
 * パッケージのテストが他パッケージのテスト内部に依存すると、向こうの門（`gpu_gate_test`）や
 * 環境変数の都合がこちらへ漏れる。流儀（判定 1 回・警告文・明示 SKIP）だけ合わせる。
 */
const detectAdapter = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  return (await gpu.requestAdapter()) !== null;
};

export const GPU_AVAILABLE: boolean = await detectAdapter();

if (!GPU_AVAILABLE) {
  console.warn(
    "[karume] GPUAdapter が無いため models の実 GPU テストを SKIP する" +
      "（リリース判定は実 GPU 緑が必須 — ADR 0005）",
  );
}

/**
 * GPU 時間診断（ADR 0021）が使えるアダプタか。
 *
 * MUST: パイプラインキー別の内訳（`lastRunTiming`）を読むテストは
 * `acquireGpu({ gpuTiming: true })` を明示的に渡す。既定は**要求しない**ので、渡し忘れると
 * `lastRunTiming` が undefined になり、`entries` を数えるキー検査が**黙って空振りする**。
 * `true` は feature 不在で fail loudly するため、列挙が無いアダプタではこのフラグでケースごと
 * SKIP する（**Metal も `timestamp-query` を広告するのでここは true 側になる** — 実際に計測が
 * 成立するかは別問題で、query set を多数同時に生かす経路は device ごと落ちる。2026-09-03 M2 実測・
 * `docs/known-issues.md`「Metal で `--diagnostics` が device ごと落ちる」節）。
 */
const detectTimestampQuery = async (): Promise<boolean> => {
  const gpu: GPU | undefined = navigator.gpu;
  if (gpu === undefined) return false;
  const adapter = await gpu.requestAdapter();
  return adapter !== null && adapter.features.has("timestamp-query");
};

export const TIMESTAMP_QUERY_AVAILABLE: boolean = await detectTimestampQuery();
