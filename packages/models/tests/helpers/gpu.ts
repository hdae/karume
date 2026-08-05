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
